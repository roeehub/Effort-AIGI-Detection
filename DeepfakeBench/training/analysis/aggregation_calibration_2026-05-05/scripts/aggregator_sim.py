"""
Core production-aggregation simulator. Pure functions, importable, multi-process safe.

Production semantics:
  - In production the model receives a video stream sampled at 3-4 fps.
  - A sliding window of W frame-level scores is aggregated to a per-window
    decision.
  - The video is flagged FAKE iff ANY window in the video flags it.

This module implements:
  apply_strategy(scores: np.ndarray, strategy: str, **params) -> bool
      Aggregate a single window of scores -> {True=fake, False=real}.

  apply_override(scores: np.ndarray, rule: str) -> bool
      OR-with override rule on the same window.

  simulate_video(scores, frame_idx, policy) -> bool
      Run a sliding window over a stream and return the video-level decision.

  simulate_suite(stream_iter, policy) -> n_videos, n_flagged
      Iterate streams from a suite and return aggregate stats.

`policy` is a dict:
    window_size: int
    strategy: str
    strategy_params: dict
    override_rule: str  ('none' for no override)
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Frame-index extraction (used for ordering frames within a stream)
# ----------------------------------------------------------------------
_FRAME_RE = re.compile(r"frame_(\d+)")
_SEG_RE = re.compile(r"_(\d+\.\d+)_frame_")  # captures "0.0" in "_s32_0.0_frame_..."


def extract_frame_index(path: str) -> int:
    m = _FRAME_RE.search(path or "")
    return int(m.group(1)) if m else 0


def extract_seg_id(path: str) -> float:
    m = _SEG_RE.search(path or "")
    return float(m.group(1)) if m else 0.0


def order_key(row) -> Tuple[float, int]:
    """Sort key: (segment_index, frame_within_segment)."""
    return (extract_seg_id(row.frame_path), extract_frame_index(row.frame_path))


# ----------------------------------------------------------------------
# Window-level aggregation strategies
# ----------------------------------------------------------------------

def _majority_vote(scores: np.ndarray, threshold: float, vote_majority: float) -> bool:
    """Fraction of frames with prob >= threshold must be >= vote_majority."""
    return (scores >= threshold).mean() >= vote_majority


def _high_vote_floor(scores: np.ndarray, threshold: float, vote_majority: float) -> bool:
    """Like majority_vote but with stronger super-majority floors."""
    return (scores >= threshold).mean() >= vote_majority


def _trimmed_mean_above_thresh(scores: np.ndarray, threshold: float, trim_pct: float) -> bool:
    """Drop top trim_pct and bottom trim_pct, then mean(remaining) >= threshold."""
    n = len(scores)
    if n == 0:
        return False
    k = int(np.floor(trim_pct * n))
    if 2 * k >= n:
        return False
    s = np.sort(scores)[k : n - k]
    return s.mean() >= threshold


def _median_above_thresh(scores: np.ndarray, threshold: float) -> bool:
    return np.median(scores) >= threshold


def _rolling_mean_then_thresh(scores: np.ndarray, smoothing_window: int, threshold: float) -> bool:
    """Smooth via rolling mean, then ANY smoothed value >= threshold flags fake."""
    if len(scores) == 0:
        return False
    w = min(smoothing_window, len(scores))
    if w <= 1:
        return (scores >= threshold).any()
    # cumulative-sum trick for fast rolling mean
    c = np.cumsum(np.insert(scores, 0, 0.0))
    smoothed = (c[w:] - c[:-w]) / w
    return (smoothed >= threshold).any()


def _run_length(scores: np.ndarray, threshold: float, M: int) -> bool:
    """Require M consecutive frames above threshold."""
    if len(scores) < M:
        return False
    above = scores >= threshold
    # rolling sum of bools — if any window of length M has all M, run-length met
    c = np.cumsum(np.insert(above.astype(np.int32), 0, 0))
    sums = c[M:] - c[:-M]
    return bool((sums >= M).any())


_STRATEGY_DISPATCH = {
    "majority_vote": _majority_vote,
    "high_vote_floor": _high_vote_floor,
    "trimmed_mean_above_thresh": _trimmed_mean_above_thresh,
    "median_above_thresh": _median_above_thresh,
    "rolling_mean_then_thresh": _rolling_mean_then_thresh,
    "run_length": _run_length,
}


def apply_strategy(scores: np.ndarray, strategy: str, **params) -> bool:
    fn = _STRATEGY_DISPATCH.get(strategy)
    if fn is None:
        raise KeyError(f"unknown strategy: {strategy}")
    return bool(fn(scores, **params))


# ----------------------------------------------------------------------
# Override rules — return True iff override fires (forces fake decision)
# ----------------------------------------------------------------------

def apply_override(scores: np.ndarray, rule: str) -> bool:
    if rule == "none":
        return False
    if rule == "one_frame_above_0.98":
        return bool((scores >= 0.98).any())
    if rule == "two_frames_above_0.98_in_window":
        return bool((scores >= 0.98).sum() >= 2)
    if rule == "three_consec_frames_above_0.95":
        return _consec_above(scores, 0.95, 3)
    if rule == "five_consec_frames_above_0.90":
        return _consec_above(scores, 0.90, 5)
    raise KeyError(f"unknown override rule: {rule}")


def _consec_above(scores: np.ndarray, t: float, k: int) -> bool:
    if len(scores) < k:
        return False
    a = scores >= t
    c = np.cumsum(np.insert(a.astype(np.int32), 0, 0))
    return bool(((c[k:] - c[:-k]) >= k).any())


# ----------------------------------------------------------------------
# Window-level + video-level decisions
# ----------------------------------------------------------------------

def window_decision(scores: np.ndarray, policy: dict) -> bool:
    """Aggregate one window — strategy OR override -> fake."""
    if apply_override(scores, policy["override_rule"]):
        return True
    return apply_strategy(scores, policy["strategy"], **policy["strategy_params"])


def simulate_video(scores: np.ndarray, policy: dict) -> bool:
    """Slide a window over a video stream and return video-level decision (any-window-fake).

    For very short streams (n < min(W, MIN_STREAM_FOR_AGG)=8) the policy is degenerate;
    we fall back to a frame-level decision derived from the policy's primary threshold.
    This is the honest treatment for sample-frame data that's not actually a video.
    """
    n = len(scores)
    if n == 0:
        return False
    W = policy["window_size"]
    # If the stream is too short to be meaningfully aggregated (e.g., 1 frame),
    # fall back to a frame-level threshold decision so the policy still evaluates.
    MIN_AGG = 8
    if n < min(W, MIN_AGG):
        # Use the policy's "threshold" if present; otherwise use 0.5 by convention.
        thr = policy["strategy_params"].get("threshold", 0.5)
        # Honor the override on this short window too.
        if apply_override(scores, policy["override_rule"]):
            return True
        return bool((scores >= thr).any())
    if n < W:
        # Short-but-aggregatable stream — evaluate as a single short window (clamp).
        return window_decision(scores, policy)
    for start in range(0, n - W + 1):
        if window_decision(scores[start : start + W], policy):
            return True
    return False


def simulate_streams(streams: List[np.ndarray], policy: dict) -> Tuple[int, int]:
    """Apply policy to every stream. Return (n_streams, n_flagged_fake)."""
    n_total = len(streams)
    n_flagged = sum(1 for s in streams if simulate_video(s, policy))
    return n_total, n_flagged


# ----------------------------------------------------------------------
# Stream construction from frame-level CSVs
# ----------------------------------------------------------------------
_SESSION_RE = re.compile(r"^(.*?)__seg_")


def extract_session(video_id: str) -> str:
    """Extract session ID from a video_id like 'Cam_Test__s32__seg_308.0__real'.
    Falls back to the full video_id when there's no __seg_ marker."""
    m = _SESSION_RE.match(video_id or "")
    return m.group(1) if m else (video_id or "")


def build_streams(df: pd.DataFrame, group_col: str = "session") -> List[np.ndarray]:
    """Group frames into pseudo-streams of contiguous score sequences.

    For teams suites: groups by session (e.g., 'Cam_Test__s32') and orders
    by (seg_id, frame_idx). For deeplive_enhanced_dev where there's only
    one session, falls back to grouping by an arbitrary partition.
    """
    if df.empty:
        return []
    df = df.copy()
    df["session"] = df["video_id"].astype(str).map(extract_session)
    df["seg_id"] = df["frame_path"].astype(str).map(extract_seg_id)
    df["frame_idx"] = df["frame_path"].astype(str).map(extract_frame_index)
    streams: List[np.ndarray] = []
    for sess, sub in df.groupby(group_col, sort=False):
        ordered = sub.sort_values(["seg_id", "frame_idx"])
        streams.append(ordered["frame_prob"].to_numpy(dtype=np.float64))
    return streams


def build_streams_for_suite(df: pd.DataFrame, suite_name: str) -> List[np.ndarray]:
    """Suite-aware stream construction.

    deeplive_enhanced_dev has one root identity; treat each video_id as its
    own (length-1) stream OR partition by chunks of N to form pseudo-streams.

    For teams_real_dor_dev (n=50, all 1-frame) — same fallback.
    """
    if df.empty:
        return []
    streams = build_streams(df, "session")

    # If the only-session fallback yielded one giant stream OR streams that are
    # all length-1 (deeplive-style), partition into chunks of ~32 to allow
    # window simulation. This is honest: chunk-grouped pseudo-streams are NOT
    # true continuous video — but they're the best we can do with sample-frame
    # data, and we explicitly flag this in FINDINGS.
    if len(streams) <= 2 or all(len(s) <= 4 for s in streams):
        all_scores = df.sort_values(
            ["video_id", "frame_path"]
        )["frame_prob"].to_numpy(dtype=np.float64)
        chunked = []
        chunk = 32  # mimic a 32-frame window
        for i in range(0, len(all_scores), chunk):
            sl = all_scores[i : i + chunk]
            if len(sl) >= 4:  # discard tail < 4 frames
                chunked.append(sl)
        if chunked:
            return chunked
    return streams


# ----------------------------------------------------------------------
# Policy expansion — Cartesian product over the grid
# ----------------------------------------------------------------------

def expand_policy_grid(grid: dict) -> List[dict]:
    """Expand the YAML policy grid into a flat list of policy dicts."""
    from itertools import product

    window_sizes = grid.get("window_size", [32])
    overrides = grid.get("override_rules", ["none"])
    out: List[dict] = []
    pid = 0
    for strat in grid["strategies"]:
        name = strat["name"]
        params = strat["params"]
        keys = list(params.keys())
        values_lists = [params[k] for k in keys]
        for vs in product(*values_lists):
            sp = dict(zip(keys, vs))
            for W in window_sizes:
                for ov in overrides:
                    out.append(
                        {
                            "policy_id": pid,
                            "window_size": int(W),
                            "strategy": name,
                            "strategy_params": dict(sp),
                            "override_rule": ov,
                        }
                    )
                    pid += 1
    return out


# ----------------------------------------------------------------------
# Path normalization for parquet join
# ----------------------------------------------------------------------

def normalize_path_for_join(p: str) -> str:
    """Strip gs://bucket/ prefix to a basename for join with parquet which
    might use blob_path or basename.
    """
    if not isinstance(p, str):
        return ""
    if p.startswith("gs://"):
        return p.split("/", 3)[-1] if p.count("/") >= 3 else p
    return p


def basename(p: str) -> str:
    if not isinstance(p, str):
        return ""
    return p.rsplit("/", 1)[-1]
