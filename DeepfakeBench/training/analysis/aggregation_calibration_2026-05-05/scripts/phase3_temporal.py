"""Phase 3 — Temporal autocorrelation of frame_prob within streams.

Per stream (sessions of >= 8 frames, sorted by (seg_id, frame_idx)):
  - Pearson lag-1, lag-3, lag-5 autocorrelation
  - Linear-regression slope (trend coefficient over frame index)
  - Run-length distribution of "above τ=0.5" segments

Aggregated to suite-level percentiles. The headline finding answers:
"Are 32 frames effectively 32 independent votes, or fewer?"

Outputs:
  data/<ckpt>/temporal_autocorr.csv    — one row per stream
  data/<ckpt>/temporal_suite_summary.csv — suite percentiles
  findings/<ckpt>/phase3_temporal.md
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from _common import ensure_dirs, get_ckpt_cfg, load_suite_csv  # noqa: E402
from aggregator_sim import extract_frame_index, extract_seg_id, extract_session  # noqa: E402


def lag_autocorr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag or x.std() == 0:
        return float("nan")
    a = x[:-lag]
    b = x[lag:]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def linear_slope(x: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    t = np.arange(n)
    return float(np.polyfit(t, x, 1)[0])


def run_lengths_above(x: np.ndarray, t: float = 0.5) -> list[int]:
    """Return run-lengths of consecutive positions where x >= t."""
    above = (x >= t).astype(np.int8)
    lengths = []
    cur = 0
    for v in above:
        if v:
            cur += 1
        else:
            if cur > 0:
                lengths.append(cur)
                cur = 0
    if cur > 0:
        lengths.append(cur)
    return lengths


def main(ckpt_name: str) -> None:
    cfg = get_ckpt_cfg(ckpt_name)
    dirs = ensure_dirs(ckpt_name)
    full_name = cfg["full_name"]
    t0 = time.time()

    rows = []
    for suite_name in cfg["suites"].keys():
        try:
            df = load_suite_csv(cfg, suite_name)
        except FileNotFoundError:
            continue
        df = df.copy()
        df["session"] = df["video_id"].astype(str).map(extract_session)
        df["seg_id"] = df["frame_path"].astype(str).map(extract_seg_id)
        df["frame_idx"] = df["frame_path"].astype(str).map(extract_frame_index)
        for sess, sub in df.groupby("session", sort=False):
            ordered = sub.sort_values(["seg_id", "frame_idx"])
            scores = ordered["frame_prob"].to_numpy(dtype=np.float64)
            n = len(scores)
            if n < 8:
                continue
            rls = run_lengths_above(scores, 0.5)
            rec = {
                "suite": suite_name,
                "label_class": df["label_class"].iloc[0],
                "stream_id": sess,
                "n_frames": int(n),
                "lag1_autocorr": lag_autocorr(scores, 1),
                "lag3_autocorr": lag_autocorr(scores, 3),
                "lag5_autocorr": lag_autocorr(scores, 5),
                "linear_slope": linear_slope(scores),
                "n_runs_above_0.5": int(len(rls)),
                "max_run_length_above_0.5": int(max(rls) if rls else 0),
                "median_run_length_above_0.5": float(np.median(rls) if rls else 0.0),
            }
            rows.append(rec)

    df_temp = pd.DataFrame(rows)
    df_temp.to_csv(dirs["data"] / "temporal_autocorr.csv", index=False)

    if df_temp.empty:
        (dirs["findings"] / "phase3_temporal.md").write_text(
            f"# Phase 3 — temporal autocorrelation — {full_name}\n\n"
            "No streams with >= 8 frames. Phase 3 inconclusive on this ckpt.\n"
            f"\n_Wall time: {time.time() - t0:.1f}s_\n"
        )
        print(f"[phase3] {full_name} — no eligible streams")
        return

    # Suite-level percentiles.
    summary = (
        df_temp.groupby(["suite", "label_class"])
        .agg(
            n_streams_geq8=("n_frames", "size"),
            lag1_p25=("lag1_autocorr", lambda s: float(np.nanpercentile(s, 25))),
            lag1_p50=("lag1_autocorr", lambda s: float(np.nanpercentile(s, 50))),
            lag1_p75=("lag1_autocorr", lambda s: float(np.nanpercentile(s, 75))),
            lag3_p50=("lag3_autocorr", lambda s: float(np.nanpercentile(s, 50))),
            lag5_p50=("lag5_autocorr", lambda s: float(np.nanpercentile(s, 50))),
            slope_p50=("linear_slope", lambda s: float(np.nanpercentile(s, 50))),
            max_run_p95=("max_run_length_above_0.5", lambda s: float(np.nanpercentile(s, 95))),
        )
        .reset_index()
    )
    summary.to_csv(dirs["data"] / "temporal_suite_summary.csv", index=False)

    md = [f"# Phase 3 — temporal autocorrelation — {full_name}", ""]
    md.append(
        "Streams with >= 8 frames only. lag1 autocorrelation tells us if 32 frames in a "
        "window are effectively independent. Strong positive lag1 (e.g., > 0.5) means the "
        "effective sample size is <<32 — sliding windows behave very differently from "
        "random subsampling.\n"
    )
    md.append("| suite | label | n_streams≥8 | lag1 p25 | lag1 p50 | lag1 p75 | lag3 p50 | lag5 p50 | slope p50 | max_run≥0.5 p95 |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        md.append(
            f"| {r['suite']} | {r['label_class']} | {int(r['n_streams_geq8'])} | "
            f"{r['lag1_p25']:.3f} | {r['lag1_p50']:.3f} | {r['lag1_p75']:.3f} | "
            f"{r['lag3_p50']:.3f} | {r['lag5_p50']:.3f} | "
            f"{r['slope_p50']:.4g} | {r['max_run_p95']:.0f} |"
        )

    md.append(f"\n_Wall time: {time.time() - t0:.1f}s_")
    (dirs["findings"] / "phase3_temporal.md").write_text("\n".join(md))
    print(f"[phase3] {full_name} done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    main(a.ckpt)
