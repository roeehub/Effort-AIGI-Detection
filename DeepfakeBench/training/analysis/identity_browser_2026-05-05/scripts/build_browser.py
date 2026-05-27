#!/usr/bin/env python3
"""Build a self-contained HTML browser of evaluation frames grouped by base identity.

Re-runnable / idempotent.  Pulls frame paths from one or more manifest CSVs,
downloads frames from GCS, generates 256-px thumbnails, then writes
`index.html`.  Adding a new bucket = pass another manifest via --manifest.

Layout produced (all relative to OUTPUT_ROOT):
  frames/<base_identity>/<suite>__<frame_filename>     # full-size originals
  thumbs/<base_identity>/<suite>__<frame_filename>.jpg # 256-px thumbnails
  data/grouped_manifest.csv                            # per-frame metadata (legacy)
  data/grouped_manifest_v2.csv                         # per-frame metadata + scores + props
  data/summary.json                                    # per-identity counts
  index.html                                           # the browser
  run.log                                              # phase timings + skips

CPU-only.  No sklearn n_jobs=-1.  Uses google-cloud-storage with a
ThreadPoolExecutor for I/O-bound downloads and multiprocessing.Pool(workers)
for thumbnailing.

Iteration 2 (2026-05-05): adds per-frame P8A score overlay, binary
property tags (face_size, quality), within-identity sort by those props
and score, and an extension point for additional checkpoint scores via
repeatable `--score-csv NAME=PATH` (PATH may glob).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from html import escape as html_escape
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_ROOT = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/"
    "DeepfakeBench/training/analysis/identity_browser_2026-05-05"
)

# Default P8A score CSV glob (8 of the 9 suite reports — drops
# visomaster_enhanced_macro_dev because it isn't represented in the
# bundled scope manifest; it's harmless if included though).
DEFAULT_P8A_GLOB = (
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/"
    "DeepfakeBench/training/analysis/cpu_followups_2026-05-04/raw_reports/"
    "*p8a_reference_step5000_frames_report.csv"
)
DEFAULT_E2B_GLOB = (
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/"
    "DeepfakeBench/training/analysis/cpu_followups_2026-05-04/raw_reports/"
    "*_e2b_top_n_step3200_frames_report.csv"
)
DEFAULT_PA_3800_GLOB = (
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/"
    "DeepfakeBench/training/analysis/pa_pc_eval_2026-05-05/raw_reports/"
    "*_pa_top_n_step3800_frames_report.csv"
)
DEFAULT_T3_S1_STEP1500_GLOB = (
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/"
    "DeepfakeBench/training/analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/"
    "slot1_step1500/*_t3_slot1_periodic_step1500_frames_report.csv"
)
DEFAULT_T3_S1_STEP2500_GLOB = (
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/"
    "DeepfakeBench/training/analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/"
    "slot1_step2500/*_t3_slot1_periodic_step2500_frames_report.csv"
)
DEFAULT_TAGS_PARQUET = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/"
    "DeepfakeBench/training/analysis/lockbox_tagging/full_tags_2026-04-27.parquet"
)

THUMBNAIL_MAX_DIM = 256
THUMBNAIL_QUALITY = 80
DEFAULT_DOWNLOAD_WORKERS = 32
DEFAULT_THUMB_WORKERS = 8

# Chronic-6 list per memory `project_chronic_offenders_partition_per_ckpt_2026-05-04`.
CHRONIC_6 = {
    "PC_Generator__s22",
    "bla_bla_chow",
    "bla_bla_chow__s2",
    "PC_Generator__s45",
    "Roy_D",  # case-corrected from chronic list
    "Q__s6",
}

# ---------------------------------------------------------------------------
# Identity extraction
# ---------------------------------------------------------------------------

_RE_SEG = re.compile(r"__seg_[\d.]+")
_RE_SEQ = re.compile(r"__seq_?\d+")
_RE_LABEL_SUFFIX = re.compile(r"__(?:real|fake)$")
_RE_FRAME_META = re.compile(r"__frame_\d+_crop_\d+__[a-f0-9]+")
_RE_TRAILING_UNDERSCORES = re.compile(r"_+$")


def base_identity(video_id: str) -> str:
    """Extract the base identity from a video_id.

    Strips: __seg_X.Y, __seq[_]NNN, trailing __real/__fake, embedded
    __frame_NNN_crop_NNN__hex (for ilan/orel-style ids), trailing underscores.
    """

    s = video_id
    s = _RE_SEG.sub("", s)
    s = _RE_SEQ.sub("", s)
    s = _RE_LABEL_SUFFIX.sub("", s)
    s = _RE_FRAME_META.sub("", s)
    s = _RE_TRAILING_UNDERSCORES.sub("", s)
    return s


# ---------------------------------------------------------------------------
# Local-path planning
# ---------------------------------------------------------------------------


def safe_segment(seg: str) -> str:
    """Make a string safe for use as a single path segment."""

    seg = seg.replace("/", "_")
    return seg


def local_paths(row: dict, output_root: Path) -> tuple[Path, Path]:
    """Compute (full_frame_path, thumb_path) for a manifest row.

    Frame extension preserved from source.  Thumb is always JPEG.
    """

    suite = safe_segment(str(row["suite"]))
    ident = safe_segment(str(row["base_identity"]))
    src_name = Path(row["frame_path"]).name
    target_name = f"{suite}__{src_name}"
    full = output_root / "frames" / ident / target_name
    thumb_name = Path(target_name).with_suffix(".jpg").name
    thumb = output_root / "thumbs" / ident / thumb_name
    return full, thumb


# ---------------------------------------------------------------------------
# Manifest assembly
# ---------------------------------------------------------------------------


def load_and_normalize_manifests(paths: Iterable[Path]) -> pd.DataFrame:
    """Concatenate one or more scope manifests and add base_identity column.

    Drops the (possibly-wrong) original `identity` column to avoid confusion.
    """

    frames: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        if "identity" in df.columns:
            df = df.drop(columns=["identity"])
        df["source_manifest"] = str(p)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    required = {"suite", "video_id", "frame_path", "label", "is_lockbox"}
    missing = required - set(out.columns)
    if missing:
        raise SystemExit(f"manifest missing required columns: {missing}")
    out["base_identity"] = out["video_id"].map(base_identity)
    out = out.drop_duplicates(subset=["frame_path"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Score CSV loading (per-checkpoint extension point)
# ---------------------------------------------------------------------------


def parse_score_specs(specs: list[str] | None) -> list[tuple[str, list[str]]]:
    """Parse `--score-csv NAME=PATH_OR_GLOB` specs into (name, [paths]).

    PATH_OR_GLOB is expanded via glob.glob.  Repeated specs with the same NAME
    are merged: the paths are concatenated (order: first appearance wins for
    NAME ordering).  Empty specs => empty list.
    """

    if not specs:
        return []
    name_to_paths: dict[str, list[str]] = {}
    name_order: list[str] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(
                f"--score-csv must be NAME=PATH (got: {spec!r})"
            )
        name, path_or_glob = spec.split("=", 1)
        name = name.strip()
        path_or_glob = path_or_glob.strip()
        if not name:
            raise SystemExit(f"--score-csv NAME is empty in: {spec!r}")
        paths = sorted(glob.glob(path_or_glob))
        if not paths:
            # Allow a literal path that doesn't glob-match
            if Path(path_or_glob).exists():
                paths = [path_or_glob]
            else:
                raise SystemExit(
                    f"--score-csv {name}: no files match {path_or_glob!r}"
                )
        if name not in name_to_paths:
            name_to_paths[name] = []
            name_order.append(name)
        name_to_paths[name].extend(paths)
    out: list[tuple[str, list[str]]] = [(n, name_to_paths[n]) for n in name_order]
    return out


def load_score_lookup(
    name: str, paths: list[str], log: logging.Logger
) -> dict[str, float]:
    """Concatenate score CSVs for a single ckpt and return frame_path -> prob."""

    lookup: dict[str, float] = {}
    total_rows = 0
    for p in paths:
        df = pd.read_csv(p)
        if "frame_path" not in df.columns or "frame_prob" not in df.columns:
            log.warning(
                f"score CSV {p} missing frame_path or frame_prob; skipping"
            )
            continue
        for fp, fp_prob in zip(df["frame_path"], df["frame_prob"]):
            # If a frame appears in multiple suite CSVs (rare but possible),
            # last-write-wins is fine — both should report the same prob from
            # the same checkpoint.
            lookup[str(fp)] = float(fp_prob)
        total_rows += len(df)
    log.info(
        f"  score lookup [{name}]: {len(lookup)} unique frame_paths "
        f"from {len(paths)} CSV(s) ({total_rows} rows)"
    )
    return lookup


# ---------------------------------------------------------------------------
# Tag parquet loading + per-suite binary property derivation
# ---------------------------------------------------------------------------


def load_tags(parquet_path: Path, log: logging.Logger) -> pd.DataFrame | None:
    """Load tag parquet; returns dataframe with gcs_uri + face_area_ratio +
    is_low_quality + is_no_face + width + height, or None if the parquet is
    missing. The width/height/is_no_face columns drive the G1+G2 production
    gate filter; missing them is non-fatal (gate_status becomes "unknown").
    """

    if not parquet_path.exists():
        log.warning(f"tag parquet not found: {parquet_path}; skipping properties")
        return None
    df = pd.read_parquet(parquet_path)
    cols_needed = {"gcs_uri", "face_area_ratio", "is_low_quality"}
    missing = cols_needed - set(df.columns)
    if missing:
        log.warning(
            f"tag parquet {parquet_path} missing cols {missing}; skipping properties"
        )
        return None
    # Gate-relevant columns: is_no_face (G1) + width/height (G2). Pull if
    # available; absent values are treated as "unknown" at gate-derivation time.
    keep_cols = list(cols_needed)
    for opt in ("is_no_face", "width", "height"):
        if opt in df.columns:
            keep_cols.append(opt)
    return df[keep_cols].copy()


def derive_binary_properties(
    df: pd.DataFrame, tags: pd.DataFrame | None, log: logging.Logger
) -> pd.DataFrame:
    """Add face_size and quality columns to df.

    face_size: 'close' / 'far' / 'unknown'
      - per-suite median split on face_area_ratio
        (face_area_ratio is the parquet's analogue of face_area_quartile;
        the spec's face_area_quartile column doesn't exist in the parquet)
      - fallback: face_area_ratio >= 0.10 maps to 'close'
        (only used if a suite has zero parquet coverage)

    quality: 'hi-q' / 'lo-q' / 'unknown'
      - is_low_quality == False -> 'hi-q'
      - is_low_quality == True  -> 'lo-q'
      - missing -> 'unknown'
    """

    df = df.copy()
    if tags is None:
        df["face_size"] = "unknown"
        df["quality"] = "unknown"
        df["face_area_ratio"] = float("nan")
        df["is_low_quality"] = pd.NA
        df["gate_status"] = "unknown"
        return df

    merged = df.merge(
        tags.rename(columns={"gcs_uri": "_join_uri"}),
        left_on="frame_path",
        right_on="_join_uri",
        how="left",
    ).drop(columns=["_join_uri"])

    # Per-suite median split for face_size; fallback per-spec to >= 0.10.
    def assign_face_size(grp: pd.DataFrame) -> pd.Series:
        valid = grp["face_area_ratio"].notna()
        if valid.sum() == 0:
            return pd.Series(["unknown"] * len(grp), index=grp.index)
        med = grp.loc[valid, "face_area_ratio"].median()
        out = pd.Series(["unknown"] * len(grp), index=grp.index)
        out.loc[valid & (grp["face_area_ratio"] >= med)] = "close"
        out.loc[valid & (grp["face_area_ratio"] < med)] = "far"
        return out

    merged["face_size"] = (
        merged.groupby("suite", group_keys=False)
        .apply(assign_face_size)
        .reindex(merged.index)
        .fillna("unknown")
    )

    # quality
    def map_quality(v) -> str:
        if pd.isna(v):
            return "unknown"
        return "lo-q" if bool(v) else "hi-q"

    merged["quality"] = merged["is_low_quality"].map(map_quality)

    # G1 + G2(200) production gate per frame. Categories:
    #   "pass":         face present AND min(W,H) >= 200  (would-be-scored in prod)
    #   "drop_no_face": is_no_face == True
    #   "drop_lowres":  min(W,H) < 200 and face present
    #   "drop_both":    is_no_face AND min(W,H) < 200
    #   "unknown":      missing parquet coverage (no width/height OR no is_no_face)
    # Frames in chronic-tiny-face identities (pc_gen_s22/s45/q__s6) typically
    # land in drop_lowres; bla_bla_chow__s2 mostly lands in drop_lowres too.
    # Roy_D has NO parquet coverage in current parquet -> "unknown".
    G2_MIN_WH = 200
    has_wh = ("width" in merged.columns) and ("height" in merged.columns)
    has_nf = "is_no_face" in merged.columns

    def derive_gate_row(row) -> str:
        if not (has_wh and has_nf):
            return "unknown"
        w = row.get("width")
        h = row.get("height")
        nf = row.get("is_no_face")
        wh_ok = (pd.notna(w) and pd.notna(h))
        nf_known = pd.notna(nf)
        if not (wh_ok and nf_known):
            return "unknown"
        is_no_face = bool(nf)
        is_lowres = (min(w, h) < G2_MIN_WH)
        if is_no_face and is_lowres:
            return "drop_both"
        if is_no_face:
            return "drop_no_face"
        if is_lowres:
            return "drop_lowres"
        return "pass"

    merged["gate_status"] = merged.apply(derive_gate_row, axis=1)

    # Coverage report
    covered = merged["face_area_ratio"].notna().sum()
    log.info(
        f"  parquet coverage on manifest: {covered}/{len(merged)} "
        f"({covered/len(merged)*100:.1f}%)"
    )
    gate_counts = Counter(merged["gate_status"])
    log.info(
        f"  gate-status distribution: pass={gate_counts['pass']} "
        f"drop_no_face={gate_counts['drop_no_face']} "
        f"drop_lowres={gate_counts['drop_lowres']} "
        f"drop_both={gate_counts['drop_both']} "
        f"unknown={gate_counts['unknown']}"
    )
    for s, sub in merged.groupby("suite"):
        c = sub["face_area_ratio"].notna().sum()
        log.info(
            f"    suite {s}: {c}/{len(sub)} ({c/len(sub)*100:.1f}%) "
            f"face_size={Counter(sub['face_size'])} "
            f"quality={Counter(sub['quality'])}"
        )

    return merged


# ---------------------------------------------------------------------------
# Score & sort enrichment
# ---------------------------------------------------------------------------

# Sort-priority maps. Lower numeric value sorts first.
FACE_SIZE_RANK = {"close": 0, "far": 1, "unknown": 2}
QUALITY_RANK = {"hi-q": 0, "lo-q": 1, "unknown": 2}


def enrich_with_scores(
    df: pd.DataFrame,
    score_specs: list[tuple[str, list[str]]],
    log: logging.Logger,
) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    """Attach a column `score_<NAME>` per ckpt and report per-suite coverage.

    Returns (enriched_df, ckpt_names_in_order, coverage_summary).
    """

    df = df.copy()
    names_in_order: list[str] = []
    coverage_summary: dict[str, float] = {}

    for name, paths in score_specs:
        names_in_order.append(name)
        lookup = load_score_lookup(name, paths, log)
        col = f"score_{name}"
        df[col] = df["frame_path"].map(lookup)
        covered = df[col].notna().sum()
        pct = covered / len(df) * 100 if len(df) > 0 else 0.0
        coverage_summary[name] = pct
        log.info(
            f"  score [{name}] coverage on manifest: {covered}/{len(df)} ({pct:.2f}%)"
        )
        if covered < len(df):
            missing_rows = df[df[col].isna()]
            for s, sub in missing_rows.groupby("suite"):
                log.info(
                    f"    suite {s}: {len(sub)} frames missing [{name}] score"
                )
    return df, names_in_order, coverage_summary


# ---------------------------------------------------------------------------
# Model-selection analytics
# ---------------------------------------------------------------------------


def _operating_metrics(
    df: pd.DataFrame, score_col: str, tau: float
) -> dict:
    """At threshold tau, return overall real FPR + per-fake-suite recall.

    Skips rows with NaN scores.  Returns {'fpr', 'recall_overall',
    'per_fake_suite': {suite: recall}, 'n_real', 'n_fake'}.
    """
    sub = df[df[score_col].notna()]
    real = sub[sub["label"] == 0]
    fake = sub[sub["label"] == 1]
    n_real = len(real)
    n_fake = len(fake)
    fpr = float((real[score_col] >= tau).sum()) / max(n_real, 1)
    recall_overall = float((fake[score_col] >= tau).sum()) / max(n_fake, 1)
    per_suite = {}
    for suite, ssub in fake.groupby("suite"):
        per_suite[str(suite)] = (
            float((ssub[score_col] >= tau).sum()) / max(len(ssub), 1)
        )
    return {
        "fpr": fpr,
        "recall_overall": recall_overall,
        "per_fake_suite": per_suite,
        "n_real": n_real,
        "n_fake": n_fake,
    }


def compute_analytics(
    df: pd.DataFrame, ckpt_names: list[str], log: logging.Logger
) -> dict:
    """Build the data for the model-selection analytics section.

    Returns dict with three keys: 'operating_points', 'per_suite_at_5fpr',
    'quality_discard'.  Each is a list of row dicts ready for HTML rendering.
    """
    if not ckpt_names:
        return {"operating_points": [], "per_suite_at_5fpr": [],
                "quality_discard": [], "fake_suites": [], "real_suites": []}

    fake_suites = sorted(df.loc[df["label"] == 1, "suite"].unique().tolist())
    real_suites = sorted(df.loc[df["label"] == 0, "suite"].unique().tolist())

    # ---- Per-ckpt tau at 5% real FPR ------------------------------------
    tau_5fpr_per_ckpt: dict[str, float] = {}
    for n in ckpt_names:
        col = f"score_{n}"
        real_scores = df.loc[(df["label"] == 0) & df[col].notna(), col].values
        if len(real_scores) == 0:
            tau_5fpr_per_ckpt[n] = 0.5
            continue
        # τ at 5% real FPR = 95th percentile of real scores
        tau_5fpr_per_ckpt[n] = float(np.percentile(real_scores, 95))

    # ---- Table A: operating points across multiple thresholds -----------
    operating_points: list[dict] = []
    fixed_taus = [0.50, 0.70, 0.90]
    for n in ckpt_names:
        col = f"score_{n}"
        tau_set = sorted(set(fixed_taus + [round(tau_5fpr_per_ckpt[n], 4)]))
        for tau in tau_set:
            m = _operating_metrics(df, col, tau)
            operating_points.append({
                "ckpt": n,
                "tau": tau,
                "tau_label": (
                    f"τ_5%FPR ({tau:.3f})"
                    if abs(tau - tau_5fpr_per_ckpt[n]) < 1e-6
                    else f"{tau:.2f}"
                ),
                "fpr": m["fpr"],
                "recall_overall": m["recall_overall"],
                "per_fake_suite": m["per_fake_suite"],
                "n_real": m["n_real"],
                "n_fake": m["n_fake"],
            })

    # ---- Table B: per-suite recall @ τ_5%FPR per ckpt -------------------
    per_suite_at_5fpr: list[dict] = []
    for n in ckpt_names:
        col = f"score_{n}"
        tau = tau_5fpr_per_ckpt[n]
        m = _operating_metrics(df, col, tau)
        per_suite_at_5fpr.append({
            "ckpt": n,
            "tau": tau,
            "fpr": m["fpr"],
            "recall_overall": m["recall_overall"],
            "per_fake_suite": m["per_fake_suite"],
            # also per-real-suite FPR breakdown (which real subset hurts most)
            "per_real_suite_fpr": {
                str(s): float(
                    (df[(df["label"] == 0) & (df["suite"] == s) & df[col].notna()][col] >= tau).sum()
                ) / max(len(df[(df["label"] == 0) & (df["suite"] == s) & df[col].notna()]), 1)
                for s in real_suites
            },
        })

    # ---- Table C: quality-discard sweep at τ_5%FPR -----------------------
    # Operates on the IQ-tagged subset (quality != 'unknown' AND face_size != 'unknown').
    iq_mask = (df["quality"].astype(str) != "unknown") & (df["face_size"].astype(str) != "unknown")
    iq_df = df[iq_mask].copy()
    log.info(
        f"  analytics: IQ-tagged subset = {len(iq_df)}/{len(df)} frames "
        f"({100 * len(iq_df) / max(len(df), 1):.1f}%)"
    )

    scenarios = [
        ("All IQ-tagged frames", lambda d: d),
        ("Drop quality=lo-q", lambda d: d[d["quality"].astype(str) != "lo-q"]),
        ("Drop face_size=far", lambda d: d[d["face_size"].astype(str) != "far"]),
        ("Drop both lo-q & far", lambda d: d[
            (d["quality"].astype(str) != "lo-q")
            & (d["face_size"].astype(str) != "far")
        ]),
    ]

    quality_discard: list[dict] = []
    for n in ckpt_names:
        col = f"score_{n}"
        # Use τ_5%FPR calibrated on the FULL real set (not just IQ subset),
        # so the threshold reflects deployment policy and the lever shows
        # what happens at a fixed operating point.
        tau = tau_5fpr_per_ckpt[n]
        for scen_label, fn in scenarios:
            sub = fn(iq_df)
            n_kept = len(sub)
            n_kept_real = int((sub["label"] == 0).sum())
            n_kept_fake = int((sub["label"] == 1).sum())
            if n_kept == 0:
                quality_discard.append({
                    "ckpt": n, "tau": tau, "scenario": scen_label,
                    "n_kept": 0, "fpr": float("nan"),
                    "recall_overall": float("nan"),
                    "n_real": 0, "n_fake": 0,
                })
                continue
            m = _operating_metrics(sub, col, tau)
            quality_discard.append({
                "ckpt": n, "tau": tau, "scenario": scen_label,
                "n_kept": n_kept,
                "fpr": m["fpr"],
                "recall_overall": m["recall_overall"],
                "n_real": n_kept_real, "n_fake": n_kept_fake,
            })

    return {
        "operating_points": operating_points,
        "per_suite_at_5fpr": per_suite_at_5fpr,
        "quality_discard": quality_discard,
        "fake_suites": fake_suites,
        "real_suites": real_suites,
        "tau_5fpr_per_ckpt": tau_5fpr_per_ckpt,
    }


def _fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{100 * x:.1f}%"


def _bar(x: float, max_x: float = 1.0, color: str = "var(--score-good)") -> str:
    """Inline horizontal bar for a 0..max_x value (no JS)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    pct = max(0.0, min(1.0, x / max(max_x, 1e-9)))
    return (
        f'<span class="ana-bar" style="--w:{pct * 100:.1f}%; '
        f'background-image: linear-gradient(to right, {color} 0 var(--w), '
        f'transparent var(--w) 100%);"></span>'
    )


def build_analytics_html(analytics: dict) -> str:
    """Render the three analytics tables to HTML."""
    if not analytics["operating_points"]:
        return ""

    fake_suites = analytics["fake_suites"]
    real_suites = analytics["real_suites"]

    # ---- Table A: operating points --------------------------------------
    # Group rows per model into a <tbody> so CSS can paint alternating bands.
    # Each recompute-able cell carries data attributes consumed by the JS
    # recomputeAnalytics() function, so identity-level filters update the
    # numbers in-place without rebuilding the page.
    rows_by_ckpt_a: dict[str, list[str]] = {}
    for r in analytics["operating_points"]:
        ckpt = r["ckpt"]
        tau = r["tau"]
        per_suite_cells = []
        for s in fake_suites:
            v = r["per_fake_suite"].get(s)
            cell = (
                f'<td class="num" data-cell-type="recall-per-suite" '
                f'data-ckpt="{html_escape(ckpt)}" data-tau="{tau:.6f}" '
                f'data-suite-filter="{html_escape(s)}">'
                f'{_fmt_pct(v)} {_bar(v or 0.0)}</td>'
                if v is not None else '<td class="num">—</td>'
            )
            per_suite_cells.append(cell)
        row = (
            f'<tr>'
            f'<td class="ck">{html_escape(ckpt)}</td>'
            f'<td class="num tau">{html_escape(r["tau_label"])}</td>'
            f'<td class="num fpr" data-cell-type="fpr" '
            f'data-ckpt="{html_escape(ckpt)}" data-tau="{tau:.6f}">'
            f'{_fmt_pct(r["fpr"])} {_bar(r["fpr"], 0.5, "var(--score-bad)")}</td>'
            f'<td class="num" data-cell-type="recall" '
            f'data-ckpt="{html_escape(ckpt)}" data-tau="{tau:.6f}">'
            f'{_fmt_pct(r["recall_overall"])} {_bar(r["recall_overall"])}</td>'
            f'{"".join(per_suite_cells)}'
            f'</tr>'
        )
        rows_by_ckpt_a.setdefault(ckpt, []).append(row)
    a_thead = (
        '<tr><th>Model</th><th>τ</th><th>Real FPR</th><th>Fake recall (overall)</th>'
        + "".join(f'<th>{html_escape(s)}</th>' for s in fake_suites)
        + '</tr>'
    )
    a_tbodies = "".join(
        f'<tbody class="model-group" data-ckpt="{html_escape(c)}">{"".join(rs)}</tbody>'
        for c, rs in rows_by_ckpt_a.items()
    )
    table_a = (
        f'<table class="ana"><thead>{a_thead}</thead>{a_tbodies}</table>'
    )

    # ---- Table B: per-suite breakdown @ τ_5%FPR --------------------------
    rows_by_ckpt_b: dict[str, list[str]] = {}
    for r in analytics["per_suite_at_5fpr"]:
        ckpt = r["ckpt"]
        tau = r["tau"]
        per_fake_cells = []
        for s in fake_suites:
            v = r["per_fake_suite"].get(s)
            per_fake_cells.append(
                f'<td class="num" data-cell-type="recall-per-suite" '
                f'data-ckpt="{html_escape(ckpt)}" data-tau="{tau:.6f}" '
                f'data-suite-filter="{html_escape(s)}">'
                f'{_fmt_pct(v)} {_bar(v or 0.0)}</td>'
                if v is not None else '<td class="num">—</td>'
            )
        per_real_cells = []
        for s in real_suites:
            v = r["per_real_suite_fpr"].get(s)
            per_real_cells.append(
                f'<td class="num" data-cell-type="fpr-per-suite" '
                f'data-ckpt="{html_escape(ckpt)}" data-tau="{tau:.6f}" '
                f'data-suite-filter="{html_escape(s)}">'
                f'{_fmt_pct(v)} {_bar(v or 0.0, 0.5, "var(--score-bad)")}</td>'
                if v is not None else '<td class="num">—</td>'
            )
        row = (
            f'<tr>'
            f'<td class="ck">{html_escape(ckpt)}</td>'
            f'<td class="num tau">{tau:.3f}</td>'
            f'<td class="num fpr" data-cell-type="fpr" '
            f'data-ckpt="{html_escape(ckpt)}" data-tau="{tau:.6f}">'
            f'{_fmt_pct(r["fpr"])}</td>'
            f'<td class="num" data-cell-type="recall" '
            f'data-ckpt="{html_escape(ckpt)}" data-tau="{tau:.6f}">'
            f'{_fmt_pct(r["recall_overall"])} {_bar(r["recall_overall"])}</td>'
            f'{"".join(per_fake_cells)}'
            f'{"".join(per_real_cells)}'
            f'</tr>'
        )
        rows_by_ckpt_b.setdefault(ckpt, []).append(row)
    b_thead = (
        '<tr>'
        '<th>Model</th><th>τ_5%FPR</th><th>Real FPR (actual)</th><th>Fake recall</th>'
        + "".join(f'<th class="fakehdr">{html_escape(s)}</th>' for s in fake_suites)
        + "".join(f'<th class="realhdr">{html_escape(s)}</th>' for s in real_suites)
        + '</tr>'
    )
    b_tbodies = "".join(
        f'<tbody class="model-group" data-ckpt="{html_escape(c)}">{"".join(rs)}</tbody>'
        for c, rs in rows_by_ckpt_b.items()
    )
    table_b = (
        f'<table class="ana"><thead>{b_thead}</thead>{b_tbodies}</table>'
    )

    # ---- Table C: quality-discard sweep ---------------------------------
    SCENARIO_KEY = {
        "All IQ-tagged frames": "iq_all",
        "Drop quality=lo-q": "iq_drop_loq",
        "Drop face_size=far": "iq_drop_far",
        "Drop both lo-q & far": "iq_drop_both",
    }
    rows_by_ckpt_c: dict[str, list[str]] = {}
    for r in analytics["quality_discard"]:
        ckpt = r["ckpt"]
        tau = r["tau"]
        scen_key = SCENARIO_KEY.get(r["scenario"], "iq_all")
        row = (
            f'<tr data-scenario="{scen_key}">'
            f'<td class="ck">{html_escape(ckpt)}</td>'
            f'<td>{html_escape(r["scenario"])}</td>'
            f'<td class="num" data-cell-type="kept" data-scenario="{scen_key}">'
            f'{r["n_kept"]} '
            f'<span class="muted">(R={r["n_real"]}, F={r["n_fake"]})</span></td>'
            f'<td class="num fpr" data-cell-type="fpr" data-scenario="{scen_key}" '
            f'data-ckpt="{html_escape(ckpt)}" data-tau="{tau:.6f}">'
            f'{_fmt_pct(r["fpr"])} {_bar(r["fpr"], 0.5, "var(--score-bad)")}</td>'
            f'<td class="num" data-cell-type="recall" data-scenario="{scen_key}" '
            f'data-ckpt="{html_escape(ckpt)}" data-tau="{tau:.6f}">'
            f'{_fmt_pct(r["recall_overall"])} {_bar(r["recall_overall"])}</td>'
            f'</tr>'
        )
        rows_by_ckpt_c.setdefault(ckpt, []).append(row)
    c_thead = (
        '<tr><th>Model</th><th>Scenario</th><th>Frames kept</th>'
        '<th>Real FPR</th><th>Fake recall</th></tr>'
    )
    c_tbodies = "".join(
        f'<tbody class="model-group" data-ckpt="{html_escape(c)}">{"".join(rs)}</tbody>'
        for c, rs in rows_by_ckpt_c.items()
    )
    table_c = (
        f'<table class="ana"><thead>{c_thead}</thead>{c_tbodies}</table>'
    )

    return (
        '<section id="analytics-section">\n'
        '  <details open>\n'
        '    <summary>Model selection — operating points and trade-offs</summary>\n'
        '    <div class="ana-body">\n'
        '      <h3>A. Operating points by threshold</h3>\n'
        '      <p class="ana-help">For each model × τ choice, real FPR (false-positive rate on '
        'all real suites) and fake recall (overall + per fake suite). '
        'τ_5%FPR is calibrated per-model on this run\'s real frames. '
        'Lower FPR + higher recall = better.</p>\n'
        f'      {table_a}\n'
        '      <h3>B. Per-suite breakdown at τ_5%FPR</h3>\n'
        '      <p class="ana-help">At each model\'s 5%-FPR-calibrated threshold: '
        'fake recall per fake-suite (where each model shines) and FPR per real-suite '
        '(which real cohort hurts most for this model).</p>\n'
        f'      {table_b}\n'
        '      <h3>C. Quality-discard sweep at τ_5%FPR</h3>\n'
        '      <p class="ana-help">Limited to frames with IQ tags '
        '(face_size + quality known). Shows what happens to the operating point '
        'when low-quality / far-face frames are pre-filtered out.</p>\n'
        f'      {table_c}\n'
        '    </div>\n'
        '  </details>\n'
        '</section>\n'
    )


# ---------------------------------------------------------------------------
# GCS download
# ---------------------------------------------------------------------------


def plan_downloads(
    df: pd.DataFrame, output_root: Path
) -> tuple[list[tuple[str, Path]], int]:
    """Return list of (gcs_uri, local_full_path) for files NOT already present.

    Also returns the count already cached.
    """

    pending: list[tuple[str, Path]] = []
    cached = 0
    for _, row in df.iterrows():
        full, _thumb = local_paths(row, output_root)
        if full.exists() and full.stat().st_size > 0:
            cached += 1
            continue
        full.parent.mkdir(parents=True, exist_ok=True)
        pending.append((row["frame_path"], full))
    return pending, cached


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"not a gs:// URI: {uri}")
    rest = uri[5:]
    bucket, _, key = rest.partition("/")
    return bucket, key


def download_via_gcs_client(
    pending: list[tuple[str, Path]],
    log: logging.Logger,
) -> tuple[int, int]:
    """Download pending files via the google-cloud-storage Python client.

    Much faster than spawning per-file gsutil subprocesses (no per-file
    auth + subprocess startup cost).  Uses ThreadPoolExecutor since the
    GCS client is thread-safe and downloads are I/O-bound.
    """

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from google.cloud import storage

    ok = 0
    failed = 0
    n = len(pending)
    if n == 0:
        return 0, 0

    log.info(
        f"Downloading {n} frames via google-cloud-storage "
        f"({DEFAULT_DOWNLOAD_WORKERS} threads)…"
    )

    # One client shared across threads; bucket cache is also thread-safe.
    client = storage.Client()
    bucket_cache: dict[str, "storage.Bucket"] = {}

    def _bucket(name: str):
        b = bucket_cache.get(name)
        if b is None:
            b = client.bucket(name)
            bucket_cache[name] = b
        return b

    def _one(uri: str, dest: Path) -> tuple[bool, str]:
        try:
            bname, key = _parse_gs_uri(uri)
            blob = _bucket(bname).blob(key)
            tmp = dest.with_suffix(dest.suffix + ".part")
            blob.download_to_filename(str(tmp))
            os.replace(tmp, dest)
            return True, ""
        except Exception as exc:  # noqa: BLE001
            return False, f"{uri} -> {exc!r}"

    with ThreadPoolExecutor(max_workers=DEFAULT_DOWNLOAD_WORKERS) as ex:
        futures = {ex.submit(_one, uri, dest): (uri, dest) for uri, dest in pending}
        last_log = time.monotonic()
        completed = 0
        for fut in as_completed(futures):
            uri, dest = futures[fut]
            success, err = fut.result()
            completed += 1
            if success:
                ok += 1
            else:
                failed += 1
                log.warning(f"download FAILED: {err}")
            now = time.monotonic()
            if now - last_log > 5:
                log.info(
                    f"  download progress: {completed}/{n} "
                    f"ok={ok} fail={failed} "
                    f"({completed / max(1, now - download_started_at()):.1f}/s)"
                )
                last_log = now
    log.info(f"  download done: ok={ok} fail={failed}")
    return ok, failed


_download_t0_holder: list[float] = []


def download_started_at() -> float:
    if not _download_t0_holder:
        _download_t0_holder.append(time.monotonic())
    return _download_t0_holder[0]


# ---------------------------------------------------------------------------
# Thumbnail generation
# ---------------------------------------------------------------------------


def _thumb_one(args: tuple[str, str]) -> tuple[bool, str]:
    src, dst = args
    try:
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return True, ""
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        img = Image.open(src)
        img = ImageOps.exif_transpose(img)
        if img.mode not in {"RGB", "L"}:
            img = img.convert("RGB")
        img.thumbnail((THUMBNAIL_MAX_DIM, THUMBNAIL_MAX_DIM), Image.LANCZOS)
        img.save(dst, format="JPEG", quality=THUMBNAIL_QUALITY, optimize=True)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{src} -> {exc!r}"


def make_thumbnails(
    df: pd.DataFrame, output_root: Path, log: logging.Logger
) -> tuple[int, int, int]:
    """Generate thumbs for every row (skips when thumb already exists)."""

    plan: list[tuple[str, str]] = []
    cached = 0
    missing_src = 0
    for _, row in df.iterrows():
        full, thumb = local_paths(row, output_root)
        if not full.exists():
            missing_src += 1
            continue
        if thumb.exists() and thumb.stat().st_size > 0:
            cached += 1
            continue
        plan.append((str(full), str(thumb)))
    if not plan:
        log.info(f"  thumbnails already cached: {cached} (missing src: {missing_src})")
        return 0, 0, missing_src
    log.info(
        f"Thumbnailing {len(plan)} new frames via Pool({DEFAULT_THUMB_WORKERS})… "
        f"already cached: {cached}"
    )
    ok = 0
    failed = 0
    with mp.get_context("spawn").Pool(DEFAULT_THUMB_WORKERS) as pool:
        for i, (success, err) in enumerate(
            pool.imap_unordered(_thumb_one, plan, chunksize=16), 1
        ):
            if success:
                ok += 1
            else:
                failed += 1
                log.warning(f"thumb FAILED: {err}")
            if i % 500 == 0:
                log.info(f"  thumb progress: {i}/{len(plan)} ok={ok} fail={failed}")
    log.info(f"  thumb done: ok={ok} fail={failed}")
    return ok, failed, missing_src


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


# CSS uses 6-column grid + a small score-table that can host arbitrary rows.
PAGE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Identity Frame Browser - {generated}</title>
<style>
  :root {{
    --bg:#111317; --card:#1b1f26; --text:#e7eaee; --muted:#8a93a3;
    --real:#3aa775; --fake:#d35454; --lockbox:#e2b54e; --accent:#5aa3ff;
    --border:#2c323b;
    --score-good:#3aa775; --score-bad:#ff6b6b;
    --pill-bg:#222831; --pill-border:#3a414c;
  }}
  /* Colorblind-safe palette: bold blue (correct) + bold orange (wrong).
     Distinguishable across deuteranopia/protanopia/tritanopia.
     Activated when <body> has the .color-mode-cb class. */
  body.color-mode-cb {{
    --score-good:#1976d2;  /* bold blue */
    --score-bad:#ff7f0e;   /* bold orange */
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ background: var(--bg); color: var(--text); margin:0;
    font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }}
  header {{ position: sticky; top:0; z-index: 5; background: rgba(17,19,23,0.95);
    backdrop-filter: blur(8px); padding: 12px 18px; border-bottom: 1px solid var(--border); }}
  h1 {{ margin:0 0 6px 0; font-size: 18px; font-weight: 600; }}
  .meta {{ color: var(--muted); font-size: 12px; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius: 99px;
    border:1px solid var(--border); margin-right: 6px; font-size: 11px; }}
  .badge.real {{ color: var(--real); border-color: var(--real); }}
  .badge.fake {{ color: var(--fake); border-color: var(--fake); }}
  .badge.lockbox {{ color: var(--lockbox); border-color: var(--lockbox); }}
  .badge.chronic {{ color: #ff8a65; border-color:#ff8a65; }}
  .controls {{ display:flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; margin-top:8px; }}
  .controls input[type=text] {{ background: var(--card); border: 1px solid var(--border);
    color: var(--text); padding: 6px 10px; border-radius: 6px; min-width: 240px; }}
  .controls label {{ color: var(--muted); font-size: 12px; }}
  .controls .grouplbl {{ color: var(--text); margin-right: 4px; font-weight: 600; }}
  .legend {{ font-size: 11px; color: var(--muted); margin-top: 4px; }}
  main {{ padding: 16px 18px 80px 18px; }}
  /* Errors section — dynamic FP/FN review for the highlighted ckpt. */
  #errors-section {{ background: var(--card);
    border: 1px solid var(--border);
    border-bottom: 2px solid var(--accent);
    border-radius: 10px;
    padding: 10px 14px 14px 14px;
    margin: 0 0 28px 0;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
  }}
  #errors-section.hidden {{ display: none !important; }}
  /* Busy state while repopulateErrorsSection is in flight (deferred via
     requestAnimationFrame so counters can paint first). */
  #errors-section.errors-busy {{ opacity: 0.55; transition: opacity 120ms ease; }}
  #errors-section.errors-busy::after {{ content: "calculating…"; position: absolute;
    top: 8px; right: 14px; font-size: 11px; color: var(--muted);
    font-style: italic;
  }}
  #errors-section {{ position: relative; }}
  /* Threshold-trajectory charts — small multiples, one panel per ckpt. */
  #threshold-charts {{ background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 6px 14px 14px 14px;
    margin: 0 0 18px 0;
    box-shadow: 0 4px 14px rgba(0,0,0,0.18);
  }}
  #threshold-charts > details > summary {{ cursor: pointer; padding: 6px 0;
    font-size: 13px; font-weight: 600; letter-spacing: 0.02em;
    color: var(--text);
  }}
  #tcharts-toolbar {{ display: flex; gap: 16px; align-items: center;
    flex-wrap: wrap; margin: 6px 0 8px 0; font-size: 11px; color: var(--muted);
  }}
  .tcharts-legend-item {{ display: inline-flex; align-items: center; gap: 6px; }}
  .tcharts-legend-item .dot {{ display: inline-block; width: 22px; height: 0;
    border-bottom: 2px solid; vertical-align: middle;
  }}
  .tcharts-legend-item .dot.fpr {{ border-bottom-style: dashed;
    border-color: var(--score-bad);
  }}
  .tcharts-legend-item .dot.recall {{ border-color: var(--score-good); }}
  .tcharts-tau-readout {{ font-variant-numeric: tabular-nums;
    color: var(--text); font-weight: 600; font-size: 12px;
  }}
  #tcharts-refresh {{ background: var(--bg); color: var(--muted);
    border: 1px solid var(--border); border-radius: 4px; padding: 2px 8px;
    font-size: 11px; cursor: pointer; margin-left: auto;
  }}
  #tcharts-refresh:hover {{ color: var(--accent); border-color: var(--accent); }}
  #tcharts-grid {{ display: grid; gap: 14px;
    grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
  }}
  .tchart-panel {{ background: var(--bg); border: 1px solid var(--border);
    border-radius: 6px; padding: 10px 12px 8px 12px; min-height: 240px;
    display: flex; flex-direction: column;
  }}
  .tchart-header {{ display: flex; justify-content: space-between;
    align-items: baseline; gap: 12px; flex-wrap: wrap;
    margin-bottom: 4px;
  }}
  .tchart-title {{ font-size: 13px; font-weight: 700; color: var(--text);
    letter-spacing: 0.04em; text-transform: uppercase;
  }}
  .tchart-title .n {{ color: var(--muted); font-weight: 400;
    text-transform: none; font-size: 11px; margin-left: 6px;
  }}
  .tchart-readout {{ font-size: 12px; font-variant-numeric: tabular-nums;
    color: var(--text); display: flex; gap: 14px;
  }}
  .tchart-readout .v-fpr {{ color: var(--score-bad); font-weight: 600; }}
  .tchart-readout .v-recall {{ color: var(--score-good); font-weight: 600; }}
  .tchart-readout .v-gap {{ color: var(--accent); font-weight: 600;
    margin-left: 4px; padding-left: 8px; border-left: 1px solid var(--border);
  }}
  .tchart-svg {{ width: 100%; height: 200px; cursor: crosshair;
    display: block;
  }}
  .tchart-svg .grid-line {{ stroke: var(--border); stroke-width: 1;
    stroke-dasharray: 2 3; opacity: 0.5;
  }}
  .tchart-svg .axis-label {{ font-size: 11px; fill: var(--muted); }}
  .tchart-svg .curve-fpr {{ fill: none; stroke: var(--score-bad);
    stroke-width: 2; stroke-dasharray: 5 3;
  }}
  .tchart-svg .curve-recall {{ fill: none; stroke: var(--score-good);
    stroke-width: 2;
  }}
  .tchart-svg .tau-line {{ stroke: var(--accent); stroke-width: 1.2;
    stroke-dasharray: 2 3;
  }}
  .tchart-svg .tau-dot {{ fill: var(--accent); stroke: var(--card);
    stroke-width: 1.5;
  }}
  .tchart-svg .hover-line {{ stroke: var(--muted); stroke-width: 1;
    stroke-dasharray: 1 2; opacity: 0; pointer-events: none;
  }}
  .tchart-svg.hovering .hover-line {{ opacity: 0.9; }}
  .tchart-empty {{ font-size: 12px; color: var(--muted);
    font-style: italic; padding: 32px 0; text-align: center;
  }}
  #tcharts-help {{ font-size: 10px; color: var(--muted); margin-top: 8px;
    font-style: italic;
  }}
  /* Analytics section — model selection trade-offs (top of page). */
  #analytics-section {{ background: var(--card);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent);
    border-radius: 10px;
    padding: 0 14px 14px 14px;
    margin: 0 0 24px 0;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
  }}
  #analytics-section > details > summary {{ list-style: none; cursor: pointer;
    padding: 12px 4px; font-size: 14px; font-weight: 600;
    user-select: none; color: var(--accent);
  }}
  #analytics-section > details > summary::-webkit-details-marker {{ display: none; }}
  #analytics-section > details > summary::before {{ content: "\\25B6";
    color: var(--accent); transition: transform .15s ease-in;
    display: inline-block; width: 14px; margin-right: 4px; font-size: 11px;
  }}
  #analytics-section > details[open] > summary::before {{ transform: rotate(90deg); }}
  #analytics-section .ana-body {{ padding-top: 4px; }}
  #analytics-section h3 {{ margin: 18px 0 4px 0; font-size: 13px; text-transform: uppercase;
    letter-spacing: .04em; color: var(--text); }}
  #analytics-section .ana-help {{ color: var(--muted); font-size: 11px;
    margin: 0 0 8px 0; max-width: 880px; line-height: 1.5; }}
  table.ana {{ border-collapse: collapse; width: 100%; font-size: 12px;
    margin-bottom: 10px; font-variant-numeric: tabular-nums; }}
  table.ana th, table.ana td {{ border: 1px solid rgba(255, 255, 255, 0.12);
    padding: 5px 8px; text-align: left; vertical-align: middle; }}
  table.ana thead th {{ background: var(--bg); color: var(--muted);
    font-weight: 600; font-size: 11px; text-transform: uppercase;
    letter-spacing: .03em; position: sticky; top: 0; }}
  table.ana td.num {{ text-align: right; }}
  table.ana td.tau {{ color: var(--accent); font-weight: 600; }}
  table.ana td.fpr {{ color: var(--score-bad); }}
  table.ana td.ck {{ font-weight: 600; }}
  table.ana th.fakehdr {{ color: var(--fake); }}
  table.ana th.realhdr {{ color: var(--real); }}
  table.ana .muted {{ color: var(--muted); font-size: 10px; }}
  /* Per-model colored bands — 3-hue cycle so each model's rows read as a
     unit.  Opacity tuned for the dark theme: subtle but unambiguous. */
  table.ana tbody.model-group:nth-of-type(3n+1) td {{ background: rgba(90, 163, 255, 0.13); }}   /* blue */
  table.ana tbody.model-group:nth-of-type(3n+2) td {{ background: rgba(214, 134, 230, 0.13); }}  /* magenta */
  table.ana tbody.model-group:nth-of-type(3n+3) td {{ background: rgba(120, 210, 150, 0.13); }}  /* green */
  /* Matching left-edge accent on the model-name column for the same hue. */
  table.ana tbody.model-group:nth-of-type(3n+1) td.ck {{ border-left: 4px solid #5aa3ff; }}
  table.ana tbody.model-group:nth-of-type(3n+2) td.ck {{ border-left: 4px solid #d686e6; }}
  table.ana tbody.model-group:nth-of-type(3n+3) td.ck {{ border-left: 4px solid #78d296; }}
  /* Stronger top border on each group's first row to separate models visually. */
  table.ana tbody.model-group + tbody.model-group td {{ border-top: 2px solid var(--accent); }}
  .ana-bar {{ display: inline-block; width: 60px; height: 8px;
    margin-left: 6px; vertical-align: middle;
    background-color: var(--border); border-radius: 2px; }}
  #errors-section details.errors-group {{ background: transparent;
    border: 1px solid var(--border);
    border-radius: 8px;
    margin: 8px 0;
    overflow: hidden;
  }}
  #errors-section details.errors-group > summary {{ list-style: none;
    cursor: pointer; padding: 8px 12px;
    display: flex; gap: 8px; align-items: center; user-select: none;
    font-size: 13px; font-weight: 600;
  }}
  #errors-section details.errors-group > summary::-webkit-details-marker {{ display: none; }}
  #errors-section details.errors-group > summary::before {{ content: "\\25B6";
    color: var(--muted); transition: transform .15s ease-in;
    display: inline-block; width: 12px; font-size: 10px;
  }}
  #errors-section details.errors-group[open] > summary::before {{ transform: rotate(90deg); }}
  #errors-section #errors-fp > summary {{ color: var(--score-bad); }}
  #errors-section #errors-fn > summary {{ color: var(--lockbox); }}
  #errors-section .count {{ color: var(--text); font-variant-numeric: tabular-nums; }}
  #errors-section .errors-grid {{ display:grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 8px; padding: 8px 12px 12px 12px;
  }}
  @media (max-width: 1100px) {{ #errors-section .errors-grid {{ grid-template-columns: repeat(4, minmax(0,1fr)); }} }}
  @media (max-width: 720px)  {{ #errors-section .errors-grid {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
  #errors-section .errors-identity-header {{
    grid-column: 1 / -1;
    padding: 8px 4px 4px 4px;
    margin-top: 6px;
    margin-bottom: 2px;
    border-bottom: 1px solid var(--border, #2a2f3a);
    font-weight: 600;
    color: var(--text);
    display: flex;
    align-items: baseline;
    gap: 10px;
    font-variant-numeric: tabular-nums;
  }}
  #errors-section .errors-identity-header .ident {{
    color: var(--text);
  }}
  #errors-section .errors-identity-header .fraction {{
    color: var(--score-bad);
    font-weight: 700;
  }}
  #errors-section #errors-fn .errors-identity-header .fraction {{
    color: var(--lockbox);
  }}
  #errors-section .errors-identity-header .pct {{
    color: var(--muted);
    font-weight: 400;
    font-size: 0.92em;
  }}
  #errors-section .errors-empty {{ color: var(--muted); font-style: italic;
    padding: 10px 14px; font-size: 12px;
  }}
  details.identity {{ background: var(--card); border:1px solid var(--border);
    border-radius: 10px; margin: 14px 0; overflow: hidden; }}
  details.identity > summary {{ list-style: none; cursor: pointer; padding: 12px 16px;
    display: flex; gap: 12px; align-items: center; user-select: none; }}
  details.identity > summary::-webkit-details-marker {{ display: none; }}
  details.identity > summary::before {{ content: "\\25B6"; color: var(--muted);
    transition: transform .15s ease-in; display: inline-block; width: 14px; }}
  details.identity[open] > summary::before {{ transform: rotate(90deg); }}
  details.identity .ident-name {{ font-weight: 700; font-size: 16px; }}
  details.identity .ident-toggle {{ margin: 0 8px 0 0; cursor: pointer;
    width: 14px; height: 14px; flex-shrink: 0; }}
  details.identity.id-deselected {{ opacity: 0.55; border-color: var(--border); }}
  details.identity.id-deselected > summary {{ background: rgba(255,255,255,0.02); }}
  details.identity.id-deselected .ident-name {{ text-decoration: line-through;
    color: var(--muted); }}
  .ids-summary {{ color: var(--muted); font-size: 11px; }}
  details.identity .ident-counts {{ color: var(--muted); font-size: 12px; }}
  .label-block {{ padding: 8px 14px 14px 14px; border-top: 1px solid var(--border); }}
  .label-block h3 {{ margin: 8px 0 4px 0; font-size: 13px; text-transform: uppercase;
    letter-spacing: .04em; color: var(--muted); }}
  .label-block.real h3 {{ color: var(--real); }}
  .label-block.fake h3 {{ color: var(--fake); }}
  .subbucket {{ margin: 8px 0; }}
  .subbucket-header {{ font-size: 11px; color: var(--muted); padding: 2px 4px;
    margin: 6px 0 4px 0; border-left: 2px solid var(--border); padding-left: 8px;
    text-transform: uppercase; letter-spacing: .03em; }}
  .subbucket-header b {{ color: var(--text); font-weight: 600; }}
  .grid {{ display:grid; grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 8px; }}
  @media (max-width: 1100px) {{ .grid {{ grid-template-columns: repeat(4, minmax(0,1fr)); }} }}
  @media (max-width: 720px)  {{ .grid {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
  .frame {{ background: #14171c; border:1px solid var(--border); border-radius: 6px;
    overflow: hidden; display: flex; flex-direction: column; text-decoration: none;
    color: inherit; }}
  .frame img {{ display: block; width: 100%; height: 160px; object-fit: cover;
    background: #0c0e12; }}
  .frame .cap {{ font-size: 10px; color: var(--muted); padding: 4px 6px 0 6px;
    word-break: break-all; line-height: 1.3; }}
  .frame .cap b {{ color: var(--text); font-weight: 600; }}
  /* Property pill row: very small, between caption and score table. */
  .pill-row {{ display: flex; flex-wrap: wrap; gap: 4px; padding: 3px 6px 0 6px; }}
  .pill {{ display: inline-block; font-size: 9px; padding: 1px 6px; border-radius: 99px;
    background: var(--pill-bg); border: 1px solid var(--pill-border); color: var(--text); }}
  .pill.fs-close {{ border-color:#5aa3ff; color:#9ec3ff; }}
  .pill.fs-far   {{ border-color:#8a93a3; color:#b6bcc7; }}
  .pill.fs-unknown {{ border-color:#3a414c; color:#666e7a; font-style: italic; }}
  .pill.q-hi-q   {{ border-color:#3aa775; color:#5fc096; }}
  .pill.q-lo-q   {{ border-color:#d3a554; color:#e9c780; }}
  .pill.q-unknown {{ border-color:#3a414c; color:#666e7a; font-style: italic; }}
  /* Score table — extension point: each ckpt is one .score-row. */
  .scores {{ padding: 3px 6px 6px 6px; display: flex; flex-direction: column;
    gap: 2px; font-size: 10px; }}
  .score-row {{ display: flex; justify-content: space-between; align-items: baseline;
    gap: 6px; line-height: 1.2; }}
  .score-row .ckpt {{ color: var(--muted); font-weight: 600; letter-spacing: .04em;
    text-transform: uppercase; }}
  .score-row .prob {{ font-variant-numeric: tabular-nums; font-weight: 600; }}
  .score-row .prob.real-good, .score-row .prob.fake-good {{ color: var(--score-good); }}
  .score-row .prob.real-bad,  .score-row .prob.fake-bad  {{ color: var(--score-bad);  }}
  .score-row .prob.missing {{ color: var(--muted); font-weight: 400; }}
  /* Highlight panel — sticky right-side controls for live model+threshold review. */
  .frame.thumb {{ border: 3px solid transparent; transition: border-color .12s ease-in; }}
  .frame.thumb.highlight-correct {{ border: 3px solid var(--score-good); }}
  .frame.thumb.highlight-wrong   {{ border: 3px solid var(--score-bad); }}
  .score-row.muted {{ opacity: 0.45; }}
  #highlight-panel {{ position: fixed; right: 16px; top: 80px; width: 240px; z-index: 10;
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 12px 14px 14px 14px;
    box-shadow: 0 6px 22px rgba(0,0,0,0.35);
    font-size: 12px;
  }}
  #highlight-panel h2 {{ margin: 0 0 8px 0; font-size: 12px; text-transform: uppercase;
    letter-spacing: .06em; color: var(--muted); font-weight: 600;
    cursor: grab; user-select: none; padding: 2px 4px; margin-left: -4px;
    margin-right: -4px; border-radius: 4px;
    display: flex; align-items: center; justify-content: space-between; }}
  #highlight-panel h2:hover {{ background: var(--bg); }}
  #highlight-panel h2:active {{ cursor: grabbing; }}
  #highlight-panel h2::before {{ content: "⋮⋮"; color: var(--muted);
    letter-spacing: -2px; font-weight: 400; opacity: 0.6; margin-right: 6px; }}
  #highlight-panel h2 .reset-pos {{ font-size: 10px; color: var(--muted);
    cursor: pointer; padding: 2px 6px; border: 1px solid var(--border);
    border-radius: 4px; text-transform: none; letter-spacing: 0;
    font-weight: 400; }}
  #highlight-panel h2 .reset-pos:hover {{ color: var(--accent);
    border-color: var(--accent); }}
  #highlight-panel .panel-section {{ margin-bottom: 10px; }}
  #highlight-panel .panel-section:last-child {{ margin-bottom: 0; }}
  #highlight-panel label.modelopt {{ display: flex; align-items: center; gap: 6px;
    color: var(--text); padding: 2px 0; cursor: pointer; }}
  #highlight-panel label.modelopt input[type=radio] {{ margin: 0; }}
  #highlight-panel label.modelopt.active {{ color: var(--accent); font-weight: 600; }}
  #highlight-panel .threshold-row {{ display: flex; align-items: center; gap: 8px; }}
  #highlight-panel .threshold-row input[type=range] {{ flex: 1; }}
  #highlight-panel .thresh-val {{ font-variant-numeric: tabular-nums; font-weight: 700;
    color: var(--text); min-width: 38px; text-align: right; }}
  #highlight-panel .counters {{ display: grid; grid-template-columns: 1fr 1fr 1fr;
    gap: 4px; text-align: center; margin-top: 4px; }}
  #highlight-panel .counters .cell {{ background: var(--bg); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 0; }}
  #highlight-panel .counters .cell .lbl {{ display: block; font-size: 9px;
    text-transform: uppercase; letter-spacing: .05em; color: var(--muted); }}
  #highlight-panel .counters .cell .val {{ display: block; font-size: 14px;
    font-weight: 700; font-variant-numeric: tabular-nums; }}
  #highlight-panel .counters .cell.correct .val {{ color: var(--score-good); }}
  #highlight-panel .counters .cell.wrong   .val {{ color: var(--score-bad); }}
  #highlight-panel .counters .cell.fp      .val {{ color: var(--score-bad); }}
  #highlight-panel .counters .cell.fn      .val {{ color: var(--lockbox); }}
  #highlight-panel .counters-fpfn {{ grid-template-columns: 1fr 1fr;
    margin-top: 6px; }}
  #highlight-panel .panel-help {{ font-size: 10px; color: var(--muted);
    margin-top: 6px; line-height: 1.35; }}
  #highlight-panel .cb-toggle {{ display: flex; align-items: center; gap: 6px;
    cursor: pointer; color: var(--muted); font-size: 11px; user-select: none; }}
  #highlight-panel .cb-toggle input {{ margin: 0; cursor: pointer; }}
  #highlight-panel #color-mode-row {{ border-top: 1px solid var(--border);
    padding-top: 8px; margin-top: 4px; }}
  body.color-mode-cb #highlight-panel .cb-toggle {{ color: var(--accent); }}
  @media (max-width: 1100px) {{ #highlight-panel {{ width: 200px; right: 8px; top: 76px; }} }}
  .hidden {{ display: none !important; }}
  .empty {{ color: var(--muted); font-style: italic; padding: 18px;
    text-align: center; }}
  .toolbar-right {{ margin-left: auto; }}
  button.linklike {{ background: transparent; color: var(--accent); border: 0;
    cursor: pointer; padding: 4px 8px; font-size: 12px; }}
  button.linklike:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<header>
  <h1>Identity Frame Browser</h1>
  <div class="meta">
    Generated {generated} &middot;
    <span id="cnt-frames">{n_frames}</span> frames &middot;
    <span id="cnt-identities">{n_identities}</span> identities &middot;
    <span class="badge real">real {n_real}</span>
    <span class="badge fake">fake {n_fake}</span>
    <span class="badge lockbox">lockbox {n_lockbox}</span>
    <span class="badge">dev {n_dev}</span>
    <span class="badge">scores: {ckpt_names_str}</span>
  </div>
  <div class="controls">
    <span class="grouplbl">Filter:</span>
    <input id="search" type="text" placeholder="filter by identity / video_id / suite (substring)…" />
    <span class="grouplbl">Suites:</span>
    {suite_checkboxes}
    <span class="grouplbl">Label:</span>
    <label><input type="checkbox" class="lblcheck" data-label="0" checked /> real</label>
    <label><input type="checkbox" class="lblcheck" data-label="1" checked /> fake</label>
    <span class="grouplbl">Lockbox:</span>
    <label><input type="checkbox" class="lockboxcheck" data-lockbox="True" checked /> lockbox</label>
    <label><input type="checkbox" class="lockboxcheck" data-lockbox="False" checked /> dev</label>
    <span class="grouplbl">Face:</span>
    <label><input type="checkbox" class="fscheck" data-face_size="close" checked /> close</label>
    <label><input type="checkbox" class="fscheck" data-face_size="far" checked /> far</label>
    <label><input type="checkbox" class="fscheck" data-face_size="unknown" checked /> unknown</label>
    <span class="grouplbl">Quality:</span>
    <label><input type="checkbox" class="qcheck" data-quality="hi-q" checked /> hi-q</label>
    <label><input type="checkbox" class="qcheck" data-quality="lo-q" checked /> lo-q</label>
    <label><input type="checkbox" class="qcheck" data-quality="unknown" checked /> unknown</label>
    <span class="grouplbl" title="Production gate. G1 = face detector; G2(200) = min(W,H) >= 200 px. Uncheck a category to drop those frames from counts/errors.">Gate:</span>
    <label title="Frame would be scored by the production pipeline."><input type="checkbox" class="gatecheck" data-gate_status="pass" checked /> pass</label>
    <label title="G2 fails: face crop min(W,H) &lt; 200. Production drops these."><input type="checkbox" class="gatecheck" data-gate_status="drop_lowres" checked /> drop:lowres</label>
    <label title="G1 fails: no face detected. Production drops these."><input type="checkbox" class="gatecheck" data-gate_status="drop_no_face" checked /> drop:no-face</label>
    <label title="Both G1 and G2 fail."><input type="checkbox" class="gatecheck" data-gate_status="drop_both" checked /> drop:both</label>
    <label title="Missing parquet coverage (e.g., Roy_D). Production gate behavior unknown."><input type="checkbox" class="gatecheck" data-gate_status="unknown" checked /> unknown</label>
    <span class="grouplbl">Identities:</span>
    <button class="linklike" id="btn-ids-all">Select all</button>
    <button class="linklike" id="btn-ids-none">Deselect all</button>
    <span class="ids-summary" id="ids-summary"></span>
    <span class="toolbar-right">
      <button class="linklike" id="btn-expand">Expand all</button>
      <button class="linklike" id="btn-collapse">Collapse all</button>
    </span>
  </div>
  <div class="legend">
    Click a thumbnail to open the full-size frame. &middot;
    <span class="badge chronic">chronic-6</span> are the FP-prone identities from the 2026-05-04 audit. &middot;
    Score color: <span class="legend-good" style="color:var(--score-good)">green</span> = model agrees with label,
    <span class="legend-bad" style="color:var(--score-bad)">red</span> = model disagrees (flag for review).
  </div>
</header>
<aside id="highlight-panel">
  <h2 id="hl-drag-handle"><span>Highlight</span><span class="reset-pos" id="hl-reset-pos" title="Reset panel position">reset</span></h2>
  <div class="panel-section" id="model-radios">
    {model_radios}
  </div>
  <div class="panel-section">
    <div class="threshold-row">
      <label for="thresh-slider" style="color:var(--muted);">τ</label>
      <input id="thresh-slider" type="range" min="0" max="1" step="0.01" value="0.50" />
      <span class="thresh-val" id="thresh-val">0.50</span>
    </div>
  </div>
  <div class="panel-section">
    <div class="counters">
      <div class="cell correct"><span class="lbl">correct</span><span class="val" id="cnt-correct">0</span></div>
      <div class="cell wrong"><span class="lbl">wrong</span><span class="val" id="cnt-wrong">0</span></div>
      <div class="cell"><span class="lbl">total</span><span class="val" id="cnt-total">0</span></div>
    </div>
    <div class="counters counters-fpfn">
      <div class="cell fp" title="False positive: real frame scored above τ"><span class="lbl">FP (real flagged)</span><span class="val" id="cnt-fp">0</span></div>
      <div class="cell fn" title="False negative: fake frame scored below τ"><span class="lbl">FN (fake missed)</span><span class="val" id="cnt-fn">0</span></div>
    </div>
    <div class="panel-help">Counters reflect VISIBLE frames with a score for the highlighted ckpt.</div>
  </div>
  <div class="panel-section" id="color-mode-row">
    <label class="cb-toggle" for="cb-toggle">
      <input id="cb-toggle" type="checkbox" />
      <span>Colorblind mode (blue/orange)</span>
    </label>
  </div>
</aside>
<main>
{analytics_html}
<section id="threshold-charts">
  <details open>
    <summary>Threshold trajectories — FPR &amp; recall as τ varies (per ckpt; respects current filter+identity selection)</summary>
    <div id="tcharts-toolbar">
      <span class="tcharts-legend-item"><span class="dot fpr"></span>FPR (real → fake) — lower is better</span>
      <span class="tcharts-legend-item"><span class="dot recall"></span>Recall (fake caught) — higher is better</span>
      <span class="tcharts-tau-readout"></span>
      <button id="tcharts-refresh" type="button" title="Recompute curves on the current visible set">recompute</button>
    </div>
    <div id="tcharts-grid"></div>
    <div id="tcharts-help">Vertical line tracks the τ slider in the Highlight panel. Drag the slider to compare ckpts at any operating point. Click anywhere on a chart to set τ to that x-coordinate.</div>
  </details>
</section>
<section id="errors-section" class="hidden">
  <details class="errors-group" id="errors-fp" open>
    <summary>Wrong: REAL flagged as fake &mdash; <span class="count">0</span> frames</summary>
    <div class="errors-grid"></div>
  </details>
  <details class="errors-group" id="errors-fn" open>
    <summary>Wrong: FAKE missed &mdash; <span class="count">0</span> frames</summary>
    <div class="errors-grid"></div>
  </details>
</section>
"""

PAGE_TAIL = """
</main>
<script>
(function() {
  const search = document.getElementById('search');
  const suiteBoxes = document.querySelectorAll('.suitecheck');
  const labelBoxes = document.querySelectorAll('.lblcheck');
  const lockboxBoxes = document.querySelectorAll('.lockboxcheck');
  const fsBoxes = document.querySelectorAll('.fscheck');
  const qBoxes = document.querySelectorAll('.qcheck');
  const gateBoxes = document.querySelectorAll('.gatecheck');
  const expandBtn = document.getElementById('btn-expand');
  const collapseBtn = document.getElementById('btn-collapse');
  const allDetails = document.querySelectorAll('details.identity');
  const idsAllBtn = document.getElementById('btn-ids-all');
  const idsNoneBtn = document.getElementById('btn-ids-none');
  const idsSummary = document.getElementById('ids-summary');
  const idToggleBoxes = document.querySelectorAll('.ident-toggle');

  // Identity-selection state — set of base_identity strings the user has
  // de-selected.  Persisted so refresh keeps the user's working set.
  //
  // IMPORTANT: prune stale entries on load. If localStorage has identities
  // from a prior build that no longer exist on this page (e.g., scope
  // manifest changed), the count math "(${total - hidden}/${total})"
  // produces negative values and EVERY current identity may end up in
  // the hidden set silently — making the highlight panel show TOTAL=0.
  // Validate against currently-rendered identity-toggle checkboxes.
  const ID_STORAGE_KEY = 'identity_browser_hidden_ids_v1';
  const hiddenIdentities = new Set();
  const knownIdentities = new Set();
  document.querySelectorAll('.ident-toggle').forEach(cb => {
    if (cb.dataset.identity) knownIdentities.add(cb.dataset.identity);
  });
  try {
    const saved = JSON.parse(localStorage.getItem(ID_STORAGE_KEY) || '[]');
    let prunedCount = 0;
    saved.forEach(id => {
      if (knownIdentities.has(id)) {
        hiddenIdentities.add(id);
      } else {
        prunedCount += 1;
      }
    });
    if (prunedCount > 0) {
      // Persist the cleaned set so we don't replay this on every load.
      try {
        localStorage.setItem(ID_STORAGE_KEY,
          JSON.stringify([...hiddenIdentities]));
      } catch (e) {}
      console.info('identity-browser: pruned', prunedCount,
        'stale identity entries from localStorage');
    }
  } catch (e) {}

  function persistHiddenIdentities() {
    try {
      localStorage.setItem(ID_STORAGE_KEY,
        JSON.stringify([...hiddenIdentities]));
    } catch (e) {}
  }

  function syncIdentityCheckboxes() {
    idToggleBoxes.forEach(cb => {
      cb.checked = !hiddenIdentities.has(cb.dataset.identity);
    });
    if (idsSummary) {
      const total = idToggleBoxes.length;
      const hidden = hiddenIdentities.size;
      if (hidden === 0) idsSummary.textContent = `(all ${total} included)`;
      else idsSummary.textContent = `(${total - hidden}/${total} included)`;
    }
  }

  // Analytics tables — recomputed in JS on every filter/identity change.
  const analyticsCells = document.querySelectorAll(
    '#analytics-section table.ana td[data-cell-type]'
  );
  function fmtPct(x) {
    if (x === null || x === undefined || Number.isNaN(x)) return '—';
    return (100 * x).toFixed(1) + '%';
  }
  function bar(x, maxX, color) {
    if (x === null || x === undefined || Number.isNaN(x)) return '';
    const pct = Math.max(0, Math.min(1, x / Math.max(maxX, 1e-9))) * 100;
    return '<span class="ana-bar" style="--w:' + pct.toFixed(1) + '%; ' +
      'background-image: linear-gradient(to right, ' + color +
      ' 0 var(--w), transparent var(--w) 100%);"></span>';
  }
  function passesScenario(fr, scenario) {
    if (!scenario || scenario === 'iq_all_ignore') return true;
    // All quality-discard scenarios require an IQ-tagged frame to begin with.
    const q = fr.dataset.quality;
    const fs = fr.dataset.face_size;
    if (q === 'unknown' || fs === 'unknown') return false;
    if (scenario === 'iq_all') return true;
    if (scenario === 'iq_drop_loq') return q !== 'lo-q';
    if (scenario === 'iq_drop_far') return fs !== 'far';
    if (scenario === 'iq_drop_both') return q !== 'lo-q' && fs !== 'far';
    return true;
  }
  function recomputeAnalytics() {
    if (!analyticsCells.length) return;
    // Collect frames that contribute to analytics — visible AND not excluded
    // by an identity-deselect.
    const vis = [];
    allFrames.forEach(fr => {
      if (fr.classList.contains('hidden')) return;
      if (fr.classList.contains('id-excluded')) return;
      vis.push(fr);
    });
    analyticsCells.forEach(cell => {
      const cellType = cell.dataset.cellType;
      const ckpt = cell.dataset.ckpt;
      const tau = parseFloat(cell.dataset.tau);
      const suiteFilter = cell.dataset.suiteFilter || null;
      const scenario = cell.dataset.scenario || null;
      const attr = ckpt ? ('data-score-' + ckpt.toLowerCase()) : null;

      let frames = vis;
      if (scenario) {
        frames = frames.filter(fr => passesScenario(fr, scenario));
      }
      if (suiteFilter) {
        frames = frames.filter(fr => fr.dataset.suite === suiteFilter);
      }
      // Apply label filter for fpr / recall variants.
      if (cellType === 'fpr' || cellType === 'fpr-per-suite') {
        frames = frames.filter(fr => fr.dataset.label === '0');
      } else if (cellType === 'recall' || cellType === 'recall-per-suite') {
        frames = frames.filter(fr => fr.dataset.label === '1');
      }

      if (cellType === 'kept') {
        const n = frames.length;
        const nR = frames.filter(fr => fr.dataset.label === '0').length;
        const nF = frames.filter(fr => fr.dataset.label === '1').length;
        cell.innerHTML = n + ' <span class="muted">(R=' + nR +
          ', F=' + nF + ')</span>';
        return;
      }

      // Compute fraction of frames at or above tau (with valid score).
      let total = 0, flagged = 0;
      frames.forEach(fr => {
        const raw = attr ? fr.getAttribute(attr) : null;
        if (raw === null || raw === '' || raw === 'NaN') return;
        const s = parseFloat(raw);
        if (Number.isNaN(s)) return;
        total += 1;
        if (s >= tau) flagged += 1;
      });
      let value = total === 0 ? null : flagged / total;
      const isFpr = (cellType === 'fpr' || cellType === 'fpr-per-suite');
      const maxX = isFpr ? 0.5 : 1.0;
      const color = isFpr ? 'var(--score-bad)' : 'var(--score-good)';
      const pctTxt = total === 0 ? '—' : fmtPct(value);
      cell.innerHTML = pctTxt + ' ' + (total === 0 ? '' : bar(value, maxX, color));
    });
  }

  // Highlight panel wiring.
  const modelRadios = document.querySelectorAll('input[name="hl-model"]');
  const threshSlider = document.getElementById('thresh-slider');
  const threshVal = document.getElementById('thresh-val');
  const cntCorrect = document.getElementById('cnt-correct');
  const cntWrong = document.getElementById('cnt-wrong');
  const cntTotal = document.getElementById('cnt-total');
  const cntFp = document.getElementById('cnt-fp');
  const cntFn = document.getElementById('cnt-fn');
  // Originals only — explicitly exclude clones living inside the errors section
  // (those carry data-error-clone="true"). We use a CSS attribute selector so
  // the negation is robust even if the errors-section node is moved.
  const allFrames = document.querySelectorAll('.frame:not([data-error-clone])');

  // Errors-section refs.
  const errorsSection = document.getElementById('errors-section');
  const errorsFp = document.getElementById('errors-fp');
  const errorsFn = document.getElementById('errors-fn');
  const errorsFpGrid = errorsFp ? errorsFp.querySelector('.errors-grid') : null;
  const errorsFnGrid = errorsFn ? errorsFn.querySelector('.errors-grid') : null;
  const errorsFpCount = errorsFp ? errorsFp.querySelector('.count') : null;
  const errorsFnCount = errorsFn ? errorsFn.querySelector('.count') : null;

  // State: highlightedCkpt (null = "Off"); threshold (float).
  const state = { highlightedCkpt: null, threshold: 0.50 };

  function activeSet(boxes, attr) {
    const out = new Set();
    boxes.forEach(b => { if (b.checked) out.add(b.dataset[attr]); });
    return out;
  }

  function applyFilter() {
    const q = (search.value || '').trim().toLowerCase();
    const suites = activeSet(suiteBoxes, 'suite');
    const labels = activeSet(labelBoxes, 'label');
    const lockboxes = activeSet(lockboxBoxes, 'lockbox');
    const fsizes = activeSet(fsBoxes, 'face_size');
    const quals = activeSet(qBoxes, 'quality');
    const gates = activeSet(gateBoxes, 'gate_status');

    allDetails.forEach(det => {
      const ident = det.dataset.identity.toLowerCase();
      const idDeselected = hiddenIdentities.has(det.dataset.identity);
      let identMatched = false;
      const frames = det.querySelectorAll('.frame');
      let visibleInIdentity = 0;
      frames.forEach(fr => {
        // Identity-deselect just marks frames as excluded from calculation;
        // it does NOT hide them. Counters, analytics, and the errors-section
        // skip frames with .id-excluded.
        fr.classList.toggle('id-excluded', idDeselected);
        const suite = fr.dataset.suite;
        const label = fr.dataset.label;
        const lockbox = fr.dataset.lockbox;
        const fsize = fr.dataset.face_size;
        const qual = fr.dataset.quality;
        const gateSt = fr.dataset.gate_status || 'unknown';
        const videoId = (fr.dataset.video || '').toLowerCase();
        let show = suites.has(suite)
          && labels.has(label)
          && lockboxes.has(lockbox)
          && fsizes.has(fsize)
          && quals.has(qual)
          && gates.has(gateSt);
        if (show && q) {
          show = ident.includes(q) || videoId.includes(q) || suite.toLowerCase().includes(q);
        }
        fr.classList.toggle('hidden', !show);
        if (show) { visibleInIdentity++; identMatched = true; }
      });
      // Hide subbuckets/labels with zero visible frames.
      det.querySelectorAll('.subbucket').forEach(sb => {
        const any = sb.querySelector('.frame:not(.hidden)');
        sb.classList.toggle('hidden', !any);
      });
      det.querySelectorAll('.label-block').forEach(lb => {
        const any = lb.querySelector('.frame:not(.hidden)');
        lb.classList.toggle('hidden', !any);
      });
      det.classList.toggle('hidden', !identMatched);
      det.classList.toggle('id-deselected', idDeselected);
      const cnt = det.querySelector('.visible-count');
      if (cnt) cnt.textContent = visibleInIdentity;
    });
    // Re-paint highlight after filtering so counters reflect visible frames.
    applyHighlight();
    // Recompute the analytics tables on the new visible subset.
    recomputeAnalytics();
    // Recompute threshold-trajectory chart curves on the new visible subset.
    // Guard for the initial-call ordering: tchartsRecompute is defined later
    // in this script's lexical scope but the function is hoisted; allFrames
    // and tchartsCkpts are also hoisted-as-const so this is safe to call.
    if (typeof tchartsRecompute === 'function') tchartsRecompute();
  }

  function applyHighlight() {
    // 1) Mute non-highlighted score rows.
    if (state.highlightedCkpt === null) {
      // Off: clear all border + muting.
      allFrames.forEach(fr => {
        fr.classList.remove('highlight-correct', 'highlight-wrong');
        fr.querySelectorAll('.score-row').forEach(sr => sr.classList.remove('muted'));
      });
      // Counters: zero out.
      cntCorrect.textContent = '0';
      cntWrong.textContent = '0';
      cntTotal.textContent = '0';
      cntFp.textContent = '0';
      cntFn.textContent = '0';
      // Sync radio active class.
      modelRadios.forEach(r => {
        const lbl = r.closest('label');
        if (lbl) lbl.classList.toggle('active', r.checked);
      });
      return;
    }
    const ckpt = state.highlightedCkpt;
    const tau = state.threshold;
    let nCorrect = 0, nWrong = 0, nTotal = 0, nFp = 0, nFn = 0;
    allFrames.forEach(fr => {
      // Per-frame border logic.
      // HTML attribute names are case-insensitive and lower-cased by the
      // parser; emit ckpt names in lower-case in the attribute name so
      // getAttribute() lookup matches deterministically.
      const attrName = 'data-score-' + ckpt.toLowerCase();
      const rawScore = fr.getAttribute(attrName);
      fr.classList.remove('highlight-correct', 'highlight-wrong');
      if (rawScore === null || rawScore === undefined || rawScore === '' || rawScore === 'NaN') {
        // Defensive: missing score => no border.
      } else {
        const s = parseFloat(rawScore);
        if (!Number.isNaN(s)) {
          const label = parseInt(fr.dataset.label, 10);
          // Real (label=0): correct iff s < tau. Fake (label=1): correct iff s >= tau.
          const isCorrect = (label === 0) ? (s < tau) : (s >= tau);
          fr.classList.add(isCorrect ? 'highlight-correct' : 'highlight-wrong');
          // Count only frames that are visible AND not excluded by an
          // identity-deselect.  Excluded frames keep their colored border
          // (so the user can still SEE the model's verdict) but don't
          // contribute to the panel's counters.
          if (!fr.classList.contains('hidden') &&
              !fr.classList.contains('id-excluded')) {
            nTotal += 1;
            if (isCorrect) {
              nCorrect += 1;
            } else {
              nWrong += 1;
              // FP = real (label=0) flagged as fake; FN = fake missed.
              if (label === 0) nFp += 1; else nFn += 1;
            }
          }
        }
      }
      // Per-frame score-row muting.
      fr.querySelectorAll('.score-row').forEach(sr => {
        const isHighlighted = sr.dataset.ckpt === ckpt;
        sr.classList.toggle('muted', !isHighlighted);
      });
    });
    cntCorrect.textContent = nCorrect.toString();
    cntWrong.textContent = nWrong.toString();
    cntTotal.textContent = nTotal.toString();
    cntFp.textContent = nFp.toString();
    cntFn.textContent = nFn.toString();
    modelRadios.forEach(r => {
      const lbl = r.closest('label');
      if (lbl) lbl.classList.toggle('active', r.checked);
    });
  }

  // ----- Errors-section: dynamic FP/FN review for the highlighted ckpt -----
  // Produces clones of the wrong-at-threshold .frame elements, partitioned
  // into FP (real-flagged-as-fake) and FN (fake-missed). Clones get
  // data-error-clone="true" so the highlight repaint loop ignores them.
  function clearErrorsSection() {
    if (!errorsFpGrid || !errorsFnGrid) return;
    errorsFpGrid.replaceChildren();
    errorsFnGrid.replaceChildren();
    if (errorsFpCount) errorsFpCount.textContent = '0';
    if (errorsFnCount) errorsFnCount.textContent = '0';
  }

  function repopulateErrorsSection() {
    if (!errorsSection) return;
    if (state.highlightedCkpt === null) {
      errorsSection.classList.add('hidden');
      clearErrorsSection();
      return;
    }
    errorsSection.classList.remove('hidden');
    const ckpt = state.highlightedCkpt;
    const tau = state.threshold;
    const attrName = 'data-score-' + ckpt.toLowerCase();

    // Walk originals only; respect the visibility set produced by applyFilter
    // (frames that applyFilter hid have the .hidden class on themselves).
    function getFrameSource(node) {
      const det = node.closest('details.identity');
      return det ? (det.dataset.identity || '') : '';
    }
    // Per-identity totals on the CURRENTLY VISIBLE + SCORED set. We use the
    // base identity (details.identity dataset.identity) as the grouping key,
    // matching the source the FP/FN items already carry. Two counters per
    // identity: total_real (denominator for FP fraction) and total_fake
    // (denominator for FN fraction). Computed once per repopulate.
    // Frames with no score for the current ckpt are excluded from the
    // denominators (mirroring the items[] population logic below).
    const realTotals = new Map();  // source -> count of visible real frames w/ score
    const fakeTotals = new Map();
    const fpItems = [];  // {score, node, source}
    const fnItems = [];
    allFrames.forEach(fr => {
      if (fr.classList.contains('hidden')) return;
      if (fr.classList.contains('id-excluded')) return;
      const rawScore = fr.getAttribute(attrName);
      if (rawScore === null || rawScore === undefined ||
          rawScore === '' || rawScore === 'NaN') return;
      const s = parseFloat(rawScore);
      if (Number.isNaN(s)) return;
      const label = parseInt(fr.dataset.label, 10);
      const isReal = (label === 0);
      const source = getFrameSource(fr);
      const totMap = isReal ? realTotals : fakeTotals;
      totMap.set(source, (totMap.get(source) || 0) + 1);
      const scoreSaysFake = (s >= tau);
      const wrong = (isReal && scoreSaysFake) || (!isReal && !scoreSaysFake);
      if (!wrong) return;
      if (isReal) {
        fpItems.push({ score: s, node: fr, source: source });
      } else {
        fnItems.push({ score: s, node: fr, source: source });
      }
    });

    // Sort: primary key = source (groups frames from same identity together);
    // secondary = score (FP desc = worst-FP first within source; FN asc = same).
    fpItems.sort((a, b) => {
      if (a.source !== b.source) return a.source < b.source ? -1 : 1;
      return b.score - a.score;
    });
    fnItems.sort((a, b) => {
      if (a.source !== b.source) return a.source < b.source ? -1 : 1;
      return a.score - b.score;
    });

    // Build clones; set markers; never carry the original id.
    function buildClone(originalNode, errorClass) {
      const clone = originalNode.cloneNode(true);
      // cloneNode copies the id; strip it to avoid duplicate-id errors.
      if (clone.id) clone.removeAttribute('id');
      // Strip ids on any inner elements too (defensive — none exist today).
      clone.querySelectorAll('[id]').forEach(el => el.removeAttribute('id'));
      // Clones must never be filter-hidden by applyFilter (they live outside
      // identity .details and applyFilter only walks `det.querySelectorAll('.frame')`,
      // but we also ensure the .hidden class — if any — is cleared.
      clone.classList.remove('hidden');
      clone.setAttribute('data-error-clone', 'true');
      clone.setAttribute('data-error-class', errorClass);
      return clone;
    }

    // Populate grids with per-identity headers showing wrong/total fractions.
    // Items are already sorted by source first (groups identity frames
    // together), then by score within source.
    function populateGridWithHeaders(grid, items, errorClass, totalsMap) {
      if (!grid) return;
      if (items.length === 0) {
        const placeholder = document.createElement('div');
        placeholder.className = 'errors-empty';
        placeholder.textContent = '(no errors at this threshold + filter)';
        grid.replaceChildren(placeholder);
        return;
      }
      // Count wrong-per-source.
      const wrongPerSource = new Map();
      items.forEach(it => {
        wrongPerSource.set(it.source, (wrongPerSource.get(it.source) || 0) + 1);
      });
      // Build grid as a sequence of (header, frames-for-this-identity).
      const frag = document.createDocumentFragment();
      let currentSource = null;
      items.forEach(it => {
        if (it.source !== currentSource) {
          currentSource = it.source;
          const wrong = wrongPerSource.get(currentSource) || 0;
          const total = totalsMap.get(currentSource) || 0;
          const pct = (total > 0) ? (100 * wrong / total) : 0;
          const header = document.createElement('div');
          header.className = 'errors-identity-header';
          header.setAttribute('data-source', currentSource);
          const identSpan = document.createElement('span');
          identSpan.className = 'ident';
          identSpan.textContent = currentSource || '(no identity)';
          const fractionSpan = document.createElement('span');
          fractionSpan.className = 'fraction';
          fractionSpan.textContent = wrong + ' / ' + total + ' wrong';
          const pctSpan = document.createElement('span');
          pctSpan.className = 'pct';
          pctSpan.textContent = '(' + pct.toFixed(1) + '%)';
          header.appendChild(identSpan);
          header.appendChild(fractionSpan);
          header.appendChild(pctSpan);
          frag.appendChild(header);
        }
        frag.appendChild(buildClone(it.node, errorClass));
      });
      grid.replaceChildren(frag);
    }
    populateGridWithHeaders(errorsFpGrid, fpItems, 'fp', realTotals);
    populateGridWithHeaders(errorsFnGrid, fnItems, 'fn', fakeTotals);

    if (errorsFpCount) errorsFpCount.textContent = fpItems.length.toString();
    if (errorsFnCount) errorsFnCount.textContent = fnItems.length.toString();
  }

  // Light debounce so search-typing doesn't rebuild on every keystroke.
  let errorsRepopTimer = null;
  function repopulateErrorsSectionDebounced(delayMs) {
    if (errorsRepopTimer) clearTimeout(errorsRepopTimer);
    errorsRepopTimer = setTimeout(() => {
      errorsRepopTimer = null;
      repopulateErrorsSection();
    }, delayMs);
  }

  // ----- Threshold trajectory charts -----
  // Per-ckpt small multiples showing FPR(τ) + Recall(τ) curves on the
  // currently visible+included frames. Vertical line tracks the highlight
  // panel's τ slider. Click on a chart panel sets τ at that x-coordinate.
  const tchartsGrid = document.getElementById('tcharts-grid');
  const tchartsTauReadout = document.querySelector('#threshold-charts .tcharts-tau-readout');
  const tchartsRefreshBtn = document.getElementById('tcharts-refresh');
  // Read ckpt names from the highlight radios (skip __off__).
  const tchartsCkpts = Array.from(modelRadios)
    .map(r => r.value)
    .filter(v => v && v !== '__off__');
  // SVG layout — wider + taller, with bigger axis fonts. The viewBox is in
  // user-units; the SVG element scales to fill its grid cell width while
  // the height stays at the CSS height set in .tchart-svg.
  const TCHART_W = 460, TCHART_H = 200;
  const TCHART_PAD = { l: 38, r: 14, t: 10, b: 26 };
  const TCHART_PLOT_W = TCHART_W - TCHART_PAD.l - TCHART_PAD.r;
  const TCHART_PLOT_H = TCHART_H - TCHART_PAD.t - TCHART_PAD.b;
  // Per-ckpt sorted score arrays (real, fake) — recomputed on filter change.
  const tchartsData = {};

  function tchartsCollect() {
    tchartsCkpts.forEach(c => { tchartsData[c] = { real: [], fake: [] }; });
    allFrames.forEach(fr => {
      if (fr.classList.contains('hidden') || fr.classList.contains('id-excluded')) return;
      const label = fr.dataset.label;
      tchartsCkpts.forEach(c => {
        const raw = fr.getAttribute('data-score-' + c.toLowerCase());
        if (raw === null || raw === '' || raw === 'NaN') return;
        const s = parseFloat(raw);
        if (Number.isNaN(s)) return;
        if (label === '0') tchartsData[c].real.push(s);
        else if (label === '1') tchartsData[c].fake.push(s);
      });
    });
    Object.values(tchartsData).forEach(d => {
      d.real.sort((a, b) => a - b);
      d.fake.sort((a, b) => a - b);
    });
  }

  // Binary search: count of arr[i] >= tau.
  function tchartsCountAbove(arr, tau) {
    let lo = 0, hi = arr.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (arr[mid] >= tau) hi = mid; else lo = mid + 1;
    }
    return arr.length - lo;
  }

  function tchartsRateAt(arr, tau) {
    if (arr.length === 0) return null;
    return tchartsCountAbove(arr, tau) / arr.length;
  }

  // Render or update one ckpt panel. Returns the panel element.
  function tchartsRenderPanel(ckpt) {
    const data = tchartsData[ckpt] || { real: [], fake: [] };
    const nReal = data.real.length;
    const nFake = data.fake.length;
    const panel = document.createElement('div');
    panel.className = 'tchart-panel';
    panel.dataset.ckpt = ckpt;

    // Header: title (left) + readout (right), both in regular HTML so they
    // don't overlap the SVG plot area.
    const header = document.createElement('div');
    header.className = 'tchart-header';
    header.innerHTML =
      '<div class="tchart-title">' + ckpt +
      '<span class="n">n_real=' + nReal + ' &middot; n_fake=' + nFake + '</span>' +
      '</div>' +
      '<div class="tchart-readout">' +
      '<span class="v-fpr">FPR —</span>' +
      '<span class="v-recall">Recall —</span>' +
      '<span class="v-gap">gap —</span>' +
      '</div>';
    panel.appendChild(header);

    if (nReal === 0 && nFake === 0) {
      const empty = document.createElement('div');
      empty.className = 'tchart-empty';
      empty.textContent = '(no scored frames in current selection)';
      panel.appendChild(empty);
      return panel;
    }

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'tchart-svg');
    svg.setAttribute('viewBox', '0 0 ' + TCHART_W + ' ' + TCHART_H);
    svg.setAttribute('preserveAspectRatio', 'none');

    // Y-axis grid lines + labels at 0, 25, 50, 75, 100%.
    [0, 0.25, 0.5, 0.75, 1.0].forEach(yv => {
      const y = TCHART_PAD.t + (1 - yv) * TCHART_PLOT_H;
      const ln = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      ln.setAttribute('x1', TCHART_PAD.l);
      ln.setAttribute('x2', TCHART_PAD.l + TCHART_PLOT_W);
      ln.setAttribute('y1', y); ln.setAttribute('y2', y);
      ln.setAttribute('class', 'grid-line');
      svg.appendChild(ln);
      const tx = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      tx.setAttribute('x', TCHART_PAD.l - 5); tx.setAttribute('y', y + 4);
      tx.setAttribute('text-anchor', 'end');
      tx.setAttribute('class', 'axis-label');
      tx.textContent = (yv * 100).toFixed(0) + '%';
      svg.appendChild(tx);
    });
    // X-axis grid lines + labels at 0, 0.25, 0.5, 0.75, 1.0
    [0, 0.25, 0.5, 0.75, 1.0].forEach(xv => {
      const x = TCHART_PAD.l + xv * TCHART_PLOT_W;
      // Vertical light grid line
      const ln = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      ln.setAttribute('x1', x); ln.setAttribute('x2', x);
      ln.setAttribute('y1', TCHART_PAD.t);
      ln.setAttribute('y2', TCHART_PAD.t + TCHART_PLOT_H);
      ln.setAttribute('class', 'grid-line');
      svg.appendChild(ln);
      const tx = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      tx.setAttribute('x', x); tx.setAttribute('y', TCHART_H - 8);
      tx.setAttribute('text-anchor', xv === 0 ? 'start' : (xv === 1 ? 'end' : 'middle'));
      tx.setAttribute('class', 'axis-label');
      tx.textContent = xv.toFixed(2);
      svg.appendChild(tx);
    });
    // Axis title
    const xTitle = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    xTitle.setAttribute('x', TCHART_PAD.l + TCHART_PLOT_W / 2);
    xTitle.setAttribute('y', TCHART_H - 1);
    xTitle.setAttribute('text-anchor', 'middle');
    xTitle.setAttribute('class', 'axis-label');
    xTitle.style.fontStyle = 'italic';
    xTitle.textContent = 'τ';
    svg.appendChild(xTitle);

    // Build the FPR + Recall curves using sample points along τ.
    const N = 201;
    let fprPath = '', recallPath = '';
    for (let i = 0; i < N; i++) {
      const tau = i / (N - 1);
      const x = TCHART_PAD.l + tau * TCHART_PLOT_W;
      const fpr = tchartsRateAt(data.real, tau);
      const rec = tchartsRateAt(data.fake, tau);
      if (fpr !== null) {
        const y = TCHART_PAD.t + (1 - fpr) * TCHART_PLOT_H;
        fprPath += (i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1) + ' ';
      }
      if (rec !== null) {
        const y = TCHART_PAD.t + (1 - rec) * TCHART_PLOT_H;
        recallPath += (i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1) + ' ';
      }
    }
    if (fprPath) {
      const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      p.setAttribute('class', 'curve-fpr'); p.setAttribute('d', fprPath);
      svg.appendChild(p);
    }
    if (recallPath) {
      const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      p.setAttribute('class', 'curve-recall'); p.setAttribute('d', recallPath);
      svg.appendChild(p);
    }

    // τ vertical line + dots — updated via tchartsUpdateTau().
    const tauLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    tauLine.setAttribute('class', 'tau-line');
    tauLine.setAttribute('y1', TCHART_PAD.t);
    tauLine.setAttribute('y2', TCHART_PAD.t + TCHART_PLOT_H);
    svg.appendChild(tauLine);
    const fprDot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    fprDot.setAttribute('class', 'tau-dot'); fprDot.setAttribute('r', '4');
    svg.appendChild(fprDot);
    const recDot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    recDot.setAttribute('class', 'tau-dot'); recDot.setAttribute('r', '4');
    svg.appendChild(recDot);

    // Hover line — visible on mouseover, shows the τ at the cursor without
    // committing the slider until click.
    const hoverLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    hoverLine.setAttribute('class', 'hover-line');
    hoverLine.setAttribute('y1', TCHART_PAD.t);
    hoverLine.setAttribute('y2', TCHART_PAD.t + TCHART_PLOT_H);
    svg.appendChild(hoverLine);

    // HTML readout refs (in the panel header).
    const readoutFpr = panel.querySelector('.v-fpr');
    const readoutRecall = panel.querySelector('.v-recall');
    const readoutGap = panel.querySelector('.v-gap');

    function tauFromEvent(ev) {
      const rect = svg.getBoundingClientRect();
      const px = ev.clientX - rect.left;
      const xRatio = (px / rect.width) * TCHART_W;
      return Math.max(0, Math.min(1, (xRatio - TCHART_PAD.l) / TCHART_PLOT_W));
    }
    function paintReadout(tau) {
      const fpr = tchartsRateAt(data.real, tau);
      const rec = tchartsRateAt(data.fake, tau);
      readoutFpr.textContent = 'FPR ' + (fpr === null ? '—' : (100 * fpr).toFixed(1) + '%');
      readoutRecall.textContent = 'Recall ' + (rec === null ? '—' : (100 * rec).toFixed(1) + '%');
      const gap = (fpr === null || rec === null) ? null : rec - fpr;
      readoutGap.textContent = 'gap ' + (gap === null ? '—' : (100 * gap).toFixed(1) + 'pp');
    }
    panel._paintReadout = paintReadout;

    svg.addEventListener('mousemove', (ev) => {
      const tau = tauFromEvent(ev);
      const x = TCHART_PAD.l + tau * TCHART_PLOT_W;
      hoverLine.setAttribute('x1', x);
      hoverLine.setAttribute('x2', x);
      svg.classList.add('hovering');
      paintReadout(tau);
    });
    svg.addEventListener('mouseleave', () => {
      svg.classList.remove('hovering');
      paintReadout(state.threshold);
    });
    svg.addEventListener('click', (ev) => {
      const tau = tauFromEvent(ev);
      threshSlider.value = tau.toFixed(2);
      threshSlider.dispatchEvent(new Event('input', { bubbles: true }));
      threshSlider.dispatchEvent(new Event('change', { bubbles: true }));
    });

    panel._tauLine = tauLine;
    panel._fprDot = fprDot;
    panel._recDot = recDot;
    panel.appendChild(svg);
    return panel;
  }

  function tchartsRender() {
    if (!tchartsGrid) return;
    tchartsGrid.replaceChildren();
    tchartsCkpts.forEach(c => {
      const panel = tchartsRenderPanel(c);
      tchartsGrid.appendChild(panel);
    });
    tchartsUpdateTau(state.threshold);
  }

  function tchartsUpdateTau(tau) {
    if (tchartsTauReadout) {
      tchartsTauReadout.textContent = 'τ = ' + tau.toFixed(2);
    }
    if (!tchartsGrid) return;
    tchartsGrid.querySelectorAll('.tchart-panel').forEach(panel => {
      const ckpt = panel.dataset.ckpt;
      const data = tchartsData[ckpt];
      if (!panel._tauLine || !data) return;
      const x = TCHART_PAD.l + tau * TCHART_PLOT_W;
      panel._tauLine.setAttribute('x1', x);
      panel._tauLine.setAttribute('x2', x);
      const fpr = tchartsRateAt(data.real, tau);
      const rec = tchartsRateAt(data.fake, tau);
      if (fpr !== null) {
        panel._fprDot.setAttribute('cx', x);
        panel._fprDot.setAttribute('cy', TCHART_PAD.t + (1 - fpr) * TCHART_PLOT_H);
        panel._fprDot.setAttribute('display', '');
      } else {
        panel._fprDot.setAttribute('display', 'none');
      }
      if (rec !== null) {
        panel._recDot.setAttribute('cx', x);
        panel._recDot.setAttribute('cy', TCHART_PAD.t + (1 - rec) * TCHART_PLOT_H);
        panel._recDot.setAttribute('display', '');
      } else {
        panel._recDot.setAttribute('display', 'none');
      }
      // Repaint the HTML readout via the per-panel paintReadout closure,
      // which has access to the panel's data + readout span refs.
      if (typeof panel._paintReadout === 'function') {
        panel._paintReadout(tau);
      }
    });
  }

  function tchartsRecompute() {
    tchartsCollect();
    tchartsRender();
  }
  if (tchartsRefreshBtn) {
    tchartsRefreshBtn.addEventListener('click', tchartsRecompute);
  }

  // Wire highlight panel.
  // applyHighlight() is fast (border + counters update). repopulateErrorsSection()
  // is slow (deep-clones up to ~1k DOM nodes per ckpt). Run them in separate
  // animation frames so the counters paint immediately and the heavy clone
  // step doesn't make the click feel stuck. Show a "calculating..." marker on
  // the errors section while the clone work is in flight.
  function setErrorsBusy(busy) {
    if (!errorsSection) return;
    errorsSection.classList.toggle('errors-busy', busy);
    if (errorsFpCount) errorsFpCount.textContent = busy ? '…' : errorsFpCount.textContent;
    if (errorsFnCount) errorsFnCount.textContent = busy ? '…' : errorsFnCount.textContent;
  }
  function deferRepopulate() {
    if (state.highlightedCkpt === null) {
      setErrorsBusy(false);
      repopulateErrorsSection();
      return;
    }
    setErrorsBusy(true);
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        repopulateErrorsSection();
        setErrorsBusy(false);
      });
    });
  }
  modelRadios.forEach(r => {
    r.addEventListener('change', () => {
      const v = r.value;
      state.highlightedCkpt = (v === '__off__') ? null : v;
      applyHighlight();
      deferRepopulate();
    });
  });
  // Slider: live border updates on `input` (drag); errors-section repop
  // only on `change` (slider release) to avoid cloning ~1k nodes per tick.
  // Threshold-trajectory chart's τ line updates live on `input` (cheap).
  threshSlider.addEventListener('input', () => {
    const v = parseFloat(threshSlider.value);
    state.threshold = v;
    threshVal.textContent = v.toFixed(2);
    applyHighlight();
    tchartsUpdateTau(v);
  });
  threshSlider.addEventListener('change', () => {
    deferRepopulate();
  });

  // Filter changes also refresh the errors section (debounced for search).
  search.addEventListener('input', () => {
    applyFilter();
    repopulateErrorsSectionDebounced(150);
  });
  function wireFilterBox(b) {
    b.addEventListener('change', () => {
      applyFilter();
      repopulateErrorsSection();
    });
  }
  suiteBoxes.forEach(wireFilterBox);
  labelBoxes.forEach(wireFilterBox);
  lockboxBoxes.forEach(wireFilterBox);
  fsBoxes.forEach(wireFilterBox);
  qBoxes.forEach(wireFilterBox);
  gateBoxes.forEach(wireFilterBox);
  expandBtn.addEventListener('click', () => allDetails.forEach(d => d.open = true));
  collapseBtn.addEventListener('click', () => allDetails.forEach(d => d.open = false));

  // Per-identity checkboxes — click toggles inclusion in counters/analytics.
  // stopPropagation keeps the click from also toggling the <details> open/close.
  idToggleBoxes.forEach(cb => {
    cb.addEventListener('click', e => e.stopPropagation());
    cb.addEventListener('change', () => {
      const id = cb.dataset.identity;
      if (cb.checked) hiddenIdentities.delete(id);
      else hiddenIdentities.add(id);
      persistHiddenIdentities();
      syncIdentityCheckboxes();
      applyFilter();
      repopulateErrorsSection();
    });
  });
  if (idsAllBtn) {
    idsAllBtn.addEventListener('click', () => {
      hiddenIdentities.clear();
      persistHiddenIdentities();
      syncIdentityCheckboxes();
      applyFilter();
      repopulateErrorsSection();
    });
  }
  if (idsNoneBtn) {
    idsNoneBtn.addEventListener('click', () => {
      idToggleBoxes.forEach(cb => hiddenIdentities.add(cb.dataset.identity));
      persistHiddenIdentities();
      syncIdentityCheckboxes();
      applyFilter();
      repopulateErrorsSection();
    });
  }

  // Initial sync — restore identity-checkbox state from persisted set.
  syncIdentityCheckboxes();
  applyFilter();
  // applyFilter already calls applyHighlight() and recomputeAnalytics().
  repopulateErrorsSection();
  // Initial threshold-trajectory chart build. Recomputes when filter changes
  // OR when identities are toggled (both go through applyFilter which calls
  // recomputeAnalytics; we hook tchartsRecompute alongside that flow via
  // the recompute button + on filter wiring above). Build once now.
  tchartsRecompute();

  // Draggable highlight panel ----------------------------------------
  // Drag from the h2 header. Position is persisted in localStorage.
  // Reset link snaps back to the default top-right position.
  (function setupDrag() {
    const panel = document.getElementById('highlight-panel');
    const handle = document.getElementById('hl-drag-handle');
    const resetBtn = document.getElementById('hl-reset-pos');
    if (!panel || !handle) return;
    const STORAGE_KEY = 'identity_browser_panel_pos_v1';

    function clampToViewport(left, top) {
      const r = panel.getBoundingClientRect();
      const maxLeft = window.innerWidth - r.width - 4;
      const maxTop = window.innerHeight - r.height - 4;
      return [Math.max(4, Math.min(maxLeft, left)),
              Math.max(4, Math.min(maxTop,  top))];
    }
    function applyPos(left, top) {
      const [L, T] = clampToViewport(left, top);
      panel.style.left = L + 'px';
      panel.style.top  = T + 'px';
      panel.style.right = 'auto';
    }
    function resetPos() {
      panel.style.left = '';
      panel.style.top = '';
      panel.style.right = '';
      try { localStorage.removeItem(STORAGE_KEY); } catch (e) {}
    }
    // Restore saved position.
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
      if (saved && typeof saved.left === 'number' && typeof saved.top === 'number') {
        applyPos(saved.left, saved.top);
      }
    } catch (e) {}

    let dragging = false, startX = 0, startY = 0, startL = 0, startT = 0;
    handle.addEventListener('mousedown', (e) => {
      // Don't start drag if user clicked the reset button.
      if (e.target && e.target.id === 'hl-reset-pos') return;
      dragging = true;
      const r = panel.getBoundingClientRect();
      startX = e.clientX; startY = e.clientY;
      startL = r.left; startT = r.top;
      handle.style.cursor = 'grabbing';
      e.preventDefault();
    });
    document.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      const dx = e.clientX - startX, dy = e.clientY - startY;
      applyPos(startL + dx, startT + dy);
    });
    document.addEventListener('mouseup', () => {
      if (!dragging) return;
      dragging = false;
      handle.style.cursor = '';
      const r = panel.getBoundingClientRect();
      try { localStorage.setItem(STORAGE_KEY,
              JSON.stringify({ left: r.left, top: r.top })); } catch (e) {}
    });
    if (resetBtn) {
      resetBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetPos();
      });
    }
    // Re-clamp on window resize so the panel doesn't end up off-screen.
    window.addEventListener('resize', () => {
      if (!panel.style.left) return;
      const r = panel.getBoundingClientRect();
      applyPos(r.left, r.top);
    });
  })();

  // Colorblind mode toggle ------------------------------------------------
  // Swaps green/red for bold blue/orange. Persisted in localStorage.
  // Updates the legend text in the page header to match.
  (function setupColorMode() {
    const cb = document.getElementById('cb-toggle');
    const STORAGE_KEY = 'identity_browser_color_mode_v1';
    const goodEl = document.querySelector('.legend-good');
    const badEl = document.querySelector('.legend-bad');
    function apply(on) {
      if (on) {
        document.body.classList.add('color-mode-cb');
        if (goodEl) goodEl.textContent = 'blue';
        if (badEl)  badEl.textContent = 'orange';
      } else {
        document.body.classList.remove('color-mode-cb');
        if (goodEl) goodEl.textContent = 'green';
        if (badEl)  badEl.textContent = 'red';
      }
    }
    // Restore.
    let saved = false;
    try { saved = localStorage.getItem(STORAGE_KEY) === '1'; } catch (e) {}
    if (cb) { cb.checked = saved; }
    apply(saved);
    if (cb) {
      cb.addEventListener('change', () => {
        apply(cb.checked);
        try { localStorage.setItem(STORAGE_KEY, cb.checked ? '1' : '0'); } catch (e) {}
      });
    }
  })();
})();
</script>
</body></html>
"""


def _score_class(label: int, score: float) -> str:
    """Return a CSS class indicating whether the model's prob agrees with the label.

    Threshold: 0.5.  Real (label=0) frame with prob >= 0.5 -> 'real-bad'.
    """

    if score >= 0.5:
        return "real-bad" if label == 0 else "fake-good"
    return "real-good" if label == 0 else "fake-bad"


def _score_format(score: float | None) -> str:
    if score is None or pd.isna(score):
        return "—"
    # 2 decimal places, no rounding mistakes from floats
    return f"{float(score):.2f}"


def _render_score_block(
    row: pd.Series, ckpt_names: list[str]
) -> str:
    """Render the .scores block for one frame with one .score-row per ckpt.

    Future ckpts: just add to ckpt_names; HTML emission is automatic.
    Each .score-row carries data-ckpt="<NAME>" so JS can target individual
    rows for muting when a model is highlighted.
    """

    parts = ['<div class="scores">']
    label = int(row["label"])
    for name in ckpt_names:
        col = f"score_{name}"
        score = row.get(col)
        if pd.isna(score):
            parts.append(
                f'<div class="score-row" data-ckpt="{html_escape(name)}">'
                f'<span class="ckpt">{html_escape(name)}</span>'
                f'<span class="prob missing">—</span>'
                f'</div>'
            )
        else:
            cls = _score_class(label, float(score))
            parts.append(
                f'<div class="score-row" data-ckpt="{html_escape(name)}">'
                f'<span class="ckpt">{html_escape(name)}</span>'
                f'<span class="prob {cls}">{_score_format(score)}</span>'
                f'</div>'
            )
    parts.append("</div>")
    return "".join(parts)


def _data_score_attr(ckpt_name: str) -> str:
    """Map a ckpt name to its DOM dataset key used in JS.

    HTML attribute is `data-score-<NAME>`. The browser DOMStringMap
    (`dataset`) lower-cases attribute names and converts `-x` to `X`,
    so JS reads `dataset['score' + sanitized]`. We sanitize matching
    the JS regex `[^a-zA-Z0-9_-]` -> '_' to keep them aligned.
    """

    # No spaces in any expected ckpt names; we still sanitize defensively.
    return re.sub(r"[^a-zA-Z0-9_-]", "_", ckpt_name)


def _sort_key(row: pd.Series, primary_ckpt: str | None) -> tuple:
    """Sort key per spec.

    Primary  : face_size (close < far < unknown)
    Secondary: quality (hi-q < lo-q < unknown)
    Tertiary : score from `primary_ckpt`. For real frames descending
               (worst-FP first); for fake frames ascending (worst-FN first).
               Missing scores sort last within their sub-bucket.
    """

    fs = FACE_SIZE_RANK.get(str(row.get("face_size", "unknown")), 2)
    q = QUALITY_RANK.get(str(row.get("quality", "unknown")), 2)
    score = None
    if primary_ckpt is not None:
        s = row.get(f"score_{primary_ckpt}")
        if not pd.isna(s):
            score = float(s)
    label = int(row["label"])
    if score is None:
        # missing → sort last within sub-bucket
        return (fs, q, 1, 0.0)
    if label == 0:  # real → descending → flip sign
        return (fs, q, 0, -score)
    return (fs, q, 0, score)  # fake → ascending


def render_html(
    df: pd.DataFrame,
    output_root: Path,
    ckpt_names: list[str],
    log: logging.Logger,
    analytics_html: str = "",
) -> Path:
    """Write index.html based on df.

    - 6-column grid
    - frames sorted within each (identity, label) section by face_size,
      quality, then primary-ckpt score
    - per-frame property pills + score table (one row per ckpt)
    """

    primary_ckpt = ckpt_names[0] if ckpt_names else None

    n_frames = len(df)
    n_identities = df["base_identity"].nunique()
    n_real = int((df["label"] == 0).sum())
    n_fake = int((df["label"] == 1).sum())
    n_lockbox = int(df["is_lockbox"].astype(str).isin(["True", "1", "true"]).sum())
    n_dev = n_frames - n_lockbox

    suites_sorted = sorted(df["suite"].unique())
    suite_checkboxes_html = "\n    ".join(
        f'<label><input type="checkbox" class="suitecheck" '
        f'data-suite="{html_escape(s)}" checked /> {html_escape(s)}</label>'
        for s in suites_sorted
    )

    # Build the model-radio set for the highlight panel.
    # First option is "Off" (default checked); then one radio per ckpt.
    model_radio_lines: list[str] = []
    model_radio_lines.append(
        '<label class="modelopt active">'
        '<input type="radio" name="hl-model" value="__off__" checked />'
        ' Off</label>'
    )
    for n in ckpt_names:
        model_radio_lines.append(
            f'<label class="modelopt">'
            f'<input type="radio" name="hl-model" value="{html_escape(n)}" />'
            f' {html_escape(n)}</label>'
        )
    model_radios_html = "\n    ".join(model_radio_lines)

    head = PAGE_HEAD.format(
        generated=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        n_frames=n_frames,
        n_identities=n_identities,
        n_real=n_real,
        n_fake=n_fake,
        n_lockbox=n_lockbox,
        n_dev=n_dev,
        suite_checkboxes=suite_checkboxes_html,
        ckpt_names_str=html_escape(", ".join(ckpt_names) if ckpt_names else "(none)"),
        model_radios=model_radios_html,
        analytics_html=analytics_html,
    )

    parts: list[str] = [head]

    # Sort identities by frame count desc.
    ident_order = (
        df["base_identity"].value_counts().sort_values(ascending=False).index.tolist()
    )
    for ident in ident_order:
        sub = df[df["base_identity"] == ident]
        nf = len(sub)
        n_r = int((sub["label"] == 0).sum())
        n_f = int((sub["label"] == 1).sum())
        n_lb = int(sub["is_lockbox"].astype(str).isin(["True", "1", "true"]).sum())
        chronic = ident in CHRONIC_6
        chronic_badge = '<span class="badge chronic">chronic-6</span>' if chronic else ""
        parts.append(
            f'<details class="identity" data-identity="{html_escape(ident)}" open>\n'
            f'  <summary>'
            f'<input type="checkbox" class="ident-toggle" '
            f'data-identity="{html_escape(ident)}" checked '
            f'title="Include this identity in filtered counters and analytics" />'
            f'<span class="ident-name">{html_escape(ident)}</span>'
            f'  {chronic_badge}'
            f'<span class="ident-counts">'
            f'  &middot; {nf} frames '
            f'(<span class="visible-count">{nf}</span> visible) '
            f'&middot; real {n_r} / fake {n_f} '
            f'&middot; lockbox {n_lb}</span>'
            f'</summary>\n'
        )
        for label_val in [0, 1]:  # real first
            label_name = "real" if label_val == 0 else "fake"
            label_sub = sub[sub["label"] == label_val]
            if label_sub.empty:
                continue
            parts.append(f'  <div class="label-block {label_name}">\n')
            parts.append(f'    <h3>{label_name} ({len(label_sub)})</h3>\n')

            # Compute sort keys and order frames.
            label_sub = label_sub.copy()
            label_sub["_sort_key"] = label_sub.apply(
                lambda r: _sort_key(r, primary_ckpt), axis=1
            )
            ordered = label_sub.sort_values(by="_sort_key", kind="stable")

            # Group by (face_size, quality) sub-bucket while preserving order.
            current_bucket: tuple[str, str] | None = None
            grid_open = False
            sb_open = False

            def _close_sb():
                nonlocal grid_open, sb_open
                if grid_open:
                    parts.append('      </div>\n')  # grid
                    grid_open = False
                if sb_open:
                    parts.append('    </div>\n')  # subbucket
                    sb_open = False

            for _, row in ordered.iterrows():
                bucket = (str(row.get("face_size", "unknown")),
                          str(row.get("quality", "unknown")))
                if bucket != current_bucket:
                    _close_sb()
                    current_bucket = bucket
                    fs, q = bucket
                    bucket_count = int(((ordered["face_size"] == fs)
                                        & (ordered["quality"] == q)).sum())
                    parts.append('    <div class="subbucket" '
                                 f'data-face_size="{html_escape(fs)}" '
                                 f'data-quality="{html_escape(q)}">\n')
                    sb_open = True
                    parts.append(
                        f'      <div class="subbucket-header">'
                        f'<b>{html_escape(fs)} + {html_escape(q)}</b> '
                        f'({bucket_count} frame{"s" if bucket_count != 1 else ""})'
                        f'</div>\n'
                    )
                    parts.append('      <div class="grid">\n')
                    grid_open = True

                full, thumb = local_paths(row, output_root)
                rel_full = os.path.relpath(full, output_root)
                rel_thumb = os.path.relpath(thumb, output_root)
                if not thumb.exists():
                    continue
                cap_video = html_escape(str(row["video_id"]))
                cap_suite = html_escape(str(row["suite"]))
                fs, q = bucket
                pill_fs = (
                    f'<span class="pill fs-{html_escape(fs)}">'
                    f'face: {html_escape(fs)}</span>'
                )
                pill_q = (
                    f'<span class="pill q-{html_escape(q)}">'
                    f'q: {html_escape(q)}</span>'
                )
                score_block = _render_score_block(row, ckpt_names)
                # Emit per-ckpt score attributes for JS-side highlight panel.
                # Attribute names are lower-cased to match getAttribute()
                # lookup in the inline JS (HTML parser lower-cases attrs).
                score_attrs_parts: list[str] = []
                for name in ckpt_names:
                    col = f"score_{name}"
                    s = row.get(col)
                    if not pd.isna(s):
                        attr = f"data-score-{name.lower()}"
                        score_attrs_parts.append(
                            f'{attr}="{float(s):.6f}"'
                        )
                score_attrs = (" " + " ".join(score_attrs_parts)) if score_attrs_parts else ""
                gate_st = str(row.get("gate_status", "unknown"))
                parts.append(
                    f'        <a class="frame thumb" '
                    f'href="{html_escape(rel_full)}" target="_blank" '
                    f'data-suite="{cap_suite}" '
                    f'data-label="{label_val}" '
                    f'data-lockbox="{html_escape(str(row["is_lockbox"]))}" '
                    f'data-face_size="{html_escape(fs)}" '
                    f'data-quality="{html_escape(q)}" '
                    f'data-gate_status="{html_escape(gate_st)}" '
                    f'data-video="{cap_video}"'
                    f'{score_attrs}>'
                    f'<img loading="lazy" src="{html_escape(rel_thumb)}" '
                    f'alt="{cap_video}" />'
                    f'<div class="cap"><b>{cap_suite}</b><br>{cap_video}</div>'
                    f'<div class="pill-row">{pill_fs}{pill_q}</div>'
                    f'{score_block}'
                    f'</a>\n'
                )
            _close_sb()
            parts.append('  </div>\n')  # label-block
        parts.append("</details>\n")

    parts.append(PAGE_TAIL)

    out_path = output_root / "index.html"
    out_path.write_text("".join(parts), encoding="utf-8")
    log.info(f"  wrote HTML: {out_path} ({out_path.stat().st_size/1024:.1f} KB)")
    return out_path


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------


def write_summary(df: pd.DataFrame, output_root: Path) -> Path:
    summary: dict = {
        "totals": {
            "frames": int(len(df)),
            "identities": int(df["base_identity"].nunique()),
            "real": int((df["label"] == 0).sum()),
            "fake": int((df["label"] == 1).sum()),
            "lockbox": int(df["is_lockbox"].astype(str).isin(["True", "1", "true"]).sum()),
        },
        "suites": dict(Counter(df["suite"])),
        "by_identity": {},
    }
    for ident, sub in df.groupby("base_identity"):
        summary["by_identity"][ident] = {
            "frames": int(len(sub)),
            "real": int((sub["label"] == 0).sum()),
            "fake": int((sub["label"] == 1).sum()),
            "lockbox": int(sub["is_lockbox"].astype(str).isin(["True", "1", "true"]).sum()),
            "suites": dict(Counter(sub["suite"])),
            "chronic_6": ident in CHRONIC_6,
        }
    out = output_root / "data" / "summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def configure_logging(output_root: Path) -> logging.Logger:
    log_path = output_root / "run.log"
    logger = logging.getLogger("identity_browser")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        action="append",
        type=Path,
        default=None,
        help="Path to a scope manifest CSV.  Repeat to merge multiple buckets.",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output directory root (default: %(default)s).",
    )
    ap.add_argument(
        "--score-csv",
        action="append",
        default=None,
        metavar="NAME=PATH_OR_GLOB",
        help=(
            "Per-checkpoint score source.  PATH_OR_GLOB is a frames-report CSV "
            "(or shell glob matching multiple suite CSVs from one ckpt).  CSVs "
            "must have columns frame_path, frame_prob.  Repeat to load multiple "
            "ckpts.  Order matters: the first ckpt is the primary sort key.  "
            "If omitted, defaults to "
            f"P8A={DEFAULT_P8A_GLOB!r}."
        ),
    )
    ap.add_argument(
        "--tags-parquet",
        type=Path,
        default=DEFAULT_TAGS_PARQUET,
        help="Parquet file with frame tags (default: %(default)s).",
    )
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the GCS download phase (use what's already on disk).",
    )
    ap.add_argument(
        "--skip-thumb",
        action="store_true",
        help="Skip the thumbnailing phase.",
    )
    ap.add_argument(
        "--skip-html",
        action="store_true",
        help="Skip the HTML render phase (useful for incremental download tests).",
    )
    args = ap.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "frames").mkdir(exist_ok=True)
    (output_root / "thumbs").mkdir(exist_ok=True)
    (output_root / "data").mkdir(exist_ok=True)
    log = configure_logging(output_root)

    # Default: the bundled scope_manifest.csv plus every *_manifest.csv in
    # data/. The latter accumulate as stage_*.py scripts add new buckets;
    # auto-discovering them prevents the next rebuild from silently dropping
    # identities (regression observed 2026-05-10: a default rebuild went from
    # 96 identities back to 34 because only scope_manifest.csv was loaded).
    if not args.manifest:
        bundled = [output_root / "scope_manifest.csv"]
        data_manifests = sorted((output_root / "data").glob("*_manifest.csv"))
        # Skip the auto-generated grouped_manifest{,_v2}.csv which isn't an input.
        data_manifests = [
            p for p in data_manifests
            if not p.name.startswith("grouped_manifest")
        ]
        args.manifest = bundled + data_manifests
        if data_manifests:
            # NOTE: don't shadow `log` — the outer script-level logger is
            # `log = configure_logging(...)` defined just above. Reuse it.
            log.info(
                "auto-discovered %d additional manifest(s) in data/: %s",
                len(data_manifests),
                ", ".join(p.name for p in data_manifests),
            )
    for m in args.manifest:
        if not m.exists():
            raise SystemExit(f"manifest not found: {m}")

    # Default: load all five wired checkpoints (P8A reference + E2B deployed +
    # PA_3800 + T3_S1_step1500 + T3_S1_step2500). Pass --score-csv NAME=PATH
    # to override.
    #
    # Each ckpt has TWO default score sources merged:
    # 1. The original frames_report.csv pool from cpu_followups / pa_pc_eval.
    # 2. The per-bucket *_scores_<NAME>.csv files in data/ (produced by
    #    stage_*.py scripts when new buckets get added). Without these,
    #    the additional manifests' frames have no score and show "—" for
    #    that ckpt — observed 2026-05-10 when only the cpu_followups glob
    #    was loaded. parse_score_specs concatenates duplicate-NAME specs.
    score_specs_raw = args.score_csv
    if score_specs_raw is None:
        per_bucket_glob = str(output_root / "data" / "*_scores_{ckpt}.csv")
        score_specs_raw = [
            f"P8A={DEFAULT_P8A_GLOB}",
            f"P8A={per_bucket_glob.format(ckpt='P8A')}",
            f"E2B={DEFAULT_E2B_GLOB}",
            f"E2B={per_bucket_glob.format(ckpt='E2B')}",
            f"PA_3800={DEFAULT_PA_3800_GLOB}",
            f"PA_3800={per_bucket_glob.format(ckpt='PA_3800')}",
            f"T3_S1_STEP1500={DEFAULT_T3_S1_STEP1500_GLOB}",
            f"T3_S1_STEP2500={DEFAULT_T3_S1_STEP2500_GLOB}",
        ]
    score_specs = parse_score_specs(score_specs_raw)

    log.info("=" * 70)
    log.info(f"Identity browser build starting (iter 2 — props + scores)")
    log.info(f"  output_root = {output_root}")
    log.info(f"  manifests   = {[str(p) for p in args.manifest]}")
    log.info(f"  score-csvs  = {score_specs_raw}")
    log.info(f"  tags-parq   = {args.tags_parquet}")

    t0 = time.monotonic()
    df = load_and_normalize_manifests(args.manifest)
    log.info(
        f"Manifest: {len(df)} unique frames, "
        f"{df['base_identity'].nunique()} base identities, "
        f"{df['suite'].nunique()} suites"
    )

    # Phase 0a — load tags + derive binary properties
    tags = load_tags(args.tags_parquet, log)
    df = derive_binary_properties(df, tags, log)

    # Phase 0b — load score lookups
    df, ckpt_names, score_coverage = enrich_with_scores(df, score_specs, log)

    # Persist the v1 grouped manifest for backwards compat AND a v2 with extras
    grouped_path = output_root / "data" / "grouped_manifest.csv"
    legacy_cols = [
        c for c in df.columns
        if c not in {f"score_{n}" for n in ckpt_names}
        and c not in {"face_size", "quality", "face_area_ratio", "is_low_quality"}
    ]
    df[legacy_cols].to_csv(grouped_path, index=False)
    log.info(f"Wrote {grouped_path} (legacy schema)")

    grouped_v2_path = output_root / "data" / "grouped_manifest_v2.csv"
    df.to_csv(grouped_v2_path, index=False)
    log.info(f"Wrote {grouped_v2_path} (with face_size, quality, score_*)")

    # Phase 1 -- download
    _download_t0_holder.clear()
    download_t0 = download_started_at()
    if args.skip_download:
        log.info("Skipping download phase (--skip-download)")
        ok_dl, fail_dl, cached_dl = 0, 0, 0
    else:
        pending, cached_dl = plan_downloads(df, output_root)
        log.info(f"Download plan: {len(pending)} pending, {cached_dl} already cached")
        ok_dl, fail_dl = download_via_gcs_client(pending, log)
    download_dt = time.monotonic() - download_t0

    # Phase 2 -- thumbnails
    thumb_t0 = time.monotonic()
    if args.skip_thumb:
        log.info("Skipping thumb phase (--skip-thumb)")
        ok_thumb, fail_thumb, missing_src = 0, 0, 0
    else:
        ok_thumb, fail_thumb, missing_src = make_thumbnails(df, output_root, log)
    thumb_dt = time.monotonic() - thumb_t0

    # Phase 3 -- HTML + summary
    html_t0 = time.monotonic()
    summary_path = write_summary(df, output_root)
    if args.skip_html:
        log.info("Skipping HTML phase (--skip-html)")
        html_path = None
    else:
        analytics = compute_analytics(df, ckpt_names, log)
        analytics_html = build_analytics_html(analytics)
        html_path = render_html(df, output_root, ckpt_names, log,
                                analytics_html=analytics_html)
    html_dt = time.monotonic() - html_t0

    # Disk usage
    def _du(path: Path) -> int:
        total = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total

    bytes_frames = _du(output_root / "frames")
    bytes_thumbs = _du(output_root / "thumbs")
    total_dt = time.monotonic() - t0

    log.info("=" * 70)
    log.info(f"Phase wall times:")
    log.info(f"  download : {download_dt:7.1f}s   (ok={ok_dl} cached={cached_dl} fail={fail_dl})")
    log.info(f"  thumbs   : {thumb_dt:7.1f}s   (ok={ok_thumb} fail={fail_thumb} missing_src={missing_src})")
    log.info(f"  html     : {html_dt:7.1f}s")
    log.info(f"  TOTAL    : {total_dt:7.1f}s")
    log.info(f"Disk usage:")
    log.info(f"  frames/  : {bytes_frames/1024/1024:7.1f} MB")
    log.info(f"  thumbs/  : {bytes_thumbs/1024/1024:7.1f} MB")
    log.info(f"Score coverage on manifest:")
    for n in ckpt_names:
        log.info(f"  {n}: {score_coverage.get(n, 0.0):.2f}%")
    log.info(f"Outputs:")
    log.info(f"  index.html       : {html_path}")
    log.info(f"  grouped manifest : {grouped_path}")
    log.info(f"  grouped (v2)     : {grouped_v2_path}")
    log.info(f"  summary          : {summary_path}")


if __name__ == "__main__":
    main()
