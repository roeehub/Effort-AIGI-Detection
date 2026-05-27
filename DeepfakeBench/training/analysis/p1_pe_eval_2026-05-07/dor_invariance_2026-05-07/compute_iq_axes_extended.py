"""
Extension of compute_iq_axes.py — re-computes IQ-axis correlations for ALL 8
ckpts (P8A, E2B, 6 P1 ckpts) using:

  - the now-updated scores_full.csv (8 score columns), and
  - cached per-frame IQ axes from axis_values_per_frame.csv
    (no image redownload, no recompute of width/height/min_dim/color_b_dev/sharpness).

Outputs:
  - axis_correlation_summary.csv (overwritten, ALL 8 ckpts)
  - axis_decoupling_trajectory.csv (focused 6-row × 6-col trajectory table for
    the 6 P1 ckpts in training-step order)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO, datefmt="%H:%M:%S")
log = logging.getLogger("iq_axes_ext")

THIS_DIR = Path(__file__).resolve().parent
SCORES = THIS_DIR / "scores_full.csv"
PRIOR_AXES = THIS_DIR / "axis_values_per_frame.csv"

# All 8 ckpts (P8A is reference for delta).
ALL_CKPTS = [
    "P8A",
    "E2B",
    "P1_BUNDLE_PERIODIC_STEP500",
    "P1_BUNDLE_TOP_N_STEP3750",
    "P1_BUNDLE_step4000",  # already in scores_full from prior run
    "P1_PAIRRANK_PERIODIC_STEP500",
    "P1_PAIRRANK_TOP_N_STEP6000",
    "P1_PAIRRANK_step6750",  # already in scores_full from prior run
]

# 6 P1 ckpts in TRAINING-STEP order (BUNDLE 500/3750/4000, PAIRRANK 500/6000/6750).
P1_TRAJECTORY = [
    ("P1_BUNDLE_PERIODIC_STEP500", "BUNDLE", 500),
    ("P1_BUNDLE_TOP_N_STEP3750",   "BUNDLE", 3750),
    ("P1_BUNDLE_step4000",         "BUNDLE", 4000),
    ("P1_PAIRRANK_PERIODIC_STEP500", "PAIRRANK", 500),
    ("P1_PAIRRANK_TOP_N_STEP6000", "PAIRRANK", 6000),
    ("P1_PAIRRANK_step6750",       "PAIRRANK", 6750),
]

AXES = ["sharpness", "min_dim", "color_b_dev"]


def pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]; y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    scores = pd.read_csv(SCORES)
    log.info("scores_full.csv: %d rows, score cols=%s",
             len(scores), [c for c in scores.columns if c.startswith("score_")])
    if len(scores) != 180:
        raise RuntimeError(f"expected 180 rows, got {len(scores)}")

    # Verify all 8 ckpts have score columns.
    missing = [ck for ck in ALL_CKPTS if f"score_{ck}" not in scores.columns]
    if missing:
        raise RuntimeError(f"missing score columns for: {missing}")

    # Pull cached IQ axes (width/height/min_dim/color_b_dev/sharpness) from prior run.
    prior = pd.read_csv(PRIOR_AXES)
    axis_only = prior[["variant", "frame_path", "width", "height", "min_dim",
                       "color_b_dev", "sharpness"]].copy()
    log.info("loaded cached per-frame axes: %d rows", len(axis_only))

    # Merge: scores_full × cached axes (180-row inner join on (variant, frame_path)).
    full = scores.merge(axis_only, on=["variant", "frame_path"], how="inner")
    if len(full) != 180:
        raise RuntimeError(f"merge produced {len(full)} rows, expected 180")

    # Compute Δ-vs-P8A for the 7 non-P8A ckpts.
    for ck in ALL_CKPTS:
        if ck == "P8A":
            continue
        full[f"delta_{ck}"] = full[f"score_{ck}"] - full["score_P8A"]

    # Persist the extended per-frame table (with new score cols + deltas).
    out_per_frame = THIS_DIR / "axis_values_per_frame.csv"
    full.to_csv(out_per_frame, index=False)
    log.info("rewrote %s (n=%d, %d cols)", out_per_frame, len(full), len(full.columns))

    # Build the long-form correlation table: per (axis, ckpt, target_kind, scope).
    rows: list[dict] = []
    variants = sorted(full["variant"].unique())
    for axis in AXES + (["min_dim"] if False else []):  # keep order: sharpness, min_dim, color_b_dev
        pass  # noop guard
    for axis in AXES:
        # Raw scores
        for ck in ALL_CKPTS:
            col = f"score_{ck}"
            r_all = pearson_safe(full[axis].values, full[col].values)
            rows.append({
                "axis": axis, "ckpt": ck, "target_kind": "raw",
                "scope": "ALL", "n": len(full), "pearson_r": r_all,
            })
            for v in variants:
                sub = full[full["variant"] == v]
                rows.append({
                    "axis": axis, "ckpt": ck, "target_kind": "raw",
                    "scope": v, "n": len(sub),
                    "pearson_r": pearson_safe(sub[axis].values, sub[col].values),
                })
        # Delta vs P8A (skip P8A itself)
        for ck in ALL_CKPTS:
            if ck == "P8A":
                continue
            col = f"delta_{ck}"
            r_all = pearson_safe(full[axis].values, full[col].values)
            rows.append({
                "axis": axis, "ckpt": ck, "target_kind": "delta_vs_P8A",
                "scope": "ALL", "n": len(full), "pearson_r": r_all,
            })
            for v in variants:
                sub = full[full["variant"] == v]
                rows.append({
                    "axis": axis, "ckpt": ck, "target_kind": "delta_vs_P8A",
                    "scope": v, "n": len(sub),
                    "pearson_r": pearson_safe(sub[axis].values, sub[col].values),
                })

    corr = pd.DataFrame(rows)
    out_corr = THIS_DIR / "axis_correlation_summary.csv"
    corr.to_csv(out_corr, index=False)
    log.info("wrote %s (n=%d rows)", out_corr, len(corr))

    # Build the focused trajectory table (6 P1 ckpts × 6 cols).
    # Cols: sharpness raw-r, sharpness Δ-r, min_dim raw-r, min_dim Δ-r,
    #       color_b_dev raw-r, color_b_dev Δ-r — all combined-suite (n=180).
    traj_rows: list[dict] = []
    for ck, arm, step in P1_TRAJECTORY:
        rec: dict = {"ckpt": ck, "arm": arm, "step": step}
        for axis in AXES:
            r_raw = corr[(corr["axis"] == axis) & (corr["ckpt"] == ck) &
                         (corr["target_kind"] == "raw") &
                         (corr["scope"] == "ALL")]["pearson_r"].iloc[0]
            r_delta = corr[(corr["axis"] == axis) & (corr["ckpt"] == ck) &
                           (corr["target_kind"] == "delta_vs_P8A") &
                           (corr["scope"] == "ALL")]["pearson_r"].iloc[0]
            rec[f"{axis}_raw_r"] = r_raw
            rec[f"{axis}_delta_r"] = r_delta
        traj_rows.append(rec)

    traj = pd.DataFrame(traj_rows)
    out_traj = THIS_DIR / "axis_decoupling_trajectory.csv"
    traj.to_csv(out_traj, index=False)
    log.info("wrote %s (n=%d rows)", out_traj, len(traj))

    # Console summaries.
    print()
    print("=" * 100)
    print("PEARSON r — RAW SCORE vs AXIS (combined ALL n=180)  —  ALL 8 CKPTS")
    print("=" * 100)
    pivot_raw = corr[(corr["scope"] == "ALL") &
                     (corr["target_kind"] == "raw")].pivot(
        index="ckpt", columns="axis", values="pearson_r"
    ).reindex(ALL_CKPTS)[AXES]
    print(pivot_raw.round(3).to_string())

    print()
    print("=" * 100)
    print("PEARSON r — DELTA vs P8A vs AXIS (combined ALL n=180)  —  7 NON-P8A CKPTS")
    print("=" * 100)
    pivot_delta = corr[(corr["scope"] == "ALL") &
                       (corr["target_kind"] == "delta_vs_P8A")].pivot(
        index="ckpt", columns="axis", values="pearson_r"
    ).reindex([c for c in ALL_CKPTS if c != "P8A"])[AXES]
    print(pivot_delta.round(3).to_string())

    print()
    print("=" * 100)
    print("AXIS DECOUPLING TRAJECTORY  —  6 P1 CKPTS × 6 COLS  (combined n=180)")
    print("=" * 100)
    print(traj.set_index(["arm", "step", "ckpt"]).round(3).to_string())


if __name__ == "__main__":
    main()
