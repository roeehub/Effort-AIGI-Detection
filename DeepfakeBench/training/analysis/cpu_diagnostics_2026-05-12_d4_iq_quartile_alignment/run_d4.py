"""D4 — Per-IQ-quartile FPR alignment, dev vs lockbox.

Tests whether the dev↔lockbox substrate gap is purely an IQ-distribution shift
(matched-IQ-bin FPRs align across substrates) or has substrate-specific residual
beyond IQ.

Method (single-file, CPU only, n_jobs=1):
  For each ckpt in {P8A, T5C_step3500, T3_S1_step1500}:
    1. Calibrate tau on teams_real_all_dev (label=0) to give 5% FPR.
    2. For each of 6 IQ axes, compute dev-derived quartile cut points on
       teams_real_all_dev real frames (non-null IQ). Apply same cut points to
       teams_real_all_lockbox real frames. Compute per-quartile FPR @ tau on
       both substrates.
    3. Report Pearson r and mean abs delta across the 4 quartile cells.
    4. Joint-bin (lap_var x min_dim) median-split sanity check.
    5. Lockbox-like subset of dev: at lockbox Q1 cut points, find dev frames
       in that range and compare FPRs.

Reads:
  analysis/cpu_diagnostics_2026-05-12_stage_a/outputs/unified_frame_matrix.csv

Writes (to analysis/cpu_diagnostics_2026-05-12_d4_iq_quartile_alignment/outputs/):
  quartile_fpr_per_ckpt_per_axis.csv
  alignment_summary.csv
  joint_bin_alignment.csv
  lockbox_like_dev_fpr.csv
  summary.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------
ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
INPUT_CSV = ROOT / "analysis/cpu_diagnostics_2026-05-12_stage_a/outputs/unified_frame_matrix.csv"
OUT_DIR = ROOT / "analysis/cpu_diagnostics_2026-05-12_d4_iq_quartile_alignment"
OUT_OUTPUTS = OUT_DIR / "outputs"
OUT_OUTPUTS.mkdir(parents=True, exist_ok=True)

CKPTS = ["P8A", "T5C_step3500", "T3_S1_step1500"]

IQ_AXES = [
    "min_dim",
    "lap_var",
    "luma_mean",
    "saturation_mean",
    "color_a_dev",
    "color_b_dev",
]

DEV_SUITE = "teams_real_all_dev"
LOCKBOX_SUITE = "teams_real_all_lockbox"
TARGET_DEV_FPR = 0.05  # tau is set so dev FPR == 5% (on full dev real cohort)


def calibrate_tau(scores: np.ndarray, target_fpr: float) -> float:
    """Return the tau such that fraction(scores > tau) == target_fpr on this cohort.

    Uses the (1 - target_fpr)-quantile of the score distribution. With strict-greater
    comparison this gives FPR <= target_fpr with negligible slack on dense scores.
    """
    return float(np.quantile(scores, 1.0 - target_fpr))


def fpr_at_tau(scores: np.ndarray, tau: float) -> float:
    """Fraction of scores strictly greater than tau."""
    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores > tau))


def per_axis_quartile_fprs(
    dev_real: pd.DataFrame,
    lb_real: pd.DataFrame,
    ckpt: str,
    axis: str,
    tau: float,
) -> tuple[list[dict], dict]:
    """Compute per-quartile FPRs on dev and lockbox, with dev-derived cut points.

    Returns
    -------
    rows : list of dicts (one per quartile)
        Keys: ckpt, axis, quartile, dev_low, dev_high, dev_n, dev_fpr,
              lockbox_n, lockbox_fpr, delta_dev_minus_lockbox.
    summary : dict
        Keys: ckpt, axis, pearson_r_dev_vs_lockbox, mean_abs_delta,
              n_dev_total, n_lockbox_total.
    """
    dev_sub = dev_real[dev_real[axis].notna()].copy()
    lb_sub = lb_real[lb_real[axis].notna()].copy()

    # Dev-derived quartile cut points (25/50/75)
    q25, q50, q75 = dev_sub[axis].quantile([0.25, 0.5, 0.75]).tolist()
    # Use -inf/+inf for outer bounds so all rows are placed.
    bounds = [(-np.inf, q25), (q25, q50), (q50, q75), (q75, np.inf)]
    qlabels = ["Q1", "Q2", "Q3", "Q4"]

    rows = []
    dev_fprs = []
    lb_fprs = []
    for qlabel, (lo, hi) in zip(qlabels, bounds):
        if qlabel == "Q1":
            dev_mask = dev_sub[axis] <= hi
            lb_mask = lb_sub[axis] <= hi
        elif qlabel == "Q4":
            dev_mask = dev_sub[axis] > lo
            lb_mask = lb_sub[axis] > lo
        else:
            dev_mask = (dev_sub[axis] > lo) & (dev_sub[axis] <= hi)
            lb_mask = (lb_sub[axis] > lo) & (lb_sub[axis] <= hi)

        dev_cell = dev_sub.loc[dev_mask, ckpt].to_numpy()
        lb_cell = lb_sub.loc[lb_mask, ckpt].to_numpy()
        dev_fpr = fpr_at_tau(dev_cell, tau)
        lb_fpr = fpr_at_tau(lb_cell, tau)
        dev_fprs.append(dev_fpr)
        lb_fprs.append(lb_fpr)
        rows.append({
            "ckpt": ckpt,
            "axis": axis,
            "quartile": qlabel,
            "cut_low": lo,
            "cut_high": hi,
            "dev_n": int(dev_mask.sum()),
            "dev_fpr": dev_fpr,
            "lockbox_n": int(lb_mask.sum()),
            "lockbox_fpr": lb_fpr,
            "delta_dev_minus_lockbox": (dev_fpr - lb_fpr) if (not np.isnan(dev_fpr) and not np.isnan(lb_fpr)) else float("nan"),
        })

    # Correlation across 4 quartile cells (only over cells where both sides are valid)
    arr_dev = np.array(dev_fprs, dtype=float)
    arr_lb = np.array(lb_fprs, dtype=float)
    mask = ~(np.isnan(arr_dev) | np.isnan(arr_lb))
    if mask.sum() >= 2 and np.std(arr_dev[mask]) > 0 and np.std(arr_lb[mask]) > 0:
        r = float(np.corrcoef(arr_dev[mask], arr_lb[mask])[0, 1])
    else:
        r = float("nan")
    mean_abs_delta = float(np.nanmean(np.abs(arr_dev - arr_lb))) if mask.any() else float("nan")
    summary = {
        "ckpt": ckpt,
        "axis": axis,
        "pearson_r_dev_vs_lockbox": r,
        "mean_abs_delta": mean_abs_delta,
        "n_dev_total": int(len(dev_sub)),
        "n_lockbox_total": int(len(lb_sub)),
    }
    return rows, summary


def joint_bin_alignment(
    dev_real: pd.DataFrame,
    lb_real: pd.DataFrame,
    ckpt: str,
    tau: float,
) -> list[dict]:
    """4-cell joint median split on (lap_var, min_dim). Dev medians define cuts."""
    axes = ["lap_var", "min_dim"]
    dev_sub = dev_real[dev_real[axes].notna().all(axis=1)].copy()
    lb_sub = lb_real[lb_real[axes].notna().all(axis=1)].copy()

    lap_med = float(dev_sub["lap_var"].median())
    md_med = float(dev_sub["min_dim"].median())

    rows = []
    for lap_label, lap_mask_d, lap_mask_l in [
        ("lap_lo", dev_sub["lap_var"] <= lap_med, lb_sub["lap_var"] <= lap_med),
        ("lap_hi", dev_sub["lap_var"] > lap_med, lb_sub["lap_var"] > lap_med),
    ]:
        for md_label, md_mask_d, md_mask_l in [
            ("md_lo", dev_sub["min_dim"] <= md_med, lb_sub["min_dim"] <= md_med),
            ("md_hi", dev_sub["min_dim"] > md_med, lb_sub["min_dim"] > md_med),
        ]:
            dev_mask = lap_mask_d & md_mask_d
            lb_mask = lap_mask_l & md_mask_l
            dev_cell = dev_sub.loc[dev_mask, ckpt].to_numpy()
            lb_cell = lb_sub.loc[lb_mask, ckpt].to_numpy()
            dev_fpr = fpr_at_tau(dev_cell, tau)
            lb_fpr = fpr_at_tau(lb_cell, tau)
            rows.append({
                "ckpt": ckpt,
                "bin": f"{lap_label}__{md_label}",
                "lap_var_median": lap_med,
                "min_dim_median": md_med,
                "dev_n": int(dev_mask.sum()),
                "dev_fpr": dev_fpr,
                "lockbox_n": int(lb_mask.sum()),
                "lockbox_fpr": lb_fpr,
                "delta_dev_minus_lockbox": (dev_fpr - lb_fpr) if (not np.isnan(dev_fpr) and not np.isnan(lb_fpr)) else float("nan"),
            })
    return rows


def lockbox_like_dev_fpr(
    dev_real: pd.DataFrame,
    lb_real: pd.DataFrame,
    ckpt: str,
    axis: str,
    tau: float,
) -> dict:
    """At the lockbox's Q1 cut point (axis), find dev frames in that range; compare FPRs."""
    dev_sub = dev_real[dev_real[axis].notna()].copy()
    lb_sub = lb_real[lb_real[axis].notna()].copy()

    # Lockbox Q1 cut point: 25th percentile of lockbox axis
    lb_q25 = float(lb_sub[axis].quantile(0.25))

    lb_q1_mask = lb_sub[axis] <= lb_q25
    lb_q1 = lb_sub.loc[lb_q1_mask, ckpt].to_numpy()
    lb_q1_fpr = fpr_at_tau(lb_q1, tau)

    dev_like_mask = dev_sub[axis] <= lb_q25
    dev_like = dev_sub.loc[dev_like_mask, ckpt].to_numpy()
    dev_like_fpr = fpr_at_tau(dev_like, tau)

    return {
        "ckpt": ckpt,
        "axis": axis,
        "lockbox_q25_cut": lb_q25,
        "dev_like_n": int(dev_like_mask.sum()),
        "dev_like_fpr": dev_like_fpr,
        "lockbox_q1_n": int(lb_q1_mask.sum()),
        "lockbox_q1_fpr": lb_q1_fpr,
        "delta_dev_minus_lockbox": dev_like_fpr - lb_q1_fpr if (not np.isnan(dev_like_fpr) and not np.isnan(lb_q1_fpr)) else float("nan"),
        "abs_delta_le_0_02": bool(abs(dev_like_fpr - lb_q1_fpr) <= 0.02) if (not np.isnan(dev_like_fpr) and not np.isnan(lb_q1_fpr)) else False,
    }


def main() -> None:
    print(f"[d4] reading {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"[d4] total rows: {len(df)}")

    dev_all = df[df["suite"] == DEV_SUITE].copy()
    lb_all = df[df["suite"] == LOCKBOX_SUITE].copy()
    dev_real = dev_all[dev_all["label"] == 0].copy()
    lb_real = lb_all[lb_all["label"] == 0].copy()
    print(f"[d4] dev_real n={len(dev_real)} lockbox_real n={len(lb_real)}")

    # Non-null IQ counts (any of 6 axes; per-axis filtering done inside fns)
    dev_iq_nn = dev_real[IQ_AXES].notna().all(axis=1).sum()
    lb_iq_nn = lb_real[IQ_AXES].notna().all(axis=1).sum()
    print(f"[d4] dev_real all-6-IQ-non-null n={dev_iq_nn}; lockbox_real n={lb_iq_nn}")

    quartile_rows = []
    alignment_summary_rows = []
    joint_bin_rows = []
    lockbox_like_rows = []

    tau_by_ckpt: dict[str, float] = {}
    dev_fpr_at_tau_check: dict[str, float] = {}
    lb_fpr_at_tau: dict[str, float] = {}

    for ckpt in CKPTS:
        # Calibrate tau on full dev real cohort (no IQ filter) for 5% FPR
        dev_scores_all = dev_real[ckpt].dropna().to_numpy()
        tau = calibrate_tau(dev_scores_all, TARGET_DEV_FPR)
        tau_by_ckpt[ckpt] = tau
        dev_fpr_check = fpr_at_tau(dev_scores_all, tau)
        lb_scores_all = lb_real[ckpt].dropna().to_numpy()
        lb_fpr_all = fpr_at_tau(lb_scores_all, tau)
        dev_fpr_at_tau_check[ckpt] = dev_fpr_check
        lb_fpr_at_tau[ckpt] = lb_fpr_all
        print(f"[d4] {ckpt} tau={tau:.6f} dev_fpr_check={dev_fpr_check:.4f} lockbox_fpr_all={lb_fpr_all:.4f}")

        # Per-axis quartile FPRs
        for axis in IQ_AXES:
            rows, summary = per_axis_quartile_fprs(dev_real, lb_real, ckpt, axis, tau)
            quartile_rows.extend(rows)
            alignment_summary_rows.append(summary)

        # Joint (lap_var x min_dim) median split
        joint_bin_rows.extend(joint_bin_alignment(dev_real, lb_real, ckpt, tau))

        # Lockbox-like-dev FPR per axis
        for axis in IQ_AXES:
            lockbox_like_rows.append(lockbox_like_dev_fpr(dev_real, lb_real, ckpt, axis, tau))

    # Write CSVs
    q_df = pd.DataFrame(quartile_rows)
    a_df = pd.DataFrame(alignment_summary_rows)
    j_df = pd.DataFrame(joint_bin_rows)
    l_df = pd.DataFrame(lockbox_like_rows)

    q_df.to_csv(OUT_OUTPUTS / "quartile_fpr_per_ckpt_per_axis.csv", index=False)
    a_df.to_csv(OUT_OUTPUTS / "alignment_summary.csv", index=False)
    j_df.to_csv(OUT_OUTPUTS / "joint_bin_alignment.csv", index=False)
    l_df.to_csv(OUT_OUTPUTS / "lockbox_like_dev_fpr.csv", index=False)

    # Aggregates
    agg_per_ckpt = a_df.groupby("ckpt").agg(
        mean_pearson_r=("pearson_r_dev_vs_lockbox", "mean"),
        mean_abs_delta=("mean_abs_delta", "mean"),
    ).reset_index()
    agg_per_ckpt.to_csv(OUT_OUTPUTS / "alignment_summary_per_ckpt.csv", index=False)

    # Identify max abs delta cell per ckpt
    q_df["abs_delta"] = q_df["delta_dev_minus_lockbox"].abs()
    max_delta_rows = q_df.sort_values("abs_delta", ascending=False).groupby("ckpt").head(1)

    # Summary JSON
    summary = {
        "tau_by_ckpt": tau_by_ckpt,
        "dev_fpr_check": dev_fpr_at_tau_check,
        "lockbox_fpr_at_dev_calibrated_tau": lb_fpr_at_tau,
        "n_dev_real": int(len(dev_real)),
        "n_dev_real_all_iq_nonnull": int(dev_iq_nn),
        "n_lockbox_real": int(len(lb_real)),
        "n_lockbox_real_all_iq_nonnull": int(lb_iq_nn),
        "mean_pearson_r_per_ckpt": {row["ckpt"]: row["mean_pearson_r"] for _, row in agg_per_ckpt.iterrows()},
        "mean_abs_delta_per_ckpt": {row["ckpt"]: row["mean_abs_delta"] for _, row in agg_per_ckpt.iterrows()},
        "largest_delta_cell": max_delta_rows[["ckpt", "axis", "quartile", "dev_fpr", "lockbox_fpr", "delta_dev_minus_lockbox", "abs_delta"]].to_dict(orient="records"),
    }
    with open(OUT_OUTPUTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # Console summary
    print("\n[d4] === per-ckpt aggregate alignment ===")
    print(agg_per_ckpt.to_string(index=False))
    print("\n[d4] === largest absolute delta cell per ckpt ===")
    print(max_delta_rows[["ckpt", "axis", "quartile", "dev_fpr", "lockbox_fpr", "delta_dev_minus_lockbox"]].to_string(index=False))
    print("\n[d4] === lockbox-like-dev vs lockbox-Q1 FPR per axis ===")
    print(l_df[["ckpt", "axis", "dev_like_fpr", "lockbox_q1_fpr", "delta_dev_minus_lockbox", "abs_delta_le_0_02"]].to_string(index=False))

    print(f"\n[d4] outputs written to {OUT_OUTPUTS}")


if __name__ == "__main__":
    main()
