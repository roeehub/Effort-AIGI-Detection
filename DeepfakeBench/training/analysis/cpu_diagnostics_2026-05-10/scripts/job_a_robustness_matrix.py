"""Job A — Multi-axis robustness matrix.

For each ckpt, calibrate tau to give 5% FPR on the overall real cohort
(teams_real_all_dev). Then bin frames by IQ axis quartile and measure
per-bin FPR + recall. Output: per-bin table + per-ckpt robustness summary.

Real and fake suites:
- real: teams_real_all_dev
- fake: visomaster_enhanced_macro_dev, deeplive_enhanced_dev, teams_fake_all_dev

Axes: lap_var (sharpness), min_dim (resolution), luma_mean (brightness),
color_a_dev, color_b_dev, saturation_mean.

Usage: python job_a_robustness_matrix.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT = ROOT / "analysis/cpu_diagnostics_2026-05-10/outputs"
OUT.mkdir(parents=True, exist_ok=True)

IQ_ATLAS = ROOT / "analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet"

# (ckpt_alias, real_suite_csv, fake_suite_csvs)
CKPT_REPORTS: Dict[str, Dict[str, Path]] = {
    "P8A": {
        "teams_real_all_dev": ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/teams_real_all_dev_p8a_reference_step5000_frames_report.csv",
        "visomaster_enhanced_macro_dev": ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/visomaster_enhanced_macro_dev_p8a_reference_step5000_frames_report.csv",
        "deeplive_enhanced_dev": ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/deeplive_enhanced_dev_p8a_reference_step5000_frames_report.csv",
        "teams_fake_all_dev": ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/teams_fake_all_dev_p8a_reference_step5000_frames_report.csv",
    },
    "E2B": {
        "teams_real_all_dev": ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/teams_real_all_dev_e2b_top_n_step3200_frames_report.csv",
        "visomaster_enhanced_macro_dev": ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/visomaster_enhanced_macro_dev_e2b_top_n_step3200_frames_report.csv",
        "deeplive_enhanced_dev": ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/deeplive_enhanced_dev_e2b_top_n_step3200_frames_report.csv",
        "teams_fake_all_dev": ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/teams_fake_all_dev_e2b_top_n_step3200_frames_report.csv",
    },
    "T3_S1_step1500": {
        "teams_real_all_dev": ROOT / "analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step1500/teams_real_all_dev_t3_slot1_periodic_step1500_frames_report.csv",
        "visomaster_enhanced_macro_dev": ROOT / "analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step1500/visomaster_enhanced_macro_dev_t3_slot1_periodic_step1500_frames_report.csv",
        "deeplive_enhanced_dev": ROOT / "analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step1500/deeplive_enhanced_dev_t3_slot1_periodic_step1500_frames_report.csv",
        "teams_fake_all_dev": ROOT / "analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step1500/teams_fake_all_dev_t3_slot1_periodic_step1500_frames_report.csv",
    },
    "T3_S1_step2500": {
        "teams_real_all_dev": ROOT / "analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step2500/teams_real_all_dev_t3_slot1_periodic_step2500_frames_report.csv",
        "visomaster_enhanced_macro_dev": ROOT / "analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step2500/visomaster_enhanced_macro_dev_t3_slot1_periodic_step2500_frames_report.csv",
        "deeplive_enhanced_dev": ROOT / "analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step2500/deeplive_enhanced_dev_t3_slot1_periodic_step2500_frames_report.csv",
        "teams_fake_all_dev": ROOT / "analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step2500/teams_fake_all_dev_t3_slot1_periodic_step2500_frames_report.csv",
    },
}

REAL_SUITE = "teams_real_all_dev"
FAKE_SUITES = ["visomaster_enhanced_macro_dev", "deeplive_enhanced_dev", "teams_fake_all_dev"]
TARGET_FPR = 0.05  # operating point for tau calibration

AXES = [
    "lap_var",
    "min_dim",
    "luma_mean",
    "color_a_dev",
    "color_b_dev",
    "saturation_mean",
]


def load_iq_atlas() -> pd.DataFrame:
    df = pd.read_parquet(IQ_ATLAS)
    return df


def load_scores(reports: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    out = {}
    for suite, p in reports.items():
        if not p.exists():
            print(f"  WARN: missing {p}", file=sys.stderr)
            continue
        df = pd.read_csv(p, usecols=["frame_path", "frame_prob", "label"])
        out[suite] = df
    return out


def calibrate_tau(real_scores: np.ndarray, target_fpr: float) -> float:
    """Find tau such that fraction of reals scoring > tau equals target_fpr."""
    sorted_desc = np.sort(real_scores)[::-1]
    n = len(sorted_desc)
    k = max(1, int(round(target_fpr * n)))
    if k >= n:
        return float(sorted_desc[-1]) - 1e-9
    return float(sorted_desc[k - 1])


def quartile_bins(values: pd.Series) -> pd.Series:
    """Return labels Q1..Q4 by quartile of values, NaN-safe."""
    vals = values.copy()
    finite = vals.dropna()
    if len(finite) < 4:
        return pd.Series(["NA"] * len(vals), index=vals.index)
    q = np.quantile(finite, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins = ["Q1", "Q2", "Q3", "Q4"]
    out = pd.Series(["NA"] * len(vals), index=vals.index)
    for i, label in enumerate(bins):
        lo, hi = q[i], q[i + 1]
        if i == 3:
            mask = (vals >= lo) & (vals <= hi)
        else:
            mask = (vals >= lo) & (vals < hi)
        out.loc[mask] = label
    return out


def main():
    print("Loading IQ atlas...")
    iq = load_iq_atlas()
    iq_keep_cols = ["frame_path"] + AXES
    iq_lite = iq[iq_keep_cols].drop_duplicates(subset="frame_path")
    print(f"  IQ atlas: {len(iq):,} rows, {len(iq_lite):,} unique frame_paths")

    rows: List[dict] = []
    summary_rows: List[dict] = []

    for ckpt_name, reports in CKPT_REPORTS.items():
        print(f"\nProcessing {ckpt_name}...")
        scores = load_scores(reports)
        if REAL_SUITE not in scores:
            print(f"  SKIP — no real suite for {ckpt_name}")
            continue

        # Calibrate tau on the full real cohort
        real_df = scores[REAL_SUITE].merge(iq_lite, on="frame_path", how="left")
        n_real_total = len(real_df)
        n_real_with_iq = real_df[AXES[0]].notna().sum()
        tau = calibrate_tau(real_df["frame_prob"].values, TARGET_FPR)
        actual_fpr = (real_df["frame_prob"] > tau).mean()
        print(f"  tau@FPR={TARGET_FPR:.0%} = {tau:.4f}  (actual FPR={actual_fpr:.4f}, n_real={n_real_total}, n_with_iq={n_real_with_iq})")

        # Build pooled fake df (across 3 fake suites) for recall
        fake_df_parts = []
        for suite in FAKE_SUITES:
            if suite not in scores:
                continue
            sub = scores[suite].merge(iq_lite, on="frame_path", how="left")
            sub["fake_suite"] = suite
            fake_df_parts.append(sub)
        if not fake_df_parts:
            print(f"  SKIP — no fake suites for {ckpt_name}")
            continue
        fake_df = pd.concat(fake_df_parts, ignore_index=True)

        # Per-axis-bin FPR and recall
        for axis in AXES:
            real_with_axis = real_df.dropna(subset=[axis]).copy()
            real_with_axis["bin"] = quartile_bins(real_with_axis[axis])

            # Quartile thresholds from REAL distribution; apply same edges to fakes
            finite = real_with_axis[axis].dropna()
            q = np.quantile(finite, [0.0, 0.25, 0.5, 0.75, 1.0])

            fake_with_axis = fake_df.dropna(subset=[axis]).copy()
            # bin fakes by REAL distribution's quartile edges (so we measure recall in same bins)
            def assign_bin(v):
                if pd.isna(v):
                    return "NA"
                if v < q[1]:
                    return "Q1"
                if v < q[2]:
                    return "Q2"
                if v < q[3]:
                    return "Q3"
                return "Q4"
            fake_with_axis["bin"] = fake_with_axis[axis].apply(assign_bin)

            for binlabel in ["Q1", "Q2", "Q3", "Q4"]:
                rb = real_with_axis[real_with_axis["bin"] == binlabel]
                fb = fake_with_axis[fake_with_axis["bin"] == binlabel]
                if len(rb) == 0:
                    continue
                fpr = (rb["frame_prob"] > tau).mean()
                recall = (fb["frame_prob"] > tau).mean() if len(fb) > 0 else float("nan")
                row = {
                    "ckpt": ckpt_name,
                    "axis": axis,
                    "bin": binlabel,
                    "n_real": int(len(rb)),
                    "n_fake": int(len(fb)),
                    "fpr": float(fpr),
                    "recall": float(recall),
                    "tau": tau,
                    "axis_lo": float(q[{"Q1":0,"Q2":1,"Q3":2,"Q4":3}[binlabel]]),
                    "axis_hi": float(q[{"Q1":1,"Q2":2,"Q3":3,"Q4":4}[binlabel]]),
                }
                rows.append(row)

        # Summary per ckpt
        ckpt_rows = [r for r in rows if r["ckpt"] == ckpt_name]
        for axis in AXES:
            axis_rows = [r for r in ckpt_rows if r["axis"] == axis]
            if not axis_rows:
                continue
            fprs = [r["fpr"] for r in axis_rows]
            recalls = [r["recall"] for r in axis_rows if not np.isnan(r["recall"])]
            summary_rows.append({
                "ckpt": ckpt_name,
                "axis": axis,
                "fpr_mean": float(np.mean(fprs)),
                "fpr_max": float(np.max(fprs)),
                "fpr_min": float(np.min(fprs)),
                "fpr_std": float(np.std(fprs)),
                "fpr_max_minus_min": float(np.max(fprs) - np.min(fprs)),
                "recall_mean": float(np.mean(recalls)) if recalls else float("nan"),
                "recall_min": float(np.min(recalls)) if recalls else float("nan"),
                "recall_std": float(np.std(recalls)) if recalls else float("nan"),
                "recall_max_minus_min": float(np.max(recalls) - np.min(recalls)) if recalls else float("nan"),
                "n_bins": len(axis_rows),
            })

    df_rows = pd.DataFrame(rows)
    df_summary = pd.DataFrame(summary_rows)

    df_rows.to_csv(OUT / "job_a_robustness_per_bin.csv", index=False)
    df_summary.to_csv(OUT / "job_a_robustness_summary.csv", index=False)
    print(f"\nWrote {OUT / 'job_a_robustness_per_bin.csv'} ({len(df_rows)} rows)")
    print(f"Wrote {OUT / 'job_a_robustness_summary.csv'} ({len(df_summary)} rows)")

    # Headline robustness coefficient
    overall = (
        df_summary.groupby("ckpt")
        .agg(
            mean_fpr_spread=("fpr_max_minus_min", "mean"),
            max_fpr_in_any_bin=("fpr_max", "max"),
            mean_recall=("recall_mean", "mean"),
            min_recall=("recall_min", "min"),
            mean_recall_spread=("recall_max_minus_min", "mean"),
        )
        .reset_index()
    )
    # Robustness score: penalize FPR spread + recall spread; reward mean recall
    overall["robustness_score"] = (
        overall["mean_recall"]
        - overall["mean_fpr_spread"]
        - 0.5 * overall["mean_recall_spread"]
    )
    overall = overall.sort_values("robustness_score", ascending=False)
    overall.to_csv(OUT / "job_a_robustness_headline.csv", index=False)
    print(f"\nHeadline robustness ranking:")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
