"""Job B — Catastrophic-tail comparison.

For each ckpt, count real frames that score "catastrophically high" (>0.9, >0.95, >0.99)
across the broadest real cohort with frame-level data for all 4 ckpts.

The "fragility" question: at deployment-grade tau, how many reals does each ckpt
score as confidently fake (>0.9)? A wider tail means more false-flag surprises in
production-divergent conditions.

Suites used: teams_real_all_dev (n=4564) — only suite where all 4 ckpts have
frame-level reports.

Also computes catastrophic-miss tail on fakes (fake frames scoring <0.05).

Usage: python job_b_catastrophic_tail.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT = ROOT / "analysis/cpu_diagnostics_2026-05-10/outputs"

# Per-frame report paths, by ckpt
REPORTS = {
    "P8A": {
        "teams_real_all_dev": ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/teams_real_all_dev_p8a_reference_step5000_frames_report.csv",
        "teams_fake_all_dev": ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/teams_fake_all_dev_p8a_reference_step5000_frames_report.csv",
        "visomaster_enhanced_macro_dev": ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/visomaster_enhanced_macro_dev_p8a_reference_step5000_frames_report.csv",
        "deeplive_enhanced_dev": ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/deeplive_enhanced_dev_p8a_reference_step5000_frames_report.csv",
    },
    "E2B": {
        "teams_real_all_dev": ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/teams_real_all_dev_e2b_top_n_step3200_frames_report.csv",
        "teams_fake_all_dev": ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/teams_fake_all_dev_e2b_top_n_step3200_frames_report.csv",
        "visomaster_enhanced_macro_dev": ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/visomaster_enhanced_macro_dev_e2b_top_n_step3200_frames_report.csv",
        "deeplive_enhanced_dev": ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/deeplive_enhanced_dev_e2b_top_n_step3200_frames_report.csv",
    },
    "T3_S1_step1500": {
        "teams_real_all_dev": ROOT / "analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step1500/teams_real_all_dev_t3_slot1_periodic_step1500_frames_report.csv",
        "teams_fake_all_dev": ROOT / "analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step1500/teams_fake_all_dev_t3_slot1_periodic_step1500_frames_report.csv",
        "visomaster_enhanced_macro_dev": ROOT / "analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step1500/visomaster_enhanced_macro_dev_t3_slot1_periodic_step1500_frames_report.csv",
        "deeplive_enhanced_dev": ROOT / "analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step1500/deeplive_enhanced_dev_t3_slot1_periodic_step1500_frames_report.csv",
    },
    "T3_S1_step2500": {
        "teams_real_all_dev": ROOT / "analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step2500/teams_real_all_dev_t3_slot1_periodic_step2500_frames_report.csv",
        "teams_fake_all_dev": ROOT / "analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step2500/teams_fake_all_dev_t3_slot1_periodic_step2500_frames_report.csv",
        "visomaster_enhanced_macro_dev": ROOT / "analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step2500/visomaster_enhanced_macro_dev_t3_slot1_periodic_step2500_frames_report.csv",
        "deeplive_enhanced_dev": ROOT / "analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step2500/deeplive_enhanced_dev_t3_slot1_periodic_step2500_frames_report.csv",
    },
}

REAL_SUITE = "teams_real_all_dev"
FAKE_SUITES = ["teams_fake_all_dev", "visomaster_enhanced_macro_dev", "deeplive_enhanced_dev"]
TARGET_FPR = 0.05


def calibrate_tau(real_scores: np.ndarray, target_fpr: float) -> float:
    sorted_desc = np.sort(real_scores)[::-1]
    n = len(sorted_desc)
    k = max(1, int(round(target_fpr * n)))
    if k >= n:
        return float(sorted_desc[-1]) - 1e-9
    return float(sorted_desc[k - 1])


def main():
    rows = []
    extreme_frames_per_ckpt: Dict[str, pd.DataFrame] = {}

    for ckpt, paths in REPORTS.items():
        real_df = pd.read_csv(paths[REAL_SUITE], usecols=["frame_path", "frame_prob", "video_id"])
        n_real = len(real_df)
        tau5 = calibrate_tau(real_df["frame_prob"].values, 0.05)
        # Count tail
        tail_thresholds = [0.5, 0.7, 0.9, 0.95, 0.99]
        for t in tail_thresholds:
            n_above = (real_df["frame_prob"] > t).sum()
            rows.append({
                "ckpt": ckpt,
                "metric": f"reals_above_{t}",
                "suite": REAL_SUITE,
                "n_total": n_real,
                "n_above_thresh": int(n_above),
                "frac_above": float(n_above / n_real),
                "calibrated_tau_5pct_FPR": tau5,
            })

        # Catastrophic-tail measure: count of reals that score above the 99th percentile of cohort
        sorted_desc = np.sort(real_df["frame_prob"].values)[::-1]
        # tau at 1% FPR (severe deployment threshold)
        tau1 = calibrate_tau(real_df["frame_prob"].values, 0.01)
        n_above_99th = (real_df["frame_prob"] > tau1).sum()
        rows.append({
            "ckpt": ckpt,
            "metric": "reals_above_calibrated_tau_1pct_FPR",
            "suite": REAL_SUITE,
            "n_total": n_real,
            "n_above_thresh": int(n_above_99th),
            "frac_above": float(n_above_99th / n_real),
            "calibrated_tau_5pct_FPR": tau5,
        })

        # Top-50 most over-fired reals: identify them per ckpt
        top = real_df.sort_values("frame_prob", ascending=False).head(50)[["frame_path", "frame_prob", "video_id"]].copy()
        top["ckpt"] = ckpt
        extreme_frames_per_ckpt[ckpt] = top

        # ---- Fake-side: catastrophic miss (fake scored very low) ----
        for fake_suite in FAKE_SUITES:
            if fake_suite not in paths:
                continue
            fake_df = pd.read_csv(paths[fake_suite], usecols=["frame_path", "frame_prob"])
            n_fake = len(fake_df)
            for t in [0.05, 0.1, 0.3]:
                n_below = (fake_df["frame_prob"] < t).sum()
                rows.append({
                    "ckpt": ckpt,
                    "metric": f"fakes_below_{t}",
                    "suite": fake_suite,
                    "n_total": n_fake,
                    "n_above_thresh": int(n_below),
                    "frac_above": float(n_below / n_fake),  # reused col name; here it's frac_below
                    "calibrated_tau_5pct_FPR": tau5,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "job_b_catastrophic_tail.csv", index=False)

    # Concatenate top-50 for inspection
    top_all = pd.concat(list(extreme_frames_per_ckpt.values()), ignore_index=True)
    top_all.to_csv(OUT / "job_b_top50_overfired_reals_per_ckpt.csv", index=False)
    print(f"Wrote {OUT / 'job_b_catastrophic_tail.csv'} ({len(df)} rows)")
    print(f"Wrote {OUT / 'job_b_top50_overfired_reals_per_ckpt.csv'} ({len(top_all)} rows)")

    # Headline: REAL-side catastrophic tail
    print("\n" + "=" * 80)
    print("Catastrophic-tail count (real frames scored above threshold)")
    print(f"On {REAL_SUITE} (n=4564)")
    print("=" * 80)
    pivot = (
        df[df["metric"].str.startswith("reals_above_")]
        .pivot(index="ckpt", columns="metric", values="n_above_thresh")
        .reindex(["P8A", "E2B", "T3_S1_step1500", "T3_S1_step2500"])
    )
    print(pivot)
    print()
    print("As % of cohort:")
    pivot_pct = pivot / 4564 * 100
    print(pivot_pct.round(2))

    # Headline: FAKE-side catastrophic miss
    print("\n" + "=" * 80)
    print("Catastrophic-miss count (fake frames scored below threshold)")
    print("=" * 80)
    for fs in FAKE_SUITES:
        sub = df[(df["suite"] == fs) & (df["metric"].str.startswith("fakes_below_"))]
        pivot_fake = sub.pivot(index="ckpt", columns="metric", values="frac_above").reindex(
            ["P8A", "E2B", "T3_S1_step1500", "T3_S1_step2500"]
        )
        print(f"\n{fs}:")
        print((pivot_fake * 100).round(1).astype(str) + "%")

    # Identify SHARED catastrophic-FP reals (frames that ALL ckpts score >0.9)
    print("\n" + "=" * 80)
    print("Shared catastrophic FPs (real frames scored >0.9 by ALL 4 ckpts)")
    print("=" * 80)
    real_dfs = {}
    for ckpt, paths in REPORTS.items():
        real_dfs[ckpt] = pd.read_csv(paths[REAL_SUITE], usecols=["frame_path", "frame_prob"]).set_index("frame_path")
    common = pd.DataFrame({ckpt: d["frame_prob"] for ckpt, d in real_dfs.items()})
    shared = common[(common > 0.9).all(axis=1)]
    print(f"  Shared catastrophic-FP reals (all 4 ckpts > 0.9): n = {len(shared)}")
    if len(shared) > 0:
        # Most-shared catastrophic frames
        shared = shared.assign(min_score=shared.min(axis=1)).sort_values("min_score", ascending=False)
        print(shared.head(10))

    # P8A-only catastrophic FPs (P8A > 0.9, others < 0.5)
    p8a_only = common[(common["P8A"] > 0.9) & (common["E2B"] < 0.5) & (common["T3_S1_step1500"] < 0.5) & (common["T3_S1_step2500"] < 0.5)]
    print(f"\n  P8A-only catastrophic FPs (P8A>0.9, others<0.5): n = {len(p8a_only)}")
    # T3 step2500-only catastrophic FPs
    t2500_only = common[(common["T3_S1_step2500"] > 0.9) & (common["P8A"] < 0.5) & (common["E2B"] < 0.5) & (common["T3_S1_step1500"] < 0.5)]
    print(f"  T3_step2500-only catastrophic FPs (step2500>0.9, others<0.5): n = {len(t2500_only)}")
    # T3 step1500-only catastrophic FPs
    t1500_only = common[(common["T3_S1_step1500"] > 0.9) & (common["P8A"] < 0.5) & (common["E2B"] < 0.5) & (common["T3_S1_step2500"] < 0.5)]
    print(f"  T3_step1500-only catastrophic FPs (step1500>0.9, others<0.5): n = {len(t1500_only)}")
    # E2B-only
    e2b_only = common[(common["E2B"] > 0.9) & (common["P8A"] < 0.5) & (common["T3_S1_step1500"] < 0.5) & (common["T3_S1_step2500"] < 0.5)]
    print(f"  E2B-only catastrophic FPs (E2B>0.9, others<0.5): n = {len(e2b_only)}")

    # Save the unique-FP frame sets
    for label, sub in [("p8a_only", p8a_only), ("e2b_only", e2b_only), ("t3_step1500_only", t1500_only), ("t3_step2500_only", t2500_only), ("shared_all_4", shared)]:
        sub.reset_index().to_csv(OUT / f"job_b_unique_FPs_{label}.csv", index=False)


if __name__ == "__main__":
    main()
