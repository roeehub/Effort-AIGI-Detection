"""Job D — Cross-ckpt disagreement structure.

Question: where do the 4 ckpts disagree maximally on the same frames?
What axes characterize the disagreement?

For each frame in teams_real_all_dev: compute score range across (P8A, E2B,
step1500, step2500). Identify "controversial" frames (range > 0.7).
Cross-tabulate by IQ axis quartile.

Usage: python job_d_disagreement.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT = ROOT / "analysis/cpu_diagnostics_2026-05-10/outputs"
IQ_ATLAS = ROOT / "analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet"

REAL_REPORTS = {
    "P8A": ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/teams_real_all_dev_p8a_reference_step5000_frames_report.csv",
    "E2B": ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/teams_real_all_dev_e2b_top_n_step3200_frames_report.csv",
    "T3_S1_step1500": ROOT / "analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step1500/teams_real_all_dev_t3_slot1_periodic_step1500_frames_report.csv",
    "T3_S1_step2500": ROOT / "analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step2500/teams_real_all_dev_t3_slot1_periodic_step2500_frames_report.csv",
}


def main():
    iq = pd.read_parquet(IQ_ATLAS)[["frame_path", "lap_var", "min_dim", "color_a_dev", "saturation_mean", "luma_mean"]]
    df_dict = {ckpt: pd.read_csv(p, usecols=["frame_path", "frame_prob"]).set_index("frame_path")["frame_prob"]
               for ckpt, p in REAL_REPORTS.items()}
    common = pd.DataFrame(df_dict)
    common["score_range"] = common.max(axis=1) - common.min(axis=1)
    common["score_max_ckpt"] = common[["P8A","E2B","T3_S1_step1500","T3_S1_step2500"]].idxmax(axis=1)
    common["score_min_ckpt"] = common[["P8A","E2B","T3_S1_step1500","T3_S1_step2500"]].idxmin(axis=1)
    common = common.reset_index().merge(iq, on="frame_path", how="left")

    print(f"Total real frames: {len(common):,}")
    print(f"  with IQ data: {common['lap_var'].notna().sum():,}")

    print("\n" + "=" * 90)
    print("Score range distribution across 4 ckpts (max - min)")
    print("=" * 90)
    for q in [0.5, 0.75, 0.9, 0.95, 0.99]:
        v = common["score_range"].quantile(q)
        print(f"  p{int(q*100):02d}: {v:.3f}")

    # Controversial: range > 0.7
    controversial = common[common["score_range"] > 0.7]
    print(f"\nControversial frames (range > 0.7): {len(controversial)} / {len(common)} ({len(controversial)/len(common)*100:.1f}%)")

    # Which ckpt is the "outlier high" most often on controversial frames?
    print("\n  Outlier-high (max-scoring) ckpt distribution on controversial frames:")
    print(controversial["score_max_ckpt"].value_counts().to_string())
    print("\n  Outlier-low (min-scoring) ckpt distribution on controversial frames:")
    print(controversial["score_min_ckpt"].value_counts().to_string())

    # IQ axis profile of controversial frames
    print("\n" + "=" * 90)
    print("IQ profile of controversial frames vs full cohort (mean values)")
    print("=" * 90)
    iq_axes = ["lap_var", "min_dim", "color_a_dev", "saturation_mean", "luma_mean"]
    for axis in iq_axes:
        full_mean = common[axis].mean()
        contro_mean = controversial[axis].mean()
        print(f"  {axis:18s}  full cohort mean={full_mean:.2f}  controversial mean={contro_mean:.2f}  delta={contro_mean - full_mean:+.2f}")

    # Per-quartile rate of controversial frames
    print("\n" + "=" * 90)
    print("Controversy rate per IQ-axis quartile (P8A-aligned bins)")
    print("=" * 90)
    for axis in iq_axes:
        finite = common[axis].dropna()
        q_edges = np.quantile(finite, [0, 0.25, 0.5, 0.75, 1.0])
        common[f"{axis}_bin"] = pd.cut(common[axis], bins=q_edges, labels=["Q1","Q2","Q3","Q4"], include_lowest=True)
        rate = common.groupby(f"{axis}_bin", observed=True)["score_range"].apply(lambda s: (s > 0.7).mean())
        print(f"\n  {axis}:")
        for binlabel in ["Q1","Q2","Q3","Q4"]:
            r = rate.get(binlabel, float("nan"))
            print(f"    {binlabel}: {r*100:.1f}% controversial")

    # Specifically: pairwise correlation of scores
    print("\n" + "=" * 90)
    print("Pairwise score correlation (Pearson) across all real frames")
    print("=" * 90)
    corr = common[["P8A", "E2B", "T3_S1_step1500", "T3_S1_step2500"]].corr()
    print(corr.round(3))

    # Save
    common.to_csv(OUT / "job_d_per_frame_disagreement.csv", index=False)
    controversial.to_csv(OUT / "job_d_controversial_frames.csv", index=False)
    print(f"\nWrote {OUT / 'job_d_per_frame_disagreement.csv'} and job_d_controversial_frames.csv")


if __name__ == "__main__":
    main()
