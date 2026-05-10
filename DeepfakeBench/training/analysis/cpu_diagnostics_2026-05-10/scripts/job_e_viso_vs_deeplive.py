"""Job E — Why is viso harder than deeplive across all our ckpts?

Across the 4 candidate ckpts {P8A, E2B, step1500, step2500}, deeplive recall
saturates near 100% on F4 while viso plateaus 67-79%. This script asks:
  - Are missed-viso frames characterized by specific IQ axes?
  - Are they identity-clustered or method-clustered?
  - Are they systematically different from caught-viso frames?
  - Are they systematically different from caught/missed deeplive frames?

If viso missed frames cluster on IQ axes the model has weakened (luma, color),
T4 face_scale_jitter wouldn't help. If they cluster on identity/method, a
data lever might. If they cluster on lower-IQ (gate-shielded), the gate
solves it.

Inputs (per ckpt):
  - F4 input frame reports for visomaster_enhanced_macro_dev + deeplive_enhanced_dev
  - IQ atlas for axis profiles

Outputs:
  - outputs/job_e_viso_vs_deeplive_summary.csv
  - outputs/job_e_viso_missed_iq_profile.csv
  - outputs/job_e_viso_PROFILE.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT = ROOT / "analysis/cpu_diagnostics_2026-05-10/outputs"
ATLAS = ROOT / "analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet"

REPORTS = {
    "P8A": {
        "visomaster_enhanced_macro_dev": ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/visomaster_enhanced_macro_dev_p8a_reference_step5000_frames_report.csv",
        "deeplive_enhanced_dev": ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/deeplive_enhanced_dev_p8a_reference_step5000_frames_report.csv",
    },
    "E2B": {
        "visomaster_enhanced_macro_dev": ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/visomaster_enhanced_macro_dev_e2b_top_n_step3200_frames_report.csv",
        "deeplive_enhanced_dev": ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/deeplive_enhanced_dev_e2b_top_n_step3200_frames_report.csv",
    },
    "T3_S1_step1500": {
        "visomaster_enhanced_macro_dev": ROOT / "analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step1500/visomaster_enhanced_macro_dev_t3_slot1_periodic_step1500_frames_report.csv",
        "deeplive_enhanced_dev": ROOT / "analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step1500/deeplive_enhanced_dev_t3_slot1_periodic_step1500_frames_report.csv",
    },
    "T3_S1_step2500": {
        "visomaster_enhanced_macro_dev": ROOT / "analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step2500/visomaster_enhanced_macro_dev_t3_slot1_periodic_step2500_frames_report.csv",
        "deeplive_enhanced_dev": ROOT / "analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step2500/deeplive_enhanced_dev_t3_slot1_periodic_step2500_frames_report.csv",
    },
}

IQ_AXES = ["lap_var", "min_dim", "color_a_dev", "color_b_dev", "luma_mean", "saturation_mean"]
TAU_FOR_MISS = 0.5  # frames scoring below this are "missed fakes"


def main():
    print("Loading IQ atlas...")
    atlas = pd.read_parquet(ATLAS)
    print(f"  atlas: {len(atlas)} rows")

    summary_rows = []
    iq_rows = []
    md_lines = ["# Job E — Why viso is harder than deeplive across ckpts", "",
                "For each ckpt, characterize the score distribution + IQ profile of MISSED fake frames (score < 0.5) vs CAUGHT fake frames on viso vs deeplive.", ""]

    for ckpt_label, suites in REPORTS.items():
        print(f"\n=== {ckpt_label} ===")
        md_lines.append(f"## {ckpt_label}\n")
        for suite, report_path in suites.items():
            if not report_path.exists():
                print(f"  SKIP {suite}: {report_path} not found")
                continue
            df = pd.read_csv(report_path)
            n = len(df)
            # Join with IQ atlas
            joined = df.merge(atlas[["frame_path"] + IQ_AXES], on="frame_path", how="left")
            n_join = joined[IQ_AXES[0]].notna().sum()
            join_pct = 100 * n_join / n if n > 0 else 0

            # Caught vs missed at TAU_FOR_MISS
            caught = df[df["frame_prob"] >= TAU_FOR_MISS]
            missed = df[df["frame_prob"] < TAU_FOR_MISS]
            n_caught = len(caught)
            n_missed = len(missed)
            recall = n_caught / n if n > 0 else 0

            print(f"  {suite}: n={n}, recall@0.5={100*recall:.1f}%, missed={n_missed}, IQ join={join_pct:.0f}%")

            row = {"ckpt": ckpt_label, "suite": suite, "n": n,
                   "n_caught": n_caught, "n_missed": n_missed,
                   "recall_at_0.5": recall,
                   "score_p50": float(df["frame_prob"].median()),
                   "score_mean": float(df["frame_prob"].mean()),
                   "score_p10": float(df["frame_prob"].quantile(0.10)),
                   "score_p90": float(df["frame_prob"].quantile(0.90)),
                   "iq_join_pct": join_pct,
                  }
            # IQ profile of MISSED frames (those that should have been caught)
            if n_join > 0 and n_missed > 0:
                missed_joined = joined[joined["frame_prob"] < TAU_FOR_MISS]
                caught_joined = joined[joined["frame_prob"] >= TAU_FOR_MISS]
                for c in IQ_AXES:
                    if c in missed_joined.columns:
                        row[f"missed_{c}_p50"] = float(missed_joined[c].median()) if missed_joined[c].notna().any() else None
                        row[f"caught_{c}_p50"] = float(caught_joined[c].median()) if caught_joined[c].notna().any() else None
            summary_rows.append(row)

            # Per-method (group_key) breakdown for viso suite
            if "visomaster" in suite:
                if "group_key" in df.columns:
                    method_breakdown = df.groupby("group_key").agg(
                        n=("frame_prob", "count"),
                        miss_rate=("frame_prob", lambda x: (x < TAU_FOR_MISS).mean()),
                        score_p50=("frame_prob", "median"),
                    ).sort_values("miss_rate", ascending=False)
                    md_lines.append(f"### {suite} — per-method miss rate at τ=0.5")
                    md_lines.append("")
                    md_lines.append("| group_key | n | miss_rate | score_p50 |")
                    md_lines.append("|---|---:|---:|---:|")
                    for gk, r in method_breakdown.iterrows():
                        md_lines.append(f"| {gk} | {int(r['n'])} | {100*r['miss_rate']:.0f}% | {r['score_p50']:.3f} |")
                    md_lines.append("")

        # Compute viso vs deeplive recall delta and IQ delta
        md_lines.append(f"### {ckpt_label} — viso miss profile vs deeplive miss profile\n")
        md_lines.append("| axis | viso_caught_p50 | viso_missed_p50 | deeplive_caught_p50 | deeplive_missed_p50 |")
        md_lines.append("|---|---:|---:|---:|---:|")
        viso_row = next((r for r in summary_rows if r["ckpt"] == ckpt_label and "visomaster" in r["suite"]), None)
        dl_row = next((r for r in summary_rows if r["ckpt"] == ckpt_label and "deeplive" in r["suite"]), None)
        if viso_row and dl_row:
            for c in IQ_AXES:
                vc = viso_row.get(f"caught_{c}_p50", None)
                vm = viso_row.get(f"missed_{c}_p50", None)
                dc = dl_row.get(f"caught_{c}_p50", None)
                dm = dl_row.get(f"missed_{c}_p50", None)
                fmt = lambda x: f"{x:.1f}" if x is not None else "-"
                md_lines.append(f"| {c} | {fmt(vc)} | {fmt(vm)} | {fmt(dc)} | {fmt(dm)} |")
        md_lines.append("")

    pd.DataFrame(summary_rows).to_csv(OUT / "job_e_viso_vs_deeplive_summary.csv", index=False)
    (OUT / "job_e_viso_PROFILE.md").write_text("\n".join(md_lines))
    print(f"\nWrote summary + markdown")

if __name__ == "__main__":
    main()
