#!/usr/bin/env python3
"""
After PA+PC frame reports land, compute IQ-signature characterization:
- Pearson r(score, laplacian_var) per ckpt on viso fakes
- Cohort overlap with prior P8A/E2B/E3 cohort assignments
- Score median per cohort

Run after `run_full_eval.py`. Reads frame_report CSVs from
analysis/pa_pc_eval_2026-05-05/raw_reports/ and merges with the
existing per-frame IQ data from analysis/p8a_signature_decomposition_2026-05-05/.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
RAW = ROOT / "analysis/pa_pc_eval_2026-05-05/raw_reports"
COHORT_CSV = ROOT / "analysis/p8a_signature_decomposition_2026-05-05/viso_cohort_assignments.csv"
OUT = ROOT / "analysis/pa_pc_eval_2026-05-05"

# We compute on the 4 PA + PC ckpts; baselines are already characterized.
TARGETS = [
    ("P8A_REFERENCE_STEP5000", "p8a_reference_step5000"),
    ("E2B_TOP_N_STEP3200", "e2b_top_n_step3200"),
    ("PA_TOP_N_STEP5600", "pa_top_n_step5600"),
    ("PA_TOP_N_STEP3800", "pa_top_n_step3800"),
    ("PA_PERIODIC_STEP5000", "pa_periodic_step5000"),
    ("PC_TOP_N_STEP7400", "pc_top_n_step7400"),
    ("PC_TOP_N_STEP5400", "pc_top_n_step5400"),
    ("PC_PERIODIC_STEP5000", "pc_periodic_step5000"),
]


def main():
    # Load existing cohort assignments (per-frame IQ + 3 ckpt scores + cohort label)
    cohort = pd.read_csv(COHORT_CSV)
    print(f"Cohort source: {len(cohort)} frames, columns={list(cohort.columns)[:10]}...")

    # For each target ckpt, load its viso_enhanced_macro_dev frame_report
    out_rows = []
    cohort_score_table_rows = []
    for ckpt_key, ckpt_low in TARGETS:
        viso_path = RAW / f"visomaster_enhanced_macro_dev_{ckpt_low}_frames_report.csv"
        if not viso_path.exists():
            print(f"[SKIP] {viso_path} not found")
            continue
        df = pd.read_csv(viso_path)
        # Filter to fakes only
        if "label" in df.columns:
            df = df[df["label"] == 1]
        merged = df.merge(cohort[["frame_path", "laplacian_var", "luma_mean", "sobel_edge_mean", "cohort"]], on="frame_path", how="inner")
        n = len(merged)
        if n < 10:
            print(f"[SKIP] {ckpt_key}: merge produced n={n} < 10 frames")
            continue

        r_lap, p_lap = pearsonr(merged["frame_prob"], merged["laplacian_var"])
        r_luma, p_luma = pearsonr(merged["frame_prob"], merged["luma_mean"])
        r_sob, p_sob = pearsonr(merged["frame_prob"], merged["sobel_edge_mean"])
        out_rows.append({
            "ckpt": ckpt_key,
            "n": n,
            "r_score_lap": r_lap, "p_lap": p_lap,
            "r_score_luma": r_luma, "p_luma": p_luma,
            "r_score_sobel": r_sob, "p_sobel": p_sob,
        })
        # Per-cohort score median
        for cohort_name, group in merged.groupby("cohort"):
            cohort_score_table_rows.append({
                "ckpt": ckpt_key,
                "cohort": cohort_name,
                "n": len(group),
                "score_median": float(group["frame_prob"].median()),
                "score_mean": float(group["frame_prob"].mean()),
                "lap_p50": float(group["laplacian_var"].median()),
            })

        print(f"{ckpt_key:30s}  r(score, lap) = {r_lap:+.3f}  r(score, luma) = {r_luma:+.3f}  r(score, sobel) = {r_sob:+.3f}  (n={n})")

    OUT.mkdir(parents=True, exist_ok=True)
    if out_rows:
        with open(OUT / "iq_signatures.json", "w") as f:
            json.dump(out_rows, f, indent=2)
        # Pretty CSV
        with open(OUT / "iq_signatures.csv", "w") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            for r in out_rows:
                w.writerow(r)
        print(f"\nWrote {OUT/'iq_signatures.json'} + .csv ({len(out_rows)} ckpts)")

    if cohort_score_table_rows:
        df_out = pd.DataFrame(cohort_score_table_rows)
        df_out.to_csv(OUT / "per_cohort_score.csv", index=False)
        # Pivot: ckpt × cohort
        pv = df_out.pivot_table(index="cohort", columns="ckpt", values="score_median")
        pv.to_csv(OUT / "per_cohort_score_pivot.csv")
        print(f"Wrote {OUT/'per_cohort_score.csv'} + per_cohort_score_pivot.csv")
        print("\n=== Per-cohort score median pivot ===")
        print(pv.to_string())


if __name__ == "__main__":
    main()
