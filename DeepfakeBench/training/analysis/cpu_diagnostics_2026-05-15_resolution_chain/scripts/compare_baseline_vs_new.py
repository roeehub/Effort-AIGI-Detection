"""Side-by-side comparison of resolution-chain stability across all 5 ckpts.

Reads:
  outputs/per_frame_per_variant.parquet (P8A, E2B, T5C from baseline probe)
  outputs/per_frame_summary.parquet
  outputs_new_ckpts/per_frame_per_variant_new_ckpts.parquet (SLOT_A, SLOT_B)
  outputs_new_ckpts/per_frame_summary_new_ckpts.parquet

Writes:
  outputs_new_ckpts/comparison_5ckpts.csv
  outputs_new_ckpts/per_cohort_range_5ckpts.csv
  outputs_new_ckpts/flip_rate_5ckpts.csv
  outputs_new_ckpts/per_size_real_score_5ckpts.csv
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR.parent / "outputs_new_ckpts"
BASE_DIR = THIS_DIR.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

base_full = pd.read_parquet(BASE_DIR / "per_frame_per_variant.parquet")
base_sum = pd.read_parquet(BASE_DIR / "per_frame_summary.parquet")
new_full = pd.read_parquet(OUT_DIR / "per_frame_per_variant_new_ckpts.parquet")
new_sum = pd.read_parquet(OUT_DIR / "per_frame_summary_new_ckpts.parquet")

# Concat into 5-ckpt view
full = pd.concat([base_full, new_full], ignore_index=True)
summary = pd.concat([base_sum, new_sum], ignore_index=True)

print("=== Loaded 5-ckpt grid ===")
print(f"  full   : {len(full)} rows ({full['ckpt'].nunique()} ckpts)")
print(f"  summary: {len(summary)} rows")
print(f"  ckpts: {sorted(summary['ckpt'].unique())}")

# 1. Per-ckpt per-cohort score range
print("\n=== 1. Median score_range per (ckpt × cohort) ===")
agg = summary.groupby(["ckpt", "cohort"]).agg(
    n=("score_range", "size"),
    range_p50=("score_range", lambda s: s.quantile(0.5)),
    range_p90=("score_range", lambda s: s.quantile(0.9)),
).reset_index()
agg.to_csv(OUT_DIR / "per_cohort_range_5ckpts.csv", index=False)

# Print a pivot table for legibility
pivot = agg.pivot(index="cohort", columns="ckpt", values="range_p50")
# Sort cols for readability
col_order = ["P8A_step5000", "E2B_step3200", "T5C_step3500", "SLOT_A_RESCHAIN", "SLOT_B_6AXIS_GRL"]
cols_present = [c for c in col_order if c in pivot.columns]
print(pivot[cols_present].to_string())

# 2. Flip rate at τ thresholds
print("\n=== 2. Flip rate per ckpt (across τ thresholds) ===")
flip_rows = []
for c in summary["ckpt"].unique():
    s = summary[summary["ckpt"] == c]
    flips = {
        "ckpt": c,
        "n": len(s),
        "flip_0.5_pct": float(((s["score_baseline"] < 0.5) & (s["score_max"] >= 0.5) |
                               (s["score_baseline"] >= 0.5) & (s["score_min"] < 0.5)).mean()),
        "flip_0.7_pct": float(((s["score_baseline"] < 0.7) & (s["score_max"] >= 0.7) |
                               (s["score_baseline"] >= 0.7) & (s["score_min"] < 0.7)).mean()),
        "flip_0.9_pct": float(((s["score_baseline"] < 0.9) & (s["score_max"] >= 0.9) |
                               (s["score_baseline"] >= 0.9) & (s["score_min"] < 0.9)).mean()),
    }
    flip_rows.append(flips)
flips_df = pd.DataFrame(flip_rows).sort_values("flip_0.5_pct")
flips_df.to_csv(OUT_DIR / "flip_rate_5ckpts.csv", index=False)
print(flips_df.to_string(index=False))

# 3. Per-size mean score on REAL frames
print("\n=== 3. Per-size mean score on REALS (label=0) per ckpt ===")
perturbed = full[full["down_size"] != -1].copy()
reals = perturbed[perturbed["label"] == 0]
size_pivot = reals.groupby(["ckpt", "down_size"])["score"].mean().reset_index()
pivot_size = size_pivot.pivot(index="down_size", columns="ckpt", values="score")
cols_present = [c for c in col_order if c in pivot_size.columns]
pivot_size = pivot_size[cols_present]
print(pivot_size.round(3).to_string())
pivot_size.to_csv(OUT_DIR / "per_size_real_score_5ckpts.csv")

# 4. Mean / median / max per ckpt of score_range on REAL cohorts only
print("\n=== 4. score_range on REAL cohorts only (label=0) per ckpt ===")
real_sum = summary[summary["label"] == 0]
real_agg = real_sum.groupby("ckpt").agg(
    n=("score_range", "size"),
    range_mean=("score_range", "mean"),
    range_p50=("score_range", lambda s: s.quantile(0.5)),
    range_p90=("score_range", lambda s: s.quantile(0.9)),
    range_max=("score_range", "max"),
).reset_index()
real_agg = real_agg.sort_values("range_p50")
real_agg.to_csv(OUT_DIR / "real_range_5ckpts.csv", index=False)
print(real_agg.to_string(index=False))

print("\nDONE — wrote 4 CSVs to outputs_new_ckpts/.")
