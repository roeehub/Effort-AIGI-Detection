"""CPU Job 2 — T5C overfire population audit.

For frames where T5C step3500 > P8A by > 0.5 on REAL cohorts (1318 in contract data):
  - cohort breakdown
  - identity clustering (which chronic / healthy identities concentrate overfires)
  - IQ profile of overfires vs baseline
  - find the minimum reference cohort whose anchor would prevent most overfires

Goal: design the reference cohort for the L11 anchor loss / output distillation loss.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT = REPO / "analysis" / "cpu_diagnostics_2026-05-12_stage_a" / "outputs"

ALL = pd.read_csv(OUT / "unified_frame_matrix.csv")

CHRONIC_6 = ["Roy_D", "PC_Generator", "bla_bla_chow",
             "Md_noyn_Sharker", "dor_shkedi", "healthy_dor"]


def extract_identity(fp):
    """Extract identity prefix from filename."""
    if not isinstance(fp, str):
        return "unknown"
    fname = fp.split("/")[-1]
    # filename format: <identity>__s<session>_<frame>_crop_<crop>__<hash>.jpg
    if "__" in fname:
        return fname.split("__")[0]
    return fname.split("_")[0]


ALL["identity"] = ALL["frame_path"].apply(extract_identity)

# Real frames only
real = ALL[ALL["label"] == 0].copy()
real["delta"] = real["T5C_step3500"] - real["P8A"]
real["overfire_T5C"] = (real["delta"] > 0.5).astype(int)

print("=" * 80)
print("OVERFIRE BREAKDOWN: T5C > P8A by > 0.5 on REAL frames")
print("=" * 80)
print(f"Total real frames: {len(real)}")
print(f"T5C overfire frames: {real['overfire_T5C'].sum()} ({100*real['overfire_T5C'].mean():.2f}%)")

# By suite
print("\nBy suite:")
suite_summary = real.groupby("suite").agg(
    n=("delta", "count"),
    overfires=("overfire_T5C", "sum"),
    pct=("overfire_T5C", "mean"),
    mean_delta=("delta", "mean"),
).round(4)
suite_summary["pct"] *= 100
print(suite_summary.to_string())

# Top 20 identities with overfires
print("\n=== Top 20 identities by absolute overfire count (real-side T5C > P8A by 0.5+) ===")
id_summary = real.groupby("identity").agg(
    n=("delta", "count"),
    overfires=("overfire_T5C", "sum"),
    overfire_rate=("overfire_T5C", "mean"),
).sort_values("overfires", ascending=False).head(20)
id_summary["overfire_rate"] = (id_summary["overfire_rate"] * 100).round(2)
print(id_summary.to_string())

# Top 20 identities by overfire RATE (with n >= 5)
print("\n=== Top 20 identities by overfire RATE (n>=5 frames) ===")
id_rate = real.groupby("identity").agg(
    n=("delta", "count"),
    overfires=("overfire_T5C", "sum"),
    overfire_rate=("overfire_T5C", "mean"),
).query("n >= 5").sort_values("overfire_rate", ascending=False).head(20)
id_rate["overfire_rate"] = (id_rate["overfire_rate"] * 100).round(2)
print(id_rate.to_string())

# Concentration: how many identities contribute X% of overfires?
print("\n=== Cumulative overfire coverage by identity rank ===")
sorted_ids = real.groupby("identity")["overfire_T5C"].sum().sort_values(ascending=False)
sorted_ids = sorted_ids[sorted_ids > 0]
cumsum = sorted_ids.cumsum()
total = cumsum.iloc[-1]
print(f"Total overfires: {total}")
print(f"Top 5 identities cover: {cumsum.iloc[4] if len(cumsum) >= 5 else cumsum.iloc[-1]} ({100*cumsum.iloc[4]/total if len(cumsum) >= 5 else 100:.1f}%)")
print(f"Top 10 identities cover: {cumsum.iloc[9] if len(cumsum) >= 10 else cumsum.iloc[-1]} ({100*cumsum.iloc[9]/total if len(cumsum) >= 10 else 100:.1f}%)")
print(f"Top 20 identities cover: {cumsum.iloc[19] if len(cumsum) >= 20 else cumsum.iloc[-1]} ({100*cumsum.iloc[19]/total if len(cumsum) >= 20 else 100:.1f}%)")
n_50pct = (cumsum < total * 0.5).sum() + 1
n_80pct = (cumsum < total * 0.8).sum() + 1
n_95pct = (cumsum < total * 0.95).sum() + 1
print(f"Identities needed to cover 50% of overfires: {n_50pct}")
print(f"Identities needed to cover 80% of overfires: {n_80pct}")
print(f"Identities needed to cover 95% of overfires: {n_95pct}")

# IQ profile
print("\n=== IQ profile: overfires vs baseline reals ===")
for axis in ["min_dim", "lap_var", "color_a_dev", "saturation_mean", "luma_mean"]:
    if axis not in real.columns:
        continue
    of = real[real["overfire_T5C"] == 1][axis].dropna()
    no_of = real[real["overfire_T5C"] == 0][axis].dropna()
    if len(of) == 0 or len(no_of) == 0:
        continue
    print(f"  {axis:<20} overfire mean={of.mean():>9.2f}  median={of.median():>9.2f}  | "
          f"non-overfire mean={no_of.mean():>9.2f}  median={no_of.median():>9.2f}")

# Is T5C's overfire population a chronic-6 effect or broader?
print("\n=== Chronic_6 contribution to overfires ===")
chronic_re = "|".join(CHRONIC_6)
real["is_chronic_6"] = real["frame_path"].str.contains(chronic_re, case=False, regex=True).astype(int)
print(f"Chronic_6 share of overfires: {real[real['overfire_T5C']==1]['is_chronic_6'].sum()} / "
      f"{real['overfire_T5C'].sum()} ({100*real[real['overfire_T5C']==1]['is_chronic_6'].mean():.1f}%)")
print(f"Chronic_6 share of all reals: {real['is_chronic_6'].sum()} / {len(real)} "
      f"({100*real['is_chronic_6'].mean():.1f}%)")

# Specifically what is the score distribution on chronic-6 vs healthy reals?
print("\n=== Score distribution: chronic_6 reals vs healthy reals ===")
chr_reals = real[real["is_chronic_6"] == 1]
hlth_reals = real[real["is_chronic_6"] == 0]
print(f"Chronic_6 reals (n={len(chr_reals)}):")
print(f"  P8A:           mean={chr_reals['P8A'].mean():.4f}  p50={chr_reals['P8A'].median():.4f}  "
      f"p90={chr_reals['P8A'].quantile(0.9):.4f}")
print(f"  T5C_step3500:  mean={chr_reals['T5C_step3500'].mean():.4f}  p50={chr_reals['T5C_step3500'].median():.4f}  "
      f"p90={chr_reals['T5C_step3500'].quantile(0.9):.4f}")
print(f"Healthy reals (n={len(hlth_reals)}):")
print(f"  P8A:           mean={hlth_reals['P8A'].mean():.4f}  p50={hlth_reals['P8A'].median():.4f}  "
      f"p90={hlth_reals['P8A'].quantile(0.9):.4f}")
print(f"  T5C_step3500:  mean={hlth_reals['T5C_step3500'].mean():.4f}  p50={hlth_reals['T5C_step3500'].median():.4f}  "
      f"p90={hlth_reals['T5C_step3500'].quantile(0.9):.4f}")

# Output the overfire frames + features
real.to_csv(OUT / "real_with_overfire_flag.csv", index=False)
print(f"\nWrote real_with_overfire_flag.csv ({len(real)} rows)")
