"""Decompose remediation effects by IQ axes + identity.

Question: is there a sub-population (e.g., low-laplacian reals) where
remediation gives a meaningful real-side drop without commensurate fake drop?

If yes, the deploy lever is: detect risky frame -> apply remediation only.
If no, the model's chronic-FP behavior is not pixel-domain-fixable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

THIS_DIR = Path(__file__).resolve().parent
OUT = THIS_DIR / "outputs"

df = pd.read_csv(OUT / "scores_per_frame.csv")
print(f"loaded {len(df)} frames")
print(f"pool counts: {df['pool'].value_counts().to_dict()}")
print()

CONDITIONS = ["orig", "wb_grayworld", "unsharp_05", "gamma_norm"]

# ============================================================================
# 1) Score response decomposed by IQ quartile (laplacian variance)
# ============================================================================
print("=" * 70)
print("1) Δscore by laplacian-variance quartile (risky_real only)")
print("=" * 70)
risky = df[df["pool"] == "risky_real"].copy()
risky["lap_q"] = pd.qcut(risky["iq_lap_var"], 4, labels=["Q1_blur", "Q2", "Q3", "Q4_sharp"])
for cond in CONDITIONS[1:]:
    print(f"\nCondition: {cond}")
    g = risky.groupby("lap_q").apply(
        lambda d: pd.Series({
            "n": len(d),
            "lap_var_med": d["iq_lap_var"].median(),
            "orig_mean": d["score_orig"].mean(),
            f"{cond}_mean": d[f"score_{cond}"].mean(),
            "delta_mean": (d[f"score_{cond}"] - d["score_orig"]).mean(),
            "delta_median": (d[f"score_{cond}"] - d["score_orig"]).median(),
            "flipped_below_50": ((d[f"score_{cond}"] < 0.5) & (d["score_orig"] >= 0.5)).sum(),
        }),
        
    )
    print(g.to_string())

# ============================================================================
# 2) Same for fakes — does the same quartile respond differently?
# ============================================================================
print()
print("=" * 70)
print("2) Δscore by laplacian quartile (tp_fake only)")
print("=" * 70)
fakes = df[df["pool"] == "tp_fake"].copy()
fakes["lap_q"] = pd.qcut(fakes["iq_lap_var"], 4, labels=["Q1_blur", "Q2", "Q3", "Q4_sharp"])
for cond in CONDITIONS[1:]:
    print(f"\nCondition: {cond}")
    g = fakes.groupby("lap_q").apply(
        lambda d: pd.Series({
            "n": len(d),
            "lap_var_med": d["iq_lap_var"].median(),
            "orig_mean": d["score_orig"].mean(),
            f"{cond}_mean": d[f"score_{cond}"].mean(),
            "delta_mean": (d[f"score_{cond}"] - d["score_orig"]).mean(),
            "delta_median": (d[f"score_{cond}"] - d["score_orig"]).median(),
            "flipped_below_50": ((d[f"score_{cond}"] < 0.5) & (d["score_orig"] >= 0.5)).sum(),
        }),
        
    )
    print(g.to_string())

# ============================================================================
# 3) Per-identity decomposition (risky reals) — find responsive subgroups
# ============================================================================
print()
print("=" * 70)
print("3) Per-base_identity Δscore (risky reals, all conditions)")
print("=" * 70)
for cond in CONDITIONS[1:]:
    print(f"\nCondition: {cond}")
    g = risky.groupby("base_identity").apply(
        lambda d: pd.Series({
            "n": len(d),
            "orig_mean": d["score_orig"].mean(),
            f"{cond}_mean": d[f"score_{cond}"].mean(),
            "delta_mean": (d[f"score_{cond}"] - d["score_orig"]).mean(),
            "flipped_below_50": int(((d[f"score_{cond}"] < 0.5) & (d["score_orig"] >= 0.5)).sum()),
        }),
        
    )
    print(g.sort_values("delta_mean").to_string())

# ============================================================================
# 4) CONDITIONAL REMEDIATION — apply best transform only when flagged
#    Detector: laplacian-variance below median (= "this looks blurry/webcam")
# ============================================================================
print()
print("=" * 70)
print("4) Conditional remediation: apply only when lap_var < median")
print("=" * 70)
lap_threshold = df["iq_lap_var"].median()
print(f"  lap_var detection threshold (median): {lap_threshold:.2f}")
df["is_risky_lap"] = df["iq_lap_var"] < lap_threshold
print(f"  flagged frames: {df['is_risky_lap'].sum()} / {len(df)}")

# Build conditional scores: use remediated if flagged, else orig
for cond in CONDITIONS[1:]:
    df[f"score_cond_{cond}"] = np.where(df["is_risky_lap"], df[f"score_{cond}"], df["score_orig"])

# Compute AUC for original and each conditional remediation (real vs fake — mixed pool)
mixed = df[df["pool"].isin(["risky_real", "clean_real", "tp_fake"])]
print(f"\n  mixed-pool AUC (n={len(mixed)}):")
print(f"  {'condition':<24} {'AUC':<10}")
auc_orig = roc_auc_score(mixed["label"], mixed["score_orig"])
print(f"  {'orig':<24} {auc_orig:<10.4f}")
for cond in CONDITIONS[1:]:
    auc_universal = roc_auc_score(mixed["label"], mixed[f"score_{cond}"])
    auc_cond = roc_auc_score(mixed["label"], mixed[f"score_cond_{cond}"])
    print(f"  {'universal_'+cond:<24} {auc_universal:<10.4f}  Δ={auc_universal-auc_orig:+.4f}")
    print(f"  {'conditional_'+cond:<24} {auc_cond:<10.4f}  Δ={auc_cond-auc_orig:+.4f}")

# And the "hard" subpool (risky + tp_fake only — high-confidence pre-existing errors/correct)
hard = df[df["pool"].isin(["risky_real", "tp_fake"])]
print(f"\n  hard-pool AUC (risky_real + tp_fake, n={len(hard)}):")
auc_orig_h = roc_auc_score(hard["label"], hard["score_orig"])
print(f"  {'orig':<24} {auc_orig_h:<10.4f}")
for cond in CONDITIONS[1:]:
    auc_uni = roc_auc_score(hard["label"], hard[f"score_{cond}"])
    auc_cnd = roc_auc_score(hard["label"], hard[f"score_cond_{cond}"])
    print(f"  {'universal_'+cond:<24} {auc_uni:<10.4f}  Δ={auc_uni-auc_orig_h:+.4f}")
    print(f"  {'conditional_'+cond:<24} {auc_cnd:<10.4f}  Δ={auc_cnd-auc_orig_h:+.4f}")

# ============================================================================
# 5) Best-case: oracle remediation — pick best per-frame
# ============================================================================
print()
print("=" * 70)
print("5) Oracle remediation (pick the lowest-fake / highest-real-correctness condition)")
print("=" * 70)
# For each frame, the "oracle" picks the condition that MOVES toward the correct label
df["oracle_score"] = df.apply(
    lambda r: min(r["score_orig"], r["score_wb_grayworld"], r["score_unsharp_05"], r["score_gamma_norm"]) if r["label"] == 0
    else max(r["score_orig"], r["score_wb_grayworld"], r["score_unsharp_05"], r["score_gamma_norm"]),
    axis=1,
)
auc_oracle = roc_auc_score(df.dropna(subset=["oracle_score"])["label"],
                            df.dropna(subset=["oracle_score"])["oracle_score"])
print(f"  oracle AUC (mixed pool, all 260 frames): {auc_oracle:.4f}")
auc_orig_all = roc_auc_score(df["label"], df["score_orig"])
print(f"  orig AUC: {auc_orig_all:.4f}  (Δ = {auc_oracle-auc_orig_all:+.4f})")
print(f"  -> upper bound on how much remediation could help if we had a perfect")
print(f"     per-frame chooser")

# Per-frame "which condition won" for the oracle on risky reals
risky_oracle = df[df["pool"] == "risky_real"].copy()
def best_cond_for_real(r):
    scs = {"orig": r["score_orig"], "wb": r["score_wb_grayworld"],
           "unsharp": r["score_unsharp_05"], "gamma": r["score_gamma_norm"]}
    return min(scs, key=scs.get)
risky_oracle["best_cond"] = risky_oracle.apply(best_cond_for_real, axis=1)
print(f"\n  risky-real: best condition per frame (where oracle picks the lowest score):")
print(risky_oracle["best_cond"].value_counts().to_string())

# Same for tp_fakes — which condition KEEPS them highest (= confirms fake)
fake_oracle = df[df["pool"] == "tp_fake"].copy()
def best_cond_for_fake(r):
    scs = {"orig": r["score_orig"], "wb": r["score_wb_grayworld"],
           "unsharp": r["score_unsharp_05"], "gamma": r["score_gamma_norm"]}
    return max(scs, key=scs.get)
fake_oracle["best_cond"] = fake_oracle.apply(best_cond_for_fake, axis=1)
print(f"\n  tp_fake: best condition per frame (where oracle picks the highest score):")
print(fake_oracle["best_cond"].value_counts().to_string())
