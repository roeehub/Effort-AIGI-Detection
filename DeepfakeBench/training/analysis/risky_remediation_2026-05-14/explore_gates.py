"""Explore (A) fine-grained G2 threshold and (B) other candidate gates.

A. Sweep G2 at 10-px resolution from 80 to 260.
B. For each cheap feature (lap_var, luma, lab_a/b_dev, edge_density),
   sweep quantile thresholds and check if adding it as a gate (on top of
   G2-current) improves any production metric.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

OUT = Path(__file__).resolve().parent / "outputs"
df = pd.read_csv(OUT / "all_cohorts_scored.csv")

# Compute min_dim from local images (cached previously in g2_pass_pool; redo here for completeness)
import cv2
import time

print("computing min(W,H)...")
t0 = time.time()
min_dims = []
for i, p in enumerate(df["local"]):
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        min_dims.append(0)
    else:
        h, w = img.shape[:2]
        min_dims.append(min(h, w))
df["min_dim"] = min_dims
print(f"done in {time.time()-t0:.0f}s")

POOL_DEFS = {
    "teams_dev": ["teams_real_all_dev", "teams_fake_all_dev"],
    "teams_lockbox": ["teams_real_all_lockbox", "teams_fake_all_lockbox"],
    "dor_cross": ["dor_evening", "dor_morning", "dor_fake_local", "visomaster_v2_dor"],
}
def assign_pool(s):
    for p, ss in POOL_DEFS.items():
        if s in ss:
            return p
    return "other"
df["pool"] = df["suite"].map(assign_pool)


def recall_at_fpr(scores, labels, target=0.05):
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = fpr <= target
    if ok.sum() == 0:
        return float("nan")
    return float(tpr[np.where(ok)[0][-1]])


def identity_verdict_score(df_sub, tau=0.49, maj=0.50):
    """Per-identity correctness: how many identities (with >= 5 frames)
    are correctly classified by majority vote?"""
    correct = 0
    total = 0
    for ident, g in df_sub.groupby("base_identity"):
        if len(g) < 5 or g["label"].nunique() != 1:
            continue
        label = int(g["label"].iloc[0])
        frac = float((g["T5C_orig"] > tau).mean())
        verdict = frac > maj
        truth = (label == 1)
        if verdict == truth:
            correct += 1
        total += 1
    return correct, total


# ============================================================================
# A) Fine-grained G2 sweep
# ============================================================================
print()
print("=" * 100)
print("A) Fine G2 sweep — 10-px resolution from 80 to 260")
print("=" * 100)
THRESHOLDS = list(range(80, 261, 10))
rows = []
for pool_name, suites in POOL_DEFS.items():
    pool_df = df[df["suite"].isin(suites)].copy()
    print(f"\n### {pool_name}")
    print(f"  {'G2':<5} {'pass_n':<8} {'pass%':<7} {'AUC':<8} {'recall@5':<10} "
          f"{'recall@1':<10} {'FP@τ=.49':<10} {'TP@τ=.49':<10} {'id_correct':<14}")
    for th in THRESHOLDS:
        sub = pool_df[pool_df["min_dim"] >= th]
        if sub["label"].nunique() < 2 or len(sub) < 30:
            print(f"  {th:<5} {len(sub):<8} insufficient")
            continue
        labels = sub["label"].values
        scores = sub["T5C_orig"].values
        auc = roc_auc_score(labels, scores)
        rec5 = recall_at_fpr(scores, labels, 0.05)
        rec1 = recall_at_fpr(scores, labels, 0.01)
        fp = ((labels == 0) & (scores > 0.49)).sum() / max((labels == 0).sum(), 1)
        tp = ((labels == 1) & (scores > 0.49)).sum() / max((labels == 1).sum(), 1)
        correct, total = identity_verdict_score(sub)
        ic = f"{correct}/{total}"
        print(f"  {th:<5} {len(sub):<8} {len(sub)/len(pool_df)*100:<6.1f}% "
              f"{auc:<8.4f} {rec5:<10.4f} {rec1:<10.4f} {fp:<10.4f} {tp:<10.4f} {ic:<14}")
        rows.append({
            "pool": pool_name, "g2": th, "pass_n": len(sub),
            "frac_pass": len(sub)/len(pool_df),
            "AUC": auc, "recall@5": rec5, "recall@1": rec1,
            "FP_rate_tau49": fp, "TP_rate_tau49": tp,
            "identity_correct": correct, "identity_total": total,
        })
g2_sweep = pd.DataFrame(rows)
g2_sweep.to_csv(OUT / "g2_fine_sweep.csv", index=False)

# Find the "ideal" G2 per pool — maximizing identity-correct rate then AUC
print()
print("=" * 100)
print("A) IDEAL G2 — by identity-correctness, then AUC tie-breaker")
print("=" * 100)
for pool_name in POOL_DEFS:
    sub = g2_sweep[g2_sweep["pool"] == pool_name].copy()
    if len(sub) == 0:
        continue
    sub["correct_rate"] = sub["identity_correct"] / sub["identity_total"].replace(0, 1)
    sub = sub.sort_values(["correct_rate", "AUC"], ascending=False)
    top = sub.head(3)
    print(f"\n  {pool_name}:")
    for _, r in top.iterrows():
        print(f"    G2={r['g2']:<5} id_correct={r['identity_correct']}/{r['identity_total']} "
              f"({r['correct_rate']:.4f}) AUC={r['AUC']:.4f} recall@5={r['recall@5']:.4f} "
              f"pass={r['frac_pass']:.4f}")

# Combined (any-pool union): identity correctness summed
print()
print("  Combined across all 3 pools (sum identity-correct, sum identity-total):")
combo = g2_sweep.groupby("g2").agg(
    correct_sum=("identity_correct", "sum"),
    total_sum=("identity_total", "sum"),
    auc_mean=("AUC", "mean"),
    pass_mean=("frac_pass", "mean"),
).reset_index()
combo["correct_rate"] = combo["correct_sum"] / combo["total_sum"]
combo = combo.sort_values(["correct_rate", "auc_mean"], ascending=False)
print(combo.head(5).to_string(index=False))

# ============================================================================
# B) Candidate gate exploration — other cheap features
# ============================================================================
print()
print("=" * 100)
print("B) Other candidate gates — test each cheap feature as a gate axis")
print("=" * 100)
print()
print("For each feature, sweep quantile thresholds (drop bottom X% or top X%);")
print("for each, compute T5C performance on KEPT pool and rate of correct id-verdicts.")
print()

FEATURES = {
    "iq_lap_var": "drop_below_qth",        # drop low-sharpness (typical "G3")
    "iq_luma": "drop_outside_quartiles",   # drop extreme dark or bright
    "iq_lab_a_dev": "drop_above_qth",      # drop high color cast
    "iq_lab_b_dev": "drop_above_qth",      # drop high color cast
    "iq_edge_density": "drop_below_qth",   # drop featureless faces
}

QUANTILES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# Per-pool baseline at G2(150) (the new ship spec)
print("Baseline: G2(150) only, no other gate")
print()
baseline = {}
for pool_name, suites in POOL_DEFS.items():
    pool_df = df[df["suite"].isin(suites)].copy()
    sub = pool_df[pool_df["min_dim"] >= 150]
    if sub["label"].nunique() < 2:
        continue
    labels = sub["label"].values
    scores = sub["T5C_orig"].values
    auc = roc_auc_score(labels, scores)
    rec5 = recall_at_fpr(scores, labels, 0.05)
    correct, total = identity_verdict_score(sub)
    baseline[pool_name] = {"n": len(sub), "AUC": auc, "recall@5": rec5,
                            "id_correct": correct, "id_total": total}
    print(f"  {pool_name:<14} n={len(sub):<5} AUC={auc:.4f} recall@5={rec5:.4f} id={correct}/{total}")

# For each candidate gate, sweep thresholds
print()
all_gate_results = []
for feat, direction in FEATURES.items():
    print(f"\n--- Feature: {feat} (direction: {direction}) ---")
    for q in QUANTILES:
        # Compute threshold from the FULL pool's distribution (not per-pool)
        if direction == "drop_below_qth":
            th_low = df[feat].quantile(q)
            gate_pass = df[feat] >= th_low
            descr = f"keep {feat} >= {th_low:.2f} (top {100*(1-q):.0f}%)"
        elif direction == "drop_above_qth":
            th_high = df[feat].quantile(1 - q)
            gate_pass = df[feat] <= th_high
            descr = f"keep {feat} <= {th_high:.2f} (bottom {100*(1-q):.0f}%)"
        elif direction == "drop_outside_quartiles":
            # Symmetric: drop bottom q and top q
            th_low = df[feat].quantile(q)
            th_high = df[feat].quantile(1 - q)
            gate_pass = (df[feat] >= th_low) & (df[feat] <= th_high)
            descr = f"keep {feat} in [{th_low:.0f}, {th_high:.0f}] (middle {100*(1-2*q):.0f}%)"

        # Combine with G2(150)
        combined_gate = gate_pass & (df["min_dim"] >= 150)
        for pool_name, suites in POOL_DEFS.items():
            mask = df["suite"].isin(suites) & combined_gate
            sub = df[mask]
            if sub["label"].nunique() < 2 or len(sub) < 30:
                continue
            labels = sub["label"].values
            scores = sub["T5C_orig"].values
            auc = roc_auc_score(labels, scores)
            rec5 = recall_at_fpr(scores, labels, 0.05)
            correct, total = identity_verdict_score(sub)
            base = baseline[pool_name]
            d_auc = auc - base["AUC"]
            d_rec = rec5 - base["recall@5"]
            d_correct = (correct/max(total,1)) - (base["id_correct"]/max(base["id_total"],1))
            all_gate_results.append({
                "feature": feat, "direction": direction,
                "quantile_drop": q, "pool": pool_name,
                "kept_n": len(sub), "frac_kept_vs_base": len(sub) / base["n"],
                "AUC": auc, "ΔAUC": d_auc,
                "recall@5": rec5, "Δrecall": d_rec,
                "id_correct": correct, "id_total": total,
                "id_correct_rate": correct/max(total,1),
                "Δid_correct_rate": d_correct,
            })

g = pd.DataFrame(all_gate_results)
g.to_csv(OUT / "candidate_gates_sweep.csv", index=False)

# Report: which (feature, q, pool) combos are net positive (ΔAUC > 0 AND Δrecall >= 0 AND Δid_correct >= 0)?
print()
print("=" * 100)
print("CANDIDATE GATES — combos that IMPROVE all 3 metrics (ΔAUC, Δrecall, Δid_correct) over G2(150)-only baseline")
print("=" * 100)
positive = g[(g["ΔAUC"] >= 0) & (g["Δrecall"] >= 0) & (g["Δid_correct_rate"] >= 0)].copy()
positive["score"] = positive["ΔAUC"] + positive["Δrecall"] + positive["Δid_correct_rate"]
positive = positive.sort_values("score", ascending=False)
if len(positive) == 0:
    print("  NONE — no candidate gate strictly dominates the G2(150)-only baseline.")
else:
    print(positive.head(20).to_string(index=False))

# Also: per-pool winner (sorted by ΔAUC alone)
print()
print("=" * 100)
print("Per-pool: top 3 candidate-gate ΔAUC results")
print("=" * 100)
for pool_name in POOL_DEFS:
    sub = g[g["pool"] == pool_name].sort_values("ΔAUC", ascending=False)
    print(f"\n  {pool_name}:")
    if len(sub) == 0:
        print("    (no data)")
        continue
    print(sub.head(5).to_string(index=False))
