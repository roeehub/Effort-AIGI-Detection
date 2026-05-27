"""G2 threshold sensitivity — what happens if we relax from 200 to 150?

For each frame, get actual min(W,H) from local image. Compute T5C performance
at G2 thresholds {150, 175, 200, 225, 250}. Per-pool AUC + recall@5%FPR.

Also: examine the newly-admitted 150-200 band specifically — what's the
real-vs-fake separability on those previously-dropped frames?
"""
from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

OUT = Path(__file__).resolve().parent / "outputs"
df = pd.read_csv(OUT / "all_cohorts_scored.csv")
print(f"loaded {len(df)} frames")

# Get min(W,H) for each frame
print("\ncomputing min(W,H) for all frames...")
t0 = time.time()
min_dims = []
for i, p in enumerate(df["local"]):
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        min_dims.append(0)
        continue
    h, w = img.shape[:2]
    min_dims.append(min(h, w))
    if (i+1) % 3000 == 0:
        print(f"  {i+1}/{len(df)} ({time.time()-t0:.0f}s)")
df["min_dim"] = min_dims
print(f"done in {time.time()-t0:.1f}s")
print()

# Distribution of min_dim
print("min(W,H) distribution per suite:")
for suite in sorted(df["suite"].unique()):
    sub = df[df["suite"] == suite]
    print(f"  {suite:<24} n={len(sub):<5} p10={sub['min_dim'].quantile(0.1):.0f} "
          f"p50={sub['min_dim'].quantile(0.5):.0f} p90={sub['min_dim'].quantile(0.9):.0f}")
print()

# Pool definitions
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


# ============================================================================
# Per-pool: sweep G2 thresholds
# ============================================================================
THRESHOLDS = [150, 175, 200, 225, 250]
print("=" * 100)
print(f"Per-pool T5C performance at G2 thresholds {THRESHOLDS}")
print("=" * 100)

rows = []
for pool_name, suites in POOL_DEFS.items():
    pool_df = df[df["suite"].isin(suites)].copy()
    print(f"\n### {pool_name} (total: {len(pool_df)} frames)")
    print(f"  {'G2':<6} {'pass_n':<8} {'pass%':<8} {'n_real':<8} {'n_fake':<8} "
          f"{'AUC':<10} {'recall@5%FPR':<14}")
    for th in THRESHOLDS:
        sub = pool_df[pool_df["min_dim"] >= th]
        if sub["label"].nunique() < 2 or len(sub) < 30:
            print(f"  {th:<6} {len(sub):<8} insufficient")
            continue
        labels = sub["label"].values
        scores = sub["T5C_orig"].values
        auc = roc_auc_score(labels, scores)
        rec = recall_at_fpr(scores, labels, 0.05)
        n_r = (labels == 0).sum()
        n_f = (labels == 1).sum()
        print(f"  {th:<6} {len(sub):<8} {len(sub)/len(pool_df)*100:<7.1f}% "
              f"{n_r:<8} {n_f:<8} {auc:<10.4f} {rec:<14.4f}")
        rows.append({
            "pool": pool_name, "g2_threshold": th,
            "n_pass": len(sub), "frac_pass": len(sub)/len(pool_df),
            "n_real": n_r, "n_fake": n_f,
            "AUC": auc, "recall@5%FPR": rec,
        })

pd.DataFrame(rows).to_csv(OUT / "g2_threshold_sweep.csv", index=False)

# ============================================================================
# What's in the 150-199 band specifically? (the newly-admitted frames at G2=150)
# ============================================================================
print()
print("=" * 100)
print("Frames in [150, 199] band — newly admitted by relaxing G2 from 200 to 150")
print("=" * 100)
band = df[(df["min_dim"] >= 150) & (df["min_dim"] < 200)].copy()
print(f"\nTotal frames in [150,199]: {len(band)}")
print(f"\nDistribution by suite × label:")
print(band.groupby(["suite", "label"]).size().to_string())

# Per-pool: AUC of T5C on this band alone
print(f"\nT5C separability on [150,199] band only (per pool):")
for pool_name in POOL_DEFS:
    pool_band = band[band["suite"].isin(POOL_DEFS[pool_name])]
    if pool_band["label"].nunique() < 2:
        print(f"  {pool_name}: insufficient (n={len(pool_band)} or single-label)")
        continue
    labels = pool_band["label"].values
    scores = pool_band["T5C_orig"].values
    auc = roc_auc_score(labels, scores)
    n_r = (labels == 0).sum()
    n_f = (labels == 1).sum()
    rec = recall_at_fpr(scores, labels, 0.05)
    # Also: fraction of reals flagged at tau=0.49
    fp_rate = float((scores[labels == 0] > 0.49).mean())
    tp_rate = float((scores[labels == 1] > 0.49).mean())
    print(f"  {pool_name:<14} n={len(pool_band):<5} (real={n_r}, fake={n_f})  "
          f"AUC={auc:.4f}  recall@5%FPR={rec:.4f}  FPR@τ=0.49={fp_rate:.4f}  TPR@τ=0.49={tp_rate:.4f}")

# ============================================================================
# Compare: at the SAME deployment τ=0.49, what changes between G2=150 vs G2=200?
# ============================================================================
print()
print("=" * 100)
print(f"DEPLOYMENT IMPACT at τ=0.49 — frame-counts above τ (fakes caught) vs below (reals cleared)")
print("=" * 100)

for pool_name, suites in POOL_DEFS.items():
    pool_df = df[df["suite"].isin(suites)].copy()
    print(f"\n### {pool_name}")
    print(f"  {'G2':<6} {'pass_n':<8} {'FP_count':<10} {'FP_rate':<10} "
          f"{'TP_count':<10} {'TP_rate':<10} {'precision':<10}")
    for th in [150, 175, 200]:
        sub = pool_df[pool_df["min_dim"] >= th]
        if len(sub) == 0:
            continue
        labels = sub["label"].values
        scores = sub["T5C_orig"].values
        # At τ=0.49
        above = scores > 0.49
        fp_count = ((labels == 0) & above).sum()
        tp_count = ((labels == 1) & above).sum()
        n_r = (labels == 0).sum()
        n_f = (labels == 1).sum()
        fp_rate = fp_count / max(n_r, 1)
        tp_rate = tp_count / max(n_f, 1)
        prec = tp_count / max(fp_count + tp_count, 1)
        print(f"  {th:<6} {len(sub):<8} {fp_count:<10} {fp_rate:<10.4f} "
              f"{tp_count:<10} {tp_rate:<10.4f} {prec:<10.4f}")

# ============================================================================
# Per-identity (majority vote) impact: G2=150 vs G2=200
# Production uses per-identity majority — does relaxing the gate change
# the IDENTITY-LEVEL verdict for any identity?
# ============================================================================
print()
print("=" * 100)
print("PER-IDENTITY VERDICT (frac>0.49 > 0.50 = flagged fake) at G2=150 vs G2=200")
print("=" * 100)
print(f"\nIdentities with >=10 frames at G2(200), per pool:")

for pool_name, suites in POOL_DEFS.items():
    pool_df = df[df["suite"].isin(suites)].copy()
    print(f"\n### {pool_name}")
    print(f"  {'identity':<28} {'lbl':<4} {'n_g200':<8} {'frac>τ_g200':<14} "
          f"{'n_g150':<8} {'frac>τ_g150':<14} {'verdict_change':<16}")

    for ident, g in pool_df.groupby("base_identity"):
        g_g150 = g[g["min_dim"] >= 150]
        g_g200 = g[g["min_dim"] >= 200]
        if len(g_g200) < 10 and len(g_g150) < 10:
            continue
        label = int(g["label"].iloc[0])
        if len(g_g200) >= 1:
            frac_g200 = float((g_g200["T5C_orig"] > 0.49).mean())
            verdict_g200 = frac_g200 > 0.50
        else:
            frac_g200, verdict_g200 = float("nan"), None
        if len(g_g150) >= 1:
            frac_g150 = float((g_g150["T5C_orig"] > 0.49).mean())
            verdict_g150 = frac_g150 > 0.50
        else:
            frac_g150, verdict_g150 = float("nan"), None

        if verdict_g200 != verdict_g150:
            change = f"CHANGED ({verdict_g200} -> {verdict_g150})"
        else:
            change = "no change"

        # Only print if change OR meaningful sample
        if change != "no change" or (len(g_g150) >= 20 or len(g_g200) >= 20):
            print(f"  {ident:<28} {label:<4} {len(g_g200):<8} {frac_g200:<14.4f} "
                  f"{len(g_g150):<8} {frac_g150:<14.4f} {change:<16}")
