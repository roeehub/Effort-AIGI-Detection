"""Statistical validation: bootstrap CIs on AUC delta + sign tests.

Per-cohort, compute a 95% bootstrap CI for AUC(blend) - AUC(orig). Combined
across cohorts, check sign test (does blend win in more cohorts than chance?).

Also reports paired per-identity Δscore Wilcoxon signed-rank.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent / "outputs"
# Use G2-filtered production-eligible pool
df = pd.read_csv(OUT / "g2_pass_pool.csv")
print(f"loaded {len(df)} G2-pass frames")

POOL_DEFS = {
    "teams_dev": ["teams_real_all_dev", "teams_fake_all_dev"],
    "teams_lockbox": ["teams_real_all_lockbox", "teams_fake_all_lockbox"],
    "dor_cross": ["dor_evening", "dor_morning", "dor_fake_local", "visomaster_v2_dor"],
}

def assign_pool(suite):
    for pool, suites in POOL_DEFS.items():
        if suite in suites:
            return pool
    return "other"

df["pool"] = df["suite"].map(assign_pool)
two_label = df[df["pool"] != "other"].copy()
print(f"production-eligible two-label: {len(two_label)}")
print()


def bootstrap_auc_delta(scores_a, scores_b, labels, n_iter=1000, seed=9501):
    """95% CI on AUC(b) - AUC(a)."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    deltas = np.zeros(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        l_b = labels[idx]
        if len(np.unique(l_b)) < 2:
            deltas[i] = float("nan")
            continue
        a_a = roc_auc_score(l_b, scores_a[idx])
        a_b = roc_auc_score(l_b, scores_b[idx])
        deltas[i] = a_b - a_a
    deltas = deltas[~np.isnan(deltas)]
    return {
        "delta_mean": float(np.mean(deltas)),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "n_iter": len(deltas),
        "p_positive_one_sided": float((deltas <= 0).mean()),
    }


# ============================================================================
# Bootstrap CI per suite, per ckpt
# ============================================================================
print("=" * 100)
print("Bootstrap 95% CI on ΔAUC = AUC(blend@0.50) - AUC(orig)")
print("=" * 100)
results = []
for pool in sorted(two_label["pool"].unique()):
    sub = two_label[two_label["pool"] == pool]
    if len(sub) < 30 or sub["label"].nunique() < 2:
        continue
    for ckpt in ["T5C", "P8A"]:
        s_o = sub[f"{ckpt}_orig"].values
        s_b = sub[f"{ckpt}_blend_050"].values
        labels = sub["label"].values
        boot = bootstrap_auc_delta(s_o, s_b, labels, n_iter=2000)
        auc_o = roc_auc_score(labels, s_o)
        results.append({
            "pool": pool,
            "ckpt": ckpt,
            "n": len(sub),
            "n_real": int((labels == 0).sum()),
            "n_fake": int((labels == 1).sum()),
            "AUC_orig": auc_o,
            "ΔAUC_mean": boot["delta_mean"],
            "CI_low": boot["ci_low"],
            "CI_high": boot["ci_high"],
            "ci_excludes_0": (boot["ci_low"] > 0) or (boot["ci_high"] < 0),
            "p_one_sided": boot["p_positive_one_sided"],
        })
rd = pd.DataFrame(results).round(4)
print()
print(rd.to_string(index=False))
rd.to_csv(OUT / "bootstrap_auc_deltas.csv", index=False)

# ============================================================================
# Sign test across cohorts: how many cohorts does blend win?
# ============================================================================
print()
print("=" * 100)
print("Sign test: in how many of the N cohorts does blend strictly improve AUC?")
print("=" * 100)
for ckpt in ["T5C", "P8A"]:
    sub = rd[rd["ckpt"] == ckpt]
    n_total = len(sub)
    n_pos = (sub["ΔAUC_mean"] > 0).sum()
    n_neg = (sub["ΔAUC_mean"] < 0).sum()
    n_sig_pos = ((sub["ΔAUC_mean"] > 0) & sub["ci_excludes_0"]).sum()
    n_sig_neg = ((sub["ΔAUC_mean"] < 0) & sub["ci_excludes_0"]).sum()
    # Binomial test: under H0, P(positive) = 0.5
    pval = stats.binomtest(int(n_pos), int(n_pos + n_neg), p=0.5, alternative="greater").pvalue if n_pos + n_neg > 0 else float("nan")
    print(f"\n{ckpt}:")
    print(f"  Cohorts: {n_total} | blend positive: {n_pos} | blend negative: {n_neg}")
    print(f"  Significant (CI excludes 0) wins: {n_sig_pos} | losses: {n_sig_neg}")
    print(f"  Sign test one-sided p-value: {pval:.4f}")

# ============================================================================
# Per-frame paired Δscore — Wilcoxon for risky-real subgroup (chronic identities)
# ============================================================================
print()
print("=" * 100)
print("Wilcoxon paired signed-rank on per-frame Δscore (full pool)")
print("=" * 100)
for ckpt in ["T5C", "P8A"]:
    for label_name, label_val in [("REAL", 0), ("FAKE", 1)]:
        sub = two_label[two_label["label"] == label_val]
        d = sub[f"{ckpt}_blend_050"].values - sub[f"{ckpt}_orig"].values
        stat = stats.wilcoxon(d, alternative="less" if label_val == 0 else "greater")
        med = float(np.median(d))
        print(f"  {ckpt} / {label_name:<5} n={len(d):<6} median Δ={med:+.4f}  "
              f"Wilcoxon p={stat.pvalue:.3e}  "
              f"(H1: Δ {'<' if label_val == 0 else '>'} 0)")
