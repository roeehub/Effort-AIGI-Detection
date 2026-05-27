"""Design A — multi-axis selective remediation.

Apply blend@0.50 only when the frame is RISKY by some cheap rule.
Compare to (a) no remediation, (b) universal blend@0.50.

Hypothesis: triggering only on lap_var < T_lap captures the win on chronic-FP
reals without disturbing clean reals or sharp fakes.

NOTE: my earlier conditional-vs-universal test on the 260-frame stress pool
showed conditional was WORSE. But that was on a hand-picked FP-rich pool.
The honest test is on the full multi-substrate data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

OUT = Path(__file__).resolve().parent / "outputs"
df = pd.read_csv(OUT / "all_cohorts_scored.csv")
print(f"loaded {len(df)} frames")

# Build production-relevant mixed pools by pairing real & fake cohorts
POOL_DEFS = {
    "teams_dev": ["teams_real_all_dev", "teams_fake_all_dev"],
    "teams_lockbox": ["teams_real_all_lockbox", "teams_fake_all_lockbox"],
    "live_prod": ["live_reals_teams_prod", "live_fakes_teams_prod"],
    "dor_cross": ["dor_evening", "dor_morning", "dor_fake_local", "visomaster_v2_dor"],
    "all_combined": None,  # everything that's two-label-compatible
}

def assign_pool(suite):
    for pool, suites in POOL_DEFS.items():
        if suites is None:
            continue
        if suite in suites:
            return pool
    return "other"

df["pool"] = df["suite"].map(assign_pool)
# For all_combined, just use everything except 'other'
df_combined = df[df["pool"] != "other"].copy()
df_combined["pool"] = "all_combined"

# Stack the pools (some frames will appear in 'all_combined' as well as their named pool)
parts = [df[df["pool"] == p].assign(_pool=p) for p in POOL_DEFS if p != "all_combined"]
parts.append(df_combined.assign(_pool="all_combined"))
two_label = pd.concat(parts, ignore_index=True)
two_label = two_label.rename(columns={"_pool": "eval_pool"})
print(f"\nEval pool composition:")
print(two_label.groupby(["eval_pool", "label"]).size().to_string())
print()


# ============================================================================
# Selective remediation: combine orig + blend conditionally
# ============================================================================
def selective_score(row, ckpt, rule_fn):
    """Pick T5C/P8A_blend_050 if rule_fn(row) is True, else _orig."""
    return row[f"{ckpt}_blend_050"] if rule_fn(row) else row[f"{ckpt}_orig"]


def rule_lap_lt(threshold):
    return lambda r: r["iq_lap_var"] < threshold


def rule_lap_lt_OR_color_high(lap_th, color_th):
    return lambda r: (r["iq_lap_var"] < lap_th) or (r["iq_lab_b_dev"] > color_th)


def rule_lap_quartile(df_sub, q):
    """Threshold at the q-th quantile of lap_var (computed on whole 2-label set)."""
    th = df_sub["iq_lap_var"].quantile(q)
    return rule_lap_lt(th), th


def evaluate(scores_orig, scores_blend, scores_selective, labels, name):
    """Report AUC and recall@5%FPR for all three."""
    auc_o = roc_auc_score(labels, scores_orig)
    auc_b = roc_auc_score(labels, scores_blend)
    auc_s = roc_auc_score(labels, scores_selective)

    def rec_at_5(s):
        fpr, tpr, _ = roc_curve(labels, s)
        ok = fpr <= 0.05
        if ok.sum() == 0: return float("nan")
        return tpr[np.where(ok)[0][-1]]
    r_o = rec_at_5(scores_orig)
    r_b = rec_at_5(scores_blend)
    r_s = rec_at_5(scores_selective)
    return {
        "name": name,
        "AUC_orig": auc_o,
        "AUC_blend": auc_b,
        "AUC_selective": auc_s,
        "AUC_Δuniv_vs_orig": auc_b - auc_o,
        "AUC_Δsel_vs_orig": auc_s - auc_o,
        "AUC_Δsel_vs_univ": auc_s - auc_b,
        "rec@5_orig": r_o,
        "rec@5_blend": r_b,
        "rec@5_selective": r_s,
    }


# ============================================================================
# Sweep lap_var thresholds — find sweet spot for "selective" rule
# ============================================================================
print("=" * 100)
print("Sweep: selective rule = (lap_var < quantile_q × lap_var across all frames)")
print("=" * 100)
quantiles = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]
# Sweep on the 'all_combined' pool first to find good thresholds, then per-pool
for ckpt in ["T5C", "P8A"]:
    print(f"\n### {ckpt} — all_combined pool")
    pool_df = two_label[two_label["eval_pool"] == "all_combined"]
    if pool_df["label"].nunique() < 2:
        print("  pool not two-label; skipping")
        continue
    results = []
    labels = pool_df["label"].values
    s_orig = pool_df[f"{ckpt}_orig"].values
    s_blend = pool_df[f"{ckpt}_blend_050"].values
    for q in quantiles:
        th = float(pool_df["iq_lap_var"].quantile(q))
        is_risky = (pool_df["iq_lap_var"] < th).values
        s_sel = np.where(is_risky, s_blend, s_orig)
        # Also try INVERTED selective: blend only the SHARP frames
        is_sharp = ~is_risky
        s_inv = np.where(is_sharp, s_blend, s_orig)
        n_risky = is_risky.sum()
        r = evaluate(s_orig, s_blend, s_sel, labels, f"blend lap<{th:.0f} (q={q:.2f})")
        r["n_risky"] = n_risky
        r["frac_risky"] = n_risky / len(pool_df)
        results.append(r)
        # Inverted
        r2 = evaluate(s_orig, s_blend, s_inv, labels, f"blend lap>={th:.0f} (q={q:.2f}) INV")
        r2["n_risky"] = (~is_risky).sum()
        r2["frac_risky"] = (~is_risky).mean()
        results.append(r2)
    rd = pd.DataFrame(results).round(4)
    print(rd[["name", "n_risky", "frac_risky", "AUC_orig", "AUC_blend",
              "AUC_selective", "AUC_Δsel_vs_orig", "AUC_Δsel_vs_univ",
              "rec@5_orig", "rec@5_blend", "rec@5_selective"]].to_string(index=False))

# ============================================================================
# Per-cohort: does selective beat universal on each suite?
# ============================================================================
print()
print("=" * 100)
print("Per-suite: selective (lap_var < global median) vs universal blend vs orig")
print("=" * 100)
median_lap = two_label["iq_lap_var"].median()
print(f"(median lap_var = {median_lap:.1f})")
print()
rows = []
for pool in sorted(two_label["eval_pool"].unique()):
    sub = two_label[two_label["eval_pool"] == pool]
    if sub["label"].nunique() < 2:
        continue
    is_risky = (sub["iq_lap_var"] < median_lap).values
    labels = sub["label"].values
    for ckpt in ["T5C", "P8A"]:
        s_o = sub[f"{ckpt}_orig"].values
        s_b = sub[f"{ckpt}_blend_050"].values
        s_s = np.where(is_risky, s_b, s_o)
        try:
            auc_o = roc_auc_score(labels, s_o)
            auc_b = roc_auc_score(labels, s_b)
            auc_s = roc_auc_score(labels, s_s)
        except Exception:
            continue
        rows.append({
            "pool": pool,
            "ckpt": ckpt,
            "n": len(sub),
            "frac_risky": is_risky.mean(),
            "AUC_orig": auc_o,
            "AUC_univ": auc_b,
            "AUC_sel": auc_s,
            "Δuniv": auc_b - auc_o,
            "Δsel": auc_s - auc_o,
            "Δsel-univ": auc_s - auc_b,
        })
rd = pd.DataFrame(rows).round(4)
print(rd.to_string(index=False))
rd.to_csv(OUT / "design_a_per_pool.csv", index=False)
