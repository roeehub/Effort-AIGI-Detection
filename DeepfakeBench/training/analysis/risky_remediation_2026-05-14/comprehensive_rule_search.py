"""Comprehensive search for the most statistically-supported rule.

Three forms of evidence:
1. Fine 4D grid (bulk_τ × bulk_x × tail_τ × tail_K, ~625 combos): is the optimum a
   stable neighborhood or a knife-edge?
2. Permutation test (1000 shuffles): null distribution of best Δ — tighter p-value
   than bootstrap.
3. Per-pool consistency: rule must win in ALL pools (not just on combined average).
4. Holdout cross-validation: train on subset, test on rest.

For each candidate rule, compute:
- identity-correct rate (combined + per-pool)
- bootstrap CI on Δ vs baseline
- # neighboring rules with similar performance (robustness)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from itertools import product

OUT = Path(__file__).resolve().parent / "outputs"
per_id = pd.read_csv(OUT / "extreme_rule_per_identity.csv")
pid = per_id[per_id["n_frames"] >= 5].copy().reset_index(drop=True)
print(f"Pool: {len(pid)} identities ({(pid['label']==0).sum()} real, {(pid['label']==1).sum()} fake)")
print(f"Per-pool: {dict(pid.groupby('pool').size())}")
print()


# We need fraction columns at finer thresholds. Add missing ones if needed.
# From the existing per_id table, we have frac_gt_{0.3, 0.4, 0.49, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99}
# I will use these directly. For "bulk_τ ∈ {0.45, 0.50, ...}" — only the existing thresholds are usable.

BULK_THRESHOLDS = [0.40, 0.49, 0.5, 0.6, 0.7]
BULK_FRACTIONS = [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
TAIL_THRESHOLDS = [0.80, 0.90]  # only 0.8 and 0.9 are useful (scores cap at ~0.94)
TAIL_COUNTS = [1, 2, 3, 5]


def evaluate_rule(pid, bulk_t, bulk_x, tail_t, tail_k):
    bulk_col = f"frac_gt_{bulk_t}"
    cnt_col = f"count_gt_{tail_t}"
    if bulk_col not in pid.columns or cnt_col not in pid.columns:
        return None
    v = ((pid[bulk_col] > bulk_x) & (pid[cnt_col] >= tail_k)).values
    truth = (pid["label"] == 1).values
    correct = (v == truth).sum()
    fp = ((v) & ~truth).sum()
    fn = ((~v) & truth).sum()
    return {"correct": int(correct), "fp": int(fp), "fn": int(fn), "n": len(v),
            "rate": correct / len(v), "verdicts": v}


# Baseline
baseline_v = (pid["frac_gt_0.49"] > 0.5).values
truth = (pid["label"] == 1).values
baseline_correct = (baseline_v == truth).sum()
baseline_rate = baseline_correct / len(pid)
print(f"Baseline: {baseline_correct}/{len(pid)} = {baseline_rate:.4f}")
print()

# ============================================================================
# 1) 4D GRID SEARCH
# ============================================================================
print("=" * 100)
print("1) 4D GRID — bulk_τ × bulk_x × tail_τ × tail_K (combos:", end=" ")
grid = list(product(BULK_THRESHOLDS, BULK_FRACTIONS, TAIL_THRESHOLDS, TAIL_COUNTS))
print(f"{len(grid)})")
print("=" * 100)

all_results = []
for bulk_t, bulk_x, tail_t, tail_k in grid:
    res = evaluate_rule(pid, bulk_t, bulk_x, tail_t, tail_k)
    if res is None:
        continue
    # Per-pool
    per_pool = {}
    for pool in ["teams_dev", "teams_lockbox", "dor_cross"]:
        mask = (pid["pool"] == pool).values
        if not mask.any(): continue
        v_pool = res["verdicts"][mask]
        t_pool = truth[mask]
        per_pool[pool] = (v_pool == t_pool).sum() / mask.sum()

    all_results.append({
        "bulk_t": bulk_t, "bulk_x": bulk_x, "tail_t": tail_t, "tail_k": tail_k,
        "correct": res["correct"], "n": res["n"],
        "rate": res["rate"], "fp": res["fp"], "fn": res["fn"],
        "rate_teams_dev": per_pool.get("teams_dev", float("nan")),
        "rate_lockbox": per_pool.get("teams_lockbox", float("nan")),
        "rate_dor_cross": per_pool.get("dor_cross", float("nan")),
    })

res_df = pd.DataFrame(all_results)
res_df["delta"] = res_df["rate"] - baseline_rate
res_df["all_pools_pos"] = (
    (res_df["rate_teams_dev"] >= 33/34) &  # baseline teams_dev rate (33/34)
    (res_df["rate_lockbox"] >= 5/7) &      # baseline lockbox rate (5/7)
    (res_df["rate_dor_cross"] >= 28/30)    # baseline dor_cross rate (28/30)
)
res_df = res_df.sort_values("rate", ascending=False)
print()
print(f"Top 15 rules by combined correct rate:")
print(f"{'bulk':<14} {'tail':<14} {'correct':<10} {'rate':<8} {'Δ':<8} {'fp':<4} {'fn':<4} {'dev':<6} {'lb':<6} {'dor':<6} {'all_pools_pos':<8}")
for _, r in res_df.head(15).iterrows():
    bulk = f"f>{r['bulk_t']}>{r['bulk_x']}"
    tail = f"c>{r['tail_t']}>={r['tail_k']}"
    print(f"  {bulk:<14} {tail:<14} {r['correct']:<3.0f}/{r['n']:<5.0f} {r['rate']:<8.4f} {r['delta']:+<8.4f} "
          f"{r['fp']:<4.0f} {r['fn']:<4.0f} {r['rate_teams_dev']:<6.3f} {r['rate_lockbox']:<6.3f} {r['rate_dor_cross']:<6.3f} {r['all_pools_pos']}")

# How many rules achieve >=baseline in each pool simultaneously?
print(f"\nRules with rate >= baseline AND no per-pool regression vs baseline:")
better_and_safe = res_df[res_df["all_pools_pos"] & (res_df["rate"] >= baseline_rate)]
print(f"  Count: {len(better_and_safe)} / {len(res_df)}")
if len(better_and_safe) > 0:
    print(f"  rates: {sorted(better_and_safe['rate'].unique(), reverse=True)[:10]}")

# Rules that achieve THE OPTIMUM (69/71)
optimum = res_df[res_df["correct"] == res_df["correct"].max()]
print(f"\nRules achieving the optimum ({optimum['correct'].iloc[0]:.0f}/{len(pid)}):")
print(f"  Count: {len(optimum)}")
for _, r in optimum.iterrows():
    print(f"    bulk f>{r['bulk_t']}>{r['bulk_x']:.2f}  AND  tail c>{r['tail_t']}>={r['tail_k']:.0f}  "
          f"(per-pool: dev={r['rate_teams_dev']:.3f} lb={r['rate_lockbox']:.3f} dor={r['rate_dor_cross']:.3f})")

res_df.to_csv(OUT / "comprehensive_rule_grid.csv", index=False)


# ============================================================================
# 2) PERMUTATION TEST — null distribution of best Δ
# ============================================================================
print()
print("=" * 100)
print("2) PERMUTATION TEST — shuffle labels, compute best rule's Δ under H0")
print("=" * 100)

N_PERM = 1000
n = len(pid)
rng = np.random.default_rng(9501)

# Compute observed best Δ
observed_best_corr = res_df["correct"].max()
observed_best_delta = observed_best_corr / n - baseline_rate
print(f"\nObserved best rate: {observed_best_corr}/{n} = {observed_best_corr/n:.4f}")
print(f"Observed Δ vs baseline: {observed_best_delta:+.4f}")

# Permutation: shuffle labels, find best rule's Δ on shuffled data
null_best_deltas = []
print("Running permutations (this finds the BEST rule's Δ under random labels)...")
for perm in range(N_PERM):
    # Shuffle labels
    perm_labels = rng.permutation(pid["label"].values)
    perm_truth = (perm_labels == 1)
    perm_baseline_correct = ((pid["frac_gt_0.49"] > 0.5).values == perm_truth).sum()
    perm_baseline_rate = perm_baseline_correct / n

    # Find best rule's correct rate on permuted labels
    best_perm_correct = 0
    for bulk_t, bulk_x, tail_t, tail_k in grid:
        bulk_col = f"frac_gt_{bulk_t}"
        cnt_col = f"count_gt_{tail_t}"
        if bulk_col not in pid.columns or cnt_col not in pid.columns:
            continue
        v = ((pid[bulk_col] > bulk_x) & (pid[cnt_col] >= tail_k)).values
        correct = (v == perm_truth).sum()
        if correct > best_perm_correct:
            best_perm_correct = correct
    null_delta = best_perm_correct / n - perm_baseline_rate
    null_best_deltas.append(null_delta)
    if (perm + 1) % 200 == 0:
        print(f"  perm {perm+1}/{N_PERM}: null_best_delta mean so far = {np.mean(null_best_deltas):.4f}")

null_arr = np.array(null_best_deltas)
p_perm = (null_arr >= observed_best_delta).mean()
print(f"\nNull distribution of best Δ:")
print(f"  mean = {null_arr.mean():+.4f}")
print(f"  q90  = {np.quantile(null_arr, 0.90):+.4f}")
print(f"  q95  = {np.quantile(null_arr, 0.95):+.4f}")
print(f"  q99  = {np.quantile(null_arr, 0.99):+.4f}")
print(f"  max  = {null_arr.max():+.4f}")
print(f"\nObserved Δ = {observed_best_delta:+.4f}")
print(f"Permutation p-value (P(null_best_Δ ≥ observed_Δ | H0)) = {p_perm:.4f}")
print(f"  (This accounts for multiple testing across {len(grid)} rules)")


# ============================================================================
# 3) Bootstrap CI on the optimum rule
# ============================================================================
print()
print("=" * 100)
print("3) BOOTSTRAP CIs on the optimum rule")
print("=" * 100)

best_row = optimum.iloc[0]
bulk_t, bulk_x = best_row["bulk_t"], best_row["bulk_x"]
tail_t, tail_k = best_row["tail_t"], int(best_row["tail_k"])
print(f"\nBest rule: bulk f>{bulk_t}>{bulk_x}  AND  tail c>{tail_t}>={tail_k}")

bulk_col = f"frac_gt_{bulk_t}"
cnt_col = f"count_gt_{tail_t}"
v_best = ((pid[bulk_col] > bulk_x) & (pid[cnt_col] >= tail_k)).values

rng = np.random.default_rng(9501)
deltas_full = []
for _ in range(5000):
    idx = rng.integers(0, n, n)
    t_b = truth[idx]
    rate_base = (baseline_v[idx] == t_b).mean()
    rate_best = (v_best[idx] == t_b).mean()
    deltas_full.append(rate_best - rate_base)
deltas_full = np.array(deltas_full)
print(f"  Δ bootstrap (5000 iter, FULL pool): mean={deltas_full.mean():+.4f}  CI=[{np.quantile(deltas_full, 0.025):+.4f}, {np.quantile(deltas_full, 0.975):+.4f}]")
print(f"  P(Δ > 0) = {(deltas_full > 0).mean():.4f}")
print(f"  P(Δ > 0.01) = {(deltas_full > 0.01).mean():.4f}")
print(f"  P(Δ > 0.02) = {(deltas_full > 0.02).mean():.4f}")

# Pool-stratified bootstrap (resample within each pool)
print(f"\nPool-stratified bootstrap (within-pool resampling):")
pools = pid["pool"].values
unique_pools = np.unique(pools)
deltas_strat = []
for _ in range(5000):
    idx_list = []
    for pool in unique_pools:
        pool_idx = np.where(pools == pool)[0]
        idx_list.append(rng.choice(pool_idx, size=len(pool_idx), replace=True))
    idx = np.concatenate(idx_list)
    t_b = truth[idx]
    rate_base = (baseline_v[idx] == t_b).mean()
    rate_best = (v_best[idx] == t_b).mean()
    deltas_strat.append(rate_best - rate_base)
deltas_strat = np.array(deltas_strat)
print(f"  Δ stratified bootstrap: mean={deltas_strat.mean():+.4f}  CI=[{np.quantile(deltas_strat, 0.025):+.4f}, {np.quantile(deltas_strat, 0.975):+.4f}]")
print(f"  P(Δ > 0) = {(deltas_strat > 0).mean():.4f}")


# ============================================================================
# 4) ROBUSTNESS — neighborhood analysis
# ============================================================================
print()
print("=" * 100)
print("4) ROBUSTNESS — how many rules near the optimum also beat baseline?")
print("=" * 100)
opt_corr = optimum["correct"].iloc[0]
# Define neighborhood: same tail-τ + neighboring tail_k values + bulk grid nearby
print(f"\nAmong all {len(res_df)} grid rules, how many achieve:")
for thr in [opt_corr - 0, opt_corr - 1, opt_corr - 2, opt_corr - 3]:
    cnt = (res_df["correct"] >= thr).sum()
    print(f"  correct >= {thr:.0f}/{n}: {cnt} rules ({cnt/len(res_df)*100:.1f}%)")

# How many rules strictly beat baseline?
above = (res_df["correct"] > baseline_correct).sum()
print(f"  Rules strictly > baseline ({baseline_correct}/{n}): {above} ({above/len(res_df)*100:.1f}%)")

# Of those, how many have all pool rates >= baseline?
above_and_safe = res_df[(res_df["correct"] > baseline_correct) & res_df["all_pools_pos"]]
print(f"  Rules strictly > baseline AND no per-pool regression: {len(above_and_safe)}")

# Within the OPTIMUM rules (69/71), what are the bulk_t × bulk_x × tail_t × tail_k ranges?
print(f"\nFor the {len(optimum)} OPTIMUM rules:")
print(f"  bulk_t range: {sorted(optimum['bulk_t'].unique())}")
print(f"  bulk_x range: {sorted(optimum['bulk_x'].unique())}")
print(f"  tail_t range: {sorted(optimum['tail_t'].unique())}")
print(f"  tail_k range: {sorted(optimum['tail_k'].unique())}")


# ============================================================================
# 5) HOLDOUT CROSS-VALIDATION
# ============================================================================
print()
print("=" * 100)
print("5) HOLDOUT CV — leave-one-pool-out: train on 2 pools, test on 3rd")
print("=" * 100)
for test_pool in ["teams_dev", "teams_lockbox", "dor_cross"]:
    train_mask = (pid["pool"] != test_pool).values
    test_mask = ~train_mask
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        continue
    train_truth = truth[train_mask]
    test_truth = truth[test_mask]
    train_baseline_correct = (baseline_v[train_mask] == train_truth).sum()
    test_baseline_correct = (baseline_v[test_mask] == test_truth).sum()
    # Find best rule on train
    best_train_correct = train_baseline_correct
    best_rule = None
    for bulk_t, bulk_x, tail_t, tail_k in grid:
        bulk_col = f"frac_gt_{bulk_t}"
        cnt_col = f"count_gt_{tail_t}"
        if bulk_col not in pid.columns or cnt_col not in pid.columns:
            continue
        v = ((pid[bulk_col] > bulk_x) & (pid[cnt_col] >= tail_k)).values
        train_correct = (v[train_mask] == train_truth).sum()
        if train_correct > best_train_correct:
            best_train_correct = train_correct
            best_rule = (bulk_t, bulk_x, tail_t, tail_k)
    if best_rule:
        bt, bx, tt, tk = best_rule
        bulk_col = f"frac_gt_{bt}"
        cnt_col = f"count_gt_{tt}"
        v = ((pid[bulk_col] > bx) & (pid[cnt_col] >= tk)).values
        test_correct = (v[test_mask] == test_truth).sum()
        print(f"\n  Test pool = {test_pool}")
        print(f"    Best train rule: bulk f>{bt}>{bx}  AND  tail c>{tt}>={tk}")
        print(f"    Train: {best_train_correct}/{train_mask.sum()} = {best_train_correct/train_mask.sum():.4f}  "
              f"(baseline {train_baseline_correct}/{train_mask.sum()} = {train_baseline_correct/train_mask.sum():.4f})")
        print(f"    Test:  {test_correct}/{test_mask.sum()} = {test_correct/test_mask.sum():.4f}  "
              f"(baseline {test_baseline_correct}/{test_mask.sum()} = {test_baseline_correct/test_mask.sum():.4f})")
        print(f"    Test Δ vs baseline: {(test_correct-test_baseline_correct)/test_mask.sum():+.4f}")
    else:
        print(f"\n  Test pool = {test_pool}: NO RULE BEATS BASELINE ON TRAIN")

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"  Best rule: bulk f>{bulk_t}>{bulk_x}  AND  tail c>{tail_t}>={tail_k}")
print(f"  Combined rate: {opt_corr}/{n} = {opt_corr/n:.4f}  (baseline {baseline_correct}/{n} = {baseline_rate:.4f}, Δ={observed_best_delta:+.4f})")
print(f"  Permutation p-value (multi-test corrected): {p_perm:.4f}")
print(f"  Bootstrap P(Δ>0): {(deltas_full > 0).mean():.4f}")
print(f"  Pool-stratified P(Δ>0): {(deltas_strat > 0).mean():.4f}")
