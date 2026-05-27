"""Comprehensive per-identity 'extremity' rule study.

Tests whether the distribution-shape of T5C scores per identity provides
discriminative signal beyond the current majority rule (frac > 0.49 > 0.50).

For each identity (suite × base_identity) in the G2(110)-passing pool with
enough frames, compute rich score-distribution statistics. Then sweep 5 rule
families across all parameter combinations and report per-pool + cross-pool
identity-correct rates with bootstrap CIs.

Output: outputs/extreme_rule_per_identity.csv (per-identity stats)
        outputs/extreme_rule_sweep_all.csv (all rule×param×pool evals)
        outputs/extreme_rule_best.csv (top rules by combined id-correct rate)
"""
from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from scipy import stats as sps

OUT = Path(__file__).resolve().parent / "outputs"
df = pd.read_csv(OUT / "all_cohorts_scored.csv")
print(f"loaded {len(df)} frames")

# --------------------------------------------------------------------------
# Apply G2(110) gate first — that's our deployment baseline
# --------------------------------------------------------------------------
print("computing min(W,H)...")
t0 = time.time()
min_dims = []
for p in df["local"]:
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        min_dims.append(0)
    else:
        h, w = img.shape[:2]
        min_dims.append(min(h, w))
df["min_dim"] = min_dims
print(f"  done in {time.time()-t0:.0f}s")

G2_THRESH = 110
df = df[df["min_dim"] >= G2_THRESH].reset_index(drop=True)
print(f"G2({G2_THRESH})-passing: {len(df)} frames")

# --------------------------------------------------------------------------
# Pool assignment + score column
# --------------------------------------------------------------------------
POOL_DEFS = {
    "teams_dev": ["teams_real_all_dev", "teams_fake_all_dev"],
    "teams_lockbox": ["teams_real_all_lockbox", "teams_fake_all_lockbox"],
    "dor_cross": ["dor_evening", "dor_morning", "dor_fake_local", "visomaster_v2_dor"],
}
def assign_pool(s):
    for p, ss in POOL_DEFS.items():
        if s in ss: return p
    return "other"
df["pool"] = df["suite"].map(assign_pool)
df = df[df["pool"] != "other"].copy()
print(f"production pools: {len(df)} frames")
print(df.groupby(["pool", "suite", "label"]).size().to_string())

SCORE_COL = "T5C_orig"

# --------------------------------------------------------------------------
# Per-identity stats
# --------------------------------------------------------------------------
THRESHOLDS = [0.3, 0.4, 0.49, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
print()
print("computing per-identity stats...")

rows = []
for (pool, suite, base_id), g in df.groupby(["pool", "suite", "base_identity"]):
    if g["label"].nunique() > 1:
        # Shouldn't happen per suite × identity, but skip mixed
        continue
    if len(g) < 5:
        # too few frames to compute meaningful stats — keep but flag
        pass
    label = int(g["label"].iloc[0])
    scores = g[SCORE_COL].values
    row = {
        "pool": pool, "suite": suite, "base_identity": base_id, "label": label,
        "n_frames": len(scores),
        "mean": np.mean(scores),
        "std": np.std(scores),
        "min": np.min(scores), "max": np.max(scores),
        "p10": np.quantile(scores, 0.10),
        "p25": np.quantile(scores, 0.25),
        "p50": np.quantile(scores, 0.50),
        "p75": np.quantile(scores, 0.75),
        "p90": np.quantile(scores, 0.90),
        "p95": np.quantile(scores, 0.95),
        "p99": np.quantile(scores, 0.99),
        "skew": float(sps.skew(scores)) if len(scores) >= 3 else 0.0,
        "kurtosis": float(sps.kurtosis(scores)) if len(scores) >= 4 else 0.0,
    }
    for t in THRESHOLDS:
        row[f"frac_gt_{t}"] = float((scores > t).mean())
        row[f"count_gt_{t}"] = int((scores > t).sum())
    # Conditional: given frame > 0.49, what fraction is >0.9?
    above_tau = scores > 0.49
    if above_tau.sum() > 0:
        row["frac_gt_0.9_cond_0.49"] = float((scores[above_tau] > 0.9).mean())
    else:
        row["frac_gt_0.9_cond_0.49"] = 0.0
    rows.append(row)

per_id = pd.DataFrame(rows)
per_id.to_csv(OUT / "extreme_rule_per_identity.csv", index=False)
print(f"per-identity table: {len(per_id)} identities saved")
print()
print("Identity distribution by pool × label × n_frames bucket:")
per_id["n_bucket"] = pd.cut(per_id["n_frames"],
                              bins=[0, 5, 10, 20, 50, 100, 100000],
                              labels=["<5", "5-9", "10-19", "20-49", "50-99", "100+"])
print(per_id.groupby(["pool", "label", "n_bucket"], observed=False).size().unstack(fill_value=0).to_string())

# --------------------------------------------------------------------------
# Baseline: current production rule (frac > 0.49 > 0.5)
# --------------------------------------------------------------------------
def apply_rule_basic(d, tau, x_majority):
    """rule = frac > tau > x_majority -> verdict_fake."""
    return (d[f"frac_gt_{tau}"] > x_majority).values

def apply_rule_AND(d, tau, x_majority, tau_e, x_e):
    """AND rule: must pass BOTH (frac>tau>x_maj) AND (frac>tau_e>x_e)."""
    pass1 = d[f"frac_gt_{tau}"] > x_majority
    pass2 = d[f"frac_gt_{tau_e}"] > x_e
    return (pass1 & pass2).values

def apply_rule_OR(d, tau, x_majority, tau_e, x_e):
    pass1 = d[f"frac_gt_{tau}"] > x_majority
    pass2 = d[f"frac_gt_{tau_e}"] > x_e
    return (pass1 | pass2).values

def apply_rule_REPLACE(d, tau_e, x_e):
    return (d[f"frac_gt_{tau_e}"] > x_e).values

def apply_rule_BLEND(d, alpha, threshold):
    """alpha * frac>0.49 + (1-alpha) * frac>0.9 > threshold."""
    s = alpha * d["frac_gt_0.49"].values + (1 - alpha) * d["frac_gt_0.9"].values
    return s > threshold

def evaluate(verdicts, labels):
    """Returns dict of identity-level metrics."""
    truth = (labels == 1)
    tp = ((verdicts == True) & truth).sum()
    fp = ((verdicts == True) & ~truth).sum()
    tn = ((verdicts == False) & ~truth).sum()
    fn = ((verdicts == False) & truth).sum()
    n = len(labels)
    correct = (verdicts == truth).sum()
    return {
        "n": n, "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "id_correct": int(correct),
        "id_correct_rate": correct / n if n > 0 else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "real_fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "fake_fnr": fn / (tp + fn) if (tp + fn) > 0 else 0.0,
    }

# --------------------------------------------------------------------------
# Define rule grids
# --------------------------------------------------------------------------
RULE_GRIDS = {
    "basic": [
        (tau, x) for tau in [0.3, 0.4, 0.49, 0.5, 0.6, 0.7, 0.8, 0.9]
        for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ],
    "AND": [
        ("0.49", 0.5, tau_e, x_e) for tau_e in [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        for x_e in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]
    ],
    "OR": [
        ("0.49", 0.5, tau_e, x_e) for tau_e in [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        for x_e in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]
    ],
    "REPLACE": [
        (tau_e, x_e) for tau_e in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        for x_e in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    ],
    "BLEND": [
        (alpha, threshold) for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ],
}

# --------------------------------------------------------------------------
# Evaluate rules
# --------------------------------------------------------------------------
# Filter to identities with enough frames — start with n>=5 (loose)
N_MIN_CUTOFFS = [5, 10, 20]
print()
print(f"\nEvaluating all rules at multiple sample-size cutoffs {N_MIN_CUTOFFS}...")

all_evals = []

for n_min in N_MIN_CUTOFFS:
    pid = per_id[per_id["n_frames"] >= n_min].reset_index(drop=True)
    if len(pid) == 0:
        continue
    pools = ["teams_dev", "teams_lockbox", "dor_cross", "all"]

    for rule_family, grid in RULE_GRIDS.items():
        for params in grid:
            for pool in pools:
                if pool == "all":
                    sub = pid
                else:
                    sub = pid[pid["pool"] == pool]
                if len(sub) == 0 or sub["label"].nunique() < 2:
                    continue
                # Apply rule
                if rule_family == "basic":
                    tau, x = params
                    verdicts = apply_rule_basic(sub, tau, x)
                    rule_str = f"frac>{tau}>{x}"
                elif rule_family == "AND":
                    tau, x_m, tau_e, x_e = params
                    verdicts = apply_rule_AND(sub, tau, x_m, tau_e, x_e)
                    rule_str = f"frac>{tau}>{x_m} AND frac>{tau_e}>{x_e}"
                elif rule_family == "OR":
                    tau, x_m, tau_e, x_e = params
                    verdicts = apply_rule_OR(sub, tau, x_m, tau_e, x_e)
                    rule_str = f"frac>{tau}>{x_m} OR frac>{tau_e}>{x_e}"
                elif rule_family == "REPLACE":
                    tau_e, x_e = params
                    verdicts = apply_rule_REPLACE(sub, tau_e, x_e)
                    rule_str = f"frac>{tau_e}>{x_e}"
                elif rule_family == "BLEND":
                    alpha, threshold = params
                    verdicts = apply_rule_BLEND(sub, alpha, threshold)
                    rule_str = f"a={alpha}*frac>.49+(1-a)*frac>.9 > {threshold}"
                m = evaluate(verdicts, sub["label"].values)
                m["family"] = rule_family
                m["rule"] = rule_str
                m["pool"] = pool
                m["n_min"] = n_min
                m["params"] = str(params)
                all_evals.append(m)

ev = pd.DataFrame(all_evals)
ev.to_csv(OUT / "extreme_rule_sweep_all.csv", index=False)
print(f"  {len(ev)} (rule × params × pool × n_min) evaluations saved")

# --------------------------------------------------------------------------
# Baseline reference: current production rule (frac>0.49 > 0.5)
# --------------------------------------------------------------------------
print()
print("=" * 100)
print("BASELINE: current production rule (frac > 0.49 > 0.5)")
print("=" * 100)
for n_min in N_MIN_CUTOFFS:
    pid = per_id[per_id["n_frames"] >= n_min]
    for pool in ["teams_dev", "teams_lockbox", "dor_cross", "all"]:
        sub = pid if pool == "all" else pid[pid["pool"] == pool]
        if len(sub) == 0 or sub["label"].nunique() < 2:
            continue
        v = apply_rule_basic(sub, 0.49, 0.5)
        m = evaluate(v, sub["label"].values)
        print(f"  n_min={n_min:<3} pool={pool:<14} n={m['n']:<3} "
              f"correct={m['id_correct']}/{m['n']} rate={m['id_correct_rate']:.4f} "
              f"FP={m['fp']} FN={m['fn']} (prec={m['precision']:.3f} rec={m['recall']:.3f})")

# --------------------------------------------------------------------------
# Best rules per family, sorted by combined id_correct_rate at n_min=5
# --------------------------------------------------------------------------
print()
print("=" * 100)
print("TOP 5 RULES PER FAMILY by combined id_correct_rate (at n_min=5)")
print("=" * 100)
combined = ev[(ev["pool"] == "all") & (ev["n_min"] == 5)].copy()
for fam in ["basic", "AND", "OR", "REPLACE", "BLEND"]:
    sub = combined[combined["family"] == fam].sort_values("id_correct_rate", ascending=False)
    print(f"\n### {fam}")
    print(f"  {'rule':<55} {'correct':<10} {'rate':<8} {'fp':<4} {'fn':<4} {'prec':<6} {'rec':<6}")
    for _, r in sub.head(5).iterrows():
        print(f"  {r['rule']:<55} {r['id_correct']}/{r['n']:<5} {r['id_correct_rate']:<8.4f} "
              f"{r['fp']:<4} {r['fn']:<4} {r['precision']:<6.3f} {r['recall']:<6.3f}")

# --------------------------------------------------------------------------
# Cross-pool consistency check — best 'AND' rule on teams_dev, validated on others
# --------------------------------------------------------------------------
print()
print("=" * 100)
print("CROSS-POOL VALIDATION: select rule on teams_dev (n_min=5), test on lockbox + dor_cross")
print("=" * 100)
TRAIN_POOL = "teams_dev"
for fam in ["AND", "OR", "REPLACE", "basic", "BLEND"]:
    train_sub = ev[(ev["family"] == fam) & (ev["pool"] == TRAIN_POOL) & (ev["n_min"] == 5)]
    if len(train_sub) == 0:
        continue
    train_sorted = train_sub.sort_values("id_correct_rate", ascending=False)
    best_params = train_sorted.iloc[0]["params"]
    print(f"\n### {fam}  best on teams_dev:")
    print(f"  rule = {train_sorted.iloc[0]['rule']}  "
          f"train rate={train_sorted.iloc[0]['id_correct_rate']:.4f}")
    for test_pool in ["teams_lockbox", "dor_cross", "all"]:
        test_row = ev[(ev["family"] == fam) & (ev["pool"] == test_pool) &
                       (ev["n_min"] == 5) & (ev["params"] == best_params)]
        if len(test_row) == 0:
            continue
        r = test_row.iloc[0]
        print(f"    {test_pool:<14} rate={r['id_correct_rate']:.4f} "
              f"correct={r['id_correct']}/{r['n']} fp={r['fp']} fn={r['fn']}")

# --------------------------------------------------------------------------
# Identity-level analysis: which identities are RESCUED by best AND rule vs baseline?
# --------------------------------------------------------------------------
print()
print("=" * 100)
print("IDENTITY-LEVEL: best AND rule on combined pool vs baseline (n_min=5)")
print("=" * 100)
best_and = combined[combined["family"] == "AND"].sort_values("id_correct_rate", ascending=False)
if len(best_and) > 0:
    top_and = best_and.iloc[0]
    print(f"\nBest AND rule: {top_and['rule']}")
    print(f"  id_correct: {top_and['id_correct']}/{top_and['n']} = {top_and['id_correct_rate']:.4f}")

    # Reproduce the verdicts
    params_str = top_and["params"]
    # params is "('0.49', 0.5, tau_e, x_e)"
    import ast
    params_tuple = ast.literal_eval(params_str)
    _, _, tau_e, x_e = params_tuple
    full = per_id[per_id["n_frames"] >= 5].copy()
    v_new = apply_rule_AND(full, "0.49", 0.5, tau_e, x_e)
    v_old = apply_rule_basic(full, 0.49, 0.5)
    truth = (full["label"] == 1).values
    correct_old = (v_old == truth)
    correct_new = (v_new == truth)
    rescued = (~correct_old) & correct_new
    newly_wrong = correct_old & (~correct_new)
    print(f"\n  Identities RESCUED (wrong->right): {rescued.sum()}")
    print(f"  Identities NEWLY WRONG (right->wrong): {newly_wrong.sum()}")

    if rescued.sum() > 0:
        print("\n  RESCUED:")
        rsub = full[rescued].copy()
        for _, r in rsub.iterrows():
            print(f"    {r['pool']:<14} {r['base_identity']:<30} label={r['label']} "
                  f"n={r['n_frames']:<4} mean={r['mean']:.3f} "
                  f"frac>0.49={r['frac_gt_0.49']:.3f} frac>{tau_e}={r[f'frac_gt_{tau_e}']:.3f}")
    if newly_wrong.sum() > 0:
        print("\n  NEWLY WRONG:")
        nsub = full[newly_wrong].copy()
        for _, r in nsub.iterrows():
            print(f"    {r['pool']:<14} {r['base_identity']:<30} label={r['label']} "
                  f"n={r['n_frames']:<4} mean={r['mean']:.3f} "
                  f"frac>0.49={r['frac_gt_0.49']:.3f} frac>{tau_e}={r[f'frac_gt_{tau_e}']:.3f}")

# --------------------------------------------------------------------------
# Bootstrap CI on best rule's combined id_correct_rate
# --------------------------------------------------------------------------
def bootstrap_id_rate(verdicts, labels, n_iter=2000, seed=9501):
    truth = (labels == 1)
    correct = (verdicts == truth).astype(int)
    rng = np.random.default_rng(seed)
    n = len(correct)
    rates = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        rates.append(correct[idx].mean())
    rates = np.array(rates)
    return float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))

print()
print("=" * 100)
print("BOOTSTRAP 95% CIs on combined id_correct_rate")
print("=" * 100)
full = per_id[per_id["n_frames"] >= 5].copy()
truth = (full["label"] == 1).values

# Baseline
v_base = apply_rule_basic(full, 0.49, 0.5)
m_base = evaluate(v_base, full["label"].values)
ci_lo, ci_hi = bootstrap_id_rate(v_base, full["label"].values)
print(f"\nBaseline (frac>0.49>0.5):  rate={m_base['id_correct_rate']:.4f}  CI=[{ci_lo:.4f}, {ci_hi:.4f}]")

# Top 3 from each rule family on combined pool
for fam in ["AND", "OR", "REPLACE", "BLEND", "basic"]:
    fam_sub = combined[combined["family"] == fam].sort_values("id_correct_rate", ascending=False)
    print(f"\n  Top 3 {fam}:")
    for i in range(min(3, len(fam_sub))):
        r = fam_sub.iloc[i]
        params_tuple = ast.literal_eval(r["params"])
        if fam == "basic":
            tau, x = params_tuple
            v = apply_rule_basic(full, tau, x)
        elif fam == "AND":
            tau, x_m, tau_e, x_e = params_tuple
            v = apply_rule_AND(full, tau, x_m, tau_e, x_e)
        elif fam == "OR":
            tau, x_m, tau_e, x_e = params_tuple
            v = apply_rule_OR(full, tau, x_m, tau_e, x_e)
        elif fam == "REPLACE":
            tau_e, x_e = params_tuple
            v = apply_rule_REPLACE(full, tau_e, x_e)
        elif fam == "BLEND":
            alpha, threshold = params_tuple
            v = apply_rule_BLEND(full, alpha, threshold)
        ci_lo, ci_hi = bootstrap_id_rate(v, full["label"].values)
        d_base = r["id_correct_rate"] - m_base["id_correct_rate"]
        sig = "*" if (ci_lo > m_base["id_correct_rate"] or ci_hi < m_base["id_correct_rate"]) else " "
        print(f"    rate={r['id_correct_rate']:.4f}  Δ={d_base:+.4f}  CI=[{ci_lo:.4f}, {ci_hi:.4f}] {sig} "
              f"-- {r['rule']}")
