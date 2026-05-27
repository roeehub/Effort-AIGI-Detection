"""
Rule #1 (think-out-loud reply 2026-05-14): test `frac > 0.7 ≥ K` as a positive-evidence
requirement on top of a bulk gate, to discriminate weak dor fakes (frac>0.7 = 17-49%)
from real-person FPs running near threshold (hypothesized frac>0.7 < ~10%).

Compared to existing options:
- Option 1 baseline:  frac>0.49 > 0.5
- Option 2 stricter:  frac>0.6  > 0.4
- Option 3 combined:  frac>0.6  > 0.4  AND count>0.9 >= 1
- N1 family:          (bulk gate) AND  frac>0.7 >= K   for K in {0.05..0.30}
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
SRC = ROOT / "outputs" / "extreme_rule_per_identity.csv"
OUT_DIR = ROOT / "outputs"


def load_pool() -> pd.DataFrame:
    df = pd.read_csv(SRC)
    print(f"Loaded {len(df)} identities from {SRC.name}")
    print(f"Per pool: {df.groupby('pool').size().to_dict()}")
    print(f"Labels: fake={(df['label']==1).sum()}, real={(df['label']==0).sum()}")
    return df


def apply_rule(df, name, predicate):
    v = predicate(df).astype(bool).values
    truth = (df["label"] == 1).values
    correct = (v == truth)
    out = {
        "rule": name,
        "n": len(df),
        "correct": int(correct.sum()),
        "rate": float(correct.mean()),
        "fp": int((v & ~truth).sum()),
        "fn": int((~v & truth).sum()),
    }
    return out, v


def per_pool_breakdown(df, verdict):
    rows = []
    truth = (df["label"] == 1).values
    correct = (verdict == truth)
    for pool, g_idx in df.groupby("pool").indices.items():
        idx = np.array(g_idx)
        rows.append({
            "pool": pool,
            "n": len(idx),
            "correct": int(correct[idx].sum()),
            "rate": float(correct[idx].mean()),
            "fp": int((verdict[idx] & ~truth[idx]).sum()),
            "fn": int((~verdict[idx] & truth[idx]).sum()),
        })
    return pd.DataFrame(rows)


def list_errors(df, verdict, kind="fp"):
    truth = (df["label"] == 1).values
    if kind == "fp":
        mask = verdict & ~truth
    else:
        mask = ~verdict & truth
    cols = ["pool", "base_identity", "n_frames", "mean",
            "frac_gt_0.49", "frac_gt_0.5", "frac_gt_0.6", "frac_gt_0.7",
            "frac_gt_0.8", "frac_gt_0.9", "count_gt_0.9"]
    return df.loc[mask, cols].copy()


def bootstrap_paired_delta(df, vA, vB, n_iter=5000, seed=9501):
    rng = np.random.default_rng(seed)
    truth = (df["label"] == 1).values
    correct_A = (vA == truth).astype(float)
    correct_B = (vB == truth).astype(float)
    n = len(df)
    deltas = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        deltas[i] = correct_B[idx].mean() - correct_A[idx].mean()
    return {
        "delta_mean": float(deltas.mean()),
        "delta_p2.5": float(np.percentile(deltas, 2.5)),
        "delta_p97.5": float(np.percentile(deltas, 97.5)),
        "p_gt_0": float((deltas > 0).mean()),
        "p_ge_0": float((deltas >= 0).mean()),
    }


def main():
    df = load_pool().reset_index(drop=True)

    # Define rules
    rules = []

    def r_baseline(d):
        return d["frac_gt_0.49"] > 0.5
    rules.append(("opt1_baseline_f>0.49>0.5", r_baseline))

    def r_opt2(d):
        return d["frac_gt_0.6"] > 0.4
    rules.append(("opt2_stricter_f>0.6>0.4", r_opt2))

    def r_opt3(d):
        return (d["frac_gt_0.6"] > 0.4) & (d["count_gt_0.9"] >= 1)
    rules.append(("opt3_combined_f>0.6>0.4_AND_c>0.9>=1", r_opt3))

    # N1 family with frac>0.5 bulk
    for K in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
        rules.append(
            (f"N1a_f>0.5>0.5_AND_f>0.7>={K:.2f}",
             lambda d, K=K: (d["frac_gt_0.5"] > 0.5) & (d["frac_gt_0.7"] >= K)),
        )

    # N1 family with frac>0.6 bulk (stricter bulk)
    for K in [0.05, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
        rules.append(
            (f"N1b_f>0.6>0.4_AND_f>0.7>={K:.2f}",
             lambda d, K=K: (d["frac_gt_0.6"] > 0.4) & (d["frac_gt_0.7"] >= K)),
        )

    # OR variant: bulk OR tail (looser, more recall)
    for K in [0.15, 0.20, 0.25, 0.30]:
        rules.append(
            (f"N1c_OR_f>0.49>0.5_OR_f>0.7>={K:.2f}",
             lambda d, K=K: (d["frac_gt_0.49"] > 0.5) | (d["frac_gt_0.7"] >= K)),
        )

    # Run all rules
    results = []
    verdicts = {}
    for name, pred in rules:
        rec, v = apply_rule(df, name, pred)
        results.append(rec)
        verdicts[name] = v

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("rate", ascending=False).reset_index(drop=True)
    df_results.to_csv(OUT_DIR / "rule_n1_summary.csv", index=False)
    print("\n=== RULE COMPARISON (sorted by rate) ===")
    print(df_results.to_string(index=False))

    # For the top 5 N1 rules, do per-pool breakdown + bootstrap vs Option 2 (the simple stricter)
    print("\n=== TOP 5 RULES — PER-POOL BREAKDOWN + BOOTSTRAP vs OPT2 ===")
    v_opt2 = verdicts["opt2_stricter_f>0.6>0.4"]
    v_opt1 = verdicts["opt1_baseline_f>0.49>0.5"]
    v_opt3 = verdicts["opt3_combined_f>0.6>0.4_AND_c>0.9>=1"]

    top_names = list(df_results["rule"].head(8).values)
    breakdowns = {}
    bootstraps = {}
    for name in top_names:
        v = verdicts[name]
        pb = per_pool_breakdown(df, v)
        breakdowns[name] = pb
        bs_vs_opt2 = bootstrap_paired_delta(df, v_opt2, v)
        bs_vs_opt1 = bootstrap_paired_delta(df, v_opt1, v)
        bs_vs_opt3 = bootstrap_paired_delta(df, v_opt3, v)
        bootstraps[name] = {"vs_opt1": bs_vs_opt1, "vs_opt2": bs_vs_opt2, "vs_opt3": bs_vs_opt3}
        print(f"\n--- {name} ---")
        print(pb.to_string(index=False))
        print(f"  vs Option 1 (baseline): Δ={bs_vs_opt1['delta_mean']:+.4f} "
              f"[{bs_vs_opt1['delta_p2.5']:+.4f}, {bs_vs_opt1['delta_p97.5']:+.4f}] "
              f"P(Δ>0)={bs_vs_opt1['p_gt_0']:.3f}")
        print(f"  vs Option 2 (stricter): Δ={bs_vs_opt2['delta_mean']:+.4f} "
              f"[{bs_vs_opt2['delta_p2.5']:+.4f}, {bs_vs_opt2['delta_p97.5']:+.4f}] "
              f"P(Δ>0)={bs_vs_opt2['p_gt_0']:.3f}")
        print(f"  vs Option 3 (combined): Δ={bs_vs_opt3['delta_mean']:+.4f} "
              f"[{bs_vs_opt3['delta_p2.5']:+.4f}, {bs_vs_opt3['delta_p97.5']:+.4f}] "
              f"P(Δ>0)={bs_vs_opt3['p_gt_0']:.3f}")

    # The core question: which dor fakes survive each rule, which FPs survive?
    print("\n=== FOCUS: dor fakes (should be flagged) — frac>0.7 by identity ===")
    dor_fakes = df[(df["label"] == 1) & (df["base_identity"].str.contains("dor_fake", na=False))]
    print(dor_fakes[["base_identity", "n_frames", "mean", "frac_gt_0.5", "frac_gt_0.6",
                    "frac_gt_0.7", "frac_gt_0.9", "count_gt_0.9"]].to_string(index=False))

    print("\n=== FOCUS: known FPs (should NOT be flagged) — frac>0.7 by identity ===")
    fp_now = df[(df["label"] == 0) & (verdicts["opt1_baseline_f>0.49>0.5"])]
    print(fp_now[["pool", "base_identity", "n_frames", "mean", "frac_gt_0.5", "frac_gt_0.6",
                 "frac_gt_0.7", "frac_gt_0.9", "count_gt_0.9"]].to_string(index=False))

    # Best N1 rule — show errors
    print("\n=== BEST N1 RULE — errors ===")
    best_n1 = [r for r in top_names if r.startswith("N1")][0]
    v_best = verdicts[best_n1]
    print(f"Best N1: {best_n1}")
    fp_best = list_errors(df, v_best, kind="fp")
    fn_best = list_errors(df, v_best, kind="fn")
    print(f"\nFPs ({len(fp_best)}):")
    print(fp_best.to_string(index=False))
    print(f"\nFNs ({len(fn_best)}):")
    print(fn_best.to_string(index=False))

    # Save outputs
    df_results.to_csv(OUT_DIR / "rule_n1_summary.csv", index=False)
    with open(OUT_DIR / "rule_n1_bootstrap.json", "w") as f:
        json.dump(bootstraps, f, indent=2)
    print(f"\nSaved: rule_n1_summary.csv, rule_n1_bootstrap.json")


if __name__ == "__main__":
    main()
