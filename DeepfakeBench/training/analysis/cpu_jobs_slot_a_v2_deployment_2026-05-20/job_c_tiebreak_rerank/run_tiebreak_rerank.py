"""Job C — Re-rank the 2026-05-20 29-suite scorecard under alternative tiebreak policies.

The standing v3-fix policy ranks ckpts by:
  1. Gate: dev_primary_real_fpr <= target_real_fpr (0.07)
  2. Gate: dev_worst_real_stress_fpr <= target_stress_fpr (0.10)
  3. Gate: dev_fake_macro_recall >= target_fake_recall_min (0.30)
  4. Tiebreak: lockbox_real_fpr ASCENDING

The P8A vs Slot A v2 step3500 gap is 0.000735 on lockbox_real_fpr (P8A 0.0184
vs SlotAv2 0.0191) — a delta of 1 lockbox-real video out of 1361. This script
asks: under what alternative tiebreak policies does Slot A v2 step3500 rank
ahead of P8A, given its +30pp lockbox_fake_recall advantage (0.688 vs 0.387)?

Reads:
  analysis/manual_canary_2026-05-20/scorecard_pull/checkpoint_summary.csv
  analysis/manual_canary_2026-05-20/scorecard_pull/promotion_winner.json

Writes:
  analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_c_tiebreak_rerank/outputs/
    rerank_table.csv         — full ranking under every policy
    policy_winners.csv       — winner per policy
    bootstrap_check.json     — quick FPR-gap bootstrap (not per-video; uses lockbox N=1361)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
SCORECARD_DIR = REPO / "analysis/manual_canary_2026-05-20/scorecard_pull"
OUT_DIR = REPO / "analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_c_tiebreak_rerank/outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(SCORECARD_DIR / "checkpoint_summary.csv")
    with open(SCORECARD_DIR / "promotion_winner.json") as f:
        contract = json.load(f)["contract"]
    return df, contract


def apply_gates(df: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """Add a `passes_gates` boolean per row."""
    out = df.copy()
    out["gate_real_fpr"] = out["dev_primary_real_fpr"] <= contract["target_real_fpr"]
    out["gate_stress_fpr"] = out["dev_worst_real_stress_fpr"] <= contract["target_stress_fpr"]
    out["gate_fake_recall"] = out["dev_fake_macro_recall"] >= contract["target_fake_recall_min"]
    out["passes_gates"] = out["gate_real_fpr"] & out["gate_stress_fpr"] & out["gate_fake_recall"]
    return out


# ---- Tiebreak policies -------------------------------------------------------


def policy_v3fix_strict(df: pd.DataFrame) -> pd.Series:
    """Current contract: lockbox_real_fpr ASC."""
    return df["lockbox_real_fpr"].rank(method="min", ascending=True)


def policy_lex_thresholded(df: pd.DataFrame, threshold: float) -> pd.Series:
    """Lex on lockbox_real_fpr only when ckpts differ by > `threshold`;
    otherwise tiebreak by lockbox_fake_recall DESC."""
    # Compute a "comparable group" based on rounding lockbox_real_fpr to bins
    # of width = threshold. Within a bin, rank by lockbox_fake_recall DESC.
    rows = df.copy()
    rows["fpr_bin"] = (rows["lockbox_real_fpr"] / threshold).round(0)
    rows = rows.sort_values(
        by=["fpr_bin", "lockbox_fake_recall"],
        ascending=[True, False],
    ).reset_index(drop=False)
    rows["rank"] = range(1, len(rows) + 1)
    return rows.set_index("index")["rank"]


def policy_composite(df: pd.DataFrame, lam: float) -> pd.Series:
    """Composite score = lockbox_fake_recall - lam * lockbox_real_fpr.
    Higher is better."""
    s = df["lockbox_fake_recall"] - lam * df["lockbox_real_fpr"]
    return s.rank(method="min", ascending=False)


def policy_dual_objective_pareto(df: pd.DataFrame) -> pd.Series:
    """Pareto-dominance count: for each ckpt count how many ckpts are
    strictly worse on BOTH lockbox_real_fpr AND lockbox_fake_recall.
    Higher count = better; ties broken by composite λ=10."""
    n = len(df)
    dominates = np.zeros(n, dtype=int)
    fpr = df["lockbox_real_fpr"].values
    rec = df["lockbox_fake_recall"].values
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # i dominates j if i has LOWER fpr AND HIGHER recall
            if fpr[i] < fpr[j] and rec[i] > rec[j]:
                dominates[i] += 1
    # Higher dominates count = better rank; ties to composite λ=10
    composite = rec - 10 * fpr
    keys = np.array(list(zip(-dominates, -composite)))
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)
    return pd.Series(rank, index=df.index)


# ---- Bootstrap on lockbox_real_fpr gap (aggregate level) --------------------


def bootstrap_lockbox_fpr_gap(p8a_fpr: float, slot_fpr: float, n_lockbox: int = 1361, n_boot: int = 100_000, seed: int = 7):
    """Quick bootstrap (per-video resampling assumed iid Bernoulli with given p).
    NOT a paired bootstrap of actual scores — that lives in Job B."""
    rng = np.random.default_rng(seed)
    p8a_n = int(round(p8a_fpr * n_lockbox))
    slot_n = int(round(slot_fpr * n_lockbox))
    p8a_labels = np.zeros(n_lockbox, dtype=int)
    p8a_labels[:p8a_n] = 1
    slot_labels = np.zeros(n_lockbox, dtype=int)
    slot_labels[:slot_n] = 1
    # Independent bootstrap (not paired — Job B does paired)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_lockbox, size=n_lockbox)
        deltas[b] = slot_labels[idx].mean() - p8a_labels[idx].mean()
    ci_lo, ci_hi = np.quantile(deltas, [0.025, 0.975])
    return {
        "p8a_fpr": p8a_fpr,
        "slot_fpr": slot_fpr,
        "p8a_fps_implied": p8a_n,
        "slot_fps_implied": slot_n,
        "absolute_gap": slot_fpr - p8a_fpr,
        "n_lockbox_videos": n_lockbox,
        "bootstrap_delta_mean": float(deltas.mean()),
        "bootstrap_95ci_lo": float(ci_lo),
        "bootstrap_95ci_hi": float(ci_hi),
        "bootstrap_p_slot_higher_fpr": float((deltas > 0).mean()),
        "ci_covers_zero": bool(ci_lo <= 0.0 <= ci_hi),
        "method_note": "Independent bootstrap on Bernoulli with implied n_FPs; treats lockbox-real videos as iid. NOT paired (Job B does paired per-video).",
        "n_boot": n_boot,
    }


def main():
    df, contract = load_data()
    df = apply_gates(df, contract)

    # All policies expect rows ordered the same way for index alignment.
    df = df.reset_index(drop=True)

    policies = {
        "v3fix_strict_lex_fpr_asc": policy_v3fix_strict(df),
        "lex_thresholded_0.001": policy_lex_thresholded(df, threshold=0.001),
        "lex_thresholded_0.005": policy_lex_thresholded(df, threshold=0.005),
        "lex_thresholded_0.010": policy_lex_thresholded(df, threshold=0.010),
        "lex_thresholded_0.030": policy_lex_thresholded(df, threshold=0.030),
        "composite_lambda_5": policy_composite(df, lam=5),
        "composite_lambda_10": policy_composite(df, lam=10),
        "composite_lambda_20": policy_composite(df, lam=20),
        "composite_lambda_50": policy_composite(df, lam=50),
        "composite_lambda_100": policy_composite(df, lam=100),
        "pareto_dominates": policy_dual_objective_pareto(df),
    }

    # Build the rerank table
    rerank = df[
        [
            "checkpoint_key",
            "dev_primary_real_fpr",
            "dev_worst_real_stress_fpr",
            "dev_fake_macro_recall",
            "lockbox_real_fpr",
            "lockbox_fake_recall",
            "passes_gates",
            "promotion_rank",
        ]
    ].copy()
    for policy_name, ranks in policies.items():
        rerank[f"rank__{policy_name}"] = ranks.values
    rerank.to_csv(OUT_DIR / "rerank_table.csv", index=False)

    # Winner per policy
    winners = []
    for policy_name in policies:
        col = f"rank__{policy_name}"
        # Only ckpts that pass gates are eligible
        eligible = rerank[rerank["passes_gates"]]
        if len(eligible) == 0:
            winner = "NONE"
        else:
            winner = eligible.loc[eligible[col].idxmin(), "checkpoint_key"]
        winners.append({"policy": policy_name, "winner": winner})
    pd.DataFrame(winners).to_csv(OUT_DIR / "policy_winners.csv", index=False)

    # Bootstrap on FPR gap
    p8a_row = df[df["checkpoint_key"] == "P8A_REFERENCE_STEP5000"].iloc[0]
    slot_row = df[df["checkpoint_key"] == "SLOT_A_ANCHOR_AWARE_STEP3500"].iloc[0]
    bootstrap_out = bootstrap_lockbox_fpr_gap(
        p8a_fpr=float(p8a_row["lockbox_real_fpr"]),
        slot_fpr=float(slot_row["lockbox_real_fpr"]),
        n_lockbox=int(p8a_row["lockbox_real_n_videos"]),
    )
    with open(OUT_DIR / "bootstrap_check.json", "w") as f:
        json.dump(bootstrap_out, f, indent=2)

    print("=" * 80)
    print("RERANK TABLE")
    print("=" * 80)
    print(rerank.to_string(index=False))
    print()
    print("=" * 80)
    print("WINNER PER POLICY")
    print("=" * 80)
    for w in winners:
        print(f"  {w['policy']:<35}  →  {w['winner']}")
    print()
    print("=" * 80)
    print("BOOTSTRAP ON LOCKBOX_REAL_FPR GAP (aggregate-level)")
    print("=" * 80)
    print(json.dumps(bootstrap_out, indent=2))


if __name__ == "__main__":
    main()
