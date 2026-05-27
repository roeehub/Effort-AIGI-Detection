"""Job J — joint compliance at FPR=5% and FPR=10%.

The user's permissive operating regime is 5-10% real FPR, with a strict
90%-across-the-board fake recall target. Reuses the full joint grid from
Job A and reports what each ckpt achieves at higher FPR floors.

Also reports: SMALLEST joint FPR at which each ckpt hits ≥90% recall on
each fake suite (or "unreachable" if max recall on the grid is <90%).
"""
from __future__ import annotations

import sys; sys.path.insert(0, "analysis/p22_eval_2026-05-02/cpu_followups/scripts")
from pathlib import Path

import numpy as np
import pandas as pd

from _common import OUT, FAKE_SUITES, REAL_SUITES, load_per_frame_scores

CKPTS = ["P8A_REFERENCE_STEP5000", "P18T_GRL_TREATMENT_STEP4000",
         "P22_AUG_STEP1000", "P22_AUG_STEP4000", "P22_AUG_STEP8000"]

GRID = pd.read_csv(OUT / "01_joint_recal_full_grid.csv")


def at_floor(floor):
    """For each ckpt, find smallest τ s.t. both real-FPRs ≤ floor; return recall row."""
    rows = []
    for ckpt in CKPTS:
        sub = GRID[GRID.ckpt == ckpt]
        valid = sub[(sub.dev_real_fpr <= floor + 1e-9) & (sub.lockbox_real_fpr <= floor + 1e-9)]
        if len(valid) == 0:
            rows.append({"ckpt": ckpt, "floor": floor, "joint_compliant": False})
            continue
        chosen = valid.loc[valid.tau.idxmin()]
        row = {"ckpt": ckpt, "floor": floor, "joint_compliant": True,
               "tau": chosen.tau, "dev_real_fpr": chosen.dev_real_fpr,
               "lockbox_real_fpr": chosen.lockbox_real_fpr}
        for s in FAKE_SUITES:
            col = f"{s}_recall"
            if col in chosen.index:
                row[s] = chosen[col]
        rows.append(row)
    return pd.DataFrame(rows)


def find_min_fpr_for_recall_target(target=0.9):
    """For each (ckpt, fake suite), find smallest joint FPR floor at which
    recall on that suite >= target (sweeping floor 0.005 to 0.50)."""
    floors = np.linspace(0.005, 0.50, 100)
    rows = []
    for ckpt in CKPTS:
        sub = GRID[GRID.ckpt == ckpt]
        for s in FAKE_SUITES:
            col = f"{s}_recall"
            if col not in sub.columns: continue
            best = None
            for floor in floors:
                valid = sub[(sub.dev_real_fpr <= floor + 1e-9) & (sub.lockbox_real_fpr <= floor + 1e-9)]
                if len(valid) == 0: continue
                # We want the τ that gives highest recall at this floor on this suite
                # (note: for a single-suite-target view, the best τ at floor for THIS suite,
                #  not the contract's lex-ordered τ)
                best_at_floor = valid[col].max()
                if best_at_floor >= target:
                    best = floor
                    break
            rows.append({"ckpt": ckpt, "fake_suite": s, "min_fpr_for_90pct_recall": best,
                         "max_recall_observed": float(sub[col].max())})
    return pd.DataFrame(rows)


def main():
    rows = []
    for floor in [0.02, 0.05, 0.10, 0.20]:
        df = at_floor(floor)
        rows.append(df)
    full = pd.concat(rows, ignore_index=True)
    full.to_csv(OUT / "08_joint_recal_higher_floors.csv", index=False)

    print("=" * 110)
    print("Joint dev+lockbox τ-recalibration across FPR floors")
    print("=" * 110)
    for floor in [0.02, 0.05, 0.10, 0.20]:
        sub = full[full.floor == floor]
        cols = ["ckpt", "tau", "dev_real_fpr", "lockbox_real_fpr",
                "teams_fake_all_dev", "visomaster_enhanced_macro_dev",
                "deeplive_enhanced_dev", "teams_fake_all_lockbox"]
        cols = [c for c in cols if c in sub.columns]
        print(f"\nFPR floor = {floor:.2f}")
        print(sub[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))

    target_df = find_min_fpr_for_recall_target(target=0.9)
    target_df.to_csv(OUT / "08_min_fpr_for_90pct_recall.csv", index=False)
    print("\n" + "=" * 110)
    print("Minimum joint FPR floor for ≥ 90% recall (per ckpt × fake suite)")
    print("(NaN means recall never reaches 90% on the joint grid; max_recall is observed peak)")
    print("=" * 110)
    pivot = target_df.pivot(index="ckpt", columns="fake_suite", values="min_fpr_for_90pct_recall")
    pivot = pivot.reindex(CKPTS)
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}" if pd.notna(x) else "—"))

    print("\nMax recall observed on each suite (regardless of FPR):")
    pivot_max = target_df.pivot(index="ckpt", columns="fake_suite", values="max_recall_observed")
    pivot_max = pivot_max.reindex(CKPTS)
    print(pivot_max.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
