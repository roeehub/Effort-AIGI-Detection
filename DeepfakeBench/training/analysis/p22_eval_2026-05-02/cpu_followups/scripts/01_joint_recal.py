"""Job A — joint dev+lockbox τ-recalibration.

Question: Is there a τ such that BOTH dev primary FPR ≤ 2% AND lockbox real
FPR ≤ 2% on P22 step8k? Compare to P8A and P18T under same constraint.

For each ckpt, sweep τ on a fine grid. For each τ:
  dev_fpr      = mean(score >= τ) on teams_real_all_dev
  lockbox_fpr  = mean(score >= τ) on teams_real_all_lockbox
  dev_recall   = per-fake-suite recall
  lockbox_recall = per-fake-suite recall
Then locate the smallest τ where both real-FPRs are ≤ floor.
"""
from __future__ import annotations

import sys; sys.path.insert(0, "analysis/p22_eval_2026-05-02/cpu_followups/scripts")
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (CKPT_FILE_SUFFIX, OUT, REAL_SUITES, FAKE_SUITES,
                     load_per_frame_scores)

CKPTS = ["P8A_REFERENCE_STEP5000", "P18T_GRL_TREATMENT_STEP4000",
         "P22_AUG_STEP1000", "P22_AUG_STEP4000", "P22_AUG_STEP8000"]

FLOOR = 0.02
TAU_GRID = np.linspace(0.001, 0.999, 999)


def measure(ckpt):
    real_dev = load_per_frame_scores(ckpt, "teams_real_all_dev")
    real_lockbox = load_per_frame_scores(ckpt, "teams_real_all_lockbox")
    if real_dev is None or real_lockbox is None:
        print(f"[{ckpt}] missing real data, skipping")
        return None
    fake_data = {}
    for s in FAKE_SUITES:
        df = load_per_frame_scores(ckpt, s)
        if df is not None:
            fake_data[s] = df["frame_prob"].to_numpy()

    rd = real_dev["frame_prob"].to_numpy()
    rl = real_lockbox["frame_prob"].to_numpy()

    rows = []
    for tau in TAU_GRID:
        dev_fpr = float((rd >= tau).mean())
        lock_fpr = float((rl >= tau).mean())
        row = {"ckpt": ckpt, "tau": float(tau),
               "dev_real_fpr": dev_fpr, "lockbox_real_fpr": lock_fpr}
        for s, scores in fake_data.items():
            row[f"{s}_recall"] = float((scores >= tau).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def find_joint_compliant(df, floor=FLOOR):
    """Smallest τ s.t. both dev_real_fpr ≤ floor AND lockbox_real_fpr ≤ floor."""
    valid = df[(df.dev_real_fpr <= floor + 1e-9) & (df.lockbox_real_fpr <= floor + 1e-9)]
    if len(valid) == 0:
        return None
    return valid.loc[valid.tau.idxmin()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    for ckpt in CKPTS:
        df = measure(ckpt)
        if df is None:
            continue
        all_dfs.append(df)
    full = pd.concat(all_dfs, ignore_index=True)
    full.to_csv(OUT / "01_joint_recal_full_grid.csv", index=False)

    # Per-floor best τ
    summary = []
    for floor in [0.005, 0.01, 0.02, 0.05]:
        for ckpt in CKPTS:
            sub = full[full.ckpt == ckpt]
            chosen = find_joint_compliant(sub, floor)
            if chosen is None:
                # Best dev-only τ that also satisfies lockbox is impossible; report
                # smallest joint-FPR-violation point
                row = {"ckpt": ckpt, "floor": floor, "joint_compliant": False}
                # If pure-dev τ exists, find it and note its lockbox FPR
                dev_valid = sub[sub.dev_real_fpr <= floor + 1e-9]
                if len(dev_valid):
                    pure_dev = dev_valid.loc[dev_valid.tau.idxmin()]
                    row.update({"tau": pure_dev.tau,
                                "dev_fpr_at_tau": pure_dev.dev_real_fpr,
                                "lockbox_fpr_at_tau": pure_dev.lockbox_real_fpr})
                    for s in FAKE_SUITES:
                        col = f"{s}_recall"
                        if col in pure_dev.index:
                            row[f"{s}_recall_at_pure_dev_tau"] = pure_dev[col]
            else:
                row = {"ckpt": ckpt, "floor": floor, "joint_compliant": True,
                       "tau": chosen.tau,
                       "dev_fpr_at_tau": chosen.dev_real_fpr,
                       "lockbox_fpr_at_tau": chosen.lockbox_real_fpr}
                for s in FAKE_SUITES:
                    col = f"{s}_recall"
                    if col in chosen.index:
                        row[f"{s}_recall"] = chosen[col]
            summary.append(row)

    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUT / "01_joint_recal_summary.csv", index=False)

    # Print at FLOOR=0.02 (the contract default)
    print("=" * 90)
    print(f"Joint dev+lockbox recalibration at FPR floor = {FLOOR}")
    print("=" * 90)
    sub = sdf[sdf.floor == FLOOR]
    cols_show = ["ckpt", "joint_compliant", "tau", "dev_fpr_at_tau", "lockbox_fpr_at_tau",
                 "teams_fake_all_dev_recall", "visomaster_enhanced_macro_dev_recall",
                 "deeplive_enhanced_dev_recall", "teams_fake_all_lockbox_recall"]
    cols_show = [c for c in cols_show if c in sub.columns]
    print(sub[cols_show].to_string(index=False))

    print(f"\nWrote: {OUT}/01_joint_recal_full_grid.csv ({len(full)} rows)")
    print(f"Wrote: {OUT}/01_joint_recal_summary.csv ({len(sdf)} rows)")


if __name__ == "__main__":
    main()
