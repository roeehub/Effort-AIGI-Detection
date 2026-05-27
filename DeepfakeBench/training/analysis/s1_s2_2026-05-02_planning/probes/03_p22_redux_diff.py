"""Probe 3 — direct comparison vs P22 step1k.

For S1 (or S2), compare the recall-vs-FPR curve against P22 step1k on
the same suites. Tells us:
  - Is the new ckpt strictly dominant, or wins on some FPR floors only?
  - Is the dor invariance preserved across the full FPR sweep?

Output: 03_diff_<packet>_vs_p22_step1k.csv + figure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("analysis/s1_s2_2026-05-02_planning/probes/outputs")
FIG = Path("analysis/s1_s2_2026-05-02_planning/probes/figures")

# P22 reference grid (from cpu_followups full grid)
P22_FULL_GRID = Path("analysis/p22_eval_2026-05-02/cpu_followups/outputs/01_joint_recal_full_grid.csv")


def joint_recal_curve(grid: pd.DataFrame, ckpt_key: str):
    """Return (fpr_floor, viso_recall, deeplive_recall, teams_fake_dev_recall, ...)
    sweep across joint dev+lockbox compliant points."""
    if "ckpt" in grid.columns:
        sub = grid[grid.ckpt == ckpt_key].copy()
    else:
        sub = grid[grid.checkpoint_key == ckpt_key].copy()
    if len(sub) == 0:
        return None
    return sub.sort_values("tau") if "tau" in sub.columns else sub.sort_values("threshold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", choices=["S1", "S2"], required=True)
    ap.add_argument("--new-grid", required=True,
                    help="threshold_grid.csv from S1/S2 scorecard")
    ap.add_argument("--new-ckpt-key", required=True)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    p22 = pd.read_csv(P22_FULL_GRID)
    new_grid = pd.read_csv(args.new_grid)

    # Build floor sweep tables for both
    floors = np.linspace(0.005, 0.50, 100)

    def best_recall_at_floor(grid, ckpt_key, floor, suite_col):
        if "ckpt" in grid.columns:
            sub = grid[grid.ckpt == ckpt_key]
            fpr_dev = "dev_real_fpr"
            fpr_lock = "lockbox_real_fpr"
            recall_col = f"{suite_col}_recall"
        else:
            sub = grid[grid.checkpoint_key == ckpt_key]
            fpr_dev = "dev_primary_real_fpr"
            fpr_lock = "teams_real_all_lockbox__real_fpr"
            recall_col = f"{suite_col}__fake_recall"
        # Lockbox FPR may not exist; fall back to dev only
        if fpr_lock in sub.columns:
            valid = sub[(sub[fpr_dev] <= floor + 1e-9) & (sub[fpr_lock] <= floor + 1e-9)]
        else:
            valid = sub[sub[fpr_dev] <= floor + 1e-9]
        if len(valid) == 0 or recall_col not in valid.columns: return None
        return float(valid[recall_col].max())

    suites = ["visomaster_enhanced_macro_dev", "deeplive_enhanced_dev",
              "teams_fake_all_dev"]
    p22_label = "P22_AUG_STEP1000"

    rows = []
    for floor in floors:
        for suite in suites:
            new_recall = best_recall_at_floor(new_grid, args.new_ckpt_key, floor, suite)
            p22_recall = best_recall_at_floor(p22, p22_label, floor, suite)
            rows.append({"floor": floor, "suite": suite,
                          "new_recall": new_recall, "p22_step1k_recall": p22_recall,
                          "delta": (new_recall - p22_recall) if (new_recall is not None and p22_recall is not None) else None})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"03_diff_{args.packet}_vs_p22_step1k.csv", index=False)

    # Figure: 3 subplots, one per suite
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, suite in enumerate(suites):
        ax = axes[i]
        sub = df[df.suite == suite].dropna()
        ax.plot(sub.floor, sub.new_recall, label=f"{args.packet} ({args.new_ckpt_key})", color="tab:green", linewidth=2)
        ax.plot(sub.floor, sub.p22_step1k_recall, label="P22 step1k (reference)", color="tab:blue", linewidth=2, linestyle="--")
        ax.set_xlabel("Joint dev+lockbox FPR floor")
        ax.set_ylabel("Recall (best τ at this floor)")
        ax.set_title(suite)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.axhline(0.9, color="red", linestyle=":", alpha=0.5, label="90% target")
        ax.set_xlim(0, 0.30); ax.set_ylim(0, 1.05)
    plt.suptitle(f"{args.packet} vs P22 step1k — recall-vs-joint-FPR curves")
    plt.tight_layout()
    out = FIG / f"03_diff_{args.packet}_vs_p22_step1k.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote: {out}")
    print(f"Wrote: {OUT}/03_diff_{args.packet}_vs_p22_step1k.csv")


if __name__ == "__main__":
    main()
