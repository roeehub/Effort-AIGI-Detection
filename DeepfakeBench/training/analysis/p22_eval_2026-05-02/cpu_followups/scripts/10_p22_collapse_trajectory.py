"""Job L+ — analyse the P22 W&B trajectory using the right metric names.

The W&B run has explicit collapse-tracking metrics:
  train/collapse/class_separation
  train/collapse/logit_std
  train/collapse/prob_entropy
  train/collapse/prob_spread
plus train/loss/overall, train/metric/auc, val_holdout/overall/auc, etc.

Goal: characterize WHEN collapse begins, what triggers it, and whether the
1000-step peak is sharp or part of a broader stable phase.
"""
from __future__ import annotations

import sys; sys.path.insert(0, "analysis/p22_eval_2026-05-02/cpu_followups/scripts")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import OUT, FIG

DF = pd.read_csv(OUT / "09_wandb_history_P22_dot1buye.csv", low_memory=False)
print(f"Loaded {len(DF)} rows")

# Pick step axis
if "train/step" in DF.columns:
    DF["step"] = DF["train/step"]
elif "global_step" in DF.columns:
    DF["step"] = DF["global_step"]
else:
    DF["step"] = DF["_step"]

KEY_COLS = {
    "train_loss": "train/loss/overall",
    "train_cls_loss": "train/loss/cls_loss",
    "train_auc": "train/metric/auc",
    "val_holdout_auc": "val_holdout/overall/auc",
    "val_in_dist_auc": "val_in_dist/overall/auc",
    "arcface_s": "train/arcface/s",
    "lr": "train/lr",
    "logit_std": "train/collapse/logit_std",
    "class_separation": "train/collapse/class_separation",
    "prob_entropy": "train/collapse/prob_entropy",
    "prob_spread": "train/collapse/prob_spread",
    "logit_range": "train/collapse/logit_range",
    "is_constant_output": "train/collapse/is_constant_output",
}

# Filter columns that exist
KEY_COLS = {k: v for k, v in KEY_COLS.items() if v in DF.columns}
print(f"Found {len(KEY_COLS)} key metric columns")


def at_step(step, metric_col):
    """Get latest non-null value of metric at or before step."""
    sub = DF[(DF.step <= step + 1)].dropna(subset=[metric_col])
    if len(sub) == 0: return None
    return float(sub.iloc[-1][metric_col])


def main():
    # 1) Build a snapshot table at canonical training milestones
    rows = []
    for step in [100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000]:
        row = {"step": step}
        for label, col in KEY_COLS.items():
            row[label] = at_step(step, col)
        rows.append(row)
    snap = pd.DataFrame(rows)
    snap.to_csv(OUT / "10_p22_step_snapshots.csv", index=False)
    print("\nP22 metrics at canonical steps:")
    cols_show = ["step", "train_loss", "train_auc", "val_holdout_auc", "val_in_dist_auc",
                 "logit_std", "class_separation", "prob_entropy", "prob_spread", "arcface_s"]
    cols_show = [c for c in cols_show if c in snap.columns]
    print(snap[cols_show].to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))

    # 2) Plot grid
    metrics_to_plot = [
        ("train_loss", "Training loss"),
        ("train_auc", "Train AUC"),
        ("val_holdout_auc", "Val holdout AUC"),
        ("val_in_dist_auc", "Val in-dist AUC"),
        ("logit_std", "Logit std (low = collapse)"),
        ("class_separation", "Class separation (low = collapse)"),
        ("prob_entropy", "Prob entropy (high = uniform predictions)"),
        ("prob_spread", "Prob spread"),
        ("arcface_s", "ArcFace s"),
        ("lr", "Learning rate"),
    ]
    metrics_to_plot = [(m, t) for (m, t) in metrics_to_plot if m in KEY_COLS]
    n = len(metrics_to_plot)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5 * rows))
    axes = axes.flatten()

    for i, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[i]
        col = KEY_COLS[metric]
        sub = DF.dropna(subset=[col])
        ax.plot(sub.step, sub[col], color="tab:blue", linewidth=1.0, alpha=0.9)
        ax.set_xlabel("step")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        for s, label, c in [(1000, "1k (peak train AUC)", "green"),
                            (4000, "4k", "orange"),
                            (8000, "8k", "red")]:
            ax.axvline(s, color=c, linestyle="--", alpha=0.6, linewidth=0.8)

    for j in range(i + 1, len(axes)):
        axes[j].set_axis_off()

    plt.suptitle("P22 (dot1buye) training trajectory — when does collapse begin?",
                 fontsize=13, y=1.0)
    plt.tight_layout()
    out = FIG / "10_p22_trajectory_full.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nWrote: {out}")

    # 3) Identify the "collapse onset" — when class_separation first drops below half its peak
    if "class_separation" in KEY_COLS:
        cs_col = KEY_COLS["class_separation"]
        sub = DF.dropna(subset=[cs_col])
        peak_cs = sub[cs_col].max()
        peak_step = sub.loc[sub[cs_col].idxmax(), "step"]
        half_peak = peak_cs / 2
        crossings = sub[(sub.step > peak_step) & (sub[cs_col] < half_peak)]
        if len(crossings) > 0:
            collapse_step = float(crossings.iloc[0].step)
            print(f"\nCLASS SEPARATION peak: {peak_cs:.4f} at step {peak_step:.0f}")
            print(f"Class separation drops below half-peak ({half_peak:.4f}) at step ~{collapse_step:.0f}")

    if "logit_std" in KEY_COLS:
        ls_col = KEY_COLS["logit_std"]
        sub = DF.dropna(subset=[ls_col])
        peak_ls = sub[ls_col].max()
        peak_step = sub.loc[sub[ls_col].idxmax(), "step"]
        half_peak = peak_ls / 2
        crossings = sub[(sub.step > peak_step) & (sub[ls_col] < half_peak)]
        if len(crossings) > 0:
            collapse_step = float(crossings.iloc[0].step)
            print(f"LOGIT STD peak: {peak_ls:.4f} at step {peak_step:.0f}")
            print(f"Logit std drops below half-peak ({half_peak:.4f}) at step ~{collapse_step:.0f}")

    # 4) Look at val_holdout AUC trajectory to find peak val performance
    if "val_holdout_auc" in KEY_COLS:
        col = KEY_COLS["val_holdout_auc"]
        sub = DF.dropna(subset=[col])
        peak_val = sub[col].max()
        peak_step = sub.loc[sub[col].idxmax(), "step"]
        print(f"\nVal holdout AUC peak: {peak_val:.4f} at step {peak_step:.0f}")

    if "val_in_dist_auc" in KEY_COLS:
        col = KEY_COLS["val_in_dist_auc"]
        sub = DF.dropna(subset=[col])
        peak_val = sub[col].max()
        peak_step = sub.loc[sub[col].idxmax(), "step"]
        print(f"Val in-dist AUC peak: {peak_val:.4f} at step {peak_step:.0f}")


if __name__ == "__main__":
    main()
