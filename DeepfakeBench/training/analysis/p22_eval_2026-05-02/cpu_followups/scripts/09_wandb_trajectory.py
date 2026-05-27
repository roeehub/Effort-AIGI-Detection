"""Job L — pull P22 W&B per-step training trajectory.

We have eval data for step 1k, 4k, 8k only. The trainer-side W&B run has
per-step metrics that can characterize the inflection point: when did
train AUC peak, when did val_holdout AUC start regressing, when did
ArcFace s saturate, etc.

Pull and plot.
"""
from __future__ import annotations

import os
import sys; sys.path.insert(0, "analysis/p22_eval_2026-05-02/cpu_followups/scripts")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wandb
from wandb.apis.public import Api

from _common import OUT, FIG

ENTITY = "dtect-vision"
PROJECT = "phase2r13-experiments"
RUNS = {
    "P22_dot1buye": "dot1buye",
    "P8A_9lmvb5b4": "9lmvb5b4",
    "P18T_xpbvc1e4": "xpbvc1e4",
}

KEY_METRICS = [
    "train_loss", "train_auc",
    "val_holdout_auc", "val_in_dist_auc",
    "ArcFace_s",
    "lr", "step",
    "global_step",
]


def fetch_run_history(api, run_path):
    run = api.run(run_path)
    history_df = run.history(samples=20000)
    return history_df, run


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    api = Api()
    all_dfs = {}
    for label, run_id in RUNS.items():
        run_path = f"{ENTITY}/{PROJECT}/{run_id}"
        try:
            df, run = fetch_run_history(api, run_path)
        except Exception as e:
            print(f"  failed to fetch {run_path}: {e}")
            continue
        df["run_label"] = label
        df["run_id"] = run_id
        # Some runs use "step", some "_step", some "global_step"
        if "global_step" in df.columns:
            df["x_step"] = df["global_step"]
        elif "step" in df.columns:
            df["x_step"] = df["step"]
        else:
            df["x_step"] = df["_step"]
        all_dfs[label] = df
        print(f"  fetched {label}: {len(df)} rows, columns: {list(df.columns)[:15]}...")

    # Save raw
    for label, df in all_dfs.items():
        df.to_csv(OUT / f"09_wandb_history_{label}.csv", index=False)

    # Plot key trajectories for P22
    p22 = all_dfs.get("P22_dot1buye")
    if p22 is not None:
        # Find columns that exist
        plot_metrics = [c for c in ["train_auc", "val_holdout_auc", "val_in_dist_auc",
                                     "train_loss", "ArcFace_s", "lr"]
                        if c in p22.columns]
        n = len(plot_metrics)
        if n > 0:
            fig, axes = plt.subplots((n+1)//2, 2, figsize=(14, 4 * ((n+1)//2)))
            axes = axes.flatten() if n > 1 else [axes]
            for i, metric in enumerate(plot_metrics):
                ax = axes[i]
                sub = p22.dropna(subset=[metric])
                ax.plot(sub["x_step"], sub[metric], color="tab:blue", linewidth=1.2)
                ax.set_xlabel("global_step")
                ax.set_ylabel(metric)
                ax.set_title(f"P22 (dot1buye) — {metric}")
                ax.grid(alpha=0.3)
                # mark step 1000, 4000, 8000
                for s, label, c in [(1000, "1k (peak)", "green"),
                                     (4000, "4k", "orange"),
                                     (8000, "8k", "red")]:
                    ax.axvline(s, color=c, linestyle="--", alpha=0.5, label=label)
                if i == 0:
                    ax.legend(fontsize=8)
            for j in range(i+1, len(axes)):
                axes[j].set_axis_off()
            plt.tight_layout()
            plt.savefig(FIG / "09_wandb_p22_trajectory.png", dpi=140, bbox_inches="tight")
            plt.close()
            print(f"  wrote {FIG}/09_wandb_p22_trajectory.png")

    # Print summary table for P22 at key steps
    if p22 is not None:
        print("\n" + "=" * 80)
        print("P22 (dot1buye) metrics at key training steps")
        print("=" * 80)
        for step in [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]:
            # Find closest row to step
            sub = p22.dropna(subset=["x_step"])
            if len(sub) == 0: continue
            idx = (sub["x_step"] - step).abs().idxmin()
            row = sub.loc[idx]
            metrics = []
            for m in ["train_auc", "val_holdout_auc", "val_in_dist_auc", "train_loss", "ArcFace_s"]:
                if m in row.index and pd.notna(row[m]):
                    metrics.append(f"{m}={row[m]:.4f}")
            print(f"  step ~{step:5d}: " + "  ".join(metrics))


if __name__ == "__main__":
    main()
