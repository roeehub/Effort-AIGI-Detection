"""Job B — score-distribution forensics across the chain.

Histograms + quantile decomposition of P8A vs P18T vs P22 step8k per suite.
Tests the "wider score distribution" hypothesis structurally.
"""
from __future__ import annotations

import sys; sys.path.insert(0, "analysis/p22_eval_2026-05-02/cpu_followups/scripts")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import FIG, OUT, SUITE_LABEL, load_per_frame_scores

CKPTS = ["P8A_REFERENCE_STEP5000", "P18T_GRL_TREATMENT_STEP4000",
         "P22_AUG_STEP1000", "P22_AUG_STEP4000", "P22_AUG_STEP8000"]

# Show all 9 suites where data exists
SUITES_PLOT = ["teams_real_all_dev", "teams_real_all_lockbox", "teams_real_dor_dev",
               "teams_real_lighting_extreme_dev", "teams_real_poor_quality_dev",
               "teams_fake_all_dev", "teams_fake_all_lockbox",
               "visomaster_enhanced_macro_dev", "deeplive_enhanced_dev"]


def quantile_table():
    rows = []
    for ckpt in CKPTS:
        for suite in SUITES_PLOT:
            df = load_per_frame_scores(ckpt, suite)
            if df is None:
                continue
            s = df["frame_prob"].to_numpy()
            row = {"checkpoint": ckpt, "suite": suite, "label": SUITE_LABEL[suite],
                   "n": len(s),
                   "min": float(s.min()), "p1": float(np.quantile(s, 0.01)),
                   "p5": float(np.quantile(s, 0.05)), "p10": float(np.quantile(s, 0.10)),
                   "p25": float(np.quantile(s, 0.25)), "p50": float(np.quantile(s, 0.50)),
                   "p75": float(np.quantile(s, 0.75)), "p90": float(np.quantile(s, 0.90)),
                   "p95": float(np.quantile(s, 0.95)), "p99": float(np.quantile(s, 0.99)),
                   "max": float(s.max()),
                   "iqr": float(np.quantile(s, 0.75) - np.quantile(s, 0.25)),
                   "p99_minus_p1": float(np.quantile(s, 0.99) - np.quantile(s, 0.01))}
            rows.append(row)
    return pd.DataFrame(rows)


def plot_histograms_grid():
    fig, axes = plt.subplots(len(SUITES_PLOT), len(CKPTS), figsize=(22, 4 * len(SUITES_PLOT)))
    for i, suite in enumerate(SUITES_PLOT):
        for j, ckpt in enumerate(CKPTS):
            ax = axes[i, j]
            df = load_per_frame_scores(ckpt, suite)
            if df is None:
                ax.set_axis_off()
                ax.set_title(f"{suite}\n{ckpt}\n(missing)", fontsize=8)
                continue
            s = df["frame_prob"].to_numpy()
            color = "tab:red" if SUITE_LABEL[suite] == 1 else "tab:blue"
            ax.hist(s, bins=40, color=color, alpha=0.7, edgecolor="black", linewidth=0.3)
            ax.axvline(0.5, color="grey", linestyle=":", alpha=0.5)
            ax.set_xlim(0, 1)
            ax.set_title(f"{suite}\n{ckpt}\nn={len(s)}, IQR={np.quantile(s,0.75)-np.quantile(s,0.25):.3f}",
                         fontsize=8)
            if j == 0:
                ax.set_ylabel("count")
            if i == len(SUITES_PLOT) - 1:
                ax.set_xlabel("score")
    plt.suptitle("Per-frame score distributions: P8A vs P18T vs P22 1k/4k/8k\n"
                 "(blue = real suites, red = fake suites)", fontsize=12, y=1.0)
    plt.tight_layout()
    out = FIG / "02_score_distributions.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    return out


def plot_iqr_summary(qtable):
    """Single bar chart: IQR per (suite, checkpoint), grouping by suite."""
    fig, ax = plt.subplots(figsize=(16, 7))
    pivot = qtable.pivot(index="suite", columns="checkpoint", values="iqr")
    pivot = pivot[CKPTS]  # column order
    pivot.plot.bar(ax=ax, width=0.8)
    ax.set_ylabel("IQR (p75 - p25)")
    ax.set_title("Score distribution width (IQR) per suite × checkpoint\n"
                 "Wider IQR = more spread / less saturation")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    out = FIG / "02_iqr_summary.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    qtable = quantile_table()
    qtable.to_csv(OUT / "02_score_quantiles.csv", index=False)

    print("=" * 80)
    print("IQR per (suite, checkpoint) — wider = more spread distribution")
    print("=" * 80)
    pivot = qtable.pivot(index="suite", columns="checkpoint", values="iqr")
    print(pivot[CKPTS].to_string(float_format=lambda x: f"{x:.3f}"))

    print("\nMedian (p50) per (suite, checkpoint):")
    pmed = qtable.pivot(index="suite", columns="checkpoint", values="p50")
    print(pmed[CKPTS].to_string(float_format=lambda x: f"{x:.3f}"))

    plot_histograms_grid()
    plot_iqr_summary(qtable)
    print(f"\nWrote: {OUT}/02_score_quantiles.csv")
    print(f"Wrote: {FIG}/02_score_distributions.png")
    print(f"Wrote: {FIG}/02_iqr_summary.png")


if __name__ == "__main__":
    main()
