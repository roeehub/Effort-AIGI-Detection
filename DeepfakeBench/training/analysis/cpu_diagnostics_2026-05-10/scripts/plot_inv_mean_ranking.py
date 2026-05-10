"""Bar plot of L11 inv_mean across all 23 trained ckpts.

Reads outputs/L11_inv_mean_with_t4.csv and produces a ranked bar plot
with T4 ckpts highlighted. Shows the prior 0.0349 ceiling as a reference
line.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
INPUT = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-10/outputs/L11_inv_mean_with_t4.csv"
OUT = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-10/outputs/L11_inv_mean_ranking.png"


def main():
    df = pd.read_csv(INPUT).sort_values("inv_mean", ascending=True)  # asc for horizontal bar plot
    n = len(df)
    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.35)))

    # Color: T4_L1 (orange), T4_L2 (red), anchors P8A/E2B (blue), other (gray)
    def color_for(name):
        if name.startswith("T4_L1"):
            return "#ff7f0e"  # orange
        if name.startswith("T4_L2"):
            return "#d62728"  # red
        if name in ("P8A", "E2B"):
            return "#1f77b4"  # blue
        return "#7f7f7f"  # gray

    colors = [color_for(c) for c in df["ckpt"]]
    bars = ax.barh(df["ckpt"], df["inv_mean"], color=colors, edgecolor="black", linewidth=0.5)

    # Annotate values
    for bar, val in zip(bars, df["inv_mean"]):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    # Reference lines
    prior_ceiling = df[~df["ckpt"].str.startswith("T4")]["inv_mean"].max()
    ax.axvline(prior_ceiling, color="red", linestyle="--", linewidth=1.5, label=f"Prior ceiling (0.0349)")
    p8a_baseline = df[df["ckpt"] == "P8A"]["inv_mean"].iloc[0] if (df["ckpt"] == "P8A").any() else None
    if p8a_baseline is not None:
        ax.axvline(p8a_baseline, color="#1f77b4", linestyle=":", linewidth=1.5, label=f"P8A baseline ({p8a_baseline:.4f})")

    ax.set_xlabel("L11 inv_mean = forgery_AUC − mean(shortcut_AUCs)")
    ax.set_title("L11 inv_mean across 23 trained ckpts (T4 highlighted)\nHigher = encoder more invariant to shortcut axes while preserving forgery signal")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_xlim(0, max(df["inv_mean"]) * 1.15)

    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT}")
    print()
    print("Color legend:")
    print("  orange = T4-λ1.0 (mainline)")
    print("  red    = T4-λ2.0 (sensitivity hedge)")
    print("  blue   = anchors (P8A, E2B)")
    print("  gray   = other prior ckpts")


if __name__ == "__main__":
    raise SystemExit(main())
