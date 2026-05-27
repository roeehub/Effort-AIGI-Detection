"""Validate the sharpness-predicts-catchability finding across all 3 models.

The crop-attribute audit (Diag #1 in the prior batch) found that pairs the model
NEVER catches in either substrate at tau=0.5 are systematically less sharp
(Laplacian variance 43.6 vs 71.4) than ever-caught pairs. That used a cross-model
"any model catches it" definition.

This script asks the per-model version: does the sharpness pattern hold for each
model's catch/miss categorisation independently?

For each model, compare laplacian_var distributions across the four pair
categories (both_caught, raw_only, teams_only, both_missed). Mann-Whitney p-values
for each pairwise comparison.

Output:
  outputs/sharpness_validation.csv
  outputs/figures/sharpness_per_model.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"


def main():
    print("[load] crop attributes + viso pairs...")
    attrs = pd.read_csv(OUT / "crop_attributes.csv")
    pairs = pd.read_csv(OUT / "viso_pairs.csv")

    # Pivot attrs to per-pair raw/teams Laplacian
    pair_attr = attrs.pivot_table(
        index="seq_id", columns="subtype",
        values=["laplacian_var", "sobel_edge_mean", "luma_mean", "luma_std"],
        aggfunc="first"
    ).reset_index()
    pair_attr.columns = [f"{a}_{b}" if b else a for a, b in pair_attr.columns]
    # Use mean of raw + teams Laplacian as per-pair sharpness summary
    pair_attr["laplacian_var_mean"] = (pair_attr["laplacian_var_raw"] + pair_attr["laplacian_var_teams"]) / 2

    summary_rows = []
    cat_order = ["both_caught", "raw_only", "teams_only", "both_missed"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, model in zip(axes, ["P8A", "P18T", "P18C"]):
        sub = pairs[pairs["model"] == model].copy()
        # Categorise at tau=0.5
        rh = sub["frame_prob_raw"] >= 0.5
        th = sub["frame_prob_teams"] >= 0.5
        sub["category"] = np.where(rh & th, "both_caught",
                          np.where(rh & ~th, "raw_only",
                          np.where(~rh & th, "teams_only", "both_missed")))
        sub = sub.merge(pair_attr[["seq_id", "laplacian_var_raw", "laplacian_var_teams", "laplacian_var_mean"]], on="seq_id", how="left")

        # For each category, distribution of laplacian_var_mean
        cat_data = {c: sub[sub["category"] == c]["laplacian_var_mean"].dropna() for c in cat_order}
        # Box plot
        positions = list(range(len(cat_order)))
        bp = ax.boxplot([cat_data[c].tolist() for c in cat_order], positions=positions, widths=0.6, patch_artist=True)
        colors = {"both_caught": "#2ca02c", "raw_only": "#1f77b4", "teams_only": "#ff7f0e", "both_missed": "#d62728"}
        for patch, c in zip(bp["boxes"], cat_order):
            patch.set_facecolor(colors[c])
            patch.set_alpha(0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{c}\n(n={len(cat_data[c])})" for c in cat_order], fontsize=8)
        ax.set_title(f"{model} — Laplacian variance per pair (mean of raw+teams)")
        ax.set_ylabel("laplacian_var_mean")
        ax.grid(alpha=0.3, axis="y")

        # Pairwise MWU vs both_missed
        for c in cat_order:
            if c == "both_missed":
                continue
            a = cat_data[c]
            b = cat_data["both_missed"]
            if len(a) < 5 or len(b) < 5:
                p, u = float("nan"), float("nan")
            else:
                u, p = mannwhitneyu(a, b, alternative="two-sided")
            summary_rows.append({
                "model": model, "category_A": c, "category_B": "both_missed",
                "n_A": int(len(a)), "n_B": int(len(b)),
                "median_A": float(np.median(a)) if len(a) else float("nan"),
                "median_B": float(np.median(b)) if len(b) else float("nan"),
                "delta_median": float(np.median(a) - np.median(b)) if len(a) and len(b) else float("nan"),
                "MWU_p": float(p),
                "significant_5pct": bool(p < 0.05),
            })

    fig.suptitle("Sharpness per pair, per category, per model — does the sharpness pattern hold across all three models?")
    fig.tight_layout()
    fig.savefig(FIG / "sharpness_per_model.png", dpi=120)
    plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "sharpness_validation.csv", index=False)
    print("\n=== Sharpness validation: laplacian_var_mean per category, per model ===")
    print("(Comparison: each category vs both_missed within the model)\n")
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False, float_format="%.4f"))

    print(f"\n[done] outputs in {OUT}")


if __name__ == "__main__":
    main()
