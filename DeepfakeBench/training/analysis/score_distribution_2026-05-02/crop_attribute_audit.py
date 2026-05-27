"""Crop-attribute audit on the 275 paired viso frames.

Goal: characterise the 144 both-missed pairs vs the 131 sometimes-caught pairs
along low-level image attributes (brightness, contrast, sharpness, edge density).
If both-missed has structurally different attributes, that's a different
intervention class than calibration.

For each of 275 unique sequences (raw + teams variant = 550 unique frames),
compute:
  - Brightness: mean luminance, std, percentiles
  - Sharpness: Laplacian variance (higher = sharper)
  - Edge density: Sobel edge magnitude mean
  - Contrast: std of luminance
  - Saturation: mean HSV.S
  - Skin-region heuristic: fraction of pixels in skin-color YCrCb range

Then label each pair by category (both_caught / raw_only / teams_only / both_missed
at τ=0.5, using P8A scores as primary), and compare distributions.

Outputs:
  outputs/crop_attributes.csv (per-frame attributes)
  outputs/crop_attributes_by_category.csv (per-pair category + attributes summary)
  outputs/figures/crop_attr_dist_<feature>.png (one per feature)
  outputs/figures/crop_attr_violin.png (multi-feature compact view)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import sobel

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FRAMES = OUT / "viso_full_paired"

PATTERN = re.compile(r"visomaster_enhanced_(raw|teams)__frame_(\d+)_(seq\d+)\.png")


def luminance(arr: np.ndarray) -> np.ndarray:
    return (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)


def laplacian_var(luma: np.ndarray) -> float:
    # 3x3 Laplacian kernel applied via convolve
    from scipy.ndimage import laplace
    lap = laplace(luma)
    return float(lap.var())


def sobel_edge_mean(luma: np.ndarray) -> float:
    sx = sobel(luma, axis=0)
    sy = sobel(luma, axis=1)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    return float(mag.mean())


def saturation_mean(rgb: np.ndarray) -> float:
    # Convert to HSV and take S
    from colorsys import rgb_to_hsv
    # Vectorised approximation
    rgb_n = rgb.astype(np.float32) / 255.0
    cmax = rgb_n.max(axis=2)
    cmin = rgb_n.min(axis=2)
    delta = cmax - cmin
    sat = np.where(cmax > 0, delta / (cmax + 1e-8), 0.0)
    return float(sat.mean())


def skin_mask_fraction(rgb: np.ndarray) -> float:
    # YCrCb conversion (BT.601)
    R, G, B = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 128.0
    Cb = (B - Y) * 0.564 + 128.0
    skin = (Cr >= 133) & (Cr <= 173) & (Cb >= 77) & (Cb <= 127)
    return float(skin.mean())


def per_frame_attributes(p: Path) -> Dict[str, float]:
    arr = np.asarray(Image.open(p).convert("RGB"))
    luma = luminance(arr)
    return {
        "h": int(arr.shape[0]),
        "w": int(arr.shape[1]),
        "luma_mean": float(luma.mean()),
        "luma_std": float(luma.std()),
        "luma_p10": float(np.percentile(luma, 10)),
        "luma_p90": float(np.percentile(luma, 90)),
        "laplacian_var": laplacian_var(luma),
        "sobel_edge_mean": sobel_edge_mean(luma),
        "saturation_mean": saturation_mean(arr),
        "skin_frac": skin_mask_fraction(arr),
    }


def main():
    print("[load] viso pairs + scoring...")
    pairs = pd.read_csv(OUT / "viso_pairs.csv")
    # Use P8A for the categorisation since that's the primary baseline.
    p8a_pairs = pairs[pairs["model"] == "P8A"].copy()
    p8a_pairs["category"] = np.where(
        (p8a_pairs["frame_prob_raw"] >= 0.5) & (p8a_pairs["frame_prob_teams"] >= 0.5), "both_caught",
        np.where((p8a_pairs["frame_prob_raw"] >= 0.5) & (p8a_pairs["frame_prob_teams"] < 0.5), "raw_only",
        np.where((p8a_pairs["frame_prob_raw"] < 0.5) & (p8a_pairs["frame_prob_teams"] >= 0.5), "teams_only",
                 "both_missed")))
    print("[category] P8A pair categories:", p8a_pairs["category"].value_counts().to_dict())

    # Multi-model "any-caught" category — where ANY of the 3 models catches either substrate
    pivot = pairs.pivot_table(index="seq_id", columns="model", values=["frame_prob_raw", "frame_prob_teams"], aggfunc="first")
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot["any_caught"] = (pivot >= 0.5).any(axis=1)
    pivot = pivot.reset_index()
    print(f"[category] cross-model: any_caught = {pivot['any_caught'].sum()}/{len(pivot)} pairs")

    print("[attrs] computing per-frame attributes for 550 frames...")
    rows = []
    files = sorted(FRAMES.glob("visomaster_enhanced_*.png"))
    for i, p in enumerate(files):
        m = PATTERN.search(p.name)
        if not m:
            continue
        subtype, frame_num, seq_id = m.groups()
        attrs = per_frame_attributes(p)
        attrs.update({"seq_id": seq_id, "subtype": subtype, "frame_num": int(frame_num), "filename": p.name})
        rows.append(attrs)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(files)}]")
    attr_df = pd.DataFrame(rows)
    attr_df.to_csv(OUT / "crop_attributes.csv", index=False)
    print(f"[attrs] computed {len(attr_df)} per-frame attributes")

    # Pivot to per-pair attributes (raw vs teams as columns)
    pair_attrs = attr_df.pivot_table(
        index="seq_id", columns="subtype",
        values=["luma_mean", "luma_std", "laplacian_var", "sobel_edge_mean", "saturation_mean", "skin_frac"],
        aggfunc="first"
    ).reset_index()
    pair_attrs.columns = [f"{a}_{b}" if b else a for a, b in pair_attrs.columns]

    # Merge with category
    merged = p8a_pairs[["seq_id", "category", "frame_prob_raw", "frame_prob_teams"]].merge(
        pair_attrs, on="seq_id", how="left"
    )
    merged.to_csv(OUT / "crop_attributes_by_category.csv", index=False)

    # Add cross-model any_caught flag
    merged = merged.merge(pivot[["seq_id", "any_caught"]], on="seq_id", how="left")
    merged["never_caught"] = ~merged["any_caught"]
    print(f"[merge] never_caught (no model, no substrate >= 0.5) = {merged['never_caught'].sum()}/{len(merged)} pairs")

    # ---- Statistical comparison: never_caught vs ever_caught
    print("\n=== Per-feature distribution: never_caught vs ever_caught ===")
    feature_cols = [c for c in merged.columns if c.endswith("_raw") or c.endswith("_teams")]
    feature_cols = [c for c in feature_cols if c not in ("frame_prob_raw", "frame_prob_teams")]
    summary = []
    from scipy.stats import mannwhitneyu
    for col in feature_cols:
        nc = merged[merged["never_caught"] == True][col].dropna()
        ec = merged[merged["never_caught"] == False][col].dropna()
        if len(nc) < 5 or len(ec) < 5:
            continue
        try:
            u, p = mannwhitneyu(nc, ec, alternative="two-sided")
        except Exception:
            u, p = float("nan"), float("nan")
        summary.append({
            "feature": col,
            "never_caught_mean": float(nc.mean()),
            "ever_caught_mean": float(ec.mean()),
            "delta_mean": float(nc.mean() - ec.mean()),
            "never_caught_n": int(len(nc)),
            "ever_caught_n": int(len(ec)),
            "mannwhitney_p": float(p),
        })
    summary_df = pd.DataFrame(summary).sort_values("mannwhitney_p")
    summary_df.to_csv(OUT / "crop_attribute_significance.csv", index=False)
    print(summary_df.to_string(index=False, float_format="%.4f"))

    # ---- Plots
    print("\n[plot] per-feature distribution (never_caught vs ever_caught)...")
    plot_features = ["luma_mean_raw", "luma_mean_teams", "laplacian_var_raw", "laplacian_var_teams",
                     "sobel_edge_mean_raw", "sobel_edge_mean_teams", "saturation_mean_raw", "skin_frac_raw"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, col in zip(axes.flat, plot_features):
        if col not in merged.columns:
            ax.axis("off")
            continue
        nc = merged[merged["never_caught"] == True][col].dropna()
        ec = merged[merged["never_caught"] == False][col].dropna()
        bins = np.linspace(merged[col].min(), merged[col].max(), 25)
        ax.hist(ec, bins=bins, alpha=0.55, color="green", label=f"ever_caught (n={len(ec)})", edgecolor="black")
        ax.hist(nc, bins=bins, alpha=0.55, color="red", label=f"never_caught (n={len(nc)})", edgecolor="black")
        try:
            from scipy.stats import mannwhitneyu
            _, p = mannwhitneyu(nc, ec, alternative="two-sided")
            sig = " *" if p < 0.05 else ""
            ax.set_title(f"{col}\np={p:.4f}{sig}", fontsize=10)
        except Exception:
            ax.set_title(col, fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Crop attribute distribution — never_caught (no model catches either substrate at τ=0.5) vs ever_caught", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "crop_attr_never_vs_ever.png", dpi=110)
    plt.close(fig)

    # ---- Score-vs-attribute scatter (P8A raw score vs feature)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, col in zip(axes.flat, plot_features):
        if col not in merged.columns:
            ax.axis("off")
            continue
        ax.scatter(merged[col], merged["frame_prob_raw"], s=10, alpha=0.55, color="steelblue", label="P8A raw_score")
        ax.scatter(merged[col], merged["frame_prob_teams"], s=10, alpha=0.55, color="orange", label="P8A teams_score")
        r_raw = merged[[col, "frame_prob_raw"]].corr().iloc[0, 1]
        r_teams = merged[[col, "frame_prob_teams"]].corr().iloc[0, 1]
        ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5)
        ax.set_title(f"{col}\nr(raw)={r_raw:.3f}  r(teams)={r_teams:.3f}", fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("P8A score")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("P8A raw and teams scores vs per-frame crop attributes (Pearson r in title)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "crop_attr_vs_score_p8a.png", dpi=110)
    plt.close(fig)

    print(f"\n[done] outputs in {OUT}")


if __name__ == "__main__":
    main()
