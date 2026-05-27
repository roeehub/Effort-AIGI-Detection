"""Sample training-data frames and compute the same crop attributes used in
the eval audit, then compare distributions side-by-side.

Pulls ~50 frames each from major training methods. Total ~400 frames.
Computes per-frame: luma_mean, luma_std, laplacian_var, sobel_edge_mean,
saturation_mean, skin_frac. Compares to the 550 eval viso frames already cached.

Output:
  outputs/train_attributes.csv (per-frame for training samples)
  outputs/train_vs_eval_attribute_summary.csv
  outputs/figures/train_vs_eval_attr_overlay.png  (per-feature overlaid distributions)
  outputs/figures/train_vs_eval_attr_box.png      (per-feature side-by-side boxes)
"""

from __future__ import annotations

import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import laplace, sobel

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
DEST = OUT / "train_data_samples"
DEST.mkdir(parents=True, exist_ok=True)


def list_blobs(bucket_prefix: str, limit: int = 100) -> List[str]:
    res = subprocess.run(["gsutil", "ls", bucket_prefix], capture_output=True, text=True)
    lines = [l.strip() for l in res.stdout.split("\n") if l.strip()]
    return lines[:limit]


def fetch(uri: str, dest_dir: Path) -> Path:
    # Make a unique filename derived from the URI's sample directory + frame name
    parts = uri.split("/")
    if "samples" in parts:
        sample_idx = parts.index("samples") + 1
        sample_id = parts[sample_idx]
    else:
        sample_id = "unknown"
    frame_name = parts[-1]
    name = f"{sample_id}__{frame_name}"
    p = dest_dir / name
    if p.exists() and p.stat().st_size > 0:
        return p
    subprocess.run(["gsutil", "-q", "cp", uri, str(p)], check=False)
    return p


def luminance(arr: np.ndarray) -> np.ndarray:
    return (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)


def per_frame_attrs(p: Path) -> Dict[str, float]:
    arr = np.asarray(Image.open(p).convert("RGB"))
    luma = luminance(arr)
    sx = sobel(luma, axis=0); sy = sobel(luma, axis=1)
    sobel_mag = np.sqrt(sx**2 + sy**2)
    rgb_n = arr.astype(np.float32) / 255.0
    cmax = rgb_n.max(axis=2); cmin = rgb_n.min(axis=2)
    sat = np.where(cmax > 0, (cmax - cmin) / (cmax + 1e-8), 0.0)
    R, G, B = arr[..., 0].astype(np.float32), arr[..., 1].astype(np.float32), arr[..., 2].astype(np.float32)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 128.0
    Cb = (B - Y) * 0.564 + 128.0
    skin = (Cr >= 133) & (Cr <= 173) & (Cb >= 77) & (Cb <= 127)
    return {
        "h": int(arr.shape[0]), "w": int(arr.shape[1]),
        "luma_mean": float(luma.mean()), "luma_std": float(luma.std()),
        "laplacian_var": float(laplace(luma).var()),
        "sobel_edge_mean": float(sobel_mag.mean()),
        "saturation_mean": float(sat.mean()),
        "skin_frac": float(skin.mean()),
        "filesize": int(p.stat().st_size),
    }


SOURCES = {
    # Training base viso (different algorithms)
    "train_visomaster_CSCS":             ("gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_CSCS_", "frames/fake/frame_0001.png", 50),
    "train_visomaster_GhostFace_v3":     ("gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_GhostFace-v3_", "frames/fake/frame_0001.png", 50),
    "train_visomaster_SimSwap512":       ("gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_SimSwap512_", "frames/fake/frame_0001.png", 50),
    "train_visomaster_Inswapper128":     ("gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_Inswapper128_", "frames/fake/frame_0001.png", 50),
    "train_visomaster_InStyleSwapper":   ("gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_InStyleSwapper256-A_", "frames/fake/frame_0001.png", 50),
    # Training enhanced viso (different enhancers)
    "train_visomaster_enhanced_codeformer": ("gs://visomaster-enhanced-face-cropped/samples/visomaster_CSCS_", "_enhanced_codeformer/frames/fake/frame_0001.png", 30),
    "train_visomaster_enhanced_gfpgan":     ("gs://visomaster-enhanced-face-cropped/samples/visomaster_CSCS_", "_enhanced_gfpgan/frames/fake/frame_0001.png", 30),
    # Training reals
    "train_realpool":                  ("gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_CSCS_", "frames/real/frame_0001.png", 50),
}


def collect_uris() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for name, (prefix, suffix, limit) in SOURCES.items():
        # List directories under prefix
        bucket_prefix = prefix.rsplit("/", 1)[0] + "/"
        res = subprocess.run(["gsutil", "ls", bucket_prefix], capture_output=True, text=True)
        dirs = [l.strip() for l in res.stdout.split("\n") if l.strip().startswith(prefix)]
        # Trim to limit
        dirs = dirs[:limit]
        # Build full URIs to frame_0001.png
        uris = []
        for d in dirs:
            # d ends with /
            if "_enhanced_" in suffix:
                # special handling for enhanced - the "suffix" replaces _NNNN/ with _NNNN_enhanced_X/
                base = d.rstrip("/")
                # base is "visomaster_CSCS_00007" → uri = "visomaster_CSCS_00007_enhanced_codeformer/frames/fake/frame_0001.png"
                # but since prefix also contains the bucket, we need to match the suffix style
                if "_enhanced_" in d:
                    # already an enhanced sample dir
                    uris.append(d + "frames/fake/frame_0001.png")
                else:
                    uris.append(d.rstrip("/") + suffix)
            else:
                uris.append(d + suffix)
        out[name] = uris
    return out


def main():
    print("[discover] listing training sample directories...")
    grouped = collect_uris()
    for k, v in grouped.items():
        print(f"  {k}: {len(v)} URIs")

    print("\n[fetch] downloading samples...")
    all_uris = []
    for group, uris in grouped.items():
        gd = DEST / group
        gd.mkdir(parents=True, exist_ok=True)
        for u in uris:
            all_uris.append((u, gd))
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(lambda args: fetch(args[0], args[1]), all_uris))

    print("\n[attrs] computing per-frame attributes...")
    rows = []
    for group, gd in [(g, DEST / g) for g in grouped]:
        files = list(gd.glob("*.png")) + list(gd.glob("*.jpg"))
        for p in files:
            try:
                attrs = per_frame_attrs(p)
                attrs["group"] = group
                attrs["filename"] = p.name
                rows.append(attrs)
            except Exception as exc:
                print(f"  [skip] {p.name}: {exc}")
    train_attrs = pd.DataFrame(rows)
    train_attrs.to_csv(OUT / "train_attributes.csv", index=False)
    print(f"[attrs] {len(train_attrs)} training-frame attributes computed")

    # Eval attributes (already computed)
    eval_attrs = pd.read_csv(OUT / "crop_attributes.csv")
    eval_attrs["group"] = eval_attrs["subtype"].apply(lambda s: f"eval_visomaster_{s}")

    # Combined
    feature_cols = ["luma_mean", "luma_std", "laplacian_var", "sobel_edge_mean", "saturation_mean", "skin_frac", "h"]
    combined = pd.concat([
        train_attrs[["group"] + feature_cols],
        eval_attrs[["group"] + feature_cols],
    ], ignore_index=True)
    summary = combined.groupby("group")[feature_cols].agg(["count", "mean", "median", "std"]).round(3)
    summary.to_csv(OUT / "train_vs_eval_attribute_summary.csv")
    print("\n=== Summary by group ===")
    print(summary.to_string())

    # ── Plot: per-feature overlaid distributions
    print("\n[plot] feature distributions...")
    plot_features = ["luma_mean", "laplacian_var", "sobel_edge_mean", "saturation_mean", "skin_frac", "h"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    groups_to_plot = list(combined["group"].unique())
    cmap = plt.get_cmap("tab20")
    colors = {g: cmap(i % 20) for i, g in enumerate(groups_to_plot)}
    for ax, col in zip(axes.flat, plot_features):
        for group in groups_to_plot:
            data = combined[combined["group"] == group][col].dropna()
            if data.empty:
                continue
            # Use percentile range to avoid extreme outliers in plots
            lo, hi = combined[col].quantile([0.01, 0.99])
            bins = np.linspace(lo, hi, 30)
            ax.hist(data, bins=bins, alpha=0.4, color=colors[group], label=f"{group} (n={len(data)})", density=True)
        ax.set_title(col)
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(alpha=0.3)
    fig.suptitle("Train vs Eval — per-feature density distributions")
    fig.tight_layout()
    fig.savefig(FIG / "train_vs_eval_attr_overlay.png", dpi=110)
    plt.close(fig)

    # ── Plot: per-feature boxplots side-by-side
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, col in zip(axes.flat, plot_features):
        data_per_group = [combined[combined["group"] == g][col].dropna() for g in groups_to_plot]
        bp = ax.boxplot(data_per_group, tick_labels=groups_to_plot, patch_artist=True)
        for patch, g in zip(bp["boxes"], groups_to_plot):
            patch.set_facecolor(colors[g])
            patch.set_alpha(0.6)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Train vs Eval — per-feature box distributions")
    fig.tight_layout()
    fig.savefig(FIG / "train_vs_eval_attr_box.png", dpi=110)
    plt.close(fig)

    print(f"\n[done] outputs in {OUT}")


if __name__ == "__main__":
    main()
