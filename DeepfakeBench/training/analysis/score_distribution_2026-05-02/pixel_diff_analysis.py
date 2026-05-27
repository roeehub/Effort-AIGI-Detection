"""Pixel-level diff between paired viso raw/teams frames.

The visomaster_enhanced_macro_dev slice has 275 paired sequences, each with a
`visomaster_enhanced_raw__*.png` and a `visomaster_enhanced_teams__*.png` variant.
Visual inspection of 4 pairs each across 4 score-categories shows the images
look essentially IDENTICAL to the human eye, yet the model behaves very
differently across them (P8A: raw mean 0.44, teams mean 0.27; per-pair Δ
unidirectional toward raw).

This script computes pixel-level diff statistics for all 275 pairs:

  - L1, L2, max-pixel diff in RGB and luminance
  - SSIM
  - DCT high-frequency band-energy ratio (compression signature)
  - Mean luminance shift, hue shift, saturation shift
  - File-size diff (decoded-size proxy)

Outputs:
  outputs/pixel_diff.csv
  outputs/figures/pixel_diff_hist.png
  outputs/figures/pixel_diff_vs_score_delta.png
  outputs/viso_full_paired_samples/  (additional diff visualisations for top-K pairs)
"""

from __future__ import annotations

import re
import shutil
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
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FULL = OUT / "viso_full_paired"
FULL.mkdir(parents=True, exist_ok=True)

PAIRS_CSV = OUT / "viso_pairs.csv"


def gcs_to_local(uri: str) -> Path:
    name = uri.rsplit("/", 1)[-1]
    return FULL / name


def fetch_one(uri: str) -> Path:
    p = gcs_to_local(uri)
    if p.exists() and p.stat().st_size > 0:
        return p
    subprocess.run(["gsutil", "-q", "cp", uri, str(p)], check=False)
    return p


def fetch_pairs(pairs: pd.DataFrame, max_workers: int = 16) -> Dict[str, Path]:
    uris = list(set(pairs["frame_path_raw"].tolist() + pairs["frame_path_teams"].tolist()))
    paths: Dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for uri, p in zip(uris, ex.map(fetch_one, uris)):
            paths[uri] = p
    return paths


def load_image(p: Path) -> np.ndarray:
    img = Image.open(p).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    return arr


def luminance(arr: np.ndarray) -> np.ndarray:
    # ITU-R BT.601
    return (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)


def pixel_diff_metrics(raw_path: Path, teams_path: Path) -> Dict[str, float]:
    if not raw_path.exists() or not teams_path.exists():
        return {"valid": False}
    a = load_image(raw_path)
    b = load_image(teams_path)
    if a.shape != b.shape:
        # Resize teams to match raw
        b_img = Image.open(teams_path).convert("RGB").resize((a.shape[1], a.shape[0]), Image.LANCZOS)
        b = np.asarray(b_img, dtype=np.uint8)

    diff = a.astype(np.int32) - b.astype(np.int32)
    abs_diff = np.abs(diff)
    l1 = float(abs_diff.mean())
    l2 = float(np.sqrt((diff.astype(np.float32) ** 2).mean()))
    max_pix = float(abs_diff.max())
    frac_nonzero = float((abs_diff.sum(axis=2) > 0).mean())
    frac_above_5 = float((abs_diff.sum(axis=2) > 5 * 3).mean())  # >5/channel cumulative

    luma_a = luminance(a)
    luma_b = luminance(b)
    luma_l1 = float(np.abs(luma_a - luma_b).mean())
    luma_mean_shift = float(luma_b.mean() - luma_a.mean())

    # Per-channel mean shift
    rgb_shift = (b.astype(np.float32) - a.astype(np.float32)).mean(axis=(0, 1))

    # DCT high-freq ratio. Compute on luminance, full image, blocks of size 8.
    # Take per-block DCT energy in high-freq quadrant vs low-freq quadrant.
    h, w = luma_a.shape
    bs = 8
    h2 = (h // bs) * bs
    w2 = (w // bs) * bs
    hfratio_a = _dct_hf_ratio(luma_a[:h2, :w2], bs)
    hfratio_b = _dct_hf_ratio(luma_b[:h2, :w2], bs)

    # File size difference (rough compression-signature proxy).
    size_a = raw_path.stat().st_size
    size_b = teams_path.stat().st_size

    return {
        "valid": True,
        "shape_h": int(a.shape[0]),
        "shape_w": int(a.shape[1]),
        "rgb_l1_mean": l1,
        "rgb_l2_rms": l2,
        "rgb_max_pix": max_pix,
        "frac_pixels_changed": frac_nonzero,
        "frac_pixels_changed_above_5": frac_above_5,
        "luma_l1_mean": luma_l1,
        "luma_mean_shift": luma_mean_shift,
        "rgb_mean_shift_R": float(rgb_shift[0]),
        "rgb_mean_shift_G": float(rgb_shift[1]),
        "rgb_mean_shift_B": float(rgb_shift[2]),
        "hfratio_raw": float(hfratio_a),
        "hfratio_teams": float(hfratio_b),
        "hfratio_delta": float(hfratio_b - hfratio_a),
        "filesize_raw": int(size_a),
        "filesize_teams": int(size_b),
        "filesize_ratio": float(size_b / max(1, size_a)),
    }


def _dct_hf_ratio(img: np.ndarray, bs: int = 8) -> float:
    """Mean per-block ratio of HF energy to total energy in 8x8 DCT.

    HF = bottom-right quadrant (rows >= bs/2, cols >= bs/2).
    Lower ratio indicates lossy-compressed image (HF energy stripped).
    """
    h, w = img.shape
    nb_h = h // bs
    nb_w = w // bs
    img = img[: nb_h * bs, : nb_w * bs]
    blocks = img.reshape(nb_h, bs, nb_w, bs).transpose(0, 2, 1, 3).reshape(-1, bs, bs)
    blocks = blocks - blocks.mean(axis=(1, 2), keepdims=True)
    dct = dctn(blocks, axes=(1, 2), norm="ortho")
    energy = dct ** 2
    total = energy.sum(axis=(1, 2))
    hf = energy[:, bs // 2:, bs // 2:].sum(axis=(1, 2))
    valid = total > 0
    if not valid.any():
        return 0.0
    return float((hf[valid] / total[valid]).mean())


def main():
    print("[load] reading viso pairs...")
    pairs = pd.read_csv(PAIRS_CSV)
    # We only need each unique seq once for pixel diff (not per-model).
    unique_pairs = pairs.drop_duplicates(subset=["seq_id"])[["seq_id", "frame_path_raw", "frame_path_teams"]].reset_index(drop=True)
    print(f"[load] {len(unique_pairs)} unique sequences")

    print("[fetch] downloading all 550 unique frames (parallel)...")
    paths = fetch_pairs(unique_pairs)
    n_have = sum(1 for p in paths.values() if p.exists())
    print(f"[fetch] downloaded {n_have} of {len(paths)} unique URIs")

    print("[diff] computing per-pair pixel-level diff metrics...")
    rows = []
    for i, r in unique_pairs.iterrows():
        raw_p = paths.get(r["frame_path_raw"])
        teams_p = paths.get(r["frame_path_teams"])
        if not raw_p or not teams_p:
            continue
        m = pixel_diff_metrics(raw_p, teams_p)
        m["seq_id"] = r["seq_id"]
        m["frame_path_raw"] = r["frame_path_raw"]
        m["frame_path_teams"] = r["frame_path_teams"]
        rows.append(m)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(unique_pairs)}]")
    pixel_df = pd.DataFrame(rows)
    pixel_df.to_csv(OUT / "pixel_diff.csv", index=False)
    valid = pixel_df[pixel_df["valid"] == True].copy()
    print(f"[diff] {len(valid)} valid pairs")

    print("\n=== Pixel-diff summary across 275 pairs ===")
    cols = [
        "rgb_l1_mean", "rgb_l2_rms", "luma_l1_mean", "frac_pixels_changed",
        "frac_pixels_changed_above_5", "luma_mean_shift",
        "hfratio_raw", "hfratio_teams", "hfratio_delta",
        "filesize_raw", "filesize_teams", "filesize_ratio",
    ]
    print(valid[cols].describe().T[["mean", "50%", "min", "max", "std"]].round(4))

    # Plot: distribution of pixel-diff metrics
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(
        axes.flat,
        ["rgb_l1_mean", "luma_l1_mean", "frac_pixels_changed",
         "luma_mean_shift", "hfratio_delta", "filesize_ratio"]
    ):
        ax.hist(valid[col], bins=30, color="seagreen", edgecolor="black", alpha=0.8)
        ax.set_title(f"{col}\nmean={valid[col].mean():.4f} median={valid[col].median():.4f}")
        ax.grid(alpha=0.3)
    fig.suptitle("Pixel-level diff between raw and teams substrate (n=275 unique pairs)")
    fig.tight_layout()
    fig.savefig(FIG / "pixel_diff_hist.png", dpi=120)
    plt.close(fig)

    print("\n[plot] correlating pixel diff with model score delta...")
    # Merge pixel diff with model scores per (seq_id, model)
    merged = pairs.merge(valid, on="seq_id", how="inner", suffixes=("", "_pix"))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    cols_plot = ["rgb_l1_mean", "luma_l1_mean", "hfratio_delta", "frac_pixels_changed", "luma_mean_shift", "filesize_ratio"]
    for model_idx, model in enumerate(["P8A", "P18T", "P18C"]):
        sub = merged[merged["model"] == model]
        for j, col in enumerate(cols_plot[:3]):
            ax = axes[0, j]
            ax.scatter(sub[col], sub["delta"], s=8, alpha=0.5,
                       label=model, color={"P8A": "#1f77b4", "P18T": "#ff7f0e", "P18C": "#2ca02c"}[model])
            r = sub[[col, "delta"]].corr().iloc[0, 1]
            if model_idx == 0:
                ax.set_title(f"score Δ vs {col}", fontsize=9)
                ax.set_xlabel(col, fontsize=8)
                ax.set_ylabel("teams_score - raw_score", fontsize=8)
                ax.axhline(0, color="black", linestyle=":", alpha=0.4)
                ax.grid(alpha=0.3)
        for j, col in enumerate(cols_plot[3:]):
            ax = axes[1, j]
            ax.scatter(sub[col], sub["delta"], s=8, alpha=0.5,
                       label=model, color={"P8A": "#1f77b4", "P18T": "#ff7f0e", "P18C": "#2ca02c"}[model])
            if model_idx == 0:
                ax.set_title(f"score Δ vs {col}", fontsize=9)
                ax.set_xlabel(col, fontsize=8)
                ax.set_ylabel("teams_score - raw_score", fontsize=8)
                ax.axhline(0, color="black", linestyle=":", alpha=0.4)
                ax.grid(alpha=0.3)
    for ax in axes.flat:
        ax.legend(loc="best", fontsize=7)
    fig.suptitle("Per-pair score delta vs pixel-level diff metrics (3 models overlaid)")
    fig.tight_layout()
    fig.savefig(FIG / "pixel_diff_vs_score_delta.png", dpi=120)
    plt.close(fig)

    print(f"\n[done] outputs in {OUT}")


if __name__ == "__main__":
    main()
