"""Torch-free TSNE + plot pass for the embedding triptych.

The full triptych.py crashes (silent SIGSEGV) inside sklearn TSNE on this
Mac when torch is loaded in the same process — Apple Accelerate / openMP
conflict between PyTorch and sklearn. Since features are already cached
to .npz by triptych.py, we run TSNE + plotting in a fresh torch-free
process here.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


def load_cached(cache_dir: Path, label: str, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    p = cache_dir / f"triptych_features__{label}__n{n_samples}.npz"
    if not p.exists():
        raise FileNotFoundError(f"missing cache: {p}")
    z = np.load(p, allow_pickle=False)
    return z["features"], z["valid_idx"]


def _color_by_label(df: pd.DataFrame) -> Tuple[np.ndarray, dict]:
    cmap = {"real": "#4477aa", "fake": "#cc6677"}
    if "label" not in df.columns:
        return np.array(["#888888"] * len(df)), {"all": "#888888"}
    colors = np.array([cmap.get(str(v), "#888888") for v in df["label"].astype(str)])
    return colors, cmap


def _color_by_capture_mode(df: pd.DataFrame) -> Tuple[np.ndarray, dict]:
    palette = ["#4477aa", "#cc6677", "#117733", "#ddcc77", "#882255", "#88ccee"]
    if "clip_capture_mode" not in df.columns:
        return np.array(["#888888"] * len(df)), {"n/a": "#888888"}
    modes = df["clip_capture_mode"].fillna("unknown").astype(str)
    unique = sorted(modes.unique())
    cmap = {m: palette[i % len(palette)] for i, m in enumerate(unique)}
    colors = np.array([cmap[m] for m in modes])
    return colors, cmap


def _color_by_face_size(df: pd.DataFrame) -> Tuple[np.ndarray, dict]:
    palette = ["#4477aa", "#117733", "#ddcc77", "#cc6677"]
    if "face_pixel_area" not in df.columns:
        return np.array(["#888888"] * len(df)), {"n/a": "#888888"}
    fa = df["face_pixel_area"].fillna(-1).astype(float).to_numpy()
    valid = fa > 0
    quantiles = np.quantile(fa[valid], [0.25, 0.5, 0.75]) if valid.sum() > 4 else [0, 0, 0]
    bucket = np.full(len(fa), -1, dtype=int)
    bucket[valid & (fa <= quantiles[0])] = 0
    bucket[valid & (fa > quantiles[0]) & (fa <= quantiles[1])] = 1
    bucket[valid & (fa > quantiles[1]) & (fa <= quantiles[2])] = 2
    bucket[valid & (fa > quantiles[2])] = 3
    labels = {
        0: f"<= {quantiles[0]:.0f}",
        1: f"{quantiles[0]:.0f}-{quantiles[1]:.0f}",
        2: f"{quantiles[1]:.0f}-{quantiles[2]:.0f}",
        3: f"> {quantiles[2]:.0f}",
    }
    cmap = {labels[i]: palette[i] for i in range(4)}
    cmap["missing"] = "#888888"
    colors = np.array([
        palette[b] if b >= 0 else "#888888"
        for b in bucket
    ])
    return colors, cmap


def render_grid(
    coords_per_ckpt: Dict[str, np.ndarray],
    metadata_df: pd.DataFrame,
    output_dir: Path,
    reducer_name: str,
) -> None:
    ckpt_labels = list(coords_per_ckpt.keys())
    n_rows = len(ckpt_labels)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4.5 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    color_funcs = [
        ("real / fake", _color_by_label),
        ("clip_capture_mode", _color_by_capture_mode),
        ("face_pixel_area bucket", _color_by_face_size),
    ]

    for r, label in enumerate(ckpt_labels):
        coords = coords_per_ckpt[label]
        for c, (col_title, fn) in enumerate(color_funcs):
            ax = axes[r, c]
            colors, cmap = fn(metadata_df)
            ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=10, alpha=0.6, edgecolors="none")
            ax.set_title(f"{label}  |  {col_title}")
            ax.set_xticks([])
            ax.set_yticks([])
            handles = [
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=v, label=str(k), markersize=6)
                for k, v in cmap.items()
            ]
            ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.7, markerscale=1.2)

    fig.suptitle(f"[CLS] feature embedding ({reducer_name.upper()}) — n={len(metadata_df)} frames per row", y=1.0)
    fig.tight_layout()
    out_png = output_dir / f"triptych_grid_{reducer_name}.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"PNG written: {out_png}")


def write_coords_csv(
    coords_per_ckpt: Dict[str, np.ndarray],
    metadata_df: pd.DataFrame,
    output_dir: Path,
    reducer_name: str,
) -> None:
    rows = []
    for label, coords in coords_per_ckpt.items():
        for i, (x, y) in enumerate(coords):
            base = metadata_df.iloc[i].to_dict()
            base.update({
                "checkpoint": label,
                f"{reducer_name}_x": float(x),
                f"{reducer_name}_y": float(y),
            })
            for drop_key in ("arcface_embed", "clip_embed"):
                base.pop(drop_key, None)
            rows.append(base)
    out_csv = output_dir / f"triptych_coords_{reducer_name}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"CSV written: {out_csv}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--labels", nargs="+", required=True,
                    help="Checkpoint labels matching cached files, in row order")
    ap.add_argument("--n_samples", type=int, default=800)
    ap.add_argument("--perplexity", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = pd.read_csv(args.output_dir / "sampled_frames.csv")
    print(f"Loaded metadata: {len(metadata_df)} rows")

    feats_per_ckpt: Dict[str, np.ndarray] = {}
    valid_per_ckpt: Dict[str, np.ndarray] = {}
    common_valid = None
    for label in args.labels:
        f, v = load_cached(args.cache_dir, label, args.n_samples)
        feats_per_ckpt[label] = f
        valid_per_ckpt[label] = v
        common_valid = v if common_valid is None else np.intersect1d(common_valid, v)
        print(f"[{label}] features={f.shape}, valid={len(v)}")

    if common_valid is None or len(common_valid) == 0:
        raise RuntimeError("no common valid frames across checkpoints")
    print(f"Common valid frames: {len(common_valid)}")

    metadata_for_each = metadata_df.iloc[common_valid].reset_index(drop=True)

    coords_per_ckpt: Dict[str, np.ndarray] = {}
    perp = min(args.perplexity, max(5, len(common_valid) // 4))
    for label in args.labels:
        feats = feats_per_ckpt[label]
        valid = valid_per_ckpt[label]
        pos_in_features = {int(orig_idx): row for row, orig_idx in enumerate(valid)}
        rows = [pos_in_features[int(c)] for c in common_valid]
        f_aligned = feats[rows]
        print(f"[{label}] running TSNE on {f_aligned.shape}, perplexity={perp}")
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            random_state=args.seed,
            init="pca",
            learning_rate="auto",
            n_jobs=1,
        )
        coords = tsne.fit_transform(f_aligned)
        coords_per_ckpt[label] = coords
        print(f"[{label}] TSNE done")

    write_coords_csv(coords_per_ckpt, metadata_for_each, args.output_dir, "tsne")
    render_grid(coords_per_ckpt, metadata_for_each, args.output_dir, "tsne")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
