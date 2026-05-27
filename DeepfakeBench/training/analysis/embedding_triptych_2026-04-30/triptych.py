"""3x3 embedding triptych: P8A | Slot 2 | Slot 3, three colorings each.

Renders a 3x3 grid of 2D embedding scatter plots:
  rows    = P8A | SLOT2 (GRL) | SLOT3 (jitter)
  columns = colored by (a) real/fake, (b) clip_capture_mode, (c) face_pixel_area bucket

Visual narrative figure for the wiki / handoff. The same 800 frames are
projected through each checkpoint independently — within a row the geometry
is shaped by what THAT checkpoint's [CLS] features carry; across rows you
compare how each model collapsed (or kept distinct) the various nuisance
axes.

How to read:
- Column 1 (real/fake): Should be cleanly separable for a healthy detector.
  If P8A separates and Slot 2/Slot 3 do NOT, the intervention degraded the
  primary task. Reading: sanity check.
- Column 2 (clip_capture_mode): Webcam-vs-non-webcam separability. Per
  project_lockbox_fpr_dominated_by_webcam_mode.md, P8A is dominated by
  webcam-mode FPR. If GRL (Slot 2) WORKED, the webcam cluster should
  collapse into the rest; if jitter (Slot 3) WORKED, less so (jitter
  doesn't directly target capture mode). This is the load-bearing column.
- Column 3 (face_pixel_area bucket): The face-size leak axis (per
  project_face_size_label_leak.md). If jitter (Slot 3) WORKED, the
  small/medium/large face-area clusters should bleed into each other; if
  not, they remain separable. This is the load-bearing column for Slot 3.

Embeddings cached per checkpoint to avoid re-running inference. CSV with
the 2D coordinates (per ckpt × frame) is written so the user can rerun
plotting without re-extracting features.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import yaml
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detectors import DETECTOR  # noqa: E402

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

DEFAULT_PARQUET = REPO_ROOT / "analysis" / "lockbox_tagging" / "full_tags_2026-04-27.parquet"
DEFAULT_DETECTOR_CONFIG = REPO_ROOT / "config" / "detector" / "effort.yaml"
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "config" / "train_config.yaml"

logger = logging.getLogger("triptych")


# -----------------------------------------------------------------------------
# Checkpoint loading.
# -----------------------------------------------------------------------------
def _maybe_download_gcs_checkpoint(ckpt_uri: str, cache_dir: Path) -> Path:
    if not ckpt_uri.startswith("gs://"):
        return Path(ckpt_uri)
    cache_dir.mkdir(parents=True, exist_ok=True)
    blob_name = ckpt_uri[5:].split("/", 1)[1]
    local_path = cache_dir / Path(blob_name).name
    if local_path.exists():
        logger.info("Checkpoint cached at %s", local_path)
        return local_path
    from google.cloud import storage

    bucket_name = ckpt_uri[5:].split("/", 1)[0]
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    if not blob.exists(client=client):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_uri}")
    logger.info("Downloading %s -> %s", ckpt_uri, local_path)
    blob.download_to_filename(str(local_path))
    return local_path


def load_effort_model(
    checkpoint_path: Path,
    detector_config: Path,
    train_config: Path,
    device: torch.device,
) -> torch.nn.Module:
    with open(detector_config, "r") as f:
        cfg = yaml.safe_load(f)
    with open(train_config, "r") as f:
        train_cfg = yaml.safe_load(f)
    cfg.update(train_cfg)

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        model_config = ckpt.get("model_config", {})
        for k, v in model_config.items():
            if k != "current_arcface_s":
                cfg[k] = v
    else:
        state_dict = ckpt
        model_config = {}

    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)
    if model_config.get("use_arcface_head", False) and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])
    clean_state = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Feature extraction.
# -----------------------------------------------------------------------------
def load_and_preprocess(local_path: Path, resolution: int) -> Optional[torch.Tensor]:
    img_bgr = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_bgr = cv2.resize(img_bgr, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])
    return transform(img_rgb)


def extract_cls_features(
    model: torch.nn.Module,
    paths: List[Path],
    device: torch.device,
    batch_size: int = 64,
    resolution: int = 224,
) -> Tuple[np.ndarray, np.ndarray]:
    feats: List[np.ndarray] = []
    valid: List[int] = []
    pending: List[Tuple[int, torch.Tensor]] = []
    for i, p in enumerate(paths):
        t = load_and_preprocess(p, resolution)
        if t is None:
            continue
        pending.append((i, t))
    for i in range(0, len(pending), batch_size):
        chunk = pending[i:i + batch_size]
        batch_idx = [c[0] for c in chunk]
        batch = torch.stack([c[1] for c in chunk]).to(device, non_blocking=True)
        with torch.inference_mode():
            outputs = model({"image": batch}, inference=True)
            f = outputs["feat"].detach().cpu().numpy()
        feats.append(f)
        valid.extend(batch_idx)
    feats_arr = np.concatenate(feats, axis=0) if feats else np.zeros((0, 0), dtype=np.float32)
    valid_arr = np.array(valid, dtype=np.int64)
    return feats_arr, valid_arr


def cache_path_for(ckpt_label: str, n_samples: int, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"triptych_features__{ckpt_label}__n{n_samples}.npz"


# -----------------------------------------------------------------------------
# Sampling: stratified across (real/fake, method, capture_mode).
# -----------------------------------------------------------------------------
def stratified_sample(
    parquet_path: Path,
    n_samples: int,
    seed: int = 0,
) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if "decode_ok" in df.columns:
        df = df[df["decode_ok"].astype(bool)]
    df = df[df["local_path"].apply(lambda p: os.path.exists(str(p)))].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("Parquet has zero rows whose local_path exists on disk.")

    # Stratify by (label, method, clip_capture_mode); fall back gracefully if a
    # column is missing.
    strata_cols = [c for c in ("label", "method", "clip_capture_mode") if c in df.columns]
    if not strata_cols:
        return df.sample(n=min(n_samples, len(df)), random_state=seed).reset_index(drop=True)

    # Proportional allocation per stratum.
    grouped = df.groupby(strata_cols, dropna=False)
    weights = grouped.size() / len(df)
    rng = np.random.default_rng(seed)
    picks: List[int] = []
    for key, sub_idx in grouped.groups.items():
        n_target = max(1, int(round(n_samples * weights.loc[key])))
        n_avail = len(sub_idx)
        n_take = min(n_avail, n_target)
        picks.extend(rng.choice(np.array(sub_idx), size=n_take, replace=False).tolist())
    # Clip to exactly n_samples (in case rounding overshoots)
    if len(picks) > n_samples:
        picks = list(rng.choice(picks, size=n_samples, replace=False))
    elif len(picks) < n_samples:
        # Top up with random rows (without replacement)
        remaining = list(set(df.index.tolist()) - set(picks))
        if remaining:
            top_up = rng.choice(remaining, size=min(n_samples - len(picks), len(remaining)), replace=False)
            picks.extend(top_up.tolist())
    return df.iloc[picks].reset_index(drop=True)


# -----------------------------------------------------------------------------
# Dimensionality reduction.
# -----------------------------------------------------------------------------
def reduce_2d(features: np.ndarray, reducer: str, perplexity: int, n_neighbors: int, seed: int) -> np.ndarray:
    if reducer == "tsne":
        from sklearn.manifold import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, max(5, len(features) // 4)),
            random_state=seed,
            init="pca",
            learning_rate="auto",
            n_jobs=1,  # explicit; do not use -1 (see feedback_sklearn_njobs.md)
        )
        return tsne.fit_transform(features)
    elif reducer == "umap":
        try:
            import umap
        except ImportError as e:
            raise ImportError(
                "umap-learn is not installed. `pip install umap-learn` or use --reducer tsne."
            ) from e
        reducer_obj = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            random_state=seed,
            min_dist=0.1,
        )
        return reducer_obj.fit_transform(features)
    else:
        raise ValueError(f"Unknown reducer: {reducer!r}")


# -----------------------------------------------------------------------------
# Plot.
# -----------------------------------------------------------------------------
def _color_by_label(df: pd.DataFrame) -> Tuple[np.ndarray, dict]:
    if "label" not in df.columns:
        return np.zeros(len(df)), {0: "all"}
    cmap = {"real": "#4477aa", "fake": "#cc6677"}
    colors = np.array([cmap.get(str(v), "#888888") for v in df["label"].astype(str)])
    return colors, cmap


def _color_by_capture_mode(df: pd.DataFrame) -> Tuple[np.ndarray, dict]:
    palette = ["#4477aa", "#cc6677", "#117733", "#ddcc77", "#882255", "#88ccee"]
    if "clip_capture_mode" not in df.columns:
        return np.zeros(len(df)), {0: "n/a"}
    modes = df["clip_capture_mode"].fillna("unknown").astype(str)
    unique = sorted(modes.unique())
    cmap = {m: palette[i % len(palette)] for i, m in enumerate(unique)}
    colors = np.array([cmap[m] for m in modes])
    return colors, cmap


def _color_by_face_size(df: pd.DataFrame) -> Tuple[np.ndarray, dict]:
    palette = ["#4477aa", "#117733", "#ddcc77", "#cc6677"]  # small -> large
    if "face_pixel_area" not in df.columns:
        return np.zeros(len(df)), {0: "n/a"}
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
        -1: "missing",
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
        # Slice metadata to rows that produced features
        for c, (col_title, fn) in enumerate(color_funcs):
            ax = axes[r, c]
            colors, cmap = fn(metadata_df)
            ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=10, alpha=0.6, edgecolors="none")
            ax.set_title(f"{label}  |  {col_title}")
            ax.set_xticks([])
            ax.set_yticks([])
            # Build legend (small)
            handles = [
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=v, label=str(k), markersize=6)
                for k, v in cmap.items()
            ]
            ax.legend(
                handles=handles, loc="upper right",
                fontsize=7, framealpha=0.7, markerscale=1.2,
            )

    fig.suptitle(f"[CLS] feature embedding ({reducer_name.upper()}) — n={len(metadata_df)} frames per row", y=1.0)
    fig.tight_layout()
    fig.savefig(output_dir / f"triptych_grid_{reducer_name}.png", dpi=120)
    plt.close(fig)


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
            # Strip embedding columns to keep CSV light
            for drop_key in ("arcface_embed", "clip_embed"):
                base.pop(drop_key, None)
            rows.append(base)
    pd.DataFrame(rows).to_csv(output_dir / f"triptych_coords_{reducer_name}.csv", index=False)


# -----------------------------------------------------------------------------
# CLI.
# -----------------------------------------------------------------------------
def parse_ckpt_args(raw_list: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in raw_list:
        if "=" not in raw:
            raise ValueError(f"--ckpts entries must be LABEL=URI; got: {raw}")
        label, uri = raw.split("=", 1)
        out[label.strip()] = uri.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="3-row triptych: (P8A | Slot2 | Slot3) x (real/fake | capture_mode | face_size)"
    )
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="Space-separated LABEL=URI entries")
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--n_samples", type=int, default=800)
    ap.add_argument("--reducer", type=str, choices=["tsne", "umap"], default="tsne")
    ap.add_argument("--perplexity", type=int, default=30)
    ap.add_argument("--n_neighbors", type=int, default=15)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--cache_dir", type=Path,
                    default=REPO_ROOT / "analysis" / "_features_cache_2026-04-30")
    ap.add_argument("--detector_config", type=Path, default=DEFAULT_DETECTOR_CONFIG)
    ap.add_argument("--train_config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    ckpts = parse_ckpt_args(args.ckpts)

    sampled_df = stratified_sample(args.parquet, args.n_samples, seed=args.seed)
    logger.info("Stratified sample: %d frames", len(sampled_df))
    paths = [Path(p) for p in sampled_df["local_path"].tolist()]

    # Cached metadata for the sample stays consistent across all ckpts; we
    # save the indices so a partial-extraction crash can be resumed.
    sampled_df.to_csv(args.output_dir / "sampled_frames.csv", index=False)

    coords_per_ckpt: Dict[str, np.ndarray] = {}
    metadata_for_each: Optional[pd.DataFrame] = None
    common_valid: Optional[np.ndarray] = None

    # First pass: extract features for each checkpoint.
    feats_per_ckpt: Dict[str, np.ndarray] = {}
    valid_per_ckpt: Dict[str, np.ndarray] = {}
    for label, uri in ckpts.items():
        cache_p = cache_path_for(label, args.n_samples, args.cache_dir)
        if cache_p.exists():
            logger.info("[%s] features cached at %s", label, cache_p)
            cached = np.load(cache_p, allow_pickle=False)
            features = cached["features"]
            valid_idx = cached["valid_idx"]
        else:
            logger.info("[%s] loading checkpoint and extracting features", label)
            ckpt_path = _maybe_download_gcs_checkpoint(uri, args.cache_dir)
            model = load_effort_model(ckpt_path, args.detector_config, args.train_config, device)
            features, valid_idx = extract_cls_features(
                model=model, paths=paths, device=device,
                batch_size=args.batch_size, resolution=args.resolution,
            )
            np.savez_compressed(cache_p, features=features, valid_idx=valid_idx)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        feats_per_ckpt[label] = features
        valid_per_ckpt[label] = valid_idx
        common_valid = valid_idx if common_valid is None else np.intersect1d(common_valid, valid_idx)

    if common_valid is None or len(common_valid) == 0:
        raise RuntimeError("No common valid frames across all checkpoints — cannot align.")

    metadata_for_each = sampled_df.iloc[common_valid].reset_index(drop=True)

    # Reduce each checkpoint's features to 2D (filtered to the common valid indices).
    for label, features in feats_per_ckpt.items():
        valid_idx = valid_per_ckpt[label]
        # Build a positional mapping from valid_idx -> features rows.
        pos_in_features = {int(orig_idx): row for row, orig_idx in enumerate(valid_idx)}
        rows = [pos_in_features[int(c)] for c in common_valid]
        f_aligned = features[rows]
        coords = reduce_2d(
            f_aligned,
            reducer=args.reducer,
            perplexity=args.perplexity,
            n_neighbors=args.n_neighbors,
            seed=args.seed,
        )
        coords_per_ckpt[label] = coords

    render_grid(coords_per_ckpt, metadata_for_each, args.output_dir, args.reducer)
    write_coords_csv(coords_per_ckpt, metadata_for_each, args.output_dir, args.reducer)

    print()
    print("=" * 70)
    print(f"EMBEDDING TRIPTYCH ({args.reducer.upper()})")
    print("=" * 70)
    for label in coords_per_ckpt:
        print(f"  {label:<10} coords shape = {coords_per_ckpt[label].shape}")
    print(f"  n_aligned_frames = {len(metadata_for_each)}")
    print()
    print(f"Grid written to: {args.output_dir / f'triptych_grid_{args.reducer}.png'}")
    print(f"CSV  written to: {args.output_dir / f'triptych_coords_{args.reducer}.csv'}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
