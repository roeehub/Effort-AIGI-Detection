"""Face-size / crop-tightness invariance probe (Slot 3 jitter validation).

This script tests whether `face_scale_jitter@scale_limit=0.50` flattened the
score-vs-tightness curve. Procedure: for each frame in --frames_dir, generate
N tightness variants (default t in [0.7, 0.85, 1.0, 1.15, 1.5]) using the same
center-crop / pad-with-edge-replication pipeline as
`analysis/crop_shortcut_2026-04-27/crop_sweep.py`, run each variant through the
loaded checkpoint, and record `prob_fake` per (frame, t).

How to read the output:
- "Flip rate" = fraction of frames whose argmax label (real/fake at threshold
  0.5) changes across the tightness grid. The 04-27 audit reported 25/47
  (53%) on P8A; if jitter@0.50 worked on Slot 3, this should drop to ~10-15%
  or lower.
- "Median |Δprob_fake|" = median of (max - min) over the tightness grid
  per frame. A small number (~0.05) means the curve is flat (good, jitter
  worked); a large number (>0.30) means the curve is steep (the shortcut is
  intact). Compare P8A vs Slot 3 directly on the same --frames_dir for an
  apples-to-apples readout.
- The PNG renders one line per frame (alpha=0.3) with a thick median line
  on top and the headline flip rate annotated in the title.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Resolve repo root and ensure project imports work
REPO_ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detectors import DETECTOR  # noqa: E402

# CLIP normalization (matches batch_inference_gcs.py)
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

DEFAULT_TIGHTNESS = (0.7, 0.85, 1.0, 1.15, 1.5)
DEFAULT_DETECTOR_CONFIG = REPO_ROOT / "config" / "detector" / "effort.yaml"
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "config" / "train_config.yaml"

logger = logging.getLogger("face-size-invariance")


# -----------------------------------------------------------------------------
# Tightness variant generator (mirrors crop_shortcut_2026-04-27/crop_sweep.py).
# -----------------------------------------------------------------------------
def make_variant(img: Image.Image, tightness: float) -> Image.Image:
    """Tightness > 1: tighter (center-crop 1/t side, resize back).

    Tightness < 1: looser (shrink to t-fraction of canvas, edge-replicate pad)."""
    if abs(tightness - 1.0) < 1e-6:
        return img.copy()
    W, H = img.size
    if tightness > 1.0:
        new_w = max(1, int(round(W / tightness)))
        new_h = max(1, int(round(H / tightness)))
        x0 = (W - new_w) // 2
        y0 = (H - new_h) // 2
        sub = img.crop((x0, y0, x0 + new_w, y0 + new_h))
        return sub.resize((W, H), Image.LANCZOS)
    inner_w = max(1, int(round(W * tightness)))
    inner_h = max(1, int(round(H * tightness)))
    inner = img.resize((inner_w, inner_h), Image.LANCZOS)
    canvas = Image.new(img.mode, (W, H), (0, 0, 0) if img.mode in ("RGB", "RGBA") else 0)
    pad_x = (W - inner_w) // 2
    pad_y = (H - inner_h) // 2
    if pad_y > 0:
        top = inner.crop((0, 0, inner_w, 1)).resize((inner_w, pad_y))
        canvas.paste(top, (pad_x, 0))
    if pad_y > 0 and (H - pad_y - inner_h) > 0:
        bot = inner.crop((0, inner_h - 1, inner_w, inner_h)).resize((inner_w, H - pad_y - inner_h))
        canvas.paste(bot, (pad_x, pad_y + inner_h))
    if pad_x > 0:
        lf = inner.crop((0, 0, 1, inner_h)).resize((pad_x, inner_h))
        canvas.paste(lf, (0, pad_y))
    if pad_x > 0 and (W - pad_x - inner_w) > 0:
        rt = inner.crop((inner_w - 1, 0, inner_w, inner_h)).resize((W - pad_x - inner_w, inner_h))
        canvas.paste(rt, (pad_x + inner_w, pad_y))
    if pad_x > 0 and pad_y > 0:
        tl = inner.crop((0, 0, 1, 1)).resize((pad_x, pad_y))
        canvas.paste(tl, (0, 0))
    if pad_x > 0 and pad_y > 0 and (W - pad_x - inner_w) > 0:
        tr = inner.crop((inner_w - 1, 0, inner_w, 1)).resize((W - pad_x - inner_w, pad_y))
        canvas.paste(tr, (pad_x + inner_w, 0))
    if pad_x > 0 and pad_y > 0 and (H - pad_y - inner_h) > 0:
        bl = inner.crop((0, inner_h - 1, 1, inner_h)).resize((pad_x, H - pad_y - inner_h))
        canvas.paste(bl, (0, pad_y + inner_h))
    if (
        pad_x > 0 and pad_y > 0
        and (W - pad_x - inner_w) > 0 and (H - pad_y - inner_h) > 0
    ):
        br = inner.crop((inner_w - 1, inner_h - 1, inner_w, inner_h)).resize(
            (W - pad_x - inner_w, H - pad_y - inner_h)
        )
        canvas.paste(br, (pad_x + inner_w, pad_y + inner_h))
    canvas.paste(inner, (pad_x, pad_y))
    return canvas


# -----------------------------------------------------------------------------
# Checkpoint loading (mirrors arena/model_arena.py and batch_inference_gcs.py).
# -----------------------------------------------------------------------------
def _maybe_download_gcs_checkpoint(ckpt_uri: str, cache_dir: Path) -> Path:
    if not ckpt_uri.startswith("gs://"):
        return Path(ckpt_uri)
    cache_dir.mkdir(parents=True, exist_ok=True)
    blob_name = ckpt_uri[5:].split("/", 1)[1]
    local_path = cache_dir / Path(blob_name).name
    if local_path.exists():
        logger.info("Checkpoint already cached at %s", local_path)
        return local_path
    from google.cloud import storage  # local import to keep no-GCS use lightweight

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
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing:
        logger.debug("Missing keys: %d", len(missing))
    if unexpected:
        logger.debug("Unexpected keys: %d", len(unexpected))
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Inference helpers.
# -----------------------------------------------------------------------------
def pil_to_tensor(img: Image.Image, resolution: int) -> torch.Tensor:
    """Match training preprocessing: BGR-aware OpenCV resize INTER_LINEAR + CLIP normalize."""
    img_rgb = np.array(img.convert("RGB"))
    # cv2 expects BGR but resize works on RGB the same way; keep INTER_LINEAR per
    # threads/preprocessing_parity_bug.md.
    img_resized = cv2.resize(img_rgb, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])
    return transform(img_resized)


def score_batch(model: torch.nn.Module, tensors: List[torch.Tensor], device: torch.device) -> np.ndarray:
    batch = torch.stack(tensors).to(device, non_blocking=True)
    with torch.inference_mode():
        outputs = model({"image": batch}, inference=True)
        probs = outputs["prob"].detach().cpu().numpy().reshape(-1)
    return probs


# -----------------------------------------------------------------------------
# Main sweep.
# -----------------------------------------------------------------------------
def collect_frames(frames_dir: Path, exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> List[Path]:
    out: List[Path] = []
    for p in sorted(frames_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


def run_sweep(
    model: torch.nn.Module,
    frame_paths: List[Path],
    tightnesses: List[float],
    resolution: int,
    batch_size: int,
    device: torch.device,
) -> List[dict]:
    """For each frame × tightness, return dicts with prob_fake and predicted_label."""
    rows: List[dict] = []
    pending: List[dict] = []  # holds dicts before scoring

    for frame_path in frame_paths:
        try:
            img = Image.open(frame_path).convert("RGB")
        except Exception as e:
            logger.warning("Failed to open %s: %s", frame_path, e)
            continue
        for t in tightnesses:
            variant = make_variant(img, t)
            tensor = pil_to_tensor(variant, resolution)
            pending.append({
                "frame_id": frame_path.stem,
                "frame_path": str(frame_path),
                "tightness": float(t),
                "tensor": tensor,
            })

    # Batch through model
    logger.info("Scoring %d (frame, tightness) pairs in batches of %d", len(pending), batch_size)
    for i in range(0, len(pending), batch_size):
        chunk = pending[i:i + batch_size]
        tensors = [c["tensor"] for c in chunk]
        probs = score_batch(model, tensors, device)
        for c, p in zip(chunk, probs):
            rows.append({
                "frame_id": c["frame_id"],
                "frame_path": c["frame_path"],
                "tightness": c["tightness"],
                "prob_fake": float(p),
                "predicted_label": "fake" if p >= 0.5 else "real",
            })
    return rows


# -----------------------------------------------------------------------------
# Aggregation + plotting.
# -----------------------------------------------------------------------------
def compute_flip_metrics(rows: List[dict]) -> dict:
    """Per-frame flip rate (label changes across tightness grid) + median |delta prob_fake|."""
    by_frame: dict[str, List[Tuple[float, float, str]]] = {}
    for r in rows:
        by_frame.setdefault(r["frame_id"], []).append(
            (r["tightness"], r["prob_fake"], r["predicted_label"])
        )

    flip_count = 0
    total = 0
    deltas: List[float] = []
    flips_by_pair: dict[Tuple[float, float], int] = {}
    for fid, recs in by_frame.items():
        if len(recs) < 2:
            continue
        recs.sort(key=lambda x: x[0])
        labels = [r[2] for r in recs]
        probs = [r[1] for r in recs]
        # Per-frame flip if any pair of (sorted) tightness levels disagrees on label
        if len(set(labels)) > 1:
            flip_count += 1
        total += 1
        deltas.append(max(probs) - min(probs))
        # Per-tightness-pair flip count (adjacent-bin)
        for (t_a, _, lab_a), (t_b, _, lab_b) in zip(recs[:-1], recs[1:]):
            if lab_a != lab_b:
                key = (round(t_a, 3), round(t_b, 3))
                flips_by_pair[key] = flips_by_pair.get(key, 0) + 1

    flip_rate = flip_count / total if total else 0.0
    median_delta = float(np.median(deltas)) if deltas else 0.0
    return {
        "n_frames": total,
        "n_flipped": flip_count,
        "flip_rate": flip_rate,
        "median_delta_prob_fake": median_delta,
        "adjacent_pair_flip_counts": dict(sorted(flips_by_pair.items())),
    }


def plot_curves(rows: List[dict], output_png: Path, summary: dict) -> None:
    by_frame: dict[str, List[Tuple[float, float]]] = {}
    for r in rows:
        by_frame.setdefault(r["frame_id"], []).append((r["tightness"], r["prob_fake"]))

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    all_xs = sorted({pt[0] for recs in by_frame.values() for pt in recs})
    median_y_per_x = []
    for x in all_xs:
        ys = []
        for recs in by_frame.values():
            for t, p in recs:
                if abs(t - x) < 1e-6:
                    ys.append(p)
        median_y_per_x.append(np.median(ys) if ys else np.nan)

    for fid, recs in by_frame.items():
        recs.sort(key=lambda r: r[0])
        xs = [r[0] for r in recs]
        ys = [r[1] for r in recs]
        ax.plot(xs, ys, "-", alpha=0.25, linewidth=0.8)

    ax.plot(all_xs, median_y_per_x, "k-", linewidth=2.5, label="median")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="threshold=0.5")
    ax.set_xlabel("crop tightness (>1 = tighter, <1 = looser)")
    ax.set_ylabel("prob_fake")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(
        f"Score vs crop tightness (n={summary['n_frames']} frames)\n"
        f"flip rate = {summary['flip_rate']:.1%} ({summary['n_flipped']}/{summary['n_frames']});  "
        f"median |Δprob_fake| = {summary['median_delta_prob_fake']:.3f}"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_png, dpi=120)
    plt.close(fig)


def write_csv(rows: List[dict], output_csv: Path) -> None:
    import csv
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["frame_id", "frame_path", "tightness", "prob_fake", "predicted_label"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Crop-tightness invariance probe: re-runs the 47-frame × 5-tightness sweep on a checkpoint."
    )
    ap.add_argument("--ckpt", required=True, type=str,
                    help="Local path or gs:// URI to a .pth checkpoint")
    ap.add_argument("--frames_dir", required=True, type=Path,
                    help="Directory of input frames (recursively glob *.png/*.jpg/*.jpeg)")
    ap.add_argument("--tightness_grid", nargs="+", type=float, default=list(DEFAULT_TIGHTNESS),
                    help=f"Tightness factors to sweep (default {DEFAULT_TIGHTNESS})")
    ap.add_argument("--output_csv", required=True, type=Path,
                    help="Where to write per-frame × tightness predictions CSV")
    ap.add_argument("--output_png", required=True, type=Path,
                    help="Where to write the score-vs-tightness curve plot")
    ap.add_argument("--cache_dir", type=Path,
                    default=REPO_ROOT / "analysis" / "_features_cache_2026-04-30",
                    help="Local cache for downloaded GCS checkpoints")
    ap.add_argument("--detector_config", type=Path, default=DEFAULT_DETECTOR_CONFIG)
    ap.add_argument("--train_config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_frames", type=int, default=None,
                    help="Optional cap on number of frames sampled (for quick smoke test)")
    ap.add_argument("--device", type=str, default=None,
                    help="cuda/mps/cpu (auto-detected if not provided)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

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

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)

    ckpt_path = _maybe_download_gcs_checkpoint(args.ckpt, args.cache_dir)
    model = load_effort_model(ckpt_path, args.detector_config, args.train_config, device)

    frame_paths = collect_frames(args.frames_dir)
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]
    logger.info("Found %d frames in %s", len(frame_paths), args.frames_dir)
    if not frame_paths:
        logger.error("No frames found under %s", args.frames_dir)
        return 1

    rows = run_sweep(
        model=model,
        frame_paths=frame_paths,
        tightnesses=list(args.tightness_grid),
        resolution=args.resolution,
        batch_size=args.batch_size,
        device=device,
    )

    write_csv(rows, args.output_csv)
    summary = compute_flip_metrics(rows)
    plot_curves(rows, args.output_png, summary)

    print()
    print("=" * 70)
    print(f"FACE-SIZE INVARIANCE :: ckpt = {Path(args.ckpt).name}")
    print("=" * 70)
    print(f"frames evaluated        : {summary['n_frames']}")
    print(f"frames whose label flips: {summary['n_flipped']}")
    print(f"flip rate               : {summary['flip_rate']:.1%}")
    print(f"median |Δprob_fake|     : {summary['median_delta_prob_fake']:.4f}")
    print()
    print("Adjacent-tightness flip counts:")
    for (t_a, t_b), n in summary["adjacent_pair_flip_counts"].items():
        print(f"  {t_a:.2f} → {t_b:.2f} : {n}")
    print()
    print(f"CSV  written: {args.output_csv}")
    print(f"PNG  written: {args.output_png}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
