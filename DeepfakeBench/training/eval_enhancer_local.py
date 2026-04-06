#!/usr/bin/env python3
"""
eval_enhancer_local.py – Local MPS evaluation of multiple checkpoints
against the WMA enhanced face-swap images.

All images are FAKE (enhanced face-swaps). The script:
  1. Downloads each checkpoint from GCS (skips if already cached)
  2. Runs inference on all images with MPS
  3. Deletes the checkpoint from disk before downloading the next
  4. Saves per-image probabilities + per-model summary to CSV

Usage:
    cd DeepfakeBench/training
    python eval_enhancer_local.py

Output:
    analysis_results/enhancer_eval_per_image.csv   – per-image probs for every model
    analysis_results/enhancer_eval_summary.csv     – per-model aggregated stats
"""

from __future__ import annotations

import csv
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image as pil_image

# ── Project imports ──
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detectors import DETECTOR  # noqa: E402

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
IMAGE_DIR = Path("/Users/roeedar/Downloads/wma_export/all_images")
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "analysis_results"
CACHE_DIR = Path(__file__).resolve().parent / "weights" / "enhancer_eval"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 5 checkpoints to evaluate — spanning different rounds and strategies
CHECKPOINTS = {
    # Current deployment winner (R2.5, k=32, base_only aug, no QE data)
    "R25_F1": {
        "gcs": "gs://training-job-outputs/phase2r2_experiments/5w453our/"
               "top_n_effort_20260212_step18500_auc0.9893_eer0.0356.pth",
        "desc": "k=32, cosine, base_only aug (deployment winner)",
    },
    # Holdout champion – fine-tuned from F1, base_only aug + QE data
    "R3_FT3": {
        "gcs": "gs://training-job-outputs/phase2r3_experiments/kzfu116l/"
               "top_n_effort_20260213_step2500_auc0.9966_eer0.0191.pth",
        "desc": "FT from F1, base_only aug, +QE data (holdout AUC 0.9966)",
    },
    # Fine-tuned with quality_robust moderate aug + QE data
    "R3_FT1": {
        "gcs": "gs://training-job-outputs/phase2r3_experiments/w7wi9lpj/"
               "top_n_effort_20260213_step500_auc0.9962_eer0.0191.pth",
        "desc": "FT from F1, quality_robust moderate aug (AUC 0.9962)",
    },
    # Fine-tuned with quality_robust light aug + QE data
    "R3_FT2": {
        "gcs": "gs://training-job-outputs/phase2r3_experiments/3cxpwxgn/"
               "top_n_effort_20260213_step500_auc0.9954_eer0.0229.pth",
        "desc": "FT from F1, quality_robust light aug (AUC 0.9954)",
    },
    # Phase 1 best — the old deployed model we know fails on enhanced fakes
    "B16_old": {
        "gcs": "gs://training-job-outputs/best_checkpoints/corrected/"
               "top_n_effort_20260119_step14000_auc0.9947_eer0.0218_B16_LAION.patched.pth",
        "desc": "Phase 1 best (k=8, CE, known to fail on enhancer)",
    },
}

# ─────────────────────────────────────────
# Logging
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("enhancer-eval")

# ─────────────────────────────────────────
# Device
# ─────────────────────────────────────────

def _resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = _resolve_device()
logger.info(f"Using device: {DEVICE}")

# ─────────────────────────────────────────
# Image preprocessing (CLIP normalization, 224×224)
# Images are already cropped faces from WMA export — no face detection needed
# ─────────────────────────────────────────
_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711]),
])


def preprocess_image(img_path: Path) -> torch.Tensor:
    """Load, resize to 224×224, normalize with CLIP stats."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    return _transform(pil_image.fromarray(img_rgb))


# ─────────────────────────────────────────
# GCS download
# ─────────────────────────────────────────
def download_checkpoint(gcs_path: str) -> Path:
    """Download from GCS, returning local path. Caches in CACHE_DIR."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = gcs_path.rsplit("/", 1)[-1]
    local_path = CACHE_DIR / filename

    if local_path.exists():
        logger.info(f"  ↳ Cached: {local_path.name}")
        return local_path

    logger.info(f"  ↳ Downloading: {gcs_path}")
    t0 = time.time()
    result = subprocess.run(
        ["gsutil", "cp", gcs_path, str(local_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gsutil failed: {result.stderr}")

    size_mb = local_path.stat().st_size / (1024 * 1024)
    logger.info(f"  ↳ Downloaded in {time.time() - t0:.1f}s ({size_mb:.1f} MB)")
    return local_path


def delete_checkpoint(local_path: Path):
    """Delete cached checkpoint to free disk space."""
    if local_path.exists():
        local_path.unlink()
        logger.info(f"  ↳ Deleted: {local_path.name}")


# ─────────────────────────────────────────
# Model loading (adapted from test_checkpoint_local.py)
# ─────────────────────────────────────────
def load_model(weights_path: Path) -> torch.nn.Module:
    """Load an EffortDetector from a checkpoint onto DEVICE."""
    ckpt = torch.load(str(weights_path), map_location=DEVICE, weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        model_config = ckpt.get("model_config", {})
    else:
        state_dict = ckpt
        model_config = {}

    # Load base config + apply checkpoint overrides
    base_cfg_path = Path(__file__).resolve().parent / "config" / "detector" / "effort.yaml"
    with open(base_cfg_path) as f:
        cfg = yaml.safe_load(f)

    if model_config:
        for key, value in model_config.items():
            if key != "current_arcface_s":
                cfg[key] = value

    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(DEVICE)

    # Restore ArcFace dynamic s if needed
    if model_config.get("use_arcface_head", False) and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])

    state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ─────────────────────────────────────────
# Batch inference
# ─────────────────────────────────────────
@torch.inference_mode()
def run_batch(model: torch.nn.Module, tensors: torch.Tensor) -> np.ndarray:
    """Run a batch through the model, return fake probabilities (N,)."""
    data = {
        "image": tensors.to(DEVICE),
        "label": torch.zeros(tensors.size(0), dtype=torch.long, device=DEVICE),
    }
    preds = model(data, inference=True)
    return preds["prob"].squeeze(-1).cpu().numpy()


def evaluate_model(model: torch.nn.Module, image_paths: List[Path],
                   batch_size: int = 32) -> Dict[str, float]:
    """Run inference on all images, return {filename: fake_prob}."""
    results: Dict[str, float] = {}
    n = len(image_paths)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_paths = image_paths[start:end]

        tensors = []
        valid_paths = []
        for p in batch_paths:
            try:
                tensors.append(preprocess_image(p))
                valid_paths.append(p)
            except Exception as e:
                logger.warning(f"  Skipping {p.name}: {e}")
                results[p.name] = float("nan")

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors)
        probs = run_batch(model, batch_tensor)

        for path, prob in zip(valid_paths, probs):
            results[path.name] = float(prob)

        done = min(end, n)
        if done % (batch_size * 5) == 0 or done == n:
            logger.info(f"    [{done}/{n}] images processed")

    return results


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    # Collect images
    image_paths = sorted([
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in IMG_EXTS
    ])
    logger.info(f"Found {len(image_paths)} images in {IMAGE_DIR}")
    if not image_paths:
        logger.error("No images found!")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Results storage: {model_name: {filename: prob}}
    all_results: Dict[str, Dict[str, float]] = {}

    for i, (model_name, ckpt_info) in enumerate(CHECKPOINTS.items(), 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{len(CHECKPOINTS)}] {model_name}: {ckpt_info['desc']}")
        logger.info(f"{'='*60}")

        # Download
        local_path = download_checkpoint(ckpt_info["gcs"])

        # Load
        t0 = time.time()
        model = load_model(local_path)
        logger.info(f"  Model loaded in {time.time() - t0:.1f}s")

        # Evaluate
        t0 = time.time()
        results = evaluate_model(model, image_paths)
        elapsed = time.time() - t0
        logger.info(f"  Inference done in {elapsed:.1f}s "
                     f"({len(image_paths)/elapsed:.1f} img/s)")

        all_results[model_name] = results

        # Quick stats (all images are FAKE — prob > 0.5 = correct)
        probs = [v for v in results.values() if not np.isnan(v)]
        correct = sum(1 for p in probs if p > 0.5)
        logger.info(f"  ── Quick stats ──")
        logger.info(f"  Accuracy (>0.5):  {correct}/{len(probs)} "
                     f"({100*correct/len(probs):.1f}%)")
        logger.info(f"  Mean fake prob:   {np.mean(probs):.4f}")
        logger.info(f"  Median fake prob: {np.median(probs):.4f}")
        logger.info(f"  Min fake prob:    {np.min(probs):.4f}")
        logger.info(f"  Max fake prob:    {np.max(probs):.4f}")

        # Free memory before next model
        del model
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        elif DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        # Delete checkpoint to save disk (keeps next download fast)
        delete_checkpoint(local_path)

    # ──────────────────────────────────────
    # Write per-image CSV
    # ──────────────────────────────────────
    per_image_path = OUTPUT_DIR / "enhancer_eval_per_image.csv"
    model_names = list(CHECKPOINTS.keys())
    all_filenames = sorted(set().union(*(r.keys() for r in all_results.values())))

    with open(per_image_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["filename"] + [f"{m}_prob" for m in model_names]
        writer.writerow(header)
        for fname in all_filenames:
            row = [fname] + [
                f"{all_results[m].get(fname, float('nan')):.6f}"
                for m in model_names
            ]
            writer.writerow(row)

    logger.info(f"\n✅ Per-image results: {per_image_path}")

    # ──────────────────────────────────────
    # Write summary CSV
    # ──────────────────────────────────────
    summary_path = OUTPUT_DIR / "enhancer_eval_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "description", "n_images",
            "accuracy_at_0.5", "accuracy_at_0.6", "accuracy_at_0.7",
            "mean_prob", "median_prob", "std_prob",
            "min_prob", "max_prob",
            "pct_above_0.9", "pct_below_0.3",
        ])
        for model_name in model_names:
            probs = np.array([
                all_results[model_name].get(fn, float("nan"))
                for fn in all_filenames
            ])
            probs = probs[~np.isnan(probs)]
            n = len(probs)
            if n == 0:
                continue
            writer.writerow([
                model_name,
                CHECKPOINTS[model_name]["desc"],
                n,
                f"{100 * np.mean(probs > 0.5):.1f}%",
                f"{100 * np.mean(probs > 0.6):.1f}%",
                f"{100 * np.mean(probs > 0.7):.1f}%",
                f"{np.mean(probs):.4f}",
                f"{np.median(probs):.4f}",
                f"{np.std(probs):.4f}",
                f"{np.min(probs):.4f}",
                f"{np.max(probs):.4f}",
                f"{100 * np.mean(probs > 0.9):.1f}%",
                f"{100 * np.mean(probs < 0.3):.1f}%",
            ])

    logger.info(f"✅ Summary results:  {summary_path}")
    logger.info("\nDone! All models evaluated.")

    # ──────────────────────────────────────
    # Print comparison table to terminal
    # ──────────────────────────────────────
    print(f"\n{'='*80}")
    print("ENHANCER EVAL RESULTS — All images are FAKE (should be prob > 0.5)")
    print(f"{'='*80}")
    print(f"{'Model':<12} {'Acc@0.5':>8} {'Acc@0.7':>8} {'Mean':>8} "
          f"{'Median':>8} {'Min':>8} {'>0.9':>8} {'<0.3':>8}")
    print("-" * 80)
    for model_name in model_names:
        probs = np.array([
            all_results[model_name].get(fn, float("nan"))
            for fn in all_filenames
        ])
        probs = probs[~np.isnan(probs)]
        if len(probs) == 0:
            continue
        print(f"{model_name:<12} "
              f"{100*np.mean(probs>0.5):>7.1f}% "
              f"{100*np.mean(probs>0.7):>7.1f}% "
              f"{np.mean(probs):>8.4f} "
              f"{np.median(probs):>8.4f} "
              f"{np.min(probs):>8.4f} "
              f"{100*np.mean(probs>0.9):>7.1f}% "
              f"{100*np.mean(probs<0.3):>7.1f}%")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
