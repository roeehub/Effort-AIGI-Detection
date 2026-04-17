#!/usr/bin/env python3
"""
V2 — Resolution Isolation Test
================================
Separates two confounded effects:
  A) GFPGAN smoothing (removes high-freq artifacts)
  B) Downscaling from higher resolution (342→224, further reduces sharpness)

Method:
  Take DL quality_enhancement_fake images (98% acc, 224×224) →
  upsample to 342×435 → downsample back to 224×224 → re-infer.

If accuracy drops significantly: resolution mismatch is a major contributor.
If accuracy stays high: GFPGAN + different crop pipeline is the primary driver.

Also tests the reverse: take WMA images but resize with INTER_LANCZOS4 (sharper)
instead of INTER_LINEAR to see if resize quality matters.

Requires:
  - DL quality_enhancement images (downloaded from GCS, ~150 images)
  - R25_F1 checkpoint
  - WMA images (for reverse test)

Usage:
    cd DeepfakeBench/training
    python ../../analysis_results/verification/V2_resolution_isolation.py

Output:
    analysis_results/verification/V2_resolution_results.csv
    analysis_results/verification/V2_resolution_summary.txt
"""

from __future__ import annotations

import csv
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image as pil_image

TRAINING_DIR = Path(__file__).resolve().parent.parent.parent / "DeepfakeBench" / "training"
sys.path.insert(0, str(TRAINING_DIR))
from detectors import DETECTOR  # noqa: E402

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
WMA_IMAGE_DIR = Path("/Users/roeedar/Downloads/wma_export/all_images")
OUTPUT_DIR = Path(__file__).resolve().parent
CACHE_DIR = TRAINING_DIR / "weights" / "enhancer_eval"
DL_CACHE_DIR = OUTPUT_DIR / "dl_qe_images"   # downloaded DL QE images go here

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

R25_F1_GCS = ("gs://training-job-outputs/phase2r2_experiments/5w453our/"
              "top_n_effort_20260212_step18500_auc0.9893_eer0.0356.pth")

# GCS path for DeepLive quality_enhancement fake images
DL_QE_BUCKET = "deepfake-detection-training-data"
DL_QE_PREFIX = "samples/quality_enhancement_fake"

BATCH_SIZE = 32
MAX_DL_IMAGES = 200  # sample from GCS

# ─────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("V2-resolution")

def _resolve_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = _resolve_device()

_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711]),
])


def compute_sharpness(img_gray):
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()


# ─────────────────────────────────────────
# Model loading (reused from V1)
# ─────────────────────────────────────────
def download_checkpoint(gcs_path):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = gcs_path.rsplit("/", 1)[-1]
    local_path = CACHE_DIR / filename
    if local_path.exists():
        return local_path
    logger.info(f"Downloading checkpoint: {gcs_path}")
    result = subprocess.run(["gsutil", "cp", gcs_path, str(local_path)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gsutil failed: {result.stderr}")
    return local_path


def load_model(weights_path):
    ckpt = torch.load(str(weights_path), map_location=DEVICE, weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model_config = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}

    cfg_path = TRAINING_DIR / "config" / "detector" / "effort.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    for k, v in model_config.items():
        if k != "current_arcface_s":
            cfg[k] = v

    model = DETECTOR[cfg["model_name"]](cfg).to(DEVICE)
    if model_config.get("use_arcface_head", False) and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])

    state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.inference_mode()
def run_batch(model, tensors):
    data = {"image": tensors.to(DEVICE),
            "label": torch.zeros(tensors.size(0), dtype=torch.long, device=DEVICE)}
    preds = model(data, inference=True)
    return preds["prob"].squeeze(-1).cpu().numpy()


# ─────────────────────────────────────────
# Download DL QE images from GCS
# ─────────────────────────────────────────
def download_dl_qe_images():
    """Download a sample of DL quality_enhancement fake images from GCS."""
    DL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    existing = list(DL_CACHE_DIR.glob("*.jpg")) + list(DL_CACHE_DIR.glob("*.png"))
    if len(existing) >= 50:
        logger.info(f"Using {len(existing)} cached DL QE images from {DL_CACHE_DIR}")
        return sorted(existing)

    logger.info("Downloading DL quality_enhancement images from GCS...")
    # List blobs
    result = subprocess.run(
        ["gsutil", "ls", f"gs://{DL_QE_BUCKET}/{DL_QE_PREFIX}*"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error(f"gsutil ls failed: {result.stderr}")
        logger.info("Trying alternative path pattern...")
        # Try listing subdirectories
        result = subprocess.run(
            ["gsutil", "ls", f"gs://{DL_QE_BUCKET}/samples/"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.error(f"Cannot list GCS bucket: {result.stderr}")
            return []
        logger.info(f"Available prefixes:\n{result.stdout[:2000]}")
        return []

    blobs = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
    image_blobs = [b for b in blobs if any(b.lower().endswith(ext) for ext in IMG_EXTS)]
    logger.info(f"Found {len(image_blobs)} DL QE images in GCS")

    # Sample
    np.random.seed(42)
    sample = np.random.choice(image_blobs, min(MAX_DL_IMAGES, len(image_blobs)), replace=False)

    for i, blob in enumerate(sample):
        fname = blob.rsplit("/", 1)[-1]
        local = DL_CACHE_DIR / fname
        if local.exists():
            continue
        subprocess.run(["gsutil", "cp", blob, str(local)], capture_output=True, text=True)
        if (i + 1) % 50 == 0:
            logger.info(f"  Downloaded {i+1}/{len(sample)}")

    paths = sorted([p for p in DL_CACHE_DIR.iterdir() if p.suffix.lower() in IMG_EXTS])
    logger.info(f"DL QE images ready: {len(paths)}")
    return paths


# ─────────────────────────────────────────
# Experimental conditions
# ─────────────────────────────────────────
def process_image(img_path, condition):
    """Load and process an image according to the experimental condition.
    
    Returns: (tensor_224, sharpness_before_resize, sharpness_after_resize, original_size)
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Cannot read: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    orig_size = (w, h)

    if condition == "dl_original":
        # DL QE image as-is (should be 224×224 already)
        resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    elif condition == "dl_updown_linear":
        # Upsample to WMA-like resolution, then downsample back
        upsampled = cv2.resize(img_rgb, (342, 435), interpolation=cv2.INTER_LINEAR)
        resized = cv2.resize(upsampled, (224, 224), interpolation=cv2.INTER_LINEAR)

    elif condition == "dl_updown_lanczos":
        # Same but with LANCZOS (sharper) interpolation on the downsample
        upsampled = cv2.resize(img_rgb, (342, 435), interpolation=cv2.INTER_LINEAR)
        resized = cv2.resize(upsampled, (224, 224), interpolation=cv2.INTER_LANCZOS4)

    elif condition == "dl_updown_area":
        # INTER_AREA is best for downsampling (anti-aliasing)
        upsampled = cv2.resize(img_rgb, (342, 435), interpolation=cv2.INTER_LINEAR)
        resized = cv2.resize(upsampled, (224, 224), interpolation=cv2.INTER_AREA)

    elif condition == "wma_linear":
        # WMA image resized with default INTER_LINEAR
        resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    elif condition == "wma_lanczos":
        # WMA image resized with LANCZOS (sharper)
        resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LANCZOS4)

    elif condition == "wma_area":
        # WMA image with INTER_AREA
        resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)

    else:
        raise ValueError(f"Unknown condition: {condition}")

    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    sharpness_224 = compute_sharpness(gray)

    gray_orig = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sharpness_orig = compute_sharpness(gray_orig)

    tensor = _transform(pil_image.fromarray(resized))
    return tensor, sharpness_orig, sharpness_224, orig_size


def run_condition(model, image_paths, condition, label="fake"):
    """Run one condition, return results list."""
    results = []
    tensors = []
    metas = []

    for p in image_paths:
        try:
            t, s_orig, s_224, orig_size = process_image(p, condition)
            tensors.append(t)
            metas.append((p.name, s_orig, s_224, orig_size))
        except Exception as e:
            logger.warning(f"Skipping {p.name}: {e}")

    all_probs = []
    for i in range(0, len(tensors), BATCH_SIZE):
        batch = torch.stack(tensors[i:i + BATCH_SIZE])
        probs = run_batch(model, batch)
        all_probs.extend(probs.tolist())

    for (fname, s_orig, s_224, orig_size), prob in zip(metas, all_probs):
        results.append({
            "condition": condition,
            "filename": fname,
            "label": label,
            "fake_prob": prob,
            "sharpness_original": s_orig,
            "sharpness_224": s_224,
            "orig_width": orig_size[0],
            "orig_height": orig_size[1],
        })

    return results


def main():
    # ── Load model ──────────────────────────────────────────────────
    ckpt_path = download_checkpoint(R25_F1_GCS)
    model = load_model(ckpt_path)
    logger.info("R25_F1 model loaded\n")

    all_results = []
    summary = []

    # ── Part A: DL QE images (test resolution round-trip) ───────────
    dl_paths = download_dl_qe_images()

    if dl_paths:
        dl_conditions = ["dl_original", "dl_updown_linear", "dl_updown_lanczos", "dl_updown_area"]
        for cond in dl_conditions:
            logger.info(f"Running: {cond} ({len(dl_paths)} images)")
            results = run_condition(model, dl_paths, cond, label="fake")
            all_results.extend(results)

            probs = np.array([r["fake_prob"] for r in results])
            s224 = np.array([r["sharpness_224"] for r in results])
            summary.append({
                "condition": cond,
                "source": "DL_QE",
                "n": len(results),
                "acc_0.5": f"{100*np.mean(probs > 0.5):.1f}%",
                "acc_0.7": f"{100*np.mean(probs > 0.7):.1f}%",
                "mean_prob": f"{np.mean(probs):.4f}",
                "median_prob": f"{np.median(probs):.4f}",
                "mean_sharp_224": f"{np.mean(s224):.1f}",
            })
    else:
        logger.warning("No DL QE images available — skipping Part A")

    # ── Part B: WMA images (test interpolation method) ──────────────
    wma_paths = sorted([p for p in WMA_IMAGE_DIR.iterdir() if p.suffix.lower() in IMG_EXTS])
    logger.info(f"\nFound {len(wma_paths)} WMA images")

    wma_conditions = ["wma_linear", "wma_lanczos", "wma_area"]
    for cond in wma_conditions:
        logger.info(f"Running: {cond} ({len(wma_paths)} images)")
        results = run_condition(model, wma_paths, cond, label="fake")
        all_results.extend(results)

        probs = np.array([r["fake_prob"] for r in results])
        s224 = np.array([r["sharpness_224"] for r in results])
        summary.append({
            "condition": cond,
            "source": "WMA",
            "n": len(results),
            "acc_0.5": f"{100*np.mean(probs > 0.5):.1f}%",
            "acc_0.7": f"{100*np.mean(probs > 0.7):.1f}%",
            "mean_prob": f"{np.mean(probs):.4f}",
            "median_prob": f"{np.median(probs):.4f}",
            "mean_sharp_224": f"{np.mean(s224):.1f}",
        })

    # ── Save per-image CSV ──────────────────────────────────────────
    detail_path = OUTPUT_DIR / "V2_resolution_results.csv"
    try:
        import pandas as pd
        pd.DataFrame(all_results).to_csv(detail_path, index=False)
    except ImportError:
        with open(detail_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
    logger.info(f"\n✅ Per-image results: {detail_path}")

    # ── Print summary ───────────────────────────────────────────────
    lines = []
    def log(msg):
        print(msg)
        lines.append(msg)

    log("\n" + "=" * 95)
    log("V2 RESOLUTION ISOLATION TEST — RESULTS")
    log("=" * 95)
    log(f"{'Condition':<25s} {'Source':>6s} {'N':>5s} {'Acc@0.5':>8s} {'Acc@0.7':>8s} "
        f"{'MeanProb':>9s} {'MedianP':>8s} {'Sharp224':>9s}")
    log("-" * 95)

    for s in summary:
        log(f"{s['condition']:<25s} {s['source']:>6s} {s['n']:>5d} {s['acc_0.5']:>8s} "
            f"{s['acc_0.7']:>8s} {s['mean_prob']:>9s} {s['median_prob']:>8s} "
            f"{s['mean_sharp_224']:>9s}")

    # ── Verdict ─────────────────────────────────────────────────────
    log("\n" + "=" * 95)
    log("VERDICT")
    log("=" * 95)

    # Compare DL original vs updown
    dl_orig = next((s for s in summary if s["condition"] == "dl_original"), None)
    dl_updown = next((s for s in summary if s["condition"] == "dl_updown_linear"), None)

    if dl_orig and dl_updown:
        orig_acc = float(dl_orig["acc_0.5"].rstrip("%"))
        updown_acc = float(dl_updown["acc_0.5"].rstrip("%"))
        delta = updown_acc - orig_acc

        log(f"\n  DL QE original (224×224):       {orig_acc:.1f}% accuracy")
        log(f"  DL QE up→342×435→down→224×224:  {updown_acc:.1f}% accuracy (Δ = {delta:+.1f}pp)")
        log(f"  Sharpness: {dl_orig['mean_sharp_224']} → {dl_updown['mean_sharp_224']}")

        if delta < -20:
            log(f"\n✅ RESOLUTION MISMATCH IS A MAJOR CONTRIBUTOR (Δ = {delta:+.1f}pp)")
            log("   Simply upsampling+downsampling destroys discriminative features.")
            log("   Training with multi-resolution augmentation is essential.")
        elif delta < -5:
            log(f"\n⚠️  Resolution has moderate effect (Δ = {delta:+.1f}pp)")
            log("   Contributes to the problem but isn't the sole cause.")
        else:
            log(f"\n🔍 Resolution has minimal effect (Δ = {delta:+.1f}pp)")
            log("   The main issue is GFPGAN smoothing / different crop pipeline,")
            log("   not the resolution round-trip.")
    else:
        log("\n  ⚠️  DL QE images not available — cannot assess resolution isolation.")

    # Compare WMA interpolation methods
    wma_linear = next((s for s in summary if s["condition"] == "wma_linear"), None)
    wma_lanczos = next((s for s in summary if s["condition"] == "wma_lanczos"), None)
    if wma_linear and wma_lanczos:
        lin_acc = float(wma_linear["acc_0.5"].rstrip("%"))
        lan_acc = float(wma_lanczos["acc_0.5"].rstrip("%"))
        log(f"\n  WMA with INTER_LINEAR:   {lin_acc:.1f}% accuracy")
        log(f"  WMA with INTER_LANCZOS4: {lan_acc:.1f}% accuracy (Δ = {lan_acc-lin_acc:+.1f}pp)")
        if lan_acc > lin_acc + 5:
            log("  → Sharper interpolation helps! Inference pipeline matters.")
        else:
            log("  → Interpolation method doesn't matter much.")

    summary_path = OUTPUT_DIR / "V2_resolution_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    log(f"\n✅ Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
