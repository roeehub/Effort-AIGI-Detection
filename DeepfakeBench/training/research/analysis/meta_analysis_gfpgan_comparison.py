#!/usr/bin/env python3
"""
meta_analysis_gfpgan_comparison.py – Compare GFPGAN-enhanced GCS samples with WMA.

Downloads frames from gs://live-deepfake-methods-real-and-fake-frames/samples/
for 4 source categories:
  1. edge_cases_enhanced (GFPGAN-enhanced fakes)
  2. minimal_processing_enhanced (GFPGAN-enhanced fakes)
  3. edge_cases (original, non-enhanced fakes)
  4. minimal_processing (original, non-enhanced fakes)

Computes 36 image properties (same as meta_analysis_enhancer.py) + model
predictions, then merges with the existing meta_analysis_properties.csv
and produces a comparison table.

Key question: do the GFPGAN-enhanced GCS frames have similar properties
to WMA enhanced fakes (the ones our model fails on)?  If so, the root
cause is GFPGAN smoothing.  If not, the root cause is
GFPGAN + resolution mismatch + different cropping pipeline.

Usage:
    cd DeepfakeBench/training
    python meta_analysis_gfpgan_comparison.py
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from tqdm import tqdm

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detectors import DETECTOR  # noqa: E402
from video_preprocessor import extract_yolo_face  # noqa: E402

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "analysis_results"
CACHE_DIR = Path(__file__).resolve().parent / "weights" / "enhancer_eval"
GCS_CACHE = Path(__file__).resolve().parent / "weights" / "gfpgan_comparison_cropped"

# GCS bucket (raw frames, NOT the cropped bucket)
RAW_BUCKET = "live-deepfake-methods-real-and-fake-frames"

SAMPLES_PER_STRATEGY = 150  # per strategy × label

# Model checkpoint (same as meta-analysis)
MODEL_CHECKPOINT = {
    "name": "R25_F1",
    "gcs": "gs://training-job-outputs/phase2r2_experiments/5w453our/"
           "top_n_effort_20260212_step18500_auc0.9893_eer0.0356.pth",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("gfpgan-compare")


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


# ═══════════════════════════════════════════
# IMAGE PROPERTY CALCULATIONS (same as meta_analysis_enhancer.py)
# ═══════════════════════════════════════════

def compute_image_properties(img_rgb: np.ndarray) -> Dict[str, float]:
    """Compute a comprehensive set of image properties from an RGB uint8 array."""
    h, w, c = img_rgb.shape
    props = {}

    # ── Basic dimensions ──
    props["width"] = w
    props["height"] = h
    props["aspect_ratio"] = w / h
    props["num_pixels"] = w * h

    # ── Convert to different color spaces ──
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    # ── Sharpness (Laplacian variance) ──
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    props["sharpness_laplacian_var"] = float(laplacian.var())
    props["sharpness_laplacian_mean"] = float(np.abs(laplacian).mean())

    # ── Sharpness (Tenengrad / Sobel gradient magnitude) ──
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(gx**2 + gy**2)
    props["sharpness_tenengrad"] = float(gradient_mag.mean())
    props["sharpness_tenengrad_var"] = float(gradient_mag.var())

    # ── Frequency spectrum analysis ──
    gray_224 = cv2.resize(img_gray, (224, 224))
    f_transform = np.fft.fft2(gray_224.astype(np.float64))
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    cy, cx = 112, 112
    Y, X = np.ogrid[:224, :224]
    radius = np.sqrt((X - cx)**2 + (Y - cy)**2)

    total_energy = magnitude.sum()
    if total_energy > 0:
        low_mask = radius < 20
        mid_mask = (radius >= 20) & (radius < 60)
        high_mask = radius >= 60
        props["freq_low_energy_ratio"] = float(magnitude[low_mask].sum() / total_energy)
        props["freq_mid_energy_ratio"] = float(magnitude[mid_mask].sum() / total_energy)
        props["freq_high_energy_ratio"] = float(magnitude[high_mask].sum() / total_energy)
        props["freq_high_to_low_ratio"] = float(
            magnitude[high_mask].sum() / max(magnitude[low_mask].sum(), 1e-8))
    else:
        props["freq_low_energy_ratio"] = 0.0
        props["freq_mid_energy_ratio"] = 0.0
        props["freq_high_energy_ratio"] = 0.0
        props["freq_high_to_low_ratio"] = 0.0

    # ── Color statistics (RGB channels) ──
    for i, ch_name in enumerate(["r", "g", "b"]):
        ch = img_rgb[:, :, i].astype(np.float64)
        props[f"color_{ch_name}_mean"] = float(ch.mean())
        props[f"color_{ch_name}_std"] = float(ch.std())

    # ── Luminance statistics ──
    L = img_lab[:, :, 0].astype(np.float64)
    props["luminance_mean"] = float(L.mean())
    props["luminance_std"] = float(L.std())

    # ── Contrast (RMS contrast) ──
    gray_float = img_gray.astype(np.float64) / 255.0
    props["contrast_rms"] = float(gray_float.std())
    props["contrast_michelson"] = float(
        (gray_float.max() - gray_float.min()) /
        max(gray_float.max() + gray_float.min(), 1e-8))

    # ── Saturation ──
    saturation = img_hsv[:, :, 1].astype(np.float64)
    props["saturation_mean"] = float(saturation.mean())
    props["saturation_std"] = float(saturation.std())

    # ── Hue statistics ──
    hue = img_hsv[:, :, 0].astype(np.float64)
    props["hue_mean"] = float(hue.mean())
    props["hue_std"] = float(hue.std())

    # ── Noise estimation (Immerkær method) ──
    noise_kernel = np.array([[1, -2, 1],
                              [-2, 4, -2],
                              [1, -2, 1]], dtype=np.float64)
    sigma = np.sum(np.abs(cv2.filter2D(img_gray.astype(np.float64), -1, noise_kernel)))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))
    props["noise_estimate"] = float(sigma)

    # ── JPEG quality estimate ──
    _, jpg_buf = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, 95])
    props["jpeg_compressibility"] = float(len(jpg_buf)) / float(w * h * 3)

    # ── Edge density (Canny) ──
    edges = cv2.Canny(img_gray, 100, 200)
    props["edge_density"] = float(edges.sum() / 255.0) / float(w * h)

    # ── Texture measure ──
    kernel_size = 3
    local_mean = cv2.blur(img_gray.astype(np.float64), (kernel_size, kernel_size))
    local_var = cv2.blur((img_gray.astype(np.float64) - local_mean)**2,
                         (kernel_size, kernel_size))
    props["texture_local_var_mean"] = float(local_var.mean())

    # ── Symmetry ──
    left_half = img_gray[:, :w//2]
    right_half = cv2.flip(img_gray[:, w//2:], 1)
    min_w = min(left_half.shape[1], right_half.shape[1])
    left_half = left_half[:, :min_w]
    right_half = right_half[:, :min_w]
    diff = np.abs(left_half.astype(np.float64) - right_half.astype(np.float64))
    props["symmetry_score"] = 1.0 - float(diff.mean() / 255.0)

    # ── Effective resolution ──
    sorted_mag = np.sort(magnitude.flatten())[::-1]
    cumsum = np.cumsum(sorted_mag)
    if cumsum[-1] > 0:
        idx_90 = np.searchsorted(cumsum, 0.90 * cumsum[-1])
        props["effective_resolution_90pct"] = float(idx_90) / float(len(sorted_mag))
    else:
        props["effective_resolution_90pct"] = 0.0

    return props


# ═══════════════════════════════════════════
# GCS DATA SAMPLING
# ═══════════════════════════════════════════

def _gcs_client():
    from google.cloud import storage
    return storage.Client()


def sample_gcs_frames(
    strategy_prefix: str,
    label: str,
    n_samples: int,
    source_name: str,
) -> List[Tuple[str, np.ndarray]]:
    """Download raw frames from GCS, apply YOLO face-crop → 224×224.

    Each downloaded 640×360 frame is passed through the same
    ``extract_yolo_face()`` pipeline used by the training preprocessor
    so that properties and model inference are directly comparable.

    Args:
        strategy_prefix: e.g. 'edge_cases_enhanced_', 'minimal_processing_'
        label: 'fake' or 'real'
        n_samples: how many samples to draw
        source_name: identifier for caching and output

    Returns:
        List of (identifier, rgb_array_224x224) tuples
    """
    client = _gcs_client()
    bucket = client.bucket(RAW_BUCKET)
    cache_dir = GCS_CACHE / source_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check cache first (cached files are already YOLO-cropped 224×224)
    cached = sorted([f for f in cache_dir.iterdir() if f.suffix in IMG_EXTS])
    if len(cached) >= n_samples:
        logger.info(f"  Using {n_samples} cached YOLO-cropped images for {source_name}")
        results = []
        for p in cached[:n_samples]:
            img = np.array(Image.open(p).convert("RGB"))
            results.append((f"{source_name}/{p.name}", img))
        return results

    logger.info(f"  Listing samples with prefix 'samples/{strategy_prefix}' ...")
    prefix = f"samples/{strategy_prefix}"

    # Collect sample directories by finding manifest.json files
    sample_dirs = []
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith("manifest.json"):
            sample_dir = blob.name.rsplit("/", 1)[0]
            sample_dirs.append(sample_dir)

    logger.info(f"  Found {len(sample_dirs)} sample directories for {source_name}")

    if not sample_dirs:
        logger.warning(f"  No samples found for {source_name}")
        return []

    rng = np.random.RandomState(42)
    rng.shuffle(sample_dirs)

    yolo_fail_count = 0
    results = []
    for sample_dir in tqdm(sample_dirs[:n_samples * 3],
                           desc=f"Downloading {source_name}", unit="sample"):
        if len(results) >= n_samples:
            break

        # Pick a random frame from frames/{label}/
        frame_prefix = f"{sample_dir}/frames/{label}/"
        frame_blobs = [b for b in bucket.list_blobs(prefix=frame_prefix)
                       if not b.name.endswith("/")]

        if not frame_blobs:
            continue

        fb = frame_blobs[rng.randint(len(frame_blobs))]

        try:
            img_bytes = fb.download_as_bytes()
            img_rgb = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

            # ── YOLO face crop (same pipeline as training data) ──
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            face_crop_bgr = extract_yolo_face(img_bgr)  # → 224×224 BGR or None
            if face_crop_bgr is None:
                yolo_fail_count += 1
                continue
            face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)

            # Cache the YOLO-cropped face (not the raw frame)
            parts = fb.name.split("/")
            sample_id = parts[1]  # e.g. edge_cases_enhanced_0000
            frame_name = parts[-1]  # e.g. frame_0005.png
            local_name = f"{sample_id}__{frame_name}"
            Image.fromarray(face_crop_rgb).save(cache_dir / local_name)

            results.append((f"{source_name}/{local_name}", face_crop_rgb))
        except Exception as e:
            logger.warning(f"  Error loading {fb.name}: {e}")

    if yolo_fail_count:
        logger.warning(f"  YOLO face detection failed on {yolo_fail_count} frames for {source_name}")
    logger.info(f"  Got {len(results)} YOLO-cropped images for {source_name}")
    return results


# ═══════════════════════════════════════════
# MODEL LOADING & INFERENCE
# ═══════════════════════════════════════════

_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711]),
])


def download_checkpoint(gcs_path: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = gcs_path.rsplit("/", 1)[-1]
    local_path = CACHE_DIR / filename
    if local_path.exists():
        return local_path
    logger.info(f"  Downloading checkpoint: {gcs_path}")
    result = subprocess.run(["gsutil", "cp", gcs_path, str(local_path)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gsutil failed: {result.stderr}")
    return local_path


def load_model(weights_path: Path) -> torch.nn.Module:
    ckpt = torch.load(str(weights_path), map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        model_config = ckpt.get("model_config", {})
    else:
        state_dict = ckpt
        model_config = {}

    base_cfg_path = Path(__file__).resolve().parent / "config" / "detector" / "effort.yaml"
    with open(base_cfg_path) as f:
        cfg = yaml.safe_load(f)

    if model_config:
        for key, value in model_config.items():
            if key != "current_arcface_s":
                cfg[key] = value

    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(DEVICE)

    if model_config.get("use_arcface_head", False) and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])

    state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def preprocess_for_model(img_rgb: np.ndarray) -> torch.Tensor:
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    return _transform(Image.fromarray(img_resized))


@torch.inference_mode()
def extract_features_and_probs(
    model: torch.nn.Module,
    images: List[np.ndarray],
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    all_features = []
    all_probs = []

    for start in tqdm(range(0, len(images), batch_size),
                      desc="Model inference", unit="batch"):
        batch_imgs = images[start:start + batch_size]
        tensors = torch.stack([preprocess_for_model(img) for img in batch_imgs])

        data = {
            "image": tensors.to(DEVICE),
            "label": torch.zeros(tensors.size(0), dtype=torch.long, device=DEVICE),
        }
        preds = model(data, inference=True)

        all_features.append(preds["feat"].cpu().numpy())
        all_probs.append(preds["prob"].squeeze(-1).cpu().numpy())

    return np.concatenate(all_features, axis=0), np.concatenate(all_probs, axis=0)


# ═══════════════════════════════════════════
# COMPARISON & REPORTING
# ═══════════════════════════════════════════

KEY_PROPS = [
    "sharpness_laplacian_var",
    "sharpness_tenengrad",
    "edge_density",
    "noise_estimate",
    "freq_high_energy_ratio",
    "freq_high_to_low_ratio",
    "contrast_rms",
    "saturation_mean",
    "jpeg_compressibility",
    "effective_resolution_90pct",
    "width",
    "height",
]


def load_existing_meta_analysis() -> Dict[str, List[Dict]]:
    """Load existing meta_analysis_properties.csv grouped by source."""
    csv_path = OUTPUT_DIR / "meta_analysis_properties.csv"
    if not csv_path.exists():
        logger.warning(f"  Existing meta-analysis not found at {csv_path}")
        return {}

    source_data: Dict[str, List[Dict]] = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for k in row:
                if k not in ("source", "label", "filename"):
                    try:
                        row[k] = float(row[k])
                    except (ValueError, TypeError):
                        pass
            source_data[row["source"]].append(row)

    logger.info(f"  Loaded existing meta-analysis: "
                f"{sum(len(v) for v in source_data.values())} rows, "
                f"{len(source_data)} sources")
    return source_data


def print_comparison_table(
    all_source_data: Dict[str, List[Dict]],
    source_order: List[str],
    source_labels: Dict[str, str],
):
    """Print a comprehensive comparison table."""

    print(f"\n{'='*130}")
    print("GFPGAN ENHANCED vs WMA vs TRAINING DATA — Property Comparison")
    print(f"{'='*130}")

    # ── Per-property comparison ──
    for prop in KEY_PROPS:
        print(f"\n── {prop} ──")
        print(f"  {'Source':<50} {'n':>5} {'Mean':>10} {'Median':>10} "
              f"{'Std':>10} {'Min':>10} {'Max':>10}")
        print(f"  {'-'*105}")
        for src in source_order:
            rows = all_source_data.get(src, [])
            if not rows:
                continue
            vals = []
            for r in rows:
                v = r.get(prop, None)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(float(v))
            if not vals:
                continue
            vals = np.array(vals)
            print(f"  {src:<50} {len(vals):>5} {np.mean(vals):>10.4f} "
                  f"{np.median(vals):>10.4f} {np.std(vals):>10.4f} "
                  f"{np.min(vals):>10.4f} {np.max(vals):>10.4f}")

    # ── Model prediction summary ──
    print(f"\n{'='*130}")
    print("MODEL PREDICTIONS (R25_F1)")
    print(f"{'='*130}")
    print(f"  {'Source':<50} {'Label':<6} {'N':>5} {'Acc@0.5':>8} "
          f"{'MeanProb':>10} {'MedianProb':>12}")
    print(f"  {'-'*105}")

    for src in source_order:
        rows = all_source_data.get(src, [])
        if not rows:
            continue
        label = source_labels.get(src, "?")
        probs = []
        for r in rows:
            p = r.get("model_fake_prob", None)
            if p is not None:
                probs.append(float(p))
        if not probs:
            continue
        ps = np.array(probs)
        if label == "fake":
            acc = np.mean(ps > 0.5)
        else:
            acc = np.mean(ps <= 0.5)
        print(f"  {src:<50} {label:<6} {len(ps):>5} {100*acc:>7.1f}% "
              f"{np.mean(ps):>10.4f} {np.median(ps):>12.4f}")

    # ── Head-to-head: GFPGAN enhanced GCS vs WMA ──
    print(f"\n{'='*130}")
    print("HEAD-TO-HEAD: GFPGAN GCS Enhanced vs WMA Enhanced")
    print(f"{'='*130}")

    gfpgan_sources = [s for s in source_order if "gfpgan" in s.lower()]
    wma_sources = [s for s in source_order if "wma" in s.lower()]
    original_sources = [s for s in source_order
                        if s.startswith("gcs_original_")]

    compare_groups = {
        "GFPGAN Enhanced (YOLO face crop)": gfpgan_sources,
        "WMA Enhanced (face crop)": wma_sources,
        "Original non-enhanced (YOLO face crop)": original_sources,
    }

    for prop in KEY_PROPS:
        print(f"\n  {prop}:")
        for group_name, srcs in compare_groups.items():
            all_vals = []
            for src in srcs:
                rows = all_source_data.get(src, [])
                for r in rows:
                    v = r.get(prop, None)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        all_vals.append(float(v))
            if all_vals:
                vals = np.array(all_vals)
                print(f"    {group_name:<45} n={len(vals):>4}  "
                      f"mean={np.mean(vals):>10.4f}  "
                      f"median={np.median(vals):>10.4f}  "
                      f"std={np.std(vals):>8.4f}")

    # ── GFPGAN effect: enhanced vs original (same bucket, same resolution) ──
    print(f"\n{'='*130}")
    print("GFPGAN EFFECT: Enhanced vs Original (same GCS bucket, same resolution)")
    print(f"{'='*130}")

    for strategy in ["edge_cases", "minimal_processing"]:
        enhanced_src = f"gfpgan_{strategy}_enhanced_fake"
        original_src = f"gcs_original_{strategy}_fake"

        enhanced_rows = all_source_data.get(enhanced_src, [])
        original_rows = all_source_data.get(original_src, [])

        if not enhanced_rows or not original_rows:
            continue

        print(f"\n  Strategy: {strategy}")
        print(f"  {'Property':<35} {'Enhanced':>12} {'Original':>12} {'Delta':>12} {'Δ%':>8}")
        print(f"  {'-'*80}")

        for prop in KEY_PROPS:
            e_vals = [float(r[prop]) for r in enhanced_rows
                      if r.get(prop) is not None]
            o_vals = [float(r[prop]) for r in original_rows
                      if r.get(prop) is not None]
            if e_vals and o_vals:
                e_mean = np.mean(e_vals)
                o_mean = np.mean(o_vals)
                delta = e_mean - o_mean
                pct = (delta / max(abs(o_mean), 1e-8)) * 100
                print(f"  {prop:<35} {e_mean:>12.4f} {o_mean:>12.4f} "
                      f"{delta:>+12.4f} {pct:>+7.1f}%")


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GCS_CACHE.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────
    # Step 1: Download frames from 4 GCS source categories
    # ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Downloading frames from GCS")
    logger.info("=" * 60)

    # Source definitions: (source_name, strategy_prefix, label, ground_truth)
    gcs_sources = [
        # GFPGAN-enhanced fakes
        ("gfpgan_edge_cases_enhanced_fake",
         "edge_cases_enhanced_", "fake", "fake"),
        ("gfpgan_minimal_processing_enhanced_fake",
         "minimal_processing_enhanced_", "fake", "fake"),
        # Original (non-enhanced) fakes for paired comparison
        ("gcs_original_edge_cases_fake",
         "edge_cases_", "fake", "fake"),
        ("gcs_original_minimal_processing_fake",
         "minimal_processing_", "fake", "fake"),
    ]

    all_source_images: List[Tuple[str, str, str, np.ndarray]] = []
    # (source_name, ground_truth_label, identifier, img_rgb)

    for source_name, prefix, label, gt_label in gcs_sources:
        logger.info(f"\n  Downloading: {source_name}")
        images = sample_gcs_frames(prefix, label, SAMPLES_PER_STRATEGY, source_name)
        for identifier, img_rgb in images:
            all_source_images.append((source_name, gt_label, identifier, img_rgb))

    logger.info(f"\nTotal new images: {len(all_source_images)}")
    for src_name, _, _, _ in gcs_sources:
        n = sum(1 for s, _, _, _ in all_source_images if s == src_name)
        logger.info(f"  {src_name}: {n}")

    # ──────────────────────────────────────
    # Step 2: Compute image properties
    # ──────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Computing image properties")
    logger.info("=" * 60)

    new_rows = []
    new_images = []
    new_meta = []

    for source_name, gt_label, identifier, img_rgb in tqdm(
        all_source_images, desc="Properties", unit="img"
    ):
        try:
            props = compute_image_properties(img_rgb)
            row = {
                "source": source_name,
                "label": gt_label,
                "filename": identifier,
                **props,
            }
            for k, v in row.items():
                if isinstance(v, (np.floating, np.integer)):
                    row[k] = float(v)
            new_rows.append(row)
            new_images.append(img_rgb)
            new_meta.append((source_name, gt_label, identifier))
        except Exception as e:
            logger.warning(f"  Error processing {identifier}: {e}")

    logger.info(f"  Computed properties for {len(new_rows)} images")

    # ──────────────────────────────────────
    # Step 3: Model inference
    # ──────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Model inference (R25_F1)")
    logger.info("=" * 60)

    ckpt_path = download_checkpoint(MODEL_CHECKPOINT["gcs"])
    model = load_model(ckpt_path)
    logger.info(f"  Model loaded: {MODEL_CHECKPOINT['name']}")

    features, probs = extract_features_and_probs(model, new_images)
    logger.info(f"  Features: {features.shape}, Probs: {probs.shape}")

    for i, row in enumerate(new_rows):
        row["model_fake_prob"] = float(probs[i])

    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    # ──────────────────────────────────────
    # Step 4: Save new data
    # ──────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Saving results")
    logger.info("=" * 60)

    # Save new properties CSV
    csv_path = OUTPUT_DIR / "gfpgan_comparison_properties.csv"
    if new_rows:
        fieldnames = list(new_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(new_rows)
        logger.info(f"  Saved: {csv_path} ({len(new_rows)} rows)")

    # Save features NPZ
    npz_path = OUTPUT_DIR / "gfpgan_comparison_features.npz"
    np.savez_compressed(
        npz_path,
        features=features,
        probs=probs,
        source_labels=np.array([m[0] for m in new_meta]),
        gt_labels=np.array([m[1] for m in new_meta]),
        filenames=np.array([m[2] for m in new_meta]),
    )
    logger.info(f"  Saved: {npz_path}")

    # ──────────────────────────────────────
    # Step 5: Load existing meta-analysis and compare
    # ──────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Comparison with existing meta-analysis")
    logger.info("=" * 60)

    existing_data = load_existing_meta_analysis()

    # Merge new data into source_data dict
    all_source_data = dict(existing_data)
    for row in new_rows:
        src = row["source"]
        if src not in all_source_data:
            all_source_data[src] = []
        all_source_data[src].append(row)

    # Source display order: new sources first, then relevant existing
    source_order = [
        # New GFPGAN-enhanced sources
        "gfpgan_edge_cases_enhanced_fake",
        "gfpgan_minimal_processing_enhanced_fake",
        # New originals for comparison
        "gcs_original_edge_cases_fake",
        "gcs_original_minimal_processing_fake",
        # Existing WMA (the problem source)
        "wma_enhanced",
        # Existing training data (cropped to 224x224)
        "deeplive_quality_enhancement_fake",
        "deeplive_minimal_processing_fake",
        "deeplive_edge_cases_fake",
        "visomaster_fake",
        "df40_fake",
        # Existing reals
        "deeplive_quality_enhancement_real",
        "deeplive_minimal_processing_real",
        "df40_real",
        "external_youtube_real",
    ]

    source_labels = {
        "gfpgan_edge_cases_enhanced_fake": "fake",
        "gfpgan_minimal_processing_enhanced_fake": "fake",
        "gcs_original_edge_cases_fake": "fake",
        "gcs_original_minimal_processing_fake": "fake",
        "wma_enhanced": "fake",
        "deeplive_quality_enhancement_fake": "fake",
        "deeplive_minimal_processing_fake": "fake",
        "deeplive_edge_cases_fake": "fake",
        "visomaster_fake": "fake",
        "df40_fake": "fake",
        "deeplive_quality_enhancement_real": "real",
        "deeplive_minimal_processing_real": "real",
        "df40_real": "real",
        "external_youtube_real": "real",
    }

    print_comparison_table(all_source_data, source_order, source_labels)

    # ──────────────────────────────────────
    # Step 6: Print key takeaways
    # ──────────────────────────────────────
    print(f"\n{'='*130}")
    print("KEY TAKEAWAYS")
    print(f"{'='*130}")

    # Gather stats for summary
    def get_mean(src: str, prop: str) -> Optional[float]:
        rows = all_source_data.get(src, [])
        vals = [float(r[prop]) for r in rows if r.get(prop) is not None]
        return np.mean(vals) if vals else None

    def get_acc(src: str, label: str) -> Optional[float]:
        rows = all_source_data.get(src, [])
        probs = [float(r["model_fake_prob"]) for r in rows
                 if r.get("model_fake_prob") is not None]
        if not probs:
            return None
        ps = np.array(probs)
        if label == "fake":
            return float(np.mean(ps > 0.5))
        return float(np.mean(ps <= 0.5))

    # Key comparisons
    gfpgan_ec_sharp = get_mean("gfpgan_edge_cases_enhanced_fake",
                               "sharpness_laplacian_var")
    gfpgan_mp_sharp = get_mean("gfpgan_minimal_processing_enhanced_fake",
                               "sharpness_laplacian_var")
    orig_ec_sharp = get_mean("gcs_original_edge_cases_fake",
                             "sharpness_laplacian_var")
    orig_mp_sharp = get_mean("gcs_original_minimal_processing_fake",
                             "sharpness_laplacian_var")
    wma_sharp = get_mean("wma_enhanced", "sharpness_laplacian_var")
    train_qe_sharp = get_mean("deeplive_quality_enhancement_fake",
                              "sharpness_laplacian_var")

    gfpgan_ec_acc = get_acc("gfpgan_edge_cases_enhanced_fake", "fake")
    gfpgan_mp_acc = get_acc("gfpgan_minimal_processing_enhanced_fake", "fake")
    orig_ec_acc = get_acc("gcs_original_edge_cases_fake", "fake")
    orig_mp_acc = get_acc("gcs_original_minimal_processing_fake", "fake")

    def _f(v, fmt=".1f"):
        """Format a value safely, returning 'N/A' for None."""
        if v is None:
            return "N/A"
        return f"{v:{fmt}}"

    def _safe_sub(a, b, fmt="+.1f"):
        if a is None or b is None:
            return "N/A"
        return f"{a - b:{fmt}}"

    def _safe_avg(a, b):
        vals = [x for x in (a, b) if x is not None]
        return np.mean(vals) if vals else None

    avg_gfpgan_sharp = _safe_avg(gfpgan_ec_sharp, gfpgan_mp_sharp)
    avg_orig_sharp = _safe_avg(orig_ec_sharp, orig_mp_sharp)
    avg_gfpgan_acc = _safe_avg(gfpgan_ec_acc, gfpgan_mp_acc)
    avg_orig_acc = _safe_avg(orig_ec_acc, orig_mp_acc)

    print(f"""
  1. GFPGAN EFFECT ON SHARPNESS (YOLO face crops, 224×224):
     Edge-cases:    enhanced={_f(gfpgan_ec_sharp)}  original={_f(orig_ec_sharp)}  Δ={_safe_sub(gfpgan_ec_sharp, orig_ec_sharp)}
     MinProc:       enhanced={_f(gfpgan_mp_sharp)}  original={_f(orig_mp_sharp)}  Δ={_safe_sub(gfpgan_mp_sharp, orig_mp_sharp)}

  2. COMPARISON WITH WMA & TRAINING DATA (all face crops):
     WMA enhanced (face crops, ~342×435):     sharpness={_f(wma_sharp)}
     GFPGAN GCS enhanced (YOLO crop, 224×224): sharpness=~{_f(avg_gfpgan_sharp)}
     Training QE fakes (cropped, 224×224):     sharpness={_f(train_qe_sharp)}

  3. MODEL ACCURACY ON GFPGAN ENHANCED FACE CROPS:
     Edge-cases enhanced:   {_f(100*gfpgan_ec_acc if gfpgan_ec_acc is not None else None)}%
     MinProc enhanced:      {_f(100*gfpgan_mp_acc if gfpgan_mp_acc is not None else None)}%
     Original edge-cases:   {_f(100*orig_ec_acc if orig_ec_acc is not None else None)}%
     Original MinProc:      {_f(100*orig_mp_acc if orig_mp_acc is not None else None)}%

  4. INTERPRETATION:
""")

    # Determine which hypothesis the data supports
    # Now comparing apples-to-apples: all sources are face crops
    if avg_gfpgan_sharp is None or avg_gfpgan_acc is None:
        print("     → Insufficient data to determine hypothesis.")
        print("       Some sources may not have been loaded. Check logs above.")
    elif avg_gfpgan_acc < 0.5:
        print("     → GFPGAN smoothing ALONE causes detection failure")
        print("       Even with proper YOLO face cropping (224×224), the model")
        print("       fails on GFPGAN-enhanced face crops.")
        if wma_sharp is not None and avg_gfpgan_sharp is not None and abs(avg_gfpgan_sharp - wma_sharp) < 10:
            print(f"       Sharpness profile ({avg_gfpgan_sharp:.1f}) matches WMA ({wma_sharp:.1f}) — same root cause.")
        elif wma_sharp is not None and avg_gfpgan_sharp is not None:
            print(f"       Sharpness ({avg_gfpgan_sharp:.1f}) differs from WMA ({wma_sharp:.1f})")
            print("       — similar failure mode but different severity.")
    elif avg_gfpgan_acc > 0.8:
        print("     → GFPGAN smoothing alone does NOT cause failure")
        print("       The model correctly detects GFPGAN-enhanced face crops.")
        print("       WMA failure must be due to other factors in the WMA pipeline")
        print("       (different face detection, resolution, cropping method).")
    else:
        print("     → PARTIAL: GFPGAN smoothing degrades but doesn't fully break detection")
        print(f"       Accuracy dropped from {_f(100*avg_orig_acc if avg_orig_acc is not None else None)}% "
              f"to {_f(100*avg_gfpgan_acc if avg_gfpgan_acc is not None else None)}%")
        wma_acc = get_acc("wma_enhanced", "fake")
        if wma_acc is not None:
            print(f"       WMA failure ({100*wma_acc:.1f}%) is worse — WMA cropping/resolution amplifies the effect.")
        else:
            print("       WMA accuracy not available for comparison.")

    print(f"\n{'='*130}")
    print(f"Output files:")
    print(f"  Properties: {csv_path}")
    print(f"  Features:   {npz_path}")
    print(f"{'='*130}")


if __name__ == "__main__":
    main()
