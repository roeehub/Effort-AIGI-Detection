#!/usr/bin/env python3
"""
meta_analysis_fix_missing.py – Download the 4 missing DeepLive sources,
compute properties & features, and merge into existing results.

Missing sources:
  - deeplive_quality_enhancement_fake  (CRITICAL: same pipeline as WMA)
  - deeplive_quality_enhancement_real
  - deeplive_minimal_processing_fake   (same tool, no enhancer)
  - deeplive_minimal_processing_real

The bug was: sample_deeplive_images() used prefix='samples/' with max_results=10000,
which was dominated by visomaster entries and never reached QE/MP manifests.
Fix: use prefix='samples/{strategy}_' to target the right directories.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detectors import DETECTOR  # noqa: E402

# ─── Config ───
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "analysis_results"
CACHE_DIR = Path(__file__).resolve().parent / "weights" / "enhancer_eval"
GCS_SAMPLE_CACHE = Path(__file__).resolve().parent / "weights" / "meta_analysis_samples"

DEEPLIVE_BUCKET = "live-deepfake-methods-real-and-fake-frames-cropped"
SAMPLES_PER_SOURCE = 150
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODEL_CHECKPOINT = {
    "name": "R25_F1",
    "gcs": "gs://training-job-outputs/phase2r2_experiments/5w453our/"
           "top_n_effort_20260212_step18500_auc0.9893_eer0.0356.pth",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("fix-missing")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


# ═══════════════════════════════════════════
# Image property computation (copied from meta_analysis_enhancer.py)
# ═══════════════════════════════════════════

def compute_image_properties(img_rgb: np.ndarray) -> Dict[str, float]:
    """Compute a comprehensive set of image properties from an RGB uint8 array."""
    h, w, c = img_rgb.shape
    props = {}
    props["width"] = w
    props["height"] = h
    props["aspect_ratio"] = w / h
    props["num_pixels"] = w * h

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    props["sharpness_laplacian_var"] = float(laplacian.var())
    props["sharpness_laplacian_mean"] = float(np.abs(laplacian).mean())

    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(gx**2 + gy**2)
    props["sharpness_tenengrad"] = float(gradient_mag.mean())
    props["sharpness_tenengrad_var"] = float(gradient_mag.var())

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

    for i, ch_name in enumerate(["r", "g", "b"]):
        ch = img_rgb[:, :, i].astype(np.float64)
        props[f"color_{ch_name}_mean"] = float(ch.mean())
        props[f"color_{ch_name}_std"] = float(ch.std())

    L = img_lab[:, :, 0].astype(np.float64)
    props["luminance_mean"] = float(L.mean())
    props["luminance_std"] = float(L.std())

    gray_float = img_gray.astype(np.float64) / 255.0
    props["contrast_rms"] = float(gray_float.std())
    gmin, gmax = gray_float.min(), gray_float.max()
    props["contrast_michelson"] = float((gmax - gmin) / max(gmax + gmin, 1e-8))

    saturation = img_hsv[:, :, 1].astype(np.float64)
    props["saturation_mean"] = float(saturation.mean())
    props["saturation_std"] = float(saturation.std())

    hue = img_hsv[:, :, 0].astype(np.float64)
    props["hue_mean"] = float(hue.mean())
    props["hue_std"] = float(hue.std())

    noise_kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    sigma = np.sum(np.abs(cv2.filter2D(img_gray.astype(np.float64), -1, noise_kernel)))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))
    props["noise_estimate"] = float(sigma)

    _, jpg_buf = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, 95])
    props["jpeg_compressibility"] = float(len(jpg_buf)) / float(w * h * 3)

    edges = cv2.Canny(img_gray, 100, 200)
    props["edge_density"] = float(edges.sum() / 255.0) / float(w * h)

    kernel_size = 3
    local_mean = cv2.blur(img_gray.astype(np.float64), (kernel_size, kernel_size))
    local_var = cv2.blur((img_gray.astype(np.float64) - local_mean)**2,
                         (kernel_size, kernel_size))
    props["texture_local_var_mean"] = float(local_var.mean())

    left_half = img_gray[:, :w//2]
    right_half = cv2.flip(img_gray[:, w//2:], 1)
    min_w = min(left_half.shape[1], right_half.shape[1])
    left_half = left_half[:, :min_w]
    right_half = right_half[:, :min_w]
    diff = np.abs(left_half.astype(np.float64) - right_half.astype(np.float64))
    props["symmetry_score"] = 1.0 - float(diff.mean() / 255.0)

    sorted_mag = np.sort(magnitude.flatten())[::-1]
    cumsum = np.cumsum(sorted_mag)
    if cumsum[-1] > 0:
        idx_90 = np.searchsorted(cumsum, 0.90 * cumsum[-1])
        props["effective_resolution_90pct"] = float(idx_90) / float(len(sorted_mag))
    else:
        props["effective_resolution_90pct"] = 0.0

    return props


# ═══════════════════════════════════════════
# GCS sampling — FIXED version
# ═══════════════════════════════════════════

def sample_deeplive_fixed(strategy: str, label: str, n: int) -> List[Tuple[str, np.ndarray]]:
    """Sample n images using strategy-specific prefix (fixed version)."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(DEEPLIVE_BUCKET)
    cache_dir = GCS_SAMPLE_CACHE / f"deeplive_{strategy}_{label}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Clear empty cache dir from previous failed attempt
    cached = sorted([f for f in cache_dir.iterdir() if f.suffix in IMG_EXTS])
    if len(cached) >= n:
        logger.info(f"  Using {n} cached images for deeplive/{strategy}/{label}")
        results = []
        for p in cached[:n]:
            img = np.array(Image.open(p).convert("RGB"))
            results.append((f"deeplive_{strategy}_{label}/{p.name}", img))
        return results

    logger.info(f"  Sampling {n} images from gs://{DEEPLIVE_BUCKET}")
    logger.info(f"  Using targeted prefix: samples/{strategy}_")

    # KEY FIX: Use strategy-specific prefix instead of scanning all samples/
    prefix = f"samples/{strategy}_"
    matching_samples = []
    blob_count = 0
    for blob in bucket.list_blobs(prefix=prefix):
        blob_count += 1
        if blob.name.endswith("manifest.json"):
            sample_dir = blob.name.rsplit("/", 1)[0]
            matching_samples.append(sample_dir)
            if len(matching_samples) >= n * 3:
                break

    logger.info(f"  Scanned {blob_count} blobs, found {len(matching_samples)} sample directories")

    if not matching_samples:
        logger.warning(f"  No samples found for strategy={strategy}")
        return []

    rng = np.random.RandomState(42)
    rng.shuffle(matching_samples)

    results = []
    errors = 0
    for sample_dir in tqdm(matching_samples, desc=f"{strategy}/{label}", unit="sample"):
        if len(results) >= n:
            break

        frame_prefix = f"{sample_dir}/frames/{label}/"
        frame_blobs = list(bucket.list_blobs(prefix=frame_prefix))
        frame_blobs = [b for b in frame_blobs if not b.name.endswith("/")]

        if not frame_blobs:
            continue

        fb = frame_blobs[rng.randint(len(frame_blobs))]

        try:
            img_bytes = fb.download_as_bytes()
            img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

            local_name = fb.name.replace("/", "_")
            Image.fromarray(img).save(cache_dir / f"{local_name}.png")

            identifier = f"deeplive_{strategy}_{label}/{fb.name.split('/')[-1]}"
            results.append((identifier, img))
        except Exception as e:
            errors += 1
            if errors <= 3:
                logger.warning(f"  Error loading {fb.name}: {e}")

    logger.info(f"  ✅ Got {len(results)} images for deeplive/{strategy}/{label} "
                f"({errors} errors)")
    return results


# ═══════════════════════════════════════════
# Model loading & feature extraction
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
def extract_features_and_probs(model, images, batch_size=32):
    all_features = []
    all_probs = []
    pbar = tqdm(total=len(images), desc="Feature extraction", unit="img")
    for start in range(0, len(images), batch_size):
        batch_imgs = images[start:start + batch_size]
        tensors = torch.stack([preprocess_for_model(img) for img in batch_imgs])
        data = {
            "image": tensors.to(DEVICE),
            "label": torch.zeros(tensors.size(0), dtype=torch.long, device=DEVICE),
        }
        preds = model(data, inference=True)
        all_features.append(preds["feat"].cpu().numpy())
        all_probs.append(preds["prob"].squeeze(-1).cpu().numpy())
        pbar.update(len(batch_imgs))
    pbar.close()
    return np.concatenate(all_features), np.concatenate(all_probs)


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("FIX: Downloading 4 missing DeepLive sources")
    logger.info("=" * 60)

    # Define the 4 missing sources
    missing_sources = [
        ("quality_enhancement", "fake", "deeplive_quality_enhancement_fake"),
        ("quality_enhancement", "real", "deeplive_quality_enhancement_real"),
        ("minimal_processing", "fake", "deeplive_minimal_processing_fake"),
        ("minimal_processing", "real", "deeplive_minimal_processing_real"),
    ]

    # Download all 4
    new_images = {}  # source_name -> [(identifier, img_rgb), ...]
    for strategy, label, source_name in missing_sources:
        logger.info(f"\n{'─'*50}")
        logger.info(f"Downloading: {source_name}")
        logger.info(f"{'─'*50}")
        imgs = sample_deeplive_fixed(strategy, label, SAMPLES_PER_SOURCE)
        new_images[source_name] = imgs
        logger.info(f"  → {len(imgs)} images")

    # Compute image properties for new sources
    logger.info(f"\n{'='*60}")
    logger.info("Computing image properties for new sources")
    logger.info(f"{'='*60}")

    new_rows = []
    new_meta = []
    new_imgs_flat = []

    for strategy, label, source_name in missing_sources:
        imgs = new_images[source_name]
        if not imgs:
            logger.warning(f"  Skipping {source_name} — no images")
            continue

        for identifier, img_rgb in tqdm(imgs, desc=f"Props: {source_name}", unit="img"):
            try:
                props = compute_image_properties(img_rgb)
                row = {"source": source_name, "label": label, "filename": identifier, **props}
                for k, v in row.items():
                    if isinstance(v, (np.floating, np.integer)):
                        row[k] = float(v)
                new_rows.append(row)
                new_meta.append((source_name, label, identifier))
                new_imgs_flat.append(img_rgb)
            except Exception as e:
                logger.warning(f"  Error processing {identifier}: {e}")

    logger.info(f"\n  New rows computed: {len(new_rows)}")

    if not new_rows:
        logger.error("No new images collected! Check GCS access.")
        return

    # Extract features + probabilities
    logger.info(f"\n{'='*60}")
    logger.info("Extracting CLIP features for new sources")
    logger.info(f"{'='*60}")

    ckpt_path = download_checkpoint(MODEL_CHECKPOINT["gcs"])
    model = load_model(ckpt_path)
    new_features, new_probs = extract_features_and_probs(model, new_imgs_flat)
    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    # Add probs to rows
    for i, row in enumerate(new_rows):
        row["model_fake_prob"] = float(new_probs[i])

    # ──────────────────────────────────────
    # Merge with existing results
    # ──────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("Merging with existing results")
    logger.info(f"{'='*60}")

    # Load existing CSV
    import pandas as pd
    csv_path = OUTPUT_DIR / "meta_analysis_properties.csv"
    existing_df = pd.read_csv(csv_path)
    logger.info(f"  Existing CSV: {len(existing_df)} rows, sources: {existing_df['source'].unique()}")

    # Create new dataframe
    new_df = pd.DataFrame(new_rows)

    # Remove any stale rows for these sources (should be 0, but be safe)
    new_source_names = [s[2] for s in missing_sources]
    existing_df = existing_df[~existing_df['source'].isin(new_source_names)]

    # Merge
    merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    merged_df.to_csv(csv_path, index=False)
    logger.info(f"  ✅ Merged CSV: {len(merged_df)} rows")
    logger.info(f"     Sources: {sorted(merged_df['source'].unique())}")

    # Load existing NPZ and merge
    npz_path = OUTPUT_DIR / "meta_analysis_features.npz"
    existing_npz = np.load(npz_path, allow_pickle=True)

    # Filter out stale entries
    existing_mask = ~np.isin(existing_npz['source_labels'], new_source_names)

    merged_features = np.concatenate([existing_npz['features'][existing_mask], new_features])
    merged_probs = np.concatenate([existing_npz['probs'][existing_mask], new_probs])
    merged_sources = np.concatenate([existing_npz['source_labels'][existing_mask],
                                     np.array([m[0] for m in new_meta])])
    merged_gt = np.concatenate([existing_npz['gt_labels'][existing_mask],
                                np.array([m[1] for m in new_meta])])
    merged_fnames = np.concatenate([existing_npz['filenames'][existing_mask],
                                    np.array([m[2] for m in new_meta])])

    np.savez_compressed(npz_path,
                        features=merged_features,
                        probs=merged_probs,
                        source_labels=merged_sources,
                        gt_labels=merged_gt,
                        filenames=merged_fnames)
    logger.info(f"  ✅ Merged NPZ: {merged_features.shape[0]} samples")

    # ──────────────────────────────────────
    # Print summary for new sources
    # ──────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("NEW SOURCE SUMMARY")
    logger.info(f"{'='*60}")

    for strategy, label, source_name in missing_sources:
        subset = new_df[new_df['source'] == source_name]
        if len(subset) == 0:
            logger.info(f"  {source_name}: NO DATA")
            continue
        probs = subset['model_fake_prob']
        is_fake = label == "fake"
        acc = (probs > 0.5).mean() * 100 if is_fake else (probs <= 0.5).mean() * 100
        sharp = subset['sharpness_laplacian_var'].median()
        edge = subset['edge_density'].median()
        noise = subset['noise_estimate'].median()
        w = subset['width'].median()
        h = subset['height'].median()

        logger.info(f"  {source_name:45s} | n={len(subset):4d} | acc={acc:5.1f}% | "
                    f"prob={probs.median():.3f} | sharp={sharp:7.1f} | edge={edge:.4f} | "
                    f"noise={noise:.3f} | res={w:.0f}×{h:.0f}")

    logger.info(f"\n{'='*60}")
    logger.info("Done! Now re-run plot_meta_analysis.py to update plots.")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
