#!/usr/bin/env python3
"""
meta_analysis_enhancer.py – Compare image properties across data sources.

Samples images from:
  1. WMA test images (local, all fake, enhanced face swaps)
  2. DeepLive quality_enhancement (GCS, fake — same software as WMA)
  3. DeepLive minimal_processing (GCS, fake — same software, no enhancer)
  4. DeepLive edge_cases (GCS, fake)
  5. VisoMaster (GCS, fake)
  6. DF40 (GCS, fake)
  7. Training reals from DeepLive (GCS, real)
  8. Training reals from DF40 (GCS, real)

For each image, computes:
  - Image properties: resolution, aspect ratio, sharpness, frequency spectrum,
    color statistics, JPEG quality estimate, contrast, saturation, noise level
  - CLIP backbone features: 512-dim pooler_output from the R25_F1 model
  - Model prediction: fake probability from R25_F1

Outputs:
  - analysis_results/meta_analysis_properties.csv (all image properties)
  - analysis_results/meta_analysis_features.npz (CLIP embeddings + metadata)

Usage:
    cd DeepfakeBench/training
    python meta_analysis_enhancer.py
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

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
WMA_DIR = Path("/Users/roeedar/Downloads/wma_export/all_images")
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "analysis_results"
CACHE_DIR = Path(__file__).resolve().parent / "weights" / "enhancer_eval"
GCS_SAMPLE_CACHE = Path(__file__).resolve().parent / "weights" / "meta_analysis_samples"

# How many images to sample from each GCS source
# Target: balanced real/fake. Fake sources: QE_fake(150) + MP_fake(150) + EC_fake(150)
#   + VM_fake(150) + DF40_fake(150) = 750 fake from GCS + 1202 WMA = 1952 fake total
# Real sources: QE_real(150) + DF40_real(150) + DeepLive_real(150) + ExtYT(200) = 650 real from GCS
# We'll also add more DeepLive real counterparts to help balance.
# Perfect balance isn't possible (WMA is 1202 and all fake) but we get enough
# real representation for meaningful property comparison.
SAMPLES_PER_SOURCE = 150
SAMPLES_EXT_YOUTUBE = 200  # More from this OOD real source — it's important

# Model checkpoint for feature extraction
MODEL_CHECKPOINT = {
    "name": "R25_F1",
    "gcs": "gs://training-job-outputs/phase2r2_experiments/5w453our/"
           "top_n_effort_20260212_step18500_auc0.9893_eer0.0356.pth",
}

# GCS bucket names
DEEPLIVE_BUCKET = "live-deepfake-methods-real-and-fake-frames-cropped"
DF40_BUCKET = "df40-frames-recropped-rfa85"
EXTERNAL_REAL_BUCKET = "effort-collected-data"
EXTERNAL_REAL_PREFIX = "real/external_youtube_avspeech"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("meta-analysis")


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
# IMAGE PROPERTY CALCULATIONS
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
    # Higher = sharper
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
    # Compute 2D FFT, look at energy distribution
    # Resize to fixed size for comparable frequency analysis
    gray_224 = cv2.resize(img_gray, (224, 224))
    f_transform = np.fft.fft2(gray_224.astype(np.float64))
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    # Avoid log(0)
    log_magnitude = np.log1p(magnitude)

    # Radial frequency analysis: low vs mid vs high frequency energy
    cy, cx = 112, 112  # center
    Y, X = np.ogrid[:224, :224]
    radius = np.sqrt((X - cx)**2 + (Y - cy)**2)

    total_energy = magnitude.sum()
    if total_energy > 0:
        # Low freq: radius < 20 (~inner 18%)
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
    # Michelson contrast
    gmin, gmax = gray_float.min(), gray_float.max()
    props["contrast_michelson"] = float((gmax - gmin) / max(gmax + gmin, 1e-8))

    # ── Saturation ──
    saturation = img_hsv[:, :, 1].astype(np.float64)
    props["saturation_mean"] = float(saturation.mean())
    props["saturation_std"] = float(saturation.std())

    # ── Hue statistics ──
    hue = img_hsv[:, :, 0].astype(np.float64)
    props["hue_mean"] = float(hue.mean())
    props["hue_std"] = float(hue.std())

    # ── Noise estimation (Immerkær method) ──
    # Uses a 3×3 Laplacian kernel for robust noise estimation
    noise_kernel = np.array([[1, -2, 1],
                              [-2, 4, -2],
                              [1, -2, 1]], dtype=np.float64)
    sigma = np.sum(np.abs(cv2.filter2D(img_gray.astype(np.float64), -1, noise_kernel)))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))
    props["noise_estimate"] = float(sigma)

    # ── JPEG quality estimate ──
    # Encode as JPEG at quality 100 and compare file size to raw size
    # Smaller ratio = more compressible = likely already JPEG-compressed
    _, jpg_buf = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, 95])
    props["jpeg_compressibility"] = float(len(jpg_buf)) / float(w * h * 3)

    # ── Edge density (Canny) ──
    edges = cv2.Canny(img_gray, 100, 200)
    props["edge_density"] = float(edges.sum() / 255.0) / float(w * h)

    # ── Local binary pattern - texture measure ──
    # Simplified: variance of local pixel differences
    kernel_size = 3
    local_mean = cv2.blur(img_gray.astype(np.float64), (kernel_size, kernel_size))
    local_var = cv2.blur((img_gray.astype(np.float64) - local_mean)**2,
                         (kernel_size, kernel_size))
    props["texture_local_var_mean"] = float(local_var.mean())

    # ── Symmetry (face images should be roughly symmetric) ──
    left_half = img_gray[:, :w//2]
    right_half = cv2.flip(img_gray[:, w//2:], 1)
    # Handle odd width
    min_w = min(left_half.shape[1], right_half.shape[1])
    left_half = left_half[:, :min_w]
    right_half = right_half[:, :min_w]
    diff = np.abs(left_half.astype(np.float64) - right_half.astype(np.float64))
    props["symmetry_score"] = 1.0 - float(diff.mean() / 255.0)

    # ── Effective resolution (power spectrum density rolloff) ──
    # The frequency at which 90% of total energy is contained
    # Lower = lower effective resolution (blurry / upscaled)
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
    """Lazy GCS client."""
    from google.cloud import storage
    return storage.Client()


def sample_deeplive_images(strategy: str, label: str, n: int) -> List[Tuple[str, np.ndarray]]:
    """Sample n face chip images from a DeepLive strategy on GCS.
    
    Returns list of (identifier_string, rgb_array) tuples.
    """
    client = _gcs_client()
    bucket = client.bucket(DEEPLIVE_BUCKET)
    cache_dir = GCS_SAMPLE_CACHE / f"deeplive_{strategy}_{label}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check cache first
    cached = sorted([f for f in cache_dir.iterdir() if f.suffix in IMG_EXTS])
    if len(cached) >= n:
        logger.info(f"  Using {n} cached images for deeplive/{strategy}/{label}")
        results = []
        for p in cached[:n]:
            img = np.array(Image.open(p).convert("RGB"))
            results.append((f"deeplive_{strategy}_{label}/{p.name}", img))
        return results

    logger.info(f"  Sampling {n} images from gs://{DEEPLIVE_BUCKET} strategy={strategy}, label={label}")
    
    # Use strategy-specific prefix to avoid scanning the entire bucket
    # DeepLive samples are like: samples/{strategy}_{id}/frames/{label}/frame_XXXX.png
    import json
    prefix = f"samples/{strategy}_"
    
    # Only list manifest files under the strategy-specific prefix
    matching_samples = []
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith("manifest.json"):
            sample_dir = blob.name.rsplit("/", 1)[0]
            matching_samples.append(sample_dir)
            if len(matching_samples) >= n * 3:  # Get more than we need for diversity
                break
    
    logger.info(f"  Found {len(matching_samples)} sample directories for strategy={strategy}")
    
    if not matching_samples:
        logger.warning(f"  No samples found for strategy={strategy}")
        return []
    
    # Randomly select samples and pick one frame from each
    rng = np.random.RandomState(42)
    rng.shuffle(matching_samples)
    
    results = []
    for sample_dir in matching_samples:
        if len(results) >= n:
            break
        
        # List frames for this sample
        frame_prefix = f"{sample_dir}/frames/{label}/"
        frame_blobs = list(bucket.list_blobs(prefix=frame_prefix))
        frame_blobs = [b for b in frame_blobs if not b.name.endswith("/")]
        
        if not frame_blobs:
            continue
        
        # Pick a random frame
        fb = frame_blobs[rng.randint(len(frame_blobs))]
        
        try:
            img_bytes = fb.download_as_bytes()
            img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
            
            # Cache locally
            local_name = fb.name.replace("/", "_")
            Image.fromarray(img).save(cache_dir / f"{local_name}.png")
            
            identifier = f"deeplive_{strategy}_{label}/{fb.name.split('/')[-1]}"
            results.append((identifier, img))
        except Exception as e:
            logger.warning(f"  Error loading {fb.name}: {e}")
            continue
    
    logger.info(f"  Got {len(results)} images for deeplive/{strategy}/{label}")
    return results


def sample_visomaster_images(label: str, n: int) -> List[Tuple[str, np.ndarray]]:
    """Sample n face chips from VisoMaster data on GCS."""
    client = _gcs_client()
    bucket = client.bucket(DEEPLIVE_BUCKET)  # VisoMaster is in same bucket
    cache_dir = GCS_SAMPLE_CACHE / f"visomaster_{label}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    cached = sorted([f for f in cache_dir.iterdir() if f.suffix in IMG_EXTS])
    if len(cached) >= n:
        logger.info(f"  Using {n} cached images for visomaster/{label}")
        results = []
        for p in cached[:n]:
            img = np.array(Image.open(p).convert("RGB"))
            results.append((f"visomaster_{label}/{p.name}", img))
        return results

    logger.info(f"  Sampling {n} images from visomaster {label}")
    
    # VisoMaster samples are prefixed with visomaster_
    import json
    prefix = "samples/visomaster_"
    blobs = list(bucket.list_blobs(prefix=prefix, max_results=5000))
    manifest_blobs = [b for b in blobs if b.name.endswith("manifest.json")]
    
    matching_samples = []
    for mb in manifest_blobs:
        sample_dir = mb.name.rsplit("/", 1)[0]
        matching_samples.append(sample_dir)
        if len(matching_samples) >= n * 2:
            break
    
    rng = np.random.RandomState(43)
    rng.shuffle(matching_samples)
    
    results = []
    for sample_dir in matching_samples:
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
            
            results.append((f"visomaster_{label}/{fb.name.split('/')[-1]}", img))
        except Exception as e:
            logger.warning(f"  Error loading {fb.name}: {e}")
    
    logger.info(f"  Got {len(results)} images for visomaster/{label}")
    return results


def sample_df40_images(label: str, n: int) -> List[Tuple[str, np.ndarray]]:
    """Sample n face chips from DF40 on GCS."""
    client = _gcs_client()
    bucket = client.bucket(DF40_BUCKET)
    cache_dir = GCS_SAMPLE_CACHE / f"df40_{label}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    cached = sorted([f for f in cache_dir.iterdir() if f.suffix in IMG_EXTS])
    if len(cached) >= n:
        logger.info(f"  Using {n} cached images for df40/{label}")
        results = []
        for p in cached[:n]:
            img = np.array(Image.open(p).convert("RGB"))
            results.append((f"df40_{label}/{p.name}", img))
        return results

    logger.info(f"  Sampling {n} images from gs://{DF40_BUCKET}/{label}/")
    
    # DF40 structure: {label}/{method_or_source}/{identity}/frame_XXXX.png
    prefix = f"{label}/"
    blobs = list(bucket.list_blobs(prefix=prefix, max_results=20000))
    image_blobs = [b for b in blobs if b.name.lower().endswith(('.png', '.jpg', '.jpeg'))
                   and not b.name.endswith("/")]
    
    if not image_blobs:
        logger.warning(f"  No images found for df40/{label}")
        return []
    
    rng = np.random.RandomState(44)
    indices = rng.choice(len(image_blobs), size=min(n, len(image_blobs)), replace=False)
    
    results = []
    for idx in indices:
        fb = image_blobs[idx]
        try:
            img_bytes = fb.download_as_bytes()
            img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
            
            local_name = fb.name.replace("/", "_")
            Image.fromarray(img).save(cache_dir / f"{local_name}.png")
            
            results.append((f"df40_{label}/{fb.name.split('/')[-1]}", img))
        except Exception as e:
            logger.warning(f"  Error loading {fb.name}: {e}")
    
    logger.info(f"  Got {len(results)} images for df40/{label}")
    return results


def sample_external_youtube_images(n: int) -> List[Tuple[str, np.ndarray]]:
    """Sample n face chip images from external YouTube AVSpeech (real) on GCS.
    
    Structure: gs://effort-collected-data/real/external_youtube_avspeech/{video_id}/{frame}.png
    """
    client = _gcs_client()
    bucket = client.bucket(EXTERNAL_REAL_BUCKET)
    cache_dir = GCS_SAMPLE_CACHE / "external_youtube_real"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    cached = sorted([f for f in cache_dir.iterdir() if f.suffix in IMG_EXTS])
    if len(cached) >= n:
        logger.info(f"  Using {n} cached images for external_youtube_real")
        results = []
        for p in cached[:n]:
            img = np.array(Image.open(p).convert("RGB"))
            results.append((f"external_youtube_real/{p.name}", img))
        return results

    logger.info(f"  Sampling {n} images from gs://{EXTERNAL_REAL_BUCKET}/{EXTERNAL_REAL_PREFIX}/")

    # Group blobs by video folder
    from collections import defaultdict as dd
    videos = dd(list)
    for blob in bucket.list_blobs(prefix=EXTERNAL_REAL_PREFIX + "/"):
        parts = blob.name.split("/")
        if len(parts) >= 4 and blob.name.lower().endswith((".png", ".jpg", ".jpeg")):
            video_id = parts[-2]
            videos[video_id].append(blob)

    logger.info(f"  Found {len(videos)} videos")

    if not videos:
        logger.warning("  No external YouTube videos found")
        return []

    # Sample one frame per video, from random videos
    rng = np.random.RandomState(45)
    video_ids = sorted(videos.keys())
    rng.shuffle(video_ids)

    results = []
    for vid_id in video_ids:
        if len(results) >= n:
            break
        frames = videos[vid_id]
        fb = frames[rng.randint(len(frames))]

        try:
            img_bytes = fb.download_as_bytes()
            img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

            local_name = fb.name.replace("/", "_")
            Image.fromarray(img).save(cache_dir / f"{local_name}.png")

            results.append((f"external_youtube_real/{fb.name.split('/')[-1]}", img))
        except Exception as e:
            logger.warning(f"  Error loading {fb.name}: {e}")

    logger.info(f"  Got {len(results)} images for external_youtube_real")
    return results


def load_wma_images(max_n: Optional[int] = None) -> List[Tuple[str, np.ndarray]]:
    """Load all (or up to max_n) WMA test images."""
    files = sorted([f for f in WMA_DIR.iterdir() if f.suffix.lower() in IMG_EXTS])
    if max_n:
        files = files[:max_n]
    
    results = []
    for f in files:
        try:
            img = np.array(Image.open(f).convert("RGB"))
            results.append((f"wma_enhanced/{f.name}", img))
        except Exception as e:
            logger.warning(f"  Error loading {f.name}: {e}")
    
    logger.info(f"  Loaded {len(results)} WMA test images")
    return results


# ═══════════════════════════════════════════
# MODEL LOADING & FEATURE EXTRACTION
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
    """Resize to 224×224, normalize with CLIP stats."""
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    return _transform(Image.fromarray(img_resized))


@torch.inference_mode()
def extract_features_and_probs(model: torch.nn.Module,
                                images: List[np.ndarray],
                                batch_size: int = 32
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """Extract CLIP features and fake probabilities for a list of images.
    
    Returns:
        features: (N, hidden_size) float32 array
        probs: (N,) float32 array of fake probabilities
    """
    all_features = []
    all_probs = []
    n_batches = (len(images) + batch_size - 1) // batch_size

    pbar = tqdm(total=len(images), desc="Feature extraction", unit="img",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

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
    return np.concatenate(all_features, axis=0), np.concatenate(all_probs, axis=0)


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GCS_SAMPLE_CACHE.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────
    # Step 1: Collect images from all sources
    # ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Collecting images from all sources")
    logger.info("=" * 60)

    # Each entry: (source_name, ground_truth_label, list of (id, rgb_array))
    sources: List[Tuple[str, str, List[Tuple[str, np.ndarray]]]] = []

    # ── FAKE sources ──

    # 1. WMA test images (all fake, enhanced) — the images we're investigating
    logger.info("\n[1/10] WMA enhanced test images (local)")
    wma_imgs = load_wma_images()
    sources.append(("wma_enhanced", "fake", wma_imgs))

    # 2. DeepLive quality_enhancement (fake) — same software as WMA
    logger.info("\n[2/10] DeepLive quality_enhancement (GCS, fake)")
    qe_fake = sample_deeplive_images("quality_enhancement", "fake", SAMPLES_PER_SOURCE)
    sources.append(("deeplive_quality_enhancement_fake", "fake", qe_fake))

    # 3. DeepLive minimal_processing (fake) — same tool, no enhancer
    logger.info("\n[3/10] DeepLive minimal_processing (GCS, fake)")
    mp_fake = sample_deeplive_images("minimal_processing", "fake", SAMPLES_PER_SOURCE)
    sources.append(("deeplive_minimal_processing_fake", "fake", mp_fake))

    # 4. DeepLive edge_cases (fake)
    logger.info("\n[4/10] DeepLive edge_cases (GCS, fake)")
    ec_fake = sample_deeplive_images("edge_cases", "fake", SAMPLES_PER_SOURCE)
    sources.append(("deeplive_edge_cases_fake", "fake", ec_fake))

    # 5. VisoMaster (fake)
    logger.info("\n[5/10] VisoMaster (GCS, fake)")
    vm_fake = sample_visomaster_images("fake", SAMPLES_PER_SOURCE)
    sources.append(("visomaster_fake", "fake", vm_fake))

    # 6. DF40 (fake)
    logger.info("\n[6/10] DF40 (GCS, fake)")
    df40_fake = sample_df40_images("fake", SAMPLES_PER_SOURCE)
    sources.append(("df40_fake", "fake", df40_fake))

    # ── REAL sources ──

    # 7. DeepLive quality_enhancement (real) — real counterparts of QE fakes
    logger.info("\n[7/10] DeepLive quality_enhancement (GCS, real)")
    qe_real = sample_deeplive_images("quality_enhancement", "real", SAMPLES_PER_SOURCE)
    sources.append(("deeplive_quality_enhancement_real", "real", qe_real))

    # 8. DeepLive minimal_processing (real) — real counterparts
    logger.info("\n[8/10] DeepLive minimal_processing (GCS, real)")
    mp_real = sample_deeplive_images("minimal_processing", "real", SAMPLES_PER_SOURCE)
    sources.append(("deeplive_minimal_processing_real", "real", mp_real))

    # 9. DF40 (real)
    logger.info("\n[9/10] DF40 (GCS, real)")
    df40_real = sample_df40_images("real", SAMPLES_PER_SOURCE)
    sources.append(("df40_real", "real", df40_real))

    # 10. External YouTube AVSpeech (real) — OOD real data used in R3 validation
    logger.info("\n[10/10] External YouTube AVSpeech (GCS, real)")
    ext_yt = sample_external_youtube_images(SAMPLES_EXT_YOUTUBE)
    sources.append(("external_youtube_real", "real", ext_yt))

    # Summary
    total = sum(len(imgs) for _, _, imgs in sources)
    logger.info(f"\nTotal images collected: {total}")
    for name, label, imgs in sources:
        logger.info(f"  {name} ({label}): {len(imgs)}")

    # ──────────────────────────────────────
    # Step 2: Compute image properties (resumable)
    # ──────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Computing image properties")
    logger.info("=" * 60)

    props_checkpoint = OUTPUT_DIR / "_meta_analysis_props_checkpoint.csv"
    all_rows = []  # Each row: {source, label, filename, ...properties}
    all_images_flat = []  # For feature extraction later
    all_meta_flat = []  # (source, label, filename) for each image

    # Check for partial progress
    completed_files = set()
    if props_checkpoint.exists():
        import csv as csv_mod
        with open(props_checkpoint, "r") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                all_rows.append(row)
                completed_files.add(row["filename"])
                all_meta_flat.append((row["source"], row["label"], row["filename"]))
        logger.info(f"  Resuming: {len(completed_files)} images already processed")

    # Flatten all images, noting which are already done
    all_source_images = []
    for source_name, label, images in sources:
        for identifier, img_rgb in images:
            all_source_images.append((source_name, label, identifier, img_rgb))

    # Rebuild all_images_flat in order (needed for feature extraction)
    # We'll re-extract features for everything — images are small, features aren't cached
    pending = [(s, l, ident, img) for s, l, ident, img in all_source_images
               if ident not in completed_files]

    if pending:
        logger.info(f"  Computing properties for {len(pending)} images "
                    f"({len(completed_files)} already cached)")
        writer_handle = None
        csv_file = None
        try:
            is_new = not props_checkpoint.exists() or len(completed_files) == 0
            csv_file = open(props_checkpoint, "a" if not is_new else "w", newline="")

            for source_name, label, identifier, img_rgb in tqdm(
                pending, desc="Image properties", unit="img",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ):
                try:
                    props = compute_image_properties(img_rgb)
                    row = {
                        "source": source_name,
                        "label": label,
                        "filename": identifier,
                        **props,
                    }
                    # Convert numeric values back to proper types
                    for k, v in row.items():
                        if isinstance(v, (np.floating, np.integer)):
                            row[k] = float(v)

                    all_rows.append(row)
                    all_meta_flat.append((source_name, label, identifier))
                    completed_files.add(identifier)

                    # Write to checkpoint file incrementally
                    if writer_handle is None:
                        fieldnames = list(row.keys())
                        writer_handle = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        if is_new:
                            writer_handle.writeheader()
                            is_new = False
                    writer_handle.writerow(row)
                    csv_file.flush()  # flush each row for crash safety

                except Exception as e:
                    logger.warning(f"  Error processing {identifier}: {e}")
        finally:
            if csv_file:
                csv_file.close()
    else:
        logger.info("  All image properties already cached — skipping Step 2")

    # Rebuild all_images_flat in source order (for feature extraction)
    completed_lookup = {ident: i for i, (_, _, ident) in enumerate(all_meta_flat)}
    for source_name, label, identifier, img_rgb in all_source_images:
        if identifier in completed_lookup:
            all_images_flat.append(img_rgb)

    # ──────────────────────────────────────
    # Step 3: Load model and extract features + predictions
    # ──────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Extracting CLIP features + model predictions")
    logger.info("=" * 60)

    ckpt_path = download_checkpoint(MODEL_CHECKPOINT["gcs"])
    model = load_model(ckpt_path)
    logger.info(f"  Model loaded: {MODEL_CHECKPOINT['name']}")

    features, probs = extract_features_and_probs(model, all_images_flat)
    logger.info(f"  Features shape: {features.shape}, Probs shape: {probs.shape}")

    # Add prob to rows
    for i, row in enumerate(all_rows):
        row["model_fake_prob"] = float(probs[i])

    # Free model memory
    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    # ──────────────────────────────────────
    # Step 4: Write outputs
    # ──────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Writing outputs")
    logger.info("=" * 60)

    # 4a: Properties CSV
    csv_path = OUTPUT_DIR / "meta_analysis_properties.csv"
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        logger.info(f"  ✅ Properties CSV: {csv_path} ({len(all_rows)} rows)")

    # 4b: Features NPZ (for t-SNE / UMAP / cosine similarity analysis)
    npz_path = OUTPUT_DIR / "meta_analysis_features.npz"
    source_labels = np.array([m[0] for m in all_meta_flat])
    gt_labels = np.array([m[1] for m in all_meta_flat])
    filenames = np.array([m[2] for m in all_meta_flat])
    np.savez_compressed(
        npz_path,
        features=features,
        probs=probs,
        source_labels=source_labels,
        gt_labels=gt_labels,
        filenames=filenames,
    )
    logger.info(f"  ✅ Features NPZ: {npz_path}")

    # ──────────────────────────────────────
    # Step 5: Print summary statistics
    # ──────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Summary statistics")
    logger.info("=" * 60)

    # Key properties to compare
    key_props = [
        "sharpness_laplacian_var", "sharpness_tenengrad",
        "freq_high_energy_ratio", "freq_high_to_low_ratio",
        "noise_estimate", "contrast_rms", "saturation_mean",
        "jpeg_compressibility", "edge_density",
        "width", "height",
        "model_fake_prob",
    ]

    # Group by source
    source_data = defaultdict(list)
    for row in all_rows:
        source_data[row["source"]].append(row)

    # Print comparison table for each key property
    source_order = [s[0] for s in sources]
    
    print(f"\n{'='*120}")
    print("META-ANALYSIS: Image Properties by Data Source")
    print(f"{'='*120}")
    
    for prop in key_props:
        print(f"\n── {prop} ──")
        print(f"  {'Source':<45} {'Mean':>10} {'Median':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print(f"  {'-'*95}")
        for src in source_order:
            rows = source_data.get(src, [])
            if not rows:
                continue
            vals = np.array([r[prop] for r in rows if not np.isnan(r.get(prop, float("nan")))])
            if len(vals) == 0:
                continue
            print(f"  {src:<45} {np.mean(vals):>10.4f} {np.median(vals):>10.4f} "
                  f"{np.std(vals):>10.4f} {np.min(vals):>10.4f} {np.max(vals):>10.4f}")

    # Model prediction summary
    print(f"\n{'='*120}")
    print("MODEL PREDICTIONS (R25_F1) — Fake prob by source")
    print(f"{'='*120}")
    print(f"  {'Source':<45} {'Label':<6} {'N':>5} {'Acc@0.5':>8} {'MeanProb':>10} {'MedianProb':>12}")
    print(f"  {'-'*95}")
    for src_name, label, imgs in sources:
        rows = source_data.get(src_name, [])
        if not rows:
            continue
        ps = np.array([r["model_fake_prob"] for r in rows])
        if label == "fake":
            acc = np.mean(ps > 0.5)
        else:
            acc = np.mean(ps <= 0.5)
        print(f"  {src_name:<45} {label:<6} {len(ps):>5} {100*acc:>7.1f}% "
              f"{np.mean(ps):>10.4f} {np.median(ps):>12.4f}")

    print(f"\n{'='*120}")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
