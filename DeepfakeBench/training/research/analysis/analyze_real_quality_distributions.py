#!/usr/bin/env python3
"""
analyze_real_quality_distributions.py — Compare quality fingerprints across real-image sources.

Motivation: R5 OOD monitoring showed ~51–60% accuracy on Zoom VCD real images
(near coin-flip), while YouTube AVSpeech reals get ~86%. This means the model
confuses Zoom video-call quality characteristics with deepfake artifacts.

This script samples frames from three real-image sources and computes per-frame
quality metrics to identify *exactly* which quality axes diverge, so we can design
targeted augmentation to close the gap without adding VCD to training.

Sources:
  1. Zoom VCD reals         — gs://effort-collected-data/real/VCD (per-image, flat)
  2. YouTube AVSpeech reals — gs://effort-collected-data/real/external_youtube_avspeech/{video_id}/
  3. DF40 paired reals      — gs://df40-frames-recropped-rfa85/real/{source}/{identity}/
  4. Webcam test reals      — gs://effort-collected-data/real_or_virtual/real/{video_name}/

Outputs:
  - analysis_results/real_quality_fingerprints.csv    (all per-frame metrics)
  - analysis_results/plots/real_quality_distributions/ (comparison plots)

Usage:
    cd DeepfakeBench/training
    python analyze_real_quality_distributions.py [--samples-per-source 300] [--no-cache]
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from tqdm import tqdm

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
EXTERNAL_BUCKET = "effort-collected-data"
VCD_PREFIX = "real/VCD"
YOUTUBE_PREFIX = "real/external_youtube_avspeech"
REAL_OR_VIRTUAL_PREFIX = "real_or_virtual/real"
DF40_BUCKET = "df40-frames-recropped-rfa85"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent.parent / "analysis_results"
PLOT_DIR = OUTPUT_DIR / "plots" / "real_quality_distributions"
CACHE_DIR = SCRIPT_DIR / "weights" / "quality_analysis_samples"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("quality-analysis")


# ═══════════════════════════════════════════
# GCS ACCESS
# ═══════════════════════════════════════════

_client = None

def _gcs_client():
    global _client
    if _client is None:
        from google.cloud import storage
        _client = storage.Client()
    return _client


def _download_image(bucket, blob_name: str) -> Optional[np.ndarray]:
    """Download a single image from GCS and return as RGB numpy array."""
    try:
        blob = bucket.blob(blob_name)
        img_bytes = blob.download_as_bytes()
        img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        return img
    except Exception as e:
        logger.warning(f"  Error downloading {blob_name}: {e}")
        return None


# ═══════════════════════════════════════════
# IMAGE QUALITY METRICS
# ═══════════════════════════════════════════

def compute_quality_metrics(img_rgb: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive quality fingerprint from an RGB uint8 array.

    Metrics are chosen to capture axes where Zoom VCD might diverge from
    YouTube/DF40 reals: compression, sharpness, noise, resolution, color.
    """
    h, w = img_rgb.shape[:2]
    props: Dict[str, float] = {}

    # ── Dimensions ──
    props["width"] = float(w)
    props["height"] = float(h)
    props["num_pixels"] = float(w * h)

    # ── Color space conversions ──
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    # ═══ GROUP 1: SHARPNESS / BLUR ═══
    # Laplacian variance — the classic blur detector
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    props["sharpness_laplacian_var"] = float(laplacian.var())
    props["sharpness_laplacian_mean_abs"] = float(np.abs(laplacian).mean())

    # Tenengrad (Sobel gradient magnitude)
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    props["sharpness_tenengrad"] = float(grad_mag.mean())

    # Edge density (Canny)
    edges = cv2.Canny(img_gray, 100, 200)
    props["edge_density"] = float(edges.sum() / 255.0) / float(w * h)

    # ═══ GROUP 2: FREQUENCY SPECTRUM ═══
    # Resize to fixed 224×224 for comparable frequency analysis
    gray_224 = cv2.resize(img_gray, (224, 224))
    f_transform = np.fft.fft2(gray_224.astype(np.float64))
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    cy, cx = 112, 112
    Y, X = np.ogrid[:224, :224]
    radius = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    total_energy = magnitude.sum()
    if total_energy > 0:
        low_mask = radius < 20
        mid_mask = (radius >= 20) & (radius < 60)
        high_mask = radius >= 60
        props["freq_low_ratio"] = float(magnitude[low_mask].sum() / total_energy)
        props["freq_mid_ratio"] = float(magnitude[mid_mask].sum() / total_energy)
        props["freq_high_ratio"] = float(magnitude[high_mask].sum() / total_energy)
        props["freq_high_to_low"] = float(
            magnitude[high_mask].sum() / max(magnitude[low_mask].sum(), 1e-8)
        )
    else:
        props["freq_low_ratio"] = 0.0
        props["freq_mid_ratio"] = 0.0
        props["freq_high_ratio"] = 0.0
        props["freq_high_to_low"] = 0.0

    # Power spectral density slope (log-log regression of radial PSD)
    # Steeper negative slope = less high-frequency content = blurrier/more compressed
    max_r = int(np.sqrt(2) * 112)
    radial_profile = np.zeros(max_r)
    radial_count = np.zeros(max_r)
    r_int = radius.astype(int)
    for ry in range(224):
        for rx in range(224):
            r = r_int[ry, rx]
            if r < max_r:
                radial_profile[r] += magnitude[ry, rx]
                radial_count[r] += 1
    radial_count[radial_count == 0] = 1
    radial_profile /= radial_count

    # Log-log slope from freq 5 to 100 (skip DC and very high)
    valid = (radial_profile[5:100] > 0)
    if valid.sum() > 10:
        freqs = np.arange(5, 100)[valid]
        power = radial_profile[5:100][valid]
        log_f = np.log10(freqs)
        log_p = np.log10(power)
        slope, intercept, r_value, _, _ = stats.linregress(log_f, log_p)
        props["psd_slope"] = float(slope)
        props["psd_r_squared"] = float(r_value ** 2)
    else:
        props["psd_slope"] = 0.0
        props["psd_r_squared"] = 0.0

    # Effective resolution (freq index where 90% of energy is captured)
    sorted_mag = np.sort(magnitude.flatten())[::-1]
    cumsum = np.cumsum(sorted_mag)
    if cumsum[-1] > 0:
        idx_90 = np.searchsorted(cumsum, 0.90 * cumsum[-1])
        props["effective_resolution_90pct"] = float(idx_90) / float(len(sorted_mag))
    else:
        props["effective_resolution_90pct"] = 0.0

    # ═══ GROUP 3: COMPRESSION ARTIFACTS ═══
    # JPEG compressibility ratio
    _, jpg_buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    props["jpeg_compressibility"] = float(len(jpg_buf)) / float(w * h * 3)

    # DCT blockiness — 8×8 block boundary discontinuity (JPEG artifact signature)
    # Compute mean absolute difference at 8-pixel boundaries vs non-boundaries
    if h > 16 and w > 16:
        gray_f = img_gray.astype(np.float64)
        # Horizontal block boundaries (every 8th column)
        h_bound_cols = np.arange(8, w - 1, 8)
        h_non_bound_cols = np.array([c for c in range(1, w - 1) if c % 8 != 0])
        if len(h_bound_cols) > 0 and len(h_non_bound_cols) > 0:
            h_bound_diff = np.abs(gray_f[:, h_bound_cols] - gray_f[:, h_bound_cols - 1]).mean()
            # Sample same count of non-boundary diffs
            rng = np.random.RandomState(0)
            sample_cols = rng.choice(h_non_bound_cols, size=min(len(h_non_bound_cols), len(h_bound_cols) * 3), replace=False)
            h_non_diff = np.abs(gray_f[:, sample_cols] - gray_f[:, sample_cols - 1]).mean()
            props["blockiness_h_ratio"] = float(h_bound_diff / max(h_non_diff, 1e-8))
        else:
            props["blockiness_h_ratio"] = 1.0

        # Vertical block boundaries (every 8th row)
        v_bound_rows = np.arange(8, h - 1, 8)
        v_non_bound_rows = np.array([r for r in range(1, h - 1) if r % 8 != 0])
        if len(v_bound_rows) > 0 and len(v_non_bound_rows) > 0:
            v_bound_diff = np.abs(gray_f[v_bound_rows, :] - gray_f[v_bound_rows - 1, :]).mean()
            sample_rows = rng.choice(v_non_bound_rows, size=min(len(v_non_bound_rows), len(v_bound_rows) * 3), replace=False)
            v_non_diff = np.abs(gray_f[sample_rows, :] - gray_f[sample_rows - 1, :]).mean()
            props["blockiness_v_ratio"] = float(v_bound_diff / max(v_non_diff, 1e-8))
        else:
            props["blockiness_v_ratio"] = 1.0

        props["blockiness_mean"] = (props["blockiness_h_ratio"] + props["blockiness_v_ratio"]) / 2.0
    else:
        props["blockiness_h_ratio"] = 1.0
        props["blockiness_v_ratio"] = 1.0
        props["blockiness_mean"] = 1.0

    # ═══ GROUP 4: NOISE ═══
    # Immerkær noise estimate (3×3 Laplacian-based)
    noise_kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    sigma = np.sum(np.abs(cv2.filter2D(img_gray.astype(np.float64), -1, noise_kernel)))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))
    props["noise_estimate"] = float(sigma)

    # High-frequency noise (difference between original and Gaussian-blurred)
    blurred = cv2.GaussianBlur(img_gray.astype(np.float64), (5, 5), 1.0)
    hf_noise = img_gray.astype(np.float64) - blurred
    props["hf_noise_std"] = float(hf_noise.std())
    props["hf_noise_mean_abs"] = float(np.abs(hf_noise).mean())

    # ═══ GROUP 5: COLOR & LUMINANCE ═══
    for i, ch_name in enumerate(["r", "g", "b"]):
        ch = img_rgb[:, :, i].astype(np.float64)
        props[f"color_{ch_name}_mean"] = float(ch.mean())
        props[f"color_{ch_name}_std"] = float(ch.std())

    L = img_lab[:, :, 0].astype(np.float64)
    props["luminance_mean"] = float(L.mean())
    props["luminance_std"] = float(L.std())

    # Contrast
    gray_float = img_gray.astype(np.float64) / 255.0
    props["contrast_rms"] = float(gray_float.std())
    gmin, gmax = gray_float.min(), gray_float.max()
    props["contrast_michelson"] = float((gmax - gmin) / max(gmax + gmin, 1e-8))

    # Saturation
    saturation = img_hsv[:, :, 1].astype(np.float64)
    props["saturation_mean"] = float(saturation.mean())
    props["saturation_std"] = float(saturation.std())

    # Color histogram entropy (measure of color diversity)
    for i, ch_name in enumerate(["r", "g", "b"]):
        hist = cv2.calcHist([img_rgb], [i], None, [64], [0, 256]).flatten()
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        props[f"color_{ch_name}_entropy"] = float(-np.sum(hist * np.log2(hist)))

    # ═══ GROUP 6: TEXTURE ═══
    local_mean = cv2.blur(img_gray.astype(np.float64), (3, 3))
    local_var = cv2.blur((img_gray.astype(np.float64) - local_mean) ** 2, (3, 3))
    props["texture_local_var_mean"] = float(local_var.mean())
    props["texture_local_var_std"] = float(local_var.std())

    return props


# ═══════════════════════════════════════════
# DATA SAMPLING FROM GCS
# ═══════════════════════════════════════════

def sample_vcd_images(n: int, use_cache: bool = True) -> List[Tuple[str, np.ndarray]]:
    """Sample n face-crop images from Zoom VCD reals on GCS."""
    cache_dir = CACHE_DIR / "vcd_real"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if use_cache:
        cached = sorted([f for f in cache_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
        if len(cached) >= n:
            logger.info(f"  Using {n} cached VCD images")
            results = []
            for p in cached[:n]:
                img = np.array(Image.open(p).convert("RGB"))
                results.append((f"vcd_real/{p.stem}", img))
            return results

    logger.info(f"  Sampling {n} images from gs://{EXTERNAL_BUCKET}/{VCD_PREFIX}/")
    client = _gcs_client()
    bucket = client.bucket(EXTERNAL_BUCKET)

    blobs = list(bucket.list_blobs(prefix=VCD_PREFIX + "/"))
    image_blobs = [
        b for b in blobs
        if any(b.name.lower().endswith(ext) for ext in IMG_EXTS) and not b.name.endswith("/")
    ]
    logger.info(f"  Found {len(image_blobs)} VCD images total")

    rng = np.random.RandomState(42)
    indices = rng.choice(len(image_blobs), size=min(n, len(image_blobs)), replace=False)

    results = []
    for idx in tqdm(indices, desc="VCD"):
        fb = image_blobs[idx]
        img = _download_image(bucket, fb.name)
        if img is not None:
            local_name = fb.name.replace("/", "__")
            Image.fromarray(img).save(cache_dir / f"{local_name}.png")
            results.append((f"vcd_real/{fb.name.split('/')[-1]}", img))

    logger.info(f"  Got {len(results)} VCD images")
    return results


def sample_youtube_images(n: int, use_cache: bool = True) -> List[Tuple[str, np.ndarray]]:
    """Sample n face-crop images from YouTube AVSpeech reals (one per video)."""
    cache_dir = CACHE_DIR / "youtube_real"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if use_cache:
        cached = sorted([f for f in cache_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
        if len(cached) >= n:
            logger.info(f"  Using {n} cached YouTube images")
            results = []
            for p in cached[:n]:
                img = np.array(Image.open(p).convert("RGB"))
                results.append((f"youtube_real/{p.stem}", img))
            return results

    logger.info(f"  Sampling {n} images from gs://{EXTERNAL_BUCKET}/{YOUTUBE_PREFIX}/")
    client = _gcs_client()
    bucket = client.bucket(EXTERNAL_BUCKET)

    # Group by video folder (1 frame per video for independence)
    videos = defaultdict(list)
    for blob in bucket.list_blobs(prefix=YOUTUBE_PREFIX + "/"):
        parts = blob.name.split("/")
        if len(parts) >= 4 and any(blob.name.lower().endswith(ext) for ext in IMG_EXTS):
            video_id = parts[-2]
            videos[video_id].append(blob)

    logger.info(f"  Found {len(videos)} YouTube videos")

    rng = np.random.RandomState(43)
    video_ids = sorted(videos.keys())
    rng.shuffle(video_ids)

    results = []
    for vid_id in tqdm(video_ids[:n * 2], desc="YouTube"):
        if len(results) >= n:
            break
        frames = videos[vid_id]
        fb = frames[rng.randint(len(frames))]
        img = _download_image(bucket, fb.name)
        if img is not None:
            local_name = fb.name.replace("/", "__")
            Image.fromarray(img).save(cache_dir / f"{local_name}.png")
            results.append((f"youtube_real/{fb.name.split('/')[-1]}", img))

    logger.info(f"  Got {len(results)} YouTube images")
    return results


def sample_real_or_virtual_images(n: int, use_cache: bool = True) -> List[Tuple[str, np.ndarray]]:
    """Sample n face-crop images from real_or_virtual webcam test videos on GCS.

    Small set (~20 videos), so we sample generously across all videos.
    """
    cache_dir = CACHE_DIR / "real_or_virtual"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if use_cache:
        cached = sorted([f for f in cache_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
        if len(cached) >= n:
            logger.info(f"  Using {n} cached real_or_virtual images")
            results = []
            for p in cached[:n]:
                img = np.array(Image.open(p).convert("RGB"))
                results.append((f"real_or_virtual/{p.stem}", img))
            return results

    logger.info(f"  Sampling {n} images from gs://{EXTERNAL_BUCKET}/{REAL_OR_VIRTUAL_PREFIX}/")
    client = _gcs_client()
    bucket = client.bucket(EXTERNAL_BUCKET)

    # Group by video folder for balanced sampling
    videos = defaultdict(list)
    for blob in bucket.list_blobs(prefix=REAL_OR_VIRTUAL_PREFIX + "/"):
        if any(blob.name.lower().endswith(ext) for ext in IMG_EXTS) and not blob.name.endswith("/"):
            parts = blob.name.split("/")
            if len(parts) >= 4:  # real_or_virtual/real/{video_name}/frame.jpg
                video_name = parts[-2]
                videos[video_name].append(blob)

    logger.info(f"  Found {len(videos)} real_or_virtual videos, {sum(len(v) for v in videos.values())} total frames")

    rng = np.random.RandomState(45)
    # Sample roughly equally from each video
    per_video = max(1, n // max(len(videos), 1))
    remainder = n - per_video * len(videos)

    results = []
    for vid_name in sorted(videos.keys()):
        frames = videos[vid_name]
        k = min(per_video + (1 if remainder > 0 else 0), len(frames))
        if remainder > 0:
            remainder -= 1
        indices = rng.choice(len(frames), size=k, replace=False)
        for idx in indices:
            fb = frames[idx]
            img = _download_image(bucket, fb.name)
            if img is not None:
                local_name = fb.name.replace("/", "__")
                Image.fromarray(img).save(cache_dir / f"{local_name}.png")
                results.append((f"real_or_virtual/{fb.name.split('/')[-1]}", img))

    # If we still need more, random-fill from all frames
    if len(results) < n:
        all_frames = [b for frames in videos.values() for b in frames]
        extra_idx = rng.choice(len(all_frames), size=min(n - len(results), len(all_frames)), replace=False)
        for idx in extra_idx:
            fb = all_frames[idx]
            img = _download_image(bucket, fb.name)
            if img is not None:
                local_name = fb.name.replace("/", "__")
                Image.fromarray(img).save(cache_dir / f"{local_name}.png")
                results.append((f"real_or_virtual/{fb.name.split('/')[-1]}", img))

    logger.info(f"  Got {len(results)} real_or_virtual images")
    return results


def sample_df40_real_images(n: int, use_cache: bool = True) -> List[Tuple[str, np.ndarray]]:
    """Sample n face-crop images from DF40 paired reals on GCS."""
    cache_dir = CACHE_DIR / "df40_real"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if use_cache:
        cached = sorted([f for f in cache_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
        if len(cached) >= n:
            logger.info(f"  Using {n} cached DF40 real images")
            results = []
            for p in cached[:n]:
                img = np.array(Image.open(p).convert("RGB"))
                results.append((f"df40_real/{p.stem}", img))
            return results

    logger.info(f"  Sampling {n} images from gs://{DF40_BUCKET}/real/")
    client = _gcs_client()
    bucket = client.bucket(DF40_BUCKET)

    blobs = list(bucket.list_blobs(prefix="real/", max_results=30000))
    image_blobs = [
        b for b in blobs
        if any(b.name.lower().endswith(ext) for ext in IMG_EXTS) and not b.name.endswith("/")
    ]
    logger.info(f"  Found {len(image_blobs)} DF40 real images total")

    rng = np.random.RandomState(44)
    indices = rng.choice(len(image_blobs), size=min(n, len(image_blobs)), replace=False)

    results = []
    for idx in tqdm(indices, desc="DF40 real"):
        fb = image_blobs[idx]
        img = _download_image(bucket, fb.name)
        if img is not None:
            local_name = fb.name.replace("/", "__")
            Image.fromarray(img).save(cache_dir / f"{local_name}.png")
            results.append((f"df40_real/{fb.name.split('/')[-1]}", img))

    logger.info(f"  Got {len(results)} DF40 real images")
    return results


# ═══════════════════════════════════════════
# ANALYSIS & PLOTTING
# ═══════════════════════════════════════════

# Metrics grouped by the quality axis they measure (for subplot organization)
METRIC_GROUPS = {
    "Sharpness / Blur": [
        "sharpness_laplacian_var",
        "sharpness_laplacian_mean_abs",
        "sharpness_tenengrad",
        "edge_density",
    ],
    "Frequency Spectrum": [
        "freq_low_ratio",
        "freq_mid_ratio",
        "freq_high_ratio",
        "freq_high_to_low",
        "psd_slope",
        "effective_resolution_90pct",
    ],
    "Compression Artifacts": [
        "jpeg_compressibility",
        "blockiness_h_ratio",
        "blockiness_v_ratio",
        "blockiness_mean",
    ],
    "Noise": [
        "noise_estimate",
        "hf_noise_std",
        "hf_noise_mean_abs",
    ],
    "Color & Luminance": [
        "luminance_mean",
        "luminance_std",
        "contrast_rms",
        "contrast_michelson",
        "saturation_mean",
        "saturation_std",
        "color_r_entropy",
        "color_g_entropy",
        "color_b_entropy",
    ],
    "Texture": [
        "texture_local_var_mean",
        "texture_local_var_std",
    ],
}

SOURCE_COLORS = {
    "vcd_real": "#e74c3c",            # red — the problem source
    "youtube_real": "#2ecc71",        # green — known-good OOD real
    "df40_real": "#3498db",           # blue — training real
    "real_or_virtual": "#9b59b6",     # purple — webcam test reals
}

SOURCE_LABELS = {
    "vcd_real": "Zoom VCD (OOD — FAILING)",
    "youtube_real": "YouTube AVSpeech (OOD — OK)",
    "df40_real": "DF40 Paired Reals (training)",
    "real_or_virtual": "Webcam Tests (real_or_virtual)",
}


def plot_metric_histograms(df: pd.DataFrame):
    """Plot overlapping histograms for each metric group."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    sources = df["source"].unique()

    for group_name, metrics in METRIC_GROUPS.items():
        # Filter to metrics that exist in the dataframe
        metrics = [m for m in metrics if m in df.columns]
        if not metrics:
            continue

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            for source in sorted(sources):
                vals = df.loc[df["source"] == source, metric].dropna()
                if len(vals) == 0:
                    continue
                ax.hist(
                    vals,
                    bins=50,
                    alpha=0.45,
                    label=SOURCE_LABELS.get(source, source),
                    color=SOURCE_COLORS.get(source, "#888"),
                    density=True,
                )
            ax.set_title(metric.replace("_", " "), fontsize=10)
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)

        fig.suptitle(f"Quality Distributions — {group_name}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        safe_name = group_name.lower().replace(" ", "_").replace("/", "_")
        fig.savefig(PLOT_DIR / f"{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved plot: {safe_name}.png")


def plot_summary_divergence(df: pd.DataFrame):
    """Bar chart of standardized mean difference (VCD vs YouTube) per metric.

    This is the "action chart" — metrics with large divergence are the ones
    we need to cover with augmentation.
    """
    numeric_cols = [c for c in df.columns if c not in ("source", "image_id")]

    vcd = df[df["source"] == "vcd_real"]
    yt = df[df["source"] == "youtube_real"]

    divergences = {}
    for col in numeric_cols:
        v_vals = vcd[col].dropna()
        y_vals = yt[col].dropna()
        if len(v_vals) < 10 or len(y_vals) < 10:
            continue
        # Cohen's d (standardized mean difference)
        pooled_std = np.sqrt((v_vals.std() ** 2 + y_vals.std() ** 2) / 2)
        if pooled_std < 1e-10:
            continue
        d = (v_vals.mean() - y_vals.mean()) / pooled_std
        # Also run KS test
        ks_stat, ks_p = stats.ks_2samp(v_vals, y_vals)
        divergences[col] = {"cohens_d": d, "ks_stat": ks_stat, "ks_p": ks_p}

    if not divergences:
        logger.warning("  No valid divergence metrics computed")
        return

    # Sort by absolute Cohen's d
    sorted_metrics = sorted(divergences.keys(), key=lambda k: abs(divergences[k]["cohens_d"]), reverse=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_metrics) * 0.35)))
    y_pos = np.arange(len(sorted_metrics))
    d_vals = [divergences[m]["cohens_d"] for m in sorted_metrics]
    colors = ["#e74c3c" if abs(d) > 0.8 else "#f39c12" if abs(d) > 0.5 else "#2ecc71" for d in d_vals]

    ax.barh(y_pos, d_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.replace("_", " ") for m in sorted_metrics], fontsize=8)
    ax.set_xlabel("Cohen's d (VCD − YouTube)")
    ax.set_title("VCD vs YouTube: Quality Metric Divergence\n(Red = large effect, Orange = medium, Green = small)", fontsize=11)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axvline(-0.8, color="#e74c3c", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.axvline(0.8, color="#e74c3c", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.axvline(-0.5, color="#f39c12", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.axvline(0.5, color="#f39c12", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "divergence_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved divergence_summary.png")

    # Also save a text summary
    summary_path = OUTPUT_DIR / "vcd_quality_divergence_summary.txt"
    with open(summary_path, "w") as f:
        f.write("VCD vs YouTube AVSpeech — Quality Metric Divergence\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Metric':<35} {'Cohen d':>10} {'KS stat':>10} {'KS p':>12} {'Severity':>10}\n")
        f.write("-" * 77 + "\n")
        for m in sorted_metrics:
            d = divergences[m]
            severity = "LARGE" if abs(d["cohens_d"]) > 0.8 else "MEDIUM" if abs(d["cohens_d"]) > 0.5 else "small"
            f.write(f"{m:<35} {d['cohens_d']:>10.3f} {d['ks_stat']:>10.3f} {d['ks_p']:>12.2e} {severity:>10}\n")

        # Add interpretation
        f.write("\n\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 70 + "\n")
        f.write("Cohen's d > 0: VCD has HIGHER values than YouTube\n")
        f.write("Cohen's d < 0: VCD has LOWER values than YouTube\n")
        f.write("|d| > 0.8: Large effect — strong augmentation target\n")
        f.write("|d| > 0.5: Medium effect — moderate augmentation target\n")
        f.write("|d| < 0.5: Small effect — low priority\n")
        f.write("\n")
        f.write("KS p < 0.01: Distributions are statistically significantly different\n")

        # Actionable summary
        large = [m for m in sorted_metrics if abs(divergences[m]["cohens_d"]) > 0.8]
        medium = [m for m in sorted_metrics if 0.5 < abs(divergences[m]["cohens_d"]) <= 0.8]
        f.write("\n\n")
        f.write("AUGMENTATION TARGETS (priority order)\n")
        f.write("=" * 70 + "\n")
        if large:
            f.write("\nLARGE divergence (must address):\n")
            for m in large:
                d = divergences[m]["cohens_d"]
                direction = "higher" if d > 0 else "lower"
                f.write(f"  - {m}: VCD is {direction} (d={d:.2f})\n")
        if medium:
            f.write("\nMEDIUM divergence (should address):\n")
            for m in medium:
                d = divergences[m]["cohens_d"]
                direction = "higher" if d > 0 else "lower"
                f.write(f"  - {m}: VCD is {direction} (d={d:.2f})\n")

    logger.info(f"  Saved divergence summary: {summary_path}")


def plot_source_summary_table(df: pd.DataFrame):
    """Print a compact summary table comparing means across sources."""
    numeric_cols = [c for c in df.columns if c not in ("source", "image_id")]
    sources = sorted(df["source"].unique())

    summary_path = OUTPUT_DIR / "real_quality_summary_table.txt"
    with open(summary_path, "w") as f:
        f.write("Real Sources — Quality Metric Summary (mean ± std)\n")
        f.write("=" * 100 + "\n\n")

        header = f"{'Metric':<35}"
        for s in sources:
            label = SOURCE_LABELS.get(s, s)[:25]
            header += f" {label:>25}"
        f.write(header + "\n")
        f.write("-" * (35 + 26 * len(sources)) + "\n")

        for col in numeric_cols:
            row = f"{col:<35}"
            for s in sources:
                vals = df.loc[df["source"] == s, col].dropna()
                if len(vals) > 0:
                    row += f" {vals.mean():>11.3f} ±{vals.std():>10.3f}"
                else:
                    row += f" {'N/A':>25}"
            f.write(row + "\n")

    logger.info(f"  Saved summary table: {summary_path}")


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Analyze quality distributions across real-image sources")
    parser.add_argument("--samples-per-source", type=int, default=300,
                        help="Number of frames to sample from each source (default: 300)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-download even if cached samples exist")
    args = parser.parse_args()

    n = args.samples_per_source
    use_cache = not args.no_cache

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Sampling {n} images per source (cache={'on' if use_cache else 'off'})")

    # ── Step 1: Sample images ──
    t0 = time.time()
    all_images: Dict[str, List[Tuple[str, np.ndarray]]] = {}

    logger.info("=" * 60)
    logger.info("STEP 1: Sampling images from GCS")
    logger.info("=" * 60)

    all_images["vcd_real"] = sample_vcd_images(n, use_cache)
    all_images["youtube_real"] = sample_youtube_images(n, use_cache)
    all_images["df40_real"] = sample_df40_real_images(n, use_cache)
    all_images["real_or_virtual"] = sample_real_or_virtual_images(n, use_cache)

    for src, imgs in all_images.items():
        logger.info(f"  {src}: {len(imgs)} images")

    sample_time = time.time() - t0
    logger.info(f"  Sampling took {sample_time:.1f}s")

    # ── Step 2: Compute quality metrics ──
    logger.info("=" * 60)
    logger.info("STEP 2: Computing quality metrics")
    logger.info("=" * 60)

    rows = []
    for source, images in all_images.items():
        logger.info(f"  Processing {source} ({len(images)} images)...")
        for image_id, img_rgb in tqdm(images, desc=source):
            props = compute_quality_metrics(img_rgb)
            props["source"] = source
            props["image_id"] = image_id
            rows.append(props)

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "real_quality_fingerprints.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"  Saved {len(df)} rows to {csv_path}")

    compute_time = time.time() - t0 - sample_time
    logger.info(f"  Metric computation took {compute_time:.1f}s")

    # ── Step 3: Generate plots ──
    logger.info("=" * 60)
    logger.info("STEP 3: Generating comparison plots")
    logger.info("=" * 60)

    plot_metric_histograms(df)
    plot_summary_divergence(df)
    plot_source_summary_table(df)

    total_time = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"DONE — Total time: {total_time:.1f}s")
    logger.info(f"  CSV: {csv_path}")
    logger.info(f"  Plots: {PLOT_DIR}/")
    logger.info(f"  Divergence summary: {OUTPUT_DIR / 'vcd_quality_divergence_summary.txt'}")
    logger.info(f"  Summary table: {OUTPUT_DIR / 'real_quality_summary_table.txt'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
