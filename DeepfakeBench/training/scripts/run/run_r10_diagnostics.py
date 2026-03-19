#!/usr/bin/env python3
"""
R10 Pre-Training Diagnostic Experiments
========================================

Three containerized experiments that run against the R9_A checkpoint
to inform Round 10 training decisions.

**Experiment 0A — JPEG Sensitivity Sweep**
    Take R9_A. Run inference on PNG validation data at native quality,
    then on the same images JPEG-compressed at Q50, Q70, Q85, Q95.
    Compare AUC and per-method accuracy.

**Experiment 0B — Lighting Sensitivity Profile**
    Take 30 real + 30 fake crops from validation. Apply systematic
    brightness multipliers and gamma corrections. Record per-frame
    fake_prob under each perturbation.

**Experiment 0C — Teams v2 Sanity Check**
    Run R9_A on a random 200-sample subset of Teams v2 (100 real,
    100 fake). Compare score distributions against baseline expectations.

All results are saved as CSVs to GCS and logged to W&B.

Usage (inside container):
    python run_r10_diagnostics.py \\
        --checkpoint_gcs_path gs://training-job-outputs/phase2r9_experiments/1551zxa8/top_n_effort_20260228_step6000_auc0.9891_eer0.0457.pth \\
        --experiments 0A,0B,0C \\
        --output_gcs_folder gs://training-job-outputs/r10_diagnostics/$(date +%Y-%m-%d_%H-%M-%S)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import random
import tempfile
import time
from collections import Counter, OrderedDict, defaultdict
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from PIL import Image

try:
    import wandb
except ImportError:
    wandb = None

from logger import create_logger
from prepare_splits import VideoInfo
from utils import choose_metric, init_seed
from utils.gcs import download_assets_from_gcs


# ──────────────────────────────────────────────────────────────────────────────
# Model Loading (same pattern as validate_custom_sources.py)
# ──────────────────────────────────────────────────────────────────────────────


def _load_config(detector_path: str, train_config_path: str, dataloader_config: str) -> dict:
    """Load and merge all config files."""
    with open(detector_path) as f:
        config = yaml.safe_load(f)
    with open(train_config_path) as f:
        config.update(yaml.safe_load(f))
    with open(dataloader_config) as f:
        config.update(yaml.safe_load(f))

    config.setdefault("ddp", False)
    config.setdefault("local_rank", 0)
    config.setdefault("metric_scoring", "auc")
    config.setdefault("test_batchSize", 32)
    config.setdefault("manualSeed", 737)
    config.setdefault("cuda", True)
    config.setdefault("cudnn", True)

    dl_params = config.get("dataloader_params", {})
    dl_params["frames_per_video"] = 8
    config["dataloader_params"] = dl_params
    return config


def _load_model(config: dict, checkpoint_path: str, logger: logging.Logger):
    """Load checkpoint, update config, create model, load weights."""
    from detectors import DETECTOR

    saved_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_config = {}

    if isinstance(saved_checkpoint, dict) and "model_config" in saved_checkpoint:
        saved_config = saved_checkpoint["model_config"]
        logger.info("Restoring model config from checkpoint:")
        for key, value in saved_config.items():
            if key == "current_arcface_s":
                continue
            old_value = config.get(key)
            config[key] = value
            if old_value != value:
                logger.info("  %s: %s -> %s", key, old_value, value)
    else:
        logger.warning("Checkpoint missing model_config; using current config.")

    # Merge gcs_assets from checkpoint for backbone downloads
    saved_assets = (saved_config or {}).get("gcs_assets") or {}
    clip_backbone = saved_assets.get("clip_backbone")
    if clip_backbone:
        if "gcs_assets" not in config:
            config["gcs_assets"] = {}
        existing = config["gcs_assets"].get("clip_backbone")
        if not existing or (not existing.get("gcs_path") and not existing.get("local_path")):
            config["gcs_assets"]["clip_backbone"] = clip_backbone
            logger.info("Added clip_backbone asset from checkpoint config.")

    download_assets_from_gcs(config, logger)

    logger.info("Creating model: %s", config.get("model_name"))
    model = DETECTOR[config["model_name"]](config)

    # Load state dict
    if isinstance(saved_checkpoint, dict) and "state_dict" in saved_checkpoint:
        state_dict = saved_checkpoint["state_dict"]
    else:
        state_dict = saved_checkpoint

    if saved_config.get("use_arcface_head") and "current_arcface_s" in saved_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(saved_config["current_arcface_s"])
            logger.info("Restored ArcFace s = %s", saved_config["current_arcface_s"])

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    logger.info("Model weights loaded.")

    device = torch.device("cuda" if config.get("cuda") and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, device, saved_config


# ──────────────────────────────────────────────────────────────────────────────
# Shared inference utilities
# ──────────────────────────────────────────────────────────────────────────────


def _get_clip_transform(config: dict):
    """Return the standard CLIP normalize transform used during validation."""
    from torchvision import transforms as T

    return T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=config.get("mean", [0.48145466, 0.4578275, 0.40821073]),
            std=config.get("std", [0.26862954, 0.26130258, 0.27577711]),
        ),
    ])


@torch.no_grad()
def _infer_single_image(
    model: torch.nn.Module,
    pil_image: Image.Image,
    transform,
    device: torch.device,
    frames_per_video: int = 1,
) -> float:
    """Run inference on a single PIL image, return fake probability."""
    tensor = transform(pil_image)  # [3, 224, 224]
    # Model expects [B, T, C, H, W]
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 224, 224]
    if frames_per_video > 1:
        tensor = tensor.expand(-1, frames_per_video, -1, -1, -1).clone()
    tensor = tensor.to(device)
    data_dict = {
        "image": tensor,
        "label": torch.zeros(1, dtype=torch.long, device=device),
    }
    out = model(data_dict, inference=True)
    prob = out["prob"]
    if prob.dim() > 0:
        prob = prob.mean()
    return prob.item()


@torch.no_grad()
def _infer_batch_images(
    model: torch.nn.Module,
    pil_images: List[Image.Image],
    transform,
    device: torch.device,
) -> List[float]:
    """Run batched inference on a list of PIL images. Returns list of fake_probs."""
    tensors = [transform(img) for img in pil_images]
    batch = torch.stack(tensors, dim=0).unsqueeze(1)  # [B, 1, 3, 224, 224]
    batch = batch.to(device)
    data_dict = {
        "image": batch,
        "label": torch.zeros(len(pil_images), dtype=torch.long, device=device),
    }
    out = model(data_dict, inference=True)
    probs = out["prob"]
    if probs.dim() == 0:
        return [probs.item()]
    return probs.cpu().tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Image transform utilities
# ──────────────────────────────────────────────────────────────────────────────


def jpeg_compress(pil_image: Image.Image, quality: int) -> Image.Image:
    """JPEG compress-decompress a PIL image at the given quality level."""
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_brightness(pil_image: Image.Image, factor: float) -> Image.Image:
    """Multiply pixel values by factor (clipped to 0-255)."""
    arr = np.array(pil_image, dtype=np.float32)
    arr = np.clip(arr * factor, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_gamma(pil_image: Image.Image, gamma: float) -> Image.Image:
    """Apply gamma correction: output = (input/255)^gamma * 255."""
    arr = np.array(pil_image, dtype=np.float32)
    arr = np.clip(((arr / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ──────────────────────────────────────────────────────────────────────────────
# Sample loading from GCS
# ──────────────────────────────────────────────────────────────────────────────


def _load_pil_from_gcs(gcs_path: str, resolution: int = 224) -> Optional[Image.Image]:
    """Load a single image from GCS and resize to resolution."""
    import fsspec

    try:
        with fsspec.open(gcs_path, "rb") as f:
            img = Image.open(f).convert("RGB")
            img = img.resize((resolution, resolution), Image.BICUBIC)
        return img
    except Exception as e:
        logging.getLogger(__name__).debug("Failed to load %s: %s", gcs_path, e)
        return None


def _stratified_sample(
    videos: List[VideoInfo],
    per_method_per_label: int,
    seed: int = 737,
    logger: Optional[logging.Logger] = None,
) -> List[VideoInfo]:
    """
    Sample uniformly across methods: up to `per_method_per_label` videos per
    (method, label) pair.  This ensures every method has enough representation
    for per-method statistics (target: ≥30 for stable CI).
    """
    log = logger or logging.getLogger(__name__)
    rng = random.Random(seed)
    buckets: Dict[Tuple[str, str], List[VideoInfo]] = defaultdict(list)
    for v in videos:
        buckets[(v.method, v.label)].append(v)

    sampled: List[VideoInfo] = []
    for key in sorted(buckets.keys()):
        pool = buckets[key]
        rng.shuffle(pool)
        take = min(per_method_per_label, len(pool))
        sampled.extend(pool[:take])
        log.info("  Stratified sample: %-40s  %d / %d available", key, take, len(pool))

    rng.shuffle(sampled)
    real_count = sum(1 for v in sampled if v.label == "real")
    fake_count = sum(1 for v in sampled if v.label == "fake")
    n_methods = len(set(v.method for v in sampled))
    log.info(
        "Stratified total: %d videos (real=%d fake=%d) across %d methods, "
        "target %d per method per label",
        len(sampled), real_count, fake_count, n_methods, per_method_per_label,
    )
    return sampled


def _load_samples_for_diagnostic(
    videos: List[VideoInfo],
    per_method_per_label: int,
    resolution: int,
    seed: int = 737,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[dict], List[dict]]:
    """
    Load a balanced, stratified set of real + fake samples from VideoInfo objects.
    Samples `per_method_per_label` videos per (method, label) pair for uniform coverage.

    Returns (real_samples, fake_samples) where each sample is:
        {"video_id": str, "method": str, "label": str, "image": PIL.Image, "frame_path": str}
    """
    log = logger or logging.getLogger(__name__)
    sampled = _stratified_sample(videos, per_method_per_label, seed, log)
    reals = [v for v in sampled if v.label == "real"]
    fakes = [v for v in sampled if v.label == "fake"]

    def _load_one(video_info: VideoInfo) -> Optional[dict]:
        # Pick one frame per video for diagnostic
        path = video_info.frame_paths[0] if video_info.frame_paths else None
        if not path:
            return None
        img = _load_pil_from_gcs(path, resolution)
        if img is None:
            return None
        return {
            "video_id": video_info.video_id,
            "method": video_info.method,
            "label": video_info.label,
            "image": img,
            "frame_path": path,
        }

    real_samples = [s for s in (_load_one(v) for v in reals) if s]
    fake_samples = [s for s in (_load_one(v) for v in fakes) if s]
    log.info("Loaded %d real, %d fake diagnostic samples", len(real_samples), len(fake_samples))
    return real_samples, fake_samples


# ──────────────────────────────────────────────────────────────────────────────
# GCS upload utility
# ──────────────────────────────────────────────────────────────────────────────


def _upload_to_gcs(local_path: str, gcs_path: str, logger: logging.Logger):
    """Upload a local file to GCS."""
    from google.cloud import storage

    bucket_name = gcs_path.replace("gs://", "").split("/")[0]
    blob_name = gcs_path.split(f"gs://{bucket_name}/", 1)[1]
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    logger.info("Uploaded %s → %s", os.path.basename(local_path), gcs_path)


def _save_csv(rows: List[list], headers: List[str], filepath: str):
    """Write CSV file."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 0A — JPEG Sensitivity Sweep
# ──────────────────────────────────────────────────────────────────────────────


def run_experiment_0a(
    model: torch.nn.Module,
    device: torch.device,
    config: dict,
    output_dir: str,
    output_gcs_folder: str,
    logger: logging.Logger,
    wandb_run=None,
    per_method_per_label: int = 30,
):
    """
    Experiment 0A: JPEG Sensitivity Sweep.

    Load PNG validation data from DeepLive (val split) + VisoMaster + DF40 target_source.
    Run inference at native PNG quality, then JPEG Q50/Q70/Q85/Q95.
    Save per-frame and per-quality CSV with all scores.
    """
    from data.validation_sources import (
        load_deeplive_validation,
        load_visomaster_validation,
        load_df40_pairs_validation,
    )
    from sklearn.metrics import roc_auc_score

    logger.info("=" * 60)
    logger.info("EXPERIMENT 0A: JPEG Sensitivity Sweep")
    logger.info("=" * 60)

    # --- Load validation videos (PNG sources) ---
    deeplive_videos = load_deeplive_validation(
        bucket_name="live-deepfake-methods-real-and-fake-frames-cropped",
        gcs_project=os.environ.get("GOOGLE_CLOUD_PROJECT", "train-cvit2"),
        split="val",
        train_split=0.9,
        val_split=0.1,
        seed=737,
    )
    visomaster_videos = load_visomaster_validation(
        cropped_bucket_name="live-deepfake-methods-real-and-fake-frames-cropped",
        frames_bucket_name="live-deepfake-methods-real-and-fake-frames",
        gcs_project=os.environ.get("GOOGLE_CLOUD_PROJECT", "train-cvit2"),
    )
    df40_pair_json = os.path.join(os.path.dirname(__file__), "dataset/df40_pairs/df40-pair-matching.json")
    df40_videos = load_df40_pairs_validation(
        pair_json_path=df40_pair_json,
        orientation="target_source",
        mode="paired",
    )

    all_videos = deeplive_videos + visomaster_videos + df40_videos
    logger.info(
        "0A: Total validation videos: %d (DeepLive=%d, VisoMaster=%d, DF40=%d)",
        len(all_videos), len(deeplive_videos), len(visomaster_videos), len(df40_videos),
    )

    # --- Stratified sampling: N per method per label for stable per-method stats ---
    # With ~15 methods × N per label × 2 labels × 5 qualities = ~15*N*10 inferences
    # At 30/method: ~4500 inferences ≈ 4 min on A100. Well within budget.
    resolution = config.get("resolution", 224)
    transform = _get_clip_transform(config)

    sampled_videos = _stratified_sample(
        all_videos, per_method_per_label, seed=737, logger=logger,
    )

    logger.info(
        "0A: Stratified %d samples (%d per method per label)...",
        len(sampled_videos), per_method_per_label,
    )

    # Load images
    qualities = ["png", "Q95", "Q85", "Q70", "Q50"]
    quality_map = {"png": None, "Q95": 95, "Q85": 85, "Q70": 70, "Q50": 50}

    frame_rows = []  # Per-frame CSV data
    loaded_count = 0
    failed_count = 0

    for idx, video in enumerate(sampled_videos):
        if idx % 50 == 0:
            logger.info("0A: Processing video %d/%d...", idx, len(sampled_videos))

        frame_path = video.frame_paths[0] if video.frame_paths else None
        if not frame_path:
            failed_count += 1
            continue

        img = _load_pil_from_gcs(frame_path, resolution)
        if img is None:
            failed_count += 1
            continue

        loaded_count += 1
        label_int = 0 if video.label == "real" else 1

        for quality_name in qualities:
            q = quality_map[quality_name]
            if q is not None:
                processed_img = jpeg_compress(img, q)
            else:
                processed_img = img

            fake_prob = _infer_single_image(model, processed_img, transform, device)
            frame_rows.append([
                video.method,
                video.label,
                video.video_id,
                frame_path,
                quality_name,
                label_int,
                round(fake_prob, 6),
                1 if (fake_prob >= 0.5) == (label_int == 1) else 0,  # is_correct
            ])

    logger.info("0A: Loaded %d samples (%d failed)", loaded_count, failed_count)

    # --- Save per-frame CSV ---
    frame_csv_path = os.path.join(output_dir, "0A_jpeg_sensitivity_frames.csv")
    _save_csv(
        frame_rows,
        ["method", "label", "video_id", "frame_path", "quality", "label_int", "fake_prob", "is_correct"],
        frame_csv_path,
    )

    # --- Compute summary metrics per quality ---
    summary_rows = []
    for quality_name in qualities:
        q_rows = [r for r in frame_rows if r[4] == quality_name]
        labels = np.array([r[5] for r in q_rows])
        probs = np.array([r[6] for r in q_rows])
        correct = np.array([r[7] for r in q_rows])

        n_total = len(labels)
        accuracy = correct.mean() if n_total > 0 else 0
        try:
            auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan")
        except Exception:
            auc = float("nan")

        # Per-label stats
        real_mask = labels == 0
        fake_mask = labels == 1
        real_acc = correct[real_mask].mean() if real_mask.sum() > 0 else float("nan")
        fake_acc = correct[fake_mask].mean() if fake_mask.sum() > 0 else float("nan")
        mean_prob_real = probs[real_mask].mean() if real_mask.sum() > 0 else float("nan")
        mean_prob_fake = probs[fake_mask].mean() if fake_mask.sum() > 0 else float("nan")

        summary_rows.append([
            quality_name, n_total,
            round(auc, 4), round(accuracy, 4),
            round(real_acc, 4), round(fake_acc, 4),
            round(mean_prob_real, 4), round(mean_prob_fake, 4),
        ])

        logger.info(
            "0A %s: AUC=%.4f  Accuracy=%.4f  Real_Acc=%.4f  Fake_Acc=%.4f  Mean_Real=%.4f  Mean_Fake=%.4f",
            quality_name, auc, accuracy, real_acc, fake_acc, mean_prob_real, mean_prob_fake,
        )

    summary_csv_path = os.path.join(output_dir, "0A_jpeg_sensitivity_summary.csv")
    _save_csv(
        summary_rows,
        ["quality", "n_samples", "auc", "accuracy", "real_accuracy", "fake_accuracy",
         "mean_prob_real", "mean_prob_fake"],
        summary_csv_path,
    )

    # --- Per-method breakdown at each quality ---
    method_rows = []
    methods = sorted(set(r[0] for r in frame_rows))
    for method in methods:
        for quality_name in qualities:
            q_rows = [r for r in frame_rows if r[0] == method and r[4] == quality_name]
            if not q_rows:
                continue
            labels = np.array([r[5] for r in q_rows])
            probs = np.array([r[6] for r in q_rows])
            correct = np.array([r[7] for r in q_rows])
            try:
                auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan")
            except Exception:
                auc = float("nan")
            method_rows.append([
                method, quality_name, len(q_rows),
                round(auc, 4), round(correct.mean(), 4),
                round(probs.mean(), 4),
            ])

    method_csv_path = os.path.join(output_dir, "0A_jpeg_sensitivity_per_method.csv")
    _save_csv(
        method_rows,
        ["method", "quality", "n_samples", "auc", "accuracy", "mean_prob"],
        method_csv_path,
    )

    # --- Upload ---
    for fname in ["0A_jpeg_sensitivity_frames.csv", "0A_jpeg_sensitivity_summary.csv",
                   "0A_jpeg_sensitivity_per_method.csv"]:
        local = os.path.join(output_dir, fname)
        _upload_to_gcs(local, f"{output_gcs_folder}/{fname}", logger)

    # --- Log to W&B ---
    if wandb_run:
        for row in summary_rows:
            wandb_run.log({
                f"0A/{row[0]}/auc": row[2],
                f"0A/{row[0]}/accuracy": row[3],
            })

    logger.info("0A: Complete. Results at %s/0A_*", output_gcs_folder)
    return {"summary": summary_rows, "n_samples": loaded_count}


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 0B — Lighting Sensitivity Profile
# ──────────────────────────────────────────────────────────────────────────────


def run_experiment_0b(
    model: torch.nn.Module,
    device: torch.device,
    config: dict,
    output_dir: str,
    output_gcs_folder: str,
    logger: logging.Logger,
    wandb_run=None,
    per_method_per_label: int = 5,
):
    """
    Experiment 0B: Lighting Sensitivity Profile.

    Load face crops stratified across methods from DeepLive val + VisoMaster.
    With ~8-10 methods × 5 per label × 2 labels ≈ 80-100 samples.
    Apply 8 brightness multipliers + 7 gamma values = 15 perturbations per sample.
    Total: ~1200-1500 inferences + cosine analysis. ~2 min on A100.
    """
    from data.validation_sources import (
        load_deeplive_validation,
        load_visomaster_validation,
    )

    logger.info("=" * 60)
    logger.info("EXPERIMENT 0B: Lighting Sensitivity Profile")
    logger.info("=" * 60)

    # Load validation videos (PNG, production-like sources)
    deeplive_videos = load_deeplive_validation(
        bucket_name="live-deepfake-methods-real-and-fake-frames-cropped",
        gcs_project=os.environ.get("GOOGLE_CLOUD_PROJECT", "train-cvit2"),
        split="val",
        train_split=0.9,
        val_split=0.1,
        seed=737,
    )
    visomaster_videos = load_visomaster_validation(
        cropped_bucket_name="live-deepfake-methods-real-and-fake-frames-cropped",
        frames_bucket_name="live-deepfake-methods-real-and-fake-frames",
        gcs_project=os.environ.get("GOOGLE_CLOUD_PROJECT", "train-cvit2"),
    )

    all_videos = deeplive_videos + visomaster_videos
    resolution = config.get("resolution", 224)
    transform = _get_clip_transform(config)

    # Stratified: per_method_per_label samples per (method, label) pair
    # For 0B we need fewer total samples since each gets 15 perturbations,
    # but we need method coverage for per-method sensitivity analysis.
    real_samples, fake_samples = _load_samples_for_diagnostic(
        all_videos, per_method_per_label=per_method_per_label,
        resolution=resolution, seed=737, logger=logger,
    )

    logger.info("0B: Using %d real + %d fake samples (%d per method per label)",
                len(real_samples), len(fake_samples), per_method_per_label)

    # --- Define perturbation grid ---
    brightness_factors = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0]
    gamma_values = [0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0]

    all_samples = real_samples + fake_samples
    frame_rows = []

    # --- Brightness sweep ---
    logger.info("0B: Running brightness sweep (%d factors × %d samples)...",
                len(brightness_factors), len(all_samples))

    for sample in all_samples:
        # Baseline (unperturbed)
        base_prob = _infer_single_image(model, sample["image"], transform, device)

        for brightness in brightness_factors:
            perturbed = apply_brightness(sample["image"], brightness)
            prob = _infer_single_image(model, perturbed, transform, device)
            frame_rows.append([
                sample["method"],
                sample["label"],
                sample["video_id"],
                sample["frame_path"],
                "brightness",
                brightness,
                round(prob, 6),
                round(base_prob, 6),
                round(abs(prob - base_prob), 6),
            ])

    # --- Gamma sweep ---
    logger.info("0B: Running gamma sweep (%d values × %d samples)...",
                len(gamma_values), len(all_samples))

    for sample in all_samples:
        base_prob = _infer_single_image(model, sample["image"], transform, device)

        for gamma in gamma_values:
            perturbed = apply_gamma(sample["image"], gamma)
            prob = _infer_single_image(model, perturbed, transform, device)
            frame_rows.append([
                sample["method"],
                sample["label"],
                sample["video_id"],
                sample["frame_path"],
                "gamma",
                gamma,
                round(prob, 6),
                round(base_prob, 6),
                round(abs(prob - base_prob), 6),
            ])

    # --- Save per-frame detailed CSV ---
    frame_csv_path = os.path.join(output_dir, "0B_lighting_sensitivity_frames.csv")
    _save_csv(
        frame_rows,
        ["method", "label", "video_id", "frame_path", "perturbation_type",
         "perturbation_value", "fake_prob", "base_prob", "delta_prob"],
        frame_csv_path,
    )

    # --- Compute summary: mean delta by perturbation × label ---
    summary_rows = []
    for perturb_type in ["brightness", "gamma"]:
        values = brightness_factors if perturb_type == "brightness" else gamma_values
        for value in values:
            for label in ["real", "fake"]:
                matching = [r for r in frame_rows
                            if r[4] == perturb_type and r[5] == value and r[1] == label]
                if not matching:
                    continue
                probs = np.array([r[6] for r in matching])
                base_probs = np.array([r[7] for r in matching])
                deltas = np.array([r[8] for r in matching])

                # Count flips: label changed when prob crossed 0.5
                flips = 0
                for r in matching:
                    base_pred = 1 if r[7] >= 0.5 else 0
                    perturbed_pred = 1 if r[6] >= 0.5 else 0
                    if base_pred != perturbed_pred:
                        flips += 1

                summary_rows.append([
                    perturb_type, value, label,
                    len(matching),
                    round(probs.mean(), 4),
                    round(base_probs.mean(), 4),
                    round(deltas.mean(), 4),
                    round(deltas.max(), 4),
                    round(np.percentile(deltas, 90), 4),
                    round(probs.std(), 4),
                    flips,
                    round(flips / len(matching), 4) if matching else 0,
                ])

    summary_csv_path = os.path.join(output_dir, "0B_lighting_sensitivity_summary.csv")
    _save_csv(
        summary_rows,
        ["perturbation_type", "perturbation_value", "label", "n_samples",
         "mean_prob", "mean_base_prob", "mean_delta", "max_delta", "p90_delta",
         "prob_std", "n_flips", "flip_rate"],
        summary_csv_path,
    )

    # --- Cosine distance analysis (if we can access embeddings) ---
    # This measures how much the CLIP embedding moves under perturbation
    cosine_rows = _compute_cosine_sensitivity(
        model, all_samples, transform, device,
        brightness_factors, gamma_values, logger,
    )
    if cosine_rows:
        cosine_csv_path = os.path.join(output_dir, "0B_cosine_distance_vs_perturbation.csv")
        _save_csv(
            cosine_rows,
            ["label", "video_id", "perturbation_type", "perturbation_value",
             "cosine_distance", "l2_distance"],
            cosine_csv_path,
        )
        _upload_to_gcs(cosine_csv_path, f"{output_gcs_folder}/0B_cosine_distance_vs_perturbation.csv", logger)

    # --- Upload ---
    for fname in ["0B_lighting_sensitivity_frames.csv", "0B_lighting_sensitivity_summary.csv"]:
        local = os.path.join(output_dir, fname)
        _upload_to_gcs(local, f"{output_gcs_folder}/{fname}", logger)

    # --- Log to W&B ---
    if wandb_run:
        for row in summary_rows:
            wandb_run.log({
                f"0B/{row[0]}_{row[1]}/{row[2]}/mean_delta": row[6],
                f"0B/{row[0]}_{row[1]}/{row[2]}/flip_rate": row[11],
            })

    # Print key findings
    bright_deltas = [r for r in summary_rows if r[0] == "brightness"]
    gamma_deltas = [r for r in summary_rows if r[0] == "gamma"]
    max_bright_delta = max((r[7] for r in bright_deltas), default=0)
    max_gamma_delta = max((r[7] for r in gamma_deltas), default=0)
    logger.info("0B: Max brightness delta=%.4f, Max gamma delta=%.4f", max_bright_delta, max_gamma_delta)
    logger.info("0B: Complete. Results at %s/0B_*", output_gcs_folder)

    return {"summary": summary_rows, "n_samples": len(all_samples)}


@torch.no_grad()
def _compute_cosine_sensitivity(
    model, samples, transform, device,
    brightness_factors, gamma_values, logger,
) -> List[list]:
    """
    Measure how much the CLIP embedding vector moves under each perturbation.
    This tells us the cosine distance that feeds into ArcFace.
    """
    rows = []

    # Try to extract the backbone for embedding extraction
    if not hasattr(model, "backbone"):
        logger.info("0B: Model has no 'backbone' attribute — skipping cosine distance analysis.")
        return rows

    backbone = model.backbone
    backbone.eval()

    def _get_embedding(pil_img):
        tensor = transform(pil_img).unsqueeze(0).to(device)
        # Most CLIP wrappers accept raw images and return features
        try:
            out = backbone(tensor)
            if isinstance(out, dict):
                emb = out.get("pooler_output", out.get("last_hidden_state", None))
                if emb is not None and emb.dim() == 3:
                    emb = emb[:, 0, :]  # CLS token
            else:
                emb = out
            return emb.squeeze(0)
        except Exception as e:
            logger.debug("Embedding extraction failed: %s", e)
            return None

    for sample in samples[:20]:  # Cap at 20 for speed
        base_emb = _get_embedding(sample["image"])
        if base_emb is None:
            continue
        base_emb_norm = torch.nn.functional.normalize(base_emb.unsqueeze(0), dim=1)

        for brightness in brightness_factors:
            perturbed = apply_brightness(sample["image"], brightness)
            emb = _get_embedding(perturbed)
            if emb is None:
                continue
            emb_norm = torch.nn.functional.normalize(emb.unsqueeze(0), dim=1)
            cos_dist = 1.0 - torch.nn.functional.cosine_similarity(base_emb_norm, emb_norm).item()
            l2_dist = torch.norm(base_emb - emb).item()
            rows.append([
                sample["label"], sample["video_id"],
                "brightness", brightness,
                round(cos_dist, 6), round(l2_dist, 4),
            ])

        for gamma in gamma_values:
            perturbed = apply_gamma(sample["image"], gamma)
            emb = _get_embedding(perturbed)
            if emb is None:
                continue
            emb_norm = torch.nn.functional.normalize(emb.unsqueeze(0), dim=1)
            cos_dist = 1.0 - torch.nn.functional.cosine_similarity(base_emb_norm, emb_norm).item()
            l2_dist = torch.norm(base_emb - emb).item()
            rows.append([
                sample["label"], sample["video_id"],
                "gamma", gamma,
                round(cos_dist, 6), round(l2_dist, 4),
            ])

    logger.info("0B: Computed %d cosine distance measurements", len(rows))
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 0C — Teams v2 Sanity Check
# ──────────────────────────────────────────────────────────────────────────────


def _load_teams_v2_validation_samples(
    bucket_name: str,
    max_real: int = 100,
    max_fake: int = 100,
    seed: int = 737,
    logger: Optional[logging.Logger] = None,
) -> List[VideoInfo]:
    """
    Load a random sample of Teams v2 pairs for sanity checking.

    Discovers samples from the Teams v2 bucket (same structure as v1),
    randomly selects pairs, and builds VideoInfo objects.
    """
    from google.cloud import storage as gcs_storage

    log = logger or logging.getLogger(__name__)
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "train-cvit2")
    client = gcs_storage.Client(project=project)
    bucket = client.bucket(bucket_name)

    # Discover manifests
    log.info("0C: Discovering Teams v2 samples from gs://%s ...", bucket_name)
    try:
        manifest_blobs = list(bucket.list_blobs(prefix="samples/", match_glob="**/manifest.json"))
    except TypeError:
        # Older google-cloud-storage versions don't support match_glob
        log.warning("0C: match_glob not supported, falling back to prefix scan")
        manifest_blobs = [
            b for b in bucket.list_blobs(prefix="samples/")
            if b.name.endswith("/manifest.json")
        ]
    log.info("0C: Found %d manifests", len(manifest_blobs))

    samples = []
    for blob in manifest_blobs:
        try:
            manifest = json.loads(blob.download_as_text())
            if not manifest.get("pair_complete", False):
                continue
            samples.append(manifest)
        except Exception as e:
            log.debug("Failed to read manifest %s: %s", blob.name, e)

    log.info("0C: %d complete paired samples", len(samples))

    rng = random.Random(seed)
    rng.shuffle(samples)
    selected = samples[: max(max_real, max_fake)]

    videos = []
    for sample in selected:
        sample_id = sample.get("sample_id", "")
        strategy = sample.get("strategy", "unknown")
        # Teams v2 manifests have frame_count_real / frame_count_fake, not frame_count
        frame_count_real = int(sample.get("frame_count_real", sample.get("frame_count", 0)))
        frame_count_fake = int(sample.get("frame_count_fake", sample.get("frame_count", 0)))
        method = f"teams_v2_{strategy}"

        # Teams frames are JPG
        anchor_indices = [0, 2, 4, 6, 8, 10, 12, 14]
        real_frames = [
            f"gs://{bucket_name}/samples/{sample_id}/frames/real/frame_{i:04d}.jpg"
            for i in anchor_indices if i < frame_count_real
        ]
        fake_frames = [
            f"gs://{bucket_name}/samples/{sample_id}/frames/fake/frame_{i:04d}.jpg"
            for i in anchor_indices if i < frame_count_fake
        ]

        if not real_frames or not fake_frames:
            continue

        identity = hash(sample_id) & 0x7FFFFFFF

        videos.append(VideoInfo(
            label="real",
            method=method,
            video_id=f"{sample_id}_real",
            frame_paths=real_frames,
            identity=identity,
        ))
        videos.append(VideoInfo(
            label="fake",
            method=method,
            video_id=f"{sample_id}_fake",
            frame_paths=fake_frames,
            identity=identity,
        ))

    real_count = sum(1 for v in videos if v.label == "real")
    fake_count = sum(1 for v in videos if v.label == "fake")
    log.info("0C: Built %d VideoInfo entries (real=%d, fake=%d)", len(videos), real_count, fake_count)
    return videos


def run_experiment_0c(
    model: torch.nn.Module,
    device: torch.device,
    config: dict,
    output_dir: str,
    output_gcs_folder: str,
    logger: logging.Logger,
    wandb_run=None,
    teams_v2_bucket: str = "live-deepfake-methods-real-and-fake-frames-cropped-teams-v2",
    max_samples: int = 100,
):
    """
    Experiment 0C: Teams v2 Sanity Check.

    Run R9_A on a random subset of Teams v2 (100 real, 100 fake).
    Compare score distributions. Look for anomalies.
    """
    from sklearn.metrics import roc_auc_score

    logger.info("=" * 60)
    logger.info("EXPERIMENT 0C: Teams v2 Sanity Check")
    logger.info("=" * 60)

    teams_videos = _load_teams_v2_validation_samples(
        bucket_name=teams_v2_bucket,
        max_real=max_samples,
        max_fake=max_samples,
        seed=737,
        logger=logger,
    )

    if not teams_videos:
        logger.error("0C: No Teams v2 samples loaded. Aborting experiment.")
        return {"error": "no_samples"}

    resolution = config.get("resolution", 224)
    transform = _get_clip_transform(config)

    # --- Run inference (1 frame per video for sanity check, then also multi-frame) ---
    frame_rows = []

    for idx, video in enumerate(teams_videos):
        if idx % 50 == 0:
            logger.info("0C: Processing video %d/%d...", idx, len(teams_videos))

        label_int = 0 if video.label == "real" else 1
        frame_probs = []

        for frame_path in video.frame_paths:
            img = _load_pil_from_gcs(frame_path, resolution)
            if img is None:
                continue

            prob = _infer_single_image(model, img, transform, device)
            frame_probs.append(prob)

            frame_rows.append([
                video.method,
                video.label,
                video.video_id,
                frame_path,
                label_int,
                round(prob, 6),
                1 if (prob >= 0.5) == (label_int == 1) else 0,
            ])

    # --- Save per-frame CSV ---
    frame_csv_path = os.path.join(output_dir, "0C_teams_v2_frames.csv")
    _save_csv(
        frame_rows,
        ["method", "label", "video_id", "frame_path", "label_int", "fake_prob", "is_correct"],
        frame_csv_path,
    )

    # --- Aggregate per-video ---
    video_groups = defaultdict(list)
    for row in frame_rows:
        video_groups[(row[0], row[1], row[2])].append(row)

    video_rows = []
    for (method, label, video_id), rows in video_groups.items():
        probs = [r[5] for r in rows]
        avg_prob = np.mean(probs)
        label_int = rows[0][4]
        prediction = 1 if avg_prob >= 0.5 else 0
        is_correct = 1 if prediction == label_int else 0
        video_rows.append([
            method, label, video_id,
            round(avg_prob, 6),
            prediction,
            is_correct,
            len(probs),
            round(np.std(probs), 6) if len(probs) > 1 else 0,
        ])

    video_csv_path = os.path.join(output_dir, "0C_teams_v2_videos.csv")
    _save_csv(
        video_rows,
        ["method", "label", "video_id", "avg_prob", "prediction", "is_correct",
         "n_frames", "intra_video_std"],
        video_csv_path,
    )

    # --- Summary stats ---
    summary_rows = []
    all_labels = np.array([r[4] for r in video_rows])
    all_probs = np.array([r[3] for r in video_rows])
    all_correct = np.array([r[5] for r in video_rows])

    try:
        overall_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    except Exception:
        overall_auc = float("nan")

    summary_rows.append(["overall", len(video_rows),
                          round(overall_auc, 4), round(all_correct.mean(), 4)])

    for label in ["real", "fake"]:
        mask = np.array([r[1] == label for r in video_rows])
        if mask.sum() == 0:
            continue
        probs = all_probs[mask]
        correct = all_correct[mask]
        summary_rows.append([
            label, int(mask.sum()),
            round(probs.mean(), 4), round(correct.mean(), 4),
            round(probs.std(), 4),
            round(np.percentile(probs, 10), 4),
            round(np.percentile(probs, 50), 4),
            round(np.percentile(probs, 90), 4),
        ])

    # Per strategy
    strategies = sorted(set(r[0] for r in video_rows))
    for strategy in strategies:
        for label in ["real", "fake"]:
            matching = [r for r in video_rows if r[0] == strategy and r[1] == label]
            if not matching:
                continue
            probs = np.array([r[3] for r in matching])
            correct = np.array([r[5] for r in matching])
            summary_rows.append([
                f"{strategy}_{label}", len(matching),
                round(probs.mean(), 4), round(correct.mean(), 4),
                round(probs.std(), 4),
                round(np.percentile(probs, 10), 4),
                round(np.percentile(probs, 50), 4),
                round(np.percentile(probs, 90), 4),
            ])

    summary_csv_path = os.path.join(output_dir, "0C_teams_v2_summary.csv")
    _save_csv(
        summary_rows,
        ["group", "n_videos", "auc_or_mean_prob", "accuracy_or_mean_correct",
         "std", "p10", "p50", "p90"],
        summary_csv_path,
    )

    # --- Score histogram data for distribution comparison ---
    hist_rows = []
    for row in video_rows:
        hist_rows.append([row[0], row[1], row[2], row[3]])

    hist_csv_path = os.path.join(output_dir, "0C_teams_v2_score_histogram.csv")
    _save_csv(hist_rows, ["method", "label", "video_id", "avg_prob"], hist_csv_path)

    # --- Upload ---
    for fname in ["0C_teams_v2_frames.csv", "0C_teams_v2_videos.csv",
                   "0C_teams_v2_summary.csv", "0C_teams_v2_score_histogram.csv"]:
        local = os.path.join(output_dir, fname)
        _upload_to_gcs(local, f"{output_gcs_folder}/{fname}", logger)

    # --- Log to W&B ---
    if wandb_run:
        wandb_run.log({
            "0C/overall_auc": overall_auc,
            "0C/overall_accuracy": all_correct.mean(),
            "0C/n_videos": len(video_rows),
        })

    logger.info("0C: AUC=%.4f  Accuracy=%.4f  (%d videos)", overall_auc, all_correct.mean(), len(video_rows))
    logger.info("0C: Complete. Results at %s/0C_*", output_gcs_folder)

    return {"auc": overall_auc, "accuracy": float(all_correct.mean()), "n_videos": len(video_rows)}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="R10 Pre-Training Diagnostic Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--detector_path", type=str, default="./config/detector/effort.yaml")
    parser.add_argument("--train_config_path", type=str, default="./config/train_config.yaml")
    parser.add_argument("--dataloader_config", type=str, default="./config/dataloader_config.yml")

    parser.add_argument("--checkpoint_gcs_path", type=str, required=True,
                        help="GCS path to R9_A checkpoint .pth file.")
    parser.add_argument("--checkpoint_local_path", type=str,
                        default="./weights/r10_diag_checkpoint.pth")

    parser.add_argument("--experiments", type=str, default="0A,0B,0C",
                        help="Comma-separated list of experiments to run (0A, 0B, 0C).")

    parser.add_argument("--output_gcs_folder", type=str, required=True,
                        help="GCS folder to save all results.")

    # 0A-specific
    parser.add_argument("--0a_per_method", type=int, default=30,
                        help="Samples per method per label for Experiment 0A (default 30).")

    # 0B-specific
    parser.add_argument("--0b_per_method", type=int, default=5,
                        help="Samples per method per label for Experiment 0B (default 5). "
                             "Lower than 0A because each sample gets 15 perturbations.")

    # 0C-specific
    parser.add_argument("--teams_v2_bucket", type=str,
                        default="live-deepfake-methods-real-and-fake-frames-cropped-teams-v2",
                        help="GCS bucket for Teams v2 data.")
    parser.add_argument("--0c_max_samples", type=int, default=100,
                        help="Max samples per label for Teams v2 sanity check.")

    parser.add_argument("--disable_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str,
                        default=os.environ.get("WANDB_PROJECT", "r10-diagnostics"))

    args = parser.parse_args()

    # --- Setup ---
    log_dir = "./logs_r10_diagnostics"
    os.makedirs(log_dir, exist_ok=True)
    logger = create_logger(os.path.join(log_dir, "diagnostics.log"))

    output_dir = tempfile.mkdtemp(prefix="r10_diag_")
    logger.info("Local temp output dir: %s", output_dir)

    experiments = [e.strip().upper() for e in args.experiments.split(",")]
    logger.info("Experiments to run: %s", experiments)

    wandb_run = None
    if not args.disable_wandb and wandb is not None:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"R10_diagnostics_{'_'.join(experiments)}",
            config={
                "checkpoint": args.checkpoint_gcs_path,
                "experiments": experiments,
            },
        )

    # --- Load config & model ---
    config = _load_config(args.detector_path, args.train_config_path, args.dataloader_config)
    init_seed(config)
    if config.get("cudnn"):
        cudnn.benchmark = True

    config.setdefault("gcs_assets", {})
    config["gcs_assets"]["base_checkpoint"] = {
        "gcs_path": args.checkpoint_gcs_path,
        "local_path": args.checkpoint_local_path,
    }
    downloaded = download_assets_from_gcs(config, logger)
    if not downloaded or "base_checkpoint" not in downloaded:
        logger.error("Failed to download checkpoint. Aborting.")
        return

    model, device, saved_config = _load_model(config, args.checkpoint_local_path, logger)

    logger.info("Model on %s, inference mode", device)

    # --- Run experiments ---
    results = {}
    t_start = time.time()

    if "0A" in experiments:
        t0 = time.time()
        results["0A"] = run_experiment_0a(
            model, device, config, output_dir, args.output_gcs_folder, logger, wandb_run,
            per_method_per_label=args.__dict__["0a_per_method"],
        )
        logger.info("0A took %.1f seconds", time.time() - t0)

    if "0B" in experiments:
        t0 = time.time()
        results["0B"] = run_experiment_0b(
            model, device, config, output_dir, args.output_gcs_folder, logger, wandb_run,
            per_method_per_label=args.__dict__["0b_per_method"],
        )
        logger.info("0B took %.1f seconds", time.time() - t0)

    if "0C" in experiments:
        t0 = time.time()
        results["0C"] = run_experiment_0c(
            model, device, config, output_dir, args.output_gcs_folder, logger, wandb_run,
            teams_v2_bucket=args.teams_v2_bucket,
            max_samples=args.__dict__["0c_max_samples"],
        )
        logger.info("0C took %.1f seconds", time.time() - t0)

    # --- Save master summary ---
    summary_path = os.path.join(output_dir, "diagnostics_summary.json")
    with open(summary_path, "w") as f:
        # Convert numpy values to native Python types
        def _sanitize(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            return obj

        json.dump({
            "checkpoint": args.checkpoint_gcs_path,
            "experiments_run": experiments,
            "total_time_seconds": round(time.time() - t_start, 1),
            "results": _sanitize(results),
        }, f, indent=2)

    _upload_to_gcs(summary_path, f"{args.output_gcs_folder}/diagnostics_summary.json", logger)

    logger.info("=" * 60)
    logger.info("ALL DIAGNOSTICS COMPLETE in %.1f seconds", time.time() - t_start)
    logger.info("Results folder: %s", args.output_gcs_folder)
    logger.info("=" * 60)

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
