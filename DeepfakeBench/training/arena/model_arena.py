#!/usr/bin/env python3
"""
Model Arena — Cross-checkpoint comparison on target-domain data.

Runs multiple EFFORT checkpoints against multiple GCS data sources,
then produces comprehensive analysis including:
  - Per-frame inference CSVs (reuses batch_inference_gcs patterns)
  - Probability health metrics (distribution, calibration, EER, threshold health)
  - Frame-to-frame stability analysis (jitter, flip rate)
  - Per-method breakdowns (weak areas, target method TPR)
  - Embedding visualization (T-SNE / UMAP)
  - Window-based voting strategy search (per model + cross-model)
  - Final leaderboard ranking models holistically

Usage:
  # Full pipeline (inference + analysis + strategy):
  python model_arena.py --config arena_config.yaml

  # Inference only (on Vertex AI):
  python model_arena.py --config arena_config.yaml --phase inference

  # Analysis only (reads pre-existing CSVs):
  python model_arena.py --config arena_config.yaml --phase analysis

  # Strategy search only:
  python model_arena.py --config arena_config.yaml --phase strategy

  # Dry run (show plan, no execution):
  python model_arena.py --config arena_config.yaml --dry-run

  # Smoke test (load all models, score ~16 frames per source, save outputs):
  python model_arena.py --config arena_config.yaml --smoke-test

  # Provide pre-existing inference dir:
  python model_arena.py --config arena_config.yaml --phase analysis --inference-dir ./arena_results/20260309_150000/inference
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time

# Ensure the training root (parent of arena/) is on sys.path so that
# top-level packages like `detectors`, `trainer`, `utils`, `data` are importable.
_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TRAINING_ROOT not in sys.path:
    sys.path.insert(0, _TRAINING_ROOT)
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("model-arena")

# ---------------------------------------------------------------------------
# Lazy imports — heavy libraries loaded only when needed
# ---------------------------------------------------------------------------
_torch = None
_cv2 = None
_yaml = None
_storage = None


def _import_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _import_cv2():
    global _cv2
    if _cv2 is None:
        import cv2
        _cv2 = cv2
    return _cv2


def _import_yaml():
    global _yaml
    if _yaml is None:
        import yaml
        _yaml = yaml
    return _yaml


def _import_gcs():
    global _storage
    if _storage is None:
        from google.cloud import storage
        _storage = storage
    return _storage


# ===========================================================================
# Configuration
# ===========================================================================
@dataclass
class CheckpointConfig:
    name: str
    path: str
    backbone: str = "B16"
    notes: str = ""


@dataclass
class DataSourceConfig:
    name: str
    bucket: str
    layout: str = "auto"
    chronological: bool = False
    notes: str = ""


@dataclass
class ArenaConfig:
    """Parsed arena configuration."""
    output_dir: str = "./arena_results"
    output_gcs_folder: str = ""
    checkpoints: List[CheckpointConfig] = field(default_factory=list)
    data_sources: List[DataSourceConfig] = field(default_factory=list)
    # Inference
    batch_size: int = 256
    num_workers: int = 12
    save_embeddings: bool = True
    detector_config: str = "./config/detector/effort.yaml"
    train_config: str = "./config/defaults.yaml"
    # Analysis
    stability_enabled: bool = True
    stability_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    prob_health_enabled: bool = True
    healthy_threshold_range: Tuple[float, float] = (0.25, 0.75)
    embeddings_enabled: bool = True
    embedding_method: str = "tsne"
    embedding_perplexity: int = 30
    embedding_max_samples: int = 5000
    per_method_enabled: bool = True
    target_methods: List[str] = field(default_factory=lambda: ["FaceSwap", "inswap", "mobileswap", "simswap"])
    # Strategy
    strategy_enabled: bool = True
    window_sizes: List[int] = field(default_factory=lambda: [16, 24, 32])
    threshold_range: Tuple[float, float] = (0.20, 0.95)
    threshold_step: float = 0.01
    k_ratio_range: Tuple[float, float] = (0.30, 0.80)
    k_ratio_step: float = 0.02
    max_teams_fpr: float = 0.0
    strategy_healthy_range: Tuple[float, float] = (0.15, 0.85)
    uncertain_margins: List[int] = field(default_factory=lambda: [0, 1, 2, 3])


def load_config(path: str) -> ArenaConfig:
    """Load arena config from YAML."""
    yaml = _import_yaml()
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = ArenaConfig()
    cfg.output_dir = raw.get("output_dir", cfg.output_dir)
    cfg.output_gcs_folder = raw.get("output_gcs_folder", "")

    # Checkpoints
    for name, cdict in raw.get("checkpoints", {}).items():
        cfg.checkpoints.append(CheckpointConfig(
            name=name,
            path=cdict["path"],
            backbone=cdict.get("backbone", "B16"),
            notes=cdict.get("notes", ""),
        ))

    # Data sources
    for name, ddict in raw.get("data_sources", {}).items():
        cfg.data_sources.append(DataSourceConfig(
            name=name,
            bucket=ddict["bucket"],
            layout=ddict.get("layout", "auto"),
            chronological=ddict.get("chronological", False),
            notes=ddict.get("notes", ""),
        ))

    # Inference
    inf = raw.get("inference", {})
    cfg.batch_size = inf.get("batch_size", cfg.batch_size)
    cfg.num_workers = inf.get("num_workers", cfg.num_workers)
    cfg.save_embeddings = inf.get("save_embeddings", cfg.save_embeddings)
    cfg.detector_config = inf.get("detector_config", cfg.detector_config)
    cfg.train_config = inf.get("train_config", cfg.train_config)

    # Analysis
    ana = raw.get("analysis", {})
    stab = ana.get("stability", {})
    cfg.stability_enabled = stab.get("enabled", True)
    cfg.stability_thresholds = stab.get("thresholds", cfg.stability_thresholds)

    ph = ana.get("prob_health", {})
    cfg.prob_health_enabled = ph.get("enabled", True)
    hr = ph.get("healthy_threshold_range", list(cfg.healthy_threshold_range))
    cfg.healthy_threshold_range = tuple(hr)

    emb = ana.get("embeddings", {})
    cfg.embeddings_enabled = emb.get("enabled", True)
    cfg.embedding_method = emb.get("method", cfg.embedding_method)
    cfg.embedding_perplexity = emb.get("perplexity", cfg.embedding_perplexity)
    cfg.embedding_max_samples = emb.get("max_samples", cfg.embedding_max_samples)

    pm = ana.get("per_method", {})
    cfg.per_method_enabled = pm.get("enabled", True)
    cfg.target_methods = pm.get("target_methods", cfg.target_methods)

    # Strategy
    strat = raw.get("strategy", {})
    cfg.strategy_enabled = strat.get("enabled", True)
    cfg.window_sizes = strat.get("window_sizes", cfg.window_sizes)
    tr = strat.get("threshold_range", list(cfg.threshold_range))
    cfg.threshold_range = tuple(tr)
    cfg.threshold_step = strat.get("threshold_step", cfg.threshold_step)
    kr = strat.get("k_ratio_range", list(cfg.k_ratio_range))
    cfg.k_ratio_range = tuple(kr)
    cfg.k_ratio_step = strat.get("k_ratio_step", cfg.k_ratio_step)
    cfg.max_teams_fpr = strat.get("max_teams_fpr", cfg.max_teams_fpr)
    shr = strat.get("healthy_threshold_range", list(cfg.strategy_healthy_range))
    cfg.strategy_healthy_range = tuple(shr)
    cfg.uncertain_margins = strat.get("uncertain_margins", cfg.uncertain_margins)

    return cfg


# ===========================================================================
# Phase 1: Inference
# ===========================================================================
# We re-use the model loading and GCS discovery logic from batch_inference_gcs
# but run multiple checkpoints × multiple buckets.

# CLIP normalization constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


@dataclass
class FrameRecord:
    """Metadata for a single frame to score."""
    bucket: str
    blob_path: str
    label: int              # 0=real, 1=fake, -1=unknown
    method: str
    video_id: str
    frame_name: str
    strategy: str = ""
    extra: Dict[str, str] = field(default_factory=dict)


# ── GCS bucket discovery ──────────────────────────────────────────────

# Known deepfake methods from poc-phase-1 bucket
_POC_METHODS = {
    "deeplivecam", "face2face", "faceshifter", "faceswap",
    "fomm", "fsgan", "infoswap", "inswap", "liveportrait",
    "megaportraits", "mobileswap", "mrfa", "oneshot",
    "simswap", "styleswap", "uniface",
}


def _extract_method_from_folder(folder_name: str) -> str:
    """Extract deepfake method from poc-phase-1 folder name."""
    lower = folder_name.lower()
    for m in _POC_METHODS:
        if lower.startswith(m):
            return m.capitalize() if m != "face2face" else "Face2Face"
    # Fallback: use the prefix before the first underscore as method
    parts = folder_name.split("_")
    return parts[0] if parts else "unknown"


def _extract_teams_segment(filename: str) -> str:
    """Extract segment ID from teams-flat filename."""
    # Pattern: Cam_Test__s32_308.0_frame_000349_crop_000__4df523de.jpg
    # Also handle new naming: {prefix}__{original_filename}.ext
    parts = filename.split("__")
    if len(parts) >= 2:
        return parts[0]
    return filename.rsplit(".", 1)[0]


def discover_poc_phase1(bucket_name: str) -> List[FrameRecord]:
    """Discover frames in poc-phase-1-test bucket."""
    storage = _import_gcs()
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    records = []

    for label_str, label_int in [("fake", 1), ("real", 0)]:
        blobs = bucket.list_blobs(prefix=f"{label_str}/")
        for blob in blobs:
            name = blob.name
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            parts = name.split("/")
            if len(parts) < 3:
                continue
            folder = parts[1]
            frame_name = parts[-1]
            method = _extract_method_from_folder(folder) if label_int == 1 else "real"
            video_id = folder
            records.append(FrameRecord(
                bucket=bucket_name,
                blob_path=name,
                label=label_int,
                method=method,
                video_id=video_id,
                frame_name=frame_name,
            ))

    logger.info("Discovered %d frames in %s (poc layout)", len(records), bucket_name)
    return records


def discover_teams_flat(bucket_name: str) -> List[FrameRecord]:
    """Discover frames in teams-flat bucket."""
    storage = _import_gcs()
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    records = []

    for label_str, label_int in [("fake", 1), ("real", 0)]:
        blobs = bucket.list_blobs(prefix=f"{label_str}/")
        for blob in blobs:
            name = blob.name
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            parts = name.split("/")
            if len(parts) < 2:
                continue
            frame_name = parts[-1]
            segment = _extract_teams_segment(frame_name)
            method = "teams_passthrough" if label_int == 1 else "real"
            records.append(FrameRecord(
                bucket=bucket_name,
                blob_path=name,
                label=label_int,
                method=method,
                video_id=segment,
                frame_name=frame_name,
            ))

    logger.info("Discovered %d frames in %s (teams_flat layout)", len(records), bucket_name)
    return records


def discover_teams_paired(bucket_name: str) -> List[FrameRecord]:
    """Discover frames in teams-paired layout (live-deepfake buckets)."""
    storage = _import_gcs()
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    records = []

    blobs = list(bucket.list_blobs(prefix="samples/"))
    for blob in blobs:
        name = blob.name
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        # Expected: samples/{sample_id}/frames/{fake,real}/frame_NNNN.jpg
        parts = name.split("/")
        if len(parts) < 5:
            continue
        sample_id = parts[1]
        label_str = parts[3]
        frame_name = parts[-1]

        if label_str not in ("fake", "real"):
            continue

        label_int = 1 if label_str == "fake" else 0
        # Extract strategy from sample_id (e.g., "edge_cases_0001" → "edge_cases")
        strategy = "_".join(sample_id.split("_")[:-1]) if "_" in sample_id else sample_id
        method = f"teams_{strategy}" if label_int == 1 else "real"

        records.append(FrameRecord(
            bucket=bucket_name,
            blob_path=name,
            label=label_int,
            method=method,
            video_id=sample_id,
            frame_name=frame_name,
            strategy=strategy,
        ))

    logger.info("Discovered %d frames in %s (teams_paired layout)", len(records), bucket_name)
    return records


def discover_bucket(source: DataSourceConfig) -> List[FrameRecord]:
    """Discover all frames in a data source bucket."""
    layout = source.layout.lower()
    if layout == "poc":
        return discover_poc_phase1(source.bucket)
    elif layout == "teams_flat":
        return discover_teams_flat(source.bucket)
    elif layout == "teams_paired":
        return discover_teams_paired(source.bucket)
    elif layout == "auto":
        # Heuristic
        bname = source.bucket
        if bname == "poc-phase-1-test":
            return discover_poc_phase1(bname)
        elif bname.startswith("teams-faces-data-test"):
            return discover_teams_flat(bname)
        elif bname.startswith("live-deepfake-methods") and "teams" in bname:
            return discover_teams_paired(bname)
        else:
            # Try paired first (check for samples/ prefix)
            logger.warning("Auto-detecting layout for %s — trying teams_paired first", bname)
            return discover_teams_paired(bname)
    else:
        raise ValueError(f"Unknown layout: {layout}")


# ── GCS Dataset ───────────────────────────────────────────────────────

class GCSFrameDataset:
    """PyTorch-compatible dataset that downloads frames from GCS on the fly."""

    def __init__(self, records: List[FrameRecord], resolution: int = 224):
        self.records = records
        self.resolution = resolution
        cv2 = _import_cv2()
        torch = _import_torch()
        import torchvision.transforms as T
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])
        self._gcs_client = None

    def _get_client(self):
        if self._gcs_client is None:
            storage = _import_gcs()
            self._gcs_client = storage.Client()
        return self._gcs_client

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        torch = _import_torch()
        cv2 = _import_cv2()

        rec = self.records[idx]
        client = self._get_client()
        bucket = client.bucket(rec.bucket)
        blob = bucket.blob(rec.blob_path)

        try:
            data = blob.download_as_bytes()
            arr = np.frombuffer(data, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            img_bgr = None

        if img_bgr is None:
            logger.warning("Failed to decode: gs://%s/%s", rec.bucket, rec.blob_path)
            return torch.zeros(3, self.resolution, self.resolution), idx

        # INTER_LINEAR to match training preprocessing (combined_paired.py collate +
        # data/batching/*). INTER_AREA here caused silent train/eval drift in retro-scoring.
        img_bgr = cv2.resize(img_bgr, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb)
        return tensor, idx


# ── Model loading ─────────────────────────────────────────────────────

def _download_checkpoint(gcs_uri: str) -> str:
    """Download checkpoint from GCS to a temp file."""
    import tempfile
    storage = _import_gcs()

    if not gcs_uri.startswith("gs://"):
        return gcs_uri  # already local

    no_scheme = gcs_uri[5:]
    parts = no_scheme.split("/", 1)
    bucket_name, blob_name = parts[0], parts[1]

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not blob.exists(client=client):
        raise FileNotFoundError(f"Checkpoint not found: {gcs_uri}")

    local_dir = tempfile.mkdtemp(prefix="arena_ckpt_")
    local_path = os.path.join(local_dir, Path(blob_name).name)
    logger.info("Downloading checkpoint %s → %s", gcs_uri, local_path)
    blob.download_to_filename(local_path)
    return local_path


def load_model(checkpoint_path: str, detector_config: str, train_config: str, device):
    """Load Effort detector from checkpoint. Returns model."""
    torch = _import_torch()
    yaml = _import_yaml()
    from detectors import DETECTOR

    with open(detector_config, "r") as f:
        cfg = yaml.safe_load(f)
    with open(train_config, "r") as f:
        train_cfg = yaml.safe_load(f)
    cfg.update(train_cfg)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

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

    # Restore ArcFace scale
    if model_config.get("use_arcface_head", False) and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])

    clean_state = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    logger.info("Model loaded on %s", device)
    return model


# ── Inference runner ──────────────────────────────────────────────────

CSV_FIELDS = [
    "model", "bucket", "gcs_uri", "blob_path", "label", "label_str",
    "method", "video_id", "frame_name", "strategy", "prob_fake", "status",
]


def run_inference_for_checkpoint(
    ckpt_cfg: CheckpointConfig,
    records_by_source: Dict[str, List[FrameRecord]],
    cfg: ArenaConfig,
    run_dir: str,
) -> Dict[str, str]:
    """
    Run a single checkpoint against all data sources.
    Returns dict mapping source_name → CSV path.
    """
    torch = _import_torch()
    import torch.utils.data as tdata

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download and load model
    local_ckpt = _download_checkpoint(ckpt_cfg.path)
    model = load_model(local_ckpt, cfg.detector_config, cfg.train_config, device)

    inference_dir = os.path.join(run_dir, "inference")
    os.makedirs(inference_dir, exist_ok=True)

    embeddings_dir = os.path.join(run_dir, "embeddings")
    if cfg.save_embeddings:
        os.makedirs(embeddings_dir, exist_ok=True)

    csv_paths = {}

    for source_name, records in records_by_source.items():
        if not records:
            logger.warning("No records for %s — skipping", source_name)
            continue

        csv_path = os.path.join(inference_dir, f"{ckpt_cfg.name}__{source_name}.csv")
        emb_path = os.path.join(embeddings_dir, f"{ckpt_cfg.name}__{source_name}.npz") if cfg.save_embeddings else None

        # RESUME: skip if CSV already exists (from a previous run)
        if os.path.exists(csv_path):
            n_existing = sum(1 for _ in open(csv_path)) - 1  # minus header
            logger.info("SKIP %s/%s — CSV already exists (%d rows)", ckpt_cfg.name, source_name, n_existing)
            csv_paths[source_name] = csv_path
            continue

        logger.info("Scoring %s on %s (%d frames) ...", ckpt_cfg.name, source_name, len(records))
        t0 = time.time()

        dataset = GCSFrameDataset(records)
        loader = tdata.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            prefetch_factor=4 if cfg.num_workers > 0 else None,
        )

        results = [None] * len(records)
        all_probs = np.zeros(len(records), dtype=np.float32)
        all_embeddings = [] if cfg.save_embeddings else None
        emb_indices = [] if cfg.save_embeddings else None

        for batch_idx, (images, indices) in enumerate(loader):
            images = images.to(device, non_blocking=True)

            with torch.inference_mode():
                outputs = model({"image": images}, inference=True)
                probs = outputs["prob"].detach().cpu().numpy().reshape(-1)

                if cfg.save_embeddings and "feat" in outputs:
                    feats = outputs["feat"].detach().cpu().numpy()
                    all_embeddings.append(feats)
                    emb_indices.extend(indices.numpy().tolist())

            for i, global_idx in enumerate(indices.numpy()):
                rec = records[int(global_idx)]
                prob = float(probs[i])
                is_failed = bool(images[i].sum().item() == 0.0)
                all_probs[int(global_idx)] = prob

                results[int(global_idx)] = {
                    "model": ckpt_cfg.name,
                    "bucket": rec.bucket,
                    "gcs_uri": f"gs://{rec.bucket}/{rec.blob_path}",
                    "blob_path": rec.blob_path,
                    "label": rec.label,
                    "label_str": "fake" if rec.label == 1 else "real",
                    "method": rec.method,
                    "video_id": rec.video_id,
                    "frame_name": rec.frame_name,
                    "strategy": rec.strategy,
                    "prob_fake": f"{prob:.8f}",
                    "status": "failed_decode" if is_failed else "ok",
                }

            if (batch_idx + 1) % 50 == 0:
                logger.info("  %s/%s: batch %d/%d", ckpt_cfg.name, source_name,
                            batch_idx + 1, len(loader))

        # Write CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for r in results:
                if r is not None:
                    writer.writerow(r)

        csv_paths[source_name] = csv_path
        elapsed = time.time() - t0
        n_ok = sum(1 for r in results if r and r["status"] == "ok")
        logger.info("  %s/%s: %d ok, %d failed, %.1fs",
                     ckpt_cfg.name, source_name, n_ok, len(records) - n_ok, elapsed)

        # Save embeddings
        if cfg.save_embeddings and all_embeddings:
            all_emb = np.concatenate(all_embeddings, axis=0)
            emb_labels = np.array([records[idx].label for idx in emb_indices])
            emb_methods = np.array([records[idx].method for idx in emb_indices])
            np.savez_compressed(emb_path,
                                embeddings=all_emb,
                                labels=emb_labels,
                                methods=emb_methods,
                                indices=np.array(emb_indices))
            logger.info("  Saved embeddings to %s (%s)", emb_path, all_emb.shape)

    # Clean up downloaded checkpoint
    if local_ckpt != ckpt_cfg.path and os.path.exists(local_ckpt):
        os.remove(local_ckpt)

    return csv_paths


def _subsample_for_smoke_test(records: List[FrameRecord], max_per_label: int = 8) -> List[FrameRecord]:
    """
    Subsample records for smoke test: up to max_per_label per label (real/fake).
    Tries to pick from varied video_ids/methods.
    """
    import random
    real = [r for r in records if r.label == 0]
    fake = [r for r in records if r.label == 1]
    random.seed(42)
    random.shuffle(real)
    random.shuffle(fake)
    sampled = real[:max_per_label] + fake[:max_per_label]
    logger.info("  Smoke-test subsample: %d real, %d fake (from %d total)",
                min(len(real), max_per_label), min(len(fake), max_per_label), len(records))
    return sampled


def run_all_inference(cfg: ArenaConfig, run_dir: str, smoke_test: bool = False) -> Dict[str, Dict[str, str]]:
    """
    Run inference for all checkpoints × all data sources.
    Returns: {model_name: {source_name: csv_path}}
    """
    # Discover frames per source
    records_by_source: Dict[str, List[FrameRecord]] = {}
    for ds in cfg.data_sources:
        logger.info("Discovering frames in %s (%s) ...", ds.name, ds.bucket)
        records = discover_bucket(ds)
        if smoke_test:
            records = _subsample_for_smoke_test(records)
        records_by_source[ds.name] = records

    all_csv_paths: Dict[str, Dict[str, str]] = {}
    for i, ckpt in enumerate(cfg.checkpoints):
        logger.info("\n" + "=" * 80)
        logger.info("Checkpoint %d/%d: %s (%s)", i + 1, len(cfg.checkpoints), ckpt.name, ckpt.backbone)
        logger.info("=" * 80)
        csv_paths = run_inference_for_checkpoint(ckpt, records_by_source, cfg, run_dir)
        all_csv_paths[ckpt.name] = csv_paths

        # Sync to GCS after each checkpoint so results are visible while
        # the remaining checkpoints are still running.
        if cfg.output_gcs_folder and not smoke_test:
            timestamp = os.path.basename(run_dir)
            logger.info("Syncing %s results to GCS ...", ckpt.name)
            _upload_results_to_gcs(run_dir, cfg.output_gcs_folder, timestamp)

    return all_csv_paths


# ===========================================================================
# Phase 2: Analysis
# ===========================================================================

def load_inference_csv(path: str) -> List[Dict[str, Any]]:
    """Load an inference CSV into a list of dicts with numeric conversion."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["label"] = int(row["label"])
            row["prob_fake"] = float(row["prob_fake"])
            rows.append(row)
    return rows


def discover_inference_csvs(inference_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Discover inference CSVs from a directory.
    Expected naming: {model}__{source}.csv
    Returns: {model_name: {source_name: csv_path}}
    """
    result = defaultdict(dict)
    for fname in sorted(os.listdir(inference_dir)):
        if not fname.endswith(".csv"):
            continue
        stem = fname[:-4]
        if "__" not in stem:
            continue
        model, source = stem.split("__", 1)
        result[model][source] = os.path.join(inference_dir, fname)
    return dict(result)


# ── 2A. Probability Health ────────────────────────────────────────────

def compute_eer(labels: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    """Compute Equal Error Rate and the corresponding threshold."""
    # Sort by threshold
    thresholds = np.sort(np.unique(probs))
    # Subsample if too many unique values
    if len(thresholds) > 2000:
        thresholds = np.linspace(probs.min(), probs.max(), 2000)

    best_eer = 1.0
    best_thr = 0.5
    for t in thresholds:
        preds = (probs >= t).astype(int)
        fpr = ((preds == 1) & (labels == 0)).sum() / max((labels == 0).sum(), 1)
        fnr = ((preds == 0) & (labels == 1)).sum() / max((labels == 1).sum(), 1)
        eer = (fpr + fnr) / 2
        if abs(fpr - fnr) < abs(best_eer * 2 - (fpr + fnr)):
            best_eer = eer
            best_thr = t
    return best_eer, best_thr


def compute_auc_simple(labels: np.ndarray, probs: np.ndarray) -> float:
    """Simple AUC via trapezoidal rule (no sklearn dependency)."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    # Sort by decreasing score
    order = np.argsort(-probs)
    labels_sorted = labels[order]

    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tp = 0
    fp = 0
    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0

    for i in range(len(labels_sorted)):
        if labels_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / n_neg
        tpr = tp / n_pos
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr = fpr
        prev_tpr = tpr

    return auc


def analyze_prob_health(
    model_name: str,
    source_name: str,
    rows: List[Dict[str, Any]],
    healthy_range: Tuple[float, float],
) -> Dict[str, Any]:
    """Compute probability health metrics for one model×source."""
    ok_rows = [r for r in rows if r["status"] == "ok"]
    if not ok_rows:
        return {}

    probs = np.array([r["prob_fake"] for r in ok_rows])
    labels = np.array([r["label"] for r in ok_rows])

    real_probs = probs[labels == 0]
    fake_probs = probs[labels == 1]

    eer, eer_threshold = compute_eer(labels, probs)
    auc = compute_auc_simple(labels, probs)

    # Threshold health: how far from 0.5?
    threshold_deviation = abs(eer_threshold - 0.5)
    threshold_healthy = healthy_range[0] <= eer_threshold <= healthy_range[1]

    # Confidence metrics
    # For fakes: mean prob should be high; for reals: mean prob should be low
    fake_confidence = float(np.mean(fake_probs)) if len(fake_probs) > 0 else float("nan")
    real_confidence = 1.0 - float(np.mean(real_probs)) if len(real_probs) > 0 else float("nan")

    # Separability: Cohen's d between distributions
    if len(real_probs) > 1 and len(fake_probs) > 1:
        pooled_std = np.sqrt((np.var(real_probs) + np.var(fake_probs)) / 2)
        cohens_d = (np.mean(fake_probs) - np.mean(real_probs)) / max(pooled_std, 1e-8)
    else:
        cohens_d = float("nan")

    return {
        "model": model_name,
        "source": source_name,
        "n_frames": len(ok_rows),
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        "auc": auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "threshold_healthy": threshold_healthy,
        "threshold_deviation": threshold_deviation,
        "real_prob_mean": float(np.mean(real_probs)) if len(real_probs) > 0 else None,
        "real_prob_std": float(np.std(real_probs)) if len(real_probs) > 0 else None,
        "real_prob_median": float(np.median(real_probs)) if len(real_probs) > 0 else None,
        "fake_prob_mean": float(np.mean(fake_probs)) if len(fake_probs) > 0 else None,
        "fake_prob_std": float(np.std(fake_probs)) if len(fake_probs) > 0 else None,
        "fake_prob_median": float(np.median(fake_probs)) if len(fake_probs) > 0 else None,
        "fake_confidence": fake_confidence,
        "real_confidence": real_confidence,
        "cohens_d": cohens_d,
    }


# ── 2B. Stability Analysis ───────────────────────────────────────────

def analyze_stability(
    model_name: str,
    source_name: str,
    rows: List[Dict[str, Any]],
    thresholds: List[float],
) -> Dict[str, Any]:
    """
    Frame-to-frame stability analysis for chronologically ordered data.

    Computes jitter (consecutive frame prob changes) and flip rates
    (how often the binary decision changes between adjacent frames).
    Groups by video_id for proper sequence boundaries.
    """
    ok_rows = [r for r in rows if r["status"] == "ok"]
    if len(ok_rows) < 2:
        return {}

    # Group by video_id, sort by frame_name within each group
    videos = defaultdict(list)
    for r in ok_rows:
        videos[r["video_id"]].append(r)

    all_diffs = []
    flip_counts = {t: 0 for t in thresholds}
    pair_count = 0
    per_video_jitter = []

    for vid, frames in videos.items():
        if len(frames) < 2:
            continue
        # Sort by frame_name (assumes alphabetical = chronological)
        frames.sort(key=lambda r: r["frame_name"])
        probs = np.array([r["prob_fake"] for r in frames])

        diffs = np.abs(np.diff(probs))
        all_diffs.extend(diffs.tolist())
        pair_count += len(diffs)

        # Per-video jitter stats
        per_video_jitter.append({
            "video_id": vid,
            "n_frames": len(frames),
            "label": frames[0]["label"],
            "mean_jitter": float(np.mean(diffs)),
            "max_jitter": float(np.max(diffs)),
            "std_jitter": float(np.std(diffs)),
            "prob_mean": float(np.mean(probs)),
            "prob_std": float(np.std(probs)),
        })

        # Flip rates per threshold
        for t in thresholds:
            decisions = (probs >= t).astype(int)
            flips = np.sum(np.abs(np.diff(decisions)))
            flip_counts[t] += int(flips)

    if not all_diffs:
        return {}

    all_diffs_arr = np.array(all_diffs)

    flip_rates = {}
    for t in thresholds:
        flip_rates[f"flip_rate_T{t:.2f}"] = flip_counts[t] / max(pair_count, 1)

    return {
        "model": model_name,
        "source": source_name,
        "n_videos": len(per_video_jitter),
        "n_frame_pairs": pair_count,
        "mean_jitter": float(np.mean(all_diffs_arr)),
        "median_jitter": float(np.median(all_diffs_arr)),
        "max_jitter": float(np.max(all_diffs_arr)),
        "p95_jitter": float(np.percentile(all_diffs_arr, 95)),
        "p99_jitter": float(np.percentile(all_diffs_arr, 99)),
        "std_jitter": float(np.std(all_diffs_arr)),
        **flip_rates,
        "_per_video_jitter": per_video_jitter,
    }


# ── 2C. Per-Method Breakdown ─────────────────────────────────────────

def analyze_per_method(
    model_name: str,
    source_name: str,
    rows: List[Dict[str, Any]],
    target_methods: List[str],
) -> List[Dict[str, Any]]:
    """Per-method AUC, accuracy, TPR for fake methods."""
    ok_rows = [r for r in rows if r["status"] == "ok"]
    if not ok_rows:
        return []

    # Get all real frames for computing FPR baseline
    real_probs = np.array([r["prob_fake"] for r in ok_rows if r["label"] == 0])
    n_real = len(real_probs)

    # Group fake frames by method
    by_method = defaultdict(list)
    for r in ok_rows:
        if r["label"] == 1:
            by_method[r["method"]].append(r["prob_fake"])

    results = []
    for method in sorted(by_method.keys()):
        fake_probs = np.array(by_method[method])
        n_fake = len(fake_probs)

        # Compute metrics at multiple thresholds
        eer_threshold = 0.5  # default
        if n_real > 0 and n_fake > 0:
            all_probs = np.concatenate([real_probs, fake_probs])
            all_labels = np.concatenate([np.zeros(n_real), np.ones(n_fake)])
            _, eer_threshold = compute_eer(all_labels, all_probs)
            auc = compute_auc_simple(all_labels, all_probs)
        else:
            auc = float("nan")

        # TPR at various thresholds
        tpr_at = {}
        for t in [0.3, 0.5, eer_threshold, 0.7, 0.9]:
            tpr = float((fake_probs >= t).sum()) / max(n_fake, 1)
            tpr_at[f"tpr_at_{t:.2f}"] = tpr

        is_target = method in target_methods

        results.append({
            "model": model_name,
            "source": source_name,
            "method": method,
            "is_target": is_target,
            "n_frames": n_fake,
            "auc": auc,
            "mean_prob": float(np.mean(fake_probs)),
            "std_prob": float(np.std(fake_probs)),
            "eer_threshold": eer_threshold,
            **tpr_at,
        })

    return results


# ── 2D. Embedding Analysis ───────────────────────────────────────────

def analyze_embeddings(
    model_name: str,
    source_name: str,
    emb_path: str,
    cfg: ArenaConfig,
    output_dir: str,
) -> Dict[str, Any]:
    """T-SNE or UMAP on saved embeddings."""
    if not os.path.exists(emb_path):
        return {}

    data = np.load(emb_path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]
    methods = data["methods"]

    n_total = len(embeddings)
    if n_total == 0:
        return {}

    # Subsample if needed
    max_samples = cfg.embedding_max_samples
    if n_total > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(n_total, max_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
        methods = methods[indices]
        n_total = max_samples

    logger.info("Running %s on %d embeddings for %s/%s ...",
                cfg.embedding_method, n_total, model_name, source_name)

    if cfg.embedding_method == "tsne":
        from sklearn.manifold import TSNE
        # sklearn ≥1.6 renamed n_iter → max_iter
        import inspect
        tsne_params = inspect.signature(TSNE.__init__).parameters
        iter_key = "max_iter" if "max_iter" in tsne_params else "n_iter"
        reducer = TSNE(n_components=2, perplexity=cfg.embedding_perplexity,
                       random_state=42, **{iter_key: 1000})
    else:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42)

    coords = reducer.fit_transform(embeddings)

    # Save coordinates
    out_path = os.path.join(output_dir, f"{model_name}__{source_name}_{cfg.embedding_method}.npz")
    np.savez_compressed(out_path,
                        coords=coords, labels=labels, methods=methods)

    # Compute intra/inter-class distances
    real_emb = embeddings[labels == 0]
    fake_emb = embeddings[labels == 1]

    result = {
        "model": model_name,
        "source": source_name,
        "n_samples": n_total,
        "method": cfg.embedding_method,
        "coords_path": out_path,
    }

    if len(real_emb) > 1 and len(fake_emb) > 1:
        # Cosine distances (sample 500 pairs for speed)
        rng = np.random.RandomState(42)
        n_pairs = min(500, len(real_emb), len(fake_emb))

        # Real-real cosine distances
        idx1 = rng.choice(len(real_emb), n_pairs)
        idx2 = rng.choice(len(real_emb), n_pairs)
        rr_cos = np.array([
            np.dot(real_emb[i], real_emb[j]) / (np.linalg.norm(real_emb[i]) * np.linalg.norm(real_emb[j]) + 1e-8)
            for i, j in zip(idx1, idx2) if i != j
        ])

        # Fake-fake
        idx1 = rng.choice(len(fake_emb), n_pairs)
        idx2 = rng.choice(len(fake_emb), n_pairs)
        ff_cos = np.array([
            np.dot(fake_emb[i], fake_emb[j]) / (np.linalg.norm(fake_emb[i]) * np.linalg.norm(fake_emb[j]) + 1e-8)
            for i, j in zip(idx1, idx2) if i != j
        ])

        # Real-fake
        idx_r = rng.choice(len(real_emb), n_pairs)
        idx_f = rng.choice(len(fake_emb), n_pairs)
        rf_cos = np.array([
            np.dot(real_emb[i], fake_emb[j]) / (np.linalg.norm(real_emb[i]) * np.linalg.norm(fake_emb[j]) + 1e-8)
            for i, j in zip(idx_r, idx_f)
        ])

        result["cos_real_real_mean"] = float(np.mean(rr_cos)) if len(rr_cos) > 0 else None
        result["cos_fake_fake_mean"] = float(np.mean(ff_cos)) if len(ff_cos) > 0 else None
        result["cos_real_fake_mean"] = float(np.mean(rf_cos)) if len(rf_cos) > 0 else None

    return result


# ── Master analysis runner ────────────────────────────────────────────

def run_all_analysis(
    cfg: ArenaConfig,
    csv_paths: Dict[str, Dict[str, str]],
    run_dir: str,
) -> Dict[str, Any]:
    """
    Run all analysis phases on pre-existing inference CSVs.
    Returns a summary dict.
    """
    analysis_dir = os.path.join(run_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # Lookup chronological sources
    chrono_sources = {ds.name for ds in cfg.data_sources if ds.chronological}

    all_health = []
    all_stability = []
    all_per_method = []
    all_embeddings = []

    for model_name, sources in csv_paths.items():
        for source_name, csv_path in sources.items():
            logger.info("Analyzing %s / %s ...", model_name, source_name)
            rows = load_inference_csv(csv_path)

            # 2A: Probability health
            if cfg.prob_health_enabled:
                try:
                    health = analyze_prob_health(model_name, source_name, rows,
                                                 cfg.healthy_threshold_range)
                    if health:
                        all_health.append(health)
                except Exception as e:
                    logger.error("prob_health failed for %s/%s: %s", model_name, source_name, e)

            # 2B: Stability (only for chronological sources)
            if cfg.stability_enabled and source_name in chrono_sources:
                try:
                    stab = analyze_stability(model_name, source_name, rows,
                                             cfg.stability_thresholds)
                    if stab:
                        # Save per-video details separately
                        per_vid = stab.pop("_per_video_jitter", [])
                        all_stability.append(stab)

                        # Save per-video CSV
                        if per_vid:
                            vid_path = os.path.join(analysis_dir,
                                                    f"stability_per_video__{model_name}__{source_name}.csv")
                            _write_csv(vid_path, per_vid)
                except Exception as e:
                    logger.error("stability failed for %s/%s: %s", model_name, source_name, e)

            # 2C: Per-method breakdown
            if cfg.per_method_enabled:
                try:
                    pm = analyze_per_method(model_name, source_name, rows,
                                            cfg.target_methods)
                    all_per_method.extend(pm)
                except Exception as e:
                    logger.error("per_method failed for %s/%s: %s", model_name, source_name, e)

            # 2D: Embeddings
            if cfg.embeddings_enabled:
                try:
                    emb_dir = os.path.join(run_dir, "embeddings")
                    emb_path = os.path.join(emb_dir, f"{model_name}__{source_name}.npz")
                    if os.path.exists(emb_path):
                        emb_result = analyze_embeddings(model_name, source_name, emb_path,
                                                        cfg, analysis_dir)
                        if emb_result:
                            all_embeddings.append(emb_result)
                except Exception as e:
                    logger.error("embeddings failed for %s/%s: %s", model_name, source_name, e)

    # Write summary CSVs
    if all_health:
        _write_csv(os.path.join(analysis_dir, "prob_health.csv"), all_health)
    if all_stability:
        _write_csv(os.path.join(analysis_dir, "stability.csv"), all_stability)
    if all_per_method:
        _write_csv(os.path.join(analysis_dir, "per_method.csv"), all_per_method)
    if all_embeddings:
        emb_summary = [{k: v for k, v in e.items() if not k.startswith("_")} for e in all_embeddings]
        _write_csv(os.path.join(analysis_dir, "embeddings_summary.csv"), emb_summary)

    # Generate leaderboard
    leaderboard = _build_leaderboard(all_health, all_stability, all_per_method, cfg)
    if leaderboard:
        _write_csv(os.path.join(analysis_dir, "leaderboard.csv"), leaderboard)

    # Generate text report
    report_path = os.path.join(analysis_dir, "arena_report.txt")
    _write_report(report_path, all_health, all_stability, all_per_method,
                  all_embeddings, leaderboard, cfg)

    return {
        "n_health": len(all_health),
        "n_stability": len(all_stability),
        "n_per_method": len(all_per_method),
        "n_embeddings": len(all_embeddings),
        "leaderboard": leaderboard,
    }


def _write_csv(path: str, rows: List[Dict]):
    """Write list of dicts to CSV.  Uses the union of all row keys as fieldnames."""
    if not rows:
        return
    # Collect all keys across every row (preserving first-seen order)
    seen = set()
    fieldnames = []
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info("Wrote %s (%d rows)", path, len(rows))


def _build_leaderboard(
    health: List[Dict],
    stability: List[Dict],
    per_method: List[Dict],
    cfg: ArenaConfig,
) -> List[Dict]:
    """
    Build a leaderboard ranking models by composite score.

    Components:
      - AUC (higher = better)
      - EER (lower = better)
      - Threshold health (closer to 0.5 = better)
      - Stability / mean jitter (lower = better)
      - Target method mean TPR (higher = better)
    """
    models = sorted(set(h["model"] for h in health))
    if not models:
        return []

    leaderboard = []
    for model in models:
        model_health = [h for h in health if h["model"] == model]
        model_stab = [s for s in stability if s["model"] == model]
        model_pm = [p for p in per_method if p["model"] == model and p["is_target"]]

        # Average AUC across sources
        aucs = [h["auc"] for h in model_health if not math.isnan(h.get("auc", float("nan")))]
        avg_auc = float(np.mean(aucs)) if aucs else float("nan")

        # Average EER
        eers = [h["eer"] for h in model_health if not math.isnan(h.get("eer", float("nan")))]
        avg_eer = float(np.mean(eers)) if eers else float("nan")

        # Threshold health
        thresholds = [h["eer_threshold"] for h in model_health
                      if h.get("eer_threshold") is not None]
        avg_threshold = float(np.mean(thresholds)) if thresholds else float("nan")
        threshold_healthy = all(h.get("threshold_healthy", False) for h in model_health)

        # Mean jitter (lower = more stable)
        jitters = [s["mean_jitter"] for s in model_stab]
        avg_jitter = float(np.mean(jitters)) if jitters else float("nan")

        # Target method TPR at EER threshold
        target_tprs = [p.get(f"tpr_at_{p.get('eer_threshold', 0.5):.2f}", 0) for p in model_pm]
        # Simpler: TPR at 0.5
        target_tprs_50 = [p.get("tpr_at_0.50", 0) for p in model_pm]
        avg_target_tpr = float(np.mean(target_tprs_50)) if target_tprs_50 else float("nan")

        # Composite score: weighted combination
        # AUC contributes 40%, (1-EER) 20%, threshold_health 10%, (1-jitter) 15%, target_tpr 15%
        score_parts = []
        if not math.isnan(avg_auc):
            score_parts.append(("auc", 0.40, avg_auc))
        if not math.isnan(avg_eer):
            score_parts.append(("1-eer", 0.20, 1.0 - avg_eer))
        if not math.isnan(avg_threshold):
            # Health = 1 - 2*|threshold - 0.5|  (1.0 when threshold=0.5, 0.0 at extremes)
            score_parts.append(("thr_health", 0.10, 1.0 - 2 * abs(avg_threshold - 0.5)))
        if not math.isnan(avg_jitter):
            # Stability = 1 - min(jitter * 10, 1)  (1.0 when jitter=0, 0 when jitter>=0.1)
            score_parts.append(("stability", 0.15, max(0, 1.0 - avg_jitter * 10)))
        if not math.isnan(avg_target_tpr):
            score_parts.append(("target_tpr", 0.15, avg_target_tpr))

        if score_parts:
            total_weight = sum(w for _, w, _ in score_parts)
            composite = sum(w * v for _, w, v in score_parts) / total_weight
        else:
            composite = float("nan")

        leaderboard.append({
            "model": model,
            "composite_score": round(composite, 4) if not math.isnan(composite) else None,
            "avg_auc": round(avg_auc, 4) if not math.isnan(avg_auc) else None,
            "avg_eer": round(avg_eer, 4) if not math.isnan(avg_eer) else None,
            "avg_eer_threshold": round(avg_threshold, 3) if not math.isnan(avg_threshold) else None,
            "threshold_healthy": threshold_healthy,
            "avg_jitter": round(avg_jitter, 5) if not math.isnan(avg_jitter) else None,
            "avg_target_tpr": round(avg_target_tpr, 4) if not math.isnan(avg_target_tpr) else None,
            "n_sources_scored": len(model_health),
        })

    leaderboard.sort(key=lambda x: x.get("composite_score") or 0, reverse=True)
    return leaderboard


# ── Text Report ───────────────────────────────────────────────────────

def _write_report(
    path: str,
    health: List[Dict],
    stability: List[Dict],
    per_method: List[Dict],
    embeddings: List[Dict],
    leaderboard: List[Dict],
    cfg: ArenaConfig,
):
    """Write a comprehensive text report."""
    lines = []
    def pr(s=""):
        lines.append(s)

    pr("=" * 100)
    pr("MODEL ARENA — Cross-Checkpoint Comparison Report")
    pr(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pr(f"Checkpoints: {[c.name for c in cfg.checkpoints]}")
    pr(f"Data sources: {[d.name for d in cfg.data_sources]}")
    pr("=" * 100)

    # ── Leaderboard ──
    pr("\n" + "─" * 100)
    pr("LEADERBOARD (ranked by composite score)")
    pr("─" * 100)
    pr(f"{'Rank':>4} {'Model':<12} {'Score':>7} {'AUC':>7} {'EER':>7} {'Thr':>6} {'Thr OK':>6} "
       f"{'Jitter':>8} {'TgtTPR':>7}")
    pr("─" * 80)
    for i, lb in enumerate(leaderboard, 1):
        pr(f"{i:>4} {lb['model']:<12} "
           f"{lb['composite_score'] or '?':>7} "
           f"{lb['avg_auc'] or '?':>7} "
           f"{lb['avg_eer'] or '?':>7} "
           f"{lb['avg_eer_threshold'] or '?':>6} "
           f"{'✓' if lb['threshold_healthy'] else '✗':>6} "
           f"{lb['avg_jitter'] or '?':>8} "
           f"{lb['avg_target_tpr'] or '?':>7}")

    # ── Probability Health details ──
    pr("\n" + "─" * 100)
    pr("PROBABILITY HEALTH — per model × source")
    pr("─" * 100)
    pr(f"{'Model':<12} {'Source':<15} {'AUC':>6} {'EER':>6} {'Thr':>6} {'OK':>3} "
       f"{'RealMu':>7} {'RealSd':>7} {'FakeMu':>7} {'FakeSd':>7} {'Cohen-d':>8}")
    pr("─" * 100)
    for h in sorted(health, key=lambda x: (x["model"], x["source"])):
        thr_ok = "✓" if h.get("threshold_healthy") else "✗"
        pr(f"{h['model']:<12} {h['source']:<15} "
           f"{h['auc']:.4f} {h['eer']:.4f} {h['eer_threshold']:.3f} {thr_ok:>3} "
           f"{h.get('real_prob_mean', 0):.4f}  {h.get('real_prob_std', 0):.4f}  "
           f"{h.get('fake_prob_mean', 0):.4f}  {h.get('fake_prob_std', 0):.4f}  "
           f"{h.get('cohens_d', 0):.3f}")

    # ── Stability ──
    if stability:
        pr("\n" + "─" * 100)
        pr("STABILITY — Frame-to-frame jitter (lower = more stable)")
        pr("─" * 100)
        pr(f"{'Model':<12} {'Source':<15} {'MeanJit':>8} {'MedJit':>8} {'P95':>8} "
           f"{'P99':>8} {'MaxJit':>8} {'Pairs':>7}")
        pr("─" * 90)
        for s in sorted(stability, key=lambda x: (x["model"], x["source"])):
            pr(f"{s['model']:<12} {s['source']:<15} "
               f"{s['mean_jitter']:.5f} {s['median_jitter']:.5f} "
               f"{s['p95_jitter']:.5f} {s['p99_jitter']:.5f} "
               f"{s['max_jitter']:.5f} {s['n_frame_pairs']:>7}")

        # Flip rates
        thresholds = cfg.stability_thresholds
        pr(f"\n  Flip rates (fraction of adjacent pairs where binary decision changes):")
        header = f"{'Model':<12} {'Source':<15}"
        for t in thresholds:
            header += f" {'T=' + f'{t:.2f}':>8}"
        pr(header)
        pr("─" * (30 + 9 * len(thresholds)))
        for s in sorted(stability, key=lambda x: (x["model"], x["source"])):
            row = f"{s['model']:<12} {s['source']:<15}"
            for t in thresholds:
                rate = s.get(f"flip_rate_T{t:.2f}", 0)
                row += f" {rate:>8.4f}"
            pr(row)

    # ── Per-method breakdown ──
    if per_method:
        pr("\n" + "─" * 100)
        pr("PER-METHOD BREAKDOWN (target methods highlighted)")
        pr("─" * 100)
        # Group by source then model
        by_source = defaultdict(list)
        for pm in per_method:
            by_source[pm["source"]].append(pm)

        for source_name, pms in sorted(by_source.items()):
            pr(f"\n  Source: {source_name}")
            pr(f"  {'Model':<12} {'Method':<20} {'Tgt':>3} {'N':>6} {'AUC':>6} "
               f"{'MeanP':>6} {'TPR@0.3':>7} {'TPR@0.5':>7} {'TPR@0.7':>7}")
            pr(f"  {'─' * 90}")
            for pm in sorted(pms, key=lambda x: (x["model"], -x.get("is_target", 0), x["method"])):
                marker = "★" if pm["is_target"] else " "
                pr(f"  {pm['model']:<12} {pm['method']:<20} {marker:>3} "
                   f"{pm['n_frames']:>6} {pm.get('auc', 0):.4f} "
                   f"{pm.get('mean_prob', 0):.4f} "
                   f"{pm.get('tpr_at_0.30', 0):.4f}  "
                   f"{pm.get('tpr_at_0.50', 0):.4f}  "
                   f"{pm.get('tpr_at_0.70', 0):.4f}")

    # ── Embeddings ──
    if embeddings:
        pr("\n" + "─" * 100)
        pr("EMBEDDING ANALYSIS")
        pr("─" * 100)
        pr(f"{'Model':<12} {'Source':<15} {'N':>6} {'CosRR':>7} {'CosFF':>7} {'CosRF':>7}")
        pr("─" * 60)
        for e in embeddings:
            pr(f"{e['model']:<12} {e['source']:<15} {e['n_samples']:>6} "
               f"{e.get('cos_real_real_mean', '?'):>7} "
               f"{e.get('cos_fake_fake_mean', '?'):>7} "
               f"{e.get('cos_real_fake_mean', '?'):>7}")

    # ── Key Insights ──
    pr("\n" + "=" * 100)
    pr("KEY INSIGHTS")
    pr("=" * 100)

    if leaderboard:
        best = leaderboard[0]
        pr(f"\n  Overall winner: {best['model']} (composite score: {best['composite_score']})")

        # Threshold warnings
        for h in health:
            if not h.get("threshold_healthy"):
                pr(f"  ⚠ {h['model']} on {h['source']}: EER threshold={h['eer_threshold']:.3f} "
                   f"is OUTSIDE healthy range {cfg.healthy_threshold_range}")

        # Stability warnings
        for s in stability:
            if s["mean_jitter"] > 0.05:
                pr(f"  ⚠ {s['model']} on {s['source']}: HIGH jitter (mean={s['mean_jitter']:.4f}, "
                   f"p95={s['p95_jitter']:.4f})")

    pr("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Report written to %s (%d lines)", path, len(lines))


# ===========================================================================
# Phase 3: Strategy Search
# ===========================================================================

def build_windows(
    rows: List[Dict[str, Any]],
    source_name: str,
    W: int,
    seed: int = 42,
) -> List[Tuple[np.ndarray, int, str, str]]:
    """
    Build non-overlapping windows of W frames.
    Returns list of (probs_array, label, method, source).

    For poc-style data: windows within videos (non-overlapping).
    For flat/paired data: shuffled pool → non-overlapping chunks.
    """
    ok_rows = [r for r in rows if r["status"] == "ok"]
    if not ok_rows:
        return []

    windows = []
    rng = np.random.RandomState(seed)

    # Check if this source has natural video grouping
    has_videos = len(set(r["video_id"] for r in ok_rows)) > 1

    if has_videos and source_name == "poc_phase1":
        # Video-grouped windows
        by_video = defaultdict(list)
        for r in ok_rows:
            by_video[r["video_id"]].append(r)

        for vid, frames in by_video.items():
            frames.sort(key=lambda r: r["frame_name"])
            probs = np.array([r["prob_fake"] for r in frames])
            label = frames[0]["label"]
            method = frames[0]["method"]
            if len(probs) < W:
                continue
            for s in range(0, len(probs) - W + 1, W):
                windows.append((probs[s:s + W], label, method, source_name))
    else:
        # Pool-based windows (shuffle within label×method groups)
        for label_int in [0, 1]:
            by_method = defaultdict(list)
            for r in ok_rows:
                if r["label"] == label_int:
                    by_method[r["method"]].append(r["prob_fake"])

            for method, prob_list in by_method.items():
                probs = np.array(prob_list)
                rng.shuffle(probs)
                for s in range(0, len(probs) - W + 1, W):
                    windows.append((probs[s:s + W], label_int, method, source_name))

    return windows


def run_strategy_search(
    cfg: ArenaConfig,
    csv_paths: Dict[str, Dict[str, str]],
    run_dir: str,
) -> Dict[str, Any]:
    """
    Run window-based voting strategy search per model.
    Searches over (W, T, K/W) and uncertain margin.
    """
    strategy_dir = os.path.join(run_dir, "strategy")
    os.makedirs(strategy_dir, exist_ok=True)

    thresholds = np.round(np.arange(cfg.threshold_range[0],
                                     cfg.threshold_range[1] + cfg.threshold_step / 2,
                                     cfg.threshold_step), 3)
    k_ratios = np.round(np.arange(cfg.k_ratio_range[0],
                                    cfg.k_ratio_range[1] + cfg.k_ratio_step / 2,
                                    cfg.k_ratio_step), 3)

    # Identify "teams" sources for FPR constraint
    teams_source_names = set()
    for ds in cfg.data_sources:
        if "teams" in ds.bucket.lower():
            teams_source_names.add(ds.name)

    all_results = []
    best_per_model = []

    for model_name, sources in csv_paths.items():
        logger.info("Strategy search for %s ...", model_name)

        # Build windows across all sources for this model
        for W in cfg.window_sizes:
            all_windows = []
            for source_name, csv_path in sources.items():
                rows = load_inference_csv(csv_path)
                ws = build_windows(rows, source_name, W)
                all_windows.extend(ws)

            if not all_windows:
                continue

            N = len(all_windows)
            prob_matrix = np.zeros((N, W))
            labels = np.zeros(N, dtype=int)
            sources_arr = []
            methods_arr = []

            for i, (p, l, m, src) in enumerate(all_windows):
                prob_matrix[i] = p
                labels[i] = l
                sources_arr.append(src)
                methods_arr.append(m)

            sources_arr = np.array(sources_arr)
            methods_arr = np.array(methods_arr)
            is_real = (labels == 0)
            is_fake = (labels == 1)
            is_teams_real = is_real & np.isin(sources_arr, list(teams_source_names))
            n_teams_real = int(is_teams_real.sum())

            # Target method masks (case-insensitive)
            target_lower = {t.lower() for t in cfg.target_methods}
            is_target = np.array([m.lower() in target_lower for m in methods_arr])

            # Teams fake masks (for computing Teams TPR)
            is_teams_fake = is_fake & np.isin(sources_arr, list(teams_source_names))
            n_teams_fake = int(is_teams_fake.sum())

            for T in thresholds:
                votes = np.sum(prob_matrix >= T, axis=1)

                for kr in k_ratios:
                    K = max(1, int(round(kr * W)))

                    for margin in cfg.uncertain_margins:
                        K_high = K + margin
                        K_low = K - margin

                        preds = np.full(N, -1)  # -1 = uncertain
                        preds[votes >= K_high] = 1  # fake
                        preds[votes < K_low] = 0    # real
                        if margin == 0:
                            preds[votes >= K] = 1
                            preds[votes < K] = 0

                        decided = (preds >= 0)
                        uncertain = ~decided
                        n_uncertain = int(uncertain.sum())

                        # Overall metrics
                        tp = int(((preds == 1) & is_fake).sum())
                        tn = int(((preds == 0) & is_real).sum())
                        fp = int(((preds == 1) & is_real).sum())
                        fn = int(((preds == 0) & is_fake).sum())

                        n_f = int(is_fake.sum())
                        n_r = int(is_real.sum())
                        tpr = tp / max(n_f, 1)
                        tnr = tn / max(n_r, 1)
                        fpr = fp / max(n_r, 1)
                        bal_acc = (tpr + tnr) / 2

                        # Teams FP
                        teams_fp = int(((preds == 1) & is_teams_real).sum())
                        teams_fpr = teams_fp / max(n_teams_real, 1)

                        # Teams fake TPR (how well do we detect Teams fakes?)
                        teams_tp = int(((preds == 1) & is_teams_fake).sum())
                        teams_tpr = teams_tp / max(n_teams_fake, 1)

                        # Target method TPR
                        target_tp = int(((preds == 1) & is_target & is_fake).sum())
                        target_n = int((is_target & is_fake).sum())
                        target_tpr = target_tp / max(target_n, 1)

                        # Decided-only metrics (exclude uncertain windows)
                        n_decided_fake = int((decided & is_fake).sum())
                        n_decided_real = int((decided & is_real).sum())
                        tpr_decided = tp / max(n_decided_fake, 1)
                        tnr_decided = tn / max(n_decided_real, 1)
                        bal_acc_decided = (tpr_decided + tnr_decided) / 2 if (n_decided_fake + n_decided_real) > 0 else 0.0

                        # Threshold health flag
                        thr_healthy = cfg.strategy_healthy_range[0] <= T <= cfg.strategy_healthy_range[1]

                        row = {
                            "model": model_name,
                            "W": W,
                            "T": T,
                            "K": K,
                            "K_ratio": kr,
                            "margin": margin,
                            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                            "tpr": round(tpr, 4),
                            "tnr": round(tnr, 4),
                            "fpr": round(fpr, 4),
                            "bal_acc": round(bal_acc, 4),
                            "teams_fp": teams_fp,
                            "teams_fpr": round(teams_fpr, 4),
                            "teams_tpr": round(teams_tpr, 4),
                            "target_tpr": round(target_tpr, 4),
                            "bal_acc_decided": round(bal_acc_decided, 4),
                            "n_uncertain": n_uncertain,
                            "uncertain_pct": round(n_uncertain / max(N, 1), 4),
                            "n_total": N,
                            "n_fake": n_f,
                            "n_real": n_r,
                            "n_teams_real": n_teams_real,
                            "n_teams_fake": n_teams_fake,
                            "thr_healthy": thr_healthy,
                        }
                        all_results.append(row)

            logger.info("  %s W=%d: %d windows, %d combos so far",
                        model_name, W, N, len(all_results))

        # Find best strategy per model (0 Teams FP, binary, healthy threshold)
        model_rows = [r for r in all_results if r["model"] == model_name]
        zero_teams = [r for r in model_rows if r["teams_fp"] == 0 and r["margin"] == 0]
        if zero_teams:
            healthy = [r for r in zero_teams if r["thr_healthy"]]
            pool = healthy if healthy else zero_teams
            best = max(pool, key=lambda r: (r["bal_acc"], r["target_tpr"], -r["fpr"]))
            best_per_model.append({
                "model": model_name,
                "best_W": best["W"],
                "best_T": best["T"],
                "best_K": best["K"],
                "best_bal_acc": best["bal_acc"],
                "best_target_tpr": best["target_tpr"],
                "best_teams_tpr": best["teams_tpr"],
                "best_teams_fp": best["teams_fp"],
                "best_thr_healthy": best["thr_healthy"],
                "best_fpr": best["fpr"],
            })

    # ── Best strategy WITH uncertain zone (up to 3% uncertain) ──
    MAX_UNCERTAIN_PCT = 0.03
    best_per_model_uncertain = []
    for model_name in csv_paths.keys():
        model_rows = [r for r in all_results if r["model"] == model_name]
        # Allow any margin, but uncertain_pct <= 3%, and teams_fp == 0
        eligible = [r for r in model_rows
                    if r["teams_fp"] == 0
                    and r["uncertain_pct"] <= MAX_UNCERTAIN_PCT]
        if eligible:
            healthy = [r for r in eligible if r["thr_healthy"]]
            pool = healthy if healthy else eligible
            best = max(pool, key=lambda r: (r["bal_acc_decided"], r["teams_tpr"], -r["fpr"]))
            best_per_model_uncertain.append({
                "model": model_name,
                "best_W": best["W"],
                "best_T": best["T"],
                "best_K": best["K"],
                "best_margin": best["margin"],
                "best_bal_acc_decided": best["bal_acc_decided"],
                "best_bal_acc": best["bal_acc"],
                "best_target_tpr": best["target_tpr"],
                "best_teams_tpr": best["teams_tpr"],
                "best_teams_fp": best["teams_fp"],
                "best_thr_healthy": best["thr_healthy"],
                "best_fpr": best["fpr"],
                "best_uncertain_pct": best["uncertain_pct"],
                "best_n_uncertain": best["n_uncertain"],
            })

    # Write results
    if all_results:
        _write_csv(os.path.join(strategy_dir, "strategy_grid_all.csv"), all_results)
    if best_per_model:
        _write_csv(os.path.join(strategy_dir, "best_strategy_per_model.csv"), best_per_model)
    if best_per_model_uncertain:
        _write_csv(os.path.join(strategy_dir, "best_strategy_uncertain.csv"), best_per_model_uncertain)

    # Write strategy report
    _write_strategy_report(
        os.path.join(strategy_dir, "strategy_report.txt"),
        all_results, best_per_model, best_per_model_uncertain, cfg)

    return {
        "n_combos": len(all_results),
        "best_per_model": best_per_model,
        "best_per_model_uncertain": best_per_model_uncertain,
    }


def _write_strategy_report(
    path: str,
    all_results: List[Dict],
    best_per_model: List[Dict],
    best_per_model_uncertain: List[Dict],
    cfg: ArenaConfig,
):
    """Write strategy search text report."""
    lines = []
    def pr(s=""):
        lines.append(s)

    pr("=" * 100)
    pr("STRATEGY SEARCH REPORT")
    pr(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pr(f"Total combos evaluated: {len(all_results):,}")
    pr("=" * 100)

    # ── Best per model (0 Teams FP) ──
    pr("\n" + "─" * 100)
    pr("BEST STRATEGY PER MODEL (Teams FP = 0, binary, healthy threshold preferred)")
    pr("─" * 100)
    pr(f"{'Model':<12} {'W':>3} {'T':>6} {'K':>3} {'BalAcc':>7} {'TgtTPR':>7} "
       f"{'TeamsTPR':>9} {'FPR':>6} {'ThrOK':>5}")
    pr("─" * 70)
    for b in best_per_model:
        ok = "✓" if b["best_thr_healthy"] else "✗"
        pr(f"{b['model']:<12} {b['best_W']:>3} {b['best_T']:>6.2f} {b['best_K']:>3} "
           f"{b['best_bal_acc']:>7.4f} {b['best_target_tpr']:>7.4f} "
           f"{b.get('best_teams_tpr', 0):>9.4f} "
           f"{b['best_fpr']:>6.4f} {ok:>5}")

    # ── Best per model WITH uncertain zone (up to 3%) ──
    if best_per_model_uncertain:
        pr("\n" + "─" * 100)
        pr("BEST STRATEGY WITH UNCERTAIN ZONE (≤3% uncertain excluded, Teams FP = 0)")
        pr("─" * 100)
        pr(f"{'Model':<12} {'W':>3} {'T':>6} {'K':>3} {'M':>2} "
           f"{'AccDecid':>8} {'BalAcc':>7} {'TgtTPR':>7} {'TeamsTPR':>9} "
           f"{'Unc%':>5} {'FPR':>6} {'ThrOK':>5}")
        pr("─" * 85)
        for b in best_per_model_uncertain:
            ok = "✓" if b["best_thr_healthy"] else "✗"
            pr(f"{b['model']:<12} {b['best_W']:>3} {b['best_T']:>6.2f} {b['best_K']:>3} "
               f"{b['best_margin']:>2} "
               f"{b['best_bal_acc_decided']:>8.4f} {b['best_bal_acc']:>7.4f} "
               f"{b['best_target_tpr']:>7.4f} {b['best_teams_tpr']:>9.4f} "
               f"{b['best_uncertain_pct'] * 100:>5.1f} "
               f"{b['best_fpr']:>6.4f} {ok:>5}")

        # Show improvement over binary
        pr("\n  Improvement over binary (AccDecided vs BalAcc binary):")
        binary_lookup = {b["model"]: b for b in best_per_model}
        for b in best_per_model_uncertain:
            binary_ba = binary_lookup.get(b["model"], {}).get("best_bal_acc", 0)
            delta = b["best_bal_acc_decided"] - binary_ba
            pr(f"    {b['model']:<12} binary={binary_ba:.4f} → decided={b['best_bal_acc_decided']:.4f} "
               f"Δ={delta:+.4f} (margin={b['best_margin']}, {b['best_uncertain_pct']*100:.1f}% uncertain)")

    # ── Cross-model comparison at each model's best strategy ──
    pr("\n" + "─" * 100)
    pr("CROSS-MODEL: How does each model perform at OTHER models' best strategies?")
    pr("─" * 100)

    models = sorted(set(r["model"] for r in all_results))
    strategies = [(b["model"], b["best_W"], b["best_T"], b["best_K"]) for b in best_per_model]

    header = f"{'StrategyOf':<12} {'W':>3} {'T':>5} {'K':>3} |"
    for m in models:
        header += f" {m:>12}"
    pr(header)
    pr("─" * (28 + 13 * len(models)))

    for strat_model, W, T, K in strategies:
        row = f"{strat_model:<12} {W:>3} {T:>5.2f} {K:>3} |"
        for m in models:
            # Find this model's result at this strategy
            match = [r for r in all_results
                     if r["model"] == m and r["W"] == W
                     and abs(r["T"] - T) < 0.001 and r["K"] == K
                     and r["margin"] == 0]
            if match:
                r = match[0]
                row += f" {r['bal_acc']:.4f}({r['teams_fp']})"
            else:
                row += f" {'—':>12}"
        pr(row)

    # ── Pareto frontier per model ──
    pr("\n" + "─" * 100)
    pr("PARETO FRONTIER: Teams FP count vs Balanced Accuracy (binary, best per FP level)")
    pr("─" * 100)

    for model in models:
        model_rows = [r for r in all_results
                      if r["model"] == model and r["margin"] == 0]
        if not model_rows:
            continue

        pr(f"\n  {model}:")
        pr(f"  {'TeamsFP':>7} {'BalAcc':>7} {'TgtTPR':>7} {'TeamsTPR':>9} {'T':>6} {'K':>3} {'W':>3} {'ThrOK':>5}")
        pr(f"  {'─' * 62}")

        seen_fp = set()
        model_rows.sort(key=lambda r: (r["teams_fp"], -r["bal_acc"]))
        best_at_fp = {}
        for r in model_rows:
            fp_key = r["teams_fp"]
            if fp_key not in best_at_fp or r["bal_acc"] > best_at_fp[fp_key]["bal_acc"]:
                best_at_fp[fp_key] = r

        for fp_level in sorted(best_at_fp.keys())[:15]:
            r = best_at_fp[fp_level]
            ok = "✓" if r["thr_healthy"] else "✗"
            pr(f"  {fp_level:>7} {r['bal_acc']:>7.4f} {r['target_tpr']:>7.4f} "
               f"{r['teams_tpr']:>9.4f} "
               f"{r['T']:>6.2f} {r['K']:>3} {r['W']:>3} {ok:>5}")

    # ── Uncertain zone Pareto frontier ──
    pr("\n" + "─" * 100)
    pr("UNCERTAIN ZONE FRONTIER: Best AccDecided by uncertain % (Teams FP = 0)")
    pr("─" * 100)

    for model in models:
        model_rows = [r for r in all_results
                      if r["model"] == model and r["teams_fp"] == 0]
        if not model_rows:
            continue

        pr(f"\n  {model}:")
        pr(f"  {'Unc%':>5} {'AccDec':>7} {'BalAcc':>7} {'TeamsTPR':>9} "
           f"{'T':>6} {'K':>3} {'M':>2} {'W':>3} {'ThrOK':>5}")
        pr(f"  {'─' * 65}")

        # Best at each uncertain percentage bucket (0%, 0.5%, 1%, 1.5%, 2%, 2.5%, 3%)
        for pct_target in [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]:
            eligible = [r for r in model_rows if r["uncertain_pct"] <= pct_target + 0.001]
            if not eligible:
                continue
            healthy = [r for r in eligible if r["thr_healthy"]]
            pool = healthy if healthy else eligible
            best = max(pool, key=lambda r: (r["bal_acc_decided"], r["teams_tpr"], -r["fpr"]))
            ok = "✓" if best["thr_healthy"] else "✗"
            pr(f"  {pct_target * 100:>5.1f} {best['bal_acc_decided']:>7.4f} {best['bal_acc']:>7.4f} "
               f"{best['teams_tpr']:>9.4f} "
               f"{best['T']:>6.2f} {best['K']:>3} {best['margin']:>2} {best['W']:>3} {ok:>5}")

    # ── Health warnings ──
    pr("\n" + "─" * 100)
    pr("THRESHOLD HEALTH WARNINGS")
    pr("─" * 100)
    unhealthy = [b for b in best_per_model if not b["best_thr_healthy"]]
    if unhealthy:
        for b in unhealthy:
            pr(f"  ⚠ {b['model']}: best strategy has T={b['best_T']:.2f} which is OUTSIDE "
               f"healthy range {cfg.strategy_healthy_range}")
    else:
        pr("  All models' best strategies have healthy thresholds ✓")

    pr("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Strategy report written to %s (%d lines)", path, len(lines))


# ===========================================================================
# Main
# ===========================================================================

def print_plan(cfg: ArenaConfig):
    """Print the execution plan without running anything."""
    print("=" * 80)
    print("MODEL ARENA — Execution Plan")
    print("=" * 80)

    print(f"\nCheckpoints ({len(cfg.checkpoints)}):")
    for c in cfg.checkpoints:
        print(f"  {c.name:<12} {c.backbone}  {c.path}")
        if c.notes:
            print(f"  {'':12} └─ {c.notes}")

    print(f"\nData Sources ({len(cfg.data_sources)}):")
    for d in cfg.data_sources:
        chrono = " [chronological]" if d.chronological else ""
        print(f"  {d.name:<15} {d.bucket}{chrono}")
        if d.notes:
            print(f"  {'':15} └─ {d.notes}")

    n_jobs = len(cfg.checkpoints) * len(cfg.data_sources)
    print(f"\nInference jobs: {n_jobs} ({len(cfg.checkpoints)} checkpoints × {len(cfg.data_sources)} sources)")
    print(f"Embeddings: {'yes' if cfg.save_embeddings else 'no'}")

    print(f"\nAnalysis:")
    print(f"  Probability health:  {'✓' if cfg.prob_health_enabled else '✗'}")
    print(f"  Stability:           {'✓' if cfg.stability_enabled else '✗'}")
    print(f"  Per-method:          {'✓' if cfg.per_method_enabled else '✗'}")
    print(f"  Embeddings:          {'✓' if cfg.embeddings_enabled else '✗'} ({cfg.embedding_method})")

    print(f"\nStrategy search:")
    print(f"  Enabled:             {'✓' if cfg.strategy_enabled else '✗'}")
    print(f"  Window sizes:        {cfg.window_sizes}")
    n_t = len(np.arange(cfg.threshold_range[0], cfg.threshold_range[1], cfg.threshold_step))
    n_k = len(np.arange(cfg.k_ratio_range[0], cfg.k_ratio_range[1], cfg.k_ratio_step))
    n_m = len(cfg.uncertain_margins)
    n_combos = n_t * n_k * n_m * len(cfg.window_sizes) * len(cfg.checkpoints)
    print(f"  Search space:        ~{n_combos:,} combos per model")
    print(f"  Teams FPR cap:       {cfg.max_teams_fpr}")
    print(f"  Healthy T range:     {cfg.strategy_healthy_range}")

    print(f"\nOutput: {{output_dir}}/{{timestamp}}/")
    print(f"  inference/   — per-frame CSVs ({n_jobs} files)")
    print(f"  embeddings/  — .npz files for T-SNE")
    print(f"  analysis/    — health, stability, per-method CSVs + report")
    print(f"  strategy/    — grid search CSVs + report")


def main():
    parser = argparse.ArgumentParser(
        description="Model Arena — Cross-checkpoint comparison on target-domain data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="arena_config.yaml",
                        help="Path to arena config YAML")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "inference", "analysis", "strategy"],
                        help="Which phase to run (default: all)")
    parser.add_argument("--inference-dir", type=str, default=None,
                        help="Path to pre-existing inference dir (skip inference phase)")
    parser.add_argument("--resume", type=str, default=None, metavar="RUN_ID",
                        help="Resume a previous run by timestamp (e.g. 20260310_140000). "
                             "Downloads existing results from GCS and skips completed checkpoint×source combos.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Smoke test: load all models, score ~16 frames/source, save all outputs")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None,
                        metavar="NAME",
                        help="Only evaluate these checkpoint names (e.g. --checkpoints R13_FT2 R12_G). "
                             "Default: all checkpoints in config.")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    cfg = load_config(args.config)

    # Filter checkpoints if --checkpoints was given
    if args.checkpoints:
        requested = set(args.checkpoints)
        available = {c.name for c in cfg.checkpoints}
        unknown = requested - available
        if unknown:
            logger.error("Unknown checkpoint(s): %s  (available: %s)",
                         ", ".join(sorted(unknown)), ", ".join(sorted(available)))
            return
        cfg.checkpoints = [c for c in cfg.checkpoints if c.name in requested]
        logger.info("Filtered to %d checkpoint(s): %s",
                    len(cfg.checkpoints), ", ".join(c.name for c in cfg.checkpoints))

    if args.dry_run:
        print_plan(cfg)
        return

    smoke_test = args.smoke_test
    if smoke_test:
        logger.info("\n🔥 SMOKE TEST MODE — limited frames, condensed search")
        # Override settings for speed
        cfg.batch_size = 8
        cfg.num_workers = 0
        cfg.window_sizes = [16]
        cfg.threshold_range = (0.3, 0.7)
        cfg.threshold_step = 0.1
        cfg.k_ratio_range = (0.4, 0.6)
        cfg.k_ratio_step = 0.1
        cfg.uncertain_margins = [0]
        cfg.embedding_max_samples = 32

    # Create run directory (or resume an existing one)
    import shutil

    if args.resume:
        timestamp = args.resume
        run_dir = os.path.join(cfg.output_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        logger.info("RESUMING run %s", timestamp)
        # Download whatever survived from GCS
        if cfg.output_gcs_folder:
            _restore_from_gcs(run_dir, cfg.output_gcs_folder, timestamp)
    else:
        tag = "smoke_test" if smoke_test else ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if tag:
            timestamp = f"{timestamp}_{tag}"
        run_dir = os.path.join(cfg.output_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)

    logger.info("Run directory: %s", run_dir)

    # Save config copy
    shutil.copy2(args.config, os.path.join(run_dir, "arena_config.yaml"))

    # Helper: incremental GCS sync after each phase to avoid data loss
    def _sync_to_gcs(phase_label: str):
        if cfg.output_gcs_folder and not smoke_test:
            logger.info("Syncing results to GCS after %s ...", phase_label)
            _upload_results_to_gcs(run_dir, cfg.output_gcs_folder, timestamp)

    # Phase routing
    csv_paths = None

    if args.phase in ("all", "inference"):
        if args.inference_dir:
            logger.info("Using pre-existing inference dir: %s", args.inference_dir)
            csv_paths = discover_inference_csvs(args.inference_dir)
        else:
            csv_paths = run_all_inference(cfg, run_dir, smoke_test=smoke_test)
        _sync_to_gcs("inference")  # save CSVs + embeddings immediately

    if args.phase in ("all", "analysis"):
        if csv_paths is None:
            inference_dir = args.inference_dir or os.path.join(run_dir, "inference")
            csv_paths = discover_inference_csvs(inference_dir)
        analysis_results = run_all_analysis(cfg, csv_paths, run_dir)
        logger.info("Analysis complete: %s", analysis_results)
        _sync_to_gcs("analysis")  # save analysis reports

    if args.phase in ("all", "strategy"):
        if csv_paths is None:
            inference_dir = args.inference_dir or os.path.join(run_dir, "inference")
            csv_paths = discover_inference_csvs(inference_dir)
        strategy_results = run_strategy_search(cfg, csv_paths, run_dir)
        logger.info("Strategy search complete: %d combos, %d models",
                     strategy_results["n_combos"], len(strategy_results["best_per_model"]))
        _sync_to_gcs("strategy")  # final sync

    if smoke_test:
        logger.info("\n" + "=" * 80)
        logger.info("🔥 SMOKE TEST PASSED — all phases completed successfully")
        logger.info("=" * 80)
        # List what was produced
        for dirpath, dirnames, filenames in os.walk(run_dir):
            for fn in sorted(filenames):
                fpath = os.path.join(dirpath, fn)
                size = os.path.getsize(fpath)
                rel = os.path.relpath(fpath, run_dir)
                logger.info("  %-60s %s", rel, f"{size:,} bytes")

    logger.info("\nDone! Results in %s", run_dir)


def _restore_from_gcs(run_dir: str, gcs_folder: str, timestamp: str):
    """Download existing results from a previous GCS sync into local run_dir."""
    try:
        storage = _import_gcs()
        uri = gcs_folder.rstrip("/") + "/" + timestamp
        no_scheme = uri[5:]
        bucket_name, prefix = no_scheme.split("/", 1)

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix + "/"))

        if not blobs:
            logger.info("No previous results found in gs://%s/%s/", bucket_name, prefix)
            return

        count = 0
        for blob in blobs:
            rel_path = blob.name[len(prefix) + 1:]  # strip prefix/
            if not rel_path:
                continue
            local_path = os.path.join(run_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            count += 1

        logger.info("Restored %d files from gs://%s/%s/", count, bucket_name, prefix)
    except Exception as e:
        logger.warning("GCS restore failed (will run from scratch): %s", e)


def _upload_results_to_gcs(run_dir: str, gcs_folder: str, timestamp: str):
    """Upload all results to GCS."""
    try:
        storage = _import_gcs()
        uri = gcs_folder.rstrip("/") + "/" + timestamp
        no_scheme = uri[5:]
        bucket_name, prefix = no_scheme.split("/", 1)

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        count = 0
        for root, dirs, files in os.walk(run_dir):
            for fname in files:
                local_path = os.path.join(root, fname)
                rel_path = os.path.relpath(local_path, run_dir)
                blob_path = f"{prefix}/{rel_path}"
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(local_path)
                count += 1

        logger.info("Uploaded %d files to gs://%s/%s/", count, bucket_name, prefix)
    except Exception as e:
        logger.warning("GCS upload failed: %s", e)


if __name__ == "__main__":
    main()
