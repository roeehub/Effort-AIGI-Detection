#!/usr/bin/env python3
"""
Batch inference on GCS test buckets — outputs per-frame CSV with probabilities.

Supports three bucket layouts:
  1) poc-phase-1-test:
       {fake,real}/{method_videoId}/NNNN.png
  2) teams-faces-data-test-*:
       {fake,real}/flat_image_files.jpg
  3) live-deepfake-methods-*-teams:
       samples/{strategy_NNNN}/frames/{fake,real}/frame_NNNN.jpg
       (with manifest.json per sample)

Usage (local):
  python batch_inference_gcs.py \
    --checkpoint gs://training-job-outputs/phase2r9_experiments/1551zxa8/top_n_effort_20260228_step6000_auc0.9891_eer0.0457.pth \
    --buckets poc-phase-1-test \
    --output_dir ./inference_results

Usage (Vertex AI — all 3 buckets):
  python batch_inference_gcs.py \
    --checkpoint gs://training-job-outputs/phase2r9_experiments/1551zxa8/top_n_effort_20260228_step6000_auc0.9891_eer0.0457.pth \
    --buckets poc-phase-1-test teams-faces-data-test-2914-fake-4420-real-feb-28 live-deepfake-methods-real-and-fake-frames-cropped-teams \
    --output_dir /gcs/training-job-outputs/batch_inference_results \
    --batch_size 128 --num_workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.utils.data as data
import yaml
from google.cloud import storage
import torchvision.transforms as T

# Project imports
from detectors import DETECTOR

logger = logging.getLogger("batch-inference-gcs")

# CLIP normalization constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


# =============================================================================
# Data structures
# =============================================================================
@dataclass
class FrameRecord:
    """Metadata for a single frame to score."""
    bucket: str
    blob_path: str          # full blob path in the bucket
    label: int              # 0=real, 1=fake
    method: str             # deepfake method or "real"
    video_id: str           # video/sample identifier
    frame_name: str         # e.g. "0001.png"
    strategy: str = ""      # for teams-paired: edge_cases, minimal_processing, etc.
    extra: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# GCS bucket discovery — build frame manifest from bucket contents
# =============================================================================
def _parse_gs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    no_scheme = uri[5:]
    parts = no_scheme.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def _download_checkpoint(gcs_uri: str) -> str:
    """Download checkpoint from GCS to a temp file. Returns local path."""
    bucket_name, blob_name = _parse_gs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not blob.exists(client=client):
        raise FileNotFoundError(f"Checkpoint not found: {gcs_uri}")

    local_dir = tempfile.mkdtemp(prefix="batch_infer_ckpt_")
    local_path = os.path.join(local_dir, Path(blob_name).name)
    logger.info("Downloading checkpoint %s → %s", gcs_uri, local_path)
    blob.download_to_filename(local_path)
    return local_path


def discover_poc_phase1(client: storage.Client, bucket_name: str) -> List[FrameRecord]:
    """
    poc-phase-1-test layout:
      fake/{Method}_{videoId}/NNNN.png
      real/{videoId}/NNNN.png
    """
    records = []
    bucket = client.bucket(bucket_name)
    image_exts = {".png", ".jpg", ".jpeg"}

    for label_str, label_int in [("fake", 1), ("real", 0)]:
        prefix = f"{label_str}/"
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            ext = os.path.splitext(blob.name)[1].lower()
            if ext not in image_exts:
                continue

            # e.g. fake/Deeplivecam_deepfake_-1muVDMvY4c/0001.png
            rel = blob.name[len(prefix):]  # Deeplivecam_deepfake_-1muVDMvY4c/0001.png
            parts = rel.split("/")
            if len(parts) < 2:
                continue

            folder_name = parts[0]
            frame_name = parts[-1]

            if label_int == 1:
                # Extract method: everything before the first underscore-separated video ID
                # Folder patterns: Method_detail_videoId  or  Method_NNN_NNN
                # Use heuristic: split by '_' and group
                method = _extract_method_from_folder(folder_name)
                video_id = folder_name
            else:
                method = "real"
                video_id = folder_name

            records.append(FrameRecord(
                bucket=bucket_name,
                blob_path=blob.name,
                label=label_int,
                method=method,
                video_id=video_id,
                frame_name=frame_name,
            ))

    logger.info("[%s] Discovered %d frames", bucket_name, len(records))
    return records


def _extract_method_from_folder(folder_name: str) -> str:
    """
    Extract the deepfake method name from a video folder name.
    
    Examples:
      Deeplivecam_deepfake_-1muVDMvY4c → Deeplivecam
      Face2Face_070_057 → Face2Face
      simswap_sequence_rtrn03_... → simswap
      inswap_-1muVDMvY4c → inswap
    """
    known_methods = [
        "Deeplivecam", "Face2Face", "FaceShifter", "FaceSwap",
        "MRAA", "NeuralTextures", "facevid2vid", "fomm", "fsgan",
        "hyperreenact", "inswap", "lia", "mcnet", "mobileswap",
        "oneshot", "pirender", "sadtalker", "simswap", "tpsm",
    ]
    for m in known_methods:
        if folder_name.startswith(m + "_") or folder_name == m:
            return m
    # Fallback: take the first token
    return folder_name.split("_")[0]


def discover_teams_flat(client: storage.Client, bucket_name: str) -> List[FrameRecord]:
    """
    teams-faces-data-test-* layout:
      fake/Cam_Test__s32_N.N_frame_NNNNNN_crop_NNN__HASH.jpg
      real/Cam_Test__s32_N.N_frame_NNNNNN_crop_NNN__HASH.jpg
    
    The 'video' concept here is the segment number (N.N part).
    """
    records = []
    bucket = client.bucket(bucket_name)
    image_exts = {".png", ".jpg", ".jpeg"}

    for label_str, label_int in [("fake", 1), ("real", 0)]:
        prefix = f"{label_str}/"
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            ext = os.path.splitext(blob.name)[1].lower()
            if ext not in image_exts:
                continue

            frame_name = os.path.basename(blob.name)
            # Extract video/segment ID from filename pattern:
            # Cam_Test__s32_N.N_frame_NNNNNN_crop_NNN__HASH.jpg
            # The segment is the float number after 's32_'
            video_id = _extract_teams_segment(frame_name)

            records.append(FrameRecord(
                bucket=bucket_name,
                blob_path=blob.name,
                label=label_int,
                method="teams_passthrough" if label_int == 1 else "real",
                video_id=video_id,
                frame_name=frame_name,
            ))

    logger.info("[%s] Discovered %d frames", bucket_name, len(records))
    return records


def _extract_teams_segment(filename: str) -> str:
    """
    Extract segment ID from teams filename.
    Cam_Test__s32_103.0_frame_003465_crop_001__a4fa23f9.jpg → seg_103.0
    """
    try:
        # Find the pattern after 's32_' and before '_frame_'
        idx_s32 = filename.index("s32_") + 4
        idx_frame = filename.index("_frame_")
        seg = filename[idx_s32:idx_frame]
        return f"seg_{seg}"
    except (ValueError, IndexError):
        return "unknown"


def discover_teams_paired(client: storage.Client, bucket_name: str) -> List[FrameRecord]:
    """
    live-deepfake-methods-*-teams layout:
      samples/{strategy_NNNN}/manifest.json
      samples/{strategy_NNNN}/frames/{fake,real}/frame_NNNN.jpg
    """
    records = []
    bucket = client.bucket(bucket_name)
    image_exts = {".png", ".jpg", ".jpeg"}

    # List all sample directories
    sample_blobs = bucket.list_blobs(prefix="samples/")
    # Build a set of sample directories
    sample_dirs = set()
    all_blobs_by_prefix: Dict[str, List[storage.Blob]] = {}

    for blob in sample_blobs:
        if blob.name.endswith("/"):
            continue
        # samples/edge_cases_0000/manifest.json
        parts = blob.name.split("/")
        if len(parts) >= 2:
            sample_key = parts[1]  # edge_cases_0000
            sample_dirs.add(sample_key)

            if sample_key not in all_blobs_by_prefix:
                all_blobs_by_prefix[sample_key] = []
            all_blobs_by_prefix[sample_key].append(blob)

    for sample_key in sorted(sample_dirs):
        # Derive strategy from the sample key: edge_cases_0000 → edge_cases
        strategy = "_".join(sample_key.split("_")[:-1])  # strip numeric suffix

        blobs = all_blobs_by_prefix.get(sample_key, [])

        # Read manifest if present (for metadata, but we can also infer)
        manifest_data = {}
        for blob in blobs:
            if blob.name.endswith("manifest.json"):
                try:
                    content = blob.download_as_text()
                    manifest_data = json.loads(content)
                except Exception:
                    pass
                break

        for blob in blobs:
            ext = os.path.splitext(blob.name)[1].lower()
            if ext not in image_exts:
                continue

            # samples/edge_cases_0000/frames/fake/frame_0001.jpg
            parts = blob.name.split("/")
            if "frames" not in parts:
                continue

            frames_idx = parts.index("frames")
            if frames_idx + 1 >= len(parts):
                continue

            label_str = parts[frames_idx + 1]  # fake or real
            if label_str not in ("fake", "real"):
                continue

            label_int = 1 if label_str == "fake" else 0
            frame_name = parts[-1]

            records.append(FrameRecord(
                bucket=bucket_name,
                blob_path=blob.name,
                label=label_int,
                method=f"teams_{strategy}" if label_int == 1 else "real",
                video_id=sample_key,
                frame_name=frame_name,
                strategy=strategy,
                extra={
                    "pair_complete": str(manifest_data.get("pair_complete", "")),
                },
            ))

    logger.info("[%s] Discovered %d frames", bucket_name, len(records))
    return records


def discover_bucket(client: storage.Client, bucket_name: str) -> List[FrameRecord]:
    """Auto-detect bucket layout and discover all frames."""
    if bucket_name == "poc-phase-1-test":
        return discover_poc_phase1(client, bucket_name)
    elif bucket_name.startswith("teams-faces-data-test"):
        return discover_teams_flat(client, bucket_name)
    elif bucket_name.startswith("live-deepfake-methods") and bucket_name.endswith("teams"):
        return discover_teams_paired(client, bucket_name)
    else:
        # Try to auto-detect
        bucket = client.bucket(bucket_name)
        # Check if it has samples/ directory → paired layout
        samples_it = bucket.list_blobs(prefix="samples/", max_results=1)
        if any(True for _ in samples_it):
            return discover_teams_paired(client, bucket_name)
        # Check if it has fake/ and real/ → flat or video-folder layout
        has_subfolders = False
        for blob in bucket.list_blobs(prefix="fake/", max_results=5):
            rel = blob.name[len("fake/"):]
            if "/" in rel:
                has_subfolders = True
            break
        if has_subfolders:
            return discover_poc_phase1(client, bucket_name)
        else:
            return discover_teams_flat(client, bucket_name)


# =============================================================================
# PyTorch Dataset — streams images from GCS
# =============================================================================
class GCSFrameDataset(data.Dataset):
    """
    Dataset that loads pre-cropped face images directly from GCS blobs.
    Images are already face-cropped, so we just resize to 224×224 and normalize.
    
    NOTE: storage.Client is not picklable, so we create per-worker clients 
    lazily (supports num_workers > 0 in DataLoader).
    """

    def __init__(
        self,
        records: List[FrameRecord],
        resolution: int = 224,
    ):
        self.records = records
        self.resolution = resolution
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])
        # Per-worker GCS client and bucket cache (created lazily)
        self._client: Optional[storage.Client] = None
        self._buckets: Dict[str, Any] = {}

    def _get_client(self) -> storage.Client:
        if self._client is None:
            self._client = storage.Client()
        return self._client

    def _get_bucket(self, name: str):
        if name not in self._buckets:
            self._buckets[name] = self._get_client().bucket(name)
        return self._buckets[name]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rec = self.records[idx]
        bucket = self._get_bucket(rec.bucket)
        blob = bucket.blob(rec.blob_path)

        # Download to memory
        try:
            img_bytes = blob.download_as_bytes()
        except Exception as e:
            logger.warning("Failed to download gs://%s/%s: %s", rec.bucket, rec.blob_path, e)
            return torch.zeros(3, self.resolution, self.resolution), idx

        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img_bgr is None:
            # Return a zeroed tensor on decode failure — we'll flag this in output
            logger.warning("Failed to decode image: gs://%s/%s", rec.bucket, rec.blob_path)
            return torch.zeros(3, self.resolution, self.resolution), idx

        # Resize to model input resolution (images are already face-cropped)
        img_bgr = cv2.resize(img_bgr, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        tensor = self.transform(img_rgb)
        return tensor, idx


# =============================================================================
# Model loading (reuses patterns from simple_inference.py / run_r8_score_manifest.py)
# =============================================================================
def load_model(
    checkpoint_path: str,
    detector_config: str,
    train_config: str,
    device: torch.device,
) -> torch.nn.Module:
    """Load Effort detector from checkpoint."""
    # Merge configs
    with open(detector_config, "r") as f:
        cfg = yaml.safe_load(f)
    with open(train_config, "r") as f:
        train_cfg = yaml.safe_load(f)
    cfg.update(train_cfg)

    # Load checkpoint
    logger.info("Loading checkpoint: %s", checkpoint_path)
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
            logger.info("Restored ArcFace scale to %.3f", model_config["current_arcface_s"])

    clean_state = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing:
        logger.debug("Missing keys: %s", missing)
    if unexpected:
        logger.debug("Unexpected keys: %s", unexpected)

    model.eval()
    logger.info("Model loaded on %s", device)
    return model


# =============================================================================
# Batch inference engine
# =============================================================================
def run_inference(
    model: torch.nn.Module,
    records: List[FrameRecord],
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 4,
) -> List[Dict[str, Any]]:
    """
    Run inference on all records. Returns list of result dicts.
    """
    dataset = GCSFrameDataset(records)
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    results = [None] * len(records)
    total_scored = 0
    total_failed = 0
    t0 = time.time()

    for batch_idx, (images, indices) in enumerate(loader):
        images = images.to(device, non_blocking=True)

        with torch.inference_mode():
            outputs = model({"image": images}, inference=True)
            probs = outputs["prob"].detach().cpu().numpy().reshape(-1)

        for i, global_idx in enumerate(indices.numpy()):
            rec = records[int(global_idx)]
            prob = float(probs[i])

            # Detect decode failures (zero tensor → unusual prob)
            is_failed = bool(images[i].sum().item() == 0.0)

            results[int(global_idx)] = {
                "bucket": rec.bucket,
                "blob_path": rec.blob_path,
                "gcs_uri": f"gs://{rec.bucket}/{rec.blob_path}",
                "label": rec.label,
                "label_str": "fake" if rec.label == 1 else "real",
                "method": rec.method,
                "video_id": rec.video_id,
                "frame_name": rec.frame_name,
                "strategy": rec.strategy,
                "prob_fake": f"{prob:.8f}",
                "status": "failed_decode" if is_failed else "ok",
            }

            if is_failed:
                total_failed += 1
            else:
                total_scored += 1

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
            elapsed = time.time() - t0
            fps = total_scored / max(elapsed, 0.001)
            logger.info(
                "  Batch %d/%d | scored=%d failed=%d | %.1f fps | %.1fs elapsed",
                batch_idx + 1, len(loader), total_scored, total_failed, fps, elapsed,
            )

    # Filter out any None entries (shouldn't happen with correct indexing)
    results = [r for r in results if r is not None]

    elapsed = time.time() - t0
    logger.info(
        "Inference complete: %d frames in %.1fs (%.1f fps), %d failed",
        total_scored + total_failed, elapsed,
        (total_scored + total_failed) / max(elapsed, 0.001),
        total_failed,
    )
    return results


# =============================================================================
# CSV output
# =============================================================================
CSV_FIELDS = [
    "bucket", "gcs_uri", "blob_path",
    "label", "label_str", "method", "video_id",
    "frame_name", "strategy", "prob_fake", "status",
]


def write_results_csv(results: List[Dict[str, Any]], output_path: str) -> None:
    """Write results to CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    logger.info("Wrote %d rows to %s", len(results), output_path)


def upload_to_gcs(local_path: str, gcs_uri: str) -> None:
    """Upload a local file to GCS."""
    bucket_name, blob_name = _parse_gs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    logger.info("Uploaded %s → %s", local_path, gcs_uri)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Batch inference on GCS test buckets — per-frame CSV output"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Checkpoint path (gs:// URI or local path)",
    )
    parser.add_argument(
        "--buckets", nargs="+", required=True,
        help="GCS bucket names to process (without gs:// prefix)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./inference_results",
        help="Directory for output CSVs (local or /gcs/ mount)",
    )
    parser.add_argument(
        "--gcs_output_bucket", type=str, default="training-job-outputs",
        help="GCS bucket for uploading results (if --upload_results is set)",
    )
    parser.add_argument(
        "--gcs_output_prefix", type=str, default="batch_inference_results",
        help="GCS prefix within output bucket for results",
    )
    parser.add_argument(
        "--upload_results", action="store_true", default=False,
        help="Upload results CSVs to GCS after completion",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128,
        help="Inference batch size (default: 128 for A100 40GB)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8,
        help="DataLoader workers for parallel image loading",
    )
    parser.add_argument(
        "--detector_config", type=str, default="config/detector/effort.yaml",
    )
    parser.add_argument(
        "--train_config", type=str, default="config/train_config.yaml",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument(
        "--run_id", type=str, default="",
        help="Optional run identifier for output naming (defaults to checkpoint basename)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        logger.info("GPU: %s (%.1f GB)", gpu_name, gpu_mem)

    # Download checkpoint if on GCS
    if args.checkpoint.startswith("gs://"):
        local_ckpt = _download_checkpoint(args.checkpoint)
    else:
        local_ckpt = args.checkpoint

    # Load model
    model = load_model(local_ckpt, args.detector_config, args.train_config, device)

    # Run ID for output naming
    run_id = args.run_id
    if not run_id:
        run_id = Path(local_ckpt).stem  # e.g. top_n_effort_20260228_step6000_auc0.9891_eer0.0457

    # GCS client for data loading
    gcs_client = storage.Client()

    # Process each bucket
    all_results = []
    os.makedirs(args.output_dir, exist_ok=True)

    for bucket_name in args.buckets:
        logger.info("=" * 70)
        logger.info("Processing bucket: %s", bucket_name)
        logger.info("=" * 70)

        # Discover frames
        t0 = time.time()
        records = discover_bucket(gcs_client, bucket_name)
        disc_time = time.time() - t0

        if not records:
            logger.warning("No frames found in bucket %s — skipping", bucket_name)
            continue

        # Log discovery summary
        fake_count = sum(1 for r in records if r.label == 1)
        real_count = sum(1 for r in records if r.label == 0)
        methods = sorted(set(r.method for r in records))
        videos = len(set(r.video_id for r in records))
        logger.info(
            "Discovered %d frames (fake=%d, real=%d) across %d videos, %d methods in %.1fs",
            len(records), fake_count, real_count, videos, len(methods), disc_time,
        )
        logger.info("Methods: %s", ", ".join(methods))

        # Run inference
        results = run_inference(
            model=model,
            records=records,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Write per-bucket CSV
        safe_bucket_name = bucket_name.replace("/", "_")
        csv_name = f"{run_id}__{safe_bucket_name}.csv"
        csv_path = os.path.join(args.output_dir, csv_name)
        write_results_csv(results, csv_path)

        all_results.extend(results)

        # Upload to GCS if requested
        if args.upload_results:
            gcs_dest = f"gs://{args.gcs_output_bucket}/{args.gcs_output_prefix}/{csv_name}"
            upload_to_gcs(csv_path, gcs_dest)

    # Write combined CSV
    if len(args.buckets) > 1 and all_results:
        combined_name = f"{run_id}__combined.csv"
        combined_path = os.path.join(args.output_dir, combined_name)
        write_results_csv(all_results, combined_path)

        if args.upload_results:
            gcs_dest = f"gs://{args.gcs_output_bucket}/{args.gcs_output_prefix}/{combined_name}"
            upload_to_gcs(combined_path, gcs_dest)

    # Print final summary
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info("Total frames scored: %d", len(all_results))
    logger.info("Output directory: %s", args.output_dir)

    for bucket_name in args.buckets:
        bucket_results = [r for r in all_results if r["bucket"] == bucket_name]
        if not bucket_results:
            continue
        ok = sum(1 for r in bucket_results if r["status"] == "ok")
        failed = sum(1 for r in bucket_results if r["status"] != "ok")
        fake_ok = [r for r in bucket_results if r["label"] == 1 and r["status"] == "ok"]
        real_ok = [r for r in bucket_results if r["label"] == 0 and r["status"] == "ok"]

        if fake_ok:
            fake_probs = [float(r["prob_fake"]) for r in fake_ok]
            fake_mean = np.mean(fake_probs)
            fake_acc = np.mean([1 if p >= 0.5 else 0 for p in fake_probs])
        else:
            fake_mean = fake_acc = 0.0

        if real_ok:
            real_probs = [float(r["prob_fake"]) for r in real_ok]
            real_mean = np.mean(real_probs)
            real_acc = np.mean([1 if p < 0.5 else 0 for p in real_probs])
        else:
            real_mean = real_acc = 0.0

        logger.info(
            "  %s: %d ok, %d failed | fake(n=%d mean_prob=%.4f acc@0.5=%.3f) | real(n=%d mean_prob=%.4f acc@0.5=%.3f)",
            bucket_name, ok, failed,
            len(fake_ok), fake_mean, fake_acc,
            len(real_ok), real_mean, real_acc,
        )


if __name__ == "__main__":
    main()
