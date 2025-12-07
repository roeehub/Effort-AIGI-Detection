#!/usr/bin/env python3
"""Minimal inference harness for the Effort detector.

This script ensures only the essential CLIP backbone artifacts are downloaded,
loads either the base or a custom checkpoint, and runs a single-frame
inference so the weights can be quantized or exported afterwards.
"""

import argparse
import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2  # type: ignore
import torch
import yaml
from google.api_core import exceptions
from google.cloud import storage
from google.cloud.storage import Bucket

import video_preprocessor
from detectors import DETECTOR

logger = logging.getLogger("simple-inference")


# ---------------------------------------------------------------------------
# GCS utilities (kept self-contained to avoid importing FastAPI apps)
# ---------------------------------------------------------------------------
def _download_gcs_asset(
    bucket: Bucket,
    gcs_path: str,
    local_path: str,
    allowed_files: Optional[List[str]] = None,
) -> bool:
    """Download a single blob or a directory from GCS.

    Args:
        bucket: The GCS bucket object.
        gcs_path: The fully qualified GCS path (gs://bucket/path or path).
        local_path: Local destination path (file or directory).
        allowed_files: Optional list of relative file paths to download when
            ``gcs_path`` represents a directory.
    """

    if gcs_path.startswith("gs://"):
        prefix_to_strip = f"gs://{bucket.name}/"
    else:
        prefix_to_strip = f"{bucket.name}/"

    if gcs_path.endswith("/"):
        prefix = gcs_path.replace(prefix_to_strip, "", 1)
        os.makedirs(local_path, exist_ok=True)

        if allowed_files:
            normalized = [p.lstrip("/") for p in allowed_files]
            for rel_path in normalized:
                blob_name = f"{prefix}{rel_path}" if prefix else rel_path
                blob = bucket.blob(blob_name)
                if not blob.exists():
                    logger.error("Missing file in GCS directory: gs://%s/%s", bucket.name, blob_name)
                    return False

                destination = Path(local_path) / rel_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    blob.download_to_filename(str(destination))
                except Exception as exc:  # pragma: no cover - transport errors
                    logger.error("Failed to download %s: %s", blob_name, exc)
                    return False

            logger.info("Downloaded %d targeted file(s) from %s", len(normalized), gcs_path)
            return True

        blobs = list(bucket.list_blobs(prefix=prefix))
        if not blobs:
            logger.error("Directory %s is empty or does not exist.", gcs_path)
            return False

        downloaded_any = False
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            rel = os.path.relpath(blob.name, prefix)
            destination = Path(local_path) / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                blob.download_to_filename(str(destination))
                downloaded_any = True
            except Exception as exc:  # pragma: no cover - transport errors
                logger.error("Failed to download %s: %s", blob.name, exc)
                return False

        if not downloaded_any:
            logger.error("Directory %s contained no downloadable files.", gcs_path)
            return False
        return True

    # Single file branch -----------------------------------------------------
    blob_name = gcs_path.replace(prefix_to_strip, "", 1)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        logger.error("File not found at gs://%s/%s", bucket.name, blob_name)
        return False

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)
    return True


def _asset_already_present(asset: Dict[str, Any]) -> bool:
    local_path = Path(asset.get("local_path", ""))
    if not local_path:
        return False

    expected_files = asset.get("files")
    if expected_files:
        return all((local_path / Path(f)).exists() for f in expected_files)
    return local_path.exists()


def ensure_assets_downloaded(assets: Dict[str, Dict[str, Any]], force_download: bool = False) -> None:
    """Ensure all required assets are present locally, downloading if needed."""
    for key, asset in assets.items():
        if not asset.get("gcs_path") and not _asset_already_present(asset):
            raise FileNotFoundError(
                f"Asset '{key}' is missing locally and no gcs_path was provided.")

    need_download = [
        (key, asset)
        for key, asset in assets.items()
        if asset.get("gcs_path") and (force_download or not _asset_already_present(asset))
    ]

    if not need_download:
        logger.info("All requested assets already exist locally. Skipping downloads.")
        return

    storage_client = storage.Client()

    for key, asset in need_download:
        gcs_path = asset["gcs_path"]
        local_path = asset["local_path"]
        allowed_files = asset.get("files")

        logger.info("Downloading '%s' from %s -> %s", key, gcs_path, local_path)
        bucket_name = gcs_path.split("gs://", 1)[1].split("/", 1)[0]
        bucket = storage_client.bucket(bucket_name)

        if not _download_gcs_asset(bucket, gcs_path, local_path, allowed_files):
            raise RuntimeError(f"Failed to download asset '{key}' from {gcs_path}")

    logger.info("All remote assets downloaded successfully.")


# ---------------------------------------------------------------------------
# Model loading and inference helpers
# ---------------------------------------------------------------------------
def load_detector(cfg: Dict[str, Any], weights_path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Instantiate the Effort detector and load checkpoint weights."""
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {weights_path}")

    logger.info("Loading weights from %s", weights_path)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    cfg_for_model = copy.deepcopy(cfg)
    model_config = {}

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        model_config = checkpoint.get("model_config", {})

        if model_config:
            logger.info("Restoring model configuration from checkpoint metadata")
            for key, value in model_config.items():
                if key == "current_arcface_s":
                    continue
                previous = cfg_for_model.get(key)
                cfg_for_model[key] = value
                if previous != value:
                    logger.debug("  %s: %s -> %s", key, previous, value)
    else:
        logger.warning("Checkpoint is in legacy format without metadata; proceeding with provided config.")
        state_dict = checkpoint

    model_cls = DETECTOR[cfg_for_model["model_name"]]
    model = model_cls(cfg_for_model).to(device)

    if model_config.get("use_arcface_head") and "current_arcface_s" in model_config:
        arcface_s = model_config["current_arcface_s"]
        if hasattr(model.head, "s"):
            model.head.s.data.fill_(arcface_s)
            logger.info("Restored ArcFace scale parameter to %.3f", arcface_s)

    clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing:
        logger.debug("Missing parameters when loading state dict: %s", missing)
    if unexpected:
        logger.debug("Unexpected parameters when loading state dict: %s", unexpected)

    model.eval()
    logger.info("Model ready on %s", device)
    return model, cfg_for_model


def run_single_frame(
    model: torch.nn.Module,
    device: torch.device,
    image_path: Path,
    threshold: float,
    recrop: bool,
    yolo_conf: float,
) -> Tuple[str, float]:
    """Run inference on a single frame and return label & probability."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Unable to decode image at {image_path}")

    if recrop:
        video_preprocessor.initialize_yolo_model()
        cropped = video_preprocessor.extract_yolo_face(bgr, yolo_conf)
        if cropped is None:
            raise RuntimeError("YOLO face detection failed for the provided frame.")
    else:
        cropped = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA)

    transform = video_preprocessor._get_transform()
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model({"image": tensor}, inference=True)
        prob_fake = float(outputs["prob"].squeeze().cpu().item())

    label = "FAKE" if prob_fake >= threshold else "REAL"
    return label, prob_fake


def summarize_backbone(backbone_dir: Path) -> None:
    if not backbone_dir.exists():
        logger.warning("Backbone directory %s does not exist.", backbone_dir)
        return

    files = sorted([p for p in backbone_dir.glob("**/*") if p.is_file()])
    total_bytes = sum(p.stat().st_size for p in files)
    logger.info("Backbone contents (%d files, %.2f GB):", len(files), total_bytes / (1024 ** 3))
    for path in files:
        size_gb = path.stat().st_size / (1024 ** 3)
        logger.info("  - %s (%.4f GB)", path.relative_to(backbone_dir), size_gb)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-frame inference with the Effort detector.")
    parser.add_argument("--detector-config", default="./config/detector/effort.yaml", help="Path to detector YAML")
    parser.add_argument("--train-config", default="./config/train_config.yaml", help="Path to train YAML")
    parser.add_argument("--image", required=True, help="Path to an image frame (BGR/PNG/JPEG)")
    parser.add_argument("--model-type", choices=["base", "custom"], default="base", help="Which model to load")
    parser.add_argument("--weights-local", help="Optional local checkpoint path to load.")
    parser.add_argument(
        "--weights-gcs",
        help="Optional gs:// path to download a checkpoint from before loading.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for FAKE vs REAL")
    parser.add_argument("--recrop", action="store_true", help="Run YOLO face detection before inference")
    parser.add_argument("--yolo-conf", type=float, default=0.2, help="YOLO confidence threshold when recropping")
    parser.add_argument("--force-download", action="store_true", help="Force re-download of assets.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    with open(args.detector_config, "r") as fp:
        detector_cfg = yaml.safe_load(fp)
    with open(args.train_config, "r") as fp:
        train_cfg = yaml.safe_load(fp)

    config = {**detector_cfg, **train_cfg}
    gcs_assets = copy.deepcopy(config.get("gcs_assets", {}))

    clip_asset = gcs_assets.get("clip_backbone")
    if not clip_asset:
        raise KeyError("clip_backbone asset not defined in configuration.")

    assets_to_fetch: Dict[str, Dict[str, Any]] = {"clip_backbone": copy.deepcopy(clip_asset)}

    # Determine which checkpoint to load --------------------------------------------------------
    weights_path: Optional[str] = args.weights_local
    if args.weights_gcs:
        if not args.weights_gcs.startswith("gs://"):
            raise ValueError("weights-gcs must start with 'gs://'")
        target_name = Path(args.weights_gcs.rstrip("/")).name
        local_target = Path(args.weights_local) if args.weights_local else Path("./weights") / target_name
        local_target.parent.mkdir(parents=True, exist_ok=True)
        asset_key = "custom_checkpoint" if args.model_type == "custom" else "base_checkpoint"
        assets_to_fetch[asset_key] = {
            "gcs_path": args.weights_gcs,
            "local_path": str(local_target),
        }
        weights_path = str(local_target)
    elif not weights_path:
        default_key = "base_checkpoint" if args.model_type == "base" else "custom_checkpoint"
        default_asset = gcs_assets.get(default_key)
        if default_asset:
            assets_to_fetch[default_key] = copy.deepcopy(default_asset)
            weights_path = default_asset.get("local_path")
        elif args.model_type == "custom":
            raise ValueError("No custom checkpoint provided. Use --weights-local or --weights-gcs.")

    if not weights_path:
        raise ValueError("Unable to determine checkpoint path to load.")

    # Ensure all relevant assets exist ----------------------------------------------------------
    try:
        ensure_assets_downloaded(assets_to_fetch, force_download=args.force_download)
    except exceptions.GoogleAPIError as exc:
        raise RuntimeError(f"GCS download failed: {exc}") from exc

    clip_dir = Path(assets_to_fetch["clip_backbone"]["local_path"])
    summarize_backbone(clip_dir)

    # Instantiate and run inference -------------------------------------------------------------
    model, effective_cfg = load_detector(config, weights_path, device)
    label, prob = run_single_frame(
        model,
        device,
        Path(args.image),
        threshold=args.threshold,
        recrop=args.recrop,
        yolo_conf=args.yolo_conf,
    )

    logger.info("Inference completed: label=%s, fake_prob=%.4f", label, prob)

    print()
    print("=== Inference Summary ===")
    print(f"Model type      : {args.model_type}")
    print(f"Checkpoint path : {weights_path}")
    print(f"Backbone path   : {clip_dir}")
    print(f"Decision        : {label}")
    print(f"Fake probability: {prob:.4f}")
    print(f"Threshold       : {args.threshold:.2f}")


if __name__ == "__main__":
    main()
