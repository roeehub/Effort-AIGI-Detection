import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import tempfile
import shutil
import time

import cv2  # noqa
import numpy as np  # noqa
import torch  # noqa
import yaml  # noqa
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Request, Query  # noqa
from fastapi.responses import JSONResponse  # noqa
from pydantic import BaseModel, Field  # noqa
from torch import nn  # noqa

import video_preprocessor
from detectors import DETECTOR, EffortDetector  # noqa
from google.cloud import storage  # noqa
from google.api_core import exceptions  # noqa
from google.cloud.storage import Bucket  # noqa

# ──────────────────────────────────────────
# Logging
# ──────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("effort-aigi-api-v3")


def _resolve_device() -> torch.device:
    """Resolve the inference device from the DEVICE env var.

    Precedence:
      1. DEVICE env var ("cuda", "mps", "cpu")
      2. CUDA if available  (preserves original default)
      3. CPU fallback
    """
    requested = os.getenv("DEVICE", "").lower().strip()
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "DEVICE=mps was requested but MPS is not available on this machine."
            )
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    # Default path: CUDA (matches original behaviour)
    if requested and requested != "cuda":
        logger.warning("Unknown DEVICE=%r — falling back to default CUDA/CPU selection.", requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


device = _resolve_device()
DEBUG_FRAME_DIR = "./debug_frames"

# ──────────────────────────────────────────
# Quality gate (deployment packet 2026-05-04)
# ──────────────────────────────────────────
# Lightweight per-frame gate. Frames that fail any criterion are rejected as
# "out of operational envelope" and get the default-real probability instead
# of going through the model. The gate criteria match the deployment packet at
# `analysis/deeplive_deployment_24h_2026-05-05/DEPLOYMENT_PACKET_DEEPLIVE.md`:
#   - min(width, height) >= QUALITY_GATE_MIN_DIM (rejects thumbnails)
#   - laplacian_var >= QUALITY_GATE_MIN_LAP_VAR (rejects extreme blur)
#   - is_no_face is enforced upstream by YOLO returning None on recrop=True paths
QUALITY_GATE_DEFAULT_PROB = 0.25
QUALITY_GATE_MIN_DIM = 80
QUALITY_GATE_MIN_LAP_VAR = 8.0


def pretty_print_batch(
    files: List,
    per_frame_status: List[Dict[str, Any]],
    confidence: float,
    threshold: float,
    pred_label: str,
) -> None:
    """ANSI-coloured per-frame readout, printed to the same logger.

    Emits a single multi-line block. Per-frame lines are GREEN when prob<threshold
    (REAL) and RED when prob>=threshold (FAKE). Gated frames are shown in DIM.
    The final mean line is colour-keyed by the batch verdict.
    """
    RED = "\033[91m"
    GREEN = "\033[92m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    total = len(files)
    bar_w = 16

    lines = []
    sep = "═" * 78
    lines.append(sep)
    head_color = RED if pred_label == "FAKE" else GREEN
    lines.append(
        f"{BOLD}[BATCH {total}] threshold={threshold:.2f}  "
        f"mean={head_color}{confidence:.3f}{RESET}{BOLD} → {head_color}{pred_label}{RESET}"
    )
    lines.append("─" * 78)

    for i, (f, s) in enumerate(zip(files, per_frame_status)):
        name = (f.filename or f"frame_{i}")[:28].ljust(28)
        kind = s.get("kind")
        if kind == "failed":
            lines.append(f"  {DIM}{name}  [decode failed]{RESET}")
            continue
        prob = s.get("prob")
        if prob is None:
            lines.append(f"  {DIM}{name}  [no prob]{RESET}")
            continue
        filled = int(round(prob * bar_w))
        bar = "█" * filled + "░" * (bar_w - filled)
        if kind == "gated":
            reason = s.get("reason", "gated")
            lines.append(f"  {DIM}{name}  {bar}  {prob:.3f}  GATED  ({reason}){RESET}")
        else:
            color = RED if prob >= threshold else GREEN
            verdict = "FAKE" if prob >= threshold else "REAL"
            lines.append(f"  {name}  {color}{bar}  {prob:.3f}  {verdict}{RESET}")

    lines.append(sep)
    logger.info("\n" + "\n".join(lines))


def quality_gate(img_bgr: Optional[np.ndarray], frame_id: str = "frame") -> tuple:
    """Per-frame deployment quality gate.

    Returns (passes, reason). When passes=False, the caller should NOT run the
    model on this frame; instead emit prob=QUALITY_GATE_DEFAULT_PROB (0.25,
    treated as REAL by the standard threshold=0.5) and log the gate hit so the
    rejection is auditable in production logs.

    Cheap to compute: one cvtColor + one Laplacian pass on the input image.
    """
    if img_bgr is None or img_bgr.size == 0:
        return False, "empty_or_none_image"
    h, w = img_bgr.shape[:2]
    if min(h, w) < QUALITY_GATE_MIN_DIM:
        return False, f"min_dim={min(h, w)}<{QUALITY_GATE_MIN_DIM}"
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception as e:
        return False, f"gate_compute_failed:{e!r}"
    if lap_var < QUALITY_GATE_MIN_LAP_VAR:
        return False, f"laplacian_var={lap_var:.2f}<{QUALITY_GATE_MIN_LAP_VAR}"
    return True, None


# ──────────────────────────────────────────
# GCS Asset Downloading Utilities
# ──────────────────────────────────────────
def download_gcs_asset(
        bucket: Bucket,
        gcs_path: str,
        local_path: str,
        logger,
        allowed_files: Optional[List[str]] = None) -> bool:
    """Downloads a single blob or a directory of blobs from GCS."""
    if not gcs_path.startswith('gs://'):
        # This function is now used for both assets and frame batches, so handle both formats
        prefix_to_strip = f"{bucket.name}/"
    else:
        prefix_to_strip = f"gs://{bucket.name}/"

    if gcs_path.endswith('/'):  # It's a directory
        prefix = gcs_path.replace(prefix_to_strip, '', 1)
        os.makedirs(local_path, exist_ok=True)

        if allowed_files:
            normalized_paths = [p.lstrip('/') for p in allowed_files]
            for rel_path in normalized_paths:
                blob_name = f"{prefix}{rel_path}" if prefix else rel_path
                blob = bucket.blob(blob_name)
                if not blob.exists():
                    logger.error(f"File not found in GCS directory: gs://{bucket.name}/{blob_name}")
                    return False

                destination_file_name = os.path.join(local_path, rel_path)
                os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)

                try:
                    blob.download_to_filename(destination_file_name)
                except Exception as e:
                    logger.error(f"Failed to download {blob.name}: {e}")
                    return False

            logger.debug(
                "Downloaded %d specific file(s) from %s", len(normalized_paths), gcs_path)
            return True

        blobs = list(bucket.list_blobs(prefix=prefix))  # Use list to check length
        if not blobs:
            logger.error(f"Directory {gcs_path} is empty or does not exist.")
            return False

        downloaded = False

        # Decide whether this folder is an "images folder" or a "generic assets folder".
        image_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        any_images = any(
            (not b.name.endswith('/')) and b.name.lower().endswith(image_exts)
            for b in blobs
        )

        for blob in blobs:
            if blob.name.endswith('/'):
                continue

            # If the directory contains images, keep previous behavior (download images only).
            # Otherwise, download **all** files (needed for model folders like CLIP backbone).
            if any_images and not blob.name.lower().endswith(image_exts):
                logger.debug(f"Skipping non-image file in GCS directory: {blob.name}")
                continue

            # Preserve relative subpaths under the prefix
            rel = blob.name.replace(prefix, '', 1)
            destination_file_name = os.path.join(local_path, rel)
            os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)

            try:
                blob.download_to_filename(destination_file_name)
                downloaded = True
            except Exception as e:
                logger.error(f"Failed to download {blob.name}: {e}")
                return False

        return downloaded

    else:  # It's a single file
        blob_name = gcs_path.replace(prefix_to_strip, '', 1)
        blob = bucket.blob(blob_name)
        if not blob.exists():
            logger.error(f"File not found at gs://{bucket.name}/{blob_name}")
            return False
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        return True


def download_assets_from_gcs(config, logger):
    """Downloads specified assets (checkpoints, models) from a GCS bucket."""
    assets_to_download = config.get('gcs_assets')
    if not assets_to_download:
        logger.info("No GCS assets configured for download. Skipping.")
        return None
    local_paths = {}
    all_exist = all(os.path.exists(asset.get('local_path', '')) for asset in assets_to_download.values())
    if all_exist:
        logger.info("All GCS assets already exist locally. Skipping downloads.")
        return {key: asset['local_path'] for key, asset in assets_to_download.items()}
    logger.info("--- GCS Asset Download ---")
    try:
        storage_client = storage.Client()
        start_time = time.time()
        for key, asset_info in assets_to_download.items():
            gcs_path, local_path = asset_info.get('gcs_path'), asset_info.get('local_path')
            if not gcs_path or not local_path:
                logger.info(f"Asset '{key}' has no gcs_path or local_path configured. Skipping.")
                continue
            if not gcs_path.startswith('gs://'):
                logger.error(f"Invalid GCS path for asset '{key}': '{gcs_path}'.")
                return None
            if os.path.exists(local_path):
                logger.info(f"Asset '{key}' already exists at {local_path}. Skipping.")
                local_paths[key] = local_path
                continue
            logger.info(f"Downloading asset '{key}': {gcs_path} -> {local_path}")
            bucket_name = gcs_path.split('gs://', 1)[1].split('/', 1)[0]
            bucket = storage_client.bucket(bucket_name)
            allowed_files = asset_info.get('files')
            if not download_gcs_asset(bucket, gcs_path, local_path, logger, allowed_files=allowed_files):
                raise RuntimeError(f"Failed to download asset '{key}'.")
            local_paths[key] = local_path
            logger.info(f"✅ SUCCESS: Downloaded '{key}'.")
        logger.info(f"✅ SUCCESS: All GCS assets downloaded in {time.time() - start_time:.2f}s.")
        return local_paths
    except (exceptions.Forbidden, exceptions.NotFound) as e:
        logger.error(f"FAILED: GCP access error for assets. Ensure permissions/paths are correct. Details: {e}")
        return None
    except Exception as e:
        logger.error(f"FAILED: An unexpected error occurred during GCS download: {e}")
        return None


# ──────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────
def load_detector(cfg: dict, weights: str) -> nn.Module:
    """Loads the EffortDetector model from config and weights with configuration validation."""
    logger.info(f"Loading detector from: {weights}")
    
    # Load checkpoint (weights_only requires PyTorch ≥ 1.13)
    try:
        ckpt = torch.load(weights, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(weights, map_location=device)
    
    # Handle both old and new checkpoint formats
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        # New format with configuration
        state_dict = ckpt['state_dict']
        model_config = ckpt.get('model_config', {})
        
        # Update config with saved configuration for exact reconstruction
        if model_config:
            logger.info("📋 Restoring model configuration from checkpoint:")
            for key, value in model_config.items():
                if key == 'current_arcface_s':  # Skip dynamic parameter
                    continue
                old_value = cfg.get(key)
                if key == 'gcs_assets' and isinstance(value, dict):
                    # Merge instead of replace to preserve runtime entries
                    # (e.g. custom_checkpoint added by startup_event)
                    cfg.setdefault('gcs_assets', {}).update(value)
                else:
                    cfg[key] = value
                if old_value != value:
                    logger.info(f"  {key}: {old_value} → {value}")
            
            logger.info(f"📊 Checkpoint: Epoch {ckpt.get('epoch')}, AUC: {ckpt.get('auc', 0):.4f}")
        else:
            logger.warning("No model configuration in checkpoint - using provided config")
    else:
        # Old format
        state_dict = ckpt
        model_config = {}
        logger.warning("⚠️  Old checkpoint format detected. Configuration validation not possible.")
    
    # If checkpoint restored gcs_assets (e.g. a different backbone), ensure those
    # assets are downloaded before we attempt to instantiate the model.
    if model_config.get('gcs_assets'):
        restored_assets = cfg.get('gcs_assets', {})
        missing = {
            k: v for k, v in restored_assets.items()
            if v.get('local_path') and not os.path.exists(v['local_path'])
        }
        if missing:
            logger.info(f"📥 Downloading {len(missing)} asset(s) restored from checkpoint config...")
            result = download_assets_from_gcs({'gcs_assets': missing}, logger)
            if result is None:
                raise RuntimeError(
                    f"Failed to download checkpoint-specified assets: {list(missing.keys())}"
                )

    # Initialize model with (possibly updated) config
    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)
    
    # Restore dynamic ArcFace parameter if available
    if model_config.get('use_arcface_head', False) and 'current_arcface_s' in model_config:
        if hasattr(model, 'head') and hasattr(model.head, 's'):
            current_s = model_config['current_arcface_s']
            model.head.s.data.fill_(current_s)
            logger.info(f"  Restored ArcFace s parameter: {current_s}")
    
    # Load state dict with module prefix handling. Capture the IncompatibleKeys
    # return value so missing/unexpected keys are surfaced explicitly. With
    # strict=False (kept for backwards compat with old checkpoint formats),
    # mismatches are otherwise silent and could hide architecture confusion.
    state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing:
        logger.warning(
            "⚠️  load_state_dict: %d MISSING key(s) (model expected, ckpt did not provide; "
            "these layers run with random init): %s%s",
            len(missing),
            missing[:10],
            " ..." if len(missing) > 10 else "",
        )
    if unexpected:
        logger.warning(
            "⚠️  load_state_dict: %d UNEXPECTED key(s) (ckpt provided, model did not need; "
            "these weights are silently dropped): %s%s",
            len(unexpected),
            unexpected[:10],
            " ..." if len(unexpected) > 10 else "",
        )
    if not missing and not unexpected:
        logger.info("✅ load_state_dict: all keys matched (no missing, no unexpected)")
    model.eval()

    # Final loaded-config summary so the operator can confirm what actually
    # loaded vs what env vars / yaml said. The checkpoint's saved model_config
    # OVERRIDES env vars and yaml (lines above) — this summary makes that
    # override explicit, since silent overrides previously produced confusion
    # about whether ArcFace was actually active at inference.
    logger.info(
        "📋 FINAL loaded config: model_name=%s use_arcface_head=%s arcface_m=%s arcface_s=%s",
        cfg.get("model_name"),
        cfg.get("use_arcface_head"),
        cfg.get("arcface_m"),
        cfg.get("arcface_s"),
    )
    logger.info("✅ Model loaded and set to evaluation mode")
    return model


# ──────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────
app = FastAPI(title="Effort-AIGI Detector API", version="0.7.0")


# --- API Models ---
class InferResponse(BaseModel):
    pred_label: str
    fake_prob: float


class GCSPathRequest(BaseModel):
    gcs_path: str


class AggregationResult(BaseModel):
    decision: str
    score: float
    frames_used: int


class AggregationSet(BaseModel):
    mean: AggregationResult
    median: AggregationResult
    std: Optional[AggregationResult] = None
    majority_vote: AggregationResult


class PolicySet(BaseModel):
    unsure_policy_off: AggregationSet
    unsure_policy_on: Optional[AggregationSet] = None


class FrameCountResults(BaseModel):
    frames_8: Optional[PolicySet] = Field(None, alias="8_frames")
    frames_16: Optional[PolicySet] = Field(None, alias="16_frames")
    frames_32: PolicySet = Field(..., alias="32_frames")
    # frames_64: Optional[PolicySet] = Field(None, alias="64_frames")


class VideoAnalysisResponse(BaseModel):
    results: FrameCountResults
    raw_frame_probs: List[float]


class BatchInferResponse(BaseModel):
    pred_label: str  # Add prediction label based on threshold
    confidence: float  # mean of frame-level fake probabilities
    probs: List[float]  # per-frame fake probabilities


# --- Analysis Helper ---
def calculate_analysis(frame_probs: List[float], threshold: float) -> VideoAnalysisResponse:
    """Performs the full analysis matrix on a list of frame probabilities."""
    analysis: Dict[str, Any] = {}

    for n_frames in [32]:  # [8, 16, 32, 64]
        # Use up to n_frames, but don't fail if fewer are available
        sample_probs = frame_probs[:n_frames]

        policy_results: Dict[str, Any] = {}
        for policy in ["off"]:  # ["off", "on"]
            if not sample_probs:
                # Handle case where initial list is empty
                agg_set = {
                    agg: {"decision": "N/A", "score": -1.0, "frames_used": 0}
                    for agg in ["mean", "median", "std", "majority_vote"]
                }
                policy_results[f"unsure_policy_{policy}"] = agg_set
                continue

            if policy == "on":
                filtered_probs = [p for p in sample_probs if not (0.4 <= p <= 0.6)]
            else:
                filtered_probs = sample_probs

            # Numpy array for easier calculations
            np_probs = np.array(filtered_probs)
            frames_used = len(filtered_probs)

            agg_results: Dict[str, Any] = {}
            if frames_used > 0:
                # Mean
                mean_score = np.mean(np_probs)
                agg_results["mean"] = {
                    "decision": "FAKE" if mean_score >= threshold else "REAL",
                    "score": mean_score,
                    "frames_used": frames_used
                }
                # Median
                median_score = np.median(np_probs)
                agg_results["median"] = {
                    "decision": "FAKE" if median_score >= threshold else "REAL",
                    "score": median_score,
                    "frames_used": frames_used
                }
                # Std (decision based on mean)
                # std_score = np.std(np_probs)
                # agg_results["std"] = {
                #     "decision": "FAKE" if mean_score >= threshold else "REAL",
                #     "score": std_score,
                #     "frames_used": frames_used
                # }
                # Majority Vote
                fake_count = np.sum(np_probs >= threshold)
                agg_results["majority_vote"] = {
                    "decision": "FAKE" if fake_count > frames_used / 2 else "REAL",
                    "score": fake_count / frames_used,
                    "frames_used": frames_used
                }
            else:  # No frames left after filtering
                agg_results = {
                    agg: {"decision": "N/A", "score": -1.0, "frames_used": 0}
                    for agg in ["mean", "median", "std", "majority_vote"]
                }

            policy_results[f"unsure_policy_{policy}"] = agg_results

        analysis[f"{n_frames}_frames"] = policy_results

    return VideoAnalysisResponse(results=analysis, raw_frame_probs=frame_probs)


# ──────────────────────────────────────────
# Startup: Load Model(s) & Assert CUDA
# ──────────────────────────────────────────
# In app3.py

@app.on_event("startup")
def startup_event() -> None:
    # 0) Initialize state
    app.state.models = {}
    app.state.loaded_weights_paths = {}

    # 1) Device Check
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA is not available. Set DEVICE=mps or DEVICE=cpu to run without CUDA.")
        raise RuntimeError("CUDA was selected but is not available")
    logger.info("Using device: %s", device)

    # 2) Define paths and check for required config files
    repo_base = Path(".")
    cfg_path = repo_base / "config/detector/effort.yaml"
    train_cfg_path = repo_base / "config/train_config.yaml"
    base_weights_path = repo_base / "weights/effort_clip_L14_trainOn_FaceForensic.pth"

    if not all([cfg_path.exists(), train_cfg_path.exists()]):
        raise RuntimeError("A required config file (effort.yaml or train_config.yaml) was not found.")

    # 3) Load and merge configurations
    try:
        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)
        with open(train_cfg_path, "r") as f:
            config.update(yaml.safe_load(f))
        logger.info("Successfully loaded and merged configuration files.")
    except Exception as e:
        logger.exception("Failed to load or merge YAML configuration files.")
        raise e

    # 4) Handle custom checkpoint from environment variable
    custom_checkpoint_gcs_path = os.getenv("CHECKPOINT_GCS_PATH")
    custom_weights_path = None
    if custom_checkpoint_gcs_path:
        logger.info("Custom checkpoint specified via environment variable.")
        if not custom_checkpoint_gcs_path.startswith("gs://"):
            raise RuntimeError(f"Invalid CHECKPOINT_GCS_PATH: '{custom_checkpoint_gcs_path}'. Must start with 'gs://'.")

        local_filename = Path(custom_checkpoint_gcs_path.split("gs://", 1)[1]).name
        custom_weights_dir = repo_base / "weights" / "custom"
        custom_weights_dir.mkdir(parents=True, exist_ok=True)
        custom_weights_path = custom_weights_dir / local_filename

        logger.info(f"  Custom GCS Path: {custom_checkpoint_gcs_path}")
        logger.info(f"  Custom Local Path: {custom_weights_path}")

        config.setdefault('gcs_assets', {})['custom_checkpoint'] = {
            'gcs_path': custom_checkpoint_gcs_path,
            'local_path': str(custom_weights_path)
        }
    else:
        logger.info("No custom checkpoint specified. Only the base model will be loaded.")

    # 5) Download all configured GCS Assets
    if not download_assets_from_gcs(config, logger):
        raise RuntimeError("Failed to prepare one or more model assets from GCS.")

    # 6) Load Base Model (optional – skip if weights are missing)
    logger.info("--- Loading Base Model ---")
    if not base_weights_path.exists():
        logger.warning(f"⚠️  Base model weights not found at {base_weights_path}. Skipping base model.")
    else:
        try:
            # The base model always uses the default config
            app.state.models['base'] = load_detector(config, str(base_weights_path))
            app.state.loaded_weights_paths['base'] = str(base_weights_path)
            logger.info(f"✅ SUCCESS: Base detector model loaded from: {base_weights_path}")
        except Exception as e:
            logger.exception("Failed to load BASE detector model")
            raise e

    # 7) Load Custom Model (if configured)
    if custom_checkpoint_gcs_path and custom_weights_path:
        logger.info("--- Loading Custom Model ---")
        if not custom_weights_path.exists():
            raise RuntimeError(
                f"Custom model weights file does not exist after GCS download attempt: {custom_weights_path}")

        # ───> START OF CHANGES <───
        # Create a copy of the config to modify specifically for the custom model.
        # This prevents affecting the base model's configuration.
        custom_config = config.copy()

        # Check for the new environment variable to toggle ArcFace head
        use_arcface_env = os.getenv("CUSTOM_MODEL_USE_ARCFACE", "false").lower()
        if use_arcface_env in ['true', '1', 't']:
            logger.info("✅ CUSTOM_MODEL_USE_ARCFACE is 'true'. Overriding config to use ArcFace head.")
            custom_config['use_arcface_head'] = True
            # You could also add other env vars for s, m, etc. if needed
            # custom_config['arcface_s'] = float(os.getenv("CUSTOM_MODEL_ARCFACE_S", 30.0))
        else:
            logger.info("CUSTOM_MODEL_USE_ARCFACE is not set or 'false'. Using default head from config file.")

        try:
            # Pass the potentially modified config to the loader function
            app.state.models['custom'] = load_detector(custom_config, str(custom_weights_path))
            app.state.loaded_weights_paths['custom'] = str(custom_weights_path)
            logger.info(f"✅ SUCCESS: Custom detector model loaded from: {custom_weights_path}")
        except Exception as e:
            logger.exception("Failed to load CUSTOM detector model")
            raise e
        # ───> END OF CHANGES <───

    # 8) Load Face Preprocessor Models (YOLO only) – optional
    try:
        video_preprocessor.initialize_yolo_model()
        app.state.yolo_available = True
        logger.info("✅ SUCCESS: YOLO face detector loaded successfully.")
    except Exception as e:
        app.state.yolo_available = False
        logger.warning("⚠️  YOLO model not available: %s. Endpoints requiring face detection will be disabled.", e)

    logger.info("Startup complete. Available models: %s, YOLO: %s",
                list(app.state.models.keys()), app.state.yolo_available)


# --- Utility: assert YOLO is loaded ---
def require_yolo(request: Request) -> None:
    """Raises 503 if YOLO face detector was not loaded at startup."""
    if not getattr(request.app.state, 'yolo_available', False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face detection (YOLO) is not available. "
                   "Use /check_frame_batch with recrop=false for pre-cropped images, "
                   "or restart the server with the YOLO model accessible."
        )


# --- Utility function to get model for endpoints ---
def get_model_for_request(request: Request, model_type: Optional[str]) -> nn.Module:
    """Gets the requested model from app state.

    `model_type` is REQUIRED — every inference request must specify "custom"
    (the user-uploaded checkpoint loaded via CHECKPOINT_GCS_PATH) or "base"
    (the local CLIP-L14 baseline). The previous default ("base") and silent
    auto-fallback to whichever model was loaded both removed 2026-05-04 — they
    masked which model was actually scoring the request, which is operationally
    confusing during A/B comparisons.
    """
    if model_type is None or model_type == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query parameter 'model_type' is REQUIRED. Pass ?model_type=custom "
                   "for the uploaded checkpoint, or ?model_type=base for the local "
                   "CLIP-L14 baseline. No default — must be explicit on every request."
        )
    if model_type not in ("base", "custom"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model_type={model_type!r}. Must be 'base' or 'custom'."
        )

    available = request.app.state.models
    if model_type not in available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Requested model_type={model_type!r} is not loaded. "
                   f"Loaded models: {sorted(available.keys())}. "
                   f"For 'custom', set CHECKPOINT_GCS_PATH before startup. "
                   f"For 'base', ensure weights/effort_clip_L14_trainOn_FaceForensic.pth exists."
        )

    model = available[model_type]
    weights_path = request.app.state.loaded_weights_paths.get(model_type)
    logger.info(f"Using '{model_type}' model for inference: {weights_path}")
    return model


# ──────────────────────────────────────────
# Health-check
# ──────────────────────────────────────────
@app.get("/ping")
def ping() -> dict:
    return {"message": "pong"}


# ──────────────────────────────────────────
# Inference Endpoints
# ──────────────────────────────────────────
@app.post("/check_frame", response_model=InferResponse)
async def check_frame(
        request: Request,
        file: UploadFile = File(...),
        model_type: Optional[str] = Query(None, description="REQUIRED: 'base' or 'custom'. No default."),
        threshold: float = Query(0.5, ge=0.0, le=1.0, description="Threshold for FAKE/REAL classification"),
        yolo_conf_threshold: float = Query(0.20, ge=0.0, le=1.0, description="YOLO confidence threshold for face detection"),
        recrop: bool = Query(False, description="Whether to perform face detection and cropping. If False, assumes image is already cropped"),
        debug: bool = False
) -> InferResponse:
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only JPEG or PNG images are accepted")

    if recrop:
        require_yolo(request)

    try:
        model = get_model_for_request(request, model_type)
        raw = await file.read()
        img_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Cannot decode image")

        # Quality gate (deployment packet 2026-05-04): reject thumbnails / extreme-blur
        # frames before model inference. Gated frames return prob=0.25 (REAL by default
        # threshold=0.5), and the gate hit is logged for debugging / audit.
        gate_passes, gate_reason = quality_gate(img_bgr, frame_id=file.filename or "frame")
        if not gate_passes:
            logger.info(
                "[QUALITY-GATE] /check_frame REJECTED %s (%s) → returning prob=%.2f",
                file.filename or "[unnamed]", gate_reason, QUALITY_GATE_DEFAULT_PROB,
            )
            return InferResponse(pred_label="REAL", fake_prob=QUALITY_GATE_DEFAULT_PROB)

        if recrop:
            processed_face_bgr = video_preprocessor.extract_yolo_face(img_bgr, yolo_conf_threshold)
            if processed_face_bgr is None:
                # is_no_face branch of the quality gate (YOLO found nothing).
                logger.info(
                    "[QUALITY-GATE] /check_frame REJECTED %s (no_face_detected) → returning prob=%.2f",
                    file.filename or "[unnamed]", QUALITY_GATE_DEFAULT_PROB,
                )
                return InferResponse(pred_label="REAL", fake_prob=QUALITY_GATE_DEFAULT_PROB)
        else:
            # INTER_LINEAR matches training preprocessing (combined_paired.py:3518).
            processed_face_bgr = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)

        if debug:
            os.makedirs(DEBUG_FRAME_DIR, exist_ok=True)
            timestamp = int(time.time() * 1000)
            crop_status = "cropped" if recrop else "precropped"
            save_path = os.path.join(DEBUG_FRAME_DIR, f"frame_{crop_status}_{timestamp}.jpg")
            cv2.imwrite(save_path, processed_face_bgr)
            logger.info(f"Debug frame saved to: {save_path}")

        transform = video_preprocessor._get_transform()
        rgb_face = cv2.cvtColor(processed_face_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = transform(rgb_face).unsqueeze(0).to(device)

        with torch.inference_mode():
            preds = model({'image': image_tensor}, inference=True)
            prob = preds["prob"].squeeze().cpu().item()
            pred_label = "FAKE" if prob >= threshold else "REAL"

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Inference failed for frame.")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Model inference failed.") from e

    logger.info("Frame inference result: label=%s, fake_prob=%.4f, threshold=%.2f", pred_label, prob, threshold)
    return InferResponse(pred_label=pred_label, fake_prob=prob)


@app.post("/check_frame_batch", response_model=BatchInferResponse)
async def check_frame_batch(
        request: Request,
        files: List[UploadFile] = File(...),
        model_type: Optional[str] = Query(None, description="REQUIRED: 'base' or 'custom'. No default."),
        threshold: float = Query(0.5, ge=0.0, le=1.0, description="Threshold for FAKE/REAL classification"),
        yolo_conf_threshold: float = Query(0.20, ge=0.0, le=1.0, description="YOLO confidence threshold for face detection"),
        recrop: bool = Query(False, description="Whether to perform face detection and cropping. If False, assumes frames are already cropped"),
        debug: bool = False
) -> BatchInferResponse:
    """
    Accepts a batch of frames (JPEG/PNG), runs the same pipeline as /check_frame on each,
    and returns the 'mean' strategy result over successfully processed frames:
      - confidence: mean of per-frame fake probabilities (from successful frames only)
      - probs: list of per-frame fake probabilities (from successful frames only)
    If no frames can be processed, returns pred_label="REAL", confidence=0.0, probs=[]
    
    Parameters:
    - recrop: If True, performs YOLO face detection and cropping. If False, assumes frames are already cropped.
    """
    if not files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No files were uploaded.")

    if recrop:
        require_yolo(request)

    # Validate content-types early
    for f in files:
        if f.content_type not in {"image/jpeg", "image/png"}:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only JPEG or PNG images are accepted")

    try:
        model = get_model_for_request(request, model_type)

        # Prepare transform once
        transform = video_preprocessor._get_transform()

        # Per-frame status tracking. The model is run only on frames that pass
        # the quality gate; gated frames get prob=QUALITY_GATE_DEFAULT_PROB.
        # Final probs list is in input order, with one entry per successfully
        # decoded file (gated + model-scored).
        per_frame_status = []   # list of dicts: {kind: 'gated'|'tensor'|'failed', tensor?, prob?, reason?}
        failed_frames = 0
        gated_frames = 0
        total_frames = len(files)

        for i, f in enumerate(files):
            try:
                raw = await f.read()
                img_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    logger.warning(f"Frame {i+1}/{total_frames}: Cannot decode image: {f.filename or '[unnamed]'}")
                    per_frame_status.append({"kind": "failed"})
                    failed_frames += 1
                    continue

                # Quality gate: reject thumbnails / extreme blur before model
                gate_passes, gate_reason = quality_gate(img_bgr, frame_id=f.filename or f"frame_{i+1}")
                if not gate_passes:
                    logger.info(
                        "[QUALITY-GATE] /check_frame_batch frame %d/%d REJECTED %s (%s) → prob=%.2f",
                        i + 1, total_frames, f.filename or "[unnamed]", gate_reason, QUALITY_GATE_DEFAULT_PROB,
                    )
                    per_frame_status.append({"kind": "gated", "prob": QUALITY_GATE_DEFAULT_PROB, "reason": gate_reason})
                    gated_frames += 1
                    continue

                if recrop:
                    # Same face extraction path as /check_frame (YOLO)
                    processed_face_bgr = video_preprocessor.extract_yolo_face(img_bgr, yolo_conf_threshold)
                    if processed_face_bgr is None:
                        # is_no_face branch — gate-equivalent: emit default-real instead of failing
                        logger.info(
                            "[QUALITY-GATE] /check_frame_batch frame %d/%d REJECTED %s (no_face_detected) → prob=%.2f",
                            i + 1, total_frames, f.filename or "[unnamed]", QUALITY_GATE_DEFAULT_PROB,
                        )
                        per_frame_status.append({"kind": "gated", "prob": QUALITY_GATE_DEFAULT_PROB, "reason": "no_face_detected"})
                        gated_frames += 1
                        continue
                else:
                    # Use the frame as-is, assuming it's already cropped, but resize to model input size.
                    # INTER_LINEAR matches training preprocessing (combined_paired.py:3518).
                    processed_face_bgr = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)

                if debug:
                    os.makedirs(DEBUG_FRAME_DIR, exist_ok=True)
                    timestamp = int(time.time() * 1000)
                    crop_status = "cropped" if recrop else "precropped"
                    save_path = os.path.join(DEBUG_FRAME_DIR, f"batch_frame_{i+1}_{crop_status}_{timestamp}.jpg")
                    cv2.imwrite(save_path, processed_face_bgr)
                    logger.info(f"Debug frame saved to: {save_path}")

                # To tensor (same as /check_frame) - convert to RGB and apply normalization
                rgb_face = cv2.cvtColor(processed_face_bgr, cv2.COLOR_BGR2RGB)
                image_tensor = transform(rgb_face).unsqueeze(0)  # (1, C, H, W)
                per_frame_status.append({"kind": "tensor", "tensor": image_tensor})

            except Exception as e:
                logger.warning(f"Frame {i+1}/{total_frames}: Processing failed: {e}")
                per_frame_status.append({"kind": "failed"})
                failed_frames += 1
                continue

        # Run the model on the model-scored subset (if any), then assemble
        # final per-frame probs list in original input order.
        tensor_indices = [i for i, s in enumerate(per_frame_status) if s["kind"] == "tensor"]
        model_probs = []
        if tensor_indices:
            tensors = [per_frame_status[i]["tensor"] for i in tensor_indices]
            batch_tensor = torch.cat(tensors, dim=0).to(device)  # (N, C, H, W)
            with torch.inference_mode():
                preds = model({'image': batch_tensor}, inference=True)
                raw_probs = preds["prob"].detach().squeeze().cpu().numpy().tolist()
            if isinstance(raw_probs, float):
                model_probs = [float(raw_probs)]
            else:
                model_probs = [float(p) for p in raw_probs]
            for idx, prob in zip(tensor_indices, model_probs):
                per_frame_status[idx]["prob"] = prob

        # Final probs list — gated frames get QUALITY_GATE_DEFAULT_PROB; failed-decode frames are dropped.
        probs_list = [s["prob"] for s in per_frame_status if s["kind"] in ("gated", "tensor")]
        successful_frames = sum(1 for s in per_frame_status if s["kind"] == "tensor")

        # Handle case where no frames were processed (all decode-failed)
        if not probs_list:
            logger.info(
                f"No frames could be processed. Failed: {failed_frames}/{total_frames}, gated: {gated_frames}"
            )
            return BatchInferResponse(pred_label="REAL", confidence=0.0, probs=[])

        # 'mean' strategy across the union of model-scored and gated frames.
        # Gated frames at 0.25 will pull confidence DOWN, which is the desired
        # behavior (uncertain inputs default to real).
        confidence = float(np.mean(probs_list))
        pred_label = "FAKE" if confidence >= threshold else "REAL"

        logger.info(
            f"Batch inference complete: {successful_frames}/{total_frames} model-scored, "
            f"{gated_frames} gated (defaulted to {QUALITY_GATE_DEFAULT_PROB}), {failed_frames} decode-failed"
        )

        pretty_print_batch(files, per_frame_status, confidence, threshold, pred_label)

        return BatchInferResponse(pred_label=pred_label, confidence=confidence, probs=probs_list)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Batch inference failed.")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Batch inference failed.") from e


@app.post("/check_video", response_model=VideoAnalysisResponse)
async def check_video(
        request: Request,
        file: UploadFile = File(...),
        model_type: Optional[str] = Query(None, description="REQUIRED: 'base' or 'custom'. No default."),
        threshold: float = Query(0.5, ge=0.0, le=1.0, description="Threshold for FAKE/REAL classification"),
        debug: bool = False,
        debug_frames_count: int = Query(2, ge=1, description="Number of frames to save when debug is enabled")
) -> VideoAnalysisResponse:
    # NOTE (2026-05-04): per-frame quality gate is NOT applied here. The
    # video_preprocessor returns a tensor of already-cropped 224×224 faces, so
    # the min(W,H)<150 check would fire on every frame (false positive). To
    # gate per-frame inside video, video_preprocessor.py needs to expose the
    # pre-resize face crop and laplacian_var per frame; left as TODO. For now,
    # video endpoints score every extracted face through the model.
    ext = Path(file.filename).suffix.lower()
    if ext not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported video format {ext!r}")

    require_yolo(request)

    tmp_dir = tempfile.mkdtemp(prefix="effort-aigi-")
    try:
        model = get_model_for_request(request, model_type)
        tmp_path = Path(tmp_dir) / file.filename
        with tmp_path.open("wb") as fp:
            shutil.copyfileobj(file.file, fp)

        debug_path = DEBUG_FRAME_DIR if debug else None
        video_tensor = video_preprocessor.preprocess_video_for_effort_model(
            str(tmp_path), pre_method="yolo", debug_save_path=debug_path, debug_frames_count=debug_frames_count if debug else None
        )

        if video_tensor is None or video_tensor.shape[1] == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Video could not be processed. This can happen if the video is too short or if a face cannot be consistently detected."
            )

        with torch.inference_mode():
            preds = model({'image': video_tensor.to(device)}, inference=True)
            frame_probs = preds["prob"].cpu().numpy().tolist()

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Video processing or inference failed.")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Video processing or inference failed.") from e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("Video inference complete: frames_processed=%d, threshold=%.2f", len(frame_probs), threshold)

    return calculate_analysis(frame_probs, threshold)


@app.post("/check_video_from_gcp", response_model=VideoAnalysisResponse)
async def check_video_from_gcp(
        request_body: GCSPathRequest,
        request: Request,
        model_type: Optional[str] = Query(None, description="REQUIRED: 'base' or 'custom'. No default."),
        threshold: float = Query(0.5, ge=0.0, le=1.0, description="Threshold for FAKE/REAL classification"),
        debug: bool = False,
        debug_frames_count: int = Query(2, ge=1, description="Number of frames to save when debug is enabled")
) -> VideoAnalysisResponse:
    # See /check_video for the per-frame quality-gate TODO. Same applies here.
    gcs_full_path = request_body.gcs_path
    logger.info(f"Received request to process video from GCS: {gcs_full_path}")

    require_yolo(request)

    try:
        bucket_name, blob_name = gcs_full_path.split('/', 1)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "Invalid GCS path format. Expected 'bucket-name/path/to/file'.")

    tmp_dir = tempfile.mkdtemp(prefix="effort-aigi-gcs-")
    try:
        model = get_model_for_request(request, model_type)
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        local_filename = Path(blob_name).name
        tmp_path = Path(tmp_dir) / local_filename

        logger.info(f"Downloading gs://{gcs_full_path} to {tmp_path}...")
        if not download_gcs_asset(bucket, blob_name, str(tmp_path), logger):
            raise exceptions.NotFound(f"File not found or failed to download from GCS at path: {gcs_full_path}")
        logger.info("Download complete.")

        debug_path = DEBUG_FRAME_DIR if debug else None
        video_tensor = video_preprocessor.preprocess_video_for_effort_model(
            str(tmp_path), pre_method="yolo", debug_save_path=debug_path, debug_frames_count=debug_frames_count if debug else None
        )

        if video_tensor is None or video_tensor.shape[1] == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Video could not be processed. This can happen if the video is too short or if a face cannot be consistently detected."
            )

        with torch.inference_mode():
            preds = model({'image': video_tensor.to(device)}, inference=True)
            frame_probs = preds["prob"].cpu().numpy().tolist()

    except HTTPException:
        raise
    except exceptions.NotFound as e:
        logger.error(f"GCS object not found: gs://{gcs_full_path}")
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))
    except Exception as e:
        logger.exception("GCS video processing or inference failed.")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "GCS video processing or inference failed.") from e
    finally:
        logger.info(f"Cleaning up temporary directory: {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("GCS video inference complete: frames_processed=%d, threshold=%.2f", len(frame_probs), threshold)

    return calculate_analysis(frame_probs, threshold)


@app.post("/check_gcs_frame_batch", response_model=VideoAnalysisResponse)
async def check_gcs_frame_batch(
        request_body: GCSPathRequest,
        request: Request,
        model_type: Optional[str] = Query(None, description="REQUIRED: 'base' or 'custom'. No default."),
        threshold: float = Query(0.5, ge=0.0, le=1.0, description="Threshold for FAKE/REAL classification"),
) -> VideoAnalysisResponse:
    gcs_dir_path = request_body.gcs_path
    if not gcs_dir_path.endswith('/'):
        gcs_dir_path += '/'
    logger.info(f"Received request to process frame batch from GCS: {gcs_dir_path}")

    try:
        bucket_name, dir_name = gcs_dir_path.split('/', 1)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "Invalid GCS path format. Expected 'bucket-name/path/to/directory/'.")

    tmp_dir = tempfile.mkdtemp(prefix="effort-aigi-gcs-batch-")
    try:
        model = get_model_for_request(request, model_type)
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        logger.info(f"Downloading frames from gs://{gcs_dir_path} to {tmp_dir}...")
        if not download_gcs_asset(bucket, dir_name, tmp_dir, logger):
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No image files found in GCS at path: {gcs_dir_path}")
        logger.info("Download complete.")

        image_files = sorted([p for p in Path(tmp_dir).glob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        if not image_files:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                f"No valid image files found after downloading from {gcs_dir_path}")

        frame_probs = []
        gated_count = 0
        transform = video_preprocessor._get_transform()
        with torch.inference_mode():
            for img_path in image_files:
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    logger.warning(f"Could not read image file: {img_path}, skipping.")
                    continue

                # Quality gate: reject thumbnails / extreme blur before model.
                gate_passes, gate_reason = quality_gate(img_bgr, frame_id=img_path.name)
                if not gate_passes:
                    logger.info(
                        "[QUALITY-GATE] /check_gcs_frame_batch REJECTED %s (%s) → prob=%.2f",
                        img_path.name, gate_reason, QUALITY_GATE_DEFAULT_PROB,
                    )
                    frame_probs.append(QUALITY_GATE_DEFAULT_PROB)
                    gated_count += 1
                    continue

                # ASSUMPTION: Frames are pre-cropped face images. Resize to model
                # input size with INTER_LINEAR (matches training preprocessing per
                # combined_paired.py:3518). Previously the resize was missing —
                # any non-224×224 input would interpolate positional embeddings
                # in the ViT and produce undefined behavior.
                resized_bgr = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
                rgb_face = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
                image_tensor = transform(rgb_face).unsqueeze(0).to(device)

                preds = model({'image': image_tensor}, inference=True)
                prob = preds["prob"].squeeze().cpu().item()
                frame_probs.append(prob)

        if gated_count:
            logger.info(
                "GCS frame batch: %d/%d frames gated (defaulted to %.2f)",
                gated_count, len(image_files), QUALITY_GATE_DEFAULT_PROB,
            )

    except HTTPException:
        raise
    except exceptions.NotFound as e:
        logger.error(f"GCS directory not found: gs://{gcs_dir_path}")
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))
    except Exception as e:
        logger.exception("GCS frame batch processing or inference failed.")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "GCS frame batch processing failed.") from e
    finally:
        logger.info(f"Cleaning up temporary directory: {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("GCS frame batch inference complete: frames_processed=%d, threshold=%.2f", len(frame_probs), threshold)
    return calculate_analysis(frame_probs, threshold)


# ──────────────────────────────────────────
# Global Exception Handler
# ──────────────────────────────────────────
@app.exception_handler(Exception)
async def unhandled_exception_handler(req: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred"},
    )
