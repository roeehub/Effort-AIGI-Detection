import os
import logging
import socket
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
import observability  # per-request capture → GCS (fail-open, never blocks inference)
from batch_assembly import assemble_probs_list  # pure helper: t5c index-aligned response probs (B1)
import live_log  # pure console-logging: IP badges, per-participant blocks, anomalies, rolling dashboard
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
# Quality gate profiles
# ──────────────────────────────────────────
# Per-frame gate that can run in one of two profiles:
#
#   legacy  — deployment packet 2026-05-04. min(W,H)>=80 AND laplacian_var>=8.
#             Frames that fail still VOTE with prob=0.25 (treated as REAL by
#             threshold=0.5), and the gate hit is logged for audit.
#
#   t5c     — ship spec 2026-05-14 (analysis/risky_remediation_2026-05-14/
#             A_SHIP_SPEC_T5C_2026-05-14.md). G1: face-detector must return a
#             face. G2: min(W,H)>=200. No laplacian check. Frames that fail
#             G1 or G2 are NOT scored — they do not vote in the per-identity
#             decision. is_fake_identity abstains when no frames pass.
#
# Default profile is set by env var GATE_PROFILE (defaults to "legacy" to
# preserve current behavior). Individual requests can override via the
# ?gate_profile= query parameter on supported endpoints.
# Sentinel probability returned for rejected/gated frames in t5c profile.
# Chosen as -1.0 because it is outside the valid model-output range [0, 1],
# so downstream consumers can filter it trivially with `p >= 0.0`. Callers
# can also count sentinels to know how many frames were rejected without any
# separate response field. NOTE: index alignment with the input file list
# is preserved in t5c profile precisely because gated frames now leave a
# sentinel in their slot — see /check_frame_batch for the contract.
GATE_SENTINEL_PROB = -1.0

GATE_PROFILES: Dict[str, Dict[str, Any]] = {
    "legacy": {
        "min_dim": 80,
        "min_lap_var": 8.0,
        "default_prob": 0.25,           # gated frames vote with this prob
        "exclude_gated_from_vote": False,
        "align_probs_to_input": False,  # decode-failure frames are dropped
    },
    "t5c": {
        # NOTE: ship spec said min_dim=200; relaxed in production after the
        # 2026-05-14 fine G2 sweep (combined-pool identity-correctness peaked
        # at 110-120, sub-200 reals were inflating the no-decision rate).
        # 2026-06-01: lowered 120→110 (the LOWER edge of that sweep-optimum band)
        # — real Teams-capture crops were arriving at 117-119px and getting fully
        # gated (the user was never scored); 110 admits them while staying inside
        # the validated band. Going below 110 is unsupported by the sweep.
        "min_dim": 110,
        "min_lap_var": None,            # laplacian check disabled
        "default_prob": GATE_SENTINEL_PROB,  # sentinel in probs (excluded from mean)
        "exclude_gated_from_vote": True,
        # Every input frame gets a slot in `probs` (real model prob, or sentinel
        # for gated/decode-failed). Downstream consumers can rely on
        # `len(probs) == len(input_files)` and use index-based participant
        # attribution safely.
        "align_probs_to_input": True,
    },
}

_env_profile = os.getenv("GATE_PROFILE", "legacy").lower().strip()
if _env_profile not in GATE_PROFILES:
    logger.warning(
        "Unknown GATE_PROFILE=%r — falling back to 'legacy'. Valid: %s",
        _env_profile, sorted(GATE_PROFILES.keys()),
    )
    _env_profile = "legacy"
GATE_PROFILE_DEFAULT = _env_profile
logger.info("Default gate profile: %s", GATE_PROFILE_DEFAULT)

# Kept for backwards compatibility with any callers / tests that import these
# directly. New code should call quality_gate(img, profile) and read from the
# profile dict.
QUALITY_GATE_DEFAULT_PROB = GATE_PROFILES["legacy"]["default_prob"]
QUALITY_GATE_MIN_DIM = GATE_PROFILES["legacy"]["min_dim"]
QUALITY_GATE_MIN_LAP_VAR = GATE_PROFILES["legacy"]["min_lap_var"]


def resolve_gate_profile(query_value: Optional[str]) -> str:
    """Pick the gate profile for one request. Query param overrides env default."""
    if query_value is None or query_value == "":
        return GATE_PROFILE_DEFAULT
    profile = query_value.lower().strip()
    if profile not in GATE_PROFILES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid gate_profile={query_value!r}. "
                   f"Must be one of {sorted(GATE_PROFILES.keys())}.",
        )
    return profile


def quality_gate(
    img_bgr: Optional[np.ndarray],
    profile: str = "legacy",
    frame_id: str = "frame",
) -> tuple:
    """Per-frame quality gate, dispatched by profile.

    Returns (passes, reason). When passes=False, the caller decides how to
    handle the frame — legacy profile votes with prob=0.25, t5c profile drops
    the frame from the vote. See GATE_PROFILES for the per-profile thresholds.
    """
    spec = GATE_PROFILES.get(profile)
    if spec is None:
        return False, f"unknown_gate_profile:{profile!r}"

    if img_bgr is None or img_bgr.size == 0:
        return False, "empty_or_none_image"

    h, w = img_bgr.shape[:2]
    min_dim = spec["min_dim"]
    if min_dim is not None and min(h, w) < min_dim:
        return False, f"min_dim={min(h, w)}<{min_dim}"

    min_lap_var = spec["min_lap_var"]
    if min_lap_var is not None:
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception as e:
            return False, f"gate_compute_failed:{e!r}"
        if lap_var < min_lap_var:
            return False, f"laplacian_var={lap_var:.2f}<{min_lap_var}"

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
    app.state.obs = None  # observability uploader (set in step 9; None = disabled)
    app.state.live = None  # live console logger (rich readout + rolling dashboard; set below)

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

    # 9) Observability uploader (per-request capture → GCS). Fail-open: never
    #    block model serving on observability init.
    try:
        if observability.OBS_ENABLED:
            static = {
                "checkpoint_paths": dict(app.state.loaded_weights_paths),
                "use_arcface": os.getenv("CUSTOM_MODEL_USE_ARCFACE", "false").lower() in ("true", "1", "t"),
                "device": str(device),
                "app_version": app.version,
                "git_sha": observability.GIT_SHA,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
            }
            app.state.obs = observability.build_uploader(static)
            app.state.obs.start()
            logger.info("✅ Observability uploader active (bucket=%s).", observability.OBS_BUCKET)
        else:
            logger.info("Observability disabled via OBS_ENABLED=false.")
    except Exception:
        logger.exception("⚠️  Observability init failed — serving inference without it.")
        app.state.obs = None

    # Live console logging (IP badges, per-participant blocks, anomalies, rolling
    # dashboard). Created unconditionally — it's pure stdout formatting, independent
    # of observability/GCS. Fail-open so a logging issue can never block startup.
    try:
        app.state.live = live_log.LiveLog(
            log_fn=logger.info,
            cfg=live_log.LogConfig.from_env(),
            pid_parser=observability.parse_pid_from_filename,
        )
        app.state.live.start()
        logger.info("✅ Live console logging active (rolling dashboard every %ss).",
                    app.state.live.cfg.dashboard_seconds)
    except Exception:
        logger.exception("⚠️  Live console logging init failed — serving without the rich readout.")
        app.state.live = None

    logger.info("Startup complete. Available models: %s, YOLO: %s",
                list(app.state.models.keys()), app.state.yolo_available)


@app.on_event("shutdown")
def shutdown_event() -> None:
    # NOTE: @app.on_event is deprecated in newer FastAPI but kept for
    # consistency with startup_event above (a lifespan migration is a separate
    # refactor that would touch the model-loading path). Best-effort drain of
    # the observability queue; SIGKILL (OOM/preemption) loss is accepted —
    # this is telemetry, not transactional data.
    obs = getattr(app.state, "obs", None)
    if obs is not None:
        try:
            obs.stop(observability.OBS_DRAIN_TIMEOUT_S)
        except Exception:
            logger.exception("Observability shutdown drain failed.")

    live = getattr(app.state, "live", None)
    if live is not None:
        try:
            live.stop()
        except Exception:
            logger.exception("Live console logging shutdown failed.")


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
    logger.debug(f"Using '{model_type}' model for inference: {weights_path}")
    return model


# ──────────────────────────────────────────
# Health-check
# ──────────────────────────────────────────
@app.get("/ping")
def ping() -> dict:
    return {"message": "pong"}


@app.get("/obs_stats")
def obs_stats() -> dict:
    """Observability uploader self-stats (queue depth, uploaded/failed/dropped)."""
    obs = getattr(app.state, "obs", None)
    return obs.stats() if obs is not None else {"enabled": False}


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
        gate_profile: Optional[str] = Query(None, description="Gate profile: 'legacy' or 't5c'. Defaults to GATE_PROFILE env var (legacy)."),
        debug: bool = False
) -> InferResponse:
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only JPEG or PNG images are accepted")

    if recrop:
        require_yolo(request)

    profile = resolve_gate_profile(gate_profile)
    spec = GATE_PROFILES[profile]
    # legacy: gated → fake_prob=0.25, pred_label="REAL".
    # t5c:    gated → fake_prob=GATE_SENTINEL_PROB (-1.0), pred_label="REAL".
    #         Callers can distinguish "gated/abstained" from a real model score
    #         via `fake_prob < 0`.
    gated_prob = spec["default_prob"]

    # Observability: build the capture record + a single frame slot. Fail-open —
    # if anything here errors the request is unaffected. One enqueue in `finally`
    # covers every exit path (decode-fail / gate-fail / no-face / success / error).
    _obs = getattr(request.app.state, "obs", None)
    _t0 = time.perf_counter()
    cap = None
    fcap = None
    if _obs is not None:
        try:
            cap = observability.new_record(
                request, "/check_frame", model_type=model_type, threshold=threshold,
                gate_profile=profile, gate_spec=spec, yolo_conf_threshold=yolo_conf_threshold,
                recrop=recrop, debug=debug,
            )
            fcap = observability.FrameCapture(
                seq=0, raw_bytes=b"", filename=file.filename, content_type=file.content_type,
            )
            # WMA encodes pid + per-pid seq into the multipart filename
            # (`pid=…__seq=…__frame_…`). Returns (None, None) for legacy
            # callers that don't carry the encoding — participant_id then
            # stays None and the uploader falls back to the flat key layout.
            fcap.participant_id, fcap.participant_seq = (
                observability.parse_pid_from_filename(file.filename)
            )
            cap.frames.append(fcap)
        except Exception:
            cap = fcap = None

    try:
        model = get_model_for_request(request, model_type)
        raw = await file.read()
        if fcap is not None:
            # Bound retained bytes so a giant upload can't pin memory (it's still scored).
            if _obs is not None and len(raw) > _obs.max_record_bytes:
                fcap.capture_skipped = True
            else:
                fcap.raw_bytes = raw
        img_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            if cap is not None:
                cap.status = "decode_failed"
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Cannot decode image")
        if fcap is not None:
            fcap.dims_hw = (int(img_bgr.shape[0]), int(img_bgr.shape[1]))

        gate_passes, gate_reason = quality_gate(img_bgr, profile=profile, frame_id=file.filename or "frame")
        if fcap is not None:
            fcap.gate_pass = bool(gate_passes)
            fcap.gate_reason = gate_reason
        if not gate_passes:
            logger.info(
                "[QUALITY-GATE:%s] /check_frame REJECTED %s (%s) → returning prob=%.2f",
                profile, file.filename or "[unnamed]", gate_reason, gated_prob,
            )
            if fcap is not None:
                fcap.prob = gated_prob
            if cap is not None:
                cap.status = "gate_failed"
                cap.pred_label = "REAL"
                cap.confidence = gated_prob
            return InferResponse(pred_label="REAL", fake_prob=gated_prob)

        if recrop:
            processed_face_bgr = video_preprocessor.extract_yolo_face(img_bgr, yolo_conf_threshold)
            if processed_face_bgr is None:
                # G1 failure: face detector returned no face.
                logger.info(
                    "[QUALITY-GATE:%s] /check_frame REJECTED %s (no_face_detected) → returning prob=%.2f",
                    profile, file.filename or "[unnamed]", gated_prob,
                )
                if fcap is not None:
                    fcap.face_found = False
                    fcap.prob = gated_prob
                if cap is not None:
                    cap.status = "no_face"
                    cap.pred_label = "REAL"
                    cap.confidence = gated_prob
                return InferResponse(pred_label="REAL", fake_prob=gated_prob)
            if fcap is not None:
                fcap.face_found = True
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

        if fcap is not None:
            fcap.prob = float(prob)
            fcap.scored = True
            fcap.verdict = pred_label
        if cap is not None:
            cap.status = "ok"
            cap.pred_label = pred_label
            cap.confidence = float(prob)

    except HTTPException:
        raise
    except Exception as e:
        if cap is not None:
            cap.status = "inference_error"
        logger.exception("Inference failed for frame.")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Model inference failed.") from e
    finally:
        if cap is not None:
            cap.latency_ms = (time.perf_counter() - _t0) * 1000.0
            observability._safe_enqueue(_obs, cap)

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
        gate_profile: Optional[str] = Query(None, description="Gate profile: 'legacy' or 't5c'. Defaults to GATE_PROFILE env var (legacy)."),
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

    profile = resolve_gate_profile(gate_profile)
    spec = GATE_PROFILES[profile]
    exclude_gated = spec["exclude_gated_from_vote"]
    align_to_input = spec["align_probs_to_input"]
    # In t5c profile this is the sentinel (-1.0). In legacy it's the 0.25 vote.
    gated_slot_prob = spec["default_prob"]

    # Observability: one record per request; one FrameCapture per input file
    # (kept 1:1 with `files`/`per_frame_status`). Fail-open; single enqueue in
    # `finally`. Defined before the try so the finally is always safe.
    _obs = getattr(request.app.state, "obs", None)
    _t0 = time.perf_counter()
    cap = None
    capture_frames: List[Any] = []
    captured_bytes = 0  # running total of retained originals (bounds in-handler memory)
    if _obs is not None:
        try:
            cap = observability.new_record(
                request, "/check_frame_batch", model_type=model_type, threshold=threshold,
                gate_profile=profile, gate_spec=spec, yolo_conf_threshold=yolo_conf_threshold,
                recrop=recrop, debug=debug,
            )
        except Exception:
            cap = None

    try:
        model = get_model_for_request(request, model_type)

        # Prepare transform once
        transform = video_preprocessor._get_transform()

        # Per-frame status tracking. The model is run only on frames that pass
        # the quality gate; in legacy profile gated frames vote with prob=0.25,
        # in t5c profile gated frames carry the sentinel (-1.0) so the response
        # `probs` list is 1:1 with the input file list.
        per_frame_status = []   # list of dicts: {kind: 'gated'|'tensor'|'failed', tensor?, prob?, reason?}
        failed_frames = 0
        gated_frames = 0
        total_frames = len(files)

        for i, f in enumerate(files):
            fc = None
            if cap is not None:
                fc = observability.FrameCapture(
                    seq=i, raw_bytes=b"", filename=f.filename, content_type=f.content_type,
                )
                # Per-frame pid extraction so the GCS capture partitions
                # by participant (see /check_frame for the contract).
                fc.participant_id, fc.participant_seq = (
                    observability.parse_pid_from_filename(f.filename)
                )
                capture_frames.append(fc)
            try:
                raw = await f.read()
                if fc is not None:
                    # Bound retained originals across the whole batch so a large
                    # batch can't pin unbounded memory (frames are still scored).
                    if _obs is not None and captured_bytes + len(raw) > _obs.max_record_bytes:
                        fc.capture_skipped = True
                    else:
                        fc.raw_bytes = raw
                        captured_bytes += len(raw)
                img_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    logger.warning(f"Frame {i+1}/{total_frames}: Cannot decode image: {f.filename or '[unnamed]'}")
                    if fc is not None:
                        fc.gate_reason = "decode_failed"
                    entry = {"kind": "failed"}
                    # In aligned profile (t5c), decode failures still get a slot
                    # with the sentinel so downstream index-based attribution
                    # is preserved.
                    if align_to_input:
                        entry["prob"] = GATE_SENTINEL_PROB
                        entry["reason"] = "decode_failed"
                    per_frame_status.append(entry)
                    failed_frames += 1
                    continue

                if fc is not None:
                    fc.dims_hw = (int(img_bgr.shape[0]), int(img_bgr.shape[1]))

                gate_passes, gate_reason = quality_gate(img_bgr, profile=profile, frame_id=f.filename or f"frame_{i+1}")
                if fc is not None:
                    fc.gate_pass = bool(gate_passes)
                    fc.gate_reason = gate_reason
                    if not gate_passes:
                        fc.prob = gated_slot_prob
                if not gate_passes:
                    logger.debug(
                        "[QUALITY-GATE:%s] /check_frame_batch frame %d/%d REJECTED %s (%s)%s",
                        profile, i + 1, total_frames, f.filename or "[unnamed]", gate_reason,
                        f" → sentinel={GATE_SENTINEL_PROB}" if exclude_gated else f" → prob={gated_slot_prob:.2f}",
                    )
                    per_frame_status.append({
                        "kind": "gated",
                        "reason": gate_reason,
                        "prob": gated_slot_prob,
                    })
                    gated_frames += 1
                    continue

                if recrop:
                    # Same face extraction path as /check_frame (YOLO)
                    processed_face_bgr = video_preprocessor.extract_yolo_face(img_bgr, yolo_conf_threshold)
                    if processed_face_bgr is None:
                        # G1 failure: face detector returned no face.
                        logger.debug(
                            "[QUALITY-GATE:%s] /check_frame_batch frame %d/%d REJECTED %s (no_face_detected)%s",
                            profile, i + 1, total_frames, f.filename or "[unnamed]",
                            f" → sentinel={GATE_SENTINEL_PROB}" if exclude_gated else f" → prob={gated_slot_prob:.2f}",
                        )
                        if fc is not None:
                            fc.face_found = False
                            fc.gate_reason = "no_face_detected"
                            fc.prob = gated_slot_prob
                        per_frame_status.append({
                            "kind": "gated",
                            "reason": "no_face_detected",
                            "prob": gated_slot_prob,
                        })
                        gated_frames += 1
                        continue
                    if fc is not None:
                        fc.face_found = True
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
                if fc is not None and not fc.gate_reason:
                    fc.gate_reason = "processing_failed"
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
                if cap is not None and idx < len(capture_frames):
                    capture_frames[idx].prob = float(prob)
                    capture_frames[idx].scored = True
                    capture_frames[idx].verdict = "FAKE" if prob >= threshold else "REAL"

        # Build the response probs list via the pure helper so the t5c alignment
        # contract is guaranteed by construction (see batch_assembly.py + tests):
        # - t5c (align_to_input=True): EVERY input frame gets one slot, in order.
        #   Real prob for scored frames; GATE_SENTINEL_PROB (-1.0) for ANY
        #   non-scored frame — gated, decode-failed, AND processing-failed (the
        #   except path below, which previously dropped its slot — B1). So
        #   len(probs) == len(files) and client-side positional per-participant
        #   attribution can't shift across pid boundaries.
        # - legacy (align_to_input=False): gated frames carry the 0.25 vote;
        #   decode/processing failures are dropped. Pre-existing contract.
        probs_list = assemble_probs_list(per_frame_status, align_to_input, GATE_SENTINEL_PROB)
        successful_frames = sum(1 for s in per_frame_status if s["kind"] == "tensor")

        # For the confidence mean, drop sentinel slots (they represent rejected
        # input, not a real verdict). In legacy profile no slot is a sentinel,
        # so the filter is a no-op.
        voting_probs = [p for p in probs_list if p >= 0.0]

        # Handle case where nothing voted (all gated/decode-failed in t5c, or
        # everything decode-failed in legacy).
        if not voting_probs:
            logger.info(
                "No frames available for scoring. Profile=%s. Failed: %d/%d, gated: %d (excluded=%s)",
                profile, failed_frames, total_frames, gated_frames, exclude_gated,
            )
            if cap is not None:
                cap.status = "no_voting_frames"
                cap.pred_label = "REAL"
                cap.confidence = 0.0
            return BatchInferResponse(pred_label="REAL", confidence=0.0, probs=probs_list)

        confidence = float(np.mean(voting_probs))
        pred_label = "FAKE" if confidence >= threshold else "REAL"

        gated_note = (
            f"{gated_frames} gated (sentinel)" if exclude_gated
            else f"{gated_frames} gated (defaulted to {gated_slot_prob})"
        )
        logger.debug(
            "Batch inference complete (profile=%s): %d/%d model-scored, %s, %d decode-failed, aligned=%s",
            profile, successful_frames, total_frames, gated_note, failed_frames, align_to_input,
        )

        live = getattr(request.app.state, "live", None)
        if live is not None:
            # IP = the same client identity the viewer sessionizes on; per-frame
            # pid is parsed from the filename inside log_batch (fail-open there).
            client_ip = observability.get_client_ip(request)[0]
            latency_ms = (time.perf_counter() - _t0) * 1000.0
            live.log_batch(
                [f.filename for f in files], per_frame_status, threshold, client_ip, latency_ms,
            )

        if cap is not None:
            cap.status = "ok"
            cap.pred_label = pred_label
            cap.confidence = confidence
        return BatchInferResponse(pred_label=pred_label, confidence=confidence, probs=probs_list)

    except HTTPException:
        raise
    except Exception as e:
        if cap is not None:
            cap.status = "inference_error"
        logger.exception("Batch inference failed.")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Batch inference failed.") from e
    finally:
        if cap is not None:
            cap.frames = capture_frames
            cap.latency_ms = (time.perf_counter() - _t0) * 1000.0
            observability._safe_enqueue(_obs, cap)


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
        gate_profile: Optional[str] = Query(None, description="Gate profile: 'legacy' or 't5c'. Defaults to GATE_PROFILE env var (legacy)."),
) -> VideoAnalysisResponse:
    gcs_dir_path = request_body.gcs_path
    if not gcs_dir_path.endswith('/'):
        gcs_dir_path += '/'
    logger.info(f"Received request to process frame batch from GCS: {gcs_dir_path}")

    profile = resolve_gate_profile(gate_profile)
    spec = GATE_PROFILES[profile]
    exclude_gated = spec["exclude_gated_from_vote"]
    gated_vote_prob = spec["default_prob"]

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

                gate_passes, gate_reason = quality_gate(img_bgr, profile=profile, frame_id=img_path.name)
                if not gate_passes:
                    logger.info(
                        "[QUALITY-GATE:%s] /check_gcs_frame_batch REJECTED %s (%s)%s",
                        profile, img_path.name, gate_reason,
                        " → excluded from vote" if exclude_gated else f" → prob={gated_vote_prob:.2f}",
                    )
                    if not exclude_gated:
                        frame_probs.append(gated_vote_prob)
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
            gated_note = (
                "excluded from vote" if exclude_gated
                else f"defaulted to {gated_vote_prob:.2f}"
            )
            logger.info(
                "GCS frame batch (profile=%s): %d/%d frames gated (%s)",
                profile, gated_count, len(image_files), gated_note,
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

    logger.info(
        "GCS frame batch inference complete (profile=%s): frames_processed=%d, threshold=%.2f",
        profile, len(frame_probs), threshold,
    )
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
