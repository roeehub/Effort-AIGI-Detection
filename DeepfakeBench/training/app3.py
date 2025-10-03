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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG_FRAME_DIR = "./debug_frames"


# ──────────────────────────────────────────
# GCS Asset Downloading Utilities
# ──────────────────────────────────────────
def download_gcs_asset(bucket: Bucket, gcs_path: str, local_path: str, logger) -> bool:
    """Downloads a single blob or a directory of blobs from GCS."""
    if not gcs_path.startswith('gs://'):
        # This function is now used for both assets and frame batches, so handle both formats
        prefix_to_strip = f"{bucket.name}/"
    else:
        prefix_to_strip = f"gs://{bucket.name}/"

    if gcs_path.endswith('/'):  # It's a directory
        prefix = gcs_path.replace(prefix_to_strip, '', 1)
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
                logger.error(f"Asset '{key}' is missing 'gcs_path' or 'local_path'.")
                return None
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
            if not download_gcs_asset(bucket, gcs_path, local_path, logger):
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
    
    # Load checkpoint
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
                if key != 'current_arcface_s':  # Skip dynamic parameter
                    old_value = cfg.get(key)
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
    
    # Initialize model with (possibly updated) config
    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)
    
    # Restore dynamic ArcFace parameter if available
    if model_config.get('use_arcface_head', False) and 'current_arcface_s' in model_config:
        if hasattr(model, 'head') and hasattr(model.head, 's'):
            current_s = model_config['current_arcface_s']
            model.head.s.data.fill_(current_s)
            logger.info(f"  Restored ArcFace s parameter: {current_s}")
    
    # Load state dict with module prefix handling
    state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
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

    # 1) CUDA Check
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This service requires a GPU.")
        raise RuntimeError("CUDA is required for this service")
    logger.info("CUDA is available. Using device: %s", device)

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

    # 6) Load Base Model
    logger.info("--- Loading Base Model ---")
    if not base_weights_path.exists():
        raise RuntimeError(f"Base model weights file not found: {base_weights_path}")
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

    # 8) Load Face Preprocessor Models (YOLO only)
    try:
        video_preprocessor.initialize_yolo_model()
        logger.info("✅ SUCCESS: YOLO face detector loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load YOLO model")
        raise RuntimeError("Failed to load YOLO model") from e

    logger.info("Startup complete. Available models: %s", list(app.state.models.keys()))


# --- Utility function to get model for endpoints ---
def get_model_for_request(request: Request, model_type: str) -> nn.Module:
    """Gets the requested model from app state and handles errors."""
    if model_type not in ["base", "custom"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid 'model_type'. Choose 'base' or 'custom'."
        )
    if model_type == "custom" and "custom" not in request.app.state.models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Custom model is not available. It must be configured via CHECKPOINT_GCS_PATH at startup."
        )

    model = request.app.state.models.get(model_type)
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
        model_type: str = Query("base", description="Model to use: 'base' or 'custom'"),
        threshold: float = Query(0.5, ge=0.0, le=1.0, description="Threshold for FAKE/REAL classification"),
        yolo_conf_threshold: float = Query(0.20, ge=0.0, le=1.0, description="YOLO confidence threshold for face detection"),
        debug: bool = False
) -> InferResponse:
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only JPEG or PNG images are accepted")

    try:
        model = get_model_for_request(request, model_type)
        raw = await file.read()
        img_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Cannot decode image")

        processed_face_bgr = video_preprocessor.extract_yolo_face(img_bgr, yolo_conf_threshold)

        if processed_face_bgr is None:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                f"Could not find a face in the image using the 'yolo' method")

        if debug:
            os.makedirs(DEBUG_FRAME_DIR, exist_ok=True)
            timestamp = int(time.time() * 1000)
            save_path = os.path.join(DEBUG_FRAME_DIR, f"frame_{timestamp}.jpg")
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
        model_type: str = Query("base", description="Model to use: 'base' or 'custom'"),
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

    # Validate content-types early
    for f in files:
        if f.content_type not in {"image/jpeg", "image/png"}:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only JPEG or PNG images are accepted")

    try:
        model = get_model_for_request(request, model_type)

        # Prepare transform once
        transform = video_preprocessor._get_transform()

        tensors = []
        failed_frames = 0
        total_frames = len(files)

        for i, f in enumerate(files):
            try:
                raw = await f.read()
                img_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    logger.warning(f"Frame {i+1}/{total_frames}: Cannot decode image: {f.filename or '[unnamed]'}")
                    failed_frames += 1
                    continue

                if recrop:
                    # Same face extraction path as /check_frame (YOLO)
                    processed_face_bgr = video_preprocessor.extract_yolo_face(img_bgr, yolo_conf_threshold)
                    if processed_face_bgr is None:
                        logger.warning(f"Frame {i+1}/{total_frames}: Could not find a face in the image using the 'yolo' method: {f.filename or '[unnamed]'}")
                        failed_frames += 1
                        continue
                else:
                    # Use the frame as-is, assuming it's already cropped, but resize to model input size
                    processed_face_bgr = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)

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
                tensors.append(image_tensor)

            except Exception as e:
                logger.warning(f"Frame {i+1}/{total_frames}: Processing failed: {e}")
                failed_frames += 1
                continue

        # Handle case where no frames were successfully processed
        if not tensors:
            logger.info(f"No frames could be processed successfully. Failed: {failed_frames}/{total_frames}")
            return BatchInferResponse(pred_label="REAL", confidence=0.0, probs=[])

        # Batch the frames to a single forward pass when possible
        batch_tensor = torch.cat(tensors, dim=0).to(device)  # (N, C, H, W)
        successful_frames = len(tensors)

        with torch.inference_mode():
            preds = model({'image': batch_tensor}, inference=True)  # same call signature as /check_frame
            # Expect preds["prob"] to be shape (N,) or (N,1)
            probs = preds["prob"].detach().squeeze().cpu().numpy().tolist()

        # Normalize to list[float]
        if isinstance(probs, float):
            probs_list = [float(probs)]
        else:
            probs_list = [float(p) for p in probs]

        # 'mean' strategy over successfully processed frames
        confidence = float(np.mean(probs_list)) if probs_list else 0.0
        pred_label = "FAKE" if confidence >= threshold else "REAL"

        logger.info(f"Batch inference complete: {successful_frames}/{total_frames} frames processed successfully, {failed_frames} failed")

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
        model_type: str = Query("base", description="Model to use: 'base' or 'custom'"),
        threshold: float = Query(0.75, ge=0.0, le=1.0, description="Threshold for FAKE/REAL classification"),
        debug: bool = False,
        debug_frames_count: int = Query(2, ge=1, description="Number of frames to save when debug is enabled")
) -> VideoAnalysisResponse:
    ext = Path(file.filename).suffix.lower()
    if ext not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported video format {ext!r}")

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
        model_type: str = Query("base", description="Model to use: 'base' or 'custom'"),
        threshold: float = Query(0.5, ge=0.0, le=1.0, description="Threshold for FAKE/REAL classification"),
        debug: bool = False,
        debug_frames_count: int = Query(2, ge=1, description="Number of frames to save when debug is enabled")
) -> VideoAnalysisResponse:
    gcs_full_path = request_body.gcs_path
    logger.info(f"Received request to process video from GCS: {gcs_full_path}")

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
        model_type: str = Query("base", description="Model to use: 'base' or 'custom'"),
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
        transform = video_preprocessor._get_transform()
        with torch.inference_mode():
            for img_path in image_files:
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    logger.warning(f"Could not read image file: {img_path}, skipping.")
                    continue

                # ASSUMPTION: Frames are pre-cropped, so we don't run face detection
                rgb_face = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                image_tensor = transform(rgb_face).unsqueeze(0).to(device)

                preds = model({'image': image_tensor}, inference=True)
                prob = preds["prob"].squeeze().cpu().item()
                frame_probs.append(prob)

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
