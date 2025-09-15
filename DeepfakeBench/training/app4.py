# app4.py

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict
import tempfile
import shutil
import time

import cv2
import numpy as np
import torch
import yaml
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from torch import nn

import video_preprocessor
from detectors import DETECTOR, EffortDetector
from google.cloud import storage
from google.api_core import exceptions
from google.cloud.storage import Bucket

# ──────────────────────────────────────────
# Logging
# ──────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("effort-aigi-api-v4")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG_FRAME_DIR = "./debug_frames"
MIN_VIDEO_FRAMES = 8


# ──────────────────────────────────────────
# GCS Asset Downloading Utilities
# ──────────────────────────────────────────
def download_gcs_asset(bucket: Bucket, gcs_path: str, local_path: str, logger) -> bool:
    """Downloads a single blob or a directory of blobs from GCS."""
    if gcs_path.endswith('/'):  # It's a directory
        prefix = gcs_path.split(bucket.name + '/', 1)[1]
        blobs = bucket.list_blobs(prefix=prefix)
        downloaded = False
        for blob in blobs:
            if blob.name.endswith('/'): continue
            destination_file_name = os.path.join(local_path, os.path.relpath(blob.name, prefix))
            os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
            try:
                blob.download_to_filename(destination_file_name)
                downloaded = True
            except Exception as e:
                logger.error(f"Failed to download {blob.name}: {e}")
                return False
        if not downloaded:
            logger.error(f"Directory {gcs_path} is empty or does not exist.")
            return False
        return True
    else:  # It's a single file
        blob_name = gcs_path.split(bucket.name + '/', 1)[1]
        blob = bucket.blob(blob_name)
        if not blob.exists():
            logger.error(f"File not found at {gcs_path}")
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
    """Loads the EffortDetector model from config and weights."""
    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)
    ckpt = torch.load(weights, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ──────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────
app = FastAPI(title="Effort-AIGI Detector API v4", version="0.7.0")


# --- API Models ---
class FrameInferResponse(BaseModel):
    pred_label: str
    fake_prob: float


class Decision(BaseModel):
    label: str
    score: float


class MultiDecisionResponse(BaseModel):
    mean_decision: Decision
    median_decision: Decision
    majority_vote_decision: Decision
    frame_probs: Optional[List[float]] = None


class GCSVideoRequest(BaseModel):
    gcs_path: str


# ──────────────────────────────────────────
# Startup: Load Model(s) & Assert CUDA
# ──────────────────────────────────────────
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
        try:
            app.state.models['custom'] = load_detector(config, str(custom_weights_path))
            app.state.loaded_weights_paths['custom'] = str(custom_weights_path)
            logger.info(f"✅ SUCCESS: Custom detector model loaded from: {custom_weights_path}")
        except Exception as e:
            logger.exception("Failed to load CUSTOM detector model")
            raise e

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
@app.post("/check_frame", response_model=FrameInferResponse)
async def check_frame(
        request: Request,
        file: UploadFile = File(...),
        model_type: str = Query("base", description="Model to use: 'base' or 'custom'"),
        threshold: float = Query(0.5, ge=0.0, le=1.0, description="Threshold for REAL/FAKE classification"),
        debug: bool = False
) -> FrameInferResponse:
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only JPEG or PNG images are accepted")

    try:
        model = get_model_for_request(request, model_type)
        raw = await file.read()
        img_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Cannot decode image")

        processed_face_bgr = video_preprocessor.extract_yolo_face(img_bgr)

        if processed_face_bgr is None:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                "Could not find a face in the image using the 'yolo' method")

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

    logger.info("Frame inference result: label=%s, fake_prob=%.4f", pred_label, prob)
    return FrameInferResponse(pred_label=pred_label, fake_prob=prob)


def _calculate_video_decisions(frame_probs: List[float], threshold: float) -> tuple:
    """Helper to calculate mean, median, and majority vote decisions."""
    probs_arr = np.array(frame_probs)

    # Mean Decision
    mean_prob = np.mean(probs_arr)
    mean_label = "FAKE" if mean_prob >= threshold else "REAL"
    mean_decision = Decision(label=mean_label, score=float(mean_prob))

    # Median Decision
    median_prob = np.median(probs_arr)
    median_label = "FAKE" if median_prob >= threshold else "REAL"
    median_decision = Decision(label=median_label, score=float(median_prob))

    # Majority Vote Decision
    votes = (probs_arr >= threshold).astype(int)  # 1 for FAKE, 0 for REAL
    num_fake_votes = np.sum(votes)
    num_real_votes = len(votes) - num_fake_votes
    if num_fake_votes > num_real_votes:
        majority_label = "FAKE"
        majority_score = num_fake_votes / len(votes)
    else:  # REAL wins or it's a tie
        majority_label = "REAL"
        majority_score = num_real_votes / len(votes)
    majority_vote_decision = Decision(label=majority_label, score=majority_score)

    return mean_decision, median_decision, majority_vote_decision


@app.post("/check_video", response_model=MultiDecisionResponse, response_model_exclude_none=True)
async def check_video(
        request: Request,
        file: UploadFile = File(...),
        model_type: str = Query("base", description="Model to use: 'base' or 'custom'"),
        threshold: float = Query(0.5, ge=0.0, le=1.0, description="Threshold for REAL/FAKE classification"),
        return_probs: bool = Query(True, description="Whether to return the list of per-frame probabilities"),
        debug: bool = False
) -> MultiDecisionResponse:
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
            str(tmp_path), pre_method='yolo', debug_save_path=debug_path
        )

        num_frames_found = 0 if video_tensor is None else video_tensor.shape[1]
        if num_frames_found < MIN_VIDEO_FRAMES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Video could not be processed. Found {num_frames_found} faces, but a minimum of {MIN_VIDEO_FRAMES} is required."
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

    mean_decision, median_decision, majority_vote_decision = _calculate_video_decisions(frame_probs, threshold)

    logger.info(
        "Video inference complete: mean_label=%s (%.4f), median_label=%s (%.4f), majority_label=%s (%.4f), frames=%d",
        mean_decision.label, mean_decision.score,
        median_decision.label, median_decision.score,
        majority_vote_decision.label, majority_vote_decision.score,
        len(frame_probs)
    )

    return MultiDecisionResponse(
        mean_decision=mean_decision,
        median_decision=median_decision,
        majority_vote_decision=majority_vote_decision,
        frame_probs=frame_probs if return_probs else None
    )


@app.post("/check_video_from_gcp", response_model=MultiDecisionResponse, response_model_exclude_none=True)
async def check_video_from_gcp(
        request_body: GCSVideoRequest,
        request: Request,
        model_type: str = Query("base", description="Model to use: 'base' or 'custom'"),
        threshold: float = Query(0.5, ge=0.0, le=1.0, description="Threshold for REAL/FAKE classification"),
        return_probs: bool = Query(True, description="Whether to return the list of per-frame probabilities"),
        debug: bool = False
) -> MultiDecisionResponse:
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
        blob = bucket.blob(blob_name)

        local_filename = Path(blob_name).name
        tmp_path = Path(tmp_dir) / local_filename

        logger.info(f"Downloading {blob_name} to {tmp_path}...")
        blob.download_to_filename(tmp_path)
        logger.info("Download complete.")

        debug_path = DEBUG_FRAME_DIR if debug else None
        video_tensor = video_preprocessor.preprocess_video_for_effort_model(
            str(tmp_path), pre_method='yolo', debug_save_path=debug_path
        )

        num_frames_found = 0 if video_tensor is None else video_tensor.shape[1]
        if num_frames_found < MIN_VIDEO_FRAMES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Video could not be processed. Found {num_frames_found} faces, but a minimum of {MIN_VIDEO_FRAMES} is required."
            )

        with torch.inference_mode():
            preds = model({'image': video_tensor.to(device)}, inference=True)
            frame_probs = preds["prob"].cpu().numpy().tolist()

    except HTTPException:
        raise
    except exceptions.NotFound:
        logger.error(f"GCS object not found: gs://{bucket_name}/{blob_name}")
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"File not found in GCS at path: gs://{bucket_name}/{blob_name}")
    except Exception as e:
        logger.exception("GCS video processing or inference failed.")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "GCS video processing or inference failed.") from e
    finally:
        logger.info(f"Cleaning up temporary directory: {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)

    mean_decision, median_decision, majority_vote_decision = _calculate_video_decisions(frame_probs, threshold)

    logger.info(
        "GCS video inference complete: mean_label=%s (%.4f), median_label=%s (%.4f), majority_label=%s (%.4f), frames=%d",
        mean_decision.label, mean_decision.score,
        median_decision.label, median_decision.score,
        majority_vote_decision.label, majority_vote_decision.score,
        len(frame_probs)
    )

    return MultiDecisionResponse(
        mean_decision=mean_decision,
        median_decision=median_decision,
        majority_vote_decision=majority_vote_decision,
        frame_probs=frame_probs if return_probs else None
    )


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
