import os
import logging
from pathlib import Path
from typing import Optional, List
import tempfile
import shutil
import time

import cv2  # noqa
import numpy as np  # noqa
import torch  # noqa
import yaml  # noqa
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Request  # noqa
from fastapi.responses import JSONResponse  # noqa
from pydantic import BaseModel  # noqa
from torch import nn  # noqa

# --- New Imports ---
# Import the new, standardized preprocessing functions.
import video_preprocessor
# Assuming detectors.py contains the model class and DETECTOR registry.
from detectors import DETECTOR, EffortDetector  # noqa

# --- ADDED IMPORTS for GCS Asset Handling ---
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
logger = logging.getLogger("effort-aigi-api")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG_FRAME_DIR = "./debug_frames"


# ──────────────────────────────────────────
# GCS Asset Downloading Utilities (Copied from train_sweep.py)
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
# Model Loading (Now uses a merged config)
# ──────────────────────────────────────────
def load_detector(cfg: dict, weights: str) -> nn.Module:
    """Loads the EffortDetector model from config and weights."""
    model_cls = DETECTOR[cfg["model_name"]]
    # The config 'cfg' now contains the 'gcs_assets' key required by the model
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
app = FastAPI(title="Effort-AIGI Detector API", version="0.2.1")  # Version bump


# --- API Response Models ---
class InferResponse(BaseModel):
    pred_label: str
    fake_prob: float


class VideoInferResponse(BaseModel):
    pred_label: str
    fake_prob: float
    frame_probs: Optional[List[float]] = None


# ──────────────────────────────────────────
# Startup: Load Model & Assert CUDA
# ──────────────────────────────────────────
@app.on_event("startup")
def startup_event() -> None:
    # 1) CUDA Check
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This service requires a GPU.")
        raise RuntimeError("CUDA is required for this service")
    logger.info("CUDA is available. Using device: %s", device)

    # 2) Define paths to configs and default weights
    repo_base = Path("/home/roee/repos/Effort-AIGI-Detection-Fork/DeepfakeBench/training")
    cfg_path = repo_base / "config/detector/effort.yaml"
    train_cfg_path = repo_base / "config/train_config.yaml"
    weights_path = repo_base / "weights/effort_clip_L14_trainOn_FaceForensic.pth"  # Default

    # Check for required config files
    if not all([cfg_path.exists(), train_cfg_path.exists()]):
        logger.error("Missing one or more required config files: effort.yaml or train_config.yaml.")
        raise RuntimeError("A required configuration file was not found.")

    # 3) Load and merge configurations
    try:
        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)
        with open(train_cfg_path, "r") as f:
            train_config = yaml.safe_load(f)
        config.update(train_config)
        logger.info("Successfully loaded and merged configuration files.")
    except Exception:
        logger.exception("Failed to load or merge YAML configuration files.")
        raise

    # 4) --- NEW: Handle custom checkpoint from environment variables ---
    custom_gcs_bucket = os.getenv("GCS_BUCKET")
    custom_checkpoint_path = os.getenv("CHECKPOINT_PATH")

    if custom_gcs_bucket and custom_checkpoint_path:
        logger.info("Custom checkpoint specified via environment variables.")
        full_gcs_path = f"gs://{custom_gcs_bucket}/{custom_checkpoint_path}"

        # Define a local path for the custom checkpoint to be saved to
        local_filename = Path(custom_checkpoint_path).name
        custom_weights_dir = repo_base / "weights" / "custom"
        custom_weights_dir.mkdir(parents=True, exist_ok=True)
        new_weights_path = custom_weights_dir / local_filename

        logger.info(f"  GCS Path: {full_gcs_path}")
        logger.info(f"  Local Path: {new_weights_path}")

        # Add this custom checkpoint to the asset download list
        config.setdefault('gcs_assets', {})['custom_checkpoint'] = {
            'gcs_path': full_gcs_path,
            'local_path': str(new_weights_path)
        }

        # Update the main weights_path variable to use the custom one
        weights_path = new_weights_path
        logger.info("Overriding default weights with custom checkpoint.")
    else:
        logger.info("Using default base checkpoint.")
        if not weights_path.exists():
            logger.warning(
                f"Default weight file not found at {weights_path}. The API will fail if it's not defined as a GCS asset in the config.")

    # 5) Download all required GCS Assets
    if not download_assets_from_gcs(config, logger):
        logger.error("Could not download required GCS assets. See logs for details.")
        raise RuntimeError("Failed to prepare model assets from GCS.")

    # 6) Final check that the target weights file exists before loading
    if not Path(weights_path).exists():
        logger.error(f"Target weights file does not exist after GCS download attempt: {weights_path}")
        raise RuntimeError("Model weights file is missing.")

    # 7) Load the PyTorch model
    try:
        app.state.model = load_detector(config, str(weights_path))
        logger.info(f"Detector model loaded successfully from: {weights_path}")
    except Exception:
        logger.exception("Failed to load detector model")
        raise

    logger.info("Startup complete. Dlib models will be loaded by the preprocessor on first use.")


# ──────────────────────────────────────────
# Health-check
# ──────────────────────────────────────────
@app.get("/ping")
def ping() -> dict:
    return {"message": "pong"}


# ──────────────────────────────────────────
# Inference Endpoints (Now using video_preprocessor)
# ──────────────────────────────────────────
@app.post("/check_frame", response_model=InferResponse)
async def check_frame(
        file: UploadFile = File(...),
        debug: bool = False
) -> InferResponse:
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only JPEG or PNG images are accepted")

    raw = await file.read()
    img_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Cannot decode image")

    aligned_face_bgr = video_preprocessor.extract_aligned_face(img_bgr)
    if aligned_face_bgr is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Could not find a face in the image")

    # --- ADDITION: Save debug frame if flag is set ---
    if debug:
        os.makedirs(DEBUG_FRAME_DIR, exist_ok=True)
        timestamp = int(time.time() * 1000)
        save_path = os.path.join(DEBUG_FRAME_DIR, f"frame_{timestamp}.jpg")
        cv2.imwrite(save_path, aligned_face_bgr)
        logger.info(f"Debug frame saved to: {save_path}")
    # --- END ADDITION ---

    transform = video_preprocessor._get_transform()
    rgb_face = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = transform(rgb_face).unsqueeze(0).to(device)

    try:
        with torch.inference_mode():
            preds = app.state.model({'image': image_tensor}, inference=True)
            prob = preds["prob"].squeeze().cpu().item()
            pred_label = "FAKE" if prob >= 0.5 else "REAL"

    except Exception as e:
        logger.exception("Inference failed for frame.")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Model inference failed.") from e

    logger.info("Frame inference result: label=%s, fake_prob=%.4f", pred_label, prob)
    return InferResponse(pred_label=pred_label, fake_prob=prob)


@app.post("/check_video", response_model=VideoInferResponse, response_model_exclude_none=True)
async def check_video(
        file: UploadFile = File(...),
        return_probs: bool = True,
        debug: bool = False
) -> VideoInferResponse:
    ext = Path(file.filename).suffix.lower()
    allowed_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    if ext not in allowed_exts:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported video format {ext!r}")

    tmp_dir = tempfile.mkdtemp(prefix="effort-aigi-")
    tmp_path = Path(tmp_dir) / file.filename
    try:
        with tmp_path.open("wb") as fp:
            shutil.copyfileobj(file.file, fp)

        # --- MODIFICATION: Conditionally pass the debug path ---
        debug_path = DEBUG_FRAME_DIR if debug else None
        video_tensor = video_preprocessor.preprocess_video_for_effort_model(
            str(tmp_path),
            debug_save_path=debug_path
        )
        # --- END MODIFICATION ---

        if video_tensor is None:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                "Video could not be processed. It may not contain enough frames with detectable faces.")

        with torch.inference_mode():
            preds = app.state.model({'image': video_tensor.to(device)}, inference=True)
            frame_probs = preds["prob"].cpu().numpy().tolist()

    except Exception as e:
        logger.exception("Video processing or inference failed.")
        # Re-raise as HTTPException to be caught by FastAPI's error handling
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Video processing or inference failed.") from e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    avg_prob = float(np.mean(frame_probs))
    pred_label = "FAKE" if avg_prob >= 0.5 else "REAL"

    logger.info("Video inference complete: label=%s, avg_fake_prob=%.4f, frames_processed=%d",
                pred_label, avg_prob, len(frame_probs))

    return VideoInferResponse(
        pred_label=pred_label,
        fake_prob=avg_prob,
        frame_probs=frame_probs if return_probs else None
    )


# ──────────────────────────────────────────
# Global Exception Handler
# ──────────────────────────────────────────
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred"},
    )
