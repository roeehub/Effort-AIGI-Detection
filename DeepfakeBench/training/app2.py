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
# Model Loading (Moved from infer.py for self-containment)
# ──────────────────────────────────────────
def load_detector(detector_cfg: str, weights: str) -> nn.Module:
    """Loads the EffortDetector model from config and weights."""
    with open(detector_cfg, "r") as f:
        cfg = yaml.safe_load(f)
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
app = FastAPI(title="Effort-AIGI Detector API", version="0.2.0")


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

    # 2) Define paths to model files
    # These paths should be configured via environment variables or a config file in a real app
    cfg_path = Path("/home/roee/repos/Effort-AIGI-Detection-Fork/DeepfakeBench/training/config/detector/effort.yaml")
    weights_path = Path(
        "/home/roee/repos/Effort-AIGI-Detection-Fork/DeepfakeBench/training/weights/effort_clip_L14_trainOn_FaceForensic.pth")

    # The dlib landmark model path is now configured inside video_preprocessor.py
    # We no longer need to load it here.

    if not cfg_path.exists() or not weights_path.exists():
        logger.error("Missing model config or weights file(s).")
        raise RuntimeError("One or more model files not found")

    # 3) Load PyTorch model
    try:
        # The 'load_detector' function is now part of this file.
        app.state.model = load_detector(str(cfg_path), str(weights_path))
        logger.info("Detector model loaded successfully.")
    except Exception:
        logger.exception("Failed to load detector model")
        raise

    # 4) Dlib loading is now handled by the video_preprocessor module on-demand.
    #    This keeps the app's startup clean and centralizes preprocessing logic.
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
            # ... (inference logic is the same) ...
            preds = app.state.model({'image': image_tensor}, inference=True)
            prob = preds["prob"].squeeze().cpu().item()
            pred_label = "FAKE" if prob >= 0.5 else "REAL"

    except Exception:
        # ... (error handling is the same) ...
        raise

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
            # ... (inference logic is the same) ...
            preds = app.state.model({'image': video_tensor.to(device)}, inference=True)
            frame_probs = preds["prob"].cpu().numpy().tolist()

    except Exception as e:
        # ... (error handling is the same) ...
        raise
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
