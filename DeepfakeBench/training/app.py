import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import dlib
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from infer import load_detector, infer_single_image

# ──────────────────────────────────────────
# Logging
# ──────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("effort-aigi-api")

# ──────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────
app = FastAPI(title="Effort-AIGI Detector API", version="0.1.0")


class InferResponse(BaseModel):
    pred_label: int       # 0 = real, 1 = fake
    fake_prob: float


# ──────────────────────────────────────────
# Startup: load model + dlib + assert CUDA
# ──────────────────────────────────────────
@app.on_event("startup")
def startup_event() -> None:
    # 1) CUDA
    assert torch.cuda.is_available(), "CUDA is required for this service"
    logger.info("CUDA is available.")

    # 2) Paths from your batch_demo invocation
    cfg_path = Path("/home/roee/repos/Effort-AIGI-Detection-Fork/DeepfakeBench/training/config/detector/effort.yaml")
    weights_path = Path("/home/roee/repos/Effort-AIGI-Detection-Fork/DeepfakeBench/training/weights/effort_clip_L14_trainOn_FaceForensic.pth")
    landmark_path = Path("/home/roee/repos/Effort-AIGI-Detection-Fork/DeepfakeBench/preprocessing/shape_predictor_81_face_landmarks.dat")

    if not cfg_path.exists() or not weights_path.exists() or not landmark_path.exists():
        logger.error("Missing file(s): %s, %s, %s", cfg_path, weights_path, landmark_path)
        raise RuntimeError("One or more model files not found")

    # 3) Load PyTorch model
    try:
        app.state.model = load_detector(str(cfg_path), str(weights_path))
        logger.info("Detector model loaded.")
    except Exception:
        logger.exception("Failed to load detector model")
        raise

    # 4) Init dlib
    try:
        app.state.face_detector = dlib.get_frontal_face_detector()
        app.state.landmark_predictor = dlib.shape_predictor(str(landmark_path))
        logger.info("Dlib face detector and landmark predictor loaded.")
    except Exception:
        logger.exception("Failed to initialize dlib components")
        raise


# ──────────────────────────────────────────
# Health-check
# ──────────────────────────────────────────
@app.get("/ping")
def ping() -> dict:
    return {"message": "pong"}


# ──────────────────────────────────────────
# Inference Endpoint
# ──────────────────────────────────────────
@app.post("/check_frame", response_model=InferResponse)
async def check_frame(file: UploadFile = File(...)) -> JSONResponse:
    # 1) MIME-type validation
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only JPEG or PNG images are accepted"
        )

    # 2) Read & decode
    raw = await file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    # 3) dlib face detection
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.state.face_detector(rgb, 1)
    face_count = len(faces)

    if face_count != 1:
        logger.warning("Expected exactly 1 face, got %d", face_count)
        return JSONResponse(
            status_code=400,
            content={
                "detail": "Expected exactly one face in the image",
                "face_count": face_count
            }
        )

    # 4) Inference
    try:
        logits, fake_prob = infer_single_image(
            img,
            face_detector=app.state.face_detector,
            landmark_predictor=app.state.landmark_predictor,
            model=app.state.model
        )
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Internal inference error")

    # 5) Map logits to label
    label_idx = int(np.argmax(logits))       # 0 = REAL, 1 = FAKE
    label_str = "FAKE" if label_idx == 1 else "REAL"
    fake_prob = float(fake_prob)             # ensure native float

    logger.info("Inference result: label=%s, fake_prob=%.4f", label_str, fake_prob)
    return InferResponse(pred_label=label_str, fake_prob=fake_prob)


# ──────────────────────────────────────────
# Global Exception Handler
# ──────────────────────────────────────────
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Unexpected server error"},
    )
