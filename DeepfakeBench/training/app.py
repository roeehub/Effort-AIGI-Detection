import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List

import cv2  # noqa
import dlib  # noqa
import numpy as np  # noqa
import torch  # noqa
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Request  # noqa
from fastapi.responses import JSONResponse  # noqa
from pydantic import BaseModel  # noqa
from tqdm import tqdm  # noqa

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


# For a singe image inference
class InferResponse(BaseModel):
    pred_label: str  # "REAL" or "FAKE"
    fake_prob: float


# For video inference
class VideoInferResponse(BaseModel):
    pred_label: str
    fake_prob: float
    frame_probs: Optional[List[float]] = None


# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────
def _sample_indices(strategy: str, n_frames: int, n_samples: int) -> List[int]:
    if strategy == "uniform":
        return np.linspace(0, n_frames - 1, n_samples, dtype=int).tolist()
    if strategy == "random":
        return sorted(np.random.choice(n_frames, size=n_samples, replace=False).tolist())
    raise ValueError(f"Unknown sample_strategy: {strategy!r}")


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
    weights_path = Path(
        "/home/roee/repos/Effort-AIGI-Detection-Fork/DeepfakeBench/training/weights/effort_clip_L14_trainOn_FaceForensic.pth")
    landmark_path = Path(
        "/home/roee/repos/Effort-AIGI-Detection-Fork/DeepfakeBench/preprocessing/shape_predictor_81_face_landmarks.dat")

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

    # 5) Map logits → string label
    label_idx = int(np.argmax(logits))  # 0 = REAL, 1 = FAKE
    label_str = "FAKE" if label_idx == 1 else "REAL"
    fake_prob = float(fake_prob)  # ensure native float

    logger.info("Inference result: label=%s, fake_prob=%.4f", label_str, fake_prob)
    return InferResponse(pred_label=label_str, fake_prob=fake_prob)


@app.post(
    "/check_video",
    response_model=VideoInferResponse,
    response_model_exclude_none=True
)
async def check_video(
        file: UploadFile = File(...),
        sample_strategy: str = "uniform",
        num_samples: int = 32,
        return_probs: bool = True
) -> JSONResponse:
    """Detect deep-fake content in an uploaded video."""

    # 1) MIME-type validation ────────────────────────────────────────────────
    if not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only video files are accepted"
        )

    # 2) Persist upload to a temp file so OpenCV can read it
    import tempfile, shutil

    tmp_dir = tempfile.mkdtemp(prefix="effort-aigi-")
    tmp_path = Path(tmp_dir) / file.filename
    with tmp_path.open("wb") as fp:
        shutil.copyfileobj(file.file, fp)

    cap = cv2.VideoCapture(str(tmp_path))
    if not cap.isOpened():
        cap.release()
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Cannot open video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Video has no frames")

    # 3) Parameter validation  ───────────────────────────────────────────────
    if num_samples <= 0 or num_samples > total_frames:
        raise HTTPException(
            status_code=400,
            detail=f"`num_samples` must be 1-{total_frames}, got {num_samples}"
        )
    if sample_strategy not in {"uniform", "random"}:
        raise HTTPException(
            status_code=400,
            detail='`sample_strategy` must be "uniform" or "random"'
        )

    idxs = _sample_indices(sample_strategy, total_frames, num_samples)

    logger.info(
        "Video accepted: %s | frames=%d | sampling=%s | samples=%d",
        file.filename, total_frames, sample_strategy, num_samples
    )

    # 4) Per-frame inference  ────────────────────────────────────────────────
    probs: List[float] = []

    try:
        with torch.inference_mode():
            for i, frame_idx in enumerate(tqdm(idxs, desc="Processing frames", unit="frame")):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    logger.warning("Failed to read frame %d", frame_idx)
                    continue

                _, fake_prob = infer_single_image(
                    frame_bgr,
                    face_detector=app.state.face_detector,
                    landmark_predictor=app.state.landmark_predictor,
                    model=app.state.model
                )
                probs.append(float(fake_prob))

                # periodic CUDA cache flush (avoid GPU-RAM bloat on long videos)
                if (i + 1) % 32 == 0:
                    torch.cuda.empty_cache()

    except Exception:
        logger.exception("Inference failed on video %s", file.filename)
        raise HTTPException(status_code=500, detail="Internal inference error")
    finally:
        cap.release()
        tmp_path.unlink(missing_ok=True)

    if not probs:
        raise HTTPException(status_code=500, detail="No frames could be processed")

    # 5) Aggregate result  ───────────────────────────────────────────────────
    avg_prob = float(np.mean(probs))
    label_str = "FAKE" if avg_prob >= 0.5 else "REAL"

    logger.info(
        "Video inference complete: label=%s  fake_prob=%.4f  frames=%d",
        label_str, avg_prob, len(probs)
    )

    return VideoInferResponse(
        pred_label=label_str,
        fake_prob=avg_prob,
        frame_probs=probs if return_probs else None
    )


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
