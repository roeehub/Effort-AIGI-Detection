import logging
import os
import sys
import time
from functools import partial
from multiprocessing import Pool, Manager
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from google.cloud import storage
from tqdm import tqdm

# Import the user-provided preprocessor. Assumes it's in the same directory.
# We will use its face extraction functions.
try:
    import video_preprocessor as vp
except ImportError:
    print("ERROR: Could not import 'video_preprocessor.py'. Make sure it's in the same directory.")
    sys.exit(1)

# ──────────────────────────
# 📍 Global Configuration
# ──────────────────────────
# --- Paths and GCS ---
DATASET_ROOT = Path("FaceForensics_data/FaceForensics++_C23/")
BUCKET_NAME = "faceforensics_pp_cropped"
GCS_ROOT_PREFIX = "fake"

# --- Processing Parameters ---
NUM_CORES = 32
FRAMES_TO_SAMPLE = 34
FRAMES_TO_KEEP = 32
FAILURE_TOLERANCE = 2
CROP_METHOD = 'yolo'

# --- Dataset Structure ---
FAKE_VIDEO_METHODS = [
    "DeepFakeDetection", "Deepfakes", "Face2Face",
    "FaceShifter", "FaceSwap", "NeuralTextures",
]

# ──────────────────────────
# ⚙️ Logging Setup
# ──────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(processName)s] [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

# --- Global caches for models within each process ---
# This ensures each worker process has its own model instance.
_model_cache = {}


def get_yolo_model():
    """Initializes and returns a cached YOLO model for the current process."""
    if "yolo" not in _model_cache:
        # We call the user's provided initializer which handles downloading etc.
        _model_cache["yolo"] = vp.initialize_yolo_model()
    return _model_cache["yolo"]


# ──────────────────────────
# ✨ OPTIMIZED Core Processing Logic
# ──────────────────────────

def sample_and_crop_faces_batched(video_path: Path) -> Optional[List[np.ndarray]]:
    """
    OPTIMIZED: Reads all sampled frames first, then runs face detection on the
    entire batch for maximum efficiency.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Could not open video: {video_path.name}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < FRAMES_TO_SAMPLE:
        # This check is now done before any processing, which is more efficient.
        cap.release()
        return None

    # --- Step 1: Read all 34 sampled frames into memory ---
    frame_indices = np.linspace(0, total_frames - 1, FRAMES_TO_SAMPLE, dtype=int)
    frames_to_process = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # The model expects RGB, so we convert here.
            frames_to_process.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames_to_process:
        return None

    # --- Step 2: Run YOLO on the entire batch of frames in one go ---
    model = get_yolo_model()
    # verbose=False reduces console spam from the model itself
    results = model.predict(frames_to_process, conf=vp.YOLO_CONF_THRESHOLD, verbose=False)

    # --- Step 3: Process results and crop faces ---
    successful_crops = []
    failed_crops = 0

    # We iterate through the original BGR frames and the results together
    original_bgr_frames = [cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR) for rgb_frame in frames_to_process]

    for frame_bgr, result in zip(original_bgr_frames, results):
        if len(successful_crops) == FRAMES_TO_KEEP:
            break

        if result.boxes.shape[0] == 0:
            failed_crops += 1
        else:
            # Find the largest face box if multiple are detected
            boxes = result.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            x0, y0, x1, y1 = boxes[np.argmax(areas)]

            # Crop and resize logic from vp.extract_yolo_face
            h, w = frame_bgr.shape[:2]
            x0 = max(0, x0 - vp.YOLO_BBOX_MARGIN)
            y0 = max(0, y0 - vp.YOLO_BBOX_MARGIN)
            x1 = min(w, x1 + vp.YOLO_BBOX_MARGIN)
            y1 = min(h, y1 + vp.YOLO_BBOX_MARGIN)

            width, height = x1 - x0, y1 - y0
            center_x, center_y = x0 + width / 2, y0 + height / 2
            side_length = max(width, height)

            sq_x0 = max(0, int(center_x - side_length / 2))
            sq_y0 = max(0, int(center_y - side_length / 2))
            sq_x1 = min(w, int(center_x + side_length / 2))
            sq_y1 = min(h, int(center_y + side_length / 2))

            cropped_face = frame_bgr[sq_y0:sq_y1, sq_x0:sq_x1]
            if cropped_face.size > 0:
                resized_face = cv2.resize(cropped_face, (vp.MODEL_IMG_SIZE, vp.MODEL_IMG_SIZE),
                                          interpolation=cv2.INTER_AREA)
                successful_crops.append(resized_face)
            else:
                failed_crops += 1

        if failed_crops > FAILURE_TOLERANCE:
            logging.warning(f"Skipping {video_path.name}: Exceeded failure tolerance during batch processing.")
            return None

    if len(successful_crops) >= FRAMES_TO_KEEP:
        return successful_crops[:FRAMES_TO_KEEP]
    else:
        logging.warning(f"Skipping {video_path.name}: Only found {len(successful_crops)} faces after batch processing.")
        return None


# ──────────────────────────
# ☁️ Google Cloud Storage Worker
# ──────────────────────────

gcs_client_worker = None


def worker_initializer():
    global gcs_client_worker
    gcs_client_worker = storage.Client()
    get_yolo_model()  # This initializes the model once per worker process.


def process_and_upload_video(video_path: Path, progress_data: dict):
    global gcs_client_worker
    bucket = gcs_client_worker.bucket(BUCKET_NAME)
    try:
        method = video_path.parent.name
        video_name = video_path.stem
        gcs_prefix = f"{GCS_ROOT_PREFIX}/{method}/{video_name}/"

        last_frame_blob_name = f"{gcs_prefix}{FRAMES_TO_KEEP - 1:04d}.png"
        if bucket.blob(last_frame_blob_name).exists():
            progress_data['skipped'] += 1
            return

        # Use the NEW batched function
        cropped_faces = sample_and_crop_faces_batched(video_path)

        if not cropped_faces:
            if video_path.name not in str(progress_data.get('failed_log', [])):  # Avoid duplicate logs
                progress_data['failed'] += 1
            return

        for i, face_img in enumerate(cropped_faces):
            _, buffer = cv2.imencode(".png", face_img)
            blob_name = f"{gcs_prefix}{i:04d}.png"
            blob = bucket.blob(blob_name)
            blob.upload_from_string(buffer.tobytes(), content_type="image/png")

        progress_data['processed'] += 1
    except Exception as e:
        logging.error(f"FATAL ERROR processing {video_path.name}: {e}")
        progress_data['failed'] += 1


# ──────────────────────────
# 🚀 Main Execution
# ──────────────────────────

def main():
    if not DATASET_ROOT.is_dir():
        logging.error(f"Dataset root directory not found: {DATASET_ROOT}")
        sys.exit(1)

    logging.info("Starting dataset preprocessing with OPTIMIZED BATCHING...")
    logging.info(f"Using {NUM_CORES} cores and '{CROP_METHOD}' crop method.")
    logging.info(f"Output bucket: gs://{BUCKET_NAME}/{GCS_ROOT_PREFIX}")

    tasks = []
    for method in FAKE_VIDEO_METHODS:
        method_path = DATASET_ROOT / method
        if method_path.exists():
            tasks.extend(list(method_path.glob("*.mp4")))
        else:
            logging.warning(f"Method directory not found, skipping: {method_path}")

    if not tasks:
        logging.error("No .mp4 files found. Check DATASET_ROOT and FAKE_VIDEO_METHODS.")
        sys.exit(1)

    logging.info(f"Found {len(tasks)} total videos to process.")

    with Manager() as manager:
        progress_data = manager.dict({'processed': 0, 'skipped': 0, 'failed': 0})
        worker_func = partial(process_and_upload_video, progress_data=progress_data)

        with Pool(processes=NUM_CORES, initializer=worker_initializer) as pool:
            list(tqdm(pool.imap_unordered(worker_func, tasks), total=len(tasks), desc="Processing Videos"))

        logging.info("=" * 50)
        logging.info("            PROCESSING COMPLETE")
        logging.info("=" * 50)
        logging.info(f"Successfully processed and uploaded: {progress_data['processed']} videos")
        logging.info(f"Skipped (already exist):           {progress_data['skipped']} videos")
        logging.info(f"Failed or skipped (errors):        {progress_data['failed']} videos")
        logging.info(f"Total videos checked:              {sum(progress_data.values())}")
        logging.info("=" * 50)


if __name__ == "__main__":
    main()
