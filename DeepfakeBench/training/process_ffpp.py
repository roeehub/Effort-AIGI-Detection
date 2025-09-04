import argparse
import logging
import os
import sys
import tempfile
import time
from functools import partial
from multiprocessing import Pool, Manager
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import cv2
import numpy as np
from google.cloud import storage
from tqdm import tqdm

try:
    import video_preprocessor as vp
except ImportError:
    print("ERROR: Could not import 'video_preprocessor.py'. Make sure it's in the same directory.")
    sys.exit(1)

# ──────────────────────────
# 📍 Global Configuration
# ──────────────────────────
GLOBAL_BUCKET_NAME = "effort-collected-data"
GCS_SOURCE_BUCKET = "veo3-creations"  # Default, can be overridden

# --- Processing Parameters ---
NUM_CORES = 4
FRAMES_TO_SAMPLE = 35
FRAMES_TO_KEEP = 32
FAILURE_TOLERANCE = FRAMES_TO_SAMPLE - FRAMES_TO_KEEP

# ──────────────────────────
# ⚙️ Logging & Initialization
# ──────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(processName)s] [%(levelname)s] - %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
_model_cache = {}
gcs_client_worker = None


def get_yolo_model():
    if "yolo" not in _model_cache:
        _model_cache["yolo"] = vp.initialize_yolo_model()
    return _model_cache["yolo"]


def worker_initializer():
    global gcs_client_worker
    gcs_client_worker = storage.Client()
    get_yolo_model()


# ──────────────────────────
# ✨ Core Processing Logic (FIXED)
# ──────────────────────────

def crop_and_resize_face(frame_bgr: np.ndarray, box: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    """
    Crops a face from a frame using bounding box coordinates, converts it to a
    square, and resizes it. This logic is adapted from video_preprocessor.py.
    """
    x0, y0, x1, y1 = [int(coord) for coord in box]
    h, w = frame_bgr.shape[:2]

    width, height = x1 - x0, y1 - y0
    center_x, center_y = x0 + width / 2, y0 + height / 2
    side_length = max(width, height)
    sq_x0 = max(0, int(center_x - side_length / 2))
    sq_y0 = max(0, int(center_y - side_length / 2))
    sq_x1 = min(w, int(center_x + side_length / 2))
    sq_y1 = min(h, int(center_y + side_length / 2))

    cropped_face = frame_bgr[sq_y0:sq_y1, sq_x0:sq_x1]
    if cropped_face.size == 0:
        return None

    # Use the image size from the imported vp module
    return cv2.resize(cropped_face, (vp.MODEL_IMG_SIZE, vp.MODEL_IMG_SIZE), interpolation=cv2.INTER_AREA)


def sample_and_crop_faces_batched(video_path: Path) -> Optional[List[np.ndarray]]:
    """
    More robustly samples frames from a video, detects faces in a batch,
    and returns a list of cropped faces. The function succeeds if at least
    FRAMES_TO_KEEP faces are found out of the FRAMES_TO_SAMPLE attempts.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.warning(f"Could not open video file: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < FRAMES_TO_SAMPLE:
        cap.release()
        return None

    frame_indices_to_sample = np.linspace(0, total_frames - 1, FRAMES_TO_SAMPLE, dtype=int)
    frames_for_yolo = []
    original_bgr_frames_read = []

    for frame_idx in frame_indices_to_sample:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Keep both the original BGR for cropping and the RGB version for YOLO
            original_bgr_frames_read.append(frame)
            frames_for_yolo.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames_for_yolo) < FRAMES_TO_KEEP:
        logging.warning(
            f"Could not read enough frames from {video_path}. Needed {FRAMES_TO_KEEP}, got {len(frames_for_yolo)}.")
        return None

    model = get_yolo_model()
    # Process all collected frames in a single batch for efficiency
    results = model.predict(frames_for_yolo, conf=vp.YOLO_CONF_THRESHOLD, verbose=False)

    successful_crops = []
    # Loop through all results to gather as many valid faces as possible
    for frame_bgr, result in zip(original_bgr_frames_read, results):
        if result.boxes.shape[0] > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_box = boxes[np.argmax(areas)]

            cropped_face = crop_and_resize_face(frame_bgr, tuple(best_box))
            if cropped_face is not None:
                successful_crops.append(cropped_face)

    # Only after trying all frames, check if we have enough
    if len(successful_crops) >= FRAMES_TO_KEEP:
        return successful_crops[:FRAMES_TO_KEEP]
    else:
        logging.warning(
            f"Found only {len(successful_crops)} faces in {video_path} after sampling {len(frames_for_yolo)} frames. Required {FRAMES_TO_KEEP}.")
        return None


# ──────────────────────────
# ☁️ Unified GCS Worker (Robust Version)
# ──────────────────────────
def process_task_worker(task: Dict, progress_data: dict, bucket_name: str, gcs_root_prefix: str):
    global gcs_client_worker
    target_bucket = gcs_client_worker.bucket(bucket_name)

    video_name, method = task['gcs_path_parts']
    source_ref = task['source_ref']
    source_name_for_logs = source_ref['name']

    try:
        gcs_prefix = f"{gcs_root_prefix}/{method}/{video_name}/"
        last_frame_blob_name = f"{gcs_prefix}{FRAMES_TO_KEEP - 1:04d}.png"
        if target_bucket.blob(last_frame_blob_name).exists():
            progress_data['skipped'] += 1
            return

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp_f:
            source_bucket = gcs_client_worker.bucket(source_ref['bucket'])
            source_blob = source_bucket.blob(source_ref['name'])
            source_blob.download_to_filename(tmp_f.name)
            local_video_path = Path(tmp_f.name)
            cropped_faces = sample_and_crop_faces_batched(local_video_path)

        if not cropped_faces:
            logging.warning(f"Failed to extract sufficient faces from '{source_name_for_logs}'.")
            progress_data['failed'] += 1
            return

        for i, face_img in enumerate(cropped_faces):
            _, buffer = cv2.imencode(".png", face_img)
            blob_name = f"{gcs_prefix}{i:04d}.png"
            blob = target_bucket.blob(blob_name)
            blob.upload_from_string(buffer.tobytes(), content_type="image/png")

        progress_data['processed'] += 1

    except Exception as e:
        logging.error(f"FATAL EXCEPTION on '{source_name_for_logs}': {e}", exc_info=True)
        progress_data['failed'] += 1


# ──────────────────────────
# 🚀 Main Execution
# ──────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos from GCS and upload cropped faces.")
    parser.add_argument('--gcs-source', action='store_true',
                        help='Process videos from a GCS bucket (this is the main mode).')
    parser.add_argument('--label', type=str, required=True,
                        help='Top-level GCS prefix for data (e.g., "fake" or "real").')
    parser.add_argument('--gcs-bucket', type=str, required=True, help=f'Source GCS bucket name.')
    parser.add_argument('--gcs-prefix', type=str, default="",
                        help='Optional path prefix within the source GCS bucket to filter videos.')
    parser.add_argument('--cores', type=int, default=NUM_CORES, help=f'Number of CPU cores to use.')
    args = parser.parse_args()

    # Ensure this mode is used, simplifies logic
    if not args.gcs_source:
        print("ERROR: This script requires the --gcs-source flag to run.", file=sys.stderr)
        sys.exit(1)

    NUM_CORES = args.cores
    tasks = []

    logging.info(f"Running in GCS-SOURCE mode.")
    client = storage.Client()
    blobs = client.list_blobs(args.gcs_bucket, prefix=args.gcs_prefix)
    method_name = args.gcs_bucket

    video_blobs = [b for b in blobs if b.name.lower().endswith((".mp4", ".mov", ".avi")) and not b.name.endswith('/')]

    for blob in video_blobs:
        # --- THIS IS THE KEY FIX FOR THE PATHING ISSUE ---
        # Instead of the file's stem (e.g., "sample_0"), get the parent directory's name
        video_id = Path(blob.name).parent.name
        tasks.append({
            "source_ref": {"bucket": blob.bucket.name, "name": blob.name},
            "gcs_path_parts": (video_id, method_name),
        })

    target_bucket = GLOBAL_BUCKET_NAME
    target_gcs_prefix = args.label
    logging.info(f"Source: gs://{args.gcs_bucket}/{args.gcs_prefix}...")
    logging.info(f"Target Structure: gs://{target_bucket}/{target_gcs_prefix}/{method_name}/<video_id>/")

    if not tasks:
        logging.error("No compatible video files found at the specified GCS path.")
        sys.exit(1)

    logging.info(f"Found {len(tasks)} videos. Initializing {NUM_CORES} workers (this may take a moment)...")

    with Manager() as manager:
        progress_data = manager.dict({'processed': 0, 'skipped': 0, 'failed': 0})
        worker_func = partial(process_task_worker,
                              progress_data=progress_data,
                              bucket_name=target_bucket,
                              gcs_root_prefix=target_gcs_prefix)

        with Pool(processes=NUM_CORES, initializer=worker_initializer) as pool:
            list(tqdm(pool.imap_unordered(worker_func, tasks), total=len(tasks), desc="Processing Videos"))

        logging.info("=" * 50 + "\n            PROCESSING COMPLETE\n" + "=" * 50)
        logging.info(f"Successfully processed: {progress_data['processed']}")
        logging.info(f"Skipped (already exist): {progress_data['skipped']}")
        logging.info(f"Failed (could not process): {progress_data['failed']}")
        logging.info("=" * 50)
