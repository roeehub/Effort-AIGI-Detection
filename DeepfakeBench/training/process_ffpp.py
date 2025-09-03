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
NUM_CORES = 32  # Number of CPU cores to use for parallel processing
FRAMES_TO_SAMPLE = 34  # Uniformly sample this many frames from each video
FRAMES_TO_KEEP = 32  # Keep the first 32 successfully cropped frames
FAILURE_TOLERANCE = 2  # Max number of failed face crops before skipping a video
# UPDATED: Changed the crop method to 'yolo' for simple square cropping.
CROP_METHOD = 'yolo'  # Face crop method: 'yolo', 'yolo_haar', or 'dlib'

# --- Dataset Structure ---
# Folders containing the fake videos to be processed
FAKE_VIDEO_METHODS = [
    "DeepFakeDetection",
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
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


# ──────────────────────────
# ✨ Core Processing Logic
# ──────────────────────────

def sample_and_crop_faces(video_path: Path) -> Optional[List[np.ndarray]]:
    """
    Samples frames from a video, crops faces, and respects failure tolerance.

    Args:
        video_path: Path to the video file.

    Returns:
        A list of 32 cropped face images (as numpy arrays), or None if processing fails.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Could not open video: {video_path.name}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < FRAMES_TO_SAMPLE:
        logging.warning(f"Skipping {video_path.name}: not enough frames ({total_frames})")
        cap.release()
        return None

    frame_indices = np.linspace(0, total_frames - 1, FRAMES_TO_SAMPLE, dtype=int)
    successful_crops = []
    failed_crops = 0

    # This dictionary dynamically selects the correct function from video_preprocessor.py
    crop_function = {
        'yolo': vp.extract_yolo_face,
        'yolo_haar': vp.extract_yolo_haar_face,
        'dlib': vp.extract_aligned_face
    }.get(CROP_METHOD)

    if not crop_function:
        raise ValueError(f"Invalid CROP_METHOD: {CROP_METHOD}")

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            failed_crops += 1
            continue

        face = crop_function(frame)

        if face is not None:
            successful_crops.append(face)
            if len(successful_crops) == FRAMES_TO_KEEP:
                break  # We have enough frames
        else:
            failed_crops += 1

        if failed_crops > FAILURE_TOLERANCE:
            logging.warning(f"Skipping {video_path.name}: Exceeded failure tolerance.")
            cap.release()
            return None

    cap.release()

    if len(successful_crops) >= FRAMES_TO_KEEP:
        return successful_crops[:FRAMES_TO_KEEP]
    else:
        logging.warning(f"Skipping {video_path.name}: Only found {len(successful_crops)} faces.")
        return None


# ──────────────────────────
# ☁️ Google Cloud Storage Worker
# ──────────────────────────

# Global GCS client for each worker process to avoid re-initialization
gcs_client_worker = None


def worker_initializer():
    """
    Initializes a GCS client and the required YOLO model for each worker process.
    This runs once per process in the pool.
    """
    global gcs_client_worker
    gcs_client_worker = storage.Client()
    # Initialize the required YOLO model for this worker process.
    # The Haar cascades and dlib models are not initialized, saving resources.
    vp.initialize_yolo_model()


def process_and_upload_video(video_path: Path, progress_data: dict):
    """
    The main worker function executed by each process. It handles checking,
    processing, and uploading for a single video.
    """
    global gcs_client_worker
    bucket = gcs_client_worker.bucket(BUCKET_NAME)

    try:
        method = video_path.parent.name
        video_name = video_path.stem
        gcs_prefix = f"{GCS_ROOT_PREFIX}/{method}/{video_name}/"

        # --- RESUMABILITY CHECK ---
        # Check if the video has already been processed by looking for the last frame.
        last_frame_blob_name = f"{gcs_prefix}{FRAMES_TO_KEEP - 1:04d}.png"
        if bucket.blob(last_frame_blob_name).exists():
            progress_data['skipped'] += 1
            return

        # --- PROCESS VIDEO ---
        cropped_faces = sample_and_crop_faces(video_path)

        if not cropped_faces:
            progress_data['failed'] += 1
            return

        # --- UPLOAD TO GCS ---
        for i, face_img in enumerate(cropped_faces):
            # Encode image to PNG in memory
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
    """Gathers all video files and processes them in parallel."""
    if not DATASET_ROOT.exists():
        logging.error(f"Dataset root directory not found: {DATASET_ROOT}")
        sys.exit(1)

    logging.info("Starting dataset preprocessing...")
    logging.info(f"Using {NUM_CORES} cores and '{CROP_METHOD}' crop method.")
    logging.info(f"Output bucket: gs://{BUCKET_NAME}/{GCS_ROOT_PREFIX}")

    # --- 1. GATHER ALL VIDEO FILES ---
    tasks = []
    for method in FAKE_VIDEO_METHODS:
        method_path = DATASET_ROOT / method
        if method_path.exists():
            # Using glob for efficiency
            tasks.extend(list(method_path.glob("*.mp4")))
        else:
            logging.warning(f"Method directory not found, skipping: {method_path}")

    if not tasks:
        logging.error("No .mp4 files found to process. Check DATASET_ROOT and FAKE_VIDEO_METHODS.")
        sys.exit(1)

    logging.info(f"Found {len(tasks)} total videos to process.")

    # --- 2. SETUP PARALLEL EXECUTION ---
    # Using a Manager dictionary to share progress stats between processes
    with Manager() as manager:
        progress_data = manager.dict({'processed': 0, 'skipped': 0, 'failed': 0})

        # Use partial to pass the progress dictionary to the worker
        worker_func = partial(process_and_upload_video, progress_data=progress_data)

        with Pool(processes=NUM_CORES, initializer=worker_initializer) as pool:
            # Use imap_unordered for efficient job distribution and tqdm for progress bar
            list(tqdm(pool.imap_unordered(worker_func, tasks), total=len(tasks), desc="Processing Videos"))

        # --- 3. PRINT FINAL SUMMARY ---
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
