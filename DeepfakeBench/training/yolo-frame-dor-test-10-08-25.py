# yolo-frame-dor-test-10-08-25.py

import os
import sys
import logging
import time
import io
from pathlib import Path
from typing import List, Optional, Set

import cv2
import numpy as np
import torch
from google.cloud import storage
from google.api_core import exceptions
from tqdm import tqdm

# Assuming video_preprocessor.py is in the same directory or accessible in PYTHONPATH
import video_preprocessor

# --- CONFIGURATION ---
# Source bucket and prefix containing the original videos
SOURCE_BUCKET_NAME = "deep-fake-test-10-08-25"
SOURCE_DATA_PREFIX = "Deep fake test 10.08.25/"

# Target buckets for the processed frames
YOLO_BUCKET_NAME = "deep-fake-test-10-08-25-frames-yolo"
YOLO_HAAR_BUCKET_NAME = "deep-fake-test-10-08-25-frames-yolo-haar"

# Local directories for temporary files and logs
BASE_DIR = Path("./preprocessing_run")
LOCAL_TEMP_DIR = BASE_DIR / "temp_videos"
LOG_FILE = BASE_DIR / "preprocessing_log.txt"

# --- FRAME SAMPLING PARAMETERS ---
INITIAL_SAMPLE_COUNT = 40
TARGET_FRAME_COUNT = 32
MIN_FRAME_COUNT = 16  # Videos with fewer than this many faces will be skipped

# --- GLOBAL VARIABLES ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger("preprocessor")


def setup_logging():
    """Configures logging to both file and console."""
    BASE_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Silence overly verbose GCS logs
    logging.getLogger('google.auth.transport.requests').setLevel(logging.WARNING)


def initialize_preprocessors():
    """Initializes the YOLO model and Haar cascades required for face extraction."""
    logger.info("Initializing face detection pre-processors...")
    try:
        video_preprocessor.initialize_yolo_model()
        video_preprocessor.initialize_haar_cascades()
        logger.info("✅ Pre-processors initialized successfully.")
        return True
    except Exception as e:
        logger.exception(f"FATAL: Failed to initialize pre-processors. Error: {e}")
        return False


def list_gcs_blobs(bucket_name: str, prefix: Optional[str] = None) -> Set[str]:
    """Lists all blob names in a GCS bucket for quick lookups."""
    try:
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
        return {blob.name for blob in blobs}
    except exceptions.Forbidden as e:
        logger.error(
            f"FATAL: GCS Permission Denied for bucket '{bucket_name}'. Have you run 'gcloud auth application-default login'?")
        raise e
    except Exception as e:
        logger.error(f"FATAL: Could not list blobs in bucket '{bucket_name}'. Error: {e}")
        raise e


def get_target_blob_name(source_blob_name: str) -> str:
    """
    Constructs the target GCS path for the .npz file from the original video path.
    Example: 'Deep fake test 10.08.25/real/tik tok/video.mp4' -> 'real/tik tok/video.npz'
    """
    relative_path = os.path.relpath(source_blob_name, SOURCE_DATA_PREFIX)
    return str(Path(relative_path).with_suffix(".npz"))


def process_video_for_faces(video_path: Path, pre_method: str) -> Optional[np.ndarray]:
    """
    Extracts face frames from a single video file.
    This is adapted from `process_and_cache_video_faces` from the original script.

    Args:
        video_path (Path): The local path to the video file.
        pre_method (str): The face extraction method ('yolo' or 'yolo_haar').

    Returns:
        An optional NumPy array of face frames if successful, otherwise None.
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}. Skipping.")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            logger.warning(f"Video has 0 frames: {video_path}. Skipping.")
            return None

        frame_indices = np.linspace(0, total_frames - 1, INITIAL_SAMPLE_COUNT, dtype=int)

        collected_faces = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            face = None
            if pre_method == 'yolo':
                face = video_preprocessor.extract_yolo_face(frame)
            elif pre_method == 'yolo_haar':
                face = video_preprocessor.extract_yolo_haar_face(frame)

            if face is not None:
                collected_faces.append(face)

        cap.release()

        if len(collected_faces) < MIN_FRAME_COUNT:
            logger.info(
                f"Found only {len(collected_faces)} faces (min: {MIN_FRAME_COUNT}) in {video_path.name}. Skipping.")
            return None

        # Take up to TARGET_FRAME_COUNT frames and return as a single numpy array
        final_faces = collected_faces[:TARGET_FRAME_COUNT]
        return np.array(final_faces)

    except cv2.error as e:
        logger.error(f"OpenCV error while processing {video_path.name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing {video_path.name}: {e}")
        return None


def upload_frames_to_gcs(storage_client: storage.Client, bucket_name: str, blob_name: str, frames_array: np.ndarray):
    """
    Saves a numpy array of frames to a .npz file in memory and uploads it to GCS.
    """
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        with io.BytesIO() as buffer:
            np.savez_compressed(buffer, faces=frames_array)
            buffer.seek(0)
            blob.upload_from_file(buffer, content_type='application/octet-stream')

        logger.debug(f"Successfully uploaded to gs://{bucket_name}/{blob_name}")
    except Exception as e:
        logger.error(f"Failed to upload frames to gs://{bucket_name}/{blob_name}. Error: {e}")


def main():
    """Main script execution function."""
    setup_logging()
    logger.info("========= Starting GCS Video Preprocessing and Upload Script =========")

    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Using CPU. This may be slow.")
    else:
        logger.info(f"CUDA is available. Using device: {device}")

    if not initialize_preprocessors():
        return

    # Create local temp directory
    LOCAL_TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Get lists of all source and existing target files ---
    logger.info("Fetching file lists from GCS for recovery check...")
    try:
        storage_client = storage.Client()
        source_blobs = list(storage_client.list_blobs(SOURCE_BUCKET_NAME, prefix=SOURCE_DATA_PREFIX))

        # Filter for actual video files
        source_videos = [b for b in source_blobs if b.name.lower().endswith(('.mp4', '.mov', '.avi', '.webm'))]

        if not source_videos:
            logger.error("No video files found at the specified source path. Halting.")
            return

        existing_yolo_blobs = list_gcs_blobs(YOLO_BUCKET_NAME)
        existing_yolo_haar_blobs = list_gcs_blobs(YOLO_HAAR_BUCKET_NAME)
        logger.info(f"Found {len(source_videos)} source videos.")
        logger.info(f"Found {len(existing_yolo_blobs)} existing files in YOLO bucket.")
        logger.info(f"Found {len(existing_yolo_haar_blobs)} existing files in YOLO_HAAR bucket.")
    except Exception as e:
        logger.error(f"Could not initialize GCS connection. Halting. Error: {e}")
        return

    # --- Phase 2: Process each video ---
    logger.info("\n--- Starting video processing loop ---")
    processed_count = 0
    skipped_count = 0

    for video_blob in tqdm(source_videos, desc="Processing Videos"):
        source_blob_name = video_blob.name
        target_blob_name = get_target_blob_name(source_blob_name)

        # Determine which methods need to be run for this video
        process_yolo = target_blob_name not in existing_yolo_blobs
        process_yolo_haar = target_blob_name not in existing_yolo_haar_blobs

        if not process_yolo and not process_yolo_haar:
            logger.debug(f"All outputs exist for {source_blob_name}. Skipping.")
            skipped_count += 1
            continue

        logger.info(f"Processing: {source_blob_name}")
        local_video_path = None
        extracted_frames = {}  # Cache frames to avoid re-processing

        try:
            # --- Download Step ---
            local_video_path = LOCAL_TEMP_DIR / Path(source_blob_name).name
            logger.debug(f"Downloading to {local_video_path}")
            video_blob.download_to_filename(local_video_path)

            # --- YOLO Processing ---
            if process_yolo:
                logger.info(f"  -> Running YOLO face extraction...")
                yolo_frames = process_video_for_faces(local_video_path, 'yolo')
                if yolo_frames is not None:
                    extracted_frames['yolo'] = yolo_frames
                    upload_frames_to_gcs(storage_client, YOLO_BUCKET_NAME, target_blob_name, yolo_frames)
                    logger.info(f"     ✅ YOLO frames uploaded.")
                else:
                    logger.warning(f"     ❌ YOLO processing failed or found too few faces.")

            # --- YOLO-HAAR Processing ---
            if process_yolo_haar:
                logger.info(f"  -> Running YOLO-HAAR face extraction...")
                yolo_haar_frames = process_video_for_faces(local_video_path, 'yolo_haar')
                if yolo_haar_frames is not None:
                    extracted_frames['yolo_haar'] = yolo_haar_frames
                    upload_frames_to_gcs(storage_client, YOLO_HAAR_BUCKET_NAME, target_blob_name, yolo_haar_frames)
                    logger.info(f"     ✅ YOLO-HAAR frames uploaded.")
                else:
                    logger.warning(f"     ❌ YOLO-HAAR processing failed or found too few faces.")

            processed_count += 1

        except Exception as e:
            logger.exception(f"An unexpected error occurred while processing {source_blob_name}: {e}")
        finally:
            # --- Cleanup Step ---
            if local_video_path and local_video_path.exists():
                os.remove(local_video_path)
                logger.debug(f"Cleaned up {local_video_path}")

    logger.info("========= Preprocessing Finished =========")
    logger.info(f"Total videos attempted: {len(source_videos)}")
    logger.info(f"Videos processed in this run: {processed_count}")
    logger.info(f"Videos skipped (already processed): {skipped_count}")
    logger.info(f"Full log available at: {LOG_FILE}")


if __name__ == "__main__":
    main()
