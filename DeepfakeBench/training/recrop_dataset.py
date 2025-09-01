"""
recrop_dataset.py
================================================
A high-performance, resumable script to re-process an entire image dataset to
a standardized Relative Face Area (RFA).

Purpose:
To eliminate face crop size as a confounding variable in a deepfake detection
dataset. This script reads every frame from a source GCS bucket, re-crops it
so the face occupies a consistent portion of the frame (TARGET_RFA), and writes
the result to a destination bucket.

Features:
- GPU Acceleration: Leverages a CUDA-enabled GPU for fast YOLO face detection.
- Parallel Processing: Uses a ThreadPoolExecutor to handle I/O and CPU tasks concurrently.
- Resumable: Checks for the existence of the output file before processing,
  allowing the script to be stopped and restarted without re-doing work.
- Robust Error Handling: Skips frames where no face is detected or other
  processing errors occur, and reports on them at the end.
"""

import os
import argparse
import concurrent.futures
from functools import partial
from collections import Counter

import cv2
import fsspec
import numpy as np
import torch
from google.cloud import storage
from tqdm import tqdm

# --- Local Imports from Project ---
try:
    # Works when running as a module from the repo root:
    #   python -m training.recrop_dataset
    from training.video_preprocessor import initialize_yolo_model, _get_yolo_face_box
except Exception:
    try:
        # Works when running the file directly from inside training/:
        #   python recrop_dataset.py
        from video_preprocessor import initialize_yolo_model, _get_yolo_face_box
    except Exception as e:
        import os, sys, traceback
        # Last-resort: add this file's directory to sys.path and retry
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        try:
            from video_preprocessor import initialize_yolo_model, _get_yolo_face_box
        except Exception:
            print("\n[ERROR] Could not import from 'video_preprocessor.py'.")
            print("Please ensure you either:")
            print("  1) run from repo root with:  python -m training.recrop_dataset")
            print("  2) or run inside training/: python recrop_dataset.py")
            print("If needed, set PYTHONPATH to the repo root.")
            print("\nDebug info:\n", traceback.format_exc())
            sys.exit(1)

# --- Configuration ---
TARGET_RFA = 0.85
FINAL_IMAGE_SIZE = (224, 224)  # Final output resolution for the model
DEFAULT_WORKERS = 32
IMAGE_QUALITY = 90  # JPEG quality for output images


# --- Core Re-cropping Logic ---

def crop_to_target_rfa(image_np: np.ndarray, yolo_model) -> np.ndarray | None:
    """
    Takes an image, detects the face, and re-crops it to match TARGET_RFA.
    """
    img_height, img_width = image_np.shape[:2]

    # 1. Detect the face in the current image.
    bbox = _get_yolo_face_box(image_np, yolo_model)
    if bbox is None:
        return None  # No face found

    x0, y0, x1, y1 = bbox
    face_width = x1 - x0
    face_height = y1 - y0
    face_area = face_width * face_height
    face_center_x = x0 + face_width / 2
    face_center_y = y0 + face_height / 2

    if face_area == 0:
        return None  # Invalid face detection

    # 2. Calculate the size of the NEW crop window needed.
    # new_crop_area = face_area / TARGET_RFA
    # new_crop_side = sqrt(new_crop_area) for a square crop
    new_crop_side = np.sqrt(face_area / (TARGET_RFA + 1e-6))

    # 3. Determine the coordinates of this new crop window, centered on the face.
    new_x0 = int(face_center_x - new_crop_side / 2)
    new_y0 = int(face_center_y - new_crop_side / 2)
    new_x1 = int(face_center_x + new_crop_side / 2)
    new_y1 = int(face_center_y + new_crop_side / 2)

    # 4. Clamp the coordinates to stay within the original image bounds.
    new_x0 = max(0, new_x0)
    new_y0 = max(0, new_y0)
    new_x1 = min(img_width, new_x1)
    new_y1 = min(img_height, new_y1)

    # 5. Perform the new crop and resize.
    final_crop = image_np[new_y0:new_y1, new_x0:new_x1]

    if final_crop.size == 0:
        return None

    return cv2.resize(final_crop, FINAL_IMAGE_SIZE, interpolation=cv2.INTER_AREA)


def process_single_frame(gcs_paths: tuple, fs: fsspec.AbstractFileSystem, yolo_model) -> str:
    """
    Worker function: takes a (source, dest) path tuple, processes, and returns status.
    This is the target for each parallel thread.
    """
    source_path, dest_path = gcs_paths
    try:
        # RESUMABILITY: Check if the destination file already exists.
        if fs.exists(dest_path):
            return 'skipped'

        # Download image
        with fs.open(source_path, 'rb') as f:
            image_bytes = np.frombuffer(f.read(), np.uint8)

        image_np_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image_np_bgr is None:
            return 'read_error'

        # Core processing
        recropped_image = crop_to_target_rfa(image_np_bgr, yolo_model)

        if recropped_image is None:
            return 'no_face'

        # Encode and upload
        _, buffer = cv2.imencode('.jpg', recropped_image, [int(cv2.IMWRITE_JPEG_QUALITY), IMAGE_QUALITY])

        with fs.open(dest_path, 'wb') as f:
            f.write(buffer.tobytes())

        return 'processed'

    except Exception as e:
        tqdm.write(f"\n[ERROR] Failed to process {source_path}: {e}")
        return 'error'


def run_recrop_pipeline(source_bucket_name, dest_bucket_name, num_workers, gcs_prefix=None, limit=None):
    """Main orchestrator for the re-cropping pipeline."""

    print("--- Initializing Recrop Pipeline ---")
    print(f"Source Bucket: gs://{source_bucket_name}/{gcs_prefix or ''}")
    print(f"Destination Bucket: gs://{dest_bucket_name}/{gcs_prefix or ''}")
    print(f"Using {num_workers} parallel workers.")

    # Initialize GCS clients and the YOLO model (once, on the main thread)
    try:
        storage_client = storage.Client()
        fs = fsspec.filesystem('gcs')
        print("Initializing YOLO model...")
        yolo_model = initialize_yolo_model()
        device = "GPU" if next(yolo_model.parameters()).is_cuda else "CPU"
        print(f"YOLO model loaded successfully on {device}.")
    except Exception as e:
        print(f"\n[FATAL] Initialization failed: {e}")
        return

    # Discover all frames to be processed
    print("Discovering frames in source bucket (this may take a moment)...")
    source_bucket = storage_client.bucket(source_bucket_name)
    blobs = list(source_bucket.list_blobs(prefix=gcs_prefix))

    tasks = []
    for blob in blobs:
        if not blob.name.endswith('/'):  # Ignore "folders"
            source_gcs_path = f"gs://{source_bucket_name}/{blob.name}"
            dest_gcs_path = f"gs://{dest_bucket_name}/{blob.name}"
            tasks.append((source_gcs_path, dest_gcs_path))

    if limit:
        print(f"Applying limit: processing only the first {limit} frames.")
        tasks = tasks[:limit]

    print(f"Found {len(tasks)} total frames to process.")
    if not tasks:
        print("No frames found. Exiting.")
        return

    # Run the processing in a parallel thread pool
    processing_func = partial(process_single_frame, fs=fs, yolo_model=yolo_model)
    status_counts = Counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(processing_func, tasks), total=len(tasks), desc="Re-cropping frames"))

    status_counts.update(results)

    # Final Report
    print("\n--- Processing Complete ---")
    print(f"Total frames processed:      {status_counts['processed']}")
    print(f"Frames skipped (already done): {status_counts['skipped']}")
    print(f"Frames with no face detected:  {status_counts['no_face']}")
    print(f"Frames with read errors:       {status_counts['read_error']}")
    print(f"Frames with other errors:      {status_counts['error']}")
    print("-----------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Re-crop an image dataset to a standard Relative Face Area.")
    parser.add_argument('--source_bucket', required=True, type=str, help="Name of the GCS source bucket.")
    parser.add_argument('--dest_bucket', required=True, type=str, help="Name of the GCS destination bucket.")
    parser.add_argument('--prefix', type=str, default=None,
                        help="Optional GCS prefix to process only a subset (e.g., 'real/').")
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help=f"Number of parallel workers. Default: {DEFAULT_WORKERS}")
    parser.add_argument('--limit', type=int, default=None, help="Process only the first N files (for testing).")
    args = parser.parse_args()

    run_recrop_pipeline(args.source_bucket, args.dest_bucket, args.workers, args.prefix, args.limit)


if __name__ == "__main__":
    main()