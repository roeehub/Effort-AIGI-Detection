"""
recrop_dataset.py (v4.2 - Production Ready with Pre-creation Mode)
================================================
A high-performance, resumable script to re-process an entire image dataset to
a standardized Relative Face Area (RFA).

New in this version:
- Feature: Adds a '--mode precreate_dirs' to create the entire local
  directory structure in advance. This is a powerful tool for debugging
  filesystem race conditions in multithreaded environments.
- Fix: All YOLO model initialization is now handled by a thread-safe
  initializer to prevent any potential conflicts.
"""
import os
import argparse
import concurrent.futures
import random
import pickle
import threading
from pathlib import Path
from functools import partial
from collections import Counter
from typing import Set

import cv2
import fsspec
import numpy as np
import torch
from google.cloud import storage
from tqdm import tqdm

try:
    from video_preprocessor import initialize_yolo_model, _get_yolo_face_box
except ImportError:
    print("\n[ERROR] Could not import from 'video_preprocessor.py'. Please ensure it's in the same directory.")
    exit(1)

# --- Configuration ---
TARGET_RFA = 0.85
FINAL_IMAGE_SIZE = (224, 224)
DEFAULT_WORKERS = 32
IMAGE_QUALITY = 90
DEFAULT_LOCAL_SAMPLE_SIZE = 200

# --- Thread-Safe YOLO Model Initializer ---
# This ensures each thread gets its own model instance if needed, preventing conflicts.
thread_local = threading.local()


def get_yolo_for_thread():
    if not hasattr(thread_local, "yolo_model"):
        # print(f"Initializing YOLO model for thread: {threading.get_ident()}") # Optional: for debugging
        thread_local.yolo_model = initialize_yolo_model()
    return thread_local.yolo_model


# --- Core Logic ---
def get_gcs_blob_names(args) -> list[str]:
    """Gets GCS blob names, using a local cache if available."""
    safe_prefix = (args.prefix or 'noprefix').replace('/', '_')
    cache_file = Path(args.cache_dir) / f'gcs_cache_{args.source_bucket}_{safe_prefix}.pkl'

    if cache_file.exists() and not args.no_cache:
        print(f"Loading file list from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        print("Discovering frames in source bucket (this may take a moment)...")
        storage_client = storage.Client()
        source_bucket = storage_client.bucket(args.source_bucket)
        blob_names = [b.name for b in tqdm(source_bucket.list_blobs(prefix=args.prefix), desc="Listing GCS blobs")]

        os.makedirs(args.cache_dir, exist_ok=True)
        print(f"Caching {len(blob_names)} file paths for future runs at: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(blob_names, f)
        return blob_names


def crop_to_target_rfa(image_np: np.ndarray, yolo_model) -> np.ndarray | None:
    img_height, img_width = image_np.shape[:2]
    bbox = _get_yolo_face_box(image_np, yolo_model)
    if bbox is None: return None
    x0, y0, x1, y1 = bbox
    face_width, face_height = x1 - x0, y1 - y0
    face_area = face_width * face_height
    face_center_x, face_center_y = x0 + face_width / 2, y0 + face_height / 2
    if face_area == 0: return None
    new_crop_side = np.sqrt(face_area / (TARGET_RFA + 1e-6))
    new_x0, new_y0 = int(face_center_x - new_crop_side / 2), int(face_center_y - new_crop_side / 2)
    new_x1, new_y1 = int(face_center_x + new_crop_side / 2), int(face_center_y + new_crop_side / 2)
    new_x0, new_y0 = max(0, new_x0), max(0, new_y0)
    new_x1, new_y1 = min(img_width, new_x1), min(img_height, new_y1)
    final_crop = image_np[new_y0:new_y1, new_x0:new_x1]
    if final_crop.size == 0: return None
    return cv2.resize(final_crop, FINAL_IMAGE_SIZE, interpolation=cv2.INTER_AREA)


def process_single_frame(paths: tuple, fs_in: fsspec.AbstractFileSystem, fs_out: fsspec.AbstractFileSystem) -> str:
    source_path, dest_path = paths
    try:
        yolo_model = get_yolo_for_thread()
        is_local_output = (fs_out.protocol == 'file')

        if is_local_output and os.path.exists(dest_path):
            return 'skipped'
        elif not is_local_output and fs_out.exists(dest_path):
            return 'skipped'

        with fs_in.open(source_path, 'rb') as f:
            image_bytes = np.frombuffer(f.read(), np.uint8)
        image_np_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image_np_bgr is None: return 'read_error'
        recropped_image = crop_to_target_rfa(image_np_bgr, yolo_model)
        if recropped_image is None: return 'no_face'
        _, buffer = cv2.imencode('.jpg', recropped_image, [int(cv2.IMWRITE_JPEG_QUALITY), IMAGE_QUALITY])
        image_data = buffer.tobytes()

        if is_local_output:
            # Pre-creating dirs is recommended, but we still ensure it exists here as a fallback.
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, 'wb') as f:
                f.write(image_data)
        else:  # GCS output
            with fs_out.open(dest_path, 'wb') as f:
                f.write(image_data)
        return 'processed'
    except Exception as e:
        tqdm.write(f"\n[ERROR] Failed during processing of {source_path} -> {dest_path}: {e}")
        return 'error'


# --- NEW: Directory Pre-creation Mode ---
def precreate_local_dirs(args):
    """Scans all blob names and creates the corresponding local directory structure."""
    if not args.local_sample_dir:
        print("[ERROR] --local_sample_dir must be specified for 'precreate_dirs' mode.")
        return

    print("--- Mode: Pre-create Local Directories ---")
    blob_names = get_gcs_blob_names(args)

    print("Identifying unique directory paths to create...")
    # Use a set for automatic deduplication
    dirs_to_create: Set[str] = set()
    for blob_name in blob_names:
        if not blob_name.endswith('/'):  # Ignore GCS "folders"
            dest_path = os.path.join(args.local_sample_dir, blob_name)
            parent_dir = os.path.dirname(dest_path)
            dirs_to_create.add(parent_dir)

    if not dirs_to_create:
        print("No directories to create.")
        return

    print(f"Found {len(dirs_to_create)} unique directories. Creating them now...")
    for directory in tqdm(sorted(list(dirs_to_create)), desc="Creating Dirs"):
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            print(f"\n[ERROR] Could not create directory {directory}: {e}")

    print("\n✅ Directory pre-creation complete.")
    print(f"You can now manually inspect the structure inside '{args.local_sample_dir}'.")
    print("Next, run the script again with '--mode run' to process the images.")


# --- Main Pipeline Runner ---
def run_recrop_pipeline(args):
    print("--- Mode: Run Recrop Pipeline ---")
    is_local_sample_mode = args.local_sample_dir is not None
    if is_local_sample_mode and not args.sample_size:
        args.sample_size = DEFAULT_LOCAL_SAMPLE_SIZE

    print(f"Source Bucket: gs://{args.source_bucket}/{args.prefix or ''}")
    if is_local_sample_mode:
        print(f"Destination (Local Sample): {args.local_sample_dir}")
    else:
        print(f"Destination Bucket: gs://{args.dest_bucket}/{args.prefix or ''}")
    print(f"Using {args.workers} parallel workers.")

    try:
        # Filesystems are lightweight and can be initialized here
        fs_in = fsspec.filesystem('gcs')
        fs_out = fsspec.filesystem('file') if is_local_sample_mode else fsspec.filesystem('gcs')
    except Exception as e:
        print(f"\n[FATAL] Filesystem initialization failed: {e}");
        return

    blob_names = get_gcs_blob_names(args)
    tasks = []
    for blob_name in blob_names:
        if not blob_name.endswith('/'):
            source_path = f"gs://{args.source_bucket}/{blob_name}"
            dest_path = os.path.join(args.local_sample_dir,
                                     blob_name) if is_local_sample_mode else f"gs://{args.dest_bucket}/{blob_name}"
            tasks.append((source_path, dest_path))

    if args.sample_size:
        print(f"\nSampling {min(args.sample_size, len(tasks))} random frames for processing.")
        tasks = random.sample(tasks, min(args.sample_size, len(tasks)))

    print(f"Found {len(tasks)} total frames to process.")
    if not tasks: print("No frames found. Exiting."); return

    # YOLO models are initialized lazily inside each thread by get_yolo_for_thread()
    print("Starting parallel processing. YOLO models will be loaded on-demand in each worker thread.")
    processing_func = partial(process_single_frame, fs_in=fs_in, fs_out=fs_out)
    status_counts = Counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(processing_func, tasks), total=len(tasks), desc="Re-cropping frames"))
    status_counts.update(results)

    print("\n--- Processing Complete ---")
    for status in sorted(status_counts.keys()):
        print(f"{status.replace('_', ' ').title():<28}: {status_counts[status]}")
    print("-----------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Re-crop a dataset to a standard Relative Face Area.")
    parser.add_argument('--mode', default='run', choices=['run', 'precreate_dirs'], help="Operation to perform.")
    parser.add_argument('--source_bucket', required=True, type=str, help="GCS source bucket.")
    parser.add_argument('--prefix', type=str, default=None, help="Optional GCS prefix to process a subset.")
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS, help="Number of parallel workers.")
    parser.add_argument('--cache_dir', type=str, default='./.gcs_cache',
                        help="Directory to store GCS file list caches.")
    parser.add_argument('--no_cache', action='store_true', help="Force re-downloading of GCS file list.")

    # Destination options
    dest_group = parser.add_argument_group('Destination')
    dest_group.add_argument('--dest_bucket', type=str, help="GCS destination bucket (for full run).")
    dest_group.add_argument('--local_sample_dir', type=str, default=None,
                            help="Save a sample to a local directory for verification.")

    # Sampling options
    sample_group = parser.add_argument_group('Sampling')
    sample_group.add_argument('--sample_size', type=int, default=None, help="Process a random sample of N files.")

    args = parser.parse_args()
    if args.mode == 'run' and not args.local_sample_dir and not args.dest_bucket:
        parser.error("For '--mode run', must specify either --dest_bucket or --local_sample_dir.")

    if args.mode == 'precreate_dirs':
        precreate_local_dirs(args)
    elif args.mode == 'run':
        run_recrop_pipeline(args)


if __name__ == "__main__":
    main()
