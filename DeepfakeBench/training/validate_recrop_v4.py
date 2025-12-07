"""
validate_recrop_v4.py
=====================
A self-contained, out-of-the-box script to quickly validate the results
of the recropping pipeline by comparing a sample of frames from the source
and destination GCS buckets.

Version 4: Robust path handling and improved error logging.
- Fix: Replaces brittle path splitting with robust string replacement to
  correctly construct destination paths.
- Feature: Adds detailed traceback logging to make debugging any future
  errors much easier.
"""
import os
import argparse
import concurrent.futures
import threading
import traceback
from functools import partial

import cv2
import fsspec
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from video_preprocessor import initialize_yolo_model, _get_yolo_face_box
except ImportError:
    print("\n[FATAL ERROR] Could not import 'video_preprocessor.py'. Please ensure it's in the same directory.")
    exit(1)

# --- Configuration ---
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}
DEFAULT_METHODS_TO_CHECK = [
    'blendface', 'danet', 'ddim', 'DiT', 'e4s', 'facedancer', 'faceswap',
    'facevid2vid', 'fomm', 'fsgan', 'hyperreenact', 'inswap', 'lia', 'mcnet',
    'mobileswap', 'MRAA', 'one_shot_free', 'RDDM', 'SiT', 'StyleGAN2', 'StyleGAN3'
]
SAMPLES_PER_METHOD = 5
DEFAULT_WORKERS = 16
DEFAULT_PREFIX = 'fake/'

# --- Thread-Safe YOLO Initializer ---
thread_local = threading.local()


def get_yolo_for_thread():
    if not hasattr(thread_local, "yolo_model"):
        thread_local.yolo_model = initialize_yolo_model()
    return thread_local.yolo_model


# --- Core Analysis Logic ---
def analyze_single_image(task: tuple, fs: fsspec.AbstractFileSystem) -> dict | None:
    source_type, method, gcs_path = task
    yolo_model = get_yolo_for_thread()
    try:
        with fs.open(gcs_path, 'rb') as f:
            image_bytes = np.frombuffer(f.read(), np.uint8)
        image_np_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image_np_bgr is None:
            tqdm.write(f"[Warning] Could not decode image: {gcs_path}")
            return None
        image_area = image_np_bgr.shape[0] * image_np_bgr.shape[1]
        bbox = _get_yolo_face_box(image_np_bgr, yolo_model)
        rfa = 0.0
        if bbox is not None and image_area > 0:
            face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            rfa = face_area / image_area
        return {"source": source_type, "method": method, "relative_face_area": rfa}
    except Exception:
        # --- IMPROVED ERROR LOGGING ---
        tqdm.write(f"\n--- [CRITICAL ERROR] ---")
        tqdm.write(f"Failed during processing of task: {(source_type, method, gcs_path)}")
        tqdm.write("Full Traceback:")
        tqdm.write(traceback.format_exc())
        tqdm.write("--- [END OF ERROR] ---\n")
        return None


def gather_validation_tasks(args, fs) -> list:
    print("Discovering sample files (searching for images in subdirectories)...")
    tasks = []

    # --- ROBUST PATH CONSTRUCTION (THE FIX) ---
    source_root = f"{args.source_bucket}/"
    dest_root = f"gs://{args.dest_bucket}/"

    for method in tqdm(args.methods_to_check, desc="Gathering Tasks"):
        prefix = f"{args.prefix}{method}/"
        source_path_pattern = f"gs://{args.source_bucket}/{prefix}**/*"

        try:
            all_files = fs.glob(source_path_pattern)
            image_files = sorted([f for f in all_files if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS])
            sample_files = image_files[:args.samples_per_method]

            if not sample_files:
                tqdm.write(f"[Warning] No image files found for method '{method}' in source bucket.")
                continue

            for source_gcs_path_full in sample_files:
                # Correctly construct the relative path by removing only the bucket name
                # Example: 'df40-frames/fake/blendface/...' -> 'fake/blendface/...'
                relative_path = source_gcs_path_full.replace(source_root, '', 1)

                # Construct the full GCS paths for both source and destination
                source_gcs_uri = f"gs://{source_gcs_path_full}"
                dest_gcs_uri = f"{dest_root}{relative_path}"

                tasks.append(('original', method, source_gcs_uri))
                tasks.append(('recropped', method, dest_gcs_uri))
        except Exception:
            tqdm.write(f"\n[ERROR] Failed to list files for method '{method}'. Traceback:")
            tqdm.write(traceback.format_exc())

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Quickly validate the recropping pipeline by comparing GCS buckets.")
    parser.add_argument('--source_bucket', required=True, type=str, help="The original GCS bucket.")
    parser.add_argument('--dest_bucket', required=True, type=str, help="The new, recropped GCS bucket.")
    parser.add_argument('--methods_to_check', nargs='+', default=DEFAULT_METHODS_TO_CHECK,
                        help="Space-separated list of method folders to check.")
    parser.add_argument('--prefix', type=str, default=DEFAULT_PREFIX,
                        help="The parent prefix for the method folders (e.g., 'fake/').")
    parser.add_argument('--samples_per_method', type=int, default=SAMPLES_PER_METHOD,
                        help="Number of frames to sample per method.")
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS, help="Number of parallel workers.")

    args = parser.parse_args()

    print("--- Recrop Validation Started (v4) ---")
    print(f"Source: gs://{args.source_bucket}")
    print(f"Destination: gs://{args.dest_bucket}")
    print("-" * 35)

    fs = fsspec.filesystem('gcs')
    tasks = gather_validation_tasks(args, fs)

    if not tasks:
        print("\nNo validation tasks could be created. Please check bucket names and prefixes.")
        return

    print(f"\nFound {len(tasks)} total images to analyze ({len(tasks) // 2} pairs). Starting analysis...")

    processing_func = partial(analyze_single_image, fs=fs)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for res in tqdm(executor.map(processing_func, tasks), total=len(tasks), desc="Analyzing Images"):
            if res:
                results.append(res)

    if not results:
        print("\nAnalysis complete, but no data was collected. Check for processing errors above.")
        return

    df = pd.DataFrame(results)
    report_df = df.groupby(['method', 'source'])['relative_face_area'].agg(
        ['mean', 'median', 'std', 'count']).sort_index()

    for col in ['mean', 'median', 'std']:
        report_df[col] = report_df[col].map('{:.3f}'.format)

    print("\n\n--- Validation Report: Relative Face Area (RFA) ---")
    print("Comparing 'original' vs. 'recropped' samples.")
    print("-" * 55)
    with pd.option_context('display.max_rows', None, 'display.width', 1000):
        print(report_df)
    print("-" * 55)

    try:
        median_original = df[df['source'] == 'original']['relative_face_area'].median()
        median_recropped = df[df['source'] == 'recropped']['relative_face_area'].median()
        print("\n--- Verdict ---")
        print(f"Median RFA of Original Samples:  {median_original:.3f}")
        print(f"Median RFA of Recropped Samples: {median_recropped:.3f}")

        if median_recropped > median_original * 1.1 and 0.75 < median_recropped < 0.90:
            print("\n✅ SUCCESS: The recropped images show a clear and consistent increase in RFA.")
        else:
            print("\n⚠️  WARNING: The results are not as expected. Please review the recrop script's logs.")
    except (KeyError, IndexError):
        print("\nCould not generate final verdict due to missing data from one of the sources.")


if __name__ == "__main__":
    main()
