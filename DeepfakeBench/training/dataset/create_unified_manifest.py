#!/usr/bin/env python3
"""
create_unified_manifest.py

Scans GCS buckets and/or local directories to create or update a unified manifest
of frame properties.

Key Features:
- Aggregates frame paths from multiple GCS buckets and local directories.
- Caches the GCS path list to a JSON file to avoid slow, costly rescans.
- Appends to an existing manifest, skipping already-processed paths (--existing-manifest).
- Supports adding only frames from newly discovered methods (--add-new-methods).
- Standardizes method names (e.g., "Hey Gen" -> "hey_gen") for consistency.
- Calculates properties (sharpness, file_size_kb, video_id, clip_id) in parallel.
- Adds a 'method_category' column for simplified downstream sampling.
"""
import argparse
import json
import logging
import sys
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Set, List

# --- Third-party imports ---
try:
    import fsspec
    import numpy as np
    import pandas as pd
    from PIL import Image, UnidentifiedImageError
    import cv2
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Missing required library -> {e.name}")
    print("Please install dependencies: pip install pandas pyarrow opencv-python-headless gcsfs Pillow tqdm")
    sys.exit(1)

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

# ==============================================================================
# --- METHOD & CATEGORY DEFINITIONS (Unchanged) ---
# ==============================================================================
# These sets contain STANDARDIZED method names (lowercase_snake_case)
BASE_FF_METHODS: Set[str] = {
    'deepfakes', 'face2face', 'faceshifter', 'faceswap', 'neuraltextures',
    'deepfakedetection',
}
BASE_DF40_EFS_METHODS: Set[str] = {
    "dit", "sit", "ddim", "rddm", "vqgan", "stylegan2", "stylegan3", "styleganxl",
}
BASE_DF40_REG_METHODS: Set[str] = {
    "simswap", "fsgan", "fomm", "facedancer", "inswap", "one_shot_free",
    "blendface", "lia", "mobileswap", "mcnet", "uniface", "mraa", "facevid2vid",
    "wav2lip", "sadtalker", "danet", "e4s", "pirender", "tpsm", "hyperreenact",
    "youtube_real", "faceforensics++", "celeb_real",
}


def standardize_method_name(name: str) -> str:
    """Converts a method name to a standard format: lowercase_snake_case."""
    return name.lower().replace(' ', '_').replace('-', '_')


def get_method_category(standardized_method: str) -> str:
    """Assigns a category to a method."""
    if standardized_method in BASE_FF_METHODS: return 'BASE_FF++'
    if standardized_method in BASE_DF40_EFS_METHODS: return 'BASE_DF40_EFS'
    if standardized_method in BASE_DF40_REG_METHODS: return 'BASE_DF40_REG'
    return standardized_method


# ==============================================================================
# --- ID EXTRACTION FUNCTIONS (Unchanged) ---
# ==============================================================================
def get_video_identity(label: str, method: str, video_id: str) -> int:
    is_synthetic = method in BASE_DF40_EFS_METHODS
    is_unmappable_real = label == "real" and method not in {"faceforensics++"}
    if not is_synthetic and not is_unmappable_real:
        try:
            if method == "wav2lip" and '_' in video_id: return int(video_id.split('_')[0])
            if '_' in video_id: return int(video_id.split('_')[1])
            return int(video_id)
        except (ValueError, IndexError):
            pass
    hashed_id = hash((method, video_id))
    return (hashed_id & 0x7FFFFFFF) + 100_000


def get_clip_id(method: str, original_video_id: str) -> int:
    hashed_id = hash((method, original_video_id))
    return hashed_id & 0x7FFFFFFF


# ==============================================================================
# --- CORE PROCESSING LOGIC (Refactored for Protocol-Agnosticism) ---
# ==============================================================================
def calculate_sharpness(image_bytes: bytes) -> float:
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('L')
        cv_image = np.array(pil_image)
        return float(cv2.Laplacian(cv_image, cv2.CV_64F).var())
    except (UnidentifiedImageError, cv2.error):
        return -1.0


# *** NEW UTILITY FUNCTION ***
def get_method_from_path(path_str: str) -> Optional[str]:
    """
    Quickly extracts and standardizes the method name from a path string.
    Assumes path structure: .../{label}/{method}/{video_id}/{frame_name}.png
    """
    try:
        # parts[-1] is filename, parts[-2] is video_id, parts[-3] is method
        method_raw = Path(path_str).parts[-3]
        return standardize_method_name(method_raw)
    except IndexError:
        log.warning(f"Could not extract method from malformed path: {path_str}")
        return None


def process_path(path_str: str) -> Optional[Dict]:
    """
    Processes a single path (local or GCS) to extract metadata and compute properties.
    NOTE: Assumes the path structure is '.../{label}/{method}/{video_id}/{frame_name}.png'
    """
    try:
        # Use fsspec to open the file, which handles local and GCS paths transparently
        with fsspec.open(path_str, 'rb') as f:
            # Get filesystem object to query info
            fs = f.fs
            info = fs.info(path_str)
            file_size_kb = info.get("size", 0) / 1024.0
            image_bytes = f.read()

        parts = Path(path_str).parts
        if len(parts) < 4:
            log.warning(f"Skipping malformed path (too short): {path_str}")
            return None

        label, method_raw, video_id_raw = parts[-4], parts[-3], parts[-2]
        method = standardize_method_name(method_raw)

        unified_id = get_video_identity(label, method, video_id_raw)
        clip_id = get_clip_id(method, video_id_raw)

        sharpness = calculate_sharpness(image_bytes)
        if sharpness < 0:
            log.warning(f"Could not calculate sharpness for {path_str}, skipping.")
            return None

        return {
            "path": path_str, "label": label, "method": method,
            "original_video_id": video_id_raw,
            "video_id": unified_id, "clip_id": clip_id,
            "file_size_kb": round(file_size_kb, 2),
            "sharpness": round(sharpness, 2),
        }
    except Exception as e:
        log.error(f"Unexpected error processing {path_str}: {e}")
        return None


# ==============================================================================
# --- PATH AGGREGATION & CACHING ---
# ==============================================================================
def _scan_bucket_gcs(bucket_name: str, fs: fsspec.AbstractFileSystem) -> List[str]:
    """Scans a single GCS bucket for image files."""
    full_bucket_path = f"gs://{bucket_name}"
    log.info(f"Scanning GCS bucket: {full_bucket_path}...")
    try:
        paths_dict = fs.find(full_bucket_path, detail=False)
        image_paths = [
            f"gs://{p}" for p in paths_dict
            if Path(p).suffix.lower() in {'.png', '.jpg', '.jpeg'}
        ]
        log.info(f"Found {len(image_paths):,} image paths in {bucket_name}.")
        return image_paths
    except Exception as e:
        log.error(f"Failed to scan bucket {bucket_name}: {e}")
        return []


def aggregate_paths_from_gcs(
        bucket_names: List[str], cache_path: Path, workers: int, force_rescan: bool = False
) -> List[str]:
    """Aggregates all frame paths from a list of GCS buckets, using a JSON cache."""
    if not force_rescan and cache_path.exists():
        log.info(f"Loading cached GCS frame paths from '{cache_path}'...")
        with open(cache_path, 'r') as f: all_paths = json.load(f)
        log.info(f"Loaded {len(all_paths):,} GCS paths from cache.")
        return all_paths

    log.info("GCS cache not found or --force-rescan enabled. Scanning GCS buckets...")
    fs = fsspec.filesystem("gcs")
    all_paths = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        func = partial(_scan_bucket_gcs, fs=fs)
        future_to_bucket = {executor.submit(func, name): name for name in bucket_names}
        progress = tqdm(as_completed(future_to_bucket), total=len(bucket_names), desc="Scanning GCS Buckets")
        for future in progress:
            bucket_name = future_to_bucket[future]
            try:
                paths = future.result()
                all_paths.extend(paths)
                progress.set_postfix({"bucket": bucket_name, "found": f"{len(paths):,}"})
            except Exception as e:
                log.error(f"Error processing bucket {bucket_name}: {e}")

    log.info(f"Writing aggregated GCS path list to cache: '{cache_path}'")
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(all_paths, f)
    except Exception as e:
        log.error(f"Failed to write path cache file: {e}")
    return all_paths


def aggregate_paths_from_local(local_dirs: List[str]) -> List[str]:
    """Scans local directories recursively for image files."""
    log.info(f"Scanning local directories: {local_dirs}")
    all_paths = []
    for dir_path in local_dirs:
        log.info(f"Scanning '{dir_path}'...")
        path_obj = Path(dir_path)
        if not path_obj.is_dir():
            log.warning(f"Skipping non-existent directory: {dir_path}")
            continue

        image_files = []
        for ext in ('**/*.png', '**/*.jpg', '**/*.jpeg'):
            image_files.extend(path_obj.glob(ext))

        # Convert Path objects to absolute path strings
        abs_paths = [str(p.resolve()) for p in image_files]
        all_paths.extend(abs_paths)
        log.info(f"Found {len(abs_paths):,} images in '{dir_path}'.")
    return all_paths


# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Create or update a unified, enriched property manifest from GCS and/or local sources.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Source Arguments ---
    parser.add_argument("--buckets", type=str, nargs='+', default=[],
                        help="List of GCS bucket names to scan (e.g., df40-frames...).")
    parser.add_argument("--local-dirs", type=str, nargs='+', default=[],
                        help="List of local directories to scan. IMPORTANT: Must follow the .../{label}/{method}/{video_id}/ structure.")
    # --- Input/Output Arguments ---
    parser.add_argument("--output-manifest", type=str, required=True,
                        help="Path to the final output Parquet manifest (e.g., ./frame_properties.parquet).")
    parser.add_argument("--existing-manifest", type=str, default=None,
                        help="Path to an existing Parquet manifest to append new data to. Will skip any paths already present.")
    parser.add_argument("--path-cache", type=str, default="./gcs_paths_cache.json",
                        help="Path to cache the aggregated list of GCS frame paths.")
    # --- Control Arguments ---
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8,
                        help="Number of parallel worker threads for processing frames.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N new records for testing.")
    parser.add_argument("--force-rescan", action="store_true",
                        help="Ignore the GCS path cache and force a full rescan of all GCS buckets.")
    # *** NEW ARGUMENT ***
    parser.add_argument("--add-new-methods", action="store_true",
                        help="Only process and add frames from methods not present in the --existing-manifest. "
                             "This ignores all new frames from existing methods.")
    args = parser.parse_args()

    # *** NEW VALIDATION LOGIC ***
    if args.add_new_methods and not args.existing_manifest:
        parser.error("--add-new-methods requires --existing-manifest to be specified.")

    if not args.buckets and not args.local_dirs:
        parser.error("You must provide at least one data source: --buckets or --local-dirs")

    # --- 1. Load Existing Manifest (if provided) ---
    existing_df = None
    existing_paths: Set[str] = set()
    existing_methods: Set[str] = set()  # *** NEW ***
    if args.existing_manifest:
        existing_path = Path(args.existing_manifest)
        if existing_path.exists():
            log.info(f"Loading existing manifest from '{existing_path}' to append data.")
            existing_df = pd.read_parquet(existing_path)
            existing_paths = set(existing_df['path'])
            existing_methods = set(existing_df['method'])  # *** NEW ***
            log.info(
                f"Found {len(existing_paths):,} paths and {len(existing_methods)} unique methods in the existing manifest.")
        else:
            log.warning(f"Existing manifest '{existing_path}' not found. A new one will be created.")

    # --- 2. Aggregate All Paths from Sources ---
    all_new_source_paths = []
    if args.buckets:
        all_new_source_paths.extend(aggregate_paths_from_gcs(
            bucket_names=args.buckets, cache_path=Path(args.path_cache),
            workers=args.workers, force_rescan=args.force_rescan
        ))
    if args.local_dirs:
        all_new_source_paths.extend(aggregate_paths_from_local(args.local_dirs))

    log.info(f"Total paths found from all sources: {len(all_new_source_paths):,}")

    # --- 3. Filter Out Duplicates (based on selected mode) ---
    paths_to_process = []

    # *** REVISED FILTERING LOGIC ***
    if args.add_new_methods:
        log.info("Filtering mode: --add-new-methods. Will only process frames from newly discovered methods.")
        newly_discovered_methods = set()
        for path in tqdm(all_new_source_paths, desc="Discovering new methods"):
            method = get_method_from_path(path)
            if method and method not in existing_methods:
                paths_to_process.append(path)
                newly_discovered_methods.add(method)

        if newly_discovered_methods:
            log.info(
                f"Discovery complete! Found {len(newly_discovered_methods)} new methods to add: {sorted(list(newly_discovered_methods))}")
        else:
            log.info("Discovery complete. No new methods found in the source paths.")

    else:
        log.info("Filtering mode: Standard append. Will skip any paths already in the manifest.")
        paths_to_process = [p for p in all_new_source_paths if p not in existing_paths]
        num_skipped = len(all_new_source_paths) - len(paths_to_process)
        if num_skipped > 0:
            log.info(f"Skipping {num_skipped:,} paths that are already in the existing manifest.")

    if not paths_to_process:
        log.info("No new paths to process. Exiting.")
        sys.exit(0)

    if args.limit:
        log.warning(f"Processing limited to the first {args.limit:,} new paths.")
        paths_to_process = paths_to_process[:args.limit]

    # --- 4. Process New Paths in Parallel ---
    log.info(f"Starting property calculation for {len(paths_to_process):,} new frames with {args.workers} workers...")
    new_results, error_count = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_path, path) for path in paths_to_process}
        progress = tqdm(as_completed(futures), total=len(paths_to_process), desc="Processing New Frames")
        for future in progress:
            result = future.result()
            if result:
                new_results.append(result)
            else:
                error_count += 1
            progress.set_postfix({"success": len(new_results), "errors": error_count})

    log.info("------ Property Calculation Complete ------")
    if not new_results:
        log.warning("No valid data was processed from new sources. Output file will not be updated.")
        sys.exit(1)

    # --- 5. Create DataFrame for New Data and Combine with Existing ---
    log.info(f"Aggregating {len(new_results):,} new results into a DataFrame...")
    new_df = pd.DataFrame(new_results)
    log.info("Adding 'method_category' column to new data...")
    new_df['method_category'] = new_df['method'].apply(get_method_category)

    if existing_df is not None:
        log.info(f"Combining {len(existing_df):,} existing records with {len(new_df):,} new records.")
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    # --- 6. Save Final Manifest ---
    output_path = Path(args.output_manifest)
    log.info(f"Writing final DataFrame ({len(final_df):,} total records) to Parquet file: '{output_path}'")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cols = ['path', 'label', 'method', 'method_category', 'video_id', 'clip_id', 'original_video_id', 'sharpness',
                'file_size_kb']
        final_df = final_df[cols]  # Enforce column order
        final_df.to_parquet(output_path, index=False, engine='pyarrow', compression='snappy')
        log.info(f"Successfully saved the unified manifest to '{output_path}'.")
    except Exception as e:
        log.error(f"Failed to write Parquet file: {e}")
        sys.exit(1)

    log.info(f"Encountered {error_count:,} processing errors (see logs for details).")


if __name__ == "__main__":
    main()
