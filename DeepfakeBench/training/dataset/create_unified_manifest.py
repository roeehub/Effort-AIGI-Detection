#!/usr/bin/env python3
"""
create_unified_manifest.py

Scans multiple GCS buckets to create a single, unified manifest of frame paths.
It then processes each frame in parallel to compute key properties (sharpness,
file size, IDs) and enriches the data with a 'method_category'.

Key Features:
- Aggregates frame paths from a list of GCS buckets.
- Caches the aggregated path list to a JSON file to avoid slow, costly rescans.
- Standardizes method names (e.g., "Hey Gen" -> "hey_gen") for consistency.
- Calculates properties (sharpness, file_size_kb, video_id, clip_id) in parallel.
- Adds a 'method_category' column for simplified downstream sampling.
"""
import argparse
import json
import logging
import sys
import io
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
# --- METHOD & CATEGORY DEFINITIONS ---
# ==============================================================================
# These sets contain STANDARDIZED method names (lowercase_snake_case)
# Used by get_method_category() to assign a 'BASE' category.
# Any method not in these lists will have its own name as its category.

BASE_FF_METHODS: Set[str] = {
    'deepfakes', 'face2face', 'faceshifter', 'faceswap', 'neuraltextures'
}
BASE_DF40_EFS_METHODS: Set[str] = {
    "dit", "sit", "ddim", "rddm", "vqgan", "stylegan2", "stylegan3", "styleganxl",
}
BASE_DF40_REG_METHODS: Set[str] = {
    "simswap", "fsgan", "fomm", "facedancer", "inswap", "one_shot_free",
    "blendface", "lia", "mobileswap", "mcnet", "uniface", "mraa", "facevid2vid",
    "wav2lip", "sadtalker", "danet", "e4s", "pirender", "tpsm",
}


def standardize_method_name(name: str) -> str:
    """Converts a method name to a standard format: lowercase_snake_case."""
    return name.lower().replace(' ', '_').replace('-', '_')


def get_method_category(standardized_method: str) -> str:
    """
    Assigns a category to a method.
    - If the method belongs to a known base set, assign the base category.
    - Otherwise, the method's own name becomes its category (e.g., 'hey_gen').
    """
    if standardized_method in BASE_FF_METHODS:
        return 'BASE_FF++'
    if standardized_method in BASE_DF40_EFS_METHODS:
        return 'BASE_DF40_EFS'
    if standardized_method in BASE_DF40_REG_METHODS:
        return 'BASE_DF40_REG'
    return standardized_method  # For novel methods, the category is the method itself


# ==============================================================================
# --- ID EXTRACTION FUNCTIONS (Unchanged) ---
# ==============================================================================
def get_video_identity(label: str, method: str, video_id: str) -> int:
    is_synthetic = method in BASE_DF40_EFS_METHODS
    is_unmappable_real = label == "real" and method not in {"faceforensics++"}

    if not is_synthetic and not is_unmappable_real:
        try:
            if method == "wav2lip" and '_' in video_id:
                return int(video_id.split('_')[0])
            if '_' in video_id:
                return int(video_id.split('_')[1])
            return int(video_id)
        except (ValueError, IndexError):
            pass
    hashed_id = hash((method, video_id))
    return (hashed_id & 0x7FFFFFFF) + 100_000


def get_clip_id(method: str, original_video_id: str) -> int:
    hashed_id = hash((method, original_video_id))
    return hashed_id & 0x7FFFFFFF


# ==============================================================================
# --- CORE PROCESSING LOGIC ---
# ==============================================================================
def calculate_sharpness(image_bytes: bytes) -> float:
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('L')
        cv_image = np.array(pil_image)
        return float(cv2.Laplacian(cv_image, cv2.CV_64F).var())
    except (UnidentifiedImageError, cv2.error):
        return -1.0


def process_path(gcs_path: str, fs: fsspec.AbstractFileSystem) -> Optional[Dict]:
    """Processes a single GCS path to extract metadata and compute properties."""
    try:
        parts = Path(gcs_path).parts
        # Expected structure: gs://BUCKET/label/method/video_id/frame.png
        # So parts[-4] is label, parts[-3] is method, etc.
        if len(parts) < 5:
            log.warning(f"Skipping malformed path (too short): {gcs_path}")
            return None

        # --- Path Parsing & Standardization ---
        label, method_raw, video_id_raw = parts[-4], parts[-3], parts[-2]
        method = standardize_method_name(method_raw)

        # --- ID Generation ---
        unified_id = get_video_identity(label, method, video_id_raw)
        clip_id = get_clip_id(method, video_id_raw)

        # --- Property Calculation ---
        info = fs.info(gcs_path)
        file_size_kb = info.get("size", 0) / 1024.0

        with fs.open(gcs_path, 'rb') as f:
            image_bytes = f.read()

        sharpness = calculate_sharpness(image_bytes)
        if sharpness < 0:
            log.warning(f"Could not calculate sharpness for {gcs_path}, skipping.")
            return None

        return {
            "path": gcs_path, "label": label, "method": method,
            "original_video_id": video_id_raw,
            "video_id": unified_id,
            "clip_id": clip_id,
            "file_size_kb": round(file_size_kb, 2),
            "sharpness": round(sharpness, 2),
        }
    except Exception as e:
        log.error(f"Unexpected error processing {gcs_path}: {e}")
        return None


# ==============================================================================
# --- NEW: PATH AGGREGATION & CACHING ---
# ==============================================================================

def _scan_bucket(bucket_name: str, fs: fsspec.AbstractFileSystem) -> List[str]:
    """Scans a single GCS bucket for image files."""
    full_bucket_path = f"gs://{bucket_name}"
    log.info(f"Scanning bucket: {full_bucket_path}...")
    try:
        # Use fs.find for recursive search, more efficient than glob for deep trees
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


def aggregate_paths_from_buckets(
        bucket_names: List[str],
        cache_path: Path,
        workers: int,
        force_rescan: bool = False
) -> List[str]:
    """
    Aggregates all frame paths from a list of GCS buckets, using a JSON cache.
    """
    if not force_rescan and cache_path.exists():
        log.info(f"Loading cached frame paths from '{cache_path}'...")
        with open(cache_path, 'r') as f:
            all_paths = json.load(f)
        log.info(f"Loaded {len(all_paths):,} paths from cache.")
        return all_paths

    log.info("Cache not found or --force-rescan enabled. Scanning GCS buckets...")
    fs = fsspec.filesystem("gcs")
    all_paths = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        func = partial(_scan_bucket, fs=fs)
        future_to_bucket = {executor.submit(func, name): name for name in bucket_names}

        progress_bar = tqdm(as_completed(future_to_bucket), total=len(bucket_names), desc="Scanning Buckets")
        for future in progress_bar:
            bucket_name = future_to_bucket[future]
            try:
                paths = future.result()
                all_paths.extend(paths)
                progress_bar.set_postfix({"bucket": bucket_name, "found": f"{len(paths):,}"})
            except Exception as e:
                log.error(f"Error processing bucket {bucket_name}: {e}")

    log.info(f"Total paths aggregated from all buckets: {len(all_paths):,}")
    log.info(f"Writing aggregated path list to cache: '{cache_path}'")
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(all_paths, f)
    except Exception as e:
        log.error(f"Failed to write path cache file: {e}")

    return all_paths


# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Create a unified, enriched property manifest from multiple GCS buckets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--buckets", type=str, nargs='+', required=True,
                        help="List of GCS bucket names to scan (e.g., df40-frames... faceforensics_pp...).")
    parser.add_argument("--path-cache", type=str, required=True,
                        help="Path to cache the aggregated list of frame paths (e.g., ./master_paths.json).")
    parser.add_argument("--output-manifest", type=str, required=True,
                        help="Path to the final output Parquet manifest (e.g., ./frame_properties.parquet).")
    parser.add_argument("--workers", type=int, default=64,
                        help="Number of parallel worker threads for processing frames.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N records for testing.")
    parser.add_argument("--force-rescan", action="store_true",
                        help="Ignore the path cache and force a full rescan of all GCS buckets.")
    args = parser.parse_args()

    # --- Phase 1: Aggregate all frame paths from GCS (with caching) ---
    all_paths = aggregate_paths_from_buckets(
        bucket_names=args.buckets,
        cache_path=Path(args.path_cache),
        workers=args.workers,
        force_rescan=args.force_rescan
    )
    if not all_paths:
        log.error("No paths were found. Exiting.")
        sys.exit(1)

    if args.limit:
        log.warning(f"Processing limited to the first {args.limit:,} paths.")
        all_paths = all_paths[:args.limit]

    # --- Phase 2: Process all paths in parallel to get properties ---
    fs = fsspec.filesystem("gcs")
    log.info(f"Starting property calculation for {len(all_paths):,} frames with {args.workers} workers...")
    results_list, error_count = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        func = partial(process_path, fs=fs)
        futures = {executor.submit(func, path) for path in all_paths}
        progress_bar = tqdm(as_completed(futures), total=len(all_paths), desc="Processing Frames")
        for future in progress_bar:
            result = future.result()
            if result:
                results_list.append(result)
            else:
                error_count += 1
            progress_bar.set_postfix({"success": len(results_list), "errors": error_count})

    log.info("------ Property Calculation Complete ------")
    if not results_list:
        log.warning("No valid data was processed. Output file will not be created.")
        sys.exit(1)

    # --- Phase 3: Create DataFrame and add the method_category column ---
    log.info(f"Aggregating {len(results_list):,} results into a DataFrame...")
    df = pd.DataFrame(results_list)

    log.info("Adding 'method_category' column...")
    df['method_category'] = df['method'].apply(get_method_category)

    # --- Phase 4: Save the final, enriched manifest ---
    output_path = Path(args.output_manifest)
    log.info(f"Writing final DataFrame to Parquet file: '{output_path}'")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Reorder columns for better readability
        cols = [
            'path', 'label', 'method', 'method_category', 'video_id', 'clip_id',
            'original_video_id', 'sharpness', 'file_size_kb'
        ]
        other_cols = [c for c in df.columns if c not in cols]
        df = df[cols + other_cols]
        df.to_parquet(output_path, index=False, engine='pyarrow', compression='snappy')
        log.info(f"Successfully saved the unified manifest to '{output_path}'.")
    except Exception as e:
        log.error(f"Failed to write Parquet file: {e}")
        sys.exit(1)

    log.info(f"Encountered {error_count:,} processing errors (see logs for details).")


if __name__ == "__main__":
    main()
