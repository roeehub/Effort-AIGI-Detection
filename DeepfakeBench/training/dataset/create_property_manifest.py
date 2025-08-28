#!/usr/bin/env python3
"""
create_property_manifest.py

Scans a list of cloud storage frame paths, computes key properties for each frame,
and saves the results to a new, enriched manifest in Apache Parquet format.

Includes a fast '--update-ids-only' mode to re-compute only the source video IDs
on an existing manifest without re-downloading any image data.
"""
import argparse
import json
import logging
import sys
import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Set

# --- Third-party imports ---
try:
    import cv2
    import fsspec
    import numpy as np
    import pandas as pd
    from PIL import Image
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Missing required library -> {e.name}")
    print("Please install dependencies: pip install pandas pyarrow opencv-python-headless gcsfs Pillow tqdm")
    sys.exit(1)

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

# ==============================================================================
# --- CORRECTED, SELF-CONTAINED IDENTITY EXTRACTION FUNCTION ---
# ==============================================================================

# --- Configuration: Method Categories (ddim is correctly placed here) ---
EFS_METHODS: Set[str] = {
    "DiT", "SiT", "ddim", "RDDM", "VQGAN", "StyleGAN2", "StyleGAN3", "StyleGANXL",
}
REG_METHODS: Set[str] = {
    "simswap", "fsgan", "faceswap", "fomm", "facedancer", "inswap", "one_shot_free",
    "blendface", "lia", "mobileswap", "mcnet", "uniface", "MRAA", "facevid2vid",
    "wav2lip", "sadtalker", "danet", "e4s", "pirender", "tpsm",
}
REV_METHODS: Set[str] = {}


def get_video_identity(label: str, method: str, video_id: str) -> int:
    """
    Determines a unique, stable integer identity for a video based on its name.
    Handles multiple formats and falls back to hashing for synthetic videos.
    """
    # Stage 1: Skip parsing for videos that don't have a numeric target ID
    is_synthetic = method in EFS_METHODS
    is_unmappable_real = label == "real" and method not in {"FaceForensics++"}

    if not is_synthetic and not is_unmappable_real:
        try:
            # Special case for 'wav2lip' format 'XXX_...'
            if method == "wav2lip" and '_' in video_id:
                source_id_str = video_id.split('_')[0]
                return int(source_id_str)

            # Case for 'target_source' format 'XXX_YYY'
            if '_' in video_id:
                source_id_str = video_id.split('_')[1]
                return int(source_id_str)

            # Case for simple 'source' format 'XXX'
            else:
                return int(video_id)
        except (ValueError, IndexError):
            pass  # Fallback to hashing if parsing fails

    # Stage 2: Fallback to hashing for synthetic or unmappable videos
    hashed_id = hash((method, video_id))
    positive_hash = hashed_id & 0x7FFFFFFF
    return positive_hash + 100_000


# ==============================================================================
# --- PROPERTY CALCULATION & PROCESSING LOGIC ---
# ==============================================================================

def calculate_sharpness(image_bytes: bytes) -> float:
    """Calculates image sharpness using the variance of the Laplacian."""
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('L')
        cv_image = np.array(pil_image)
        sharpness = cv2.Laplacian(cv_image, cv2.CV_64F).var()
        return float(sharpness)
    except Exception:
        return -1.0


def process_path(gcs_path: str, fs: fsspec.AbstractFileSystem) -> Optional[Dict]:
    """Processes a single GCS path to extract metadata and compute properties."""
    try:
        parts = Path(gcs_path).parts
        if len(parts) < 4:
            log.warning(f"Skipping malformed path (too short): {gcs_path}")
            return None
        label, method, video_id_raw = parts[-4], parts[-3], parts[-2]
        unified_id = get_video_identity(label, method, video_id_raw)

        info = fs.info(gcs_path)
        file_size_kb = info.get("size", 0) / 1024.0

        with fs.open(gcs_path, 'rb') as f:
            image_bytes = f.read()
        sharpness = calculate_sharpness(image_bytes)
        if sharpness < 0:
            return None

        return {
            "path": gcs_path, "label": label, "method": method,
            "original_video_id": video_id_raw,
            "video_id": unified_id,
            "file_size_kb": round(file_size_kb, 2),
            "sharpness": round(sharpness, 2),
        }
    except Exception as e:
        log.error(f"Unexpected error processing {gcs_path}: {e}")
        return None


# ==============================================================================
# --- MODIFIED: FAST ID-ONLY UPDATE FUNCTION ---
# ==============================================================================

def update_manifest_ids(input_path: Path, output_path: Path, workers: int):
    """
    Loads an existing manifest, re-computes the video IDs, and saves the
    updated manifest. The new unified ID is placed in 'video_id', and the
    original is preserved in 'original_video_id'.
    """
    log.info(f"Loading existing manifest from '{input_path}'...")
    try:
        df = pd.read_parquet(input_path)
    except FileNotFoundError:
        log.error(f"Input manifest not found at: {input_path}")
        sys.exit(1)

    if 'original_video_id' not in df.columns:
        log.info("Renaming 'video_id' to 'original_video_id' to preserve it.")
        df.rename(columns={'video_id': 'original_video_id'}, inplace=True)
    else:
        log.info("Found existing 'original_video_id' column. Will use it as source.")

    log.info(f"Loaded {len(df):,} records. Re-computing IDs with {workers} workers...")

    def compute_new_ids(chunk_df: pd.DataFrame) -> pd.Series:
        return chunk_df.apply(
            lambda row: get_video_identity(row['label'], row['method'], row['original_video_id']),
            axis=1
        )

    df_chunks = np.array_split(df, workers)
    new_id_chunks = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(compute_new_ids, chunk) for chunk in df_chunks}
        progress_bar = tqdm(as_completed(futures), total=len(futures), desc="Updating IDs")
        for future in progress_bar:
            new_id_chunks.append(future.result())

    log.info("ID computation complete. Updating DataFrame...")
    df['video_id'] = pd.concat(new_id_chunks)

    log.info(f"Writing updated DataFrame to Parquet file: '{output_path}'")
    try:
        cols = ['path', 'label', 'method', 'video_id', 'original_video_id']
        other_cols = [c for c in df.columns if c not in cols]
        df = df[cols + other_cols]
        df.to_parquet(output_path, index=False, engine='pyarrow', compression='snappy')
        log.info("Successfully saved the updated manifest.")
    except Exception as e:
        log.error(f"Failed to write Parquet file: {e}")
        sys.exit(1)


# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create or update an enhanced property manifest in Parquet format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-manifest", type=str, required=True,
                        help="Path to input. For full mode, a JSON list of paths. For update mode, the Parquet manifest to update.")
    parser.add_argument("--output-manifest", type=str, required=True,
                        help="Path to the output Parquet manifest.")
    parser.add_argument("--workers", type=int, default=32,
                        help="Number of parallel worker threads.")
    parser.add_argument("--update-ids-only", action="store_true",
                        help="Enable fast mode to only update IDs on an existing Parquet manifest.")
    parser.add_argument("--limit", type=int, default=None,
                        help="[Full Mode Only] Process only the first N records for testing.")
    args = parser.parse_args()

    input_path = Path(args.input_manifest)
    output_path = Path(args.output_manifest)

    if args.update_ids_only:
        log.info("--- Running in Fast ID Update-Only Mode ---")
        update_manifest_ids(input_path, output_path, args.workers)
        sys.exit(0)

    log.info("--- Running in Full Processing Mode ---")
    if not input_path.exists():
        log.error(f"Input JSON manifest not found at: {input_path}")
        sys.exit(1)
    if output_path.exists():
        if input(f"Output file '{output_path}' already exists. Overwrite? (y/N): ").lower() != 'y':
            log.info("Operation cancelled.")
            sys.exit(0)

    log.info(f"Loading frame paths from '{input_path}'...")
    with open(input_path, 'r') as f:
        all_paths = json.load(f)
    log.info(f"Found {len(all_paths):,} total paths.")
    if args.limit:
        all_paths = all_paths[:args.limit]
        log.warning(f"Processing limited to the first {len(all_paths):,} paths.")

    fs = fsspec.filesystem("gcs")
    log.info(f"Starting processing with {args.workers} worker threads...")
    results_list, error_count = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        func = partial(process_path, fs=fs)
        futures = {executor.submit(func, path) for path in all_paths}
        progress_bar = tqdm(as_completed(futures), total=len(all_paths), desc="Processing frames")
        for future in progress_bar:
            result = future.result()
            if result:
                results_list.append(result)
            else:
                error_count += 1
            progress_bar.set_postfix({"success": len(results_list), "errors": error_count})

    log.info("------ Processing Complete ------")
    if not results_list:
        log.warning("No valid data was processed. Output file will not be created.")
    else:
        log.info(f"Aggregating {len(results_list):,} results into a DataFrame...")
        df = pd.DataFrame(results_list)
        df.to_parquet(output_path, index=False, engine='pyarrow', compression='snappy')
        log.info(f"Successfully saved the manifest to '{output_path}'.")
    log.info(f"Encountered {error_count:,} errors (see logs for details).")


if __name__ == "__main__":
    main()
