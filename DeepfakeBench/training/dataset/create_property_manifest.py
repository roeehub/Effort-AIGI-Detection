#!/usr/bin/env python3
"""
create_property_manifest.py

Scans a list of cloud storage frame paths, computes key properties for each frame,
and saves the results to a new, enriched manifest in Apache Parquet format.

This is intended as a one-time, offline preprocessing step to enable
property-based balanced sampling during model training.

Properties computed:
- File Size (from metadata, no download needed)
- Sharpness (variance of the Laplacian, requires image download and processing)

Example Usage:
python create_property_manifest.py \
    --input-manifest frame_manifest.json \
    --output-manifest frame_properties.parquet \
    --workers 64

"""
import argparse
import json
import logging
import sys
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

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
    print("Please install the necessary dependencies by running:")
    print("pip install pandas pyarrow opencv-python-headless gcsfs Pillow tqdm")
    sys.exit(1)

# --- Setup basic logging ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def calculate_sharpness(image_bytes: bytes) -> float:
    """Calculates the sharpness of an image using the variance of the Laplacian.

    Args:
        image_bytes: The raw bytes of the image file.

    Returns:
        A float representing the sharpness score.
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('L')
        cv_image = np.array(pil_image)
        sharpness = cv2.Laplacian(cv_image, cv2.CV_64F).var()
        return float(sharpness)
    except Exception:
        return -1.0


def process_path(gcs_path: str, fs: fsspec.AbstractFileSystem) -> Optional[Dict]:
    """
    Processes a single GCS path to extract metadata and compute properties.

    Args:
        gcs_path: The full gs:// path to the frame.
        fs: An initialized fsspec filesystem object for GCS.

    Returns:
        A dictionary containing the properties, or None if an error occurs.
    """
    try:
        parts = Path(gcs_path).parts
        if len(parts) < 4:
            log.warning(f"Skipping malformed path (too short): {gcs_path}")
            return None
        label, method, video_id = parts[-4], parts[-3], parts[-2]

        info = fs.info(gcs_path)
        file_size_kb = info.get("size", 0) / 1024.0

        with fs.open(gcs_path, 'rb') as f:
            image_bytes = f.read()

        sharpness = calculate_sharpness(image_bytes)
        if sharpness < 0:
            log.warning(f"Could not calculate sharpness for image: {gcs_path}")
            return None

        return {
            "path": gcs_path,
            "label": label,
            "method": method,
            "video_id": video_id,
            "file_size_kb": round(file_size_kb, 2),
            "sharpness": round(sharpness, 2),
        }

    except FileNotFoundError:
        log.warning(f"File not found on GCS, skipping: {gcs_path}")
        return None
    except Exception as e:
        log.error(f"Unexpected error processing {gcs_path}: {e}")
        return None


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Create an enhanced property manifest in Parquet format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-manifest",
        type=str,
        required=True,
        help="Path to the input JSON manifest containing a list of GCS file paths.",
    )
    parser.add_argument(
        "--output-manifest",
        type=str,
        required=True,
        help="Path to the output Parquet (.parquet) manifest to be created.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of parallel worker threads to use for fetching and processing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: Process only the first N records for testing.",
    )
    args = parser.parse_args()

    # --- 1. Pre-flight Checks ---
    input_path = Path(args.input_manifest)
    output_path = Path(args.output_manifest)

    if not input_path.exists():
        log.error(f"Input manifest not found at: {input_path}")
        sys.exit(1)

    if output_path.exists():
        overwrite = input(f"Output file '{output_path}' already exists. Overwrite? (y/N): ").lower()
        if overwrite != 'y':
            log.info("Operation cancelled by user.")
            sys.exit(0)

    # --- 2. Load and Prepare Data ---
    log.info(f"Loading frame paths from '{input_path}'...")
    with open(input_path, 'r') as f:
        all_paths = json.load(f)
    log.info(f"Found {len(all_paths):,} total paths.")

    if args.limit:
        all_paths = all_paths[:args.limit]
        log.warning(f"Processing limited to the first {len(all_paths):,} paths.")

    # --- 3. Initialize GCS Filesystem ---
    fs = fsspec.filesystem("gcs")

    # --- 4. Parallel Processing and Data Collection ---
    log.info(f"Starting processing with {args.workers} worker threads...")

    results_list = []
    error_count = 0

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

            progress_bar.set_postfix(
                {"success": len(results_list), "errors": error_count}
            )

    # --- 5. Final Conversion and Writing to Parquet ---
    log.info("------ Processing Complete ------")
    if not results_list:
        log.warning("No valid data was processed. Output file will not be created.")
    else:
        log.info(f"Aggregating {len(results_list):,} results into a DataFrame...")
        df = pd.DataFrame(results_list)

        log.info(f"Writing DataFrame to Parquet file: '{output_path}'")
        try:
            df.to_parquet(output_path, index=False, engine='pyarrow', compression='snappy')
            log.info("Successfully saved the manifest.")
        except Exception as e:
            log.error(f"Failed to write Parquet file: {e}")
            sys.exit(1)

    # --- 6. Final Report ---
    log.info(f"Successfully processed and wrote {len(results_list):,} records.")
    log.info(f"Encountered {error_count:,} errors (see logs for details).")


if __name__ == "__main__":
    main()
