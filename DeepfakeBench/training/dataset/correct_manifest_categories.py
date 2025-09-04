#!/usr/bin/env python3
"""
correct_manifest_categories.py

Loads an existing frame properties manifest and corrects the 'method_category'
for a predefined list of methods. This is intended as a one-off script to fix
categorization errors in previously generated manifests.
"""
import argparse
import logging
import sys
import pandas as pd
from pathlib import Path

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

# --- CORRECTION MAPPING ---
# This dictionary defines the required corrections.
# Key: The 'method' name to find.
# Value: The 'method_category' it should be assigned.
CORRECTIONS = {
    # Re-categorize these real-video sources into the main DF40 pool
    'youtube_real': 'BASE_DF40_REG',
    'faceforensics++': 'BASE_DF40_REG',
    'celeb_real': 'BASE_DF40_REG',

    # Re-categorize this fake method into the main DF40 pool
    'hyperreenact': 'BASE_DF40_REG',

    # Re-categorize this dataset into the FF++ pool
    'deepfakedetection': 'BASE_FF++',
}


def main():
    parser = argparse.ArgumentParser(
        description="Correct method categories in an existing Parquet manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("manifest_path", type=str,
                        help="Path to the frame_properties.parquet file to correct.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        log.error(f"Error: Manifest file not found at '{manifest_path}'")
        sys.exit(1)

    log.info(f"Loading manifest from '{manifest_path}'...")
    try:
        df = pd.read_parquet(manifest_path)
    except Exception as e:
        log.error(f"Failed to read Parquet file: {e}")
        sys.exit(1)

    log.info(f"Loaded {len(df):,} records. Applying corrections...")

    for method, new_category in CORRECTIONS.items():
        # Create a boolean mask for rows that match the method and have the wrong category
        mask = (df['method'] == method) & (df['method_category'] != new_category)

        num_rows_to_update = mask.sum()

        if num_rows_to_update > 0:
            log.info(
                f"Found {num_rows_to_update:>7,} rows for method '{method}' to be re-categorized to '{new_category}'.")
            df.loc[mask, 'method_category'] = new_category
        else:
            log.info(f"No rows needed updating for method '{method}'.")

    log.info("Corrections applied. Saving updated manifest...")
    try:
        # Overwrite the original file with the corrected DataFrame
        df.to_parquet(manifest_path, index=False, engine='pyarrow', compression='snappy')
        log.info(f"Successfully saved corrected manifest to '{manifest_path}'.")
    except Exception as e:
        log.error(f"Failed to write updated Parquet file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
