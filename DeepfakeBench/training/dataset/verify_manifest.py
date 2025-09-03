#!/usr/bin/env python3
"""
verify_manifest.py

Reads a Parquet manifest file and prints a comprehensive summary to verify
its structure, schema, and content.

Usage:
python verify_manifest.py path/to/your/manifest.parquet
"""

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is not installed. Please run `pip install pandas pyarrow`")
    sys.exit(1)


def verify_manifest(manifest_path: Path):
    """Loads and verifies the Parquet manifest file."""
    if not manifest_path.exists():
        print(f"Error: File not found at '{manifest_path}'")
        sys.exit(1)

    print(f"--- Verifying Manifest: {manifest_path} ---")

    try:
        df = pd.read_parquet(manifest_path)
    except Exception as e:
        print(f"\nError: Failed to read the Parquet file. It may be corrupt.")
        print(f"Details: {e}")
        sys.exit(1)

    # 1. Basic Information (Shape)
    print("\n--- 1. Basic Information ---")
    print(f"Total records (frames): {df.shape[0]:,}")
    print(f"Total columns (properties): {df.shape[1]}")

    # 2. Schema, Data Types, and Non-Null Counts
    print("\n--- 2. Schema and Data Types (Dtypes) ---")
    print("This shows columns, non-null counts, and memory usage.")
    # Use a buffer to capture the output of df.info() and print it
    buffer = pd.io.common.StringIO()
    df.info(buf=buffer)
    print(buffer.getvalue())

    # 3. Numerical Properties Summary
    print("\n--- 3. Numerical Properties Summary ---")
    # describe() provides stats like mean, std, min, max for numeric columns
    numeric_cols = ['file_size_kb', 'sharpness']
    if all(col in df.columns for col in numeric_cols):
        print(df[numeric_cols].describe().round(2))
    else:
        print("Warning: 'file_size_kb' or 'sharpness' column not found.")

    # 4. Categorical Value Counts
    print("\n--- 4. Label and Method Counts ---")
    if 'label' in df.columns:
        print("\n[Label Distribution]")
        print(df['label'].value_counts())

    if 'method' in df.columns:
        print("\n[Method Distribution (Top 10)]")
        print(df['method'].value_counts().nlargest(10))

    # 5. Spot Check (First 5 Rows)
    print("\n--- 5. Spot Check (First 5 Rows) ---")
    with pd.option_context('display.max_colwidth', 100):  # Widen column for paths
        print(df.head())

    print("\n--- Verification Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify a Parquet manifest file.")
    parser.add_argument(
        "manifest_path",
        type=Path,
        help="Path to the .parquet manifest file to verify."
    )
    args = parser.parse_args()
    verify_manifest(args.manifest_path)
