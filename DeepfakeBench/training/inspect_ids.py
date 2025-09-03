#!/usr/bin/env python3
"""
inspect_ids.py

A debugging tool to inspect the relationship between the unified 'source_video_id'
and the raw 'video_id' (folder names) in the property manifest.

This script demonstrates how multiple fake video folders map to a single
source video, explaining the discrepancy in video counts between different
analysis scripts.

Example Usage:
# Show all video folders derived from the source video '120' from FaceForensics++
python inspect_ids.py 120

# Show all video folders derived from the source video 'id00017' from Celeb-real
python inspect_ids.py id00017
"""
import sys
import argparse
import pandas as pd

MANIFEST_PATH = "frame_properties.parquet"


def main():
    parser = argparse.ArgumentParser(
        description="Inspect the mapping from a source_video_id to its raw video_id folders."
    )
    parser.add_argument(
        "source_id",
        type=str,
        help="The unified source_video_id to inspect (e.g., '120', 'id00017').",
    )
    args = parser.parse_args()

    print(f"Loading manifest '{MANIFEST_PATH}'...")
    try:
        df = pd.read_parquet(MANIFEST_PATH)
    except FileNotFoundError:
        print(f"ERROR: Manifest not found at '{MANIFEST_PATH}'.")
        print("Please run `create_property_manifest.py` first.")
        sys.exit(1)

    # Convert source_id column to string to ensure the query works correctly
    # as IDs can be numeric or alphanumeric.
    df['source_video_id'] = df['source_video_id'].astype(str)

    print(f"Searching for all entries with source_video_id == '{args.source_id}'...")

    # Filter the DataFrame to find all frames related to this one source video
    result_df = df[df['source_video_id'] == args.source_id]

    if result_df.empty:
        print("\n" + "=" * 70)
        print(f"❌ No records found for source_video_id '{args.source_id}'.")
        print("Please check the ID and try again.")
        print("=" * 70)
        sys.exit(0)

    # --- Generate the Report ---
    print("\n" + "=" * 70)
    print(f"✅ Found {len(result_df)} frames from one source video: '{args.source_id}'")
    print("=" * 70)
    print("These frames are grouped under the following (method, raw_folder_name) combinations:")

    # Group by method and the raw video_id (folder name) and count frames
    grouped = result_df.groupby(['method', 'video_id']).size().reset_index(name='frame_count')

    for _, row in grouped.iterrows():
        print(f"  - Method: {row['method']:<15} | Raw Folder: {row['video_id']:<30} | Frames: {row['frame_count']}")

    print("\nThis shows that one 'source_video_id' correctly groups together the original")
    print("real video (if present) and all of its fake derivatives from different methods.")


if __name__ == "__main__":
    main()