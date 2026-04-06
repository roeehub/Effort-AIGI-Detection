#!/usr/bin/env python3
"""
upload_wma_validation.py — Upload WMA face-crop images to GCS for OOD validation.

Reads 1,202 pre-cropped face images from a local directory, groups them into
"video" folders by participant name, and uploads to GCS in the layout expected
by the validation data loaders:

    gs://<bucket>/wma_validation/enhanced_fake/<participant_id>/<frame>.jpg

The participant id is extracted from the filename prefix (before the first
double-underscore). If fewer than --min_frames images belong to a participant,
they are merged into a catch-all "misc" folder.

All images are uploaded as-is (no resizing). The validation pipeline resizes to
224×224 on the fly.

This is a ONE-TIME upload script.  After running it you can validate with:

    python validate_custom_sources.py \\
        --checkpoint_gcs_path <GCS_CHECKPOINT> \\
        --wma_bucket effort-collected-data \\
        ...

Usage:
    python upload_wma_validation.py \\
        --src /Users/roeedar/Downloads/wma_export/all_images \\
        --bucket effort-collected-data \\
        --prefix wma_validation/enhanced_fake \\
        --dry_run          # remove to actually upload
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_participant(filename: str) -> str:
    """Extract participant id from e.g. 'dor_shkedi__20260214_152126__frame_017935_crop_001.jpg'.

    Convention: everything before the first double-underscore is the participant.
    """
    stem = Path(filename).stem
    parts = stem.split("__", 1)
    return parts[0] if len(parts) > 1 else "unknown"


def group_images(src_dir: Path, min_frames: int) -> dict[str, list[Path]]:
    """Group image files by participant, merging small groups into 'misc'."""
    by_participant: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(src_dir.iterdir()):
        if f.suffix.lower() in IMG_EXTS and f.is_file():
            pid = extract_participant(f.name)
            by_participant[pid].append(f)

    # Merge participants with < min_frames into "misc"
    merged: dict[str, list[Path]] = {}
    misc: list[Path] = []
    for pid, files in by_participant.items():
        if len(files) >= min_frames:
            merged[pid] = files
        else:
            misc.extend(files)
    if misc:
        merged["misc"] = misc

    return merged


def main():
    parser = argparse.ArgumentParser(description="Upload WMA validation images to GCS.")
    parser.add_argument("--src", type=str,
                        default="/Users/roeedar/Downloads/wma_export/all_images",
                        help="Local directory with WMA face-crop images.")
    parser.add_argument("--bucket", type=str, default="effort-collected-data",
                        help="GCS bucket name.")
    parser.add_argument("--prefix", type=str, default="wma_validation/enhanced_fake",
                        help="GCS prefix (path under bucket).")
    parser.add_argument("--min_frames", type=int, default=8,
                        help="Minimum frames to keep a participant as a separate video (default 8).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be uploaded without uploading.")
    args = parser.parse_args()

    src_dir = Path(args.src)
    if not src_dir.is_dir():
        print(f"ERROR: source directory not found: {src_dir}")
        sys.exit(1)

    groups = group_images(src_dir, args.min_frames)

    total_files = sum(len(files) for files in groups.values())
    print(f"Source:       {src_dir}")
    print(f"Destination:  gs://{args.bucket}/{args.prefix}/<participant>/<frame>.jpg")
    print(f"Total images: {total_files}")
    print(f"Video groups: {len(groups)}")
    print()

    for pid, files in sorted(groups.items()):
        print(f"  {pid:<30} {len(files):>5} frames")
    print()

    if args.dry_run:
        print("DRY RUN — no files uploaded.")
        print("\nSample GCS paths:")
        for pid, files in sorted(groups.items()):
            for f in files[:2]:
                print(f"  gs://{args.bucket}/{args.prefix}/{pid}/{f.name}")
            if len(files) > 2:
                print(f"  ... ({len(files) - 2} more)")
        return

    # Upload
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(args.bucket)

    uploaded = 0
    for pid, files in sorted(groups.items()):
        for f in files:
            blob_path = f"{args.prefix}/{pid}/{f.name}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(f), content_type="image/jpeg")
            uploaded += 1
            if uploaded % 100 == 0:
                print(f"  Uploaded {uploaded}/{total_files} ...")

    print(f"\nDone — uploaded {uploaded} images to gs://{args.bucket}/{args.prefix}/")
    print(f"\nTo validate, add to validate_custom_sources.py:")
    print(f"  --wma_bucket {args.bucket}")
    print(f"  --wma_prefix {args.prefix}")


if __name__ == "__main__":
    main()
