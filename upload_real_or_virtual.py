#!/usr/bin/env python3
"""
upload_real_or_virtual.py
=========================
Extract 24 uniform face-cropped frames per video from /Users/roeedar/Downloads/Videos
and upload to gs://effort-collected-data/real_or_virtual/real/<video_name>/<frame>.jpg

Uses the same YOLO face cropping settings as video_preprocessor.py:
- YOLOv8s-face model
- conf=0.20, margin=20px
- Square crop → 224x224 INTER_AREA
"""
import os
import sys
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np

# ── Add training dir to path so we can import video_preprocessor ──
TRAINING_DIR = os.path.join(os.path.dirname(__file__), "DeepfakeBench", "training")
sys.path.insert(0, TRAINING_DIR)

from video_preprocessor import extract_yolo_face, MODEL_IMG_SIZE

# ── Config ──
SOURCE_DIR = "/Users/roeedar/Downloads/Videos"
GCS_BUCKET = "gs://effort-collected-data/real_or_virtual/real"
NUM_FRAMES = 24
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def discover_videos(root: str) -> list[tuple[str, str]]:
    """
    Walk the directory tree and return (video_path, unique_video_name) pairs.
    Unique name uses relative path components to avoid collisions
    (e.g. Test/9/V_Mac.mp4 → Test_9_V_Mac).
    """
    videos = []
    root_path = Path(root)
    for path in sorted(root_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            # Build unique name from relative path
            rel = path.relative_to(root_path)
            parts = list(rel.parent.parts) + [path.stem]
            unique_name = "_".join(parts)
            videos.append((str(path), unique_name))
    return videos


def sample_and_crop(video_path: str, num_frames: int) -> list[np.ndarray]:
    """
    Open video, sample `num_frames` uniformly, YOLO face-crop each.
    Returns list of 224x224 BGR face crops.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        print(f"  [ERROR] No frames in: {video_path}")
        cap.release()
        return []

    # Generate uniform indices
    indices = set(np.linspace(0, total - 1, num_frames, dtype=int))

    frames = {}
    frame_num = 0
    while frame_num <= max(indices):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num in indices:
            frames[frame_num] = frame
        frame_num += 1
    cap.release()

    if not frames:
        print(f"  [WARN] Could not read any frames from {video_path}")
        return []

    # YOLO face crop each frame
    crops = []
    for idx in sorted(frames.keys()):
        face = extract_yolo_face(frames[idx])
        if face is not None:
            crops.append((idx, face))
        # else: skip frames where no face is found

    return crops


def upload_to_gcs(crops: list[tuple[int, np.ndarray]], video_name: str, dry_run: bool = False) -> int:
    """Save crops to temp dir, then gsutil -m cp to GCS. Returns count uploaded."""
    if not crops:
        return 0

    with tempfile.TemporaryDirectory() as tmpdir:
        local_paths = []
        for idx, face_bgr in crops:
            fname = f"frame_{idx:06d}.jpg"
            local_path = os.path.join(tmpdir, fname)
            cv2.imwrite(local_path, face_bgr)
            local_paths.append(local_path)

        gcs_dest = f"{GCS_BUCKET}/{video_name}/"
        if dry_run:
            print(f"  [DRY RUN] Would upload {len(local_paths)} frames → {gcs_dest}")
            return len(local_paths)
        else:
            # Use gsutil -m for parallel upload
            cmd = f'gsutil -m cp {tmpdir}/*.jpg {gcs_dest}'
            ret = os.system(cmd)
            if ret != 0:
                print(f"  [ERROR] gsutil upload failed for {video_name}")
                return 0
            return len(local_paths)


def main():
    parser = argparse.ArgumentParser(description="Extract face crops from videos and upload to GCS")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be uploaded without uploading")
    parser.add_argument("--num_frames", type=int, default=NUM_FRAMES, help="Frames to sample per video")
    args = parser.parse_args()

    videos = discover_videos(SOURCE_DIR)
    print(f"Found {len(videos)} videos in {SOURCE_DIR}\n")

    total_uploaded = 0
    total_failed_videos = 0

    for i, (vpath, vname) in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {vname}")
        print(f"  Source: {vpath}")

        crops = sample_and_crop(vpath, args.num_frames)
        if not crops:
            print(f"  [SKIP] No faces extracted")
            total_failed_videos += 1
            continue

        print(f"  Extracted {len(crops)}/{args.num_frames} face crops")
        count = upload_to_gcs(crops, vname, dry_run=args.dry_run)
        total_uploaded += count
        print(f"  {'Would upload' if args.dry_run else 'Uploaded'} {count} frames → {GCS_BUCKET}/{vname}/")

    print(f"\n{'=' * 50}")
    print(f"Done — {'would upload' if args.dry_run else 'uploaded'} {total_uploaded} face crops from {len(videos) - total_failed_videos}/{len(videos)} videos")
    print(f"Destination: {GCS_BUCKET}/")


if __name__ == "__main__":
    main()
