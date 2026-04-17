#!/usr/bin/env python3
"""
Send all image frames from a directory to the Effort-AIGI API (no cropping)
and print per-frame probabilities.

Usage:
    python debug_frames_dir.py /Users/roeedar/Downloads/dor_musk_test
    python debug_frames_dir.py /path/to/frames --model-type base --threshold 0.5
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import requests

API_BASE = "http://34.16.217.28:8999"
ENDPOINT = f"{API_BASE}/check_frame_batch"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def send_batch(image_paths: list[Path],
               model_type: str = "custom",
               threshold: float = 0.5) -> dict:
    """Read images, encode as JPEG, POST to /check_frame_batch with recrop=false."""
    files = []
    names = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  ⚠ Cannot read {p.name}, skipping")
            continue
        ok, buf = cv2.imencode(".jpg", img)
        if not ok:
            print(f"  ⚠ Failed to encode {p.name}")
            continue
        files.append(("files", (p.name, buf.tobytes(), "image/jpeg")))
        names.append(p.name)

    if not files:
        return {"error": "No frames could be encoded", "names": []}

    params = {"model_type": model_type, "threshold": threshold, "recrop": False}
    resp = requests.post(ENDPOINT, files=files, params=params, timeout=300)
    resp.raise_for_status()
    result = resp.json()
    result["_names"] = names
    return result


def print_results(dir_path: str, result: dict):
    probs = result.get("probs", [])
    names = result.get("_names", [])
    confidence = result.get("confidence", -1)
    label = result.get("pred_label", "?")

    print(f"\n{'=' * 74}")
    print(f"📁  {dir_path}")
    print(f"{'─' * 74}")
    print(f"  Overall  →  {label}  (mean confidence: {confidence:.4f})")
    print(f"  {'─' * 64}")

    if probs:
        for i, prob in enumerate(probs):
            name = names[i] if i < len(names) else f"frame_{i}"
            bar_len = int(prob * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            tag = "FAKE" if prob >= 0.5 else "REAL"
            print(f"  {name:>30s}  │ {bar} │ {prob:.4f}  ({tag})")
    else:
        print("  No probabilities returned.")

    if probs:
        arr = np.array(probs)
        print(f"  {'─' * 64}")
        print(f"  Stats: mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
              f"std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")
        fake_count = np.sum(arr >= 0.5)
        print(f"  Frames: {len(arr)} total, {fake_count} FAKE, {len(arr) - fake_count} REAL")
    print(f"{'=' * 74}")


def main():
    parser = argparse.ArgumentParser(description="Send directory of frames to Effort-AIGI API")
    parser.add_argument("path", help="Directory containing image frames")
    parser.add_argument("--model-type", "-m", default="custom", choices=["base", "custom"])
    parser.add_argument("--threshold", "-t", type=float, default=0.5)
    args = parser.parse_args()

    dir_path = Path(args.path)
    if not dir_path.is_dir():
        print(f"❌ Not a directory: {dir_path}")
        sys.exit(1)

    images = sorted([p for p in dir_path.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not images:
        print(f"❌ No image files found in {dir_path}")
        sys.exit(1)

    # Health check
    try:
        r = requests.get(f"{API_BASE}/ping", timeout=5)
        r.raise_for_status()
        print(f"✅ API is reachable at {API_BASE}")
    except Exception as e:
        print(f"❌ Cannot reach API at {API_BASE}: {e}")
        sys.exit(1)

    print(f"\nDirectory: {dir_path}")
    print(f"Found {len(images)} images, sending with recrop=false")
    print(f"Settings: model={args.model_type}, threshold={args.threshold}")

    try:
        result = send_batch(images, model_type=args.model_type, threshold=args.threshold)
        if "error" in result:
            print(f"❌ {result['error']}")
            sys.exit(1)
        print_results(str(dir_path), result)
    except requests.HTTPError as e:
        print(f"❌ API error: {e.response.status_code} — {e.response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
