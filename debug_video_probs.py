#!/usr/bin/env python3
"""
Debug script: sample frames from videos, send to the Effort-AIGI API,
and print per-frame fake probabilities.

Usage:
    python debug_video_probs.py video1.mp4 video2.mp4 video3.mp4
    python debug_video_probs.py /path/to/*.mp4
    python debug_video_probs.py --video-list videos.txt   # one path per line
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import requests
from ultralytics import YOLO

API_BASE = "http://34.16.217.28:8999"
ENDPOINT = f"{API_BASE}/check_frame_batch"
NUM_FRAMES = 64
YOLO_MODEL_PATH = "/Users/roeedar/Downloads/yolov8s-face.pt"
FACE_MARGIN = 10  # pixels to grow the bounding box on each side
MODEL_INPUT_SIZE = (224, 224)

_yolo_model = None


def get_yolo_model() -> YOLO:
    """Lazy-load the YOLO face model."""
    global _yolo_model
    if _yolo_model is None:
        print(f"Loading YOLO face model from {YOLO_MODEL_PATH} ...")
        # Allow unsafe load – the checkpoint contains custom ultralytics classes
        import torch
        _orig = torch.load
        torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
        try:
            _yolo_model = YOLO(YOLO_MODEL_PATH)
        finally:
            torch.load = _orig
    return _yolo_model


def crop_face(frame: np.ndarray, margin: int = FACE_MARGIN) -> np.ndarray | None:
    """Detect the largest face in `frame`, grow box by `margin` px, crop & resize to 224x224."""
    model = get_yolo_model()
    results = model(frame, verbose=False)

    if not results or len(results[0].boxes) == 0:
        return None

    # Pick the box with the highest confidence
    boxes = results[0].boxes
    best_idx = int(boxes.conf.argmax())
    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

    # Grow by margin, clamp to image bounds
    h, w = frame.shape[:2]
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return cv2.resize(crop, MODEL_INPUT_SIZE, interpolation=cv2.INTER_AREA)


def sample_frames(video_path: str, n: int = NUM_FRAMES) -> list[np.ndarray]:
    """Sample `n` evenly-spaced frames (as BGR numpy arrays) from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Video has 0 frames: {video_path}")

    # Evenly spaced indices (avoid the very first and last frame)
    indices = np.linspace(0, total - 1, n + 2, dtype=int)[1:-1]  # skip first/last
    if len(indices) == 0:
        indices = np.linspace(0, total - 1, n, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append((int(idx), frame))
        else:
            print(f"  ⚠ Could not read frame {idx}")
    cap.release()
    return frames


def send_batch(frames: list[tuple[int, np.ndarray]],
               model_type: str = "custom",
               threshold: float = 0.5,
               recrop: bool = False) -> dict:
    """Crop faces locally, encode as JPEG, and POST to /check_frame_batch."""
    files = []
    for i, (idx, frame) in enumerate(frames):
        # Local face crop with YOLO + margin
        cropped = crop_face(frame)
        if cropped is None:
            print(f"  ⚠ No face detected in frame {idx}, skipping")
            continue

        ok, buf = cv2.imencode(".jpg", cropped)
        if not ok:
            print(f"  ⚠ Failed to encode frame {idx}")
            continue
        files.append(
            ("files", (f"frame_{idx:05d}.jpg", buf.tobytes(), "image/jpeg"))
        )

    if not files:
        return {"error": "No frames could be encoded"}

    params = {
        "model_type": model_type,
        "threshold": threshold,
        "recrop": recrop,
    }
    resp = requests.post(ENDPOINT, files=files, params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


def print_results(video_path: str, frame_indices: list[int], result: dict):
    """Pretty-print per-frame probabilities."""
    probs = result.get("probs", [])
    confidence = result.get("confidence", -1)
    label = result.get("pred_label", "?")

    print(f"\n{'=' * 70}")
    print(f"📹  {video_path}")
    print(f"{'─' * 70}")
    print(f"  Overall  →  {label}  (mean confidence: {confidence:.4f})")
    print(f"  {'─' * 60}")

    if probs:
        for i, prob in enumerate(probs):
            idx = frame_indices[i] if i < len(frame_indices) else "?"
            bar_len = int(prob * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            tag = "FAKE" if prob >= 0.5 else "REAL"
            print(f"  Frame {str(idx):>6s}  │ {bar} │ {prob:.4f}  ({tag})")
    else:
        print("  No probabilities returned.")

    if probs:
        arr = np.array(probs)
        print(f"  {'─' * 60}")
        print(f"  Stats: mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
              f"std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Debug video frame probabilities via Effort-AIGI API")
    parser.add_argument("videos", nargs="*", help="Video file paths")
    parser.add_argument("--video-list", "-l", help="Text file with one video path per line")
    parser.add_argument("--model-type", "-m", default="custom", choices=["base", "custom"])
    parser.add_argument("--threshold", "-t", type=float, default=0.5)
    parser.add_argument("--num-frames", "-n", type=int, default=NUM_FRAMES)
    parser.add_argument("--server-recrop", action="store_true",
                        help="Let the server do YOLO cropping instead of local cropping")
    args = parser.parse_args()

    video_paths = list(args.videos)
    if args.video_list:
        with open(args.video_list) as f:
            video_paths.extend(line.strip() for line in f if line.strip())

    if not video_paths:
        parser.print_help()
        sys.exit(1)

    # Quick health check
    try:
        r = requests.get(f"{API_BASE}/ping", timeout=5)
        r.raise_for_status()
        print(f"✅ API is reachable at {API_BASE}")
    except Exception as e:
        print(f"❌ Cannot reach API at {API_BASE}: {e}")
        sys.exit(1)

    print(f"\nSettings: model={args.model_type}, threshold={args.threshold}, "
          f"frames={args.num_frames}, crop=local (YOLO + {FACE_MARGIN}px margin)")

    # Pre-load YOLO model once
    if not args.server_recrop:
        get_yolo_model()

    for vp in video_paths:
        if not Path(vp).exists():
            print(f"\n⚠ File not found: {vp} — skipping")
            continue

        try:
            print(f"\n⏳ Sampling {args.num_frames} frames from: {vp}")
            frames = sample_frames(vp, n=args.num_frames)
            if not frames:
                print(f"  ⚠ No frames could be read from {vp}")
                continue

            frame_indices = [idx for idx, _ in frames]
            print(f"  Sampled frame indices: {frame_indices}")
            print(f"  Sending {len(frames)} frames to API...")

            result = send_batch(
                frames,
                model_type=args.model_type,
                threshold=args.threshold,
                recrop=args.server_recrop,
            )
            print_results(vp, frame_indices, result)

        except requests.HTTPError as e:
            print(f"  ❌ API error: {e.response.status_code} — {e.response.text}")
        except Exception as e:
            print(f"  ❌ Error processing {vp}: {e}")


if __name__ == "__main__":
    main()
