"""Step 1 — Crop the side-by-side WhatsApp screenshot into Roy_D / Guest halves,
run YOLO face detection on each half, save face crops using the same preprocessing
as the training pipeline (YOLO_BBOX_MARGIN=20 around detected face box).

Input:  /Users/roeedar/Downloads/WhatsApp Image 2026-05-19 at 16.14.22.jpeg
Output: analysis/teams_account_natural_experiment_2026-05-19/crops/
        - panel_roy_d.png      (left half, full frame)
        - panel_guest.png      (right half, full frame)
        - face_roy_d.png       (YOLO face crop, BBOX+margin, raw resolution)
        - face_guest.png       (YOLO face crop, BBOX+margin, raw resolution)
        - face_roy_d_224.png   (resized 224×224, INTER_LINEAR — model input)
        - face_guest_224.png   (resized 224×224, INTER_LINEAR — model input)
"""
from __future__ import annotations

import sys, os
from pathlib import Path
import cv2
import numpy as np

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
sys.path.insert(0, str(REPO))
import video_preprocessor as vp  # noqa

SRC = Path("/Users/roeedar/Downloads/WhatsApp Image 2026-05-19 at 16.14.22.jpeg")
OUT = Path("analysis/teams_account_natural_experiment_2026-05-19/crops")
OUT.mkdir(parents=True, exist_ok=True)

# Load image
img = cv2.imread(str(SRC), cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f"Could not read {SRC}")
H, W = img.shape[:2]
print(f"Source image: {W}x{H}")

# Split into left (Roy_D) / right (Guest) halves
mid = W // 2
left = img[:, :mid].copy()
right = img[:, mid:].copy()

cv2.imwrite(str(OUT / "panel_roy_d.png"), left)
cv2.imwrite(str(OUT / "panel_guest.png"), right)
print(f"Saved panels: {left.shape} / {right.shape}")

# YOLO face detection on each half
vp.initialize_yolo_model()

def detect_and_crop(panel: np.ndarray, label: str) -> dict:
    """Return dict with bbox, raw face crop, 224x224 model-ready crop."""
    bbox = vp._get_yolo_face_box(panel, conf_threshold=vp.YOLO_CONF_THRESHOLD)
    if bbox is None:
        raise RuntimeError(f"YOLO failed to detect face in {label} panel")
    x0, y0, x1, y1 = map(int, bbox)
    h, w = panel.shape[:2]
    # Make a square crop around the YOLO box (production behavior — square crops)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    side = max(x1 - x0, y1 - y0)
    half = side // 2
    sx0 = max(0, cx - half); sy0 = max(0, cy - half)
    sx1 = min(w, cx + half); sy1 = min(h, cy + half)
    face_raw = panel[sy0:sy1, sx0:sx1].copy()
    # Resize to 224×224 with INTER_LINEAR (matches training)
    face_224 = cv2.resize(face_raw, (224, 224), interpolation=cv2.INTER_LINEAR)
    return {
        "bbox_xyxy": (int(x0), int(y0), int(x1), int(y1)),
        "square_bbox_xyxy": (int(sx0), int(sy0), int(sx1), int(sy1)),
        "face_raw": face_raw,
        "face_224": face_224,
        "face_raw_shape": face_raw.shape,
    }

r_roy = detect_and_crop(left, "Roy_D")
r_guest = detect_and_crop(right, "Guest")

print(f"Roy_D bbox: {r_roy['bbox_xyxy']}  square_crop_shape: {r_roy['face_raw_shape']}")
print(f"Guest bbox: {r_guest['bbox_xyxy']}  square_crop_shape: {r_guest['face_raw_shape']}")

# Save crops
cv2.imwrite(str(OUT / "face_roy_d.png"), r_roy["face_raw"])
cv2.imwrite(str(OUT / "face_guest.png"), r_guest["face_raw"])
cv2.imwrite(str(OUT / "face_roy_d_224.png"), r_roy["face_224"])
cv2.imwrite(str(OUT / "face_guest_224.png"), r_guest["face_224"])
print(f"\nWrote face crops to {OUT}")

# Quick visual diff at full resolution
import json
meta = {
    "src": str(SRC),
    "src_shape": list(img.shape),
    "roy_d": {
        "panel_shape": list(left.shape),
        "yolo_bbox_xyxy": r_roy["bbox_xyxy"],
        "square_crop_bbox_xyxy": r_roy["square_bbox_xyxy"],
        "face_raw_shape": list(r_roy["face_raw"].shape),
    },
    "guest": {
        "panel_shape": list(right.shape),
        "yolo_bbox_xyxy": r_guest["bbox_xyxy"],
        "square_crop_bbox_xyxy": r_guest["square_bbox_xyxy"],
        "face_raw_shape": list(r_guest["face_raw"].shape),
    },
}
with open(OUT / "crop_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(json.dumps(meta, indent=2))
