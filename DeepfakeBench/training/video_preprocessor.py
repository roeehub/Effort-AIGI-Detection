"""
video_preprocessor.py
================================================
A modular library to preprocess images and videos for the EffortDetector model.

This module implements three face detection/cropping strategies:
- 'dlib': Dlib-based landmark alignment. Highest accuracy, heaviest dependency.
- 'yolo': YOLOv8-based simple square crop. Robust and fast detection.
- 'yolo_haar': A hybrid method using YOLO for face detection and a lightweight
  OpenCV Haar Cascade for eye-based alignment. Good balance of speed and
  accuracy.
"""
from __future__ import annotations
import sys
import os
from typing import List, Optional, Tuple
from pathlib import Path
from urllib.request import urlretrieve

import cv2  # noqa
import numpy as np  # noqa
import torch  # noqa
import torchvision.transforms as T  # noqa
# import dlib  # noqa
from skimage import transform as trans  # noqa
from ultralytics import YOLO  # noqa

# ──────────────────────────
# 📍 Global Configuration
# ──────────────────────────
MODEL_IMG_SIZE = 224

# --- Dlib Settings ---
DLIB_LANDMARK_MODEL_PATH = "shape_predictor_81_face_landmarks.dat"

# --- YOLO Settings ---
YOLO_MODEL_PATH = "yolov8s-face.pt"
# YOLO_MODEL_PATH = "/Users/roeedar/Library/Application Support/JetBrains/PyCharmCE2024.2/scratches/yolov8s-face.pt"
YOLO_BBOX_MARGIN = 20
YOLO_CONF_THRESHOLD = 0.20

# --- Haar Cascade Settings ---
# Path is resolved automatically from cv2 library

# --- Frame Sampling Settings for Video ---
NUM_SAMPLES = 32
# NUM_SAMPLES = 64

# --- Model Input Settings ---
MODEL_NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
MODEL_NORM_STD = [0.26862954, 0.26130258, 0.27577711]

# --- Global Caches ---
_dlib_cache = {}
_yolo_cache = {}
_haar_cache = {}
_transform_cache = {}


# ──────────────────────────
# 🚀 Model Initializers (called once at startup)
# ──────────────────────────

def initialize_dlib_predictors():
    """Initializes and caches dlib models."""
    global _dlib_cache
    if "face_detector" not in _dlib_cache:
        if not os.path.exists(DLIB_LANDMARK_MODEL_PATH):
            raise FileNotFoundError(f"Dlib landmark model not found: {DLIB_LANDMARK_MODEL_PATH}")
        print("[*] Initializing Dlib models...")
        _dlib_cache["face_detector"] = dlib.get_frontal_face_detector()
        _dlib_cache["landmark_predictor"] = dlib.shape_predictor(DLIB_LANDMARK_MODEL_PATH)
    return _dlib_cache["face_detector"], _dlib_cache["landmark_predictor"]


def initialize_yolo_model():
    """Initializes and caches the YOLO model, downloading weights if needed."""
    global _yolo_cache
    if "model" not in _yolo_cache:
        model_path = Path(YOLO_MODEL_PATH)
        if not model_path.exists():
            print(f"[*] YOLO model not found at '{model_path}'. Downloading...")
            try:
                url = "https://github.com/akanametov/yolo-face/releases/download/v8.0/yolov8s-face.pt"
                urlretrieve(url, model_path)
            except Exception as e:
                raise IOError(f"Failed to download YOLO model. Details: {e}")
        print("[*] Initializing YOLO model...")
        _yolo_cache["model"] = YOLO(str(model_path))
    return _yolo_cache["model"]


def initialize_haar_cascades():
    """Initializes and caches the OpenCV Haar Cascade for eye detection."""
    global _haar_cache
    if "eye_cascade" not in _haar_cache:
        haar_xml_path = os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')
        if not os.path.exists(haar_xml_path):
            raise FileNotFoundError(f"Could not find Haar Cascade file: {haar_xml_path}")
        print("[*] Initializing OpenCV Haar Cascade for eyes...")
        _haar_cache["eye_cascade"] = cv2.CascadeClassifier(haar_xml_path)
    return _haar_cache["eye_cascade"]


# ──────────────────────────
# 🖼️ Core Face Detection & Cropping Logic
# ──────────────────────────

# --- Method 1: Dlib ---
def extract_aligned_face(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Detects, aligns, and crops one face from a BGR frame using dlib."""
    face_detector, landmark_predictor = initialize_dlib_predictors()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    faces = face_detector(frame_rgb, 1)
    if not faces: return None

    face = max(faces, key=lambda r: r.width() * r.height())
    shape = landmark_predictor(frame_rgb, face)
    lmk_idxs = [37, 44, 30, 49, 55]
    keypoints = np.array([[shape.part(i).x, shape.part(i).y] for i in lmk_idxs], dtype=np.float32)

    dst = np.array([[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
                    [33.5493, 92.3655], [62.7299, 92.2041]], dtype=np.float32)
    dst[:, 0] += 8.0
    dst = (dst * MODEL_IMG_SIZE / 112.0)
    tform = trans.SimilarityTransform()
    tform.estimate(keypoints, dst)
    M = tform.params[0:2, :]
    warped_rgb = cv2.warpAffine(frame_rgb, M, (MODEL_IMG_SIZE, MODEL_IMG_SIZE), borderValue=0.0)
    return cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2BGR)


def _get_yolo_face_box(frame_bgr: np.ndarray, model=None) -> Optional[np.ndarray]:
    """Internal helper to get the largest face box from YOLO."""
    # If a model isn't passed, initialize it. Otherwise, use the provided one.
    if model is None:
        model = initialize_yolo_model()

    h, w = frame_bgr.shape[:2]
    # Use the 'model' variable for prediction
    results = model.predict(frame_bgr, conf=YOLO_CONF_THRESHOLD, iou=0.4, verbose=False)
    if not results or results[0].boxes.shape[0] == 0: return None

    boxes = results[0].boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    x0, y0, x1, y1 = boxes[np.argmax(areas)]

    x0 = max(0, x0 - YOLO_BBOX_MARGIN)
    y0 = max(0, y0 - YOLO_BBOX_MARGIN)
    x1 = min(w, x1 + YOLO_BBOX_MARGIN)
    y1 = min(h, y1 + YOLO_BBOX_MARGIN)
    return np.array([x0, y0, x1, y1])


# --- Method 2: YOLO (Simple Crop) ---
def extract_yolo_face(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Detects and crops one face using YOLOv8 with a simple square crop."""
    model = initialize_yolo_model() # Initialize the model here
    box = _get_yolo_face_box(frame_bgr, model=model) # Pass the model
    if box is None: return None
    x0, y0, x1, y1 = box.astype(int)
    h, w = frame_bgr.shape[:2]

    width, height = x1 - x0, y1 - y0
    center_x, center_y = x0 + width / 2, y0 + height / 2
    side_length = max(width, height)
    sq_x0 = max(0, int(center_x - side_length / 2))
    sq_y0 = max(0, int(center_y - side_length / 2))
    sq_x1 = min(w, int(center_x + side_length / 2))
    sq_y1 = min(h, int(center_y + side_length / 2))

    cropped_face = frame_bgr[sq_y0:sq_y1, sq_x0:sq_x1]
    if cropped_face.size == 0: return None
    return cv2.resize(cropped_face, (MODEL_IMG_SIZE, MODEL_IMG_SIZE), interpolation=cv2.INTER_AREA)


# --- Method 3: YOLO + Haar (Best-Effort Alignment) ---
def extract_yolo_haar_face(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Detects with YOLO, aligns with Haar Cascades if possible, then crops."""
    h, w = frame_bgr.shape[:2]
    eye_cascade = initialize_haar_cascades()

    box = _get_yolo_face_box(frame_bgr)
    if box is None: return None
    x0, y0, x1, y1 = box

    # Attempt to find eyes for alignment within the upper part of the face box
    roi_y0, roi_y1 = int(y0), int(y0 + (y1 - y0) * 0.6)
    roi_x0, roi_x1 = int(x0), int(x1)
    roi = frame_bgr[roi_y0:roi_y1, roi_x0:roi_x1]

    final_crop = None
    if roi.size > 0:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    else:
        eyes = []

    # If alignment is possible (2 eyes found), rotate the whole image
    if len(eyes) == 2:
        eye_centers = sorted([(ex + ew // 2, ey + eh // 2) for ex, ey, ew, eh in eyes], key=lambda p: p[0])
        left_eye_roi, right_eye_roi = eye_centers

        angle = np.degrees(np.arctan2(right_eye_roi[1] - left_eye_roi[1], right_eye_roi[0] - left_eye_roi[0]))

        face_center = (x0 + (x1 - x0) / 2, y0 + (y1 - y0) / 2)
        M = cv2.getRotationMatrix2D(face_center, angle, 1.0)
        rotated_frame = cv2.warpAffine(frame_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))

        width, height = x1 - x0, y1 - y0
        side = max(width, height)
        sq_x0 = max(0, int(face_center[0] - side / 2))
        sq_y0 = max(0, int(face_center[1] - side / 2))
        final_crop = rotated_frame[sq_y0:int(sq_y0 + side), sq_x0:int(sq_x0 + side)]

    # Fallback: If alignment failed, use the simple square crop on the original image
    if final_crop is None or final_crop.size == 0:
        return extract_yolo_face(frame_bgr)

    return cv2.resize(final_crop, (MODEL_IMG_SIZE, MODEL_IMG_SIZE), interpolation=cv2.INTER_AREA)


# ──────────────────────────
# 🎥 Video Preprocessing Logic
# ──────────────────────────

def _find_and_prepare_faces(
        video_path: str,
        pre_method: str,
        debug_save_path: Optional[str] = None
) -> Optional[List[np.ndarray]]:
    """Internal helper to sample a video and return a list of processed face images."""
    if debug_save_path:
        os.makedirs(debug_save_path, exist_ok=True)
        print(f"[*] Debug mode ON. Saving processed faces to: {debug_save_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}", file=sys.stderr)
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < NUM_SAMPLES:
        cap.release()
        return None

    collected_faces, segment_len = [], total_frames / NUM_SAMPLES
    search_points = [0.5, 0.25, 0.75, 0.1, 0.9]

    for i in range(NUM_SAMPLES):
        found_in_segment = False
        for point in search_points:
            frame_idx = int(i * segment_len + point * segment_len)
            if frame_idx >= total_frames: continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: continue

            if pre_method == 'dlib':
                face = extract_aligned_face(frame)
            elif pre_method == 'yolo':
                face = extract_yolo_face(frame)
            else:  # yolo_haar
                face = extract_yolo_haar_face(frame)

            if face is not None:
                collected_faces.append(face)
                found_in_segment = True
                if debug_save_path:
                    save_name = f"{Path(video_path).stem}_seg{i + 1}_frame{frame_idx}_{pre_method}.jpg"
                    cv2.imwrite(os.path.join(debug_save_path, save_name), face)
                break

        if not found_in_segment:
            cap.release()
            print(f"[*] Preprocessing failed: No valid face in segment {i + 1} using '{pre_method}'.")
            return None

    cap.release()
    return collected_faces


def _get_transform():
    """Returns a cached torchvision transform pipeline."""
    if 'transform' not in _transform_cache:
        _transform_cache['transform'] = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=MODEL_NORM_MEAN, std=MODEL_NORM_STD),
        ])
    return _transform_cache['transform']


def preprocess_video_for_effort_model(
        video_path: str,
        pre_method: str,
        debug_save_path: Optional[str] = None
) -> Optional[torch.Tensor]:
    """Preprocesses a video clip for the EffortDetector model."""
    print(f"[*] Starting video preprocessing for: {os.path.basename(video_path)} using '{pre_method}' method.")

    selected_faces = _find_and_prepare_faces(video_path, pre_method, debug_save_path=debug_save_path)
    if not selected_faces:
        print(f"[*] Video rejected: Failed to sample {NUM_SAMPLES} valid faces.")
        return None

    print(f"[*] Successfully processed {len(selected_faces)} faces. Creating tensor.")
    transform = _get_transform()
    tensor_frames = [transform(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)) for face in selected_faces]

    final_tensor = torch.stack(tensor_frames, dim=0).unsqueeze(0)
    print(f"[*] Preprocessing complete. Final tensor shape: {final_tensor.shape}")
    return final_tensor
