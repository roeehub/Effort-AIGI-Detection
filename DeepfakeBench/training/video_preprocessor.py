"""
video_preprocessor.py
================================================
A modular library to preprocess images and videos for the EffortDetector model.

This module implements the "Path A" strategy: prioritizing accuracy by using
the same dlib-based face alignment and cropping logic as the original model's
training and demonstration code.

It exposes two main functions for use in an application:
1. `extract_aligned_face(frame_bgr)`:
   - Processes a single image frame (NumPy array).
   - Detects the largest face, aligns it using 81 facial landmarks, and
     crops it to the model's required 224x224 size.
   - Returns a cropped face image (NumPy array) or None.

2. `preprocess_video_for_effort_model(video_path)`:
   - Processes a full video file.
   - Samples frames from the video, finds an aligned face in each one,
     and stacks them into a single 5D tensor ([1, T, C, H, W]).
   - This tensor is ready for a single, efficient forward pass in the model.
   - Returns the tensor or None if the video is invalid.

To run this script, you will need:
- The dlib landmark model: `shape_predictor_81_face_landmarks.dat`
- pip install opencv-python dlib scikit-image numpy torch torchvision
"""
from __future__ import annotations
import sys
import os
from typing import List, Optional, Tuple
from pathlib import Path

import cv2  # noqa
import numpy as np  # noqa
import torch  # noqa
import torchvision.transforms as T  # noqa
import dlib  # noqa
from skimage import transform as trans  # noqa

# ──────────────────────────
# 📍 Global Configuration
# ──────────────────────────
# --- Dlib Face Alignment Settings (from original `infer.py`) ---
# ⚠️ IMPORTANT: You must download this file and provide the correct path.
DLIB_LANDMARK_MODEL_PATH = "shape_predictor_81_face_landmarks.dat"
ALIGN_SCALE = 1.3  # Scale factor for the face crop, as used in the original demo
MODEL_IMG_SIZE = 224  # The input image size for the model (CLIP is 224x224)

# --- Frame Sampling Settings for Video ---
NUM_SAMPLES = 32  # Number of frames to sample from the video (T in model input)

# --- Model Input Settings (for CLIP-based EffortDetector) ---
MODEL_NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
MODEL_NORM_STD = [0.26862954, 0.26130258, 0.27577711]

# --- Global Caches ---
_dlib_cache = {}
_transform_cache = {}


# ──────────────────────────
# 🖼️ Core Face Detection & Alignment Logic (Ported from `infer.py`)
# ──────────────────────────

def _get_dlib_predictors():
    """Initializes and caches dlib models to avoid reloading."""
    global _dlib_cache
    if "face_detector" not in _dlib_cache:
        if not os.path.exists(DLIB_LANDMARK_MODEL_PATH):
            raise FileNotFoundError(
                f"Dlib landmark model not found at: {DLIB_LANDMARK_MODEL_PATH}\n"
                "Please download 'shape_predictor_81_face_landmarks.dat' and place it "
                "in the correct directory or update the path in video_preprocessor.py."
            )
        print("[*] Initializing Dlib models (this happens only once)...")
        _dlib_cache["face_detector"] = dlib.get_frontal_face_detector()
        _dlib_cache["landmark_predictor"] = dlib.shape_predictor(DLIB_LANDMARK_MODEL_PATH)
    return _dlib_cache["face_detector"], _dlib_cache["landmark_predictor"]


def _get_keypts(image_rgb: np.ndarray, face: dlib.rectangle, predictor: dlib.shape_predictor) -> np.ndarray:
    """Gets the 5 key landmarks (eyes, nose, mouth) for alignment."""
    shape = predictor(image_rgb, face)
    # Original indices from `infer.py`: left eye, right eye, nose, left mouth, right mouth
    lmk_idxs = [37, 44, 30, 49, 55]
    return np.array([[shape.part(i).x, shape.part(i).y] for i in lmk_idxs], dtype=np.float32)


def _align_and_crop_face_from_landmarks(
        img_rgb: np.ndarray,
        landmark: np.ndarray,
        outsize: Tuple[int, int] = (MODEL_IMG_SIZE, MODEL_IMG_SIZE),
        scale: float = ALIGN_SCALE
) -> np.ndarray:
    """
    Performs similarity transform to align and crop the face.
    This is a direct port of `img_align_crop` from the original `infer.py`.
    """
    # Reference landmarks for a 112x112 image, which are then scaled
    dst = np.array([
        [30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
        [33.5493, 92.3655], [62.7299, 92.2041],
    ], dtype=np.float32)
    dst[:, 0] += 8.0

    dst = (dst * outsize[0] / 112.0)

    # Estimate the transformation matrix and warp the image
    tform = trans.SimilarityTransform()
    tform.estimate(landmark, dst)
    M = tform.params[0:2, :]
    warped = cv2.warpAffine(img_rgb, M, (outsize[1], outsize[0]), borderValue=0.0)
    return warped


def extract_aligned_face(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Detects, aligns, and crops one face from a BGR frame.

    This is the primary function for processing a single image.

    Args:
        frame_bgr: An image frame in BGR format (from cv2.imread or cv2.VideoCapture).

    Returns:
        A 224x224 cropped & aligned face image in BGR format, or None if no face is found.
    """
    face_detector, landmark_predictor = _get_dlib_predictors()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    faces = face_detector(frame_rgb, 1)

    if not faces:
        return None

    # Take the largest face, same as the original logic
    face = max(faces, key=lambda r: r.width() * r.height())
    keypoints = _get_keypts(frame_rgb, face, landmark_predictor)

    # The original demo uses a scale of 1.3, but the alignment function itself
    # doesn't use it directly, it's baked into the `img_align_crop` logic.
    # The `extract_aligned_face_dlib` in infer.py uses a different alignment method.
    # Let's use the most faithful one from `img_align_crop`.
    aligned_face_rgb = _align_and_crop_face_from_landmarks(frame_rgb, keypoints, scale=ALIGN_SCALE)

    return cv2.cvtColor(aligned_face_rgb, cv2.COLOR_RGB2BGR)


# ──────────────────────────
# 🎥 Video Preprocessing Logic
# ──────────────────────────

def _find_and_prepare_faces(
        video_path: str,
        debug_save_path: Optional[str] = None
) -> Optional[List[np.ndarray]]:
    """
    Internal helper to sample a video and return a list of aligned face images.
    If debug_save_path is provided, it saves the processed frames.
    """
    # --- ADDITION: Create debug directory if needed ---
    if debug_save_path:
        os.makedirs(debug_save_path, exist_ok=True)
        print(f"[*] Debug mode ON. Saving aligned faces to: {debug_save_path}")
    # --- END ADDITION ---

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}", file=sys.stderr)
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < NUM_SAMPLES:
        print(f"[*] Video rejected: Not enough frames ({total_frames}) for sampling ({NUM_SAMPLES}).")
        cap.release()
        return None

    collected_faces = []
    segment_len = total_frames / NUM_SAMPLES
    search_points = [0.5, 0.25, 0.75, 0.1, 0.9]

    for i in range(NUM_SAMPLES):
        found_in_segment = False
        for point in search_points:
            frame_idx = int(i * segment_len + point * segment_len)
            if frame_idx >= total_frames: continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: continue

            cropped_face_bgr = extract_aligned_face(frame)

            if cropped_face_bgr is not None:
                collected_faces.append(cropped_face_bgr)
                found_in_segment = True

                # --- ADDITION: Save the debug frame ---
                if debug_save_path:
                    video_name = Path(video_path).stem
                    save_name = f"{video_name}_seg{i + 1}_frame{frame_idx}.jpg"
                    cv2.imwrite(os.path.join(debug_save_path, save_name), cropped_face_bgr)
                # --- END ADDITION ---

                break

        if not found_in_segment:
            cap.release()
            print(f"[*] Preprocessing failed: Could not find a valid face in segment {i + 1}.")
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
        debug_save_path: Optional[str] = None
) -> Optional[torch.Tensor]:
    """
    Preprocesses a video clip for the EffortDetector model.

    Args:
        video_path: The path to the video clip.
        debug_save_path: If set, saves aligned faces to this directory.

    Returns:
        A torch.Tensor of shape [1, T, C, H, W] or None.
    """
    print(f"[*] Starting video preprocessing for: {os.path.basename(video_path)}")

    # 1. Sample video and get a list of cropped/aligned face images
    selected_faces = _find_and_prepare_faces(video_path, debug_save_path=debug_save_path)

    if not selected_faces:
        print(f"[*] Video rejected: Failed to sample {NUM_SAMPLES} valid faces.")
        return None

    print(f"[*] Successfully sampled and aligned {len(selected_faces)} faces. Now creating tensor.")

    # 2. Get the transformation pipeline
    transform = _get_transform()

    # 3. Process each face image and stack into a tensor
    tensor_frames = []
    for face_img_bgr in selected_faces:
        rgb_face = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
        tensor_frame = transform(rgb_face)
        tensor_frames.append(tensor_frame)

    video_tensor = torch.stack(tensor_frames, dim=0)
    final_tensor = video_tensor.unsqueeze(0)

    print(f"[*] Preprocessing complete. Final tensor shape: {final_tensor.shape}")
    return final_tensor
