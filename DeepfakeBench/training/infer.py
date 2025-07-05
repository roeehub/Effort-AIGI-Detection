#!/usr/bin/env python3
"""training/infer.py – **verbatim inference path** from demo.py

This version now includes the full `img_align_crop` implementation with
`scale=1.3`, matching the original single‑image demo exactly. Running any
script that imports these helpers will produce *identical* logits and fake
probabilities to `training/demo.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import dlib
import numpy as np
import torch
import yaml
from imutils import face_utils
from PIL import Image as pil_image
from skimage import transform as trans
import torchvision.transforms as T
from torch import nn

from detectors import DETECTOR

# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Model forward helper
# -----------------------------------------------------------------------------

@torch.no_grad()
def inference(model: nn.Module, data_dict):
    data, label = data_dict["image"], data_dict["label"]
    data_dict["image"], data_dict["label"] = data.to(device), label.to(device)
    return model(data_dict, inference=True)

# -----------------------------------------------------------------------------
# Face‑alignment helpers (copied 1‑to‑1 from demo.py)
# -----------------------------------------------------------------------------

_DEF_LMK_IDXS = [37, 44, 30, 49, 55]  # leye, reye, nose, l‑mouth, r‑mouth


def _get_keypts(image: np.ndarray, face, predictor) -> np.ndarray:
    shape = predictor(image, face)
    return np.array([[shape.part(i).x, shape.part(i).y] for i in _DEF_LMK_IDXS], dtype=np.float32)


def img_align_crop(img_rgb: np.ndarray, landmark: np.ndarray, *, outsize: Tuple[int, int], scale: float = 1.3):
    """Exact copy of `img_align_crop` from demo.py (with mask branch removed)."""
    target_size = [112, 112]
    dst = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041],
    ], dtype=np.float32)

    if target_size[1] == 112:
        dst[:, 0] += 8.0

    dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
    dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

    margin_rate = scale - 1.0
    x_margin = outsize[0] * margin_rate / 2.0
    y_margin = outsize[1] * margin_rate / 2.0

    dst[:, 0] += x_margin
    dst[:, 1] += y_margin

    dst[:, 0] *= outsize[0] / (outsize[0] + 2 * x_margin)
    dst[:, 1] *= outsize[1] / (outsize[1] + 2 * y_margin)

    src = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(src, dst)
    M = tform.params[0:2, :]

    warped = cv2.warpAffine(img_rgb, M, (outsize[1], outsize[0]))
    warped = cv2.resize(warped, (outsize[1], outsize[0]))
    return warped


def extract_aligned_face_dlib(face_detector, predictor, image_bgr: np.ndarray, res: int = 224):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    faces = face_detector(rgb, 1)
    if not faces:
        return None, None, None

    face = max(faces, key=lambda r: r.width() * r.height())
    kpts = _get_keypts(rgb, face, predictor)
    cropped_rgb = img_align_crop(rgb, kpts, outsize=(res, res), scale=1.3)
    cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)

    faces2 = face_detector(cropped_bgr, 1)
    landmark_arr = None
    if faces2:
        shape2 = predictor(cropped_bgr, faces2[0])
        landmark_arr = face_utils.shape_to_np(shape2)
    return cropped_bgr, landmark_arr, face

# -----------------------------------------------------------------------------
# Pre‑processing (identical mean/std, PIL path)
# -----------------------------------------------------------------------------

_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711]),
])


def preprocess_face(img_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    return _transform(pil_image.fromarray(img_rgb)).unsqueeze(0)  # 1×3×H×W

# -----------------------------------------------------------------------------
# Inference wrapper (returns *exact* tensors demo prints)
# -----------------------------------------------------------------------------

@torch.inference_mode()
def infer_single_image(img_bgr: np.ndarray, face_detector, landmark_predictor, model: nn.Module):
    if face_detector and landmark_predictor:
        face_aligned, _, _ = extract_aligned_face_dlib(face_detector, landmark_predictor, img_bgr, res=224)
        if face_aligned is None:
            face_aligned = img_bgr  # fallback, exactly like demo.py
    else:
        face_aligned = img_bgr

    face_tensor = preprocess_face(face_aligned).to(device)
    data = {"image": face_tensor, "label": torch.tensor([0]).to(device)}
    preds = inference(model, data)

    cls_out = preds["cls"].squeeze().cpu().numpy()
    prob = preds["prob"].squeeze().cpu().numpy()
    return cls_out, prob

# -----------------------------------------------------------------------------
# File handling
# -----------------------------------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def collect_image_paths(path_str: str) -> List[Path]:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(path_str)

    if p.is_file():
        if p.suffix.lower() not in IMG_EXTS:
            raise ValueError(f"Invalid image: {p.name}")
        return [p]

    imgs = [fp for fp in sorted(p.iterdir()) if fp.suffix.lower() in IMG_EXTS]
    if not imgs:
        raise RuntimeError(f"No images in directory: {path_str}")
    return imgs

# -----------------------------------------------------------------------------
# Detector loader (unchanged)
# -----------------------------------------------------------------------------

def load_detector(detector_cfg: str, weights: str) -> nn.Module:
    with open(detector_cfg, "r") as f:
        cfg = yaml.safe_load(f)
    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)
    ckpt = torch.load(weights, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model
