"""Face scale-jitter — randomize face pixel-area before the canonical resize.

Counters the face-pixel-area shortcut documented in
``memory/project_face_size_label_leak.md`` (each fake method clusters at a
tight face-size band; the model uses face area as a fake predictor).

Applied symmetrically across labels in the collate-fn loop, BEFORE the
hardcoded 224×224 resize. Effect: an image of original size H×W is
resized to (H * s, W * s) where s ∈ [1 - limit, 1 + limit], then the
existing collate resize squashes it to 224×224. Net effect on the model
input is a randomized "effective face crop area" within [0.75², 1.25²].

Configuration is module-level (not per-call) because the collate function
takes no extra args. ``set_face_scale_jitter_config`` is called once at
trainer init from the yaml top-level ``face_scale_jitter`` block.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np


_CONFIG = {
    "enabled": False,
    "scale_limit": 0.0,
    "interpolation_down": cv2.INTER_AREA,
    "interpolation_up": cv2.INTER_LINEAR,
}


def set_face_scale_jitter_config(
    enabled: bool,
    scale_limit: float,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Configure scale-jitter at trainer init time.

    Idempotent: subsequent calls overwrite the prior config.
    """
    _CONFIG["enabled"] = bool(enabled) and float(scale_limit) > 0.0
    _CONFIG["scale_limit"] = float(scale_limit)
    log = logger or logging.getLogger(__name__)
    if _CONFIG["enabled"]:
        log.info(
            "FaceScaleJitter ENABLED: scale_limit=%.3f (range [%.2f, %.2f])",
            _CONFIG["scale_limit"],
            1.0 - _CONFIG["scale_limit"],
            1.0 + _CONFIG["scale_limit"],
        )
    else:
        log.info("FaceScaleJitter DISABLED")


def apply_face_scale_jitter(img: np.ndarray) -> np.ndarray:
    """Return a scale-jittered copy of ``img`` if enabled, else ``img`` itself.

    The returned image has dimensions different from the input — callers
    must follow up with the canonical resize-to-target.
    """
    if not _CONFIG["enabled"] or not isinstance(img, np.ndarray):
        return img
    if img.ndim < 2:
        return img

    limit = _CONFIG["scale_limit"]
    scale = float(np.random.uniform(1.0 - limit, 1.0 + limit))
    if abs(scale - 1.0) < 1e-3:
        return img

    h, w = img.shape[:2]
    new_h = max(8, int(round(h * scale)))
    new_w = max(8, int(round(w * scale)))
    interp = (
        _CONFIG["interpolation_down"] if scale < 1.0
        else _CONFIG["interpolation_up"]
    )
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def get_face_scale_jitter_config() -> dict:
    """Read-only snapshot of current config (used by tests and diagnostics)."""
    return dict(_CONFIG)
