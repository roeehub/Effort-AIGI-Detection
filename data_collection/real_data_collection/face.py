#!/usr/bin/env python3
"""
face.py – CUDA-accelerated MediaPipe detector with yaw estimate.
"""

from __future__ import annotations
import math
import cv2  # noqa
import numpy as np  # noqa
from dataclasses import dataclass
from typing import List

import mediapipe as mp  # noqa

# lazy-init on first call
_mp_det = None  # lazy-initialized detector
_gpu_capable: bool | None = None  # True / False / None = “unknown”


@dataclass(slots=True, frozen=True)
class Face:
    x: int
    y: int
    w: int
    h: int
    conf: float
    yaw: float  # signed, rad


def _init(det_conf: float):
    global _mp_det
    _mp_det = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=det_conf)

    # Newer MediaPipe removes private _graph_runner; fall back to “unknown”
    try:
        _gpu_capable = bool(getattr(_mp_det, "_graph_runner").runner_options.use_gpu)  # type: ignore[attr-defined]
    except Exception:  # AttributeError or anything else
        _gpu_capable = None


def detect_faces_bgr(frame_bgr, det_conf=0.5) -> List[Face]:
    """Return list[Face] detected in BGR frame."""
    if _mp_det is None:
        _init(det_conf)
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = _mp_det.process(rgb)
    faces = []
    if res.detections:
        for det in res.detections:
            box = det.location_data.relative_bounding_box
            x, y = int(box.xmin * w), int(box.ymin * h)
            fw, fh = int(box.width * w), int(box.height * h)
            conf = det.score[0]
            # naive yaw from keypoints (ear-to-ear); good enough for gate
            lm = det.location_data.relative_keypoints
            if len(lm) >= 2:
                dx = lm[1].x - lm[0].x
                yaw = math.atan2(dx, box.width)
            else:
                yaw = 0.0
            faces.append(Face(x, y, fw, fh, conf, yaw))
    return faces


def gpu_enabled() -> bool | None:
    if _gpu_capable is None:
        _init(0.1)
    return _gpu_capable
