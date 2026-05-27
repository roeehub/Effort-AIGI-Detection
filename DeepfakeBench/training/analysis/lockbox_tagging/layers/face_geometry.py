"""Layer 2 — face geometry via MediaPipe FaceMesh + cv2.solvePnP for pose."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# MediaPipe FaceMesh canonical 3D model points (mm), six landmark indexes.
# Indices target the 478-landmark FaceMesh:
#  1   = nose tip
#  152 = chin
#  263 = left-eye outer corner (subject-left)
#  33  = right-eye outer corner
#  287 = left mouth corner
#  57  = right mouth corner
_PNP_LANDMARK_IDX = (1, 152, 263, 33, 287, 57)
_PNP_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)

_LEFT_EYE = (33, 160, 158, 133, 153, 144)   # outer, top×2, inner, bottom×2
_RIGHT_EYE = (263, 387, 385, 362, 380, 373)
_MOUTH = (61, 291, 0, 17)  # left-corner, right-corner, upper-mid, lower-mid

_face_mesh_singleton = None


def _face_mesh():
    global _face_mesh_singleton
    if _face_mesh_singleton is None:
        import mediapipe as mp

        _face_mesh_singleton = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=2,
            refine_landmarks=False,
            min_detection_confidence=0.3,
        )
    return _face_mesh_singleton


def _eye_aspect_ratio(lm: np.ndarray, idx: tuple[int, ...]) -> float:
    p = lm[list(idx)]
    # vertical / horizontal in normalized landmark space
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3]) + 1e-6
    return float((v1 + v2) / (2 * h))


def _mouth_aspect_ratio(lm: np.ndarray) -> float:
    p = lm[list(_MOUTH)]
    v = np.linalg.norm(p[2] - p[3])
    h = np.linalg.norm(p[0] - p[1]) + 1e-6
    return float(v / h)


def _solve_pose(lm_xy: np.ndarray, w: int, h: int) -> tuple[float, float, float]:
    """Return (yaw, pitch, roll) in degrees from 6-point PnP."""
    image_points = lm_xy[list(_PNP_LANDMARK_IDX)].astype(np.float64)
    focal = float(w)
    center = (w / 2.0, h / 2.0)
    cam = np.array(
        [[focal, 0.0, center[0]], [0.0, focal, center[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros((4, 1))
    ok, rvec, _ = cv2.solvePnP(
        _PNP_MODEL_POINTS, image_points, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return float("nan"), float("nan"), float("nan")
    rmat, _ = cv2.Rodrigues(rvec)
    # pitch = x-rot, yaw = y-rot, roll = z-rot, decoded from rotation matrix
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy < 1e-6:
        pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
        yaw = np.arctan2(-rmat[2, 0], sy)
        roll = 0.0
    else:
        pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
        yaw = np.arctan2(-rmat[2, 0], sy)
        roll = np.arctan2(rmat[1, 0], rmat[0, 0])
    return float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll))


def compute_face_geometry(path: Path) -> dict:
    out: dict = {
        "face_count": 0,
        "face_pixel_area": None,
        "face_area_ratio": None,
        "face_bbox_x": None,
        "face_bbox_y": None,
        "face_bbox_w": None,
        "face_bbox_h": None,
        "yaw_deg": None,
        "pitch_deg": None,
        "roll_deg": None,
        "eye_aspect_ratio_left": None,
        "eye_aspect_ratio_right": None,
        "mouth_aspect_ratio": None,
    }
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return out
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = _face_mesh().process(rgb)
    if not res.multi_face_landmarks:
        return out

    out["face_count"] = len(res.multi_face_landmarks)
    # Use the largest face if multiple
    best = None
    best_area = -1.0
    for f in res.multi_face_landmarks:
        pts = np.array([(p.x * w, p.y * h) for p in f.landmark], dtype=np.float64)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        a = float((x1 - x0) * (y1 - y0))
        if a > best_area:
            best_area = a
            best = (pts, (x0, y0, x1 - x0, y1 - y0))
    assert best is not None
    pts, (bx, by, bw, bh) = best

    out["face_pixel_area"] = float(bw * bh)
    out["face_area_ratio"] = float(bw * bh / (w * h))
    out["face_bbox_x"] = float(bx)
    out["face_bbox_y"] = float(by)
    out["face_bbox_w"] = float(bw)
    out["face_bbox_h"] = float(bh)

    # Pose from PnP. Normalized landmark coords in pts are in pixel space.
    lm_norm = pts / np.array([w, h])
    out["yaw_deg"], out["pitch_deg"], out["roll_deg"] = _solve_pose(pts, w, h)
    out["eye_aspect_ratio_left"] = _eye_aspect_ratio(lm_norm, _LEFT_EYE)
    out["eye_aspect_ratio_right"] = _eye_aspect_ratio(lm_norm, _RIGHT_EYE)
    out["mouth_aspect_ratio"] = _mouth_aspect_ratio(lm_norm)
    return out


if __name__ == "__main__":
    import json
    import sys

    for arg in sys.argv[1:]:
        print(json.dumps({"path": arg, **compute_face_geometry(Path(arg))}, indent=2))
