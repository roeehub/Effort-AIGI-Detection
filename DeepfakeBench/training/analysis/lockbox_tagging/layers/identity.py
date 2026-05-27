"""Layer 3 — ArcFace (InsightFace buffalo_l/w600k_r50) via onnxruntime.

We *do not* use the manifest's identity_key for grouping (it collides across
people). The embedding is stored raw and used downstream for k-NN-based
property-style outlier detection (see analyze.py)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ARCFACE_ONNX = Path.home() / ".insightface/models/buffalo_l/w600k_r50.onnx"
INPUT_SIZE = 112  # buffalo_l recognition input

_session_singleton = None


def _session():
    """Lazy-load the ArcFace ONNX session, preferring CoreML EP on macOS."""
    global _session_singleton
    if _session_singleton is not None:
        return _session_singleton
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    providers: list = []
    if "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
    providers.append("CPUExecutionProvider")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(ARCFACE_ONNX), providers=providers, sess_options=so)
    _session_singleton = sess
    return sess


def _crop_face(img: np.ndarray, bbox: tuple[float, float, float, float] | None) -> np.ndarray:
    """Return a 112×112 BGR crop. If bbox is None, center-resize the full image."""
    h, w = img.shape[:2]
    if bbox is not None:
        x, y, bw, bh = bbox
        # Expand 12% on each side and clamp.
        cx, cy = x + bw / 2.0, y + bh / 2.0
        side = max(bw, bh) * 1.12
        x0 = int(max(0, round(cx - side / 2.0)))
        y0 = int(max(0, round(cy - side / 2.0)))
        x1 = int(min(w, round(cx + side / 2.0)))
        y1 = int(min(h, round(cy + side / 2.0)))
        if x1 - x0 < 4 or y1 - y0 < 4:
            crop = img
        else:
            crop = img[y0:y1, x0:x1]
    else:
        crop = img
    return cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)


def _preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    """ArcFace input: RGB, mean=127.5, std=128.0, NCHW float32."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - 127.5) / 128.0
    return np.transpose(rgb, (2, 0, 1))[None, ...]


def compute_identity(path: Path, face_bbox: tuple[float, float, float, float] | None = None) -> dict:
    out: dict = {
        "arcface_embed": None,
        "arcface_norm": None,
    }
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return out
    crop = _crop_face(img, face_bbox)
    x = _preprocess(crop)
    sess = _session()
    out_name = sess.get_outputs()[0].name
    in_name = sess.get_inputs()[0].name
    emb = sess.run([out_name], {in_name: x})[0][0]  # (512,)
    norm = float(np.linalg.norm(emb))
    if norm > 0:
        emb_unit = (emb / norm).astype(np.float16)
    else:
        emb_unit = emb.astype(np.float16)
    out["arcface_embed"] = emb_unit.tolist()
    out["arcface_norm"] = norm
    return out


if __name__ == "__main__":
    import json
    import sys

    for arg in sys.argv[1:]:
        r = compute_identity(Path(arg))
        if r.get("arcface_embed") is not None:
            r["arcface_embed"] = f"<512-d float16, first 4: {r['arcface_embed'][:4]}>"
        print(json.dumps({"path": arg, **r}, indent=2))
