"""Layer 1 — file & technical-quality metrics from a JPEG on disk."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _jpeg_quality_factor(qtables) -> float | None:
    """Estimate JPEG QF from the luma quantization table.

    Pillow's `Image.quantization` is a dict with int keys (0=luma, 1=chroma);
    each value is a 64-element flat list/tuple of int quantization values.
    Returns None for non-JPEG files."""
    if not qtables:
        return None
    # Be lenient about key types (older Pillow may use str/int).
    luma = None
    if isinstance(qtables, dict):
        for k in (0, "0"):
            if k in qtables:
                luma = qtables[k]
                break
        if luma is None and qtables:
            # Pick the first table if neither 0 nor "0" is a key.
            luma = next(iter(qtables.values()))
    elif isinstance(qtables, (list, tuple)) and qtables:
        luma = qtables[0]
    if luma is None or len(luma) == 0:
        return None
    arr = np.asarray(luma, dtype=np.float32)
    s = float(arr.sum())
    if s <= 0:
        return None
    # libjpeg canonical luma table sums to 1117 at QF=50.
    if s >= 1117 * 2:
        qf = 5000.0 / s
    else:
        qf = 100.0 - s / (1117.0 * 2.0) * 50.0
    return max(1.0, min(100.0, qf))


def compute_quality(path: Path) -> dict:
    """Return the Layer-1 dict for one image. Robust to corrupt files."""
    out: dict = {
        "file_bytes": path.stat().st_size,
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "actual_format": None,   # PIL format string (e.g. "JPEG", "PNG")
        "image_mode": None,      # PIL mode (e.g. "RGB", "RGBA")
        "has_alpha": None,
        "jpeg_qf_estimate": None,
        "sharpness_laplacian": None,
        "brightness_v_mean": None,
        "brightness_v_std": None,
        "contrast_rms": None,
        "saturation_s_mean": None,
        "is_clipped_highlights": None,
        "decode_ok": False,
    }
    try:
        with Image.open(path) as im:
            out["width"], out["height"] = im.size
            out["aspect_ratio"] = round(im.size[0] / im.size[1], 4) if im.size[1] else None
            out["actual_format"] = im.format
            out["image_mode"] = im.mode
            out["has_alpha"] = "A" in (im.mode or "")
            qt = getattr(im, "quantization", None)
            out["jpeg_qf_estimate"] = _jpeg_quality_factor(qt) if qt else None
    except Exception:
        return out

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return out
    out["decode_ok"] = True

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out["sharpness_laplacian"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    out["contrast_rms"] = float(gray.std())

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    s = hsv[..., 1]
    out["brightness_v_mean"] = float(v.mean())
    out["brightness_v_std"] = float(v.std())
    out["saturation_s_mean"] = float(s.mean())
    out["is_clipped_highlights"] = float((v == 255).mean())
    return out


if __name__ == "__main__":
    import json
    import sys

    for arg in sys.argv[1:]:
        print(json.dumps({"path": arg, **compute_quality(Path(arg))}, indent=2))
