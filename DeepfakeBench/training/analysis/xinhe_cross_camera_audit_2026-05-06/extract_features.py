"""Extract per-frame IQ features for the Xinhe cross-camera audit (2026-05-06).

Compares 92 may6 (false-flag) frames against 60 may5 (correct) reference frames.
CPU-only. n_jobs<=2.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

# silence absl/protobuf chatter from mediapipe
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import mediapipe as mp  # noqa: E402

ROOT = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis/xinhe_cross_camera_audit_2026-05-06"
)
RAW = ROOT / "raw"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def sobel_mean(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(mag.mean())


def hf_energy_ratio(gray: np.ndarray, frac: float = 0.5) -> float:
    """Radial FFT — fraction of total spectral energy past `frac` of Nyquist radius."""
    g = gray.astype(np.float32)
    # zero-mean to drop DC
    g = g - g.mean()
    F = np.fft.fftshift(np.fft.fft2(g))
    P = (F.real * F.real + F.imag * F.imag).astype(np.float64)
    h, w = P.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices(P.shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = float(rr.max())
    total = P.sum() + 1e-12
    hf = P[rr >= rmax * frac].sum()
    return float(hf / total)


def detect_face_bbox(rgb: np.ndarray, detector) -> Optional[tuple[int, int, int, int]]:
    """Returns (x0, y0, x1, y1) bounding box or None."""
    res = detector.process(rgb)
    if not res.detections:
        return None
    h, w = rgb.shape[:2]
    # take highest-confidence detection
    det = max(res.detections, key=lambda d: d.score[0] if d.score else 0.0)
    rb = det.location_data.relative_bounding_box
    x0 = max(int(rb.xmin * w), 0)
    y0 = max(int(rb.ymin * h), 0)
    x1 = min(int((rb.xmin + rb.width) * w), w)
    y1 = min(int((rb.ymin + rb.height) * h), h)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def extract_for_path(path: Path, detector) -> dict:
    """Compute IQ features for one image path."""
    rec: dict = {"path": str(path), "filename": path.name}
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        rec["error"] = "imread_failed"
        return rec
    h, w = img.shape[:2]
    rec["width"] = w
    rec["height"] = h
    rec["pixels"] = w * h

    # Color
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rec["luma_mean"] = float(yuv[..., 0].mean())
    rec["luma_std"] = float(yuv[..., 0].std())
    rec["sat_mean"] = float(hsv[..., 1].mean())
    rec["sat_std"] = float(hsv[..., 1].std())
    rec["hue_mean"] = float(hsv[..., 0].mean())
    rec["r_mean"] = float(rgb[..., 0].mean())
    rec["g_mean"] = float(rgb[..., 1].mean())
    rec["b_mean"] = float(rgb[..., 2].mean())
    rec["r_std"] = float(rgb[..., 0].std())
    rec["g_std"] = float(rgb[..., 1].std())
    rec["b_std"] = float(rgb[..., 2].std())

    # Sharpness — full image
    rec["lap_var_full"] = laplacian_var(gray)
    rec["sobel_mean_full"] = sobel_mean(gray)
    rec["hf_ratio_full"] = hf_energy_ratio(gray, frac=0.5)

    # Face crop sharpness
    bbox = detect_face_bbox(rgb, detector)
    if bbox is None:
        rec["face_detected"] = 0
        rec["face_x0"] = np.nan
        rec["face_y0"] = np.nan
        rec["face_x1"] = np.nan
        rec["face_y1"] = np.nan
        rec["face_w"] = np.nan
        rec["face_h"] = np.nan
        rec["face_area"] = np.nan
        rec["face_area_frac"] = np.nan
        rec["lap_var_face"] = np.nan
        rec["sobel_mean_face"] = np.nan
        rec["hf_ratio_face"] = np.nan
        rec["luma_mean_face"] = np.nan
    else:
        x0, y0, x1, y1 = bbox
        face_gray = gray[y0:y1, x0:x1]
        face_rgb = rgb[y0:y1, x0:x1]
        face_yuv = yuv[y0:y1, x0:x1]
        rec["face_detected"] = 1
        rec["face_x0"] = x0
        rec["face_y0"] = y0
        rec["face_x1"] = x1
        rec["face_y1"] = y1
        rec["face_w"] = x1 - x0
        rec["face_h"] = y1 - y0
        rec["face_area"] = (x1 - x0) * (y1 - y0)
        rec["face_area_frac"] = rec["face_area"] / max(w * h, 1)
        rec["lap_var_face"] = laplacian_var(face_gray)
        rec["sobel_mean_face"] = sobel_mean(face_gray)
        rec["hf_ratio_face"] = hf_energy_ratio(face_gray, frac=0.5) if face_gray.size > 16 else np.nan
        rec["luma_mean_face"] = float(face_yuv[..., 0].mean())

    return rec


def main() -> int:
    # Load deployment scores for may6
    tags_path = OUT / "frame_tags.json"
    score_by_filename: dict[str, float] = {}
    if tags_path.exists():
        with open(tags_path) as f:
            tags = json.load(f)
        for tag in tags.get("tags", []):
            for it in tag.get("items", []):
                fname = it.get("source_filename") or ""
                participant = it.get("participant") or ""
                score = it.get("score", float("nan"))
                # both bare and prefixed forms — disk has "Generator PC__<bare>"
                score_by_filename[fname] = score
                if participant:
                    score_by_filename[f"{participant}__{fname}"] = score

    # model_selection=0 = short-range model — required because input frames
    # are already 173–190 px face crops; the full-range model (1) returns
    # zero detections at this resolution.
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.3
    )

    rows: list[dict] = []
    skipped: list[str] = []

    for pop, raw_dir in [("may6_falseflag", RAW / "may6"), ("may5_correct", RAW / "may5")]:
        files = sorted(glob.glob(str(raw_dir / "*.png"))) + sorted(
            glob.glob(str(raw_dir / "*.jpg"))
        )
        print(f"[{pop}] {len(files)} files in {raw_dir}", flush=True)
        for i, fp in enumerate(files):
            rec = extract_for_path(Path(fp), detector)
            rec["population"] = pop
            if pop == "may6_falseflag":
                rec["deploy_score"] = score_by_filename.get(Path(fp).name, np.nan)
            else:
                rec["deploy_score"] = np.nan
            if rec.get("face_detected", 0) == 0:
                skipped.append(f"{pop}:{Path(fp).name}")
            rows.append(rec)
            if (i + 1) % 25 == 0:
                print(f"  [{pop}] {i+1}/{len(files)}", flush=True)

    detector.close()

    df = pd.DataFrame(rows)
    print(f"Total frames: {len(df)} | face-detect skips: {len(skipped)}")
    df.to_csv(OUT / "per_frame_features.csv", index=False)
    with open(OUT / "face_skips.txt", "w") as f:
        f.write("\n".join(skipped))
    print(f"Wrote {OUT / 'per_frame_features.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
