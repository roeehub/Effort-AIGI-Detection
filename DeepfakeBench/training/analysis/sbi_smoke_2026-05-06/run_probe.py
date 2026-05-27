"""SBI_SMOKE_2026-05-06 — generate self-blended pseudo-fakes from target-domain real
face crops; verify visual plausibility, IQ-axis separability, score distribution.

Validates the foundation of the planned PE_SBI packet (NEXT_STEPS_PLAN_2026-05-06.md §8.2 P2).

Inputs (locally cached, no GCS pulls):
  - Real source frames: analysis/team_sanity_check_2026-05-05/frames/{Dor,Noyn,Roee,Xiang,Xinhe}/
    (210 face crops, ~163x163, target-domain real teams captures, used in Probe 1)
  - Known fake frames: analysis/identity_browser_2026-05-05/frames/live_prod__*/
    (~1480 face crops from live_fakes_teams_prod for IQ comparison)
  - Cached P8A/E2B/PA_3800 scores per gs:// frame path:
    analysis/identity_browser_2026-05-05/data/{team_sanity_may5,live_fakes}_scores_*.csv

Outputs (analysis/sbi_smoke_2026-05-06/outputs/):
  - pseudofakes/      sample pseudo-fake images (24 sxs panels + 100-300 standalone)
  - iq_comparison.csv per-axis means/stds + Cohen's d for {real, sbi, fake}
  - score_distributions.csv  cached scores for {real, fake}; SBI scores via blueprint
  - summary.json      per-criterion verdict
  - verdict.json      top-level GREEN/AMBER/RED for P2 launch
  - FINDINGS.md       1-2 page synthesis

CPU only.  No sklearn n_jobs=-1.  No gs:// pulls.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import pandas as pd
from scipy import stats

# silence absl/protobuf chatter from mediapipe
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import mediapipe as mp  # noqa: E402

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("sbi_smoke")

THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "outputs"
PSEUDO_DIR = OUT_DIR / "pseudofakes"
PSEUDO_DIR.mkdir(parents=True, exist_ok=True)

REAL_FRAMES_ROOT = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis/team_sanity_check_2026-05-05/frames"
)
FAKE_FRAMES_ROOT = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis/identity_browser_2026-05-05/frames"
)
SCORES_DIR = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis/identity_browser_2026-05-05/data"
)

REAL_IDENTITIES = ["Dor", "Noyn", "Roee", "Xiang", "Xinhe"]
N_FAKE_SAMPLES = 200      # how many known fakes to sample for IQ comparison
N_SBI_TARGET = 200        # number of SBI pseudo-fakes to generate
N_VISUAL_PANELS = 24      # how many side-by-side real|sbi panels to dump

RNG_SEED = 20260506


# ─────────────────────────── face-mask construction ───────────────────────────


# Convex-hull mask using a coarse ring of FaceMesh landmarks (jaw + brows).
# Indices from MediaPipe FaceMesh canonical 468-point topology — outer face oval.
# https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
FACE_OVAL_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
]


def detect_landmarks(rgb: np.ndarray, mesher) -> Optional[np.ndarray]:
    """Returns Nx2 landmark array in pixel coords (468 points), or None."""
    res = mesher.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    h, w = rgb.shape[:2]
    pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
    return pts


def build_face_mask(landmarks: np.ndarray, h: int, w: int,
                    feather_px: int = 11,
                    contraction_px: int = 0) -> np.ndarray:
    """Build feathered face mask (uint8 0–255) from landmark hull.

    contraction_px: optionally erode the hull a bit (negative => dilate) so the
    blend boundary is INSIDE the face rather than at the silhouette."""
    if landmarks is None:
        return np.zeros((h, w), dtype=np.uint8)
    oval_pts = landmarks[FACE_OVAL_LANDMARKS]
    hull = cv2.convexHull(oval_pts.astype(np.int32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if contraction_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * contraction_px + 1, 2 * contraction_px + 1)
        )
        mask = cv2.erode(mask, kernel)
    elif contraction_px < 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (-2 * contraction_px + 1, -2 * contraction_px + 1)
        )
        mask = cv2.dilate(mask, kernel)
    if feather_px > 0:
        # Gaussian blur of the mask gives soft alpha boundary
        k = max(3, 2 * feather_px + 1)
        mask = cv2.GaussianBlur(mask, (k, k), feather_px / 2.0)
    return mask


def fallback_elliptical_mask(rgb: np.ndarray, feather_px: int = 11) -> np.ndarray:
    """Used only when FaceMesh fails — pad-elliptical mask centred on image."""
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cy, cx = h // 2, w // 2
    cv2.ellipse(mask, (cx, cy), (int(w * 0.36), int(h * 0.46)), 0, 0, 360, 255, -1)
    if feather_px > 0:
        k = max(3, 2 * feather_px + 1)
        mask = cv2.GaussianBlur(mask, (k, k), feather_px / 2.0)
    return mask


# ─────────────────────────── source-side transforms ───────────────────────────


def affine_transform(img: np.ndarray, tx: float, ty: float,
                     rotation_deg: float, scale: float) -> np.ndarray:
    """Translation + rotation + isotropic scale around image centre."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), rotation_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def color_jitter(img: np.ndarray, brightness: float, contrast: float,
                 hue_shift_deg: float, sat_scale: float,
                 noise_std: float, rng: np.random.RandomState) -> np.ndarray:
    """Brightness, contrast, hue/sat shift, additive Gaussian noise."""
    out = img.astype(np.float32)
    # brightness  (additive in 0..255)
    out += brightness * 255.0
    # contrast around the mean
    mean = out.mean()
    out = (out - mean) * contrast + mean
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    # hue / sat in HSV
    if abs(hue_shift_deg) > 1e-6 or abs(sat_scale - 1.0) > 1e-6:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + hue_shift_deg / 2.0) % 180.0  # opencv H in [0,180)
        hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0.0, 255.0)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # additive Gaussian noise on the source side
    if noise_std > 0:
        noise = rng.normal(0.0, noise_std * 255.0, size=out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
    return out


# ─────────────────────────── SBI pipeline ─────────────────────────────────────


def make_sbi(img_bgr: np.ndarray, mesher, rng: np.random.RandomState,
             feather_px: Optional[int] = None,
             geom_strength: float = 1.0,
             color_strength: float = 1.0,
             noise_std: float = 0.01) -> dict:
    """Self-blend a real face crop into a pseudo-fake.

    Returns dict with: sbi_image, mask, params, fallback_mask_used, landmarks_ok.
    """
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    landmarks = detect_landmarks(rgb, mesher)
    landmarks_ok = landmarks is not None

    # Source-side transforms (small magnitudes — SBI premise)
    tx = rng.uniform(-3.0, 3.0) * geom_strength
    ty = rng.uniform(-3.0, 3.0) * geom_strength
    rot = rng.uniform(-3.0, 3.0) * geom_strength
    scale = 1.0 + rng.uniform(-0.03, 0.03) * geom_strength

    bri = rng.uniform(-0.04, 0.04) * color_strength
    con = 1.0 + rng.uniform(-0.04, 0.04) * color_strength
    hue = rng.uniform(-2.0, 2.0) * color_strength
    sat = 1.0 + rng.uniform(-0.05, 0.05) * color_strength

    transformed = affine_transform(img_bgr, tx, ty, rot, scale)
    transformed = color_jitter(transformed, bri, con, hue, sat,
                               noise_std, rng)

    # Face mask: convex hull of landmarks (preferred) or elliptical fallback
    if feather_px is None:
        feather_px = int(rng.choice([5, 7, 9, 11, 13, 15]))
    fallback = False
    if landmarks_ok:
        mask = build_face_mask(landmarks, h, w, feather_px=feather_px,
                               contraction_px=0)
    else:
        mask = fallback_elliptical_mask(rgb, feather_px=feather_px)
        fallback = True

    alpha = mask.astype(np.float32)[..., None] / 255.0
    blended = alpha * transformed.astype(np.float32) + (1.0 - alpha) * img_bgr.astype(np.float32)
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    return {
        "sbi_image": blended,
        "mask": mask,
        "fallback_mask_used": fallback,
        "landmarks_ok": landmarks_ok,
        "params": {
            "tx": tx, "ty": ty, "rot_deg": rot, "scale": scale,
            "brightness": bri, "contrast": con, "hue_shift": hue,
            "sat_scale": sat, "noise_std": noise_std, "feather_px": feather_px,
        },
    }


# ─────────────────────────── IQ axis extraction ───────────────────────────────


def laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def sobel_mean(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(mag.mean())


def hf_energy_ratio(gray: np.ndarray, frac: float = 0.5) -> float:
    g = gray.astype(np.float32)
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
    res = detector.process(rgb)
    if not res.detections:
        return None
    h, w = rgb.shape[:2]
    det = max(res.detections, key=lambda d: d.score[0] if d.score else 0.0)
    rb = det.location_data.relative_bounding_box
    x0 = max(int(rb.xmin * w), 0)
    y0 = max(int(rb.ymin * h), 0)
    x1 = min(int((rb.xmin + rb.width) * w), w)
    y1 = min(int((rb.ymin + rb.height) * h), h)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def extract_iq(img_bgr: np.ndarray, detector) -> dict:
    """Same axes used in Probe 1 / xinhe_cross_camera_audit."""
    rec: dict = {}
    h, w = img_bgr.shape[:2]
    rec["width"] = w
    rec["height"] = h
    rec["pixels"] = w * h

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

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

    rec["lap_var_full"] = laplacian_var(gray)
    rec["sobel_mean_full"] = sobel_mean(gray)
    rec["hf_ratio_full"] = hf_energy_ratio(gray, frac=0.5)

    bbox = detect_face_bbox(rgb, detector)
    if bbox is None:
        rec["face_detected"] = 0
        rec["face_area"] = np.nan
        rec["face_area_frac"] = np.nan
        rec["lap_var_face"] = np.nan
        rec["sobel_mean_face"] = np.nan
        rec["hf_ratio_face"] = np.nan
        rec["luma_mean_face"] = np.nan
    else:
        x0, y0, x1, y1 = bbox
        face_gray = gray[y0:y1, x0:x1]
        face_yuv = yuv[y0:y1, x0:x1]
        rec["face_detected"] = 1
        rec["face_area"] = (x1 - x0) * (y1 - y0)
        rec["face_area_frac"] = rec["face_area"] / max(w * h, 1)
        rec["lap_var_face"] = laplacian_var(face_gray)
        rec["sobel_mean_face"] = sobel_mean(face_gray)
        rec["hf_ratio_face"] = hf_energy_ratio(face_gray, frac=0.5) if face_gray.size > 16 else np.nan
        rec["luma_mean_face"] = float(face_yuv[..., 0].mean())

    return rec


# ─────────────────────────── Stats helpers ────────────────────────────────────


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sa = a.std(ddof=1); sb = b.std(ddof=1)
    pooled = float(np.sqrt(((len(a) - 1) * sa * sa + (len(b) - 1) * sb * sb) /
                            max(len(a) + len(b) - 2, 1)))
    if pooled <= 1e-12:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def summary_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=float); arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "p10": float("nan"), "p90": float("nan"), "std": float("nan")}
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    }


# ─────────────────────────── Frame collection ────────────────────────────────


def collect_real_frames() -> list[Path]:
    out: list[Path] = []
    for ident in REAL_IDENTITIES:
        d = REAL_FRAMES_ROOT / ident
        if not d.is_dir():
            continue
        out.extend(sorted(d.glob("*.png")))
        out.extend(sorted(d.glob("*.jpg")))
    return out


def collect_fake_frames(n: int, rng: np.random.RandomState) -> list[Path]:
    out: list[Path] = []
    fake_dirs = sorted([p for p in FAKE_FRAMES_ROOT.iterdir()
                        if p.is_dir() and p.name.startswith("live_prod__")])
    for d in fake_dirs:
        out.extend(sorted(d.glob("*.png")))
    rng.shuffle(out)
    return out[:n]


def gs_uri_from_real_local(p: Path) -> str:
    """Reconstruct gs:// path used in score CSVs from local frame path."""
    # local: analysis/team_sanity_check_2026-05-05/frames/<ident>/<filename>
    # gs   : gs://real-teams-dor-roee/Roee-Dor-Xiang-Xinhe-noyn-may5/frames/<ident>/<filename>
    return f"gs://real-teams-dor-roee/Roee-Dor-Xiang-Xinhe-noyn-may5/frames/{p.parent.name}/{p.name}"


def gs_uri_from_fake_local(p: Path) -> str:
    """Reconstruct gs:// path for live_fakes_teams_prod frames."""
    # local: analysis/identity_browser_2026-05-05/frames/live_prod__<suite>/<file>.png
    # local-file: live_fakes_teams_prod__Generator PC__frame_001977_seq3465.png
    # gs file:    Generator PC__frame_001977_seq3465.png
    suite = p.parent.name.replace("live_prod__", "")
    fname = p.name.replace("live_fakes_teams_prod__", "")
    return f"gs://live-fakes-teams-prod/fake/session_20260414_112354/{suite}/{fname}"


def load_score_index(scores_csv: Path) -> dict[str, float]:
    if not scores_csv.exists():
        return {}
    out: dict[str, float] = {}
    with scores_csv.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                out[row["frame_path"]] = float(row["frame_prob"])
            except (KeyError, ValueError):
                continue
    return out


# ─────────────────────────── Main run ─────────────────────────────────────────


def make_sbs_panel(panels: Sequence[tuple[np.ndarray, np.ndarray, str]],
                   out_path: Path, n_cols: int = 4, tile: int = 180) -> None:
    """Save N (real | sbi) side-by-side panels arranged in a grid.
    Each cell is resized to (tile, tile) to handle heterogeneous source sizes.
    """
    if not panels:
        return
    h = w = tile
    sep_w = 4
    label_h = 22
    cell_w = w * 2 + sep_w
    cell_h = h + label_h
    n = len(panels)
    n_rows = (n + n_cols - 1) // n_cols
    canvas = np.full((n_rows * cell_h, n_cols * cell_w, 3), 255, dtype=np.uint8)
    for i, (real, sbi, lbl) in enumerate(panels):
        r = i // n_cols
        c = i % n_cols
        y0 = r * cell_h
        x0 = c * cell_w
        real_t = cv2.resize(real, (w, h), interpolation=cv2.INTER_LINEAR)
        sbi_t = cv2.resize(sbi, (w, h), interpolation=cv2.INTER_LINEAR)
        canvas[y0:y0 + h, x0:x0 + w] = real_t
        canvas[y0:y0 + h, x0 + w + sep_w:x0 + 2 * w + sep_w] = sbi_t
        cv2.putText(canvas, lbl, (x0 + 4, y0 + h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "real", (x0 + 4, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "sbi", (x0 + w + sep_w + 4, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def main() -> int:
    log.info("=" * 70)
    log.info("SBI_SMOKE_2026-05-06 — generate self-blended pseudo-fakes")
    log.info("=" * 70)

    rng = np.random.RandomState(RNG_SEED)
    random.seed(RNG_SEED)

    real_paths = collect_real_frames()
    log.info("real frames available: %d", len(real_paths))
    if len(real_paths) < 100:
        log.error("not enough real frames (need 100-300, have %d)", len(real_paths))
        return 2

    fake_paths = collect_fake_frames(N_FAKE_SAMPLES, rng)
    log.info("known-fake frames sampled: %d", len(fake_paths))

    # Cap real to N_SBI_TARGET (200), respecting the 210 cache bound
    real_for_sbi = real_paths[:N_SBI_TARGET]

    mesher = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.3,
        static_image_mode=True,
    )
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.3,
    )

    # ──────── 1.  Generate SBI pseudo-fakes ────────
    log.info("[1/4] generating %d SBI pseudo-fakes ...", len(real_for_sbi))
    t0 = time.time()
    sbi_records: list[dict] = []
    sbi_images: dict[str, np.ndarray] = {}
    n_landmarks_ok = 0
    n_fallback = 0
    panel_indices = sorted(rng.choice(len(real_for_sbi),
                                       size=min(N_VISUAL_PANELS, len(real_for_sbi)),
                                       replace=False).tolist())
    panels: list[tuple[np.ndarray, np.ndarray, str]] = []

    for i, p in enumerate(real_for_sbi):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            log.warning("[1/4] decode-fail %s", p)
            continue
        result = make_sbi(img, mesher, rng)
        if result["landmarks_ok"]:
            n_landmarks_ok += 1
        if result["fallback_mask_used"]:
            n_fallback += 1
        sbi_id = f"sbi_{i:04d}_{p.parent.name}_{p.stem}.png"
        sbi_path = PSEUDO_DIR / sbi_id
        # Only save first ~60 standalone images to keep dir size reasonable
        if i < 60:
            cv2.imwrite(str(sbi_path), result["sbi_image"])
        sbi_images[sbi_id] = result["sbi_image"]
        sbi_records.append({
            "sbi_id": sbi_id,
            "real_source_path": str(p),
            "real_source_basename": p.name,
            "real_source_identity": p.parent.name,
            "landmarks_ok": int(result["landmarks_ok"]),
            "fallback_mask": int(result["fallback_mask_used"]),
            **result["params"],
        })
        if i in panel_indices:
            panels.append((img, result["sbi_image"],
                           f"{p.parent.name}/{p.stem[:14]}"))
        if (i + 1) % 50 == 0:
            log.info("[1/4]   generated %d/%d", i + 1, len(real_for_sbi))

    log.info("[1/4] done: landmarks_ok=%d/%d  fallback_mask=%d  elapsed=%.1fs",
             n_landmarks_ok, len(real_for_sbi), n_fallback, time.time() - t0)

    pd.DataFrame(sbi_records).to_csv(OUT_DIR / "sbi_records.csv", index=False)
    panel_path = OUT_DIR / "pseudofakes" / "_visual_panel_real_vs_sbi.png"
    make_sbs_panel(panels, panel_path, n_cols=4)
    log.info("[1/4] visual panel -> %s", panel_path)

    # ──────── 2.  IQ axis comparison ────────
    log.info("[2/4] computing IQ axes for {real, sbi, fake} ...")
    t0 = time.time()
    real_iq: list[dict] = []
    for i, p in enumerate(real_for_sbi):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        rec = extract_iq(img, detector)
        rec["population"] = "real_source"
        rec["frame_id"] = p.name
        real_iq.append(rec)
        if (i + 1) % 100 == 0:
            log.info("[2/4]   real %d/%d", i + 1, len(real_for_sbi))

    sbi_iq: list[dict] = []
    sbi_ids = list(sbi_images.keys())
    for i, sid in enumerate(sbi_ids):
        img = sbi_images[sid]
        rec = extract_iq(img, detector)
        rec["population"] = "sbi_pseudofake"
        rec["frame_id"] = sid
        sbi_iq.append(rec)
        if (i + 1) % 100 == 0:
            log.info("[2/4]   sbi  %d/%d", i + 1, len(sbi_ids))

    fake_iq: list[dict] = []
    for i, p in enumerate(fake_paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        rec = extract_iq(img, detector)
        rec["population"] = "known_fake"
        rec["frame_id"] = p.name
        fake_iq.append(rec)
        if (i + 1) % 100 == 0:
            log.info("[2/4]   fake %d/%d", i + 1, len(fake_paths))

    df_iq = pd.DataFrame(real_iq + sbi_iq + fake_iq)
    df_iq.to_csv(OUT_DIR / "iq_per_frame.csv", index=False)
    log.info("[2/4] iq_per_frame.csv (%d rows) elapsed=%.1fs",
             len(df_iq), time.time() - t0)

    # Per-axis aggregate
    iq_axes = ["luma_mean", "luma_std", "sat_mean", "sat_std",
               "hue_mean", "r_mean", "g_mean", "b_mean",
               "r_std", "g_std", "b_std",
               "lap_var_full", "sobel_mean_full", "hf_ratio_full",
               "face_area", "face_area_frac",
               "lap_var_face", "sobel_mean_face", "hf_ratio_face",
               "luma_mean_face"]
    rows = []
    real_arr_by_axis = {ax: df_iq.loc[df_iq.population == "real_source", ax].dropna().values for ax in iq_axes}
    sbi_arr_by_axis = {ax: df_iq.loc[df_iq.population == "sbi_pseudofake", ax].dropna().values for ax in iq_axes}
    fake_arr_by_axis = {ax: df_iq.loc[df_iq.population == "known_fake", ax].dropna().values for ax in iq_axes}
    for ax in iq_axes:
        r = real_arr_by_axis[ax]; s = sbi_arr_by_axis[ax]; f = fake_arr_by_axis[ax]
        d_real_sbi = cohen_d(r, s)
        d_real_fake = cohen_d(r, f)
        d_sbi_fake = cohen_d(s, f)
        rows.append({
            "axis": ax,
            "n_real": len(r), "n_sbi": len(s), "n_fake": len(f),
            "mean_real": float(np.mean(r)) if len(r) else float("nan"),
            "mean_sbi":  float(np.mean(s)) if len(s) else float("nan"),
            "mean_fake": float(np.mean(f)) if len(f) else float("nan"),
            "std_real":  float(np.std(r, ddof=1)) if len(r) > 1 else 0.0,
            "std_sbi":   float(np.std(s, ddof=1)) if len(s) > 1 else 0.0,
            "std_fake":  float(np.std(f, ddof=1)) if len(f) > 1 else 0.0,
            "cohen_d_real_vs_sbi":  d_real_sbi,
            "cohen_d_real_vs_fake": d_real_fake,
            "cohen_d_sbi_vs_fake":  d_sbi_fake,
            "abs_d_real_vs_sbi":  abs(d_real_sbi)  if d_real_sbi  == d_real_sbi  else float("nan"),
            "abs_d_real_vs_fake": abs(d_real_fake) if d_real_fake == d_real_fake else float("nan"),
            "abs_d_sbi_vs_fake":  abs(d_sbi_fake)  if d_sbi_fake  == d_sbi_fake  else float("nan"),
        })
    df_iqcmp = pd.DataFrame(rows)
    df_iqcmp.to_csv(OUT_DIR / "iq_comparison.csv", index=False)
    log.info("[2/4] iq_comparison.csv written")

    # ──────── 3.  Score distributions (cached real + cached fake; SBI = blueprint) ────────
    log.info("[3/4] reading cached scores for real_source + known_fake ...")
    score_files = {
        "P8A": ("team_sanity_may5_scores_P8A.csv", "live_fakes_scores_P8A.csv"),
        "E2B": ("team_sanity_may5_scores_E2B.csv", "live_fakes_scores_E2B.csv"),
        "PA_3800": ("team_sanity_may5_scores_PA_3800.csv", "live_fakes_scores_PA_3800.csv"),
    }
    sd_rows: list[dict] = []
    real_score_idx: dict[str, dict[str, float]] = {}
    fake_score_idx: dict[str, dict[str, float]] = {}
    for ckpt, (real_csv, fake_csv) in score_files.items():
        ridx = load_score_index(SCORES_DIR / real_csv)
        fidx = load_score_index(SCORES_DIR / fake_csv)
        real_score_idx[ckpt] = ridx
        fake_score_idx[ckpt] = fidx
        log.info("[3/4]   %s: real_idx=%d  fake_idx=%d", ckpt, len(ridx), len(fidx))

        real_scores = []
        miss_r = 0
        for p in real_for_sbi:
            uri = gs_uri_from_real_local(p)
            sc = ridx.get(uri)
            if sc is None:
                miss_r += 1
            else:
                real_scores.append(sc)

        fake_scores = []
        miss_f = 0
        for p in fake_paths:
            uri = gs_uri_from_fake_local(p)
            sc = fidx.get(uri)
            if sc is None:
                miss_f += 1
            else:
                fake_scores.append(sc)

        log.info("[3/4]   %s: real cache-hits=%d/%d (miss=%d) fake cache-hits=%d/%d (miss=%d)",
                 ckpt, len(real_scores), len(real_for_sbi), miss_r,
                 len(fake_scores), len(fake_paths), miss_f)

        rs = summary_stats(np.array(real_scores))
        fs = summary_stats(np.array(fake_scores))
        sd_rows.append({"ckpt": ckpt, "population": "real_source", **rs})
        sd_rows.append({"ckpt": ckpt, "population": "known_fake", **fs})
        # SBI scores: blueprint placeholder — local CPU inference requires GS download.
        sd_rows.append({"ckpt": ckpt, "population": "sbi_pseudofake_BLUEPRINT_NOT_RUN",
                        "n": 0, "mean": float("nan"), "median": float("nan"),
                        "p10": float("nan"), "p90": float("nan"), "std": float("nan")})

    pd.DataFrame(sd_rows).to_csv(OUT_DIR / "score_distributions.csv", index=False)
    log.info("[3/4] score_distributions.csv written (SBI inference deferred — see FINDINGS.md)")

    # ──────── 4.  Verdicts ────────
    log.info("[4/4] computing verdicts ...")
    # Plausibility verdict: face-mask hit rate + visual panel saved
    landmark_rate = n_landmarks_ok / max(len(real_for_sbi), 1)
    plausibility = "GREEN" if landmark_rate >= 0.95 else ("AMBER" if landmark_rate >= 0.80 else "RED")

    # IQ-similarity verdict:
    #   pseudofakes should be SIMILAR to reals (small |d_real_vs_sbi|) and
    #   NOT especially close to fakes  (|d_sbi_vs_fake| ~ |d_real_vs_fake|).
    # GREEN  if median |d_real_vs_sbi| <= 0.30 AND mean |d_sbi_vs_fake| > median |d_real_vs_sbi|
    # AMBER  if median |d_real_vs_sbi| in (0.30, 0.60]
    # RED    otherwise
    valid_axes = df_iqcmp["abs_d_real_vs_sbi"].dropna().values
    valid_axes_fake = df_iqcmp["abs_d_real_vs_fake"].dropna().values
    valid_axes_sbi_fake = df_iqcmp["abs_d_sbi_vs_fake"].dropna().values
    med_d_real_sbi = float(np.median(valid_axes))   if len(valid_axes)   else float("nan")
    med_d_real_fake = float(np.median(valid_axes_fake)) if len(valid_axes_fake) else float("nan")
    med_d_sbi_fake = float(np.median(valid_axes_sbi_fake)) if len(valid_axes_sbi_fake) else float("nan")
    if med_d_real_sbi != med_d_real_sbi:
        iq_verdict = "RED"
    elif med_d_real_sbi <= 0.30 and med_d_sbi_fake > med_d_real_sbi:
        iq_verdict = "GREEN"
    elif med_d_real_sbi <= 0.60:
        iq_verdict = "AMBER"
    else:
        iq_verdict = "RED"

    # Score-sanity verdict — for cached real + fake we already know the gap is large;
    # the SBI score-distribution is BLUEPRINT (not yet run on local CPU). Verdict here
    # is structural: do the real and fake distributions show enough headroom for SBI to
    # be intermediate?
    p8a_real = [r for r in sd_rows if r["ckpt"] == "P8A" and r["population"] == "real_source"][0]
    p8a_fake = [r for r in sd_rows if r["ckpt"] == "P8A" and r["population"] == "known_fake"][0]
    score_headroom_p8a = float(p8a_fake["median"] - p8a_real["median"]) if (
        p8a_real["median"] == p8a_real["median"] and p8a_fake["median"] == p8a_fake["median"]
    ) else float("nan")
    if score_headroom_p8a >= 0.50:
        score_verdict = "GREEN"   # ample room for SBI to land intermediate
    elif score_headroom_p8a >= 0.20:
        score_verdict = "AMBER"
    else:
        score_verdict = "RED"

    # Top-level: GREEN only if all three are GREEN
    levels = {"GREEN": 2, "AMBER": 1, "RED": 0}
    inv_levels = {v: k for k, v in levels.items()}
    top = inv_levels[min(levels[plausibility], levels[iq_verdict], levels[score_verdict])]

    summary = {
        "job": "SBI_SMOKE_2026-05-06",
        "n_real_frames_used": int(len(real_for_sbi)),
        "n_known_fake_frames_used": int(len(fake_paths)),
        "n_sbi_pseudofakes_generated": int(len(sbi_images)),
        "n_pseudofakes_saved_to_disk": int(min(60, len(sbi_images))),
        "n_visual_panel_pairs": int(len(panels)),
        "landmark_detection": {
            "ok": int(n_landmarks_ok),
            "total": int(len(real_for_sbi)),
            "rate": float(landmark_rate),
            "fallback_mask_count": int(n_fallback),
        },
        "iq_summary": {
            "median_abs_cohen_d_real_vs_sbi":  med_d_real_sbi,
            "median_abs_cohen_d_real_vs_fake": med_d_real_fake,
            "median_abs_cohen_d_sbi_vs_fake":  med_d_sbi_fake,
            "n_axes_evaluated": int(len(valid_axes)),
        },
        "score_summary": {
            "P8A_real_median":  p8a_real["median"],
            "P8A_fake_median":  p8a_fake["median"],
            "P8A_headroom":     score_headroom_p8a,
            "sbi_scores_status": "BLUEPRINT_NOT_RUN — see FINDINGS.md §Score-collection blueprint",
        },
        "verdicts": {
            "visual_plausibility": plausibility,
            "iq_similarity": iq_verdict,
            "score_distribution": score_verdict,
        },
    }
    with (OUT_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    verdict = {
        "job": "SBI_SMOKE_2026-05-06",
        "verdict": top,
        "p2_pe_sbi_launch_readiness": top,
        "criteria": summary["verdicts"],
        "headline": (
            "GREEN: SBI recipe is plausible foundation for PE_SBI." if top == "GREEN" else
            "AMBER: SBI recipe is workable but needs parameter tuning before launch." if top == "AMBER" else
            "RED: SBI recipe is dead in current form."
        ),
    }
    with (OUT_DIR / "verdict.json").open("w") as f:
        json.dump(verdict, f, indent=2)

    mesher.close()
    detector.close()
    log.info("DONE — top-level verdict: %s", top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
