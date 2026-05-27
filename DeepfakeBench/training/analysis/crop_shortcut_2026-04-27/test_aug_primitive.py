"""Empirically test whether widening the scale aug primitive actually breaks
the per-method face_pixel_area signature on already-cropped frames.

Approach:
1. Load 5 deeplive frames + 5 teams_capture_pc_generator_s4 frames + 5 dev_real
2. Apply A.ShiftScaleRotate(scale_limit=X) for X in [0.12 (current), 0.30, 0.50]
   and A.Affine(scale=...) variants
3. For each augmented sample, run MediaPipe face detection and compute face_pixel_area
4. Compare distributions: does scale-aug widen the face_pixel_area distribution
   for each method, breaking the per-method signature?

Run from training/:
  python3 -m analysis.crop_shortcut_2026-04-27.test_aug_primitive
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from statistics import mean, median, stdev

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from PIL import Image

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
PARQUET = REPO / "analysis/lockbox_tagging/full_tags_2026-04-27.parquet"
TMP = Path("/tmp/aug_primitive_test")
TMP.mkdir(exist_ok=True)


def fetch_to_local(gcs_uri: str, idx: int) -> Path | None:
    fn = TMP / f"sample_{idx}_{Path(gcs_uri).name}"
    if not fn.exists():
        rc = os.system(f"gsutil -q cp '{gcs_uri}' '{fn}' 2>/dev/null")
        if rc != 0 or not fn.exists():
            return None
    return fn


def measure_face_area_via_mediapipe(img_path: Path) -> tuple[float | None, float | None]:
    """Returns (face_pixel_area, face_area_ratio) using MediaPipe via the
    existing layer code. None if no face detected.
    """
    try:
        from analysis.lockbox_tagging.layers.face_geometry import compute_face_geometry
        d = compute_face_geometry(img_path)
        fpa = d.get("face_pixel_area")
        far = d.get("face_area_ratio")
        return (float(fpa) if fpa else None, float(far) if far else None)
    except Exception as e:
        print(f"  mediapipe error: {e}")
        return (None, None)


def aug_pipeline(scale_limit: float) -> A.Compose:
    return A.Compose([
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=scale_limit,
            rotate_limit=7,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=1.0,
        ),
    ])


def aug_pipeline_affine(scale_min: float, scale_max: float) -> A.Compose:
    """A.Affine version — different albumentations primitive.

    A.Affine in albumentations 0.4.6 might not exist; fall back to
    A.ShiftScaleRotate with the equivalent effective scale.
    """
    if hasattr(A, "Affine"):
        return A.Compose([
            A.Affine(
                scale=(scale_min, scale_max),
                translate_percent=(-0.05, 0.05),
                rotate=(-7, 7),
                interpolation=cv2.INTER_LINEAR,
                cval=0,
                p=1.0,
            ),
        ])
    # Fallback: ShiftScaleRotate symmetric around 1.0
    sr = (scale_max - scale_min) / 2.0
    center = (scale_max + scale_min) / 2.0  # not 1.0, e.g., for (0.5, 1.5) center=1.0
    return aug_pipeline(scale_limit=sr)


def main() -> None:
    df = pd.read_parquet(PARQUET)
    fakes_dl = df[(df["method"] == "deeplive_enhanced") & (df["split"] == "dev")].head(5)
    fakes_pc = df[(df["method"] == "teams_capture_pc_generator_s4") & (df["split"] == "dev")].head(5)
    reals = df[(df["method"] == "teams_real") & (df["split"] == "dev")].head(5)

    samples = []
    for i, r in enumerate(fakes_dl.itertuples(), 1):
        samples.append(("deeplive", r.gcs_uri, i, r.face_pixel_area, r.face_area_ratio))
    for i, r in enumerate(fakes_pc.itertuples(), 11):
        samples.append(("pc_generator_s4", r.gcs_uri, i, r.face_pixel_area, r.face_area_ratio))
    for i, r in enumerate(reals.itertuples(), 21):
        samples.append(("teams_real", r.gcs_uri, i, r.face_pixel_area, r.face_area_ratio))

    # Fetch all
    print(f"[aug-test] downloading {len(samples)} sample frames ...")
    local_paths: dict[int, Path] = {}
    for method, uri, idx, _, _ in samples:
        p = fetch_to_local(uri, idx)
        if p is not None:
            local_paths[idx] = p
        else:
            print(f"  failed to fetch {uri}")

    print(f"[aug-test] fetched {len(local_paths)} / {len(samples)}")

    # Apply each aug variant N times and measure face area distribution
    aug_variants = {
        "current_0.12": aug_pipeline(0.12),
        "wider_0.30": aug_pipeline(0.30),
        "heavy_0.50": aug_pipeline(0.50),
    }
    if hasattr(A, "Affine"):
        aug_variants["affine_0.6_1.6"] = aug_pipeline_affine(0.6, 1.6)
        aug_variants["affine_0.5_2.0"] = aug_pipeline_affine(0.5, 2.0)

    N_REPS = 8

    rows = []
    for method, uri, idx, baseline_fpa, baseline_far in samples:
        if idx not in local_paths:
            continue
        pil = Image.open(local_paths[idx]).convert("RGB")
        img = np.array(pil)
        h, w = img.shape[:2]
        # Baseline is the parquet's face_pixel_area
        rows.append({
            "method": method, "idx": idx, "image_h": h, "image_w": w,
            "aug": "BASELINE", "rep": 0,
            "face_pixel_area": baseline_fpa,
            "face_area_ratio": baseline_far,
        })
        for aug_name, aug in aug_variants.items():
            for rep in range(N_REPS):
                augmented = aug(image=img)["image"]
                # Save to temp and measure with mediapipe
                aug_path = TMP / f"aug_{aug_name}_{idx}_{rep}.png"
                Image.fromarray(augmented).save(aug_path)
                fpa, far = measure_face_area_via_mediapipe(aug_path)
                rows.append({
                    "method": method, "idx": idx, "image_h": h, "image_w": w,
                    "aug": aug_name, "rep": rep,
                    "face_pixel_area": fpa,
                    "face_area_ratio": far,
                })

    out = pd.DataFrame(rows)
    out.to_csv(REPO / "analysis/crop_shortcut_2026-04-27/aug_primitive_test.csv", index=False)

    # ===== AGGREGATE =====
    print("\n=== Per-(method, aug) face_pixel_area distribution ===")
    print(f"{'method':<18} {'aug':<20} {'n':<4} {'baseline_med':<14} {'aug_med':<10} {'aug_min':<10} {'aug_max':<10} {'aug_std':<10}")
    for method in ("deeplive", "pc_generator_s4", "teams_real"):
        baseline_vals = out[(out["method"] == method) & (out["aug"] == "BASELINE")]["face_pixel_area"].dropna()
        bm = baseline_vals.median() if len(baseline_vals) > 0 else float("nan")
        for aug_name in ("BASELINE", "current_0.12", "wider_0.30", "heavy_0.50") + (("affine_0.6_1.6", "affine_0.5_2.0") if "affine_0.6_1.6" in aug_variants else ()):
            sub = out[(out["method"] == method) & (out["aug"] == aug_name)]["face_pixel_area"].dropna()
            if len(sub) == 0:
                continue
            print(f"{method:<18} {aug_name:<20} {len(sub):<4} {bm:<14.0f} {sub.median():<10.0f} {sub.min():<10.0f} {sub.max():<10.0f} {sub.std():<10.0f}")

    print("\n=== Per-(method, aug) face_area_ratio distribution ===")
    print(f"{'method':<18} {'aug':<20} {'n':<4} {'aug_med':<10} {'aug_min':<10} {'aug_max':<10} {'aug_std':<10}")
    for method in ("deeplive", "pc_generator_s4", "teams_real"):
        for aug_name in ("BASELINE", "current_0.12", "wider_0.30", "heavy_0.50") + (("affine_0.6_1.6", "affine_0.5_2.0") if "affine_0.6_1.6" in aug_variants else ()):
            sub = out[(out["method"] == method) & (out["aug"] == aug_name)]["face_area_ratio"].dropna()
            if len(sub) == 0:
                continue
            print(f"{method:<18} {aug_name:<20} {len(sub):<4} {sub.median():<10.3f} {sub.min():<10.3f} {sub.max():<10.3f} {sub.std():<10.3f}")

    # Verdict logic
    print("\n=== VERDICT ===")
    target = 0.50  # we want each method's face_pixel_area std/median ratio ≥ 0.50 after aug to call it "broken"
    for aug_name in ("current_0.12", "wider_0.30", "heavy_0.50") + (("affine_0.6_1.6", "affine_0.5_2.0") if "affine_0.6_1.6" in aug_variants else ()):
        scores = []
        for method in ("deeplive", "pc_generator_s4", "teams_real"):
            sub = out[(out["method"] == method) & (out["aug"] == aug_name)]["face_pixel_area"].dropna()
            if len(sub) >= 5 and sub.median() > 0:
                rel_spread = (sub.std() / sub.median())
                scores.append((method, rel_spread))
        if scores:
            print(f"  {aug_name}: " + ", ".join(f"{m}={s:.2f}" for m, s in scores))


if __name__ == "__main__":
    main()
