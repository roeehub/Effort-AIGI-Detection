"""Compute per-frame image properties on the locally-cached lockbox real crops.

Candidate shortcuts (from existing memories):
  - sharpness_laplacian  (memory: project_image_quality_shortcut)
  - mean_luma            (memory: project_image_quality_shortcut)
  - luma_std             (texture proxy)
  - lab_a_mean / lab_b_mean / lab_a_std / lab_b_std
    (memory: project_dor_drift_named_axes_2026-05-06)
  - skin_frac_hsv        (rough HSV skin-color mask fraction)
  - edge_density         (Sobel magnitude > threshold fraction)
  - face_size            (min(W, H) of crop)
  - file_size_bytes      (compression signature, partly redundant with sharpness)

Single-threaded (per memory feedback_sklearn_njobs).
"""
from __future__ import annotations
import os
import re
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
CACHE = HERE / 'cache'
OUT = HERE / 'outputs'


def extract_identity_from_path(p: str) -> str:
    """Map crop filename to identity. Filenames look like:
    - dor_shkedi__s17_3.0_frame_000058_crop_001__xyz.jpg
    - PC_Generator__s22_...jpg
    - real_dor__frame_000531_seq5188.png
    - bla_bla_chow__...
    """
    base = os.path.basename(p)
    # Strip extension
    name = re.sub(r'\.(jpg|jpeg|png)$', '', base, flags=re.I)
    # Identity is everything before first '__'
    head = name.split('__')[0]
    return head


def compute_props(img_bgr: np.ndarray, file_size: int) -> dict:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Sharpness: variance of Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(lap.var())

    # Luma stats
    luma_mean = float(gray.mean())
    luma_std = float(gray.std())

    # LAB stats (a is green-red, b is blue-yellow)
    a_mean = float(lab[..., 1].mean())
    b_mean = float(lab[..., 2].mean())
    a_std = float(lab[..., 1].std())
    b_std = float(lab[..., 2].std())
    # Deviations from neutral (a=128, b=128 in 8-bit LAB)
    a_dev = abs(a_mean - 128)
    b_dev = abs(b_mean - 128)

    # Skin frac via HSV: H in [0,25] or [160,180], S>50, V>50
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    skin_mask = (((H <= 25) | (H >= 160)) & (S > 50) & (V > 50))
    skin_frac = float(skin_mask.mean())

    # Edge density: Sobel magnitude > 50 fraction
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    edge_density = float((mag > 50).mean())

    return {
        'width': w,
        'height': h,
        'min_dim': min(w, h),
        'sharpness_laplacian': sharpness,
        'luma_mean': luma_mean,
        'luma_std': luma_std,
        'lab_a_mean': a_mean,
        'lab_b_mean': b_mean,
        'lab_a_std': a_std,
        'lab_b_std': b_std,
        'lab_a_dev': a_dev,
        'lab_b_dev': b_dev,
        'skin_frac_hsv': skin_frac,
        'edge_density': edge_density,
        'file_size_bytes': file_size,
    }


def main():
    cache_files = sorted(CACHE.glob('*'))
    print(f'Cache has {len(cache_files)} files')

    rows = []
    for i, fp in enumerate(cache_files):
        if i % 200 == 0:
            print(f'  [{i}/{len(cache_files)}] {fp.name}')
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            print(f'  WARN: unreadable {fp.name}')
            continue
        file_size = fp.stat().st_size
        props = compute_props(img, file_size)
        props['crop_basename'] = fp.name
        props['identity'] = extract_identity_from_path(fp.name)
        rows.append(props)

    df = pd.DataFrame(rows)
    out = OUT / 'frame_properties.csv'
    df.to_csv(out, index=False)
    print(f'\nWrote {out} ({len(df)} rows)')
    print('Per-identity counts:')
    print(df['identity'].value_counts())
    print('\nProperty summary:')
    cols = ['sharpness_laplacian', 'luma_mean', 'luma_std', 'lab_a_dev', 'lab_b_dev',
            'skin_frac_hsv', 'edge_density', 'min_dim', 'file_size_bytes']
    print(df[cols].describe().T[['mean', 'std', 'min', '50%', 'max']].to_string())


if __name__ == '__main__':
    main()
