"""Roy_D substrate IQ characterization.

The parquet has zero Roy_D coverage so the F4 manifest does not have any IQ
features for these frames. This script downloads a representative sample of
Roy_D real frames + computes the same IQ proxies the parquet would carry:
  width, height, sharpness (Laplacian variance), brightness (V channel mean),
  contrast (RMS), JPEG QF estimate (PNG -> N/A), face crop region size.

Output: roy_d_iq_features.csv + summary stats vs chronic-6 medians.
"""
from __future__ import annotations
import csv
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

ROOT = '/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training'
OUT_DIR = os.path.join(ROOT, 'analysis/best_candidate_search_2026-05-13/_roy_d_probe')
os.makedirs(OUT_DIR, exist_ok=True)
CACHE_DIR = '/tmp/roy_d_iq_probe/frames'
os.makedirs(CACHE_DIR, exist_ok=True)

SAMPLE_CSV = '/tmp/roy_d_iq_probe/sample_frames.csv'


def laplacian_var(img_gray: np.ndarray) -> float:
    """Variance of Laplacian — standard sharpness proxy (no cv2 dep)."""
    # 3x3 Laplacian kernel
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    H, W = img_gray.shape
    out = np.zeros_like(img_gray, dtype=np.float32)
    g = img_gray.astype(np.float32)
    out[1:-1, 1:-1] = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1])
    return float(out.var())


def main():
    df = pd.read_csv(SAMPLE_CSV)
    rows = []
    for _, r in df.iterrows():
        gcs = r['frame_path']
        fname = gcs.rsplit('/', 1)[-1]
        local = os.path.join(CACHE_DIR, fname)
        if not os.path.exists(local):
            try:
                subprocess.run(['gsutil', '-q', 'cp', gcs, local], check=True)
            except Exception as e:
                print(f'  ERR downloading {gcs}: {e}')
                continue
        try:
            img = Image.open(local).convert('RGB')
        except Exception as e:
            print(f'  ERR opening {local}: {e}')
            continue
        W, H = img.size
        arr = np.asarray(img)
        gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
        lap_var = laplacian_var(gray)
        # HSV brightness
        from colorsys import rgb_to_hsv
        # vectorized HSV: just compute V (max channel)
        rgb_norm = arr.astype(np.float32) / 255.0
        v = rgb_norm.max(axis=-1)
        brightness_mean = float(v.mean()) * 255
        brightness_std = float(v.std()) * 255
        contrast_rms = float(gray.std())
        # Saturation S
        cmin = rgb_norm.min(axis=-1)
        cmax = rgb_norm.max(axis=-1)
        sat = np.where(cmax > 0, (cmax - cmin) / np.maximum(cmax, 1e-9), 0.0)
        sat_mean = float(sat.mean()) * 255
        min_wh = min(W, H)
        face_area = W * H  # PNG crops are already face-cropped
        rows.append({
            'video_id': r['video_id'],
            'T5C_score': r['frame_prob'],
            'width': W,
            'height': H,
            'min_wh': min_wh,
            'sharpness_laplacian': round(lap_var, 1),
            'brightness_v_mean': round(brightness_mean, 1),
            'brightness_v_std': round(brightness_std, 1),
            'contrast_rms': round(contrast_rms, 1),
            'saturation_s_mean': round(sat_mean, 1),
            'face_pixel_area_crop': face_area,
        })
        print(f'{r["video_id"]:25s} W={W} H={H} sharp={lap_var:.0f} bright={brightness_mean:.0f} sat={sat_mean:.0f}  T5C={r["frame_prob"]:.3f}')

    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, 'roy_d_iq_features.csv')
    out_df.to_csv(out_csv, index=False)

    # Summary
    if len(out_df):
        summary = {
            'n_sampled': int(len(out_df)),
            'min_wh': {'p50': float(out_df['min_wh'].median()), 'p10': float(out_df['min_wh'].quantile(0.1)), 'p90': float(out_df['min_wh'].quantile(0.9))},
            'sharpness': {'p50': float(out_df['sharpness_laplacian'].median()), 'min': float(out_df['sharpness_laplacian'].min()), 'max': float(out_df['sharpness_laplacian'].max())},
            'brightness': {'p50': float(out_df['brightness_v_mean'].median())},
            'saturation': {'p50': float(out_df['saturation_s_mean'].median())},
            'face_area': {'p50': float(out_df['face_pixel_area_crop'].median())},
        }
        import json
        with open(os.path.join(OUT_DIR, 'roy_d_iq_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print('\n=== Roy_D IQ summary ===')
        print(json.dumps(summary, indent=2))

    # Compare to chronic-6 medians (from chronic6_fingerprint per_identity_feature_medians.csv)
    chr_csv = os.path.join(ROOT, 'analysis/chronic6_fingerprint_2026-05-05/per_identity_feature_medians.csv')
    if os.path.exists(chr_csv):
        chr_df = pd.read_csv(chr_csv)
        print('\n=== Comparison: Roy_D (this probe) vs chronic-6 medians ===')
        print(f'{"identity":25s}{"min_wh":>10s}{"sharpness":>12s}{"brightness":>13s}{"saturation":>13s}{"face_area":>13s}')
        for _, row in chr_df.iterrows():
            print(f'{row["identity"][:23]:25s}{row.get("width_p50",0):10.0f}{row.get("sharpness_laplacian_p50",0):12.0f}'
                  f'{row.get("brightness_v_mean_p50",0):13.0f}{row.get("saturation_s_mean_p50",0):13.0f}{row.get("face_pixel_area_p50",0):13.0f}')
        if len(out_df):
            print(f'{"Roy_D (this probe)":25s}{out_df["min_wh"].median():10.0f}{out_df["sharpness_laplacian"].median():12.0f}'
                  f'{out_df["brightness_v_mean"].median():13.0f}{out_df["saturation_s_mean"].median():13.0f}{out_df["face_pixel_area_crop"].median():13.0f}')


if __name__ == '__main__':
    main()
