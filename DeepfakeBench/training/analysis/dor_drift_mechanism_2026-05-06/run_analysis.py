"""Mechanism audit for Dor real-side score drift across recording conditions.

For each recording session of REAL Dor frames, compute IQ axes per frame and fit
a ridge regression score ~ IQ. Then decompose the score drift between
dor_evening (low score) and teams_real_dor_dev (high score) along each IQ axis.

Run from training/ directory (cwd):
    cd /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training
    python3 analysis/dor_drift_mechanism_2026-05-06/run_analysis.py
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2  # type: ignore
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path('/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training')
OUT = ROOT / 'analysis' / 'dor_drift_mechanism_2026-05-06' / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)

MANIFEST = ROOT / 'analysis' / 'identity_browser_2026-05-05' / 'data' / 'grouped_manifest_v2.csv'
PARQUET = ROOT / 'analysis' / 'lockbox_tagging' / 'full_tags_2026-04-27.parquet'
FRAMES_ROOT = ROOT / 'analysis' / 'identity_browser_2026-05-05' / 'frames'

CKPTS = ['P8A', 'E2B', 'PA_3800']


def find_local(suite: str, base_identity: str, frame_path: str) -> str | None:
    bn = os.path.basename(frame_path)
    cand = FRAMES_ROOT / base_identity / f"{suite}__{bn}"
    if cand.exists():
        return str(cand)
    cand2 = FRAMES_ROOT / base_identity / bn
    if cand2.exists():
        return str(cand2)
    return None


def load_dor_reals() -> pd.DataFrame:
    df = pd.read_csv(MANIFEST)
    patterns = ['dor_evening', 'dor_morning', 'dor_shkedi', 'real_dor', 'team_may5__Dor']
    mask_id = df['base_identity'].astype(str).str.contains('|'.join(patterns), na=False, case=False)
    mask_suite = df['suite'] == 'teams_real_dor_dev'
    mask_label = df['label'] == 0
    dor = df[(mask_id | mask_suite) & mask_label].copy()
    dor['local_path'] = dor.apply(lambda r: find_local(r['suite'], r['base_identity'], r['frame_path']), axis=1)
    dor = dor.dropna(subset=['local_path']).reset_index(drop=True)
    dor['session'] = dor['suite'] + ' / ' + dor['base_identity'].astype(str)
    return dor


def compute_iq_axes(local_path: str) -> dict | None:
    """Compute pixel-domain IQ axes for a single frame.

    Returns None on failure. Uses cv2 only; n_jobs=2-3 multiprocessing-safe.
    """
    try:
        img = cv2.imread(local_path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Sharpness: variance of laplacian on gray
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        sharpness_lap = float(lap.var())

        # Luma stats (V channel of HSV used as proxy for luminance)
        v = hsv[..., 2]
        luma_mean = float(v.mean())
        luma_std = float(v.std())

        # Saturation
        sat_mean = float(hsv[..., 1].mean())

        # Contrast (L channel of LAB std)
        contrast_l = float(lab[..., 0].std())

        # Edge magnitude: Sobel
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = float(np.sqrt(gx * gx + gy * gy).mean())

        # HF energy ratio: high-frequency energy / total energy via FFT magnitude
        # Use a fairly small image window (resize for speed)
        sm = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        f = np.fft.fft2(sm)
        f_shift = np.fft.fftshift(f)
        mag = np.abs(f_shift)
        cy, cx = mag.shape[0] // 2, mag.shape[1] // 2
        # mask center 16x16 = low frequency; remaining = high
        Y, X = np.ogrid[:mag.shape[0], :mag.shape[1]]
        rad = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        lf_mask = rad <= 16
        total = float(mag.sum() + 1e-9)
        lf = float(mag[lf_mask].sum())
        hf = total - lf
        hf_ratio = hf / total

        # Color cast: mean of A and B channels in LAB (deviation from neutral 128)
        a_dev = float(np.abs(lab[..., 1] - 128).mean())
        b_dev = float(np.abs(lab[..., 2] - 128).mean())

        # Highlights / shadow fractions
        highlight_frac = float((v >= 240).mean())
        shadow_frac = float((v <= 16).mean())

        # JPEG-blockiness proxy: difference between block boundaries (8x8) and mid-block
        # Compute mean abs diff at 8-pixel boundaries vs interior; approximate
        # Subtract abs(diff at column 8k) - abs(diff at column 8k+4)
        gx_abs = np.abs(np.diff(gray, axis=1))
        # boundary cols: 7,15,23,...
        bnd_cols = np.arange(7, gx_abs.shape[1], 8)
        mid_cols = np.arange(3, gx_abs.shape[1], 8)
        try:
            block_score = float(gx_abs[:, bnd_cols].mean() - gx_abs[:, mid_cols].mean())
        except Exception:
            block_score = 0.0

        return dict(
            local_path=local_path,
            width=w,
            height=h,
            min_dim=min(w, h),
            sharpness_lap=sharpness_lap,
            luma_mean=luma_mean,
            luma_std=luma_std,
            sat_mean=sat_mean,
            contrast_l=contrast_l,
            edge_mag=edge_mag,
            hf_ratio=hf_ratio,
            color_a_dev=a_dev,
            color_b_dev=b_dev,
            highlight_frac=highlight_frac,
            shadow_frac=shadow_frac,
            block_score=block_score,
        )
    except Exception as e:
        return None


def compute_iq_parallel(paths: list[str], n_jobs: int = 2) -> pd.DataFrame:
    rows = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for f in as_completed([ex.submit(compute_iq_axes, p) for p in paths]):
            r = f.result()
            if r is not None:
                rows.append(r)
    return pd.DataFrame(rows)


def per_session_summary(dor: pd.DataFrame) -> pd.DataFrame:
    df = dor.copy()
    # face_area_ratio + face_size + is_low_quality may be strings/NaN; coerce
    for col in ['face_area_ratio', 'face_size', 'is_low_quality']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    g = df.groupby('session')
    out = g.agg(
        n_frames=('frame_path', 'count'),
        suite=('suite', 'first'),
        base_identity=('base_identity', 'first'),
        mean_score_P8A=('score_P8A', 'mean'),
        mean_score_E2B=('score_E2B', 'mean'),
        mean_score_PA_3800=('score_PA_3800', 'mean'),
        std_score_P8A=('score_P8A', 'std'),
        mean_face_area_ratio=('face_area_ratio', 'mean'),
        low_quality_frac=('is_low_quality', 'mean'),
        mean_face_size=('face_size', 'mean'),
    ).reset_index()
    return out


def fit_ridge(X: np.ndarray, y: np.ndarray, feature_names: list[str], alpha: float = 1.0):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    model = Ridge(alpha=alpha)
    model.fit(Xs, y)
    yhat = model.predict(Xs)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot
    coefs = {fn: float(c) for fn, c in zip(feature_names, model.coef_)}
    return dict(
        r2=r2,
        intercept=float(model.intercept_),
        coefficients=coefs,
        scaler_mean={fn: float(m) for fn, m in zip(feature_names, sc.mean_)},
        scaler_scale={fn: float(s) for fn, s in zip(feature_names, sc.scale_)},
    ), model, sc


def attribute_drift(model_info: dict, low_session: pd.DataFrame, high_session: pd.DataFrame, feature_names: list[str], y_col: str) -> dict:
    """Decompose score drift using standardized regression."""
    coefs = model_info['coefficients']
    sc_mean = model_info['scaler_mean']
    sc_scale = model_info['scaler_scale']
    obs_low = float(low_session[y_col].mean())
    obs_high = float(high_session[y_col].mean())
    obs_drift = obs_high - obs_low

    # Predicted drift via standardized features:
    # contribution_axis = coef_standardized * (mean_high_std - mean_low_std)
    contributions = {}
    pred_drift = 0.0
    for fn in feature_names:
        m = sc_mean[fn]
        s = sc_scale[fn] if sc_scale[fn] > 1e-12 else 1.0
        low_std = (low_session[fn].mean() - m) / s
        high_std = (high_session[fn].mean() - m) / s
        contrib = coefs[fn] * (high_std - low_std)
        contributions[fn] = contrib
        pred_drift += contrib

    residual = obs_drift - pred_drift
    return dict(
        observed_drift=obs_drift,
        predicted_drift=pred_drift,
        residual=residual,
        observed_low=obs_low,
        observed_high=obs_high,
        contributions=contributions,
    )


def main(sample_per_session: int = 200):
    print('Loading dor reals manifest...')
    dor = load_dor_reals()
    print(f' {len(dor)} dor real rows after local-path resolution')
    print(' By session:')
    print(dor.groupby('session').size())

    # Per-session summary
    pss = per_session_summary(dor)
    pss.to_csv(OUT / 'per_session_summary.csv', index=False)
    print(' wrote per_session_summary.csv')

    # Sample frames: cap at sample_per_session per session to bound cost
    sampled = []
    rng = np.random.default_rng(42)
    for s, g in dor.groupby('session'):
        if len(g) > sample_per_session:
            idx = rng.choice(len(g), size=sample_per_session, replace=False)
            sampled.append(g.iloc[idx])
        else:
            sampled.append(g)
    dor_s = pd.concat(sampled, ignore_index=True)
    print(f' sampled to {len(dor_s)} frames for IQ feature extraction')

    # Compute IQ axes
    print(' computing IQ axes (n_jobs=2)...')
    iq_df = compute_iq_parallel(dor_s['local_path'].tolist(), n_jobs=2)
    print(f' computed IQ for {len(iq_df)} frames')

    # Merge by local_path
    merged = dor_s.merge(iq_df, on='local_path', how='inner')
    print(f' merged: {len(merged)} rows')

    # Try to enrich with parquet sharpness for lockbox sessions
    try:
        pq = pd.read_parquet(PARQUET)
        # Keys we can match on are frame_path-style basenames; parquet has gcs_uri
        merged_basename = merged['frame_path'].apply(lambda p: os.path.basename(str(p)))
        pq_basename = pq['gcs_uri'].apply(lambda p: os.path.basename(str(p)))
        pq_small = pq[['gcs_uri', 'sharpness_laplacian']].copy()
        pq_small['_bn'] = pq_basename
        merged['_bn'] = merged_basename
        merged = merged.merge(pq_small[['_bn', 'sharpness_laplacian']], on='_bn', how='left', suffixes=('', '_pq'))
        n_pq = merged['sharpness_laplacian'].notna().sum()
        print(f' parquet sharpness joined for {n_pq}/{len(merged)} rows')
    except Exception as e:
        print(f' parquet enrichment skipped: {e}')

    # Save per-frame features
    merged.to_csv(OUT / 'per_frame_features.csv', index=False)
    print(' wrote per_frame_features.csv')

    # Feature list for regression
    iq_features = [
        'sharpness_lap', 'luma_mean', 'luma_std', 'sat_mean', 'contrast_l',
        'edge_mag', 'hf_ratio', 'color_a_dev', 'color_b_dev',
        'highlight_frac', 'shadow_frac', 'block_score',
        'min_dim',
    ]
    # Include face_area_ratio when available (not for all sessions). Coerce first.
    far_num = pd.to_numeric(merged['face_area_ratio'], errors='coerce')
    far_med = far_num.median()
    if pd.isna(far_med):
        far_med = 0.0
    merged['face_area_ratio_imp'] = far_num.fillna(far_med)
    iq_features.append('face_area_ratio_imp')

    # Coerce score and feature cols to numeric
    for c in CKPTS:
        merged[f'score_{c}'] = pd.to_numeric(merged[f'score_{c}'], errors='coerce')
    for c in iq_features:
        merged[c] = pd.to_numeric(merged[c], errors='coerce')

    # Drop rows missing any feature
    fit_df = merged.dropna(subset=iq_features + [f'score_{c}' for c in CKPTS]).reset_index(drop=True)
    print(f' fitting on {len(fit_df)} rows, {len(iq_features)} features')

    X = fit_df[iq_features].values
    results = {}
    for ckpt in CKPTS:
        y = fit_df[f'score_{ckpt}'].values
        info, model, sc = fit_ridge(X, y, iq_features, alpha=1.0)
        results[ckpt] = info
        with open(OUT / f'regression_{ckpt.lower()}.json', 'w') as fp:
            json.dump(info, fp, indent=2)
        print(f' {ckpt}: R^2={info["r2"]:.3f}')

    # Drift attribution: dor_evening (low) vs teams_real_dor_dev (high)
    low_name = 'dor_evening / dor_evening'
    high_name = 'teams_real_dor_dev / dor_shkedi'
    low_df = fit_df[fit_df['session'] == low_name]
    high_df = fit_df[fit_df['session'] == high_name]
    print(f' attribution: low={low_name} (n={len(low_df)}), high={high_name} (n={len(high_df)})')

    rows = []
    attribution_summary = {}
    for ckpt in CKPTS:
        a = attribute_drift(results[ckpt], low_df, high_df, iq_features, f'score_{ckpt}')
        attribution_summary[ckpt] = a
        for fn, contrib in a['contributions'].items():
            rows.append(dict(
                ckpt=ckpt,
                axis=fn,
                contribution=contrib,
                observed_drift=a['observed_drift'],
                predicted_drift=a['predicted_drift'],
                residual=a['residual'],
            ))
    pivot = pd.DataFrame(rows).pivot(index='axis', columns='ckpt', values='contribution')
    # Add an observed/predicted/residual rows
    extra_rows = []
    for ckpt in CKPTS:
        a = attribution_summary[ckpt]
        extra_rows.append(dict(axis='__OBSERVED_DRIFT__', ckpt=ckpt, contribution=a['observed_drift']))
        extra_rows.append(dict(axis='__PREDICTED_DRIFT__', ckpt=ckpt, contribution=a['predicted_drift']))
        extra_rows.append(dict(axis='__RESIDUAL_NAMED__', ckpt=ckpt, contribution=a['residual']))
        extra_rows.append(dict(axis='__OBSERVED_LOW_MEAN__', ckpt=ckpt, contribution=a['observed_low']))
        extra_rows.append(dict(axis='__OBSERVED_HIGH_MEAN__', ckpt=ckpt, contribution=a['observed_high']))
    extra_pivot = pd.DataFrame(extra_rows).pivot(index='axis', columns='ckpt', values='contribution')
    full = pd.concat([pivot, extra_pivot])
    full.to_csv(OUT / 'drift_attribution.csv')
    print(' wrote drift_attribution.csv')

    with open(OUT / 'attribution_summary.json', 'w') as fp:
        json.dump(attribution_summary, fp, indent=2)

    # Per-axis distribution overlap stats
    overlap_rows = []
    for fn in iq_features:
        l = low_df[fn].values
        h = high_df[fn].values
        if len(l) == 0 or len(h) == 0:
            continue
        overlap_rows.append(dict(
            axis=fn,
            mean_low=float(np.mean(l)),
            mean_high=float(np.mean(h)),
            std_low=float(np.std(l)),
            std_high=float(np.std(h)),
            cohen_d=float((np.mean(h) - np.mean(l)) / (np.sqrt((np.var(l) + np.var(h)) / 2) + 1e-9)),
        ))
    pd.DataFrame(overlap_rows).to_csv(OUT / 'iq_axis_distribution_overlap.csv', index=False)
    print(' wrote iq_axis_distribution_overlap.csv')

    print()
    print('Done. Now run write_findings.py for FINDINGS.md')


if __name__ == '__main__':
    main()
