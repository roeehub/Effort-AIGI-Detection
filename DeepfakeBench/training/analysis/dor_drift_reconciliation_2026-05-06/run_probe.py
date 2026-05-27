"""Reconciliation re-computation of the three dor-drift metrics.

For job DOR_DRIFT_R2_RECONCILIATION_2026-05-06.

Reads `analysis/dor_drift_mechanism_2026-05-06/outputs/per_frame_features.csv`,
re-fits both the all-sessions ridge AND the endpoint-union ridge, and computes the
three metrics that were cited inconsistently:

  1. R² (P8A) for the all-sessions fit  -> compare with regression_p8a.json:0.142
  2. all-sessions-fit predicted/observed drift on the (low,high) endpoints
     -> compare with drift_attribution.csv:0.064/0.316 ~= 0.20
  3. union-fit predicted/observed drift; R²_union
     -> compare with thread "predicted 0.285 / total 0.316 = 90%, R²_union 0.42-0.62"
        and with supplementary_attribution_union.json

Run from repo training/ directory:
    python3 analysis/dor_drift_reconciliation_2026-05-06/run_probe.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path('/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training')
SRC = ROOT / 'analysis' / 'dor_drift_mechanism_2026-05-06' / 'outputs'
OUT = ROOT / 'analysis' / 'dor_drift_reconciliation_2026-05-06' / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)

CKPTS = ['P8A', 'E2B', 'PA_3800']

IQ_FEATURES_ALL_SESSIONS = [
    'sharpness_lap', 'luma_mean', 'luma_std', 'sat_mean', 'contrast_l',
    'edge_mag', 'hf_ratio', 'color_a_dev', 'color_b_dev',
    'highlight_frac', 'shadow_frac', 'block_score',
    'min_dim', 'face_area_ratio_imp',  # original used face_area_ratio_imp
]
IQ_FEATURES_UNION = [
    'sharpness_lap', 'luma_mean', 'luma_std', 'sat_mean', 'contrast_l',
    'edge_mag', 'hf_ratio', 'color_a_dev', 'color_b_dev',
    'highlight_frac', 'shadow_frac', 'block_score',
    'min_dim',
]

LOW_NAME = 'dor_evening / dor_evening'
HIGH_NAME = 'teams_real_dor_dev / dor_shkedi'


def fit_ridge(df: pd.DataFrame, features: list[str], y_col: str, alpha: float = 1.0):
    sc = StandardScaler()
    Xs = sc.fit_transform(df[features].values)
    y = df[y_col].values
    model = Ridge(alpha=alpha)
    model.fit(Xs, y)
    yhat = model.predict(Xs)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot
    return model, sc, r2


def attribute(model, sc, features, low_df, high_df, y_col):
    obs_drift = float(high_df[y_col].mean() - low_df[y_col].mean())
    pred = 0.0
    for i, fn in enumerate(features):
        m = sc.mean_[i]
        s = sc.scale_[i] if sc.scale_[i] > 1e-12 else 1.0
        delta_std = (high_df[fn].mean() - low_df[fn].mean()) / s
        pred += float(model.coef_[i] * delta_std)
    return obs_drift, pred


def main():
    df = pd.read_csv(SRC / 'per_frame_features.csv')
    print(f'rows in per_frame_features.csv: {len(df)}')

    # Reconstruct face_area_ratio_imp the same way run_analysis.py does:
    # numeric-coerce face_area_ratio, then fill missing with the median.
    far_num = pd.to_numeric(df['face_area_ratio'], errors='coerce')
    far_med = far_num.median()
    if pd.isna(far_med):
        far_med = 0.0
    df['face_area_ratio_imp'] = far_num.fillna(far_med)

    # Coerce numeric
    for c in [f'score_{x}' for x in CKPTS] + IQ_FEATURES_ALL_SESSIONS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=IQ_FEATURES_ALL_SESSIONS + [f'score_{c}' for c in CKPTS]).reset_index(drop=True)
    print(f'rows after numeric-coerce/dropna: {len(df)}')

    low_df = df[df['session'] == LOW_NAME].copy()
    high_df = df[df['session'] == HIGH_NAME].copy()
    union_df = pd.concat([low_df, high_df], ignore_index=True)
    print(f'low n={len(low_df)} | high n={len(high_df)} | union n={len(union_df)}')

    out: dict = {'all_sessions_fit': {}, 'union_fit': {}}

    # === All-sessions fit (matches run_analysis.py methodology) ===
    print('\n--- ALL-SESSIONS FIT (run_analysis.py: 14 features, 819 frames, 7 sessions) ---')
    for ckpt in CKPTS:
        model, sc, r2 = fit_ridge(df, IQ_FEATURES_ALL_SESSIONS, f'score_{ckpt}')
        obs, pred = attribute(model, sc, IQ_FEATURES_ALL_SESSIONS, low_df, high_df, f'score_{ckpt}')
        ratio = pred / obs if abs(obs) > 1e-9 else float('nan')
        print(f'  {ckpt}: R^2={r2:.4f} | obs_drift={obs:.4f} | pred_drift={pred:.4f} | pred/obs={ratio:.3f}')
        out['all_sessions_fit'][ckpt] = dict(r2=r2, observed_drift=obs, predicted_drift=pred, pred_over_obs=ratio)

    # === Endpoint-union fit (matches run_supplementary.py methodology) ===
    print('\n--- ENDPOINT-UNION FIT (run_supplementary.py: 13 features, low+high frames only) ---')
    for ckpt in CKPTS:
        model, sc, r2 = fit_ridge(union_df, IQ_FEATURES_UNION, f'score_{ckpt}')
        obs, pred = attribute(model, sc, IQ_FEATURES_UNION, low_df, high_df, f'score_{ckpt}')
        ratio = pred / obs if abs(obs) > 1e-9 else float('nan')
        print(f'  {ckpt}: R^2_union={r2:.4f} | obs_drift={obs:.4f} | pred_drift={pred:.4f} | pred/obs={ratio:.3f}')
        out['union_fit'][ckpt] = dict(r2_union=r2, observed_drift=obs, predicted_drift=pred, pred_over_obs=ratio)

    # === Diagnostic: also fit union with face_area_ratio_imp included to check whether the
    # 14-feature variant changes the verdict. Several lockbox/dor_evening sessions lack
    # face_area_ratio so this is a cross-check, not an alternative recipe.
    print('\n--- DIAGNOSTIC: union fit with face_area_ratio_imp included (14 features) ---')
    for ckpt in CKPTS:
        model, sc, r2 = fit_ridge(union_df, IQ_FEATURES_ALL_SESSIONS, f'score_{ckpt}')
        obs, pred = attribute(model, sc, IQ_FEATURES_ALL_SESSIONS, low_df, high_df, f'score_{ckpt}')
        ratio = pred / obs if abs(obs) > 1e-9 else float('nan')
        print(f'  {ckpt}: R^2={r2:.4f} | obs_drift={obs:.4f} | pred_drift={pred:.4f} | pred/obs={ratio:.3f}')

    # Save
    with open(OUT / 'reconciliation_recompute.json', 'w') as fp:
        json.dump(out, fp, indent=2)
    print(f"\nWrote {OUT / 'reconciliation_recompute.json'}")


if __name__ == '__main__':
    main()
