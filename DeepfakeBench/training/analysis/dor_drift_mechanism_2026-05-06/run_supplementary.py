"""Supplementary analyses for drift attribution.

1. Re-fit on UNION of low+high session frames only (more apples-to-apples).
2. Per-axis univariate slope: how much score changes per 1-SD axis change, fit
   only on the low+high union.
3. Total Cohen's d ranked.
4. Pearson correlations of each axis with score (across all sampled frames).
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path('/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training')
OUT = ROOT / 'analysis' / 'dor_drift_mechanism_2026-05-06' / 'outputs'
df = pd.read_csv(OUT / 'per_frame_features.csv')
print('rows:', len(df))

CKPTS = ['P8A', 'E2B', 'PA_3800']
IQ_FEATURES = [
    'sharpness_lap', 'luma_mean', 'luma_std', 'sat_mean', 'contrast_l',
    'edge_mag', 'hf_ratio', 'color_a_dev', 'color_b_dev',
    'highlight_frac', 'shadow_frac', 'block_score',
    'min_dim',
]

low_name = 'dor_evening / dor_evening'
high_name = 'teams_real_dor_dev / dor_shkedi'
low_df = df[df['session'] == low_name].copy()
high_df = df[df['session'] == high_name].copy()
union = pd.concat([low_df, high_df], ignore_index=True)
print(f'low n={len(low_df)} high n={len(high_df)} union n={len(union)}')

for c in [f'score_{x}' for x in CKPTS] + IQ_FEATURES:
    union[c] = pd.to_numeric(union[c], errors='coerce')
union = union.dropna(subset=IQ_FEATURES + [f'score_{c}' for c in CKPTS]).reset_index(drop=True)


def fit_and_attribute(union, low_df, high_df, ckpt):
    sc = StandardScaler()
    Xs = sc.fit_transform(union[IQ_FEATURES].values)
    y = union[f'score_{ckpt}'].values
    model = Ridge(alpha=1.0)
    model.fit(Xs, y)
    yhat = model.predict(Xs)
    r2 = 1.0 - float(np.sum((y - yhat) ** 2)) / (float(np.sum((y - y.mean()) ** 2)) + 1e-12)

    obs_drift = float(high_df[f'score_{ckpt}'].mean() - low_df[f'score_{ckpt}'].mean())
    contributions = {}
    pred = 0.0
    for i, fn in enumerate(IQ_FEATURES):
        m = sc.mean_[i]
        s = sc.scale_[i] if sc.scale_[i] > 1e-12 else 1.0
        delta_std = (high_df[fn].mean() - low_df[fn].mean()) / s
        c = float(model.coef_[i] * delta_std)
        contributions[fn] = c
        pred += c
    return dict(
        r2=r2,
        observed_drift=obs_drift,
        predicted_drift=pred,
        residual=obs_drift - pred,
        contributions=contributions,
        coefficients={fn: float(model.coef_[i]) for i, fn in enumerate(IQ_FEATURES)},
    )


supp_attr = {}
for ckpt in CKPTS:
    a = fit_and_attribute(union, low_df, high_df, ckpt)
    supp_attr[ckpt] = a
    print(f'\n== {ckpt} ==  union R^2={a["r2"]:.3f}  obs={a["observed_drift"]:.3f}  pred={a["predicted_drift"]:.3f}  resid={a["residual"]:.3f}')
    s = sorted(a['contributions'].items(), key=lambda kv: abs(kv[1]), reverse=True)
    for fn, v in s:
        print(f'  {fn:<22s} contrib {v:+.4f}  ({100 * v / max(abs(a["observed_drift"]), 1e-9):+.1f}%)')

with open(OUT / 'supplementary_attribution_union.json', 'w') as fp:
    json.dump(supp_attr, fp, indent=2)


# Pearson r of each axis with each score across the FULL sampled per_frame set
print('\nPer-axis Pearson r (full sampled set, all sessions):')
df_all = df.copy()
for c in [f'score_{x}' for x in CKPTS] + IQ_FEATURES:
    df_all[c] = pd.to_numeric(df_all[c], errors='coerce')
df_all = df_all.dropna(subset=IQ_FEATURES + [f'score_{c}' for c in CKPTS])
rows = []
for fn in IQ_FEATURES:
    row = {'axis': fn}
    for ckpt in CKPTS:
        r = df_all[fn].corr(df_all[f'score_{ckpt}'])
        row[f'r_{ckpt}'] = float(r)
    rows.append(row)
pd.DataFrame(rows).to_csv(OUT / 'per_axis_pearson_r_all_sessions.csv', index=False)
print('  wrote per_axis_pearson_r_all_sessions.csv')

# Univariate within-(low,high)-union: r per axis
print('\nUnion-only Pearson r:')
rows = []
for fn in IQ_FEATURES:
    row = {'axis': fn}
    for ckpt in CKPTS:
        r = union[fn].corr(union[f'score_{ckpt}'])
        row[f'r_{ckpt}'] = float(r)
    rows.append(row)
pd.DataFrame(rows).to_csv(OUT / 'per_axis_pearson_r_union.csv', index=False)
print('  wrote per_axis_pearson_r_union.csv')

# Reverse-driver check: how much would observed_drift change if we just used a single-axis model?
print('\nSingle-axis model attribution (union-fit):')
single_rows = []
for ckpt in CKPTS:
    y = union[f'score_{ckpt}'].values
    for fn in IQ_FEATURES:
        x = union[fn].values
        s = x.std() + 1e-12
        xs = (x - x.mean()) / s
        m = Ridge(alpha=1.0)
        m.fit(xs.reshape(-1, 1), y)
        coef = float(m.coef_[0])
        delta = float((high_df[fn].mean() - low_df[fn].mean()) / s)
        contrib = coef * delta
        single_rows.append(dict(ckpt=ckpt, axis=fn, single_axis_contribution=contrib, single_axis_coef=coef))
pd.DataFrame(single_rows).to_csv(OUT / 'single_axis_attribution_union.csv', index=False)
print('  wrote single_axis_attribution_union.csv')

print('\nDone supplementary.')
