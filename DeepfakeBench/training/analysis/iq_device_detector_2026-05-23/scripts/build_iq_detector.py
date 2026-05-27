"""Build IQ-based device-OOD detector and test cross-person generalization.

The CLIP-based detector (analysis/device_ood_detector_2026-05-23/) overfit to
"Roee on his specific Windows laptop." This script tests whether explicit IQ
features (sharpness, brightness, color cast, jpeg_qf, ...) generalize as a
device discriminator.

Critical test: train Mac-Roee vs Windows-Roee on IQ features, then project
all team-humans onto the axis. If non-Roee in-distribution cohorts (Xinhe,
Xiang, Noyn, dor on Windows captures) end up on the "Windows" side, the
detector generalizes. If they end up "Mac-like" like the CLIP detector did,
IQ alone is also identity-confounded.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.stats import spearmanr

ROOT = Path('.')
SRC = ROOT / "analysis/iq_device_detector_2026-05-23/outputs/per_frame_iq_v2.parquet"

df = pd.read_parquet(SRC)
print(f"IQ rows: {len(df)}")
print(f"per-human: {df.groupby('human').size().to_dict()}")

META = {'frame_path','human','base_identity','deploy_relevant','prob_P8A'}
# Drop jpeg_qf — it's None for non-JPEG frames (mostly PNGs here)
SKIP = META | {'jpeg_qf'}
iq_cols = [c for c in df.columns if c not in SKIP and df[c].dtype != object]
print(f"IQ feature cols ({len(iq_cols)}): {iq_cols}")

# Drop rows with NaN in any remaining IQ feature
df_clean = df.dropna(subset=iq_cols).copy()
print(f"after NaN drop: {len(df_clean)} rows")
print(f"  per-human after NaN drop: {df_clean.groupby('human').size().to_dict()}")

# Build train set: Roee-Mac vs Roee-Windows
mac_mask = (df_clean['human']=='Roee_Mac')
win_mask = (df_clean['human']=='Roee_Windows')
X_mac = df_clean[mac_mask][iq_cols].to_numpy(dtype=float)
X_win = df_clean[win_mask][iq_cols].to_numpy(dtype=float)
print(f"\nRoee-Mac: {len(X_mac)} rows; Roee-Windows: {len(X_win)} rows")

if len(X_mac) == 0 or len(X_win) == 0:
    print("INSUFFICIENT TRAINING DATA — aborting")
    sys.exit(1)

X = np.concatenate([X_win, X_mac], axis=0)
y = np.concatenate([np.zeros(len(X_win), dtype=int), np.ones(len(X_mac), dtype=int)])

# IQ-LR with scaling
pipe = Pipeline([('scale', StandardScaler()), ('lr', LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver='lbfgs'))])
cv_acc = cross_val_score(pipe, X, y, cv=5, scoring='accuracy', n_jobs=1)
cv_auc = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc', n_jobs=1)
print(f"\nIQ-LR Mac-vs-Windows (Roee): CV accuracy={cv_acc.mean():.4f}±{cv_acc.std():.4f}, AUC={cv_auc.mean():.4f}")

# Fit + project all
pipe.fit(X, y)
all_iq = df_clean[iq_cols].to_numpy(dtype=float)
margins = pipe.decision_function(all_iq)
df_clean = df_clean.assign(iq_margin=margins)

# Per-cohort margin + correlation with P8A FPR
print('\n' + '='*120)
print('Per-cohort IQ-detector margins (positive = Mac-like)')
print('='*120)
summary = df_clean.groupby(['human','base_identity']).agg(
    n=('iq_margin','size'),
    margin_mean=('iq_margin','mean'),
    margin_p50=('iq_margin','median'),
    margin_p95=('iq_margin', lambda s: s.quantile(0.95)),
    frac_above_0=('iq_margin', lambda s: (s>0).mean()),
    p8a_fpr_010=('prob_P8A', lambda s: (s>=0.10).mean()),
    p8a_fpr_059=('prob_P8A', lambda s: (s>=0.59).mean()),
    deploy_relevant=('deploy_relevant','first'),
).reset_index().sort_values('margin_mean', ascending=False)
pd.set_option('display.float_format','{:.3f}'.format)
pd.set_option('display.width', 200)
print(summary.to_string(index=False))
summary.to_csv('analysis/iq_device_detector_2026-05-23/outputs/per_cohort_iq_margin.csv', index=False)

# ---- Generalization test: how is each non-Roee in-dist cohort classified? ----
print('\n' + '='*120)
print('GENERALIZATION TEST: non-Roee in-dist cohorts (should be classified "Windows-like" / margin<0)')
print('='*120)
non_roee = summary[(summary['human'].isin(['Xinhe','Xiang','Noyn','dor'])) & (summary['deploy_relevant']==True)]
print(non_roee.to_string(index=False))
n_total = len(non_roee)
n_windows_like = (non_roee['margin_mean'] < 0).sum()
print(f"\n{n_windows_like} of {n_total} non-Roee in-dist cohorts classified as Windows-like (margin<0)")
print(f"  → {'GENERALIZES' if n_windows_like / max(n_total,1) > 0.6 else 'OVERFITS / IDENTITY-CONFOUNDED'}")

# Compare to CLIP detector result
print('\n' + '='*120)
print('Compare to CLIP detector (most non-Roee cohorts were classified Mac-like = identity-confounded)')
print('='*120)
clip_summary = pd.read_csv('analysis/device_ood_detector_2026-05-23/outputs/per_cohort_device_ood.csv')
clip_summary['detector'] = 'CLIP'
iq_renamed = non_roee.rename(columns={'margin_mean':'iq_margin_mean'})[['base_identity','human','iq_margin_mean']]
clip_subset = clip_summary[(clip_summary['human'].isin(['Xinhe','Xiang','Noyn','dor'])) & (clip_summary['in_scope']=='deploy')][['base_identity','human','margin_mean']].rename(columns={'margin_mean':'clip_margin_mean'})
comparison = pd.merge(iq_renamed, clip_subset, on=['base_identity','human'], how='outer')
comparison['both_windows'] = (comparison['iq_margin_mean'] < 0) & (comparison['clip_margin_mean'] < 0)
comparison['iq_better'] = (comparison['iq_margin_mean'] < 0) & (comparison['clip_margin_mean'] > 0)
comparison['clip_better'] = (comparison['iq_margin_mean'] > 0) & (comparison['clip_margin_mean'] < 0)
print(comparison.to_string(index=False))

# Per-frame correlation: IQ-margin → P8A score
print('\n' + '='*120)
print('Per-frame IQ-margin → P8A score correlation (Spearman)')
print('='*120)
rho, p = spearmanr(df_clean['iq_margin'], df_clean['prob_P8A'])
print(f'all real frames: rho={rho:.4f}, p={p:.2e}, n={len(df_clean)}')

in_dist = df_clean[df_clean['deploy_relevant']==True]
rho_in, p_in = spearmanr(in_dist['iq_margin'], in_dist['prob_P8A'])
print(f'in-dist only: rho={rho_in:.4f}, p={p_in:.2e}, n={len(in_dist)}')

# Per-margin-decile P8A FPR
print('\n' + '='*120)
print('Per-margin-decile P8A FPR — does Mac-like (margin>0) predict elevated FPR?')
print('='*120)
deciles = pd.qcut(df_clean['iq_margin'], q=10, labels=False, duplicates='drop')
df_clean['decile'] = deciles
dec_stats = df_clean.groupby('decile').agg(
    n=('iq_margin','size'),
    margin_min=('iq_margin','min'),
    margin_max=('iq_margin','max'),
    p8a_fpr_010=('prob_P8A', lambda s: (s>=0.10).mean()),
    p8a_fpr_059=('prob_P8A', lambda s: (s>=0.59).mean()),
    in_dist_share=('deploy_relevant', 'mean'),
    n_humans=('human', 'nunique'),
).reset_index()
print(dec_stats.to_string(index=False))

# LR coefficient: which features are most loading-bearing?
print('\n' + '='*120)
print('LR feature importance (top features by abs coefficient)')
print('='*120)
lr_coefs = pipe.named_steps['lr'].coef_[0]
feat_imp = pd.DataFrame({'feature': iq_cols, 'coef': lr_coefs, 'abs_coef': np.abs(lr_coefs)}).sort_values('abs_coef', ascending=False)
print(feat_imp.to_string(index=False))
feat_imp.to_csv('analysis/iq_device_detector_2026-05-23/outputs/feature_importance.csv', index=False)
