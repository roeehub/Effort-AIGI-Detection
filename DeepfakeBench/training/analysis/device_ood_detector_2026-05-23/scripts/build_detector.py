"""Device-OOD detector via frozen-CLIP LR.

Train on Mac-Roee (~498) vs Windows-Roee (~330) frozen-CLIP L11 features.
Project all team frames onto the axis to identify which in-distribution
cohorts look Mac-like (and therefore at risk of high per-frame FPR).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = Path('.')
TEAM_NPZ = ROOT / 'analysis/frozen_clip_team_identity_baseline_2026-05-23/outputs/clip_frozen_l11__team_identity_n5941.npz'
MAC_NPZ = ROOT / 'analysis/device_ood_detector_2026-05-23/outputs/clip_frozen_l11__mac_roee_n498.npz'
PER_FRAME = ROOT / 'analysis/team_identity_deploy_readout_expanded_2026-05-23/outputs/per_frame_full.csv'
OUT = ROOT / 'analysis/device_ood_detector_2026-05-23/outputs'
OUT.mkdir(parents=True, exist_ok=True)

# Load TEAM + MAC caches and combine into one path-keyed feature lookup
team_data = np.load(TEAM_NPZ, allow_pickle=True)
mac_data = np.load(MAC_NPZ, allow_pickle=True)
print(f"TEAM cache: {team_data['features'].shape}; MAC cache: {mac_data['features'].shape}")

feats = np.concatenate([team_data['features'], mac_data['features']], axis=0)
fpaths = np.concatenate([team_data['frame_paths'], mac_data['frame_paths']], axis=0)

# Build path → feature index
fp_to_idx = {fp: i for i, fp in enumerate(fpaths)}

# Per-frame metadata
df = pd.read_csv(PER_FRAME)
df['_clip_idx'] = df['frame_path'].map(fp_to_idx)
df = df.dropna(subset=['_clip_idx']).copy()
df['_clip_idx'] = df['_clip_idx'].astype(int)

# Pull device-labeled REAL frames only
windows_roee = df[(df['human'] == 'Roee_Windows') & (df['role'] == 'real')]
mac_roee = df[(df['human'] == 'Roee_Mac') & (df['role'] == 'real')]

print(f'Windows-Roee real frames: {len(windows_roee)}')
print(f'Mac-Roee real frames: {len(mac_roee)}')

X_win = feats[windows_roee['_clip_idx'].to_numpy()]
X_mac = feats[mac_roee['_clip_idx'].to_numpy()]
X = np.concatenate([X_win, X_mac], axis=0)
y = np.concatenate([np.zeros(len(X_win), dtype=int), np.ones(len(X_mac), dtype=int)])

# Train LR + 5-fold CV
lr = LogisticRegression(C=1.0, max_iter=2000, n_jobs=1, solver='lbfgs')
cv_acc = cross_val_score(lr, X, y, cv=5, scoring='accuracy', n_jobs=1)
cv_auc = cross_val_score(lr, X, y, cv=5, scoring='roc_auc', n_jobs=1)
print(f'\nDevice-OOD LR: 5-fold CV accuracy={cv_acc.mean():.4f}±{cv_acc.std():.4f}  AUC={cv_auc.mean():.4f}')

# Train on all data; save axis
lr.fit(X, y)
np.save(OUT / 'device_ood_lr_coef.npy', lr.coef_[0])

# Project ALL frames onto the axis (signed margin from LR decision function)
all_margins = lr.decision_function(feats)

# Attach margins to per_frame
df['device_ood_margin'] = all_margins[df['_clip_idx'].to_numpy()]

# Also pull per-frame P8A scores for cross-reference (already in df as prob_P8A)
# Compute per-cohort margin statistics
def cohort_stats(g):
    return pd.Series({
        'n': len(g),
        'human': g['human'].iloc[0],
        'in_scope': 'deploy' if g['deploy_relevant'].iloc[0] else 'mac_oos',
        'role': g['role'].iloc[0],
        'margin_mean': g['device_ood_margin'].mean(),
        'margin_p25': g['device_ood_margin'].quantile(0.25),
        'margin_p50': g['device_ood_margin'].quantile(0.50),
        'margin_p75': g['device_ood_margin'].quantile(0.75),
        'margin_p95': g['device_ood_margin'].quantile(0.95),
        'frac_above_0': (g['device_ood_margin'] > 0).mean(),  # frac classified as Mac
        'p8a_mean': g['prob_P8A'].mean(),
        'p8a_fpr_010': (g['prob_P8A'] >= 0.10).mean() if (g['role'] == 'real').iloc[0] else float('nan'),
        'p8a_fpr_059': (g['prob_P8A'] >= 0.59).mean() if (g['role'] == 'real').iloc[0] else float('nan'),
    })

real_df = df[df['role'] == 'real']
cohort_summary = real_df.groupby('base_identity').apply(cohort_stats).reset_index()
cohort_summary = cohort_summary.sort_values(['in_scope', 'margin_mean'], ascending=[True, False])

pd.set_option('display.float_format', '{:.3f}'.format)
pd.set_option('display.width', 200)
print('\n' + '='*120)
print('Per-cohort: device-OOD margin (positive = Mac-like) vs P8A FPR')
print('='*120)
print(cohort_summary.to_string(index=False))

cohort_summary.to_csv(OUT / 'per_cohort_device_ood.csv', index=False)

# Correlation: does device-OOD margin predict P8A per-frame FPR at τ=0.10?
real_only = real_df.copy()
real_only['flagged_010'] = (real_only['prob_P8A'] >= 0.10).astype(int)
real_only['flagged_059'] = (real_only['prob_P8A'] >= 0.59).astype(int)

# Bucket by margin
buckets = pd.qcut(real_only['device_ood_margin'], q=10, labels=False, duplicates='drop')
real_only['margin_bucket'] = buckets
bucket_stats = real_only.groupby('margin_bucket').agg(
    n=('frame_path', 'size'),
    margin_min=('device_ood_margin', 'min'),
    margin_max=('device_ood_margin', 'max'),
    p8a_fpr_010=('flagged_010', 'mean'),
    p8a_fpr_059=('flagged_059', 'mean'),
    in_dist_share=('deploy_relevant', 'mean'),
).reset_index()
print('\n' + '='*120)
print('Per-margin-decile: does Mac-like margin predict elevated P8A FPR?')
print('='*120)
print(bucket_stats.to_string(index=False))

bucket_stats.to_csv(OUT / 'margin_decile_p8a_fpr.csv', index=False)

# Final: per-cohort margin > 0 implies Mac-like; cross-check against the worst-FPR in-dist cohorts
print('\n' + '='*120)
print('Top-15 highest-margin (most Mac-like) deploy-relevant REAL cohorts')
print('='*120)
deploy_real_cohorts = cohort_summary[cohort_summary['in_scope']=='deploy']
top15 = deploy_real_cohorts.nlargest(15, 'margin_mean')[['base_identity','human','n','margin_mean','margin_p95','frac_above_0','p8a_fpr_010','p8a_fpr_059']]
print(top15.to_string(index=False))
