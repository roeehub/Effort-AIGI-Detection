"""
FP-tail characterization for P8A / E2B_3200 / E3_6600 across real-class suites.

Outputs:
- fp_cohort_per_ckpt.csv      : per (ckpt, suite, frame) with score + axes + is_FP
- concentration_per_ckpt.csv  : per (ckpt, suite, axis, axis_value) FP-rate / counts
- shared_fp_tail.csv          : frame-level on teams_real_all_dev with all 3 ckpts'
                                 scores and n_ckpts_FP
- FINDINGS.md                 : verdict
"""

import os
import re
import sys
import numpy as np
import pandas as pd

ROOT = '/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training'
RAW  = os.path.join(ROOT, 'analysis/cpu_followups_2026-05-04/raw_reports')
OUT  = os.path.join(ROOT, 'analysis/fp_tail_characterization_2026-05-04')
os.makedirs(OUT, exist_ok=True)

CKPTS = {
    'P8A':  'p8a_reference_step5000',
    'E2B':  'e2b_top_n_step3200',
    'E3':   'e3_top_n_step6600',
}
SUITES = [
    'teams_real_all_dev',
    'teams_real_all_lockbox',
    'teams_real_dor_dev',
    'teams_real_poor_quality_dev',
    'teams_real_lighting_extreme_dev',
]
CALIB_SUITE = 'teams_real_all_dev'  # FPR=10% τ calibrated on this per-ckpt
TARGET_FPR  = 0.10


# ---------- helpers ----------

_IDENT_STRIP = re.compile(
    r'(__seq\d+|__seg_[\d.]+|__s\d+|_frame_\d+|_seq\d+|__real|__fake|_real|_fake|_crop_\d+|__[0-9a-f]{6,})',
    flags=re.IGNORECASE,
)

def extract_identity_from_video_id(vid):
    if pd.isna(vid):
        return 'UNK'
    s = str(vid)
    # Iteratively strip the suffix patterns
    prev = None
    while prev != s:
        prev = s
        s = _IDENT_STRIP.sub('', s)
    return s.strip('_').lower() or 'UNK'


def quartile_label(series, n=4):
    """Return labelled quartile for each value (Q1=lowest)."""
    s = pd.to_numeric(series, errors='coerce')
    try:
        return pd.qcut(s, q=n, labels=[f'Q{i+1}' for i in range(n)], duplicates='drop')
    except Exception:
        return pd.Series(['Qx'] * len(s), index=s.index)


def fpr_threshold(scores, target_fpr=0.10):
    """τ at which fraction(scores > τ) == target_fpr (for a real-only set).
       Uses the (1-target_fpr) quantile."""
    s = pd.to_numeric(scores, errors='coerce').dropna()
    if len(s) == 0:
        return np.nan
    return float(np.quantile(s, 1.0 - target_fpr))


# ---------- load metadata parquet ----------

print('Loading parquet metadata...')
pq = pd.read_parquet(os.path.join(ROOT, 'analysis/lockbox_tagging/full_tags_2026-04-27.parquet'))

# Restrict columns we'll use
META_COLS = [
    'gcs_uri', 'identity_key', 'session_id', 'video_id',
    'sharpness_laplacian', 'face_pixel_area', 'width', 'height',
    'clip_capture_mode', 'clip_lighting', 'clip_quality',
    'is_low_quality', 'is_pose_extreme', 'is_no_face',
]
pq_meta = pq[META_COLS].copy()
pq_meta = pq_meta.rename(columns={'video_id': 'meta_video_id', 'session_id': 'meta_session_id'})
print(f'parquet rows: {len(pq_meta):,}')


# ---------- per-suite × per-ckpt assembly ----------

per_frame_rows = []
concentration_rows = []

# Compute calibration τ for each ckpt on teams_real_all_dev
calib_tau = {}
for ck_label, ck_id in CKPTS.items():
    fp_calib = os.path.join(RAW, f'{CALIB_SUITE}_{ck_id}_frames_report.csv')
    df_c = pd.read_csv(fp_calib)
    tau = fpr_threshold(df_c['frame_prob'], TARGET_FPR)
    calib_tau[ck_label] = tau
    print(f'  τ@FPR={TARGET_FPR:.0%} for {ck_label} on {CALIB_SUITE}: {tau:.4f}')


def add_axes_columns(df, suite_name):
    """Join parquet meta and derive axis columns. Quartiles defined per-suite on the
       reals (i.e. the input frame set, since label==0 here)."""
    merged = df.merge(
        pq_meta, how='left', left_on='frame_path', right_on='gcs_uri',
    )
    merged['identity_video_id'] = merged['video_id'].apply(extract_identity_from_video_id)
    merged['identity'] = merged['identity_key'].fillna(merged['identity_video_id'])

    # face_pixel_area quartiles (per suite, on the available subset)
    merged['face_area_q'] = quartile_label(merged['face_pixel_area'])
    merged['sharpness_q'] = quartile_label(merged['sharpness_laplacian'])

    minWH = merged[['width', 'height']].min(axis=1)
    merged['min_wh'] = minWH
    merged['source_resolution_bucket'] = np.where(
        minWH.isna(), 'unknown',
        np.where(minWH < 200, 'low_lt200', 'hi_ge200')
    )
    merged['suite'] = suite_name
    return merged


print('\nProcessing per-suite × per-ckpt frame tables...')
all_frame_dfs = {}  # (ckpt, suite) -> df with is_FP flag
for ck_label, ck_id in CKPTS.items():
    tau = calib_tau[ck_label]
    for suite in SUITES:
        fp_csv = os.path.join(RAW, f'{suite}_{ck_id}_frames_report.csv')
        if not os.path.exists(fp_csv):
            print(f'  MISSING: {fp_csv}')
            continue
        df = pd.read_csv(fp_csv)
        df = add_axes_columns(df, suite)
        df['ckpt'] = ck_label
        df['tau_used'] = tau
        df['is_FP'] = (df['frame_prob'] > tau).astype(int)

        per_frame_rows.append(df[[
            'ckpt', 'suite', 'video_id', 'frame_path', 'frame_prob', 'tau_used',
            'identity', 'clip_capture_mode', 'face_pixel_area', 'face_area_q',
            'sharpness_laplacian', 'sharpness_q', 'min_wh', 'source_resolution_bucket',
            'clip_lighting', 'clip_quality', 'is_low_quality', 'is_pose_extreme',
            'is_FP',
        ]])
        all_frame_dfs[(ck_label, suite)] = df

        # concentration metrics
        n_total = len(df)
        n_fp = int(df['is_FP'].sum())

        def _add_axis(axis_name, axis_col, restrict_to_meta=False):
            sub = df.copy()
            if restrict_to_meta:
                sub = sub[sub['gcs_uri'].notna()]
            if len(sub) == 0:
                return
            grp = sub.groupby(axis_col, dropna=False).agg(
                n_total=('frame_prob', 'size'),
                n_fp=('is_FP', 'sum'),
            ).reset_index().rename(columns={axis_col: 'axis_value'})
            grp['fp_rate'] = grp['n_fp'] / grp['n_total'].clip(lower=1)
            denom = max(1, int(grp['n_fp'].sum()))
            grp['share_of_fp_fires'] = grp['n_fp'] / denom
            grp.insert(0, 'axis', axis_name)
            grp.insert(0, 'suite', suite)
            grp.insert(0, 'ckpt', ck_label)
            concentration_rows.append(grp)

        _add_axis('identity',                 'identity')
        _add_axis('clip_capture_mode',        'clip_capture_mode',        restrict_to_meta=True)
        _add_axis('face_area_quartile',       'face_area_q',              restrict_to_meta=True)
        _add_axis('sharpness_quartile',       'sharpness_q',              restrict_to_meta=True)
        _add_axis('source_resolution_bucket', 'source_resolution_bucket', restrict_to_meta=True)
        _add_axis('clip_lighting',            'clip_lighting',            restrict_to_meta=True)
        _add_axis('clip_quality',             'clip_quality',             restrict_to_meta=True)
        _add_axis('is_low_quality',           'is_low_quality',           restrict_to_meta=True)

        print(f'  {ck_label:4s} | {suite:35s}: n={n_total:5d}  FP={n_fp:5d} ({100*n_fp/max(1,n_total):4.1f}%)  τ={tau:.3f}')


print('\nWriting per-frame cohort csv...')
fp_cohort = pd.concat(per_frame_rows, ignore_index=True)
fp_cohort.to_csv(os.path.join(OUT, 'fp_cohort_per_ckpt.csv'), index=False)
print(f'  fp_cohort_per_ckpt.csv: {len(fp_cohort):,} rows')

print('Writing concentration csv...')
concentration = pd.concat(concentration_rows, ignore_index=True)
concentration = concentration[['ckpt', 'suite', 'axis', 'axis_value',
                               'n_total', 'n_fp', 'fp_rate', 'share_of_fp_fires']]
concentration.to_csv(os.path.join(OUT, 'concentration_per_ckpt.csv'), index=False)
print(f'  concentration_per_ckpt.csv: {len(concentration):,} rows')


# ---------- shared FP-tail on teams_real_all_dev ----------

print('\nBuilding shared FP-tail on teams_real_all_dev...')
shared_base = None
for ck_label in CKPTS.keys():
    df = all_frame_dfs[(ck_label, CALIB_SUITE)][[
        'frame_path', 'video_id', 'identity', 'clip_capture_mode',
        'face_area_q', 'sharpness_q', 'source_resolution_bucket',
        'frame_prob', 'is_FP',
    ]].copy()
    df = df.rename(columns={
        'frame_prob': f'{ck_label}_score',
        'is_FP':      f'{ck_label}_is_FP',
    })
    if shared_base is None:
        shared_base = df
    else:
        meta_cols = ['video_id', 'identity', 'clip_capture_mode',
                     'face_area_q', 'sharpness_q', 'source_resolution_bucket']
        shared_base = shared_base.merge(
            df.drop(columns=meta_cols, errors='ignore'),
            on='frame_path', how='outer'
        )

shared_base['n_ckpts_FP'] = (
    shared_base[[f'{c}_is_FP' for c in CKPTS]].fillna(0).sum(axis=1).astype(int)
)
shared_base.to_csv(os.path.join(OUT, 'shared_fp_tail.csv'), index=False)
print(f'  shared_fp_tail.csv: {len(shared_base):,} rows')

shared_dist = shared_base['n_ckpts_FP'].value_counts().sort_index()
print('  n_ckpts_FP distribution:')
for k, v in shared_dist.items():
    print(f'    {k} ckpts: {v} frames ({100*v/len(shared_base):.1f}%)')

# Of all FP fires (sum across ckpts), what fraction are on frames flagged by ≥2?
fired_frames = shared_base[shared_base['n_ckpts_FP'] > 0]
total_fires = int(shared_base[[f'{c}_is_FP' for c in CKPTS]].fillna(0).sum().sum())
fires_in_geq2 = int(
    shared_base[shared_base['n_ckpts_FP'] >= 2][[f'{c}_is_FP' for c in CKPTS]].fillna(0).sum().sum()
)
fires_in_eq3 = int(
    shared_base[shared_base['n_ckpts_FP'] == 3][[f'{c}_is_FP' for c in CKPTS]].fillna(0).sum().sum()
)
print(f'  total FP fires across 3 ckpts: {total_fires}')
print(f'    fires on frames FP for ≥2 ckpts: {fires_in_geq2} ({100*fires_in_geq2/max(1,total_fires):.1f}%)')
print(f'    fires on frames FP for ALL 3:    {fires_in_eq3} ({100*fires_in_eq3/max(1,total_fires):.1f}%)')

# ----- top-10% offending identities on teams_real_all_dev -----
print('\nTop-10% offending identities (teams_real_all_dev):')
for ck_label in CKPTS.keys():
    df = all_frame_dfs[(ck_label, CALIB_SUITE)]
    grp = df.groupby('identity').agg(
        n_total=('frame_prob', 'size'),
        n_fp=('is_FP', 'sum'),
    ).reset_index()
    grp = grp.sort_values('n_fp', ascending=False)
    n_ids = len(grp)
    top_n = max(1, int(round(0.10 * n_ids)))
    top = grp.head(top_n)
    total_fp = int(grp['n_fp'].sum())
    top_fp = int(top['n_fp'].sum())
    share = 100 * top_fp / max(1, total_fp)
    print(f'  {ck_label}: {n_ids} identities; top-{top_n} ({100*top_n/n_ids:.0f}%) account for '
          f'{top_fp}/{total_fp} = {share:.1f}% of FP fires')

print('\nDone.')
