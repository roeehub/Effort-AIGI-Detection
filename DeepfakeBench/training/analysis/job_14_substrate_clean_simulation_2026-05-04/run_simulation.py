"""
Job 14 — Substrate-cleaning simulation for P8A / E2B_3200 / E3_6600.

Filter sets on teams_real_all_dev (4564 frames), keep tau frozen at F0-calibrated
FPR=10% to see deployed-tau performance on cleaner substrate; also recalibrate on F4
and re-eval fake-side recall.

Outputs:
- filter_inventory.csv
- per_ckpt_filter_fpr.csv
- per_ckpt_filter_recall_lift.csv
- fpr_drop_decomposition.csv
- FINDINGS_FACTS.md
- FINDINGS_INTERPRETATION.md
"""
import os
import re
import numpy as np
import pandas as pd

ROOT = '/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training'
RAW = os.path.join(ROOT, 'analysis/cpu_followups_2026-05-04/raw_reports')
OUT = os.path.join(ROOT, 'analysis/job_14_substrate_clean_simulation_2026-05-04')
os.makedirs(OUT, exist_ok=True)

CKPTS = {
    'P8A':       'p8a_reference_step5000',
    'E2B_3200':  'e2b_top_n_step3200',
    'E3_6600':   'e3_top_n_step6600',
}
REAL_DEV = 'teams_real_all_dev'
FAKE_SUITES = {
    'visomaster_enhanced_macro_dev': 'visomaster_enhanced_macro_dev',
    'deeplive_enhanced_dev':         'deeplive_enhanced_dev',
    'teams_fake_all_dev':            'teams_fake_all_dev',
    'teams_fake_all_lockbox':        'teams_fake_all_lockbox',
}
TARGET_FPR = 0.10

CHRONIC_6 = {
    'bla_bla_chow',
    'bla_bla_chow__s2',
    'pc_generator__s22',
    'pc_generator__s45',
    'roy_d',
    'q__s6',
}

# ---------- helpers ----------
_IDENT_STRIP = re.compile(
    r'(__seq\d+|__seg_[\d.]+|_frame_\d+|_seq\d+|__real|__fake|_real|_fake|_crop_\d+|__[0-9a-f]{6,})',
    flags=re.IGNORECASE,
)
# NOTE: Chronic-6 includes `bla_bla_chow__s2` (subject-session token). So we should NOT
# strip `__s\d+` before matching against the chronic list, because `__s\d+` is part of
# the canonical identity for the chronic list. We canonicalize with __s\d+ kept.

def extract_identity_from_video_id(vid):
    """Strip frame/segment suffixes but preserve __s<digits> session token."""
    if pd.isna(vid):
        return 'UNK'
    s = str(vid)
    prev = None
    while prev != s:
        prev = s
        s = _IDENT_STRIP.sub('', s)
    return s.strip('_').lower() or 'UNK'

def fpr_threshold(scores, target_fpr=0.10):
    s = pd.to_numeric(scores, errors='coerce').dropna().values
    if len(s) == 0:
        return float('nan')
    # FPR = fraction(scores > tau) -> tau is (1-target_fpr) quantile
    return float(np.quantile(s, 1.0 - target_fpr))

def fpr_at_tau(scores, tau):
    s = pd.to_numeric(scores, errors='coerce').dropna().values
    if len(s) == 0:
        return float('nan')
    return float((s > tau).sum() / len(s))

def recall_at_tau(scores, tau):
    """For fakes (label=1), recall = fraction(scores > tau)."""
    s = pd.to_numeric(scores, errors='coerce').dropna().values
    if len(s) == 0:
        return float('nan')
    return float((s > tau).sum() / len(s))

# ---------- load metadata ----------
print('Loading parquet metadata...')
pq = pd.read_parquet(os.path.join(ROOT, 'analysis/lockbox_tagging/full_tags_2026-04-27.parquet'))
pq_meta = pq[['gcs_uri', 'identity_key', 'width', 'height', 'is_no_face', 'clip_capture_mode']].copy()
pq_meta['min_wh'] = pq_meta[['width', 'height']].min(axis=1)
print(f'  parquet rows: {len(pq_meta):,}')

# ---------- load + tag dev real frames per ckpt ----------
print('\nLoading teams_real_all_dev per ckpt + tagging...')
real_dfs = {}
for ck_label, ck_id in CKPTS.items():
    df = pd.read_csv(os.path.join(RAW, f'{REAL_DEV}_{ck_id}_frames_report.csv'))
    df['identity'] = df['video_id'].apply(extract_identity_from_video_id)
    merged = df.merge(pq_meta, how='left', left_on='frame_path', right_on='gcs_uri')
    # When parquet identity_key exists prefer it, but lower-cased
    merged['identity_pq'] = merged['identity_key'].astype(str).str.lower()
    merged.loc[merged['identity_key'].isna(), 'identity_pq'] = merged['identity']
    # Use parquet identity if covered, else regex-derived
    merged['identity_canon'] = np.where(merged['identity_key'].notna(), merged['identity_pq'], merged['identity'])
    merged['ckpt'] = ck_label
    real_dfs[ck_label] = merged

# Sanity: chronic-6 identity coverage
print('\nChronic-6 identity coverage (regex-derived video_id identity):')
for ck_label, df in real_dfs.items():
    covered = df['identity'].isin(CHRONIC_6).sum()
    print(f'  {ck_label}: {covered} frames match chronic-6')

# Also try parquet identity_key match (lowercased)
print('\nChronic-6 identity coverage (parquet identity_key, lowered):')
for ck_label, df in real_dfs.items():
    covered = df['identity_pq'].isin(CHRONIC_6).sum()
    print(f'  {ck_label}: {covered} frames match chronic-6')

print('\nDistinct identities (regex from video_id) sample:')
print(real_dfs['P8A']['identity'].value_counts().head(20))

# ---------- F0..F4 filters ----------
def build_filters(df):
    """Return dict mask_name -> boolean Series (True = KEEP).
    F0: keep all
    F1: drop chronic-6 identities (use identity from video_id regex; also union with parquet identity_key)
    F2: drop frames with min_wh < 200 (only restricts where parquet covers; uncovered KEPT)
    F3: drop frames where is_no_face == True (only where parquet covers; uncovered KEPT)
    F4: F1 ∧ F2 ∧ F3
    """
    n = len(df)
    is_chronic = df['identity'].isin(CHRONIC_6) | df['identity_pq'].isin(CHRONIC_6)
    keep_F1 = ~is_chronic

    # F2: drop where min_wh<200 AND parquet covers; uncovered frames are kept (caveat documented)
    minWH = pd.to_numeric(df['min_wh'], errors='coerce')
    keep_F2 = ~(minWH < 200).fillna(False)  # NaN treated as kept

    # F3: drop where is_no_face True AND parquet covers; NaN kept
    no_face = df['is_no_face'].fillna(False).astype(bool)
    keep_F3 = ~no_face

    keep_F0 = pd.Series(True, index=df.index)
    keep_F4 = keep_F1 & keep_F2 & keep_F3
    return {
        'F0': keep_F0,
        'F1': keep_F1,
        'F2': keep_F2,
        'F3': keep_F3,
        'F4': keep_F4,
    }

# ---------- filter inventory ----------
print('\nBuilding filter inventory...')
inventory_rows = []
for ck_label, df in real_dfs.items():
    masks = build_filters(df)
    n_total = len(df)
    parquet_cov = df['gcs_uri'].notna().sum()
    for fname, mask in masks.items():
        kept = int(mask.sum())
        dropped = n_total - kept
        # Per-axis dropped counts (independent)
        is_chronic = (df['identity'].isin(CHRONIC_6) | df['identity_pq'].isin(CHRONIC_6))
        dropped_chronic = int(is_chronic.sum())
        minWH = pd.to_numeric(df['min_wh'], errors='coerce')
        dropped_lowres = int((minWH < 200).fillna(False).sum())
        no_face = df['is_no_face'].fillna(False).astype(bool)
        dropped_noface = int(no_face.sum())
        inventory_rows.append({
            'ckpt': ck_label,
            'filter': fname,
            'n_kept': kept,
            'n_dropped': dropped,
            'n_total': n_total,
            'pct_kept': round(100 * kept / n_total, 2),
            'parquet_covered_total': int(parquet_cov),
            'parquet_coverage_pct': round(100 * parquet_cov / n_total, 2),
            'axis_drop_chronic6_alone': dropped_chronic,
            'axis_drop_lowres_lt200_alone': dropped_lowres,
            'axis_drop_isnoface_alone': dropped_noface,
        })
inv_df = pd.DataFrame(inventory_rows)
inv_df.to_csv(os.path.join(OUT, 'filter_inventory.csv'), index=False)
print(inv_df.to_string(index=False))

# ---------- per-ckpt filter FPR ----------
print('\nComputing per-ckpt filter FPR (F0-calibrated tau)...')
calib_tau_F0 = {}
for ck_label, df in real_dfs.items():
    calib_tau_F0[ck_label] = fpr_threshold(df['frame_prob'], TARGET_FPR)
    print(f'  {ck_label} F0 tau@FPR=10%: {calib_tau_F0[ck_label]:.4f}')

fpr_rows = []
for ck_label, df in real_dfs.items():
    masks = build_filters(df)
    tau0 = calib_tau_F0[ck_label]
    for fname, mask in masks.items():
        sub = df.loc[mask, 'frame_prob']
        fpr = fpr_at_tau(sub, tau0)
        s = pd.to_numeric(sub, errors='coerce').dropna()
        fpr_rows.append({
            'ckpt': ck_label,
            'filter': fname,
            'n_frames': len(sub),
            'tau_used_F0_calibrated': round(tau0, 4),
            'fpr_actual': round(fpr, 4),
            'fpr_actual_pct': round(100 * fpr, 2),
            'mean_score': round(float(s.mean()), 4) if len(s) else float('nan'),
            'p50': round(float(s.quantile(0.50)), 4) if len(s) else float('nan'),
            'p90': round(float(s.quantile(0.90)), 4) if len(s) else float('nan'),
            'p99': round(float(s.quantile(0.99)), 4) if len(s) else float('nan'),
        })
fpr_df = pd.DataFrame(fpr_rows)
fpr_df.to_csv(os.path.join(OUT, 'per_ckpt_filter_fpr.csv'), index=False)
print(fpr_df.to_string(index=False))

# ---------- recalibrate tau on F4 reals + apply to fakes ----------
print('\nRecalibrating tau on F4 reals (FPR=10%) + applying to fakes...')
calib_tau_F4 = {}
for ck_label, df in real_dfs.items():
    masks = build_filters(df)
    f4_scores = df.loc[masks['F4'], 'frame_prob']
    calib_tau_F4[ck_label] = fpr_threshold(f4_scores, TARGET_FPR)
    print(f'  {ck_label} F4 tau@FPR=10%: {calib_tau_F4[ck_label]:.4f} (F0 was {calib_tau_F0[ck_label]:.4f})')

recall_rows = []
for ck_label, ck_id in CKPTS.items():
    tau0 = calib_tau_F0[ck_label]
    tau4 = calib_tau_F4[ck_label]
    for suite_short, suite_full in FAKE_SUITES.items():
        fp_csv = os.path.join(RAW, f'{suite_full}_{ck_id}_frames_report.csv')
        if not os.path.exists(fp_csv):
            print(f'  MISSING: {fp_csv}')
            continue
        df_fake = pd.read_csv(fp_csv)
        n = len(df_fake)
        rec0 = recall_at_tau(df_fake['frame_prob'], tau0)
        rec4 = recall_at_tau(df_fake['frame_prob'], tau4)
        recall_rows.append({
            'ckpt': ck_label,
            'fake_suite': suite_short,
            'n_fake_frames': n,
            'tau_F0': round(tau0, 4),
            'tau_F4': round(tau4, 4),
            'recall_at_tau_F0': round(rec0, 4),
            'recall_at_tau_F4': round(rec4, 4),
            'recall_at_tau_F0_pct': round(100 * rec0, 2),
            'recall_at_tau_F4_pct': round(100 * rec4, 2),
            'lift_abs_pp': round(100 * (rec4 - rec0), 2),
            'lift_rel_pct': round(100 * (rec4 - rec0) / max(1e-9, rec0), 1),
        })
rec_df = pd.DataFrame(recall_rows)
rec_df.to_csv(os.path.join(OUT, 'per_ckpt_filter_recall_lift.csv'), index=False)
print(rec_df.to_string(index=False))

# ---------- FPR-drop axis decomposition ----------
print('\nDecomposing F4 FPR drop by axis (each axis applied alone)...')
decomp_rows = []
for ck_label, df in real_dfs.items():
    tau0 = calib_tau_F0[ck_label]
    fpr_F0 = fpr_at_tau(df['frame_prob'], tau0)
    masks = build_filters(df)

    fpr_chronic_only = fpr_at_tau(df.loc[masks['F1'], 'frame_prob'], tau0)
    fpr_lowres_only  = fpr_at_tau(df.loc[masks['F2'], 'frame_prob'], tau0)
    fpr_noface_only  = fpr_at_tau(df.loc[masks['F3'], 'frame_prob'], tau0)
    fpr_F4 = fpr_at_tau(df.loc[masks['F4'], 'frame_prob'], tau0)

    total_drop_pp = 100 * (fpr_F0 - fpr_F4)

    for axis, fpr_axis in [
        ('chronic6', fpr_chronic_only),
        ('lowres_lt200', fpr_lowres_only),
        ('is_no_face', fpr_noface_only),
    ]:
        drop_pp = 100 * (fpr_F0 - fpr_axis)
        share = (drop_pp / total_drop_pp * 100) if total_drop_pp > 0 else float('nan')
        decomp_rows.append({
            'ckpt': ck_label,
            'axis': axis,
            'fpr_F0_pct': round(100 * fpr_F0, 2),
            'fpr_axis_alone_pct': round(100 * fpr_axis, 2),
            'fpr_drop_pp_axis_alone': round(drop_pp, 2),
            'fpr_F4_pct': round(100 * fpr_F4, 2),
            'fpr_drop_pp_F4_total': round(total_drop_pp, 2),
            'axis_share_of_F4_drop_pct': round(share, 1) if pd.notna(share) else float('nan'),
        })
decomp_df = pd.DataFrame(decomp_rows)
decomp_df.to_csv(os.path.join(OUT, 'fpr_drop_decomposition.csv'), index=False)
print(decomp_df.to_string(index=False))

print('\nDone — wrote outputs to', OUT)
