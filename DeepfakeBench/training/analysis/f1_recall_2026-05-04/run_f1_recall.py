"""
F1 recall simulation — 2026-05-04

Extends job_14 by adding F1-calibrated tau (drop chronic-6 only, keep low-res).
F1 is the deployment-defensible filter: chronic-6 are over-represented identities
in the dev eval pool; excluding them from calibration is defensible for a
production system that won't systematically over-represent those same identities.

Outputs (per_ckpt_f1_recall.csv):
  For each (ckpt, fake_suite): recall at tau_F0, tau_F1, tau_F4 side-by-side.

Additional outputs:
  f1_tau_fpr.csv   — tau and actual FPR for F0/F1/F4 per ckpt
  FINDINGS_FACTS.txt
"""
import os, re
import numpy as np
import pandas as pd

ROOT = '/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training'
RAW  = os.path.join(ROOT, 'analysis/cpu_followups_2026-05-04/raw_reports')
OUT  = os.path.join(ROOT, 'analysis/f1_recall_2026-05-04')
os.makedirs(OUT, exist_ok=True)

CKPTS = {
    'P8A':      'p8a_reference_step5000',
    'E2B_3200': 'e2b_top_n_step3200',
    'E3_6600':  'e3_top_n_step6600',
}
REAL_DEV   = 'teams_real_all_dev'
FAKE_SUITES = {
    'visomaster_enhanced_macro_dev': 'visomaster_enhanced_macro_dev',
    'deeplive_enhanced_dev':         'deeplive_enhanced_dev',
    'teams_fake_all_dev':            'teams_fake_all_dev',
    'teams_fake_all_lockbox':        'teams_fake_all_lockbox',
}
TARGET_FPR = 0.10

CHRONIC_6 = {
    'bla_bla_chow', 'bla_bla_chow__s2',
    'pc_generator__s22', 'pc_generator__s45',
    'roy_d', 'q__s6',
}

_STRIP = re.compile(
    r'(__seq\d+|__seg_[\d.]+|_frame_\d+|_seq\d+|__real|__fake|_real|_fake|_crop_\d+|__[0-9a-f]{6,})',
    flags=re.IGNORECASE,
)

def extract_identity(vid):
    if pd.isna(vid):
        return 'UNK'
    s = str(vid)
    prev = None
    while prev != s:
        prev = s
        s = _STRIP.sub('', s)
    return s.strip('_').lower() or 'UNK'

def tau_at_fpr(scores, fpr=0.10):
    s = pd.to_numeric(scores, errors='coerce').dropna().values
    return float(np.quantile(s, 1.0 - fpr)) if len(s) else float('nan')

def fpr_at_tau(scores, tau):
    s = pd.to_numeric(scores, errors='coerce').dropna().values
    return float((s > tau).mean()) if len(s) else float('nan')

def recall_at_tau(scores, tau):
    s = pd.to_numeric(scores, errors='coerce').dropna().values
    return float((s > tau).mean()) if len(s) else float('nan')

# Load parquet metadata for low-res filter (F4 cross-check only)
pq = pd.read_parquet(os.path.join(ROOT, 'analysis/lockbox_tagging/full_tags_2026-04-27.parquet'))
pq_meta = pq[['gcs_uri', 'identity_key', 'width', 'height', 'is_no_face']].copy()
pq_meta['min_wh'] = pq_meta[['width', 'height']].min(axis=1)

print('Loading real-dev frames per ckpt...')
real_dfs = {}
for ck, ck_id in CKPTS.items():
    df = pd.read_csv(os.path.join(RAW, f'{REAL_DEV}_{ck_id}_frames_report.csv'))
    df['identity'] = df['video_id'].apply(extract_identity)
    m = df.merge(pq_meta, how='left', left_on='frame_path', right_on='gcs_uri')
    m['identity_pq'] = m['identity_key'].astype(str).str.lower()
    m.loc[m['identity_key'].isna(), 'identity_pq'] = m['identity']
    m['identity_canon'] = np.where(m['identity_key'].notna(), m['identity_pq'], m['identity'])
    real_dfs[ck] = m
    print(f'  {ck}: {len(df)} frames')

def build_masks(df):
    is_chronic = df['identity'].isin(CHRONIC_6) | df['identity_pq'].isin(CHRONIC_6)
    keep_F1 = ~is_chronic
    minWH  = pd.to_numeric(df['min_wh'], errors='coerce')
    keep_F2 = ~(minWH < 200).fillna(False)
    no_face = df['is_no_face'].fillna(False).astype(bool)
    keep_F3 = ~no_face
    return {
        'F0': pd.Series(True, index=df.index),
        'F1': keep_F1,
        'F4': keep_F1 & keep_F2 & keep_F3,
    }

# Calibrate tau per filter per ckpt
print('\nCalibrating tau (FPR=10%) per filter per ckpt...')
taus = {}
tau_fpr_rows = []
for ck, df in real_dfs.items():
    masks = build_masks(df)
    taus[ck] = {}
    for fname, mask in masks.items():
        sub = df.loc[mask, 'frame_prob']
        tau = tau_at_fpr(sub, TARGET_FPR)
        actual_fpr = fpr_at_tau(sub, tau)
        taus[ck][fname] = tau
        s = pd.to_numeric(sub, errors='coerce').dropna()
        tau_fpr_rows.append({
            'ckpt': ck, 'filter': fname,
            'n_reals_kept': int(mask.sum()),
            'tau': round(tau, 5),
            'actual_fpr': round(actual_fpr, 4),
            'actual_fpr_pct': round(100 * actual_fpr, 2),
            'real_mean': round(float(s.mean()), 4),
            'real_p50': round(float(s.quantile(0.50)), 5),
            'real_p90': round(float(s.quantile(0.90)), 5),
            'real_p99': round(float(s.quantile(0.99)), 4),
        })
        print(f'  {ck} {fname}: n={mask.sum():4d}  tau={tau:.5f}  FPR={100*actual_fpr:.2f}%')

tau_fpr_df = pd.DataFrame(tau_fpr_rows)
tau_fpr_df.to_csv(os.path.join(OUT, 'f1_tau_fpr.csv'), index=False)
print('\nf1_tau_fpr.csv written.')

# Apply per-filter taus to fake suites
print('\nApplying per-filter taus to fake suites...')
recall_rows = []
for ck, ck_id in CKPTS.items():
    for suite_short, suite_full in FAKE_SUITES.items():
        fp = os.path.join(RAW, f'{suite_full}_{ck_id}_frames_report.csv')
        if not os.path.exists(fp):
            print(f'  MISSING: {fp}')
            continue
        df_fake = pd.read_csv(fp)
        n = len(df_fake)
        row = {'ckpt': ck, 'fake_suite': suite_short, 'n_fake_frames': n}
        for fname in ['F0', 'F1', 'F4']:
            tau = taus[ck][fname]
            rec = recall_at_tau(df_fake['frame_prob'], tau)
            row[f'tau_{fname}'] = round(tau, 5)
            row[f'recall_{fname}'] = round(rec, 4)
            row[f'recall_{fname}_pct'] = round(100 * rec, 2)
        row['lift_F1_over_F0_pp'] = round(row['recall_F1_pct'] - row['recall_F0_pct'], 2)
        row['lift_F4_over_F0_pp'] = round(row['recall_F4_pct'] - row['recall_F0_pct'], 2)
        row['lift_F1_vs_F4_pp']   = round(row['recall_F1_pct'] - row['recall_F4_pct'], 2)
        recall_rows.append(row)

rec_df = pd.DataFrame(recall_rows)
rec_df.to_csv(os.path.join(OUT, 'per_ckpt_f1_recall.csv'), index=False)
print(rec_df.to_string(index=False))

# Also compute: cross-filter FPR on full real population (how bad is F1-tau on F0 reals?)
print('\n--- Cross-check: F1 tau applied to F0 real population (over-firing check) ---')
for ck, df in real_dfs.items():
    tau_f1 = taus[ck]['F1']
    fpr_on_f0 = fpr_at_tau(df['frame_prob'], tau_f1)
    print(f'  {ck}: tau_F1={tau_f1:.5f}  FPR on full F0 reals={100*fpr_on_f0:.2f}%  '
          f'(includes chronic-6; expected > 10%)')

# FINDINGS_FACTS
facts = []
facts.append('# F1 Recall Simulation — FACTS (2026-05-04)\n')
facts.append('## Setup\n')
facts.append(f'- Three filters: F0 (all reals), F1 (drop chronic-6), F4 (drop chronic-6 + lowres<200 + no-face)\n')
facts.append(f'- tau calibrated at FPR=10% on each filter\'s real subset\n')
facts.append(f'- Recall measured on same fake-suite frames for all three taus\n\n')

facts.append('## Tau and FPR per filter per ckpt\n')
facts.append(tau_fpr_df.to_string(index=False))
facts.append('\n\n')

facts.append('## Recall comparison F0 vs F1 vs F4 per (ckpt x fake_suite)\n')
facts.append(rec_df.to_string(index=False))
facts.append('\n\n')

facts.append('## Key numbers (viso, FPR=10%)\n')
viso = rec_df[rec_df['fake_suite'] == 'visomaster_enhanced_macro_dev']
for _, r in viso.iterrows():
    facts.append(
        f"  {r['ckpt']:10s}: F0={r['recall_F0_pct']:5.1f}%  "
        f"F1={r['recall_F1_pct']:5.1f}% (+{r['lift_F1_over_F0_pp']:.1f}pp)  "
        f"F4={r['recall_F4_pct']:5.1f}% (F1 vs F4 diff={r['lift_F1_vs_F4_pp']:+.1f}pp)\n"
    )

with open(os.path.join(OUT, 'FINDINGS_FACTS.txt'), 'w') as fh:
    fh.writelines(facts)

print(f'\nDone. Outputs in {OUT}')
