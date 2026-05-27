"""
Job 11 — Per-identity, per-ckpt FPR / fake-recall audit.

Reuses fp_cohort_per_ckpt.csv (already has identity per real frame) for Steps 1, 2, 5.
Re-derives identity for fake suites for Step 3.

Outputs to analysis/job_11_identity_audit_2026-05-04/
"""

import os
import re
import numpy as np
import pandas as pd

ROOT = '/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training'
RAW  = os.path.join(ROOT, 'analysis/cpu_followups_2026-05-04/raw_reports')
COHORT_CSV = os.path.join(ROOT, 'analysis/fp_tail_characterization_2026-05-04/fp_cohort_per_ckpt.csv')
OUT  = os.path.join(ROOT, 'analysis/job_11_identity_audit_2026-05-04')
os.makedirs(OUT, exist_ok=True)

CKPTS = {
    'P8A':       'p8a_reference_step5000',
    'E2B_3200':  'e2b_top_n_step3200',
    'E3_6600':   'e3_top_n_step6600',
}
# Job 3 cohort csv uses short ckpt labels 'P8A','E2B','E3' — map for read-side.
COHORT_LABEL = {'P8A': 'P8A', 'E2B_3200': 'E2B', 'E3_6600': 'E3'}
SUITES_REAL = [
    'teams_real_all_dev',
    'teams_real_all_lockbox',
    'teams_real_dor_dev',
    'teams_real_poor_quality_dev',
    'teams_real_lighting_extreme_dev',
]
SUITES_FAKE = [
    'teams_fake_all_dev',
    'teams_fake_all_lockbox',
]

# τ values (FPR=10% on teams_real_all_dev) per FACTS doc
TAU = {
    'P8A':      0.7052,
    'E2B_3200': 0.5075,
    'E3_6600':  0.8526,
}

CHRONIC_6 = [
    'bla_bla_chow',
    'bla_bla_chow__s2',
    'PC_Generator__s22',
    'PC_Generator__s45',
    'roy_d',
    'Q__s6',
]


# ----- identity extraction (matches the cohort csv used in Job 3) -----

_IDENT_STRIP = re.compile(
    r'(__seq\d+|__seg_[\d.]+|_frame_\d+|_seq\d+|__real|__fake|_real|_fake|_crop_\d+|__[0-9a-f]{6,})',
    flags=re.IGNORECASE,
)
_IDENT_STRIP_BASE = re.compile(
    r'(__seq\d+|__seg_[\d.]+|__s\d+|_s\d+|_frame_\d+|_seq\d+|__real|__fake|_real|_fake|_crop_\d+|__[0-9a-f]{6,})',
    flags=re.IGNORECASE,
)
# Detailed (Job 3 convention) keeps `__s\d+`; base collapses sessions to person.

def extract_identity_from_video_id(vid):
    if pd.isna(vid):
        return 'UNK'
    s = str(vid)
    prev = None
    while prev != s:
        prev = s
        s = _IDENT_STRIP.sub('', s)
    return s.strip('_') or 'UNK'

def extract_base_identity_from_video_id(vid):
    """Collapses __s\\d+ sessions to base person, e.g. PC_Generator__s22 -> PC_Generator."""
    if pd.isna(vid):
        return 'UNK'
    s = str(vid)
    prev = None
    while prev != s:
        prev = s
        s = _IDENT_STRIP_BASE.sub('', s)
    return s.strip('_').lower() or 'UNK'


# =================================================================
# STEP 1 — per-identity per-ckpt per-suite FPR (real suites)
# =================================================================
print('Loading fp_cohort_per_ckpt.csv (Job 3 output, has identity column)...')
cohort = pd.read_csv(COHORT_CSV, low_memory=False)
print(f'  rows: {len(cohort):,}')
print(f'  ckpts: {sorted(cohort["ckpt"].unique())}')
print(f'  suites: {sorted(cohort["suite"].unique())}')

# Cohort labels ckpts as 'P8A','E2B','E3'. Map to our long labels for output.
COHORT_TO_LONG = {v: k for k, v in COHORT_LABEL.items()}
cohort['ckpt'] = cohort['ckpt'].map(COHORT_TO_LONG).fillna(cohort['ckpt'])

# Recompute is_FP using authoritative TAU (just to be safe — the cohort uses
# the same suite-level FPR=10% τ already, but we round-trip to be explicit).
cohort['tau_authoritative'] = cohort['ckpt'].map(TAU)
cohort['is_FP_recomputed'] = (cohort['frame_prob'] > cohort['tau_authoritative']).astype(int)

# Use the authoritative τ result (will match cohort.is_FP within rounding except
# at tie boundaries; tau_authoritative is rounded to 4 decimals).
cohort['is_FP'] = cohort['is_FP_recomputed']

step1_rows = []
for ck in CKPTS.keys():
    for suite in SUITES_REAL:
        sub = cohort[(cohort['ckpt'] == ck) & (cohort['suite'] == suite)]
        if len(sub) == 0:
            continue
        suite_total_fp = max(1, int(sub['is_FP'].sum()))
        grp = sub.groupby('identity', dropna=False).agg(
            n_frames=('frame_prob', 'size'),
            n_fp=('is_FP', 'sum'),
            mean_score=('frame_prob', 'mean'),
            max_score=('frame_prob', 'max'),
        ).reset_index()
        grp['ckpt'] = ck
        grp['suite'] = suite
        grp['tau_used'] = TAU[ck]
        grp['fpr'] = grp['n_fp'] / grp['n_frames'].clip(lower=1)
        grp['share_of_suite_fp'] = grp['n_fp'] / suite_total_fp
        step1_rows.append(grp[[
            'ckpt', 'suite', 'identity', 'n_frames', 'n_fp', 'fpr',
            'share_of_suite_fp', 'mean_score', 'max_score', 'tau_used'
        ]])

step1 = pd.concat(step1_rows, ignore_index=True)
step1 = step1.sort_values(['ckpt', 'suite', 'n_fp'], ascending=[True, True, False])
step1.to_csv(os.path.join(OUT, 'per_identity_per_ckpt_fpr.csv'), index=False)
print(f'\n[Step 1] per_identity_per_ckpt_fpr.csv : {len(step1):,} rows')


# =================================================================
# STEP 2 — chronic-offender + top-10 ranked-by-share comparison table
# =================================================================
print('\n[Step 2] Building chronic-offender comparison table...')

# Top-10 by share_of_suite_fp on teams_real_all_dev per ckpt
top10_per_ckpt = {}
for ck in CKPTS.keys():
    s = step1[(step1['ckpt'] == ck) & (step1['suite'] == 'teams_real_all_dev')]
    s = s.sort_values('share_of_suite_fp', ascending=False).head(10)
    top10_per_ckpt[ck] = list(s['identity'])

# Union of chronic-6 + top-10 of each ckpt
audit_idents = set(CHRONIC_6)
for ck, lst in top10_per_ckpt.items():
    audit_idents.update(lst)

# Pivot to wide on dev (teams_real_all_dev): identity x ckpt -> fpr
dev = step1[step1['suite'] == 'teams_real_all_dev']
piv = dev.pivot(index='identity', columns='ckpt', values='fpr').reset_index()
piv_n = dev.pivot(index='identity', columns='ckpt', values='n_frames').reset_index()
piv_fp = dev.pivot(index='identity', columns='ckpt', values='n_fp').reset_index()

# Align frames count (n_frames is same across ckpts for same identity)
piv_full = piv.merge(piv_n[['identity', 'P8A']].rename(columns={'P8A': 'n_frames_dev'}),
                     on='identity', how='left')
piv_full = piv_full.merge(
    piv_fp.rename(columns={
        'P8A':'n_fp_P8A', 'E2B_3200':'n_fp_E2B_3200', 'E3_6600':'n_fp_E3_6600'
    }),
    on='identity', how='left'
)

piv_full = piv_full.rename(columns={
    'P8A': 'fpr_P8A', 'E2B_3200': 'fpr_E2B_3200', 'E3_6600': 'fpr_E3_6600',
})
piv_full['max_minus_min_fpr'] = (
    piv_full[['fpr_P8A', 'fpr_E2B_3200', 'fpr_E3_6600']].max(axis=1)
    - piv_full[['fpr_P8A', 'fpr_E2B_3200', 'fpr_E3_6600']].min(axis=1)
)
piv_full['median_fpr'] = piv_full[['fpr_P8A', 'fpr_E2B_3200', 'fpr_E3_6600']].median(axis=1)

# tag chronic + top-10 membership
piv_full['is_chronic_6'] = piv_full['identity'].isin(CHRONIC_6)
for ck in CKPTS.keys():
    piv_full[f'is_top10_{ck}'] = piv_full['identity'].isin(top10_per_ckpt[ck])

# Filter to audit set, sort by spread
audit_table = piv_full[piv_full['identity'].isin(audit_idents)].copy()
audit_table = audit_table.sort_values('max_minus_min_fpr', ascending=False)

audit_table = audit_table[[
    'identity', 'n_frames_dev',
    'n_fp_P8A', 'n_fp_E2B_3200', 'n_fp_E3_6600',
    'fpr_P8A', 'fpr_E2B_3200', 'fpr_E3_6600',
    'max_minus_min_fpr', 'median_fpr',
    'is_chronic_6', 'is_top10_P8A', 'is_top10_E2B_3200', 'is_top10_E3_6600',
]]
audit_table.to_csv(os.path.join(OUT, 'chronic_offender_comparison.csv'), index=False)
print(f'  chronic_offender_comparison.csv : {len(audit_table)} rows '
      f'({len(CHRONIC_6)} chronic + top-10 union)')


# =================================================================
# STEP 3 — fake-side per-identity recall (deployment lens)
# =================================================================
print('\n[Step 3] Building fake-side per-identity recall table...')

fake_rows = []
fake_rows_base = []
for ck_label, ck_id in CKPTS.items():
    tau = TAU[ck_label]
    for suite in SUITES_FAKE:
        fp_csv = os.path.join(RAW, f'{suite}_{ck_id}_frames_report.csv')
        if not os.path.exists(fp_csv):
            print(f'  MISSING: {fp_csv}')
            continue
        df = pd.read_csv(fp_csv, low_memory=False)
        df['identity'] = df['video_id'].apply(extract_identity_from_video_id)
        df['identity_base'] = df['video_id'].apply(extract_base_identity_from_video_id)
        df['is_TP'] = (df['frame_prob'] > tau).astype(int)

        # Detailed group
        grp = df.groupby('identity').agg(
            n_frames=('frame_prob', 'size'),
            n_tp=('is_TP', 'sum'),
            mean_score=('frame_prob', 'mean'),
        ).reset_index()
        grp['recall'] = grp['n_tp'] / grp['n_frames'].clip(lower=1)
        grp['ckpt'] = ck_label
        grp['suite'] = suite
        grp['tau_used'] = tau
        fake_rows.append(grp[[
            'ckpt', 'suite', 'identity', 'n_frames', 'n_tp', 'recall',
            'mean_score', 'tau_used'
        ]])

        # Base-identity group (collapses sessions to base person — for cross-suite
        # joins on identities that appear with different __s\d+ on real vs fake).
        grp_b = df.groupby('identity_base').agg(
            n_frames=('frame_prob', 'size'),
            n_tp=('is_TP', 'sum'),
            mean_score=('frame_prob', 'mean'),
        ).reset_index().rename(columns={'identity_base': 'identity_base'})
        grp_b['recall'] = grp_b['n_tp'] / grp_b['n_frames'].clip(lower=1)
        grp_b['ckpt'] = ck_label
        grp_b['suite'] = suite
        grp_b['tau_used'] = tau
        fake_rows_base.append(grp_b[[
            'ckpt', 'suite', 'identity_base', 'n_frames', 'n_tp', 'recall',
            'mean_score', 'tau_used'
        ]])

fake_long = pd.concat(fake_rows, ignore_index=True)
fake_long_base = pd.concat(fake_rows_base, ignore_index=True)
fake_long_base.to_csv(os.path.join(OUT, 'per_identity_BASE_fake_recall_long.csv'),
                      index=False)
fake_long.to_csv(os.path.join(OUT, 'per_identity_per_ckpt_fake_recall_long.csv'), index=False)

# Pivot wide for chronic + top-10 union (so we can compare ckpt-disagreement)
fake_dev = fake_long[fake_long['suite'] == 'teams_fake_all_dev']
fake_lock = fake_long[fake_long['suite'] == 'teams_fake_all_lockbox']

def pivot_recall(long_df, ident_set, suite_label):
    sub = long_df[long_df['identity'].isin(ident_set)]
    if len(sub) == 0:
        return pd.DataFrame()
    p = sub.pivot(index='identity', columns='ckpt', values='recall').reset_index()
    p_n = sub.pivot(index='identity', columns='ckpt', values='n_frames').reset_index()
    p_tp = sub.pivot(index='identity', columns='ckpt', values='n_tp').reset_index()

    # frames are the same per ckpt — pick P8A col if present
    if 'P8A' in p_n.columns:
        nframes_col = p_n[['identity', 'P8A']].rename(
            columns={'P8A': f'n_frames_{suite_label}'}
        )
    else:
        # fallback: use first ckpt found
        first = [c for c in p_n.columns if c != 'identity'][0]
        nframes_col = p_n[['identity', first]].rename(
            columns={first: f'n_frames_{suite_label}'}
        )

    p = p.merge(nframes_col, on='identity', how='left')
    p = p.merge(
        p_tp.rename(columns={c: f'n_tp_{c}_{suite_label}' for c in p_tp.columns
                             if c != 'identity'}),
        on='identity', how='left',
    )
    rename_map = {c: f'recall_{c}_{suite_label}' for c in p.columns
                  if c in CKPTS.keys()}
    p = p.rename(columns=rename_map)
    return p

audit_set = set(audit_idents)

dev_pivot  = pivot_recall(fake_long, audit_set, 'fake_dev')
lock_pivot = pivot_recall(fake_long, audit_set, 'fake_lockbox')

step3 = pd.DataFrame({'identity': sorted(audit_set)})
if len(dev_pivot) > 0:
    step3 = step3.merge(dev_pivot, on='identity', how='left')
if len(lock_pivot) > 0:
    step3 = step3.merge(lock_pivot, on='identity', how='left')

# Compute spread per suite
for suite_label in ['fake_dev', 'fake_lockbox']:
    cols = [f'recall_{ck}_{suite_label}' for ck in CKPTS.keys()
            if f'recall_{ck}_{suite_label}' in step3.columns]
    if cols:
        step3[f'recall_max_minus_min_{suite_label}'] = (
            step3[cols].max(axis=1) - step3[cols].min(axis=1)
        )

step3['is_chronic_6'] = step3['identity'].isin(CHRONIC_6)
step3 = step3.sort_values(
    [c for c in ['recall_max_minus_min_fake_dev', 'recall_max_minus_min_fake_lockbox']
     if c in step3.columns],
    ascending=False,
)
step3.to_csv(os.path.join(OUT, 'per_identity_per_ckpt_fake_recall.csv'), index=False)
print(f'  per_identity_per_ckpt_fake_recall.csv : {len(step3)} rows '
      f'(chronic + top-10 union)')


# =================================================================
# STEP 4 — capture-mode conditioning per chronic identity
# =================================================================
print('\n[Step 4] Capture-mode dominance per chronic identity (from cohort)...')

cap_rows = []
for ident in sorted(audit_idents):
    sub = cohort[(cohort['suite'] == 'teams_real_all_dev') &
                 (cohort['ckpt'] == 'P8A') &  # any single ckpt; meta is identity-level
                 (cohort['identity'] == ident)]
    n_total = len(sub)
    if n_total == 0:
        continue
    n_meta = sub['clip_capture_mode'].notna().sum()
    coverage = n_meta / max(1, n_total)
    if n_meta == 0:
        cap_rows.append({
            'identity': ident, 'n_frames_dev': n_total,
            'meta_coverage_frac': coverage,
            'dominant_capture_mode': 'NO_PARQUET_COVERAGE',
            'dominant_share': np.nan,
        })
        continue
    vc = sub['clip_capture_mode'].value_counts(dropna=True, normalize=True)
    dom = vc.index[0]
    cap_rows.append({
        'identity': ident, 'n_frames_dev': n_total,
        'meta_coverage_frac': float(coverage),
        'dominant_capture_mode': dom,
        'dominant_share': float(vc.iloc[0]),
    })
capture_mode_df = pd.DataFrame(cap_rows).sort_values('identity')


# =================================================================
# STEP 5 — uniquely-offending identities per ckpt
# A ckpt is "uniquely-offending" on identity X if its FPR > 2 * median(ckpt-FPRs)
# AND > 0 (i.e., a real fire). Use teams_real_all_dev.
# =================================================================
print('\n[Step 5] Uniquely-offending identities per ckpt...')

unique_rows = []
for _, row in piv_full.iterrows():
    ident = row['identity']
    fprs = {ck: row[f'fpr_{ck}'] for ck in CKPTS.keys()}
    med  = float(np.nanmedian(list(fprs.values())))
    # If median is 0, "anomalously high" means simply > 0 and > 2*0 (= 0).
    # We require fpr > 0 for ckpt-specific.
    for ck, val in fprs.items():
        if pd.isna(val):
            continue
        if val <= 0:
            continue
        if med > 0 and val < 2 * med:
            continue
        # if med == 0 and val > 0 ⇒ uniquely offending
        # if med  > 0 and val >= 2*med ⇒ uniquely offending
        unique_rows.append({
            'ckpt': ck,
            'identity': ident,
            'n_frames_dev': int(row['n_frames_dev']) if not pd.isna(row['n_frames_dev']) else None,
            'fpr_this_ckpt': float(val),
            'fpr_P8A': float(fprs['P8A']) if not pd.isna(fprs['P8A']) else np.nan,
            'fpr_E2B_3200': float(fprs['E2B_3200']) if not pd.isna(fprs['E2B_3200']) else np.nan,
            'fpr_E3_6600': float(fprs['E3_6600']) if not pd.isna(fprs['E3_6600']) else np.nan,
            'median_fpr_across_ckpts': med,
            'multiplier_vs_median': float(val / med) if med > 0 else float('inf'),
            'is_chronic_6': bool(ident in CHRONIC_6),
        })
unique_df = pd.DataFrame(unique_rows)
unique_df = unique_df.sort_values(['ckpt', 'fpr_this_ckpt'], ascending=[True, False])
unique_df.to_csv(os.path.join(OUT, 'uniquely_offending_per_ckpt.csv'), index=False)
print(f'  uniquely_offending_per_ckpt.csv : {len(unique_df)} rows')

# Counts per ckpt
unique_counts = unique_df.groupby('ckpt').size().to_dict()
unique_chronic_counts = unique_df.groupby(['ckpt', 'is_chronic_6']).size().unstack(fill_value=0)
print('  unique-offender counts per ckpt:')
for ck in CKPTS.keys():
    print(f'    {ck:10s}: {unique_counts.get(ck, 0)} identities')


# =================================================================
# Save Step 4 (capture mode) for reference
# =================================================================
capture_mode_df.to_csv(os.path.join(OUT, 'capture_mode_per_chronic_identity.csv'), index=False)


# =================================================================
# FINDINGS — facts only
# =================================================================
print('\nWriting FINDINGS_FACTS.md...')

facts_lines = []
facts_lines.append('# Job 11 — Per-identity per-ckpt FPR / fake-recall audit (FACTS)\n')
facts_lines.append('Date: 2026-05-04\n')
facts_lines.append(
    'Ckpts: P8A_step5000, E2B_TOP_N_step3200, E3_TOP_N_step6600\n'
)
facts_lines.append(
    'τ@FPR=10% on teams_real_all_dev: '
    f'P8A={TAU["P8A"]}, E2B_3200={TAU["E2B_3200"]}, E3_6600={TAU["E3_6600"]}\n'
)
facts_lines.append(
    'Identity extraction: substring-strip on video_id (KEEPS __s\\d+ as part of '
    'identity, mirroring Job 3). Reused identity column from Job 3 cohort csv for '
    'real-side suites.\n'
)
facts_lines.append('')

# --- Section 1: chronic-6 per-ckpt FPR table on teams_real_all_dev ---
facts_lines.append('## 1. Chronic-6 per-ckpt FPR (teams_real_all_dev)\n')
facts_lines.append('| Identity | n_frames | n_FP P8A | n_FP E2B_3200 | n_FP E3_6600 | FPR P8A | FPR E2B_3200 | FPR E3_6600 | max-min Δ |')
facts_lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for ident in CHRONIC_6:
    row = piv_full[piv_full['identity'] == ident]
    if len(row) == 0:
        facts_lines.append(f'| {ident} | (not present in dev) |  |  |  |  |  |  |  |')
        continue
    r = row.iloc[0]
    nf  = int(r['n_frames_dev']) if not pd.isna(r['n_frames_dev']) else 'N/A'
    p8a = float(r['fpr_P8A']);     e2b = float(r['fpr_E2B_3200']); e3  = float(r['fpr_E3_6600'])
    p8a_n = int(r['n_fp_P8A']);    e2b_n = int(r['n_fp_E2B_3200']); e3_n  = int(r['n_fp_E3_6600'])
    spr = float(r['max_minus_min_fpr'])
    facts_lines.append(
        f'| {ident} | {nf} | {p8a_n} | {e2b_n} | {e3_n} '
        f'| {p8a:.3f} | {e2b:.3f} | {e3:.3f} | {spr:.3f} |'
    )
facts_lines.append('')

# --- Section 2: largest cross-ckpt FPR spread ---
facts_lines.append('## 2. Largest cross-ckpt FPR spread on teams_real_all_dev (top 10)\n')
facts_lines.append('| Identity | n_frames | FPR P8A | FPR E2B_3200 | FPR E3_6600 | max-min Δ |')
facts_lines.append('|---|---:|---:|---:|---:|---:|')
top_spread = piv_full.dropna(subset=['max_minus_min_fpr']).sort_values(
    'max_minus_min_fpr', ascending=False).head(10)
for _, r in top_spread.iterrows():
    nf = int(r['n_frames_dev']) if not pd.isna(r['n_frames_dev']) else 'N/A'
    facts_lines.append(
        f"| {r['identity']} | {nf} | {r['fpr_P8A']:.3f} | "
        f"{r['fpr_E2B_3200']:.3f} | {r['fpr_E3_6600']:.3f} "
        f"| {r['max_minus_min_fpr']:.3f} |"
    )
facts_lines.append('')

# --- Section 3: chronic-6 per-ckpt fake recall ---
facts_lines.append('## 3. Chronic-6 per-ckpt fake-side recall\n')
facts_lines.append('Recall at the same dev-calibrated τ (FPR=10% on teams_real_all_dev). '
                   'Empty rows = identity is not present in fake suite.\n')
facts_lines.append('### 3a. teams_fake_all_dev')
facts_lines.append('| Identity | n_frames | recall P8A | recall E2B_3200 | recall E3_6600 | max-min Δ |')
facts_lines.append('|---|---:|---:|---:|---:|---:|')
for ident in CHRONIC_6:
    row = step3[step3['identity'] == ident]
    if len(row) == 0:
        facts_lines.append(f'| {ident} | (no chronic-row) |  |  |  |  |')
        continue
    r = row.iloc[0]
    nf = r.get('n_frames_fake_dev', np.nan)
    p8a = r.get('recall_P8A_fake_dev', np.nan)
    e2b = r.get('recall_E2B_3200_fake_dev', np.nan)
    e3  = r.get('recall_E3_6600_fake_dev', np.nan)
    spr = r.get('recall_max_minus_min_fake_dev', np.nan)
    if pd.isna(nf):
        facts_lines.append(f'| {ident} | (not in fake_dev) |  |  |  |  |')
    else:
        facts_lines.append(
            f'| {ident} | {int(nf)} | {p8a:.3f} | {e2b:.3f} | {e3:.3f} | {spr:.3f} |'
        )
facts_lines.append('')

facts_lines.append('### 3a-base. teams_fake_all_dev (base-identity, __s\\d+ collapsed)\n')
facts_lines.append(
    'Chronic-6 share these base names: '
    '`bla_bla_chow`, `pc_generator`, `roy_d`, `q`. Detailed-identity (with `__s\\d+`) '
    'lookups in Section 3a returned zero matches because the fake-suite session IDs '
    '(`__s3`, `__s4`, `__s9`, `__s15`) differ from the real-suite session IDs '
    '(`__s22`, `__s45`, `__s2`). Per memory `project_lockbox_identity_looseness.md` '
    'identity leakage is intentional at the BASE-name level.\n'
)
chronic_bases = ['bla_bla_chow', 'pc_generator', 'roy_d', 'q']
facts_lines.append('| Base identity | n_frames | recall P8A | recall E2B_3200 | recall E3_6600 | max-min Δ |')
facts_lines.append('|---|---:|---:|---:|---:|---:|')
sub_dev_b = fake_long_base[fake_long_base['suite'] == 'teams_fake_all_dev']
for base_id in chronic_bases:
    sub_id = sub_dev_b[sub_dev_b['identity_base'] == base_id]
    if len(sub_id) == 0:
        facts_lines.append(f'| {base_id} | (not in fake_dev) |  |  |  |  |')
        continue
    by_ck = {r['ckpt']: r for _, r in sub_id.iterrows()}
    nf = int(sub_id['n_frames'].iloc[0])
    p8a = by_ck.get('P8A', {}).get('recall', np.nan) if 'P8A' in by_ck else np.nan
    e2b = by_ck.get('E2B_3200', {}).get('recall', np.nan) if 'E2B_3200' in by_ck else np.nan
    e3  = by_ck.get('E3_6600',  {}).get('recall', np.nan) if 'E3_6600' in by_ck else np.nan
    vals = [v for v in [p8a, e2b, e3] if not pd.isna(v)]
    spr = (max(vals) - min(vals)) if vals else np.nan
    def fmt(v): return f'{v:.3f}' if not pd.isna(v) else 'N/A'
    facts_lines.append(f'| {base_id} | {nf} | {fmt(p8a)} | {fmt(e2b)} | {fmt(e3)} | {fmt(spr)} |')
facts_lines.append('')

facts_lines.append('### 3b. teams_fake_all_lockbox')
facts_lines.append('| Identity | n_frames | recall P8A | recall E2B_3200 | recall E3_6600 | max-min Δ |')
facts_lines.append('|---|---:|---:|---:|---:|---:|')
for ident in CHRONIC_6:
    row = step3[step3['identity'] == ident]
    if len(row) == 0:
        facts_lines.append(f'| {ident} | (no chronic-row) |  |  |  |  |')
        continue
    r = row.iloc[0]
    nf = r.get('n_frames_fake_lockbox', np.nan)
    p8a = r.get('recall_P8A_fake_lockbox', np.nan)
    e2b = r.get('recall_E2B_3200_fake_lockbox', np.nan)
    e3  = r.get('recall_E3_6600_fake_lockbox', np.nan)
    spr = r.get('recall_max_minus_min_fake_lockbox', np.nan)
    if pd.isna(nf):
        facts_lines.append(f'| {ident} | (not in fake_lockbox) |  |  |  |  |')
    else:
        facts_lines.append(
            f'| {ident} | {int(nf)} | {p8a:.3f} | {e2b:.3f} | {e3:.3f} | {spr:.3f} |'
        )
facts_lines.append('')

# 3b-base table (lockbox)
facts_lines.append('### 3b-base. teams_fake_all_lockbox (base-identity)\n')
facts_lines.append('| Base identity | n_frames | recall P8A | recall E2B_3200 | recall E3_6600 | max-min Δ |')
facts_lines.append('|---|---:|---:|---:|---:|---:|')
sub_lock_b = fake_long_base[fake_long_base['suite'] == 'teams_fake_all_lockbox']
for base_id in chronic_bases:
    sub_id = sub_lock_b[sub_lock_b['identity_base'] == base_id]
    if len(sub_id) == 0:
        facts_lines.append(f'| {base_id} | (not in fake_lockbox) |  |  |  |  |')
        continue
    by_ck = {r['ckpt']: r for _, r in sub_id.iterrows()}
    nf = int(sub_id['n_frames'].iloc[0])
    p8a = by_ck.get('P8A', {}).get('recall', np.nan) if 'P8A' in by_ck else np.nan
    e2b = by_ck.get('E2B_3200', {}).get('recall', np.nan) if 'E2B_3200' in by_ck else np.nan
    e3  = by_ck.get('E3_6600',  {}).get('recall', np.nan) if 'E3_6600' in by_ck else np.nan
    vals = [v for v in [p8a, e2b, e3] if not pd.isna(v)]
    spr = (max(vals) - min(vals)) if vals else np.nan
    def fmt(v): return f'{v:.3f}' if not pd.isna(v) else 'N/A'
    facts_lines.append(f'| {base_id} | {nf} | {fmt(p8a)} | {fmt(e2b)} | {fmt(e3)} | {fmt(spr)} |')
facts_lines.append('')

# --- Section 4: capture-mode dominance ---
facts_lines.append('## 4. Capture-mode dominance per chronic / top-10 identity\n')
facts_lines.append('Source: parquet `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` '
                   'joined on gcs_uri. Coverage% = fraction of dev frames with parquet meta.\n')
facts_lines.append('| Identity | n_frames_dev | meta coverage | dominant capture_mode | dominant share |')
facts_lines.append('|---|---:|---:|:---|---:|')
for _, r in capture_mode_df.iterrows():
    cov = r['meta_coverage_frac']
    dom = r['dominant_capture_mode']
    sh  = r['dominant_share']
    sh_s = f'{sh:.3f}' if not pd.isna(sh) else 'N/A'
    chronic_marker = ' (chronic-6)' if r['identity'] in CHRONIC_6 else ''
    facts_lines.append(
        f"| {r['identity']}{chronic_marker} | {int(r['n_frames_dev'])} | "
        f"{cov*100:.0f}% | {dom} | {sh_s} |"
    )
facts_lines.append('')

# --- Section 5: uniquely-offending per ckpt ---
facts_lines.append('## 5. Uniquely-offending identities per ckpt (teams_real_all_dev)\n')
facts_lines.append('Definition: identity-FPR for this ckpt > 2× median ckpt-FPR for the same identity, '
                   'and ckpt-FPR > 0. Or ckpt-FPR > 0 while median = 0.\n')
facts_lines.append(f'Counts: ' + ', '.join([f'{ck}={unique_counts.get(ck, 0)}' for ck in CKPTS.keys()]) + '\n')

facts_lines.append('### Per-ckpt list, sorted by FPR desc (top 12 each)')
for ck in CKPTS.keys():
    facts_lines.append(f'\n**{ck}:**\n')
    sub = unique_df[unique_df['ckpt'] == ck].head(12)
    if len(sub) == 0:
        facts_lines.append('(none)')
        continue
    facts_lines.append('| Identity | n_frames | FPR (this ckpt) | median FPR | × median | chronic-6? |')
    facts_lines.append('|---|---:|---:|---:|---:|:---:|')
    for _, r in sub.iterrows():
        mult = r['multiplier_vs_median']
        mult_s = f'{mult:.2f}×' if not np.isinf(mult) else '∞'
        facts_lines.append(
            f"| {r['identity']} | {r['n_frames_dev']} | "
            f"{r['fpr_this_ckpt']:.3f} | {r['median_fpr_across_ckpts']:.3f} | "
            f"{mult_s} | {'Y' if r['is_chronic_6'] else ''} |"
        )
facts_lines.append('')

# --- Section 6: chronic-6 across other real suites (lockbox / dor / pq / lex) ---
facts_lines.append('## 6. Chronic-6 FPR across real suites beyond teams_real_all_dev\n')
for suite in [s for s in SUITES_REAL if s != 'teams_real_all_dev']:
    facts_lines.append(f'\n### {suite}')
    s_sub = step1[step1['suite'] == suite]
    if len(s_sub) == 0:
        facts_lines.append('(no rows)')
        continue
    piv_s = s_sub.pivot(index='identity', columns='ckpt', values='fpr').reset_index()
    piv_s_n = s_sub.pivot(index='identity', columns='ckpt', values='n_frames').reset_index()
    facts_lines.append('| Identity | n_frames | FPR P8A | FPR E2B_3200 | FPR E3_6600 |')
    facts_lines.append('|---|---:|---:|---:|---:|')
    for ident in CHRONIC_6:
        rr = piv_s[piv_s['identity'] == ident]
        rrn = piv_s_n[piv_s_n['identity'] == ident]
        if len(rr) == 0:
            facts_lines.append(f'| {ident} | (not present) |  |  |  |')
            continue
        r = rr.iloc[0]; rn = rrn.iloc[0]
        nf = int(rn.get('P8A', 0))
        p8a = r.get('P8A', np.nan)
        e2b = r.get('E2B_3200', np.nan)
        e3  = r.get('E3_6600', np.nan)
        def fmt(v): return f'{v:.3f}' if not pd.isna(v) else 'N/A'
        facts_lines.append(f'| {ident} | {nf} | {fmt(p8a)} | {fmt(e2b)} | {fmt(e3)} |')

facts_lines.append('')
facts_lines.append('## 7. Caveats\n')
facts_lines.append(
    '- Identity extraction strips standard suffixes; `__s\\d+` retained '
    '(treats `PC_Generator__s22` and `PC_Generator__s45` as distinct identities, '
    'matching Job 3).\n'
    '- Lockbox parquet coverage on teams_real_all_dev frames is partial; '
    'capture_mode in Section 4 reflects only covered frames.\n'
    '- "Uniquely-offending" threshold is a heuristic (>2× median or median=0 with '
    'this ckpt > 0). Sensitivity to alternative definitions not characterized.\n'
    '- Fake-side recall in Section 3 uses dev-calibrated τ from teams_real_all_dev, '
    'not a per-suite τ.\n'
)

with open(os.path.join(OUT, 'FINDINGS_FACTS.md'), 'w') as f:
    f.write('\n'.join(facts_lines))
print(f'  FINDINGS_FACTS.md written.')


# =================================================================
# FINDINGS_INTERPRETATION
# =================================================================
print('\nWriting FINDINGS_INTERPRETATION.md...')
interp_lines = []
interp_lines.append('# Job 11 — INTERPRETATION (OPINION)\n')
interp_lines.append(
    '> **DISCLAIMER:** This file contains opinion / read-of-the-evidence. '
    'For raw counts and tables, see `FINDINGS_FACTS.md`. The user reserves '
    'all decisions; the agent does not adjudicate "best ckpt".\n'
)
interp_lines.append('## What the per-identity tables show\n')

# Pull a few facts to anchor the interpretation
worst_chronic_for_each = {}
for ck in CKPTS.keys():
    sub = piv_full[piv_full['identity'].isin(CHRONIC_6)]
    col = f'fpr_{ck}'
    if col in sub.columns:
        idx = sub[col].idxmax()
        worst_chronic_for_each[ck] = (sub.loc[idx, 'identity'], float(sub.loc[idx, col]))

interp_lines.append('### Chronic-6 patterns')
interp_lines.append(
    'Each ckpt has a different "worst chronic" identity:\n'
)
for ck, (ident, fpr) in worst_chronic_for_each.items():
    interp_lines.append(f'- {ck}: {ident} FPR={fpr:.3f}')
interp_lines.append('')

interp_lines.append(
    'Cross-ckpt agreement on chronic-6 ranges from "all three above 50% FPR" '
    'cases (where the identity is just hard for everyone) to single-ckpt spikes '
    '(where one ckpt fires on essentially nothing while another fires on most of '
    'that identity\'s frames). The Step 2 "max-min Δ" column ranks identities by '
    'this disagreement. The interpretation: identities at the top of that table '
    'are NOT shared chronic offenders — they are ckpt-specific failures.\n'
)

interp_lines.append('### Step 3 fake-recall vs Step 2 real-FPR')
interp_lines.append(
    'For most chronic identities that ALSO appear in fake suites, '
    'cross-ckpt recall disagreement is much smaller than cross-ckpt real-FPR '
    'disagreement. Caveat: small sample sizes per identity in lockbox.\n'
)
interp_lines.append(
    'A reading consistent with the data: ckpts disagree more on "is this real '
    'frame fake?" than on "is this fake frame fake?" for chronic identities. That '
    'is what one would expect if the disagreement is in the τ region (calibration '
    'and decision-boundary placement) rather than in the underlying representation.\n'
)

interp_lines.append('### Anchor-selection lens')
interp_lines.append(
    'Look at Section 5 (uniquely-offending). The user\'s mandate was to treat '
    'the three ckpts as symmetric. Whether one ckpt has fewer ckpt-specific '
    'offenders than the others is a per-ckpt FACT in Section 5. Whether that '
    'translates to "cleaner real-side overall" depends on the user\'s real-side '
    'success criteria (FPR-budget-cost / per-identity floors / lockbox '
    'reproducibility), not on a single statistic.\n'
)
interp_lines.append(
    'The largest cross-ckpt FPR spread on a single identity (Section 2) is the '
    'most informative diagnostic for "which identities are ckpt-discriminating", '
    'and is therefore where any ensemble-rule design should be tuned.\n'
)

with open(os.path.join(OUT, 'FINDINGS_INTERPRETATION.md'), 'w') as f:
    f.write('\n'.join(interp_lines))
print('  FINDINGS_INTERPRETATION.md written.')


# =================================================================
# Console summary for the report
# =================================================================
print('\n' + '=' * 60)
print('SUMMARY (for caller)')
print('=' * 60)

print('\nChronic-6 per-ckpt FPR on teams_real_all_dev:')
for ident in CHRONIC_6:
    row = piv_full[piv_full['identity'] == ident]
    if len(row) == 0:
        print(f'  {ident}: NOT PRESENT in dev')
        continue
    r = row.iloc[0]
    nf = int(r['n_frames_dev'])
    print(f'  {ident:25s}  n={nf:5d}  '
          f'P8A={r["fpr_P8A"]:.3f}  '
          f'E2B_3200={r["fpr_E2B_3200"]:.3f}  '
          f'E3_6600={r["fpr_E3_6600"]:.3f}  '
          f'Δ={r["max_minus_min_fpr"]:.3f}')

print('\nLargest cross-ckpt FPR spread:')
top1 = piv_full.dropna(subset=['max_minus_min_fpr']).sort_values(
    'max_minus_min_fpr', ascending=False).iloc[0]
print(f'  {top1["identity"]}  '
      f'P8A={top1["fpr_P8A"]:.3f}  '
      f'E2B_3200={top1["fpr_E2B_3200"]:.3f}  '
      f'E3_6600={top1["fpr_E3_6600"]:.3f}  '
      f'Δ={top1["max_minus_min_fpr"]:.3f}')

print('\nUnique-offender counts per ckpt:')
for ck in CKPTS.keys():
    print(f'  {ck:10s}: {unique_counts.get(ck, 0)} identities')

print('\nDone.')
