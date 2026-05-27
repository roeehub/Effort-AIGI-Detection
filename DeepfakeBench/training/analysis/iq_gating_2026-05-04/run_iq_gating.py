"""
IQ-quartile conditional recall analysis — 2026-05-04

Tests whether runtime input-quality gating is viable as a deployment lever.
Uses existing per-frame IQ features (crop_attributes.csv, 550 viso fakes) joined
with per-ckpt scores. Bins viso fakes by laplacian_var quartile and measures:
- recall at tau_F0 (10% FPR, full dev reals)
- recall at tau_F1 (10% FPR, chronic-6-dropped dev reals)
- fraction of fakes that fall in each IQ quartile
- "gating defensibility" metric: at what laplacian_var cutoff does recall collapse
  below 50%? Is that cutoff at an extreme quantile (defensible) or near the median?

Outputs:
  viso_recall_by_iq_quartile.csv   — per (ckpt, iq_feature, quartile): recall, n
  viso_iq_recall_collapse.csv      — threshold at which recall drops below 50%
  FINDINGS_FACTS.txt
"""
import os, re
import numpy as np
import pandas as pd

ROOT  = '/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training'
RAW   = os.path.join(ROOT, 'analysis/cpu_followups_2026-05-04/raw_reports')
IQ_CSV = os.path.join(ROOT, 'analysis/score_distribution_2026-05-02/outputs/crop_attributes.csv')
F1_TAU_CSV = os.path.join(ROOT, 'analysis/f1_recall_2026-05-04/f1_tau_fpr.csv')
OUT   = os.path.join(ROOT, 'analysis/iq_gating_2026-05-04')
os.makedirs(OUT, exist_ok=True)

CKPTS = {
    'P8A':      'p8a_reference_step5000',
    'E2B_3200': 'e2b_top_n_step3200',
    'E3_6600':  'e3_top_n_step6600',
}
VISO_SUITE = 'visomaster_enhanced_macro_dev'

IQ_FEATURES = ['laplacian_var', 'sobel_edge_mean', 'luma_p10', 'luma_p90',
                'luma_mean', 'saturation_mean', 'skin_frac']

# Load IQ attributes (filename-keyed, 550 viso fakes)
print('Loading crop_attributes.csv...')
iq_df = pd.read_csv(IQ_CSV)
print(f'  shape: {iq_df.shape}, columns: {list(iq_df.columns)}')

# Load per-ckpt viso scores; join on filename extracted from frame_path
print('Loading per-ckpt viso scores...')
score_dfs = {}
for ck, ck_id in CKPTS.items():
    fp = os.path.join(RAW, f'{VISO_SUITE}_{ck_id}_frames_report.csv')
    df = pd.read_csv(fp)
    df['filename'] = df['frame_path'].str.split('/').str[-1]
    score_dfs[ck] = df[['filename', 'frame_prob']].rename(columns={'frame_prob': f'score_{ck}'})
    print(f'  {ck}: {len(df)} rows')

# Join all ckpt scores + IQ features on filename
base = score_dfs['P8A'].copy()
for ck in ['E2B_3200', 'E3_6600']:
    base = base.merge(score_dfs[ck], on='filename', how='inner')
merged = base.merge(iq_df, on='filename', how='inner')
print(f'\nJoined: {len(merged)} frames ({len(merged)}/{len(iq_df)} viso fakes covered)')

# Load taus from F1 recall simulation (F0 and F1)
taus = {}
if os.path.exists(F1_TAU_CSV):
    tau_df = pd.read_csv(F1_TAU_CSV)
    for _, r in tau_df.iterrows():
        ck = r['ckpt']
        if ck not in taus:
            taus[ck] = {}
        taus[ck][r['filter']] = r['tau']
    print(f'\nLoaded taus from {F1_TAU_CSV}')
else:
    # Fallback: hardcoded from job_14 outputs
    taus = {
        'P8A':      {'F0': 0.7050, 'F1': None},
        'E2B_3200': {'F0': 0.5064, 'F1': None},
        'E3_6600':  {'F0': 0.8520, 'F1': None},
    }
    print('\nWARNING: f1_tau_fpr.csv not found, using F0 taus only from hardcoded values.')

for ck in CKPTS:
    for f in ['F0', 'F1']:
        t = taus.get(ck, {}).get(f)
        print(f'  {ck} tau_{f}: {t}')

# Per-iq-feature quartile recall
print('\nComputing per-iq-quartile recall...')
quartile_rows = []
for iq_feat in IQ_FEATURES:
    if iq_feat not in merged.columns:
        print(f'  SKIP (missing): {iq_feat}')
        continue
    iq_vals = pd.to_numeric(merged[iq_feat], errors='coerce')
    quartiles = pd.qcut(iq_vals, q=4, labels=['Q1_low', 'Q2', 'Q3', 'Q4_high'], duplicates='drop')
    merged[f'_q_{iq_feat}'] = quartiles
    for ck in CKPTS:
        score_col = f'score_{ck}'
        for q_label in merged[f'_q_{iq_feat}'].cat.categories:
            mask = merged[f'_q_{iq_feat}'] == q_label
            sub = merged.loc[mask, score_col]
            n = int(mask.sum())
            iq_sub = iq_vals[mask]
            iq_lo = float(iq_sub.min()) if n else float('nan')
            iq_hi = float(iq_sub.max()) if n else float('nan')
            row = {
                'ck': ck, 'iq_feature': iq_feat, 'quartile': str(q_label),
                'n_fakes': n,
                'iq_min': round(iq_lo, 3), 'iq_max': round(iq_hi, 3),
                'iq_median': round(float(iq_sub.median()), 3) if n else float('nan'),
            }
            for fname in ['F0', 'F1']:
                tau = taus.get(ck, {}).get(fname)
                if tau is not None:
                    rec = float((pd.to_numeric(sub, errors='coerce') > tau).mean()) if n else float('nan')
                else:
                    rec = float('nan')
                row[f'recall_{fname}'] = round(rec, 4) if pd.notna(rec) else float('nan')
                row[f'recall_{fname}_pct'] = round(100 * rec, 2) if pd.notna(rec) else float('nan')
            quartile_rows.append(row)

q_df = pd.DataFrame(quartile_rows)
q_df.to_csv(os.path.join(OUT, 'viso_recall_by_iq_quartile.csv'), index=False)
print(q_df[q_df['iq_feature'] == 'laplacian_var'].to_string(index=False))

# Collapse-threshold analysis: laplacian_var percentile at which recall falls below 50%
print('\nFinding laplacian_var collapse thresholds per ckpt...')
lap = merged['laplacian_var']
collapse_rows = []
for ck in CKPTS:
    score_col = f'score_{ck}'
    for fname in ['F0', 'F1']:
        tau = taus.get(ck, {}).get(fname)
        if tau is None:
            continue
        # Sweep percentiles 5, 10, 15, ... 95 as a cutoff (keep frames ABOVE cutoff)
        results = []
        for pct in range(5, 100, 5):
            cutoff = float(np.percentile(lap.dropna(), pct))
            mask_above = lap >= cutoff
            sub_scores = merged.loc[mask_above, score_col]
            sub_lap = lap[mask_above]
            n = int(mask_above.sum())
            rec = float((pd.to_numeric(sub_scores, errors='coerce') > tau).mean()) if n else float('nan')
            results.append({
                'ck': ck, 'tau_filter': fname,
                'lap_pct_cutoff': pct,
                'lap_value_cutoff': round(cutoff, 2),
                'n_fakes_kept': n,
                'pct_fakes_kept': round(100 * n / len(merged), 1),
                'recall_pct': round(100 * rec, 2) if pd.notna(rec) else float('nan'),
            })
        for r in results:
            collapse_rows.append(r)

coll_df = pd.DataFrame(collapse_rows)
coll_df.to_csv(os.path.join(OUT, 'viso_iq_recall_collapse.csv'), index=False)

# Print summary: collapse point for P8A at F0 and F1
for ck in ['P8A', 'E2B_3200']:
    for fname in ['F0', 'F1']:
        sub = coll_df[(coll_df['ck'] == ck) & (coll_df['tau_filter'] == fname)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values('lap_pct_cutoff')
        # Find first percentile cutoff where recall >= 70%
        above70 = sub[sub['recall_pct'] >= 70]
        below50_all = sub[sub['recall_pct'] < 50]
        # When keeping everything (pct_cutoff=5, drop bottom 5%), recall is?
        base_rec = sub[sub['lap_pct_cutoff'] == 5]['recall_pct'].values
        print(f'\n{ck} tau_{fname}:')
        print(f'  Recall keeping top-95% lap frames (drop bottom 5%): {base_rec[0] if len(base_rec) else "n/a"}%')
        if not above70.empty:
            r70 = above70.iloc[0]
            print(f'  First cutoff with recall>=70%: drop bottom {r70["lap_pct_cutoff"]}% '
                  f'(lap>={r70["lap_value_cutoff"]:.1f}), '
                  f'keep {r70["pct_fakes_kept"]:.0f}% of fakes')
        else:
            print('  No cutoff achieves recall>=70%')

# FINDINGS_FACTS
facts = ['# IQ Gating Analysis — FACTS (2026-05-04)\n\n']
facts.append(f'Joined {len(merged)} viso fakes with IQ attributes and per-ckpt scores.\n\n')
facts.append('## Per-laplacian-quartile recall (P8A and E2B at F0 tau)\n')
lap_q = q_df[q_df['iq_feature'] == 'laplacian_var'][['ck', 'quartile', 'n_fakes', 'iq_min', 'iq_max', 'recall_F0_pct', 'recall_F1_pct']]
facts.append(lap_q.to_string(index=False))
facts.append('\n\n')
facts.append('## Laplacian collapse thresholds (see viso_iq_recall_collapse.csv for full table)\n')
for ck in ['P8A', 'E2B_3200']:
    for fname in ['F0', 'F1']:
        sub = coll_df[(coll_df['ck'] == ck) & (coll_df['tau_filter'] == fname)]
        if sub.empty:
            continue
        facts.append(f'\n### {ck} tau_{fname}\n')
        facts.append(sub[['lap_pct_cutoff', 'lap_value_cutoff', 'n_fakes_kept', 'pct_fakes_kept', 'recall_pct']].to_string(index=False))
        facts.append('\n')

with open(os.path.join(OUT, 'FINDINGS_FACTS.txt'), 'w') as fh:
    fh.writelines(facts)

print(f'\nDone. Outputs in {OUT}')
