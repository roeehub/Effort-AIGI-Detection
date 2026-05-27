"""Aggregate properties by (identity × Slot-β over-fire status) and look for
separating axes.

Primary contrast: Slot β over-firing frames (n=124) vs Slot β non-over-firing
frames (n=1294). Secondary: dor_shkedi-over vs dor_shkedi-non-over vs real_dor.
"""
from __future__ import annotations
import re
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / 'outputs'

TAUS = {'P8A': 0.916, 'T5C': 0.831, 'SLOT_B': 0.816}


def basename_from_gcs(p: str) -> str:
    return os.path.basename(p)


def main():
    props = pd.read_csv(OUT / 'frame_properties.csv')
    idx = pd.read_csv(OUT / 'frame_index.csv')
    idx['crop_basename'] = idx['frame_path'].apply(basename_from_gcs)

    merged = idx.merge(props, on='crop_basename', how='inner', suffixes=('_idx', ''))
    print(f'Merged shape: {merged.shape}  (expect ~1418)')

    # Mark over-fire status per ckpt
    for ckpt, tau in TAUS.items():
        merged[f'{ckpt}_overfire'] = (merged[ckpt] >= tau).astype(int)

    # Headline cohort: Slot β over-fire vs not
    print('\n=== Slot β over-fire counts ===')
    print(merged['SLOT_B_overfire'].value_counts())

    PROPS = ['sharpness_laplacian', 'luma_mean', 'luma_std', 'lab_a_dev', 'lab_b_dev',
             'lab_a_std', 'lab_b_std', 'skin_frac_hsv', 'edge_density',
             'min_dim', 'file_size_bytes']

    # Contrast 1: Slot β over-fire vs non-over-fire (ALL identities)
    print('\n=== Contrast 1: Slot β over-fire (n=124) vs non-over-fire (n=1294) — ALL identities ===')
    over = merged[merged['SLOT_B_overfire'] == 1]
    notover = merged[merged['SLOT_B_overfire'] == 0]
    rows = []
    for p in PROPS:
        u = stats.mannwhitneyu(over[p], notover[p], alternative='two-sided')
        # Cohen's d (rough)
        pooled_sd = np.sqrt((over[p].var() + notover[p].var()) / 2)
        d = (over[p].mean() - notover[p].mean()) / pooled_sd if pooled_sd > 0 else 0
        rows.append({'property': p,
                     'over_median': over[p].median(),
                     'notover_median': notover[p].median(),
                     'delta_median': over[p].median() - notover[p].median(),
                     'over_mean': over[p].mean(),
                     'notover_mean': notover[p].mean(),
                     'cohens_d': d,
                     'mw_p': u.pvalue})
    c1 = pd.DataFrame(rows).sort_values('cohens_d', key=abs, ascending=False)
    print(c1.to_string(index=False))
    c1.to_csv(OUT / 'contrast_slot_b_overfire_vs_not_ALL.csv', index=False)

    # Contrast 2: dor_shkedi over-fire vs dor_shkedi non-over-fire (within same identity)
    print('\n=== Contrast 2: dor_shkedi over (n=116) vs dor_shkedi non-over (n=1054) — same identity ===')
    dor = merged[merged['identity'] == 'dor_shkedi']
    over_d = dor[dor['SLOT_B_overfire'] == 1]
    notover_d = dor[dor['SLOT_B_overfire'] == 0]
    rows = []
    for p in PROPS:
        u = stats.mannwhitneyu(over_d[p], notover_d[p], alternative='two-sided')
        pooled_sd = np.sqrt((over_d[p].var() + notover_d[p].var()) / 2)
        d = (over_d[p].mean() - notover_d[p].mean()) / pooled_sd if pooled_sd > 0 else 0
        rows.append({'property': p,
                     'over_median': over_d[p].median(),
                     'notover_median': notover_d[p].median(),
                     'delta_median': over_d[p].median() - notover_d[p].median(),
                     'over_mean': over_d[p].mean(),
                     'notover_mean': notover_d[p].mean(),
                     'cohens_d': d,
                     'mw_p': u.pvalue})
    c2 = pd.DataFrame(rows).sort_values('cohens_d', key=abs, ascending=False)
    print(c2.to_string(index=False))
    c2.to_csv(OUT / 'contrast_dor_shkedi_overfire_vs_not.csv', index=False)

    # Contrast 3: dor_shkedi (all) vs real_dor — same person, different tag
    print('\n=== Contrast 3: dor_shkedi (n=1170) vs real_dor (n=109) — same person, different tag ===')
    real_dor = merged[merged['identity'] == 'real_dor']
    rows = []
    for p in PROPS:
        u = stats.mannwhitneyu(dor[p], real_dor[p], alternative='two-sided')
        pooled_sd = np.sqrt((dor[p].var() + real_dor[p].var()) / 2)
        d = (dor[p].mean() - real_dor[p].mean()) / pooled_sd if pooled_sd > 0 else 0
        rows.append({'property': p,
                     'dor_median': dor[p].median(),
                     'real_dor_median': real_dor[p].median(),
                     'delta_median': dor[p].median() - real_dor[p].median(),
                     'dor_mean': dor[p].mean(),
                     'real_dor_mean': real_dor[p].mean(),
                     'cohens_d': d,
                     'mw_p': u.pvalue})
    c3 = pd.DataFrame(rows).sort_values('cohens_d', key=abs, ascending=False)
    print(c3.to_string(index=False))
    c3.to_csv(OUT / 'contrast_dor_shkedi_vs_real_dor.csv', index=False)

    # Per-identity property medians (overview)
    print('\n=== Per-identity property medians ===')
    overview = merged.groupby('identity')[PROPS].median().reset_index()
    print(overview.to_string(index=False))
    overview.to_csv(OUT / 'per_identity_property_medians.csv', index=False)

    # Per-identity Slot β over-fire rate
    print('\n=== Per-identity Slot β over-fire rate ===')
    rate = merged.groupby('identity').agg(
        n_frames=('SLOT_B_overfire', 'size'),
        n_overfires=('SLOT_B_overfire', 'sum'),
    ).reset_index()
    rate['rate'] = rate['n_overfires'] / rate['n_frames']
    print(rate.to_string(index=False))
    rate.to_csv(OUT / 'per_identity_slot_b_overfire_rate.csv', index=False)

    # Save merged for downstream cross-check
    merged.to_csv(OUT / 'frames_with_props_and_scores.csv', index=False)


if __name__ == '__main__':
    main()
