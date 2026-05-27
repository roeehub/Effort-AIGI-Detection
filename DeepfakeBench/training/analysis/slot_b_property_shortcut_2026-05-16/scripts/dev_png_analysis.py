"""Property analysis on the 3 PNG identities in teams_real_all_dev:
Roy_D (79% over-fire), ilan (0%), orel (0%).
"""
from __future__ import annotations
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent.parent
CACHE = HERE / 'dev_cache'
OUT = HERE / 'outputs'

import sys
sys.path.insert(0, str(HERE / 'scripts'))
from compute_properties import compute_props, extract_identity_from_path


def main():
    cache_files = sorted(CACHE.glob('*'))
    print(f'Cache has {len(cache_files)} dev PNG files')

    rows = []
    for fp in cache_files:
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            continue
        props = compute_props(img, fp.stat().st_size)
        props['crop_basename'] = fp.name
        props['identity'] = extract_identity_from_path(fp.name)
        rows.append(props)

    pdf = pd.DataFrame(rows)
    pdf.to_csv(OUT / 'dev_png_frame_properties.csv', index=False)

    # Merge with scores
    scores = pd.read_csv('dev_inputs/dev_combined.csv')
    scores['crop_basename'] = scores['frame_path'].apply(os.path.basename)
    scores_pivot = scores.pivot_table(index='crop_basename', columns='ckpt',
                                       values='frame_prob', aggfunc='first').reset_index()
    merged = pdf.merge(scores_pivot, on='crop_basename', how='inner')

    TAUS = {'P8A': 0.916, 'T5C': 0.831, 'SLOT_B': 0.816}
    for c, t in TAUS.items():
        merged[f'{c}_overfire'] = (merged[c] >= t).astype(int)

    merged.to_csv(OUT / 'dev_png_frames_with_props_and_scores.csv', index=False)
    print(f'\nMerged: {merged.shape}')

    PROPS = ['sharpness_laplacian', 'luma_mean', 'luma_std', 'lab_a_dev', 'lab_b_dev',
             'lab_a_std', 'lab_b_std', 'skin_frac_hsv', 'edge_density',
             'min_dim', 'file_size_bytes']

    print('\n=== Per-identity property medians (dev PNG) ===')
    med = merged.groupby('identity')[PROPS].median().reset_index()
    print(med.to_string(index=False))
    med.to_csv(OUT / 'dev_png_per_identity_medians.csv', index=False)

    # Also crop dimensions
    print('\n=== Per-identity crop dimension distribution ===')
    dim = merged.groupby('identity').agg(
        n=('width', 'size'),
        w_mean=('width', 'mean'),
        w_std=('width', 'std'),
        h_mean=('height', 'mean'),
        h_std=('height', 'std'),
        aspect_mean=('width', lambda s: (merged.loc[s.index, 'width'] / merged.loc[s.index, 'height']).mean()),
    ).reset_index()
    print(dim.to_string(index=False))

    # Roy_D over vs non-over contrast
    print('\n=== Roy_D Slot β over (n=103) vs non-over (n=27) ===')
    roy = merged[merged['identity'] == 'Roy_D']
    over = roy[roy['SLOT_B_overfire'] == 1]
    notover = roy[roy['SLOT_B_overfire'] == 0]
    rows_c = []
    for p in PROPS:
        try:
            u = stats.mannwhitneyu(over[p], notover[p], alternative='two-sided')
            pval = u.pvalue
        except:
            pval = np.nan
        pooled_sd = np.sqrt((over[p].var() + notover[p].var()) / 2)
        d = (over[p].mean() - notover[p].mean()) / pooled_sd if pooled_sd > 0 else 0
        rows_c.append({'property': p,
                       'over_med': over[p].median(),
                       'notover_med': notover[p].median(),
                       'over_mean': over[p].mean(),
                       'notover_mean': notover[p].mean(),
                       'cohens_d': d,
                       'mw_p': pval})
    c = pd.DataFrame(rows_c).sort_values('cohens_d', key=abs, ascending=False)
    print(c.to_string(index=False))
    c.to_csv(OUT / 'dev_png_roy_d_overfire_contrast.csv', index=False)

    # Cross-identity: Roy_D (over-firing) vs ilan + orel (clean) — what differs?
    print('\n=== Roy_D (n=130) vs ilan+orel (n=64) cross-identity property contrast ===')
    roy_all = merged[merged['identity'] == 'Roy_D']
    clean = merged[merged['identity'].isin(['ilan', 'orel'])]
    rows_c2 = []
    for p in PROPS:
        try:
            u = stats.mannwhitneyu(roy_all[p], clean[p], alternative='two-sided')
            pval = u.pvalue
        except:
            pval = np.nan
        pooled_sd = np.sqrt((roy_all[p].var() + clean[p].var()) / 2)
        d = (roy_all[p].mean() - clean[p].mean()) / pooled_sd if pooled_sd > 0 else 0
        rows_c2.append({'property': p,
                        'roy_med': roy_all[p].median(),
                        'clean_med': clean[p].median(),
                        'roy_mean': roy_all[p].mean(),
                        'clean_mean': clean[p].mean(),
                        'cohens_d': d,
                        'mw_p': pval})
    c2 = pd.DataFrame(rows_c2).sort_values('cohens_d', key=abs, ascending=False)
    print(c2.to_string(index=False))
    c2.to_csv(OUT / 'dev_png_roy_d_vs_clean_contrast.csv', index=False)

    # Compare Roy_D to dor_shkedi.png (the lockbox PNG over-firing identity)
    lockbox = pd.read_csv(OUT / 'frames_with_props_and_scores.csv')
    dor_png = lockbox[(lockbox['identity'] == 'dor_shkedi')
                       & (lockbox['crop_basename'].str.endswith('.png'))]
    print('\n=== Roy_D dev PNG (n=130) vs dor_shkedi lockbox PNG (n=895) — both over-firing on Slot β ===')
    rows_c3 = []
    for p in PROPS:
        try:
            u = stats.mannwhitneyu(roy_all[p], dor_png[p], alternative='two-sided')
            pval = u.pvalue
        except:
            pval = np.nan
        pooled_sd = np.sqrt((roy_all[p].var() + dor_png[p].var()) / 2)
        d = (roy_all[p].mean() - dor_png[p].mean()) / pooled_sd if pooled_sd > 0 else 0
        rows_c3.append({'property': p,
                        'roy_med': roy_all[p].median(),
                        'dor_png_med': dor_png[p].median(),
                        'cohens_d': d,
                        'mw_p': pval})
    c3 = pd.DataFrame(rows_c3).sort_values('cohens_d', key=abs, ascending=False)
    print(c3.to_string(index=False))


if __name__ == '__main__':
    main()
