"""Test for band-shortcuts in property axes.

A band shortcut means: over-fire rate is high in a SPECIFIC RANGE of property
values, not monotonically increasing/decreasing. The previous Cohen's d and
Mann-Whitney tests would miss this. Per-quantile binning detects it.

Pool: all PNG frames (lockbox 1004 + dev 194 = 1198), with their Slot β scores.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import os

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / 'outputs'

TAU_SLOT_B = 0.816


def main():
    # Load lockbox PNG frames
    lockbox = pd.read_csv(OUT / 'frames_with_props_and_scores.csv')
    lockbox['ext'] = lockbox['crop_basename'].str.lower().str.rsplit('.', n=1).str[-1]
    lockbox_png = lockbox[lockbox['ext'] == 'png'].copy()
    lockbox_png['source'] = 'lockbox'

    # Load dev PNG frames
    dev = pd.read_csv(OUT / 'dev_png_frames_with_props_and_scores.csv')
    dev['source'] = 'dev'

    # Unify columns
    cols = ['crop_basename', 'identity', 'source',
            'sharpness_laplacian', 'luma_mean', 'luma_std', 'lab_a_dev', 'lab_b_dev',
            'lab_a_std', 'lab_b_std', 'skin_frac_hsv', 'edge_density',
            'min_dim', 'file_size_bytes', 'SLOT_B', 'P8A', 'T5C']
    pool = pd.concat([lockbox_png[cols], dev[cols]], ignore_index=True)
    pool['SLOT_B_overfire'] = (pool['SLOT_B'] >= TAU_SLOT_B).astype(int)
    pool['P8A_overfire'] = (pool['P8A'] >= 0.916).astype(int)
    pool['T5C_overfire'] = (pool['T5C'] >= 0.831).astype(int)
    pool.to_csv(OUT / 'all_png_pool.csv', index=False)

    print(f'Total PNG pool: {len(pool)}')
    print(f'Slot β over-fires: {pool["SLOT_B_overfire"].sum()} ({pool["SLOT_B_overfire"].mean()*100:.2f}%)')
    print(f'T5C over-fires: {pool["T5C_overfire"].sum()} ({pool["T5C_overfire"].mean()*100:.2f}%)')
    print(f'P8A over-fires: {pool["P8A_overfire"].sum()} ({pool["P8A_overfire"].mean()*100:.2f}%)')

    PROPS = ['sharpness_laplacian', 'luma_mean', 'luma_std', 'lab_a_dev', 'lab_b_dev',
             'lab_a_std', 'lab_b_std', 'skin_frac_hsv', 'edge_density',
             'min_dim', 'file_size_bytes']

    # Per-decile over-fire rate for Slot β
    print('\n=== Per-decile Slot β over-fire rate per property ===')
    decile_rows = []
    for p in PROPS:
        pool['decile'] = pd.qcut(pool[p], q=10, duplicates='drop', labels=False)
        bins = pool.groupby('decile').agg(
            n=('SLOT_B_overfire', 'size'),
            n_over=('SLOT_B_overfire', 'sum'),
            prop_min=(p, 'min'),
            prop_max=(p, 'max'),
            prop_med=(p, 'median'),
        ).reset_index()
        bins['rate'] = bins['n_over'] / bins['n']
        bins['property'] = p
        decile_rows.append(bins)
        print(f'\n[{p}]')
        print(bins[['decile', 'prop_min', 'prop_max', 'prop_med', 'n', 'n_over', 'rate']].to_string(index=False))

    all_deciles = pd.concat(decile_rows, ignore_index=True)
    all_deciles.to_csv(OUT / 'per_decile_overfire_rate.csv', index=False)

    # Same for T5C and P8A
    print('\n=== Per-decile T5C over-fire rate per property (only top-3 noisiest) ===')
    for p in ['sharpness_laplacian', 'lab_b_dev', 'luma_mean']:
        pool['decile'] = pd.qcut(pool[p], q=10, duplicates='drop', labels=False)
        bins = pool.groupby('decile').agg(
            n=('T5C_overfire', 'size'),
            n_over=('T5C_overfire', 'sum'),
            prop_min=(p, 'min'),
            prop_max=(p, 'max'),
            prop_med=(p, 'median'),
        ).reset_index()
        bins['rate'] = bins['n_over'] / bins['n']
        print(f'\n[T5C × {p}]')
        print(bins[['decile', 'prop_min', 'prop_max', 'prop_med', 'n', 'n_over', 'rate']].to_string(index=False))

    print('\n=== Per-decile P8A over-fire rate per property (only top-3 noisiest) ===')
    for p in ['sharpness_laplacian', 'lab_b_dev', 'luma_mean']:
        pool['decile'] = pd.qcut(pool[p], q=10, duplicates='drop', labels=False)
        bins = pool.groupby('decile').agg(
            n=('P8A_overfire', 'size'),
            n_over=('P8A_overfire', 'sum'),
            prop_min=(p, 'min'),
            prop_max=(p, 'max'),
            prop_med=(p, 'median'),
        ).reset_index()
        bins['rate'] = bins['n_over'] / bins['n']
        print(f'\n[P8A × {p}]')
        print(bins[['decile', 'prop_min', 'prop_max', 'prop_med', 'n', 'n_over', 'rate']].to_string(index=False))


if __name__ == '__main__':
    main()
