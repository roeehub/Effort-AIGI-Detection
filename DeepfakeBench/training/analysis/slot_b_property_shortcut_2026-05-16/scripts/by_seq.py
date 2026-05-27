"""Per-seq decomposition of dor_shkedi over-fires.

dor_shkedi/real_dor frames are tagged `..._seqNNN_...`. seq might cluster into
capture sessions or processing batches. Also check file-extension (.jpg vs .png)
since that often signals different processing pipelines.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / 'outputs'

SEQ_RE = re.compile(r'seq(\d+)')


def seq_of(basename: str):
    m = SEQ_RE.search(basename)
    return int(m.group(1)) if m else None


def ext_of(basename: str):
    return basename.lower().split('.')[-1]


def main():
    df = pd.read_csv(OUT / 'frames_with_props_and_scores.csv')
    df['seq'] = df['crop_basename'].apply(seq_of)
    df['ext'] = df['crop_basename'].apply(ext_of)

    print('=== File extension distribution per identity ===')
    print(df.groupby('identity')['ext'].value_counts())

    print('\n=== Slot β over-fire rate by extension per identity ===')
    for ident in ['dor_shkedi', 'real_dor']:
        sub = df[df['identity'] == ident]
        ext_break = sub.groupby('ext').agg(
            n=('SLOT_B_overfire', 'size'),
            n_over=('SLOT_B_overfire', 'sum'),
            mean_score=('SLOT_B', 'mean'),
            p95_score=('SLOT_B', lambda s: s.quantile(0.95)),
            sharpness_med=('sharpness_laplacian', 'median'),
            lab_b_dev_med=('lab_b_dev', 'median'),
            lab_a_std_med=('lab_a_std', 'median'),
            skin_frac_med=('skin_frac_hsv', 'median'),
            min_dim_med=('min_dim', 'median'),
            file_size_med=('file_size_bytes', 'median'),
        ).reset_index()
        ext_break['rate'] = ext_break['n_over'] / ext_break['n']
        print(f'\n[{ident}]')
        print(ext_break.to_string(index=False))

    # Bin seq into deciles and see if over-fires concentrate
    print('\n=== dor_shkedi: seq deciles ===')
    dor = df[df['identity'] == 'dor_shkedi'].copy()
    dor['seq_decile'] = pd.qcut(dor['seq'], q=10, duplicates='drop', labels=False)
    deciles = dor.groupby('seq_decile').agg(
        seq_min=('seq', 'min'),
        seq_max=('seq', 'max'),
        n=('SLOT_B_overfire', 'size'),
        n_over=('SLOT_B_overfire', 'sum'),
        mean_score=('SLOT_B', 'mean'),
        sharpness_med=('sharpness_laplacian', 'median'),
        lab_b_dev_med=('lab_b_dev', 'median'),
        ext_mix=('ext', lambda s: s.value_counts().to_dict()),
    ).reset_index()
    deciles['rate'] = deciles['n_over'] / deciles['n']
    print(deciles.to_string(index=False))

    print('\n=== real_dor: seq deciles ===')
    rd = df[df['identity'] == 'real_dor'].copy()
    rd['seq_decile'] = pd.qcut(rd['seq'], q=10, duplicates='drop', labels=False)
    deciles_rd = rd.groupby('seq_decile').agg(
        seq_min=('seq', 'min'),
        seq_max=('seq', 'max'),
        n=('SLOT_B_overfire', 'size'),
        mean_score=('SLOT_B', 'mean'),
        sharpness_med=('sharpness_laplacian', 'median'),
        lab_b_dev_med=('lab_b_dev', 'median'),
    ).reset_index()
    print(deciles_rd.to_string(index=False))


if __name__ == '__main__':
    main()
