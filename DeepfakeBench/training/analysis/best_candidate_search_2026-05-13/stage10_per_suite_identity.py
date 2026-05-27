"""Stage 10: Per-fake-identity caught/missed breakdown at recommended tau.

For T5C @ tau=0.50 (the recommended deployment config), show each fake
identity's per-identity frame-fraction-above-tau. This reveals:
  - Which fake identities are caught with huge margin (frac → 1.0)
  - Which are marginal (frac near 0.5)
  - If any fake identity is missed (frac < 0.5)

Also for each real identity (P2 substrate), show frame-fraction. Highlights
which real identities are closest to the 0.5 boundary.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

ROOT = '/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training'
sys.path.insert(0, os.path.join(ROOT, 'analysis/substrate_cleaning_eval_2026-05-05'))
sys.path.insert(0, os.path.join(ROOT, 'analysis/best_candidate_search_2026-05-13'))
from run_clean_eval import MIN_WH_PX, load_parquet_meta  # noqa: E402
from stage6_per_identity_majority import CK_FILES, CHRONIC, base_identity  # noqa: E402

OUT_DIR = os.path.join(ROOT, 'analysis/best_candidate_search_2026-05-13/_stage10')
os.makedirs(OUT_DIR, exist_ok=True)

TAU_RECS = {
    'T5C_step3500':         0.50,
    'P2D_fourier_step3000': 0.20,
    'P8A_step5000':         0.05,
    'E2B_step3200':         0.50,
}


def load_anno(csv_path):
    df = pd.read_csv(csv_path)
    df['identity'] = df['video_id'].astype(str).apply(base_identity)
    df['frame_prob'] = pd.to_numeric(df['frame_prob'], errors='coerce')
    return df


def add_substrate(real_df, parquet_meta):
    a = real_df.merge(parquet_meta, how='left', left_on='frame_path', right_on='gcs_uri')
    min_wh = pd.to_numeric(a['min_wh'], errors='coerce')
    is_lr = (min_wh < MIN_WH_PX).fillna(False)
    is_nf = a['is_no_face'].fillna(False).astype(bool)
    a['keep_P2'] = ~(is_lr | is_nf)
    a['keep_F4'] = a['keep_P2'] & ~a['identity'].isin(CHRONIC)
    return a


def per_id_summary(df, tau, label):
    s = df.groupby('identity').agg(
        n_frames=('frame_prob','count'),
        mean_prob=('frame_prob','mean'),
        frac_above_tau=('frame_prob', lambda x: float((x > tau).mean())),
    ).reset_index()
    s['label'] = label
    s['flagged_fake'] = s['frac_above_tau'] > 0.5
    return s


def main():
    parquet_meta = load_parquet_meta()
    all_rows = []
    for ckpt, csvs in CK_FILES.items():
        tau = TAU_RECS[ckpt]
        real = add_substrate(load_anno(csvs['teams_real_all_dev']), parquet_meta)
        real_F4 = real[real['keep_F4']]
        real_P2 = real[real['keep_P2']]
        real_summary_F4 = per_id_summary(real_F4, tau, label='real_F4')
        real_summary_P2 = per_id_summary(real_P2, tau, label='real_P2')

        fake_summaries = []
        for s in ['visomaster_enhanced_macro_dev','deeplive_enhanced_dev','teams_fake_all_dev']:
            fdf = load_anno(csvs[s])
            fs = per_id_summary(fdf, tau, label=f'fake_{s}')
            fake_summaries.append(fs)

        ckpt_df = pd.concat([real_summary_F4, real_summary_P2] + fake_summaries, ignore_index=True)
        ckpt_df['ckpt'] = ckpt
        ckpt_df['tau'] = tau
        all_rows.append(ckpt_df)

        print(f'\n=== {ckpt} @ tau={tau:.2f} ===')
        # Real F4 — sorted desc by frac
        rf4 = real_summary_F4.sort_values('frac_above_tau', ascending=False)
        print(f'\nReal (F4 substrate, n={len(rf4)} identities, max frac={rf4["frac_above_tau"].max():.3f}):')
        print(rf4[['identity','n_frames','frac_above_tau','mean_prob','flagged_fake']].to_string(index=False))
        # Real P2-only (the chronic-cohort that survives G2)
        rp2_extra = real_summary_P2[~real_summary_P2['identity'].isin(rf4['identity'])]
        if len(rp2_extra):
            print(f'\nReal P2-only (chronic survivors of G2, n={len(rp2_extra)}):')
            print(rp2_extra[['identity','n_frames','frac_above_tau','mean_prob','flagged_fake']].sort_values('frac_above_tau', ascending=False).to_string(index=False))
        for fs in fake_summaries:
            suite = fs['label'].iloc[0].replace('fake_','')
            print(f'\nFake [{suite}] (n={len(fs)} identities):')
            print(fs[['identity','n_frames','frac_above_tau','mean_prob','flagged_fake']].sort_values('frac_above_tau').to_string(index=False))

    df = pd.concat(all_rows, ignore_index=True)
    out_csv = os.path.join(OUT_DIR, 'STAGE10_PER_SUITE.csv')
    df.to_csv(out_csv, index=False)
    print(f'\nwrote {out_csv}')


if __name__ == '__main__':
    main()
