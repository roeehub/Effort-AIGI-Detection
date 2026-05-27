"""Build a deployment threshold table for the 3 deployment-candidate ckpts.

For each ckpt (T5C, P8A, E2B), report:
  - tau at FPR target on P2 (lowres+noface — production-realistic) substrate
  - tau at FPR target on F4 (chronic-6 dropped) substrate
  - per-suite recall at each tau
  - on the same row: the production-FPR you'd observe if you deploy at this tau
    over the FULL F0 pool (i.e., when no production filter is applied)
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import pandas as pd

ROOT = '/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training'
sys.path.insert(0, os.path.join(ROOT, 'analysis/substrate_cleaning_eval_2026-05-05'))
sys.path.insert(0, os.path.join(ROOT, 'analysis/best_candidate_search_2026-05-13'))
from run_clean_eval import MIN_WH_PX, extract_identity_from_video_id, load_parquet_meta  # noqa: E402
from stage1_run_all_candidates import CANDIDATES  # noqa: E402

OUT_DIR = os.path.join(ROOT, 'analysis/best_candidate_search_2026-05-13/_thresholds')
os.makedirs(OUT_DIR, exist_ok=True)

DEPLOY_CKPTS = ['T5C_step3500', 'P8A_step5000', 'E2B_step3200', 'P2D_fourier_step3000']
TARGET_FPRS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]


def annotate(real_csv, parquet_meta):
    df = pd.read_csv(real_csv)
    df['identity'] = df['video_id'].apply(extract_identity_from_video_id)
    merged = df.merge(parquet_meta, how='left',
                      left_on='frame_path', right_on='gcs_uri')
    merged['identity_pq'] = merged['identity_key'].astype(str).str.lower()
    merged.loc[merged['identity_key'].isna(), 'identity_pq'] = merged['identity']
    return merged


def keep_F4(d):
    chronic = {'bla_bla_chow','bla_bla_chow__s2','pc_generator__s22','pc_generator__s45','roy_d','q__s6'}
    is_chr = d['identity'].isin(chronic) | d['identity_pq'].isin(chronic)
    min_wh = pd.to_numeric(d['min_wh'], errors='coerce')
    is_lr = (min_wh < MIN_WH_PX).fillna(False)
    is_nf = d['is_no_face'].fillna(False).astype(bool)
    return ~(is_chr | is_lr | is_nf)


def keep_P2(d):
    min_wh = pd.to_numeric(d['min_wh'], errors='coerce')
    is_lr = (min_wh < MIN_WH_PX).fillna(False)
    is_nf = d['is_no_face'].fillna(False).astype(bool)
    return ~(is_lr | is_nf)


def main():
    parquet_meta = load_parquet_meta()
    cand_map = {c['ckpt_name']: c for c in CANDIDATES}
    out_rows = []
    for ckpt in DEPLOY_CKPTS:
        spec = cand_map[ckpt]
        real_csv = spec['real_csv']
        real = annotate(real_csv, parquet_meta)
        f4_mask = keep_F4(real)
        p2_mask = keep_P2(real)
        real_F4 = real[f4_mask]
        real_P2 = real[p2_mask]
        scores_F0 = pd.to_numeric(real['frame_prob'], errors='coerce').dropna().to_numpy()
        scores_F4 = pd.to_numeric(real_F4['frame_prob'], errors='coerce').dropna().to_numpy()
        scores_P2 = pd.to_numeric(real_P2['frame_prob'], errors='coerce').dropna().to_numpy()

        # Pre-read fakes
        fakes = {}
        for suite, p in spec['fake_csvs'].items():
            if os.path.exists(p):
                fdf = pd.read_csv(p)
                fakes[suite] = pd.to_numeric(fdf['frame_prob'], errors='coerce').dropna().to_numpy()

        for fpr in TARGET_FPRS:
            # F4 calibration
            tau_F4 = float(np.quantile(scores_F4, 1 - fpr))
            # P2 calibration (production-realistic)
            tau_P2 = float(np.quantile(scores_P2, 1 - fpr))
            for label, tau in [('F4', tau_F4), ('P2', tau_P2)]:
                row = {
                    'ckpt': ckpt,
                    'calibration_substrate': label,
                    'target_FPR_pct': round(fpr * 100, 2),
                    'tau': round(tau, 4),
                    'realized_FPR_F4_pct': round(100 * float((scores_F4 > tau).mean()), 2),
                    'realized_FPR_P2_pct': round(100 * float((scores_P2 > tau).mean()), 2),
                    'realized_FPR_F0_pct': round(100 * float((scores_F0 > tau).mean()), 2),
                }
                short_map = {
                    'visomaster_enhanced_macro_dev': 'viso',
                    'deeplive_enhanced_dev': 'dl',
                    'teams_fake_all_dev': 'tfake',
                }
                for suite, fs in fakes.items():
                    short = short_map.get(suite, suite)
                    row[f'recall_{short}_pct'] = round(100 * float((fs > tau).mean()), 2)
                # Macro
                recs = [v for k, v in row.items() if k.startswith('recall_')]
                row['recall_macro_pct'] = round(float(np.mean(recs)), 2)
                out_rows.append(row)

    df = pd.DataFrame(out_rows)
    out_csv = os.path.join(OUT_DIR, 'DEPLOYMENT_THRESHOLDS.csv')
    df.to_csv(out_csv, index=False)
    print(f'wrote {out_csv}')

    # Pretty print: for each ckpt, the P2-calibration table at typical FPR points
    for ckpt in DEPLOY_CKPTS:
        print(f'\n=== {ckpt} — calibrated on P2 (lowres+noface filter) ===')
        sub = df[(df.ckpt == ckpt) & (df.calibration_substrate == 'P2')]
        cols = ['target_FPR_pct','tau','realized_FPR_F4_pct','realized_FPR_F0_pct',
                'recall_viso_pct','recall_dl_pct','recall_tfake_pct','recall_macro_pct']
        print(sub[cols].to_string(index=False))
        print(f'\n=== {ckpt} — calibrated on F4 (chronic-6 dropped) ===')
        sub = df[(df.ckpt == ckpt) & (df.calibration_substrate == 'F4')]
        print(sub[cols].to_string(index=False))


if __name__ == '__main__':
    main()
