"""Stage 2: Quantify how extreme each chronic-6 identity is and rank them.

For each chronic identity, compute:
  - n frames in teams_real_all_dev
  - per-ckpt FPR contribution at deployment tau (over a few reference ckpts)
  - IQ extremeness: distance to non-chronic real-pool distribution on
    (face_pixel_area, sharpness_laplacian, min_dim) — Mahalanobis-like z-score
  - dominant capture_mode

Output: identity_extremeness_table.csv with a single "extremeness" rank.

The goal is to answer: "if we drop just 1, 2, or 3 of the chronic identities,
which should we drop and what's the FPR/recall payoff?"
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np
import pandas as pd

ROOT = '/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training'
sys.path.insert(0, os.path.join(ROOT, 'analysis/substrate_cleaning_eval_2026-05-05'))
from run_clean_eval import (  # type: ignore  # noqa: E402
    CHRONIC_6, MIN_WH_PX, extract_identity_from_video_id, load_parquet_meta,
)

CHRONIC_FP = os.path.join(ROOT, 'analysis/chronic6_fingerprint_2026-05-05/per_identity_feature_medians.csv')
OUT_DIR = os.path.join(ROOT, 'analysis/best_candidate_search_2026-05-13/_stage2')
os.makedirs(OUT_DIR, exist_ok=True)

# Reference ckpts to compute per-identity FPR contribution.
# Pick a spread of recipes.
REF_CKPTS = [
    ('P8A',        os.path.join(ROOT, 'analysis/cpu_followups_2026-05-04/raw_reports/teams_real_all_dev_p8a_reference_step5000_frames_report.csv')),
    ('E2B',        os.path.join(ROOT, 'analysis/cpu_followups_2026-05-04/raw_reports/teams_real_all_dev_e2b_top_n_step3200_frames_report.csv')),
    ('E3_6600',    os.path.join(ROOT, 'analysis/cpu_followups_2026-05-04/raw_reports/teams_real_all_dev_e3_top_n_step6600_frames_report.csv')),
    ('T3_S1_1500', os.path.join(ROOT, 'analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step1500/teams_real_all_dev_t3_slot1_periodic_step1500_frames_report.csv')),
    ('T5C_3500',   os.path.join(ROOT, 'analysis/cpu_diagnostics_2026-05-12_stage_a/_scorecard_reports/teams_real_all_dev_t5c_periodic_step3500_frames_report.csv')),
    ('PA_5600',    os.path.join(ROOT, 'analysis/pa_pc_eval_2026-05-05/raw_reports/teams_real_all_dev_pa_top_n_step5600_frames_report.csv')),
    ('SLOT1_step1500', os.path.join(ROOT, 'analysis/r13_overnight_slot1_shift_analysis_2026-05-13/_frame_cache/teams_real_all_dev_slot1_lora_p8a_periodic_step1500_frames_report.csv')),
]

CHRONIC_PRINTABLE = {
    'bla_bla_chow':       'bla_bla_chow (n>200, screen, sharp~525)',
    'bla_bla_chow__s2':   'bla_bla_chow__s2 (phone_screen, BLURRY 55 Laplacian)',
    'pc_generator__s22':  'pc_generator__s22 (webcam, tiny 88x94 face)',
    'pc_generator__s45':  'pc_generator__s45 (webcam, tiny 90x97 face)',
    'roy_d':              'roy_d (need to characterize)',
    'q__s6':              'q__s6 (webcam, tiny 94x102 face)',
}


def annotate(real_csv: str, parquet_meta: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(real_csv)
    df['identity'] = df['video_id'].apply(extract_identity_from_video_id)
    merged = df.merge(parquet_meta, how='left',
                      left_on='frame_path', right_on='gcs_uri')
    merged['identity_pq'] = merged['identity_key'].astype(str).str.lower()
    merged.loc[merged['identity_key'].isna(), 'identity_pq'] = merged['identity']
    return merged


def main():
    parquet_meta = load_parquet_meta()
    print(f'parquet rows: {len(parquet_meta):,}')

    # Annotate the canonical real pool to extract per-identity frame counts +
    # IQ stats.
    canon = annotate(REF_CKPTS[0][1], parquet_meta)
    print(f'canonical real pool size: {len(canon)}')

    # Build per-identity row.
    rows = []
    chronic_set = set(CHRONIC_6)
    canon['identity_final'] = canon['identity_pq']
    canon.loc[canon['identity_final'].isin(['nan','unk',''])
              | canon['identity_final'].isnull(),
              'identity_final'] = canon['identity']

    n_total = len(canon)
    is_chronic_mask = canon['identity_final'].isin(chronic_set) | canon['identity'].isin(chronic_set)
    n_chronic = int(is_chronic_mask.sum())
    n_nonchronic = n_total - n_chronic
    print(f'chronic frames: {n_chronic} ({100*n_chronic/n_total:.1f}%) / non-chronic: {n_nonchronic}')

    # Per-identity stats
    for ident in CHRONIC_6:
        mask = (canon['identity_final'] == ident) | (canon['identity'] == ident)
        sub = canon[mask]
        row = {
            'identity': ident,
            'description': CHRONIC_PRINTABLE.get(ident, ident),
            'n_frames_dev': int(len(sub)),
            'pct_of_dev_pool': round(100 * len(sub) / n_total, 2),
        }
        # IQ proxies (we only have width/height in parquet by default).
        wh_min = pd.to_numeric(sub['min_wh'], errors='coerce')
        row['min_wh_p50'] = float(wh_min.median()) if len(wh_min) else None
        row['min_wh_lt200_pct'] = round(100 * (wh_min < 200).fillna(False).mean(), 1) if len(wh_min) else None
        row['no_face_pct'] = round(100 * sub['is_no_face'].fillna(False).astype(bool).mean(), 1) if len(sub) else None
        rows.append(row)

    # Add per-ckpt FPR contribution at deployment tau (tau set to FPR=10% on full F0)
    for ckpt_name, real_csv in REF_CKPTS:
        if not os.path.exists(real_csv):
            print(f'  skip {ckpt_name}: missing {real_csv}')
            continue
        df = annotate(real_csv, parquet_meta)
        df['identity_final'] = df['identity_pq']
        df.loc[df['identity_final'].isin(['nan','unk',''])
              | df['identity_final'].isnull(),
              'identity_final'] = df['identity']
        scores = pd.to_numeric(df['frame_prob'], errors='coerce')
        # Compute tau such that overall F0 FPR == 10%
        tau = float(np.quantile(scores.dropna(), 0.90))
        # Total FPR at this tau (sanity = 10%)
        global_fpr = float((scores > tau).mean())
        # For each chronic identity: fraction of frames > tau (this identity's FPR)
        # AND fraction of total positives that come from this identity
        n_positives_total = int((scores > tau).sum())
        for r in rows:
            ident = r['identity']
            mask = (df['identity_final'] == ident) | (df['identity'] == ident)
            sub_scores = scores[mask]
            n_pos = int((sub_scores > tau).sum())
            r[f'{ckpt_name}_tau'] = round(tau, 4)
            r[f'{ckpt_name}_n_frames'] = int(mask.sum())
            r[f'{ckpt_name}_fpr_pct'] = round(100 * n_pos / max(1, int(mask.sum())), 1)
            r[f'{ckpt_name}_share_of_total_pos_pct'] = round(100 * n_pos / max(1, n_positives_total), 1)

    # Output
    df_out = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, 'CHRONIC_IDENTITY_EXTREMENESS.csv')
    df_out.to_csv(out_csv, index=False)
    print(f'\nwrote {out_csv}')
    # Pretty print
    print('\n=== CHRONIC IDENTITY EXTREMENESS ===')
    cols = ['identity', 'n_frames_dev', 'pct_of_dev_pool', 'min_wh_p50', 'min_wh_lt200_pct']
    for ckpt_name, _ in REF_CKPTS:
        cols.append(f'{ckpt_name}_fpr_pct')
    print(df_out[cols].to_string(index=False))
    print()
    print('=== SHARE OF TOTAL FPR-FRAMES ===')
    cols = ['identity'] + [f'{ck}_share_of_total_pos_pct' for ck,_ in REF_CKPTS]
    print(df_out[cols].to_string(index=False))


if __name__ == '__main__':
    main()
