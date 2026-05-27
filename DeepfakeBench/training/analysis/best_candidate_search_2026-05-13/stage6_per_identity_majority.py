"""Stage 6: Per-identity majority-vote aggregation.

In production, many frames are captured per identity/session, and the
identity is flagged as fake when >50% of its frames pass the per-frame
threshold tau. This evaluation matches that aggregation.

For each candidate ckpt × tau:
  - Compute per-identity fraction(frame_prob > tau)
  - identity is "fake_flagged" if fraction > 0.5
  - Identity-level FPR = fake_flagged_real_identities / total_real_identities
  - Identity-level recall = fake_flagged_fake_identities / total_fake_identities

Outputs: STAGE6_PER_IDENTITY.csv + summary printout.
"""
from __future__ import annotations
import os
import re
import sys
import numpy as np
import pandas as pd

ROOT = '/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training'
sys.path.insert(0, os.path.join(ROOT, 'analysis/substrate_cleaning_eval_2026-05-05'))
from run_clean_eval import MIN_WH_PX, extract_identity_from_video_id, load_parquet_meta  # noqa: E402

OUT_DIR = os.path.join(ROOT, 'analysis/best_candidate_search_2026-05-13/_stage6')
os.makedirs(OUT_DIR, exist_ok=True)


def base_identity(vid):
    s = re.sub(r'__seg_[\d.]+', '', vid or '')
    s = re.sub(r'__seq_?\d+', '', s)
    s = re.sub(r'__(?:real|fake)$', '', s)
    s = re.sub(r'__frame_\d+_crop_\d+__[a-f0-9]+', '', s)
    return s.strip('_').lower() or 'UNK'


# Candidate -> per-suite CSV paths
CK_FILES = {
    'T5C_step3500': {
        'teams_real_all_dev': 'analysis/cpu_diagnostics_2026-05-12_stage_a/_scorecard_reports/teams_real_all_dev_t5c_periodic_step3500_frames_report.csv',
        'visomaster_enhanced_macro_dev': 'analysis/cpu_diagnostics_2026-05-12_stage_a/_scorecard_reports/visomaster_enhanced_macro_dev_t5c_periodic_step3500_frames_report.csv',
        'deeplive_enhanced_dev': 'analysis/cpu_diagnostics_2026-05-12_stage_a/_scorecard_reports/deeplive_enhanced_dev_t5c_periodic_step3500_frames_report.csv',
        'teams_fake_all_dev': 'analysis/cpu_diagnostics_2026-05-12_stage_a/_scorecard_reports/teams_fake_all_dev_t5c_periodic_step3500_frames_report.csv',
    },
    'P8A_step5000': {
        'teams_real_all_dev': 'analysis/cpu_followups_2026-05-04/raw_reports/teams_real_all_dev_p8a_reference_step5000_frames_report.csv',
        'visomaster_enhanced_macro_dev': 'analysis/cpu_followups_2026-05-04/raw_reports/visomaster_enhanced_macro_dev_p8a_reference_step5000_frames_report.csv',
        'deeplive_enhanced_dev': 'analysis/cpu_followups_2026-05-04/raw_reports/deeplive_enhanced_dev_p8a_reference_step5000_frames_report.csv',
        'teams_fake_all_dev': 'analysis/cpu_followups_2026-05-04/raw_reports/teams_fake_all_dev_p8a_reference_step5000_frames_report.csv',
    },
    'E2B_step3200': {
        'teams_real_all_dev': 'analysis/cpu_followups_2026-05-04/raw_reports/teams_real_all_dev_e2b_top_n_step3200_frames_report.csv',
        'visomaster_enhanced_macro_dev': 'analysis/cpu_followups_2026-05-04/raw_reports/visomaster_enhanced_macro_dev_e2b_top_n_step3200_frames_report.csv',
        'deeplive_enhanced_dev': 'analysis/cpu_followups_2026-05-04/raw_reports/deeplive_enhanced_dev_e2b_top_n_step3200_frames_report.csv',
        'teams_fake_all_dev': 'analysis/cpu_followups_2026-05-04/raw_reports/teams_fake_all_dev_e2b_top_n_step3200_frames_report.csv',
    },
    'P2D_fourier_step3000': {
        'teams_real_all_dev': 'analysis/p2_eval_2026-05-08/p2_deeper_analysis/frame_reports/teams_real_all_dev_p2_d_fourier_periodic_step3000_frames_report.csv',
        'visomaster_enhanced_macro_dev': 'analysis/p2_eval_2026-05-08/p2_deeper_analysis/frame_reports/visomaster_enhanced_macro_dev_p2_d_fourier_periodic_step3000_frames_report.csv',
        'deeplive_enhanced_dev': 'analysis/p2_eval_2026-05-08/p2_deeper_analysis/frame_reports/deeplive_enhanced_dev_p2_d_fourier_periodic_step3000_frames_report.csv',
        'teams_fake_all_dev': 'analysis/p2_eval_2026-05-08/p2_deeper_analysis/frame_reports/teams_fake_all_dev_p2_d_fourier_periodic_step3000_frames_report.csv',
    },
}

# Substrate filter: chronic-6 identities to exclude from FPR pool when evaluating "F4"
CHRONIC = {'bla_bla_chow','bla_bla_chow__s2','pc_generator__s22','pc_generator__s45','roy_d','q__s6'}


def per_identity_pivot(suite_df, parquet_meta=None):
    """Add identity column and (optionally) parquet IQ. Return DataFrame indexed by identity."""
    df = suite_df.copy()
    df['identity'] = df['video_id'].astype(str).apply(base_identity)
    df['frame_prob'] = pd.to_numeric(df['frame_prob'], errors='coerce')
    return df


def evaluate(ckpt, csvs, parquet_meta):
    real_df = per_identity_pivot(pd.read_csv(csvs['teams_real_all_dev']))
    fake_dfs = {suite: per_identity_pivot(pd.read_csv(p)) for suite, p in csvs.items() if suite != 'teams_real_all_dev'}

    # Apply F2+F3 (production filter): need parquet to know min_wh & is_no_face
    real_anno = real_df.merge(parquet_meta, how='left', left_on='frame_path', right_on='gcs_uri')
    min_wh = pd.to_numeric(real_anno['min_wh'], errors='coerce')
    is_lr = (min_wh < MIN_WH_PX).fillna(False)
    is_nf = real_anno['is_no_face'].fillna(False).astype(bool)
    real_anno['keep_P2'] = ~(is_lr | is_nf)
    real_anno['keep_F4'] = real_anno['keep_P2'] & ~real_anno['identity'].isin(CHRONIC)

    out_rows = []
    # Tau grid focused on relevant range
    tau_grid = np.concatenate([np.linspace(0.05, 0.95, 19), np.array([0.97, 0.99])])
    for substrate_name, mask_col in [('P0_raw','keep_all'), ('P2_lowres','keep_P2'), ('F4_full','keep_F4')]:
        if mask_col == 'keep_all':
            real_sub = real_anno
        else:
            real_sub = real_anno[real_anno[mask_col]]
        # Identity-level totals
        real_identities = sorted(real_sub['identity'].unique())
        for tau in tau_grid:
            # Per-identity fraction(>tau) for real
            real_frac = real_sub.groupby('identity')['frame_prob'].apply(lambda s: float((s > tau).mean()))
            real_n = real_sub.groupby('identity').size()
            n_real_id = len(real_frac)
            flagged_real = real_frac > 0.5
            n_fpr = int(flagged_real.sum())
            # Worst real identity
            worst_real = real_frac.idxmax()
            worst_real_frac = float(real_frac.max())

            # Per-fake-suite identity-level recall (each suite's identities are pool)
            per_suite_recall = {}
            all_fake_identities = []
            for suite, fdf in fake_dfs.items():
                fake_frac = fdf.groupby('identity')['frame_prob'].apply(lambda s: float((s > tau).mean()))
                flagged_fake = fake_frac > 0.5
                per_suite_recall[suite] = float(flagged_fake.mean()) if len(flagged_fake) else None
                all_fake_identities.extend([f'{suite}::{i}' for i in fake_frac.index if flagged_fake[i]])
            macro_recall = float(np.mean([v for v in per_suite_recall.values() if v is not None])) if per_suite_recall else None

            out_rows.append({
                'ckpt': ckpt,
                'substrate': substrate_name,
                'tau': round(float(tau), 4),
                'n_real_identities': n_real_id,
                'identity_FPR_pct': round(100 * n_fpr / max(1, n_real_id), 2),
                'n_real_flagged': n_fpr,
                'worst_real_identity': worst_real,
                'worst_real_frac_above_tau': round(worst_real_frac, 3),
                **{f'recall_{s}_pct': round(100 * v, 2) if v is not None else None for s, v in per_suite_recall.items()},
                'macro_recall_pct': round(100 * macro_recall, 2) if macro_recall is not None else None,
            })

    return out_rows


def main():
    parquet_meta = load_parquet_meta()
    all_rows = []
    for ckpt, csvs in CK_FILES.items():
        print(f'\n=== {ckpt} ===')
        rows = evaluate(ckpt, csvs, parquet_meta)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(OUT_DIR, 'STAGE6_PER_IDENTITY.csv')
    df.to_csv(out_csv, index=False)
    print(f'\nwrote {out_csv}')

    # Pretty print: per-ckpt, best tau for each FPR budget
    for ckpt in CK_FILES:
        print(f'\n=== {ckpt} — per-identity FPR/recall under majority-vote (>50% rule) ===')
        for substrate in ['P2_lowres','F4_full']:
            sub = df[(df.ckpt == ckpt) & (df.substrate == substrate)].sort_values('tau')
            print(f'\n-- substrate = {substrate} (n_real_identities = {sub.iloc[0]["n_real_identities"]}) --')
            cols = ['tau','identity_FPR_pct','n_real_flagged','worst_real_identity','worst_real_frac_above_tau','macro_recall_pct']
            print(sub[cols].to_string(index=False))


if __name__ == '__main__':
    main()
