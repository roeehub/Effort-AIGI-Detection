"""Stage 11: Fine tau sweep + per-identity fraction-average ensemble.

Finer-grained tau exploration on T5C step3500 (current winner). Stage 7 used
0.05 increments — here we use 0.01.

Also tests a new ensemble rule: per-identity FRACTION-AVERAGE.
  ensemble_frac(id) = mean(frac_T5C(id), frac_P2D(id))
  flag if ensemble_frac > 0.5

This differs from Stage 8's vote-AND (both ckpts must independently flag).
Fraction-average can produce a larger or smaller gap depending on how the
two ckpts correlate per-identity.
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

OUT_DIR = os.path.join(ROOT, 'analysis/best_candidate_search_2026-05-13/_stage11')
os.makedirs(OUT_DIR, exist_ok=True)


def per_id_frac(df, tau):
    return df.groupby('identity')['frame_prob'].apply(
        lambda s: float((pd.to_numeric(s, errors='coerce') > tau).mean())
    )


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


def main():
    parquet_meta = load_parquet_meta()
    print('=' * 80)
    print('Part 1: Fine tau sweep (T5C step3500 on F4)')
    print('=' * 80)

    csvs = CK_FILES['T5C_step3500']
    real = add_substrate(load_anno(csvs['teams_real_all_dev']), parquet_meta)
    real_F4 = real[real['keep_F4']]
    fakes = {s: load_anno(csvs[s]) for s in csvs if s != 'teams_real_all_dev'}

    rows = []
    for tau in np.arange(0.05, 0.95, 0.01):
        real_frac = per_id_frac(real_F4, tau)
        n_real = len(real_frac)
        n_flagged = int((real_frac > 0.5).sum())
        max_real = float(real_frac.max())
        max_real_id = real_frac.idxmax()
        per_suite_rec, per_suite_min = {}, {}
        for s, fdf in fakes.items():
            fk = per_id_frac(fdf, tau)
            per_suite_rec[s] = float((fk > 0.5).mean())
            per_suite_min[s] = float(fk.min())
        macro = float(np.mean(list(per_suite_rec.values())))
        min_fake = min(per_suite_min.values())
        gap = min_fake - max_real
        rows.append({
            'tau': round(tau, 3),
            'fpr_id_pct': round(100 * n_flagged / max(1, n_real), 2),
            'macro_recall_id_pct': round(100 * macro, 2),
            'max_real_frac': round(max_real, 4),
            'min_fake_frac': round(min_fake, 4),
            'safety_gap': round(gap, 4),
            'max_real_id': max_real_id,
        })
    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, 'STAGE11_FINE_TAU.csv')
    df.to_csv(out_csv, index=False)

    # Pick best tau by gap, subject to FPR=0 + recall=100
    best = df[(df.fpr_id_pct == 0) & (df.macro_recall_id_pct == 100)].sort_values('safety_gap', ascending=False)
    if len(best):
        print(f'\nBest tau for T5C step3500 on F4 (FPR=0, recall=100):')
        print(best.head(10).to_string(index=False))
    else:
        print('  None hit FPR=0 + recall=100 across tau grid.')

    # Also: window of safe taus
    safe = df[(df.fpr_id_pct == 0) & (df.macro_recall_id_pct == 100)]
    if len(safe):
        print(f'\nSafe-tau window (FPR=0 AND recall=100): tau in [{safe.tau.min():.2f}, {safe.tau.max():.2f}]')
        print(f'  Best gap @ tau={safe.loc[safe.safety_gap.idxmax(),"tau"]:.2f}: gap={safe.safety_gap.max():.4f}')

    # ======================================================================
    print('\n' + '=' * 80)
    print('Part 2: Per-identity fraction-average ensemble (T5C + P2D)')
    print('=' * 80)
    t5c_csvs = CK_FILES['T5C_step3500']
    p2d_csvs = CK_FILES['P2D_fourier_step3000']
    t5c_real = add_substrate(load_anno(t5c_csvs['teams_real_all_dev']), parquet_meta)
    p2d_real = add_substrate(load_anno(p2d_csvs['teams_real_all_dev']), parquet_meta)
    t5c_real_F4 = t5c_real[t5c_real['keep_F4']]
    p2d_real_F4 = p2d_real[p2d_real['keep_F4']]
    t5c_fakes = {s: load_anno(t5c_csvs[s]) for s in t5c_csvs if s != 'teams_real_all_dev'}
    p2d_fakes = {s: load_anno(p2d_csvs[s]) for s in p2d_csvs if s != 'teams_real_all_dev'}

    tau_t_grid = np.arange(0.30, 0.71, 0.05)
    tau_p_grid = np.arange(0.10, 0.51, 0.05)
    rows = []
    for tau_t in tau_t_grid:
        real_frac_t = per_id_frac(t5c_real_F4, tau_t)
        for tau_p in tau_p_grid:
            real_frac_p = per_id_frac(p2d_real_F4, tau_p)
            # Intersection of identities
            ids = sorted(set(real_frac_t.index) & set(real_frac_p.index))
            real_avg = (real_frac_t.reindex(ids).fillna(0) + real_frac_p.reindex(ids).fillna(0)) / 2.0
            n_real = len(ids)
            n_flagged = int((real_avg > 0.5).sum())
            max_real = float(real_avg.max())

            per_suite_rec, per_suite_min = {}, {}
            for s in t5c_fakes:
                t_frac = per_id_frac(t5c_fakes[s], tau_t)
                p_frac = per_id_frac(p2d_fakes[s], tau_p)
                f_ids = sorted(set(t_frac.index) & set(p_frac.index))
                f_avg = (t_frac.reindex(f_ids).fillna(0) + p_frac.reindex(f_ids).fillna(0)) / 2.0
                per_suite_rec[s] = float((f_avg > 0.5).mean())
                per_suite_min[s] = float(f_avg.min())
            macro = float(np.mean(list(per_suite_rec.values())))
            min_fake = min(per_suite_min.values())
            gap = min_fake - max_real
            rows.append({
                'tau_T5C': round(tau_t, 2),
                'tau_P2D': round(tau_p, 2),
                'fpr_id_pct': round(100 * n_flagged / max(1, n_real), 2),
                'macro_recall_id_pct': round(100 * macro, 2),
                'max_real_frac_avg': round(max_real, 4),
                'min_fake_frac_avg': round(min_fake, 4),
                'safety_gap_avg': round(gap, 4),
            })
    df_ens = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, 'STAGE11_FRAC_AVG_ENSEMBLE.csv')
    df_ens.to_csv(out_csv, index=False)
    print(f'\nWrote {out_csv}')

    qual = df_ens[(df_ens.fpr_id_pct == 0) & (df_ens.macro_recall_id_pct == 100)]
    if len(qual):
        best = qual.sort_values('safety_gap_avg', ascending=False).head(10)
        print(f'\nBest (FPR=0, recall=100) fraction-average ensemble:')
        print(best.to_string(index=False))
        best_gap = qual.safety_gap_avg.max()
        single_t5c_best = 0.5257  # from Stage 7
        if best_gap > single_t5c_best:
            print(f'\n*** Ensemble BEATS T5C-alone: {best_gap:.4f} > {single_t5c_best:.4f} ***')
        else:
            print(f'\nEnsemble (gap {best_gap:.4f}) does NOT beat T5C-alone (gap {single_t5c_best:.4f})')
    else:
        print('  No ensemble pair meets FPR=0 + recall=100.')


if __name__ == '__main__':
    main()
