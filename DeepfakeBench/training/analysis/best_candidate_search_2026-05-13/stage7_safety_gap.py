"""Stage 7: Safety-gap analysis for per-identity majority-vote.

For each candidate × tau × substrate, compute:
  gap = min(fake_id_frac_above_tau across all fake suites)
        − max(real_id_frac_above_tau)

Higher gap = more margin between the worst-detected fake and worst-misbehaving
real. Find the tau that maximizes gap subject to:
  - identity_FPR = 0% on the chosen substrate
  - macro identity recall ≥ 95%

Output: STAGE7_SAFETY_GAP.csv with one row per (ckpt × tau × substrate) and
a SUMMARY block showing best-tau-by-gap for each ckpt.
"""
from __future__ import annotations
import os
import re
import sys
import numpy as np
import pandas as pd

ROOT = '/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training'
sys.path.insert(0, os.path.join(ROOT, 'analysis/substrate_cleaning_eval_2026-05-05'))
sys.path.insert(0, os.path.join(ROOT, 'analysis/best_candidate_search_2026-05-13'))
from run_clean_eval import MIN_WH_PX, load_parquet_meta  # noqa: E402
from stage6_per_identity_majority import CK_FILES, CHRONIC, base_identity  # noqa: E402

OUT_DIR = os.path.join(ROOT, 'analysis/best_candidate_search_2026-05-13/_stage7')
os.makedirs(OUT_DIR, exist_ok=True)


def per_identity_fracs(df, tau):
    """Return per-identity frame-fraction above tau."""
    g = df.groupby('identity')['frame_prob'].apply(lambda s: float((pd.to_numeric(s, errors='coerce') > tau).mean()))
    return g


def evaluate(ckpt, csvs, parquet_meta):
    real_df = pd.read_csv(csvs['teams_real_all_dev'])
    real_df['identity'] = real_df['video_id'].astype(str).apply(base_identity)
    real_df['frame_prob'] = pd.to_numeric(real_df['frame_prob'], errors='coerce')
    real_anno = real_df.merge(parquet_meta, how='left', left_on='frame_path', right_on='gcs_uri')
    min_wh = pd.to_numeric(real_anno['min_wh'], errors='coerce')
    is_lr = (min_wh < MIN_WH_PX).fillna(False)
    is_nf = real_anno['is_no_face'].fillna(False).astype(bool)
    real_anno['keep_P2'] = ~(is_lr | is_nf)
    real_anno['keep_F4'] = real_anno['keep_P2'] & ~real_anno['identity'].isin(CHRONIC)

    fake_dfs = {}
    for suite, p in csvs.items():
        if suite == 'teams_real_all_dev':
            continue
        d = pd.read_csv(p)
        d['identity'] = d['video_id'].astype(str).apply(base_identity)
        d['frame_prob'] = pd.to_numeric(d['frame_prob'], errors='coerce')
        fake_dfs[suite] = d

    tau_grid = np.concatenate([np.linspace(0.05, 0.95, 19), np.array([0.97, 0.99])])
    rows = []
    for substrate_name, mask_col in [('P2_lowres','keep_P2'), ('F4_full','keep_F4')]:
        real_sub = real_anno[real_anno[mask_col]]
        for tau in tau_grid:
            real_fracs = per_identity_fracs(real_sub, tau)
            n_real_id = len(real_fracs)
            n_real_flagged = int((real_fracs > 0.5).sum())
            max_real_frac = float(real_fracs.max()) if len(real_fracs) else None
            max_real_id = real_fracs.idxmax() if len(real_fracs) else None
            fpr_id = n_real_flagged / max(1, n_real_id)

            # Per fake suite
            per_suite_min_frac = {}
            per_suite_recall = {}
            for suite, fdf in fake_dfs.items():
                fake_fracs = per_identity_fracs(fdf, tau)
                per_suite_recall[suite] = float((fake_fracs > 0.5).mean())
                per_suite_min_frac[suite] = float(fake_fracs.min())
            macro_recall = float(np.mean(list(per_suite_recall.values())))
            # Worst-detected fake across all suites
            min_fake_frac = min(per_suite_min_frac.values())
            min_fake_suite = min(per_suite_min_frac, key=per_suite_min_frac.get)

            gap = min_fake_frac - max_real_frac

            row = {
                'ckpt': ckpt,
                'substrate': substrate_name,
                'tau': round(float(tau), 4),
                'n_real_id': n_real_id,
                'identity_fpr_pct': round(100 * fpr_id, 2),
                'macro_recall_id_pct': round(100 * macro_recall, 2),
                'max_real_frac': round(max_real_frac, 4),
                'max_real_id': max_real_id,
                'min_fake_frac': round(min_fake_frac, 4),
                'min_fake_suite': min_fake_suite,
                'safety_gap': round(gap, 4),
            }
            for s, r in per_suite_recall.items():
                row[f'recall_{s}_pct'] = round(100 * r, 2)
                row[f'min_frac_{s}'] = round(per_suite_min_frac[s], 4)
            rows.append(row)
    return rows


def main():
    parquet_meta = load_parquet_meta()
    all_rows = []
    for ckpt, csvs in CK_FILES.items():
        print(f'evaluating {ckpt} ...')
        all_rows.extend(evaluate(ckpt, csvs, parquet_meta))

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(OUT_DIR, 'STAGE7_SAFETY_GAP.csv')
    df.to_csv(out_csv, index=False)
    print(f'\nwrote {out_csv}')

    # Best tau by safety gap per ckpt + substrate (subject to FPR=0, recall>=95)
    print('\n=== BEST-TAU-BY-GAP (FPR=0% AND macro_recall>=95% required) ===')
    print(f'{"ckpt":25s}{"substrate":12s}{"tau":>8s}{"gap":>8s}{"fpr_id":>10s}{"recall_id":>12s}{"max_real":>10s}{"min_fake":>10s}{"max_real_id":>20s}{"min_fake_suite":>35s}')
    for ckpt in CK_FILES:
        for sub in ['P2_lowres','F4_full']:
            qual = df[(df.ckpt==ckpt) & (df.substrate==sub) & (df.identity_fpr_pct==0) & (df.macro_recall_id_pct>=95)]
            if len(qual) == 0:
                print(f'{ckpt:25s}{sub:12s}  NO TAU MEETS FPR=0 + recall>=95%')
                continue
            best = qual.loc[qual['safety_gap'].idxmax()]
            print(f'{ckpt:25s}{sub:12s}{best["tau"]:8.4f}{best["safety_gap"]:8.4f}{best["identity_fpr_pct"]:10.2f}{best["macro_recall_id_pct"]:12.2f}{best["max_real_frac"]:10.4f}{best["min_fake_frac"]:10.4f}{str(best["max_real_id"]):>20s}{str(best["min_fake_suite"])[:35]:>35s}')

    # Also: best tau by gap with no FPR/recall constraint (for context)
    print('\n=== BEST-TAU-BY-GAP (unconstrained) ===')
    for ckpt in CK_FILES:
        for sub in ['P2_lowres','F4_full']:
            sub_df = df[(df.ckpt==ckpt) & (df.substrate==sub)]
            best = sub_df.loc[sub_df['safety_gap'].idxmax()]
            print(f'{ckpt:25s}{sub:12s}{best["tau"]:8.4f}{best["safety_gap"]:8.4f}{best["identity_fpr_pct"]:10.2f}{best["macro_recall_id_pct"]:12.2f}{best["max_real_frac"]:10.4f}{best["min_fake_frac"]:10.4f}')


if __name__ == '__main__':
    main()
