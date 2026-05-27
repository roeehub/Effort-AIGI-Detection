"""Stage 8: Two-model ensemble with per-identity majority vote.

For each (tau_T5C, tau_P2D) pair on a grid:
  - Compute per-identity vote_T5C = (frac > tau_T5C) > 0.5
  - Compute per-identity vote_P2D = (frac > tau_P2D) > 0.5
  - AND-rule: identity flagged fake iff BOTH votes are fake
  - OR-rule: identity flagged fake iff EITHER vote is fake

Then compute:
  - identity FPR on F4 and P2 substrates
  - identity macro recall
  - safety gap (now defined for the ensemble decision boundary)

Hypothesis: two structurally-distinct models should give a strictly larger
safety margin than either alone, with FPR=0/recall=100 preserved.
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

OUT_DIR = os.path.join(ROOT, 'analysis/best_candidate_search_2026-05-13/_stage8')
os.makedirs(OUT_DIR, exist_ok=True)


def per_id_frac(df, tau):
    return df.groupby('identity')['frame_prob'].apply(
        lambda s: float((pd.to_numeric(s, errors='coerce') > tau).mean())
    )


def load_anno(csv_path, parquet_meta):
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
    t5c_csv = CK_FILES['T5C_step3500']
    p2d_csv = CK_FILES['P2D_fourier_step3000']

    suites = list(t5c_csv.keys())
    print(f'suites: {suites}')

    # Per-suite frame DFs (real + fakes) for both ckpts
    t5c = {s: load_anno(t5c_csv[s], parquet_meta) for s in suites}
    p2d = {s: load_anno(p2d_csv[s], parquet_meta) for s in suites}

    real_t5c = add_substrate(t5c['teams_real_all_dev'], parquet_meta)
    real_p2d = add_substrate(p2d['teams_real_all_dev'], parquet_meta)
    fake_suites = [s for s in suites if s != 'teams_real_all_dev']

    # Tau grids — focus on promising regions for each ckpt
    tau_t5c_grid = np.arange(0.10, 0.91, 0.05)
    tau_p2d_grid = np.arange(0.05, 0.91, 0.05)

    rows = []
    for substrate in ['P2', 'F4']:
        rmask = real_t5c['keep_P2'] if substrate == 'P2' else real_t5c['keep_F4']
        real_t5c_sub = real_t5c[rmask].copy()
        rmask_p = real_p2d['keep_P2'] if substrate == 'P2' else real_p2d['keep_F4']
        real_p2d_sub = real_p2d[rmask_p].copy()
        # Identity intersection — only score identities present in BOTH ckpts
        real_ids = sorted(set(real_t5c_sub['identity'].unique()) & set(real_p2d_sub['identity'].unique()))

        for tau_t in tau_t5c_grid:
            real_frac_t = per_id_frac(real_t5c_sub[real_t5c_sub['identity'].isin(real_ids)], tau_t)
            for tau_p in tau_p2d_grid:
                real_frac_p = per_id_frac(real_p2d_sub[real_p2d_sub['identity'].isin(real_ids)], tau_p)
                vote_real_t = (real_frac_t > 0.5).reindex(real_ids).fillna(False)
                vote_real_p = (real_frac_p > 0.5).reindex(real_ids).fillna(False)
                and_real = vote_real_t & vote_real_p
                or_real = vote_real_t | vote_real_p

                # Fake-side
                and_recalls = []
                or_recalls = []
                min_and_frac = 1.0
                for fs in fake_suites:
                    fake_t = per_id_frac(t5c[fs], tau_t)
                    fake_p = per_id_frac(p2d[fs], tau_p)
                    # Use intersection of fake identities
                    fake_ids = sorted(set(fake_t.index) & set(fake_p.index))
                    vt = (fake_t.reindex(fake_ids) > 0.5)
                    vp = (fake_p.reindex(fake_ids) > 0.5)
                    and_recalls.append(float((vt & vp).mean()))
                    or_recalls.append(float((vt | vp).mean()))
                    # safety margin proxy: for each fake identity, take min(frac_t, frac_p) (the AND-vote-rule analogue)
                    # then identity-level vote requires BOTH > 0.5; safety = min over identities of (min(frac_t, frac_p))
                    min_per_id = pd.DataFrame({'t': fake_t.reindex(fake_ids), 'p': fake_p.reindex(fake_ids)}).min(axis=1)
                    if len(min_per_id):
                        min_and_frac = min(min_and_frac, float(min_per_id.min()))

                # Real-side worst (for AND rule, both must agree → real "and-vote-fraction" = min(frac_t,frac_p))
                max_real_and = float(pd.DataFrame({'t': real_frac_t.reindex(real_ids), 'p': real_frac_p.reindex(real_ids)}).min(axis=1).max())
                gap_and = min_and_frac - max_real_and

                rows.append({
                    'substrate': substrate,
                    'tau_T5C': round(float(tau_t), 2),
                    'tau_P2D': round(float(tau_p), 2),
                    'AND_fpr_id_pct': round(100*float(and_real.mean()), 2),
                    'AND_macro_recall_pct': round(100*float(np.mean(and_recalls)), 2),
                    'OR_fpr_id_pct': round(100*float(or_real.mean()), 2),
                    'OR_macro_recall_pct': round(100*float(np.mean(or_recalls)), 2),
                    'AND_gap_proxy': round(gap_and, 4),
                    'AND_min_fake_frac': round(min_and_frac, 4),
                    'AND_max_real_frac': round(max_real_and, 4),
                })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, 'STAGE8_ENSEMBLE.csv')
    df.to_csv(out_csv, index=False)
    print(f'wrote {out_csv}')

    for substrate in ['P2', 'F4']:
        print(f'\n=== Substrate {substrate} — TOP-10 AND-rule by safety-gap-proxy (FPR=0, recall>=95) ===')
        q = df[(df.substrate == substrate) & (df.AND_fpr_id_pct == 0) & (df.AND_macro_recall_pct >= 95)]
        if len(q) == 0:
            print('  NO (tau_T5C, tau_P2D) pair meets FPR=0 + recall>=95% under AND-rule')
            # Show best by relaxed criterion
            q2 = df[(df.substrate == substrate) & (df.AND_fpr_id_pct == 0)]
            if len(q2):
                print(f'  Relaxed (FPR=0 only) — TOP-5 by recall:')
                print(q2.sort_values('AND_macro_recall_pct', ascending=False).head(5).to_string(index=False))
            continue
        print(q.sort_values('AND_gap_proxy', ascending=False).head(10).to_string(index=False))


if __name__ == '__main__':
    main()
