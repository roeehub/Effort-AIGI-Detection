"""Stage 9: Bootstrap-frame stability of per-identity majority vote.

Production sessions yield 50-500+ frames per identity. The eval data has
1-815 frames per identity (median ~200 for non-chronic identities). Question:
how stable is the majority-vote verdict as we sample fewer/more frames?

For each candidate at its recommended tau:
  - For N in [10, 25, 50, 100, all]:
      - Bootstrap 200 times: sample N frames per identity (with replacement)
      - Compute identity-level frac_above_tau
      - Decide flag (>50%)
      - Track variance of (identity_FPR, identity_recall) across bootstraps

Output stability curves so we know the minimum N for reliable verdict.
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

OUT_DIR = os.path.join(ROOT, 'analysis/best_candidate_search_2026-05-13/_stage9')
os.makedirs(OUT_DIR, exist_ok=True)

# Use the best-gap taus from Stage 7
TAU_RECS = {
    'T5C_step3500':         0.50,
    'P8A_step5000':         0.05,
    'E2B_step3200':         0.50,
    'P2D_fourier_step3000': 0.20,
}

N_BOOT = 200
N_GRID = [10, 25, 50, 100, 200, 500]
RNG = np.random.default_rng(202605)


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


def boot_id_vote(df_by_id, tau, n_sample):
    """For each identity in df_by_id, sample n_sample frames (with replacement)
    and decide flag (>50% above tau). Returns boolean Series."""
    out = {}
    for ident, group in df_by_id:
        probs = pd.to_numeric(group['frame_prob'], errors='coerce').dropna().to_numpy()
        if len(probs) == 0:
            continue
        sampled = RNG.choice(probs, size=n_sample, replace=True)
        frac = (sampled > tau).mean()
        out[ident] = bool(frac > 0.5)
    return pd.Series(out)


def main():
    parquet_meta = load_parquet_meta()
    rows = []
    for ckpt, csvs in CK_FILES.items():
        tau = TAU_RECS[ckpt]
        real = add_substrate(load_anno(csvs['teams_real_all_dev']), parquet_meta)
        real_F4 = real[real['keep_F4']]
        fakes = {s: load_anno(csvs[s]) for s in csvs if s != 'teams_real_all_dev'}

        for n_sample in N_GRID:
            fprs, recs = [], []
            for b in range(N_BOOT):
                real_votes = boot_id_vote(real_F4.groupby('identity'), tau, n_sample)
                fpr = float(real_votes.mean())
                fprs.append(fpr)
                # Per-fake-suite recall
                suite_recalls = []
                for s, fdf in fakes.items():
                    fake_votes = boot_id_vote(fdf.groupby('identity'), tau, n_sample)
                    suite_recalls.append(float(fake_votes.mean()))
                recs.append(float(np.mean(suite_recalls)))

            row = {
                'ckpt': ckpt,
                'tau': tau,
                'n_sample': n_sample,
                'n_boot': N_BOOT,
                'fpr_mean_pct':  round(100*float(np.mean(fprs)), 2),
                'fpr_std_pct':   round(100*float(np.std(fprs)), 2),
                'fpr_p95_pct':   round(100*float(np.quantile(fprs, 0.95)), 2),
                'recall_mean_pct': round(100*float(np.mean(recs)), 2),
                'recall_std_pct':  round(100*float(np.std(recs)), 2),
                'recall_p5_pct':   round(100*float(np.quantile(recs, 0.05)), 2),
            }
            rows.append(row)
            print(f'{ckpt:25s} tau={tau:.2f} N={n_sample:4d}: '
                  f'FPR {row["fpr_mean_pct"]:5.2f}±{row["fpr_std_pct"]:.2f}%  '
                  f'recall {row["recall_mean_pct"]:5.2f}±{row["recall_std_pct"]:.2f}%')

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, 'STAGE9_BOOTSTRAP.csv')
    df.to_csv(out_csv, index=False)
    print(f'wrote {out_csv}')


if __name__ == '__main__':
    main()
