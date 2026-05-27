"""Stage 4: Deep-dive on the top-N finalists.

For each finalist:
  - shortcut indicators (pearson r on full pool AND on F4-cleaned reals)
  - forgery AUC (full pool AND F4-cleaned reals)
  - per-suite recall at FPR = {2, 5, 10}% on F4-cleaned reals
  - minimum FPR (on F4 reals) needed to reach 90% macro recall — and the resulting
    in-eval FPR on the full F0 substrate
  - per-suite recall under D5_drop_roy (== F4) for each fake suite

Output: STAGE4_FINALISTS.csv + a per-candidate JSON.
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = '/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training'
sys.path.insert(0, os.path.join(ROOT, 'analysis/substrate_cleaning_eval_2026-05-05'))
sys.path.insert(0, os.path.join(ROOT, 'analysis/best_candidate_search_2026-05-13'))
from run_clean_eval import MIN_WH_PX, extract_identity_from_video_id, load_parquet_meta  # noqa: E402
from stage1_run_all_candidates import CANDIDATES  # noqa: E402

OUT_DIR = os.path.join(ROOT, 'analysis/best_candidate_search_2026-05-13/_stage4')
os.makedirs(OUT_DIR, exist_ok=True)

# Top-N finalists from Stage 3 (sorted by F4 macro recall at FPR=5%).
FINALISTS = [
    'T5C_step3500',
    'P2D_fourier_step3000',
    'P1_pairrank_step6750',
    'P1_pairrank_step6000',
    'P1_bundle_step4000',
    'T3_SLOT1_step1500',
    'T3_SLOT1_step2500',
    'P1_bundle_step3750',
    'P8A_step5000',           # gold-standard invariance baseline
    'E2B_step3200',           # currently deployed
    'P2D_fourier_step8000',
    'E3_step6600',
]


def annotate_real(real_csv, parquet_meta):
    df = pd.read_csv(real_csv)
    df['identity'] = df['video_id'].apply(extract_identity_from_video_id)
    merged = df.merge(parquet_meta, how='left',
                      left_on='frame_path', right_on='gcs_uri')
    merged['identity_pq'] = merged['identity_key'].astype(str).str.lower()
    merged.loc[merged['identity_key'].isna(), 'identity_pq'] = merged['identity']
    return merged


def make_F4_mask(real_df):
    """F4 = drop chronic-6 + lowres + no-face."""
    chronic = ['bla_bla_chow','bla_bla_chow__s2','pc_generator__s22','pc_generator__s45','roy_d','q__s6']
    chr_set = set(chronic)
    is_chr = real_df['identity'].isin(chr_set) | real_df['identity_pq'].isin(chr_set)
    min_wh = pd.to_numeric(real_df['min_wh'], errors='coerce')
    is_lr = (min_wh < MIN_WH_PX).fillna(False)
    is_nf = real_df['is_no_face'].fillna(False).astype(bool)
    return ~(is_chr | is_lr | is_nf)


def pearson(a, b):
    a = pd.to_numeric(a, errors='coerce')
    b = pd.to_numeric(b, errors='coerce')
    mask = a.notna() & b.notna()
    if mask.sum() < 50:
        return None
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def auc_real_vs_fake(real_scores, fake_scores):
    r = pd.to_numeric(real_scores, errors='coerce').dropna().to_numpy()
    f = pd.to_numeric(fake_scores, errors='coerce').dropna().to_numpy()
    if len(r) == 0 or len(f) == 0:
        return None
    y = np.concatenate([np.zeros(len(r)), np.ones(len(f))])
    s = np.concatenate([r, f])
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return None


def min_fpr_for_recall_target(real_scores, fake_scores_list, target_recall=0.90):
    """For each candidate tau (descending from max), compute macro recall and
    return the LOWEST FPR on real_scores where macro_recall >= target.
    fake_scores_list is a list of np arrays (one per suite)."""
    r = pd.to_numeric(real_scores, errors='coerce').dropna().to_numpy()
    if len(r) == 0:
        return None, None
    # Sample candidate taus from the merged score space
    all_s = np.concatenate([r] + [pd.to_numeric(f, errors='coerce').dropna().to_numpy() for f in fake_scores_list])
    taus = np.unique(np.quantile(all_s, np.linspace(0.0, 1.0, 2001)))
    best_fpr = None
    best_tau = None
    # Sort descending so that as tau decreases, FPR grows
    for tau in np.sort(taus)[::-1]:
        fpr = float((r > tau).mean())
        recalls = []
        for f in fake_scores_list:
            fs = pd.to_numeric(f, errors='coerce').dropna().to_numpy()
            if len(fs) > 0:
                recalls.append(float((fs > tau).mean()))
        if not recalls:
            continue
        rec = float(np.mean(recalls))
        if rec >= target_recall:
            if best_fpr is None or fpr < best_fpr:
                best_fpr = fpr
                best_tau = float(tau)
    return best_fpr, best_tau


def fpr_at_tau(scores, tau):
    s = pd.to_numeric(scores, errors='coerce').dropna().to_numpy()
    if len(s) == 0:
        return None
    return float((s > tau).mean())


def main():
    parquet_meta = load_parquet_meta()
    cand_map = {c['ckpt_name']: c for c in CANDIDATES}
    rows = []
    for name in FINALISTS:
        spec = cand_map.get(name)
        if spec is None:
            print(f'  skip {name}: not in candidates list')
            continue
        real_csv = spec['real_csv']
        if not os.path.exists(real_csv):
            print(f'  skip {name}: missing real csv')
            continue
        real = annotate_real(real_csv, parquet_meta)
        f4_keep = make_F4_mask(real)
        real_F4 = real[f4_keep]

        # Pre-read fakes
        fakes = {}
        for suite, p in spec['fake_csvs'].items():
            if os.path.exists(p):
                fakes[suite] = pd.read_csv(p)

        row = {
            'ckpt': name,
            'n_F0': int(len(real)),
            'n_F4': int(f4_keep.sum()),
        }
        # Pearson r on full and F4
        row['pearson_F0'] = round(pearson(real['frame_prob'], real['min_wh']) or 0, 4)
        row['pearson_F4'] = round(pearson(real_F4['frame_prob'], real_F4['min_wh']) or 0, 4)
        # AUC on full and F4
        aucs_F0, aucs_F4 = [], []
        for suite, fdf in fakes.items():
            short = suite.replace('_enhanced_macro_dev','viso').replace('_enhanced_dev','').replace('_all_dev','')
            row[f'auc_F0_{short}'] = round(auc_real_vs_fake(real['frame_prob'], fdf['frame_prob']) or 0, 4)
            row[f'auc_F4_{short}'] = round(auc_real_vs_fake(real_F4['frame_prob'], fdf['frame_prob']) or 0, 4)
            aucs_F0.append(row[f'auc_F0_{short}'])
            aucs_F4.append(row[f'auc_F4_{short}'])
        row['auc_F0_macro'] = round(float(np.mean(aucs_F0)), 4) if aucs_F0 else None
        row['auc_F4_macro'] = round(float(np.mean(aucs_F4)), 4) if aucs_F4 else None

        # Per-suite recall at FPR = {2, 5, 10} on F4 reals
        r_F4 = pd.to_numeric(real_F4['frame_prob'], errors='coerce').dropna().to_numpy()
        for target_fpr in [0.02, 0.05, 0.10]:
            tau = float(np.quantile(r_F4, 1 - target_fpr))
            recs = []
            for suite, fdf in fakes.items():
                short = suite.replace('_enhanced_macro_dev','viso').replace('_enhanced_dev','').replace('_all_dev','')
                fs = pd.to_numeric(fdf['frame_prob'], errors='coerce').dropna().to_numpy()
                rec = float((fs > tau).mean()) * 100
                row[f'recall_{short}_FPR{int(target_fpr*100)}'] = round(rec, 2)
                recs.append(rec)
            row[f'macro_recall_FPR{int(target_fpr*100)}'] = round(float(np.mean(recs)), 2)
            row[f'tau_FPR{int(target_fpr*100)}_F4'] = round(tau, 4)
            # In-eval (F0) FPR at this tau
            row[f'fpr_F0_at_tau_FPR{int(target_fpr*100)}'] = round(100 * (fpr_at_tau(real['frame_prob'], tau) or 0), 2)

        # Min FPR (on F4) for macro recall >= 90% / 95%
        fake_arrays = [fdf['frame_prob'] for fdf in fakes.values()]
        for target in [0.90, 0.95]:
            fpr, tau = min_fpr_for_recall_target(real_F4['frame_prob'], fake_arrays, target)
            row[f'min_fpr_F4_recall{int(target*100)}'] = round(100 * fpr, 2) if fpr is not None else None
            row[f'tau_at_recall{int(target*100)}'] = round(tau, 4) if tau is not None else None
            # F0 FPR at this tau (production-realistic when including chronics)
            if tau is not None:
                row[f'fpr_F0_at_recall{int(target*100)}'] = round(100 * (fpr_at_tau(real['frame_prob'], tau) or 0), 2)

        rows.append(row)
        print(f'{name}:')
        print(f'  pearson F0={row["pearson_F0"]} F4={row["pearson_F4"]}  auc F0={row["auc_F0_macro"]} F4={row["auc_F4_macro"]}')
        print(f'  recall F4@FPR2={row["macro_recall_FPR2"]} @FPR5={row["macro_recall_FPR5"]} @FPR10={row["macro_recall_FPR10"]}')
        print(f'  min_FPR_F4 for 90% recall = {row.get("min_fpr_F4_recall90")}%  (F0 FPR = {row.get("fpr_F0_at_recall90")}%)')

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, 'STAGE4_FINALISTS.csv')
    df.to_csv(out_csv, index=False)
    print(f'\nwrote {out_csv}')


if __name__ == '__main__':
    main()
