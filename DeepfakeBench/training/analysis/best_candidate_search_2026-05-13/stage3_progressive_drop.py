"""Stage 3 (v2): For each candidate, compute fake recall under stepwise
drop policies that isolate the marginal effect of each filter.

KEY OBSERVATION from v1: the 3 tiny-face chronics (pc_gen_s22 / pc_gen_s45 /
q__s6) all have min(W,H) < 200, so the F2 (lowres) filter already drops them.
That means we cannot distinguish "drop tiny-face chronic" from "apply F2 filter."

Policies (each is a superset of the previous):
  P0_F0_raw          : no drops at all (raw 4564 frames)
  P1_F3_noface       : + drop is_no_face frames only
  P2_F2_lowres       : + drop min(W,H) < 200 (captures the 3 tiny-face chronics)
  P3_drop_bbc        : + drop bla_bla_chow         (wide 399px face)
  P4_drop_bbc_s2     : + drop bla_bla_chow__s2     (small 146px, sharpness 55)
  P5_drop_roy        : + drop roy_d                (T3-FT recipe-bad cohort)
  P6_full_F4         : = P5 + bla_bla_chow + bla_bla_chow__s2 cumulative
                       (this is the Job 14 F4 contract)

Calibration: tau picked on each policy's filtered real pool at FPR=10% and 5%.
Recall measured on the unfiltered fake pools (we want to catch fakes regardless
of identity).
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

ROOT = '/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training'
sys.path.insert(0, os.path.join(ROOT, 'analysis/substrate_cleaning_eval_2026-05-05'))
from run_clean_eval import (  # type: ignore  # noqa: E402
    MIN_WH_PX, extract_identity_from_video_id, load_parquet_meta,
)
from stage1_run_all_candidates import CANDIDATES  # type: ignore  # noqa: E402

OUT_DIR = os.path.join(ROOT, 'analysis/best_candidate_search_2026-05-13/_stage3')
os.makedirs(OUT_DIR, exist_ok=True)


def annotate_real(real_csv: str, parquet_meta: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(real_csv)
    df['identity'] = df['video_id'].apply(extract_identity_from_video_id)
    merged = df.merge(parquet_meta, how='left',
                      left_on='frame_path', right_on='gcs_uri')
    merged['identity_pq'] = merged['identity_key'].astype(str).str.lower()
    merged.loc[merged['identity_key'].isna(), 'identity_pq'] = merged['identity']
    return merged


def policy_masks(real_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Return cumulative KEEP masks for each policy."""
    n = len(real_df)
    min_wh = pd.to_numeric(real_df['min_wh'], errors='coerce')
    no_face = real_df['is_no_face'].fillna(False).astype(bool)

    def is_id(name):
        return (real_df['identity'] == name) | (real_df['identity_pq'] == name)

    drop_noface = no_face
    drop_lowres = (min_wh < MIN_WH_PX).fillna(False)
    drop_bbc    = is_id('bla_bla_chow')
    drop_bbc_s2 = is_id('bla_bla_chow__s2')
    drop_roy    = is_id('roy_d')

    masks = {}
    keep = pd.Series(True, index=real_df.index)
    masks['P0_F0_raw'] = keep.copy()

    keep = keep & ~drop_noface
    masks['P1_F3_noface'] = keep.copy()

    keep = keep & ~drop_lowres
    masks['P2_F2_lowres'] = keep.copy()

    keep = keep & ~drop_bbc
    masks['P3_drop_bbc'] = keep.copy()

    keep = keep & ~drop_bbc_s2
    masks['P4_drop_bbc_s2'] = keep.copy()

    keep = keep & ~drop_roy
    masks['P5_drop_roy'] = keep.copy()

    # P6 is the same as P5 by construction; record explicitly as Job-14 F4 alias.
    masks['P6_full_F4'] = keep.copy()
    return masks


def shortcut_indicator(real_df: pd.DataFrame) -> dict:
    """Pearson r between score and face-size proxy (min_wh). Closer to 0 = less shortcut."""
    s = pd.to_numeric(real_df['frame_prob'], errors='coerce')
    w = pd.to_numeric(real_df['min_wh'], errors='coerce')
    mask = s.notna() & w.notna()
    if mask.sum() < 50:
        return {'pearson_r_score_min_wh': None, 'n_corr': int(mask.sum())}
    r = float(np.corrcoef(s[mask], w[mask])[0, 1])
    return {'pearson_r_score_min_wh': round(r, 4), 'n_corr': int(mask.sum())}


def forgery_auc(real_scores: pd.Series, fake_csvs: dict) -> dict:
    """AUC of fake vs real scores per suite."""
    from sklearn.metrics import roc_auc_score
    r = pd.to_numeric(real_scores, errors='coerce').dropna().to_numpy()
    out = {}
    aucs = []
    for suite, f_csv in fake_csvs.items():
        if not os.path.exists(f_csv):
            continue
        f_df = pd.read_csv(f_csv)
        f = pd.to_numeric(f_df['frame_prob'], errors='coerce').dropna().to_numpy()
        if len(f) == 0 or len(r) == 0:
            continue
        all_vals = np.concatenate([r, f])
        y = np.concatenate([np.zeros(len(r)), np.ones(len(f))])
        try:
            auc = float(roc_auc_score(y, all_vals))
        except Exception:
            auc = float('nan')
        suite_short = suite.replace('_enhanced_macro_dev','viso').replace('_enhanced_dev','').replace('_all_dev','')
        out[f'auc_real_vs_{suite_short}'] = round(auc, 4)
        if not np.isnan(auc):
            aucs.append(auc)
    if aucs:
        out['auc_macro'] = round(float(np.mean(aucs)), 4)
    return out


def score_range(real_scores: pd.Series) -> dict:
    r = pd.to_numeric(real_scores, errors='coerce').dropna().to_numpy()
    if len(r) == 0:
        return {}
    return {
        'real_score_min': round(float(np.min(r)), 4),
        'real_score_p50': round(float(np.median(r)), 4),
        'real_score_max': round(float(np.max(r)), 4),
        'real_score_iqr': round(float(np.quantile(r,0.75)-np.quantile(r,0.25)), 4),
    }


def main():
    parquet_meta = load_parquet_meta()
    print(f'parquet rows: {len(parquet_meta):,}')
    rows = []
    for spec in CANDIDATES:
        name = spec['ckpt_name']
        real_csv = spec['real_csv']
        fake_csvs = spec['fake_csvs']
        if not os.path.exists(real_csv):
            print(f'  SKIP {name}: missing real csv')
            continue
        try:
            real_anno = annotate_real(real_csv, parquet_meta)
        except Exception as e:
            print(f'  ERR {name}: {e}')
            continue
        fake_df_cache = {s: pd.read_csv(p) for s, p in fake_csvs.items() if os.path.exists(p)}

        row = {'ckpt': name}
        row.update(shortcut_indicator(real_anno))
        row.update(score_range(real_anno['frame_prob']))
        row.update(forgery_auc(real_anno['frame_prob'], fake_csvs))

        masks = policy_masks(real_anno)
        for pname, mask in masks.items():
            sub = real_anno.loc[mask, 'frame_prob']
            s = pd.to_numeric(sub, errors='coerce').dropna().to_numpy()
            if len(s) < 50:
                continue
            tau10 = float(np.quantile(s, 0.90))
            tau5  = float(np.quantile(s, 0.95))
            row[f'{pname}_n_real'] = int(len(s))
            row[f'{pname}_tau10'] = round(tau10, 4)
            row[f'{pname}_tau5']  = round(tau5, 4)
            recalls10, recalls5 = [], []
            for suite, fdf in fake_df_cache.items():
                fs = pd.to_numeric(fdf['frame_prob'], errors='coerce').dropna().to_numpy()
                if len(fs) == 0:
                    continue
                r10 = float((fs > tau10).mean()) * 100
                r5  = float((fs > tau5).mean())  * 100
                suite_short = suite.replace('_enhanced_macro_dev','viso').replace('_enhanced_dev','').replace('_all_dev','')
                row[f'{pname}_{suite_short}_r10'] = round(r10, 2)
                row[f'{pname}_{suite_short}_r5']  = round(r5, 2)
                recalls10.append(r10)
                recalls5.append(r5)
            row[f'{pname}_macro_r10'] = round(float(np.mean(recalls10)), 2) if recalls10 else None
            row[f'{pname}_macro_r5']  = round(float(np.mean(recalls5)), 2)  if recalls5  else None
        rows.append(row)
        print(f'{name}: pearson={row.get("pearson_r_score_min_wh")} auc_macro={row.get("auc_macro")} '
              f'P0_r10={row.get("P0_F0_raw_macro_r10")} P2_r10={row.get("P2_F2_lowres_macro_r10")} '
              f'P5_r10={row.get("P5_drop_roy_macro_r10")} P5_r5={row.get("P5_drop_roy_macro_r5")}')

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, 'STAGE3_PROGRESSIVE.csv')
    df = df.sort_values('P5_drop_roy_macro_r10', ascending=False, na_position='last')
    df.to_csv(out_csv, index=False)
    print(f'\nwrote {out_csv}')


if __name__ == '__main__':
    main()
