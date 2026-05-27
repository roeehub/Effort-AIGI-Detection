"""Stage 5: T5C + P8A ensemble probes.

Hypothesis: T5C catches more fakes (higher ceiling) but has Roy_D + may6
production-drift failures (lower floor). P8A handles those cases cleanly.
An AND-rule ensemble — flag fake only if BOTH ckpts agree — should keep
T5C's high ceiling while inheriting P8A's safety on the failure modes.

Probes:
  1. AND-rule: predict fake iff (T5C > tau_T5C) AND (P8A > tau_P8A)
  2. MIN-rule: ensemble_score = min(T5C, P8A); apply single tau
  3. AVG-rule: ensemble_score = 0.5*T5C + 0.5*P8A; apply single tau
  4. P8A-gate then T5C: if P8A < tau_P8A_low → real (else use T5C)

For each rule, sweep tau values; report best macro recall at FPR=5%/10% on
P2 (production-realistic) and F4 substrates.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

ROOT = '/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training'
sys.path.insert(0, os.path.join(ROOT, 'analysis/substrate_cleaning_eval_2026-05-05'))
from run_clean_eval import MIN_WH_PX, extract_identity_from_video_id, load_parquet_meta  # noqa: E402

OUT_DIR = os.path.join(ROOT, 'analysis/best_candidate_search_2026-05-13/_stage5')
os.makedirs(OUT_DIR, exist_ok=True)

T5C_DIR = os.path.join(ROOT, 'analysis/cpu_diagnostics_2026-05-12_stage_a/_scorecard_reports')
P8A_DIR = os.path.join(ROOT, 'analysis/cpu_followups_2026-05-04/raw_reports')

SUITES = {
    'teams_real_all_dev': 'real',
    'visomaster_enhanced_macro_dev': 'fake',
    'deeplive_enhanced_dev': 'fake',
    'teams_fake_all_dev': 'fake',
}


def load_pair(suite: str):
    t5c = pd.read_csv(os.path.join(T5C_DIR, f'{suite}_t5c_periodic_step3500_frames_report.csv'))
    p8a = pd.read_csv(os.path.join(P8A_DIR, f'{suite}_p8a_reference_step5000_frames_report.csv'))
    merged = t5c.merge(p8a, on='frame_path', suffixes=('_t5c', '_p8a'))
    if 'video_id_t5c' in merged.columns:
        merged = merged.rename(columns={'video_id_t5c': 'video_id'})
    return merged


def annotate(real_df, parquet_meta):
    d = real_df.copy()
    d['identity'] = d['video_id'].apply(extract_identity_from_video_id)
    m = d.merge(parquet_meta, how='left', left_on='frame_path', right_on='gcs_uri')
    m['identity_pq'] = m['identity_key'].astype(str).str.lower()
    m.loc[m['identity_key'].isna(), 'identity_pq'] = m['identity']
    return m


def f4_mask(d):
    chronic = {'bla_bla_chow','bla_bla_chow__s2','pc_generator__s22','pc_generator__s45','roy_d','q__s6'}
    is_chr = d['identity'].isin(chronic) | d['identity_pq'].isin(chronic)
    min_wh = pd.to_numeric(d['min_wh'], errors='coerce')
    is_lr = (min_wh < MIN_WH_PX).fillna(False)
    is_nf = d['is_no_face'].fillna(False).astype(bool)
    return ~(is_chr | is_lr | is_nf)


def p2_mask(d):
    min_wh = pd.to_numeric(d['min_wh'], errors='coerce')
    is_lr = (min_wh < MIN_WH_PX).fillna(False)
    is_nf = d['is_no_face'].fillna(False).astype(bool)
    return ~(is_lr | is_nf)


def main():
    pq = load_pair('teams_real_all_dev')  # real
    fakes = {s: load_pair(s) for s in SUITES if SUITES[s] == 'fake'}
    parquet_meta = load_parquet_meta()
    real = annotate(pq, parquet_meta)
    print(f'real frames: F0 n={len(real)}, P2 n={int(p2_mask(real).sum())}, F4 n={int(f4_mask(real).sum())}')
    print(f'fake suites: {[(s, len(f)) for s, f in fakes.items()]}')

    # Substrate filters for real
    real_F4 = real[f4_mask(real)]
    real_P2 = real[p2_mask(real)]

    # Build ensemble scores per real / per fake suite
    def t5c(d): return pd.to_numeric(d['frame_prob_t5c'], errors='coerce').to_numpy()
    def p8a(d): return pd.to_numeric(d['frame_prob_p8a'], errors='coerce').to_numpy()
    def rule_min(d): return np.minimum(t5c(d), p8a(d))
    def rule_max(d): return np.maximum(t5c(d), p8a(d))
    def rule_avg(d): return 0.5*(t5c(d) + p8a(d))

    def fpr(scores_real, tau, mask=None):
        s = scores_real if mask is None else scores_real[mask]
        return float((s > tau).mean()) if len(s) else None

    def recall_macro(scores_fakes, tau):
        recs = [float((s > tau).mean()) for s in scores_fakes.values() if len(s)]
        return float(np.mean(recs)) if recs else None

    def best_recall_at_fpr(real_scores, fake_scores_dict, target_fpr, real_mask):
        s = real_scores[real_mask]
        tau = float(np.quantile(s, 1 - target_fpr))
        return recall_macro(fake_scores_dict, tau), tau

    rules = {
        'rule_min_T5C_P8A': rule_min,
        'rule_max_T5C_P8A': rule_max,
        'rule_avg_T5C_P8A': rule_avg,
        'T5C_only':        lambda d: t5c(d),
        'P8A_only':        lambda d: p8a(d),
    }

    rows = []
    for rule_name, rule_fn in rules.items():
        real_s = rule_fn(real)
        fakes_s = {s: rule_fn(df) for s, df in fakes.items()}
        for substrate_name, mask in [('P2', p2_mask(real).to_numpy()), ('F4', f4_mask(real).to_numpy())]:
            for fpr_t in [0.02, 0.05, 0.10]:
                rec, tau = best_recall_at_fpr(real_s, fakes_s, fpr_t, mask)
                # Per-suite recall + tail FPRs
                per_suite = {f's_{name}': round(100*float((s > tau).mean()), 2) for name, s in fakes_s.items()}
                # Compute P2 FPR + F4 FPR + F0 FPR at this tau
                fpr_P2 = round(100 * fpr(real_s, tau, p2_mask(real).to_numpy()), 2)
                fpr_F4 = round(100 * fpr(real_s, tau, f4_mask(real).to_numpy()), 2)
                fpr_F0 = round(100 * fpr(real_s, tau), 2)
                rows.append({
                    'rule': rule_name,
                    'calib_substrate': substrate_name,
                    'target_FPR_pct': round(100*fpr_t, 2),
                    'tau': round(tau, 4),
                    'macro_recall_pct': round(100*rec, 2),
                    'fpr_P2_pct': fpr_P2,
                    'fpr_F4_pct': fpr_F4,
                    'fpr_F0_pct': fpr_F0,
                    **per_suite,
                })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, 'STAGE5_ENSEMBLE.csv')
    df.to_csv(out_csv, index=False)
    print(f'\nwrote {out_csv}\n')

    # Pretty print: side-by-side P2-cal @ FPR=5% across all rules
    for substrate in ['P2','F4']:
        for fpr_t in [2.0, 5.0, 10.0]:
            sub = df[(df.calib_substrate==substrate) & (df.target_FPR_pct==fpr_t)]
            print(f'\n=== {substrate}-calibrated, target FPR={fpr_t}% ===')
            cols = ['rule','tau','macro_recall_pct','fpr_P2_pct','fpr_F0_pct',
                    's_visomaster_enhanced_macro_dev','s_deeplive_enhanced_dev','s_teams_fake_all_dev']
            print(sub[cols].sort_values('macro_recall_pct', ascending=False).to_string(index=False))


if __name__ == '__main__':
    main()
