"""Stage 1: Run F4 substrate-cleaning eval on every candidate ckpt with frame CSVs.

Compiles one master scoreboard CSV with recall@FPR=5/10% per suite, plus a F0 baseline.
"""
from __future__ import annotations
import json
import os
import sys
import pandas as pd

ROOT = '/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training'
sys.path.insert(0, os.path.join(ROOT, 'analysis/substrate_cleaning_eval_2026-05-05'))
from run_clean_eval import evaluate_checkpoint, load_parquet_meta  # type: ignore  # noqa: E402

OUT_DIR = os.path.join(ROOT, 'analysis/best_candidate_search_2026-05-13/_stage1')
os.makedirs(OUT_DIR, exist_ok=True)

CPU_FU = os.path.join(ROOT, 'analysis/cpu_followups_2026-05-04/raw_reports')
PA_PC  = os.path.join(ROOT, 'analysis/pa_pc_eval_2026-05-05/raw_reports')
T3_DIR = os.path.join(ROOT, 'analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs')
STAGEA = os.path.join(ROOT, 'analysis/cpu_diagnostics_2026-05-12_stage_a/_scorecard_reports')
SLOT1  = os.path.join(ROOT, 'analysis/r13_overnight_slot1_shift_analysis_2026-05-13/_frame_cache')
P1_DIR = os.path.join(ROOT, 'analysis/p1_pe_eval_2026-05-07/raw_reports/phase_a')
P2_DIR = os.path.join(ROOT, 'analysis/p2_eval_2026-05-08/p2_deeper_analysis/frame_reports')
PD_DIR = os.path.join(ROOT, 'analysis/pd_scorecard_artifacts_2026-05-06/reports_original')


def _ck(name, dir_, tag):
    """Build (real, viso, deeplive, teams_fake) tuple for a ckpt-tag in given dir."""
    return {
        'ckpt_name': name,
        'real_csv':   os.path.join(dir_, f'teams_real_all_dev_{tag}_frames_report.csv'),
        'fake_csvs': {
            'visomaster_enhanced_macro_dev': os.path.join(dir_, f'visomaster_enhanced_macro_dev_{tag}_frames_report.csv'),
            'deeplive_enhanced_dev':         os.path.join(dir_, f'deeplive_enhanced_dev_{tag}_frames_report.csv'),
            'teams_fake_all_dev':            os.path.join(dir_, f'teams_fake_all_dev_{tag}_frames_report.csv'),
        },
    }


CANDIDATES = [
    # === Classic baselines ===
    _ck('P8A_step5000',           CPU_FU, 'p8a_reference_step5000'),
    _ck('E2B_step3200',           CPU_FU, 'e2b_top_n_step3200'),
    _ck('E3_step4800',            CPU_FU, 'e3_top_n_step4800'),
    _ck('E3_step6600',            CPU_FU, 'e3_top_n_step6600'),
    _ck('E3_step7200',            CPU_FU, 'e3_top_n_step7200'),
    # === PA / PC ===
    _ck('PA_top_n_step3800',      PA_PC,  'pa_top_n_step3800'),
    _ck('PA_top_n_step5600',      PA_PC,  'pa_top_n_step5600'),
    _ck('PA_periodic_step5000',   PA_PC,  'pa_periodic_step5000'),
    _ck('PC_top_n_step5400',      PA_PC,  'pc_top_n_step5400'),
    _ck('PC_top_n_step7400',      PA_PC,  'pc_top_n_step7400'),
    _ck('PC_periodic_step5000',   PA_PC,  'pc_periodic_step5000'),
    # === T3 slot 1/2/3 ===
    _ck('T3_SLOT1_step1500', os.path.join(T3_DIR, 'slot1_step1500'), 't3_slot1_periodic_step1500'),
    _ck('T3_SLOT1_step2500', os.path.join(T3_DIR, 'slot1_step2500'), 't3_slot1_periodic_step2500'),
    _ck('T3_SLOT2_step1000', os.path.join(T3_DIR, 'slot2_step1000'), 't3_slot2_periodic_step1000'),
    _ck('T3_SLOT3_step1500', os.path.join(T3_DIR, 'slot3_step1500'), 't3_slot3_periodic_step1500'),
    _ck('T3_SLOT3_step3500', os.path.join(T3_DIR, 'slot3_step3500'), 't3_slot3_periodic_step3500'),
    # === T5C ===
    _ck('T5C_step3500',           STAGEA, 't5c_periodic_step3500'),
    # === Today's batch — Slot 1 LoRA-P8A ===
    _ck('SLOT1_LORA_step1500',    SLOT1,  'slot1_lora_p8a_periodic_step1500'),
    _ck('SLOT1_LORA_step3500',    SLOT1,  'slot1_lora_p8a_periodic_step3500'),
    _ck('SLOT1_LORA_TopN_step2000', SLOT1, 'slot1_lora_p8a_top_n_step2000'),
    # === P1 / P2 ===
    _ck('P1_bundle_step500',      P1_DIR, 'p1_bundle_periodic_step500'),
    _ck('P1_bundle_step3750',     P1_DIR, 'p1_bundle_top_n_step3750'),
    _ck('P1_bundle_step4000',     P1_DIR, 'p1_bundle_top_n_step4000'),
    _ck('P1_pairrank_step500',    P1_DIR, 'p1_pairrank_periodic_step500'),
    _ck('P1_pairrank_step6000',   P1_DIR, 'p1_pairrank_top_n_step6000'),
    _ck('P1_pairrank_step6750',   P1_DIR, 'p1_pairrank_top_n_step6750'),
    _ck('P2C_pairrank_step3000',  P2_DIR, 'p2_c_pairrank_periodic_step3000'),
    _ck('P2C_pairrank_step7000',  P2_DIR, 'p2_c_pairrank_top_n_step7000'),
    _ck('P2D_fourier_step3000',   P2_DIR, 'p2_d_fourier_periodic_step3000'),
    _ck('P2D_fourier_step8000',   P2_DIR, 'p2_d_fourier_periodic_step8000'),
    _ck('P2D_fourier_step19000',  P2_DIR, 'p2_d_fourier_top_n_step19000'),
    # === PD (correlation-penalty) ===
    _ck('PD_viso_step600',        PD_DIR, 'viso_corr_top_n_step600'),
    _ck('PD_viso_step1000',       PD_DIR, 'viso_corr_periodic_step1000'),
    _ck('PD_viso_step2000',       PD_DIR, 'viso_corr_periodic_step2000'),
    _ck('PD_dl_step1800',         PD_DIR, 'deeplive_corr_top_n_step1800'),
    _ck('PD_dl_step2000',         PD_DIR, 'deeplive_corr_periodic_step2000'),
    _ck('PD_dl_step4800',         PD_DIR, 'deeplive_corr_top_n_step4800'),
]


def main():
    print(f'loading parquet meta...')
    parquet_meta = load_parquet_meta()
    print(f'parquet rows: {len(parquet_meta):,}')
    print(f'running F4 eval on {len(CANDIDATES)} candidates...')
    rows = []
    fails = []
    for spec in CANDIDATES:
        name = spec['ckpt_name']
        missing = [p for p in [spec['real_csv'], *spec['fake_csvs'].values()] if not os.path.exists(p)]
        if missing:
            print(f'  SKIP {name}: missing {len(missing)} files')
            fails.append((name, missing))
            continue
        try:
            summary = evaluate_checkpoint(
                ckpt_name=name,
                real_csv=spec['real_csv'],
                fake_csvs=spec['fake_csvs'],
                out_dir=OUT_DIR,
                parquet_meta=parquet_meta,
            )
        except Exception as e:
            print(f'  ERR {name}: {e}')
            fails.append((name, str(e)))
            continue
        per_suite = {r['fake_suite']: r for r in summary['per_suite']}
        row = {
            'ckpt': name,
            'n_real_F0': summary['n_real_frames_F0'],
            'n_real_F4': summary['n_real_frames_F4'],
            'tau_F0_at_FPR10': summary['tau_F0_calibrated'],
            'tau_F4_at_FPR10': summary['tau_F4_at_FPR_10pct'],
            'tau_F4_at_FPR5':  summary['tau_F4_at_FPR_5pct'],
            'fpr_F4_at_tauF0_pct': summary['fpr_F4_pct'],
        }
        for suite, r in per_suite.items():
            short = suite.replace('_enhanced_macro_dev','').replace('_enhanced_dev','').replace('_all_dev','')
            row[f'{short}_F0_pct']      = r['recall_at_tau_F0_pct']
            row[f'{short}_F4_FPR10_pct']= r['recall_at_tau_F4_FPR10_pct']
            row[f'{short}_F4_FPR5_pct'] = r['recall_at_tau_F4_FPR5_pct']
        # Macro-fake recall at F4 (average of three suites at FPR=10%)
        f4_10 = [r['recall_at_tau_F4_FPR10_pct'] for r in summary['per_suite']]
        f4_5  = [r['recall_at_tau_F4_FPR5_pct']  for r in summary['per_suite']]
        row['macro_F4_FPR10_pct'] = round(sum(f4_10)/len(f4_10), 2) if f4_10 else None
        row['macro_F4_FPR5_pct']  = round(sum(f4_5)/len(f4_5), 2)   if f4_5  else None
        rows.append(row)
        print(f'  {name}: macro@FPR10={row["macro_F4_FPR10_pct"]} macro@FPR5={row["macro_F4_FPR5_pct"]}')

    out_csv = os.path.join(OUT_DIR, 'STAGE1_SCOREBOARD.csv')
    df = pd.DataFrame(rows)
    df = df.sort_values('macro_F4_FPR10_pct', ascending=False, na_position='last')
    df.to_csv(out_csv, index=False)
    print(f'\nwrote {out_csv}')
    print(f'fails: {len(fails)}')
    for name, why in fails:
        print(f'  - {name}: {why if isinstance(why,str) else len(why)} missing')

    summary_path = os.path.join(OUT_DIR, '_fails.json')
    with open(summary_path, 'w') as f:
        json.dump(fails, f, indent=2, default=str)


if __name__ == '__main__':
    main()
