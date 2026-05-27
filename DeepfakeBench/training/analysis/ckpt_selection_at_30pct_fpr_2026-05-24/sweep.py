"""Find the best ckpt at ≤30% per-frame FPR target on diverse deployment-relevant data.

User constraint: production uses MV > 50% per N-frame window. If per-frame FPR ≤ ~30%,
MV-simple-maj is robust (per the MV simulation analysis). So the question becomes:
  Which ckpt has the highest fake recall at the τ where per-frame FPR = 30%?

Data sources (per-frame caches available):
  - cpu_diagnostics_2026-05-12_stage_a/_scorecard_reports/ — P8A, T5C, T3_SLOT1 on full suites
  - iq_shortcut_decomp_2026-05-08/scores_cache/ — P8A, E2B on viso + HDTF suites
  - auto_mode_2026-05-16_eval/scorecard_outputs/ — Slot_A_v2 (CLS), Slot_B on teams_real
  - face_pool_scorecard_2026-05-22/reports/ — Slot_A_v2 FACE-pool per-video (=per-frame on viso suite)
  - team_identity_deploy_readout_expanded_2026-05-23/outputs/per_frame_full.csv — P8A, T5C, E2B,
    SlotAv2_CLS, SlotAv2_FACE all scored on team-identity (dor 32 variants, Xinhe, Xiang)
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('.')

# ---- Suites to evaluate ----
# Aggregating from multiple sources to maximize coverage; map suite -> (frame_prob col, file pattern)

# Sources for per-frame stage_a data (P8A, T5C, T3_SLOT1)
STAGE_A = ROOT / "analysis/cpu_diagnostics_2026-05-12_stage_a/_scorecard_reports"
AUTO_MODE = ROOT / "analysis/auto_mode_2026-05-16_eval/scorecard_outputs"
IQ_CACHE = ROOT / "analysis/iq_shortcut_decomp_2026-05-08/scores_cache"
FACE_POOL = ROOT / "analysis/face_pool_scorecard_2026-05-22/reports"

# ckpt → mapping suite → (file path, score column)
CKPT_SUITES = {
    "P8A_step5000": {
        "viso_enhanced_macro_dev":   (IQ_CACHE / "P8A_REFERENCE_STEP5000__visomaster_enhanced_macro_dev.csv", "frame_prob"),
        "teams_fake_all_dev":         (IQ_CACHE / "P8A_REFERENCE_STEP5000__teams_fake_all_dev.csv", "frame_prob"),
        "teams_fake_all_lockbox":     (IQ_CACHE / "P8A_REFERENCE_STEP5000__teams_fake_all_lockbox.csv", "frame_prob"),
        "deeplive_enhanced_dev":      (IQ_CACHE / "P8A_REFERENCE_STEP5000__deeplive_enhanced_dev.csv", "frame_prob"),
        "teams_real_all_dev":         (IQ_CACHE / "P8A_REFERENCE_STEP5000__teams_real_all_dev.csv", "frame_prob"),
        "teams_real_all_lockbox":     (IQ_CACHE / "P8A_REFERENCE_STEP5000__teams_real_all_lockbox.csv", "frame_prob"),
        "teams_real_dor_dev":         (IQ_CACHE / "P8A_REFERENCE_STEP5000__teams_real_dor_dev.csv", "frame_prob"),
        "teams_real_lighting_extreme_dev": (IQ_CACHE / "P8A_REFERENCE_STEP5000__teams_real_lighting_extreme_dev.csv", "frame_prob"),
        "teams_real_poor_quality_dev": (IQ_CACHE / "P8A_REFERENCE_STEP5000__teams_real_poor_quality_dev.csv", "frame_prob"),
        "hdtf_real_clean_dev":        (IQ_CACHE / "P8A_REFERENCE_STEP5000__hdtf_real_clean_dev.csv", "frame_prob"),
        "hdtf_real_teams_dev":        (IQ_CACHE / "P8A_REFERENCE_STEP5000__hdtf_real_teams_dev.csv", "frame_prob"),
        "hdtf_real_teams_lockbox":    (IQ_CACHE / "P8A_REFERENCE_STEP5000__hdtf_real_teams_lockbox.csv", "frame_prob"),
        "hdtf_fake_teams_dev":        (IQ_CACHE / "P8A_REFERENCE_STEP5000__hdtf_fake_teams_dev.csv", "frame_prob"),
        "hdtf_fake_teams_lockbox":    (IQ_CACHE / "P8A_REFERENCE_STEP5000__hdtf_fake_teams_lockbox.csv", "frame_prob"),
    },
    "T5C_step3500": {
        "viso_enhanced_macro_dev":   (STAGE_A / "visomaster_enhanced_macro_dev_t5c_periodic_step3500_frames_report.csv", "prob"),
        "teams_fake_all_dev":         (STAGE_A / "teams_fake_all_dev_t5c_periodic_step3500_frames_report.csv", "prob"),
        "teams_fake_all_lockbox":     (STAGE_A / "teams_fake_all_lockbox_t5c_periodic_step3500_frames_report.csv", "prob"),
        "deeplive_enhanced_dev":      (STAGE_A / "deeplive_enhanced_dev_t5c_periodic_step3500_frames_report.csv", "prob"),
        "teams_real_all_dev":         (STAGE_A / "teams_real_all_dev_t5c_periodic_step3500_frames_report.csv", "prob"),
        "teams_real_all_lockbox":     (STAGE_A / "teams_real_all_lockbox_t5c_periodic_step3500_frames_report.csv", "prob"),
        "teams_real_dor_dev":         (STAGE_A / "teams_real_dor_dev_t5c_periodic_step3500_frames_report.csv", "prob"),
        "teams_real_lighting_extreme_dev": (STAGE_A / "teams_real_lighting_extreme_dev_t5c_periodic_step3500_frames_report.csv", "prob"),
        "teams_real_poor_quality_dev": (STAGE_A / "teams_real_poor_quality_dev_t5c_periodic_step3500_frames_report.csv", "prob"),
    },
    "T3_SLOT1_step1500": {
        "viso_enhanced_macro_dev":   (STAGE_A / "visomaster_enhanced_macro_dev_t3_slot1_periodic_step1500_frames_report.csv", "prob"),
        "teams_fake_all_dev":         (STAGE_A / "teams_fake_all_dev_t3_slot1_periodic_step1500_frames_report.csv", "prob"),
        "teams_fake_all_lockbox":     (STAGE_A / "teams_fake_all_lockbox_t3_slot1_periodic_step1500_frames_report.csv", "prob"),
        "deeplive_enhanced_dev":      (STAGE_A / "deeplive_enhanced_dev_t3_slot1_periodic_step1500_frames_report.csv", "prob"),
        "teams_real_all_dev":         (STAGE_A / "teams_real_all_dev_t3_slot1_periodic_step1500_frames_report.csv", "prob"),
        "teams_real_all_lockbox":     (STAGE_A / "teams_real_all_lockbox_t3_slot1_periodic_step1500_frames_report.csv", "prob"),
        "teams_real_dor_dev":         (STAGE_A / "teams_real_dor_dev_t3_slot1_periodic_step1500_frames_report.csv", "prob"),
        "teams_real_lighting_extreme_dev": (STAGE_A / "teams_real_lighting_extreme_dev_t3_slot1_periodic_step1500_frames_report.csv", "prob"),
        "teams_real_poor_quality_dev": (STAGE_A / "teams_real_poor_quality_dev_t3_slot1_periodic_step1500_frames_report.csv", "prob"),
    },
    "E2B_step3200": {
        "viso_enhanced_macro_dev":   (IQ_CACHE / "E2B_TOP_N_STEP3200__visomaster_enhanced_macro_dev.csv", "frame_prob"),
        "teams_fake_all_dev":         (IQ_CACHE / "E2B_TOP_N_STEP3200__teams_fake_all_dev.csv", "frame_prob"),
        "teams_fake_all_lockbox":     (IQ_CACHE / "E2B_TOP_N_STEP3200__teams_fake_all_lockbox.csv", "frame_prob"),
        "deeplive_enhanced_dev":      (IQ_CACHE / "E2B_TOP_N_STEP3200__deeplive_enhanced_dev.csv", "frame_prob"),
        "teams_real_all_dev":         (IQ_CACHE / "E2B_TOP_N_STEP3200__teams_real_all_dev.csv", "frame_prob"),
        "teams_real_all_lockbox":     (IQ_CACHE / "E2B_TOP_N_STEP3200__teams_real_all_lockbox.csv", "frame_prob"),
        "teams_real_dor_dev":         (IQ_CACHE / "E2B_TOP_N_STEP3200__teams_real_dor_dev.csv", "frame_prob"),
        "teams_real_lighting_extreme_dev": (IQ_CACHE / "E2B_TOP_N_STEP3200__teams_real_lighting_extreme_dev.csv", "frame_prob"),
        "teams_real_poor_quality_dev": (IQ_CACHE / "E2B_TOP_N_STEP3200__teams_real_poor_quality_dev.csv", "frame_prob"),
    },
    "SLOT_A_v2_anchor_aware_step3500": {
        # CLS-pool: teams_real / teams_fake only from auto_mode + cpu_diagnostics_2026-05-19_pre_plan
        "teams_real_all_dev":         (AUTO_MODE / "teams_real_all_dev_slot_a_anchor_aware_step3500_frames_report.csv", "prob"),
        "teams_real_all_lockbox":     (AUTO_MODE / "teams_real_all_lockbox_slot_a_anchor_aware_step3500_frames_report.csv", "prob"),
        "teams_fake_all_lockbox":     (ROOT / "analysis/cpu_diagnostics_2026-05-19_pre_plan/gcs_cache_auto_mode/teams_fake_all_lockbox_slot_a_anchor_aware_step3500_frames_report.csv", "prob"),
    },
    "SLOT_A_v2_anchor_aware_step1500": {
        "teams_real_all_dev":         (AUTO_MODE / "teams_real_all_dev_slot_a_anchor_aware_step1500_frames_report.csv", "prob"),
        "teams_real_all_lockbox":     (AUTO_MODE / "teams_real_all_lockbox_slot_a_anchor_aware_step1500_frames_report.csv", "prob"),
        "teams_fake_all_lockbox":     (ROOT / "analysis/cpu_diagnostics_2026-05-19_pre_plan/gcs_cache_auto_mode/teams_fake_all_lockbox_slot_a_anchor_aware_step1500_frames_report.csv", "prob"),
    },
}

# Auto-detect 'prob' column name
def load_csv(path):
    df = pd.read_csv(path)
    return df

def find_prob_col(df):
    for cand in ("prob","frame_prob","fake_prob","p","probability","score"):
        if cand in df.columns:
            return cand
    # Fallback: a numeric column
    for c in df.columns:
        if df[c].dtype in (np.float64, np.float32):
            return c
    raise ValueError(f"No prob col found: {list(df.columns)}")

# ---- Build per-ckpt suite-score dict {ckpt: {suite: np.array of probs, labels}} ----
data = {}  # data[ckpt][suite] = (probs, is_fake)
for ckpt, suites in CKPT_SUITES.items():
    data[ckpt] = {}
    for suite, (path, col) in suites.items():
        if not path.exists():
            continue
        df = load_csv(path)
        prob_col = col if col in df.columns else find_prob_col(df)
        probs = df[prob_col].to_numpy()
        is_fake = ("fake" in suite or "visomaster" in suite or "deeplive" in suite)
        data[ckpt][suite] = (probs, is_fake)

# Print coverage
print("=" * 110)
print("Coverage summary (n_frames per ckpt × suite)")
print("=" * 110)
all_suites = sorted({s for d in data.values() for s in d.keys()})
print(f"{'ckpt':40s} " + " ".join(f"{s[:20]:>20s}" for s in all_suites))
for ckpt, suites in data.items():
    print(f"{ckpt:40s} " + " ".join(f"{len(suites.get(s, (np.array([]),))[0]):>20d}" for s in all_suites))

# ---- For each ckpt, sweep τ and find τ such that teams_real_all_lockbox FPR ≈ 30% ----
TARGET_FPR = 0.30
FPR_GATE_SUITE = "teams_real_all_lockbox"

print()
print("=" * 110)
print(f"τ selection: find τ such that {FPR_GATE_SUITE} per-frame FPR ≤ {TARGET_FPR:.0%}")
print("=" * 110)
tau_selected = {}
for ckpt, suites in data.items():
    if FPR_GATE_SUITE not in suites:
        tau_selected[ckpt] = None
        continue
    real_probs, _ = suites[FPR_GATE_SUITE]
    # Find τ such that frac(probs >= τ) just barely <= TARGET_FPR
    sorted_probs = np.sort(real_probs)[::-1]
    n = len(sorted_probs)
    target_n = int(np.floor(TARGET_FPR * n))
    if target_n == 0:
        # min τ that gives 0 FPR
        tau = float(sorted_probs[0]) + 1e-9
    elif target_n >= n:
        tau = -np.inf
    else:
        tau = float(sorted_probs[target_n])
    tau_selected[ckpt] = tau
    achieved = (real_probs >= tau).mean()
    print(f"  {ckpt:40s} τ={tau:.4f}  achieves FPR={achieved:.2%}  (n_real={n})")

# ---- At each ckpt's τ, measure fake recall across all suites ----
print()
print("=" * 130)
print(f"At τ chosen for {FPR_GATE_SUITE} FPR ≤ {TARGET_FPR:.0%} — fake recall on all fake suites + cross-FPR sanity")
print("=" * 130)

fake_suites_of_interest = [
    "viso_enhanced_macro_dev",
    "deeplive_enhanced_dev",
    "teams_fake_all_dev",
    "teams_fake_all_lockbox",
    "hdtf_fake_teams_dev",
    "hdtf_fake_teams_lockbox",
]
real_suites_of_interest = [
    "teams_real_all_dev",
    "teams_real_all_lockbox",
    "teams_real_dor_dev",
    "teams_real_lighting_extreme_dev",
    "teams_real_poor_quality_dev",
    "hdtf_real_clean_dev",
    "hdtf_real_teams_dev",
    "hdtf_real_teams_lockbox",
]

# Build summary table
rows = []
for ckpt, tau in tau_selected.items():
    if tau is None:
        continue
    row = {"ckpt": ckpt, "tau": round(tau, 4)}
    for s in fake_suites_of_interest:
        if s in data[ckpt]:
            probs, _ = data[ckpt][s]
            row[f"rec/{s}"] = round((probs >= tau).mean(), 4)
            row[f"n/{s}"] = len(probs)
        else:
            row[f"rec/{s}"] = np.nan
            row[f"n/{s}"] = 0
    for s in real_suites_of_interest:
        if s in data[ckpt]:
            probs, _ = data[ckpt][s]
            row[f"fpr/{s}"] = round((probs >= tau).mean(), 4)
        else:
            row[f"fpr/{s}"] = np.nan
    rows.append(row)
summary = pd.DataFrame(rows)
pd.set_option('display.float_format','{:.3f}'.format)
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 50)

# Print recall side
print("\nFAKE RECALL (at τ achieving ≤30% lockbox real FPR):")
rec_cols = ['ckpt','tau'] + [f"rec/{s}" for s in fake_suites_of_interest]
print(summary[rec_cols].to_string(index=False))

print("\nREAL FPR (at same τ — sanity across other real cohorts):")
fpr_cols = ['ckpt','tau'] + [f"fpr/{s}" for s in real_suites_of_interest]
print(summary[fpr_cols].to_string(index=False))

# Aggregate score: harmonic mean of (viso, deeplive, teams_fake_all_dev, teams_fake_all_lockbox)
# (excluding HDTF since HDTF substrate is much easier)
print()
print("=" * 110)
print("AGGREGATE RANKING — geometric mean across {viso_enh_macro, deeplive_enh, teams_fake_all_dev, teams_fake_all_lockbox}")
print("=" * 110)
key_fakes = ['viso_enhanced_macro_dev', 'deeplive_enhanced_dev', 'teams_fake_all_dev', 'teams_fake_all_lockbox']
for r in rows:
    vals = [r.get(f"rec/{s}", np.nan) for s in key_fakes]
    have = [v for v in vals if not np.isnan(v)]
    if len(have) > 0:
        # Geometric mean (better than arithmetic for ranking when some are very low)
        gm = np.exp(np.mean(np.log(np.maximum(np.array(have), 1e-6))))
        r['geom_mean_recall'] = round(gm, 4)
        r['n_suites_covered'] = len(have)

ranked = sorted([r for r in rows if 'geom_mean_recall' in r], key=lambda x: -x['geom_mean_recall'])
print()
print(f"{'rank':>4s} {'ckpt':40s} {'τ':>7s} {'geomean':>8s} {'n_suites':>9s}   key-fake breakdown")
for i, r in enumerate(ranked, 1):
    parts = " ".join(f"{s.replace('_enhanced_macro_dev','').replace('_enhanced_dev','').replace('_all_dev','/dev').replace('_all_lockbox','/lockbox')[:10]:>10s}={r.get(f'rec/{s}', float('nan')):>5.2f}" for s in key_fakes)
    print(f"  {i:>3d}. {r['ckpt']:40s} {r['tau']:>7.3f} {r['geom_mean_recall']:>8.3f} {r['n_suites_covered']:>9d}   {parts}")

# ---- Now add the team-identity per-frame data (P8A, T5C, E2B, SlotAv2_CLS, SlotAv2_FACE on dor variants + Xinhe + Xiang) ----
print()
print("=" * 110)
print("TEAM-IDENTITY CROSS-CKPT (P8A, T5C, E2B, SlotAv2_CLS, SlotAv2_FACE all on same 6,439 frames)")
print("=" * 110)
ti = pd.read_csv('analysis/team_identity_deploy_readout_expanded_2026-05-23/outputs/per_frame_full.csv')
real = ti[ti['role']=='real']
fakes = ti[ti['role'].str.startswith('fake')]
dor_fakes = ti[ti['role']=='fake_target_dor']

# For each ckpt-prob-column, find τ s.t. teams_real_all_lockbox approx (using deploy_relevant=True as lockbox proxy)
prob_cols = ['prob_P8A', 'prob_T5C', 'prob_E2B', 'prob_SlotAv2_CLS', 'prob_SlotAv2_FACE']
# For lockbox FPR target, we use real-team-identity frames (5 humans, deploy_relevant=True; same data as user's deployment)
real_deploy = real[real['deploy_relevant']==True]
print(f"Using {len(real_deploy)} deploy-relevant real-team-identity frames as FPR gate")

ti_rows = []
for col in prob_cols:
    probs_real = real_deploy[col].to_numpy()
    sorted_p = np.sort(probs_real)[::-1]
    n = len(sorted_p)
    target_n = int(np.floor(TARGET_FPR * n))
    tau = float(sorted_p[target_n]) if target_n < n else -np.inf
    achieved = (probs_real >= tau).mean()
    # Per-human FPR
    fprs = real_deploy.groupby('human').apply(lambda g: (g[col]>=tau).mean())
    # Per-role recall
    recs = fakes.groupby('role').apply(lambda g: (g[col]>=tau).mean())
    # Dor variant — worst, median, best, count of variants ≥ 90% recall
    dor_var_recs = dor_fakes.groupby('base_identity').apply(lambda g: (g[col]>=tau).mean())
    row = {
        'ckpt': col, 'tau': round(tau, 4), 'overall_fpr': round(achieved, 4),
        'worst_human_fpr': round(fprs.max(), 4), 'fprs': fprs.to_dict(),
        'dor_recall': round(recs.get('fake_target_dor', np.nan), 4),
        'xinhe_recall': round(recs.get('fake_target_Xinhe', np.nan), 4),
        'xiang_recall': round(recs.get('fake_target_Xiang', np.nan), 4),
        'dor_var_median': round(dor_var_recs.median(), 4),
        'dor_var_worst': round(dor_var_recs.min(), 4),
        'dor_var_n_high': int((dor_var_recs >= 0.9).sum()),
        'dor_var_total': len(dor_var_recs),
    }
    ti_rows.append(row)
ti_df = pd.DataFrame(ti_rows)
print()
print("At τ chosen so overall FPR ≤ 30%:")
print(ti_df[['ckpt','tau','overall_fpr','worst_human_fpr','dor_recall','xinhe_recall','xiang_recall','dor_var_median','dor_var_worst','dor_var_n_high']].to_string(index=False))
print()
print("Per-human FPR detail:")
for r in ti_rows:
    fpr_strs = ', '.join(f"{h}={v:.1%}" for h,v in r['fprs'].items())
    print(f"  {r['ckpt']:25s} τ={r['tau']:.3f}: {fpr_strs}")

# Save
Path('analysis/ckpt_selection_at_30pct_fpr_2026-05-24').mkdir(parents=True, exist_ok=True)
summary.to_csv('analysis/ckpt_selection_at_30pct_fpr_2026-05-24/contract_suite_summary.csv', index=False)
ti_df.drop(columns=['fprs']).to_csv('analysis/ckpt_selection_at_30pct_fpr_2026-05-24/team_identity_summary.csv', index=False)
print()
print("=" * 110)
print("Outputs saved to analysis/ckpt_selection_at_30pct_fpr_2026-05-24/")
