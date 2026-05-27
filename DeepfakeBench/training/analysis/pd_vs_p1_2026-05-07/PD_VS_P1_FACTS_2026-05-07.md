# PD vs P1 head-to-head at calibrated τ — FACTS

**Date authored**: 2026-05-07
**Author**: Claude (research-agent)
**Discipline**: FACTS-doc (no opinions; one-line interpretive observations only against numerics shown in-line)

## Question

Of the two anti-shortcut lever classes evaluated this week — **PD's correlation_penalty(λ=1.0) on sharpness/luma/face_area** vs **P1's pair_rank + GroupDRO bundle** — which produces the cleaner joint-performance checkpoint on the Teams promotion contract at **calibrated τ** (not τ=0.5)? Both packets target the documented R13 image-quality / face-size shortcut class. Both arms FT from the same `E2B_TOP_N_STEP3200` base.

## Method

### What was already available
- **P1 scorecard** at calibrated τ: `analysis/p1_pe_eval_2026-05-07/scorecard/selected_threshold_scorecard.csv` (4 P1 ckpts + P8A + E2B baselines, all 9 contract suites, calibrated to `target_real_fpr=0.07`, `target_stress_fpr=0.10`, then maximized on `dev_fake_macro_recall`).
- **PD scorecard** at τ=0.5 only: GCS `gs://training-job-outputs/test_results/pd_corr_penalty/pd-corr-penalty-scorecard-2026-05-06/reports/`, 8 ckpts × 20 suites, per-video probability `videos_report.csv` files. The launcher labels the τ=0.5 numbers "diagnostic-only" — there is no calibrated τ scorecard for PD.
- Local cache `analysis/pd_scorecard_artifacts_2026-05-06/` confirms the τ=0.5 caveat in its README §Caveats.

### What I did
1. Pulled `pd-corr-penalty-scorecard-2026-05-06/reports/*videos_report.csv` (157 files) into `analysis/pd_vs_p1_2026-05-07/raw/pd_videos_reports/` via gcloud (gcloud auth verified, `roee@dtectvision.ai`).
2. **Re-calibrated τ for each PD checkpoint to match P1's contract policy**:
   - candidate grid = 10001-step linear union 0..1 ∪ percentiles of (real_dev ∪ all dev fakes)
   - filter τ such that real_FPR(`teams_real_all_dev`) ≤ 0.07 AND each stress-suite real_FPR ≤ 0.10
   - among survivors, select τ that maximizes mean fake_recall over (`teams_fake_all_dev`, `visomaster_enhanced_macro_dev`, `deeplive_enhanced_dev`); ties broken by smaller τ
3. Applied that calibrated τ across all 9 contract suites for each PD ckpt; loaded P1 ckpts directly from P1's CSV at their already-calibrated τ.
4. Sanity check: P8A and E2B baselines from the **PD eval run** vs the **P1 eval run** match within 0.0019 in calibrated τ and ≤0.002 in any per-suite FPR/recall — confirms the two scorecard runs use comparable substrates and the calibration logic is sound.

### Builder script and outputs
- Script: `analysis/pd_vs_p1_2026-05-07/build_head_to_head.py`
- Long form: `analysis/pd_vs_p1_2026-05-07/head_to_head_at_calibrated_tau.csv` (108 rows = 12 ckpts × 9 suites)
- Wide form: `analysis/pd_vs_p1_2026-05-07/head_to_head_pivot.csv` (12 rows × 9 suites + τ + dev_macro_recall)

## Head-to-head table at calibrated τ (lex policy: real_FPR ≤ 7%, stress_FPR ≤ 10%, then max dev_fake_macro_recall)

Cells display `real_fpr` for real suites and `fake_recall` for fake suites. All values are video-level. All rows are at each row's own calibrated τ. **τ varies per row** (this is intentional — single-tau calibration applied independently per checkpoint, exactly mirroring P1's promotion-contract scorecard).

| Checkpoint | τ | dev_macro_recall | teams_real_all_dev (FPR) | teams_real_poor_quality_dev (FPR) | teams_real_lighting_extreme_dev (FPR) | teams_fake_all_dev (recall) | visomaster_enhanced_macro_dev (recall) | deeplive_enhanced_dev (recall) | teams_real_all_lockbox (FPR) | teams_fake_all_lockbox (recall) | teams_real_dor_dev (FPR) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 (PD eval) | 0.914 | **0.300** | 0.070 | 0.026 | 0.069 | 0.526 | 0.136 | 0.239 | **0.018** | 0.387 | 0.080 |
| P8A_REFERENCE_STEP5000 (P1 eval) | 0.916 | 0.300 | 0.070 | 0.026 | 0.069 | 0.526 | 0.135 | 0.239 | 0.018 | 0.387 | 0.080 |
| E2B_TOP_N_STEP3200 (PD eval) | 0.711 | **0.508** | 0.067 | 0.081 | 0.100 | 0.678 | 0.051 | 0.796 | 0.024 | 0.628 | 0.120 |
| E2B_TOP_N_STEP3200 (P1 eval) | 0.711 | 0.508 | 0.067 | 0.081 | 0.100 | 0.678 | 0.051 | 0.796 | 0.024 | 0.628 | 0.120 |
| **DEEPLIVE_CORR_PERIODIC_STEP2000** | 0.805 | 0.532 | 0.068 | 0.073 | 0.100 | 0.718 | 0.136 | 0.741 | 0.038 | 0.708 | **0.620** |
| **DEEPLIVE_CORR_TOP_N_STEP1800** | 0.675 | 0.562 | 0.066 | 0.068 | 0.100 | 0.736 | 0.136 | 0.813 | 0.047 | 0.672 | **0.640** |
| **DEEPLIVE_CORR_TOP_N_STEP4800** | 0.574 | **0.657** | 0.069 | 0.063 | 0.100 | **0.799** | 0.206 | **0.967** | **0.159** | 0.632 | **0.620** |
| **VISO_CORR_PERIODIC_STEP1000** | 0.878 | 0.256 | 0.061 | 0.066 | 0.100 | 0.525 | 0.091 | 0.151 | 0.016 | **0.937** | **0.460** |
| **VISO_CORR_PERIODIC_STEP2000** | 0.950 | 0.183 | 0.058 | 0.054 | 0.100 | 0.458 | 0.054 | 0.035 | 0.005 | 0.640 | 0.100 |
| **VISO_CORR_TOP_N_STEP600** | 0.774 | 0.416 | 0.064 | 0.076 | 0.100 | 0.633 | 0.114 | 0.501 | 0.027 | 0.621 | **0.480** |
| P1_BUNDLE_PERIODIC_STEP500 | 0.992 | 0.075 | 0.067 | 0.046 | 0.100 | 0.220 | 0.005 | **0.000** | 0.033 | **0.826** | 0.040 |
| P1_PAIRRANK_PERIODIC_STEP500 (P1 promotion winner) | 0.768 | 0.354 | 0.061 | 0.035 | 0.099 | 0.509 | 0.216 | 0.336 | 0.018 | 0.708 | 0.160 |

Bold = notable extreme value. PD ckpts highlighted in **bold** by name to distinguish from the P1 row.

## Direct observations

### O1. Sanity check — PD eval and P1 eval are on the same substrate
P8A and E2B baselines computed from PD's videos_reports vs P1's selected_threshold_scorecard.csv match within ≤0.002 absolute on every contract metric. The two evals are directly comparable — calibration logic and substrate are identical.

### O2. PD's deeplive arm achieves the highest dev_macro_recall in the comparison set
| ckpt | dev_macro_recall |
|---|---:|
| DEEPLIVE_CORR_TOP_N_STEP4800 | 0.657 |
| DEEPLIVE_CORR_TOP_N_STEP1800 | 0.562 |
| DEEPLIVE_CORR_PERIODIC_STEP2000 | 0.532 |
| E2B_TOP_N_STEP3200 (FT base) | 0.508 |
| VISO_CORR_TOP_N_STEP600 | 0.416 |
| P1_PAIRRANK_PERIODIC_STEP500 | 0.354 |
| P8A_REFERENCE_STEP5000 | 0.300 |
| VISO_CORR_PERIODIC_STEP1000 | 0.256 |
| VISO_CORR_PERIODIC_STEP2000 | 0.183 |
| P1_BUNDLE_PERIODIC_STEP500 | 0.075 |

PD's `DEEPLIVE_CORR_TOP_N_STEP4800` lifts dev_macro_recall +0.149pp over E2B FT-base and +0.357pp over P8A. P1's best ckpt (PAIRRANK_500) is **below** the FT-base E2B (-0.155pp) and roughly matches P8A (+0.054pp).

### O3. The PD deeplive arm has catastrophic dor invariance regression
PD deeplive ckpts FPR on `teams_real_dor_dev` (50 videos):

| ckpt | dor_FPR |
|---|---:|
| DEEPLIVE_CORR_PERIODIC_STEP2000 | 0.620 |
| DEEPLIVE_CORR_TOP_N_STEP1800 | 0.640 |
| DEEPLIVE_CORR_TOP_N_STEP4800 | 0.620 |
| VISO_CORR_TOP_N_STEP600 | 0.480 |
| VISO_CORR_PERIODIC_STEP1000 | 0.460 |
| VISO_CORR_PERIODIC_STEP2000 | 0.100 |
| P1_PAIRRANK_PERIODIC_STEP500 | 0.160 |
| E2B_TOP_N_STEP3200 (FT base) | 0.120 |
| VISO_CORR_PERIODIC_STEP2000 | 0.100 |
| P8A_REFERENCE_STEP5000 | 0.080 |
| P1_BUNDLE_PERIODIC_STEP500 | 0.040 |

5 of 6 PD ckpts have dor_FPR ≥ 0.46 (5-8× E2B's 0.12). This is the canonical Phase 1A `is_dor_shkedi`-axis regression documented in memory `project_p18_diagnostics_complete_2026-05-02.md`: when an FT lever fails to preserve E2B's dor invariance, the model collapses on the same-identity-axis shortcut. **Only VISO_CORR_PERIODIC_STEP2000 preserves dor invariance** (0.100 vs E2B 0.120) — but at the cost of dev_macro_recall = 0.183 (the worst in the comparison set).

### O4. PD's `DEEPLIVE_CORR_TOP_N_STEP4800` also breaks lockbox real invariance
| ckpt | lockbox_real_FPR |
|---|---:|
| DEEPLIVE_CORR_TOP_N_STEP4800 | 0.159 |
| DEEPLIVE_CORR_TOP_N_STEP1800 | 0.047 |
| DEEPLIVE_CORR_PERIODIC_STEP2000 | 0.038 |
| P1_BUNDLE_PERIODIC_STEP500 | 0.033 |
| VISO_CORR_TOP_N_STEP600 | 0.027 |
| E2B_TOP_N_STEP3200 (FT base) | 0.024 |
| P8A_REFERENCE_STEP5000 | 0.018 |
| P1_PAIRRANK_PERIODIC_STEP500 | 0.018 |
| VISO_CORR_PERIODIC_STEP1000 | 0.016 |
| VISO_CORR_PERIODIC_STEP2000 | 0.005 |

The arm with the highest dev_macro_recall (0.657) also has 9× lockbox real FPR vs P8A. Lockbox calibration is independent of dev calibration; this lockbox FPR was NOT a constraint in the dev-only contract policy and is a substrate-mismatch warning.

### O5. PD's viso arm has the highest lockbox fake recall (VISO_CORR_PERIODIC_STEP1000 = 0.937) but bombs everywhere else
VISO_CORR_PERIODIC_STEP1000 is the only ckpt in the comparison set with lockbox_fake_recall > 0.85, ahead of P1_BUNDLE (0.826) and far ahead of P8A (0.387). However it has dev_macro_recall = 0.256 (below E2B's 0.508), deeplive_enhanced_dev recall = 0.151, and dor_FPR = 0.460. **The single high-recall metric is not joint** — it doesn't generalize to dev fakes or preserve real invariance.

### O6. P1_PAIRRANK_500 (P1 promotion winner) clears the contract floors but at lower dev_macro than the FT base
P1_PAIRRANK_500 is the only ckpt that:
- clears the floor `dev_fake_macro_recall ≥ 0.30` (0.354)
- preserves dor_FPR ≤ 0.20 (0.160)
- preserves lockbox_real_FPR ≤ 0.025 (0.018)
- has lockbox_fake_recall ≥ 0.70 (0.708)

But its dev_macro 0.354 is **below E2B's 0.508** (FT-base regression of −0.155pp). The lift it offers is **lockbox transfer**, not raw dev recall: lockbox_fake_recall 0.708 vs E2B 0.628 (+0.080pp) at lockbox_real_FPR equal-or-better.

### O7. P1_BUNDLE_500 is operationally broken
Its calibrated τ=0.992 ceiling-saturates the dev fake suites: deeplive_enhanced_dev recall = 0.000, visomaster recall = 0.005. The FT signal cannot be operationalized at deployment-grade FPR.

### O8. None of the 11 ckpts in the comparison set clears the PD packet's own F1 close criterion
F1 = lockbox recall ≥ 0.90 at FPR ≤ 0.10 (excluding `is_no_face` / extreme low-IQ frames per `eval_substrate_data_hygiene.md`).

| ckpt | lockbox_real_FPR | lockbox_fake_recall |
|---|---:|---:|
| VISO_CORR_PERIODIC_STEP1000 | 0.016 | 0.937 |
| P1_BUNDLE_PERIODIC_STEP500 | 0.033 | 0.826 |
| DEEPLIVE_CORR_PERIODIC_STEP2000 | 0.038 | 0.708 |
| P1_PAIRRANK_PERIODIC_STEP500 | 0.018 | 0.708 |

**VISO_CORR_PERIODIC_STEP1000 is the only ckpt to clear F1 strictly** (0.937 ≥ 0.90 with lockbox_real_FPR 0.016 ≤ 0.10). However per O5 and O3, the same ckpt fails on dev fake suites and dor invariance.

## Direct verdict

Question: which lever class produces the cleaner joint-performance ckpt?

**Neither lever class produces a deployment-grade ckpt that is jointly clean across the 9-suite contract. PD lifts headline dev fake recall but pays in dor and lockbox real FPR; P1 preserves real invariance but does not lift dev fake recall above the FT-base E2B.**

Joint-performance reading by ckpt:
- **PD's deeplive arm** lifts dev fake recall (+0.149pp dev_macro vs E2B at the highest-recall ckpt) but consistently breaks dor invariance (5-8× E2B baseline), and the highest-recall ckpt also breaks lockbox real (9× P8A). This is the canonical "dev_fake gain at the price of substrate invariance" failure mode.
- **PD's viso arm** has one ckpt (PERIODIC_STEP1000) that beats every other ckpt in the comparison set on lockbox_fake_recall (0.937) but the same ckpt has the worst dev fake recall outside P1_BUNDLE — no joint signal.
- **P1's pairrank arm** preserves real invariance (dor 0.16, lockbox real 0.018) and gives a modest lockbox transfer lift (lockbox_fake +0.080pp over E2B) but does NOT lift dev fake recall above E2B (it regresses by 0.155pp).
- **P1's bundle arm** is operationally broken at calibrated τ — saturates the score ceiling on dev fakes (deeplive_enhanced_dev recall = 0).

The cleanest single ckpt **on real invariance + lockbox transfer** is **P1_PAIRRANK_PERIODIC_STEP500**. The cleanest single ckpt **on dev fake recall** is **DEEPLIVE_CORR_TOP_N_STEP4800**. They are not the same ckpt and neither dominates the other.

If one had to pick a single deployment ckpt from the comparison set under joint-performance discipline, the strict reading is **E2B_TOP_N_STEP3200 (the FT base) is not strictly dominated by any of the four FT children on the 9-suite contract**: dev_macro 0.508 (only DEEPLIVE_CORR ckpts beat it; all of those break dor/lockbox), dor_FPR 0.120 (only P8A and 2 ckpts beat it), lockbox_fake 0.628 (4 ckpts beat it; all by ≤0.10 except VISO_CORR_STEP1000 which fails dev). The two FT-children that beat E2B in lockbox_fake without breaking real invariance — P1_PAIRRANK and DEEPLIVE_CORR_PERIODIC — both regress on dev_macro vs E2B's 0.508.

This is consistent with the pattern documented in memories `project_p18_diagnostics_complete_2026-05-02.md` (GRL was defensive against FT regression, not additive over P8A) and `project_data_axis_lever_pulled_twice_no_lift.md` (data-axis lever pulled twice, no lift). **Both lever classes here are also pulled and produce no joint lift over E2B.** The FT-from-E2B trajectory continues to either preserve or degrade — the trajectory has not produced an FT-child that strictly dominates its base.

## Caveats

1. **PD scorecard ran τ=0.5 only** in its native pipeline; my calibrated-τ values are derived by re-running calibration logic on the per-video probabilities. The scorecard launcher labels τ=0.5 outputs "diagnostic-only" because dev calibration is the deployment path. My re-calibration uses identical policy to P1 (lex on real_FPR ≤ 0.07, stress_FPR ≤ 0.10, then max dev_macro_recall) so the two are policy-comparable.
2. **The candidate τ grid** I built (10001-step linear ∪ percentiles) is finer than P1's 5,549-point grid (`threshold_candidate_count` field), so PD's τs may resolve to slightly different values than what a fully-faithful re-calibration on the same grid would produce. The sanity check (O1) confirms convergence to within 0.002 absolute, which is below the per-cell rounding noise.
3. **Lockbox stress suites** (`teams_real_lighting_extreme_lockbox`, `teams_real_poor_quality_lockbox`) ARE present in PD's videos_reports but the contract policy does not gate on them. They are not shown in the head-to-head table (kept lockbox to the two suites P1 contract uses).
4. **F4 cross-substrate validation (HDTF) is NOT in this scorecard's scope** for either packet — would require running each ckpt against the HDTF suite manifest separately. PA's F4 lift not generalizing to HDTF (memory `project_pa_does_not_generalize_to_hdtf_2026-05-05.md`) is the cautionary prior; F4 is undetermined here.
5. **F2/F3 (5-axis correlation audit)** — not in this comparison's scope; shortcut-weakening verdict for PD is undetermined here. The Phase 1 baseline lives at `analysis/deeplive_viso_corr_eval_2026-05-06/abs_pearson_summary.csv`; Phase 2 (these PD ckpts vs that baseline) has not been run.

## Files

- `analysis/pd_vs_p1_2026-05-07/PD_VS_P1_FACTS_2026-05-07.md` (this file)
- `analysis/pd_vs_p1_2026-05-07/build_head_to_head.py` (the comparison-builder)
- `analysis/pd_vs_p1_2026-05-07/head_to_head_at_calibrated_tau.csv` (long form, 108 rows)
- `analysis/pd_vs_p1_2026-05-07/head_to_head_pivot.csv` (wide form, 12 ckpts × 9 suites)
- `analysis/pd_vs_p1_2026-05-07/raw/scorecards/scorecard.{csv,wide.csv,json}` (PD's resume-run rollup at τ=0.5; not used in this comparison but mirrored locally)
- `analysis/pd_vs_p1_2026-05-07/raw/pd_videos_reports/*.csv` (157 PD per-video reports from GCS)
- Source: `gs://training-job-outputs/test_results/pd_corr_penalty/pd-corr-penalty-scorecard-2026-05-06/reports/`
- Source: `gs://training-job-outputs/test_results/pd_corr_penalty/pd-corr-penalty-scorecard-2026-05-06-resume/scorecards/`
- Source: `analysis/p1_pe_eval_2026-05-07/scorecard/selected_threshold_scorecard.csv`
- Local prior PD cache: `analysis/pd_scorecard_artifacts_2026-05-06/` (τ=0.5 only; this comparison does not use it).
