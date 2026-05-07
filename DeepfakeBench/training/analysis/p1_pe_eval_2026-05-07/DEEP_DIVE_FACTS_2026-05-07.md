# P1 (PE_PAIR_RANK_DRO) — Deep-dive Results, 2026-05-07

**Status**: factual-only consolidating record. No interpretation. No verdict. For independent re-reading.

This doc consolidates everything that has been computed for the P1 evaluation as of 2026-05-07. It supersedes the earlier `ANALYSIS_DEPRECATED_2026-05-07.md` and the `RESULTS_F1_F5_FACTS_2026-05-07.md` synthesis on points where new data has landed (specifically: F3 on `color_b_dev` + `min_dim` re-pass; F2(a) feasibility; ROC degeneracy mechanism; roy_d transition matrix). Earlier docs are kept on disk for forensic provenance.

Where there is an interpretation gap or a methodological caveat, it is stated directly. Where a number could be misread, the reading frame is given mechanically.

**Companion docs (raw artifact pointers)**:
- `RESULTS_FACTS_2026-05-07.md` — Phase A scorecard tables + Phase C diagnostic.
- `RESULTS_F1_F5_FACTS_2026-05-07.md` — F1/F4/F5 verdicts (F5 was buggy at first; corrected).
- `ANALYSIS_DEPRECATED_2026-05-07.md` — earlier structured analysis (some claims now superseded).
- `f2_pair_rank/F2_PAIR_RANK_FACTS_2026-05-07.md` — F2(a) compute + caveats.
- `f3_color_b_dev/F3_COLOR_B_DEV_FACTS_2026-05-07.md` — F3 color_b_dev axis.
- `roc_degeneracy/ROC_DEGENERACY_FACTS_2026-05-07.md` — threshold-grid forensic.
- `roy_d_regression/ROY_D_REGRESSION_FACTS_2026-05-07.md` — roy_d per-frame transitions.

---

## 1. Per-ckpt verdict matrix — F1-F5

Cells show pass/fail/partial against the close criterion as written in `experiments/phase2_round13/R13_P1_BUNDLE_FT_FROM_P8A.yaml` header. No softening.

| ckpt | F1 lockbox≥90% | F2(a) pair-rank lift | F2(b) chronic/dor lift | F3 untargeted-axis | F4 HDTF≤5% | F5 pc_generator drop≥0.10 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| P8A_REFERENCE_STEP5000 | FAIL (38.7%) | n/a | n/a | (baseline) | PASS | n/a |
| E2B_TOP_N_STEP3200 | FAIL (62.9%) | n/a | n/a | FAIL face_area | PASS | n/a |
| P1_BUNDLE_PERIODIC_STEP500 | FAIL (82.6%) | not testable | partial | PARTIAL (face_area only) | PASS | **PASS Δ +0.629** |
| P1_BUNDLE_TOP_N_STEP3750 | FAIL (15.4%) | not testable | partial | PARTIAL (face_area only) | PASS | **PASS Δ +0.623** |
| P1_BUNDLE_TOP_N_STEP4000 | FAIL (26.1%) | not testable | partial | PARTIAL (face_area only) | PASS | **PASS Δ +0.598** |
| P1_PAIRRANK_PERIODIC_STEP500 | FAIL (70.8%) | not testable | partial | PARTIAL (face_area only) | PASS | n/a |
| P1_PAIRRANK_TOP_N_STEP6000 | FAIL (32.4%) | not testable | partial | PARTIAL (face_area only) | PASS | n/a |
| P1_PAIRRANK_TOP_N_STEP6750 | FAIL (48.6%) | not testable | partial | PARTIAL (face_area only) | PASS | n/a |

Cell legend:
- **F1**: contract-selected-τ lockbox fake recall vs 0.90 bar.
- **F2(a)**: "not testable" = the criterion as written cannot be evaluated against Phase A's eval substrate; see §3.
- **F2(b)**: "partial" = chronic-6 covered (see §6, F5 cluster passes), dor-drift covered through `teams_real_dor_dev` (see §2 worst-group recall row).
- **F3**: PARTIAL = passes on `sharpness_laplacian`, `min_dim`, `color_b_dev`; fails on `face_area_fraction`. See §4.
- **F4**: max FPR across 4 HDTF real suites at calibrated τ vs 0.05 bar.
- **F5** (BUNDLE-only by yaml): pc_generator chronic FPR drop vs P8A.

**Aggregate read of the YAML 4-of-4 BUNDLE gate** (F1+F2+F3+F4+F5, where F5 is BUNDLE-specific):
- Strict pass count for any BUNDLE ckpt = **2 of 5** (F4 + F5). F1 fails. F3 fails on 1 of 4 axes (counts as fail per yaml wording). F2(a) is "not testable", so does not contribute to the pass count and does not count as fail either.

---

## 2. F1 — lockbox fake recall ≥ 90% at FPR ≤ 10%

Source: `scorecard/selected_threshold_scorecard.csv`. Calibrated τ per ckpt.

| ckpt | τ_selected | lockbox_fake_recall (n=253) | lockbox_real_fpr (n=1361) | F1 |
|---|---:|---:|---:|:---:|
| P8A_REFERENCE_STEP5000 | 0.9156 | 0.387 | 0.018 | FAIL |
| E2B_TOP_N_STEP3200 | 0.7108 | 0.629 | 0.024 | FAIL |
| P1_BUNDLE_PERIODIC_STEP500 | 0.9919 | **0.826** | 0.033 | FAIL |
| P1_BUNDLE_TOP_N_STEP3750 | 0.9994 | 0.154 | 0.013 | FAIL |
| P1_BUNDLE_TOP_N_STEP4000 | 0.9989 | 0.261 | 0.021 | FAIL |
| P1_PAIRRANK_PERIODIC_STEP500 | 0.7677 | **0.708** | 0.018 | FAIL |
| P1_PAIRRANK_TOP_N_STEP6000 | 0.9878 | 0.324 | 0.020 | FAIL |
| P1_PAIRRANK_TOP_N_STEP6750 | 0.9900 | 0.486 | 0.026 | FAIL |

Closest to F1: **BUNDLE_PERIODIC_STEP500 at 0.826** (8.4 pp short of bar).

`teams_real_dor_dev` FPR (relevant to F2(b) dor-drift sub-criterion; n=50):

| ckpt | dor_dev FPR | n_FP |
|---|---:|---:|
| P8A | 0.080 | 4/50 |
| E2B | 0.120 | 6/50 |
| BUNDLE_step500 | **0.040** | 2/50 |
| BUNDLE_step3750 | 0.060 | 3/50 |
| BUNDLE_step4000 | 0.080 | 4/50 |
| PAIRRANK_step500 | 0.160 | 8/50 |
| PAIRRANK_step6000 | 0.080 | 4/50 |
| PAIRRANK_step6750 | **0.040** | 2/50 |

---

## 3. F2(a) — pair-rank lift on previously-missed fakes

Source: `f2_pair_rank/F2_PAIR_RANK_FACTS_2026-05-07.md` (full report on disk).

**Headline number**: 0 of 6 P1 ckpts pass the "≥30% relative lift on ≥2 of 6 paired lanes" bar.

**The number is mechanically forced by 5 stacked structural caveats** (each documented in the F2 doc and reproduced here in compressed form for visibility):

1. **Of 6 yaml-named training lanes, only 1 (`deeplive_teams`) has any Phase A proxy.** The other 5 (`df40`, `deeplive`, `visomaster_v1_base`, `visomaster_enhanced`, `visomaster_teams_enhanced`) score against substrates absent from Phase A reports (`gs://local/...`, `gs://visomaster-enhanced-face-cropped-v2/...`).

2. **Within the `deeplive_teams` lane, the F2 audit subdivides 8 canonical_subjects.** After applying the "previously-missed fake" filter (P8A `frame_prob < 0.5`), 5 of those 8 sub-lanes have `n_pairs = 0` — P8A's score is ≥0.5 on every fake in those subjects, so there are no missed fakes to lift on.

3. **The 3 sub-lanes with non-zero n_pairs** are `dor_shkedi__s16` (n=31), `test_cam__s76` (n=27), `xiang_xiang2_feng` (n=2228). The dominant lane has 97.4% of all pairs.

4. **P8A baseline `frac_fake_gt_real` is 1.000 / 1.000 / 0.971** on the 3 active sub-lanes. At 1.000 baseline, lift is structurally bounded ≤ 0. At 0.971 baseline, the maximum possible relative lift is +2.99% (= (1.000 − 0.971)/0.971 × 100). The 30% bar is arithmetically unreachable.

5. **Pairing convention used: cross-product within `canonical_subject`**, NOT `(sample_id, frame_idx)` tight pairs that the training-time pair_rank_loss actually fires on. Per `pair_gap_audit_2026-05-06/outputs/FINDINGS.md`, the eval-substrate teams_real and teams_fake frames come from different sessions; there is no tight (sample_id, frame_idx) link in the eval data. Cross-product is a coarse proxy.

**Maximum observed lift across the (6 P1 ckpts × 3 active sub-lanes)** grid = +1.99% (PAIRRANK_step6000 on xiang_xiang2_feng). Minimum = −3.23% (3 ckpts on dor_shkedi__s16, where 1 of 31 pair-rank flips lost from baseline).

**Implication for verdict reading**: the F2(a) "not testable" cell in the per-ckpt verdict table is not a model-quality statement. It is a statement that the close criterion as written cannot be evaluated against Phase A's substrate. To get a true F2(a) verdict, a future eval would need to score the model on each of the 6 paired training-lane substrates directly.

---

## 4. F3 — untargeted-axis amplification audit (4 axes)

Source: `abs_pearson_summary.csv`, `min_dim_correlations.csv`, `f3_color_b_dev/abs_pearson_summary.csv`. Per-frame Pearson r between `frame_prob` and the axis value, aggregated as mean |r| across same-kind suites (real-side: `teams_real_all_dev`; fake-side: `visomaster_enhanced_macro_dev`, `deeplive_enhanced_dev`, `teams_fake_all_dev`). N per real-side ≈ 4006 (after lockbox-tagging join coverage); n per fake-side ≈ 545-3039 depending on suite.

### 4.1 Real-side mean |r| — Δ percent vs P8A

P8A baseline `mean_abs_r` per axis on real-side (this is the column the +50% bar is computed against):

| axis | P8A mean_abs_r |
|---|---:|
| sharpness_laplacian | 0.188 |
| face_area_fraction | 0.117 |
| min_dim | **0.491** |
| color_b_dev | 0.091 |

Δ percent vs P8A on real-side per (ckpt × axis):

| ckpt | sharpness_laplacian | face_area_fraction | min_dim | color_b_dev |
|---|---:|---:|---:|---:|
| E2B | +96.4 AMP | +171.0 AMP | −69.1 | −73.1 |
| P1_BUNDLE_step500 | −38.5 | **+266.1 AMP** | −26.5 | −84.8 |
| P1_BUNDLE_step3750 | +1.0 | **+166.6 AMP** | −61.3 | −75.4 |
| P1_BUNDLE_step4000 | +4.5 | **+138.4 AMP** | −59.7 | −83.4 |
| P1_PAIRRANK_step500 | −33.3 | **+165.4 AMP** | −28.4 | −82.1 |
| P1_PAIRRANK_step6000 | +20.8 | **+105.8 AMP** | −73.0 | −95.3 |
| P1_PAIRRANK_step6750 | +33.7 | **+170.1 AMP** | −70.3 | −99.8 |

Cells marked "AMP" exceed +50% (the F3 bar for failure). Negative values indicate decoupling (model relies LESS on the axis than P8A).

**Mechanical F3 status per axis × ckpt:**
- `sharpness_laplacian`: PASSES for all 6 P1 ckpts (no amplification > +50%).
- `face_area_fraction`: FAILS for all 6 P1 ckpts (every ckpt amplifies > +50%; range +106% to +266%).
- `min_dim`: PASSES for all 6 P1 ckpts (decouple by 26-73%).
- `color_b_dev`: PASSES for all 6 P1 ckpts (decouple by 75-100%).

**Per the yaml's strict "no untargeted axis amplifies +50%" wording**: F3 fails for all 6 P1 ckpts (and for E2B), driven entirely by `face_area_fraction`.

### 4.2 Fake-side mean |r| — Δ percent vs P8A

P8A baseline `mean_abs_r` per axis on fake-side:

| axis | P8A mean_abs_r |
|---|---:|
| sharpness_laplacian | 0.188 |
| face_area_fraction | 0.412 |
| min_dim | 0.240 |
| color_b_dev | 0.197 |

Δ percent vs P8A on fake-side per (ckpt × axis):

| ckpt | sharpness_laplacian | face_area_fraction | min_dim | color_b_dev |
|---|---:|---:|---:|---:|
| E2B | −13.9 | −21.9 | +18.8 | −10.5 |
| P1_BUNDLE_step500 | −62.2 | −62.9 | −54.3 | −26.2 |
| P1_BUNDLE_step3750 | −37.6 | −64.9 | −42.0 | −38.3 |
| P1_BUNDLE_step4000 | −43.8 | −67.9 | −49.4 | −41.2 |
| P1_PAIRRANK_step500 | −14.0 | −45.7 | −54.3 | −16.6 |
| P1_PAIRRANK_step6000 | −26.1 | −40.4 | −26.4 | −34.5 |
| P1_PAIRRANK_step6750 | −31.4 | −61.5 | −19.3 | −32.8 |

On the fake side, every P1 ckpt × every axis decouples (negative Δ). No amplification anywhere.

### 4.3 Magnitude perspective on F3 real-side

P8A's strongest IQ shortcut on reals is `min_dim` at |r| = 0.491. P1 reduces this to 0.13-0.36. Simultaneously P1 raises `face_area_fraction` from 0.117 to 0.24-0.43. Absolute magnitude comparison:

| axis | P8A | best P1 (BUNDLE_step3750) | worst P1 (BUNDLE_step500) |
|---|---:|---:|---:|
| min_dim |r| | 0.491 | 0.190 | 0.361 |
| face_area_fraction |r| | 0.117 | 0.312 | 0.428 |
| color_b_dev |r| | 0.091 | 0.022 | 0.014 |
| sharpness |r| | 0.188 | 0.190 | 0.116 |

The `face_area_fraction` real-side coupling under any P1 ckpt is below P8A's `min_dim` coupling. Whether that's a net improvement or net regression depends on the comparison axis.

---

## 5. F4 — HDTF cross-substrate FPR ≤ 5% at calibrated τ

Source: `hdtf/scorecard_calibrated_tau.csv`. Phase C `promotion_contract/` failed; per-frame reports were complete; we applied Phase A per-ckpt τ to those reports locally.

| ckpt | τ | proper_real_teams_dev | proper_real_teams_lockbox | proper_real_clean_dev | proper_real_clean_lockbox | max | F4 |
|---|---:|---:|---:|---:|---:|---:|:---:|
| P8A | 0.9156 | 0.0079 | 0.0088 | 0.0026 | 0.0000 | **0.0088** | PASS |
| E2B | 0.7108 | 0.0015 | 0.0013 | 0.0016 | 0.0013 | 0.0016 | PASS |
| BUNDLE_step500 | 0.9919 | 0.0008 | 0.0026 | 0.0067 | 0.0141 | **0.0141** | PASS |
| BUNDLE_step3750 | 0.9994 | 0.0000 | 0.0000 | 0.0010 | 0.0000 | 0.0010 | PASS |
| BUNDLE_step4000 | 0.9989 | 0.0000 | 0.0000 | 0.0010 | 0.0013 | 0.0013 | PASS |
| PAIRRANK_step500 | 0.7677 | 0.0021 | 0.0020 | 0.0069 | 0.0065 | 0.0069 | PASS |
| PAIRRANK_step6000 | 0.9878 | 0.0028 | 0.0003 | 0.0002 | 0.0020 | 0.0028 | PASS |
| PAIRRANK_step6750 | 0.9900 | 0.0042 | 0.0007 | 0.0014 | 0.0052 | 0.0052 | PASS |

All 8 ckpts under 5%. The 25% FPR observed at τ=0.5 on BUNDLE_step500 was an uncalibrated-τ artifact; at deployment τ it's 1.41%.

HDTF fake recall at calibrated τ (diagnostic — not a gate):
- P8A: 83-95% across 8 fake suites (cross-substrate ceiling).
- E2B: bimodal — 84-93% on `*_clean_*`, 6-50% on `*_teams_*`.
- BUNDLE_step3750/4000: also bimodal — 95-97% clean / 17-40% teams.
- PAIRRANK_step500: more balanced — 64-98% range.

---

## 6. F5 — chronic-FP pc_generator cluster drop ≥ 0.10

Source: `phase_d/chronic6_aggregate_fpr_FIXED.csv` and `phase_d/per_identity_fpr_FIXED.csv` (after fixing a regex bug in the original `run_chronic_filter.py` that stripped session tokens before matching).

### 6.1 PC_Generator cluster (s22 + s45) FPR @ calibrated τ

| ckpt | n_pc | pc_fpr | Δ vs P8A | F5 (BUNDLE-only) |
|---|---:|---:|---:|:---:|
| P8A | 318 | **0.629** | (baseline) | n/a |
| E2B | 318 | 0.107 | +0.522 | n/a |
| BUNDLE_step500 | 318 | **0.000** | **+0.629** | PASS |
| BUNDLE_step3750 | 318 | 0.006 | +0.623 | PASS |
| BUNDLE_step4000 | 318 | 0.031 | +0.598 | PASS |
| PAIRRANK_step500 | 318 | 0.066 | +0.563 | n/a |
| PAIRRANK_step6000 | 318 | 0.142 | +0.487 | n/a |
| PAIRRANK_step6750 | 318 | 0.135 | +0.494 | n/a |

P8A's `pc_fpr = 0.629` (200/318 PC_Generator chronic frames false-positive at calibrated τ). All 3 BUNDLE ckpts drop this to ≤ 0.031.

### 6.2 Per-identity chronic-6 breakdown @ calibrated τ

| identity | n | P8A | E2B | BUNDLE_500 | BUNDLE_3750 | BUNDLE_4000 | PAIRRANK_500 | PAIRRANK_6000 | PAIRRANK_6750 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PC_Generator__s22 | 227 | 0.788 | 0.031 | 0.000 | 0.009 | 0.040 | 0.092 | 0.163 | 0.123 |
| PC_Generator__s45 | 91 | 0.231 | 0.297 | 0.000 | 0.000 | 0.011 | 0.000 | 0.088 | 0.165 |
| Q__s6 | 54 | 0.889 | 0.278 | 0.167 | 0.167 | 0.167 | 0.185 | 0.167 | 0.185 |
| bla_bla_chow | 491 | 0.063 | 0.295 | 0.163 | 0.100 | 0.100 | 0.126 | 0.112 | 0.112 |
| bla_bla_chow__s2 | 180 | 0.111 | 0.283 | 0.317 | 0.072 | 0.078 | 0.189 | 0.133 | 0.106 |
| roy_d | 130 | 0.292 | 0.154 | **0.931** | **0.869** | **0.854** | **0.785** | **0.815** | **0.777** |

### 6.3 Aggregate chronic-6 FPR @ calibrated τ

| ckpt | n_chronic | chronic_fpr | Δ vs P8A |
|---|---:|---:|---:|
| P8A | 993 | 0.319 | (baseline) |
| E2B | 993 | 0.215 | +0.103 |
| BUNDLE_step500 | 993 | 0.212 | +0.108 |
| BUNDLE_step3750 | 993 | 0.174 | +0.145 |
| BUNDLE_step4000 | 993 | 0.180 | +0.139 |
| PAIRRANK_step500 | 993 | 0.196 | +0.123 |
| PAIRRANK_step6000 | 993 | 0.217 | +0.103 |
| PAIRRANK_step6750 | 993 | 0.211 | +0.108 |

Every P1 ckpt reduces aggregate chronic-6 FPR by ≥ 0.103 absolute.

---

## 7. roy_d regression — per-frame transition mechanism

Source: `roy_d_regression/ROY_D_REGRESSION_FACTS_2026-05-07.md` and `roy_d_per_frame_scores.csv`.

`teams_real_all_dev` contains 130 roy_d frames (130 unique video_ids; each is 1 frame).

### 7.1 Score distribution per ckpt

| ckpt | τ_selected | p25 | p50 | p75 | p95 | max | mean | frac ≥ τ_sel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P8A | 0.9156 | 0.042 | 0.438 | 0.940 | 0.991 | 0.994 | 0.477 | **0.292** |
| E2B | 0.7108 | 0.053 | 0.286 | 0.619 | 0.917 | 0.962 | 0.353 | 0.154 |
| BUNDLE_step500 | 0.9919 | 0.992 | 0.993 | 0.993 | 0.993 | 0.993 | 0.992 | **0.931** |
| BUNDLE_step3750 | 0.9994 | 0.9998 | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.998 | 0.869 |
| BUNDLE_step4000 | 0.9989 | 0.9997 | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.994 | 0.854 |
| PAIRRANK_step500 | 0.7677 | 0.819 | 0.982 | 0.988 | 0.989 | 0.989 | 0.860 | 0.785 |
| PAIRRANK_step6000 | 0.9878 | 0.997 | 0.9998 | 0.9999 | 0.9999 | 0.9999 | 0.964 | 0.815 |
| PAIRRANK_step6750 | 0.9900 | 0.993 | 0.9998 | 0.9999 | 0.9999 | 0.9999 | 0.962 | 0.777 |

P8A score distribution on roy_d is bimodal (range 0.04-0.99). Every P1 ckpt collapses the distribution into the upper tail near 0.99.

### 7.2 P8A → BUNDLE transition matrix

| transition | BUNDLE_step500 | BUNDLE_step3750 | BUNDLE_step4000 |
|---|---:|---:|---:|
| P8A correct → BUNDLE correct | 6 | 17 | 19 |
| P8A correct → BUNDLE FP | **86** | **75** | **73** |
| P8A FP → BUNDLE correct | 3 | 0 | 0 |
| P8A FP → BUNDLE FP | 35 | 38 | 38 |
| **Total** | 130 | 130 | 130 |

For BUNDLE_step500: 86 of 92 P8A-correct roy_d frames flipped to FP under BUNDLE (93%). Per-frame score Δ stats (BUNDLE_step500 − P8A on the 130 roy_d frames): p25 +0.05, p50 +0.55, p75 +0.95, max +0.99, min −0.002, mean +0.515.

---

## 8. ROC degeneracy — why BUNDLE_step3750 / step4000 land at τ ≈ 0.999x

Source: `roc_degeneracy/ROC_DEGENERACY_FACTS_2026-05-07.md`. Threshold-grid analysis on `scorecard/threshold_grid.csv` (5300+ τ samples per ckpt).

| ckpt | n_grid | n_budget_OK | n_tier0 (recall ≥ 0.30) | τ_min in budget | max recall in budget |
|---|---:|---:|---:|---:|---:|
| P8A | 5361 | 1257 | 2 | 0.9148 | 0.300 |
| E2B | 5599 | 1800 | **561** | 0.7108 | 0.509 |
| BUNDLE_step500 | 5552 | 720 | **0** | 0.9919 | 0.075 |
| BUNDLE_step3750 | 5339 | 923 | **0** | 0.9994 | 0.192 |
| BUNDLE_step4000 | 5292 | 1032 | **0** | 0.9989 | 0.285 |
| PAIRRANK_step500 | 5592 | 1408 | 117 | 0.7675 | 0.354 |
| PAIRRANK_step6000 | 5390 | 1281 | 94 | 0.9876 | 0.339 |
| PAIRRANK_step6750 | 5447 | 1273 | 98 | 0.9900 | 0.343 |

For each of the 3 BUNDLE ckpts, **zero grid points have macro_recall ≥ 0.30 anywhere** in the FPR ≤ 0.07 region. The contract sort key (per `arena/score_teams_promotion_contract.py:485-499`) prefers tier-0 when available; for these 3 ckpts no tier-0 candidates exist, so within-tier-1 selection lands at the highest τ in budget (τ_max_budget = 1.0000 for all 8 ckpts).

The recall-floor flag (set at 0.30 by the launcher) **worked correctly** — it correctly demoted these ckpts in the cross-checkpoint rank. The within-tier-1 τ position at 0.999x is not a τ-tail collapse bug; it is the highest macro_recall available to those ckpts within the FPR budget.

P8A scrapes into tier-0 with 2 grid points; max recall in budget = 0.300 (just above floor). BUNDLE_step4000 misses tier-0 by 0.015 absolute (max recall in budget = 0.285).

---

## 9. W&B logging gap — pair_rank_loss + GroupDRO scalar magnitude not logged for BUNDLE

Already documented in `RESULTS_FACTS_2026-05-07.md` §4 and `PHASE_F_SYNTHESIS_TEMPLATE_FACTS.md` §3 caveat. Reprised here for self-containment:

`trainer/trainer.py:1718-1727` — when `use_group_dro=True`:
```python
per_sample_losses_dict = loss_fn_owner.get_losses(...)
per_sample_loss = per_sample_losses_dict['overall']
losses = self.calculate_group_dro_loss(data_dict, per_sample_loss)
```

The 11+ diagnostic scalars from `effort_detector.py:1345-1366` (including `pair_rank_loss` at line 1364, `cls_loss`, `corr_penalty_loss`, `feat_norm_loss`, `quality_domain_loss`, `keepsv_loss`, `reg_loss`) are silently discarded after `[‌'overall']` extraction. The W&B log loop at `trainer.py:1847-1857` only sees the GroupDRO mixin's 4-key return dict.

**Verifiable from W&B history:**
- PAIRRANK_ONLY (`s2mp5fxm`): logs `train/loss/pair_rank_loss` directly. Median 0.113 / max 0.919 / final-step 0.025. Fired throughout, no collapse.
- BUNDLE (`tznuar61`): logs only `train/loss/group_dro_in_warmup` (binary, transitions 1→0 at step 50) and `train/diagnostic/{group_losses_ema,group_weights}` (histograms). No scalar magnitude for either pair_rank_loss or DRO loss component.

**Recoverability**: pair_rank_loss cannot be reconstructed from the BUNDLE checkpoint alone — it depends on per-batch `pair_id` composition, which is not checkpointed. To recover: replay a few training batches against the BUNDLE checkpoint with the same data loader seed, OR fix `trainer.py:1727` and rerun.

**One-line fix candidate**:
```python
losses.update({k: v for k, v in per_sample_losses_dict.items() if k != 'overall'})
```

---

## 10. Phase E weight-delta — diagnostic, NOT a gate

Source: `outputs/weight_delta_verdict.csv` (8 P1 ckpts after the overlap extension).

| ckpt | qkv_mean_fnorm | out_proj_mean_fnorm | mlp_mean_fnorm | qkv/out_proj | qkv/mlp |
|---|---:|---:|---:|---:|---:|
| p1_bundle_step500 | 0.0080 | 0.1010 | 0.2449 | 0.0796 | 0.0328 |
| p1_bundle_step1000 | 0.0125 | 0.1741 | 0.3984 | 0.0715 | 0.0313 |
| p1_bundle_step3750 | 0.0192 | 0.2529 | 0.5837 | 0.0759 | 0.0329 |
| p1_bundle_step4000 | 0.0192 | 0.2528 | 0.5847 | 0.0759 | 0.0328 |
| p1_pairrank_step500 | 0.0078 | 0.1021 | 0.2488 | 0.0767 | 0.0315 |
| p1_pairrank_step1000 | 0.0123 | 0.1711 | 0.4082 | 0.0716 | 0.0300 |
| p1_pairrank_step6000 | 0.0190 | 0.2574 | 0.5889 | 0.0739 | 0.0323 |
| p1_pairrank_step6750 | 0.0203 | 0.2681 | 0.6163 | 0.0758 | 0.0330 |

The qkv/out_proj ratio sits in [0.0715, 0.0796] across all 8 P1 ckpts; qkv/mlp in [0.0300, 0.0330]. The in_proj-SVD lever is active and consistent across both arms × all training steps.

**Caveat**: there is no `apply_svd_to_in_proj=False` baseline run in this packet. Ratios are relative-only. Cannot answer "is the lever load-bearing for outcomes?" without that baseline.

---

## 11. Dor probe — IQ-axis trajectory across all 6 P1 ckpts (180 frames)

Source: `dor_invariance_2026-05-07/axis_decoupling_trajectory.csv`. Reference: P8A real-side `sharpness raw_r = −0.644` on the dor probe substrate.

| arm | step | sharpness raw_r | min_dim raw_r | color_b_dev raw_r |
|---|---:|---:|---:|---:|
| BUNDLE | 500 | −0.272 | +0.338 | −0.016 |
| BUNDLE | 3750 | −0.342 | +0.192 | −0.045 |
| BUNDLE | 4000 | −0.317 | +0.181 | −0.044 |
| PAIRRANK | 500 | −0.517 | +0.298 | +0.218 |
| PAIRRANK | 6000 | −0.600 | +0.354 | +0.243 |
| PAIRRANK | 6750 | −0.631 | +0.407 | +0.196 |

Per-variant FPR @ τ=0.5 on the 180-frame dor probe:

| variant | n | P8A | E2B | BUNDLE_500 | BUNDLE_3750 | BUNDLE_4000 | PAIRRANK_500 | PAIRRANK_6000 | PAIRRANK_6750 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dor_laptop_whiteish | 30 | 0.000 | 0.000 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dor_laptop_yellowish | 30 | 0.000 | 0.000 | 0.933 | 0.400 | 0.300 | 0.067 | 0.067 | 0.200 |
| dor_session_0411 | 30 | 0.033 | 0.033 | 0.033 | 0.033 | 0.033 | 0.033 | 0.033 | 0.033 |
| dor_session_0424 | 30 | 0.700 | 0.367 | 1.000 | 0.667 | 0.600 | 0.367 | 0.867 | 0.867 |
| dor_webcam_no_vbg | 30 | 0.800 | 0.367 | 1.000 | 0.433 | 0.333 | 0.567 | 0.700 | 0.767 |
| dor_webcam_with_vbg | 30 | 0.767 | 0.000 | 1.000 | 0.300 | 0.233 | 0.433 | 0.533 | 0.667 |
| **Combined** | 180 | 0.383 | 0.128 | 0.745 | 0.306 | 0.250 | 0.244 | 0.367 | 0.422 |

(τ=0.5 here, which is NOT the deployment τ from Phase A's contract. At calibrated τ from Phase A, BUNDLE_step500's `teams_real_dor_dev` FPR is 0.04 (2/50). The 180-frame dor probe and the 50-frame `teams_real_dor_dev` are different substrates with different frame counts.)

---

## 12. Phase A.5 partial — diagnostic substrate inference (4 valid suites)

Source: `diagnostic_substrates/per_suite_comparison_tau_0.5.csv`. 4 suites had valid frame paths; 5 had stale `gs://local/...` placeholders.

| suite | n | label | P8A | E2B | PA_3800 | BUNDLE_step4000 | PAIRRANK_step6750 |
|---|---:|---|---:|---:|---:|---:|---:|
| xinhe_may6_falseflag | 92 | real (FPR) | 0.0000 | 0.5761 | 0.1630 | 0.0978 | 0.1304 |
| live_fakes_teams_prod | 1675 | fake (recall) | 0.7534 | 0.8113 | 0.5039 | 0.7409 | 0.7887 |
| visomaster_v2_dor | 2073 | fake (recall) | 0.7381 | 0.5596 | 0.2523 | 0.6522 | 0.7771 |
| team_sanity_may5 | 210 | real (FPR) | 0.0095 | 0.0095 | 0.0048 | 0.0095 | 0.0143 |

Open loop: `grouped-manifest-v2-stale-paths` — the 5 invalid suites cannot be re-evaluated until the manifest is regenerated.

---

## 13. Phase C HDTF — diagnostic τ=0.5 vs calibrated τ deltas

Already covered in §5 with calibrated τ. The diagnostic τ=0.5 readout reveals the shape of the score distribution shift between P8A and P1; saved as `hdtf/scorecard.wide.csv`. Notable cross-substrate-vs-substrate comparisons:

P8A on `proper_visomaster_enhanced_teams_dev` (n=1182, HDTF substrate, τ=0.5): fake recall = 0.9357.
P8A on `visomaster_enhanced_macro_dev` (n=550, production substrate, τ=0.9156): fake recall = 0.1345.

Same model, structurally similar attack class (visomaster-enhanced + teams transport), 7× recall gap due to substrate.

---

## 14. Open follow-ups still NOT computed

1. **Phase C failure root cause** — the `replica workerpool0-0 exited with non-zero status of 1` message is generic. Vertex job logs not yet inspected.
2. **F2(a) on the 5 missing training lanes** — would require scoring the 6 P1 ckpts against the training-loader's substrates (`gs://local/...`, `gs://visomaster-enhanced-face-cropped-v2/...`, etc.). Not in Phase A.
3. **PD scorecard comparison** — `analysis/pd_scorecard_artifacts_2026-05-06/unified_scorecard_simple.csv` not yet joined.
4. **Per-axis attribution for the roy_d regression** — joining `roy_d_per_frame_scores.csv` to lockbox tags and running an axis-correlation analysis on roy_d-only frames could identify what feature pattern P1 picked up on. Not run.
5. **Ablation: what happens at lower τ for BUNDLE step500?** — at calibrated τ=0.9919, lockbox recall is 82.6%. At τ=0.95 it would be higher; at τ=0.99 it would be different. Phase A picked one τ; the relationship across τ is in `threshold_grid.csv` if needed.

---

## 15. Cross-references

**Verdict-bearing artifacts** (most authoritative numbers):
- `scorecard/promotion_winner.json` — Phase A contract verdict (PAIRRANK_PERIODIC_STEP500 rank-1).
- `scorecard/selected_threshold_scorecard.csv` — 8 ckpts × 9 suites at calibrated τ.
- `hdtf/scorecard_calibrated_tau.csv` — F4 at calibrated τ (Phase C reports + Phase A τ).
- `phase_d/chronic6_aggregate_fpr_FIXED.csv`, `per_identity_fpr_FIXED.csv` — F5 + per-identity chronic-6.
- `f2_pair_rank/per_lane_per_ckpt_lift.csv` — F2(a) data.
- `f3_color_b_dev/abs_pearson_summary.csv` — F3 color_b_dev.
- `min_dim_correlations.csv`, `abs_pearson_summary.csv`, `correlations.csv` — F3 sharpness, face_area, min_dim, luma.
- `dor_invariance_2026-05-07/per_variant_fpr.csv`, `axis_decoupling_trajectory.csv` — dor probe.
- `outputs/weight_delta_verdict.csv` — Phase E weight-delta.

**Memory entries that are now LOAD-BEARING for re-reading P1**:
- `project_promotion_contract.md` — the lexicographic contract policy.
- `project_contract_policy_bug.md` — the τ-tail collapse failure mode.
- `project_p8a_breakthrough.md` — P8A as anchor / FT base.
- `project_deployment_is_e2b_2026-05-06.md` — E2B is the production deployment.
- `project_signature_shortcut_finding.md` — RLP6_04 ceiling on dor anchor.
- `project_chronic_offenders_partition_per_ckpt_2026-05-04.md` — chronic-6 list provenance.
- `project_in_proj_svd_gradient_bug.md` — `apply_svd_to_in_proj` was zero-gradient pre-`2feea58`; P1 is the FIRST post-fix lever-class test.

**Memory entries that should be re-checked after this evaluation** (some claims may no longer hold without modification):
- The "P1 is the advisor's primary recommendation" framing in `NEXT_STEPS_PLAN_2026-05-06.md` — P1 ran; results in this doc.
- The 27% viso ceiling line — P1 PAIRRANK ckpts hit 22% on visomaster_enhanced_macro_dev at calibrated τ; not a clean breakthrough but worth checking trajectory.
- The "P8A is the dor-invariance winner" framing — P8A `pc_fpr=0.629` on the `teams_real_all_dev` chronic PC_Generator cluster; BUNDLE drops it to 0-3%. The "winner" claim depends on which substrate.

---

## 16. Self-correction log

This evaluation went through 2 known reversals from the agent's first read:

1. **F5 buggy → corrected** — `phase_d/run_chronic_filter.py` original implementation stripped `__s22/__s45/__s2/__s6` session tokens via regex before matching against the chronic-6 list. This dropped PC_Generator and Q chronic identities entirely. Corrected version uses prefix-on-raw-video_id matching (`f2_pair_rank/compute_f2.py` and the inline F5 fix). The original buggy CSVs are kept on disk at `phase_d/{chronic6_aggregate_fpr,per_identity_fpr,pc_generator_cluster_fpr}.csv` for forensic provenance — DO NOT USE for verdict; use the `_FIXED` versions.

2. **F3 "completely fail" → 1-of-4 fail** — initial F3 read from `run_audit.py` only measured 3 axes (sharpness, face_area, luma) and concluded "all P1 amplifies face_area_fraction". Extension to `min_dim` (data already in tags) showed P1 reduces min_dim coupling 26-73%; extension to `color_b_dev` (frame downloads + B-channel std) showed P1 reduces color_b_dev coupling 75-100%. Of the 4 axes named in the F3 close criterion, only 1 amplifies. The strict F3 verdict is still FAIL (close criterion = "no untargeted axis amplifies +50%"), but the per-axis picture is mixed.

These are documented for the next agent's awareness — when re-reading the data, the same kind of partial reading could happen again on a different axis or filter. The `_FIXED` suffix and the inline caveats are the disambiguating tags.
