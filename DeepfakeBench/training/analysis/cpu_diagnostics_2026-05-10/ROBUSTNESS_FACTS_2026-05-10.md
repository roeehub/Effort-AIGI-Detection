# Robustness diagnostics FACTS (2026-05-10)

> **Status: factual-only.** Forbidden words: succeeds, fails, wins, promotes, deployment-grade.
> Numbers + tables + cross-references only. Interpretation belongs in `ROBUSTNESS_OPINIONS_2026-05-10.md`.
>
> **Scope**: 4 candidate ckpts (P8A, E2B, T3_SLOT1_step1500, T3_SLOT1_step2500) compared on 4 CPU diagnostic axes.
> Driver scripts at `analysis/cpu_diagnostics_2026-05-10/scripts/`. Outputs at `outputs/`.

---

## 1. Question

For each of {P8A_REFERENCE_STEP5000, E2B_TOP_N_STEP3200, T3_SLOT1_PERIODIC_STEP1500, T3_SLOT1_PERIODIC_STEP2500}, characterize:

1. Per-axis-bin FPR + recall variance (axis-decoupling) — Job A
2. Catastrophic-tail count (count of reals scored very high) — Job B
3. Post-IQ-gate FPR (production-deployable robustness) — Job C
4. Cross-ckpt disagreement structure (where they differ on the same frames) — Job D

## 2. Method

All jobs share data sources:
- Per-frame score CSVs from contract scorecards (P8A/E2B at `analysis/cpu_followups_2026-05-04/raw_reports/`; T3 at `analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/`).
- IQ atlas at `analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet` (15,236 frames; 6 IQ axes per frame).
- Real cohort: `teams_real_all_dev` (n=4564 frames; 1898 with IQ atlas data, 39.2% join rate).
- Fake cohorts: `teams_fake_all_dev`, `visomaster_enhanced_macro_dev`, `deeplive_enhanced_dev`.

## 3. Job A — Multi-axis robustness matrix

For each ckpt, calibrated τ to give 5% overall FPR on `teams_real_all_dev`. Then binned reals (and pooled fakes) by IQ axis quartiles and measured per-bin FPR + recall.

### 3.1 Calibrated τ at 5% overall FPR

| ckpt | τ | actual FPR |
|---|---:|---:|
| P8A | 0.9756 | 4.97% |
| E2B | 0.7442 | 4.97% |
| T3_S1_step1500 | 0.7757 | 4.97% |
| T3_S1_step2500 | 0.9761 | 4.97% |

### 3.2 Worst FPR axis-bin per ckpt

| ckpt | worst bin | FPR there | recall there |
|---|---|---:|---:|
| P8A | min_dim Q1 (lowest res) | 10.4% | 31.2% |
| E2B | lap_var Q1 (blurriest) | 13.1% | 34.4% |
| T3_S1_step1500 | color_a_dev Q4 (warmest) | 14.3% | 66.7% |
| T3_S1_step2500 | color_a_dev Q4 (warmest) | 20.4% | 48.7% |

### 3.3 Per-axis-bin FPR matrix (FPR @ 5% overall calibrated τ)

P8A:
| axis | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| color_a_dev | 7.9% | 2.0% | 0.0% | 4.7% |
| color_b_dev | 8.4% | 2.0% | 3.4% | 0.7% |
| lap_var | 4.7% | 0.2% | 3.2% | 6.4% |
| luma_mean | 8.4% | 4.7% | 1.5% | 0.0% |
| min_dim | **10.4%** | 0.7% | 3.4% | 0.0% |
| saturation_mean | 5.7% | 3.0% | 2.5% | 3.4% |

T3_S1_step1500:
| axis | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| color_a_dev | 3.7% | 1.2% | 0.5% | **14.3%** |
| color_b_dev | 4.9% | 2.5% | 10.6% | 1.7% |
| lap_var | 13.3% | 0.2% | 1.0% | 5.2% |
| luma_mean | 6.4% | 12.1% | 1.0% | 0.2% |
| min_dim | 7.9% | 0.5% | 10.7% | 0.5% |
| saturation_mean | 3.2% | 1.7% | 2.7% | 12.1% |

T3_S1_step2500:
| axis | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| color_a_dev | 1.5% | 1.5% | 1.2% | **20.4%** |
| color_b_dev | 2.0% | 7.4% | 12.8% | 2.5% |
| lap_var | 18.5% | 1.5% | 1.2% | 3.4% |
| luma_mean | 4.2% | 18.3% | 2.2% | 0.0% |
| min_dim | 8.1% | 2.5% | 13.4% | 0.5% |
| saturation_mean | 1.2% | 1.7% | 7.1% | 14.5% |

### 3.4 Per-axis FPR spread (max-min)

| ckpt | mean spread | max bin FPR |
|---|---:|---:|
| P8A | 0.073 | 10.4% |
| E2B | 0.087 | 13.1% |
| T3_S1_step1500 | 0.114 | 14.3% |
| T3_S1_step2500 | 0.153 | 20.4% |

Source: `outputs/job_a_robustness_per_bin.csv`, `outputs/job_a_robustness_summary.csv`.

## 4. Job B — Catastrophic-tail comparison

Count of reals from `teams_real_all_dev` (n=4564) scoring above various thresholds.

### 4.1 Real-side catastrophic-tail count (% of cohort scoring above threshold)

| ckpt | reals > 0.5 | reals > 0.7 | reals > 0.9 | reals > 0.95 | reals > 0.99 |
|---|---:|---:|---:|---:|---:|
| **T3_S1_step1500** | **9.03%** | 6.79% | 5.30% | 4.69% | 3.79% |
| E2B | 10.08% | 6.81% | 4.52% | 4.05% | 3.13% |
| P8A | 12.20% | 8.92% | 6.53% | 5.46% | 4.12% |
| T3_S1_step2500 | **19.96%** | 13.52% | 8.30% | 6.09% | 3.66% |

### 4.2 Unique catastrophic-FP structure

Frames where one ckpt scored > 0.9 AND other 3 ckpts all scored < 0.5:

| ckpt | unique catastrophic-FP count |
|---|---:|
| T3_S1_step1500 | **0** |
| E2B | 2 |
| P8A | 11 |
| T3_S1_step2500 | 60 |

Shared catastrophic-FP reals (all 4 ckpts > 0.9): n = 17.

### 4.3 Fake-side catastrophic-miss count (% of fake cohort scoring below threshold)

| Suite | ckpt | fakes < 0.05 | fakes < 0.1 | fakes < 0.3 |
|---|---|---:|---:|---:|
| teams_fake_all_dev | P8A | 10.3% | 13.0% | 20.0% |
| teams_fake_all_dev | E2B | 12.4% | 13.9% | 17.0% |
| teams_fake_all_dev | T3_S1_step1500 | 6.3% | 8.1% | 14.1% |
| teams_fake_all_dev | T3_S1_step2500 | **1.7%** | 2.5% | 4.4% |
| visomaster_enhanced_macro_dev | P8A | 37.5% | 43.1% | 58.5% |
| visomaster_enhanced_macro_dev | E2B | 66.5% | 73.8% | 86.5% |
| visomaster_enhanced_macro_dev | T3_S1_step1500 | 34.4% | 42.0% | 58.5% |
| visomaster_enhanced_macro_dev | T3_S1_step2500 | **9.5%** | 13.5% | 22.2% |
| deeplive_enhanced_dev | P8A | 16.0% | 22.2% | 37.1% |
| deeplive_enhanced_dev | E2B | 0.0% | 0.0% | 0.7% |
| deeplive_enhanced_dev | T3_S1_step1500 | 0.0% | 0.6% | 10.5% |
| deeplive_enhanced_dev | T3_S1_step2500 | **0.0%** | 0.0% | 0.0% |

Source: `outputs/job_b_catastrophic_tail.csv`, `job_b_unique_FPs_*.csv`, `job_b_top50_overfired_reals_per_ckpt.csv`.

## 5. Job C — IQ-gate filterability

Question: at multiple proposed (lap_var, min_dim) gate thresholds, does the gate filter out the chronic-6 + warm-color (Roy_D-equivalent) populations?

### 5.1 Per-chronic-identity IQ profile (mean values, n=1898 IQ-joined real cohort)

| Identity | n | lap_var | min_dim | color_a_dev | sat_mean |
|---|---:|---:|---:|---:|---:|
| Roy_D | 63 | **89.6** | 275 | **19.16** | 124 |
| PC_Generator | 424 | 283 | 224 | 6.19 | 73 |
| bla_bla_chow | 243 | 286 | 284 | 12.89 | 106 |
| Md_noyn_Sharker | 240 | 591 | **169** | 14.32 | 97 |
| Test_Cam | 420 | 229 | 268 | 10.69 | 76 |
| Xiang_Xiang2_Feng | 198 | 541 | 306 | 4.62 | 36 |

### 5.2 Filterability at gate thresholds

Gate retention (frames passing gate / total in category):

| Gate | total reals | chronic-6 | color_a_dev Q4 |
|---|---:|---:|---:|
| none | 100% | 100% | 100% |
| very_lenient (lap_var≥30, min_dim≥100) | 91.1% | 94.7% | 96.8% |
| lenient (lap_var≥50, min_dim≥150) | 74.7% | 76.6% | 87.2% |
| medium (lap_var≥100, min_dim≥200) | 33.8% | 38.5% | 29.1% |
| strict (lap_var≥200, min_dim≥200) | 21.5% | 25.5% | 18.3% |

### 5.3 Per-chronic-identity retention at medium gate (lap_var≥100, min_dim≥200)

| Identity | kept / total | retention |
|---|---|---:|
| Roy_D | 4 / 63 | 6.3% |
| PC_Generator | 69 / 424 | 16.3% |
| bla_bla_chow | 118 / 243 | 48.6% |
| Md_noyn_Sharker | 11 / 240 | 4.6% |
| Test_Cam | 253 / 420 | 60.2% |
| Xiang_Xiang2_Feng | 157 / 198 | 79.3% |

Source: `outputs/job_c_iq_gate_filterability.csv`.

## 6. Job C2 — Post-gate FPR per ckpt

For each ckpt × gate threshold, compute FPR @ τ=0.5 on the gate-PASSING real cohort.

### 6.1 FPR @ τ=0.5 on gate-passing reals (production-honest)

| Gate | P8A | E2B | T3_S1_step1500 | T3_S1_step2500 |
|---|---:|---:|---:|---:|
| none | 10.70% | 13.01% | **8.43%** | 20.76% |
| very_lenient | 7.57% | 9.88% | **6.76%** | 17.80% |
| lenient | 6.14% | 8.04% | **6.21%** | 14.88% |
| medium | 4.21% | 10.76% | **2.65%** | 14.20% |
| strict | 4.41% | 16.18% | **4.17%** | 18.38% |

### 6.2 FPR reduction multiplier (pre-gate / post-gate)

| Gate | P8A | E2B | T3_S1_step1500 | T3_S1_step2500 |
|---|---:|---:|---:|---:|
| very_lenient | 1.41× | 1.32× | 1.25× | 1.17× |
| lenient | 1.74× | 1.62× | 1.36× | 1.40× |
| medium | 2.54× | 1.21× | 3.18× | 1.46× |
| strict | 2.42× | 0.80× | 2.02× | 1.13× |

### 6.3 FPR @ τ=0.5 on gate-passing AND warm-color (color_a_dev Q4) — Roy_D-type production population

| Gate | P8A | E2B | T3_S1_step1500 | T3_S1_step2500 |
|---|---:|---:|---:|---:|
| none + color_q4 | 18.90% | 24.40% | 22.01% | 35.17% |
| medium + color_q4 | 13.27% | 36.28% | **11.50%** | 33.63% |
| strict + color_q4 | 16.22% | **55.41%** | 17.57% | **50.00%** |

Source: `outputs/job_c2_post_gate_fpr.csv`.

## 7. Job D — Cross-ckpt disagreement structure

For each frame in `teams_real_all_dev` (n=4839), compute score range across the 4 ckpts. Identify "controversial" frames (range > 0.7).

### 7.1 Score range distribution

| percentile | range |
|---|---:|
| p50 | 0.029 |
| p75 | 0.335 |
| p90 | 0.698 |
| p95 | 0.830 |
| p99 | 0.956 |

Controversial frames (range > 0.7): 474 / 4839 (9.8%).

### 7.2 Outlier-high distribution on controversial frames

Which ckpt scores HIGHEST when ckpts disagree?

| ckpt | times outlier-high | % |
|---|---:|---:|
| T3_S1_step2500 | 251 | 53% |
| P8A | 172 | 36% |
| E2B | 51 | 11% |
| T3_S1_step1500 | **0** | **0%** |

### 7.3 Outlier-low distribution on controversial frames

Which ckpt scores LOWEST when ckpts disagree?

| ckpt | times outlier-low | % |
|---|---:|---:|
| E2B | 240 | 51% |
| P8A | 184 | 39% |
| T3_S1_step1500 | 50 | 11% |
| T3_S1_step2500 | **0** | **0%** |

### 7.4 Pairwise score correlation (Pearson)

|  | P8A | E2B | step1500 | step2500 |
|---|---:|---:|---:|---:|
| P8A | 1.000 | 0.524 | **0.825** | 0.719 |
| E2B | 0.524 | 1.000 | 0.622 | 0.683 |
| step1500 | 0.825 | 0.622 | 1.000 | **0.837** |
| step2500 | 0.719 | 0.683 | 0.837 | 1.000 |

### 7.5 IQ profile of controversial frames vs full cohort

| axis | full mean | controversial mean | Δ |
|---|---:|---:|---:|
| lap_var | 302.04 | 265.17 | -36.87 |
| min_dim | 252.59 | 235.60 | -16.99 |
| color_a_dev | 10.32 | 11.48 | +1.16 |
| saturation_mean | 80.75 | 88.45 | +7.70 |
| luma_mean | 159.83 | 148.72 | -11.11 |

### 7.6 Per-quartile controversy rate

| axis | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| lap_var | 15.4% | 4.9% | 10.3% | 8.0% |
| min_dim | 11.3% | 7.8% | 12.0% | 7.4% |
| color_a_dev | 9.7% | 5.9% | 8.9% | **14.1%** |
| saturation_mean | 8.0% | 9.5% | 7.0% | **14.1%** |
| luma_mean | 10.9% | **17.1%** | 6.1% | 4.4% |

Source: `outputs/job_d_per_frame_disagreement.csv`, `job_d_controversial_frames.csv`.

## 7.5 may6 retest — production-frame drift FACTS

T3_S1_step1500 + T3_S1_step2500 scored on the 92 may6 + 60 may5 frames at
`analysis/xinhe_cross_camera_audit_2026-05-06/raw/`. P8A, E2B, PA_3800
scores from prior 2026-05-06 audit; T3 scores added 2026-05-10. Inference:
local CPU, INTER_LINEAR resize to 224, BGR2RGB, CLIP normalization (matches
training/eval preprocessing).

### 7.5.1 Per-cohort score distribution

| ckpt | may5 mean | may6 mean | Δ (drift) | may5 > 0.5 | may6 > 0.5 | may6 > 0.7 | may6 > 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P8A | 0.021 | 0.021 | +0.000 | 0/60 (0%) | 0/92 (0%) | 0/92 (0%) | 0/92 (0%) |
| E2B | 0.053 | 0.511 | +0.459 | 1/60 (1.7%) | 53/92 (57.6%) | 34/92 (37.0%) | 16/92 (17.4%) |
| PA_3800 | 0.054 | 0.269 | +0.215 | 0/60 (0%) | 15/92 (16.3%) | 9/92 (9.8%) | 2/92 (2.2%) |
| T3_S1_STEP1500 | 0.012 | 0.109 | +0.096 | 0/60 (0%) | 6/92 (6.5%) | 2/92 (2.2%) | 0/92 (0%) |
| T3_S1_STEP2500 | 0.063 | 0.732 | +0.670 | 0/60 (0%) | 71/92 (77.2%) | 59/92 (64.1%) | 40/92 (43.5%) |

### 7.5.2 Reading

- may5 frames (n=60): all 5 ckpts score them as real with ≤1.7% > 0.5.
  Same identity, day before, no false-flag.
- may6 frames (n=92): 4 of 5 ckpts have non-zero false-flag rates. Order
  from least to most fragile on may6: P8A (0%) < step1500 (6.5%) < PA_3800
  (16.3%) < E2B (57.6%) < step2500 (77.2%).
- The day-to-day drift (may6 mean - may5 mean) is the empirical signature
  of production-frame fragility: P8A 0.000, step1500 +0.096, PA_3800 +0.215,
  E2B +0.459, step2500 +0.670.
- Memory `project_xinhe_may6_falseflag_2026-05-06` recorded scores 0.83-0.93
  on may6 frames at deploy; per `project_deployment_is_e2b_2026-05-06`
  (Pearson r=+1.000 deploy ↔ E2B local), the deployed model that produced
  those may6 false-flags is E2B. P8A scores 0/92 may6 over 0.5.

Source: `analysis/xinhe_cross_camera_audit_2026-05-06/outputs/scores_T3_S1_STEP{1500,2500}.csv`,
`scores_all_5_ckpts_may6_may5.csv`.

## 8. Caveats

- IQ atlas join coverage on `teams_real_all_dev`: 1898/4564 (39.2%). Sub-cohort statistics use n=1898; whole-cohort statistics (Job B catastrophic tail) use n=4564.
- Frame reports for T3 ckpts use F4 input CSVs (4 suites: real_all_dev, viso_enhanced_macro, deeplive_enhanced, teams_fake_all_dev). Stress + lockbox suites have step1500 reports only — not used in this analysis.
- All FPR / recall numbers at fixed τ=0.5 unless explicitly noted; "calibrated τ" rows use the 5% overall FPR on the ckpt's own real distribution.
- The IQ atlas covers a sub-population of the eval substrate; un-joined frames are excluded from per-axis analysis but included in Job B macro counts.
- Production frame distribution (camera, lighting, identity composition) is not the eval cohort's distribution. The "warm-color (color_a_dev Q4)" subset's behavior is the closest available proxy for "Roy_D-type production users" — see memory `project_dor_drift_named_axes_2026-05-06`.
- This is offline analysis; no in-the-wild production-frame retest performed (Xinhe-may6 retest deferred — T3 ckpts not yet scored against may6 frames).

## 9. Cross-references

- Driver scripts: `analysis/cpu_diagnostics_2026-05-10/scripts/job_{a,b,c,c2,d}_*.py`
- Companion (interpretation): `ROBUSTNESS_OPINIONS_2026-05-10.md`
- IQ atlas: `analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet`
- Prior packet retro: `docs/packet_retrospectives/packets/T3.md`
- Morning brief: `analysis/cpu_diagnostics_2026-05-09/MORNING_BRIEF_2026-05-10.md`
- Memory entries (relevant): `project_dor_drift_named_axes_2026-05-06`, `project_chronic_offenders_partition_per_ckpt_2026-05-04`, `project_xinhe_may6_falseflag_2026-05-06`, `project_canary_below_production_resolution_2026-05-08`.
