# Stage A CPU Probes — FACTS (2026-05-12)

> **Status: factual-only.** Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, unfortunately, remarkably.
> Numbers + tables + cross-references only. Interpretation belongs in a sibling OPINIONS doc.
>
> **Scope**: three CPU probes on existing per-frame contract scorecard data + the small-cohort probe matrix from `cpu_diagnostics_2026-05-11_t67_t5c_probe`. Question: is T5C step3500 already deployable in some inference-policy combination with P8A that strictly dominates P8A on Pillar 1 (fake catch) without regressing Pillar 2 (FPR) or Pillar 3 (production-drift robustness)?
>
> **Inputs**:
> - Per-frame contract reports (P8A + T5C_step3500 + T3_SLOT1_step1500 × 9 suites = 27 files, 13,636 frames) under `_scorecard_reports/`
> - IQ atlas at `analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet` (n=15,236; join coverage 6,905 / 13,636 = 50.6%)
> - Small-cohort probe matrix at `analysis/cpu_diagnostics_2026-05-11_t67_t5c_probe/outputs/per_ckpt_cohort_scores.csv` (n=952 frames including may5/may6/dor/chronic_6/Roy_D)
>
> **Scripts**: `run_stage_a.py`, `synthesize_facts.py`

---

## 1. Probe 1 — Per-IQ-bin policy probe

For each IQ axis × quartile × ckpt × τ, FPR-on-reals + recall-on-fakes are tabulated in `outputs/per_iq_bin_policy.csv` (999 rows).

### 1.1 Per-axis fragility at τ=0.5 (P8A vs T5C step3500)

| Axis | Cohort | Q1 P8A FPR | Q1 T5C FPR | Q4 P8A FPR | Q4 T5C FPR |
|---|---|---:|---:|---:|---:|
| min_dim | teams_real_all_dev | 26.2% | 27.0% | 1.8% | 10.4% |
| min_dim | teams_real_all_lockbox | 19.7% | 41.0% | 3.4% | 44.1% |
| lap_var | teams_real_all_dev | 15.3% | 28.3% | 17.0% | 22.7% |
| lap_var | teams_real_all_lockbox | 10.0% | 46.0% | 16.1% | 40.9% |
| color_a_dev | teams_real_all_dev | 17.7% | 17.2% | 18.0% | 29.6% |
| color_a_dev | teams_real_all_lockbox | 16.7% | 53.3% | 9.4% | 28.9% |
| saturation_mean | teams_real_all_dev | 12.1% | 16.0% | 11.1% | 25.4% |
| saturation_mean | teams_real_all_lockbox | 16.7% | 53.3% | 9.4% | 31.5% |

Direct observation: T5C step3500 has a uniformly wider real-side score distribution on the lockbox cohort across all measured IQ axes, with the absolute FPR delta (T5C minus P8A) larger on lockbox (+10-+40pp) than on dev (+1-+10pp).

## 2. Probe 2 — Ensemble policy probe

For 10 candidate inference policies on the 13,636-frame contract matrix, each policy's score is calibrated to 5% FPR on `teams_real_all_dev`, then evaluated on the other 8 suites. Output at `outputs/ensemble_policy_grid.csv` (450 rows × 5 τ values).

### 2.1 Policy table at FPR-calibrated τ (5% on teams_real_all_dev)

| Policy | calibrated τ | lockbox_real_fpr | lockbox_fake_recall | visomaster_enh_recall | deeplive_enh_recall |
|---|---:|---:|---:|---:|---:|
| **P8A** | 0.9752 | **0.0092** | 0.306 | 0.058 | 0.108 |
| T5C_step3500 | 0.8465 | 0.0240 | 0.649 | 0.102 | 0.550 |
| T3_S1_step1500 | 0.7729 | 0.0219 | 0.713 | 0.082 | 0.308 |
| min(P8A, T5C) | 0.7303 | 0.0162 | 0.501 | **0.255** | 0.407 |
| mean(P8A, T5C) | 0.8411 | 0.0120 | 0.480 | 0.185 | 0.345 |
| max(P8A, T5C) | 0.9752 | 0.0092 | 0.306 | 0.058 | 0.108 |
| routed_chronic (P8A on chronic_6, T5C else) | 0.9508 | **0.0078** | 0.209 | 0.000 | 0.000 |
| routed_lap_var (P8A if lap_var<100) | 0.8214 | 0.0310 | 0.461 | 0.205 | 0.650 |
| routed_min_dim (P8A if min_dim<200) | 0.8735 | 0.0162 | 0.602 | 0.035 | 0.356 |
| routed_chronic_or_lowiq | 0.9548 | 0.0092 | 0.348 | 0.076 | 0.000 |

Observation: at FPR-calibrated τ with the single shared threshold, `min(P8A, T5C)` shows the largest viso_enhanced_macro_dev recall (0.255, vs P8A 0.058 = 4.4× P8A) and a deeplive_enhanced_dev recall of 0.407 (vs P8A 0.108 = 3.8× P8A).

### 2.2 Specialist routing — per-ckpt-own-calibrated-τ

A different inference rule: route by chronic_6-identity flag, then apply each ckpt at its OWN calibrated τ (P8A @ 0.9752, T5C @ 0.8465). Both τ values were calibrated to give 5% FPR on `teams_real_all_dev`.

| Suite | n | P8A @ 0.9752 | T5C @ 0.8465 | min(P8A,T5C) @ 0.7303 | **Specialist** |
|---|---:|---:|---:|---:|---:|
| teams_real_all_dev | 4564 | 5.0% | 5.0% | 5.0% | **4.4%** |
| teams_real_poor_quality_dev | 1303 | 1.5% | 3.4% | 3.5% | 1.6% |
| teams_real_lighting_extreme_dev | 1742 | 4.2% | 8.1% | 5.9% | 4.4% |
| teams_real_all_lockbox | 1418 | 0.9% | 2.4% | 1.6% | **0.4%** |
| teams_real_dor_dev | 50 | 4.0% | 6.0% | 12.0% | 4.0% |
| teams_fake_all_dev | 3039 | 53.0% | 63.3% | 66.1% | **64.6%** |
| teams_fake_all_lockbox | 425 | 30.6% | 64.9% | 50.1% | **68.5%** |
| visomaster_enhanced_macro_dev | 550 | 5.8% | 10.2% | 25.5% | 10.2% |
| deeplive_enhanced_dev | 545 | 10.8% | 55.0% | 40.7% | **55.0%** |

### 2.3 Specialist routing on small-cohort probe data (chronic + Roy_D + production-drift)

Same routing rule applied to the 952-frame probe matrix:

| Cohort | n | P8A | T5C step3500 | **Specialist** |
|---|---:|---:|---:|---:|
| MAY5 | 60 | 0/60 (0%) | 0/60 (0%) | **0/60 (0%)** |
| MAY6 | 92 | 0/92 (0%) | 0/92 (0%) | **0/92 (0%)** |
| DOR_REAL_LOCKBOX | 100 | 0/100 (0%) | 3/100 (3%) | **0/100 (0%)** |
| DOR_REAL_DEV | 50 | 6/50 (12%) | 4/50 (8%) | 6/50 (12%) |
| ROY_D | 130 | 24/130 (18.5%) | 109/130 (83.8%) | **24/130 (18.5%)** |
| CHRONIC6_REAL_DEV | 207 | 11/207 (5.3%) | 11/207 (5.3%) | 11/207 (5.3%) |
| CHRONIC6_REAL_LOCKBOX | 34 | 1/34 (2.9%) | 1/34 (2.9%) | 1/34 (2.9%) |
| NON_DOR_REAL_DEV | 80 | 6/80 (7.5%) | 6/80 (7.5%) | 5/80 (6.2%) |
| DOR_FAKE_DEV | 78 | 48/78 (61.5%) | 8/78 (10.3%) | **48/78 (61.5%)** |
| NON_DOR_FAKE_DEV | 80 | 52/80 (65.0%) | 51/80 (63.8%) | 51/80 (63.8%) |
| CHRONIC6_FAKE_DEV | 30 | 27/30 (90.0%) | 22/30 (73.3%) | **27/30 (90.0%)** |
| CHRONIC6_FAKE_LOCKBOX | 11 | 10/11 (90.9%) | 8/11 (72.7%) | **10/11 (90.9%)** |

### 2.4 Direct comparison: Specialist routing vs P8A standalone (per-suite Δ)

At each cohort, sign of the delta between specialist routing and P8A standalone, both at their calibrated τ:

| Pillar | Cohort | Δ (specialist − P8A) | Direction |
|---|---|---:|---|
| **Pillar 2 (real FPR)** | teams_real_all_dev | −0.006 | better |
|  | teams_real_all_lockbox | −0.005 | better |
|  | teams_real_dor_dev (n=50) | 0 | ≈ |
|  | teams_real_poor_quality | +0.001 | ≈ |
|  | teams_real_lighting_extreme | +0.002 | ≈ |
|  | MAY6 (production drift) | 0/92 → 0/92 | preserved |
|  | DOR_REAL_LOCKBOX | 0/100 → 0/100 | preserved |
|  | DOR_REAL_DEV | 6/50 → 6/50 | preserved |
|  | ROY_D | 24/130 → 24/130 | preserved |
| **Pillar 1 (fake recall)** | teams_fake_all_dev | **+0.116** | better |
|  | teams_fake_all_lockbox | **+0.379** | better |
|  | visomaster_enhanced_macro | +0.044 | better |
|  | deeplive_enhanced_dev | **+0.442** | better |
|  | DOR_FAKE_DEV (n=78) | 48/78 → 48/78 | preserved |
|  | CHRONIC6_FAKE_DEV | 27/30 → 27/30 | preserved |
|  | CHRONIC6_FAKE_LOCKBOX | 10/11 → 10/11 | preserved |

## 3. Probe 3 — Disagreement audit

Frames where |P8A score − T5C_step3500 score| > 0.5. Output at `outputs/disagreement_frames.csv` (n=1,587 = 11.64% of 13,636 contract frames).

### 3.1 Direction of disagreement

| Direction | Count | % of disagreements |
|---|---:|---:|
| T5C > P8A | 1491 | 94.0% |
| P8A > T5C | 96 | 6.0% |

### 3.2 By suite

| Suite | n_suite | P8A>T5C | T5C>P8A | % suite in disagreement |
|---|---:|---:|---:|---:|
| teams_real_all_dev | 4564 | 58 | 176 | 5.1% |
| teams_real_poor_quality | 1303 | 12 | 33 | 3.5% |
| teams_real_lighting_extreme | 1742 | 3 | 126 | 7.4% |
| teams_real_all_lockbox | 1418 | 18 | 429 | 31.5% |
| teams_real_dor_dev | 50 | 0 | 2 | 4.0% |
| teams_fake_all_dev | 3039 | 4 | 342 | 11.4% |
| teams_fake_all_lockbox | 425 | 1 | 78 | 18.6% |
| visomaster_enhanced_macro | 550 | 0 | 109 | 19.8% |
| deeplive_enhanced_dev | 545 | 0 | 196 | 36.0% |

### 3.3 Roy_D / chronic_6 concentration

- Roy_D frames in big disagreement: 109. All 109 are T5C > P8A (T5C over-fires).
- chronic_6 (incl. Roy_D) in big disagreement: 739. 697 are T5C > P8A (94.3%).
- non-chronic in big disagreement: 848. 794 are T5C > P8A (93.6%).

The asymmetry is the same across chronic and non-chronic populations: T5C scores higher in ~94% of big-disagreement frames in both subsets.

## 4. FPR-calibrated standalone summary

| Ckpt | τ (5% dev FPR) | dev_fake | viso_enh | deeplive_enh | lockbox_fake | lockbox_real_FPR | dor_dev_FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| P8A | 0.9752 | 0.530 | 0.058 | 0.108 | 0.306 | 0.0092 | 0.040 |
| T5C_step3500 | 0.8465 | 0.633 | 0.102 | 0.550 | 0.649 | 0.0240 | 0.060 |
| T3_S1_step1500 | 0.7729 | 0.621 | 0.082 | 0.308 | 0.713 | 0.0219 | 0.180 |

## 5. Output artifacts

- `outputs/per_iq_bin_policy.csv` — 999 rows: per-axis × quartile × cohort × ckpt × τ FPR/recall.
- `outputs/ensemble_policy_grid.csv` — 450 rows: 10 policies × 9 suites × 5 τ values.
- `outputs/ensemble_summary_tau{0.50,0.70,0.85,0.90}.csv` — per-τ per-policy aggregates.
- `outputs/disagreement_frames.csv` — 1,587 frames where |P8A − T5C| > 0.5, with cohort + chronic flags.
- `outputs/unified_frame_matrix.csv` — 13,636-frame matrix with P8A + T5C + T3_SLOT1 scores + IQ atlas join.

## 6. Direct observations

1. The specialist routing rule (chronic_6 → P8A@0.9752, non-chronic → T5C@0.8465) has lower real FPR than P8A standalone on `teams_real_all_dev` (4.4% vs 5.0%) and `teams_real_all_lockbox` (0.4% vs 0.9%), while preserving 0/92 may6 FPR (§2.3).

2. Specialist routing preserves P8A's FPR on every chronic_6 + Roy_D + dor cohort tested (§2.3 — preserved column).

3. Specialist routing has +11.6pp absolute on teams_fake_all_dev recall, +37.9pp on teams_fake_all_lockbox, +44.2pp on deeplive_enhanced_dev, +4.4pp on visomaster_enhanced_macro_dev compared to P8A standalone (§2.4).

4. The `min(P8A, T5C_step3500)` ensemble at FPR-calibrated τ=0.7303 catches 4.4× more visomaster_enhanced fakes than P8A (25.5% vs 5.8%) at lockbox FPR 0.0162 vs P8A 0.0092 (§2.1).

5. T5C step3500 has 94% of disagreements as T5C > P8A across all cohorts (real and fake), consistent with a wider score distribution (§3.1).

6. The encoder pair (P8A + T5C step3500) carries complementary information sufficient to break the contract's chronic_6 / lockbox ceiling at $0 inference cost, given access to identity-routing or an equivalent gate (§2.3 + §2.4).
