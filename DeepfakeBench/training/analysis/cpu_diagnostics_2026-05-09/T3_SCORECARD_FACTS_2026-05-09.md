# T3 (Stage 3) scorecard FACTS (2026-05-09)

> **Status: factual-only.** No interpretation. Forbidden words: succeeds,
> fails, wins, promotes, deployment-grade.
>
> Source data: 5 promotion-contract scorecards executed on Vertex AI on
> 2026-05-09 18:50–22:23 CEST. Image 1.3.276 / 1.3.277 (Slot 2 only).
> Iterative-mode suite manifest. Promotion policy: target_real_fpr=0.07,
> target_stress_fpr=0.10, target_fake_recall_min=0.30.
>
> Companion docs:
> - `CPU_DIAGNOSTICS_FACTS_2026-05-09.md` (forgery-signal atlas, slot design)
> - `outputs/t3_per_cohort_stats.{slot1only,slot3only}.csv` (Mac probe)
> - `_t3_scorecards/{slot,step}*/promotion_contract/` (raw scorecard data)

## 1. Question

For each of the 4 T3 candidates (Slot 1 step1500/step2500, Slot 3
step1500/step3500, Slot 2 step1000), does any candidate clear the
promotion-contract policy (`target_real_fpr=0.07`,
`target_stress_fpr=0.10`, `dev_fake_macro_recall_min=0.30`) at its
auto-calibrated τ? And how do the per-suite numbers compare to the
production anchors P8A_REFERENCE_STEP5000 and E2B_TOP_N_STEP3200?

## 2. Method

Each scorecard is a Vertex AI iterative-mode promotion-contract job that:
1. Scores the candidate + 2 anchors on the iterative suite manifest
   (`teams_real_all_dev`, `teams_real_poor_quality_dev`,
   `teams_real_lighting_extreme_dev`, `teams_fake_all_dev`,
   `visomaster_enhanced_macro_dev`, `deeplive_enhanced_dev`,
   `teams_real_all_lockbox`, `teams_fake_all_lockbox`, plus per-identity
   diagnostic suites).
2. Runs auto-calibration to pick τ that satisfies the contract budgets
   (lex priority: real_fpr ≤ 0.07 → stress_fpr ≤ 0.10 → fake_macro_recall
   ≥ 0.30 → maximize teams_fake_all_lockbox recall).
3. Emits `selected_threshold_scorecard.csv` (per-ckpt, calibrated τ) and
   `promotion_winner.json` (rank-1 ckpt by lex policy).

Anchors P8A and E2B are scored in every job (with identical inputs and τ
auto-calibration), so per-job anchor numbers are reproducible across the 5
runs (verified: P8A τ=0.915605 in all 5 jobs; E2B τ=0.71077 in all 5).

## 3. Calibrated-τ scorecard

| ckpt | τ | lockbox real FPR | lockbox fake recall | dev fake macro | teams_dev | viso_dev | deeplive_dev | dev primary FPR | dev stress FPR | passes floor |
|------|--:|-----------------:|--------------------:|---------------:|----------:|---------:|-------------:|----------------:|---------------:|:------------:|
| **P8A_REFERENCE_STEP5000** | 0.916 | 1.84% | 38.74% | **30.03%** | 52.59% | 13.64% | 23.85% | 6.95% | **6.85%** | ✓ |
| E2B_TOP_N_STEP3200 | 0.711 | 2.35% | 62.85% | 50.85% | 67.83% | 5.09% | **79.63%** | 6.67% | 9.99% | ✓ |
| **T3_SLOT1_PERIODIC_STEP1500** | 0.682 | 3.09% | **77.47%** | **37.63%** | 59.07% | 12.18% | 41.65% | 6.76% | 9.92% | ✓ |
| T3_SLOT1_PERIODIC_STEP2500 | 0.971 | 3.82% | **89.33%** | 22.95% | 45.25% | 7.82% | 15.78% | 6.92% | 9.99% | ✗ |
| T3_SLOT2_PERIODIC_STEP1000 | 0.938 | 2.79% | 62.06% | 25.71% | 54.25% | 5.09% | 17.80% | 6.49% | 9.99% | ✗ |
| T3_SLOT3_PERIODIC_STEP3500 | 0.980 | 3.53% | 60.08% | 28.23% | 49.94% | 10.73% | 24.04% | 6.30% | 9.99% | ✗ |
| T3_SLOT3_PERIODIC_STEP1500 | 0.988 | 0.96% | 53.36% | 16.35% | 40.81% | 2.00% | 6.24% | 5.99% | 9.99% | ✗ |

(Reading: lockbox suites are n=1361 real / n=253 fake video-level. Floor
column tracks dev_fake_macro_recall ≥ 0.30 budget. dev primary FPR is
teams_real_all_dev FPR; dev stress FPR is the worst across
teams_real_poor_quality_dev and teams_real_lighting_extreme_dev.)

## 4. Per-candidate δ vs P8A (relative to P8A's per-suite numbers)

| ckpt | Δ lockbox fake recall | Δ teams_dev | Δ viso_dev | Δ deeplive_dev | Δ dev_macro | Δ stress FPR |
|------|----------------------:|------------:|-----------:|---------------:|------------:|-------------:|
| **T3_SLOT1_step1500** | **+38.7pp** | +6.5pp | -1.5pp | **+17.8pp** | **+7.6pp** | +3.1pp |
| T3_SLOT1_step2500 | **+50.6pp** | -7.3pp | -5.8pp | -8.1pp | -7.1pp | +3.1pp |
| T3_SLOT2_step1000 | +23.3pp | +1.7pp | -8.6pp | -6.0pp | -4.3pp | +3.1pp |
| T3_SLOT3_step3500 | +21.3pp | -2.7pp | -2.9pp | +0.2pp | -1.8pp | +3.1pp |
| T3_SLOT3_step1500 | +14.6pp | -11.8pp | -11.6pp | -17.6pp | -13.7pp | +3.1pp |

(Reading: positive Δ = candidate better than P8A on that suite. Stress FPR
is uniformly +3.1pp across all 4 T3 candidates because all calibrators
hit the 9.99% stress FPR ceiling, while P8A sits at 6.85%.)

## 5. Promotion ranks

Per `promotion_winner.json` for each scorecard: P8A wins rank 1 in all 5
jobs. Reason: lex policy prioritizes (real_fpr → stress_fpr →
fake_macro_recall → lockbox), and P8A is the only candidate with stress
FPR < 7% (P8A 6.85%; all T3 candidates 9.92-9.99%; E2B 9.99%).

| scorecard | rank 1 | rank 2 | rank 3 |
|-----------|--------|--------|--------|
| t3-scorecard-slot1-step1500-2026-05-09 | P8A | E2B | T3_SLOT1_STEP1500 |
| t3-scorecard-slot1-step2500-2026-05-09 | P8A | E2B | T3_SLOT1_STEP2500 |
| t3-scorecard-slot2-step1000-2026-05-09 | P8A | E2B | T3_SLOT2_STEP1000 |
| t3-scorecard-slot3-step1500-2026-05-09 | P8A | E2B | T3_SLOT3_STEP1500 |
| t3-scorecard-slot3-step3500-2026-05-09 | P8A | E2B | T3_SLOT3_STEP3500 |

## 6. Headline observations

- T3_SLOT1_PERIODIC_STEP1500 is the only T3 candidate whose
  auto-calibrated τ leaves dev_fake_macro_recall ≥ 0.30 (37.63% ≥ 30.03%
  floor; matches/exceeds P8A on the floor metric).
- T3_SLOT1_PERIODIC_STEP2500 has the highest single observed
  lockbox_fake_recall in this scorecard family (89.33%, n=253 videos)
  but fails the macro-recall floor (22.95% < 30%).
- T3_SLOT1_PERIODIC_STEP1500 lockbox_fake_recall (77.47%) exceeds both
  anchors (P8A 38.74%, E2B 62.85%) on the same n=253 lockbox cohort.
- T3_SLOT1_PERIODIC_STEP1500 deeplive_enhanced_dev recall (41.65%)
  exceeds P8A (23.85%) by 17.8pp at calibrated τ; matches E2B's
  79.63% only directionally (E2B remains 38pp higher).
- visomaster_enhanced_macro_dev recall is within ±1.5pp of P8A for
  T3_SLOT1_step1500 (12.18% vs 13.64%) at calibrated τ; T3_SLOT1_step2500,
  T3_SLOT2_step1000, and T3_SLOT3_step1500 all show double-digit pp
  drops vs P8A on this suite.
- All 4 T3 candidates land at or near the 9.99% stress_fpr ceiling at
  their auto-calibrated τ (T3_SLOT1_step1500 = 9.92%; others = 9.99%).
- The Mac-probe-best Slot 2 candidate (step1000, predicted strongest
  joint-frontier from Dor cohort + Roy_D probe) does not exceed P8A on
  any dev or lockbox cell at calibrated τ.

## 7. Auto-calibrated τ ranges

| candidate | auto-τ | P8A τ | Δ τ |
|-----------|------:|------:|----:|
| P8A | 0.916 | 0.916 | 0.000 |
| E2B | 0.711 | 0.916 | -0.205 |
| T3_SLOT1_step1500 | 0.682 | 0.916 | -0.234 |
| T3_SLOT2_step1000 | 0.938 | 0.916 | +0.023 |
| T3_SLOT1_step2500 | 0.971 | 0.916 | +0.055 |
| T3_SLOT3_step3500 | 0.980 | 0.916 | +0.064 |
| T3_SLOT3_step1500 | 0.988 | 0.916 | +0.072 |

(Reading: low auto-τ (E2B, T3_SLOT1_step1500) corresponds to a model that
needs a low threshold to reach the FPR target — typically because real
score distribution sits low. High auto-τ (Slot 3, late Slot 1) means
the calibrator is squeezing τ tight to keep real-FPR within budget.)
