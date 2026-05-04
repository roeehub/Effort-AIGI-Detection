# Preliminary score distribution analysis (PA + PC, partial F0 data)

**Date authored**: 2026-05-05 (during PA+PC eval, suite 1 of 9 complete)
**Data**: PA+PC F0 contract scorecard run 7756239039929253888, suite `teams_real_all_dev` (n=4564) for all 8 ckpts.
**Status**: PRELIMINARY — fake-suite data not yet available; this characterizes only score distributions on reals.

---

## τ calibration on `teams_real_all_dev`

| Ckpt | n | τ@FPR=10% | τ@FPR=5% | τ@FPR=2% | mean | p50 |
|---|---:|---:|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 4564 | 0.7055 | 0.9756 | 0.9926 | 0.1345 | 0.0070 |
| E2B_TOP_N_STEP3200 | 4564 | 0.5082 | 0.7442 | 0.8971 | 0.1202 | 0.0074 |
| PA_TOP_N_STEP5600 | 4564 | 0.6327 | 0.9034 | 0.9839 | 0.1339 | 0.0072 |
| PA_TOP_N_STEP3800 | 4564 | **0.4981** | **0.7208** | 0.9352 | 0.1402 | 0.0161 |
| PA_PERIODIC_STEP5000 | 4564 | **0.8640** | 0.9615 | 0.9882 | **0.2132** | 0.0165 |
| PC_TOP_N_STEP7400 | 4564 | 0.7597 | 0.9431 | 0.9871 | 0.1652 | 0.0115 |
| PC_TOP_N_STEP5400 | 4564 | **0.8708** | 0.9725 | 0.9916 | **0.2048** | 0.0170 |
| PC_PERIODIC_STEP5000 | 4564 | 0.7032 | 0.9044 | 0.9774 | 0.1890 | 0.0332 |

## Observations

1. **PA_TOP_N_STEP3800 is the most E2B-like** (τ@FPR=10% = 0.50, similar to E2B's 0.51). Mid-training, scores are spread.
2. **PA_PERIODIC_STEP5000 is MORE SATURATED than P8A** (τ@FPR=10% = 0.86 vs P8A's 0.71). Late-step PA drifts toward score saturation.
3. **PA_TOP_N_STEP5600 is intermediate** between E2B and P8A.
4. **PC_TOP_N_STEP5400 is the MOST saturated of all 8 ckpts** (τ@FPR=10% = 0.87). Codec aug + early-step convergence saturates scores.
5. **PC_TOP_N_STEP7400 is less saturated than PC_TOP_N_STEP5400** — interesting non-monotonic behavior.
6. **Mean real-side score is elevated for PA_PERIODIC, PC_TOP_N_5400 (0.20-0.21)** vs P8A/E2B (0.12-0.13). PA/PC are over-confident on reals at saturation.

## Implications

A higher τ at the same FPR target means deployment τ moves UP, which makes it HARDER to catch fakes whose scores fall below the τ. The τ-tail collapse pattern (per `project_p16_data_axis_does_not_promote_2026-04-30.md`) is amplified for ckpts with higher saturation.

**Predicted ordering of τ-tail collapse severity** (most → least):
1. PC_TOP_N_STEP5400 (highest τ@FPR=10% at 0.87)
2. PA_PERIODIC_STEP5000 (0.86)
3. PC_TOP_N_STEP7400 (0.76)
4. P8A_REFERENCE_STEP5000 (0.71)
5. PC_PERIODIC_STEP5000 (0.70)
6. PA_TOP_N_STEP5600 (0.63)
7. E2B_TOP_N_STEP3200 (0.51)
8. PA_TOP_N_STEP3800 (0.50)

By this prediction, PA_TOP_N_STEP3800 should preserve fake recall best at deployment τ; PC_TOP_N_STEP5400 should collapse worst.

This contradicts the AUC-based selection (PA's best top_n is step5600 by AUC; the model dashboard runner uses AUC for top_n selection). The most-deployable PA ckpt may be top_n_step3800 not top_n_step5600.

## Cross-ckpt notes

- **PA top_n_step3800 vs E2B**: very similar score distributions on reals (τ@FPR=10% within 0.01). PA might inherit E2B's deployment-tau behavior.
- **PA periodic_step5000 vs PA top_n_step5600**: same training run (`26u8bn1t`), different selection. Periodic 5000 saturates harder than top_n at step5600 (which is later in training but selected for AUC). Suggests AUC and saturation are de-correlated in this regime.
- **PC's 3 ckpts have notably different saturation profiles** (0.70, 0.76, 0.87). Stage of training matters more for PC than for PA.

## What this suggests for full F0 verdict

**Likely outcomes when fake-suite data lands**:
- At deployment τ (FPR=2%): the most-saturated ckpts (PC_TOP_N_STEP5400, PA_PERIODIC_STEP5000) will have the worst fake recall on viso (canonical pattern: saturation eats recall).
- At τ=0.5: all ckpts will have moderate-to-high recall — but that's not deployment-relevant.
- F4 (chronic-6 cleaned) may help more on saturated ckpts (lower required τ).

I will revisit these predictions when fake-suite data lands. Pre-registered here so they can be checked.

## Cross-references

- `PRE_LANDING_PREDICTIONS.md` — original numerical predictions
- `IQ_VALLEY_FINDING.md` — the IQ-shortcut valley framing
- `per_ckpt_iq_signatures.md` — Pearson r per existing ckpts
- `analysis/pa_pc_eval_2026-05-05/raw_reports/` — source frame_report CSVs (gitignored)
