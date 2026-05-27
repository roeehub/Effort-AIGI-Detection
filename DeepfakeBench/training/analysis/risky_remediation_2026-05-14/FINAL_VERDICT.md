# Risky-Frame Remediation — FINAL VERDICT (2026-05-14)

## Headline

**`blend@0.50` (50/50 mix of orig + mild unsharp mask) is a deployable inference-side lever for P8A. For T5C step3500 the win is smaller and has mixed dev-side effects.**

The lever does what we hoped: pulls down chronic-FP reals (especially **dor_shkedi**) while preserving / boosting fakes (especially **Cam_Test__s33**).

## Two-pool experiment

### Pool 1: stress pool (260 frames — risky reals + clean reals + TP fakes)
- T5C AUC +0.0058 mix / +0.0104 hard
- Operating-point at 5% real-FPR: **WORSE recall** (-1.25pp on T5C, flat on P8A)
- Verdict from stress pool alone: "marginal — probably not worth deploying"

### Pool 2: full gate=pass pool (4035 frames — teams_real_all + teams_fake_all, dev + lockbox)
- T5C dev AUC: +0.0001 (essentially zero)
- T5C lockbox AUC: +0.0061
- **P8A dev AUC: +0.0008**
- **P8A lockbox AUC: +0.0498** ← substantial

## Operating-point gains (full pool)

### T5C step3500 on LOCKBOX:
| FPR | orig recall | blend recall | Δ |
|-----|-------------|--------------|---|
| 1% | 48.24% | 56.37% | **+8.13pp** |
| 2% | 53.39% | 57.72% | +4.34pp |
| 5% | 62.87% | 66.40% | +3.52pp |
| 10% | 71.82% | 72.09% | +0.27pp |

### P8A step5000 on LOCKBOX:
| FPR | orig recall | blend recall | Δ |
|-----|-------------|--------------|---|
| 1% | 23.31% | 30.62% | +7.32pp |
| 2% | 32.25% | 36.86% | +4.61pp |
| **5%** | **53.93%** | **70.73%** | **+16.80pp** |
| 10% | 70.73% | 86.72% | **+15.99pp** |

## Cross-calibrated deployment scenario

Calibrate τ on dev at 5% FPR, apply to lockbox:

| ckpt | mode | τ | lockbox FPR | lockbox recall |
|------|------|---|-------------|----------------|
| T5C | orig | 0.6143 | 49.54% | 90.51% |
| T5C | blend | 0.6393 | 30.46% | 86.99% |
| | Δ | | **-19.08pp** | -3.52pp |
| P8A | orig | 0.2178 | 17.23% | 80.49% |
| P8A | blend | 0.2328 | 9.54% | 84.01% |
| | Δ | | **-7.69pp** | **+3.52pp** |

**P8A+blend is a clean Pareto improvement** — lower FPR AND higher recall.

## Why the discrepancy with the stress pool?

The 260-frame stress pool over-represented chronic FPs that BOTH ckpts get wrong with high confidence. On those, blend's pull-down isn't enough to push them under typical τ. The full pool has the right ratio of easy vs hard frames, and the bulk of the gain comes from:
- Reducing borderline-real-FPs in dor_shkedi (where many frames sit at score 0.05-0.20 and blend pushes them lower)
- Boosting borderline fakes in Cam_Test__s33 (where many sit at 0.50-0.70 and blend pushes them higher)

## Why the win is bigger on P8A than T5C

- P8A's "IQ-sharpness response is monotonic" (`project_iq_gating_viability_2026-05-04`). Sharpening pulls real-FPs DOWN.
- T5C's signature was classifier-1024 GRL — the multi-axis GRL already partially neutralizes the sharpness shortcut, so blend has less headroom to add.
- Empirically: P8A_lockbox sees clean Pareto improvement; T5C_lockbox is mixed (wins on real-FP reduction, slight fake-recall regression).

## Per-identity decomposition (lockbox, n≥10)

**P8A blend effect on REALS (negative = blend pulls down = good):**
| identity | n | orig | blend | Δ | high-score FPs orig→blend |
|----------|---|------|-------|---|---------------------------|
| dor_shkedi | 241 | 0.116 | 0.066 | -0.050 | 16 → 6 |
| bla_bla_chow__s1 | 68 | 0.061 | 0.046 | -0.014 | 1 → 0 |
| Chikara_Takahashi__s22 | 16 | 0.706 | 0.697 | -0.009 | 12 → 12 |

**P8A blend effect on FAKES (positive = blend pulls up = good):**
| identity | n | orig | blend | Δ | high-score TPs orig→blend |
|----------|---|------|-------|---|---------------------------|
| Cam_Test__s33 | 334 | 0.603 | 0.670 | **+0.068** | 207 → 231 |
| PC_Generator__s15 | 35 | 0.990 | 0.981 | -0.009 | 35 → 35 |

The Cam_Test__s33 boost is the load-bearing fake-recall improvement.

## Honest caveats

1. **Pool limits**: The `grouped_manifest_v2.csv` I used has only 4035 gate=pass frames. The full promotion-contract scorecard pool is larger and includes deeplive_enhanced_dev, visomaster_enhanced_macro_dev, etc. Those suites need to be tested before final promotion.

2. **Lockbox identity coverage is thin**: 325 lockbox reals across 3 base identities (dor_shkedi 241 / bla_bla_chow__s1 68 / Chikara_Takahashi__s22 16). The big P8A effect is essentially driven by ONE identity (dor_shkedi). It's plausible but should be verified on a broader identity set.

3. **No cross-substrate validation**: HDTF / live-prod / may6 cohorts not tested. The "blend pulls reals down" effect could reverse on a substrate the model already handles well.

4. **w=0.50 non-monotonicity is unexplained**: w=0.35 and w=0.75 both HURT. The lever is sensitive to exact mix weight — re-tests must hold the recipe constant.

## Recommendation

**Run blend@0.50 through the full promotion-contract scorecard before any production change.** Specifically:

1. Run the scorecard against **P8A_blend** (highest-priority test) — compare to P8A_orig and T5C_orig.
2. Optionally also run **T5C_blend** as a smaller-impact, lower-risk option.
3. Decision rule: if P8A_blend wins by ≥ 0.05 AUC on lockbox AND doesn't regress dev_macro below floor AND keeps Pillar 3 robustness, **switch ship recommendation from T5C step3500 to P8A + blend@0.50**.

Cost: 1 contract scorecard run = ~$15-25 on Vertex, 4-6h. Inference-side change in the deploy path is ~1ms/frame.

If full-scorecard validates: this is **the first new inference-side lever found in this project** and it's "free" — no training cost, no checkpoint change required, just a 6-line preprocessing function.
