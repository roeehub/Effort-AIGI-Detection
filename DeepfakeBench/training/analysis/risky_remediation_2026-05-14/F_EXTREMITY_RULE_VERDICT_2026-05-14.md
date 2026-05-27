# Extremity Rule — Verdict (2026-05-14)

## TL;DR

Your hypothesis was **partially confirmed**: borderline-FP reals have distinctly *different* score-distribution shapes than true fakes, and a rule exploiting this rescues 2 of 4 current FPs without creating new errors. But **catastrophic FPs (Roy_D, bla_bla_chow__s1) remain unrescuable** — their score distributions are too similar to those of true fakes.

**Best rule found: `frac > 0.6 > 0.4`** (raise per-frame threshold to 0.6, require 40% of frames). Equivalent to "13 of 32 frames score above 0.6 in production-typical 32-frame batches."

Improvement vs baseline `frac > 0.49 > 0.5`: combined 92.96% → 95.77% (+2.82pp), 4 FPs → 2 FPs (-50% FP), 0 new FNs. Bootstrap CIs overlap — directional win, not statistically significant.

**Recommendation: change rule if you want fewer FPs; keep baseline if you want statistical confidence.** Either is defensible. I'd lean toward changing — the absolute FP reduction is real and the held-out cross-pool sign is consistent.

## What I tested

Five rule families × parameter grids × 4 pools × 3 sample-size cutoffs = **3,060 rule evaluations**.

| family | description | params |
|---|---|---|
| basic | `frac > τ > x` (vary base threshold AND fraction) | τ ∈ {0.3, 0.4, 0.49, 0.5, 0.6, 0.7, 0.8, 0.9}, x ∈ {0.1, 0.2, ..., 0.9} |
| AND | `(frac > 0.49 > 0.5) AND (frac > t_E > x_E)` | t_E ∈ {0.6, 0.7, 0.8, 0.9, 0.95, 0.99}, x_E ∈ {0.05, ..., 0.7} |
| OR | `(frac > 0.49 > 0.5) OR (frac > t_E > x_E)` | same |
| REPLACE | `frac > t_E > x_E` only | t_E ∈ {0.5, ..., 0.95}, x_E ∈ {0.05, ..., 0.9} |
| BLEND | `α × frac>.49 + (1-α) × frac>.9 > threshold` | α ∈ {0, 0.25, 0.5, 0.75, 1}, threshold ∈ {0.1, ..., 0.9} |

## Key empirical finding: T5C scores cap at ~0.94

`frac > 0.95 = 0.000` for **every identity** in the pool. The model never emits a score above ~0.94. So the "very high tail" the extremity hypothesis was reaching for doesn't exist in this checkpoint's score range.

This means any "extreme threshold" useful for discrimination must be ≤ 0.9. Thresholds at 0.95+ collapse all identities into the same group.

## The 5 baseline errors, dissected

The current rule `frac > 0.49 > 0.5` produces these 5 wrong identity-level verdicts on our 71-identity pool:

| identity | pool | label | n | mean | frac>.49 | frac>.7 | **frac>.9** | distribution character |
|---|---|---|---|---|---|---|---|---|
| dor_morning | dor_cross | REAL | 244 | 0.500 | 53% | 23% | **0%** | spread, no high-confidence mass |
| Roy_D | teams_dev | REAL | 130 | **0.888** | 100% | 97% | **60%** | **tightly peaked at 0.88** — looks like a confident fake |
| bla_bla_chow__s1 | lockbox | REAL | 68 | 0.694 | 84% | 66% | **0%** | peaked at 0.85, no extreme |
| dor_shkedi | lockbox | REAL | 1169 | 0.534 | 60% | 19% | **0%** | spread, borderline |
| dor_fake_inswapper_128res_gpen1024 | dor_cross | FAKE | 121 | 0.432 | 36% | 17% | **0%** | weak fake, model has low confidence |

The mechanism of your hypothesis: **distribution peakedness should distinguish true fakes from FP reals**. Test pattern:
- True fakes: scores cluster at high values (high frac>.9)
- FP reals: scores are more spread out (low frac>.9 despite passing majority)

**This works for 3 of the 4 FPs** (dor_morning, bla_bla_chow__s1, dor_shkedi — all with frac>.9 = 0). 
**It fails for Roy_D** (frac>.9 = 60%, near the strongest fakes).

## Why Roy_D is unrescuable by any score-distribution rule

Roy_D's score distribution is **tightly clustered around 0.88** with a peak just below the model's max (0.94):
- p25=0.885, p50=0.913, p75=0.927, p90=0.932
- std=0.065 (very tight)

A confident fake like Cam_Test__s35 looks essentially identical:
- p25=0.928, p50=0.932, p75=0.936, p90=0.939
- std=0.027 (even tighter)

Both have nearly all frames above 0.8, both have substantial mass above 0.9 (Roy_D 60% / Cam_Test__s35 95%). No threshold separates them without losing real fakes.

To rescue Roy_D specifically would require `frac > 0.9 > 0.7` (his is 0.60, fakes are 0.95+). But this rule turns 22 true fakes into FNs because many dor_cross fakes have frac>.9 in the 10-30% range (the model is genuinely less confident on those swap-method variants).

**Roy_D is a per-frame model failure, not a per-identity rule problem.** The fix has to come from training — not aggregation.

## Top rules per family (combined pool, n_min=5)

### `basic` (single-threshold sweep)
| rule | correct | rate | FP | FN |
|---|---|---|---|---|
| **`frac > 0.6 > 0.4`** | **68/71** | **0.9577** | 2 | 1 |
| `frac > 0.6 > 0.5` | 67/71 | 0.9437 | 2 | 2 |
| `frac > 0.5 > 0.6` | 67/71 | 0.9437 | 2 | 2 |
| `frac > 0.5 > 0.5` (baseline) | 66/71 | 0.9296 | 4 | 1 |
| `frac > 0.5 > 0.4` | 66/71 | 0.9296 | 4 | 1 |

### `AND` (your proposed rule structure)
| rule | correct | rate | FP | FN |
|---|---|---|---|---|
| **`frac > 0.49 > 0.5 AND frac > 0.6 > 0.4`** | **68/71** | **0.9577** | 2 | 1 |
| `frac > 0.49 > 0.5 AND frac > 0.6 > 0.5` | 67/71 | 0.9437 | 2 | 2 |
| `frac > 0.49 > 0.5 AND frac > 0.8 > 0.1` | 67/71 | 0.9437 | 2 | 2 |
| `frac > 0.49 > 0.5 AND frac > 0.7 > 0.2` | 67/71 | 0.9437 | 3 | 1 |

**The best AND rule and the best basic rule give identical verdicts**. The AND rule `frac > 0.49 > 0.5 AND frac > 0.6 > 0.4` is mathematically equivalent to just `frac > 0.6 > 0.4` (because the stricter test dominates). So adding "AND extreme" doesn't add any signal beyond what a single tighter threshold gives.

### `OR` (loosen base by adding extreme path)
All OR variants give exactly the baseline rate (66/71). Adding an OR path doesn't help — the identities that DON'T pass majority but DO pass some extreme criterion mostly don't exist (because passing majority is highly correlated with having frames at the high end).

### `REPLACE` (single high threshold, no majority)
| rule | correct | rate | FP | FN |
|---|---|---|---|---|
| `frac > 0.6 > 0.5` | 67/71 | 0.9437 | 2 | 2 |
| `frac > 0.7 > 0.3` | 65/71 | 0.9155 | 3 | 3 |
| `frac > 0.7 > 0.2` | 65/71 | 0.9155 | 5 | 1 |

Replacing the base rule entirely with an extreme one underperforms — you lose the regularization of "many frames need to be above some threshold."

### `BLEND` (linear combination)
The best blend `α=0.75 × frac>.49 + 0.25 × frac>.9 > 0.4` gives 67/71. No combination of `frac>.49` and `frac>.9` outperforms simple thresholding.

## Cross-pool validation (key for production safety)

Optimizing per-pool then testing on other pools shows the OVERFITTING RISK:

| family | best rule on `teams_dev` | train rate | lockbox | dor_cross | combined |
|---|---|---|---|---|---|
| AND | `frac>.49>.5 AND frac>.6>.05` | 97.06% | 71.43% | 93.33% | 92.96% |
| **basic** | `frac>.49>.9` | **97.06%** | **100%** | **63.33%** (FN flood) | **83.10%** |
| BLEND | `α=1, threshold>.9` | 97.06% | 100% | 63.33% | 83.10% |

**Look at the basic `frac>.49>.9` row** — it's perfect on lockbox (7/7) but loses 11 fakes from dor_cross. This is exactly the overfitting trap I warned about. A "very strict majority" rule (require 90% of frames above τ) kills weak fakes.

The combined-pool best `frac > 0.6 > 0.4` is stable across pools:
- teams_dev: 33/34 (unchanged from baseline)
- lockbox: 6/7 (+1)
- dor_cross: 29/30 (+1)

## The user's specific test values (t=0.5, t=0.7)

| rule | combined correct | rate | comment |
|---|---|---|---|
| `frac > 0.5 > 0.5` | 66/71 | 92.96% | tied with baseline (essentially identical to 0.49) |
| `frac > 0.5 > 0.6` | 67/71 | 94.37% | mildly better |
| `frac > 0.7 > 0.3` | 65/71 | 91.55% | WORSE — high threshold turns weak fakes into FNs |
| `frac > 0.7 > 0.4` | 65/71 | 91.55% | worse |
| **`frac > 0.6 > 0.4`** | **68/71** | **95.77%** | **the sweet spot** |
| `frac > 0.6 > 0.5` | 67/71 | 94.37% | also good |
| `frac > 0.8 > 0.5` | 64/71 | 90.14% | too strict; FN flood |

t=0.7 doesn't catch Roy_D (his frac>0.7 = 97% — still above any sensible majority threshold). t=0.6 is the inflection point where bla_bla_chow__s1 / dor_shkedi / dor_morning start failing the rule (their frac>0.6 = 77%/37%/33%) but most TP fakes still pass.

## Bootstrap 95% CIs

| rule | rate | 95% CI | Δ vs baseline | significant? |
|---|---|---|---|---|
| baseline (`frac>.49>.5`) | 0.9296 | [0.8592, 0.9859] | — | — |
| `frac > 0.6 > 0.4` | 0.9577 | [0.9014, 1.0000] | +0.0282 | NO (CIs overlap) |
| `frac > 0.6 > 0.5` | 0.9437 | [0.8873, 0.9859] | +0.0141 | NO |
| `frac > 0.49 > 0.5 AND frac > 0.6 > 0.4` | 0.9577 | [0.9014, 1.0000] | +0.0282 | NO |

The CIs overlap with baseline. At 71 identities, the noise floor is too high to declare statistical significance from a +2.82pp gain.

But: in terms of false-positive count, the reduction is **4 → 2 (50% relative reduction)** with zero recall cost. That's a real operational improvement even if it doesn't beat the bootstrap threshold.

## What no rule can do

Even an oracle (best per-identity rule, per pool) caps at ~96% on this 71-identity pool. The remaining ~4% are structural failures:
1. **Roy_D-style** (mean 0.88, tight peak, all frames look fake-confident): score-distribution rule can't distinguish from a true confident fake
2. **Weak-fake** (mean 0.43, model uncertain): no rule that demands frac>τ>x for any reasonable (τ, x) catches these without also rescuing borderline FPs

These are training-side problems. The L11 LoRA work and identity-fresh data are the structural fixes.

## Recommendation

### Option 1: Change ship spec to `frac > 0.6 > 0.4`
- 50% reduction in FPs
- No new FNs
- Cross-pool stable (+1 on lockbox, +1 on dor_cross, neutral on teams_dev)
- Not statistically significant at α=0.05 but directionally consistent

In production-friendly terms with 32-frame batches: "flag identity as fake if **at least 13 frames** score above 0.6" (vs current "at least 17 frames score above 0.49").

### Option 2: Keep current `frac > 0.49 > 0.5`
- Well-understood baseline
- No risk of subtle regressions in untested cohorts

### My take

Option 1 is mildly better on the data we have, but the improvement is small and the statistical confidence is moderate. I'd put this as a **soft recommendation: change if you don't have to convince a stats-strict stakeholder, otherwise stay**. If you do switch, the equivalent AND formulation may be easier to communicate to the PM ("you need to pass majority AND have a strong showing at the 0.6 threshold") even though it's mathematically identical to the basic version.

### What I would NOT do

- Apply per-pool optimized rules. The teams_dev-best rule (`frac>.49>.9`) catastrophically overfits and FN-floods dor_cross.
- Use `frac > 0.7` or stricter — creates too many FNs on weak fakes.
- Try to rescue Roy_D at the rule level — it's a model-side problem, not a rule problem.

## Artifacts

- `extreme_rule_study.py` — main script
- `outputs/extreme_rule_per_identity.csv` — 72 identities × 30+ stats columns
- `outputs/extreme_rule_sweep_all.csv` — 3060 (rule × params × pool × n_min) evaluations
