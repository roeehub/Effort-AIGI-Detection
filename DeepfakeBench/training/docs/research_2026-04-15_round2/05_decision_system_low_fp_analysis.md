# Decision-System Low-FP Analysis

## 1. Why this file exists

The current repo can lose the deployment objective in two different ways:

- the model is genuinely weak on target-domain real Teams data
- the model is usable, but the thresholding / aggregation / gating policy is wrong

Round 2 treats low-FP operation as a first-class system problem instead of assuming all regression is in the backbone.

## 2. Established selection-layer problem

The repo currently mixes:

- training-time checkpoint selection by holdout/OOD AUC
- scorecard-time deployment judgment at threshold `0.5`

This is not a small bookkeeping issue. For a low-FP Teams deployment, thresholding policy is part of the model system.

## 3. What tooling the repo already has

The repo already contains machinery for a more serious decision layer:

- `scripts/run/run_r8_calibration_fit.py`
  - fits Platt / isotonic calibrators
  - uses identity-safe split logic
  - supports FP-first threshold search with a fake-TPR constraint
- `scripts/run/run_r8_apply_calibrator.py`
  - applies saved calibrator bundles
- `app4.py`
  - can consume a calibrator bundle
  - supports video-level aggregation modes such as mean, median, and majority

So a better low-FP policy does not require inventing a new stack.

## 4. What stored decision artifacts already suggest

These artifacts are older than the current R12/R13 finalist shortlist, so they are not authoritative for promotion. They are still useful for estimating decision-layer leverage.

### Binary vs uncertain strategies from `best_strategies_summary.csv`

Selected `teams_only` binary strategy:

- threshold `0.56`
- `K = 16`
- Teams-only `TPR = 0.9691`
- Teams-only `FPR = 0.0126`

Selected `teams_only` uncertain strategy:

- threshold `0.48`
- `K = 16`
- margin `2`
- Teams-only `TPR = 0.9537`
- Teams-only `FPR = 0.0063`
- uncertain rate `0.0365`

Another strong binary configuration selected by the broader `all_three` criterion:

- threshold `0.47`
- `K = 18`
- Teams-only `TPR = 0.9537`
- Teams-only `FPR = 0.0063`

Selected `all_three` uncertain strategy:

- threshold `0.32`
- `K = 20`
- margin `4`
- Teams-only `TPR = 0.9073`
- Teams-only `FPR = 0.0032`
- uncertain rate `0.0642`

Interpretation:

- even in older artifacts, threshold policy alone moves Teams FPR substantially
- an abstain band can cut FPR further at manageable uncertain rates

### Extremity gate results from `extremity_gate_full.csv`

Baseline ungated operating point from `analyze_gate_results.py`:

- `T = 0.30`
- `K = 18`
- `TPR = 0.9378`
- `FP = 18`
- `BalAcc = 0.9464`

Useful gated alternatives at the same `T` and `K`:

- best mild gate:
  - `FP = 15`
  - `TPR = 0.9341`
  - `BalAcc = 0.9483`
- stronger gate:
  - `FP = 12`
  - `TPR = 0.9206`
  - `BalAcc = 0.9453`

Interpretation:

- simple gating can often save `3` false positives with only about `0.0037` absolute TPR cost on that older dataset
- aggressive gating can save more, but starts to trade too much recall

## 5. What this means for the current shortlist

### Established

- the current deployment contract is threshold-sensitive
- fixed threshold `0.5` is not a trustworthy final operating point
- the repo already has practical calibration and voting primitives

### Plausible

- some of the current real Teams false-positive pain can be reduced without new training
- some checkpoint orderings may flip after per-checkpoint calibration

### Still unknown

- whether calibrated `R13_FT7` or `R13_FT9` beats calibrated `R12_G`
- whether `R13_E` recovers enough real-side behavior when judged on the actual low-FP contract

## 6. Recommended decision-layer changes before more heavy retraining

### 1. Per-checkpoint target-domain calibration and threshold sweep

Run this first on the frozen shortlist:

- `R12_G_FP32`
- `R13_A_STEP15500`
- `R13_E_BESTSOFAR`
- `R13_FT7_FP32`
- `R13_FT9_FP32`

Use the dev portion of the frozen Teams suite to fit a per-checkpoint threshold with the contract from `01_evaluation_contract_and_shortlist.md`.

Why first:

- highest value
- lowest engineering risk
- directly addresses the current selection mismatch

### 2. Video-level temporal aggregation plus hysteresis

Use the existing aggregation machinery, but add deployment logic such as:

- median or EMA score aggregation
- minimum consecutive positive windows before raising an alert
- separate raise / clear thresholds

Why second:

- directly targets score instability
- cheaper than retraining
- consistent with live-call use

### 3. Disagreement or extremity gate

Use a small gate based on one of:

- crop perturbation disagreement
- temporal score volatility
- margin-to-threshold
- multi-crop extremity consistency

Why third:

- older repo artifacts suggest a modest FPR win is realistic
- should be tested only after calibration, not instead of calibration

### 4. Narrow abstain band

For operationally sensitive cases, use a reject band around the decision threshold.

Why fourth:

- useful if the business can tolerate uncertain outputs
- especially relevant if the model becomes unstable near the threshold under shift

## 7. What this will not solve

Decision-layer improvements cannot create missing data conditions.

They will not:

- manufacture enhanced-through-Teams evidence
- fix a missing enhanced fake lockbox
- replace a real curriculum/sampler redesign if the model still lacks exposure to the actual target condition

## 8. Practical recommendation

The next serious measurement move should be:

1. calibrated threshold sweep on the frozen shortlist
2. one temporal aggregation policy
3. one small gate or abstain variant

Do that before launching another family of weight-only fine-tunes.

## 9. Status

- Established:
  - low-FP performance here is a system design problem, not just a model-score problem
  - repo-local evidence already supports threshold and gate leverage
- Plausible:
  - calibrated selection may recover some of the apparent Track A regression story
- Still unknown:
  - how much of the current gap is fixable at the decision layer versus only by better target-domain training
