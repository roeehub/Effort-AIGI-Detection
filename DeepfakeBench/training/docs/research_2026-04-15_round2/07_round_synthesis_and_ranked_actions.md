# Round 2 Synthesis And Ranked Actions

## 1. What changed from the previous round

Five conclusions tightened materially.

### A. The evaluation contract is now clearer

The repo was still blending:

- training checkpoint metrics
- fixed-threshold Teams scorecards
- mixed-source regression suites

Round 2 separates them. Promotion should be decided on the frozen Teams lane under a calibrated low-FP threshold policy, not on holdout AUC and not on mixed-source mega-eval.

### B. The target-domain data story is harsher than the labels suggest

Local truth supports this stronger claim:

- real Teams data exists
- direct Teams fake data exists
- enhanced clean fake data exists
- enhanced-through-Teams fake data is still structurally absent

That missing condition is not "underweighted." It is missing.

### C. The sampler is weaker than the YAML names imply

The key Track A merged source is dominated by clean-fallback VTE rows.

- baseline merged lane: only about `17.29` expected true Teams-companion VTE selections per epoch
- corrected teamsonly lane: only about `20.18` expected VTE selections per epoch, even after filtering to true Teams companions

So the current sampler/curriculum is not delivering the late target-domain emphasis the experiment names imply.

### D. The augmentation story was overstated

The current R13 configs appear to request gamma-up, but runtime does not activate it. Direct Teams rows also still bypass the stronger nuisance pipeline and receive only a light passthrough augmentation.

### E. The decision layer is more promising than the current process admits

Repo-local artifacts already suggest that calibration, aggregation, and small gates can move the low-FP tradeoff without retraining. That is cheaper and better constrained than another weight-only fine-tune family.

## 2. What now seems more likely

- The immediate bottleneck is evaluation/selection discipline plus missing-condition coverage, not yet another family-weight sweep.
- Some of the apparent checkpoint ordering may flip once candidates are judged on a calibrated Teams low-FP contract.
- Track A-like gains are real on some fake slices, but they still have not been translated into a trustworthy deployment promotion result.
- The most valuable training change is probably a real sampler/curriculum change, not a simple weight change inside the current sampler.

## 3. What now seems less likely

- `FT8` and `FT10` represent meaningful new evidence beyond `FT7` and `FT9` if local external-real truth is representative.
- Current teamsonly configs are delivering a strong target-domain late lesson.
- Current R13 augmentation YAML already covers the intended brightness-up hypothesis.
- The best next move is a backbone search.

## 4. Top ranked next actions

### 1. Measure before training

Run the frozen shortlist on the frozen Teams suite with a calibrated low-FP contract.

Shortlist for that run:

- `R12_G_FP32`
- `R13_A_STEP15500`
- `R13_E_BESTSOFAR`
- `R13_FT7_FP32`
- `R13_FT9_FP32`

Collapse `FT8` and `FT10` unless external-real loading is proven.

Why this is first:

- it is the highest-value unresolved question
- it can invalidate several comforting stories quickly
- it may reduce unnecessary new training

### 2. Fix augmentation plumbing, then run the smallest real nuisance ablation

Do not launch more lighting-robustness claims until the augmentation path is trustworthy.

The minimum worthwhile sequence is:

- tiny preset-key allowlist fix
- plumbing-control run
- gamma-up-only run
- Teams-spatial-only run

### 3. Redesign the sampler/curriculum around true Teams-companion exposure

Either:

- split VTE into explicit true-Teams and clean-fallback families
- or change the sampler so overlapping direct-Teams / VTE identities are not winner-take-most

Why this is third:

- current weight tuning has weak leverage
- the missing-condition story will not improve without a more explicit curriculum

### 4. Only then revisit capacity/backbone questions

The small-model question is still live, but it is not the next constraint to attack.

## 5. What should explicitly **not** be done next

- Do not run another family of `FT7/8/9/10`-style weight tweaks and call it progress.
- Do not promote a checkpoint from threshold-`0.5` tables alone.
- Do not treat mixed-source mega-eval as the promotion contract.
- Do not spend the next round mainly on generic model-capacity brainstorming.
- Do not assume "teamsonly" means the model is finally seeing real enhanced-through-Teams supervision.

## 6. Bottom line

The best next move is **measure more before training more**.

Specifically:

- freeze the calibrated promotion contract
- score the reduced shortlist on it
- fix the augmentation plumbing
- then test one real sampler/curriculum intervention

Everything else should wait behind that.

## 7. Status

- Established:
  - the repo's current decision contract is misaligned
  - the missing target condition is still missing
  - current sampler leverage is weaker than experiment naming suggests
  - augmentation intent and augmentation runtime are not the same thing
- Plausible:
  - calibrated low-FP selection and a better decision layer may recover meaningful value before more retraining
- Still unknown:
  - whether any current R13 checkpoint truly beats `R12_G` once judged on the right contract
