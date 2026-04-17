# Literature Deepening

## 1. Scope

Round 1 already covered general stability / calibration / augmentation themes. This round went narrower:

- video-call / conferencing robustness
- calibration under distribution shift
- identity leakage and pair-aware sampling
- hard-negative mining and curricula
- deployment-safe low-FP decision systems
- compact-model robustness under post-processing

This file ties each literature point back to a live repo uncertainty.

## 2. Video-call and post-processing robustness

### `Deepfake Detection In Video Conferencing Scenarios` (VCF, 2025)

Why it matters:

- it directly treats conferencing pipelines as a distinct evaluation regime instead of assuming generic deepfake benchmarks are enough
- that is aligned with this repo's problem statement more tightly than most deepfake papers

What it reinforces here:

- Teams-like transport is not just "extra compression"
- evaluation and training need a dedicated conferencing lane
- current use of a frozen Teams suite is justified

What it does **not** solve for this repo:

- it does not replace the missing enhanced-through-Teams lane
- it does not answer the current low-FP selection problem by itself

### `Reduced Spatial Dependency for More General Video-level Deepfake Detection` (2024)

Why it matters:

- the repo already suspects crop and spatial sensitivity
- current direct Teams training rows receive almost no spatial perturbation

What it suggests here:

- detectors that over-rely on fixed crop-local cues can lose generalization under conferencing crops, resizes, and framing shifts
- crop-robust evaluation and light target-branch spatial augmentation are worth testing before larger architecture churn

### `Towards More General Video-based Deepfake Detection through Facial Component Guided Adaptation for Foundation Model` (CVPR 2025)

Why it matters:

- if backbone questions are revisited later, component-aware adaptation is a more plausible direction than a blind "bigger is better" move

What it suggests here:

- structured facial-component adaptation may improve robustness under post-processing and partial cue corruption
- but this is a later-stage backbone question, not the next move while data and decision issues remain unresolved

## 3. Calibration under distribution shift

### `Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift`

Why it matters:

- this is one of the cleanest general warnings that confidence quality degrades under shift

What it suggests here:

- a threshold tuned on one population will not stay trustworthy by default on Teams target data
- post-hoc uncertainty and calibration need to be validated on the actual target lane, not assumed

### `Frustratingly Easy Uncertainty Estimation for Distribution Shift` (2021)

Why it matters:

- it argues that surprisingly simple uncertainty signals remain useful baselines under shift

What it suggests here:

- before building a complex uncertainty head, the repo should test simple score-margin, energy, or disagreement gates on the frozen shortlist

### Recent deepfake-specific calibration work, including `A Study of Calibration for Deepfake Video Detection` (2025)

Why it matters:

- calibration is not just a generic classification afterthought for deepfakes; ambiguous and transitional samples matter disproportionately

What it suggests here:

- low-FP deployment should treat calibration as part of the research target
- soft-label or ambiguity-aware training may eventually be useful, but the next actionable step is still per-checkpoint target-domain calibration on current frozen checkpoints

## 4. Identity leakage and pair-aware sampling

### `Implicit Identity Leakage: The Stumbling Block to Improving Deepfake Detection` (CVPR 2023)

Why it matters:

- this is the strongest literature match to the repo's current sampler problem

What it reinforces here:

- identity can dominate measured improvements if split and pairing design are weak
- a model can look better because it learns identity shortcuts or identity-correlated artifacts rather than the manipulation signal

Repo-specific connection:

- Round 2 sampler analysis shows that the meaningful family competition lives on a tiny overlap set of identities
- that means pair-aware and per-identity-per-source sampling design is not an academic extra here; it is central

## 5. Hard-negative mining and curricula

### `Representative Forgery Mining for Fake Face Detection` (CVPR 2021)

Why it matters:

- it argues against naive "more fake data is always better" thinking

What it suggests here:

- late-stage selection of the right hard forgeries can matter more than simply increasing merged-source row count
- this matches the current repo situation, where raw VTE size hides the fact that true Teams-companion exposure is tiny

### Reward- or curriculum-based sample-weighting ideas such as `TSRL` (2026 preprint)

Why it matters:

- the field is moving toward adaptive sample weighting rather than static family weights

What it suggests here:

- learned or feedback-driven curricula are plausible future directions
- but for this repo they are still lower priority than explicit source splitting and simple sampler redesign

## 6. Deployment-safe low-FP strategies

### `SelectiveNet: A Deep Neural Network with an Integrated Reject Option` (ICML 2019)

Why it matters:

- it gives a principled framing for abstention when mistakes are asymmetric

What it suggests here:

- if false positives on real Teams participants are expensive, a narrow abstain band is not a hack; it is a valid system design choice

### Selective classification literature more broadly

Why it matters:

- reject-option systems often outperform forced binary decisions when the deployment cost of one error class is much higher

What it suggests here:

- the repo should be willing to test uncertain / abstain policies instead of insisting that every window must produce a hard binary label

## 7. Compact models and robustness after post-processing

### `FE-UNet: Lightweight and Efficient Deepfake Detection with Quantized Vision State Space Models` (IJCAI 2025)

Why it matters:

- it is evidence that compact detectors are still an active research direction, not a dead end

What it suggests here:

- small-model improvement is still plausible without immediately abandoning the deployment budget
- but architecture choice only becomes the dominant question after the repo resolves data truth, target exposure, and low-FP selection

## 8. What the literature now makes more likely

- conferencing-specific robustness is a real subproblem, not just a relabelled generic deepfake problem
- calibration and selective prediction are first-class concerns under shift
- identity-aware sampling design matters more than static weight narratives admit
- hard-negative curricula are more plausible than more merged-row count alone

## 9. What the literature does **not** yet justify

- jumping straight to a larger model without first fixing evaluation and data truth
- assuming more generic compression augmentation will substitute for true conferencing-domain examples
- trusting uncertainty scores without target-domain validation
- treating a merged source name as evidence that the missing condition is present

## 10. Status

- Established:
  - the field supports conferencing-domain evaluation, identity-leakage caution, and shift-aware calibration
- Plausible:
  - selective prediction and targeted curricula can give safer deployment behavior here
- Still unknown:
  - whether a compact architecture change will beat better target-domain data and selection discipline in this specific repo
