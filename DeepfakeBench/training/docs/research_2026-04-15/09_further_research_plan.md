# Further Research Plan

## What We Now Know

- The current project is not blocked by lack of ideas. It is blocked by lack of clean truth on which mechanism is actually responsible for the current tradeoffs.
- The strongest structural finding so far is that the current `visomaster_teams_enhanced` training source is mostly `clean_fallback`, and the loader never emits fake frames that are both enhanced and Teams-processed. The project is therefore still under-training the exact failure mode that matters most.
- The current identity-balanced sampler makes many family weights much less powerful than they appear in YAML. Most identities do not present real cross-family competition, so more small weight edits are unlikely to materially change the training diet.
- `R12_G` remains the strongest established B/16 generic-robustness branch, while Track A is best understood as a branch that improved some target fake slices at the cost of worse real Teams false-positive behavior.
- Real Teams FPR regressions should be treated as more trustworthy than enhanced-fake gains, because the real-side frozen evaluation has stronger lockbox support than the enhanced-fake side.
- The current augmentation story is still partly untrusted. There is credible evidence that intended `GammaUp` settings in recent R13 configs may not have been active because of key mismatch.
- The repo still mixes incompatible selection objectives: holdout AUC during training, optional OOD composite for checkpointing, then fixed-threshold scorecards in arena. That makes "best model" claims less reliable than they appear.
- Low-FP operation is not only a model-training issue. It is also a calibration, thresholding, temporal aggregation, and uncertainty-handling problem.
- The Effort whitepaper is still useful as a bias toward constrained residual adaptation on top of a pretrained semantic backbone, but it does not answer the Teams-specific questions that now dominate the project.

## What Is Still Uncertain

- Which current candidate is actually best on the real deployment objective once judged on one frozen lane with one threshold policy. The shortlist now includes at least `R12_G`, best Track A checkpoint, `R13_E`, and `R13_FT10`, and the repo still does not contain a final apples-to-apples answer.
- Whether the enhanced-through-Teams weakness is mostly a missing-data problem, an augmentation/invariance problem, or a combined problem.
- How much real target-domain capture is worth relative to synthetic Teams-style augmentation for this exact detector and target domain.
- Whether the real Teams FPR regressions are mainly driven by lighting/spatial fragility, by fake-side sharpening of the decision boundary, or by checkpoint-selection mismatch.
- Whether current frame instability is mostly reducible by training changes, or whether most of the near-term gain is actually in the decision layer through calibration, temporal smoothing, and disagreement gating.
- How much the smaller `ViT-B/16` backbone is a real limiting factor after data truth and selection truth are fixed. Capacity may matter, but it is still not the cleanest explanation for the current failures.
- Whether the current frozen target-domain suites are sufficient to support final promotion decisions on enhanced fake generalization, given the weak lockbox coverage on those slices.

## Priority Research Tracks For Round 2

### Track 1: Freeze The Evaluation And Selection Contract

This is the highest-value track because all later conclusions depend on it. Round 2 should first determine how checkpoints are compared, how thresholds are chosen, which suites are authoritative, and which metrics are allowed to drive promotion decisions.

The output of this track should not be another model recommendation. It should be a written evaluation contract that all later research follows.

### Track 2: Reconstruct The True Target-Domain Data Story

This track should focus on data truth, not training yet. It should map exactly which samples exist for:

- real Teams reals
- Teams passthrough fakes
- enhanced clean fakes
- enhanced-through-Teams fakes
- merged-source fallback rows
- external webcam-like reals

The key goal is to stop reasoning from source names and start reasoning from emitted training conditions.

### Track 3: Determine Whether Sampler And Curriculum Changes Can Materially Alter The Effective Diet

This track should answer whether curriculum and source splitting are truly high-leverage, or whether the current pool structure is so constrained that only new data can move the needle.

The research focus should be on effective exposure per epoch, identity competition structure, and late-stage specialization options, not on more nominal weight proposals.

### Track 4: Verify Nuisance-Invariance Hypotheses Cleanly

This track should revisit lighting, spatial variance, crop sensitivity, and Teams-like perturbation robustness, but only after the config path is made trustworthy enough to know which augmentations were truly active.

The goal is not to recommend one augmentation recipe yet. The goal is to decide which nuisance factors are still first-order blockers and which are mostly narrative.

### Track 5: Separate Model Weakness From Decision-System Weakness

This track should study calibration, threshold drift, temporal smoothing, hysteresis, and perturbation disagreement using frozen checkpoints only.

It is important that this remain a research track, not an implementation rush. The project needs evidence on how much low-FP gain is available without retraining before it decides where to spend the next training budget.

### Track 6: Capacity And Representation Ceiling, But Only As A Late Track

This track should stay lower priority unless earlier tracks show that data truth, selection truth, and nuisance robustness are already reasonably controlled.

The research question here is narrower than "should we use a bigger model." It is whether the current B/16 lane is still failing after structural issues are corrected, and if so, which capacity change would actually be justified.

## Questions Each Track Must Answer

### Track 1: Freeze The Evaluation And Selection Contract

- Which exact checkpoints belong in the final shortlist for fair comparison?
- Which suites are mandatory for promotion: Teams real dev, Teams real lockbox, Teams fake dev, enhanced fake dev, generic OOD, or some subset?
- What threshold policy is allowed for comparison: fixed `0.5`, in-dist EER, target-domain calibrated threshold, or multiple reported operating points?
- What metric ranking is authoritative for deployment: low-FP Teams score, OOD composite, holdout AUC, or a new explicit weighted scorecard?
- Which current historical claims become invalid once the new contract is enforced?

### Track 2: Reconstruct The True Target-Domain Data Story

- How many true enhanced-through-Teams fake examples actually exist today, by identity, method, and quality bucket?
- How much of each nominal Teams-oriented source is true Teams signal versus clean fallback or cross-domain pairing?
- Which external real lanes are truly present and discoverable versus only configured?
- Which target-domain slices have enough support to drive training decisions, and which are still too small to trust?
- Where are the critical holes: enhanced-through-Teams, poor lighting real Teams, method-specific Teams captures, or real webcam diversity?

### Track 3: Determine Whether Sampler And Curriculum Changes Can Materially Alter The Effective Diet

- If `visomaster_teams_enhanced` is split into truth-based subgroups, how much would the effective epoch-level diet actually change?
- Would a staged curriculum change exposure enough to matter, or would most identities still force essentially the same diet?
- How much late-stage DF40 and clean VisoMaster exposure can be removed before generic robustness collapses?
- Which hard slices are currently under-exposed enough to justify their own late lesson?
- Is pair-aware or source-aware sampling missing a material advantage beyond the current identity balancing?

### Track 4: Verify Nuisance-Invariance Hypotheses Cleanly

- Were the intended R13 lighting augmentations actually inactive, and if so, which prior conclusions are no longer reliable?
- How sensitive are current candidate checkpoints to crop jitter, detector-box shifts, gamma/brightness change, sharpening, blur, and Teams-like rescale/compression changes?
- Which nuisance factors correlate most strongly with real Teams false positives?
- Are enhanced-fake failures explained more by Teams transport effects, by lighting/context variation, or by both interacting?
- Which augmentation ideas deserve another round because they are both mechanistically plausible and currently under-tested?

### Track 5: Separate Model Weakness From Decision-System Weakness

- How much can Teams real FPR be reduced using only calibration, temporal aggregation, hysteresis, and disagreement gating on frozen checkpoints?
- Which checkpoint benefits most from a decision-system upgrade, and does that change the model ranking?
- How unstable are scores on near-identical adjacent frames and on small crop perturbations once evaluated systematically?
- Is there a practical abstain band that meaningfully lowers false positives without destroying fake recall?
- Does a decision-layer improvement reduce the urgency of some training-side stability work, or only mask it?

### Track 6: Capacity And Representation Ceiling, But Only As A Late Track

- After Tracks 1 through 5 are clarified, which failures remain that look like real representation limits rather than data or decision issues?
- Is the problem lack of total backbone capacity, lack of adaptation capacity, or mismatch between current backbone prior and Teams-style nuisance variation?
- Would a larger or different backbone likely improve the exact low-FP Teams objective, or only generic benchmark metrics?
- What evidence would justify spending time on architecture change instead of more target-domain data work?

## Artifacts Or Data We Should Gather Before Round 2

- One frozen comparison bundle for the shortlist checkpoints: checkpoint IDs, exact training configs, effective runtime config dumps, and the exact metric lane used to pick each checkpoint.
- A target-domain manifest audit that explicitly labels each sample or paired object as:
  - true Teams real
  - true Teams original fake
  - true enhanced-through-Teams fake
  - clean enhanced fake
  - clean fallback companion
  - other
- Per-source and per-subgroup counts by identity, method, split, and expected epoch-level exposure under the current sampler.
- A gap table for target-domain coverage showing where lockbox exists and where only dev exists, especially for enhanced fake slices.
- A reproducible augmentation audit for the recent candidate configs, including which keys actually propagated into the active transform preset.
- A matched-frame or near-identical-frame instability set for frozen checkpoints, built specifically to measure score variance under:
  - adjacent frames
  - small crop shifts
  - lighting changes
  - mild Teams-like perturbations
- A calibration study bundle on the frozen target-domain suites, including raw scores, calibrated scores, and threshold sweeps.
- A data-collection ledger for new captures in progress, with counts for real Teams, Teams fake, and enhanced-through-Teams by identity and method.

## What Would Change The Final Recommendation The Most

- Evidence that one current checkpoint already wins clearly on a frozen low-FP Teams contract after calibration and temporal decision logic. That could downgrade the urgency of new training work.
- Evidence that a small amount of real enhanced-through-Teams data causes a large jump relative to much larger synthetic-augmentation substitutes. That would strongly favor more capture over augmentation research.
- Evidence that current real Teams FPR problems fall sharply once lighting/spatial invariance is fixed with verified-active augmentations. That would raise augmentation/invariance work above data restructuring.
- Evidence that the shortlist checkpoints remain poor even after decision-layer improvements and data-truth corrections. That would make capacity or representation changes more credible.
- Evidence that the current enhanced-fake gains do not survive stronger lockbox-style evaluation. That would significantly reduce confidence in the Track A direction.
- Evidence that source splitting and curriculum materially change the effective diet under the current identity sampler. That would justify making data-organization work a primary engineering priority before collecting much more data.

## How Round 2 Should Be Structured

Round 2 should not start as another training sprint. It should start as a short research sprint with explicit gates.

### Phase 1: Measurement Contract

- Finalize the checkpoint shortlist.
- Finalize the frozen evaluation suites and threshold policy.
- Produce one written comparison contract and use it for every later claim.

### Phase 2: Data Truth

- Audit the target-domain sources and merged-source branches.
- Quantify actual target-condition coverage and missing slices.
- Decide whether source splitting is mandatory before further interpretation.

### Phase 3: Mechanism Separation

- Run frozen-checkpoint studies for calibration, temporal smoothing, disagreement, and instability.
- In parallel, finish the augmentation propagation audit and nuisance-sensitivity analysis.
- Only after this decide whether the next bottleneck is mainly data realism, nuisance invariance, or decision policy.

### Phase 4: Round-3 Candidate Definition

- Use the evidence from Phases 1 through 3 to define the next actual experiment set.
- Limit that set to the few experiments that answer the most expensive open questions, rather than attempting to improve everything at once.

The main purpose of Round 2 is to make the next recommendations harder to fool. Right now the project has enough signals to act, but still too many ways to tell itself the wrong story.
