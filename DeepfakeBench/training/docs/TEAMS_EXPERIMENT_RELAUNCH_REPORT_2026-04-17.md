# Teams Experiment Relaunch Report And Parallel Worktree Coordination

**Date:** April 17, 2026

## Purpose

This report is a sober relaunch memo for the next experiment wave after the
April 17, 2026 VisoMaster bad-data correction.

It is intentionally **not** an experiment-writing brief. The goal is to decide
which directions still deserve attention, which ones must be re-examined under
the corrected data interpretation, and which new questions become important now
that we are separating weak-signal VisoMaster hints from truly target-domain
data.

The working use case is:

- launch some work **today** without new target-domain data
- launch some work **tomorrow** once new proper data begins to exist

## How To Use This Document

This file is now the top-level coordination note for the post-April-17 Teams
relaunch state.

Use it to:

- classify work by problem type before writing experiments
- decide whether a task belongs to `today / current data` or `tomorrow / proper data`
- assign non-overlapping worktree ownership
- find the deeper supporting document before touching code

If this file and an older pre-April-17 memo disagree, this file should be
treated as the current coordination reference unless a newer document supersedes
it.

## Related Documents

These are the main supporting files that workers should read before editing.

- [`VISOMASTER_BAD_DATA_POLICY_PLAN_IMPACT_2026-04-17.md`](VISOMASTER_BAD_DATA_POLICY_PLAN_IMPACT_2026-04-17.md)
  - planning impact of the April 17 policy
  - weak-signal ablation framing
  - current reference state for hint-lane interpretation
- [`research_2026-04-15_round2/02_target_domain_data_truth.md`](research_2026-04-15_round2/02_target_domain_data_truth.md)
  - what the current loader and frozen eval manifest actually contain
  - proof that the repo still does not emit an actual enhanced-through-Teams fake lane
- [`research_2026-04-15/05_target_domain_gap_and_teams_enhanced_structural_findings.md`](research_2026-04-15/05_target_domain_gap_and_teams_enhanced_structural_findings.md)
  - why Track A improved fake recall but regressed real Teams false positives
  - current bottleneck ranking
  - structural limits of `visomaster_teams_enhanced`
- [`research_2026-04-15_round2/01_evaluation_contract_and_shortlist.md`](research_2026-04-15_round2/01_evaluation_contract_and_shortlist.md)
  - calibrated low-FP promotion contract
  - current shortlist
  - current mismatch between training selection and deployment selection
- [`TRACK_C_SCORECARD_RUNBOOK_2026-04-07.md`](TRACK_C_SCORECARD_RUNBOOK_2026-04-07.md)
  - current compact scorecard path
  - latest fixed-threshold Track A vs `R12_G` readout
- [`research_2026-04-15/04_data_composition_and_curriculum_opportunities.md`](research_2026-04-15/04_data_composition_and_curriculum_opportunities.md)
  - actual epoch-level exposure under the current identity-balanced sampler
  - why raw row counts and YAML weights can be misleading
- [`research_2026-04-15_round2/04_nuisance_invariance_and_augmentation_truth.md`](research_2026-04-15_round2/04_nuisance_invariance_and_augmentation_truth.md)
  - augmentation runtime truth
  - `GammaUp` key mismatch and current nuisance-invariance evidence
- [`SPATIAL_AUGMENTATION_PROPOSAL_2026-04-14.md`](SPATIAL_AUGMENTATION_PROPOSAL_2026-04-14.md)
  - current spatial-jitter proposal
  - strongest concrete recommendation for crop / face-geometry robustness
- [`TEAMS_AUGMENTATION_SIDE_TRACK_HANDOFF_2026-04-06.md`](TEAMS_AUGMENTATION_SIDE_TRACK_HANDOFF_2026-04-06.md)
  - current Teams-simulation side-track state
  - why old single-mode `TeamsCodecSimulation` should stay off the main line
- [`research_2026-04-15_round2/05_decision_system_low_fp_analysis.md`](research_2026-04-15_round2/05_decision_system_low_fp_analysis.md)
  - threshold, temporal aggregation, hysteresis, and abstain-band analysis
- [`research_2026-04-15_round3/04_internal_instability_mitigation_summary_for_parallel_review.md`](research_2026-04-15_round3/04_internal_instability_mitigation_summary_for_parallel_review.md)
  - current instability evidence
  - why decision-layer and stability work should stay explicit and measured
- [`../arena/score_teams_promotion_contract.py`](../arena/score_teams_promotion_contract.py)
  - implementation of the calibrated Teams promotion contract scorer
- [`../experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml`](../experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml)
  - current active Track A baseline config
- [`../data/sources/combined_paired.py`](../data/sources/combined_paired.py) and [`../data/sources/visomaster.py`](../data/sources/visomaster.py)
  - source-of-truth loader behavior for what training can actually emit

## Operating Assumptions

These assumptions should be treated as the current ground truth for planning.

- The April 17, 2026 VisoMaster bad-data policy is now the reference
  interpretation for historical bad `visomaster_*` data.
- Historical bad `visomaster_*` data is no longer method-faithful supervision.
- The retained weak-signal pool is:
  - `480` sample IDs as `visomaster hints`
  - `202` sample IDs as `visomaster hints (teams)`
- `4904` bad sample IDs are ignored and `3` are delete-only.
- We currently do **not** have any VisoMaster lane that should be described as
  the exact target domain.
- Older runs therefore remain useful only as **old-composition evidence** unless
  they are rebuilt under the corrected data interpretation.

The most important planning consequence is simple:

**There is currently no honest basis for saying that we already have precise
target-domain VisoMaster training data.**

That changes how current experiments should be framed:

- today’s work is mostly about decision-layer leverage, nuisance robustness, and
  corrected-composition truth
- tomorrow’s work can begin to test true target-domain realism, but only if the
  new data is kept explicit and provenance-clean from day one

## What Still Stands From The Upgrade Work

The April 17 correction invalidates parts of the old composition story, but it
does **not** invalidate the core deployment lessons.

### 1. The deployment objective is still the same

- Real Teams false positives remain the primary safety gate.
- Promotion should still be judged on the frozen Teams lane, not on generic
  holdout AUC or mixed-source OOD composites.
- The calibrated low-FP promotion contract still makes more sense than fixed
  threshold `0.5` comparisons.

### 2. `enhanced + Teams` is still a distinct mode

- The old structural finding still holds: enhanced fakes after Teams processing
  are not equivalent to clean enhanced fakes plus generic call degradation.
- The current loader/runtime truth still does not emit an actual
  enhanced-through-Teams fake lane.
- That means the main realism gap remains a real bottleneck until proper data
  arrives.

### 3. The merged `visomaster_teams_enhanced` source is still not an honest single family

- Even before April 17, the merged source was structurally diluted.
- After April 17, that caution is even stronger because the Teams-connected
  subset is itself weak-signal bad VisoMaster-through-Teams content.
- This source can still be diagnostically useful, but it should not be treated
  as a clean stand-in for the target condition.

### 4. Stability and nuisance robustness still matter

- Near-identical-frame instability is still a real failure mode.
- Generic stability regularization has already produced a negative result in the
  repo and should not be treated as an open default answer.
- Lighting, crop sensitivity, score calibration, temporal aggregation, and
  threshold policy all remain relevant.

### 5. More blind family-weight tuning is still low-value

- The old conclusion still holds that small family-weight sweeps are weaker than
  they look under the current sampler.
- The April 17 correction only makes old weighting narratives less trustworthy,
  not more.

## What Must Be Re-Examined Before Interpreting New Runs

These are not optional housekeeping items. They directly determine whether a new
run is even interpretable.

### 1. Whether training actually consumes the April 17 policy

The viewer and MCP surface now expose the corrected interpretation, but the
checked-in training path should not be assumed to use it until proven.

Planning implication:

- do not describe a new training run as "policy-corrected" unless the training
  loader path is shown to consume the manifest or an equivalent source rewrite

### 2. What the current R13 configs mean after the correction

Any config still built around:

- `combined_paired.visomaster`
- `combined_paired.teams`
- `combined_paired.visomaster_teams_enhanced`

needs to be re-read under the new semantics.

Planning implication:

- new runs on these configs can still be useful, but only if they are described
  as weak-signal or old-lane experiments rather than exact target-domain
  experiments

### 3. The meaning of the direct Teams lane

The direct Teams lane is still relevant, but it is no longer cleanly equal to
"true target-domain non-VisoMaster Teams data."

Planning implication:

- direct Teams results should now be read as a mix of:
  - real target-domain Teams signal
  - weak-signal bad VisoMaster-through-Teams signal

### 4. Any per-method VisoMaster reasoning

Historical bad VisoMaster rows should no longer be used to support clean
per-method conclusions about `GhostFace`, `Inswapper`, or related families.

Planning implication:

- stop using the old bad lane to justify per-method balancing or method-specific
  generalization claims

## Current Data Setup: How It Can Be Used Right Now

The corrected April 17 setup does still support useful work, but only if each
lane is described honestly.

### For retraining right now

- `visomaster hints` can be used only as weak-signal auxiliary data.
- `visomaster hints (teams)` can be used only as weak-signal auxiliary
  Teams-processed data.
- the direct Teams lane is still usable as deployment-relevant data, but it
  should now be understood as a mixed lane rather than a perfectly clean one.
- `visomaster_teams_enhanced` can still be used for structural diagnostics or
  controlled ablations, but it should not be presented as an exact
  target-domain lane.

### For reevaluation right now

- the frozen Teams suite remains the right reevaluation base
- real Teams lockbox remains the primary safety signal
- calibrated low-FP scoring remains more informative than raw threshold `0.5`
- mixed-source mega-eval remains diagnostic only

### For interpretation right now

- if a run uses only the retained hint lanes under corrected semantics, it
  should be called a `weak-signal corrected-data run`
- if a run still depends on old-lane semantics, it should be called an
  `old-semantics reference run`
- neither should be described as a true target-domain VisoMaster experiment

## Category Taxonomy For This Relaunch

The categories below are **problem types**, not worktree ownership. A single
worktree may own one or more categories, but these categories are the shared
vocabulary for planning and reporting.

### 1. Data Truth / Lane Semantics

Question:

- What do current training and evaluation lanes actually mean after the April 17
  policy?

In scope:

- policy-aware training path or lack of it
- explicit hint-lane semantics versus old `visomaster` semantics
- contamination accounting inside direct Teams and merged VTE lanes
- corrected composition truth for promotion-relevant configs

This category should produce:

- a reproducible per-lane composition packet
- a yes/no answer on whether a config is `policy-corrected`, `weak-signal-only`,
  or `old-semantics`

### 2. New Proper-Data Provenance / Lane Design

Question:

- When new proper VisoMaster and Teams data arrive, how must they be represented
  so they stay honest and useful?

In scope:

- provenance schema
- clean lane boundaries
- split / lockbox hygiene by base capture or session
- future train/eval slice naming for exact target conditions

This category should produce:

- the canonical future lane schema for proper clean data
- rules for how clean and Teams-parallel versions of the same base capture are linked

### 3. Weak-Signal Training Ablations

Question:

- Do the retained weak-signal hint lanes help, hurt, or add noise once they are
  treated honestly?

In scope:

- `no bad VisoMaster lane`
- `visomaster hints only`
- `visomaster hints + visomaster hints (teams)`

This category should produce:

- one clean three-arm ablation family under corrected interpretation
- no claims about exact target-domain realism

### 4. Augmentation / Nuisance Invariance

Question:

- Can truthful nuisance robustness improve real Teams behavior without pretending
  to replace missing target-domain data?

In scope:

- augmentation runtime truth
- `GammaUp` plumbing
- lighting robustness
- crop / face-geometry robustness
- Teams-simulation sidecars kept separate from the main line

This category should produce:

- one narrow truthful nuisance-sidecar family
- evidence about whether real Teams invariance improves

### 5. Stability / Decision-Layer Behavior

Question:

- How much can deployment behavior improve without new training?

In scope:

- calibrated threshold sweeps
- temporal aggregation
- hysteresis
- narrow gate / abstain-band policy
- near-identical-frame and crop-jitter score variance

This category should produce:

- a checkpoint ranking under the actual low-FP contract
- a decision on whether decision-layer leverage is strong enough to justify immediate deployment-side work

### 6. Evaluation Contract / Measurement

Question:

- What should count as "better" for this repo stage, and how is that different
  from current default behavior?

In scope:

- calibrated promotion contract
- frozen Teams suite usage
- real lockbox safety ordering
- fake lockbox inclusion
- fake-side provenance visibility

This category should produce:

- one authoritative promotion path
- one clear rule for what is diagnostic only versus promotion-authoritative

### 7. Structural Lane / Repo Issues

Question:

- Which lane names or repo behaviors are misleading even before new experiments
  start?

In scope:

- misleading lane names
- lanes whose emitted condition does not match their label
- missing exact-condition train/eval lanes
- sampler/runtime mismatches that make experiment names overstate what changed

This category should produce:

- a prioritized fix list with proof and direct repair path

## Lane Glossary And Current Mismatches

For this report, a `lane` means one explicit source or exact condition family
used in training or evaluation.

- `clean lane`
  - provenance-clean and method-faithful
- `weak-signal lane`
  - retained historical bad data with limited supervision value
- `mixed lane`
  - lane known to contain more than one semantically distinct condition
- `target-condition lane`
  - the exact deployment-relevant condition we ultimately care about

Current high-value lane mismatches:

| Lane | Current reality | Why the name is misleading | What would make it right |
| --- | --- | --- | --- |
| `visomaster` | still appears in active training YAMLs as a normal source | the April 17 policy says historical bad `visomaster_*` rows should be treated as weak-signal hints, not normal method-faithful supervision | replace or explicitly split into `visomaster_hints`, `visomaster_hints_teams`, and a future proper clean VisoMaster lane |
| `teams` | still deployment-relevant, but no longer a monolithic pure Teams lane | part of it is weak-signal bad-VisoMaster-through-Teams content | preserve or expose provenance so Teams-clean and Teams-hint contributions can be distinguished |
| `visomaster_teams_enhanced` | mostly `clean_fallback`; true Teams-companion subset is tiny; enhancer branches still load clean enhanced fake frames | the name sounds like an exact enhanced-through-Teams lane, but the loader never emits that exact fake condition | split by companion domain and branch, or stop using it as a target-labeled lane |
| `enhanced-through-Teams` training lane | structurally absent right now | it is the central unresolved target condition, but current training does not emit it | collect and package it as its own clean proper lane |
| `teams_fake_all_*` eval slices | useful operationally, but fake-side provenance is still coarse | fake-side wins are hard to attribute cleanly between weak-signal Teams-played VisoMaster and other fake families | keep this deployment slice, but add richer proper-data sub-slices when new data arrives |
| `external_training_reals` | configured in some YAMLs, but locally absent in current discovery truth | "realboost" narratives can overread runtime truth if the lane is not actually loaded | verify remote runtime truth or stop treating it as active leverage until proven |

Proof for these mismatches lives primarily in:

- [`VISOMASTER_BAD_DATA_POLICY_PLAN_IMPACT_2026-04-17.md`](VISOMASTER_BAD_DATA_POLICY_PLAN_IMPACT_2026-04-17.md)
- [`research_2026-04-15_round2/02_target_domain_data_truth.md`](research_2026-04-15_round2/02_target_domain_data_truth.md)
- [`research_2026-04-15/05_target_domain_gap_and_teams_enhanced_structural_findings.md`](research_2026-04-15/05_target_domain_gap_and_teams_enhanced_structural_findings.md)
- [`research_2026-04-15/04_data_composition_and_curriculum_opportunities.md`](research_2026-04-15/04_data_composition_and_curriculum_opportunities.md)

## Measuring Data Truth / Composition

This category is not measured by AUC. It is measured by whether the lane names,
loader behavior, and effective sampler exposure all describe the same thing.

Minimum artifact set that any composition-truth worker should produce for each
promotion-relevant config:

1. `lane_counts_by_split`
   - paired-object counts per lane for train / val / test
2. `lane_contamination_accounting`
   - how much of each supposed lane is actually another condition
3. `effective_epoch_exposure`
   - expected selected objects per epoch under the actual identity-balanced sampler
4. `loader_emission_truth`
   - what the loader can actually emit, not what the YAML names imply
5. `eval_coverage_truth`
   - which exact conditions exist in `dev` and `lockbox`

For this repo stage, the key measurements are:

- per-lane counts after the April 17 policy
- direct Teams contamination accounting
- `teams_v2_companion` versus `clean_fallback` inside merged VTE
- whether current training is policy-aware or only the viewer is
- whether the repo can emit:
  - `Teams real + Teams original fake`
  - `Teams real + Teams enhanced fake`
  - `clean real + clean enhanced fake`

Current hard fact:

- the repo can emit `Teams real + Teams original fake`
- the repo can emit `Teams real + clean enhanced fake`
- the repo does **not** currently emit `Teams real + Teams enhanced fake`

That is why composition truth is a first-class task rather than bookkeeping.

## Interpreting A Practical Proper-Data Package

The following kind of collection wave is directly relevant to Category 2 if it
is kept provenance-clean:

- about `300` high-quality real videos
- about `600` matching VisoMaster + enhancer fake videos in very high quality
- about `400` medium-quality **big-face** real videos where the face is closer before cropping
- about `1200` corresponding big-face fake videos with VisoMaster methods with and without enhancement
- some subset of the above passed through Teams, creating explicit parallel Teams versions

Planning interpretation:

- the high-value additions are **not all equal**
- the best additions are:
  - medium-quality big-face real/fake coverage
  - any clean-to-Teams parallel pairs
  - especially any **enhanced-through-Teams** subset
- high-quality clean real/fake data are still useful, but mainly as clean anchor lanes rather than as a substitute for the Teams target condition

Why the proposed big-face data matters:

- it directly targets a likely face-scale / crop-regime mismatch
- current cropped training faces are not dominated by close-call framing
- many real video-call examples do show closer faces than the current training mix

Important packaging consequence:

- think in terms of **derived conditions from the same base capture**, not only in terms of "2x" or "4x" identity counts
- under the current identity-balanced sampler, more variants per identity improve condition coverage, but do **not** translate linearly into sampler exposure

Recommended minimum clean lane schema for future proper data:

- `proper_real_clean`
- `proper_visomaster_clean`
- `proper_visomaster_enhanced_clean`
- `proper_real_teams`
- `proper_visomaster_teams`
- `proper_visomaster_enhanced_teams`

Recommended metadata to preserve from day one:

- `base_capture_id`
- `session_id`
- `identity_id`
- generator / method
- enhancement status
- playback path
- quality band
- face-scale band such as `big_face` versus ordinary framing

Packaging rules that should be treated as mandatory:

- split by base capture / session / identity **before** deriving clean versus Teams variants
- keep clean and Teams-parallel versions explicit as separate lanes
- do not hide proper data inside `visomaster hints`
- do not hide proper data inside the old merged `visomaster_teams_enhanced` convenience lane

## Evaluation Contract Clarification: What We Do Now Versus What We Should Do

The repo already has most of the pieces for the right contract, but workers
should be explicit about the difference between the current default path and the
desired deployment path.

| Topic | Current repo behavior | Desired behavior for promotion |
| --- | --- | --- |
| training-side checkpoint selection | primary checkpointing is still driven by `val_holdout` and the configured primary metric, usually AUC | keep this as training-monitoring only |
| auxiliary checkpoint lane | `ood_composite` checkpoints are already saved and are useful for shortlist reduction | keep as shortlist evidence, not the final promotion contract |
| scorecard threshold | current scorecard rows are reported at fixed threshold `0.5` | fit a checkpoint-specific threshold on dev and freeze it before lockbox |
| primary deployment gate | real Teams false positives are already treated as the key operational issue | keep `teams_real_all_lockbox` as the primary safety gate |
| fake lockbox | older docs often treated this as incomplete | the checked-in frozen suite now includes `teams_fake_all_lockbox`; use the current suite file plus the promotion-contract scorer as the authoritative implementation path |
| promotion implementation | often described through compact `0.5` scorecards | use [`../arena/score_teams_promotion_contract.py`](../arena/score_teams_promotion_contract.py) as the decisive promotion path |

Practical worker rule:

- do not describe a fixed-threshold `0.5` table as the final promotion contract
- do not describe mixed-source mega-eval as the promotion contract
- if a worker claims a winner, that worker should either:
  - run the calibrated promotion-contract scorer, or
  - explicitly say that the result is still provisional and fixed-threshold only

## Parallel Review Lanes Before Writing Experiments

The cleanest way to proceed is to split the thinking into a few bounded review
lanes. These can be done by separate reviewers if needed.

### Lane A: Decision And Stability Review

Question:

- How much of today’s real-Teams pain can be improved without new training?

Review focus:

- calibrated per-checkpoint threshold sweeps on the frozen shortlist
- temporal aggregation and hysteresis choices
- narrow gate or abstain-band options
- whether checkpoint ranking changes under the actual low-FP contract

Why this lane still matters:

- it does not depend on new target-domain VisoMaster data
- it directly addresses the repo’s strongest still-valid operational finding:
  threshold and decision policy are part of the model system

Output this lane should produce:

- a decision on whether there is enough decision-layer leverage to justify
  immediate no-new-data deployment-side experiments

### Lane B: Current Data Semantics And Training-Path Review

Question:

- If we launch training today, what data are we actually training on under the
  corrected April 17 understanding?

Review focus:

- whether the training loader is policy-aware or still old-lane
- corrected composition reports for the shortlist-relevant configs
- direct Teams contamination accounting
- meaning of `visomaster_teams_enhanced` after the correction

Why this lane still matters:

- without it, a "current-data retrain" cannot be interpreted honestly

Output this lane should produce:

- a yes/no answer on whether a same-day retrain can be labeled as
  policy-corrected, weak-signal-only, or old-semantics-only

### Lane C: Nuisance And Runtime-Truth Review

Question:

- Which stability and invariance interventions are still worth testing, and
  which ones are already ruled out or not even active?

Review focus:

- augmentation runtime truth rather than YAML intent
- `GammaUp` plumbing status
- direct Teams passthrough augmentation weakness
- whether a minimal light/spatial nuisance ablation is still the right small
  training-side experiment family

Why this lane still matters:

- the repo still has real evidence for lighting and crop sensitivity
- but it also has evidence that some intended augmentations were never active

Output this lane should produce:

- the smallest set of truthful nuisance/stability experiment classes worth
  considering today on current data

### Lane D: New Proper Data Packaging And Evaluation Review

Question:

- If new VisoMaster and Teams data are actually target-domain faithful, how
  should they enter the system without repeating the current ambiguity?

Review focus:

- provenance schema
- lane boundaries
- split and lockbox hygiene
- evaluation slices for the exact target condition
- whether proper new data should replace, coexist with, or simply be isolated
  from the weak-signal hint lanes

Why this lane matters:

- tomorrow’s experiments become meaningful only if the new data enters cleanly

Output this lane should produce:

- the rules for how proper new data must be represented before a target-domain
  training experiment can be trusted

## Parallel Worktree Split

The review lanes above are conceptual. The worktree split below is the
**non-overlapping ownership model** for parallel execution.

Important rule:

- categories are shared vocabulary
- worktree tracks are file-ownership boundaries
- if two workers need the same file, they are not truly non-overlapping

### Non-Overlapping Worktree Tracks

#### `WT-A` Data Truth / Policy Integration

Owns:

- [`../data/sources/visomaster.py`](../data/sources/visomaster.py)
- [`../data/sources/combined_paired.py`](../data/sources/combined_paired.py)
- policy-aware source redesign docs
- corrected composition / contamination reporting artifacts

Avoid:

- `../arena/*`
- `../data/augmentations/*`
- active `phase2_round13` YAML edits unless this is the only owning worktree

Outputs:

- proof whether current training is policy-aware
- corrected per-lane counts and contamination accounting
- explicit plan for `visomaster_hints` and `visomaster_hints_teams`

Start state:

- can start immediately

#### `WT-B` Weak-Signal Ablation Configs

Owns:

- new experiment YAMLs only under `../experiments/phase2_round13/`
- new weak-signal runbooks / launch notes

Avoid:

- `../data/sources/*`
- `../arena/*`
- `../data/augmentations/*`

Outputs:

- one three-arm weak-signal ablation family:
  - `no hints`
  - `hints only`
  - `hints + teams hints`

Start state:

- draft-only until `WT-A` freezes lane names and semantics

#### `WT-C` Augmentation / Nuisance Invariance

Owns:

- `../data/augmentations/*`
- `../tests/test_lighting_transforms.py`
- `../tests/test_teams_adaptive_simulation.py`
- new sidecar YAMLs under fresh names

Avoid:

- `../data/sources/*`
- `../arena/*`
- active main-line Track A YAMLs

Outputs:

- runtime-truth confirmation for `GammaUp`
- one narrow light / spatial nuisance-sidecar family
- no silent edits to active main-line configs

Start state:

- can start immediately

#### `WT-D` Stability / Decision-System Analysis

Owns:

- new analysis scripts and docs for:
  - threshold sweeps
  - temporal aggregation
  - hysteresis
  - abstain-band or narrow-gate analysis
- stability-specific reports built on frozen checkpoints and reports

Avoid:

- `../data/sources/*`
- `../data/augmentations/*`
- `../arena/score_teams_promotion_contract.py`

Outputs:

- decision-layer comparison on the frozen shortlist
- frame-to-frame / crop-jitter stability evidence

Start state:

- can start immediately

#### `WT-E` Evaluation Contract / Promotion Tooling

Owns:

- [`../arena/run_target_domain_validation_sequential.py`](../arena/run_target_domain_validation_sequential.py)
- [`../arena/score_teams_promotion_contract.py`](../arena/score_teams_promotion_contract.py)
- `../arena/target_domain_suites*.yaml`
- checkpoint maps
- scorecard runbooks

Avoid:

- `../data/sources/*`
- `../data/augmentations/*`
- active training YAMLs unless explicitly reassigned

Outputs:

- authoritative calibrated promotion path
- decisive scorecard / contract artifacts
- current suite ownership, including fake-lockbox usage

Start state:

- can start immediately

#### `WT-F` New Proper-Data Schema / Future Manifests

Owns:

- new proper-data schema docs
- new manifest builders or templates for **future proper data only**
- capture / inventory / provenance docs

Avoid:

- current frozen suite files owned by `WT-E`
- current loader code owned by `WT-A`
- active training YAMLs owned by `WT-B` or other experiment tracks

Outputs:

- first-class clean lane schema for incoming data
- future eval slice specification for clean versus Teams-parallel conditions

Start state:

- can start immediately in spec mode, even before the data lands

### Overlapping Coordination Zones

The tracks above are file-non-overlapping. The topics below still overlap
conceptually and therefore need explicit coordination.

#### `WT-A` <-> `WT-B`

- `WT-B` cannot finalize ablation YAMLs until `WT-A` freezes the actual hint-lane semantics
- if `WT-A` changes lane names, `WT-B` must rebase rather than invent local aliases

#### `WT-A` <-> `WT-E`

- contamination accounting and evaluation slice naming must describe the same lane truth
- any promotion claim that ignores corrected lane semantics should be treated as provisional

#### `WT-C` <-> `WT-D`

- `WT-C` proposes nuisance / augmentation interventions
- `WT-D` judges whether those interventions actually improve stability or low-FP behavior
- `WT-D` should own the measurement rules; `WT-C` should not self-certify promotion claims

#### `WT-E` <-> `WT-F`

- future proper-data lane names and future eval slice names must be agreed together
- do not let future manifest builders invent names that the promotion contract cannot score

#### Shared Hot Files And Merge Hotspots

- this report file should have one owner at a time
- [`../experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml`](../experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml) should be treated as a baseline reference, not as a shared scratchpad
- if a track needs an experiment variant, create a **new YAML** instead of editing the active baseline in place
- [`../data/sources/combined_paired.py`](../data/sources/combined_paired.py) belongs to `WT-A`
- `../data/augmentations/*` belongs to `WT-C`
- [`../arena/run_target_domain_validation_sequential.py`](../arena/run_target_domain_validation_sequential.py) and [`../arena/score_teams_promotion_contract.py`](../arena/score_teams_promotion_contract.py) belong to `WT-E`

### Minimum Coordination Rules For Parallel Workers

- no worker should claim a target-domain win from old-semantics training
- no worker should claim a promotion winner from fixed-threshold `0.5` tables alone
- no worker should silently reinterpret `visomaster_teams_enhanced` as if it were an exact enhanced-through-Teams lane
- any worker touching active baseline files should say so explicitly in the handoff
- every track should link back to this report plus its deeper supporting document

## Experiments Worth Considering Today Without New Data

This section is intentionally narrow. The point is not to generate many
experiments. The point is to identify the few experiment families that remain
defensible **today**.

### 1. Decision-layer experiments on the frozen shortlist

Status:

- still clearly worth doing

Why:

- they use existing checkpoints
- they do not depend on the missing proper VisoMaster lane
- repo evidence already suggests threshold, aggregation, and gating can move
  Teams FPR materially

What this family should answer:

- whether the shortlist ordering changes under calibrated low-FP scoring
- whether temporal aggregation or a mild gate reduces real Teams false positives
  enough to matter operationally

What this family should **not** claim:

- that it solves the missing target-domain realism problem

### 2. One narrow nuisance-invariance training family

Status:

- still worth considering, but only after runtime-truth review

Why:

- lighting and crop robustness remain open
- current config intent and current runtime behavior are not the same thing
- this is still a better bet than another generic weight sweep

What this family should answer:

- whether truthful upward-brightness robustness or light spatial perturbation
  materially improves real Teams behavior

What this family should **not** claim:

- that it is a substitute for real enhanced-through-Teams data

### 3. Weak-signal current-data retrains only if they are described honestly

Status:

- conditionally worth considering

Why:

- the retained hint lanes may still carry some weak signal
- the Teams-played hint subset may still teach something about transport
  artifacts

What this family should answer:

- whether the retained hints still help once they are no longer treated as
  method-faithful supervision
- whether removing the bad VisoMaster mass improves or harms the tradeoff

What this family should **not** claim:

- that it is already testing the exact target domain

Practical framing:

- if launched today, these runs should be framed as
  `weak-signal ablations under corrected interpretation`, not as target-domain
  realism experiments

## Experiments That Should Wait For Tomorrow’s Proper Data

These are the experiment families that become much more important once new
proper data starts to exist.

### 1. High-purity target-domain fine-tune

This becomes the highest-value training family once the new data is genuinely
target-domain faithful.

What it would answer:

- whether the main bottleneck was realism of the fake condition rather than only
  nuisance robustness or threshold policy

Why it should wait:

- today we do not have an honest exact-target-domain VisoMaster lane

### 2. Clean replacement-lane versus weak-hint comparison

Once proper data exists, the right question is no longer "how much can we
squeeze from bad VisoMaster?" It becomes:

- how much does clean proper data outperform the weak-hint stopgap?

Why this matters:

- it tells us whether the hint lanes should remain auxiliary, become negligible,
  or be dropped entirely

### 3. Proper target-condition evaluation slices

New proper data should not only feed training. It should also create explicit
evaluation slices for:

- proper VisoMaster-through-Teams
- proper enhanced-through-Teams if available
- session-held-out or lockbox target-domain stress lanes

Why this matters:

- otherwise the training lane becomes more target-faithful but the evaluation
  contract still cannot see the exact condition

## Implications Of Future Proper VisoMaster And Teams Data

If future VisoMaster and Teams data are in fact proper and target-domain
faithful, several planning assumptions change immediately.

### 1. Proper data should be a first-class clean lane, not a relabel of hints

- do not fold proper data back into `visomaster hints`
- do not hide it inside the old merged `visomaster_teams_enhanced` convenience
  lane
- keep it explicit as its own provenance-clean source family

### 2. Proper data re-opens curriculum and weighting questions

Right now weighting is not the main story because the current data is structurally
mis-specified. Once proper target-domain data exists, weighting and curriculum
become more meaningful again.

### 3. Proper data should carry richer provenance from day one

At minimum, future data should preserve:

- generator / method identity
- enhancement status
- playback path
- whether the sample is direct Teams, enhanced-through-Teams, or another exact
  deployment-relevant condition
- session identity for split hygiene

### 4. Proper data changes how we interpret the weak-signal lanes

The retained hint lanes stop being a stopgap proxy for target-domain realism.
They become optional auxiliary data that must justify themselves against the
clean lane.

### 5. Proper data can finally test the central structural thesis directly

The repo’s strongest unresolved thesis is that the model is missing the exact
`enhanced fake after Teams processing` condition. Proper new data is the first
chance to test that directly instead of by proxy.

## What Not To Spend Time On Right Now

These directions remain low-priority under the corrected planning state.

- Do not treat current bad `visomaster_*` data as exact target-domain
  supervision.
- Do not launch another round of blind family-weight tuning as if the old
  composition story were still trustworthy.
- Do not describe `visomaster_teams_enhanced` as an honest target-domain lane.
- Do not claim that generic stability regularization is still an open default
  path. The repo already has a negative result there.
- Do not let today’s no-new-data experiments stand in for tomorrow’s proper-data
  realism experiments.

## Practical Relaunch Stance

The cleanest planning stance is:

### Today

- treat the strongest immediate work as:
  - decision-layer experiments on the frozen shortlist
  - a corrected-composition / training-path review
  - at most one narrow truthful nuisance-invariance experiment family
- treat any current-data retrain as a weak-signal or corrected-semantics
  experiment, not as a true target-domain experiment

### Tomorrow

- if proper new data is ready, treat it as a clean new lane
- use it to test one high-purity target-domain training direction
- compare it explicitly against the weak-hint stopgap rather than silently
  merging the two

## Bottom Line

The April 17 correction does **not** leave us with "nothing to do." It leaves us
with a clearer split between:

- what is still worth testing now without new data
- what must be re-labeled and re-interpreted under the corrected composition
- what only becomes meaningful once proper target-domain data exists

The right immediate posture is therefore:

- use today for decision-layer leverage, truthful nuisance review, and
  corrected-composition clarity
- use tomorrow’s proper data to finally test the missing target-domain realism
  hypothesis directly
