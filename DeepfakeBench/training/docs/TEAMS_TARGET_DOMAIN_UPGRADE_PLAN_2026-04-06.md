# Teams Target-Domain Upgrade Plan

**Date:** April 6, 2026  
**Context:** R12 is strong overall, but deployment is live Microsoft Teams inference on local video streams, not generic holdout classification. The next round should optimize for that reality.

## 1. Objective

Build a better post-R12 model for live Microsoft Teams calls.

Primary goal:
- Reduce false positives on real low-quality Teams video from varied webcams, lighting, and network conditions.

Secondary goals:
- Improve detection of `Deeplivecam` regular + enhanced.
- Improve detection of `VisoMaster` regular, with extra focus on `SimSwap512` and enhanced variants.
- Preserve acceptable DF40/general performance.
- Preserve accuracy after INT8 quantization.

## 2. What We Know

### 2.1 Current Best Recipe

`R12_G` is still the practical reference point:
- Scratch ViT-B/16 DataComp-XL.
- Identity-balanced weighted sampling.
- DF40 + DeepLive + VisoMaster + Teams + external VCD reals.
- Family-aware augmentation with `quality_targeted_family` and `vcd_targeted`.
- Strong generic holdout and strong but imperfect OOD behavior.

Important caveat:
- The current training/evaluation loop is still not fully aligned to the production target. `val_holdout` is useful, but it is not the deployment metric.

### 2.2 The Real Deployment Problem

Observed in production-like use:
- False positives on poor real video.
- Sensitivity to lighting / location / camera characteristics.
- Weakness on VisoMaster enhanced faces.
- Weakness on VisoMaster `SimSwap512`.

The repo history supports this diagnosis:
- Data and domain coverage consistently mattered more than clever losses or generic curriculum.
- Lighting and webcam-domain mismatch are already documented as real failure modes.
- The Teams channel itself changes images in structured ways that the current pipeline only partially models.

### 2.3 Data Sources That Matter

Main training/eval universe:
- `DF40` fake and real: useful as a broad negative anchor, but not the deployment domain.
- `DeepLive` clean bucket: important fake family coverage.
- `VisoMaster` clean bucket: important fake family coverage.
- `Teams-v2` bucket: the closest existing production-like paired source.
- `VCD` reals: useful webcam-domain real anchor.
- `teams-faces-data-test-2914-fake-4420-real-feb-28`: strong target-domain benchmark bucket.

### 2.4 Two Different Enhanced VisoMaster Buckets

There are two distinct enhanced buckets and they should not be treated as interchangeable:

1. `gs://visomaster-enhanced-face-cropped/`
- Older loader-compatible bucket.
- Stored per enhancer variant.
- Current code already supports it.
- It behaves like a clean enhanced source, not a Teams-native merged source.
- Current top-level sample count observed: `8788`.

2. `gs://enhanced-visomaster-cropped/`
- Newer bucket.
- Stored by base `sample_id`, with enhancer subfolders under `frames/{enhancer}/`.
- Explicitly claims linkage to the Teams-v2 companion bucket using `sample_id`.
- Current top-level sample count observed: `999`.
- This is the more relevant bucket for the deployment goal because the data was passed through Teams.

### 2.5 Linkage Is Real, But Must Be Audited

The linkage mechanism is valid in principle:
- Example verified by user and by repo-side inspection: `visomaster_CSCS_00007`.
- The enhanced manifest points to:
  - `real` and `fake` from the companion bucket via `sample_id`
  - `enhanced` variants from `enhanced-visomaster-cropped`

However, spot checks also showed a second reality:
- Some enhanced manifests point to Teams-v2 sample IDs that do **not** currently resolve in Teams-v2.
- Example IDs checked: `visomaster_CSCS_00029`, `visomaster_CSCS_00051`.
- Those IDs currently resolve in `live-deepfake-methods-real-and-fake-frames-cropped`, not in Teams-v2.

Working assumption:
- The join key is correct.
- The manifest metadata is not sufficient by itself.
- Companion resolution should be implemented by existence checks, not by trusting the manifest blindly.

This is important because it changes whether the new bucket is safe for training immediately.

### 2.6 The Current Sampler Has a Hidden Multiplicity Problem

The current `combined_paired` identity-balanced sampler selects **one sample object per identity per epoch**, then applies family weighting across those sample objects.

That matters for enhanced data:
- If one base sample becomes 8 separate enhanced sample objects, that identity gets 8 extra weighted chances to be served as enhanced.
- That is not the same as “add one more family with a weight.”
- It can quietly distort the effective training distribution.

Conclusion:
- The new Teams-enhanced VisoMaster source should not be integrated as “999 base samples -> 8,000+ independent paired samples” in the same style as the older clean enhanced bucket.
- It should be integrated as a **merged base sample** keyed by `sample_id`, with controlled fake-branch selection at iteration time.

### 2.7 TeamsCodecSimulation Is Useful, But Not Settled

Current state:
- `TeamsCodecSimulation` exists and is wired.
- The old dead-code wiring issue was already fixed after R9.
- The transform was validated against matched before/after Teams frame pairs.

Current implementation models:
- brightness increase,
- contrast increase,
- blur / smoothing,
- JPEG-like compression,
- bilateral deblocking,
- optional chroma blur.

What the repo already says:
- The first bucket-level assumptions were wrong.
- Matched-pair validation was necessary.
- Teams behavior is at least partly bimodal.
- A single fixed transform is only an approximation.

Important implication:
- “Re-enable TeamsCodecSimulation at `p=0.15`” is not a final decision.
- It is a placeholder hypothesis.
- We should investigate the Teams channel again using the newer before/after data and decide whether the right answer is:
  - one better global transform,
  - a two-mode mixture,
  - per-family application,
  - or even an offline synthetic “Teamsified” dataset for selected families.

## 3. Guiding Decisions

1. Optimize for the target domain first.
2. Keep DF40 as a guardrail, not the main target.
3. Treat Teams-native data as more valuable than clean enhanced data.
4. Do not let multi-enhancer multiplicity silently dominate the sampler.
5. Do not assume the current Teams simulator is optimal just because it already exists.
6. Test curriculum once, in a narrow and falsifiable form.

## 4. Success Criteria

Primary success:
- Lower false-positive rate on poor-quality real Teams data.

Secondary success:
- Better recall on:
  - `Deeplivecam` regular,
  - `Deeplivecam` enhanced,
  - `VisoMaster SimSwap512`,
  - `VisoMaster enhanced` macro and per-enhancer.

Guardrails:
- DF40 headline metric may drop a little, but not materially.
- Generic holdout AUC may dip slightly, but should remain close to R12.
- INT8 should retain the target-domain gain.

Recommended ranking order for candidates:
1. `teams_real_poor_quality` FPR
2. target fake macro recall
3. INT8 retention
4. overall Teams-real FPR
5. holdout AUC as tie-breaker only

## 5. Sequential Gates

These tasks should happen in order because downstream work depends on them.

### S0. Freeze Baseline

Goal:
- Freeze the exact `R12_G` baseline and one or two best post-R12 enhanced checkpoints on a shared scorecard.

Deliverables:
- Baseline evaluation report in FP32.
- Baseline evaluation report in INT8.
- Single agreed metric table for later comparisons.

### S1. Audit `enhanced-visomaster-cropped`

Goal:
- Turn the new enhanced bucket from “promising but ambiguous” into a training-safe source.

Required output:
- Resolver manifest with one row per `sample_id`.
- Resolution status:
  - `teams_v2_companion`
  - `clean_companion_only`
  - `missing_companion`
- Available enhancers per sample.
- Whether all 8 enhancers are present.
- Companion frame counts.

Hard rule:
- Training code must use the resolver manifest, not trust the raw manifest metadata.

### S2. Freeze Target-Domain Evaluation Splits

Goal:
- Create a stable `dev` / `lockbox` split for the Teams-like benchmark bucket.

Deliverables:
- Deterministic split by identity/video.
- Deterministic slices:
  - `teams_real_all`
  - `teams_real_poor_quality`
  - `teams_real_lighting_extreme`
  - `deeplive_regular`
  - `deeplive_enhanced`
  - `visomaster_original_macro`
  - `visomaster_simswap512`
  - `visomaster_enhanced_macro`
  - per-enhancer slices if possible

Only after `S1` and `S2` are done should model changes be merged and experiment configs be finalized.

## 6. Parallel Workstreams

These are designed for separate worktrees with minimal write overlap.

### Track A: New Teams-Enhanced Data Source

**Purpose**
- Integrate `enhanced-visomaster-cropped` in a way that matches the production use case and avoids sample explosion.

**Write scope**
- `DeepfakeBench/training/data/sources/visomaster.py`
- `DeepfakeBench/training/data/sources/combined_paired.py`
- tests for the new source

**Core design**
- Add a new config block, e.g. `combined_paired.visomaster_teams_enhanced`.
- Use `sample_id` as the base join key.
- Build merged base samples, not 8 independent enhanced sample objects.
- Each base sample contains:
  - real Teams frames
  - original fake Teams frames
  - available enhanced fake branches
- At iteration time choose exactly one fake branch:
  - original fake branch, or
  - one enhanced branch

**Recommended default branch policy**
- `p_original = 0.5`
- `p_enhanced = 0.5`
- `enhanced_choice = uniform`

**Why**
- This preserves per-enhancer diversity without multiplying the identity weight by 8.

**Acceptance**
- One base sample contributes one fake branch per selection.
- Identity leakage remains impossible.
- Resolver-manifest audit is enforced.

### Track B: Teams Channel Re-Characterization And Augmentation Redesign

**Purpose**
- Investigate the Teams transform more deeply using current before/after data, not just reuse the existing R9-era approximation.

**Write scope**
- `DeepfakeBench/training/data/augmentations/teams_simulation.py`
- `DeepfakeBench/training/tests/test_teams_simulation.py`
- a new analysis/report script or doc under `DeepfakeBench/training/docs/` or `tools/`

**What to investigate**
- Re-run matched-pair analysis on current data:
  - clean -> Teams real/fake
  - VisoMaster original fake -> Teams fake
  - optionally enhanced -> Teams-enhanced if aligned pairs exist
- Break out metrics by:
  - strategy
  - swap model
  - enhancer
  - content family
  - brightness / sharpness regime

**Specific questions**
- Is Teams truly one transform family, or two or more modes?
- Is the existing brightness/contrast overshoot still present?
- Does the sharpness bimodality cluster by method or by input quality?
- Is chroma handling worth modeling?
- Should the transform be:
  - one better preset,
  - a mixture of two presets,
  - or applied selectively by family?

**Recommended deeper target**
- Replace the single hard-coded policy:
  - “non-Teams families only, `p=0.15`”
- with a data-backed decision among:
  - `p=0.10`, `0.15`, `0.25`
  - single-mode simulation
  - mixture simulation
  - or offline cached Teamsified data for selected families

**Acceptance**
- New report using matched pairs from current buckets.
- Chosen augmentation policy justified by measured deltas, not carried over by habit.

### Track C: Target-Domain Evaluation And INT8 Scorecard

**Purpose**
- Make candidate selection reflect the real deployment goal.

**Write scope**
- `DeepfakeBench/training/arena/`
- `DeepfakeBench/training/validate_custom_sources.py`
- eval manifests or runner scripts

**Tasks**
- Build dev/lockbox split for `teams-faces-data-test-*`.
- Compute poor-quality and lighting-extreme real subsets.
- Add scorecard export for:
  - Teams-real FPR,
  - target fake recall,
  - per-method/per-enhancer breakdown,
  - INT8 delta.
- Make the scorecard easy to run across multiple checkpoints.

**Acceptance**
- One command or one short runner produces comparable model tables.

### Track D: Experiment Configs And Launch Matrix

**Purpose**
- Turn the plan into runnable YAMLs once Tracks A-C produce their outputs.

**Write scope**
- `DeepfakeBench/training/experiments/`
- optionally one coordinating doc under `DeepfakeBench/training/docs/`

**Tasks**
- Create a clean experiment family derived from `R12_G`.
- Use corrected seed plumbing explicitly.
- Add configs for:
  - merged Teams-enhanced source baseline
  - heavier enhanced emphasis
  - stronger Teams-real emphasis
  - Teams-simulation ablations driven by Track B
  - one curriculum pair
  - one seed repeat

**Acceptance**
- Config set is consistent, runnable, and directly traceable to the plan.

### Track E: Narrow Curriculum Ablation

**Purpose**
- Test the DF40-first idea once, in a bounded way.

**Write scope**
- experiment YAMLs only
- optional small doc note

**Important background**
- Repo history already suggests flat mixed training generally beat earlier curriculum ideas.
- So this should be a single falsifiable ablation, not the new default.

**Recommended design**
- Stage 1:
  - `DF40 + VCD real`
  - short run
  - moderate augmentation
- Stage 2:
  - initialize from Stage 1
  - full target-domain mixture, including merged Teams-enhanced source

**Matched control**
- One flat run from the same starting checkpoint and same total budget.

**Acceptance**
- Curriculum only survives if it improves target-domain metrics without losing the production fit.

## 7. Recommended Execution Order

### Phase 0: Truth And Baseline
1. `S0` Freeze baseline scorecard.
2. `S1` Audit `enhanced-visomaster-cropped`.
3. `S2` Freeze dev/lockbox evaluation manifests.

### Phase 1: Parallel Implementation
4. Track A implements merged Teams-enhanced loader.
5. Track B re-characterizes Teams channel and proposes augmentation policy.
6. Track C finalizes target-domain + INT8 scorecard tooling.
7. Track D drafts experiment YAMLs in parallel, but only finalizes after A-C outputs land.
8. Track E prepares the narrow curriculum pair.

### Phase 2: Integration
9. Merge Track A first.
10. Merge Track C next so evaluation is stable.
11. Merge Track B once the augmentation policy is selected.
12. Finalize Track D YAMLs.
13. Keep Track E as an optional ablation branch.

### Phase 3: Experiment Batches
14. Launch the flat merged-source baseline first.
15. Launch weight and augmentation ablations next.
16. Launch the curriculum pair only after at least one flat merged-source baseline exists.
17. Promote finalists only using the target-domain scorecard, with INT8 included.

## 8. Initial Experiment Set

These are the first experiments worth running once the implementation tracks land.

### Flat Mainline
- `FT_MERGED_BASE`
  - merged Teams-enhanced source
  - moderate enhanced weight
- `FT_MERGED_ENH_HEAVY`
  - same, heavier enhanced emphasis
- `FT_MERGED_REAL_BOOST`
  - same, more Teams-real / external-real emphasis

### Teams Transform Ablations
- `FT_MERGED_TEAMSIM_SINGLE`
  - best flat baseline + chosen single-mode Teams sim
- `FT_MERGED_TEAMSIM_MIX`
  - if Track B finds a real two-mode benefit
- `FT_MERGED_LIGHTING_PLUS_TEAMSIM`
  - best Teams-sim config + updated lighting block

### Stability / Reproducibility
- `FT_MERGED_BEST_SEED2`
  - exact repeat of best flat config with true seed change

### Curriculum
- `FT_CURRIC_STAGE1`
- `FT_CURRIC_STAGE2`
- `FT_FLAT_MATCHED_CTRL`

## 9. Non-Negotiable Checks

- Do not train the new Teams-enhanced bucket by exploding it into one sample per enhancer variant inside the existing sampler.
- Do not trust `real_fake_bucket` metadata without resolver verification.
- Do not choose winners from generic holdout alone.
- Do not accept a target-domain win that vanishes in INT8.
- Do not assume the existing Teams simulator is final.

## 10. Suggested Agent Split

If using separate worktrees, use these boundaries:

### Agent 1: Data Integration
- Owns Track A only.
- Avoids evaluation and experiment YAML files.

### Agent 2: Teams Simulation Investigation
- Owns Track B only.
- Avoids data loader code and YAMLs.

### Agent 3: Evaluation / Scorecard
- Owns Track C only.
- Avoids augmentation and data loader code.

### Agent 4: Experiment Configs
- Owns Track D and optionally Track E.
- Should wait until the outputs from A-C are stable.

## 11. Bottom Line

The right next round is not just “R12 + more enhanced VisoMaster.”

It is:
- make the production metric primary,
- integrate the new Teams-passed-through enhanced source correctly,
- audit and probably redesign the Teams simulation using current matched-pair data,
- keep DF40 as an anchor instead of a target,
- and test curriculum once, narrowly, instead of betting the round on it.

That is the shortest path to a model that is better for the actual Teams deployment rather than just better on internal validation.
