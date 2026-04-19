# VisoMaster Bad-Data Policy Plan Impact

**Date:** April 17, 2026

## Purpose

Consolidate the April 17, 2026 VisoMaster bad-data correction against the current
Teams target-domain plan.

This note is about the planning impact, not about defending any specific model
result. The main distinction is:

- pre-April-17 runs and composition reports describe the **old composition**
- the April 17 policy artifacts plus the local viewer / MCP surface define the
  **current reference interpretation**

So older run outcomes should now be treated as evidence from an outdated
composition unless the run is explicitly rebuilt under the new policy.

## Source Of Truth

- `DeepfakeBench/training/policy/visomaster_bad_data/VISOMASTER_BAD_DATA_POLICY_MANIFEST_2026-04-17.csv`
- `DeepfakeBench/training/policy/visomaster_bad_data/VISOMASTER_BAD_DATA_POLICY_SUMMARY_2026-04-17.json`
- `DeepfakeBench/training/policy/visomaster_bad_data/VISOMASTER_BAD_DATA_POLICY_REPORT_2026-04-17.md`
- `DeepfakeBench/training/policy/visomaster_bad_data/VISOMASTER_BAD_DATA_UPLOAD_AUDIT.md`
- `DeepfakeBench/training/docs/research_2026-04-15/04_data_composition_and_curriculum_opportunities.md`
- `DeepfakeBench/training/docs/research_2026-04-15/05_target_domain_gap_and_teams_enhanced_structural_findings.md`
- `DeepfakeBench/training/docs/research_2026-04-15_round2/01_evaluation_contract_and_shortlist.md`
- `DeepfakeBench/training/docs/research_2026-04-15_round2/02_target_domain_data_truth.md`
- `DeepfakeBench/training/experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml`
- local viewer / MCP served from `http://127.0.0.1:8501` and currently running
  `experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml`

## Current Reference State

The April 17 policy changes the interpretation of the historical bad
`visomaster_*` family:

- old bad `visomaster_*` rows are no longer method-faithful supervision
- `480` sample IDs are retained as `visomaster hints`
- `202` sample IDs are retained as `visomaster hints (teams)`
- `4904` sample IDs are ignored
- `3` sample IDs are delete-only

The policy report is explicit that any experiment or config still treating this
family as normal per-method VisoMaster supervision is outdated.

The upload audit also matters for Track A and Teams planning:

- `237 / 1348` `pair_complete` Teams rows in the direct Teams bucket are bad
  VisoMaster-through-Teams rows
- in the old `R13_A_trackA_teams_enhanced` train split that meant `198` direct
  Teams paired objects were bad VisoMaster-through-Teams
- the `54` resolver-confirmed `teams_v2_companion` rows inside
  `visomaster_teams_enhanced` are a subset of those same bad Teams-played
  VisoMaster rows
- in the old Track A train split that meant `42` merged-source train pairs sat
  on top of that contaminated subset

So the April 17 change is not a cosmetic relabel. It changes the meaning of:

- the direct baseline `visomaster` lane
- the direct Teams lane
- the Teams-connected subset inside `visomaster_teams_enhanced`

## What Still Holds

These parts of the forward plan remain valid and should stay:

- Freeze promotion on the Teams deployment objective, not on holdout AUC.
- Keep the frozen Teams scorecard / manifest tooling and the calibrated low-FP
  promotion-contract idea.
- Keep the split between:
  - training-selection metrics
  - deployment-promotion metrics
  - diagnostic regression suites
- Keep the conclusion that enhanced-through-Teams is its own mode and is not
  well-modeled by one generic Teams simulator.
- Keep the conclusion that the current merged `visomaster_teams_enhanced`
  source is structurally diluted and should not be treated as one honest
  "Teams-enhanced" family.
- Keep nuisance-invariance work on lighting, crop robustness, and decision-layer
  stability / calibration.
- Keep the rule that real Teams safety is the primary deployment gate.

In short: the **evaluation contract**, the **deployment objective**, and the
core **structural lessons** survive.

## What Must Be Re-Examined

These parts of the current plan should not be carried forward unchanged.

### 1. Old training-diet and weighting narratives

Any narrative built on the older composition memo should be re-read. That memo
assumed:

- `visomaster` was a normal method-faithful lane
- `visomaster_fake` was a large clean fake family
- direct Teams rows were a cleaner target-domain lane than they really were

Under the corrected viewer interpretation for the active `R13_A` config:

- total rows drop from `31,064` to `20,776`
- the old `visomaster` lane no longer exists as a normal source in the viewer
- retained bad-VisoMaster rows shrink from `11,178` old `visomaster` rows to
  `1,364` hint rows total
- old train bad-VisoMaster fake pairs shrink from `4,735` to `571`
  (`396` baseline hints + `175` Teams hints)

So any earlier claim about sampler balance, family leverage, or target-domain
exposure that depended on the old `visomaster` size should be treated as stale.

### 2. All active R13 YAMLs that still rely on the old VisoMaster lane

The checked-in training configs still reference old VisoMaster lanes such as:

- `combined_paired.visomaster`
- `combined_paired.teams`
- `combined_paired.visomaster_teams_enhanced`

I did not find checked-in training-side policy-manifest wiring outside the
viewer stack. That means the viewer/MCP now reflects the corrected
interpretation layer, but the checked-in training path should still be assumed
to follow old lane semantics until explicitly updated.

Practically, most active Round 13 YAMLs are affected, including:

- `R13_A_trackA_teams_enhanced`
- all `R13_FT7/8/9/10/11/12/13/14/15/16/17` Track-A follow-ons
- `R13_E`, `R13_F`, `R13_H`, `R13_I`
- the Track B sidecars that still include the same source family mix

### 3. The meaning of the direct Teams lane

The old plan treated `combined_paired.teams` as direct target-domain signal.
That is only partly true now.

The April 17 upload audit shows:

- `237 / 1348` `pair_complete` Teams rows are bad VisoMaster-through-Teams
- the old Track A train split contained `198` such paired objects

So direct Teams data is still relevant, but it is not a clean monolithic lane.
It now needs to be split mentally into:

- direct Teams non-VisoMaster signal
- weak-signal bad-VisoMaster-through-Teams signal

### 4. The meaning of `visomaster_teams_enhanced`

This source already needed structural caution before April 17 because it was
mostly `clean_fallback` and never emitted enhanced-through-Teams fake frames.
After April 17 it needs even more caution:

- its true Teams companion subset comes from the same bad Teams-played
  VisoMaster family
- only the `original` branch emits the original Teams fake
- with `p_original = 0.5`, only about half of those emissions use that branch

So the forward plan should still treat `visomaster_teams_enhanced` as useful
diagnostic structure, but not as a clean source of method-faithful or
deployment-faithful supervision.

### 5. Per-method VisoMaster reasoning

Any plan step that reasons about old `GhostFace-v1`, `GhostFace-v2`,
`GhostFace-v3`, `InStyleSwapper256-*`, or `Inswapper128` rows as if those old
rows were correct method labels should be reconsidered.

The retained rows are now weak labels:

- `visomaster hints`
- `visomaster hints (teams)`

They are no longer evidence for clean per-method balancing or per-method
generalization claims.

## What Should Be Newly Examined

These checks were not optional before, but they are now clearly first-class.

### 1. Training-path integration of the policy

We now need one explicit answer for every forward config:

- does the actual training loader consume the April 17 policy manifest?
- or is the corrected state only visible in the viewer/MCP interpretation layer?

Until this is answered with code-path evidence, new runs should be assumed to be
at risk of old-composition semantics.

### 2. Corrected composition reports for the active shortlist configs

Before more training, rebuild diet/composition truth under the corrected policy
for the configs that still matter:

- `R13_A_trackA_teams_enhanced`
- the current shortlist FT follow-ons
- any sidecar that is still promotion-relevant

The old composition memo is no longer enough because the baseline VisoMaster
mass has collapsed under the corrected interpretation.

### 3. Policy-aware source redesign

If the bad-data family is kept at all, it should be explicit in training as:

- `visomaster_hints`
- `visomaster_hints_teams`

with their own family semantics and with no pretense that they are normal
method-faithful VisoMaster supervision.

This is a design question now, not just a viewer-label question.

### 4. Evaluation provenance backfill on the fake side

The promotion contract itself is still the right idea, but fake-side provenance
needs better visibility.

New question:

- how much of `teams_fake_all_*` or related target-domain fake slices is driven
  by weak-signal VisoMaster-through-Teams content versus other fake families?

Right now the frozen eval lane is still useful, especially for real Teams FPR,
but it does not preserve enough provenance to answer this cleanly.

### 5. Weak-signal value ablation

The April 17 policy preserves `visomaster hints` and `visomaster hints (teams)`
because they may still carry weak signal. That should now be tested directly.

The key new ablation is:

- no bad VisoMaster lane at all
- hints-only baseline lane
- hints-plus-Teams lane

That will tell us whether the retained weak-signal subset is actually helping,
hurting, or just adding noise.

### 6. Clean replacement lane planning

The policy is a stopgap, not a final steady state. More actually correct
VisoMaster data is expected soon.

So the forward plan should now treat:

- corrected future VisoMaster data as the desired replacement lane
- current hint lanes as temporary weak-signal scaffolding only

## Planning Stance After April 17

The best way to talk about the current state is:

- the structural plan was directionally right
- the data interpretation changed materially
- old run results may not be representative because they were trained against
  an older composition
- the next phase should preserve the evaluation contract and structural lessons,
  but should reset composition assumptions before trusting more run outcomes

## Practical Reset Of The Forward Plan

1. Keep the frozen Teams promotion contract and scorecard path.
2. Freeze the April 17 policy as the current reference state for composition.
3. Treat old run outcomes as old-composition evidence, not as current-truth
   evidence.
4. Rebuild composition truth for the active candidate configs under the new
   policy.
5. Decide explicitly whether weak-signal hint lanes stay in the mainline or move
   to sidecar-only status.
6. Only then run new training or new promotion measurement intended to represent
   the corrected dataset state.

## Bottom Line

The April 17 correction does **not** kill the whole plan.

It does kill the parts of the plan that relied on old bad `visomaster_*` data
being normal method-faithful supervision.

What survives is the deployment objective, the evaluation contract, the
structural finding that enhanced-through-Teams is still under-modeled, and the
need for better nuisance invariance and calibration.

What resets is the composition story, the meaning of several Round 13 source
lanes, and the trust we should place in old run outcomes as representatives of
the now-corrected data state.
