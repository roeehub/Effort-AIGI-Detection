# Teams Worktree Agent Prompts

**Root branch:** `teams-relaunch-root-2026-04-17`  
**Anchor commit:** `6c0d1fb`  
**Repo root:** `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision`

This file turns the relaunch report into ready-to-run worktree prompts.

Source of truth:

- `DeepfakeBench/training/docs/TEAMS_EXPERIMENT_RELAUNCH_REPORT_2026-04-17.md`
- shared task board: `DeepfakeBench/training/docs/relaunch_handoffs/TASK_BOARD_2026-04-17.md`

Recommended launch order right now:

1. `WT-A`
2. `WT-E`
3. `WT-D`
4. `WT-C`
5. `WT-F`

Deferred until `WT-A` merges a lane-semantics freeze:

- `WT-B`

## Generic Dispatcher Prompt

Use this when you want to launch multiple agents from the same root branch and let each one pick an unclaimed track.

```text
You are starting from the shared repo root:
/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision

Root branch:
teams-relaunch-root-2026-04-17

Read these coordination docs first:
1. DeepfakeBench/training/docs/TEAMS_EXPERIMENT_RELAUNCH_REPORT_2026-04-17.md
2. DeepfakeBench/training/docs/TEAMS_WORKTREE_AGENT_PROMPTS_2026-04-17.md
3. DeepfakeBench/training/docs/relaunch_handoffs/TASK_BOARD_2026-04-17.md
4. DeepfakeBench/training/docs/relaunch_handoffs/README.md

Your job is to choose one free worktree track, claim it so the next agent sees it is taken, then execute that track in its own separate worktree.

Claiming rules:
- only choose a track whose status is `free`
- do not choose `WT-B` unless the task board says WT-A has merged a lane-semantics freeze and WT-B is now `free`
- once you choose a track, immediately update the task board and set:
  - status: `claimed`
  - owner: your agent name or identifier
  - branch
  - worktree path
  - claim time
- commit the task-board claim to `teams-relaunch-root-2026-04-17` before doing the track work so other agents can see it

Execution rules:
- after claiming, create or reuse the branch and worktree specified for that track in TEAMS_WORKTREE_AGENT_PROMPTS_2026-04-17.md
- read the track-specific prompt in that file and follow it exactly
- re-investigate the task before editing; refine the scope if the report is wrong or incomplete
- stay inside the ownership boundary for the claimed track
- commit your work on the track branch
- if the work is complete and verified, rebase onto teams-relaunch-root-2026-04-17, merge back with --no-ff, and update the task board status to `merged`
- if blocked, update the task board status to `blocked` with a short blocker note
- if you finish analysis but not a mergeable change, update the task board status to `analysis-done`

Selection priority:
1. WT-A
2. WT-E
3. WT-D
4. WT-C
5. WT-F
6. WT-B only after WT-A freeze is merged

Important constraints:
- do not edit another track's owned files
- do not treat the baseline R13 YAML as a shared scratchpad
- do not claim a target-domain win from old-semantics training
- do not claim a promotion winner from fixed-threshold 0.5 tables alone
```

## Shared Rules For Every Agent

Every agent should:

1. Read the relaunch report first, then the track-specific supporting docs, then the owned code.
2. Re-derive the task before editing. If the report overstates, understates, or mis-scopes the track, refine the scope and record that explicitly in the handoff.
3. Create its own worktree and branch from `teams-relaunch-root-2026-04-17`. If the worktree already exists, reuse it.
4. Stay inside the owned paths. Do not edit files in another track's ownership zone unless you stop and document why the ownership model is wrong.
5. Add or update verification that matches the track. If no automated test is appropriate, produce a concrete reproducible check.
6. Commit on the track branch.
7. Only merge back into `teams-relaunch-root-2026-04-17` if the work is actually complete, verified, and still within ownership.
8. Before merging back, rebase or merge the latest `teams-relaunch-root-2026-04-17` into the track branch.
9. Merge back with `--no-ff`, then leave a short handoff note in a uniquely named file under `DeepfakeBench/training/docs/relaunch_handoffs/`.
10. If blocked, do not guess across ownership boundaries. Write the blocker and stop.

Suggested merge-back safety check:

- ensure the main workspace at `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision` is still on `teams-relaunch-root-2026-04-17` and clean before merging
- if another agent already changed the same ownership boundary, stop and hand off instead of forcing the merge

## Prompt: WT-A Data Truth / Policy Integration

```text
You own WT-A Data Truth / Policy Integration for the Teams relaunch.

Start from the shared repo root:
/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision

Root branch:
teams-relaunch-root-2026-04-17

Create or reuse this worktree:
../wt-a-data-truth-2026-04-17

Create or reuse this branch:
wt-a-data-truth-2026-04-17

First read these sources of truth in this order:
1. DeepfakeBench/training/docs/TEAMS_EXPERIMENT_RELAUNCH_REPORT_2026-04-17.md
2. DeepfakeBench/training/docs/VISOMASTER_BAD_DATA_POLICY_PLAN_IMPACT_2026-04-17.md
3. DeepfakeBench/training/docs/research_2026-04-15_round2/02_target_domain_data_truth.md
4. DeepfakeBench/training/docs/research_2026-04-15/04_data_composition_and_curriculum_opportunities.md
5. DeepfakeBench/training/data/sources/combined_paired.py
6. DeepfakeBench/training/data/sources/visomaster.py

Your first job is to critically re-investigate the task and refine it if needed before editing. Do not assume the report is perfectly scoped.

You own:
- DeepfakeBench/training/data/sources/visomaster.py
- DeepfakeBench/training/data/sources/combined_paired.py
- policy-aware source redesign docs
- corrected composition and contamination reporting artifacts

Avoid:
- DeepfakeBench/training/arena/*
- DeepfakeBench/training/data/augmentations/*
- active phase2_round13 YAML edits unless you conclude the ownership split is wrong and document that

Deliverables:
- prove whether current training is policy-aware or not
- produce corrected per-lane counts and contamination accounting for promotion-relevant configs
- define explicit lane semantics for visomaster hints and visomaster hints (teams)
- make it unambiguous whether same-day retrains should be labeled policy-corrected, weak-signal-only, or old-semantics

Verification:
- add or update focused tests if loader behavior changes
- include a reproducible composition-truth artifact or script

Merge-back rule:
- if complete, commit on wt-a-data-truth-2026-04-17, rebase onto teams-relaunch-root-2026-04-17, then merge back to the root branch with --no-ff
- add a handoff file named DeepfakeBench/training/docs/relaunch_handoffs/WT-A_YYYY-MM-DD.md
```

## Prompt: WT-C Augmentation / Nuisance Invariance

```text
You own WT-C Augmentation / Nuisance Invariance for the Teams relaunch.

Start from the shared repo root:
/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision

Root branch:
teams-relaunch-root-2026-04-17

Create or reuse this worktree:
../wt-c-nuisance-2026-04-17

Create or reuse this branch:
wt-c-nuisance-2026-04-17

First read these sources of truth in this order:
1. DeepfakeBench/training/docs/TEAMS_EXPERIMENT_RELAUNCH_REPORT_2026-04-17.md
2. DeepfakeBench/training/docs/research_2026-04-15_round2/04_nuisance_invariance_and_augmentation_truth.md
3. DeepfakeBench/training/docs/SPATIAL_AUGMENTATION_PROPOSAL_2026-04-14.md
4. DeepfakeBench/training/docs/TEAMS_AUGMENTATION_SIDE_TRACK_HANDOFF_2026-04-06.md
5. DeepfakeBench/training/data/augmentations/transforms.py
6. DeepfakeBench/training/data/augmentations/pipelines.py
7. DeepfakeBench/training/data/augmentations/teams_simulation.py

Your first job is to critically re-investigate the track. Confirm which augmentations are actually live at runtime, which ones are dead config intent, and whether the narrow truthful nuisance sidecar still makes sense.

You own:
- DeepfakeBench/training/data/augmentations/*
- DeepfakeBench/training/tests/test_lighting_transforms.py
- DeepfakeBench/training/tests/test_teams_adaptive_simulation.py
- new sidecar YAMLs under fresh names

Avoid:
- DeepfakeBench/training/data/sources/*
- DeepfakeBench/training/arena/*
- active main-line Track A YAMLs

Deliverables:
- runtime-truth confirmation for GammaUp and related lighting paths
- one narrow truthful nuisance sidecar family
- no silent edits to active baseline configs

Verification:
- add or update augmentation tests
- if you add YAMLs, ensure they are clearly sidecars and not baseline rewrites
- treat WT-D as the owner of decision-layer measurement and promotion judgment; do not self-certify promotion wins from this track

Merge-back rule:
- if complete, commit on wt-c-nuisance-2026-04-17, rebase onto teams-relaunch-root-2026-04-17, then merge back to the root branch with --no-ff
- add a handoff file named DeepfakeBench/training/docs/relaunch_handoffs/WT-C_YYYY-MM-DD.md
```

## Prompt: WT-D Stability / Decision-System Analysis

```text
You own WT-D Stability / Decision-System Analysis for the Teams relaunch.

Start from the shared repo root:
/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision

Root branch:
teams-relaunch-root-2026-04-17

Create or reuse this worktree:
../wt-d-decision-system-2026-04-17

Create or reuse this branch:
wt-d-decision-system-2026-04-17

First read these sources of truth in this order:
1. DeepfakeBench/training/docs/TEAMS_EXPERIMENT_RELAUNCH_REPORT_2026-04-17.md
2. DeepfakeBench/training/docs/research_2026-04-15_round2/05_decision_system_low_fp_analysis.md
3. DeepfakeBench/training/docs/research_2026-04-15_round3/04_internal_instability_mitigation_summary_for_parallel_review.md
4. DeepfakeBench/training/docs/TRACK_C_SCORECARD_RUNBOOK_2026-04-07.md

Your first job is to critically re-investigate whether today’s real-Teams pain can move materially without new training. Re-rank the immediate decision-layer opportunities before writing or changing scripts.

You own:
- new analysis scripts and docs for threshold sweeps, temporal aggregation, hysteresis, abstain-band, and stability analysis
- stability-specific reports built on frozen checkpoints and reports

Avoid:
- DeepfakeBench/training/data/sources/*
- DeepfakeBench/training/data/augmentations/*
- DeepfakeBench/training/arena/score_teams_promotion_contract.py

Deliverables:
- checkpoint ranking under the actual low-FP contract assumptions
- explicit judgment on whether decision-layer leverage is strong enough for immediate no-new-data work
- frame-to-frame or crop-jitter stability evidence

Verification:
- every result must be reproducible from a checked-in script or documented command
- do not claim a promotion winner from fixed-threshold 0.5 tables alone

Merge-back rule:
- if complete, commit on wt-d-decision-system-2026-04-17, rebase onto teams-relaunch-root-2026-04-17, then merge back to the root branch with --no-ff
- add a handoff file named DeepfakeBench/training/docs/relaunch_handoffs/WT-D_YYYY-MM-DD.md
```

## Prompt: WT-E Evaluation Contract / Promotion Tooling

```text
You own WT-E Evaluation Contract / Promotion Tooling for the Teams relaunch.

Start from the shared repo root:
/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision

Root branch:
teams-relaunch-root-2026-04-17

Create or reuse this worktree:
../wt-e-promotion-contract-2026-04-17

Create or reuse this branch:
wt-e-promotion-contract-2026-04-17

First read these sources of truth in this order:
1. DeepfakeBench/training/docs/TEAMS_EXPERIMENT_RELAUNCH_REPORT_2026-04-17.md
2. DeepfakeBench/training/docs/research_2026-04-15_round2/01_evaluation_contract_and_shortlist.md
3. DeepfakeBench/training/docs/TRACK_C_SCORECARD_RUNBOOK_2026-04-07.md
4. DeepfakeBench/training/arena/score_teams_promotion_contract.py
5. DeepfakeBench/training/arena/run_target_domain_validation_sequential.py

Your first job is to critically re-investigate the current promotion path. Confirm what is already authoritative, what is still provisional, and where the current suite or tooling still misleads workers.

You own:
- DeepfakeBench/training/arena/run_target_domain_validation_sequential.py
- DeepfakeBench/training/arena/score_teams_promotion_contract.py
- DeepfakeBench/training/arena/target_domain_suites*.yaml
- DeepfakeBench/training/arena/checkpoint_maps/*
- scorecard runbooks tied to the promotion path

Avoid:
- DeepfakeBench/training/data/sources/*
- DeepfakeBench/training/data/augmentations/*
- active training YAMLs unless you conclude reassignment is required and document it

Deliverables:
- one authoritative calibrated promotion path
- decisive scorecard or contract artifacts
- explicit statement of what is promotion-authoritative versus diagnostic-only

Verification:
- add or update tests around the promotion-contract scorer and suite behavior where appropriate
- do not leave fixed-threshold-only paths looking authoritative if they are not
- keep the lane truth aligned with WT-A and keep future naming aligned with WT-F

Merge-back rule:
- if complete, commit on wt-e-promotion-contract-2026-04-17, rebase onto teams-relaunch-root-2026-04-17, then merge back to the root branch with --no-ff
- add a handoff file named DeepfakeBench/training/docs/relaunch_handoffs/WT-E_YYYY-MM-DD.md
```

## Prompt: WT-F New Proper-Data Schema / Future Manifests

```text
You own WT-F New Proper-Data Schema / Future Manifests for the Teams relaunch.

Start from the shared repo root:
/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision

Root branch:
teams-relaunch-root-2026-04-17

Create or reuse this worktree:
../wt-f-proper-data-schema-2026-04-17

Create or reuse this branch:
wt-f-proper-data-schema-2026-04-17

First read these sources of truth in this order:
1. DeepfakeBench/training/docs/TEAMS_EXPERIMENT_RELAUNCH_REPORT_2026-04-17.md
2. DeepfakeBench/training/docs/VISOMASTER_BAD_DATA_POLICY_PLAN_IMPACT_2026-04-17.md
3. DeepfakeBench/training/docs/research_2026-04-15_round2/02_target_domain_data_truth.md
4. COORDINATED_CAPTURE_PLAN.md
5. TEAMS_CAPTURE_README.md

Your first job is to critically re-investigate how future proper data should enter the repo so it stays provenance-clean and evaluable. If the current lane schema or capture assumptions are weak, correct them before proposing templates.

You own:
- new proper-data schema docs
- new manifest builders or templates for future proper data only
- capture, inventory, and provenance docs for incoming clean data

Avoid:
- current frozen suite files owned by WT-E
- current loader code owned by WT-A
- active training YAMLs owned by WT-B or another track

Deliverables:
- canonical clean lane schema for incoming proper data
- rules for keeping clean and Teams-parallel variants explicit
- future eval slice specification for exact target conditions

Verification:
- make the schema concrete enough that another worker could implement it without guessing
- do not hide proper data inside hint lanes or the old merged VTE lane
- coordinate lane naming with WT-E so future promotion tooling can score the new slices without ad hoc aliases

Merge-back rule:
- if complete, commit on wt-f-proper-data-schema-2026-04-17, rebase onto teams-relaunch-root-2026-04-17, then merge back to the root branch with --no-ff
- add a handoff file named DeepfakeBench/training/docs/relaunch_handoffs/WT-F_YYYY-MM-DD.md
```

## Prompt: WT-B Weak-Signal Ablation Configs

This track is intentionally gated. Do not launch it until `WT-A` has merged a lane-semantics freeze that makes the hint-lane names and meanings explicit.

```text
You own WT-B Weak-Signal Ablation Configs for the Teams relaunch.

Start from the shared repo root:
/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision

Root branch:
teams-relaunch-root-2026-04-17

Create or reuse this worktree:
../wt-b-weak-signal-ablations-2026-04-17

Create or reuse this branch:
wt-b-weak-signal-ablations-2026-04-17

Do not start until WT-A has merged a lane freeze into teams-relaunch-root-2026-04-17.

First read these sources of truth in this order:
1. DeepfakeBench/training/docs/TEAMS_EXPERIMENT_RELAUNCH_REPORT_2026-04-17.md
2. The merged WT-A handoff under DeepfakeBench/training/docs/relaunch_handoffs/
3. DeepfakeBench/training/docs/research_2026-04-15_round2/02_target_domain_data_truth.md
4. DeepfakeBench/training/experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml

Your first job is to critically re-investigate whether the three-arm weak-signal ablation family is still the right minimal experiment family after WT-A freezes semantics. If not, refine it and explain why.

You own:
- new experiment YAMLs only under DeepfakeBench/training/experiments/phase2_round13/
- new weak-signal runbooks and launch notes

Avoid:
- DeepfakeBench/training/data/sources/*
- DeepfakeBench/training/arena/*
- DeepfakeBench/training/data/augmentations/*
- editing the active baseline YAML in place

Deliverables:
- one honest three-arm weak-signal ablation family:
  - no hints
  - hints only
  - hints plus teams hints
- framing that explicitly says weak-signal corrected-data ablation, not target-domain realism

Verification:
- if possible, validate YAMLs with the repo’s existing dry-run path
- create new YAMLs under fresh names; do not use the baseline as a scratchpad

Merge-back rule:
- if complete, commit on wt-b-weak-signal-ablations-2026-04-17, rebase onto teams-relaunch-root-2026-04-17, then merge back to the root branch with --no-ff
- add a handoff file named DeepfakeBench/training/docs/relaunch_handoffs/WT-B_YYYY-MM-DD.md
```
