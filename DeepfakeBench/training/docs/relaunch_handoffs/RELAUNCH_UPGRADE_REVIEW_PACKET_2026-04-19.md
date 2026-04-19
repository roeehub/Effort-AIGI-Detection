# Relaunch Upgrade Review Packet

**Date:** 2026-04-19  
**Branch:** `teams-relaunch-root-2026-04-17`

This is the single review entry point for the current relaunch state.

It consolidates:

- the original April 6 upgrade intent in
  `DeepfakeBench/training/docs/TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md`
- the April 17 relaunch worktree handoffs (`WT-A` through `WT-F`)
- the April 18-19 WT-B runtime/smoke closeout
- the April 19 proper-data loader / artifact / smoke follow-up for the new
  buckets

Use this file when handing the repo to the next reviewer.

## Start Here

Read in this order:

1. this file
2. `DeepfakeBench/training/docs/relaunch_handoffs/TASK_BOARD_2026-04-17.md`
3. `DeepfakeBench/training/docs/relaunch_handoffs/WT_B_AND_NEW_DATA_READINESS_2026-04-19.md`
4. `DeepfakeBench/training/docs/relaunch_handoffs/NEW_DATA_LOADER_AND_EXPERIMENT_HANDOFF_2026-04-19.md`
5. `DeepfakeBench/training/docs/TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md`

Then review the current code patch in:

- `DeepfakeBench/training/data/sources/combined_paired.py`
- `DeepfakeBench/training/data/sources/proper_data.py`
- `DeepfakeBench/training/utils/grouping.py`
- `DeepfakeBench/training/train_sweep.py`
- `DeepfakeBench/training/tests/test_phase4_family_pipeline.py`
- `DeepfakeBench/training/arena/build_visomaster_proper_data_artifacts.py`
- `DeepfakeBench/training/arena/inspect_visomaster_proper_buckets.py`
- `DeepfakeBench/training/experiments/phase2_round13/R13_STARTUP_SMOKE_WTB3_weak_signal_hints_plus_teams_hints.yaml`
- `DeepfakeBench/training/experiments/phase2_round13/R13_SMOKE_WTB3_weak_signal_hints_plus_teams_hints.yaml`
- `DeepfakeBench/training/experiments/phase2_round13/R13_STARTUP_SMOKE_PROPER_DATA_WTF_PROVISIONAL.yaml`

## Executive Summary

- The April 6 upgrade plan is no longer just a plan. Most of the core
  implementation and evaluation scaffolding it called for has landed.
- On the relaunch task board, all `WT-*` tracks are now marked `merged`.
- The WT-B weak-signal runtime is no longer a local-only claim. Both WT-B
  launcher smokes succeeded remotely.
- The April 19 new buckets are now on an explicit training-side
  `combined_paired.proper_data` path. They are not being folded into legacy
  `visomaster`, `visomaster_hints`, or weak-signal aliases.
- A proper-data launcher startup smoke also succeeded remotely.
- What remains is not basic loader uncertainty anymore. The remaining work is
  operational and planning-oriented:
  - close out the current patch in git
  - regenerate provisional proper-data artifacts after Teams propagation
    stabilizes
  - decide whether the current ragged-clean intersection policy is acceptable
    for the first packet
  - finalize and launch the first real experiment matrix
  - optionally run the WT-E promotion contract if promotion judgment is needed

## Naming Note

The April 17 relaunch `WT-*` labels do **not** map one-to-one to the original
April 6 "Track A/B/C/D/E" labels.

Treat the documents like this:

- the April 6 plan defines the original intent and gating logic
- the April 17-19 `WT-*` handoffs are the relaunch execution layer
- WT-F and the April 19 proper-data follow-up are genuine extensions beyond the
  original April 6 plan

## Original Plan To Current State

### S0: Freeze Baseline / Promotion Logic

Original goal:

- freeze the exact baseline and strongest post-R12 candidates on a shared
  target-domain scorecard
- use calibrated promotion logic instead of generic holdout stories

Current state:

- target-domain scorecard and manifest-backed evaluation tooling already exist
- WT-D added decision-layer tooling for threshold sweeps, abstain bands,
  temporal aggregation, and hysteresis
- WT-E added the authoritative promotion-contract suite, shortlist checkpoint
  map, and runnable wrappers
- the repo now has the correct machinery to decide a promotion winner

Still left:

- run the five-checkpoint shortlist through the WT-E promotion contract and the
  WT-D decision tools on the actual frozen report root
- promotion is therefore tooling-ready, but not fully operationally closed

Primary docs:

- `DeepfakeBench/training/docs/relaunch_handoffs/WT-D_2026-04-17.md`
- `DeepfakeBench/training/docs/relaunch_handoffs/WT-E_2026-04-17.md`

### S1: Audit Enhanced / Policy-Sensitive Data Truth

Original goal:

- turn ambiguous enhanced/Teams-related data into training-safe sources with a
  stable truth contract

Current state:

- the original Track A resolver-driven Teams-enhanced path had already landed
  before the relaunch
- WT-A then re-froze the honest policy-aware truth in tracked artifacts and
  docs, instead of pretending the old mixed semantics were already clean
- WT-B then landed explicit weak-signal hint lanes plus direct Teams
  policy-filtering so the WT-B family is actually honest
- the April 19 proper-data work added a separate training-side path for the new
  buckets instead of forcing them into the old VisoMaster or hints taxonomy

Still left:

- exact corrected `val` / `test` hint counts still depend on the missing raw
  April 17 policy packet
- the current proper-data snapshot is still provisional because Teams
  propagation is incomplete and a regeneration pass is still required later

Primary docs:

- `DeepfakeBench/training/docs/relaunch_handoffs/WT-A_2026-04-17.md`
- `DeepfakeBench/training/docs/relaunch_handoffs/WT-B_2026-04-18.md`
- `DeepfakeBench/training/docs/relaunch_handoffs/WT_B_AND_NEW_DATA_READINESS_2026-04-19.md`
- `DeepfakeBench/training/docs/relaunch_handoffs/NEW_DATA_LOADER_AND_EXPERIMENT_HANDOFF_2026-04-19.md`

### S2: Freeze Target-Domain Evaluation Splits

Original goal:

- create deterministic dev/lockbox target-domain manifests and slices that are
  stable enough to score checkpoints fairly

Current state:

- the original plan's manifest-backed target-domain evaluation path exists
- WT-E aligned promotion-facing suite names and shortlist scoring against that
  path
- WT-F aligned future `proper_*` suite naming so future proper-data evaluation
  can stay on the same scoring contract

Still left:

- no structural blocker remains here
- the remaining work is operational execution of the scoring and promotion
  wrappers, not missing split/freeze machinery

Primary docs:

- `DeepfakeBench/training/docs/TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md`
- `DeepfakeBench/training/docs/relaunch_handoffs/WT-E_2026-04-17.md`
- `DeepfakeBench/training/docs/relaunch_handoffs/WT-F_2026-04-17.md`

## Relaunch Track Status

### WT-A

Status:

- merged

What landed:

- policy-truth freeze artifact
- validator and focused test
- lane-semantics redesign doc

What it means now:

- WT-A is the authoritative honesty correction layer for the old mixed
  VisoMaster / Teams semantics
- it intentionally refused to overclaim a tracked-tree loader rewrite

What remains:

- exact corrected `val` / `test` hint counts need the missing raw policy packet

Primary doc:

- `DeepfakeBench/training/docs/relaunch_handoffs/WT-A_2026-04-17.md`

### WT-B

Status:

- merged
- runtime- and smoke-proven

What landed:

- explicit `combined_paired.visomaster_hints`
- explicit `combined_paired.visomaster_hints_teams`
- policy-aware direct Teams filtering
- grouping/family routing for the new hint lanes
- runnable `WTB1` / `WTB2` / `WTB3` configs
- launcher-oriented startup and integration smokes
- discovery-cache support for expensive source listings

Remote proof:

- startup smoke succeeded
  - Vertex custom job id: `9012653657048481792`
  - W&B run id: `8bjcadyr`
- integration smoke succeeded
  - Vertex custom job id: `2921535161029885952`
  - W&B run id: `g6f8fovi`

What remains:

- broader unrelated `test_phase4_family_pipeline.py` failures still exist
- a reporting oddity (`unknown_fake: 20` alongside `external_real: 20`) is
  still worth keeping in mind when touching family accounting

Primary docs:

- `DeepfakeBench/training/docs/relaunch_handoffs/WT-B_2026-04-18.md`
- `DeepfakeBench/training/docs/R13_WTB_WEAK_SIGNAL_RUNBOOK_2026-04-18.md`
- `DeepfakeBench/training/docs/relaunch_handoffs/WT_B_AND_NEW_DATA_READINESS_2026-04-19.md`

### WT-C

Status:

- merged

What landed:

- truthful GammaUp / nuisance-control plumbing
- Teams passthrough special-augmentation knobs
- three sidecar configs for controlled augmentation experiments
- focused augmentation tests

What remains:

- these are still sidecars, not promotion claims
- the sidecars still need disciplined measurement through the existing
  evaluation path before they should influence the mainline packet

Primary doc:

- `DeepfakeBench/training/docs/relaunch_handoffs/WT-C_2026-04-17.md`

### WT-D

Status:

- merged

What landed:

- report-driven analysis tooling for threshold sweeps
- abstain-band analysis
- temporal aggregation comparison
- hysteresis / minimum-positive-run comparison

What remains:

- someone still has to run the tooling on the actual frozen shortlist report
  root to name a calibrated winner

Primary doc:

- `DeepfakeBench/training/docs/relaunch_handoffs/WT-D_2026-04-17.md`

### WT-E

Status:

- merged

What landed:

- authoritative promotion-contract suite
- frozen five-checkpoint shortlist map
- promotion wrappers and runbook
- diagnostic scorecards explicitly demoted to non-authoritative status

What remains:

- run the promotion contract on the shortlist to emit `promotion_winner.json`

Primary doc:

- `DeepfakeBench/training/docs/relaunch_handoffs/WT-E_2026-04-17.md`

### WT-F

Status:

- merged

What landed:

- canonical future `proper_*` lane schema
- inventory / provenance contract
- future manifest builder
- future suite template aligned with WT-E naming

What remains:

- WT-F itself did not wire those lanes into training
- that gap was intentionally outside WT-F and is what the April 19 proper-data
  follow-up addressed

Primary docs:

- `DeepfakeBench/training/docs/relaunch_handoffs/WT-F_2026-04-17.md`
- `DeepfakeBench/training/docs/PROPER_DATA_SCHEMA_AND_FUTURE_MANIFESTS_2026-04-17.md`

## Post-WT Proper-Data Follow-Up

This is the main April 19 extension beyond the original relaunch packet.

What happened:

- the new HDTF / quickclips VisoMaster buckets were inspected and classified as
  `proper_data`
- provisional WT-F inventory / manifest / suite artifacts were generated for
  the current snapshot
- a real training-side `combined_paired.proper_data` path was added
- a tiny launcher-only startup smoke proved that the new proper-data lanes load
  honestly through training

Training-side proof:

- config:
  `DeepfakeBench/training/experiments/phase2_round13/R13_STARTUP_SMOKE_PROPER_DATA_WTF_PROVISIONAL.yaml`
- Vertex custom job id: `2064725331922649088`
- W&B run id: `bxay0n66`
- final state: `JOB_STATE_SUCCEEDED`
- proof lines included:
  - `ProperData: enabled=True -> 128 samples`
  - nonzero lane counts for:
    - `proper_visomaster_clean`
    - `proper_visomaster_enhanced_clean`
    - `proper_visomaster_teams`
    - `proper_visomaster_enhanced_teams`
  - `MAX STEPS REACHED: 2/2`

Current meaning:

- the new data is now real training input, not just a document or manifest idea
- the honest taxonomy is explicit `proper_*` lanes
- the new data is ready to enter experiment planning

Still provisional:

- Teams propagation is still incomplete; roughly `30%` more data is expected to
  land
- the current inventory / manifest / suite files are therefore provisional and
  must be regenerated later
- the current loader keeps ragged-clean pairs by intersecting available anchor
  indices across the real/fake pair; that policy still needs an explicit human
  accept/reject decision before longer runs

Primary docs:

- `DeepfakeBench/training/docs/relaunch_handoffs/WT_B_AND_NEW_DATA_READINESS_2026-04-19.md`
- `DeepfakeBench/training/docs/relaunch_handoffs/NEW_DATA_LOADER_AND_EXPERIMENT_HANDOFF_2026-04-19.md`

## Current Review Scope In The Working Tree

If the next agent is reviewing the current local patch, these are the most
important changed surfaces:

- runtime:
  - `DeepfakeBench/training/data/sources/combined_paired.py`
  - `DeepfakeBench/training/data/sources/proper_data.py`
  - `DeepfakeBench/training/utils/grouping.py`
  - `DeepfakeBench/training/train_sweep.py`
- tests:
  - `DeepfakeBench/training/tests/test_phase4_family_pipeline.py`
  - `DeepfakeBench/training/tests/test_build_visomaster_proper_data_artifacts.py`
  - `DeepfakeBench/training/tests/test_inspect_visomaster_proper_buckets.py`
- proper-data artifact tooling:
  - `DeepfakeBench/training/arena/build_visomaster_proper_data_artifacts.py`
  - `DeepfakeBench/training/arena/inspect_visomaster_proper_buckets.py`
- generated provisional artifacts:
  - `DeepfakeBench/training/arena/inventories/proper_visomaster_wave_2026_04_19_provisional.yaml`
  - `DeepfakeBench/training/arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json`
  - `DeepfakeBench/training/arena/target_domain_suites.proper_data_future.provisional_2026-04-19.yaml`
  - `DeepfakeBench/training/arena/reports/*.json`
- smoke configs:
  - `DeepfakeBench/training/experiments/phase2_round13/R13_STARTUP_SMOKE_WTB3_weak_signal_hints_plus_teams_hints.yaml`
  - `DeepfakeBench/training/experiments/phase2_round13/R13_SMOKE_WTB3_weak_signal_hints_plus_teams_hints.yaml`
  - `DeepfakeBench/training/experiments/phase2_round13/R13_STARTUP_SMOKE_PROPER_DATA_WTF_PROVISIONAL.yaml`
- packaging / repo closeout:
  - `DeepfakeBench/training/.dockerignore`
  - `DeepfakeBench/training/VERSION`

## Current Planning Stance

The current docs are aligned on the first-night planning shape:

- keep the WT-B weak-signal family as one axis
- keep `proper_data` as one explicit `off/on` axis
- do not explode the first packet into many proper-data sub-arms until the
  Teams snapshot stabilizes and the provisional artifacts are regenerated

Recommended first planning packet:

- control: `WTB3`
- `WTB3 + proper` using the retained unenhanced proper lanes
- `WTB3 + full retained proper snapshot` using both unenhanced and enhanced
  proper lanes

Primary docs:

- `DeepfakeBench/training/docs/relaunch_handoffs/WT_B_AND_NEW_DATA_READINESS_2026-04-19.md`
- `DeepfakeBench/training/docs/relaunch_handoffs/NEW_DATA_LOADER_AND_EXPERIMENT_HANDOFF_2026-04-19.md`

## What Is Still Open

- commit / land the current proper-data patch cleanly
- decide whether the current ragged-clean intersection policy is acceptable for
  the first packet or whether the provisional artifacts should be regenerated
  with a stricter clean-side `16/16` filter
- regenerate the proper-data inventory / manifest / suite after Teams
  propagation finishes
- run the first real experiment packet
- run WT-E / WT-D promotion analysis if model promotion is needed in parallel

## Bottom Line

The repo is past the "do we have the necessary plumbing?" phase.

The current state is:

- all relaunch `WT-*` tracks are merged
- WT-B is remotely proven
- proper-data training integration exists and is remotely proven at startup
- experiment planning with the new data can begin now

The remaining uncertainty is about:

- provisional data counts
- clean-versus-ragged policy on the new proper-data path
- operational closeout and experiment selection

That is the right context to hand to the next reviewer.
