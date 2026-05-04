# Thread: Weak-signal hints track (WT-B arc)

## The question

After the 2026-04-17 WT-A bad-data audit retained `480` `visomaster_hints` rows and `202` `visomaster_hints_teams` rows as "weak-signal" (not method-faithful supervision; not deployment-distribution; not clean target-domain truth) — does the retained subset carry any positive training signal at all? Three explicit arms were proposed: (`no_hints` → `hints_only` → `hints + teams_hints`).

## Initial belief

At April-17 merge, the team's stance was that the retained hint rows might still carry weak signal worth recovering, since the policy preserved them rather than deleting them outright (`docs/VISOMASTER_BAD_DATA_POLICY_PLAN_IMPACT_2026-04-17.md:230-242`). The plan called for an explicit three-arm ablation so that "the retained weak-signal subset is actually helping, hurting, or just adding noise" could be answered by training, not by judgment (`VISOMASTER_BAD_DATA_POLICY_PLAN_IMPACT_2026-04-17.md:230-242`). WT-B was the dedicated worktree for this experiment.

## What changed our mind

- **2026-04-17 WT-B claim and immediate block (`docs/relaunch_handoffs/TASK_BOARD_2026-04-17.md:35-47`).** The board kept WT-B blocked behind WT-A's lane-semantics freeze ("WT-B stays blocked until WT-A merges a lane-semantics freeze," `TASK_BOARD_2026-04-17.md:13-15`). Once WT-A merged 19:05 CEST, WT-B was unblocked at 20:36 CEST.
- **2026-04-17 WT-B draft-only verdict (`docs/relaunch_handoffs/WT-B_2026-04-17.md:14-87`, commits `733b343` → `44743be` → `5533985` → `1fa0360`).** WT-B inspected the tracked tree and reported it could not honestly deliver runnable hints arms because (1) `DeepfakeBench/training/data/sources/{combined_paired,visomaster}.py` were not part of the tracked checkout in this repo layout, and (2) the runtime loader did not consume the April-17 policy packet (`WT-B_2026-04-17.md:42-55`). WT-B shipped three draft-only YAML manifests (`R13_WTB1/2/3_*_draft_only.yaml`) plus a runbook (`R13_WTB_WEAK_SIGNAL_DRAFT_RUNBOOK_2026-04-17.md`), updated the task-board status to `blocked` rather than papering it over, and explicitly refused to take ownership of the WT-A-side loader gap (`WT-B_2026-04-17.md:74-87`). The honest same-day framing was *"draft-only weak-signal plan blocked on source integration"* (`WT-B_2026-04-17.md:48-55`).
- **2026-04-19 WT-B unblock and runnable package (memory `wt_a_to_wt_b_handoff` chain, `docs/POLICY_AWARE_SOURCE_REDESIGN_2026-04-17.md:23-31`).** Slice-2 territory: a `data/` ignore-rule exception for `*.py` files made the runtime loader committable; the canonical policy packet landed under `training/policy/visomaster_bad_data/`. WT-B's runnable package (`combined_paired.visomaster_hints`, `combined_paired.visomaster_hints_teams`, `combined_paired.teams.apply_bad_data_policy`) merged 2026-04-19 12:31 CEST as commit `f9303eb` ([WT-infrastructure packet retro](../packets/WT_infrastructure.md), "Conclusions drawn in-session — WT-B").
- **RLP1 verdict (Slice 2 ownership): hints fail.** Per [WT_infrastructure packet retro](../packets/WT_infrastructure.md), section "WT-B hypothesis failed cleanly": RLP1 showed hints as a net drag (control `RLP1_01` beat `RLP1_02`/`RLP1_03` across all 8 matched holdout checkpoints). RLP2 dropped hints from the main slate; RLP3+ kept hints off by default.
- **2026-04-18 WT-B re-investigation finalized (`docs/relaunch_handoffs/WT-B_2026-04-18.md:25-90`).** The re-arc explicitly added a fourth requirement to the three-arm ablation: *"the direct `teams` lane must also be cleaned by policy, otherwise the retained Teams-played hint rows are silently duplicated in the control arm and the family stops being honest"* (`WT-B_2026-04-18.md:13-22`). Resulting deliverables: runtime support for `combined_paired.visomaster_hints` / `visomaster_hints_teams` / `teams.apply_bad_data_policy` (`combined_paired.py`, `utils/grouping.py`, `data/augmentations/pipelines.py`); fresh runnable configs `R13_WTB{1,2,3}_*.yaml` plus `R13_SMOKE_WTB3_*.yaml`; tracked policy packet at `training/policy/visomaster_bad_data/`; `.gitignore` exception so `data/sources/*.py` becomes reviewable. Targeted WT-B test pass green (`pytest test_phase4_family_pipeline.py -k "wt_b OR family_router OR group_key OR hint OR teams_passthrough"`). The runbook lives at `docs/R13_WTB_WEAK_SIGNAL_RUNBOOK_2026-04-18.md` (the 2026-04-17 draft was marked superseded but kept).
- **2026-04-19 WT-B remote smoke proof (`docs/relaunch_handoffs/WT_B_AND_NEW_DATA_READINESS_2026-04-19.md:57-98`).** Two Vertex jobs succeeded under `JOB_STATE_SUCCEEDED`: startup smoke (Vertex `9012653657048481792`, W&B `8bjcadyr`, 2-step run with `visomaster_hints_fake=480`, `visomaster_hints_teams_fake=202`) and integration smoke (Vertex `2921535161029885952`, W&B `g6f8fovi`, 100-step run reaching `val_in_dist/overall/auc=0.97534`). Lane-count proof confirms the runtime loader emits the explicit hint lanes with the WT-A retained counts.
- **2026-04-19 hints + proper-data overlap configs land (`docs/relaunch_handoffs/NEW_DATA_LOADER_AND_EXPERIMENT_HANDOFF_2026-04-19.md:419-428`, commit `77facfc`).** First combined `WTB3 + proper_data` configs exist locally (`R13_WTB3_with_proper_data_unenhanced_provisional.yaml`, `R13_WTB3_with_proper_data_full_snapshot_provisional.yaml`); these become the seeds of `RLP1_04`/`RLP1_05`. The branch lays out hints + proper-data orthogonally, so RLP1's hint ladder isolates hint signal even with proper-data superimposed.
- **2026-04-20 RLP1 hints-arm results — control wins at every comparison point (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_LIVE_MONITORING_2026-04-20.md:108-123, 405-421`, [RLP1.md:38-58](../packets/RLP1.md)).** Best-checkpoint composites: `RLP1_01` (no hints) `0.98915` > `RLP1_02` (hints_only) `0.98524` > `RLP1_03` (hints + teams_hints) `0.98442`. At all 8 matched holdout checkpoints `RLP1_01` beat `RLP1_02` by `0.0077` to `0.0115` AUC. `RLP1_03` beat `RLP1_02` only at 1 of 8 matched holdouts (extra Teams hints actively unhelpful, not just neutral). Live monitoring noted that *"`RLP1_03` has the strongest current `val_in_dist` AUC in the FT family (`0.99054`) but a much weaker current holdout AUC (`0.98258`)"* — Teams-hints learn something on the easier in-dist slice without helping the harder holdout slice (`R13_RELAUNCH_PACKET1_LIVE_MONITORING_2026-04-20.md:391-401`).
- **2026-04-21 RLP2 drops hints from main slate (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET2_EXPERIMENT_PLAN_2026-04-21.md:67-74`).** Packet-2 explicitly excludes hints: *"No hints in the main packet. Packet 1 already failed to justify them."* RLP2's `RLP2_01` (no-hints fresh control) and `RLP2_03` (no-hints + unenhanced + teams_enhanced) confirm the hint family is dead even with proper-data added. The April-21 source-of-truth handoff (`R13_RELAUNCH_TRAINING_SOURCE_OF_TRUTH_2026-04-21.md:99-100`) codifies it: *"Packet 1 did **not** justify the hint ladder."*

## Current stance (2026-04-29)

The hints-as-supervision hypothesis is closed cleanly negative. WT-B's negative result is load-bearing because it eliminated a candidate lever without re-litigation. The retained `visomaster_hints` / `visomaster_hints_teams` rows have value as **policy bookkeeping** (so the data lineage is auditable) but not as **training signal**: the runtime loader supports the lanes, default RLP3+ configs leave them off, and the suite-side `path_exclude_contains: ["/visomaster_"]` on `teams_ood_fake` (the gate-alignment hygiene rule from RLP6) makes the lanes cleanly hint-only on both training and evaluation sides.

The three-arm ablation pattern (`no_hints` → `hints_only` → `hints + teams_hints`) inherited from the WT-A→WT-B handoff is the canonical example of "test the policy-retained subset before relying on it." The Slice-1 "draft-only blocked on source integration" pattern is the canonical honest-handoff template when a worktree dependency is not yet met.

## Packet timeline

- [WT-infrastructure](../packets/WT_infrastructure.md) — WT-B claimed 2026-04-17 20:36 CEST, draft-only merged 20:41 CEST, runnable package merged 2026-04-19 12:31 CEST.
- [RLP1](../packets/RLP1.md) — three arms tested as `RLP1_01` (control), `RLP1_02` (`hints_only`), `RLP1_03` (`hints + teams_hints`). Control wins across all 8 matched holdout checkpoints; hints-as-supervision closed.
- [RLP2](../packets/RLP2.md) — hints dropped from the main slate.

## Evidence locations

- `docs/relaunch_handoffs/WT-B_2026-04-17.md:14-87` — the draft-only verdict and blocking-dependency framing
- `docs/relaunch_handoffs/WT-B_2026-04-18.md:25-90` — the April-18 finalization (Slice 2 reading)
- `docs/relaunch_handoffs/WT_B_AND_NEW_DATA_READINESS_2026-04-19.md:57-118` — startup + integration smoke proofs (Slice 2 reading)
- `docs/relaunch_handoffs/TASK_BOARD_2026-04-17.md:13-49` — coordination + dependency rule + claim log
- `docs/VISOMASTER_BAD_DATA_POLICY_PLAN_IMPACT_2026-04-17.md:230-242` — the original three-arm ablation framing
- `docs/POLICY_AWARE_SOURCE_REDESIGN_2026-04-17.md:23-31` — April-19 unblock notes
- `docs/R13_WTB_WEAK_SIGNAL_DRAFT_RUNBOOK_2026-04-17.md` — superseded draft runbook (WT-B 04-17)
- `docs/R13_WTB_WEAK_SIGNAL_RUNBOOK_2026-04-18.md` — final runbook (Slice 2 reading)
- Memory: *(none yet — WT-B's outcome lives in `WT_infrastructure.md` packet retro and the `gate_alignment_story` thread; no auto-memory entry has been written for the hints arc itself)*
- Commits: `733b343`, `44743be`, `5533985`, `1fa0360`, `f9303eb`

## Open loops

### Open loop: wt-b-april-17-source-integration
status: resolved
severity: medium
first_seen: 2026-04-17
last_verified: 2026-04-29
close_criterion: WT-B has a runnable training package (not just draft-only YAMLs) that consumes the April-17 policy and a smoke-passed launcher

The April-17 attempt shipped draft-only because tracked-tree source integration was missing (`docs/relaunch_handoffs/WT-B_2026-04-17.md:74-87`). Resolution shipped 2026-04-19 12:31 CEST when the data-loader package was committed alongside a relaxed `.gitignore` rule and a tracked policy packet (commit `f9303eb`; see [WT_infrastructure packet retro](../packets/WT_infrastructure.md)). Slice 2 owns the actual readout from the smoke runs.

### Open loop: hints-as-supervision-hypothesis
status: resolved
severity: medium
first_seen: 2026-04-17
last_verified: 2026-04-29
close_criterion: an explicit ablation produces evidence whether the retained `visomaster_hints` / `visomaster_hints_teams` rows help, hurt, or are neutral as training signal

Closed negative by [RLP1](../packets/RLP1.md) (Slice 2): control `RLP1_01` beat hints arms `RLP1_02`/`RLP1_03` across all 8 matched holdout checkpoints. Hints removed from the main slate on RLP2 onward. Result is load-bearing — eliminates a candidate lever without re-litigation. See [WT_infrastructure packet retro](../packets/WT_infrastructure.md) "WT-B hypothesis failed cleanly".
