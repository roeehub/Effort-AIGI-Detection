# R13 Relaunch Training Source Of Truth

**Date:** 2026-04-21  
**Purpose:** top-level signal file for any agent picking up the training thread
from the April 6 target-domain upgrade plan through the current packet-2 state.

## Short Answer

There is **not** one older file that is sufficient on its own anymore.

The closest partial anchors are:

- the April 6 plan for the original strategy
- the April 17 relaunch report for the post-policy-correction interpretation
- the April 19 review packet for the consolidated implementation checkpoint
- the April 21 packet docs for the current experiment train of thought

That means the right answer now is:

- keep the old documents
- use **this file** as the top-level source-of-truth index
- treat the linked documents below as the canonical stack

## The Canonical Stack

### 1. Strategic root

This is the original intent document. It defines the project-level goal shift:

- optimize for Teams / target-domain usefulness
- evaluate through the calibrated low-FP promotion contract
- stop over-trusting generic holdout stories

Primary file:

- [TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md](../TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md)

### 2. Relaunch interpretation root

This is the best single document for understanding why the April 17 relaunch
exists and how to think about the corrected data semantics:

- old bad VisoMaster data became `hints`, not clean supervision
- the direct Teams lane remains useful but is not perfectly clean
- proper target-domain data must be kept explicit
- decision / stability / augmentation work must stay honest and measured

Primary file:

- [TEAMS_EXPERIMENT_RELAUNCH_REPORT_2026-04-17.md](../TEAMS_EXPERIMENT_RELAUNCH_REPORT_2026-04-17.md)

### 3. Consolidated implementation / review checkpoint

This is the best single checkpoint for the April 17-19 implementation state:

- all `WT-*` tracks
- WT-B smoke closeout
- proper-data runtime integration
- code-review surfaces

Primary file:

- [RELAUNCH_UPGRADE_REVIEW_PACKET_2026-04-19.md](./RELAUNCH_UPGRADE_REVIEW_PACKET_2026-04-19.md)

### 4. Packet-1 evidence bridge

This is the best bridge from relaunch implementation into actual experiment
conclusions:

- what packet 1 tested
- what packet 1 taught us
- what was weak, what was promising, what should be carried forward

Primary file:

- [R13_RELAUNCH_PACKET1_RESULTS_AND_PLANNING_CONTEXT_2026-04-21.md](./R13_RELAUNCH_PACKET1_RESULTS_AND_PLANNING_CONTEXT_2026-04-21.md)

### 5. Current active packet

This is the current experiment-writing root for the next training wave:

- the next 6 FT runs
- why hints are out
- why unenhanced proper-data is the main bet
- how enhanced proper-data is being reintroduced more carefully
- how prediction stability is now being measured explicitly

Primary file:

- [R13_RELAUNCH_PACKET2_EXPERIMENT_PLAN_2026-04-21.md](./R13_RELAUNCH_PACKET2_EXPERIMENT_PLAN_2026-04-21.md)

## Current Training Thesis

This is the current train of thought the next agent should inherit.

- The project goal is still Teams / target-domain deployment quality, not
  generic holdout victory.
- `hints` are now a deliberately small weak-signal residue from the old bad
  VisoMaster story. They are not the new truth source.
- `proper_data` is the explicit new high-value target-domain path. It is the
  important new data axis.
- Packet 1 showed that the clean no-hints FT baseline (`RLP1_01`) is still the
  burden-of-proof control.
- Packet 1 also showed that unenhanced proper-data (`RLP1_04`) is the strongest
  genuinely new signal.
- Packet 1 did **not** justify the hint ladder, the WT-C sidecars, or the full
  proper-data jump at the live packet-1 scale.
- Packet 1 scratch is still a hedge, not the main current winner.
- Packet 2 is therefore built around:
  - refreshed no-hints control
  - no-hints plus unenhanced proper-data
  - moderate enhanced-proper follow-ons
  - explicit prediction-stability probes

## What Is Settled

These points should be treated as current ground truth unless a newer document
or live run disproves them.

- The April 6 target-domain orientation still stands.
- All `WT-*` relaunch tracks are merged on the branch.
- The old bad VisoMaster semantics were corrected rather than hand-waved.
- WT-B landed explicit `hints` and `hints_teams` runtime lanes.
- WT-F plus the April 19 follow-up landed an explicit
  `combined_paired.proper_data` training path.
- The new HDTF / quickclips buckets belong on `proper_data`, not on `hints`.
- Packet 1 established that the strongest new data idea is not “more hints” but
  “explicit proper-data.”
- Generic `stability_lambda` is not the preferred stability lever anymore.
- Prediction stability now needs to be measured explicitly, not implied.

## What Is Still Provisional

- Proper-data counts remain tied to the latest completed artifact build.
- If the builder was interrupted, rerun it before trusting the proper-data
  manifests or launching new proper-data experiments.
- Packet-1 tail status should be verified live before making any final
  packet-wide claims:
  - `RLP1_03`
  - `RLP1_08`
- Packet-2 launch status should be verified live rather than assumed from docs.

## Recommended Read Order

### Fast path for a new training agent

If the agent needs the shortest path that still preserves the current train of
thought, read in this order:

1. this file
2. [TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md](../TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md)
3. [TEAMS_EXPERIMENT_RELAUNCH_REPORT_2026-04-17.md](../TEAMS_EXPERIMENT_RELAUNCH_REPORT_2026-04-17.md)
4. [RELAUNCH_UPGRADE_REVIEW_PACKET_2026-04-19.md](./RELAUNCH_UPGRADE_REVIEW_PACKET_2026-04-19.md)
5. [R13_RELAUNCH_PACKET1_RESULTS_AND_PLANNING_CONTEXT_2026-04-21.md](./R13_RELAUNCH_PACKET1_RESULTS_AND_PLANNING_CONTEXT_2026-04-21.md)
6. [R13_RELAUNCH_PACKET2_EXPERIMENT_PLAN_2026-04-21.md](./R13_RELAUNCH_PACKET2_EXPERIMENT_PLAN_2026-04-21.md)

### Deeper operational read

If the agent needs to touch implementation or interpret runtime behavior, then
add these:

1. [TASK_BOARD_2026-04-17.md](./TASK_BOARD_2026-04-17.md)
2. [WT_B_AND_NEW_DATA_READINESS_2026-04-19.md](./WT_B_AND_NEW_DATA_READINESS_2026-04-19.md)
3. [NEW_DATA_LOADER_AND_EXPERIMENT_HANDOFF_2026-04-19.md](./NEW_DATA_LOADER_AND_EXPERIMENT_HANDOFF_2026-04-19.md)
4. [WT-C_2026-04-17.md](./WT-C_2026-04-17.md)
5. [WT-F_2026-04-17.md](./WT-F_2026-04-17.md)
6. [R13_RELAUNCH_PACKET1_MONITORING_HANDOFF_2026-04-20.md](./R13_RELAUNCH_PACKET1_MONITORING_HANDOFF_2026-04-20.md)
7. [R13_RELAUNCH_PACKET1_LIVE_MONITORING_2026-04-20.md](./R13_RELAUNCH_PACKET1_LIVE_MONITORING_2026-04-20.md)

## Topic Map

Use this map if the next agent is only touching one area.

### Data semantics / honesty

- [WT-A_2026-04-17.md](./WT-A_2026-04-17.md)
- [WT-B_2026-04-18.md](./WT-B_2026-04-18.md)
- [VISOMASTER_BAD_DATA_POLICY_PLAN_IMPACT_2026-04-17.md](../VISOMASTER_BAD_DATA_POLICY_PLAN_IMPACT_2026-04-17.md)
- [POLICY_AWARE_SOURCE_REDESIGN_2026-04-17.md](../POLICY_AWARE_SOURCE_REDESIGN_2026-04-17.md)

### Proper-data / new buckets

- [WT-F_2026-04-17.md](./WT-F_2026-04-17.md)
- [PROPER_DATA_SCHEMA_AND_FUTURE_MANIFESTS_2026-04-17.md](../PROPER_DATA_SCHEMA_AND_FUTURE_MANIFESTS_2026-04-17.md)
- [NEW_DATA_LOADER_AND_EXPERIMENT_HANDOFF_2026-04-19.md](./NEW_DATA_LOADER_AND_EXPERIMENT_HANDOFF_2026-04-19.md)
- [WT_B_AND_NEW_DATA_READINESS_2026-04-19.md](./WT_B_AND_NEW_DATA_READINESS_2026-04-19.md)

### Evaluation / promotion / decision system

- [WT-D_2026-04-17.md](./WT-D_2026-04-17.md)
- [WT-E_2026-04-17.md](./WT-E_2026-04-17.md)
- [WT_D_MINIMAL_BASELINE_PACKET_2026-04-17.md](../WT_D_MINIMAL_BASELINE_PACKET_2026-04-17.md)
- [TEAMS_DECISION_SYSTEM_REINVESTIGATION_2026-04-17.md](../TEAMS_DECISION_SYSTEM_REINVESTIGATION_2026-04-17.md)

### Augmentation / nuisance / stability

- [WT-C_2026-04-17.md](./WT-C_2026-04-17.md)
- [TEAMS_AUGMENTATION_SIDE_TRACK_HANDOFF_2026-04-06.md](../TEAMS_AUGMENTATION_SIDE_TRACK_HANDOFF_2026-04-06.md)
- [SPATIAL_AUGMENTATION_PROPOSAL_2026-04-14.md](../SPATIAL_AUGMENTATION_PROPOSAL_2026-04-14.md)

### Packet execution / live monitoring

- [R13_RELAUNCH_PACKET1_MONITORING_HANDOFF_2026-04-20.md](./R13_RELAUNCH_PACKET1_MONITORING_HANDOFF_2026-04-20.md)
- [R13_RELAUNCH_PACKET1_LIVE_MONITORING_2026-04-20.md](./R13_RELAUNCH_PACKET1_LIVE_MONITORING_2026-04-20.md)
- [R13_RELAUNCH_PACKET1_RESULTS_AND_PLANNING_CONTEXT_2026-04-21.md](./R13_RELAUNCH_PACKET1_RESULTS_AND_PLANNING_CONTEXT_2026-04-21.md)
- [R13_RELAUNCH_PACKET2_EXPERIMENT_PLAN_2026-04-21.md](./R13_RELAUNCH_PACKET2_EXPERIMENT_PLAN_2026-04-21.md)

## Current Source Of Truth Verdict

If a new agent asks “what file should I trust first?” the answer should now be:

1. [R13_RELAUNCH_TRAINING_SOURCE_OF_TRUTH_2026-04-21.md](./R13_RELAUNCH_TRAINING_SOURCE_OF_TRUTH_2026-04-21.md)
2. [TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md](../TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md)
3. [R13_RELAUNCH_PACKET2_EXPERIMENT_PLAN_2026-04-21.md](./R13_RELAUNCH_PACKET2_EXPERIMENT_PLAN_2026-04-21.md)

Use the April 6 plan for strategic intent, use this file for navigation, and
use the packet-2 plan for the current active experiment wave.

## What A New Agent Should Do First

1. Read the fast-path stack above.
2. Verify the current live W&B state rather than assuming docs are still fresh.
3. Verify whether packet 2 has already been launched.
4. Verify whether the proper-data builder last completed cleanly.
5. Continue from the current packet-2 logic unless new live evidence clearly
   invalidates it.
