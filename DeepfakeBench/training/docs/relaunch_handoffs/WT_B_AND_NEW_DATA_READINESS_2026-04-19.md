# WT-B And New Data Readiness

## Scope

This note captures the current proven state for WT-B and the April 19
proper-data path before the first real experiment packet is planned.

Single consolidated review packet:

- `DeepfakeBench/training/docs/relaunch_handoffs/RELAUNCH_UPGRADE_REVIEW_PACKET_2026-04-19.md`

Detailed working handoff for the active proper-data task:

- `DeepfakeBench/training/docs/relaunch_handoffs/NEW_DATA_LOADER_AND_EXPERIMENT_HANDOFF_2026-04-19.md`

## What Is Already Landed

- explicit weak-signal lanes exist in the training runtime:
  - `combined_paired.visomaster_hints`
  - `combined_paired.visomaster_hints_teams`
- the direct `teams` lane can be policy-filtered so WT-B does not silently
  duplicate retained Teams-played hint rows
- tracked policy artifacts live under
  `DeepfakeBench/training/policy/visomaster_bad_data/`
- runnable WT-B family configs exist:
  - `R13_WTB1_weak_signal_no_hints.yaml`
  - `R13_WTB2_weak_signal_hints_only.yaml`
  - `R13_WTB3_weak_signal_hints_plus_teams_hints.yaml`
- launcher smoke configs exist for both WT-B and proper-data:
  - `R13_STARTUP_SMOKE_WTB3_weak_signal_hints_plus_teams_hints.yaml`
  - `R13_SMOKE_WTB3_weak_signal_hints_plus_teams_hints.yaml`
  - `R13_STARTUP_SMOKE_PROPER_DATA_WTF_PROVISIONAL.yaml`
- training-side proper-data loading now exists on the explicit
  `combined_paired.proper_data` path and consumes the WT-F provisional
  inventory/manifest artifacts without folding the data into legacy
  `visomaster`, `visomaster_hints`, or other weak-signal lanes
- discovery-cache wiring exists for:
  - DeepLive
  - VisoMaster hints
  - Teams passthrough
  - external VCD training reals
- the April 19 provisional WT-F artifacts exist in-repo:
  - `DeepfakeBench/training/arena/inventories/proper_visomaster_wave_2026_04_19_provisional.yaml`
  - `DeepfakeBench/training/arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json`
  - `DeepfakeBench/training/arena/target_domain_suites.proper_data_future.provisional_2026-04-19.yaml`

## Remote Proofs

### WT-B Startup Smoke

- config:
  `DeepfakeBench/training/experiments/phase2_round13/R13_STARTUP_SMOKE_WTB3_weak_signal_hints_plus_teams_hints.yaml`
- display name:
  `exp-R13_STARTUP_SMOKE_WTB3_weak_signal_hints_plus_teams_hints-20260419-125302`
- custom job id: `9012653657048481792`
- W&B run id: `8bjcadyr`
- Vertex state: `JOB_STATE_SUCCEEDED`
- runtime window:
  - start: `2026-04-19T11:01:25Z`
  - end: `2026-04-19T12:03:14Z`
- proof lines:
  - `MAX STEPS REACHED: 2/2`
  - `val_in_dist/overall/auc 0.4856`
  - checkpoint upload succeeded to
    `gs://training-job-outputs/phase2r13_experiments/8bjcadyr/...`

### WT-B Integration Smoke

- config:
  `DeepfakeBench/training/experiments/phase2_round13/R13_SMOKE_WTB3_weak_signal_hints_plus_teams_hints.yaml`
- display name:
  `exp-R13_SMOKE_WTB3_weak_signal_hints_plus_teams_hints-20260419-125308`
- custom job id: `2921535161029885952`
- W&B run id: `g6f8fovi`
- Vertex state: `JOB_STATE_SUCCEEDED`
- runtime window:
  - start: `2026-04-19T11:01:01Z`
  - end: `2026-04-19T12:23:56Z`
- proof lines:
  - `MAX STEPS REACHED: 100/100`
  - `val_in_dist/overall/auc 0.97534`
  - checkpoint upload succeeded to
    `gs://training-job-outputs/phase2r13_experiments/g6f8fovi/...`
  - family counts (all) included:
    - `visomaster_hints_fake: 480`
    - `visomaster_hints_teams_fake: 202`
    - `deeplive_teams_real: 1111`
    - `deeplive_teams_fake: 1111`
    - `external_real: 20`

### Proper-Data Startup Smoke

- config:
  `DeepfakeBench/training/experiments/phase2_round13/R13_STARTUP_SMOKE_PROPER_DATA_WTF_PROVISIONAL.yaml`
- display name:
  `exp-R13_STARTUP_SMOKE_PROPER_DATA_WTF_PROVISIONAL-20260419-194708`
- custom job id: `2064725331922649088`
- W&B run id: `bxay0n66`
- Vertex state: `JOB_STATE_SUCCEEDED`
- runtime window:
  - start: `2026-04-19T17:54:05Z`
  - end: `2026-04-19T17:58:07Z`
- proof lines:
  - `ProperData: enabled=True -> 128 samples`
  - `Proper-data lane counts (all): {'proper_visomaster_clean': 32, 'proper_visomaster_enhanced_clean': 32, 'proper_visomaster_enhanced_teams': 32, 'proper_visomaster_teams': 32}`
  - `Proper-data lane counts (train): {'proper_visomaster_clean': 29, 'proper_visomaster_enhanced_clean': 25, 'proper_visomaster_enhanced_teams': 25, 'proper_visomaster_teams': 29}`
  - `MAX STEPS REACHED: 2/2`
  - `val_in_dist/overall/auc 0.19444`
  - checkpoint upload succeeded to
    `gs://training-job-outputs/phase2r13_experiments/bxay0n66/...`

## What Is Still Provisional

- the current Teams propagation is still incomplete; per user report, roughly
  `30%` more Teams-propagated data is expected to land
- the April 19 proper-data inventory, manifest, suite, and smoke proof are
  valid for the current proof boundary but still provisional for count-sensitive
  experiment planning
- the current proper-data training loader keeps the exact `proper_*` lanes but
  tolerates clean-side ragged residue by taking the intersection of requested
  anchor indices available on both sides of a pair
- the current repo tree still needs commit closeout; the successful smoke used
  the built working tree image, not a finalized committed state

## Readiness Summary

Current state:

- WT-B code and configs: ready and remotely proven
- WT-B remote startup smoke: succeeded
- WT-B remote integration smoke: succeeded
- training-side proper-data loader: landed and remotely proven via startup smoke
- proper-data snapshot: provisional until Teams propagation finishes and the
  WT-F artifacts are regenerated
- experiment planning with the new data: can begin now

Recommendation for the first planning pass:

- keep the WT-B family as one axis
- keep `proper_data` as one explicit `off/on` axis
- do not explode the first packet into many proper-data sub-arms until the
  Teams snapshot stabilizes and the provisional artifacts are regenerated
