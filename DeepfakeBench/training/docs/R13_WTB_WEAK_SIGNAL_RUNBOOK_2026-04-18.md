# WT-B Weak-Signal Runbook

## Status

WT-B now has:

- a runnable three-arm family
- a dedicated smoke config for the highest-coverage hint path
- shared discovery-cache wiring for DeepLive, VisoMaster hints, and Teams passthrough

Configs:

- `DeepfakeBench/training/experiments/phase2_round13/R13_WTB1_weak_signal_no_hints.yaml`
- `DeepfakeBench/training/experiments/phase2_round13/R13_WTB2_weak_signal_hints_only.yaml`
- `DeepfakeBench/training/experiments/phase2_round13/R13_WTB3_weak_signal_hints_plus_teams_hints.yaml`
- `DeepfakeBench/training/experiments/phase2_round13/R13_SMOKE_WTB3_weak_signal_hints_plus_teams_hints.yaml`

## Family Shape

All three real arms keep the `R13_FT11_trackA_scorecard_base` schedule, base
checkpoint, deeplive mix, df40 mix, and clean direct Teams lane.

The intended variable is retained weak-signal exposure:

- `WTB1`: no explicit hint lanes
- `WTB2`: `visomaster_hints` only
- `WTB3`: `visomaster_hints` plus `visomaster_hints_teams`

## Runtime Contract

The training runtime supports two explicit weak-signal source blocks:

- `combined_paired.visomaster_hints`
- `combined_paired.visomaster_hints_teams`

WT-B also requires `combined_paired.teams.apply_bad_data_policy: true` so the
direct Teams lane is partitioned cleanly and does not silently duplicate the
retained Teams-played hint rows.

For honesty, the WT-B configs disable:

- `combined_paired.visomaster`
- `combined_paired.visomaster_teams_enhanced`

## Tracked Policy Packet

The WT-B configs point at:

- `DeepfakeBench/training/policy/visomaster_bad_data/VISOMASTER_BAD_DATA_POLICY_MANIFEST_2026-04-17.csv`
- `DeepfakeBench/training/policy/visomaster_bad_data/VISOMASTER_BAD_DATA_POLICY_SUMMARY_2026-04-17.json`

The policy report and upload audit live alongside that bundle in the same
tracked directory. `viewer/visomaster_policy.py` prefers this tracked location
before falling back to legacy local copies.

## Smoke Gate

Use the smoke config first:

- `DeepfakeBench/training/experiments/phase2_round13/R13_SMOKE_WTB3_weak_signal_hints_plus_teams_hints.yaml`

`WTB3` is the right smoke target because it exercises the full WT-B runtime
surface in one launch:

- `visomaster_hints`
- `visomaster_hints_teams`
- `teams.apply_bad_data_policy`

Smoke for this path should go through the standard launcher, not a local
`python train_sweep.py` invocation:

```bash
cd DeepfakeBench/training
./launch_experiment.sh -y phase2r13-experiments asia-southeast1 \
  experiments/phase2_round13/R13_SMOKE_WTB3_weak_signal_hints_plus_teams_hints.yaml
```

Launch only after rebuilding or otherwise publishing an image that contains
this committed WT-B runtime package and the new YAMLs. The remote job uses the
published image; it does not sync the current working tree.

Smoke pass criteria:

- nonzero `visomaster_hints_samples` in logs
- nonzero `visomaster_hints_teams_samples` in logs
- nonzero clean direct Teams samples after policy filtering
- training reaches `max_train_steps: 100`
- checkpoint write succeeds under `gs://training-job-outputs/phase2r13_experiments/`

Do not treat the smoke checkpoint as a quality result.

## Verification

Targeted WT-B checks:

- `python3 -m py_compile DeepfakeBench/training/utils/grouping.py DeepfakeBench/training/data/augmentations/pipelines.py DeepfakeBench/training/data/sources/combined_paired.py DeepfakeBench/training/viewer/visomaster_policy.py DeepfakeBench/training/tests/test_phase4_family_pipeline.py`
- `pytest DeepfakeBench/training/tests/test_phase4_family_pipeline.py::test_group_key_mapping_representative_cases DeepfakeBench/training/tests/test_phase4_family_pipeline.py::test_quality_targeted_family_router_forward_pass DeepfakeBench/training/tests/test_phase4_family_pipeline.py::test_quality_targeted_family_router_registers_hint_families DeepfakeBench/training/tests/test_phase4_family_pipeline.py::test_wt_b_policy_partition_helpers_split_clean_and_hint_lanes DeepfakeBench/training/tests/test_phase4_family_pipeline.py::test_discover_teams_passthrough_samples_uses_fresh_cache_without_gcs DeepfakeBench/training/tests/test_phase4_family_pipeline.py::test_iterate_wt_b_hint_lane_samples_keep_explicit_source_names -q`

Current proof boundary:

- the targeted WT-B subset is green
- `pytest DeepfakeBench/training/tests/test_phase4_family_pipeline.py -q` still has unrelated failures outside WT-B

## Launch Sequence

1. Build and publish the training image from the current committed tree, or use another launch path that definitely includes current local code.
2. Run the smoke config above via `./launch_experiment.sh`.
3. If the smoke passes, launch the real family:

```bash
cd DeepfakeBench/training
./launch_experiment.sh -y phase2r13-experiments asia-southeast1 \
  experiments/phase2_round13/R13_WTB1_weak_signal_no_hints.yaml

cd DeepfakeBench/training
./launch_experiment.sh -y phase2r13-experiments asia-southeast1 \
  experiments/phase2_round13/R13_WTB2_weak_signal_hints_only.yaml

cd DeepfakeBench/training
./launch_experiment.sh -y phase2r13-experiments asia-southeast1 \
  experiments/phase2_round13/R13_WTB3_weak_signal_hints_plus_teams_hints.yaml
```

The shared discovery-cache paths are intended to let the smoke prime the Teams
and hint discovery state before the real launches.
