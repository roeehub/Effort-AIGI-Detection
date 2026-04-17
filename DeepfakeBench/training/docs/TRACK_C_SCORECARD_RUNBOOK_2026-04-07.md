# Track C Scorecard Runbook

**Date:** April 7, 2026  
**Scope:** frozen Teams target-domain scorecard execution, checkpoint aliasing, and reusable comparison outputs

## Goal

Turn the frozen Teams target-domain manifest into a repeatable comparison table for:

- `S0` baseline freezing
- future `Track A` candidate evaluation
- later FP32 vs INT8 retention checks

Current operator decision for the April 11, 2026 close-out:

- continue Track C in FP32 only for now
- defer INT8 until later
- treat the current best live Track A checkpoint as the active `TRACK_A_CANDIDATE`

The current ready-to-run assets are:

- concrete local suite manifest:
  - `DeepfakeBench/training/arena/target_domain_suites.teams_manifest.frozen_2026-04-06.yaml`
- generated breakdown suite manifest:
  - `DeepfakeBench/training/arena/target_domain_suites.teams_manifest.breakdown_2026-04-07.yaml`
- suite-manifest generator:
  - `DeepfakeBench/training/arena/build_teams_target_domain_suites.py`
- checkpoint-map template:
  - `DeepfakeBench/training/arena/checkpoint_maps/teams_target_domain.template.yaml`
- seeded FP32 candidate map:
  - `DeepfakeBench/training/arena/checkpoint_maps/teams_target_domain.seed_candidates_2026-04-07.yaml`
- one-command wrapper:
  - `DeepfakeBench/training/arena/run_teams_target_domain_scorecard.sh`
- Vertex launcher:
  - `DeepfakeBench/training/arena/launch_target_domain_scorecard.sh`
- frozen manifest:
  - `DeepfakeBench/training/arena/manifests/teams_target_domain_manifest_2026-04-06_frozen.json`

## What Changed In Track C

The target-domain sequential runner now supports:

- custom checkpoint aliases through `--checkpoint_map`
- `--checkpoints ALL` to expand every available alias in the map
- scorecard export outputs:
  - `--scorecard_csv`
  - `--scorecard_wide_csv`
  - `--scorecard_delta_csv`
  - `--scorecard_json`

Operational meaning:

- you are no longer limited to `FT1..FT8`
- baseline aliases like `R12_G_FP32`, `R13_FT2_FP32`, and `TRACK_A_CANDIDATE` are now valid
- one run can emit both the underlying detailed reports and:
  - a compact checkpoint-by-suite table
  - an FP32-vs-INT8 delta table for matched alias pairs

Even though INT8 export is wired, the current working path is FP32-only until
real INT8 artifacts exist and become worth comparing.

The new suite generator now also turns the frozen manifest into a wider
breakdown manifest that covers:

- core real slices
- fake family slices
- fake method slices

Current limitation:

- the frozen manifest still collapses VisoMaster enhanced rows to
  `visomaster_enhanced_macro`, so true per-enhancer scorecard rows are still
  blocked on richer provenance than the current frozen artifact carries.

## Recommended Alias Set

Use these aliases in the checkpoint map unless there is a strong reason not to:

- `R12_G_FP32`
- `R13_FT2_FP32`
- `TRACK_A_CANDIDATE`

For a wider historical readout, keep `R13_FT1_FP32` and `R13_FT3_FP32` in the
map as optional extra baselines.

## Step 1: Fill The Checkpoint Map

Start from:

- `DeepfakeBench/training/arena/checkpoint_maps/teams_target_domain.template.yaml`

Replace the placeholder `gs://...pth` values with real checkpoint paths.

If you want a faster starting point, copy or edit:

- `DeepfakeBench/training/arena/checkpoint_maps/teams_target_domain.seed_candidates_2026-04-07.yaml`

That seeded map already includes repo-known FP32 entries for:

- `R12_G_FP32`
- `R13_FT1_FP32`
- `R13_FT2_FP32`
- `R13_FT3_FP32`
- `TRACK_A_CANDIDATE`

The seeded map is now runnable as-is for the current FP32-only Track C path.

## Step 2: Dry Run The Scorecard

From the training root:

```bash
cd DeepfakeBench/training

bash arena/run_teams_target_domain_scorecard.sh \
  --checkpoint-map arena/checkpoint_maps/teams_target_domain.seed_candidates_2026-04-07.yaml \
  --checkpoints R12_G_FP32,R13_FT2_FP32,TRACK_A_CANDIDATE \
  --dry-run
```

Or run every alias currently present in the map:

```bash
cd DeepfakeBench/training

bash arena/run_teams_target_domain_scorecard.sh \
  --checkpoint-map arena/checkpoint_maps/teams_target_domain.seed_candidates_2026-04-07.yaml \
  --checkpoints ALL \
  --dry-run
```

If you want the wider fake-family + fake-method breakdown, switch the suite
manifest:

```bash
cd DeepfakeBench/training

bash arena/run_teams_target_domain_scorecard.sh \
  --checkpoint-map arena/checkpoint_maps/teams_target_domain.seed_candidates_2026-04-07.yaml \
  --suite-manifest arena/target_domain_suites.teams_manifest.breakdown_2026-04-07.yaml \
  --checkpoints ALL \
  --dry-run
```

Current practical compact comparison:

```bash
cd DeepfakeBench/training

bash arena/run_teams_target_domain_scorecard.sh \
  --checkpoint-map arena/checkpoint_maps/teams_target_domain.seed_candidates_2026-04-07.yaml \
  --checkpoints R12_G_FP32,R13_FT2_FP32,TRACK_A_CANDIDATE \
  --dry-run
```

## Step 3: Run The Real Scorecard

```bash
cd DeepfakeBench/training

bash arena/run_teams_target_domain_scorecard.sh \
  --checkpoint-map arena/checkpoint_maps/teams_target_domain.seed_candidates_2026-04-07.yaml \
  --checkpoints ALL
```

Current practical FP32-only comparison:

```bash
cd DeepfakeBench/training

bash arena/run_teams_target_domain_scorecard.sh \
  --checkpoint-map arena/checkpoint_maps/teams_target_domain.seed_candidates_2026-04-07.yaml \
  --checkpoints R12_G_FP32,R13_FT2_FP32,TRACK_A_CANDIDATE
```

If you want the real scorecard to run on Vertex using the freshly built image,
use:

```bash
cd DeepfakeBench/training

bash arena/launch_target_domain_scorecard.sh
```

That default Vertex launch now includes:

- `R12_G_FP32`
- `R13_FT1_FP32`
- `R13_FT2_FP32`
- `R13_FT3_FP32`
- `TRACK_A_CANDIDATE`

That launcher uses the image version from `VERSION`, loads the packaged suite
manifest + checkpoint map from `/workspace/arena/...`, and writes:

- detailed reports to:
  - `gs://training-job-outputs/test_results/teams_target_domain_scorecard/<job-name>/reports/`
- scorecards to:
  - `gs://training-job-outputs/test_results/teams_target_domain_scorecard/<job-name>/scorecards/`
- Vertex system output to:
  - `gs://training-job-outputs/vertex-output/<job-name>/`

For the wider family/method breakdown on Vertex:

```bash
cd DeepfakeBench/training

bash arena/launch_target_domain_scorecard.sh --breakdown
```

To regenerate the breakdown suite manifest from the frozen JSON:

Use the path form that the runner will later consume. For local training-root
runs, that means `arena/...`, not a repo-root-prefixed path.

```bash
cd DeepfakeBench/training

python arena/build_teams_target_domain_suites.py \
  --manifest arena/manifests/teams_target_domain_manifest_2026-04-06_frozen.json \
  --output arena/target_domain_suites.teams_manifest.breakdown_2026-04-07.yaml
```

## Outputs

Local scorecard outputs are written to:

- `DeepfakeBench/training/arena/scorecards/<timestamp>/`

Artifacts:

- `scorecard.csv`
  - long-form one-row-per-checkpoint-per-suite table
- `scorecard.wide.csv`
  - wide comparison table, one row per checkpoint
- `scorecard.int8_delta.csv`
  - one row per matched `*_FP32` / `*_INT8` checkpoint-pair per suite
- `scorecard.json`
  - machine-readable long + wide + pair-delta payload

Validation detailed reports still land under the configured GCS output folder:

- default:
  - `gs://training-job-outputs/test_results/target_domain_validation`

## How To Read The Scorecard

Primary suite metric at the default `0.5` threshold:

- real-only suites:
  - `real_fpr_at_0p5`
- fake-only suites:
  - `fake_recall_at_0p5`
- mixed suites:
  - `accuracy_at_0p5`

Secondary fields are also exported:

- `n_videos`
- `accuracy_at_0p5`
- `mean_prob`
- `p50_prob`
- `p90_prob`
- confusion counts (`tn`, `fp`, `fn`, `tp`)

Pair-delta rows include:

- `score_metric_delta_int8_minus_fp32`
- `score_metric_directional_delta`
  - positive means INT8 improved or preserved the primary metric direction
  - negative means INT8 regressed
- `accuracy_at_0p5_delta_int8_minus_fp32`

## Current Practical Use

Right now the most useful comparison is:

1. freeze the current FP32-only finalist table with `R12_G_FP32`, one strong post-`R12` enhanced reference, and `TRACK_A_CANDIDATE`
2. use `R13_FT2_FP32` as the default post-`R12` enhanced reference unless there is a reason to keep all three FT baselines
3. use the breakdown suite manifest when you need family/method attribution, not just the compact freeze table
4. revisit INT8 only after the FP32 ranking is stable enough to be worth quantizing

## Latest Compact Result

Latest Track A compact comparison:

- display name: `td-scorecard-compact-20260410-132025`
- compared:
  - `R12_G_FP32`
  - `TRACK_A_CANDIDATE`
- scorecards:
  - `gs://training-job-outputs/test_results/teams_target_domain_scorecard/td-scorecard-compact-20260410-132025/scorecards/`

Key readout at threshold `0.5`:

- real Teams objective regressed for `TRACK_A_CANDIDATE` versus `R12_G`:
  - `teams_real_all_dev` FPR:
    - `R12_G`: `0.1980`
    - `TRACK_A_CANDIDATE`: `0.2355`
  - `teams_real_poor_quality_dev` FPR:
    - `R12_G`: `0.1679`
    - `TRACK_A_CANDIDATE`: `0.2319`
  - `teams_real_all_lockbox` FPR:
    - `R12_G`: `0.6334`
    - `TRACK_A_CANDIDATE`: `0.7252`
- target fake recall improved for `TRACK_A_CANDIDATE`:
  - `teams_fake_all_dev` recall:
    - `R12_G`: `0.8510`
    - `TRACK_A_CANDIDATE`: `0.8896`
  - `visomaster_enhanced_macro_dev` recall:
    - `R12_G`: `0.4527`
    - `TRACK_A_CANDIDATE`: `0.7400`
- mixed family detail:
  - `deeplive_enhanced_dev` still favors `R12_G`:
    - `0.9706` vs `0.9138`
  - `teams_capture_cam_test_dev`, `teams_capture_pc_generator_dev`, and `teams_capture_test_cam_dev` are unchanged or effectively tied

Practical conclusion:

- `TRACK_A_CANDIDATE` clearly improves the known VisoMaster-enhanced / target-fake weakness
- `TRACK_A_CANDIDATE` does not beat `R12_G` on the primary real-Teams false-positive objective
- on the current compact Track C lane, this is a tradeoff result, not a clean promotion result
- run the breakdown suite only if we need finer attribution before deciding what to tune next

## Remaining Gap

This runbook completes the reusable scorecard lane.

What still remains outside this doc:

- deciding whether the current FP32 tradeoff is good enough to justify Track A follow-on tuning
- producing the actual INT8 checkpoint artifacts later if quantized retention becomes worth checking
- reading the scorecard and promoting finalists
- richer VisoMaster enhanced provenance if per-enhancer rows become mandatory
