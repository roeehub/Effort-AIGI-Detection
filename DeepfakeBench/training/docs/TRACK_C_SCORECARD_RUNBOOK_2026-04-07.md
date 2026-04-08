# Track C Scorecard Runbook

**Date:** April 7, 2026  
**Scope:** frozen Teams target-domain scorecard execution, checkpoint aliasing, and reusable comparison outputs

## Goal

Turn the frozen Teams target-domain manifest into a repeatable comparison table for:

- `S0` baseline freezing
- future `Track A` candidate evaluation
- later FP32 vs INT8 retention checks

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
- baseline aliases like `R12_G_FP32`, `R12_G_INT8`, and `TRACK_A_CANDIDATE` are now valid
- one run can emit both the underlying detailed reports and:
  - a compact checkpoint-by-suite table
  - an FP32-vs-INT8 delta table for matched alias pairs

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
- `R12_G_INT8`
- `POST_R12_ENH_FP32`
- `POST_R12_ENH_INT8`
- `TRACK_A_CANDIDATE`

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

It still requires real INT8 and `TRACK_A_CANDIDATE` paths.

## Step 2: Dry Run The Scorecard

From the training root:

```bash
cd DeepfakeBench/training

bash arena/run_teams_target_domain_scorecard.sh \
  --checkpoint-map arena/checkpoint_maps/teams_target_domain.template.yaml \
  --checkpoints R12_G_FP32,R12_G_INT8 \
  --dry-run
```

Or run every alias currently present in the map:

```bash
cd DeepfakeBench/training

bash arena/run_teams_target_domain_scorecard.sh \
  --checkpoint-map arena/checkpoint_maps/teams_target_domain.template.yaml \
  --checkpoints ALL \
  --dry-run
```

If you want the wider fake-family + fake-method breakdown, switch the suite
manifest:

```bash
cd DeepfakeBench/training

bash arena/run_teams_target_domain_scorecard.sh \
  --checkpoint-map arena/checkpoint_maps/teams_target_domain.template.yaml \
  --suite-manifest arena/target_domain_suites.teams_manifest.breakdown_2026-04-07.yaml \
  --checkpoints ALL \
  --dry-run
```

## Step 3: Run The Real Scorecard

```bash
cd DeepfakeBench/training

bash arena/run_teams_target_domain_scorecard.sh \
  --checkpoint-map arena/checkpoint_maps/teams_target_domain.template.yaml \
  --checkpoints ALL
```

If you want the real scorecard to run on Vertex using the freshly built image,
use:

```bash
cd DeepfakeBench/training

bash arena/launch_target_domain_scorecard.sh
```

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

1. freeze `S0` with baseline FP32 + INT8
2. add one or two best post-R12 enhanced checkpoints
3. later append `TRACK_A_CANDIDATE` once the first full-length Track A checkpoint exists
4. use the breakdown suite manifest when you need family/method attribution, not just the compact freeze table

## Remaining Gap

This runbook completes the reusable scorecard lane.

What still remains outside this doc:

- producing the actual INT8 checkpoint artifacts if they do not already exist
- deciding which exact baseline and post-R12 checkpoints become the frozen `S0` set
- reading the scorecard and promoting finalists
- richer VisoMaster enhanced provenance if per-enhancer rows become mandatory
