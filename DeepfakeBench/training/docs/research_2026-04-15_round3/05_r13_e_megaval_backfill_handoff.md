# R13_E Mega-Eval Backfill Handoff

## Goal

Launch one additional Vertex job in parallel that backfills the missing `R13_E_BESTSOFAR` rows for the active mega-eval:

- active mega-eval job: `r13-best-megaval-20260413-155050`
- Vertex job id: `2766548001978580992`
- active reports root:
  `gs://training-job-outputs/test_results/r13_best_megaval/r13-best-megaval-20260413-155050/reports/`

The backfill job must produce report artifacts that can later be merged with the active mega-eval's report artifacts and used to rebuild a complete fixed-threshold diagnostic scorecard.

This task is about the broad diagnostic mega-eval only. It is not the calibrated promotion decision task.

## Executive Summary

`R13_E_BESTSOFAR` is not a training-run name. It is an evaluation alias that was pinned to a specific checkpoint object in the checkpoint map:

- alias: `R13_E_BESTSOFAR`
- stale path:
  `gs://training-job-outputs/phase2r13_experiments/14d5exx0/ood_composite_effort_20260412_step12000_auc0.9844_eer0.0369.pth`

That object existed when the evaluation maps were written, but it no longer exists now. The `R13_E` training run kept training and the trainer pruned older `ood_composite` checkpoints because `ood_composite_top_n_size` is `3`.

Result:

- the first two `R13_E` suite evaluations in the mega-eval succeeded
- all later `R13_E` suite evaluations fail immediately with a `404` when the runner tries to download the stale checkpoint object
- the rest of the mega-eval keeps moving for the other checkpoints

The practical fix is:

1. keep the alias name `R13_E_BESTSOFAR`
2. repoint it to a surviving substitute checkpoint from the same run
3. run an `R13_E`-only backfill job on the suites whose `R13_E` report artifacts are missing
4. later merge the live mega-eval reports and the backfill reports into one staging location
5. rebuild the fixed-threshold scorecards from the merged reports

## Current Situation

### 1. Active mega-eval

The active mega-eval was launched from:

- launcher: `DeepfakeBench/training/arena/launch_r13_best_megaval.sh`
- checkpoint map baked into launch:
  `DeepfakeBench/training/arena/checkpoint_maps/r13_best_eval_2026-04-13.yaml`
- suite manifest path passed at launch:
  `DeepfakeBench/training/arena/target_domain_suites.r13_best_megaval_2026-04-13.yaml`
- image at launch:
  `us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:1.3.180`

Important nuance:

- the checked-in suite YAML now includes `teams_fake_all_lockbox`
- the live mega-eval image was launched before that correction was baked into the image
- the active run is therefore still following the old effective 16-suite mega-eval lane, not the updated 17-suite checked-in file

As of the latest inspection before writing this handoff:

- the active mega-eval is still `RUNNING`
- the active mega-eval has `88` `*_videos_report.csv` artifacts
- checkpoint completion counts:
  - `R12_G_FP32`: `15`
  - `R13_A_STEP15500`: `15`
  - `R13_FT7_FP32`: `14`
  - `R13_FT8_FP32`: `14`
  - `R13_FT9_FP32`: `14`
  - `R13_FT10_FP32`: `14`
  - `R13_E_BESTSOFAR`: `2`
- the latest visible progress is `visomaster_all_sources_r13_a_step15500`

Interpretation:

- the mega-eval is still progressing for the non-`R13_E` checkpoints
- `R13_E` is the only checkpoint showing systematic repeated failure

### 2. Active shortlist promotion run

There is also a separate live shortlist run:

- job: `teams-promo-shortlist-20260415-171322`
- Vertex job id: `8519157654130524160`

This run is not the scope of this handoff. However, it uses the same stale `R13_E_BESTSOFAR` alias, so it is exposed to the same missing-object problem unless someone restores or repoints that alias path for future work.

## Root Cause

### 1. What `R13_E_BESTSOFAR` actually is

`R13_E_BESTSOFAR` is an evaluation alias in the checkpoint maps. It is not the underlying experiment name.

The underlying experiment is:

- `DeepfakeBench/training/experiments/phase2_round13/R13_E_trackA_teams_enhanced_realboost.yaml`

The alias was documented as:

- "best OOD-composite checkpoint available as of April 12, 2026 while that run is still live"

### 2. The exact failure

The first failing `R13_E` mega-eval suite logs:

- suite:
  `teams_real_lighting_extreme_dev`
- alias:
  `R13_E_BESTSOFAR`
- failure:
  `404 GET ... No such object`

The missing object is:

- `gs://training-job-outputs/phase2r13_experiments/14d5exx0/ood_composite_effort_20260412_step12000_auc0.9844_eer0.0369.pth`

The runner aborts that suite-checkpoint evaluation immediately after failing to download the checkpoint.

### 3. Why the object disappeared

The `R13_E` training config enables bounded `ood_composite` checkpoint retention:

- `ood_composite_enabled: true`
- `ood_composite_top_n_size: 3`

The trainer keeps only the best retained `ood_composite` checkpoints and deletes the worst retained object when the list grows past the configured top-N. This is exactly what the trainer code does.

So the active evaluation maps referenced a real object that later got pruned when the training run produced stronger `ood_composite` checkpoints.

## Evidence That Matters

### 1. Surviving checkpoint objects under `14d5exx0`

The run folder still contains these relevant checkpoints:

- `top_n_effort_20260412_step12000_auc0.9844_eer0.0369.pth`
- `ood_composite_effort_20260413_step23000_auc0.9889_eer0.0330.pth`
- `ood_composite_effort_20260413_step25000_auc0.9887_eer0.0291.pth`
- `ood_composite_effort_20260414_step27000_auc0.9877_eer0.0330.pth`

The stale path does not exist.

### 2. Recommended substitute

The best semantic substitute for backfill is:

- `gs://training-job-outputs/phase2r13_experiments/14d5exx0/top_n_effort_20260412_step12000_auc0.9844_eer0.0369.pth`

Why this is the recommended choice:

- same run id: `14d5exx0`
- same step: `12000`
- same AUC in filename: `0.9844`
- same EER in filename: `0.0369`
- same checkpoint serializer is used for both `top_n` and `ood_composite` checkpoint saves
- this preserves the original intent of evaluating the April 12 "best so far" state instead of silently substituting a later, stronger checkpoint

What is not recommended:

- silently changing `R13_E_BESTSOFAR` to `step23000`, `step25000`, or `step27000`

Those later checkpoints may be useful for a separate "best current R13_E" question, but they are not the cleanest replacement for the originally launched candidate.

## What This Side Agent Should Produce

The deliverable is:

1. one new Vertex custom job in `RUNNING` state
2. that job evaluates only `R13_E_BESTSOFAR`
3. the alias is repointed to the recommended surviving substitute checkpoint
4. the suite list contains only the `R13_E` mega-eval suites whose reports are still missing from the active mega-eval
5. the output goes to a separate backfill GCS root, not into the live mega-eval root

This job is meant to be merged later with the live mega-eval outputs.

## Hard Constraints

- Do not rerun non-`R13_E` checkpoints.
- Do not rerun `FT8` or `FT10` separately.
- Do not use the updated fake-lockbox-aware promotion suite for this task.
- Do not write the backfill outputs into the live mega-eval output folder.
- Do not claim that the merged mega-eval answers the calibrated promotion contract.
- Do not mutate the current live mega-eval code path.

## Important Background About the Live Mega-Eval Suite Set

For merge compatibility, the backfill must follow the effective live 16-suite mega-eval lane, not the current checked-in 17-suite YAML.

The effective live 16 suites are:

1. `teams_real_all_dev`
2. `teams_real_poor_quality_dev`
3. `teams_real_lighting_extreme_dev`
4. `teams_real_all_lockbox`
5. `visomaster_enhanced_macro_dev`
6. `deeplive_enhanced_dev`
7. `teams_capture_cam_test_dev`
8. `teams_capture_pc_generator_dev`
9. `teams_capture_test_cam_dev`
10. `teams_capture_noyn_sharker_dev`
11. `teams_capture_dor_shkedi_dev`
12. `teams_flat_xiang_xiang2_feng_dev`
13. `teams_fake_all_dev`
14. `deeplive_all_sources`
15. `visomaster_all_sources`
16. `visomaster_enhanced_v2_all`

`teams_fake_all_lockbox` is intentionally not in this list because the active live mega-eval did not include it.

## Current Missing `R13_E` Suites

At the time of writing, the active mega-eval has `R13_E` reports only for:

- `teams_real_all_dev`
- `teams_real_poor_quality_dev`

The missing `R13_E` suites are therefore currently:

1. `teams_real_lighting_extreme_dev`
2. `teams_real_all_lockbox`
3. `visomaster_enhanced_macro_dev`
4. `deeplive_enhanced_dev`
5. `teams_capture_cam_test_dev`
6. `teams_capture_pc_generator_dev`
7. `teams_capture_test_cam_dev`
8. `teams_capture_noyn_sharker_dev`
9. `teams_capture_dor_shkedi_dev`
10. `teams_flat_xiang_xiang2_feng_dev`
11. `teams_fake_all_dev`
12. `deeplive_all_sources`
13. `visomaster_all_sources`
14. `visomaster_enhanced_v2_all`

However, the side agent should compute this list dynamically at launch time.

Reason:

- if someone restores the missing checkpoint object or if the active mega-eval recovers some future `R13_E` rows before the backfill launches, the backfill should skip those suites

## Recommended Operational Decision

### Default choice

Use the alias:

- `R13_E_BESTSOFAR`

but repoint it to:

- `gs://training-job-outputs/phase2r13_experiments/14d5exx0/top_n_effort_20260412_step12000_auc0.9844_eer0.0369.pth`

This keeps the report filenames compatible with the active mega-eval because filenames are based on the alias, not the raw checkpoint filename.

### Optional pre-step if operator approves artifact repair

If the operator is comfortable mutating GCS object history, there is an optional salvage action:

1. copy the surviving `top_n ... step12000 ...` object
2. recreate the missing `ood_composite ... step12000 ...` object key

Potential benefit:

- the still-running mega-eval may stop failing on future not-yet-reached `R13_E` suites
- the still-running shortlist run may avoid the same future `R13_E` failure

Limit:

- this does not repair the mega-eval suites that already failed before the object is restored

This is optional. The backfill job is still needed unless the operator also wants to rerun the already-failed suites another way.

## Recommended Backfill Strategy

### 1. Compute the missing suite list dynamically

At launch time:

1. list the live mega-eval reports root
2. for each suite in the live 16-suite set, check whether
   `<suite>_r13_e_bestsofar_videos_report.csv`
   exists under the live mega-eval reports root
3. include only the suites where that file is missing

If the list is empty, do not launch a backfill job.

### 2. Build a runtime-only checkpoint map

Use a temporary JSON checkpoint map containing only:

```json
{
  "R13_E_BESTSOFAR": "gs://training-job-outputs/phase2r13_experiments/14d5exx0/top_n_effort_20260412_step12000_auc0.9844_eer0.0369.pth"
}
```

This is intentionally not a code change to the repo.

### 3. Build a runtime-only suite manifest

Use a temporary JSON suite manifest containing only the missing suites from the live 16-suite set.

The agent should not rely on the checked-in current mega-eval YAML because that file no longer matches the effective live job contents.

### 4. Launch a custom Vertex job directly

Recommended environment:

- project: `train-cvit2`
- region: `asia-southeast1`
- service account:
  `vertex-job-runner-train-cvit2@train-cvit2.iam.gserviceaccount.com`
- image:
  `us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:1.3.180`

Recommended job naming:

- Vertex display name:
  `r13-best-megaval-r13e-backfill-<timestamp>`

Recommended output root:

- `gs://training-job-outputs/test_results/r13_best_megaval_backfill/r13-best-megaval-20260413-155050-r13e-backfill-<timestamp>/reports/`

Why use a separate output root:

- avoids collision with the live job
- avoids confusion about what the live runner did itself
- makes later merge explicit and auditable

### 5. Run only the sequential validation runner

Run:

- `arena/run_target_domain_validation_sequential.py`

with:

- `--checkpoints R13_E_BESTSOFAR`
- `--checkpoint_map <temp_json>`
- `--suite_manifest <temp_json>`
- `--output_gcs_folder <separate_backfill_reports_root>`
- `--wandb_project enhanced-aug-test`

Do not request fixed-threshold scorecards from this backfill job.

Reason:

- a one-checkpoint partial scorecard is not the final merged mega-eval scorecard
- the only useful backfill output here is the per-suite detailed reports

### 6. Prefer disabled W&B for the backfill

Recommended:

- `WANDB_MODE=disabled`

Reason:

- this is operational repair work, not a new scientific run
- it reduces noise and reduces the chance of avoidable external logging failure

## Concrete Submission Shape

The safest submission style is the same pattern used for the shortlist rerun:

1. submit a raw Vertex custom job via REST or Python requests
2. use the existing container image
3. run `/bin/bash -lc "<script>"`
4. inside the container, write temporary JSON files to `/tmp`
5. run the sequential validation runner directly

Conceptual container script:

```bash
set -euo pipefail
cd /workspace

cat > /tmp/r13e_backfill_checkpoint_map.json <<'JSON'
{ ... }
JSON

cat > /tmp/r13e_backfill_suite_manifest.json <<'JSON'
{ ... }
JSON

export GOOGLE_CLOUD_PROJECT=train-cvit2
export WANDB_MODE=disabled

python -u arena/run_target_domain_validation_sequential.py \
  --checkpoints "R13_E_BESTSOFAR" \
  --checkpoint_map /tmp/r13e_backfill_checkpoint_map.json \
  --suite_manifest /tmp/r13e_backfill_suite_manifest.json \
  --output_gcs_folder "gs://training-job-outputs/test_results/r13_best_megaval_backfill/<job>/reports" \
  --wandb_project enhanced-aug-test
```

## Validation Checklist After Submission

The side agent should verify:

1. the new custom job reaches `RUNNING`
2. the backfill output root starts receiving `*_frames_report.csv`, `*_videos_report.csv`, `*_group_metrics.csv`, and `*_summary_report.txt`
3. the filenames end with `_r13_e_bestsofar_*`
4. the suite names match the live mega-eval suite names exactly

If the first backfill suite still fails immediately:

- confirm the substitute checkpoint path exists
- confirm the suite manifest used the live 16-suite semantics
- confirm the backfill map still uses alias `R13_E_BESTSOFAR`

## Merge Plan After Both Jobs Finish

This is not the immediate deliverable, but the side agent should understand it now.

### 1. Do not rely on the live job's in-memory scorecard

The live runner stores scorecard rows in memory as it runs and only writes final scorecards from that in-memory list at the end.

That means:

- simply dropping backfill report files into the live run folder will not make the current live process include them in its final scorecards

### 2. Create a merged staging root

Recommended merged root:

- `gs://training-job-outputs/test_results/r13_best_megaval_merged/r13-best-megaval-20260413-155050-plus-r13e-backfill-<timestamp>/reports/`

### 3. Populate the merged staging root

1. copy all live mega-eval report artifacts into the merged root
2. copy all backfill `R13_E` report artifacts into the same merged root
3. there should be no filename collisions except possible intentional replacement if the live job later also produced a same-name `R13_E` artifact

### 4. Rebuild the fixed-threshold scorecards from merged reports

There is no checked-in standalone CLI that rebuilds the fixed-threshold mega-eval scorecards from an arbitrary reports folder.

However, the helper functions already exist in:

- `DeepfakeBench/training/arena/run_target_domain_validation_sequential.py`

Relevant functions:

- `_load_scorecard_row`
- `_build_wide_scorecard_rows`
- `_build_pair_delta_rows`
- `_write_scorecard_outputs`

When rebuilding the merged scorecard:

- use the original 7-checkpoint alias set
- use the live 16-suite mega-eval suite set
- use a checkpoint map where `R13_E_BESTSOFAR` points to the actual substitute checkpoint used for the backfill

That final point matters because the merged scorecard metadata should reflect the checkpoint that was actually used for the repaired `R13_E` rows.

### 5. Label the merged result clearly

The merged scorecard should be labeled as:

- post-hoc merged diagnostic evidence

It is not a native output of the original live mega-eval process.

## Suggested Decision Tree For The Side Agent

1. Confirm the active mega-eval is still live and list the currently missing `R13_E` suites.
2. Decide whether artifact repair is allowed.
3. If allowed, optionally recreate the missing `ood_composite ... step12000 ...` object key from the same-step `top_n` file.
4. Regardless of step 3, prepare an `R13_E`-only backfill job using the same-step `top_n` substitute.
5. Launch the backfill job to a separate output root.
6. Verify the first artifacts land.
7. Stop there unless explicitly asked to perform the merge and scorecard rebuild.

## Things The Side Agent Should Not Misinterpret

- `R13_E_BESTSOFAR` is not a placeholder that can be arbitrarily redefined to any later stronger checkpoint.
- The active mega-eval still does not become the promotion contract even if fully merged.
- The backfill job should not use the updated fake-lockbox-aware promotion suite.
- The side agent should not spend time on `FT8` or `FT10`.
- The side agent should not modify repo code just to launch the backfill unless there is a strong operational reason.

## Minimal Facts To Carry Forward

- The active mega-eval is still useful and still progressing for non-`R13_E` checkpoints.
- The `R13_E` failure is operational, not a model-quality conclusion.
- The failure mode is a stale alias to a pruned checkpoint object.
- A same-step same-metrics surviving substitute exists.
- A parallel `R13_E`-only backfill can complete the missing rows.
- The merged result will still be diagnostic only.

