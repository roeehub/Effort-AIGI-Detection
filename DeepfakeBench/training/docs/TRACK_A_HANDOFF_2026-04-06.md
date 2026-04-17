# Track A Handoff

**Date:** April 6, 2026  
**Scope:** Teams target-domain upgrade, Track A merged Teams-enhanced loader  
**Status:** full-length Vertex run is active; first arena candidate already exists

## Current State

- `S1` resolver audit is done.
- Track A core loader is implemented and tested.
- Vertex-safe resolver artifacts are staged at:
  - `gs://training-job-outputs/cache/visomaster/enhanced_visomaster_resolver_2026-04-06.json`
  - `gs://training-job-outputs/cache/visomaster/enhanced_visomaster_resolver_2026-04-06.csv`
- Smoke config is ready at:
  - `DeepfakeBench/training/experiments/phase2_round13/R13_SMOKE_trackA_teams_enhanced.yaml`
- Full-length Track A config is prepared at:
  - `DeepfakeBench/training/experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml`
- Track A arena one-off config is prepared at:
  - `DeepfakeBench/training/arena/arena_config.track_a.yaml`
- Runtime runbook is prepared at:
  - `DeepfakeBench/training/docs/TRACK_A_RUNTIME_RUNBOOK_2026-04-06.md`
- Current full-length Track A baseline is live on Vertex after rebuilding the image to include the new YAML:
  - display name: `exp-R13_A_trackA_teams_enhanced-20260407-165545`
  - job id: `4166954730590830592`
  - full resource name: `projects/700371397073/locations/asia-southeast1/customJobs/4166954730590830592`
  - region: `asia-southeast1`
  - image: `us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:1.3.171`
  - create time (UTC): `2026-04-07T14:55:49.609358Z`
  - start time (UTC): `2026-04-07T15:03:06Z`
  - current observed state as of April 10, 2026: `JOB_STATE_RUNNING`
  - W&B run id: `f04l917o`
- First usable full-length Track A candidate so far:
  - selection lane: `best_ood_composite`
  - step: `11500`
  - holdout AUC: `0.9848`
  - OOD AUC: `0.9664`
  - OOD-composite: `0.9755`
  - checkpoint:
    - `gs://training-job-outputs/phase2r13_experiments/f04l917o/ood_composite_effort_20260409_step11500_auc0.9848_eer0.0369.pth`
- Completed Vertex smoke:
  - display name: `exp-R13_SMOKE_trackA_teams_enhanced-20260406-123308`
  - job id: `3841428920024956928`
  - region: `asia-southeast1`
  - final state: `JOB_STATE_SUCCEEDED`
  - create time (UTC): `2026-04-06T10:33:13.444938Z`
  - start time (UTC): `2026-04-06T10:41:02Z`
  - end time (UTC): `2026-04-06T13:08:48Z`
  - final Vertex update time (UTC): `2026-04-06T13:08:55.503508Z`
  - W&B run id: `2h0rhxun`
  - checkpoints written under:
    - `gs://training-job-outputs/phase2r13_experiments/2h0rhxun/`

## What Was Already Verified

- Loader behavior:
  - one unified sample per base `sample_id`
  - no 8x enhancer sample explosion
  - fake branch selected at iteration time
- Real-manifest load snapshot:
  - `997` merged training-eligible samples
  - `54` `teams_v2`
  - `943` `clean_fallback`
  - `2` unresolved rows excluded
- Targeted tests already passed:
  - `9 passed`
  - `0 failed`
- Smoke gate facts confirmed from logs:
  - merged source loaded cleanly from the staged resolver manifest
  - `128` merged `visomaster_teams_enhanced` samples loaded for the smoke slice
  - companion-domain split in the smoke slice:
    - `12` `teams_v2`
    - `116` `clean_fallback`
  - nonzero merged samples appeared in train / val / test logs
  - checkpoints landed successfully in GCS
- Important interpretation:
  - the smoke passed the runtime / integration gate
  - the smoke checkpoint metrics were weak / near-random
  - the smoke checkpoint itself should **not** be treated as a model-quality result or sent to arena

## What To Do Right Now

Do not relaunch the smoke. Treat it as passed.

Do not submit a second copy of the full-length Track A run while the current job is active.

Use the current best checkpoint to unblock arena now:

- set `TRACK_A_CANDIDATE` to:
  - `gs://training-job-outputs/phase2r13_experiments/f04l917o/ood_composite_effort_20260409_step11500_auc0.9848_eer0.0369.pth`
- launch the one-off Track A arena path
- keep monitoring the live training run only for a strictly better OOD-composite checkpoint

Monitor the live Vertex job with:

```bash
gcloud ai custom-jobs describe projects/700371397073/locations/asia-southeast1/customJobs/4166954730590830592 \
  --project=train-cvit2 \
  --region=asia-southeast1

gcloud ai custom-jobs stream-logs projects/700371397073/locations/asia-southeast1/customJobs/4166954730590830592 \
  --project=train-cvit2 \
  --region=asia-southeast1
```

- update `DeepfakeBench/training/arena/arena_config.track_a.yaml`
- run arena on the full-length checkpoint alias `TRACK_A_CANDIDATE`

## What Must Wait

- weight tuning
- `p_original` tuning
- any new bucket audit
- sending the smoke checkpoint to arena
- promoting the Track A source swap into additional Round 13 configs before the first arena readout

## Smoke Outcome

Outcome is already known:

- runtime / integration gate: **passed**
- operational meaning:
  - the first full-length Track A launch is unblocked
  - arena should wait for a real full-length Track A checkpoint, not the smoke checkpoint

Next steps:

1. Update `DeepfakeBench/training/arena/arena_config.track_a.yaml` with the current best checkpoint path:
   - `gs://training-job-outputs/phase2r13_experiments/f04l917o/ood_composite_effort_20260409_step11500_auc0.9848_eer0.0369.pth`
2. Run arena on that real checkpoint via:
   - `./launch_arena.sh --config arena_config.track_a.yaml --checkpoints TRACK_A_CANDIDATE`
3. Keep monitoring the live Vertex job for a better OOD-composite checkpoint.
4. Decide whether to promote the same source swap into additional Round 13 configs after the first arena readout.

## Key Files

- `DeepfakeBench/training/data/sources/visomaster.py`
- `DeepfakeBench/training/data/sources/combined_paired.py`
- `DeepfakeBench/training/tests/test_phase4_family_pipeline.py`
- `DeepfakeBench/training/experiments/phase2_round13/R13_SMOKE_trackA_teams_enhanced.yaml`
- `DeepfakeBench/training/experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml`
- `DeepfakeBench/training/arena/arena_config.track_a.yaml`
- `DeepfakeBench/training/docs/TRACK_A_RUNTIME_RUNBOOK_2026-04-06.md`
- `DeepfakeBench/training/docs/TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md`

## Things Explicitly Not To Spend Time On

- re-deriving the `54 / 943 / 2` split
- re-checking clean companion existence
- changing Teams simulation
- deep weight tuning before the first full-length readout
