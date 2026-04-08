# Track A Handoff

**Date:** April 6, 2026  
**Scope:** Teams target-domain upgrade, Track A merged Teams-enhanced loader  
**Status:** full-length Vertex run launched on April 7, 2026; awaiting first checkpoint + arena

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
- First full-length Track A baseline is now live on Vertex:
  - display name: `exp-R13_A_trackA_teams_enhanced-20260407-160710`
  - job id: `7406731712530481152`
  - full resource name: `projects/700371397073/locations/asia-southeast1/customJobs/7406731712530481152`
  - region: `asia-southeast1`
  - create time (UTC): `2026-04-07T14:07:17.673276Z`
  - start time (UTC): `2026-04-07T14:07:17.834510Z`
  - current observed state: `JOB_STATE_PENDING`
  - current observed Vertex update time (UTC): `2026-04-07T14:07:26.861950Z`
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

Launch command already used for the current live job:

```bash
cd DeepfakeBench/training
./launch_experiment.sh -y phase2r13-experiments asia-southeast1 \
  experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml
```

Then:

- monitor the live Vertex job:

```bash
gcloud ai custom-jobs describe projects/700371397073/locations/asia-southeast1/customJobs/7406731712530481152 \
  --project=train-cvit2 \
  --region=asia-southeast1

gcloud ai custom-jobs stream-logs projects/700371397073/locations/asia-southeast1/customJobs/7406731712530481152 \
  --project=train-cvit2 \
  --region=asia-southeast1
```

- wait for the first full-length checkpoint
- update `DeepfakeBench/training/arena/arena_config.track_a.yaml`
- run arena on the full-length checkpoint alias `TRACK_A_CANDIDATE`

## What Must Wait

- arena evaluation, until a **full-length** checkpoint exists
- weight tuning
- `p_original` tuning
- any new bucket audit
- sending the smoke checkpoint to arena

## Smoke Outcome

Outcome is already known:

- runtime / integration gate: **passed**
- operational meaning:
  - the first full-length Track A launch is unblocked
  - arena should wait for a real full-length Track A checkpoint, not the smoke checkpoint

Next steps:

1. Wait for the first full-length checkpoint from `exp-R13_A_trackA_teams_enhanced-20260407-160710`.
2. Update `DeepfakeBench/training/arena/arena_config.track_a.yaml` with the first full-length checkpoint path.
3. Run arena on that real checkpoint via:
   - `./launch_arena.sh --config arena_config.track_a.yaml --checkpoints TRACK_A_CANDIDATE`
4. Decide whether to promote the same source swap into additional Round 13 configs.

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
