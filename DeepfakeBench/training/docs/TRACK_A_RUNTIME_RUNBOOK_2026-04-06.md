# Track A Runtime Runbook

**Date:** April 6, 2026  
**Scope:** launch and evaluate the first resolver-driven Track A candidate after the smoke gate

## Smoke Outcome

- display name: `exp-R13_SMOKE_trackA_teams_enhanced-20260406-123308`
- job id: `3841428920024956928`
- region: `asia-southeast1`
- final state: `JOB_STATE_SUCCEEDED`
- create time (UTC): `2026-04-06T10:33:13.444938Z`
- start time (UTC): `2026-04-06T10:41:02Z`
- end time (UTC): `2026-04-06T13:08:48Z`
- final Vertex update time (UTC): `2026-04-06T13:08:55.503508Z`
- W&B run id: `2h0rhxun`
- smoke checkpoints written under:
  - `gs://training-job-outputs/phase2r13_experiments/2h0rhxun/`
- smoke interpretation:
  - runtime / integration gate: **passed**
  - the smoke checkpoint metrics were weak / near-random
  - do **not** send the smoke checkpoint to arena
  - launch a real full-length Track A run next

## Prepared Artifacts

- Full-length Track A config:
  - `DeepfakeBench/training/experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml`
- Smoke config:
  - `DeepfakeBench/training/experiments/phase2_round13/R13_SMOKE_trackA_teams_enhanced.yaml`
- One-off arena config:
  - `DeepfakeBench/training/arena/arena_config.track_a.yaml`

## April 7 Launch Update

The first full-length Track A baseline has now been submitted on Vertex.

- display name: `exp-R13_A_trackA_teams_enhanced-20260407-160710`
- job id: `7406731712530481152`
- full resource name: `projects/700371397073/locations/asia-southeast1/customJobs/7406731712530481152`
- region: `asia-southeast1`
- create time (UTC): `2026-04-07T14:07:17.673276Z`
- start time (UTC): `2026-04-07T14:07:17.834510Z`
- current observed state: `JOB_STATE_PENDING`
- current observed Vertex update time (UTC): `2026-04-07T14:07:26.861950Z`
- base output directory:
  - `gs://training-job-outputs/vertex-output/exp-R13_A_trackA_teams_enhanced-20260407-160710`

Do **not** submit a second copy of this full-length run while it is still active.

Monitor the current job with:

```bash
gcloud ai custom-jobs describe projects/700371397073/locations/asia-southeast1/customJobs/7406731712530481152 \
  --project=train-cvit2 \
  --region=asia-southeast1

gcloud ai custom-jobs stream-logs projects/700371397073/locations/asia-southeast1/customJobs/7406731712530481152 \
  --project=train-cvit2 \
  --region=asia-southeast1
```

## Important Arena CLI Caveat

- `./launch_arena.sh --checkpoints ...` filters checkpoint **names** already defined in the selected arena config.
- It does **not** accept a raw `gs://...pth` checkpoint URI directly.
- For Track A, update `arena/arena_config.track_a.yaml` with the real checkpoint path first, then pass `TRACK_A_CANDIDATE`.

## Launch The First Full-Length Track A Run

1. Launch the full-length Track A baseline:

```bash
cd DeepfakeBench/training
./launch_experiment.sh -y phase2r13-experiments asia-southeast1 \
  experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml
```

2. Monitor the launched training job with the job name or id printed by `launch_experiment.sh`:

```bash
gcloud ai custom-jobs describe <job_name_or_id> \
  --project=train-cvit2 \
  --region=asia-southeast1

gcloud ai custom-jobs stream-logs <job_name_or_id> \
  --project=train-cvit2 \
  --region=asia-southeast1
```

3. Once the first checkpoint lands, find the path:

```bash
gsutil ls gs://training-job-outputs/phase2r13_experiments/*/*.pth | tail -n 20
```

4. Update `DeepfakeBench/training/arena/arena_config.track_a.yaml`:
   - set `checkpoints.TRACK_A_CANDIDATE.path` to the real checkpoint URI;
   - update `notes` with the run label if useful.

5. Launch arena with the Track A one-off config:

```bash
cd DeepfakeBench/training
./launch_arena.sh --config arena_config.track_a.yaml --checkpoints TRACK_A_CANDIDATE
```

6. Monitor the arena job with the submitted job name:

```bash
gcloud ai custom-jobs stream-logs <arena_job_name> \
  --project=train-cvit2 \
  --region=asia-southeast1
```

7. Arena results will be written under:
   - `gs://training-job-outputs/arena_results/<timestamp>/`

## Quick Go / No-Go Checklist

- smoke already reached training steps cleanly
- merged Teams-enhanced sample count was nonzero in smoke logs
- smoke checkpoint landed under `gs://training-job-outputs/phase2r13_experiments/2h0rhxun/`
- first **full-length** checkpoint lands under `gs://training-job-outputs/phase2r13_experiments/<wandb_run_id>/`
- `arena_config.track_a.yaml` is updated with the real checkpoint path
- arena is launched with checkpoint alias `TRACK_A_CANDIDATE`, not the raw URI
- smoke checkpoint is not used as a quality result
