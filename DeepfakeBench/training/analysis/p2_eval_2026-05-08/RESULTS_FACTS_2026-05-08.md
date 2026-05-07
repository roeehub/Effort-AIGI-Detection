# P2 packet — RESULTS_FACTS_2026-05-08

> **Status: factual-only.** No interpretation. Forbidden words: succeeds, fails, wins,
> promotes, deployment-grade. Per `docs/packet_retrospectives/eval_folder_template.md`.
>
> **Population status (2026-05-08 00:50 local)**: skeleton — to be filled per slot as
> training terminates. Slot A submit 2026-05-07 21:38:26 UTC, Slot B 21:42:11, Slot C
> 21:36:36, Slot D 22:43:33. Expected wall-clock 10-12h each.

## Per-slot training outcomes

| slot | yaml | vertex job id | region | image | wandb run id | submit (UTC) | running (UTC) | terminal (UTC) | terminal state | total ckpts | canary first-fire step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A — BUNDLE | `R13_P2_SCRATCH_BUNDLE.yaml` | `8395459915946131456` | us-east1 | 1.3.271 | `u22wz1vf` | 2026-05-07 21:38 | 21:38:26 (per STATE.md) | TBD | TBD | TBD | TBD |
| B — CORR_ONLY | `R13_P2_SCRATCH_CORR_ONLY.yaml` | `5580112770428305408` | us-west4 | 1.3.271 | `mlo5vfe8` | 2026-05-07 21:42 | 21:42:11 (per STATE.md) | TBD | TBD | TBD | TBD |
| C — PAIRRANK_ONLY | `R13_P2_SCRATCH_PAIRRANK_ONLY.yaml` | `3126673777023254528` | us-central1 | 1.3.271 | `oaur8odo` | 2026-05-07 21:36 | 21:36:36 (per STATE.md) | TBD | TBD | TBD | TBD |
| D — FOURIER | `R13_P2_SCRATCH_FOURIER.yaml` | `8486657808400384000` | us-east1 | 1.3.272 | TBD (pending first log) | 2026-05-07 22:43:33 | TBD | TBD | TBD | TBD | TBD |

Populated as each slot terminates by reading `gcloud ai custom-jobs describe` + listing
`gs://training-job-outputs/best_checkpoints/<wandb_run_id>/` + counting periodic_saves.

## Per-ckpt × suite scorecard

> Pending — populates after a Phase A scorecard is launched on the P2 ckpts.
> The next session (after these 4 slots terminate) decides whether/when to run
> the Phase A scorecard. This eval-folder is OUTCOME data only; scorecard data
> arrives in a separate run.

## Artifacts

- W&B project: <https://wandb.ai/dtect-vision/phase2-round13>
- GCS ckpt prefix: `gs://training-job-outputs/best_checkpoints/<wandb_run_id>/`
- Stream logs: `gcloud ai custom-jobs stream-logs <full_job_path> --region=<region>`
- Per-slot canary CSV: `<slot>_canary_history.csv` (this folder, populated post-terminal)
