# P2 packet — RESULTS_FACTS_2026-05-08

> **Status: factual-only.** No interpretation. Forbidden words: succeeds, fails, wins,
> promotes, deployment-grade. Per `docs/packet_retrospectives/eval_folder_template.md`.
>
> **Population status (2026-05-08 morning)**: outcomes for all 4 slots populated. Canary
> trajectory facts in `CANARY_TRAJECTORY_FACTS_2026-05-08.md`. Per-ckpt × suite scorecard
> deferred to a Phase A scorecard that has not been authorized at the time of writing.

## Per-slot training outcomes

Source: `gcloud ai custom-jobs describe`, `wandb.Api().run(...)`, and `gs://training-job-outputs/best_checkpoints/<wandb_run_id>/` listings.

| slot | yaml | vertex job id | region | image | wandb run id | submit (UTC) | running (UTC) | terminal (UTC) | wall clock | terminal state | wandb final _step | wandb final epoch |
|---|---|---|---|---|---|---|---|---|---|---|---:|---:|
| A — BUNDLE | `R13_P2_SCRATCH_BUNDLE.yaml` | `8395459915946131456` | us-east1 | 1.3.271 | `u22wz1vf` | 2026-05-07 21:38 | 21:38:26 | 23:49:26 | 2h 11m | `JOB_STATE_SUCCEEDED` | 7508 | 3 |
| B — CORR_ONLY | `R13_P2_SCRATCH_CORR_ONLY.yaml` | `5580112770428305408` | us-west4 | 1.3.271 | `mlo5vfe8` | 2026-05-07 21:42 | 21:42:11 | 2026-05-08 00:39:28 | 2h 57m | `JOB_STATE_SUCCEEDED` | 6508 | 3 |
| C — PAIRRANK_ONLY | `R13_P2_SCRATCH_PAIRRANK_ONLY.yaml` | `3126673777023254528` | us-central1 | 1.3.271 | `oaur8odo` | 2026-05-07 21:36 | 21:36:36 | 2026-05-08 00:53:42 | 3h 17m | `JOB_STATE_SUCCEEDED` | 13013 | 5 |
| D — FOURIER | `R13_P2_SCRATCH_FOURIER.yaml` | `8486657808400384000` | us-east1 | 1.3.272 | `89tt9xyz` | 2026-05-07 22:43 | 22:48:25 | 2026-05-08 05:31:20 | 6h 43m | `JOB_STATE_SUCCEEDED` | 25013 | 9 |

**Direct observations**:

1. All 4 slots reached `JOB_STATE_SUCCEEDED`. None reached the configured `total_training_steps: 8000` × `nEpochs: 16` cap; `early_stopping_patience: 12` triggered (or another stop condition fired) before epoch 16 in every slot.
2. Slot A and Slot B terminated at epoch 3 with `_step` in [6508, 7508].
3. Slot C terminated at epoch 5 with `_step=13013`.
4. Slot D terminated at epoch 9 with `_step=25013`. Slot D's wall-clock is ~3× the other 3 slots; consistent with the per-frame FFT overhead introduced by `data/augmentations/fourier_band_aug.py` (~3 channel FFT2 + IFFT2 per frame at p_apply=0.5).
5. Discrepancy between yaml `total_training_steps: 8000` and observed `_step` values (7508-25013) is unresolved at the level of this FACTS doc (open observation; see CANARY_TRAJECTORY_FACTS §"Cadence anomaly").

## Per-slot checkpoint inventory

Source: `gs://training-job-outputs/best_checkpoints/<wandb_run_id>/` listings (2026-05-08 morning).

### Slot A (BUNDLE) — `u22wz1vf`

| filename pattern | optimizer step | best metric (filename suffix) |
|---|---:|---|
| `first_best_effort_*_ep1` | ep1 | auc=0.6802 / eer=0.3772 |
| `periodic_*_step{500,1000,2000,4000,5000,7000}` | 500-7000 | step500 auc=0.6802 → step7000 auc=0.5100 (monotonic descent) |
| `top_n_*_step500` | 500 | auc=0.6802 |
| `value_composite_*_step500` | 500 | auc=0.6802 |

9 ckpts total. Training-AUC trajectory descends from 0.6802 (step500) → 0.5100 (step7000).

### Slot B (CORR_ONLY) — `mlo5vfe8`

| filename pattern | optimizer step | best metric |
|---|---:|---|
| `first_best_effort_*_ep1` | ep1 | auc=0.6800 / eer=0.3853 |
| `periodic_*_step{500,1000,2000,3000,4000,5000,6000}` | 500-6000 | step500 auc=0.6800 → step2000 auc=0.3831 (below-chance) → step6000 auc=0.4374 |
| `top_n_*_step500` | 500 | auc=0.6800 |
| `value_composite_*_step{500,1500,3500}` | 500-3500 | step500 auc=0.6800 → step3500 auc=0.3815 |

11 ckpts total. Training-AUC trajectory drops below chance (0.3815-0.4374) for steps 2000-6000.

### Slot C (PAIRRANK_ONLY) — `oaur8odo`

| filename pattern | optimizer step | best metric |
|---|---:|---|
| `first_best_effort_*_ep1` | ep1 | auc=0.8840 / eer=0.2146 |
| `periodic_*_step{500,1000,2000,3000,4000,5000,6000,7000,8000}` | 500-8000 | step500 auc=0.8840 → step7000 auc=0.9862 (peak) → step8000 auc=0.9849 |
| `top_n_*_step{2500,3500,4000,5500,6000,7000}` | 2500-7000 | step7000 auc=0.9862 |
| `value_composite_*_step500` | 500 | auc=0.8840 |

17 ckpts total. Training-AUC monotone-rising 0.8840 → 0.9862, peak at step7000.

### Slot D (FOURIER) — `89tt9xyz`

| filename pattern | optimizer step | best metric |
|---|---:|---|
| `first_best_effort_*_ep1` | ep1 | auc=0.8743 / eer=0.1983 |
| `periodic_*_step{500,1000,2000,3000,4000,5000,6000,7000,8000}` | 500-8000 | step500 auc=0.8743 → step6000 auc=0.9878 (peak in this set) → step8000 auc=0.9859 |
| `top_n_*_step{3500,4500,6000,10500,14500,19000}` | 3500-19000 | step19000 auc=0.9908 (highest top_n) |
| `value_composite_*_step500` | 500 | auc=0.8743 |

17 ckpts total. Training-AUC monotone-rising 0.8743 → 0.9908; top_n picks include step19000 = highest in chain. Slot D's run produced 2× as many top_n ckpts as Slot C (because Slot D ran ~2× more `_step`s).

## Per-ckpt × suite scorecard

Pending — populates after a Phase A scorecard is launched on the P2 ckpts. The next session
(after this CPU-diagnostics round) decides whether to authorize a Phase A scorecard. This
eval-folder is OUTCOME data only; scorecard data arrives in a separate run.

The CPU-diagnostics round (D1-D4) under `analysis/p2_eval_2026-05-08/d1_d4_cpu/` produces
per-frame canary-substrate scoring on a curated 5-ckpt subset (A/top_n_step500, B/top_n_step500,
C/top_n_step7000, D/top_n_step6000, D/top_n_step19000) — these are NOT a Phase-A scorecard
substitute but inform whether Phase A is worth the GPU spend.

## Artifacts

- W&B project: <https://wandb.ai/dtect-vision/phase2-round13>
- GCS ckpt prefix: `gs://training-job-outputs/best_checkpoints/<wandb_run_id>/`
- Stream logs: `gcloud ai custom-jobs stream-logs <full_job_path> --region=<region>`
- Per-slot canary CSVs: `slot_{A,B,C,D}_canary_history.csv` (this folder)
- D1-D4 CPU diagnostic outputs: `d1_d4_cpu/outputs/` and `d1_d4_cpu/figs/`
