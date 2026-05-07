# P2 packet — CANARY_TRAJECTORY_FACTS_2026-05-08

> **Status: factual-only.** No interpretation. Forbidden words: succeeds, fails, wins,
> promotes, deployment-grade. Per `docs/packet_retrospectives/eval_folder_template.md`
> and `docs/packet_retrospectives/threads/in_training_canary_signal.md`.
>
> **Population status (2026-05-08 00:50 local)**: skeleton. Canary first-fire is at
> training step 1000 (per `frequency_steps: 1000` in the four yamls). Slot A was at
> step 501 at 22:05 UTC; Slots A/B/C should hit step 1000 between approx 22:30 UTC
> and 00:00 UTC tonight depending on data-loader latency. Slot D's first canary fire
> follows whenever it reaches step 1000 from the 22:43:33 UTC start.

## Logged canary scalars (per probe, every 1000 steps)

Per `trainer/mixins/canary_probe.py` and the canary thread:

1. `canary/score_p50_on_reals`, `canary/score_p95_on_reals`, `canary/score_mean_on_reals`, `canary/score_std_on_reals`
2. `canary/score_p50_on_fakes`, `canary/score_p05_on_fakes`, `canary/score_mean_on_fakes`
3. `canary/chronic_mean/<identity>` (6 keys) + `canary/max_per_identity_mean_score`, `canary/mean_per_identity_mean_score`
4. `canary/lockbox_recall_at_FPR_10pct`, `canary/lockbox_recall_at_FPR_5pct`, `canary/lockbox_tau_at_FPR_10pct`
5. `canary/recall_at_tau05/<method_cohort>` (4 keys: lockbox / viso / deeplive / teams)
6. `canary/wilcoxon_stat_vs_p8a_reals`, `canary/wilcoxon_pval_vs_p8a_reals`, `canary/mean_score_drift_vs_p8a_reals`, `canary/abs_mean_score_drift_vs_p8a_reals`
7. `canary/n_frames_evaluated`, `canary/n_reals`, `canary/n_fakes`

Pulled to local CSV per slot via `wandb.Api().run('dtect-vision/phase2-round13/<wandb_run_id>').history(...)` post-terminal.

## Per-slot canary trajectory tables

> Each slot's row gets filled as the slot terminates and W&B history is pulled.
> One row per slot × probe step (1000, 2000, ..., final). Columns mirror §"Logged
> canary scalars". Populated by §5.3 of the Slot D task spec.

### Slot A (BUNDLE) — `u22wz1vf`

Pending — slot still RUNNING.

### Slot B (CORR_ONLY) — `mlo5vfe8`

Pending — slot still RUNNING.

### Slot C (PAIRRANK_ONLY) — `oaur8odo`

Pending — slot still RUNNING.

### Slot D (FOURIER) — TBD

Pending — slot in JOB_STATE_PENDING at 22:43:33 UTC submit; W&B run id assigned
on first step log.

## Threshold-crossing observations

Per Slot D task spec §5.2, observations are *recorded only* when canary signals
cross these thresholds (no interpretation here):

- `canary/score_p95_on_reals` rising past 0.85 mid-training
- `canary/lockbox_recall_at_FPR_10pct` falling between two consecutive probes
- `canary/max_per_identity_mean_score` rising past 0.50
- `canary/n_frames_evaluated < 700` (canary loader dropped frames; capture
  trainer.log warning explaining why)

| slot | step | metric | value | crossing direction | trainer.log notes (if applicable) |
|---|---:|---|---:|---|---|
| (no observations recorded yet) | | | | | |

## Per-frame disagreement / agreement structure

Pending — populates when a non-trivial number of canary fires have landed across
all 4 slots and a comparison can be tabulated. Driven from the per-slot CSVs.

## Artifacts

- Per-slot CSVs: `<slot>_canary_history.csv` (this folder)
- Canary parquet (frames + P8A reference scores): `arena/canaries/teams_chronic_diverse_800_2026-05-07.parquet`
- Canary build script: `arena/canaries/build_canary_2026-05-07.py`
- Trainer mixin: `trainer/mixins/canary_probe.py`
- W&B project: <https://wandb.ai/dtect-vision/phase2-round13>
