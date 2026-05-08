# P2 packet — CANARY_TRAJECTORY_FACTS_2026-05-08

> **Status: factual-only.** No interpretation. Forbidden words: succeeds, fails, wins,
> promotes, deployment-grade. Per `docs/packet_retrospectives/eval_folder_template.md`
> and `docs/packet_retrospectives/threads/in_training_canary_signal.md`.
>
> **Pulled 2026-05-08 morning from W&B `dtect-vision/phase2-round13`** via
> `wandb.Api().run(...).scan_history(keys=...)`. CSVs at `slot_{A,B,C,D}_canary_history.csv`.

## Cadence anomaly (open observation)

The yaml configured `canary_probe.frequency_steps: 1000`. The W&B `_step` values logged at
canary-fire time are at multiples of 3000 (Slot C, Slot D) — not 1000. Slot A and Slot B
emitted only ONE canary fire each at `_step` 7000 and 6000 respectively. Mapping between
optimizer-step and W&B `_step` is unverified at the level of this FACTS doc; possible
causes (none verified): batch-vs-optimizer step counting, gradient accumulation, or a
silent skip when the canary fires near training termination.

| slot | wandb final `_step` | wandb final epoch | canary fire count | canary fire `_step`s |
|---|---:|---:|---:|---|
| A — BUNDLE | 7508 | 3 | 1 | 7000 |
| B — CORR_ONLY | 6508 | 3 | 1 | 6000 |
| C — PAIRRANK | 13013 | 5 | 4 | 3000, 6000, 9000, 12000 |
| D — FOURIER | 25013 | 9 | 8 | 3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000 |

## Logged canary scalars (29 keys per slot)

Per `trainer/mixins/canary_probe.py` (verified by inspecting `r.summary` keys for each run):

- Distribution (8): `canary/score_{p50,p95,mean,std}_on_reals`, `canary/score_{p50,p05,mean}_on_fakes`
- Per-chronic-identity mean (6): `canary/chronic_mean/<identity>` for `PC_Generator__s22`, `PC_Generator__s45`, `Q__s6`, `Roy_D`, `bla_bla_chow`, `bla_bla_chow__s2`
- Per-identity aggregates (2): `canary/{max,mean}_per_identity_mean_score`
- Lockbox-FPR-calibrated (3): `canary/lockbox_recall_at_FPR_{5pct,10pct}`, `canary/lockbox_tau_at_FPR_10pct`
- Per-method recall@τ=0.5 (3): `canary/recall_at_tau05/{lockbox_fake,viso_fake,deeplive_fake}` (3 keys, not the 4 listed in `threads/in_training_canary_signal.md` — the `teams` key is absent in actual logs)
- Wilcoxon vs P8A reference (4): `canary/{wilcoxon_stat,wilcoxon_pval,mean_score_drift,abs_mean_score_drift}_vs_p8a_reals`
- Sanity counters (3): `canary/n_{frames_evaluated,reals,fakes}`

`canary/n_frames_evaluated` = 800 in every fire across all 4 slots. No drops below 700.
`canary/probe_step` is also logged.

## Per-slot canary trajectory tables

### Slot A (BUNDLE) — `u22wz1vf`

Single fire at `_step=7000`:

| metric | value |
|---|---:|
| `score_p50_on_reals` | 0.4994 |
| `score_p95_on_reals` | 0.4994 |
| `score_mean_on_reals` | 0.4991 |
| `score_p50_on_fakes` | 0.4994 |
| `score_p05_on_fakes` | 0.4993 |
| `score_mean_on_fakes` | 0.4994 |
| `lockbox_recall_at_FPR_10pct` | 0.0000 |
| `lockbox_recall_at_FPR_5pct` | 0.0000 |
| `lockbox_tau_at_FPR_10pct` | 0.4994 |
| `recall_at_tau05/lockbox_fake` | 0.0000 |
| `recall_at_tau05/viso_fake` | 0.0000 |
| `recall_at_tau05/deeplive_fake` | 0.0000 |
| `max_per_identity_mean_score` | 0.4994 |
| `mean_per_identity_mean_score` | 0.4988 |
| `wilcoxon_pval_vs_p8a_reals` | <0.0001 |
| `mean_score_drift_vs_p8a_reals` | +0.1523 |
| `abs_mean_score_drift_vs_p8a_reals` | 0.4280 |

Chronic-6 means at fire: PC_Gen_s22 0.4975, PC_Gen_s45 0.4994, Q_s6 0.4994, Roy_D 0.4994, bla_bla_chow 0.4994, bla_bla_chow_s2 0.4977.

### Slot B (CORR_ONLY) — `mlo5vfe8`

Single fire at `_step=6000`:

| metric | value |
|---|---:|
| `score_p50_on_reals` | 0.5003 |
| `score_p95_on_reals` | 0.5005 |
| `score_mean_on_reals` | 0.5003 |
| `score_p50_on_fakes` | 0.5002 |
| `score_p05_on_fakes` | 0.4985 |
| `score_mean_on_fakes` | 0.5004 |
| `lockbox_recall_at_FPR_10pct` | 0.0100 |
| `lockbox_recall_at_FPR_5pct` | 0.0000 |
| `lockbox_tau_at_FPR_10pct` | 0.5004 |
| `recall_at_tau05/lockbox_fake` | 0.4800 |
| `recall_at_tau05/viso_fake` | 1.0000 |
| `recall_at_tau05/deeplive_fake` | 1.0000 |
| `max_per_identity_mean_score` | 0.5009 |
| `mean_per_identity_mean_score` | 0.5004 |
| `wilcoxon_pval_vs_p8a_reals` | <0.0001 |
| `mean_score_drift_vs_p8a_reals` | +0.1535 |
| `abs_mean_score_drift_vs_p8a_reals` | 0.4282 |

Chronic-6 means: PC_Gen_s22 0.5009, PC_Gen_s45 0.5004, Q_s6 0.5004, Roy_D 0.5002, bla_bla_chow 0.5002, bla_bla_chow_s2 0.5003.

### Slot C (PAIRRANK_ONLY) — `oaur8odo`

| `_step` | p50_reals | p95_reals | mean_reals | p50_fakes | p05_fakes | lockbox@10 | lockbox@5 | tau@10 | max_per_id | r_lockbox | r_viso | r_deeplive | wilcox_p | drift_mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3000 | 0.0452 | 0.9983 | 0.2496 | 0.5092 | 0.0045 | 0.020 | 0.000 | 0.9580 | 0.8693 | 0.270 | 0.500 | 1.000 | <1e-4 | -0.0973 |
| 6000 | 0.0568 | 0.9998 | 0.3090 | 0.8981 | 0.0146 | 0.120 | 0.010 | 0.9978 | 0.9476 | 0.690 | 0.460 | 1.000 | 0.0231 | -0.0378 |
| 9000 | 0.0276 | 0.9999 | 0.2919 | 0.9099 | 0.0078 | 0.060 | 0.000 | 0.9988 | 0.9447 | 0.700 | 0.440 | 1.000 | 7e-4 | -0.0549 |
| 12000 | 0.1407 | 0.9998 | 0.3885 | 0.9960 | 0.1317 | 0.260 | 0.010 | 0.9995 | 0.9731 | 0.940 | 0.720 | 1.000 | 0.0018 | +0.0416 |

Chronic-6 trajectory:

| `_step` | PC_Gen_s22 | PC_Gen_s45 | Q_s6 | Roy_D | bla_bla_chow | bla_bla_chow_s2 |
|---:|---:|---:|---:|---:|---:|---:|
| 3000 | 0.0730 | 0.1499 | 0.2916 | 0.8693 | 0.5479 | 0.3964 |
| 6000 | 0.1945 | 0.6475 | 0.2221 | 0.9476 | 0.7329 | 0.6550 |
| 9000 | 0.1527 | 0.6596 | 0.1867 | 0.9447 | 0.7555 | 0.5653 |
| 12000 | 0.6646 | 0.7768 | 0.2440 | 0.9731 | 0.6314 | 0.8149 |

### Slot D (FOURIER) — `89tt9xyz`

| `_step` | p50_reals | p95_reals | mean_reals | p50_fakes | p05_fakes | lockbox@10 | lockbox@5 | tau@10 | max_per_id | r_lockbox | r_viso | r_deeplive | wilcox_p | drift_mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3000 | 0.2014 | 0.8828 | 0.2953 | 0.6659 | 0.1394 | 0.370 | 0.130 | 0.7792 | 0.7571 | 0.880 | 0.180 | 0.960 | 0.5399 | -0.0515 |
| 6000 | 0.1118 | 0.9910 | 0.3363 | 0.8495 | 0.0429 | 0.060 | 0.000 | 0.9705 | 0.9308 | 0.740 | 0.480 | 0.980 | 0.1667 | -0.0105 |
| 9000 | 0.1447 | 0.9940 | 0.3643 | 0.8342 | 0.0776 | 0.100 | 0.010 | 0.9832 | 0.9606 | 0.650 | 0.500 | 0.980 | 0.0020 | +0.0175 |
| 12000 | 0.1110 | 0.9835 | 0.3008 | 0.6127 | 0.0206 | 0.090 | 0.000 | 0.9197 | 0.8644 | 0.390 | 0.520 | 0.960 | 0.7464 | -0.0461 |
| 15000 | 0.0614 | 0.9943 | 0.3190 | 0.5947 | 0.0250 | 0.030 | 0.010 | 0.9893 | 0.9450 | 0.430 | 0.280 | 1.000 | 0.2019 | -0.0278 |
| 18000 | 0.0444 | 0.9623 | 0.2455 | 0.7659 | 0.0161 | 0.410 | 0.220 | 0.8910 | 0.6795 | 0.600 | 0.360 | 0.980 | 0.0029 | -0.1013 |
| 21000 | 0.0479 | 0.9926 | 0.3288 | 0.9257 | 0.0194 | 0.170 | 0.010 | 0.9852 | 0.8950 | 0.620 | 0.620 | 0.980 | 0.5624 | -0.0181 |
| 24000 | 0.0754 | 0.9942 | 0.3726 | 0.9637 | 0.0248 | 0.170 | 0.010 | 0.9911 | 0.9383 | 0.790 | 0.620 | 0.980 | 0.0030 | +0.0258 |

Chronic-6 trajectory:

| `_step` | PC_Gen_s22 | PC_Gen_s45 | Q_s6 | Roy_D | bla_bla_chow | bla_bla_chow_s2 |
|---:|---:|---:|---:|---:|---:|---:|
| 3000 | 0.2821 | 0.5868 | 0.3941 | 0.7571 | 0.3595 | 0.3751 |
| 6000 | 0.2226 | 0.6776 | 0.3416 | 0.9308 | 0.7798 | 0.2919 |
| 9000 | 0.3029 | 0.7392 | 0.3755 | 0.9606 | 0.8214 | 0.4706 |
| 12000 | 0.3373 | 0.6216 | 0.3299 | 0.8644 | 0.5649 | 0.2015 |
| 15000 | 0.1648 | 0.7364 | 0.3246 | 0.9450 | 0.9059 | 0.3976 |
| 18000 | 0.5098 | 0.5589 | 0.2448 | 0.6795 | 0.4067 | 0.3245 |
| 21000 | 0.6998 | 0.6236 | 0.2210 | 0.8950 | 0.8063 | 0.3800 |
| 24000 | 0.8128 | 0.7250 | 0.2461 | 0.9383 | 0.8578 | 0.5175 |

## Threshold-crossing observations (per task spec §5.2)

Recorded mechanically against the four flagged thresholds:

| slot | step | metric | value | threshold | crossing | trainer.log notes |
|---|---:|---|---:|---|---|---|
| A | 7000 | `score_p95_on_reals` | 0.4994 | <0.85 | below | n/a |
| B | 6000 | `score_p95_on_reals` | 0.5005 | <0.85 | below | n/a |
| C | 3000 | `score_p95_on_reals` | 0.9983 | >0.85 | **crossed up** | n/a |
| C | 3000 → 12000 | `max_per_identity_mean_score` | 0.8693 → 0.9731 | >0.50 | **crossed up** at first fire | n/a |
| C | 6000 → 9000 | `lockbox_recall_at_FPR_10pct` | 0.120 → 0.060 | falling | **crossed down** | n/a |
| C | 9000 → 12000 | `lockbox_recall_at_FPR_10pct` | 0.060 → 0.260 | rising | (recovered) | n/a |
| D | 3000 | `score_p95_on_reals` | 0.8828 | >0.85 | **crossed up** | n/a |
| D | 3000 → 6000 | `lockbox_recall_at_FPR_10pct` | 0.370 → 0.060 | falling | **crossed down** | n/a |
| D | 12000 → 18000 | `max_per_identity_mean_score` | 0.8644 → 0.6795 | falling | crossed down (then back up) | n/a |
| D | 15000 → 18000 | `lockbox_recall_at_FPR_10pct` | 0.030 → 0.410 | rising | recovered to peak | n/a |
| D | 18000 → 21000 | `max_per_identity_mean_score` | 0.6795 → 0.8950 | rising | re-crossed up | n/a |

`n_frames_evaluated < 700`: never observed (always 800 across all fires for all 4 slots).

## Per-slot ckpt-list summaries

See `RESULTS_FACTS_2026-05-08.md` §"Per-slot checkpoint inventory".

## Artifacts

- Per-slot CSVs: `slot_{A,B,C,D}_canary_history.csv` (this folder)
- Canary parquet (frames + P8A reference scores): `arena/canaries/teams_chronic_diverse_800_2026-05-07.parquet`
- Canary build script: `arena/canaries/build_canary_2026-05-07.py`
- Trainer mixin: `trainer/mixins/canary_probe.py`
- W&B project: <https://wandb.ai/dtect-vision/phase2-round13>
