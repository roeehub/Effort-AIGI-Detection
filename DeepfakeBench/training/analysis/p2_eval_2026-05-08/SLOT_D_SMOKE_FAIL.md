# SLOT D — fourier_band_aug smoke test outcome (2026-05-08)

> **Status: factual-only.** This doc records what the smoke test produced; per task spec
> §3.7 the next agent (operational) does NOT change aug parameters unilaterally — surface
> to the user and wait for explicit direction.

## Tiers run

Per task spec `docs/packet_retrospectives/SLOT_D_FOURIER_TASK_2026-05-08.md` §3.7.

| Tier | Source | Outcome | Numbers |
|---|---|---|---|
| 1 — synthetic | uint8 random 224×224×3, p_apply=1.0 | **PASS** | mean abs diff = 7.65; p99 = 26.0; bounds (>0.5, <30) satisfied |
| 2 — disabled | uint8 random 224×224×3, enabled=False, p_apply=1.0 | **PASS** | exact byte equality; class form + function form both pass-through |
| 3 — real face crop | 5 RNG draws on `PC_Generator__s22_1002.0_frame_034941_crop_000__c83e5f44.jpg` | **FAIL** on `mean_abs_diff > 1.0` floor | mean abs diff per trial = [0.539, 0.532, 0.567, 0.601, 0.543]; p99 per trial = [2.0, 2.0, 2.0, 3.0, 2.0]; bound `<80` satisfied with massive headroom |

## Diversity probe — to test whether Tier 3 failure is one-frame or systematic

Re-ran the aug on one frame from each of the 15 canary cohorts (chronic-6 + healthy-5 +
HDTF clean reals + lockbox fakes + viso fakes + deeplive fakes), 5 RNG draws each, all
with `enabled=True, p_apply=1.0`. Source:
`arena/canaries/teams_chronic_diverse_800_2026-05-07.parquet`.

| cohort | label | mean abs diff (avg) | p99 abs diff (avg) | max abs diff |
|---|---:|---:|---:|---:|
| chronic_PCGen_s22 | real | 0.551 | 2.00 | 4 |
| chronic_PCGen_s45 | real | 0.515 | 1.40 | 4 |
| chronic_Q_s6 | real | 0.555 | 2.00 | 5 |
| chronic_Roy_D | real | 0.535 | 1.80 | 5 |
| chronic_bla_bla_chow | real | 0.758 | 5.00 | 9 |
| chronic_bla_bla_chow_s2 | real | 0.577 | 2.40 | 6 |
| deeplive_fake | fake | 0.601 | 2.20 | 5 |
| hdtf_clean_real | real | 0.656 | 2.60 | 7 |
| healthy_dor | real | 0.586 | 2.40 | 8 |
| healthy_dor_shkedi | real | 0.563 | 2.00 | 6 |
| healthy_md_noyn_sharker | real | 0.537 | 1.80 | 5 |
| healthy_test_cam | real | 0.546 | 1.80 | 6 |
| healthy_xiang_xiang2_feng | real | 0.708 | 3.20 | 14 |
| lockbox_fake | fake | 0.504 | 1.00 | 3 |
| viso_fake | fake | 0.573 | 2.00 | 5 |

**Direct observations** (numbered, no interpretation):

1. Across all 15 cohorts, mean abs diff lands in [0.50, 0.76]. None reach the spec's 1.0 floor.
2. p99 abs diff is in [1.0, 5.0] for every cohort. The spec's <80 ceiling is satisfied by ≥16× margin everywhere.
3. The aug is NOT a no-op on any cohort (every frame shows non-zero change, max abs diff 3-14 per cohort).
4. The aug is NOT garbled on any cohort (max single-pixel change is at most 14 / 255).
5. Tier 1's synthetic uniform image gives mean abs diff = 7.65 (≈14× larger than the face mean). The synthetic image has flat power across all bands; faces have low amplitude at radii 8-13.

## What the spec authorizes

Per §3.7: *"If Tier 3 fails (output is garbled or unchanged), the aug parameters need
adjustment but you do not change them unilaterally. Stop, write what you saw to
`analysis/p2_eval_2026-05-08/SLOT_D_SMOKE_FAIL.md`, and surface the issue to the user.
Do not launch."*

Operational reading: Tier 3 mathematically failed the `mean_abs > 1.0` floor, but neither
"garbled" nor "unchanged" describes the actual outcome. The aug produces a measurable,
bounded effect on every cohort tested.

## Decision points (mapped to user)

The user picks one of the following before launch can proceed:

- **A — Relax the Tier-3 floor and proceed unchanged.** Treat the spec's 1.0 floor as
  mis-calibrated for natural 224×224 face crops. Re-run smoke as `mean_abs_diff > 0.4`
  (every cohort passes that floor) and launch with the spec's stated parameter set
  (`bands_randomize=[8,9,10,11,12,13]`, `bands_preserve=[5,6]`, `noise_log_range=[-0.3,+0.3]`).
- **B — Boost noise to produce a more visible effect.** Change `noise_log_range` from
  `[-0.3,+0.3]` (factor ∈ [0.74, 1.35]) to e.g. `[-0.5,+0.5]` (factor ∈ [0.61, 1.65]).
  Estimated mean abs diff per cohort would scale ~linearly with the log range, landing
  around 0.85-1.30 average. This deviates from the spec's stated parameter pin.
- **C — Decline to launch Slot D.** Investigate whether a stronger aug variant is needed
  before committing GPU. Defer to a future session.

## Artifacts

- Smoke output images per trial: `/tmp/fourier_smoke/tier3_out_{0..4}.png` (Tier 3 source frame).
- Diverse-probe outputs cached under: `/tmp/fourier_smoke/diverse/`.
- Source: `data/augmentations/fourier_band_aug.py` (working tree, uncommitted).
- Wiring sites: `data/sources/combined_paired.py:3781-3791`, `data/batching/df40_paired.py:368-378`,
  `trainer/trainer.py:686-691`, `train_sweep.py:354-364`.

## Resolution (2026-05-08)

User picked option **A — relax floor, keep params**. Tier-3 floor relaxed from
`mean_abs_diff > 1.0` to `mean_abs_diff > 0.4`. Spec parameters unchanged.

Re-run outcome (same source frame, 5 RNG draws):
- mean abs diffs = [0.549, 0.544, 0.565, 0.563, 0.565] — all > 0.4 ✓
- p99 abs diffs = [2.00, 2.00, 2.00, 2.00, 2.00] — all < 80 ✓
- max single-pixel diffs = [3, 4, 5, 5, 4] — bounded ✓

All 3 tiers pass under the relaxed Tier-3 floor. Proceeding with commit + Cloud Build +
Slot D launch.
