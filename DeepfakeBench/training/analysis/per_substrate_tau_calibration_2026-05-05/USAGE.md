# Per-substrate Tau Calibration

Purpose: pick a single deployable tau that satisfies a worst-substrate-FPR
ceiling on a held-out real-frame set. Substrate is unobservable at deployment
(per user constraint: no per-mode tau in production), so we cannot ship a
substrate-conditional policy — but we CAN use offline substrate metadata to
pick a single global tau that won't blow up on the worst substrate.

## Quick start (Packet A / Packet C-codec when they finish)

When `frames_report.csv` files for the new checkpoint land in
`analysis/cpu_followups_2026-05-04/raw_reports/` (or any other dir):

```bash
python3 analysis/per_substrate_tau_calibration_2026-05-05/run_calibration.py \
    --real-dev    PATH_TO/teams_real_all_dev_<ckpt>_frames_report.csv \
    --real-lockbox PATH_TO/teams_real_all_lockbox_<ckpt>_frames_report.csv \
    --fake-suite "name=viso_dev path=PATH_TO/visomaster_enhanced_macro_dev_<ckpt>_frames_report.csv" \
    --fake-suite "name=deeplive_dev path=PATH_TO/deeplive_enhanced_dev_<ckpt>_frames_report.csv" \
    --fake-suite "name=teams_fake_dev path=PATH_TO/teams_fake_all_dev_<ckpt>_frames_report.csv" \
    --fake-suite "name=teams_fake_lockbox path=PATH_TO/teams_fake_all_lockbox_<ckpt>_frames_report.csv" \
    --score-col frame_prob --label-col label --path-col frame_path \
    --out analysis/per_substrate_tau_calibration_2026-05-05/run_<ckpt>/
```

To verify the tool against the P8A reference, use the built-in profile:

```bash
python3 analysis/per_substrate_tau_calibration_2026-05-05/run_calibration.py \
    --profile p8a_reference_step5000 \
    --out analysis/per_substrate_tau_calibration_2026-05-05/reference_run_p8a/
```

## Inputs expected

Each frames_report CSV needs columns:

- `frame_path` (the GCS uri or full path used in the lockbox-tagging parquet)
- `frame_prob` (model score in [0,1])
- `label` (0 = real, 1 = fake)

Other columns are ignored.

## Substrate definition

Substrate := `clip_capture_mode` from
`analysis/lockbox_tagging/full_tags_2026-04-27.parquet`
(7334 frames, 5 buckets: `normal_photo, webcam, phone_screen, screen,
screen_recording`). Frames not in that parquet are bucketed as `unknown` and
appear in sweep columns as `fpr_dev_unknown` / `fpr_lockbox_unknown`. They
are still treated as real frames for the worst-substrate-FPR computation.

The substrate manifest is written to `substrate_manifest.json` in the
output dir for reproducibility.

## Outputs

| File | Contents |
|---|---|
| `tau_sweep.csv` | One row per tau in the grid. Columns: `tau`, `fpr_dev_<substrate>`, `fpr_dev_worst_substrate`, `fpr_dev_global`, `fpr_lockbox_<substrate>`, `fpr_lockbox_worst_substrate`, `fpr_lockbox_global`, `recall_<suite>` for each fake suite. |
| `tau_recommendations.json` | Three taus picked by FPR target. For each, full per-substrate FPR + per-suite recall, plus a naive global-pooled baseline and an oracle per-mode tau policy. |
| `substrate_manifest.json` | Substrate definition + per-frame-set substrate counts. |
| `substrate_assignment.csv` | Per-frame substrate label, for audit. |

## How to read `tau_recommendations.json`

`selections.tau_strict / tau_moderate / tau_loose` correspond to FPR ceilings
of 5% / 10% / 20% on the worst substrate. Each has three policy slots:

1. `substrate_aware_single_tau` — what to ship: lowest tau where worst-substrate
   FPR on dev reals ≤ ceiling. Single number; substrate not needed at inference.
2. `naive_global_pooled_single_tau` — calibration baseline: lowest tau where
   pooled global dev FPR ≤ ceiling. Mis-controls per-substrate FPR; usually
   over-fires on webcam.
3. `per_mode_tau_oracle` — UPPER BOUND, NOT shippable. Different tau per
   substrate, calibrated to the same within-substrate FPR ceiling.

`recall_lift_substrate_aware_minus_naive_pp` shows the per-suite recall delta
between substrate-aware and naive policies. Negative values mean naive has
higher recall at the cost of higher worst-substrate FPR (the naive baseline
is structurally weaker and only included as a comparator).

## Substrates exposed at deployment time

The deployable tau (`substrate_aware_single_tau.tau`) does NOT require
substrate at inference. It's a single global threshold whose FPR was checked
against substrate-stratified dev reals.

## Reference run on P8A (see `reference_run_p8a/`)

| Policy | tau | dev worst-substrate FPR | lockbox global FPR | recall (teams_fake_lockbox) | recall (viso_dev) | recall (deeplive_dev) |
|---|---|---|---|---|---|---|
| **strict** (5% ceiling) — substrate-aware | 0.9941 | 3.91% | 0.07% | 16.24% | 0.18% | 0.00% |
| **moderate** (10% ceiling) — substrate-aware | 0.9891 | 9.50% | 0.35% | 24.94% | 1.64% | 4.04% |
| **loose** (20% ceiling) — substrate-aware | 0.8409 | 19.81% | 2.68% | 46.82% | 20.91% | 31.01% |

For Job-7 reproduction (per-mode oracle vs single global tau at TAU_F0_DEV
0.70503): per-mode dev-calibrated achieves teams_fake_lockbox recall 78.59%
at 16.91% lockbox real FPR vs 54.12% recall at 4.30% lockbox FPR for the
single global tau — a 24.5pp recall lift at 4× the FPR. This validates the
~21pp lift claim in memory `project_job7_head_retrain_REFUTED_2026-05-04`.

## CLI options

| Flag | Default | Notes |
|---|---|---|
| `--profile p8a_reference_step5000` | — | Pin all paths to the P8A reference |
| `--real-dev` | required (when no profile) | Path to dev real-frame scores CSV |
| `--real-lockbox` | optional | Held-out lockbox real-frame scores CSV (for diagnostic FPR readout) |
| `--fake-suite "name=X path=Y"` | repeatable | One per fake suite |
| `--score-col` | `frame_prob` | Column name for model score |
| `--label-col` | `label` | Column name for {0,1} label |
| `--path-col` | `frame_path` | Column name matching the parquet's `gcs_uri` |
| `--tags-parquet` | `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` | Substrate source |
| `--substrates` | the 5 capture modes | Override if you want a different bucketing |
| `--tau-grid-n` | 100 | Linear grid size |
| `--tau-grid-lo` `--tau-grid-hi` | 0.05 0.95 | Linear grid range; quantile anchors are added automatically |
| `--out` | required | Output dir |

## Limits

- Substrate coverage is partial: lockbox_tagging parquet has 7334 frames; some
  fake-suite frames (notably `viso_dev`) have 0 substrate hits. Their
  per-suite recall under the single-tau policy is unaffected (suite recall
  doesn't condition on substrate), but the oracle per-mode policy can't score
  them.
- The "worst-substrate FPR" semantics treat `unknown` substrate as a substrate
  bucket. If you don't want to penalise tau picks for the unknown bucket,
  filter the substrate set with `--substrates` to only include the named
  capture modes (default behaviour: include only the 5 named modes; unknown
  is not in the worst-substrate calculation).
