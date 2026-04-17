# WT-D Minimal Baseline Packet

**Date:** April 17, 2026  
**Purpose:** define the smallest worthwhile WT-D measurement pass before the new
retraining wave finishes

## Why this exists

WT-D should not become an open-ended measurement sink.

The goal is **not** to run every possible stability and decision experiment
right now. The goal is to create one compact, repeatable baseline packet that:

- gives a fair pre-retrain reference
- is cheap enough to run alongside the other tracks
- can be rerun after retraining with the same rules

## The minimal packet

### Scope to keep

Checkpoint set:

- `R12_G_FP32`
- `R13_A_STEP15500`
- `R13_E_BESTSOFAR`
- `R13_FT7_FP32`
- `R13_FT9_FP32`

Do **not** add:

- `R13_FT8_FP32`
- `R13_FT10_FP32`
- any new speculative checkpoints

Suite set:

- use the frozen Teams target-domain suite manifest already in repo:
  `arena/target_domain_suites.teams_manifest.frozen_2026-04-06.yaml`

Primary outputs:

1. threshold-ranked checkpoint summary
2. abstain-band summary
3. frame-stability summary
4. compact temporal/hysteresis policy summary

### Scope to skip for now

Do **not** spend baseline budget on:

- the mixed-source mega-eval
- extra per-session deep dives unless the baseline result is ambiguous
- crop-disagreement / perturbation-gate experiments
- more than one EMA setting
- more than one or two hysteresis settings
- reopening `FT8/FT10`

## Minimal execution plan

### Option A: cheapest path if reports already exist

If a shortlist reports root already exists and contains:

- `*_videos_report.csv`
- `*_frames_report.csv`

then do **not** rerun validation first. Just run the WT-D tools on that reports
root.

From `DeepfakeBench/training`:

```bash
python3 tools/teams_video_policy_analysis.py \
  --report_root /path/to/reports \
  --checkpoint_map arena/checkpoint_maps/teams_target_domain.r13_finalists_2026-04-12.yaml \
  --checkpoints R12_G_FP32,R13_A_STEP15500,R13_E_BESTSOFAR,R13_FT7_FP32,R13_FT9_FP32 \
  --output_dir /tmp/wt_d_minimal_video_2026-04-17
```

```bash
python3 tools/teams_frame_policy_analysis.py \
  --report_root /path/to/reports \
  --checkpoint_map arena/checkpoint_maps/teams_target_domain.r13_finalists_2026-04-12.yaml \
  --checkpoints R12_G_FP32,R13_A_STEP15500,R13_E_BESTSOFAR,R13_FT7_FP32,R13_FT9_FP32 \
  --checkpoint_summary_csv /tmp/wt_d_minimal_video_2026-04-17/checkpoint_summary.csv \
  --policy_families mean,median,majority,ema_last,hysteresis \
  --ema_alphas 0.35 \
  --hysteresis_margins 0.05 \
  --hysteresis_raise_runs 2 \
  --hysteresis_clear_runs 1 \
  --output_dir /tmp/wt_d_minimal_frame_2026-04-17
```

### Option B: one fresh baseline run if reports do not already exist

If no suitable reports root exists, do one shortlist-only validation pass using
the frozen Teams suite, with detailed reports enabled so frame and video reports
are both emitted.

From `DeepfakeBench/training`:

```bash
python3 arena/run_target_domain_validation_sequential.py \
  --checkpoint_map arena/checkpoint_maps/teams_target_domain.r13_finalists_2026-04-12.yaml \
  --checkpoints R12_G_FP32,R13_A_STEP15500,R13_E_BESTSOFAR,R13_FT7_FP32,R13_FT9_FP32 \
  --suite_manifest arena/target_domain_suites.teams_manifest.frozen_2026-04-06.yaml \
  --output_gcs_folder gs://REPLACE_ME/test_results/wt_d_minimal_baseline_2026-04-17/reports \
  --detailed_reports
```

Then run the two WT-D tools against that reports root.

## What to read first

If time is short, read these outputs in this order:

1. `checkpoint_summary.csv`
   - primary checkpoint ordering under the low-FP contract
2. `policy_checkpoint_summary.csv`
   - whether simple temporal or hysteresis logic changes the practical story
3. `stability_summary.csv`
   - whether some checkpoints are much more flickery than others
4. `abstain_band_summary.csv`
   - only if abstain outputs are operationally acceptable

## What counts as success for this minimal packet

This baseline packet is successful if it answers these four questions:

1. Does `R12_G_FP32` still lead on the low-FP contract, or does a challenger
   beat it when thresholded correctly?
2. Does simple temporal logic materially reduce real Teams false positives?
3. Is one checkpoint clearly more unstable than the others on frame jitter /
   threshold flips?
4. Is an abstain band worth operational discussion, or can it be ignored for
   now?

If those four questions are answered, the packet is already good enough for the
current stage.

## Suggested follow-up only if the result is ambiguous

Only expand beyond the minimal packet if:

- two checkpoints remain too close to call after the baseline readout
- or a policy choice changes the ordering enough that you need a narrower second
  pass

If that happens, the first expansion should be:

- rerun frame-policy comparison only on the top 2 checkpoints from
  `checkpoint_summary.csv`

Do **not** broaden the checkpoint list first.

## Relationship to retraining

This packet is intended to be run twice:

1. now, as the pre-retrain baseline
2. later, on the new experiment checkpoints

The second run should reuse the same:

- shortlist rule
- suite rule
- policy family
- output tables

That is the whole point of keeping the packet small.
