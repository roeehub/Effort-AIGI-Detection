# Teams Decision-System Reinvestigation

**Date:** April 17, 2026  
**Track:** `WT-D` Stability / Decision-System Analysis

## Purpose

Re-investigate the same-day no-new-training question before changing scripts:

- can today’s real-Teams pain move materially at the decision layer
- which decision-layer opportunities should be ranked first
- what can be stated honestly from the current runtime versus what still needs a
  shortlist rerun

## Runtime Boundary

The usable local evidence boundary in this runtime is narrower than the ideal
measurement path.

Established here:

- the reduced five-checkpoint shortlist is still the right one:
  - `R12_G_FP32`
  - `R13_A_STEP15500`
  - `R13_E_BESTSOFAR`
  - `R13_FT7_FP32`
  - `R13_FT9_FP32`
- fixed-threshold `0.5` is still not the promotion contract
- older repo-local strategy artifacts still show real decision-layer leverage
- no accessible calibrated shortlist scorecard or raw per-video shortlist dump is
  present in this runtime

Practical runtime blocker:

- this shell has no `gcloud`, no `gsutil`, no `google-cloud-storage`, and no
  Google credentials configured
- current GCS-hosted shortlist reports therefore cannot be pulled or rescored
  from this runtime as-is

Planning consequence:

- `WT-D` can still produce checked-in analysis tooling, tests, commands, and an
  explicit judgment
- `WT-D` cannot honestly claim a calibrated five-checkpoint promotion winner
  from this runtime alone

## Re-Ranked Decision-Layer Opportunities

### 1. Per-checkpoint calibrated threshold sweep on the reduced shortlist

This remains the highest-value same-day no-new-training move.

Why it stays first:

- it directly addresses the known selection mismatch
- it is the cleanest way to re-rank the five frozen checkpoints
- it uses existing reports and avoids new training risk

Required contract:

1. minimize `teams_real_all_dev` FPR
2. then minimize worst-slice FPR across:
   - `teams_real_poor_quality_dev`
   - `teams_real_lighting_extreme_dev`
3. then maximize fake recall on:
   - `teams_fake_all_dev`
   - `visomaster_enhanced_macro_dev`
   - `deeplive_enhanced_dev`
4. freeze threshold before lockbox

### 2. Frame-report temporal aggregation on checkpoint-specific thresholds

This is the best next measurement once threshold selection is frozen.

Why it is second:

- repo inference already exposes mean, median, and majority-style video logic
- it directly targets frame flicker without retraining
- it gives a clean bridge from score instability to deployable alert policy

Recommended first comparison set:

- mean
- median
- majority
- one EMA-last variant

### 3. Hysteresis or minimum-positive-run policies

This should be tested immediately after basic aggregation, not before it.

Why it is third:

- it is the simplest stateful way to suppress single-frame or short-run spikes
- it is operationally closer to live alerting than raw frame thresholds
- it is cheaper and more honest than another generic stability-loss retrain

Recommended first sweep:

- one selected threshold per checkpoint
- raise/clear margins around that threshold
- `raise_run >= 2`

### 4. Narrow abstain band

This remains valuable, but only if uncertain outputs are operationally allowed.

Why it is fourth:

- older repo artifacts already suggested FPR can drop materially with a small
  uncertain zone
- the gain should be judged together with uncertain rate and coverage, not only
  with clean accuracy

### 5. Crop-disagreement or perturbation gate

This stays behind the items above.

Why it is fifth:

- the idea is plausible
- older artifact support is weaker than calibration + aggregation
- it depends on richer per-frame or perturbation outputs than are currently
  accessible here

## Explicit Judgment

**Yes: decision-layer leverage is strong enough to justify immediate no-new-data
work.**

Reasoning:

- repo-local history already shows threshold policy can move Teams FPR
  materially
- repo-local history already shows abstain logic and mild gates are not
  cosmetic
- repo-local instability evidence still points to frame flicker and threshold
  portability as live problems
- the strongest completed model-side stability intervention was negative, which
  increases the relative value of decision-layer work right now

What this judgment does **not** mean:

- it does not mean the missing enhanced-through-Teams condition is solved
- it does not mean a calibrated promotion winner is already known
- it does not mean decision policy replaces future target-condition data

## Checkpoint Ranking: Current Rankable State

An honest full `1..5` calibrated ranking is still unavailable in this runtime
because the required shortlist reports are not locally accessible.

What can be ranked honestly now:

### Tier 1: current incumbent deployment reference

- `R12_G_FP32`

Why:

- it remains the strongest directly evidenced real-side safety reference
- no locally accessible calibrated shortlist artifact disproves it
- the only locally documented fixed-threshold frozen Teams comparison still puts
  `R13_A_STEP15500` below it on the primary real-side objective

### Tier 2: unresolved calibrated challengers

- `R13_FT7_FP32`
- `R13_FT9_FP32`
- `R13_E_BESTSOFAR`

Why:

- they belong in the measured shortlist
- they must still be judged on the calibrated low-FP contract
- this runtime has no accessible shortlist reports or raw prediction dumps for a
  defensible total order

### Tier 3: locally evidenced fake-gain / real-regression branch

- `R13_A_STEP15500`

Why:

- this checkpoint clearly improves some fake-side slices in the available frozen
  Track C readout
- the same readout also shows worse real Teams FPR than `R12_G_FP32` at
  threshold `0.5`
- without calibrated shortlist reports, it should be treated as a real
  contender, but not as the incumbent safety leader

Bottom line:

- `R12_G_FP32` stays the incumbent safety reference
- the remaining shortlist must be rerun before a calibrated promotion order is
  named
- do not force a fake-certainty total order from the current runtime

## Stability Evidence

### Existing repo evidence still stands

- near-identical frames were already observed to produce materially different
  scores
- production later still reported frame-to-frame volatility on real Teams calls
- generic stability regularization and label smoothing remain negative repo
  results

### New checked-in measurement path

This track adds two report-driven tools:

- `tools/teams_video_policy_analysis.py`
  - consumes `*_videos_report.csv`
  - sweeps thresholds under the low-FP contract
  - emits checkpoint ranking tables and abstain-band comparisons
- `tools/teams_frame_policy_analysis.py`
  - consumes `*_frames_report.csv`
  - emits frame-jitter and threshold-flip summaries
  - compares temporal aggregation, consecutive-positive, and hysteresis policies

The frame-policy tool measures:

- mean jitter
- p95 jitter
- threshold flip rate at selected thresholds
- contract-style suite metrics for:
  - mean
  - median
  - majority
  - EMA-last
  - consecutive-positive
  - hysteresis

Note:

- the checked-in tests validate tool behavior on synthetic fixtures
- they are not themselves production or shortlist model results

## Reproducible Commands

All commands below are intended to be run from `DeepfakeBench/training`.

### 1. Calibrated threshold + abstain analysis from video reports

```bash
python3 tools/teams_video_policy_analysis.py \
  --report_root /path/to/reports \
  --checkpoint_map arena/checkpoint_maps/teams_target_domain.r13_finalists_2026-04-12.yaml \
  --checkpoints R12_G_FP32,R13_A_STEP15500,R13_E_BESTSOFAR,R13_FT7_FP32,R13_FT9_FP32 \
  --output_dir /tmp/wt_d_video_policy_2026-04-17
```

If the report root is `gs://...`, run the same command inside the training
environment that already has `google-cloud-storage` available.

### 2. Frame stability + temporal / hysteresis analysis from frame reports

```bash
python3 tools/teams_frame_policy_analysis.py \
  --report_root /path/to/reports \
  --checkpoint_map arena/checkpoint_maps/teams_target_domain.r13_finalists_2026-04-12.yaml \
  --checkpoints R12_G_FP32,R13_A_STEP15500,R13_E_BESTSOFAR,R13_FT7_FP32,R13_FT9_FP32 \
  --checkpoint_summary_csv /tmp/wt_d_video_policy_2026-04-17/checkpoint_summary.csv \
  --output_dir /tmp/wt_d_frame_policy_2026-04-17
```

### 3. Local verification for the checked-in tooling

```bash
python3 -m unittest DeepfakeBench.training.tests.test_teams_video_policy_analysis -v
python3 -m unittest DeepfakeBench.training.tests.test_teams_frame_policy_analysis -v
```

## Practical Recommendation

Use the next no-new-training measurement budget in this order:

1. rerun the five-checkpoint shortlist on the calibrated low-FP contract
2. carry the selected threshold per checkpoint into frame-report temporal and
   hysteresis analysis
3. only then decide whether an abstain band or a small gate is worth the extra
   operational complexity

Do not spend the next same-day cycle on another weight-only fine-tune family
before the calibrated shortlist and frame-policy comparisons are available.
