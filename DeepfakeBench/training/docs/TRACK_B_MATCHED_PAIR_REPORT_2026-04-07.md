# Track B Matched-Pair Report

**Date:** April 7, 2026  
**Scope:** first current-bucket Track B rerun for Microsoft Teams channel re-characterization  
**Status:** sidecar-only, no active `phase2_round13` config changes

## 1. What Was Run

New rerunnable analysis tool:

- `DeepfakeBench/training/tools/analyze_teams_matched_pairs.py`

Command used for this first report:

```bash
python DeepfakeBench/training/tools/analyze_teams_matched_pairs.py \
  --standard-max-samples 6 \
  --enhanced-max-base-samples 3 \
  --frame-indices 0,4,8,12 \
  --sim-repeats 1 \
  --output-json /tmp/teams_matched_pairs_2026-04-07.json \
  --output-csv /tmp/teams_matched_pairs_2026-04-07.csv
```

This compares matched pairs from:

- clean original -> Teams v1 real
- clean original -> Teams v1 fake
- clean original -> Teams v2 real
- clean original -> Teams v2 fake
- clean enhanced VisoMaster fake -> Teams-enhanced VisoMaster fake

Active config sanity check:

- `rg -n "teams_codec_simulation" DeepfakeBench/training/experiments/phase2_round13 -S`
- result: no matches
- interpretation: active `phase2_round13` configs still do not enable `teams_codec_simulation`

## 2. Coverage

This first rerun processed `186` matched frame pairs:

- `teams_v1_real`: `21` frames from `6` samples
- `teams_v1_fake`: `21` frames from `6` samples
- `teams_v2_real`: `24` frames from `6` samples
- `teams_v2_fake`: `24` frames from `6` samples
- `visomaster_enhanced_to_teams`: `96` frames from `3` base sample IDs across all `8` enhancers

This is a deterministic sample-based report, not a full-bucket exhaustive sweep.

## 3. Main Findings

### 3.1 Ordinary current Teams slices are not uniformly blur-dominant

Aggregate deltas on the current ordinary buckets:

| Group | Sharpness | Brightness | Noise | HF energy | BPP |
|---|---:|---:|---:|---:|---:|
| `teams_v1_real` | `+24.1%` | `+17.6%` | `+20.0%` | `-31.5%` | `+7.6%` |
| `teams_v1_fake` | `+103.4%` | `+34.8%` | `+42.1%` | `+109.3%` | `+28.2%` |
| `teams_v2_real` | `+14.8%` | `+19.0%` | `+33.3%` | `+51.8%` | `+11.9%` |
| `teams_v2_fake` | `+45.6%` | `+23.4%` | `+26.0%` | `+47.6%` | `+10.1%` |

Interpretation:

- the current ordinary v1/v2 buckets do **not** support a single “Teams mostly smooths everything” story
- brightness increase is still stable
- sharpness / noise / high-frequency behavior is mixed or positive on most ordinary current slices

### 3.2 The enhanced-through-Teams slice is materially different

Aggregate deltas for clean enhanced VisoMaster fake -> Teams-enhanced fake:

- sharpness: `-23.2%`
- brightness: `+33.3%`
- contrast: `+21.9%`
- noise: `+17.1%`
- high-frequency energy: `-81.8%`
- BPP: `+8.0%`

Interpretation:

- this slice is the only aggregate group that still looks broadly blur / HF-loss dominant
- the enhanced-through-Teams problem is real, but it should not define the whole Teams policy by itself

### 3.3 The current `TeamsCodecSimulation` does not fit the ordinary current slices

Direction-match accuracy of the existing simulator on this rerun:

- `teams_v1_real`: `37.5%`
- `teams_v1_fake`: `50.0%`
- `teams_v2_real`: `37.5%`
- `teams_v2_fake`: `50.0%`
- `visomaster_enhanced_to_teams`: `62.5%`

Largest recurrent misses:

- it pushes sharpness negative where ordinary current buckets are often mixed or positive
- it pushes noise negative while all five measured groups were positive in aggregate
- it pushes BPP negative while all five measured groups were positive in aggregate

What it still gets roughly right:

- brightness increase
- some of the enhanced-through-Teams high-frequency loss behavior

Interpretation:

- the existing single blur-heavy simulator is not training-safe as a general current-Teams approximation
- it is closer to the enhanced-through-Teams slice than to the ordinary current slices

### 3.4 Mixed behavior remains visible inside the enhanced slice

Enhancer-level sharpness deltas:

- `gpen-256`: `-30.3%`
- `restoreformer++`: `-19.2%`
- `gpen-1024`: `-33.9%`
- `gpen-2048`: `-33.9%`
- `gpen-512`: `-29.1%`
- `codeformer`: `-9.6%`
- `gfpgan`: `+0.8%`
- `vqfr-v2`: `+1.2%`

Swap-model-level sharpness deltas on the enhanced-through-Teams slice:

- `GhostFace-v1`: `-22.3%`
- `InStyleSwapper256-A`: `-42.1%`
- `InStyleSwapper256-B`: `+8.2%`

Interpretation:

- even the enhanced-through-Teams slice is not one clean mode
- a single fixed preset is probably too coarse
- the data is already pointing toward a mixture or family-conditional policy

## 4. Track B Recommendation

For now:

1. Keep the active `phase2_round13` configs unchanged.
2. Do **not** re-enable the current `TeamsCodecSimulation` globally.
3. Do **not** treat `vcd_targeted` as equivalent to a validated current-Teams model.

If Track B continues into code experiments, the next sidecar step should be one of:

1. A two-mode synthetic policy:
   - ordinary current Teams mode with brightness/contrast lift and a mixed sharpness branch
   - enhanced-through-Teams mode with stronger HF-loss / smoothing
2. A family-conditional policy:
   - keep ordinary non-enhanced families off the old blur-heavy preset
   - target `visomaster_enhanced_fake` first
   - optionally include `deeplive_enhanced_fake` later

The current data does **not** justify promoting the old R9-era single-mode Teams simulator back into the active main line.

## 5. Limits Of This First Rerun

- deterministic sample, not exhaustive full-bucket coverage
- `sim_repeats=1` for speed
- enhanced slice covers `3` base sample IDs in this first pass

So this report should be used as:

- a Track B gating note that blocks blind promotion of the current simulator
- a starting point for the next sidecar experiment

It should **not** yet be treated as the final Teams augmentation redesign.

## 6. Sidecar Prototype Comparison

After the first readout, three sidecar policies were compared on the same
`186` matched frame pairs:

1. `legacy_single`
2. `adaptive_mixture`
3. `family_split`

New code added for that comparison:

- `TeamsAdaptiveCodecSimulation`
- `TeamsHybridCodecSimulation`

Raw local artifacts:

- `/tmp/teams_matched_pairs_legacy_2026-04-07.json`
- `/tmp/teams_matched_pairs_adaptive_2026-04-07.json`
- `/tmp/teams_matched_pairs_family_split_2026-04-07.json`

Macro comparison:

| Policy | Ordinary dir acc | Ordinary MAE | Enhanced dir acc | Enhanced MAE |
|---|---:|---:|---:|---:|
| `legacy_single` | `43.8%` | `36.7` | `62.5%` | `21.9` |
| `adaptive_mixture` | `62.5%` | `32.8` | `50.0%` | `24.3` |
| `family_split` | `62.5%` | `31.6` | `62.5%` | `22.4` |

Interpretation:

- `adaptive_mixture` improved the ordinary current Teams slices materially
- but it gave back too much on the true enhanced-through-Teams slice
- `family_split` kept the ordinary-slice gain while restoring the enhanced
  slice to legacy-level direction match

Current best sidecar candidate:

- `family_split`

What it means:

- enhanced fake families should keep the blur-heavy legacy policy
- ordinary non-enhanced families should use the adaptive mixture

This is the first Track B result that looks good enough to justify a dedicated
sidecar ablation.

## 7. Candidate Sidecar Config

This is the current best candidate policy to test in a sidecar experiment:

```yaml
augmentation:
  teams_codec_simulation:
    enabled: true
    probability: 0.15
    policy: "family_split"
    exclude_families:
      - "deeplive_teams_fake"
      - "deeplive_teams_real"
    enhanced_families:
      - "visomaster_enhanced_fake"
      - "deeplive_enhanced_fake"
    ordinary_mode_probability_non_enhanced: 0.75
    ordinary_mode_probability_enhanced: 0.25
```

Important:

- this should stay sidecar-only until it earns promotion with a real training
  ablation and target-domain scorecard readout
- it should not be silently merged into the active Track A baseline

Drafted sidecar experiment:

- `DeepfakeBench/training/experiments/phase2_round13/R13_TB1_trackB_family_split_sidecar.yaml`

Suggested launch command:

```bash
cd DeepfakeBench/training
./launch_experiment.sh -y phase2r13-experiments asia-southeast1 \
  experiments/phase2_round13/R13_TB1_trackB_family_split_sidecar.yaml
```
