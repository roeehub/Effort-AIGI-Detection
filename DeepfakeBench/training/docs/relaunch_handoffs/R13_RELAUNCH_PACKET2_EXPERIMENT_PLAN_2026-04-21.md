# R13 Relaunch Packet 2 Experiment Plan - 2026-04-21

## Goal

Use the next `6` FT slots while `RLP1_08` scratch and the tail of `RLP1_03`
are still live to answer the next relaunch questions in the right order:

1. does unenhanced proper-data still help once hints are removed entirely?
2. if enhanced proper-data is useful, is a **moderate** enhanced addition
   better than the packet-1 all-in full proper jump?
3. can we improve **prediction stability** on the best no-hints proper packet
   without reopening the old generic stability-loss dead end?

This packet is intentionally **FT-only**:

- scratch is already being covered by `RLP1_08`
- hints are already on a short leash from packet 1
- WT-C sidecars are deferred until the data packet is cleaner

## Packet-1 Conclusions That Drive Packet 2

The relevant packet-1 read is now stable enough to act on:

- `RLP1_01` remains the best overall FT control.
- `RLP1_04` is the only new data intervention that looked genuinely promising.
- `RLP1_02` and `RLP1_03` do not justify the hint family.
- `RLP1_05` shows that the live full-proper jump was too aggressive.
- `RLP1_06` and `RLP1_07` did not justify extra augmentation complexity.
- `RLP1_08` is still a hedge, not the current promotion candidate.

The live packet also already exposed a real prediction-stability question.
Current W&B `ood/score_jitter/*` summaries show:

| Run | YT jitter | Teams real jitter | Teams fake jitter | Read |
| --- | ---: | ---: | ---: | --- |
| `RLP1_01` | `0.04286` | `0.03946` | `0.08373` | strongest overall FT control |
| `RLP1_04` | `0.04038` | `0.04862` | `0.08640` | better YT jitter, worse Teams real jitter |
| `RLP1_05` | `0.04113` | `0.05464` | `0.08480` | full proper worsens Teams real jitter further |

So packet 2 should not just chase composite AUC. It should also keep explicit
eyes on prediction stability.

## Shared Scaffold

Unless a config explicitly says otherwise, packet-2 runs keep the current R13
relaunch FT scaffold:

- backbone: `vit_b_16_laion_datacomp` / `ViT-B-16-DataComp-XL`
- rank: `736` (`k=32`)
- frames per video: `8`
- frames per batch: `32`
- resolution: `224`
- seed / split seed / identity split seed: `737`
- train / val / test split: `0.85 / 0.10 / 0.05`
- base checkpoint: Track C `R12_G_FP32`
- learning rate: `3e-5`
- total steps: `10000`
- warmup: `400`
- ArcFace schedule: `s 6 -> 12`
- checkpoint selection: `best_ood_composite`
- gradient clipping: `1.0`

The packet keeps the same backbone and FT scaffold on purpose. Packet 2 is
about isolating the next **data** and **prediction-stability** choices, not
opening a broad hyperparameter sweep.

## What Packet 2 Is Not Doing

- No hints in the main packet. Packet 1 already failed to justify them.
- No generic `stability_lambda`. Repo history already showed that this hurts
  OOD and does not reduce jitter meaningfully.
- No new scratch run. `RLP1_08` is already covering the scratch hedge.
- No WT-C GammaUp / Teams-shadow sidecars. Packet 1 did not justify carrying
  that complexity forward yet.

## Why A Fresh Control Is Included

Mutable live-discovery sources are still changing, and proper-data must be
rebuilt before launch to pick up the latest HDTF / quickclips uploads.

That means packet 2 should not compare only against yesterday's packet-1
numbers. It needs one fresh control on the **same launch-time snapshot**.

## The Six Runs

| Slot | Config | Purpose | Main question |
| --- | --- | --- | --- |
| `01` | `R13_RLP2_01_FT_WTB1_no_hints_refresh_live.yaml` | fresh control on the current mutable-source snapshot | what does the honest no-hints packet do right now? |
| `02` | `R13_RLP2_02_FT_WTB1_plus_proper_unenhanced_live.yaml` | main packet-2 hypothesis | does no-hints + unenhanced proper-data beat the refreshed control? |
| `03` | `R13_RLP2_03_FT_WTB1_plus_proper_unenhanced_plus_teams_enhanced_live.yaml` | moderate target-domain enhanced step | does adding only Teams-enhanced proper-data help without the packet-1 full-proper overshoot? |
| `04` | `R13_RLP2_04_FT_WTB1_plus_proper_unenhanced_plus_clean_enhanced_live.yaml` | moderate clean-enhanced step | is the enhanced value coming from clean enhanced, or is that the wrong direction for target-domain work? |
| `05` | `R13_RLP2_05_FT_WTB1_plus_proper_unenhanced_spatial_stability_live.yaml` | spatial-jitter ablation on the best expected data packet | can stronger spatial variation reduce score flicker without giving away the packet? |
| `06` | `R13_RLP2_06_FT_WTB1_plus_proper_unenhanced_low_arcface_live.yaml` | lower-`s` stability / calibration hedge | does a gentler ArcFace surface improve stability on the same data packet? |

## Packet Logic

### 1. The main packet-2 bet

The highest-value unresolved question from packet 1 is:

`RLP1_04` looked good even though it was still carrying the hint ladder.

That makes the best next-step hypothesis:

- no hints
- unenhanced proper-data
- same FT scaffold

That is `RLP2_02`.

### 2. Moderate enhanced-proper tests

Packet 1 showed that the full live proper jump was too big. But that does not
prove that all enhanced proper-data is bad. It only proves that the
`RLP1_04 -> RLP1_05` jump was too aggressive.

Packet 2 therefore tests the enhanced proper-data question in a more
interpretable way:

- `RLP2_03`: add only `proper_visomaster_enhanced_teams`
- `RLP2_04`: add only `proper_visomaster_enhanced_clean`

This is better than a blind family-weight sweep because it keeps the data story
legible.

### 3. Prediction-stability probes

Packet 2 includes two explicit prediction-stability probes on the strongest
expected data packet:

- `RLP2_05`: stronger spatial augmentation
- `RLP2_06`: lower ArcFace endpoint

This is deliberately **not** generic stability regularization. The repo already
showed that `stability_lambda` was a negative result. The packet instead uses
the two levers that still make sense in-repo:

- more realistic spatial crop jitter / perturbation exposure
- a slightly gentler probability surface

## Approximate Data Shapes

These are launch-time approximations only. Final startup counts must be taken
from W&B after the pre-launch proper-data rebuild.

Using the current packet-1 launch snapshot as the reference:

- refreshed control (`RLP2_01`): about current `WTB1`
- `RLP2_02`: `RLP2_01` plus unenhanced proper lanes
  (`~ +684` proper fake rows at the last launch)
- `RLP2_03`: `RLP2_02` plus `proper_visomaster_enhanced_teams`
  (`~ +1484` at the last builder snapshot)
- `RLP2_04`: `RLP2_02` plus `proper_visomaster_enhanced_clean`
  (`~ +1484` at the last builder snapshot)
- `RLP2_05` / `RLP2_06`: same data packet as `RLP2_02`

The packet intentionally avoids the packet-1 full-proper `3186`-row jump.

## Prediction-Stability Measurement Rules

Packet 2 should be judged on **two** stability layers.

### A. Live training-time stability read

Every run should be monitored on:

- `val_primary/ood_composite`
- `ood/overall/auc`
- `val_holdout/overall/auc`
- `ood/score_jitter/external_youtube_avspeech`
- `ood/score_jitter/teams_ood_real`
- `ood/score_jitter/teams_ood_fake`
- `train/collapse/is_constant_output`
- `train/params_with_grad`

Interpretation rule:

- do not treat a run as a stability win just because one jitter number moves
  down if the packet collapses or the main balanced metric falls apart

### B. Required post-train stability pass

For the top `2-3` packet-2 checkpoints, run the frozen WT-D frame-policy pass
so we get:

- `stability_summary.csv`
- `stability_per_video.csv`
- `policy_checkpoint_summary.csv`

Use:

- `WT_D_MINIMAL_BASELINE_PACKET_2026-04-17.md`
- `tools/teams_frame_policy_analysis.py`

This is the required packet-2 prediction-stability closeout. Packet 2 should
not be judged only on the live training metrics.

## Freshness Rules Before Launch

### Mutable-source rule

Keep the packet cacheless for live mutable sources, same as packet 1:

- `deeplive`
- `visomaster_hints` if ever re-enabled later
- `teams`
- `external_training_reals`

### Proper-data rebuild rule

Immediately before launch, rerun:

```bash
python3 DeepfakeBench/training/arena/build_visomaster_proper_data_artifacts.py
```

Then capture the startup sample counts from W&B for:

- control packet total
- unenhanced proper lanes
- enhanced clean lane
- enhanced Teams lane

Those launch-time counts are the source of truth for packet-2 interpretation.

## Expected Reads

- `RLP2_02` is the main expected winner or near-winner.
- `RLP2_03` is the best enhanced-proper rescue hypothesis because it is still
  target-domain-oriented.
- `RLP2_04` is the diagnostic control for whether clean-enhanced is helping or
  just diluting the target-domain story.
- `RLP2_05` and `RLP2_06` are not expected to be obvious packet leaders on the
  main composite metric; they exist to see whether prediction stability can be
  improved honestly on the strongest clean data packet.

## Launch Order

If slots are constrained, launch in this order:

1. `RLP2_01`
2. `RLP2_02`
3. `RLP2_03`
4. `RLP2_04`
5. `RLP2_05`
6. `RLP2_06`

That ordering preserves the highest-value data answers first and keeps the
stability-specific probes attached to a known-good data packet.
