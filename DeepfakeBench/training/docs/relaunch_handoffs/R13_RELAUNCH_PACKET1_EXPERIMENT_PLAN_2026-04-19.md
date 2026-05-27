# R13 Relaunch Packet 1 Experiment Plan - 2026-04-19

## Goal

Use the first 8-slot launch window to answer the relaunch questions in the
right order:

1. What is the best honest data packet under the April 17 relaunch contract?
2. Once proper-data is present, do the retained weak-signal hints still help?
3. Does one light nuisance sidecar help the strongest data packet?
4. Is the FT family enough, or does a scratch hedge still matter?

This packet is intentionally data-dominant:

- 5 runs on data composition
- 2 WT-C-style augmentation sidecars on the strongest expected FT packet
- 1 scratch hedge on the strongest expected data packet

## Shared Scaffold

These packet decisions stay fixed unless a config explicitly says otherwise:

- backbone: `vit_b_16_laion_datacomp` / `ViT-B-16-DataComp-XL`
- rank: `736` (`k=32`)
- frames per video: `8`
- frames per batch: `32`
- resolution: `224`
- sampling strategy: `identity_resample_weighted`
- seed: `737`
- train / val / test splits: `0.85 / 0.10 / 0.05`
- OOD monitoring cadence and checkpointing stay aligned with the current R13 relaunch family

Fine-tune runs (`RLP1_01` through `RLP1_07`) share the current relaunch FT
scaffold:

- init: Track C `R12_G_FP32` scorecard-base checkpoint
- learning rate: `3e-5`
- total steps: `10000`
- warmup: `400`
- ArcFace schedule: `s 6 -> 12`

The scratch hedge (`RLP1_08`) keeps the same backbone, splits, and data packet
but switches to an R12_G-style scratch schedule:

- no base checkpoint
- learning rate: `2e-4`
- total steps: `30000`
- warmup: `1500`
- ArcFace schedule: `s 10 -> 14`

## Data Reality That Drives This Packet

### Proper-data

Post-launch correction from `2026-04-20`: the packet did **not** launch with
the `221 / 985 / 2412` numbers originally copied into this plan.

Current in-tree WT-F artifact snapshot:

- retained strict clean / Teams captures: `1826`
- total manifest videos: `7304`
- fake-lane totals in the checked-in builder report / manifest:
  - `proper_visomaster_clean`: `342`
  - `proper_visomaster_teams`: `342`
  - `proper_visomaster_enhanced_clean`: `1484`
  - `proper_visomaster_enhanced_teams`: `1484`

Launched RLP1 packet scale to use when interpreting the current live runs:

- `RLP1_04` live proper fake rows: `684` = `342 + 342`
- `RLP1_05/06/07/08` live proper fake rows: `3186` = `342 + 1251 + 1251 + 342`

Implications:

- the `03 -> 04` step is larger than the written plan said
- the `04 -> 05` step is materially larger than the written plan said
- future packets must record counts from the builder report and then re-check
  them against the launched run's startup W&B summary

The packet uses the strict retained subset on purpose. Runtime is still robust
to ragged future manifests, but tonight's packet should not silently pretend it
is training on every raw bucket row.

### Hints

Hints remain a deliberately small weak-signal residue from the old bad
VisoMaster pool:

- full bad-data pool: `5589`
- retained hints total: `682`
- retained fraction: `12.2%`

Hint composition:

- baseline hints: `480`
- Teams-played hints: `202`

Important: the hint lane ignores enhancer distinctions even if source metadata
mentions them. Hints are treated only as:

- `visomaster_hints`
- `visomaster_hints_teams`

### Core non-hint packet before proper-data

The honest non-hint relaunch baseline before proper-data is:

- `DF40`: `4698`
- DeepLive family total: `3151`
  - non-enhanced: `1659`
  - enhanced: `1492`
- external VCD reals: `63`

Outside hints, legacy non-hint VisoMaster is no longer part of this packet.
Non-hint VisoMaster only returns through explicit `proper_data`.

## Why These 8 Runs

The packet is designed to answer the largest-value questions first.

### Data-composition ladder

`RLP1_01` through `RLP1_05` walks the data story in the order we actually care
about:

1. clean honest baseline
2. add baseline hints
3. add Teams-played hints
4. add unenhanced proper-data
5. add full proper-data snapshot

This tells us:

- whether hints help at all
- whether Teams-played hints add anything beyond baseline hints
- whether proper-data adds signal beyond weak-signal hints
- whether enhanced proper-data matters beyond the unenhanced slice

### Augmentation sidecars

`RLP1_06` and `RLP1_07` do not create a separate training family. They hold the
strongest expected FT packet fixed (`RLP1_05`) and test two narrow overlays:

- truthful GammaUp
- light Teams shadow passthrough augmentation

This keeps WT-C informative without letting nuisance tests consume the whole
night.

### Scratch hedge

`RLP1_08` exists for one reason: a pure FT-only packet could miss a better
solution because the initializer still carries the old data semantics. One
scratch run on the strongest expected data packet is the minimum honest hedge.

## The 8 Configs

| Slot | Config | Purpose | Main question |
| --- | --- | --- | --- |
| 01 | `R13_RLP1_01_FT_WTB1_no_hints_live.yaml` | Clean relaunch control | What does the honest no-hints baseline do on its own? |
| 02 | `R13_RLP1_02_FT_WTB2_hints_only_live.yaml` | Add `480` baseline hints | Do small weak-signal hints help over the clean baseline? |
| 03 | `R13_RLP1_03_FT_WTB3_hints_plus_teams_hints_live.yaml` | Add all `682` retained hints | Do Teams-played hints add value beyond baseline hints? |
| 04 | `R13_RLP1_04_FT_WTB3_plus_proper_unenhanced_live.yaml` | Add `684` unenhanced proper fake samples at launch | Does clean target-domain proper-data already beat hints-only WTB3? |
| 05 | `R13_RLP1_05_FT_WTB3_plus_proper_full_live.yaml` | Add `3186` proper fake samples at launch | Does the full retained proper snapshot materially shift target-domain performance? |
| 06 | `R13_RLP1_06_FT_WTB3_plus_proper_full_truthful_gammaup.yaml` | `RLP1_05` + truthful GammaUp | Does lighting uplift help the strongest FT data packet? |
| 07 | `R13_RLP1_07_FT_WTB3_plus_proper_full_teams_shadow.yaml` | `RLP1_05` + Teams shadow sidecar | Does a light Teams nuisance overlay help the strongest FT data packet? |
| 08 | `R13_RLP1_08_SCRATCH_WTB3_plus_proper_full_live.yaml` | Scratch hedge on `RLP1_05` data | Is FT leaving performance on the table under the new honest data contract? |

## Approximate Packet Sizes

These are the current expected training sample totals for the core packet arms:

| Config | Total | Key meaning |
| --- | ---: | --- |
| `RLP1_01` | `7912` | clean honest baseline |
| `RLP1_02` | `8392` | `RLP1_01` + `480` hints |
| `RLP1_03` | `8594` | `RLP1_02` + `202` Teams hints |
| `RLP1_04` | `9278` | `RLP1_03` + `684` unenhanced proper samples at launch |
| `RLP1_05/06/07/08` | `11780` | `RLP1_03` + full launched proper snapshot |

The most important composition shift is `RLP1_05`:

- `DF40` drops from `54.67%` of `RLP1_03` to `39.88%`
- enhanced proper-data becomes `21.24%` of the packet
- unenhanced proper-data adds another `5.81%`

That is why `RLP1_05` is the correct anchor for sidecars and for the scratch
hedge.

## Freshness Rules Before Launch

### Live-discovery rule

All packet configs intentionally remove discovery cache manifests for active
mutable sources:

- `deeplive`
- `visomaster_hints`
- `teams`
- `external_training_reals`

That prevents stale discovery manifests from hiding new uploads while tonight's
packet is being prepared.

### Proper-data rebuild rule

`proper_data` is not live-bucket discovery at training runtime. It is consumed
through local generated artifacts. That means new HDTF / quickclips uploads are
not visible until the builder is rerun.

Required command immediately before launch:

```bash
python3 DeepfakeBench/training/arena/build_visomaster_proper_data_artifacts.py
```

This refreshes:

- `DeepfakeBench/training/arena/inventories/proper_visomaster_wave_2026_04_19_provisional.yaml`
- `DeepfakeBench/training/arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json`
- `DeepfakeBench/training/arena/target_domain_suites.proper_data_future.provisional_2026-04-19.yaml`
- `DeepfakeBench/training/arena/reports/proper_visomaster_wave_2026_04_19_provisional_build_report.json`

If the rebuild changes lane counts materially or drops any lane to zero, rerun a
2-step startup smoke before the full packet.

### Identity-split rule

Any rerun or follow-up comparison packet should set:

```yaml
combined_paired:
  identity_split_mode: "hash_stable"
```

The legacy global-shuffle identity split can move existing identities between
`train` / `val` / `test` when later arms add new data, which weakens strict
like-for-like packet comparisons.

## Evaluation Rule

The comparison reference remains `R12_G_FP32`, but only through the current
frozen target-domain evaluation contract.

Do not compare tonight's results to stale historical logged numbers. Rescore the
incumbent and all packet winners on the same current scorecard lane.

## Expected Readout After Packet 1

By the end of this packet we should know:

- whether hints deserve to survive once proper-data exists
- whether the full proper snapshot clearly beats the weak-signal-only packet
- whether one WT-C overlay is worth carrying forward
- whether the best packet should stay in the FT family or get a bigger scratch
  follow-up in the next round
