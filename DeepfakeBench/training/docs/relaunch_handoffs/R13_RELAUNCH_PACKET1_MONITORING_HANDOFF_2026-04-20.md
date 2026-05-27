# R13 Relaunch Packet 1 Monitoring Handoff - 2026-04-20

## Purpose

This document is the centralized handoff for the agent responsible for:

- monitoring the currently running `R13` relaunch packet-1 experiments
- documenting what happens during training in a way that is useful for the next packet
- drawing honest intermediate and final conclusions
- helping decide what the next round of experiments should be

This handoff is intentionally opinionated about process, but it should not force
the monitoring agent into our conclusions. The monitoring job is to be
critical, skeptical, and explicit about uncertainty.

## The Live Project

The live W&B project was verified on **April 20, 2026** as:

- `dtect-vision/phase2r13-rlp1-overnight-20260420`

Important operational note:

- the live W&B run display names include timestamp suffixes
- the config `name` values do **not** include those suffixes
- in the current API view, W&B tags are not reliably populated for these runs
- so the safest way to find runs is by **project path + display name prefix + run id**

## Verified Live Runs

Verified from the live project on **April 20, 2026**:

| Slot | Live run display name | Run id | Config `name` | Status at check |
| --- | --- | --- | --- | --- |
| 01 | `R13_RLP1_01_FT_WTB1_no_hints_live_0419-2241` | `40xok4cb` | `R13_RLP1_01_FT_WTB1_no_hints_live` | `running` |
| 02 | `R13_RLP1_02_FT_WTB2_hints_only_live_0419-2242` | `vj7b4z21` | `R13_RLP1_02_FT_WTB2_hints_only_live` | `running` |
| 03 | `R13_RLP1_03_FT_WTB3_hints_plus_teams_hints_live_0419-2241` | `dxrt0cjs` | `R13_RLP1_03_FT_WTB3_hints_plus_teams_hints_live` | `running` |
| 04 | `R13_RLP1_04_FT_WTB3_plus_proper_unenhanced_live_0419-2242` | `omqszujc` | `R13_RLP1_04_FT_WTB3_plus_proper_unenhanced_live` | `running` |
| 05 | `R13_RLP1_05_FT_WTB3_plus_proper_full_live_0419-2241` | `qbwyyen0` | `R13_RLP1_05_FT_WTB3_plus_proper_full_live` | `running` |
| 06 | `R13_RLP1_06_FT_WTB3_plus_proper_full_truthful_gammaup_0419-2242` | `djyg3dfb` | `R13_RLP1_06_FT_WTB3_plus_proper_full_truthful_gammaup` | `running` |
| 07 | `R13_RLP1_07_FT_WTB3_plus_proper_full_teams_shadow_0419-2242` | `pdduz69x` | `R13_RLP1_07_FT_WTB3_plus_proper_full_teams_shadow` | `running` |
| 08 | `R13_RLP1_08_SCRATCH_WTB3_plus_proper_full_live_0419-2242` | `wofj4hlp` | `R13_RLP1_08_SCRATCH_WTB3_plus_proper_full_live` | `running` |

## Background

The relaunch is not just "another 8-run overnight batch". It is the first
packet after several training-contract corrections.

### What changed before this packet

#### WT-A / WT-B: data honesty

The old questionable VisoMaster data is no longer treated as clean truth.
Instead:

- only a small retained slice survives
- that slice is explicitly labeled as `hints`
- hints are now a weak-signal axis, not the main source of target-domain truth

The current hint packet is deliberate, not accidental:

- baseline hints: `480`
- Teams-played hints: `202`
- total retained hints: `682`

#### WT-F: proper-data integration

New high-value target-domain VisoMaster data from the HDTF / quickclips buckets
was added on an explicit `proper_data` path.

This is the important distinction:

- `hints` = weak-signal residue from old questionable data
- `proper_data` = explicit new target-domain data path

Post-launch correction from `2026-04-20`: the earlier `221 / 985 / 2412`
proper-data copy is stale.

Current in-tree WT-F artifact snapshot:

- retained strict clean / Teams captures: `1826`
- total manifest videos: `7304`
- fake-lane totals:
  - `proper_visomaster_clean`: `342`
  - `proper_visomaster_teams`: `342`
  - `proper_visomaster_enhanced_clean`: `1484`
  - `proper_visomaster_enhanced_teams`: `1484`

Launched RLP1 packet comparison scale:

- `RLP1_04` live proper fake rows: `684`
- `RLP1_05/06/07/08` live proper fake rows: `3186`

Use the builder report plus the launched run's startup W&B summary as the
count source of truth for any rerun, retry, or packet-2 planning.

#### WT-C: augmentation honesty

WT-C is not the main data story. It exists so augmentation sidecars are:

- truthful
- explicit
- independently testable

That matters here because the current packet includes two WT-C-style overlays:

- truthful GammaUp
- light Teams shadow passthrough augmentation

#### WT-D / WT-E: evaluation discipline

The relaunch is supposed to move model selection toward the target-domain
contract, not just generic holdout metrics.

This means:

- in-training metrics are important
- but they are not the final promotion decision
- the final winner should still be judged on the frozen target-domain scorecard

## What This Packet Is Trying To Learn

This packet is designed to answer the questions in the right order.

### Primary question: data composition

The biggest question is:

- what data packet best serves the target-domain objective?

That is why 5 of the 8 runs are a data ladder:

1. clean honest baseline
2. add baseline hints
3. add Teams-played hints
4. add unenhanced proper-data
5. add full proper-data snapshot

### Secondary question: sidecar nuisance robustness

Once the strongest expected data packet exists, the next question is:

- does one narrow augmentation overlay help that packet?

That is why runs 06 and 07 are:

- `RLP1_05` + truthful GammaUp
- `RLP1_05` + Teams shadow sidecar

### Tertiary question: FT versus scratch

We did not want an FT-only packet because a pure fine-tune family can inherit
old assumptions from the initializer.

We also did not want an all-scratch packet because that would burn too much of
the 8-slot budget on slower exploration.

So the compromise is:

- most runs are FT
- one run is a scratch hedge on the strongest expected data packet

## The Shared Scaffold

These features are intentionally held fixed for the fine-tune packet unless a
run explicitly says otherwise:

- backbone: `vit_b_16_laion_datacomp` / `ViT-B-16-DataComp-XL`
- rank: `736` (`k=32`)
- frames per video: `8`
- frames per batch: `32`
- resolution: `224`
- sampling strategy: `identity_resample_weighted`
- seed: `737`
- train / val / test split: `0.85 / 0.10 / 0.05`
- fine-tune init: Track C `R12_G_FP32` scorecard-base checkpoint
- FT learning rate: `3e-5`
- FT total steps: `10000`
- FT warmup: `400`
- FT ArcFace schedule: `s 6 -> 12`
- in-training OOD monitoring starts at step `5000`

The scratch hedge differs on purpose:

- no base checkpoint
- learning rate: `2e-4`
- total steps: `30000`
- warmup: `1500`
- ArcFace schedule: `s 10 -> 14`

## What Each Run Means

| Slot | Meaning | Main comparison |
| --- | --- | --- |
| `RLP1_01` | clean honest baseline | reference point for all hint value |
| `RLP1_02` | add baseline hints | `01` vs `02` = do hints help at all? |
| `RLP1_03` | add Teams-played hints | `02` vs `03` = does the extra retained hint slice help? |
| `RLP1_04` | add unenhanced proper-data | `03` vs `04` = does explicit proper-data already beat hints-only WTB3? |
| `RLP1_05` | add full proper-data snapshot | `04` vs `05` = does enhanced proper-data add further value? |
| `RLP1_06` | `RLP1_05` + truthful GammaUp | `05` vs `06` = does lighting uplift help the strongest FT packet? |
| `RLP1_07` | `RLP1_05` + Teams shadow sidecar | `05` vs `07` = does a light Teams nuisance overlay help? |
| `RLP1_08` | scratch hedge on `RLP1_05` data | `05` vs `08` = is FT leaving performance on the table? |

## Monitoring Mindset

The monitoring agent should be honest, not loyal to the packet design.

### Required mindset

- Do not defend the plan.
- Do not assume proper-data must win.
- Do not assume hints are obsolete.
- Do not assume augmentation sidecars should help.
- Do not assume scratch is better just because it is more expensive.

Instead:

- state what is actually happening
- separate observed facts from interpretations
- say when the evidence is too early or too weak
- say plainly if one of the hypotheses appears wrong

## What To Watch During Training

### 1. Startup and loader sanity

Check early logs and config surfaces for:

- correct run identity
- FT versus scratch schedule being what it should be
- source counts looking plausible
- proper-data lanes present where expected
- hint lanes present where expected
- no obvious loader collapse, missing lane, or repeated download failure

If any run appears to have silently lost a source family, that is a serious
finding and should be called out immediately.

### 2. Mid-training signals

Monitor:

- holdout metrics
- OOD metrics after the first OOD event
- OOD-composite style checkpoint behavior
- whether one run is clearly unstable or obviously dominated
- whether a run is merely slow versus actually bad

Be careful with timing:

- FT runs should not be judged seriously before the first meaningful OOD phase
- the FT packet only begins OOD monitoring at step `5000`
- the scratch run should be judged even more patiently

Important historical lesson:

- `R12_G` was a late-improving scratch winner
- so early scratch weakness is not enough to dismiss `RLP1_08`

### 3. Like-for-like comparisons

Do not compare everything to everything at once.

Use the intended comparison chain:

- `01` vs `02`
- `02` vs `03`
- `03` vs `04`
- `04` vs `05`
- `05` vs `06`
- `05` vs `07`
- `05` vs `08`

That preserves the logic of the packet.

### 4. Final training-side takeaways

At the end of the runs, the monitoring agent should be able to say:

- whether hints still deserve space once proper-data exists
- whether full proper-data is better than the unenhanced slice
- whether either WT-C sidecar deserves follow-up
- whether the scratch hedge is interesting enough to expand in the next round

## What Not To Overclaim

The monitoring agent must not confuse training-side evidence with final
deployment proof.

Do **not** overclaim:

- that the best in-training run is automatically the final model
- that holdout AUC alone decides the winner
- that mixed in-training OOD monitoring is the full target-domain contract
- that historical logged numbers from older rounds are directly comparable

The right stance is:

- training-side monitoring is directional and decision-supporting
- final promotion still needs the shared frozen target-domain evaluation

## Documentation Requirements

The monitoring agent should produce a living record that is useful for the next
packet, not just a one-paragraph "winner".

For each run, document:

- run display name
- run id
- hypothesis being tested
- important schedule facts
- startup sanity notes
- important mid-training observations
- best checkpoint behavior
- final training-side verdict
- what the run teaches us for the next round

For the packet as a whole, document:

- what was learned
- what was surprising
- what failed
- what looks promising but is not yet proven
- what should be repeated, expanded, or dropped

When writing conclusions, separate:

- **observed fact**
- **inference**
- **next-step recommendation**

## Critical Guidance

Please be explicitly critical and honest when reviewing.

That means:

- say when a run is not answering the question it was supposed to answer
- say when a run loses to a simpler baseline
- say when a sidecar adds complexity without evidence
- say when a result might just be variance or timing
- say when the packet design itself seems wrong in hindsight

The job is not to make the packet look good.
The job is to make the next decision better.

## Operational Freshness Note

For the currently running packet, the launch has already happened.

But for any rerun, retry, or follow-up packet:

- mutable discovery sources were intentionally made cacheless
- `proper_data` is still artifact-backed, not live-discovered at runtime
- so new HDTF / quickclips uploads require a fresh rebuild before launch:

```bash
python3 DeepfakeBench/training/arena/build_visomaster_proper_data_artifacts.py
```

This matters for future packets even if it does not change the current live
runs.

Future split-hygiene rule:

- any rerun or follow-up comparison packet should set
  `combined_paired.identity_split_mode: "hash_stable"`
- the legacy global-shuffle identity split can move identities between
  `train` / `val` / `test` when later arms add new data, so packet-1 arm
  comparisons are directionally useful but not a perfectly frozen shared
  holdout slice

Historical reporting note:

- older W&B summaries that show `unknown_fake` alongside matching
  `external_real` counts are slightly polluted by a reporting bug in unpaired
  external-real family accounting

## Recommended Context Files

If deeper context is needed, read these docs in roughly this order:

1. `DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_EXPERIMENT_PLAN_2026-04-19.md`
2. `DeepfakeBench/training/docs/relaunch_handoffs/RELAUNCH_UPGRADE_REVIEW_PACKET_2026-04-19.md`
3. `DeepfakeBench/training/docs/relaunch_handoffs/NEW_DATA_LOADER_AND_EXPERIMENT_HANDOFF_2026-04-19.md`
4. `DeepfakeBench/training/docs/relaunch_handoffs/WT_B_AND_NEW_DATA_READINESS_2026-04-19.md`
5. `DeepfakeBench/training/docs/relaunch_handoffs/WT-C_2026-04-17.md`
6. `DeepfakeBench/training/docs/relaunch_handoffs/WT-F_2026-04-17.md`
7. `DeepfakeBench/training/docs/TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md`

## Suggested Prompt To The Monitoring Agent

```text
You are monitoring the live R13 relaunch packet-1 runs in:
`dtect-vision/phase2r13-rlp1-overnight-20260420`

Start with:
`DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_MONITORING_HANDOFF_2026-04-20.md`

Your job:
- monitor the 8 live runs carefully
- document what happens during training in a way that helps the next packet
- draw honest conclusions
- be critical and explicit about uncertainty

Important:
- do not assume the packet is correct
- do not force a positive story
- do not overclaim from early metrics
- do not confuse training-side metrics with final promotion proof

The exact live run display names and run ids are listed in the handoff doc.
Use those, not only config names or tags.

When you report findings, separate:
- observed fact
- inference
- recommendation

Deliverables:
1. a live-running summary while training is ongoing
2. a post-run packet synthesis after the 8 runs are done
3. a next-round recommendation with clear rationale
```
