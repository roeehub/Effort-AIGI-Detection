# R13 Packet-2 Planning Handoff From Packet-1 Results - 2026-04-21

## Split-mode caveat (pinned 2026-04-21, A5)

**Packet 1 used `identity_split_mode: shuffle` (legacy default). Packet 2 and packet 3 use `identity_split_mode: hash_stable`.** The switch re-partitions identities across train / val / test, so any AUC delta between a packet-1 run and a packet-2/3 run carries an **unquantified split-mode component** on top of whatever training-recipe change you are actually testing.

- Packet-3 `RLP3_01` is the correct within-packet baseline for packet-3 interpretation.
- The packet-2 fresh-control drop (`RLP2_01 = 0.98662` vs `RLP1_01 = 0.98915`, Δ −0.00253) has a split-mode component and cannot be attributed to mutable-source drift alone.
- When comparing packet-1 numbers to packet-2+ numbers, do **not** treat the delta as a training-regime effect without first bounding the split-mode artifact.

## Purpose

This document is for the agent whose job is to design the **next 6
experiments** after the first relaunch 8-run packet.

This is not just a summary of packet 1. The point is to help the next planner:

- understand the relaunch background
- understand what packet 1 actually tested
- understand what packet 1 actually taught us
- identify what is settled, what is still uncertain, and what must be measured
- propose the next 6 runs with honest reasoning

The next planner should not anchor blindly on any prior summary, including this
one. They should read the linked materials, verify the live project directly,
and then produce a deliberate packet-2 plan.

Live snapshot referenced below:

- W&B project: `dtect-vision/phase2r13-rlp1-overnight-20260420`
- snapshot time: `2026-04-21 00:23:50 CEST`

## What The Next Planner Must Deliver

The output of the next planning step should be a concrete 6-run experiment
proposal, not a vague brainstorming note.

The planner should hand back:

1. a proposed 6-run slate
2. the purpose of each run
3. what stays fixed across the slate
4. what question each run is answering
5. what metric or decision rule will be used to interpret the slate
6. how the next slate will measure **prediction robustness stability**
7. what assumptions remain provisional

The next planner should be explicit about tradeoffs and uncertainty. They should
not force a false conclusion from packet 1.

## Read First

Read these before writing the next-6 plan:

1. [Relaunch Upgrade Review Packet](./RELAUNCH_UPGRADE_REVIEW_PACKET_2026-04-19.md)
2. [Teams Target-Domain Upgrade Plan](../TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md)
3. [New Data Loader And Experiment Handoff](./NEW_DATA_LOADER_AND_EXPERIMENT_HANDOFF_2026-04-19.md)
4. [WT-F Handoff](./WT-F_2026-04-17.md)
5. [WT-C Handoff](./WT-C_2026-04-17.md)
6. [R13 Packet-1 Experiment Plan](./R13_RELAUNCH_PACKET1_EXPERIMENT_PLAN_2026-04-19.md)
7. [R13 Packet-1 Monitoring Handoff](./R13_RELAUNCH_PACKET1_MONITORING_HANDOFF_2026-04-20.md)
8. [R13 Packet-1 Live Monitoring Record](./R13_RELAUNCH_PACKET1_LIVE_MONITORING_2026-04-20.md)

Then verify the live W&B project directly before finalizing any plan.

## Background

### Why the relaunch exists

The relaunch is not a routine tuning cycle. It exists because the project
needed to correct several training-contract issues and move closer to the real
deployment objective.

The core background is:

- the April 6 plan changed the project goal from generic holdout strength to
  Teams / target-domain usefulness
- WT-A and WT-B made the old questionable VisoMaster story honest by carving a
  small weak-signal residue into explicit `hints`
- WT-F introduced the new HDTF / quickclips VisoMaster data on an explicit
  `proper_data` path
- WT-C separated nuisance / augmentation ideas into truthful sidecars instead of
  letting them remain implied or partially dead config intent

The clean mental model now is:

- `hints` = weak-signal residue from old questionable data
- `proper_data` = explicit new high-value target-domain VisoMaster path
- packet 1 = first data-dominant relaunch packet to test those axes honestly

### What packet 1 was supposed to answer

Packet 1 was intentionally built to answer these questions in order:

1. what is the best honest data packet under the relaunch contract?
2. once proper-data exists, do hints still help?
3. does a light WT-C-style sidecar help the strongest proper-data packet?
4. is FT enough, or does scratch still matter?

Packet 1 was therefore designed as:

- 5 data-composition runs
- 2 sidecars
- 1 scratch hedge

That packet structure should matter when interpreting results.

## Shared Scaffold From Packet 1

Unless the next planner makes an explicit case to change something, these are
the default inherited packet-1 scaffold choices:

- backbone: `ViT-B-16-DataComp-XL` / `vit_b_16_laion_datacomp`
- low-rank setup: rank `736`, `k=32`
- frames per video: `8`
- frames per batch: `32`
- image size: `224`
- sampling strategy: `identity_resample_weighted`
- seed: `737`
- train / val / test split ratio: `0.85 / 0.10 / 0.05`

Fine-tune family (`01` through `07`):

- base checkpoint: `R12_G_FP32` scorecard-base initialization
- LR: `3e-5`
- total steps: `10000`
- warmup: `400`
- ArcFace schedule: `s 6 -> 12`

Scratch hedge (`08`):

- no base checkpoint
- LR: `2e-4`
- total steps: `30000`
- warmup: `1500`
- ArcFace schedule: `s 10 -> 14`

The next planner should treat these as defaults, not as sacred rules.

## Data Reality That Must Inform Packet 2

### Hints

Hints are not “the main VisoMaster training data.” They are an intentionally
small retained weak-signal slice from the old questionable pool.

Packet-1 hint sizes:

- baseline hints: `480`
- Teams-played hints: `202`
- total retained hints in the full hint packet: `682`

Planning implication:

- the next planner should not talk about hints as if they represent the true new
  target-domain direction
- if hints remain in the search space, there should be a concrete reason

### Proper-data

Proper-data is the explicit new target-domain path from the HDTF / quickclips
buckets. This is the important new data story.

Use the corrected live packet-1 scale, not the earliest draft counts:

- `04` live proper fake rows: `684`
- `05/06/07/08` live proper fake rows: `3186`

Planning implication:

- the `04 -> 05` jump is large
- packet 1 did not merely test “some more proper-data”; it tested a much heavier
  proper-data injection at `05`

### Legacy non-hint VisoMaster

For the relaunch packet, the important non-hint VisoMaster story is the new
`proper_data` path. The planner should not reason as if packet 1 was still
using legacy VisoMaster as clean supervision.

## Packet-1 Runs

| Slot | Config | What it was testing |
| --- | --- | --- |
| `01` | [`R13_RLP1_01_FT_WTB1_no_hints_live`](../../experiments/phase2_round13/R13_RLP1_01_FT_WTB1_no_hints_live.yaml) | honest FT baseline without hints or proper-data |
| `02` | [`R13_RLP1_02_FT_WTB2_hints_only_live`](../../experiments/phase2_round13/R13_RLP1_02_FT_WTB2_hints_only_live.yaml) | add baseline hints |
| `03` | [`R13_RLP1_03_FT_WTB3_hints_plus_teams_hints_live`](../../experiments/phase2_round13/R13_RLP1_03_FT_WTB3_hints_plus_teams_hints_live.yaml) | add Teams-played hints |
| `04` | [`R13_RLP1_04_FT_WTB3_plus_proper_unenhanced_live`](../../experiments/phase2_round13/R13_RLP1_04_FT_WTB3_plus_proper_unenhanced_live.yaml) | add unenhanced proper-data |
| `05` | [`R13_RLP1_05_FT_WTB3_plus_proper_full_live`](../../experiments/phase2_round13/R13_RLP1_05_FT_WTB3_plus_proper_full_live.yaml) | add full retained proper-data snapshot |
| `06` | [`R13_RLP1_06_FT_WTB3_plus_proper_full_truthful_gammaup`](../../experiments/phase2_round13/R13_RLP1_06_FT_WTB3_plus_proper_full_truthful_gammaup.yaml) | truthful GammaUp sidecar on `05` |
| `07` | [`R13_RLP1_07_FT_WTB3_plus_proper_full_teams_shadow`](../../experiments/phase2_round13/R13_RLP1_07_FT_WTB3_plus_proper_full_teams_shadow.yaml) | Teams-shadow sidecar on `05` |
| `08` | [`R13_RLP1_08_SCRATCH_WTB3_plus_proper_full_live`](../../experiments/phase2_round13/R13_RLP1_08_SCRATCH_WTB3_plus_proper_full_live.yaml) | scratch hedge on the `05` packet |

## Latest Live Snapshot

At the snapshot used for this handoff:

- finished: `01`, `02`, `04`, `05`, `06`, `07`
- still running: `03`, `08`
- `03` is live at `11123` steps, already past nominal FT budget
- `08` is live at `11000 / 30000` steps

### Best-checkpoint ranking

| Rank | Slot | State | Best composite | Best step | Latest composite | Latest holdout AUC | Latest OOD AUC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `01` | `finished` | `0.98915` | `6000` | `0.98857` | `0.99540` | `0.98183` |
| 2 | `04` | `finished` | `0.98773` | `8000` | `0.98768` | `0.98840` | `0.98705` |
| 3 | `05` | `finished` | `0.98542` | `8000` | `0.98507` | `0.98196` | `0.98816` |
| 4 | `02` | `finished` | `0.98524` | `9000` | `0.98524` | `0.98748` | `0.98298` |
| 5 | `07` | `finished` | `0.98500` | `7000` | `0.98470` | `0.98167` | `0.98775` |
| 6 | `06` | `finished` | `0.98492` | `10000` | `0.98492` | `0.98187` | `0.98795` |
| 7 | `03` | `running` | `0.98442` | `7000` | `0.98403` | `0.98465` | `0.98341` |
| 8 | `08` | `running` | `0.98164` | `8000` | `0.97399` | `0.97476` | `0.98115` |

## What Packet 1 Taught Us

These are the main lessons the next planner should reason from.

### 1. `01` is still the burden-of-proof control

`01` remains the packet leader on the active balanced metric.

Planning implication:

- any packet-2 idea should be framed relative to whether it is trying to beat
  `01`, complement `01`, or test a new deployment-weighted criterion that
  `01` may not optimize

### 2. `04` is the strongest new signal

`04` is the best proper-data arm and the strongest new idea from packet 1.

Compared with `03`, `04` improved:

- composite by `+0.00331`
- holdout by `+0.00353`
- OOD by `+0.00308`

Planning implication:

- packet 2 should treat the positive `04` signal as real
- the next planner should think carefully about what exactly `04` is buying
  relative to the hint packet underneath it

### 3. The hint ladder is weak

`02` and `03` both trail `01`, and `03` does not rescue the hint story.

Planning implication:

- hints should no longer be treated as a default ingredient
- if the next planner keeps hints in the search space, that decision needs a
  precise hypothesis

### 4. `05` suggests the full proper-data jump is too heavy at current scale

`05` is better than `03`, but clearly worse than `04`.

Planning implication:

- packet 2 should not lazily equate “more proper-data” with “better”
- the next planner should reason about dose, composition, and interaction, not
  just binary include / exclude

### 5. The WT-C sidecars are not currently justified

Neither `06` nor `07` beats `05`.

Planning implication:

- sidecars did not earn promotion from packet 1
- but this was tested on top of the heavy `05` packet, so the result is more
  negative than decisive

### 6. Scratch is not currently winning

`08` has strong OOD upside, but weak balanced performance and clear trajectory
volatility.

Planning implication:

- packet 2 should not assume scratch deserves multiple slots by default
- if scratch remains in the search space, the planner should articulate why

### 7. Packet 1 is optimization-stable, but not yet robust in the stronger sense

Current packet-1 support:

- no collapse signal
- gradient flow is intact
- FT runs are fairly stable after their best checkpoints

Current packet-1 limits:

- no seed-stability proof
- no direct prediction-robustness measurement
- no direct perturbation-consistency training signal in packet 1

Planning implication:

- packet 2 should explicitly define how it will measure prediction robustness
  stability

## What The Next Planner Should Treat As Settled

These are not perfectly final truths, but packet 1 makes them strong enough to
act on unless a later frozen scorecard reverses them:

- the honest no-hints FT baseline is strong
- unenhanced proper-data is the most promising new direction from packet 1
- the hint ladder is weak
- the heavy full-proper packet did not justify itself
- the two sidecars did not justify themselves on top of `05`

## What Is Still Uncertain

These are the real packet-2 planning uncertainties:

- whether the value in `04` comes from unenhanced proper-data itself, from the
  specific packet composition, or from both
- whether a smaller / cleaner proper-data extension than `05` would beat `04`
- whether hints still have any role once proper-data is used more cleanly
- whether WT-C-style sidecars might help on a different base packet
- whether scratch should remain only a hedge or leave the search space entirely
- how to define and measure prediction-robustness stability

## Non-Negotiable Planning Questions

Before proposing the next 6 runs, the next planner should answer these
questions explicitly:

1. What is the exact role of `01` in packet 2: control, fallback, or both?
2. What is the cleanest hypothesis for why `04` worked?
3. Is the `05` underperformance mainly a scale problem, a composition problem,
   or an interaction problem?
4. What concrete evidence still justifies giving any slot to hints?
5. Should packet 2 spend any slot on WT-C sidecars before the data story is
   tighter?
6. Should packet 2 spend any slot on scratch before the FT data story is tighter?
7. What metric should define success in packet 2: balanced composite only, or a
   more deployment-weighted criterion?
8. How will packet 2 measure **prediction robustness stability** in a way that
   is concrete enough to compare runs?

## Required Packet-2 Planning Dimensions

The next planner should explicitly reason across these axes:

- data composition
- hints usage or removal
- proper-data amount and composition
- FT vs scratch allocation
- augmentation / sidecar allocation
- stability measurement

The planner should not let the packet become a vague mixture of all axes at
once. Six slots are scarce.

## What The Next Planner Should Hand Back

The next planner’s response should include:

1. a 6-run slate
2. a one-line purpose for each run
3. what is fixed across the 6 runs
4. what varies across the 6 runs
5. why this is the right use of only 6 slots
6. how the slate measures prediction robustness stability
7. what conclusions from packet 1 are being exploited, and what uncertainties are
   still being tested

The planner should be critical and honest. They should not optimize for novelty
for its own sake, and they should not preserve packet-1 ingredients that no
longer have a good reason to survive.

