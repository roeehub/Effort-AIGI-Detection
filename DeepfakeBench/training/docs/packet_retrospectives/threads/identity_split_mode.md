# Thread: Identity-split-mode methodology (`shuffle` → `hash_stable`)

## The question

When the training-side identity-split mode flipped from the legacy global `shuffle` (`combined_paired.split_samples_by_identity`) to `hash_stable` between [RLP1](../packets/RLP1.md) and [RLP2](../packets/RLP2.md), how much of any later cross-packet AUC delta is attributable to that split-mode change versus the actual training-recipe change being tested? Concretely: if a fresh control (`RLP2_01 = 0.98662`) drops `−0.00253` against the prior packet's control (`RLP1_01 = 0.98915`), is that mutable-source drift, split-mode artifact, or both — and how does any later packet make sure it is comparing what it thinks it is comparing?

## Initial belief

Until the live RLP1 startup summaries surfaced on 2026-04-20, the team treated the legacy global-shuffle identity split as an inert default — the `.85/.10/.05` partition was assumed to behave like a deterministic seed-defined partition across runs and packets (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_EXPERIMENT_PLAN_2026-04-19.md:23-31`). That belief made `01 → 02 → 03 → 04 → 05` and `05 → 06`, `05 → 07`, `05 → 08` directly readable as a frozen A/B ladder.

## What changed our mind

- **2026-04-20 RLP1 mid-flight diagnosis (`docs/relaunch_handoffs/RELAUNCH_UPGRADE_REVIEW_PACKET_2026-04-19.md:406-416`).** Live introspection of `combined_paired.split_samples_by_identity` showed that *"when later arms add identities, earlier identities can move between `train` / `val` / `test` even with the same `split_seed`"*. RLP1's later arms (`04`, `05/06/07/08`) carry the proper-data identities that the early arms (`01`, `02`, `03`) do not see — so the cross-arm comparison is no longer perfectly identity-frozen. The patch was landed in-tree in the same window: `combined_paired.identity_split_mode: "hash_stable"` (`RELAUNCH_UPGRADE_REVIEW_PACKET_2026-04-19.md:412-415`). The future rule was *"any comparison packet or rerun that cares about arm-to-arm fairness should use `hash_stable`"*.
- **2026-04-20 RLP1 live monitoring acknowledges the caveat (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_LIVE_MONITORING_2026-04-20.md:367-377`).** *"the current live packet used the legacy global-shuffle identity split … `01 -> 05` still teaches something real, but it is not a perfectly frozen shared holdout slice in the strictest sense"*. This is the explicit downgrade from "frozen A/B ladder" to "directionally useful, but not a perfectly frozen A/B ladder" — captured in [RLP1's status card retrospective](../packets/RLP1.md) under "Directional, not frozen."
- **2026-04-21 RLP2 flips to `hash_stable` (`experiments/phase2_round13/R13_RLP2_06_FT_WTB1_plus_proper_unenhanced_low_arcface_live.yaml:79`, [RLP2.md:23](../packets/RLP2.md)).** Every `R13_RLP2_*` config uses `combined_paired.identity_split_mode: "hash_stable"` from launch — confirmed by direct yaml inspection. The split mode is never the variable being tested; it changes once at the RLP1→RLP2 boundary and then stays.
- **2026-04-21 cross-packet split-mode caveat pinned at A5 of the packet-1 results handoff (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_RESULTS_AND_PLANNING_CONTEXT_2026-04-21.md:3-10`).** *"Packet 1 used `identity_split_mode: shuffle` (legacy default). Packet 2 and packet 3 use `identity_split_mode: hash_stable`. The switch re-partitions identities across train / val / test, so any AUC delta between a packet-1 run and a packet-2/3 run carries an **unquantified split-mode component** on top of whatever training-recipe change you are actually testing."* The packet-2 fresh-control drop (`RLP2_01 = 0.98662` vs `RLP1_01 = 0.98915`, Δ `−0.00253`) is explicitly named as "has a split-mode component and cannot be attributed to mutable-source drift alone". This is the load-bearing methodology rule the next planner has to honor. The handoff also records the within-packet correction: *"Packet-3 `RLP3_01` is the correct within-packet baseline for packet-3 interpretation"*, foreclosing the temptation to compare any RLP3 number directly to RLP1.
- **2026-04-21 source-of-truth handoff codifies it (`docs/relaunch_handoffs/R13_RELAUNCH_TRAINING_SOURCE_OF_TRUTH_2026-04-21.md:113-114`).** Settled bullet: *"the new HDTF / quickclips buckets belong on `proper_data`, not on `hints`"* — and the operational read through hash_stable is what makes that comparable across packets.

## Current stance (2026-04-21)

The `hash_stable` identity-split mode is the operational baseline from RLP2 onward. RLP1 is the only packet on legacy `shuffle`, and any cross-packet AUC delta that crosses the RLP1↔RLP2+ boundary carries an unquantified split-mode component. Within-packet baselines (`RLP2_01` for RLP2; `RLP3_01` for RLP3; etc.) are the correct comparison anchors — the "fresh control on the same launch-time snapshot" pattern (`R13_RELAUNCH_PACKET2_EXPERIMENT_PLAN_2026-04-21.md:77-83`) was added precisely so packet-2+ would have a same-snapshot, same-split-mode reference. Treat any "RLP2 control dropped vs RLP1 control" framing as a category error unless the split-mode component is bounded first (no agent has done so yet through Slice 2).

## Packet timeline

- [WT-infrastructure](../packets/WT_infrastructure.md) — WT-F's April-19 follow-up shipped the `hash_stable` mode in-tree (`RELAUNCH_UPGRADE_REVIEW_PACKET_2026-04-19.md:406-416`); not yet active.
- [RLP1](../packets/RLP1.md) — used legacy `shuffle` (default at launch); split-mode caveat surfaced mid-flight on 2026-04-20.
- [RLP2](../packets/RLP2.md) — first packet on `hash_stable`; introduces the "fresh same-snapshot control" pattern as the cross-packet drift compensator.
- RLP3+ — inherit `hash_stable`; `RLP3_01` becomes the within-packet baseline (Slice-3 ownership).

## Evidence locations

- `docs/relaunch_handoffs/RELAUNCH_UPGRADE_REVIEW_PACKET_2026-04-19.md:406-416` — the diagnostic write-up of the global-shuffle behavior; in-tree fix landed
- `docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_LIVE_MONITORING_2026-04-20.md:367-377` — RLP1 mid-flight downgrade from "frozen A/B" to "directionally useful"
- `docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_RESULTS_AND_PLANNING_CONTEXT_2026-04-21.md:3-10` — the pinned split-mode caveat (A5 of packet-2 planning); the load-bearing rule
- `docs/relaunch_handoffs/R13_RELAUNCH_PACKET2_EXPERIMENT_PLAN_2026-04-21.md:77-83` — fresh same-snapshot control pattern
- `experiments/phase2_round13/R13_RLP2_06_FT_WTB1_plus_proper_unenhanced_low_arcface_live.yaml:79` — `identity_split_mode: "hash_stable"` is on every RLP2_* config
- [RLP1.md](../packets/RLP1.md) "Directional, not frozen" bullet
- [RLP2.md](../packets/RLP2.md) "Configuration" — `Split mode flipped shuffle → hash_stable`
- Memory: *(none load-bearing — this is a methodology rule encoded in handoffs and packet retros; no auto-memory entry has been written for it through Slice 2)*

## Open loops

### Open loop: split-mode-delta-unquantified
status: open
severity: medium
first_seen: 2026-04-21
last_verified: 2026-04-29
close_criterion: a within-packet ablation produces a numeric bound on the AUC delta attributable to `shuffle → hash_stable` alone (e.g., re-run `RLP1_01` recipe with `hash_stable` on the same data snapshot, or re-run `RLP2_01` recipe with `shuffle` and measure the difference under matched mutable-source state)

The `−0.00253` delta between `RLP2_01` (0.98662, `hash_stable`) and `RLP1_01` (0.98915, `shuffle`) is treated as a noisy upper bound that mixes mutable-source drift and split-mode artifact, and through Slice 2 no agent has separated those components. The methodology rule is conservative enough that no RLP1↔RLP2+ comparison is being made, so the loop is not actively damaging — but it is not closed either, and the same kind of caveat will recur whenever another methodology shift moves through the packet stream. Next slice agents should watch for any later packet that ablates the same recipe under matched mutable-source state, since that would close this loop incidentally.

**Slice 3 verification (2026-04-29).** RLP3's A2b retroactive runner (`tools/retroactive_final_eval.py`) reapplied the A10 `blake2b(video_id)` hash-partition to RLP1/RLP2 checkpoints (`RLP1_01, RLP1_04, RLP2_01, RLP2_02, RLP2_06`) — but this re-evaluates the same checkpoints under matched eval pools, **it does not re-train under matched split-mode**. The split-mode delta is a training-time partition decision; A2b operates only at eval time. So A2b is not a close-criterion event for this loop. The RLP3 retro-score results doc (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_RETRO_SCORE_RESULTS_2026-04-22.md`) does not address split-mode separately. RLP3's design rule (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_EXPERIMENT_PLAN_2026-04-21.md:341`) handles the question conservatively by making `RLP3_01` the within-packet baseline, so no RLP3 number is compared to an RLP1 number across the split-mode boundary — but that's *avoidance*, not a quantified bound. Loop stays `open`. The condition for closure (re-run `RLP1_01` recipe with `hash_stable` on the same snapshot, or re-run `RLP2_01` recipe with `shuffle`) was not constructed in any RLP3/RLP3.5/RLP4 yaml.
