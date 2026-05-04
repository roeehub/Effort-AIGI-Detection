# Packet RLP1  ·  Relaunch baseline — 8-run probe of hints, proper-data, FT-vs-scratch

> **Template contract**: fill every section. If a section truly has no content, write `*(none — reason)*` rather than deleting the heading. Keep the status-card table intact.

## Status card

| Field | Value |
|---|---|
| Dates | 2026-04-19 → 2026-04-21 |
| Slots | 8 (RLP1_01 … RLP1_08) |
| Headline lever | Three-axis probe: hints ladder, proper-data integration, FT-vs-scratch |
| Leader slot | RLP1_01 (no-hints FT control) |
| Leader metric | best_ood_composite = 0.98915 @ step 6000 (holdout 0.99628, OOD 0.98213) |
| Verdict | confirmed — control wins; unenhanced proper-data is the only promising new signal; hints fail; scratch not competitive |
| Next-packet decision | Packet-2: drop hints, make unenhanced proper-data the main bet, defer sidecars, switch to `hash_stable` split |
| Themes touched | [gate_alignment_story](../threads/gate_alignment_story.md) (light); [promotion_contract_evolution](../threads/promotion_contract_evolution.md) (baseline, pre-`value_composite`); [preprocessing_parity_bug](../threads/preprocessing_parity_bug.md) (pre-fix) |

## Configuration

- **First packet under the April-17 relaunch contract.** WT-A/B reclassified the old questionable VisoMaster pool as weak-signal `hints` (480 baseline + 202 Teams = 682 retained); WT-F brought explicit `proper_data` on the HDTF/quickclips path. See `docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_EXPERIMENT_PLAN_2026-04-19.md:66-105`.
- **Control slot**: `RLP1_01` — clean FT baseline, no hints, no proper-data, 7912 rows.
- **Shared FT scaffold** (01–07): ViT-B-16-DataComp-XL, rank 736 (k=32), `R12_G_FP32` base, lr 3e-5, 10000 steps, warmup 400, ArcFace s 6→12, seed 737, 0.85/0.10/0.05 split. Scratch (08): no base ckpt, lr 2e-4, 30000 steps, ArcFace s 10→14. See `R13_RELAUNCH_PACKET1_EXPERIMENT_PLAN_2026-04-19.md:19-49`.
- **Identity split mode**: legacy `shuffle` (global) — later packets switched to `hash_stable`.
- **Variants** (all yamls live in `experiments/phase2_round13/R13_RLP1_{01..08}_*.yaml`):
  - `01` — honest no-hints FT baseline (control)
  - `02` — `01` + 480 baseline hints
  - `03` — `02` + 202 Teams-played hints (WTB3)
  - `04` — WTB3 + **684** unenhanced proper fakes (342 clean + 342 teams)
  - `05` — WTB3 + **3186** full proper snapshot (adds enhanced lanes)
  - `06` — `05` + truthful GammaUp sidecar
  - `07` — `05` + Teams-shadow passthrough sidecar
  - `08` — scratch on the `05` data packet
- **W&B project**: `dtect-vision/phase2r13-rlp1-overnight-20260420`.

## Results at the time

Ranking is on `best_ood_composite` (AUC-based; pre-`value_composite`). See `R13_RELAUNCH_PACKET1_RESULTS_AND_PLANNING_CONTEXT_2026-04-21.md:202-214`.

| Rank | Slot | Best step | Best composite | Best holdout | Best OOD | Δ vs `01` |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `01` | 6000 | **0.98915** | 0.99628 | 0.98213 | — |
| 2 | `04` | 8000 | 0.98773 | 0.98868 | 0.98677 | −0.00143 |
| 3 | `05` | 8000 | 0.98542 | 0.98240 | 0.98845 | −0.00374 |
| 4 | `02` | 9000 | 0.98524 | 0.98751 | 0.98298 | −0.00391 |
| 5 | `07` | 7000 | 0.98500 | 0.98179 | 0.98824 | −0.00415 |
| 6 | `06` | 10000 | 0.98492 | 0.98191 | 0.98795 | −0.00423 |
| 7 | `03` | 7000 | 0.98442 | 0.98515 | 0.98369 | −0.00474 |
| 8 | `08` | 8000 | 0.98164 | 0.97182 | 0.99167 | −0.00751 |

Key reads:
- **Control wins.** `RLP1_01` leads by 0.00143; nothing in the 7 add-on arms displaces it.
- **Unenhanced proper-data is the best new signal.** `RLP1_04` beats `RLP1_03` by +0.00331 composite (+0.00353 holdout, +0.00308 OOD). See `R13_RELAUNCH_PACKET1_LIVE_MONITORING_2026-04-20.md:41-54`.
- **Hints fail.** Both `02` and `03` trail `01`; Teams-played hints on top of baseline hints made things worse, not better.
- **Full proper is too heavy at current scale.** `05` is worse than `04` despite 5× more proper-fake rows; the enhanced lanes look like dilution or imbalance.
- **Sidecars unjustified.** Neither GammaUp (`06`) nor Teams-shadow (`07`) beats base `05`. See `R13_RELAUNCH_PACKET1_RESULTS_AND_PLANNING_CONTEXT_2026-04-21.md:266-274`.
- **Scratch has OOD spike, poor balance.** `RLP1_08` recorded the packet's top OOD AUC (0.99167) but worst holdout (0.97182); latest composite fell further to 0.97399.
- **Live packet larger than planned.** RLP1_04 ran 684 proper fake rows vs planned 442; RLP1_05–08 ran 3186 vs planned 2412. Live counts are authoritative. See `R13_RELAUNCH_PACKET1_LIVE_MONITORING_2026-04-20.md:336-367`.

## Conclusions drawn in-session

- **Hints are a net drag.** `02` trailed `01` at all 8 matched holdout checkpoints (−0.0077 to −0.0115 AUC); Teams-played slice did not rescue them (`R13_RELAUNCH_PACKET1_LIVE_MONITORING_2026-04-20.md:405-415`).
- **Unenhanced proper-data > more hints.** `04` beat `03` at all 8 matched holdout checkpoints by 0.0017–0.0070 AUC.
- **Enhanced proper-data dilutes.** The `04 → 05` dose was larger than documented; extra enhanced lanes did not earn their slots.
- **FT is the default.** Scratch was non-competitive on balance despite a late OOD spike; `R12_G`-style late-scratch improvement was not observed at comparable steps.
- **Directional, not frozen.** `shuffle` split means later arms adding identities could shift train/val/test membership; strict A/B only under `hash_stable`. See `R13_RELAUNCH_PACKET1_LIVE_MONITORING_2026-04-20.md:367-377`.
- **No collapse, gradients intact.** `is_constant_output = 0`, `params_with_grad = 145` — a real comparison, not a debugging failure.
- **Quote** (session `3f5b35f6`, packet-2 confirmation): *"This is a clean win on a shared-snapshot A/B and confirms that with hints fully removed, unenhanced proper-data is the single most valuable new signal — consistent with the packet-1 read on RLP1_04."*
- **Session IDs**: `fcd6be9f`, `3f5b35f6`

## Retrospective (as of 2026-04-24)

- **Hints verdict confirmed across two packets.** RLP2_01 and RLP2_03 (no-hints arms) also underperformed; the hint ladder stayed dead. Packet-2 removed hints from the main slate entirely on the strength of RLP1's finding (`R13_RELAUNCH_PACKET2_EXPERIMENT_PLAN_2026-04-21.md:69-74`).
- **Split-mode caveat, pinned.** Packet 1 used `identity_split_mode: shuffle`; packet 2+ use `hash_stable`. The RLP2 fresh-control drop (RLP2_01 = 0.98662 vs RLP1_01 = 0.98915, Δ −0.00253) has an unquantified split-mode component and cannot be attributed to mutable-source drift alone. See `R13_RELAUNCH_PACKET1_RESULTS_AND_PLANNING_CONTEXT_2026-04-21.md:4-10`.
- **Scratch settled — not revisited.** `RLP1_08` closed the scratch question; FT-from-R12_G-FP32 remains the default.
- **Unenhanced proper-data recipe survived.** `RLP1_04` became the main bet of RLP2_02 and the anchor of RLP3's proper-data thesis. Confirmed.
- **WT-C sidecars deferred, not revived.** Packet-2 explicitly excluded them (`R13_RELAUNCH_PACKET2_EXPERIMENT_PLAN_2026-04-21.md:73-74`); the packet-1 negatives were tested on top of a dominated base, so the negative was more negative than decisive — but nothing later has challenged it.
- **Preprocessing parity — pre-fix.** All RLP1 numbers predate commit `855871e` (2026-04-24), which changed `cv2.INTER_AREA → cv2.INTER_LINEAR` in the retro-score preprocessing path. RLP1 is **not** a post-fix baseline; any RLP1 checkpoint re-scored on the arena/lockbox path today would move. See [../threads/preprocessing_parity_bug.md](../threads/preprocessing_parity_bug.md).
- **Metric-version note.** RLP1 numbers are `best_ood_composite` (AUC-based). The deployment-grade `value_composite` metric was introduced in RLP3; RLP1 leaders would need re-score under current gates for deployment-grade comparison. Directional conclusions (hints fail, unenhanced proper-data is the cleanest new signal, FT > scratch) survive the metric change; numeric rankings do not transfer. See [../threads/promotion_contract_evolution.md](../threads/promotion_contract_evolution.md).
- **Gate-alignment link.** RLP1 is where hints were first tested honestly as weak-signal residue rather than clean supervision — upstream of the later `worst_pool_fpr` / real-pool evaluation-hygiene arc. See [../threads/gate_alignment_story.md](../threads/gate_alignment_story.md).

## Source files

- **Handoffs**:
  - `docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_EXPERIMENT_PLAN_2026-04-19.md:66-105` — design rationale
  - `docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_MONITORING_HANDOFF_2026-04-20.md:30-42` — live run IDs + W&B project
  - `docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_LIVE_MONITORING_2026-04-20.md:108-123` — best-checkpoint ranking; `:336-367` — live-vs-planned count discrepancy
  - `docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_RESULTS_AND_PLANNING_CONTEXT_2026-04-21.md:4-10` — split-mode caveat; `:216-315` — packet lessons (primary retrospective source)
  - `docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_EVALUATOR_HANDOFF_2026-04-19.md` — independent review scaffolding
  - `docs/relaunch_handoffs/R13_RELAUNCH_PACKET2_EXPERIMENT_PLAN_2026-04-21.md:20-30` — packet-1 conclusions that drove packet-2
  - `docs/relaunch_handoffs/R13_RELAUNCH_TRAINING_SOURCE_OF_TRUTH_2026-04-21.md:66-76` — doc-stack placement
- **Yamls**: `experiments/phase2_round13/R13_RLP1_{01..08}_*.yaml`
- **Scorecards / analysis**: *(none — packet-1 was scored only on training-side `best_ood_composite` via W&B; the value_composite + lockbox stack landed in RLP3)*
- **Memory pointers**: *(none load-bearing — relevant entries reference downstream packets)*
