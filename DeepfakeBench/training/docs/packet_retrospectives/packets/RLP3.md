# Packet RLP3  ·  R13 Relaunch Packet 3 — the pivotal instrumentation packet

> **Template contract**: fill every section. If a section truly has no content, write `*(none — reason)*` rather than deleting the heading. Keep the status-card table intact.

## Status card

| Field | Value |
|---|---|
| Dates | 2026-04-21 → 2026-04-22 |
| Slots | 9 (`R13_RLP3_00_PRELAUNCH_SMOKE`, `R13_RLP3_01..08`) |
| Headline lever | Land A1–A10 instrumentation + introduce deployment-aligned `value_composite` |
| Leader slot | `R13_RLP3_05` (low_arcface + spatial, stacked) |
| Leader metric | retro NEW-gate `value_composite = 0.7232` (gates `(0.03, 0.05, p95)`); legacy `best_value_composite = 0.6224` |
| Verdict | ✅ confirmed (instrumentation + metric shift held); specific leader numbers partially superseded |
| Next-packet decision | RLP3.5 arcface-margin scan (m ∈ {0.10, 0.15, 0.20}) on the spatial-enabled baseline; defer heterogeneous stacking |
| Themes touched | [`promotion_contract_evolution`](../threads/promotion_contract_evolution.md), [`processing_signature_shortcut`](../threads/processing_signature_shortcut.md), [`preprocessing_parity_bug`](../threads/preprocessing_parity_bug.md) |

## Configuration

- **Biggest lever**: A1–A10 instrumentation landed end-to-end in every slot. Plan at `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_EXPERIMENT_PLAN_2026-04-21.md:56-304`.
  - A1 per-video jitter (max/p95/spike_rate), mirrored onto val_holdout
  - A2/A2b dual-checkpoint final_eval (save BOTH `best_ood_composite` and `best_value_composite`) + retroactive runner for packets 1-2
  - A3 / A3b three deterministic lighting + three deterministic spatial stress OOD lanes (`R13_RLP3_02_FT_proper_main.yaml:268-320`)
  - A4 proper-data `build_id` surfaced into W&B config
  - A6 enhanced-proper OOD lanes carved OUT of `ood_composite` but logged into `value_composite` — kills the selection-bias loop on the enhanced question
  - A7 `summary/*` 10-metric namespace + `per_video_scores.parquet` artifact
  - A8 first OOD at step 1000 (not 5000) for FT runs (`R13_RLP3_02_FT_proper_main.yaml:223-225`)
  - A9 `value_composite` = `0.6 * teams_fakes_tpr + 0.3 * other_fakes_tpr + 0.1 * stability` at the τ meeting `mean_FPR ≤ 0.02` AND `max_FPR ≤ 0.04` across 6 real pools (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_EXPERIMENT_PLAN_2026-04-21.md:224-274`)
  - A10 blake2b video-id 90/10 held-out partition per OOD pool
- **Control**: `R13_RLP3_01_FT_control.yaml` — packet-3's OWN baseline; the packet-1→2 split-mode switch already opened a confounding gap.
- **Slate** (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_EXPERIMENT_PLAN_2026-04-21.md:338-348`): `01` clean FT; `02` main proper, seed 737; `03` low-ArcFace `s_end=10`; `04` +spatial; `05` stacked low-ArcFace+spatial (`R13_RLP3_05_FT_proper_low_arcface_spatial.yaml:65-77`); `06` seed 239; `07` lighting aug; `08` +dose-matched enhanced-teams (~342 rows).
- Selection STAYED on `best_ood_composite` throughout RLP3; `value_composite` was readout-only per `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_EXPERIMENT_PLAN_2026-04-21.md:259-264`.

## Results at the time

Retro-scored NEW gates, from `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_RETRO_SCORE_RESULTS_2026-04-22.md:77-82`:

| Slot | retro NEW VC | legacy-gate VC | teams_tpr | other_tpr | stability | τ | mean_fpr | max_fpr |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 02 main | 0.7027 | 0.6097 | 0.6673 | 0.8587 | 0.4475 | 0.9696 | 0.0298 | 0.0408 |
| 04 spatial | 0.7215 | 0.6237 | 0.6586 | 0.9271 | 0.4818 | 0.9629 | 0.0300 | 0.0436 |
| 05 low_arc+spatial ★ | **0.7232** | 0.6224 | 0.6611 | 0.9280 | 0.4811 | 0.9609 | 0.0300 | 0.0436 |

- Leader vs 02 control delta: **+0.02** under the NEW metric — a real but modest lift.
- **Ranking flip**: legacy gates had slot 04 leading slot 05 (`0.6237 > 0.6224`); NEW gates flipped it (`0.7232 > 0.7215`). Effectively tied; inside seed noise (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_RETRO_SCORE_RESULTS_2026-04-22.md:83`).
- Sanity drift `retro 0.6030 vs training 0.6097` (`-0.0067`) — systematic, documented `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_RETRO_SCORE_RESULTS_2026-04-22.md:30-58`.
- **The "5× smaller deltas" observation**: legacy spread ~0.10–0.11 VC; NEW spread ~0.02 for the same checkpoints. The most consequential readout of the packet.
- Slot 07 lighting and slot 08 dose-matched enhanced: non-promoting in-session (07 failed ≥2/3 lighting-preset threshold; 08 did not beat 02 on enhanced-proper by more than σ_seed). Slot 01 trajectory was bracketed by RLP2_01 — the drift question stayed open.
- Retroactive `value_composite` also ran for `RLP1_01 / RLP1_04 / RLP2_01 / RLP2_02 / RLP2_06` via `tools/retroactive_final_eval.py` to put past packets on the new scale.

## Conclusions drawn in-session

- `value_composite` is the first metric shaped by the deployment value hierarchy (worst-pool FPR gate + 0.6/0.3/0.1 weighting + NaN escape for spiky real pools). It surfaces signal legacy `ood_composite` was masking.
- Session `99732ab8`: pulling `best_value_composite` instead of current flipped the packet's ranking and showed *"all runs are already past their peak"* — dual-checkpoint A2 is the only reason we could say that.
- `train_sweep.py` originally dropped the nested `value_composite` block (fix in commit `872502c`; session `02873fd0`: *"it's a nested dict that falls through"*). Without this every RLP3.5 new-gate run would have silently used legacy gates.
- Slot-07 lighting probe was flagged in-session as seeding a camera/ISP-signature investigation line. This is the anchor the `processing_signature_shortcut` thread attributes to RLP3.
- Packet-4 call (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_RETRO_SCORE_RESULTS_2026-04-22.md:25`): *"focus packet-4 experimentation on arcface-margin scanning ... rather than stacking heterogeneous hypotheses."*
- Split-mode caveat (A5) explicitly locked in: packet-1 used `shuffle`, packet-2+ uses `hash_stable`, so the RLP2_01 drop cannot be attributed to source drift alone (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_EXPERIMENT_PLAN_2026-04-21.md:29`).
- **Session IDs**: `eabf9ffb`, `1e871e7e`, `e65d2f50`, `f990ef28`, `6a95f3da`, `02873fd0`, `99732ab8`, `4f89363c`, `69cf0ffe`.

## Retrospective (as of 2026-04-24)

- **`value_composite` survived, but is not the deployment gate.** User memory is authoritative: trainer's `value_composite` is not deployment-grade; authoritative readout is dev-calibrated lexicographic τ then lockbox. RLP3 permanently reshaped in-experiment ranking, but the promotion contract built later runs lexicographically over different axes. See [`promotion_contract_evolution`](../threads/promotion_contract_evolution.md).
- **Preprocessing-parity status: PRE-FIX.** Every RLP3 number — training-time `best_value_composite`, the three NEW-gate retro rows (`0.7027 / 0.7215 / 0.7232`), legacy sanity, AND the A2b retroactive pass over RLP1/RLP2 checkpoints — predates commit `855871e` (`cv2.INTER_AREA → cv2.INTER_LINEAR` at `batch_inference_gcs.py:407` and `arena/model_arena.py:472`). The retro-score path IS the code path that changed, so RLP3's retro numbers carry silent interpolation drift. The post-fix comparable is `analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json`. See [`preprocessing_parity_bug`](../threads/preprocessing_parity_bug.md).
- **Leader-ranking numbers are partially superseded.** The slot-04/05 flip sits inside sub-σ_seed noise, and no post-fix rescore has been applied to RLP3 checkpoints. Read the ordering as: spatial-ish stack ≈ +0.02 NEW over plain-02; low-ArcFace adds nothing distinguishable from noise. Do not cite `0.7232` without the pre-fix caveat.
- **Slot-07 seeded the processing-signature shortcut.** The lighting-aug probe (0.40 scale, stronger than baseline) produced the first sign lighting/processing cues move the classifier independently of identity/fakeness. The line culminates in Packet-7 camera-signature work and the user-memory finding "same person dor_shkedi vs real_dor flips model output; 90/90 on lockbox is NOT threshold-reachable." RLP3 is where the hypothesis first became instrumentable. See [`processing_signature_shortcut`](../threads/processing_signature_shortcut.md).
- **Split-mode artifact persists.** The packet-2 fresh-control drop (`RLP2_01 = 0.98662` vs `RLP1_01 = 0.98915`) cannot be attributed to source drift alone — part is pure split-mode artifact. RLP3_01 was designed to resolve this; trajectory was bracketed but not conclusive, and no final "drift is/isn't real" verdict was produced.
- **What held.** The A1–A10 instrumentation stack is still operational in every packet after this. Dual-checkpoint saving, A6 carve-out, A10 held-out partition, and the A9 composite definition are unchanged in RLP3.5 through RLP7 eval paths.
- **What later packets did.** RLP3.5 ran the arcface-margin scan recommended here; its slot-02 (m=0.15) hit retro-NEW `value_composite=0.7442`, +0.0415 over RLP3_02 under matched gates. The "RLP3 leader was 0.7232" framing was effectively retired within 24h.

## Source files

- **Handoffs**:
  - `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_EXPERIMENT_PLAN_2026-04-21.md:1-615` (plan + A1–A10 + critic revision log)
  - `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_RETRO_SCORE_RESULTS_2026-04-22.md:1-139` (retro output, leader table, slot-07 don't-fire decision)
  - `docs/relaunch_handoffs/R13_RELAUNCH_TRAINING_SOURCE_OF_TRUTH_2026-04-21.md`
- **Yamls**: `experiments/phase2_round13/R13_RLP3_00_PRELAUNCH_SMOKE.yaml`, `R13_RLP3_01_FT_control.yaml` … `R13_RLP3_08_FT_proper_main_plus_dose_matched_enhanced.yaml`; `R13_RLP3_02_FT_proper_main__LEGACY_GATES_FOR_RETRO_SANITY.yaml` and `R13_RLP3_{02,04,05}_*__NEW_GATES_FOR_RETRO.yaml` (gate-shift A/B yamls used by the retro runner).
- **Scorecards / analysis**: W&B project `phase2r13_experiments` (checkpoint run `zifvogm6`; retro jobs `1213846543413542912`, `2744540314675970048`, `6225227028770062336`). Retro runner: `retro_score_value_composite.py`. Trainer helper: `trainer/trainer.py:219` (`_compute_value_composite`) invoked at `trainer/trainer.py:3092`.
- **Memory pointers**: `project_promotion_contract.md`; `project_signature_shortcut_finding.md` (slot-07 seed); `reference_image_rebuild.md` (retro needed images `1.3.194` + `1.3.195`).
