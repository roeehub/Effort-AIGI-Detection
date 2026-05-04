# Packet RLP5  ·  enhancement weighting + dose-matched E3 — the headline breakthrough

> **Template contract**: fill every section. If a section truly has no content, write `*(none — reason)*` rather than deleting the heading. Keep the status-card table intact.

## Status card

| Field | Value |
|---|---|
| Dates | 2026-04-23 → 2026-04-23 |
| Slots | 8 (`R13_RLP5_01..08`; slot 7 = breakthrough, slot 8 = E1 control) |
| Headline lever | **E3 recipe** — enable `proper_visomaster_enhanced_teams` lane + `proper_visomaster_teams_fake` family weight 1.0 → 2.5, on the spatial + `arcface_m=0.15` backbone |
| Leader slot | `R13_RLP5_07_E3_seedB` |
| Leader metric | `best_value_composite = 0.7736` @ step 20500 |
| Verdict | 🔬 superseded — **training-side breakthrough real** (VisoMaster-Teams fake recall 67 % → 94 %; enhanced-teams 18 % → 96 %); **deployment interpretation shortcut-driven** (slot-07 pipeline-signature flip; 90/90 lockbox not threshold-reachable) |
| Next-packet decision | RLP5_07 adopted as RLP6 base; RLP6 pivots from training-knobs to gate alignment (drop avspeech + VCD, hint-clean `teams_ood_fake`, drop `wma_failure_fake`) |
| Themes touched | [processing_signature_shortcut](../threads/processing_signature_shortcut.md) (**primary**), [gate_alignment_story](../threads/gate_alignment_story.md), [promotion_contract_evolution](../threads/promotion_contract_evolution.md), [preprocessing_parity_bug](../threads/preprocessing_parity_bug.md) |

## Configuration

No dedicated plan doc; composed in-session as a pivot off the failed overnight packet-4 relaunch, driven by `docs/relaunch_handoffs/R13_RELAUNCH_VISOMASTER_TEAMS_POOL_DIAGNOSTIC_FINDINGS_2026-04-23.md:16-27` (67 %/18 % gap) and :186-226 (E1/E2/E3 levers). Retrospective carrier: `docs/relaunch_handoffs/R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md:11-18`.

Deltas vs. RLP3.5 winner / RLP3 spatial backbone (`R13_RLP3_05`):

- **Enable `proper_visomaster_enhanced_teams` lane** (1484 captures previously off; `R13_RLP5_01_E3_teams2_5.yaml:186-190`) — dominant lever.
- **Reweight** `proper_visomaster_teams_fake` 1.0 → 2.5 + add `proper_visomaster_enhanced_teams_fake` family weight (`R13_RLP5_01_E3_teams2_5.yaml:212`).
- Unchanged: `arcface_m=0.15`, `anneal_steps=8000`, spatial aug 0.08/0.40, A9 gate `(0.03, 0.05, p95)` (`R13_RLP5_07_E3_seedB.yaml:57-60, 332-337`).

**Control**: `R13_RLP5_08_E1_teams2_5` — weight lever alone, **no** enhanced lane (`R13_RLP5_08_E1_teams2_5.yaml:186-189`). A/B anchor.

**Variants** — all 8 share the backbone; 7 of 8 enable the enhanced lane. Deltas on fake family weights:

| # | Slot | `teams_fake` | `enhanced_teams_fake` | Extra |
|---|---|---:|---:|---|
| 1 | `RLP5_01_E3_teams2_5` | 2.5 | 1.0 | E3 core (seed 737) |
| 2 | `RLP5_02_E3_teams4_0` | 4.0 | 1.0 | teams-weight ceiling |
| 3 | `RLP5_03_E3_both2_5` | 2.5 | 2.5 | matched |
| 4 | `RLP5_04_E3_enh4_0` | 2.5 | 4.0 | enhanced-maximal |
| 5 | `RLP5_05_E3_both_heavy` | 4.0 | 3.0 | kill-or-cure |
| 6 | `RLP5_06_E3_deeplive_up` | 2.5 | 1.0 | DeepLive 5→6/3→4/2.5→3.0 |
| **7 ★** | `RLP5_07_E3_seedB` | 2.5 | 1.0 | **seed 742** (split_seed stays 737; `R13_RLP5_07_E3_seedB.yaml:10-15, 87-89`) |
| 8 | `RLP5_08_E1_teams2_5` | 2.5 | — (lane off) | E1 control (`R13_RLP5_08_E1_teams2_5.yaml:9-14`) |

Launch layout (session `cd19ddfb`, 08:49 UTC 2026-04-23): us-west4 ×2, us-east1 ×2, asia-southeast1 ×2 (relocated from europe-west4 after ~20 min PENDING), us-central1 ×2. All 8 RUNNING, 0 failures, image `1.3.196` (fresh build after the packet-4 `FileNotFoundError` wipeout the prior night).

## Results at the time

**Composite leaders** (`best_value_composite`, per `docs/relaunch_handoffs/R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md:11-18, 82-87`):

| Slot | `best_value_composite` | Notes |
|---|---:|---|
| `RLP5_07` E3 seedB ★ | **0.7736** @ step 20500 | +0.0294 vs RLP3.5 leader (0.7442), +0.0505 vs RLP3 leader (0.7232) |
| `RLP5_01` E3 core | 0.7691 | seed-737 sibling → inside seed-noise band |
| `RLP5_08` E1 control | 0.7691 (≈) | "packet-5 robust" per RLP6 plan :87 |

**Per-source detector readouts on the slot-07 checkpoint** (`analysis/feature_space_2026-04-23/per_source_summary.csv:12-15`, same checkpoint as the composite leader):

| Source | `prob_fake_rate_tau05` | Pre-RLP5 reading |
|---|---:|---|
| `proper_visomaster_teams_fake` | **0.9367** (94 %) | 0.672 (`…DIAGNOSTIC_FINDINGS_2026-04-23.md:22`) |
| `proper_visomaster_enhanced_teams_fake` | **0.9600** (96 %) | 0.186 (`…DIAGNOSTIC_FINDINGS_2026-04-23.md:23`) |
| `proper_visomaster_clean_fake` | 0.9867 | baseline high |
| `proper_real_teams__paired` | FPR 0.0067 | 99 % held |
| `tv2_deeplive_fake` | 1.0000 | DeepLive untouched |

**Headline lift**: the two VisoMaster-Teams fake lanes the packet was built to address went 67 %/18 % → **94 %/96 %** at τ=0.5. Full closure of the diagnostic gap.

**Suite-level readout** via the calibrated contract used the post-hoc sanity maps `arena/checkpoint_maps/teams_target_domain.r13_rlp5_07_sanity_{,_dor_}2026-04-23.yaml` against `arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml:72-77` (`teams_real_dor_dev`). The Dor slice is the probe that later surfaced the pipeline-signature failure.

**Why composite is 0.7736, not ~0.90**: the A9 gate forces τ high to keep `external_youtube_avspeech_real` under 5 % FPR (avspeech = 54.2 % / 45.8 % FPR at natural τ; `R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md:54-58`). An **RLP6 finding**, not RLP5. The breakthrough landed inside a metric whose ceiling was gate-bound, not representation-bound.

**Leader checkpoint**: `gs://training-job-outputs/phase2r13_experiments/6jwwb526/value_composite_effort_20260423_step20500_auc0.9908_eer0.0304.pth`.

## Conclusions drawn in-session

- **The enhanced-teams lane was the gap, not the margin.** With `m=0.15` already locked from RLP3.5, enabling `proper_visomaster_enhanced_teams` + the 1.0 → 2.5 weight bump moved the two proper-VisoMaster-Teams fake lanes into the 93-96 % band. Session `cd19ddfb`: *"1,484 captures × enhancers — 5× the visomaster-teams data budget is locked behind a single `include_lanes` line."*
- **DeepLive uplift did not move composite.** User framing (session `cd19ddfb`): *"DeepLive pools at 91% — good, can be better"*; but slot 6's weight increase did not exceed seed-noise. The ceiling there is not weight-reachable.
- **`teams_ood_fake` acknowledged as old-collection.** Session `cd19ddfb`: *"teams_ood_fake — less relevant, this is an old collection. If we do succeed there it's a good signal."* Retained in the gate for visibility, not designed around.
- **Slot-07 is a seed replicate, not an architectural variant.** +0.0045 over slot 01 is inside the RLP3.5 ±0.005 seed-noise envelope. The ★ is because its step-20500 checkpoint landed slightly ahead and became the RLP6 base.
- **Image-rebuild discipline enforced.** Packet-4 overnight failure (all 8 `FileNotFoundError` — yamls not in image `1.3.195`) forced the explicit `./dev.sh build-prod -y` + canary-first workflow; image `1.3.196` was validated in us-west4 before the remaining 7 launched. See `reference_image_rebuild.md`.
- **Session IDs**: `cd19ddfb` (slate design, launch, status); diagnostic ancestry via the FINDINGS handoff.

## Retrospective (as of 2026-04-24)

**The training-side breakthrough is real. The deployment interpretation is not.** Both parts have to be stated explicitly — the gap between them is the most important thing to surface about this packet.

- **What held.** The E3 recipe (dose-matched enhanced + spatial backbone + `arcface_m=0.15`) is carried forward wholesale into RLP6. Every RLP6 slot inherits from `R13_RLP5_07_E3_seedB` (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md:82-84`). The 67 %/18 % → 94 %/96 % closure on the proper-VisoMaster-Teams fake lanes is genuine training-side evidence that the data-composition intervention worked.
- **Why it is shortcut-driven.** Post-hoc, slot-07 was scored on the calibrated contract with the Dor slice (`…_sanity_dor_2026-04-23.yaml`). Memory `project_signature_shortcut_finding.md` is decisive: on `dor_shkedi` (1138 lockbox fakes) the checkpoint gives `mean prob_fake = 0.457`, 14.6 % > 0.9; on `real_dor` (109 frames of **the same face**, different `identity_key`) it gives `mean = 0.038`, 0 % > 0.9. The determinant is not face-artifact features — it is the pipeline/encoding/capture signature. The τ=0.5 gains on enhanced-teams partly reflect the model having learned a pipeline-label correlation.
- **Slot-01 shortcut status: unknown.** Slot-07 beats slot-01 by `+0.0045` composite (0.7736 vs 0.7691) — inside the RLP3.5 `±0.005` seed-noise envelope, with both on the same E3 recipe differing only in top-level seed (742 vs 737). Slot-07 is the **only** RLP5 checkpoint that was retro-scored against the Dor slice and surfaced the shortcut; slot-01 was never probed separately. If slot-01 carries the same shortcut (likely, since it is a seed sibling on the same recipe), the training-side breakthrough framing weakens further — the E3 recipe as a whole is shortcut-entangled, not just the checkpoint that happened to be retro-scored. **Open loop.** No slot-01 probe is scheduled; the Dor slice was re-used to build RLP6's base instead, and the question is moot for deployment because RLP6_04 (descended from slot-07) is the checkpoint actually on the critical path. Worth noting so the slot-07-specific framing is not read as an architectural claim.
- **Lockbox 90/90 is NOT threshold-reachable.** Per the memory file: at τ catching 90 % fakes, real FPR is 63.6 %. Representation problem, not calibration. The dev-calibrated lexicographic τ + lockbox readout is the only place this is legible (see [promotion_contract_evolution](../threads/promotion_contract_evolution.md), `project_promotion_contract.md`); trainer's `value_composite` cannot see it.
- **Entry point to RLP7.** RLP6's slot `R13_RLP6_04_add_enh_clean` later surfaced the webcam false-flag (Dor webcam 0.94 vs same person on laptop 0.02) when gate alignment + the enhanced_clean lane exposed more of the shortcut. That drove RLP7's camera-signature diagnostic packet. See [processing_signature_shortcut](../threads/processing_signature_shortcut.md).
- **Preprocessing-parity: PRE-FIX.** All RLP5 numbers — `best_value_composite=0.7736`, the feature-space table, and the Dor/real_dor scores that drove the shortcut finding — predate commit `855871e` (`cv2.INTER_AREA → cv2.INTER_LINEAR` at `batch_inference_gcs.py:407` and `arena/model_arena.py:472`). The retro-score path used for the sanity maps (`arena/model_arena.py:472`) silently used the wrong interpolation. Post-fix comparable: `analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json`. The *direction* of the shortcut (same-person pipeline flip) is robust across pre/post-fix; *magnitudes* are not. See [preprocessing_parity_bug](../threads/preprocessing_parity_bug.md).
- **Gate-alignment reframing.** RLP6 reframed the ~0.77 ceiling as gate-bound, not representation-bound: removing `external_youtube_avspeech_real` + `zoom_vcd_real` from the real gate pushes composite toward ~0.90 without retraining. RLP5's training-knob lift and RLP6's gate-alignment lift are orthogonal and additive; neither touches the shortcut. See [gate_alignment_story](../threads/gate_alignment_story.md).
- **Story continues** in `RLP6.md`, `RLP6B.md`, `RLP7.md`, and the three threads linked above.

## Source files

- **Handoffs**:
  - `docs/relaunch_handoffs/R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md:11-18, 82-106` — primary retrospective carrier (no dedicated RLP5 plan); §1 TL;DR records 67 %/18 % → 94 %/96 % + composite 0.7736, and §"The 8 experiments" records the RLP6 inheritance from `R13_RLP5_07_E3_seedB`.
  - `docs/relaunch_handoffs/R13_RELAUNCH_VISOMASTER_TEAMS_POOL_DIAGNOSTIC_FINDINGS_2026-04-23.md:16-27, 99-108, 186-226` — diagnostic that motivated the slate and named the E1/E2/E3 levers.
  - `docs/relaunch_handoffs/R13_RELAUNCH_TRAINING_SOURCE_OF_TRUTH_2026-04-21.md` — baseline / split / checkpoint conventions inherited.
- **Yamls**: `experiments/phase2_round13/R13_RLP5_0{1..8}_*.yaml`; special-case `R13_RLP5_07_E3_seedB.yaml:10-15` (seed 742 top-level, 737 split) and `R13_RLP5_08_E1_teams2_5.yaml:186-189` (enhanced lane explicitly off).
- **Scorecards / analysis**:
  - `analysis/feature_space_2026-04-23/REPORT.md` + `per_source_summary.csv` — feature-space readout on slot-07 checkpoint.
  - `analysis/bucket_comparison_2026-04-23/REPORT.md:16-22, 79-97, 169-187` — pixel-statistics distribution analysis framing the structural gap.
  - `arena/checkpoint_maps/teams_target_domain.r13_rlp5_07_sanity_2026-04-23.yaml` + `…_dor_2026-04-23.yaml` — post-hoc sanity maps; the Dor variant surfaced the webcam false-flag later.
  - `arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml:72-77` — `teams_real_dor_dev` suite entry that made the shortcut visible.
- **Memory pointers** (load-bearing):
  - `project_signature_shortcut_finding.md` — **retrospective pivot**: "same person `dor_shkedi` vs `real_dor` flips model output; 90/90 on lockbox is NOT threshold-reachable."
  - `project_promotion_contract.md` — authoritative readout is dev-calibrated lexicographic τ + lockbox, not trainer's `value_composite`.
  - `reference_image_rebuild.md` — canary-first + VERSION-bump workflow forced by packet-4 overnight wipeout.
