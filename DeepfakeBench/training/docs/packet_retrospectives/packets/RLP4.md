# Packet RLP4  ·  ArcFace-margin fine-scan + Teams reweight, half-and-half on control vs spatial backbone

> **Template contract**: fill every section. If a section truly has no content, write `*(none — reason)*` rather than deleting the heading. Keep the status-card table intact.

## Status card

| Field | Value |
|---|---|
| Dates | 2026-04-22 → 2026-04-23 |
| Slots | 8 (`R13_RLP4_01..08`) |
| Headline lever | ArcFace margin fine-scan `m ∈ {0.125, 0.15, 0.175}` × `{control, spatial}` backbone, plus Teams-fake family-reweight (1.0 → 3.0) on both backbones |
| Leader slot | `R13_RLP4_06_arcface_m015_spatial` (intended candidate; never produced a trained metric — see Results) |
| Leader metric | *(none — all 8 jobs failed before first W&B metric; lever was carried forward on RLP3.5 evidence)* |
| Verdict | ✅ confirmed (m=0.15 control + spatial-aug block carried into RLP5 on RLP3.5 evidence; Teams-reweight lever re-planned and reformulated in RLP5/RLP6) |
| Next-packet decision | RLP5 adopts m=0.15 + spatial aug (0.08/0.40) as the E3 base; drops per-family margin-scan; pivots Teams-gap attack to `include_lanes` + `family_weights` on `proper_visomaster_enhanced_teams_fake` |
| Themes touched | [`promotion_contract_evolution`](../threads/promotion_contract_evolution.md), [`processing_signature_shortcut`](../threads/processing_signature_shortcut.md), [`gate_alignment_story`](../threads/gate_alignment_story.md), [`preprocessing_parity_bug`](../threads/preprocessing_parity_bug.md) |

## Configuration

All 8 slots inherit `R13_RLP35_02_arcface_m015.yaml` and preserve the P3.5 `value_composite` block `(target_mean_fpr=0.03, max_pool_fpr=0.05, stability_jitter_stat=p95)` and `anneal_steps: 8000` (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET4_EXPERIMENT_PLAN_2026-04-22.md:13`).

Deltas vs the RLP3.5 leader (`R13_RELAUNCH_PACKET4_EXPERIMENT_PLAN_2026-04-22.md:25-34`):

- **Control-backbone quartet**: arcface-margin fine scan (m=0.125, 0.175), seed-B replication, Teams reweight (C2).
- **Spatial-backbone quartet**: same 4 levers stacked on the P3-leader spatial-aug block (`context_variation_shift: 0.04 → 0.08`, `context_variation_individual_p: 0.15 → 0.40`). Slot 6 was flagged going in as *"Strongest new-leader candidate — stacks the P3.5 margin win on the P3 spatial win"* (`R13_RELAUNCH_PACKET4_EXPERIMENT_PLAN_2026-04-22.md:32`).

Variants (slot → lever):

- `R13_RLP4_01_arcface_m0125_control` — margin gap-fill between tied m=0.10 and m=0.15.
- `R13_RLP4_02_arcface_m0175_control` — upper-margin probe (the RLP3.5 `m=0.20` ceiling).
- `R13_RLP4_03_arcface_m015_seedB` — top-level `seed: 737 → 742`; `combined_paired.split_seed` and `identity_split_seed` held at 737 to preserve the identity partition (`R13_RELAUNCH_PACKET4_EXPERIMENT_PLAN_2026-04-22.md:116`).
- `R13_RLP4_04_arcface_m015_teams_reweight` — `proper_visomaster_teams_fake: 1.0 → 3.0` on control backbone (`experiments/phase2_round13/R13_RLP4_04_arcface_m015_teams_reweight.yaml:215`). Per-**deployment-category**, not per-method — distinct from the facedancer case the user had declined (`R13_RELAUNCH_PACKET4_EXPERIMENT_PLAN_2026-04-22.md:117`).
- `R13_RLP4_05/06/07_arcface_m{0125,015,0175}_spatial` — margin scan on spatial backbone (`experiments/phase2_round13/R13_RLP4_06_arcface_m015_spatial.yaml:79-82`).
- `R13_RLP4_08_arcface_m015_spatial_teams_reweight` — mirror of slot 04 on spatial backbone.

C1 (add `proper_visomaster_teams` as an OOD fake lane) was **deferred in-session**: `external_fake_sources` does a blind bucket walk with VCD-only identity-regex support, and a manifest-aware OOD source was scoped at ~2–4h in `combined_paired.py` (`R13_RELAUNCH_PACKET4_EXPERIMENT_PLAN_2026-04-22.md:120-127`). Slot 8 was originally C1-on-spatial; the launcher agent replaced it with C2-on-spatial to keep C1 out of the packet's code path.

## Results at the time

**All 8 jobs FAILED before first metric.** Root cause (session `cd19ddfb`): `FileNotFoundError` — the 8 yamls were still untracked in git when image `1.3.195` was built. `launch_experiment.sh` bakes yamls into the container at `/workspace/experiments/…`; all 8 containers died ~30s after start with `FileNotFoundError: [Errno 2]… /workspace/experiments/phase2_round13/R13_RLP4_<slot>_*.yaml`. No W&B init, no OOD composite. **Zero trained `value_composite` for RLP4.**

The RLP3.5 leader `0.7442` was never challenged with new data; the intended comparisons (m=0.125 vs 0.15 vs 0.175 on each backbone; C2 effect on the 22pp Teams gap) never ran. Artifacts: 8 yamls on disk, the plan doc, and the launcher postmortem in session `cd19ddfb`.

## Conclusions drawn in-session

- **Slot-6 was the intended new-leader candidate.** Plan doc: *"Strongest new-leader candidate — stacks the P3.5 margin win on the P3 spatial win"* (`R13_RELAUNCH_PACKET4_EXPERIMENT_PLAN_2026-04-22.md:32`). No in-session refutation; the slot never returned a metric.
- **Teams reweight is a deployment-category lever, not a per-method lever** (`R13_RELAUNCH_PACKET4_EXPERIMENT_PLAN_2026-04-22.md:117`). Distinguished in-session from the facedancer per-method case the user had declined.
- **C1 deferral was structural, not scheduling.** Manifest-aware OOD fake sources did not exist; a bucket-walk schema cannot honor the inventory-level lockbox split. Deferred to a post-training eval task (`R13_RELAUNCH_PACKET4_EXPERIMENT_PLAN_2026-04-22.md:120-127`).
- **Image/yaml hygiene gap.** Session `cd19ddfb`: *"The 8 packet-4 yamls are still untracked in git and were never committed before `1.3.195` was built, so they don't exist inside the image."* This became the forcing function behind the image-rebuild-before-launch rule in memory (`reference_image_rebuild.md`) and the canary-first pattern RLP5 adopted.
- **Session IDs**: `a1a6c1bf` (plan + launch), `cd19ddfb` (failure postmortem + pivot).

## Retrospective (as of 2026-04-24)

- **Verdict ✅ confirmed — by carry-forward, not direct evidence.** RLP5 adopted `arcface_m=0.15` + the spatial-aug block (`context_variation_shift: 0.08`, `context_variation_individual_p: 0.40`) as the E3 base for 7 of 8 slots (session `cd19ddfb`). Defensible because the margin win rests on RLP3.5 slot-02 evidence (`0.7442`, +0.0415 over retro-P3_02) — RLP4 was going to re-test, not originate. Fine-grain `m=0.125/0.175` did not displace `m=0.15`: RLP6's questions already assume m=0.15 as settled.
- **Spatial backbone produced the first real post-RLP3.5 gain.** RLP5's slot-07 (E3 seedB) hit `best_value_composite = 0.7736` on the spatial stack (`R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md:11-14`) — +0.029 over RLP3.5's 0.7442. The lever RLP4_06 was commissioned to demonstrate is the one that moved the needle in RLP5. Settled.
- **Deferred C1 landed in RLP5 via training, not via OOD instrumentation.** RLP5's E3 lane (`include_lanes: + proper_visomaster_enhanced_teams`, weighted) closed the VisoMaster-Teams gap directly: *"Packet 5 E3 successfully closed the headline VisoMaster-Teams training gap (proper lanes from 67%/18% → 94%/96%… composite = 0.7736)"* (`R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md:10-14`). Session `87386521` captured the motivating question: *"teams_tpr - wow, this might be important… we have OTHER teams data (newer, proper) - how are we doing over there?"* The answer came from training data, not from an OOD lane. Slot 8's C1→C2 swap was the right call.
- **Teams-reweight (slots 4 + 8) informed RLP6's framing but not RLP6's fix.** RLP6 found the block to ~0.90 was **not** on the reweight side but on the gate side: avspeech `54%`/`46% FPR` driving τ, and hint contamination in `teams_ood_fake` (`R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md:55-63, 87`). The correction was **real-pool alignment + hint exclusion**, not more reweighting (see [`gate_alignment_story`](../threads/gate_alignment_story.md)). RLP4's Teams-reweight contributed the framing ("Teams is a deployment category, treat it separately"); the lever itself was superseded.
- **Preprocessing-parity status: PRE-FIX.** RLP4 produced no numbers so nothing to rescore, but all downstream RLP5/RLP6 numbers that **validate** the RLP4 carry-forward (the 0.7736 and the 94%/96% per-lane readouts) predate commit `855871e` (`cv2.INTER_AREA → cv2.INTER_LINEAR` at `batch_inference_gcs.py:407` and `arena/model_arena.py:472`). See [`preprocessing_parity_bug`](../threads/preprocessing_parity_bug.md).
- **The launcher postmortem is load-bearing.** RLP4's contribution was procedural: it forced the untracked-yaml → image-rebuild → canary-then-batch discipline that RLP5 (`./dev.sh build-prod -y` bumping `1.3.195 → 1.3.196`, single canary in us-west4 before fan-out) and every subsequent packet inherit. `reference_image_rebuild.md` exists because of this packet.

## Source files

- **Handoffs**:
  - `docs/relaunch_handoffs/R13_RELAUNCH_PACKET4_PLANNING_HANDOFF_2026-04-22.md` — user-preference list :60-66, candidate menu :110-187.
  - `docs/relaunch_handoffs/R13_RELAUNCH_PACKET4_EXPERIMENT_PLAN_2026-04-22.md` — slate :25-34, coverage matrix :38-47, C1 deferral :120-127, gotchas :113-118.
  - `docs/relaunch_handoffs/R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md` — downstream validation; E3 result :10-14, avspeech gate-driver :55-63, hint exclusion :87.
  - `docs/relaunch_handoffs/R13_RELAUNCH_TRAINING_SOURCE_OF_TRUTH_2026-04-21.md` — inherited baseline/split/checkpoint conventions.
- **Yamls**: `experiments/phase2_round13/R13_RLP4_{01..08}_*.yaml` (8 files, 360 lines each, baked from `R13_RLP35_02_arcface_m015.yaml` with single/paired deltas). Representative: `experiments/phase2_round13/R13_RLP4_06_arcface_m015_spatial.yaml:79-82` (spatial block); `:335` (`arcface_m: 0.15`). Slot-03 seed: `R13_RLP4_03_arcface_m015_seedB.yaml` (top-level `seed: 742`; split/identity seeds held at 737).
- **Scorecards / analysis**: *(none — no run produced metrics)*. Launch-infra: `scripts/launch/launch_experiment.sh` (bakes yamls into the image — the failure mode); `scripts/launch/launch_experiment_gcs.sh` (runtime-config alternative considered during pivot; RLP5 chose `./dev.sh build-prod -y` + rebuild instead).
- **Memory pointers**: `reference_image_rebuild.md` (image-rebuild-before-launch rule, created in response to this packet); `feedback_decision_points.md` (C1 vs C2 scope decision).
