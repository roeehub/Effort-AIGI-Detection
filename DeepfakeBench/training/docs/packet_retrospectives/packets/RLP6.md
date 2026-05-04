# Packet RLP6  ·  Gate alignment + fake-pool hygiene (and the slot that later broke the shortcut model open)

> **Template contract**: fill every section. If a section truly has no content, write `*(none — reason)*` rather than deleting the heading. Keep the status-card table intact.

## Status card

| Field | Value |
|---|---|
| Dates | 2026-04-23 → 2026-04-24 |
| Slots | 8 (`R13_RLP6_01..08`) — side-branch `RLP6B` (quality-lambda sweep, 8 slots) tracked separately in [RLP6B.md](RLP6B.md) |
| Headline lever | Gate-alignment canary + fake-pool hygiene — three corrective levers on top of RLP5_07_E3_seedB |
| Leader slot | `R13_RLP6_04_add_enh_clean` |
| Leader metric | `value_composite = 0.9006` (step 23500, `auc=0.9942`, `eer=0.0169`) |
| Verdict | ✅ confirmed (gate-alignment levers) / 🔬 superseded (RLP6_04 leader-checkpoint scorecard) |
| Next-packet decision | Take `RLP6_04` forward to Packet-7 as the base for spatial/codec aug variants (`RLP7_04/05`) after WS-P0 preprocessing-parity fix + post-fix rescore re-validation |
| Themes touched | [gate_alignment_story](../threads/gate_alignment_story.md) · [preprocessing_parity_bug](../threads/preprocessing_parity_bug.md) · [processing_signature_shortcut](../threads/processing_signature_shortcut.md) · [promotion_contract_evolution](../threads/promotion_contract_evolution.md) |

## Configuration

All 8 slots inherit from `R13_RLP5_07_E3_seedB` (packet-5 leader at `composite=0.7736`). Packet-6 was a **yaml-only** packet plus one already-landed code change (`path_exclude_contains` on external fake/real loaders, `data/validation_sources.py` / `data/sources/combined_paired.py`, noted at `docs/relaunch_handoffs/R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md:128-145`).

Three corrective levers (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md:22-40`):

1. **Drop `external_youtube_avspeech_real` (+ `zoom_vcd_real`) from `external_real_sources`** — avspeech sits at 54.2% accuracy / 45.8% FPR at natural thresholds and is the dominant *real-pool* gate driver, while not being deployment-distribution (deployment target = Teams-passed reals, covered by `teams_ood_real` at 97.8% / 2.2% FPR).
2. **Hint-exclude `teams_ood_fake`** with `path_exclude_contains: ["/visomaster_"]` — drops ~535 failed-deepfake hint folders from the gate. Per `docs/VISOMASTER_BAD_DATA_POLICY_PLAN_IMPACT_2026-04-17.md:39-44`: 4,904 visomaster-hint sample IDs are `ignore`, 480 retained as `visomaster hints`, 202 as `visomaster hints (teams)`. The fake-side gate had been grading the detector against near-real content labeled fake.
3. **Drop `wma_failure_fake`** — sits 14.55 Fréchet from `deeplive_enh_fake` (effectively in-distribution, 99.4% detected). Packet-5's hypothesis that it drove `worst_pool_fpr` was wrong; `worst_pool_fpr` is computed over real pools only (`trainer/trainer.py:182-216`).

Control slot is `RLP5_07_E3_seedB` itself (retro-baseline), not a same-packet run.

Variants (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md:80-94`):

- **`R13_RLP6_01_gate_align_canary`** — all three levers stacked, launched alone first. `experiments/phase2_round13/R13_RLP6_01_gate_align_canary.yaml`.
- **`R13_RLP6_02_hint_clean_only`** — lever 2 only (isolation probe for hint contamination).
- **`R13_RLP6_03_drop_wma_only`** — lever 3 only (confirms wma is not a gate driver; corroborates the corrected model).
- **`R13_RLP6_04_add_enh_clean`** — slot 1 + `proper_visomaster_enhanced_clean` added to `proper_data.include_lanes` with `family_weights = 2.0`. `experiments/phase2_round13/R13_RLP6_04_add_enh_clean.yaml:1-21`. **Became packet leader.**
- **`R13_RLP6_05_seed_variance`** — slot 1 + `seed 742 → 745` (seed-stability on the deployment-aligned gate).
- **`R13_RLP6_06_avspeech_readout_only`** — slot 1 + avspeech/VCD moved to a readout-only OOD block (observability-preserving variant).
- **`R13_RLP6_07_heavy_enh`** — slot 4 + `enhanced_clean_fake` weight `2.0 → 4.0` (saturation probe).
- **`R13_RLP6_08_E1_gate_align`** — E1 backbone + slot-1 gate changes (second anchor).

Base code change (landed this session, not packet-specific): `data/validation_sources.py` and `data/sources/combined_paired.py` now accept `path_exclude_contains` on external real/fake + stress-OOD loaders. Opt-in; legacy yamls unaffected.

## Results at the time

All eight slots launched. `R13_RLP6_04_add_enh_clean` became the leader at step 23500 with **`value_composite = 0.9006`** (`auc=0.9942`, `eer=0.0169`) — the biggest within-packet jump in the relaunch sequence (packet-5 leader was 0.7736; this is +0.127 under the same gate definition). Checkpoint at `gs://training-job-outputs/phase2r13_experiments/h2pdu6i5/value_composite_effort_20260424_step23500_auc0.9942_eer0.0169.pth` (cited in `HANDOFF.md:158-159` and `docs/relaunch_handoffs/R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md:1-10`).

Gate behaviour on slot-1/4/5 confirmed the corrected analysis:

- Dropping avspeech + VCD unblocked `τ` — the bisection no longer had to push up to tame 45.8% avspeech FPR.
- `teams_fakes_tpr` and `other_fakes_tpr` both rose well above the packet-5 0.70–0.85 band.
- `wma_failure_fake` removal was cleanup only, as predicted (slot-3 stayed near packet-5 numbers — see `docs/relaunch_handoffs/R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md:201-207`).

Suite-level: per the subsequent camera-signature work, RLP6_04's contract-suite scorecard had it clear for promotion under `value_composite`. No aggregate real-pool FPR blow-up was visible at in-session rescore.

Artifacts:

- `arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json` — target-domain manifest Dor-annotated for this packet.
- `experiments/phase2_round13/R13_RLP6_*.yaml` — all 8 slot yamls.
- `HANDOFF.md` — Packet-7 pre-launch gate, anchored on the RLP6_04 leader checkpoint.

## Conclusions drawn in-session

- **The corrected gate-block model is the packet's first-class finding.** The earlier framing that `wma_failure_fake` drove `worst_pool_fpr` was wrong: `trainer/trainer.py:161-216` shows `worst_pool_fpr` is computed over real pools only (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md:43-48`). The real gate drivers were three real pools — and two of them (avspeech, zoom_vcd) are not deployment-distribution.
- **avspeech is a gate blocker, not a deployment pool.** At τ≈0.44 it sits at 54.2%/45.8% FPR; keeping it in the gate force-pushes τ to ~0.995 and crushes TPR (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md:53-63`). Dropping it was predicted to recover ≥+0.10 on the composite; slot-4 delivered +0.127.
- **Hint exclusion was hygiene, not a driver.** Predicted swing ≤ ±0.01 (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md:88`). Slot-2 corroborated.
- **`wma_failure_fake` drop was cleanup only.** Predicted composite unchanged; slot-3 confirmed. The earlier `bucket_comparison` causation claim is explicitly marked superseded (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md:217-220`).
- **`proper_visomaster_enhanced_clean` at weight 2.0 adds signal on top of the gate fix.** Slot-4's jump above slot-1 justified the lever.
- **Session IDs**: convmem coverage for RLP6 is thin (packet launched 2026-04-23, sessions largely still open at the time of this writing) — `convmem search "RLP6"` returns only upstream RLP3/RLP4/RLP5 threads. Primary traceability is via the handoff + post-fix rescore artifact (see § Source files).

## Retrospective (as of 2026-04-24)

Split verdict. The corrective levers held; the leader-checkpoint scorecard did not.

**What held.** The three gate-alignment levers are the operating baseline for Packet-7 and beyond. Avspeech and zoom_vcd are out of the deployment gate permanently; `path_exclude_contains: ["/visomaster_"]` on `teams_ood_fake` is now standard; `wma_failure_fake` is a demoted readout pool. The corrected real-pool-only `worst_pool_fpr` model is the anchor for deployment vs. readout FPR reasoning going forward.

**What fell.** `R13_RLP6_04_add_enh_clean` became infamous within 24 hours. Controlled 2-camera tests on 2026-04-24 (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md:5-19`): Dor laptop no-VB → 0.02; Dor webcam same scene no-VB → 0.94; Roee Mac webcam with VB → 0.90; Roee Windows laptop → 0.01. Same subject, same lighting, different camera flipped the score — camera/ISP-signature shortcut, made undeniable. Packet-5 slot-07 had already flipped on `dor_shkedi` vs `real_dor` (memory `project_signature_shortcut_finding.md`), and lockbox 90/90 is **not** threshold-reachable on RLP6_04.

**Two distinct failures stacked.** (a) **Preprocessing drift.** The 0.94 Dor number came from `arena/model_arena.py:472` and the deploy server both running `cv2.INTER_AREA`, while training uses `INTER_LINEAR` (`combined_paired.py:3455`). WS-P0 commit `855871e` fixed the two repo paths. (b) **Structural shortcut.** The post-fix rescore (`analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json`) shows the fix alone does not close the gap — anchor pool `dor-real-webcam-false-flag-no-virtual-bg` moved only `-0.008` (pre-fix 0.94 → post-fix 0.932); decision-tree verdict **`fully_structural__launch_full_subset`**. Clean pools stayed <0.05, false-flag pools stayed 0.93–0.96. Preprocessing was noise; the shortcut is the dominant residual.

**Preprocessing-parity note (explicit).** In-session RLP6_04 scorecard numbers are **PRE-fix** (pre-commit `855871e`). The authoritative post-fix readout on the 6 Dor/Roee pools is `analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json` — cite that, not the W&B `value_composite` log, when comparing RLP6_04 to post-fix baselines. See [preprocessing_parity_bug](../threads/preprocessing_parity_bug.md).

**Calibration probe (WS-P1).** `analysis/calibration_probe_2026-04-24.summary.json` — 30.1% average gap closure across target FPRs {5, 10, 15, 25}%. Below the ≥60% "calibration is the lever" threshold, at the ≤30% "training-aug is the lever" threshold. Training-aug is dominant; calibration is complementary.

**What the story becomes.** RLP6_04 forces three reinterpretations: (i) `value_composite` is not deployment-grade — a legal-under-contract checkpoint can false-flag real Teams participants on the wrong camera; (ii) preprocessing parity across training and every inference path is load-bearing; (iii) fake recall is not the binding failure — camera/ISP signature is. All three point forward to Packet-7 (spatial + codec aug, per-camera calibration, per-identity reducer). See [processing_signature_shortcut](../threads/processing_signature_shortcut.md) and [calibration_vs_training_aug](../threads/calibration_vs_training_aug.md).

**Relation to `RLP6B`.** Separate side-branch, 8 slots sweeping `quality_lambda` + regularization knobs (`experiments/phase2_round13/R13_RLP6B_01..08_*.yaml`). Does not test gate alignment. See [RLP6B.md](RLP6B.md).

## Source files

- **Handoffs**:
  - `docs/relaunch_handoffs/R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md` — primary plan. Three-lever framing :22-40, corrected gate-block analysis :43-48, gate-driver table :53-58, math :66-79, slate :80-94, coverage matrix :96-106, code-change notes :128-145, deferred / dropped :217-225.
  - `docs/relaunch_handoffs/R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md` — primary retrospective. 2-camera evidence :5-19, WS-P0 preprocessing fix :27-36, WS-P1 calibration probe :40-50, per-identity reducer :50-58, launch subset rationale :92-106.
  - `HANDOFF.md` — Packet-7 pre-launch gate. RLP6_04 checkpoint path :158-159, decision tree :200-210, open workstreams :45-54.
  - `docs/VISOMASTER_BAD_DATA_POLICY_PLAN_IMPACT_2026-04-17.md:39-44` — 4904/480/202 hint-exclusion accounting.
- **YAMLs**: `experiments/phase2_round13/R13_RLP6_0{1..8}_*.yaml` (all eight slots listed in §Configuration). `RLP6B` yamls `R13_RLP6B_0{1..8}_*.yaml` are sibling-packet (see [RLP6B.md](RLP6B.md)).
- **Scorecards / analysis**:
  - `analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json` — post-fix readout on 6 Dor/Roee pools; verdict `fully_structural__launch_full_subset`.
  - `analysis/rlp6_04_postfix_rescore_2026-04-24.per_frame.csv` — per-frame pre/post scores.
  - `analysis/calibration_probe_2026-04-24.summary.json` — WS-P1 probe (0.301 avg gap closure).
  - `analysis/lockbox_failure_contact_2026-04-24.index.json` — failure-frame contact sheet.
  - `arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json` — Dor-annotated manifest.
  - `trainer/trainer.py:161-294` — authoritative on `_find_threshold_for_mean_fpr` + `_compute_value_composite`; the source for the real-pool-only correction.
- **Memory pointers**: `project_signature_shortcut_finding.md` (lockbox 90/90 not threshold-reachable on RLP6_04); `project_promotion_contract.md` (dev-calibrated lockbox readout, not trainer's `value_composite`, is deployment-grade — RLP6_04 is the concrete slot that made this rule load-bearing); `reference_image_rebuild.md` (VERSION bump for the `path_exclude_contains` code change).
