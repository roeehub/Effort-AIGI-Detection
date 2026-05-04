# Packet RLP7  ·  Camera/ISP-signature shortcut countermeasure (Teams spatial + codec aug, per-camera calibration, per-identity reducer)

> **Template contract**: fill every section. If a section truly has no content, write `*(none — reason)*` rather than deleting the heading. Keep the status-card table intact.

## Status card

| Field | Value |
|---|---|
| Dates | 2026-04-24 (authored; in-flight / pre-launch gate) |
| Slots | 5 (`R13_RLP7_01..05`) |
| Headline lever | Training-time augmentation (Teams-passthrough spatial `ShiftScaleRotate`, Teams codec-sim, wide-CCT / wide-gamma lighting) to break the same-person-different-webcam score flip; paired with per-camera calibration probe (WS-P1) and per-identity reducer (WS-P2.b) as complementary readout-side levers |
| Leader slot | *(none — no checkpoint trained yet; pre-launch gate cleared but launch pending user trigger)* |
| Leader metric | *(none — not yet launched)* |
| Verdict | 🟡 in-flight (pre-launch gate cleared with verdict `fully_structural__launch_full_subset`; launch pending user pick of subset) |
| Next-packet decision | Launch the suggested subset (`RLP7_05`, `RLP7_02`, `RLP7_04`) forked off RLP6_04 per the post-fix re-score decision tree; `RLP7_01`/`_03` in reserve — controlled-pool evidence does not point at lighting as dominant |
| Themes touched | [processing_signature_shortcut](../threads/processing_signature_shortcut.md) (primary — this packet IS the response) · [preprocessing_parity_bug](../threads/preprocessing_parity_bug.md) (WS-P0 commit `855871e` landed here) · [calibration_vs_training_aug](../threads/calibration_vs_training_aug.md) (WS-P1 30.1% result is the core data point) · [gate_alignment_story](../threads/gate_alignment_story.md) (Teams spatial-aug wiring WS-P4.a is adjacent) |

## Configuration

All five slots fork the Packet-6 leader `R13_RLP6_04_add_enh_clean` at `gs://training-job-outputs/phase2r13_experiments/h2pdu6i5/value_composite_effort_20260424_step23500_auc0.9942_eer0.0169.pth` (`experiments/phase2_round13/R13_RLP7_04_teams_spatial_only.yaml:62-64`). Every slot is a narrow-delta augmentation override; backbone, ArcFace head, LR schedule, sampling weights, gate, and 10k-step budget are held constant at RLP6_04 values. Seeds 740–745; RLP7_01/02/03 all share 742, RLP7_04=744, RLP7_05=745 — the 742 collision is a flagged user decision point (`R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md:122-125`).

- Control slot: `RLP6_04` itself (retro-baseline); no in-packet control.
- **`R13_RLP7_01_lighting_aggressive`** — wide CCT `[2200, 9500]K`, brightness `[-0.35, 0.70]`, `gamma_up_p=0.25`, `shadow_p=0.20`, `teams_codec_sim_p=0.15`. Tests the lighting-dominant hypothesis (wb_rb_ratio, highlight_clip, specular_hotspots). `R13_RLP7_01_lighting_aggressive.yaml:77-99`.
- **`R13_RLP7_02_codec_aggressive`** — `webcam_codec_p=0.35`, `webcam_codec_quality=[20,65]`, `teams_codec_sim_p=0.40`, `jpeg_lower=30`, `downscale_min=0.35`. Tightest match to the fingerprint-diff evidence (`dct_hf_ratio` max_sep=3.46, `mean_cb`=2.23, `mean_cr`=1.97). `R13_RLP7_02_codec_aggressive.yaml:72-86`.
- **`R13_RLP7_03_combined`** — V1+V2 at reduced intensities (`cct_p=0.40`, `webcam_codec_p=0.25`, `teams_codec_sim_p=0.25`). Matches the `V3_combined` verdict (`dor_pool_fingerprints_2026-04-24.summary.json:220-223`). `R13_RLP7_03_combined.yaml:72-94`.
- **`R13_RLP7_04_teams_spatial_only`** — *new this session.* Opt-in `teams_passthrough_special_aug_enabled=true` + `ShiftScaleRotate` (shift ±3%, scale ±5%, rotate ±3°, p=0.5). No codec, no extra lighting. Isolates the spatial-geometry shortcut. `R13_RLP7_04_teams_spatial_only.yaml:83-92`.
- **`R13_RLP7_05_teams_spatial_plus_codec`** — *new this session.* Slot-04 deltas + `teams_codec_sim_p=0.25`, `teams_codec_sim_quality=[25,70]`. Complementarity test: slot-05 > slot-04 and > slot-02 means the axes combine. `R13_RLP7_05_teams_spatial_plus_codec.yaml:75-86`.

Yaml-only enablement — no model or loss change. `teams_passthrough_special_*` and `teams_codec_sim_*` all survive the `combined_paired.py:4772` preset-key filter (verified via `_build_teams_passthrough_pipeline`). The one load-bearing code change is WS-P0: `cv2.INTER_AREA → cv2.INTER_LINEAR` in `batch_inference_gcs.py:407` and `arena/model_arena.py:472` (commit `855871e`), matching `combined_paired.py:3455`. Guarded by `tests/test_inference_train_preprocessing_parity.py` (5/5 pass).

## Results at the time

*(no training results yet — packet has not launched.)* Four non-training artifacts gate and shape the launch:

- **WS-P0 preprocessing-parity fix** — commit `855871e`. Both repo inference paths now match training. Every pre-fix retro-score is now labelled suspect. See [preprocessing_parity_bug](../threads/preprocessing_parity_bug.md).
- **WS-P1 per-camera calibration probe** — `analysis/calibration_probe_2026-04-24.summary.json:446-459`. Average gap closure **0.301** across target FPRs {5, 10, 15, 25}%; per-target {0.05: 0.000, 0.10: 0.333, 0.15: 0.286, 0.25: 0.583}. Verdict `mixed_both_levers_needed` — at the **≤0.30 training-aug-is-right-lever** threshold. Per-pool means confirm the pattern (`dor-real-webcam-false-flag-no-virtual-bg`: 0.940 vs `dor-real-laptop-correct-no-virtual-bg-whiteish`: 0.018).
- **WS-P2.b per-identity reducer** — `arena/postprocess_per_identity.py`. Groups `videos_report.csv` by `group_key` and recomputes `real_fpr` / `fake_recall` per identity with `--verify`. Enables "narrowest per-identity FPR spread" as the explicit Packet-7 leader-pick metric (`HANDOFF.md:233-240`).
- **Post-fix baseline re-score (the pre-launch gate itself)** — `analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json`. RLP6_04 through the fixed `INTER_LINEAR` path on the 6 Dor/Roee pools. Anchor pool `dor-real-webcam-false-flag-no-virtual-bg` moved only **-0.008** (pre-fix mean 0.940 → post-fix 0.932). Clean pools ≤0.02; false-flag pools 0.75–0.96. `decision_tree.verdict = "fully_structural__launch_full_subset"` (:80-84). Preprocessing drift was noise; the camera/ISP shortcut is the dominant residual — **launch the full subset**.

Fingerprint diagnostics that shape launch priority (not in-packet training results): `dor_baseline_repro_2026-04-24.summary.json:947-983` ranks `dct_hf_ratio` (max_sep 3.46), `wb_rb_ratio` (2.32), `mean_cb` (2.23), `mean_cr` (1.97) as the most consistent pool-separating signals — two codec-aligned, two lighting/chroma. `dor_pool_fingerprints_2026-04-24.summary.json:219-223` gives `recommended_variant = V3_combined` (backs RLP7_03). `lockbox_failure_contact_2026-04-24.index.json` shows `Cam_Test__s33` fakes at `frame_prob` 0.01–0.05 on pre-fix RLP6_04 — not threshold-reachable, per memory `project_signature_shortcut_finding.md`.

## Conclusions drawn in-session

- **Camera/ISP signature is the structural residual, not preprocessing drift.** The post-fix anchor delta of -0.008 ruled out the "WS-P0 alone descopes the packet" branch of the decision tree (`HANDOFF.md:17-19`).
- **Training-augmentation dominates; calibration is complementary.** 0.301 is at the ≤0.30 training-aug threshold (`calibration_probe_2026-04-24.summary.json:454-459`). Handoff ordering (`RLP7_05` first, then `_02`, then `_04`) reflects this: attack both axes, fall back to codec-only, hold spatial-only as the isolation read.
- **Launch priority is codec-aligned, not lighting-aligned.** The yellow-vs-white clean control stayed near 0 on the 2-camera data (`R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md:11-16, 104-105`), so `RLP7_01` and `RLP7_03` were demoted despite the fingerprint-diff recommending `V3_combined`.
- **`arena/model_arena.py:472` is the load-bearing INTER fix**, not `batch_inference_gcs.py:407` — the 0.94 Dor number came through the retro-score path (`R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md:31-37`).
- **Deploy-server preprocessing is untouched.** The original 0.94 came from `http://34.16.217.28:8999` (`HANDOFF.md:53`); WS-P0 does not address production. Separate workstream.
- **Packet-7 success criterion is per-camera FPR spread, not fake recall.** `HANDOFF.md:242-247`: lowest `max(per-identity real_fpr)` with `fake_recall ≥ RLP6_04 - 0.01` and `value_composite ≥ 0.89`. Fake-recall regression is the main failure mode to guard against.
- **Session IDs**: *(none indexed — session still live; convmem coverage pending. Traceability via `HANDOFF.md`, `R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md`, and commits `855871e`, `deac44e`, `c4affa1`, `584118e`, `5798b3d`, `0f797a1`.)*

## Retrospective (as of 2026-04-24)

- **Packet is not yet launched.** Five yamls authored (two new this session), four support workstreams shipped, pre-launch gate cleared. Next on-path action is the user's launch pick of `{RLP7_05, RLP7_02, RLP7_04}` vs the full five, gated by W&B credentials per `feedback_promotion_contract_launch.md`.
- **Structural response to the slot-07 shortcut.** Packet-5 slot-07 flipped on `dor_shkedi` vs `real_dor`; memory `project_signature_shortcut_finding.md` codified *"lockbox 90/90 is NOT threshold-reachable"* on RLP6_04. The 2026-04-24 2-camera test promoted that finding from curiosity to deployment-blocker. RLP7 is the training-side answer; [processing_signature_shortcut](../threads/processing_signature_shortcut.md) is the narrative spine.
- **Preprocessing-parity note (explicit).** This packet **contains** the parity fix (commit `855871e`). RLP6_04 scorecard numbers pre-date it; post-fix readout on the 6 Dor/Roee pools is `analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json`. Every RLP7 training checkpoint will be evaluated post-fix by construction. See [preprocessing_parity_bug](../threads/preprocessing_parity_bug.md).
- **Calibration is a live complement, not a throwaway.** The 0.301 ruled out "calibrate and ship", not "calibrate alongside". If an RLP7 checkpoint lands with narrower but nonzero pool FPR spread, per-camera τ recovers the residual without retraining. See [calibration_vs_training_aug](../threads/calibration_vs_training_aug.md).
- **`RLP7_04` vs `RLP7_05` is the informative contrast** (`R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md:94-101`). `RLP7_05 ≥ RLP7_04` ⇒ codec complements spatial; `RLP7_05 ≈ RLP7_04` ⇒ spatial alone carries the break.
- **Two failed approaches recorded (`HANDOFF.md:59-63`).** (i) Initial WS-P1 probe at single 5% target FPR with 15 test frames/pool — FPR quantized to 1/15, `gap_closure=0.000`; fixed by averaging over {5,10,15,25}%. (ii) Parity-test regex broke on nested parens in `cv2.resize(img, (self.resolution, self.resolution), ...)`; fixed by line-by-line scan.
- **Open workstreams carried forward.** WS-P2.a (stress-variant retro-score suites, reusing `pipelines.py:1702-1738`) and WS-P3 (lockbox-scale fingerprint diagnostics, 2–3 day GCS fetch) are planned, not started (`HANDOFF.md:51-52`). WS-P4.b/c gated on WS-P3 axis ranking.
- **Cross-reference.** Story continues in [RLP6](RLP6.md) (reinterpreted parent), [processing_signature_shortcut](../threads/processing_signature_shortcut.md), [preprocessing_parity_bug](../threads/preprocessing_parity_bug.md), [calibration_vs_training_aug](../threads/calibration_vs_training_aug.md), [gate_alignment_story](../threads/gate_alignment_story.md).

## Source files

- **Handoffs**:
  - `docs/relaunch_handoffs/R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md` — **primary source.** 2-camera evidence :5-19, WS-P0 :26-37, WS-P1 :40-48, WS-P2.b :50-62, WS-P4.a :65-68, RLP7_04/_05 :69-79, launch subset :92-106, decision points :122-130.
  - `HANDOFF.md` — pre-launch gate state. TL;DR :17-26, completed :30-40, not-yet-done :45-54, failed approaches :59-63, key decisions :69-76, files-to-know :94-114, resume :164-247.
  - `docs/relaunch_handoffs/R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md` — upstream plan behind RLP6_04.
- **Yamls** (all in `experiments/phase2_round13/`): `R13_RLP7_01_lighting_aggressive.yaml`, `_02_codec_aggressive.yaml`, `_03_combined.yaml` (all seed 742); `_04_teams_spatial_only.yaml` (seed 744, new, commit `c4affa1`); `_05_teams_spatial_plus_codec.yaml` (seed 745, new, commit `c4affa1`). Two exploratory siblings `_06_teams_cct_only.yaml` and `_07_teams_spatial_plus_codec_plus_cct.yaml` exist but are not in the handoff subset.
- **Code touchpoints**: `batch_inference_gcs.py:407` + `arena/model_arena.py:472` (WS-P0 `INTER_LINEAR` fix; second is load-bearing). `data/sources/combined_paired.py:3455` (training reference). `data/sources/combined_paired.py:4772` (override preset-key filter). `data/augmentations/pipelines.py:787` (`_TEAMS_PASSTHROUGH_DEFAULTS`), `:1086` (`_build_teams_passthrough_special_block`; `aug_enabled` must be true at :1088), `:1386` (`_build_teams_passthrough_pipeline`). `arena/postprocess_per_identity.py` (WS-P2.b CLI). `analysis/calibration_probe_2026-04-24.py` (WS-P1, seed 42). `tests/test_inference_train_preprocessing_parity.py`, `tests/test_postprocess_per_identity.py`.
- **Scorecards / analysis**: `analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json` (pre-launch gate data), `.per_frame.csv`; `analysis/calibration_probe_2026-04-24.summary.json` (0.301); `analysis/dor_baseline_repro_2026-04-24.summary.json`; `analysis/dor_pool_fingerprints_2026-04-24.summary.json` (V3_combined verdict); `analysis/dor_roee_combined_2026-04-24.summary.json`; `analysis/lockbox_failure_contact_2026-04-24.index.json`.
- **Reference data**: `/tmp/dor_roee_combined_2026-04-24/combined_frame_tags.json` (6 tags × 30 frames); `gs://real-teams-dor-roee/session_20260424_combined_tags_121458_121007/`; RLP6_04 ckpt at `gs://training-job-outputs/phase2r13_experiments/h2pdu6i5/value_composite_effort_20260424_step23500_auc0.9942_eer0.0169.pth`.
- **Commits (branch `teams-relaunch-root-2026-04-17`)**: `855871e` (WS-P0 INTER_LINEAR fix), `deac44e` (WS-P1 probe + WS-P2.b reducer), `c4affa1` (RLP7_04/05 yamls), `584118e` (Packet-7 handoff), `5798b3d` (HANDOFF.md refresh), `0f797a1` (track RLP7_01/02/03 yamls).
- **Memory pointers**: `project_signature_shortcut_finding.md` (lockbox 90/90 not threshold-reachable; RLP7 is the structural response); `project_promotion_contract.md` (dev-calibrated τ + lockbox is the deployment readout, not `value_composite`); `feedback_promotion_contract_launch.md` (launcher needs `WANDB_*` env vars); `reference_image_rebuild.md` (new in-image yamls require `./dev.sh build-prod -y`); `feedback_decision_points.md` (launch subset is a user call).
