# OPEN_LOOPS — Mechanically Generated Issue Inventory

> ⚙ **GENERATED on 2026-05-05** by `tools/regenerate_open_loops.py`. Do not hand-edit. To change an entry, edit the corresponding `### Open loop:` block in the **owning thread file** under `threads/` and re-run the script.

## Summary

- **Open**: 27
- **In progress**: 4
- **Resolved**: 8
- **Superseded**: 1

## Entries

### Open (27)

### `contract-policy-bug-fix-not-committed`
- **status**: open
- **severity**: high
- **first_seen**: 2026-04-23
- **last_verified**: 2026-04-30
- **close_criterion**: the recall-floor + budget-aware τ-selection patch (working-tree diff: `arena/score_teams_promotion_contract.py +98`, `arena/run_target_domain_validation_sequential.py +19`, `tests/test_score_teams_promotion_contract.py +125`, total +207 net lines as of 2026-04-29) is committed to `teams-relaunch-root-2026-04-17`, the image is rebuilt with the fix in (`./dev.sh build-prod -y` auto-bumps VERSION patch), and a contract scorecard run on a representative recent checkpoint (e.g. P8A reference step 5000) is invoked with `--promotion_target_fake_recall_min 0.30` AND the resulting scorecard is documented as having selected τ via the recall-floor path (not via the legacy no-budget minimize-FPR-only path). All three commit + image + verified-readout components are required. **2026-04-30 evening update**: the `mclioexb` scorecard run did NOT exercise the v3 fix — the launcher (`arena/launch_teams_promotion_contract.sh`) omitted `--promotion_target_fake_recall_min`, so the scorecard ran the DEFAULT policy (τ=0.975, recall=0.13). Component 3 still NOT MET. The verdict was independently meaningful (model fails contract under either policy — see [`promotion_contract_evolution`](promotion_contract_evolution.md) "2026-04-30 evening" subsection) but does not close this loop.
- **source**: `threads/contract_policy_bug.md:108`

### `apply-svd-in-proj-attribution-revision-needed`
- **status**: open
- **severity**: high
- **first_seen**: 2026-04-26
- **last_verified**: 2026-04-29
- **close_criterion**: every prior R12g / RLP / P-* run with `apply_svd_to_in_proj: true` is either (a) re-trained from a matched fork point with the fixed routing and the deltas vs broken-bug counterpart documented in a per-packet table, or (b) explicitly annotated in its packet retro with "in_proj-SVD trained on zero classification gradient — q/k/v residuals contributed only via regularizer drift." Until one of these is done the published attribution claims (P8A breakthrough = "more reach", P7 ceiling = "FT-only structurally bounded with all levers on") are imprecise.
- **source**: `threads/in_proj_svd_gradient_bug.md:53`

### `face-size-label-leak`
- **status**: open
- **severity**: high
- **first_seen**: 2026-04-27
- **last_verified**: 2026-04-30
- **close_criterion**: a downstream packet demonstrates that with a face-size-targeted intervention actually live (face-scale-jitter, symmetric crop-aug, bucket-balanced sampling, or a structurally-different lever yet to be designed), per-method face-pixel-area distributions overlap (Cohen's d on dev_fake vs dev_real ≤ 0.10), AND a tightness-sweep at inference flips ≤ 10% of frame predictions on the production-honest cache — i.e., the leak is no longer exploitable at the size of effect that the 04-27 sweep demonstrated. As of 2026-04-30 afternoon the flip-rate half is affirmatively NOT MET on the strongest available leader (`mclioexb` jitter@0.50: 43.9% on 180 frames; P8A baseline 39.4% on the same 180 frames).
- **source**: `threads/face_size_label_leak.md:124`

### `eval-production-crop-tightness-mismatch`
- **status**: open
- **severity**: high
- **first_seen**: 2026-04-29
- **last_verified**: 2026-04-30
- **close_criterion**: a written disposition is recorded — either (a) the crop-tightness delta between the eval substrate and production deployment crops is quantified (e.g., distributions of face_area_fraction or background_to_face ratio measured on a representative sample of eval frames vs production captures, with median/p90 of the delta) AND a rule is documented for translating eval FPR to production FPR (or for re-cropping eval at production tightness), OR (b) a packet retrains/re-evaluates with eval re-cropped at production tightness and the resulting headline FPR / recall numbers are recorded as the deployment-relevant readout, replacing the looser-crop substrate as the canonical reporting surface
- **source**: `threads/eval_production_crop_tightness_gap.md:88`

### `sharpness-metric-computed-on-full-image-not-face`
- **status**: open
- **severity**: high
- **first_seen**: 2026-04-29
- **last_verified**: 2026-04-29
- **close_criterion**: `analysis/lockbox_tagging/layers/quality.py:82` is updated to compute the laplacian on a face-crop region (using the existing face-geometry layer's bbox tags), the parquet `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` (or its successor) is re-tagged with the corrected metric, AND any downstream FPR-by-quartile report that depends on `sharpness_laplacian` is either rerun under the corrected metric OR explicitly annotated as "indexed on full-image laplacian, not face laplacian — read as a confounded U-shape". The 2026-04-27 investigation's *"very sharp (>402): 45% FPR"* table entry is the canonical downstream report; flagging it covers the most-cited claim.
- **source**: `threads/sharpness_metric_bug.md:69`

### `split-mode-delta-unquantified`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-04-21
- **last_verified**: 2026-04-29
- **close_criterion**: a within-packet ablation produces a numeric bound on the AUC delta attributable to `shuffle → hash_stable` alone (e.g., re-run `RLP1_01` recipe with `hash_stable` on the same data snapshot, or re-run `RLP2_01` recipe with `shuffle` and measure the difference under matched mutable-source state)
- **source**: `threads/identity_split_mode.md:43`

### `deploy-server-preprocessing-drift`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-04-24
- **last_verified**: 2026-04-29
- **close_criterion**: the deployment server at `http://34.16.217.28:8999` (or its successor) is verified through a `tests/test_inference_train_preprocessing_parity`-style guard to use `cv2.INTER_LINEAR` (matching `combined_paired.py:3455`), and a smoke probe through the production endpoint reproduces the post-fix anchor-pool number within ±0.02 of `analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json`
- **source**: `threads/preprocessing_parity_bug.md:61`

### `residual-70-pct-fpr-gap-no-lever-past-rlp7`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-04-24
- **last_verified**: 2026-04-29
- **close_criterion**: a Packet-7+ readout reports per-pool FPR spread on the 6 Dor/Roee pools (anchor `dor-real-webcam-false-flag-no-virtual-bg` ≤ 0.30 + clean pools ≤ 0.05) — i.e., the training-aug + calibration combination demonstrably closes ≥70% of the cross-pool FPR gap; OR a documented next-lever proposal (representation loss, data-side intervention, architectural change) is filed for the residual
- **source**: `threads/calibration_vs_training_aug.md:73`

### `in-proj-svd-residual-capacity-not-exercised`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-04-26
- **last_verified**: 2026-04-29
- **close_criterion**: a recipe (yaml or code change) is identified that produces a measurable lift on a deployment-relevant metric (anchor pool composite, lockbox real-FPR, or per-method recall) when `apply_svd_to_in_proj: true` is enabled vs disabled at matched fork point + LR + schedule + data — i.e., the lever is shown to be live in some configuration, not just live in principle.
- **source**: `threads/in_proj_svd_gradient_bug.md:62`

### `wandb-side-surface-hygiene-not-systematic`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-04-26
- **last_verified**: 2026-04-29
- **close_criterion**: a single mechanism (test, lint, schema, or shared helper) catches both (a) silent-fallback config flattening AND (b) artifact-name validation issues at submission time before they fire in production — i.e., the wandb-side surface is hardened wholesale rather than fix-by-fix
- **source**: `threads/wandb_yaml_propagation_bugs.md:98`

### `webcam-mode-fpr-dominance-headline-misleading`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-04-27
- **last_verified**: 2026-04-29
- **close_criterion**: the promotion contract scorecard (`arena/score_teams_promotion_contract.py` or its successor contract yaml) reports v2-filtered lockbox FPR as a first-class metric alongside baseline lockbox FPR — i.e., a candidate that has 5% baseline lockbox FPR but 0.7% v2 FPR is read as "deployment-FPR ≤ 5%" by the contract, not the other way around. Includes per-identity breakdown to surface the dor_shkedi-skew caveat as a hard sub-gate (e.g. "no single v2 identity has FPR > 30%" — the catastrophic outlier filter).
- **source**: `threads/webcam_fpr_dominance.md:109`

### `silent-feature-failures-pattern`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-04-28
- **last_verified**: 2026-04-29
- **close_criterion**: a single pre-launch CI / smoke / lint step verifies that every yaml-declared trainer feature actually emits an "ENABLED" log line at trainer init — i.e., for every nested-dict block in the launched yaml that the trainer reads via `self.config.get('<block>')`, a runtime check fails fast (and a CI test fails offline) if the block is silently fall-through-disabled
- **source**: `threads/wandb_flattening.md:69`

### `frame-level-vs-clip-level-scorer-mismatch`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-04-29
- **last_verified**: 2026-04-29
- **close_criterion**: the contract scorer's clip-level recall numbers are reconciled with the frame-level AUCs from cached predictions — either (a) the corrected contract policy with `target_fake_recall_min=0.30` produces clip-level recalls within ~5pp of `(1 - threshold-implied-FNR)` from the frame-level distribution, OR (b) the contract scorer is documented as measuring something genuinely different from the frame-level signal (e.g. clip-aggregation thresholds, video-level voting policy) and the headline reporting is normalized so future agents do not compare clip-level recall to AUC-implied recall.
- **source**: `threads/viso_bucket_gap.md:138`

### `shortcut-block-criterion-may-be-scorer-artifact-bound`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-04-29
- **last_verified**: 2026-04-29
- **close_criterion**: a written disposition is recorded — either (a) the `shortcut-deployment-block` close criterion's `lockbox_fake_recall ≥ 0.60` half is reaffirmed under the corrected contract scorer policy (`target_fake_recall_min=0.30`) and a downstream packet hits both halves at the same τ, OR (b) the criterion is explicitly revised to a frame-level-AUC-based version (e.g. `dor-real-webcam-false-flag-no-virtual-bg` mean prob ≤ 0.30 AND `visomaster_enhanced_macro_dev` frame-level AUC ≥ 0.85) with a documented rationale referencing memory `project_p8a_frame_level_auc_2026-04-29.md`. Surfaced as a sub-question to the parent `shortcut-deployment-block` (critical) loop, not a replacement for it.
- **source**: `threads/processing_signature_shortcut.md:164`

### `source-image-resolution-floor-not-applied-to-eval`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-04-29
- **last_verified**: 2026-04-29
- **close_criterion**: a source-image-resolution minimum (e.g., `min(width, height) >= 200`) is applied as an eval-scope bound, the headline FPR / recall numbers on at least one canonical checkpoint (P8A_step5000) are rerun under the bound, AND the bound is either (a) added to the modern_lockbox_v2 filter set as a permanent piece of the v2 definition, OR (b) reported as a side-by-side column alongside v2 (so a reader can see both the v2 number and the v2-plus-resolution-floor number).
- **source**: `threads/eval_substrate_data_hygiene.md:65`

### `very-sharp-fp-slice-mixes-two-populations`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-04-29
- **last_verified**: 2026-04-29
- **close_criterion**: the very-sharp-FP slice (n=421 in the 2026-04-27 substrate) is re-tagged with face-crop laplacian and re-partitioned into the two populations the audit identified — (a) genuinely sharp faces in sharp environments, (b) soft faces in tiny / busy / compressed crops where non-face content drives the laplacian; per-population FPR is reported separately; the wiki narrative around the very-sharp-FP slice is updated to reflect the split.
- **source**: `threads/sharpness_metric_bug.md:78`

### `anti-shortcut-bundle-needs-single-lever-discipline`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-04-30
- **last_verified**: 2026-04-30
- **close_criterion**: the next anti-shortcut packet that stacks more than one intervention is structured with at least one single-lever ablation slot (the strongest lever alone, FT init + data + LR fixed) AND the packet retro records whether the bundle is net-additive vs the single-lever baseline. The discipline either becomes a `BUILD_SCAFFOLD.md`-style operational rule for future packets, OR a counter-example (a stacked bundle that demonstrably beats its single-strongest component on a deployment-relevant axis) is filed and this loop is resolved as superseded.
- **source**: `threads/anti_shortcut_bundle_decomposition.md:60`

### `corrected-val-test-hint-split-counts-unrecoverable`
- **status**: open
- **severity**: low
- **first_seen**: 2026-04-17
- **last_verified**: 2026-04-29
- **close_criterion**: the corrected `val`/`test` hint-split counts (`84 + 174 + 27` reconstructable rows in `WT-A_policy_truth_artifact_2026-04-17.json:117-122`) get explicit splits
- **source**: `threads/gate_alignment_story.md:82`

### `rlp3-retro-score-systematic-drift`
- **status**: open
- **severity**: low
- **first_seen**: 2026-04-22
- **last_verified**: 2026-04-29
- **close_criterion**: a root cause for the `−0.0067` retro-vs-training drift (vs the ≤1e-3 spec) is identified — either pinned to one of the candidate sources (OOD sampler / identity-split / frame-order non-determinism, augmentation RNG, GPU kernel) or the retro pipeline is reproducible to within `1e-3` on the sanity slot
- **source**: `threads/promotion_contract_evolution.md:179`

### `value-composite-cross-packet-comparability`
- **status**: open
- **severity**: low
- **first_seen**: 2026-04-22
- **last_verified**: 2026-04-29
- **close_criterion**: a per-packet table records which `value_composite` definition (gate thresholds, stability aggregator, real-pool set, fake-pool set) was active for the leader checkpoint, so that absolute composite numbers are quotable with provenance
- **source**: `threads/value_composite_semantics.md:74`

### `per-identity-reducer-not-a-contract-gate`
- **status**: open
- **severity**: low
- **first_seen**: 2026-04-24
- **last_verified**: 2026-04-29
- **close_criterion**: `arena/score_teams_promotion_contract.py` (or its successor contract yaml) hard-gates on `max(per-identity real_fpr) ≤ <budget>` rather than admitting it as a sort key only — i.e., a checkpoint that has 95% mean real-pool TPR but a single-identity 50% FPR fails the contract by construction
- **source**: `threads/calibration_vs_training_aug.md:82`

### `prior-leader-rescore-sweep`
- **status**: open
- **severity**: low
- **first_seen**: 2026-04-24
- **last_verified**: 2026-04-29
- **close_criterion**: every pre-fix leader (RLP1_01, RLP2_02, RLP3_05, RLP3.5_02, RLP5_07, RLP6_04 — RLP6_04 is the only one already done) is re-scored through the `INTER_LINEAR` path on a matched eval suite, and the resulting post-fix ranking is documented in `analysis/preprocessing_parity_post_fix_sweep_*.summary.json` (or equivalent)
- **source**: `threads/preprocessing_parity_bug.md:70`

### `ws-p1-rerun-on-post-fix-inputs`
- **status**: open
- **severity**: low
- **first_seen**: 2026-04-24
- **last_verified**: 2026-04-29
- **close_criterion**: WS-P1 (`analysis/calibration_probe_2026-04-24.py`) is re-run on the post-INTER_LINEAR re-scored RLP6_04 frames, and the resulting `avg_gap_closure_across_target_fprs` is recorded — confirming or shifting the 0.301 verdict that placed per-camera calibration on the training-aug side of the boundary
- **source**: `threads/preprocessing_parity_bug.md:79`

### `is-no-face-slice-is-data-degeneracy`
- **status**: open
- **severity**: low
- **first_seen**: 2026-04-29
- **last_verified**: 2026-04-29
- **close_criterion**: the 219 is_no_face frames (174 real + 45 fake) are removed or relocated from any eval substrate that informs FPR/recall/FN-lift conclusions, AND the 2026-04-27 investigation's *"small but high-lift, FN lift 2.09×"* reading on this slice is documented as data-degeneracy (correct MediaPipe verdict — the frames don't contain face content the model could reasonably score) rather than model weakness. Headline FPR numbers will not move materially (n=219 of ~7,334) but the interpretation of "the model misses fakes when no face is present" changes. Documenting the frames-removed delta on at least one canonical checkpoint (P8A_step5000) makes the close criterion auditable.
- **source**: `threads/eval_substrate_data_hygiene.md:56`

### `bundle-failure-mode-attribution-revision`
- **status**: open
- **severity**: low
- **first_seen**: 2026-04-30
- **last_verified**: 2026-04-30
- **close_criterion**: the P13 retro's γ-verdict attribution ("anti-shortcut interventions on a from-scratch substrate are insufficient") is either reaffirmed by a from-scratch single-lever-jitter-only run, OR explicitly revised in the P13 retro to "anti-shortcut bundle as composed in P13 was insufficient on from-scratch; whether the single-lever jitter@0.50 would have been enough is unknown without re-running."
- **source**: `threads/anti_shortcut_bundle_decomposition.md:69`

### `jitter-on-training-substrate-not-yet-tested`
- **status**: open
- **severity**: low
- **first_seen**: 2026-04-30
- **last_verified**: 2026-04-30
- **close_criterion**: the face-size invariance sweep is re-run on a frame substrate stratified across the training face-pixel-area distribution (e.g., 200 frames sampled from `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` with stratification by face-area quartile), with the same 5-tightness grid and same 4 ckpts as the 04-30 afternoon run; if `mclioexb` flip rate ≤ 10% on the training-distribution substrate while still being > 10% on the production-honest substrate, mechanism (3) (test-substrate mismatch) is supported
- **source**: `threads/jitter_winner_mechanism_unknown.md:122`

### `jitter-winner-mechanism-unknown`
- **status**: open
- **severity**: low
- **first_seen**: 2026-04-30
- **last_verified**: 2026-04-30
- **close_criterion**: at least one of the three candidate mechanisms (decision-boundary / intermediate-layer / test-substrate) is empirically supported on a probe whose design isolates that mechanism, AND the supported mechanism produces a measurable signature on `mclioexb` that does NOT also appear on `9lmvb5b4` step 5000 baseline at comparable magnitude — i.e., the mechanism is specific to the value_composite winner, not a feature shared with the FT base
- **source**: `threads/jitter_winner_mechanism_unknown.md:113`

### In progress (4)

### `shortcut-deployment-block`
- **status**: in-progress
- **severity**: critical
- **first_seen**: 2026-04-24
- **last_verified**: 2026-04-29
- **close_criterion**: a Packet-7+ checkpoint achieves `dor-real-webcam-false-flag-no-virtual-bg` mean prob_fake ≤ 0.30 (vs RLP6_04's 0.932 post-fix) AND `lockbox_fake_recall ≥ 0.60` at a τ that holds `teams_ood_real` FPR ≤ 5%, demonstrating the camera/ISP shortcut has been broken without sacrificing fake recall
- **source**: `threads/processing_signature_shortcut.md:175`

### `fpr-minimization-no-budget-tau-collapse`
- **status**: in-progress
- **severity**: high
- **first_seen**: 2026-04-17
- **last_verified**: 2026-04-30
- **close_criterion**: the `score_teams_promotion_contract.py` runner enforces a recall floor or τ ceiling that prevents `selected_threshold ≈ 0.995` configurations passing silently
- **source**: `threads/promotion_contract_evolution.md:160`

### `data-axis-clean-single-lever-retest-in-progress`
- **status**: in-progress
- **severity**: high
- **first_seen**: 2026-05-04
- **last_verified**: 2026-05-05
- **close_criterion**: Packet A (`R13_PA_VISOMASTER_ENHANCED_DATA.yaml`, run `3140330896851206144`, JOB_STATE_RUNNING in us-east1) and/or Packet C-codec (`R13_PC_CODEC_PLUS_DATA.yaml`, run `1202657157175050240`, JOB_STATE_RUNNING in us-east1) complete training, the resulting checkpoints are scored under both the F0 (full eval substrate) and F4 (substrate-cleaning, `analysis/substrate_cleaning_eval_2026-05-05/`) lenses, and a documented verdict is recorded for whether enabling `visomaster_enhanced + visomaster_teams_enhanced` data sources at fw=4.0 lifts visomaster_enhanced_macro_dev recall above the E2B_3200 baseline (8.4% F0 / 30.9% F4) at deployment-honest single-τ at 5% FPR ceiling. The verdict closes EITHER as (a) data-axis lever is dispositive (Packet A/C-codec materially beats E2B on viso recall under deployment policy), OR (b) data-axis lever as cleanly tested still does not lift, in which case memory `project_data_axis_lever_pulled_twice_no_lift.md` is amended to "pulled three times" with the bundle-confound caveat lifted from the prior two attempts. Either outcome closes the loop.
- **source**: `threads/viso_bucket_gap.md:127`

### `enhanced-vs-unenhanced-val-pool-confound`
- **status**: in-progress
- **severity**: medium
- **first_seen**: 2026-04-21
- **last_verified**: 2026-04-29
- **close_criterion**: a packet trains an enhanced-proper arm and measures it against a validation pool that explicitly contains enhanced-proper rows, with dose matched to the unenhanced arm — so "enhanced hurts" / "enhanced helps" is readable without the pool-composition + dose confound
- **source**: `threads/gate_alignment_story.md:91`

### Resolved (8)

### `hints-as-supervision-hypothesis`
- **status**: resolved
- **severity**: medium
- **first_seen**: 2026-04-17
- **last_verified**: 2026-04-29
- **close_criterion**: an explicit ablation produces evidence whether the retained `visomaster_hints` / `visomaster_hints_teams` rows help, hurt, or are neutral as training signal
- **source**: `threads/weak_signal_hints_track.md:59`

### `wt-a-tracked-source-integration-gap`
- **status**: resolved
- **severity**: medium
- **first_seen**: 2026-04-17
- **last_verified**: 2026-04-29
- **close_criterion**: a runtime training source consumes the April-17 policy manifest and is part of the tracked Git tree on `teams-relaunch-root-2026-04-17`
- **source**: `threads/gate_alignment_story.md:73`

### `wt-b-april-17-source-integration`
- **status**: resolved
- **severity**: medium
- **first_seen**: 2026-04-17
- **last_verified**: 2026-04-29
- **close_criterion**: WT-B has a runnable training package (not just draft-only YAMLs) that consumes the April-17 policy and a smoke-passed launcher
- **source**: `threads/weak_signal_hints_track.md:50`

### `value-composite-semantics-undefined`
- **status**: resolved
- **severity**: medium
- **first_seen**: 2026-04-21
- **last_verified**: 2026-04-29
- **close_criterion**: a doc-level statement explicitly defines `value_composite` as a trainer-side directional metric, names what it can and cannot detect, and codifies the operational rule that promotion decisions go through the lockbox-anchored contract scorecard rather than the trainer composite
- **source**: `threads/value_composite_semantics.md:65`

### `rlp4-image-rebuild-discipline-forged`
- **status**: resolved
- **severity**: medium
- **first_seen**: 2026-04-22
- **last_verified**: 2026-04-29
- **close_criterion**: a packet successfully launches yaml-only or yaml+code changes after a deliberate `./dev.sh build-prod -y` rebuild + canary-first pattern (no untracked-yaml FileNotFoundError class repeats)
- **source**: `threads/image_rebuild_discipline.md:58`

### `train-sweep-allowlist-not-closed-under-schema-changes`
- **status**: resolved
- **severity**: medium
- **first_seen**: 2026-04-22
- **last_verified**: 2026-04-29
- **close_criterion**: every new nested dict in yaml requires explicit re-application in train_sweep.py allowlist (~lines 176-282) — recurrence indicates pattern not generalized.
- **source**: `threads/wandb_yaml_propagation_bugs.md:85`

### `wt-e-promotion-winner-deferred`
- **status**: resolved
- **severity**: low
- **first_seen**: 2026-04-17
- **last_verified**: 2026-04-29
- **close_criterion**: at least one shortlist run produces a written `promotion_winner.json`/`checkpoint_summary.csv` artifact pair on the WT-E manifest+shortlist
- **source**: `threads/promotion_contract_evolution.md:151`

### `pair-loss-asymmetric-variant-untested`
- **status**: resolved
- **severity**: low
- **first_seen**: 2026-05-04
- **last_verified**: 2026-05-04
- **close_criterion**: either (a) a $5 GPU probe extracts E2B features on the same 550 (raw, teams) viso pairs and re-runs Q1/Q2 directly on E2B geometry, with documented verdict on whether the P8A-as-proxy assumption hides a real signal; OR (b) an asymmetric-pair-loss variant (only aligning teams toward raw when raw scores higher than teams — never the reverse) is scoped, the cohort math is recomputed under that asymmetry, and a go/no-go decision is documented; OR (c) the loop is explicitly closed-as-not-pursued with a one-line note that the symmetric pair loss verdict is dispositive enough to deprioritize the asymmetric variant given the 9pp upper-bound ceiling and the existence of cheaper alternatives (substrate cleaning, per-substrate τ-calibration).
- **source**: `threads/clean_teams_identity_pairing.md:93`

### Superseded (1)

### `p14-data-fix-not-launched`
- **status**: superseded
- **severity**: medium
- **first_seen**: 2026-04-29
- **last_verified**: 2026-04-30
- **close_criterion**: `R13_P14_DATA_FIX.yaml` is committed, the smoke loads enhanced viso samples cleanly, the retrain completes, and a contract scorecard run on the resulting checkpoint shows `visomaster_enhanced_macro_dev` recall lifted materially above the P8A baseline (≥ 24% target under the corrected contract policy with `target_fake_recall_min=0.30`) — verifying the bucket-gap closure on the actual headline metric.
- **source**: `threads/viso_bucket_gap.md:64`

