# OPEN_LOOPS — Mechanically Generated Issue Inventory

> ⚙ **GENERATED on 2026-05-22** by `tools/regenerate_open_loops.py`. Do not hand-edit. To change an entry, edit the corresponding `### Open loop:` block in the **owning thread file** under `threads/` and re-run the script.

## Summary

- **Open**: 58
- **In progress**: 6
- **Resolved**: 17
- **Superseded**: 1
- **Unknown status**: 1

## Entries

### Open (58)

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

### `deployment-vs-p8a-substrate-tradeoff-not-quantified`
- **status**: open
- **severity**: high
- **first_seen**: 2026-05-06
- **last_verified**: 2026-05-06
- **close_criterion**: a written quantification of (a) the populations on which P8A would have lower real-side FPR than the currently-deployed E2B at matched τ — quantified across the identity_browser-style substrates (xinhe_may6_falseflag, dor_evening, dor_morning, extra_roy_d, the chronic-6 set, etc.) and the production live-fake suites — AND (b) the populations on which E2B would have higher fake recall than P8A at matched τ. The disposition then explicitly chooses the deployment model with the trade-off documented (memory `project_promotion_contract.md` is the contract surface for the promotion decision; this loop is the deployment-side disposition that depends on which failure cost the operator weights higher). Until the disposition is recorded, the user has the standing option of swapping deployment to P8A on substrate-classes where E2B is dispositively false-flagging (xinhe_may6_falseflag is one such confirmed substrate).
- **source**: `threads/processing_signature_shortcut.md:257`

### `diagnostic-substrates-not-in-contract`
- **status**: open
- **severity**: high
- **first_seen**: 2026-05-07
- **last_verified**: 2026-05-07
- **close_criterion**: either (a) the canonical contract suite manifest is extended with the 9 diagnostic substrates listed in this thread (`xinhe_may6_falseflag`, `live_*_teams_prod`, `dor_evening`/`dor_morning`, `team_sanity_may5`, `dor_fake_local`, `extra`, `visomaster_v2_dor`) so the GPU scorecard scores them automatically, OR (b) a `verdict_template.md` is authored that mandates a `Phase A.5 — diagnostic substrates` step in every packet retro and is referenced by `AGENTS.md` as a pre-launch checklist item. Either path closes the loop; (a) is the more durable fix.
- **source**: `threads/eval_substrate_layering.md:64`

### `eval-manifests-version-pinning`
- **status**: open
- **severity**: high
- **first_seen**: 2026-05-07
- **last_verified**: 2026-05-07
- **close_criterion**: every eval manifest under `arena/manifests/` is either (a) committed with a date-pinned filename (`..._wave_<YYYY-MM-DD>.json` or `..._wave_<YYYY-MM-DD>_v2.json`) so cross-version comparisons are explicit, OR (b) the manifest filename embeds a content-hash that the scorer/runner records alongside its results so a stale comparison is auto-flagged. A pre-launch lint at `tools/lint/preflight_launch.sh` fails the launch if `git status arena/manifests/` is non-empty.
- **source**: `threads/eval_substrate_layering.md:73`

### `f2-not-testable-from-phase-a`
- **status**: open
- **severity**: high
- **first_seen**: 2026-05-07
- **last_verified**: 2026-05-07
- **close_criterion**: either (a) the F2(a) close criterion ("≥30% relative pair-rank lift on ≥2 of 6 paired training lanes among previously-missed fakes") is rewritten as a Phase-A-evaluable form (e.g., "lift on per-frame pair gaps where P8A is confident-wrong" — uses P8A's own score as the gating condition, doesn't require the training-loader's `(sample_id, frame_idx)` pair structure), OR (b) the eval substrate is extended with score sets on each of the 6 paired training lanes (`df40`, `deeplive`, `visomaster_v1_base`, `visomaster_enhanced`, `visomaster_teams_enhanced`, `deeplive_teams`) so the criterion as written becomes testable, OR (c) the criterion is dropped from packet retros' close-criterion table with a written disposition referencing this loop.
- **source**: `threads/eval_substrate_layering.md:104`

### `grouped-manifest-v2-stale-paths`
- **status**: open
- **severity**: high
- **first_seen**: 2026-05-07
- **last_verified**: 2026-05-07
- **close_criterion**: `analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv` is regenerated against current GCS state (bucket layout has migrated to `session_<timestamp>/...` for `live-fakes-teams-prod` and the `roee_tester_real_2026-03-24/` folder no longer exists), AND every row whose `frame_path` starts with `gs://local/...` is either (a) marked with an explicit `is_local: True` flag so scoring scripts can branch (download from local mirror or skip), OR (b) re-pointed to a real GCS URI. Phase A.5 verifies by re-running the broken suites (`live_reals_teams_prod`, `dor_evening`, `dor_morning`, `dor_fake_local`, `extra` — 2,768 frames total) without zero-tensor decode failures.
- **source**: `threads/eval_substrate_layering.md:82`

### `pair-rank-non-paired-lane-collateral`
- **status**: open
- **severity**: high
- **first_seen**: 2026-05-07
- **last_verified**: 2026-05-07
- **close_criterion**: a follow-up CPU or training experiment establishes whether pair_rank_loss is the specific lever responsible for non-paired-lane real-side regressions. Either: (a) a from-scratch CLIP+pair_rank run reproduces the Roy_D-class regression on its own substrate (confirms pair_rank), OR (b) a controlled FT-from-P8A run with pair_rank disabled reaches similar fake recall without the regression (refutes pair_rank as the exclusive cause).
- **source**: `threads/pair_rank_collateral.md:52`

### `t3-step2500-deployment-decision-pending`
- **status**: open
- **severity**: high
- **first_seen**: 2026-05-10
- **last_verified**: 2026-05-10
- **close_criterion**: a written deployment-direction decision is recorded that picks ONE of (a) ship T3_SLOT1_PERIODIC_STEP2500 with FPR-calibrated τ on a production-realistic real cohort (lockbox / F4-cleaned / IQ-pre-gated reals — depending on production substrate match), then optionally launch a refinement packet (T4 — face_scale_jitter @0.50 stacked with Slot 1 keep-list lever, OR Roy_D hard-negative mining); (b) ship T3_SLOT1_PERIODIC_STEP1500 instead (passes F0 strict floor; F4 viso 73.3% vs step2500 79.3% — slightly weaker capability but stricter contract pass); (c) keep P8A as deployment, treat T3 as a research result and pursue T4 with an additional viso-targeted intervention before next deployment cycle. The decision should be backed by an explicit determination of whether production substrate matches v2 (T3 wins), HDTF teams (P8A wins), or HDTF clean (T3 marginally wins) — see memory `project_v2_substrate_is_dor_diverse_swap` for prior framing of the v2-substrate-specificity question.
- **source**: `threads/viso_bucket_gap.md:265`

### `dev-to-lockbox-substrate-transfer-gap`
- **status**: open
- **severity**: high
- **first_seen**: 2026-05-11
- **last_verified**: 2026-05-12
- **close_criterion**: EITHER (a) a head-retrain on a substrate-diverse pool (dev + lockbox-style data, properly held-out for eval) lifts T4 lockbox AUC ≥ 0.90 — confirms the gap is closeable by exposing the head to substrate-diverse training, OR (b) the gap is REPRODUCED on multiple ckpts (T4, P8A, T3) and on the full lockbox cohort — closes as "structural data gap; requires new training data ingestion, not architecture/loss changes"
- **source**: `threads/iq_shortcut_deconvolution_program_2026-05-08.md:756`

### `roy-d-specific-anchor-pool-packet`
- **status**: open
- **severity**: high
- **first_seen**: 2026-05-16
- **last_verified**: 2026-05-16
- **close_criterion**: launch a packet with Roy_D anchor pool (30-50 frames) + dor anchor pool combined. If Roy_D dev FPR drops below 30%, mechanism generalizes. If not, Roy_D encoder region needs a different intervention class.
- **source**: `threads/iq_shortcut_deconvolution_program_2026-05-08.md:1294`

### `slot-b-real-rebalance-via-vcd-reaches-roy-d-region`
- **status**: open
- **severity**: high
- **first_seen**: 2026-05-16
- **last_verified**: 2026-05-16
- **close_criterion**: scorecard outcome on Slot B step3500 against the auto_mode_2026-05-16 checkpoint map. CONFIRMER: lockbox_real_fpr ≤ 0.03 AND dev_fake_macro_recall ≥ 0.40 AND viso_enhanced_macro_dev ≥ 0.15. FALSIFIER: any of dev_fake_macro_recall < 0.30, lockbox_real_fpr > 0.05, viso_enhanced_macro_dev < 0.10.
- **source**: `threads/iq_shortcut_deconvolution_program_2026-05-08.md:1226`

### `slot-b-viso-lift-mechanism-and-lockbox-fpr-localization`
- **status**: open
- **severity**: high
- **first_seen**: 2026-05-16
- **last_verified**: 2026-05-16
- **close_criterion**: a per-identity decomposition of Slot β step3500's lockbox_real_fpr (~$0 MPS, ~1h) determines whether the +5.5pp absolute penalty is concentrated on 2-3 chronic identities (rule-rescuable per memory `project_blend_unsharp_lever_2026-05-14`) or distributed across new identities. AND a 5-axis sister-variant (axes = current 4 + color_b_dev_high only, no luma_mean_high) tests whether the viso lift comes from color_b alone or requires both new axes. The combination of these two tests resolves whether Slot β is a deployment path.
- **source**: `threads/iq_shortcut_deconvolution_program_2026-05-08.md:1143`

### `per-base-substrate-pair-cohort-math-untested`
- **status**: open
- **severity**: high
- **first_seen**: 2026-05-22
- **last_verified**: 2026-05-22
- **close_criterion**: run Wilcoxon on Slot A v2 step3500 on the 275 paired viso fakes from analysis/pair_loss_effect_verification_2026-05-05; CPU-2 in next Phase 1
- **source**: `threads/clean_teams_identity_pairing.md:146`

### `viso-fake-signature-non-face-vs-face-localization`
- **status**: open
- **severity**: high
- **first_seen**: 2026-05-22
- **last_verified**: 2026-05-22
- **close_criterion**: per-patch ablation on viso vs deeplive fake misses; CPU-1 in next Phase 1
- **source**: `threads/viso_bucket_gap.md:317`

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
- **source**: `threads/viso_bucket_gap.md:288`

### `shortcut-block-criterion-may-be-scorer-artifact-bound`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-04-29
- **last_verified**: 2026-04-29
- **close_criterion**: a written disposition is recorded — either (a) the `shortcut-deployment-block` close criterion's `lockbox_fake_recall ≥ 0.60` half is reaffirmed under the corrected contract scorer policy (`target_fake_recall_min=0.30`) and a downstream packet hits both halves at the same τ, OR (b) the criterion is explicitly revised to a frame-level-AUC-based version (e.g. `dor-real-webcam-false-flag-no-virtual-bg` mean prob ≤ 0.30 AND `visomaster_enhanced_macro_dev` frame-level AUC ≥ 0.85) with a documented rationale referencing memory `project_p8a_frame_level_auc_2026-04-29.md`. Surfaced as a sub-question to the parent `shortcut-deployment-block` (critical) loop, not a replacement for it.
- **source**: `threads/processing_signature_shortcut.md:339`

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
- **source**: `threads/anti_shortcut_bundle_decomposition.md:61`

### `corr-penalty-frozen-head-shifting-axes-not-targeted`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-05-06
- **last_verified**: 2026-05-06
- **close_criterion**: either (a) the encoder fine-tune scorecard's 5-axis audit shows |Pearson r| on `is_webcam` AND `is_screen` did NOT amplify > 50% relative to E2B baseline (predicted shifting did not occur on the encoder; resolves favorably) OR (b) the audit shows amplification > 50% on at least one of those axes (the frozen-head prediction holds) AND a successor packet design includes `is_webcam` or `is_screen` in the penalty axes, with a path to deploy them at inference (currently blocked because Teams does not surface capture mode — would need an upstream classifier or a different inference-time signal that proxies for capture mode). The frozen-head prototype showed `face_area` 0.032→0.157 and `is_webcam` 0.081→0.205 as λ ramped 0→1, so the encoder is the place this gets adjudicated. Resolution flips this loop's status to `resolved` favorably or `superseded` (lever-class cap discovered).
- **source**: `threads/correlation_penalty_loss.md:115`

### `canary-empirical-validation`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-05-07
- **last_verified**: 2026-05-07
- **close_criterion**: at least 2 P2 ckpts have BOTH (a) a canary readout at the corresponding training step AND (b) a post-training contract scorecard verdict. Compute Pearson correlation between `canary/lockbox_recall_at_FPR_10pct` and the scorecard's `lockbox_fake_recall` at calibrated τ. Compute the same for `canary/max_per_identity_mean_score` and the F5 close-criterion's binding identity. If r > 0.7 on both, the canary is empirically validated as a deployment proxy. If r < 0.3, the canary's design is wrong and the metric set needs re-deriving from the P2 outcomes. Either result closes the loop.
- **source**: `threads/in_training_canary_signal.md:99`

### `open-loops-stale-state-claims`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-05-07
- **last_verified**: 2026-05-07
- **close_criterion**: `tools/regenerate_open_loops.py` is extended to (a) flag entries whose `last_verified` is more than 30 days old AND whose `close_criterion` text contains state-claim keywords (`uncommitted`, `in working tree`, `not yet committed`, `pending commit`), AND (b) auto-check those claims against current `git status` / `git log` output where possible. Output is a warning section in `OPEN_LOOPS.md` listing entries that need re-verification.
- **source**: `threads/eval_substrate_layering.md:95`

### `roy-d-color-b-dev-mechanism`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-05-07
- **last_verified**: 2026-05-07
- **close_criterion**: determine whether P8A's r(score, color_b_dev) = −0.714 on roy_d is (a) generalization (roy_d is held-out from P8A's train set; P8A learned a useful identity-invariant feature), or (b) memorization (roy_d is in train set; P8A's association is set-specific). The (a) interpretation makes the P1 regression more concerning; (b) makes it less so. A grep of the train manifest for Roy_D would close the loop cheaply.
- **source**: `threads/pair_rank_collateral.md:61`

### `roy-d-mechanism-not-fully-diagnosed`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-05-10
- **last_verified**: 2026-05-10
- **close_criterion**: a CPU diagnostic determines the actual mechanism behind T3's universal Roy_D regression. Two cheap candidates are: (a) cross-correlate Roy_D regression magnitude with each identity's color_a_dev × saturation profile across all 18 T3 ckpts (need to score the 12 un-scored Slot 2 + Slot 3 step ckpts on Mac first, ~30 min CPU); (b) probe whether disabling visomaster_teams_enhanced (one of PA's data sources) on a T3-style retrain restores Roy_D handling — this isolates whether PA's data-source addition or T3's keep-list addition is the Roy_D-shifting culprit (requires GPU retrain, not cheap). Loop closes when EITHER (a) confirms the color_a_dev mechanism (in which case T4 hard-negative-mining or face_scale_jitter is the right intervention) OR (b) refutes both candidate mechanisms (in which case Roy_D regression mechanism remains unknown and a deeper representation-level probe is needed). Current evidence: Lap-shortcut hypothesis FALSIFIED (Roy_D Lap p50=66 is LOW, would not have been dropped by keep-list). Source: `MORNING_BRIEF_2026-05-10.md` §5 + §11.
- **source**: `threads/viso_bucket_gap.md:274`

### `chronic-6-encoder-iq-angle-drift-during-ft`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-05-12
- **last_verified**: 2026-05-12
- **close_criterion**: a counterfactual training experiment establishes whether the chronic-6 IQ-PC1 angle drift from CLIP-frozen's 83.68° to FT'd ckpts' 74.95°-78.54° is causally responsible for chronic-cohort FPR, OR is a correlated side-effect. Candidate experiment: FT with continuous-axis-GRL on 6 IQ axes that explicitly pulls the encoder direction toward IQ-orthogonality. If post-training chronic-6 angle ≥ 82° AND chronic-cohort FPR drops by ≥ 0.05 absolute at FPR-cal τ, the drift is causally implicated. If angle goes ≥ 82° but FPR doesn't change, the drift is incidental.
- **source**: `threads/iq_shortcut_deconvolution_program_2026-05-08.md:868`

### `chronic-6-encoder-iq-angle-drift-during-ft`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-05-12
- **last_verified**: 2026-05-12
- **close_criterion**: a counterfactual training experiment establishes whether the chronic-6 IQ-PC1 angle drift from CLIP-frozen's 83.68° to FT'd ckpts' 74.95°-78.54° is causally responsible for chronic-cohort FPR, OR is a correlated side-effect. Candidate experiment: FT with continuous-axis-GRL on 6 IQ axes that explicitly pulls the encoder direction toward IQ-orthogonality. If post-training chronic-6 angle ≥ 82° AND chronic-cohort FPR drops by ≥ 0.05 absolute at FPR-cal τ, the drift is causally implicated. If angle goes ≥ 82° but FPR doesn't change, the drift is incidental.
- **source**: `threads/iq_shortcut_deconvolution_program_2026-05-08.md:1025`

### `t5c-classifier-capacity-mechanism`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-05-12
- **last_verified**: 2026-05-12
- **close_criterion**: L11 atlas inv_mean recompute on T5C step1500 + step3500 + step3750 (a la A3 2026-05-11) — if T5C step3500 chronic_6 inv_mean ≥ P8A's (vs T4 step10500's −0.0315 absolute), classifier capacity is the lever responsible for the chronic_6 lift; otherwise the dor-cohort lift comes from somewhere else and §3.1 of the opinion-doc is refuted
- **source**: `threads/iq_shortcut_deconvolution_program_2026-05-08.md:812`

### `cpu-probe-mechanism-discrimination`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-05-16
- **last_verified**: 2026-05-16
- **close_criterion**: the 2026-05-15 resolution-chain probe template is amended to report fake-vs-real AUC on the panel for each ckpt scored. The AUC distinguishes encoder-level invariance (AUC preserved) from score-distribution compression (AUC reduced). Once added, any single-lever IQ-axis-attack packet must demonstrate that AUC is preserved as the pre-launch CPU gate.
- **source**: `threads/iq_shortcut_deconvolution_program_2026-05-08.md:1127`

### `lockbox-real-fpr-tiebreak-is-load-bearing`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-05-16
- **last_verified**: 2026-05-16
- **close_criterion**: a $0 CPU re-rank of the most recent scorecard using `dev_fake_macro_recall` as the tiebreak (instead of `lockbox_real_fpr`) AND a per-IQ-quartile cell decomposition of `lockbox_real_fpr` at each ckpt's selected τ. If the per-quartile FPR is uniformly below dev FPR (per D4 2026-05-12 finding), the tiebreak is measuring τ-tail-density not lockbox substrate difficulty. Decision: whether to amend the contract policy.
- **source**: `threads/iq_shortcut_deconvolution_program_2026-05-08.md:1160`

### `slot-a-anchor-aware-bounded-by-pool-content`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-05-16
- **last_verified**: 2026-05-16
- **close_criterion**: scorecard outcome on Slot A step3500. Slot A's anchor pool is dor-webcam content only (30 frames). The encoder embedding probe shows anchor pool frames at PC1=+2.67 (clean side), only 16.7% closer to Roy_D. Predicts Slot A reduces dor chronic FP but does NOT generalize to Roy_D / xiang / PC_Generator. CONFIRMER: dor chronic_FP reduces ≥30%. NO_TRANSFER FINDING: Roy_D over-fire rate ≈ unchanged from T5C step3500.
- **source**: `threads/iq_shortcut_deconvolution_program_2026-05-08.md:1242`

### `slot-a-bla-bla-chow-regression`
- **status**: open
- **severity**: medium
- **first_seen**: 2026-05-16
- **last_verified**: 2026-05-16
- **close_criterion**: encoder probe on Slot A v2's bla_bla_chow embeddings. If bla_bla_chow shifted toward Roy_D region (away from clean), confirm the spillover mechanism and tune weight to reduce.
- **source**: `threads/iq_shortcut_deconvolution_program_2026-05-08.md:1309`

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
- **source**: `threads/anti_shortcut_bundle_decomposition.md:70`

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

### `hdtf-promotion-contract-failure-recurrence`
- **status**: open
- **severity**: low
- **first_seen**: 2026-05-05
- **last_verified**: 2026-05-10
- **close_criterion**: `score_teams_promotion_contract.py:748` is patched so the hard-coded `--dev_real_suite teams_real_all_dev` default is replaced by a configurable flag OR a fallback that detects suite-name mismatches gracefully (e.g., "no v2-style real suite in manifest, skipping contract scoring with verdict=PERSUITE_ONLY"). All HDTF runs since 2026-05-05 (P8A-on-HDTF, PA-on-HDTF, P2_D-on-HDTF, T3-on-HDTF, T3-step2500-on-HDTF) hit `JOB_STATE_FAILED` at the contract-scoring tail because the HDTF suite manifest does not include `teams_real_all_dev`. Per-suite reports always complete cleanly so the failure is non-blocking, but the FAILED state interferes with the persistent monitor pattern and adds operational friction. Closes when the patch is committed and a fresh HDTF run reaches `JOB_STATE_SUCCEEDED`.
- **source**: `threads/viso_bucket_gap.md:281`

### `canary-finer-resolution`
- **status**: open
- **severity**: low
- **first_seen**: 2026-05-07
- **last_verified**: 2026-05-07
- **close_criterion**: a "tiny canary" companion of ~60 frames runs every 200 steps for finer resolution at the cost of ~0.1% extra training time. Useful specifically because P1 BUNDLE_step500 was already in the failed regime by step 500 — the current 1000-step cadence might miss the inflection. Implement only if the 1000-step cadence proves to be too coarse on the P2 runs. Tiny canary composition would be: 5 chronic identities × 6 frames + 30 lockbox fakes = 60 frames.
- **source**: `threads/in_training_canary_signal.md:113`

### `canary-wilcoxon-cohort-split`
- **status**: open
- **severity**: low
- **first_seen**: 2026-05-07
- **last_verified**: 2026-05-07
- **close_criterion**: the Wilcoxon-vs-P8A metric is split into two cohort-specific stats: `canary/wilcoxon_stat_vs_p8a_healthy_reals` (where small drift = good) and `canary/wilcoxon_stat_vs_p8a_chronic_reals` (where large negative drift = good). The aggregate stat is preserved but augmented. Implementation is ~30 lines in `trainer/mixins/canary_probe.py:_aggregate_metrics`. Defer until P2 results validate the basic metric set; if the basic Wilcoxon already correlates with deployment quality, the cohort split may not be needed.
- **source**: `threads/in_training_canary_signal.md:106`

### `clip-frozen-chronic-6-auc-robustness`
- **status**: open
- **severity**: low
- **first_seen**: 2026-05-12
- **last_verified**: 2026-05-12
- **close_criterion**: extend the CLIP-frozen chronic-6 probe to a larger sample (e.g., 500-1000 chronic-6 reals from contract suites + 100-200 chronic-6 fakes). If 5-fold CV probe AUC remains ≥ 0.95 at the larger sample, the "forgery signal is in raw CLIP" framing is robust. If the AUC drops below 0.90 at scale, the n=282 triptych result is sample-size-inflated and the framing weakens.
- **source**: `threads/iq_shortcut_deconvolution_program_2026-05-08.md:879`

### `clip-frozen-chronic-6-auc-robustness`
- **status**: open
- **severity**: low
- **first_seen**: 2026-05-12
- **last_verified**: 2026-05-12
- **close_criterion**: extend the CLIP-frozen chronic-6 probe to a larger sample (e.g., 500-1000 chronic-6 reals from contract suites + 100-200 chronic-6 fakes). If 5-fold CV probe AUC remains ≥ 0.95 at the larger sample, the "forgery signal is in raw CLIP" framing is robust. If the AUC drops below 0.90 at scale, the n=282 triptych result is sample-size-inflated and the framing weakens.
- **source**: `threads/iq_shortcut_deconvolution_program_2026-05-08.md:1036`

### `train-bucket-identity-overlap-gcs-audit`
- **status**: open
- **severity**: low
- **first_seen**: 2026-05-12
- **last_verified**: 2026-05-12
- **close_criterion**: a GCS-side enumeration (`gsutil ls gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2/samples/ | head -1000`) plus identity-name extraction from per-sample `manifest.json`'s `original_video_name` field determines whether the SAME HUMANS as `real_dor`/`Cam_Test`/`PC_Generator`/etc. appear in the training-side teams-v2 bucket under different sample_ids or sessions. If yes, the "P8A memorization" framing is plausible and the critic's §3.2 C4 reframe should be partially walked back. If no, the "P8A invariance" framing is strengthened and Slot C (L11 anchor on 5-identity cohort) becomes safer. Cost: <10 min CPU + GCS list quota; <$1.
- **source**: `threads/iq_shortcut_deconvolution_program_2026-05-08.md:905`

### `canary-probe-not-default-in-yaml-templates`
- **status**: open
- **severity**: low
- **first_seen**: 2026-05-16
- **last_verified**: 2026-05-16
- **close_criterion**: the next packet author adds a default-on canary block to the packet yaml template OR explicitly documents the rationale for keeping it off. The canary infrastructure has been available since 2026-05-07; both Slot α and Slot β yamls inherited the canary-disabled state from `R13_T5C_T4_BIG_CLASSIFIER_2026-05-11.yaml`. The canary would have surfaced fake-side score crash by step 500-1000 in Slot α, allowing early-stop.
- **source**: `threads/iq_shortcut_deconvolution_program_2026-05-08.md:1177`

### In progress (6)

### `shortcut-deployment-block`
- **status**: in-progress
- **severity**: critical
- **first_seen**: 2026-04-24
- **last_verified**: 2026-05-07
- **close_criterion**: a Packet-7+ checkpoint achieves `dor-real-webcam-false-flag-no-virtual-bg` mean prob_fake ≤ 0.30 (vs RLP6_04's 0.932 post-fix) AND `lockbox_fake_recall ≥ 0.60` at a τ that holds `teams_ood_real` FPR ≤ 5%, demonstrating the camera/ISP shortcut has been broken without sacrificing fake recall
- **source**: `threads/processing_signature_shortcut.md:350`

### `fpr-minimization-no-budget-tau-collapse`
- **status**: in-progress
- **severity**: high
- **first_seen**: 2026-04-17
- **last_verified**: 2026-04-30
- **close_criterion**: the `score_teams_promotion_contract.py` runner enforces a recall floor or τ ceiling that prevents `selected_threshold ≈ 0.995` configurations passing silently
- **source**: `threads/promotion_contract_evolution.md:160`

### `corr-penalty-deployment-grade-verdict-pending`
- **status**: in-progress
- **severity**: high
- **first_seen**: 2026-05-06
- **last_verified**: 2026-05-06
- **close_criterion**: a promotion-contract scorecard run on the 8-entry ckpt map `arena/checkpoint_maps/teams_target_domain.deeplive_viso_corr_2026-05-06.yaml` against `arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml` lands and is read against the 4/4 close criterion (F1 lockbox recall ≥ 90% at FPR ≤ 10%; F2 shortcut weakening ≥ 30% on ≥ 2 of 5 axes vs E2B baseline; F3 no untargeted axis +50%; F4 HDTF cross-substrate FPR ≤ 5%). The Phase-2 5-axis audit (`analysis/deeplive_viso_corr_eval_2026-05-06/run_audit.py` Phase 2 block, currently commented out — uncomment after `gcloud storage cp -r <scorecard_output>/raw_reports analysis/deeplive_viso_corr_eval_2026-05-06/raw_reports/` lands) is the comparison frame. Either (a) verdict is positive on all 4 criteria → corr-penalty is the first R13 anti-shortcut intervention to clear deployment grade and gets a "Confirmed good" entry in the README; or (b) any criterion fails → packet is muddled / failed and the lever class joins anchor_aware, GRL, and face_scale_jitter on the "structurally working but not deployment-grade" list.
- **source**: `threads/correlation_penalty_loss.md:106`

### `enhanced-vs-unenhanced-val-pool-confound`
- **status**: in-progress
- **severity**: medium
- **first_seen**: 2026-04-21
- **last_verified**: 2026-04-29
- **close_criterion**: a packet trains an enhanced-proper arm and measures it against a validation pool that explicitly contains enhanced-proper rows, with dose matched to the unenhanced arm — so "enhanced hurts" / "enhanced helps" is readable without the pool-composition + dose confound
- **source**: `threads/gate_alignment_story.md:91`

### `quality-enhancement-strategy-misrouted-fix-pending`
- **status**: in-progress
- **severity**: medium
- **first_seen**: 2026-05-05
- **last_verified**: 2026-05-05
- **close_criterion**: routing fix is applied — `quality_enhancement` is removed from `DEFAULT_ENHANCED_STRATEGIES` in `utils/grouping.py:13-17` AND from every R13 yaml's `enhanced_strategy_names` override (~15 yamls under `experiments/phase2_round13/`); a unit test asserts `infer_family_key` returns `deeplive_non_enhanced_fake` for a `quality_enhancement` fake input; image is rebuilt via `./dev.sh build-prod -y` (auto-bumps VERSION); and a critical-reading banner is added to the README and to historical packet retros (P8A, E2B, PA, PC) noting they trained on the contaminated routing. Validation re-run on at least one FT-from-base packet measuring the delta in deeplive enhanced-vs-non-enhanced family balance is recommended but not required for closure (could be folded into the in-scoping deeplive ship experiment instead). Verification step (visual inspection) is COMPLETE 2026-05-05 evening — user reported "quality enhancement is non-GFPGAN like we suspected." Bug confirmed; fix pending authorization.
- **source**: `threads/quality_enhancement_strategy_misrouting.md:175`

### `chronic_6-feature-regression-on-t4`
- **status**: in-progress
- **severity**: low
- **first_seen**: 2026-05-11
- **last_verified**: 2026-05-12
- **close_criterion**: a training-time intervention (T5-C stronger online classifier; OR T5-B multi-layer GRL attachment; OR head-only retrain on a substrate-diverse pool including lockbox-style data) measurably restores chronic_6 inv_mean Δ to ≥ 0 vs P8A on the FULL chronic_6 cohort (including PC_Generator/Roy_D/Q identities, not just bla_bla_chow) while preserving lockbox + HDTF performance
- **source**: `threads/iq_shortcut_deconvolution_program_2026-05-08.md:741`

### Resolved (17)

### `contract-policy-bug-fix-not-committed`
- **status**: resolved
- **severity**: high
- **first_seen**: 2026-04-23
- **last_verified**: 2026-05-07
- **close_criterion**: the recall-floor + budget-aware τ-selection patch is (1) committed to `teams-relaunch-root-2026-04-17`, (2) the image is rebuilt with the fix in, and (3) a contract scorecard run on a representative recent checkpoint is invoked with `--promotion_target_fake_recall_min 0.30` AND the resulting scorecard is documented as having selected τ via the recall-floor path (not via the legacy no-budget minimize-FPR-only path). All three components are required. **2026-04-30 evening update**: the `mclioexb` scorecard run did NOT exercise the v3 fix — the launcher (`arena/launch_teams_promotion_contract.sh`) omitted `--promotion_target_fake_recall_min`, so the scorecard ran the DEFAULT policy (τ=0.975, recall=0.13). Component 3 still NOT MET. **2026-05-07 update — components 1 and 2 are MET; component 3 is in flight.** Audit of the tree (commit `974e033` 2026-04-29) shows the scorer code with `target_real_fpr=0.07 / target_stress_fpr=0.10 / target_fake_recall_min=0.70` defaults was committed a week ago; the runner with `--promotion_target_*` CLI args was committed at the same time. The actual missing piece was the launcher: `arena/launch_teams_promotion_contract.sh` did not pass these flags through. Today's commits `ad070d3` (passing the flags + setting v3 defaults `0.07 / 0.10 / 0.30`) and `7f81e7a` (canonicalizing the 500GB scorecard template to prevent disk-exhaustion) close the launcher gap. Image `1.3.270` (commit `5dccfa4`, Cloud Build `40cf4d7c-4b8b-4af7-81ca-0991f2083450`) bakes everything in. The Phase A scorecard run for P1 (Vertex job `7995519158412378112`, us-east1, submitted 2026-05-07T08:19:20Z) explicitly invokes `--promotion_target_fake_recall_min 0.30` — verifiable in the gcloud `containerSpec.args` trace. Component 3 closes when the resulting `promotion_winner.json` shows τ selected via the recall-floor path; verdict pending Phase A finish (~12:30 UTC).
- **source**: `threads/contract_policy_bug.md:108`

### `data-axis-clean-single-lever-retest-in-progress`
- **status**: resolved
- **severity**: high
- **first_seen**: 2026-05-04
- **last_verified**: 2026-05-05
- **close_criterion**: Packet A (`R13_PA_VISOMASTER_ENHANCED_DATA.yaml`, run `3140330896851206144`, JOB_STATE_RUNNING in us-east1) and/or Packet C-codec (`R13_PC_CODEC_PLUS_DATA.yaml`, run `1202657157175050240`, JOB_STATE_RUNNING in us-east1) complete training, the resulting checkpoints are scored under both the F0 (full eval substrate) and F4 (substrate-cleaning, `analysis/substrate_cleaning_eval_2026-05-05/`) lenses, and a documented verdict is recorded for whether enabling `visomaster_enhanced + visomaster_teams_enhanced` data sources at fw=4.0 lifts visomaster_enhanced_macro_dev recall above the E2B_3200 baseline (8.4% F0 / 30.9% F4) at deployment-honest single-τ at 5% FPR ceiling. The verdict closes EITHER as (a) data-axis lever is dispositive (Packet A/C-codec materially beats E2B on viso recall under deployment policy), OR (b) data-axis lever as cleanly tested still does not lift, in which case memory `project_data_axis_lever_pulled_twice_no_lift.md` is amended to "pulled three times" with the bundle-confound caveat lifted from the prior two attempts. Either outcome closes the loop.
- **source**: `threads/viso_bucket_gap.md:160`

### `dev-to-lockbox-substrate-transfer-gap`
- **status**: resolved
- **severity**: high
- **first_seen**: 2026-05-11
- **last_verified**: 2026-05-12
- **close_criterion**: EITHER (a) a head-retrain on a substrate-diverse pool (dev + lockbox-style data, properly held-out for eval) lifts T4 lockbox AUC ≥ 0.90 — confirms the gap is closeable by exposing the head to substrate-diverse training, OR (b) the gap is REPRODUCED on multiple ckpts (T4, P8A, T3) and on the full lockbox cohort — closes as "structural data gap; requires new training data ingestion, not architecture/loss changes"
- **source**: `threads/iq_shortcut_deconvolution_program_2026-05-08.md:1001`

### `canary-silence-when-multi-axis-grl-active`
- **status**: resolved
- **severity**: high
- **first_seen**: 2026-05-20
- **last_verified**: 2026-05-20
- **close_criterion**: a code fix lands such that for a yaml configuration with BOTH `multi_axis_grl.enabled: true` AND `canary_probe.enabled: true`, the next training run logs ≥1 `canary/score_p95_on_reals` value to W&B within the first 1000 steps. The fix's root-cause investigation must name the specific raise site in `trainer/mixins/canary_probe.py:_run_canary_probe` or its callee that the try/except is currently swallowing (candidates documented in `analysis/manual_canary_2026-05-20/DEEP_DIVE_FACTS_2026-05-20.md §1.2`: model forward `data_dict` missing GRL-axis labels; ArcFace head 2-tuple unpack at `detectors/effort_detector.py:1813`; dict-key extraction at `canary_probe.py:312-318`). Acceptance: a smoke test in `tests/test_canary_with_grl_wiring.py` (new file) exercises a model with `multi_axis_grl.enabled: true` and `use_arcface_head: true`, invokes `_run_canary_probe` once, and asserts ≥1 metric was logged + 0 warnings of pattern `Canary probe.*FAILED|disabled`. The smoke test must be added to CI alongside `tests/test_lora_adapter.py`. Pending the fix, the workaround is off-line replication via `analysis/manual_canary_2026-05-20/score_canary.py` — slow ($0 CPU, ~30 sec per ckpt) but correctness-preserving for non-LoRA ckpts.
- **source**: `threads/in_training_canary_signal.md:120`

### `lora-enabled-not-propagated-by-load-model`
- **status**: resolved
- **severity**: high
- **first_seen**: 2026-05-20
- **last_verified**: 2026-05-20
- **close_criterion**: a code fix lands at one of (a) the trainer ckpt-save path (`trainer/trainer.py` `save_checkpoint` or equivalent) so the `lora` block from cfg is embedded into the ckpt's `model_config` dict, OR (b) the `load_model` function (`batch_inference_gcs.py:420-466`) infers `lora.enabled=true` from presence of `lora_A`/`lora_B` keys in `state_dict` and derives `target_layers` from the keys, OR (c) the ckpt-save path embeds the full active yaml under a new `training_yaml` key in the ckpt and `load_model` reads it explicitly. Acceptance: a new test `tests/test_lora_ckpt_roundtrip.py` saves a LoRA ckpt with `enabled=true, target_layers=[8,9], rank=8, alpha=16`, calls `load_model` on the saved path, and asserts (1) the resulting model's `named_parameters` includes `backbone.visual.transformer.resblocks.8.attn.out_proj.lora_A.weight`, AND (2) `model.load_state_dict(...)` returned zero `unexpected_keys` from the lora_* family. The test must be added to CI alongside `tests/test_lora_adapter.py` and `tests/test_train_sweep_reapply_allowlist.py`. Until the fix lands, off-line analyses of LoRA ckpts via `load_model` must NOT cite their numbers without first verifying the LoRA tensors loaded (e.g., assert `model.named_parameters()` contains `lora_A` keys; or use `scripts/smoke_lora_wiring_2026-05-12.py` to confirm).
- **source**: `threads/wandb_yaml_propagation_bugs.md:130`

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

### `fourier-aug-band-overlap-not-resolved`
- **status**: resolved
- **severity**: medium
- **first_seen**: 2026-05-06
- **last_verified**: 2026-05-06
- **close_criterion**: a follow-up probe partitions FFT amplitude bands into "shortcut-only" (high contribution to Probe 1's may6/may5 separation, low contribution to Probe 3's fake/real separation) vs "signal-carrying" (the inverse), with a quantitative overlap measure. If the overlap is small (e.g., separable bands give ≥80% of shortcut signal while preserving ≥80% of fake signal), the band-limited Fourier-aug recipe is greenlit as a packet candidate. If overlap is large (any randomization that hurts the shortcut also hurts the fake signal substantially), the lever class moves from "viable" to "abandoned"; the next-packet decision pivots fully to AugMix consistency + SBI.
- **source**: `threads/processing_signature_shortcut.md:214`

### `wt-e-promotion-winner-deferred`
- **status**: resolved
- **severity**: low
- **first_seen**: 2026-04-17
- **last_verified**: 2026-04-29
- **close_criterion**: at least one shortlist run produces a written `promotion_winner.json`/`checkpoint_summary.csv` artifact pair on the WT-E manifest+shortlist
- **source**: `threads/promotion_contract_evolution.md:151`

### `face-scale-jitter-composability`
- **status**: resolved
- **severity**: low
- **first_seen**: 2026-04-30
- **last_verified**: 2026-05-12
- **close_criterion**: empirical test of jitter@0.50 stacked on top of a non-bundle base (T3 SLOT1 or T4 multi-axis-L11-GRL) — if dev_macro_recall holds within 0.05 of base, composability is supported; if regression > 0.05 absolute, lever is bundle-replacement only
- **source**: `threads/iq_shortcut_deconvolution_program_2026-05-08.md:800`

### `pair-loss-asymmetric-variant-untested`
- **status**: resolved
- **severity**: low
- **first_seen**: 2026-05-04
- **last_verified**: 2026-05-04
- **close_criterion**: either (a) a $5 GPU probe extracts E2B features on the same 550 (raw, teams) viso pairs and re-runs Q1/Q2 directly on E2B geometry, with documented verdict on whether the P8A-as-proxy assumption hides a real signal; OR (b) an asymmetric-pair-loss variant (only aligning teams toward raw when raw scores higher than teams — never the reverse) is scoped, the cohort math is recomputed under that asymmetry, and a go/no-go decision is documented; OR (c) the loop is explicitly closed-as-not-pursued with a one-line note that the symmetric pair loss verdict is dispositive enough to deprioritize the asymmetric variant given the 9pp upper-bound ceiling and the existence of cheaper alternatives (substrate cleaning, per-substrate τ-calibration).
- **source**: `threads/clean_teams_identity_pairing.md:93`

### `encoder-vs-head-locus-on-t4-lockbox`
- **status**: resolved
- **severity**: low
- **first_seen**: 2026-05-11
- **last_verified**: 2026-05-11
- **close_criterion**: frozen-encoder linear probe on T4 lockbox features yields AUC ≥ 0.85 across multiple T4 ckpts (= α-outcome, encoder retains separation, head is the failure point) OR AUC ≤ trained-head AUC (= β-outcome, encoder lost separation)
- **source**: `threads/iq_shortcut_deconvolution_program_2026-05-08.md:728`

### `train-bucket-identity-overlap-gcs-audit`
- **status**: resolved
- **severity**: low
- **first_seen**: 2026-05-12
- **last_verified**: 2026-05-12
- **close_criterion**: a GCS-side enumeration (`gsutil ls gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2/samples/ | head -1000`) plus identity-name extraction from per-sample `manifest.json`'s `original_video_name` field determines whether the SAME HUMANS as `real_dor`/`Cam_Test`/`PC_Generator`/etc. appear in the training-side teams-v2 bucket under different sample_ids or sessions. If yes, the "P8A memorization" framing is plausible and the critic's §3.2 C4 reframe should be partially walked back. If no, the "P8A invariance" framing is strengthened and Slot C (L11 anchor on 5-identity cohort) becomes safer. Cost: <10 min CPU + GCS list quota; <$1.
- **source**: `threads/iq_shortcut_deconvolution_program_2026-05-08.md:1014`

### Superseded (1)

### `p14-data-fix-not-launched`
- **status**: superseded
- **severity**: medium
- **first_seen**: 2026-04-29
- **last_verified**: 2026-04-30
- **close_criterion**: `R13_P14_DATA_FIX.yaml` is committed, the smoke loads enhanced viso samples cleanly, the retrain completes, and a contract scorecard run on the resulting checkpoint shows `visomaster_enhanced_macro_dev` recall lifted materially above the P8A baseline (≥ 24% target under the corrected contract policy with `target_fake_recall_min=0.30`) — verifying the bucket-gap closure on the actual headline metric.
- **source**: `threads/viso_bucket_gap.md:64`

### Unknown status (1)

### `dev-to-lockbox-substrate-transfer-gap`
- **status**: resolvable
- **severity**: high
- **first_seen**: 2026-05-11
- **last_verified**: 2026-05-12
- **close_criterion**: EITHER (a) a head-retrain on a substrate-diverse pool (dev + lockbox-style data, properly held-out for eval) lifts T4 lockbox AUC ≥ 0.90 — confirms the gap is closeable by exposing the head to substrate-diverse training, OR (b) the gap is REPRODUCED on multiple ckpts (T4, P8A, T3) and on the full lockbox cohort — closes as "structural data gap; requires new training data ingestion, not architecture/loss changes"
- **source**: `threads/iq_shortcut_deconvolution_program_2026-05-08.md:856`

