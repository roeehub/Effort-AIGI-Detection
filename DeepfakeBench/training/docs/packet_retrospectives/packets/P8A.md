# Packet P8A · CLIP-backbone unfreeze breaks the camera-signature anchor ceiling (with a per-method fake-recall regression)

> **⚠ Critical-reading note (added 2026-05-05)**: P8A was trained with the `quality_enhancement` routing bug active (commit 38558ee5 → fixed 2026-05-05). At training time, ~5,120 deeplive `quality_enhancement_*` frames were routed to the `deeplive_enhanced_fake` family at weight 3.0 (instead of the correct `deeplive_non_enhanced_fake` family at weight 2.5). All numerical results below are conditional on that contamination. The anchor-pool / FPR / fake-recall framing within P8A remains internally valid (relative readings are uniform across the chain); claims about *how* P8A handles GFPGAN-enhanced fakes specifically need revisiting once a post-fix retraining run lands. See [`quality_enhancement_strategy_misrouting`](../threads/quality_enhancement_strategy_misrouting.md) for full evidence and fix details.

> **Template contract**: fill every section. If a section truly has no content, write `*(none — reason)*` rather than deleting the heading. Keep the status-card table intact.

## Status card

| Field | Value |
|---|---|
| Dates | 2026-04-24 (drafted) → 2026-04-25 (run + rescore + state-of-detector) |
| Slots | 2 (`R13_RLP8_01_unfreeze_clip_codec` = P8A, `R13_RLP8_02_fresh_head_plain_clip` = P8B) |
| Headline lever | Unfreeze CLIP backbone surgically (`unfreeze_final_proj` + `unfreeze_final_ln` + `apply_svd_to_mlp`) on top of RLP7_02 codec-aggressive aug |
| Leader slot | `R13_RLP8_01` (P8A, run `9lmvb5b4`, value_composite step 5000) |
| Leader metric | trainer `value_composite=0.99+` step 5000, `auc=0.9926`, `eer=0.0270`; **anchor pool Δ=−0.188 vs RLP6_04** (best P7 was −0.097) |
| Verdict | ⚠️ **muddled — single-variable winner on anchor + real-pool FPR; aggregate fake recall regressed −13.6 pp on `teams_fake_all_dev`; not crownable on the lockbox-anchored contract** |
| Next-packet decision | Packet-9 = "softened P8A" — single-variable disentangle of magnitude (`backbone_lr_mult`), topology (`apply_svd_to_mlp` removal), data (real-codec uplift), plus replication seed and freeze-native control |
| Themes touched | [processing_signature_shortcut](../threads/processing_signature_shortcut.md) (primary — first packet to break the anchor ceiling) · [in_proj_svd_gradient_bug](../threads/in_proj_svd_gradient_bug.md) (retroactively; in_proj-SVD was no-op during this run) · [value_composite_semantics](../threads/value_composite_semantics.md) (concrete proof composite ≠ deployment-grade) · [promotion_contract_evolution](../threads/promotion_contract_evolution.md) (the per-method fake recall regression is the load-bearing case) |

## Configuration

Both P8A and P8B fork from a different premise than the rest of the R13 chain:

- **P8A** (`R13_RLP8_01_unfreeze_clip_codec.yaml`) — single-variable delta from RLP7_02. Three flags flipped: `backbone.unfreeze_final_proj: true`, `backbone.unfreeze_final_ln: true`, `backbone.apply_svd_to_mlp: true`. Everything else identical: SVD rank 736 on in_proj, codec-aggressive aug, combined_paired data, LR 3e-5, arcface m=0.15, 10k schedule, seed 749. Base checkpoint RLP6_04 step 23500. Trained 2026-04-24 21:41 UTC → 23:48 UTC, ~2h7min on us-east1, image 1.3.206 (commit `d5be7ce`).
- **P8B** (`R13_RLP8_02_fresh_head_plain_clip.yaml`) — scratch from plain CLIP-DataComp-XL: 30k steps, LR 2e-4, ArcFace m=0 + s-anneal 10→14, no R12g/R13 inheritance. Hypothesis: is the camera/ISP shortcut in the R12g/R13 weight chain or in the data mix? Run `n8yk2hox` on us-west4. Hung at step 12000 / 30000 on `teams_ood_fake` data loader at 00:58 UTC 2026-04-25; cancelled at 03:13 UTC (cancel was the right call; was a process mistake to do without authorization, see memory `feedback_no_cancelling_vertex_jobs.md`).

Yaml-only enablement of three flags + new yaml — no model-side or loss-side code change in this packet. The `backbone.unfreeze_*` and `apply_svd_to_mlp` flags were already supported in the trainer (introduced in R12g per memory `project_p8a_breakthrough.md`); this packet was the first to **simultaneously** enable all three on top of RLP6_04 + codec-aggressive aug + the full combined_paired mix.

**Retroactive correction (2026-04-26):** P8A's published trainable-parameter description was wrong by ~3M params per attention block × 12 blocks. Every P8A run had `apply_svd_to_in_proj: true` but the q/k/v residuals received zero classification gradient until the 2026-04-26 fix (commit `2feea58`). See [`in_proj_svd_gradient_bug`](../threads/in_proj_svd_gradient_bug.md). The actual lever set during P8A was: head + out_proj-SVD + MLP-SVD + unfrozen `visual.proj` + unfrozen `ln_post`. The "more reach" qualitative explanation still holds; the parameter-count headline does not.

## Results at the time

### Anchor-pool (180 frames across 6 Dor/Roee pools, `analysis/pool_rescore_rlp8_a.summary.json`)

| Pool | n | RLP6_04 | **P8A** | Δ | frac > 0.9 |
|---|---:|---:|---:|---:|---:|
| dor-laptop-correct-whiteish | 30 | 0.0126 | 0.0064 | −0.0062 | 0/30 |
| dor-laptop-correct-yellowish | 30 | 0.0144 | 0.0072 | −0.0073 | 0/30 |
| roee-windows-laptop-correct | 30 | 0.0060 | 0.0054 | −0.0006 | 0/30 |
| dor-webcam-false-flag-VBG | 30 | 0.9640 | 0.7304 | **−0.234** | 14/30 |
| **dor-webcam-no-VBG (ANCHOR)** | 30 | 0.9317 | 0.7437 | **−0.188** | 13/30 |
| roee-mac-false-flag-VBG | 30 | 0.7515 | 0.3528 | **−0.399** | 3/30 |

Best of P7 was `RLP7_08` at anchor Δ=−0.097 / Roee-mac Δ=+0.072; P8A roughly **doubles** the anchor improvement and converts the Roee-mac regression to a strong improvement. All three real-correct pools also moved further toward zero (no FPR regression).

### Trainer-side W&B summary at value_composite step 5000 (run `9lmvb5b4`)

- AUC = 0.9926, EER = 0.0270, value_composite = 0.99+.
- Per-method fake recall on training-time eval (39 fake methods): macro acc 0.977, 33/39 at 100%. Imperfect: facedancer 0.75 (n=4, noisy), mobileswap 0.83 (regressed from 1.0 at step 2500), blendface 0.92 (regressed from 1.0 at step 2500), simswap 0.93, deeplive_edge_cases 0.94, deeplive_teams_edge_cases 0.96, deeplive_teams_minimal_processing 0.98, deeplive_minimal_processing_enhanced 0.97. One real method: `external_vcd_real` 0.80.
- Threshold curve: τ at FPR 1% → TPR 0.952; FPR 5% → TPR 0.972.
- OOD aug-stress slices (200 reals from external_youtube_avspeech, eval-time augs): P8A leads `teams_ood_real_real` (0.995 vs RLP6_04 0.968), tied for `teams_ood_fake_fake` (0.995). Wins or ties three lighting-stress slices. **Regressed vs RLP7_05** on `crop_shift` (0.639 vs 0.681) and `scale` (0.653 vs 0.708) — the unfreeze undid RLP7_05's spatial-aug gain. All runs are weak on lighting/spatial stress (50–70% acc on stressed reals).

### 2026-04-25 promotion-contract scorecard readout (job `2162379556655202304`, asia-southeast1)

This is what makes P8A muddled.

#### Real-pool FPR (lower is better, default τ)

| Slice | RLP6_04 | RLP7_05 | **P8A** |
|---|---:|---:|---:|
| `teams_real_all_dev` (3253) | 15.92% | 13.74% | **12.11%** |
| `teams_real_poor_quality_dev` (923) | 13.33% | 10.83% | **8.13%** |
| `teams_real_lighting_extreme_dev` (1401) | 15.85% | 12.63% | **11.13%** |

P8A leads all three. The harder the slice, the bigger the relative improvement (`poor_quality_dev` shows a 39% relative FPR reduction).

#### Per-method fake recall (`teams_fake_all_dev`, 2409 videos, default τ)

| Method | n | RLP6_04 | **P8A** | Δ |
|---|---:|---:|---:|---:|
| deeplive_enhanced | 545 | 79.6% | 53.0% | **−26.6 pp** |
| teams_flat_xiang_xiang2_feng | 135 | 78.5% | 45.2% | **−33.3 pp** |
| visomaster_enhanced_macro | 550 | 57.3% | 35.6% | **−21.6 pp** |
| teams_capture_noyn_sharker_s23 | 204 | 97.1% | 91.2% | −5.9 pp |
| teams_capture_cam_test_s35 | 244 | 97.5% | 94.3% | −3.3 pp |
| (12 other `teams_capture_*` methods) | varied | ≥98% | ≥98% | ~0 |
| **Aggregate** | **2409** | **83.9%** | **70.2%** | **−13.6 pp** |

#### Failure-mode analysis (`analysis/p8a_fake_failure_analysis_2026-04-25/`)

329 videos where RLP6_04 catches the fake but P8A misses, with **zero compensating gains**. ~80% of P8A's misses sit at scores 0.10–0.40 (with a substantial cluster <0.10), only ~19% in the τ-recoverable [0.40, 0.50) band. Implication: lockbox τ-tuning will recover at most ~3.8 pp of the 13.6 pp gap. The remaining loss is **separability collapse, not threshold drift**. Visomaster regression concentrates on `_teams` sub-pool (1.5× harder than `_raw`); the codec-pressure component of P8A's training appears to have specifically hurt the slice closest to the deployment distribution.

### Lockbox readout (P9_MID_FLIGHT_HANDOFF readback, 2026-04-26)

- P8A: `lockbox_real_fpr=0.147%`, `lockbox_fake_recall=23.7%`, `viso_macro=1.1%`, `deeplive_enh=2.4%`, τ=0.991.
- RLP6_04: `lockbox_real_fpr=0.441%`, `lockbox_fake_recall=23.3%`, `viso_macro=3.3%`, `deeplive_enh=6.4%`, τ=0.992.

P8A roughly **halves** lockbox real FPR (0.441% → 0.147%) at a τ that is also 0.991 (close to but below the 0.995 τ-tail-collapse signature). Lockbox aggregate fake recall is similar (23.7% vs 23.3%). The per-method recall regressions on `viso_macro` and `deeplive_enh` are the load-bearing failure: P8A trades **deployment FPR (good)** for **per-method fake recall (bad)** on the methods we care about most.

## Conclusions drawn in-session

- **P8A is the first run to break the P7 anchor ceiling** at single-variable cost. Anchor Δ −0.188 vs best P7 −0.097 ≈ 2× improvement. Same data, same aug, same schedule, same LR — only the three unfreeze flags moved.
- **Backbone freeze was reach-limited, not the camera/ISP shortcut being uniquely difficult.** Memory `project_p8a_breakthrough.md`: "head + attention-only SVD wasn't enough degree of freedom." Unfreezing 2–3 CLIP visual modules + adding SVD residuals to MLP gave enough redistribution room.
- **P8B refuted "the shortcut is in the weight chain"** — scratch on plain CLIP performed *worse* than RLP6_04 baseline. Anchor Δ +0.066 at step 11000 / 30k. Confirms the shortcut lives in the data mix, not in R12g/R13 weights. The CLIP-DataComp-XL prior preserved by FT is load-bearing as a regularizer that scratch lacks. **Implication**: data-side intervention is the only path to *eliminate* the shortcut, but P8A's recipe goes a long way without one.
- **Anchor pool is now bimodal**, not flat-ceilinged. Mean 0.74, std 0.27 — 13/30 frames still flip > 0.9, 17/30 are pulled below. Frame-level analysis shows pinned and escaped frames cluster temporally in a 6-second clip — pinned early, escaping late. The model has the capacity to escape; content variation already shakes some frames loose. Read: "the shortcut is no longer reach-bound, it is data-bound."
- **Roee-mac is mostly solved** — 23/30 frames decisively correct, mean 0.35 vs baseline 0.75. Cross-camera generalization confirmed.
- **2026-04-25 reframe: P8A's regression is separability loss, not threshold drift.** Failure-mode analysis shows zero compensating wins; bimodal P8A miss distribution sits below the τ-recoverable band; per-method regressions concentrate on `_teams` sub-pool of visomaster (the deployment-distribution slice). Lockbox τ-tuning will help marginally; will not fix this. **Implication**: Packet-9 must be a *softened* P8A, not "P8A more / longer / harder."
- **`value_composite=0.99+` on a checkpoint that regresses fake recall by −13.6 pp aggregate is the concrete proof of `value_composite`'s deployment-blindness.** See [`value_composite_semantics`](../threads/value_composite_semantics.md). The regression is invisible at the trainer composite layer because the composite's training-time fake pools are dominated by `teams_capture_*` methods which P8A still solves at ≥98%.
- **`backbone_lr_mult` plumbing landed** (commit `0f2f342`): optional `optimizer.adam.backbone_lr_mult` field applies a per-group multiplier to unfrozen-backbone parameters (SVD residuals + visual.proj + visual.ln_post) while keeping the head at base LR. Default 1.0 (preserves prior behavior). Foundation for P9_01's "softened P8A".
- **`real_codec_uplift` plumbing landed** (commit `ebce585`): augmentation router now supports a real-side codec uplift flag. Foundation for P9_05's data-axis hypothesis (codec exposure asymmetry).
- **State-of-detector doc landed** (`docs/relaunch_handoffs/R13_RELAUNCH_STATE_OF_THE_TEAMS_DETECTOR_2026-04-25.md`): full readout of P7 matrix + P8A/B; pre-Packet-9 status. The R13_FULL_STORY second-opinion doc (`docs/relaunch_handoffs/R13_FULL_STORY_PRE_PACKET_9_2026-04-25.md`) reframes the question as "is the right next packet a softened P8A, a return to RLP7_05, or a different intervention class entirely?" — explicitly inviting external pushback.
- **Session IDs**: convmem coverage thin for the P8A scorecard session; primary traceability via the two state-of-detector docs, the failure-mode analysis dir, and commits `d047da9`, `d5be7ce`, `1573c59`, `856ea08`, `8a9ae90`, `e326b81`, `3426bdf`, `0f2f342`, `f766ccd`, `ebce585`.

## Retrospective (as of 2026-04-26)

**Packet is muddled, not a single-verdict win.** P8A is the first checkpoint that **simultaneously** improves real-pool FPR substantially **and** breaks the camera-signature anchor ceiling, but it does so at the cost of −13.6 pp aggregate fake recall on the deployment-distribution methods (`visomaster_enhanced_macro`, `deeplive_enhanced`, `teams_flat_xiang_xiang2_feng`). The P9 crowning protocol explicitly excludes P8A on per-method recall floors (the `deeplive_enhanced ≥ 3.4%` and `visomaster_enhanced_macro ≥ 0.3%` floors are computed against RLP6_04 minus 3pt slack — P8A's 2.4% / 1.1% per-method numbers fall below the floors).

**Retroactive in_proj-SVD correction (2026-04-26, commit `2feea58`).** Every P8A claim that depended on "in_proj-SVD residuals are training" is imprecise — the q/k/v residuals received zero classification gradient through the entire run. The lever set was MLP-SVD + visual.proj + ln_post unfreeze; the in_proj-SVD piece contributed only via regularizer drift. The headline anchor improvement is real (the checkpoint exists, the rescores ran on it, the numbers stand); the *attribution* to "more reach" needs the in_proj piece subtracted. The Phase C overnight slate's C-ablation slot (image 1.3.218, commit `6665910`) tied with C.1 within noise on the P10_SYM-on-P8A recipe — confirms in_proj-SVD does not materially move the canonical P8A configuration. See [`in_proj_svd_gradient_bug`](../threads/in_proj_svd_gradient_bug.md).

**Two structural shifts driven by this packet.**

1. The team's mental model of "where is the shortcut" updates. P7 ceiling at −0.10 read as "the FT chain has bounded reach"; P8A's −0.188 says "the FT chain had bounded *parameter set*; with more parameters the same data + same aug + same schedule moves further." Combined with P8B's failure (scratch is worse), the framing converges: **the shortcut is data-bound; capacity helps; scratch is wrong**.
2. `value_composite` is operationally demoted to "directional, not deployment-grade." The R13_FULL_STORY doc names this rule for the first time; the P9 crowning protocol excludes it from the gate set; both `value_composite` and `ood_composite` should be quoted side-by-side at in-flight reporting. See [`value_composite_semantics`](../threads/value_composite_semantics.md).

**Preprocessing-parity note.** P8A scorecard numbers are **post-fix** (commit `855871e` landed in Packet-7 before P8A). The anchor-pool readout at `analysis/pool_rescore_rlp8_a.summary.json` runs through the fixed `INTER_LINEAR` path. RLP6_04 and RLP7 baselines used for comparison are also post-fix (re-scored against `rlp6_04_postfix_rescore_2026-04-24.summary.json`).

**Open workstreams as of 2026-04-26.** P9 in-flight (5 single-variable + 5 longshot variants); promotion-contract scorecard `2877145196556976128` running; P10 anti-shortcut packet (symmetric router + GRL slate, commit `2c9778b`) drafted on the assumption P8A is the right base but trained too aggressively. The "are we wrong about the codec aug being part of the problem" open question (`R13_FULL_STORY_PRE_PACKET_9_2026-04-25.md:252`) remains live: P9_05 (real-codec uplift) is the data-axis test.

**Cross-references.** Story continues in [P9](P9.md), [P10](P10.md), [`processing_signature_shortcut`](../threads/processing_signature_shortcut.md), [`value_composite_semantics`](../threads/value_composite_semantics.md), [`in_proj_svd_gradient_bug`](../threads/in_proj_svd_gradient_bug.md), [`promotion_contract_evolution`](../threads/promotion_contract_evolution.md).

## Source files

- **Handoffs**:
  - `docs/relaunch_handoffs/R13_RELAUNCH_STATE_OF_THE_TEAMS_DETECTOR_2026-04-25.md` — pre-Packet-9 state of the detector with P7 matrix, P8A/B detail, anchor table, OOD aug-stress trajectory, Packet-9 candidate scoring.
  - `docs/relaunch_handoffs/R13_FULL_STORY_PRE_PACKET_9_2026-04-25.md` — second-opinion narrative with the failure-mode analysis embedded; the document that names "value_composite is not deployment-grade" verbatim.
  - `docs/relaunch_handoffs/PACKET_9_MID_FLIGHT_HANDOFF_2026-04-26.md` — handoff written mid-flight that summarizes P8A's trade and pre-commits the P9 crowning protocol that excludes P8A.
- **Yamls**: `experiments/phase2_round13/R13_RLP8_01_unfreeze_clip_codec.yaml` (P8A); `experiments/phase2_round13/R13_RLP8_02_fresh_head_plain_clip.yaml` (P8B).
- **Code touchpoints**: `utils/setup.py::choose_optimizer` — `backbone_lr_mult` plumbing (commit `0f2f342`). `data/augmentations/pipelines.py` — `real_codec_uplift` plumbing (commit `ebce585`). `detectors/effort_detector.py` — backbone unfreeze flags around `:736`, `:744` per memory.
- **Scorecards / analysis**:
  - `analysis/pool_rescore_rlp8_a.summary.json` — P8A anchor pool rescore (leader checkpoint).
  - `analysis/pool_rescore_rlp8_a_step2500.summary.json` — P8A step 2500 anchor (confirmed step 5000 is the leader; commit `3426bdf`).
  - `analysis/pool_rescore_rlp8_b.summary.json` + `_step5000.summary.json` — P8B anchor (worse than RLP6_04).
  - `analysis/p8a_fake_failure_analysis_2026-04-25/` — 329-video regression dir with `analyze.py`, `summary.json`, `regressed_videos.csv`. The load-bearing evidence for "separability collapse, not threshold drift."
  - `analysis/overnight_packet7_packet8_summary_2026-04-25.md` — full overnight readout combining P7 matrix and P8A/B detail.
  - `gs://training-job-outputs/test_results/teams_promotion_contract/p8a-review-scorecard-20260425/reports/` — promotion-contract scorecard output dir (job `2162379556655202304`).
- **Reference checkpoints**: P8A `gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth`. P8B `gs://training-job-outputs/phase2r13_experiments/n8yk2hox/...` (step 11000 — partial, hung at 12000).
- **Commits**: `d047da9` (P8A/B yamls drafted), `d5be7ce` (image 1.3.206), `1573c59` (P7/8 anchor rescores + state-of-detector), `856ea08` (P8A scorecard inputs), `8a9ae90` (image 1.3.207 = P8A scorecard), `e326b81` (state-of-detector with W&B per-method + OOD stress slices), `3426bdf` (P8A step 2500 ood_composite anchor rescore), `0f2f342` (`backbone_lr_mult` option), `f766ccd` (R13 full-story pre-Packet-9 doc), `ebce585` (`real_codec_uplift` flag).
- **Memory pointers**: `project_p8a_breakthrough.md` (the breakthrough; with 2026-04-26 CORRECTION on in_proj-SVD); `project_signature_shortcut_finding.md` (90/90 not threshold-reachable on RLP5_07; same shortcut family); `project_shortcut_is_upstream.md` (with 2026-04-26 CAVEAT on FT-only ceiling re-validation needed); `project_promotion_contract.md` (lockbox is the deployment readout); `feedback_no_cancelling_vertex_jobs.md` (P8B cancellation incident learning); `feedback_decision_points.md` (Packet-9 path is a user call).
