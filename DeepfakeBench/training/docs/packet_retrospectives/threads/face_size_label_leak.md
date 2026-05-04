# Thread: Face-pixel-area label leak

> **Slice 6 finding**: each fake method clusters at a near-deterministic face-pixel-area band; reals span a wider range. The model can use face size as a direct fake-prediction shortcut. This is the camera-signature shortcut viewed from a different axis. Memory `project_face_size_label_leak.md` is the auto-memory anchor; this thread is the deliberated synthesis.

## The question

Does training data have a face-pixel-area distribution that lets the detector predict the fake label from the geometric crop tightness of the face alone — independent of any artifact-content signal? If yes, every fake-method's frames carry a near-uniform face-size signature (because each pipeline crops at fixed tightness) while real frames span a wider range, and the detector's "fake recall" is partly an artifact of the data layout rather than the manipulation surface.

## Initial belief

Through Slices 1–5 the team treated face crop size as **deployment-domain-relevant** rather than **leakage-relevant**. The "modern subset" filtering work (Slice 6 morning, `analysis/modern_lockbox_v2_2026-04-27/`) was scoped as a deployment-conditioning question — *"what subset of the lockbox actually represents Teams production capture?"* — not as an audit of label leakage in training. The processing-signature shortcut ([`processing_signature_shortcut`](processing_signature_shortcut.md)) was framed as a camera/ISP signal: codec, color profile, sharpness, ISP-specific noise patterns. Crop tightness was on the list of suspected confounders but had not been measured.

## What changed our mind

- **2026-04-27 21:30 CEST — Crop-tightness brittleness CONFIRMED, label-independent** (`april-26-training-master-plan-v2.LOG.md:750-817`). The user reported that the same `deeplive_dor` frame scored prob_fake=0.30 at native crop but 0.98 when manually tightened — a 68pp swing. The agent ran `analysis/crop_shortcut_2026-04-27/crop_sweep.py` + `population_sweep.py` over 47 unique frames × 5 tightnesses ≈ 250 check-frame predictions. Findings (LOG `:767-795`):
  - 25 of 47 frames (53%) flipped the predicted label across the tightness sweep — severe model brittleness, label-independent, not a clean axis (per-frame valleys scattered across t∈[0.7, 1.5]).
  - Production-honest pool sensitivity, 30 frames × 5 tightnesses — FAIL regime sits in a "FAKE valley" at native crop: any perturbation away from native (loosening or tightening) drops FPR ~30-40pp; OK regime is robust at any tightness.
  - Per-FAIL-tag breakdown is uneven: 2 of 3 FAIL tags are largely a crop-shortcut artifact; the third uses a non-crop signal (likely a webcam-specific signature).

- **2026-04-27 22:40 CEST — Discovered third shortcut: face-pixel-area training data leak** (`april-26-training-master-plan-v2.LOG.md:825-826`). Per-method face-size signatures measured on `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` (n=7,334 dev+lockbox):
  - `deeplive_enhanced` (n=545): 100% of frames in **22.6-24.6k px²** bucket — single-point distribution.
  - `teams_capture_pc_generator_s4` (n=30): median 144k px².
  - `teams_capture_test_cam_s76` (n=138): median 154k px².
  - `teams_capture_noyn_sharker_s23` (n=324): median 14k px².
  - Dev REAL (n=3,832): face_pixel_area median 24,061 px², p90 93,735 px².
  - Dev FAKE (n=2,444): face_pixel_area median 40,349 px², p90 121,148 px².
  - Cohen's d ≈ 0.37 between dev_real and dev_fake — moderate label leak.
  - 5–10k px² bucket: 7.4% of reals, 0.6% of fakes (12× under-rep).
  - 75–100k px² bucket: 6.5% of reals, 11.1% of fakes (1.7× over-rep).
  - **A frame with face >75k is ~1.7× more likely to be FAKE in training; a frame <10k is ~12× more likely REAL.** This means the model can identify the fake method (and thus the fake label) from face size alone in many cases.

- **2026-04-27 22:40 CEST — Production-honest FAIL frames sit in fake-rich face-area zones** (memory `project_face_size_label_leak.md`):
  - `dor-real-webcam-false-flag` (FAIL): median 58k → in the 50-75k bucket where fake/real ratio = 1.46×.
  - `roee-mac-laptop-false-flag-virtual-bg` (FAIL): median 45k → in the 30-50k bucket where fake/real ratio = 1.39×.
  - `dor-real-laptop-correct-no-virtual-bg-whiteish` (OK): median 36k → 30-50k bucket. Same bucket as a FAIL — and it works because the visual content matches "real-Dor" enough to override the bucket prior. So the bucket signal is real but not dominant.

- **`A.ShiftScaleRotate(scale_limit=0.50)` empirically NOT inert on already-cropped data** (`april-26-training-master-plan-v2.LOG.md:826`). Deeplive face_pixel_area span goes from 23-27k baseline → 7-50k under aug, dissolving per-method signatures (per-method spread / median 0.50). Yaml-only change to existing `context_variation_scale` knob (currently default 0.10 in P8A, plumbed at `pipelines.py:1037`).

- **2026-04-27 23:13 CEST — P11 portfolio incorporates face-size axis disruption.** P11_MILD (ctx_scale 0.30) and P11_HEAVY (ctx_scale 0.50) directly attack the face-size leak via training-time symmetric crop variation. P11_HEAVY is the main test of the heavy variant. See [P11 retro](../packets/P11.md).

- **2026-04-28 morning — Anti-shortcut interventions for P13 include face-size mitigation.** Commit `cab2909` adds `face_scale_jitter` as a top-level yaml block (alongside `anchor_aware` loss + pipeline-random aug). The face-scale-jitter block was the response to the face-size leak finding. **The block silently failed on P13_FROM_SCRATCH** (the wandb-flattening bug; see [`wandb_flattening`](wandb_flattening.md)) — i.e., the intervention designed to break the face-size leak was itself blocked by a separate "added-but-not-firing" bug. The fix lands as commit `c366026`; the actual P13_FROM_SCRATCH run with face-scale-jitter live is Slice 7's verdict.

## Current stance (2026-04-29)

Face pixel area is a **dual-role variable**: deployment-domain-relevant AND a training-data label leak. Three findings converge on this stance:

1. **The leak is empirically present in training data**: each fake method clusters tightly (deeplive_enhanced is a near-single-point at 22.6-24.6k); reals span wider; Cohen's d ≈ 0.37; bucket-distribution leak ranges from 12× under-rep (small reals) to 1.7× over-rep (large fakes).
2. **The leak is exploited by the model**: the crop-tightness sweep flips 53% of frame predictions; production FAIL frames sit in fake-rich face-area zones; FAIL regime is in a "FAKE valley" at native crop with brittle escape under perturbation.
3. **The face-size leak is the same shortcut from a different axis as the camera-signature shortcut**. Memory `project_face_size_label_leak.md` says so explicitly: *"This is the same shortcut from a different axis. Confirmed empirically by the crop-sweep findings: same person flips real↔fake under tightness perturbation alone."*

The lever is symmetric crop / face-scale jitter aug applied at training time. P11_HEAVY (`context_variation_scale: 0.50`) is the first attempt. P13's `face_scale_jitter` block is the second (post the Slice-6 wandb-flattening fix `c366026` that finally let the block fire).

### Slice 7 update (2026-04-29) — face_scale_jitter fired live in P13 post-fix; insufficient on its own

Per `docs/relaunch_handoffs/P13_DAY4_VERDICT_2026-04-29.md`: P13_FROM_SCRATCH ran post-`c366026` with `face_scale_jitter` ENABLED at trainer init (verified in init logs). The Day-4 verdict γ shows: (a) Axis 3 shortcut Δ moved directionally on best ckpt (P13_step18000 = 0.520 vs P8A 0.659, ~14pp improvement), (b) Axis 3 Δ is still 3.5× the 0.15 gate, (c) cross-domain capability collapsed (the from-scratch trade). **face_scale_jitter as deployed in P13 is necessary-but-not-sufficient on its own.** The close criterion below — "Cohen's d ≤ 0.10 + tightness-sweep flips ≤ 10%" — has not been re-measured against the P13 candidate; that audit is queued for the post-P14 (or post-P14_DATA_FIX) verdict. **Loop stays open**; the `face_scale_jitter` block firing this time is a precondition for the next measurement, not the measurement itself.

A separate question — **whether bucket-balanced sampling** (explicitly equalizing the marginal P(fake | face_pixel_area)) is a higher-leverage lever than scale-aug — is still open and not yet tested.

### 2026-04-30 afternoon update — Direct measurement: jitter@0.50 did NOT close the leak; mechanism of the value_composite win is unknown

The face-size invariance probe authored 2026-04-30 morning (`analysis/face_size_invariance_2026-04-30/face_size_invariance.py`) ran the same afternoon on the broader 180-frame production cache (`analysis/deployment_honest_eval_2026-04-27/_prod_cache/frames/`, not the 47-frame audit subset — so absolute flip-rate magnitude is not directly comparable to the 53% baseline; the cross-checkpoint comparison on the same 180 frames is the canonical readout). All four overnight ckpts evaluated at tightness grid `[0.7, 0.85, 1.0, 1.15, 1.5]`:

| Ckpt | flip rate | median \|Δprob_fake\| | Reading |
|---|---:|---:|---|
| P8A baseline (`9lmvb5b4` step 5000) | 39.4% | 0.128 | reference |
| **`mclioexb` (jitter@0.50, the WINNER)** | **43.9%** | **0.303** | **WORSE** — curve 2.4× steeper than P8A |
| `w5tky6ss` (P15 GRL) | 22.8% | 0.124 | better (incidental) |
| `xan4dfto` (DATA_FIX) | 22.8% | 0.143 | better (incidental) |

Source artifacts: `analysis/face_size_invariance_2026-04-30/outputs/{p8a,slot3_mclioexb,slot2_w5tky6ss,datafix_xan4dfto}_invariance.{csv,png}`; per-frame predictions and tightness sweep curves saved.

**The hypothesis that jitter@0.50 closed the face-pixel-area shortcut is REFUTED.** Jitter alone — the supposed face-size lever — actually made the model *more* sensitive to crop tightness on production-honest frames, not less. GRL and DATA_FIX (neither designed to target face-size) incidentally halved the flip rate. The 5.7× value_composite gap from jitter-isolated remains real but its mechanism is no longer attributed to face-size invariance on this substrate.

The triptych t-SNE visualization (`analysis/embedding_triptych_2026-04-30/outputs/triptych_p8a_slot2_slot3/triptych_grid_tsne.png`) corroborates: the face_pixel_area-bucket coloring shows similar quartile structure across all three ckpts' [CLS] embeddings, with no row showing the size buckets bleeding together. The geometry of the leak in [CLS] space is conserved across interventions.

Three implications for this thread:

1. **The 2026-04-30 morning update's "indirect evidence is strong" framing is invalidated by the direct measurement.** What jitter@0.50 invariantizes (relative to jitter@0.25) is not face-pixel-area on production-honest frames. Possibilities for the actual mechanism: (a) jitter@0.50 invariantizes a *different* axis (e.g., aug-induced texture noise) that correlates with the value_composite components but not with crop-tightness on this substrate; (b) the production-honest 180 frames have a narrower face-size range than what jitter@0.50 was trained against, so the test is on the wrong substrate to detect what jitter learned; (c) jitter@0.50 changes calibration / decision-boundary geometry without changing the underlying [CLS] feature manifold.
2. **The face-size leak's close criterion's flip-rate half is now affirmatively NOT MET on the leader.** The leader checkpoint `mclioexb` registered 43.9% > 10% close-criterion threshold. The Cohen's d half on per-method predictions has not been re-measured because the in-vivo flip-rate failure already invalidates the close-criterion conjunction. Loop stays open with this affirmative no-close evidence recorded.
3. **The face-size leak is the same shortcut from a different axis as the camera-signature shortcut** — that framing (from memory `project_face_size_label_leak.md` and `processing_signature_shortcut`) survives because the 04-27 audit's mechanism (each method's frames come from a fixed pipeline that crops at fixed tightness) is structurally about training data, not about the trained model. The night's learning is that fixing a model with input-space perturbation (jitter@0.50) does not in itself bring per-frame predictions to invariance under crop-tightness perturbation at inference. The leak in *training data* persists; the model *trained against* that leak via jitter has not learned crop-tightness invariance on production-honest frames.

The mechanism of the night's value_composite ranking is now an open question. New thread [`jitter_winner_mechanism_unknown`](jitter_winner_mechanism_unknown.md) (2026-04-30 afternoon) tracks the puzzle and the candidate next probes (intermediate-layer features, calibration-only diff, training-substrate face-size sweep, etc.).

### 2026-04-30 update — face_scale_jitter@0.50 is the load-bearing lever (indirect evidence)

P14_FACE_SCALE_JITTER_ISOLATED (`mclioexb`, seed 2273, FT-from-P8A + face_scale_jitter@0.50 ONLY — anchor_aware DISABLED, pipeline_random DISABLED) won the overnight slate at trainer-side `value_composite=0.661`, beating the P14 bundle (0.116) and P14_DATA_FIX (0.126) by 5.7×. Cross-method generalization preserved (`other_fakes_tpr=0.591` vs the bundle's 0.020). Source: W&B summary on `dtect-vision/phase2-experiments/runs/mclioexb`; checkpoint `gs://training-job-outputs/phase2r13_experiments/mclioexb/value_composite_effort_20260429_step500_auc0.9797_eer0.0521.pth`.

This is **strong indirect evidence that the face-pixel-area shortcut was the load-bearing problem in the bundle** — jitter at scale_limit=0.50 (matching the t∈[0.7, 1.5] tightness range that flipped 53% of frames in the 04-27 audit) finally has enough range to bite. But it is NOT the close-criterion measurement: the close criterion below — Cohen's d ≤ 0.10 on per-method dev_fake vs dev_real face-pixel-area distributions AND tightness-sweep flips ≤ 10% on FAIL frames — requires direct re-measurement on the new ckpt. That measurement is authored at `analysis/face_size_invariance_2026-04-30/face_size_invariance.py` (sub-agent 2026-04-30) and pending execution; the script re-runs the 47-frame × 5-tightness sweep on Slot 3 vs P8A, producing the flip-rate comparison.

Three implications for this thread:

1. **Jitter at 0.25 (the bundle's strength) was undertrained against the leak**; jitter at 0.50 was the right dose. Future packets attacking the face-size leak should use scale_limit ≥ 0.50 by default unless a specific reason to back off.
2. **The bundle was net-negative against jitter alone**, motivating the new [`anti_shortcut_bundle_decomposition`](anti_shortcut_bundle_decomposition.md) thread. The face-size leak's close criterion does not depend on bundle composition; what changed is the operational rule about how to *test* the leak going forward (single-lever ablation slot required when stacking).
3. **Loop stays open until direct measurement lands.** The 5.7× value_composite lift licenses *expecting* the close criterion to be hit, but does not itself satisfy it. The flip-rate audit is the load-bearing measurement.

### Post-Slice-7 update (2026-04-29 afternoon) — face-area thresholds miss small-source-image cases

A 2026-04-29 morning visual audit of eval-substrate slices (the very-sharp-FP and is_no_face slices from the 2026-04-27 investigation; see [`eval_production_crop_tightness_gap`](eval_production_crop_tightness_gap.md), [`sharpness_metric_bug`](sharpness_metric_bug.md), [`eval_substrate_data_hygiene`](eval_substrate_data_hygiene.md)) surfaced a parallel cut on the same substrate: **the eval pool contains 99×110-pixel source crops** (e.g. `PC_Generator__s15_1754.4_frame_058496_crop_002__d83b853d.jpg`) that pass the face-area floor (`face_pixel_area > 1k` floor in routine analyses; `face_pixel_area > 57k` quick-win bound in the 2026-04-27 investigation) but fail any reasonable source-resolution bound like `min(width, height) >= 200`. The 99×110 frame's full image is ~10.9k px², so a face_pixel_area inside it sits at roughly the same magnitude as the bucket boundaries this thread's per-method analysis uses. Implication for this thread: **face-pixel-area thresholds are not a complete eval-scope bound** — a source-resolution floor is a different axis that should compose with the existing face-area filter rather than replace it. The two findings together (face-pixel-area leak in training data; small-source-image presence in eval substrate) bear on the same set of FPR/recall numbers but address different mechanisms.

Additionally, the audit's structurally upstream Finding 5 — the eval-vs-production crop-tightness gap (see [`eval_production_crop_tightness_gap`](eval_production_crop_tightness_gap.md)) — bears on this thread's mechanism in a load-bearing way. **The face-size leak's root cause is "each method's frames come from a fixed pipeline that crops at fixed tightness"**; if the eval substrate's crop tightness does not match production's, then closing the leak at the eval-substrate level may leave a residual gap when the model deploys to tighter production crops. The Cohen's d ≤ 0.10 close criterion in the open loop below is measured on the eval substrate; whether hitting that target translates to the same delta on production-tightness crops is now a separately-open question. **Do not unilaterally rewrite this thread's close criterion** — but a future agent should read it knowing the eval substrate has a known crop-tightness mismatch with production.

## Packet timeline

- [P10](../packets/P10.md) — anti-shortcut packet authored 2026-04-26; the symmetric-router half ran on Phase C and was tied within noise. Did not yet have the face-size axis isolated as a separate intervention class.
- [P11](../packets/P11.md) — first packet to incorporate face-size axis disruption as a training-time lever (`context_variation_scale` 0.30 / 0.50 across MILD / HEAVY variants). Verdict β at end of Slice 6; HEAVY recipe direction confirmed but step-1000 checkpoint policy artifact distorts the readout.
- [P12](../packets/P12.md) — P12_HEAVY_LONG was the long-schedule HEAVY recipe attempt; periodic_saves silent failure means no usable mid-step checkpoints landed. P12 dud.
- [P13](../packets/P13.md) — `face_scale_jitter` block introduced (commit `cab2909`); silently DISABLED by wandb-flattening bug; fixed in `c366026`. Slice 7 carries the post-fix verdict.

## Evidence locations

- `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` — n=7,334 dev+lockbox tagged frames; the parquet that supports all face-size statistics in this thread.
- `analysis/crop_shortcut_2026-04-27/{crop_sweep.py, population_sweep.py, synthesize.py, synthesis.md}` — 47-frame crop-tightness sweep; 53% prediction-flip rate.
- `analysis/crop_shortcut_2026-04-27/p8a_lockbox_join_2026-04-27.csv` — P8A predictions × R9A-tagged parquet join used for the face-size filter sweep that produced the +41pp Pareto lift at FPR=5% (`april-26-training-master-plan-v2.LOG.md:838-857`).
- `data/augmentations/pipelines.py:1037` — `context_variation_scale` plumbing (existing yaml knob; default 0.10 in P8A).
- `experiments/phase2_round13/R13_P11_HEAVY.yaml` — first packet variant to set `context_variation_scale: 0.50` as the heavy face-size disruption lever.
- `experiments/phase2_round13/R13_P13_FROM_SCRATCH.yaml:88-91` — `face_scale_jitter` nested block (`enabled: true`, `scale_limit: 0.25`).
- `data/augmentations/face_scale_jitter.py` — new module added by `cab2909` (face-size canonicalization aug).
- Master plan LOG: `april-26-training-master-plan-v2.LOG.md:825-857` — the 04-27 22:40 entry where the face-pixel-area leak was discovered and the +41pp filtered Pareto lift was measured.
- Memory: `project_face_size_label_leak.md` — auto-memory anchor with the per-method face-size signatures and bucket-distribution leak numbers.
- Commits: `cab2909` (P13 anti-shortcut interventions including face_scale_jitter), `c366026` (wandb-flattening fix that lets face_scale_jitter actually fire).

## Open loops

### Open loop: face-size-label-leak
status: open
severity: high
first_seen: 2026-04-27
last_verified: 2026-04-30
close_criterion: a downstream packet demonstrates that with a face-size-targeted intervention actually live (face-scale-jitter, symmetric crop-aug, bucket-balanced sampling, or a structurally-different lever yet to be designed), per-method face-pixel-area distributions overlap (Cohen's d on dev_fake vs dev_real ≤ 0.10), AND a tightness-sweep at inference flips ≤ 10% of frame predictions on the production-honest cache — i.e., the leak is no longer exploitable at the size of effect that the 04-27 sweep demonstrated. As of 2026-04-30 afternoon the flip-rate half is affirmatively NOT MET on the strongest available leader (`mclioexb` jitter@0.50: 43.9% on 180 frames; P8A baseline 39.4% on the same 180 frames).

The leak is empirically present and exploited by the model. Four interventions tested so far: P11_HEAVY's `context_variation_scale: 0.50` (training-time symmetric scale aug; verdict β with checkpoint policy artifact distorting readout), P13's `face_scale_jitter@0.25` (verdict γ — bundle-level, cross-domain collapsed), P14_FACE_SCALE_JITTER_ISOLATED's `face_scale_jitter@0.50` (2026-04-30 — leader of overnight slate at trainer-side value_composite=0.661 but **flip-rate close criterion FAILED on direct measurement: 43.9% > 10% threshold**), and P15_GRL_FROM_P8A's `quality_domain_loss_weight=0.20` (incidentally cut flip rate to 22.8% — not a face-size-targeted lever and still > 10% close-criterion threshold; structurally interesting because GRL was not designed to attack this axis). **Severity: high** because the leak directly invalidates "fake recall on lockbox" claims when the lockbox face-size distribution differs from training; this is structurally adjacent to the deployment block and can confound any post-fix readout that doesn't audit the face-size distribution explicitly.

**2026-04-30 afternoon status note**: indirect evidence framing from the morning (the 5.7× value_composite lift of jitter@0.50 alone vs the bundle) is **REFUTED by direct measurement**. The flip-rate component of the close criterion is now affirmatively NOT MET on the strongest available leader (43.9% on `mclioexb`, worse than P8A's 39.4% on the same 180-frame substrate). Cohen's d half intentionally not re-measured — the conjunction is already broken. The next-packet design discussion needs to treat "jitter@0.50 closes the face-size leak" as REFUTED, even though jitter@0.50 wins value_composite by 5.7× via some other mechanism. See companion thread [`jitter_winner_mechanism_unknown`](jitter_winner_mechanism_unknown.md) for the open-question tracking. Source artifacts: `analysis/face_size_invariance_2026-04-30/outputs/*_invariance.{csv,png}` (per-frame predictions and curves for all 4 ckpts) and `analysis/embedding_triptych_2026-04-30/outputs/triptych_p8a_slot2_slot3/triptych_grid_tsne.png` (face_pixel_area-bucket geometry conserved across the 3 ckpts).

### Cross-thread refs

- [`processing_signature_shortcut`](processing_signature_shortcut.md) — face-size leak is the same shortcut from a different axis as the camera-signature shortcut (memory `project_face_size_label_leak.md`). The `shortcut-deployment-block` (critical, in-progress) loop in that thread interlocks with this one — both close on the same kind of evidence (a packet that breaks the per-pool false-flag pattern).
- [`wandb_flattening`](wandb_flattening.md) — the P13 `face_scale_jitter` block was silently DISABLED by the wandb-flattening bug; this thread's intervention designed to break the leak was itself blocked by an "added-but-not-firing" bug class. The two threads' verdicts are coupled in Slice 7.
- [`webcam_fpr_dominance`](webcam_fpr_dominance.md) — modern_lockbox_v2 filter (Slice 6 morning) drops `webcam` mode + tiny crops; both filters interact with face_pixel_area as a confound. Memory `project_lockbox_fpr_dominated_by_webcam_mode.md` notes the v2 number is what's deployment-relevant, but face-pixel-area is the leakage axis underneath the deployment-domain question.
- [`promotion_contract_evolution`](promotion_contract_evolution.md) — any "fake recall" number from the contract scorecard inherits the face-size leak unless the eval pool is stratified by face_pixel_area; this is a *latent* contamination of every contract readout pre-fix.
