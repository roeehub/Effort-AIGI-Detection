# R13 Full Story — Pre-Packet-9 State of the Teams Detector

**Date**: 2026-04-25
**Branch**: `teams-relaunch-root-2026-04-17`
**Author**: Claude Opus 4.7 (Anthropic), in collaboration with Roee Dar
**Purpose**: Full self-contained narrative of the Teams deepfake detector program — the model, the loss, the target, the data, the failure modes, and every experiment so far. Written as a second-opinion document: an external reviewer with no prior context should be able to read this, understand the situation, and push back productively on our current direction.

> **For reviewers**: I have explicitly flagged ⚠ next to assumptions or interpretations where the evidence is suggestive but not decisive. Push back hardest on those. The "Open questions" section at the end lists the specific places where a second opinion would be most valuable.

---

## 1. TL;DR

We are training a deepfake-detection model for use inside a Microsoft Teams deployment context. The base architecture is **Effort** — an OpenCLIP ViT-B/16 backbone (DataComp-XL pretraining) with an SVD-rank residual on its attention input projection and an ArcFace classification head. After a long packet sequence (R12g → R13 RLP6 → RLP7 → RLP8), our current best run (P8A) achieves a real-world camera-shortcut break: it cuts the false-flag rate on Roee's "anchor" pool — a single Dor-on-webcam clip the model historically over-flagged — by roughly half, with no regression on training-time fake recall. We just discovered (today, 2026-04-25) that **P8A's broader fake-recall has regressed substantially** on harder out-of-pool methods (deeplive, visomaster, xiang_xiang flat) — a 13.6 pp aggregate drop at default τ. Failure-mode analysis indicates this is **separability loss**, not a recoverable threshold shift. The next training packet (Packet-9) will be one or more *softened* P8A variants. The lockbox-calibrated promotion-contract scorecard is still running — its result will sharpen but probably not reverse the picture.

**The single decision we want second opinion on**: Given the failure-mode evidence, is the right next packet (a) a softened P8A, (b) a return to RLP7_05 territory with augmentation pressure as the main lever, or (c) a different intervention class entirely (e.g., an explicit fake-recall rescue arm via loss reweighting, or a fresh-head approach we haven't tried)?

---

## 2. The system

### 2.1 Target

Production target: a deepfake detector that runs inside a Microsoft Teams deployment to flag synthetic faces in video calls. Three success criteria, all load-bearing — improving any single one at the cost of the others is not acceptable:

1. **Fake recall on target methods** at chosen operating threshold τ (the families likely to be encountered in adversarial use: realtime swap tools like deeplive, visomaster, etc.).
2. **False-positive rate < 5%** on real Teams traffic.
3. **Robustness** across lighting, camera, codec, and color-pipeline variation typical of consumer hardware and Teams compression.

The promotion-contract scorecard (see §4) operationalizes (1) and (2). Robustness is checked through dedicated eval slices and the anchor pool (§5).

### 2.2 Architecture

The Effort detector (`detectors/effort_detector.py`):

- **Backbone**: OpenCLIP ViT-B/16, weights `ViT-B-16-DataComp-XL` from LAION (`datacomp_xl_s13b_b90k`). Resolution 224. Hidden size 512 (after the 768→512 final projection).
- **SVD residual on attention `in_proj`**: rank 736. Three trainable matrices `U_residual ∈ ℝ^(d×r)`, `S_residual ∈ ℝ^r`, `V_residual ∈ ℝ^(r×d)` are added on top of the frozen attention input projection per transformer block. ⚠ The choice of rank 736 is inherited from earlier R12g work; we have not re-validated it under the current data mix.
- **Backbone freeze policy**: by default the entire CLIP backbone is frozen except for the SVD residuals. Three optional flags (introduced in R12g, used in P8A) extend trainability:
  - `unfreeze_final_proj` — makes the final 768→512 projection trainable.
  - `unfreeze_final_ln` — makes the final LayerNorm trainable.
  - `apply_svd_to_mlp` — adds SVD residuals to the MLP `c_fc` / `c_proj` weights inside each transformer block (in addition to the default attention `in_proj` residuals).
- **Head**: ArcFace head with feature normalization. Margin `m = 0.15` (current). Scale `s` initialized and learned. Binary classification (real vs fake). Standard cross-entropy is used as the underlying loss — ArcFace's margin replaces the role of focal-loss-style hard-example weighting.
- **Auxiliary loss**: `quality_domain_loss` — a gradient-reversal head that classifies clip quality/codec domain. Its job is to *prevent* the backbone from latching onto codec/quality cues. Default weight 0.1 (configurable). Used because the program had a prior history of "shortcut" failures where the model learned to detect specific cameras/codecs rather than synthetic-face artifacts.

### 2.3 Training setup

- **Optimizer**: Adam, base LR 3e-5, weight decay 0.05, eps 1e-8.
- **As of 2026-04-25 (today)**, the optimizer also supports an opt-in `backbone_lr_mult` field that applies a per-group multiplier to unfrozen-backbone parameters (SVD residuals + visual.proj + visual.ln_post) while keeping the head at base LR. Default 1.0 (preserves prior behavior). This was added because P8A's regression analysis identified per-layer LR control as the missing knob to soften the unfreeze recipe; see §8.
- **Schedule**: cosine with warmup, 400 warmup steps, total 10,000 training steps. `nEpochs: 15`.
- **Batches**: 32 video clips × 8 frames each = 256 frames per batch.
- **Base checkpoint**: most R13 packets fork from `RLP6_04 step 23500` — `gs://training-job-outputs/phase2r13_experiments/h2pdu6i5/value_composite_effort_20260424_step23500_auc0.9942_eer0.0169.pth`. RLP6_04 itself forked from R12g.
- **In-training selection metric**: `value_composite` — a weighted combination of dev-set FPR (target mean ≤ 0.03, max-pool FPR ≤ 0.05) and per-method recall, with p95 stability jitter as a tiebreaker. ⚠ This is *not* deployment-grade — see §4.

---

## 3. The data

### 3.1 Real pool

The "real" class is composed of multiple identity-balanced sources, mixed via `data/sources/combined_paired.py` with a per-family weight (`combined_paired.sampling.family_weights`):

- **Teams real captures** (`teams_capture_*`, `realpool_real`) — the in-domain target distribution. Genuine Teams calls, varied subjects, varied cameras (consumer webcams, phone, laptop). *This is what we are protecting from false-flagging.*
- **df40 real partition** (`df40_real`) — the original DeepfakeBench training pool's real side. Down-weighted (0.4 in current mix) because it is *not* representative of Teams traffic and over-presence pushes the model toward irrelevant generalization.
- **External AVSpeech** (`external_youtube_avspeech`) — open-domain real videos. Used both directly and as the substrate for synthetic OOD-stress eval slices (§3.3).
- **Identity-specific personal recordings** (`dor_shkedi`, `roee` captures) — small high-importance sets used as *anchor pools* (§3.4).

### 3.2 Fake pool

- **df40 fakes** (`df40_fake`) — broad multi-method DeepfakeBench fake pool. Down-weighted (0.15) for the same reason as df40_real.
- **deeplive** family (`deeplive_non_enhanced_fake`, `deeplive_enhanced_fake`, `deeplive_teams_fake`) — realtime face-swap tool. Particularly important because it represents the kind of adversary likely to appear in Teams. Up-weighted aggressively (2.5–5.0).
- **visomaster** family (`proper_visomaster_clean_fake`, `proper_visomaster_teams_fake`, `proper_visomaster_enhanced_clean_fake`, `proper_visomaster_enhanced_teams_fake`) — high-quality offline face-swap tool with a realistic camera-pipeline post-processing stage. Up-weighted (1.0–2.5). The `_teams` variants are post-processed to mimic Teams codec; the `_enhanced` variants apply additional realism enhancement.
- **xiang_xiang flat** — a smaller method family.
- **teams_capture_*_s* fakes** — internal in-domain fake captures (different `_s##` indicate different production runs / setups).

### 3.3 Augmentation

`data/augmentations/pipelines.py` provides the augmentation system. Current configuration uses `quality_targeted_family` routing — different fake/real families get different augmentation strategies. The "codec-aggressive" preset used since RLP7_02 / P8A pushes codec-style degradation hard:

| Knob | P8A value |
|---|---|
| `webcam_codec_p` | 0.35 |
| `webcam_codec_quality` | [20, 65] |
| `quality_p` (JPEG/downscale aggregation) | 0.72 |
| `jpeg_lower` | 30 |
| `downscale_min` | 0.35 |
| `teams_codec_sim_p` | 0.40 |
| `teams_codec_sim_quality` | [20, 65] |
| `context_variation_enabled` | true |

The intent is to make the real pass robust to codec/quality variation by training through aggressive synthetic codec noise. ⚠ As of today's analysis, we suspect this aug pressure is part of what makes P8A's calibration shift "real-leaning" — by stressing the real pass hard in training, the model learns to confidently call quality-degraded inputs "real," which hurts when test fakes (visomaster especially) carry similar quality artifacts.

### 3.4 Evaluation suites

We evaluate on multiple slices:

- **Anchor pools** (small, ~30 frames, very high-signal): `dor-real-webcam-false-flag-no-virtual-bg`, `dor-real-webcam-false-flag-virtual-bg`, `roee-real-mac-no-virtual-bg`, `dor-real-correct-whiteish`, `dor-real-correct-mobile`, `roee-real-iphone-correct`. The **anchor metric** is the false-flag rate on `dor-real-webcam-false-flag-no-virtual-bg` — a single Dor-on-his-webcam clip the model has historically over-flagged. This pool is the canonical test bed for the camera-signature shortcut (§5).
- **Dev slices** (`teams_real_all_dev`, `teams_real_poor_quality_dev`, `teams_real_lighting_extreme_dev`): broader real-pool suites of 923–3253 videos. Used for FPR estimation pre-deployment.
- **Fake suites** (`teams_fake_all_dev`, 2409 videos across 15 fake methods): per-method fake recall.
- **Lockbox**: held-out partition the promotion-contract uses for *final* FPR + recall calibration. Never used for selection during training.
- **OOD aug-stress slices** (`ood_lighting_stress_*_real`, `ood_spatial_stress_*_real`): real videos from `external_youtube_avspeech` with eval-time augs applied. Probes generalization to lighting/spatial perturbations not seen in training.

---

## 4. The promotion contract

We have a two-stage scorecard that operationalizes "is this checkpoint deployable":

1. **Calibrate τ** lexicographically on dev (`arena/score_teams_promotion_contract.py`): find τ that minimizes false-positive rate subject to per-method recall floors. The lexicographic ordering matters — the policy is "as much fake recall as we can get under the FPR budget" rather than a single weighted objective.
2. **Read out lockbox FPR** at the calibrated τ. This is the deployment-side number we trust. Per-method recall is also reported at the same τ.

**Why the trainer's `value_composite` is not deployment-grade**: it is computed against a fixed snapshot of dev data and uses heuristic thresholds (target_mean_fpr 0.03, max_pool_fpr 0.05). It does not reflect the lockbox distribution and does not perform the lexicographic τ search. A run can score `value_composite ≈ 0.99` and still fail the contract — exactly what is happening with P8A right now.

**A known bug** (memory `project_contract_policy_bug`): when the FPR budget is set too tight, the contract τ-search drives τ toward ~0.995, which crushes fake recall to near-zero. We always check `selected_threshold` in scorecard outputs before trusting any contract-pass claim.

---

## 5. The shortcut problem

This section is the central technical challenge of the program — the reason for everything from RLP7 onward.

### 5.1 What we observed

When R13 RLP6_04 was running well at training metrics, it was simultaneously *flipping* on a small Dor-on-webcam clip with no virtual background. Same person, same lighting, same face — but on his webcam (a specific Logitech model in his home), the model declared the clip fake at high confidence.

### 5.2 Hypothesis evolution

We initially suspected **identity** (the model had memorized Dor's identity as "fake-correlated" because some Dor footage was in training). This was ruled out: the same identity flips correctly on different cameras (mobile, MacBook FaceTime), and Dor's whiteish-correct pool scores ~0.99 real.

We then suspected **camera ISP signature**. The Logitech webcam has a distinctive ISP pipeline (specific noise profile, color handling, compression). When forced through that pipeline, real footage looks like training-distribution fake-side artifacts — particularly the codec-style artifacts from the heavy aug pipeline. This hypothesis has held up.

A *fingerprint diff* analysis (`analysis/dor_pool_fingerprint_diff_2026-04-24.py`) showed the cleanest pool separators are `dct_hf_ratio` (high-frequency DCT content) and `bits_per_pixel` (compression intensity). Both are codec/ISP signals.

### 5.3 What we ruled out (with experiments)

- **Late-RLP6_04 consolidation**: RLP7_08 forked from RLP6_04 *step 4500* instead of *step 23500*. Same anchor-pool ceiling (~0.84–0.89). The shortcut is not formed late in RLP6_04 training — it's upstream.
- **R12g/R13 chain itself**: P8B trained a fresh head on plain CLIP from scratch (no R12g/RLP6 inheritance). It performed *worse* on the anchor than RLP6_04. So the shortcut is not in the R13 weight chain; it is in the data mix interacting with frozen-CLIP features.
- **Fine-tuning reach (refuted)**: until P8A, all FT was head + SVD-on-attention residuals only. P8A unfroze the final projection and LayerNorm and added SVD on MLP. Anchor ceiling broke (Δ −0.188 vs RLP6_04, vs −0.097 for the best P7). So **the shortcut was reach-limited** — frozen CLIP backbone weights had no way to redistribute features away from the camera signature; head + attention-only SVD wasn't enough degree of freedom.

### 5.4 The remaining anchor signal

P8A still flips 13/30 anchor frames (frac > 0.9 = 0.43). The shortcut is *weakened*, not eliminated. Frame-level analysis shows pinned and escaped frames cluster temporally in a 6-second clip — pinned early, escaping late. The model has the *capacity* to escape; content variation already shakes some frames loose. This is consistent with "the shortcut is no longer reach-bound, it is data-bound" — what's needed now is more diverse Teams real-camera data, not more model capacity.

---

## 6. Experiment chain (compact)

| Run | Key delta | Anchor Δ vs RLP6_04 | Verdict |
|---|---|---:|---|
| RLP6_04 (baseline) | production base | 0.000 (ref) | Production. Strong on training metrics (value_composite 0.9006, AUC 0.9942), weak on Dor anchor. |
| RLP7_02 | add codec-aggressive aug | −0.076 | Modest anchor improvement; smaller than hoped. |
| RLP7_04 | spatial-only aug | −0.045 | Spatial-only is the weakest single-axis improvement. |
| RLP7_05 | spatial + moderate codec | −0.089 | Best of P7 on anchor. Best balanced trade. |
| RLP7_06 | (variant) | (similar) | — |
| RLP7_07 | triple-axis (CCT + spatial + codec) | −0.097 | Marginally better anchor than RLP7_05; aug regressions on `crop_shift` etc. |
| RLP7_08 | RLP7_02 aug, fork from RLP6_04 *step 4500* | (similar to RLP7_02) | Refuted "late consolidation" hypothesis. |
| **P8A** (`9lmvb5b4`) | RLP7_02 aug + unfreeze visual.proj + visual.ln_post + SVD-on-MLP | **−0.188** | **Shortcut broken** by ~half. Anchor: 0.932 → 0.744. 13/30 frames still pinned. Best so far on real-pool FPR (12.11% on `teams_real_all_dev` vs 15.92% RLP6_04). **But hard-fake recall regressed −13.6 pp** (§7). |
| P8B (`n8yk2hox`) | fresh head on plain CLIP from scratch (no R12g chain) | +0.066 | *Worse* than RLP6_04. Confirmed shortcut isn't in the R13 weight chain. Job hung at step 12000 / 30000 on `teams_ood_fake` data loader; cancelled. |

P8A's checkpoint at step 5000 is the current leader — `gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth`.

---

## 7. Today's findings: the P8A regression

Today (2026-04-25) we ran a promotion-contract scorecard to validate P8A, RLP7_05, and RLP6_04 against the calibrated lockbox + per-method-recall pipeline. The aggregator is still running, but the partial slice-level data is in.

### 7.1 Real-pool FPR (lower is better, default τ)

| Slice (videos) | RLP6_04 | RLP7_05 | **P8A** |
|---|---:|---:|---:|
| `teams_real_all_dev` (3253) | 15.92% | 13.74% | **12.11%** |
| `teams_real_poor_quality_dev` (923) | 13.33% | 10.83% | **8.13%** |
| `teams_real_lighting_extreme_dev` (1401) | 15.85% | 12.63% | **11.13%** |

P8A leads all three. **The harder the slice, the bigger the relative improvement** — `poor_quality_dev` shows a 39% relative FPR reduction. Note that `lighting_extreme_dev` is only marginally harder than `all_dev`; the W&B-tracked synthetic `ood_lighting_stress_*_real` aug-stress slices showed 30–50% FPR, but those are eval-time augs on `external_youtube_avspeech` reals, not naturally-extreme Teams clips. ⚠ This means our previously-flagged "lighting weakness" is largely a *synthetic-stress* phenomenon, not a natural-distribution one.

### 7.2 Per-method fake recall (`teams_fake_all_dev`, 2409 videos, default τ)

This is where P8A regresses.

| Method | n | RLP6_04 | **P8A** | Δ |
|---|---:|---:|---:|---:|
| deeplive_enhanced | 545 | 79.6% | **53.0%** | **−26.6 pp** |
| teams_flat_xiang_xiang2_feng | 135 | 78.5% | **45.2%** | **−33.3 pp** |
| visomaster_enhanced_macro | 550 | 57.3% | **35.6%** | **−21.6 pp** |
| teams_capture_noyn_sharker_s23 | 204 | 97.1% | 91.2% | −5.9 pp |
| teams_capture_cam_test_s35 | 244 | 97.5% | 94.3% | −3.3 pp |
| (12 other `teams_capture_*` methods) | varied | ≥98% | ≥98% | ~0 |
| **Aggregate** | **2409** | **83.9%** | **70.2%** | **−13.6 pp** |

The "easy" production-style fakes (`teams_capture_*_s*`) are essentially unaffected. The harder out-of-pool methods collapse.

### 7.3 Failure-mode analysis (today, `analysis/p8a_fake_failure_analysis_2026-04-25/`)

We pulled the per-video predictions for both checkpoints and identified 329 videos where RLP6_04 caught the fake but P8A missed it. Findings:

- **Zero compensating gains**: there are *no* videos where P8A wins and RLP6_04 lost. The −13.6 pp gap is pure regression, not a redistribution.
- **Bimodal score distribution on missed fakes**: only ~19% of P8A's misses sit in [0.40, 0.50) — the threshold-recoverable band. The other 80% are at 0.10–0.40, with a substantial cluster below 0.10. ⚠ Implication: lockbox-calibrated τ tuning will recover at most ~3.8 pp of the 13.6 pp gap. **The remaining loss is genuine separability collapse, not a threshold drift.**
- **Visomaster regression concentrates on `_teams` sub-pool**: the in-domain Teams-codec variant of visomaster regresses 1.5× harder than the `_raw` variant (26.2% vs 17.1% regression rate). The codec-pressure component of P8A's training appears to have specifically hurt the slice closest to the deployment distribution.
- **Global score compression even where P8A stays correct**: on easy `teams_capture_*` methods where both checkpoints are ~99% correct, P8A's score distribution is wider — p10 score 0.90 vs RLP6_04's 0.97. The unfreeze didn't just shift borderline cases; it pulled the entire fake-side distribution toward the decision boundary.

### 7.4 The reframe

Before today, we believed P8A's regression might be threshold-recoverable. The failure-mode analysis says no — the unfreeze recipe pushed the model into a regime where real and fake are *less separable*, not just *differently centered*. Lockbox τ-tuning will help marginally; it will not fix this.

---

## 8. Packet-9 design space

Given §7, Packet-9 cannot simply be "P8A more, longer, harder." The question is which softening lever to pull.

### 8.1 Available knobs (audited today)

| Knob | Status | Mechanism |
|---|---|---|
| Independent `unfreeze_*` toggles (proj / ln / mlp-svd) | ✓ yaml | Drop visual.proj while keeping ln_post + mlp-svd. The most aggressive layer is visual.proj — dropping it might recover separability while preserving most of the anchor win. |
| `svd_blocks: [9, 10, 11]` | ✓ yaml | Restrict SVD residuals to the last 3 of 12 transformer blocks; reduce backbone reach. |
| Family weights in sampler | ✓ yaml | Up-weight visomaster_enhanced_macro and deeplive_enhanced fakes to directly counter their regressions. |
| Codec-aggressive aug intensity | ✓ yaml | Dial back `quality_p`, `jpeg_lower`, `webcam_codec_p`. Reduces the "codec stress on real pass" pressure that may underlie the conservative shift. |
| `quality_domain_loss_weight` | ✓ yaml | Bump from 0.1 → 0.25–0.30; harder gradient-reversal pressure against codec shortcuts. |
| Group-DRO (dynamic per-method loss reweight) | ✓ yaml | Enable to dynamically up-weight low-recall methods during training. |
| **`backbone_lr_mult`** | ✓ **yaml** (added today, default 1.0) | **Cap unfrozen-backbone LR (e.g., 5e-6) while keeping head at base LR (3e-5).** Highest-leverage clean lever. |
| Per-method classification loss weighting | ✗ code | Would require modifying `get_losses()` in effort_detector.py. Not pursuing today. |
| Asymmetric fake-FN vs real-FN loss | ✗ code | Same. Not pursuing today. |
| Selective per-block unfreeze beyond `svd_blocks` | ⚠ partial | Limited to SVD application gating; per-block weight unfreeze would need code work. |

### 8.2 Candidate Packet-9 variants (drafts, not committed)

Working hypotheses for variants we'd consider firing in parallel:

- **RLP9_01 — P8A-soft, drop visual.proj**: Same as P8A but `unfreeze_final_proj: false`. Keeps `unfreeze_final_ln + apply_svd_to_mlp`. ⚠ Hypothesis: visual.proj is the layer that did most of the score-distribution compression. If true, this should recover ~half the fake-recall regression while keeping most of the anchor improvement.
- **RLP9_02 — P8A + `backbone_lr_mult: 0.17`**: Same recipe, but cap unfrozen-backbone LR at 5e-6 while keeping head at 3e-5. Single cleanest lever. ⚠ Hypothesis: lower backbone LR limits how far the score distribution drifts; the anchor still gets reach but less reach-per-step.
- **RLP9_03 — P8A + hard-fake oversample + lower codec aug**: Up-weight `visomaster_enhanced_*` and `deeplive_enhanced` family weights ~1.5×; dial `quality_p` 0.72 → 0.60, `jpeg_lower` 30 → 38. Same backbone unfreeze. ⚠ Hypothesis: the regression is partly a training-mix imbalance issue; more visomaster/deeplive presence directly counters it.
- **(maybe) RLP9_04 — RLP7_05 lineage longer schedule**: A control to test "did we even need P8A's unfreeze recipe, or is the spatial+codec aug recipe converging on the same anchor with fewer side-effects"? Lower priority but a useful sanity check.

We have not committed to any of these yet. The decision waits on (a) the lockbox-calibrated aggregator landing and (b) Roee's pick.

### 8.3 What we are *not* doing for Packet-9

- **No scratch-on-plain-CLIP variants.** P8B refuted that direction (worse anchor than RLP6_04, real-pool also regressed).
- **No P8A-amplification recipes** (longer schedule at same recipe, harder unfreeze). Failure-mode evidence says these would amplify the regression.
- **No per-method classification loss reweighting in code** today. The yaml-only `family_weights` lever is sufficient and avoids new code blast radius before launch.

---

## 9. Open questions for second-opinion review

The places where we most want pushback:

1. **Is the "softened P8A" framing right?** Or is the cleaner intervention to back away from the unfreeze recipe entirely and pursue something orthogonal (e.g., explicit fake-recall rescue with loss reweighting, or a head-only retrain with a different SVD-rank choice)? The failure-mode evidence shows P8A's regression is separability loss; we are betting that a *softer* version of the same recipe sits in a regime where the trade is acceptable. Could that be wrong — i.e., is *any* version of "unfreeze visual.proj/ln_post + SVD-on-MLP" doomed to compress fake-side scores?

2. **Are we wrong about the codec aug being part of the problem?** We're suspicious of the codec-aggressive aug because it stresses the real pass hard, and we observe a "real-leaning" calibration shift in P8A. But the codec aug is also why P8A (and RLP7_05) outperform RLP6_04 on real-pool FPR. Dial-back risks losing the real-pool gain. Is the right move to dial it back, hold it constant, or even *increase* it on the fake-pool side only?

3. **Is the visomaster_enhanced_macro at ~57%-base recall acceptable to begin with?** Even RLP6_04 (the baseline) only gets 57% on this method. The method may be a hard-truth limit of CLIP-DataComp-XL representational capacity — in which case chasing recall here is misguided and we should accept it as a known weakness rather than design Packet-9 around it. ⚠ We have not run a "what's the maximum achievable visomaster_macro recall under this architecture, regardless of trade?" probe.

4. **Should we have run the lockbox-calibrated scorecard *first*, before anchor rescores?** Our workflow has been: rescore on anchor pool → identify candidate → run promotion contract. With P8A, the anchor rescore looked decisive in P8A's favor; only after running the contract today did we discover the fake-recall regression. Is the lesson "run the contract earlier, even on partial slices"?

5. **Is the program target right?** We are training one model that has to be Teams-deployable, robust to real-camera variation, and good at hard fakes. These are partly in tension. Should the program instead be a *cascade* — a fast Teams-tuned model that hands off uncertain cases to a heavier model trained primarily on hard fakes?

---

## 10. Pending readout (as of writing, 2026-04-25 ~16:30 UTC)

The promotion-contract scorecard for `[P8A, RLP7_05, RLP6_04]` is still running on Vertex AI (job `2162379556655202304`, asia-southeast1, started 08:33 UTC). Real-pool slices are complete (numbers in §7.1). Fake-method slices are partially complete (P8A done, RLP6_04 done, RLP7_05 in progress as of 16:30 UTC). After all per-method slices finish, the aggregator runs the lexicographic τ calibration and produces:

- `promotion_winner.json` with calibrated τ and lockbox FPR
- per-method recall at calibrated τ for each candidate

Realistic terminal time: 18:00–20:00 UTC.

**What we expect the aggregator to say** (based on §7's failure-mode analysis):

- P8A's lockbox FPR will be lower than RLP6_04's at calibrated τ (the real-pool gain holds).
- P8A's per-method recall on visomaster_enhanced_macro / deeplive_enhanced / xiang_xiang flat will be partially recovered by τ-tuning (perhaps +3–4 pp on aggregate) but will *still* sit below RLP6_04 at the contract τ.
- The lexicographic policy will likely refuse to promote P8A on the per-method recall floors, even though its FPR is better.

**What would surprise us**: if P8A's per-method recall *fully* recovers under τ-tuning. That would mean our separability-loss read of the failure-mode data is wrong, and the regression is just a calibration shift after all. We'd want to look hard at why the score-distribution analysis missed it.

---

## 11. References & artifacts

**State-of-detector longer-form doc (anchor-pool focus)**:
`docs/relaunch_handoffs/R13_RELAUNCH_STATE_OF_THE_TEAMS_DETECTOR_2026-04-25.md`

**Today's failure-mode analysis**:
`analysis/p8a_fake_failure_analysis_2026-04-25/`
  - `analyze.py` — reproducible script
  - `summary.json` — structured findings
  - `regressed_videos.csv` — 329 videos P8A loses where RLP6_04 wins

**Today's codebase audit (transient analysis, captured in §8.1 above)**

**Anchor-pool rescores**:
`analysis/pool_rescore_rlp8_a.summary.json` (P8A step 5000)
`analysis/pool_rescore_rlp8_a_step2500.summary.json` (P8A step 2500 — confirmed step 5000 is the leader)
`analysis/pool_rescore_rlp8_b.summary.json` (P8B at step 11000)

**Promotion-contract scorecard (in progress)**:
`gs://training-job-outputs/test_results/teams_promotion_contract/p8a-review-scorecard-20260425/reports/`

**Key W&B runs**:
- RLP6_04 (production base): `dtect-vision/phase2-experiments/h2pdu6i5`
- P8A: `dtect-vision/enhanced-aug-test/9lmvb5b4`
- RLP7_05: `dtect-vision/enhanced-aug-test/hhc8quq9`
- P8B (cancelled, hung): `dtect-vision/enhanced-aug-test/n8yk2hox`

**Yaml definitions**:
- P8A: `experiments/phase2_round13/R13_RLP8_01_unfreeze_clip_codec.yaml`
- P8B: `experiments/phase2_round13/R13_RLP8_02_fresh_head_plain_clip.yaml`
- RLP7_05 et al.: `experiments/phase2_round13/R13_RLP7_*.yaml`

**Code change today (uncommitted)**:
- `utils/setup.py:choose_optimizer` — added optional `optimizer.adam.backbone_lr_mult` (default 1.0). Sanity-checked. Not yet committed; awaits Packet-9 launch decision.

---

*Document prepared 2026-04-25 by Claude Opus 4.7. Numbers in §7 are from scorecard reports already on disk; all other numbers are from prior session memory or repo files. Where we marked ⚠, the claim is interpretive and the evidence is suggestive, not decisive.*
