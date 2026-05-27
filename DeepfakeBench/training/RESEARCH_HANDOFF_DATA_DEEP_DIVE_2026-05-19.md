# Research Handoff — Data-Side Bottleneck in a CLIP-Based Deepfake Detector

> **Audience.** A researcher coming to this problem cold. The goal of this doc is to give you enough context to walk into the academic literature (white papers, arXiv, recent CVPR / ECCV / NeurIPS / WACV proceedings, blog posts, GitHub repos) and tell us *"this published technique is structurally aligned with your bottleneck — try it."* It is written assuming you have a strong vision/ML background but **no specific context** on this codebase or its training history.
>
> **Date.** 2026-05-19. The "where we are" reflects 5 weeks of dense empirical work (~50 training packets, ~30 CPU-side probes) on a deployment-grade deepfake detector for Microsoft Teams video calls.

---

## 1. What we are actually building

A binary classifier for **face deepfakes in live Microsoft Teams video calls**. Concretely:

- **Input**: a single RGB face crop, typically 224×224, aligned and roughly frontalized by an upstream face detector + cropper. The "image" the model sees is a face image with a small amount of background context (typical crop-to-face area ratio 0.5–0.7).
- **Output**: a scalar probability `p_fake ∈ [0, 1]`. We compare to a calibrated threshold τ. If `p_fake ≥ τ`, the frame is flagged.
- **Deployment context**: production attackers use real-time face-swap toolkits (DeepFaceLive, Visomaster, DFDC-style methods) to impersonate real people in Teams calls. The model must catch these *during* a live call.

False positives are very costly: flagging a real person on a real call as a deepfake destroys the product. The product target is **fake recall as high as possible at real-side FPR ≤ 5%** on a held-out evaluation set whose composition is meant to mirror real Teams traffic.

## 2. The three pillars (operational requirements)

Any candidate model is evaluated against three orthogonal requirements:

1. **Fake recall** on the targeted generation families (`deeplive`, `visomaster`).
2. **Real-side FPR ≤ 5%** at the deployment threshold.
3. **Robustness** across lighting, cameras, codecs, capture pipelines, and identities.

All three are load-bearing. A "good" model that catches 95% of fakes but false-flags real Teams users at 30% is useless. A model with 0% FPR that misses 80% of fakes is useless. A model that performs well on the test set but collapses on a 30%-darker lighting condition is useless.

## 3. The training data

### 3.1 Buckets and labeling

Training data is stored on Google Cloud Storage. Two primary pools:

- **`teams-v2` bucket** (hundreds of thousands of face-crop frames). Real-side: YouTube-origin videos of public-figure faces, processed through a *Teams-capture pipeline* (resolution downsample → codec recompression → simulated webcam capture). Fake-side: the same YouTube-origin faces with deepfake methods (`visomaster`, `deeplive`, `deepfake-detection-challenge-style`) applied, passed through the same Teams-capture pipeline.
- **`proper_data` bucket** (hundreds of thousands of frames). Drawn from HDTF (high-definition talking-face) and QCLIPS public datasets. Includes clean and Teams-pipelined versions of the same identities.

Both pools use **frame-level labels**. A given frame is either `real` or `fake`; we do not currently use temporal, clip-level, or video-level supervision.

### 3.2 Identity coverage

- ~430 unique identities in the `visomaster` fake-method pool.
- ~1900 unique identities in the `deeplive` fake-method pool.
- An additional ~2300 identities exist in unused buckets — *increasing identity count alone has been refuted as a lever* (see §6).
- Per-method face-size band: each fake method generates faces at a tight face-pixel-area band (e.g., `deeplive` outputs cluster at ~22-25k pixels² regardless of source video resolution). Real-side spans wider face-pixel-areas. **This is a known label leak** — the model can learn "small face = real, medium face = `deeplive`-shaped" rather than learning a forgery feature.

### 3.3 Transport / capture pipeline

The `teams_capture` pipeline applied during training is a sequence of: down-resample to a typical Teams resolution (~640×360 or similar), JPEG-style recompression, optional simulated bandwidth artifacts, then face re-detection and re-crop. *This pipeline is the same one applied to both the real and fake sides of the training pool.*

The critical fact: **the training real-side frames are YouTube videos that were post-processed through this synthetic pipeline**. They are *not* directly captured Teams calls. The model has never seen a true webcam capture of a real person on a Teams call during training.

## 4. The evaluation data

Two tiers:

- **Dev pool**. In-distribution validation. Used for threshold (τ) calibration. ~29 substrates covering different methods × transport (clean vs Teams) × identity sets. Dev real frames are 100% YouTube-derived face crops with the Teams pipeline applied — the same construction as training reals.

- **Lockbox pool**. Held-out evaluation, never seen by training or τ-calibration. Contains:
  - **Directly captured Teams-call frames of real people** (live recordings on actual hardware, not YouTube-derived).
  - Fake-side examples generated by the targeted methods and captured through the same pipeline as deployment.
  - Diagnostic substrates: `dor_evening`, `dor_morning`, `xinhe_may6_falseflag`, `live_*_teams_prod`, etc.

**The dev↔lockbox gap is structurally large** and is one of the central facts of this program:

- A KLIEP density-ratio discriminator separates **dev-real vs lockbox-real** at **99.09% balanced accuracy** in *frozen* CLIP feature space (i.e., before any FT). The two distributions are almost linearly separable in vanilla pretrained CLIP embeddings.
- Wasserstein-1 distances between pool centroids in CLIP-feature space:
  - W(train, dev) = 0.77
  - W(train, lockbox) = 5.57 → **7.25× ratio**

Source: `analysis/cpu_diagnostics_2026-05-12_d10_training_pool_position/D10_FACTS_2026-05-12.md`.

## 5. The model and training setup

- **Backbone**: OpenAI CLIP ViT-L/14 (24 transformer layers, 1024-dim hidden, 224px input, 14px patches).
- **Head**: ArcFace margin head over the `[CLS]` token of layer 23 (final layer), with margin scheduling (m=0.10 → 0.20) and scale s∈[6, 12]. Also tested: plain linear head + cross-entropy.
- **Fine-tuning**: typically full-encoder FT with SVD-reparameterized attention projections (rank ≤ hidden_size−1) to constrain capacity. Recent packets also test LoRA (rank 16, layers 10-11) stacked on top of the SVD constraint.
- **Auxiliary objectives explored**: GroupDRO over chronic-FP identities, gradient-reversal-layer (GRL) adversarial heads against capture-mode / image-quality axes, pair-rank loss on `(real, fake)` twin pairs, Fourier-band augmentation, anchor-aware contrastive loss against a fixed identity anchor pool.
- **Augmentations explored**: face-scale jitter, resolution-chain randomization (multi-step downsample → upsample), color jitter, codec recompression.

## 6. What we have measured (load-bearing findings)

These are *measured*, not hypothesized. Each cites the FACTS document with raw numbers.

### 6.1 The forgery signal is already present in raw, frozen CLIP

`analysis/cpu_diagnostics_2026-05-12_d2_encoder_separation/D2_FACTS_2026-05-12.md`

A 5-fold cross-validated linear probe on **frozen** CLIP ViT-L/14 L11 features separates fake from real at **AUC = 1.000** on the "chronic-6" sub-cohort (n=282 frames, 41 fakes). The encoder, with **zero fine-tuning**, can already see the forgery signal.

This is dispositive on encoder capacity: the failure mode is *not* "the encoder doesn't have enough representational power to see fakes."

### 6.2 Fine-tuning amplifies an image-property shortcut

The model learns to predict "fake" using **image properties** (sharpness, color cast, face size, crop tightness, capture mode) that are correlated with the fake-method labels in training but are independent of forgery in deployment.

- Per-frame logit correlates negatively with Laplacian-variance sharpness on most evaluation suites → the model treats "soft / less-sharp" as a fake predictor.
- D6 bootstrap (B=50, n=282): the chronic-6 fake/real direction in L11 is **83.68° from the IQ-PC1 axis** in frozen CLIP (near-orthogonal), but only **69-75° after FT** — every FT recipe **rotates the discrimination direction toward the IQ axis**, by 5° (P8A) to 10° (E2B).
- D7 block-drop partial R² on per-frame FPR: the IQ-axis block contributes **+0.2359 (P8A) / +0.3143 (T5C)**; the substrate-axis block contributes **+0.0146 / +0.0001**. **Per-frame FPR is dominated by image-property variation, not by substrate or identity per se.**

### 6.3 A small set of "chronic" identities concentrates almost all FP mass

Six identities — `PC_Generator`, `Roy_D`, `bla_bla_chow`, `dor_shkedi`, `xiang`, `dor` (some appearing across sub-sessions like `PC_Generator__s22`, `dor_shkedi__s16`, etc.) — account for **>95% of lockbox real-side false positives** across most checkpoints, while comprising ~10% of the lockbox real set by frame count.

Different checkpoints fail on different members of chronic-6 (e.g., P8A scores `PC_Generator__s22` at 0.91 while E3 scores it at 0.06; PA handles `Roy_D` but P8A regresses it). The information to separate them is somewhere in feature space — no single checkpoint extracts it cleanly, and no label-free ensembler achieves more than 16% viso recall vs P8A's 27%.

### 6.4 The train pool does not span the deployment image-property distribution

In addition to the dev↔lockbox gap in §4, the train pool also lives in a tight region of image-property space:
- Train real-side frames have a tight `face_pixel_area` band, a tight `sharpness_laplacian` band, and a single capture-mode mode (because they are all YouTube-derived passed through the same synthetic pipeline).
- Lockbox real-side spans capture modes that include webcam, phone-screen, and screen-capture variants the training pool barely contains.
- Memory: `project_lockbox_fpr_dominated_by_webcam_mode` — `clip_capture_mode == webcam` accounts for 65.7% of lockbox real-side FPR; modern-v2 filter (which removes the most-divergent capture modes from eval) drops headline FPR from 4.6% → 0.71%.

### 6.5 The "viso 27% ceiling" is partly an eval-substrate artifact

The headline metric `visomaster_recall_at_FPR_10%` has been stuck at **~27%** across 13+ packets and 3 architecturally distinct training recipes (full-FT L/14, scratch-train B/16, scratch-train L/14 with cross-entropy).

`analysis/cpu_diagnostics_2026-05-04/.../job14`: if you remove chronic-6 + frames with `min(W, H) < 200` + frames the face detector skipped (`is_no_face`), the **same checkpoints** jump to **67–78% viso recall** with FPR < 5%. The 27% ceiling is largely substrate-pollution; but it remains a real deployment failure because production captures contain those same problematic frames.

### 6.6 Two recent training-side levers have shown structural bite

Most recent (2026-05-15 → 2026-05-16):

- **`resolution_chain_aug`** (multi-step downsample → upsample applied during training) cut the per-resolution score-range from 0.605 to 0.448 (25% reduction) — the model is no longer using source-resolution as a fake predictor.
- **`anchor_aware`** (a contrastive loss that pulls embeddings of chronic-identity examples toward a fixed anchor pool) eliminated chronic-FP on 2 of 6 chronic identities (`Chikara_Takahashi` 26%→0%, `PC_Generator` 28%→0%) but introduced a `Roy_D`-specific failure (29%→81%).

These are the only levers in the last 5 weeks that bit *structurally* (changed a known failure mode by ≥10pp absolute) without being a substrate-cleaning artifact.

## 7. What has been refuted as a lever

- **More data quantity / identity diversity.** ~2300 additional identities exist; previous packets adding them never lifted operational metrics. `project_data_inventory_identity_diversity`.
- **Rebalancing existing buckets** (family weights, lane composition). Tested twice. P14 (fw=8.0 + anti-shortcut bundle) collapsed to `value_composite=0.126`; P16 (fw=2.0 standalone) did not lift viso recall above 1.1%. `project_data_axis_lever_pulled_twice_no_lift`.
- **Backbone alternatives.** Scratch-trained B/16 (E2B) and L/14 (E3) with cross-entropy heads both improved `deeplive` recall but failed the `viso` ceiling and regressed chronic-6 invariance. `project_e2b_breaks_deeplive_ceiling`, `project_l14_does_not_break_viso_ceiling`.
- **Head retraining only.** Re-training the head on a substrate-diverse pool while keeping the frozen FT encoder gave 66-72% lockbox FPR at dev-cal τ → no operational lift. `project_job7_head_retrain_REFUTED_2026-05-04`.
- **Single-lever FT-recipe interventions** in isolation: LoRA on L10-L11 alone, GRL with static λ, pair_rank λ=0.2, GroupDRO over `chronic_flag`, `face_scale_jitter` alone — each tested as a single lever, none broke the ceiling. Most regressed at least one of the three pillars.

## 8. Primary hypothesis (high confidence)

**The training pool's joint marginal distribution on five image-property axes —**
- `sharpness` (Laplacian variance of the face region),
- `color_cast` (per-channel color standard deviation, particularly `color_a_dev`),
- `face_pixel_area` (face size in pixels),
- `crop_tightness` (face area as a fraction of crop area),
- `capture_mode` (webcam / phone-screen / screen-capture / studio),

**does not span the deployment (lockbox / production) distribution on these same axes.** The model learns these axes as fake-predictors during FT because they happen to correlate with fake labels in training. Deployment frames sit in regions of this joint space the model has never seen, so when a real frame lands in a region that was "fake-dense" in training, it false-positives.

This hypothesis is **coupled**: it is a data-design problem (the training pool's joint marginal is constructed by sampling decisions, not by deployment) AND a training-recipe problem (the FT recipe rotates the discrimination axis toward these properties — proven by the D6 angle measurements). Either side alone is plausibly fixable; together they require either:

- **Targeted data ingestion** — new real-side frames whose joint distribution on the five image-property axes *matches deployment*. Distinguished from the refuted "more data" lever because the targeting axis is not identity-count or bucket-balance but the IQ marginal itself.
- **Training-recipe deconvolution** — continuous-axis-GRL or anchor-aware against the specific IQ axes, holding data fixed. This is the current frontier; `resolution_chain_aug` and `anchor_aware` are partial bites in this direction.

The two paths are complementary, not mutually exclusive. A clean test of which dominates would re-tag the train pool on the five axes and compute the fraction of lockbox / chronic-6 mass in regions where train density is < (5%, 10%, 25%) of deployment density. *That measurement is in flight as a companion to this document* (see §11).

## 9. Less-likely candidates (lower priors, not zero)

These are alternative explanations the literature might have direct techniques for. We give each a rough subjective prior.

| # | Candidate hypothesis | Prior | What would change our mind |
|---|---|---:|---|
| 1 | **Encoder capacity is insufficient for the targeted manipulation classes.** | 5% | If a frozen DINOv2/SigLIP-2 probe at the chronic-6 level gave AUC > CLIP-frozen's 1.000 AND IQ-orthogonality > 88° — i.e., a backbone that holds more forgery signal in a more IQ-invariant direction. |
| 2 | **Binary `real / fake` label space is too coarse.** Each fake method has its own artifact signature; collapsing them forces the head to learn a least-common-denominator that *is* the image-property axis. | 20% | A multi-class method-conditional head out-performs the binary head on at least one of the three pillars. (Note: 12-class method-conditional GRL was tested in P18 and was a null result — but as a *regularizer*, not as the primary head.) |
| 3 | **Frame-level scoring is the wrong unit.** Real Teams calls have temporal coherence the frame model cannot see; clip-level or video-level pooling might recover signal. | 25% | A temporal aggregator over frame-level scores breaks the viso ceiling on the same encoder. |
| 4 | **CLIP's pretraining objective is hostile to the task.** Image-text contrastive learning compresses image-property variation that is "linguistically irrelevant" but is exactly where forensic cues live. DINOv2 / SigLIP-2 with dense-feature objectives might preserve more local forensic information. | 10-25% | Frozen DINOv2 / SigLIP-2 chronic-6 probe shows materially different (higher) AUC OR materially better IQ-orthogonality. Companion document covers this. |
| 5 | **Capture-mode × identity entanglement.** A real Teams capture with `dor`'s specific webcam, lighting, and color cast may anchor an identity-cluster the model cannot disentangle from manipulation, even with anchor-aware loss. | 30% | A capture-mode-randomized augmentation collapses the chronic-6 FP cluster without changing data. |
| 6 | **The forgery signal is in spectral / phase information CLIP discards.** Frequency-domain artifacts (DCT coefficient statistics, FFT band residuals, phase consistency) may be the actual ground truth, and CLIP-on-pixels is an oblique proxy. | 15% | A frequency-domain feature, attached as auxiliary input or as a separate head, breaks the viso ceiling. |
| 7 | **The eval contract itself is mis-specified.** Per-substrate τ-calibration on a production-realistic real cohort gives different numbers than the current dev-calibrated headline. We have evidence this is partially true (Job 14, F1 simulation memory) but it doesn't *break the model* — it just re-frames how to read existing checkpoints. | 30% | Already partially confirmed; what would *fully* change our framing is if the production-honest contract showed P8A meeting all three pillars at deployment τ. |

## 10. Where to point the literature search

The non-obvious question is whether any existing technique's *mechanism* specifically targets the joint-marginal mismatch between train and deploy on image-property axes — as opposed to identity-invariance, method-invariance, or generic OOD robustness in isolation.

Search angles, ordered by structural alignment to our problem:

1. **Spurious-correlation last-layer retraining for vision encoders**. JTT (Just Train Twice), DFR (Deep Feature Reweighting), CRT (Classifier Retraining), GroupDRO and its variants, BalancedERM. Especially: the techniques that work *without* group labels but with image-property tags.
2. **Robust face manipulation detection**. CVPR/ECCV 2023-2026 — esp. work on cross-dataset generalization (FaceForensics++, DFDC, DeeperForensics, KoDF cross-eval), domain-invariant feature learning specifically for deepfake detection.
3. **Frequency-domain / spectral methods for forgery detection**. F3-Net, SPSL, FreqNet, MultiAtt — anything where the input or an auxiliary head operates on DCT / FFT representations rather than pixel features.
4. **Self-supervised visual backbones evaluated on forensic transfer**. DINOv2, MAE, BEIT-3, SAM as frozen feature extractors for downstream forgery heads. Anchor: would the chronic-6 frozen-probe AUC = 1.000 hold for these, and with greater IQ-orthogonality?
5. **Anchor-based / prototype contrastive losses for chronic false-positive mitigation**. SupCon, Prototype Networks, deep metric learning specifically designed for OOD-robust face recognition (e.g. ArcFace variants with anti-shortcut regularizers).
6. **Image-quality-invariant face representations**. Curricula that randomize JPEG QF, resolution, color cast at train time; IQ-conditional normalization; robust ArcFace variants for low-quality face recognition (TinyFace, IJB-C protocols).
7. **Capture-pipeline-aware data augmentation**. Synthesia, RealForensics datasets, "synthetic-to-real" gap closure work for face capture.
8. **Density-ratio matching for training data resampling**. KLIEP, RuLSIF, importance-weighted ERM specifically applied to image-property axes (as opposed to class balance).
9. **Test-time adaptation for face forensics**. TENT, T3A, TTT — any technique that adapts per-sample at inference time given image-property metadata.
10. **Production-substrate-matched deepfake benchmarks**. Anything reporting on directly-captured webcam / Teams / Zoom real-call data, especially with both real and fake examples generated through the same capture pipeline.

A "**hit**" would look like: a paper whose mechanism specifically deconvolves image-property correlates of class labels (or trains an encoder whose discriminative direction is provably orthogonal to a chosen nuisance axis) AND has been validated on a face-deepfake or face-forensic benchmark.

## 11. Companion audit (in flight as of 2026-05-19)

The primary hypothesis in §8 makes a falsifiable prediction: a large fraction of lockbox / chronic-6 frames sit in regions where train density is very low (≪ deployment density) on the five-axis joint marginal.

A CPU-side audit producing this measurement is being computed alongside this document:

- Inputs: existing tagged parquets at `analysis/iq_data_atlas_2026-05-08/_cache/*.parquet` (29 pools × ~500 frames, 7 image-property axes) and `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` (7334 frames with identity_key + richer schema).
- Outputs: per-axis distribution overlap (Wasserstein-1, K-S) and joint-marginal coverage (5-axis KDE / multidim-binning) between train and each of {dev, lockbox, chronic-6}.
- Decision rule: if ≥30% of lockbox-real mass and ≥50% of chronic-6 mass lies in regions where train density is <10% of lockbox density, the **targeted data-ingestion lever is structurally live** and prior packets ("rebalance buckets", "more data") were the wrong specification of "data." If <10% / <20% of mass lies in those regions, the data thesis is **not** the binding constraint and the training-recipe deconvolution path (continuous-axis-GRL, extended anchor-aware) is the residual lever.

Output document expected at `analysis/joint_marginal_audit_2026-05-19/JOINT_MARGINAL_FACTS_2026-05-19.md`.

---

## Appendix A — Suggested reading order in this repository for the curious researcher

Start with FACTS docs, not OPINIONS docs (the codebase has a strong convention separating measurement from interpretation):

1. `docs/packet_retrospectives/MODEL_GOALS.md` — the three pillars in detail.
2. `analysis/cpu_diagnostics_2026-05-12_d2_encoder_separation/D2_FACTS_2026-05-12.md` — frozen-CLIP chronic-6 AUC = 1.000.
3. `analysis/cpu_diagnostics_2026-05-12_d7_fpr_decomposition/D7_FACTS_2026-05-12.md` — IQ vs substrate vs chronic block-drop partial R².
4. `analysis/cpu_diagnostics_2026-05-12_d10_training_pool_position/D10_FACTS_2026-05-12.md` — KLIEP discriminator and 7.25× Wasserstein ratio.
5. `analysis/cpu_diagnostics_2026-05-04/.../job14` — substrate-cleaning lift on viso.
6. `docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md` — the program-level synthesis of the IQ-shortcut hypothesis.
7. `docs/packet_retrospectives/threads/processing_signature_shortcut.md` — the foundational shortcut thread.
8. `docs/packet_retrospectives/STATE.md` — the rolling current-state snapshot.

## Appendix B — Glossary of terms used inside the codebase

- **Packet**: a planned training experiment with one or more checkpoints (named e.g. P8A, T3, T5C, P18, RLP6_04). Each packet has a `packets/<name>.md` retrospective doc.
- **Checkpoint / ckpt**: a specific model snapshot at a specific training step.
- **Substrate**: an evaluation sub-population (e.g., `teams_real_all_dev`, `visomaster_enhanced_macro_dev`, `lockbox_fake_recall`). 29 substrates make up the contract suite manifest.
- **Lockbox**: held-out evaluation, never used in training or τ-calibration. Should be the closest proxy to production.
- **Chronic-6**: the six identities responsible for >95% of lockbox real-side FPs across most checkpoints.
- **IQ axis**: image-quality axis (sharpness / brightness / color cast / face size / etc.) — five-to-seven of these are the "binding axes" the model uses as shortcut.
- **F0 / F1 / F2 / F3 / F4 / F5**: close-criterion families for the deployment contract; F4 specifically means "evaluation substrate cleaned of chronic-6 + low-res + no-face frames."
- **Pillar 1 / 2 / 3**: fake recall / real FPR / robustness — see §2.
