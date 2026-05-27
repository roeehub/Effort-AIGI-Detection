# Anti-Shortcut Techniques — Ranked Catalogue and Next-Packet Sequencing

> **Status as of 2026-05-06 ~11:00 UTC.** PD (correlation_penalty) scorecard `8207399447131324416` running in us-east1 against the 8-entry ckpt map (`teams_target_domain.deeplive_viso_corr_2026-05-06.yaml`); verdict pending. This document is the forward-looking strategy doc for what comes after PD, regardless of PD's outcome.

## Why this document exists

R13 has run ~30+ packets attacking shortcut learning. Each lever has produced a modest, bounded effect; none has cleanly cleared the 4/4 deployment-grade close criterion. The recurring pattern is **the shortcut shifts onto an untargeted axis** rather than disappearing. Two new pieces of evidence on 2026-05-06 raise the urgency:

1. **Same-person score drift on production data**. CPU diagnostics on `analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv` show P8A score on Dor real frames moves from `dor_evening` mean 0.015 → `dor_morning` mean 0.157 → `teams_real_dor_dev.dor_shkedi` mean 0.331 (a **52× drift on the same person across recording conditions**). E2B and PA show the same pattern at smaller magnitude.
2. **Cross-camera Xinhe inversion (user observation 2026-05-06)**. The same person (Xinhe) tested under different cameras produced **completely opposite results** — caught as fake under one camera, missed entirely under another. This is qualitative confirmation of the same phenomenon the diagnostics measure: the model is reading capture-condition features and mixing them into the manipulation score.

Both confirm the same finding: **shortcut learning is still active despite the R13 anti-shortcut sequence**. Whatever we run next must either weaken this directly (consistency / invariance) or change the training distribution so the shortcut signal doesn't generalize (pseudo-fakes / SBI).

## What R13 has already tried (so we don't repeat)

| Axis | Levers exercised |
|---|---|
| Augmentation | `pipeline_random`, `face_scale_jitter@0.25/0.50`, `TeamsCodecSimulation`, `WEBCAM_HARDEN`, ShiftScaleRotate, `context_variation_*`, P22 augmentation curriculum, S1/S2/S3, "heavy aug" baseline. **Codec aug HURT viso 35-50pp on PC (refuted)**. **face_scale_jitter@0.50 isolated won trainer composite but did NOT promote and did NOT close the design-intent face-size flip rate**. |
| Loss / regularizer | arcface margin (`m=0.15` retained), `anchor_aware`, GRL/DANN (`λ=0.20` static), stability-KL, **`correlation_penalty` (PD, in flight)**. Drafted but never launched: mixup, label_smoothing, feat_norm_reg, quality_lambda. |
| Data | `visomaster_enhanced` + `visomaster_teams_enhanced` at fw=4.0 (PA — F4 v2 lift but did NOT generalize to HDTF — substrate-bound memorization). `visomaster_hints*` permanently disabled (bad data). DF40, proper_data lanes. |
| Architecture | CLIP `visual.proj`+`ln_post`+`MLP-SVD` unfreeze (P8A — broke camera-signature ceiling), B16 scratch (E2B — broke deeplive ceiling, regressed viso), L14 scratch (E3 — no help on viso). |
| Methodology | Bundle decomposition discipline, single-lever testing, cross-substrate validation (HDTF vs v2), F4 substrate cleaning (chronic-6 + low-res + no-face removed). |

**The pattern across all of these**: each lever pushes the shortcut around without eliminating it. PD's frozen-head prototype directly demonstrated this — penalizing `sharpness_laplacian` and `luma_mean` at λ=1 caused `face_area` and `is_webcam` correlations to *rebound* (0.032→0.157 and 0.081→0.205 respectively). Encoder-side fine-tune may show the same shifting; that's part of what the in-flight scorecard measures (F3 close criterion).

## Critical evaluation of the 6 outside suggestions

Verbatim assessments, with theoretical fit, evidence in this problem class, cost, risks.

### #1 Fourier / amplitude-domain augmentation — strong, with one mandatory pre-validation

**Mechanism**: APR-S / FACT-style. Decompose image into amplitude (style) and phase (content); randomize amplitude across same-label samples; keep phase. Optionally add a consistency loss between original and Fourier-augmented views.

**Why this fits**: documented R13 shortcuts (sharpness, luma, HF energy) all live in the amplitude spectrum. Same-label restriction is structurally correct (cross-label mixing breaks the label).

**The mandatory caveat**: face-swap manipulation signals can ALSO live in amplitude (Wang et al. 2020 "CNN-generated images are surprisingly easy to spot" showed GAN frequency fingerprints; several deepfake-frequency papers report the same). If the manipulation signal is mostly amplitude-domain, naive amplitude randomization erases it. The same-label restriction does not protect against this — it just keeps the label valid.

**Required pre-validation (~1 day, free, do BEFORE any packet)**:
- Train tiny linear classifier on FFT magnitude alone vs FFT phase alone, using `teams_fake_all_dev` vs `teams_real_all_dev`.
- If amplitude-only AUC ≥ 0.85: amplitude carries the manipulation signal; naive randomization will hurt. Move to band-limited or partial amplitude mixing.
- If phase-only AUC ≥ 0.85 and amplitude-only is materially lower: amplitude randomization is safe. Proceed with confidence.

**Theoretical fit**: HIGH. **Evidence in this problem class**: MIXED (FreqAug-for-deepfake reports gains, especially cross-dataset). **Implementation cost**: LOW (~50 LOC, FFT ops fast in PyTorch). **Risk-of-regression**: MED-HIGH without pre-validation.

### #2 AugMix-style consistency, customized — right idea, hybrid composition is the right answer

**Mechanism**: For each frame, generate 2 augmented views from the documented nuisance axes (codec, luma, sharpness, crop scale, face area, compression). Train with main-CE on the original + JSD/KL consistency between original and views.

**Critique of "customized only"**: PD's prototype proved that closing 2 axes shifts the shortcut onto untargeted axes. AugMix's strength is its *diversity* — random chaining of N=3 ops with random severity. If we only branch along known axes, we train invariance to those, but the next shortcut axis we didn't include becomes the new fake predictor.

**Hybrid composition (recommended)**: 60% custom-substrate branches (codec, luma, sharpness, crop, face-size, compression — their list) + 40% generic image-level ops (RandAugment subset: auto-contrast, equalize, posterize, color jitter, gaussian noise). All chained, JSD on the predictions.

**Risks**: with high β consistency weight, model can collapse to constant prediction. Standard β = 10-12 vs CE main loss; needs sweep.

**Theoretical fit**: HIGH. **Evidence**: AugMix is well-established (Hendrycks 2020, ImageNet-C). **Implementation cost**: MED (~1.5d eng — dataloader emits triplets, JSD loss, β sweep). **Risk-of-regression**: LOW-MED.

**Most direct fit to the user's lived experience and to the same-person drift the diagnostics measure**.

### #3 SBI / FreqBlender pseudo-fake — the most paradigm-shifting suggestion in their list

**Mechanism**: SBI (Self-Blended Images, Shiohara & Yamasaki, CVPR 2022). Take a real face, apply small geometric/color transforms to itself, blend at a face-shaped mask boundary, label the result "fake." The artifact is the **blending boundary** — the universal artifact across face-swap methods (deeplive, viso, SimSwap, all of them). Mix with real fakes at ~40/40/20 ratio (real / known-fake / pseudo-fake).

**Why this fits**: PA collapsed on HDTF because it learned the v2 substrate's identity space, not a generalizable manipulation signal. SBI's pseudo-fakes have no specific generator's fingerprint — they have ONLY the blending artifact. Trains a method-agnostic boundary detector. **Strongest known answer in the literature for cross-method generalization.**

**Risks specific to us**:
- SBI was tuned for FaceForensics++. Production-quality face-swaps may have different blending characteristics; pseudo-fakes might be too obvious or too subtle.
- Mixing pseudo-fakes with real fakes shifts the loss surface; recipe may need re-tuning.
- 40/40/20 ratio is a starting guess; ratio sweep adds runs.

**FreqBlender** is less established in the literature; treat as ablation-only after SBI baseline lands.

**Theoretical fit**: HIGH. **Evidence**: STRONG for cross-method deepfake generalization. **Implementation cost**: HIGH (~3d eng — needs face landmark detection (have via MediaPipe), self-blending pipeline, new data lane in `combined_paired.py`). **Risk-of-regression**: MED — could hurt in-distribution recall if pseudo-fake distribution drifts from real fakes.

### #4 ViT patch-based negative augmentation — good ViT-specific addition, second-tier

**Mechanism**: DropPatch / patch noise outside core face landmarks, with consistency loss between original and patch-perturbed view.

**Why this fits us specifically**: Job 3 audit showed **90% of P8A FPs come from frames with `min(W,H) < 200`** (low-resolution crops where peripheral content drives Laplacian). The shortcut likely parks in periphery patches (background, hair, neck shadow, ambient lighting) rather than face content. Patch-aware masking outside landmarks directly attacks this geometry.

**Risk**: small face crops have few non-face patches; the augmentation is weakest where it's most needed. Need patch-coverage sanity check on training data.

**Composition**: pairs naturally with consistency loss as one of the JSD branches. **Don't run as a separate packet**; fold into #2 as a branch.

**Theoretical fit**: MED-HIGH (ViT-specific). **Evidence**: MED (proven for general ViT robustness; less specific to deepfake). **Implementation cost**: MED (~1d if folded into #2, 2d standalone). **Risk-of-regression**: LOW.

### #5 Style / texture randomization — overlap with current R13 aug; only useful as paired counterfactual

**Mechanism (their framing)**: histogram matching, gamma, white balance, camera noise transfer, sharpening/denoising variation.

**Critique**: R13's existing `pipeline_random` and `data/augmentations/` infrastructure already does most of this. The only structurally new piece is **paired counterfactuals** — pair each frame with a style-matched version of itself from a different recording session, force consistency. That's just one specific instantiation of #2.

**Verdict**: DEMOTE / FOLD into #2 as one of the augmentation branches. Don't run as a separate packet.

### #6 Lightweight adversarial augmentation — right place in the ranking, lowest priority

**Mechanism**: Adversarial perturbation targeting substrate axes (PGD on sharpness/luma directions), then train detector to be invariant.

**Theoretical strength**: HIGHEST. The adversary searches for the worst augmentation; provably stronger than random.

**Why it's last anyway**:
- 2-5× training cost (PGD inner loop). 8h becomes 30-40h on Vertex.
- Adversarial training is famously unstable; we'd burn 1-2 packets on tuning.
- Memory `project_image_quality_shortcut.md` already characterized the shortcut axes; we don't need adversarial discovery to find them.

**Theoretical fit**: HIGH. **Evidence**: STRONG for general robustness, less specific to deepfake. **Implementation cost**: HIGH. **Risk-of-regression**: HIGH (instability + cost).

## My prior-list re-evaluation

| Prior suggestion | Honest re-read |
|---|---|
| AugMix consistency (generic) | Their #2 (customized hybrid) is strictly better. Replace with their version. |
| HSIC nonlinear penalty (PD successor) | Not in their list. **Still stands** — only worth running if PD shows shortcut-shifting on the scorecard. Cheap (~30 LOC delta to PD). |
| GroupDRO / V-REx | Not in their list. **Still stands** — different geometry from anything in their list (outcome-equalization, not feature-invariance). Targets the same-person-drift observation at the loss level. |
| Frequency-domain randomization | Their #1 is a strictly better, more theoretically-grounded version. Drop mine. |
| Test-time augmentation | Not in their list. **Still stands** — free, deployable today, orthogonal to all training-time levers. |
| Stylized training (texture-bias breaking) | Lower-priority bet with risk of in-distribution accuracy drop. Adjacent to their #5 but more aggressive. Keep as deep-bench option. |

**What I missed in my prior list**: SBI / pseudo-fake. Their #3 is genuinely the highest-leverage option in either list. Should have been there.

**What they missed**: HSIC (nonlinear PD), GroupDRO, TTA. All three deserve places in the merged list.

## Merged ranked list

Ranked by `(theoretical fit × empirical evidence in this problem class) / (implementation cost × risk-of-regression)`. Each entry includes explicit caveats.

| # | Technique | Fit | Evidence | Cost | Risk | Summary |
|---|---|---|---|---|---|---|
| **1** | **Fourier amplitude perturbation + consistency** *(after pre-validation)* | High | Med | Low (~1d eng + 1 run) | Med-High before validation; Low after | Most direct attack on documented amplitude-domain shortcut axes (sharpness, luma, HF). **Pre-validate amplitude-vs-phase fake-signal location FIRST**. If validation passes, this is the highest-leverage cheap option. |
| **2** | **Customized + generic AugMix consistency (JSD)** | High | High | Med (~1.5d eng + 1 run) | Low-Med | Direct attack on "same person, different camera → score moves." Hybrid 60% substrate + 40% generic ops + JSD between original and 2 views. Composes with PD if PD clears. |
| **3** | **Self-Blended Images (SBI) pseudo-fake lane** | High | High | High (~3d eng + 1 run + ratio sweep) | Med | Strongest known answer to cross-method generalization. Method-agnostic blending-boundary signal. Directly addresses the PA-on-HDTF-collapse pattern. |
| **4** | **Test-Time Augmentation at inference** | Med-High | High | Very Low (~0.5d eng, no training run) | Very Low | Free productivity gain available **today**. Damps same-person drift by averaging score over N augmented views. Won't fix structural fail (xinhe-fake-2 at 16% recall) but cuts the 52× Dor drift to maybe 5-10×. **Run in parallel with everything else.** |
| **5** | **GroupDRO over (identity × capture-condition) groups** | High | Med-High | Med-High (~2d eng + group sampler + 1 run) | Med | Different geometry from GRL — outcome-equalization, not feature-invariance. Penalizes worst-group risk directly. Targets the user's "score moves on me" observation at the loss level. |
| **6** | **HSIC nonlinear penalty (PD-shifted successor)** | High *conditional* | Low-Med | Low (~30 LOC delta to PD) | Low | **Only worth running IF** PD's scorecard shows shortcut-shifting onto untargeted axes. Catches what Pearson misses (all dependence, not just linear). |
| **7** | **ViT patch-aware masking + consistency** | Med | Med | Med (~1d folded into #2; 2d standalone) | Low | Attacks the background/peripheral shortcut path specifically. **Fold into #2 as one of the JSD branches**, not a separate packet. |
| **8** | **Stylized training (texture-bias breaking)** | Med | Med | Med (~1.5d + style data prep) | Med-High | Forces shape-bias over texture-bias. Risk of in-distribution accuracy drop. Only after #1-#5 exhausted. |
| **9** | **Lightweight adversarial augmentation on substrate axes** | High | High | High (3-5× training cost) | High (instability + tuning) | Theoretically strongest. Practically last-resort due to cost and instability. |

## Pre-validation probes (cheap, do before committing to a packet)

| Probe | Cost | Decides |
|---|---|---|
| **Amplitude-vs-phase fake-signal probe** | 1 day, CPU | Whether #1 (Fourier aug) is gold or poison. Fits a tiny linear classifier on FFT magnitude vs phase from training data. |
| **TTA on cached frame reports** | 0.5 day, CPU | Quantifies same-person drift reduction at zero training cost. Re-scores P8A and E2B on identity-browser frames with N augmented views averaged. |
| **PD scorecard verdict (in flight)** | already paid | Determines whether explicit Pearson decorrelation works on the encoder, OR shifts onto untargeted axes. Output gates whether HSIC (#6) jumps the queue. |
| **SBI smoke on 100 frames** | 1 day, CPU | Validates the SBI pipeline produces visually-plausible pseudo-fakes on our face crops before committing to a 3d implementation packet. |

## Recommended packet sequence

**Conditional on PD scorecard verdict** (lands in 2-3h from this writeup).

### Branch A — PD clears 4/4 (positive)

1. **PE = AugMix-customized + PD** (`PE_AUGMIX_ON_PD_BASE`): single-lever AugMix consistency on top of PD's best ckpt. Tests whether invariance training composes with explicit decorrelation.
2. **PF = SBI pseudo-fake on PE base**: stack the structural lever on top once invariance has bitten.

### Branch B — PD muddles or shifts (most likely outcome per prototype)

1. **PE = SBI pseudo-fake on E2B base** (skip AugMix first because PD's failure tells us decorrelation alone is insufficient; jump to the paradigm shift).
2. **PF = HSIC nonlinear penalty on E2B** (in parallel — cheap follow-up to PD on a different loss class).
3. **PG = AugMix-customized on whichever base wins PE/PF**.

### Free actions to run in parallel (no GPU spend)

- **Implement TTA** (#4) for inference. Re-score the identity-browser. Get the same-person-drift quantification at zero cost.
- **Run amplitude-vs-phase probe** (#1 pre-validation). Decides whether Fourier aug is on the table at all.
- **SBI smoke** on 100 face crops to validate visual quality before committing eng time.

## Discipline rules carried forward (from R13 retros)

- **Single-lever testing** (`anti_shortcut_bundle_decomposition.md`). When stacking, include single-lever ablation slot. PA tried this and the v2-substrate confound still bit; SBI/AugMix proposals must declare the FT base + which legacy R13 levers are OFF.
- **Cross-substrate validation in close criterion** (`project_pa_does_not_generalize_to_hdtf_2026-05-05.md`). HDTF F4 gate must be in every packet's close criterion. PD already encodes this (4/4 includes F4); successors must too.
- **Pre-validation before training-time packets** (new rule). The PA-HDTF-walkback cost a packet's worth of engineering on a substrate-bound winner. Cheap CPU probes before GPU runs save packets.

## Open questions this doc does NOT settle

- **Why xinhe-fake-1/2/3 (no-glasses) miss**: is it the mask quality, the codec, the lighting, or interaction? Worth a focused audit on her variants alone — same person, different masks, same setup. Compare frame-level features across her catchable vs uncatchable variants.
- **Whether the chronic-6 reals are addressable via training or only via deployment-side filtering**. Job 3 + Job 11 audit showed the chronic-6 partition by ckpt; some ckpts handle some chronic identities cleanly. Ensemble or substrate-aware τ may be the right answer (but user said no ensemble).
- **Whether the ceiling is architectural** (saturate this CLIP-B16 family) or method-class (saturate "FT-from-CLIP" as a paradigm). E2B, P8A, E2B-scratch, L14-scratch all hit the same v2 cap. SBI is a training-distribution change, not architecture; if SBI doesn't crack it, architectural change becomes the next axis.

## Cross-references

- This doc: `docs/relaunch_handoffs/ANTI_SHORTCUT_TECHNIQUES_RANKED_2026-05-06.md`
- PD packet retro: `docs/packet_retrospectives/packets/PD.md`
- correlation_penalty thread: `docs/packet_retrospectives/threads/correlation_penalty_loss.md`
- Shortcut taxonomy: `docs/packet_retrospectives/threads/processing_signature_shortcut.md`
- Bundle discipline: `docs/packet_retrospectives/threads/anti_shortcut_bundle_decomposition.md`
- Identity browser dataset: `analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv`
- Open loops: `docs/packet_retrospectives/OPEN_LOOPS.md` — `corr-penalty-deployment-grade-verdict-pending`, `corr-penalty-frozen-head-shifting-axes-not-targeted`
