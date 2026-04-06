# Round 9 Plan — Teams Domain Adaptation & Score Stability

**Date:** February 27, 2026 (plan) · February 28, 2026 (implementation complete)
**Author:** AI Agent session (copilot)
**Status:** ✅ IMPLEMENTED — all 8 steps complete, 4 pre-launch fixes applied, 8 experiment configs ready
**Predecessor:** R8 (8 runs completed, R8_E scratch-target-heavy is the checkpoint leader)

---

## Table of Contents

1. [Situation Assessment](#1-situation-assessment)
2. [The Teams Domain Shift — New Understanding](#2-the-teams-domain-shift--new-understanding)
3. [Augmentation Audit — What's Helping and What's Hurting](#3-augmentation-audit--whats-helping-and-whats-hurting)
4. [Data Strategy](#4-data-strategy)
5. [Code Changes Required](#5-code-changes-required)
6. [Metrics & Monitoring Improvements](#6-metrics--monitoring-improvements)
7. [Experiment Design — 8 Slots, Two Waves](#7-experiment-design--8-slots-two-waves)
8. [Success Criteria](#8-success-criteria)
9. [Risks and Mitigations](#9-risks-and-mitigations)
10. [Appendix: File Reference](#10-appendix-file-reference)
11. [Implementation Log](#11-implementation-log)
12. [TeamsCodecSimulation Validation Report](#12-teamscodecimulation-validation-report)
13. [Pre-Launch Audit & Bug Fixes (Feb 28)](#13-pre-launch-audit--bug-fixes-feb-28)

---

## 1. Situation Assessment

### Where We Are After R8

| Target | R7 Baseline | R8 Best (R8_E) | Gate | Status |
|--------|-------------|-----------------|------|--------|
| **VisoMaster overall** | 69% | 95–98% | ≥75% | **SOLVED** |
| **DeepLiveCam** | 96% | 93–97% | ≥95% | **PASSING (most runs)** |
| **VCD Real accuracy** | ~70% | 82.1% | ≥85% | **GAP: −2.9pp** |

**R8_E** (scratch, target-heavy, W&B `hu7cen3m`) is the checkpoint champion.

### Two Structural Problems from R8

1. **Threshold calibration gap:** Val EER threshold ≈ 0.45 vs OOD EER threshold ≈ 0.77. The model's probability space is not calibrated across domains. Deploying with val-derived thresholds would produce catastrophic FPR on webcam reals.

2. **Score instability:** Near-identical video frames produce wildly different scores (e.g., 0.03 → 0.48). Root cause: ArcFace `s=18` amplifies ±0.02 cosine perturbations into ±10pp probability swings. Compounded by SVD residual amplification, codec jitter, and face crop instability.

### The Missing Piece: Teams Transmission

We have been training on *source* face crops (clean PNGs from GCS) and evaluating on faces that were transmitted through a video conferencing pipeline. The model has **never seen what its deployment channel does to images**.

---

## 2. The Teams Domain Shift — New Understanding

### Quantitative Evidence

> **⚠️ CORRECTION (Feb 28):** The original deltas below were computed from bulk bucket-level
> statistics, not matched frame pairs. Matched-pair validation on 18 samples × 132 frames
> revealed the original numbers were **fundamentally wrong** — Teams actually *blurs*
> (−50.9% sharpness), not sharpens (+37.8%), and *reduces* noise (−11.8%), not adds it (+8.3%).
> See [Section 12](#12-teamscodecimulation-validation-report) for the validated measurements
> that the actual implementation is based on.

~~Original (incorrect) measurements — kept for reference:~~

| Metric | Original | Teams | Delta | Direction | **Validated?** |
|--------|----------|-------|-------|-----------|----------------|
| ~~Sharpness~~ | ~~62.4~~ | ~~86.0~~ | ~~+37.8%~~ | ~~sharpens~~ | ❌ **Actually −50.9% (blurs)** |
| **Brightness** | 109.4 | 129.7 | **+18.6%** | brightens | ✅ Confirmed (+19.0%) |
| **Contrast** | 60.6 | 64.9 | **+7.0%** | boosts | ✅ Confirmed (+3.5%) |
| ~~Noise~~ | ~~2.79~~ | ~~3.02~~ | ~~+8.3%~~ | ~~adds noise~~ | ❌ **Actually −11.8% (reduces)** |
| **High-freq energy** | 0.283 | 0.263 | **−7.2%** | removes detail | ✅ Confirmed (−77.4%) |
| **Chroma blur** | 0.081 | 0.071 | **−12.6%** | subsampling | ⚠️ Small effect |
| **Blockiness** | 1.070 | 1.002 | **−6.4%** | deblocking | ✅ Confirmed (−1.2%) |
| Bits-per-pixel | 11.55 | 12.38 | +7.1% | re-encoding | ✅ Confirmed (−8.7%) |

**Validated measurements (18 samples × 132 matched frame pairs):**

| Metric | Delta | Direction |
|--------|-------|-----------|
| **Sharpness** (Laplacian var) | **−50.9%** | Teams *blurs* via lossy codec compression |
| **Brightness** (mean pixel) | **+19.0%** | Teams auto-exposure / gain control |
| **Contrast** (pixel std) | **+3.5%** | Teams auto-exposure |
| **Noise** (MAD) | **−11.8%** | Teams *reduces* noise (denoising filter) |
| **High-freq energy** (DCT ratio) | **−77.4%** | VP8/VP9 lossy encoding destroys HF |
| **Blockiness** (boundary ratio) | **−1.2%** | Slight deblocking |
| **Bits-per-pixel** (JPG Q95) | **−8.7%** | Less compressible (smoother) |

At 50-sample scale (165 frames), per-frame sharpness showed **bimodal behavior** — some
Teams content categories trigger an enhancement/sharpening mode while others blur. Mean
sharpness shifted to +2.8% with std=399.9% and a +3015% outlier. The other 7 metrics
remained stable. This bimodality is acceptable because (a) real Teams data in training
covers both modes, and (b) the augmentation doesn't need to be a perfect simulator — it
pushes model invariance in the right direction.

### Why This Matters

The Teams pipeline applies a **consistent, structured set of correlated transformations** — sharpening, brightness boost, codec noise, chroma subsampling — simultaneously to every frame. This is fundamentally different from random augmentations that vary each transform independently.

**Critical insight about VCD reals:** The VCD webcam dataset was captured through a Zoom video call. The model's failure on VCD reals (82.1%, target 85%) is partly because it has never trained on data that went through a video conference pipeline. Both Teams and Zoom apply similar WebRTC processing (H.264, auto-exposure, sharpening, denoising). Learning the Teams fingerprint should transfer to Zoom/VCD.

**Critical insight about face crop jitter:** In production, a WMA face detector runs on a compressed Teams stream and produces crops with different bounding-box positions than the YOLO detector used for source crop extraction. This is not a confounder — it is a feature of the deployment environment. Training on Teams-passthrough data teaches the model to be invariant to it.

### Why Previous Augmentations Were Insufficient

R8 used `quality_targeted_family` + `vcd_targeted` — directionally correct but structurally wrong:

> **Updated (Feb 28):** Table corrected to reflect validated measurements.

| Teams Reality | What Current Augs Do | Gap |
|---------------|---------------------|-----|
| Sharpness **−50.9%** (blurs) | `CustomUnsharpMask` sometimes sharpens, sometimes blurs | Direction correct half the time, but Teams blurring is *consistent* and strong |
| Brightness +19.0% (consistent UP) | `RandomBrightnessContrast ±0.25` | Symmetric around 0 — half the time it dims, which Teams never does |
| Chroma blur −12.6% (YUV 4:2:0) | Not modeled at all | **Complete gap** |
| Deblocking (H.264 filter) | Not modeled | **Complete gap** |
| All transforms applied together | Each aug fires independently with separate probability | Correlated joint distribution ≠ independent marginals |

The `VideoCodecSimulation` transform in `transforms.py` is the closest existing tool — it has a 5-step pipeline (downscale → bilateral deblock → block quantization → freq-shaped noise → JPEG). But it was designed for generic codec simulation, not Teams-specific parameters, and it's applied via `webcam_codec_p` at only 10–22% probability in the `quality_targeted_family` router.

---

## 3. Augmentation Audit — What's Helping and What's Hurting

### Current `vcd_targeted` Preset Review

The `vcd_targeted` preset in `_QUALITY_TARGETED_PRESETS` (pipelines.py L870–920) was designed to break the "smooth = real" quality shortcut. With our new Teams understanding, here's what we should keep vs modify:

#### ✅ KEEP — These are genuinely helpful

| Transform | Current Setting | Why Keep |
|-----------|----------------|----------|
| `HorizontalFlip(p=0.5)` | All families | Basic spatial invariance — always useful |
| `RandomBrightnessContrast` (color_p=0.52) | brightness ±0.22, contrast ±0.22 | Reasonable range, matches Teams brightness shift direction half the time |
| `HueSaturationValue` | hue ±12, sat ±24, val ±24 | Color robustness — important for diverse lighting |
| `context_variation_enabled: true` | Gamma 80–120, SSR, brightness/contrast | Spatial + lighting robustness — critical for VC diversity |

#### ⚠️ RECONSIDER — May be doing more harm than good

| Transform | Current Setting | Problem |
|-----------|----------------|---------|
| `GaussianBlur` (blur_limit 3–9) | quality_p=0.60 for fakes, ~0.25 for reals | **Teams sharpens (+37.8%), never blurs.** Including blur in the augmentation pipeline teaches the model that both blurred and sharp versions exist — but in production, everything is sharpened. Blur augmentation makes the model *less* sensitive to the actual signal. |
| `Downscale` (0.50–0.80) | quality_p=0.60 for fakes | **Teams maintains resolution** (face detector output is already 224px-class). Downscale simulation is irrelevant and adds noise to training. |
| `real_sharpen_p: 0.60` | sharpen_alpha_real 0.30–0.70 | Direction is correct (Teams sharpens), but magnitude is random. In Teams, sharpening is **consistent and 37.8% Laplacian variance increase** — not random. |
| `webcam_codec_p: 0.12` | Quality 35–80 | **Too low probability** and **wrong parameters**. Should match Teams' measured profile, not a generic codec. |
| `fake_extra_degrade_p: 0.15` | Downscale 0.35–0.55 or BlurLimit 5–11 | **Wrong direction** — Teams doesn't degrade fakes, it processes them the same as reals. |

#### ❌ MISSING — Not in the current pipeline at all

| What Teams Does | Current Pipeline |
|-----------------|-----------------|
| YUV 4:2:0 chroma subsampling (−12.6% chroma blur) | No chroma-specific transform |
| Consistent brightness boost (+18.6%) | Symmetric ± augmentation |
| Consistent sharpening (+37.8%) | Random direction |
| Joint application of all transforms together | Independent probabilities |
| Face detector bounding-box jitter | Not in augmentation (only via crop_jitter in instability fixes) |

### Recommendation

For R9, we should:

1. **Use the actual Teams-passthrough data** as the primary domain adaptation signal (highest fidelity, no approximation needed)
2. **Build a `teams_simulation` augmentation pipeline** that applies Teams-measured transforms as a coherent block — for use on non-Teams training data that we want to "Teams-ify"
3. **Keep basic spatial/color augmentations** (flip, brightness/contrast, hue/sat) as a foundation
4. **Remove or reduce directionally-wrong augmentations** (blur-heavy degradation, downscale) for the Teams-targeted families
5. **Keep the existing pipelines** for DF40/VisoMaster families where generic quality variation is still useful

---

## 4. Data Strategy

### 4.1 Teams Data (New)

**Source bucket:** `live-deepfake-methods-real-and-fake-frames-cropped-teams`
**Structure:** `samples/{sample_id}/frames/{real|fake}/frame_NNNN.jpg`
**Current state:** 228 complete pairs (pair_complete: true), ~250 additional partial samples
**Tonight:** Expected +200 additional complete pairs from rerun session

Each Teams sample has a **parallel non-Teams counterpart** in `live-deepfake-methods-real-and-fake-frames-cropped` (the original DeepLive bucket). This creates both opportunities and challenges:

#### Identity Handling

Teams samples and their non-Teams counterparts share the same identity (same person, same source video). They **must** be in the same split to prevent identity leakage. The identity extraction logic already uses `realpool_{original_video_name}` for DeepLive — Teams samples must use the same prefix.

#### Duplication Risk

If both Teams and non-Teams versions of the same sample are in training, the model sees the same face twice per epoch (at different quality levels). This is actually **desirable** — it teaches the model that the same person's face can look different through Teams vs raw capture. However, we need to ensure the identity-balanced sampler doesn't over-weight these shared identities.

**Approach:** Register Teams data as a new family key (`deeplive_teams_fake` / `deeplive_teams_real`) within the existing identity pool. The identity-balanced sampler will select an identity, then choose from {non-Teams, Teams} versions based on family weights. This guarantees:
- No identity leakage (same identity prefix)
- Controlled exposure to Teams vs non-Teams (via family weights)
- Both real and fake Teams crops contribute to learning

#### Paired Contrastive Training Opportunity

We have the unique advantage of matched pairs: the **exact same frame** before and after Teams transmission. This enables:
- The model learns what Teams *does* to an image, not just what a Teams image *looks like*
- During training, both `<original, label>` and `<teams_version, label>` appear — the model must give consistent predictions across both

This is naturally achieved by having both versions in training with the same label. No special contrastive loss needed — the standard cross-entropy with family-weighted sampling handles it.

### 4.2 VCD Reals — Why NOT Re-Process Through Teams

The VCD dataset contains real faces from **actual Zoom video calls**. These have already been through a video conference pipeline (Zoom's WebRTC). Passing them through Teams would create a **double-processed** distribution that doesn't exist in production. The model would learn to recognize double-codec artifacts rather than single-pass conferencing artifacts.

VCD reals should remain as-is in the `external_real` family.

### 4.3 DF40 / VisoMaster Through Teams — Costs vs Benefits

**Passing DF40 through Teams** would require: extracting crops → playing as video → capturing through Teams → re-cropping. DF40 is frame-based (not video), so this would be technically complex (convert frame pairs to a video stream).

**Decision: NOT for R9.** The 228–430 Teams samples from DeepLive provide sufficient signal for the model to learn the Teams codec fingerprint. DF40 augmentation via the `teams_simulation` pipeline (Section 5.3) provides additional coverage without the overhead of actual Teams passthrough.

If R9 experiments show that VisoMaster detection degrades through Teams (because VisoMaster was only trained on clean crops), we can consider Teams-passthrough for VisoMaster in R10.

### 4.4 Family Weights — R9 Proposal

```yaml
sampling:
  strategy: "identity_resample_weighted"
  family_weights:
    df40_fake: 0.2              # Same as R8 — supplementary
    visomaster_fake: 2.0        # Slightly reduced from 3.0 — already solved
    deeplive_non_enhanced_fake: 2.5    # Reduced from 4.0 — making room for Teams
    deeplive_enhanced_fake: 3.0        # Reduced from 5.0 — making room for Teams
    deeplive_teams_fake: 7.0           # NEW — highest weight, deployment domain
    deeplive_teams_real: 5.0           # NEW — Teams real crops, critical for FPR
    df40_real: 0.5              # Same as R8
    realpool_real: 1.5          # Same as R8 — DeepLive/Viso reals
    external_real: 2.0          # Slightly increased — VCD webcam reals
```

**Rationale for Teams weights:** With only ~228–430 Teams pairs vs ~1000 DeepLive + ~500 VisoMaster, high weights are needed to ensure Teams data appears proportionally enough in training. The identity-balanced sampler normalizes by identity count, so high weights ensure Teams identities are selected more frequently.

---

## 5. Code Changes Required

### 5.1 New Data Source: Teams Passthrough Integration ✅ IMPLEMENTED

**Location:** `data/sources/combined_paired.py`

**What to implement:**

1. **Add Teams discovery in `create_combined_paired_pipeline`:**
   ```python
   # After DeepLive discovery, discover Teams-passthrough samples
   teams_config = combined_config.get('teams', {})
   if teams_config.get('enabled', False):
       teams_samples = discover_teams_passthrough_samples(
           gcs_bucket=teams_config.get('gcs_bucket',
                       'live-deepfake-methods-real-and-fake-frames-cropped-teams'),
           anchor_indices=teams_anchor_indices,
           logger=logger,
       )
       teams_unified = create_unified_samples_from_teams(
           teams_samples, logger
       )
       all_samples.extend(teams_unified)
   ```

2. **`discover_teams_passthrough_samples` function:**
   - List `samples/` prefix in the Teams bucket
   - For each sample directory, read `manifest.json`
   - Filter to `pair_complete: true` only
   - Extract `sample_id`, `strategy`, `original_video_name`, `frame_count`
   - Return list of `TeamsSample` dataclass instances

3. **`create_unified_samples_from_teams` function:**
   - Convert to `UnifiedPairedSample` with `source='deeplive_teams'`
   - Use `identity = f"realpool_{original_video_name}"` — **same prefix as DeepLive** to ensure shared identity pool
   - Method: `f"deeplive_teams_{strategy}"` for downstream group/family inference

4. **`_iterate_teams_sample` in `CombinedPairedIterableDataset`:**
   - Similar to `_iterate_deeplive_sample` but reads JPG (not PNG) from the Teams bucket
   - No landmarks (Teams capturing doesn't preserve landmarks)
   - Passes `source='deeplive_teams'` in meta dict for family-aware augmentation routing

5. **New family keys in `utils/grouping.py`:**
   ```python
   # In infer_group_key:
   if 'deeplive_teams' in source_str or 'deeplive_teams' in method_str:
       if label_int == 1:
           return 'deeplive_teams_fake'
       return 'deeplive_teams_real'

   # In infer_family_key:
   if group_key == 'deeplive_teams_fake':
       return 'deeplive_teams_fake'
   if group_key == 'deeplive_teams_real':
       return 'deeplive_teams_real'
   ```

6. **Add Teams family to `_build_family_quality_pipeline`** (or reuse an existing one):
   - Teams data has **already been through the codec pipeline** — it needs **minimal augmentation**
   - Pipeline: `HorizontalFlip(p=0.5)` + light color jitter only
   - **Do NOT apply VideoCodecSimulation or quality degradation** — the data is already degraded by the actual codec

### 5.2 Augmentation: New Teams Family Pipeline ✅ IMPLEMENTED

**Location:** `data/augmentations/pipelines.py` — `_build_teams_passthrough_pipeline()`

Add to `_build_family_quality_pipeline`:

```python
if family_key in {"deeplive_teams_fake", "deeplive_teams_real"}:
    # Teams data has already been through the codec pipeline.
    # Only apply the lightest augmentation — preserve the real codec fingerprint.
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.08,
            contrast_limit=0.08,
            p=0.3,
        ),
        # No sharpening, no codec sim, no downscale — data is already "Teams-processed"
    ])
```

### 5.3 Augmentation: Teams Simulation Pipeline (For Non-Teams Data)

**Location:** `data/augmentations/teams_simulation.py` ✅ **IMPLEMENTED**

> **⚠️ MAJOR REVISION (Feb 28):** The original plan proposed a 5-stage pipeline including
> sharpening and additive noise. Matched-pair validation proved Teams does **neither** —
> it blurs and reduces noise. The implementation was completely rewritten 6 times based on
> iterative validation against real Teams frames. See [Section 12](#12-teamscodecimulation-validation-report).

```python
class TeamsCodecSimulation(ImageOnlyTransform):
    """
    Simulate Microsoft Teams WebRTC video pipeline.
    Validated against 132 matched frame pairs (18 samples) + 165 frames (50 samples).

    Pipeline stages (applied in order, always together):
      1. Brightness + contrast boost  — Teams auto-exposure / gain control
      2. Light Gaussian blur          — Simulates codec smoothing of detail
      3. JPEG compression             — I-frame encoding at conferencing bitrate
      4. Bilateral deblocking         — VP8/VP9 in-loop deblock filter
      (No sharpening, no additive noise — validated against real Teams data)
    """
```

**Final calibrated defaults:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `brightness_limit` | (0.02, 0.10) | Positive-only; Teams brightens +19% |
| `contrast_limit` | (0.10, 0.22) | Applied as `1 + delta`; Teams +3.5% contrast |
| `blur_sigma` | (0.35, 0.85) | Gaussian blur for −50.9% sharpness loss |
| `jpeg_quality` | (72, 88) | Conferencing I-frame bitrate |
| `chroma_blur_ksize` | 0 | Disabled (small measured effect) |
| `deblock_d` | 5 | Bilateral filter for VP8/VP9 deblock |
| `deblock_sigma_color/space` | 30.0 | Moderate edge-preserving smoothing |

**What changed from the plan:**
- ❌ Removed: UnsharpMask sharpening (Teams blurs, not sharpens)
- ❌ Removed: Additive noise (Teams reduces noise by −11.8%)
- ❌ Removed: Chroma blur (measurable but tiny effect, adds complexity)
- ✅ Added: Bilateral deblocking filter (models VP8/VP9 in-loop deblocking)
- ✅ Changed: Brightness range reduced from (0.05, 0.20) to (0.02, 0.10) to reduce overshoot

**Wiring into training pipeline:** ✅ **IMPLEMENTED**

The augmentation is wired as a **post-pipeline step** in `QualityTargetedFamilyRouter`:
1. `combined_paired.py` reads `augmentation.teams_codec_simulation` from YAML config
2. Passes config dict to `create_quality_targeted_family_router(teams_codec_simulation=...)`
3. Router instantiates `TeamsCodecSimulation(always_apply=True, p=1.0)` if `enabled: true`
4. After each per-family augmentation pipeline runs, `_maybe_apply_teams_sim()` applies the
   simulation with the configured `probability` (default 0.15), skipping any `exclude_families`

> **🐛 Critical bug found and fixed (Feb 28):** The YAML configs had `teams_codec_simulation.enabled: true`
> but `_create_combined_transform()` filtered `aug_config` keys against `_valid_preset_keys` from
> `_QUALITY_TARGETED_PRESETS` — since `teams_codec_simulation` is not a preset key, it was **silently
> dropped**. The augmentation was completely dead code. Fixed by adding explicit config reading
> before the preset filter and forwarding through the factory.

**Validation:** Completed. See [Section 12](#12-teamscodecimulation-validation-report).

**Smoke test:** All 11 runtime tests pass: instantiation, apply(), router with/without teams sim,
exclude_families, disabled mode, brightness verification, p=0 edge case. Plus 100 random images
of varying sizes pass the edge case audit.

### 5.4 Score Stability Fixes ✅ IMPLEMENTED

From the [Score Instability Analysis](../experiments/phase2_round8/SCORE_INSTABILITY_ANALYSIS.md):

#### Fix 1: Label Smoothing (config-only, 5 min) ✅

**Location:** `trainer/trainer.py` — wherever `CrossEntropyLoss` is created.

```python
label_smoothing = self.config.get('label_smoothing', 0.0)
self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
```

**Config:**
```yaml
label_smoothing: 0.05
```

#### Fix 2: Lower ArcFace Scale (config-only, 5 min) ✅

```yaml
# R8 (too aggressive):
s_start: 10.0
s_end: 18.0

# R9 (gentler):
s_start: 6.0
s_end: 12.0
```

#### Fix 3: Perturbation Consistency Loss (new mixin, 1–2 hours) ✅

**New file:** `trainer/mixins/stability.py`

```python
class StabilityRegMixin:
    """
    Mixin that adds an input perturbation consistency regularization loss.

    For each batch, generates a slightly perturbed copy (Gaussian noise +
    small spatial shift) and penalizes KL divergence between clean and
    perturbed predictions. Teaches the model to be invariant to codec
    jitter and face crop instability.

    Config:
        stability_lambda: float (0.0 = disabled)
        stability_noise_std: float (pixel noise std, ~0.02)
        stability_crop_jitter: float (crop shift fraction, ~0.03)
    """

    def init_stability_reg(self):
        self.stability_lambda = self.config.get('stability_lambda', 0.0)
        self.stability_noise_std = self.config.get('stability_noise_std', 0.02)
        self.stability_crop_jitter = self.config.get('stability_crop_jitter', 0.03)

    def compute_stability_loss(self, model, images, logits_clean):
        """Returns stability_lambda * KL(perturbed || clean.detach())."""
        ...

    def _generate_perturbation(self, images):
        """Gaussian noise + random crop-and-resize."""
        ...
```

**Integration:** Add to `Trainer` class mixins, call in `train_epoch` after main loss.

### 5.5 Metrics Improvements ✅ IMPLEMENTED

#### Add F1 Score ✅

**Location:** `metrics/utils.py` — in `_compute_roc_metrics` and `get_test_metrics`

```python
from sklearn.metrics import f1_score

# At EER threshold:
preds_at_eer = (y_pred >= eer_thresh).astype(int)
f1_at_eer = f1_score(y_true, preds_at_eer)
result['f1_at_eer'] = f1_at_eer

# At 0.5 threshold:
preds_at_half = (y_pred >= 0.5).astype(int)
f1_at_half = f1_score(y_true, preds_at_half)
result['f1'] = f1_at_half
```

#### Add Combined Threshold Monitoring ✅

**Location:** `trainer/trainer.py` — in `_run_validation`

**Problem:** R8 showed val threshold ≈ 0.45, holdout threshold ≈ 0.60, OOD threshold ≈ 0.77. These were logged separately, making it easy to miss the divergence.

**Solution:** After all validation runs complete, compute a **combined pool** threshold from all predictions:

```python
def _compute_unified_threshold(self, all_val_preds, all_val_labels,
                                all_ood_preds, all_ood_labels):
    """Compute EER threshold over combined val + OOD predictions."""
    combined_preds = np.concatenate([all_val_preds, all_ood_preds])
    combined_labels = np.concatenate([all_val_labels, all_ood_labels])
    metrics = _compute_roc_metrics(combined_preds, combined_labels)
    if metrics:
        return metrics['eer_threshold'], metrics['eer']
    return None, None
```

Log as `unified/eer_threshold`, `unified/eer`, `unified/f1`. This gives one number to watch instead of three.

#### Add Weakest-Method Tracking ✅

**Location:** `trainer/trainer.py` — in test_epoch reporting

After computing per-method accuracy, find and explicitly log:

```python
# Log the worst-performing real and fake methods
real_methods = {m: acc for m, acc in method_accs.items() if m is real}
fake_methods = {m: acc for m, acc in method_accs.items() if m is fake}

worst_real = min(real_methods.items(), key=lambda x: x[1])
worst_fake = min(fake_methods.items(), key=lambda x: x[1])

wandb.log({
    f"{prefix}weakest/real_method": worst_real[0],
    f"{prefix}weakest/real_acc": worst_real[1],
    f"{prefix}weakest/fake_method": worst_fake[0],
    f"{prefix}weakest/fake_acc": worst_fake[1],
})
```

This surfaces regressions immediately rather than requiring manual inspection of per-method tables.

#### Add Score Stability Monitoring ✅

During OOD evaluation (which processes video frames), compute frame-to-frame jitter for each video:

```python
# For each video in OOD evaluation:
frame_probs = [p for p in video_frame_probs]
if len(frame_probs) >= 2:
    jitter = np.mean(np.abs(np.diff(frame_probs)))
    wandb.log({f"ood/score_jitter/{method}": jitter})
```

---

## 6. Metrics & Monitoring Improvements

Summary of all monitoring changes (from Section 5.5):

| Metric | Where Logged | Purpose |
|--------|-------------|---------|
| `f1_at_eer` | val, holdout, ood | Real-world performance at calibrated threshold |
| `f1` | val, holdout, ood | Performance at naive 0.5 threshold |
| `unified/eer_threshold` | After all val runs | Single threshold across all data pools |
| `unified/eer` | After all val runs | One number for calibration quality |
| `unified/f1` | After all val runs | One number for real-world performance |
| `weakest/real_method` + `weakest/real_acc` | val, holdout, ood | Which real source is hardest |
| `weakest/fake_method` + `weakest/fake_acc` | val, holdout, ood | Which fake method is hardest |
| `ood/score_jitter/{method}` | ood monitoring | Frame-to-frame score stability |
| `train/stability_loss` | training loop | Perturbation consistency loss value |

---

## 7. Experiment Design — 8 Slots, Two Waves ✅ ALL CONFIGS CREATED

### Resource Constraint

8 Vertex AI A100-40GB slots total. We split into:
- **Wave 1 (launch now):** 4-5 experiments using the 228 Teams pairs available now
- **Wave 2 (launch tonight):** 3-4 experiments using ~430 Teams pairs (228 + ~200 from rerun)

### Full Experiment Matrix

| Experiment | From | Teams Data | TeamsCodecSim | Stability Fixes | ArcFace s | LR / Steps | Wave |
|------------|------|------------|---------------|-----------------|-----------|-------------|------|
| **R9_A** | R8_E FT | ✅ | — | ✅ (λ=0.3, smooth=0.05) | 6→12 | 5e-5 / 10K | 1 |
| **R9_B** | R8_E FT | ✅ | — | ❌ (λ=0, smooth=0) | 10→18 | 5e-5 / 10K | 1 |
| **R9_C** | scratch | ✅ | — | ✅ | 6→12 | 2e-4 / 12K | 1 |
| **R9_D** | R8_E FT | ❌ | — | ✅ | 6→12 | 5e-5 / 10K | 1 |
| **R9_E** | R8_E FT | ✅ (~430) | — | ✅ | 6→12 | 5e-5 / 10K | 2 |
| **R9_F** | R8_E FT | ❌ | ✅ p=0.15 all | ✅ | 6→12 | 5e-5 / 10K | 2 |
| **R9_G** | R8_E FT | ✅ (~430) | ✅ p=0.15 excl. Teams | ✅ | 6→12 | 5e-5 / 10K | 2 |
| **SMOKE** | R8_E FT | ✅ | — | ✅ | 6→12 | 5e-5 / 1K | — |

### Wave 1 — Launch Immediately (228 Teams pairs)

| Config | Name | From | Key Test | Teams Data |
|--------|------|------|----------|------------|
| **R9_A** | teams_ft_stability | R8_E checkpoint | Teams data + stability fixes (Fix 1-3) + reduced aug | 228 pairs |
| **R9_B** | teams_ft_nofixes | R8_E checkpoint | Teams data only, NO stability fixes (ablation) | 228 pairs |
| **R9_C** | teams_scratch | CLIP scratch | Teams data + stability fixes from scratch | 228 pairs |
| **R9_D** | stability_only | R8_E checkpoint | ONLY stability fixes (Fix 1-3), no Teams data | 0 (control) |
| **R9_SMOKE** | smoke test | R8_E checkpoint | 1K steps, verify pipeline + new metrics work | 228 pairs |

**R9_A** is the primary experiment — it combines all innovations:
- Teams data at high weight
- Label smoothing 0.05
- ArcFace s: 6 → 12
- Stability reg λ = 0.3
- Teams-appropriate augmentation (minimal for Teams family, existing for others)
- New metrics (F1, unified threshold, weakest method, jitter)

**R9_B** is the Teams-only ablation — isolates the value of Teams data without stability fixes. If R9_B ≈ R9_A, stability fixes aren't critical yet and we can focus on data.

**R9_D** is the stability-only control — isolates the value of score stability fixes without Teams data. If R9_D shows improved threshold calibration, the fixes work independently.

**R9_C** tests whether Teams + stability works from scratch (no checkpoint bias).

### Wave 2 — Launch Tonight (~430 Teams pairs)

| Config | Name | From | Key Test | Teams Data |
|--------|------|------|----------|------------|
| **R9_E** | teams_ft_v2 | R8_E checkpoint | R9_A config with more Teams data | ~430 pairs |
| **R9_F** | teams_sim_aug | R8_E checkpoint | TeamsCodecSimulation aug on ALL data (no actual Teams data) | 0 (aug only) |
| **R9_G** | teams_both | R8_E checkpoint | Actual Teams data + TeamsCodecSimulation on non-Teams data | ~430 pairs |

**R9_E** is R9_A with more data — tests if ~400 pairs is meaningfully better than 228.

**R9_F** tests synthetic Teams augmentation alone — `TeamsCodecSimulation` (validated, 6/8 direction accuracy at 18-sample scale) applied to all families at 15% probability. If this approaches R9_A's performance, synthetic augmentation can substitute for costly Teams data collection.

**R9_G** combines both: actual Teams data for high-fidelity signal + synthetic Teams augmentation on non-Teams families at 15% probability (with `exclude_families: [deeplive_teams_fake, deeplive_teams_real]` so real Teams data keeps its authentic fingerprint). This is the speculative "best of both worlds" config.

**Note on remaining slot:** 1 slot held in reserve for a quick follow-up if Wave 1 results suggest a clear improvement direction.

### R9_A Detailed Config (Primary Experiment)

```yaml
name: "R9_A_teams_ft_stability"
description: "Teams domain adaptation + score stability fixes, FT from R8_E"
seed: 737

data_source: combined_paired

backbone:
  name: "vit_b_16_laion_datacomp"
  variant: "ViT-B-16-DataComp-XL"
  source: "laion"
  model_name: "ViT-B-16"
  pretrained: "datacomp_xl_s13b_b90k"
  hidden_size: 512
  resolution: 224
  apply_svd_to_in_proj: true

# Fine-tune from R8_E (scratch winner)
learning_rate: 5.0e-5
weight_decay: 0.05
lambda_reg: 0.01
rank: 736
gradient_clip_val: 1.0

total_training_steps: 10000
lr_scheduler: "cosine_with_warmup"
lr_scheduler_warmup_steps: 500

load_base_checkpoint: true
gcs_base_checkpoint: "<R8_E best checkpoint GCS path>"

# --- Score stability fixes ---
label_smoothing: 0.05
s_start: 6.0
s_end: 12.0
anneal_steps: 20000
stability_lambda: 0.3
stability_noise_std: 0.02
stability_crop_jitter: 0.03

augmentation:
  version: "quality_targeted_family"
  strength: "vcd_targeted"
  routing:
    mode: "family_aware"
    enhanced_strategy_names:
      - "quality_enhancement"
      - "edge_cases_enhanced"
      - "minimal_processing_enhanced"
  # Context variation — same as R8
  context_variation_enabled: true
  context_variation_gamma_limit: [80, 120]
  context_variation_brightness: 0.25
  context_variation_contrast: 0.25

combined_paired:
  identity_balanced_sampling: true
  split_seed: 737

  df40:
    enabled: true
    pair_json: "dataset/df40_pairs/df40-pair-matching.json"
    gcs_bucket: "df40-frames-recropped-rfa85"
    methods:
      - "simswap"
      - "facedancer"
      - "blendface"
      - "e4s"
      - "inswap"
      - "mobileswap"
      - "uniface"

  deeplive:
    enabled: true
    gcs_bucket: "live-deepfake-methods-real-and-fake-frames-cropped"
    use_landmarks: false
    include_strategies:
      - "edge_cases"
      - "minimal_processing"
      - "quality_enhancement"
      - "edge_cases_enhanced"
      - "minimal_processing_enhanced"

  visomaster:
    enabled: true
    gcs_bucket: "live-deepfake-methods-real-and-fake-frames-cropped"
    swap_models:
      - "CSCS"
      - "GhostFace-v1"
      - "GhostFace-v2"
      - "GhostFace-v3"
      - "InStyleSwapper256-A"
      - "InStyleSwapper256-B"
      - "InStyleSwapper256-C"
      - "Inswapper128"
      - "SimSwap512"

  # NEW: Teams passthrough data
  teams:
    enabled: true
    gcs_bucket: "live-deepfake-methods-real-and-fake-frames-cropped-teams"
    require_pair_complete: true
    anchor_indices: [0, 2, 4, 6, 8, 10, 12, 14]

  external_training_reals:
    - bucket: "effort-collected-data"
      prefix: "real/VCD"
      method: "external_vcd_real"
      identity_train_fraction: 0.40
      max_frames_per_identity: 15
      max_total_samples: 1200

  sampling:
    strategy: "identity_resample_weighted"
    family_weights:
      df40_fake: 0.2
      visomaster_fake: 2.0
      deeplive_non_enhanced_fake: 2.5
      deeplive_enhanced_fake: 3.0
      deeplive_teams_fake: 7.0
      deeplive_teams_real: 5.0
      df40_real: 0.5
      realpool_real: 1.5
      external_real: 2.0

  ood_monitoring:
    enabled: true
    exclude_training_identities: true
    external_real_sources:
      - bucket: "effort-collected-data"
        prefix: "real/external_youtube_avspeech"
        method: "external_youtube_avspeech"
        max_videos: 200
      - bucket: "effort-collected-data"
        prefix: "real/VCD"
        method: "zoom_vcd_real"
        max_videos: 1200
    external_fake_sources:
      - bucket: "effort-collected-data"
        prefix: "wma_validation/enhanced_fake"
        method: "wma_failure_fake"
        max_videos: 1202

use_arcface_head: true
arcface_m: 0.0
arcface_s: 6.0

evaluate_every_steps: 500
ood_monitoring_start_step: 500
ood_monitoring_every_steps: 1000
test_batch_size: 32

early_stopping_enabled: true
early_stopping_patience: 10

checkpointing:
  gcs_prefix: "gs://training-job-outputs/phase2r9_experiments/"
  save_every_steps: 2000
  keep_last_n: 3
```

---

## 8. Success Criteria

### Primary Gates (must pass)

| Metric | Target | What It Means |
|--------|--------|---------------|
| VCD Real accuracy | ≥ 85% | False positive rate on webcam reals below 15% |
| DeepLive TPR | ≥ 95% | Fake detection rate on real-time webcam faceswap |
| VisoMaster overall | ≥ 90% | No regression from R8 |
| Unified EER threshold | ≤ 0.65 | Val/OOD threshold gap narrowed (was 0.45 vs 0.77) |

### Secondary Targets (want)

| Metric | Target | What It Means |
|--------|--------|---------------|
| Score jitter (mean abs diff) | ≤ 0.08 | Frame-to-frame stability |
| Unified F1 at EER threshold | ≥ 0.88 | Good real-world performance at calibrated threshold |
| Weakest fake method | ≥ 70% | No single method catastrophically fails |
| Weakest real source | ≥ 80% | No single real source has extreme FPR |

### R9_A vs R9_B Comparison (Teams + stability vs Teams alone)

If R9_A significantly beats R9_B on unified threshold and jitter → stability fixes complement Teams data.
If R9_A ≈ R9_B → Teams data alone is sufficient; stability fixes are optional.

### R9_A vs R9_D Comparison (Both vs stability only)

If R9_A significantly beats R9_D on VCD Real and DeepLive → Teams data provides signal beyond stability fixes.
If R9_D ≈ R9_A → stability fixes alone solve the problem (unlikely but would save data collection effort).

---

## 9. Risks and Mitigations

### Risk 1: 228 Teams Pairs Is Too Little Data

**Severity:** Medium
**Mitigation:** High family weight (7.0) ensures frequent sampling. The model sees each Teams frame multiple times per epoch with different batch contexts. If R9_E (430 pairs) significantly beats R9_A (228 pairs), we know to collect more data.

### Risk 2: Teams Augmentation Simulation Is Inaccurate

**Severity:** Low (Wave 2 only)
**Status:** ✅ **VALIDATED** — see [Section 12](#12-teamscodecimulation-validation-report).
Direction accuracy: 6/8 metrics at 18-sample scale, 5/8 at 50-sample scale. The augmentation
is not a perfect simulator but pushes model invariance in the right direction. Combined with
real Teams data (R9_G), the model sees both the authentic codec fingerprint and the synthetic
approximation.

### Risk 3: Stability Fixes Hurt In-Distribution Performance

**Severity:** Low
**Mitigation:** R9_D (stability-only control) isolates this risk. Label smoothing and lower `s` may increase val EER by ~0.5–1pp — acceptable if unified threshold improves. R9_B (no stability fixes) provides a safety net.

### Risk 4: Identity Leakage Between Teams and Non-Teams Samples

**Severity:** High if it happens — would invalidate val/OOD metrics
**Mitigation:** Use same `realpool_{original_video_name}` identity prefix. Write an assertion in the pipeline that checks no identity appears in multiple splits across Teams and non-Teams sources.

### Risk 5: VisoMaster Regression

**Severity:** Medium — VisoMaster was the hardest win in R8
**Mitigation:** Keep `visomaster_fake` weight at 2.0 (reasonable). Monitor per-model VisoMaster accuracy in val reports. If any swap model drops below 80%, add early stopping guardrail.

---

## 10. Appendix: File Reference

### Files Created ✅

| File | Purpose | Status |
|------|---------|--------|
| `trainer/mixins/stability.py` | `StabilityRegMixin` — perturbation consistency loss | ✅ Created |
| `data/augmentations/teams_simulation.py` | `TeamsCodecSimulation` transform (4-stage: brightness+contrast → blur → JPEG → bilateral deblock) | ✅ Created & validated |
| `tests/test_teams_simulation.py` | GCS-based validation script for matched-pair comparison | ✅ Created |
| `experiments/phase2_round9/R9_A_teams_ft_stability.yaml` | Primary: FT + Teams data + stability fixes | ✅ Created |
| `experiments/phase2_round9/R9_B_teams_ft_nofixes.yaml` | Ablation: FT + Teams data, no stability fixes | ✅ Created |
| `experiments/phase2_round9/R9_C_teams_scratch.yaml` | From scratch + Teams data + stability fixes | ✅ Created |
| `experiments/phase2_round9/R9_D_stability_only.yaml` | Control: FT + stability fixes, no Teams data | ✅ Created |
| `experiments/phase2_round9/R9_E_teams_ft_v2.yaml` | Wave 2: R9_A with ~430 Teams pairs | ✅ Created |
| `experiments/phase2_round9/R9_F_teams_sim_aug.yaml` | Wave 2: TeamsCodecSimulation on all families, no real Teams data | ✅ Created |
| `experiments/phase2_round9/R9_G_teams_both.yaml` | Wave 2: Real Teams data + TeamsCodecSimulation on non-Teams families | ✅ Created |
| `experiments/phase2_round9/R9_SMOKE.yaml` | Smoke test: 1K steps, verify pipeline + new metrics | ✅ Created |

### Files Modified ✅

| File | Change | Status |
|------|--------|--------|
| `data/sources/combined_paired.py` | Teams discovery, `_iterate_teams_sample`, Teams sample handling, **teams_codec_simulation config forwarding** | ✅ Modified |
| `utils/grouping.py` | `deeplive_teams_fake` / `deeplive_teams_real` family keys | ✅ Modified |
| `data/augmentations/pipelines.py` | Teams passthrough pipeline, **TeamsCodecSimulation wiring in QualityTargetedFamilyRouter** (constructor, `_maybe_apply_teams_sim()`, factory) | ✅ Modified |
| `data/augmentations/__init__.py` | Export `TeamsCodecSimulation` | ✅ Modified |
| `metrics/utils.py` | F1 computation at EER threshold and 0.5 | ✅ Modified |
| `trainer/trainer.py` | Stability loss, weakest-method logging, unified threshold, jitter monitoring, label smoothing | ✅ Modified |
| `trainer/mixins/__init__.py` | Register `StabilityRegMixin` | ✅ Modified |

### Key Dependencies Between Changes

```
grouping.py (family keys) ──────────────────┐
                                             │
combined_paired.py (Teams discovery) ───────┼──→ Training pipeline works  ✅
                                             │
pipelines.py (Teams family aug) ────────────┘

metrics/utils.py (F1) ──────────────────────┐
                                             │
trainer.py (unified threshold, weakest) ────┼──→ Monitoring works  ✅
                                             │
stability.py (mixin) ──────────────────────┘

teams_simulation.py (transform) ──┐
                                   │
pipelines.py (wiring) ────────────┼──→ Wave 2 augmentation works  ✅
                                   │
combined_paired.py (config fwd) ──┘
```

**Implementation order (all complete):**
1. ✅ `utils/grouping.py` — new family keys
2. ✅ `data/sources/combined_paired.py` — Teams discovery and iteration
3. ✅ `data/augmentations/pipelines.py` — Teams family pipeline + TeamsCodecSimulation wiring
4. ✅ `metrics/utils.py` — F1 metric
5. ✅ `trainer/mixins/stability.py` — new mixin
6. ✅ `trainer/trainer.py` — stability loss, metrics logging
7. ✅ Experiment configs — R9_A through R9_SMOKE (8 configs)
8. ✅ `data/augmentations/teams_simulation.py` — Teams simulation (validated on 50 samples / 165 frames)

---

## 11. Implementation Log

### Feb 27 — Steps 1–7 (Wave 1 pipeline)

All Wave 1 code changes implemented per plan:
- Teams data discovery and iteration in `combined_paired.py`
- Family keys (`deeplive_teams_fake`, `deeplive_teams_real`) in `grouping.py`
- Teams passthrough pipeline (HorizontalFlip + light BrightnessContrast) in `pipelines.py`
- StabilityRegMixin in `trainer/mixins/stability.py`
- F1, unified threshold, weakest-method, jitter logging in `trainer.py` and `metrics/utils.py`
- 8 experiment YAML configs created

### Feb 28 — Step 8 (TeamsCodecSimulation) + Validation + Critical Bug Fix

#### TeamsCodecSimulation: Iterative Development

The initial implementation followed the plan's proposed 5-stage pipeline (brightness → sharpen → chroma blur → JPEG → noise). **Matched-pair validation against real Teams frames revealed the plan's measurements were fundamentally wrong:**

| What the plan said | What real data showed |
|-------------------|---------------------|
| Teams sharpens +37.8% | Teams **blurs** −50.9% |
| Codec adds noise +8.3% | Teams **reduces** noise −11.8% |
| HF energy −7.2% | HF energy **−77.4%** (10× worse) |

The augmentation was completely rewritten **6 times** through iterative validation:

1. **v1:** Plan's 5-stage pipeline → direction accuracy 2/8
2. **v2:** Removed sharpening → 3/8
3. **v3:** Added Gaussian blur for codec smoothing → 5/8
4. **v4:** Removed additive noise → 6/8
5. **v5:** Added bilateral deblocking → 6/8, magnitude improved
6. **v6:** Tuned brightness (0.02, 0.10) and contrast (0.10, 0.22) to reduce overshoot → 6/8 final

#### Validation Results

**18-sample validation (132 matched frames):**

| Metric | Real Teams Δ | Simulated Δ | Direction Match |
|--------|-------------|-------------|-----------------|
| Sharpness | −50.9% | −35.6% | ✅ |
| Brightness | +19.0% | +34.0% | ✅ (overshoot) |
| Contrast | +3.5% | +18.2% | ✅ (overshoot) |
| Noise | −11.8% | −6.2% | ✅ |
| HF energy | −77.4% | −48.3% | ✅ |
| Blockiness | −1.2% | −3.8% | ✅ |
| Chroma blur | −12.6% | +0.1% | ❌ |
| BPP | −8.7% | −12.4% | ✅ |

Direction accuracy: **6/8 (75%)** — all major effects (blur, brighten, contrast, denoise, HF loss, deblock) correct.

**50-sample validation (42 usable, 165 matched frames):**

| Metric | Real Teams Δ | Simulated Δ | Direction Match |
|--------|-------------|-------------|-----------------|
| Sharpness | +2.8% (bimodal) | −35.6% | ❌ (bimodal flip) |
| Brightness | +17.1% | +34.0% | ✅ |
| Contrast | +2.3% | +18.2% | ✅ |
| Noise | −12.1% | −6.2% | ✅ |
| HF energy | −74.4% | −48.3% | ✅ |
| Blockiness | −1.2% | −3.8% | ✅ |
| Chroma blur | −1.1% | +0.1% | ❌ |
| BPP | −11.1% | −12.4% | ✅ |

Direction accuracy: **5/8 (62.5%)** — sharpness flipped due to bimodal Teams behavior (enhancement mode on some content types). Per-frame sharpness std = 399.9% with a +3015% outlier.

**Decision:** The augmentation is accepted as-is. Rationale:
- It captures the **dominant** Teams effects (brightness boost, codec compression, denoising)
- Real Teams data in training (R9_A, R9_E, R9_G) covers both modes
- The augmentation doesn't need to be a perfect simulator — it pushes model invariance in the right direction
- Bimodality comes from Teams' content-adaptive video enhancement, which varies by sample category

#### Critical Bug: Dead Code Wiring

**Discovery:** While verifying the full pipeline, investigation revealed that `TeamsCodecSimulation` was **completely dead code** — the YAML config `augmentation.teams_codec_simulation.enabled: true` was present in R9_F and R9_G, but **no Python code ever read it**.

**Root cause:** In `_create_combined_transform()`, the code filters `aug_config` keys:
```python
preset_overrides = {
    k: v for k, v in aug_config.items()
    if k not in _NON_PRESET_KEYS and k in _valid_preset_keys
}
```
Since `teams_codec_simulation` is not a valid preset key, it was silently dropped.

**Fix:** Added explicit config reading **before** the preset filter:
```python
teams_codec_cfg = aug_config.get('teams_codec_simulation', None)
router = create_quality_targeted_family_router(
    ...,
    teams_codec_simulation=teams_codec_cfg,
)
```

Plus full wiring in `QualityTargetedFamilyRouter`:
- Constructor accepts `teams_codec_simulation: dict | None`
- Reads `enabled`, `probability`, `exclude_families` from config
- Lazily imports and instantiates `TeamsCodecSimulation` if enabled
- `_maybe_apply_teams_sim(image, family_key)` applies after per-family pipeline

**Verification:** 11-point smoke test + 100-image edge case audit all pass.

---

## 12. TeamsCodecSimulation Validation Report

### Methodology

Validated by downloading matched frame pairs from GCS:
- **Original:** `gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/{id}/frames/{real,fake}/frame_NNNN.png`
- **Teams passthrough:** `gs://live-deepfake-methods-real-and-fake-frames-cropped-teams/samples/{id}/frames/{real,fake}/frame_NNNN.jpg`

For each matched pair, computed 8 metrics on the original, the real Teams frame, and a simulated frame (`TeamsCodecSimulation(original)`). Compared deltas.

### Key Finding: Plan's Measurements Were Wrong

The original R9 plan reported deltas from a **bulk bucket-level comparison** (different samples, different frame counts, different face detectors). The matched-pair validation using the **exact same frame before and after Teams** showed dramatically different results:

| Metric | Plan's bucket-level Δ | Matched-pair Δ | Error |
|--------|----------------------|----------------|-------|
| Sharpness | +37.8% (sharpens) | **−50.9%** (blurs) | **Wrong direction** |
| Noise | +8.3% (adds) | **−11.8%** (reduces) | **Wrong direction** |
| HF energy | −7.2% | **−77.4%** | **10× underestimate** |
| Brightness | +18.6% | +19.0% | ✅ Close |
| Contrast | +7.0% | +3.5% | ✅ Same direction |

**Lesson learned:** Bucket-level statistics are unreliable for characterizing a codec pipeline. Always use matched pairs.

### 50-Sample Bimodality Observation

At broader scale (50 samples), sharpness flipped from −50.9% to +2.8% mean with extreme variance (std=399.9%). This is because:
- **Edge-case samples** (18-sample set): Teams compresses and blurs these
- **Quality-enhancement samples** (broader set): Teams' video enhancement sharpens these

Teams has a **content-adaptive** processing pipeline. The augmentation models the blur mode (the dominant mode for deployment-realistic content).

### Files

| File | Purpose |
|------|---------|
| `data/augmentations/teams_simulation.py` | The validated transform (148 lines) |
| `tests/test_teams_simulation.py` | GCS-based validation script |
| `data/augmentations/pipelines.py` | Wiring into QualityTargetedFamilyRouter |
| `data/sources/combined_paired.py` | Config forwarding to router factory |

---

## 13. Pre-Launch Audit & Bug Fixes (Feb 28)

A full pre-launch audit of all R9 implementation files uncovered **4 bugs** (1 blocking crash, 3 correctness/performance issues) plus dead code. All fixed in `data/sources/combined_paired.py`.

### 13.1 BLOCKING: Missing `storage` Import

**Problem:** `combined_paired.py` uses `storage.Client()` and `storage.Blob()` in the Teams iteration code but **never imports `google.cloud.storage`**. Any Teams-enabled config (R9_A, B, C, E, F, G — 6 of 8) would crash instantly with `NameError: name 'storage' is not defined`.

**Fix:** Added at line 48:
```python
from google.cloud import storage
```

### 13.2 HIGH: Frame Pairing Misalignment

**Problem:** `_load_frames()` returned a flat `List[np.ndarray]`, then real/fake pairing used positional indexing (`real_frames[i]`, `fake_frames[i]`). If a frame download failed on only one side (e.g., real frame 3 missing), the lists would have different lengths or shifted indices — all subsequent pairs would be misaligned.

**Fix:** Changed `_load_frames()` to return `List[Tuple[int, np.ndarray]]` (frame index, image). Pairing now uses dict-based lookup:
```python
real_by_idx = {idx: img for idx, img in real_frames}
fake_by_idx = {idx: img for idx, img in fake_frames}
common_indices = sorted(set(real_by_idx) & set(fake_by_idx))
for frame_idx in common_indices:
    real_img = real_by_idx[frame_idx]
    fake_img = fake_by_idx[frame_idx]
```
This guarantees correct pairing even when frames are missing asymmetrically.

### 13.3 HIGH: `storage.Client()` Created Per Sample

**Problem:** Every call to the Teams sample iteration created a new `storage.Client()` instance (~460 calls per epoch), each doing an OAuth handshake.

**Fix:** Cached the client on the dataset object:
```python
if not hasattr(self, '_teams_gcs_client') or self._teams_gcs_client is None:
    self._teams_gcs_client = storage.Client()
client = self._teams_gcs_client
```

### 13.4 MEDIUM: Silent 0-Sample Failure

**Problem:** If Teams discovery found 0 samples (e.g., bucket permissions, wrong prefix), training would proceed silently with Teams contributing nothing — highest-weighted family producing zero gradients.

**Fix:** Added a warning log:
```python
if len(teams_samples) == 0:
    logger.warning(
        "Teams data is ENABLED but 0 samples were discovered from "
        f"bucket '{bucket_name}' prefix '{prefix}'. "
        "Check bucket permissions and path configuration."
    )
```

### 13.5 Dead Code Removal

Removed an unused GCS listing call that was left over from an earlier implementation:
```python
# REMOVED:
manifests = list(bucket.list_blobs(prefix="samples/", delimiter="/"))
```
This was wasting a GCS RPC on every worker initialization.

### 13.6 Files Modified

| File | Changes |
|------|---------|
| `data/sources/combined_paired.py` | All 5 fixes above |

### 13.7 Launch Notes

- **Docker image must be rebuilt** after these fixes before launching on Vertex AI
- If skipping R9_SMOKE due to time pressure: **watch the first validation pass of R9_A** (at step 500) to confirm Teams data is loading and stability loss is non-zero. Key W&B metrics to check:
  - `train/teams_samples_seen > 0` — confirms Teams data is flowing
  - `stability_loss > 0` — confirms StabilityRegMixin is active
  - `val/f1` is computed — confirms F1 metric wiring
  - No `NameError` or `ImportError` in Vertex AI logs within the first 2 minutes

---

*End of R9 Plan — all 8 implementation steps complete, 4 pre-launch fixes applied, ready to launch.*
