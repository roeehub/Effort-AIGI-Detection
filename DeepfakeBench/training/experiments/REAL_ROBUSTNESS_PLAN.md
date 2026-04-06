# Real Data Robustness Improvement Plan

**Date**: February 18, 2026  
**Status**: Ready for implementation  
**Scope**: Extend training pipeline to eliminate false positives on out-of-distribution (OOD) real images without degrading fake detection capability

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Discovery and Root Cause](#2-problem-discovery-and-root-cause)
3. [Quantitative Evidence](#3-quantitative-evidence)
4. [What We Already Tried](#4-what-we-already-tried)
5. [Intervention Strategy](#5-intervention-strategy)
6. [Implementation Plan — Part A: Unpaired Real Support](#6-implementation-plan--part-a-unpaired-real-support)
7. [Implementation Plan — Part B: Augmentation Tuning](#7-implementation-plan--part-b-augmentation-tuning)
8. [Implementation Plan — Part C: Gradient Reversal (Quality-Domain Adversarial Head)](#8-implementation-plan--part-c-gradient-reversal-quality-domain-adversarial-head)
9. [Experiment Configs](#9-experiment-configs)
10. [Validation and Success Criteria](#10-validation-and-success-criteria)
11. [File Reference](#11-file-reference)
12. [Appendix: Key Data Tables](#appendix-key-data-tables)

---

## 1. Executive Summary

The Effort detector (SVD residual fine-tuning on CLIP ViT-B-16) achieves excellent in-distribution performance (AUC >0.99, enhanced fake TPR >98%) but fails catastrophically on **Zoom VCD real images** — classifying ~40-49% of genuine video-call face crops as fake. The root cause is a **quality-property shortcut**: DF40 training reals are unusually soft and smooth compared to real-world webcam captures, and the model has learned "soft smooth face = real." Any real image with higher sharpness, more texture, or different frequency characteristics gets pushed toward the fake decision boundary.

This plan implements three coordinated interventions:
1. **Add ~800 VCD real images directly to training** (identity-split, 20% of VCD identities) as unpaired real samples in the existing `CombinedPairedIterableDataset`
2. **Tune augmentation presets** to broaden the training-real quality distribution (sharpen reals, degrade fakes) — breaking the quality shortcut from both directions
3. **Add a gradient reversal quality-domain head** that forces the SVD residual subspace to be uninformative about image quality, making the model provably quality-invariant

These are ordered by implementation effort and can be deployed incrementally.

---

## 2. Problem Discovery and Root Cause

### 2.1 The Failure

R4's best model (FT7) achieves:
- DF40 in-distribution fake TPR: **86.85%**
- Enhanced DeepLive fake TPR: **99.53%**
- WMA flat per-image detection: **88.35%**
- External real FPR: **4.38%** (YouTube AVSpeech)

But on **Zoom VCD real webcam captures**: **~51-60% classified as fake** (near random chance). These are genuine face crops from Zoom video calls — the exact deployment domain for the detector.

### 2.2 Phase 1 Quality Fingerprint Analysis

We ran a comprehensive quality fingerprint analysis (`analyze_real_quality_distributions.py`) comparing 300 images from each of 4 real sources:
- **DF40 paired reals** (training distribution)
- **Webcam test captures** (`real_or_virtual` bucket)
- **Zoom VCD reals** (OOD — failing)
- **YouTube AVSpeech** (OOD — passing)

Cached samples are at `weights/quality_analysis_samples/{df40_real,real_or_virtual,vcd_real,youtube_real}/` (300 images each).

### 2.3 The Root Cause: DF40 Training Reals Are the Outlier

The critical finding (**the single most important insight**) is that DF40 training reals are not representative of real-world faces. They are the outlier:

| Metric | DF40 (train) | VCD (OOD, failing) | YouTube (OOD, passing) | Webcam Tests |
|---|---:|---:|---:|---:|
| sharpness_laplacian_var | **38.9** | 295.3 | 430.3 | 251.2 |
| sharpness_tenengrad | **32.8** | 40.9 | 54.5 | 45.4 |
| edge_density | **0.022** | 0.044 | 0.070 | 0.050 |
| freq_high_ratio | **0.145** | 0.249 | 0.166 | 0.228 |
| psd_slope | **-2.071** | -1.665 | -1.942 | -1.742 |
| effective_resolution | **0.210** | 0.405 | 0.253 | 0.378 |
| texture_local_var_mean | **3.6** | 23.3 | 36.9 | 20.7 |
| hf_noise_std | **1.97** | 4.49 | 5.57 | 4.58 |
| noise_estimate | **0.38** | 1.08 | 1.19 | 1.00 |

DF40 reals are by far the softest (sharpness 38.9 vs 295+), smoothest (texture 3.6 vs 23+), and least noisy (HF noise 1.97 vs 4.49+). The model has learned: "soft, smooth, low-texture face = real." VCD reals are sharper, noisier, and have more high-frequency content — properties that overlap with fake artifacts.

### 2.4 VCD vs YouTube Divergence

The file `analysis_results/vcd_quality_divergence_summary.txt` shows which metrics diverge most between VCD and YouTube (the passing OOD source):

| Metric | Cohen's d | Severity |
|---|---:|---|
| effective_resolution_90pct | 1.831 | LARGE |
| freq_high_ratio | 1.812 | LARGE |
| psd_slope | 1.713 | LARGE |
| freq_high_to_low | 1.445 | LARGE |
| freq_low_ratio | -1.174 | LARGE |
| edge_density | -0.676 | MEDIUM |
| sharpness_tenengrad | -0.621 | MEDIUM |

VCD's unique signature: **much higher effective resolution** (0.41 vs 0.25) and **much higher HF ratio** (0.25 vs 0.17) with a **flatter PSD slope** (-1.67 vs -1.94). This is the characteristic "webcam codec fingerprint" — Zoom's encoder preserves more high-frequency detail than YouTube's heavier compression, but introduces codec noise (ringing, mosquito noise) that elevates the HF energy ratio.

### 2.5 Why the Model Fails

The Effort model trains SVDResidualLinear layers with only **k=32 trainable singular directions** per linear layer (out of 512-768 total). This is an extremely constrained subspace. With training reals occupying a very narrow quality band (soft, smooth, low-noise) and fakes showing diverse quality characteristics, the residual subspace has encoded quality as a proxy for real/fake classification. When VCD reals arrive with quality properties outside the training-real band, they fall into the "fake" zone.

This is a **shortcut learning** problem, not a model capacity problem. The model has enough capacity to detect actual forgery artifacts — it just also learned to use quality as a shortcut because the training data made quality and label correlated.

---

## 3. Quantitative Evidence

### 3.1 Full Quality Metric Table

Source: `analysis_results/real_quality_summary_table.txt`

This file contains the complete mean ± std for all 37 quality metrics across all 4 real sources (DF40, Webcam Tests, Zoom VCD, YouTube). The raw per-image data is in `analysis_results/real_quality_fingerprints.csv`.

### 3.2 Codec Simulation Gap Analysis

We implemented a `VideoCodecSimulation` transform (5-step chain: downscale → bilateral filter → block quantization → frequency-shaped noise → JPEG compression) and ran a visual demo on 50 real cached images.

**Key finding: Codec simulation closes only 25-40% of the gap and moves the WRONG direction on sharpness/edge metrics.**

| Metric | DF40 Original | DF40 + Codec Sim | VCD (target) | Gap Closed |
|---|---:|---:|---:|---:|
| psd_slope | -2.07 | ~-1.92 | -1.67 | ~37% |
| freq_high_ratio | 0.145 | ~0.17 | 0.249 | ~24% |
| sharpness_tenengrad | 32.8 | ~25 | 40.9 | **WRONG DIRECTION** |
| edge_density | 0.022 | ~0.015 | 0.044 | **WRONG DIRECTION** |

The codec sim makes images blurrier and lower-quality (less sharp, lower edge density), but VCD reals are actually SHARPER than DF40 (tenengrad 40.9 vs 32.8). VCD = sharp + noisy (real webcam capture + codec ringing). Codec sim produces blurry + noisy — fundamentally wrong direction for the sharpness/edge dimension.

**Implication**: VideoCodecSimulation is useful for adding general quality diversity (keep at 10-15% probability) but cannot solve the VCD gap alone. Must not be cranked up to 40-50%.

Demo plots: `analysis_results/plots/codec_simulation_demo/` (5 PNGs: before/after grid, frequency shift distributions, distribution overlap, radar comparison, single-image deep dive with FFT)

---

## 4. What We Already Tried

### 4.1 R5 Experiment Matrix (Overnight, Feb 17-18)

Six parallel runs launched on Vertex AI:
- **R5_S1**: Baseline FT7-style data mix (no augmentation tuning)
- **R5_S2**: FT7 mix + weighted family resampling
- **R5_S3**: FT7 mix + weighted sampling + `quality_targeted_family` augmentation at "strong" strength

Each run with 2 seeds (737, 1337). All monitor VCD + WMA + YouTube every 2000 steps in-training.

### 4.2 Existing Augmentation Infrastructure

The `quality_targeted_family` pipeline (`data/augmentations/pipelines.py`) already provides:
- **Family-routed augmentation**: Different pipelines for `df40_fake`, `deeplive_non_enhanced_fake`, `deeplive_enhanced_fake`, `visomaster_fake`, `df40_real`, `realpool_real`, `external_real`
- **`QualityTargetedFamilyRouter`**: Callable that inspects `meta['label']`, `meta['source']`, `meta['method']` to route images to family-specific pipelines
- **Three preset strengths**: `light`, `moderate`, `strong` — controlling JPEG compression, blur, noise, downscale, codec simulation, sharpening
- **`VideoCodecSimulation`** integrated into all family pipelines at 10-22% probability

### 4.3 Existing OOD Monitoring

R5 configs include in-training OOD monitoring:
- 1200 VCD real images (`per_image` grouping, deterministic)
- 200 YouTube real videos (`by_folder` grouping, deterministic)
- 1202 WMA fake images (`per_image` grouping, deterministic)

Monitoring starts at step 1000, runs every 2000 steps. **This is read-only — does not affect gradients.**

### 4.4 What We Know Doesn't Work

1. **Heavy codec sim alone** — moves wrong direction on sharpness/edge. Closes <40% of the gap.
2. **Generic "more augmentation"** — R3 proved this counterproductive. Over-augmentation destroys training signal without targeting the actual quality shortcut.
3. **Using only augmentation on existing data** — fundamental limit: you can't augment DF40 reals to be SHARPER AND NOISIER (VCD's profile) without also making them look fake-like. The quality distributions need actual target-domain data.

---

## 5. Intervention Strategy

### 5.1 Three-Pronged Approach

| Intervention | Impact | Effort | Mechanism |
|---|---|---|---|
| **A. Add VCD reals to training** | Direct | Moderate | Expands real class support to cover VCD quality domain |
| **B. Tune augmentation bidirectionally** | Indirect | Low | Sharpen reals + degrade fakes → breaks quality-label correlation from both sides |
| **C. Gradient reversal quality head** | Principled | Moderate | Forces SVD subspace to be quality-invariant via adversarial training |

**These are complementary, not alternatives.** A provides the data, B prevents new shortcuts, C provides the theoretical guarantee.

### 5.2 Why Each Is Necessary

**A alone is insufficient**: Adding VCD reals teaches the model "webcam-quality faces can be real" but also risks a new shortcut "webcam quality = always real." WMA fakes have similar quality properties (they were also recorded via webcam/video-call pipeline). Without also ensuring the model sees degraded fakes, you trade false positives for false negatives.

**B alone is insufficient**: Augmentation can broaden distributions but can't perfectly replicate VCD's characteristic codec fingerprint. Also, with k=32, there's a limit to how much quality diversity the residual subspace can absorb without losing fake-detection signal.

**C alone is insufficient**: Gradient reversal needs quality-domain examples in the batch to learn what "quality-invariant" means. Without VCD reals in training, the quality head only sees the narrow DF40 quality range and can't learn to be invariant across the full deployment quality spectrum.

### 5.3 Critical Constraint: Protect Fake Detection

With only k=32 trainable singular directions per layer, the SVD residual subspace is tiny. Every gradient update is precious. Key guardrails:

1. **Cap unpaired reals at 10-15% of total training samples** — prevents real-class gradients from overwhelming fake-detection learning
2. **Use identity-weighted sampling** with modest weight for external reals (0.5-0.8 vs 1.0 for training sources)
3. **Augment both directions** — sharpen reals AND degrade fakes
4. **Monitor regression continuously** — R5's OOD loop checks WMA fake detection + VCD real FPR + in-dist AUC every 2000 steps

---

## 6. Implementation Plan — Part A: Unpaired Real Support

### 6.1 Overview

Extend `CombinedPairedIterableDataset` to handle real-only samples from external GCS sources, integrated via the existing identity-balanced sampling infrastructure.

### 6.2 Data Source

- **GCS bucket**: `effort-collected-data`
- **Prefix**: `real/VCD`
- **Total images**: ~1200 across ~136 unique identities
- **Identity encoding**: MD5 hash in filename, format: `real__VCD__<md5>_<WxH>_<fps>__<frame>.png`
- **Identity split**: Use first 20% of sorted MD5 hashes for training (~27 identities, ~200-300 images), rest for OOD eval
- **Currently**: Weight 0.0 (OOD monitoring only), no existing train/test split

### 6.3 Step-by-Step Implementation

#### Step A1: Add `UnifiedUnpairedRealSample` dataclass

**File**: `data/sources/combined_paired.py` (near line 85, after `UnifiedPairedSample`)

```python
@dataclass
class UnifiedUnpairedRealSample:
    """
    Wrapper for unpaired real-only samples from external sources.
    
    Unlike UnifiedPairedSample, these have no fake counterpart.
    Each sample represents a single identity with one or more real frames
    from an external GCS source (e.g., VCD webcam captures).
    """
    identity: str           # Unique identity (e.g., "external_vcd_<md5>")
    source: str             # 'external_vcd_real', 'external_webcam_real', etc.
    method: str             # For family routing (must contain "external" for grouping)
    gcs_bucket: str         # e.g., "effort-collected-data"
    frame_paths: List[str]  # GCS paths to individual frame PNGs
    sample_id: str          # Unique sample ID
    has_landmarks: bool = False
```

**Key design decisions**:
- `source` and `method` must contain `"external"` to trigger `external_real` routing through `infer_group_key` (see grouping routing below)
- `frame_paths` holds pre-resolved GCS paths (no lazy discovery at iteration time)
- Reuses existing `_apply_transform` with `meta={'label': 0, 'source': source, 'method': method}`

#### Step A2: Fix family routing for VCD reals

**File**: `utils/grouping.py` (line ~115)

**Current issue**: The existing `infer_group_key` function routes to `external_real` when:
```python
source_norm == "external"
or method_norm.startswith("external_")
or "external" in method_norm
```

The OOD monitoring config uses `method: "zoom_vcd_real"` which does NOT match any of these conditions. For training, we need to ensure VCD reals route to `external_real`. 

**Solution**: Use method names containing "external" for training samples. For example:
- `method="external_vcd_real"` → matches `method_norm.startswith("external_")` → routes to `external_real` ✓
- `source="external"` → also matches ✓

Either approach works. Recommend using `method="external_vcd_real"` and `source="external"` to be explicit.

**No code change needed in `grouping.py`** — just use the right method/source strings in the sample dataclass.

#### Step A3: Add `_discover_external_training_reals()` function

**File**: `data/sources/combined_paired.py` (near line 637, before `_build_external_ood_videos()`)

This function reads the `combined_paired.external_training_reals` config section, discovers images from GCS, groups them by identity, performs the identity split, and returns `List[UnifiedUnpairedRealSample]`.

```python
def _discover_external_training_reals(
    combined_config: Dict[str, Any],
    logger: logging.Logger,
) -> List[UnifiedUnpairedRealSample]:
    """
    Discover and group external real images for training.
    
    Reads external_training_reals config, lists GCS objects, groups by identity
    (MD5 hash extracted from filename), selects the training split of identities,
    and returns UnifiedUnpairedRealSample objects.
    """
```

**Implementation details**:
1. Read `combined_config.get("external_training_reals", [])` — returns list of source configs
2. For each source config, list GCS objects using `fsspec` (same pattern as `load_external_real_videos` in `data/validation_sources.py`)
3. Extract identity from filename: parse `real__VCD__<md5>_<WxH>_<fps>__<frame>.png` → identity = MD5 hash
4. Group frames by identity
5. Apply identity split: sort identities deterministically, take first `identity_train_fraction` (default 0.2) for training
6. For each training identity, create a `UnifiedUnpairedRealSample` with all its frame paths
7. Apply optional `max_samples` cap
8. Log: number of total identities, training identities, total frames, frames per identity distribution

**Config schema**:
```yaml
combined_paired:
  external_training_reals:
    - bucket: "effort-collected-data"
      prefix: "real/VCD"
      method: "external_vcd_real"        # MUST contain "external" for routing
      grouping: "per_image"              # Each image is a separate frame
      identity_pattern: "real__VCD__(?P<md5>[a-f0-9]{32})_"  # Regex to extract identity
      identity_train_fraction: 0.20      # 20% of identities for training
      identity_split_seed: 737           # Deterministic split
      max_frames_per_identity: 10        # Cap to avoid identity imbalance
      max_total_samples: 800             # Hard cap on total frames
      deterministic: true
```

#### Step A4: Integrate unpaired samples into identity-balanced sampling

**File**: `data/sources/combined_paired.py`

The key insight is that `CombinedPairedIterableDataset` currently only handles `UnifiedPairedSample` objects. We need to make it also handle `UnifiedUnpairedRealSample`.

**Option A (Recommended)**: Make `UnifiedUnpairedRealSample` a thin wrapper that is duck-type compatible with `UnifiedPairedSample`. Add an `is_unpaired_real` flag:

```python
@dataclass
class UnifiedUnpairedRealSample:
    identity: str
    source: str
    method: str
    gcs_bucket: str
    frame_paths: List[str]
    sample_id: str
    has_landmarks: bool = False
    is_unpaired_real: bool = True  # Distinguishes from paired samples
    original_sample: Any = None    # Compat: not used, but present for duck typing
```

The dataset's `__init__` already groups samples by identity via `self._samples_by_identity`. Unpaired real samples get unique identities (e.g., `"external_vcd_a1b2c3d4..."`) that don't collide with DF40/DeepLive identities. They participate in identity-balanced sampling like any other sample.

**Family routing**: `_sample_family_for_sampling()` currently prioritizes fake-family routing (since paired samples yield both real and fake). For unpaired reals, there is no fake side. The function should handle this:

```python
def _sample_family_for_sampling(sample, enhanced_strategy_names):
    # For unpaired reals, route directly to real family
    if getattr(sample, 'is_unpaired_real', False):
        return infer_family_key(0, method=sample.method, source=sample.source,
                                enhanced_strategy_names=enhanced_strategy_names)
    # ... existing paired logic
```

**Weight control**: The existing `identity_family_weights` config controls how often each family appears:
```yaml
sampling:
  strategy: "identity_resample_weighted"
  family_weights:
    external_real: 0.6    # Modest weight — don't overwhelm fakes
    df40_fake: 0.9
    # ... rest unchanged
```

#### Step A5: Add `_iterate_unpaired_real_sample()` method

**File**: `data/sources/combined_paired.py` (next to `_iterate_visomaster_sample`, line ~1293)

```python
def _iterate_unpaired_real_sample(
    self,
    unified_sample: UnifiedUnpairedRealSample,
    rng: random.Random,
) -> Iterator[Dict[str, Any]]:
    """Load and yield frames from an unpaired real-only sample."""
    from data.gcs_utils import download_frame_from_gcs  # Or equivalent
    
    frame_paths = unified_sample.frame_paths
    # Sample up to frames_per_sample frames if more are available
    n = min(len(frame_paths), self.config.frames_per_sample)
    selected = rng.sample(frame_paths, n) if len(frame_paths) > n else frame_paths
    
    for i, gcs_path in enumerate(selected):
        img = download_and_decode_frame(unified_sample.gcs_bucket, gcs_path)
        if img is None:
            continue
        
        if self.transform:
            img = self._apply_transform(
                img, None,
                {'label': 0, 'source': unified_sample.source,
                 'method': unified_sample.method},
            )
        
        yield {
            'image': img,
            'label': 0,
            'identity': unified_sample.identity,
            'source': unified_sample.source,
            'method': unified_sample.method,
            'method_id': self.method_mapping.get(unified_sample.method, -1),
            'sample_id': unified_sample.sample_id,
            'frame_idx': i,
        }
```

**Frame loading**: Reuse the same GCS download pattern used by `_iterate_visomaster_sample` (which calls `load_visomaster_frames` from `data/sources/visomaster.py`). The VCD images are individual PNGs, not video folder sequences, so loading is simpler.

#### Step A6: Route in `__iter__`

**File**: `data/sources/combined_paired.py` (line ~1098-1110)

Current dispatch:
```python
for unified_sample in samples_to_iterate:
    try:
        if unified_sample.source == 'df40':
            yield from self._iterate_df40_sample(unified_sample, rng)
        elif unified_sample.source == 'visomaster':
            yield from self._iterate_visomaster_sample(unified_sample, rng)
        else:  # deeplive
            yield from self._iterate_deeplive_sample(unified_sample, rng)
```

**Add**:
```python
        elif getattr(unified_sample, 'is_unpaired_real', False):
            yield from self._iterate_unpaired_real_sample(unified_sample, rng)
```

Place this BEFORE the existing `else` clause.

#### Step A7: Wire into `create_combined_paired_pipeline`

**File**: `data/sources/combined_paired.py` (line ~1800, in the pipeline assembly section)

After loading DF40 + DeepLive + VisoMaster samples and before identity-stratified splitting:

```python
# Discover external training reals (VCD, webcam, etc.)
external_reals = _discover_external_training_reals(combined_config, logger)
if external_reals:
    logger.info(f"Adding {len(external_reals)} unpaired real samples to training pool")
    all_samples.extend(external_reals)
```

Since unpaired reals have unique identity prefixes (`external_vcd_...`), they won't collide with existing identities and will be split correctly by `split_samples_by_identity()`.

#### Step A8: Adjust total frame count estimation

**File**: `trainer/trainer.py` (or wherever `total_frames` is computed from `num_samples_per_epoch`)

Currently the training loop estimates: `total_frames = num_samples_per_epoch * frames_per_sample * 2` (the `* 2` assumes every sample yields both real and fake frames). With unpaired reals, some samples yield `frames_per_sample * 1`. The simplest fix is to count paired and unpaired separately:

```python
n_paired = sum(1 for s in train_samples if not getattr(s, 'is_unpaired_real', False))
n_unpaired = sum(1 for s in train_samples if getattr(s, 'is_unpaired_real', False))
total_frames = (n_paired * frames_per_sample * 2) + (n_unpaired * frames_per_sample)
```

#### Step A9: Collate function — no changes needed

The `combined_paired_collate_fn` (line ~1360) groups frames by `(sample_id, label)`. Unpaired reals produce real-only groups. The collater handles this fine — it creates fewer video groups per batch when some are real-only. The batch might be 55/45 real/fake instead of 50/50, which is acceptable and controlled by family weights.

#### Step A10: Identity split for OOD eval

Currently the OOD monitoring loads ALL VCD images (1200). Once some VCD identities are in training, the OOD monitor must exclude them. Two approaches:

**Option A**: Create separate OOD monitoring configs that use the complementary identity set (training gets 20% of identities, OOD gets the other 80%).

**Option B** (simpler): Pass `exclude_identities` to `_build_external_ood_videos()` so the OOD loader skips any identity present in the training set.

Recommend **Option B** — add an `exclude_identities` parameter to the OOD video builder and pass in the training VCD identities.

---

## 7. Implementation Plan — Part B: Augmentation Tuning

### 7.1 Overview

Tune the `quality_targeted_family` pipeline presets to break the quality-label correlation from both sides: **sharpen/add-texture to reals** (moving them toward VCD's profile) and **degrade fakes** (so low-quality fakes are also seen during training).

### 7.2 What NOT to Do

**Do NOT increase `VideoCodecSimulation` probability beyond 15%.** Our analysis showed codec sim makes images blurrier (wrong direction for VCD, which is sharper than DF40). Keep it at current levels for general quality diversity.

### 7.3 Specific Changes

#### Change B1: Increase sharpening on real pipelines

**File**: `data/augmentations/pipelines.py`, `_build_family_quality_pipeline()`

**Current** (line ~997, `df40_real`):
```python
A.IAASharpen(alpha=p["sharpen_alpha_real"], lightness=(0.7, 1.0), p=0.50),
```

**Target**: Increase probability and alpha range to push DF40 reals toward VCD's sharpness profile:
```python
A.IAASharpen(alpha=(0.30, 0.70), lightness=(0.7, 1.0), p=0.60),
```

The VCD tenengrad (40.9) is only ~25% above DF40 (32.8), but the laplacian variance (295 vs 38.9) is 7.6x higher, indicating much more fine detail/texture. Sharpening is the right tool.

**Same change for `realpool_real` and `external_real`** pipelines (line ~1010):
```python
A.IAASharpen(alpha=(0.15, 0.35), lightness=(0.7, 1.0), p=0.18),
```
→ Increase to `alpha=(0.20, 0.45), p=0.30`.

#### Change B2: Add GaussNoise to real pipelines

**Rationale**: VCD reals have 2.3x higher HF noise std (4.49 vs 1.97) and 2.8x higher noise estimate (1.08 vs 0.38) than DF40. Adding noise to training reals helps the model learn that noise ≠ fake.

**File**: `data/augmentations/pipelines.py`, `_build_family_quality_pipeline()`

For `df40_real`, add a noise step to the OneOf degradation block:
```python
A.OneOf(
    [
        A.ImageCompression(quality_lower=max(58, p["jpeg_lower"]), quality_upper=95, p=1.0),
        A.GaussNoise(var_limit=(3.0, max(12.0, p["noise_var"][1] * 0.6)), p=1.0),
        A.GaussNoise(var_limit=(8.0, 25.0), p=1.0),  # NEW: dedicated noise-only option
    ],
    p=0.30,  # Increased from 0.20
),
```

#### Change B3: Add quality degradation to fake pipelines

**Rationale**: If we're sharpening reals and adding noise, we must also show the model low-quality fakes. Otherwise the model learns a new shortcut: "sharp+noisy = real." WMA fakes already have webcam-like quality but other fakes don't.

For `df40_fake`, add a downscale+blur branch:
```python
# In the OneOf balanced_degrade block, add:
A.Downscale(
    scale_min=0.45,
    scale_max=0.65,
    interpolation=cv2.INTER_AREA,
    p=1.0,
),  # Already exists, but lower the scale range
```

Also increase the overall degradation probability for `deeplive_enhanced_fake` (which represents GFPGAN-enhanced fakes — these are unrealistically sharp and should be degraded more).

#### Change B4: Create "vcd_targeted" preset

Add a new preset to `_QUALITY_TARGETED_PRESETS` that encodes the above changes:

```python
"vcd_targeted": {
    # Same base as "strong" but with adjusted real-side augmentation
    "jpeg_lower": 40,
    "jpeg_upper": 90,
    "blur_limit": (3, 9),
    "noise_var": (10.0, 45.0),
    "downscale_min": 0.50,
    "downscale_max": 0.80,
    "quality_p": 0.60,
    "color_p": 0.52,
    "color_brightness": 0.22,
    "color_contrast": 0.22,
    "hue_shift": 12,
    "sat_shift": 24,
    "val_shift": 24,
    # Codec sim stays modest — wrong direction for VCD sharpness gap
    "webcam_codec_p": 0.12,
    "webcam_codec_quality": (35, 80),
    # Increased sharpening for reals
    "sharpen_alpha_balanced": (0.24, 0.60),
    "sharpen_alpha_real": (0.30, 0.70),
    # Real-side noise injection (new parameter)
    "real_noise_p": 0.25,
    "real_noise_var": (5.0, 20.0),
},
```

Then modify `_build_family_quality_pipeline` to use `real_noise_p` and `real_noise_var` when building real-family pipelines.

---

## 8. Implementation Plan — Part C: Gradient Reversal (Quality-Domain Adversarial Head)

### 8.1 Overview

Add a small quality-domain classifier head after the CLIP backbone with a **gradient reversal layer** (GRL). During forward pass, this head predicts which quality domain an image comes from. During backward pass, the GRL reverses the gradient sign, so the backbone learns to produce features that are **uninformative about quality domain** while remaining useful for real/fake classification.

This is the principled solution — it directly forces the SVD residual subspace to not encode quality.

### 8.2 Theory

Based on Ganin & Lempitsky (2015), "Domain-Adversarial Training of Neural Networks" (DANN). The key idea:

```
                    Features
CLIP Backbone → ─────┬────────→ Classification Head (real/fake)
                      │              ↑ normal gradient
                      └────────→ [GRL] → Quality Head (domain label)
                                          ↑ reversed gradient
```

The GRL multiplies gradients by `-λ` during backprop, where `λ` is a scaling factor that increases over training (curriculum). This creates a minimax game: the quality head tries to predict quality domain, the backbone tries to fool it.

### 8.3 Quality Domain Labels

Define quality domains based on measured property clusters:

| Domain ID | Label | Source(s) | Characteristic |
|---|---|---|---|
| 0 | `clean_academic` | DF40 reals | Soft, smooth, low-noise |
| 1 | `webcam_codec` | VCD reals, external webcam | Sharp, noisy, codec artifacts |
| 2 | `social_media` | YouTube reals | Variable quality, heavier compression |
| 3 | `enhanced` | GFPGAN-enhanced fakes | Unnaturally sharp, low noise |

The domain label is derived from `meta['source']` and `meta['method']` — same routing as `infer_family_key`. The quality head receives the **same features** as the classification head but predicts domain instead of real/fake.

### 8.4 Implementation

#### Step C1: Gradient Reversal Layer

**File**: `detectors/effort_detector.py` (new class, add before `EffortDetector`)

```python
class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer — reverses gradient by factor lambda during backprop."""
    
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_val=1.0):
        super().__init__()
        self.lambda_val = lambda_val
    
    def set_lambda(self, val):
        self.lambda_val = val
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)
```

#### Step C2: Quality Domain Head

**File**: `detectors/effort_detector.py` (new class)

```python
class QualityDomainHead(nn.Module):
    """Small MLP that predicts quality domain from backbone features."""
    
    def __init__(self, in_features, num_domains=4, hidden_dim=128):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_domains),
        )
    
    def set_lambda(self, val):
        self.grl.set_lambda(val)
    
    def forward(self, features):
        reversed_features = self.grl(features)
        return self.classifier(reversed_features)
```

#### Step C3: Integrate into EffortDetector

**File**: `detectors/effort_detector.py`, `EffortDetector.__init__` and `forward`

In `__init__` (around line 201+):
```python
# Quality domain adversarial head (optional)
self.use_quality_head = config.get('use_quality_domain_head', False)
if self.use_quality_head:
    num_domains = config.get('quality_domain_count', 4)
    self.quality_head = QualityDomainHead(
        in_features=self.hidden_size,
        num_domains=num_domains,
        hidden_dim=config.get('quality_head_hidden_dim', 128),
    )
```

In `forward` (around line 940, after features are computed):
```python
# Quality domain prediction (if enabled)
if self.use_quality_head and not inference:
    pred_dict['quality_domain_logits'] = self.quality_head(features)
```

#### Step C4: Add quality domain loss to `get_losses`

**File**: `detectors/effort_detector.py`, `get_losses` (line 688)

```python
# Quality domain adversarial loss
if self.use_quality_head and 'quality_domain_logits' in pred_dict:
    domain_labels = data_dict.get('quality_domain')
    if domain_labels is not None:
        quality_loss = F.cross_entropy(
            pred_dict['quality_domain_logits'],
            domain_labels.long(),
            reduction=reduction,
        )
        quality_weight = self.config.get('quality_domain_loss_weight', 0.1)
        losses['quality_domain'] = quality_loss * quality_weight
        losses['overall'] = losses['overall'] + quality_loss * quality_weight
```

#### Step C5: Lambda scheduling

In the training loop (`trainer/trainer.py`), update the GRL's lambda value based on training progress:

```python
# Curriculum lambda for gradient reversal
if hasattr(model, 'quality_head') and model.use_quality_head:
    # Sigmoid schedule: starts at 0, reaches ~1.0 at 70% through training
    progress = current_step / total_steps
    lambda_val = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
    model.quality_head.set_lambda(lambda_val)
```

This follows the standard DANN schedule: the quality head starts with zero reversal (learns the domain task first), then gradually increases reversal strength.

#### Step C6: Assign domain labels in the data pipeline

The quality domain label must be added to each frame's dict in the iteration methods. In each `_iterate_*_sample` method, add:

```python
yield {
    'image': img,
    'label': label,
    # ... existing fields ...
    'quality_domain': DOMAIN_MAP.get(source, 0),  # 0=clean_academic, 1=webcam_codec, etc.
}
```

The `DOMAIN_MAP` can be defined as:
```python
QUALITY_DOMAIN_MAP = {
    'df40': 0,          # clean_academic
    'external': 1,      # webcam_codec (VCD, webcam reals)
    'deeplive': 2,      # studio_capture
    'visomaster': 2,    # studio_capture (same real source as deeplive)
    'youtube': 3,       # social_media
}
```

The collate function needs a minor update to stack `quality_domain` tensors alongside `image`, `label`, etc.

### 8.5 Config Additions

```yaml
# In experiment YAML
use_quality_domain_head: true
quality_domain_count: 4
quality_head_hidden_dim: 128
quality_domain_loss_weight: 0.1    # Start modest
quality_domain_lambda_schedule: "sigmoid"  # or "linear"
```

---

## 9. Experiment Configs

### 9.1 Experiment Matrix

Create configs in `experiments/phase2_round6/` (or phase2_round5 extension):

| Config | External Reals | Augmentation | Quality Head | Purpose |
|---|---|---|---|---|
| **R6_S1_vcd_reals_aug** | 800 VCD | `vcd_targeted` | No | Test data + augmentation |
| **R6_S2_aug_only** | None | `vcd_targeted` | No | Ablation: augmentation alone |
| **R6_S3_vcd_reals_aug_grl** | 800 VCD | `vcd_targeted` | Yes | Full stack |
| **R6_SMOKE_30MIN** | 200 VCD | `vcd_targeted` | Yes | Quick integration test |

### 9.2 Example Config: R6_S1

```yaml
name: "R6_S1_vcd_reals_aug"
description: "Add VCD reals to training + vcd-targeted augmentation"
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

# Training hyperparams (same as R5_S2 baseline)
learning_rate: 2.0e-4
weight_decay: 0.05
optimizer_eps: 1.0e-8
lambda_reg: 0.01
rank: 736
gradient_clip_val: 1.0
nEpochs: 30
total_training_steps: 65000
lr_scheduler: "cosine_with_warmup"
lr_scheduler_warmup_steps: 1000
dataloader_strategy: combined_paired
frames_per_batch: 32
frames_per_video: 8
load_base_checkpoint: false

combined_paired:
  identity_balanced_sampling: true
  split_seed: 737

  df40:
    enabled: true
    pair_json: "dataset/df40_pairs/df40-pair-matching.json"
    gcs_bucket: "df40-frames-recropped-rfa85"
    sampling_mode: "sparse"
    anchor_indices: [0, 4, 8, 12, 16, 20, 24, 28]

  deeplive:
    enabled: true
    gcs_bucket: "live-deepfake-methods-real-and-fake-frames-cropped"
    use_landmarks: false
    sampling_mode: "sparse"
    anchor_indices: [0, 2, 4, 6, 8, 10, 12, 14]
    include_strategies:
      - "edge_cases"
      - "minimal_processing"
      - "quality_enhancement"
      - "edge_cases_enhanced"
      - "minimal_processing_enhanced"

  visomaster:
    enabled: true
    gcs_bucket: "live-deepfake-methods-real-and-fake-frames-cropped"
    frames_bucket: "live-deepfake-methods-real-and-fake-frames"
    anchor_indices: [0, 2, 4, 6, 8, 10, 12, 14]
    swap_models:
      - "CSCS"
      - "GhostFace-v1"
      - "GhostFace-v2"
      - "InStyleSwapper256-A"
      - "InStyleSwapper256-B"
      - "Inswapper128"

  # === NEW: External training reals ===
  external_training_reals:
    - bucket: "effort-collected-data"
      prefix: "real/VCD"
      method: "external_vcd_real"
      grouping: "per_image"
      identity_pattern: "real__VCD__(?P<md5>[a-f0-9]{32})_"
      identity_train_fraction: 0.20
      identity_split_seed: 737
      max_frames_per_identity: 10
      max_total_samples: 800
      deterministic: true

  sampling:
    strategy: "identity_resample_weighted"
    family_weights:
      df40_fake: 0.9
      visomaster_fake: 0.8
      deeplive_non_enhanced_fake: 2.4
      deeplive_enhanced_fake: 3.0
      df40_real: 1.0
      realpool_real: 1.0
      external_real: 0.6        # Modest — avoid overwhelming fake detection

  train_split: 0.85
  val_split: 0.10
  test_split: 0.05
  seed: 737

  holdout:
    mode: "method_holdout"
    df40_orientation: "source_target"   # NOTE: fixed from target_source
    methods:
      - "visomaster_Inswapper128"
      - "visomaster_GhostFace-v2"
    max_samples_per_method: 300

  ood_monitoring:
    enabled: true
    exclude_training_identities: true   # NEW: exclude VCD training identities
    external_real_sources:
      - bucket: "effort-collected-data"
        prefix: "real/external_youtube_avspeech"
        method: "external_youtube_avspeech"
        grouping: "by_folder"
        max_videos: 200
        deterministic: true
      - bucket: "effort-collected-data"
        prefix: "real/VCD"
        method: "zoom_vcd_real"
        grouping: "per_image"
        max_videos: 1200              # Will be reduced by identity exclusion
        deterministic: true
    external_fake_sources:
      - bucket: "effort-collected-data"
        prefix: "wma_validation/enhanced_fake"
        method: "wma_failure_fake"
        grouping: "per_image"
        max_videos: 1202
        deterministic: true

augmentation:
  version: "quality_targeted_family"
  strength: "vcd_targeted"              # NEW preset
  routing:
    mode: "family_aware"
    enhanced_strategy_names:
      - "quality_enhancement"
      - "edge_cases_enhanced"
      - "minimal_processing_enhanced"

use_arcface_head: true
arcface_m: 0.0
arcface_s: 10.0
s_start: 10.0
s_end: 18.0
anneal_steps: 30000

evaluate_every_steps: 500
ood_monitoring_start_step: 1000
ood_monitoring_every_steps: 2000
test_batch_size: 32
seed: 737

early_stopping_enabled: true
early_stopping_patience: 10

gcs_assets:
  clip_backbone:
    gcs_path: "gs://base-checkpoints/effort-aigi/CLIP-ViT-B-16-DataComp.XL-s13B-b90K/"
    local_path: "./weights/CLIP-ViT-B-16-DataComp.XL-s13B-b90K/"
    files:
      - open_clip_config.json
      - open_clip_pytorch_model.bin

checkpointing:
  gcs_prefix: "gs://training-job-outputs/phase2r6_experiments/"
```

---

## 10. Validation and Success Criteria

### 10.1 Primary Success Metrics

| Metric | Baseline (R4 FT7) | Target | Hard Minimum |
|---|---:|---:|---:|
| VCD real accuracy | ~51-60% | **>80%** | >70% |
| External real FPR (YouTube) | 4.38% | ≤8% | ≤10% |
| WMA flat per-image detection | 88.35% | ≥85% | ≥50% |
| Enhanced DeepLive fake TPR | 99.53% | ≥95% | ≥90% |
| DF40 fake TPR | 86.85% | ≥85% | ≥80% |
| In-dist AUC | 0.9941 | ≥0.98 | ≥0.97 |

### 10.2 Regression Gates

Any candidate MUST NOT regress below hard minimums. If VCD real accuracy improves to >80% but WMA detection drops below 50%, the run is a failure.

### 10.3 Verification Steps

1. **Unit tests**: New tests for `UnifiedUnpairedRealSample`, `_discover_external_training_reals()`, `_iterate_unpaired_real_sample()`, and family routing for external reals
2. **Smoke run**: `R6_SMOKE_30MIN` with `total_training_steps: 1000` to verify data loading, batch composition, and loss convergence
3. **In-training OOD monitoring**: VCD + WMA + YouTube checked every 2000 steps — look for VCD accuracy improvement and WMA stability
4. **Post-training sidecar validation**: Full sidecar suite on all external sources with canonical grouping semantics

### 10.4 Rollback Criteria

If after 30K steps:
- VCD accuracy is still <65% AND
- WMA detection has regressed more than 5 percentage points

→ Kill the run and revert to R5 baseline. The interventions are not working as expected.

---

## 11. File Reference

### Files to CREATE

| File | Purpose |
|---|---|
| `experiments/phase2_round6/R6_S1_vcd_reals_aug.yaml` | VCD reals + augmentation tuning |
| `experiments/phase2_round6/R6_S2_aug_only.yaml` | Augmentation-only ablation |
| `experiments/phase2_round6/R6_S3_vcd_reals_aug_grl.yaml` | Full stack with gradient reversal |
| `experiments/phase2_round6/R6_SMOKE_30MIN.yaml` | Fast integration smoke test |
| `experiments/phase2_round6/README.md` | Round 6 documentation |
| `tests/test_unpaired_reals.py` | Unit tests for unpaired real support |

### Files to MODIFY

| File | Lines | Change |
|---|---|---|
| `data/sources/combined_paired.py` | ~85 | Add `UnifiedUnpairedRealSample` dataclass |
| `data/sources/combined_paired.py` | ~637 | Add `_discover_external_training_reals()` function |
| `data/sources/combined_paired.py` | ~820 | Update `_sample_family_for_sampling()` for unpaired reals |
| `data/sources/combined_paired.py` | ~1098 | Add dispatch for unpaired reals in `__iter__` |
| `data/sources/combined_paired.py` | ~1293 | Add `_iterate_unpaired_real_sample()` method |
| `data/sources/combined_paired.py` | ~1800 | Wire external reals into `create_combined_paired_pipeline()` |
| `data/augmentations/pipelines.py` | ~770 | Add `"vcd_targeted"` preset to `_QUALITY_TARGETED_PRESETS` |
| `data/augmentations/pipelines.py` | ~860 | Tune real-family augmentation parameters |
| `detectors/effort_detector.py` | ~200 | Add `GradientReversalLayer`, `QualityDomainHead` classes |
| `detectors/effort_detector.py` | ~201 | Add quality head init to `EffortDetector.__init__` |
| `detectors/effort_detector.py` | ~940 | Add quality domain prediction to `forward()` |
| `detectors/effort_detector.py` | ~688 | Add quality domain loss to `get_losses()` |
| `trainer/trainer.py` | Training loop | Add lambda scheduling for GRL |
| `trainer/trainer.py` | Frame counting | Fix total frame estimation for mixed paired/unpaired |

### Files for REFERENCE (read-only)

| File | Purpose |
|---|---|
| `analysis_results/real_quality_summary_table.txt` | Quality metric means ± std for all 4 real sources |
| `analysis_results/vcd_quality_divergence_summary.txt` | VCD vs YouTube Cohen's d for each metric |
| `analysis_results/real_quality_fingerprints.csv` | Per-image quality data (37 metrics × 1200 images) |
| `analysis_results/plots/codec_simulation_demo/` | 5 visual demo plots showing codec sim limitations |
| `weights/quality_analysis_samples/` | 300 cached images per source (df40_real, vcd_real, youtube_real, real_or_virtual) |
| `utils/grouping.py` | `infer_group_key` and `infer_family_key` — routing logic for family-aware augmentation |
| `data/validation_sources.py` | `load_external_real_videos` — GCS listing pattern to reuse |
| `data/augmentations/transforms.py` | `VideoCodecSimulation` class (line 958) |
| `experiments/phase2_round5/R5_S3_scratch_ft7mix_weighted_targetdomain_aug.yaml` | Closest existing config to base R6 on |

---

## Appendix: Key Data Tables

### A.1 Quality Metric Summary (mean ± std)

Full table in `analysis_results/real_quality_summary_table.txt`. Most critical rows:

```
Metric                             DF40 Paired Reals    Webcam Tests         Zoom VCD (FAILING)  YouTube AVSpeech
---------------------------------------------------------------------------
sharpness_laplacian_var              38.854 ±  49.278   251.178 ± 157.897   295.293 ± 339.996   430.321 ± 659.290
sharpness_tenengrad                  32.793 ±   8.711    45.384 ±   9.454    40.894 ±  14.136    54.487 ±  27.545
edge_density                          0.022 ±   0.015     0.050 ±   0.016     0.044 ±   0.026     0.070 ±   0.047
freq_high_ratio                       0.145 ±   0.023     0.228 ±   0.047     0.249 ±   0.053     0.166 ±   0.037
psd_slope                            -2.071 ±   0.119    -1.742 ±   0.132    -1.665 ±   0.161    -1.942 ±   0.162
effective_resolution_90pct            0.210 ±   0.049     0.378 ±   0.085     0.405 ±   0.088     0.253 ±   0.078
texture_local_var_mean                3.593 ±   4.540    20.688 ±  11.951    23.310 ±  23.698    36.857 ±  58.905
hf_noise_std                          1.971 ±   0.959     4.580 ±   1.199     4.491 ±   2.060     5.569 ±   3.209
noise_estimate                        0.381 ±   0.111     0.996 ±   0.434     1.078 ±   0.813     1.185 ±   0.747
```

### A.2 VCD vs YouTube Divergence (Top 10 by |Cohen's d|)

```
Metric                                 Cohen d    Severity
effective_resolution_90pct               1.831      LARGE
freq_high_ratio                          1.812      LARGE
psd_slope                                1.713      LARGE
freq_high_to_low                         1.445      LARGE
freq_low_ratio                          -1.174      LARGE
psd_r_squared                            0.989      LARGE
width                                    0.938      LARGE
height                                   0.915      LARGE
num_pixels                               0.829      LARGE
edge_density                            -0.676     MEDIUM
```

### A.3 VCD GCS Data Structure

```
gs://effort-collected-data/real/VCD/
├── <md5_hash_1>_<WxH>_<fps>/
│   ├── 0000.png
│   ├── 0001.png
│   └── ...
├── <md5_hash_2>_<WxH>_<fps>/
│   ├── 0000.png
│   └── ...
└── ...

~136 unique identities (MD5 hashes)
~1200 total images
Filename pattern: real__VCD__<32-char-md5>_<W>x<H>_<fps>__<frame>.png
```

### A.4 Current R5 Family Weight Config

From `R5_S3_scratch_ft7mix_weighted_targetdomain_aug.yaml`:
```yaml
sampling:
  strategy: "identity_resample_weighted"
  family_weights:
    df40_fake: 0.9
    visomaster_fake: 0.8
    deeplive_non_enhanced_fake: 2.4
    deeplive_enhanced_fake: 3.0
    df40_real: 1.0
    realpool_real: 1.0
    external_real: 1.0
```

### A.5 Current Augmentation Presets

From `_QUALITY_TARGETED_PRESETS` in `data/augmentations/pipelines.py`:

| Parameter | light | moderate | strong |
|---|---:|---:|---:|
| jpeg_lower | 62 | 48 | 40 |
| jpeg_upper | 95 | 92 | 90 |
| blur_limit | (3, 5) | (3, 7) | (3, 9) |
| noise_var | (4, 20) | (8, 35) | (10, 45) |
| downscale_min | 0.72 | 0.58 | 0.50 |
| downscale_max | 0.90 | 0.85 | 0.80 |
| quality_p | 0.38 | 0.52 | 0.60 |
| webcam_codec_p | 0.10 | 0.15 | 0.22 |
| webcam_codec_quality | (40, 80) | (30, 75) | (25, 70) |
| sharpen_alpha_real | (0.18, 0.45) | (0.24, 0.55) | (0.26, 0.62) |

### A.6 R4 Scoreboard (Current Best)

| Run | External Real FPR | WMA Flat | Enhanced DL TPR | DF40 Fake TPR | AUC |
|---|---:|---:|---:|---:|---:|
| FT5 | 3.35% | 74.29% | 98.38% | 83.08% | 0.9943 |
| FT7 | 4.38% | 88.35% | 99.53% | 86.85% | 0.9941 |

### A.7 Architecture Constants

- **Backbone**: CLIP ViT-B-16 (LAION DataComp-XL pretrained)
- **Hidden size**: 512 (ViT-B-16) — features shape is [B, 512]
- **SVD rank config**: `rank: 736` → for 512-dim layers, k = 512 - (512 - rank%512) = 32 trainable singular directions per layer
- **Head**: ArcFace (ArcMarginProduct) with s=10→18 annealing, m=0.0
- **Loss**: Cross-entropy on ArcFace-penalized logits + orthogonal regularization (λ=0.01)
- **Training**: 65K steps, cosine LR with 1K warmup, batch size 32 frames

---

*End of plan. All code changes should be implemented against the codebase at `DeepfakeBench/training/` in the workspace root.*
