# Phase 2 Round 6 — Real Robustness Intervention: Full Experiment Report

**Date**: February 19, 2026  
**Status**: All 8 runs in progress (~6K/12K steps at time of writing)  
**W&B Project**: `dtect-vision/phase2-round6`  
**Vertex AI Region**: `asia-southeast1`  

**Post-report engineering update (February 19, 2026, later)**:
- Added explicit config propagation for `use_quality_domain_head` and related GRL keys in `utils/config_helpers.py`.
- Added fail-fast runtime guards so GRL training now errors if `quality_domain` labels or logits are missing.
- Added `quality_domain_require_labels: true` to all GRL-enabled R6 configs to prevent silent zero-loss runs.
- Added smoke-startup acceleration: DeepLive discovery cache + per-strategy cap, external-real cache manifests, and smoke-only skip of OOD loader build at startup to avoid hour-long pre-training discovery.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Root Cause Analysis](#2-root-cause-analysis)
3. [Intervention Plan (3 Parts)](#3-intervention-plan)
4. [Implementation Details](#4-implementation-details)
5. [Experiment Matrix (8 Runs)](#5-experiment-matrix)
6. [Training Configuration](#6-training-configuration)
7. [Results at ~6K Steps](#7-results-at-6k-steps)
8. [Analysis and Findings](#8-analysis-and-findings)
9. [Critical Bug: GRL Quality Domain Loss = 0](#9-critical-bug-grl-quality-domain-loss--0)
10. [Investigation Seeds and Next Steps](#10-investigation-seeds-and-next-steps)

---

## 1. Problem Statement

The Effort detector (SVD residual fine-tuning on CLIP ViT-B-16, ICML 2025 Oral) achieves excellent in-distribution performance (AUC >0.99, enhanced fake TPR >98%) but **fails catastrophically on Zoom VCD real images** — classifying ~40-49% of genuine video-call face crops as fake (FPR 40-49%). VCD (Video Call Detection) represents the exact deployment domain: real people on live Zoom calls.

### Baseline Performance (R4/R5 best — FT7)

| Source | Metric | Value |
|---|---|---:|
| DF40 (in-dist) | Fake TPR | 86.85% |
| Enhanced DeepLive (in-dist) | Fake TPR | 99.53% |
| WMA flat (OOD) | Per-image accuracy | 88.35% |
| YouTube AVSpeech (OOD real) | Real accuracy | ~96% |
| **Zoom VCD (OOD real)** | **Real accuracy** | **~51-60%** |

The YouTube and VCD OOD divergence is the smoking gun: both are real videos from external sources, but only VCD fails. Something about VCD's image characteristics triggers false positives.

---

## 2. Root Cause Analysis

### 2.1 DF40 Training Reals Are Quality Outliers

A comprehensive quality fingerprint analysis (300 images × 4 sources, 37 quality metrics) revealed that **DF40 training reals are not representative of real-world faces — they are the outlier**:

| Metric | DF40 (train) | VCD (failing) | YouTube (passing) | Webcam Tests |
|---|---:|---:|---:|---:|
| sharpness_laplacian_var | **38.9** | 295.3 | 430.3 | 251.2 |
| sharpness_tenengrad | **32.8** | 40.9 | 54.5 | 45.4 |
| edge_density | **0.022** | 0.044 | 0.070 | 0.050 |
| freq_high_ratio | **0.145** | 0.249 | 0.166 | 0.228 |
| texture_local_var_mean | **3.6** | 23.3 | 36.9 | 20.7 |
| hf_noise_std | **1.97** | 4.49 | 5.57 | 4.58 |
| noise_estimate | **0.38** | 1.08 | 1.19 | 1.00 |
| effective_resolution | **0.210** | 0.405 | 0.253 | 0.378 |
| psd_slope | **-2.071** | -1.665 | -1.942 | -1.742 |

DF40 reals are by far the **softest** (sharpness 38.9 vs 295+), **smoothest** (texture 3.6 vs 23+), and **least noisy** (HF noise 1.97 vs 4.49+). The model learned: **"soft, smooth, low-texture face = real."** VCD reals are sharper, noisier, and have more high-frequency content — properties that overlap with fake artifacts.

### 2.2 VCD's Unique Codec Fingerprint

VCD vs YouTube divergence (highest Cohen's d):

| Metric | Cohen's d | Interpretation |
|---|---:|---|
| effective_resolution_90pct | 1.831 | VCD has much higher effective resolution |
| freq_high_ratio | 1.812 | VCD retains far more HF energy |
| psd_slope | 1.713 | VCD has flatter power spectrum |
| freq_high_to_low | 1.445 | VCD ratio is much higher |

VCD's signature: Zoom's encoder preserves more high-frequency detail than YouTube's heavier compression, but introduces codec noise (ringing, mosquito noise) that elevates the HF energy ratio. This is the "webcam codec fingerprint": **sharp + noisy**, which the model interprets as fake-like.

### 2.3 Why SVD Residual Is Particularly Vulnerable

The Effort model trains `SVDResidualLinear` layers with only **k=32 trainable singular directions per layer** (out of 512-768 total). This extremely constrained subspace is efficient but makes the model vulnerable to shortcut learning: with training reals in a narrow quality band and fakes showing diverse quality, the residual subspace encoded quality as a proxy for real/fake classification.

### 2.4 What We Already Tried (R5)

- **Codec simulation augention**: Closes only 25-40% of the quality gap and moves the **wrong direction** on sharpness/edge metrics (makes images blurrier, but VCD reals are sharper than DF40)
- **Generic "more augmentation"**: R3 proved counterproductive — over-augmentation destroys training signal without targeting the actual quality shortcut
- **Augmentation-only on existing data**: Fundamental limit — can't augment DF40 reals to be simultaneously sharper AND noisier (VCD's profile) without making them look fake-like

---

## 3. Intervention Plan

Three coordinated, complementary interventions implemented together:

### Part A: Unpaired VCD Reals in Training

Add ~200-300 Zoom VCD real images directly to the training set. These come from the `effort-collected-data` GCS bucket, `real/VCD` prefix (~1200 total images, ~136 unique identities). We use 20% of identities for training (deterministic split by sorted MD5 hash), the rest stays reserved for OOD monitoring.

**Key constraints**:
- Identity-split: no identity leakage between train and OOD eval
- Modest sampling weight (`external_real: 0.6`, vs `df40_real: 1.0`) to avoid overwhelming fake-detection gradients
- Cap at 800 frames, 10 per identity

### Part B: Augmentation Tuning (`vcd_targeted` Preset)

A new augmentation preset designed to break the quality-label correlation from **both sides**:
- **Sharpen reals**: Push DF40 reals toward VCD's quality profile (tenengrad 40.9 vs DF40's 32.8)
- **Add noise to reals**: Inject VCD-like noise (noise_estimate 1.08 vs DF40's 0.38)
- **Extra-degrade fakes**: Make some fakes softer/lower-quality to overlap with the training-real quality band
- **Modest codec sim** (12% probability): Keep it low since codec sim moves the wrong direction on sharpness

### Part C: Gradient Reversal Quality-Domain Head

A DANN-style (Ganin et al., 2015) gradient reversal layer that forces the SVD residual subspace to be **uninformative about image quality domain**:
- 4 quality domains: `df40` (0), `external/webcam` (1), `deeplive/visomaster` (2), `youtube` (3)
- MLP head: `GRL → Linear(512, 128) → ReLU → Dropout(0.3) → Linear(128, 4)`
- Sigmoid lambda schedule: ramps from 0→1 over training, so GRL starts gentle
- Loss added to total training loss, weighted by `quality_domain_loss_weight`

**Why all three are needed**: A alone risks new shortcut ("webcam quality = always real"). B alone can't fully replicate VCD's codec fingerprint. C alone can't learn quality-invariance without seeing quality-diverse reals in training.

---

## 4. Implementation Details

### 4.1 Modified Files

Four source files were modified:

| File | Changes |
|---|---|
| `data/sources/combined_paired.py` | `UnifiedUnpairedRealSample` dataclass, `QUALITY_DOMAIN_MAP`, `_discover_external_training_reals()`, `_iterate_unpaired_real_sample()`, `quality_domain` in collate function, `is_unpaired_real` dispatch in `__iter__` |
| `data/augmentations/pipelines.py` | `vcd_targeted` preset in `_QUALITY_TARGETED_PRESETS` |
| `detectors/effort_detector.py` | `GradientReversalFunction`, `GradientReversalLayer`, `QualityDomainHead` classes, quality head init/forward/get_losses integration |
| `trainer/trainer.py` | `_update_quality_domain_lambda()` sigmoid scheduling method |

### 4.2 Part A Implementation: Unpaired Real Support

#### UnifiedUnpairedRealSample (combined_paired.py ~L106-123)

```python
@dataclass
class UnifiedUnpairedRealSample:
    identity: str           # "external_vcd_<md5>"
    source: str             # 'external' (triggers external_real routing)
    method: str             # "external_vcd_real" (must contain "external")
    gcs_bucket: str         # "effort-collected-data"
    frame_paths: List[str]  # GCS paths to individual frame PNGs
    sample_id: str = ""
    has_landmarks: bool = False
    is_unpaired_real: bool = True   # Distinguishes from paired samples
    original_sample: Any = None     # Duck typing compat
```

#### Quality Domain Map (combined_paired.py ~L60-66)

```python
QUALITY_DOMAIN_MAP = {
    "df40": 0,           # clean_academic — soft, smooth, low-noise
    "external": 1,       # webcam_codec — sharp, noisy, codec artifacts
    "deeplive": 2,       # studio_capture — studio lighting
    "visomaster": 2,     # studio_capture — same real source as deeplive
    "youtube": 3,        # social_media — heavier compression
}
```

#### Discovery Function (combined_paired.py ~L675-691)

`_discover_external_training_reals()` reads the `external_training_reals` config, lists GCS objects via `fsspec`, groups by MD5-identity from filenames (regex: `real__VCD__(?P<md5>[a-f0-9]{32})_`), takes the first `identity_train_fraction` (20%) of deterministically sorted identities for training, returns `List[UnifiedUnpairedRealSample]` and a set of training identity strings. The identity set is used to exclude training identities from OOD monitoring.

#### Iteration (combined_paired.py ~L1597-1654)

`_iterate_unpaired_real_sample()` loads individual PNG frames from GCS via `fsspec`/`cv2.imdecode`, applies the transform with `meta={'label': 0, 'source': source, 'method': method}`, and yields dicts with `quality_domain` set via `_quality_domain_for_source()`.

#### Collate (combined_paired.py ~L1689-1748)

The collate function collects `quality_domain` from each video group's first frame, stacks into `torch.long` tensor, and includes it in the batch dict alongside `image`, `label`, `video_id`, `method_id`.

#### Dispatch (combined_paired.py ~L1334-1340)

In `__iter__`, when a sample has `is_unpaired_real=True`, the code routes to `_iterate_unpaired_real_sample()` instead of the standard paired iterator.

All 7 yield-point dicts across all iterators (DF40, DeepLive, VisoMaster, unpaired reals, etc.) include the `quality_domain` field to ensure every sample has a domain label for the GRL head.

### 4.3 Part B Implementation: vcd_targeted Augmentation Preset

Added to `_QUALITY_TARGETED_PRESETS` in `pipelines.py` (~L850-884):

```python
"vcd_targeted": {
    "jpeg_lower": 40,           # vs "strong"'s 30
    "jpeg_upper": 90,
    "blur_limit": (3, 9),
    "webcam_codec_p": 0.12,     # vs "strong"'s 0.22 — modest, wrong direction
    "sharpen_alpha_real": (0.30, 0.70),   # Push reals toward VCD sharpness
    "real_noise_p": 0.25,       # Inject VCD-like noise on reals
    "real_noise_var": (5.0, 20.0),
    "real_sharpen_p": 0.60,     # Extra sharpening for reals
    "fake_extra_degrade_p": 0.15,  # Degrade some fakes to overlap real quality
}
```

Key design: lower `webcam_codec_p` (0.12 vs "strong"'s 0.22) since codec sim moves the wrong direction on sharpness. Instead, adds `real_noise_p` and `real_sharpen_p` to push reals toward VCD's "sharp + noisy" profile, and `fake_extra_degrade_p` to pull some fakes toward the real quality band.

### 4.4 Part C Implementation: Gradient Reversal

#### Core Classes (effort_detector.py ~L204-270)

```python
class GradientReversalFunction(torch.autograd.Function):
    """Reverses gradient by factor lambda during backprop (DANN, Ganin 2015)."""
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
    def set_lambda(self, val): self.lambda_val = val
    def forward(self, x): return GradientReversalFunction.apply(x, self.lambda_val)

class QualityDomainHead(nn.Module):
    """GRL → Linear(512,128) → ReLU → Dropout(0.3) → Linear(128,4)"""
    def __init__(self, in_features, num_domains=4, hidden_dim=128):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(hidden_dim, num_domains),
        )
    def set_lambda(self, val): self.grl.set_lambda(val)
    def forward(self, features):
        return self.classifier(self.grl(features))
```

#### Init (effort_detector.py ~L340-352)

Gated by `use_quality_domain_head` config flag. Creates `QualityDomainHead` with configurable `num_domains`, `hidden_dim`, `loss_weight`.

#### Forward (effort_detector.py ~L1058-1060)

```python
if self.use_quality_head and not inference:
    pred_dict['quality_domain_logits'] = self.quality_head(features)
```

Only runs during training (not inference). Uses the same `features` tensor that feeds the ArcFace classification head.

#### Loss (effort_detector.py ~L857-870)

```python
quality_loss = torch.tensor(0.0, device=device)
if self.use_quality_head and 'quality_domain_logits' in pred_dict and self.training:
    domain_labels = data_dict.get('quality_domain')
    if domain_labels is not None:
        domain_labels = domain_labels.to(device)
        # Expand for video batches (B quality labels → B*T frame-level logits)
        if pred_dict['quality_domain_logits'].shape[0] > domain_labels.shape[0]:
            B_q = domain_labels.shape[0]
            T_q = pred_dict['quality_domain_logits'].shape[0] // B_q
            domain_labels = domain_labels.repeat_interleave(T_q)
        quality_loss = F.cross_entropy(
            pred_dict['quality_domain_logits'], domain_labels.long()
        ) * self.quality_domain_loss_weight
```

The `quality_loss` is later added to `overall_loss` (classification + orthogonal regularization + quality domain).

#### Lambda Scheduling (trainer.py ~L226-246)

```python
def _update_quality_domain_lambda(self, step_cnt):
    progress = min(step_cnt / max(total_steps, 1), 1.0)
    lambda_val = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0  # DANN sigmoid
    model_instance.quality_head.set_lambda(lambda_val)
```

Called in the training loop. Reports `train/quality_grl_lambda` to W&B. Starts near 0 (no gradient reversal) and ramps to ~1.0 (full reversal) by end of training. At 50% training (6K steps), lambda ≈ 0.73.

---

## 5. Experiment Matrix

### 5.1 Core Ablation (S1–S4)

| Experiment | VCD Reals | Augmentation | GRL Head | Purpose |
|---|:---:|:---:|:---:|---|
| **S1** | ✅ | vcd_targeted | ❌ | Data + aug without adversarial training |
| **S2** | ❌ | vcd_targeted | ❌ | Aug-only ablation — baseline for comparison |
| **S3** | ✅ | vcd_targeted | ✅ (0.1) | **Full stack** — all three interventions |
| **S4** | ❌ | vcd_targeted | ✅ (0.1) | GRL-only ablation — can GRL help without VCD data? |

### 5.2 Extended Experiments (S5–S8)

| Experiment | Diff from S3 | Purpose |
|---|---|---|
| **S5** | `strength: "strong"` instead of `"vcd_targeted"` | Isolate vcd_targeted aug tuning contribution |
| **S6** | `external_real` weight 1.5 instead of 0.6 | Test if higher VCD sampling frequency helps |
| **S7** | `quality_domain_loss_weight: 0.25` instead of 0.1 | Test if stronger GRL adversarial signal helps |
| **S8** | `identity_train_fraction: 0.40` instead of 0.20 | Test if more VCD identity diversity helps |

### 5.3 W&B Run IDs

| Experiment | Run ID | Name |
|---|---|---|
| S1 | `s3tx3fk4` | R6_S1_vcd_reals_aug |
| S2 | `0ip38ipo` | R6_S2_aug_only |
| S3 | `iwk4pe1j` | R6_S3_vcd_reals_aug_grl |
| S4 | `l1wj2i8k` | R6_S4_grl_only |
| S5 | `cm8sgqe9` | R6_S5_vcd_reals_strong_aug |
| S6 | `rlyym6yn` | R6_S6_full_stack_high_vcd_weight |
| S7 | `z54lwhda` | R6_S7_full_stack_strong_grl |
| S8 | `mverbl00` | R6_S8_full_stack_more_vcd_ids |
| Smoke | `9e5jc6hx` | R6_SMOKE_30MIN (crashed as expected after 2 epochs) |

---

## 6. Training Configuration

All 8 experiments share these settings:

| Parameter | Value |
|---|---|
| Backbone | CLIP ViT-B-16 (LAION DataComp-XL, `open_clip`) |
| Hidden size | 512 |
| SVD residual rank | k=32 trainable singular directions per layer |
| Total rank (frozen + trainable) | 736 |
| Learning rate | 2.0e-4 |
| LR scheduler | Cosine with warmup |
| Warmup steps | 1000 |
| Weight decay | 0.05 |
| Gradient clip | 1.0 |
| Total training steps | 12,000 |
| Epochs | 10 |
| ArcFace scale annealing | s=10 → 18 over 24,000 steps (conservative; intentionally exceeds training length) |
| ArcFace margin | m=0.0 |
| Batch size | 32 frames |
| Frames per video | 8 |
| Orthogonal regularization | λ=0.01 |
| Data source | `combined_paired` (DF40 + DeepLive + VisoMaster + external reals) |
| Holdout methods | visomaster_Inswapper128, visomaster_GhostFace-v2 (300 samples each) |
| OOD monitoring | VCD reals (1200), YouTube reals (200), WMA fakes (1202) |

**Why `anneal_steps=24000`**: Even though training is only 12K steps, the ArcFace scale only reaches ~14 (midway through the anneal). This is intentional — aggressive scale annealing caused training instability in earlier rounds.

---

## 7. Results at ~6K Steps

### 7.1 Scoreboard (Sorted by Holdout AUC)

| Exp | Holdout AUC | VCD Real Acc | YouTube Acc | OOD AUC | In-Dist AUC | Best Epoch |
|---|---:|---:|---:|---:|---:|---:|
| **S3** (full stack) | **0.9694** | **0.698** | 0.853 | **0.955** | 0.9912 | 3 |
| **S6** (high VCD wt) | 0.9692 | **0.701** | 0.853 | 0.953 | 0.9912 | 2 |
| **S1** (VCD+aug) | 0.9691 | 0.698 | 0.853 | 0.954 | 0.9911 | 3 |
| **S7** (strong GRL) | 0.9689 | 0.700 | 0.853 | 0.953 | 0.9910 | 3 |
| S5 (strong aug) | 0.9665 | 0.616 | 0.822 | 0.952 | 0.9911 | 3 |
| S8 (more VCD ids) | 0.9660 | 0.663 | 0.808 | 0.948 | 0.9905 | 2 |
| S4 (GRL only) | 0.9655 | 0.549 | 0.826 | 0.922 | 0.9919 | 3 |
| S2 (aug only) | 0.9654 | 0.550 | 0.831 | 0.921 | 0.9919 | 3 |

### 7.2 Per-Method In-Dist Accuracy (Best Runs)

S1/S3/S6/S7 show nearly identical in-distribution per-method accuracy:

| Validation Source | Accuracy |
|---|---:|
| df40_fake (holdout) | ~97% |
| df40_real | ~97% |
| deeplive_enhanced_fake | ~99% |
| deeplive_non_enhanced_fake | ~89% |
| visomaster_fake | ~95% |
| **external_vcd_real (in-dist)** | **0%** |

### 7.3 OOD WMA Fake Detection

| Exp | WMA Fake Acc |
|---|---:|
| S1/S3/S6/S7 | ~95-96% |
| S2/S4 | ~92% |
| S5/S8 | ~93-95% |

No degradation of fake detection from the interventions.

---

## 8. Analysis and Findings

### 8.1 Finding 1: VCD Reals in Training Are the Primary Lever

The clearest signal: **S1/S3/S6/S7 (with VCD reals) achieve ~70% VCD real accuracy vs S2/S4 (without) at ~55%**. This is a +20pp improvement from baseline (which was at ~50-60%). It confirms that the model needs actual target-domain data to expand its real-class support.

The ~55% accuracy for S2/S4 (without VCD reals) is still slightly above the R4/R5 baseline of ~50-60%, suggesting the `vcd_targeted` augmentation alone provides a small boost — but not nearly enough.

### 8.2 Finding 2: GRL Has Zero Observable Effect

**S1 (VCD+aug, no GRL) ≈ S3 (full stack with GRL)** across all metrics. The GRL head adds literally nothing measurable. Similarly, S2 (aug only) ≈ S4 (aug + GRL). This is because the GRL is broken — see Section 9.

### 8.3 Finding 3: `vcd_targeted` Augmentation Matters

**S3 (vcd_targeted) outperforms S5 (strong) on VCD real accuracy: 69.8% vs 61.6%**. The "strong" preset has higher `webcam_codec_p` (0.22 vs 0.12) which moves the wrong direction on sharpness, and lacks the bidirectional real-sharpening/fake-degradation components. This validates the targeted augmentation design.

S5 also shows lower YouTube accuracy (82.2% vs 85.3%), suggesting the "strong" preset is too aggressive and hurts general OOD robustness.

### 8.4 Finding 4: More VCD Data Doesn't Help (Counterintuitively)

**S8 (40% VCD identities, ~double the data) underperforms S3 (20% identities): 66.3% vs 69.8% VCD real accuracy**. S8 also has lower holdout AUC (0.9660 vs 0.9694) and much lower YouTube accuracy (80.8% vs 85.3%).

Possible explanations:
- More VCD reals may dilute the fake-class signal with k=32 constrained residual
- The `max_total_samples=800` cap means S8 has the same total frames but from more identities (fewer frames per identity), reducing per-identity diversity
- With `external_real: 0.6` weight, doubling identities may cause the real-class distribution to shift too far toward VCD, creating a new domain imbalance

### 8.5 Finding 5: Higher VCD Weight Has No Effect

**S6 (VCD weight 1.5) ≈ S3 (VCD weight 0.6)** — 70.1% vs 69.8% VCD real accuracy. The identity-balanced sampling may be buffering the weight difference. Or the bottleneck isn't frequency of VCD samples in training but something more fundamental.

### 8.6 Finding 6: In-Distribution AUC Slightly Lower with VCD Reals

S2/S4 (no VCD reals) achieve in-dist AUC of 0.9919 vs S1/S3's 0.9911-0.9912. This is a minor regression (~0.08pp) but consistent — the VCD reals are slightly diluting the in-distribution signal. This is an acceptable tradeoff for the +20pp VCD real accuracy gain.

### 8.7 Summary: Tier Ranking

**Tier 1 (Best)**: S1 ≈ S3 ≈ S6 ≈ S7 — All achieve VCD ~70%, holdout AUC ~0.969, YouTube ~85%  
**Tier 2**: S5 — VCD 61.6%, strong aug too aggressive  
**Tier 3**: S8 — VCD 66.3%, more VCD data hurts  
**Tier 4**: S2 ≈ S4 — VCD ~55%, no target domain data, GRL does nothing alone  

The simplest effective configuration is **S1** (VCD reals + vcd_targeted aug, no GRL), which matches S3's performance without the broken GRL head.

---

## 9. Critical Bug: GRL Quality Domain Loss = 0

### 9.1 The Symptom

In **ALL 8 runs**, including S3 and S7 (which have `use_quality_domain_head: true`), the W&B metric `quality_domain_loss` reads **exactly 0.0** throughout training. The GRL head is not producing any loss signal. This explains why S3 ≈ S1 and S4 ≈ S2.

Additionally, `train/quality_grl_lambda` is being logged correctly (sigmoid ramp from 0 to ~0.73 at 6K steps), confirming the scheduling code runs. The issue is in the loss computation path, not the lambda scheduling.

### 9.2 Another Anomaly: external_vcd_real In-Dist Accuracy = 0%

In validation, the `external_vcd_real` source shows **0% accuracy** in all runs with VCD data (S1, S3, S5, S6, S7). This means every VCD real in the in-dist validation split is classified as fake. Strangely, S8 shows 100% — possibly a validation set size artifact (S8's larger identity split may have very few validation samples, which happened to be correct).

This is at odds with the OOD VCD real accuracy of ~70%, which evaluates on the remaining 80% of VCD identities. The in-dist validation VCD reals (from the training-split 20% of identities) are all misclassified, while the OOD VCD reals (different identities) are 70% correct. This suggests the model may be overfitting to the training-split VCD identities in a negative way (perhaps the small number of training identities has extreme quality characteristics).

### 9.3 Hypotheses for GRL Loss = 0

1. **`quality_domain` tensor missing from batch**: The collate function builds `quality_domain` and includes it in the batch dict, but the `data_dict` received by `get_losses` might not contain it. Possible causes:
   - The tensor might be dropped somewhere in the data pipeline between collate and model forward
   - The key name might not match (`quality_domain` vs `video_quality_domains` or similar)

2. **domain_labels is None at loss time**: The `get_losses` code checks `data_dict.get('quality_domain')` — if this returns `None`, quality loss stays at 0.0. The code doesn't log whether domain_labels was None, making this hard to diagnose from W&B alone.

3. **quality_domain_logits missing from pred_dict**: The forward code checks `self.use_quality_head` — if the config key `use_quality_domain_head` doesn't properly propagate to `self.use_quality_head`, the logits are never computed.

4. **Shape mismatch in repeat_interleave**: The loss code expands domain labels for video batches. If the shape calculation is wrong, it could error silently or produce a zero tensor.

5. **Not logging the right metric**: The quality_loss value might be computed but reported under a different key or not reported at all — and the loss might actually be contributing to `overall_loss` without being separately visible.

### 9.4 Most Likely Root Cause

The most likely issue is **hypothesis 1 or 2**: the `quality_domain` key from the collate function is not reaching the model's `get_losses` method. The training pipeline may pass only specific keys from the batch dict (e.g., `image`, `label`, `method_id`) and drop unknown keys. This would need to be verified by checking the trainer code that transforms batch dict into `data_dict` before calling `model.get_losses()`.

---

## 10. Investigation Seeds and Next Steps

### 10.1 Immediate: Fix the GRL Bug

Before launching any new experiments, the GRL must be debugged:

1. **Add logging** in `get_losses` to check whether `data_dict.get('quality_domain')` is None
2. **Trace the data path** from collate → trainer batch handling → `data_dict` passed to model. Look for key filtering/dropping.
3. **Add assertion** that quality_domain_logits exists in pred_dict when use_quality_head is True
4. **Log `quality_loss` as a separate W&B metric** (not just whether it contributed to overall_loss)

### 10.2 Investigation: Why external_vcd_real In-Dist = 0%

The in-dist VCD accuracy of 0% while OOD VCD is 70% is very suspicious. Possible investigations:

- Check how many VCD samples are in the validation split (could be very few)
- Verify the validation pipeline uses the correct augmentation (or no augmentation)
- Check if the validation VCD samples are being augmented differently or loaded incorrectly
- Compare the quality characteristics of training-split vs OOD-split VCD identities

### 10.3 Next Round Ideas (After GRL Fix)

1. **Increase VCD weight with identity-balanced sampling tuning**: S6 showed no effect from higher weight — investigate why identity-balanced sampling may be capping VCD exposure

2. **More diverse real sources**: Add YouTube reals or webcam captures to training (not just VCD) to broaden the quality spectrum further. Currently training reals span DF40 (very soft) and VCD (sharp+noisy) but nothing in between.

3. **Per-frame domain labels**: Currently quality_domain is per-video (first frame's domain). With video batches of 8 frames, all frames get the same domain label. This is correct for single-source videos but doesn't capture frame-level augmentation-induced quality variation.

4. **Separate quality-head learning rate**: The GRL head may need a different LR than the main backbone. Too high LR → GRL oscillates; too low → never learns.

5. **Directly optimizing VCD accuracy**: Consider adding VCD real samples (training-split) to the validation loop as a direct optimization target, with early stopping based on a composite metric (e.g., `0.6 * holdout_AUC + 0.4 * vcd_real_acc`).

6. **Feature analysis**: Extract CLIP features from VCD vs DF40 reals and visualize with t-SNE/UMAP. This would show whether the VCD reals cluster separately in feature space and whether the trained model has started merging these clusters.

### 10.4 Final Runs Status

All 8 runs are still active at ~6K/12K steps. Several show `epochs_without_improvement` = 2-3, suggesting they may be plateauing. Wait for completion and evaluate final checkpoints before launching R7.

### 10.5 Key Result to Carry Forward

**The VCD real accuracy improved from ~50-60% (R4/R5 baseline) to ~70% (R6 S1/S3/S6/S7) by adding VCD reals to training with vcd_targeted augmentation.** This is meaningful progress but still far from the target of >90%. The GRL adversarial training — the theoretically strongest intervention — never activated due to a bug. Fixing this is the highest priority for R7.

---

## Appendix A: Config File Locations

| File | Path (relative to `DeepfakeBench/training/`) |
|---|---|
| S1 config | `experiments/phase2_round6/R6_S1_vcd_reals_aug.yaml` |
| S2 config | `experiments/phase2_round6/R6_S2_aug_only.yaml` |
| S3 config | `experiments/phase2_round6/R6_S3_vcd_reals_aug_grl.yaml` |
| S4 config | `experiments/phase2_round6/R6_S4_grl_only.yaml` |
| S5 config | `experiments/phase2_round6/R6_S5_vcd_reals_strong_aug.yaml` |
| S6 config | `experiments/phase2_round6/R6_S6_full_stack_high_vcd_weight.yaml` |
| S7 config | `experiments/phase2_round6/R6_S7_full_stack_strong_grl.yaml` |
| S8 config | `experiments/phase2_round6/R6_S8_full_stack_more_vcd_ids.yaml` |
| Original plan | `experiments/REAL_ROBUSTNESS_PLAN.md` |
| This report | `experiments/phase2_round6/R6_EXPERIMENT_REPORT.md` |

## Appendix B: Implementation File References

| Component | File (relative to `DeepfakeBench/training/`) | Key Lines |
|---|---|---|
| QUALITY_DOMAIN_MAP | `data/sources/combined_paired.py` | ~L60-66 |
| UnifiedUnpairedRealSample | `data/sources/combined_paired.py` | ~L106-123 |
| _discover_external_training_reals | `data/sources/combined_paired.py` | ~L675-700 |
| _iterate_unpaired_real_sample | `data/sources/combined_paired.py` | ~L1597-1654 |
| is_unpaired_real dispatch | `data/sources/combined_paired.py` | ~L1334-1340 |
| collate quality_domain | `data/sources/combined_paired.py` | ~L1689-1748 |
| vcd_targeted preset | `data/augmentations/pipelines.py` | ~L850-884 |
| GradientReversalFunction | `detectors/effort_detector.py` | ~L204-215 |
| GradientReversalLayer | `detectors/effort_detector.py` | ~L217-226 |
| QualityDomainHead | `detectors/effort_detector.py` | ~L229-270 |
| Quality head init | `detectors/effort_detector.py` | ~L340-352 |
| Quality domain forward | `detectors/effort_detector.py` | ~L1058-1060 |
| Quality domain loss | `detectors/effort_detector.py` | ~L857-870 |
| _update_quality_domain_lambda | `trainer/trainer.py` | ~L226-246 |

## Appendix C: Launch Commands

```bash
cd DeepfakeBench/training

# S1-S4 (core ablation)
./launch_experiment.sh phase2-round6 asia-southeast1 experiments/phase2_round6/R6_S1_vcd_reals_aug.yaml -y
./launch_experiment.sh phase2-round6 asia-southeast1 experiments/phase2_round6/R6_S2_aug_only.yaml -y
./launch_experiment.sh phase2-round6 asia-southeast1 experiments/phase2_round6/R6_S3_vcd_reals_aug_grl.yaml -y
./launch_experiment.sh phase2-round6 asia-southeast1 experiments/phase2_round6/R6_S4_grl_only.yaml -y

# S5-S8 (extended)
./launch_experiment.sh phase2-round6 asia-southeast1 experiments/phase2_round6/R6_S5_vcd_reals_strong_aug.yaml -y
./launch_experiment.sh phase2-round6 asia-southeast1 experiments/phase2_round6/R6_S6_full_stack_high_vcd_weight.yaml -y
./launch_experiment.sh phase2-round6 asia-southeast1 experiments/phase2_round6/R6_S7_full_stack_strong_grl.yaml -y
./launch_experiment.sh phase2-round6 asia-southeast1 experiments/phase2_round6/R6_S8_full_stack_more_vcd_ids.yaml -y
```

## Appendix D: Quality Fingerprint Data Sources

| Source | Location |
|---|---|
| Full quality metrics table | `analysis_results/real_quality_summary_table.txt` |
| Per-image quality data | `analysis_results/real_quality_fingerprints.csv` |
| VCD vs YouTube divergence | `analysis_results/vcd_quality_divergence_summary.txt` |
| Codec sim demo plots | `analysis_results/plots/codec_simulation_demo/` |
| Quality analysis samples | `weights/quality_analysis_samples/{df40_real,real_or_virtual,vcd_real,youtube_real}/` |
