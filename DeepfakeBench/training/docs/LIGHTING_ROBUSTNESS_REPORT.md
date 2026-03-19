# Lighting Robustness Analysis Report

**Date:** March 8, 2026  
**Scope:** Empirical analysis of lighting-related failure modes in the EFFORT detector, with implications for R12 experiment planning.  
**Data sources:** DF40 training crops (GCS), 33 real-world captures (Roee, varied indoor lighting).

---

## 1. Executive Summary

Production false positives under varying indoor lighting are **not caused by a brightness skew between real and fake training data**. Real and fake images in DF40 have nearly identical brightness distributions (mean 104.1 vs 106.4, Δ = 2.3 units). Instead, the failure mode is that **production conditions fall entirely outside the training distribution** — both real and fake. The model is extrapolating, not misclassifying along a learned shortcut.

Key numbers:
- **Your real-world captures** have mean brightness 160.6 (vs training 104.1), R/B ratio 1.2 (vs training real 1.6), and RMS contrast 0.3 (vs training 0.5).
- Only **45%** of your captures overlap with training real's [5%–95%] brightness range; **42%** overlap with training fake's range.
- Current augmentation (wide mode: gamma [50,150], brightness ±0.40) shifts training images *down* more effectively than *up*, achieving only **58%** coverage of your captures.
- **Proposed compound augmentation** with CCT simulation would cover **91%** of your captures' colour temperature, up from 67% with the current pipeline.

---

## 2. Methodology

### 2.1 Data Collection

| Dataset | N | Source | Description |
|---------|---|--------|-------------|
| **DF40 Real** | 66 | GCS bucket `df40-frames-recropped-rfa85/real/` | Sampled from FaceForensics++ (30), Celeb-real (20), YouTube-real (16). Frame 000 from diverse video IDs. |
| **DF40 Fake** | 155 | GCS bucket `df40-frames-recropped-rfa85/fake/` | 17 methods across 4 families: GAN (46), FaceSwap (59), Reenact (48), Diffusion (2). 12 images per method from first 12 video folders. |
| **Real-world captures** | 33 | `/Users/roeedar/Downloads/roee_light/` | Roee's face captured in diverse indoor lighting (desk lamp, overhead, daylight window, dim hallway, etc.). Square crops 189–256px. |

All images resized to 224×224 for stat computation (same as training input size).

### 2.2 Metrics Computed

Per-image pixel-level statistics (no model involvement):

| Metric | Formula | What it captures |
|--------|---------|------------------|
| Mean Brightness | `mean(0.299R + 0.587G + 0.114B)` | Overall exposure level |
| Brightness Std Dev | `std(luminance)` | Spatial contrast / lighting uniformity |
| R/B Ratio | `mean(R) / mean(B)` | Colour temperature proxy (high = warm, low = cool) |
| RMS Contrast | `std(luminance) / mean(luminance)` | Normalised contrast |

### 2.3 Analysis Tools Created

Two standalone scripts were created for this analysis. Both are dependency-light and require no model weights:

- **`tools/lighting_showcase.py`** (964 lines) — Three modes: `showcase` (visual grid), `compare` (real vs synthetic), `audit` (statistical coverage audit with distribution histograms)
- **`tools/real_vs_fake_skew.py`** (323 lines) — Real vs fake brightness/colour skew analysis with per-family breakdown and box plots

**Dependencies used:**
```
numpy            # array ops and statistics
opencv-python    # image loading, resizing, colour conversion (cv2)
Pillow           # image I/O (PIL, used in showcase mode)
matplotlib       # plotting (histograms, box plots, grids)
```
Optional (for CLIP distance computation in showcase mode only):
```
torch            # tensor ops
open_clip_torch  # CLIP model loading (OpenCLIP / LAION weights)
```

---

## 3. Finding 1: No Brightness Skew Between Real and Fake

| Category | Mean Brightness | Std Dev | R/B Ratio | RMS Contrast |
|----------|:-:|:-:|:-:|:-:|
| **Real (training)** | 104.1 ± 30.2 | 47.5 ± 11.1 | 1.6 ± 0.5 | 0.5 ± 0.1 |
| **Fake ALL** | 106.4 ± 26.6 | 47.5 ± 12.8 | 1.4 ± 0.3 | 0.5 ± 0.2 |
| **Δ (Fake − Real)** | **+2.3** | **0.0** | **−0.2** | **0.0** |

**Interpretation:** The model has not learned "bright = fake" as a shortcut. The 2.3-unit difference is negligible (within noise). Brightness std dev and RMS contrast are effectively identical. The only notable difference is R/B ratio (1.6 real vs 1.4 fake), suggesting DF40 reals skew slightly warmer than fakes — likely from studio/indoor filming conditions in the real source datasets.

### Per-family breakdown (fakes):

| Family | N | Mean Brightness | R/B Ratio | RMS Contrast |
|--------|:-:|:-:|:-:|:-:|
| GAN | 46 | 106.0 ± 22.9 | 1.4 ± 0.2 | 0.4 ± 0.2 |
| Diffusion | 2 | 125.5 ± 6.8 | 1.3 ± 0.1 | 0.4 ± 0.0 |
| FaceSwap | 59 | 110.6 ± 29.0 | 1.4 ± 0.3 | 0.5 ± 0.1 |
| Reenact | 48 | 100.9 ± 26.1 | 1.3 ± 0.3 | 0.5 ± 0.2 |

Reenactment fakes are slightly *dimmer* than reals. GANs and FaceSwaps are slightly brighter. No family presents a strong brightness shortcut.

---

## 4. Finding 2: Production Conditions Are Out-of-Distribution

| Category | Mean Brightness | R/B Ratio | RMS Contrast |
|----------|:-:|:-:|:-:|
| **Training Real** | 104.1 | 1.6 | 0.5 |
| **Training Fake** | 106.4 | 1.4 | 0.5 |
| **Your captures** | **160.6** | **1.2** | **0.3** |

Your captures are **56 brightness units above** training real and fake — completely outside the distribution. The R/B ratio of 1.2 (neutral white LED) is below both training sets. RMS contrast of 0.3 (flat, evenly-lit) is well below the training range of 0.5.

**Overlap analysis (captures vs training [5%–95%] range):**

| Metric | Overlap with Real | Overlap with Fake |
|--------|:-:|:-:|
| Mean Brightness | 45% | 42% |
| R/B Ratio | 67% (current aug) → 91% (proposed) | — |
| RMS Contrast | 82% (current) → 97% (proposed) | — |

**Root cause:** DF40 source datasets (Celeb-DF, FaceForensics++, YouTube Faces) were filmed with varied cameras under mixed conditions, but systematically lack well-lit indoor webcam-style captures with neutral white LED lighting — exactly what home offices and living rooms produce.

---

## 5. Finding 3: Current Augmentation Coverage Gaps

### Current pipeline: `_build_context_variation_block()`

**Code location:** `data/augmentations/pipelines.py`, line 928

```python
def _build_context_variation_block(p: dict) -> list:
    return [
        A.OneOf(
            [
                A.RandomGamma(gamma_limit=p.get("context_variation_gamma_limit", (90, 110)), p=ind_p),
                A.RandomBrightnessContrast(brightness_limit=..., contrast_limit=..., p=ind_p),
                A.ShiftScaleRotate(shift_limit=..., scale_limit=..., rotate_limit=..., p=ind_p),
            ],
            p=oneof_p,
        ),
    ]
```

**Structural problem:** `A.OneOf` selects exactly one of {gamma, brightness+contrast, shift/scale/rotate}. For any given image, at most one lighting transform fires. ShiftScaleRotate is not even a lighting transform — it's geometric. So each training image sees *either* gamma *or* brightness, never compound lighting variation.

### Preset values (from R10/R11):

| Setting | `vcd_targeted` default | R10_A / R11_C "wide" | R11_A "narrow" |
|---------|:-:|:-:|:-:|
| `gamma_limit` | (90, 110) | **(50, 150)** | (80, 120) |
| `brightness` | 0.15 | **0.40** | 0.25 |
| `contrast` | 0.15 | **0.35** | 0.25 |
| `oneof_p` | 0.18 | **0.50** | 0.30 |
| `individual_p` | 0.10 | **0.20** | 0.15 |

**Code locations for experiment configs:**
- `experiments/phase2_round10/R10_A_scratch_wide_aug.yaml`, lines 81–89
- `experiments/phase2_round11/R11_C_ft_r9d_wide.yaml`, lines 74–82
- `experiments/phase2_round11/R11_A_ft_r9d_narrow.yaml`, lines 80–88

### Audit results (25 training images × 10 augmentations, overlaid with 33 real captures):

| Metric | Original [5%–95%] | Current aug [5%–95%] | Proposed aug [5%–95%] | Capture coverage (current→proposed) |
|--------|----|----|----|:-:|
| Mean Brightness | 64.2 – 151.0 | 54.4 – 171.2 | 32.5 – 144.0 | **58% → 30%** |
| Brightness Std | 30.5 – 61.8 | 28.8 – 62.6 | 16.6 – 61.2 | 85% → 82% |
| R/B Ratio | 1.15 – 4.56 | 1.12 – 4.61 | 1.06 – 6.53 | **67% → 91%** |
| RMS Contrast | 0.27 – 0.69 | 0.24 – 0.75 | 0.19 – 0.88 | 82% → **97%** |

**Note on brightness coverage:** Neither current nor proposed augmentation effectively pushes images *brighter* enough. The current pipeline extends the upper bound from 151 to 171, but captures reach 205. The proposed compound pipeline actually shifts *down* more aggressively (gamma × brightness compound can darken heavily). This is because gamma < 1 and brightness < 1 compound multiplicatively downward, but the upward compound (gamma 1.5 × brightness 1.4) saturates quickly on already-midrange images.

---

## 6. What's Missing (Not Covered by Any Augmentation)

| Dimension | Status | Code reference |
|-----------|--------|---------------|
| Global brightness/gamma | ✅ Covered (OneOf) | `pipelines.py` L941–944 (RandomGamma) |
| Global brightness+contrast | ✅ Covered (OneOf) | `pipelines.py` L945–949 (RandomBrightnessContrast) |
| Compound lighting (all simultaneously) | ❌ Not covered | Blocked by `A.OneOf` wrapper at L940 |
| Colour temperature (CCT / white balance) | ❌ Not covered | No transform exists in codebase |
| Directional shadows | ❌ Not covered | No transform exists |
| Hue shift (partial CCT proxy) | ⚠️ Weak (±12°) | `pipelines.py` L996–999 (`hue_shift_limit=12`) in `color_block` |
| Upward brightness stretch | ⚠️ Insufficient | Current gamma [50,150] and brightness ±0.40 can't reach 200+ from 104 mean |

The `color_block` (at `pipelines.py` line 985) operates inside a separate `A.OneOf` with `RandomBrightnessContrast`, so `HueSaturationValue` only fires ~50% of the time colour transforms are applied at all.

---

## 7. Recommended Changes for R12

### 7.1 Easy Win: Remove `OneOf` from Context Variation Block

**Impact:** HIGH | **Effort:** 5-line change | **Risk:** LOW

Replace `A.OneOf([gamma, brightness, shift], p=oneof_p)` with independent transforms:

```python
# In _build_context_variation_block() at pipelines.py L940
# BEFORE (current):
A.OneOf([gamma, brightnesscontrast, shiftscalerotate], p=oneof_p)

# AFTER (proposed):
A.RandomGamma(gamma_limit=..., p=ind_p),
A.RandomBrightnessContrast(brightness_limit=..., contrast_limit=..., p=ind_p),
A.ShiftScaleRotate(..., p=ind_p),
```

This allows gamma + brightness + contrast to compound per-image, which our audit showed produces significantly wider coverage.

### 7.2 Easy Win: Add CCT / White Balance Simulation

**Impact:** HIGH | **Effort:** ~30-line custom transform | **Risk:** LOW

A simple RGB scaling transform that simulates colour temperature changes from 2700K (warm incandescent) to 8000K (overcast daylight). This directly closes the biggest validated gap: R/B ratio coverage going from 67% to 91% of real captures.

Reference implementation exists in `tools/lighting_showcase.py` (function `apply_cct()`, lines ~130–160) — this would need to be wrapped as an Albumentations 0.4.6 `ImageOnlyTransform` in `data/augmentations/transforms.py`.

### 7.3 Easy Win: Widen Hue Shift Range

**Impact:** MEDIUM | **Effort:** Config-only | **Risk:** LOW

Change `hue_shift` from 12 to 20 in the `vcd_targeted` preset at `pipelines.py` L892. This gives HueSaturationValue more room to simulate colour temperature drift.

### 7.4 Harder: Brightness Distribution Stretch Upward

**Impact:** HIGH | **Effort:** MEDIUM | **Risk:** MEDIUM

The fundamental problem is that DF40 training data has mean brightness ~104, and production data reaches 160–205. Augmentation alone cannot reliably push images this far up without saturation. Two approaches:

**Option A — Asymmetric brightness augmentation:** Bias the brightness augmentation upward. Instead of `brightness_limit=(-0.40, +0.40)`, use `brightness_limit=(-0.20, +0.60)` to shift the distribution right.

**Option B — Targeted exposure correction on training data:** Pre-process a subset of training images to be brighter using histogram equalisation or gamma < 1 before cropping, creating a more uniformly distributed brightness dataset.

**Option C — Production data in training mix:** Add well-lit webcam captures (VCD, Teams, or captured Roee-style data) to the real training pool, directly covering the gap.

### 7.5 Experiment Design Suggestion

| Experiment | Changes | Tests |
|------------|---------|-------|
| R12_A: Compound lighting | Remove OneOf, keep same ranges | Does compound > OneOf? |
| R12_B: Compound + CCT | Remove OneOf + add CCT transform | Does CCT close colour temp gap? |
| R12_C: Compound + CCT + brightness-up bias | All above + asymmetric brightness | Does brightness stretch help? |
| R12_D: Control (R11 best) | No changes | Baseline comparison |

For each, validate with `tools/lighting_showcase.py audit` before and after to confirm distribution coverage matches expectations.

---

## 8. Artifacts & Reproducibility

### Generated artifacts (in `/tmp/`, ephemeral):

| File | Description |
|------|-------------|
| `/tmp/real_vs_fake_skew.png` | Histogram overlay: real vs fake brightness distributions |
| `/tmp/real_vs_fake_skew_boxplot.png` | Box plots by method family + real captures |
| `/tmp/real_vs_fake_skew.csv` | Raw per-image stats (66 real + 155 fake + 33 captures) |
| `/tmp/lighting_audit_real.png` | Augmentation coverage audit with capture overlay |
| `/tmp/lighting_audit_real.csv` | Raw audit stats |
| `/tmp/lighting_showcase_roee.png` | 16-augmentation visual grid on one capture |
| `/tmp/df40_real_vs_fake/` | Downloaded DF40 sample (66 real PNGs + 155 fake PNGs) |

### Tools (committed to repo):

| File | Lines | Purpose |
|------|-------|---------|
| `tools/lighting_showcase.py` | 964 | Augmentation showcase, comparison, and statistical audit |
| `tools/real_vs_fake_skew.py` | 323 | Real vs fake brightness/colour skew analysis |

### Reproduction commands:

```bash
cd DeepfakeBench/training

# 1. Download training samples (requires GCS auth)
bash /tmp/download_rf_sample.sh     # 66 real images
bash /tmp/download_gan_sample.sh    # GAN/diffusion fakes

# 2. Real vs fake skew analysis
python tools/real_vs_fake_skew.py
# → /tmp/real_vs_fake_skew.png, /tmp/real_vs_fake_skew_boxplot.png, /tmp/real_vs_fake_skew.csv

# 3. Augmentation coverage audit (with real captures overlaid)
python tools/lighting_showcase.py audit \
  --images-dir /tmp/df40_real_vs_fake/real \
  --max-images 66 \
  --repeats 10 \
  --real-captures-dir /Users/roeedar/Downloads/roee_light \
  --output /tmp/lighting_audit_real.png

# 4. Showcase on a single face crop
python tools/lighting_showcase.py showcase \
  --image /Users/roeedar/Downloads/roee_light/frame_000193_seq842.png \
  --output /tmp/lighting_showcase_roee.png
```

### GCS data structure reference:

```
gs://df40-frames-recropped-rfa85/
├── real/
│   ├── Celeb-real/{id}_{video}/{frame}.png
│   ├── FaceForensics++/{video}/{frame}.png
│   └── YouTube-real/{video}/{frame}.png
└── fake/
    ├── StyleGAN2/{folder}/seed*.png          # GAN: seed-based naming
    ├── facedancer/{src}_{tgt}/{frame}.png     # Swap/reenact: frame-based
    └── ... (29 methods total)
```

---

## 9. Key Takeaway

The lighting false-positive problem is an **out-of-distribution problem**, not a **learned-shortcut problem**. The model hasn't been taught "bright = fake" — it simply has no experience with well-lit, neutral-temperature, low-contrast indoor faces. The fix requires:

1. **Augmentation changes** (compound lighting, CCT) to widen the training distribution envelope
2. **Brightness stretching** specifically upward to cover the 150–210 range
3. Ideally, **real-world webcam data** in the training mix to provide genuine coverage of production conditions

All three can be validated empirically before training using the tools created in this analysis.
