# Sharpness Shortcut — Verification Plan

**Date:** 2026-02-15  
**Context:** All models fail on WMA enhanced face-swaps (best: 55.7%, deployed R25_F1: 21%)
AND on DF40 real data (89% false positive rate). The meta-analysis in §15 of
PROGRESS_REPORT.md identified a **sharpness-based shortcut** as the likely root cause.

**Goal:** Before investing in new training runs, we need to **verify** the sharpness
shortcut theory with targeted experiments. Each test below either confirms or refutes
a specific aspect of the theory.

---

## Why Verify First?

Round 3 already taught us a painful lesson: "quality-robust" augmentation (designed to
fix exactly this problem) made OOD performance **worse**, not better. If we rush to train
again without understanding the failure mechanism precisely, we risk repeating that mistake.

The two alarming findings:

1. **WMA enhanced fakes → 21% detection (R25_F1).** 1,202 GFPGAN-enhanced face-swaps
   from DeepLiveCam. The model confidently says "real" — 65% of images get <0.3 fake prob.
2. **DF40 real faces → 89% false positive rate.** The model says "fake" with median
   probability 1.000 for DF40 real faces (sharpness 15.4). This was hidden because DF40
   reals were never evaluated in isolation before.

Both failures correlate with **image sharpness** (Laplacian variance):
- Low sharpness (<40) → model says "real" regardless of ground truth
- High sharpness (50–130) → model says "fake" regardless of ground truth

---

## Verification Experiments

### V5 — Feature-Prediction Correlation Matrix ⏱️ ~2 min
**File:** `V5_correlation_matrix.py`  
**Data needed:** `meta_analysis_properties.csv` (already on disk)

**Question:** Is sharpness really the #1 predictor of model output, or is it one of many?

**Method:** Compute Pearson & Spearman correlations between each of the 36 image
properties and `model_fake_prob`. Rank by absolute correlation strength.

**Expected result if theory is correct:** `sharpness_laplacian_var`, `edge_density`,
`sharpness_tenengrad`, and `freq_high_energy_ratio` should dominate the top of the
ranking (|r| > 0.3), well above color/noise features.

**Why this matters:** If sharpness is only weakly correlated and some other feature
dominates, our fix strategy changes entirely.

---

### V4 — CLIP Feature Probe: Does CLIP Encode Sharpness? ⏱️ ~5 min
**File:** `V4_clip_feature_probe.py`  
**Data needed:** `meta_analysis_features.npz` + `meta_analysis_properties.csv` (on disk)

**Question:** Does the sharpness shortcut live in CLIP's frozen features, or did the
SVD residual layers learn it?

**Method:** Train a linear regression predicting `sharpness_laplacian_var` from the
512-dim CLIP features. Report R². Also train a logistic regression predicting
`model_fake_prob > 0.5` from the same features (as a sanity check that the features
carry the model's decision signal).

**Expected results:**
- If CLIP R² for sharpness is HIGH (>0.5): CLIP itself encodes quality → the fix
  must happen at the data/augmentation level (can't change CLIP's frozen features).
- If CLIP R² for sharpness is LOW (<0.2): The SVD layers learned sharpness on their
  own → architectural fixes (orthogonalization against quality) might work.

**Why this matters:** Determines whether we need data-level or architecture-level fixes.

---

### V6 — Cross-Model Sharpness Consistency ⏱️ ~2 min
**File:** `V6_cross_model_consistency.py`  
**Data needed:** `enhancer_eval_per_image.csv` + `meta_analysis_properties.csv` (on disk)

**Question:** Do all 5 models use the sharpness shortcut, or is it model-specific?

**Method:** Join WMA per-image predictions (5 models) with WMA image properties (sharpness).
Compute sharpness↔probability correlation for each model separately.

**Expected results:**
- If ALL 5 models show strong correlation: Systemic data issue — the training
  distribution itself teaches this shortcut regardless of architecture/loss.
- If B16_old (55.7% accuracy) shows weaker correlation: It partially learned real
  forgery features, suggesting the shortcut is not inevitable.

**Why this matters:** If the shortcut is universal across training configs, only
data changes can fix it. If some configs avoid it, we have a training recipe clue.

---

### V1 — Sharpness Causality Test (Intervention) ⏱️ ~30 min
**File:** `V1_sharpness_causality.py`  
**Data needed:** WMA images (local), R25_F1 checkpoint (GCS download)

**Question:** Does artificially changing sharpness causally change the model's decision?

**Method:**
1. Take WMA images → **sharpen** (unsharp mask, target Laplacian var ~60–80) → re-infer
2. Take WMA images → leave original → re-infer (baseline)
3. Optionally: Take DeepLive training fakes → **blur** (target sharpness ~20) → re-infer

**Expected results:**
- Sharpening WMA images should dramatically increase fake detection (21% → ???)
- Blurring training fakes should dramatically decrease detection
- This is the strongest test: it's a causal intervention, not correlation

**Why this matters:** This is the "smoking gun" — if we can flip predictions by changing
sharpness alone, the shortcut is proven beyond doubt.

---

### V2 — Resolution Isolation Test ⏱️ ~30 min
**File:** `V2_resolution_isolation.py`  
**Data needed:** DL quality_enhancement images (GCS), R25_F1 checkpoint

**Question:** How much of the WMA failure comes from downscaling vs GFPGAN smoothing?

**Method:** Take the 150 DL `quality_enhancement_fake` images (98% acc, 224×224) →
upsample to 342×435 → downsample back to 224×224 → re-infer. Compare accuracy to
original 98%.

**Expected results:**
- If accuracy drops significantly (e.g., to <70%): Resolution mismatch is a major
  contributor — training needs multi-resolution augmentation.
- If accuracy stays high (>90%): GFPGAN + different crop pipeline is the primary
  driver — resolution is not the issue.

---

## Running the Experiments

```bash
cd /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/analysis_results/verification

# Easy ones first (no model inference needed):
python V5_correlation_matrix.py      # ~2 min, existing CSV
python V4_clip_feature_probe.py      # ~5 min, existing CSV + NPZ
python V6_cross_model_consistency.py # ~2 min, existing CSVs

# Harder ones (need model checkpoint + images):
python V1_sharpness_causality.py     # ~30 min, needs WMA images + checkpoint
python V2_resolution_isolation.py    # ~30 min, needs GCS images + checkpoint
```

## After Verification

Once we have results from V1–V6, we'll know:
1. Is sharpness the dominant shortcut? (V5)
2. Where does it live — CLIP or SVD? (V4)
3. Is it universal across models? (V6)
4. Is it causal? (V1)
5. How much is resolution vs GFPGAN? (V2)

This tells us exactly what kind of training fix to design — and equally importantly,
what fixes will NOT work (like the R3 quality-robust augmentation that backfired).
