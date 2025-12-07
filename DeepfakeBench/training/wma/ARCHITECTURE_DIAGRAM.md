# Four-Model Ensemble: Architecture & Data Flow

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      WMA Production Server                       │
│                         (server.py)                              │
└─────────────────────────────────────────────────────────────────┘
                               ▼
                     Incoming Participant
                    (1-32 frame images)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              FourModelAPIClient.process_frames()                 │
│                  (test_four_model_fusion.py)                     │
└─────────────────────────────────────────────────────────────────┘
                               ▼
        ┌──────────────┬──────────────┬──────────────┬──────────────┐
        ▼              ▼              ▼              ▼
  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Model 1  │  │ Model 2  │  │ Model 3  │  │ Model 4  │
  │9rfa62j1  │  │1mjgo9w1  │  │dfsesrgu  │  │4vtny88m  │
  │  (EFS+   │  │(FaceSwap)│  │(Reenact) │  │(Noisy-OR)│
  │Platform) │  │          │  │          │  │          │
  │ IP1:8999 │  │ IP2:8999 │  │ IP3:8999 │  │ IP4:8999 │
  └──────────┘  └──────────┘  └──────────┘  └──────────┘
        │              │              │              │
        │ [p1_f1, ... │ [p2_f1, ... │ [p3_f1, ... │ [p4_f1, ...
        │  p1_f27]    │  p2_f27]    │  p3_f27]    │  p4_f27]
        │              │              │              │
        └──────────────┴──────────────┴──────────────┴──────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│           Aggregate per model (model-specific method)            │
│   Model 1: topk4 → 0.9234     Model 2: softmax_b5 → 0.9156     │
│   Model 3: topk4 → 0.9401     Model 4: topk4 → 0.9287          │
└─────────────────────────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              Apply Isotonic Calibration (per model)              │
│   Model 1: 0.9234 → 0.9245    Model 2: 0.9156 → 0.9134         │
│   Model 3: 0.9401 → 0.9423    Model 4: 0.9287 → 0.9301         │
└─────────────────────────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Noisy-OR Fusion                               │
│   fusion = 1 - ∏(1 - p_i)                                       │
│   fusion = 1 - (1-0.9245)(1-0.9134)(1-0.9423)(1-0.9301)        │
│   fusion = 1 - (0.0755)(0.0866)(0.0577)(0.0699)                │
│   fusion = 1 - 0.0000257 = 0.9999743                           │
└─────────────────────────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Three-Way Classification                        │
│   If fusion < 0.996700  →  REAL                                 │
│   If fusion > 0.998248  →  FAKE                                 │
│   Otherwise             →  UNCERTAIN                            │
│                                                                  │
│   Result: fusion = 0.9999743 > 0.998248                         │
│   ✓ Verdict: FAKE, Confidence: High                             │
└─────────────────────────────────────────────────────────────────┘
                               ▼
                      Return to WMA Server
                     (Generate banner action)
```

## Data Flow Example: Single Participant

### Input
```python
participant_name = "John Doe"
frames = [frame1.jpg, frame2.jpg, ..., frame27.jpg]  # 27 frames @ 2fps
```

### Step 1: Parallel API Calls (async)
```python
# All 4 calls happen simultaneously
predictions = {
    '9rfa62j1': [0.92, 0.89, 0.94, ..., 0.91],  # 27 per-frame probs
    '1mjgo9w1': [0.88, 0.87, 0.93, ..., 0.90],  # 27 per-frame probs
    'dfsesrgu': [0.91, 0.93, 0.96, ..., 0.92],  # 27 per-frame probs
    '4vtny88m': [0.90, 0.88, 0.95, ..., 0.89],  # 27 per-frame probs
}
```

### Step 2: Aggregate (model-specific)
```python
# Model 1: topk4 (top 4 highest values)
model1_frames = [0.92, 0.89, 0.94, 0.87, 0.96, 0.88, ...]  # 27 values
sorted_desc = [0.96, 0.94, 0.92, 0.91, ...]               # Sort descending
top4 = [0.96, 0.94, 0.92, 0.91]                            # Select top 4
aggregated1 = mean([0.96, 0.94, 0.92, 0.91]) = 0.9325

# Model 2: softmax_b5 (exponentially-weighted average)
model2_frames = [0.88, 0.87, 0.93, ...]                    # 27 values
# Higher values get exponentially more weight with β=5
aggregated2 = softmax_pool(model2_frames, beta=5.0) = 0.9156

# Models 3 & 4: topk4 (same as Model 1)
aggregated3 = 0.9401
aggregated4 = 0.9287
```

### Step 3: Calibrate (isotonic regression)
```python
# Apply pre-fitted calibration per model
calibrated = {
    '9rfa62j1': calibrator1.transform([0.9325]) = 0.9345,
    '1mjgo9w1': calibrator2.transform([0.9156]) = 0.9134,
    'dfsesrgu': calibrator3.transform([0.9401]) = 0.9423,
    '4vtny88m': calibrator4.transform([0.9287]) = 0.9301,
}
```

### Step 4: Fuse (Noisy-OR)
```python
# Assumes independent errors
fusion = 1 - (1 - 0.9345) * (1 - 0.9134) * (1 - 0.9423) * (1 - 0.9301)
fusion = 1 - (0.0655) * (0.0866) * (0.0577) * (0.0699)
fusion = 1 - 0.0000217
fusion = 0.9999783
```

### Step 5: Classify
```python
T_low = 0.996700
T_high = 0.998248

if fusion < T_low:
    verdict = "REAL"
elif fusion > T_high:
    verdict = "FAKE"     # ← This case!
else:
    verdict = "UNCERTAIN"

# Confidence calculation (for FAKE)
range_span = 1.0 - T_high = 0.001752
relative = fusion - T_high = 0.9999783 - 0.998248 = 0.0017303
relative / range_span = 0.9876

# 0.9876 > 0.67 → High confidence

result = {
    'verdict': 'FAKE',
    'confidence': 'High',
    'fusion_score': 0.9999783
}
```

## Comparison: Single Model vs Four Models

### Single Model System (Current)
```
27 frames → Single Model → [p1, p2, ..., p27]
                           ↓
                    Simple Average
                           ↓
                     mean = 0.89
                           ↓
            threshold = 0.75 (fixed)
                           ↓
               0.89 > 0.75 → FAKE
```

**Issues:**
- Single point of failure
- No specialization per deepfake type
- Binary threshold (no UNCERTAIN)
- Simple averaging (no frame selection)

### Four Model System (Target)
```
27 frames → Model 1 (EFS specialist)    → topk4 → calibrate → 0.9345
         ↘ Model 2 (FaceSwap specialist) → softmax → calibrate → 0.9134
         ↘ Model 3 (Reenact specialist)  → topk4 → calibrate → 0.9423
         ↘ Model 4 (General specialist)  → topk4 → calibrate → 0.9301
                           ↓
                    Noisy-OR Fusion
                           ↓
                   fusion = 0.9999783
                           ↓
             Three-way thresholds
              T_low = 0.9967
              T_high = 0.9982
                           ↓
         fusion > T_high → FAKE (High confidence)
```

**Advantages:**
- ✅ 4 specialist models (ensemble diversity)
- ✅ Model-specific aggregation (topk4 vs softmax_b5)
- ✅ Calibrated probabilities (isotonic regression)
- ✅ Statistically sound fusion (Noisy-OR)
- ✅ Three-way classification (REAL/UNCERTAIN/FAKE)
- ✅ Confidence levels (High/Medium/Low)

## Key Technical Concepts

### 1. Why topk4 for 27 frames?
```
27 frames total
Top 4 frames = ~15% of frames
```

**Intuition:** Select the "most suspicious" frames. If a video contains 27 frames but only 4 show clear manipulation, those 4 frames should drive the decision.

**Training validated this:** topk4 > mean for Models 1, 3, 4.

### 2. Why softmax_b5 for Model 2?
```python
# Regular average: all frames equal weight
mean([0.9, 0.5, 0.8]) = 0.733

# Softmax (β=5): higher values get more weight
softmax_pool([0.9, 0.5, 0.8], β=5) = 0.847

# The suspicious frames (0.9, 0.8) dominate
```

**Intuition:** FaceSwap artifacts may be subtle but consistent. Exponential weighting captures this better than topk4.

### 3. Why Noisy-OR fusion?
```python
# If models make independent errors:
# P(all miss the fake) = P(miss1) × P(miss2) × P(miss3) × P(miss4)
# P(at least one catches it) = 1 - P(all miss)

# Example: Each model 90% accurate
P(all miss) = 0.1 × 0.1 × 0.1 × 0.1 = 0.0001
P(detect) = 1 - 0.0001 = 0.9999 ← Very high!
```

**Why it works:** Your 4 models are trained on different data clusters (EFS, FaceSwap, Reenact, general). Their errors are largely independent.

### 4. Why such high thresholds?
```python
# Given 4 models with scores ~0.90-0.95:
fusion_score = 0.9999+

# Need thresholds that match this distribution:
T_low = 0.9967   # Below this = confident REAL
T_high = 0.9982  # Above this = confident FAKE
# Between = UNCERTAIN (need more data)
```

**Training validated:** These thresholds achieve 90.52% TPR, 4.35% FPR on validation set.

## Performance Expectations

### Latency
```
Per-participant processing:
├── API calls (parallel): 2-4 seconds
├── Aggregation: < 10 milliseconds
├── Calibration: < 1 millisecond
├── Fusion: < 1 millisecond
└── Classification: < 1 millisecond

Total: ~2-4 seconds for 27 frames
```

**Optimization tips:**
- Reduce frames: 15 instead of 27 (faster, slightly less accurate)
- Batch multiple participants (if your server supports it)
- Cache calibrators in memory (don't reload each time)

### Accuracy (Expected)
Based on training results:

| Metric | Single Model | Four Models | Improvement |
|--------|--------------|-------------|-------------|
| TPR (catch fakes) | ~85% | 90.52% | +5.5% |
| TNR (catch reals) | ~90% | 93.81% | +3.8% |
| FPR (false alarms) | ~10% | 4.35% | -56% |
| Uncertain rate | 0% | 2.18% | New feature |

## Debugging Guide

### Issue: Fusion scores always near 1.0

**Check:**
```python
# Are calibrated scores high?
print(calibrated_scores)
# {'9rfa62j1': 0.95, '1mjgo9w1': 0.92, ...}

# If all > 0.90, fusion will be 0.999+
# This is CORRECT if inputs are high!

# Verify inputs are reasonable:
for model_id, probs in predictions.items():
    print(f"{model_id}: mean={np.mean(probs):.3f}")
```

**Expected:** 
- Real videos: mean ~0.1-0.3 → fusion ~0.5-0.8
- Fake videos: mean ~0.85-0.95 → fusion ~0.999

### Issue: All verdicts are UNCERTAIN

**Check thresholds:**
```python
# Are you seeing scores in the narrow band?
# T_low = 0.9967, T_high = 0.9982
# Band width = 0.0015 (only 0.15% of range!)

# Check fusion score distribution:
print(f"Fusion score: {fusion_score:.6f}")
print(f"T_low: {T_low:.6f}")
print(f"T_high: {T_high:.6f}")
```

**Fix:** If too many UNCERTAINs, you may need to widen the band or collect more training data.

### Issue: Different results than training

**Common causes:**
1. ❌ Wrong aggregator (using mean instead of topk4)
2. ❌ Calibrators not loaded (identity function)
3. ❌ Different frame counts (15 vs 27)
4. ❌ Model servers returning different probabilities

**Verify each step:**
```python
# Log everything:
logger.info(f"Raw predictions: {predictions}")
logger.info(f"Aggregated: {aggregated}")
logger.info(f"Calibrated: {calibrated}")
logger.info(f"Fused: {fusion_score}")
logger.info(f"Verdict: {verdict}")
```

## Summary

This four-model ensemble system:

1. **Calls 4 specialist models** in parallel
2. **Aggregates per model** using validated methods (topk4 or softmax_b5)
3. **Calibrates** with isotonic regression (fitted on OOF data)
4. **Fuses** with Noisy-OR (assumes independent errors)
5. **Classifies** with three-way thresholds (validated on test set)

**Result:** 90% TPR, 4% FPR, with confidence levels and UNCERTAIN category.

**Next step:** Run the standalone test! 🚀

```bash
python test_four_model_fusion.py --mock
```
