# B16_LAION + ArcFace Feasibility Investigation

## Executive Summary

**Current Status: B16_LAION + ArcFace is NOT working as configured**

After implementing full SVD coverage (q, k, v, out_proj = 48 total layers) and running controlled experiments, the results clearly show:

| Configuration | Best val_holdout AUC | Training Behavior |
|---------------|---------------------|-------------------|
| B16_LAION + ArcFace (full SVD) | **~0.65** | Collapse at epoch ~46 |
| B16_LAION + NO ArcFace (full SVD) | **~0.78** | Stable, continues improving |
| L14_OpenAI + ArcFace | **~0.85+** | Stable (reference success) |

The ~13% AUC gap is significant and the collapse behavior is catastrophic.

---

## Detailed Log Analysis

### B16_LAION + ArcFace (s18_full_svd)

**Early Training (Steps 0-2000):**
- Step 500: val_holdout AUC = 0.5551
- Step 1000: val_holdout AUC = 0.6318
- Step 1500: val_holdout AUC = 0.6503 ← **PEAK**
- Step 2000: val_holdout AUC = 0.6503

**Late Training (Steps 2500+):**
- Step 2500: val_holdout AUC = 0.6225 (degrading)
- Step 3000: val_holdout AUC = 0.6214 (still degrading)

**Collapse Indicators Starting at Epoch ~46 (Step ~3800):**
```
⚠️ COLLAPSE WARNING: Logit std=0.004364, range=0.015625
Predictions: pred_fake=0, pred_real=48  ← All predicting REAL
Fake frames: 0/24 predicted correctly
Feature norm mean: 21.0000  ← HIGH norm
Loss climbing: 1.1402 → 1.2205 → 1.1507
```

**Critical Observation:** The model enters a mode where logit variance approaches zero and all predictions flip to one class.

### B16_LAION + NO ArcFace (full SVD)

**Early Training (Steps 0-2000):**
- Step 500: val_holdout AUC = 0.6515
- Step 1000: val_holdout AUC = 0.7154
- Step 1500: val_holdout AUC = 0.7619
- Step 2000: val_holdout AUC = 0.7821

**Late Training (Steps 2500+):**
- Step 2500: val_holdout AUC = 0.7803 (stable)
- Step 3000: val_holdout AUC = 0.7838 (slight improvement)
- Step 4000: val_holdout AUC = 0.7828 (stable plateau)

**Batch Diagnostics (Healthy):**
```
Logit diff stats (fake-real): mean=-0.1890, std=0.6475  ← Good spread
Predictions: pred_fake=28, pred_real=20  ← Balanced predictions
Real frames: 15/24 predicted correctly
Fake frames: 19/24 predicted correctly
Feature norm mean: 17.2969  ← Moderate norm
Loss stable: 0.5863 → 0.6075 → 0.5619
```

---

## Root Cause Analysis

Based on the investigation report and log analysis, the following factors contribute to the failure:

### Primary Cause: ArcFace + Effort Parameter Interaction

1. **Feature Norm Inflation**: ArcFace requires normalized features, but with Effort's constrained optimization:
   - With ArcFace: Feature norm mean = **21.0** (inflated)
   - Without ArcFace: Feature norm mean = **17.3** (moderate)
   
2. **Logit Collapse Mechanism**:
   - ArcFace's angular margin creates sharp decision boundaries
   - Effort's orthogonality constraints fight against the required weight updates
   - Result: Model finds a degenerate solution where logit variance → 0

3. **Loss Dynamics**:
   - With ArcFace: Training loss climbs (0.7 → 4.6 → constant at ~1.0-1.2 during collapse)
   - Without ArcFace: Training loss stable around 0.5-0.7

### Contributing Factors

1. **Hidden Size Mismatch with L14**:
   - L14_OpenAI: hidden_size = **1024**
   - B16_LAION: hidden_size = **512** (output), embed_dim = **768** (internal)
   - ArcFace's scale `s` may need different tuning for different embedding dimensions

2. **SVD Rank Configuration**:
   - Currently using rank=767 (embed_dim - 1) for full SVD coverage
   - This gives k=1 trainable singular direction per layer
   - ~73,776 trainable backbone params total

3. **Weight Decay Conflict** (Potential):
   - If weight decay is applied to SVD parameters (U, S, V), it fights orthogonality constraints
   - Need to verify SVD params have `weight_decay=0`

---

## Priority Checklist for Investigation

### ✅ VERIFIED (Not The Issue)

#### ~~1. Verify SVD Parameters Have weight_decay=0~~
**Status:** ✅ CONFIRMED - Already correctly implemented
**Evidence from logs:**
```
2026-01-13 11:03:43   - SVD residual params: 73,776 (weight_decay=0.0)
2026-01-13 11:03:43   - Other params: 1,024 (weight_decay=0.05)
```
The SVD params are correctly separated into their own param group with `weight_decay=0.0`.

---

### 🔴 HIGH PRIORITY (Try First)

#### 1. ArcFace Scale (s) Reduction for B16
**Why:** Smaller embedding dimension (512 output) may need smaller scale
**Current:** s_start=10, s_end=18
**Try:**
- s_start=5, s_end=10 (HIGH PRIORITY)
- s_start=3, s_end=8
**Rationale:** Scale controls sharpness of angular margin; smaller embeddings may saturate faster
**Experiment Config:** See `Experiment A` below

#### 2. Feature Normalization Experiment
**Why:** The feature norm difference (21 vs 17) suggests ArcFace is pushing norms up
**Action:** Add explicit L2 normalization BEFORE ArcFace head:
```python
features = F.normalize(features, p=2, dim=1)  # Force unit norm
logits = arcface_head(features, labels)
```

### 🟡 MEDIUM PRIORITY (Try Second)

#### 4. ArcFace Margin (m) Reduction
**Why:** Angular margin may be too aggressive for B16's representation space
**Current:** m=0.28
**Try:**
- m=0.15
- m=0.10
- m=0.0 (equivalent to CosFace with m=0)

#### 5. Hybrid Loss Strategy: Start BCE, Switch to ArcFace
**Why:** Let features organize first, then apply angular margin
**Implementation:**
```yaml
loss_transition:
  warmup_with_bce: 2000  # Train with BCE first
  transition_steps: 500   # Gradual transition to ArcFace
```

#### 6. Log Separate Loss Components
**Why:** Need to see if orthogonality loss or singular value constraint is exploding
**Action:** Add logging for:
- `cls_loss` (classification component)
- `orth_loss` (orthogonality constraint)
- `sv_loss` (singular value constraint)
- Gradient norms for SVD params vs head params

### 🟢 LOWER PRIORITY (Try If Above Fail)

#### 7. Alternative Angular Losses
**Options:**
- **CosFace** (additive margin): More stable, less aggressive
- **AdaCos** (adaptive scale): Auto-tunes scale parameter
- **SphereFace** (multiplicative margin): Different geometry

#### 8. Reduce ArcFace Annealing Speed
**Current:** anneal_steps=1000 (aligned with warmup)
**Try:** anneal_steps=3000 (slower introduction of full margin)

#### 9. Two-Phase Training
**Strategy:**
1. Phase 1: Train without ArcFace until convergence (~0.78 AUC)
2. Phase 2: Fine-tune with ArcFace from the converged checkpoint

#### 10. Feature Space Analysis
**Action:** Save feature embeddings at different training stages and analyze:
- Intra-class compactness
- Inter-class separation
- Angular distribution of features

---

## Overnight Experiment Suite (8 Experiments)

All configs located in: `experiments/arcface_scale_ablation/B16_LAION_optimized/`

### Batch 1: Scale Experiments
| Experiment | Config File | Key Change | Hypothesis |
|------------|-------------|------------|------------|
| A | `vit_B16_laion_s10_full_svd.yaml` | s_end=10 | Lower scale for smaller embeddings |
| B | `vit_B16_laion_s6_full_svd.yaml` | s_end=6 | Very low scale (CosFace-like behavior) |

### Batch 2: Margin Experiments
| Experiment | Config File | Key Change | Hypothesis |
|------------|-------------|------------|------------|
| C | `vit_B16_laion_m15_full_svd.yaml` | m=0.15 | Less aggressive angular margin |
| D | `vit_B16_laion_cosface_style_full_svd.yaml` | m=0 | No margin (is margin the problem?) |

### Batch 3: Combined Reductions
| Experiment | Config File | Key Change | Hypothesis |
|------------|-------------|------------|------------|
| E | `vit_B16_laion_s10_m15_full_svd.yaml` | s_end=10, m=0.15 | Maximum gentleness |

### Batch 4: Structural Changes
| Experiment | Config File | Key Change | Hypothesis |
|------------|-------------|------------|------------|
| F | `vit_B16_laion_feat_norm_full_svd.yaml` | L2 normalize features | High norms cause instability |
| G | `vit_B16_laion_slow_anneal_full_svd.yaml` | anneal_steps=3000 | Too fast margin introduction |
| H | `vit_B16_laion_low_lambda_full_svd.yaml` | lambda_reg=0.1 | Constraints dominating loss |

### Code Changes Required
- ✅ `normalize_features_before_head` flag added to `effort_detector.py`

---

## Quick Experiments Config Templates

### Experiment A: Low-Scale ArcFace
```yaml
# experiments/arcface_scale_ablation/B16_LAION_optimized/vit_B16_laion_s10_full_svd.yaml
arcface:
  enabled: true
  s_start: 5.0
  s_end: 10.0  # Reduced from 18
  m: 0.28
  anneal_steps: 1000
```

### Experiment B: Low-Margin ArcFace
```yaml
# experiments/arcface_scale_ablation/B16_LAION_optimized/vit_B16_laion_m15_full_svd.yaml
arcface:
  enabled: true
  s_start: 10.0
  s_end: 18.0
  m: 0.15  # Reduced from 0.28
  anneal_steps: 1000
```

### Experiment C: Feature Normalization + ArcFace
```yaml
# experiments/arcface_scale_ablation/B16_LAION_optimized/vit_B16_laion_feat_norm_full_svd.yaml
normalize_features_before_head: true  # L2 normalize to unit length
arcface:
  enabled: true
  s_start: 10.0
  s_end: 18.0
  m: 0.28
```

---

## Key Metrics to Track

For each experiment, log and compare:

| Metric | Healthy Range | Collapse Indicator |
|--------|--------------|-------------------|
| Logit std | > 0.2 | < 0.05 |
| Feature norm | 15-20 | > 25 or < 5 |
| Loss | Decreasing or stable | Monotonically increasing |
| Prediction balance | 40-60% per class | 90%+ one class |
| val_holdout AUC | Improving | Degrading after peak |

**New diagnostic metrics (already logged):**
- `cls_loss` - Classification loss before regularization
- `reg_loss` - Regularization term (lambda_reg * avg(orth + keepsv))
- `orthogonal_loss` - Sum of orthogonality losses
- `keepsv_loss` - Sum of singular value preservation losses
- `reg_cls_ratio` - **KEY**: If this >> 1.0, regularization is dominating

---

## Conclusion

The B16_LAION + ArcFace combination fails due to a fundamental incompatibility between:
1. ArcFace's angular margin requirements
2. Effort's SVD orthogonality constraints
3. The smaller embedding dimension of B16

The **first experiments to run** should be:
1. ✅ Verify weight_decay=0 for SVD parameters
2. 🔬 Low-scale ArcFace (s_end=10)
3. 🔬 Add explicit feature normalization before ArcFace

If none of these work, the pragmatic path is to **accept that B16_LAION works best without ArcFace** (achieving 0.78 AUC), and focus ArcFace experiments on L14_OpenAI which has demonstrated success.
