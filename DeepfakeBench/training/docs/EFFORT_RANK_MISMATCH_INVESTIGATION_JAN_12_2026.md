# Effort Method Rank Mismatch Investigation

**Date**: January 12, 2026  
**Branch**: `refactor-training`  
**Status**: Fix implemented, awaiting validation results

---

## Executive Summary

We discovered the **root cause** of the training collapse observed in all ArcFace scale ablation experiments: a **critical mismatch between the configured SVD rank and the actual attention layer dimensions**. 

The ViT-B-16 LAION backbone uses **768-dimensional** attention output projections, but our config specified `rank=511` (intended for 512-dim layers). This resulted in **k=257 trainable singular directions per layer** instead of the Effort paper's recommended **k=1**, causing:

1. **257× more trainable parameters** than intended in SVD layers
2. **Regularization loss growing unbounded** (orthogonality constraints harder to maintain)
3. **Training loss growing 8× (2.8 → 22+)** while validation loss stayed stable
4. **Model collapse at step ~4600** regardless of ArcFace scale

This report documents the investigation journey, the discovery, and the implemented fixes.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Investigation Timeline](#2-investigation-timeline)
3. [The Discovery: Rank Mismatch](#3-the-discovery-rank-mismatch)
4. [Root Cause Analysis](#4-root-cause-analysis)
5. [Implemented Fixes](#5-implemented-fixes)
6. [Expected Outcomes](#6-expected-outcomes)
7. [Validation Plan](#7-validation-plan)
8. [Technical Details](#8-technical-details)
9. [Lessons Learned](#9-lessons-learned)

---

## 1. Problem Statement

### Observed Symptoms

All four ArcFace scale ablation experiments (s_end = 12, 15, 18, 20) exhibited identical failure patterns:

| Symptom | Observation |
|---------|-------------|
| **Collapse timing** | All collapsed at step ~4600-4800 (±3% variance) |
| **Training loss** | Grew from 2.8 → 22+ (8× increase) over training |
| **Validation loss** | Stayed relatively stable (~0.68-0.70) |
| **Collapse signature** | Logit std < 0.01, model outputs near-constant predictions |
| **Scale independence** | s_end had NO effect on collapse timing |

### The Anomaly That Led to Discovery

The most telling anomaly was the **training loss growth pattern**:

```
Epoch 1:  loss ≈ 2.8
Epoch 20: loss ≈ 6.4
Epoch 40: loss ≈ 12.3
Epoch 57: loss ≈ 22+ (collapse)
```

In standard classification training, training loss should **decrease** as the model learns. A steadily **increasing** training loss while validation loss remains stable indicates a **non-classification loss component is growing unbounded**.

---

## 2. Investigation Timeline

### January 9, 2026: Identity Leakage Fix
- **Discovery**: Validation AUC was artificially inflated due to identity overlap between train/val splits
- **Fix**: Implemented identity-stratified splitting
- **Result**: More realistic metrics, but collapse persisted

### January 10, 2026: ArcFace Scale Hypothesis
- **Hypothesis**: High ArcFace scale (s=30) amplifies logit magnitudes, causing gradient instability
- **Proposed Solution**: Ablation study with s_end ∈ {12, 15, 18, 20}
- **Result**: All experiments collapsed at the same step, **invalidating the hypothesis**

### January 11, 2026 (Morning): Regularization Dominance Hypothesis
- **New Hypothesis**: The SVD regularization losses (orthogonal + keepsv) are growing faster than classification loss, eventually dominating the gradient signal
- **Evidence**: Training loss grows 8× while val loss stable → non-classification component
- **Action**: Added diagnostic logging for `cls_loss`, `reg_loss`, `reg_cls_ratio`

### January 11, 2026 (Afternoon): THE BREAKTHROUGH 🎯

While analyzing the training logs with new diagnostics, we noticed something critical in the parameter analysis:

```log
backbone.transformer.resblocks.0.attn.out_proj.S_residual: [257] (257)
backbone.transformer.resblocks.0.attn.out_proj.U_residual: [768, 257] (197,376)
backbone.transformer.resblocks.0.attn.out_proj.V_residual: [257, 768] (197,376)
```

**Wait... `U_residual` has shape [768, 257]?**

The config said `hidden_size: 512` and `rank: 511`, which should give `k = 512 - 511 = 1` trainable direction. But the actual layers are **768-dimensional**, giving `k = 768 - 511 = 257`!

---

## 3. The Discovery: Rank Mismatch

### The Confusion: hidden_size vs Attention Dimension

| Config Parameter | What We Thought | Reality |
|-----------------|-----------------|---------|
| `hidden_size: 512` | Attention layer width | CLIP's **output embedding dimension** |
| `rank: 511` | For 512×512 attention layers | Applied to **768×768** attention layers |

### ViT-B-16 Architecture Reality

The ViT-B-16 model has:
- **Patch embedding dimension**: 768
- **Attention head dimension**: 768 / 12 heads = 64 per head
- **Attention out_proj**: **768 × 768** (NOT 512 × 512)
- **CLIP output pooler**: 768 → 512 (this is where 512 comes from!)

```
Input Image (224×224)
    ↓
Patch Embedding → [B, 197, 768]  ← 768 dimensions!
    ↓
12× Transformer Blocks
    ├── MultiheadAttention (768 dim)
    │   └── out_proj: Linear(768, 768)  ← SVD applied HERE
    └── MLP
    ↓
CLS Token → [B, 768]
    ↓
Projection Head → [B, 512]  ← This is where 512 comes from!
```

### The Math: Why k=257 Instead of k=1

The Effort method decomposes weight matrices via SVD:
```
W = U @ diag(S) @ V^T

where:
- W: [out_features, in_features] = [768, 768]
- U: [768, 768]  (left singular vectors)
- S: [768]       (singular values)
- V: [768, 768]  (right singular vectors)
```

With `rank = r`, we keep the top-r singular components frozen and train the remaining `k = min(d, d) - r` components:

| Configured rank | Actual d | Result k | Status |
|----------------|----------|----------|--------|
| 511 | 768 | **257** | ❌ WRONG |
| 767 | 768 | **1** | ✅ CORRECT |

---

## 4. Root Cause Analysis

### Why k=257 Causes Collapse

#### 4.1 Parameter Explosion

With k=257 trainable directions per layer across 12 transformer blocks:

| Component | Per Layer | 12 Layers | Total |
|-----------|-----------|-----------|-------|
| U_residual [768, 257] | 197,376 | 2,368,512 | |
| S_residual [257] | 257 | 3,084 | |
| V_residual [257, 768] | 197,376 | 2,368,512 | |
| **Total SVD params** | 395,009 | **4,740,108** | ~4.74M |

Expected with k=1:
| Component | Per Layer | 12 Layers |
|-----------|-----------|-----------|
| U_residual [768, 1] | 768 | 9,216 |
| S_residual [1] | 1 | 12 |
| V_residual [1, 768] | 768 | 9,216 |
| **Total SVD params** | 1,537 | **~18,444** |

**We were training 257× more SVD parameters than the Effort paper recommends!**

#### 4.2 Orthogonality Constraint Explosion

The orthogonal loss enforces:
```
L_orth = ||[U_r | U_res][U_r | U_res]^T - I||_F + ||[V_r; V_res][V_r; V_res]^T - I||_F
```

With k=1: This is a (d+1)×(d+1) identity constraint → relatively easy to maintain
With k=257: This is a (d+257)×(d+257) identity constraint → **much harder to maintain**

As the 257 residual vectors drift during training, maintaining mutual orthogonality becomes increasingly difficult, causing `orthogonal_loss` to grow.

#### 4.3 KeepSV Loss Amplification

The keepsv loss enforces:
```
L_keepsv = |W_current_fnorm² - W_original_fnorm²|
```

With 257 trainable singular directions, the weight matrix can change much more dramatically than with k=1, leading to larger Frobenius norm deviations.

#### 4.4 Weight Decay Conflict (Secondary Issue)

The optimizer was configured with `weight_decay=0.05` applied uniformly to ALL parameters, including U_residual, S_residual, V_residual.

Weight decay pushes parameter norms toward zero:
```
θ_new = θ_old - lr * (∇L + weight_decay * θ_old)
```

But the SVD constraints want to maintain orthogonality and original Frobenius norm. This creates a **constant tug-of-war**:
- Weight decay: "Shrink the SVD factors!"
- Orthogonal loss: "Keep them orthogonal!"
- KeepSV loss: "Maintain the original norm!"

---

## 5. Implemented Fixes

### Fix 1: Correct Rank Configuration (PRIMARY)

**Files Modified**: All `*_batch256.yaml` experiment configs

**Change**:
```yaml
# BEFORE
rank: 511  # Gives k=257 for 768-dim layers

# AFTER
# FIXED Jan 11, 2026: ViT-B-16 attention layers are 768x768, NOT 512x512
# rank=767 gives k=1 trainable singular direction (matching Effort paper default)
# Previously rank=511 gave k=257, causing reg_loss to dominate
rank: 767  # Gives k=1 for 768-dim layers
```

**Files Updated**:
- `experiments/arcface_scale_ablation/vit_B16_laion_s12_batch256.yaml`
- `experiments/arcface_scale_ablation/vit_B16_laion_s15_batch256.yaml`
- `experiments/arcface_scale_ablation/vit_B16_laion_s18_batch256.yaml`
- `experiments/arcface_scale_ablation/vit_B16_laion_s20_batch256.yaml`

### Fix 2: Disable Weight Decay for SVD Parameters

**File Modified**: `utils/setup.py` → `choose_optimizer()`

**Change**: SVD residual parameters now have their own param group with `weight_decay=0.0`

```python
def choose_optimizer(model, config):
    # Separate SVD residual parameters from other parameters
    svd_param_names = ['U_residual', 'S_residual', 'V_residual']
    
    svd_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(svd_name in name for svd_name in svd_param_names):
            svd_params.append(param)
        else:
            other_params.append(param)
    
    param_groups = [
        {'params': svd_params, 'weight_decay': 0.0, 'name': 'svd_residual'},
        {'params': other_params, 'weight_decay': weight_decay, 'name': 'other'}
    ]
    
    optimizer = optim.Adam(param_groups, lr=lr, eps=eps)
```

### Fix 3: SVD Layer Dimension Diagnostic

**File Modified**: `detectors/effort_detector.py`

**New Function**: `log_svd_layer_dimensions(model, configured_rank)`

This diagnostic function runs at model initialization and:
1. Inspects all SVDResidualLinear layers
2. Reports actual k value vs expected k=1
3. **Warns loudly if k ≠ 1** with the correct rank to use
4. Shows parameter counts for verification

**Sample Output (Correct Config)**:
```
======================================================================
🔬 SVD LAYER DIMENSION ANALYSIS (CRITICAL FOR EFFORT METHOD)
======================================================================
   Configured rank: 767
   Effort paper default: k=1 trainable singular direction
----------------------------------------------------------------------
   Found 12 SVDResidualLinear layers
   Layer dimension (d): 768
   Trainable singular directions per layer (k): 1
   Total trainable directions across all layers: 12
   ✅ CORRECT: k=1 matches Effort paper default
----------------------------------------------------------------------
   Total trainable params in SVD layers: 18,444
   Expected if k=1: ~18,444
======================================================================
```

**Sample Output (Incorrect Config)**:
```
======================================================================
🔬 SVD LAYER DIMENSION ANALYSIS (CRITICAL FOR EFFORT METHOD)
======================================================================
   Configured rank: 511
   Effort paper default: k=1 trainable singular direction
----------------------------------------------------------------------
   Found 12 SVDResidualLinear layers
   Layer dimension (d): 768
   Trainable singular directions per layer (k): 257
   Total trainable directions across all layers: 3084
   ⚠️ WARNING: k=257 != 1 (Effort paper default)
   ⚠️ This means you're training 257× more directions than intended!
   ⚠️ For k=1 with 768-dim layers, set rank=767
----------------------------------------------------------------------
   Total trainable params in SVD layers: 4,740,108
   Expected if k=1: ~18,444
   ⚠️ SVD params are 257.0× higher than k=1 baseline!
======================================================================
```

### Fix 4: Loss Component Logging (Already Implemented)

The `get_losses()` method now returns detailed breakdown:

| Key | Description | Use Case |
|-----|-------------|----------|
| `cls_loss` | Classification loss BEFORE regularization | Track learning signal |
| `reg_loss` | Total regularization term | Track constraint penalty |
| `orthogonal_loss` | Sum of orthogonal losses (raw) | Identify drift |
| `keepsv_loss` | Sum of keepsv losses (raw) | Identify norm deviation |
| `reg_cls_ratio` | `reg_loss / cls_loss` | **KEY**: Should stay ~1.0 |

---

## 6. Expected Outcomes

### After Fixes, We Expect:

| Metric | Before (k=257) | After (k=1) |
|--------|----------------|-------------|
| Trainable SVD params | ~4.74M | ~18K |
| Training loss trend | Grows 8× (2.8→22) | Stable or decreasing |
| reg_cls_ratio | Grows >> 1.0 | Stays ~1.0 |
| Collapse step | ~4600 | **None** (or much later) |
| Peak val_holdout AUC | ~0.67-0.70 | Higher (TBD) |

### Diagnostic Checkpoints

During the next training run, we should verify:

1. **At initialization**: SVD diagnostic shows `k=1` with ✅
2. **At step 100**: `reg_cls_ratio` < 2.0
3. **At step 1000**: Training loss decreasing or stable
4. **At step 4000**: No collapse warning, `reg_cls_ratio` < 5.0
5. **At step 6000+**: Model still learning, no constant predictions

---

## 7. Validation Plan

### Experiment to Run

We will re-run the `s18_batch256` configuration with all fixes applied:

```yaml
# Key changes from previous run:
rank: 767  # Was 511
# Weight decay for SVD params: 0.0 (via optimizer param groups)
```

### Success Criteria

| Criterion | Threshold | Measurement Point |
|-----------|-----------|-------------------|
| No collapse | Logit std > 0.1 | Throughout training |
| reg_cls_ratio bounded | < 10.0 | At step 6000 |
| Training loss stable | Not growing > 2× | Steps 4000-8000 |
| Val AUC improvement | > 0.70 | Best checkpoint |
| Training completes | 12000 steps | End of run |

### Metrics to Monitor

1. **W&B Dashboard**:
   - `train/loss/cls_loss` vs `train/loss/reg_loss`
   - `train/loss/reg_cls_ratio` (should stay bounded)
   - `train/loss/orthogonal_loss` and `train/loss/keepsv_loss`

2. **Console Logs**:
   - SVD dimension diagnostic at startup
   - Optimizer param group summary
   - No COLLAPSE WARNINGs

---

## 8. Technical Details

### Effort Method Background

The Effort paper (ICML 2025 Oral) proposes SVD-based fine-tuning for CLIP backbones:

1. **Decompose** attention projection weights via SVD: `W = U @ diag(S) @ V^T`
2. **Freeze** top-r singular components (preserve pretrained knowledge)
3. **Train** remaining k components (learn task-specific features)
4. **Regularize** to maintain orthogonality and weight norm

The paper explicitly ablates over k ∈ {64, 16, 6, 1} and finds **k=1 works best**.

### Why k=1 Works Best

With k=1:
- Minimal perturbation to pretrained representations
- Easier to maintain orthogonality (only 1 new vector to keep orthogonal)
- Prevents overfitting to training distribution
- Better generalization to unseen deepfake methods

### Rank Configuration by Backbone

| Backbone | Attention Dim | Correct rank | Result k |
|----------|--------------|--------------|----------|
| ViT-B-16 (LAION) | 768 | 767 | 1 |
| ViT-B-32 | 768 | 767 | 1 |
| ViT-L-14 | 1024 | 1023 | 1 |
| ViT-H-14 | 1280 | 1279 | 1 |

**General Rule**: `rank = attention_dim - 1` for k=1

---

## 9. Lessons Learned

### 9.1 Configuration Naming Ambiguity

The `hidden_size: 512` parameter was misleading because:
- It refers to CLIP's **output embedding dimension** (after pooling)
- NOT the **attention layer width** (which is 768 for ViT-B-16)

**Action**: Added comment in experiment configs explaining the distinction.

### 9.2 Importance of Diagnostic Logging

The SVD layer dimension diagnostic would have caught this issue immediately. 

**Action**: Added `log_svd_layer_dimensions()` that runs at model init and warns if k ≠ 1.

### 9.3 Weight Decay Interaction with Constraints

Applying weight decay to parameters that have explicit loss constraints (orthogonality, norm preservation) creates conflicting optimization objectives.

**Action**: SVD residual parameters now have separate param group with `weight_decay=0.0`.

### 9.4 Parameter Count as Sanity Check

The log showed "Trainable parameters: 4,741,132" which should have been a red flag. With k=1, we'd expect ~18K trainable backbone params.

**Action**: SVD diagnostic now shows expected vs actual param counts.

---

## Appendix A: File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `experiments/.../vit_B16_laion_s12_batch256.yaml` | Config | rank: 511 → 767 |
| `experiments/.../vit_B16_laion_s15_batch256.yaml` | Config | rank: 511 → 767 |
| `experiments/.../vit_B16_laion_s18_batch256.yaml` | Config | rank: 511 → 767 |
| `experiments/.../vit_B16_laion_s20_batch256.yaml` | Config | rank: 511 → 767 |
| `utils/setup.py` | Code | Separate param groups for SVD |
| `detectors/effort_detector.py` | Code | Add `log_svd_layer_dimensions()` |
| `detectors/effort_detector.py` | Code | Loss component breakdown (already had) |

---

## Appendix B: Effort Paper Reference

**Title**: "Effort: Efficient Orthogonal Modeling for Generalizable AI-Generated Image Detection"  
**Venue**: ICML 2025 (Oral)  
**Key Insight**: Fine-tune CLIP with k=1 trainable singular direction per attention layer

**Relevant Equations**:

SVD Decomposition:
```
W = U_r @ diag(S_r) @ V_r^T + U_res @ diag(S_res) @ V_res^T
     └── frozen (top-r) ──┘   └── trainable (remaining k) ──┘
```

Orthogonal Loss:
```
L_orth = ||[U_r | U_res]^T [U_r | U_res] - I||_F + ||[V_r; V_res] [V_r; V_res]^T - I||_F
```

KeepSV Loss:
```
L_keepsv = |||W_current||_F^2 - ||W_original||_F^2|
```

---

## Appendix C: Results (To Be Updated)

*This section will be updated with results from the validation run.*

### Validation Run Details
- **Start Time**: TBD
- **Config**: `vit_B16_laion_s18_batch256.yaml` with fixes
- **W&B Run**: TBD

### Results Summary
| Metric | Value |
|--------|-------|
| Collapse occurred? | TBD |
| Peak val_holdout AUC | TBD |
| Final reg_cls_ratio | TBD |
| Total steps completed | TBD |

### Training Curves
*W&B plots to be added*

---

**Document Version**: 1.0  
**Last Updated**: January 12, 2026  
**Authors**: Investigation Team
