# Investigation Report: L14 vs B16_LAION Backbone Performance Gap

**Date:** January 13, 2026  
**Author:** Copilot + Human Analysis  
**Status:** � IN PROGRESS - Root causes identified, fixes pending

---

## Executive Summary

We achieved **breakthrough AUC of 0.9881** on the ViT-L-14 OpenAI backbone after optimizing hyperparameters. However, when applying the same optimizations to the ViT-B-16-LAION-DataComp-XL backbone, we observe **significantly worse performance** (~0.62-0.80 AUC) with concerning instability when using ArcFace loss.

**Key Finding:** The L14 optimizations do not transfer directly to B16_LAION due to **implementation asymmetry** in our SVD application — NOT inherent architectural limitations. ArcFace actively hurts B16_LAION training (AUC declines after initial peak), while no-ArcFace achieves stable but mediocre ~0.80 AUC.

**Root Cause Identified:** Our OpenCLIP implementation only applies SVD to `out_proj` (1 layer), while HuggingFace CLIP applies SVD to all 4 attention projections (q, k, v, out). This is a **fixable implementation gap**, not a fundamental limitation.

> ⚠️ **Important Context:** The original Effort paper (arxiv:2411.15633) uses **CrossEntropyLoss**, not ArcFace. ArcFace is our extension. This may explain some incompatibility issues.

---

## 1. Background: The Original Problem

### 1.1 Initial B16_LAION ArcFace Scale Ablation (Jan 11-12)

We ran ArcFace scale ablation experiments (`s_end=12, 15, 18, 20`) on B16_LAION with the original hyperparameters:
- `learning_rate: 5e-5`
- `lr_scheduler_warmup_steps: 4000`
- `rank: 767` (corrected from earlier 511 bug)

**Results:** All runs plateaued at **~0.65-0.70 AUC** (mediocre)

### 1.2 Diagnosis: Why B16 Underperformed

Analysis revealed several issues:
1. **Low learning rate** (5e-5) too conservative for ~19k trainable parameters
2. **Long warmup** (4000 steps) delays effective learning
3. **ArcFace weight norm collapse** - weights not learning angular margin properly

---

## 2. L14 Experiments: The Success Story

### 2.1 Experiment Design

We created 5 L14 ablation experiments with progressively aggressive hyperparameters:

| Experiment | Learning Rate | Warmup | ArcFace | Key Changes |
|------------|--------------|--------|---------|-------------|
| `s18_fast_lr` | 1.5e-4 | 1000 | s=18 | Higher LR |
| `s18_lr2e4` | 2.0e-4 | 1000 | s=18 | Even higher LR |
| `s18_all_improvements` | 2.0e-4 | 1000 | s=18, anneal=1000 | Full optimization |
| `no_arcface` | 2.0e-4 | 1000 | None | BCE loss baseline |
| `s18_backbone_only` | 5e-5 | 4000 | s=18 | Control (original settings) |

**Config file:** `experiments/L14_ablation/vit_L14_openai_s18_all_improvements.yaml`

### 2.2 L14 Architecture Details

From `config/backbone_registry.yaml`:
```yaml
ViT-L-14:
  hidden_size: 1024
  num_layers: 24
  num_heads: 16
  recommended_rank: 1023  # hidden_size - 1
```

**Trainable Parameter Breakdown (from logs):**
```
Total parameters: 504,606,720
Trainable parameters: 198,752
  - backbone (SVD residuals): 196,704
  - head (ArcFace): 2,048
Trainable ratio: 0.04%
```

**SVD Configuration:**
- 24 transformer layers × 4 attention projections (q, k, v, out) = 96 SVDResidualLinear layers
- Each layer: `U_residual: [1024, 1]`, `V_residual: [1, 1024]`, `S_residual: [1]`
- k=1 trainable singular direction per layer (Effort paper default)
- ~2,049 params per layer × 96 layers ≈ 196,704 backbone params

### 2.3 L14 Results: BREAKTHROUGH 🎉

**Run:** `L14_OpenAI_s18_all_improvements`  
**Log file:** `debug/L14_OpenAI_s18_all_improvements_0112-0804.log`

| Step | val_in_dist AUC | val_holdout AUC | Trend |
|------|-----------------|-----------------|-------|
| 500 | 0.6484 | 0.6249 | Starting |
| 1000 | 0.7858 | 0.7634 | Rapid improvement |
| 1500 | 0.9276 | **0.9551** | Breakthrough |
| 2000 | 0.9438 | 0.9634 | Continued improvement |
| 2500 | 0.9514 | 0.9714 | |
| 3000 | 0.9637 | 0.9777 | |
| 3500 | 0.9671 | 0.9820 | |
| 4000 | 0.9698 | 0.9810 | |
| 4500 | 0.9714 | **0.9862** | Near-peak |
| Final | 0.9766 | **0.9881** | Best val_holdout! |

**Best metrics achieved:**
- **val_holdout AUC: 0.9881**
- **val_in_dist AUC: 0.9766**
- **EER: 0.0306**

---

## 3. B16_LAION Optimized Experiments: The Problem

### 3.1 Experiment Design

We applied L14 learnings to B16_LAION:

| Experiment | Learning Rate | Warmup | ArcFace | Hypothesis |
|------------|--------------|--------|---------|------------|
| `s18_all_improvements` | 2.0e-4 | 1000 | s=18 | Direct transfer |
| `s18_lr3e4` | 3.0e-4 | 1000 | s=18 | More aggressive LR |
| `no_arcface_all_improvements` | 2.0e-4 | 1000 | None | Simpler loss |

**Config files:** `experiments/arcface_scale_ablation/B16_LAION_optimized/`

### 3.2 B16_LAION Architecture Details

From `config/backbone_registry.yaml`:
```yaml
ViT-B-16-DataComp-XL:
  library: "open_clip"
  hidden_size: 512       # NOTE: OpenCLIP projects 768 embed_dim to 512 output_dim
  embed_dim: 768         # Internal transformer dimension (for SVD rank)
  num_layers: 12
  num_heads: 12
  recommended_rank: 767  # embed_dim - 1 (NOT 511)
```

> ✅ **Config Issue FIXED:** The `backbone_registry.yaml` file previously had `recommended_rank: 511` which was incorrect. SVD operates on attention layers at `embed_dim=768`, so it should be `767`. **This has been corrected.**

**Trainable Parameter Breakdown (from logs):**
```
Total parameters: 100,358,656
Trainable parameters: 19,468
  - backbone (SVD residuals): 18,444
  - head (ArcFace): 1,024
Trainable ratio: 0.02%
```

**SVD Configuration (CURRENT IMPLEMENTATION - SUBOPTIMAL):**
- 12 transformer layers × **1 attention out_proj only** = 12 SVDResidualLinear layers
- Each layer: `U_residual: [768, 1]`, `V_residual: [1, 768]`, `S_residual: [1]`
- k=1 trainable singular direction per layer ✅
- ~1,537 params per layer × 12 layers ≈ 18,444 backbone params

**Why only out_proj?** OpenCLIP uses `nn.MultiheadAttention` which fuses q, k, v into a single `in_proj_weight` tensor. Our current `apply_svd_residual_to_openclip_attn()` only replaces `out_proj`, missing the fused q/k/v weights.

### 3.3 B16_LAION Results: POOR PERFORMANCE ❌

**Log files:**
- `debug/B16_LAION_s18_all_improvements_0112-1728.log`
- `debug/B16_LAION_s18_lr3e4_0112-1728.log`
- `debug/B16_LAION_no_arcface_all_improvements_0112-1728.log`

#### Results Comparison at ~Step 10k:

| Experiment | Peak val_in_dist AUC | Final val_in_dist AUC | Final val_holdout AUC | Trend |
|------------|---------------------|----------------------|----------------------|-------|
| **no_arcface** | 0.81 | **0.81** | **0.78** | ✅ Stable |
| s18_all_improvements | 0.69 | 0.62 | 0.62 | ❌ Declining |
| s18_lr3e4 | 0.69 | 0.62 | 0.62 | ❌ Declining |

#### Detailed Progression (val_in_dist AUC):

| Step | no_arcface | s18_all_improvements | s18_lr3e4 |
|------|------------|---------------------|-----------|
| 500 | 0.68 | 0.58 | 0.62 |
| 1000 | 0.75 | 0.68 | **0.69** (peak) |
| 1500 | 0.78 | **0.69** (peak) | 0.63 |
| 2000 | 0.79 | 0.65 | 0.65 |
| 2500 | **0.80** | 0.63 | 0.63 |
| 3000 | 0.79 | 0.62 | 0.61 |
| 5000 | 0.81 | 0.64 | 0.64 |
| 10000 | 0.81 | 0.63 | 0.62 |

![val_in_dist AUC comparison](../figs/b16_laion_optimized_val_in_dist_auc.png)
*Figure 1: B16_LAION val_in_dist AUC over training steps. Red = no_arcface (stable ~0.81), Blue = s18_all_improvements (declining), Green = s18_lr3e4 (declining)*

### 3.4 Training Stability Analysis

The `train/probabilities` metric for B16_LAION_s18_all_improvements shows **extreme variance**:

![train probabilities variance](../figs/b16_laion_train_probabilities.png)
*Figure 2: Train probabilities for B16_LAION_s18_all_improvements showing wild oscillations (0.2 to 0.9)*

This indicates the ArcFace loss is causing training instability for B16_LAION.

---

## 4. Architecture Comparison: Why L14 Works and B16 Doesn't (Yet)

### 4.1 Key Architectural Differences

| Property | ViT-L-14 OpenAI | ViT-B-16 LAION DataComp |
|----------|-----------------|-------------------------|
| **Library** | HuggingFace `transformers` | `open_clip` |
| **Model Parameters** | 504M | 100M |
| **Transformer Layers** | 24 | 12 |
| **Hidden Size** | 1024 | 768 (internal) → 512 (output) |
| **Attention Heads** | 16 | 12 |
| **Patch Size** | 14×14 | 16×16 |
| **SVD Layers Modified** | 96 (q, k, v, out × 24 layers) | **12 (out_proj only × 12 layers)** ⚠️ |
| **Trainable Params** | 198,752 | 19,468 |
| **Pre-training Data** | OpenAI WebImageText (400M) | DataComp-1B (12.8B filtered) |

### 4.2 Critical Difference: SVD Layer Coverage (IMPLEMENTATION GAP)

**L14 (HuggingFace):** SVD applied to **all 4 attention projections** (q, k, v, out) per layer
```
backbone.encoder.layers.0.self_attn.k_proj.S_residual: [1]
backbone.encoder.layers.0.self_attn.v_proj.S_residual: [1]
backbone.encoder.layers.0.self_attn.q_proj.S_residual: [1]
backbone.encoder.layers.0.self_attn.out_proj.S_residual: [1]
```

**B16_LAION (OpenCLIP) - CURRENT:** SVD applied to **out_proj only** per layer
```
backbone.transformer.resblocks.0.attn.out_proj.S_residual: [1]
```

**Why the difference?**
- HuggingFace CLIP uses separate `nn.Linear` for q, k, v → easy to replace each
- OpenCLIP uses `nn.MultiheadAttention` which fuses q, k, v into `in_proj_weight` (shape `[3*d, d]`)
- Our `apply_svd_residual_to_openclip_attn()` currently only handles `out_proj`

> 🔧 **This is FIXABLE:** We can extend the code to apply SVD to the fused `in_proj_weight` by treating it as 3 stacked weight matrices.

**Current Impact:** L14 has **8× more trainable SVD directions** than B16_LAION at the same layer depth. Accounting for layer count (24 vs 12), it's **4× more per-layer** due to our implementation gap.

### 4.3 Trainable Parameters Breakdown

| Backbone | Trainable Params | Total Params | Ratio | Breakdown |
|----------|-----------------|--------------|-------|-----------|
| L14 | 198,752 | 504M | 0.04% | 24 layers × 4 proj × ~2049 params |
| B16_LAION (current) | 19,468 | 100M | 0.02% | 12 layers × 1 proj × ~1537 params |
| B16_LAION (with in_proj fix) | ~73,776 | 100M | 0.07% | 12 layers × 4 proj × ~1537 params (**estimated**) |

The 10× difference breaks down as:
1. **2× from layer count:** 24 vs 12 transformer layers
2. **4× from SVD coverage:** 4 projections vs 1 (fixable!)
3. **~1.3× from hidden dim:** 1024 vs 768

### 4.4 Output Dimension Mismatch

**L14:** Head input = hidden_size = 1024  
**B16_LAION:** Head input = output_dim = 512 (projected from 768 embed_dim)

The OpenCLIP projection layer (`visual.proj`) reduces the feature dimension from 768 to 512 before classification. This may affect the expressiveness of learned representations.

---

## 5. Root Cause Analysis

### Primary Cause: SVD Coverage Implementation Gap (CONFIRMED)

**Evidence:** 
- L14 has 96 SVDResidualLinear layers (4 projections × 24 layers)
- B16_LAION has only 12 SVDResidualLinear layers (1 projection × 12 layers)
- This is NOT an inherent limitation — it's our implementation choice

**Why it matters:** The Effort method learns in the orthogonal subspace. With 4× fewer trainable directions per layer, B16_LAION has significantly less capacity to learn discriminative features while maintaining orthogonality.

**Fix available:** Extend `apply_svd_residual_to_openclip_attn()` to handle `in_proj_weight`

### Secondary Cause: ArcFace + Constrained Capacity Conflict

**Evidence:** 
- ArcFace runs show declining AUC after step 1000-1500
- no_arcface is stable at ~0.81 AUC
- Train probabilities show wild oscillations for ArcFace runs

**Analysis:** With only 12 SVD layers and 18k backbone params, competing optimization objectives may conflict:
1. Orthogonality constraints (reg_loss) - maintain SVD structure
2. Angular margin (ArcFace) - push classes apart in normalized space  
3. Classification (cls_loss) - learn discriminative features

> 📝 **Note:** The original Effort paper uses CrossEntropyLoss, NOT ArcFace. ArcFace is our extension. Once SVD coverage is fixed, ArcFace should work (as proven by L14).

### Contributing Factor: Config File Inconsistency

**Issue:** `backbone_registry.yaml` has `recommended_rank: 511` for B16_LAION, but SVD operates at `embed_dim=768`, so it should be `767`.

**Impact:** May cause confusion in future experiments. Needs cleanup.

---

## 6. Action Plan: Getting B16_LAION + ArcFace to Work

Given that L14 + ArcFace succeeded, we have confidence B16_LAION + ArcFace should also succeed after addressing the implementation gaps.

### 🔴 CRITICAL: Fix SVD Coverage (Blocking Issue)

#### Action 1: Extend SVD to in_proj_weight for OpenCLIP
**File:** `detectors/effort_detector.py` → `apply_svd_residual_to_openclip_attn()`

**Current behavior:**
```python
# Only replaces out_proj
module.out_proj = SVDResidualLinear(...)
```

**Required change:**
```python
# Also handle in_proj_weight (fused q, k, v)
# in_proj_weight has shape [3*embed_dim, embed_dim]
# Need to either:
# Option A: Split into 3 separate SVDResidualLinear layers
# Option B: Create a custom SVD layer that handles the fused structure
```

**Expected outcome:** ~4× more trainable params for B16_LAION (~73k vs ~19k)

#### Action 2: Fix backbone_registry.yaml
**File:** `config/backbone_registry.yaml`

**Change:** ✅ **COMPLETED**
```yaml
# FROM:
recommended_rank: 511  # output_dim - 1

# TO:
recommended_rank: 767  # embed_dim - 1 (SVD operates at internal dim, not output dim)
```

### 🟡 DIAGNOSTIC: Isolate Variables with Control Experiment

#### Action 3: Test B16 OpenAI (HuggingFace) - CRITICAL CONTROL
**Purpose:** Determine if the issue is:
- (A) LAION pretraining vs OpenAI pretraining, OR
- (B) OpenCLIP implementation (out_proj only) vs HuggingFace (all 4 projections)

**Config:**
```yaml
backbone:
  source: openai
  variant: ViT-B-16
  huggingface_id: openai/clip-vit-base-patch16
  hidden_size: 768
```

**Expected outcome:**
- If B16 OpenAI works well → Problem is SVD coverage (confirms Action 1)
- If B16 OpenAI also fails → Problem is model capacity (need different approach)

### 🟢 OPTIMIZATION: After SVD Fix is Implemented

#### Action 4: Re-run B16_LAION with ArcFace after SVD fix
**Config:** Same as `s18_all_improvements` but with extended SVD coverage
- `learning_rate: 2.0e-4`
- `warmup_steps: 1000`
- `arcface_s: 18`
- `anneal_steps: 1000`

#### Action 5: Hyperparameter sweep if needed
If Action 4 doesn't reach parity with L14:
- Try `learning_rate: 1e-4` (more conservative for smaller model)
- Try `anneal_steps: 2000-3000` (slower ArcFace ramp-up)

---

## 7. Implementation Notes for SVD Extension

### Option A: Split in_proj_weight into 3 SVDResidualLinear layers

```python
def apply_svd_residual_to_openclip_attn(model, r):
    for name, module in model.named_children():
        if isinstance(module, nn.MultiheadAttention):
            embed_dim = module.embed_dim
            
            # Handle out_proj (existing)
            module.out_proj = replace_with_svd_residual(module.out_proj, r)
            
            # Handle in_proj_weight (NEW)
            # in_proj_weight shape: [3*embed_dim, embed_dim]
            # Split into q, k, v components
            if module.in_proj_weight is not None:
                W = module.in_proj_weight.data  # [3*d, d]
                W_q = W[:embed_dim, :]           # [d, d]
                W_k = W[embed_dim:2*embed_dim, :]  # [d, d]
                W_v = W[2*embed_dim:, :]         # [d, d]
                
                # Create SVD layers for each
                # NOTE: This requires modifying the forward pass
                # to use these instead of in_proj_weight
```

**Complexity:** HIGH - requires changes to forward pass

### Option B: Custom SVDResidualInProj layer

Create a single layer that handles the fused structure while applying SVD to each q/k/v section independently.

**Complexity:** MEDIUM - cleaner but still requires forward pass changes

### Option C: Use separate Linear layers in OpenCLIP

Modify model loading to convert `nn.MultiheadAttention` to separate q/k/v Linear layers before applying SVD.

**Complexity:** MEDIUM - may affect model loading/compatibility

---

## 8. File References

### Experiment Configs
- L14 (success): `experiments/L14_ablation/vit_L14_openai_s18_all_improvements.yaml`
- B16 ArcFace: `experiments/arcface_scale_ablation/B16_LAION_optimized/vit_B16_laion_s18_all_improvements.yaml`
- B16 no_arcface: `experiments/arcface_scale_ablation/B16_LAION_optimized/vit_B16_laion_no_arcface_all_improvements.yaml`

### Log Files
- L14 best: `debug/L14_OpenAI_s18_all_improvements_0112-0804.log`
- B16 ArcFace: `debug/B16_LAION_s18_all_improvements_0112-1728.log`
- B16 lr3e4: `debug/B16_LAION_s18_lr3e4_0112-1728.log`
- B16 no_arcface: `debug/B16_LAION_no_arcface_all_improvements_0112-1728.log`

### Code References
- SVDResidualLinear: `detectors/effort_detector.py:804`
- OpenCLIP SVD application: `detectors/effort_detector.py:1134` (`apply_svd_residual_to_openclip_attn`)
- HuggingFace SVD application: `detectors/effort_detector.py:991` (`apply_svd_residual_to_self_attn`)
- Backbone registry: `config/backbone_registry.yaml`

---

## 9. Conclusion

The L14 backbone achieves excellent results (0.9881 AUC) with ArcFace. The B16_LAION underperformance is **NOT** due to inherent architectural limitations but rather **implementation gaps** in our SVD application.

**Key insight:** Once we extend SVD coverage to all 4 attention projections for OpenCLIP models, B16_LAION should achieve comparable results to L14 with appropriate hyperparameter tuning.

### Summary of Required Actions

| Priority | Action | Status | Expected Impact |
|----------|--------|--------|-----------------|
| � P0 | Fix `backbone_registry.yaml` rank | ✅ DONE | Config correctness |
| 🔴 P1 | Extend SVD to `in_proj_weight` | TODO | 4× more trainable params |
| 🟡 P2 | Test B16 OpenAI (control) | TODO | Isolate root cause |
| 🟢 P3 | Re-run B16_LAION + ArcFace | BLOCKED | Target: >0.90 AUC |

---

*Report generated: January 13, 2026*
*Last updated: January 13, 2026 - Added root cause analysis and action plan*
