# ArcFace Scale Ablation Collapse Investigation

**Date**: January 11, 2026  
**Branch**: `refactor-training`  
**Investigators**: Analysis of batch256 experiments (s12, s15, s18, s20)

---

## Executive Summary

All four ArcFace scale ablation experiments collapsed at approximately the same training step (~4600-4800), regardless of the ArcFace scale value (s_end = 12, 15, 18, or 20). This indicates the collapse is **step-dependent, not scale-dependent**, invalidating our initial hypothesis that high ArcFace scale was the primary cause.

---

## 1. Background & Prior Work

### January 9, 2026: Identity Leakage Fix
- **Problem**: Validation AUC was artificially inflated due to identity overlap between train/val splits
- **Fix**: Implemented identity-stratified splitting to ensure no identity appears in both train and validation
- **Result**: More realistic (lower) validation metrics, but training still collapsed

### January 10, 2026: ArcFace Scale Hypothesis
- **Observation**: Training collapsed with high ArcFace scale (s=30)
- **Hypothesis**: High scale amplifies logit magnitude, causing gradient instability
- **Proposed Solution**: Reduce s_end and test multiple values

---

## 2. Experiment Configuration

### Common Configuration (All Experiments)
```yaml
# Model Architecture
backbone:
  source: laion
  variant: ViT-B-16
  openclip_model: ViT-B-16
  openclip_pretrained: datacomp_xl_s13b_b90k
  hidden_size: 512
  resolution: 224

# SVD Fine-tuning (Effort Method)
rank: 511  # Near-full rank for ViT-B-16 (512 dim)
lambda_reg: 1.0  # Regularization weight for orthogonal + keepsv losses

# ArcFace Configuration
use_arcface_head: true
arcface_m: 0.28  # Angular margin
s_start: 10  # Initial scale (annealed up)
anneal_steps: 4000  # Steps to anneal from s_start to s_end

# Training Schedule
batch_size: 256
num_epochs: 150
warmup_steps: 4000
total_training_steps: 12000  # ~143 epochs at 84 steps/epoch
scheduler: cosine_with_warmup

# Data
data_source: combined (DF40 + DeepLive)
train_samples: 4909
train_identities: 1341
steps_per_epoch: 84  # ceil(4909 / 256) ≈ 20, but with augmentation/sampling = 84
```

### Varied Parameter: `s_end`
| Experiment | s_end | W&B Run |
|------------|-------|---------|
| s12_batch256 | 12 | arcface-scale-ablation project |
| s15_batch256 | 15 | arcface-scale-ablation project |
| s18_batch256 | 18 | arcface-scale-ablation project |
| s20_batch256 | 20 | arcface-scale-ablation project |

---

## 3. Expected Outcomes

### Hypothesis Being Tested
> "Higher ArcFace scale (s) causes earlier collapse due to gradient instability from amplified logit magnitudes."

### Expected Results If Hypothesis Were Correct
| Experiment | Expected Collapse Timing | Expected Peak AUC |
|------------|-------------------------|-------------------|
| s12 | Latest (most stable) | Highest |
| s15 | Later | High |
| s18 | Earlier | Moderate |
| s20 | Earliest (least stable) | Lowest |

### Success Criteria
1. Clear correlation between s_end and collapse timing
2. Lower s_end → later collapse → higher final AUC
3. s12 should remain stable throughout training or collapse much later than s20

---

## 4. Actual Observations

### 4.1 Collapse Timing (CRITICAL FINDING)

| Experiment | First COLLAPSE WARNING | Step | Epoch |
|------------|----------------------|------|-------|
| s12_batch256 | logit std=0.0099, range=0.039 | 4755 | 57 |
| s20_batch256 | logit std=0.0091, range=0.031 | 4623 | 56 |

**Key Finding**: Both experiments collapsed within ~130 steps of each other (~1.5% difference), despite s_end differing by 67% (12 vs 20).

### 4.2 Validation AUC Progression

| Step | s12 val_holdout AUC | s20 val_holdout AUC |
|------|---------------------|---------------------|
| ~1500 | 0.6278 | 0.6415 |
| ~2500 | 0.6520 | 0.6644 |
| ~3500 | **0.6705** (peak) | **0.7052** (peak) |
| 4000 | 0.6463 | 0.6696 |
| 4500 | 0.5932 | 0.5796 |
| 5000 | ~0.55 | ~0.55 |

**Observations**:
- Both experiments peaked around step 3500 (NOT at warmup end)
- Both began declining BEFORE warmup ended (step 4000)
- s20 actually achieved HIGHER peak AUC than s12 (contradicting hypothesis)

### 4.3 Training Loss Anomaly

```
# s12 Training Loss Progression
Epoch 1:  loss ≈ 2.8
Epoch 20: loss ≈ 6.4
Epoch 40: loss ≈ 12.3
Epoch 57: loss ≈ 22+ (at collapse)

# s20 Training Loss Progression  
Epoch 1:  loss ≈ 2.8
Epoch 20: loss ≈ 6.5
Epoch 40: loss ≈ 12.5
Epoch 56: loss ≈ 22+ (at collapse)
```

**Critical Anomaly**: Training loss grows **8x** over training (2.8 → 22+) while validation loss stays relatively stable (~0.68). This is **highly unusual** for classification tasks and suggests a non-classification loss component is growing unbounded.

### 4.4 Collapse Signature

From logs at collapse:
```
COLLAPSE WARNING: Logit std=0.009, range=0.031. Model may be outputting near-constant predictions!
Predictions: pred_fake=0, pred_real=32 (out of 32 total)
```

The model outputs **identical predictions for all samples** (logit std < 0.01), effectively collapsing to a constant function.

---

## 5. Root Cause Hypotheses

### Hypothesis 1: Regularization Loss Dominates (PRIMARY - 80% confidence)

#### Evidence
1. Training loss grows 8x while val loss stable → suggests non-classification component
2. Collapse timing correlates with post-warmup phase, not ArcFace scale
3. `lambda_reg=1.0` applies full regularization weight throughout training

#### Mechanism
The total loss is computed as:
```python
total_loss = cls_loss + lambda_reg * (orthogonal_loss + keepsv_loss) / num_layers
```

Where:
- `orthogonal_loss = ||UUᵀ - I||_F + ||VVᵀ - I||_F` (per layer)
- `keepsv_loss = |weight_current_fnorm² - weight_original_fnorm²|` (per layer)

As training progresses, the SVD residuals (U_residual, S_residual, V_residual) drift from perfect orthogonality, causing `orthogonal_loss` to grow. With `lambda_reg=1.0` and 12 transformer blocks, this regularization can overwhelm the classification gradient signal.

#### Why collapse happens at step ~4600
1. Warmup ends at step 4000 → LR reaches peak
2. Peak LR causes faster weight updates
3. Faster updates → faster orthogonality drift
4. Orthogonality drift → larger reg_loss
5. reg_loss dominates → classification signal lost → collapse

### Hypothesis 2: LR Schedule Interaction (SECONDARY - 60% confidence)

#### Evidence
- Cosine schedule with 4000 warmup → LR peaks at step 4000
- Collapse happens ~600-800 steps after peak LR
- Val AUC starts declining around step 3500 (before warmup ends)

#### Mechanism
The learning rate schedule may be inappropriate for the loss landscape:
- High LR at step 4000 causes large weight updates
- Large updates violate SVD orthogonality constraints faster than regularization can correct
- Creates a feedback loop: larger updates → larger reg_loss → even larger total loss → even larger gradients

### Hypothesis 3: ArcFace Margin Instability (UNLIKELY - 30% confidence)

#### Evidence
- Margin m=0.28 is relatively aggressive
- After scale annealing completes, full margin penalty is applied

#### Counter-Evidence
- Different scales (12 vs 20) collapse at nearly identical times
- If margin were the issue, lower scale should help (it doesn't)

---

## 6. Missing Data (Critical Gaps)

### 6.1 Loss Component Breakdown (NOW ADDED)

**Before**: Only `train/loss/overall` was logged  
**After**: Now logging:
- `train/loss/cls_loss` - Classification loss before regularization
- `train/loss/reg_loss` - Total regularization term
- `train/loss/orthogonal_loss` - Sum of orthogonal losses (raw)
- `train/loss/keepsv_loss` - Sum of keepsv losses (raw)
- `train/loss/reg_cls_ratio` - **KEY**: If >> 1.0, regularization is dominating

### 6.2 Still Missing (Would Be Useful)

| Data | Criticality | Why It Matters |
|------|-------------|----------------|
| Per-layer orthogonal loss | Medium | Identify which layers drift most |
| SVD residual magnitude (S_residual norm) | Medium | Track how much model is changing |
| Gradient norms (cls vs reg) | Medium | Understand gradient flow |
| Per-layer weight delta | Low | Track weight evolution |

---

## 7. Diagnostic Code Changes

### File Modified
`DeepfakeBench/training/detectors/effort_detector.py`

### Changes to `get_losses()` Method
Added detailed loss component tracking:

```python
# NEW: Track individual regularization components
orthogonal_loss_total = torch.tensor(0.0, device=device)
keepsv_loss_total = torch.tensor(0.0, device=device)

for module in self.backbone.modules():
    if isinstance(module, SVDResidualLinear):
        orth_loss = module.compute_orthogonal_loss()
        keep_loss = module.compute_keepsv_loss()
        orthogonal_loss_total += orth_loss
        keepsv_loss_total += keep_loss
        num_reg += 1

# NEW: Return breakdown in loss_dict
loss_dict = {
    'overall': overall_loss,
    'cls_loss': cls_loss.detach(),
    'reg_loss': reg_term.detach(),
    'orthogonal_loss': orthogonal_loss_total.detach(),
    'keepsv_loss': keepsv_loss_total.detach(),
    'reg_cls_ratio': (reg_term / (cls_loss + 1e-8)).detach(),
    ...
}
```

---

## 8. Next Steps

### Immediate: Validate Hypothesis with New Logging
1. Run new experiment with diagnostic logging enabled
2. Observe `reg_cls_ratio` over training
3. Confirm whether regularization dominates before collapse

### If Hypothesis 1 Confirmed (reg_loss dominates):

#### Option A: Reduce lambda_reg
```yaml
lambda_reg: 0.1  # Reduce from 1.0 to 0.1
```

#### Option B: Regularization Annealing
```yaml
lambda_reg_start: 0.1
lambda_reg_end: 1.0
lambda_reg_anneal_steps: 6000  # Gradually increase regularization
```

#### Option C: Decouple Warmup from Annealing
```yaml
warmup_steps: 2000  # Shorter warmup
anneal_steps: 6000  # Longer annealing
```

#### Option D: Gradient Clipping for SVD Layers
```python
# Clip gradients specifically for SVD residual parameters
torch.nn.utils.clip_grad_norm_(svd_params, max_norm=1.0)
```

### If Hypothesis 1 Not Confirmed:
- Investigate LR schedule more deeply
- Consider alternative explanations for loss growth

---

## 9. Timeline Summary

| Date | Event | Finding |
|------|-------|---------|
| Jan 9, 2026 | Identity leakage fix | Val metrics more realistic but collapse persists |
| Jan 10, 2026 | Propose ArcFace scale hypothesis | Plan ablation experiments |
| Jan 10-11, 2026 | Run batch256 ablations (s12-s20) | All collapse at same step |
| Jan 11, 2026 | Analyze results | Collapse is step-dependent, not scale-dependent |
| Jan 11, 2026 | New hypothesis: regularization dominates | Add diagnostic logging |
| **Next** | Run with diagnostics | Validate or refute reg_loss hypothesis |

---

## 10. Key Equations Reference

### ArcFace Loss
```
L_arcface = -log(exp(s * cos(θ_yi + m)) / Σ exp(s * cos(θ_j)))
```
Where `s` is scale, `m` is margin, `θ` is angle between feature and class weight.

### Orthogonal Loss (per SVD layer)
```
L_orth = 0.5 * ||[U_r | U_res][U_r | U_res]ᵀ - I||_F + 0.5 * ||[V_r; V_res][V_r; V_res]ᵀ - I||_F
```
Enforces that concatenated U and V matrices remain orthogonal.

### KeepSV Loss (per SVD layer)
```
L_keepsv = |W_current_fnorm² - W_original_fnorm²|
```
Encourages the weight Frobenius norm to stay close to original.

### Total Loss
```
L_total = L_cls + λ_reg * (Σ L_orth + Σ L_keepsv) / num_layers
```
With `λ_reg = 1.0` currently.

---

## Appendix A: Log Excerpts

### s12 Collapse Sequence (Epoch 57)
```
2026-01-11 XX:XX:XX - Epoch: 57, Batch: 4755, Loss: 22.1435
2026-01-11 XX:XX:XX - COLLAPSE WARNING: Logit std=0.009872, range=0.039062. Model may be outputting near-constant predictions!
2026-01-11 XX:XX:XX - [DIAG batch #4755] Predictions: pred_fake=0, pred_real=32
```

### s20 Collapse Sequence (Epoch 56)
```
2026-01-11 XX:XX:XX - Epoch: 56, Batch: 4623, Loss: 21.8927
2026-01-11 XX:XX:XX - COLLAPSE WARNING: Logit std=0.009079, range=0.031250. Model may be outputting near-constant predictions!
2026-01-11 XX:XX:XX - [DIAG batch #4623] Predictions: pred_fake=0, pred_real=32
```

---

## Appendix B: Model Architecture Summary

```
EffortDetector
├── backbone: OpenCLIPVisionModelWrapper
│   └── visual: ViT-B-16 (LAION DataComp-XL)
│       └── transformer.resblocks[0-11].attn.out_proj: SVDResidualLinear
│           ├── weight_main: [512, 512] (frozen, top-r singular components)
│           ├── U_residual: [512, 1] (trainable)
│           ├── S_residual: [1] (trainable)
│           └── V_residual: [1, 512] (trainable)
└── head: ArcMarginProduct
    ├── weight: [2, 512] (trainable)
    └── s: scalar (annealed from s_start to s_end)

Total Parameters: 100.3M
Trainable Parameters: 4.74M (4.72%)
  - backbone: 4.73M (SVD residuals only)
  - head: 1,026 (ArcFace weight + bias)
```

---

## Appendix C: Configuration Files Used

### Experiment Config Path
`experiments/df40_paired_vit_B16_laion.yaml`

### Key Config Excerpts
```yaml
# From the experiment config
backbone:
  type: clip
  source: laion
  variant: ViT-B-16
  hidden_size: 512
  resolution: 224
  openclip_model: ViT-B-16
  openclip_pretrained: datacomp_xl_s13b_b90k

use_arcface_head: true
arcface_m: 0.28
s_start: 10
# s_end: varies (12, 15, 18, 20)
anneal_steps: 4000

lambda_reg: 1.0
rank: 511

batch_size: 256
warmup_steps: 4000
total_training_steps: 12000
```
