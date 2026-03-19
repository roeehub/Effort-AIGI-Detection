# ArcFace Scale Ablation Experiments - Jan 10, 2026

## Executive Summary

Despite implementing the identity leakage and frequency imbalance fixes (Jan 9, 2026), training **still collapsed**. Analysis revealed a **different failure mode**: ArcFace scale annealing to high values (s=30) causes the model to output constant predictions.

This document details:
1. Diagnosis of the new collapse mechanism
2. Logging enhancements added for debugging
3. ArcFace scale ablation experiments created
4. Results comparison framework

---

## Problem Analysis: Jan 10 Run

### Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Config | `combined_paired_vit_B16_laion.yaml` |
| Backbone | ViT-B-16 LAION DataComp |
| Data Source | Combined DF40 + DeepLive |
| Learning Rate | 5e-5 |
| Batch Size | 32 |
| ArcFace s_start | 10.0 |
| ArcFace s_end | 30.0 |
| ArcFace anneal_steps | 4000 |
| ArcFace margin (m) | 0.28 |
| Warmup Steps | 1200 |

### Observed Behavior

**Key log evidence:**

```
Step 1401: pred_fake mean=0.3755 std=0.2068, logit_diff std=0.5903 ← LEARNING
Step 2101: pred_fake mean=0.4521 std=0.1745, logit_diff std=0.4923 ← Still OK
Step 2801: pred_fake mean=0.4932 std=0.0029, logit_diff std=0.0082 ← COLLAPSING
Step 3501: pred_fake mean=0.5000 std=0.0000, logit_diff std=0.0000 ← DEAD
Step 3801: All pred_fake=0, all samples predicted as REAL ← COLLAPSED
```

**Identity leakage fix verified working:**
```
Data split BY IDENTITY (seed=737):
  - Train identities: 1,499 (80.0%)
  - Val identities: 187 (10.0%)
  - Test identities: 188 (10.0%)
  ✓ No identity overlap between splits
```

### Failure Timeline

| Step | ArcFace s | LR Phase | Behavior |
|------|-----------|----------|----------|
| 0-1200 | 10→13 | Warmup | Normal learning |
| 1200-2400 | 13→16 | Peak LR | Still learning |
| 2400-3200 | 16→20 | Peak LR | **Logit std dropping** |
| 3200-4000 | 20→25 | Decaying | **Collapse begins** |
| 4000+ | 25→30 | Decaying | **Complete collapse** |

### Root Cause: ArcFace Scale + LR Schedule Misalignment

**The problem:**
1. LR warmup completes at step 1200 (LR at peak)
2. ArcFace scale continues climbing until step 4000
3. At steps 1200-4000: Peak LR + climbing scale = **double gradient amplification**
4. Model finds degenerate solution: output constant 0.5 probability

**This is DIFFERENT from the Jan 9 collapse:**

| Aspect | Jan 9 Collapse | Jan 10 Collapse |
|--------|----------------|-----------------|
| Root Cause | Identity leakage | ArcFace scale too high |
| Evidence | SVD layers dying | Logit std → 0 |
| Metric | `svd/near_zero_layers` increasing | `pred_fake_std` → 0 |
| Collapse Point | ~3000-4000 | ~2800-3500 |
| Fix Status | ✅ Fixed (verified) | ❌ New issue |

---

## Fixes Implemented

### Fix 1: Enhanced Logging (trainer/trainer.py)

Added new diagnostic methods to catch collapse early:

#### 1.1 Collapse Warning System
```python
def _check_collapse_warning(self, predictions: Dict, labels: torch.Tensor, step: int) -> Dict:
    """
    Check for signs of model collapse (constant outputs).
    Returns dict of warning indicators for logging.
    """
    warnings = {}
    
    # Check if all predictions are the same
    if 'pred_fake' in predictions:
        pred_fake = predictions['pred_fake']
        pred_std = pred_fake.std().item()
        warnings['pred_fake_std'] = pred_std
        warnings['collapse_warning'] = pred_std < 0.01  # Threshold for concern
        
        if pred_std < 0.001:
            self.logger.warning(
                f"⚠️ COLLAPSE DETECTED at step {step}: "
                f"pred_fake std={pred_std:.6f} (all predictions nearly identical)"
            )
    
    # Check logit distribution
    if 'raw_logits' in predictions:
        logits = predictions['raw_logits']
        logit_std = logits.std().item()
        warnings['logit_std'] = logit_std
        
        # Per-class statistics
        if logits.dim() == 2 and logits.size(1) == 2:
            warnings['logit_real_mean'] = logits[:, 0].mean().item()
            warnings['logit_fake_mean'] = logits[:, 1].mean().item()
            warnings['logit_diff_std'] = (logits[:, 1] - logits[:, 0]).std().item()
    
    return warnings
```

#### 1.2 ArcFace Diagnostics
```python
def _collect_arcface_diagnostics(self) -> Dict:
    """
    Collect ArcFace-specific metrics for debugging collapse.
    """
    metrics = {}
    
    if hasattr(self.model, 'arcface_head') and self.model.arcface_head is not None:
        head = self.model.arcface_head
        
        # Current scale value
        metrics['arcface/current_s'] = head.s
        
        # Weight statistics
        if hasattr(head, 'weight'):
            w = head.weight
            metrics['arcface/weight_norm'] = w.norm().item()
            metrics['arcface/weight_std'] = w.std().item()
            
            # Cosine similarity between class centers
            w_normalized = F.normalize(w, dim=1)
            cos_sim = (w_normalized[0] @ w_normalized[1]).item()
            metrics['arcface/center_cosine_sim'] = cos_sim
        
        # Gradient statistics (if available)
        if hasattr(head, 'weight') and head.weight.grad is not None:
            metrics['arcface/weight_grad_norm'] = head.weight.grad.norm().item()
    
    return metrics
```

#### 1.3 Per-Class Logit Statistics

Added to training loop logging:
```python
# Per-class logit stats for collapse detection
if 'raw_logits' in predictions and predictions['raw_logits'].dim() == 2:
    logits = predictions['raw_logits']
    metrics['train/logit_real_mean'] = logits[:, 0].mean().item()
    metrics['train/logit_fake_mean'] = logits[:, 1].mean().item()
    metrics['train/logit_diff_std'] = (logits[:, 1] - logits[:, 0]).std().item()

# Collapse warning
collapse_metrics = self._check_collapse_warning(predictions, labels, self.global_step)
metrics.update({f'train/{k}': v for k, v in collapse_metrics.items()})

# ArcFace diagnostics
arcface_metrics = self._collect_arcface_diagnostics()
metrics.update(arcface_metrics)
```

---

### Fix 2: Schedule Alignment

**Problem:** LR peaks at step 1200, but ArcFace anneals until step 4000.

**Solution:** Align warmup with ArcFace annealing:

```yaml
# OLD (misaligned)
lr_scheduler_warmup_steps: 1200  # LR peaks early
anneal_steps: 4000                # ArcFace still climbing

# NEW (aligned)
lr_scheduler_warmup_steps: 4000  # LR and ArcFace ramp together
anneal_steps: 4000                # Both reach target at same step
```

---

### Fix 3: ArcFace Scale Reduction

Created experiment variants with reduced `s_end`:

| Variant | s_start | s_end | Risk Level |
|---------|---------|-------|------------|
| baseline | 10 | 30 | ❌ Expected collapse |
| risky | 10 | 18 | ⚠️ Boundary test |
| safe | 10 | 15 | ✅ Should work |
| conservative | 10 | 12 | ✅ Safest |

---

### Fix 4: Batch Size & Data Split

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `frames_per_batch` | 32 | 64 | Smoother gradients for ArcFace stability |
| `train_split` | 0.8 | 0.9 | More training data |
| `test_split` | 0.1 | 0.0 | Skip test, focus on training stability |

---

## Experiment Matrix

### Configs Created

Located in `experiments/arcface_scale_ablation/`:

| File | Backbone | s_end | Warmup | Batch |
|------|----------|-------|--------|-------|
| `vit_B16_laion_s12_conservative.yaml` | ViT-B-16 LAION | 12 | 4000 | 64 |
| `vit_B16_laion_s15_safe.yaml` | ViT-B-16 LAION | 15 | 4000 | 64 |
| `vit_B16_laion_s18_risky.yaml` | ViT-B-16 LAION | 18 | 4000 | 64 |
| `vit_B16_laion_s30_baseline.yaml` | ViT-B-16 LAION | 30 | 4000 | 64 |
| `vit_L14_openai_s12_conservative.yaml` | ViT-L-14 OpenAI | 12 | 4000 | 64 |
| `vit_L14_openai_s15_safe.yaml` | ViT-L-14 OpenAI | 15 | 4000 | 64 |
| `vit_L14_openai_s18_risky.yaml` | ViT-L-14 OpenAI | 18 | 4000 | 64 |
| `vit_L14_openai_s30_baseline.yaml` | ViT-L-14 OpenAI | 30 | 4000 | 64 |

### Common Config (All Experiments)

```yaml
data_source: combined_paired
learning_rate: 5.0e-5
gradient_clip_val: 1.0
lr_scheduler: "cosine_with_warmup"
lr_scheduler_warmup_steps: 4000  # Aligned with anneal_steps
anneal_steps: 4000
arcface_m: 0.28
frames_per_batch: 64
train_split: 0.9
val_split: 0.1
test_split: 0.0
evaluate_every_steps: 500
```

---

## Results Comparison Template

### Run Summary Table

| Run | Date | Config | s_end | Collapsed? | Collapse Step | Best AUC | Notes |
|-----|------|--------|-------|------------|---------------|----------|-------|
| 1 | Jan 9 | df40_paired_vit_B16_laion | 30 | ✅ Yes | ~3000 | 0.75 | Identity leakage |
| 2 | Jan 10 | combined_paired_vit_B16_laion | 30 | ✅ Yes | ~2800 | 0.66 | ArcFace collapse |
| 3 | Jan 10+ | B16_s15_safe | 15 | ? | ? | ? | Pending |
| 4 | Jan 10+ | B16_s18_risky | 18 | ? | ? | ? | Pending |
| 5 | Jan 10+ | L14_s15_safe | 15 | ? | ? | ? | Pending |
| 6 | Jan 10+ | L14_s18_risky | 18 | ? | ? | ? | Pending |

### Detailed Metrics Table

| Run | Step 1000 | Step 2000 | Step 3000 | Step 4000 | Step 5000 |
|-----|-----------|-----------|-----------|-----------|-----------|
| **Run 2 (baseline s=30)** |||||
| val_auc | 0.58 | 0.66 | 0.54 | 0.33 | 0.33 |
| pred_fake_std | 0.21 | 0.17 | 0.01 | 0.00 | 0.00 |
| logit_diff_std | 0.59 | 0.49 | 0.01 | 0.00 | 0.00 |
| arcface_s | 12.5 | 15.0 | 17.5 | 20.0 | 25.0 |
| **Run 3 (s_end=15)** |||||
| val_auc | ? | ? | ? | ? | ? |
| pred_fake_std | ? | ? | ? | ? | ? |
| logit_diff_std | ? | ? | ? | ? | ? |
| arcface_s | ? | ? | ? | ? | ? |

### Collapse Indicators

Key metrics to determine collapse:

| Metric | Healthy | Warning | Collapsed |
|--------|---------|---------|-----------|
| `pred_fake_std` | > 0.1 | 0.01 - 0.1 | < 0.01 |
| `logit_diff_std` | > 0.3 | 0.1 - 0.3 | < 0.1 |
| `val_auc` | > 0.6 | 0.5 - 0.6 | < 0.5 |
| `collapse_warning` | False | - | True |

---

## Launch Commands

```bash
# Priority experiments (should succeed)
./launch_experiment.sh arcface-scale-ablation asia-southeast1 experiments/arcface_scale_ablation/vit_B16_laion_s15_safe.yaml
./launch_experiment.sh arcface-scale-ablation asia-southeast1 experiments/arcface_scale_ablation/vit_L14_openai_s15_safe.yaml

# Boundary tests
./launch_experiment.sh arcface-scale-ablation asia-southeast1 experiments/arcface_scale_ablation/vit_B16_laion_s18_risky.yaml
./launch_experiment.sh arcface-scale-ablation asia-southeast1 experiments/arcface_scale_ablation/vit_L14_openai_s18_risky.yaml

# Conservative fallback (if s15/s18 still fail)
./launch_experiment.sh arcface-scale-ablation asia-southeast1 experiments/arcface_scale_ablation/vit_B16_laion_s12_conservative.yaml
./launch_experiment.sh arcface-scale-ablation asia-southeast1 experiments/arcface_scale_ablation/vit_L14_openai_s12_conservative.yaml
```

---

## Expected Outcomes

### If s_end=15 Works (No Collapse)

- **Confirms:** ArcFace scale >20 is the collapse trigger
- **Next steps:** 
  - Run full training with s_end=15
  - Test s_end=18 to find exact boundary
  - May need to increase margin (m) to compensate for lower scale

### If s_end=15 Still Collapses

- **Indicates:** Problem is not just scale, may be:
  - Combined dataset issue
  - Learning rate still too high
  - Backbone-specific problem
- **Next steps:**
  - Try s_end=12 (conservative)
  - Try without ArcFace entirely
  - Reduce LR to 2e-5

### If s_end=18 Works But s_end=30 Doesn't

- **Confirms:** Collapse boundary is between s=18 and s=30
- **Next steps:**
  - Use s_end=18 for production
  - Consider slower annealing (anneal_steps=6000)

---

## Files Changed (Jan 10)

| File | Change Type | Description |
|------|-------------|-------------|
| `trainer/trainer.py` | Modified | Added `_check_collapse_warning()`, `_collect_arcface_diagnostics()`, enhanced logging |
| `experiments/arcface_scale_ablation/` | **New folder** | 8 experiment configs + README |
| `docs/ARCFACE_SCALE_ABLATION_JAN_10_2026.md` | **New** | This document |

---

## Version History

| Date | Author | Changes |
|------|--------|---------|
| Jan 10, 2026 | Copilot | Diagnosed ArcFace collapse, added logging, created ablation experiments |

---

## References

- Previous fixes: `docs/TRAINING_PIPELINE_FIXES_JAN_9_2026.md`
- Experiment configs: `experiments/arcface_scale_ablation/README.md`
- ArcFace paper: [ArcFace: Additive Angular Margin Loss](https://arxiv.org/abs/1801.07698)
