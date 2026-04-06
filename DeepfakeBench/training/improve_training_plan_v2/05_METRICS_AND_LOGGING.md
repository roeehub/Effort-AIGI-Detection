# Metrics & Logging

This document explains the W&B metrics logged during training and how to interpret them.

## 1. Metric Categories

### 1.1 Train Metrics

Logged every step (or every `log_progress_steps` steps):

| Metric | Type | Description |
|--------|------|-------------|
| `train/loss` | float | Total loss (CE + regularization) |
| `train/real_loss` | float | Loss on real samples only |
| `train/fake_loss` | float | Loss on fake samples only |
| `train/lr` | float | Current learning rate |
| `train/step` | int | Global training step |

### 1.2 Gradient Metrics

| Metric | Type | Healthy Range | Description |
|--------|------|---------------|-------------|
| `train/grad_norm` | float | 0.1 - 10.0 | L2 norm of all gradients |
| `train/params_with_grad` | int | 36-37 | Count of params with non-zero grad |

**Interpretation:**
- `grad_norm = 0`: ❌ No gradients flowing (frozen model or bug)
- `grad_norm > 100`: ⚠️ Exploding gradients, consider clipping
- `params_with_grad < 36`: ❌ Some residual params not receiving gradients

### 1.3 Prediction Balance Metrics

| Metric | Type | Healthy Range | Description |
|--------|------|---------------|-------------|
| `train/pred_balance/fake_ratio` | float | 0.4 - 0.6 | Fraction of predictions that are "fake" |
| `train/probabilities` | histogram | Wide distribution | Raw prediction probabilities |

**Interpretation:**
- `fake_ratio = 1.0`: ❌ Model collapsed to predict all "fake"
- `fake_ratio = 0.0`: ❌ Model collapsed to predict all "real"
- `fake_ratio ≈ 0.5`: ⚠️ Could be random guessing OR balanced learning

### 1.4 Logit Diagnostics

| Metric | Type | Healthy Range | Description |
|--------|------|---------------|-------------|
| `train/logits/diff_mean` | float | Varies | Mean of (fake_logit - real_logit) |
| `train/logits/diff_std` | float | > 0.1 | Std of logit differences |

**Interpretation:**
- `diff_std ≈ 0`: ❌ **Critical bug!** Model outputs identical predictions for all samples
- `diff_std > 0.5`: ✅ Model is differentiating between samples
- `diff_mean` increasing: Model is learning to predict "fake" more confidently

### 1.5 Confidence Metrics

| Metric | Type | Healthy Range | Description |
|--------|------|---------------|-------------|
| `train/confidence/mean` | float | 0.6 - 0.95 | Average prediction confidence |
| `train/confidence/std` | float | > 0 | Variability in confidence |
| `train/confidence/fraction_confident` | float | 0.2 - 0.9 | Fraction with >0.8 confidence |

**Interpretation:**
- `confidence/mean` increasing: Model becoming more decisive
- `fraction_confident = 1.0` early: ⚠️ Might be overfitting or collapsed

### 1.6 ArcFace Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `train/arcface/s` | float | Current scale factor (during annealing) |

### 1.7 SVD Residual Diagnostics (Added Jan 4, 2026)

These metrics help diagnose issues with the EFFORT SVD residual training:

| Metric | Type | Healthy Range | Description |
|--------|------|---------------|-------------|
| `svd/S_residual_min` | float | > 1e-6 | Minimum S_residual value across all layers |
| `svd/S_residual_max` | float | varies | Maximum S_residual value across all layers |
| `svd/S_residual_mean` | float | varies | Mean S_residual value across all layers |
| `svd/layer_count` | int | 12-24 | Number of SVDResidualLinear layers found |
| `svd/near_zero_layers` | int | 0 | Count of layers with S_residual < 1e-6 |

**Interpretation:**
- `near_zero_layers > 0`: ⚠️ Some layers have effectively stopped learning
- `S_residual_min` decreasing to near zero: ⚠️ Potential dying gradient issue
- `layer_count = 0`: ❌ SVD residual not applied correctly

**Use Case:**
These metrics were added to diagnose the `params_with_grad` drop observed in LAION B16 training.
If `S_residual` values approach zero, the corresponding gradients will vanish.

## 2. Validation Metrics

### 2.1 Primary Metrics

| Metric | Type | Good | Description |
|--------|------|------|-------------|
| `val/auc` | float | > 0.7 | Area Under ROC Curve |
| `val/eer` | float | < 0.3 | Equal Error Rate |
| `val/accuracy` | float | > 0.6 | Classification accuracy |
| `val/ap` | float | > 0.7 | Average Precision |

### 2.2 Per-Dataset Metrics

Metrics are logged separately for each validation set:

```
val_in_dist/overall/auc      # In-distribution validation
val_holdout/overall/auc      # Held-out test set
ood/overall/auc              # Out-of-distribution (if configured)
```

### 2.3 Per-Class Metrics

```
val_holdout/real/accuracy    # Accuracy on real samples
val_holdout/fake/accuracy    # Accuracy on fake samples
val_holdout/real/precision   # Precision for real class
val_holdout/fake/recall      # Recall for fake class
```

## 3. System Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `system/gpu_memory_used` | float | GPU memory in GB |
| `system/gpu_utilization` | float | GPU utilization % |
| `system/samples_per_second` | float | Training throughput |

## 4. Diagnostic Logging

### 4.1 Console Diagnostic Messages

Every 100 batches, the training code logs:

```
[DIAG batch #1] Label distribution: real=16, fake=16
[DIAG batch #1] Logit diff stats (fake-real): mean=0.27, std=0.15
[DIAG batch #1] Predictions: pred_fake=20, pred_real=12
[DIAG batch #1] Real frames: 10/16 predicted correctly
[DIAG batch #1] Fake frames: 14/16 predicted correctly
```

**What to look for:**
- `std=0.0000`: ❌ Critical bug (was the OpenCLIP SVD issue)
- Balanced `pred_fake/pred_real`: ✅ Model is learning
- Improving accuracy over time: ✅ Model is converging

### 4.2 First Forward Pass Log

On the first batch:

```
[FIRST FORWARD] Input image shape: torch.Size([32, 3, 224, 224])
[FIRST FORWARD] Backbone output (features) shape: torch.Size([32, 768])
[FIRST FORWARD] Expected hidden_size: 768
```

**Check for dimension mismatches!**

## 5. Interpreting Training Progress

### 5.1 Healthy Training Pattern

| Step | Loss | AUC | grad_norm | diff_std | fake_ratio |
|------|------|-----|-----------|----------|------------|
| 0 | 0.69 | 0.50 | 0.5 | 0.05 | 0.5 |
| 100 | 0.55 | 0.60 | 0.8 | 0.2 | 0.45 |
| 500 | 0.40 | 0.75 | 0.6 | 0.5 | 0.48 |
| 1000 | 0.30 | 0.85 | 0.4 | 0.8 | 0.52 |

**Characteristics:**
- Loss decreasing steadily
- AUC increasing
- grad_norm stable, non-zero
- diff_std increasing (model differentiating)
- fake_ratio staying near 0.5

### 5.2 Model Collapse Pattern

| Step | Loss | AUC | grad_norm | diff_std | fake_ratio |
|------|------|-----|-----------|----------|------------|
| 0 | 0.69 | 0.50 | 0.5 | 0.05 | 0.5 |
| 100 | 0.50 | 0.50 | 1.2 | 0.00 | 1.0 |
| 500 | 0.35 | 0.50 | 1.5 | 0.00 | 1.0 |

**Symptoms:**
- AUC stuck at 0.5 (random)
- diff_std = 0 (identical predictions)
- fake_ratio = 1.0 (all "fake")
- Loss may still decrease (learning to be confident about wrong answer)

### 5.3 Overfitting Pattern

| Step | Train Loss | Val AUC | Train AUC |
|------|------------|---------|-----------|
| 0 | 0.69 | 0.50 | 0.50 |
| 500 | 0.20 | 0.80 | 0.90 |
| 1000 | 0.05 | 0.75 | 0.99 |
| 1500 | 0.01 | 0.70 | 1.00 |

**Symptoms:**
- Train loss → 0, train AUC → 1.0
- Val AUC peaks then decreases
- Gap between train/val metrics grows

## 6. W&B Dashboard Setup

### 6.1 Recommended Panels

**Overview Panel:**
- `train/loss` (line chart)
- `val_holdout/overall/auc` (line chart)
- `train/lr` (line chart)

**Gradient Health Panel:**
- `train/grad_norm` (line chart)
- `train/params_with_grad` (line chart)

**Prediction Quality Panel:**
- `train/logits/diff_std` (line chart) ⭐ Key metric
- `train/pred_balance/fake_ratio` (line chart)
- `train/confidence/mean` (line chart)

**Validation Panel:**
- `val_in_dist/overall/auc` vs `val_holdout/overall/auc`
- `val_holdout/overall/eer`

### 6.2 Alerts to Configure

| Condition | Alert |
|-----------|-------|
| `train/logits/diff_std < 0.01` | 🚨 Model collapse detected |
| `train/pred_balance/fake_ratio > 0.95` | 🚨 Predicting all fake |
| `train/grad_norm > 100` | ⚠️ Gradient explosion |
| `val_holdout/overall/auc < previous - 0.1` | ⚠️ Catastrophic forgetting |

## 7. Common Issues & Solutions

### 7.1 diff_std = 0 (The SVD Bug)

**Symptoms:**
- `train/logits/diff_std ≈ 0`
- All predictions same class
- 50% accuracy despite decreasing loss

**Cause:** SVD wrapper was zeroing MHA out_proj weight.

**Solution:** Fixed in Jan 2026 - SVDResidualLinear now properly replaces out_proj.

**Verification:**
```bash
python scripts/test_svd_residual.py
```

### 7.2 grad_norm = 0

**Symptoms:**
- `train/grad_norm = 0`
- `train/params_with_grad = 0`
- Model not learning

**Cause:** All parameters frozen incorrectly.

**Solution:** Check `requires_grad` on residual parameters:
```python
for name, param in model.named_parameters():
    if 'residual' in name:
        print(f"{name}: requires_grad={param.requires_grad}")
```

### 7.3 Loss = NaN

**Symptoms:**
- Loss becomes NaN after some steps
- Training crashes

**Causes:**
- Learning rate too high
- Gradient explosion
- Numerical instability in ArcFace

**Solutions:**
1. Reduce `learning_rate`
2. Add `gradient_clip_val: 1.0`
3. Use ArcFace annealing (low s_start)

### 7.4 AUC Not Improving

**Symptoms:**
- AUC stuck around 0.5-0.6
- Loss decreasing slowly

**Causes:**
- Learning rate too low
- Insufficient model capacity
- Data issue

**Solutions:**
1. Increase `learning_rate`
2. Decrease `rank` (more trainable params)
3. Check data distribution

## 8. Post-Training Analysis

### 8.1 Export W&B Data

```python
import wandb
api = wandb.Api()
run = api.run("entity/project/run_id")
history = run.history()
```

### 8.2 Key Questions to Answer

1. **Did the model learn?**
   - Final AUC > 0.7? ✅
   - diff_std increased over training? ✅

2. **Is it generalizing?**
   - val_holdout AUC close to val_in_dist AUC? ✅
   - No large gap between train/val metrics? ✅

3. **What's the failure mode?**
   - Higher error on real or fake?
   - Specific manipulation methods harder?

---

*See also: [01_TRAINING_FLOW.md](01_TRAINING_FLOW.md) for training pipeline details*
