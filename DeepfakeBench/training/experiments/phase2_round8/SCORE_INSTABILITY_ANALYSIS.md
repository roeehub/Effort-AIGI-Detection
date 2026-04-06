# Score Instability on Near-Identical Frames — Analysis & Fix Plan

**Date:** February 27, 2026  
**Context:** Observed during R8 evaluation — nearly identical video frames produce wildly different scores (e.g., 0.03 vs 0.48). Primarily seen on B16 backbone models.

---

## 1. Root Cause Analysis

### Primary Cause: ArcFace Scale Amplification + Decision Boundary Proximity

The ArcFace logit is `s * cos(θ)`. With the current R8 config (`s_end: 18.0`), a tiny angular change in feature space gets amplified 18× before the sigmoid:

| cos(θ) | logit (s=18) | sigmoid | Probability |
|--------|-------------|---------|-------------|
| +0.05  | +0.90       | 0.711   | 71.1%       |
| +0.01  | +0.18       | 0.545   | 54.5%       |
| −0.01  | −0.18       | 0.455   | 45.5%       |
| −0.05  | −0.90       | 0.289   | 28.9%       |

A 0.02 cosine perturbation → **10pp probability swing**. A 0.10 perturbation → **42pp swing**. This matches the observed 0.03 → 0.48 behavior: the samples sit near the decision boundary and the scale factor amplifies microscopic feature differences.

### Contributing Factors

| Factor | Mechanism |
|--------|-----------|
| **Pixel-level codec jitter** | Consecutive video frames differ in DCT rounding errors. ViT patch embeddings are linear projections of raw pixels — no pooling absorbs this. |
| **JPEG/codec block ↔ ViT patch misalignment** | ViT-B-16 uses 16×16 patches. H.264 uses 4/8/16px macroblocks. A 1px shift in face crop realigns different codec blocks to different ViT patches. |
| **SVD residual amplification** | Effort trains on the orthogonal residual of the weight matrix — a high-frequency, low-energy subspace. Small input perturbations produce proportionally large changes in this subspace. |
| **Face crop instability** | Slightly different bounding boxes on adjacent frames → different 224×224 resampled inputs. |

### Connection to Threshold Calibration Gap (R8 Report §7)

The val EER threshold (~0.45) vs OOD EER threshold (~0.77) gap shares the same root cause: over-scaled logits compress the useful probability range. Fixing score instability will also narrow the calibration gap.

---

## 2. Is This a Known Problem?

**Yes.** Known in the literature as:

- **Prediction instability / flickering** — frame-to-frame score oscillation in video deepfake detection
- **Calibration brittleness** — high-confidence models with sharp decision boundaries
- **Local Lipschitz violation** — small ‖Δx‖ → large ‖Δf(x)‖
- **Temporal incoherence** — any frame-level classifier applied to video

Especially prevalent with ViT backbones (no built-in translation invariance), high ArcFace scale, and residual/subspace methods like Effort.

---

## 3. Training-Time Fixes — Ranked by Effort

### Fix 1: Label Smoothing ⏱️ 5 min | Config-only

**Impact:** Medium | **Backward compatible:** ✅ fully | **Risk:** Near-zero

Prevents logits from being pushed to ±∞ during training. Target becomes [0.05, 0.95] instead of [0, 1], so the score surface near the boundary is flatter.

**Implementation:**

```yaml
# experiment config
label_smoothing: 0.05
```

```python
# In ArcFaceMixin or wherever CrossEntropyLoss is created:
label_smoothing = getattr(self.config, 'label_smoothing', 0.0)
self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
```

---

### Fix 2: Lower ArcFace Scale ⏱️ 5 min | Config-only

**Impact:** HIGH (directly addresses root cause) | **Backward compatible:** ✅ fully | **Risk:** ~0.5–1pp EER increase on val set

```yaml
# Current (too aggressive):
s_start: 10.0
s_end: 18.0

# Proposed (gentler):
s_start: 6.0
s_end: 12.0
```

With `s=12` the same ±0.02 cosine perturbation produces ~0.06 probability swing instead of ~0.10. Combined with label smoothing, the 0.03→0.48 behavior becomes ~0.45→0.55.

**Trade-off note:** Lower `s` slightly reduces in-distribution sharpness, but improves OOD calibration. Expect narrowing of the val/OOD threshold gap (R8 §7).

---

### Fix 3: Input Perturbation Consistency Loss (Mixin) ⏱️ 1–2 hours

**Impact:** HIGH | **Backward compatible:** ✅ new mixin, gated by `stability_lambda: 0.0` | **Risk:** Slightly slower training (one extra forward pass)

For each batch, generate a slightly perturbed copy (Gaussian noise + small spatial shift), and penalize KL divergence between clean and perturbed predictions. This explicitly teaches the SVD residual subspace to be invariant to input noise while retaining sensitivity to spatially coherent artifacts.

**Config:**

```yaml
stability_lambda: 0.5        # [0.1, 1.0] range; 0.0 = disabled
stability_noise_std: 0.02     # ~5/255 pixel noise
stability_crop_jitter: 0.03   # ~7px shift on 224x224
```

**Implementation:** New `StabilityRegMixin` that:
1. Generates perturbed inputs (Gaussian noise + random crop-and-resize)
2. Forward passes the perturbed batch through the model
3. Computes `KL(softmax(logits_perturbed) ‖ softmax(logits_clean.detach()))`
4. Returns `stability_lambda * kl_loss` to add to the main loss

Follows the existing mixin pattern (`ArcFaceMixin`, `CurriculumMixin`, etc.).

**Code sketch:**

```python
# trainer/mixins/stability_mixin.py
class StabilityRegMixin:
    def _init_stability_reg(self):
        self.stability_lambda = getattr(self.config, 'stability_lambda', 0.0)
        self.stability_noise_std = getattr(self.config, 'stability_noise_std', 0.02)
        self.stability_crop_jitter = getattr(self.config, 'stability_crop_jitter', 0.03)

    def compute_stability_loss(self, model, images, logits_clean):
        if self.stability_lambda <= 0:
            return torch.tensor(0.0, device=images.device)
        perturbed = self._generate_perturbation(images)
        logits_perturbed = model(perturbed)
        p_clean = F.softmax(logits_clean.detach(), dim=-1)
        p_pert = F.log_softmax(logits_perturbed, dim=-1)
        kl = F.kl_div(p_pert, p_clean, reduction='batchmean')
        return self.stability_lambda * kl

    def _generate_perturbation(self, images):
        B, C, H, W = images.shape
        perturbed = images.clone()
        # Gaussian noise
        noise = torch.randn_like(perturbed) * self.stability_noise_std
        perturbed = (perturbed + noise).clamp(0, 1)
        # Crop jitter
        margin = int(H * self.stability_crop_jitter)
        if margin >= 1:
            top = torch.randint(0, margin + 1, (1,)).item()
            left = torch.randint(0, margin + 1, (1,)).item()
            perturbed = perturbed[:, :, top:top+H-margin, left:left+W-margin]  
            perturbed = F.interpolate(perturbed, size=(H, W), mode='bilinear', align_corners=False)
        return perturbed
```

**Integration in training loop:**

```python
# After main loss computation:
stability_loss = self.compute_stability_loss(self.model, images, logits)
total_loss = main_loss + stability_loss
wandb.log({"train/stability_loss": stability_loss.item()})
```

---

### Fix 4: SVD Residual Feature Normalization ⏱️ 2–3 hours

**Impact:** Medium | **Backward compatible:** ✅ flag-gated | **Risk:** May need `s` re-tuning

L2-normalize features before the classification head. ArcFace conceptually assumes normalized features (`cos(θ)`), but if actual feature magnitudes vary (they do with SVD residuals), the effective scale varies per-sample.

```yaml
residual_feature_norm: true
```

```python
# In effort_detector.py forward, before ArcFace head:
if getattr(self.config, 'residual_feature_norm', False):
    features = F.normalize(features, p=2, dim=-1)
```

---

### Fix 5: Multi-Crop Training ⏱️ 1–2 hours

**Impact:** Medium-High | **Backward compatible:** ✅ augmentation wrapper | **Risk:** 1.5–2× slower training

Feed 2 slightly different crops of the same face per sample and average logits before the loss. Forces learned features to be robust to crop boundary alignment with ViT patches.

```yaml
multi_crop_n: 2              # number of crops per face (1 = disabled)
multi_crop_jitter: 0.04      # max shift fraction
```

---

## 4. Recommended R9 Configuration

Apply Fixes 1–3 together for maximum impact:

```yaml
# === Score stability fixes ===
label_smoothing: 0.05
s_start: 6.0
s_end: 12.0
stability_lambda: 0.5
stability_noise_std: 0.02
stability_crop_jitter: 0.03
```

Expected outcome:
- Frame-to-frame jitter reduced from ±0.20 probability to ±0.05
- Val/OOD threshold calibration gap narrows
- In-distribution EER may increase ~0.5–1pp (acceptable trade-off)

---

## 5. Quick Summary Table

| # | Fix | Time | Impact | Config-Only? | Risk |
|---|-----|------|--------|-------------|------|
| 1 | Label smoothing | 5 min | Medium | ✅ | Near-zero |
| 2 | Lower ArcFace `s` | 5 min | **High** | ✅ | ~0.5pp EER |
| 3 | Perturbation consistency loss | 1–2 hr | **High** | New mixin, gated | Slower training |
| 4 | Residual feature normalization | 2–3 hr | Medium | Flag-gated | Needs `s` re-tune |
| 5 | Multi-crop training | 1–2 hr | Medium-High | Aug wrapper | 1.5–2× slower |
