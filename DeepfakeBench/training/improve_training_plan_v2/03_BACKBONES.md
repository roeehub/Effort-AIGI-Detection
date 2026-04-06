# Backbones & Model Architecture

This document covers the vision backbones used for deepfake detection and the SVD residual training approach.

## 1. Overview

The EFFORT method fine-tunes pre-trained CLIP vision encoders for deepfake detection by:
1. **Freezing** most of the backbone (preserving pre-trained knowledge)
2. **Training only** a low-rank residual on attention output projections
3. Using an **ArcFace head** for margin-based binary classification

This approach is parameter-efficient and prevents catastrophic forgetting.

## 2. Available Backbones

### 2.1 Backbone Registry

All backbones are defined in `config/backbone_registry.yaml`:

| Backbone | Source | Hidden Size | Resolution | Params | Status |
|----------|--------|-------------|------------|--------|--------|
| ViT-L-14 | OpenAI | 1024 | 224 | ~304M | ✅ Baseline |
| ViT-B-16 | OpenAI | 768 | 224 | ~86M | ✅ Available |
| ViT-B-32 | OpenAI | 768 | 224 | ~88M | ✅ Available |
| ViT-B-16-DataComp-XL | LAION | 512* | 224 | ~86M | ✅ Available |

*LAION ViT-B-16 projects from 768 internal dim to 512 output dim.

### 2.2 OpenAI CLIP (HuggingFace)

```yaml
backbone:
  source: "openai"
  variant: "ViT-B-16"
  huggingface_id: "openai/clip-vit-base-patch16"
  hidden_size: 768
```

**Module Structure:**
```
CLIPVisionModel
└── encoder
    └── layers[0..11]
        └── self_attn
            ├── q_proj (nn.Linear)
            ├── k_proj (nn.Linear)
            ├── v_proj (nn.Linear)
            └── out_proj (nn.Linear)  ← SVD applied here
```

**Loading Code:**
```python
from transformers import CLIPModel
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')
vision_encoder = model.vision_model
```

### 2.3 LAION OpenCLIP

```yaml
backbone:
  source: "laion"
  variant: "ViT-B-16-DataComp-XL"
  openclip_model: "ViT-B-16"
  openclip_pretrained: "datacomp_xl_s13b_b90k"
  hidden_size: 512
```

**Module Structure:**
```
OpenCLIP.visual
└── transformer
    └── resblocks[0..11]
        └── attn (nn.MultiheadAttention)
            └── out_proj (nn.Linear)  ← SVD applied here
```

**Loading Code:**
```python
import open_clip
model, _, _ = open_clip.create_model_and_transforms(
    'ViT-B-16', 
    pretrained='datacomp_xl_s13b_b90k'
)
visual_encoder = model.visual
```

### 2.4 Key Differences

| Aspect | OpenAI (HuggingFace) | LAION (OpenCLIP) |
|--------|---------------------|------------------|
| Module type | Separate `nn.Linear` layers | Fused `nn.MultiheadAttention` |
| SVD target | `self_attn.out_proj` | `attn.out_proj` |
| Hidden sizes | 768, 1024 | 512 (projected), 768 internal |
| Training data | 400M image-text pairs | 12.8B DataComp samples |

## 3. SVD Residual Approach

### 3.1 Concept

Instead of fine-tuning all parameters, we decompose the output projection weight:

```
W_original = U @ Σ @ V^T    (SVD decomposition)

W_main = U[:, :r] @ Σ[:r] @ V^T[:r, :]    (frozen, top-r singular values)
W_residual = U_res @ Σ_res @ V_res^T       (trainable, remaining)

W_effective = W_main + W_residual
```

### 3.2 Implementation

**For HuggingFace CLIP (`apply_svd_residual_to_self_attn`):**
- Each `self_attn.out_proj` (nn.Linear) is replaced with `SVDResidualLinear`
- The `SVDResidualLinear` stores frozen `weight_main` and trainable residual components

**For OpenCLIP (`apply_svd_residual_to_openclip_attn`):**
- The `attn.out_proj` inside `nn.MultiheadAttention` is replaced with `SVDResidualLinear`
- A `weight` property provides dynamic weight computation for MHA compatibility

### 3.3 SVDResidualLinear Module

```python
class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        # Frozen main weight (top-r singular components)
        self.weight_main = nn.Parameter(weight_main, requires_grad=False)
        
        # Trainable residual components
        self.S_residual = nn.Parameter(S_residual, requires_grad=True)
        self.U_residual = nn.Parameter(U_residual, requires_grad=True)
        self.V_residual = nn.Parameter(V_residual, requires_grad=True)
        
        # Frozen singular vectors for orthogonality loss
        self.U_r = nn.Parameter(U_r, requires_grad=False)
        self.V_r = nn.Parameter(V_r, requires_grad=False)
    
    @property
    def weight(self):
        """Dynamic weight for nn.MultiheadAttention compatibility."""
        return self.compute_current_weight()
    
    def compute_current_weight(self):
        residual = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        return self.weight_main + residual
    
    def forward(self, x):
        return F.linear(x, self.compute_current_weight(), self.bias)
```

### 3.4 Regularization Losses

**Orthogonality Loss:**
```python
def compute_orthogonal_loss(self):
    # Ensure [U_r | U_residual] stays orthogonal
    UUT = cat([U_r, U_residual], dim=1) @ cat([U_r, U_residual], dim=1).T
    loss = ||UUT - I||_F  # Frobenius norm
    return loss
```

**Keep-Singular-Value Loss:**
```python
def compute_keepsv_loss(self):
    # Preserve overall weight norm
    W_current = weight_main + residual
    loss = |||W_current||_F^2 - ||W_original||_F^2|
    return loss
```

## 4. Classification Head

### 4.1 Standard Linear Head
```python
self.head = nn.Linear(hidden_size, 2)  # Binary classification
```

### 4.2 ArcFace Head (Recommended)
```python
class ArcMarginProduct(nn.Module):
    def forward(self, features, label=None):
        # Normalize features and weights
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        
        if label is None:
            return self.s * cosine  # Inference
        
        # Add angular margin to target class
        theta = torch.acos(cosine)
        margin_theta = torch.where(one_hot, theta + self.m, theta)
        
        return self.s * torch.cos(margin_theta)
```

**Why ArcFace?**
- Enforces angular margin between classes in feature space
- Better discrimination than softmax alone
- Scale parameter `s` controls prediction sharpness
- Margin `m` controls inter-class separation

## 5. Trainable Parameters

### 5.1 Parameter Counts

For ViT-B-16 with 12 attention layers:

| Component | Count | Trainable |
|-----------|-------|-----------|
| Backbone (total) | ~86M | ❌ Frozen |
| out_proj weights | 12 × 768 × 768 = 7M | ❌ Frozen (main) |
| Residual S | 12 × 1 = 12 | ✅ |
| Residual U | 12 × 768 × 1 = 9.2K | ✅ |
| Residual V | 12 × 1 × 768 = 9.2K | ✅ |
| ArcFace head | 768 × 2 = 1.5K | ✅ |
| **Total trainable** | ~20K | ✅ |

### 5.2 Verification

Run the test script to verify:
```bash
python scripts/test_svd_residual.py
```

Expected output:
```
✅ Replaced out_proj in 12 MultiheadAttention modules
✅ Backward pass: 36 residual params with gradients
✅ Parameter freezing: 37 trainable / 188 frozen
```

## 6. Forward Pass Summary

```
Input: [B, 3, 224, 224]
    ↓
CLIP Vision Encoder
├── Patch Embedding → [B, 197, 768]
├── Transformer Blocks ×12
│   ├── LayerNorm
│   ├── MultiheadAttention (with SVD out_proj)
│   ├── LayerNorm
│   └── MLP
└── Pooler (CLS token) → [B, 768]
    ↓
ArcFace Head
├── Normalize features
├── Cosine similarity with class weights
├── Add angular margin (training only)
└── Scale by s → [B, 2]
    ↓
Output: logits [B, 2], prob [B], features [B, 768]
```

## 7. Backbone Selection Guide

| Use Case | Recommended Backbone | Reason |
|----------|---------------------|--------|
| Baseline experiments | ViT-L-14 (OpenAI) | Largest, best pretrained |
| Fast iteration | ViT-B-16 (OpenAI) | 3x faster, good quality |
| Different pretraining | ViT-B-16 (LAION) | More diverse training data |
| Resource constrained | ViT-B-32 (OpenAI) | Fastest, larger patches |

---

*See also: [04_DATA_AND_AUGMENTATION.md](04_DATA_AND_AUGMENTATION.md) for data handling*
