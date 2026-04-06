# CLIP Backbone Compatibility Report for SVD Decomposition

**Date:** December 29, 2025  
**Purpose:** Evaluate compatibility of alternative CLIP backbones with the Effort detector's SVD decomposition approach

---

## Overview

The Effort detector uses SVD decomposition on CLIP's vision transformer attention layers as a unique training strategy. The original paper and current implementation use **OpenAI's CLIP ViT-L/14**. This report evaluates the feasibility of using smaller/alternative CLIP backbones.

---

## Candidate Models

| Model | HuggingFace ID | Source |
|-------|----------------|--------|
| Current | `openai/clip-vit-large-patch14` | OpenAI |
| Candidate 1 | `openai/clip-vit-base-patch16` | OpenAI |
| Candidate 2 | `openai/clip-vit-base-patch32` | OpenAI |
| Candidate 3 | `laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K` | LAION/OpenCLIP |

---

## Architecture Comparison

| Model | Vision Hidden Size | Patch Size | Layers | Attention Heads | Image Size | Library |
|-------|-------------------|------------|--------|-----------------|------------|---------|
| **clip-vit-large-patch14** (current) | **1024** | 14 | 24 | 16 | 224 | `transformers` |
| clip-vit-base-patch16 | **768** | 16 | 12 | 12 | 224 | `transformers` |
| clip-vit-base-patch32 | **768** | 32 | 12 | 12 | 224 | `transformers` |
| CLIP-ViT-B-16-DataComp.XL | **768** | 16 | 12 | 12 | 224 | **`open_clip`** |

### Key Architectural Differences

1. **Hidden Size**: ViT-L uses 1024, ViT-B uses 768
2. **Depth**: ViT-L has 24 layers, ViT-B has 12 layers
3. **Attention Heads**: ViT-L has 16 heads, ViT-B has 12 heads
4. **Parameters**: ViT-L ~304M (vision), ViT-B ~86M (vision)

---

## SVD Decomposition Compatibility

### How SVD Works in Effort

The `apply_svd_residual_to_self_attn()` function:
1. Finds all `nn.Linear` layers within `self_attn` modules
2. Performs SVD decomposition: `W = U @ diag(S) @ V`
3. Keeps top `r` singular values fixed (main weight)
4. Makes remaining singular values trainable (residual)

### Compatibility Assessment

| Model | SVD Compatible | Notes |
|-------|---------------|-------|
| clip-vit-base-patch16 | ✅ Yes | Same transformers API, different dimensions |
| clip-vit-base-patch32 | ✅ Yes | Same transformers API, different dimensions |
| CLIP-ViT-B-16-DataComp.XL | ⚠️ Requires Changes | Different library (OpenCLIP) |

The SVD decomposition is **architecture-agnostic** - it operates on standard `nn.Linear` layers. The key consideration is the **rank parameter** which should be adjusted based on hidden size.

---

## Detailed Analysis

### ✅ OpenAI Base Models (clip-vit-base-patch16, clip-vit-base-patch32)

**Status: Should work with configuration changes only**

These models use the same HuggingFace `transformers` library, so `CLIPModel.from_pretrained()` works identically.

#### Required Changes

1. **Hidden Size for Head Layer**
   - Current: 1024 (for ViT-L-14)
   - Required: 768 (for ViT-B models)
   - The head layer (`nn.Linear` or `ArcMarginProduct`) must match

2. **Rank Parameter**
   - Current: `rank: 1023` (for 1024 hidden size)
   - Recommended: `rank: 767` or lower (for 768 hidden size)
   - Formula: `rank = hidden_size - 1` for maximum rank

3. **Attention Layer Dimensions**
   - ViT-L-14: Q/K/V projections are 1024×1024
   - ViT-B-16/32: Q/K/V projections are 768×768

#### Code Support (Already Exists)

The `_get_hidden_size()` method in `effort_detector.py` already handles variant mapping:

```python
VARIANT_HIDDEN_SIZES = {
    'ViT-B-16': 768,
    'ViT-B-32': 512,  # ⚠️ BUG: Should be 768
    'ViT-L-14': 1024,
    'ViT-L-14-336': 1024,
    'ViT-H-14': 1280,
    'ViT-G-14': 1664,
}
```

> ⚠️ **Bug Found**: `ViT-B-32` is incorrectly mapped to 512. The HuggingFace config shows the **vision model hidden_size is 768**, not 512. The 512 value is the `projection_dim`, not `hidden_size`.

---

### ⚠️ LAION DataComp Model (CLIP-ViT-B-16-DataComp.XL)

**Status: Requires code changes**

This model uses the **OpenCLIP library**, not HuggingFace transformers.

#### Incompatibility Details

| Aspect | Current Code | LAION Model |
|--------|--------------|-------------|
| Library | `transformers` | `open_clip` |
| Loading | `CLIPModel.from_pretrained()` | `open_clip.create_model_and_transforms()` |
| Weight Format | HuggingFace format | OpenCLIP format |
| Module Names | `vision_model.encoder.layers[i].self_attn` | May differ |

#### Required Changes for OpenCLIP Support

1. **Add dependency**: `pip install open_clip_torch`

2. **Modify `build_backbone()`** to support OpenCLIP:

```python
def build_backbone(self, config):
    backbone_config = config.get('backbone', {})
    source = backbone_config.get('source', 'openai')
    
    if source == 'openclip':
        import open_clip
        model_name = backbone_config.get('openclip_model', 'ViT-B-16')
        pretrained = backbone_config.get('openclip_pretrained', 'datacomp_xl_s13b_b90k')
        
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        vision_model = model.visual
        
        # Apply SVD (verify module structure matches)
        vision_model = apply_svd_residual_to_self_attn(vision_model, r=self.rank)
        return vision_model
    else:
        # Existing HuggingFace loading path
        clip_model = CLIPModel.from_pretrained(self.clip_backbone_path, local_files_only=True)
        clip_model.vision_model = apply_svd_residual_to_self_attn(clip_model.vision_model, r=self.rank)
        return clip_model.vision_model
```

3. **Verify module naming**: OpenCLIP may use different names for attention modules. Test with:
```python
for name, module in vision_model.named_modules():
    if 'attn' in name.lower():
        print(name, type(module))
```

---

## Configuration Examples

### For clip-vit-base-patch16

```yaml
# In experiment config
backbone:
  type: clip
  variant: 'ViT-B-16'
  source: openai
  hidden_size: 768
  resolution: 224

gcs_assets:
  clip_backbone:
    local_path: "./weights/models--openai--clip-vit-base-patch16/"

rank: 767  # Adjusted for 768 hidden size
```

### For clip-vit-base-patch32

```yaml
backbone:
  type: clip
  variant: 'ViT-B-32'
  source: openai
  hidden_size: 768  # NOT 512!
  resolution: 224

gcs_assets:
  clip_backbone:
    local_path: "./weights/models--openai--clip-vit-base-patch32/"

rank: 767
```

### For LAION DataComp (Future)

```yaml
backbone:
  type: clip
  variant: 'ViT-B-16'
  source: openclip
  openclip_model: 'ViT-B-16'
  openclip_pretrained: 'datacomp_xl_s13b_b90k'
  hidden_size: 768
  resolution: 224

rank: 767
```

---

## Action Items

### Immediate (Low Effort)

- [ ] **Fix Bug**: Correct `ViT-B-32` hidden size from 512 to 768 in `effort_detector.py`
- [ ] Download `openai/clip-vit-base-patch16` weights
- [ ] Create config for ViT-B-16 experiments
- [ ] Run baseline comparison experiment

### Future (Medium Effort)

- [ ] Add OpenCLIP support to `build_backbone()`
- [ ] Test module structure compatibility for OpenCLIP models
- [ ] Download and test LAION DataComp model

---

## Recommendations

### Start With: `openai/clip-vit-base-patch16`

**Rationale:**
1. **Minimal code changes** - Just configuration
2. **Smaller model** - ~3.5x fewer parameters, faster iteration
3. **Same library** - Uses familiar HuggingFace transformers
4. **Good baseline** - Well-documented, widely used

### Benefits of Smaller Backbones

1. **Faster training** - Fewer parameters to process
2. **Lower memory** - Can use larger batch sizes
3. **Quick experiments** - Test hypotheses faster
4. **Deployment** - Smaller models are easier to deploy

### Potential Tradeoffs

1. **Capacity** - May have lower representational capacity
2. **Performance** - May achieve slightly lower accuracy
3. **Generalization** - Needs empirical testing

---

## References

- [OpenAI CLIP Paper](https://arxiv.org/abs/2103.00020)
- [HuggingFace clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
- [HuggingFace clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16)
- [HuggingFace clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- [LAION CLIP-ViT-B-16-DataComp.XL](https://huggingface.co/laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K)
- [OpenCLIP GitHub](https://github.com/mlfoundations/open_clip)
- [DataComp Paper](https://arxiv.org/abs/2304.14108)
