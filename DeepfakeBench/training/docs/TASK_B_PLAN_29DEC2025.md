# Task B: DeepLive Dataset Integration Plan

**Date:** December 29, 2025  
**Status:** � Implementation Phase  
**Predecessor:** Task A (Baseline Validation) - ✅ COMPLETED

---

## Executive Summary

Task B integrates the new DeepLive dataset (paired real/fake frames) into the training pipeline with:
1. **4 backbone variants**: ViT-L-14 (current), ViT-B-16, ViT-B-32, and LAION DataComp (future)
2. A new dataloader for the `live-deepfake-methods-real-and-fake-frames` GCS bucket
3. Landmark-based intelligent occlusion augmentations (image augmentations TBD)

The goal is to leverage the refactored modular architecture (Phases 1-3 complete) to add these features cleanly.

---

## User Specifications (Confirmed Dec 29)

| Aspect | Decision | Notes |
|--------|----------|-------|
| **Backbones** | 4 variants | ViT-L-14, ViT-B-16, ViT-B-32, LAION (future) |
| **Frame Sampling** | Sparse (8 anchors) | Modular for easy switching |
| **Batch Composition** | Paired real/fake | Config-controlled batch_size=32 |
| **Image Augmentations** | None for now | Easy to connect later |
| **Landmark Augmentations** | Intelligent occlusions | Yes, use landmarks |

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Component Checklist](#2-component-checklist)
3. [Backbone Configuration (4 Variants)](#3-backbone-configuration-4-variants)
4. [Implementation Plan](#4-implementation-plan)
5. [Integration Points](#5-integration-points)
6. [Testing Strategy](#6-testing-strategy)
7. [Progress Log](#7-progress-log)

---

## 1. Architecture Overview

### Current Refactored Structure (Leveraging)

```
training/
├── data/
│   ├── augmentations/
│   │   ├── registry.py       ← Register new "deeplive_v1" pipeline
│   │   ├── pipelines.py      ← Add deeplive augmentations here
│   │   └── transforms.py     ← Add landmark-based transforms if needed
│   ├── batching/
│   │   ├── factory.py        ← Register new "deeplive" strategy
│   │   ├── base.py           ← BatchingStrategy interface
│   │   └── [NEW] deeplive.py ← New batching strategy implementation
│   └── splitting/            ← May need to create deeplive splitter
├── detectors/
│   └── effort_detector.py    ← Modify for configurable backbone
└── dataset/
    └── [NEW] deeplive_dataset.py ← New dataset class
```

### Data Flow for DeepLive

```
┌─────────────────────────────────────────────────────────────────┐
│                    DeepLive GCS Bucket                          │
│  gs://live-deepfake-methods-real-and-fake-frames/samples/       │
├─────────────────────────────────────────────────────────────────┤
│  {sample_id}/                                                   │
│  ├── manifest.json          ← Sample metadata (completion marker│
│  ├── frames/real/*.png      ← 16 real frames (ground truth)     │
│  ├── frames/fake/*.png      ← 16 fake frames (deepfake)         │
│  └── landmarks/*.json       ← MediaPipe landmarks + blendshapes │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  DeepLiveDataset (NEW)                          │
│  - Discovers samples via manifest.json                          │
│  - Loads paired real/fake frames                                │
│  - Optionally loads landmarks                                   │
│  - Filters by strategy (edge_cases, etc.)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│            DeepLiveBatchingStrategy (NEW)                       │
│  - Creates DataLoader with paired sampling                      │
│  - Options: anchor-only, full 16, temporal pairs                │
│  - Applies DeepLive augmentations                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Model (Configurable Backbone)                   │
│  - Currently: CLIP ViT-L-14 (hardcoded)                        │
│  - Goal: Configurable via config (openclip, siglip, dinov2?)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Checklist

### 🔧 Phase B0: Bug Fixes (Prerequisite)
| ID | Task | Status | Notes |
|----|------|--------|-------|
| B0.1 | Fix ViT-B-32 hidden_size bug (512→768) | ⬜ | In `effort_detector.py` |

### 🎯 Phase B1: Backbone Configuration (4 Variants)
| ID | Task | Status | Notes |
|----|------|--------|-------|
| B1.1 | Download `openai/clip-vit-base-patch16` weights | ⬜ | HuggingFace |
| B1.2 | Download `openai/clip-vit-base-patch32` weights | ⬜ | HuggingFace |
| B1.3 | Create backbone registry config | ⬜ | `config/backbone_registry.yaml` |
| B1.4 | Update GCS asset download logic | ⬜ | Support multiple backbones |
| B1.5 | Test ViT-L-14 (existing) forward pass | ⬜ | Sanity check |
| B1.6 | Test ViT-B-16 forward pass | ⬜ | New backbone |
| B1.7 | Test ViT-B-32 forward pass | ⬜ | New backbone |
| B1.8 | (Future) Add OpenCLIP support for LAION DataComp | ⬜ | Different library |

### 📦 Phase B2: DeepLive Dataset
| ID | Task | Status | Notes |
|----|------|--------|-------|
| B2.1 | Create `DeepLiveSample` dataclass | ⬜ | Sample metadata structure |
| B2.2 | Create `DeepLiveDataset` class | ⬜ | `dataset/deeplive_dataset.py` |
| B2.3 | Implement manifest discovery from GCS | ⬜ | List all samples |
| B2.4 | Implement sparse frame loading (8 anchors) | ⬜ | Indices 0,2,4,6,8,10,12,14 |
| B2.5 | Implement paired real/fake loading | ⬜ | Same frame index |
| B2.6 | Implement landmark loading | ⬜ | For occlusion augmentation |
| B2.7 | Add strategy filtering (edge_cases, etc.) | ⬜ | Via manifest.json |
| B2.8 | Make frame_sampling mode configurable | ⬜ | sparse/full/pairs |
| B2.9 | Unit test with mock data | ⬜ | Local test |
| B2.10 | Integration test with real GCS data | ⬜ | Download subset |

### 🔄 Phase B3: DeepLive Batching Strategy
| ID | Task | Status | Notes |
|----|------|--------|-------|
| B3.1 | Create `DeepLiveBatchingStrategy` class | ⬜ | `data/batching/deeplive.py` |
| B3.2 | Implement paired sampling logic | ⬜ | Real+fake from same sample |
| B3.3 | Add configurable batch_size (default 32) | ⬜ | Config parameter |
| B3.4 | Create collate function for paired batches | ⬜ | Stack correctly |
| B3.5 | Register in `data/batching/factory.py` | ⬜ | Strategy name: "deeplive" |
| B3.6 | Support train/val split | ⬜ | Random or by strategy |
| B3.7 | Unit test batch structure | ⬜ | Verify shape, labels |

### 🎨 Phase B4: Landmark-Based Occlusion Augmentation
| ID | Task | Status | Notes |
|----|------|--------|-------|
| B4.1 | Create `LandmarkOcclusion` transform | ⬜ | `data/augmentations/transforms.py` |
| B4.2 | Implement eye region occlusion | ⬜ | Using landmark bbox |
| B4.3 | Implement mouth region occlusion | ⬜ | Using landmark bbox |
| B4.4 | Implement nose region occlusion | ⬜ | Using landmark bbox |
| B4.5 | Add configurable occlusion probability | ⬜ | Per-region control |
| B4.6 | Add occlusion methods (black, blur, noise) | ⬜ | Configurable |
| B4.7 | Create `deeplive_v1` augmentation pipeline | ⬜ | Landmarks only for now |
| B4.8 | Register in `data/augmentations/registry.py` | ⬜ | Easy to add image augs later |
| B4.9 | Visual unit test | ⬜ | Save sample images |

### ⚙️ Phase B5: Configuration & Integration
| ID | Task | Status | Notes |
|----|------|--------|-------|
| B5.1 | Create `config/deeplive_exp_vit_l_14.yaml` | ⬜ | Experiment 1 |
| B5.2 | Create `config/deeplive_exp_vit_b_16.yaml` | ⬜ | Experiment 2 |
| B5.3 | Create `config/deeplive_exp_vit_b_32.yaml` | ⬜ | Experiment 3 |
| B5.4 | Wire DeepLive data source in train_sweep.py | ⬜ | Support new bucket |
| B5.5 | Add data_source config section | ⬜ | bucket, type, strategies |
| B5.6 | End-to-end dry run (no GPU) | ⬜ | Verify pipeline |

### ✅ Phase B6: Verification & Launch
| ID | Task | Status | Notes |
|----|------|--------|-------|
| B6.1 | Local test with subset (100 samples) | ⬜ | Download and test |
| B6.2 | Short training run (100 steps) | ⬜ | Verify gradients flow |
| B6.3 | Launch Experiment 1: ViT-L-14 | ⬜ | Full run |
| B6.4 | Launch Experiment 2: ViT-B-16 | ⬜ | Full run |
| B6.5 | Launch Experiment 3: ViT-B-32 | ⬜ | Full run |
| B6.6 | (Future) Launch Experiment 4: LAION DataComp | ⬜ | After OpenCLIP support |

---

## 3. Backbone Configuration (4 Variants)

### Bug Fix Required

From `clip_backbone_compatibility.md`:

> ⚠️ **Bug Found**: `ViT-B-32` is incorrectly mapped to 512. The HuggingFace config shows the **vision model hidden_size is 768**, not 512.

**File:** `detectors/effort_detector.py`  
**Line:** ~255  
**Fix:**
```python
# Current (WRONG)
'ViT-B-32': 512,

# Fixed (CORRECT)
'ViT-B-32': 768,
```

### Backbone Specifications

| Variant | HuggingFace ID | Hidden Size | Rank | Resolution | Status |
|---------|----------------|-------------|------|------------|--------|
| ViT-L-14 | `openai/clip-vit-large-patch14` | 1024 | 1023 | 224 | ✅ Existing |
| ViT-B-16 | `openai/clip-vit-base-patch16` | 768 | 767 | 224 | 🔧 To add |
| ViT-B-32 | `openai/clip-vit-base-patch32` | 768 | 767 | 224 | 🔧 To add |
| DataComp | `laion/CLIP-ViT-B-16-DataComp.XL` | 768 | 767 | 224 | 📋 Future (OpenCLIP) |

### Config Examples

**Experiment 1: ViT-L-14 (Baseline)**
```yaml
backbone:
  type: clip
  variant: ViT-L-14
  source: openai
  hidden_size: 1024
  resolution: 224
rank: 1023
```

**Experiment 2: ViT-B-16**
```yaml
backbone:
  type: clip
  variant: ViT-B-16
  source: openai
  hidden_size: 768
  resolution: 224
rank: 767
```

**Experiment 3: ViT-B-32**
```yaml
backbone:
  type: clip
  variant: ViT-B-32
  source: openai
  hidden_size: 768
  resolution: 224
rank: 767
```

---

## 4. Implementation Plan

### Order of Implementation

```
B0 (Bug Fix) ─────────────────────────────────────────────┐
                                                          │
B1 (Backbone Setup) ──────────────────────────────────────┼──► B5 (Integration)
                                                          │
B2 (Dataset) ─────┬───────────────────────────────────────┤
                  │                                       │
B3 (Batching) ────┘                                       │
                                                          │
B4 (Landmark Occlusion) ──────────────────────────────────┘
```

**Parallelization:**
- B2 (Dataset) and B4 (Landmark Occlusion) can be developed in parallel
- B3 (Batching) depends on B2
- B5 (Integration) needs B1-B4 complete

### Recommended Execution Order

| Step | Phase | What | Why First |
|------|-------|------|-----------|
| 1 | **B0** | Fix ViT-B-32 hidden_size bug | Unblocks B1 |
| 2 | **B1.1-B1.3** | Download weights, create registry | Unblocks experiments |
| 3 | **B2.1-B2.5** | Core dataset implementation | Foundation |
| 4 | **B4.1-B4.6** | Landmark occlusion transforms | Can parallel with B2 |
| 5 | **B2.6-B2.8** | Dataset config options | After core works |
| 6 | **B3** | Batching strategy | Needs B2 |
| 7 | **B4.7-B4.8** | Register augmentation pipeline | After transforms |
| 8 | **B5** | Integration & configs | After B1-B4 |
| 9 | **B6** | Verification & launch | Final |

### Estimated Effort

| Phase | Estimated Time | Complexity |
|-------|----------------|------------|
| B0: Bug Fix | 5 min | Trivial |
| B1: Backbone Setup | 1-2 hours | Low |
| B2: Dataset | 3-4 hours | Medium |
| B3: Batching | 2-3 hours | Medium |
| B4: Landmark Occlusion | 3-4 hours | Medium |
| B5: Integration | 2-3 hours | Medium |
| B6: Verification | 2-4 hours | Low |

**Total:** ~15-20 hours

---

## 5. Data Configuration (Finalized)

### DeepLive Data Source
```yaml
deeplive_data:
  bucket_name: "live-deepfake-methods-real-and-fake-frames"
  
  # Frame sampling (sparse = 8 anchor frames)
  frame_sampling: "sparse"
  sparse_indices: [0, 2, 4, 6, 8, 10, 12, 14]
  
  # Strategy filtering (can be configured per experiment)
  strategies: "all"  # or ["edge_cases", "minimal_processing", ...]
  
  # Landmark loading (required for occlusion augmentation)
  use_landmarks: true
  
  # Train/val split
  train_val_split: 0.8
```

### Batching Configuration
```yaml
batching:
  strategy: "deeplive"
  batch_size: 32  # Configurable
  
  # Pairing mode
  pairing_mode: "paired"  # Real+fake from same sample
  
  # Per sample: 8 frames × 2 (real+fake) = 16 images per sample
  # With batch_size=32: 2 samples per batch → 32 images
  samples_per_batch: 2  # Derived from batch_size / (frames × 2)
```

### Augmentation Configuration
```yaml
augmentation:
  version: "deeplive_v1"
  
  # Image augmentations: NONE for now (easy to add later)
  use_compression: false
  use_blur: false
  use_noise: false
  use_color_jitter: false
  use_flip: true  # Usually harmless
  
  # Landmark-based augmentations: YES
  landmark_occlusion:
    enabled: true
    probability: 0.3  # 30% of images get occlusion
    regions:
      - name: "left_eye"
        probability: 0.3
        method: "gaussian_blur"  # or "black", "noise"
      - name: "right_eye"
        probability: 0.3
        method: "gaussian_blur"
      - name: "mouth"
        probability: 0.3
        method: "gaussian_blur"
      - name: "nose"
        probability: 0.2
        method: "gaussian_blur"
```

---

## 6. Integration Points

### Where New Code Hooks In

```python
# 1. Dataset (NEW file)
# dataset/deeplive_dataset.py
from dataset.deeplive_dataset import DeepLiveDataset, DeepLiveSample

# 2. Batching Factory (MODIFY existing)
# data/batching/factory.py
def get_batching_strategy(name: str, config: dict) -> BatchingStrategy:
    if name == "deeplive":
        from data.batching.deeplive import DeepLiveBatchingStrategy
        return DeepLiveBatchingStrategy(config)
    # ... existing strategies

# 3. Augmentation Registry (MODIFY existing)
# data/augmentations/registry.py
AUGMENTATION_REGISTRY = {
    # ... existing pipelines
    "deeplive_v1": create_deeplive_pipeline_v1,
}

# 4. Landmark Transforms (NEW in existing file)
# data/augmentations/transforms.py
class LandmarkOcclusion:
    """Occludes facial regions based on landmark bounding boxes."""
    ...

# 5. train_sweep.py (MODIFY)
# - Add handling for data_source_type == "deeplive"
# - Route to DeepLiveDataset + DeepLiveBatchingStrategy
```

### File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `detectors/effort_detector.py` | MODIFY | Fix ViT-B-32 hidden_size |
| `dataset/deeplive_dataset.py` | CREATE | DeepLiveDataset class |
| `data/batching/deeplive.py` | CREATE | DeepLiveBatchingStrategy |
| `data/batching/factory.py` | MODIFY | Register "deeplive" strategy |
| `data/augmentations/transforms.py` | MODIFY | Add LandmarkOcclusion |
| `data/augmentations/pipelines.py` | MODIFY | Add deeplive_v1 pipeline |
| `data/augmentations/registry.py` | MODIFY | Register deeplive_v1 |
| `config/backbone_registry.yaml` | CREATE | Backbone configs |
| `config/deeplive_exp_*.yaml` | CREATE | 3 experiment configs |
| `train_sweep.py` | MODIFY | Support deeplive data source |

---

## 7. Testing Strategy

### Unit Tests

```python
# tests/test_deeplive_dataset.py
def test_manifest_discovery():
    """Test that we can find all samples with manifest.json"""
    
def test_sparse_frame_loading():
    """Test loading 8 anchor frames (indices 0,2,4,6,8,10,12,14)"""
    
def test_paired_real_fake_loading():
    """Test that real/fake pairs have matching indices"""
    
def test_landmark_loading():
    """Test loading MediaPipe landmarks from JSON"""
    
def test_strategy_filtering():
    """Test filtering by edge_cases, minimal_processing, etc."""

# tests/test_deeplive_batching.py
def test_batch_size_32():
    """Test that batches have 32 images"""
    
def test_paired_sampling():
    """Test that real/fake from same sample are in batch"""
    
def test_label_balance():
    """Test 50/50 real/fake split in batch"""

# tests/test_landmark_occlusion.py
def test_eye_occlusion():
    """Test eye region occlusion using landmarks"""
    
def test_mouth_occlusion():
    """Test mouth region occlusion using landmarks"""
    
def test_occlusion_probability():
    """Test that occlusion respects probability config"""
```

### Integration Tests

```bash
# 1. Dry run (no GPU, verify pipeline)
python train_sweep.py \
  --param-config config/deeplive_exp_vit_l_14.yaml \
  --data.subset_percentage 0.01 \
  --training.max_steps 10 \
  --dry_run

# 2. Short run with each backbone
for backbone in vit_l_14 vit_b_16 vit_b_32; do
  python train_sweep.py \
    --param-config config/deeplive_exp_${backbone}.yaml \
    --data.subset_percentage 0.05 \
    --training.max_steps 100
done
```

---

## 8. Progress Log

| Date | Task ID | Activity | Outcome |
|------|---------|----------|---------|
| 2025-12-29 | - | Created planning document | This file |
| 2025-12-29 | - | Updated with user specifications | 4 backbones, sparse, paired, landmarks |
| 2025-12-29 | B0.1 | Fixed ViT-B-32 hidden_size bug | Changed 512→768 in effort_detector.py |
| 2025-12-29 | B1.1-B1.3 | Created backbone registry | config/backbone_registry.yaml |
| 2025-12-29 | B1.8 | Added OpenCLIP support | Modified build_backbone() in effort_detector.py |
| 2025-12-29 | B1.5-B1.7 | Created backbone test script | tests/test_backbone_variants.py |
| 2025-12-29 | B2 | Created DeepLive dataset | dataset/deeplive_dataset.py |
| 2025-12-29 | B3 | Created batching strategy | data/batching/deeplive.py |
| 2025-12-29 | B4 | Created landmark occlusions | data/augmentations/transforms.py |
| 2025-12-29 | B5 | Created experiment configs | experiments/deeplive_vit_*.yaml |
| 2025-12-29 | B6 | Verified all 4 backbones | All pass forward/backward tests |
| 2025-12-29 | B9 | Added SVD weight fusion | fuse_weights.py |

---

## 9. Implementation Status (Updated Dec 29, 2025)

### ✅ COMPLETED

| Component | Status | Files | Notes |
|-----------|--------|-------|-------|
| **B0: Bug Fix** | ✅ | `effort_detector.py` | ViT-B-32 hidden_size 512→768 |
| **B1: Backbones** | ✅ | `backbone_registry.yaml`, `effort_detector.py` | All 4 variants work |
| **B2: Dataset** | ✅ (code) | `dataset/deeplive_dataset.py` | Needs E2E testing |
| **B3: Batching** | ✅ (code) | `data/batching/deeplive.py`, `factory.py` | Needs E2E testing |
| **B4: Occlusions** | ✅ (code) | `transforms.py`, `pipelines.py`, `registry.py` | Needs E2E testing |
| **B5: Configs** | ✅ | `experiments/deeplive_vit_*.yaml` | 4 experiment files |
| **B6: Verification** | ✅ | `tests/test_backbone_variants.py` | All 4 backbones pass |
| **B9: SVD Fusion** | ✅ | `fuse_weights.py` | `fuse_openclip_svd_weights()` |

### ⚠️ NEEDS TESTING

| Component | What's Missing | How to Test |
|-----------|----------------|-------------|
| **DeepLive Dataset** | Real GCS data access | Connect to `gs://live-deepfake-methods-real-and-fake-frames/` |
| **Batching Strategy** | Integration with dataset | Run with actual samples |
| **Landmark Occlusions** | Real landmarks JSON | Test with actual MediaPipe output |
| **Full Pipeline** | End-to-end training | Short training run (100 steps) |

---

## 10. Next Steps for New Chat

### Priority 1: End-to-End Data Pipeline Test

**Goal:** Verify DeepLive dataset + batching + augmentations work together

```bash
# Test script to create:
python -m tests.test_deeplive_pipeline \
    --bucket gs://live-deepfake-methods-real-and-fake-frames \
    --num-samples 10 \
    --save-visualizations
```

**Tasks:**
1. Create `tests/test_deeplive_pipeline.py`
2. Test GCS access with `fsspec`/`gcsfs`
3. Verify sample discovery from manifest.json
4. Verify paired frame loading (real/fake)
5. Verify landmark JSON parsing
6. Verify augmentation application (visualize output)
7. Verify batch structure (shapes, labels)

### Priority 2: Short Training Run

**Goal:** Verify gradients flow through entire pipeline

```bash
python train_sweep.py \
    --config experiments/deeplive_vit_B16.yaml \
    --dry-run \
    --max-steps 100
```

**Tasks:**
1. Wire DeepLive data source in `train_sweep.py`
2. Add `deeplive` strategy selection
3. Run 100 steps, verify loss decreases
4. Check W&B logging works

### Priority 3: Launch Experiments

**Order of experiments:**
1. **ViT-B-16 (OpenAI)** — Smallest HuggingFace model, fastest iteration
2. **ViT-B-32 (OpenAI)** — Compare patch size effect
3. **ViT-L-14 (OpenAI)** — Compare to baseline
4. **ViT-B-16 (LAION DataComp)** — Compare training data effect

---

## 11. Key Technical Details

### OpenCLIP SVD Wrapping

The LAION DataComp backbone uses OpenCLIP which has `nn.MultiheadAttention`.
PyTorch's MHA uses functional API that accesses `out_proj.weight` directly.

**Solution:** `SVDMultiheadAttentionWrapper` in `effort_detector.py`:
- Wraps entire MHA module
- Zeros original `out_proj.weight`
- Applies SVD projection after MHA forward
- Trainable: `S_residual`, `U_residual`, `V_residual`

**Inference/Fusion:** Use `fuse_openclip_svd_weights()` in `fuse_weights.py`:
```python
from fuse_weights import fuse_openclip_svd_weights

# Option 1: Get fused state dict
fused_state_dict = fuse_openclip_svd_weights(model, return_state_dict=True)
torch.save(fused_state_dict, 'inference_checkpoint.pth')

# Option 2: In-place fusion
fuse_openclip_svd_weights(model)
# model now has standard weights
```

### GCS Bucket Paths (Production)

```
gs://base-checkpoints/effort-aigi/clip-vit-base-patch16/
gs://base-checkpoints/effort-aigi/clip-vit-base-patch32/
gs://base-checkpoints/effort-aigi/CLIP-ViT-B-16-DataComp.XL-s13B-b90K/
```

### Hidden Size Reference

| Backbone | Hidden Size | Rank | Notes |
|----------|-------------|------|-------|
| ViT-L-14 (OpenAI) | 1024 | 1023 | Baseline |
| ViT-B-16 (OpenAI) | 768 | 767 | |
| ViT-B-32 (OpenAI) | 768 | 767 | Larger patches |
| ViT-B-16 (LAION) | **512** | 511 | Output dim (not embed_dim!) |

---

## 12. Files Created/Modified (For PR)

### New Files
```
config/backbone_registry.yaml
dataset/deeplive_dataset.py
data/batching/deeplive.py
experiments/deeplive_vit_L14.yaml
experiments/deeplive_vit_B16.yaml
experiments/deeplive_vit_B32.yaml
experiments/deeplive_vit_B16_laion.yaml
tests/test_backbone_variants.py
```

### Modified Files
```
detectors/effort_detector.py      # OpenCLIP support, SVD wrapper, bug fix
data/batching/factory.py          # DeepLive strategy registration
data/augmentations/transforms.py  # Landmark occlusion classes
data/augmentations/pipelines.py   # Landmark occlusion pipeline
data/augmentations/registry.py    # Pipeline registration
data/augmentations/__init__.py    # Exports
fuse_weights.py                   # OpenCLIP SVD fusion
```

---

## 13. Quick Reference Commands

```bash
# Test all backbones
python tests/test_backbone_variants.py --variant all --skip-download

# Test specific backbone
python tests/test_backbone_variants.py --variant datacomp --skip-download

# Fuse OpenCLIP weights for inference
python fuse_weights.py \
    --checkpoint-gcs-path gs://path/to/checkpoint.pth \
    --config experiments/deeplive_vit_B16_laion.yaml \
    --openclip
```

---

## 10. File Locations (Final)

```
training/
├── data/
│   ├── augmentations/
│   │   ├── pipelines.py        # ADD: create_deeplive_pipeline_v1()
│   │   ├── registry.py         # ADD: "deeplive_v1" entry
│   │   └── transforms.py       # ADD: LandmarkOcclusion class
│   └── batching/
│       ├── deeplive.py         # NEW: DeepLiveBatchingStrategy
│       └── factory.py          # ADD: "deeplive" strategy
├── dataset/
│   └── deeplive_dataset.py     # NEW: DeepLiveDataset, DeepLiveSample
├── detectors/
│   └── effort_detector.py      # FIX: ViT-B-32 hidden_size
├── config/
│   ├── backbone_registry.yaml  # NEW: Backbone definitions
│   ├── deeplive_exp_vit_l_14.yaml  # NEW: Experiment 1
│   ├── deeplive_exp_vit_b_16.yaml  # NEW: Experiment 2
│   └── deeplive_exp_vit_b_32.yaml  # NEW: Experiment 3
└── docs/
    └── TASK_B_PLAN_29DEC2025.md    # This file
```
