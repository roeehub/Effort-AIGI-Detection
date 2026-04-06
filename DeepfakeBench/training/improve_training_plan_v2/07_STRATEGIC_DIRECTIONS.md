# Strategic Directions - Training Improvement Plan

**Created:** January 4, 2026  
**Status:** Active  
**Based on:** Experiment Analysis from Jan 4, 2026 (B16/B16-LAION/B32 comparison)

---

## Executive Summary

Our initial experiments show promising results (92% val accuracy with LAION B16) but reveal that **data quantity is the primary bottleneck**. This document outlines parallel work streams to systematically improve model performance toward SOTA+ deepfake detection.

### Current State
- ✅ EFFORT method (CLIP + SVD residual + ArcFace) is working
- ✅ LAION ViT-B-16 shows best performance on small dataset
- ⚠️ Only ~340 paired samples from single deepfake method
- ⚠️ Some training anomalies need investigation

### Target State
- 🎯 SOTA+ detection across multiple live deepfake methods
- 🎯 Robust to occlusions, compressions, real-world conditions
- 🎯 Validated on held-out methods (true generalization)

---

## Work Stream Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PARALLEL WORK STREAMS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  STREAM A   │    │  STREAM B   │    │  STREAM C   │    │  STREAM D   │  │
│  │    DATA     │    │  BACKBONE   │    │   ARCH      │    │  TRAINING   │  │
│  │  EXPANSION  │    │ COMPARISON  │    │ EXPERIMENTS │    │  STABILITY  │  │
│  ├─────────────┤    ├─────────────┤    ├─────────────┤    ├─────────────┤  │
│  │ Priority: 1 │    │ Priority: 2 │    │ Priority: 3 │    │ Priority: 2 │  │
│  │ ★★★★★      │    │ ★★★★☆      │    │ ★★★☆☆      │    │ ★★★★☆      │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                  │                  │                  │         │
│         ▼                  ▼                  ▼                  ▼         │
│  More DeepLive      ViT-L-14 test      Multi-center      Grad clipping    │
│  + DF40 pairing     + LAION L14        ArcFace           + LR tuning      │
│  + New methods      + Investigation    + Ensemble         + Diagnostics   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Stream A: Data Expansion ★★★★★

**Rationale:** ~340 samples is critically small. Even 2-3x data increase could significantly improve generalization.

### Task A1: Expand DeepLive Dataset
| Field | Value |
|-------|-------|
| **Goal** | Increase DeepLive pairs from ~340 to 1000+ |
| **Method** | Continue collection pipeline with existing method |
| **Effort** | Medium (pipeline exists) |
| **Impact** | High - direct improvement to current best setup |
| **Status** | ✅ **DONE** (Jan 8, 2026) - Expanded to ~740 pairs |

### Task A2: Add Second Deepfake Method
| Field | Value |
|-------|-------|
| **Goal** | Introduce diversity in training data |
| **Method** | Add another live deepfake method to collection |
| **Effort** | Medium-High |
| **Impact** | High - essential for generalization |
| **Status** | ✅ **DONE** (Jan 8, 2026) - Added ~8 new deepfake methods |

### Task A3: Investigate DF40 Pairing
| Field | Value |
|-------|-------|
| **Goal** | Determine if DF40 data can be paired |
| **Method** | Analyze DF40 structure, find real/fake correspondences |
| **Effort** | Low (investigation only) |
| **Impact** | Potentially High - large existing dataset |
| **Status** | ✅ **DONE** (Jan 8, 2026) - Created df40_paired data source with 5,379 pairs |

**Key Questions for A3:**
- [x] Does DF40 have metadata linking fake→real source? **YES - pair JSON created**
- [x] Can we extract frame-aligned pairs? **YES - 32 frames per pair**
- [x] What manipulation methods are in DF40? **8 methods: simswap, facedancer, blendface, e4s, faceswap, inswap, mobileswap, uniface**
- [x] Quality/resolution compatibility with DeepLive? **YES - both face-cropped**

### Task A4: Augmentation Expansion
| Field | Value |
|-------|-------|
| **Goal** | Increase effective data via augmentation |
| **Method** | Test higher occlusion prob (20%→40%), add compression aug |
| **Effort** | Low |
| **Impact** | Medium |
| **Status** | 🔲 Not Started |
| **Dependency** | After baseline stabilized |

---

## Stream B: Backbone Comparison ★★★★☆

**Rationale:** Need to establish which backbone is truly optimal before scaling data.

### Task B1: Run ViT-L-14 OpenAI Experiment
| Field | Value |
|-------|-------|
| **Goal** | Compare L-14 against B-16 family |
| **Method** | Create experiment config, run on same data |
| **Effort** | Low (config exists) |
| **Impact** | High - your original strong results were with L14 |
| **Status** | ✅ **DONE** (Jan 8, 2026) - Config created, experiment run & tested |
| **Deliverable** | `experiments/deeplive_vit_L14.yaml` |

### Task B2: Run ViT-L-14 LAION Experiment
| Field | Value |
|-------|-------|
| **Goal** | Test if LAION advantage scales to L-14 |
| **Method** | OpenCLIP ViT-L-14 with DataComp weights |
| **Effort** | Low |
| **Impact** | Medium-High |
| **Status** | 🔲 Not Started |
| **Dependency** | After B1 completes |

### Task B3: Investigate params_with_grad Drop
| Field | Value |
|-------|-------|
| **Goal** | Understand why LAION B16 params dropped 140→40 |
| **Method** | Add S_residual logging, inspect saved checkpoints |
| **Effort** | Medium |
| **Impact** | Medium - may reveal training bug |
| **Status** | 🔲 Not Started |
| **Deliverable** | Diagnostic logging code |

**Investigation Steps:**
1. [ ] Add W&B logging for `S_residual` min/max/mean per layer
2. [ ] Check if any S_residual values → 0 (dying gradients)
3. [ ] Compare checkpoint weights before/after the drop
4. [ ] Verify OpenCLIP SVD application is correct

---

## Stream C: Architecture Experiments ★★★☆☆

**Rationale:** Architecture changes should come after data/backbone decisions, but design can start now.

### Task C1: Design Multi-Center ArcFace
| Field | Value |
|-------|-------|
| **Goal** | Multiple embedding centers for different deepfake methods |
| **Method** | Extend ArcMarginProduct with K centers |
| **Effort** | Medium |
| **Impact** | Potentially High - better multi-method discrimination |
| **Status** | 🔲 Not Started |
| **Dependency** | Multiple methods in training data (A2) |

**Design Sketch:**
```python
class MultiCenterArcMarginProduct(nn.Module):
    def __init__(self, in_features, num_classes=2, num_centers=4, s=30.0, m=0.28):
        # num_centers: one per deepfake method + 1 for real
        self.centers = nn.Parameter(torch.randn(num_centers, in_features))
        # Route samples to appropriate center based on method label
```

### Task C2: Explore Ensemble Approach
| Field | Value |
|-------|-------|
| **Goal** | Combine predictions from multiple backbones |
| **Method** | Train LAION B16 + OpenAI L14, ensemble at inference |
| **Effort** | Low (once models trained) |
| **Impact** | Medium |
| **Status** | 🔲 Not Started |
| **Dependency** | B1 complete |

### Task C3: Temporal Modeling (Future)
| Field | Value |
|-------|-------|
| **Goal** | Leverage temporal consistency across frames |
| **Method** | Add temporal attention or consistency loss |
| **Effort** | High |
| **Impact** | Potentially High for video deepfakes |
| **Status** | 🔲 Not Started |
| **Dependency** | Baseline established |

---

## Stream D: Training Stability ★★★★☆

**Rationale:** Address observed anomalies before scaling experiments.

### Task D1: Add Gradient Clipping
| Field | Value |
|-------|-------|
| **Goal** | Prevent gradient spikes seen in B16 training |
| **Method** | Add `gradient_clip_val: 1.0` to configs |
| **Effort** | Very Low |
| **Impact** | Medium - more stable training |
| **Status** | ✅ **DONE** (Jan 4, 2026) |
| **Deliverable** | Updated experiment configs |

### Task D2: Learning Rate Tuning
| Field | Value |
|-------|-------|
| **Goal** | Optimize LR for small dataset |
| **Method** | Test 5e-5 vs 1e-4 with longer warmup |
| **Effort** | Low |
| **Impact** | Medium |
| **Status** | 🔲 Not Started |
| **Dependency** | After D1 |

### Task D3: Add Diagnostic Logging
| Field | Value |
|-------|-------|
| **Goal** | Better visibility into training dynamics |
| **Method** | Log SVD residual stats, per-class confidences |
| **Effort** | Medium |
| **Impact** | Medium - helps debugging |
| **Status** | ✅ **DONE** (Jan 4, 2026) |
| **Deliverable** | Enhanced logging code |

**Metrics Added:**
- [x] `svd/S_residual_min` - minimum singular value in residual
- [x] `svd/S_residual_max` - maximum singular value in residual  
- [x] `svd/S_residual_mean` - mean across all layers
- [x] `svd/layer_count` - number of SVD layers found
- [x] `svd/near_zero_layers` - count of layers with S_residual < 1e-6

---

## Execution Timeline

### Week 1 (Jan 4-10)

| Day | Tasks | Owner |
|-----|-------|-------|
| Jan 4 | Document conclusions (this doc) | ✅ Done |
| Jan 5-6 | B1: Create & run L14 experiment | |
| Jan 5-6 | D1: Add gradient clipping to configs | |
| Jan 7-8 | D3: Add diagnostic logging | |
| Jan 9-10 | A3: Investigate DF40 structure | |

### Week 2 (Jan 11-17)

| Day | Tasks | Owner |
|-----|-------|-------|
| Jan 11-12 | B3: Investigate params_with_grad drop | |
| Jan 13-14 | A1: Begin DeepLive expansion | |
| Jan 15-17 | B2: Run LAION L14 experiment | |

### Week 3+ (Jan 18+)

- A2: Add second deepfake method
- C1: Implement multi-center ArcFace (if needed)
- Full-scale experiments with expanded data

---

## Success Metrics

### Short-term (2 weeks)
- [ ] ViT-L-14 experiment complete with comparison
- [ ] Gradient clipping implemented
- [ ] DF40 pairing feasibility determined
- [ ] Diagnostic logging active

### Medium-term (1 month)
- [ ] 1000+ DeepLive pairs collected
- [ ] Second deepfake method integrated
- [ ] Val accuracy > 95% on expanded dataset

### Long-term (3 months)
- [ ] Multi-method training with >3 methods
- [ ] Held-out method generalization > 90%
- [ ] Paper-ready results

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| Jan 4, 2026 | Deprioritize ViT-B-32 | Patch size too coarse for deepfake artifacts |
| Jan 4, 2026 | Focus on data expansion first | 340 samples is primary bottleneck |
| Jan 4, 2026 | Keep LAION B16 as current best | 92% val acc, fast convergence |

---

## References

- `06_EXPERIMENT_ANALYSIS_JAN4.md` - Detailed analysis of current experiments
- `02_HYPERPARAMETERS.md` - Parameter reference
- `03_BACKBONES.md` - Backbone documentation
- Original EFFORT paper: https://arxiv.org/abs/2411.15633

---

*Last Updated: January 4, 2026*
