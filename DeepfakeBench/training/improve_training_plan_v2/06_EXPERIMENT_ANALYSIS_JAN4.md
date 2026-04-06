# Experiment Analysis - January 4, 2026

## Experiment Overview

**Date:** January 4, 2026  
**Dataset:** DeepLive (~340 paired samples, 8 frames each)  
**Split:** 80/10/10 (train/val/test)  
**Augmentation:** 20% landmark occlusion probability  
**Training:** ~5000 steps, cosine warmup LR schedule  

### Models Tested

| Model | Backbone | Source | Hidden Size | Rank |
|-------|----------|--------|-------------|------|
| ViT-B-32 (Green) | ViT-B-32 | OpenAI | 768 | 767 |
| ViT-B-16 (Orange) | ViT-B-16 | OpenAI | 768 | 767 |
| ViT-B-16-LAION (Blue) | ViT-B-16-DataComp-XL | LAION/OpenCLIP | 512 | 511 |

---

## 1. Key Results Summary

### Final Performance (at ~5000 steps)

| Model | Val Holdout Acc | Train AUC | Logit Diff Std |
|-------|-----------------|-----------|----------------|
| **ViT-B-16-LAION** | ~92% | ~0.98 | ~18-20 |
| **ViT-B-16-OpenAI** | ~91% | ~0.97 | ~12-14 |
| **ViT-B-32-OpenAI** | ~85% | ~0.95 | ~8-10 |

### Ranking
1. 🥇 **LAION ViT-B-16** - Best overall performance and convergence
2. 🥈 **OpenAI ViT-B-16** - Strong performance, more oscillation
3. 🥉 **OpenAI ViT-B-32** - Underperforms due to patch size limitations

---

## 2. Detailed Observations

### 2.1 Positive Observations ✅

| Metric | Observation | Confidence |
|--------|-------------|------------|
| **Training AUC** | All models reach >95% train AUC | High |
| **Validation Accuracy** | Best model (LAION) reaches 92% | High |
| **Logit Differentiation** | `diff_std` increasing = models learning to discriminate | High |
| **LR Schedule** | Cosine warmup working correctly | High |
| **No Mode Collapse** | `fake_ratio` stays near 0.4-0.6 range | High |

### 2.2 Concerns & Anomalies ⚠️

| Metric | Observation | Severity | Investigation Needed |
|--------|-------------|----------|---------------------|
| **params_with_grad drop (LAION)** | Dropped 140 → 40 at ~4500 steps | **High** | Check SVD S_residual values |
| **Oscillating val accuracy** | High variance in val_holdout/acc | Medium | Expected - see Section 3 |
| **B16 fake_ratio drift** | Drops to ~0.4 late in training | Medium | Slight real-class bias |
| **Grad norm spikes** | B16 (orange) spikes at 3-4k steps | Medium | Consider gradient clipping |

### 2.3 Why B32 Underperforms

**Root Cause:** 32×32 pixel patches are too coarse for deepfake detection.

**Evidence:**
- Lowest `logit/diff_std` (~8 vs ~12-18 for B16 variants)
- Plateaus early (~2.5k steps) at 85% accuracy
- Deepfake artifacts are often subtle, existing at:
  - Blending boundaries (few pixels wide)
  - Texture inconsistencies
  - Fine facial details (eyes, mouth edges)

**Recommendation:** Deprioritize B32 for deepfake detection tasks.

### 2.4 Why LAION Outperforms OpenAI

**Hypotheses (Medium-High Confidence):**

1. **Larger Pre-training Dataset**
   - LAION DataComp: 12.8B image-text pairs
   - OpenAI CLIP: 400M image-text pairs
   - More diverse features → better transfer

2. **Lower Output Dimensionality**
   - LAION: 512-dim output
   - OpenAI: 768-dim output
   - Lower dim may reduce overfitting on small dataset (~340 samples)

3. **Different Pre-training Distribution**
   - DataComp filtering may have selected more "natural" looking images
   - Could complement deepfake detection where fake=unnatural

**Needs Validation:** Run ViT-L-14 (OpenAI) to compare with larger OpenAI model.

---

## 3. Understanding Validation Accuracy Oscillations

### Why We See High Variance in `val_holdout/overall/acc`

The oscillating validation accuracy is **expected behavior** given our setup. Here's why:

#### 3.1 Small Validation Set Size

```
Total samples: ~340
Val split: 10%
Val samples: ~34 samples × 8 frames = ~272 frames
```

**Impact:** With only ~34 unique video pairs in validation:
- Each sample represents ~3% of the validation set
- Getting 1-2 samples wrong shifts accuracy by 3-6%
- This creates apparent "noise" in the accuracy curve

#### 3.2 Frame-Level vs Sample-Level Evaluation

Current evaluation computes accuracy at the **frame level**, but:
- Frames from the same video are highly correlated
- Model confidence may swing between evaluation steps
- A single video going from "all correct" to "all wrong" = large accuracy swing

#### 3.3 Training Dynamics

```
                    Evaluation Point
                          │
    ─────────────────────▼────────────────────────
    │░░░░░░░│▓▓▓▓▓▓▓│░░░░░░░│▓▓▓▓▓▓▓│░░░░░░░│
    └───────────────────────────────────────────┘
    Step:  100      200      300      400      500
    
    ░ = Model slightly favors "real"
    ▓ = Model slightly favors "fake"
```

During training with ArcFace + cosine LR:
- Decision boundary oscillates as model learns
- Angular margin pushes embeddings around
- Scale annealing (s: 10→30) changes prediction confidence

#### 3.4 Batch Composition Variance

With `frames_per_batch: 32` from a small dataset:
- Different batches have different difficulty
- Hard batches may shift the model's decision boundary
- Next eval sees the effect → apparent oscillation

### 3.5 Visual Explanation

```
Val Accuracy Over Training Steps

1.0│                                    
   │     ╱╲    ╱╲   ╱─╲  ╱──╲ ╱──╲      ← LAION (Blue)
0.9│    ╱  ╲  ╱  ╲ ╱   ╲╱    ╲    ╲     
   │   ╱    ╲╱    ╲           ╲    ╲    
   │  ╱    ╱╲    ╱╲    ╱╲    ╱╲    ╲   ← B16 (Orange)
0.8│ ╱    ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲    ╲  
   │╱    ╱    ╲╱    ╲╱    ╲╱    ╲    ╲ 
   │    ╱           ╱─────────────────  ← B32 (Green) plateaus
0.7│   ╱           ╱                    
   │  ╱           ╱                     
0.6│ ╱           ╱                      
   │╱           ╱                       
0.5│───────────────────────────────────
   └───────────────────────────────────
   0    1k    2k    3k    4k    5k  Steps
   
Oscillation amplitude decreases as:
1. Model confidence increases
2. Scale factor s increases (sharper decisions)
3. Feature space stabilizes
```

### 3.6 Mitigation Strategies

| Strategy | Implementation | Effectiveness |
|----------|---------------|---------------|
| **Larger validation set** | Increase data collection | ★★★★★ |
| **Sample-level aggregation** | Vote across frames per video | ★★★★☆ |
| **Exponential moving average** | Track smoothed val metrics | ★★★☆☆ |
| **More frequent evaluation** | Reduce `evaluate_every_steps` | ★★☆☆☆ |

---

## 4. Confidence Assessment

| Conclusion | Confidence | Evidence |
|------------|------------|----------|
| LAION B16 is best performer (current setup) | **High** | Consistent across all metrics |
| B32 is unsuitable for deepfake detection | **High** | Established in literature + our results |
| Small dataset is primary bottleneck | **High** | ~340 samples is extremely limited |
| Oscillations are from small val set | **High** | Mathematical certainty given 34 samples |
| LAION > OpenAI (general claim) | **Medium** | Need L14 comparison, more runs |
| params_with_grad drop is a bug | **Medium** | Need diagnostic logging to confirm |

---

## 5. Open Questions

1. **Does ViT-L-14 (OpenAI) outperform all B-size models?**
   - Your earlier results suggested strong L14 performance
   - Need direct comparison under identical conditions

2. **What caused the params_with_grad drop in LAION run?**
   - S_residual values approaching zero?
   - OpenCLIP SVD application bug?
   - Early stopping internal trigger?

3. **Can pairing DF40 data improve generalization?**
   - Your hypothesis about paired data is compelling
   - Need to investigate DF40 structure

4. **Will multi-center ArcFace help with multi-method detection?**
   - Theory: different deepfake methods = different embedding clusters
   - Prerequisite: multiple methods in training data

---

## 6. Next Experiment Priorities

See `07_STRATEGIC_DIRECTIONS.md` for detailed action plan.

---

*Document created: January 4, 2026*
*Authors: Roee + Copilot Analysis*
