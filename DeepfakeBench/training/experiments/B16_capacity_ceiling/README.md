# B16 Capacity Ceiling Experiments

**Purpose:** Determine if ViT-B-16 architecture can achieve 0.95+ AUC on deepfake detection

**Key Question:** Is the limitation from:
1. Limited Effort SVD capacity (~74K params) - FIXABLE
2. B16's representation space - NOT FIXABLE (need different backbone)

---

## The Ceiling Test Logic

### Intuitive Explanation

Think of it like testing a student's potential:

| Scenario | What It Tests |
|----------|---------------|
| **Effort (k=1)** | "Can you pass using only a 3-page cheat sheet?" |
| **Full finetune** | "Can you pass if I give you unlimited study time?" |

If the student fails even with unlimited study time, the problem isn't the cheat sheet size - **they lack the foundational knowledge**.

For B16:
- If it can't reach 0.95 even with **all parameters trainable**, then B16's pretrained features simply don't contain the signal needed for this task
- If it CAN reach 0.95 with full finetune, then the signal exists - we just need to find a more efficient way to access it

### Architectural Explanation

The capacity ceiling test answers: **Do B16's pretrained CLIP features encode the information needed to distinguish real from fake faces?**

CLIP was trained on image-text pairs. Deepfake detection requires:
- Subtle compression artifacts
- GAN fingerprints / diffusion noise patterns  
- Inconsistent lighting/geometry around face boundaries

**If B16 can learn these with full finetune → the signal exists in the representation**
**If B16 can't learn even with full finetune → the signal was lost during pretraining**

---

## Experiment Design

We progressively unfreeze B16 to find its ceiling:

| Experiment | What's Trainable | Params ~approx | Purpose |
|------------|------------------|----------------|---------|
| `1_baseline_effort` | SVD residuals only (k=1) | ~74K | Current best (0.83 AUC) |
| `2_unfreeze_last_2` | Last 2 transformer blocks + head | ~25M | Check if late layers help |
| `3_unfreeze_last_4` | Last 4 transformer blocks + head | ~50M | More capacity |
| `4_full_finetune` | Entire backbone + head | ~86M | **THEORETICAL CEILING** |

---

## Key Metrics to Track

1. **Train AUC** - Can it memorize? (overfitting ceiling)
2. **Val AUC** - Can it generalize? (what we actually care about)
3. **Gap (Train - Val)** - Overfitting indicator

### Interpretation Guide

| Train AUC | Val AUC | Gap | Conclusion |
|-----------|---------|-----|------------|
| ≥0.99 | ≥0.95 | <0.05 | ✅ B16 is viable, find efficient method |
| ≥0.99 | 0.90-0.94 | 0.05-0.10 | ⚠️ B16 close, needs regularization or distillation |
| ≥0.99 | <0.88 | >0.10 | ⚠️ Severe overfitting, try different approach |
| <0.92 | <0.88 | <0.05 | ❌ B16 can't learn the task - features insufficient |

**The worst case** is if full finetune gets train=0.88, val=0.85. That means B16 literally cannot fit the training data, proving the features don't contain the signal.

---

## Launch Commands

```bash
# From DeepfakeBench/training directory
chmod +x launch_ceiling_experiment.sh

# 1. Baseline (uses regular train_sweep.py for fair comparison)
./launch_experiment.sh B16-capacity-ceiling asia-southeast1 experiments/B16_capacity_ceiling/1_baseline_effort.yaml

# 2. Unfreeze last 2 blocks (uses train_capacity_ceiling.py)
./launch_ceiling_experiment.sh 2

# 3. Unfreeze last 4 blocks  
./launch_ceiling_experiment.sh 3

# 4. Full finetune (THE CEILING TEST)
./launch_ceiling_experiment.sh 4

# Or run all ceiling experiments at once:
./launch_ceiling_experiment.sh all
```

---

## Expected Timeline

Each experiment runs for max 30-50 epochs with early stopping.
- Baseline: ~2-3 hours
- Unfreeze experiments: ~3-4 hours each (more params = slower)
- Full finetune: ~5-6 hours

**Total: ~15-18 hours to definitive answer**

---

## After Experiments

### If ceiling ≥ 0.95:
B16 IS VIABLE! Next steps:
1. Knowledge distillation from L14 teacher
2. Improved Effort with more capacity
3. ArcFace with lower margin on well-initialized model

### If ceiling 0.90-0.94:
B16 is CLOSE. Try:
1. Stronger augmentation to reduce overfitting gap
2. Distillation from L14
3. Different pretraining (OpenAI B16 vs LAION B16)

### If ceiling < 0.88:
B16 is NOT VIABLE for 0.95 target. Options:
1. Accept lower performance for deployment model
2. Use L14 for production (larger but works)
3. Investigate other efficient backbones (MobileViT, EfficientNet, etc.)

---

## Files Created

- `train_capacity_ceiling.py` - Dedicated training script (no SVD)
- `launch_ceiling_experiment.sh` - Launch script for GCP Vertex AI
- `1_baseline_effort.yaml` - Control experiment
- `2_unfreeze_last_2.yaml` - Medium capacity
- `3_unfreeze_last_4.yaml` - High capacity  
- `4_full_finetune.yaml` - THE CEILING TEST
- Updated `entrypoint.sh` with `ceiling` mode
