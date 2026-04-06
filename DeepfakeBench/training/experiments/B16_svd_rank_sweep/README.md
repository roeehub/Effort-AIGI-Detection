# B16 SVD Rank Sweep Experiments

**Date:** January 15, 2026  
**Goal:** Find optimal SVD capacity that achieves ≥0.95 AUC (ideally 0.98)  
**Reference:** B16_BREAKTHROUGH_RESULTS.md

---

## Background

The full finetune experiment proved B16 can achieve **0.9827 AUC** (comparable to L14).
This confirms the bottleneck was **SVD capacity**, not backbone capacity.

**Key Insight:** The standard Effort config uses k=1 trainable singular direction per layer 
(~74K params total), which is insufficient to reshape embedding geometry for this task.

---

## Experiment Pack A: SVD Rank Sweep

Test increasing numbers of trainable singular directions:

| Experiment | k (trainable dirs) | Approx Params | Learning Rate | Rationale |
|------------|-------------------|---------------|---------------|-----------|
| `rank_k1`  | 1 (baseline)      | ~74K          | 2e-4          | Control   |
| `rank_k2`  | 2                 | ~148K         | 2e-4          | 2x capacity |
| `rank_k4`  | 4                 | ~296K         | 1e-4          | 4x capacity |
| `rank_k8`  | 8                 | ~592K         | 1e-4          | 8x capacity |
| `rank_k16` | 16                | ~1.18M        | 1e-4          | 16x capacity |

**Notes:**
- Lower LR for higher ranks to prevent overshoot
- All use CrossEntropyLoss (no ArcFace) - simple loss first
- Gradient clipping at 1.0 for stability

---

## Experiment Pack B: Late Layers Only

Apply SVD adapters only to final transformer blocks:

| Experiment | Blocks | k | Approx Params | Hypothesis |
|------------|--------|---|---------------|------------|
| `rank_k8_late_3_blocks` | [9, 10, 11] | 8 | ~148K | Final layers most task-specific |
| `rank_k8_late_5_blocks` | [7, 8, 9, 10, 11] | 8 | ~247K | More capacity, still efficient |

**✅ svd_blocks support implemented** - can run Pack B in parallel with Pack A.

---

## How to Launch

```bash
cd DeepfakeBench/training

# Launch individual experiments
./launch_experiment.sh B16-svd-sweep asia-southeast1 experiments/B16_svd_rank_sweep/rank_k1_baseline.yaml
./launch_experiment.sh B16-svd-sweep asia-southeast1 experiments/B16_svd_rank_sweep/rank_k2.yaml
./launch_experiment.sh B16-svd-sweep asia-southeast1 experiments/B16_svd_rank_sweep/rank_k4.yaml
./launch_experiment.sh B16-svd-sweep asia-southeast1 experiments/B16_svd_rank_sweep/rank_k8.yaml
./launch_experiment.sh B16-svd-sweep asia-southeast1 experiments/B16_svd_rank_sweep/rank_k16.yaml
```

---

## Success Metrics

- **Minimum Success:** ≥0.95 AUC with stable training
- **Target Success:** ≥0.98 AUC with <1M trainable params
- **Stretch Goal:** ≥0.98 AUC with <500K trainable params

---

## Configuration Notes

### SVD Rank vs k (trainable directions)

For embed_dim=768 (B16 internal dimension):
- `rank` parameter = number of singular values to FREEZE (top-r)
- `k` = embed_dim - rank = number of TRAINABLE directions

| k (trainable) | rank (frozen) | Formula |
|---------------|---------------|---------|
| 1 | 767 | 768 - 1 |
| 2 | 766 | 768 - 2 |
| 4 | 764 | 768 - 4 |
| 8 | 760 | 768 - 8 |
| 16 | 752 | 768 - 16 |

### Parameter Count Estimation

For B16 with full SVD coverage (48 layers: 12 blocks × 4 projections):
- Per layer: ~(2 × embed_dim × k) + k ≈ 2 × 768 × k params
- Total: 48 × 2 × 768 × k = 73,728 × k params

| k | Approx Total Params |
|---|---------------------|
| 1 | ~74K |
| 2 | ~148K |
| 4 | ~296K |
| 8 | ~592K |
| 16 | ~1.18M |
