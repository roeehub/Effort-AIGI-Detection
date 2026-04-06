# ArcFace Scale Ablation Experiments

**Created:** January 10, 2026  
**Purpose:** Investigate optimal ArcFace scale (s) to prevent training collapse

## Background

Training runs with `s_end: 30` consistently collapse around steps 2800-3500, where:
- Logit std drops to 0.0000 (constant predictions)
- AUC drops from ~0.66 to ~0.33 (random)
- Model outputs constant ~0.5 probability for all inputs

**Hypothesis:** ArcFace scale annealing to high values (>20) causes gradient instability
leading to model collapse into degenerate constant-output solution.

## Key Parameter Changes (Applied to All Configs)

### 1. Schedule Alignment
```yaml
# OLD: Misaligned schedules
lr_scheduler_warmup_steps: 1200   # LR peaks at step 1200
anneal_steps: 4000                 # ArcFace still climbing

# NEW: Aligned schedules  
lr_scheduler_warmup_steps: 4000   # LR and ArcFace both ramp together
anneal_steps: 4000                 # Both reach target at same step
```

**Why:** When LR is at peak while ArcFace scale is still climbing → double gradient amplification → collapse.

### 2. More Granular Evaluation
```yaml
# OLD
evaluate_every_steps: 500

# NEW  
evaluate_every_steps: 250   # Catch collapse earlier
```

## Experiment Matrix

| Backbone | Variant | s_end | Risk Level |
|----------|---------|-------|------------|
| ViT-B-16 LAION | s12_conservative | 12 | ✅ Safest |
| ViT-B-16 LAION | s15_safe | 15 | ✅ Safe |
| ViT-B-16 LAION | s18_risky | 18 | ⚠️ Boundary |
| ViT-B-16 LAION | s30_baseline | 30 | ❌ Control |
| ViT-L-14 OpenAI | s12_conservative | 12 | ✅ Safest |
| ViT-L-14 OpenAI | s15_safe | 15 | ✅ Safe |
| ViT-L-14 OpenAI | s18_risky | 18 | ⚠️ Boundary |
| ViT-L-14 OpenAI | s30_baseline | 30 | ❌ Control |

## Recommended Run Order

**Priority 1 - Should succeed:**
```bash
./launch_experiment.sh arcface-scale-ablation asia-southeast1 experiments/arcface_scale_ablation/vit_B16_laion_s15_safe.yaml
./launch_experiment.sh arcface-scale-ablation asia-southeast1 experiments/arcface_scale_ablation/vit_L14_openai_s15_safe.yaml
```

**Priority 2 - Test boundaries:**
```bash
./launch_experiment.sh arcface-scale-ablation asia-southeast1 experiments/arcface_scale_ablation/vit_B16_laion_s18_risky.yaml
./launch_experiment.sh arcface-scale-ablation asia-southeast1 experiments/arcface_scale_ablation/vit_L14_openai_s18_risky.yaml
```

**Priority 3 - Safest fallback:**
```bash
./launch_experiment.sh arcface-scale-ablation asia-southeast1 experiments/arcface_scale_ablation/vit_B16_laion_s12_conservative.yaml
./launch_experiment.sh arcface-scale-ablation asia-southeast1 experiments/arcface_scale_ablation/vit_L14_openai_s12_conservative.yaml
```

## Metrics to Watch

- `logit_std`: Should stay > 0.1 throughout training
- `pred_fake_std`: Should stay > 0.01 (not constant)
- `val_auc`: Should improve or stay stable, NOT drop to 0.33
- `collapse_warning`: New diagnostic (added Jan 10), should stay False
- `arcface/current_s`: Track scale progression

## Files

- `vit_B16_laion_s12_conservative.yaml` - B16 safest (s_end=12)
- `vit_B16_laion_s15_safe.yaml` - B16 safe (s_end=15)
- `vit_B16_laion_s18_risky.yaml` - B16 boundary test (s_end=18)
- `vit_B16_laion_s30_baseline.yaml` - B16 baseline/control (s_end=30)
- `vit_L14_openai_s12_conservative.yaml` - L14 safest (s_end=12)
- `vit_L14_openai_s15_safe.yaml` - L14 safe (s_end=15)
- `vit_L14_openai_s18_risky.yaml` - L14 boundary test (s_end=18)
- `vit_L14_openai_s30_baseline.yaml` - L14 baseline/control (s_end=30)
