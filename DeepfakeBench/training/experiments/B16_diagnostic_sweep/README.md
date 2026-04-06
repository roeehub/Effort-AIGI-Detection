# B16 Diagnostic Sweep - January 15, 2026

## Purpose

After the initial SVD rank sweep showed **no capacity scaling** (k=1 performed as well as k=16), this diagnostic sweep identifies the **actual bottleneck**.

## Background

Initial sweep results at ~5k steps:

| Experiment | k | Best AUC | Notes |
|------------|---|----------|-------|
| k=1 @ 2e-4 | 1 | **0.8203** | Best performer! |
| k=2 @ 2e-4 | 2 | 0.8200 | Same as k=1 |
| k=4 @ 1e-4 | 4 | 0.8121 | Dropped |
| k=8 @ 1e-4 | 8 | 0.8123 | Same plateau |
| k=16 @ 1e-4 | 16 | 0.8120 | No improvement |

**Problem**: 16x more trainable params → 0.8% WORSE AUC

This is NOT expected behavior for capacity scaling. Something else is limiting.

## Hypotheses

### H1: LR Confound
k=1,2 used LR=2e-4 while k≥4 used LR=1e-4. The "capacity doesn't help" conclusion may be LR-driven.

### H2: Constraint Dominance
The orthogonal regularization (`lambda_reg=1.0`) may be "pinning" the solution, preventing extra capacity from being used.

### H3: Wrong Control Surface
SVD on attention projections may not be the right modules to adapt. Full finetune winning with CE suggests B16 can solve it, but via different pathways (MLP, final projection, LN).

## Experiments

### Set A: LR Confound Elimination

| ID | Config | Change | Expected Outcome |
|----|--------|--------|------------------|
| A1 | `A1_k1_lr1e4.yaml` | k=1 @ LR=1e-4 | If drops to ~0.81, LR explains k=1 superiority |
| A2 | `A2_k8_lr2e4.yaml` | k=8 @ LR=2e-4 | If rises to ~0.82, LR explains k≥4 inferiority |

### Set B: Constraint Dominance

| ID | Config | Change | Expected Outcome |
|----|--------|--------|------------------|
| B1 | `B1_k8_lambda0.01.yaml` | k=8 @ λ=0.01 | If >0.85, constraint is bottleneck |
| B2 | `B2_k8_lambda0.yaml` | k=8 @ λ=0 | If >0.85 (may be unstable), constraint is bottleneck |

### Set C: Control Surface

| ID | Config | Change | Expected Outcome |
|----|--------|--------|------------------|
| C1 | `C1_k4_late3_unfreeze_proj_ln.yaml` | +unfreeze proj/LN | If >0.85, final layers matter |
| C2 | `C2_k4_late3_svd_mlp.yaml` | +SVD on MLP | If >0.85, MLP pathway matters |
| C3 | `C3_k4_late3_both.yaml` | +proj/LN +MLP | Combined max signal |

## Decision Tree

```
After ~5k steps, check results:

┌─ A1 drops to ~0.81 AND A2 rises to ~0.82?
│  └─ YES: LR was the confound. Re-run rank sweep with consistent LR.
│  └─ NO: Continue to B tests
│
├─ B1 or B2 > 0.85?
│  └─ YES: Constraint is bottleneck. 
│          → Reduce lambda_reg for production
│          → Consider annealing lambda_reg during training
│  └─ NO: Continue to C tests
│
├─ C1, C2, or C3 > 0.85?
│  └─ YES: Control surface is bottleneck.
│          → Use winning config as new baseline
│          → Consider LoRA/Adapters on MLP as alternative to SVD
│  └─ NO: None of these help
│
└─ All still ~0.82?
   └─ Data/architecture bottleneck:
      → Try distillation from full-finetune teacher
      → Try larger dataset
      → Consider different approach entirely
```

## Launch Commands

```bash
cd /path/to/DeepfakeBench/training

# Launch all 7 experiments
./launch_diagnostic_sweep.sh all asia-southeast1 B16-diagnostic

# Or launch by set
./launch_diagnostic_sweep.sh A asia-southeast1 B16-diagnostic
./launch_diagnostic_sweep.sh B asia-southeast1 B16-diagnostic
./launch_diagnostic_sweep.sh C asia-southeast1 B16-diagnostic

# Or individual experiments
./launch_diagnostic_sweep.sh A1 asia-southeast1 B16-diagnostic
```

## Code Changes Made

1. **`effort_detector.py`**: Added support for:
   - `backbone.apply_svd_to_mlp`: Apply SVD to MLP layers (c_fc, c_proj)
   - `backbone.unfreeze_final_proj`: Unfreeze visual.proj (768→512)
   - `backbone.unfreeze_final_ln`: Unfreeze visual.ln_post

## Files Created

```
experiments/B16_diagnostic_sweep/
├── A1_k1_lr1e4.yaml
├── A2_k8_lr2e4.yaml
├── B1_k8_lambda0.01.yaml
├── B2_k8_lambda0.yaml
├── C1_k4_late3_unfreeze_proj_ln.yaml
├── C2_k4_late3_svd_mlp.yaml
├── C3_k4_late3_both.yaml
└── README.md (this file)

launch_diagnostic_sweep.sh
```

## Target Metrics

- **Minimum Success**: Any experiment breaks the 0.82 plateau (>0.85 AUC)
- **Optimal Outcome**: Identify clear bottleneck with minimal params
- **Reference**: Full finetune achieved 0.9827 AUC on same data

## Timeline

- Experiments launched: Jan 15, 2026
- Expected results: ~3-4 hours for 5k steps
- Decision point: After Set A/B results, may cancel C if bottleneck found
