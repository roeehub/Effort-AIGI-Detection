# Phase 2 — Round 6: Real-Image Robustness Experiments

## Goal

Improve robustness on low-quality / compressed real images (VCD, YouTube) without regressing fake detection. Round 6 introduces three new levers on top of the R5_S3 baseline:

1. **VCD training reals** — a 20% identity-split of VCD data mixed into the training set
2. **`vcd_targeted` augmentation** — quality-targeted augmentation pipeline with family-aware routing
3. **Gradient Reversal Quality Head (GRL)** — adversarial domain classifier encouraging quality-invariant features

## Experiment Matrix

| Config | VCD Reals | Augmentation | GRL Head | Seed | Purpose |
|--------|-----------|-------------|----------|------|---------|
| `R6_S1_vcd_reals_aug` | ✅ 800 | vcd_targeted | ❌ | 737 | Data + aug |
| `R6_S1_vcd_reals_aug_seed1337` | ✅ 800 | vcd_targeted | ❌ | 1337 | Seed variance |
| `R6_S2_aug_only` | ❌ | vcd_targeted | ❌ | 737 | Aug-only ablation |
| `R6_S2_aug_only_seed1337` | ❌ | vcd_targeted | ❌ | 1337 | Seed variance |
| `R6_S3_vcd_reals_aug_grl` | ✅ 800 | vcd_targeted | ✅ | 737 | **Full stack** |
| `R6_S3_vcd_reals_aug_grl_seed1337` | ✅ 800 | vcd_targeted | ✅ | 1337 | Seed variance |
| `R6_S4_grl_only` | ❌ | vcd_targeted | ✅ | 737 | GRL-only ablation |
| `R6_SMOKE_30MIN` | ✅ 200 | vcd_targeted | ✅ | 737 | Quick sanity check |

## Config Descriptions

### S1 — VCD Reals + Augmentation
Adds 800 VCD real images (20% identity split, max 10 frames/identity) to the training set with `vcd_targeted` augmentation. Tests whether exposure to compressed reals alone improves OOD real accuracy.

### S2 — Augmentation Only (ablation)
Applies `vcd_targeted` augmentation but does **not** add VCD reals to training. Isolates the contribution of augmentation.

### S3 — Full Stack (VCD Reals + Augmentation + GRL)
Combines all three levers. The gradient reversal head classifies samples into 4 quality domains and adversarially trains the backbone toward quality-invariant representations.

### S4 — GRL Only (ablation)
Applies the GRL quality head without VCD training reals. The external_real domain will have 0 training samples (domain count kept at 4 for consistency).

### SMOKE — 30-Minute Smoke Test
Full-stack config with reduced steps (1000 total, 2 epochs) and smaller VCD sample (200). Use to verify the pipeline runs end-to-end before committing GPU hours.

Smoke startup is optimized to fail fast:
- DeepLive discovery uses cached manifests + per-strategy cap.
- VisoMaster discovery cache is enabled.
- OOD loader construction is skipped at startup (`build_loader_at_startup: false`) and runtime OOD monitoring is disabled (`ood_monitoring_enabled: false`).

## Launch Commands

```bash
cd DeepfakeBench/training

# Smoke test first
./launch_experiment.sh <WANDB_PROJECT> <REGION> experiments/phase2_round6/R6_SMOKE_30MIN.yaml

# Full experiments
./launch_experiment.sh <WANDB_PROJECT> <REGION> experiments/phase2_round6/R6_S1_vcd_reals_aug.yaml
./launch_experiment.sh <WANDB_PROJECT> <REGION> experiments/phase2_round6/R6_S1_vcd_reals_aug_seed1337.yaml
./launch_experiment.sh <WANDB_PROJECT> <REGION> experiments/phase2_round6/R6_S2_aug_only.yaml
./launch_experiment.sh <WANDB_PROJECT> <REGION> experiments/phase2_round6/R6_S2_aug_only_seed1337.yaml
./launch_experiment.sh <WANDB_PROJECT> <REGION> experiments/phase2_round6/R6_S3_vcd_reals_aug_grl.yaml
./launch_experiment.sh <WANDB_PROJECT> <REGION> experiments/phase2_round6/R6_S3_vcd_reals_aug_grl_seed1337.yaml
./launch_experiment.sh <WANDB_PROJECT> <REGION> experiments/phase2_round6/R6_S4_grl_only.yaml
```

**Example:**
```bash
./launch_experiment.sh df40-experiments asia-southeast1 experiments/phase2_round6/R6_S3_vcd_reals_aug_grl.yaml
```

## Key Differences from Round 5

| Aspect | Round 5 (R5_S3) | Round 6 |
|--------|-----------------|---------|
| VCD reals in training | ❌ | ✅ (800 samples, 20% identity split) |
| Augmentation | base | `vcd_targeted` (quality-targeted, family-aware routing) |
| GRL quality head | ❌ | ✅ (4 domains, λ=0.1, hidden=128) |
| OOD identity exclusion | N/A | `exclude_training_identities: true` for VCD configs |
| Ablation controls | — | S2 (aug-only), S4 (GRL-only) |
| Seed replicates | — | Each strategy has 737 + 1337 variant |

## Success Criteria

| Metric | Target | Monitored Via |
|--------|--------|---------------|
| VCD real accuracy | **> 80%** | `ood_monitoring` → `zoom_vcd_real` |
| YouTube real FPR | **≤ 8%** | `ood_monitoring` → `external_youtube_avspeech` |
| WMA failure-fake accuracy | **≥ 85%** | `ood_monitoring` → `wma_failure_fake` |
| Enhanced DeepLive TPR | **≥ 95%** | In-distribution validation |
| DF40 in-dist AUC | No regression vs R5_S3 | In-distribution validation |
| Holdout method AUC | No regression vs R5_S3 | Holdout evaluation |

## Seed Strategy

- `seed` (top-level) + `combined_paired.seed`: controls training randomness (init, shuffle, augmentation)
- `combined_paired.split_seed: 737`: **always 737** — keeps identity splits deterministic across seed variants
- `identity_split_seed: 737` in `external_training_reals`: keeps VCD train/OOD split consistent

This ensures seed-1337 variants differ only in training dynamics, not in data composition.
