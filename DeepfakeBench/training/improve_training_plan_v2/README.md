# Improve Training Plan v2 - Documentation

This directory contains comprehensive documentation for understanding and improving the DeepLive/EFFORT training system.

## Document Index

| Document | Purpose |
|----------|---------|
| [01_TRAINING_FLOW.md](01_TRAINING_FLOW.md) | End-to-end training pipeline, from launch to model output |
| [02_HYPERPARAMETERS.md](02_HYPERPARAMETERS.md) | All configurable parameters and their effects |
| [03_BACKBONES.md](03_BACKBONES.md) | Vision backbones, SVD residual approach, and model architecture |
| [04_DATA_AND_AUGMENTATION.md](04_DATA_AND_AUGMENTATION.md) | Datasets, dataloaders, sampling, and augmentation strategies |
| [05_METRICS_AND_LOGGING.md](05_METRICS_AND_LOGGING.md) | W&B metrics, what they mean, and how to interpret them |
| [06_EXPERIMENT_ANALYSIS_JAN4.md](06_EXPERIMENT_ANALYSIS_JAN4.md) | **NEW:** Analysis of B16/B16-LAION/B32 experiments |
| [07_STRATEGIC_DIRECTIONS.md](07_STRATEGIC_DIRECTIONS.md) | **NEW:** Strategic plan with parallel work streams |
| [08_IMPLEMENTATION_LOG_JAN4.md](08_IMPLEMENTATION_LOG_JAN4.md) | **NEW:** Log of changes made on Jan 4, 2026 |

## Quick Reference

### Launch Training
```bash
# Build production image
./dev.sh build-prod -y

# Launch experiment
./launch_experiment.sh <wandb_project> <region> <experiment_yaml>

# Example
./launch_experiment.sh deeplive-jan2026 asia-southeast1 experiments/deeplive_vit_B16.yaml
```

### Key Configuration Files
- `experiments/*.yaml` - Experiment-specific configs
- `config/backbone_registry.yaml` - Available backbones
- `train_parameters.yaml` - Default training parameters

### Test SVD Residual Fix
```bash
cd DeepfakeBench/training
python scripts/test_svd_residual.py
```

---

*Last updated: January 4, 2026*
