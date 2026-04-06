# Unified Training Script - Implementation Complete

> **Completed:** January 1, 2026  
> **Status:** ✅ IMPLEMENTED  
> **Based on:** unified_training_plan.md

---

## Summary

This document describes the implementation of the unified training script that consolidates multiple training scripts into one config-driven approach.

### What Changed

| Before | After |
|--------|-------|
| `train_sweep.py` for manifest data | Single `train_sweep.py` for ALL data sources |
| `train_deeplive.py` (broken) | ❌ No longer needed |
| `train_deeplive_v2.py` (separate script) | ❌ No longer needed |
| ~130 lines of data loading code | Single factory call |

### New Files Created

```
data/sources/
├── __init__.py      # Factory function and DataPipelineResult
├── manifest.py      # Manifest-based data pipeline
└── deeplive.py      # DeepLive GCS data pipeline

experiments/
├── example_manifest.yaml   # Example manifest config
└── example_deeplive.yaml   # Example DeepLive config

test_data_sources.py        # Validation tests
```

---

## Usage

### Traditional Manifest-Based Training

```bash
# Same as before - nothing changes for existing configs!
python train_sweep.py --param-config experiments/example_manifest.yaml
```

### DeepLive Training (Now uses SAME script!)

```bash
# Use the same train_sweep.py - just different config
python train_sweep.py --param-config experiments/example_deeplive.yaml
```

### Key Configuration

The data source is determined by the `data_source` field in your config:

```yaml
# For manifest-based training (default)
data_source: manifest

# For DeepLive training
data_source: deeplive
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         train_sweep.py                          │
│                    (single entry point)                         │
│                                                                 │
│  1. Load base configs                                           │
│  2. Apply W&B overrides                                         │
│  3. Call create_data_pipeline(config, data_config, logger)      │
│     ↓                                                           │
│  4. Factory returns DataPipelineResult with:                    │
│     - train_loader                                              │
│     - val_in_dist_loader                                        │
│     - val_holdout_loader                                        │
│     - train_samples                                             │
│     - data_stats                                                │
│     - ood_loader (optional)                                     │
│  5. Create model, optimizer, scheduler                          │
│  6. Run training loop with Trainer                              │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    data/sources/manifest.py        data/sources/deeplive.py
    - prepare_video_splits_v2       - DeepLiveDataset
    - create_dataloaders            - DeepLiveBatchingStrategy
    - weighted sampling             - landmark augmentation
```

---

## API Reference

### `create_data_pipeline`

```python
from data.sources import create_data_pipeline, DataPipelineResult

result: DataPipelineResult = create_data_pipeline(
    config,      # Main training config
    data_config, # Data-specific config (includes data_source)
    logger,      # Logger instance
)

# Access results
train_loader = result.train_loader
val_loader = result.val_in_dist_loader
train_samples = result.train_samples
```

### `DataPipelineResult`

```python
@dataclass
class DataPipelineResult:
    train_loader: DataLoader
    val_in_dist_loader: DataLoader
    val_holdout_loader: Optional[DataLoader]
    train_samples: List[Any]
    data_stats: Dict[str, Any]
    ood_loader: Optional[DataLoader] = None
```

### Registering New Data Sources

```python
from data.sources import register_data_source, DataPipelineResult

@register_data_source('my_new_source')
def create_my_pipeline(config, data_config, logger, **kwargs) -> DataPipelineResult:
    # Your custom data loading logic
    ...
    return DataPipelineResult(
        train_loader=train_loader,
        val_in_dist_loader=val_loader,
        val_holdout_loader=None,
        train_samples=samples,
        data_stats={'custom': 'stats'},
    )
```

---

## Migration Guide

### For Existing Configs

**No changes needed!** Existing configs that don't specify `data_source` will automatically use `'manifest'` (the default).

### For DeepLive Training

Instead of:
```bash
python train_deeplive_v2.py --config experiments/deeplive_vit_B16.yaml
```

Now use:
```bash
python train_sweep.py --param-config experiments/deeplive_vit_B16.yaml
```

Just add this to your DeepLive config if not already present:
```yaml
data_source: deeplive
```

---

## Testing

Testing should be performed via Vertex AI training jobs to ensure the correct cloud environment with all dependencies.

### Test Manifest Data Source (Default Behavior)

Use an existing manifest-based config - no changes needed:
```bash
# Launch via cloud build or Vertex job
python train_sweep.py --param-config experiments_configs/example_exp.yaml
```

### Test DeepLive Data Source

Update a DeepLive config to include `data_source: deeplive` and run:
```bash
# Launch via cloud build or Vertex job  
python train_sweep.py --param-config experiments/example_deeplive.yaml
```

### Verification Checklist

1. ✅ Manifest training starts and loads data correctly
2. ✅ DeepLive training starts and loads from GCS bucket
3. ✅ Both use the same Trainer class
4. ✅ W&B logging works for both
5. ✅ Checkpointing works for both

---

## Future Extensions

Adding a new data source (e.g., HuggingFace datasets) is now trivial:

1. Create `data/sources/hf_dataset.py`
2. Use the `@register_data_source('hf_dataset')` decorator
3. Create an experiment config with `data_source: hf_dataset`
4. Use the same `train_sweep.py` - no new training script needed!

---

## Files That Can Be Deprecated

The following files are now redundant and can be removed in a future cleanup:

- `train_deeplive.py` - Original broken script
- `train_deeplive_v2.py` - Fixed but separate script (now integrated)

Keep them for now for reference, but all training should go through `train_sweep.py`.
