# Unified Training Script Plan

> **Created:** January 1, 2026  
> **Status:** 📋 PLANNING  
> **Priority:** HIGH - This is the correct way to complete the refactoring

---

## The Problem

We currently have multiple training scripts:
- `train_sweep.py` - The production script (works with Trainer, all features)
- `train_deeplive.py` - A separate script for DeepLive (broken, bypasses Trainer)
- `train_deeplive_v2.py` - A fixed version (uses Trainer, but still a separate script)

**This is wrong.** DeepLive is just another training configuration - different backbone, different data source, different batching strategy. It should NOT require a separate training script.

---

## The Correct Design

### One Script, Many Configurations

```
┌─────────────────────────────────────────────────────────────────┐
│                         train.py                                │
│                    (single entry point)                         │
│                                                                 │
│  1. Load experiment config (YAML)                               │
│  2. Config specifies: backbone, data_source, strategy           │
│  3. Factory pattern creates appropriate components              │
│  4. Trainer runs the training loop (same for ALL experiments)   │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            experiments/            experiments/
            ffpp_clip.yaml          deeplive_laion.yaml
```

### Usage Should Be Identical

```bash
# Old experiments
python train.py --config experiments/ffpp_clip_vit_b16.yaml

# DeepLive experiments - SAME SCRIPT
python train.py --config experiments/deeplive_laion_vit_b16.yaml

# Any future experiment - SAME SCRIPT
python train.py --config experiments/future_dataset_new_backbone.yaml
```

---

## What Needs to Change

### 1. Add `data_source` Config Option

The experiment YAML should specify where data comes from:

```yaml
# experiments/ffpp_clip.yaml (OLD STYLE)
data_source: manifest  # or "video_manifest" 
data_params:
  manifest_path: "gs://bucket/manifests/ffpp.json"
  # ... existing params

# experiments/deeplive_laion.yaml (DEEPLIVE STYLE)
data_source: deeplive
deeplive:
  gcs_bucket: "live-deepfake-methods-real-and-fake-frames"
  sampling_mode: sparse
  anchor_indices: [0, 2, 4, 6, 8, 10, 12, 14]
  train_split: 0.8
  val_split: 0.1
```

### 2. Create Data Source Factory

```python
# data/sources/__init__.py (NEW)

def create_data_pipeline(config: Dict, data_config: Dict, logger) -> Tuple[DataLoader, DataLoader, List]:
    """
    Factory that creates the appropriate data pipeline based on config.
    
    Returns:
        train_loader, val_loader, train_samples (for epoch length calculation)
    """
    data_source = data_config.get('data_source', 'manifest')
    
    if data_source == 'manifest':
        # Existing flow: prepare_video_splits_v2 + create_dataloaders
        return create_manifest_pipeline(config, data_config, logger)
    
    elif data_source == 'deeplive':
        # DeepLive flow: DeepLiveDataset + DeepLiveBatchingStrategy
        return create_deeplive_pipeline(config, data_config, logger)
    
    elif data_source == 'hf_dataset':
        # Future: HuggingFace datasets
        return create_hf_pipeline(config, data_config, logger)
    
    else:
        raise ValueError(f"Unknown data_source: {data_source}")
```

### 3. Integrate into `train_sweep.py` (or rename to `train.py`)

```python
# train.py (simplified pseudocode)

def main():
    # 1. Parse args and load config
    config, data_config = load_and_merge_configs(args)
    
    # 2. Apply W&B overrides (if running sweep)
    if wandb_enabled:
        apply_all_wandb_overrides(config, data_config, wandb.config)
    
    # 3. Create model (already uses factory pattern via DETECTOR registry)
    model = DETECTOR[config['model_name']](config)
    
    # 4. Create data pipeline (NEW - uses data_source factory)
    train_loader, val_loader, train_samples = create_data_pipeline(
        config, data_config, logger
    )
    
    # 5. Create optimizer/scheduler (existing code)
    optimizer = choose_optimizer(model, config)
    scheduler = choose_scheduler(config, optimizer)
    
    # 6. Create Trainer and run (existing code)
    trainer = Trainer(config, model, optimizer, scheduler, ...)
    
    for epoch in range(config['nEpochs']):
        trainer.train_epoch(train_loader, epoch, train_samples)
```

### 4. Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `data/sources/__init__.py` | CREATE | Data source factory |
| `data/sources/manifest.py` | CREATE | Existing manifest-based pipeline (extract from train_sweep.py) |
| `data/sources/deeplive.py` | CREATE | DeepLive pipeline (extract from train_deeplive_v2.py) |
| `train_sweep.py` → `train.py` | RENAME/MODIFY | Use data source factory |
| `train_deeplive.py` | DELETE | No longer needed |
| `train_deeplive_v2.py` | DELETE | No longer needed |

### 5. Experiment Config Schema

```yaml
# Full experiment config schema

# === Model ===
model_name: effort  # Always 'effort' for now

# === Backbone ===
backbone:
  source: huggingface | laion | timm
  # For huggingface:
  huggingface_id: "openai/clip-vit-base-patch16"
  # For laion/openclip:
  model_name: "ViT-B-16"
  pretrained: "datacomp_xl_s13b_b90k"
  # Common:
  resolution: 224
  hidden_size: 768

# === Data Source ===
data_source: manifest | deeplive | hf_dataset

# For manifest data source:
data_params:
  manifest_path: "gs://..."
  methods: [...]

# For deeplive data source:
deeplive:
  gcs_bucket: "..."
  sampling_mode: sparse
  train_split: 0.8

# === Batching ===
dataloader_params:
  strategy: per_method | video_level | frame_level | property_balancing | deeplive
  frames_per_batch: 32
  frames_per_video: 8

# === Training ===
learning_rate: 1e-4
weight_decay: 0.05
nEpochs: 50
# ... etc
```

---

## Benefits of This Approach

1. **Single source of truth** - One training script to maintain
2. **Consistent behavior** - All experiments get same Trainer features
3. **Easy to add new data sources** - Just add a new factory method
4. **Config-driven** - Change experiments by changing YAML, not code
5. **Testable** - Each component can be tested in isolation
6. **W&B sweeps work** - Same override system for all experiments

---

## Migration Steps

### Step 1: Create Data Source Factory
- Create `data/sources/` directory
- Extract manifest pipeline from `train_sweep.py`
- Extract deeplive pipeline from `train_deeplive_v2.py`
- Create factory function

### Step 2: Update train_sweep.py
- Import data source factory
- Replace direct data loading with factory call
- Test with existing experiments (should work identically)

### Step 3: Test DeepLive via Unified Script
- Create proper `experiments/deeplive_*.yaml` config
- Run `python train_sweep.py --param-config experiments/deeplive_laion.yaml`
- Verify it works the same as `train_deeplive_v2.py`

### Step 4: Cleanup
- Rename `train_sweep.py` → `train.py` (optional, for clarity)
- Delete `train_deeplive.py` and `train_deeplive_v2.py`
- Update documentation

### Step 5: Update Launch Scripts
- `launch_experiment_jobs.sh` - Should already work (uses train_sweep.py)
- `launch_deeplive.sh` - Update to use `train.py` with deeplive config

---

## Example: Adding a New Data Source in the Future

With this design, adding support for HuggingFace datasets would be:

```python
# data/sources/hf_dataset.py
def create_hf_pipeline(config, data_config, logger):
    from datasets import load_dataset
    
    hf_config = data_config.get('hf_dataset', {})
    dataset = load_dataset(hf_config['name'], split='train')
    # ... create loaders
    return train_loader, val_loader, samples
```

```yaml
# experiments/hf_celeba.yaml
data_source: hf_dataset
hf_dataset:
  name: "celeb_df"
  split: "train"
```

No new training script needed!

---

## Summary

**Current State:** Multiple training scripts with duplicated/inconsistent logic  
**Target State:** One unified `train.py` with config-driven data sources  
**Key Insight:** DeepLive is a configuration, not a separate pipeline  

This is the correct completion of the refactoring effort.
