# Experiment Configuration Guide

**Last Updated:** December 23, 2025

This document describes all configurable parameters for training experiments and how to override them.

---

## Configuration Hierarchy

Configuration is loaded from multiple sources with later sources overriding earlier ones:

```
1. config/defaults.yaml          ← Base defaults (NEW - consolidated config)
2. config/detector/effort.yaml   ← Model-specific settings
3. config/train_config.yaml      ← Training environment (GCS paths, labels)
4. config/dataloader_config.yml  ← Data loading settings
5. experiments_configs/*.yaml    ← Your experiment overrides (via --param-config)
6. W&B sweep config              ← Hyperparameter sweep values (if using sweep mode)
```

---

## A) Configuration Files Summary

### Current State (December 2025)

| File | Purpose | Status |
|------|---------|--------|
| `config/defaults.yaml` | **Consolidated defaults** with backbone registry | ✅ Updated with backbone config |
| `config/detector/effort.yaml` | Model architecture settings | ✅ In use |
| `config/train_config.yaml` | GCS assets, checkpoints, labels | ✅ In use |
| `config/dataloader_config.yml` | Data buckets, methods, loader params | ✅ In use |
| `experiments_configs/*.yaml` | **Your experiment overrides** | ✅ Fully supported |

**Note:** `defaults.yaml` now includes a `backbone_registry` that maps backbone configurations to GCS paths. The backbone path is automatically resolved based on `backbone.source` and `backbone.variant`.

---

## B) What You Can Override via `--param-config`

### ✅ Fully Supported Parameters

These are read from your experiment YAML and applied via `utils/config_helpers.py`:

#### 🆕 Backbone Configuration (NEW!)
```yaml
# Select a different backbone model
backbone:
  type: clip                    # Options: clip, openclip, siglip, dinov2 (future)
  variant: ViT-L-14             # Options: ViT-B-16, ViT-B-32, ViT-L-14, ViT-L-14-336, ViT-H-14
  source: openai                # Options: openai, laion
  resolution: 224               # Input resolution (must match variant)
  
  # Optional: Override GCS paths (if not using backbone_registry)
  gcs_path_override: null       # e.g., "gs://my-bucket/custom-backbone/"
  local_path_override: null     # e.g., "./weights/custom-backbone/"

# Or use shorthand (just resolution)
resolution: 336                 # Direct resolution override
```

**Available Backbones (in backbone_registry):**

| Source | Variant | Resolution | Hidden Size | Status |
|--------|---------|------------|-------------|--------|
| openai | ViT-L-14 | 224 | 1024 | ✅ Default |
| openai | ViT-L-14-336 | 336 | 1024 | ✅ Available |
| openai | ViT-B-16 | 224 | 768 | ✅ Available |
| openai | ViT-B-32 | 224 | 512 | ✅ Available |
| laion | ViT-L-14 | 224 | 1024 | 📦 Needs upload |
| laion | ViT-H-14 | 224 | 1280 | 📦 Needs upload |

#### 🆕 Data Source Configuration (NEW!)
```yaml
# Override data buckets
bucket_name: "my-custom-bucket"           # Main training data bucket
ood_bucket_name: "my-ood-bucket"          # OOD test data bucket

# Or use nested structure
data_source:
  bucket_name: "my-custom-bucket"
  ood_bucket_name: "my-ood-bucket"
  manifest_path: "gs://my-bucket/manifests/custom_manifest.json"
  property_manifest_path: "gs://my-bucket/manifests/custom_properties.parquet"
```

**Available Buckets:**

| Bucket | Description | Status |
|--------|-------------|--------|
| `df40-frames-recropped-rfa85` | RFA85 recropped frames (default) | ✅ Default |
| `df40-frames` | Original unrecropped frames | ✅ Available |
| `deep-fake-test-10-08-25-frames-v2` | OOD test frames (default) | ✅ Default |
| `deep-fake-test-10-08-25-frames-yolo` | YOLO-cropped OOD frames | ✅ Available |

#### Optimizer & Training
```yaml
# Optimizer
learning_rate: 1.0e-4          # → config['optimizer']['adam']['lr']
weight_decay: 0.05             # → config['optimizer']['adam']['weight_decay']
optimizer_eps: 1.0e-8          # → config['optimizer']['adam']['eps']
lambda_reg: 1.0                # → config['lambda_reg'] (EFFORT SVD regularization)
rank: 1023                     # → config['rank'] (SVD rank)

# Training schedule
nEpochs: 20                    # → config['nEpochs']
gradient_clip_val: 0.0         # → config['gradient_clip_val']

# Learning rate scheduler
lr_scheduler: 'cosine_with_warmup'  # Options: null, 'cosine', 'cosine_with_warmup'
total_training_steps: 8000          # → config['total_training_steps']
lr_scheduler_warmup_steps: 800      # → config['lr_scheduler_warmup_steps']
```

#### Checkpointing
```yaml
load_base_checkpoint: true     # Whether to load a pretrained checkpoint
gcs_base_checkpoint: "gs://..."  # Override base checkpoint path (optional)
```

#### Dataloader
```yaml
dataloader_strategy: property_balancing  # Options: 'frame_level', 'video_level', 'per_method', 'property_balancing'
frames_per_batch: 64           # Batch size in frames (for property_balancing)
frames_per_video: 8            # Frames sampled per video
videos_per_batch: 8            # Videos per batch (for video_level/per_method)
num_workers: 12                # DataLoader workers
prefetch_factor: 3             # Prefetch batches per worker
test_batch_size: 8             # Validation batch size
real_label_ratio: null         # For balanced sampling (null = auto)
```

#### Data & Splitting
```yaml
seed: 737                      # Random seed for reproducibility
val_split_ratio: 0.1           # Validation split ratio
data_subset_percentage: 1.0    # Use subset (1.0 = all, 0.1 = 10%)
evaluation_frequency: 3        # Validations per epoch (overridden by evaluate_every_steps)
property_balancing_enabled: true  # Enable property-based sampling
```

#### ArcFace Head
```yaml
use_arcface_head: true         # Enable ArcFace classification head
arcface_s: 30.0                # Scale parameter
arcface_m: 0.30                # Margin parameter
train_arcface: true            # Trainable head

# Curriculum annealing (scale warmup)
s_start: 10.0                  # Starting scale
s_end: 30.0                    # Ending scale  
anneal_steps: 8000             # Steps to anneal over
```

#### Loss Functions
```yaml
use_focal_loss: false          # Use focal loss instead of CE
focal_loss_gamma: 2.0          # Focal loss gamma
focal_loss_alpha: null         # null = balanced, or float

use_group_dro: false           # Distributionally robust optimization
group_dro_beta: 3.0            # DRO beta parameter
group_dro_clip_min: 1.0        # Min weight clip
group_dro_clip_max: 4.0        # Max weight clip
group_dro_ema_alpha: 0.1       # EMA smoothing
```

#### Early Stopping
```yaml
early_stopping_enabled: false  # Enable early stopping
early_stopping_patience: 4     # Epochs without improvement
early_stopping_min_delta: 0.001  # Minimum improvement
```

#### Augmentation
```yaml
augmentation_version: 7        # Options: 3, 4, 5, 6, 7, 'surgical'

# Fine-grained augmentation params (optional)
augmentation_params:
  use_geometric: true
  use_advanced_noise: false
  use_color_jitter: true
  use_occlusion: true
  sharpness_adjust_prob: 0.6
  occlusion_prob: 0.4
```

#### Curriculum Learning (Lesson Gate)
```yaml
max_train_steps: 8000          # Total training steps (overrides nEpochs)
evaluate_every_steps: 700      # Validation frequency in steps

lesson_gate:
  enabled: true
  checks:
    - metric: "macro_auc"
      dataset: "val_in_dist"
      threshold: 0.93
      comparison: "ge"
    - metric: "hardest5_auc"
      dataset: "val_in_dist"
      threshold: 0.90
      comparison: "ge"
  plateau_check:
    enabled: true
    patience: 2
    min_delta: 0.001
  guardrail_check:
    enabled: true
    metric: "real_real_auc"
    dataset: "val_holdout"
    max_drop: 0.01
```

#### Dataset Methods Control
```yaml
dataset_methods:
  use_real_sources:
    - "celeb_real"
    - "youtube_real"
    - "external_youtube_avspeech"
  
  use_fake_methods_for_training:
    - "deepfakes"
    - "faceswapff"
    - "simswap"
    # ... more methods
  
  use_fake_methods_for_validation:
    - "stylegan3"
    - "sadtalker"
    # ... holdout methods
```

#### Lesson Data Control (Method Weighting)
```yaml
lesson_data_control:
  enabled: true
  fake_method_groups:
    L0_deepfakes:
      methods: ["deepfakes"]
      weight: 0.035
    L1_simswap:
      methods: ["simswap"]
      weight: 0.13
    # ... more groups
```

#### Property Balancing Weights (Advanced)
```yaml
real_category_weights:
  celeb_real: 0.4
  youtube_real: 0.6

fake_category_weights:
  face_swap: 0.5
  face_reenact: 0.3
  entire_face_synth: 0.2
```

---

## C) Recently Added Configuration Options

The following configuration options were added in the December 2025 refactoring:

### ✅ Backbone Model Selection (IMPLEMENTED)

You can now select different backbone models via experiment config:

```yaml
backbone:
  type: clip
  variant: ViT-L-14-336        # Use 336px resolution model
  source: openai
  resolution: 336
```

**Implementation Details:**
- `utils/config_helpers.py`: Added `apply_wandb_backbone_params()` and `resolve_backbone_paths()`
- `detectors/effort_detector.py`: Added `_resolve_backbone_path()` and `_get_hidden_size()` methods
- `config/defaults.yaml`: Added `backbone_registry` with GCS paths for each backbone variant

**To add a new backbone:**
1. Upload the backbone files to GCS: `gs://base-checkpoints/effort-aigi/models--{source}--{variant}/`
2. Add an entry to `backbone_registry` in `config/defaults.yaml`
3. Use in experiment config with `backbone.source` and `backbone.variant`

---

### ✅ Data Source/Bucket Selection (IMPLEMENTED)

You can now override data buckets via experiment config:

```yaml
bucket_name: "my-custom-bucket"
ood_bucket_name: "my-ood-bucket"

# Or with nested structure:
data_source:
  bucket_name: "my-custom-bucket"
  ood_bucket_name: "my-ood-bucket"
  manifest_path: "gs://my-bucket/manifests/custom.json"
```

**Implementation Details:**
- `utils/config_helpers.py`: Added `apply_wandb_data_source_params()`
- `config/defaults.yaml`: Updated `data` section with documented bucket options

---

### ✅ Image Resolution (IMPLEMENTED)

Resolution can be set directly or via backbone config:

```yaml
# Direct override
resolution: 336

# Or via backbone config
backbone:
  variant: ViT-L-14-336
  resolution: 336
```

**Implementation Details:**
- `utils/config_helpers.py`: Added `apply_wandb_resolution_params()`
- Resolution is automatically synced between `config['resolution']` and `config['backbone']['resolution']`

---

### ✅ Normalization Mean/Std (IMPLEMENTED)

Normalization values are now auto-set based on backbone type:

| Backbone Type | Mean | Std |
|---------------|------|-----|
| clip | [0.48145466, 0.4578275, 0.40821073] | [0.26862954, 0.26130258, 0.27577711] |
| openclip | [0.48145466, 0.4578275, 0.40821073] | [0.26862954, 0.26130258, 0.27577711] |
| siglip | [0.5, 0.5, 0.5] | [0.5, 0.5, 0.5] |
| dinov2 | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] |

You can also override explicitly:
```yaml
backbone:
  mean: [0.5, 0.5, 0.5]
  std: [0.5, 0.5, 0.5]
```

---

## D) Quick Reference: Running Experiments

### Single Experiment
```bash
# Local (in container)
python train_sweep.py --param-config experiments_configs/27sep/exp5.yaml

# GCP
./launch_experiment_jobs.sh --param-config experiments_configs/27sep/exp5.yaml
```

### Multiple Experiments (Same Image)
```bash
# Build image once
gcloud builds submit --config cloudbuild.yaml

# Run different experiments
./launch_experiment_jobs.sh --param-config experiments_configs/exp_arcface.yaml
./launch_experiment_jobs.sh --param-config experiments_configs/exp_focal_loss.yaml
./launch_experiment_jobs.sh --param-config experiments_configs/exp_new_methods.yaml
```

### Creating a New Experiment

1. Copy an existing experiment config:
   ```bash
   cp experiments_configs/27sep/exp5.yaml experiments_configs/my_new_exp.yaml
   ```

2. Modify parameters you want to change

3. Run with:
   ```bash
   python train_sweep.py --param-config experiments_configs/my_new_exp.yaml
   ```

---

## E) Full Parameter Reference Table

| Parameter | Type | Default | Location | Overridable |
|-----------|------|---------|----------|-------------|
| `learning_rate` | float | 0.0002 | optimizer | ✅ |
| `weight_decay` | float | 0.0005 | optimizer | ✅ |
| `optimizer_eps` | float | 1e-8 | optimizer | ✅ |
| `lambda_reg` | float | 1.0 | model | ✅ |
| `rank` | int | 1023 | model | ✅ |
| `nEpochs` | int | 10 | training | ✅ |
| `max_train_steps` | int | null | curriculum | ✅ |
| `lr_scheduler` | str | null | scheduler | ✅ |
| `total_training_steps` | int | 35000 | scheduler | ✅ |
| `lr_scheduler_warmup_steps` | int | 1000 | scheduler | ✅ |
| `gradient_clip_val` | float | 0.0 | training | ✅ |
| `dataloader_strategy` | str | property_balancing | dataloader | ✅ |
| `frames_per_batch` | int | null | dataloader | ✅ |
| `frames_per_video` | int | 8 | dataloader | ✅ |
| `videos_per_batch` | int | 8 | dataloader | ✅ |
| `num_workers` | int | 2 | dataloader | ✅ |
| `prefetch_factor` | int | 3 | dataloader | ✅ |
| `test_batch_size` | int | null | dataloader | ✅ |
| `seed` | int | 737 | data | ✅ |
| `val_split_ratio` | float | 0.3 | data | ✅ |
| `data_subset_percentage` | float | 1.0 | data | ✅ |
| `evaluation_frequency` | int | 1 | dataloader | ✅ |
| `evaluate_every_steps` | int | null | curriculum | ✅ |
| `property_balancing_enabled` | bool | true | property_balancing | ✅ |
| `use_arcface_head` | bool | false | arcface | ✅ |
| `arcface_s` | float | 30.0 | arcface | ✅ |
| `arcface_m` | float | 0.35 | arcface | ✅ |
| `s_start` | float | null | arcface | ✅ |
| `s_end` | float | null | arcface | ✅ |
| `anneal_steps` | int | 0 | arcface | ✅ |
| `use_focal_loss` | bool | false | loss | ✅ |
| `focal_loss_gamma` | float | 2.0 | loss | ✅ |
| `focal_loss_alpha` | float | null | loss | ✅ |
| `use_group_dro` | bool | false | group_dro | ✅ |
| `early_stopping_enabled` | bool | true | early_stopping | ✅ |
| `early_stopping_patience` | int | 4 | early_stopping | ✅ |
| `augmentation_version` | int/str | surgical | augmentation | ✅ |
| `lesson_gate` | dict | {enabled: false} | curriculum | ✅ |
| `lesson_data_control` | dict | {enabled: false} | curriculum | ✅ |
| `dataset_methods` | dict | {} | methods | ✅ |
| `load_base_checkpoint` | bool | false | checkpointing | ✅ |
| `gcs_base_checkpoint` | str | null | checkpointing | ✅ |
| `backbone` | dict | {type: clip, variant: ViT-L-14, source: openai} | model | ✅ NEW |
| `backbone.type` | str | clip | model | ✅ NEW |
| `backbone.variant` | str | ViT-L-14 | model | ✅ NEW |
| `backbone.source` | str | openai | model | ✅ NEW |
| `backbone.resolution` | int | 224 | model | ✅ NEW |
| `resolution` | int | 224 | model | ✅ NEW |
| `bucket_name` | str | df40-frames-recropped-rfa85 | data | ✅ NEW |
| `ood_bucket_name` | str | deep-fake-test-10-08-25-frames-v2 | data | ✅ NEW |
| `data_source.manifest_path` | str | null | data | ✅ NEW |

---

## F) Recommended Next Steps

1. **To run your first experiment:** Just use your existing `exp5.yaml` - it will work.

2. **To use a different backbone:** Add `backbone` config to your experiment YAML (see section B).

3. **To use a different data bucket:** Add `bucket_name` to your experiment YAML.

4. **To test different augmentations:** Already supported! Just set `augmentation_version: 7` or `augmentation_version: surgical`.

5. **To test different dataloaders:** Already supported! Set `dataloader_strategy: frame_level` or `video_level` or `per_method`.

6. **To add a new backbone variant:**
   - Upload backbone files to GCS
   - Add entry to `backbone_registry` in `config/defaults.yaml`
   - Reference in experiment config
