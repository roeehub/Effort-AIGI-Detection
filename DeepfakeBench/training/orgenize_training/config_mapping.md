# Configuration System Mapping

> **Purpose:** Map all configuration keys across the 4+ config sources  
> **Goal:** Design a unified, validated configuration schema

---

## Current Config Sources

1. **`config/detector/effort.yaml`** - Model architecture, defaults
2. **`config/train_config.yaml`** - GCS paths, checkpointing, labels
3. **`config/dataloader_config.yml`** - Data params, method lists, dataloader params
4. **W&B config (sweep or param-config)** - Runtime overrides

---

## Complete Config Key Inventory

### Model Configuration (`effort.yaml`)

| Key | Type | Default | Description | Override Source |
|-----|------|---------|-------------|-----------------|
| `model_name` | str | `"effort"` | Model registry key | - |
| `backbone_name` | str | `"vit"` | Backbone type | - |
| `resolution` | int | `224` | Input image size | - |
| `mean` | list | `[0.481, 0.458, 0.408]` | Normalization mean | - |
| `std` | list | `[0.269, 0.261, 0.276]` | Normalization std | - |
| `use_data_augmentation` | bool | `true` | Enable augmentation | - |

### Optimizer Configuration (`effort.yaml` + W&B)

| Key | Type | Default | Description | Override Source |
|-----|------|---------|-------------|-----------------|
| `optimizer.type` | str | `"adam"` | Optimizer type | - |
| `optimizer.adam.lr` | float | `0.0002` | Learning rate | W&B `learning_rate` |
| `optimizer.adam.eps` | float | `1e-8` | Adam epsilon | W&B `optimizer_eps` |
| `optimizer.adam.weight_decay` | float | `0.0005` | Weight decay | W&B `weight_decay` |
| `optimizer.adam.beta1` | float | `0.9` | Adam beta1 | - |
| `optimizer.adam.beta2` | float | `0.999` | Adam beta2 | - |

### Training Configuration (`effort.yaml` + `train_config.yaml` + W&B)

| Key | Type | Default | Description | Override Source |
|-----|------|---------|-------------|-----------------|
| `nEpochs` | int | `10` | Number of epochs | W&B `nEpochs` |
| `start_epoch` | int | `0` | Starting epoch | - |
| `lr_scheduler` | str | `null` | Scheduler type | W&B `lr_scheduler` |
| `total_training_steps` | int | `35000` | For warmup scheduler | W&B |
| `lr_scheduler_warmup_steps` | int | `1000` | Warmup steps | W&B |
| `gradient_clip_val` | float | `0` | Gradient clipping | W&B |
| `manualSeed` | int | `1024` | Random seed | - |
| `cuda` | bool | `true` | Use CUDA | - |
| `cudnn` | bool | `true` | Use cuDNN | - |

### GCS Assets (`train_config.yaml`)

| Key | Type | Default | Description | Override Source |
|-----|------|---------|-------------|-----------------|
| `gcs_assets.base_checkpoint.gcs_path` | str | (GCS URL) | Checkpoint source | W&B `gcs_base_checkpoint` |
| `gcs_assets.base_checkpoint.local_path` | str | `"./weights/base.pth"` | Local destination | - |
| `gcs_assets.clip_backbone.gcs_path` | str | (GCS URL) | CLIP weights | - |
| `gcs_assets.clip_backbone.local_path` | str | `"./weights/..."` | Local destination | - |
| `gcs_assets.clip_backbone.files` | list | `["config.json", "pytorch_model.bin"]` | Files to download | - |
| `gcs_assets.frame_manifest_json.gcs_path` | str | (GCS URL) | Frame manifest | - |
| `gcs_assets.property_manifest_parquet.gcs_path` | str | (GCS URL) | Property parquet | - |

### Checkpointing (`train_config.yaml`)

| Key | Type | Default | Description | Override Source |
|-----|------|---------|-------------|-----------------|
| `checkpointing.gcs_prefix` | str | `"gs://training-job-outputs/..."` | GCS output path | - |
| `load_base_checkpoint` | bool | `false` | Load checkpoint | W&B |

### Early Stopping (`train_config.yaml` + W&B)

| Key | Type | Default | Description | Override Source |
|-----|------|---------|-------------|-----------------|
| `early_stopping.enabled` | bool | `true` | Enable early stopping | W&B `early_stopping_enabled` |
| `early_stopping.patience` | int | `4` | Patience epochs | W&B `early_stopping_patience` |
| `early_stopping.min_delta` | float | `0.001` | Minimum improvement | W&B `early_stopping_min_delta` |

### Data Configuration (`dataloader_config.yml`)

| Key | Type | Default | Description | Override Source |
|-----|------|---------|-------------|-----------------|
| `gcp.bucket_name` | str | `"df40-frames-recropped-rfa85"` | Main data bucket | - |
| `gcp.ood_bucket_name` | str | `"deep-fake-test-..."` | OOD data bucket | - |
| `data_params.seed` | int | `737` | Data split seed | W&B `seed` |
| `data_params.val_split_ratio` | float | `0.3` | Validation ratio | W&B `val_split_ratio` |
| `data_params.data_subset_percentage` | float | `1.0` | Subset for debugging | W&B `data_subset_percentage` |
| `data_params.num_frames_per_video` | int | `8` | Frames per video | - |
| `data_params.evaluation_frequency` | int | ? | Evals per epoch | W&B `evaluation_frequency` |

### Property Balancing (`dataloader_config.yml` + W&B)

| Key | Type | Default | Description | Override Source |
|-----|------|---------|-------------|-----------------|
| `property_balancing.enabled` | bool | `true` | Enable property balancing | W&B `property_balancing_enabled` |
| `property_balancing.frame_properties_parquet_path` | str | (set at runtime) | Parquet file path | - |

### Dataloader Parameters (`dataloader_config.yml` + W&B)

| Key | Type | Default | Description | Override Source |
|-----|------|---------|-------------|-----------------|
| `dataloader_params.strategy` | str | ? | Batching strategy | W&B `dataloader_strategy` |
| `dataloader_params.batch_size` | int | `4` | Batch size | - |
| `dataloader_params.frames_per_batch` | int | ? | Frames per batch | W&B `frames_per_batch` |
| `dataloader_params.videos_per_batch` | int | `8` | Videos per batch | W&B `videos_per_batch` |
| `dataloader_params.frames_per_video` | int | `8` | Frames per video | W&B `frames_per_video` |
| `dataloader_params.real_label_ratio` | float | `null` | Real/fake ratio | W&B `real_label_ratio` |
| `dataloader_params.test_batch_size` | int | ? | Test batch size | W&B `test_batch_size` |
| `dataloader_params.num_workers` | int | `2` | Dataloader workers | W&B `num_workers` |
| `dataloader_params.prefetch_factor` | int | `3` | Prefetch batches | W&B `prefetch_factor` |

### Method Configuration (`dataloader_config.yml` + W&B)

| Key | Type | Default | Description | Override Source |
|-----|------|---------|-------------|-----------------|
| `methods.use_real_sources` | list | (defined in YAML) | Real data sources | W&B `dataset_methods` |
| `methods.use_fake_methods_for_training` | list | (defined in YAML) | Train fake methods | W&B `dataset_methods` |
| `methods.use_fake_methods_for_validation` | list | (defined in YAML) | Val fake methods | W&B `dataset_methods` |
| `methods.use_real_methods_for_validation_only` | list | `[]` | Val-only real methods | W&B `dataset_methods` |
| `methods.method_multipliers` | dict | `{}` | Per-method weights | W&B |

### Loss Configuration (W&B only)

| Key | Type | Default | Description | Override Source |
|-----|------|---------|-------------|-----------------|
| `use_focal_loss` | bool | `false` | Use focal loss | W&B |
| `focal_loss_gamma` | float | `2.0` | Focal loss gamma | W&B |
| `focal_loss_alpha` | float | `null` | Focal loss alpha | W&B |
| `lambda_reg` | float | `1.0` | SVD regularization | W&B |
| `rank` | int | `1023` | SVD rank | W&B |

### ArcFace Configuration (W&B only)

| Key | Type | Default | Description | Override Source |
|-----|------|---------|-------------|-----------------|
| `use_arcface_head` | bool | `false` | Use ArcFace head | W&B |
| `train_arcface` | bool | `true` | Train ArcFace params | W&B |
| `arcface_s` | float | `30.0` | ArcFace scale | W&B |
| `arcface_m` | float | `0.35` | ArcFace margin | W&B |
| `s_start` | float | `arcface_s` | Annealing start | W&B |
| `s_end` | float | `arcface_s` | Annealing end | W&B |
| `anneal_steps` | int | `0` | Annealing duration | W&B |

### Group-DRO Configuration (W&B only)

| Key | Type | Default | Description | Override Source |
|-----|------|---------|-------------|-----------------|
| `use_group_dro` | bool | `false` | Use Group-DRO | W&B |
| `group_dro_params.beta` | float | `3.0` | DRO beta | W&B `group_dro_beta` |
| `group_dro_params.clip_min` | float | `1.0` | Weight clip min | W&B `group_dro_clip_min` |
| `group_dro_params.clip_max` | float | `4.0` | Weight clip max | W&B `group_dro_clip_max` |
| `group_dro_params.ema_alpha` | float | `0.1` | EMA smoothing | W&B `group_dro_ema_alpha` |

### Augmentation Configuration (W&B only)

| Key | Type | Default | Description | Override Source |
|-----|------|---------|-------------|-----------------|
| `augmentation_version` | int/str | `"surgical"` | Aug pipeline version | W&B |
| `augmentation_params.use_geometric` | bool | `true` | Geometric augs | W&B |
| `augmentation_params.use_advanced_noise` | bool | `false` | Advanced noise | W&B |
| `augmentation_params.use_color_jitter` | bool | `true` | Color jitter | W&B |
| `augmentation_params.use_occlusion` | bool | `true` | Cutout/occlusion | W&B |
| `augmentation_params.sharpness_adjust_prob` | float | `0.6` | Sharpness aug prob | W&B |
| `augmentation_params.occlusion_prob` | float | `0.4` | Occlusion prob | W&B |

### Curriculum Learning (W&B only)

| Key | Type | Default | Description | Override Source |
|-----|------|---------|-------------|-----------------|
| `max_train_steps` | int | `null` | Max training steps | W&B |
| `evaluate_every_steps` | int | `null` | Step-based eval | W&B |
| `lesson_gate.enabled` | bool | `false` | Enable lesson gates | W&B |
| `lesson_gate.checks` | list | `[]` | Gate conditions | W&B |
| `lesson_gate.plateau_check` | dict | `{}` | Plateau detection | W&B |
| `lesson_gate.guardrail_check` | dict | `{}` | Guardrail settings | W&B |
| `lesson_data_control.enabled` | bool | `false` | Dynamic method grouping | W&B |

### Property Balancing Weights (W&B only)

| Key | Type | Default | Description | Override Source |
|-----|------|---------|-------------|-----------------|
| `real_category_weights` | dict | `{}` | Weights for real categories | W&B |
| `fake_category_weights` | dict | `{}` | Weights for fake categories | W&B |

---

## Config Loading Flow (Current)

```python
# train_sweep.py main()

# Step 1: Load base configs
with open(args.detector_path, 'r') as f:
    config = yaml.safe_load(f)  # effort.yaml

with open('./config/train_config.yaml', 'r') as f:
    config.update(yaml.safe_load(f))  # Merge train_config

with open(args.dataloader_config, 'r') as f:
    data_config = yaml.safe_load(f)  # Separate dict!

# Step 2: W&B initialization
wandb_run = wandb.init(mode="online", config=single_cfg)

# Step 3: Override from W&B (~100 lines!)
config['optimizer']['adam']['lr'] = float(wandb.config.learning_rate)
config['optimizer']['adam']['eps'] = float(wandb.config.optimizer_eps)
# ... 50+ more lines

data_config['dataloader_params']['strategy'] = wandb.config.dataloader_strategy
# ... 20+ more lines

# Step 4: Final merge
config.update(data_config)
```

---

## Problems with Current System

### 1. Two Parallel Config Dicts
```python
config = {}      # From effort.yaml + train_config.yaml
data_config = {} # From dataloader_config.yml
# Eventually merged, but passed separately to some functions
```

### 2. No Validation
```python
# Silent failures possible:
config['lr_schedular'] = wandb.config.get('lr_scheduler')  # Typo in key name
config['use_focal_los'] = wandb.config.get('use_focal_loss')  # Won't fail until runtime
```

### 3. Inconsistent Key Naming
```python
# W&B uses:
wandb.config.learning_rate
wandb.config.optimizer_eps

# Config expects:
config['optimizer']['adam']['lr']
config['optimizer']['adam']['eps']
```

### 4. Magic Defaults Scattered
```python
# Defaults in multiple places:
config.get('gradient_clip_val', 0)  # In train_sweep.py
wandb.config.get('frames_per_video', 8)  # In train_sweep.py
config.get('lr_scheduler_warmup_steps', 0)  # In choose_scheduler()
```

### 5. Conditional Feature Configs
```python
if config['use_arcface_head']:
    config['arcface_s'] = float(wandb.config.get('arcface_s', 30.0))
    # Only set if feature enabled, but accessed unconditionally later
```

---

## Proposed Unified Schema

```python
# config/schema.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Literal
from enum import Enum

class BatchingStrategy(str, Enum):
    FRAME_LEVEL = "frame_level"
    VIDEO_LEVEL = "video_level"
    PER_METHOD = "per_method"
    PROPERTY_BALANCING = "property_balancing"

class AugmentationVersion(str, Enum):
    V3 = "3"
    V4 = "4"
    V5 = "5"
    V6 = "6"
    V7 = "7"
    SURGICAL = "surgical"

@dataclass
class OptimizerConfig:
    type: Literal["adam", "sgd"] = "adam"
    lr: float = 0.0002
    eps: float = 1e-8
    weight_decay: float = 0.0005
    beta1: float = 0.9
    beta2: float = 0.999

@dataclass
class SchedulerConfig:
    type: Optional[Literal["cosine", "cosine_with_warmup"]] = None
    total_steps: int = 35000
    warmup_steps: int = 1000

@dataclass
class ModelConfig:
    name: str = "effort"
    backbone: str = "clip-vit-large-patch14"
    resolution: int = 224
    mean: List[float] = field(default_factory=lambda: [0.48145466, 0.4578275, 0.40821073])
    std: List[float] = field(default_factory=lambda: [0.26862954, 0.26130258, 0.27577711])
    rank: int = 1023
    lambda_reg: float = 1.0

@dataclass
class ArcFaceConfig:
    enabled: bool = False
    train: bool = True
    s: float = 30.0
    m: float = 0.35
    s_start: float = 30.0
    s_end: float = 30.0
    anneal_steps: int = 0

@dataclass
class FocalLossConfig:
    enabled: bool = False
    gamma: float = 2.0
    alpha: Optional[float] = None

@dataclass
class GroupDROConfig:
    enabled: bool = False
    beta: float = 3.0
    clip_min: float = 1.0
    clip_max: float = 4.0
    ema_alpha: float = 0.1

@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 4
    min_delta: float = 0.001

@dataclass
class AugmentationConfig:
    version: AugmentationVersion = AugmentationVersion.SURGICAL
    use_geometric: bool = True
    use_color_jitter: bool = True
    use_advanced_noise: bool = False
    use_occlusion: bool = True
    sharpness_adjust_prob: float = 0.6
    occlusion_prob: float = 0.4

@dataclass
class MethodConfig:
    real_sources: List[str] = field(default_factory=list)
    train_fakes: List[str] = field(default_factory=list)
    val_fakes: List[str] = field(default_factory=list)
    val_only_reals: List[str] = field(default_factory=list)
    multipliers: Dict[str, float] = field(default_factory=dict)

@dataclass
class BatchingConfig:
    strategy: BatchingStrategy = BatchingStrategy.PROPERTY_BALANCING
    batch_size: int = 64
    frames_per_video: int = 8
    videos_per_batch: int = 8
    real_label_ratio: Optional[float] = None
    num_workers: int = 4
    prefetch_factor: int = 3

@dataclass
class SplittingConfig:
    seed: int = 737
    val_split_ratio: float = 0.1
    data_subset_percentage: float = 1.0

@dataclass
class PropertyBalancingConfig:
    enabled: bool = True
    parquet_path: Optional[str] = None
    real_category_weights: Dict[str, float] = field(default_factory=dict)
    fake_category_weights: Dict[str, float] = field(default_factory=dict)

@dataclass
class DataConfig:
    gcs_bucket: str = "df40-frames-recropped-rfa85"
    ood_bucket: Optional[str] = None
    manifest_path: str = "./frame_manifest.json"
    splitting: SplittingConfig = field(default_factory=SplittingConfig)
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    methods: MethodConfig = field(default_factory=MethodConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    property_balancing: PropertyBalancingConfig = field(default_factory=PropertyBalancingConfig)

@dataclass  
class GCSAssetConfig:
    gcs_path: str
    local_path: str
    files: Optional[List[str]] = None  # For directories

@dataclass
class GCSConfig:
    base_checkpoint: Optional[GCSAssetConfig] = None
    clip_backbone: Optional[GCSAssetConfig] = None
    frame_manifest: Optional[GCSAssetConfig] = None
    property_manifest: Optional[GCSAssetConfig] = None
    output_prefix: str = "gs://training-job-outputs/best_checkpoints/"

@dataclass
class CurriculumConfig:
    max_train_steps: Optional[int] = None
    evaluate_every_steps: Optional[int] = None
    lesson_gate_enabled: bool = False
    lesson_gate_config: Dict = field(default_factory=dict)

@dataclass
class TrainingConfig:
    """Root configuration for a training run."""
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    gcs: GCSConfig = field(default_factory=GCSConfig)
    
    # Loss configurations
    arcface: ArcFaceConfig = field(default_factory=ArcFaceConfig)
    focal_loss: FocalLossConfig = field(default_factory=FocalLossConfig)
    group_dro: GroupDROConfig = field(default_factory=GroupDROConfig)
    
    # Training control
    epochs: int = 10
    gradient_clip: float = 0.0
    seed: int = 1024
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    
    # Checkpointing
    load_checkpoint: bool = False
    save_checkpoints: bool = True
    top_n_checkpoints: int = 6
    
    # Logging
    wandb_project: Optional[str] = None
    log_interval: int = 100
    eval_frequency: int = 2
```

---

## Config Loader Design

```python
# config/loader.py
import yaml
from typing import Optional, Dict, Any
from dataclasses import asdict
from .schema import TrainingConfig

def load_config(
    yaml_path: str,
    wandb_overrides: Optional[Dict[str, Any]] = None
) -> TrainingConfig:
    """
    Load configuration from YAML and optionally apply W&B overrides.
    
    Args:
        yaml_path: Path to the main config YAML file
        wandb_overrides: Dictionary of overrides from wandb.config
        
    Returns:
        Validated TrainingConfig instance
    """
    # Load base YAML
    with open(yaml_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    # Apply W&B overrides
    if wandb_overrides:
        raw_config = _apply_overrides(raw_config, wandb_overrides)
    
    # Convert to dataclass (validates structure)
    config = _dict_to_config(raw_config)
    
    # Run semantic validation
    _validate_config(config)
    
    return config

def _apply_overrides(base: dict, overrides: dict) -> dict:
    """Apply W&B config overrides with path mapping."""
    # Map W&B flat keys to nested config paths
    OVERRIDE_MAP = {
        'learning_rate': 'optimizer.lr',
        'optimizer_eps': 'optimizer.eps',
        'weight_decay': 'optimizer.weight_decay',
        'nEpochs': 'epochs',
        'dataloader_strategy': 'data.batching.strategy',
        'frames_per_batch': 'data.batching.batch_size',
        # ... etc
    }
    
    result = deep_copy(base)
    for wandb_key, value in overrides.items():
        if wandb_key in OVERRIDE_MAP:
            config_path = OVERRIDE_MAP[wandb_key]
            _set_nested(result, config_path, value)
        else:
            # Unknown keys go to a special section
            result.setdefault('_extra', {})[wandb_key] = value
    
    return result

def _validate_config(config: TrainingConfig) -> None:
    """Run semantic validation rules."""
    # Example validations
    if config.scheduler.type == 'cosine_with_warmup':
        if config.scheduler.total_steps <= 0:
            raise ValueError("cosine_with_warmup requires total_steps > 0")
    
    if config.arcface.enabled and config.focal_loss.enabled:
        # Maybe warn about unusual combination?
        pass
    
    if config.data.batching.strategy == 'property_balancing':
        if not config.data.property_balancing.enabled:
            raise ValueError("property_balancing strategy requires property_balancing.enabled=True")
```

---

## Migration Path

### Phase 1: Add Schema Alongside Current System
- Create `config/schema.py` with dataclasses
- Create `config/loader.py` with conversion functions
- Add validation without breaking existing code

### Phase 2: Update `train_sweep.py` to Use New Loader
- Replace 100 lines of `wandb.config.get()` with:
  ```python
  config = load_config('config/training.yaml', dict(wandb.config))
  ```
- Keep old code as fallback

### Phase 3: Create Single Config File Template
- Merge `effort.yaml`, `train_config.yaml`, `dataloader_config.yml` into one
- Provide example templates for common scenarios

### Phase 4: Remove Legacy Config Loading
- Delete old YAML files
- Remove `wandb.config.get()` scattered throughout code
