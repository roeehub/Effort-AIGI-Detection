# Training Codebase Organization & Refactoring Plan

> **Status:** Investigation Phase  
> **Goal:** Understand the current architecture, identify pain points, and plan modular refactoring  
> **Date:** December 2024

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture Overview](#2-current-architecture-overview)
3. [Key Pain Points Identified](#3-key-pain-points-identified)
4. [File-by-File Analysis](#4-file-by-file-analysis)
5. [Configuration System Analysis](#5-configuration-system-analysis)
6. [Proposed Modular Architecture](#6-proposed-modular-architecture)
7. [Refactoring Priorities](#7-refactoring-priorities)
8. [Next Steps](#8-next-steps)

---

## 1. Executive Summary

The training codebase has grown organically around the Effort AIGI paper's approach (SVD decomposition for CLIP backbone training). What started as a simple training script (`train.py`) evolved into `train_sweep.py` (intended for W&B sweeps) and became the de-facto main entry point for **all** training scenarios.

### Core Issues
- **Monolithic files:** `train_sweep.py` (~900 lines), `dataloaders.py` (~1670 lines), `trainer.py` (~1800 lines)
- **Scattered configuration:** 4+ YAML files with overlapping concerns and runtime overrides via W&B
- **Tightly coupled components:** Data splitting, dataloader creation, and training logic are intertwined
- **Difficult experimentation:** Changing the batching strategy or data pipeline requires touching multiple files

### Opportunity
Refactoring into a modular structure would enable:
- Easy swapping of data pipelines (batching strategies, augmentations)
- Clear separation between research experiments and production code
- Better testability and debugging
- Faster onboarding for new collaborators

---

## 2. Current Architecture Overview

### High-Level Training Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           train_sweep.py (Entry Point)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Parse CLI args & Load YAML configs (4 files!)                           │
│  2. Initialize W&B (wandb.init) & merge configs                             │
│  3. Download GCS assets (checkpoints, manifests)                            │
│  4. Call prepare_video_splits_v2() → train_data, val_in_dist, val_holdout   │
│  5. Call create_dataloaders() → train_loader, val_loaders                   │
│  6. Optionally rebuild train_loader with WeightedRandomSampler              │
│  7. Create OOD loader (optional)                                            │
│  8. Instantiate model (DETECTOR registry) & optimizer & scheduler           │
│  9. Create Trainer instance                                                 │
│  10. Load checkpoint (if configured)                                        │
│  11. Training loop: for epoch → trainer.train_epoch()                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   GCS Buckets    │────▶│  prepare_splits  │────▶│  dataloaders.py  │
│  (frame images)  │     │    .py           │     │                  │
│                  │     │                  │     │ create_dataloaders│
│  - df40-frames   │     │ - Load manifest  │     │ - Strategy-based │
│  - OOD bucket    │     │ - Filter methods │     │   loader creation│
│                  │     │ - Split by ID    │     │ - Augmentations  │
│                  │     │ - Balance        │     │ - Collate fn     │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                  │                        │
                                  ▼                        ▼
                         ┌──────────────────┐     ┌──────────────────┐
                         │    VideoInfo     │     │   DataLoader     │
                         │    dataclass     │     │   (PyTorch)      │
                         │                  │     │                  │
                         │ - label          │     │ Strategies:      │
                         │ - method         │     │ - frame_level    │
                         │ - video_id       │     │ - video_level    │
                         │ - frame_paths    │     │ - per_method     │
                         │ - identity       │     │ - property_bal   │
                         └──────────────────┘     └──────────────────┘
```

### Model & Training Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              trainer/trainer.py                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  Trainer class (~1800 lines):                                               │
│  - train_epoch(), train_step(), _run_train_step()                          │
│  - _run_validation(), validate()                                            │
│  - save_ckpt(), load_ckpt(), _upload_to_gcs()                              │
│  - Early stopping, step-based training, lesson gates                       │
│  - Group-DRO loss calculation                                              │
│  - ArcFace parameter annealing                                             │
│  - OOD monitoring                                                          │
│  - Detailed reporting & W&B logging                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         detectors/effort_detector.py                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  EffortDetector (nn.Module):                                                │
│  - CLIP backbone (ViT-L/14)                                                 │
│  - SVDResidualLinear layers (the "Effort" contribution)                     │
│  - ArcMarginProduct head (optional)                                         │
│  - FocalLoss / CrossEntropyLoss                                             │
│  - get_losses(), forward(), features(), classifier()                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Key Pain Points Identified

### 3.1 `train_sweep.py` - The "God Object" Problem

**Current state:** 900 lines doing everything from config loading to training orchestration

**Specific issues:**
- Lines 1-70: Imports and global setup (some duplicated!)
- Lines 70-120: Utility functions (`init_seed`, `choose_optimizer`, `choose_scheduler`, `choose_metric`)
- Lines 130-200: GCS download helpers (`download_gcs_asset`, `download_assets_from_gcs`)
- Lines 290-580: **main()** function doing:
  - Config loading and merging from 4 sources
  - W&B config extraction (~100 lines of `wandb.config.get()` calls)
  - Dataset/dataloader configuration overrides
  - GCS asset downloading
  - Run naming and logging setup
- Lines 580-750: Data preparation and loader creation with inline logic
- Lines 750-900: Model setup, checkpoint loading, training loop

**Why this hurts:**
- Adding a new config parameter requires touching 3+ places
- Can't easily test data loading without running the whole script
- Mixing concerns: config, data, model, training all in one file

### 3.2 `dataloaders.py` - Strategy Explosion

**Current state:** 1670 lines with multiple batching strategies and augmentation pipelines

**Specific issues:**
- Lines 50-230: **7 different augmentation pipelines** (V3, V4, V5, V6, V7, plus custom)
- Lines 300-430: More augmentation helpers (`create_surgical_augmentation_pipeline`, `create_general_augmentation_pipeline`)
- Lines 440-550: Custom DataPipe classes (`CustomRoundRobinDataPipe`, `MateFinderDataPipe`)
- Lines 550-800: Multiple frame loading functions:
  - `load_and_process_frame_batch()` - for frame_level strategy
  - `load_and_process_property_batch()` - for property_balancing
  - `load_and_process_video()` - for video_level/per_method
  - `load_and_process_video_detailed()` - for validation with extra metadata
- Lines 800-1670: **`create_dataloaders()`** function (~800 lines!) with strategy branching

**Why this hurts:**
- Adding a new batching strategy means navigating 800+ lines of `if/elif` blocks
- Augmentation selection is scattered and inconsistent
- Hard to understand which code path is actually running

### 3.3 `prepare_splits.py` - Reasonable but Growing

**Current state:** ~750 lines, relatively well-structured

**Good parts:**
- `VideoInfo` dataclass is clean
- `prepare_video_splits_v2()` has clear branching between strategies
- `prepare_splits_property_based()` handles the complex case

**Issues:**
- Two parallel splitting paths (legacy vs property-based) with duplicated logic
- Method categorization (`EFS_METHODS`, `REG_METHODS`) is hardcoded
- Balancing logic (`_balance_video_list`, `_balance_df_by_label`) could be extracted

### 3.4 `trainer.py` - Feature Accumulation

**Current state:** ~1800 lines with many optional features

**Features piled up:**
- Basic training loop
- Early stopping
- Step-based training control
- Lesson gates (curriculum learning)
- Group-DRO loss
- ArcFace parameter annealing
- OOD monitoring
- GCS checkpoint upload
- Detailed CSV report generation
- W&B logging

**Why this hurts:**
- Hard to understand the core training logic amid all the optional features
- Each feature adds state to the Trainer class
- Testing requires mocking many subsystems

### 3.5 Configuration System - Scattered and Overridden

**Current files:**
1. `config/detector/effort.yaml` - Model architecture, augmentation defaults, optimizer defaults
2. `config/train_config.yaml` - GCS paths, checkpointing, early stopping, label mappings
3. `config/dataloader_config.yml` - GCP bucket names, data params, method lists, dataloader params
4. `train_parameters.yaml` (external) - W&B sweep parameters that override everything

**Config merging in `train_sweep.py`:**
```python
# Load base configs
with open(args.detector_path, 'r') as f:
    config = yaml.safe_load(f)
with open('./config/train_config.yaml', 'r') as f:
    config.update(yaml.safe_load(f))
with open(dataloader_config_path, 'r') as f:
    data_config = yaml.safe_load(f)

# Then W&B overrides ~50 individual keys:
config['optimizer']['adam']['lr'] = float(wandb.config.learning_rate)
config['optimizer']['adam']['eps'] = float(wandb.config.optimizer_eps)
# ... 50 more lines like this
```

**Why this hurts:**
- No single source of truth for what config a run used
- Easy to have config key typos that fail silently
- Impossible to reproduce a run without the exact W&B config

---

## 4. File-by-File Analysis

### `train_sweep.py` Decomposition Candidates

| Lines | Responsibility | Proposed Module |
|-------|---------------|-----------------|
| 1-70 | Imports & globals | Keep minimal |
| 70-130 | `init_seed`, `choose_*` helpers | `training/utils/setup.py` |
| 130-200 | GCS download | `training/utils/gcs.py` |
| 290-450 | Config loading & W&B merge | `training/config/config_manager.py` |
| 450-580 | Data config overrides | `training/config/config_manager.py` |
| 580-700 | Data prep & loader creation | `training/data/pipeline.py` |
| 700-800 | Logging & stats | `training/utils/logging.py` |
| 800-900 | Model setup & training loop | Keep in `train.py` (simplified) |

### `dataloaders.py` Decomposition Candidates

| Lines | Responsibility | Proposed Module |
|-------|---------------|-----------------|
| 50-300 | Augmentation pipelines | `training/data/augmentations/` (multiple files) |
| 300-430 | Augmentation factory | `training/data/augmentations/factory.py` |
| 440-550 | Custom DataPipes | `training/data/datapipes.py` |
| 550-700 | Frame loading functions | `training/data/loaders/frame_loader.py` |
| 700-800 | Video loading functions | `training/data/loaders/video_loader.py` |
| 800-1200 | `create_dataloaders` (strategy dispatch) | `training/data/dataloader_factory.py` |
| 1200-1670 | Strategy-specific loader creation | `training/data/strategies/` (per file) |

### `trainer.py` Decomposition Candidates

| Lines | Responsibility | Proposed Module |
|-------|---------------|-----------------|
| 1-130 | Trainer init & basic state | Keep core in `trainer.py` |
| 130-200 | Group-DRO logic | `training/trainer/mixins/group_dro.py` |
| 200-330 | ArcFace annealing, lesson gates | `training/trainer/mixins/curriculum.py` |
| 330-500 | Checkpointing & GCS | `training/trainer/mixins/checkpointing.py` |
| 500-1000 | train_epoch, train_step | Keep core in `trainer.py` |
| 1000-1500 | Validation logic | `training/trainer/validation.py` |
| 1500-1800 | Reporting & CSV generation | `training/trainer/reporting.py` |

---

## 5. Configuration System Analysis

### Current Config Hierarchy

```
                    ┌─────────────────────────┐
                    │   detector/effort.yaml   │ (Base model config)
                    └───────────┬─────────────┘
                                │ config.update()
                    ┌───────────▼─────────────┐
                    │    train_config.yaml     │ (GCS, checkpointing)
                    └───────────┬─────────────┘
                                │ separate dict
                    ┌───────────▼─────────────┐
                    │  dataloader_config.yml   │ (Data params, methods)
                    └───────────┬─────────────┘
                                │ wandb.config overrides
                    ┌───────────▼─────────────┐
                    │   W&B Sweep / YAML       │ (Runtime overrides)
                    └───────────┬─────────────┘
                                │ final merge
                    ┌───────────▼─────────────┐
                    │   Runtime config dict    │ (Used by training)
                    └─────────────────────────┘
```

### Config Key Overlap Analysis

| Key | effort.yaml | train_config.yaml | dataloader_config.yml | W&B |
|-----|-------------|-------------------|----------------------|-----|
| `optimizer.adam.lr` | ✓ | | | ✓ (override) |
| `nEpochs` | ✓ | | | ✓ (override) |
| `use_data_augmentation` | ✓ | | | |
| `gcs_assets` | | ✓ | | |
| `early_stopping` | | ✓ | | ✓ (override) |
| `methods.use_real_sources` | | | ✓ | ✓ (can override) |
| `dataloader_params.strategy` | | | ✓ | ✓ (override) |
| `property_balancing.enabled` | | | ✓ | ✓ (override) |

### Proposed Config Structure

```yaml
# Single config file: training_config.yaml
# Validated by a Pydantic model or dataclass

model:
  name: effort
  backbone: clip-vit-large-patch14
  resolution: 224
  checkpoint_path: null  # Optional

training:
  epochs: 10
  learning_rate: 0.0002
  weight_decay: 0.0005
  scheduler: cosine_with_warmup
  warmup_steps: 1000
  gradient_clip: 1.0

data:
  gcs_bucket: df40-frames-recropped-rfa85
  ood_bucket: null
  manifest_path: ./frame_manifest.json
  
  splitting:
    strategy: property_balancing  # or: legacy
    val_split_ratio: 0.1
    seed: 737
    
  batching:
    strategy: property_balancing  # frame_level | video_level | per_method | property_balancing
    batch_size: 64
    frames_per_video: 8  # for video_level
    num_workers: 4
    
  methods:
    real_sources: [FaceForensics++, Celeb-real, YouTube-real]
    train_fakes: [simswap, faceswap, ...]
    val_fakes: [facedancer, DiT, ...]
    
  augmentation:
    version: 7  # 3, 4, 5, 6, 7, or surgical
    # Additional params for surgical:
    use_geometric: true
    use_color_jitter: true

losses:
  primary: cross_entropy  # or: focal
  focal_gamma: 2.0
  focal_alpha: null
  use_arcface: false
  arcface_s: 30.0
  arcface_m: 0.35

features:
  early_stopping:
    enabled: true
    patience: 4
    min_delta: 0.001
  group_dro:
    enabled: false
    beta: 3.0
  curriculum:
    enabled: false
    lesson_gate: {}

logging:
  wandb_project: deepfake-detection
  log_interval: 100
  eval_frequency: 2  # times per epoch

checkpointing:
  gcs_prefix: gs://training-job-outputs/best_checkpoints/
  top_n: 6
```

---

## 6. Proposed Modular Architecture

### Directory Structure

```
training/
├── __init__.py
├── train.py                    # Simplified entry point (~200 lines)
├── README.md                   # How to run, configure, extend
│
├── config/
│   ├── __init__.py
│   ├── schema.py               # Pydantic/dataclass config validation
│   ├── defaults.py             # Default config values
│   └── loader.py               # Config loading, merging, W&B integration
│
├── data/
│   ├── __init__.py
│   ├── pipeline.py             # High-level: get_training_pipeline(config) -> DataLoader
│   ├── splitting/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract splitter interface
│   │   ├── legacy.py           # Original splitting logic
│   │   └── property_based.py   # Property-balancing split
│   ├── batching/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract batching interface
│   │   ├── frame_level.py      # Frame-level strategy
│   │   ├── video_level.py      # Video-level strategy
│   │   ├── per_method.py       # Per-method strategy
│   │   └── property_balanced.py # Property-balancing strategy
│   ├── augmentations/
│   │   ├── __init__.py
│   │   ├── registry.py         # Augmentation version registry
│   │   ├── v3_legacy.py
│   │   ├── v4_moderate.py
│   │   ├── v5_hybrid.py
│   │   ├── v6_source_dependent.py
│   │   ├── v7_portfolio.py
│   │   └── surgical.py         # Dynamic property-based
│   └── loaders/
│       ├── __init__.py
│       ├── frame_loader.py     # Single frame loading
│       ├── video_loader.py     # Video batch loading
│       └── gcs_utils.py        # GCS/fsspec helpers
│
├── models/
│   ├── __init__.py
│   ├── registry.py             # Model registry (existing DETECTOR)
│   ├── effort/
│   │   ├── __init__.py
│   │   ├── detector.py         # EffortDetector
│   │   ├── svd_layers.py       # SVDResidualLinear
│   │   └── heads.py            # ArcMarginProduct, classifier heads
│   └── losses/
│       ├── __init__.py
│       ├── focal.py
│       └── cross_entropy.py
│
├── trainer/
│   ├── __init__.py
│   ├── base.py                 # Core Trainer class (~500 lines)
│   ├── validation.py           # Validation logic
│   ├── reporting.py            # CSV/W&B reporting
│   └── mixins/
│       ├── __init__.py
│       ├── checkpointing.py    # Checkpoint save/load/GCS upload
│       ├── early_stopping.py   # Early stopping logic
│       ├── group_dro.py        # Group-DRO loss weighting
│       └── curriculum.py       # Lesson gates, ArcFace annealing
│
└── utils/
    ├── __init__.py
    ├── gcs.py                  # GCS download/upload helpers
    ├── setup.py                # Seed init, optimizer/scheduler factories
    └── logging.py              # Logger setup, W&B integration
```

### Key Abstractions

#### 1. Data Pipeline Interface

```python
# data/pipeline.py
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

class DataPipeline(ABC):
    """High-level interface for creating training data pipelines."""
    
    @abstractmethod
    def get_train_loader(self) -> DataLoader:
        pass
    
    @abstractmethod
    def get_val_loaders(self) -> tuple[DataLoader, DataLoader]:
        """Returns (in_distribution_loader, holdout_loader)."""
        pass
    
    @abstractmethod
    def get_ood_loader(self) -> DataLoader | None:
        pass

def create_pipeline(config: TrainingConfig) -> DataPipeline:
    """Factory function to create the appropriate pipeline."""
    strategy = config.data.batching.strategy
    if strategy == 'property_balancing':
        return PropertyBalancedPipeline(config)
    elif strategy == 'video_level':
        return VideoLevelPipeline(config)
    # ... etc
```

#### 2. Batching Strategy Interface

```python
# data/batching/base.py
from abc import ABC, abstractmethod

class BatchingStrategy(ABC):
    """Interface for different batching approaches."""
    
    @abstractmethod
    def create_train_loader(
        self, 
        train_data: list,  # VideoInfo or frame dicts
        config: dict
    ) -> DataLoader:
        pass
    
    @abstractmethod
    def create_val_loader(
        self,
        val_videos: list[VideoInfo],
        config: dict
    ) -> DataLoader:
        pass
```

#### 3. Augmentation Registry

```python
# data/augmentations/registry.py
import albumentations as A

AUGMENTATION_REGISTRY = {}

def register_augmentation(version: int | str):
    def decorator(fn):
        AUGMENTATION_REGISTRY[version] = fn
        return fn
    return decorator

@register_augmentation(version=3)
def create_v3_pipeline() -> A.Compose:
    return revised_augmentation_pipeline_legacy

@register_augmentation(version='surgical')
def create_surgical_pipeline(config: dict, frame_props: dict) -> A.Compose:
    return create_surgical_augmentation_pipeline(config, frame_props)

def get_augmentation(version: int | str, **kwargs) -> A.Compose:
    factory = AUGMENTATION_REGISTRY.get(version)
    if factory is None:
        raise ValueError(f"Unknown augmentation version: {version}")
    return factory(**kwargs) if kwargs else factory()
```

#### 4. Trainer Mixins

```python
# trainer/mixins/checkpointing.py
class CheckpointingMixin:
    """Mixin for checkpoint save/load/upload functionality."""
    
    def save_checkpoint(self, epoch: int, metrics: dict) -> str:
        """Save checkpoint locally and optionally to GCS."""
        ...
    
    def load_checkpoint(self, path: str, validate: bool = True) -> None:
        """Load checkpoint with optional config validation."""
        ...
    
    def _upload_to_gcs(self, local_path: str, gcs_path: str) -> bool:
        ...

# trainer/base.py
class Trainer(CheckpointingMixin, EarlyStoppingMixin, ...):
    """Core trainer with feature mixins."""
    
    def __init__(self, config, model, optimizer, ...):
        # Only core initialization
        self.config = config
        self.model = model
        # Mixins add their own state
        
    def train_epoch(self, train_loader, epoch):
        # Clean training loop, delegates to mixins
        ...
```

---

## 7. Refactoring Priorities

### Phase 1: Low-Risk Extractions (1-2 weeks)

These can be done without changing any interfaces:

1. **Extract augmentation pipelines** → `data/augmentations/`
   - Move all `A.Compose` definitions to separate files
   - Create registry with `get_augmentation(version)` factory
   - Update `dataloaders.py` to use registry
   
2. **Extract GCS utilities** → `utils/gcs.py`
   - Move `download_gcs_asset`, `download_assets_from_gcs`
   - Add proper error handling and retry logic
   
3. **Extract setup helpers** → `utils/setup.py`
   - Move `init_seed`, `choose_optimizer`, `choose_scheduler`, `choose_metric`

### Phase 2: Config System Overhaul (1-2 weeks)

4. **Create config schema** → `config/schema.py`
   - Define Pydantic models or dataclasses for all config sections
   - Add validation (e.g., `augmentation_version in [3,4,5,6,7,'surgical']`)
   
5. **Create config loader** → `config/loader.py`
   - Single function: `load_config(yaml_path, wandb_overrides) -> TrainingConfig`
   - Replaces 100+ lines of `wandb.config.get()` calls in `train_sweep.py`

### Phase 3: Data Pipeline Refactoring (2-3 weeks)

6. **Extract splitting logic** → `data/splitting/`
   - Move `prepare_video_splits_v2` and helpers
   - Create clear interface: `Splitter.split(config) -> TrainData, ValData`
   
7. **Extract batching strategies** → `data/batching/`
   - Break apart `create_dataloaders` into strategy classes
   - Each strategy: ~100-200 lines instead of 800+ combined
   
8. **Create pipeline abstraction** → `data/pipeline.py`
   - Combine splitting + batching into single interface
   - `create_pipeline(config) -> DataPipeline`

### Phase 4: Trainer Cleanup (1-2 weeks)

9. **Extract trainer features into mixins**
   - `CheckpointingMixin`
   - `EarlyStoppingMixin`
   - `GroupDROMixin`
   - `CurriculumMixin`
   
10. **Simplify core trainer**
    - Aim for ~500 lines for core `Trainer` class
    - Clean `train_epoch` and `validate` methods

### Phase 5: Entry Point Simplification (1 week)

11. **Rewrite `train.py`**
    - Target: ~200 lines
    - Clear flow: load_config → create_pipeline → create_model → create_trainer → train

---

## 8. Next Steps

### Immediate Actions

1. [ ] **Create this document's companion files:**
   - `augmentations.md` - Detailed analysis of augmentation pipelines
   - `batching_strategies.md` - Analysis of each batching strategy
   - `config_mapping.md` - Full mapping of config keys across files

2. [ ] **Set up testing infrastructure:**
   - Unit tests for extracted utilities
   - Integration test for full training pipeline (small data)

3. [ ] **Begin Phase 1 extractions:**
   - Start with augmentations (lowest risk, highest clarity benefit)

### Questions to Resolve

1. **Backward compatibility:** Do we need to support the old config format?
2. **W&B integration:** Keep sweep-style overrides or move to pure YAML?
3. **OOD loader:** Is this used actively? Should it be a first-class feature?
4. **Multiple validation sets:** Is the dual-validation (in-dist + holdout) pattern standard?

### Success Metrics

- [ ] `train_sweep.py` reduced to <300 lines
- [ ] `dataloaders.py` split into <400-line files
- [ ] `trainer.py` core <500 lines with mixins
- [ ] New batching strategy can be added in <100 lines
- [ ] Config is validated at load time (no runtime KeyErrors)
- [ ] Full test coverage for extracted utilities

---

## 9. Planning Documents Index

All planning documents for this refactoring effort are in `/training/orgenize_training/`:

| Document | Purpose |
|----------|---------|
| **[main.md](./main.md)** | This file - Master overview and architecture analysis |
| **[batching_strategies.md](./batching_strategies.md)** | Deep dive into the 4 batching strategies |
| **[augmentations.md](./augmentations.md)** | Analysis of 7+ augmentation pipelines |
| **[config_mapping.md](./config_mapping.md)** | Full mapping of config keys across 4 YAML files |
| **[checklist.md](./checklist.md)** | Task tracking system with phases and status |
| **[logging_conventions.md](./logging_conventions.md)** | Standardized logging and error handling approach |
| **[future_features.md](./future_features.md)** | Research directions and future architecture needs |
| **[testing_plan.md](./testing_plan.md)** | Testing strategy with examples at all levels |

### Quick Start Guide

1. **Understand the current state:** Read this document (main.md)
2. **Pick up context on specific areas:** 
   - For data loading → batching_strategies.md
   - For image transforms → augmentations.md
   - For config confusion → config_mapping.md
3. **Start working:** Check checklist.md for current status and next tasks
4. **Follow conventions:** Use logging_conventions.md as reference
5. **Plan for the future:** Review future_features.md before major decisions
6. **Test your changes:** Follow testing_plan.md for verification

---

## Appendix: Code Smell Inventory

### `train_sweep.py`

```python
# Smell 1: Duplicated imports (lines 1-40)
from tqdm import tqdm  # imported twice
import torch  # imported twice
import yaml  # imported twice

# Smell 2: Global arg parsing before imports complete
args, _ = parser.parse_known_args()
torch.cuda.set_device(args.local_rank)  # Side effect at import time

# Smell 3: Massive config merging in main()
config['optimizer']['adam']['lr'] = float(wandb.config.learning_rate)
config['optimizer']['adam']['eps'] = float(wandb.config.optimizer_eps)
# ... 50+ more lines like this

# Smell 4: Inline conditional loader rebuilding (lines 630-700)
if is_property_balancing:
    logger.info("...")
elif (dataset_obj is not None) and (not isinstance(dataset_obj, IterableDataset)):
    if not train_data or 'sample_weight' not in train_data[0]:
        # ...
    else:
        # 30+ lines of loader rebuilding
```

### `dataloaders.py`

```python
# Smell 1: Multiple similar functions (lines 700-850)
def load_and_process_video(video_info, config, mode, frame_count_override=None):
    # 50 lines
    
def load_and_process_video_detailed(video_info, config, mode, frame_count_override=None):
    # 60 lines, nearly identical with extra return values

# Smell 2: Giant function with strategy branching (create_dataloaders ~800 lines)
def create_dataloaders(train_data, val_in_dist_videos, val_holdout_videos, config, data_config):
    strategy = config['dataloader_params']['strategy']
    if strategy == 'frame_level':
        # 100+ lines
    elif strategy == 'video_level':
        # 100+ lines
    elif strategy == 'per_method':
        # 150+ lines
    elif strategy == 'property_balancing':
        # 200+ lines
```

### `trainer.py`

```python
# Smell 1: __init__ with feature flag branching (50+ lines of conditionals)
if self.early_stopping_enabled:
    self.logger.info(...)
else:
    self.logger.info(...)

if self.max_train_steps:
    self.logger.info(...)
if self.evaluate_every_steps:
    self.logger.info(...)

if self.use_group_dro:
    self._init_group_dro(config)

if self.gate_enabled:
    # 20 lines of setup

# Smell 2: Methods that do too much (train_epoch ~300 lines)
def train_epoch(self, train_loader, epoch, train_videos, val_method_loaders=None):
    # Setup, iteration, logging, validation, early stopping all in one
```

---

*This document will be updated as the refactoring progresses.*