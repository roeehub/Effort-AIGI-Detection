# Augmentation Pipelines Analysis

> **Purpose:** Document all augmentation pipelines and their use cases  
> **Goal:** Consolidate into a clean registry-based system

---

## Overview

The codebase has **7+ augmentation pipelines** defined in `dataloaders.py`, plus dynamic "surgical" augmentation. This document catalogs each and recommends consolidation.

---

## Pipeline Inventory

### V3: `revised_augmentation_pipeline_legacy`

**Location:** `dataloaders.py` lines 118-148

**Purpose:** Original "calibrated" pipeline for compatibility with albumentations 0.4.6

**Components:**
```python
A.Compose([
    A.HorizontalFlip(p=0.5),
    
    # Quality transformation (custom unsharp mask)
    CustomUnsharpMask(blur_limit=(3, 9), alpha=(0.5, 1.0), threshold=10, p=0.7),
    
    # Compression & noise
    A.OneOf([
        A.ImageCompression(quality_lower=50, quality_upper=90, p=0.5),
        A.GaussNoise(var_limit=(10.0, 60.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    ], p=0.6),
    
    # Color augmentation
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.4),
])
```

**Characteristics:**
- Moderate sharpening (alpha 0.5-1.0)
- Compression quality 50-90 (moderate artifacts)
- Gentle color shifts
- ~60% chance of any compression/noise

**Use Case:** Conservative baseline, backward compatibility

---

### V4: `augmentation_pipeline_v4`

**Location:** `dataloaders.py` lines 154-186

**Purpose:** "Moderately aggressive" step-up from V3

**Changes from V3:**
| Parameter | V3 | V4 |
|-----------|----|----|
| Unsharp alpha | 0.5-1.0 | 0.6-1.2 |
| Unsharp probability | 0.7 | 0.75 |
| Compression quality_lower | 50 | 45 |
| GaussNoise var_limit upper | 60.0 | 65.0 |
| Compression/noise overall p | 0.6 | 0.7 |
| Brightness/contrast limit | 0.1 | 0.12 |
| HueSaturationValue hue_shift | 10 | 12 |
| HueSaturationValue sat_shift | 15 | 20 |
| HSV probability | 0.4 | 0.45 |

**Use Case:** More challenging training data without being destructive

---

### V5: `augmentation_pipeline_v5`

**Location:** `dataloaders.py` lines 192-227

**Purpose:** Hybrid pipeline with optional heavy degradation

**Structure:**
```python
A.Compose([
    # First: Apply V4 to every image
    augmentation_pipeline_v4,
    
    # Then: 50% chance of heavy degradation
    A.OneOf([
        A.Compose([degradation_block], p=0.5),
        NoOp(p=0.5)
    ], p=1.0)
])
```

**Degradation block:**
```python
degradation_block = A.Compose([
    A.OneOf([
        A.Compose([
            A.Downscale(scale_min=0.3, scale_max=0.6, interpolation=INTER_AREA, p=0.8),
            A.Resize(height=224, width=224, interpolation=INTER_LINEAR, always_apply=True)
        ], p=0.7),
        NoOp(p=0.3)
    ], p=1.0),
    A.ImageCompression(quality_lower=25, quality_upper=70, p=0.7),
    A.GaussianBlur(blur_limit=(3, 11), p=0.4),
])
```

**Key characteristics:**
- Aggressive downscaling (30-60% of original size)
- Heavy compression (quality 25-70)
- Strong blur (kernel up to 11)

**Use Case:** Robustness to heavily degraded inputs (social media, low-quality sources)

---

### V6: Source-Dependent Portfolio

**Location:** `dataloaders.py` lines 233-268

**Purpose:** Apply different augmentations based on data source

**Architecture:**
```
Frame path contains "df40-frames-recropped-rfa85" (Train-Primary)?
├── YES → Portfolio Sampling (40/30/30)
│   ├── AUG_PIPELINE_V6_SIMULATOR (40%)
│   ├── AUG_PIPELINE_V4_GENERALIST (30%)
│   └── AUG_PIPELINE_PURIST (30%)
└── NO → Train-Effort (50/50)
    ├── AUG_PIPELINE_V3_MILD (50%)
    └── HorizontalFlip only (50%)
```

**Sub-pipelines:**

**AUG_PIPELINE_V6_SIMULATOR** - Social media simulation
```python
A.Compose([
    A.HorizontalFlip(p=0.5),
    CustomUnsharpMask(blur_limit=(3, 9), alpha=(0.7, 1.5), threshold=10, p=0.9),
    A.ImageCompression(quality_lower=40, quality_upper=85, p=0.9),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
])
```

**AUG_PIPELINE_V4_GENERALIST** - Kitchen sink
```python
A.Compose([
    A.HorizontalFlip(p=0.5),
    CustomUnsharpMask(blur_limit=(3, 9), alpha=(0.6, 1.2), threshold=10, p=0.75),
    A.OneOf([
        A.ImageCompression(quality_lower=45, quality_upper=90, p=0.5),
        A.GaussNoise(var_limit=(10.0, 65.0), p=0.3),
    ], p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
])
```

**AUG_PIPELINE_PURIST** - Minimal
```python
A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
])
```

**AUG_PIPELINE_V3_MILD** - For non-primary data
```python
A.Compose([
    A.HorizontalFlip(p=0.5),
    CustomUnsharpMask(blur_limit=(3, 7), alpha=(0.2, 0.7), threshold=10, p=0.5),
    A.OneOf([
        A.ImageCompression(quality_lower=60, quality_upper=95, p=0.5), 
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.5)
    ], p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
])
```

**Use Case:** Differentiated treatment of high-quality vs lower-quality training data

---

### V7: Portfolio (Simplified)

**Location:** `dataloaders.py` lines 298-302

**Purpose:** Weighted portfolio without source checking

```python
def apply_augmentation_v7(img_np: np.ndarray) -> np.ndarray:
    pipelines = [AUG_PIPELINE_V6_SIMULATOR, AUG_PIPELINE_V4_GENERALIST, AUG_PIPELINE_PURIST]
    weights = [0.4, 0.45, 0.15]
    chosen_pipeline = random.choices(pipelines, weights=weights, k=1)[0]
    return chosen_pipeline(image=img_np)['image']
```

**Weights:** 40% Simulator, 45% Generalist, 15% Purist

**Use Case:** Simplified V6 without source-based branching

---

### Surgical: Dynamic Property-Based

**Location:** `dataloaders.py` lines 335-384

**Purpose:** Construct augmentation pipeline based on frame properties at runtime

**Key function:** `create_surgical_augmentation_pipeline(config, frame_properties)`

**Logic:**
```python
def create_surgical_augmentation_pipeline(config, frame_properties):
    transforms = []
    
    # Always: horizontal flip
    transforms.append(A.HorizontalFlip(p=0.5))
    
    # Optional: geometric transforms
    if config.get('use_geometric', False):
        transforms.append(A.ShiftScaleRotate(...))
    
    # Optional: color jitter
    if config.get('use_color_jitter', False):
        transforms.extend([
            A.RandomBrightnessContrast(...),
            A.HueSaturationValue(...)
        ])
    
    # SURGICAL: Property-based sharpness adjustment
    sharpness_bucket = frame_properties.get('sharpness_bucket')
    if random.random() < config.get('sharpness_adjust_prob', 0.5):
        if sharpness_bucket == 'q4':  # Very sharp
            transforms.append(degrade_quality_pipeline)
        elif sharpness_bucket == 'q1':  # Very blurry
            transforms.append(enhance_quality_pipeline)
    
    # Optional: advanced noise
    if config.get('use_advanced_noise', False):
        transforms.append(A.OneOf([
            A.ISONoise(...),
            social_media_pipeline
        ], p=config.get('advanced_noise_prob', 0.6)))
    
    # Optional: occlusion
    if config.get('use_occlusion', False):
        transforms.append(A.Cutout(...))
    
    return A.Compose(transforms)
```

**Helper pipelines:**
```python
degrade_quality_pipeline = A.Compose([
    A.OneOf([
        A.ImageCompression(quality_lower=40, quality_upper=70, p=0.8),
        A.GaussianBlur(blur_limit=(5, 11), p=0.6),
        A.GaussNoise(var_limit=(20.0, 80.0), p=0.4),
    ], p=1.0)
])

enhance_quality_pipeline = A.Compose([
    A.IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.9),
])

social_media_pipeline = A.Compose([
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.Downscale(scale_min=0.5, scale_max=0.75, ..., p=0.8),
    A.ImageCompression(quality_lower=30, quality_upper=60, p=1.0),
])
```

**Use Case:** Counter-shortcut learning by breaking spurious correlations (e.g., "sharp = real")

---

## Pipeline Usage by Strategy

| Strategy | Default Augmentation | Configurable |
|----------|---------------------|--------------|
| `frame_level` | `data_aug_v2()` → general pipeline | Via `augmentation_params` |
| `video_level` | `data_aug_v2()` → general pipeline | Via `augmentation_params` |
| `per_method` | `data_aug_v2()` → general pipeline | Via `augmentation_params` |
| `property_balancing` | Version-based in `load_and_process_property_batch` | Via `augmentation_version` |

### `data_aug_v2()` Logic
```python
def data_aug_v2(img, config, augmentation_seed=None):
    aug_params = config.get('augmentation_params', {})
    pipeline = create_general_augmentation_pipeline(aug_params)
    return pipeline(image=np.array(img))['image']

def create_general_augmentation_pipeline(config):
    aug_version = config.get('version')
    if aug_version == 3:
        return revised_augmentation_pipeline_legacy
    elif aug_version == 4:
        return augmentation_pipeline_v4
    elif aug_version == 5:
        return augmentation_pipeline_v5
    else:
        # Build custom pipeline from config
        ...
```

### `load_and_process_property_batch()` Logic
```python
def _load_single(frame_dict):
    if use_aug:
        if aug_version == 5:
            augmented = augmentation_pipeline_v5(image=img_np)['image']
        elif aug_version == 4:
            augmented = augmentation_pipeline_v4(image=img_np)['image']
        elif aug_version == 3:
            augmented = revised_augmentation_pipeline_legacy(image=img_np)['image']
        elif aug_version == 6:
            augmented = apply_augmentation_v6(img_np, frame_dict)
        elif aug_version == 7:
            augmented = apply_augmentation_v7(img_np)
        else:
            # Default to surgical
            pipeline = create_surgical_augmentation_pipeline(aug_params, frame_dict)
            augmented = pipeline(image=img_np)['image']
```

---

## Issues with Current System

### 1. Duplicated Pipelines
V4_GENERALIST in V6 is nearly identical to augmentation_pipeline_v4, but with minor differences:
- Different `brightness_limit`: 0.15 vs 0.12
- Different overall structure

### 2. Inconsistent Naming
- `revised_augmentation_pipeline_legacy` (V3)
- `augmentation_pipeline_v4` (V4)
- `augmentation_pipeline_v5` (V5)
- `AUG_PIPELINE_V6_SIMULATOR` (part of V6)
- `apply_augmentation_v6()` (V6 entry point)
- `apply_augmentation_v7()` (V7 entry point)

### 3. Scattered Selection Logic
Version selection happens in:
- `create_general_augmentation_pipeline()` (for non-property strategies)
- `load_and_process_property_batch()` (for property_balancing)
- Both have different default behaviors

### 4. Custom Transform Class
`CustomUnsharpMask` is defined in `dataloaders.py` but is a general-purpose transform that should be in its own module.

### 5. Helper Pipelines Not Reused
`degrade_quality_pipeline`, `enhance_quality_pipeline`, `social_media_pipeline` are defined twice (one version commented out).

---

## Proposed Consolidation

### Directory Structure
```
training/data/augmentations/
├── __init__.py           # Registry exports
├── registry.py           # Pipeline registration
├── transforms/
│   ├── __init__.py
│   └── unsharp_mask.py   # CustomUnsharpMask
├── pipelines/
│   ├── __init__.py
│   ├── v3_legacy.py
│   ├── v4_moderate.py
│   ├── v5_hybrid.py
│   ├── v6_portfolio.py
│   ├── v7_simple.py
│   └── surgical.py
└── helpers/
    ├── __init__.py
    ├── degradation.py    # degrade_quality_pipeline
    ├── enhancement.py    # enhance_quality_pipeline
    └── social_media.py   # social_media_pipeline
```

### Registry API
```python
# augmentations/registry.py
from typing import Callable, Dict, Any, Union
import albumentations as A

AugmentationFactory = Callable[[Dict[str, Any]], A.Compose]

_REGISTRY: Dict[Union[int, str], AugmentationFactory] = {}

def register(version: Union[int, str]):
    """Decorator to register an augmentation pipeline factory."""
    def decorator(factory: AugmentationFactory):
        _REGISTRY[version] = factory
        return factory
    return decorator

def get_pipeline(
    version: Union[int, str],
    config: Dict[str, Any] = None,
    frame_properties: Dict[str, Any] = None
) -> A.Compose:
    """Get augmentation pipeline by version."""
    if version not in _REGISTRY:
        raise ValueError(f"Unknown augmentation version: {version}. "
                        f"Available: {list(_REGISTRY.keys())}")
    
    factory = _REGISTRY[version]
    
    # Call factory with appropriate args
    if version == 'surgical':
        if frame_properties is None:
            raise ValueError("Surgical augmentation requires frame_properties")
        return factory(config or {}, frame_properties)
    elif version in (6,):
        # V6 needs frame_dict for source detection
        return factory(config or {}, frame_properties or {})
    else:
        return factory(config or {})

def list_versions() -> list:
    """List all registered augmentation versions."""
    return list(_REGISTRY.keys())
```

### Pipeline Registration
```python
# augmentations/pipelines/v3_legacy.py
from ..registry import register
from ..transforms.unsharp_mask import CustomUnsharpMask
import albumentations as A

@register(3)
@register('v3')
@register('legacy')
def create_v3_pipeline(config: dict = None) -> A.Compose:
    """V3: Conservative baseline pipeline."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        CustomUnsharpMask(blur_limit=(3, 9), alpha=(0.5, 1.0), threshold=10, p=0.7),
        A.OneOf([
            A.ImageCompression(quality_lower=50, quality_upper=90, p=0.5),
            A.GaussNoise(var_limit=(10.0, 60.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        ], p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.4),
    ])
```

### Usage After Refactor
```python
# In load_and_process_property_batch
from training.data.augmentations import get_pipeline

def _load_single(frame_dict):
    if use_aug:
        pipeline = get_pipeline(
            version=aug_version,
            config=aug_params,
            frame_properties=frame_dict
        )
        augmented = pipeline(image=img_np)['image']
```

---

## Version Selection Recommendations

Based on the analysis, here are recommendations for when to use each version:

| Version | Best For | When to Avoid |
|---------|----------|---------------|
| V3 | Debugging, baseline comparison | Production training |
| V4 | General training, slight improvement over V3 | When you need heavy robustness |
| V5 | Training for low-quality input robustness | When input quality is consistently high |
| V6 | Multi-source training with quality differences | Single-source training |
| V7 | Simplified portfolio without source awareness | When source differentiation matters |
| Surgical | Property-balanced strategy with rich metadata | Strategies without frame properties |

### Default Recommendation
For most training runs:
- **Without property_balancing:** Use V4 or V7
- **With property_balancing:** Use Surgical (default) or V6 if sources differ significantly
