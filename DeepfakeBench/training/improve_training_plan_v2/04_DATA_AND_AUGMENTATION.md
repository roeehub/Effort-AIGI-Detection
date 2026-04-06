# Data & Augmentation

This document covers datasets, data loading strategies, and augmentation pipelines.

## 1. Datasets Overview

### 1.1 DeepLive Dataset

**Location:** `gs://live-deepfake-methods-real-and-fake-frames/`

**Structure:**
```
samples/{sample_id}/
├── manifest.json           # Sample metadata
├── frames/
│   ├── real/*.png         # 16 real frames
│   └── fake/*.png         # 16 fake frames
└── landmarks/*.json        # MediaPipe 478-point landmarks
```

**Characteristics:**
- ~340 paired samples (real/fake from same source)
- 16 frames per sample (sparse sampling uses 8)
- Includes face landmarks for occlusion augmentation
- Multiple manipulation strategies

**Configuration:**
```yaml
data_source: deeplive

deeplive:
  gcs_bucket: "live-deepfake-methods-real-and-fake-frames"
  sampling_mode: "sparse"  # sparse | full | pairs
  anchor_indices: [0, 2, 4, 6, 8, 10, 12, 14]
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  seed: 737
```

### 1.2 Frame Sampling Modes

| Mode | Frames | Indices | Use Case |
|------|--------|---------|----------|
| `sparse` | 8 | [0,2,4,6,8,10,12,14] | Default, efficient training |
| `full` | 16 | [0..15] | Maximum data usage |
| `pairs` | 8 pairs | anchor + consecutive | Temporal analysis |

### 1.3 Other Supported Datasets

The codebase also supports:
- **FaceForensics++**: Classic deepfake benchmark
- **Manifest-based datasets**: Custom unified manifests from GCS

## 2. Dataloader Strategies

### 2.1 Strategy Overview

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `deeplive` | Paired real/fake sampling | DeepLive dataset |
| `frame_level` | Random frame sampling | Large datasets |
| `video_level` | Sample frames from selected videos | Video-aware training |
| `property_balancing` | Balance by frame properties | Addressing distribution bias |
| `per_method` | Balance across manipulation methods | Multi-method datasets |

### 2.2 DeepLive Strategy

```python
# dataset/deeplive_dataset.py
class DeepLiveIterableDataset(IterableDataset):
    def __iter__(self):
        for sample in samples:
            for frame_idx in anchor_indices:
                # Yield paired real and fake frames
                real_frame = load_frame(sample.real_frame_paths[frame_idx])
                fake_frame = load_frame(sample.fake_frame_paths[frame_idx])
                
                yield {'image': real_frame, 'label': 0}  # Real
                yield {'image': fake_frame, 'label': 1}  # Fake
```

**Key Properties:**
- Always balanced (1:1 real:fake ratio)
- Frames are paired (same source video)
- Landmarks available for augmentation

### 2.3 Property Balancing Strategy

```python
# Frames are bucketed by sharpness (q1=blurry, q4=sharp)
# DataLoader samples equally from all buckets
frames_by_bucket = {
    'real_q1': [...],  # Blurry real frames
    'real_q4': [...],  # Sharp real frames
    'fake_q1': [...],  # Blurry fake frames
    'fake_q4': [...],  # Sharp fake frames
}
```

**Configuration:**
```yaml
dataloader_strategy: property_balancing
dataloader_params:
  frames_per_batch: 64
  frames_per_video: 2
```

### 2.4 Per-Method Strategy

For datasets with multiple manipulation methods:

```python
# Weighted sampling based on method frequency
method_weights = {
    'Deepfakes': 0.3,
    'Face2Face': 0.25,
    'FaceSwap': 0.25,
    'NeuralTextures': 0.2,
}

# Each batch samples from methods according to weights
```

## 3. Data Splits

### 3.1 Default Split
```yaml
train_split: 0.8  # 80% training
val_split: 0.1    # 10% validation (in-distribution)
test_split: 0.1   # 10% test (holdout)
seed: 737         # Reproducible splits
```

### 3.2 Validation Sets

| Set | Purpose | Typical Use |
|-----|---------|-------------|
| `val_in_dist` | In-distribution validation | Early stopping, hyperparameter tuning |
| `val_holdout` | Held-out test set | Final evaluation |
| `ood_loader` | Out-of-distribution | Generalization testing |

## 4. Augmentation Pipelines

### 4.1 Version Overview

| Version | Name | Components |
|---------|------|------------|
| 3 | Legacy | Flip, brightness, contrast |
| 4 | Generalist | Comprehensive image augmentations |
| 5 | Medium | Balanced augmentation strength |
| 6 | Social Media Simulator | JPEG, blur, downscale |
| 7 | Landmark Occlusion | Face part masking |

### 4.2 Augmentation V7 (Landmark Occlusion)

**Purpose:** Force model to learn from partial face regions, improving robustness.

**Configuration:**
```yaml
augmentation:
  version: "landmark_occlusion"
  occlusion_type: "mixed"       # solid | blur | pixelate | mixed
  regions:
    - "left_eye"
    - "right_eye"
    - "nose"
    - "mouth"
  num_regions: [1, 2]           # Occlude 1-2 regions per image
  occlusion_prob: 0.2           # 20% of images get occlusion
  landmark_format: "mediapipe"  # 478-point landmarks
```

**Occlusion Types:**
- `solid`: Fill region with solid color
- `blur`: Gaussian blur on region
- `pixelate`: Pixelate region
- `mixed`: Random selection

**Regions Available:**
```python
LANDMARK_REGIONS = {
    'left_eye': [33, 133, 160, 159, 158, 144, 145, 153],
    'right_eye': [362, 263, 387, 386, 385, 373, 374, 380],
    'nose': [1, 2, 98, 327, 168],
    'mouth': [13, 14, 78, 308, 402, 311, 312, 317, 318],
    'left_eyebrow': [70, 63, 105, 66, 107],
    'right_eyebrow': [336, 296, 334, 293, 300],
}
```

### 4.3 Augmentation V6 (Social Media Simulator)

**Purpose:** Simulate real-world image degradation.

```python
social_media_pipeline = A.Compose([
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.Downscale(scale_min=0.5, scale_max=0.75, p=0.8),
    A.ImageCompression(quality_lower=30, quality_upper=60, p=1.0),
])
```

### 4.4 Surgical Augmentation (Property-Based)

**Purpose:** Counter-example generation based on frame properties.

```python
def create_surgical_augmentation_pipeline(config, frame_properties):
    transforms = [A.HorizontalFlip(p=0.5)]
    
    # Sharp images → degrade quality
    if frame_properties['sharpness_bucket'] == 'q4':
        transforms.append(degrade_quality_pipeline)
    
    # Blurry images → enhance quality
    elif frame_properties['sharpness_bucket'] == 'q1':
        transforms.append(enhance_quality_pipeline)
    
    return A.Compose(transforms)
```

### 4.5 Using Augmentations

```python
# Registry-based access
from data.augmentations import get_pipeline

pipeline = get_pipeline(version=7)  # or 'landmark_occlusion'
augmented = pipeline(image=image)['image']

# Version-specific functions
from data.augmentations.pipelines import apply_augmentation_v7

augmented = apply_augmentation_v7(
    image, 
    landmarks=landmarks,
    config=augmentation_config
)
```

## 5. Normalization

All images are normalized using CLIP's ImageNet-derived statistics:

```yaml
mean: [0.48145466, 0.4578275, 0.40821073]
std: [0.26862954, 0.26130258, 0.27577711]
```

```python
normalize = T.Normalize(mean=mean, std=std)
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    normalize,
])
```

## 6. Batch Construction

### 6.1 DeepLive Batch Example

```python
batch = {
    'image': torch.Tensor([B, 3, 224, 224]),  # Normalized images
    'label': torch.Tensor([B]),                # 0=real, 1=fake
    'frame_path': ['path1', 'path2', ...],    # For debugging
    'sample_id': ['id1', 'id1', ...],         # Sample grouping
}
```

### 6.2 Video-Level Batch

```python
# When frames_per_video > 1
batch = {
    'image': torch.Tensor([B, T, 3, 224, 224]),  # B videos, T frames each
    'label': torch.Tensor([B]),                   # One label per video
}

# Flattened in forward pass:
image = image.view(B * T, 3, 224, 224)
label = label.repeat_interleave(T)
```

## 7. Data Loading Configuration

### 7.1 Typical DeepLive Config

```yaml
dataloader_strategy: deeplive
frames_per_batch: 32     # GPU batch size
frames_per_video: 8      # Frames per sample
num_workers: 4           # DataLoader workers
prefetch_factor: 2       # Batches per worker to prefetch
```

### 7.2 Calculation Example

For ~340 DeepLive samples with 80/10/10 split:
- Train samples: 272
- Frames per sample: 8 anchors × 2 (real+fake) = 16
- Total train frames: 272 × 16 = 4,352
- Steps per epoch: 4,352 / 32 = ~136

## 8. GCS Integration

### 8.1 Frame Loading

```python
from google.cloud import storage

def load_frame_from_gcs(bucket_name, blob_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    image_bytes = blob.download_as_bytes()
    image = Image.open(io.BytesIO(image_bytes))
    return image
```

### 8.2 Local Caching

Optional local caching for faster iteration:

```python
DeepLiveDataset(
    bucket_name="...",
    local_cache_dir="/tmp/deeplive_cache"
)
```

---

*See also: [05_METRICS_AND_LOGGING.md](05_METRICS_AND_LOGGING.md) for monitoring training*
