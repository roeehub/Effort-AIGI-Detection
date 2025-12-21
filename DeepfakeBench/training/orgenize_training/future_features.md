# Future Features & Research Directions

> **Purpose:** Document planned features, experiments, and architectural extensions  
> **Goal:** Design the modular system to accommodate these future needs

---

## Overview

This document captures the research directions and features that the refactored training system should support. Each feature includes:
- **Motivation:** Why this matters
- **Requirements:** What the system needs to support it
- **Design considerations:** How it affects the architecture

---

## Feature Categories

1. [Model Architecture Extensions](#1-model-architecture-extensions)
2. [Data Pipeline Enhancements](#2-data-pipeline-enhancements)
3. [Augmentation Experiments](#3-augmentation-experiments)
4. [Training Strategies](#4-training-strategies)
5. [Observability & Debugging](#5-observability--debugging)

---

## 1. Model Architecture Extensions

### 1.1 Configurable CLIP Backbone

**Motivation:**  
Currently hardcoded to CLIP ViT-L/14. Want to experiment with:
- ViT-B/16 (smaller, faster)
- ViT-L/14@336 (higher resolution)
- ViT-H/14 (larger)
- OpenCLIP variants
- SigLIP
- DINOv2

**Current State:**
```python
# effort_detector.py - hardcoded path
self.clip_backbone_path = config['gcs_assets']['clip_backbone']['local_path']
# Always loads: models--openai--clip-vit-large-patch14
```

**Requirements:**
- [ ] Config option for backbone selection: `model.backbone: "openai/clip-vit-large-patch14"`
- [ ] Registry of supported backbones with their configs (resolution, embed dim, etc.)
- [ ] Dynamic SVD rank adjustment based on backbone size
- [ ] Normalization mean/std should come from backbone config, not hardcoded

**Proposed Config:**
```yaml
model:
  name: effort
  backbone:
    type: clip  # or: openclip, siglip, dinov2
    variant: ViT-L-14  # backbone-specific
    source: openai  # or: laion, apple
    pretrained: true
    resolution: 224  # can override default
  svd:
    rank: 1023  # or: auto (based on backbone)
    lambda_reg: 1.0
```

**Design Considerations:**
- Backbone registry should handle download/caching
- Each backbone type may need different feature extraction
- SVD layers need to know embedding dimension

---

### 1.2 Temporal Architecture (Future)

**Motivation:**  
Current model processes frames independently. Temporal information could help detect:
- Flickering artifacts
- Temporal inconsistencies in lip sync
- Motion artifacts in face swaps

**Requirements:**
- [ ] Sequential frame loading (N consecutive frames)
- [ ] Temporal modeling options (LSTM, Transformer, 3D Conv)
- [ ] Frame-level vs video-level predictions

**Proposed Config:**
```yaml
model:
  temporal:
    enabled: false
    type: transformer  # or: lstm, conv3d
    sequence_length: 8
    aggregation: mean  # or: attention, last
```

**Design Considerations:**
- Data loaders need to return consecutive frames
- Batching becomes: `[B, T, C, H, W]` instead of `[B, C, H, W]`
- May need different augmentation (consistent across sequence)

---

### 1.3 Multi-Head Detection

**Motivation:**  
Single binary classifier might miss nuances. Could benefit from:
- Auxiliary heads (manipulation type, method family)
- Multi-task learning
- Hierarchical classification

**Requirements:**
- [ ] Pluggable head architecture
- [ ] Multi-loss combination
- [ ] Per-head metrics

**Proposed Config:**
```yaml
model:
  heads:
    - name: binary
      type: linear
      output_dim: 2
      loss: cross_entropy
      weight: 1.0
    - name: method_family
      type: linear
      output_dim: 4  # FS, FR, EFS, Real
      loss: cross_entropy
      weight: 0.3
```

---

## 2. Data Pipeline Enhancements

### 2.1 Paired Real/Fake Video Batching ⭐ HIGH PRIORITY

**Motivation:**  
New dataset has paired videos where each fake has a known source real video. Training with pairs allows:
- Direct comparison learning
- Contrastive approaches
- Quality matching preprocessing

**Current State:**
- Real and fake videos are sampled independently
- No way to link a fake video to its source
- Metadata about source relationships not used

**Requirements:**
- [ ] Metadata schema for source-fake relationships
- [ ] New batching strategy: `paired_video`
- [ ] Batch structure: `[(real_1, fake_1), (real_2, fake_2), ...]`
- [ ] Option to apply matching preprocessing (see 2.2)

**Proposed Data Structure:**
```
gs://bucket/
├── real/
│   └── source_name/
│       └── video_id/
│           ├── frame_001.png
│           └── frame_002.png
└── fake/
    └── method_name/
        └── video_id/  # Same ID as source!
            ├── frame_001.png
            ├── frame_002.png
            └── metadata.json  # Contains source_video_id
```

**Proposed Config:**
```yaml
data:
  batching:
    strategy: paired_video
    pairs_per_batch: 4  # 4 pairs = 8 videos
    frames_per_video: 8
    # Ensures real and fake from same source are in same batch
    pairing:
      method: by_video_id  # or: by_metadata_field
      field: source_video_id  # metadata field for matching
```

**New Batching Strategy:**
```python
class PairedVideoBatchingStrategy(BatchingStrategy):
    """
    Creates batches where each fake video is paired with its source real video.
    """
    def create_train_loader(self, train_data, config):
        # Group fakes by their source
        fake_to_source = self._build_pairing_map(train_data)
        
        # Create pairs
        pairs = []
        for fake_video in fake_videos:
            source_id = fake_to_source[fake_video.video_id]
            source_video = self._find_source(source_id, real_videos)
            pairs.append((source_video, fake_video))
        
        # Batch pairs
        # Each batch: [(real_1, fake_1), (real_2, fake_2), ...]
        ...
```

---

### 2.2 Quality Matching Preprocessing ⭐ HIGH PRIORITY

**Motivation:**  
Fake videos often have obvious quality differences from real:
- Over-smoothed skin (common in face swaps)
- Compression artifacts from double-encoding
- Color/lighting inconsistencies

These are "shortcut" features that don't generalize. Preprocessing can hide them.

**Requirements:**
- [ ] Preprocessing registry (like augmentation registry)
- [ ] Configurable preprocessing pipeline
- [ ] Can be applied per-pair or globally

**Preprocessing Options:**
1. **Quality Degradation** - Degrade real to match fake quality
2. **Quality Enhancement** - Enhance fake to match real quality
3. **Mutual Degradation** - Degrade both to common lower quality
4. **Skin Smoothing** - Apply smoothing to real faces
5. **Noise Injection** - Add matching noise patterns

**Proposed Config:**
```yaml
data:
  preprocessing:
    enabled: true
    strategy: mutual_degradation  # or: match_to_fake, match_to_real
    operations:
      - type: gaussian_blur
        kernel_size: [3, 5]
        apply_to: both  # or: real_only, fake_only
      - type: jpeg_compression
        quality: [70, 90]
        apply_to: both
      - type: skin_smoothing
        strength: 0.3
        apply_to: real_only  # Make real look more "fake-like"
```

**Design Considerations:**
- Should happen BEFORE augmentation
- Needs face detection/segmentation for skin smoothing
- May need quality estimation to match levels

---

### 2.3 Facial Landmark Integration ⭐ MEDIUM PRIORITY

**Motivation:**  
Have facial landmarks in cloud storage. Can use for:
- Landmark-based augmentation (occlude specific regions)
- Attention guidance
- Region-specific feature extraction

**Current State:**
- Landmarks computed and stored but not loaded
- No integration with data pipeline

**Requirements:**
- [ ] Landmark loading alongside frames
- [ ] Landmark format standardization (68-point, MediaPipe, etc.)
- [ ] Landmark-aware transforms

**Proposed Data Structure:**
```
gs://bucket/
└── real/
    └── source_name/
        └── video_id/
            ├── frame_001.png
            ├── frame_001_landmarks.json  # or .npy
            └── frame_002.png
```

**Proposed Config:**
```yaml
data:
  loading:
    load_landmarks: true
    landmark_format: mediapipe  # or: dlib_68, fan
    landmark_file_pattern: "{frame_name}_landmarks.json"
```

---

### 2.4 Metadata-Aware Data Loading

**Motivation:**  
Want to track and use rich metadata:
- Source video origin (YouTube, custom, etc.)
- Recording quality
- Face properties (pose, expression, occlusion)
- Manipulation parameters (for synthetic data)

**Requirements:**
- [ ] Flexible metadata schema
- [ ] Metadata-based filtering
- [ ] Metadata available in batch for logging/analysis

**Proposed Config:**
```yaml
data:
  metadata:
    enabled: true
    source: parquet  # or: json_sidecar, database
    path: gs://bucket/metadata/frame_metadata.parquet
    required_fields:
      - video_id
      - method
      - label
    optional_fields:
      - source_video_id
      - quality_score
      - face_pose
      - landmarks_path
```

---

## 3. Augmentation Experiments

### 3.1 AugMix Integration ⭐ MEDIUM PRIORITY

**Motivation:**  
AugMix is a consistency-regularization augmentation that:
- Creates multiple augmented versions of same image
- Mixes them together
- Adds consistency loss between original and augmented

Paper: "AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty"

**Requirements:**
- [ ] AugMix implementation compatible with albumentations
- [ ] JSD consistency loss option
- [ ] Integration with training loop (needs multiple views)

**Proposed Config:**
```yaml
augmentation:
  version: augmix
  augmix:
    severity: 3
    width: 3  # number of augmentation chains
    depth: -1  # -1 = random
    alpha: 1.0  # mixing coefficient
    use_jsd_loss: true
    jsd_weight: 12.0
```

**Design Considerations:**
- Batch becomes: `{image, aug1, aug2, aug3}` for each sample
- Loss: `CE(image) + λ * JSD(image, aug1, aug2, aug3)`
- Memory usage increases ~4x

---

### 3.2 Facial Region Occlusion ⭐ MEDIUM PRIORITY

**Motivation:**  
Force model to not rely on single facial region by randomly occluding:
- Mouth (lip sync artifacts)
- Eyes (gaze inconsistencies)
- Nose (blending boundary)
- Forehead (hair boundary)

**Requirements:**
- [ ] Landmark-based region detection (see 2.3)
- [ ] Region occlusion transform
- [ ] Configurable occlusion strategies (black, blur, noise)

**Proposed Config:**
```yaml
augmentation:
  facial_occlusion:
    enabled: true
    probability: 0.3
    regions:
      - name: mouth
        probability: 0.4
        method: gaussian_blur  # or: black, noise, pixelate
        blur_sigma: [10, 20]
      - name: eyes
        probability: 0.3
        method: black
      - name: nose
        probability: 0.2
        method: noise
```

**Implementation:**
```python
class FacialOcclusionTransform:
    def __init__(self, config, landmark_indices):
        self.regions = config.regions
        self.landmark_indices = landmark_indices  # mapping name -> indices
    
    def __call__(self, image, landmarks):
        if random.random() > self.probability:
            return image
        
        # Select region to occlude
        region = random.choices(
            self.regions, 
            weights=[r.probability for r in self.regions]
        )[0]
        
        # Get region mask from landmarks
        mask = self._get_region_mask(landmarks, region.name)
        
        # Apply occlusion
        return self._apply_occlusion(image, mask, region.method)
```

---

### 3.3 Copy-Paste Augmentation

**Motivation:**  
Paste faces from other images onto current image to:
- Simulate face swaps during training
- Create hard negatives
- Increase face diversity

**Requirements:**
- [ ] Face crop database
- [ ] Face alignment for pasting
- [ ] Blending options

---

## 4. Training Strategies

### 4.1 Contrastive Learning with Pairs

**Motivation:**  
With paired real/fake videos (2.1), can use contrastive losses:
- Pull real/fake from different sources apart
- Push different frames from same video together
- Learn manipulation-invariant features

**Requirements:**
- [ ] Contrastive loss implementations (InfoNCE, NT-Xent)
- [ ] Pair/triplet mining strategies
- [ ] Feature extraction for contrastive (before classifier)

**Proposed Config:**
```yaml
training:
  contrastive:
    enabled: true
    loss: infonce
    temperature: 0.07
    weight: 0.5  # combined with CE loss
    pairs:
      positive: same_video  # frames from same video
      negative: different_label  # real vs fake
```

---

### 4.2 Curriculum Learning Improvements

**Current State:**  
Have basic curriculum via "lesson gates" but it's complex and brittle.

**Improvements Needed:**
- [ ] Cleaner curriculum stage definitions
- [ ] Automatic progression based on metrics
- [ ] Data curriculum (easy→hard samples)

**Proposed Config:**
```yaml
training:
  curriculum:
    enabled: true
    stages:
      - name: warmup
        epochs: 2
        data_filter:
          quality: high  # only high-quality samples
        augmentation: v3  # mild augmentation
      - name: main
        epochs: 8
        data_filter: null  # all samples
        augmentation: v5  # aggressive augmentation
      - name: finetune
        epochs: 2
        learning_rate: 0.00002  # lower LR
        augmentation: v3
```

---

## 5. Observability & Debugging ⭐ HIGH PRIORITY

### 5.1 Pipeline Visualization

**Motivation:**  
Need to SEE what's happening in the pipeline:
- What does a batch look like?
- What augmentations are applied?
- What's the data distribution?

**Requirements:**
- [ ] Batch visualization utility
- [ ] Augmentation before/after viewer
- [ ] Data distribution plots

**Proposed Tools:**
```python
# Visualize a batch
from training.debug import visualize_batch

loader = create_train_loader(...)
batch = next(iter(loader))
visualize_batch(batch, save_path="debug/batch_sample.png")
# Creates grid showing images, labels, metadata

# Visualize augmentation
from training.debug import visualize_augmentation

visualize_augmentation(
    image_path="gs://bucket/sample.png",
    aug_version=5,
    n_samples=10,
    save_path="debug/aug_samples.png"
)

# Data distribution
from training.debug import plot_data_distribution

plot_data_distribution(
    train_data,
    by=['label', 'method', 'sharpness_bucket'],
    save_path="debug/data_dist.png"
)
```

---

### 5.2 Dry Run Mode

**Motivation:**  
Quickly verify pipeline without full training:
- Check data loads correctly
- Verify batch shapes
- Test augmentations
- Validate config

**Proposed CLI:**
```bash
# Dry run - just setup, no training
python train.py --config config.yaml --dry-run

# Dry run with batch samples
python train.py --config config.yaml --dry-run --save-samples 10

# Dry run with full data stats
python train.py --config config.yaml --dry-run --full-stats
```

**Output:**
```
=== DRY RUN MODE ===

Configuration:
  Model: effort
  Backbone: openai/clip-vit-large-patch14
  Epochs: 10
  Batch size: 64

Data Pipeline:
  ✓ GCS bucket accessible
  ✓ Loaded 125,432 frame records
  ✓ Split: 100,345 train / 12,543 val_in / 12,544 val_out
  
  Method distribution:
    FaceForensics++ .......... 15,432 (15.4%)
    simswap .................. 12,345 (12.3%)
    ...

Batch Test:
  ✓ Created train loader: 1,568 batches
  ✓ Sample batch shape: [64, 3, 224, 224]
  ✓ Labels balanced: 32 real, 32 fake
  
  Saved 10 sample batches to debug/dry_run_samples/

Model Test:
  ✓ Model initialized
  ✓ Forward pass successful
  ✓ Backward pass successful
  ✓ Parameter count: 428,234,567

Ready to train!
```

---

### 5.3 Interactive Debugging

**Motivation:**  
Sometimes need to debug interactively:
- Inspect specific frames that fail
- Test augmentations manually
- Check model predictions

**Proposed Tools:**
```python
# Interactive batch inspector
from training.debug import BatchInspector

inspector = BatchInspector(train_loader)
inspector.show_batch(0)  # Show first batch
inspector.find_by_method('simswap')  # Find samples by method
inspector.show_failures(model)  # Show misclassified samples

# Augmentation playground
from training.debug import AugmentationPlayground

playground = AugmentationPlayground()
playground.load_image("gs://bucket/sample.png")
playground.apply(version=5)
playground.compare()  # Side-by-side view
playground.save("debug/aug_comparison.png")
```

---

## Architecture Implications

### For the modular system to support these features:

1. **Backbone Abstraction**
   - Registry of backbones with their properties
   - Common interface: `backbone.extract_features(images) -> embeddings`

2. **Data Pipeline Abstraction**
   - Pluggable data sources (GCS, local, database)
   - Pluggable metadata (parquet, JSON, none)
   - Pluggable pairing (none, by_id, by_metadata)
   - Pluggable batching (frame, video, paired, temporal)

3. **Augmentation Abstraction**
   - Registry with versioned pipelines
   - Support for stateful augmentations (AugMix)
   - Support for landmark-dependent augmentations
   - Pre-augmentation preprocessing hooks

4. **Training Abstraction**
   - Pluggable losses (CE, focal, contrastive, JSD)
   - Pluggable heads (single, multi-task)
   - Curriculum stages as first-class concept

5. **Observability**
   - Every component should be inspectable
   - Dry-run should test entire pipeline
   - Debug utilities as part of the package

---

## Priority Matrix

| Feature | Impact | Effort | Priority |
|---------|--------|--------|----------|
| Configurable backbone | High | Medium | P1 |
| Paired video batching | High | Medium | P1 |
| Quality matching preproc | High | Medium | P1 |
| Pipeline visualization | High | Low | P1 |
| Dry run mode | High | Low | P1 |
| Facial landmark loading | Medium | Low | P2 |
| Facial occlusion aug | Medium | Medium | P2 |
| AugMix | Medium | Medium | P2 |
| Metadata-aware loading | Medium | Low | P2 |
| Temporal architecture | Medium | High | P3 |
| Contrastive learning | Medium | High | P3 |
| Multi-head detection | Low | Medium | P3 |

---

## Next Steps

1. **Ensure base refactoring supports extension points** for these features
2. **Start with observability** (dry run, visualization) - helps with all other development
3. **Then paired batching** - unlocks new dataset structure
4. **Then backbone configurability** - enables architecture experiments
