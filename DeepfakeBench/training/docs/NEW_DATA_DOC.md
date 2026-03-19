# GCS Dataset Documentation

> **Last Updated:** February 2026  
> **Dataset Version:** 1.1

This document describes the deepfake detection dataset stored on Google Cloud Storage (GCS). The dataset contains paired real/fake video frames extracted from various face-swapping methods, with identity-based quality tiers and comprehensive metadata.

---

## Table of Contents

1. [Overview](#overview)
2. [GCS Buckets](#gcs-buckets)
3. [Data Structure](#data-structure)
4. [Manifest Schema](#manifest-schema)
5. [Tier System](#tier-system)
6. [Swap Models](#swap-models)
7. [Sample Distribution](#sample-distribution)
8. [Frame Extraction Details](#frame-extraction-details)
9. [Cropped Frames Bucket](#cropped-frames-bucket)
10. [Data Loader Considerations](#data-loader-considerations)
11. [Example Queries](#example-queries)

---

## Overview

The dataset consists of **temporally-aligned real/fake frame pairs** extracted from face-swapping videos. Each sample contains:

- **16 frames** from the real video
- **16 corresponding frames** from the fake (swapped) video
- **Facial landmarks** for each frame
- **Metadata** including identity deltas and quality tiers

The frames are extracted at the **same indices** from both real and fake videos, ensuring temporal alignment for paired comparison.

---

## GCS Buckets

| Bucket Name | Contents | Sample Count |
|-------------|----------|--------------|
| `live-deepfake-methods-real-and-fake-frames` | Full frames + landmarks + manifests | 5,089 |
| `live-deepfake-methods-real-and-fake-videos` | Source videos (real.mp4 + fake.mp4) | 6,453 |
| `live-deepfake-methods-real-and-fake-frames-cropped` | Cropped face frames | 6,449 |

### Primary Bucket for Training: `live-deepfake-methods-real-and-fake-frames-cropped`

This bucket contains **cropped face regions** optimized for training deepfake detection models.

---

## Data Structure

### Frames Bucket Structure
```
gs://live-deepfake-methods-real-and-fake-frames/
└── samples/
    └── {sample_id}/
        ├── manifest.json
        ├── frames/
        │   ├── real/
        │   │   ├── frame_0000.png
        │   │   ├── frame_0001.png
        │   │   └── ... (16 frames total)
        │   └── fake/
        │       ├── frame_0000.png
        │       ├── frame_0001.png
        │       └── ... (16 frames total)
        └── landmarks/
            ├── real_landmarks.json
            ├── fake_landmarks.json
            └── metadata.json
```

### Videos Bucket Structure
```
gs://live-deepfake-methods-real-and-fake-videos/
└── {sample_id}/
    ├── real.mp4
    └── fake.mp4
```

### Cropped Frames Bucket Structure
```
gs://live-deepfake-methods-real-and-fake-frames-cropped/
└── samples/
    └── {sample_id}/
        ├── manifest.json
        └── frames/
            ├── real/
            │   ├── frame_0000.png
            │   ├── frame_0001.png
            │   └── ... (16 frames)
            └── fake/
                ├── frame_0000.png
                ├── frame_0001.png
                └── ... (16 frames)
```

---

## Manifest Schema

### Frames Bucket Manifest (v1.1)

```json
{
  "sample_id": "visomaster_CSCS_00010",
  "strategy": "consecutive",
  "source": "visomaster",
  "swap_model": "CSCS",
  "frame_count": 16,
  "has_landmarks": true,
  "tier_data": {
    "identity_delta": 0.7234,
    "identity_delta_tier": "STRONG",
    "identity_delta_std": 0.0156
  },
  "source_face": {
    "face_id": "face_001",
    "path": "source_faces/face_001.png"
  },
  "manifest_version": "1.1",
  "manifest_updated_at": "2026-02-06T12:34:56Z"
}
```

### Cropped Bucket Manifest

```json
{
  "sample_id": "visomaster_CSCS_00010",
  "strategy": "consecutive",
  "source": "visomaster",
  "swap_model": "CSCS",
  "frame_count": 16
}
```

> **Note:** Cropped bucket manifests may not include `tier_data`. Use the frames bucket manifest as the authoritative source for tier information, matching by `sample_id`.

---

## Tier System

Samples are categorized by **identity delta** - a metric measuring the difference between the source face and the swapped face in the fake video. Higher deltas indicate more visible face-swapping artifacts.

### Tier Definitions

| Tier | Identity Delta Range | Description |
|------|---------------------|-------------|
| **STRONG** | δ ≥ 0.30 | High identity difference, easier to detect |
| **MODERATE** | 0.15 ≤ δ < 0.30 | Medium identity difference |
| **MINIMAL** | δ < 0.15 | Low identity difference, harder to detect |

### Tier Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `identity_delta` | float | Mean cosine distance between source face and swapped face embeddings across all frames |
| `identity_delta_tier` | string | Categorical tier: "STRONG", "MODERATE", or "MINIMAL" |
| `identity_delta_std` | float | Standard deviation of identity delta across frames |

### Why Tiers Matter

- **Balanced Training:** Use tier-aware sampling to avoid bias toward easy-to-detect swaps
- **Curriculum Learning:** Start with STRONG tier, progressively add MODERATE and MINIMAL
- **Evaluation:** Report metrics per-tier to understand model performance on different difficulty levels

---

## Swap Models

The dataset includes samples from multiple face-swapping methods:

### VisoMaster Models (Primary)
| Model | Sample Count | Description |
|-------|--------------|-------------|
| `CSCS` | ~601 | VisoMaster CSCS method |
| `GhostFace-v1` | ~602 | GhostFace version 1 |
| `GhostFace-v2` | ~601 | GhostFace version 2 |
| `GhostFace-v3` | ~602 | GhostFace version 3 |
| `InStyleSwapper256-A` | ~601 | InStyle variant A |
| `InStyleSwapper256-B` | ~648 | InStyle variant B |
| `Inswapper128` | ~575 | Inswapper 128px |

### DeepLiveCam Models (Legacy)
| Model | Sample Count | Description |
|-------|--------------|-------------|
| `edge_cases` | ~434 | Edge case samples |
| `minimal_processing` | ~425 | Minimal processing samples |

### Additional Models (Cropped Bucket Only)
| Model | Sample Count | Notes |
|-------|--------------|-------|
| `SimSwap512` | ~602 | Only in cropped bucket |
| `InStyleSwapper256-C` | ~702 | Only in cropped bucket |

---

## Sample Distribution

### By Source
- **VisoMaster:** ~4,230 samples (83%)
- **DeepLiveCam:** ~859 samples (17%)

### By Tier (Approximate)
- **STRONG:** ~35%
- **MODERATE:** ~40%
- **MINIMAL:** ~25%

---

## Frame Extraction Details

### Extraction Strategy: "consecutive"

Each sample contains **16 frames** extracted using a consecutive sampling strategy:

1. **8 Anchor Frames:** Evenly spaced across the video
2. **8 Consecutive Frames:** One frame immediately following each anchor

```
Video Timeline:  [====|=====|=====|=====|=====|=====|=====|=====|====]
                   ↑     ↑      ↑      ↑      ↑      ↑      ↑      ↑
Anchors:          A0    A1     A2     A3     A4     A5     A6     A7
Consecutive:      C0    C1     C2     C3     C4     C5     C6     C7

Frame Pairs: [(A0,C0), (A1,C1), (A2,C2), (A3,C3), (A4,C4), (A5,C5), (A6,C6), (A7,C7)]
```

### Frame Naming Convention

```
frame_0000.png  # First anchor
frame_0001.png  # First consecutive
frame_0002.png  # Second anchor
frame_0003.png  # Second consecutive
...
frame_0014.png  # Eighth anchor
frame_0015.png  # Eighth consecutive
```

### Temporal Alignment

**Critical:** Real and fake frames are extracted at the **exact same frame indices** from their respective videos. This ensures:

- `real/frame_0000.png` and `fake/frame_0000.png` correspond to the same moment in time
- Direct pixel-level comparison is meaningful
- Temporal artifacts can be studied consistently

### Edge Cases

~129 samples have **14-15 frames** instead of 16 due to shorter source videos. Check `frame_count` in manifest before loading.

---

## Cropped Frames Bucket

### Recommended for Training

The cropped frames bucket contains **face-region crops** that are:

- Pre-aligned to face bounding boxes
- Consistent crop sizes within each sample
- Ready for direct input to detection models

### Crop Specifications

- **Detection Method:** YOLO face detection
- **Crop Padding:** Includes margin around detected face
- **Alignment:** Center-aligned on face

### Sample ID Matching

To get tier data for cropped samples:

```python
# Cropped bucket sample
cropped_sample_id = "visomaster_CSCS_00010"

# Fetch tier_data from frames bucket manifest
frames_manifest_path = f"gs://live-deepfake-methods-real-and-fake-frames/samples/{cropped_sample_id}/manifest.json"
```

---

## Data Loader Considerations

### Recommended Loading Pattern

```python
class DeepfakeDataset:
    def __init__(self, 
                 bucket_name: str,
                 tier_filter: List[str] = None,      # ["STRONG", "MODERATE", "MINIMAL"]
                 model_filter: List[str] = None,     # ["CSCS", "GhostFace-v1", ...]
                 source_filter: List[str] = None):   # ["visomaster", "deeplive"]
        pass
    
    def __getitem__(self, idx):
        # Return: {
        #   "real_frames": Tensor[16, C, H, W],
        #   "fake_frames": Tensor[16, C, H, W],
        #   "sample_id": str,
        #   "tier": str,
        #   "swap_model": str,
        #   "identity_delta": float
        # }
        pass
```

### Filtering Recommendations

1. **Tier-Balanced Sampling:** Sample equally from each tier to avoid bias
2. **Model-Stratified:** Ensure all swap models represented in each batch
3. **Train/Val/Test Split:** Split by sample_id, not by frame

### Caching Strategy

1. **Manifest Cache:** Download all manifests once, filter locally
2. **Frame Cache:** Stream frames on-demand or pre-download to local SSD
3. **Tier Index:** Build local index mapping sample_id → tier for fast filtering

### GCS Access Patterns

```python
# List all samples
gsutil ls gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/

# Download single sample
gsutil -m cp -r gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/{sample_id}/ ./local/

# Stream with Python
from google.cloud import storage
client = storage.Client()
bucket = client.bucket("live-deepfake-methods-real-and-fake-frames-cropped")
blob = bucket.blob(f"samples/{sample_id}/frames/real/frame_0000.png")
image_bytes = blob.download_as_bytes()
```

---

## Example Queries

### List All Samples
```bash
gsutil ls gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/
```

### Get Manifest
```bash
gsutil cat gs://live-deepfake-methods-real-and-fake-frames/samples/visomaster_CSCS_00010/manifest.json
```

### Download Sample with Tier Data
```bash
# Get tier from frames bucket
TIER=$(gsutil cat gs://live-deepfake-methods-real-and-fake-frames/samples/$SAMPLE_ID/manifest.json | jq -r '.tier_data.identity_delta_tier')

# Download cropped frames
gsutil -m cp -r gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/$SAMPLE_ID/ ./data/
```

### Count Samples by Model
```bash
gsutil ls gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/ | \
  sed 's/.*samples\/\([^_]*_[^_]*\).*/\1/' | \
  sort | uniq -c
```

---

## Quick Reference Card

| Item | Value |
|------|-------|
| **Total Samples (Cropped)** | ~6,449 |
| **Frames per Sample** | 16 (8 anchor + 8 consecutive) |
| **Frame Format** | PNG |
| **Manifest Version** | 1.1 |
| **Tier Categories** | STRONG, MODERATE, MINIMAL |
| **Swap Models** | 11 (9 in frames bucket, 2 additional in cropped) |
| **Temporal Alignment** | Yes - same frame indices for real/fake |

---

## Contact

For questions about this dataset, refer to the generation scripts in:
- `Deep-Live-Cam/dataset/` - Frame extraction and processing
- `VisoMaster/cli/` - Face swapping pipeline

---

*This documentation is auto-generated. For the latest sample counts and statistics, query the buckets directly.*
