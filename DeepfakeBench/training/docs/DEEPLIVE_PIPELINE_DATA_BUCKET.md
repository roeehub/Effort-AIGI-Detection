# GCS Dataset Structure for Deepfake Detection Training

## Quick Start

```python
from google.cloud import storage

PROJECT = "train-cvit2"
BUCKET_FRAMES = "live-deepfake-methods-real-and-fake-frames"

client = storage.Client(project=PROJECT)
bucket = client.bucket(BUCKET_FRAMES)

# List all available samples
for blob in bucket.list_blobs(prefix="samples/", delimiter="/"):
    if blob.name.endswith("/manifest.json"):
        print(blob.name.split("/")[1])  # sample_id
```

## Bucket Overview

| Bucket | Purpose | Contents |
|--------|---------|----------|
| `live-deepfake-methods-real-and-fake-frames` | **Training data** | Frames, landmarks, manifests |
| `live-deepfake-methods-real-and-fake-videos` | Reference only | Source videos (not needed for training) |

## Data Structure

```
gs://live-deepfake-methods-real-and-fake-frames/
└── samples/
    └── {sample_id}/                    # e.g., "edge_cases_0004"
        ├── manifest.json               # Sample metadata (uploaded last = completion marker)
        ├── frames/
        │   ├── real/                   # Original person (ground truth)
        │   │   ├── frame_0000.png      # 16 frames per video
        │   │   ├── frame_0001.png
        │   │   └── ... frame_0015.png
        │   └── fake/                   # Deepfake output
        │       ├── frame_0000.png
        │       └── ... frame_0015.png
        └── landmarks/
            ├── real_landmarks.json     # Face landmarks for real frames
            ├── fake_landmarks.json     # Face landmarks for fake frames
            └── metadata.json           # Landmark extraction status
```

## Labels

| Folder | Label | Description |
|--------|-------|-------------|
| `frames/real/` | **Real (0)** | Original unmodified face |
| `frames/fake/` | **Fake (1)** | Deepfake-generated face |

Each sample contains **paired frames** - the same frame index in `real/` and `fake/` corresponds to the same moment in time, enabling direct comparison.

## Frame Selection Strategy

Each video yields **16 frames** organized as 8 anchor-consecutive pairs:

| Index | Type | Description |
|-------|------|-------------|
| 0, 2, 4, 6, 8, 10, 12, 14 | **Anchor** | Uniformly distributed across video duration |
| 1, 3, 5, 7, 9, 11, 13, 15 | **Consecutive** | Frame immediately after each anchor |

This enables:
- **Sparse sampling**: Use only anchors (8 frames) for efficiency
- **Temporal analysis**: Compare anchor vs consecutive for motion/temporal artifacts
- **Full sampling**: Use all 16 frames for maximum data

## Landmarks Format

Each `*_landmarks.json` contains per-frame face landmarks from MediaPipe Face Landmarker:

```json
{
  "source": "real",
  "sample_id": "edge_cases_0004",
  "model": "face_landmarker.task",
  "frames": [
    {
      "frame_index": 0,
      "face_detected": true,
      "regions": {
        "left_eye": {
          "landmarks": [{"x": 0.35, "y": 0.42, "z": 0.01}, ...],
          "bbox": {"x_min": 0.30, "y_min": 0.38, "x_max": 0.42, "y_max": 0.46}
        },
        "right_eye": { ... },
        "nose": { ... },
        "lips": { ... },
        "left_eyebrow": { ... },
        "right_eyebrow": { ... },
        "face_oval": { ... }
      },
      "blendshapes": {
        "browDownLeft": 0.12,
        "browDownRight": 0.11,
        "eyeBlinkLeft": 0.02,
        ...
      }
    },
    ...
  ]
}
```

**Landmark coordinates** are normalized [0, 1] relative to image dimensions.

### Using Landmarks for Augmentation

```python
def get_eye_region(landmarks, frame_shape):
    """Extract eye bounding box in pixel coordinates."""
    h, w = frame_shape[:2]
    left_eye = landmarks["regions"]["left_eye"]["bbox"]
    return (
        int(left_eye["x_min"] * w),
        int(left_eye["y_min"] * h),
        int(left_eye["x_max"] * w),
        int(left_eye["y_max"] * h)
    )

def get_blendshape_features(landmarks):
    """Extract 52 facial expression features."""
    return np.array(list(landmarks["blendshapes"].values()))
```

## Sample Manifest

Each sample has a `manifest.json` marking it as complete:

```json
{
  "sample_id": "edge_cases_0004",
  "strategy": "edge_cases",
  "uploaded_at": "2025-12-23T15:05:36",
  "frame_count": 16,
  "anchor_frames": 8,
  "consecutive_frames": 8,
  "has_landmarks": true,
  "real_faces_detected": 16,
  "fake_faces_detected": 16,
  "pipeline_version": "1.0"
}
```

## Strategies (Data Subsets)

Samples are grouped by **strategy** (difficulty/characteristics):

| Strategy | Description |
|----------|-------------|
| `edge_cases` | Challenging cases (occlusions, lighting, expressions) |
| `minimal_processing` | Clean, minimal post-processing |
| `quality_enhancement` | Enhanced quality deepfakes |
| `random_mixed` | Random sampling across difficulties |

Filter by strategy using the manifest:
```python
def get_samples_by_strategy(bucket, strategy="edge_cases"):
    samples = []
    for blob in bucket.list_blobs(prefix="samples/"):
        if blob.name.endswith("manifest.json"):
            manifest = json.loads(blob.download_as_text())
            if manifest.get("strategy") == strategy:
                samples.append(manifest["sample_id"])
    return samples
```

## Example: PyTorch DataLoader

```python
from google.cloud import storage
from PIL import Image
import io
import torch
from torch.utils.data import Dataset

class DeepfakeGCSDataset(Dataset):
    def __init__(self, bucket_name, sample_ids, sparse=True, cache_dir="/tmp/data"):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.sample_ids = sample_ids
        self.sparse = sparse  # Use only anchor frames
        self.cache_dir = Path(cache_dir)
        self.frame_indices = range(0, 16, 2) if sparse else range(16)
    
    def __len__(self):
        return len(self.sample_ids) * len(self.frame_indices) * 2  # real + fake
    
    def __getitem__(self, idx):
        # Calculate which sample, frame, and label
        frames_per_sample = len(self.frame_indices) * 2
        sample_idx = idx // frames_per_sample
        remainder = idx % frames_per_sample
        frame_idx = self.frame_indices[remainder // 2]
        is_fake = remainder % 2
        
        sample_id = self.sample_ids[sample_idx]
        source = "fake" if is_fake else "real"
        
        # Load frame
        blob_path = f"samples/{sample_id}/frames/{source}/frame_{frame_idx:04d}.png"
        blob = self.bucket.blob(blob_path)
        img_bytes = blob.download_as_bytes()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Load landmarks (optional)
        lm_path = f"samples/{sample_id}/landmarks/{source}_landmarks.json"
        lm_blob = self.bucket.blob(lm_path)
        landmarks = json.loads(lm_blob.download_as_text())["frames"][frame_idx]
        
        return {
            "image": torch.tensor(np.array(img)).permute(2, 0, 1) / 255.0,
            "label": is_fake,
            "landmarks": landmarks,
            "sample_id": sample_id,
            "frame_idx": frame_idx
        }
```

## Downloading Data Locally

```bash
# Download all frames for training
gsutil -m cp -r gs://live-deepfake-methods-real-and-fake-frames/samples/ ./data/

# Download specific strategy
gsutil -m cp -r "gs://live-deepfake-methods-real-and-fake-frames/samples/edge_cases_*" ./data/

# Download only landmarks (small, fast)
gsutil -m cp -r "gs://live-deepfake-methods-real-and-fake-frames/samples/*/landmarks/" ./landmarks/
```

## Data Statistics

| Metric | Value |
|--------|-------|
| Frames per sample | 16 (8 anchor + 8 consecutive) |
| Frame pairs per sample | 16 real + 16 fake = 32 images |
| Landmark files per sample | 3 (real, fake, metadata) |
| Typical frame size | ~400-500 KB (PNG) |
| Typical sample size | ~15-20 MB total |

## Incremental Updates

The dataset grows incrementally. To check for new samples:

```python
def get_all_sample_ids(bucket):
    """List all complete samples (those with manifest.json)."""
    sample_ids = set()
    for blob in bucket.list_blobs(prefix="samples/"):
        if blob.name.endswith("/manifest.json"):
            sample_id = blob.name.split("/")[1]
            sample_ids.add(sample_id)
    return sorted(sample_ids)
```

Samples are atomic: if `manifest.json` exists, all frames and landmarks are guaranteed to be present.
