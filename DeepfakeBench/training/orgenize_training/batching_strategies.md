# Batching Strategies Analysis

> **Purpose:** Detailed analysis of each batching strategy in `dataloaders.py`  
> **Goal:** Understand what each strategy does and how to extract them into separate modules

---

## Overview

The codebase supports **4 batching strategies**, selected via `config['dataloader_params']['strategy']`:

1. `frame_level` - Independent frame sampling
2. `video_level` - Sample videos, then frames from each video
3. `per_method` - Balanced sampling across generation methods
4. `property_balancing` - Sophisticated sampling based on frame properties

Each strategy fundamentally differs in:
- What the "unit" of sampling is (frame vs video)
- How balance across labels/methods is achieved
- What metadata is available to the augmentation pipeline

---

## Strategy 1: `frame_level`

### Concept
Treats each frame as an independent sample. Simple, but loses video-level context.

### Current Implementation (dataloaders.py ~L1000-1100)

```python
# Simplified pseudocode
def create_frame_level_loader(train_data, config):
    # train_data: List[VideoInfo]
    
    # 1. Flatten all frames into (path, label) tuples
    all_frames = []
    for video in train_data:
        label_id = 0 if video.label == 'real' else 1
        for path in video.frame_paths:
            all_frames.append((path, label_id))
    
    # 2. Create DataPipe
    datapipe = IterableWrapper(all_frames)
    datapipe = datapipe.shuffle(buffer_size=10000)
    datapipe = datapipe.batch(batch_size)
    datapipe = datapipe.flatmap(load_and_process_frame_batch)
    
    # 3. Create DataLoader
    return DataLoader(datapipe, batch_size=None, num_workers=...)
```

### Pros
- Simple to implement and debug
- Maximizes frame diversity per batch
- Works well with standard augmentations

### Cons
- Loses temporal context (frames from same video are independent)
- No control over how many frames per video appear in a batch
- No access to video-level metadata for augmentation

### Extraction Plan
```python
# data/batching/frame_level.py
class FrameLevelStrategy(BatchingStrategy):
    def __init__(self, config: BatchingConfig):
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        
    def create_train_loader(self, train_data: List[VideoInfo], config: dict) -> DataLoader:
        frames = self._flatten_frames(train_data)
        datapipe = self._build_datapipe(frames, config)
        return DataLoader(datapipe, batch_size=None, ...)
    
    def _flatten_frames(self, videos: List[VideoInfo]) -> List[Tuple[str, int]]:
        # Extract (path, label_id) for each frame
        ...
```

---

## Strategy 2: `video_level`

### Concept
Sample complete videos, then extract N frames from each. Preserves some temporal context.

### Current Implementation (dataloaders.py ~L1100-1250)

```python
# Simplified pseudocode
def create_video_level_loader(train_data, config):
    # train_data: List[VideoInfo]
    frames_per_video = config['dataloader_params']['frames_per_video']  # e.g., 8
    videos_per_batch = config['dataloader_params']['videos_per_batch']  # e.g., 4
    
    # 1. Create DataPipe from videos
    datapipe = IterableWrapper(train_data)
    datapipe = datapipe.shuffle(buffer_size=1000)
    datapipe = datapipe.map(lambda v: load_and_process_video(v, config, 'train'))
    datapipe = datapipe.filter(_not_none)  # Skip failed loads
    datapipe = datapipe.batch(videos_per_batch)
    
    # 2. Collate: stack video tensors
    return DataLoader(datapipe, batch_size=None, collate_fn=collate_fn, ...)
```

### Batch Shape
- Input: `videos_per_batch` videos
- Each video: `[frames_per_video, C, H, W]` tensor
- Final batch: `[videos_per_batch, frames_per_video, C, H, W]` or flattened

### Pros
- Maintains video context (frames from same video are grouped)
- Allows video-level augmentation consistency (same aug seed for all frames)
- Natural for video-level evaluation

### Cons
- Requires minimum frames per video (videos with fewer frames are skipped)
- Less frame diversity per batch than frame_level
- Memory-intensive for high `frames_per_video`

### Extraction Plan
```python
# data/batching/video_level.py
class VideoLevelStrategy(BatchingStrategy):
    def __init__(self, config: BatchingConfig):
        self.frames_per_video = config.frames_per_video
        self.videos_per_batch = config.videos_per_batch
        
    def create_train_loader(self, train_data: List[VideoInfo], config: dict) -> DataLoader:
        datapipe = self._build_video_datapipe(train_data, config)
        return DataLoader(datapipe, batch_size=None, collate_fn=self._video_collate)
```

---

## Strategy 3: `per_method`

### Concept
Explicitly balance sampling across generation methods (e.g., simswap, faceswap, StyleGAN).

### Current Implementation (dataloaders.py ~L1250-1450)

```python
# Simplified pseudocode
def create_per_method_loader(train_data, config, real_methods, fake_methods):
    # 1. Group videos by method
    videos_by_method = defaultdict(list)
    for video in train_data:
        videos_by_method[video.method].append(video)
    
    # 2. Create per-method DataPipes
    real_pipes = {m: IterableWrapper(videos_by_method[m]).cycle() 
                  for m in real_methods if m in videos_by_method}
    fake_pipes = {m: IterableWrapper(videos_by_method[m]).cycle() 
                  for m in fake_methods if m in videos_by_method}
    
    # 3. Weighted sampling
    # Each batch: sample from each method proportionally
    def sample_batch():
        batch = []
        for method, weight in method_weights.items():
            n_samples = int(weight * batch_size)
            pipe = real_pipes[method] if method in real_methods else fake_pipes[method]
            for _ in range(n_samples):
                batch.append(next(pipe))
        return batch
```

### Pros
- Explicit control over method representation
- Prevents dominant methods from overwhelming training
- Good for studying per-method generalization

### Cons
- Complex implementation (multiple iterators to manage)
- Requires explicit method listing in config
- Can over-sample rare methods (tiny dataset for that method)

### Current Issues in Code
- Method weighting logic is scattered across `train_sweep.py` and `dataloaders.py`
- Iterator management happens in `trainer.py` (`self.real_method_iters`, `self.fake_method_iters`)
- Coupling between data loading and training loop

### Extraction Plan
```python
# data/batching/per_method.py
class PerMethodStrategy(BatchingStrategy):
    def __init__(self, config: BatchingConfig, method_config: MethodConfig):
        self.real_methods = method_config.real_sources
        self.fake_methods = method_config.train_fakes
        self.method_weights = self._compute_weights(method_config)
        
    def create_train_loader(self, train_data: List[VideoInfo], config: dict) -> DataLoader:
        grouped = self._group_by_method(train_data)
        sampler = MethodBalancedSampler(grouped, self.method_weights)
        # ... return DataLoader with sampler
```

---

## Strategy 4: `property_balancing`

### Concept
Most sophisticated strategy. Uses frame-level properties (sharpness, source, etc.) to create balanced, diverse batches.

### Current Implementation (dataloaders.py ~L1450-1670)

```python
# Simplified pseudocode
def create_property_balanced_loader(train_frames, config):
    # train_frames: List[Dict] with keys: path, label, method, sharpness_bucket, sample_weight, ...
    
    # 1. Separate by label
    real_frames = [f for f in train_frames if f['label'] == 'real']
    fake_frames = [f for f in train_frames if f['label'] == 'fake']
    
    # 2. Bucket by property (e.g., sharpness)
    real_by_bucket = group_by(real_frames, 'sharpness_bucket')  # q1, q2, q3, q4
    fake_by_bucket = group_by(fake_frames, 'sharpness_bucket')
    
    # 3. Create round-robin streams per label
    real_stream = CustomRoundRobinDataPipe(*[IterableWrapper(b).cycle() for b in real_by_bucket.values()])
    fake_stream = CustomRoundRobinDataPipe(*[IterableWrapper(b).cycle() for b in fake_by_bucket.values()])
    
    # 4. Interleave real/fake at 50/50 ratio
    master_stream = CustomSampleMultiplexerDataPipe([real_stream, fake_stream], weights=[0.5, 0.5])
    
    # 5. Find "mates" (other frames from same video)
    lookup = build_clip_to_frames_lookup(train_frames)
    master_stream = MateFinderDataPipe(master_stream, lookup, frames_per_video)
    
    # 6. Batch and load
    master_stream = master_stream.batch(batch_size)
    master_stream = master_stream.flatmap(load_and_process_property_batch)
    
    return DataLoader(master_stream, batch_size=None, ...)
```

### Key Components

#### `CustomRoundRobinDataPipe`
Cycles through multiple streams, taking one item from each in turn:
```
Stream A: [a1, a2, a3, ...]
Stream B: [b1, b2, b3, ...]
Output:   [a1, b1, a2, b2, a3, b3, ...]
```

#### `CustomSampleMultiplexerDataPipe`
Probabilistically samples from streams based on weights:
```
Stream Real: [...] (weight 0.5)
Stream Fake: [...] (weight 0.5)
Each sample: 50% chance real, 50% chance fake
```

#### `MateFinderDataPipe`
Given an anchor frame, finds N-1 more frames from the same video:
```
Input: anchor frame from video X
Lookup: {video_id: [frame1, frame2, ...]}
Output: [anchor, mate1, mate2, ...] (all from video X)
```

### Data Structure
Unlike other strategies, `property_balancing` expects `train_data` to be `List[Dict]`:
```python
{
    'path': 'gs://bucket/real/method/video/frame.png',
    'label': 'real',
    'label_id': 0,
    'method': 'FaceForensics++',
    'method_id': 5,
    'video_id': 123,  # Numeric identity
    'original_video_id': '000_003',  # String folder name
    'clip_id': 'method/video',
    'sharpness': 0.78,
    'sharpness_bucket': 'q3',
    'sample_weight': 0.00042,
    # ... other properties
}
```

### Pros
- Fine-grained control over data distribution
- Property-aware augmentation (surgical augmentation)
- Explicit weight-based sampling
- Maintains video context via MateFinderDataPipe

### Cons
- Requires pre-computed property parquet file
- Most complex implementation (~200 lines)
- Custom DataPipe classes add cognitive load
- `sample_weight` computation is external (in `prepare_splits.py`)

### Extraction Plan
```python
# data/batching/property_balanced.py

class PropertyBalancedStrategy(BatchingStrategy):
    def __init__(self, config: BatchingConfig, property_config: PropertyConfig):
        self.frames_per_video = config.frames_per_video
        self.batch_size = config.batch_size
        self.real_label_ratio = config.real_label_ratio or 0.5
        
    def create_train_loader(self, train_frames: List[Dict], config: dict) -> DataLoader:
        real_stream = self._create_label_stream(train_frames, label='real')
        fake_stream = self._create_label_stream(train_frames, label='fake')
        master_stream = self._interleave_streams(real_stream, fake_stream)
        master_stream = self._add_mate_finding(master_stream, train_frames)
        return self._wrap_in_dataloader(master_stream, config)
    
    def _create_label_stream(self, frames: List[Dict], label: str) -> IterDataPipe:
        filtered = [f for f in frames if f['label'] == label]
        buckets = self._bucket_by_property(filtered, 'sharpness_bucket')
        return RoundRobinDataPipe(*[IterableWrapper(b).cycle() for b in buckets.values()])
```

---

## Strategy Comparison Table

| Aspect | frame_level | video_level | per_method | property_balancing |
|--------|-------------|-------------|------------|-------------------|
| **Sampling unit** | Frame | Video | Video (grouped by method) | Frame (with mates) |
| **Input data type** | `List[VideoInfo]` | `List[VideoInfo]` | `List[VideoInfo]` | `List[Dict]` |
| **Balance control** | None | None | Method weights | Property buckets + weights |
| **Video context** | ❌ | ✅ | ✅ | ✅ (via MateFinderDataPipe) |
| **Augmentation info** | None | None | Method name | All frame properties |
| **Complexity** | Low | Medium | High | Very High |
| **Lines of code** | ~100 | ~150 | ~200 | ~250 |

---

## Refactoring Recommendations

### 1. Common Interface
All strategies should implement:
```python
class BatchingStrategy(ABC):
    @abstractmethod
    def create_train_loader(self, train_data, config) -> DataLoader:
        pass
    
    @abstractmethod
    def create_val_loader(self, val_data, config) -> DataLoader:
        pass
    
    @property
    @abstractmethod
    def expected_data_format(self) -> str:
        """'video_info' or 'frame_dict'"""
        pass
```

### 2. Move Custom DataPipes
Create `data/datapipes.py`:
- `RoundRobinDataPipe` (rename from `CustomRoundRobinDataPipe`)
- `WeightedMultiplexerDataPipe` (rename from `CustomSampleMultiplexerDataPipe`)
- `MateFinderDataPipe`

### 3. Unified Frame Loading
Create `data/loaders/unified_loader.py`:
```python
def load_frame(
    path: str,
    config: dict,
    mode: str,
    properties: dict = None  # For property-aware augmentation
) -> torch.Tensor:
    # Single source of truth for frame loading
    ...
```

### 4. Strategy Registry
```python
# data/batching/__init__.py
BATCHING_REGISTRY = {
    'frame_level': FrameLevelStrategy,
    'video_level': VideoLevelStrategy,
    'per_method': PerMethodStrategy,
    'property_balancing': PropertyBalancedStrategy,
}

def get_batching_strategy(name: str, config: BatchingConfig) -> BatchingStrategy:
    cls = BATCHING_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown batching strategy: {name}")
    return cls(config)
```

---

## Questions for Future Design

1. **Should `per_method` be deprecated?** Property balancing with method-based weights achieves similar goals.

2. **Video context in `frame_level`:** Should we add optional mate-finding to frame_level?

3. **Validation strategy:** Should validation always use video_level (for cleaner metrics)?

4. **Dynamic strategies:** Should we support mixing strategies (e.g., 70% property_balanced, 30% random)?
