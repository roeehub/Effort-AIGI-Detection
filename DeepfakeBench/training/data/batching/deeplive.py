"""
DeepLive Batching Strategy - Paired real/fake frame batching for DeepLive dataset.

This strategy creates batches with paired real and fake frames from the same sample,
supporting the DeepLive dataset structure with landmark-based augmentation.

Usage:
    from data.batching.deeplive import DeepLiveBatchingStrategy
    
    strategy = DeepLiveBatchingStrategy(config, data_config)
    train_loader = strategy.create_train_loader(samples)
"""

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .base import BatchingStrategy, BatchingStrategyConfig

logger = logging.getLogger(__name__)


@dataclass
class DeepLiveBatchingConfig(BatchingStrategyConfig):
    """Configuration specific to DeepLive batching."""
    
    # Pairing mode
    pairing_mode: str = 'paired'  # 'paired', 'mixed', 'contrastive'
    
    # Frame sampling
    frame_sampling: str = 'sparse'  # 'sparse', 'full', 'pairs'
    sparse_indices: List[int] = field(default_factory=lambda: [0, 2, 4, 6, 8, 10, 12, 14])
    
    # Samples per batch (derived from batch_size)
    # batch_size = samples_per_batch * frames_per_sample * 2 (real + fake)
    samples_per_batch: Optional[int] = None
    
    @classmethod
    def from_data_config(cls, data_config: Dict[str, Any]) -> "DeepLiveBatchingConfig":
        """Create config from data_config dictionary."""
        dl_params = data_config.get('dataloader_params', {})
        deeplive_config = data_config.get('deeplive_data', {})
        batching_config = data_config.get('batching', {})
        
        return cls(
            batch_size=dl_params.get('batch_size', batching_config.get('batch_size', 32)),
            num_workers=dl_params.get('num_workers', 8),
            prefetch_factor=dl_params.get('prefetch_factor', 4),
            pairing_mode=batching_config.get('pairing_mode', 'paired'),
            frame_sampling=deeplive_config.get('frame_sampling', 'sparse'),
            sparse_indices=deeplive_config.get('sparse_indices', [0, 2, 4, 6, 8, 10, 12, 14]),
            samples_per_batch=batching_config.get('samples_per_batch'),
        )


class DeepLiveIterableDataset(IterableDataset):
    """
    Iterable dataset that yields paired real/fake frame batches.
    
    For efficient GCS loading, we load all frames for a sample at once,
    then yield individual (image, label, metadata) tuples.
    """
    
    def __init__(
        self,
        samples: List[Any],  # List of DeepLiveSample
        dataset: Any,  # DeepLiveDataset for loading
        config: DeepLiveBatchingConfig,
        transform: Optional[Callable] = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Initialize the iterable dataset.
        
        Args:
            samples: List of DeepLiveSample objects
            dataset: DeepLiveDataset instance for loading frames/landmarks
            config: Batching configuration
            transform: Optional transform/augmentation function
            shuffle: Whether to shuffle samples
            seed: Random seed
        """
        self.samples = samples
        self.dataset = dataset
        self.config = config
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed
        
        # Get frame indices
        self.frame_indices = config.sparse_indices if config.frame_sampling == 'sparse' else list(range(16))
        
        logger.info(f"DeepLiveIterableDataset initialized:")
        logger.info(f"  - Samples: {len(samples)}")
        logger.info(f"  - Frames per sample: {len(self.frame_indices)} x 2 (real+fake)")
        logger.info(f"  - Pairing mode: {config.pairing_mode}")
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples, yielding (image, label, metadata) dicts."""
        
        # Get worker info for distributed loading
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Split samples across workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            samples = self.samples[worker_id::num_workers]
            rng = random.Random(self.seed + worker_id)
        else:
            samples = self.samples
            rng = random.Random(self.seed)
        
        # Shuffle samples
        if self.shuffle:
            samples = samples.copy()
            rng.shuffle(samples)
        
        # Iterate through samples
        for sample in samples:
            try:
                # Load frames for this sample
                real_frames, fake_frames = self.dataset.load_sample_frames(
                    sample, 
                    frame_indices=self.frame_indices,
                    as_array=True
                )
                
                # Load landmarks if available
                real_landmarks, fake_landmarks = None, None
                if self.dataset.use_landmarks and sample.has_landmarks:
                    real_landmarks, fake_landmarks = self.dataset.load_landmarks(sample)
                
                # Yield frames based on pairing mode
                if self.config.pairing_mode == 'paired':
                    # Yield paired: real_0, fake_0, real_1, fake_1, ...
                    for i, frame_idx in enumerate(self.frame_indices):
                        # Real frame
                        real_img = real_frames[i]
                        real_lm = real_landmarks[frame_idx] if real_landmarks and frame_idx < len(real_landmarks) else None
                        
                        if self.transform:
                            real_img = self.transform(real_img, real_lm)
                        
                        yield {
                            'image': real_img,
                            'label': 0,  # Real
                            'sample_id': sample.sample_id,
                            'frame_idx': frame_idx,
                            'source': 'real',
                            'strategy': sample.strategy,
                        }
                        
                        # Fake frame
                        fake_img = fake_frames[i]
                        fake_lm = fake_landmarks[frame_idx] if fake_landmarks and frame_idx < len(fake_landmarks) else None
                        
                        if self.transform:
                            fake_img = self.transform(fake_img, fake_lm)
                        
                        yield {
                            'image': fake_img,
                            'label': 1,  # Fake
                            'sample_id': sample.sample_id,
                            'frame_idx': frame_idx,
                            'source': 'fake',
                            'strategy': sample.strategy,
                        }
                
                else:  # 'mixed' mode - shuffle real and fake independently
                    # Yield all real frames
                    for i, frame_idx in enumerate(self.frame_indices):
                        real_img = real_frames[i]
                        real_lm = real_landmarks[frame_idx] if real_landmarks and frame_idx < len(real_landmarks) else None
                        
                        if self.transform:
                            real_img = self.transform(real_img, real_lm)
                        
                        yield {
                            'image': real_img,
                            'label': 0,
                            'sample_id': sample.sample_id,
                            'frame_idx': frame_idx,
                            'source': 'real',
                            'strategy': sample.strategy,
                        }
                    
                    # Yield all fake frames
                    for i, frame_idx in enumerate(self.frame_indices):
                        fake_img = fake_frames[i]
                        fake_lm = fake_landmarks[frame_idx] if fake_landmarks and frame_idx < len(fake_landmarks) else None
                        
                        if self.transform:
                            fake_img = self.transform(fake_img, fake_lm)
                        
                        yield {
                            'image': fake_img,
                            'label': 1,
                            'sample_id': sample.sample_id,
                            'frame_idx': frame_idx,
                            'source': 'fake',
                            'strategy': sample.strategy,
                        }
                        
            except Exception as e:
                logger.warning(f"Failed to load sample {sample.sample_id}: {e}")
                continue


def deeplive_collate_fn(batch: List[Dict[str, Any]], target_size: Tuple[int, int] = (224, 224)) -> Dict[str, Any]:
    """
    Collate function for DeepLive batches - produces VIDEO-LEVEL batches.
    
    Groups frames by sample_id to create video-style batches with shape [B, T, C, H, W]
    where B is the number of unique samples and T is the number of frames per sample.
    This matches the format expected by trainer.py's test_epoch.
    
    For DeepLive, each sample produces both real (label=0) and fake (label=1) frames.
    We group these separately to maintain label consistency within each "video".
    
    Args:
        batch: List of sample dicts with 'image', 'label', 'sample_id', etc.
        target_size: (height, width) to resize images to
        
    Returns:
        Dict with:
            'image': [B, T, C, H, W] tensor
            'label': [B] tensor (label per video)
            'video_id': List of sample_ids
    """
    import cv2
    from collections import defaultdict
    
    # CLIP normalization values
    CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    
    # Group frames by (sample_id, source) to separate real and fake
    # This creates separate "videos" for real and fake frames from the same sample
    groups = defaultdict(list)
    for item in batch:
        # Use sample_id + source as the video key (e.g., "sample123_real", "sample123_fake")
        video_key = f"{item['sample_id']}_{item['source']}"
        groups[video_key].append(item)
    
    # Sort frames within each group by frame_idx for consistency
    for key in groups:
        groups[key].sort(key=lambda x: x['frame_idx'])
    
    # Process each group into a video
    video_images = []
    video_labels = []
    video_ids = []
    
    for video_key, frames in groups.items():
        if len(frames) == 0:
            continue
        
        # Process frames for this video
        frame_tensors = []
        for item in frames:
            img = item['image']
            
            # Convert to numpy if needed
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            
            # Ensure image has correct shape (H, W, C)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            
            # Resize to target size if needed
            if img.shape[:2] != target_size:
                img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
            
            # Convert to tensor: HWC -> CHW, normalize to [0, 1]
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            
            # Apply CLIP normalization
            img_tensor = (img_tensor - CLIP_MEAN) / CLIP_STD
            
            frame_tensors.append(img_tensor)
        
        # Stack frames for this video: [T, C, H, W]
        video_tensor = torch.stack(frame_tensors)
        video_images.append(video_tensor)
        
        # All frames in a group have the same label (real=0 or fake=1)
        video_labels.append(frames[0]['label'])
        video_ids.append(video_key)
    
    if len(video_images) == 0:
        # Return empty batch with correct shape
        return {
            'image': torch.zeros(0, 1, 3, target_size[0], target_size[1]),
            'label': torch.zeros(0, dtype=torch.long),
            'video_id': [],
        }
    
    # Find max frames to pad (should be consistent, but just in case)
    max_frames = max(v.shape[0] for v in video_images)
    
    # Pad videos to have same number of frames
    padded_videos = []
    for video in video_images:
        if video.shape[0] < max_frames:
            # Pad with zeros (or repeat last frame)
            padding = torch.zeros(max_frames - video.shape[0], *video.shape[1:])
            video = torch.cat([video, padding], dim=0)
        padded_videos.append(video)
    
    # Stack all videos: [B, T, C, H, W]
    images = torch.stack(padded_videos)
    labels = torch.tensor(video_labels, dtype=torch.long)
    
    return {
        'image': images,
        'label': labels,
        'video_id': video_ids,
    }


class DeepLiveBatchingStrategy(BatchingStrategy):
    """
    Batching strategy for DeepLive dataset with paired real/fake frames.
    
    This strategy:
    1. Loads samples from DeepLiveDataset
    2. Creates batches with paired real/fake frames
    3. Supports landmark-based augmentation
    4. Handles train/val splitting
    """
    
    @property
    def name(self) -> str:
        return 'deeplive'
    
    def __init__(
        self,
        config: Dict[str, Any],
        data_config: Dict[str, Any],
        strategy_config: Optional[DeepLiveBatchingConfig] = None,
        dataset: Optional[Any] = None,  # DeepLiveDataset
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the DeepLive batching strategy.
        
        Args:
            config: Main training configuration
            data_config: Data-specific configuration
            strategy_config: Optional pre-built strategy config
            dataset: Optional pre-created DeepLiveDataset
            transform: Optional transform/augmentation function
        """
        # Use DeepLive-specific config
        if strategy_config is None:
            strategy_config = DeepLiveBatchingConfig.from_data_config(data_config)
        
        super().__init__(config, data_config, strategy_config)
        
        self.deeplive_config = strategy_config
        self.dataset = dataset
        self.transform = transform
        
        logger.info(f"DeepLiveBatchingStrategy initialized:")
        logger.info(f"  - Batch size: {strategy_config.batch_size}")
        logger.info(f"  - Pairing mode: {strategy_config.pairing_mode}")
        logger.info(f"  - Frame sampling: {strategy_config.frame_sampling}")
    
    def set_dataset(self, dataset: Any) -> None:
        """Set the DeepLiveDataset instance."""
        self.dataset = dataset
    
    def set_transform(self, transform: Callable) -> None:
        """Set the augmentation transform."""
        self.transform = transform
    
    def create_train_loader(
        self, 
        train_data: List[Any]  # List of DeepLiveSample
    ) -> DataLoader:
        """
        Create a training dataloader.
        
        Args:
            train_data: List of DeepLiveSample objects
            
        Returns:
            DataLoader for training
        """
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset() first or pass to constructor.")
        
        logger.info(f"Creating training dataloader with {len(train_data)} samples")
        
        # Create iterable dataset
        iterable_dataset = DeepLiveIterableDataset(
            samples=train_data,
            dataset=self.dataset,
            config=self.deeplive_config,
            transform=self.transform,
            shuffle=True,
            seed=self.config.get('manualSeed', 42),
        )
        
        # Create dataloader
        loader = DataLoader(
            iterable_dataset,
            batch_size=self.deeplive_config.batch_size,
            num_workers=self.deeplive_config.num_workers,
            prefetch_factor=self.deeplive_config.prefetch_factor if self.deeplive_config.num_workers > 0 else None,
            collate_fn=deeplive_collate_fn,
            pin_memory=True,
        )
        
        logger.info(f"Training dataloader created:")
        logger.info(f"  - Batch size: {self.deeplive_config.batch_size}")
        logger.info(f"  - Num workers: {self.deeplive_config.num_workers}")
        
        return loader
    
    def create_validation_loader(
        self,
        val_data: List[Any],  # List of DeepLiveSample
        mode: str = 'test'
    ) -> DataLoader:
        """
        Create a validation dataloader.
        
        Args:
            val_data: List of DeepLiveSample objects
            mode: 'train' or 'test' mode
            
        Returns:
            DataLoader for validation
        """
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset() first or pass to constructor.")
        
        logger.info(f"Creating validation dataloader with {len(val_data)} samples")
        
        # Create iterable dataset (no shuffle, no augmentation for validation)
        iterable_dataset = DeepLiveIterableDataset(
            samples=val_data,
            dataset=self.dataset,
            config=self.deeplive_config,
            transform=None,  # No augmentation for validation
            shuffle=False,
            seed=self.config.get('manualSeed', 42),
        )
        
        # Create dataloader
        test_batch_size = self.config.get('test_batchSize', self.deeplive_config.batch_size)
        
        loader = DataLoader(
            iterable_dataset,
            batch_size=test_batch_size,
            num_workers=self.deeplive_config.num_workers,
            prefetch_factor=self.deeplive_config.prefetch_factor if self.deeplive_config.num_workers > 0 else None,
            collate_fn=deeplive_collate_fn,
            pin_memory=True,
        )
        
        logger.info(f"Validation dataloader created:")
        logger.info(f"  - Batch size: {test_batch_size}")
        logger.info(f"  - Samples: {len(val_data)}")
        
        return loader
