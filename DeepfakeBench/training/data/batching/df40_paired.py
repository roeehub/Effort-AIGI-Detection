"""
DF40 Paired Batching Strategy - Paired real/fake frame batching for DF40 dataset.

This strategy creates batches with paired real and fake frames from the DF40 dataset,
with the key difference that NO LANDMARKS are available (unlike DeepLive).

Usage:
    from data.batching.df40_paired import DF40PairedBatchingStrategy
    
    strategy = DF40PairedBatchingStrategy(config, data_config)
    train_loader = strategy.create_train_loader(samples)
"""

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from .base import BatchingStrategy, BatchingStrategyConfig

logger = logging.getLogger(__name__)


@dataclass
class DF40PairedBatchingConfig(BatchingStrategyConfig):
    """Configuration specific to DF40 paired batching."""
    
    # Pairing mode
    pairing_mode: str = 'paired'  # 'paired', 'mixed'
    
    # Frame sampling - DF40 has 32 frames
    frame_sampling: str = 'sparse'  # 'sparse', 'full'
    sparse_indices: List[int] = field(default_factory=lambda: [0, 4, 8, 12, 16, 20, 24, 28])
    
    # Method filtering
    methods: Optional[List[str]] = None  # None = all methods
    
    # Identity-balanced sampling: sample ONE method per identity per epoch
    # This prevents identities with more methods from dominating training
    identity_balanced_sampling: bool = True
    
    @classmethod
    def from_data_config(cls, data_config: Dict[str, Any]) -> "DF40PairedBatchingConfig":
        """Create config from data_config dictionary."""
        dl_params = data_config.get('dataloader_params', {})
        df40_config = data_config.get('df40_paired', {})
        batching_config = data_config.get('batching', {})
        
        return cls(
            batch_size=dl_params.get('batch_size', batching_config.get('batch_size', 32)),
            num_workers=dl_params.get('num_workers', 8),
            prefetch_factor=dl_params.get('prefetch_factor', 4),
            pairing_mode=batching_config.get('pairing_mode', 'paired'),
            frame_sampling=df40_config.get('sampling_mode', 'sparse'),
            sparse_indices=df40_config.get('anchor_indices', [0, 4, 8, 12, 16, 20, 24, 28]),
            methods=df40_config.get('methods'),
            identity_balanced_sampling=df40_config.get('identity_balanced_sampling', True),
        )


class DF40PairedIterableDataset(IterableDataset):
    """
    Iterable dataset that yields paired real/fake frame batches from DF40.
    
    Key features:
    - NO LANDMARKS - transform receives (image, None)
    - IDENTITY-BALANCED SAMPLING: Each identity seen exactly once per epoch,
      with a randomly selected method. This prevents identities with more
      fake methods from dominating training.
    """
    
    def __init__(
        self,
        samples: List[Any],  # List of DF40PairedSample
        dataset: Any,  # DF40PairedDataset for loading
        config: DF40PairedBatchingConfig,
        transform: Optional[Callable] = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Initialize the iterable dataset.
        
        Args:
            samples: List of DF40PairedSample objects
            dataset: DF40PairedDataset instance for loading frames
            config: Batching configuration
            transform: Optional transform/augmentation function
                       NOTE: Will receive (image, None) - no landmarks!
            shuffle: Whether to shuffle samples
            seed: Random seed
        """
        self.samples = samples
        self.dataset = dataset
        self.config = config
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0  # Track epoch for varying random selections
        
        # Get frame indices
        self.frame_indices = config.sparse_indices if config.frame_sampling == 'sparse' else list(range(32))
        
        # Group samples by identity for identity-balanced sampling
        self._samples_by_identity: Dict[str, List[Any]] = defaultdict(list)
        for sample in samples:
            self._samples_by_identity[sample.target_identity].append(sample)
        
        self._identities = list(self._samples_by_identity.keys())
        
        # Log statistics
        methods_per_identity = [len(s) for s in self._samples_by_identity.values()]
        logger.info(f"DF40PairedIterableDataset initialized:")
        logger.info(f"  - Total samples (pairs): {len(samples)}")
        logger.info(f"  - Unique identities: {len(self._identities)}")
        logger.info(f"  - Methods per identity: min={min(methods_per_identity)}, max={max(methods_per_identity)}, avg={sum(methods_per_identity)/len(methods_per_identity):.1f}")
        logger.info(f"  - Frames per sample: {len(self.frame_indices)} x 2 (real+fake)")
        logger.info(f"  - Identity-balanced sampling: {config.identity_balanced_sampling}")
        logger.info(f"  - ⚠️  NO LANDMARKS - transform will receive None for landmarks")
        
        if config.identity_balanced_sampling:
            logger.info(f"  - Samples per epoch: {len(self._identities)} (one method per identity)")
        else:
            logger.info(f"  - Samples per epoch: {len(samples)} (all pairs)")
    
    def set_epoch(self, epoch: int):
        """Set the current epoch for varying random method selection."""
        self._epoch = epoch
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples, yielding (image, label, metadata) dicts."""
        
        # Get worker info for distributed loading
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            # Use epoch + worker_id for seed to vary across epochs and workers
            rng = random.Random(self.seed + self._epoch * 1000 + worker_id)
        else:
            worker_id = 0
            num_workers = 1
            rng = random.Random(self.seed + self._epoch * 1000)
        
        # Choose sampling strategy
        if self.config.identity_balanced_sampling:
            # IDENTITY-BALANCED: One random method per identity per epoch
            samples_to_iterate = self._get_identity_balanced_samples(rng, worker_id, num_workers)
        else:
            # LEGACY: All samples (some identities seen more than others)
            samples_to_iterate = self.samples[worker_id::num_workers]
            if self.shuffle:
                samples_to_iterate = samples_to_iterate.copy()
                rng.shuffle(samples_to_iterate)
        
        # Log method distribution for this epoch (only from worker 0)
        if worker_id == 0:
            method_counts = defaultdict(int)
            for s in samples_to_iterate:
                method_counts[s.method] += 1
            logger.info(f"Epoch {self._epoch} method distribution (worker 0): {dict(method_counts)}")
        
        # Iterate through selected samples
        yield from self._iterate_samples(samples_to_iterate, rng)
    
    def _get_identity_balanced_samples(
        self, 
        rng: random.Random,
        worker_id: int,
        num_workers: int
    ) -> List[Any]:
        """
        Get one sample per identity with randomly selected method.
        
        This ensures each identity is seen exactly once per epoch,
        regardless of how many fake methods exist for that identity.
        """
        selected_samples = []
        
        for identity in self._identities:
            identity_samples = self._samples_by_identity[identity]
            # Randomly select one method for this identity this epoch
            selected = rng.choice(identity_samples)
            selected_samples.append(selected)
        
        # Shuffle the order of identities
        if self.shuffle:
            rng.shuffle(selected_samples)
        
        # Split across workers
        return selected_samples[worker_id::num_workers]
    
    def _iterate_samples(
        self, 
        samples: List[Any], 
        rng: random.Random
    ) -> Iterator[Dict[str, Any]]:
        """Iterate through samples and yield frames."""
        
        for sample in samples:
            try:
                # Load frames for this sample
                real_frames, fake_frames = self.dataset.load_sample_frames(
                    sample, 
                    frame_indices=self.frame_indices,
                    as_array=True
                )
                
                # DF40 has NO landmarks
                # We pass None to transform - it should handle this gracefully
                
                # Yield frames based on pairing mode
                if self.config.pairing_mode == 'paired':
                    # Yield paired: real_0, fake_0, real_1, fake_1, ...
                    for i, frame_idx in enumerate(self.frame_indices):
                        if i >= len(real_frames) or i >= len(fake_frames):
                            continue
                            
                        # Real frame
                        real_img = real_frames[i]
                        
                        if self.transform:
                            # Pass None for landmarks - transform must handle this!
                            real_img = self.transform(real_img, None)
                        
                        yield {
                            'image': real_img,
                            'label': 0,  # Real
                            'sample_id': sample.pair_id,
                            'identity': sample.target_identity,
                            'frame_idx': frame_idx,
                            'source': 'real',
                            'method': sample.method,
                            'real_source': sample.real_source,
                        }
                        
                        # Fake frame
                        fake_img = fake_frames[i]
                        
                        if self.transform:
                            # Pass None for landmarks - transform must handle this!
                            fake_img = self.transform(fake_img, None)
                        
                        yield {
                            'image': fake_img,
                            'label': 1,  # Fake
                            'sample_id': sample.pair_id,
                            'identity': sample.target_identity,
                            'frame_idx': frame_idx,
                            'source': 'fake',
                            'method': sample.method,
                            'real_source': sample.real_source,
                        }
                
                else:  # 'mixed' mode - yield all real then all fake
                    for i, frame_idx in enumerate(self.frame_indices):
                        if i >= len(real_frames):
                            continue
                            
                        real_img = real_frames[i]
                        
                        if self.transform:
                            real_img = self.transform(real_img, None)
                        
                        yield {
                            'image': real_img,
                            'label': 0,
                            'sample_id': sample.pair_id,
                            'identity': sample.target_identity,
                            'frame_idx': frame_idx,
                            'source': 'real',
                            'method': sample.method,
                            'real_source': sample.real_source,
                        }
                    
                    for i, frame_idx in enumerate(self.frame_indices):
                        if i >= len(fake_frames):
                            continue
                            
                        fake_img = fake_frames[i]
                        
                        if self.transform:
                            fake_img = self.transform(fake_img, None)
                        
                        yield {
                            'image': fake_img,
                            'label': 1,
                            'sample_id': sample.pair_id,
                            'identity': sample.target_identity,
                            'frame_idx': frame_idx,
                            'source': 'fake',
                            'method': sample.method,
                            'real_source': sample.real_source,
                        }
                        
            except Exception as e:
                logger.warning(f"Failed to load sample {sample.pair_id}: {e}")
                continue


def df40_paired_collate_fn(batch: List[Dict[str, Any]], target_size: Tuple[int, int] = (224, 224)) -> Dict[str, Any]:
    """
    Collate function for DF40 paired batches - produces VIDEO-LEVEL batches.
    
    Groups frames by sample_id to create video-style batches with shape [B, T, C, H, W].
    This matches the format expected by trainer.py's test_epoch.
    
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
    
    # CLIP normalization values
    CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    
    # Group frames by (sample_id, source) to separate real and fake
    groups = defaultdict(list)
    for item in batch:
        video_key = f"{item['sample_id']}_{item['source']}"
        groups[video_key].append(item)
    
    # Sort frames within each group by frame_idx
    for key in groups:
        groups[key].sort(key=lambda x: x['frame_idx'])
    
    # Process each group into a video
    video_images = []
    video_labels = []
    video_ids = []
    
    for video_key, frames in groups.items():
        if len(frames) == 0:
            continue
        
        frame_tensors = []
        for item in frames:
            img = item['image']
            
            # Convert to numpy if needed
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            
            # Ensure image has correct shape (H, W, C)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            
            # Face scale-jitter (anti-shortcut, label-symmetric) — runs BEFORE
            # the canonical 224×224 resize so the final face area is randomized.
            from data.augmentations.face_scale_jitter import apply_face_scale_jitter
            img = apply_face_scale_jitter(img)

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
        
        video_labels.append(frames[0]['label'])
        video_ids.append(video_key)
    
    if len(video_images) == 0:
        return {
            'image': torch.zeros(0, 1, 3, target_size[0], target_size[1]),
            'label': torch.zeros(0, dtype=torch.long),
            'video_id': [],
        }
    
    # Find max frames to pad
    max_frames = max(v.shape[0] for v in video_images)
    
    # Pad videos to have same number of frames
    padded_videos = []
    for video in video_images:
        if video.shape[0] < max_frames:
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


class DF40PairedBatchingStrategy(BatchingStrategy):
    """
    Batching strategy for DF40 paired dataset with real/fake frames.
    
    This strategy:
    1. Loads samples from DF40PairedDataset
    2. Creates batches with paired real/fake frames
    3. Does NOT support landmark-based augmentation (no landmarks in DF40)
    4. Handles train/val splitting
    """
    
    @property
    def name(self) -> str:
        return 'df40_paired'
    
    def __init__(
        self,
        config: Dict[str, Any],
        data_config: Dict[str, Any],
        strategy_config: Optional[DF40PairedBatchingConfig] = None,
        dataset: Optional[Any] = None,  # DF40PairedDataset
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the DF40 paired batching strategy.
        
        Args:
            config: Main training configuration
            data_config: Data-specific configuration
            strategy_config: Optional pre-built strategy config
            dataset: Optional pre-created DF40PairedDataset
            transform: Optional transform/augmentation function
                       NOTE: Transform must handle None landmarks gracefully!
        """
        if strategy_config is None:
            strategy_config = DF40PairedBatchingConfig.from_data_config(data_config)
        
        super().__init__(config, data_config, strategy_config)
        
        self.df40_config = strategy_config
        self.dataset = dataset
        self.transform = transform
        
        logger.info(f"DF40PairedBatchingStrategy initialized:")
        logger.info(f"  - Batch size: {strategy_config.batch_size}")
        logger.info(f"  - Pairing mode: {strategy_config.pairing_mode}")
        logger.info(f"  - Frame sampling: {strategy_config.frame_sampling}")
        logger.info(f"  - ⚠️  NO LANDMARKS - landmark augmentation will be skipped")
    
    def set_dataset(self, dataset: Any) -> None:
        """Set the DF40PairedDataset instance."""
        self.dataset = dataset
    
    def set_transform(self, transform: Callable) -> None:
        """Set the augmentation transform."""
        self.transform = transform
    
    def create_train_loader(
        self, 
        train_data: List[Any]  # List of DF40PairedSample
    ) -> DataLoader:
        """
        Create a training dataloader.
        
        Args:
            train_data: List of DF40PairedSample objects
            
        Returns:
            DataLoader for training
            
        Note:
            The returned DataLoader has a `.dataset` attribute that supports
            `set_epoch(epoch)` for varying random method selection per epoch.
        """
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset() first or pass to constructor.")
        
        # Log identity-balanced sampling info
        if self.df40_config.identity_balanced_sampling:
            # Count unique identities
            identities = set(s.target_identity for s in train_data)
            logger.info(f"Creating training dataloader with IDENTITY-BALANCED sampling:")
            logger.info(f"  - Total samples (pairs): {len(train_data)}")
            logger.info(f"  - Unique identities: {len(identities)}")
            logger.info(f"  - Samples per epoch: {len(identities)} (one method per identity)")
        else:
            logger.info(f"Creating training dataloader with {len(train_data)} samples")
        
        iterable_dataset = DF40PairedIterableDataset(
            samples=train_data,
            dataset=self.dataset,
            config=self.df40_config,
            transform=self.transform,
            shuffle=True,
            seed=self.config.get('manualSeed', 42),
        )
        
        loader = DataLoader(
            iterable_dataset,
            batch_size=self.df40_config.batch_size,
            num_workers=self.df40_config.num_workers,
            prefetch_factor=self.df40_config.prefetch_factor if self.df40_config.num_workers > 0 else None,
            collate_fn=df40_paired_collate_fn,
            pin_memory=True,
        )
        
        logger.info(f"Training dataloader created:")
        logger.info(f"  - Batch size: {self.df40_config.batch_size}")
        logger.info(f"  - Num workers: {self.df40_config.num_workers}")
        logger.info(f"  - Identity-balanced: {self.df40_config.identity_balanced_sampling}")
        
        return loader
    
    def create_validation_loader(
        self,
        val_data: List[Any],  # List of DF40PairedSample
        mode: str = 'test'
    ) -> DataLoader:
        """
        Create a validation dataloader.
        
        Args:
            val_data: List of DF40PairedSample objects
            mode: 'train' or 'test' mode
            
        Returns:
            DataLoader for validation
        """
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset() first or pass to constructor.")
        
        logger.info(f"Creating validation dataloader with {len(val_data)} samples")
        
        iterable_dataset = DF40PairedIterableDataset(
            samples=val_data,
            dataset=self.dataset,
            config=self.df40_config,
            transform=None,  # No augmentation for validation
            shuffle=False,
            seed=self.config.get('manualSeed', 42),
        )
        
        test_batch_size = self.config.get('test_batchSize', self.df40_config.batch_size)
        
        loader = DataLoader(
            iterable_dataset,
            batch_size=test_batch_size,
            num_workers=self.df40_config.num_workers,
            prefetch_factor=self.df40_config.prefetch_factor if self.df40_config.num_workers > 0 else None,
            collate_fn=df40_paired_collate_fn,
            pin_memory=True,
        )
        
        logger.info(f"Validation dataloader created:")
        logger.info(f"  - Batch size: {test_batch_size}")
        logger.info(f"  - Samples: {len(val_data)}")
        
        return loader
