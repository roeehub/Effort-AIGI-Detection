"""
Frame-level batching strategy implementation.

This strategy maximizes shuffle by sampling at the individual frame level,
treating each frame independently regardless of its source video.
"""
from functools import partial
from typing import Any, Dict, List

from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper

from .base import BatchingStrategy
from .loaders import load_and_process_frame_batch, collate_fn


class FrameLevelStrategy(BatchingStrategy):
    """
    Batching strategy that samples at the individual frame level.
    
    This strategy creates a single dataloader that:
    1. Flattens all videos into a stream of individual frames
    2. Shuffles frames with a large buffer for maximum randomness
    3. Batches frames for efficient I/O (larger batches to overcome cloud latency)
    4. Creates GPU batches for training
    
    Best for:
    - Maximum data diversity per batch
    - When temporal consistency doesn't matter
    - Large-scale training where you want frames from many different videos
    """
    
    @property
    def name(self) -> str:
        return "frame_level"
    
    def create_train_loader(self, train_data: List) -> DataLoader:
        """Create a frame-level training dataloader.
        
        Args:
            train_data: List of VideoInfo objects
            
        Returns:
            DataLoader that samples individual frames randomly
        """
        gpu_batch_size = self.strategy_config.frames_per_batch or 64
        num_workers = self.strategy_config.num_workers
        
        # 1. Start with an iterable of the training videos
        pipe = IterableWrapper(train_data)
        
        # 2. Lazily flatten into a stream of frame tuples (path, label_id)
        def video_to_frame_tuples(video):
            label_id = 0 if video.label == 'real' else 1
            for frame_path in video.frame_paths:
                yield (frame_path, label_id)
        
        pipe = pipe.flatmap(video_to_frame_tuples)
        
        # 3. Shuffle the stream of paths using a larger buffer for better randomness
        pipe = pipe.shuffle(buffer_size=50000)
        
        # 4. Batch the PATHS before mapping to the loading function
        # A larger io_batch_size is better for cloud storage throughput
        io_batch_size = gpu_batch_size * 4
        pipe = pipe.batch(io_batch_size)
        
        # 5. Map the batched loading function - it yields multiple processed frames
        pipe = pipe.flatmap(
            partial(load_and_process_frame_batch, config=self.config, mode='train')
        )
        
        return DataLoader(
            pipe,
            batch_size=gpu_batch_size,  # Final batch size for the GPU
            num_workers=num_workers,
            collate_fn=collate_fn,
            persistent_workers=True,
            prefetch_factor=self.strategy_config.prefetch_factor
        )
