"""
Video-level batching strategy implementation.

This strategy samples at the video level, loading all frames from selected
videos and creating batches of full videos (not individual frames).
"""
from functools import partial
from typing import Any, Dict, List

from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper, Mapper, Filter

from .base import BatchingStrategy
from .loaders import (
    load_and_process_video,
    collate_fn,
    _not_none,
)


class VideoLevelStrategy(BatchingStrategy):
    """
    Batching strategy that samples at the video level.
    
    This strategy creates a single dataloader that:
    1. Shuffles videos randomly
    2. Loads all frames from each selected video
    3. Creates batches of complete videos
    
    Best for:
    - When temporal consistency within videos is important
    - When you want to see multiple frames from the same video together
    - Models that operate on video-level features
    """
    
    @property
    def name(self) -> str:
        return "video_level"
    
    def create_train_loader(self, train_data: List) -> DataLoader:
        """Create a video-level training dataloader.
        
        Args:
            train_data: List of VideoInfo objects
            
        Returns:
            DataLoader that samples videos randomly
        """
        # Create map function
        map_fn = partial(load_and_process_video, config=self.config, mode='train')
        
        # Build datapipe
        pipe = IterableWrapper(train_data).shuffle()
        pipe = Mapper(pipe, map_fn)
        pipe = Filter(pipe, _not_none)
        
        # Get batch size (videos per batch)
        batch_size = self.strategy_config.videos_per_batch or 8
        
        return DataLoader(
            pipe,
            batch_size=batch_size,
            num_workers=self.strategy_config.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True,
            prefetch_factor=self.strategy_config.prefetch_factor
        )
