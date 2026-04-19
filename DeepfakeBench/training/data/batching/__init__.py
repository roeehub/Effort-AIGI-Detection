"""
Batching strategies for dataloader creation.

This module provides modular implementations of different batching/sampling strategies
for training and validation dataloaders.

Strategies:
- per_method: Separate loaders per generation method, round-robin iteration
- video_level: Random sampling at video level, all frames from selected videos
- frame_level: Random sampling at frame level, maximum shuffle
- property_balancing: Hierarchical sampling based on method categories and properties
"""

from .base import BatchingStrategy, BatchingStrategyConfig
from .datapipes import (
    CustomRoundRobinDataPipe,
    CustomSampleMultiplexerDataPipe,
    MateFinderDataPipe,
)
from .loaders import (
    load_and_process_frame_batch,
    load_and_process_property_batch,
    load_and_process_video,
    collate_fn,
    collate_fn_detailed,
)
from .per_method import PerMethodStrategy, LazyDataLoaderManager
from .video_level import VideoLevelStrategy
from .frame_level import FrameLevelStrategy
from .property_balanced import PropertyBalancedStrategy
from .factory import get_batching_strategy, create_dataloaders

__all__ = [
    # Base
    "BatchingStrategy",
    "BatchingStrategyConfig",
    # DataPipes
    "CustomRoundRobinDataPipe",
    "CustomSampleMultiplexerDataPipe", 
    "MateFinderDataPipe",
    # Loaders
    "load_and_process_frame_batch",
    "load_and_process_property_batch",
    "load_and_process_video",
    "collate_fn",
    "collate_fn_detailed",
    # Strategies
    "PerMethodStrategy",
    "LazyDataLoaderManager",
    "VideoLevelStrategy",
    "FrameLevelStrategy",
    "PropertyBalancedStrategy",
    # Factory
    "get_batching_strategy",
    "create_dataloaders",
]
