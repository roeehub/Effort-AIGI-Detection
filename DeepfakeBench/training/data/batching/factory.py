"""
Factory functions for batching strategies and dataloader creation.

This module provides the main entry points for creating dataloaders using
the modular batching strategy system.
"""
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from torch.utils.data import DataLoader

from .base import BatchingStrategy, BatchingStrategyConfig
from .per_method import PerMethodStrategy, LazyDataLoaderManager
from .video_level import VideoLevelStrategy
from .frame_level import FrameLevelStrategy
from .property_balanced import PropertyBalancedStrategy
from .loaders import (
    load_and_process_video,
    load_and_process_video_detailed,
    collate_fn,
    collate_fn_detailed,
    _not_none,
)


# Registry of available strategies
STRATEGY_REGISTRY: Dict[str, Type[BatchingStrategy]] = {
    'per_method': PerMethodStrategy,
    'video_level': VideoLevelStrategy,
    'frame_level': FrameLevelStrategy,
    'property_balancing': PropertyBalancedStrategy,
    'deeplive': None,  # Lazy import to avoid circular dependencies
}


def get_batching_strategy(
    strategy_name: str,
    config: Dict[str, Any],
    data_config: Dict[str, Any],
    strategy_config: Optional[BatchingStrategyConfig] = None
) -> BatchingStrategy:
    """Factory function to get a batching strategy by name.
    
    Args:
        strategy_name: Name of the strategy ('per_method', 'video_level', etc.)
        config: Main training configuration dictionary
        data_config: Data-specific configuration dictionary
        strategy_config: Optional pre-built strategy config
        
    Returns:
        Instantiated batching strategy
        
    Raises:
        ValueError: If strategy_name is not recognized
        
    Example:
        >>> strategy = get_batching_strategy('property_balancing', config, data_config)
        >>> train_loader = strategy.create_train_loader(train_data)
    """
    if strategy_name not in STRATEGY_REGISTRY:
        available = ', '.join(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown batching strategy: '{strategy_name}'. Available: {available}")
    
    strategy_class = STRATEGY_REGISTRY[strategy_name]
    
    # Handle lazy imports for strategies with special dependencies
    if strategy_class is None:
        if strategy_name == 'deeplive':
            from .deeplive import DeepLiveBatchingStrategy
            strategy_class = DeepLiveBatchingStrategy
            STRATEGY_REGISTRY[strategy_name] = strategy_class  # Cache for future use
        else:
            raise ValueError(f"Strategy '{strategy_name}' registered as None but no lazy loader defined")
    
    return strategy_class(config, data_config, strategy_config)


def register_strategy(name: str, strategy_class: Type[BatchingStrategy]) -> None:
    """Register a new batching strategy.
    
    Args:
        name: Strategy name to register
        strategy_class: Strategy class (must inherit from BatchingStrategy)
        
    Example:
        >>> class MyCustomStrategy(BatchingStrategy): ...
        >>> register_strategy('my_custom', MyCustomStrategy)
    """
    STRATEGY_REGISTRY[name] = strategy_class


def create_validation_loader(
    val_videos: List,
    config: Dict[str, Any],
    data_config: Dict[str, Any],
    max_loaders: int = 8
) -> LazyDataLoaderManager:
    """Helper to create a standard validation dataloader manager.
    
    This is used for both in-distribution and holdout validation sets.
    
    Args:
        val_videos: List of VideoInfo objects
        config: Main training configuration
        data_config: Data-specific configuration
        max_loaders: Maximum loaders to keep in memory
        
    Returns:
        LazyDataLoaderManager for per-method validation
    """
    if not val_videos:
        print("WARNING: Received an empty list of videos. The validation loader will be empty.")
    
    videos_by_method = defaultdict(list)
    for v in val_videos:
        videos_by_method[v.method].append(v)
    
    return LazyDataLoaderManager(
        videos_by_method=videos_by_method,
        config=config,
        data_config=data_config,
        batch_size=config.get('test_batchSize', 32),
        mode='test',
        max_loaders=max_loaders
    )


def create_ood_loader(
    ood_videos: List,
    config: Dict[str, Any],
    data_config: Dict[str, Any],
    max_loaders: int = 8
) -> Optional[LazyDataLoaderManager]:
    """Factory function to create the OOD (out-of-distribution) dataloader.
    
    Args:
        ood_videos: List of VideoInfo objects for OOD evaluation
        config: Main training configuration
        data_config: Data-specific configuration
        max_loaders: Maximum loaders to keep in memory
        
    Returns:
        LazyDataLoaderManager or None if no OOD videos
    """
    print("--- Creating OOD dataloader ---")
    if not ood_videos:
        print("No OOD videos found, returning None for the OOD loader.")
        return None
    
    videos_by_method = defaultdict(list)
    for v in ood_videos:
        videos_by_method[v.method].append(v)
    
    # Use custom map function with frame_count_override
    ood_map_fn = partial(load_and_process_video, config=config, mode='test', frame_count_override=4)
    
    return LazyDataLoaderManager(
        videos_by_method=videos_by_method,
        config=config,
        data_config=data_config,
        batch_size=config.get('test_batchSize', 32),
        mode='test',
        max_loaders=max_loaders,
        map_fn=ood_map_fn
    )


def create_pure_validation_loader(
    videos: List,
    config: Dict[str, Any],
    data_config: Dict[str, Any],
    detailed_reporting: bool = False,
    max_loaders: int = 8
) -> LazyDataLoaderManager:
    """Creates a validation dataloader for a specific, pre-selected set of videos.
    
    This function is designed to work with a prepared list of VideoInfo objects
    to create a dataloader that mimics the behavior of standard validation loaders.
    
    Args:
        videos: List of VideoInfo objects for the validation set
        config: Main training configuration
        data_config: Data-specific configuration
        detailed_reporting: If True, include video_id and frame_paths in output
        max_loaders: Maximum loaders to keep in memory
        
    Returns:
        LazyDataLoaderManager instance ready for validation
    """
    print("--- Creating Pure Validation Loader ---")
    
    if not videos:
        print("Received an empty list of videos. The validation loader will be empty.")
        return LazyDataLoaderManager(
            videos_by_method={},
            config=config,
            data_config=data_config,
            batch_size=config.get('test_batchSize', 32),
            mode='test',
            max_loaders=max_loaders
        )
    
    # Group videos by method, detecting duplicates
    videos_by_method = defaultdict(list)
    video_ids_seen = set()
    duplicates_found = 0
    
    for v in videos:
        video_key = f"{v.method}_{v.video_id}"
        if video_key in video_ids_seen:
            duplicates_found += 1
            print(f"WARNING: Duplicate video found: {video_key}")
        else:
            video_ids_seen.add(video_key)
            videos_by_method[v.method].append(v)
    
    if duplicates_found > 0:
        print(f"WARNING: Found and removed {duplicates_found} duplicate videos during grouping.")
    
    total_videos = sum(len(vids) for vids in videos_by_method.values())
    print(f"Assembled validation data for {len(videos_by_method)} method(s) with {total_videos} total unique videos.")
    
    # Choose appropriate map function and collate function
    if detailed_reporting:
        from torchdata.datapipes.iter import IterableWrapper, Mapper, Filter
        
        map_fn = partial(load_and_process_video_detailed, config=config, mode='test')
        collate_function = collate_fn_detailed
        print("Using detailed reporting mode with metadata preservation.")
        
        # Create specialized manager for detailed mode
        class DetailedLazyDataLoaderManager(LazyDataLoaderManager):
            def _create_loader(self, method):
                """Creates a DataLoader with detailed map function."""
                videos = self.videos_by_method[method]
                
                pipe = IterableWrapper(videos)
                pipe = Mapper(pipe, map_fn)
                pipe = Filter(pipe, _not_none)
                
                loader = DataLoader(
                    pipe,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    collate_fn=collate_function,
                    persistent_workers=False,
                    prefetch_factor=self.prefetch_factor
                )
                return loader
        
        return DetailedLazyDataLoaderManager(
            videos_by_method=videos_by_method,
            config=config,
            data_config=data_config,
            batch_size=config.get('test_batchSize', 32),
            mode='test',
            max_loaders=max_loaders
        )
    else:
        print("Using standard validation mode.")
        return LazyDataLoaderManager(
            videos_by_method=videos_by_method,
            config=config,
            data_config=data_config,
            batch_size=config.get('test_batchSize', 32),
            mode='test',
            max_loaders=max_loaders
        )


def create_dataloaders(
    train_data: List,
    val_in_dist_videos: List,
    val_holdout_videos: List,
    config: Dict[str, Any],
    data_config: Dict[str, Any]
) -> Tuple[Union[DataLoader, LazyDataLoaderManager], LazyDataLoaderManager, LazyDataLoaderManager]:
    """Factory function to create all dataloaders based on the specified strategy.
    
    This function creates three distinct dataloaders:
    1. train_loader: For training, configured by the chosen strategy
    2. val_in_dist_loader: For validating on unseen videos from training methods
    3. val_holdout_loader: For validating on unseen videos from held-out methods
    
    Args:
        train_data: Training data (format depends on strategy):
            - per_method/video_level/frame_level: List[VideoInfo]
            - property_balancing: List[dict] (frame dictionaries)
        val_in_dist_videos: List of VideoInfo for in-distribution validation
        val_holdout_videos: List of VideoInfo for holdout validation
        config: Main training configuration dictionary
        data_config: Data-specific configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_in_dist_loader, val_holdout_loader)
    """
    strategy_name = data_config.get('dataloader_params', {}).get('strategy', 'per_method')
    print(f"--- Creating dataloaders with strategy: '{strategy_name}' ---")
    
    # Get the strategy and create training loader
    strategy = get_batching_strategy(strategy_name, config, data_config)
    train_loader = strategy.create_train_loader(train_data)
    
    # Create validation loaders
    print("\n--- Creating In-Distribution Validation Loader ---")
    val_in_dist_loader = create_validation_loader(val_in_dist_videos, config, data_config)
    
    print("\n--- Creating Holdout Validation Loader ---")
    val_holdout_loader = create_validation_loader(val_holdout_videos, config, data_config)
    
    return train_loader, val_in_dist_loader, val_holdout_loader
