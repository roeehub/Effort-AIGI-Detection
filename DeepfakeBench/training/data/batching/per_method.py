"""
Per-method batching strategy implementation.

This strategy creates separate dataloaders for each generation method and
iterates through them in round-robin fashion, ensuring balanced exposure
to all methods during training.
"""
from collections import OrderedDict, defaultdict
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import torch
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper, Mapper, Filter

from .base import BatchingStrategy
from .loaders import (
    load_and_process_video,
    collate_fn,
    _not_none,
)


# Maximum number of loaders to keep in memory at once
MAX_LOADERS_IN_MEMORY = 8


class LazyDataLoaderManager:
    """
    A memory-efficient manager for creating and cycling through per-method dataloaders.
    
    This manager lazily creates dataloaders for each generation method on-demand,
    maintaining only a limited number of active loaders in memory (LRU eviction).
    It provides both iteration (for training loops) and method-wise access (for validation).
    
    Features:
    - Lazy loader creation: Loaders are only created when needed
    - LRU eviction: Old loaders are evicted when memory limit is reached
    - Round-robin iteration: Cycles through all methods evenly
    - Method-wise access: Get a specific method's loader for targeted validation
    
    Usage:
        >>> manager = LazyDataLoaderManager(videos_by_method, config, data_config, ...)
        >>> for batch in manager:  # Round-robin across all methods
        ...     train_step(batch)
        >>> 
        >>> # Or iterate per method:
        >>> for method in manager.methods:
        ...     for batch in manager.get_loader(method):
        ...         validate_step(batch)
    """
    
    def __init__(
        self,
        videos_by_method: Dict[str, List],
        config: Dict[str, Any],
        data_config: Dict[str, Any],
        batch_size: int,
        mode: str,
        max_loaders: int = MAX_LOADERS_IN_MEMORY,
        map_fn: Optional[Callable] = None
    ):
        """Initialize the lazy dataloader manager.
        
        Args:
            videos_by_method: Dict mapping method name to list of VideoInfo objects
            config: Main training configuration
            data_config: Data-specific configuration
            batch_size: Batch size for each loader
            mode: 'train' or 'test'
            max_loaders: Maximum number of loaders to keep in memory
            map_fn: Optional custom map function (defaults to load_and_process_video)
        """
        self.videos_by_method = videos_by_method
        self.config = config
        self.data_config = data_config
        self.batch_size = batch_size
        self.mode = mode
        self.max_loaders = max_loaders
        
        # Use custom map_fn if provided, otherwise use default
        self.map_fn = map_fn or partial(load_and_process_video, config=config, mode=mode)
        
        # LRU cache for active loaders
        self._active_loaders: OrderedDict[str, DataLoader] = OrderedDict()
        
        # Extract dataloader params
        dl_params = data_config.get('dataloader_params', {})
        self.num_workers = dl_params.get('num_workers', 8)
        self.prefetch_factor = dl_params.get('prefetch_factor', 4)
    
    @property
    def methods(self) -> List[str]:
        """Return list of all method names."""
        return list(self.videos_by_method.keys())
    
    def _create_loader(self, method: str) -> DataLoader:
        """Creates a DataLoader for a specific method."""
        videos = self.videos_by_method[method]
        is_train = self.mode == 'train'

        # For training, shuffle; for validation, don't
        if is_train:
            pipe = IterableWrapper(videos).shuffle()
        else:
            pipe = IterableWrapper(videos)
        
        pipe = Mapper(pipe, self.map_fn)
        pipe = Filter(pipe, _not_none)

        loader = DataLoader(
            pipe,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=is_train and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor
        )
        return loader
    
    def _get_or_create_loader(self, method: str) -> DataLoader:
        """Gets a loader from cache or creates it, managing memory via LRU eviction."""
        if method in self._active_loaders:
            # Move to end to mark as recently used
            self._active_loaders.move_to_end(method)
            return self._active_loaders[method]
        
        # Evict oldest loader if at capacity
        if len(self._active_loaders) >= self.max_loaders:
            oldest_method = next(iter(self._active_loaders))
            del self._active_loaders[oldest_method]
        
        # Create and cache new loader
        loader = self._create_loader(method)
        self._active_loaders[method] = loader
        return loader
    
    def get_loader(self, method: str) -> DataLoader:
        """Get the DataLoader for a specific method.
        
        Args:
            method: Method name to get loader for
            
        Returns:
            DataLoader for the specified method
        """
        if method not in self.videos_by_method:
            raise ValueError(f"Unknown method: {method}")
        return self._get_or_create_loader(method)
    
    def __iter__(self) -> Iterator:
        """Iterate through all methods in round-robin fashion.
        
        Yields batches from each method's loader, cycling through methods
        to ensure balanced training across all generation methods.
        """
        methods = list(self.videos_by_method.keys())
        if not methods:
            return
        
        # Create iterators for each method
        method_iterators = {}
        for method in methods:
            loader = self._get_or_create_loader(method)
            method_iterators[method] = iter(loader)
        
        # Round-robin through methods
        active_methods = set(methods)
        method_idx = 0
        
        while active_methods:
            method = methods[method_idx % len(methods)]
            method_idx += 1
            
            if method not in active_methods:
                continue
            
            try:
                batch = next(method_iterators[method])
                yield batch
            except StopIteration:
                active_methods.discard(method)
    
    def __len__(self) -> int:
        """Approximate total number of batches across all methods."""
        total = 0
        for method, videos in self.videos_by_method.items():
            # Estimate: total videos / batch_size
            total += (len(videos) + self.batch_size - 1) // self.batch_size
        return total


class PerMethodStrategy(BatchingStrategy):
    """
    Batching strategy that creates separate loaders per generation method.
    
    This strategy ensures balanced exposure to all generation methods during
    training by cycling through method-specific loaders in round-robin fashion.
    
    Best for:
    - Training when you want explicit control over method balancing
    - Scenarios with highly imbalanced method distributions
    - When you need per-method validation metrics
    """
    
    @property
    def name(self) -> str:
        return "per_method"
    
    def create_train_loader(self, train_data: List) -> LazyDataLoaderManager:
        """Create a LazyDataLoaderManager for training.
        
        Args:
            train_data: List of VideoInfo objects
            
        Returns:
            LazyDataLoaderManager that iterates through methods round-robin
        """
        # Group videos by method
        videos_by_method = defaultdict(list)
        for v in train_data:
            videos_by_method[v.method].append(v)
        
        # Calculate effective batch size (half of configured, min 1)
        train_batch_size = self.strategy_config.batch_size // 2
        if train_batch_size == 0:
            train_batch_size = 1
        
        return LazyDataLoaderManager(
            videos_by_method=videos_by_method,
            config=self.config,
            data_config=self.data_config,
            batch_size=train_batch_size,
            mode='train',
            max_loaders=MAX_LOADERS_IN_MEMORY
        )
    
    def create_validation_loader(self, val_videos: List, mode: str = 'test') -> LazyDataLoaderManager:
        """Create a LazyDataLoaderManager for validation.
        
        Args:
            val_videos: List of VideoInfo objects
            mode: 'train' or 'test'
            
        Returns:
            LazyDataLoaderManager for per-method validation
        """
        videos_by_method = defaultdict(list)
        for v in val_videos:
            videos_by_method[v.method].append(v)
        
        return LazyDataLoaderManager(
            videos_by_method=videos_by_method,
            config=self.config,
            data_config=self.data_config,
            batch_size=self.config.get('test_batchSize', 32),
            mode=mode,
            max_loaders=MAX_LOADERS_IN_MEMORY
        )
