"""
Base interface and types for batching strategies.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Union

from torch.utils.data import DataLoader


@dataclass
class BatchingStrategyConfig:
    """Configuration for a batching strategy.
    
    This consolidates all parameters needed to configure batching behavior.
    """
    batch_size: int = 32
    num_workers: int = 8
    prefetch_factor: int = 4
    
    # Strategy-specific parameters
    frames_per_batch: Optional[int] = None  # For frame_level
    videos_per_batch: Optional[int] = None  # For video_level
    frames_per_video: int = 2  # For property_balancing
    
    # Real/fake balance
    real_label_ratio: Optional[float] = None  # None means 0.5
    
    # Category weights for property balancing
    real_category_weights: Dict[str, float] = field(default_factory=dict)
    fake_category_weights: Dict[str, float] = field(default_factory=dict)
    
    @classmethod
    def from_data_config(cls, data_config: Dict[str, Any]) -> "BatchingStrategyConfig":
        """Create config from data_config dictionary."""
        dl_params = data_config.get('dataloader_params', {})
        
        return cls(
            batch_size=dl_params.get('batch_size', 32),
            num_workers=dl_params.get('num_workers', 8),
            prefetch_factor=dl_params.get('prefetch_factor', 4),
            frames_per_batch=dl_params.get('frames_per_batch'),
            videos_per_batch=dl_params.get('videos_per_batch'),
            frames_per_video=dl_params.get('frames_per_video', 2),
            real_label_ratio=dl_params.get('real_label_ratio'),
            real_category_weights=dl_params.get('real_category_weights', {}),
            fake_category_weights=dl_params.get('fake_category_weights', {}),
        )


class BatchingStrategy(ABC):
    """Abstract base class for batching strategies.
    
    Defines the interface that all batching strategies must implement.
    Each strategy handles data differently for training dataloaders.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        data_config: Dict[str, Any],
        strategy_config: Optional[BatchingStrategyConfig] = None
    ):
        """Initialize the batching strategy.
        
        Args:
            config: Main training configuration dictionary
            data_config: Data-specific configuration dictionary
            strategy_config: Optional pre-built strategy config
        """
        self.config = config
        self.data_config = data_config
        self.strategy_config = strategy_config or BatchingStrategyConfig.from_data_config(data_config)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name (e.g., 'per_method', 'frame_level')."""
        pass
    
    @abstractmethod
    def create_train_loader(self, train_data: List[Any]) -> Union[DataLoader, "LazyDataLoaderManager"]:
        """Create a training dataloader from the given data.
        
        Args:
            train_data: Training data (format depends on strategy)
                - per_method/video_level/frame_level: List[VideoInfo]
                - property_balancing: List[dict] (frame dictionaries)
        
        Returns:
            A DataLoader or LazyDataLoaderManager for training
        """
        pass
    
    def create_validation_loader(
        self,
        val_videos: List[Any],
        mode: str = 'test'
    ) -> "LazyDataLoaderManager":
        """Create a validation dataloader.
        
        Default implementation uses LazyDataLoaderManager for per-method validation.
        Strategies can override if needed.
        
        Args:
            val_videos: List of VideoInfo objects for validation
            mode: 'train' or 'test' mode
            
        Returns:
            A LazyDataLoaderManager for validation
        """
        from .per_method import LazyDataLoaderManager
        from collections import defaultdict
        
        videos_by_method = defaultdict(list)
        for v in val_videos:
            videos_by_method[v.method].append(v)
        
        return LazyDataLoaderManager(
            videos_by_method=videos_by_method,
            config=self.config,
            data_config=self.data_config,
            batch_size=self.config.get('test_batchSize', 32),
            mode=mode,
            max_loaders=8  # Default max loaders
        )


# Type alias for loader manager (forward reference)
class LazyDataLoaderManager:
    """Forward reference to LazyDataLoaderManager from per_method module."""
    pass
