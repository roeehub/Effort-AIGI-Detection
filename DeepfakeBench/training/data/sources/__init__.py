"""
Data Source Factory Module

This module provides a unified interface for creating data pipelines from different
sources (manifest files, DeepLive GCS bucket, HuggingFace datasets, etc.).

The factory pattern allows the training script to remain source-agnostic while
supporting multiple data loading strategies through configuration.

Usage:
    from data.sources import create_data_pipeline
    
    # Create pipeline based on config
    train_loader, val_in_dist_loader, val_holdout_loader, data_stats = create_data_pipeline(
        config, data_config, logger
    )
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    import logging

# Registry of data source creators
_DATA_SOURCE_REGISTRY: Dict[str, Callable] = {}


@dataclass
class DataPipelineResult:
    """Result from creating a data pipeline.

    Attributes:
        train_loader: DataLoader for training data
        val_in_dist_loader: DataLoader for in-distribution validation
        val_holdout_loader: DataLoader for holdout validation (currently populated
            from the 5% test split — kept for backward compatibility).
        train_samples: Raw training samples (for epoch length calculation, weighted sampling, etc.)
        data_stats: Dictionary with dataset statistics for logging
        ood_loader: Optional OOD evaluation loader (monitored partition after A10).
        test_loader: Optional final-test DataLoader (5% test slice), exposed separately
            from val_holdout_loader so A2 final_eval can run a frozen pass over it.
        ood_heldout_loader: Optional held-out OOD slice (A10, ~10% by blake2b hash).
            Never used during training; consumed only by A2 final_eval.
    """
    train_loader: DataLoader
    val_in_dist_loader: DataLoader
    val_holdout_loader: Optional[DataLoader]
    train_samples: List[Any]
    data_stats: Dict[str, Any]
    ood_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None
    ood_heldout_loader: Optional[DataLoader] = None


def register_data_source(name: str):
    """Decorator to register a data source creator function.
    
    Args:
        name: Name of the data source (e.g., 'manifest', 'deeplive', 'hf_dataset')
        
    Example:
        @register_data_source('manifest')
        def create_manifest_pipeline(config, data_config, logger):
            ...
    """
    def decorator(func: Callable) -> Callable:
        _DATA_SOURCE_REGISTRY[name] = func
        return func
    return decorator


def get_available_data_sources() -> List[str]:
    """Return list of registered data source names."""
    return list(_DATA_SOURCE_REGISTRY.keys())


def create_data_pipeline(
    config: Dict[str, Any],
    data_config: Dict[str, Any],
    logger: "logging.Logger",
    **kwargs
) -> DataPipelineResult:
    """
    Factory function that creates the appropriate data pipeline based on config.
    
    This is the main entry point for data loading in the unified training script.
    The data source is determined by `data_config['data_source']` or defaults to 'manifest'.
    
    Args:
        config: Main training configuration dictionary
        data_config: Data-specific configuration dictionary
        logger: Logger instance for output
        **kwargs: Additional arguments passed to the specific pipeline creator
        
    Returns:
        DataPipelineResult containing loaders and metadata
        
    Raises:
        ValueError: If the specified data_source is not registered
        
    Example:
        >>> result = create_data_pipeline(config, data_config, logger)
        >>> for batch in result.train_loader:
        ...     trainer.train_step(batch)
    """
    # Determine data source from config
    data_source = data_config.get('data_source', 'manifest')
    
    logger.info(f"=" * 60)
    logger.info(f"Creating data pipeline with source: '{data_source}'")
    logger.info(f"=" * 60)
    
    if data_source not in _DATA_SOURCE_REGISTRY:
        available = ', '.join(get_available_data_sources())
        raise ValueError(
            f"Unknown data_source: '{data_source}'. "
            f"Available sources: {available}. "
            f"Make sure the source module is imported."
        )
    
    creator_fn = _DATA_SOURCE_REGISTRY[data_source]
    return creator_fn(config, data_config, logger, **kwargs)


# Import source modules to register them
# These imports populate the registry via @register_data_source decorators
from . import manifest
from . import deeplive
from . import df40_paired
from . import combined_paired
from . import visomaster

__all__ = [
    'create_data_pipeline',
    'register_data_source',
    'get_available_data_sources',
    'DataPipelineResult',
]
