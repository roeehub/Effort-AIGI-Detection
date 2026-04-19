"""
Manifest-based Data Source

This module provides the data pipeline for manifest-based training data.
It wraps the existing prepare_video_splits_v2 and create_dataloaders functions
to provide a consistent interface with other data sources.

This is the traditional data loading approach used by the original train_sweep.py,
supporting GCS manifests, property balancing, and various batching strategies.

Usage:
    from data.sources import create_data_pipeline
    
    # In config:
    data_config['data_source'] = 'manifest'
    
    # Create pipeline
    result = create_data_pipeline(config, data_config, logger)
"""

import logging
from collections import Counter
from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset, IterableDataset
from torch.utils.data.distributed import DistributedSampler

from . import register_data_source, DataPipelineResult

if TYPE_CHECKING:
    pass


@register_data_source('manifest')
def create_manifest_pipeline(
    config: Dict[str, Any],
    data_config: Dict[str, Any],
    logger: logging.Logger,
    **kwargs
) -> DataPipelineResult:
    """
    Create data pipeline from manifest-based data.
    
    This wraps the existing prepare_video_splits_v2 and create_dataloaders
    functions, handling:
    - Video manifest loading from GCS
    - Train/val/holdout splitting
    - Various batching strategies (per_method, video_level, frame_level, property_balancing)
    - Weighted sampling for class imbalance
    - DDP-compatible sampling
    
    Args:
        config: Main training configuration
        data_config: Data-specific configuration including:
            - dataset_methods: Real sources and fake methods configuration
            - dataloader_params: Strategy, batch size, etc.
            - property_balancing: If enabled, configuration for property-based batching
            - gcp: GCS bucket configuration
        logger: Logger for output
        **kwargs: Additional arguments (unused)
        
    Returns:
        DataPipelineResult with train_loader, validation loaders, and metadata
    """
    # Import dependencies here to avoid circular imports
    from prepare_splits import prepare_video_splits_v2, prepare_ood_videos
    from dataset.dataloaders import create_dataloaders, create_ood_loader
    
    logger.info("------- Manifest Data Source: Loading Data -------")
    
    # Pre-flight check: Verify critical config keys
    logger.info("[Pre-flight] Checking required config keys...")
    logger.info(f"  - property_balancing.enabled: {data_config.get('property_balancing', {}).get('enabled', False)}")
    logger.info(f"  - dataset_methods present: {'dataset_methods' in data_config}")
    
    if 'dataset_methods' in data_config:
        dm = data_config['dataset_methods']
        logger.info(f"    - use_real_sources: {len(dm.get('use_real_sources', []))} sources")
        logger.info(f"    - use_fake_methods_for_training: {len(dm.get('use_fake_methods_for_training', []))} methods")
        logger.info(f"    - use_fake_methods_for_validation: {len(dm.get('use_fake_methods_for_validation', []))} methods")
    else:
        logger.error("[Pre-flight] CRITICAL: 'dataset_methods' NOT in data_config!")
        logger.error(f"[Pre-flight] Available keys in data_config: {list(data_config.keys())}")
    
    # Prepare video splits
    train_data, val_in_dist_videos, val_holdout_videos, data_split_stats = prepare_video_splits_v2(data_config)
    
    # Transfer method_mapping to config if Group DRO is enabled
    if config.get('use_group_dro', False):
        if 'method_mapping' in data_split_stats:
            if 'data_params' not in config:
                config['data_params'] = {}
            config['data_params']['method_mapping'] = data_split_stats['method_mapping']
            logger.info("Successfully transferred method_mapping from data prep to main config for Group-DRO.")
        else:
            raise ValueError(
                "Group-DRO is enabled, but 'method_mapping' was not found in data_split_stats. "
                "Ensure prepare_splits.py is adding it."
            )
    
    # Debug logging for lesson configs
    logger.info(f"=== PRE-DATALOADER CONFIG CHECK ===")
    logger.info(f"  config['lesson_data_control'] = {config.get('lesson_data_control', 'NOT SET')}")
    logger.info(f"  config['lesson_gate'] = {config.get('lesson_gate', 'NOT SET')}")
    logger.info(f"  data_config.property_balancing.enabled = {data_config.get('property_balancing', {}).get('enabled', False)}")
    
    # Create dataloaders
    train_loader, val_in_dist_loader, val_holdout_loader = create_dataloaders(
        train_data, val_in_dist_videos, val_holdout_videos, config, data_config
    )
    
    # Optionally rebuild train loader with weighted sampling
    train_loader = _maybe_apply_weighted_sampling(
        train_loader, train_data, config, data_config, logger
    )
    
    # Create OOD loader if configured
    ood_loader = None
    if data_config.get('gcp', {}).get('ood_bucket_name'):
        logger.info("------- OOD Data Loading -------")
        ood_videos = prepare_ood_videos(data_config)
        if ood_videos:
            ood_loader = create_ood_loader(ood_videos, config, data_config)
    else:
        logger.info("No 'ood_bucket_name' in config, skipping OOD loader creation.")
    
    # Combine validation sets for statistics
    all_val_videos = val_in_dist_videos + val_holdout_videos
    
    # Update data_split_stats with final counts
    data_split_stats['val_video_count'] = len(all_val_videos)
    data_split_stats['val_frame_count'] = sum(len(v.frame_paths) for v in all_val_videos)
    
    # Add additional stats for logging
    data_split_stats['val_in_dist_videos'] = val_in_dist_videos
    data_split_stats['val_holdout_videos'] = val_holdout_videos
    data_split_stats['all_val_videos'] = all_val_videos
    
    logger.info(f"Manifest pipeline created:")
    logger.info(f"  - Train samples: {len(train_data)}")
    logger.info(f"  - Val in-dist videos: {len(val_in_dist_videos)}")
    logger.info(f"  - Val holdout videos: {len(val_holdout_videos)}")
    
    return DataPipelineResult(
        train_loader=train_loader,
        val_in_dist_loader=val_in_dist_loader,
        val_holdout_loader=val_holdout_loader,
        train_samples=train_data,
        data_stats=data_split_stats,
        ood_loader=ood_loader,
    )


def _maybe_apply_weighted_sampling(
    train_loader: DataLoader,
    train_data: List[Any],
    config: Dict[str, Any],
    data_config: Dict[str, Any],
    logger: logging.Logger
) -> DataLoader:
    """
    Optionally rebuild train_loader with WeightedRandomSampler.
    
    Property balancing already implements its own sampling, so we skip for that.
    IterableDatasets don't support samplers, so we skip for those too.
    
    Args:
        train_loader: Original train loader
        train_data: Raw training samples
        config: Main config
        data_config: Data config
        logger: Logger
        
    Returns:
        Possibly rebuilt DataLoader with weighted sampling
    """
    # Log weight summary if available
    try:
        import pandas as pd
        if isinstance(train_loader.dataset, pd.DataFrame) and 'sample_weight' in train_loader.dataset.columns:
            logger.info(f"Weight summary: mean={train_loader.dataset['sample_weight'].mean():.6f} "
                        f"min={train_loader.dataset['sample_weight'].min():.6f} "
                        f"max={train_loader.dataset['sample_weight'].max():.6f}")
    except Exception:
        pass

    is_property_balancing = data_config.get('property_balancing', {}).get('enabled', False)
    dataset_obj = getattr(train_loader, 'dataset', None)
    
    # Property balancing uses its own sampling
    if is_property_balancing:
        logger.info(
            "Property balancing enabled -> using its internal sampling. "
            "Skipping WeightedRandomSampler rebuild."
        )
        return train_loader
    
    # IterableDatasets don't support samplers
    if dataset_obj is not None and isinstance(dataset_obj, IterableDataset):
        logger.info(
            "Train dataset is an IterableDataset -> samplers are not supported. "
            "Leaving loader as-is."
        )
        return train_loader
    
    # Check if we have sample weights
    if not train_data or 'sample_weight' not in train_data[0]:
        logger.warning("No 'sample_weight' in train_data; leaving train_loader as-is (unweighted).")
        return train_loader
    
    sample_weights = np.asarray(
        [ex.get('sample_weight', 0.0) for ex in train_data], 
        dtype=np.float64
    )
    
    if sample_weights.sum() == 0:
        logger.warning("All sample_weight are zero; leaving train_loader as-is (unweighted).")
        return train_loader
    
    # Get dataloader params
    frames_per_batch = config['dataloader_params']['frames_per_batch']
    num_workers = config['dataloader_params'].get('num_workers', 4)
    prefetch_factor = config['dataloader_params'].get('prefetch_factor', 2)
    
    if not config.get('ddp', False):
        # Single-GPU weighted sampling
        weights_t = torch.as_tensor(sample_weights, dtype=torch.double)
        epoch_size = len(sample_weights)
        w_sampler = WeightedRandomSampler(
            weights=weights_t, 
            num_samples=epoch_size, 
            replacement=True
        )
        
        train_loader = DataLoader(
            dataset_obj,
            batch_size=frames_per_batch,
            sampler=w_sampler,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
        )
        logger.info("Rebuilt train_loader with WeightedRandomSampler (single-GPU).")
    else:
        # DDP-compatible weighted sampling
        def _make_ddp_weighted_loader():
            p = sample_weights / sample_weights.sum()
            n = len(sample_weights)
            idx = np.random.choice(np.arange(n), size=n, replace=True, p=p).tolist()
            subset = Subset(dataset_obj, idx)
            dist_sampler = DistributedSampler(subset, shuffle=True, drop_last=False)
            return DataLoader(
                subset,
                batch_size=frames_per_batch,
                sampler=dist_sampler,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=True,
            )
        
        train_loader = _make_ddp_weighted_loader()
        config['_ddp_weight_helper'] = {'fn': _make_ddp_weighted_loader}
        logger.info("Rebuilt train_loader with DDP-weighted epoch materialization.")
    
    return train_loader


def compute_manifest_data_stats(
    train_data: List[Any],
    val_in_dist_videos: List[Any],
    val_holdout_videos: List[Any],
    data_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute detailed statistics for manifest-based data.
    
    Args:
        train_data: Training samples
        val_in_dist_videos: In-distribution validation videos
        val_holdout_videos: Holdout validation videos
        data_config: Data configuration
        
    Returns:
        Dictionary of computed statistics
    """
    real_source_names = data_config.get('dataset_methods', {}).get('use_real_sources', [])
    is_property_balancing = data_config.get('property_balancing', {}).get('enabled', False)
    all_val_videos = val_in_dist_videos + val_holdout_videos
    
    # Count by method for train set
    if is_property_balancing:
        # train_data is a list of frame dictionaries
        train_counts = Counter(frame['method'] for frame in train_data)
    else:
        # train_data is a list of VideoInfo objects
        train_counts = Counter(v.method for v in train_data)
    
    # Count by method for validation set
    val_counts = Counter(v.method for v in all_val_videos)
    
    # Calculate real/fake splits
    train_real_count = sum(count for method, count in train_counts.items() if method in real_source_names)
    train_fake_count = sum(count for method, count in train_counts.items() if method not in real_source_names)
    val_real_count = sum(count for method, count in val_counts.items() if method in real_source_names)
    val_fake_count = sum(count for method, count in val_counts.items() if method not in real_source_names)
    
    return {
        'train_counts': dict(train_counts),
        'val_counts': dict(val_counts),
        'train_real_count': train_real_count,
        'train_fake_count': train_fake_count,
        'val_real_count': val_real_count,
        'val_fake_count': val_fake_count,
    }
