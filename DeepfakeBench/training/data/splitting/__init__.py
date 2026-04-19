"""
Data splitting module.

This module provides:
- Splitter interface and implementations for train/val splitting
- Video and frame-level splitting strategies
- Property-based splitting with identity isolation

Usage:
    from data.splitting import get_splitter, VideoInfo
    
    splitter = get_splitter('property_based', config)
    train_data, val_in_dist, val_holdout, stats = splitter.split()
"""

from .base import Splitter, SplitResult
from .video_info import VideoInfo
from .constants import EFS_METHODS, REG_METHODS, REV_METHODS, EXCLUDE_METHODS
from .splitters import (
    LegacySplitter,
    PropertyBasedSplitter,
    get_splitter,
)
from .utils import (
    extract_target_id,
    compute_frame_weights_vectorized,
)

__all__ = [
    # Base classes
    'Splitter',
    'SplitResult',
    'VideoInfo',
    # Constants
    'EFS_METHODS',
    'REG_METHODS', 
    'REV_METHODS',
    'EXCLUDE_METHODS',
    # Splitter implementations
    'LegacySplitter',
    'PropertyBasedSplitter',
    'get_splitter',
    # Utilities
    'extract_target_id',
    'compute_frame_weights_vectorized',
]
