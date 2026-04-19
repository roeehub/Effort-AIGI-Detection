"""
Splitter implementations.

Contains concrete implementations of the Splitter interface:
- LegacySplitter: Original method-based splitting
- PropertyBasedSplitter: Property-aware splitting with identity isolation
"""

import json
import random
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Set, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .base import Splitter, SplitResult
from .video_info import VideoInfo
from .constants import EFS_METHODS, EXCLUDE_METHODS
from .utils import (
    extract_target_id,
    balance_video_list,
    balance_df_by_label,
    compute_frame_weights_vectorized,
    get_method_multipliers,
)

log = logging.getLogger(__name__)


# ==============================================================================
# Legacy Splitter
# ==============================================================================

class LegacySplitter(Splitter):
    """
    Legacy splitting strategy based on method separation.
    
    Splits data by assigning different methods to train/val pools,
    then balances each pool separately.
    """
    
    @property
    def name(self) -> str:
        return "legacy"
    
    def split(self) -> SplitResult:
        """Perform legacy method-based split."""
        cfg = self.config
        BUCKET = f"gs://{cfg['gcp']['bucket_name']}"
        SUBSET = cfg['data_params'].get('data_subset_percentage', 1.0)
        
        real_methods, train_fake_methods, val_fake_methods, val_only_real_methods = self._get_method_sets()
        all_val_real_methods = real_methods | val_only_real_methods
        all_allowed_methods = real_methods | train_fake_methods | val_fake_methods | val_only_real_methods
        
        stats = {}
        random.seed(self.seed)
        
        # Load manifest
        manifest_path = Path(__file__).parent.parent.parent.parent / "frame_manifest.json"
        frame_paths = self._load_or_create_manifest(manifest_path, BUCKET)
        stats['total_frames_in_manifest'] = len(frame_paths)
        
        # Group frames into videos
        log.info("[discovery] Grouping frames into video objects...")
        vids_dict = self._group_frames_to_videos(frame_paths, all_allowed_methods)
        
        stats['discovered_videos'] = len(vids_dict)
        stats['discovered_methods'] = len(all_allowed_methods)
        log.info(f"[discovery] Discovered {stats['discovered_videos']:,} videos")
        
        # Create pools
        training_pool, validation_pool = self._create_pools(
            vids_dict, real_methods, train_fake_methods, val_fake_methods, val_only_real_methods
        )
        
        stats['unbalanced_train_count'] = len(training_pool)
        stats['unbalanced_val_count'] = len(validation_pool)
        
        # Apply subset
        if SUBSET < 1.0:
            log.info(f"[subset] Applying subset percentage: {SUBSET:.2f}")
            random.shuffle(training_pool)
            random.shuffle(validation_pool)
            training_pool = training_pool[:int(len(training_pool) * SUBSET)]
            validation_pool = validation_pool[:int(len(validation_pool) * SUBSET)]
        
        # Balance pools
        train_videos = balance_video_list(training_pool, list(real_methods), self.seed)
        val_videos = balance_video_list(validation_pool, list(all_val_real_methods), self.seed)
        
        stats['balanced_train_count'] = len(train_videos)
        stats['balanced_val_count'] = len(val_videos)
        
        random.shuffle(train_videos)
        random.shuffle(val_videos)
        
        # Legacy returns empty in-dist, all val as holdout
        return SplitResult(
            train_data=train_videos,
            val_in_dist=[],
            val_holdout=val_videos,
            stats=stats,
        )
    
    def _load_or_create_manifest(self, manifest_path: Path, bucket: str) -> List[str]:
        """Load manifest from cache or create from GCS."""
        if manifest_path.exists():
            frame_paths = json.loads(manifest_path.read_text())
            log.info(f"[manifest] Loaded {len(frame_paths):,} paths from cache")
        else:
            log.info("[manifest] Cache not found. Listing from GCS...")
            from fsspec.core import url_to_fs
            fs = url_to_fs(bucket)[0]
            frame_paths = [f"gs://{p}" for p in fs.glob(f"{bucket}/**")
                          if Path(p).suffix.lower() in {'.png', '.jpg', '.jpeg'}]
            log.info(f"[manifest] Found {len(frame_paths):,} files")
            manifest_path.write_text(json.dumps(frame_paths))
        return frame_paths
    
    def _group_frames_to_videos(
        self, 
        frame_paths: List[str], 
        allowed_methods: Set[str]
    ) -> Dict:
        """Group frame paths into videos by (label, method, video_id)."""
        vids_dict = defaultdict(list)
        for p in frame_paths:
            try:
                parts = Path(p).parts
                label, method, vid = parts[-4], parts[-3], parts[-2]
                if method in allowed_methods and method not in EXCLUDE_METHODS:
                    vids_dict[(label, method, vid)].append(p)
            except IndexError:
                continue
        return vids_dict
    
    def _create_pools(
        self,
        vids_dict: Dict,
        real_methods: Set[str],
        train_fake_methods: Set[str],
        val_fake_methods: Set[str],
        val_only_real_methods: Set[str],
    ) -> tuple:
        """Create training and validation pools from video dict."""
        training_pool = []
        validation_pool = []
        
        for (label, method, vid), frames in vids_dict.items():
            frames.sort()
            tid = extract_target_id(label, method, vid)
            if tid is None:
                tid = (hash((method, vid)) & 0x7FFFFFFF) + 100_000
            
            video = VideoInfo(label, method, vid, frames, tid)
            
            if method in real_methods:
                training_pool.append(video)
                validation_pool.append(video)
            elif method in val_only_real_methods:
                validation_pool.append(video)
            elif method in train_fake_methods:
                training_pool.append(video)
            elif method in val_fake_methods:
                validation_pool.append(video)
        
        return training_pool, validation_pool


# ==============================================================================
# Property-Based Splitter
# ==============================================================================

class PropertyBasedSplitter(Splitter):
    """
    Property-based splitting with identity isolation.
    
    Uses frame properties from a Parquet file and ensures no identity
    leakage between train and validation sets.
    """
    
    @property
    def name(self) -> str:
        return "property_based"
    
    def split(self) -> SplitResult:
        """Perform property-based split with identity isolation."""
        cfg = self.config
        VAL_SPLIT_RATIO = cfg['data_params'].get('val_split_ratio', 0.1)
        PROPERTIES_FILE = cfg['property_balancing']['frame_properties_parquet_path']
        
        log.info(f"--- Property-Based Split (Seed: {self.seed}) ---")
        log.info(f"Loading properties from: {PROPERTIES_FILE}")
        
        real_methods, train_fake_methods, val_fake_methods, _ = self._get_method_sets()
        all_allowed_methods = real_methods | train_fake_methods | val_fake_methods
        
        stats = {}
        
        # Load and filter data
        df = pd.read_parquet(PROPERTIES_FILE)
        
        # Create method mapping
        all_method_names = sorted(list(df['method'].unique()))
        method_mapping = {name: i for i, name in enumerate(all_method_names)}
        stats['method_mapping'] = method_mapping
        df['method_id'] = df['method'].map(method_mapping)
        
        stats['discovered_videos'] = df.groupby(['method', 'original_video_id']).ngroups
        stats['discovered_methods'] = df['method'].nunique()
        stats['total_frames_in_parquet'] = len(df)
        
        df_filtered = df[df['method'].isin(all_allowed_methods) & ~df['method'].isin(EXCLUDE_METHODS)].copy()
        stats['total_frames_after_filtering'] = len(df_filtered)
        
        # Create sharpness buckets
        log.info("Creating sharpness buckets (quartiles)...")
        df_filtered['sharpness_bucket'] = pd.qcut(
            df_filtered['sharpness'],
            q=4,
            labels=['q1', 'q2', 'q3', 'q4'],
            duplicates='drop'
        )
        
        # Guarantee-aware identity split
        train_df, val_df, train_ids, val_ids = self._identity_split(
            df_filtered, real_methods, train_fake_methods, VAL_SPLIT_RATIO
        )
        
        stats['unbalanced_train_count'] = len(train_df)
        stats['unbalanced_val_count'] = val_df['video_id'].nunique()
        
        # Balance training set
        log.info("--- Balancing Training Set ---")
        train_df_balanced = balance_df_by_label(train_df, real_methods, self.seed)
        
        # Compute frame weights
        method_multipliers = get_method_multipliers(cfg)
        train_df_balanced['sample_weight'] = compute_frame_weights_vectorized(
            df=train_df_balanced,
            real_category_weights=cfg['dataloader_params']['real_category_weights'],
            fake_category_weights=cfg['dataloader_params']['fake_category_weights'],
            method_multipliers=method_multipliers,
            by='video',
        )
        
        final_train_data = train_df_balanced.to_dict('records')
        stats['train_frame_count'] = len(final_train_data)
        stats['train_video_count'] = len(train_df_balanced['video_id'].unique())
        
        # Create validation videos
        val_in_dist, val_holdout = self._create_validation_pools(
            val_df, real_methods, train_fake_methods, val_fake_methods
        )
        
        stats['val_in_dist_video_count'] = len(val_in_dist)
        stats['val_holdout_video_count'] = len(val_holdout)
        
        log.info(
            f"Final splits: Train {stats['train_video_count']} videos, "
            f"Val In-Dist {len(val_in_dist)}, Val Holdout {len(val_holdout)}"
        )
        
        random.shuffle(final_train_data)
        random.shuffle(val_in_dist)
        random.shuffle(val_holdout)
        
        return SplitResult(
            train_data=final_train_data,
            val_in_dist=val_in_dist,
            val_holdout=val_holdout,
            stats=stats,
        )
    
    def _identity_split(
        self,
        df: pd.DataFrame,
        real_methods: Set[str],
        train_fake_methods: Set[str],
        val_ratio: float,
    ) -> tuple:
        """Perform identity-based split with method guarantees."""
        all_unique_ids = df['video_id'].unique()
        required_train_methods = real_methods | train_fake_methods
        
        # Map methods to video IDs
        method_to_ids = df.groupby('method')['video_id'].unique().apply(set).to_dict()
        
        # Reserve one video_id per required method
        reserved_ids = set()
        random.seed(self.seed)
        
        for method in sorted(list(required_train_methods)):
            if method not in method_to_ids:
                raise ValueError(f"Method '{method}' not found in data")
            
            candidates = list(method_to_ids[method] - reserved_ids)
            if not candidates:
                log.info(f"Method '{method}' already covered by reservations")
                continue
            
            chosen = random.choice(candidates)
            reserved_ids.add(chosen)
            log.info(f"Reserved '{chosen}' for method '{method}'")
        
        # Split remaining IDs
        remaining_ids = np.array([vid for vid in all_unique_ids if vid not in reserved_ids])
        target_val_size = int(len(all_unique_ids) * val_ratio)
        
        if len(remaining_ids) < target_val_size:
            log.warning(f"Cannot achieve target val size {target_val_size}")
            val_ids = set(remaining_ids)
            train_from_remaining = set()
        else:
            train_rem, val_rem = train_test_split(
                remaining_ids, test_size=target_val_size, random_state=self.seed
            )
            train_from_remaining = set(train_rem)
            val_ids = set(val_rem)
        
        train_ids = reserved_ids | train_from_remaining
        
        # Sanity check
        if train_ids & val_ids:
            raise RuntimeError("Identity leakage detected!")
        
        train_df = df[df['video_id'].isin(train_ids)]
        val_df = df[df['video_id'].isin(val_ids)]
        
        return train_df, val_df, train_ids, val_ids
    
    def _create_validation_pools(
        self,
        val_df: pd.DataFrame,
        real_methods: Set[str],
        train_fake_methods: Set[str],
        val_fake_methods: Set[str],
    ) -> tuple:
        """Create in-dist and holdout validation pools."""
        val_pool = []
        for (label, method, orig_vid), group in val_df.groupby(['label', 'method', 'original_video_id']):
            frame_paths = group['path'].tolist()
            identity = int(group['video_id'].iloc[0])
            video = VideoInfo(label, method, orig_vid, frame_paths, identity)
            val_pool.append(video)
        
        val_in_dist_pool = []
        val_holdout_pool = []
        
        for video in val_pool:
            if video.method in real_methods:
                val_in_dist_pool.append(video)
                val_holdout_pool.append(video)
            elif video.method in train_fake_methods:
                val_in_dist_pool.append(video)
            elif video.method in val_fake_methods:
                val_holdout_pool.append(video)
        
        log.info("--- Balancing In-Dist Validation ---")
        val_in_dist = balance_video_list(val_in_dist_pool, list(real_methods), self.seed)
        
        log.info("--- Balancing Holdout Validation ---")
        val_holdout = balance_video_list(val_holdout_pool, list(real_methods), self.seed)
        
        return val_in_dist, val_holdout


# ==============================================================================
# Factory Function
# ==============================================================================

def get_splitter(strategy: str, config: Dict[str, Any]) -> Splitter:
    """
    Get a splitter instance by strategy name.
    
    Args:
        strategy: 'legacy' or 'property_based'
        config: Data configuration dictionary
    
    Returns:
        Splitter instance
    
    Raises:
        ValueError: If strategy is unknown
    """
    splitters = {
        'legacy': LegacySplitter,
        'property_based': PropertyBasedSplitter,
    }
    
    if strategy not in splitters:
        raise ValueError(f"Unknown splitter strategy: '{strategy}'. Available: {list(splitters.keys())}")
    
    return splitters[strategy](config)
