"""
Utility functions for data splitting.

Contains helper functions for identity extraction, frame weighting,
and data balancing.
"""

import re
import random
import logging
from collections import defaultdict
from typing import List, Dict, Set, Optional

import numpy as np
import pandas as pd

from .video_info import VideoInfo
from .constants import EFS_METHODS, REG_METHODS, REV_METHODS, RE_3DIGIT

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Identity Extraction
# ---------------------------------------------------------------------------

def extract_target_id(label: str, method: str, vid_folder: str) -> Optional[int]:
    """
    Extract integer target ID from video folder name.
    
    For face-swap/reenact methods, extracts the target identity from
    the folder name pattern (e.g., "001_002" -> 2 for REG methods).
    
    Args:
        label: 'real' or 'fake'
        method: Generation method name
        vid_folder: Video folder name
    
    Returns:
        Integer target ID, or None for synthetic/EFS methods
    """
    if method in EFS_METHODS:
        return None
    if label == "real" and method != "FaceForensics++":
        return None  # Celeb-real / YouTube-real
    
    ids = [int(tok) for tok in RE_3DIGIT.findall(vid_folder)]
    if not ids:
        return None
    if method in REV_METHODS:
        return ids[0]
    return ids[-1]  # default → REG


# ---------------------------------------------------------------------------
# Frame Weighting
# ---------------------------------------------------------------------------

def compute_frame_weights_vectorized(
    df: pd.DataFrame,
    real_category_weights: Dict[str, float],
    fake_category_weights: Dict[str, float],
    method_multipliers: Optional[Dict[str, float]] = None,
    by: str = "video",
    eps: float = 1e-12,
) -> pd.Series:
    """
    Compute normalized per-row weights for balanced sampling.
    
    This vectorized version provides significantly better performance
    than row-by-row computation.
    
    Args:
        df: DataFrame with columns: label, method, method_category, video_id
        real_category_weights: Weights for real categories
        fake_category_weights: Weights for fake categories  
        method_multipliers: Optional per-method weight multipliers
        by: 'video' or 'frame' - unit for counting
        eps: Small value to prevent division by zero
    
    Returns:
        Series of normalized weights (sum=1)
    """
    required = {'label', 'method', 'method_category', 'video_id'}
    if not required.issubset(df.columns):
        raise KeyError(f"compute_frame_weights: df missing {required - set(df.columns)}")

    df_working = df[list(required)].copy()

    # 1. Vectorized category mass calculation
    is_real = df_working['label'] == 'real'
    df_working['cat_mass'] = 0.0
    df_working.loc[is_real, 'cat_mass'] = df_working.loc[is_real, 'method_category'].map(
        real_category_weights).fillna(0.0)
    df_working.loc[~is_real, 'cat_mass'] = df_working.loc[~is_real, 'method_category'].map(
        fake_category_weights).fillna(0.0)

    # 2. Calculate effective counts per method
    mm = defaultdict(lambda: 1.0)
    if method_multipliers:
        mm.update(method_multipliers)

    if by == 'video':
        base = df_working[['label', 'method', 'method_category', 'video_id']].drop_duplicates()
    else:  # 'frame'
        base = df_working

    eff = base.groupby(['label', 'method', 'method_category']).size().reset_index(name='internal_count')
    eff['eff_weight'] = eff['internal_count'] * eff['method'].map(mm)
    eff['cat_eff_sum'] = eff.groupby(['label', 'method_category'])['eff_weight'].transform('sum') + eps
    eff['p_m_given_cat'] = eff['eff_weight'] / eff['cat_eff_sum']

    # 3. Vectorized merge to apply p_m_given_cat
    key_cols = ['label', 'method', 'method_category']
    df_working = df_working.merge(eff[key_cols + ['p_m_given_cat']], on=key_cols, how='left')
    df_working['p_m_given_cat'] = df_working['p_m_given_cat'].fillna(0.0)

    # 4. Final weight calculation
    sample_weight = df_working['cat_mass'] * df_working['p_m_given_cat']
    total = sample_weight.sum() + eps
    normalized_weight = sample_weight / total

    return normalized_weight


# ---------------------------------------------------------------------------
# Balancing Functions
# ---------------------------------------------------------------------------

def balance_video_list(
    videos: List[VideoInfo],
    real_source_names: List[str],
    seed: Optional[int] = None,
) -> List[VideoInfo]:
    """
    Balance a list of videos to have equal real and fake counts.
    
    Undersamples the majority class to match the minority class.
    
    Args:
        videos: List of VideoInfo objects
        real_source_names: Method names that are considered 'real'
        seed: Random seed for reproducibility
    
    Returns:
        Balanced list of videos
    """
    if not videos:
        return []
    
    if seed is not None:
        random.seed(seed)
    
    log.info(f"[balance] Balancing list of {len(videos)} videos.")

    # Categorize videos
    real_videos = [v for v in videos if v.method in real_source_names]
    fake_videos = [v for v in videos if v.method not in real_source_names]

    num_reals = len(real_videos)
    num_fakes = len(fake_videos)
    log.info(f"[balance] Initial counts: {num_reals} real, {num_fakes} fake videos.")

    if num_reals == 0 or num_fakes == 0:
        log.warning("[balance] One class has 0 videos. Cannot balance. Returning original list.")
        random.shuffle(videos)
        return videos

    # Undersample to match smaller class
    target_size = min(num_reals, num_fakes)
    log.info(f"[balance] Target videos per class: {target_size}")

    random.shuffle(real_videos)
    random.shuffle(fake_videos)

    balanced_videos = real_videos[:target_size] + fake_videos[:target_size]
    random.shuffle(balanced_videos)

    log.info(f"[balance] Final: {target_size} real, {target_size} fake. Total: {len(balanced_videos)}")
    return balanced_videos


def balance_df_by_label(
    df: pd.DataFrame,
    real_methods: Set[str],
    seed: int,
) -> pd.DataFrame:
    """
    Balance a DataFrame to have equal real and fake video counts.
    
    Undersamples frames from the majority class by video_id.
    
    Args:
        df: DataFrame with 'method' and 'video_id' columns
        real_methods: Set of method names considered 'real'
        seed: Random seed for reproducibility
    
    Returns:
        Balanced DataFrame
    """
    if df.empty:
        return pd.DataFrame()

    log.info(f"[balance_df] Balancing DataFrame with {len(df):,} frames.")

    # Identify real and fake videos
    is_real = df['method'].isin(real_methods)
    real_df = df[is_real]
    fake_df = df[~is_real]

    real_video_ids = real_df['video_id'].unique()
    fake_video_ids = fake_df['video_id'].unique()

    num_real_videos = len(real_video_ids)
    num_fake_videos = len(fake_video_ids)
    log.info(f"[balance_df] Initial video counts: {num_real_videos} real, {num_fake_videos} fake.")

    if num_real_videos == 0 or num_fake_videos == 0:
        log.warning("[balance_df] One class has 0 videos. Cannot balance.")
        return df

    # Undersample majority class
    target_size = min(num_real_videos, num_fake_videos)
    log.info(f"[balance_df] Target videos per class: {target_size}")

    rng = np.random.default_rng(seed=seed)
    if num_real_videos > target_size:
        sampled_real_ids = rng.choice(real_video_ids, size=target_size, replace=False)
        balanced_real_df = real_df[real_df['video_id'].isin(sampled_real_ids)]
        balanced_fake_df = fake_df
    elif num_fake_videos > target_size:
        sampled_fake_ids = rng.choice(fake_video_ids, size=target_size, replace=False)
        balanced_fake_df = fake_df[fake_df['video_id'].isin(sampled_fake_ids)]
        balanced_real_df = real_df
    else:
        balanced_real_df = real_df
        balanced_fake_df = fake_df

    balanced_df = pd.concat([balanced_real_df, balanced_fake_df])
    log.info(
        f"[balance_df] Final: {len(balanced_real_df['video_id'].unique())} real, "
        f"{len(balanced_fake_df['video_id'].unique())} fake. Total frames: {len(balanced_df):,}"
    )

    return balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)


def get_method_multipliers(config: Dict) -> Dict[str, float]:
    """
    Extract method multipliers from configuration.
    
    Checks both 'methods' and 'dataset_methods' keys for backward compatibility.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Dictionary mapping method names to their weight multipliers
    """
    mm = ((config.get('methods') or {}).get('method_multipliers')) or \
         ((config.get('dataset_methods') or {}).get('method_multipliers'))
    return mm or {}
