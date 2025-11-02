# --- prepare_splits.py ---

# !/usr/bin/env python3
"""prepare_splits.py – train/val split with GroupShuffleSplit + local manifest cache.
"""
from __future__ import annotations

import json
import random
import re
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple, Dict

import yaml  # noqa
from fsspec.core import url_to_fs  # pip install gcsfs  # noqa
from sklearn.model_selection import GroupShuffleSplit  # pip install scikit-learn  # noqa
import pandas as pd  # pip install pandas pyarrow  # noqa
import numpy as np  # noqa
from sklearn.model_selection import train_test_split  # noqa

log = logging.getLogger(__name__)


# --- Method-aware hierarchical weighting (training only) ----------------------
def _get_method_multipliers_from_cfg(cfg: dict) -> dict:
    """
    Prefer cfg['methods']['method_multipliers'], fallback to
    cfg['dataset_methods']['method_multipliers'] if present.
    """
    mm = ((cfg.get('methods') or {}).get('method_multipliers')) or \
         ((cfg.get('dataset_methods') or {}).get('method_multipliers'))
    return mm or {}


def compute_frame_weights_vectorized(
        df: pd.DataFrame,
        real_category_weights: dict,
        fake_category_weights: dict,
        method_multipliers: dict | None = None,
        by: str = "video",  # 'video' recommended for talking-head datasets
        eps: float = 1e-12,
) -> pd.Series:
    """
    Returns normalized per-row weights (sum=1).
    This version is vectorized for significantly better performance and robustness.
    """
    required = {'label', 'method', 'method_category', 'video_id'}
    if not required.issubset(df.columns):
        raise KeyError(f"compute_frame_weights: df missing {required - set(df.columns)}")

    df_working = df[list(required)].copy()

    # 1. Vectorized category mass calculation
    is_real = df_working['label'] == 'real'
    df_working['cat_mass'] = 0.0
    df_working.loc[is_real, 'cat_mass'] = df_working.loc[is_real, 'method_category'].map(real_category_weights).fillna(
        0.0)
    df_working.loc[~is_real, 'cat_mass'] = df_working.loc[~is_real, 'method_category'].map(
        fake_category_weights).fillna(0.0)

    # 2. Calculate effective counts per method (same logic as before)
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

    # 3. Vectorized merge to apply p_m_given_cat (replaces the slow dict lookup)
    # This is the core of the improvement, replacing the slow itertuples loop.
    key_cols = ['label', 'method', 'method_category']
    df_working = df_working.merge(eff[key_cols + ['p_m_given_cat']], on=key_cols, how='left')
    df_working['p_m_given_cat'] = df_working['p_m_given_cat'].fillna(0.0)

    # 4. Final weight calculation
    sample_weight = df_working['cat_mass'] * df_working['p_m_given_cat']
    total = sample_weight.sum() + eps
    normalized_weight = sample_weight / total

    return normalized_weight


# ---------------------------------------------------------------------------
# 1 Method categories (edit REG/REV if needed)
# ---------------------------------------------------------------------------
EFS_METHODS: Set[str] = {
    "DiT", "SiT", "ddim", "RDDM",
    "VQGAN", "StyleGAN2", "StyleGAN3", "StyleGANXL",
}

# source_target (target = 2nd token)
REG_METHODS: Set[str] = {
    "simswap", "fsgan", "faceswap", "fomm", "facedancer", "inswap",
    "one_shot_free", "blendface", "lia", "mobileswap", "mcnet",
    "uniface", "MRAA", "facevid2vid", "wav2lip", "sadtalker", "danet",
    "e4s", "pirender", "tpsm",
}

# target_source (target = 1st token)
REV_METHODS: Set[str] = {}

EXCLUDE_METHODS: Set[str] = {
    "hyperreenact",
}

_RE_3DIGIT = re.compile(r"\b(\d{3})\b")


# ---------------------------------------------------------------------------
@dataclass
class VideoInfo:
    label: str  # 'real' | 'fake'
    method: str  # generation/source method
    video_id: str  # folder name
    frame_paths: List[str]  # gs:// paths
    identity: int  # target identity (numeric)
    label_id: int = 0  # a numeric label ID

    def __post_init__(self):
        # 1) strict label check
        if self.label not in {"real", "fake"}:
            raise ValueError(
                f"[VideoInfo] Invalid label '{self.label}' "
                f"for video '{self.method}/{self.video_id}'. "
                "Allowed: 'real' or 'fake'."
            )
        self.label_id = 0 if self.label == 'real' else 1

        # 2) basic sanity on frame list
        if not self.frame_paths:
            raise ValueError(
                f"[VideoInfo] Empty frame list for video "
                f"'{self.method}/{self.video_id}'."
            )


# ---------------------------------------------------------------------------
# 2 Identity-extraction helper
# ---------------------------------------------------------------------------
def extract_target_id(label: str, method: str, vid_folder: str) -> int | None:
    """Return integer target ID or None (= synthetic)."""
    if method in EFS_METHODS:
        return None
    if label == "real" and method != "FaceForensics++":
        return None  # Celeb-real / YouTube-real
    ids = [int(tok) for tok in _RE_3DIGIT.findall(vid_folder)]
    if not ids:
        return None
    if method in REV_METHODS:
        return ids[0]
    return ids[-1]  # default → REG


# ---------------------------------------------------------------------------
# BALANCING HELPER FUNCTION
# ---------------------------------------------------------------------------
def _balance_video_list(
        videos: List[VideoInfo],
        real_source_names: List[str]
) -> List[VideoInfo]:
    """Balances a given list of videos to have an equal number of real and fake videos."""
    if not videos:
        return []
    print(f"[balance] Balancing list of {len(videos)} videos.")

    # 1. Categorize videos
    real_videos = [v for v in videos if v.method in real_source_names]
    fake_videos = [v for v in videos if v.method not in real_source_names]

    num_reals = len(real_videos)
    num_fakes = len(fake_videos)
    print(f"[balance] Initial counts: {num_reals} real, {num_fakes} fake videos.")

    if num_reals == 0 or num_fakes == 0:
        print("[balance] WARN: One class has 0 videos. Cannot balance. Returning original list.")
        # Returning a shuffled version of the original list
        random.shuffle(videos)
        return videos

    # 2. Determine target size (size of the smaller class)
    target_size = min(num_reals, num_fakes)
    print(f"[balance] Target videos per class: {target_size}")

    # 3. Shuffle and trim the larger list to match the smaller one
    random.shuffle(real_videos)
    random.shuffle(fake_videos)

    balanced_videos = real_videos[:target_size] + fake_videos[:target_size]
    random.shuffle(balanced_videos)

    print(
        f"[balance] Final balanced counts: {target_size} real, {target_size} fake videos. Total: {len(balanced_videos)}")

    return balanced_videos


def _balance_df_by_label(
        df: pd.DataFrame,
        real_methods: Set[str],
        seed: int
) -> pd.DataFrame:
    """
    Balances a DataFrame to have an equal number of real and fake videos.
    This is achieved by undersampling the majority class.
    """
    if df.empty:
        return pd.DataFrame()

    log.info(f"[balance_df] Balancing DataFrame with {len(df):,} frames.")

    # 1. Identify real and fake videos
    is_real = df['method'].isin(real_methods)
    real_df = df[is_real]
    fake_df = df[~is_real]

    real_video_ids = real_df['video_id'].unique()
    fake_video_ids = fake_df['video_id'].unique()

    num_real_videos = len(real_video_ids)
    num_fake_videos = len(fake_video_ids)
    log.info(f"[balance_df] Initial video counts: {num_real_videos} real, {num_fake_videos} fake.")

    if num_real_videos == 0 or num_fake_videos == 0:
        log.warning("[balance_df] WARN: One class has 0 videos. Cannot balance. Returning original DataFrame.")
        return df

    # 2. Determine target size (number of videos in the smaller class)
    target_size = min(num_real_videos, num_fake_videos)
    log.info(f"[balance_df] Target videos per class: {target_size}")

    # 3. Undersample the majority class by video_id
    rng = np.random.default_rng(seed=seed)
    if num_real_videos > target_size:
        sampled_real_ids = rng.choice(real_video_ids, size=target_size, replace=False)
        balanced_real_df = real_df[real_df['video_id'].isin(sampled_real_ids)]
        balanced_fake_df = fake_df
    elif num_fake_videos > target_size:
        sampled_fake_ids = rng.choice(fake_video_ids, size=target_size, replace=False)
        balanced_fake_df = fake_df[fake_df['video_id'].isin(sampled_fake_ids)]
        balanced_real_df = real_df
    else:  # Already balanced
        balanced_real_df = real_df
        balanced_fake_df = fake_df

    # 4. Concatenate and shuffle
    balanced_df = pd.concat([balanced_real_df, balanced_fake_df])
    log.info(
        f"[balance_df] Final balanced video counts: {len(balanced_real_df['video_id'].unique())} real, "
        f"{len(balanced_fake_df['video_id'].unique())} fake. Total frames: {len(balanced_df):,}"
    )

    return balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)


def prepare_video_splits_v2(data_cfg: dict) -> Tuple[List, List, List, Dict]:
    """
    [MODIFIED]
    Prepares video splits. This function now adheres to a new return signature to support
    dual-validation, returning (train_data, val_in_dist, val_holdout, stats).

    - If 'property_balancing' is enabled, it delegates to `prepare_splits_property_based`,
      which implements the full dual-validation logic.
    - Otherwise, it runs the legacy splitting logic and returns its single validation set
      as `val_holdout`, with an empty list for `val_in_dist`, ensuring compatibility.
    """

    if data_cfg.get('property_balancing', {}).get('enabled', False):
        # This function is expected to return the 4-tuple: (train_data, val_in_dist, val_holdout, stats)
        # This call assumes that change has been made in prepare_splits_property_based.py
        return prepare_splits_property_based(data_cfg)

    # --- LEGACY EXECUTION PATH (for non-property-balancing strategies) ---
    # The following logic remains unchanged, except for the final return statement.

    cfg = data_cfg
    BUCKET = f"gs://{cfg['gcp']['bucket_name']}"
    SEED = cfg['data_params']['seed']
    SUBSET = cfg['data_params']['data_subset_percentage']

    # --- 1. Define method sets ---
    real_methods = set(cfg['methods']['use_real_sources'])
    train_fake_methods = set(cfg['methods']['use_fake_methods_for_training'])
    val_fake_methods = set(cfg['methods']['use_fake_methods_for_validation'])

    # --- Read the new config key and create a combined set for validation ---
    val_only_real_methods = set(cfg['methods'].get('use_real_methods_for_validation_only', []))
    all_val_real_methods = real_methods | val_only_real_methods  # This is crucial for the balancer later

    all_allowed_methods = real_methods | train_fake_methods | val_fake_methods | val_only_real_methods

    stats = {}
    random.seed(SEED)
    manifest_path = Path(__file__).with_name("frame_manifest.json")

    # --- 2. Load manifest ---
    if manifest_path.exists():
        frame_paths = json.loads(manifest_path.read_text())
        log.info(f"[manifest] Loaded {len(frame_paths):,} frame paths from cache")
    else:
        log.info("[manifest] Cache not found. Listing frame objects on GCS...")
        fs = url_to_fs(BUCKET)[0]
        frame_paths = [f"gs://{p}" for p in fs.glob(f"{BUCKET}/**")
                       if Path(p).suffix.lower() in {'.png', '.jpg', '.jpeg'}]
        log.info(f"[manifest] Found {len(frame_paths):,} frame files – writing to cache.")
        manifest_path.write_text(json.dumps(frame_paths))
    stats['total_frames_in_manifest'] = len(frame_paths)

    # --- 3. Group frames and calculate initial discovery stats ---
    log.info("[discovery] Grouping frames into video objects...")
    vids_dict: dict[tuple[str, str, str], List[str]] = defaultdict(list)
    for p in frame_paths:
        try:
            parts = Path(p).parts
            label, method, vid = parts[-4], parts[-3], parts[-2]
            if method in all_allowed_methods and method not in EXCLUDE_METHODS:
                vids_dict[(label, method, vid)].append(p)
        except IndexError:
            continue

    stats['discovered_videos'] = len(vids_dict)
    stats['discovered_methods'] = len(all_allowed_methods)
    log.info(
        f"[discovery] Discovered {stats['discovered_videos']:,} videos from {stats['discovered_methods']} allowed methods.")

    # --- 4. Create separate pools and calculate unbalanced stats ---
    log.info("[split] Creating separate pools for training and validation methods...")
    training_pool: List[VideoInfo] = []
    validation_pool: List[VideoInfo] = []
    for (label, method, vid), fr in vids_dict.items():
        fr.sort()
        tid = extract_target_id(label, method, vid)
        if tid is None: tid = (hash((method, vid)) & 0x7FFFFFFF) + 100_000
        video = VideoInfo(label, method, vid, fr, tid, label_id=(0 if label == 'real' else 1))

        # --- MODIFICATION 2: Update the pool creation logic ---
        if method in real_methods:
            training_pool.append(video)
            validation_pool.append(video)
        elif method in val_only_real_methods:
            validation_pool.append(video)
        elif method in train_fake_methods:
            training_pool.append(video)
        elif method in val_fake_methods:
            validation_pool.append(video)

    stats['unbalanced_train_count'] = len(training_pool)
    stats['unbalanced_val_count'] = len(validation_pool)
    log.info(
        f"[split] Created pools ▶ train: {stats['unbalanced_train_count']:,} | val: {stats['unbalanced_val_count']:,} videos")

    # --- 5. Apply subset percentage and calculate subset stats ---
    if SUBSET < 1.0:
        log.info(f"[subset] Applying `data_subset_percentage`={SUBSET:.2f} to each pool.")
        if training_pool:
            random.shuffle(training_pool)
            training_pool = training_pool[:int(len(training_pool) * SUBSET)]
        if validation_pool:
            random.shuffle(validation_pool)
            validation_pool = validation_pool[:int(len(validation_pool) * SUBSET)]
        log.info(f"[subset] Subset pools ▶ train: {len(training_pool):,} | val: {len(validation_pool):,} videos")

    # This is calculated *after* the optional subsetting block to be correct in all cases.
    stats['subset_video_count'] = len(training_pool) + len(validation_pool)

    # --- 6. Balance pools and calculate final balanced stats ---
    log.info("--- Balancing Training Set from its pool ---")
    train_videos = _balance_video_list(videos=training_pool, real_source_names=list(real_methods))

    log.info("--- Balancing Validation Set from its pool ---")
    # --- Use the combined list of reals for balancing the validation set ---
    val_videos = _balance_video_list(videos=validation_pool, real_source_names=list(all_val_real_methods))

    stats['balanced_train_count'] = len(train_videos)
    stats['balanced_val_count'] = len(val_videos)
    log.info(
        f"Final balanced split ▶ train: {stats['balanced_train_count']:,} videos | val: {stats['balanced_val_count']:,} videos")

    random.shuffle(train_videos)
    random.shuffle(val_videos)

    # --- 7. MODIFIED RETURN STATEMENT FOR COMPATIBILITY  ---
    val_in_dist_videos = []
    val_holdout_videos = val_videos

    return train_videos, val_in_dist_videos, val_holdout_videos, stats


def prepare_splits_property_based(data_cfg: dict) -> Tuple[List[Dict], List[VideoInfo], List[VideoInfo], Dict]:
    """
    [REVISED & FIXED v2]
    Prepares data splits using a guarantee-aware, strict identity-based split.
    This version ensures that at least one video for every required training method
    is reserved for the training set before performing the random split, preventing
    data starvation for rare methods.
    """
    cfg = data_cfg
    SEED = cfg['data_params']['seed']
    VAL_SPLIT_RATIO = cfg['data_params'].get('val_split_ratio', 0.1)
    PROPERTIES_FILE = cfg['property_balancing']['frame_properties_parquet_path']

    log.info(f"--- Preparing splits with Property-Based Strategy (Seed: {SEED}) ---")
    log.info(f"Loading frame properties from: {PROPERTIES_FILE}")

    # --- 1. Define method sets from config ---
    real_methods = set(cfg['dataset_methods']['use_real_sources'])
    train_fake_methods = set(cfg['dataset_methods']['use_fake_methods_for_training'])
    val_fake_methods = set(cfg['dataset_methods']['use_fake_methods_for_validation'])
    all_allowed_methods = real_methods | train_fake_methods | val_fake_methods
    stats = {}

    # --- 2. Load and filter data ---
    df = pd.read_parquet(PROPERTIES_FILE)

    all_method_names = sorted(list(df['method'].unique()))
    method_mapping = {name: i for i, name in enumerate(all_method_names)}
    stats['method_mapping'] = method_mapping
    df['method_id'] = df['method'].map(method_mapping)
    stats['discovered_videos'] = df.groupby(['method', 'original_video_id']).ngroups
    stats['discovered_methods'] = df['method'].nunique()
    stats['total_frames_in_parquet'] = len(df)
    log.info(f"Loaded {len(df):,} frame records from Parquet file.")
    log.info(
        f"[discovery] Discovered {stats['discovered_videos']:,} videos from {stats['discovered_methods']} methods in Parquet.")

    df_filtered = df[df['method'].isin(all_allowed_methods) & ~df['method'].isin(EXCLUDE_METHODS)].copy()
    stats['total_frames_after_filtering'] = len(df_filtered)
    log.info(f"Filtered to {len(df_filtered):,} frames from allowed methods.")

    # --- Create sharpness buckets from the continuous sharpness value ---
    log.info("Creating sharpness buckets (quartiles)...")
    df_filtered['sharpness_bucket'] = pd.qcut(
        df_filtered['sharpness'],
        q=4,
        labels=['q1', 'q2', 'q3', 'q4'],
        duplicates='drop'
    )
    log.info(f"Value counts for new 'sharpness_bucket' column:\n{df_filtered['sharpness_bucket'].value_counts()}")

    # --- 3. [NEW] Guarantee-Aware Identity-Based Split ---
    log.info(f"Performing guarantee-aware identity split with validation ratio: ~{VAL_SPLIT_RATIO}")

    # Step 3.1: Identify all unique IDs and required training methods
    all_unique_ids = df_filtered['video_id'].unique()
    required_train_methods = real_methods | train_fake_methods

    # Step 3.2: Create a map of methods to the video_ids that contain them
    method_to_ids_map = df_filtered.groupby('method')['video_id'].unique().apply(set).to_dict()

    # Step 3.3: Reserve one video_id for each required training method
    reserved_for_train_ids = set()
    for method in sorted(list(required_train_methods)):
        if method not in method_to_ids_map:
            raise ValueError(
                f"Configuration error: Method '{method}' is required for training, "
                "but was not found in the filtered Parquet data."
            )

        # Find candidate IDs for this method that are not already reserved
        candidate_ids = method_to_ids_map[method]
        available_candidates = list(candidate_ids - reserved_for_train_ids)

        if not available_candidates:
            # This is not an error; it means all videos for this method were already
            # reserved by another required method they share a video_id with.
            log.info(f"Method '{method}' is already covered by existing reservations. Skipping.")
            continue

        # Reserve one random video ID for this method
        random.seed(SEED)  # Ensure reproducibility
        chosen_id = random.choice(available_candidates)
        reserved_for_train_ids.add(chosen_id)
        log.info(f"Reserved video_id '{chosen_id}' to guarantee method '{method}' in training set.")

    log.info(f"Reserved a total of {len(reserved_for_train_ids)} video_ids for the training set.")

    # Step 3.4: Split the *remaining* IDs randomly
    remaining_ids = np.array([vid for vid in all_unique_ids if vid not in reserved_for_train_ids])

    # Adjust split size to maintain the overall validation ratio
    target_val_size = int(len(all_unique_ids) * VAL_SPLIT_RATIO)
    if len(remaining_ids) < target_val_size:
        log.warning(
            f"Cannot achieve target validation size of {target_val_size} after reservations. "
            f"Using all {len(remaining_ids)} remaining IDs for validation."
        )
        val_from_remaining_ids = remaining_ids
        train_from_remaining_ids = np.array([])
    else:
        train_from_remaining_ids, val_from_remaining_ids = train_test_split(
            remaining_ids,
            test_size=target_val_size,
            random_state=SEED
        )

    # Step 3.5: Combine reserved IDs with the split training IDs
    train_ids = reserved_for_train_ids.union(set(train_from_remaining_ids))
    val_ids = set(val_from_remaining_ids)

    # Final sanity check for leakage
    if train_ids.intersection(val_ids):
        raise RuntimeError("Identity leakage detected after split! This should not happen.")

    train_df = df_filtered[df_filtered['video_id'].isin(train_ids)]
    val_df = df_filtered[df_filtered['video_id'].isin(val_ids)]

    # --- The rest of the function proceeds as before ---

    stats['unbalanced_train_count'] = len(train_df)
    stats['unbalanced_val_count'] = val_df['video_id'].nunique()
    log.info(
        f"Split pools by identity ▶ Train: {stats['unbalanced_train_count']:,} frames ({len(train_ids)} videos) | "
        f"Val: {stats['unbalanced_val_count']:,} videos"
    )

    # --- 4. Balance the training set (DataFrame) and convert to dicts ---
    log.info("--- Balancing Training Set ---")
    train_df_balanced = _balance_df_by_label(train_df, real_methods, SEED)

    # ... (the rest of the function, including weight calculation, val pool assembly, etc., is unchanged) ...
    method_multipliers = _get_method_multipliers_from_cfg(cfg)
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

    log.info("Assembling validation videos and subdividing...")
    val_pool: List[VideoInfo] = []
    for (label, method, original_video_id), group in val_df.groupby(['label', 'method', 'original_video_id']):
        frame_paths = group['path'].tolist()
        identity = int(group['video_id'].iloc[0])
        video = VideoInfo(label, method, original_video_id, frame_paths, identity)
        val_pool.append(video)

    val_in_dist_pool, val_holdout_pool = [], []
    for video in val_pool:
        is_real = video.method in real_methods
        is_train_fake = video.method in train_fake_methods
        is_val_fake = video.method in val_fake_methods

        if is_real:
            val_in_dist_pool.append(video)
            val_holdout_pool.append(video)
        elif is_train_fake:
            val_in_dist_pool.append(video)
        elif is_val_fake:
            val_holdout_pool.append(video)

    log.info("--- Balancing In-Distribution Validation Set ---")
    val_in_dist_videos = _balance_video_list(val_in_dist_pool, list(real_methods))

    log.info("--- Balancing Holdout Validation Set ---")
    val_holdout_videos = _balance_video_list(val_holdout_pool, list(real_methods))

    stats['val_in_dist_video_count'] = len(val_in_dist_videos)
    stats['val_holdout_video_count'] = len(val_holdout_videos)

    log.info(
        f"Final splits ▶ "
        f"Train: {stats['train_video_count']:,} videos ({stats['train_frame_count']:,} frames) | "
        f"Val In-Dist: {len(val_in_dist_videos):,} videos | "
        f"Val Holdout: {len(val_holdout_videos):,} videos"
    )

    random.shuffle(final_train_data)
    random.shuffle(val_in_dist_videos)
    random.shuffle(val_holdout_videos)

    return final_train_data, val_in_dist_videos, val_holdout_videos, stats


def prepare_ood_videos(data_cfg: dict) -> List[VideoInfo]:
    """
    Scans a specified OOD bucket, discovers all videos, and returns them as a list.
    This function uses its own manifest for caching and does NOT perform any
    splitting, subsetting, or balancing.
    """
    ood_bucket_name = data_cfg.get('gcp', {}).get('ood_bucket_name')
    if not ood_bucket_name:
        log.warning("[OOD] `ood_bucket_name` not found in config. Skipping OOD data preparation.")
        return []

    BUCKET = f"gs://{ood_bucket_name}"
    log.info(f"[OOD] Preparing Out-of-Distribution videos from bucket: {BUCKET}")

    # Use a separate manifest file for the OOD set to avoid conflicts
    manifest_path = Path(__file__).with_name("ood_frame_manifest.json")

    # 1. Load or create the manifest for the OOD bucket
    if manifest_path.exists():
        frame_paths = json.loads(manifest_path.read_text())
        log.info(f"[OOD] Loaded {len(frame_paths):,} OOD frame paths from cache")
    else:
        log.info("[OOD] Caching not found for OOD set. Listing frame objects on GCS...")
        fs = url_to_fs(BUCKET)[0]
        frame_paths = [f"gs://{p}" for p in fs.glob(f"{BUCKET}/**")
                       if Path(p).suffix.lower() in {'.png', '.jpg', '.jpeg'}]
        log.info(f"[OOD] Found {len(frame_paths):,} frame files – writing to cache.")
        manifest_path.write_text(json.dumps(frame_paths))

    # 2. Group frames into VideoInfo objects
    vids_dict: dict[tuple[str, str, str], List[str]] = defaultdict(list)
    for p in frame_paths:
        try:
            # Structure is gs://.../<label>/<method>/<video_id>/<frame_name>
            parts = Path(p).parts
            label, method, vid = parts[-4], parts[-3], parts[-2]
            vids_dict[(label, method, vid)].append(p)
        except IndexError:
            continue

    ood_videos: List[VideoInfo] = []
    for (label, method, vid), fr in vids_dict.items():
        fr.sort()
        # Use a deterministic hash for identity for consistency
        tid = (hash((method, vid)) & 0x7FFFFFFF)
        video = VideoInfo(label, method, vid, fr, tid, label_id=(0 if label == 'real' else 1))
        ood_videos.append(video)

    log.info(f"[OOD] Discovered {len(ood_videos):,} videos across {len(vids_dict)} methods in the OOD set.")
    return ood_videos


def prepare_pure_validation(
        data_cfg: Dict,
        methods: List[str]
) -> List[VideoInfo]:
    """
    Creates a simple, pure validation set from a properties file.

    This function loads frame data, filters it to include only the specified
    methods, and converts the data into a list of VideoInfo objects. It does
    not perform any balancing, splitting, or subsetting.

    Args:
        data_cfg: The configuration dictionary, which must contain the path
                  to the frame properties Parquet file.
        methods: A list of method names to include in the validation set.

    Returns:
        A list of VideoInfo objects representing the validation set.
    """
    PROPERTIES_FILE = data_cfg.get('property_balancing', {}).get('frame_properties_parquet_path')
    if not PROPERTIES_FILE:
        raise ValueError("Path to 'frame_properties_parquet_path' not found in data_cfg.")

    log.info(f"--- Preparing Pure Validation Set from {len(methods)} methods ---")
    log.info(f"Loading frame properties from: {PROPERTIES_FILE}")

    # 1. Load and filter the data
    df = pd.read_parquet(PROPERTIES_FILE)
    log.info(f"Loaded {len(df):,} total frame records from Parquet file.")

    df_filtered = df[df['method'].isin(methods)].copy()
    if df_filtered.empty:
        log.warning(f"No frames found for the specified methods: {methods}. Returning empty list.")
        return []

    log.info(f"Filtered to {len(df_filtered):,} frames matching the provided methods.")

    # 2. Group frames into VideoInfo objects
    validation_videos: List[VideoInfo] = []
    # Group by the unique video identifier components
    grouped = df_filtered.groupby(['label', 'method', 'original_video_id'])
    
    log.info(f"Found {len(grouped)} unique video groups.")

    for (label, method, original_video_id), group in grouped:
        # Sort frames to ensure consistent order
        frame_paths = sorted(group['path'].tolist())

        # Identity is the numeric 'video_id' which is consistent per group
        identity = int(group['video_id'].iloc[0])

        video = VideoInfo(
            label=label,
            method=method,
            video_id=original_video_id,
            frame_paths=frame_paths,
            identity=identity
        )
        validation_videos.append(video)

    log.info(f"Assembled {len(validation_videos)} videos for the pure validation set.")

    return validation_videos


if __name__ == "__main__":
    # This is an example configuration.
    # You can modify this or load it from a YAML file.
    config = {
        'gcp': {
            'bucket_name': 'df40-frames',
            'ood_bucket_name': 'deep-fake-test-10-08-25-frames-yolo'
        },
        'data_params': {
            'seed': 42,
            'data_subset_percentage': 1.0,  # Use 100% of the data
        },
        'methods': {
            # Define which sources are considered 'real'
            'use_real_sources': [
                "FaceForensics++",
                "Celeb-real",
                "youtube-real"
            ],
            # Use all discovered fake methods for both training and validation pools
            'use_fake_methods_for_training': list(EFS_METHODS | REG_METHODS),
            'use_fake_methods_for_validation': list(EFS_METHODS | REG_METHODS),
        }
    }

    # Configure logging to see the output from the script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    log.info("Starting data preparation to generate manifest and splits...")
    log.info("Starting OOD data preparation to generate OOD manifest...")
    ood_videos = prepare_ood_videos(data_cfg=config)
    log.info(f"Discovered {len(ood_videos)} OOD videos.")
    log.info("Manifest 'ood_frame_manifest.json' has been created/updated.")

    # log.info("--- Data Preparation Summary ---")
    # log.info(f"Total videos in training set: {len(train_videos)}")
    # log.info(f"Total videos in validation set: {len(val_videos)}")
    # log.info("Detailed stats:")
    # log.info(json.dumps(stats, indent=2))
    # log.info("Manifest 'frame_manifest.json' has been created/updated in the current directory.")
