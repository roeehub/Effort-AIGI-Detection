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
    all_allowed_methods = real_methods | train_fake_methods | val_fake_methods

    # The stats dictionary will be built progressively.
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

        if method in real_methods:
            training_pool.append(video)
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
    val_videos = _balance_video_list(videos=validation_pool, real_source_names=list(real_methods))

    stats['balanced_train_count'] = len(train_videos)
    stats['balanced_val_count'] = len(val_videos)
    log.info(
        f"Final balanced split ▶ train: {stats['balanced_train_count']:,} videos | val: {stats['balanced_val_count']:,} videos")

    random.shuffle(train_videos)
    random.shuffle(val_videos)

    # --- 7. MODIFIED RETURN STATEMENT FOR COMPATIBILITY ---
    # The legacy path returns its single validation set as the "holdout" set
    # and an empty list for the "in-distribution" set. This makes it compatible
    # with the new 4-tuple return signature expected by the main training script.
    val_in_dist_videos = []
    val_holdout_videos = val_videos

    return train_videos, val_in_dist_videos, val_holdout_videos, stats


def prepare_splits_property_based(data_cfg: dict) -> Tuple[List[Dict], List[VideoInfo], List[VideoInfo], Dict]:
    """
    [REVISED & FIXED]
    Prepares data splits using frame properties and a strict identity-based split.
    This version correctly populates the stats dictionary for the run overview.
    """
    cfg = data_cfg
    SEED = cfg['data_params']['seed']
    VAL_SPLIT_RATIO = cfg['data_params'].get('val_split_ratio', 0.1)
    PROPERTIES_FILE = cfg['property_balancing']['frame_properties_parquet_path']

    log.info(f"--- Preparing splits with Property-Based Strategy (Seed: {SEED}) ---")
    log.info(f"Loading frame properties from: {PROPERTIES_FILE}")

    # --- 1. Define method sets from config ---
    real_methods = set(cfg['methods']['use_real_sources'])
    train_fake_methods = set(cfg['methods']['use_fake_methods_for_training'])
    val_fake_methods = set(cfg['methods']['use_fake_methods_for_validation'])
    all_allowed_methods = real_methods | train_fake_methods | val_fake_methods
    stats = {}

    # --- 2. Load and filter data ---
    df = pd.read_parquet(PROPERTIES_FILE)

    # [FIX] Populate initial discovery stats from the raw Parquet file
    # before any filtering, to match the legacy path's logic and prevent the error.
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

    # --- 3. Perform strict identity-based split using the 'video_id' column ---
    log.info(f"Performing identity-based split with validation ratio: {VAL_SPLIT_RATIO}")
    unique_ids = df_filtered['video_id'].unique()
    train_ids, val_ids = train_test_split(unique_ids, test_size=VAL_SPLIT_RATIO, random_state=SEED)
    train_ids, val_ids = set(train_ids), set(val_ids)

    train_df = df_filtered[df_filtered['video_id'].isin(train_ids)]
    val_df = df_filtered[df_filtered['video_id'].isin(val_ids)]

    # [FIX] Rename keys to match what the train script expects for the overview.
    # The train script expects 'unbalanced_train_count' for frames and 'unbalanced_val_count' for videos.
    stats['unbalanced_train_count'] = len(train_df)
    stats['unbalanced_val_count'] = val_df['video_id'].nunique()
    log.info(
        f"Split pools by identity ▶ Train: {stats['unbalanced_train_count']:,} frames | Val: {stats['unbalanced_val_count']:,} videos")

    # --- 4. Balance the training set (DataFrame) and convert to dicts ---
    log.info("--- Balancing Training Set ---")
    train_df_balanced = _balance_df_by_label(train_df, real_methods, SEED)
    final_train_data = train_df_balanced.to_dict('records')
    stats['train_frame_count'] = len(final_train_data)
    stats['train_video_count'] = len(train_df_balanced['video_id'].unique())

    # --- 5. Assemble val pool into VideoInfo objects and subdivide ---
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

    # --- 6. Balance both validation sets independently ---
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
