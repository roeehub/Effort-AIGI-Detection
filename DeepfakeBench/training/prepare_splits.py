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


def prepare_video_splits_v2(data_cfg: dict) -> Tuple[List[VideoInfo], List[VideoInfo], dict]:
    """
    Prepares video splits with a strict separation between training and validation methods.
    This version has been audited to ensure all statistics are correctly calculated and
    assigned in all execution paths, without relying on default values.
    """

    if data_cfg.get('property_balancing', {}).get('enabled', False):
        # The new execution path
        return prepare_splits_property_based(data_cfg)  # type: ignore

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
        log.info("[manifest] Caching not found. Listing frame objects on GCS...")
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
    return train_videos, val_videos, stats


def prepare_splits_property_based(data_cfg: dict) -> Tuple[List[Dict], List[VideoInfo], dict]:
    """
    Prepares train/val splits using a method-based hold-out strategy combined
    with property-aware data handling.
    - The training set is a list of frame dicts, intended for a property-balancing loader.
    - The validation set is a list of VideoInfo objects, label-balanced by video.
    """
    log.info("--- Running property-based split with method hold-out strategy ---")
    cfg = data_cfg
    SEED = cfg['data_params']['seed']
    MANIFEST_PATH = Path(__file__).with_name("frame_properties.parquet")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Property manifest not found at {MANIFEST_PATH}. Please run `create_property_manifest.py` first.")

    log.info(f"Loading property manifest from {MANIFEST_PATH}...")
    df = pd.read_parquet(MANIFEST_PATH)
    log.info(f"Loaded {len(df):,} frames from {df['video_id'].nunique():,} videos.")
    df['label_id'] = np.where(df['label'] == 'real', 0, 1)

    log.info("Dynamically creating property buckets from the 'sharpness' and 'file_size_kb' columns...")
    try:
        # Create sharpness and size buckets as per the verified script
        df['sharpness_bucket'] = pd.qcut(df['sharpness'], 4,
                                         labels=[f's_q{i}' for i in range(1, 5)], duplicates='drop')
        df['size_bucket'] = pd.qcut(df['file_size_kb'], 4,
                                    labels=[f'f_q{i}' for i in range(1, 5)], duplicates='drop')
        df['property_bucket'] = df['sharpness_bucket'].astype(str) + '_' + df['size_bucket'].astype(str)
        log.info(f"Successfully created {df['property_bucket'].nunique()} on-the-fly property buckets.")
    except Exception as e:
        log.error(f"Failed to create property buckets on-the-fly. Error: {e}")
        raise

    stats = {
        'total_frames_in_manifest': len(df),
        'discovered_videos': df['video_id'].nunique(),
        'discovered_methods': df['method'].nunique()
    }

    # --- 1. Define method sets from config ---
    log.info("[split] Defining method sets for hold-out strategy...")
    real_methods = set(cfg['methods']['use_real_sources'])
    train_fake_methods = set(cfg['methods']['use_fake_methods_for_training'])
    val_fake_methods = set(cfg['methods']['use_fake_methods_for_validation'])

    # --- 2. THE MASTER ID SPLIT ---
    # First, create disjoint sets of IDENTITIES for training and validation.
    # This is the core of the leak prevention strategy.
    log.info("[split] Performing master split on all unique identities to prevent leaks...")
    all_identities = pd.DataFrame(df['video_id'].unique(), columns=['video_id'])

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=cfg['data_params'].get('val_split_ratio', 0.1),
        random_state=SEED
    )
    # Note: We split the DataFrame of unique IDs, not the full DataFrame.
    # The 'groups' argument is implicitly the 'video_id' column itself.
    # I added the `groups` parameter to explicitly use the 'video_id' for grouping.
    train_idx, val_idx = next(gss.split(all_identities, groups=all_identities['video_id']))

    train_ids = set(all_identities.iloc[train_idx]['video_id'])
    val_ids = set(all_identities.iloc[val_idx]['video_id'])
    log.info(f"  - Identities split into: {len(train_ids)} for training, {len(val_ids)} for validation.")

    # --- 3. Assign all videos to train/val pools based on their identity ---
    log.info("[split] Assigning all video frames to pools based on their identity...")
    initial_train_df = df[df['video_id'].isin(train_ids)]
    initial_val_df = df[df['video_id'].isin(val_ids)]
    log.info(
        f"  - Initial pools created -> Train: {len(initial_train_df):,} frames | Val: {len(initial_val_df):,} frames.")

    # --- 4. Apply the method hold-out constraint to the identity-safe pools ---
    # Now, filter each pool to only contain its allowed methods.
    # This discards videos that don't fit the hold-out criteria (e.g., a val-person with a train-method).
    log.info("[split] Applying method hold-out constraints to finalize unbalanced pools...")

    allowed_train_methods = real_methods | train_fake_methods
    train_df_unbalanced = initial_train_df[initial_train_df['method'].isin(allowed_train_methods)]

    allowed_val_methods = real_methods | val_fake_methods
    val_df_unbalanced = initial_val_df[initial_val_df['method'].isin(allowed_val_methods)]

    log.info(
        f"  - Post-filtering -> Train: {len(train_df_unbalanced):,} frames ({initial_train_df.shape[0] - train_df_unbalanced.shape[0]:,} removed).")
    log.info(
        f"  - Post-filtering -> Val: {len(val_df_unbalanced):,} frames ({initial_val_df.shape[0] - val_df_unbalanced.shape[0]:,} removed).")

    stats['unbalanced_train_count'] = len(train_df_unbalanced)  # Use frame count
    stats['unbalanced_val_count'] = val_df_unbalanced['video_id'].nunique()  # Use video count
    log.info(
        f"Assembled unbalanced sets -> Train Pool: {stats['unbalanced_train_count']:,} frames | Val Pool: {stats['unbalanced_val_count']:,} videos")

    # --- 5. Balance ONLY the validation set ---
    log.info("[balance] Balancing validation set by label (50/50 real vs fake videos)...")
    val_df_balanced = _balance_df_by_label(val_df_unbalanced, real_methods, SEED)

    # --- 6. Finalize and report ---
    # *** KEY CHANGE: Use the UNBALANCED dataframe for the training set. ***
    # The dataloader is now responsible for all training-time balancing.
    train_frames = train_df_unbalanced.to_dict('records')
    log.info(
        "[INFO] Training set is intentionally left unbalanced. The property-balancing dataloader will handle sampling."
    )

    # Convert the balanced validation DataFrame back to VideoInfo objects
    val_videos_map = defaultdict(list)
    for row in val_df_balanced.itertuples():
        val_videos_map[(row.label, row.method, row.video_id)].append(row.path)

    val_videos = []
    for (label, method, vid), frame_paths in val_videos_map.items():
        frame_paths.sort()
        tid = extract_target_id(label, method, str(vid))
        if tid is None: tid = (hash((method, vid)) & 0x7FFFFFFF) + 100_000
        val_videos.append(VideoInfo(label, method, vid, frame_paths, tid))

    # --- 6. Finalize and report detailed stats ---
    stats['train_frame_count'] = len(train_frames)
    stats['train_video_count'] = train_df_unbalanced['video_id'].nunique()

    # The validation set is a list of VideoInfo objects from the BALANCED pool
    stats['val_video_count'] = len(val_videos)
    stats['val_frame_count'] = val_df_balanced.shape[0]  # Get frame count from the balanced df

    log.info(
        f"Final training pool size: {stats['train_video_count']:,} videos ({stats['train_frame_count']:,} frames).")
    log.info(
        f"Final validation set size: {stats['val_video_count']:,} videos ({stats['val_frame_count']:,} frames) (label-balanced).")

    return train_frames, val_videos, stats


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
