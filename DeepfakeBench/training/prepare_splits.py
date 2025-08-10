# --- prepare_splits.py ---

# !/usr/bin/env python3
"""prepare_splits.py – train/val split with GroupShuffleSplit + local manifest cache.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple

import yaml  # noqa
from fsspec.core import url_to_fs  # pip install gcsfs  # noqa
from sklearn.model_selection import GroupShuffleSplit  # pip install scikit-learn  # noqa

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

    def __post_init__(self):
        # 1) strict label check
        if self.label not in {"real", "fake"}:
            raise ValueError(
                f"[VideoInfo] Invalid label '{self.label}' "
                f"for video '{self.method}/{self.video_id}'. "
                "Allowed: 'real' or 'fake'."
            )

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


# ---------------------------------------------------------------------------
# 3 Main entry
# ---------------------------------------------------------------------------
def prepare_video_splits(cfg_path: str = "config.yaml"
                         ) -> Tuple[List[VideoInfo], List[VideoInfo], dict, dict]:
    """
    Prepares video splits for training and validation.

    Returns:
        Tuple containing:
        - train_videos (List[VideoInfo]): The balanced list of training videos.
        - val_videos (List[VideoInfo]): The balanced list of validation videos.
        - cfg (dict): The loaded configuration.
        - stats (dict): A dictionary with statistics about the data preparation process.
    """
    cfg = yaml.safe_load(open(cfg_path))
    BUCKET = f"gs://{cfg['gcp']['bucket_name']}"
    SEED = cfg['data_params']['seed']
    VAL_RATIO = cfg['data_params']['val_split_ratio']
    SUBSET = cfg['data_params']['data_subset_percentage']
    allowed = set(cfg['methods']['use_real_sources']
                  + cfg['methods']['use_fake_methods'])

    # NEW: Dictionary to hold statistics
    stats = {}

    random.seed(SEED)
    manifest_path = Path(__file__).with_name("frame_manifest.json")

    # ------------------------------------------------------------------ #
    # 1) Load cached manifest if it exists, else list & cache it        #
    # ------------------------------------------------------------------ #
    if manifest_path.exists():
        frame_paths = json.loads(manifest_path.read_text())
        print(f"[manifest] Loaded {len(frame_paths):,} frame paths from cache")
    else:
        print("Listing frame objects on GCS (first run – may take a minute)…")
        fs = url_to_fs(BUCKET)[0]
        frame_paths = [
            f"gs://{p}"
            for p in fs.glob(f"{BUCKET}/**")
            if Path(p).suffix.lower() in {'.png', '.jpg', '.jpeg'}
        ]
        print(f"Found {len(frame_paths):,} frame files – caching manifest")
        manifest_path.write_text(json.dumps(frame_paths))

    # Count and print the number of frames from the specified method
    avspeech_count = sum(1 for p in frame_paths if "external_youtube_avspeech" in p)
    print(f"[manifest] Found {avspeech_count:,} frames from 'external_youtube_avspeech'")
    stats['total_frames_in_manifest'] = len(frame_paths)

    # ------------------------------------------------------------------ #
    # 2) Group frames → VideoInfo objects                                #
    # ------------------------------------------------------------------ #
    vids_dict: dict[tuple[str, str, str], List[str]] = defaultdict(list)
    for p in frame_paths:
        parts = Path(p).parts
        try:
            label, method, vid = parts[-4], parts[-3], parts[-2]
            if label not in {"real", "fake"}:
                print(f"[WARN] Invalid label '{label}' in path '{p}'. Skipping.")
                continue
        except Exception:
            continue
        if method not in allowed:
            continue
        vids_dict[(label, method, vid)].append(p)

    videos: List[VideoInfo] = []
    warned = set()
    for (label, method, vid), fr in vids_dict.items():
        if method in EXCLUDE_METHODS:
            if method not in warned:
                print(f"[WARN] excluding all videos from method '{method}'")
                warned.add(method)
            continue
        fr.sort()
        tid = extract_target_id(label, method, vid)
        if tid is None:  # synthetic unique
            tid = (hash((method, vid)) & 0x7FFFFFFF) + 100_000
        videos.append(VideoInfo(label, method, vid, fr, tid))

    print(f"Discovered {len(videos):,} videos across {len(allowed)} methods")
    stats['discovered_videos'] = len(videos)
    stats['discovered_methods'] = len(allowed)
    stats['subset_video_count'] = len(videos)  # Default value

    # ------------------------------------------------------------------ #
    # 3. (NEW) Apply subset percentage to the ENTIRE dataset FIRST       #
    # ------------------------------------------------------------------ #
    if SUBSET < 1.0:
        print(f"\n[subset] Applying `data_subset_percentage`={SUBSET:.2f} to the entire dataset.")
        # Use GroupShuffleSplit to get a representative subset while respecting identities
        subset_splitter = GroupShuffleSplit(n_splits=1, train_size=SUBSET, random_state=SEED)
        all_identities = [v.identity for v in videos]
        try:
            # The 'train_idx' from this split will be our subset
            subset_indices, _ = next(subset_splitter.split(X=videos, groups=all_identities))
            videos = [videos[i] for i in subset_indices]
            print(f"[subset] Dataset reduced to {len(videos):,} videos.")
            stats['subset_video_count'] = len(videos)
        except ValueError as e:
            print(f"[subset] WARN: Could not create a subset while respecting groups. Error: {e}")
            print("[subset] Using a random sample instead. This may cause minor identity leakage in the subset.")
            random.shuffle(videos)
            subset_size = int(len(videos) * SUBSET)
            videos = videos[:subset_size]
            stats['subset_video_count'] = len(videos)

    # ------------------------------------------------------------------ #
    # 4. Stratified GroupShuffleSplit (on the potentially smaller dataset)#
    # ------------------------------------------------------------------ #
    print(f"\n[split] Performing Stratified GroupShuffleSplit on {len(videos):,} videos...")
    train_idx, val_idx = [], []
    videos_by_method = defaultdict(list)
    for i, v in enumerate(videos):
        videos_by_method[v.method].append((i, v))

    for method, method_videos_with_indices in videos_by_method.items():
        method_indices = [item[0] for item in method_videos_with_indices]
        method_groups = [item[1].identity for item in method_videos_with_indices]

        if len(method_indices) < 2 or len(set(method_groups)) < 2:
            train_idx.extend(method_indices)
            continue

        gss = GroupShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
        try:
            method_train_indices_local, method_val_indices_local = next(gss.split(
                X=method_videos_with_indices, groups=method_groups
            ))
            train_idx.extend([method_indices[i] for i in method_train_indices_local])
            val_idx.extend([method_indices[i] for i in method_val_indices_local])
        except ValueError:
            train_idx.extend(method_indices)

    # These are the complete, unbalanced splits
    train_videos_unbalanced = [videos[i] for i in train_idx]
    val_videos_unbalanced = [videos[i] for i in val_idx]
    print(f"Split complete (unbalanced) ▶ train {len(train_videos_unbalanced):,} | val {len(val_videos_unbalanced):,}")
    stats['unbalanced_train_count'] = len(train_videos_unbalanced)
    stats['unbalanced_val_count'] = len(val_videos_unbalanced)

    # ------------------------------------------------------------------ #
    # 5. (NEW) Balance BOTH the training and validation sets             #
    # ------------------------------------------------------------------ #
    print("\n--- Balancing Training Set ---")
    train_videos = _balance_video_list(
        videos=train_videos_unbalanced,
        real_source_names=cfg['methods']['use_real_sources']
    )

    print("\n--- Balancing Validation Set ---")
    val_videos = _balance_video_list(
        videos=val_videos_unbalanced,
        real_source_names=cfg['methods']['use_real_sources']
    )

    print(f"\nFinal balanced split ▶ train {len(train_videos):,} | val {len(val_videos):,}")
    stats['balanced_train_count'] = len(train_videos)
    stats['balanced_val_count'] = len(val_videos)

    random.shuffle(train_videos)
    random.shuffle(val_videos)
    return train_videos, val_videos, cfg, stats


if __name__ == "__main__":
    tr, va, _ = prepare_video_splits()
    print("train example:", tr[0].method, tr[0].video_id, tr[0].identity)
    print("val   example:", va[0].method, va[0].video_id, va[0].identity)
