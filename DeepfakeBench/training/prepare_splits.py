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


def _balance_and_subset_videos(
        videos: List[VideoInfo],
        subset_percentage: float,
        real_source_names: List[str]
) -> List[VideoInfo]:
    """Balances the number of real and fake videos and applies subset percentage."""
    if subset_percentage >= 1.0:
        print("[balance] `data_subset_percentage` >= 1.0, using full dataset but balancing classes.")
        # Even if using 100%, we still balance the classes to the size of the smaller one
        subset_percentage = 1.0

    print(f"[balance] Balancing dataset with subset percentage: {subset_percentage:.2f}")

    # 1. Categorize videos by label and then by method/source
    reals_by_source = defaultdict(list)
    fakes_by_method = defaultdict(list)
    for v in videos:
        if v.method in real_source_names:
            reals_by_source[v.method].append(v)
        else:
            fakes_by_method[v.method].append(v)

    num_total_reals = sum(len(vids) for vids in reals_by_source.values())
    num_total_fakes = sum(len(vids) for vids in fakes_by_method.values())

    if num_total_reals == 0 or num_total_fakes == 0:
        print("[balance] WARN: One class has 0 videos. Cannot balance.")
        return videos

    print(f"[balance] Initial counts: {num_total_reals} real, {num_total_fakes} fake videos.")

    # 2. Determine smaller class and target number of videos
    smaller_class_size = min(num_total_reals, num_total_fakes)
    target_videos_per_class = max(1, int(smaller_class_size * subset_percentage))
    print(f"[balance] Target videos per class: {target_videos_per_class}")

    # 3. Subsample REAL videos proportionally from each source
    final_real_videos = []
    if num_total_reals > 0:
        for source, source_videos in reals_by_source.items():
            proportion = len(source_videos) / num_total_reals
            num_to_take = round(proportion * target_videos_per_class)
            random.shuffle(source_videos)
            final_real_videos.extend(source_videos[:int(num_to_take)])

    # 4. Subsample FAKE videos proportionally from each method
    final_fake_videos = []
    if num_total_fakes > 0:
        for method, method_videos in fakes_by_method.items():
            proportion = len(method_videos) / num_total_fakes
            num_to_take = round(proportion * target_videos_per_class)
            random.shuffle(method_videos)
            final_fake_videos.extend(method_videos[:int(num_to_take)])

    # 5. Adjust counts to be exactly equal (due to rounding)
    while len(final_real_videos) > target_videos_per_class:
        final_real_videos.pop()
    while len(final_fake_videos) > target_videos_per_class:
        final_fake_videos.pop()

    final_target = min(len(final_real_videos), len(final_fake_videos))
    final_real_videos = final_real_videos[:final_target]
    final_fake_videos = final_fake_videos[:final_target]

    balanced_videos = final_real_videos + final_fake_videos
    random.shuffle(balanced_videos)

    print(f"[balance] Final balanced counts: {len(final_real_videos)} real, {len(final_fake_videos)} fake videos.")

    return balanced_videos


# ---------------------------------------------------------------------------
# 3 Main entry
# ---------------------------------------------------------------------------
def prepare_video_splits(cfg_path: str = "config.yaml"
                         ) -> Tuple[List[VideoInfo], List[VideoInfo], dict]:
    cfg = yaml.safe_load(open(cfg_path))
    BUCKET = f"gs://{cfg['gcp']['bucket_name']}"
    SEED = cfg['data_params']['seed']
    VAL_RATIO = cfg['data_params']['val_split_ratio']
    SUBSET = cfg['data_params']['data_subset_percentage']
    allowed = set(cfg['methods']['use_real_sources']
                  + cfg['methods']['use_fake_methods'])

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

    # ------------------------------------------------------------------ #
    # 2 Group frames → VideoInfo objects                                 #
    # ------------------------------------------------------------------ #
    vids_dict: dict[tuple[str, str, str], List[str]] = defaultdict(list)
    for p in frame_paths:
        parts = Path(p).parts
        try:
            label, method, vid = parts[-4], parts[-3], parts[-2]
            if label not in {"real", "fake"}:
                # print(f"[WARN] Skipping invalid label: {label} in {p}")
                continue
        except Exception:
            # print(f"[WARN] Skipping problematic path: {p}")
            continue
        if method not in allowed:
            # print(f"[WARN] Skipping path with disallowed method: {method} in {p}")
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

    # --- START OF PATCH ---
    # Apply balancing and subsetting to the *entire* dataset *before* splitting.
    # This ensures the train and validation sets are scaled down proportionally.
    print("\n--- Applying Balancing and Subsetting to the Entire Dataset ---")
    balanced_subset_videos = _balance_and_subset_videos(
        videos=videos,
        subset_percentage=SUBSET,
        real_source_names=cfg['methods']['use_real_sources']
    )

    if not balanced_subset_videos:
        print(
            "[ERROR] No videos remaining after balancing/subsetting. Check your data and config. Returning empty lists.")
        return [], [], cfg

    print("\n--- Performing Train/Validation Split on the Scaled Dataset ---")
    # ------------------------------------------------------------------ #
    # 3 GroupShuffleSplit (identity-aware, video-balanced)               #
    # ------------------------------------------------------------------ #
    gss = GroupShuffleSplit(n_splits=1,
                            test_size=VAL_RATIO,
                            random_state=SEED)
    indices = list(range(len(balanced_subset_videos)))
    groups = [v.identity for v in balanced_subset_videos]
    train_idx, val_idx = next(gss.split(indices, groups=groups))

    train_videos = [balanced_subset_videos[i] for i in train_idx]
    val_videos = [balanced_subset_videos[i] for i in val_idx]

    # sanity check
    total_subset_videos = len(balanced_subset_videos)
    actual_ratio = len(val_videos) / total_subset_videos if total_subset_videos > 0 else 0
    if abs(actual_ratio - VAL_RATIO) > 0.005:
        print(f"[NOTE] Val ratio {actual_ratio:.3f} differs >0.5 % from "
              f"target {VAL_RATIO:.3f}")

    print(f"Split complete ▶ train {len(train_videos):,} | "
          f"val {len(val_videos):,}")

    random.shuffle(train_videos)
    random.shuffle(val_videos)
    return train_videos, val_videos, cfg


if __name__ == "__main__":
    tr, va, _ = prepare_video_splits()
    print("train example:", tr[0].method, tr[0].video_id, tr[0].identity)
    print("val   example:", va[0].method, va[0].video_id, va[0].identity)
