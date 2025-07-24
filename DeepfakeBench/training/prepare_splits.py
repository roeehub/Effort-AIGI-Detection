#!/usr/bin/env python3
"""prepare_splits.py – train/val split with GroupShuffleSplit + local manifest cache.

See README_prefetch.md for notes on tiny-method exclusion and resource tuning.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple

import yaml
from fsspec.core import url_to_fs  # pip install gcsfs
from sklearn.model_selection import GroupShuffleSplit  # pip install scikit-learn
import itertools ### ADAM CHANGED
TOTAL_FILES_ESTIMATE = 1_000_000  # An estimate of the total number of images ### ADAM CHANGED
SAMPLE_PERCENTAGE = 0.05 ### ADAM CHANGED
limit = int(TOTAL_FILES_ESTIMATE * SAMPLE_PERCENTAGE) ### ADAM CHANGED

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
    # 1 Load or build manifest of frame paths                            #
    # ------------------------------------------------------------------ #
    ###########################################################################
    # if manifest_path.exists():
    #     frame_paths = json.loads(manifest_path.read_text())
    #     print(f"[manifest] Loaded {len(frame_paths):,} frame paths from cache")
    # else:
    ###########################################################################
    print("Listing frame objects on GCS (first run – may take a minute)…")
    fs = url_to_fs(BUCKET)[0]
    #################### ADAM CHANGED ################################
    # frame_paths = [f"gs://{p}" for p in fs.glob(f"{BUCKET}/**")
    #                 if Path(p).suffix.lower() in {'.png', '.jpg', '.jpeg'}]
    # fs.glob returns an iterator, which is memory-efficient
    MANIFEST_FILENAME = "/home/roee/repos/Effort-AIGI-Detection/partial_manifest.json"
    manifest_path = Path(MANIFEST_FILENAME)

    if not manifest_path.exists():
        print(f"Error: Manifest file not found at '{manifest_path}'")
        print("Please run 'create_partial_manifest.py' first to generate the file list.")
        return None

    print(f"Loading cached file paths from '{manifest_path}'...")
    with open(manifest_path, 'r') as f:
        frame_paths = json.load(f)
    #################### ADAM CHANGED ################################
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
        except IndexError:
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

    # ------------------------------------------------------------------ #
    # 3 Optional per-method subset (for fast debugging)                  #
    # ------------------------------------------------------------------ #
    if SUBSET < 1.0:
        per_method = defaultdict(list)
        for v in videos:
            per_method[v.method].append(v)
        selected = []
        for m, lst in per_method.items():
            random.shuffle(lst)
            keep = max(1, int(len(lst) * SUBSET))
            selected.extend(lst[:keep])
        print(f"Subset active ({SUBSET * 100:.0f} %) → {len(selected):,} videos")
    else:
        selected = videos

    # ------------------------------------------------------------------ #
    # 4 GroupShuffleSplit (identity-aware, video-balanced)               #
    # ------------------------------------------------------------------ #
    gss = GroupShuffleSplit(n_splits=1,
                            test_size=VAL_RATIO,
                            random_state=SEED)
    indices = list(range(len(selected)))
    groups = [v.identity for v in selected]
    train_idx, val_idx = next(gss.split(indices, groups=groups))

    train_videos = [selected[i] for i in train_idx]
    val_videos = [selected[i] for i in val_idx]

    # sanity check
    actual_ratio = len(val_videos) / len(selected)
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