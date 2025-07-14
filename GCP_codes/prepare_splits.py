#!/usr/bin/env python3
"""prepare_splits.py–train/val split with **target‑identity awareness**

Three method groups
===================
* **EFS_METHODS**– entire‑face synthesis → each video gets its own synthetic
  identity (no clash with real IDs).
* **REG_METHODS**– folder name pattern`source_target`(→ *target* is **2nd**
  3‑digit token or the only one present).
* **REV_METHODS**– pattern`target_source`(→ *target* is **1st** 3‑digit
  token).

Real videos
-----------
* **FaceForensics++**(real) retain their numeric ID (000‑999).
* **Celeb‑real / YouTube‑real** have no 3‑digit token → they are treated like
  EFS and assigned a synthetic unique ID so they can safely land in any split.

Fill `REG_METHODS` and `REV_METHODS` below with your final lists; sensible
pre‑fills are provided from DF‑40 documentation.
"""
from __future__ import annotations

import random
import re
import yaml  # noqa
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple

from fsspec.core import get_fs_for_uri  # noqa

# ---------------------------------------------------------------------------
# 1Method categories (edit REG/REV if needed)
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
REV_METHODS = {}

EXCLUDE_METHODS: Set[str] = {
    "hyperreenact"
}  # methods to exclude from the split

_RE_3DIGIT = re.compile(r"\b(\d{3})\b")


# ---------------------------------------------------------------------------
@dataclass
class VideoInfo:
    label: str  # 'real' | 'fake'
    method: str  # generation/source method
    video_id: str  # folder name
    frame_paths: List[str]  # gs:// paths to frames
    identity: int  # target identity (numeric)


# ---------------------------------------------------------------------------
# 2Identity extraction helper
# ---------------------------------------------------------------------------

def extract_target_id(label: str, method: str, vid_folder: str) -> int | None:
    """Return integer target ID or None (synthetic).

    • For EFS and non‑FF++ real clips (Celeb‑real / YouTube‑real) ⇒ None.
    • For REG  ⇒ last 3‑digit token (or only token).
    • For REV  ⇒ first 3‑digit token.
    """
    if method in EFS_METHODS:
        return None

    if label == "real" and method != "FaceForensics++":
        return None  # Celeb‑real / YouTube‑real treated as synthetic

    ids = [int(tok) for tok in _RE_3DIGIT.findall(vid_folder)]
    if not ids:
        return None  # fallback to synthetic if no 3‑digit chunk

    if method in REV_METHODS:
        return ids[0]
    # default → REG
    return ids[-1]


# ---------------------------------------------------------------------------
# 3Main function
# ---------------------------------------------------------------------------

def prepare_video_splits(cfg_path: str = "config.yaml") -> Tuple[List[VideoInfo], List[VideoInfo], dict]:
    cfg = yaml.safe_load(open(cfg_path))

    BUCKET = f"gs://{cfg['gcp']['bucket_name']}"
    SEED = cfg['data_params']['seed']
    VAL_RATIO = cfg['data_params']['val_split_ratio']
    SUBSET = cfg['data_params']['data_subset_percentage']

    allowed_methods = set(cfg['methods']['use_real_sources'] + cfg['methods']['use_fake_methods'])

    fs = get_fs_for_uri(BUCKET)[0]

    print("Listing frame objects …")
    frame_paths = [p for p in fs.glob(f"{BUCKET}/**") if Path(p).suffix.lower() in {'.png', '.jpg', '.jpeg'}]
    print(f"Found {len(frame_paths):,} frame files")

    # group by video --------------------------------------------------------
    vids_dict: dict[tuple[str, str, str], List[str]] = defaultdict(list)
    for p in frame_paths:
        parts = Path(p).parts
        try:
            label, method, vid = parts[-4], parts[-3], parts[-2]
        except IndexError:
            continue
        if method not in allowed_methods:
            continue
        vids_dict[(label, method, vid)].append(p)

    seen_excluded = set()
    videos: List[VideoInfo] = []
    for (label, method, vid), fr in vids_dict.items():
        if method in EXCLUDE_METHODS:
            if method not in seen_excluded:
                print(f"[WARN] excluding all videos from method '{method}'")  # single warn
                seen_excluded.add(method)
            continue
        fr.sort()
        tid = extract_target_id(label, method, vid)
        if tid is None:
            tid = (hash((method, vid)) & 0x7FFFFFFF) + 100000  # synthetic unique
        videos.append(VideoInfo(label, method, vid, fr, tid))

    print(f"Discovered {len(videos):,} videos across {len(allowed_methods)} methods")

    random.seed(SEED)

    # optional per‑method subset ------------------------------------------
    if SUBSET < 1.0:
        per_method = defaultdict(list)
        for v in videos:
            per_method[v.method].append(v)
        selected: List[VideoInfo] = []
        for m, lst in per_method.items():
            random.shuffle(lst)
            keep = max(1, int(len(lst) * SUBSET))
            selected.extend(lst[:keep])
    else:
        selected = videos

    total = len(selected)
    print(f"After subset: {total:,} videos")

    # group by identity ----------------------------------------------------
    id2videos = defaultdict(list)
    for v in selected:
        id2videos[v.identity].append(v)

    identities = list(id2videos)
    random.shuffle(identities)

    target_val = int(total * VAL_RATIO)
    train_videos: List[VideoInfo] = []
    val_videos: List[VideoInfo] = []
    val_cnt = 0
    for i in identities:
        vids = id2videos[i]
        if val_cnt < target_val:
            val_videos.extend(vids)
            val_cnt += len(vids)
        else:
            train_videos.extend(vids)

    print(f"Split complete ▶ train {len(train_videos):,} | val {len(val_videos):,}")

    random.shuffle(train_videos)
    random.shuffle(val_videos)

    return train_videos, val_videos, cfg


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tr, va, _ = prepare_video_splits("config.yaml")
    if tr:
        print("train example:", tr[0].method, tr[0].video_id, tr[0].identity)
    if va:
        print("val   example:", va[0].method, va[0].video_id, va[0].identity)
