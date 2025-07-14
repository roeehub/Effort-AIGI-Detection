# dataloaders.py
import random
from collections import defaultdict
from itertools import cycle
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import (
    IterableWrapper, Zipper, Demultiplexer, CyclingIterator
)
from torchvision import transforms
from fsspec.core import open_files

from prepare_splits import VideoInfo


# ---------------------------------------------------------------------------
# 1 Frame loading helper
# ---------------------------------------------------------------------------
def load_video_frames(
        video_info: VideoInfo,
        num_frames: int,
        augmentation: callable | None = None
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Load `num_frames` random frames from one video and return a tensor,
    its label-tensor, and the method name.
    """
    paths = (random.sample(video_info.frame_paths, num_frames)
             if len(video_info.frame_paths) > num_frames
             else random.choices(video_info.frame_paths, k=num_frames))

    files = open_files(paths, mode="rb")
    frames: List[torch.Tensor] = []
    for fobj in files:
        with fobj as stream:
            img = Image.open(stream).convert("RGB")
            frames.append(transforms.ToTensor()(img))

    # --------------------------------------------------------------- #
    # NOTE: tensor shape is (C, T, H, W).  If your model expects
    #       (T, C, H, W) **swap dims here** (e.g. video_tensor = video_tensor.permute(1,0,2,3)).
    # --------------------------------------------------------------- #
    video_tensor = torch.stack(frames, dim=1)

    # Developer-friendly comment on label dtype
    # ----------------------------------------
    # • For BCEWithLogitsLoss → float {0.0/1.0} is fine (current default).
    # • For CrossEntropyLoss  → use dtype=torch.long and values {0,1}.
    label_tensor = torch.tensor(1.0 if video_info.label == "real" else 0.0)

    if augmentation:
        video_tensor = augmentation(video_tensor)

    return video_tensor, label_tensor, video_info.method


# ---------------------------------------------------------------------------
# 2 Base DataPipe builder
# ---------------------------------------------------------------------------
def create_base_videopipe(videos: List[VideoInfo], num_frames: int,
                          augmentation):
    return (
        IterableWrapper(videos)
        .shuffle()
        .sharding_filter()
        .map(lambda x: load_video_frames(x, num_frames, augmentation))
        .prefetch(10)  # tune in README_prefetch.md
    )


# ---------------------------------------------------------------------------
# 3 Naive 50/50 real-fake loader (now non-truncating)
# ---------------------------------------------------------------------------
def create_naive_dataloader(train_videos, config):
    cfg_data = config["data_params"]
    cfg_dl = config["dataloader_params"]

    real_videos = [v for v in train_videos if v.label == "real"]
    fake_videos = [v for v in train_videos if v.label == "fake"]
    print(f"Naive DL: {len(real_videos)} real | {len(fake_videos)} fake")

    real_pipe = create_base_videopipe(real_videos,
                                      cfg_data["num_frames_per_video"], None)
    fake_pipe = create_base_videopipe(fake_videos,
                                      cfg_data["num_frames_per_video"], None)

    # CyclingIterator keeps the *shorter* side looping so nothing is dropped
    combined_pipe = (
        Zipper(CyclingIterator(real_pipe), CyclingIterator(fake_pipe))
        .mapcat(lambda pair: pair)
    )

    return DataLoader(
        combined_pipe,
        batch_size=cfg_dl["batch_size"],
        num_workers=cfg_dl["num_workers"],
    )


# ---------------------------------------------------------------------------
# 4 Method-aware loaders
# ---------------------------------------------------------------------------
def create_method_aware_dataloaders(train_videos, config):
    cfg_data = config["data_params"]
    cfg_dl = config["dataloader_params"]

    videos_by_method = defaultdict(list)
    for v in train_videos:
        videos_by_method[v.method].append(v)

    dataloaders = {}
    for method, vids in videos_by_method.items():
        print(f"[method] {method:15s} → {len(vids):4d} videos")
        pipe = create_base_videopipe(vids,
                                     cfg_data["num_frames_per_video"], None)
        loader = DataLoader(
            pipe,
            batch_size=cfg_dl["batch_size"],
            num_workers=cfg_dl["num_workers"],
        )
        loader.dataset_size = len(vids)  # expose for weighting
        dataloaders[method] = loader
    return dataloaders


# ---------------------------------------------------------------------------
# 5 Utility: temperature-based inverse-frequency weighting
# ---------------------------------------------------------------------------
def temperatured_weights(sizes: List[int], T: float = 0.5) -> List[float]:
    """
    sizes → list of dataset sizes.
    Returns a *normalized* weight list ∝ (1/size)^T.
    """
    inv = np.array([1 / s for s in sizes], dtype=np.float64)
    w = inv ** T
    w /= w.sum()
    return w.tolist()
