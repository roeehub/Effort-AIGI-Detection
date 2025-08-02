# --- dataloaders.py ---

import os
import sys


def add_relative_path(levels_up):
    path = os.path.abspath(__file__)
    for _ in range(levels_up):
        path = os.path.dirname(path)
    sys.path.append(path)


add_relative_path(1)  # go 1 levels up (..)
import torch  # noqa
from torch.utils.data import DataLoader, IterDataPipe  # noqa
from torchdata.datapipes.iter import IterableWrapper, Mapper, Filter  # noqa
import random
import fsspec  # noqa
from PIL import Image  # noqa
from torchvision import transforms as T  # noqa
import numpy as np  # noqa
import albumentations as A  # noqa
from collections import defaultdict
from prepare_splits import VideoInfo  # noqa , we import VideoInfo from prepare_splits.py which is 1 level up
from itertools import chain  # noqa


# This data augmentation function is fine, we can keep it.
def data_aug(img, landmark=None, mask=None, augmentation_seed=None):
    if augmentation_seed is not None:
        random.seed(augmentation_seed)
        np.random.seed(augmentation_seed)
    transform = A.Compose([
        A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.5), A.HueSaturationValue(p=0.3),
        A.ImageCompression(quality_lower=40, p=0.1), A.GaussNoise(p=0.1), A.MotionBlur(p=0.1),
        A.CLAHE(p=0.1), A.ChannelShuffle(p=0.1), A.Cutout(p=0.1), A.RandomGamma(p=0.3), A.GlassBlur(p=0.3),
    ])
    transformed = transform(image=np.array(img))
    return Image.fromarray(transformed['image'])


def load_and_process_video(video_info: VideoInfo, config: dict, mode: str):
    """
    This function replaces `load_video_frames_as_dataset`.
    It takes a VideoInfo object and loads the required frames from GCP.
    It's resilient to individual corrupted frames, trying all available frames
    before giving up on the video.
    It returns None on failure, so it can be filtered out.
    """
    frame_num = config['frame_num'][mode]
    resolution = config['resolution']
    all_frame_paths = list(video_info.frame_paths)

    if len(all_frame_paths) < frame_num:
        return None

    random.shuffle(all_frame_paths)

    images = []
    loaded_frame_count = 0
    for path in all_frame_paths:
        if loaded_frame_count == frame_num:
            break
        try:
            with fsspec.open(path, "rb") as f:
                img = Image.open(f).convert("RGB")
                img = img.resize((resolution, resolution), Image.BICUBIC)
            images.append(img)
            loaded_frame_count += 1
        except Exception:
            continue

    if loaded_frame_count < frame_num:
        return None

    if mode == 'train' and config['use_data_augmentation']:
        aug_seed = random.randint(0, 2 ** 32 - 1)
        images = [data_aug(img, augmentation_seed=aug_seed) for img in images]

    normalize_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=config['mean'], std=config['std'])
    ])

    image_tensors = [normalize_transform(img) for img in images]
    video_tensor = torch.stack(image_tensors, dim=0)
    label = 0 if video_info.label == 'real' else 1

    return video_tensor, label, None, None, f"{video_info.method}/{video_info.video_id}"


def collate_fn(batch):
    """
    A simplified collate_fn, matching the one in abstract_dataset.py.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return {'image': torch.empty(0), 'label': torch.empty(0), 'path': []}

    images, labels, landmarks, masks, paths = zip(*batch)

    images = torch.stack(images, dim=0)
    labels = torch.LongTensor(labels)

    data_dict = {
        'image': images,
        'label': labels,
        'landmark': None,
        'mask': None,
        'path': list(paths)
    }
    return data_dict


def create_method_aware_dataloaders(train_videos: list[VideoInfo], val_videos: list[VideoInfo], config: dict,
                                    data_config: dict, train_batch_size: int):
    """
    Creates separate dataloaders for train (per-method) and val (per-method).
    """
    # --- 1. Create Training DataLoaders (per fake method AND per real source) ---
    train_loaders = {}
    videos_by_method_train = defaultdict(list)
    for v in train_videos:
        videos_by_method_train[v.method].append(v)

    all_train_methods = data_config['methods']['use_real_sources'] + data_config['methods']['use_fake_methods']

    for method in all_train_methods:
        if method in videos_by_method_train and videos_by_method_train[method]:
            pipe = IterableWrapper(videos_by_method_train[method]).shuffle()
            pipe = Mapper(pipe, lambda v: load_and_process_video(v, config, 'train'))
            pipe = Filter(pipe, lambda x: x is not None)
            train_loaders[method] = DataLoader(
                pipe,
                batch_size=train_batch_size,
                num_workers=data_config['dataloader_params']['num_workers'],
                collate_fn=collate_fn,
                persistent_workers=True,
                prefetch_factor=data_config['dataloader_params']['prefetch_factor']
            )

    # --- 2. Create Validation DataLoaders (per method, both real and fake) ---
    val_loaders = {}
    videos_by_method_val = defaultdict(list)
    for v in val_videos:
        videos_by_method_val[v.method].append(v)

    for name, videos in videos_by_method_val.items():
        # --- FIX: Check if the list of videos is empty BEFORE creating the loader ---
        if not videos: continue
        pipe = IterableWrapper(videos)  # No shuffle for validation
        pipe = Mapper(pipe, lambda v: load_and_process_video(v, config, 'test'))
        pipe = Filter(pipe, lambda x: x is not None)
        val_loaders[name] = DataLoader(
            pipe,
            batch_size=config['test_batchSize'],
            num_workers=data_config['dataloader_params']['num_workers'],
            collate_fn=collate_fn,
            persistent_workers=True,
            prefetch_factor=data_config['dataloader_params']['prefetch_factor']
        )

    return train_loaders, val_loaders
