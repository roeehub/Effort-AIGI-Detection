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
from collections import defaultdict, OrderedDict
from prepare_splits import VideoInfo  # noqa , we import VideoInfo from prepare_splits.py which is 1 level up
from itertools import chain  # noqa

# --- New controllable parameter for max data loaders in memory ---
MAX_LOADERS_IN_MEMORY = 13


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


# --- New Class for managing data loaders in a memory-efficient way ---
class LazyDataLoaderManager:
    """
    Manages a dictionary of DataLoader objects, but only keeps a fixed number
    in memory at a time. Loaders are created on-demand and the oldest one is
    evicted when the memory capacity is reached.
    """

    def __init__(self, videos_by_method, config, data_config, batch_size, mode, max_loaders):
        self.videos_by_method = videos_by_method
        self.config = config
        self.data_config = data_config
        self.batch_size = batch_size
        self.mode = mode
        self.max_loaders = max_loaders
        self.active_loaders = OrderedDict()
        self.all_methods = list(self.videos_by_method.keys())

    def _create_loader(self, method):
        """Creates a DataLoader for a specific method."""
        videos = self.videos_by_method[method]
        is_train = self.mode == 'train'

        pipe = IterableWrapper(videos).shuffle() if is_train else IterableWrapper(videos)
        pipe = Mapper(pipe, lambda v: load_and_process_video(v, self.config, self.mode))
        pipe = Filter(pipe, lambda x: x is not None)

        loader = DataLoader(
            pipe,
            batch_size=self.batch_size,
            num_workers=self.data_config['dataloader_params']['num_workers'],
            collate_fn=collate_fn,
            persistent_workers=is_train,
            prefetch_factor=self.data_config['dataloader_params']['prefetch_factor'] if is_train else 1
        )
        return loader

    def __len__(self):
        """Returns the total number of methods this manager can handle."""
        return len(self.all_methods)

    def keys(self):
        """Returns all possible method keys, not just active ones."""
        return self.all_methods

    def __getitem__(self, method):
        """
        Returns a DataLoader. If not in memory, it may create it,
        potentially evicting the oldest loader if at capacity.
        """
        if method not in self.all_methods:
            raise KeyError(f"Method '{method}' is not a valid method.")

        if method in self.active_loaders:
            # Move to end to mark as recently used
            self.active_loaders.move_to_end(method)
            return self.active_loaders[method]

        if len(self.active_loaders) >= self.max_loaders:
            # Evict the first item (oldest)
            oldest_method, oldest_loader = self.active_loaders.popitem(last=False)
            print(f"INFO: Evicting data loader for method '{oldest_method}' to free up memory.")
            del oldest_loader  # Help garbage collector

        # Load the new data loader
        print(f"INFO: Activating data loader for method '{method}'.")
        new_loader = self._create_loader(method)
        self.active_loaders[method] = new_loader
        return new_loader

    def keys(self):
        """Returns all possible method keys, not just active ones."""
        return self.all_methods

    def __contains__(self, method):
        """Checks if a method is available, even if not active."""
        return method in self.all_methods


def create_method_aware_dataloaders(train_videos: list[VideoInfo], val_videos: list[VideoInfo], config: dict,
                                    data_config: dict, train_batch_size: int):
    """
    Creates separate, memory-efficient, rotating dataloader managers for
    train (per-method) and val (per-method).
    """
    # --- 1. Prepare Training Data by Method ---
    videos_by_method_train = defaultdict(list)
    for v in train_videos:
        videos_by_method_train[v.method].append(v)

    # --- 2. Create a Lazy Manager for Training Loaders ---
    train_loaders_manager = LazyDataLoaderManager(
        videos_by_method=videos_by_method_train,
        config=config,
        data_config=data_config,
        batch_size=train_batch_size,
        mode='train',
        max_loaders=MAX_LOADERS_IN_MEMORY
    )

    # --- 3. Prepare Validation Data by Method ---
    videos_by_method_val = defaultdict(list)
    for v in val_videos:
        videos_by_method_val[v.method].append(v)

    # --- 4. Create a Lazy Manager for Validation Loaders ---
    val_loaders_manager = LazyDataLoaderManager(
        videos_by_method=videos_by_method_val,
        config=config,
        data_config=data_config,
        batch_size=config['test_batchSize'],
        mode='test',
        max_loaders=MAX_LOADERS_IN_MEMORY
    )

    return train_loaders_manager, val_loaders_manager
