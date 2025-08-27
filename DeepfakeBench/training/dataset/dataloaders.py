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
from torchdata.datapipes.iter import Multiplexer  # noqa
import random
import fsspec  # noqa
from PIL import Image  # noqa
from torchvision import transforms as T  # noqa
import numpy as np  # noqa
import albumentations as A  # noqa
from collections import defaultdict, OrderedDict
from prepare_splits import VideoInfo  # noqa , we import VideoInfo from prepare_splits.py which is 1 level up
from itertools import chain  # noqa
from concurrent.futures import ThreadPoolExecutor
# --- helper callables for DataPipes (must be top-level & picklable) ---
from functools import partial
from typing import List, Dict


def _map_video(video_info, config, mode):
    return load_and_process_video(video_info, config, mode)


def _not_none(x):
    return x is not None


def _flatmap_frame_batch(batch_of_paths, config, mode):
    # flatmap expects an iterable; load_and_process_frame_batch yields/iterates
    return load_and_process_frame_batch(batch_of_paths, config, mode)


# --- New controllable parameter for max data loaders in memory ---
MAX_LOADERS_IN_MEMORY = 2


def data_aug_v2(img, augmentation_seed=None):
    """
    Applies a two-stage augmentation pipeline compatible with albumentations==0.4.6.
    1. Base augmentations for general variety (color, orientation).
    2. A targeted "Quality Attack" using multi-level degradation to neutralize sharpness as a trivial cue.
    """
    if augmentation_seed is not None:
        random.seed(augmentation_seed)
        np.random.seed(augmentation_seed)

    # Stage 1: Base augmentations for general variety.
    # This part remains the same.
    transform_base = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.2),
    ])

    # Stage 2: The "Quality Attack" adapted for v0.4.6
    # We use a mix of degradations and the NoOp() trick.
    transform_quality = A.Compose([
        A.OneOf([
            # Heavy Degradation Options
            A.ImageCompression(quality_lower=50, quality_upper=80, p=0.5),
            A.MotionBlur(blur_limit=9, p=0.2),
            A.GaussNoise(var_limit=(20.0, 60.0), p=0.2),

            # The "Do Nothing" Option
            A.NoOp(p=0.1)  # Gives a chance for the image to pass through untouched
        ], p=0.9)  # "Safety Valve": 90% of images will enter this lottery.
    ])

    img_np = np.array(img)
    transformed_base = transform_base(image=img_np)
    transformed_quality = transform_quality(image=transformed_base['image'])

    return Image.fromarray(transformed_quality['image'])


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


def load_and_process_frame(frame_info: tuple, config: dict, mode: str):
    """
    Loads and processes a single frame.
    Input: A tuple (frame_path, label_id)
    Returns None on failure.
    """
    frame_path, label_id = frame_info
    resolution = config['resolution']

    try:
        with fsspec.open(frame_path, "rb") as f:
            img = Image.open(f).convert("RGB")
            img = img.resize((resolution, resolution), Image.BICUBIC)
    except Exception:
        return None  # Fail silently for a single frame

    if mode == 'train' and config['use_data_augmentation']:
        aug_seed = random.randint(0, 2 ** 32 - 1)
        img = data_aug_v2(img, augmentation_seed=aug_seed)

    normalize_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=config['mean'], std=config['std'])
    ])

    image_tensor = normalize_transform(img)
    # The output format must match collate_fn's expectation
    return image_tensor, label_id, None, None, frame_path


def load_and_process_frame_batch(frame_info_batch: list[tuple], config: dict, mode: str):
    """
    Loads and processes a BATCH of frames in parallel using a thread pool.
    This is much more efficient for cloud storage as it overlaps I/O latency.
    """
    resolution = config['resolution']
    use_aug = mode == 'train' and config['use_data_augmentation']

    normalize_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=config['mean'], std=config['std'])
    ])

    def _load_single(frame_info):
        """The work for a single thread."""
        frame_path, label_id = frame_info
        try:
            with fsspec.open(frame_path, "rb") as f:
                img = Image.open(f).convert("RGB")
                img = img.resize((resolution, resolution), Image.BICUBIC)

            if use_aug:
                # Give each augmentation its own seed for variety within a batch
                aug_seed = random.randint(0, 2 ** 32 - 1)
                img = data_aug_v2(img, augmentation_seed=aug_seed)

            image_tensor = normalize_transform(img)
            # The output format must match collate_fn's expectation
            return image_tensor, label_id, None, None, frame_path
        except Exception:
            return None  # Fail silently for a single frame

    # Use a ThreadPoolExecutor to fetch multiple frames concurrently within this single function call.
    # The number of threads here (e.g., 8-16) is key. It allows one dataloader worker
    # to perform multiple GCS requests in parallel.
    with ThreadPoolExecutor(max_workers=24) as executor:
        results = list(executor.map(_load_single, frame_info_batch))

    # The function should yield individual samples, not a list
    for result in results:
        if result is not None:
            yield result


def load_and_process_video(video_info: VideoInfo, config: dict, mode: str, frame_count_override: int = None):
    """
    This function replaces `load_video_frames_as_dataset`.
    It takes a VideoInfo object and loads the required frames from GCP in parallel.
    It's resilient to individual corrupted frames.
    It returns None on failure, so it can be filtered out.
    """
    if frame_count_override is not None:
        frame_num = frame_count_override
    else:
        dl_params = config.get('dataloader_params', {})
        frame_num = dl_params.get('frames_per_video', 8)

    resolution = config['resolution']
    all_frame_paths = list(video_info.frame_paths)

    if len(all_frame_paths) < frame_num:
        # For OOD, we can be more lenient and take what we can get.
        if frame_count_override is not None and len(all_frame_paths) > 0:
            frame_num = len(all_frame_paths)
        else:
            return None

    random.shuffle(all_frame_paths)
    selected_paths = all_frame_paths[:frame_num]

    def _load_single_frame(path):
        """Helper function for a single thread to load one frame."""
        try:
            with fsspec.open(path, "rb") as f:
                img = Image.open(f).convert("RGB")
                img = img.resize((resolution, resolution), Image.BICUBIC)
            return img
        except Exception:
            return None  # Fail silently for one frame

    # Use a ThreadPoolExecutor to fetch all frames for this video in parallel
    with ThreadPoolExecutor(max_workers=16) as executor:
        # map() maintains the order of the input paths
        image_results = list(executor.map(_load_single_frame, selected_paths))

    # Filter out any frames that failed to load
    images = [img for img in image_results if img is not None]

    if len(images) < frame_num:
        # Not enough frames could be loaded for this video
        return None

    if mode == 'train' and config['use_data_augmentation']:
        aug_seed = random.randint(0, 2 ** 32 - 1)
        images = [data_aug_v2(img, augmentation_seed=aug_seed) for img in images]

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

    def __init__(self, videos_by_method, config, data_config, batch_size, mode, max_loaders, **kwargs):
        self.videos_by_method = videos_by_method
        self.config = config
        self.data_config = data_config
        self.batch_size = batch_size
        self.mode = mode
        self.max_loaders = max_loaders
        self.active_loaders = OrderedDict()
        self.all_methods = list(self.videos_by_method.keys())
        # Accept a custom mapping function for flexibility (e.g., for OOD)
        self.map_fn = kwargs.get('map_fn', None)

        dl_params = self.data_config.get('dataloader_params', {})
        if self.mode == 'train':
            # For training, use the standard worker/prefetch config
            self.num_workers = dl_params.get('num_workers', 2)
            self.prefetch_factor = dl_params.get('prefetch_factor', 1)
        else:  # For 'test' or validation mode
            self.num_workers = 6
            self.prefetch_factor = 4

    def _create_loader(self, method):
        """Creates a DataLoader for a specific method."""
        videos = self.videos_by_method[method]
        is_train = self.mode == 'train'

        pipe = IterableWrapper(videos).shuffle() if is_train else IterableWrapper(videos)

        # Use the custom map_fn if provided, otherwise default to the standard one.
        map_function = self.map_fn
        if map_function is None:
            map_function = partial(_map_video, config=self.config, mode=self.mode)

        pipe = Mapper(pipe, map_function)
        pipe = Filter(pipe, _not_none)

        loader = DataLoader(
            pipe,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=is_train and self.num_workers > 0,  # Fix for num_workers=0 case
            prefetch_factor=self.prefetch_factor
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


def _create_per_method_loaders(train_videos, val_videos, config, data_config):
    """Creates the dataloaders for the 'per_method' strategy."""
    # This is the original logic, refactored into a helper.
    train_videos_by_method = defaultdict(list)
    for v in train_videos:
        train_videos_by_method[v.method].append(v)

    train_batch_size = data_config['dataloader_params']['batch_size'] // 2
    if train_batch_size == 0:
        train_batch_size = 1

    train_loaders_manager = LazyDataLoaderManager(
        videos_by_method=train_videos_by_method, config=config,
        data_config=data_config, batch_size=train_batch_size, mode='train',
        max_loaders=MAX_LOADERS_IN_MEMORY
    )

    val_videos_by_method = defaultdict(list)
    for v in val_videos:
        val_videos_by_method[v.method].append(v)

    val_loaders_manager = LazyDataLoaderManager(
        videos_by_method=val_videos_by_method, config=config,
        data_config=data_config, batch_size=config['test_batchSize'], mode='test',
        max_loaders=MAX_LOADERS_IN_MEMORY
    )
    return train_loaders_manager, val_loaders_manager


def _create_random_video_loader(train_videos, config, data_config):
    """Creates a single dataloader for the 'video_level' strategy."""
    pipe = IterableWrapper(train_videos).shuffle()
    pipe = Mapper(pipe, partial(_map_video, config=config, mode='train'))
    pipe = Filter(pipe, _not_none)

    dl_params = data_config.get('dataloader_params', {})
    batch_size = dl_params.get('videos_per_batch', 8)

    return DataLoader(
        pipe,
        batch_size=batch_size,
        num_workers=data_config['dataloader_params']['num_workers'],
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=data_config['dataloader_params']['prefetch_factor']
    )


def _create_random_frame_loader(train_videos: list[VideoInfo], config: dict, data_config: dict):
    """
    Creates a single, highly efficient dataloader for the 'frame_level' strategy.
    This implementation batches I/O requests to overcome cloud storage latency.
    """
    dl_params = data_config.get('dataloader_params', {})
    gpu_batch_size = dl_params.get('frames_per_batch', 64)
    num_workers = dl_params.get('num_workers', 16)

    # 1. Start with an iterable of the training videos.
    pipe = IterableWrapper(train_videos)

    # 2. Lazily flatten into a stream of frame tuples (path, label_id).
    def video_to_frame_tuples(video: VideoInfo):
        label_id = 0 if video.label == 'real' else 1
        for frame_path in video.frame_paths:
            yield (frame_path, label_id)

    pipe = pipe.flatmap(video_to_frame_tuples)

    # 3. Shuffle the stream of paths using a larger buffer for better randomness.
    pipe = pipe.shuffle(buffer_size=50000)

    # 4. *** KEY CHANGE ***: Batch the PATHS before mapping to the loading function.
    # We create a batch of paths to be processed by a single worker.
    # A larger io_batch_size is better. It should be a multiple of your GPU batch size.
    io_batch_size = gpu_batch_size * 4
    pipe = pipe.batch(io_batch_size)

    # 5. Map the new BATCHED loading function. It will get a list of paths.
    # `flatmap` is used because our new function yields multiple processed frames.
    pipe = pipe.flatmap(partial(_flatmap_frame_batch, config=config, mode='train'))

    # The filter is no longer needed here, as the batch loader handles it.

    return DataLoader(
        pipe,
        batch_size=gpu_batch_size,  # This is the final batch size for the GPU
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=data_config['dataloader_params']['prefetch_factor']
    )


def _create_property_balanced_loader(train_frames: List[Dict], config: dict, data_config: dict):
    """
    Implements the "Anchor-Mate" sampling strategy.
    1. Samples 1 "anchor" frame from each of the 32 property buckets.
    2. For each anchor, finds a random "mate" frame from the same video.
    3. This creates a batch that is perfectly balanced by anchor and regularized by the mate.
    """
    dl_params = data_config.get('dataloader_params', {})
    gpu_batch_size = dl_params.get('frames_per_batch', 64)
    num_workers = dl_params.get('num_workers', 16)

    # This strategy requires a batch size that is a multiple of the number of buckets (32)
    if gpu_batch_size % 32 != 0:
        raise ValueError(f"For Anchor-Mate sampling, `frames_per_batch` ({gpu_batch_size}) must be a multiple of 32.")

    # 1. Pre-computation: Organize frames for fast lookups
    # Bucket dictionary for sampling anchors
    anchor_buckets = defaultdict(list)
    for frame_dict in train_frames:
        anchor_buckets[frame_dict['property_bucket']].append(frame_dict)

    # Video dictionary for finding mates
    frames_by_video = defaultdict(list)
    for frame_dict in train_frames:
        frames_by_video[frame_dict['video_id']].append(frame_dict)

    num_buckets = len(anchor_buckets)
    log_msg = f"Anchor-Mate loader using {num_buckets} buckets. "
    log_msg += f"Smallest anchor bucket has {min(len(v) for v in anchor_buckets.values())} frames."
    print(log_msg)

    # 2. Create a datapipe for each anchor bucket
    anchor_pipes = []
    for bucket_key, frames in anchor_buckets.items():
        pipe = IterableWrapper(frames).cycle().shuffle()
        anchor_pipes.append(pipe)

    # 3. Use Multiplexer to sample one anchor from each pipe
    samples_per_bucket = gpu_batch_size // (num_buckets * 2)  # e.g., 64 / (32*2) = 1
    if samples_per_bucket < 1:
        samples_per_bucket = 1  # Fallback for smaller batches, though not recommended

    # This pipe yields tuples of anchor frames, e.g., (anchor1, anchor2, ..., anchor32)
    combined_pipe = Multiplexer(*anchor_pipes, n_instances_per_iter=samples_per_bucket)

    # 4. Define the Anchor-Mate pairing function
    def find_mates_and_flatten(anchor_tuple):
        batch_frames = []
        for anchor_frame in anchor_tuple:
            batch_frames.append(anchor_frame)  # Add the anchor

            video_id = anchor_frame['video_id']
            siblings = frames_by_video[video_id]

            # Find a mate that is not the anchor itself
            if len(siblings) > 1:
                mate_frame = random.choice(siblings)
                while mate_frame['path'] == anchor_frame['path']:
                    mate_frame = random.choice(siblings)
            else:
                mate_frame = anchor_frame

            batch_frames.append(mate_frame)  # Add the mate

        random.shuffle(batch_frames)  # Shuffle to mix anchors and mates
        return batch_frames

    # 5. Build the final pipeline
    # Map the pairing function to the stream of anchor tuples
    final_pipe = combined_pipe.map(find_mates_and_flatten)
    # Flatten the stream of lists into a stream of individual frames
    final_pipe = final_pipe.flatmap(lambda x: x)
    # Map the function that loads and processes a single frame
    final_pipe = final_pipe.map(lambda f: load_and_process_frame((f['path'], f['label_id']), config, 'train'))
    final_pipe = final_pipe.filter(_not_none)

    return DataLoader(
        final_pipe,
        batch_size=gpu_batch_size,  # DataLoader now handles the final batching
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=data_config['dataloader_params']['prefetch_factor']
    )


def create_ood_loader(ood_videos: list[VideoInfo], config: dict, data_config: dict):
    """
    Factory function to create the OOD dataloader.
    It is always per-method for detailed analysis and uses a fixed number of frames.
    """
    print("--- Creating OOD dataloader ---")
    if not ood_videos:
        print("No OOD videos found, returning None for the OOD loader.")
        return None

    ood_videos_by_method = defaultdict(list)
    for v in ood_videos:
        ood_videos_by_method[v.method].append(v)

    # Create a special mapping function that enforces 4 frames per video
    ood_map_fn = partial(
        load_and_process_video,
        config=config,
        mode='test',  # No augmentations
        frame_count_override=4
    )

    ood_loader = LazyDataLoaderManager(
        videos_by_method=ood_videos_by_method,
        config=config,
        data_config=data_config,
        batch_size=config['test_batchSize'],
        mode='test',
        max_loaders=MAX_LOADERS_IN_MEMORY,
        map_fn=ood_map_fn  # Pass the custom mapping function
    )

    return ood_loader


def create_dataloaders(train_data: List, val_videos: List[VideoInfo], config: dict,
                       data_config: dict):
    """
    Factory function to create dataloaders based on the specified strategy.
    The validation loader is ALWAYS per-method for consistent evaluation.
    The `train_data` can be a list of VideoInfo or a list of frame dictionaries.
    """
    strategy = data_config.get('dataloader_params', {}).get('strategy', 'per_method')
    print(f"--- Creating dataloaders with strategy: '{strategy}' ---")

    # --- 1. Create Training Loader based on strategy ---
    if strategy == 'per_method':
        # The per_method helper returns both train and val loaders
        train_loader, val_loader = _create_per_method_loaders(train_data, val_videos, config, data_config)
        return train_loader, val_loader

    elif strategy == 'video_level':
        train_loader = _create_random_video_loader(train_data, config, data_config)

    elif strategy == 'frame_level':
        train_loader = _create_random_frame_loader(train_data, config, data_config)

    elif strategy == 'property_balancing':
        # This new strategy expects a list of frame dictionaries as `train_data`
        train_loader = _create_property_balanced_loader(train_data, config, data_config)

    else:
        raise ValueError(f"Unknown dataloader strategy: '{strategy}'")

    # --- 2. Create the Validation Loader (always per-method) ---
    val_videos_by_method = defaultdict(list)
    for v in val_videos:
        val_videos_by_method[v.method].append(v)

    val_loader = LazyDataLoaderManager(
        videos_by_method=val_videos_by_method,
        config=config,
        data_config=data_config,
        batch_size=config['test_batchSize'],
        mode='test',
        max_loaders=MAX_LOADERS_IN_MEMORY
    )

    return train_loader, val_loader
