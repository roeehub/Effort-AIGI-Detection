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
from torchdata.datapipes.iter import Zipper  # noqa
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
import cv2  # noqa


def _map_video(video_info, config, mode):
    return load_and_process_video(video_info, config, mode)


def _not_none(x):
    return x is not None


def _flatmap_frame_batch(batch_of_paths, config, mode):
    # flatmap expects an iterable; load_and_process_frame_batch yields/iterates
    return load_and_process_frame_batch(batch_of_paths, config, mode)


# ==============================================================================
# --- NEW: FLEXIBLE AUGMENTATION PIPELINES (albumentations==0.4.6 compatible) ---
# ==============================================================================

# --- Pipeline to aggressively degrade image quality ---
degrade_quality_pipeline = A.Compose([
    A.OneOf([
        A.ImageCompression(quality_lower=40, quality_upper=70, p=0.8),
        A.GaussianBlur(blur_limit=(5, 11), p=0.6),
        A.GaussNoise(var_limit=(20.0, 80.0), p=0.4),
    ], p=1.0)
])

# --- Pipeline to enhance image quality ---
enhance_quality_pipeline = A.Compose([
    A.IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.9),
])

# --- Pipeline to simulate social media compression ---
social_media_pipeline = A.Compose([
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.Downscale(scale_min=0.5, scale_max=0.75, interpolation=cv2.INTER_AREA, p=0.8),
    A.ImageCompression(quality_lower=30, quality_upper=60, p=1.0),
], p=1.0)


def create_surgical_augmentation_pipeline(
        config: dict,
        frame_properties: dict
) -> A.Compose:
    """
    Dynamically constructs an Albumentations pipeline based on a configuration
    and specific frame properties.
    """
    transforms_list = []

    # 1. Base & Geometric
    transforms_list.append(A.HorizontalFlip(p=0.5))
    if config.get('use_geometric', False):
        transforms_list.append(A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.12, rotate_limit=7,
            interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, p=0.7
        ))

    # Added back to prevent the model from "forgetting" its robustness to color/lighting.
    # We use mild parameters and moderate probability.
    if config.get('use_color_jitter', False):
        transforms_list.extend([
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5
            )
        ])

    # 2. Surgical Sharpness Adjustment
    sharpness_bucket = frame_properties.get('sharpness_bucket')
    chance_for_sharpness_adjustment = config.get('sharpness_adjust_prob', 0.5)
    if random.random() < chance_for_sharpness_adjustment:
        if sharpness_bucket == 'q4':
            transforms_list.append(degrade_quality_pipeline)
        elif sharpness_bucket == 'q1':
            transforms_list.append(enhance_quality_pipeline)

    # 3. Advanced Noise & Artifact Simulation
    if config.get('use_advanced_noise', False):
        transforms_list.append(A.OneOf([
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            social_media_pipeline
        ], p=config.get('advanced_noise_prob', 0.6)))

    # 4. Occlusion
    if config.get('use_occlusion', False):
        transforms_list.append(A.Cutout(
            num_holes=8, max_h_size=24, max_w_size=24, fill_value=0,
            p=config.get('occlusion_prob', 0.5)
        ))

    return A.Compose(transforms_list)


def create_general_augmentation_pipeline(config: dict) -> A.Compose:
    """
    Creates a robust, general-purpose augmentation pipeline for strategies
    that do not have access to frame-level properties.
    """
    transforms_list = [
        A.HorizontalFlip(p=0.5),
    ]
    if config.get('use_geometric', False):
        transforms_list.append(A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.12, rotate_limit=7,
            interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, p=0.7
        ))

    # --- NEW: Optional Color Jitter (also added here for consistency) ---
    if config.get('use_color_jitter', False):
        transforms_list.extend([
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5
            )
        ])

    # General quality variations
    transforms_list.append(A.OneOf([
        A.ImageCompression(quality_lower=50, quality_upper=80, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    ], p=0.8))  # Apply one of the quality transforms 80% of the time

    if config.get('use_occlusion', False):
        transforms_list.append(A.Cutout(
            num_holes=8, max_h_size=24, max_w_size=24, fill_value=0,
            p=config.get('occlusion_prob', 0.5)
        ))
    return A.Compose(transforms_list)


# ======================================================================================
# === NEW DataPipes and helpers for the property-balanced strategy ===
# ======================================================================================


class CustomRoundRobinDataPipe(IterDataPipe):
    def __init__(self, *datapipes):
        super().__init__()
        self.datapipes = datapipes

    def __iter__(self):
        iterators = [iter(dp) for dp in self.datapipes]
        while True:
            for it in iterators:
                try:
                    yield next(it)
                except StopIteration:
                    return


class CustomSampleMultiplexerDataPipe(IterDataPipe):
    def __init__(self, datapipes, weights):
        super().__init__()
        self.datapipes = datapipes
        self.weights = weights

    def __iter__(self):
        iterators = [iter(dp) for dp in self.datapipes]
        indices = np.arange(len(self.datapipes))
        while True:
            chosen_index = np.random.choice(indices, p=self.weights)
            try:
                yield next(iterators[chosen_index])
            except StopIteration:
                return


def _consolidate_small_buckets(frames_by_bucket: dict, min_size: int, label_name: str) -> dict:
    consolidated_buckets = {}
    misc_frames = []
    for bucket_name, frames in frames_by_bucket.items():
        if len(frames) < min_size:
            misc_frames.extend(frames)
        else:
            consolidated_buckets[bucket_name] = frames
    if misc_frames:
        consolidated_buckets[f'misc_{label_name}'] = misc_frames
    return consolidated_buckets


def _create_master_stream_for_label(consolidated_buckets: dict) -> IterDataPipe:
    if not consolidated_buckets: return None
    bucket_datapipes = [IterableWrapper(frames).cycle() for frames in consolidated_buckets.values()]
    return CustomRoundRobinDataPipe(*bucket_datapipes)


def _build_clip_to_frames_lookup(all_frames: list[dict]) -> dict[str, list[dict]]:
    """Builds a lookup table mapping clip_id to a list of its frame dictionaries."""
    lookup = defaultdict(list)
    for frame in all_frames:
        # NOTE: Using clip_id for pairing, as per the verified script.
        # Ensure 'clip_id' exists in your frame_properties.parquet file.
        lookup[frame['clip_id']].append(frame)
    return lookup


class MateFinderDataPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe, lookup: dict):
        super().__init__()
        self.source_datapipe = source_datapipe
        self.lookup = lookup

    def __iter__(self):
        for anchor_frame in self.source_datapipe:
            yield anchor_frame  # Yield anchor first

            clip_id = anchor_frame.get('clip_id')
            if clip_id is None:
                print(
                    f"WARNING: [MateFinder] Frame missing 'clip_id'. Path: {anchor_frame.get('path')}. Using self as mate.")
                yield anchor_frame  # Fallback: use self as mate
                continue

            possible_mates = self.lookup.get(clip_id, [])
            mates_pool = [m for m in possible_mates if m['path'] != anchor_frame['path']]

            if not mates_pool:
                # This is a valid case for clips with only one frame.
                # Use the anchor frame itself as the mate to maintain batch structure.
                print(f"INFO: [MateFinder] No unique mate found for clip_id '{clip_id}'. Using self as mate.")
                yield anchor_frame
            else:
                yield random.choice(mates_pool)


# ======================================================================================

# --- New controllable parameter for max data loaders in memory ---
MAX_LOADERS_IN_MEMORY = 2

# --- Base augmentations for general variety (always applied) ---
transform_base = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.HueSaturationValue(p=0.1),
])


# # --- Pipeline to aggressively degrade image quality ---
# degrade_quality_pipeline = A.Compose([
#     A.OneOf([
#         A.ImageCompression(quality_lower=40, quality_upper=70, p=0.8),
#         A.GaussianBlur(blur_limit=(5, 11), p=0.6),
#         A.GaussNoise(var_limit=(20.0, 80.0), p=0.4),
#     ], p=1.0)  # Always apply one of the degradations from this list
# ])
#
# # --- Pipeline to enhance image quality ---
# enhance_quality_pipeline = A.Compose([
#     # This is an effective "unsharp mask"
#     A.IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.9),
# ])


def surgical_data_aug(img: Image.Image, frame_properties: dict) -> Image.Image:
    """
    Applies augmentations dynamically based on frame properties to create "counter-examples"
    and break trivial correlations like "blurry = fake" or "sharp = real".
    """
    img_np = np.array(img)
    chance_for_sharpness_adjustment = 0.5  # 50% chance to adjust sharpness-related properties

    # 1. Always apply base transformations for general robustness
    img_np = transform_base(image=img_np)['image']

    sharpness_bucket = frame_properties.get('sharpness_bucket')

    # 2. Apply surgical, property-based transformations
    if sharpness_bucket == 'q4':  # This is a very sharp image
        # Degrade it with high probability to teach the model that sharp images can also be low quality
        if random.random() < chance_for_sharpness_adjustment:
            img_np = degrade_quality_pipeline(image=img_np)['image']

    elif sharpness_bucket == 'q1':  # This is a very blurry image
        # Enhance it with high probability to teach the model that blurry images can be sharpened
        if random.random() < chance_for_sharpness_adjustment:
            img_np = enhance_quality_pipeline(image=img_np)['image']

    # For middle buckets (q2, q3), apply a mix with lower probability to create more variety
    elif sharpness_bucket in ['q2', 'q3']:
        if random.random() < chance_for_sharpness_adjustment / 2:
            if random.random() < 0.5:
                img_np = degrade_quality_pipeline(image=img_np)['image']
            else:
                img_np = enhance_quality_pipeline(image=img_np)['image']

    return Image.fromarray(img_np)


def data_aug_v2(img, config: dict, augmentation_seed=None):
    """
    A general-purpose quality augmentation pipeline that is now configurable.
    """
    if augmentation_seed is not None:
        random.seed(augmentation_seed)
        np.random.seed(augmentation_seed)

    aug_params = config.get('augmentation_params', {})
    pipeline = create_general_augmentation_pipeline(aug_params)

    transformed = pipeline(image=np.array(img))
    return Image.fromarray(transformed['image'])


def load_and_process_frame_batch(frame_info_batch: list[tuple], config: dict, mode: str):
    """
    Loads and processes a BATCH of frames in parallel using a thread pool.
    Used by the 'frame_level' strategy.
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
                # This strategy doesn't have frame properties, so it uses the general aug
                aug_seed = random.randint(0, 2 ** 32 - 1)
                img = data_aug_v2(img, config, augmentation_seed=aug_seed)

            image_tensor = normalize_transform(img)
            return image_tensor, label_id, None, None, frame_path
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=24) as executor:
        results = list(executor.map(_load_single, frame_info_batch))

    for result in results:
        if result is not None:
            yield result


def load_and_process_property_batch(frame_dict_batch: list[dict], config: dict, mode: str):
    """
    Parallel frame loader for the property-balanced strategy.
    Now uses the new configurable, surgical augmentation pipeline.
    """
    resolution = config['resolution']
    use_aug = mode == 'train' and config['use_data_augmentation']
    aug_params = config.get('augmentation_params', {})  # Get the new config dict

    normalize_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=config['mean'], std=config['std'])
    ])

    def _load_single(frame_dict):
        frame_path = frame_dict['path']
        label_id = frame_dict['label_id']
        try:
            with fsspec.open(frame_path, "rb") as f:
                img = Image.open(f).convert("RGB")
                img = img.resize((resolution, resolution), Image.BICUBIC)

            if use_aug:
                # Dynamically create and apply the surgical pipeline for each frame
                pipeline = create_surgical_augmentation_pipeline(aug_params, frame_dict)
                img_np = np.array(img)
                augmented_img = pipeline(image=img_np)['image']
                img = Image.fromarray(augmented_img)

            image_tensor = normalize_transform(img)
            return image_tensor, label_id, None, None, frame_path
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=24) as executor:
        results = list(executor.map(_load_single, frame_dict_batch))

    for result in results:
        if result is not None:
            yield result


def load_and_process_video(video_info: VideoInfo, config: dict, mode: str, frame_count_override: int = None):
    """
    Loads frames for a video in parallel. Used by 'video_level' and 'per_method' strategies.
    """
    if frame_count_override is not None:
        frame_num = frame_count_override
    else:
        dl_params = config.get('dataloader_params', {})
        frame_num = dl_params.get('frames_per_video', 8)

    resolution = config['resolution']
    all_frame_paths = list(video_info.frame_paths)

    if len(all_frame_paths) < frame_num:
        # If a video doesn't have enough frames to create a tensor of the standard
        # size for this batch, it must be skipped. Returning None achieves this
        # as the Filter datapipe will discard it.
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
            return None

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
        # These strategies don't have properties, so they use the general augmentation
        images = [data_aug_v2(img, config, augmentation_seed=aug_seed) for img in images]

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

    return DataLoader(
        pipe,
        batch_size=gpu_batch_size,  # This is the final batch size for the GPU
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=data_config['dataloader_params']['prefetch_factor']
    )


def _create_property_balanced_loader(all_train_frames: list[dict], config: dict, data_config: dict):
    """
    Creates the property-balanced dataloader using the Anchor-Mate strategy.
    This implementation is based on the verified pipeline.
    """
    dl_params = data_config['dataloader_params']
    BATCH_SIZE = dl_params['frames_per_batch']
    NUM_WORKERS = dl_params['num_workers']
    MIN_BUCKET_SIZE = BATCH_SIZE  # A sensible minimum to avoid tiny cycled iterators

    # 1. Build the lookup table for finding mates efficiently
    clip_lookup = _build_clip_to_frames_lookup(all_train_frames)
    print(f"Built clip_id lookup table with {len(clip_lookup)} unique clips.")

    # 2. Segregate all frames by label and then by property bucket
    real_frames_by_bucket = defaultdict(list)
    fake_frames_by_bucket = defaultdict(list)
    for frame_info in all_train_frames:
        bucket = frame_info['property_bucket']
        if frame_info['label_id'] == 0:
            real_frames_by_bucket[bucket].append(frame_info)
        else:
            fake_frames_by_bucket[bucket].append(frame_info)

    # 3. Consolidate small buckets to prevent overhead
    consolidated_real = _consolidate_small_buckets(real_frames_by_bucket, MIN_BUCKET_SIZE, 'real')
    consolidated_fake = _consolidate_small_buckets(fake_frames_by_bucket, MIN_BUCKET_SIZE, 'fake')

    # 4. Create master streams that cycle through property buckets for each label
    real_master_stream = _create_master_stream_for_label(consolidated_real)
    fake_master_stream = _create_master_stream_for_label(consolidated_fake)
    if real_master_stream is None or fake_master_stream is None:
        raise ValueError("Cannot create dataloader: data for one or both labels is missing.")

    # 5. Probabilistically sample from master streams to create a 50/50 balanced "anchor" stream
    anchor_pipe = CustomSampleMultiplexerDataPipe([real_master_stream, fake_master_stream], [0.5, 0.5])

    # 6. Add sharding (for DDP) and shuffling
    if NUM_WORKERS > 0:
        anchor_pipe = anchor_pipe.sharding_filter()
    anchor_pipe = anchor_pipe.shuffle(buffer_size=10000)

    # 7. Use the MateFinderDataPipe to double the stream with paired frames
    # This pipe takes one anchor and yields (anchor, mate)
    combined_pipe = MateFinderDataPipe(anchor_pipe, clip_lookup)

    # 8. Batch the frame dictionaries FOR I/O EFFICIENCY, then use the parallel loader
    # The flatmap will yield individual processed frames.
    # We create a larger IO batch to send to each worker for efficiency.
    io_batch_size = BATCH_SIZE * 4  # A good default, e.g., 256
    combined_pipe = combined_pipe.batch(io_batch_size).flatmap(
        partial(load_and_process_property_batch, config=config, mode='train')
    )

    return DataLoader(
        combined_pipe,
        batch_size=BATCH_SIZE,  # Let the DataLoader create the final GPU batch
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,  # Now this collate_fn will receive a list of tuples, which is correct
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=dl_params['prefetch_factor']
    )


def create_ood_loader(ood_videos: list[VideoInfo], config: dict, data_config: dict):
    """
    Factory function to create the OOD dataloader.
    """
    print("--- Creating OOD dataloader ---")
    if not ood_videos:
        print("No OOD videos found, returning None for the OOD loader.")
        return None

    ood_videos_by_method = defaultdict(list)
    for v in ood_videos:
        ood_videos_by_method[v.method].append(v)

    ood_map_fn = partial(load_and_process_video, config=config, mode='test', frame_count_override=4)

    ood_loader = LazyDataLoaderManager(
        videos_by_method=ood_videos_by_method, config=config, data_config=data_config,
        batch_size=config['test_batchSize'], mode='test', max_loaders=MAX_LOADERS_IN_MEMORY,
        map_fn=ood_map_fn
    )
    return ood_loader


def create_dataloaders(train_data: List, val_videos: List[VideoInfo], config: dict,
                       data_config: dict):
    """
    Factory function to create dataloaders based on the specified strategy.
    """
    strategy = data_config.get('dataloader_params', {}).get('strategy', 'per_method')
    print(f"--- Creating dataloaders with strategy: '{strategy}' ---")

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

    # Create the Validation Loader (always per-method for consistent evaluation)
    val_videos_by_method = defaultdict(list)
    for v in val_videos:
        val_videos_by_method[v.method].append(v)

    val_loader = LazyDataLoaderManager(
        videos_by_method=val_videos_by_method, config=config, data_config=data_config,
        batch_size=config['test_batchSize'], mode='test', max_loaders=MAX_LOADERS_IN_MEMORY
    )
    return train_loader, val_loader
