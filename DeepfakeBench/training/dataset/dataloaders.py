import math
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
from albumentations.core.transforms_interface import ImageOnlyTransform  # noqa
from albumentations.core.transforms_interface import BasicTransform  # noqa

# ==============================================================================
# --- Augmentation Imports from Refactored Module ---
# ==============================================================================
# All augmentation pipelines are now defined in data/augmentations/
# Import them from there to maintain backward compatibility
from data.augmentations import (
    # Custom transforms
    CustomUnsharpMask,
    NoOp,
    # Registry function
    get_pipeline,
)
from data.augmentations.pipelines import (
    # Pre-instantiated pipelines (for backward compatibility)
    revised_augmentation_pipeline_legacy,
    augmentation_pipeline_v4,
    augmentation_pipeline_v5,
    AUG_PIPELINE_V6_SIMULATOR,
    AUG_PIPELINE_V4_GENERALIST,
    AUG_PIPELINE_PURIST,
    AUG_PIPELINE_V3_MILD,
    degrade_quality_pipeline,
    enhance_quality_pipeline,
    social_media_pipeline,
    # Callable functions
    apply_augmentation_v6,
    apply_augmentation_v7,
    # Factory functions
    create_surgical_augmentation_pipeline,
    create_general_augmentation_pipeline,
)

# ==============================================================================
# --- Batching Module Imports for Backward Compatibility ---
# ==============================================================================
# The batching strategies, DataPipes, and loader functions have been refactored
# into data/batching/ module. Import them here for backward compatibility.
#
# New modular structure:
#   - data/batching/base.py           - BatchingStrategy interface
#   - data/batching/datapipes.py      - CustomRoundRobinDataPipe, CustomSampleMultiplexerDataPipe, MateFinderDataPipe
#   - data/batching/loaders.py        - load_and_process_*, collate_fn functions
#   - data/batching/per_method.py     - PerMethodStrategy, LazyDataLoaderManager
#   - data/batching/video_level.py    - VideoLevelStrategy
#   - data/batching/frame_level.py    - FrameLevelStrategy
#   - data/batching/property_balanced.py - PropertyBalancedStrategy
#   - data/batching/factory.py        - get_batching_strategy(), create_dataloaders()
#
# To use the new system:
#   from data.batching import get_batching_strategy, create_dataloaders
#   strategy = get_batching_strategy('property_balancing', config, data_config)
#   train_loader = strategy.create_train_loader(train_data)
#
# The original code in this file is preserved for backward compatibility.
# ==============================================================================

def _map_video(video_info, config, mode):
    return load_and_process_video(video_info, config, mode)


def _map_video_detailed(video_info, config, mode):
    return load_and_process_video_detailed(video_info, config, mode)


def _not_none(x):
    return x is not None


def _flatmap_frame_batch(batch_of_paths, config, mode):
    # flatmap expects an iterable; load_and_process_frame_batch yields/iterates
    return load_and_process_frame_batch(batch_of_paths, config, mode)


# ==============================================================================
# --- AUGMENTATION CODE MOVED TO data/augmentations/ MODULE ---
# ==============================================================================
# 
# All augmentation pipelines (V3-V7, surgical, etc.) have been refactored to:
#   - data/augmentations/transforms.py   (CustomUnsharpMask, NoOp)
#   - data/augmentations/pipelines.py    (all pipeline definitions)
#   - data/augmentations/registry.py     (get_pipeline factory)
#
# The following are now imported at the top of this file:
#   - CustomUnsharpMask, NoOp (custom transforms)
#   - revised_augmentation_pipeline_legacy (V3)
#   - augmentation_pipeline_v4 (V4)  
#   - augmentation_pipeline_v5 (V5)
#   - AUG_PIPELINE_V6_SIMULATOR, AUG_PIPELINE_V4_GENERALIST, etc. (V6 components)
#   - apply_augmentation_v6, apply_augmentation_v7 (callable functions)
#   - degrade_quality_pipeline, enhance_quality_pipeline, social_media_pipeline
#   - create_surgical_augmentation_pipeline, create_general_augmentation_pipeline
#
# To use the new registry pattern:
#   pipeline = get_pipeline(version=3)  # or 4, 5, 6, 7, 'surgical', 'general'
#
# ==============================================================================


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
    def __init__(self, source_datapipe: IterDataPipe, lookup: dict, frames_per_video: int):
        super().__init__()
        self.source_datapipe = source_datapipe
        self.lookup = lookup
        self.frames_per_video = frames_per_video

    def __iter__(self):
        for anchor_frame in self.source_datapipe:
            # If we only need 1 frame per video, just yield the anchor and we're done.
            if self.frames_per_video <= 1:
                yield anchor_frame
                continue

            clip_id = anchor_frame.get('clip_id')
            if clip_id is None:
                print(
                    f"WARNING: [MateFinder] Frame missing 'clip_id'. Path: {anchor_frame.get('path')}. Yielding self {self.frames_per_video} times.")
                for _ in range(self.frames_per_video):
                    yield anchor_frame
                continue

            # Pool of potential mates are all frames in the same clip EXCEPT the anchor.
            possible_mates = self.lookup.get(clip_id, [])
            mates_pool = [m for m in possible_mates if m['path'] != anchor_frame['path']]

            # This list will hold the final group of frames to be yielded sequentially.
            frames_to_yield = [anchor_frame]
            num_mates_needed = self.frames_per_video - 1

            if not mates_pool:
                # Edge case: Clip has only one frame. Repeat the anchor frame to fill the slots.
                frames_to_yield.extend([anchor_frame] * num_mates_needed)
            else:
                # Use random.choices which samples with replacement if k > len(population).
                # This elegantly handles cases where we need more mates than are available.
                chosen_mates = random.choices(mates_pool, k=num_mates_needed)
                frames_to_yield.extend(chosen_mates)

            # Yield all the selected frames one by one into the stream.
            for frame in frames_to_yield:
                yield frame


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
            # MODIFICATION START: Return 4 items: image, label, method_id (placeholder), path
            return image_tensor, label_id, -1, frame_path
            # MODIFICATION END
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
    Now supports selecting between the surgical (dynamic) and v3 (static) augmentation pipelines.
    """
    resolution = config['resolution']
    use_aug = mode == 'train' and config['use_data_augmentation']
    aug_params = config.get('augmentation_params', {})
    aug_version = aug_params.get('version')  # Check for the version key

    normalize_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=config['mean'], std=config['std'])
    ])

    def _load_single(frame_dict):
        frame_path, label_id, method_id = frame_dict['path'], frame_dict['label_id'], frame_dict['method_id']
        try:
            with fsspec.open(frame_path, "rb") as f:
                img = Image.open(f).convert("RGB")
                img = img.resize((resolution, resolution), Image.BICUBIC)

            if use_aug:
                img_np = np.array(img)
                augmented_img_np = None

                if aug_version == 5:
                    augmented_img_np = augmentation_pipeline_v5(image=img_np)['image']
                elif aug_version == 4:
                    augmented_img_np = augmentation_pipeline_v4(image=img_np)['image']
                elif aug_version == 3:
                    augmented_img_np = revised_augmentation_pipeline_legacy(image=img_np)['image']
                elif aug_version == 6:
                    augmented_img_np = apply_augmentation_v6(img_np, frame_dict)
                elif aug_version == 7:
                    augmented_img_np = apply_augmentation_v7(img_np)
                else:
                    # Default to the dynamic "surgical" pipeline
                    pipeline = create_surgical_augmentation_pipeline(aug_params, frame_dict)
                    augmented_img_np = pipeline(image=img_np)['image']

                img = Image.fromarray(augmented_img_np)

            image_tensor = normalize_transform(img)
            return image_tensor, label_id, method_id, frame_path
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

    # A3 / A3b: deterministic eval-time stress preset. Applied for ANY mode
    # when video_info.eval_aug_preset is set — this is how OOD stress lanes
    # work. Never invoked during training because training videos don't carry
    # this attribute.
    eval_aug_preset = getattr(video_info, "eval_aug_preset", None)
    if eval_aug_preset:
        try:
            from data.augmentations.pipelines import apply_eval_stress_preset
            images = [
                Image.fromarray(
                    apply_eval_stress_preset(np.array(img), eval_aug_preset, seed=42)
                )
                for img in images
            ]
        except Exception as eval_stress_err:
            # Do not silently degrade to un-stressed eval — that would make the
            # stress pool meaningless. Log and return None so the video is
            # dropped.
            print(
                f"[A3/A3b] Eval stress preset '{eval_aug_preset}' failed on "
                f"{video_info.method}/{video_info.video_id}: {eval_stress_err}"
            )
            return None

    normalize_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=config['mean'], std=config['std'])
    ])

    image_tensors = [normalize_transform(img) for img in images]
    video_tensor = torch.stack(image_tensors, dim=0)
    label = 0 if video_info.label == 'real' else 1

    # MODIFICATION START: Return 4 items: image, label, method_id (placeholder), path
    return video_tensor, label, -1, f"{video_info.method}/{video_info.video_id}"
    # MODIFICATION END


def load_and_process_video_detailed(video_info: VideoInfo, config: dict, mode: str, frame_count_override: int = None):
    """
    Enhanced version of load_and_process_video that returns additional metadata for detailed reporting.
    Returns: (video_tensor, label, method_id, path, video_id, frame_paths)
    """
    if frame_count_override is not None:
        frame_num = frame_count_override
    else:
        dl_params = config.get('dataloader_params', {})
        frame_num = dl_params.get('frames_per_video', 8)

    resolution = config['resolution']
    all_frame_paths = list(video_info.frame_paths)

    if len(all_frame_paths) < frame_num:
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
        image_results = list(executor.map(_load_single_frame, selected_paths))

    # Filter out any frames that failed to load
    images = [img for img in image_results if img is not None]

    if len(images) < frame_num:
        return None

    if mode == 'train' and config['use_data_augmentation']:
        aug_seed = random.randint(0, 2 ** 32 - 1)
        images = [data_aug_v2(img, config, augmentation_seed=aug_seed) for img in images]

    normalize_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=config['mean'], std=config['std'])
    ])

    image_tensors = [normalize_transform(img) for img in images]
    video_tensor = torch.stack(image_tensors, dim=0)
    label = 0 if video_info.label == 'real' else 1

    # Return 6 items for detailed reporting
    return (
        video_tensor, 
        label, 
        -1, 
        f"{video_info.method}/{video_info.video_id}",
        video_info.video_id,
        selected_paths
    )


def collate_fn(batch):
    """
    A simplified collate_fn, matching the one in abstract_dataset.py.
    Enhanced to support detailed reporting with video_id and frame_paths.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return {
            'image': torch.empty(0), 
            'label': torch.empty(0), 
            'method_id': torch.empty(0), 
            'path': [],
            'video_id': [],
            'frame_paths': []
        }

    # Handle both old 4-tuple format and new 6-tuple format for backward compatibility
    if len(batch[0]) == 4:
        # Old format: (image, label, method_id, path)
        images, labels, method_ids, paths = zip(*batch)
        video_ids = []
        frame_paths = []
    elif len(batch[0]) == 6:
        # New format: (image, label, method_id, path, video_id, frame_paths)
        images, labels, method_ids, paths, video_ids, frame_paths = zip(*batch)
        video_ids = list(video_ids)
        frame_paths = list(frame_paths)
    else:
        raise ValueError(f"Unexpected batch format with {len(batch[0])} elements")
        
    images = torch.stack(images, dim=0)
    labels = torch.LongTensor(labels)
    method_ids = torch.LongTensor(method_ids)  # Convert method_ids to a tensor

    data_dict = {
        'image': images,
        'label': labels,
        'method_id': method_ids,
        'path': list(paths),
        'video_id': video_ids,
        'frame_paths': frame_paths
    }
    return data_dict


def collate_fn_detailed(batch):
    """
    Specialized collate function for detailed reporting that expects 6-tuple format.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return {
            'image': torch.empty(0), 
            'label': torch.empty(0), 
            'method_id': torch.empty(0), 
            'path': [],
            'video_id': [],
            'frame_paths': []
        }

    if len(batch[0]) != 6:
        raise ValueError(f"collate_fn_detailed expects 6-tuple format, got {len(batch[0])} elements")
        
    images, labels, method_ids, paths, video_ids, frame_paths = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.LongTensor(labels)
    method_ids = torch.LongTensor(method_ids)

    data_dict = {
        'image': images,
        'label': labels,
        'method_id': method_ids,
        'path': list(paths),
        'video_id': list(video_ids),
        'frame_paths': list(frame_paths)
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
            self.num_workers = dl_params.get(
                'eval_num_workers',
                dl_params.get('num_workers_eval', 6)
            )
            self.prefetch_factor = dl_params.get(
                'eval_prefetch_factor',
                dl_params.get('prefetch_factor_eval', 4)
            )

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

        loader_kwargs = dict(
            dataset=pipe,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=is_train and self.num_workers > 0,  # Fix for num_workers=0 case
        )
        if self.num_workers > 0:
            loader_kwargs['prefetch_factor'] = self.prefetch_factor

        loader = DataLoader(**loader_kwargs)
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
    This implementation performs hierarchical sampling based on weights defined in the config.
    It first samples by 'method_category' and then balances by 'property_bucket' within that category.
    
    The real/fake ratio can be customized via the 'real_label_ratio' parameter in dataloader_params.
    If not specified or set to None, defaults to 0.5 (50/50 balance).
    """

    # Check if using the new lesson data control system first
    lesson_data_control = config.get('lesson_data_control', {})
    use_lesson_control = lesson_data_control.get('enabled', False)
    
    # DEBUG: Print what we're receiving
    print(f"[DATALOADER DEBUG] config.get('lesson_data_control') = {lesson_data_control}")
    print(f"[DATALOADER DEBUG] use_lesson_control = {use_lesson_control}")

    # The data from the Parquet file has 'label' (str) but not 'label_id' (int).
    # This loop ensures 'label_id' is present before the main logic begins.
    # This is confirmed to be necessary by the provided Parquet schema.
    for frame_info in all_train_frames:
        if 'label_id' not in frame_info:
            # We must have the 'label' key to proceed
            if 'label' not in frame_info:
                raise KeyError(
                    f"Frame dictionary is missing both 'label' and 'label_id'. Cannot proceed. Frame info: {frame_info}")
            # Create 'label_id' based on the string 'label'
            frame_info['label_id'] = 0 if frame_info['label'] == 'real' else 1

    dl_params = data_config['dataloader_params']
    BATCH_SIZE = dl_params['frames_per_batch']
    NUM_WORKERS = dl_params['num_workers']
    # Get frames_per_video, default to 2 for backward compatibility
    FRAMES_PER_VIDEO = dl_params.get('frames_per_video', 2)
    print(f"Configuring property-balanced loader with {FRAMES_PER_VIDEO} frames per video.")
    
    # Get real_label_ratio parameter for customizing real/fake balance
    # If None or not specified, defaults to 0.5 (50/50 balance)
    real_label_ratio = dl_params.get('real_label_ratio', None)
    if real_label_ratio is None:
        real_label_ratio = 0.5
    elif not (0.0 < real_label_ratio < 1.0):
        print(f"real_label_ratio must be between 0 and 1 (exclusive). Got: {real_label_ratio}")
        print("Defaulting to 0.5 (50/50 balance).")
        real_label_ratio = 0.5

    fake_label_ratio = 1.0 - real_label_ratio
    print(f"Using real/fake label ratio: {real_label_ratio:.2f} / {fake_label_ratio:.2f}")

    # A sensible minimum to avoid creating tiny, inefficient cycled iterators for small buckets
    MIN_BUCKET_SIZE = BATCH_SIZE

    # --- 1. Get category weights from config ---
    # These dictionaries define the sampling probability for each high-level category.
    # The keys MUST match the 'method_category' strings in your Parquet manifest.
    # Only validate legacy weights if NOT using lesson control
    if not use_lesson_control:
        real_category_weights = dl_params.get('real_category_weights', {})
        fake_category_weights = dl_params.get('fake_category_weights', {})

        if not real_category_weights or abs(sum(real_category_weights.values()) - 1.0) > 1e-6:
            raise ValueError(f"Real category weights do not sum to 1.0! Got: {sum(real_category_weights.values())}")

        if not fake_category_weights or abs(sum(fake_category_weights.values()) - 1.0) > 1e-6:
            raise ValueError(f"Fake category weights do not sum to 1.0! Got: {sum(fake_category_weights.values())}")

        print(f"Using REAL category weights: {real_category_weights}")
        print(f"Using FAKE category weights: {fake_category_weights}")
    else:
        # When using lesson control, set empty weights to avoid the validation
        real_category_weights = {}
        fake_category_weights = {}
        print("Using lesson data control - skipping legacy weight validation")

    # --- 2. Build the lookup table for finding mates efficiently ---
    clip_lookup = _build_clip_to_frames_lookup(all_train_frames)
    print(f"Built clip_id lookup table with {len(clip_lookup)} unique clips.")

    # --- 3. Segregate all frames by method category ---
    # [MODIFICATION] This section is now less critical for 'real' but still useful for legacy 'fake' path.
    # We will build the real frame list more directly in the next step.
    real_frames_by_category = defaultdict(list)
    fake_frames_by_category = defaultdict(list)

    # Get the list of fake categories for the legacy path
    fake_category_weights = dl_params.get('fake_category_weights', {})

    for frame_info in all_train_frames:
        category = frame_info.get('method_category')
        if category is None:
            continue

        if frame_info['label_id'] == 0:
            # We no longer need to segregate real frames by category here.
            # This will be handled in the next step.
            pass
        else:
            if category in fake_category_weights:
                fake_frames_by_category[category].append(frame_info)

    # --- 4. Create the hierarchical master stream for REAL data ---
    real_category_streams = []
    real_weights = []

    lesson_data_control = config.get('lesson_data_control', {})

    if lesson_data_control.get('enabled', False) and lesson_data_control.get('real_method_groups'):
        # --- NEW: Dynamic Grouping Logic for Real Methods (mirrors fake logic) ---
        print("--- Using DYNAMIC method grouping for REAL data stream. ---")
        
        real_method_groups = lesson_data_control.get('real_method_groups', {})
        all_real_frames = [f for f in all_train_frames if f['label_id'] == 0]
        
        # Track group info for logging
        group_info_log = []
        
        for group_name, group_info in real_method_groups.items():
            group_methods = set(group_info.get('methods', []))
            group_weight = group_info.get('weight', None)  # Allow None for default equal weighting
            
            frames_for_group = [f for f in all_real_frames if f.get('method') in group_methods]
            
            if not frames_for_group:
                print(f"WARNING: No frames found for real group '{group_name}', skipping.")
                continue
            
            print(f"  - Group '{group_name}': Found {len(frames_for_group)} frames from {len(group_methods)} method(s).")
            
            # Group by property bucket
            frames_by_bucket = defaultdict(list)
            for frame in frames_for_group:
                frames_by_bucket[frame['sharpness_bucket']].append(frame)
            
            # Create property-balanced stream for this group
            consolidated = _consolidate_small_buckets(frames_by_bucket, MIN_BUCKET_SIZE, group_name)
            group_stream = _create_master_stream_for_label(consolidated)
            
            if group_stream:
                real_category_streams.append(group_stream)
                real_weights.append(group_weight if group_weight is not None else 1.0)  # Default to 1.0 for equal weighting
                group_info_log.append({
                    'name': group_name,
                    'methods': list(group_methods),
                    'frame_count': len(frames_for_group),
                    'specified_weight': group_weight
                })
        
        if not real_category_streams:
            raise ValueError("Cannot create dataloader: no valid REAL data streams created.")
        
        # Normalize weights to sum to 1.0 (handles both explicit weights and default equal weighting)
        weight_sum = sum(real_weights)
        if not math.isclose(weight_sum, 1.0):
            if any(w is None for w in [g.get('weight') for g in real_method_groups.values()]):
                print(f"INFO: Real weights not specified or don't sum to 1.0 (sum={weight_sum:.3f}). Applying equal weighting.")
            else:
                print(f"WARNING: Real weights sum to {weight_sum:.3f}, renormalizing to 1.0")
            real_weights = [w/weight_sum for w in real_weights]
        
        # --- Enhanced Logging: Show final weight distribution ---
        print("\n=== REAL METHOD SAMPLING SUMMARY ===")
        for i, group_info in enumerate(group_info_log):
            print(f"  Group '{group_info['name']}':")
            print(f"    Methods: {', '.join(group_info['methods'])}")
            print(f"    Frame count: {group_info['frame_count']:,}")
            print(f"    Specified weight: {group_info['specified_weight']}")
            print(f"    Final sampling weight: {real_weights[i]:.4f} ({real_weights[i]*100:.2f}%)")
        print(f"Total real streams: {len(real_category_streams)}")
        print(f"Total real weight: {sum(real_weights):.6f}")
        print("=" * 40 + "\n")
    
        master_real_stream = CustomSampleMultiplexerDataPipe(real_category_streams, real_weights)

    else:
        # --- FALLBACK: Original logic (all real methods pooled together) ---
        print("--- Using UNIFIED real data stream (no per-method weighting). ---")
        
        allowed_real_sources = data_config.get('dataset_methods', {}).get('use_real_sources', [])
        if not allowed_real_sources:
            raise ValueError("`dataset_methods.use_real_sources` is empty or not defined.")
        
        allowed_real_sources_set = set(allowed_real_sources)
        all_real_frames = [
            f for f in all_train_frames
            if f['label_id'] == 0 and f.get('method') in allowed_real_sources_set
        ]
        
        if not all_real_frames:
            raise ValueError("No REAL frames found for the sources specified in `use_real_sources`.")
        
        real_frames_by_bucket = defaultdict(list)
        for frame in all_real_frames:
            real_frames_by_bucket[frame['sharpness_bucket']].append(frame)
        
        consolidated_real_buckets = _consolidate_small_buckets(real_frames_by_bucket, MIN_BUCKET_SIZE, "real_master")
        master_real_stream = _create_master_stream_for_label(consolidated_real_buckets)
        
        if not master_real_stream:
            raise ValueError("Cannot create dataloader: the master REAL data stream could not be created.")

    # --- 5. Create the hierarchical master stream for FAKE data (Symmetrical or Dynamic Logic) ---
    fake_category_streams = []
    fake_weights = []

    lesson_data_control = config.get('lesson_data_control', {})

    if lesson_data_control.get('enabled', False):
        # --- NEW: Dynamic Grouping Logic for Lessons ---
        print("--- Using DYNAMIC method grouping for FAKE data stream. ---")

        fake_method_groups = lesson_data_control.get('fake_method_groups', {})
        if not fake_method_groups:
            raise ValueError("`lesson_data_control` is enabled, but `fake_method_groups` is empty or missing.")

        # Pre-filter fake frames for efficiency
        all_fake_frames = [f for f in all_train_frames if f['label_id'] == 1]
        
        # Track group info for logging
        group_info_log = []

        for group_name, group_info in fake_method_groups.items():
            group_methods = set(group_info.get('methods', []))
            group_weight = group_info.get('weight', None)  # Allow None for default equal weighting

            if not group_methods:
                print(f"WARNING: Dynamic group '{group_name}' has no methods defined, skipping.")
                continue

            # A. Gather all frames belonging to this dynamic group
            frames_for_group = [f for f in all_fake_frames if f['method'] in group_methods]

            if not frames_for_group:
                print(f"WARNING: No frames found for dynamic group '{group_name}', skipping.")
                continue

            print(
                f"  - Group '{group_name}': Found {len(frames_for_group)} frames across {len(group_methods)} methods.")

            # B. Group frames within this dynamic group by property bucket
            frames_by_bucket = defaultdict(list)
            for frame in frames_for_group:
                frames_by_bucket[frame['sharpness_bucket']].append(frame)

            # C. Consolidate small buckets and create a property-balanced stream for this group
            consolidated = _consolidate_small_buckets(frames_by_bucket, MIN_BUCKET_SIZE, group_name)
            group_stream = _create_master_stream_for_label(consolidated)

            if group_stream:
                fake_category_streams.append(group_stream)
                fake_weights.append(group_weight if group_weight is not None else 1.0)  # Default to 1.0 for equal weighting
                group_info_log.append({
                    'name': group_name,
                    'methods': list(group_methods),
                    'frame_count': len(frames_for_group),
                    'specified_weight': group_weight
                })

    else:
        # --- ORIGINAL: Static Grouping Logic (Fallback) ---
        print("--- Using STATIC method_category grouping for FAKE data stream. ---")
        for category, weight in fake_category_weights.items():
            frames = fake_frames_by_category.get(category)
            if not frames:
                print(f"WARNING: No frames found for FAKE category '{category}', skipping.")
                continue

            frames_by_bucket = defaultdict(list)
            for frame in frames:
                frames_by_bucket[frame['sharpness_bucket']].append(frame)

            consolidated = _consolidate_small_buckets(frames_by_bucket, MIN_BUCKET_SIZE, category)
            category_stream = _create_master_stream_for_label(consolidated)

            if category_stream:
                fake_category_streams.append(category_stream)
                fake_weights.append(weight)

    if not fake_category_streams:
        raise ValueError(
            "Cannot create dataloader: no valid FAKE data streams were created. Check config and Parquet file.")

    # Normalize weights to sum to 1.0 (handles both explicit weights and default equal weighting)
    weight_sum = sum(fake_weights)
    if not math.isclose(weight_sum, 1.0):
        if lesson_data_control.get('enabled', False):
            # Check if any weights were unspecified (None)
            fake_method_groups = lesson_data_control.get('fake_method_groups', {})
            if any(g.get('weight') is None for g in fake_method_groups.values()):
                print(f"INFO: Fake weights not specified or don't sum to 1.0 (sum={weight_sum:.3f}). Applying equal weighting.")
            else:
                print(f"WARNING: Fake weights sum to {weight_sum:.3f}, renormalizing to 1.0")
        else:
            print(f"WARNING: Fake weights sum to {weight_sum:.3f}, renormalizing to 1.0")
        fake_weights = [w/weight_sum for w in fake_weights]
    
    # --- Enhanced Logging: Show final weight distribution ---
    if lesson_data_control.get('enabled', False):
        print("\n=== FAKE METHOD SAMPLING SUMMARY ===")
        for i, group_info in enumerate(group_info_log):
            print(f"  Group '{group_info['name']}':")
            print(f"    Methods: {', '.join(group_info['methods'])}")
            print(f"    Frame count: {group_info['frame_count']:,}")
            print(f"    Specified weight: {group_info['specified_weight']}")
            print(f"    Final sampling weight: {fake_weights[i]:.4f} ({fake_weights[i]*100:.2f}%)")
        print(f"Total fake streams: {len(fake_category_streams)}")
        print(f"Total fake weight: {sum(fake_weights):.6f}")
        print("=" * 40 + "\n")

    master_fake_stream = CustomSampleMultiplexerDataPipe(fake_category_streams, fake_weights)

    # --- 6. The final multiplexer uses the new master streams with configurable real/fake balance ---
    anchor_pipe = CustomSampleMultiplexerDataPipe([master_real_stream, master_fake_stream], [real_label_ratio, fake_label_ratio])

    # --- 7. Downstream logic remains the same (sharding, finding mates, parallel loading) ---
    if NUM_WORKERS > 0:
        anchor_pipe = anchor_pipe.sharding_filter()
    anchor_pipe = anchor_pipe.shuffle(buffer_size=10000)

    # <<< MODIFIED: This pipe now takes one anchor frame and yields a sequence of frames_per_video frames
    combined_pipe = MateFinderDataPipe(anchor_pipe, clip_lookup, frames_per_video=FRAMES_PER_VIDEO)

    # Batch frame dictionaries for I/O efficiency, then use the parallel loader
    # A larger IO batch is sent to each worker for better cloud storage throughput.
    io_batch_size = BATCH_SIZE * 4
    combined_pipe = combined_pipe.batch(io_batch_size).flatmap(
        partial(load_and_process_property_batch, config=config, mode='train')
    )

    return DataLoader(
        combined_pipe,
        batch_size=BATCH_SIZE,  # Let the DataLoader create the final GPU batch
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
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


def create_pure_validation_loader(
        videos: List[VideoInfo],
        config: Dict,
        data_config: Dict,
        detailed_reporting: bool = False
) -> "LazyDataLoaderManager":
    """
    Creates a validation dataloader for a specific, pre-selected set of videos.

    This function is designed to work with a prepared list of `VideoInfo` objects
    (e.g., from `prepare_pure_validation`) to create a dataloader that perfectly
    mimics the behavior of the standard validation loaders used during training.

    It groups the provided videos by their method and uses the LazyDataLoaderManager,
    ensuring that the data loading, batching, and lack of augmentation match the
    standard validation pipeline.

    Args:
        videos: A list of VideoInfo objects for the validation set.
        config: The main model/training configuration dictionary.
        data_config: The data-specific configuration dictionary.
        detailed_reporting: If True, use detailed version that includes video_id and frame_paths.

    Returns:
        A configured LazyDataLoaderManager instance ready for validation.
    """
    print("--- Creating Pure Validation Loader ---")

    if not videos:
        print("Received an empty list of videos. The validation loader will be empty.")
        # Return an empty but valid manager to prevent downstream errors
        return LazyDataLoaderManager(
            videos_by_method={},
            config=config,
            data_config=data_config,
            batch_size=config['test_batchSize'],
            mode='test',
            max_loaders=MAX_LOADERS_IN_MEMORY
        )

    # 1. Group the provided videos by their method name, just like in the original helper.
    videos_by_method = defaultdict(list)
    video_ids_seen = set()
    duplicates_found = 0
    
    for v in videos:
        video_key = f"{v.method}_{v.video_id}"
        if video_key in video_ids_seen:
            duplicates_found += 1
            print(f"WARNING: Duplicate video found: {video_key}")
        else:
            video_ids_seen.add(video_key)
            videos_by_method[v.method].append(v)

    if duplicates_found > 0:
        print(f"WARNING: Found and removed {duplicates_found} duplicate videos during grouping.")
        
    print(f"Assembled validation data for {len(videos_by_method)} method(s) with {len(sum(videos_by_method.values(), []))} total unique videos.")

    # 2. Choose the appropriate map function and collate function based on detailed_reporting
    if detailed_reporting:
        map_fn = partial(_map_video_detailed, config=config, mode='test')
        collate_function = collate_fn_detailed
        print("Using detailed reporting mode with metadata preservation.")
    else:
        map_fn = partial(_map_video, config=config, mode='test')
        collate_function = collate_fn
        print("Using standard validation mode.")

    # 3. Create a specialized LazyDataLoaderManager that supports the detailed map function
    class DetailedLazyDataLoaderManager(LazyDataLoaderManager):
        def _create_loader(self, method):
            """Creates a DataLoader for a specific method with custom map function."""
            videos = self.videos_by_method[method]
            is_train = self.mode == 'train'

            pipe = IterableWrapper(videos).shuffle() if is_train else IterableWrapper(videos)
            pipe = Mapper(pipe, map_fn)
            pipe = Filter(pipe, _not_none)

            loader_kwargs = dict(
                dataset=pipe,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=collate_function,
                persistent_workers=is_train and self.num_workers > 0,
            )
            if self.num_workers > 0:
                loader_kwargs['prefetch_factor'] = self.prefetch_factor

            loader = DataLoader(**loader_kwargs)
            return loader

    # 4. Instantiate the specialized LazyDataLoaderManager
    validation_loader_manager = DetailedLazyDataLoaderManager(
        videos_by_method=videos_by_method,
        config=config,
        data_config=data_config,
        batch_size=config['test_batchSize'],
        mode='test',
        max_loaders=MAX_LOADERS_IN_MEMORY
    )

    return validation_loader_manager


def _create_validation_loader(
        val_videos: List[VideoInfo],
        config: dict,
        data_config: dict
) -> LazyDataLoaderManager:
    """
    [NEW HELPER FUNCTION]
    Creates a validation dataloader manager from a list of VideoInfo objects.

    This function encapsulates the logic for creating a per-method validation loader,
    making it reusable for both in-distribution and holdout validation sets.

    Args:
        val_videos: A list of VideoInfo objects for this validation set.
        config: The main model/training configuration dictionary.
        data_config: The data-specific configuration dictionary.

    Returns:
        A configured LazyDataLoaderManager instance.
    """
    if not val_videos:
        print("WARNING: Received an empty list of videos. The corresponding validation loader will be empty.")

    val_videos_by_method = defaultdict(list)
    for v in val_videos:
        val_videos_by_method[v.method].append(v)

    val_loader_manager = LazyDataLoaderManager(
        videos_by_method=val_videos_by_method,
        config=config,
        data_config=data_config,
        batch_size=config['test_batchSize'],
        mode='test',
        max_loaders=MAX_LOADERS_IN_MEMORY
    )
    return val_loader_manager


def create_dataloaders(
        train_data: List,
        val_in_dist_videos: List[VideoInfo],
        val_holdout_videos: List[VideoInfo],
        config: dict,
        data_config: dict
):
    """
    [MODIFIED FACTORY FUNCTION]
    Factory function to create all dataloaders based on the specified strategy.

    This function is upgraded to support a dual-validation setup by creating three
    distinct dataloaders:
    1. train_loader: For training, configured by the chosen strategy.
    2. val_in_dist_loader: For validating on unseen videos from training methods.
    3. val_holdout_loader: For validating on unseen videos from held-out methods.

    Args:
        train_data: The data for the training set (format depends on strategy).
        val_in_dist_videos: List of VideoInfo objects for in-distribution validation.
        val_holdout_videos: List of VideoInfo objects for holdout validation.
        config: The main model/training configuration dictionary.
        data_config: The data-specific configuration dictionary.

    Returns:
        A tuple of three dataloaders: (train_loader, val_in_dist_loader, val_holdout_loader).
    """
    strategy = data_config.get('dataloader_params', {}).get('strategy', 'per_method')
    print(f"--- Creating dataloaders with strategy: '{strategy}' ---")

    # --- 1. Create the Training Loader (Logic is largely unchanged) ---
    if strategy == 'per_method':
        # This logic was previously in `_create_per_method_loaders`.
        # We bring it here for clarity and consistency.
        train_videos_by_method = defaultdict(list)
        for v in train_data:
            train_videos_by_method[v.method].append(v)

        train_batch_size = data_config['dataloader_params']['batch_size'] // 2
        if train_batch_size == 0: train_batch_size = 1

        train_loader = LazyDataLoaderManager(
            videos_by_method=train_videos_by_method, config=config,
            data_config=data_config, batch_size=train_batch_size, mode='train',
            max_loaders=MAX_LOADERS_IN_MEMORY
        )
    elif strategy == 'video_level':
        train_loader = _create_random_video_loader(train_data, config, data_config)
    elif strategy == 'frame_level':
        train_loader = _create_random_frame_loader(train_data, config, data_config)
    elif strategy == 'property_balancing':
        train_loader = _create_property_balanced_loader(train_data, config, data_config)
    else:
        raise ValueError(f"Unknown dataloader strategy: '{strategy}'")

    # --- 2. Create the TWO Validation Loaders using the new helper ---
    print("\n--- Creating In-Distribution Validation Loader ---")
    val_in_dist_loader = _create_validation_loader(val_in_dist_videos, config, data_config)

    print("\n--- Creating Holdout Validation Loader ---")
    val_holdout_loader = _create_validation_loader(val_holdout_videos, config, data_config)

    return train_loader, val_in_dist_loader, val_holdout_loader
