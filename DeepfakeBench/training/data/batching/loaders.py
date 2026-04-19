"""
Frame and video loading functions for dataloaders.

This module contains the core loading logic that processes frames and videos
from cloud storage, applies augmentations, and returns tensors.
"""
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Generator, List, Optional, Tuple

import fsspec
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

# Import augmentation pipelines from the refactored module
from data.augmentations import get_pipeline
from data.augmentations.pipelines import (
    revised_augmentation_pipeline_legacy,
    augmentation_pipeline_v4,
    augmentation_pipeline_v5,
    apply_augmentation_v6,
    apply_augmentation_v7,
    create_surgical_augmentation_pipeline,
    create_general_augmentation_pipeline,
)

# Try importing VideoInfo from different locations for flexibility
try:
    from data.splitting import VideoInfo
except ImportError:
    try:
        from prepare_splits import VideoInfo
    except ImportError:
        # Fallback - VideoInfo will be passed as a duck-typed object
        VideoInfo = Any


def data_aug_v2(img: Image.Image, config: dict, augmentation_seed: Optional[int] = None) -> Image.Image:
    """
    A general-purpose quality augmentation pipeline that is now configurable.
    
    Args:
        img: PIL Image to augment
        config: Configuration dict containing 'augmentation_params'
        augmentation_seed: Optional seed for reproducible augmentation
        
    Returns:
        Augmented PIL Image
    """
    if augmentation_seed is not None:
        random.seed(augmentation_seed)
        np.random.seed(augmentation_seed)

    aug_params = config.get('augmentation_params', {})
    pipeline = create_general_augmentation_pipeline(aug_params)

    transformed = pipeline(image=np.array(img))
    return Image.fromarray(transformed['image'])


def load_and_process_frame_batch(
    frame_info_batch: List[Tuple[str, int]],
    config: dict,
    mode: str
) -> Generator[Tuple[torch.Tensor, int, int, str], None, None]:
    """
    Loads and processes a BATCH of frames in parallel using a thread pool.
    Used by the 'frame_level' strategy.
    
    Args:
        frame_info_batch: List of (frame_path, label_id) tuples
        config: Training configuration dict
        mode: 'train' or 'test'
        
    Yields:
        Tuples of (image_tensor, label_id, method_id, frame_path)
    """
    resolution = config['resolution']
    use_aug = mode == 'train' and config.get('use_data_augmentation', False)

    normalize_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=config['mean'], std=config['std'])
    ])

    def _load_single(frame_info: Tuple[str, int]) -> Optional[Tuple]:
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
            # Return 4 items: image, label, method_id (placeholder), path
            return image_tensor, label_id, -1, frame_path
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=24) as executor:
        results = list(executor.map(_load_single, frame_info_batch))

    for result in results:
        if result is not None:
            yield result


def load_and_process_property_batch(
    frame_dict_batch: List[Dict],
    config: dict,
    mode: str
) -> Generator[Tuple[torch.Tensor, int, int, str], None, None]:
    """
    Parallel frame loader for the property-balanced strategy.
    Supports selecting between different augmentation pipelines (surgical, v3-v7).
    
    Args:
        frame_dict_batch: List of frame dictionaries with 'path', 'label_id', 'method_id'
        config: Training configuration dict
        mode: 'train' or 'test'
        
    Yields:
        Tuples of (image_tensor, label_id, method_id, frame_path)
    """
    resolution = config['resolution']
    use_aug = mode == 'train' and config.get('use_data_augmentation', False)
    aug_params = config.get('augmentation_params', {})
    aug_version = aug_params.get('version')  # Check for the version key

    normalize_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=config['mean'], std=config['std'])
    ])

    def _load_single(frame_dict: Dict) -> Optional[Tuple]:
        frame_path = frame_dict['path']
        label_id = frame_dict['label_id']
        method_id = frame_dict['method_id']
        
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


def load_and_process_video(
    video_info: "VideoInfo",
    config: dict,
    mode: str,
    frame_count_override: Optional[int] = None
) -> Optional[Tuple[torch.Tensor, int, int, str]]:
    """
    Loads frames for a video in parallel. Used by 'video_level' and 'per_method' strategies.
    
    Args:
        video_info: VideoInfo object containing video metadata
        config: Training configuration dict
        mode: 'train' or 'test'
        frame_count_override: Optional override for number of frames to load
        
    Returns:
        Tuple of (video_tensor, label, method_id, path) or None if loading failed
    """
    if frame_count_override is not None:
        frame_num = frame_count_override
    else:
        dl_params = config.get('dataloader_params', {})
        frame_num = dl_params.get('frames_per_video', 8)

    resolution = config['resolution']
    all_frame_paths = list(video_info.frame_paths)

    if len(all_frame_paths) < frame_num:
        # If a video doesn't have enough frames, skip it
        return None

    random.shuffle(all_frame_paths)
    selected_paths = all_frame_paths[:frame_num]

    def _load_single_frame(path: str) -> Optional[Image.Image]:
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
        # Not enough frames could be loaded for this video
        return None

    if mode == 'train' and config.get('use_data_augmentation', False):
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

    # Return 4 items: image, label, method_id (placeholder), path
    return video_tensor, label, -1, f"{video_info.method}/{video_info.video_id}"


def load_and_process_video_detailed(
    video_info: "VideoInfo",
    config: dict,
    mode: str,
    frame_count_override: Optional[int] = None
) -> Optional[Tuple[torch.Tensor, int, int, str, str, List[str]]]:
    """
    Enhanced version of load_and_process_video that returns additional metadata for detailed reporting.
    
    Args:
        video_info: VideoInfo object containing video metadata
        config: Training configuration dict
        mode: 'train' or 'test'
        frame_count_override: Optional override for number of frames to load
        
    Returns:
        Tuple of (video_tensor, label, method_id, path, video_id, frame_paths) or None
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

    def _load_single_frame(path: str) -> Optional[Image.Image]:
        """Helper function for a single thread to load one frame."""
        try:
            with fsspec.open(path, "rb") as f:
                img = Image.open(f).convert("RGB")
                img = img.resize((resolution, resolution), Image.BICUBIC)
            return img
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=16) as executor:
        image_results = list(executor.map(_load_single_frame, selected_paths))

    images = [img for img in image_results if img is not None]

    if len(images) < frame_num:
        return None

    if mode == 'train' and config.get('use_data_augmentation', False):
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


def collate_fn(batch: List) -> Dict[str, Any]:
    """
    A simplified collate_fn for batching video/frame data.
    Supports both 4-tuple and 6-tuple formats for backward compatibility.
    
    Args:
        batch: List of tuples from the dataloader
        
    Returns:
        Dictionary with 'image', 'label', 'method_id', 'path', 'video_id', 'frame_paths'
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

    # Handle both old 4-tuple format and new 6-tuple format
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
    method_ids = torch.LongTensor(method_ids)

    return {
        'image': images,
        'label': labels,
        'method_id': method_ids,
        'path': list(paths),
        'video_id': video_ids,
        'frame_paths': frame_paths
    }


def collate_fn_detailed(batch: List) -> Dict[str, Any]:
    """
    Specialized collate function for detailed reporting that expects 6-tuple format.
    
    Args:
        batch: List of 6-tuples from the dataloader
        
    Returns:
        Dictionary with full metadata including video_id and frame_paths
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

    return {
        'image': images,
        'label': labels,
        'method_id': method_ids,
        'path': list(paths),
        'video_id': list(video_ids),
        'frame_paths': list(frame_paths)
    }


# Helper functions for map operations
def _not_none(x) -> bool:
    """Filter function to remove None values from DataPipe."""
    return x is not None


def _map_video(video_info, config, mode):
    """Map function wrapper for load_and_process_video."""
    return load_and_process_video(video_info, config, mode)


def _map_video_detailed(video_info, config, mode):
    """Map function wrapper for load_and_process_video_detailed."""
    return load_and_process_video_detailed(video_info, config, mode)


def _flatmap_frame_batch(batch_of_paths, config, mode):
    """Flatmap function wrapper for load_and_process_frame_batch."""
    return load_and_process_frame_batch(batch_of_paths, config, mode)
