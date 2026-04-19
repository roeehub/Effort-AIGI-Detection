"""
DeepLive Data Source

This module provides the data pipeline for the DeepLive dataset stored in GCS.
It creates paired real/fake frame batches with landmark-based augmentation support.

The DeepLive dataset consists of video samples with synchronized real and fake frames,
ideal for contrastive learning approaches.

Usage:
    from data.sources import create_data_pipeline
    
    # In config:
    data_config['data_source'] = 'deeplive'
    data_config['deeplive'] = {
        'gcs_bucket': 'live-deepfake-methods-real-and-fake-frames-cropped',
        'sampling_mode': 'sparse',
        'train_split': 0.8,
        'val_split': 0.1,
    }
    
    # Create pipeline
    result = create_data_pipeline(config, data_config, logger)
"""

import logging
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from . import register_data_source, DataPipelineResult


class DeepLiveValidationAdapter:
    """
    Adapter that wraps a DeepLive DataLoader to provide the interface expected by trainer.py.
    
    The trainer expects validation loaders to have:
    - `videos_by_method`: Dict[str, List] mapping methods to video lists
    - `keys()`: Method to iterate over method names
    - `__getitem__(method)`: Get dataloader for a specific method
    - Truthiness check without calling len()
    
    This adapter provides that interface while wrapping the simple DataLoader from DeepLive.
    """
    
    def __init__(
        self,
        dataloader: DataLoader,
        samples: List[Any],
        strategy_name: str = 'deeplive'
    ):
        """
        Args:
            dataloader: The underlying PyTorch DataLoader
            samples: List of DeepLiveSample objects
            strategy_name: Name to use as the single "method" (default: 'deeplive')
        """
        self._dataloader = dataloader
        self._samples = samples
        self._strategy_name = strategy_name
        
        # Build videos_by_method mapping
        # For DeepLive, we treat the whole dataset as a single "method"
        # since it doesn't have the multi-method structure of the manifest pipeline
        self.videos_by_method = {strategy_name: samples}
    
    def keys(self):
        """Return method names (just 'deeplive' for this adapter)."""
        return self.videos_by_method.keys()
    
    def __getitem__(self, method: str) -> DataLoader:
        """Get the dataloader. Ignores method since we only have one."""
        return self._dataloader
    
    def __bool__(self) -> bool:
        """Truthiness check - returns True if we have samples."""
        return len(self._samples) > 0
    
    def __iter__(self):
        """Iterate over the underlying dataloader."""
        return iter(self._dataloader)
    
    @property
    def dataset(self):
        """Access to underlying dataset."""
        return self._dataloader.dataset


@register_data_source('deeplive')
def create_deeplive_pipeline(
    config: Dict[str, Any],
    data_config: Dict[str, Any],
    logger: logging.Logger,
    **kwargs
) -> DataPipelineResult:
    """
    Create data pipeline from DeepLive GCS bucket.
    
    This function:
    1. Connects to the DeepLive GCS bucket
    2. Discovers all available samples
    3. Splits into train/val/test sets
    4. Creates dataloaders with paired real/fake frames
    5. Optionally sets up landmark-based augmentation
    
    Args:
        config: Main training configuration
        data_config: Data-specific configuration including:
            - deeplive: DeepLive-specific settings
                - gcs_bucket: GCS bucket name
                - gcs_project: GCP project (or from GOOGLE_CLOUD_PROJECT env)
                - sampling_mode: 'sparse' or 'full'
                - anchor_indices: Frame indices for sparse mode
                - train_split: Training set proportion (default 0.8)
                - val_split: Validation set proportion (default 0.1)
            - augmentation: Augmentation settings
        logger: Logger for output
        **kwargs: Additional arguments
            - transform: Optional pre-built transform function
            
    Returns:
        DataPipelineResult with train_loader, validation loaders, and metadata
    """
    from dataset.deeplive_dataset import DeepLiveDataset
    from data.batching.deeplive import DeepLiveBatchingStrategy, DeepLiveBatchingConfig
    
    deeplive_config = data_config.get('deeplive', {})
    seed = config.get('manualSeed', config.get('seed', 737))
    
    logger.info("------- DeepLive Data Source: Connecting to GCS -------")
    
    # Create DeepLive dataset
    gcs_bucket = deeplive_config.get('gcs_bucket', 'live-deepfake-methods-real-and-fake-frames')
    gcs_project = deeplive_config.get('gcs_project', os.environ.get('GOOGLE_CLOUD_PROJECT', 'train-cvit2'))
    use_landmarks = deeplive_config.get('use_landmarks', True)
    
    logger.info(f"Creating DeepLive dataset:")
    logger.info(f"  - GCS bucket: {gcs_bucket}")
    logger.info(f"  - GCS project: {gcs_project}")
    logger.info(f"  - Use landmarks: {use_landmarks}")
    
    dataset = DeepLiveDataset(
        bucket_name=gcs_bucket,
        gcs_project=gcs_project,
        use_landmarks=use_landmarks,
    )
    
    # Discover samples
    logger.info("Discovering samples from GCS...")
    samples = dataset.discover_samples()
    logger.info(f"Discovered {len(samples)} samples")
    
    if len(samples) == 0:
        raise ValueError(
            f"No samples discovered from GCS bucket '{gcs_bucket}'! "
            "Check bucket name and permissions."
        )
    
    # Split samples
    train_split = deeplive_config.get('train_split', 0.8)
    val_split = deeplive_config.get('val_split', 0.1)
    
    train_samples, val_samples, test_samples = _split_samples(
        samples, train_split, val_split, seed, logger
    )
    
    # Create augmentation transform
    transform = kwargs.get('transform')
    if transform is None:
        transform = _create_augmentation_transform(config, deeplive_config, logger, data_config)
    
    # Determine num_workers
    no_multiprocessing = os.environ.get('NO_MULTIPROCESSING', '').lower() in ('1', 'true', 'yes')
    device_is_cpu = not torch.cuda.is_available()
    
    if no_multiprocessing or device_is_cpu:
        num_workers = 0
        logger.info("Using num_workers=0 (multiprocessing disabled)")
    else:
        num_workers = config.get('num_workers', deeplive_config.get('num_workers', 4))
    
    # Create batching config
    batching_config = DeepLiveBatchingConfig(
        batch_size=config.get('frames_per_batch', deeplive_config.get('frames_per_batch', 32)),
        num_workers=num_workers,
        prefetch_factor=config.get('prefetch_factor', 2) if num_workers > 0 else None,
        frame_sampling=deeplive_config.get('sampling_mode', 'sparse'),
        sparse_indices=deeplive_config.get('anchor_indices', [0, 2, 4, 6, 8, 10, 12, 14]),
    )
    
    # Create batching strategy
    strategy = DeepLiveBatchingStrategy(
        config=config,
        data_config=data_config,
        strategy_config=batching_config,
        dataset=dataset,
        transform=transform,
    )
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader = strategy.create_train_loader(train_samples)
    val_loader_raw = strategy.create_validation_loader(val_samples, mode='test')
    test_loader_raw = strategy.create_validation_loader(test_samples, mode='test')
    
    # Wrap validation loaders with adapter to provide the interface trainer.py expects
    # (videos_by_method, keys(), __getitem__, truthiness without len())
    val_loader = DeepLiveValidationAdapter(val_loader_raw, val_samples, 'deeplive_val')
    test_loader = DeepLiveValidationAdapter(test_loader_raw, test_samples, 'deeplive_holdout')
    
    logger.info("Validation loaders wrapped with DeepLiveValidationAdapter for trainer compatibility")
    
    # Compute statistics
    strategies = _count_by_strategy(samples)
    data_stats = {
        'total_samples': len(samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples),
        'train_split': train_split,
        'val_split': val_split,
        'gcs_bucket': gcs_bucket,
        'sampling_mode': batching_config.frame_sampling,
        'frames_per_sample': len(batching_config.sparse_indices) * 2,  # real + fake
        # Strategy counts (fake only since DeepLive always has paired real/fake)
        'strategies': strategies,
        # Required by train_sweep.py for run overview logging
        'discovered_videos': len(samples),  # Each sample is a video pair
        'discovered_methods': len(strategies),  # Number of unique deepfake strategies
        'unbalanced_train_count': len(train_samples),
        'unbalanced_val_count': len(val_samples) + len(test_samples),
        'train_video_count': len(train_samples),
        'train_frame_count': len(train_samples) * len(batching_config.sparse_indices) * 2,
        'val_video_count': len(val_samples),
        'val_frame_count': len(val_samples) * len(batching_config.sparse_indices) * 2,
    }
    
    logger.info(f"DeepLive pipeline created:")
    logger.info(f"  - Train samples: {len(train_samples)}")
    logger.info(f"  - Val samples: {len(val_samples)}")
    logger.info(f"  - Test samples: {len(test_samples)}")
    
    return DataPipelineResult(
        train_loader=train_loader,
        val_in_dist_loader=val_loader,
        val_holdout_loader=test_loader,  # Use test as holdout
        train_samples=train_samples,
        data_stats=data_stats,
        ood_loader=None,  # No OOD for DeepLive
    )


def _split_samples(
    samples: List[Any],
    train_split: float,
    val_split: float,
    seed: int,
    logger: logging.Logger
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Split samples into train/val/test sets.
    
    Args:
        samples: All samples
        train_split: Proportion for training (e.g., 0.8)
        val_split: Proportion for validation (e.g., 0.1)
        seed: Random seed for reproducibility
        logger: Logger
        
    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)
    
    n_total = len(shuffled)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    train_samples = shuffled[:n_train]
    val_samples = shuffled[n_train:n_train + n_val]
    test_samples = shuffled[n_train + n_val:]
    
    logger.info(f"Data split (seed={seed}):")
    logger.info(f"  - Train: {len(train_samples)} samples ({len(train_samples)/n_total*100:.1f}%)")
    logger.info(f"  - Val: {len(val_samples)} samples ({len(val_samples)/n_total*100:.1f}%)")
    logger.info(f"  - Test: {len(test_samples)} samples ({len(test_samples)/n_total*100:.1f}%)")
    
    return train_samples, val_samples, test_samples


def _create_base_augmentation_pipeline(aug_config: Dict[str, Any], logger: logging.Logger):
    """
    Create base augmentation pipeline with color/quality augmentations.
    
    These augmentations help prevent shortcut learning from:
    - Color shifts between real/fake (deepfake methods often add color cast)
    - Quality differences (compression, blur artifacts)
    - Orientation biases
    
    All augmentations are low-risk, high-reward choices that don't destroy
    the signal but add robustness.
    
    Args:
        aug_config: Augmentation config dict
        logger: Logger
        
    Returns:
        Albumentations Compose pipeline or None
    """
    try:
        import albumentations as A
    except ImportError:
        logger.warning("albumentations not installed - skipping base augmentations")
        return None
    
    # Check if base augmentations are disabled
    if aug_config.get('disable_base_augmentations', False):
        logger.info("Base augmentations disabled by config")
        return None
    
    # Get configurable parameters with sensible defaults
    base_config = aug_config.get('base', {})
    
    # Build augmentation list
    aug_list = []
    
    # 1. Horizontal flip - always useful for faces (p=0.5)
    if base_config.get('horizontal_flip', True):
        aug_list.append(A.HorizontalFlip(p=0.5))
        
    # 2. Color augmentations - CRITICAL for preventing color shift shortcuts
    if base_config.get('color_augmentations', True):
        # Brightness/Contrast - breaks "darker = fake" shortcuts
        aug_list.append(A.RandomBrightnessContrast(
            brightness_limit=base_config.get('brightness_limit', 0.15),
            contrast_limit=base_config.get('contrast_limit', 0.15),
            p=base_config.get('brightness_contrast_p', 0.4)
        ))
        
        # Hue/Saturation/Value - breaks color cast shortcuts
        aug_list.append(A.HueSaturationValue(
            hue_shift_limit=base_config.get('hue_shift_limit', 15),
            sat_shift_limit=base_config.get('sat_shift_limit', 20),
            val_shift_limit=base_config.get('val_shift_limit', 15),
            p=base_config.get('hsv_p', 0.3)
        ))
        
        # RGB shift - additional color robustness
        aug_list.append(A.RGBShift(
            r_shift_limit=base_config.get('rgb_shift_limit', 10),
            g_shift_limit=base_config.get('rgb_shift_limit', 10),
            b_shift_limit=base_config.get('rgb_shift_limit', 10),
            p=base_config.get('rgb_shift_p', 0.2)
        ))
    
    # 3. Quality augmentations - robustness to compression/blur
    if base_config.get('quality_augmentations', True):
        # JPEG compression - very common in real-world deepfakes
        aug_list.append(A.ImageCompression(
            quality_lower=base_config.get('jpeg_quality_lower', 70),
            quality_upper=base_config.get('jpeg_quality_upper', 95),
            p=base_config.get('compression_p', 0.2)
        ))
        
        # Slight blur - robustness to different capture quality
        aug_list.append(A.GaussianBlur(
            blur_limit=(3, 5),
            p=base_config.get('blur_p', 0.1)
        ))
    
    # 4. Geometric augmentations - mild, preserves face structure
    if base_config.get('geometric_augmentations', True):
        # Small rotation - natural head tilt variation
        aug_list.append(A.Rotate(
            limit=base_config.get('rotation_limit', 6),
            border_mode=0,  # cv2.BORDER_CONSTANT
            p=base_config.get('rotation_p', 0.2)
        ))
        
        # Small scale variation
        aug_list.append(A.RandomScale(
            scale_limit=base_config.get('scale_limit', 0.1),
            p=base_config.get('scale_p', 0.1)
        ))
    
    if not aug_list:
        return None
        
    pipeline = A.Compose(aug_list)
    
    logger.info(f"Created base augmentation pipeline with {len(aug_list)} augmentations:")
    logger.info(f"  - Horizontal flip: {base_config.get('horizontal_flip', True)}")
    logger.info(f"  - Color augmentations: {base_config.get('color_augmentations', True)}")
    logger.info(f"  - Quality augmentations: {base_config.get('quality_augmentations', True)}")
    logger.info(f"  - Geometric augmentations: {base_config.get('geometric_augmentations', True)}")
    
    return pipeline


def _create_augmentation_transform(
    config: Dict[str, Any],
    deeplive_config: Dict[str, Any],
    logger: logging.Logger,
    data_config: Optional[Dict[str, Any]] = None
) -> Optional[Callable]:
    """
    Create augmentation transform based on config.
    
    Now composes:
    1. Base augmentations (color, quality, geometric) - prevents shortcut learning
    2. Landmark-specific augmentations (occlusion) - forces artifact learning
    
    Args:
        config: Main config (should have config['augmentation'] from experiment YAML)
        deeplive_config: DeepLive-specific config (data_config['deeplive'])
        logger: Logger
        data_config: Full data config (optional, for fallback)
        
    Returns:
        Transform function or None
        
    Config Resolution Order:
        1. config['augmentation'] - Main config (populated from experiment YAML)
        2. data_config['augmentation'] - Data config (if data_config passed)
        3. deeplive_config['augmentation'] - DeepLive-specific (legacy fallback)
    """
    # Try multiple config locations (W&B flattening can cause issues)
    aug_config = config.get('augmentation')
    config_source = 'config'
    if not aug_config and data_config:
        aug_config = data_config.get('augmentation')
        config_source = 'data_config'
    if not aug_config:
        aug_config = deeplive_config.get('augmentation', {})
        config_source = 'deeplive_config'
    
    if not aug_config:
        logger.info("No augmentation config found in any config source - skipping augmentation")
        logger.info("  Checked: config['augmentation'], data_config['augmentation'], deeplive_config['augmentation']")
        return None
    
    logger.info(f"Found augmentation config from '{config_source}': {aug_config}")
    
    # Create base augmentation pipeline (color, quality, geometric)
    base_pipeline = _create_base_augmentation_pipeline(aug_config, logger)
    
    aug_version = aug_config.get('version')
    
    if aug_version == 'landmark_occlusion':
        try:
            from data.augmentations.transforms import RegionBBoxOcclusion
            
            occlusion_prob = aug_config.get('occlusion_prob', 0.8)
            regions = aug_config.get('regions', ['left_eye', 'right_eye', 'nose', 'mouth'])
            num_regions = aug_config.get('num_regions', [1, 2])
            occlusion_type = aug_config.get('occlusion_type', 'mixed')
            
            logger.info(f"Creating landmark occlusion augmentation:")
            logger.info(f"  - Occlusion probability: {occlusion_prob}")
            logger.info(f"  - Regions: {regions}")
            logger.info(f"  - Num regions per image: {num_regions}")
            
            occlusion_transform = RegionBBoxOcclusion(
                regions=regions,
                num_regions=tuple(num_regions),
                occlusion_type=occlusion_type,
                p=occlusion_prob,
            )
            
            def transform(image, landmarks=None):
                # Apply base augmentations first (color, quality, geometric)
                if base_pipeline is not None:
                    import numpy as np
                    img_np = np.array(image) if not isinstance(image, np.ndarray) else image
                    result = base_pipeline(image=img_np)
                    image = result['image']
                
                # Then apply landmark occlusion
                if landmarks is None:
                    return image
                    
                import numpy as np
                img_np = np.array(image) if not isinstance(image, np.ndarray) else image
                result = occlusion_transform(image=img_np, landmarks=landmarks)
                return result['image']
            
            return transform
            
        except ImportError as e:
            logger.warning(f"Could not import RegionBBoxOcclusion: {e}")
            logger.warning("Falling back to base augmentations only")
            
            # Still return base augmentations if available
            if base_pipeline is not None:
                def transform(image, landmarks=None):
                    import numpy as np
                    img_np = np.array(image) if not isinstance(image, np.ndarray) else image
                    result = base_pipeline(image=img_np)
                    return result['image']
                return transform
            return None
    
    elif aug_version:
        # Try to use standard augmentation pipeline
        try:
            from data.augmentations import get_pipeline
            
            pipeline = get_pipeline(version=aug_version)
            logger.info(f"Using augmentation pipeline version: {aug_version}")
            
            def transform(image, landmarks=None):
                # Apply base augmentations first
                if base_pipeline is not None:
                    import numpy as np
                    img_np = np.array(image) if not isinstance(image, np.ndarray) else image
                    result = base_pipeline(image=img_np)
                    image = result['image']
                
                # Then apply version-specific pipeline
                import numpy as np
                img_np = np.array(image) if not isinstance(image, np.ndarray) else image
                result = pipeline(image=img_np)
                return result['image']
            
            return transform
            
        except Exception as e:
            logger.warning(f"Could not create augmentation pipeline: {e}")
            
            # Still return base augmentations if available
            if base_pipeline is not None:
                def transform(image, landmarks=None):
                    import numpy as np
                    img_np = np.array(image) if not isinstance(image, np.ndarray) else image
                    result = base_pipeline(image=img_np)
                    return result['image']
                return transform
            return None
    
    # No specific version, but base augmentations are available
    if base_pipeline is not None:
        logger.info("No specific augmentation version, using base augmentations only")
        def transform(image, landmarks=None):
            import numpy as np
            img_np = np.array(image) if not isinstance(image, np.ndarray) else image
            result = base_pipeline(image=img_np)
            return result['image']
        return transform
    
    return None


def _count_by_strategy(samples: List[Any]) -> Dict[str, int]:
    """Count samples by deepfake strategy/method."""
    from collections import Counter
    
    try:
        return dict(Counter(s.strategy for s in samples))
    except AttributeError:
        return {}


def build_deeplive_trainer_config(
    experiment_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build a config dict that matches what Trainer expects from DeepLive experiment config.
    
    This bridges the DeepLive experiment config format to the format
    expected by the Trainer class (same as train_sweep.py uses).
    
    Args:
        experiment_config: DeepLive experiment YAML config
        
    Returns:
        Config dict compatible with Trainer
    """
    config = {}
    
    # --- Model config ---
    backbone_config = experiment_config.get('backbone', {})
    config['model_name'] = 'effort'  # Always use effort detector
    config['backbone'] = backbone_config
    config['rank'] = experiment_config.get('rank', backbone_config.get('hidden_size', 768) - 1)
    config['pretrained'] = True
    
    # Backbone path resolution
    if backbone_config.get('huggingface_id'):
        config['backbone_path'] = backbone_config['huggingface_id']
        config['huggingface_id'] = backbone_config['huggingface_id']
    
    # Handle LAION/OpenCLIP backbones
    if backbone_config.get('source') == 'laion':
        config['backbone_source'] = 'laion'
        config['openclip_model_name'] = backbone_config.get('model_name', 'ViT-B-16')
        config['openclip_pretrained'] = backbone_config.get('pretrained', 'datacomp_xl_s13b_b90k')
    
    # --- ArcFace config ---
    config['use_arcface_head'] = experiment_config.get('use_arcface_head', False)
    if config['use_arcface_head']:
        config['arcface_s'] = experiment_config.get('arcface_s', 30.0)
        config['arcface_m'] = experiment_config.get('arcface_m', 0.28)
        config['s_start'] = experiment_config.get('s_start', config['arcface_s'])
        config['s_end'] = experiment_config.get('s_end', config['arcface_s'])
        config['anneal_steps'] = experiment_config.get('anneal_steps', 0)
        config['train_arcface'] = experiment_config.get('train_arcface', True)
    
    # --- Loss config ---
    config['use_focal_loss'] = experiment_config.get('use_focal_loss', False)
    if config['use_focal_loss']:
        config['focal_loss_gamma'] = experiment_config.get('focal_loss_gamma', 2.0)
        config['focal_loss_alpha'] = experiment_config.get('focal_loss_alpha', None)
    
    # --- Optimizer config ---
    config['optimizer'] = {
        'type': 'adam',
        'adam': {
            'lr': experiment_config.get('learning_rate', 1e-4),
            'weight_decay': experiment_config.get('weight_decay', 0.05),
            'eps': experiment_config.get('optimizer_eps', 1e-8),
            'beta1': 0.9,
            'beta2': 0.999,
            'amsgrad': False,
        }
    }
    config['lambda_reg'] = experiment_config.get('lambda_reg', 1.0)
    
    # --- Scheduler config ---
    config['lr_scheduler'] = experiment_config.get('lr_scheduler', None)
    config['total_training_steps'] = experiment_config.get('total_training_steps', 10000)
    config['lr_scheduler_warmup_steps'] = experiment_config.get('lr_scheduler_warmup_steps', 1000)
    
    # --- Training config ---
    config['nEpochs'] = experiment_config.get('nEpochs', 50)
    config['start_epoch'] = 0
    config['manualSeed'] = experiment_config.get('seed', 737)
    config['seed'] = experiment_config.get('seed', 737)
    
    # Step-based training
    config['max_train_steps'] = experiment_config.get('max_train_steps', None)
    config['evaluate_every_steps'] = experiment_config.get('evaluate_every_steps', None)
    
    # --- Early stopping ---
    config['early_stopping_enabled'] = experiment_config.get('early_stopping_enabled', True)
    config['early_stopping_patience'] = experiment_config.get('early_stopping_patience', 10)
    config['early_stopping_min_delta'] = experiment_config.get('early_stopping_min_delta', 0.001)
    
    # --- Gradient clipping ---
    config['gradient_clip_val'] = experiment_config.get('gradient_clip_val', None)
    
    # --- Normalization (CLIP defaults) ---
    config['mean'] = experiment_config.get('mean', [0.48145466, 0.4578275, 0.40821073])
    config['std'] = experiment_config.get('std', [0.26862954, 0.26130258, 0.27577711])
    config['resolution'] = experiment_config.get('backbone', {}).get('resolution', 224)
    
    # --- Batch sizes ---
    config['train_batchSize'] = experiment_config.get('frames_per_batch', 32)
    config['test_batchSize'] = experiment_config.get('test_batch_size', 32)
    
    # --- Dataloader params (for compatibility) ---
    config['dataloader_params'] = {
        'strategy': 'deeplive',
        'frames_per_batch': experiment_config.get('frames_per_batch', 32),
        'frames_per_video': experiment_config.get('frames_per_video', 8),
        'num_workers': experiment_config.get('num_workers', 4),
        'prefetch_factor': experiment_config.get('prefetch_factor', 2),
    }
    
    # --- Curriculum / Lesson Gate (if specified) ---
    if 'lesson_gate' in experiment_config:
        config['lesson_gate'] = experiment_config['lesson_gate']
    
    # --- GCS Checkpointing ---
    checkpointing = experiment_config.get('checkpointing', {})
    config['gcs_checkpoint_prefix'] = checkpointing.get('gcs_prefix', '')
    config['keep_top_n_checkpoints'] = checkpointing.get('keep_last_n', 3)
    
    # --- Misc ---
    config['ddp'] = False  # DeepLive doesn't use DDP for now
    config['cuda'] = True
    config['cudnn'] = True
    config['save_ckpt'] = True
    config['local_rank'] = 0
    
    # --- Group DRO (disabled for DeepLive) ---
    config['use_group_dro'] = experiment_config.get('use_group_dro', False)
    
    # --- W&B logging config ---
    wandb_config = experiment_config.get('wandb', {})
    config['wandb'] = {
        'log_progress_steps': wandb_config.get('log_progress_steps', 50),
    }
    
    return config
