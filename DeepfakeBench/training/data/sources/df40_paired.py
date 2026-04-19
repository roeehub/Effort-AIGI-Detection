"""
DF40 Paired Data Source

This module provides the data pipeline for the DF40 paired dataset stored in GCS.
It creates paired real/fake frame batches using the pre-computed pair matching JSON.

Key differences from DeepLive:
- NO LANDMARKS available - landmark-based augmentation is disabled
- 32 frames per pair (vs 16 in DeepLive)
- Multiple deepfake methods (8 methods, ~5,379 pairs total)

Usage:
    from data.sources import create_data_pipeline
    
    # In config:
    data_config['data_source'] = 'df40_paired'
    data_config['df40_paired'] = {
        'pair_json': 'dataset/df40_pairs/df40-pair-matching.json',
        'gcs_bucket': 'df40-frames-recropped-rfa85',
        'sampling_mode': 'sparse',
        'methods': ['simswap', 'facedancer', 'blendface'],  # or None for all
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


class DF40ValidationAdapter:
    """
    Adapter that wraps a DF40 DataLoader to provide the interface expected by trainer.py.
    
    The trainer expects validation loaders to have:
    - `videos_by_method`: Dict[str, List] mapping methods to video lists
    - `keys()`: Method to iterate over method names
    - `__getitem__(method)`: Get dataloader for a specific method
    - Truthiness check without calling len()
    """
    
    def __init__(
        self,
        dataloader: DataLoader,
        samples: List[Any],
        strategy_name: str = 'df40_paired'
    ):
        self._dataloader = dataloader
        self._samples = samples
        self._strategy_name = strategy_name
        
        # Build videos_by_method mapping
        self.videos_by_method = {strategy_name: samples}
    
    def keys(self):
        """Return method names."""
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
        """Return the underlying dataset."""
        return self._dataloader.dataset


@register_data_source('df40_paired')
def create_df40_paired_pipeline(
    config: Dict[str, Any],
    data_config: Dict[str, Any],
    logger: logging.Logger,
    **kwargs
) -> DataPipelineResult:
    """
    Create data pipeline from DF40 paired dataset.
    
    This function:
    1. Loads pair matching JSON
    2. Filters by methods if specified
    3. Splits into train/val/test sets
    4. Creates dataloaders with paired real/fake frames
    5. Sets up augmentation (WITHOUT landmarks!)
    
    Args:
        config: Main training configuration
        data_config: Data-specific configuration including:
            - df40_paired: DF40-specific settings
                - pair_json: Path to pair matching JSON
                - gcs_bucket: GCS bucket name
                - methods: List of methods to include (or None for all)
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
    from dataset.df40_paired_dataset import DF40PairedDataset
    from data.batching.df40_paired import DF40PairedBatchingStrategy, DF40PairedBatchingConfig
    
    df40_config = data_config.get('df40_paired', {})
    seed = config.get('manualSeed', config.get('seed', 737))
    
    logger.info("=" * 70)
    logger.info("DF40 Paired Data Source: Initializing")
    logger.info("=" * 70)
    
    # Get pair JSON path
    pair_json_path = df40_config.get('pair_json', 'dataset/df40_pairs/df40-pair-matching.json')
    
    # Handle relative paths
    if not os.path.isabs(pair_json_path):
        # Try relative to training directory
        training_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        pair_json_path = os.path.join(training_dir, pair_json_path)
    
    if not os.path.exists(pair_json_path):
        raise FileNotFoundError(
            f"DF40 pair JSON not found: {pair_json_path}\n"
            f"Please ensure df40-pair-matching.json exists at the specified path."
        )
    
    # Create DF40 dataset
    gcs_bucket = df40_config.get('gcs_bucket', 'df40-frames-recropped-rfa85')
    gcs_project = df40_config.get('gcs_project', os.environ.get('GOOGLE_CLOUD_PROJECT', 'train-cvit2'))
    methods = df40_config.get('methods')  # None = all methods
    
    logger.info(f"Creating DF40 Paired dataset:")
    logger.info(f"  - Pair JSON: {pair_json_path}")
    logger.info(f"  - GCS bucket: {gcs_bucket}")
    logger.info(f"  - GCS project: {gcs_project}")
    logger.info(f"  - Methods filter: {methods if methods else 'ALL'}")
    logger.info("")
    logger.info("⚠️  WARNING: DF40 has NO LANDMARKS")
    logger.info("⚠️  Landmark-based augmentation (occlusion) will be DISABLED")
    logger.info("")
    
    dataset = DF40PairedDataset(
        pair_json_path=pair_json_path,
        bucket_name=gcs_bucket,
        gcs_project=gcs_project,
        methods=methods,
    )
    
    # Discover samples
    logger.info("Discovering samples from pair JSON...")
    samples = dataset.discover_samples()
    logger.info(f"Discovered {len(samples)} paired samples")
    
    if len(samples) == 0:
        raise ValueError(
            f"No samples found in DF40 pair JSON: {pair_json_path}\n"
            f"Methods filter: {methods}"
        )
    
    # Split samples
    train_split = df40_config.get('train_split', 0.8)
    val_split = df40_config.get('val_split', 0.1)
    
    train_samples, val_samples, test_samples = _split_samples(
        samples, train_split, val_split, seed, logger
    )
    
    # Create augmentation transform (WITHOUT landmarks)
    transform = kwargs.get('transform')
    if transform is None:
        transform = _create_augmentation_transform_no_landmarks(
            config, df40_config, logger, data_config
        )
    
    # Determine num_workers
    no_multiprocessing = os.environ.get('NO_MULTIPROCESSING', '').lower() in ('1', 'true', 'yes')
    device_is_cpu = not torch.cuda.is_available()
    
    if no_multiprocessing or device_is_cpu:
        num_workers = 0
        logger.info("Using num_workers=0 (multiprocessing disabled or CPU-only)")
    else:
        num_workers = df40_config.get('num_workers', 4)
    
    # Create batching config
    batching_config = DF40PairedBatchingConfig(
        batch_size=config.get('frames_per_batch', df40_config.get('frames_per_batch', 32)),
        num_workers=num_workers,
        prefetch_factor=config.get('prefetch_factor', 2) if num_workers > 0 else None,
        frame_sampling=df40_config.get('sampling_mode', 'sparse'),
        sparse_indices=df40_config.get('anchor_indices', [0, 4, 8, 12, 16, 20, 24, 28]),
        methods=methods,
    )
    
    # Create batching strategy
    strategy = DF40PairedBatchingStrategy(
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
    
    # Wrap validation loaders with adapter
    val_loader = DF40ValidationAdapter(val_loader_raw, val_samples, 'df40_val')
    test_loader = DF40ValidationAdapter(test_loader_raw, test_samples, 'df40_holdout')
    
    logger.info("Validation loaders wrapped with DF40ValidationAdapter for trainer compatibility")
    
    # Compute statistics
    method_counts = _count_by_method(samples)
    data_stats = {
        'total_samples': len(samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples),
        'train_split': train_split,
        'val_split': val_split,
        'gcs_bucket': gcs_bucket,
        'pair_json': pair_json_path,
        'sampling_mode': batching_config.frame_sampling,
        'frames_per_sample': len(batching_config.sparse_indices) * 2,
        'methods': method_counts,
        'methods_count': len(method_counts),
        'has_landmarks': False,  # DF40 has no landmarks!
        'discovered_videos': len(samples),
        'discovered_methods': len(method_counts),
        'unbalanced_train_count': len(train_samples),
        'unbalanced_val_count': len(val_samples) + len(test_samples),
        'train_video_count': len(train_samples),
        'train_frame_count': len(train_samples) * len(batching_config.sparse_indices) * 2,
        'val_video_count': len(val_samples),
        'val_frame_count': len(val_samples) * len(batching_config.sparse_indices) * 2,
    }
    
    logger.info(f"DF40 Paired pipeline created:")
    logger.info(f"  - Train samples: {len(train_samples)}")
    logger.info(f"  - Val samples: {len(val_samples)}")
    logger.info(f"  - Test samples: {len(test_samples)}")
    logger.info(f"  - Methods: {list(method_counts.keys())}")
    logger.info(f"  - Total train frames: {data_stats['train_frame_count']}")
    
    return DataPipelineResult(
        train_loader=train_loader,
        val_in_dist_loader=val_loader,
        val_holdout_loader=test_loader,
        train_samples=train_samples,
        data_stats=data_stats,
        ood_loader=None,
    )


def _split_samples(
    samples: List[Any],
    train_split: float,
    val_split: float,
    seed: int,
    logger: logging.Logger
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Split samples into train/val/test sets BY IDENTITY.
    
    CRITICAL: This splits by target_identity to prevent data leakage.
    No identity will appear in multiple splits, ensuring the model cannot
    memorize identities seen during training to cheat on validation.
    
    The split is done at the identity level, then all pairs for each identity
    are assigned to that identity's split.
    """
    from collections import defaultdict
    
    rng = random.Random(seed)
    
    # Group samples by target_identity (the real person)
    by_identity = defaultdict(list)
    for sample in samples:
        by_identity[sample.target_identity].append(sample)
    
    # Get list of unique identities and shuffle
    identities = list(by_identity.keys())
    rng.shuffle(identities)
    
    # Split identities (not pairs!)
    n_identities = len(identities)
    n_train_ids = int(n_identities * train_split)
    n_val_ids = int(n_identities * val_split)
    
    train_identities = set(identities[:n_train_ids])
    val_identities = set(identities[n_train_ids:n_train_ids + n_val_ids])
    test_identities = set(identities[n_train_ids + n_val_ids:])
    
    # Assign ALL pairs for each identity to that identity's split
    train_samples = []
    val_samples = []
    test_samples = []
    
    for sample in samples:
        identity = sample.target_identity
        if identity in train_identities:
            train_samples.append(sample)
        elif identity in val_identities:
            val_samples.append(sample)
        else:
            test_samples.append(sample)
    
    # Final shuffle to mix methods within each split
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)
    
    # Log split statistics
    n_total = len(samples)
    logger.info(f"Data split BY IDENTITY (seed={seed}):")
    logger.info(f"  - Total unique identities: {n_identities}")
    logger.info(f"  - Train identities: {len(train_identities)} ({len(train_identities)/n_identities*100:.1f}%)")
    logger.info(f"  - Val identities: {len(val_identities)} ({len(val_identities)/n_identities*100:.1f}%)")
    logger.info(f"  - Test identities: {len(test_identities)} ({len(test_identities)/n_identities*100:.1f}%)")
    logger.info(f"  - Train samples: {len(train_samples)} ({len(train_samples)/n_total*100:.1f}%)")
    logger.info(f"  - Val samples: {len(val_samples)} ({len(val_samples)/n_total*100:.1f}%)")
    logger.info(f"  - Test samples: {len(test_samples)} ({len(test_samples)/n_total*100:.1f}%)")
    
    # Log method distribution per split to verify stratification
    def _count_methods(sample_list):
        counts = defaultdict(int)
        for s in sample_list:
            counts[s.method] += 1
        return dict(counts)
    
    train_methods = _count_methods(train_samples)
    val_methods = _count_methods(val_samples)
    test_methods = _count_methods(test_samples)
    
    logger.info(f"  - Train methods: {train_methods}")
    logger.info(f"  - Val methods: {val_methods}")
    logger.info(f"  - Test methods: {test_methods}")
    
    # Sanity check: verify no identity overlap
    train_ids_check = set(s.target_identity for s in train_samples)
    val_ids_check = set(s.target_identity for s in val_samples)
    test_ids_check = set(s.target_identity for s in test_samples)
    
    train_val_overlap = train_ids_check & val_ids_check
    train_test_overlap = train_ids_check & test_ids_check
    val_test_overlap = val_ids_check & test_ids_check
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        logger.error(f"IDENTITY LEAKAGE DETECTED!")
        logger.error(f"  Train-Val overlap: {len(train_val_overlap)} identities")
        logger.error(f"  Train-Test overlap: {len(train_test_overlap)} identities")
        logger.error(f"  Val-Test overlap: {len(val_test_overlap)} identities")
        raise ValueError("Identity leakage between splits - this should never happen!")
    else:
        logger.info(f"  ✓ No identity overlap between splits (leakage check passed)")
    
    return train_samples, val_samples, test_samples


def _create_augmentation_transform_no_landmarks(
    config: Dict[str, Any],
    df40_config: Dict[str, Any],
    logger: logging.Logger,
    data_config: Optional[Dict[str, Any]] = None
) -> Optional[Callable]:
    """
    Create augmentation transform for DF40 (NO LANDMARKS).
    
    This creates only base augmentations (color, quality, geometric).
    Landmark-based augmentation (occlusion) is NOT available for DF40.
    
    Args:
        config: Main config
        df40_config: DF40-specific config
        logger: Logger
        data_config: Full data config (optional)
        
    Returns:
        Transform function that accepts (image, landmarks=None)
    """
    # Try multiple config locations
    aug_config = config.get('augmentation')
    config_source = 'config'
    if not aug_config and data_config:
        aug_config = data_config.get('augmentation')
        config_source = 'data_config'
    if not aug_config:
        aug_config = df40_config.get('augmentation')
        config_source = 'df40_config'
    
    if not aug_config:
        logger.info("No augmentation config found - using base augmentations only")
        aug_config = {}
    
    logger.info(f"Found augmentation config from '{config_source}'")
    
    # Check if landmark occlusion was requested
    aug_version = aug_config.get('version')
    if aug_version == 'landmark_occlusion':
        logger.warning("")
        logger.warning("=" * 70)
        logger.warning("⚠️  LANDMARK OCCLUSION REQUESTED BUT DF40 HAS NO LANDMARKS!")
        logger.warning("⚠️  Falling back to base augmentations only.")
        logger.warning("⚠️  To use landmark occlusion, use DeepLive data source instead.")
        logger.warning("=" * 70)
        logger.warning("")
    
    # Create base augmentation pipeline only
    base_pipeline = _create_base_augmentation_pipeline(aug_config, logger)
    
    if base_pipeline is not None:
        def transform_fn(image, landmarks=None):
            """Apply base augmentations. Ignores landmarks (DF40 has none)."""
            # landmarks is always None for DF40, but we accept it for API compatibility
            import numpy as np
            if isinstance(image, np.ndarray):
                augmented = base_pipeline(image=image)
                return augmented['image']
            return image
        
        logger.info("Created augmentation transform (base augmentations only, no landmarks)")
        return transform_fn
    
    return None


def _create_base_augmentation_pipeline(aug_config: Dict[str, Any], logger: logging.Logger):
    """
    Create base augmentation pipeline (color, quality, geometric).
    
    Same as DeepLive but without any landmark-dependent augmentations.
    """
    try:
        import albumentations as A
    except ImportError:
        logger.warning("albumentations not installed - skipping base augmentations")
        return None
    
    if aug_config.get('disable_base_augmentations', False):
        logger.info("Base augmentations disabled by config")
        return None
    
    base_config = aug_config.get('base', {})
    
    aug_list = []
    
    # 1. Horizontal flip
    if base_config.get('horizontal_flip', True):
        aug_list.append(A.HorizontalFlip(p=0.5))
        
    # 2. Color augmentations
    if base_config.get('color_augmentations', True):
        aug_list.append(A.RandomBrightnessContrast(
            brightness_limit=base_config.get('brightness_limit', 0.15),
            contrast_limit=base_config.get('contrast_limit', 0.15),
            p=base_config.get('brightness_contrast_p', 0.4)
        ))
        
        aug_list.append(A.HueSaturationValue(
            hue_shift_limit=base_config.get('hue_shift_limit', 15),
            sat_shift_limit=base_config.get('sat_shift_limit', 20),
            val_shift_limit=base_config.get('val_shift_limit', 15),
            p=base_config.get('hsv_p', 0.3)
        ))
        
        aug_list.append(A.RGBShift(
            r_shift_limit=base_config.get('rgb_shift_limit', 10),
            g_shift_limit=base_config.get('rgb_shift_limit', 10),
            b_shift_limit=base_config.get('rgb_shift_limit', 10),
            p=base_config.get('rgb_shift_p', 0.2)
        ))
    
    # 3. Quality augmentations
    if base_config.get('quality_augmentations', True):
        aug_list.append(A.ImageCompression(
            quality_lower=base_config.get('jpeg_quality_lower', 70),
            quality_upper=base_config.get('jpeg_quality_upper', 95),
            p=base_config.get('compression_p', 0.2)
        ))
        
        aug_list.append(A.GaussianBlur(
            blur_limit=(3, 5),
            p=base_config.get('blur_p', 0.1)
        ))
    
    # 4. Geometric augmentations
    if base_config.get('geometric_augmentations', True):
        aug_list.append(A.Rotate(
            limit=base_config.get('rotation_limit', 6),
            border_mode=0,
            p=base_config.get('rotation_p', 0.2)
        ))
    
    if not aug_list:
        return None
        
    pipeline = A.Compose(aug_list)
    
    logger.info(f"Created base augmentation pipeline with {len(aug_list)} augmentations:")
    logger.info(f"  - Horizontal flip: {base_config.get('horizontal_flip', True)}")
    logger.info(f"  - Color augmentations: {base_config.get('color_augmentations', True)}")
    logger.info(f"  - Quality augmentations: {base_config.get('quality_augmentations', True)}")
    logger.info(f"  - Geometric augmentations: {base_config.get('geometric_augmentations', True)}")
    logger.info(f"  - Landmark occlusion: DISABLED (no landmarks in DF40)")
    
    return pipeline


def _count_by_method(samples: List[Any]) -> Dict[str, int]:
    """Count samples by deepfake method."""
    from collections import Counter
    return dict(Counter(s.method for s in samples))


def build_df40_trainer_config(
    experiment_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build a config dict that matches what Trainer expects from DF40 experiment config.
    
    This bridges the DF40 experiment config format to the format
    expected by the Trainer class.
    
    Args:
        experiment_config: DF40 experiment YAML config
        
    Returns:
        Config dict compatible with Trainer
    """
    config = {}
    
    # --- Model config ---
    backbone_config = experiment_config.get('backbone', {})
    config['model_name'] = 'effort'
    config['backbone'] = backbone_config
    config['rank'] = experiment_config.get('rank', backbone_config.get('hidden_size', 768) - 1)
    config['pretrained'] = True
    
    if backbone_config.get('huggingface_id'):
        config['pretrained_path'] = backbone_config['huggingface_id']
    
    if backbone_config.get('source') == 'laion':
        config['clip_variant'] = backbone_config.get('variant', 'ViT-B-16')
        config['clip_pretrained'] = backbone_config.get('pretrained', 'datacomp_xl_s13b_b90k')
    
    # --- ArcFace config ---
    config['use_arcface_head'] = experiment_config.get('use_arcface_head', False)
    if config['use_arcface_head']:
        config['arcface_s'] = experiment_config.get('arcface_s', 30.0)
        config['arcface_m'] = experiment_config.get('arcface_m', 0.28)
        config['s_start'] = experiment_config.get('s_start', 10.0)
        config['s_end'] = experiment_config.get('s_end', 30.0)
        config['anneal_steps'] = experiment_config.get('anneal_steps', 2000)
    
    # --- Loss config ---
    config['use_focal_loss'] = experiment_config.get('use_focal_loss', False)
    if config['use_focal_loss']:
        config['focal_alpha'] = experiment_config.get('focal_alpha', 0.25)
        config['focal_gamma'] = experiment_config.get('focal_gamma', 2.0)
    
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
    
    # --- Dataloader params ---
    config['dataloader_params'] = {
        'strategy': 'df40_paired',
        'frames_per_batch': experiment_config.get('frames_per_batch', 32),
        'frames_per_video': experiment_config.get('frames_per_video', 8),
        'num_workers': experiment_config.get('num_workers', 4),
        'prefetch_factor': experiment_config.get('prefetch_factor', 2),
    }
    
    # --- GCS Checkpointing ---
    checkpointing = experiment_config.get('checkpointing', {})
    config['gcs_checkpoint_prefix'] = checkpointing.get('gcs_prefix', '')
    config['keep_top_n_checkpoints'] = checkpointing.get('keep_last_n', 3)
    
    # --- Misc ---
    config['ddp'] = False
    config['cuda'] = True
    config['cudnn'] = True
    config['save_ckpt'] = True
    config['local_rank'] = 0
    config['use_group_dro'] = experiment_config.get('use_group_dro', False)
    
    # --- W&B logging config ---
    wandb_config = experiment_config.get('wandb', {})
    config['wandb'] = {
        'log_progress_steps': wandb_config.get('log_progress_steps', 50),
    }
    
    return config
