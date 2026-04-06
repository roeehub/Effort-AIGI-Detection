#!/usr/bin/env python3
"""
# =============================================================================
# ⚠️  DEPRECATED - January 1, 2026
# =============================================================================
# This script is DEPRECATED and will be removed in a future cleanup.
# 
# USE INSTEAD:
#   python train_sweep.py --param-config experiments/deeplive_vit_B16.yaml
#
# The unified training script (train_sweep.py) now supports DeepLive via
# the data_source: deeplive configuration option.
#
# See: orgenize_training/UNIFIED_TRAINING_COMPLETE.md
# =============================================================================

DeepLive Training Script (Refactored) - DEPRECATED

This script trains the deepfake detector using the DeepLive dataset.
It uses the SAME Trainer class and pipeline as train_sweep.py, just with
DeepLive-specific data loading.

This is the CORRECT approach: new data sources should plug into the existing
training pipeline, not create a separate one.

Usage:
    # Local training with a specific experiment config
    python train_deeplive.py --config experiments/deeplive_vit_B16.yaml
    
    # Training with W&B logging
    python train_deeplive.py --config experiments/deeplive_vit_B16.yaml --wandb
    
    # Dry run (no actual training)
    python train_deeplive.py --config experiments/deeplive_vit_B16.yaml --dry-run
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# Setup path for imports
TRAINING_DIR = Path(__file__).parent
sys.path.insert(0, str(TRAINING_DIR))

# Import from the refactored modules (same as train_sweep.py)
from detectors import DETECTOR
from trainer.trainer import Trainer
from logger import create_logger

# Import utilities from refactored utils module
from utils import (
    init_seed,
    choose_optimizer,
    choose_scheduler,
    choose_metric,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
module_logger = logging.getLogger(__name__)


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    module_logger.info(f"Loaded config from {config_path}")
    return config


def apply_wandb_overrides_deeplive(
    experiment_config: Dict[str, Any],
    wandb_config: Any,
    logger: Any = None
) -> None:
    """
    Apply W&B config overrides to DeepLive experiment config (in-place).
    
    This allows sweeps to override hyperparameters dynamically.
    Mirrors the behavior of apply_all_wandb_overrides() from train_sweep.py.
    
    Args:
        experiment_config: The experiment config dict to update in-place
        wandb_config: W&B config object (from wandb.config or sweep agent)
        logger: Optional logger for debug output
    """
    log = logger.info if logger else print
    
    # --- Learning rate and optimizer params ---
    if wandb_config.get('learning_rate') is not None:
        experiment_config['learning_rate'] = wandb_config.get('learning_rate')
        log(f"W&B override: learning_rate = {experiment_config['learning_rate']}")
    
    if wandb_config.get('weight_decay') is not None:
        experiment_config['weight_decay'] = wandb_config.get('weight_decay')
        log(f"W&B override: weight_decay = {experiment_config['weight_decay']}")
    
    # --- Batch size params ---
    if wandb_config.get('frames_per_batch') is not None:
        experiment_config['frames_per_batch'] = wandb_config.get('frames_per_batch')
        log(f"W&B override: frames_per_batch = {experiment_config['frames_per_batch']}")
    
    if wandb_config.get('frames_per_video') is not None:
        experiment_config['frames_per_video'] = wandb_config.get('frames_per_video')
        log(f"W&B override: frames_per_video = {experiment_config['frames_per_video']}")
    
    # --- ArcFace params ---
    if wandb_config.get('use_arcface_head') is not None:
        experiment_config['use_arcface_head'] = wandb_config.get('use_arcface_head')
        log(f"W&B override: use_arcface_head = {experiment_config['use_arcface_head']}")
    
    if wandb_config.get('arcface_s') is not None:
        experiment_config['arcface_s'] = wandb_config.get('arcface_s')
        log(f"W&B override: arcface_s = {experiment_config['arcface_s']}")
    
    if wandb_config.get('arcface_m') is not None:
        experiment_config['arcface_m'] = wandb_config.get('arcface_m')
        log(f"W&B override: arcface_m = {experiment_config['arcface_m']}")
    
    if wandb_config.get('s_start') is not None:
        experiment_config['s_start'] = wandb_config.get('s_start')
        log(f"W&B override: s_start = {experiment_config['s_start']}")
    
    if wandb_config.get('s_end') is not None:
        experiment_config['s_end'] = wandb_config.get('s_end')
        log(f"W&B override: s_end = {experiment_config['s_end']}")
    
    if wandb_config.get('anneal_steps') is not None:
        experiment_config['anneal_steps'] = wandb_config.get('anneal_steps')
        log(f"W&B override: anneal_steps = {experiment_config['anneal_steps']}")
    
    # --- Loss params ---
    if wandb_config.get('use_focal_loss') is not None:
        experiment_config['use_focal_loss'] = wandb_config.get('use_focal_loss')
        log(f"W&B override: use_focal_loss = {experiment_config['use_focal_loss']}")
    
    if wandb_config.get('focal_loss_gamma') is not None:
        experiment_config['focal_loss_gamma'] = wandb_config.get('focal_loss_gamma')
        log(f"W&B override: focal_loss_gamma = {experiment_config['focal_loss_gamma']}")
    
    # --- Training params ---
    if wandb_config.get('max_train_steps') is not None:
        experiment_config['max_train_steps'] = wandb_config.get('max_train_steps')
        log(f"W&B override: max_train_steps = {experiment_config['max_train_steps']}")
    
    if wandb_config.get('nEpochs') is not None:
        experiment_config['nEpochs'] = wandb_config.get('nEpochs')
        log(f"W&B override: nEpochs = {experiment_config['nEpochs']}")
    
    if wandb_config.get('evaluate_every_steps') is not None:
        experiment_config['evaluate_every_steps'] = wandb_config.get('evaluate_every_steps')
        log(f"W&B override: evaluate_every_steps = {experiment_config['evaluate_every_steps']}")
    
    # --- Early stopping ---
    if wandb_config.get('early_stopping_patience') is not None:
        experiment_config['early_stopping_patience'] = wandb_config.get('early_stopping_patience')
        log(f"W&B override: early_stopping_patience = {experiment_config['early_stopping_patience']}")
    
    # --- Backbone params ---
    if 'backbone' not in experiment_config:
        experiment_config['backbone'] = {}
    
    if wandb_config.get('rank') is not None:
        experiment_config['rank'] = wandb_config.get('rank')
        log(f"W&B override: rank = {experiment_config['rank']}")
    
    if wandb_config.get('resolution') is not None:
        experiment_config['backbone']['resolution'] = wandb_config.get('resolution')
        log(f"W&B override: backbone.resolution = {experiment_config['backbone']['resolution']}")
    
    # --- DeepLive-specific params ---
    if 'deeplive' not in experiment_config:
        experiment_config['deeplive'] = {}
    
    if wandb_config.get('sampling_mode') is not None:
        experiment_config['deeplive']['sampling_mode'] = wandb_config.get('sampling_mode')
        log(f"W&B override: deeplive.sampling_mode = {experiment_config['deeplive']['sampling_mode']}")
    
    if wandb_config.get('train_split') is not None:
        experiment_config['deeplive']['train_split'] = wandb_config.get('train_split')
        log(f"W&B override: deeplive.train_split = {experiment_config['deeplive']['train_split']}")
    
    # --- Augmentation params ---
    if wandb_config.get('occlusion_prob') is not None:
        if 'augmentation' not in experiment_config:
            experiment_config['augmentation'] = {}
        experiment_config['augmentation']['occlusion_prob'] = wandb_config.get('occlusion_prob')
        log(f"W&B override: augmentation.occlusion_prob = {experiment_config['augmentation']['occlusion_prob']}")
    
    # --- Scheduler params ---
    if wandb_config.get('lr_scheduler') is not None:
        experiment_config['lr_scheduler'] = wandb_config.get('lr_scheduler')
        log(f"W&B override: lr_scheduler = {experiment_config['lr_scheduler']}")
    
    if wandb_config.get('lr_scheduler_warmup_steps') is not None:
        experiment_config['lr_scheduler_warmup_steps'] = wandb_config.get('lr_scheduler_warmup_steps')
        log(f"W&B override: lr_scheduler_warmup_steps = {experiment_config['lr_scheduler_warmup_steps']}")


def build_trainer_config(experiment_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a config dict that matches what Trainer expects.
    
    This bridges the DeepLive experiment config format to the format
    expected by the Trainer class (same as train_sweep.py uses).
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


def create_deeplive_dataloaders(
    experiment_config: Dict[str, Any],
    logger
) -> Tuple[Any, Any, Any, List]:
    """
    Create train, val, and test dataloaders for DeepLive dataset.
    
    Returns:
        train_loader, val_loader, test_loader, train_samples
    """
    from dataset.deeplive_dataset import DeepLiveDataset
    from data.batching.deeplive import DeepLiveBatchingStrategy, DeepLiveBatchingConfig
    from data.augmentations.transforms import RegionBBoxOcclusion
    
    deeplive_config = experiment_config.get('deeplive', {})
    seed = experiment_config.get('seed', 737)
    
    # Create dataset
    logger.info("Creating DeepLive dataset...")
    dataset = DeepLiveDataset(
        bucket_name=deeplive_config.get('gcs_bucket', 'live-deepfake-methods-real-and-fake-frames-cropped'),
        gcs_project=os.environ.get('GOOGLE_CLOUD_PROJECT', 'train-cvit2'),
        use_landmarks=True,
    )
    
    # Discover samples
    logger.info("Discovering samples from GCS...")
    samples = dataset.discover_samples()
    logger.info(f"Discovered {len(samples)} samples")
    
    if len(samples) == 0:
        raise ValueError("No samples discovered from GCS bucket!")
    
    # Split samples (80/10/10)
    train_split = deeplive_config.get('train_split', 0.8)
    val_split = deeplive_config.get('val_split', 0.1)
    
    # Shuffle samples with seed
    rng = random.Random(seed)
    shuffled_samples = samples.copy()
    rng.shuffle(shuffled_samples)
    
    n_total = len(shuffled_samples)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    train_samples = shuffled_samples[:n_train]
    val_samples = shuffled_samples[n_train:n_train + n_val]
    test_samples = shuffled_samples[n_train + n_val:]
    
    logger.info(f"Data split:")
    logger.info(f"  - Train: {len(train_samples)} samples ({len(train_samples)/n_total*100:.1f}%)")
    logger.info(f"  - Val: {len(val_samples)} samples ({len(val_samples)/n_total*100:.1f}%)")
    logger.info(f"  - Test: {len(test_samples)} samples ({len(test_samples)/n_total*100:.1f}%)")
    
    # Create augmentation transform
    aug_config = experiment_config.get('augmentation', {})
    transform = None
    
    if aug_config.get('version') == 'landmark_occlusion':
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
            if landmarks is None:
                return image
            result = occlusion_transform(image=image, landmarks=landmarks)
            return result['image']
    
    # Determine num_workers
    no_multiprocessing = os.environ.get('NO_MULTIPROCESSING', '').lower() in ('1', 'true', 'yes')
    device_is_cpu = not torch.cuda.is_available()
    
    if no_multiprocessing or device_is_cpu:
        num_workers = 0
        logger.info("Using num_workers=0 (multiprocessing disabled)")
    else:
        num_workers = experiment_config.get('num_workers', 4)
    
    # Create batching config
    batching_config = DeepLiveBatchingConfig(
        batch_size=experiment_config.get('frames_per_batch', 32),
        num_workers=num_workers,
        prefetch_factor=experiment_config.get('prefetch_factor', 2) if num_workers > 0 else None,
        frame_sampling=deeplive_config.get('sampling_mode', 'sparse'),
        sparse_indices=deeplive_config.get('anchor_indices', [0, 2, 4, 6, 8, 10, 12, 14]),
    )
    
    # Create batching strategy
    strategy = DeepLiveBatchingStrategy(
        config=experiment_config,
        data_config={'deeplive_data': deeplive_config},
        strategy_config=batching_config,
        dataset=dataset,
        transform=transform,
    )
    
    # Create dataloaders
    train_loader = strategy.create_train_loader(train_samples)
    val_loader = strategy.create_validation_loader(val_samples, mode='test')
    test_loader = strategy.create_validation_loader(test_samples, mode='test')
    
    return train_loader, val_loader, test_loader, train_samples


def main():
    parser = argparse.ArgumentParser(description='Train DeepLive detector (using Trainer)')
    parser.add_argument('--config', '--param-config', type=str, required=True,
                        dest='config',
                        help='Path to experiment config YAML')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run setup only, no actual training')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable W&B logging')
    parser.add_argument('--wandb-project', type=str, default='deeplive-experiments',
                        help='W&B project name')
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Load experiment config
    experiment_config = load_experiment_config(args.config)
    
    # Build Trainer-compatible config
    config = build_trainer_config(experiment_config)
    
    # Initialize seed
    init_seed(config)
    
    # Setup logger (same as train_sweep.py)
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=experiment_config.get('name', 'deeplive-experiment'),
            config=experiment_config,
        )
        log_dir = wandb_run.dir
        
        # Apply W&B config overrides (for sweeps)
        # This allows sweep agents to override hyperparameters
        apply_wandb_overrides_deeplive(experiment_config, wandb.config)
        
        # Rebuild config after overrides
        config = build_trainer_config(experiment_config)
        
        # Update W&B config with final config
        wandb.config.update({'final_trainer_config': config}, allow_val_change=True)
    else:
        log_dir = './logs/deeplive'
        os.makedirs(log_dir, exist_ok=True)
    
    logger = create_logger(os.path.join(log_dir, 'training.log'))
    logger.info(f"=" * 60)
    logger.info(f"DeepLive Training (Refactored - Using Trainer)")
    logger.info(f"=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"W&B: {'Enabled' if args.wandb else 'Disabled'}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("=" * 60)
    logger.info("Creating dataloaders...")
    logger.info("=" * 60)
    train_loader, val_loader, test_loader, train_samples = create_deeplive_dataloaders(
        experiment_config, logger
    )
    
    # Create model using DETECTOR registry (same as train_sweep.py)
    logger.info("=" * 60)
    logger.info("Creating model...")
    logger.info("=" * 60)
    model = DETECTOR[config['model_name']](config)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created:")
    logger.info(f"  - Total parameters: {total_params:,}")
    logger.info(f"  - Trainable parameters: {trainable_params:,}")
    
    # Dry run check
    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN - Testing one batch...")
        logger.info("=" * 60)
        
        model = model.to(device)
        batch = next(iter(train_loader))
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        logger.info(f"Batch shape: {images.shape}")
        
        with torch.no_grad():
            outputs = model({'image': images, 'label': labels}, inference=True)
        if isinstance(outputs, dict):
            logger.info(f"Output keys: {outputs.keys()}")
        else:
            logger.info(f"Output shape: {outputs.shape}")
        
        logger.info("Dry run successful!")
        return
    
    # Create optimizer (using utils function - same as train_sweep.py)
    optimizer = choose_optimizer(model, config)
    
    # Create scheduler (using utils function - same as train_sweep.py)
    scheduler = choose_scheduler(config, optimizer)
    
    # Choose metric (same as train_sweep.py)
    metric_scoring = choose_metric(config)
    
    # Create Trainer (THE KEY CHANGE - using same Trainer as train_sweep.py)
    logger.info("=" * 60)
    logger.info("Initializing Trainer...")
    logger.info("=" * 60)
    
    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        val_in_dist_loader=val_loader,  # Use val as in-dist
        val_holdout_loader=test_loader,  # Use test as holdout
        metric_scoring=metric_scoring,
        wandb_run=wandb_run,
        ood_loader=None,  # No OOD for DeepLive
        use_group_dro=config.get('use_group_dro', False),
    )
    
    # Training loop (same structure as train_sweep.py)
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    for epoch in range(config['start_epoch'], config['nEpochs']):
        trainer.train_epoch(
            train_loader=train_loader,
            epoch=epoch,
            train_videos=train_samples,  # Pass samples for compatibility
        )
        
        if trainer.early_stop_triggered:
            logger.info(f"Gracefully terminating training at epoch {epoch + 1} due to early stopping.")
            if wandb_run:
                import wandb
                wandb.log({"train/status": "Early Stopped"})
            break
    
    # Finish W&B
    if wandb_run:
        wandb_run.finish()
    
    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed / 60:.2f} minutes")


if __name__ == '__main__':
    main()
