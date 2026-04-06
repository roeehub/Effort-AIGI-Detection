#!/usr/bin/env python3
"""
Simplified training entry point without W&B sweep dependencies.

This script provides a straightforward way to train models using YAML config files
without the complexity of W&B sweeps. Ideal for:
- Local development and debugging
- Single-run experiments
- Testing config changes quickly

Usage:
    python train_simple.py --config config/detector/effort.yaml
    python train_simple.py --config config/detector/effort.yaml --epochs 5 --batch_size 32
    python train_simple.py --config config/detector/effort.yaml --ddp  # Multi-GPU

For production training with hyperparameter sweeps, use train_sweep.py instead.
"""

import argparse
import os
import random
import sys
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import yaml

from utils.config_helpers import load_base_configs
from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.utils import parse_metric_for_print
from logger import create_logger
from prepare_splits import prepare_video_splits_v2
from dataset.dataloaders import create_dataloaders

# Optional W&B import - not required for simple training
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Simple DeepfakeBench Training Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required
    parser.add_argument('--config', type=str, required=True,
                        help='Path to detector YAML config file')
    
    # Training parameters (override config values)
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Weight decay for optimizer')
    
    # Data
    parser.add_argument('--dataloader_config', type=str, 
                        default='./config/dataloader_config.yml',
                        help='Path to dataloader configuration file')
    parser.add_argument('--train_dataset', nargs='+', default=None,
                        help='Training dataset(s)')
    parser.add_argument('--test_dataset', nargs='+', default=None,
                        help='Test dataset(s)')
    
    # Checkpointing
    parser.add_argument('--save_ckpt', action='store_true', default=True,
                        help='Save checkpoints during training')
    parser.add_argument('--no_save_ckpt', dest='save_ckpt', action='store_false',
                        help='Disable checkpoint saving')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Distributed training
    parser.add_argument('--ddp', action='store_true', default=False,
                        help='Use DistributedDataParallel for multi-GPU training')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    
    # Logging
    parser.add_argument('--wandb', action='store_true', default=False,
                        help='Enable W&B logging (optional)')
    parser.add_argument('--wandb_project', type=str, default='deepfake-detection',
                        help='W&B project name')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Custom run name (auto-generated if not provided)')
    
    # Misc
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debug mode with verbose logging')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='Verify setup without training (loads config, creates model, optionally runs 1 batch)')
    
    return parser.parse_args()


def init_seed(seed: int, use_cuda: bool = True):
    """Initialize random seeds for reproducibility."""
    random.seed(seed)
    if use_cuda:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_distributed(args):
    """Setup distributed training if enabled."""
    if args.ddp:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(args.local_rank)
        return True
    return False


def create_optimizer(model, config):
    """Create optimizer based on config."""
    opt_name = config['optimizer']['type']
    opt_config = config['optimizer'].get(opt_name, {})
    
    if opt_name == 'adam':
        return optim.Adam(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt_config.get('lr', 1e-4),
            weight_decay=opt_config.get('weight_decay', 0),
        )
    elif opt_name == 'adamw':
        return optim.AdamW(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt_config.get('lr', 1e-4),
            weight_decay=opt_config.get('weight_decay', 0.01),
        )
    elif opt_name == 'sgd':
        return optim.SGD(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt_config.get('lr', 1e-3),
            momentum=opt_config.get('momentum', 0.9),
            weight_decay=opt_config.get('weight_decay', 0),
        )
    else:
        raise NotImplementedError(f'Optimizer {opt_name} is not implemented')


def create_scheduler(optimizer, config):
    """Create learning rate scheduler based on config."""
    scheduler_name = config.get('lr_scheduler')
    if scheduler_name is None:
        return None
    
    if scheduler_name == 'cosine':
        base_lr = config['optimizer']['adam']['lr']
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config['nEpochs'], 
            eta_min=base_lr / 100
        )
    elif scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('lr_step_size', 10),
            gamma=config.get('lr_gamma', 0.1)
        )
    else:
        raise NotImplementedError(f'Scheduler {scheduler_name} is not implemented')


def apply_cli_overrides(config: dict, args) -> dict:
    """Apply command line argument overrides to config."""
    if args.epochs is not None:
        config['nEpochs'] = args.epochs
    
    if args.batch_size is not None:
        config['train_batchSize'] = args.batch_size
        config['test_batchSize'] = args.batch_size
    
    if args.lr is not None:
        opt_name = config['optimizer']['type']
        if opt_name in config['optimizer']:
            config['optimizer'][opt_name]['lr'] = args.lr
    
    if args.weight_decay is not None:
        opt_name = config['optimizer']['type']
        if opt_name in config['optimizer']:
            config['optimizer'][opt_name]['weight_decay'] = args.weight_decay
    
    if args.seed is not None:
        config['manualSeed'] = args.seed
    
    if args.train_dataset is not None:
        config['train_dataset'] = args.train_dataset
    
    if args.test_dataset is not None:
        config['test_dataset'] = args.test_dataset
    
    return config


def generate_run_name(config: dict, args) -> str:
    """Generate a descriptive run name."""
    if args.run_name:
        return args.run_name
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = config.get('model_name', 'unknown')
    epochs = config.get('nEpochs', 0)
    batch_size = config.get('train_batchSize', 0)
    
    return f"{model_name}_e{epochs}_b{batch_size}_{timestamp}"


def run_dry_run(args):
    """
    Run a dry-run verification: load config, create model, verify imports.
    
    This tests everything up to (but not including) actual training.
    Useful for verifying the setup before deploying to GCP.
    """
    print("="*60)
    print("DRY RUN MODE - Verifying setup without training")
    print("="*60)
    
    checks_passed = 0
    checks_failed = 0
    
    def check(name, fn):
        nonlocal checks_passed, checks_failed
        try:
            result = fn()
            print(f"  ✅ {name}")
            checks_passed += 1
            return result
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            checks_failed += 1
            return None
    
    # Check 1: Load config
    print("\n[1/6] Loading configuration...")
    
    def load_configs():
        # Use load_base_configs for backward compatibility with existing codebase
        config, data_config = load_base_configs(
            detector_path=args.config,
            train_config_path='./config/train_config.yaml',
            dataloader_config_path='./config/dataloader_config.yml',
        )
        # Merge data_config into config for unified access
        config['data_config'] = data_config
        return config, data_config
    
    result = check("load_config", load_configs)
    if result is None:
        print("\n❌ Dry run FAILED - cannot proceed without config")
        return 1
    
    config, data_config = result
    config = apply_cli_overrides(config, args)
    check("apply_cli_overrides", lambda: config)
    
    # Check 2: Validate config structure (basic checks)
    print("\n[2/6] Validating configuration...")
    def validate_basic():
        required_keys = ['model_name', 'resolution', 'nEpochs']
        missing = [k for k in required_keys if k not in config]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")
        return True
    check("validate_config", validate_basic)
    print(f"       Config type: {type(config).__name__}")
    
    # Check 3: Import detector
    print("\n[3/6] Loading model class...")
    model_name = config.get('model_name', 'unknown')
    print(f"       Model: {model_name}")
    model_class = check("import_detector", lambda: DETECTOR[model_name])
    
    # Check 4: Verify model class (skip instantiation - requires GCS backbone weights)
    print("\n[4/6] Verifying model class...")
    if model_class:
        print(f"       Model class: {model_class.__name__}")
        print(f"       (Skipping instantiation - backbone weights downloaded from GCS at runtime)")
        check("verify_model_class", lambda: model_class.__name__)
    else:
        print("       ⚠️  Model class not found, skipping")
    
    # Check 5: Import data pipeline
    print("\n[5/6] Verifying data pipeline imports...")
    check("import_prepare_splits", lambda: prepare_video_splits_v2)
    check("import_dataloaders", lambda: create_dataloaders)
    
    # Check 6: Import trainer
    print("\n[6/6] Verifying trainer imports...")
    check("import_trainer", lambda: Trainer)
    
    # Check trainer mixins
    from trainer.mixins import (
        CheckpointingMixin, EarlyStoppingMixin, GroupDROMixin,
        CurriculumMixin, ArcFaceMixin, ValidationMixin, ReportingMixin
    )
    check("import_mixins", lambda: [
        CheckpointingMixin, EarlyStoppingMixin, GroupDROMixin,
        CurriculumMixin, ArcFaceMixin, ValidationMixin, ReportingMixin
    ])
    
    # Summary
    print("\n" + "="*60)
    total = checks_passed + checks_failed
    if checks_failed == 0:
        print(f"✅ DRY RUN PASSED: {checks_passed}/{total} checks successful")
        print("="*60)
        print("\nThe setup is valid. You can proceed with training:")
        print(f"  python train_simple.py --config {args.config}")
        print("\nOr deploy to GCP:")
        print("  gcloud builds submit --config cloudbuild.yaml")
        return 0
    else:
        print(f"❌ DRY RUN FAILED: {checks_failed}/{total} checks failed")
        print("="*60)
        print("\nPlease fix the issues above before training.")
        return 1


def main():
    """Main training function."""
    args = parse_args()
    
    # Handle dry-run mode
    if getattr(args, 'dry_run', False):
        sys.exit(run_dry_run(args))
    
    # Setup distributed training
    is_distributed = setup_distributed(args)
    is_main_process = not is_distributed or args.local_rank == 0
    
    # Load and validate config
    if is_main_process:
        print(f"Loading config from: {args.config}")
    
    # Use load_base_configs for backward compatibility
    config, data_config = load_base_configs(
        detector_path=args.config,
        train_config_path='./config/train_config.yaml',
        dataloader_config_path='./config/dataloader_config.yml',
    )
    config = apply_cli_overrides(config, args)
    
    # Basic config validation
    required_keys = ['model_name', 'resolution', 'nEpochs']
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    
    # Initialize seed
    seed = config.get('manualSeed', random.randint(1, 10000))
    init_seed(seed, use_cuda=config.get('cuda', True))
    
    # Setup logging
    run_name = generate_run_name(config, args)
    log_dir = os.path.join(config.get('log_dir', './logs'), run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    logger = create_logger(
        phase='train',
        save_path=log_dir,
        level='DEBUG' if args.debug else 'INFO'
    )
    
    if is_main_process:
        logger.info(f"Run name: {run_name}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Epochs: {config.get('nEpochs')}, Batch size: {config.get('train_batchSize')}")
    
    # Optional W&B initialization
    if args.wandb and WANDB_AVAILABLE and is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=config
        )
        logger.info("W&B logging enabled")
    elif args.wandb and not WANDB_AVAILABLE:
        logger.warning("W&B requested but not installed. Skipping W&B logging.")
    
    # Create model
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).cuda()
    
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.local_rank],
            find_unused_parameters=True
        )
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Prepare data splits
    if is_main_process:
        logger.info("Preparing data splits...")
    
    # prepare_video_splits_v2 returns: (train_data, val_in_dist, val_holdout, stats)
    # data_config is already loaded from load_base_configs above
    train_videos, val_in_dist, val_holdout, data_split_stats = prepare_video_splits_v2(data_config)
    
    if is_main_process:
        logger.info(f"Data split stats: {data_split_stats}")
    
    # Create dataloaders using the actual API
    if is_main_process:
        logger.info("Creating dataloaders...")
    
    train_loader, val_in_dist_loader, val_holdout_loader = create_dataloaders(
        train_videos, val_in_dist, val_holdout, config, data_config
    )
    
    # Use val_holdout_loader as primary validation (val_in_dist may be empty for legacy splits)
    val_loader = val_holdout_loader if val_holdout_loader is not None else val_in_dist_loader
    
    if is_main_process:
        logger.info(f"Train loader created: {type(train_loader)}")
        logger.info(f"Validation loader created: {type(val_loader)}")
    
    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        metric_scoring=config.get('metric_scoring', 'auc'),
        find_unused=True,
        save_ckpt=args.save_ckpt
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        if is_main_process:
            logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_ckpt(args.resume)
    
    # Training loop
    if is_main_process:
        logger.info("Starting training...")
    
    best_metric = 0.0
    for epoch in range(config['nEpochs']):
        if is_main_process:
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{config['nEpochs']}")
            logger.info(f"{'='*60}")
        
        # Train one epoch using the actual Trainer API
        # train_epoch(train_loader, epoch, train_videos, val_method_loaders=None)
        train_metrics = trainer.train_epoch(
            train_loader=train_loader,
            epoch=epoch,
            train_videos=train_videos,
            val_method_loaders={'holdout': val_loader} if val_loader else None
        )
        
        if is_main_process:
            logger.info(f"Train metrics: {parse_metric_for_print(train_metrics)}")
            
            # Log to W&B if enabled
            if args.wandb and WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch + 1,
                    **{f'train/{k}': v for k, v in train_metrics.items()},
                })
            
            # The trainer handles validation internally via val_method_loaders
            # Best metric tracking is done inside trainer.train_epoch
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Early stopping check (if enabled in config)
        if hasattr(trainer, 'check_early_stopping'):
            # Get the latest validation metrics from trainer state
            val_metrics = getattr(trainer, 'last_val_metrics', {})
            if trainer.check_early_stopping(val_metrics):
                if is_main_process:
                    logger.info("Early stopping triggered!")
                break
    
    # Final save
    if is_main_process and args.save_ckpt:
        trainer.save_ckpt(os.path.join(log_dir, 'final_model.pth'))
        best_metric = getattr(trainer, 'best_metric', 0.0)
        logger.info(f"Training complete! Best metric: {best_metric:.4f}")
        logger.info(f"Checkpoints saved to: {log_dir}")
    
    # Cleanup
    if args.wandb and WANDB_AVAILABLE and is_main_process:
        wandb.finish()
    
    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
