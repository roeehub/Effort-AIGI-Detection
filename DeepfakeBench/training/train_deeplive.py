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

DeepLive Training Script (DEPRECATED)

This script trains the deepfake detector using the DeepLive dataset with
paired real/fake frames and landmark-based occlusion augmentations.

Usage:
    # Local training with a specific experiment config
    python train_deeplive.py --config experiments/deeplive_vit_B16.yaml
    
    # Training with custom parameters
    python train_deeplive.py --config experiments/deeplive_vit_B16.yaml --max-steps 1000
    
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
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

# Setup path for imports
TRAINING_DIR = Path(__file__).parent
sys.path.insert(0, str(TRAINING_DIR))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def setup_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Set random seed to {seed}")


def create_augmentation_transform(config: Dict[str, Any]):
    """Create the landmark-based occlusion augmentation transform."""
    from data.augmentations.transforms import RegionBBoxOcclusion
    
    aug_config = config.get('augmentation', {})
    occlusion_prob = aug_config.get('occlusion_prob', 0.8)
    regions = aug_config.get('regions', ['left_eye', 'right_eye', 'nose', 'mouth'])
    num_regions = aug_config.get('num_regions', [1, 2])
    occlusion_type = aug_config.get('occlusion_type', 'mixed')
    
    logger.info(f"Creating augmentation transform:")
    logger.info(f"  - Occlusion probability: {occlusion_prob}")
    logger.info(f"  - Regions: {regions}")
    logger.info(f"  - Num regions per image: {num_regions}")
    logger.info(f"  - Occlusion type: {occlusion_type}")
    
    transform = RegionBBoxOcclusion(
        regions=regions,
        num_regions=tuple(num_regions),
        occlusion_type=occlusion_type,
        p=occlusion_prob,
    )
    
    def augmentation_fn(image: np.ndarray, landmarks=None) -> np.ndarray:
        """Apply augmentation with landmarks."""
        if landmarks is None:
            return image
        result = transform(image=image, landmarks=landmarks)
        return result['image']
    
    return augmentation_fn


def create_model(config: Dict[str, Any], device: torch.device):
    """Create the EFFORT detector model with specified backbone."""
    from detectors.effort_detector import EffortDetector
    
    backbone_config = config.get('backbone', {})
    backbone_name = backbone_config.get('name', 'vit_b_16_openai')
    source = backbone_config.get('source', 'openai')
    hidden_size = backbone_config.get('hidden_size', 768)
    huggingface_id = backbone_config.get('huggingface_id', 'openai/clip-vit-base-patch16')
    
    logger.info(f"Creating model:")
    logger.info(f"  - Backbone: {backbone_name}")
    logger.info(f"  - Source: {source}")
    logger.info(f"  - Hidden size: {hidden_size}")
    logger.info(f"  - HuggingFace ID: {huggingface_id}")
    
    # Build model config - pass backbone config for hidden_size resolution
    model_config = {
        'model_name': backbone_name,
        'rank': config.get('rank', hidden_size - 1),
        'pretrained': True,
        'backbone': backbone_config,  # Pass full backbone config for hidden_size
    }
    
    # Handle different backbone sources
    if source == 'laion':
        model_config['backbone_source'] = 'laion'
        model_config['openclip_model_name'] = backbone_config.get('model_name', 'ViT-B-16')
        model_config['openclip_pretrained'] = backbone_config.get('pretrained', 'datacomp_xl_s13b_b90k')
    else:
        # For HuggingFace models, pass the model ID directly
        # The detector will download from HuggingFace Hub
        model_config['backbone_path'] = huggingface_id
        model_config['huggingface_id'] = huggingface_id
    
    # Add ArcFace head config if specified
    if config.get('use_arcface_head', False):
        model_config['use_arcface_head'] = True
        model_config['arcface_s'] = config.get('arcface_s', 30.0)
        model_config['arcface_m'] = config.get('arcface_m', 0.28)
    
    model = EffortDetector(model_config)
    model = model.to(device)
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created:")
    logger.info(f"  - Total parameters: {total_params:,}")
    logger.info(f"  - Trainable parameters: {trainable_params:,}")
    
    return model


def create_dataloaders(
    config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test dataloaders for DeepLive dataset."""
    from dataset.deeplive_dataset import DeepLiveDataset, create_deeplive_dataset
    from data.batching.deeplive import DeepLiveBatchingStrategy, DeepLiveBatchingConfig
    
    deeplive_config = config.get('deeplive', {})
    
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
    seed = config.get('seed', 737)
    
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
    transform = create_augmentation_transform(config)
    
    # Determine num_workers based on environment
    # Cloud Build has limited shared memory, so we disable multiprocessing there
    # Detection: NO_MULTIPROCESSING env var OR running on CPU (typical for Cloud Build)
    no_multiprocessing = os.environ.get('NO_MULTIPROCESSING', '').lower() in ('1', 'true', 'yes')
    device_is_cpu = not torch.cuda.is_available()
    
    if no_multiprocessing:
        num_workers = 0
        logger.info("NO_MULTIPROCESSING env var set - using num_workers=0")
    elif device_is_cpu:
        # CPU-only environments (like Cloud Build) often have limited shared memory
        num_workers = 0
        logger.info("Running on CPU - using num_workers=0 to avoid shared memory issues")
    else:
        num_workers = config.get('num_workers', 4)
    
    # Create batching config
    batching_config = DeepLiveBatchingConfig(
        batch_size=config.get('frames_per_batch', 32),
        num_workers=num_workers,
        prefetch_factor=config.get('prefetch_factor', 2) if num_workers > 0 else None,
        frame_sampling=deeplive_config.get('sampling_mode', 'sparse'),
        sparse_indices=deeplive_config.get('anchor_indices', [0, 2, 4, 6, 8, 10, 12, 14]),
    )
    
    # Create batching strategy
    strategy = DeepLiveBatchingStrategy(
        config=config,
        data_config={'deeplive_data': deeplive_config},
        strategy_config=batching_config,
        dataset=dataset,
        transform=transform,
    )
    
    # Create dataloaders
    train_loader = strategy.create_train_loader(train_samples)
    val_loader = strategy.create_validation_loader(val_samples, mode='test')
    test_loader = strategy.create_validation_loader(test_samples, mode='test')
    
    return train_loader, val_loader, test_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    step = 0
    
    for batch_idx, batch in enumerate(train_loader):
        if max_steps and step >= max_steps:
            break
        
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Model expects dict with 'image' and 'label' for training (ArcFace needs labels)
        outputs = model({'image': images, 'label': labels})
        
        # Handle different output formats
        if isinstance(outputs, dict):
            logits = outputs.get('logits', outputs.get('cls'))
        else:
            logits = outputs
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        step += 1
        
        if step % 10 == 0:
            logger.info(f"Epoch {epoch} Step {step}: Loss={loss.item():.4f}")
    
    avg_loss = total_loss / max(step, 1)
    accuracy = total_correct / max(total_samples, 1)
    
    return {
        'train_loss': avg_loss,
        'train_accuracy': accuracy,
        'train_steps': step,
    }


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    prefix: str = 'val',
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    n_batches = 0  # Count batches during iteration (IterableDataset has no len())
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Model expects dict with 'image', use inference=True for eval
            outputs = model({'image': images, 'label': labels}, inference=True)
            
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('cls'))
            else:
                logits = outputs
            
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            n_batches += 1
    
    avg_loss = total_loss / max(n_batches, 1)
    accuracy = total_correct / max(total_samples, 1)
    
    # Compute additional metrics
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels, all_predictions)
    except:
        auc = 0.0
    
    return {
        f'{prefix}_loss': avg_loss,
        f'{prefix}_accuracy': accuracy,
        f'{prefix}_auc': auc,
        f'{prefix}_samples': total_samples,
    }


def main():
    parser = argparse.ArgumentParser(description='Train DeepLive detector')
    parser.add_argument('--config', '--param-config', type=str, required=True,
                        dest='config',
                        help='Path to experiment config YAML (--param-config is alias for compatibility)')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Maximum training steps (for testing)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run setup only, no actual training')
    parser.add_argument('--output-dir', type=str, default='outputs/deeplive',
                        help='Output directory for checkpoints')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable W&B logging')
    parser.add_argument('--wandb-project', type=str, default='deeplive-experiments',
                        help='W&B project name')
    args = parser.parse_args()
    
    # Load config
    config = load_experiment_config(args.config)
    
    # Setup
    seed = config.get('seed', 737)
    setup_seed(seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize W&B if enabled
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=config.get('name', 'deeplive-experiment'),
            config=config,
        )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    logger.info("=" * 60)
    logger.info("Creating model...")
    logger.info("=" * 60)
    model = create_model(config, device)
    
    # Create dataloaders
    logger.info("=" * 60)
    logger.info("Creating dataloaders...")
    logger.info("=" * 60)
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN - Testing one batch...")
        logger.info("=" * 60)
        
        # Test forward pass
        batch = next(iter(train_loader))
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        logger.info(f"Batch shape: {images.shape}")
        
        with torch.no_grad():
            # Model expects a dict with 'image' and optionally 'label'
            outputs = model({'image': images, 'label': labels}, inference=True)
        if isinstance(outputs, dict):
            logger.info(f"Output keys: {outputs.keys()}")
        else:
            logger.info(f"Output shape: {outputs.shape}")
        
        logger.info("Dry run successful!")
        return
    
    # Setup training
    logger.info("=" * 60)
    logger.info("Setting up training...")
    logger.info("=" * 60)
    
    # Optimizer
    learning_rate = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 0.05)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    n_epochs = config.get('nEpochs', 50)
    best_val_acc = 0.0
    patience = config.get('early_stopping_patience', 10)
    patience_counter = 0
    
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    for epoch in range(1, n_epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{n_epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            max_steps=args.max_steps
        )
        logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                    f"Train Acc: {train_metrics['train_accuracy']:.4f}")
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, prefix='val')
        logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                    f"Val Acc: {val_metrics['val_accuracy']:.4f}, "
                    f"Val AUC: {val_metrics['val_auc']:.4f}")
        
        # Log to W&B
        if args.wandb:
            import wandb
            wandb.log({
                'epoch': epoch,
                **train_metrics,
                **val_metrics,
            })
        
        # Save best model
        if val_metrics['val_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['val_accuracy']
            patience_counter = 0
            
            checkpoint_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc,
                'config': config,
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping after {patience} epochs without improvement")
                break
    
    # Final evaluation on test set
    logger.info("=" * 60)
    logger.info("Final evaluation on test set...")
    logger.info("=" * 60)
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, device, prefix='test')
    logger.info(f"Test Loss: {test_metrics['test_loss']:.4f}, "
                f"Test Acc: {test_metrics['test_accuracy']:.4f}, "
                f"Test AUC: {test_metrics['test_auc']:.4f}")
    
    if args.wandb:
        import wandb
        wandb.log(test_metrics)
        wandb.finish()
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
