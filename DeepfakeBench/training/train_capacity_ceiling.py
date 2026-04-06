#!/usr/bin/env python3
"""
B16 Capacity Ceiling Experiment - Training Script

This script is designed specifically to test the maximum capacity of the B16 backbone
WITHOUT the Effort SVD decomposition. It allows progressive unfreezing of transformer
layers to determine the theoretical ceiling of the architecture.

Usage:
    python train_capacity_ceiling.py --config experiments/B16_capacity_ceiling/4_full_finetune.yaml

Key difference from train_sweep.py:
    - Does NOT apply SVD decomposition to attention layers
    - Supports selective layer unfreezing via 'unfreeze_blocks' config
    - Simplified training loop focused on ceiling detection
"""

import argparse
import os
import yaml
import logging
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import numpy as np

from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score, accuracy_score

# Import data pipeline
from data.sources import create_data_pipeline, DataPipelineResult

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# SIMPLE B16 CLASSIFIER (No Effort SVD)
# =============================================================================

class B16Classifier(nn.Module):
    """
    Simple B16 classifier without Effort SVD decomposition.
    Used for capacity ceiling experiments.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        backbone_config = config.get('backbone', {})
        source = backbone_config.get('source', 'laion')
        
        if source in ('laion', 'openclip'):
            self.backbone = self._build_openclip_backbone(backbone_config)
            self.hidden_size = backbone_config.get('hidden_size', 512)
        else:
            raise ValueError(f"Unsupported backbone source: {source}")
        
        # Simple classification head (no ArcFace for clarity)
        self.head = nn.Linear(self.hidden_size, 2)
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Configure layer freezing
        self._configure_trainable_params(config)
    
    def _build_openclip_backbone(self, backbone_config):
        """Load OpenCLIP backbone WITHOUT SVD modification."""
        import open_clip
        
        openclip_model = backbone_config.get('model_name', 'ViT-B-16')
        openclip_pretrained = backbone_config.get('pretrained', 'datacomp_xl_s13b_b90k')
        
        logger.info(f"Loading OpenCLIP backbone: {openclip_model} (pretrained: {openclip_pretrained})")
        
        model, _, _ = open_clip.create_model_and_transforms(
            openclip_model,
            pretrained=openclip_pretrained
        )
        
        return model.visual
    
    def _configure_trainable_params(self, config):
        """
        Configure which parameters are trainable based on config.
        
        Config options:
            unfreeze_mode: 'full' | 'last_n_blocks' | 'effort_baseline'
            unfreeze_n_blocks: Number of blocks to unfreeze (from the end)
        """
        unfreeze_mode = config.get('unfreeze_mode', 'full')
        n_blocks = config.get('unfreeze_n_blocks', 0)
        
        # First freeze everything in backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Head is always trainable
        for param in self.head.parameters():
            param.requires_grad = True
        
        if unfreeze_mode == 'full':
            # Unfreeze entire backbone
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("FULL FINETUNE: All backbone parameters trainable")
            
        elif unfreeze_mode == 'last_n_blocks':
            # Unfreeze last N transformer blocks
            # OpenCLIP structure: visual.transformer.resblocks[0..11]
            if hasattr(self.backbone, 'transformer') and hasattr(self.backbone.transformer, 'resblocks'):
                total_blocks = len(self.backbone.transformer.resblocks)
                start_unfreeze = total_blocks - n_blocks
                
                logger.info(f"Unfreezing blocks {start_unfreeze} to {total_blocks-1} (last {n_blocks} of {total_blocks})")
                
                for idx in range(start_unfreeze, total_blocks):
                    for param in self.backbone.transformer.resblocks[idx].parameters():
                        param.requires_grad = True
                
                # Also unfreeze final LayerNorm and projection if they exist
                if hasattr(self.backbone, 'ln_post'):
                    for param in self.backbone.ln_post.parameters():
                        param.requires_grad = True
                    logger.info("Unfreezing ln_post (final LayerNorm)")
                
                if hasattr(self.backbone, 'proj') and self.backbone.proj is not None:
                    self.backbone.proj.requires_grad = True
                    logger.info("Unfreezing projection layer")
            else:
                logger.warning("Could not find resblocks in backbone, falling back to full finetune")
                for param in self.backbone.parameters():
                    param.requires_grad = True
                    
        elif unfreeze_mode == 'effort_baseline':
            # Keep everything frozen (head only trainable)
            # This simulates Effort with 0 capacity for comparison
            logger.info("EFFORT BASELINE: Only head trainable (backbone fully frozen)")
            
        elif unfreeze_mode == 'projection_only':
            # Just the final projection and head
            if hasattr(self.backbone, 'proj') and self.backbone.proj is not None:
                self.backbone.proj.requires_grad = True
            if hasattr(self.backbone, 'ln_post'):
                for param in self.backbone.ln_post.parameters():
                    param.requires_grad = True
            logger.info("PROJECTION ONLY: Just projection layer + head trainable")
        
        else:
            raise ValueError(f"Unknown unfreeze_mode: {unfreeze_mode}")
        
        # Log trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    def forward(self, images, labels=None):
        """Forward pass."""
        # Get features from backbone
        features = self.backbone(images)
        
        # Classification
        logits = self.head(features)
        
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return logits, loss
        
        return logits
    
    def get_probs(self, logits):
        """Convert logits to probabilities."""
        return F.softmax(logits, dim=1)[:, 1]  # Probability of fake class


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, config):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    num_batches = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Handle [B, F, C, H, W] input from DF40 pipeline - flatten to [B*F, C, H, W]
        if images.dim() == 5:
            B, F, C, H, W = images.shape
            images = images.view(B * F, C, H, W)
            # Repeat labels for each frame: [B] -> [B*F]
            labels = labels.unsqueeze(1).expand(B, F).reshape(B * F)
        
        optimizer.zero_grad()
        
        logits, loss = model(images, labels)
        
        loss.backward()
        
        # Gradient clipping
        clip_val = config.get('gradient_clip_val', 1.0)
        if clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        probs = model.get_probs(logits)
        all_preds.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if batch_idx % 100 == 0:
            logger.info(f"Epoch {epoch} [batch {batch_idx}] Loss: {loss.item():.4f}")
    
    # Compute metrics
    avg_loss = total_loss / max(num_batches, 1)
    train_auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    train_acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    
    return {
        'loss': avg_loss,
        'auc': train_auc,
        'acc': train_acc
    }


def evaluate(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    
    total_loss = 0
    num_batches = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Handle [B, F, C, H, W] input from DF40 pipeline - flatten to [B*F, C, H, W]
            if images.dim() == 5:
                B, F, C, H, W = images.shape
                images = images.view(B * F, C, H, W)
                # Repeat labels for each frame: [B] -> [B*F]
                labels = labels.unsqueeze(1).expand(B, F).reshape(B * F)
            
            logits, loss = model(images, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            probs = model.get_probs(logits)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / max(num_batches, 1)
    val_auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    val_acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    
    return {
        'loss': avg_loss,
        'auc': val_auc,
        'acc': val_acc
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='B16 Capacity Ceiling Experiment')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to experiment config YAML')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Also load default dataloader config
    dataloader_config_path = config.get('dataloader_config', './config/dataloader_config.yml')
    if os.path.exists(dataloader_config_path):
        with open(dataloader_config_path, 'r') as f:
            data_config = yaml.safe_load(f)
    else:
        data_config = {}
    
    # CRITICAL: Merge experiment config into data_config
    # The data pipeline factory looks for 'data_source' and dataset configs in data_config
    data_config['data_source'] = config.get('data_source', 'df40_paired')
    
    # Merge df40_paired config if present
    if 'df40_paired' in config:
        data_config['df40_paired'] = config['df40_paired']
    
    # Merge other data-related configs
    for key in ['dataloader_strategy', 'frames_per_batch', 'frames_per_video', 
                'num_workers', 'prefetch_factor', 'augmentation', 'mean', 'std',
                'test_batch_size', 'seed']:
        if key in config:
            data_config[key] = config[key]
    
    logger.info(f"Data source: {data_config.get('data_source')}")
    
    # Initialize W&B
    wandb_run = wandb.init(
        project=os.environ.get('WANDB_PROJECT', 'B16-capacity-ceiling'),
        entity=os.environ.get('WANDB_ENTITY', 'dtect-vision'),
        name=config.get('name', f'capacity_ceiling_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
        config=config
    )
    
    logger.info(f"="*60)
    logger.info(f"B16 CAPACITY CEILING EXPERIMENT")
    logger.info(f"Config: {args.config}")
    logger.info(f"Unfreeze mode: {config.get('unfreeze_mode', 'full')}")
    logger.info(f"="*60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = B16Classifier(config)
    model = model.to(device)
    
    # Create data pipeline
    data_pipeline: DataPipelineResult = create_data_pipeline(
        config=config,
        data_config=data_config,
        logger=logger
    )
    
    train_loader = data_pipeline.train_loader
    val_loader = data_pipeline.val_in_dist_loader
    
    # Setup optimizer (different LR for backbone vs head)
    backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if 'head' in n and p.requires_grad]
    
    backbone_lr = config.get('backbone_lr', config.get('learning_rate', 1e-5))
    head_lr = config.get('head_lr', config.get('learning_rate', 1e-4))
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': head_lr}
    ], weight_decay=config.get('weight_decay', 0.05))
    
    logger.info(f"Backbone LR: {backbone_lr}, Head LR: {head_lr}")
    
    # Setup scheduler - use total_training_steps from config (IterableDataset has no len())
    total_steps = config.get('total_training_steps')
    if total_steps is None:
        raise ValueError("'total_training_steps' must be configured (IterableDataset has no len())")
    warmup_steps = config.get('lr_scheduler_warmup_steps', int(total_steps * 0.1))
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_auc = 0.0
    patience_counter = 0
    patience = config.get('early_stopping_patience', 10)
    
    for epoch in range(config.get('nEpochs', 30)):
        # Set epoch on dataset if it supports it (for varying random selections)
        if hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(epoch)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, config)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        
        # Compute gap (overfitting indicator)
        gap = train_metrics['auc'] - val_metrics['auc']
        
        # Log to W&B
        wandb.log({
            'epoch': epoch,
            'train/loss': train_metrics['loss'],
            'train/auc': train_metrics['auc'],
            'train/acc': train_metrics['acc'],
            'val/loss': val_metrics['loss'],
            'val/auc': val_metrics['auc'],
            'val/acc': val_metrics['acc'],
            'gap/auc': gap,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        logger.info(f"Epoch {epoch}: Train AUC={train_metrics['auc']:.4f}, Val AUC={val_metrics['auc']:.4f}, Gap={gap:.4f}")
        
        # Track best
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            patience_counter = 0
            
            wandb.log({
                'best/val_auc': best_val_auc,
                'best/epoch': epoch,
                'best/train_auc': train_metrics['auc'],
                'best/gap': gap
            })
            
            # Save checkpoint
            checkpoint_path = f"checkpoints/ceiling_{config.get('name', 'exp')}_best.pth"
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_metrics['auc'],
                'train_auc': train_metrics['auc'],
                'config': config
            }, checkpoint_path)
            logger.info(f"Saved best checkpoint: {checkpoint_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
            break
    
    # Final summary
    logger.info(f"="*60)
    logger.info(f"EXPERIMENT COMPLETE")
    logger.info(f"Best Val AUC: {best_val_auc:.4f}")
    logger.info(f"="*60)
    
    wandb.finish()


if __name__ == '__main__':
    main()
