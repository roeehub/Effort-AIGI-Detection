"""
Configuration validation.

Provides functions to validate TrainingConfig instances,
checking for required fields, valid value ranges, and logical consistency.
"""

from typing import List, Optional
from dataclasses import fields

from .schema import TrainingConfig


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    
    def __init__(self, errors: List[str]):
        self.errors = errors
        message = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(message)


def validate_config(config: TrainingConfig, strict: bool = False) -> List[str]:
    """
    Validate a TrainingConfig instance.
    
    Args:
        config: The configuration to validate
        strict: If True, raise ConfigValidationError on any errors
    
    Returns:
        List of validation error messages (empty if valid)
    
    Raises:
        ConfigValidationError: If strict=True and validation fails
    """
    errors = []
    
    # Validate model config
    errors.extend(_validate_model(config))
    
    # Validate optimizer config
    errors.extend(_validate_optimizer(config))
    
    # Validate scheduler config
    errors.extend(_validate_scheduler(config))
    
    # Validate training params
    errors.extend(_validate_training(config))
    
    # Validate data config
    errors.extend(_validate_data(config))
    
    # Validate dataloader config
    errors.extend(_validate_dataloader(config))
    
    # Validate loss config
    errors.extend(_validate_loss(config))
    
    # Validate ArcFace config
    errors.extend(_validate_arcface(config))
    
    # Validate Group-DRO config
    errors.extend(_validate_group_dro(config))
    
    if strict and errors:
        raise ConfigValidationError(errors)
    
    return errors


def _validate_model(config: TrainingConfig) -> List[str]:
    """Validate model configuration."""
    errors = []
    model = config.model
    
    # Model name
    if not model.model_name:
        errors.append("model.model_name is required")
    
    # Resolution
    if model.resolution <= 0:
        errors.append(f"model.resolution must be positive, got {model.resolution}")
    
    # Normalization
    if len(model.mean) != 3:
        errors.append(f"model.mean must have 3 values, got {len(model.mean)}")
    if len(model.std) != 3:
        errors.append(f"model.std must have 3 values, got {len(model.std)}")
    
    # SVD rank
    if model.rank <= 0:
        errors.append(f"model.rank must be positive, got {model.rank}")
    
    return errors


def _validate_optimizer(config: TrainingConfig) -> List[str]:
    """Validate optimizer configuration."""
    errors = []
    opt = config.optimizer
    
    # Optimizer type
    if opt.type not in ('adam', 'sgd'):
        errors.append(f"optimizer.type must be 'adam' or 'sgd', got '{opt.type}'")
    
    # Learning rate
    lr = opt.lr
    if lr <= 0:
        errors.append(f"optimizer.lr must be positive, got {lr}")
    if lr > 1:
        errors.append(f"optimizer.lr seems too high: {lr}")
    
    # Weight decay
    wd = opt.adam.weight_decay if opt.type == 'adam' else opt.sgd.weight_decay
    if wd < 0:
        errors.append(f"optimizer.weight_decay must be non-negative, got {wd}")
    
    # Adam epsilon
    if opt.type == 'adam' and opt.adam.eps <= 0:
        errors.append(f"optimizer.adam.eps must be positive, got {opt.adam.eps}")
    
    return errors


def _validate_scheduler(config: TrainingConfig) -> List[str]:
    """Validate scheduler configuration."""
    errors = []
    sched = config.scheduler
    
    valid_types = (None, 'none', 'null', 'cosine', 'cosine_with_warmup')
    if sched.type and sched.type.lower() not in [t for t in valid_types if t]:
        errors.append(f"scheduler.type must be one of {valid_types}, got '{sched.type}'")
    
    # Warmup scheduler requires total_training_steps
    if sched.type == 'cosine_with_warmup':
        if not sched.total_training_steps:
            errors.append("scheduler.total_training_steps is required for 'cosine_with_warmup' scheduler")
        if sched.warmup_steps < 0:
            errors.append(f"scheduler.warmup_steps must be non-negative, got {sched.warmup_steps}")
        if sched.total_training_steps and sched.warmup_steps > sched.total_training_steps:
            errors.append(f"scheduler.warmup_steps ({sched.warmup_steps}) cannot exceed total_training_steps ({sched.total_training_steps})")
    
    return errors


def _validate_training(config: TrainingConfig) -> List[str]:
    """Validate training parameters."""
    errors = []
    train = config.training
    
    # Epochs
    if train.epochs <= 0:
        errors.append(f"training.epochs must be positive, got {train.epochs}")
    
    # Metric
    valid_metrics = ('auc', 'acc', 'eer', 'ap')
    if train.metric_scoring not in valid_metrics:
        errors.append(f"training.metric_scoring must be one of {valid_metrics}, got '{train.metric_scoring}'")
    
    # Gradient clipping
    if train.gradient_clip_val < 0:
        errors.append(f"training.gradient_clip_val must be non-negative, got {train.gradient_clip_val}")
    
    return errors


def _validate_data(config: TrainingConfig) -> List[str]:
    """Validate data configuration."""
    errors = []
    data = config.data
    
    # Bucket name
    if not data.bucket_name:
        errors.append("data.bucket_name is required")
    
    # Val split ratio
    if not 0 < data.val_split_ratio < 1:
        errors.append(f"data.val_split_ratio must be between 0 and 1, got {data.val_split_ratio}")
    
    # Data subset percentage
    if not 0 < data.data_subset_percentage <= 1:
        errors.append(f"data.data_subset_percentage must be between 0 and 1, got {data.data_subset_percentage}")
    
    # Frames per video
    if data.num_frames_per_video <= 0:
        errors.append(f"data.num_frames_per_video must be positive, got {data.num_frames_per_video}")
    
    return errors


def _validate_dataloader(config: TrainingConfig) -> List[str]:
    """Validate dataloader configuration."""
    errors = []
    dl = config.dataloader
    
    # Strategy
    valid_strategies = ('frame_level', 'video_level', 'per_method', 'property_balancing')
    if dl.strategy not in valid_strategies:
        errors.append(f"dataloader.strategy must be one of {valid_strategies}, got '{dl.strategy}'")
    
    # Batch size
    if dl.batch_size <= 0:
        errors.append(f"dataloader.batch_size must be positive, got {dl.batch_size}")
    
    # Workers
    if dl.num_workers < 0:
        errors.append(f"dataloader.num_workers must be non-negative, got {dl.num_workers}")
    
    # Prefetch factor
    if dl.prefetch_factor <= 0:
        errors.append(f"dataloader.prefetch_factor must be positive, got {dl.prefetch_factor}")
    
    return errors


def _validate_loss(config: TrainingConfig) -> List[str]:
    """Validate loss configuration."""
    errors = []
    loss = config.loss
    
    # Focal loss params
    if loss.use_focal_loss:
        if loss.focal_loss_gamma < 0:
            errors.append(f"loss.focal_loss_gamma must be non-negative, got {loss.focal_loss_gamma}")
    
    return errors


def _validate_arcface(config: TrainingConfig) -> List[str]:
    """Validate ArcFace configuration."""
    errors = []
    af = config.arcface
    
    if af.enabled:
        if af.scale <= 0:
            errors.append(f"arcface.scale must be positive, got {af.scale}")
        if not 0 <= af.margin <= 1:
            errors.append(f"arcface.margin should be between 0 and 1, got {af.margin}")
        if af.anneal_steps < 0:
            errors.append(f"arcface.anneal_steps must be non-negative, got {af.anneal_steps}")
    
    return errors


def _validate_group_dro(config: TrainingConfig) -> List[str]:
    """Validate Group-DRO configuration."""
    errors = []
    dro = config.group_dro
    
    if dro.enabled:
        if dro.beta < 0:
            errors.append(f"group_dro.beta must be non-negative, got {dro.beta}")
        if dro.clip_min > dro.clip_max:
            errors.append(f"group_dro.clip_min ({dro.clip_min}) must be <= clip_max ({dro.clip_max})")
        if not 0 <= dro.ema_alpha <= 1:
            errors.append(f"group_dro.ema_alpha must be between 0 and 1, got {dro.ema_alpha}")
    
    return errors


def print_config_summary(config: TrainingConfig) -> str:
    """
    Generate a human-readable summary of the configuration.
    
    Useful for logging at the start of training.
    """
    lines = [
        "=" * 60,
        "TRAINING CONFIGURATION SUMMARY",
        "=" * 60,
        "",
        "Model:",
        f"  Name: {config.model.model_name}",
        f"  Backbone: {config.model.backbone_name}",
        f"  Resolution: {config.model.resolution}",
        f"  SVD Rank: {config.model.rank}",
        "",
        "Optimizer:",
        f"  Type: {config.optimizer.type}",
        f"  Learning Rate: {config.optimizer.lr}",
        f"  Weight Decay: {config.optimizer.adam.weight_decay if config.optimizer.type == 'adam' else config.optimizer.sgd.weight_decay}",
        "",
        "Training:",
        f"  Epochs: {config.training.epochs}",
        f"  Metric: {config.training.metric_scoring}",
        f"  Gradient Clip: {config.training.gradient_clip_val or 'None'}",
        "",
        "Scheduler:",
        f"  Type: {config.scheduler.type or 'None'}",
    ]
    
    if config.scheduler.type == 'cosine_with_warmup':
        lines.extend([
            f"  Total Steps: {config.scheduler.total_training_steps}",
            f"  Warmup Steps: {config.scheduler.warmup_steps}",
        ])
    
    lines.extend([
        "",
        "Data:",
        f"  Bucket: {config.data.bucket_name}",
        f"  Val Split: {config.data.val_split_ratio}",
        f"  Subset %: {config.data.data_subset_percentage}",
        "",
        "Dataloader:",
        f"  Strategy: {config.dataloader.strategy}",
        f"  Batch Size: {config.dataloader.batch_size}",
        f"  Workers: {config.dataloader.num_workers}",
        "",
        "Features:",
        f"  Augmentation: v{config.augmentation.version} ({'enabled' if config.augmentation.enabled else 'disabled'})",
        f"  Focal Loss: {'enabled' if config.loss.use_focal_loss else 'disabled'}",
        f"  ArcFace: {'enabled' if config.arcface.enabled else 'disabled'}",
        f"  Group-DRO: {'enabled' if config.group_dro.enabled else 'disabled'}",
        f"  Property Balancing: {'enabled' if config.property_balancing.enabled else 'disabled'}",
        "",
        "Early Stopping:",
        f"  Enabled: {config.early_stopping.enabled}",
        f"  Patience: {config.early_stopping.patience}",
        f"  Monitor: {config.early_stopping.monitor}",
        "",
        "=" * 60,
    ])
    
    return "\n".join(lines)
