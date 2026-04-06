"""
Configuration system for training pipeline.

This module provides:
- Typed dataclasses for all configuration sections
- Unified config loader that merges YAML files and W&B overrides
- Validation for required fields, types, and value ranges

Usage:
    from config_system import load_config, TrainingConfig
    
    # Load and merge all configs with W&B overrides
    config = load_config(
        detector_path='config/detector/effort.yaml',
        train_config_path='config/train_config.yaml',
        dataloader_config_path='config/dataloader_config.yml',
        wandb_config=wandb.config  # Optional W&B overrides
    )
    
    # Access typed config sections
    lr = config.optimizer.lr
    batch_size = config.dataloader.batch_size
"""

from .schema import (
    # Main config
    TrainingConfig,
    # Sub-configs
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingParams,
    EarlyStoppingConfig,
    GCSAssetsConfig,
    CheckpointingConfig,
    DataConfig,
    DataloaderConfig,
    PropertyBalancingConfig,
    MethodsConfig,
    AugmentationConfig,
    LossConfig,
    ArcFaceConfig,
    GroupDROConfig,
)
from .loader import load_config, merge_wandb_overrides, DEFAULT_CONFIG_PATH
from .validation import validate_config, ConfigValidationError

__all__ = [
    # Main config
    'TrainingConfig',
    # Sub-configs
    'ModelConfig',
    'OptimizerConfig',
    'SchedulerConfig', 
    'TrainingParams',
    'EarlyStoppingConfig',
    'GCSAssetsConfig',
    'CheckpointingConfig',
    'DataConfig',
    'DataloaderConfig',
    'PropertyBalancingConfig',
    'MethodsConfig',
    'AugmentationConfig',
    'LossConfig',
    'ArcFaceConfig',
    'GroupDROConfig',
    # Functions
    'load_config',
    'merge_wandb_overrides',
    'validate_config',
    'ConfigValidationError',
    # Paths
    'DEFAULT_CONFIG_PATH',
]
