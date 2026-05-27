"""
Configuration helpers for train_sweep.py.

This module provides helper functions to reduce boilerplate in the main training
entry point while maintaining backward compatibility with the existing config structure.
"""
# =============================================================================
# CODE VERSION STAMP - Update this when making changes to verify deployment
# =============================================================================
CONFIG_HELPERS_VERSION = "2026-02-19-QUALITY-HEAD-PROPAGATION-FIX-V1"  # Ensure GRL/quality-head params propagate from run config
print(f"📦 CONFIG_HELPERS.PY VERSION: {CONFIG_HELPERS_VERSION}")
# =============================================================================

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import os


# ==============================================================================
# Backbone Configuration Constants
# ==============================================================================
# Default normalization values for different backbone types
BACKBONE_NORMALIZATION = {
    'clip': {
        'mean': [0.48145466, 0.4578275, 0.40821073],
        'std': [0.26862954, 0.26130258, 0.27577711],
    },
    'openclip': {
        'mean': [0.48145466, 0.4578275, 0.40821073],
        'std': [0.26862954, 0.26130258, 0.27577711],
    },
    'siglip': {
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
    },
    'dinov2': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    },
}

# Default hidden sizes for different backbone variants
BACKBONE_HIDDEN_SIZES = {
    'ViT-B-16': 768,
    'ViT-B-32': 512,
    'ViT-L-14': 1024,
    'ViT-L-14-336': 1024,
    'ViT-H-14': 1280,
    'ViT-G-14': 1664,
}


def load_base_configs(
    detector_path: str = './config/detector/effort.yaml',
    train_config_path: str = './config/train_config.yaml',
    dataloader_config_path: str = './config/dataloader_config.yml',
    defaults_path: str = './config/defaults.yaml',
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load and merge base configuration files.
    
    Args:
        detector_path: Path to detector YAML
        train_config_path: Path to train config YAML
        dataloader_config_path: Path to dataloader config YAML
        defaults_path: Path to defaults YAML (contains backbone_registry, etc.)
        
    Returns:
        Tuple of (config, data_config) dictionaries
    """
    # Start with defaults (base layer - contains backbone_registry, etc.)
    config = {}
    if os.path.exists(defaults_path):
        with open(defaults_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    
    # Load and merge detector config
    with open(detector_path, 'r') as f:
        detector_cfg = yaml.safe_load(f) or {}
        config.update(detector_cfg)
    
    # Merge train config
    with open(train_config_path, 'r') as f:
        train_cfg = yaml.safe_load(f) or {}
        config.update(train_cfg)
    
    # Load dataloader config separately
    with open(dataloader_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    return config, data_config


def _coerce_bool(value: Any) -> bool:
    """Best-effort conversion for bool-like config values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def apply_wandb_optimizer_params(config: Dict, wandb_config: Any) -> None:
    """
    Apply W&B optimizer parameters to config (in-place).
    
    Args:
        config: Main config dict to update
        wandb_config: W&B config object
    """
    # Use .get() with defaults to handle missing attributes gracefully
    config['load_base_checkpoint'] = wandb_config.get('load_base_checkpoint', config.get('load_base_checkpoint', False))
    config['optimizer']['adam']['lr'] = float(wandb_config.get('learning_rate', config['optimizer']['adam'].get('lr', 1e-4)))
    config['optimizer']['adam']['eps'] = float(wandb_config.get('optimizer_eps', config['optimizer']['adam'].get('eps', 1e-8)))
    config['optimizer']['adam']['weight_decay'] = float(wandb_config.get('weight_decay', config['optimizer']['adam'].get('weight_decay', 0.05)))
    config['nEpochs'] = int(wandb_config.get('nEpochs', config.get('nEpochs', 10)))
    config['lambda_reg'] = float(wandb_config.get('lambda_reg', config.get('lambda_reg', 1.0)))
    config['rank'] = int(wandb_config.get('rank', config.get('rank', 1023)))
    config['lr_scheduler'] = wandb_config.get('lr_scheduler', config.get('lr_scheduler', None))
    config['total_training_steps'] = int(wandb_config.get('total_training_steps', config.get('total_training_steps', 35000)))
    config['lr_scheduler_warmup_steps'] = int(wandb_config.get('lr_scheduler_warmup_steps', config.get('lr_scheduler_warmup_steps', 1000)))
    config['gradient_clip_val'] = float(wandb_config.get('gradient_clip_val', config.get('gradient_clip_val', 0)))


def apply_wandb_loss_params(config: Dict, wandb_config: Any) -> None:
    """
    Apply W&B loss parameters to config (in-place).
    
    Args:
        config: Main config dict to update
        wandb_config: W&B config object
    """
    # Focal Loss
    config['use_focal_loss'] = _coerce_bool(
        wandb_config.get('use_focal_loss', config.get('use_focal_loss', False))
    )
    config['focal_loss_gamma'] = float(
        wandb_config.get('focal_loss_gamma', config.get('focal_loss_gamma', 2.0))
    )
    focal_alpha = wandb_config.get('focal_loss_alpha', config.get('focal_loss_alpha', None))
    if focal_alpha in ('null', 'None'):
        focal_alpha = None
    config['focal_loss_alpha'] = focal_alpha

    # Label smoothing (used by EffortDetector's CrossEntropyLoss)
    config['label_smoothing'] = float(
        wandb_config.get('label_smoothing', config.get('label_smoothing', 0.0))
    )


def apply_wandb_stability_params(config: Dict, wandb_config: Any) -> None:
    """
    Apply W&B stability regularisation parameters to config (in-place).

    These are consumed by ``StabilityRegMixin.init_stability_reg()``.

    Args:
        config: Main config dict to update
        wandb_config: W&B config object
    """
    config['stability_lambda'] = float(
        wandb_config.get('stability_lambda', config.get('stability_lambda', 0.0))
    )
    config['stability_noise_std'] = float(
        wandb_config.get('stability_noise_std', config.get('stability_noise_std', 0.02))
    )
    config['stability_crop_jitter'] = float(
        wandb_config.get('stability_crop_jitter', config.get('stability_crop_jitter', 0.03))
    )


def apply_wandb_group_dro_params(config: Dict, wandb_config: Any) -> None:
    """
    Apply W&B Group-DRO parameters to config (in-place).
    
    Args:
        config: Main config dict to update
        wandb_config: W&B config object
    """
    config['use_group_dro'] = _coerce_bool(
        wandb_config.get('use_group_dro', config.get('use_group_dro', False))
    )
    if config['use_group_dro']:
        config['group_dro_params'] = {
            'beta': float(wandb_config.get('group_dro_beta', 3.0)),
            'clip_min': float(wandb_config.get('group_dro_clip_min', 1.0)),
            'clip_max': float(wandb_config.get('group_dro_clip_max', 4.0)),
            'ema_alpha': float(wandb_config.get('group_dro_ema_alpha', 0.1))
        }


def apply_wandb_arcface_params(config: Dict, wandb_config: Any, logger: Any = None) -> None:
    """
    Apply W&B ArcFace parameters to config (in-place).
    
    Args:
        config: Main config dict to update
        wandb_config: W&B config object
        logger: Optional logger for debug output
    """
    config['use_arcface_head'] = _coerce_bool(
        wandb_config.get('use_arcface_head', config.get('use_arcface_head', False))
    )
    config['train_arcface'] = _coerce_bool(
        wandb_config.get('train_arcface', config.get('train_arcface', True))
    )
    
    if config['use_arcface_head']:
        config['arcface_s'] = float(wandb_config.get('arcface_s', 30.0))
        config['arcface_m'] = float(wandb_config.get('arcface_m', 0.35))
        config['s_start'] = float(wandb_config.get('s_start', config['arcface_s']))
        config['s_end'] = float(wandb_config.get('s_end', config['arcface_s']))
        config['anneal_steps'] = int(wandb_config.get('anneal_steps', 0))
        
        if logger and config['train_arcface']:
            logger.info("--- ArcFace curriculum learning enabled with parameters: ---")
            logger.info(f"   - arcface_s: {config['arcface_s']}")
            logger.info(f"   - arcface_m: {config['arcface_m']}")
            logger.info(f"   - s_start: {config['s_start']}")
            logger.info(f"   - s_end: {config['s_end']}")
            logger.info(f"   - anneal_steps: {config['anneal_steps']}")


def apply_wandb_quality_domain_params(
    config: Dict,
    wandb_config: Any,
    logger: Any = None,
) -> None:
    """
    Apply quality-domain adversarial head params to config (in-place).

    This wiring is critical for Round 6+ experiments that enable the GRL head.
    """
    use_quality_head = _coerce_bool(
        wandb_config.get(
            'use_quality_domain_head',
            config.get('use_quality_domain_head', False),
        )
    )
    config['use_quality_domain_head'] = use_quality_head

    # Keep this explicit so missing supervision fails fast instead of silently
    # training with a zero quality-domain loss.
    config['quality_domain_require_labels'] = _coerce_bool(
        wandb_config.get(
            'quality_domain_require_labels',
            config.get('quality_domain_require_labels', True),
        )
    )

    if not use_quality_head:
        return

    config['quality_domain_count'] = int(
        wandb_config.get(
            'quality_domain_count',
            config.get('quality_domain_count', 4),
        )
    )
    config['quality_head_hidden_dim'] = int(
        wandb_config.get(
            'quality_head_hidden_dim',
            config.get('quality_head_hidden_dim', 128),
        )
    )
    config['quality_domain_loss_weight'] = float(
        wandb_config.get(
            'quality_domain_loss_weight',
            config.get('quality_domain_loss_weight', 0.1),
        )
    )

    if logger:
        logger.info(
            "Applied quality-domain head config: enabled=%s domains=%d hidden_dim=%d loss_weight=%.4f require_labels=%s",
            config['use_quality_domain_head'],
            config['quality_domain_count'],
            config['quality_head_hidden_dim'],
            config['quality_domain_loss_weight'],
            config['quality_domain_require_labels'],
        )


def apply_wandb_early_stopping_params(config: Dict, wandb_config: Any) -> None:
    """
    Apply W&B early stopping parameters to config (in-place).
    
    Args:
        config: Main config dict to update
        wandb_config: W&B config object
    """
    config['early_stopping'] = {
        'enabled': wandb_config.get('early_stopping_enabled', False),
        'patience': wandb_config.get('early_stopping_patience', 3),
        'min_delta': wandb_config.get('early_stopping_min_delta', 0.0001)
    }


def apply_wandb_curriculum_params(config: Dict, wandb_config: Any) -> None:
    """
    Apply W&B curriculum/lesson gate parameters to config (in-place).
    
    Args:
        config: Main config dict to update
        wandb_config: W&B config object
    """
    config['max_train_steps'] = wandb_config.get('max_train_steps', config.get('max_train_steps', None))
    config['evaluate_every_steps'] = wandb_config.get(
        'evaluate_every_steps',
        config.get('evaluate_every_steps', None),
    )
    config['ood_monitoring_enabled'] = wandb_config.get(
        'ood_monitoring_enabled',
        config.get('ood_monitoring_enabled', True),
    )
    config['ood_monitoring_start_step'] = wandb_config.get(
        'ood_monitoring_start_step',
        config.get('ood_monitoring_start_step', 0),
    )
    config['ood_monitoring_every_steps'] = wandb_config.get(
        'ood_monitoring_every_steps',
        config.get('ood_monitoring_every_steps', None),
    )
    # Keep top-level seed wired so canonical seed resolution in train_sweep reflects param-config.
    config['seed'] = wandb_config.get('seed', config.get('seed', None))


def apply_wandb_dataloader_params(data_config: Dict, wandb_config: Any) -> None:
    """
    Apply W&B dataloader parameters to data_config (in-place).
    
    Args:
        data_config: Data config dict to update
        wandb_config: W&B config object
    """
    # Use .get() with fallbacks to existing config values
    dl_params = data_config.get('dataloader_params', {})
    data_params = data_config.get('data_params', {})
    prop_bal = data_config.get('property_balancing', {})
    
    data_config['dataloader_params']['strategy'] = wandb_config.get('dataloader_strategy', dl_params.get('strategy', 'property_balancing'))
    data_config['dataloader_params']['frames_per_batch'] = wandb_config.get('frames_per_batch', dl_params.get('frames_per_batch', 64))
    data_config['dataloader_params']['videos_per_batch'] = wandb_config.get('videos_per_batch', dl_params.get('videos_per_batch', 8))
    data_config['dataloader_params']['frames_per_video'] = wandb_config.get('frames_per_video', dl_params.get('frames_per_video', 8))
    data_config['dataloader_params']['real_label_ratio'] = wandb_config.get('real_label_ratio', dl_params.get('real_label_ratio', None))
    data_config['data_params']['val_split_ratio'] = wandb_config.get('val_split_ratio', data_params.get('val_split_ratio', 0.1))
    data_config['data_params']['evaluation_frequency'] = wandb_config.get('evaluation_frequency', data_params.get('evaluation_frequency', 3))
    data_config['property_balancing']['enabled'] = wandb_config.get('property_balancing_enabled', prop_bal.get('enabled', True))
    
    # Fixed context params
    data_config['data_params']['seed'] = wandb_config.get('seed', data_params.get('seed', 737))
    data_config['data_params']['data_subset_percentage'] = wandb_config.get('data_subset_percentage', data_params.get('data_subset_percentage', 1.0))
    data_config['dataloader_params']['test_batch_size'] = wandb_config.get('test_batch_size', dl_params.get('test_batch_size', 8))
    data_config['dataloader_params']['num_workers'] = wandb_config.get('num_workers', dl_params.get('num_workers', 12))
    data_config['dataloader_params']['prefetch_factor'] = wandb_config.get('prefetch_factor', dl_params.get('prefetch_factor', 3))


def apply_wandb_augmentation_params(config: Dict, wandb_config: Any, logger: Any = None) -> None:
    """
    Apply W&B augmentation parameters to config (in-place).
    
    Args:
        config: Main config dict to update
        wandb_config: W&B config object
        logger: Optional logger for debug output
    """
    aug_params = wandb_config.get('augmentation_params')
    if aug_params:
        config['augmentation_params'] = dict(aug_params)
        if logger:
            logger.info("Successfully loaded augmentation parameters from the run's configuration.")
            logger.info(f"Augmentation settings: {config['augmentation_params']}")
    else:
        config['augmentation_params'] = {
            "use_geometric": True,
            "use_advanced_noise": False,
            "use_color_jitter": True,
            "use_occlusion": True,
            "sharpness_adjust_prob": 0.6,
            "occlusion_prob": 0.4,
        }
    
    # Augmentation version
    aug_version = wandb_config.get('augmentation_version')
    if aug_version:
        config['augmentation_params']['version'] = aug_version
        if logger:
            logger.info(f"SET augmentation version to: {aug_version}")
    else:
        config['augmentation_params']['version'] = 'surgical'
        if logger:
            logger.info("`augmentation_version` not in wandb config, defaulting to 'surgical'.")


def apply_wandb_lesson_gate_params(config: Dict, wandb_config: Any, logger: Any = None) -> None:
    """
    Apply W&B lesson gate parameters to config (in-place).
    
    Args:
        config: Main config dict to update
        wandb_config: W&B config object
        logger: Optional logger for debug output
    """
    # Check if already set directly from single_cfg (bypassing W&B flattening)
    existing_gate = config.get('lesson_gate', {})
    if existing_gate.get('enabled', False):
        if logger:
            logger.info("✅ lesson_gate already set (from direct config), skipping W&B override.")
        return
    
    lesson_gate_config = wandb_config.get('lesson_gate', {})
    if lesson_gate_config and lesson_gate_config.get('enabled', False):
        config['lesson_gate'] = dict(lesson_gate_config)
        if logger:
            logger.info("✅ Loaded Lesson Gate configuration from wandb.config.")
            logger.info(f"   - Gate settings: {config['lesson_gate']}")
    elif not existing_gate:  # Only set to disabled if not already set
        config['lesson_gate'] = {'enabled': False}


def apply_wandb_lesson_data_control_params(config: Dict, wandb_config: Any, logger: Any = None) -> None:
    """
    Apply W&B lesson data control parameters to config (in-place).
    
    Args:
        config: Main config dict to update
        wandb_config: W&B config object
        logger: Optional logger for debug output
    """
    # Check if already set directly from single_cfg (bypassing W&B flattening)
    existing_ldc = config.get('lesson_data_control', {})
    if existing_ldc.get('enabled', False):
        if logger:
            logger.info("✅ lesson_data_control already set (from direct config), skipping W&B override.")
        return
    
    lesson_data_control_config = wandb_config.get('lesson_data_control', {})
    if lesson_data_control_config and lesson_data_control_config.get('enabled', False):
        config['lesson_data_control'] = dict(lesson_data_control_config)
        if logger:
            logger.info("✅ Loaded Lesson Data Control configuration for dynamic method grouping.")
            logger.info(f"   - Group settings: {config['lesson_data_control']}")
    elif not existing_ldc:  # Only set to disabled if not already set
        config['lesson_data_control'] = {'enabled': False}


def apply_wandb_dataset_methods_override(data_config: Dict, wandb_config: Any, logger: Any = None) -> None:
    """
    Apply W&B dataset methods override to data_config (in-place).
    
    Args:
        data_config: Data config dict to update
        wandb_config: W&B config object
        logger: Optional logger for debug output
    """
    # Skip for non-manifest data sources (e.g., deeplive)
    data_source = data_config.get('data_source', wandb_config.get('data_source', 'manifest'))
    if data_source != 'manifest':
        if logger:
            logger.info(f"Skipping dataset_methods override for data_source='{data_source}' (not manifest-based)")
        return
    
    dataset_methods = wandb_config.get('dataset_methods')
    
    # Debug: Log what we're getting from wandb_config
    if logger:
        logger.info(f"[DEBUG] wandb_config type: {type(wandb_config)}")
        logger.info(f"[DEBUG] wandb_config.get('dataset_methods'): {dataset_methods}")
        # Also try to see if it's accessible differently
        if hasattr(wandb_config, 'keys'):
            logger.info(f"[DEBUG] wandb_config keys: {list(wandb_config.keys())[:20]}...")  # First 20 keys
    
    if dataset_methods:
        if logger:
            logger.info("--- Overriding dataset methods from the run's configuration. ---")
        
        if 'dataset_methods' not in data_config:
            data_config['dataset_methods'] = {}
        
        data_config['dataset_methods'] = dict(dataset_methods)
        if logger:
            logger.info(f"Successfully overrode dataset methods. Keys: {list(data_config['dataset_methods'].keys())}")
    else:
        if logger:
            logger.warning("--- `dataset_methods` NOT found in wandb config! ---")
            logger.info(f"Default FAKE TRAINING methods: {data_config.get('dataset_methods', {}).get('use_fake_methods_for_training', [])}")
            logger.info(f"Default FAKE VALIDATION methods: {data_config.get('dataset_methods', {}).get('use_fake_methods_for_validation', [])}")
        
        # CRITICAL: If dataset_methods not in data_config either, this will cause KeyError later
        if 'dataset_methods' not in data_config or not data_config.get('dataset_methods'):
            if logger:
                logger.error("CRITICAL: `dataset_methods` is missing from both wandb config AND data_config!")
                logger.error("This will cause a KeyError in prepare_splits.py. Check your experiment YAML.")


def apply_wandb_property_balancing_weights(data_config: Dict, wandb_config: Any, logger: Any = None, config: Dict = None) -> None:
    """
    Apply W&B property balancing weights to data_config (in-place).
    
    Args:
        data_config: Data config dict to update
        wandb_config: W&B config object
        logger: Optional logger for debug output
        config: Main config dict (to check lesson_data_control)
    """
    if not data_config.get('property_balancing', {}).get('enabled', False):
        return
    
    if logger:
        logger.info("--- Transferring property balancing weights from run configuration ---")
    
    # Check if lesson_data_control is enabled - check main config first (set directly from single_cfg),
    # then fall back to wandb_config
    uses_lesson_control = False
    if config and config.get('lesson_data_control', {}).get('enabled', False):
        uses_lesson_control = True
    else:
        lesson_data_control = wandb_config.get('lesson_data_control', {})
        uses_lesson_control = lesson_data_control.get('enabled', False) if isinstance(lesson_data_control, dict) else False
    
    if logger:
        logger.info(f"  lesson_data_control enabled: {uses_lesson_control}")
    
    real_weights = wandb_config.get('real_category_weights')
    if real_weights:
        data_config['dataloader_params']['real_category_weights'] = dict(real_weights)
        if logger:
            logger.info(f"Loaded real_category_weights: {data_config['dataloader_params']['real_category_weights']}")
    else:
        if logger:
            if uses_lesson_control:
                logger.info("`real_category_weights` not in config, but lesson_data_control is enabled - weights will be derived from method groups.")
            else:
                logger.warning("`real_category_weights` not found in run config. Dataloader will likely fail.")
        data_config['dataloader_params']['real_category_weights'] = {}
    
    fake_weights = wandb_config.get('fake_category_weights')
    if fake_weights:
        data_config['dataloader_params']['fake_category_weights'] = dict(fake_weights)
        if logger:
            logger.info(f"Loaded fake_category_weights: {data_config['dataloader_params']['fake_category_weights']}")
    else:
        if logger:
            if uses_lesson_control:
                logger.info("`fake_category_weights` not in config, but lesson_data_control is enabled - weights will be derived from method groups.")
            else:
                logger.warning("`fake_category_weights` not found in run config. Dataloader will likely fail.")
        data_config['dataloader_params']['fake_category_weights'] = {}


def apply_wandb_backbone_params(config: Dict, wandb_config: Any, logger: Any = None) -> None:
    """
    Apply W&B backbone parameters to config (in-place).
    
    This function handles backbone configuration including:
    - Backbone type (clip, openclip, siglip, dinov2)
    - Variant (ViT-B-16, ViT-L-14, etc.)
    - Source (openai, laion, etc.)
    - Resolution (224, 336)
    - Auto-resolution of GCS paths from backbone registry
    - Normalization values
    
    Args:
        config: Main config dict to update
        wandb_config: W&B config object
        logger: Optional logger for debug output
    """
    # Check if backbone config is provided
    backbone_config = wandb_config.get('backbone', {})
    if not backbone_config:
        if logger:
            logger.info("No backbone override in config. Using defaults from train_config.yaml.")
        return
    
    # Initialize backbone section in config if not present
    if 'backbone' not in config:
        config['backbone'] = {}
    
    # Apply backbone type
    backbone_type = backbone_config.get('type', 'clip')
    config['backbone']['type'] = backbone_type
    
    # Apply variant
    variant = backbone_config.get('variant', 'ViT-L-14')
    config['backbone']['variant'] = variant
    
    # Apply source
    source = backbone_config.get('source', 'openai')
    config['backbone']['source'] = source
    
    # Apply resolution
    resolution = backbone_config.get('resolution', 224)
    config['backbone']['resolution'] = resolution
    config['resolution'] = resolution  # Also set at top level for backward compat
    
    # Resolve GCS paths from backbone registry or use overrides
    gcs_path_override = backbone_config.get('gcs_path_override')
    local_path_override = backbone_config.get('local_path_override')
    
    if gcs_path_override and local_path_override:
        # Use explicit overrides
        gcs_path = gcs_path_override
        local_path = local_path_override
        if logger:
            logger.info(f"Using explicit backbone path overrides: {gcs_path}")
    else:
        # Resolve from backbone registry
        backbone_registry = config.get('backbone_registry', {})
        source_registry = backbone_registry.get(source, {})
        variant_entry = source_registry.get(variant, {})
        
        if variant_entry:
            gcs_path = variant_entry.get('gcs_path')
            local_path = variant_entry.get('local_path')
            # Honor explicit yaml-supplied hidden_size (e.g. P17 intermediate-layer
            # readout sets 768 to override the registry's projected 512). Fall back
            # to registry/default only if the yaml didn't specify.
            explicit_hidden_size = backbone_config.get('hidden_size')
            if explicit_hidden_size is not None:
                hidden_size = explicit_hidden_size
            else:
                hidden_size = variant_entry.get('hidden_size', BACKBONE_HIDDEN_SIZES.get(variant, 1024))
            config['backbone']['hidden_size'] = hidden_size
        else:
            # Fallback: construct path from naming convention
            variant_slug = variant.lower().replace('-', '')
            gcs_path = f"gs://base-checkpoints/effort-aigi/models--{source}--clip-{variant_slug}/"
            local_path = f"./weights/models--{source}--clip-{variant_slug}/"
            if logger:
                logger.warning(f"Backbone {source}/{variant} not in registry. Using constructed path: {gcs_path}")
    
    # Update GCS assets
    if 'gcs_assets' not in config:
        config['gcs_assets'] = {}
    if 'clip_backbone' not in config['gcs_assets']:
        config['gcs_assets']['clip_backbone'] = {}
    
    config['gcs_assets']['clip_backbone']['gcs_path'] = gcs_path
    config['gcs_assets']['clip_backbone']['local_path'] = local_path
    
    # Apply normalization
    norm_config = backbone_config.get('mean') and backbone_config.get('std')
    if norm_config:
        config['backbone']['mean'] = backbone_config['mean']
        config['backbone']['std'] = backbone_config['std']
    else:
        # Auto-set normalization based on backbone type
        norm_defaults = BACKBONE_NORMALIZATION.get(backbone_type, BACKBONE_NORMALIZATION['clip'])
        config['backbone']['mean'] = norm_defaults['mean']
        config['backbone']['std'] = norm_defaults['std']
    
    if logger:
        logger.info(f"--- Backbone Configuration ---")
        logger.info(f"   Type: {backbone_type}")
        logger.info(f"   Variant: {variant}")
        logger.info(f"   Source: {source}")
        logger.info(f"   Resolution: {resolution}")
        logger.info(f"   GCS Path: {gcs_path}")
        logger.info(f"   Hidden Size: {config['backbone'].get('hidden_size', 'default')}")


def apply_wandb_data_source_params(data_config: Dict, wandb_config: Any, logger: Any = None) -> None:
    """
    Apply W&B data source parameters to data_config (in-place).
    
    This function handles data bucket configuration:
    - Main training data bucket
    - OOD (out-of-distribution) test data bucket
    - Manifest paths
    
    Args:
        data_config: Data config dict to update
        wandb_config: W&B config object
        logger: Optional logger for debug output
    """
    # data_source can be a string (e.g., 'deeplive') or a dict with nested config
    data_source_raw = wandb_config.get('data_source', {})
    if isinstance(data_source_raw, str):
        # It's just a string like 'deeplive' or 'manifest'
        data_config['data_source'] = data_source_raw
        data_source_config = {}  # No nested config
        if logger:
            logger.info(f"Data source type: {data_source_raw}")
    else:
        # It's a dict with nested configuration
        data_source_config = data_source_raw
    
    # Check for direct bucket_name override (legacy/shorthand)
    bucket_name = wandb_config.get('bucket_name') or data_source_config.get('bucket_name') if data_source_config else wandb_config.get('bucket_name')
    if bucket_name:
        if 'gcp' not in data_config:
            data_config['gcp'] = {}
        data_config['gcp']['bucket_name'] = bucket_name
        if logger:
            logger.info(f"Overriding bucket_name to: {bucket_name}")
    
    # Check for ood_bucket_name override
    ood_bucket_name = wandb_config.get('ood_bucket_name') or (data_source_config.get('ood_bucket_name') if data_source_config else None)
    if ood_bucket_name:
        if 'gcp' not in data_config:
            data_config['gcp'] = {}
        data_config['gcp']['ood_bucket_name'] = ood_bucket_name
        if logger:
            logger.info(f"Overriding ood_bucket_name to: {ood_bucket_name}")
    
    # Check for manifest path overrides (only relevant for manifest data source)
    manifest_path = data_source_config.get('manifest_path') if data_source_config else None
    if manifest_path:
        data_config['manifest_path'] = manifest_path
        if logger:
            logger.info(f"Overriding manifest_path to: {manifest_path}")
    
    property_manifest_path = data_source_config.get('property_manifest_path') if data_source_config else None
    if property_manifest_path:
        data_config['property_manifest_path'] = property_manifest_path
        if logger:
            logger.info(f"Overriding property_manifest_path to: {property_manifest_path}")


def apply_wandb_resolution_params(config: Dict, wandb_config: Any, logger: Any = None) -> None:
    """
    Apply W&B resolution parameters to config (in-place).
    
    Resolution can be set directly or via backbone config. This function handles
    the direct resolution override case.
    
    Args:
        config: Main config dict to update
        wandb_config: W&B config object
        logger: Optional logger for debug output
    """
    # Direct resolution override (takes precedence over backbone.resolution)
    resolution = wandb_config.get('resolution')
    if resolution is not None:
        config['resolution'] = int(resolution)
        if 'backbone' in config:
            config['backbone']['resolution'] = int(resolution)
        if logger:
            logger.info(f"Overriding resolution to: {resolution}")


def resolve_backbone_paths(config: Dict, logger: Any = None) -> Dict:
    """
    Resolve backbone GCS/local paths from backbone config.
    
    Call this after all backbone params are applied to ensure paths are resolved.
    This is useful when loading configs without W&B (e.g., train_simple.py).
    
    Args:
        config: Main config dict with backbone settings
        logger: Optional logger for debug output
        
    Returns:
        Updated config dict with resolved paths
    """
    backbone = config.get('backbone', {})
    if not backbone:
        return config
    
    source = backbone.get('source', 'openai')
    variant = backbone.get('variant', 'ViT-L-14')
    
    # Check for explicit overrides
    gcs_override = backbone.get('gcs_path_override')
    local_override = backbone.get('local_path_override')
    
    if gcs_override and local_override:
        gcs_path = gcs_override
        local_path = local_override
    else:
        # Resolve from registry
        registry = config.get('backbone_registry', {})
        source_entry = registry.get(source, {})
        variant_entry = source_entry.get(variant, {})
        
        if variant_entry:
            gcs_path = variant_entry.get('gcs_path')
            local_path = variant_entry.get('local_path')
            hidden_size = variant_entry.get('hidden_size', 1024)
            config['backbone']['hidden_size'] = hidden_size
        else:
            # Construct default path
            variant_slug = variant.lower().replace('-', '')
            gcs_path = f"gs://base-checkpoints/effort-aigi/models--{source}--clip-{variant_slug}/"
            local_path = f"./weights/models--{source}--clip-{variant_slug}/"
    
    # Update gcs_assets
    if 'gcs_assets' not in config:
        config['gcs_assets'] = {}
    if 'clip_backbone' not in config['gcs_assets']:
        config['gcs_assets']['clip_backbone'] = {}
    
    config['gcs_assets']['clip_backbone']['gcs_path'] = gcs_path
    config['gcs_assets']['clip_backbone']['local_path'] = local_path
    
    if logger:
        logger.info(f"Resolved backbone path: {gcs_path}")
    
    return config


def apply_all_wandb_overrides(
    config: Dict,
    data_config: Dict,
    wandb_config: Any,
    logger: Any = None
) -> None:
    """
    Apply all W&B overrides to config and data_config (in-place).
    
    This is a convenience function that calls all the individual override functions.
    
    Args:
        config: Main config dict to update
        data_config: Data config dict to update
        wandb_config: W&B config object
        logger: Optional logger for debug output
    """
    # Backbone and resolution overrides (apply early as they affect other settings)
    apply_wandb_backbone_params(config, wandb_config, logger)
    apply_wandb_resolution_params(config, wandb_config, logger)
    
    # Main config overrides
    apply_wandb_optimizer_params(config, wandb_config)
    apply_wandb_loss_params(config, wandb_config)
    apply_wandb_stability_params(config, wandb_config)
    apply_wandb_group_dro_params(config, wandb_config)
    apply_wandb_arcface_params(config, wandb_config, logger)
    apply_wandb_quality_domain_params(config, wandb_config, logger)
    apply_wandb_early_stopping_params(config, wandb_config)
    apply_wandb_curriculum_params(config, wandb_config)
    apply_wandb_augmentation_params(config, wandb_config, logger)
    apply_wandb_lesson_gate_params(config, wandb_config, logger)
    apply_wandb_lesson_data_control_params(config, wandb_config, logger)
    
    # Data config overrides
    apply_wandb_dataloader_params(data_config, wandb_config)
    apply_wandb_data_source_params(data_config, wandb_config, logger)
    apply_wandb_dataset_methods_override(data_config, wandb_config, logger)
    apply_wandb_property_balancing_weights(data_config, wandb_config, logger, config)
    
    # Set metric scoring
    config['metric_scoring'] = 'auc'


def generate_run_name(config: Dict, wandb_config: Any) -> str:
    """
    Generate a descriptive run name based on configuration.
    
    Args:
        config: Main config dict
        wandb_config: W&B config object
        
    Returns:
        Generated run name string
    """
    import time
    
    if wandb_config.get("name"):
        timestamp = time.strftime("%m%d-%H%M")
        return f"{wandb_config.get('name')}_{timestamp}"
    
    model_name = config.get('model_name', 'model')
    strategy = wandb_config.get('dataloader_strategy', 'property_balancing')
    
    if strategy == 'frame_level':
        batch_info = f"frames{wandb_config.get('frames_per_batch', 64)}"
    else:
        batch_info = f"vids{wandb_config.get('videos_per_batch', 8)}x{wandb_config.get('frames_per_video', 8)}f"
    
    lr = wandb_config.get('learning_rate', 1e-4)
    wd = wandb_config.get('weight_decay', 0.05)
    local_rank = wandb_config.get('rank', 1023)
    
    run_name = (
        f"{model_name}"
        f"_{strategy}"
        f"_{batch_info}"
        f"_lr{lr:.0e}"
        f"_wd{wd:.0e}"
        f"_r{local_rank}"
    ).replace("+", "")
    
    return run_name


def create_curated_config_log(config: Dict, data_config: Dict) -> Dict:
    """
    Create a curated config snapshot for W&B logging/filtering.
    
    Args:
        config: Main config dict
        data_config: Data config dict
        
    Returns:
        Curated config dict for logging
    """
    return {
        'metric_scoring': config.get('metric_scoring'),
        'nEpochs': config.get('nEpochs'),
        'model_name': config.get('model_name'),
        'use_quality_domain_head': config.get('use_quality_domain_head'),
        'quality_domain_count': config.get('quality_domain_count'),
        'quality_head_hidden_dim': config.get('quality_head_hidden_dim'),
        'quality_domain_loss_weight': config.get('quality_domain_loss_weight'),
        'quality_domain_require_labels': config.get('quality_domain_require_labels'),
        'data_params': data_config.get('data_params'),
        'dataloader_params': data_config.get('dataloader_params'),
        'gcs_base_checkpoint': config.get('gcs_assets', {}).get('base_checkpoint', {}).get('gcs_path', 'N/A'),
    }
