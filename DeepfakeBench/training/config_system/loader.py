"""
Configuration loader.

Provides functions to:
- Load and merge multiple YAML config files
- Apply W&B overrides with proper key mapping
- Create typed TrainingConfig instances
"""

import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import fields, is_dataclass

from .schema import (
    TrainingConfig,
    ModelConfig,
    OptimizerConfig,
    AdamConfig,
    SGDConfig,
    SchedulerConfig,
    TrainingParams,
    EarlyStoppingConfig,
    GCSAsset,
    GCSAssetsConfig,
    CheckpointingConfig,
    DataConfig,
    DataloaderConfig,
    PropertyBalancingConfig,
    MethodsConfig,
    AugmentationConfig,
    AugmentationParams,
    LossConfig,
    ArcFaceConfig,
    GroupDROConfig,
    LessonGateConfig,
    CurriculumConfig,
)


# ==============================================================================
# W&B Key Mapping
# ==============================================================================
# Maps W&B config keys to their location in TrainingConfig
# Format: 'wandb_key': ('config_section', 'config_key') or callable

WANDB_KEY_MAP: Dict[str, tuple] = {
    # Optimizer
    'learning_rate': ('optimizer.adam', 'lr'),
    'optimizer_eps': ('optimizer.adam', 'eps'),
    'weight_decay': ('optimizer.adam', 'weight_decay'),
    
    # Scheduler
    'lr_scheduler': ('scheduler', 'type'),
    'total_training_steps': ('scheduler', 'total_training_steps'),
    'lr_scheduler_warmup_steps': ('scheduler', 'warmup_steps'),
    
    # Training
    'nEpochs': ('training', 'epochs'),
    'gradient_clip_val': ('training', 'gradient_clip_val'),
    'metric_scoring': ('training', 'metric_scoring'),
    
    # Early stopping
    'early_stopping_enabled': ('early_stopping', 'enabled'),
    'early_stopping_patience': ('early_stopping', 'patience'),
    'early_stopping_min_delta': ('early_stopping', 'min_delta'),
    
    # Checkpointing
    'load_base_checkpoint': ('checkpointing', 'load_base_checkpoint'),
    'gcs_base_checkpoint': ('gcs_assets.base_checkpoint', 'gcs_path'),
    
    # Data
    'seed': ('data', 'seed'),
    'val_split_ratio': ('data', 'val_split_ratio'),
    'data_subset_percentage': ('data', 'data_subset_percentage'),
    
    # Dataloader
    'dataloader_strategy': ('dataloader', 'strategy'),
    'frames_per_batch': ('dataloader', 'frames_per_batch'),
    'videos_per_batch': ('dataloader', 'videos_per_batch'),
    'frames_per_video': ('dataloader', 'frames_per_video'),
    'real_label_ratio': ('dataloader', 'real_label_ratio'),
    'test_batch_size': ('dataloader', 'test_batch_size'),
    'num_workers': ('dataloader', 'num_workers'),
    'prefetch_factor': ('dataloader', 'prefetch_factor'),
    'evaluation_frequency': ('dataloader', 'evaluation_frequency'),
    
    # Property balancing
    'property_balancing_enabled': ('property_balancing', 'enabled'),
    'real_category_weights': ('property_balancing', 'real_category_weights'),
    'fake_category_weights': ('property_balancing', 'fake_category_weights'),
    
    # Augmentation
    'use_data_augmentation': ('augmentation', 'enabled'),
    'augmentation_version': ('augmentation', 'version'),
    'use_geometric': ('augmentation.params', 'use_geometric'),
    'use_advanced_noise': ('augmentation.params', 'use_advanced_noise'),
    'use_color_jitter': ('augmentation.params', 'use_color_jitter'),
    'use_occlusion': ('augmentation.params', 'use_occlusion'),
    'sharpness_adjust_prob': ('augmentation.params', 'sharpness_adjust_prob'),
    'occlusion_prob': ('augmentation.params', 'occlusion_prob'),
    
    # Loss
    'use_focal_loss': ('loss', 'use_focal_loss'),
    'focal_loss_gamma': ('loss', 'focal_loss_gamma'),
    'focal_loss_alpha': ('loss', 'focal_loss_alpha'),
    'lambda_reg': ('model', 'lambda_reg'),
    'rank': ('model', 'rank'),
    
    # ArcFace
    'use_arcface_head': ('arcface', 'enabled'),
    'train_arcface': ('arcface', 'trainable'),
    'arcface_s': ('arcface', 'scale'),
    'arcface_m': ('arcface', 'margin'),
    's_start': ('arcface', 's_start'),
    's_end': ('arcface', 's_end'),
    'anneal_steps': ('arcface', 'anneal_steps'),
    
    # Group-DRO
    'use_group_dro': ('group_dro', 'enabled'),
    'group_dro_beta': ('group_dro', 'beta'),
    'group_dro_clip_min': ('group_dro', 'clip_min'),
    'group_dro_clip_max': ('group_dro', 'clip_max'),
    'group_dro_ema_alpha': ('group_dro', 'ema_alpha'),
    
    # Curriculum
    'max_train_steps': ('curriculum', 'max_train_steps'),
    'evaluate_every_steps': ('curriculum', 'evaluate_every_steps'),
    'lesson_gate_enabled': ('curriculum.lesson_gate', 'enabled'),
    'lesson_data_control_enabled': ('curriculum', 'lesson_data_control_enabled'),
}


# ==============================================================================
# YAML Loading
# ==============================================================================

def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file and return as dictionary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Values from 'override' take precedence. Nested dicts are merged recursively.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ==============================================================================
# Dict to Dataclass Conversion
# ==============================================================================

def dict_to_dataclass(data: Dict[str, Any], cls: type) -> Any:
    """
    Convert a dictionary to a dataclass instance.
    
    Handles nested dataclasses and type conversion.
    """
    if not is_dataclass(cls):
        return data
    
    field_values = {}
    for f in fields(cls):
        if f.name in data:
            value = data[f.name]
            # Handle nested dataclasses
            if is_dataclass(f.type) and isinstance(value, dict):
                value = dict_to_dataclass(value, f.type)
            # Handle Optional types
            elif hasattr(f.type, '__origin__') and f.type.__origin__ is Union:
                # Get the non-None type from Optional
                args = [a for a in f.type.__args__ if a is not type(None)]
                if args and is_dataclass(args[0]) and isinstance(value, dict):
                    value = dict_to_dataclass(value, args[0])
            field_values[f.name] = value
    
    return cls(**field_values)


# ==============================================================================
# Config Loading
# ==============================================================================

def _map_detector_config(detector_dict: Dict) -> Dict:
    """Map detector YAML structure to TrainingConfig structure."""
    result = {}
    
    # Model config
    result['model'] = {
        'model_name': detector_dict.get('model_name', 'effort'),
        'backbone_name': detector_dict.get('backbone_name', 'vit'),
        'resolution': detector_dict.get('resolution', 224),
        'num_classes': detector_dict.get('backbone_config', {}).get('num_classes', 2),
        'pretrained': detector_dict.get('pretrained'),
        'mean': detector_dict.get('mean', [0.48145466, 0.4578275, 0.40821073]),
        'std': detector_dict.get('std', [0.26862954, 0.26130258, 0.27577711]),
    }
    
    # Optimizer config
    if 'optimizer' in detector_dict:
        opt = detector_dict['optimizer']
        result['optimizer'] = {
            'type': opt.get('type', 'adam'),
            'adam': opt.get('adam', {}),
            'sgd': opt.get('sgd', {}),
        }
    
    # Scheduler config
    result['scheduler'] = {
        'type': detector_dict.get('lr_scheduler'),
    }
    
    # Training params
    result['training'] = {
        'epochs': detector_dict.get('nEpochs', 10),
        'start_epoch': detector_dict.get('start_epoch', 0),
        'manual_seed': detector_dict.get('manualSeed', 1024),
        'cuda': detector_dict.get('cuda', True),
        'cudnn': detector_dict.get('cudnn', True),
        'ngpu': detector_dict.get('ngpu', 1),
        'log_dir': detector_dict.get('log_dir', './logs'),
        'rec_iter': detector_dict.get('rec_iter', 100),
        'save_epoch': detector_dict.get('save_epoch', 1),
        'save_ckpt': detector_dict.get('save_ckpt', True),
        'save_feat': detector_dict.get('save_feat', True),
        'save_avg': detector_dict.get('save_avg', True),
    }
    
    # Augmentation
    result['augmentation'] = {
        'enabled': detector_dict.get('use_data_augmentation', True),
    }
    
    # Loss
    result['loss'] = {
        'type': detector_dict.get('loss_func', 'cross_entropy'),
    }
    
    return result


def _map_train_config(train_dict: Dict) -> Dict:
    """Map train_config YAML structure to TrainingConfig structure."""
    result = {}
    
    # Mode and flags
    result['mode'] = train_dict.get('mode', 'train')
    result['dry_run'] = train_dict.get('dry_run', False)
    
    # GCS assets
    if 'gcs_assets' in train_dict:
        gcs = train_dict['gcs_assets']
        result['gcs_assets'] = {}
        for key, asset in gcs.items():
            if isinstance(asset, dict):
                result['gcs_assets'][key] = {
                    'gcs_path': asset.get('gcs_path'),
                    'local_path': asset.get('local_path'),
                    'files': asset.get('files'),
                }
    
    # Checkpointing
    if 'checkpointing' in train_dict:
        result['checkpointing'] = {
            'gcs_prefix': train_dict['checkpointing'].get('gcs_prefix'),
        }
    
    # Early stopping
    if 'early_stopping' in train_dict:
        es = train_dict['early_stopping']
        result['early_stopping'] = {
            'enabled': es.get('enabled', True),
            'patience': es.get('patience', 4),
            'min_delta': es.get('min_delta', 0.001),
            'mode': es.get('mode', 'max'),
            'monitor': es.get('monitor', 'val/auc'),
        }
    
    # Label dict
    if 'label_dict' in train_dict:
        result['label_dict'] = train_dict['label_dict']
    
    return result


def _map_dataloader_config(data_dict: Dict) -> Dict:
    """Map dataloader_config YAML structure to TrainingConfig structure."""
    result = {}
    
    # GCP bucket info -> data config
    if 'gcp' in data_dict:
        gcp = data_dict['gcp']
        result['data'] = {
            'bucket_name': gcp.get('bucket_name'),
            'ood_bucket_name': gcp.get('ood_bucket_name'),
        }
    
    # Data params
    if 'data_params' in data_dict:
        dp = data_dict['data_params']
        if 'data' not in result:
            result['data'] = {}
        result['data'].update({
            'seed': dp.get('seed', 737),
            'val_split_ratio': dp.get('val_split_ratio', 0.3),
            'data_subset_percentage': dp.get('data_subset_percentage', 1.0),
            'num_frames_per_video': dp.get('num_frames_per_video', 8),
        })
    
    # Dataloader params
    if 'dataloader_params' in data_dict:
        dlp = data_dict['dataloader_params']
        result['dataloader'] = {
            'batch_size': dlp.get('batch_size', 4),
            'test_batch_size': dlp.get('test_batch_size'),
            'num_workers': dlp.get('num_workers', 2),
            'prefetch_factor': dlp.get('prefetch_factor', 3),
        }
    
    # Property balancing
    if 'property_balancing' in data_dict:
        pb = data_dict['property_balancing']
        result['property_balancing'] = {
            'enabled': pb.get('enabled', True),
            'frame_properties_parquet_path': pb.get('frame_properties_parquet_path'),
        }
    
    # Methods
    if 'methods' in data_dict:
        m = data_dict['methods']
        result['methods'] = {
            'use_real_sources': m.get('use_real_sources', []),
            'use_fake_methods_for_training': m.get('use_fake_methods_for_training', []),
            'use_fake_methods_for_validation': m.get('use_fake_methods_for_validation', []),
            'use_real_methods_for_validation_only': m.get('use_real_methods_for_validation_only', []),
            'method_multipliers': m.get('method_multipliers', {}),
        }
    
    return result


# ==============================================================================
# Default Config Path
# ==============================================================================

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / 'config' / 'defaults.yaml'


def load_config(
    defaults_path: Optional[Union[str, Path]] = None,
    detector_path: Optional[Union[str, Path]] = None,
    train_config_path: Optional[Union[str, Path]] = None,
    dataloader_config_path: Optional[Union[str, Path]] = None,
    wandb_config: Optional[Any] = None,
    extra_overrides: Optional[Dict[str, Any]] = None,
) -> TrainingConfig:
    """
    Load and merge configuration from multiple sources.
    
    Priority (highest to lowest):
    1. extra_overrides (dict)
    2. wandb_config (W&B sweep/run config)
    3. dataloader_config YAML
    4. train_config YAML
    5. detector YAML
    6. defaults.yaml (base defaults)
    7. Default values in schema dataclasses
    
    Args:
        defaults_path: Path to defaults config (defaults to config/defaults.yaml)
        detector_path: Path to detector config (e.g., config/detector/effort.yaml)
        train_config_path: Path to training config (e.g., config/train_config.yaml)
        dataloader_config_path: Path to dataloader config (e.g., config/dataloader_config.yml)
        wandb_config: W&B config object (wandb.config)
        extra_overrides: Additional overrides as a dictionary
    
    Returns:
        TrainingConfig: Fully populated configuration object
    """
    merged = {}
    
    # Load defaults first (base layer)
    defaults_file = defaults_path or DEFAULT_CONFIG_PATH
    if Path(defaults_file).exists():
        defaults_dict = load_yaml(defaults_file)
        merged = deep_merge(merged, defaults_dict)
    
    # Load and map detector config
    if detector_path:
        detector_dict = load_yaml(detector_path)
        merged = deep_merge(merged, _map_detector_config(detector_dict))
    
    # Load and map train config
    if train_config_path:
        train_dict = load_yaml(train_config_path)
        merged = deep_merge(merged, _map_train_config(train_dict))
    
    # Load and map dataloader config
    if dataloader_config_path:
        data_dict = load_yaml(dataloader_config_path)
        merged = deep_merge(merged, _map_dataloader_config(data_dict))
    
    # Apply W&B overrides
    if wandb_config is not None:
        merged = merge_wandb_overrides(merged, wandb_config)
    
    # Apply extra overrides
    if extra_overrides:
        merged = deep_merge(merged, extra_overrides)
    
    # Convert to TrainingConfig
    return _dict_to_training_config(merged)


def merge_wandb_overrides(config: Dict, wandb_config: Any) -> Dict:
    """
    Apply W&B config overrides to the config dictionary.
    
    Uses WANDB_KEY_MAP to translate W&B keys to config paths.
    """
    result = config.copy()
    
    for wandb_key, (config_path, config_key) in WANDB_KEY_MAP.items():
        # Get value from W&B config
        value = getattr(wandb_config, wandb_key, None)
        if value is None:
            continue
        
        # Navigate to the correct nested location
        parts = config_path.split('.')
        target = result
        for part in parts:
            if part not in target:
                target[part] = {}
            target = target[part]
        
        # Set the value
        target[config_key] = value
    
    return result


def _dict_to_training_config(data: Dict) -> TrainingConfig:
    """Convert a merged dictionary to a TrainingConfig instance."""
    # Build each sub-config
    model = dict_to_dataclass(data.get('model', {}), ModelConfig)
    
    # Handle optimizer with nested configs
    opt_data = data.get('optimizer', {})
    adam = dict_to_dataclass(opt_data.get('adam', {}), AdamConfig)
    sgd = dict_to_dataclass(opt_data.get('sgd', {}), SGDConfig)
    optimizer = OptimizerConfig(
        type=opt_data.get('type', 'adam'),
        adam=adam,
        sgd=sgd,
    )
    
    scheduler = dict_to_dataclass(data.get('scheduler', {}), SchedulerConfig)
    training = dict_to_dataclass(data.get('training', {}), TrainingParams)
    early_stopping = dict_to_dataclass(data.get('early_stopping', {}), EarlyStoppingConfig)
    
    # Handle GCS assets
    gcs_data = data.get('gcs_assets', {})
    gcs_assets = GCSAssetsConfig(
        base_checkpoint=dict_to_dataclass(gcs_data.get('base_checkpoint', {}), GCSAsset) if gcs_data.get('base_checkpoint') else None,
        clip_backbone=dict_to_dataclass(gcs_data.get('clip_backbone', {}), GCSAsset) if gcs_data.get('clip_backbone') else None,
        frame_manifest_json=dict_to_dataclass(gcs_data.get('frame_manifest_json', {}), GCSAsset) if gcs_data.get('frame_manifest_json') else None,
        property_manifest_parquet=dict_to_dataclass(gcs_data.get('property_manifest_parquet', {}), GCSAsset) if gcs_data.get('property_manifest_parquet') else None,
    )
    
    checkpointing = dict_to_dataclass(data.get('checkpointing', {}), CheckpointingConfig)
    data_config = dict_to_dataclass(data.get('data', {}), DataConfig)
    dataloader = dict_to_dataclass(data.get('dataloader', {}), DataloaderConfig)
    property_balancing = dict_to_dataclass(data.get('property_balancing', {}), PropertyBalancingConfig)
    methods = dict_to_dataclass(data.get('methods', {}), MethodsConfig)
    
    # Handle augmentation with nested params
    aug_data = data.get('augmentation', {})
    aug_params = dict_to_dataclass(aug_data.get('params', {}), AugmentationParams)
    augmentation = AugmentationConfig(
        enabled=aug_data.get('enabled', True),
        version=aug_data.get('version', 'surgical'),
        params=aug_params,
    )
    
    loss = dict_to_dataclass(data.get('loss', {}), LossConfig)
    arcface = dict_to_dataclass(data.get('arcface', {}), ArcFaceConfig)
    group_dro = dict_to_dataclass(data.get('group_dro', {}), GroupDROConfig)
    
    # Handle curriculum with nested lesson_gate
    curr_data = data.get('curriculum', {})
    lesson_gate = dict_to_dataclass(curr_data.get('lesson_gate', {}), LessonGateConfig)
    curriculum = CurriculumConfig(
        max_train_steps=curr_data.get('max_train_steps'),
        evaluate_every_steps=curr_data.get('evaluate_every_steps'),
        lesson_gate=lesson_gate,
        lesson_data_control_enabled=curr_data.get('lesson_data_control_enabled', False),
    )
    
    return TrainingConfig(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training=training,
        early_stopping=early_stopping,
        gcs_assets=gcs_assets,
        checkpointing=checkpointing,
        data=data_config,
        dataloader=dataloader,
        property_balancing=property_balancing,
        methods=methods,
        augmentation=augmentation,
        loss=loss,
        arcface=arcface,
        group_dro=group_dro,
        curriculum=curriculum,
        label_dict=data.get('label_dict', {}),
        mode=data.get('mode', 'train'),
        dry_run=data.get('dry_run', False),
        ddp=data.get('ddp', False),
        local_rank=data.get('local_rank', 0),
    )
