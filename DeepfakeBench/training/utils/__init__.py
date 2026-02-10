# Utils module for training pipeline
# Contains GCS helpers, setup utilities, config helpers, and common functions

from .gcs import download_gcs_asset, download_assets_from_gcs
from .setup import init_seed, choose_optimizer, choose_scheduler, choose_metric
from .config_helpers import (
    # Constants
    BACKBONE_NORMALIZATION,
    BACKBONE_HIDDEN_SIZES,
    # Base loading
    load_base_configs,
    # Optimizer & training params
    apply_wandb_optimizer_params,
    apply_wandb_loss_params,
    apply_wandb_group_dro_params,
    apply_wandb_arcface_params,
    apply_wandb_early_stopping_params,
    apply_wandb_curriculum_params,
    # Dataloader params
    apply_wandb_dataloader_params,
    apply_wandb_augmentation_params,
    # Lesson/curriculum params
    apply_wandb_lesson_gate_params,
    apply_wandb_lesson_data_control_params,
    apply_wandb_dataset_methods_override,
    apply_wandb_property_balancing_weights,
    # NEW: Backbone & data source params
    apply_wandb_backbone_params,
    apply_wandb_data_source_params,
    apply_wandb_resolution_params,
    resolve_backbone_paths,
    # Aggregate functions
    apply_all_wandb_overrides,
    generate_run_name,
    create_curated_config_log,
)

__all__ = [
    # Constants
    'BACKBONE_NORMALIZATION',
    'BACKBONE_HIDDEN_SIZES',
    # GCS utilities
    'download_gcs_asset',
    'download_assets_from_gcs',
    # Setup utilities
    'init_seed',
    'choose_optimizer', 
    'choose_scheduler',
    'choose_metric',
    # Config helpers - base loading
    'load_base_configs',
    # Config helpers - optimizer & training
    'apply_wandb_optimizer_params',
    'apply_wandb_loss_params',
    'apply_wandb_group_dro_params',
    'apply_wandb_arcface_params',
    'apply_wandb_early_stopping_params',
    'apply_wandb_curriculum_params',
    # Config helpers - dataloader
    'apply_wandb_dataloader_params',
    'apply_wandb_augmentation_params',
    # Config helpers - lesson/curriculum
    'apply_wandb_lesson_gate_params',
    'apply_wandb_lesson_data_control_params',
    'apply_wandb_dataset_methods_override',
    'apply_wandb_property_balancing_weights',
    # Config helpers - backbone & data source (NEW)
    'apply_wandb_backbone_params',
    'apply_wandb_data_source_params',
    'apply_wandb_resolution_params',
    'resolve_backbone_paths',
    # Config helpers - aggregate
    'apply_all_wandb_overrides',
    'generate_run_name',
    'create_curated_config_log',
]
