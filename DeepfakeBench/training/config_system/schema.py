"""
Configuration schema using dataclasses.

Defines typed configuration classes for all training parameters.
These classes provide:
- Type hints for IDE autocompletion
- Default values
- Clear documentation of all config options
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union


# ==============================================================================
# Model Configuration
# ==============================================================================

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_name: str = "effort"
    backbone_name: str = "vit"
    resolution: int = 224
    num_classes: int = 2
    pretrained: Optional[str] = None
    
    # Normalization (CLIP defaults)
    mean: List[float] = field(default_factory=lambda: [0.48145466, 0.4578275, 0.40821073])
    std: List[float] = field(default_factory=lambda: [0.26862954, 0.26130258, 0.27577711])
    
    # SVD parameters
    rank: int = 1023
    lambda_reg: float = 1.0


# ==============================================================================
# Optimizer Configuration
# ==============================================================================

@dataclass
class AdamConfig:
    """Adam optimizer parameters."""
    lr: float = 0.0002
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0005
    amsgrad: bool = False


@dataclass
class SGDConfig:
    """SGD optimizer parameters."""
    lr: float = 0.0002
    momentum: float = 0.9
    weight_decay: float = 0.0005


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    type: str = "adam"
    adam: AdamConfig = field(default_factory=AdamConfig)
    sgd: SGDConfig = field(default_factory=SGDConfig)
    
    @property
    def lr(self) -> float:
        """Get learning rate from active optimizer."""
        if self.type == "adam":
            return self.adam.lr
        return self.sgd.lr


# ==============================================================================
# Scheduler Configuration
# ==============================================================================

@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    type: Optional[str] = None  # None, 'cosine', 'cosine_with_warmup'
    total_training_steps: Optional[int] = None
    warmup_steps: int = 0


# ==============================================================================
# Training Parameters
# ==============================================================================

@dataclass
class TrainingParams:
    """Core training parameters."""
    epochs: int = 10
    start_epoch: int = 0
    manual_seed: int = 1024
    gradient_clip_val: float = 0.0
    
    # Device
    cuda: bool = True
    cudnn: bool = True
    ngpu: int = 1
    
    # Logging
    log_dir: str = "./logs"
    rec_iter: int = 100
    save_epoch: int = 1
    save_ckpt: bool = True
    save_feat: bool = True
    save_avg: bool = True
    
    # Metric
    metric_scoring: str = "auc"  # 'auc', 'acc', 'eer', 'ap'


# ==============================================================================
# Early Stopping
# ==============================================================================

@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""
    enabled: bool = True
    patience: int = 4
    min_delta: float = 0.001
    mode: str = "max"  # 'max' or 'min'
    monitor: str = "val/auc"


# ==============================================================================
# GCS Assets
# ==============================================================================

@dataclass
class GCSAsset:
    """Single GCS asset configuration."""
    gcs_path: str
    local_path: str
    files: Optional[List[str]] = None  # For directories, specific files to download


@dataclass
class GCSAssetsConfig:
    """GCS asset download configuration."""
    base_checkpoint: Optional[GCSAsset] = None
    clip_backbone: Optional[GCSAsset] = None
    frame_manifest_json: Optional[GCSAsset] = None
    property_manifest_parquet: Optional[GCSAsset] = None


# ==============================================================================
# Checkpointing
# ==============================================================================

@dataclass
class CheckpointingConfig:
    """Checkpoint saving configuration."""
    gcs_prefix: str = "gs://training-job-outputs/best_checkpoints/"
    load_base_checkpoint: bool = False


# ==============================================================================
# Data Configuration
# ==============================================================================

@dataclass
class DataConfig:
    """Data loading and splitting configuration."""
    # GCS bucket
    bucket_name: str = "df40-frames-recropped-rfa85"
    ood_bucket_name: str = "deep-fake-test-10-08-25-frames-v2"
    
    # Data splits
    seed: int = 737
    val_split_ratio: float = 0.3
    data_subset_percentage: float = 1.0
    num_frames_per_video: int = 8


# ==============================================================================
# Dataloader Configuration
# ==============================================================================

@dataclass
class DataloaderConfig:
    """Dataloader parameters."""
    strategy: str = "property_balancing"  # 'frame_level', 'video_level', 'per_method', 'property_balancing'
    batch_size: int = 4
    test_batch_size: Optional[int] = None
    frames_per_batch: Optional[int] = None
    videos_per_batch: int = 8
    frames_per_video: int = 8
    real_label_ratio: Optional[float] = None
    num_workers: int = 2
    prefetch_factor: int = 3
    evaluation_frequency: int = 1


# ==============================================================================
# Property Balancing
# ==============================================================================

@dataclass
class PropertyBalancingConfig:
    """Property-based batch balancing configuration."""
    enabled: bool = True
    frame_properties_parquet_path: Optional[str] = None
    real_category_weights: Dict[str, float] = field(default_factory=dict)
    fake_category_weights: Dict[str, float] = field(default_factory=dict)


# ==============================================================================
# Methods Configuration
# ==============================================================================

@dataclass
class MethodsConfig:
    """Dataset method selection configuration."""
    use_real_sources: List[str] = field(default_factory=list)
    use_fake_methods_for_training: List[str] = field(default_factory=list)
    use_fake_methods_for_validation: List[str] = field(default_factory=list)
    use_real_methods_for_validation_only: List[str] = field(default_factory=list)
    method_multipliers: Dict[str, float] = field(default_factory=dict)


# ==============================================================================
# Augmentation Configuration
# ==============================================================================

@dataclass
class AugmentationParams:
    """Detailed augmentation parameters for surgical pipeline."""
    use_geometric: bool = True
    use_advanced_noise: bool = False
    use_color_jitter: bool = True
    use_occlusion: bool = True
    sharpness_adjust_prob: float = 0.6
    occlusion_prob: float = 0.4


@dataclass
class AugmentationConfig:
    """Augmentation pipeline configuration."""
    enabled: bool = True
    version: Union[int, str] = "surgical"  # 3, 4, 5, 6, 7, 'surgical'
    params: AugmentationParams = field(default_factory=AugmentationParams)


# ==============================================================================
# Loss Configuration
# ==============================================================================

@dataclass
class LossConfig:
    """Loss function configuration."""
    type: str = "cross_entropy"  # 'cross_entropy', 'focal'
    use_focal_loss: bool = False
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: Optional[float] = None


# ==============================================================================
# ArcFace Configuration
# ==============================================================================

@dataclass
class ArcFaceConfig:
    """ArcFace head configuration."""
    enabled: bool = False
    trainable: bool = True
    scale: float = 30.0
    margin: float = 0.35
    # Annealing
    s_start: Optional[float] = None
    s_end: Optional[float] = None
    anneal_steps: int = 0


# ==============================================================================
# Group-DRO Configuration
# ==============================================================================

@dataclass
class GroupDROConfig:
    """Group-DRO training configuration."""
    enabled: bool = False
    beta: float = 3.0
    clip_min: float = 1.0
    clip_max: float = 4.0
    ema_alpha: float = 0.1


# ==============================================================================
# Curriculum Learning
# ==============================================================================

@dataclass
class LessonGateConfig:
    """Lesson gate configuration for curriculum learning."""
    enabled: bool = False
    checks: List[Dict[str, Any]] = field(default_factory=list)
    plateau_check: Dict[str, Any] = field(default_factory=dict)
    guardrail_check: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CurriculumConfig:
    """Curriculum learning configuration."""
    max_train_steps: Optional[int] = None
    evaluate_every_steps: Optional[int] = None
    lesson_gate: LessonGateConfig = field(default_factory=LessonGateConfig)
    lesson_data_control_enabled: bool = False


# ==============================================================================
# Main Training Configuration
# ==============================================================================

@dataclass
class TrainingConfig:
    """
    Complete training configuration.
    
    This is the main config class that contains all sub-configurations.
    Use load_config() to create an instance from YAML files and W&B overrides.
    """
    # Core configs
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingParams = field(default_factory=TrainingParams)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    
    # Data configs
    data: DataConfig = field(default_factory=DataConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    property_balancing: PropertyBalancingConfig = field(default_factory=PropertyBalancingConfig)
    methods: MethodsConfig = field(default_factory=MethodsConfig)
    
    # GCS and checkpointing
    gcs_assets: GCSAssetsConfig = field(default_factory=GCSAssetsConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    
    # Training features
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    arcface: ArcFaceConfig = field(default_factory=ArcFaceConfig)
    group_dro: GroupDROConfig = field(default_factory=GroupDROConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    
    # Label mapping
    label_dict: Dict[str, int] = field(default_factory=dict)
    
    # Runtime flags
    mode: str = "train"
    dry_run: bool = False
    ddp: bool = False
    local_rank: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to nested dictionary for backward compatibility."""
        from dataclasses import asdict
        return asdict(self)
