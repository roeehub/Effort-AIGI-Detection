"""
Tests for utils/config_helpers.py module.

These tests verify the config helper functions work correctly
for setting up training configurations from W&B and other sources.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import os


class TestConfigHelpersImport:
    """Test that config helpers can be imported."""
    
    def test_import_config_helpers(self):
        """Test config_helpers module can be imported."""
        from utils import config_helpers
        assert config_helpers is not None
    
    def test_import_specific_functions(self):
        """Test specific functions can be imported."""
        from utils.config_helpers import (
            load_base_configs,
            apply_wandb_optimizer_params,
            apply_wandb_loss_params,
            apply_wandb_group_dro_params,
            apply_wandb_arcface_params,
            apply_wandb_quality_domain_params,
            apply_wandb_early_stopping_params,
            apply_wandb_curriculum_params,
            apply_wandb_dataloader_params,
            apply_wandb_augmentation_params,
            apply_all_wandb_overrides,
            generate_run_name,
            create_curated_config_log,
        )
        
        # All should be callable
        assert callable(load_base_configs)
        assert callable(apply_wandb_optimizer_params)
        assert callable(apply_wandb_loss_params)
        assert callable(apply_wandb_group_dro_params)
        assert callable(apply_wandb_arcface_params)
        assert callable(apply_wandb_quality_domain_params)
        assert callable(apply_wandb_early_stopping_params)
        assert callable(apply_wandb_curriculum_params)
        assert callable(apply_wandb_dataloader_params)
        assert callable(apply_wandb_augmentation_params)
        assert callable(apply_all_wandb_overrides)
        assert callable(generate_run_name)
        assert callable(create_curated_config_log)


class TestGenerateRunName:
    """Tests for generate_run_name function."""
    
    def test_run_name_with_custom_name(self):
        """Test run name generation with custom name in wandb_config."""
        from utils.config_helpers import generate_run_name
        
        config = {'model_name': 'effort'}
        
        wandb_config = Mock()
        # Use return_value for the 'name' attribute
        wandb_config.get = Mock(side_effect=lambda k, default=None: {
            'name': 'my_custom_run',
        }.get(k, default))
        wandb_config.name = 'my_custom_run'  # Also set as attribute
        
        name = generate_run_name(config, wandb_config)
        
        assert 'my_custom_run' in name
    
    def test_run_name_auto_generated(self):
        """Test run name auto-generation without custom name."""
        from utils.config_helpers import generate_run_name
        
        config = {'model_name': 'effort'}
        
        wandb_config = Mock()
        wandb_config.get = Mock(return_value=None)
        wandb_config.dataloader_strategy = 'video_level'
        wandb_config.videos_per_batch = 8
        wandb_config.frames_per_video = 4
        wandb_config.learning_rate = 1e-4
        wandb_config.weight_decay = 1e-5
        wandb_config.rank = 0
        
        name = generate_run_name(config, wandb_config)
        
        assert 'effort' in name
        assert 'video_level' in name


class TestApplyWandbOptimizerParams:
    """Tests for apply_wandb_optimizer_params function."""
    
    def test_applies_optimizer_params(self):
        """Test optimizer params are applied to config."""
        from utils.config_helpers import apply_wandb_optimizer_params
        
        config = {
            'optimizer': {'adam': {}},
        }
        
        wandb_config = Mock()
        wandb_config.load_base_checkpoint = True
        wandb_config.learning_rate = 1e-4
        wandb_config.optimizer_eps = 1e-8
        wandb_config.weight_decay = 0.01
        wandb_config.nEpochs = 10
        wandb_config.lambda_reg = 0.5
        wandb_config.rank = 0
        wandb_config.get = Mock(side_effect=lambda k, default=None: {
            'lr_scheduler': 'cosine',
            'total_training_steps': 10000,
            'lr_scheduler_warmup_steps': 500,
            'gradient_clip_val': 1.0,
        }.get(k, default))
        
        apply_wandb_optimizer_params(config, wandb_config)
        
        assert config['optimizer']['adam']['lr'] == 1e-4
        assert config['optimizer']['adam']['weight_decay'] == 0.01
        assert config['nEpochs'] == 10


class TestApplyWandbLossParams:
    """Tests for apply_wandb_loss_params function."""
    
    def test_applies_focal_loss_params(self):
        """Test focal loss params are applied to config."""
        from utils.config_helpers import apply_wandb_loss_params
        
        config = {}
        
        wandb_config = Mock()
        wandb_config.get = Mock(side_effect=lambda k, default=None: {
            'use_focal_loss': True,
            'focal_loss_gamma': 2.5,
            'focal_loss_alpha': 0.25,
        }.get(k, default))
        
        apply_wandb_loss_params(config, wandb_config)
        
        assert config['use_focal_loss'] is True
        assert config['focal_loss_gamma'] == 2.5
        assert config['focal_loss_alpha'] == 0.25
    
    def test_handles_null_focal_alpha(self):
        """Test focal loss alpha can be null."""
        from utils.config_helpers import apply_wandb_loss_params
        
        config = {}
        
        wandb_config = Mock()
        wandb_config.get = Mock(side_effect=lambda k, default=None: {
            'use_focal_loss': True,
            'focal_loss_gamma': 2.0,
            'focal_loss_alpha': 'null',
        }.get(k, default))
        
        apply_wandb_loss_params(config, wandb_config)
        
        assert config['focal_loss_alpha'] is None


class TestApplyWandbGroupDROParams:
    """Tests for apply_wandb_group_dro_params function."""
    
    def test_group_dro_disabled(self):
        """Test Group-DRO setup when disabled."""
        from utils.config_helpers import apply_wandb_group_dro_params
        
        config = {}
        
        wandb_config = Mock()
        wandb_config.get = Mock(return_value=False)
        
        apply_wandb_group_dro_params(config, wandb_config)
        
        assert config['use_group_dro'] is False
    
    def test_group_dro_enabled(self):
        """Test Group-DRO setup when enabled."""
        from utils.config_helpers import apply_wandb_group_dro_params
        
        config = {}
        
        wandb_config = Mock()
        wandb_config.get = Mock(side_effect=lambda k, default=None: {
            'use_group_dro': True,
            'group_dro_beta': 3.5,
            'group_dro_clip_min': 0.5,
            'group_dro_clip_max': 5.0,
            'group_dro_ema_alpha': 0.2,
        }.get(k, default))
        
        apply_wandb_group_dro_params(config, wandb_config)
        
        assert config['use_group_dro'] is True
        assert config['group_dro_params']['beta'] == 3.5


class TestApplyWandbArcfaceParams:
    """Tests for apply_wandb_arcface_params function."""
    
    def test_arcface_disabled(self):
        """Test ArcFace setup when disabled."""
        from utils.config_helpers import apply_wandb_arcface_params
        
        config = {}
        
        wandb_config = Mock()
        wandb_config.get = Mock(return_value=False)
        
        apply_wandb_arcface_params(config, wandb_config)
        
        assert config['use_arcface_head'] is False
    
    def test_arcface_enabled(self):
        """Test ArcFace setup when enabled."""
        from utils.config_helpers import apply_wandb_arcface_params
        
        config = {}
        
        wandb_config = Mock()
        wandb_config.get = Mock(side_effect=lambda k, default=None: {
            'use_arcface_head': True,
            'train_arcface': True,
            'arcface_s': 64.0,
            'arcface_m': 0.5,
            's_start': 30.0,
            's_end': 64.0,
            'anneal_steps': 1000,
        }.get(k, default))
        
        apply_wandb_arcface_params(config, wandb_config)
        
        assert config['use_arcface_head'] is True
        assert config['arcface_s'] == 64.0


class TestApplyWandbQualityDomainParams:
    """Tests for apply_wandb_quality_domain_params function."""

    def test_quality_head_disabled(self):
        """Quality head config should stay disabled when not requested."""
        from utils.config_helpers import apply_wandb_quality_domain_params

        config = {}
        wandb_config = Mock()
        wandb_config.get = Mock(return_value=None)

        apply_wandb_quality_domain_params(config, wandb_config)

        assert config['use_quality_domain_head'] is False
        assert config['quality_domain_require_labels'] is True

    def test_quality_head_enabled(self):
        """Quality head config should propagate all key hyperparameters."""
        from utils.config_helpers import apply_wandb_quality_domain_params

        config = {}
        wandb_config = Mock()
        wandb_config.get = Mock(side_effect=lambda k, default=None: {
            'use_quality_domain_head': True,
            'quality_domain_count': 4,
            'quality_head_hidden_dim': 128,
            'quality_domain_loss_weight': 0.25,
            'quality_domain_require_labels': True,
        }.get(k, default))

        apply_wandb_quality_domain_params(config, wandb_config)

        assert config['use_quality_domain_head'] is True
        assert config['quality_domain_count'] == 4
        assert config['quality_head_hidden_dim'] == 128
        assert config['quality_domain_loss_weight'] == 0.25
        assert config['quality_domain_require_labels'] is True


class TestApplyWandbEarlyStoppingParams:
    """Tests for apply_wandb_early_stopping_params function."""
    
    def test_early_stopping_disabled(self):
        """Test early stopping setup when disabled."""
        from utils.config_helpers import apply_wandb_early_stopping_params
        
        config = {}
        
        wandb_config = Mock()
        wandb_config.get = Mock(side_effect=lambda k, default=None: {
            'early_stopping_enabled': False,
            'early_stopping_patience': 3,
            'early_stopping_min_delta': 0.0001,
        }.get(k, default))
        
        apply_wandb_early_stopping_params(config, wandb_config)
        
        assert config['early_stopping']['enabled'] is False
    
    def test_early_stopping_enabled(self):
        """Test early stopping setup when enabled."""
        from utils.config_helpers import apply_wandb_early_stopping_params
        
        config = {}
        
        wandb_config = Mock()
        wandb_config.get = Mock(side_effect=lambda k, default=None: {
            'early_stopping_enabled': True,
            'early_stopping_patience': 5,
            'early_stopping_min_delta': 0.001,
        }.get(k, default))
        
        apply_wandb_early_stopping_params(config, wandb_config)
        
        assert config['early_stopping']['enabled'] is True
        assert config['early_stopping']['patience'] == 5


class TestCreateCuratedConfigLog:
    """Tests for create_curated_config_log function."""
    
    def test_creates_curated_log(self):
        """Test curated config log creation."""
        from utils.config_helpers import create_curated_config_log
        
        config = {
            'metric_scoring': 'auc',
            'nEpochs': 10,
            'model_name': 'effort',
            'gcs_assets': {
                'base_checkpoint': {
                    'gcs_path': 'gs://bucket/checkpoint.pth'
                }
            }
        }
        
        data_config = {
            'data_params': {'some': 'params'},
            'dataloader_params': {'batch_size': 32},
        }
        
        result = create_curated_config_log(config, data_config)
        
        assert result['metric_scoring'] == 'auc'
        assert result['nEpochs'] == 10
        assert result['model_name'] == 'effort'
        assert 'gs://' in result['gcs_base_checkpoint']
        assert 'use_quality_domain_head' in result
        assert 'quality_domain_loss_weight' in result


class TestLoadBaseConfigs:
    """Tests for load_base_configs function."""
    
    def test_load_base_configs_signature(self):
        """Test load_base_configs has expected signature."""
        from utils.config_helpers import load_base_configs
        import inspect
        
        sig = inspect.signature(load_base_configs)
        params = list(sig.parameters.keys())
        
        # Should accept paths for detector, train config, and dataloader config
        assert 'detector_path' in params
        assert 'train_config_path' in params
        assert 'dataloader_config_path' in params


class TestUtilsModuleExports:
    """Test that utils module exports config helpers."""
    
    def test_utils_exports_config_helpers(self):
        """Test utils __init__ exports config helper functions."""
        from utils import (
            apply_all_wandb_overrides,
            generate_run_name,
            load_base_configs,
            create_curated_config_log,
            apply_wandb_quality_domain_params,
        )
        
        assert callable(apply_all_wandb_overrides)
        assert callable(generate_run_name)
        assert callable(load_base_configs)
        assert callable(create_curated_config_log)
        assert callable(apply_wandb_quality_domain_params)
    
    def test_utils_exports_new_config_helpers(self):
        """Test utils __init__ exports new backbone/data source functions."""
        from utils import (
            apply_wandb_backbone_params,
            apply_wandb_data_source_params,
            apply_wandb_resolution_params,
            resolve_backbone_paths,
            BACKBONE_NORMALIZATION,
            BACKBONE_HIDDEN_SIZES,
        )
        
        assert callable(apply_wandb_backbone_params)
        assert callable(apply_wandb_data_source_params)
        assert callable(apply_wandb_resolution_params)
        assert callable(resolve_backbone_paths)
        assert isinstance(BACKBONE_NORMALIZATION, dict)
        assert isinstance(BACKBONE_HIDDEN_SIZES, dict)


class TestBackboneNormalizationConstants:
    """Tests for backbone normalization constants."""
    
    def test_clip_normalization(self):
        """Test CLIP normalization values are present."""
        from utils.config_helpers import BACKBONE_NORMALIZATION
        
        assert 'clip' in BACKBONE_NORMALIZATION
        assert 'mean' in BACKBONE_NORMALIZATION['clip']
        assert 'std' in BACKBONE_NORMALIZATION['clip']
        assert len(BACKBONE_NORMALIZATION['clip']['mean']) == 3
        assert len(BACKBONE_NORMALIZATION['clip']['std']) == 3
    
    def test_all_backbone_types_have_normalization(self):
        """Test all defined backbone types have normalization."""
        from utils.config_helpers import BACKBONE_NORMALIZATION
        
        expected_types = ['clip', 'openclip', 'siglip', 'dinov2']
        for backbone_type in expected_types:
            assert backbone_type in BACKBONE_NORMALIZATION, f"Missing {backbone_type}"
            assert 'mean' in BACKBONE_NORMALIZATION[backbone_type]
            assert 'std' in BACKBONE_NORMALIZATION[backbone_type]


class TestBackboneHiddenSizes:
    """Tests for backbone hidden size constants."""
    
    def test_vit_l_14_hidden_size(self):
        """Test ViT-L-14 hidden size is 1024."""
        from utils.config_helpers import BACKBONE_HIDDEN_SIZES
        
        assert BACKBONE_HIDDEN_SIZES['ViT-L-14'] == 1024
    
    def test_all_variants_have_hidden_sizes(self):
        """Test all common variants have hidden sizes defined."""
        from utils.config_helpers import BACKBONE_HIDDEN_SIZES
        
        expected_variants = ['ViT-B-16', 'ViT-B-32', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14']
        for variant in expected_variants:
            assert variant in BACKBONE_HIDDEN_SIZES, f"Missing {variant}"


class TestApplyWandbBackboneParams:
    """Tests for apply_wandb_backbone_params function."""
    
    def test_no_backbone_config(self):
        """Test when no backbone config is provided."""
        from utils.config_helpers import apply_wandb_backbone_params
        
        config = {}
        wandb_config = Mock()
        wandb_config.get = Mock(return_value=None)
        
        apply_wandb_backbone_params(config, wandb_config)
        
        # Should not add backbone if not provided
        assert 'backbone' not in config or not config.get('backbone')
    
    def test_backbone_config_applied(self):
        """Test backbone config is applied correctly."""
        from utils.config_helpers import apply_wandb_backbone_params
        
        config = {
            'backbone_registry': {
                'openai': {
                    'ViT-L-14-336': {
                        'gcs_path': 'gs://bucket/vit-l-14-336/',
                        'local_path': './weights/vit-l-14-336/',
                        'hidden_size': 1024,
                    }
                }
            },
            'gcs_assets': {}
        }
        
        wandb_config = Mock()
        wandb_config.get = Mock(side_effect=lambda k, default=None: {
            'backbone': {
                'type': 'clip',
                'variant': 'ViT-L-14-336',
                'source': 'openai',
                'resolution': 336,
            }
        }.get(k, default))
        
        apply_wandb_backbone_params(config, wandb_config)
        
        assert config['backbone']['type'] == 'clip'
        assert config['backbone']['variant'] == 'ViT-L-14-336'
        assert config['backbone']['source'] == 'openai'
        assert config['backbone']['resolution'] == 336
        assert config['resolution'] == 336
    
    def test_backbone_gcs_path_resolved(self):
        """Test backbone GCS path is resolved from registry."""
        from utils.config_helpers import apply_wandb_backbone_params
        
        config = {
            'backbone_registry': {
                'openai': {
                    'ViT-B-16': {
                        'gcs_path': 'gs://bucket/vit-b-16/',
                        'local_path': './weights/vit-b-16/',
                        'hidden_size': 768,
                    }
                }
            },
            'gcs_assets': {'clip_backbone': {}}
        }
        
        wandb_config = Mock()
        wandb_config.get = Mock(side_effect=lambda k, default=None: {
            'backbone': {
                'type': 'clip',
                'variant': 'ViT-B-16',
                'source': 'openai',
            }
        }.get(k, default))
        
        apply_wandb_backbone_params(config, wandb_config)
        
        assert config['gcs_assets']['clip_backbone']['gcs_path'] == 'gs://bucket/vit-b-16/'
        assert config['gcs_assets']['clip_backbone']['local_path'] == './weights/vit-b-16/'
    
    def test_backbone_explicit_override(self):
        """Test backbone with explicit path overrides."""
        from utils.config_helpers import apply_wandb_backbone_params
        
        config = {'gcs_assets': {'clip_backbone': {}}}
        
        wandb_config = Mock()
        wandb_config.get = Mock(side_effect=lambda k, default=None: {
            'backbone': {
                'type': 'clip',
                'variant': 'custom',
                'source': 'custom',
                'gcs_path_override': 'gs://my-bucket/custom-backbone/',
                'local_path_override': './weights/custom/',
            }
        }.get(k, default))
        
        apply_wandb_backbone_params(config, wandb_config)
        
        assert config['gcs_assets']['clip_backbone']['gcs_path'] == 'gs://my-bucket/custom-backbone/'
        assert config['gcs_assets']['clip_backbone']['local_path'] == './weights/custom/'
    
    def test_backbone_normalization_auto_set(self):
        """Test normalization values are auto-set based on backbone type."""
        from utils.config_helpers import apply_wandb_backbone_params, BACKBONE_NORMALIZATION
        
        config = {'gcs_assets': {'clip_backbone': {}}}
        
        wandb_config = Mock()
        wandb_config.get = Mock(side_effect=lambda k, default=None: {
            'backbone': {
                'type': 'siglip',
                'variant': 'ViT-L-14',
                'source': 'apple',
            }
        }.get(k, default))
        
        apply_wandb_backbone_params(config, wandb_config)
        
        assert config['backbone']['mean'] == BACKBONE_NORMALIZATION['siglip']['mean']
        assert config['backbone']['std'] == BACKBONE_NORMALIZATION['siglip']['std']


class TestApplyWandbDataSourceParams:
    """Tests for apply_wandb_data_source_params function."""
    
    def test_no_data_source_config(self):
        """Test when no data source config is provided."""
        from utils.config_helpers import apply_wandb_data_source_params
        
        data_config = {'gcp': {'bucket_name': 'original'}}
        wandb_config = Mock()
        wandb_config.get = Mock(return_value=None)
        
        apply_wandb_data_source_params(data_config, wandb_config)
        
        # Original value should be preserved
        assert data_config['gcp']['bucket_name'] == 'original'
    
    def test_bucket_name_override(self):
        """Test bucket_name override."""
        from utils.config_helpers import apply_wandb_data_source_params
        
        data_config = {'gcp': {'bucket_name': 'original'}}
        
        wandb_config = Mock()
        wandb_config.get = Mock(side_effect=lambda k, default=None: {
            'bucket_name': 'new-bucket',
        }.get(k, default))
        
        apply_wandb_data_source_params(data_config, wandb_config)
        
        assert data_config['gcp']['bucket_name'] == 'new-bucket'
    
    def test_ood_bucket_name_override(self):
        """Test ood_bucket_name override."""
        from utils.config_helpers import apply_wandb_data_source_params
        
        data_config = {'gcp': {'ood_bucket_name': 'original-ood'}}
        
        wandb_config = Mock()
        wandb_config.get = Mock(side_effect=lambda k, default=None: {
            'ood_bucket_name': 'new-ood-bucket',
        }.get(k, default))
        
        apply_wandb_data_source_params(data_config, wandb_config)
        
        assert data_config['gcp']['ood_bucket_name'] == 'new-ood-bucket'
    
    def test_data_source_nested_config(self):
        """Test data_source nested config."""
        from utils.config_helpers import apply_wandb_data_source_params
        
        data_config = {'gcp': {}}
        
        wandb_config = Mock()
        wandb_config.get = Mock(side_effect=lambda k, default=None: {
            'data_source': {
                'bucket_name': 'nested-bucket',
                'ood_bucket_name': 'nested-ood',
                'manifest_path': 'gs://bucket/manifest.json',
            }
        }.get(k, default))
        
        apply_wandb_data_source_params(data_config, wandb_config)
        
        assert data_config['gcp']['bucket_name'] == 'nested-bucket'
        assert data_config['gcp']['ood_bucket_name'] == 'nested-ood'
        assert data_config['manifest_path'] == 'gs://bucket/manifest.json'


class TestApplyWandbResolutionParams:
    """Tests for apply_wandb_resolution_params function."""
    
    def test_no_resolution_override(self):
        """Test when no resolution is provided."""
        from utils.config_helpers import apply_wandb_resolution_params
        
        config = {'resolution': 224}
        wandb_config = Mock()
        wandb_config.get = Mock(return_value=None)
        
        apply_wandb_resolution_params(config, wandb_config)
        
        # Original value should be preserved
        assert config['resolution'] == 224
    
    def test_resolution_override(self):
        """Test resolution override."""
        from utils.config_helpers import apply_wandb_resolution_params
        
        config = {'resolution': 224, 'backbone': {}}
        
        wandb_config = Mock()
        wandb_config.get = Mock(side_effect=lambda k, default=None: {
            'resolution': 336,
        }.get(k, default))
        
        apply_wandb_resolution_params(config, wandb_config)
        
        assert config['resolution'] == 336
        assert config['backbone']['resolution'] == 336


class TestResolveBackbonePaths:
    """Tests for resolve_backbone_paths function."""
    
    def test_resolve_from_registry(self):
        """Test path resolution from backbone registry."""
        from utils.config_helpers import resolve_backbone_paths
        
        config = {
            'backbone': {
                'source': 'openai',
                'variant': 'ViT-L-14',
            },
            'backbone_registry': {
                'openai': {
                    'ViT-L-14': {
                        'gcs_path': 'gs://bucket/openai-vit-l-14/',
                        'local_path': './weights/openai-vit-l-14/',
                        'hidden_size': 1024,
                    }
                }
            },
            'gcs_assets': {},
        }
        
        result = resolve_backbone_paths(config)
        
        assert result['gcs_assets']['clip_backbone']['gcs_path'] == 'gs://bucket/openai-vit-l-14/'
        assert result['backbone']['hidden_size'] == 1024
    
    def test_resolve_with_explicit_override(self):
        """Test path resolution with explicit overrides."""
        from utils.config_helpers import resolve_backbone_paths
        
        config = {
            'backbone': {
                'source': 'custom',
                'variant': 'custom',
                'gcs_path_override': 'gs://custom/path/',
                'local_path_override': './custom/path/',
            },
            'gcs_assets': {},
        }
        
        result = resolve_backbone_paths(config)
        
        assert result['gcs_assets']['clip_backbone']['gcs_path'] == 'gs://custom/path/'
        assert result['gcs_assets']['clip_backbone']['local_path'] == './custom/path/'
    
    def test_no_backbone_config(self):
        """Test when no backbone config is present."""
        from utils.config_helpers import resolve_backbone_paths
        
        config = {}
        
        result = resolve_backbone_paths(config)
        
        # Should return config unchanged
        assert result == config


class TestApplyAllWandbOverridesIncludesNewFunctions:
    """Test that apply_all_wandb_overrides includes new functions."""
    
    def test_backbone_params_called(self):
        """Test that backbone params function is called."""
        from utils.config_helpers import apply_all_wandb_overrides
        
        config = {
            'optimizer': {'adam': {}},
            'gcs_assets': {},
        }
        data_config = {
            'dataloader_params': {},
            'data_params': {},
            'property_balancing': {},
            'gcp': {},
        }
        
        wandb_config = Mock()
        wandb_config.get = Mock(side_effect=lambda k, default=None: {
            'backbone': {
                'type': 'clip',
                'variant': 'ViT-B-16',
                'source': 'openai',
            },
            'resolution': 224,
            'use_quality_domain_head': True,
            'quality_domain_count': 4,
            'quality_head_hidden_dim': 128,
            'quality_domain_loss_weight': 0.1,
            'quality_domain_require_labels': True,
        }.get(k, default))
        
        # Set required attributes
        wandb_config.load_base_checkpoint = True
        wandb_config.learning_rate = 1e-4
        wandb_config.optimizer_eps = 1e-8
        wandb_config.weight_decay = 0.01
        wandb_config.nEpochs = 10
        wandb_config.lambda_reg = 1.0
        wandb_config.rank = 1023
        wandb_config.dataloader_strategy = 'video_level'
        wandb_config.frames_per_batch = 32
        wandb_config.videos_per_batch = 8
        wandb_config.frames_per_video = 4
        wandb_config.real_label_ratio = None
        wandb_config.val_split_ratio = 0.1
        wandb_config.evaluation_frequency = 1
        wandb_config.seed = 42
        wandb_config.data_subset_percentage = 1.0
        wandb_config.test_batch_size = 8
        wandb_config.num_workers = 4
        wandb_config.prefetch_factor = 2
        wandb_config.property_balancing_enabled = True
        
        apply_all_wandb_overrides(config, data_config, wandb_config)
        
        # Backbone should be configured
        assert 'backbone' in config
        assert config['backbone']['type'] == 'clip'
        assert config['use_quality_domain_head'] is True
        assert config['quality_domain_count'] == 4
        assert config['quality_head_hidden_dim'] == 128
        assert config['quality_domain_loss_weight'] == 0.1
