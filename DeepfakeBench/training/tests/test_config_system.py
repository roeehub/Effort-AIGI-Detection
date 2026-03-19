"""
Tests for the config_system module.
"""
import pytest
from pathlib import Path


class TestConfigLoader:
    """Tests for config loading functionality."""
    
    def test_load_yaml(self):
        """Test that YAML files can be loaded."""
        from config_system.loader import load_yaml, DEFAULT_CONFIG_PATH
        
        if DEFAULT_CONFIG_PATH.exists():
            data = load_yaml(DEFAULT_CONFIG_PATH)
            assert isinstance(data, dict)
    
    def test_deep_merge_simple(self):
        """Test deep merge with simple dicts."""
        from config_system.loader import deep_merge
        
        base = {'a': 1, 'b': 2}
        override = {'b': 3, 'c': 4}
        
        result = deep_merge(base, override)
        
        assert result['a'] == 1
        assert result['b'] == 3
        assert result['c'] == 4
    
    def test_deep_merge_nested(self):
        """Test deep merge with nested dicts."""
        from config_system.loader import deep_merge
        
        base = {
            'level1': {
                'a': 1,
                'b': 2,
                'level2': {'x': 10}
            }
        }
        override = {
            'level1': {
                'b': 20,
                'c': 3,
                'level2': {'y': 20}
            }
        }
        
        result = deep_merge(base, override)
        
        assert result['level1']['a'] == 1      # preserved
        assert result['level1']['b'] == 20     # overridden
        assert result['level1']['c'] == 3      # added
        assert result['level1']['level2']['x'] == 10  # preserved nested
        assert result['level1']['level2']['y'] == 20  # added nested
    
    def test_load_config_with_defaults(self):
        """Test loading config uses defaults."""
        from config_system import load_config
        
        config = load_config()
        
        # Check structure exists
        assert hasattr(config, 'model')
        assert hasattr(config, 'optimizer')
        assert hasattr(config, 'training')
    
    def test_config_has_expected_defaults(self):
        """Test that config has expected default values."""
        from config_system import load_config
        
        config = load_config()
        
        # These should match defaults.yaml
        assert config.model.model_name == "effort"
        assert config.optimizer.type == "adam"
        assert config.training.metric_scoring == "auc"


class TestConfigSchema:
    """Tests for config schema dataclasses."""
    
    def test_model_config_defaults(self):
        """Test ModelConfig has sensible defaults."""
        from config_system.schema import ModelConfig
        
        config = ModelConfig()
        
        assert config.model_name == "effort"
        assert config.resolution == 224
        assert config.num_classes == 2
    
    def test_optimizer_config_defaults(self):
        """Test OptimizerConfig has sensible defaults."""
        from config_system.schema import OptimizerConfig
        
        config = OptimizerConfig()
        
        assert config.type == "adam"
        assert config.adam.lr == 0.0002
    
    def test_training_config_to_dict(self):
        """Test TrainingConfig can be converted to dict."""
        from config_system import TrainingConfig
        
        config = TrainingConfig()
        
        d = config.to_dict()
        
        assert isinstance(d, dict)
        assert 'model' in d
        assert 'optimizer' in d


class TestWandBKeyMapping:
    """Tests for W&B config key mapping."""
    
    def test_wandb_key_map_exists(self):
        """Test WANDB_KEY_MAP is defined."""
        from config_system.loader import WANDB_KEY_MAP
        
        assert isinstance(WANDB_KEY_MAP, dict)
        assert len(WANDB_KEY_MAP) > 0
    
    def test_wandb_key_map_has_essential_keys(self):
        """Test essential W&B keys are mapped."""
        from config_system.loader import WANDB_KEY_MAP
        
        essential_keys = [
            'learning_rate',
            'nEpochs',
            'dataloader_strategy',
            'seed',
        ]
        
        for key in essential_keys:
            assert key in WANDB_KEY_MAP, f"Missing W&B key mapping: {key}"
