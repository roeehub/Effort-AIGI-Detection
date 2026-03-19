"""
Tests for trainer mixins.

These tests verify the mixin classes work correctly in isolation
and when composed together in the Trainer class.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os


class TestMixinImports:
    """Test that all mixins can be imported."""
    
    def test_import_checkpointing_mixin(self):
        """Test CheckpointingMixin can be imported."""
        from trainer.mixins import CheckpointingMixin
        assert CheckpointingMixin is not None
    
    def test_import_early_stopping_mixin(self):
        """Test EarlyStoppingMixin can be imported."""
        from trainer.mixins import EarlyStoppingMixin
        assert EarlyStoppingMixin is not None
    
    def test_import_group_dro_mixin(self):
        """Test GroupDROMixin can be imported."""
        from trainer.mixins import GroupDROMixin
        assert GroupDROMixin is not None
    
    def test_import_curriculum_mixin(self):
        """Test CurriculumMixin can be imported."""
        from trainer.mixins import CurriculumMixin
        assert CurriculumMixin is not None
    
    def test_import_arcface_mixin(self):
        """Test ArcFaceMixin can be imported."""
        from trainer.mixins import ArcFaceMixin
        assert ArcFaceMixin is not None
    
    def test_import_validation_mixin(self):
        """Test ValidationMixin can be imported."""
        from trainer.mixins import ValidationMixin
        assert ValidationMixin is not None
    
    def test_import_reporting_mixin(self):
        """Test ReportingMixin can be imported."""
        from trainer.mixins import ReportingMixin
        assert ReportingMixin is not None
    
    def test_import_all_from_init(self):
        """Test all mixins are exported from __init__."""
        from trainer.mixins import (
            CheckpointingMixin,
            EarlyStoppingMixin,
            GroupDROMixin,
            CurriculumMixin,
            ArcFaceMixin,
            ValidationMixin,
            ReportingMixin,
        )
        assert all([
            CheckpointingMixin,
            EarlyStoppingMixin,
            GroupDROMixin,
            CurriculumMixin,
            ArcFaceMixin,
            ValidationMixin,
            ReportingMixin,
        ])


class TestEarlyStoppingMixin:
    """Test EarlyStoppingMixin functionality."""
    
    def test_init_early_stopping_enabled(self):
        """Test early stopping initialization when enabled."""
        from trainer.mixins import EarlyStoppingMixin
        
        class MockTrainer(EarlyStoppingMixin):
            def __init__(self):
                self.config = {
                    'early_stopping': {
                        'enabled': True,
                        'patience': 5,
                        'min_delta': 0.001,
                        'mode': 'max'
                    }
                }
                self.logger = Mock()
                self.init_early_stopping()
        
        trainer = MockTrainer()
        
        assert trainer.early_stopping_enabled is True
        assert trainer.early_stopping_patience == 5
        assert trainer.early_stopping_min_delta == 0.001
        assert trainer.early_stopping_mode == 'max'
        assert trainer.epochs_without_improvement == 0
    
    def test_init_early_stopping_disabled(self):
        """Test early stopping initialization when disabled."""
        from trainer.mixins import EarlyStoppingMixin
        
        class MockTrainer(EarlyStoppingMixin):
            def __init__(self):
                self.config = {
                    'early_stopping': {'enabled': False}
                }
                self.logger = Mock()
                self.init_early_stopping()
        
        trainer = MockTrainer()
        
        assert trainer.early_stopping_enabled is False
    
    def test_check_early_stopping_returns_false_when_disabled(self):
        """Test check_early_stopping returns False when disabled."""
        from trainer.mixins import EarlyStoppingMixin
        
        class MockTrainer(EarlyStoppingMixin):
            def __init__(self):
                self.config = {'early_stopping': {'enabled': False}}
                self.logger = Mock()
                self.init_early_stopping()
        
        trainer = MockTrainer()
        
        result = trainer.check_early_stopping(0.8)
        assert result is False
    
    def test_check_early_stopping_improvement(self):
        """Test early stopping resets counter on improvement."""
        from trainer.mixins import EarlyStoppingMixin
        
        class MockTrainer(EarlyStoppingMixin):
            def __init__(self):
                self.config = {
                    'early_stopping': {
                        'enabled': True,
                        'patience': 5,
                        'min_delta': 0.001,
                        'mode': 'max'
                    }
                }
                self.logger = Mock()
                self.init_early_stopping()
        
        trainer = MockTrainer()
        
        # First call sets baseline
        trainer.check_early_stopping(0.8)
        assert trainer._early_stopping_best_metric == 0.8
        
        # Second call with improvement
        trainer.check_early_stopping(0.85)
        assert trainer._early_stopping_best_metric == 0.85
        assert trainer.epochs_without_improvement == 0


class TestCurriculumMixin:
    """Test CurriculumMixin functionality."""
    
    def test_init_curriculum_enabled(self):
        """Test curriculum initialization when enabled."""
        from trainer.mixins import CurriculumMixin
        
        class MockTrainer(CurriculumMixin):
            def __init__(self):
                self.config = {
                    'lesson_gate': {
                        'enabled': True,
                        'checks': [{'metric': 'auc', 'threshold': 0.7, 'comparison': '>='}],
                        'plateau_check': {},
                        'guardrail_check': {},
                    }
                }
                self.logger = Mock()
                self.init_curriculum()
        
        trainer = MockTrainer()
        
        assert trainer.gate_enabled is True
        assert len(trainer.gate_checks) == 1
    
    def test_init_curriculum_disabled(self):
        """Test curriculum initialization when disabled."""
        from trainer.mixins import CurriculumMixin
        
        class MockTrainer(CurriculumMixin):
            def __init__(self):
                self.config = {
                    'lesson_gate': {'enabled': False}
                }
                self.logger = Mock()
                self.init_curriculum()
        
        trainer = MockTrainer()
        
        assert trainer.gate_enabled is False
    
    def test_evaluate_lesson_gate_disabled(self):
        """Test evaluate_lesson_gate when disabled."""
        from trainer.mixins import CurriculumMixin
        
        class MockTrainer(CurriculumMixin):
            def __init__(self):
                self.config = {'lesson_gate': {'enabled': False}}
                self.logger = Mock()
                self.init_curriculum()
        
        trainer = MockTrainer()
        
        result = trainer.evaluate_lesson_gate({'auc': 0.8})
        
        assert result['passed'] is False
        assert result['should_end_lesson'] is False
        assert 'disabled' in result['reason'].lower()


class TestArcFaceMixin:
    """Test ArcFaceMixin functionality."""
    
    def test_init_arcface_enabled(self):
        """Test ArcFace initialization when enabled."""
        from trainer.mixins import ArcFaceMixin
        
        class MockTrainer(ArcFaceMixin):
            def __init__(self):
                self.config = {
                    'use_arcface_head': True,
                    'train_arcface': True,
                    's_start': 30.0,
                    's_end': 64.0,
                    'anneal_steps': 1000,
                }
                self.logger = Mock()
                self.init_arcface()
        
        trainer = MockTrainer()
        
        assert trainer.use_arcface_head is True
        assert trainer.arcface_s_start == 30.0
        assert trainer.arcface_s_end == 64.0
        assert trainer.arcface_anneal_steps == 1000
    
    def test_init_arcface_disabled(self):
        """Test ArcFace initialization when disabled."""
        from trainer.mixins import ArcFaceMixin
        
        class MockTrainer(ArcFaceMixin):
            def __init__(self):
                self.config = {'use_arcface_head': False}
                self.logger = Mock()
                self.init_arcface()
        
        trainer = MockTrainer()
        
        assert trainer.use_arcface_head is False
    
    def test_update_arcface_s_disabled(self):
        """Test update_arcface_s returns None when disabled."""
        from trainer.mixins import ArcFaceMixin
        
        class MockTrainer(ArcFaceMixin):
            def __init__(self):
                self.config = {'use_arcface_head': False}
                self.logger = Mock()
                self.init_arcface()
        
        trainer = MockTrainer()
        
        result = trainer.update_arcface_s(100)
        assert result is None


class TestGroupDROMixin:
    """Test GroupDROMixin functionality."""
    
    def test_group_dro_mixin_has_init(self):
        """Test GroupDROMixin has init_group_dro method."""
        from trainer.mixins import GroupDROMixin
        
        assert hasattr(GroupDROMixin, 'init_group_dro')
        assert callable(getattr(GroupDROMixin, 'init_group_dro'))
    
    def test_group_dro_mixin_has_calculate_loss(self):
        """Test GroupDROMixin has calculate_group_dro_loss method."""
        from trainer.mixins import GroupDROMixin
        
        assert hasattr(GroupDROMixin, 'calculate_group_dro_loss')
        assert callable(getattr(GroupDROMixin, 'calculate_group_dro_loss'))


class TestTrainerComposition:
    """Test that mixins work correctly when composed in Trainer."""
    
    def test_trainer_has_all_mixins(self):
        """Test Trainer inherits from all mixins."""
        from trainer.trainer import Trainer
        from trainer.mixins import (
            CheckpointingMixin,
            EarlyStoppingMixin,
            GroupDROMixin,
            CurriculumMixin,
            ArcFaceMixin,
            ValidationMixin,
            ReportingMixin,
        )
        
        # Check inheritance
        assert issubclass(Trainer, CheckpointingMixin)
        assert issubclass(Trainer, EarlyStoppingMixin)
        assert issubclass(Trainer, GroupDROMixin)
        assert issubclass(Trainer, CurriculumMixin)
        assert issubclass(Trainer, ArcFaceMixin)
        assert issubclass(Trainer, ValidationMixin)
        assert issubclass(Trainer, ReportingMixin)
    
    def test_trainer_mixin_methods_available(self):
        """Test mixin methods are available on Trainer class."""
        from trainer.trainer import Trainer
        
        # These methods should be available from mixins
        mixin_methods = [
            'init_early_stopping',
            'init_curriculum',
            'init_arcface',
            'check_early_stopping',
            'evaluate_lesson_gate',
            'update_arcface_s',
        ]
        
        for method_name in mixin_methods:
            assert hasattr(Trainer, method_name), f"Missing method: {method_name}"
