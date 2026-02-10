"""
Trainer module for DeepfakeBench.

This module provides:
- Trainer: The main training orchestrator
- Mixins: Modular functionality components for building custom trainers

The mixins can be used to compose new trainer classes or as reference
implementations for the functionality they encapsulate.
"""
from trainer.trainer import Trainer
from utils.registry import TRAINER

# Export mixins for composition
from trainer.mixins import (
    CheckpointingMixin,
    EarlyStoppingMixin,
    GroupDROMixin,
    CurriculumMixin,
    ArcFaceMixin,
    ValidationMixin,
    ReportingMixin,
)

__all__ = [
    'Trainer',
    'TRAINER',
    # Mixins
    'CheckpointingMixin',
    'EarlyStoppingMixin',
    'GroupDROMixin',
    'CurriculumMixin',
    'ArcFaceMixin',
    'ValidationMixin',
    'ReportingMixin',
]