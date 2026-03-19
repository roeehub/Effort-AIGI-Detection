"""
Trainer mixins for modular functionality.

This module provides mixin classes that can be composed with the base Trainer
to add specific capabilities:

- CheckpointingMixin: Model saving, top-N checkpoints, GCS upload
- EarlyStoppingMixin: Patience-based early stopping
- GroupDROMixin: Distributionally Robust Optimization
- CurriculumMixin: Lesson gates and curriculum learning
- ArcFaceMixin: ArcFace head parameter annealing
- ValidationMixin: Validation state management and utilities
- ReportingMixin: Report generation and GCS upload

Usage:
    class MyTrainer(CheckpointingMixin, EarlyStoppingMixin, BaseTrainer):
        pass
"""

from .checkpointing import CheckpointingMixin
from .early_stopping import EarlyStoppingMixin
from .group_dro import GroupDROMixin
from .curriculum import CurriculumMixin
from .arcface import ArcFaceMixin
from .validation import ValidationMixin
from .reporting import ReportingMixin
from .stability import StabilityRegMixin

__all__ = [
    'CheckpointingMixin',
    'EarlyStoppingMixin',
    'GroupDROMixin',
    'CurriculumMixin',
    'ArcFaceMixin',
    'ValidationMixin',
    'ReportingMixin',
    'StabilityRegMixin',
]
