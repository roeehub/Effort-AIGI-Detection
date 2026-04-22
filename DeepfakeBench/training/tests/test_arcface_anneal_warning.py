"""
Tests for ArcFaceMixin anneal-vs-training mismatch warning.

Packet 3 YAMLs had anneal_steps=15000 but total_training_steps=10000, so the
ArcFace scale anneal never completed. This warning flags that silently-broken
configuration at init time so future yamls don't inherit it.
"""
from __future__ import annotations

from unittest.mock import Mock


def test_arcface_warns_when_anneal_exceeds_total_training():
    """When anneal_steps > total_training_steps * 1.1, a warning fires."""
    from trainer.mixins.arcface import ArcFaceMixin

    class MockTrainer(ArcFaceMixin):
        def __init__(self):
            self.config = {
                "use_arcface_head": True,
                "train_arcface": True,
                "s_start": 6.0,
                "s_end": 12.0,
                "anneal_steps": 15000,
                "total_training_steps": 10000,
            }
            self.logger = Mock()
            self.init_arcface()

    trainer = MockTrainer()
    assert trainer.logger.warning.called, "Expected warning when anneal_steps=15000 >> total_training_steps=10000"
    (warn_fmt, *warn_args), _ = trainer.logger.warning.call_args
    # Rendered message should contain the offending numbers.
    rendered = warn_fmt % tuple(warn_args)
    assert "15000" in rendered
    assert "10000" in rendered
    # Effective s_end should be < configured s_end.
    assert "effective s_end" in rendered.lower()


def test_arcface_no_warn_when_anneal_matches_training():
    """anneal_steps <= total_training_steps * 1.1 → no warning."""
    from trainer.mixins.arcface import ArcFaceMixin

    class MockTrainer(ArcFaceMixin):
        def __init__(self):
            self.config = {
                "use_arcface_head": True,
                "train_arcface": True,
                "s_start": 6.0,
                "s_end": 12.0,
                "anneal_steps": 8000,
                "total_training_steps": 10000,
            }
            self.logger = Mock()
            self.init_arcface()

    trainer = MockTrainer()
    assert not trainer.logger.warning.called, "Expected no warning when anneal_steps=8000 < 10000"


def test_arcface_no_warn_when_total_training_steps_missing():
    """If total_training_steps is 0 / missing, skip the check (legacy configs)."""
    from trainer.mixins.arcface import ArcFaceMixin

    class MockTrainer(ArcFaceMixin):
        def __init__(self):
            self.config = {
                "use_arcface_head": True,
                "train_arcface": True,
                "s_start": 6.0,
                "s_end": 12.0,
                "anneal_steps": 15000,
                # no total_training_steps
            }
            self.logger = Mock()
            self.init_arcface()

    trainer = MockTrainer()
    assert not trainer.logger.warning.called, "Expected no warning when total_training_steps is absent"


def test_arcface_no_warn_when_arcface_disabled():
    """use_arcface_head=False → mixin returns early, no warning."""
    from trainer.mixins.arcface import ArcFaceMixin

    class MockTrainer(ArcFaceMixin):
        def __init__(self):
            self.config = {
                "use_arcface_head": False,
                "anneal_steps": 15000,
                "total_training_steps": 10000,
            }
            self.logger = Mock()
            self.init_arcface()

    trainer = MockTrainer()
    assert not trainer.logger.warning.called, "Expected no warning when ArcFace is disabled"


def test_arcface_effective_s_end_math():
    """Verify the reported effective s_end matches the linear anneal math."""
    from trainer.mixins.arcface import ArcFaceMixin

    class MockTrainer(ArcFaceMixin):
        def __init__(self):
            self.config = {
                "use_arcface_head": True,
                "train_arcface": True,
                "s_start": 6.0,
                "s_end": 12.0,
                "anneal_steps": 15000,
                "total_training_steps": 10000,
            }
            self.logger = Mock()
            self.init_arcface()

    trainer = MockTrainer()
    (warn_fmt, *warn_args), _ = trainer.logger.warning.call_args
    rendered = warn_fmt % tuple(warn_args)
    # Expected: s_start + (10000/15000) * (12-6) = 6 + 4 = 10.00
    assert "10.00" in rendered
