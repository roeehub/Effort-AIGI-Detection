"""Smoke test for data/augmentations/face_scale_jitter.py."""
from __future__ import annotations

import logging

import numpy as np

from data.augmentations.face_scale_jitter import (
    apply_face_scale_jitter,
    get_face_scale_jitter_config,
    set_face_scale_jitter_config,
)


def _img(h: int = 224, w: int = 224) -> np.ndarray:
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


class TestFaceScaleJitter:
    def setup_method(self):
        # Always reset config between tests to keep them order-independent.
        set_face_scale_jitter_config(
            enabled=False, scale_limit=0.0, logger=logging.getLogger('test'),
        )

    def test_disabled_returns_input_unchanged(self):
        set_face_scale_jitter_config(
            enabled=False, scale_limit=0.25, logger=logging.getLogger('test'),
        )
        img = _img()
        out = apply_face_scale_jitter(img)
        assert out is img

    def test_zero_limit_disables_even_if_enabled_flag_set(self):
        set_face_scale_jitter_config(
            enabled=True, scale_limit=0.0, logger=logging.getLogger('test'),
        )
        cfg = get_face_scale_jitter_config()
        assert cfg["enabled"] is False  # zero limit collapses to disabled

    def test_enabled_changes_shape(self):
        set_face_scale_jitter_config(
            enabled=True, scale_limit=0.25, logger=logging.getLogger('test'),
        )
        # Run multiple times since scale=1.0 short-circuits to no-op.
        any_changed = False
        for _ in range(20):
            img = _img(224, 224)
            out = apply_face_scale_jitter(img)
            if out.shape != img.shape:
                any_changed = True
                # Shape change must be within the expected range
                h, w = out.shape[:2]
                assert 224 * 0.74 <= h <= 224 * 1.26, f"h={h}"
                assert 224 * 0.74 <= w <= 224 * 1.26, f"w={w}"
        assert any_changed, "20 random draws should produce ≥1 non-unit scale"

    def test_dtype_preserved(self):
        set_face_scale_jitter_config(
            enabled=True, scale_limit=0.25, logger=logging.getLogger('test'),
        )
        img = _img()
        out = apply_face_scale_jitter(img)
        assert out.dtype == np.uint8

    def test_non_array_passes_through(self):
        set_face_scale_jitter_config(
            enabled=True, scale_limit=0.25, logger=logging.getLogger('test'),
        )
        result = apply_face_scale_jitter("not an image")
        assert result == "not an image"

    def test_min_dim_floor(self):
        # Tiny input shouldn't collapse to 0×0 even with extreme scale.
        set_face_scale_jitter_config(
            enabled=True, scale_limit=0.99, logger=logging.getLogger('test'),
        )
        img = _img(16, 16)
        for _ in range(20):
            out = apply_face_scale_jitter(img)
            assert out.shape[0] >= 8
            assert out.shape[1] >= 8
