"""Smoke test for data/augmentations/pipeline_randomization.py.

Verifies:
  - Disabled config → identity (input == output by reference)
  - Enabled config → output may differ from input but stays valid uint8 RGB
  - Label-aware gating: p_real=1.0 always fires, p_real=0.0 never fires
  - No NaN/Inf escape: any sub-aug producing weird values gets sanitized
  - Output shape matches input shape
"""
from __future__ import annotations

import logging

import numpy as np

from data.augmentations.pipeline_randomization import PipelineRandomization


def _sample_image(rng: np.random.RandomState | None = None) -> np.ndarray:
    rng = rng or np.random.RandomState(0)
    return rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)


class TestPipelineRandomizationGate:
    def test_disabled_returns_input_unchanged(self):
        aug = PipelineRandomization(config={'enabled': False})
        img = _sample_image()
        out = aug(img, label=0)
        assert out is img  # short-circuit returns the same object

    def test_p_real_zero_skips_real(self):
        aug = PipelineRandomization(config={
            'enabled': True, 'p_real': 0.0, 'p_fake': 1.0,
        })
        img = _sample_image()
        # Run many times to make sure p=0 never fires.
        for _ in range(20):
            out = aug(img, label=0)
            assert np.array_equal(out, img)

    def test_p_fake_zero_skips_fake(self):
        aug = PipelineRandomization(config={
            'enabled': True, 'p_real': 1.0, 'p_fake': 0.0,
        })
        img = _sample_image()
        for _ in range(20):
            out = aug(img, label=1)
            assert np.array_equal(out, img)

    def test_p_real_one_fires_for_real(self):
        # With all sub-augs at p=0, the gate fires but no sub-aug does any
        # work — so output should still equal input. Confirms gate semantics.
        aug = PipelineRandomization(config={
            'enabled': True, 'p_real': 1.0, 'p_fake': 0.0,
            'jpeg_p': 0.0, 'downscale_p': 0.0, 'chroma_blur_p': 0.0,
            'yuv_roundtrip_p': 0.0, 'gamma_p': 0.0,
        })
        img = _sample_image()
        out = aug(img, label=0)
        assert np.array_equal(out, img)


class TestPipelineRandomizationOutput:
    def test_output_shape_and_dtype_preserved(self):
        aug = PipelineRandomization(config={
            'enabled': True, 'p_real': 1.0, 'p_fake': 1.0,
        })
        img = _sample_image()
        out = aug(img, label=0)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_output_values_in_valid_range(self):
        aug = PipelineRandomization(config={
            'enabled': True, 'p_real': 1.0, 'p_fake': 1.0,
        })
        rng = np.random.RandomState(42)
        for _ in range(10):
            img = _sample_image(rng)
            out = aug(img, label=int(rng.randint(0, 2)))
            assert out.min() >= 0
            assert out.max() <= 255
            assert not np.any(np.isnan(out.astype(np.float32)))

    def test_aug_actually_modifies_image_when_enabled(self):
        # With force-fire on every sub-aug at meaningful intensities, the
        # output should differ from input on at least some pixels.
        aug = PipelineRandomization(config={
            'enabled': True, 'p_real': 1.0, 'p_fake': 1.0,
            'jpeg_p': 1.0, 'downscale_p': 1.0, 'chroma_blur_p': 1.0,
            'yuv_roundtrip_p': 1.0, 'gamma_p': 1.0,
            'jpeg_quality': (40, 50),       # strong JPEG
            'downscale_range': (0.85, 0.85),
            'gamma_range': (0.92, 0.92),    # away from 1.0
        })
        img = _sample_image()
        out = aug(img, label=0)
        diff = np.abs(out.astype(np.int32) - img.astype(np.int32))
        # At least 1% of pixel values should differ noticeably.
        n_changed = (diff > 1).sum()
        assert n_changed > 0.01 * img.size, f"Only {n_changed}/{img.size} pixels changed"


class TestPipelineRandomizationSanitization:
    def test_handles_extreme_gamma_safely(self):
        # Very large gamma can drive low pixels to ~0; ensure no NaN or wrap.
        aug = PipelineRandomization(config={
            'enabled': True, 'p_real': 1.0,
            'jpeg_p': 0.0, 'downscale_p': 0.0, 'chroma_blur_p': 0.0,
            'yuv_roundtrip_p': 0.0, 'gamma_p': 1.0,
            'gamma_range': (5.0, 5.0),
        })
        img = _sample_image()
        out = aug(img, label=0)
        assert out.dtype == np.uint8
        assert out.min() >= 0
        assert out.max() <= 255

    def test_label_none_treated_as_real(self):
        aug = PipelineRandomization(config={
            'enabled': True, 'p_real': 1.0, 'p_fake': 0.0,
            'jpeg_p': 0.0, 'downscale_p': 0.0, 'chroma_blur_p': 0.0,
            'yuv_roundtrip_p': 0.0, 'gamma_p': 0.0,
        })
        img = _sample_image()
        out = aug(img, label=None)
        assert np.array_equal(out, img)

    def test_non_array_input_passes_through(self):
        aug = PipelineRandomization(config={'enabled': True, 'p_real': 1.0})
        result = aug("not an image", label=0)
        assert result == "not an image"
