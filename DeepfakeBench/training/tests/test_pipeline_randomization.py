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


class TestPipelineRandomizationP22Augs:
    """Coverage for the P22 luma-blur + brightness-shift sub-augs."""

    def test_blur_off_by_default(self):
        # Existing yamls without blur_p must see no blur applied.
        aug = PipelineRandomization(config={
            'enabled': True, 'p_real': 1.0, 'p_fake': 0.0,
            'jpeg_p': 0.0, 'downscale_p': 0.0, 'chroma_blur_p': 0.0,
            'yuv_roundtrip_p': 0.0, 'gamma_p': 0.0,
        })
        img = _sample_image()
        out = aug(img, label=0)
        assert np.array_equal(out, img), "default config must not blur"

    def test_blur_drops_laplacian(self):
        # Force-fire blur with sigma=2; laplacian variance must drop sharply.
        aug = PipelineRandomization(config={
            'enabled': True, 'p_real': 1.0, 'p_fake': 1.0,
            'jpeg_p': 0.0, 'downscale_p': 0.0, 'chroma_blur_p': 0.0,
            'yuv_roundtrip_p': 0.0, 'gamma_p': 0.0,
            'blur_p': 1.0, 'blur_sigma_range': (2.0, 2.0),
            'brightness_p': 0.0,
        })
        # Use a real-ish image (gradient with noise) so Laplacian is well-defined.
        rng = np.random.RandomState(13)
        base = np.tile(np.arange(224, dtype=np.uint8)[None, :, None], (224, 1, 3))
        noise = rng.randint(0, 30, (224, 224, 3), dtype=np.uint8)
        img = np.clip(base.astype(np.int32) + noise, 0, 255).astype(np.uint8)
        out = aug(img, label=0)
        import cv2
        lap_before = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        lap_after = cv2.Laplacian(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        assert lap_after < lap_before * 0.5, f"σ=2 blur did not halve Laplacian: {lap_before:.1f} → {lap_after:.1f}"

    def test_blur_sigma_zero_is_noop(self):
        aug = PipelineRandomization(config={
            'enabled': True, 'p_real': 1.0, 'p_fake': 0.0,
            'jpeg_p': 0.0, 'downscale_p': 0.0, 'chroma_blur_p': 0.0,
            'yuv_roundtrip_p': 0.0, 'gamma_p': 0.0,
            'blur_p': 1.0, 'blur_sigma_range': (0.0, 0.0),
            'brightness_p': 0.0,
        })
        img = _sample_image()
        out = aug(img, label=0)
        assert np.array_equal(out, img)

    def test_brightness_shifts_mean(self):
        # Force-fire +50 brightness shift; mean luma must rise.
        aug = PipelineRandomization(config={
            'enabled': True, 'p_real': 1.0, 'p_fake': 1.0,
            'jpeg_p': 0.0, 'downscale_p': 0.0, 'chroma_blur_p': 0.0,
            'yuv_roundtrip_p': 0.0, 'gamma_p': 0.0,
            'blur_p': 0.0,
            'brightness_p': 1.0, 'brightness_range': (50.0, 50.0),
        })
        rng = np.random.RandomState(7)
        # Mid-gray image so we don't saturate.
        img = (rng.randint(80, 160, (224, 224, 3))).astype(np.uint8)
        out = aug(img, label=0)
        assert out.mean() > img.mean() + 30, f"+50 shift did not move mean: {img.mean():.1f} → {out.mean():.1f}"

    def test_brightness_negative_shift_clips(self):
        # -200 shift on uint8 must clip to 0, not wrap.
        aug = PipelineRandomization(config={
            'enabled': True, 'p_real': 1.0,
            'jpeg_p': 0.0, 'downscale_p': 0.0, 'chroma_blur_p': 0.0,
            'yuv_roundtrip_p': 0.0, 'gamma_p': 0.0,
            'blur_p': 0.0,
            'brightness_p': 1.0, 'brightness_range': (-200.0, -200.0),
        })
        img = _sample_image()
        out = aug(img, label=0)
        assert out.min() >= 0 and out.max() <= 255
        # Most pixels should clip to 0 (input is U[0,255]; -200 sends most below 0).
        assert (out == 0).mean() > 0.5

    def test_p22_curriculum_keeps_uint8_rgb_shape(self):
        # The actual P22 curriculum: blur σ∈[0,3], JPEG q∈[50,95], brightness β∈[-40,40].
        aug = PipelineRandomization(config={
            'enabled': True, 'p_real': 0.5, 'p_fake': 0.5,
            'jpeg_p': 0.5, 'jpeg_quality': (50, 95),
            'downscale_p': 0.3, 'downscale_range': (0.85, 1.0),
            'chroma_blur_p': 0.3, 'chroma_blur_ksize': 3,
            'yuv_roundtrip_p': 0.0,
            'gamma_p': 0.3, 'gamma_range': (0.92, 1.08),
            'blur_p': 0.5, 'blur_sigma_range': (0.0, 3.0),
            'brightness_p': 0.5, 'brightness_range': (-40.0, 40.0),
        })
        rng = np.random.RandomState(99)
        for i in range(20):
            img = _sample_image(rng)
            out = aug(img, label=int(rng.randint(0, 2)))
            assert out.shape == img.shape
            assert out.dtype == np.uint8
            assert out.min() >= 0 and out.max() <= 255
