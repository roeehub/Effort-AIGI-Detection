"""
Tests for lighting robustness transforms (DirectionalShadow, GammaUp).

Also covers ColorTemperatureShift for completeness since it was untested.
All transforms must be compatible with albumentations==0.4.6.
"""

import sys
import os
import importlib
import numpy as np
import pytest

# Ensure training/ is on sys.path for imports
_TRAINING_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _TRAINING_DIR)

# We must bypass data/__init__.py because it chains into torch-dependent
# batching code.  Import the leaf module directly via importlib.
_spec = importlib.util.spec_from_file_location(
    "data.augmentations.transforms",
    os.path.join(_TRAINING_DIR, "data", "augmentations", "transforms.py"),
)
_transforms_mod = importlib.util.module_from_spec(_spec)
sys.modules["data.augmentations.transforms"] = _transforms_mod
_spec.loader.exec_module(_transforms_mod)

ColorTemperatureShift = _transforms_mod.ColorTemperatureShift
DirectionalShadow = _transforms_mod.DirectionalShadow
GammaUp = _transforms_mod.GammaUp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image():
    """A 224×224 RGB uint8 image with mid-range brightness (~128)."""
    rng = np.random.RandomState(42)
    return rng.randint(40, 200, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def dark_image():
    """A uniformly dark image (mean brightness ~30)."""
    rng = np.random.RandomState(7)
    return rng.randint(10, 50, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def bright_image():
    """A uniformly bright image (mean brightness ~220)."""
    rng = np.random.RandomState(12)
    return rng.randint(200, 255, (224, 224, 3), dtype=np.uint8)


# ===================================================================
# DirectionalShadow
# ===================================================================

class TestDirectionalShadow:

    def test_output_shape_and_dtype(self, sample_image):
        t = DirectionalShadow(always_apply=True)
        out = t.apply(sample_image)
        assert out.shape == sample_image.shape
        assert out.dtype == np.uint8

    def test_output_clipped_0_255(self, sample_image):
        t = DirectionalShadow(intensity_range=(0.8, 0.99), always_apply=True)
        out = t.apply(sample_image)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_shadow_darkens_image(self, sample_image):
        """Shadow should never brighten; mean brightness must decrease."""
        t = DirectionalShadow(
            intensity_range=(0.3, 0.3), softness_range=(0.3, 0.3),
            directions=("left",), always_apply=True,
        )
        out = t.apply(sample_image.copy())
        assert out.astype(float).mean() < sample_image.astype(float).mean()

    def test_cardinal_directions(self, sample_image):
        """All 4 cardinal directions should produce valid output."""
        for d in ("left", "right", "top", "bottom"):
            t = DirectionalShadow(
                directions=(d,), intensity_range=(0.3, 0.3),
                softness_range=(0.3, 0.3), always_apply=True,
            )
            out = t.apply(sample_image.copy())
            assert out.shape == sample_image.shape
            assert out.dtype == np.uint8

    def test_diagonal_directions(self, sample_image):
        """All 4 diagonal directions should produce valid output."""
        for d in ("top_left", "top_right", "bottom_left", "bottom_right"):
            t = DirectionalShadow(
                directions=(d,), intensity_range=(0.4, 0.4),
                softness_range=(0.3, 0.3), always_apply=True,
            )
            out = t.apply(sample_image.copy())
            assert out.shape == sample_image.shape

    def test_left_shadow_gradient(self, sample_image):
        """Left shadow: left columns should be darker than right columns."""
        t = DirectionalShadow(
            directions=("left",), intensity_range=(0.5, 0.5),
            softness_range=(0.2, 0.2), always_apply=True,
        )
        out = t.apply(sample_image.copy()).astype(float)
        left_mean = out[:, :56, :].mean()   # left quarter
        right_mean = out[:, 168:, :].mean()  # right quarter
        assert left_mean < right_mean, f"left {left_mean:.1f} should be < right {right_mean:.1f}"

    def test_top_shadow_gradient(self, sample_image):
        """Top shadow: top rows should be darker than bottom rows."""
        t = DirectionalShadow(
            directions=("top",), intensity_range=(0.5, 0.5),
            softness_range=(0.2, 0.2), always_apply=True,
        )
        out = t.apply(sample_image.copy()).astype(float)
        top_mean = out[:56, :, :].mean()
        bot_mean = out[168:, :, :].mean()
        assert top_mean < bot_mean, f"top {top_mean:.1f} should be < bottom {bot_mean:.1f}"

    def test_zero_intensity_is_identity(self, sample_image):
        """Intensity=0 should leave the image unchanged."""
        t = DirectionalShadow(
            intensity_range=(0.0, 0.0), always_apply=True,
        )
        out = t.apply(sample_image.copy())
        np.testing.assert_array_equal(out, sample_image)

    def test_various_sizes(self):
        """Works on non-square and small images."""
        for h, w in [(100, 200), (200, 100), (16, 16), (1, 1)]:
            img = np.full((h, w, 3), 128, dtype=np.uint8)
            t = DirectionalShadow(always_apply=True)
            out = t.apply(img)
            assert out.shape == (h, w, 3)

    def test_albumentations_compose_integration(self, sample_image):
        """Works inside an albumentations Compose pipeline."""
        import albumentations as A
        pipe = A.Compose([DirectionalShadow(p=1.0)])
        result = pipe(image=sample_image)
        assert result["image"].shape == sample_image.shape
        assert result["image"].dtype == np.uint8

    def test_get_transform_init_args_names(self):
        t = DirectionalShadow()
        args = t.get_transform_init_args_names()
        assert "intensity_range" in args
        assert "softness_range" in args
        assert "directions" in args


# ===================================================================
# GammaUp
# ===================================================================

class TestGammaUp:

    def test_output_shape_and_dtype(self, sample_image):
        t = GammaUp(always_apply=True)
        out = t.apply(sample_image)
        assert out.shape == sample_image.shape
        assert out.dtype == np.uint8

    def test_always_brightens(self, sample_image):
        """Mean brightness must increase (gamma < 1 brightens)."""
        t = GammaUp(gamma_range=(0.5, 0.5), always_apply=True)
        out = t.apply(sample_image.copy())
        assert out.astype(float).mean() > sample_image.astype(float).mean()

    def test_brightens_dark_image(self, dark_image):
        """Even a dark image gets pushed brighter."""
        t = GammaUp(gamma_range=(0.45, 0.45), always_apply=True)
        out = t.apply(dark_image.copy())
        assert out.astype(float).mean() > dark_image.astype(float).mean()

    def test_output_clipped(self, bright_image):
        """Output should be clipped to [0, 255] even on bright input."""
        t = GammaUp(gamma_range=(0.45, 0.45), always_apply=True)
        out = t.apply(bright_image.copy())
        assert out.min() >= 0
        assert out.max() <= 255

    def test_stronger_gamma_brighter(self, sample_image):
        """Lower gamma → brighter output."""
        t_strong = GammaUp(gamma_range=(0.45, 0.45), always_apply=True)
        t_mild = GammaUp(gamma_range=(0.85, 0.85), always_apply=True)
        out_strong = t_strong.apply(sample_image.copy())
        out_mild = t_mild.apply(sample_image.copy())
        assert out_strong.astype(float).mean() > out_mild.astype(float).mean()

    def test_black_stays_black(self):
        """Pure black pixels (0) must remain 0 regardless of gamma."""
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        t = GammaUp(gamma_range=(0.45, 0.45), always_apply=True)
        out = t.apply(img)
        np.testing.assert_array_equal(out, img)

    def test_white_stays_white(self):
        """Pure white pixels (255) must remain 255."""
        img = np.full((32, 32, 3), 255, dtype=np.uint8)
        t = GammaUp(gamma_range=(0.45, 0.45), always_apply=True)
        out = t.apply(img)
        np.testing.assert_array_equal(out, img)

    def test_invalid_range_raises(self):
        """gamma_range outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError):
            GammaUp(gamma_range=(0.5, 1.0))  # upper bound = 1.0
        with pytest.raises(ValueError):
            GammaUp(gamma_range=(0.0, 0.8))  # lower bound = 0.0
        with pytest.raises(ValueError):
            GammaUp(gamma_range=(1.1, 1.5))  # both > 1

    def test_albumentations_compose_integration(self, sample_image):
        """Works inside an albumentations Compose pipeline."""
        import albumentations as A
        pipe = A.Compose([GammaUp(p=1.0)])
        result = pipe(image=sample_image)
        assert result["image"].shape == sample_image.shape
        assert result["image"].dtype == np.uint8

    def test_get_transform_init_args_names(self):
        t = GammaUp()
        args = t.get_transform_init_args_names()
        assert "gamma_range" in args

    def test_various_sizes(self):
        """Works on non-square and small images."""
        for h, w in [(100, 200), (200, 100), (16, 16)]:
            img = np.full((h, w, 3), 128, dtype=np.uint8)
            t = GammaUp(always_apply=True)
            out = t.apply(img)
            assert out.shape == (h, w, 3)


# ===================================================================
# ColorTemperatureShift (existing, but previously untested)
# ===================================================================

class TestColorTemperatureShift:

    def test_output_shape_and_dtype(self, sample_image):
        t = ColorTemperatureShift(always_apply=True)
        out = t.apply(sample_image)
        assert out.shape == sample_image.shape
        assert out.dtype == np.uint8

    def test_warm_cct_increases_red_ratio(self, sample_image):
        """Low CCT (warm) should increase R/B ratio."""
        t = ColorTemperatureShift(cct_range=(2700, 2700), always_apply=True)
        out = t.apply(sample_image.copy())
        orig_rb = sample_image[:, :, 0].astype(float).mean() / max(1, sample_image[:, :, 2].astype(float).mean())
        new_rb = out[:, :, 0].astype(float).mean() / max(1, out[:, :, 2].astype(float).mean())
        assert new_rb > orig_rb, f"Warm CCT should increase R/B ratio: {new_rb:.2f} vs {orig_rb:.2f}"

    def test_cool_cct_decreases_red_ratio(self, sample_image):
        """High CCT (cool) should decrease R/B ratio."""
        t = ColorTemperatureShift(cct_range=(8000, 8000), always_apply=True)
        out = t.apply(sample_image.copy())
        orig_rb = sample_image[:, :, 0].astype(float).mean() / max(1, sample_image[:, :, 2].astype(float).mean())
        new_rb = out[:, :, 0].astype(float).mean() / max(1, out[:, :, 2].astype(float).mean())
        assert new_rb < orig_rb, f"Cool CCT should decrease R/B ratio: {new_rb:.2f} vs {orig_rb:.2f}"

    def test_daylight_cct_near_identity(self, sample_image):
        """6500K (daylight reference) should be near-identity."""
        t = ColorTemperatureShift(cct_range=(6500, 6500), always_apply=True)
        out = t.apply(sample_image.copy())
        diff = np.abs(out.astype(float) - sample_image.astype(float)).mean()
        assert diff < 2.0, f"6500K should be near-identity, mean diff was {diff:.1f}"

    def test_output_clipped(self, bright_image):
        t = ColorTemperatureShift(cct_range=(2700, 2700), always_apply=True)
        out = t.apply(bright_image.copy())
        assert out.min() >= 0
        assert out.max() <= 255

    def test_albumentations_compose_integration(self, sample_image):
        import albumentations as A
        pipe = A.Compose([ColorTemperatureShift(p=1.0)])
        result = pipe(image=sample_image)
        assert result["image"].shape == sample_image.shape

    def test_get_transform_init_args_names(self):
        t = ColorTemperatureShift()
        assert "cct_range" in t.get_transform_init_args_names()


# ===================================================================
# Pipeline integration: _build_context_variation_block
# ===================================================================

class TestContextVariationBlock:
    """Verify the new transforms are wired correctly into the pipeline builder."""

    @pytest.fixture(autouse=True)
    def _skip_without_cv2(self):
        """Pipeline builder needs cv2 for ShiftScaleRotate border_mode."""
        pytest.importorskip("cv2")

    def _build(self, p):
        # Import pipelines module directly to avoid data/__init__.py → torch chain.
        _spec = importlib.util.spec_from_file_location(
            "data.augmentations.pipelines",
            os.path.join(_TRAINING_DIR, "data", "augmentations", "pipelines.py"),
        )
        _pipelines_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_pipelines_mod)
        return _pipelines_mod._build_context_variation_block(p)

    def test_disabled_returns_empty(self):
        result = self._build({"context_variation_enabled": False})
        assert result == []

    def test_default_has_no_shadow_or_gamma_up(self):
        """With default preset, shadow and gamma_up are off (p=0.0)."""
        transforms = self._build({"context_variation_enabled": True, "context_variation_individual_p": 0.15})
        transform_types = [type(t).__name__ for t in transforms]
        assert "DirectionalShadow" not in transform_types
        assert "GammaUp" not in transform_types

    def test_shadow_enabled(self):
        """Setting shadow_p > 0 adds DirectionalShadow to the block."""
        transforms = self._build({
            "context_variation_enabled": True,
            "context_variation_individual_p": 0.15,
            "context_variation_shadow_p": 0.10,
        })
        transform_types = [type(t).__name__ for t in transforms]
        assert "DirectionalShadow" in transform_types

    def test_gamma_up_enabled(self):
        """Setting gamma_up_p > 0 adds GammaUp to the block."""
        transforms = self._build({
            "context_variation_enabled": True,
            "context_variation_individual_p": 0.15,
            "context_variation_gamma_up_p": 0.15,
        })
        transform_types = [type(t).__name__ for t in transforms]
        assert "GammaUp" in transform_types

    def test_all_new_transforms_enabled(self):
        """All three optional transforms enabled."""
        transforms = self._build({
            "context_variation_enabled": True,
            "context_variation_individual_p": 0.15,
            "context_variation_cct_p": 0.15,
            "context_variation_shadow_p": 0.10,
            "context_variation_gamma_up_p": 0.10,
        })
        transform_types = [type(t).__name__ for t in transforms]
        assert "ColorTemperatureShift" in transform_types
        assert "DirectionalShadow" in transform_types
        assert "GammaUp" in transform_types
        # Base transforms should still be there
        assert len(transforms) >= 6  # 3 base + CCT + shadow + gamma_up
