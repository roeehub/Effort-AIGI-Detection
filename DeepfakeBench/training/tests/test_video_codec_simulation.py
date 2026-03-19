"""
Tests for VideoCodecSimulation transform and webcam_codec pipeline.

Validates:
1. Transform produces correct output shape and dtype
2. Different severity levels produce proportional degradation
3. Integration with A.Compose works correctly
4. Quality metrics shift in the expected direction (flatter PSD, more high-freq)
5. Registry lookup works for 'webcam_codec' and 'codec' keys
6. Integration with quality_targeted_family router works
"""

import numpy as np
import cv2
import pytest


def _make_face_image(size=224, seed=42):
    """Create a synthetic face-like image with gradients and edges."""
    rng = np.random.RandomState(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Base gradient (simulates skin tone variation)
    for c in range(3):
        base = rng.randint(80, 180)
        grad = np.linspace(base - 30, base + 30, size).reshape(1, -1)
        img[:, :, c] = np.clip(grad, 0, 255).astype(np.uint8)
    # Add some "facial features" (circles, lines)
    cv2.circle(img, (80, 90), 15, (60, 40, 30), -1)    # left eye
    cv2.circle(img, (144, 90), 15, (60, 40, 30), -1)    # right eye
    cv2.ellipse(img, (112, 150), (30, 15), 0, 0, 180, (120, 60, 60), 2)  # mouth
    # Add texture noise
    noise = rng.normal(0, 8, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def _compute_psd_slope(img_gray):
    """Compute PSD slope for a grayscale image (simplified)."""
    from scipy import stats
    gray_224 = cv2.resize(img_gray, (224, 224))
    f_transform = np.fft.fft2(gray_224.astype(np.float64))
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    
    cy, cx = 112, 112
    max_r = int(np.sqrt(2) * 112)
    radial_profile = np.zeros(max_r)
    radial_count = np.zeros(max_r)
    Y, X = np.ogrid[:224, :224]
    r_int = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    for ry in range(224):
        for rx in range(224):
            r = r_int[ry, rx]
            if r < max_r:
                radial_profile[r] += magnitude[ry, rx]
                radial_count[r] += 1
    radial_count[radial_count == 0] = 1
    radial_profile /= radial_count
    
    valid = (radial_profile[5:100] > 0)
    if valid.sum() > 10:
        freqs = np.arange(5, 100)[valid]
        power = radial_profile[5:100][valid]
        slope, _, _, _, _ = stats.linregress(np.log10(freqs), np.log10(power))
        return slope
    return 0.0


def _freq_high_ratio(img_gray):
    """Compute high-frequency energy ratio."""
    gray_224 = cv2.resize(img_gray, (224, 224))
    f_transform = np.fft.fft2(gray_224.astype(np.float64))
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    
    Y, X = np.ogrid[:224, :224]
    radius = np.sqrt((X - 112) ** 2 + (Y - 112) ** 2)
    total = magnitude.sum()
    if total == 0:
        return 0.0
    return float(magnitude[radius >= 60].sum() / total)


class TestVideoCodecSimulation:
    """Test the VideoCodecSimulation transform."""

    def test_import(self):
        pytest.importorskip("albumentations")
        from data.augmentations.transforms import VideoCodecSimulation
        t = VideoCodecSimulation(p=1.0)
        assert t is not None

    def test_output_shape_and_dtype(self):
        pytest.importorskip("albumentations")
        from data.augmentations.transforms import VideoCodecSimulation
        img = _make_face_image()
        t = VideoCodecSimulation(codec_quality=(30, 70), p=1.0)
        result = t(image=img)['image']
        assert result.shape == img.shape, f"Shape mismatch: {result.shape} vs {img.shape}"
        assert result.dtype == np.uint8, f"Dtype mismatch: {result.dtype}"

    def test_p_zero_no_change(self):
        pytest.importorskip("albumentations")
        from data.augmentations.transforms import VideoCodecSimulation
        img = _make_face_image()
        t = VideoCodecSimulation(codec_quality=(30, 70), p=0.0)
        result = t(image=img)['image']
        np.testing.assert_array_equal(result, img)

    def test_heavy_degradation_changes_image(self):
        pytest.importorskip("albumentations")
        from data.augmentations.transforms import VideoCodecSimulation
        img = _make_face_image()
        t = VideoCodecSimulation(codec_quality=(10, 30), p=1.0)
        result = t(image=img)['image']
        diff_pct = (img != result).mean() * 100
        assert diff_pct > 50, f"Heavy codec sim should change >50% pixels, got {diff_pct:.1f}%"

    def test_severity_proportional(self):
        """Heavier codec quality should produce more change."""
        pytest.importorskip("albumentations")
        from data.augmentations.transforms import VideoCodecSimulation
        img = _make_face_image()
        
        diffs = []
        for q_min, q_max in [(70, 90), (40, 60), (10, 30)]:
            total_diff = 0
            n_trials = 10
            for _ in range(n_trials):
                t = VideoCodecSimulation(codec_quality=(q_min, q_max), p=1.0)
                result = t(image=img)['image']
                total_diff += np.abs(img.astype(float) - result.astype(float)).mean()
            diffs.append(total_diff / n_trials)
        
        # Heavy (low quality) should produce more change than light (high quality)
        assert diffs[2] > diffs[0], (
            f"Heavy degradation ({diffs[2]:.1f}) should exceed light ({diffs[0]:.1f})"
        )

    def test_frequency_shift_direction(self):
        """Codec sim should flatten PSD slope (make it less negative) and increase high-freq ratio."""
        pytest.importorskip("albumentations")
        from data.augmentations.transforms import VideoCodecSimulation
        img = _make_face_image()
        gray_orig = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        slope_orig = _compute_psd_slope(gray_orig)
        hf_orig = _freq_high_ratio(gray_orig)
        
        # Apply heavy codec simulation multiple times and average
        slopes = []
        hfs = []
        for _ in range(15):
            t = VideoCodecSimulation(codec_quality=(15, 35), p=1.0)
            result = t(image=img)['image']
            gray_result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
            slopes.append(_compute_psd_slope(gray_result))
            hfs.append(_freq_high_ratio(gray_result))
        
        avg_slope = np.mean(slopes)
        avg_hf = np.mean(hfs)
        
        # PSD slope should be flatter (less negative = higher value)
        assert avg_slope > slope_orig, (
            f"Codec sim should flatten PSD slope: orig={slope_orig:.3f}, after={avg_slope:.3f}"
        )
        # High-frequency ratio should increase (codec noise adds high-freq energy)
        assert avg_hf > hf_orig, (
            f"Codec sim should increase high-freq ratio: orig={hf_orig:.4f}, after={avg_hf:.4f}"
        )

    def test_compose_compatibility(self):
        """Works inside A.Compose with other transforms."""
        A = pytest.importorskip("albumentations")
        from data.augmentations.transforms import VideoCodecSimulation
        img = _make_face_image()
        pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            VideoCodecSimulation(codec_quality=(30, 70), p=1.0),
            A.RandomBrightnessContrast(p=0.5),
        ])
        result = pipeline(image=img)['image']
        assert result.shape == img.shape
        assert result.dtype == np.uint8


class TestWebcamCodecPipeline:
    """Test the standalone webcam_codec pipeline registration."""

    def test_registry_lookup(self):
        pytest.importorskip("albumentations")
        from data.augmentations import get_pipeline
        pipeline = get_pipeline(version='webcam_codec')
        assert pipeline is not None

    def test_codec_alias(self):
        pytest.importorskip("albumentations")
        from data.augmentations import get_pipeline
        pipeline = get_pipeline(version='codec')
        assert pipeline is not None

    def test_pipeline_runs(self):
        pytest.importorskip("albumentations")
        from data.augmentations import get_pipeline
        img = _make_face_image()
        pipeline = get_pipeline(version='webcam_codec')
        result = pipeline(image=img)['image']
        assert result.shape == img.shape
        assert result.dtype == np.uint8


class TestQualityTargetedFamilyIntegration:
    """Test that VideoCodecSimulation is integrated into family router."""

    def test_family_pipelines_include_codec(self):
        """Each family pipeline should include VideoCodecSimulation as a step."""
        pytest.importorskip("albumentations")
        from data.augmentations.pipelines import _build_family_quality_pipeline, _QUALITY_TARGETED_PRESETS
        from data.augmentations.transforms import VideoCodecSimulation
        
        p = _QUALITY_TARGETED_PRESETS["moderate"]
        families = [
            "df40_fake", "deeplive_non_enhanced_fake", "deeplive_enhanced_fake",
            "visomaster_fake", "df40_real", "realpool_real", "external_real",
        ]
        for family in families:
            pipeline = _build_family_quality_pipeline(family, p)
            # Check that VideoCodecSimulation is somewhere in the pipeline transforms
            has_codec = any(
                isinstance(t, VideoCodecSimulation)
                for t in pipeline.transforms
            )
            assert has_codec, f"Family '{family}' pipeline missing VideoCodecSimulation step"

    def test_router_applies_codec_sometimes(self):
        """With enough runs, VideoCodecSimulation should fire for some images."""
        pytest.importorskip("albumentations")
        from data.augmentations.pipelines import create_quality_targeted_family_router
        
        router = create_quality_targeted_family_router(strength="strong")
        img = _make_face_image()
        
        # Run many times — with p=0.22 (strong preset), ~22% should get codec sim
        n_runs = 100
        changes = []
        for _ in range(n_runs):
            result = router(img, meta={"label": 0, "source": "df40", "method": None})
            diff = np.abs(img.astype(float) - result.astype(float)).mean()
            changes.append(diff)
        
        # Just verify it doesn't crash and produces some variation
        assert len(set(f"{c:.1f}" for c in changes)) > 1, "Router should produce variable output"

    def test_preset_webcam_codec_p_values(self):
        """Verify all presets have webcam_codec_p configured."""
        pytest.importorskip("albumentations")
        from data.augmentations.pipelines import _QUALITY_TARGETED_PRESETS
        
        for strength, preset in _QUALITY_TARGETED_PRESETS.items():
            assert "webcam_codec_p" in preset, f"Preset '{strength}' missing webcam_codec_p"
            assert "webcam_codec_quality" in preset, f"Preset '{strength}' missing webcam_codec_quality"
            assert 0.0 < preset["webcam_codec_p"] <= 1.0, f"webcam_codec_p should be in (0, 1]"
            q = preset["webcam_codec_quality"]
            assert len(q) == 2 and q[0] < q[1], f"webcam_codec_quality should be (min, max) tuple"


class TestExportAndImport:
    """Test that VideoCodecSimulation is properly exported."""

    def test_import_from_augmentations(self):
        pytest.importorskip("albumentations")
        from data.augmentations import VideoCodecSimulation
        assert VideoCodecSimulation is not None

    def test_in_all(self):
        pytest.importorskip("albumentations")
        import data.augmentations as aug
        assert 'VideoCodecSimulation' in aug.__all__
