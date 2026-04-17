from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("albumentations")
import numpy as np


_TRAINING_DIR = Path(__file__).resolve().parent.parent
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))


def _ensure_augmentations_package() -> None:
    if "data" not in sys.modules:
        data_pkg = types.ModuleType("data")
        data_pkg.__path__ = [str(_TRAINING_DIR / "data")]
        sys.modules["data"] = data_pkg
    if "data.augmentations" not in sys.modules:
        aug_pkg = types.ModuleType("data.augmentations")
        aug_pkg.__path__ = [str(_TRAINING_DIR / "data" / "augmentations")]
        sys.modules["data.augmentations"] = aug_pkg


def _ensure_utils_grouping_module() -> None:
    if "utils" not in sys.modules:
        utils_pkg = types.ModuleType("utils")
        utils_pkg.__path__ = [str(_TRAINING_DIR / "utils")]
        sys.modules["utils"] = utils_pkg
    if "utils.grouping" not in sys.modules:
        _load_module("utils.grouping", _TRAINING_DIR / "utils" / "grouping.py")


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_teams_module():
    _ensure_augmentations_package()
    return _load_module(
        "data.augmentations.teams_simulation",
        _TRAINING_DIR / "data" / "augmentations" / "teams_simulation.py",
    )


def _load_pipelines_module():
    _ensure_augmentations_package()
    _ensure_utils_grouping_module()
    _load_module(
        "data.augmentations.transforms",
        _TRAINING_DIR / "data" / "augmentations" / "transforms.py",
    )
    _load_teams_module()
    return _load_module(
        "data.augmentations.pipelines",
        _TRAINING_DIR / "data" / "augmentations" / "pipelines.py",
    )


def _make_detail_image(size: int = 224) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            img[y, x, 0] = (x * 3 + y) % 255
            img[y, x, 1] = (x * 5 + y * 2) % 255
            img[y, x, 2] = (x * 7 + y * 3) % 255
    cv2.circle(img, (size // 3, size // 3), size // 8, (30, 20, 10), -1)
    cv2.circle(img, (2 * size // 3, size // 3), size // 8, (30, 20, 10), -1)
    cv2.rectangle(img, (size // 4, 3 * size // 5), (3 * size // 4, 3 * size // 5 + 8), (220, 90, 90), -1)
    return img


def _sharpness(img_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))


def _transform_names(compose) -> list[str]:
    return [transform.__class__.__name__ for transform in compose.transforms]


def test_adaptive_transform_preserves_shape_and_dtype():
    teams_module = _load_teams_module()
    transform = teams_module.TeamsAdaptiveCodecSimulation(always_apply=True, p=1.0)
    image = _make_detail_image()

    result = transform(image=image)["image"]

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_adaptive_transform_family_bias_changes_sharpness_direction():
    teams_module = _load_teams_module()
    transform = teams_module.TeamsAdaptiveCodecSimulation(
        ordinary_mode_probability_non_enhanced=1.0,
        ordinary_mode_probability_enhanced=0.0,
        always_apply=True,
        p=1.0,
    )
    image = _make_detail_image()

    ordinary = transform.apply_for_family(image, family_key="realpool_real")
    enhanced = transform.apply_for_family(image, family_key="visomaster_enhanced_fake")

    assert _sharpness(ordinary) > _sharpness(enhanced)


def test_router_supports_adaptive_teams_policy():
    pipelines = _load_pipelines_module()
    router = pipelines.create_quality_targeted_family_router(
        strength="light",
        teams_codec_simulation={
            "enabled": True,
            "probability": 1.0,
            "policy": "adaptive_mixture",
        },
    )
    image = _make_detail_image(size=96)

    out = router(
        image=image,
        landmarks=None,
        meta={"label": 1, "source": "visomaster_enhanced", "method": "visomaster_enhanced_gfpgan"},
    )

    assert router._teams_sim.__class__.__name__ == "TeamsAdaptiveCodecSimulation"
    assert isinstance(out, np.ndarray)
    assert out.shape == image.shape


def test_router_keeps_legacy_policy_as_default():
    pipelines = _load_pipelines_module()
    router = pipelines.create_quality_targeted_family_router(
        strength="light",
        teams_codec_simulation={
            "enabled": True,
            "probability": 1.0,
        },
    )

    assert router._teams_sim.__class__.__name__ == "TeamsCodecSimulation"


def test_router_supports_family_split_policy():
    pipelines = _load_pipelines_module()
    router = pipelines.create_quality_targeted_family_router(
        strength="light",
        teams_codec_simulation={
            "enabled": True,
            "probability": 1.0,
            "policy": "family_split",
        },
    )

    assert router._teams_sim.__class__.__name__ == "TeamsHybridCodecSimulation"


def test_default_teams_passthrough_pipeline_is_minimal():
    pipelines = _load_pipelines_module()

    assert _transform_names(pipelines._build_teams_passthrough_pipeline()) == [
        "HorizontalFlip",
        "RandomBrightnessContrast",
    ]


def test_teams_passthrough_special_aug_requires_explicit_enable():
    pipelines = _load_pipelines_module()
    pipeline = pipelines._build_teams_passthrough_pipeline(
        {
            "teams_passthrough_special_shadow_p": 0.15,
            "teams_passthrough_special_gamma_up_p": 0.15,
        }
    )

    assert _transform_names(pipeline) == [
        "HorizontalFlip",
        "RandomBrightnessContrast",
    ]


def test_teams_passthrough_special_aug_can_enable_multiple_optional_transforms():
    pipelines = _load_pipelines_module()
    pipeline = pipelines._build_teams_passthrough_pipeline(
        {
            "teams_passthrough_special_aug_enabled": True,
            "teams_passthrough_special_shadow_p": 0.15,
            "teams_passthrough_special_gamma_up_p": 0.15,
        }
    )

    assert _transform_names(pipeline) == [
        "HorizontalFlip",
        "RandomBrightnessContrast",
        "DirectionalShadow",
        "GammaUp",
    ]


def test_gammaup_sidecar_override_does_not_mutate_teams_passthrough_branch():
    pipelines = _load_pipelines_module()
    router = pipelines.create_quality_targeted_family_router(
        strength="vcd_targeted",
        preset_overrides={
            "context_variation_gamma_up_p": 0.15,
            "context_variation_gamma_up_range": (0.45, 0.85),
        },
    )

    non_teams_names = _transform_names(router._pipelines["visomaster_enhanced_fake"])
    teams_names = _transform_names(router._pipelines["deeplive_teams_real"])

    assert "GammaUp" in non_teams_names
    assert teams_names == ["HorizontalFlip", "RandomBrightnessContrast"]


def test_teams_special_shadow_override_does_not_mutate_non_teams_branch():
    pipelines = _load_pipelines_module()
    router = pipelines.create_quality_targeted_family_router(
        strength="vcd_targeted",
        preset_overrides={
            "teams_passthrough_special_aug_enabled": True,
            "teams_passthrough_special_shadow_p": 0.15,
        },
    )

    non_teams_names = _transform_names(router._pipelines["visomaster_enhanced_fake"])
    teams_names = _transform_names(router._pipelines["deeplive_teams_real"])

    assert "DirectionalShadow" not in non_teams_names
    assert teams_names == [
        "HorizontalFlip",
        "RandomBrightnessContrast",
        "DirectionalShadow",
    ]
