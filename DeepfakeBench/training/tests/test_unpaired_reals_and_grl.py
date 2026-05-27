"""
Round 6 tests: Unpaired real support, quality domain labels, GRL integration.

Tests Part A (UnifiedUnpairedRealSample, family routing, collate),
Part B (vcd_targeted augmentation preset), and
Part C (GradientReversalLayer, QualityDomainHead, quality_domain in data pipeline).
"""

from __future__ import annotations

import importlib.util
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helper: import modules from source tree without full package install
# ---------------------------------------------------------------------------

_TRAINING_ROOT = Path(__file__).resolve().parents[1]

if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))


def _load_module(rel_path: str, module_name: str):
    module_path = _TRAINING_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_pipelines():
    """Import data.augmentations.pipelines via the package machinery so its
    `from .transforms import ...` relative import resolves. spec_from_file_location
    bypasses package context and breaks the relative import.
    """
    import importlib

    return importlib.import_module("data.augmentations.pipelines")


def _load_combined_paired():
    """Load combined_paired.py with proper package context for relative imports."""
    import types

    # Ensure utils.grouping is importable first (combined_paired imports it)
    if "utils" not in sys.modules:
        utils_pkg = types.ModuleType("utils")
        utils_pkg.__path__ = [str(_TRAINING_ROOT / "utils")]
        sys.modules["utils"] = utils_pkg
    if "utils.grouping" not in sys.modules:
        grouping_mod = _load_module("utils/grouping.py", "utils.grouping")
        sys.modules["utils.grouping"] = grouping_mod

    # Create data package stub
    if "data" not in sys.modules:
        data_pkg = types.ModuleType("data")
        data_pkg.__path__ = [str(_TRAINING_ROOT / "data")]
        sys.modules["data"] = data_pkg

    # Load the REAL data/sources/__init__.py so register_data_source etc. are available
    if "data.sources" not in sys.modules:
        init_path = _TRAINING_ROOT / "data" / "sources" / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            "data.sources", init_path,
            submodule_search_locations=[str(_TRAINING_ROOT / "data" / "sources")],
        )
        src_mod = importlib.util.module_from_spec(spec)
        sys.modules["data.sources"] = src_mod
        spec.loader.exec_module(src_mod)

    # Remove stale cached module if present
    sys.modules.pop("data.sources.combined_paired", None)

    # Now load combined_paired as a child of data.sources
    module_path = _TRAINING_ROOT / "data" / "sources" / "combined_paired.py"
    spec = importlib.util.spec_from_file_location(
        "data.sources.combined_paired", module_path,
        submodule_search_locations=[],
    )
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "data.sources"
    sys.modules["data.sources.combined_paired"] = module
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Part A — UnifiedUnpairedRealSample & Family Routing
# =============================================================================


class TestUnifiedUnpairedRealSample:
    """Tests for the UnifiedUnpairedRealSample dataclass."""

    def _import_dataclass(self):
        mod = _load_combined_paired()
        return mod.UnifiedUnpairedRealSample

    def test_construction_defaults(self):
        cls = self._import_dataclass()
        sample = cls(
            identity="external_vcd_abc123",
            source="external",
            method="external_vcd_real",
            gcs_bucket="effort-collected-data",
            frame_paths=["gs://effort-collected-data/real/VCD/abc_640x480_30/0000.png"],
        )
        assert sample.has_landmarks is False
        assert sample.is_unpaired_real is True
        assert sample.original_sample is None
        assert sample.sample_id == ""

    def test_is_unpaired_real_flag(self):
        cls = self._import_dataclass()
        sample = cls(
            identity="ext_id",
            source="external",
            method="external_vcd_real",
            gcs_bucket="bucket",
            frame_paths=[],
        )
        assert getattr(sample, "is_unpaired_real", False) is True


class TestFamilyRoutingForExternalReals:
    """Ensure external_vcd_real routes to external_real family."""

    def test_infer_family_key_for_external(self):
        grouping = _load_module("utils/grouping.py", "grouping_mod")
        # label=0 (real), method contains "external" → should map to external_real
        _group, family = grouping.infer_group_and_family(
            label=0, method="external_vcd_real", source="external"
        )
        assert family == "external_real"

    def test_sample_family_for_unpaired(self):
        """_sample_family_for_sampling should return real family for unpaired samples."""
        mod = _load_combined_paired()
        sample = mod.UnifiedUnpairedRealSample(
            identity="ext_test",
            source="external",
            method="external_vcd_real",
            gcs_bucket="bucket",
            frame_paths=["gs://bucket/img.png"],
        )
        family = mod._sample_family_for_sampling(sample, enhanced_strategy_names=[])
        assert family == "external_real"


# =============================================================================
# Part A — Quality Domain Labels in Data Pipeline
# =============================================================================


class TestQualityDomainMap:
    """Validate the QUALITY_DOMAIN_MAP and helper function."""

    def test_domain_map_values(self):
        mod = _load_combined_paired()
        assert mod.QUALITY_DOMAIN_MAP["df40"] == 0
        assert mod.QUALITY_DOMAIN_MAP["external"] == 1
        assert mod.QUALITY_DOMAIN_MAP["deeplive"] == 2
        assert mod.QUALITY_DOMAIN_MAP["visomaster"] == 2
        assert mod.QUALITY_DOMAIN_MAP["youtube"] == 3

    def test_quality_domain_for_source(self):
        mod = _load_combined_paired()
        assert mod._quality_domain_for_source("df40") == 0
        assert mod._quality_domain_for_source("external") == 1
        assert mod._quality_domain_for_source("deeplive") == 2
        assert mod._quality_domain_for_source("unknown_source") == 0  # default


# =============================================================================
# Part A — Collate function quality_domain support
# =============================================================================


class TestCollateQualityDomain:
    """Test that combined_paired_collate_fn handles quality_domain."""

    def _get_collate(self):
        mod = _load_combined_paired()
        return mod.combined_paired_collate_fn

    def _make_batch(self, n_videos=2, frames_per_video=4):
        """Create a minimal batch of frame dicts."""
        batch = []
        for vid_idx in range(n_videos):
            label = vid_idx % 2  # alternate real / fake
            source = "df40" if vid_idx < n_videos // 2 else "deeplive"
            domain = 0 if source == "df40" else 2
            for frame_idx in range(frames_per_video):
                batch.append({
                    "image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                    "label": label,
                    "identity": f"id_{vid_idx}",
                    "source": source,
                    "method": f"method_{vid_idx}",
                    "method_id": vid_idx,
                    "sample_id": f"sample_{vid_idx}",
                    "frame_idx": frame_idx,
                    "quality_domain": domain,
                })
        return batch

    def test_quality_domain_present_in_output(self):
        collate = self._get_collate()
        batch = self._make_batch(n_videos=4, frames_per_video=2)
        result = collate(batch)
        assert "quality_domain" in result
        assert isinstance(result["quality_domain"], torch.Tensor)
        assert result["quality_domain"].dtype == torch.long
        assert result["quality_domain"].shape[0] == result["label"].shape[0]

    def test_empty_batch_has_quality_domain(self):
        collate = self._get_collate()
        result = collate([])
        assert "quality_domain" in result
        assert result["quality_domain"].shape[0] == 0

    def test_quality_domain_values_correct(self):
        collate = self._get_collate()
        # 2 videos: first is df40 (domain 0), second is deeplive (domain 2)
        batch = self._make_batch(n_videos=2, frames_per_video=2)
        result = collate(batch)
        domains = result["quality_domain"].tolist()
        assert 0 in domains or 2 in domains  # at least one of the expected domains


# =============================================================================
# Part B — VCD-Targeted Augmentation Preset
# =============================================================================


class TestVCDTargetedPreset:
    """Test the vcd_targeted augmentation preset exists and has required keys."""

    def test_vcd_targeted_in_presets(self):
        pytest.importorskip("albumentations")
        mod = _load_pipelines()
        presets = mod._QUALITY_TARGETED_PRESETS
        assert "vcd_targeted" in presets

    def test_vcd_targeted_has_real_noise_keys(self):
        pytest.importorskip("albumentations")
        mod = _load_pipelines()
        preset = mod._QUALITY_TARGETED_PRESETS["vcd_targeted"]
        assert "real_noise_p" in preset
        assert "real_noise_var" in preset
        assert "real_sharpen_p" in preset
        assert preset["real_noise_p"] > 0
        assert preset["real_sharpen_p"] > 0

    def test_vcd_targeted_codec_modest(self):
        """VCD codec sim should stay modest (plan says ≤15%)."""
        pytest.importorskip("albumentations")
        mod = _load_pipelines()
        preset = mod._QUALITY_TARGETED_PRESETS["vcd_targeted"]
        assert preset["webcam_codec_p"] <= 0.15

    def test_create_router_with_vcd_targeted(self):
        pytest.importorskip("albumentations")
        mod = _load_pipelines()
        router = mod.create_quality_targeted_family_router(strength="vcd_targeted")
        assert router is not None


# =============================================================================
# Part C — Gradient Reversal Layer
# =============================================================================


class TestGradientReversalLayer:
    """Test the GradientReversalFunction and GradientReversalLayer."""

    def _get_classes(self):
        mod = _load_module("detectors/effort_detector.py", "effort_mod")
        return mod.GradientReversalLayer, mod.GradientReversalFunction

    def test_forward_passes_through(self):
        GRL, _ = self._get_classes()
        grl = GRL(lambda_val=1.0)
        x = torch.randn(4, 512)
        y = grl(x)
        assert torch.allclose(x, y)

    def test_backward_reverses_gradient(self):
        GRL, _ = self._get_classes()
        grl = GRL(lambda_val=1.0)
        x = torch.randn(4, 512, requires_grad=True)
        y = grl(x)
        loss = y.sum()
        loss.backward()
        # Gradient should be -1 * ones (reversed)
        assert x.grad is not None
        expected = -torch.ones_like(x)
        assert torch.allclose(x.grad, expected)

    def test_set_lambda(self):
        GRL, _ = self._get_classes()
        grl = GRL(lambda_val=0.5)
        x = torch.randn(2, 64, requires_grad=True)
        y = grl(x)
        y.sum().backward()
        expected = -0.5 * torch.ones_like(x)
        assert torch.allclose(x.grad, expected)

        grl.set_lambda(2.0)
        x2 = torch.randn(2, 64, requires_grad=True)
        y2 = grl(x2)
        y2.sum().backward()
        expected2 = -2.0 * torch.ones_like(x2)
        assert torch.allclose(x2.grad, expected2)


# =============================================================================
# Part C — Quality Domain Head
# =============================================================================


class TestQualityDomainHead:
    """Test the QualityDomainHead module."""

    def _get_class(self):
        mod = _load_module("detectors/effort_detector.py", "effort_mod")
        return mod.QualityDomainHead

    def test_forward_shape(self):
        QDH = self._get_class()
        head = QDH(in_features=512, num_domains=4, hidden_dim=128)
        x = torch.randn(8, 512)
        logits = head(x)
        assert logits.shape == (8, 4)

    def test_domain_map_consistency(self):
        """DOMAIN_MAP in detector should match QUALITY_DOMAIN_MAP in data pipeline."""
        detector_mod = _load_module("detectors/effort_detector.py", "effort_mod")
        data_mod = _load_combined_paired()
        detector_map = detector_mod.QualityDomainHead.DOMAIN_MAP
        data_map = data_mod.QUALITY_DOMAIN_MAP
        for key in detector_map:
            assert key in data_map, f"Key {key} in detector DOMAIN_MAP but not in data QUALITY_DOMAIN_MAP"
            assert detector_map[key] == data_map[key], \
                f"Mismatch for {key}: detector={detector_map[key]}, data={data_map[key]}"

    def test_set_lambda_propagates(self):
        QDH = self._get_class()
        head = QDH(in_features=256, num_domains=3)
        head.set_lambda(0.75)
        assert head.grl.lambda_val == 0.75


# =============================================================================
# Part C — EffortDetector quality head integration
# =============================================================================


class TestEffortDetectorQualityHead:
    """Test that EffortDetector initializes quality head when configured."""

    def _make_config(self, use_head=True):
        """Minimal config dict for EffortDetector init (won't load real weights)."""
        return {
            "backbone": {
                "source": "laion",
                "variant": "ViT-B-16-DataComp-XL",
                "model_name": "ViT-B-16",
                "pretrained": "datacomp_xl_s13b_b90k",
                "hidden_size": 512,
                "resolution": 224,
                "apply_svd_to_in_proj": True,
            },
            "rank": 736,
            "lambda_reg": 0.01,
            "use_arcface_head": True,
            "arcface_s": 10.0,
            "arcface_m": 0.0,
            "use_quality_domain_head": use_head,
            "quality_domain_count": 4,
            "quality_head_hidden_dim": 128,
            "quality_domain_loss_weight": 0.1,
        }

    @pytest.mark.skipif(
        not importlib.util.find_spec("open_clip"),
        reason="open_clip not installed"
    )
    def test_quality_head_created_when_enabled(self):
        mod = _load_module("detectors/effort_detector.py", "effort_mod")
        config = self._make_config(use_head=True)
        detector = mod.EffortDetector(config)
        assert hasattr(detector, "quality_head")
        assert detector.use_quality_head is True

    @pytest.mark.skipif(
        not importlib.util.find_spec("open_clip"),
        reason="open_clip not installed"
    )
    def test_quality_head_not_created_when_disabled(self):
        mod = _load_module("detectors/effort_detector.py", "effort_mod")
        config = self._make_config(use_head=False)
        detector = mod.EffortDetector(config)
        assert detector.use_quality_head is False
        assert not hasattr(detector, "quality_head")


# =============================================================================
# Experiment Config Validation
# =============================================================================


class TestR6ExperimentConfigs:
    """Validate that all R6 experiment YAML configs exist and are well-formed."""

    EXPECTED_CONFIGS = [
        "R6_S1_vcd_reals_aug.yaml",
        "R6_S1_vcd_reals_aug_seed1337.yaml",
        "R6_S2_aug_only.yaml",
        "R6_S2_aug_only_seed1337.yaml",
        "R6_S3_vcd_reals_aug_grl.yaml",
        "R6_S3_vcd_reals_aug_grl_seed1337.yaml",
        "R6_S4_grl_only.yaml",
        "R6_SMOKE_30MIN.yaml",
    ]

    @pytest.fixture
    def config_dir(self):
        return _TRAINING_ROOT / "experiments" / "phase2_round6"

    def test_all_configs_exist(self, config_dir):
        for name in self.EXPECTED_CONFIGS:
            path = config_dir / name
            assert path.exists(), f"Missing experiment config: {name}"

    def test_configs_are_valid_yaml(self, config_dir):
        yaml = pytest.importorskip("yaml")
        for name in self.EXPECTED_CONFIGS:
            path = config_dir / name
            with open(path) as f:
                data = yaml.safe_load(f)
            assert isinstance(data, dict), f"{name} did not parse to a dict"
            assert "name" in data, f"{name} missing 'name' field"
            assert "backbone" in data, f"{name} missing 'backbone' field"

    def test_s3_configs_have_quality_head(self, config_dir):
        yaml = pytest.importorskip("yaml")
        grl_configs = [
            "R6_S3_vcd_reals_aug_grl.yaml",
            "R6_S3_vcd_reals_aug_grl_seed1337.yaml",
            "R6_S4_grl_only.yaml",
            "R6_SMOKE_30MIN.yaml",
        ]
        for name in grl_configs:
            with open(config_dir / name) as f:
                data = yaml.safe_load(f)
            assert data.get("use_quality_domain_head") is True, \
                f"{name} should have use_quality_domain_head: true"

    def test_s1_s2_no_quality_head(self, config_dir):
        yaml = pytest.importorskip("yaml")
        no_grl_configs = [
            "R6_S1_vcd_reals_aug.yaml",
            "R6_S1_vcd_reals_aug_seed1337.yaml",
            "R6_S2_aug_only.yaml",
            "R6_S2_aug_only_seed1337.yaml",
        ]
        for name in no_grl_configs:
            with open(config_dir / name) as f:
                data = yaml.safe_load(f)
            assert not data.get("use_quality_domain_head", False), \
                f"{name} should NOT have use_quality_domain_head"

    def test_vcd_real_configs_have_external_training_reals(self, config_dir):
        yaml = pytest.importorskip("yaml")
        vcd_configs = [
            "R6_S1_vcd_reals_aug.yaml",
            "R6_S1_vcd_reals_aug_seed1337.yaml",
            "R6_S3_vcd_reals_aug_grl.yaml",
            "R6_S3_vcd_reals_aug_grl_seed1337.yaml",
            "R6_SMOKE_30MIN.yaml",
        ]
        for name in vcd_configs:
            with open(config_dir / name) as f:
                data = yaml.safe_load(f)
            cp = data.get("combined_paired", {})
            assert "external_training_reals" in cp, \
                f"{name} should have external_training_reals"

    def test_ablation_configs_no_external_training_reals(self, config_dir):
        yaml = pytest.importorskip("yaml")
        no_vcd_configs = [
            "R6_S2_aug_only.yaml",
            "R6_S2_aug_only_seed1337.yaml",
            "R6_S4_grl_only.yaml",
        ]
        for name in no_vcd_configs:
            with open(config_dir / name) as f:
                data = yaml.safe_load(f)
            cp = data.get("combined_paired", {})
            assert "external_training_reals" not in cp, \
                f"{name} should NOT have external_training_reals"

    def test_all_configs_use_vcd_targeted_augmentation(self, config_dir):
        yaml = pytest.importorskip("yaml")
        for name in self.EXPECTED_CONFIGS:
            with open(config_dir / name) as f:
                data = yaml.safe_load(f)
            aug = data.get("augmentation", {})
            assert aug.get("strength") == "vcd_targeted", \
                f"{name} should use augmentation strength 'vcd_targeted'"

    def test_smoke_has_reduced_steps(self, config_dir):
        yaml = pytest.importorskip("yaml")
        with open(config_dir / "R6_SMOKE_30MIN.yaml") as f:
            data = yaml.safe_load(f)
        assert data["total_training_steps"] <= 2000
        assert data["nEpochs"] <= 5

    def test_smoke_has_fast_startup_settings(self, config_dir):
        yaml = pytest.importorskip("yaml")
        with open(config_dir / "R6_SMOKE_30MIN.yaml") as f:
            data = yaml.safe_load(f)

        cp = data.get("combined_paired", {})
        deeplive_cfg = cp.get("deeplive", {})
        visomaster_cfg = cp.get("visomaster", {})
        ood_cfg = cp.get("ood_monitoring", {})

        assert deeplive_cfg.get("cache_manifest_uri"), \
            "R6_SMOKE_30MIN should enable DeepLive discovery caching"
        assert int(deeplive_cfg.get("max_samples_per_strategy", 0)) > 0, \
            "R6_SMOKE_30MIN should cap DeepLive per-strategy discovery for faster startup"
        assert visomaster_cfg.get("cache_manifest_uri"), \
            "R6_SMOKE_30MIN should enable VisoMaster discovery caching"
        assert ood_cfg.get("build_loader_at_startup") is False, \
            "R6_SMOKE_30MIN should skip OOD loader build at startup"
        assert data.get("ood_monitoring_enabled") is False, \
            "R6_SMOKE_30MIN should disable runtime OOD monitoring for smoke speed"

    def test_seed_1337_variants(self, config_dir):
        yaml = pytest.importorskip("yaml")
        seed_1337_configs = [
            "R6_S1_vcd_reals_aug_seed1337.yaml",
            "R6_S2_aug_only_seed1337.yaml",
            "R6_S3_vcd_reals_aug_grl_seed1337.yaml",
        ]
        for name in seed_1337_configs:
            with open(config_dir / name) as f:
                data = yaml.safe_load(f)
            assert data["seed"] == 1337, f"{name} top-level seed should be 1337"
            # split_seed should remain 737 for deterministic identity split
            cp = data.get("combined_paired", {})
            assert cp.get("split_seed") == 737, \
                f"{name} split_seed should be 737"
