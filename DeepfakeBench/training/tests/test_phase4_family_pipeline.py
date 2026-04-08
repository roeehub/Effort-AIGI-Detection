"""Phase 4 tests: family-aware augmentation, data plumbing, and grouped reporting."""

from __future__ import annotations

import logging
import random
import sys
import types
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest


def _load_grouping_module():
    module_path = Path(__file__).resolve().parents[1] / "utils" / "grouping.py"
    spec = importlib.util.spec_from_file_location("grouping_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_group_key_mapping_representative_cases():
    grouping = _load_grouping_module()

    cases = [
        (0, "faceforensics++", None, "df40_real", "df40_real"),
        (1, "blendface", None, "df40_fake", "df40_fake"),
        (0, "deeplive_edge_cases", None, "deeplive_edge_cases_real", "realpool_real"),
        (1, "deeplive_edge_cases_enhanced", None, "deeplive_edge_cases_enhanced_fake", "deeplive_enhanced_fake"),
        (1, "visomaster_CSCS", None, "visomaster_fake", "visomaster_fake"),
        # VisoMaster enhanced (post-hoc face enhancement) — source-based routing
        (1, "visomaster_enhanced_gfpgan", "visomaster_enhanced", "visomaster_enhanced_fake", "visomaster_enhanced_fake"),
        (1, "visomaster_enhanced_codeformer", "visomaster_enhanced", "visomaster_enhanced_fake", "visomaster_enhanced_fake"),
        (0, "visomaster_enhanced_gfpgan", "visomaster_enhanced", "visomaster_enhanced_real", "realpool_real"),
        # Method-only routing (no source) — falls back to method prefix detection
        (1, "visomaster_enhanced_gpen_bfr_512", None, "visomaster_enhanced_fake", "visomaster_enhanced_fake"),
        (0, "external_youtube_avspeech", None, "external_real", "external_real"),
        (1, "wma_failure_fake", None, "wma_failure_fake", "wma_failure_fake"),
    ]

    for label, method, source, expected_group, expected_family in cases:
        group_key, family_key = grouping.infer_group_and_family(label=label, method=method, source=source)
        assert group_key == expected_group
        assert family_key == expected_family


def test_quality_targeted_family_router_forward_pass():
    pytest.importorskip("albumentations")
    from data.augmentations.pipelines import create_quality_targeted_family_router

    router = create_quality_targeted_family_router(strength="light")
    image = np.full((64, 64, 3), 127, dtype=np.uint8)

    out = router(
        image=image,
        landmarks=None,
        meta={"label": 1, "source": "deeplive", "method": "deeplive_edge_cases_enhanced"},
    )

    assert isinstance(out, np.ndarray)
    assert out.shape == image.shape


def _load_pipelines_module():
    """Load pipelines module directly (avoids torch dependency via data/__init__.py)."""
    import cv2  # noqa: F401 — needed by pipelines.py at module level
    module_path = Path(__file__).resolve().parents[1] / "data" / "augmentations" / "pipelines.py"
    transforms_path = Path(__file__).resolve().parents[1] / "data" / "augmentations" / "transforms.py"
    # Load transforms first (pipelines.py imports from .transforms)
    transforms_spec = importlib.util.spec_from_file_location(
        "data.augmentations.transforms", transforms_path
    )
    transforms_module = importlib.util.module_from_spec(transforms_spec)
    sys.modules["data.augmentations.transforms"] = transforms_module
    transforms_spec.loader.exec_module(transforms_module)
    # Now load pipelines with the transforms available
    spec = importlib.util.spec_from_file_location(
        "data.augmentations.pipelines", module_path,
        submodule_search_locations=[],
    )
    module = importlib.util.module_from_spec(spec)
    # Patch the relative import to use absolute
    sys.modules["data.augmentations.pipelines"] = module
    spec.loader.exec_module(module)
    return module


def _ensure_torch_stub():
    if "torch" in sys.modules:
        return

    torch_module = types.ModuleType("torch")
    torch_utils_module = types.ModuleType("torch.utils")
    torch_utils_data_module = types.ModuleType("torch.utils.data")

    class _StubDataLoader:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _StubIterableDataset:
        pass

    torch_utils_data_module.DataLoader = _StubDataLoader
    torch_utils_data_module.IterableDataset = _StubIterableDataset
    torch_utils_module.data = torch_utils_data_module
    torch_module.utils = torch_utils_module
    torch_module.cuda = types.SimpleNamespace(is_available=lambda: False)

    sys.modules["torch"] = torch_module
    sys.modules["torch.utils"] = torch_utils_module
    sys.modules["torch.utils.data"] = torch_utils_data_module


def _ensure_source_package_stubs():
    data_module = sys.modules.get("data")
    if data_module is None:
        data_module = types.ModuleType("data")
        data_module.__path__ = []
        sys.modules["data"] = data_module

    sources_module = sys.modules.get("data.sources")
    if sources_module is None:
        sources_module = types.ModuleType("data.sources")
        sources_module.__path__ = []

        @dataclass
        class _StubDataPipelineResult:
            train_loader: object
            val_in_dist_loader: object
            val_holdout_loader: object
            train_samples: list
            data_stats: dict
            ood_loader: object = None

        def _register_data_source(_name):
            def decorator(func):
                return func
            return decorator

        sources_module.DataPipelineResult = _StubDataPipelineResult
        sources_module.register_data_source = _register_data_source
        sys.modules["data.sources"] = sources_module
        data_module.sources = sources_module

    utils_module = sys.modules.get("utils")
    if utils_module is None:
        utils_module = types.ModuleType("utils")
        utils_module.__path__ = []
        sys.modules["utils"] = utils_module

    if "utils.grouping" not in sys.modules:
        sys.modules["utils.grouping"] = _load_grouping_module()


def _load_visomaster_source_module():
    _ensure_torch_stub()
    _ensure_source_package_stubs()

    module_name = "data.sources.visomaster"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = Path(__file__).resolve().parents[1] / "data" / "sources" / "visomaster.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_combined_paired_source_module():
    _ensure_torch_stub()
    _ensure_source_package_stubs()
    _load_visomaster_source_module()

    module_name = "data.sources.combined_paired"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = Path(__file__).resolve().parents[1] / "data" / "sources" / "combined_paired.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _loader_option(loader, name):
    if hasattr(loader, "kwargs"):
        return loader.kwargs[name]
    return getattr(loader, name)


def test_context_variation_disabled_by_default_for_non_vcd_presets():
    """Non-vcd presets should NOT include context variation transforms."""
    pytest.importorskip("albumentations")
    pytest.importorskip("cv2")
    mod = _load_pipelines_module()
    presets = mod._QUALITY_TARGETED_PRESETS

    for name in ("light", "moderate", "strong"):
        preset = presets[name]
        assert preset.get("context_variation_enabled") is False, (
            f"Preset '{name}' should have context_variation_enabled=False"
        )


def test_context_variation_enabled_for_vcd_targeted():
    """vcd_targeted preset should have context variation enabled by default."""
    pytest.importorskip("albumentations")
    pytest.importorskip("cv2")
    mod = _load_pipelines_module()
    presets = mod._QUALITY_TARGETED_PRESETS

    assert presets["vcd_targeted"]["context_variation_enabled"] is True


def test_context_variation_block_produces_transforms():
    """_build_context_variation_block should return transforms when enabled."""
    pytest.importorskip("albumentations")
    pytest.importorskip("cv2")
    mod = _load_pipelines_module()
    _build_context_variation_block = mod._build_context_variation_block

    enabled_p = {"context_variation_enabled": True}
    disabled_p = {"context_variation_enabled": False}
    empty_p = {}

    assert len(_build_context_variation_block(enabled_p)) == 1  # OneOf wrapper
    assert len(_build_context_variation_block(disabled_p)) == 0
    assert len(_build_context_variation_block(empty_p)) == 0


def test_context_variation_forward_pass_all_families():
    """Context variation should not break pipeline for any family when enabled."""
    pytest.importorskip("albumentations")
    pytest.importorskip("cv2")
    mod = _load_pipelines_module()
    _build_family_quality_pipeline = mod._build_family_quality_pipeline
    presets = mod._QUALITY_TARGETED_PRESETS

    preset = presets["vcd_targeted"]
    image = np.full((64, 64, 3), 127, dtype=np.uint8)

    families = [
        "df40_fake", "deeplive_non_enhanced_fake", "deeplive_enhanced_fake",
        "visomaster_fake", "visomaster_enhanced_fake",
        "df40_real", "realpool_real", "external_real",
    ]
    for family in families:
        pipeline = _build_family_quality_pipeline(family, preset)
        out = pipeline(image=image)["image"]
        assert isinstance(out, np.ndarray), f"Pipeline failed for family '{family}'"
        assert out.shape == image.shape, f"Shape mismatch for family '{family}'"


def test_vcd_targeted_router_forward_pass_with_context_variation():
    """End-to-end: vcd_targeted router with context variation should work for all families."""
    pytest.importorskip("albumentations")
    pytest.importorskip("torch")
    from data.augmentations.pipelines import create_quality_targeted_family_router

    router = create_quality_targeted_family_router(strength="vcd_targeted")
    image = np.full((64, 64, 3), 127, dtype=np.uint8)

    test_cases = [
        {"label": 1, "source": "df40", "method": "blendface"},
        {"label": 1, "source": "deeplive", "method": "deeplive_edge_cases"},
        {"label": 1, "source": "deeplive", "method": "deeplive_edge_cases_enhanced"},
        {"label": 1, "source": "visomaster", "method": "visomaster_CSCS"},
        {"label": 1, "source": "visomaster_enhanced", "method": "visomaster_enhanced_gfpgan"},
        {"label": 1, "source": "visomaster_enhanced", "method": "visomaster_enhanced_codeformer"},
        {"label": 0, "source": "df40", "method": "faceforensics++"},
        {"label": 0, "source": "deeplive", "method": "deeplive_edge_cases"},
        {"label": 0, "source": "external", "method": "external_vcd_real"},
    ]
    for meta in test_cases:
        out = router(image=image, landmarks=None, meta=meta)
        assert isinstance(out, np.ndarray), f"Failed for meta={meta}"
        assert out.shape == image.shape, f"Shape mismatch for meta={meta}"


def test_deeplive_effective_strategy_resolution_from_sample_id():
    module_path = Path(__file__).resolve().parents[1] / "dataset" / "deeplive_dataset.py"
    spec = importlib.util.spec_from_file_location("deeplive_dataset_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    resolve_effective_strategy = module.resolve_effective_strategy

    effective, enhancement = resolve_effective_strategy(
        sample_id="edge_cases_enhanced_0007",
        raw_strategy="edge_cases",
        enhancement=None,
    )
    assert effective == "edge_cases_enhanced"
    assert enhancement == "enhanced"

    effective_plain, enhancement_plain = resolve_effective_strategy(
        sample_id="quality_enhancement_0010",
        raw_strategy="quality_enhancement",
        enhancement=None,
    )
    assert effective_plain == "quality_enhancement"
    assert enhancement_plain == "none"


@dataclass
class _FakeDF40Sample:
    pair_id: str
    method: str
    target_identity: str


@dataclass
class _FakeDeepLiveSample:
    sample_id: str
    strategy: str
    raw_strategy: str
    effective_strategy: str
    has_landmarks: bool
    original_video_name: str


class _FakeDF40Dataset:
    last_methods = None

    def __init__(self, pair_json_path, bucket_name, gcs_project, methods=None, **kwargs):
        self.methods = methods
        _FakeDF40Dataset.last_methods = methods

    def discover_samples(self):
        samples = [
            _FakeDF40Sample(pair_id="p1", method="blendface", target_identity="001"),
            _FakeDF40Sample(pair_id="p2", method="simswap", target_identity="002"),
        ]
        if self.methods is None:
            return samples
        return [s for s in samples if s.method in set(self.methods)]

    def load_sample_frames(self, sample, frame_indices, as_array=True):
        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in frame_indices]
        return frames, frames


class _FakeDeepLiveDataset:
    last_strategies = None

    def __init__(self, bucket_name, gcs_project, use_landmarks=True, strategies="all", **kwargs):
        self.strategies = strategies
        _FakeDeepLiveDataset.last_strategies = strategies
        self.use_landmarks = use_landmarks

    def discover_samples(self):
        samples = [
            _FakeDeepLiveSample(
                sample_id="edge_cases_0001",
                strategy="edge_cases",
                raw_strategy="edge_cases",
                effective_strategy="edge_cases",
                has_landmarks=False,
                original_video_name="cropped_vid_a.mp4",
            ),
            _FakeDeepLiveSample(
                sample_id="minimal_processing_0001",
                strategy="minimal_processing",
                raw_strategy="minimal_processing",
                effective_strategy="minimal_processing",
                has_landmarks=False,
                original_video_name="cropped_vid_b.mp4",
            ),
            _FakeDeepLiveSample(
                sample_id="edge_cases_enhanced_0001",
                strategy="edge_cases",  # raw manifest strategy (base)
                raw_strategy="edge_cases",
                effective_strategy="edge_cases_enhanced",
                has_landmarks=False,
                original_video_name="cropped_vid_c.mp4",
            ),
        ]
        if self.strategies == "all" or self.strategies is None:
            return samples
        wanted = set(self.strategies)
        return [s for s in samples if s.effective_strategy in wanted]

    def load_sample_frames(self, sample, frame_indices, as_array=True):
        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in frame_indices]
        return frames, frames


def test_combined_paired_wires_df40_methods_and_deeplive_strategy_filters(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("albumentations")
    from data.sources.combined_paired import create_combined_paired_pipeline

    monkeypatch.setitem(
        sys.modules,
        "dataset.df40_paired_dataset",
        types.SimpleNamespace(DF40PairedDataset=_FakeDF40Dataset),
    )
    monkeypatch.setitem(
        sys.modules,
        "dataset.deeplive_dataset",
        types.SimpleNamespace(DeepLiveDataset=_FakeDeepLiveDataset),
    )

    config = {
        "manualSeed": 737,
        "frames_per_batch": 2,
        "frames_per_video": 1,
    }
    data_config = {
        "data_source": "combined_paired",
        "combined_paired": {
            "split_seed": 123,
            "identity_balanced_sampling": True,
            "df40": {
                "enabled": True,
                "pair_json": "dataset/df40_pairs/df40-pair-matching.json",
                "methods": ["blendface"],
                "anchor_indices": [0],
            },
            "deeplive": {
                "enabled": True,
                "anchor_indices": [0],
                "include_strategies": ["edge_cases", "minimal_processing"],
                "exclude_strategies": ["edge_cases_enhanced"],
                "strategy_expectations": {"mode": "noenhanced"},
            },
            "visomaster": {"enabled": False},
            "train_split": 0.8,
            "val_split": 0.1,
        },
    }

    result = create_combined_paired_pipeline(config, data_config, logging.getLogger("test"))

    assert _FakeDF40Dataset.last_methods == ["blendface"]
    assert _FakeDeepLiveDataset.last_strategies == ["edge_cases", "minimal_processing"]
    assert result.data_stats["df40_samples"] == 1
    assert result.data_stats["deeplive_samples"] == 2
    assert result.data_stats["split_seed"] == 123


def test_combined_paired_preflight_fails_when_withenhanced_missing(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("albumentations")
    from data.sources.combined_paired import create_combined_paired_pipeline

    monkeypatch.setitem(
        sys.modules,
        "dataset.df40_paired_dataset",
        types.SimpleNamespace(DF40PairedDataset=_FakeDF40Dataset),
    )
    monkeypatch.setitem(
        sys.modules,
        "dataset.deeplive_dataset",
        types.SimpleNamespace(DeepLiveDataset=_FakeDeepLiveDataset),
    )

    config = {
        "manualSeed": 737,
        "frames_per_batch": 2,
        "frames_per_video": 1,
    }
    data_config = {
        "data_source": "combined_paired",
        "combined_paired": {
            "split_seed": 123,
            "identity_balanced_sampling": True,
            "df40": {
                "enabled": True,
                "pair_json": "dataset/df40_pairs/df40-pair-matching.json",
                "methods": ["blendface"],
                "anchor_indices": [0],
            },
            "deeplive": {
                "enabled": True,
                "anchor_indices": [0],
                "include_strategies": ["edge_cases", "minimal_processing", "edge_cases_enhanced"],
                "exclude_strategies": [],
                "strategy_expectations": {
                    "mode": "withenhanced",
                    "min_counts": {
                        "edge_cases_enhanced": 400,
                        "minimal_processing_enhanced": 390,
                    },
                },
            },
            "visomaster": {"enabled": False},
            "train_split": 0.8,
            "val_split": 0.1,
        },
    }

    with pytest.raises(ValueError, match="DeepLive strategy preflight failed"):
        create_combined_paired_pipeline(config, data_config, logging.getLogger("test"))


def test_combined_paired_pipeline_uses_top_level_num_workers_and_persistent_workers(monkeypatch):
    combined_paired_module = _load_combined_paired_source_module()
    create_combined_paired_pipeline = combined_paired_module.create_combined_paired_pipeline

    monkeypatch.setitem(
        sys.modules,
        "dataset.df40_paired_dataset",
        types.SimpleNamespace(DF40PairedDataset=_FakeDF40Dataset),
    )
    monkeypatch.setitem(
        sys.modules,
        "dataset.deeplive_dataset",
        types.SimpleNamespace(DeepLiveDataset=_FakeDeepLiveDataset),
    )
    monkeypatch.setattr(combined_paired_module.torch.cuda, "is_available", lambda: True)

    config = {
        "manualSeed": 737,
        "frames_per_batch": 2,
        "frames_per_video": 1,
        "num_workers": 3,
        "prefetch_factor": 5,
    }
    data_config = {
        "data_source": "combined_paired",
        "combined_paired": {
            "split_seed": 123,
            "identity_balanced_sampling": True,
            "df40": {
                "enabled": True,
                "pair_json": "dataset/df40_pairs/df40-pair-matching.json",
                "methods": ["blendface"],
                "anchor_indices": [0],
            },
            "deeplive": {
                "enabled": True,
                "anchor_indices": [0],
                "include_strategies": ["edge_cases", "minimal_processing"],
                "exclude_strategies": ["edge_cases_enhanced"],
                "strategy_expectations": {"mode": "noenhanced"},
            },
            "visomaster": {"enabled": False},
            "train_split": 0.8,
            "val_split": 0.1,
        },
    }

    result = create_combined_paired_pipeline(
        config,
        data_config,
        logging.getLogger("test"),
        transform=lambda image, landmarks, meta=None: image,
    )

    assert _loader_option(result.train_loader, "num_workers") == 3
    assert _loader_option(result.train_loader, "prefetch_factor") == 5
    assert _loader_option(result.train_loader, "persistent_workers") is True
    assert _loader_option(result.val_in_dist_loader._dataloader, "persistent_workers") is True
    assert _loader_option(result.val_holdout_loader._dataloader, "persistent_workers") is True


class _MiniDF40Dataset:
    def load_sample_frames(self, sample, frame_indices, as_array=True):
        frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in frame_indices]
        return frames, frames


def test_transform_backward_compatibility_two_arg_signature():
    pytest.importorskip("torch")
    pytest.importorskip("albumentations")
    from data.sources.combined_paired import (
        CombinedBatchingConfig,
        CombinedPairedIterableDataset,
        UnifiedPairedSample,
    )

    transform_calls = {"count": 0}

    def two_arg_transform(image, landmarks):
        transform_calls["count"] += 1
        return image

    sample = UnifiedPairedSample(
        identity="df40_001",
        source="df40",
        original_sample=_FakeDF40Sample(pair_id="p1", method="blendface", target_identity="001"),
        method="blendface",
        has_landmarks=False,
        sample_id="p1",
    )

    ds = CombinedPairedIterableDataset(
        samples=[sample],
        df40_dataset=_MiniDF40Dataset(),
        deeplive_dataset=None,
        config=CombinedBatchingConfig(df40_sparse_indices=[0], deeplive_sparse_indices=[0], visomaster_sparse_indices=[0]),
        transform=two_arg_transform,
        shuffle=False,
        seed=1,
        method_mapping={"blendface": 0},
    )

    it = iter(ds)
    next(it)
    next(it)
    assert transform_calls["count"] == 2


def test_weighted_identity_sampler_shifts_family_distribution():
    pytest.importorskip("torch")
    from data.sources.combined_paired import (
        CombinedBatchingConfig,
        CombinedPairedIterableDataset,
        UnifiedPairedSample,
    )

    class _NoopDataset:
        def load_sample_frames(self, sample, frame_indices, as_array=True):
            frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in frame_indices]
            return frames, frames

    samples = []
    for idx in range(40):
        identity = f"realpool_identity_{idx:03d}"
        samples.append(
            UnifiedPairedSample(
                identity=identity,
                source="visomaster",
                original_sample=object(),
                method="visomaster_CSCS",
                has_landmarks=False,
                sample_id=f"vm_{idx:03d}",
            )
        )
        samples.append(
            UnifiedPairedSample(
                identity=identity,
                source="deeplive",
                original_sample=object(),
                method="deeplive_edge_cases_enhanced",
                has_landmarks=False,
                sample_id=f"dl_{idx:03d}",
            )
        )

    uniform_cfg = CombinedBatchingConfig(
        identity_balanced_sampling=True,
        identity_sampling_strategy="identity_resample_uniform",
        df40_sparse_indices=[0],
        deeplive_sparse_indices=[0],
        visomaster_sparse_indices=[0],
    )
    weighted_cfg = CombinedBatchingConfig(
        identity_balanced_sampling=True,
        identity_sampling_strategy="identity_resample_weighted",
        identity_family_weights={
            "visomaster_fake": 0.6,
            "deeplive_enhanced_fake": 4.0,
        },
        df40_sparse_indices=[0],
        deeplive_sparse_indices=[0],
        visomaster_sparse_indices=[0],
    )

    uniform_ds = CombinedPairedIterableDataset(
        samples=samples,
        df40_dataset=_NoopDataset(),
        deeplive_dataset=_NoopDataset(),
        config=uniform_cfg,
        transform=None,
        shuffle=False,
        seed=1,
        method_mapping={},
    )
    weighted_ds = CombinedPairedIterableDataset(
        samples=samples,
        df40_dataset=_NoopDataset(),
        deeplive_dataset=_NoopDataset(),
        config=weighted_cfg,
        transform=None,
        shuffle=False,
        seed=1,
        method_mapping={},
    )

    uniform_hits = 0
    weighted_hits = 0
    total_selected = 0

    for trial in range(60):
        rng_uniform = random.Random(1000 + trial)
        rng_weighted = random.Random(1000 + trial)
        uniform_sel = uniform_ds._get_identity_balanced_samples(rng_uniform, 0, 1)
        weighted_sel = weighted_ds._get_identity_balanced_samples(rng_weighted, 0, 1)

        uniform_hits += sum(1 for s in uniform_sel if s.method == "deeplive_edge_cases_enhanced")
        weighted_hits += sum(1 for s in weighted_sel if s.method == "deeplive_edge_cases_enhanced")
        total_selected += len(uniform_sel)

    uniform_ratio = uniform_hits / total_selected
    weighted_ratio = weighted_hits / total_selected

    assert weighted_ratio > uniform_ratio + 0.15


def test_combined_validation_uses_method_id_mapping_for_per_method_metrics():
    torch = pytest.importorskip("torch")
    from trainer.trainer import Trainer

    class _DummyModel:
        def __init__(self):
            self.device = torch.device("cpu")

        def eval(self):
            return self

        def __call__(self, data_dict, inference=True):
            # B=2, T=1 => flatten length 2
            return {"prob": torch.tensor([0.1, 0.9], dtype=torch.float32)}

        def get_losses(self, data_dict, predictions):
            return {"overall": torch.tensor(0.1, dtype=torch.float32)}

    class _DummyValidationLoader:
        def __init__(self):
            self.videos_by_method = {"combined_val": [object(), object()]}
            self.method_id_to_name = {0: "faceforensics++", 1: "deeplive_edge_cases_enhanced"}

        def keys(self):
            return self.videos_by_method.keys()

        def __getitem__(self, method):
            data_dict = {
                "image": torch.zeros((2, 1, 3, 8, 8), dtype=torch.float32),
                "label": torch.tensor([0, 1], dtype=torch.long),
                "method_id": torch.tensor([0, 1], dtype=torch.long),
                "video_id": ["v_real", "v_fake"],
                "frame_paths": [["v_real_f0.png"], ["v_fake_f0.png"]],
            }
            return [data_dict]

    trainer = Trainer.__new__(Trainer)
    trainer.model = _DummyModel()
    trainer.logger = Mock()
    trainer.config = {
        "local_rank": 0,
        "dataset_methods": {"use_real_sources": ["faceforensics++"]},
        "save_ckpt": False,
    }
    trainer.metric_scoring = "auc"
    trainer.best_val_metric = -1.0
    trainer.best_val_epoch = 0
    trainer.epochs_without_improvement = 0
    trainer.early_stopping_enabled = False
    trainer.early_stopping_min_delta = 0.0
    trainer.early_stopping_patience = 10
    trainer.early_stop_triggered = False
    trainer.wandb_run = None
    trainer.top_n_checkpoints = []
    trainer.top_n_size = 3
    trainer.first_best_gcs_path = None

    metrics = trainer.test_epoch(
        epoch=0,
        step_cnt=10,
        validation_loader=_DummyValidationLoader(),
        log_prefix="unit_combined",
        is_primary_metric=False,
        generate_detailed_reports=False,
    )

    per_method = metrics.get("per_method", {})
    assert "faceforensics++" in per_method
    assert "deeplive_edge_cases_enhanced" in per_method


def test_report_generation_emits_group_metrics_and_group_summary(tmp_path):
    pytest.importorskip("torch")
    from trainer.trainer import Trainer

    trainer = Trainer.__new__(Trainer)
    trainer.logger = Mock()

    uploaded = {}

    def _fake_upload(local_path, gcs_path):
        uploaded[gcs_path] = Path(local_path).read_text()

    trainer._upload_to_gcs = _fake_upload

    frame_data = [
        ["faceforensics++", 0, "v1", "v1_f0.png", 0.10, "df40_real", "df40_real"],
        ["blendface", 1, "v2", "v2_f0.png", 0.90, "df40_fake", "df40_fake"],
        ["wma_failure_fake", 1, "v3", "v3_f0.png", 0.30, "wma_failure_fake", "wma_failure_fake"],
    ]
    video_data = [
        ["faceforensics++", 0, "v1", 0.10, 0, 1, "df40_real", "df40_real"],
        ["blendface", 1, "v2", 0.90, 1, 1, "df40_fake", "df40_fake"],
        ["wma_failure_fake", 1, "v3", 0.30, 0, 0, "wma_failure_fake", "wma_failure_fake"],
    ]

    trainer._generate_and_upload_reports(
        log_prefix="unit_test",
        frame_data=frame_data,
        video_data=video_data,
        all_preds=[0.1, 0.9, 0.3],
        all_labels=[0, 1, 1],
        method_preds={},
        method_labels={},
        generate_detailed_reports=True,
        run_name="unit",
        output_gcs_folder="gs://dummy-bucket/unit-tests",
        output_filename_prefix="phase4_",
    )

    group_csv_paths = [p for p in uploaded if p.endswith("phase4_group_metrics.csv")]
    assert group_csv_paths, "group_metrics.csv was not uploaded"
    assert "group_key,family_key,n_videos" in uploaded[group_csv_paths[0]]

    summary_paths = [p for p in uploaded if p.endswith("phase4_summary_report.txt")]
    assert summary_paths, "summary_report.txt was not uploaded"
    summary_text = uploaded[summary_paths[0]]
    assert "Per-Group Performance" in summary_text
    assert "Best-vs-worst group accuracy gap" in summary_text


# ==============================================================================
# VisoMaster Enhanced Integration Tests
# ==============================================================================


def test_visomaster_enhanced_augmentation_pipeline_all_presets():
    """visomaster_enhanced_fake pipeline should work for all preset strengths."""
    pytest.importorskip("albumentations")
    pytest.importorskip("cv2")
    mod = _load_pipelines_module()
    _build = mod._build_family_quality_pipeline
    presets = mod._QUALITY_TARGETED_PRESETS

    image = np.full((64, 64, 3), 127, dtype=np.uint8)
    for strength_name, preset in presets.items():
        pipeline = _build("visomaster_enhanced_fake", preset)
        out = pipeline(image=image)["image"]
        assert isinstance(out, np.ndarray), f"Failed for strength '{strength_name}'"
        assert out.shape == image.shape, f"Shape mismatch for strength '{strength_name}'"


def test_visomaster_enhanced_grouping_does_not_collide_with_visomaster():
    """visomaster_enhanced methods must NOT map to visomaster_fake family."""
    grouping = _load_grouping_module()

    # Enhanced samples routed by source
    g1, f1 = grouping.infer_group_and_family(
        label=1, method="visomaster_enhanced_gfpgan", source="visomaster_enhanced"
    )
    assert f1 == "visomaster_enhanced_fake"
    assert g1 == "visomaster_enhanced_fake"

    # Original visomaster stays separate
    g2, f2 = grouping.infer_group_and_family(
        label=1, method="visomaster_CSCS", source="visomaster"
    )
    assert f2 == "visomaster_fake"
    assert g2 == "visomaster_fake"


def test_visomaster_enhanced_real_maps_to_realpool():
    """Real frames from visomaster_enhanced should share realpool identity space."""
    grouping = _load_grouping_module()

    _, family = grouping.infer_group_and_family(
        label=0, method="visomaster_enhanced_gfpgan", source="visomaster_enhanced"
    )
    assert family == "realpool_real"


def test_visomaster_enhanced_quality_domain_map():
    """visomaster_enhanced should be mapped in QUALITY_DOMAIN_MAP."""
    src = Path(__file__).resolve().parents[1] / "data" / "sources" / "combined_paired.py"
    text = src.read_text()
    assert '"visomaster_enhanced"' in text or "'visomaster_enhanced'" in text


def test_visomaster_enhanced_router_registered():
    """QualityTargetedFamilyRouter should include visomaster_enhanced_fake pipeline."""
    pytest.importorskip("albumentations")
    from data.augmentations.pipelines import create_quality_targeted_family_router

    router = create_quality_targeted_family_router(strength="vcd_targeted")
    assert "visomaster_enhanced_fake" in router._pipelines


def _make_visomaster_teams_enhanced_sample(companion_domain="teams_v2", enhancers=("gfpgan", "codeformer")):
    visomaster_module = _load_visomaster_source_module()
    VisoMasterTeamsEnhancedSample = visomaster_module.VisoMasterTeamsEnhancedSample

    return VisoMasterTeamsEnhancedSample(
        sample_id="visomaster_CSCS_00007",
        strategy="visomaster_CSCS",
        swap_model="CSCS",
        frame_count=16,
        companion_bucket=(
            "live-deepfake-methods-real-and-fake-frames-cropped-teams-v2"
            if companion_domain == "teams_v2"
            else "live-deepfake-methods-real-and-fake-frames-cropped"
        ),
        companion_domain=companion_domain,
        companion_real_ext=".jpg" if companion_domain == "teams_v2" else ".png",
        companion_fake_ext=".jpg" if companion_domain == "teams_v2" else ".png",
        companion_real_frame_count=16,
        companion_fake_frame_count=16,
        enhanced_bucket="enhanced-visomaster-cropped",
        available_enhancers=tuple(enhancers),
        enhancer_frame_counts={name: 16 for name in enhancers},
        manifest={},
    )


def test_visomaster_teams_enhanced_resolver_manifest_loads_merged_samples(tmp_path):
    visomaster_module = _load_visomaster_source_module()
    discover_visomaster_teams_enhanced_samples = (
        visomaster_module.discover_visomaster_teams_enhanced_samples
    )

    resolver_manifest = {
        "version": 1,
        "rows": [
            {
                "sample_id": "visomaster_CSCS_00007",
                "strategy": "visomaster_CSCS",
                "resolved_companion_bucket": "live-deepfake-methods-real-and-fake-frames-cropped-teams-v2",
                "resolution_status": "teams_v2_companion",
                "resolved_real_frame_count": 22,
                "resolved_fake_frame_count": 18,
                "resolved_extensions": {
                    "real": {".jpg": 22},
                    "fake": {".jpg": 18},
                },
                "available_enhancers": ["gfpgan", "codeformer"],
                "enhancer_frame_counts": {"gfpgan": 16, "codeformer": 17},
                "all_expected_enhancers_present": True,
            },
            {
                "sample_id": "visomaster_SimSwap512_00099",
                "strategy": "visomaster_SimSwap512",
                "resolved_companion_bucket": "live-deepfake-methods-real-and-fake-frames-cropped",
                "resolution_status": "clean_companion_only",
                "resolved_real_frame_count": 16,
                "resolved_fake_frame_count": 16,
                "resolved_extensions": {
                    "real": {".png": 16},
                    "fake": {".png": 16},
                },
                "available_enhancers": ["gpen-512"],
                "enhancer_frame_counts": {"gpen-512": 16},
                "all_expected_enhancers_present": False,
            },
            {
                "sample_id": "visomaster_CSCS_00999",
                "strategy": "visomaster_CSCS",
                "resolved_companion_bucket": "",
                "resolution_status": "missing_companion",
                "resolved_real_frame_count": 0,
                "resolved_fake_frame_count": 0,
                "resolved_extensions": {},
                "available_enhancers": ["gfpgan"],
                "enhancer_frame_counts": {"gfpgan": 16},
                "all_expected_enhancers_present": True,
            },
        ],
    }
    manifest_path = tmp_path / "resolver.json"
    manifest_path.write_text(json.dumps(resolver_manifest))

    samples = discover_visomaster_teams_enhanced_samples(
        resolver_manifest_uri=str(manifest_path),
        enhanced_bucket="enhanced-visomaster-cropped",
    )

    assert len(samples) == 2
    teams_sample = samples[0]
    clean_sample = samples[1]

    assert teams_sample.companion_domain == "teams_v2"
    assert teams_sample.companion_real_ext == ".jpg"
    assert teams_sample.companion_fake_ext == ".jpg"
    assert teams_sample.method_variants == (
        "visomaster_CSCS",
        "visomaster_enhanced_codeformer",
        "visomaster_enhanced_gfpgan",
    )

    assert clean_sample.companion_domain == "clean_fallback"
    assert clean_sample.method_variants == (
        "visomaster_SimSwap512",
        "visomaster_enhanced_gpen_bfr_512",
    )


def test_visomaster_teams_enhanced_unified_samples_do_not_multiply_by_enhancer():
    combined_paired_module = _load_combined_paired_source_module()
    create_unified_samples_from_visomaster_teams_enhanced = (
        combined_paired_module.create_unified_samples_from_visomaster_teams_enhanced
    )

    merged_sample = _make_visomaster_teams_enhanced_sample(
        companion_domain="teams_v2",
        enhancers=("gfpgan", "codeformer", "vqfr-v2"),
    )
    unified = create_unified_samples_from_visomaster_teams_enhanced(
        [merged_sample],
        logging.getLogger("test"),
    )

    assert len(unified) == 1
    assert unified[0].sample_id == "visomaster_CSCS_00007"
    assert unified[0].method == "visomaster_CSCS"
    assert unified[0].method_variants == (
        "visomaster_CSCS",
        "visomaster_enhanced_gfpgan",
        "visomaster_enhanced_codeformer",
        "visomaster_enhanced_vqfr_v2",
    )
    assert unified[0].sampling_family_key == "visomaster_enhanced_fake"


def test_visomaster_teams_enhanced_iteration_switches_branch_metadata(monkeypatch):
    combined_paired_module = _load_combined_paired_source_module()
    visomaster_module = _load_visomaster_source_module()
    CombinedBatchingConfig = combined_paired_module.CombinedBatchingConfig
    CombinedPairedIterableDataset = combined_paired_module.CombinedPairedIterableDataset
    UnifiedPairedSample = combined_paired_module.UnifiedPairedSample

    merged_sample = _make_visomaster_teams_enhanced_sample(
        companion_domain="teams_v2",
        enhancers=("gfpgan",),
    )
    unified_sample = UnifiedPairedSample(
        identity="realpool_00007",
        source="visomaster_teams_enhanced",
        original_sample=merged_sample,
        method=merged_sample.original_method,
        has_landmarks=False,
        sample_id=merged_sample.sample_id,
        method_variants=merged_sample.method_variants,
        sampling_family_key="visomaster_enhanced_fake",
    )

    seen_branches = []

    def _fake_load(sample, anchor_indices, fake_branch="original", as_array=True, client=None):
        seen_branches.append(fake_branch)
        real = [np.full((8, 8, 3), 10, dtype=np.uint8) for _ in anchor_indices]
        fake_value = 20 if fake_branch == "original" else 30
        fake = [np.full((8, 8, 3), fake_value, dtype=np.uint8) for _ in anchor_indices]
        return real, fake

    monkeypatch.setattr(
        visomaster_module,
        "load_visomaster_teams_enhanced_frames",
        _fake_load,
    )

    original_ds = CombinedPairedIterableDataset(
        samples=[unified_sample],
        df40_dataset=None,
        deeplive_dataset=None,
        config=CombinedBatchingConfig(
            visomaster_teams_enhanced_sparse_indices=[0, 2],
            visomaster_teams_enhanced_p_original=1.0,
        ),
        transform=None,
        shuffle=False,
        seed=1,
        method_mapping={
            "visomaster_CSCS": 0,
            "visomaster_enhanced_gfpgan": 1,
        },
    )
    original_items = list(
        original_ds._iterate_visomaster_teams_enhanced_sample(
            unified_sample,
            random.Random(1),
        )
    )

    assert seen_branches[-1] == "original"
    assert original_items[0]["label"] == 0
    assert original_items[0]["method"] == "visomaster_CSCS"
    assert original_items[0]["quality_domain"] == 1
    assert original_items[1]["label"] == 1
    assert original_items[1]["method"] == "visomaster_CSCS"
    assert original_items[1]["quality_domain"] == 1
    assert original_items[1]["method_id"] == 0

    enhanced_ds = CombinedPairedIterableDataset(
        samples=[unified_sample],
        df40_dataset=None,
        deeplive_dataset=None,
        config=CombinedBatchingConfig(
            visomaster_teams_enhanced_sparse_indices=[0, 2],
            visomaster_teams_enhanced_p_original=0.0,
        ),
        transform=None,
        shuffle=False,
        seed=1,
        method_mapping={
            "visomaster_CSCS": 0,
            "visomaster_enhanced_gfpgan": 1,
        },
    )
    enhanced_items = list(
        enhanced_ds._iterate_visomaster_teams_enhanced_sample(
            unified_sample,
            random.Random(1),
        )
    )

    assert seen_branches[-1] == "gfpgan"
    assert enhanced_items[0]["label"] == 0
    assert enhanced_items[0]["method"] == "visomaster_enhanced_gfpgan"
    assert enhanced_items[0]["quality_domain"] == 1
    assert enhanced_items[1]["label"] == 1
    assert enhanced_items[1]["method"] == "visomaster_enhanced_gfpgan"
    assert enhanced_items[1]["quality_domain"] == 2
    assert enhanced_items[1]["method_id"] == 1


def test_visomaster_iteration_reuses_cached_gcs_client(monkeypatch):
    combined_paired_module = _load_combined_paired_source_module()
    visomaster_module = _load_visomaster_source_module()
    CombinedBatchingConfig = combined_paired_module.CombinedBatchingConfig
    CombinedPairedIterableDataset = combined_paired_module.CombinedPairedIterableDataset
    UnifiedPairedSample = combined_paired_module.UnifiedPairedSample
    VisoMasterSample = visomaster_module.VisoMasterSample

    sentinel_client = object()
    client_calls = {"count": 0}

    def _fake_client_factory():
        client_calls["count"] += 1
        return sentinel_client

    monkeypatch.setattr(combined_paired_module.storage, "Client", _fake_client_factory)

    seen_clients = []

    def _fake_load(sample, anchor_indices, as_array=True, client=None):
        seen_clients.append(client)
        real = [np.full((8, 8, 3), 10, dtype=np.uint8) for _ in anchor_indices]
        fake = [np.full((8, 8, 3), 20, dtype=np.uint8) for _ in anchor_indices]
        return real, fake

    monkeypatch.setattr(visomaster_module, "load_visomaster_frames", _fake_load)

    sample = VisoMasterSample(
        sample_id="visomaster_CSCS_00007",
        swap_model="CSCS",
        frame_count=16,
        tier="MINIMAL",
        identity_delta=0.0,
        bucket_name="bucket",
        manifest={"original_video_name": "cropped_vid_a.mp4"},
    )
    unified_sample = UnifiedPairedSample(
        identity="realpool_vid_a",
        source="visomaster",
        original_sample=sample,
        method="visomaster_CSCS",
        has_landmarks=False,
        sample_id=sample.sample_id,
    )

    ds = CombinedPairedIterableDataset(
        samples=[unified_sample],
        df40_dataset=None,
        deeplive_dataset=None,
        config=CombinedBatchingConfig(visomaster_sparse_indices=[0, 2]),
        transform=None,
        shuffle=False,
        seed=1,
        method_mapping={"visomaster_CSCS": 0},
    )

    list(ds._iterate_visomaster_sample(unified_sample, random.Random(1)))
    list(ds._iterate_visomaster_sample(unified_sample, random.Random(2)))

    assert client_calls["count"] == 1
    assert seen_clients == [sentinel_client, sentinel_client]
