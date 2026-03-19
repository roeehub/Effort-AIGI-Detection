"""Tests for external validation grouping and deterministic shaping semantics."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_r4_validation_sequential as r4_runner
import run_target_domain_validation_sequential as td_runner


def _load_module(module_name: str, relative_path: str):
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_validation_sources_module():
    @dataclass
    class _VideoInfo:
        label: str
        method: str
        video_id: str
        frame_paths: List[str]
        identity: int
        label_id: int = 0

        def __post_init__(self):
            if self.label not in {"real", "fake"}:
                raise ValueError(f"Invalid label: {self.label}")
            if not self.frame_paths:
                raise ValueError("frame_paths cannot be empty")
            self.label_id = 0 if self.label == "real" else 1

    prepare_stub = types.ModuleType("prepare_splits")
    prepare_stub.VideoInfo = _VideoInfo

    dataset_stub = types.ModuleType("dataset")
    dataset_stub.__path__ = []  # Mark as package for submodule import.

    deeplive_stub = types.ModuleType("dataset.deeplive_dataset")

    def _resolve_effective_strategy(sample_id=None, raw_strategy=None, enhancement=None):
        strategy = str(raw_strategy or "unknown")
        enh = str(enhancement or "none")
        return strategy, enh

    deeplive_stub.resolve_effective_strategy = _resolve_effective_strategy

    previous_prepare = sys.modules.get("prepare_splits")
    previous_dataset = sys.modules.get("dataset")
    previous_deeplive = sys.modules.get("dataset.deeplive_dataset")

    sys.modules["prepare_splits"] = prepare_stub
    sys.modules["dataset"] = dataset_stub
    sys.modules["dataset.deeplive_dataset"] = deeplive_stub

    try:
        return _load_module("validation_sources_test_module", "data/validation_sources.py")
    finally:
        if previous_prepare is None:
            sys.modules.pop("prepare_splits", None)
        else:
            sys.modules["prepare_splits"] = previous_prepare

        if previous_dataset is None:
            sys.modules.pop("dataset", None)
        else:
            sys.modules["dataset"] = previous_dataset

        if previous_deeplive is None:
            sys.modules.pop("dataset.deeplive_dataset", None)
        else:
            sys.modules["dataset.deeplive_dataset"] = previous_deeplive


validation_sources = _load_validation_sources_module()


class _FakeFS:
    def __init__(self, paths: List[str]):
        self._paths = paths

    def glob(self, _pattern: str) -> List[str]:
        return list(self._paths)


def _build_wma_paths(total_images: int = 1202, groups: int = 7) -> List[str]:
    if total_images < groups:
        raise ValueError("total_images must be >= groups")
    base = total_images // groups
    remainder = total_images % groups

    out: List[str] = []
    frame_idx = 0
    for group_idx in range(groups):
        count = base + (1 if group_idx < remainder else 0)
        group_name = f"group_{group_idx + 1:02d}"
        for _ in range(count):
            frame_idx += 1
            out.append(
                f"effort-collected-data/wma_validation/enhanced_fake/{group_name}/frame_{frame_idx:04d}.jpg"
            )
    return out


@pytest.fixture
def _patched_wma_glob(monkeypatch):
    fake_paths = _build_wma_paths(total_images=1202, groups=7)
    monkeypatch.setattr(
        validation_sources,
        "url_to_fs",
        lambda _base: (_FakeFS(fake_paths), None),
    )
    return fake_paths


def test_external_fake_per_image_mode_returns_1202_samples(_patched_wma_glob):
    videos = validation_sources.load_external_fake_videos(
        bucket_name="effort-collected-data",
        prefix="wma_validation/enhanced_fake",
        method_name="wma_failure_fake",
        grouping="per_image",
    )
    assert len(videos) == 1202
    assert all(video.label == "fake" for video in videos)
    assert all(len(video.frame_paths) == 1 for video in videos)


def test_external_fake_by_folder_mode_preserves_7_groups(_patched_wma_glob):
    videos = validation_sources.load_external_fake_videos(
        bucket_name="effort-collected-data",
        prefix="wma_validation/enhanced_fake",
        method_name="wma_failure_fake",
        grouping="by_folder",
    )
    assert len(videos) == 7
    assert sum(len(video.frame_paths) for video in videos) == 1202


def test_external_fake_per_image_deterministic_shapes_frames(_patched_wma_glob):
    videos = validation_sources.load_external_fake_videos(
        bucket_name="effort-collected-data",
        prefix="wma_validation/enhanced_fake",
        method_name="wma_failure_fake",
        grouping="per_image",
        deterministic_frame_count=8,
    )
    assert len(videos) == 1202
    assert all(len(video.frame_paths) == 8 for video in videos)
    assert all(len(set(video.frame_paths)) == 1 for video in videos[:20])


def test_external_fake_per_image_sampling_is_seed_stable(_patched_wma_glob):
    videos_a = validation_sources.load_external_fake_videos(
        bucket_name="effort-collected-data",
        prefix="wma_validation/enhanced_fake",
        method_name="wma_failure_fake",
        grouping="per_image",
        max_videos=1200,
        seed=737,
    )
    videos_b = validation_sources.load_external_fake_videos(
        bucket_name="effort-collected-data",
        prefix="wma_validation/enhanced_fake",
        method_name="wma_failure_fake",
        grouping="per_image",
        max_videos=1200,
        seed=737,
    )
    assert len(videos_a) == 1200
    assert [v.video_id for v in videos_a] == [v.video_id for v in videos_b]


def _r4_args(**overrides):
    base = dict(
        df40_orientation="target_source",
        deeplive_bucket="live-deepfake-methods-real-and-fake-frames-cropped",
        visomaster_bucket="live-deepfake-methods-real-and-fake-frames-cropped",
        visomaster_frames_bucket="live-deepfake-methods-real-and-fake-frames",
        external_real_bucket="effort-collected-data/real/external_youtube_avspeech",
        max_external_real=2000,
        external_real_deterministic=False,
        external_fake_bucket="effort-collected-data",
        external_fake_prefix="wma_validation/enhanced_fake",
        external_fake_method="wma_failure_fake",
        external_fake_grouping="by_folder",
        external_fake_deterministic=False,
        max_external_fake=1202,
        output_gcs_folder="gs://training-job-outputs/test_results/r4_validation",
        wandb_project="phase2-experiments",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_r4_runner_defaults_keep_legacy_behavior_when_new_flags_omitted():
    args = _r4_args()
    job_args = r4_runner._build_job_args(
        key="FT7",
        ckpt_path="gs://training-job-outputs/phase2r4_experiments/udgwsu7o/top_n_step500.pth",
        args=args,
    )
    assert "--external_fake_grouping" in job_args
    grouping_value = job_args[job_args.index("--external_fake_grouping") + 1]
    assert grouping_value == "by_folder"
    assert "--external_real_deterministic" not in job_args
    assert "--external_fake_deterministic" not in job_args


def test_r4_runner_can_emit_per_image_wma_flags():
    args = _r4_args(
        external_real_deterministic=True,
        external_fake_grouping="per_image",
        external_fake_deterministic=True,
    )
    job_args = r4_runner._build_job_args(
        key="FT7",
        ckpt_path="gs://training-job-outputs/phase2r4_experiments/udgwsu7o/top_n_step500.pth",
        args=args,
    )
    assert "--external_fake_grouping" in job_args
    grouping_value = job_args[job_args.index("--external_fake_grouping") + 1]
    assert grouping_value == "per_image"
    assert "--external_real_deterministic" in job_args
    assert "--external_fake_deterministic" in job_args


def test_target_domain_runner_threads_per_image_wma_semantics():
    common = argparse.Namespace(
        output_gcs_folder="gs://training-job-outputs/test_results/target_domain_validation",
        wandb_project="phase2-experiments",
        frames_per_video=8,
        detailed_reports=True,
        df40_mode="none",
        df40_orientation="target_source",
    )
    suite = {
        "name": "zoom_deeplive_enhanced_fake",
        "df40_mode": "none",
        "external_fake_bucket": "effort-collected-data",
        "external_fake_prefix": "zoom_validation/deeplive_enhanced_fake",
        "external_fake_method": "zoom_deeplive_enhanced_fake",
        "external_fake_grouping": "per_image",
        "external_fake_deterministic": True,
        "max_external_fake": 4000,
    }

    job_args = td_runner._build_job_args(
        checkpoint_key="FT7",
        checkpoint_path="gs://training-job-outputs/phase2r4_experiments/udgwsu7o/top_n_step500.pth",
        suite=suite,
        common=common,
    )

    assert "--external_fake_grouping" in job_args
    grouping_value = job_args[job_args.index("--external_fake_grouping") + 1]
    assert grouping_value == "per_image"
    assert "--external_fake_deterministic" in job_args
