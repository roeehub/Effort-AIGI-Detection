"""Tests for external validation grouping and deterministic shaping semantics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(module_name: str, relative_path: str):
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
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
r4_runner = _load_module(
    "run_r4_validation_sequential_test_module",
    "scripts/run/run_r4_validation_sequential.py",
)
td_runner = _load_module(
    "run_target_domain_validation_sequential_test_module",
    "arena/run_target_domain_validation_sequential.py",
)


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


def test_external_manifest_filters_split_slice_and_shapes_frames(tmp_path):
    manifest_path = tmp_path / "teams_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "videos": [
                    {
                        "label": "real",
                        "method": "teams_real",
                        "video_id": "teams_real_dev",
                        "frame_paths": ["/tmp/frame_a.jpg"],
                        "identity_key": "cam_test__s32",
                        "split": "dev",
                        "slices": ["teams_real_all", "teams_real_poor_quality"],
                    },
                    {
                        "label": "real",
                        "method": "teams_real",
                        "video_id": "teams_real_lockbox",
                        "frame_paths": ["/tmp/frame_b.jpg"],
                        "identity_key": "cam_test__s33",
                        "split": "lockbox",
                        "slices": ["teams_real_all"],
                    },
                    {
                        "label": "fake",
                        "method": "visomaster_enhanced_macro",
                        "video_id": "fake_macro_dev",
                        "frame_path": "/tmp/frame_c.jpg",
                        "identity_key": "viso_fake",
                        "split": "dev",
                        "slices": ["teams_fake_all", "visomaster_enhanced_macro"],
                    },
                ]
            }
        )
    )

    videos = validation_sources.load_external_manifest_videos(
        manifest_path=str(manifest_path),
        label="real",
        split="dev",
        slices=["teams_real_poor_quality"],
        deterministic_frame_count=4,
    )

    assert len(videos) == 1
    assert videos[0].label == "real"
    assert videos[0].method == "teams_real"
    assert videos[0].video_id == "teams_real_dev"
    assert len(videos[0].frame_paths) == 4
    assert len(set(videos[0].frame_paths)) == 1
    assert isinstance(videos[0].identity, int)


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


def test_target_domain_runner_threads_manifest_suite_flags():
    common = argparse.Namespace(
        output_gcs_folder="gs://training-job-outputs/test_results/target_domain_validation",
        wandb_project="phase2-experiments",
        frames_per_video=8,
        detailed_reports=True,
        df40_mode="none",
        df40_orientation="target_source",
    )
    suite = {
        "name": "teams_real_poor_quality_dev",
        "df40_mode": "none",
        "external_real_manifest": "gs://training-job-outputs/manifests/teams_target_domain.json",
        "external_real_manifest_split": "dev",
        "external_real_manifest_slices": "teams_real_poor_quality",
        "external_real_method": "teams_real",
        "max_external_real": 1200,
        "external_real_deterministic": True,
        "external_fake_manifest": "gs://training-job-outputs/manifests/teams_target_domain.json",
        "external_fake_manifest_split": "dev",
        "external_fake_manifest_slices": "visomaster_enhanced_macro",
        "external_fake_method": "visomaster_enhanced_macro",
        "max_external_fake": 600,
    }

    job_args = td_runner._build_job_args(
        checkpoint_key="FT7",
        checkpoint_path="gs://training-job-outputs/phase2r4_experiments/udgwsu7o/top_n_step500.pth",
        suite=suite,
        common=common,
    )

    assert "--external_real_manifest" in job_args
    assert job_args[job_args.index("--external_real_manifest") + 1] == suite["external_real_manifest"]
    assert "--external_real_manifest_split" in job_args
    assert job_args[job_args.index("--external_real_manifest_split") + 1] == "dev"
    assert "--external_real_manifest_slices" in job_args
    assert job_args[job_args.index("--external_real_manifest_slices") + 1] == "teams_real_poor_quality"
    assert "--external_real_deterministic" in job_args

    assert "--external_fake_manifest" in job_args
    assert job_args[job_args.index("--external_fake_manifest") + 1] == suite["external_fake_manifest"]
    assert "--external_fake_manifest_split" in job_args
    assert job_args[job_args.index("--external_fake_manifest_split") + 1] == "dev"
    assert "--external_fake_manifest_slices" in job_args
    assert job_args[job_args.index("--external_fake_manifest_slices") + 1] == "visomaster_enhanced_macro"


def test_target_domain_runner_accepts_custom_checkpoint_aliases(tmp_path):
    checkpoint_map_path = tmp_path / "checkpoint_map.yaml"
    checkpoint_map_path.write_text(
        "\n".join(
            [
                'r12_g_fp32: "gs://bucket/r12_g_fp32.pth"',
                'track_a_candidate: "gs://bucket/track_a_candidate.pth"',
            ]
        )
    )

    selected = td_runner._resolve_requested_checkpoint_keys(
        checkpoints_arg="r12_g_fp32,track_a_candidate",
        checkpoint_map_path=str(checkpoint_map_path),
    )
    resolved = td_runner._resolve_checkpoints(selected, str(checkpoint_map_path))

    assert selected == ["R12_G_FP32", "TRACK_A_CANDIDATE"]
    assert resolved == {
        "R12_G_FP32": "gs://bucket/r12_g_fp32.pth",
        "TRACK_A_CANDIDATE": "gs://bucket/track_a_candidate.pth",
    }


def test_target_domain_runner_all_expands_available_checkpoint_aliases(tmp_path):
    checkpoint_map_path = tmp_path / "checkpoint_map.yaml"
    checkpoint_map_path.write_text(
        "\n".join(
            [
                'r12_g_fp32: "gs://bucket/r12_g_fp32.pth"',
                'r12_g_int8: "gs://bucket/r12_g_int8.pth"',
                'track_a_candidate: "gs://bucket/track_a_candidate.pth"',
            ]
        )
    )

    selected = td_runner._resolve_requested_checkpoint_keys(
        checkpoints_arg="ALL",
        checkpoint_map_path=str(checkpoint_map_path),
    )

    assert selected == ["R12_G_FP32", "R12_G_INT8", "TRACK_A_CANDIDATE"]


def test_target_domain_runner_builds_absolute_validation_command():
    cmd = td_runner._build_validation_command(["--checkpoint_gcs_path", "gs://bucket/model.pth"])

    assert cmd[0] == sys.executable
    assert cmd[1] == "-u"
    assert Path(cmd[2]).name == "validate_custom_sources.py"
    assert Path(cmd[2]).exists()


def test_target_domain_scorecard_row_uses_fpr_for_real_only_suite():
    suite = {
        "name": "teams_real_all_dev",
        "external_real_manifest": "gs://training-job-outputs/manifests/teams_target_domain.json",
        "external_real_manifest_split": "dev",
        "external_real_manifest_slices": "teams_real_all",
        "external_real_method": "teams_real",
    }
    video_rows = [
        {
            "method": "teams_real",
            "label": 0,
            "video_id": "real_001",
            "avg_video_prob": 0.10,
            "prediction": 0,
            "group_key": "teams_real",
            "family_key": "teams_real",
        },
        {
            "method": "teams_real",
            "label": 0,
            "video_id": "real_002",
            "avg_video_prob": 0.91,
            "prediction": 1,
            "group_key": "teams_real",
            "family_key": "teams_real",
        },
        {
            "method": "teams_real",
            "label": 0,
            "video_id": "real_003",
            "avg_video_prob": 0.30,
            "prediction": 0,
            "group_key": "teams_real",
            "family_key": "teams_real",
        },
    ]

    row = td_runner._build_scorecard_row(
        checkpoint_key="FT7",
        checkpoint_path="gs://training-job-outputs/phase2r13_experiments/example/model.pth",
        suite=suite,
        report_path="/tmp/teams_real_all_dev_ft7_videos_report.csv",
        video_rows=video_rows,
    )

    assert row["label_mode"] == "real_only"
    assert row["split_hint"] == "dev"
    assert row["slice_hint"] == "teams_real_all"
    assert row["method_hint"] == "teams_real"
    assert row["n_videos"] == 3
    assert row["fp"] == 1
    assert row["tn"] == 2
    assert row["score_metric_name"] == "real_fpr_at_0p5"
    assert row["score_metric_value"] == pytest.approx(1 / 3, rel=1e-6)
    assert row["real_tnr_at_0p5"] == pytest.approx(2 / 3, rel=1e-6)


def test_target_domain_scorecard_row_uses_recall_for_fake_only_suite():
    suite = {
        "name": "visomaster_enhanced_macro_dev",
        "external_fake_manifest": "gs://training-job-outputs/manifests/teams_target_domain.json",
        "external_fake_manifest_split": "dev",
        "external_fake_manifest_slices": "visomaster_enhanced_macro",
        "external_fake_method": "visomaster_enhanced_macro",
    }
    video_rows = [
        {
            "method": "visomaster_enhanced_macro",
            "label": 1,
            "video_id": "fake_001",
            "avg_video_prob": 0.89,
            "prediction": 1,
            "group_key": "visomaster_enhanced_fake",
            "family_key": "visomaster_enhanced_fake",
        },
        {
            "method": "visomaster_enhanced_macro",
            "label": 1,
            "video_id": "fake_002",
            "avg_video_prob": 0.78,
            "prediction": 1,
            "group_key": "visomaster_enhanced_fake",
            "family_key": "visomaster_enhanced_fake",
        },
        {
            "method": "visomaster_enhanced_macro",
            "label": 1,
            "video_id": "fake_003",
            "avg_video_prob": 0.21,
            "prediction": 0,
            "group_key": "visomaster_enhanced_fake",
            "family_key": "visomaster_enhanced_fake",
        },
    ]

    row = td_runner._build_scorecard_row(
        checkpoint_key="FT7",
        checkpoint_path="gs://training-job-outputs/phase2r13_experiments/example/model.pth",
        suite=suite,
        report_path="/tmp/visomaster_enhanced_macro_dev_ft7_videos_report.csv",
        video_rows=video_rows,
    )

    assert row["label_mode"] == "fake_only"
    assert row["split_hint"] == "dev"
    assert row["slice_hint"] == "visomaster_enhanced_macro"
    assert row["method_hint"] == "visomaster_enhanced_macro"
    assert row["n_videos"] == 3
    assert row["tp"] == 2
    assert row["fn"] == 1
    assert row["score_metric_name"] == "fake_recall_at_0p5"
    assert row["score_metric_value"] == pytest.approx(2 / 3, rel=1e-6)
    assert row["fake_fnr_at_0p5"] == pytest.approx(1 / 3, rel=1e-6)


def test_target_domain_scorecard_row_falls_back_to_methods_seen_for_method_hint():
    suite = {
        "name": "teams_capture_cam_test_dev",
        "external_fake_manifest": "gs://training-job-outputs/manifests/teams_target_domain.json",
        "external_fake_manifest_split": "dev",
        "external_fake_manifest_slices": "teams_capture_cam_test",
    }
    video_rows = [
        {
            "method": "teams_capture_cam_test_s32",
            "label": 1,
            "video_id": "fake_001",
            "avg_video_prob": 0.91,
            "prediction": 1,
            "group_key": "teams_capture_cam_test",
            "family_key": "teams_capture_cam_test",
        },
        {
            "method": "teams_capture_cam_test_s33",
            "label": 1,
            "video_id": "fake_002",
            "avg_video_prob": 0.22,
            "prediction": 0,
            "group_key": "teams_capture_cam_test",
            "family_key": "teams_capture_cam_test",
        },
    ]

    row = td_runner._build_scorecard_row(
        checkpoint_key="R12_G_FP32",
        checkpoint_path="gs://training-job-outputs/phase2r12_experiments/example/model_fp32.pth",
        suite=suite,
        report_path="/tmp/teams_capture_cam_test_dev_r12_g_fp32_videos_report.csv",
        video_rows=video_rows,
    )

    assert row["checkpoint_precision"] == "fp32"
    assert row["checkpoint_pair_key"] == "R12_G"
    assert row["method_hint"] == "teams_capture_cam_test_s32|teams_capture_cam_test_s33"
    assert row["methods_seen"] == "teams_capture_cam_test_s32|teams_capture_cam_test_s33"


def test_target_domain_scorecard_wide_rows_pivot_primary_metrics():
    scorecard_rows = [
        {
            "checkpoint_key": "FT7",
            "checkpoint_path": "gs://bucket/model_ft7.pth",
            "checkpoint_precision": "other",
            "checkpoint_pair_key": "FT7",
            "suite_name": "teams_real_all_dev",
            "score_metric_name": "real_fpr_at_0p5",
            "score_metric_value": 0.025,
            "accuracy_at_0p5": 0.975,
            "n_videos": 400,
        },
        {
            "checkpoint_key": "FT7",
            "checkpoint_path": "gs://bucket/model_ft7.pth",
            "checkpoint_precision": "other",
            "checkpoint_pair_key": "FT7",
            "suite_name": "visomaster_enhanced_macro_dev",
            "score_metric_name": "fake_recall_at_0p5",
            "score_metric_value": 0.820,
            "accuracy_at_0p5": 0.820,
            "n_videos": 275,
        },
        {
            "checkpoint_key": "FT8",
            "checkpoint_path": "gs://bucket/model_ft8.pth",
            "checkpoint_precision": "other",
            "checkpoint_pair_key": "FT8",
            "suite_name": "teams_real_all_dev",
            "score_metric_name": "real_fpr_at_0p5",
            "score_metric_value": 0.018,
            "accuracy_at_0p5": 0.982,
            "n_videos": 400,
        },
    ]

    wide_rows = td_runner._build_wide_scorecard_rows(scorecard_rows)

    assert len(wide_rows) == 2

    ft7 = next(row for row in wide_rows if row["checkpoint_key"] == "FT7")
    assert ft7["checkpoint_precision"] == "other"
    assert ft7["checkpoint_pair_key"] == "FT7"
    assert ft7["teams_real_all_dev__real_fpr_at_0p5"] == pytest.approx(0.025, rel=1e-6)
    assert ft7["teams_real_all_dev__n_videos"] == 400
    assert ft7["visomaster_enhanced_macro_dev__fake_recall_at_0p5"] == pytest.approx(0.82, rel=1e-6)
    assert ft7["visomaster_enhanced_macro_dev__accuracy_at_0p5"] == pytest.approx(0.82, rel=1e-6)

    ft8 = next(row for row in wide_rows if row["checkpoint_key"] == "FT8")
    assert ft8["teams_real_all_dev__real_fpr_at_0p5"] == pytest.approx(0.018, rel=1e-6)
    assert ft8["teams_real_all_dev__n_videos"] == 400


def test_target_domain_scorecard_pair_delta_rows_compare_fp32_and_int8():
    scorecard_rows = [
        {
            "checkpoint_key": "R12_G_FP32",
            "checkpoint_path": "gs://bucket/r12_g_fp32.pth",
            "checkpoint_precision": "fp32",
            "checkpoint_pair_key": "R12_G",
            "suite_name": "teams_real_all_dev",
            "label_mode": "real_only",
            "split_hint": "dev",
            "slice_hint": "teams_real_all",
            "method_hint": "teams_real",
            "score_metric_name": "real_fpr_at_0p5",
            "score_metric_value": 0.020,
            "accuracy_at_0p5": 0.980,
            "n_videos": 400,
        },
        {
            "checkpoint_key": "R12_G_INT8",
            "checkpoint_path": "gs://bucket/r12_g_int8.pth",
            "checkpoint_precision": "int8",
            "checkpoint_pair_key": "R12_G",
            "suite_name": "teams_real_all_dev",
            "label_mode": "real_only",
            "split_hint": "dev",
            "slice_hint": "teams_real_all",
            "method_hint": "teams_real",
            "score_metric_name": "real_fpr_at_0p5",
            "score_metric_value": 0.028,
            "accuracy_at_0p5": 0.972,
            "n_videos": 400,
        },
        {
            "checkpoint_key": "R12_G_FP32",
            "checkpoint_path": "gs://bucket/r12_g_fp32.pth",
            "checkpoint_precision": "fp32",
            "checkpoint_pair_key": "R12_G",
            "suite_name": "visomaster_enhanced_macro_dev",
            "label_mode": "fake_only",
            "split_hint": "dev",
            "slice_hint": "visomaster_enhanced_macro",
            "method_hint": "visomaster_enhanced_macro",
            "score_metric_name": "fake_recall_at_0p5",
            "score_metric_value": 0.840,
            "accuracy_at_0p5": 0.840,
            "n_videos": 275,
        },
        {
            "checkpoint_key": "R12_G_INT8",
            "checkpoint_path": "gs://bucket/r12_g_int8.pth",
            "checkpoint_precision": "int8",
            "checkpoint_pair_key": "R12_G",
            "suite_name": "visomaster_enhanced_macro_dev",
            "label_mode": "fake_only",
            "split_hint": "dev",
            "slice_hint": "visomaster_enhanced_macro",
            "method_hint": "visomaster_enhanced_macro",
            "score_metric_name": "fake_recall_at_0p5",
            "score_metric_value": 0.830,
            "accuracy_at_0p5": 0.830,
            "n_videos": 275,
        },
    ]

    delta_rows = td_runner._build_pair_delta_rows(scorecard_rows)

    assert len(delta_rows) == 2

    real_delta = next(row for row in delta_rows if row["suite_name"] == "teams_real_all_dev")
    assert real_delta["checkpoint_pair_key"] == "R12_G"
    assert real_delta["metric_direction"] == "lower_is_better"
    assert real_delta["score_metric_delta_int8_minus_fp32"] == pytest.approx(0.008, rel=1e-6)
    assert real_delta["score_metric_directional_delta"] == pytest.approx(-0.008, rel=1e-6)
    assert real_delta["accuracy_at_0p5_delta_int8_minus_fp32"] == pytest.approx(-0.008, rel=1e-6)
    assert real_delta["n_videos_match"] is True

    fake_delta = next(row for row in delta_rows if row["suite_name"] == "visomaster_enhanced_macro_dev")
    assert fake_delta["metric_direction"] == "higher_is_better"
    assert fake_delta["score_metric_delta_int8_minus_fp32"] == pytest.approx(-0.01, rel=1e-6)
    assert fake_delta["score_metric_directional_delta"] == pytest.approx(-0.01, rel=1e-6)
