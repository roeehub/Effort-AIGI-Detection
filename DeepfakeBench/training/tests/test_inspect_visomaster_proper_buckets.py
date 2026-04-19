"""Tests for pre-integration VisoMaster proper-bucket inspection tooling."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


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


tool = _load_module(
    "inspect_visomaster_proper_buckets_test_module",
    "arena/inspect_visomaster_proper_buckets.py",
)


def _frame_names(ext: str = ".png", count: int = 16):
    return [f"frame_{index:04d}{ext}" for index in range(count)]


def _manifest(
    *,
    sample_id: str,
    real_id: str,
    dataset_key: str,
    strategy: str,
    swap_model: str,
    enhancer: str,
    target_video_name: str,
    real_identity: str,
    ext: str = ".png",
) -> dict:
    frame_files = _frame_names(ext=ext)
    return {
        "sample_id": sample_id,
        "real_id": real_id,
        "dataset_key": dataset_key,
        "frame_counts": {
            "expected_total_per_stream": len(frame_files),
            "cropped_real": len(frame_files),
            "cropped_fake": len(frame_files),
        },
        "frame_files": {
            "real": frame_files,
            "fake": frame_files,
        },
        "pair": {
            "sample_id": sample_id,
            "strategy": strategy,
            "metadata": {
                "parameters": {
                    "swap_model": swap_model,
                    "enhancer": enhancer,
                },
                "origin": {
                    "dataset_key": dataset_key,
                    "task_strategy": f"visomaster_{strategy}",
                    "combo_key": f"{swap_model}|{enhancer}",
                },
                "target_metadata": {
                    "video_name": target_video_name,
                    "real_identity": real_identity,
                },
                "real_video": {
                    "real_id": real_id,
                    "clip_stem": target_video_name,
                },
                "fake_video": {
                    "video_name": target_video_name,
                    "swap_model": swap_model,
                    "enhancer": enhancer,
                },
                "bucket_prep": {
                    "real_id": real_id,
                    "dataset_key": dataset_key,
                },
            },
        },
    }


def _write_manifest(root: Path, sample_id: str, manifest: dict) -> Path:
    path = root / "samples" / sample_id / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest))
    return path


def test_extract_manifest_record_reads_nested_fields():
    manifest = _manifest(
        sample_id="HDTF20260416_00000",
        real_id="RD_Radio14_000__rank01__clip01__s0022.00__d12.0__9bb1330e2164",
        dataset_key="hdtf_20260416",
        strategy="hdtf_20260416",
        swap_model="CSCS",
        enhancer="None",
        target_video_name="RD_Radio14_000__rank01__clip01__s0022.00__d12.0",
        real_identity="RD_Radio14",
    )

    record = tool.extract_manifest_record(
        manifest,
        manifest_ref="gs://hdtf_visomaster_cropped_frames/samples/HDTF20260416_00000/manifest.json",
        source_root="gs://hdtf_visomaster_cropped_frames",
        source_label="hdtf_visomaster_cropped_frames",
    )

    assert record.sample_id == "HDTF20260416_00000"
    assert record.sample_prefix == "HDTF20260416"
    assert record.real_id.endswith("9bb1330e2164")
    assert record.dataset_key == "hdtf_20260416"
    assert record.strategy == "hdtf_20260416"
    assert record.task_strategy == "visomaster_hdtf_20260416"
    assert record.swap_model == "CSCS"
    assert record.enhancer == "None"
    assert record.combo_key == "CSCS|None"
    assert record.target_video_name == "RD_Radio14_000__rank01__clip01__s0022.00__d12.0"
    assert record.clip_stem == "RD_Radio14_000__rank01__clip01__s0022.00__d12.0"
    assert record.real_identity == "RD_Radio14"
    assert record.real_exts == (".png",)
    assert record.fake_exts == (".png",)
    assert record.candidate_keys["real_id"] == record.real_id


def test_build_census_report_flags_duplicate_real_ids_and_stable_sample_ids(tmp_path):
    root = tmp_path / "clean_bucket"
    _write_manifest(
        root,
        "HDTF20260416_00000",
        _manifest(
            sample_id="HDTF20260416_00000",
            real_id="RD_Radio14_000__rank01__clip01__s0022.00__d12.0__9bb1330e2164",
            dataset_key="hdtf_20260416",
            strategy="hdtf_20260416",
            swap_model="CSCS",
            enhancer="None",
            target_video_name="RD_Radio14_000__rank01__clip01__s0022.00__d12.0",
            real_identity="RD_Radio14",
        ),
    )
    _write_manifest(
        root,
        "HDTF20260416_00001",
        _manifest(
            sample_id="HDTF20260416_00001",
            real_id="RD_Radio14_000__rank01__clip01__s0022.00__d12.0__9bb1330e2164",
            dataset_key="hdtf_20260416",
            strategy="hdtf_20260416",
            swap_model="GhostFace-v1",
            enhancer="None",
            target_video_name="RD_Radio14_000__rank01__clip01__s0022.00__d12.0",
            real_identity="RD_Radio14",
        ),
    )

    report = tool.build_census_report([str(root)])
    source = report["sources"][0]

    assert source["sample_count"] == 2
    assert source["counts"]["dataset_keys"] == {"hdtf_20260416": 2}
    assert source["counts"]["swap_models"] == {
        "CSCS": 1,
        "GhostFace-v1": 1,
    }
    assert source["frame_layout"]["expected_total_matches_both"] == 2
    assert source["join_key_candidates"]["sample_id"]["stable_one_to_one"] is True
    assert source["join_key_candidates"]["real_id"]["stable_one_to_one"] is False
    assert source["join_key_candidates"]["real_id"]["duplicate_key_count"] == 1
    assert source["join_key_candidates"]["target_video_name"]["duplicate_key_count"] == 1


def test_build_join_validation_report_prefers_real_id_when_sample_ids_change(tmp_path):
    left_root = tmp_path / "clean_bucket"
    right_root = tmp_path / "teams_bucket"

    left_rows = [
        (
            "HDTF20260416_00000",
            "RD_Radio14_000__rank01__clip01__s0022.00__d12.0__9bb1330e2164",
            "RD_Radio14_000__rank01__clip01__s0022.00__d12.0",
        ),
        (
            "HDTF20260416_00001",
            "RD_Radio14_001__rank01__clip02__s0045.00__d12.0__111111111111",
            "RD_Radio14_001__rank01__clip02__s0045.00__d12.0",
        ),
    ]
    right_rows = [
        (
            "HDTF20260416T_00000",
            "RD_Radio14_000__rank01__clip01__s0022.00__d12.0__9bb1330e2164",
            "RD_Radio14_000__rank01__clip01__s0022.00__d12.0",
        ),
        (
            "HDTF20260416T_00001",
            "RD_Radio14_001__rank01__clip02__s0045.00__d12.0__111111111111",
            "RD_Radio14_001__rank01__clip02__s0045.00__d12.0",
        ),
    ]

    for sample_id, real_id, video_name in left_rows:
        _write_manifest(
            left_root,
            sample_id,
            _manifest(
                sample_id=sample_id,
                real_id=real_id,
                dataset_key="hdtf_20260416",
                strategy="hdtf_20260416",
                swap_model="CSCS",
                enhancer="None",
                target_video_name=video_name,
                real_identity="RD_Radio14",
            ),
        )

    for sample_id, real_id, video_name in right_rows:
        _write_manifest(
            right_root,
            sample_id,
            _manifest(
                sample_id=sample_id,
                real_id=real_id,
                dataset_key="hdtf_20260416_teams",
                strategy="hdtf_20260416_teams",
                swap_model="CSCS",
                enhancer="None",
                target_video_name=video_name,
                real_identity="RD_Radio14",
            ),
        )

    report = tool.build_join_validation_report(str(left_root), str(right_root))
    candidate_reports = {
        candidate_report["candidate"]: candidate_report
        for candidate_report in report["candidate_reports"]
    }

    assert candidate_reports["sample_id"]["one_to_one_overlap_key_count"] == 0
    assert candidate_reports["real_id"]["one_to_one_overlap_key_count"] == 2
    assert candidate_reports["real_id"]["left_stable_one_to_one"] is True
    assert candidate_reports["real_id"]["right_stable_one_to_one"] is True
    assert report["recommended_candidate"]["candidate"] == "real_id"
    assert report["recommended_candidate"]["confidence"] == "strong"
