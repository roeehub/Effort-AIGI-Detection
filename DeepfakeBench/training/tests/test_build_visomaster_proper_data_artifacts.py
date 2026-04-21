"""Tests for provisional WT-F proper-data artifact generation."""

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
    "build_visomaster_proper_data_artifacts_test_module",
    "arena/build_visomaster_proper_data_artifacts.py",
)


def _frame_names(ext: str = ".png", count: int = 16):
    return [f"frame_{index:04d}{ext}" for index in range(count)]


def _clean_manifest(
    *,
    sample_id: str,
    real_id: str,
    dataset_key: str,
    swap_model: str,
    enhancer: str,
    clip_stem: str,
    real_frame_count: int = 16,
    fake_frame_count: int = 16,
    include_frame_files: bool = True,
) -> dict:
    real_frames = _frame_names(count=real_frame_count)
    fake_frames = _frame_names(count=fake_frame_count)
    manifest = {
        "sample_id": sample_id,
        "real_id": real_id,
        "dataset_key": dataset_key,
        "frame_counts": {
            "expected_total_per_stream": 16,
            "cropped_real": real_frame_count,
            "cropped_fake": fake_frame_count,
        },
        "pair": {
            "sample_id": sample_id,
            "strategy": dataset_key,
            "metadata": {
                "parameters": {
                    "swap_model": swap_model,
                    "enhancer": enhancer,
                },
                "origin": {
                    "dataset_key": dataset_key,
                    "task_strategy": f"visomaster_{dataset_key}",
                    "swap_model": swap_model,
                    "enhancer": enhancer,
                    "task_id": sample_id,
                },
                "real_video": {
                    "real_id": real_id,
                    "clip_stem": clip_stem,
                },
                "fake_video": {
                    "video_name": clip_stem,
                    "swap_model": swap_model,
                    "enhancer": enhancer,
                },
            },
        },
    }
    if include_frame_files:
        manifest["frame_files"] = {
            "real": real_frames,
            "fake": fake_frames,
        }
    return manifest


def _teams_manifest(clean_manifest: dict, *, selected_real: int, selected_fake: int) -> dict:
    return {
        "sample_id": clean_manifest["sample_id"],
        "source": "teams_capture",
        "pipeline_version": "teams_cropped_v1",
        "method_name": (
            f"{clean_manifest['pair']['metadata']['parameters']['swap_model']}__"
            f"{clean_manifest['pair']['metadata']['parameters']['enhancer']}"
        ),
        "frame_counts": {
            "selected_real": selected_real,
            "selected_fake": selected_fake,
            "target_per_stream": 16,
        },
        "frame_files": {
            "real": _frame_names(count=selected_real),
            "fake": _frame_names(count=selected_fake),
        },
        "original_manifest": clean_manifest,
    }


def _write_manifest(root: Path, sample_id: str, manifest: dict) -> None:
    path = root / "samples" / sample_id / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest))


def test_build_inventory_snapshot_uses_sample_overlap_and_filters_ragged_teams(tmp_path):
    clean_root = tmp_path / "clean_bucket"
    teams_root = tmp_path / "teams_bucket"

    keep_clean = _clean_manifest(
        sample_id="HDTF20260416_00000",
        real_id="RD_Radio14_000__rank01__clip01__s0022.00__d12.0__9bb1330e2164",
        dataset_key="hdtf_20260416",
        swap_model="CSCS",
        enhancer="None",
        clip_stem="RD_Radio14_000__rank01__clip01__s0022.00__d12.0",
    )
    ragged_clean = _clean_manifest(
        sample_id="HDTF20260416_00001",
        real_id="RD_Radio14_001__rank01__clip02__s0045.00__d12.0__111111111111",
        dataset_key="hdtf_20260416",
        swap_model="GhostFace-v1",
        enhancer="CodeFormer",
        clip_stem="RD_Radio14_001__rank01__clip02__s0045.00__d12.0",
    )
    clean_only = _clean_manifest(
        sample_id="HDTF20260416_00002",
        real_id="RD_Radio16_000__rank01__clip01__s0010.00__d12.0__222222222222",
        dataset_key="hdtf_20260416",
        swap_model="GhostFace-v2",
        enhancer="None",
        clip_stem="RD_Radio16_000__rank01__clip01__s0010.00__d12.0",
    )

    _write_manifest(clean_root, keep_clean["sample_id"], keep_clean)
    _write_manifest(clean_root, ragged_clean["sample_id"], ragged_clean)
    _write_manifest(clean_root, clean_only["sample_id"], clean_only)

    _write_manifest(
        teams_root,
        keep_clean["sample_id"],
        _teams_manifest(keep_clean, selected_real=16, selected_fake=16),
    )
    _write_manifest(
        teams_root,
        ragged_clean["sample_id"],
        _teams_manifest(ragged_clean, selected_real=16, selected_fake=14),
    )

    inventory, report = tool.build_inventory_snapshot(
        [
            tool.SourcePair(
                name="hdtf",
                clean_source=str(clean_root),
                teams_source=str(teams_root),
            )
        ],
        wave_id="proper_visomaster_wave_test",
        source_logs={},
    )

    assert report["inventory_capture_count"] == 1
    pair_report = report["source_pairs"][0]
    assert pair_report["clean_sample_count"] == 3
    assert pair_report["teams_sample_count"] == 2
    assert pair_report["overlap_sample_count"] == 2
    assert pair_report["kept_sample_count"] == 1
    assert pair_report["clean_only_sample_count"] == 1
    assert pair_report["skipped_ragged_teams_count"] == 1

    capture = inventory["captures"][0]
    assert capture["base_capture_id"] == "HDTF20260416_00000"
    assert capture["identity_id"] == "RD_Radio14"
    assert capture["capture_session_id"] == "RD_Radio14_000"
    assert capture["split_group_id"] == "RD_Radio14__RD_Radio14_000"
    assert capture["quality_band"] == "high"
    assert capture["face_scale_band"] == "big_face"

    fake_clean = next(
        variant
        for variant in capture["variants"]
        if variant["label"] == "fake" and variant["transport"] == "clean"
    )
    assert fake_clean["generator_method"] == "CSCS"
    assert fake_clean["enhancement"] == "none"
    assert fake_clean["frame_paths"][0].endswith(
        "/samples/HDTF20260416_00000/frames/fake/frame_0000.png"
    )


def test_build_inventory_snapshot_preserves_enhanced_method_metadata(tmp_path):
    clean_root = tmp_path / "clean_bucket"
    teams_root = tmp_path / "teams_bucket"

    clean_manifest = _clean_manifest(
        sample_id="QCLIP20260417R2_00000",
        real_id="322__-jlI8ntZrEY__rank01__s0210.98__d06.0__77e5c51a73b1",
        dataset_key="quickclips_20260417_20260418_combined",
        swap_model="GhostFace-v1",
        enhancer="GFPGAN-v1.4",
        clip_stem="322__-jlI8ntZrEY__rank01__s0210.98__d06.0",
    )
    _write_manifest(clean_root, clean_manifest["sample_id"], clean_manifest)
    _write_manifest(
        teams_root,
        clean_manifest["sample_id"],
        _teams_manifest(clean_manifest, selected_real=16, selected_fake=16),
    )

    inventory, report = tool.build_inventory_snapshot(
        [
            tool.SourcePair(
                name="quickclips",
                clean_source=str(clean_root),
                teams_source=str(teams_root),
            )
        ],
        wave_id="proper_visomaster_wave_test",
        source_logs={},
    )

    assert report["inventory_capture_count"] == 1
    capture = inventory["captures"][0]
    assert capture["identity_id"] == "322"
    assert capture["capture_session_id"] == "-jlI8ntZrEY"
    assert capture["quality_band"] == "medium"
    assert capture["face_scale_band"] == "standard"

    fake_variants = [
        variant
        for variant in capture["variants"]
        if variant["label"] == "fake"
    ]
    assert {variant["enhancement"] for variant in fake_variants} == {"enhanced"}
    assert {variant["generator_method"] for variant in fake_variants} == {
        "GhostFace-v1__GFPGAN-v1.4"
    }


def test_build_inventory_snapshot_filters_ragged_clean_by_default(tmp_path):
    clean_root = tmp_path / "clean_bucket"
    teams_root = tmp_path / "teams_bucket"

    keep_clean = _clean_manifest(
        sample_id="QCLIP20260417R2_00000",
        real_id="322__-jlI8ntZrEY__rank01__s0210.98__d06.0__77e5c51a73b1",
        dataset_key="quickclips_20260417_20260418_combined",
        swap_model="GhostFace-v1",
        enhancer="None",
        clip_stem="322__-jlI8ntZrEY__rank01__s0210.98__d06.0",
    )
    ragged_clean = _clean_manifest(
        sample_id="QCLIP20260417R2_00001",
        real_id="323__-jlI8ntZrEY__rank01__s0211.98__d06.0__77e5c51a73b2",
        dataset_key="quickclips_20260417_20260418_combined",
        swap_model="GhostFace-v2",
        enhancer="None",
        clip_stem="323__-jlI8ntZrEY__rank01__s0211.98__d06.0",
        fake_frame_count=14,
    )

    _write_manifest(clean_root, keep_clean["sample_id"], keep_clean)
    _write_manifest(clean_root, ragged_clean["sample_id"], ragged_clean)
    _write_manifest(
        teams_root,
        keep_clean["sample_id"],
        _teams_manifest(keep_clean, selected_real=16, selected_fake=16),
    )
    _write_manifest(
        teams_root,
        ragged_clean["sample_id"],
        _teams_manifest(ragged_clean, selected_real=16, selected_fake=16),
    )

    inventory, report = tool.build_inventory_snapshot(
        [
            tool.SourcePair(
                name="quickclips",
                clean_source=str(clean_root),
                teams_source=str(teams_root),
            )
        ],
        wave_id="proper_visomaster_wave_test",
        source_logs={},
    )

    assert report["inventory_capture_count"] == 1
    pair_report = report["source_pairs"][0]
    assert pair_report["kept_sample_count"] == 1
    assert pair_report["skipped_ragged_clean_count"] == 1
    assert pair_report["skipped_ragged_teams_count"] == 0
    assert inventory["captures"][0]["base_capture_id"] == "QCLIP20260417R2_00000"


def test_render_suite_template_replaces_manifest_path_and_parses_yaml():
    rendered, parsed = tool.render_suite_template(
        "arena/manifests/proper_visomaster_manifest.json"
    )

    assert "<manifest-path>" not in rendered
    assert parsed["suites"][0]["external_real_manifest"] == "arena/manifests/proper_visomaster_manifest.json"
    assert parsed["suites"][4]["external_fake_manifest_slices"] == "proper_fake_teams_all"


def test_build_inventory_snapshot_fails_fast_when_explicit_frame_files_are_missing(tmp_path):
    clean_root = tmp_path / "clean_bucket"
    teams_root = tmp_path / "teams_bucket"

    clean_manifest = _clean_manifest(
        sample_id="HDTF20260416_00000",
        real_id="RD_Radio14_000__rank01__clip01__s0022.00__d12.0__9bb1330e2164",
        dataset_key="hdtf_20260416",
        swap_model="CSCS",
        enhancer="None",
        clip_stem="RD_Radio14_000__rank01__clip01__s0022.00__d12.0",
        include_frame_files=False,
    )
    _write_manifest(clean_root, clean_manifest["sample_id"], clean_manifest)
    _write_manifest(
        teams_root,
        clean_manifest["sample_id"],
        _teams_manifest(
            _clean_manifest(
                sample_id=clean_manifest["sample_id"],
                real_id=clean_manifest["real_id"],
                dataset_key="hdtf_20260416",
                swap_model="CSCS",
                enhancer="None",
                clip_stem="RD_Radio14_000__rank01__clip01__s0022.00__d12.0",
            ),
            selected_real=16,
            selected_fake=16,
        ),
    )

    try:
        tool.build_inventory_snapshot(
            [
                tool.SourcePair(
                    name="hdtf",
                    clean_source=str(clean_root),
                    teams_source=str(teams_root),
                )
            ],
            wave_id="proper_visomaster_wave_test",
            source_logs={},
        )
    except ValueError as exc:
        assert "Missing explicit frame_files" in str(exc)
    else:
        raise AssertionError("Expected build_inventory_snapshot to fail without explicit frame_files")


def test_default_output_paths_follow_runtime_artifact_names():
    paths = tool._default_output_paths("proper_visomaster_wave_2026_04_19_provisional")

    assert paths["inventory"].endswith(
        "arena/inventories/proper_visomaster_wave_2026_04_19_provisional.yaml"
    )
    assert paths["manifest"].endswith(
        "arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json"
    )
    assert paths["suite"].endswith(
        "arena/target_domain_suites.proper_data_future.provisional_2026-04-19.yaml"
    )
    assert paths["report"].endswith(
        "arena/reports/proper_visomaster_wave_2026_04_19_provisional_build_report.json"
    )
