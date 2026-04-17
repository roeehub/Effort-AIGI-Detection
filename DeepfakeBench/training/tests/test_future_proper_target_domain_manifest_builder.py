"""Tests for future proper-data manifest/schema artifacts."""

from __future__ import annotations

import base64
import importlib.util
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


builder = _load_module(
    "future_proper_target_domain_manifest_builder_test_module",
    "arena/build_future_proper_target_domain_manifest.py",
)

_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7ZL5kAAAAASUVORK5CYII="
)


def _write_image(path: Path, color: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    del color
    path.write_bytes(_TINY_PNG)


def test_build_manifest_emits_canonical_lanes_rollups_and_contract(tmp_path):
    wave_root = tmp_path / "wave"
    layouts = {
        "proper_real_clean": [32, 33],
        "proper_real_teams": [34, 35],
        "proper_visomaster_clean": [36, 37],
        "proper_visomaster_enhanced_clean": [38, 39],
        "proper_visomaster_teams": [40, 41],
        "proper_visomaster_enhanced_teams": [42, 43],
    }
    for lane, colors in layouts.items():
        lane_root = wave_root / lane
        for index, color in enumerate(colors):
            _write_image(lane_root / f"frame_{index:04d}.jpg", color=color)

    inventory = {
        "inventory_version": 1,
        "wave_id": "proper_visomaster_wave_test",
        "split_seed": 737,
        "lockbox_ratio": 0.20,
        "source_logs": {"playlist_uri": "gs://bucket/proper_wave/playlist.json"},
        "captures": [
            {
                "base_capture_id": "roy_d_bigface_0001",
                "identity_id": "roy_d",
                "capture_session_id": "clean_cam_s01",
                "split_group_id": "roy_d__clean_cam_s01",
                "quality_band": "medium",
                "face_scale_band": "big_face",
                "variants": [
                    {
                        "variant_id": "roy_d_bigface_0001__real_clean",
                        "label": "real",
                        "transport": "clean",
                        "enhancement": "none",
                        "playback_path": "direct_capture",
                        "frame_root": str(wave_root / "proper_real_clean"),
                    },
                    {
                        "variant_id": "roy_d_bigface_0001__real_teams",
                        "label": "real",
                        "transport": "teams",
                        "enhancement": "none",
                        "playback_path": "obs_virtual_cam_to_teams",
                        "frame_root": str(wave_root / "proper_real_teams"),
                    },
                    {
                        "variant_id": "roy_d_bigface_0001__ghostface_v2_clean",
                        "label": "fake",
                        "transport": "clean",
                        "generator_family": "visomaster",
                        "generator_method": "GhostFace-v2",
                        "enhancement": "none",
                        "playback_path": "direct_capture",
                        "frame_root": str(wave_root / "proper_visomaster_clean"),
                    },
                    {
                        "variant_id": "roy_d_bigface_0001__ghostface_v2_enhanced_clean",
                        "label": "fake",
                        "transport": "clean",
                        "generator_family": "visomaster",
                        "generator_method": "GhostFace-v2",
                        "enhancement": "enhanced",
                        "playback_path": "direct_capture",
                        "frame_root": str(wave_root / "proper_visomaster_enhanced_clean"),
                    },
                    {
                        "variant_id": "roy_d_bigface_0001__ghostface_v2_teams",
                        "label": "fake",
                        "transport": "teams",
                        "generator_family": "visomaster",
                        "generator_method": "GhostFace-v2",
                        "enhancement": "none",
                        "playback_path": "obs_virtual_cam_to_teams",
                        "frame_root": str(wave_root / "proper_visomaster_teams"),
                    },
                    {
                        "variant_id": "roy_d_bigface_0001__ghostface_v2_enhanced_teams",
                        "label": "fake",
                        "transport": "teams",
                        "generator_family": "visomaster",
                        "generator_method": "GhostFace-v2",
                        "enhancement": "enhanced",
                        "playback_path": "obs_virtual_cam_to_teams",
                        "frame_root": str(wave_root / "proper_visomaster_enhanced_teams"),
                    },
                ],
            }
        ],
    }

    manifest = builder.build_manifest(inventory, source_inventory_path="inventory.yaml")

    assert manifest["schema_name"] == "future_proper_target_domain_v1"
    assert manifest["summary"]["videos_total"] == 6
    assert manifest["summary"]["lane_counts"] == {
        "proper_real_clean": 1,
        "proper_real_teams": 1,
        "proper_visomaster_clean": 1,
        "proper_visomaster_enhanced_clean": 1,
        "proper_visomaster_enhanced_teams": 1,
        "proper_visomaster_teams": 1,
    }

    rows = manifest["videos"]
    assert len({row["split"] for row in rows}) == 1

    real_teams = next(row for row in rows if row["lane"] == "proper_real_teams")
    assert real_teams["method"] == "proper_real_teams"
    assert "proper_real_all" in real_teams["slices"]
    assert "proper_quality_medium" in real_teams["slices"]
    assert "proper_face_scale_big_face" in real_teams["slices"]

    enhanced_teams = next(
        row for row in rows if row["lane"] == "proper_visomaster_enhanced_teams"
    )
    assert enhanced_teams["method"] == "proper_visomaster_enhanced_teams__ghostface_v2"
    assert "proper_fake_teams_all" in enhanced_teams["slices"]
    assert "proper_visomaster_enhanced_teams__ghostface_v2" in enhanced_teams["slices"]
    assert enhanced_teams["generator_method"] == "GhostFace-v2"

    assert manifest["recommended_contract"] == {
        "dev_real_suite": "proper_real_teams_dev",
        "dev_real_stress_suites": [],
        "dev_fake_suites": [
            "proper_fake_teams_all_dev",
            "proper_visomaster_teams_dev",
            "proper_visomaster_enhanced_teams_dev",
        ],
        "lockbox_real_suite": "proper_real_teams_lockbox",
        "lockbox_fake_suite": "proper_fake_teams_all_lockbox",
    }


def test_build_manifest_rejects_fake_variant_without_method(tmp_path):
    frame_path = tmp_path / "frames" / "frame_0000.jpg"
    _write_image(frame_path, color=64)

    inventory = {
        "inventory_version": 1,
        "wave_id": "broken_wave",
        "captures": [
            {
                "base_capture_id": "broken_0001",
                "identity_id": "roy_d",
                "capture_session_id": "clean_cam_s01",
                "split_group_id": "roy_d__clean_cam_s01",
                "quality_band": "medium",
                "face_scale_band": "big_face",
                "variants": [
                    {
                        "variant_id": "broken_0001__fake",
                        "label": "fake",
                        "transport": "teams",
                        "generator_family": "visomaster",
                        "enhancement": "none",
                        "playback_path": "obs_virtual_cam_to_teams",
                        "frame_paths": [str(frame_path)],
                    }
                ],
            }
        ],
    }

    try:
        builder.build_manifest(inventory)
    except ValueError as exc:
        assert "generator_method" in str(exc)
    else:
        raise AssertionError("Expected missing generator_method to raise ValueError")


def test_suite_template_pins_future_proper_contract_names():
    text = (
        ROOT / "arena" / "target_domain_suites.proper_data_future.template.yaml"
    ).read_text()
    assert "name: proper_real_teams_lockbox" in text
    assert 'external_fake_manifest_slices: "proper_fake_teams_all"' in text
    assert "name: proper_visomaster_enhanced_teams_dev" in text
