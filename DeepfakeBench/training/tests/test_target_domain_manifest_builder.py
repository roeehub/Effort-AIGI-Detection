"""Tests for deterministic Teams target-domain manifest building."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

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


manifest_builder = _load_module(
    "target_domain_manifest_builder_test_module",
    "arena/build_teams_target_domain_manifest.py",
)


def _write_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def test_build_manifest_groups_videos_and_tags_real_slices(tmp_path):
    root = tmp_path / "teams_bucket"

    dark = np.full((32, 32, 3), 10, dtype=np.uint8)
    bright_checker = np.zeros((32, 32, 3), dtype=np.uint8)
    bright_checker[::2, ::2] = 255
    bright_checker[1::2, 1::2] = 255

    _write_image(
        root / "real" / "Cam_Test__s32_0.0_frame_000001_crop_000__a.jpg",
        dark,
    )
    _write_image(
        root / "real" / "Cam_Test__s32_0.0_frame_000002_crop_000__b.jpg",
        dark,
    )
    _write_image(
        root / "real" / "Cam_Test__s32_1.0_frame_000010_crop_000__c.jpg",
        bright_checker,
    )
    _write_image(
        root / "real" / "Cam_Test__s32_1.0_frame_000011_crop_000__d.jpg",
        bright_checker,
    )
    _write_image(
        root / "fake" / "visomaster_enhanced_raw__frame_000001_seq1001.png",
        bright_checker,
    )
    _write_image(
        root / "fake" / "Cam_Test__s33_10.0_frame_000100_crop_000__x.jpg",
        bright_checker,
    )

    manifest = manifest_builder.build_manifest(
        source_root=str(root),
        lockbox_ratio=0.5,
        split_seed=123,
        stats_frames_per_video=1,
    )

    videos = manifest["videos"]
    assert manifest["summary"]["videos_total"] == 4
    assert manifest["summary"]["label_counts"] == {"fake": 2, "real": 2}
    assert manifest["summary"]["slice_counts"]["teams_real_all"] == 2
    assert "teams_real_poor_quality" in manifest["summary"]["slice_counts"]
    assert "teams_real_lighting_extreme" in manifest["summary"]["slice_counts"]

    real_rows = [row for row in videos if row["label"] == "real"]
    assert len(real_rows) == 2
    assert {row["method"] for row in real_rows} == {"teams_real"}
    assert len({row["split"] for row in real_rows}) == 1

    dark_row = next(row for row in real_rows if row["segment_id"] == "0.0")
    assert "teams_real_all" in dark_row["slices"]
    assert "teams_real_poor_quality" in dark_row["slices"]
    assert "teams_real_lighting_extreme" in dark_row["slices"]

    viso_row = next(
        row for row in videos
        if row["label"] == "fake" and row["prefix"] == "visomaster_enhanced_raw"
    )
    assert viso_row["method"] == "visomaster_enhanced_macro"
    assert "teams_fake_all" in viso_row["slices"]
    assert "visomaster_enhanced_macro" in viso_row["slices"]

    manifest_repeat = manifest_builder.build_manifest(
        source_root=str(root),
        lockbox_ratio=0.5,
        split_seed=123,
        stats_frames_per_video=1,
    )
    repeat_splits = {row["video_id"]: row["split"] for row in manifest_repeat["videos"]}
    assert repeat_splits == {row["video_id"]: row["split"] for row in videos}


def test_build_manifest_applies_prefix_rules(tmp_path):
    root = tmp_path / "teams_bucket"
    img = np.full((24, 24, 3), 128, dtype=np.uint8)

    _write_image(
        root / "real" / "Test_Cam__s41_100.0_frame_000001_crop_000__a.jpg",
        img,
    )
    _write_image(
        root / "fake" / "Cam_Test__s33_10.0_frame_000100_crop_000__x.jpg",
        img,
    )

    rules_path = tmp_path / "prefix_rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "fake": {
                    "Cam_Test__s33": {
                        "method": "deeplive_regular",
                        "slices": ["deeplive_regular"],
                    }
                }
            }
        )
    )

    manifest = manifest_builder.build_manifest(
        source_root=str(root),
        lockbox_ratio=0.5,
        split_seed=123,
        stats_frames_per_video=1,
        prefix_rules_path=str(rules_path),
    )

    fake_row = next(row for row in manifest["videos"] if row["label"] == "fake")
    assert fake_row["method"] == "deeplive_regular"
    assert fake_row["matched_rule"] == "Cam_Test__s33"
    assert "teams_fake_all" in fake_row["slices"]
    assert "deeplive_regular" in fake_row["slices"]
