"""Tests for the frozen Teams target-domain manifest visualizer."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

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


visualizer = _load_module(
    "teams_target_domain_manifest_visualizer_test_module",
    "arena/visualize_teams_target_domain_manifest.py",
)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (48, 48), color)
    image.save(path)


def test_build_gallery_writes_html_cards_and_selection(tmp_path):
    manifest_dir = tmp_path / "manifest_data"
    frames_dir = manifest_dir / "frames"
    output_dir = tmp_path / "gallery"

    real_a = frames_dir / "real_a.jpg"
    real_b = frames_dir / "real_b.jpg"
    fake_a = frames_dir / "fake_a.jpg"
    fake_b = frames_dir / "fake_b.jpg"
    _write_image(real_a, (10, 120, 40))
    _write_image(real_b, (20, 130, 50))
    _write_image(fake_a, (170, 20, 40))
    _write_image(fake_b, (180, 30, 50))

    manifest_path = manifest_dir / "teams_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "summary": {
                    "videos_total": 2,
                    "slice_counts": {
                        "teams_real_poor_quality": 1,
                        "visomaster_enhanced_macro": 1,
                    },
                },
                "videos": [
                    {
                        "label": "real",
                        "method": "teams_real",
                        "video_id": "real_dev_001",
                        "frame_paths": [str(real_a), str(real_b)],
                        "identity": 1,
                        "identity_key": "real_key",
                        "split": "dev",
                        "slices": ["teams_real_all", "teams_real_poor_quality"],
                        "prefix": "Cam_Test",
                        "session_id": "s32",
                        "sequence_id": None,
                        "source_kind": "session_capture",
                        "matched_rule": None,
                    },
                    {
                        "label": "fake",
                        "method": "visomaster_enhanced_macro",
                        "video_id": "fake_dev_001",
                        "frame_paths": [str(fake_a), str(fake_b)],
                        "identity": 2,
                        "identity_key": "fake_key",
                        "split": "dev",
                        "slices": ["teams_fake_all", "visomaster_enhanced_macro"],
                        "prefix": "visomaster_enhanced_raw",
                        "session_id": None,
                        "sequence_id": "seq12345",
                        "source_kind": "flat_upload",
                        "matched_rule": "visomaster_enhanced_raw",
                    },
                ],
            }
        )
    )

    result = visualizer.build_gallery(
        manifest_path=str(manifest_path),
        output_dir=str(output_dir),
        split="dev",
        slices=["teams_real_poor_quality", "visomaster_enhanced_macro"],
        samples_per_slice=1,
        frames_per_video=2,
    )

    html_path = Path(result["html_path"])
    selection_path = Path(result["selection_path"])

    assert html_path.exists()
    assert selection_path.exists()
    assert result["rendered_card_count"] == 2
    assert result["slice_count"] == 2

    html_text = html_path.read_text()
    assert "teams_real_poor_quality" in html_text
    assert "visomaster_enhanced_macro" in html_text

    selection = json.loads(selection_path.read_text())
    assert len(selection["slice_sections"]) == 2
    assert selection["slice_sections"][0]["cards"]
