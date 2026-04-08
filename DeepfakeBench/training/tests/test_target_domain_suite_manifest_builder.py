"""Tests for Teams target-domain scorecard suite generation."""

from __future__ import annotations

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


suite_builder = _load_module(
    "target_domain_suite_builder_test_module",
    "arena/build_teams_target_domain_suites.py",
)


def test_build_suite_manifest_emits_real_fake_family_and_method_suites():
    manifest = {
        "videos": [
            {
                "label": "real",
                "split": "dev",
                "method": "teams_real",
                "slices": ["teams_real_all", "teams_real_poor_quality"],
            },
            {
                "label": "real",
                "split": "dev",
                "method": "teams_real",
                "slices": ["teams_real_all", "teams_real_lighting_extreme"],
            },
            {
                "label": "real",
                "split": "lockbox",
                "method": "teams_real",
                "slices": ["teams_real_all"],
            },
            {
                "label": "fake",
                "split": "dev",
                "method": "visomaster_enhanced_macro",
                "slices": ["teams_fake_all", "visomaster_enhanced_macro"],
            },
            {
                "label": "fake",
                "split": "dev",
                "method": "teams_capture_cam_test_s33",
                "slices": ["teams_fake_all", "teams_capture_cam_test", "teams_capture_cam_test_s33"],
            },
            {
                "label": "fake",
                "split": "lockbox",
                "method": "teams_capture_cam_test_s33",
                "slices": ["teams_fake_all", "teams_capture_cam_test", "teams_capture_cam_test_s33"],
            },
        ]
    }

    payload = suite_builder.build_suite_manifest(
        manifest=manifest,
        manifest_path="arena/manifests/example_manifest.json",
    )

    suites = payload["suites"]
    suite_names = {suite["name"] for suite in suites}

    assert payload["real_splits"] == ["dev", "lockbox"]
    assert payload["fake_splits"] == ["dev"]

    assert "teams_real_all_dev" in suite_names
    assert "teams_real_poor_quality_dev" in suite_names
    assert "teams_real_lighting_extreme_dev" in suite_names
    assert "teams_real_all_lockbox" in suite_names

    assert "teams_fake_all_dev" in suite_names
    assert "teams_capture_cam_test_dev" in suite_names
    assert "teams_capture_cam_test_s33_dev" in suite_names
    assert "visomaster_enhanced_macro_dev" in suite_names

    assert "teams_fake_all_lockbox" not in suite_names
    assert "teams_capture_cam_test_s33_lockbox" not in suite_names

    family_suite = next(suite for suite in suites if suite["name"] == "teams_capture_cam_test_dev")
    assert family_suite["external_fake_manifest_slices"] == "teams_capture_cam_test"
    assert "external_fake_method" not in family_suite

    method_suite = next(suite for suite in suites if suite["name"] == "teams_capture_cam_test_s33_dev")
    assert method_suite["external_fake_manifest_slices"] == "teams_capture_cam_test_s33"
    assert method_suite["external_fake_method"] == "teams_capture_cam_test_s33"

    viso_suite = next(suite for suite in suites if suite["name"] == "visomaster_enhanced_macro_dev")
    assert viso_suite["external_fake_manifest_slices"] == "visomaster_enhanced_macro"
    assert viso_suite["external_fake_method"] == "visomaster_enhanced_macro"
