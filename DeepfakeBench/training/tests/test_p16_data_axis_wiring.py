"""Tests for R13_P16_DATA_AXIS yaml wiring + §10.5 split-audit gate.

The P16 packet is the data-axis lever (Move B1 in HANDOFF 2026-04-30).
Structurally it inherits R13_P14_DATA_FIX with three deltas:

    1. visomaster_enhanced_fake family_weight  8.0 → 2.0 (conservative; DATA_FIX
       collapsed at 8.0).
    2. SINGLE-LEVER DISCIPLINE: face_scale_jitter / anchor_aware /
       pipeline_randomization all DISABLED (per anti_shortcut_bundle_decomposition).
    3. seed 737 → 4422.

These tests lock in those deltas plus the §10.5 split-audit gate semantics:

    test_yaml_*           — parse the yaml, assert the three deltas hold
    test_yaml_disables_bad_data_lanes — visomaster_hints + _hints_teams stay OFF
                            (memory project_visomaster_hints_lanes_bad_data.md)
    test_audit_*          — synthetic-manifest unit tests for the audit checks
                            in analysis/p16_split_audit_2026-04-30/p16_split_audit.py
    test_audit_integration_real_manifests — end-to-end audit on cached real
                            manifests; SKIPPED if resolver cache absent.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

YAML_PATH = ROOT / "experiments" / "phase2_round13" / "R13_P16_DATA_AXIS.yaml"
AUDIT_PATH = ROOT / "analysis" / "p16_split_audit_2026-04-30" / "p16_split_audit.py"
RESOLVER_CACHE = (
    ROOT / "analysis" / "p16_split_audit_2026-04-30" / "_cache"
    / "enhanced_visomaster_resolver_2026-04-06.json"
)
EVAL_MANIFEST = ROOT / "arena" / "manifests" / "teams_target_domain_manifest_2026-04-06_frozen.json"


def _load_yaml() -> dict:
    yaml = pytest.importorskip("yaml")
    with open(YAML_PATH, "r") as f:
        return yaml.safe_load(f)


def _load_audit_module():
    """Load the audit script as a module. We register in sys.modules first so
    ``@dataclass`` can resolve the class's module via cls.__module__."""
    if "p16_split_audit" in sys.modules:
        return sys.modules["p16_split_audit"]
    spec = importlib.util.spec_from_file_location("p16_split_audit", AUDIT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["p16_split_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


# -----------------------------------------------------------------------------
# Yaml structural tests.
# -----------------------------------------------------------------------------
def test_yaml_exists_and_loads():
    cfg = _load_yaml()
    assert cfg["name"] == "R13_P16_DATA_AXIS"
    assert cfg["seed"] == 4422
    assert cfg["data_source"] == "combined_paired"


def test_yaml_visomaster_enhanced_fake_weight_is_conservative():
    """fw=2.0 (DATA_FIX collapsed at 8.0)."""
    cfg = _load_yaml()
    fws = cfg["combined_paired"]["sampling"]["family_weights"]
    assert fws["visomaster_enhanced_fake"] == 2.0, (
        f"Expected fw=2.0, got {fws['visomaster_enhanced_fake']}; DATA_FIX collapse rationale broken."
    )
    # Hard upper bound — never go above 4.0.
    assert fws["visomaster_enhanced_fake"] <= 4.0


def test_yaml_single_lever_discipline():
    """anti-shortcut bundle DISABLED: face_scale_jitter / anchor_aware / pipeline_random."""
    cfg = _load_yaml()
    assert cfg["face_scale_jitter"]["enabled"] is False
    assert cfg["anchor_aware"]["enabled"] is False
    assert cfg["augmentation"]["pipeline_randomization"]["enabled"] is False


def test_yaml_visomaster_teams_enhanced_enabled():
    """The data-axis lever block."""
    cfg = _load_yaml()
    block = cfg["combined_paired"]["visomaster_teams_enhanced"]
    assert block["enabled"] is True
    assert block["companion_domains"] == ["teams_v2"]
    assert "teams_v2_companion" in block["include_statuses"]
    assert block["sampling_family_key"] == "visomaster_enhanced_fake"
    # Resolver manifest URI must be the canonical 2026-04-06 one.
    assert "enhanced_visomaster_resolver_2026-04-06.json" in block["resolver_manifest_uri"]


def test_yaml_disables_bad_data_lanes():
    """memory project_visomaster_hints_lanes_bad_data.md — hints lanes are bad data."""
    cfg = _load_yaml()
    cp = cfg["combined_paired"]
    # visomaster_hints is bad data; if the key is missing that's also fine (stays off by default).
    if "visomaster_hints" in cp:
        assert cp["visomaster_hints"].get("enabled", False) is False
    if "visomaster_hints_teams" in cp:
        assert cp["visomaster_hints_teams"].get("enabled", False) is False
    # visomaster_enhanced (broad enhancer-pass, no Teams transport) explicitly DISABLED for P16.
    assert cp["visomaster_enhanced"]["enabled"] is False


def test_yaml_ft_init_points_at_p8a_step5000():
    cfg = _load_yaml()
    assert cfg["load_base_checkpoint"] is True
    ckpt = cfg["gcs_base_checkpoint"]
    assert "9lmvb5b4" in ckpt
    assert "step5000" in ckpt


def test_yaml_p8a_unfreeze_recipe_preserved():
    """SVD + visual.proj + ln_post + MLP-SVD trainable surface (P8A recipe)."""
    cfg = _load_yaml()
    bb = cfg["backbone"]
    assert bb["apply_svd_to_in_proj"] is True
    assert bb["unfreeze_final_proj"] is True
    assert bb["unfreeze_final_ln"] is True
    assert bb["apply_svd_to_mlp"] is True


def test_yaml_no_grl_block():
    """P16 is single-lever data-axis; GRL is a separate (orthogonal) packet."""
    cfg = _load_yaml()
    assert cfg.get("use_quality_domain_head", False) is False
    assert cfg.get("quality_domain_head", {}) == {} or cfg.get("quality_domain_head") is None


# -----------------------------------------------------------------------------
# Audit unit tests (synthetic manifests).
# -----------------------------------------------------------------------------
def _make_synth_eval_payload(bucket: str, sequences: list[str]) -> dict:
    return {
        "videos": [
            {
                "frame_paths": [f"gs://{bucket}/fake/visomaster_enhanced_raw__frame_001595_{seq}.png"],
                "identity": 1305929251,
                "identity_key": "visomaster_enhanced_raw",
                "label": "fake",
                "method": "visomaster_enhanced_macro",
                "sequence_id": seq,
                "session_id": None,
                "slices": ["visomaster_enhanced_macro"],
                "split": "dev",
            }
            for seq in sequences
        ]
    }


def _make_synth_resolver_payload(rows: list[dict]) -> dict:
    return {"version": 1, "rows": rows}


def test_audit_bucket_disjointness_passes_when_disjoint():
    audit = _load_audit_module()
    train_buckets = {"enhanced-visomaster-cropped", "live-...-teams-v2"}
    eval_buckets = {"teams-faces-data-test-2914-fake-4420-real-feb-28"}
    r = audit.check_bucket_disjointness(train_buckets, eval_buckets)
    assert r.passed
    assert "overlap=0" in r.detail


def test_audit_bucket_disjointness_fails_on_overlap():
    audit = _load_audit_module()
    train_buckets = {"enhanced-visomaster-cropped", "teams-faces-data-test-2914-fake-4420-real-feb-28"}
    eval_buckets = {"teams-faces-data-test-2914-fake-4420-real-feb-28"}
    r = audit.check_bucket_disjointness(train_buckets, eval_buckets)
    assert not r.passed
    assert "teams-faces-data-test-2914-fake-4420-real-feb-28" in r.overlap_examples


def test_audit_sequence_id_disjointness_passes_when_disjoint():
    audit = _load_audit_module()
    train_sample_ids = {"visomaster_CSCS_00007", "visomaster_GhostFace-v1_00019"}
    eval_seq_ids = {"seq12349", "seq12351"}
    r = audit.check_sequence_id_disjointness(train_sample_ids, eval_seq_ids)
    assert r.passed


def test_audit_sequence_id_disjointness_fails_on_substring_match():
    audit = _load_audit_module()
    # Construct a worst-case where a sample_id contains an eval seq_id substring.
    train_sample_ids = {"visomaster_CSCS_seq12349_repackaged", "good_sample"}
    eval_seq_ids = {"seq12349"}
    r = audit.check_sequence_id_disjointness(train_sample_ids, eval_seq_ids)
    assert not r.passed
    assert any("seq12349" in ex for ex in r.overlap_examples)


def test_audit_frame_path_disjointness_passes_when_disjoint():
    audit = _load_audit_module()
    train_paths = {"gs://enhanced-visomaster-cropped/path/a.png"}
    eval_paths = {"gs://teams-faces-data-test-2914-fake-4420-real-feb-28/fake/x.png"}
    r = audit.check_frame_path_disjointness(train_paths, eval_paths)
    assert r.passed


def test_audit_frame_path_disjointness_fails_on_shared_path():
    audit = _load_audit_module()
    shared = "gs://enhanced-visomaster-cropped/path/shared.png"
    r = audit.check_frame_path_disjointness({shared}, {shared, "other.png"})
    assert not r.passed
    assert shared in r.overlap_examples


# -----------------------------------------------------------------------------
# Integration: real manifests if cached.
# -----------------------------------------------------------------------------
@pytest.mark.skipif(not RESOLVER_CACHE.exists(), reason="resolver cache absent; run gsutil cp first")
@pytest.mark.skipif(not EVAL_MANIFEST.exists(), reason="eval manifest absent")
def test_audit_integration_real_manifests_pass():
    """End-to-end: real eval manifest + real resolver cache must yield PASS verdict.

    If this fails, P16 launch is BLOCKED — there is genuine train/eval overlap
    in the wired sources. Investigate before image rebuild.
    """
    audit = _load_audit_module()
    eval_payload = audit.load_eval_payload()
    eval_videos = audit.filter_eval_videos(eval_payload)
    assert eval_videos, "No eval videos matched visomaster_enhanced_macro/dev"

    eval_seq_ids = audit.extract_eval_sequence_ids(eval_videos)
    eval_buckets = audit.extract_eval_buckets(eval_videos)
    eval_frame_paths = audit.extract_eval_frame_paths(eval_videos)

    resolver_payload = audit.load_resolver_payload()
    assert resolver_payload is not None
    train_sample_ids = audit.extract_train_sample_ids(resolver_payload)
    train_buckets = audit.extract_train_buckets(resolver_payload)
    train_frame_paths = audit.extract_train_frame_paths(resolver_payload)

    r1 = audit.check_bucket_disjointness(train_buckets, eval_buckets)
    r2 = audit.check_sequence_id_disjointness(train_sample_ids, eval_seq_ids)
    r3 = audit.check_frame_path_disjointness(train_frame_paths, eval_frame_paths)

    assert r1.passed, f"BUCKET overlap: {r1.overlap_examples}"
    assert r2.passed, f"SEQUENCE_ID overlap: {r2.overlap_examples}"
    assert r3.passed, f"FRAME_PATH overlap: {r3.overlap_examples}"


def test_audit_run_audit_returns_pass_on_real_cache():
    """If both real manifests are present, the full run_audit() should exit 0."""
    if not RESOLVER_CACHE.exists() or not EVAL_MANIFEST.exists():
        pytest.skip("manifests not cached; integration check skipped")
    audit = _load_audit_module()
    rc = audit.run_audit()
    assert rc == 0, f"run_audit returned {rc}; expected 0 (PASS)"
