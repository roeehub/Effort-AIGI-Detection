"""Wiring test for R13_P22_AUG_CURRICULUM.yaml.

Verifies single-lever discipline:
  - anchor_aware            DISABLED
  - face_scale_jitter       DISABLED
  - GRL / quality_domain    DISABLED
  - pipeline_randomization  ENABLED with the P22 curriculum params

Plus the FT base ckpt and contract-relevant settings.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

YAML_PATH = Path("experiments/phase2_round13/R13_P22_AUG_CURRICULUM.yaml")


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(YAML_PATH.read_text())


class TestP22BasicWiring:
    def test_yaml_parses(self, cfg):
        assert cfg["name"] == "R13_P22_AUG_CURRICULUM"
        assert cfg["data_source"] == "combined_paired"

    def test_seed_is_2226(self, cfg):
        assert cfg["seed"] == 2226
        assert cfg["combined_paired"]["seed"] == 2226
        assert cfg["combined_paired"]["split_seed"] == 2226

    def test_ft_from_p8a_step5000(self, cfg):
        assert cfg["load_base_checkpoint"] is True
        ckpt = cfg["gcs_base_checkpoint"]
        assert "9lmvb5b4" in ckpt
        assert "step5000" in ckpt
        assert "auc0.9926" in ckpt

    def test_total_steps_8000(self, cfg):
        assert cfg["total_training_steps"] == 8000
        # Periodic saves cover the full window
        ps = cfg["periodic_saves"]["step_list"]
        assert 500 in ps and 8000 in ps


class TestP22SingleLeverDiscipline:
    """The other anti-shortcut levers MUST be off — single-lever ablation."""

    def test_anchor_aware_disabled(self, cfg):
        assert cfg["anchor_aware"]["enabled"] is False

    def test_face_scale_jitter_disabled(self, cfg):
        assert cfg["face_scale_jitter"]["enabled"] is False

    def test_no_grl_block(self, cfg):
        # P15/P18 used a `grl` block or quality_domain_count toggle. None
        # in P22.
        assert "grl" not in cfg
        assert "quality_domain_count" not in cfg
        # GRL would need quality_domain_head and lambda; ensure none.
        for key in ("quality_domain_head", "grl_lambda", "grl_lambda_schedule"):
            assert key not in cfg, f"{key} must not be set in P22 single-lever config"


class TestP22PipelineRandomizationCurriculum:
    """The single-lever IS pipeline_randomization. Verify exact params."""

    def test_enabled(self, cfg):
        pr = cfg["augmentation"]["pipeline_randomization"]
        assert pr["enabled"] is True

    def test_symmetric_label_gates(self, cfg):
        pr = cfg["augmentation"]["pipeline_randomization"]
        assert pr["p_real"] == 0.5
        assert pr["p_fake"] == 0.5, "symmetric labels: model can't use aug presence as signal"

    def test_blur_curriculum(self, cfg):
        pr = cfg["augmentation"]["pipeline_randomization"]
        assert pr["blur_p"] == 0.7
        sigma_lo, sigma_hi = pr["blur_sigma_range"]
        assert sigma_lo == pytest.approx(0.5)
        assert sigma_hi == pytest.approx(4.0)

    def test_jpeg_curriculum(self, cfg):
        pr = cfg["augmentation"]["pipeline_randomization"]
        assert pr["jpeg_p"] == 0.6
        q_lo, q_hi = pr["jpeg_quality"]
        assert q_lo == 40 and q_hi == 95

    def test_brightness_curriculum(self, cfg):
        pr = cfg["augmentation"]["pipeline_randomization"]
        assert pr["brightness_p"] == 0.5
        b_lo, b_hi = pr["brightness_range"]
        assert b_lo == -40.0 and b_hi == 40.0

    def test_yuv_off(self, cfg):
        pr = cfg["augmentation"]["pipeline_randomization"]
        assert pr["yuv_roundtrip_p"] == 0.0


class TestP22ArchitecturalParityWithP8A:
    """Backbone + head config matches P8A so the lever isolated to aug."""

    def test_backbone_unchanged(self, cfg):
        bb = cfg["backbone"]
        assert bb["name"] == "vit_b_16_laion_datacomp"
        assert bb["unfreeze_final_proj"] is True
        assert bb["unfreeze_final_ln"] is True
        assert bb["apply_svd_to_in_proj"] is True
        assert bb["apply_svd_to_mlp"] is True
        assert bb["hidden_size"] == 512

    def test_arcface_head_consistent(self, cfg):
        assert cfg["use_arcface_head"] is True
        assert cfg["arcface_m"] == 0.15
        assert cfg["s_start"] == 6.0
        assert cfg["s_end"] == 12.0
        assert cfg["anneal_steps"] == 8000  # matches total_training_steps
