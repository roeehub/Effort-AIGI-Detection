"""Wiring test for R13_S1_P22_REDUX_SHORT.yaml.

S1 = P22 aug curriculum + 1000-step cap + ArcFace s capped at 8.0.
Verifies single-lever discipline and the specific changes vs P22.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

YAML_PATH = Path("experiments/phase2_round13/R13_S1_P22_REDUX_SHORT.yaml")


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(YAML_PATH.read_text())


class TestS1BasicWiring:
    def test_yaml_parses(self, cfg):
        assert cfg["name"] == "R13_S1_P22_REDUX_SHORT"
        assert cfg["data_source"] == "combined_paired"

    def test_seed_is_2227(self, cfg):
        # Distinct from P22=2226
        assert cfg["seed"] == 2227
        assert cfg["combined_paired"]["seed"] == 2227
        assert cfg["combined_paired"]["split_seed"] == 2227

    def test_ft_from_p8a_step5000_same_as_p22(self, cfg):
        # SAME base as P22 — isolates the cap
        assert cfg["load_base_checkpoint"] is True
        ckpt = cfg["gcs_base_checkpoint"]
        assert "9lmvb5b4" in ckpt
        assert "step5000" in ckpt
        assert "auc0.9926" in ckpt


class TestS1ShortTrainingCap:
    def test_total_steps_1000(self, cfg):
        # vs P22's 8000
        assert cfg["total_training_steps"] == 1000

    def test_n_epochs_1(self, cfg):
        # nEpochs caps the outer loop; combined with total_training_steps=1000,
        # the trainer exits after ~1000 steps regardless of dataloader cycling.
        assert cfg["nEpochs"] == 1

    def test_warmup_is_10pct(self, cfg):
        # Maintains ~10% warmup ratio (P22: 400/8000 = 5%; S1: 100/1000 = 10%)
        assert cfg["lr_scheduler_warmup_steps"] == 100

    def test_periodic_saves_fine_grained(self, cfg):
        # Save every 100 steps (vs P22's [500, 1000, 2000, ...])
        ps = cfg["periodic_saves"]["step_list"]
        expected = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        assert ps == expected

    def test_eval_every_100_steps(self, cfg):
        # Must be fine-grained for short runs (P22: 500)
        assert cfg["evaluate_every_steps"] == 100


class TestS1ArcFaceSCap:
    def test_arcface_s_capped_at_8(self, cfg):
        # vs P22's s_end=12.0 (collapse trigger ~s=10.5)
        assert cfg["s_start"] == 6.0
        assert cfg["s_end"] == 8.0

    def test_anneal_matches_total_steps(self, cfg):
        # Anneal completes by end of training
        assert cfg["anneal_steps"] == 1000
        assert cfg["anneal_steps"] == cfg["total_training_steps"]

    def test_arcface_m_unchanged(self, cfg):
        # Same margin as P22 — only s is capped
        assert cfg["arcface_m"] == 0.15


class TestS1SingleLeverDiscipline:
    """Single-lever rule: ONLY pipeline_randomization is the experimental lever."""

    def test_anchor_aware_disabled(self, cfg):
        assert cfg["anchor_aware"]["enabled"] is False

    def test_face_scale_jitter_disabled(self, cfg):
        assert cfg["face_scale_jitter"]["enabled"] is False

    def test_no_grl_block(self, cfg):
        assert "grl" not in cfg
        assert "quality_domain_count" not in cfg
        for key in ("quality_domain_head", "grl_lambda", "grl_lambda_schedule"):
            assert key not in cfg, f"{key} must not be set in S1"


class TestS1PipelineRandomizationIdenticalToP22:
    """The lever IS pipeline_randomization. Curriculum must match P22 exactly so
    the only varied factor is training cap + s_end."""

    def test_enabled_with_symmetric_labels(self, cfg):
        pr = cfg["augmentation"]["pipeline_randomization"]
        assert pr["enabled"] is True
        assert pr["p_real"] == 0.5
        assert pr["p_fake"] == 0.5

    def test_blur_curriculum_matches_p22(self, cfg):
        pr = cfg["augmentation"]["pipeline_randomization"]
        assert pr["blur_p"] == 0.7
        sigma_lo, sigma_hi = pr["blur_sigma_range"]
        assert sigma_lo == pytest.approx(0.5)
        assert sigma_hi == pytest.approx(4.0)

    def test_jpeg_curriculum_matches_p22(self, cfg):
        pr = cfg["augmentation"]["pipeline_randomization"]
        assert pr["jpeg_p"] == 0.6
        q_lo, q_hi = pr["jpeg_quality"]
        assert q_lo == 40 and q_hi == 95

    def test_brightness_curriculum_matches_p22(self, cfg):
        pr = cfg["augmentation"]["pipeline_randomization"]
        assert pr["brightness_p"] == 0.5
        b_lo, b_hi = pr["brightness_range"]
        assert b_lo == -40.0 and b_hi == 40.0

    def test_yuv_off_matches_p22(self, cfg):
        pr = cfg["augmentation"]["pipeline_randomization"]
        assert pr["yuv_roundtrip_p"] == 0.0


class TestS1ArchitecturalParityWithP22:
    """Backbone, optimizer, LR, head all match P22 — only training cap differs."""

    def test_backbone_matches_p22(self, cfg):
        bb = cfg["backbone"]
        assert bb["name"] == "vit_b_16_laion_datacomp"
        assert bb["unfreeze_final_proj"] is True
        assert bb["unfreeze_final_ln"] is True
        assert bb["apply_svd_to_in_proj"] is True
        assert bb["apply_svd_to_mlp"] is True
        assert bb["hidden_size"] == 512

    def test_lr_matches_p22(self, cfg):
        assert cfg["learning_rate"] == 3.0e-5
        assert cfg["weight_decay"] == 0.05
        assert cfg["lr_scheduler"] == "cosine_with_warmup"

    def test_arcface_head_enabled(self, cfg):
        assert cfg["use_arcface_head"] is True
