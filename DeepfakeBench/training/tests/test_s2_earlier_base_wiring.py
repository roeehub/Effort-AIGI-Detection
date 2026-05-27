"""Wiring test for R13_S2_P22_EARLIER_BASE.yaml.

S2 = same as S1 but FT from P8A_step2500 (less-saturated base) instead of step5000.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

YAML_PATH = Path("experiments/phase2_round13/R13_S2_P22_EARLIER_BASE.yaml")
S1_YAML_PATH = Path("experiments/phase2_round13/R13_S1_P22_REDUX_SHORT.yaml")


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(YAML_PATH.read_text())


@pytest.fixture(scope="module")
def s1_cfg() -> dict:
    return yaml.safe_load(S1_YAML_PATH.read_text())


class TestS2BasicWiring:
    def test_yaml_parses(self, cfg):
        assert cfg["name"] == "R13_S2_P22_EARLIER_BASE"
        assert cfg["data_source"] == "combined_paired"

    def test_seed_is_2228(self, cfg):
        # Distinct from P22=2226, S1=2227
        assert cfg["seed"] == 2228
        assert cfg["combined_paired"]["seed"] == 2228
        assert cfg["combined_paired"]["split_seed"] == 2228


class TestS2BaseCheckpoint:
    """The ONLY structural difference from S1 is the base ckpt."""

    def test_ft_from_p8a_step2500(self, cfg):
        # CRITICAL: this is the lever isolated by S2 vs S1
        assert cfg["load_base_checkpoint"] is True
        ckpt = cfg["gcs_base_checkpoint"]
        assert "9lmvb5b4" in ckpt
        assert "step2500" in ckpt
        assert "auc0.9936" in ckpt
        assert "step5000" not in ckpt

    def test_base_is_real_in_gcs(self, cfg):
        # Sanity: the chosen ckpt path matches one that actually exists in GCS.
        # At step 2500, only `top_n_effort_*` and `ood_composite_effort_*`
        # variants exist (no `value_composite_*` at this step).
        assert cfg["gcs_base_checkpoint"] == (
            "gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/"
            "top_n_effort_20260424_step2500_auc0.9936_eer0.0270.pth"
        )

    def test_base_uses_top_n_or_ood_prefix(self, cfg):
        # The corrective constraint after S2's first launch failed: only the
        # top_n_effort and ood_composite_effort variants exist at step 2500.
        ckpt = cfg["gcs_base_checkpoint"]
        assert ("top_n_effort" in ckpt) or ("ood_composite_effort" in ckpt), \
            f"step 2500 has no value_composite ckpt; must use top_n_effort or ood_composite_effort. Got: {ckpt}"


class TestS2MatchesS1Otherwise:
    """Every other lever must match S1 exactly so the comparison is
    base-vs-base (single-lever discipline applied across packets)."""

    def test_total_steps_matches_s1(self, cfg, s1_cfg):
        assert cfg["total_training_steps"] == s1_cfg["total_training_steps"] == 1000

    def test_arcface_s_matches_s1(self, cfg, s1_cfg):
        assert cfg["s_start"] == s1_cfg["s_start"] == 6.0
        assert cfg["s_end"] == s1_cfg["s_end"] == 8.0
        assert cfg["anneal_steps"] == s1_cfg["anneal_steps"] == 1000

    def test_periodic_saves_matches_s1(self, cfg, s1_cfg):
        assert cfg["periodic_saves"]["step_list"] == s1_cfg["periodic_saves"]["step_list"]

    def test_pipeline_randomization_matches_s1(self, cfg, s1_cfg):
        pr_s1 = s1_cfg["augmentation"]["pipeline_randomization"]
        pr_s2 = cfg["augmentation"]["pipeline_randomization"]
        for key in ["enabled", "p_real", "p_fake", "blur_p", "blur_sigma_range",
                    "jpeg_p", "jpeg_quality", "brightness_p", "brightness_range",
                    "yuv_roundtrip_p", "downscale_p", "chroma_blur_p", "gamma_p"]:
            assert pr_s2[key] == pr_s1[key], f"S2 differs from S1 on {key}"

    def test_single_lever_matches_s1(self, cfg, s1_cfg):
        assert cfg["anchor_aware"]["enabled"] == s1_cfg["anchor_aware"]["enabled"] == False
        assert cfg["face_scale_jitter"]["enabled"] == s1_cfg["face_scale_jitter"]["enabled"] == False

    def test_lr_matches_s1(self, cfg, s1_cfg):
        assert cfg["learning_rate"] == s1_cfg["learning_rate"]
        assert cfg["weight_decay"] == s1_cfg["weight_decay"]
        assert cfg["lr_scheduler_warmup_steps"] == s1_cfg["lr_scheduler_warmup_steps"]

    def test_eval_freq_matches_s1(self, cfg, s1_cfg):
        assert cfg["evaluate_every_steps"] == s1_cfg["evaluate_every_steps"] == 100


class TestS2NoGRLOrAnchor:
    """Belt-and-suspenders: explicit single-lever check."""

    def test_no_grl(self, cfg):
        assert "grl" not in cfg
        for key in ("quality_domain_head", "grl_lambda", "grl_lambda_schedule",
                    "quality_domain_count"):
            assert key not in cfg
