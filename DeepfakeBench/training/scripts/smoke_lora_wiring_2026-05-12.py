"""Tier-2 smoke for LoRA-on-layers-10-11 wiring.

Exercises the same code path ``train_sweep.py`` follows after checkpoint
load (without actually loading P8A or running data loaders): build an
``EffortDetector`` from the R13_LORA yaml, apply LoRA + freeze base,
rebuild the optimizer, and verify:

  1. ``[LoRA] enabled at layers ...`` log line fires.
  2. Trainable encoder param fraction is <5%.
  3. Forward pass on a random tensor produces a finite output.
  4. ``optim.Adam.param_groups`` contains a ``lora`` group with the
     expected LR (= ``learning_rate × lora_lr_mult``).

This does NOT exercise the actual training step or the data loader; that
requires GCS auth + dataset download. The unit tests in
``tests/test_lora_adapter.py`` cover the LoRA math itself.

Usage:
    python scripts/smoke_lora_wiring_2026-05-12.py
"""

from __future__ import annotations

import logging
import os
import sys

import torch
import yaml

# Wire up logging early so the [LoRA] info line shows up on stdout.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("smoke_lora")

# Repo root on sys.path so `detectors`/`utils` imports resolve.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from detectors.effort_detector import EffortDetector  # noqa: E402
from detectors.lora_adapter import (  # noqa: E402
    DEFAULT_TARGET_MODULES,
    apply_lora_to_openclip_visual,
    count_lora_parameters,
    freeze_base_clip_encoder,
)
from utils.setup import choose_optimizer  # noqa: E402


YAML_PATH = os.path.join(
    REPO_ROOT, "experiments", "phase2_round13", "R13_LORA_L10_L11_2026-05-13.yaml"
)


def _build_minimal_config() -> dict:
    """Build the config the EffortDetector ctor expects, drawing the LoRA
    block from the experiment yaml and filling in the optimizer/backbone
    skeleton from the detector-level defaults.
    """
    with open(YAML_PATH, "r") as f:
        exp_cfg = yaml.safe_load(f)

    # Merge with detector defaults (mirrors what load_base_configs does in
    # the real path; for smoke we only need a minimal subset).
    detector_defaults_path = os.path.join(REPO_ROOT, "config", "detector", "effort.yaml")
    with open(detector_defaults_path, "r") as f:
        detector_defaults = yaml.safe_load(f) or {}

    config: dict = {}
    config.update(detector_defaults)
    config.update(exp_cfg)

    # Make sure the optimizer block has the lr / eps / weight_decay top-level
    # mirrored into the nested optimizer.adam slots (apply_wandb_optimizer_params
    # normally does this; we do it manually for the smoke).
    config.setdefault("optimizer", {}).setdefault("adam", {})
    config["optimizer"]["adam"]["lr"] = float(config.get("learning_rate", 1e-5))
    config["optimizer"]["adam"]["eps"] = float(config.get("optimizer_eps", 1e-8))
    config["optimizer"]["adam"]["weight_decay"] = float(config.get("weight_decay", 0.05))
    # lora_lr_mult is set in the yaml's nested optimizer.adam block; it should
    # have survived the dict.update above. Default to 1.0 if not present.
    config["optimizer"]["adam"].setdefault("lora_lr_mult", 1.0)

    # Point at local CLIP weights so EffortDetector doesn't try to fetch.
    config.setdefault("gcs_assets", {}).setdefault("clip_backbone", {})
    config["gcs_assets"]["clip_backbone"]["local_path"] = os.path.join(
        REPO_ROOT, "weights", "CLIP-ViT-B-16-DataComp.XL-s13B-b90K"
    )

    # Disable any nice-to-have-but-heavy components for this smoke.
    config["load_base_checkpoint"] = False
    config.setdefault("multi_axis_grl", {})["enabled"] = False
    config.setdefault("anchor_aware", {})["enabled"] = False
    config.setdefault("correlation_penalty", {})["enabled"] = False
    config.setdefault("face_scale_jitter", {})["enabled"] = False
    config.setdefault("canary_probe", {})["enabled"] = False

    return config


def _build_model(config: dict) -> EffortDetector:
    logger.info("Building EffortDetector (SVD on, no checkpoint load) ...")
    return EffortDetector(config)


def _apply_lora(model: EffortDetector, config: dict) -> tuple[int, int]:
    lora_cfg = config["lora"]
    visual = model.backbone.visual

    target_layers = list(lora_cfg["target_layers"])
    lora_rank = int(lora_cfg["rank"])
    lora_alpha = float(lora_cfg["alpha"])
    target_modules = tuple(lora_cfg.get("target_modules", DEFAULT_TARGET_MODULES))

    n_wrapped = apply_lora_to_openclip_visual(
        visual,
        target_layers=target_layers,
        rank=lora_rank,
        alpha=lora_alpha,
        target_modules=target_modules,
    )
    trainable, total = freeze_base_clip_encoder(visual)
    pct = 100.0 * trainable / total if total > 0 else 0.0
    logger.info(
        f"[LoRA] enabled at layers {target_layers} | rank={lora_rank} alpha={lora_alpha:g} | "
        f"target_modules={list(target_modules)} | wrapped {n_wrapped} layers | "
        f"trainable encoder params: {trainable:,} of {total:,} ({pct:.2f}%) | "
        f"freeze_base=True"
    )
    return trainable, total


def _verify_optimizer_has_lora_group(model, config: dict) -> None:
    optimizer = choose_optimizer(model, config)
    group_names = [g.get("name", "?") for g in optimizer.param_groups]
    logger.info(f"Optimizer param groups: {group_names}")
    assert "lora" in group_names, f"Expected `lora` param group, got {group_names}"

    lora_group = next(g for g in optimizer.param_groups if g.get("name") == "lora")
    base_lr = float(config["optimizer"]["adam"]["lr"])
    lora_lr_mult = float(config["optimizer"]["adam"].get("lora_lr_mult", 1.0))
    expected_lora_lr = base_lr * lora_lr_mult
    actual_lr = lora_group["lr"]
    assert abs(actual_lr - expected_lora_lr) < 1e-12, (
        f"Expected LoRA LR = {expected_lora_lr:g}, got {actual_lr:g}"
    )
    assert lora_group["weight_decay"] == 0.0, (
        f"Expected LoRA weight_decay=0.0, got {lora_group['weight_decay']}"
    )
    n_lora_params = sum(p.numel() for p in lora_group["params"])
    logger.info(
        f"`lora` group OK: lr={actual_lr:g} weight_decay=0.0 "
        f"params={n_lora_params:,}"
    )


def _verify_forward_pass(model) -> None:
    logger.info("Running a forward pass on a random 224×224 input ...")
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model.backbone(x)
    assert isinstance(out, dict) and "pooler_output" in out, (
        f"Expected dict with `pooler_output`, got {type(out).__name__}"
    )
    pooled = out["pooler_output"]
    assert torch.isfinite(pooled).all(), "Forward produced non-finite values"
    logger.info(f"Forward OK: pooler_output shape={tuple(pooled.shape)} all-finite")


def main() -> int:
    config = _build_minimal_config()
    model = _build_model(config)

    lora_cfg = config.get("lora", {})
    if not lora_cfg.get("enabled", False):
        logger.error("Smoke expected lora.enabled=true in the yaml")
        return 1

    trainable, total = _apply_lora(model, config)

    # Tier-2 smoke checks per task spec §3.5:
    pct = 100.0 * trainable / total if total > 0 else 0.0
    assert pct < 5.0, (
        f"Trainable encoder param fraction {pct:.2f}% exceeds 5% threshold"
    )
    logger.info(f"PASS: trainable fraction {pct:.2f}% < 5%")

    _verify_forward_pass(model)
    _verify_optimizer_has_lora_group(model, config)

    logger.info("ALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
