"""Tier-2 smoke for R13_SLOT1_HEAD_RETRAIN_2026-05-13 wiring.

Exercises the same code path ``train_sweep.py`` follows for a head-only
retrain on Slot 1's LoRA-trained checkpoint:

  1. Build ``EffortDetector`` from the head-retrain yaml (SVD on, no ckpt yet).
  2. Pre-install LoRA modules at the configured layers (mirrors train_sweep's
     `_head_only_enabled` pre-load LoRA block).
  3. Optionally load Slot 1's ckpt and verify the trained `lora_A`/`lora_B`
     weights populate (non-zero `lora_B`).
  4. Re-initialise the head (Kaiming uniform; matches train_sweep's
     `head_only_retrain.reinit_head: true` path).
  5. Freeze every non-head param.
  6. Verify:
     a. Only `head.weight` and `head.bias` have `requires_grad=True`.
     b. Trainable-fraction < 0.01% of total params.
     c. ``choose_optimizer`` builds an optimizer with ONLY head params in the
        active param groups (the SVD / backbone-native / LoRA groups have
        either zero params or are absent since their members have
        `requires_grad=False`).
     d. A forward pass on a random tensor produces a finite output.
     e. A backward pass produces gradients ONLY on head params.

The Slot-1 ckpt load is best-effort: if the ckpt isn't on disk locally
(it's a 941MB GCS object), the smoke runs without it — that path still
verifies the LoRA-pre-install + freeze + optimizer-rebuild logic. To run
WITH the ckpt:

    gsutil cp gs://training-job-outputs/best_checkpoints/gf6l06rf/top_n_effort_20260513_step2000_auc0.9929_eer0.0229.pth /tmp/slot1_step2000.pth
    SLOT1_CKPT=/tmp/slot1_step2000.pth python scripts/smoke_slot1_head_retrain_2026-05-13.py

Usage:
    python scripts/smoke_slot1_head_retrain_2026-05-13.py
"""

from __future__ import annotations

import logging
import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import yaml

# Wire up logging early so the [head_only_retrain] info line shows up on stdout.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("smoke_slot1_head_retrain")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from detectors.effort_detector import EffortDetector  # noqa: E402
from detectors.lora_adapter import (  # noqa: E402
    DEFAULT_TARGET_MODULES,
    LORA_PARAM_MARKERS,
    apply_lora_to_openclip_visual,
)
from utils.setup import choose_optimizer  # noqa: E402


YAML_PATH = os.path.join(
    REPO_ROOT,
    "experiments",
    "phase2_round13",
    "R13_SLOT1_HEAD_RETRAIN_2026-05-13.yaml",
)

# Set SLOT1_CKPT env var to the local path of the Slot 1 ckpt to run the
# full ckpt-load verification path. Otherwise the smoke skips that step.
SLOT1_CKPT = os.environ.get("SLOT1_CKPT", "")


def _build_minimal_config() -> dict:
    """Build the config the EffortDetector ctor expects, drawing the
    head_only_retrain + LoRA blocks from the experiment yaml.
    """
    with open(YAML_PATH, "r") as f:
        exp_cfg = yaml.safe_load(f)

    detector_defaults_path = os.path.join(REPO_ROOT, "config", "detector", "effort.yaml")
    with open(detector_defaults_path, "r") as f:
        detector_defaults = yaml.safe_load(f) or {}

    config: dict = {}
    config.update(detector_defaults)
    config.update(exp_cfg)

    # Mirror the optimizer.adam shape that apply_wandb_optimizer_params produces.
    config.setdefault("optimizer", {}).setdefault("adam", {})
    config["optimizer"]["adam"]["lr"] = float(config.get("learning_rate", 1e-4))
    config["optimizer"]["adam"]["eps"] = float(config.get("optimizer_eps", 1e-8))
    config["optimizer"]["adam"]["weight_decay"] = float(config.get("weight_decay", 0.05))
    config["optimizer"]["adam"].setdefault("lora_lr_mult", 1.0)

    # Point at local CLIP weights so EffortDetector doesn't try to fetch.
    config.setdefault("gcs_assets", {}).setdefault("clip_backbone", {})
    config["gcs_assets"]["clip_backbone"]["local_path"] = os.path.join(
        REPO_ROOT, "weights", "CLIP-ViT-B-16-DataComp.XL-s13B-b90K"
    )

    # Smoke skips ckpt download — flip the trainer-driving flag so the
    # production-style load path isn't taken in this driver (we exercise
    # the ckpt-load path inline below if SLOT1_CKPT is set).
    config["load_base_checkpoint"] = False
    # Disable nice-to-have-but-heavy components for this smoke.
    config.setdefault("multi_axis_grl", {})["enabled"] = False
    config.setdefault("anchor_aware", {})["enabled"] = False
    config.setdefault("correlation_penalty", {})["enabled"] = False
    config.setdefault("face_scale_jitter", {})["enabled"] = False
    config.setdefault("canary_probe", {})["enabled"] = False

    return config


def _build_model(config: dict) -> EffortDetector:
    logger.info("Building EffortDetector (SVD on, no ckpt load via trainer) ...")
    return EffortDetector(config)


def _preinstall_lora(model: EffortDetector, config: dict) -> int:
    """Mirror train_sweep.py's `_head_only_enabled` pre-load LoRA block."""
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
    logger.info(
        f"[head_only_retrain] LoRA pre-installed BEFORE checkpoint load: "
        f"layers={target_layers} rank={lora_rank} alpha={lora_alpha:g} | "
        f"wrapped {n_wrapped} layers"
    )
    return n_wrapped


def _load_slot1_ckpt(model: EffortDetector, ckpt_path: str) -> tuple[int, int, int]:
    """Load Slot 1's checkpoint into the LoRA-equipped model, verifying that
    the trained LoRA A/B weights populate (non-zero ``lora_B``).
    Returns (num_lora_keys_in_ckpt, num_lora_keys_loaded, num_head_keys_loaded).
    """
    logger.info(f"Loading Slot 1 ckpt from {ckpt_path} ...")
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = saved["state_dict"] if isinstance(saved, dict) and "state_dict" in saved else saved

    # Normalise module prefix.
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    lora_ckpt_keys = [k for k in new_state_dict if any(m in k for m in LORA_PARAM_MARKERS)]
    head_ckpt_keys = [k for k in new_state_dict if k.startswith("head.")]
    logger.info(
        f"Ckpt has {len(lora_ckpt_keys)} LoRA keys, {len(head_ckpt_keys)} head keys"
    )

    # Load with strict=False (matches Trainer.load_ckpt). Capture missing/unexpected.
    incompatible = model.load_state_dict(new_state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    # Sanity: the LoRA keys we expect should NOT be in `unexpected` (because
    # LoRA was pre-installed, the model has them). They MAY be in the model
    # but absent from the ckpt → would appear in `missing` (problematic).
    model_lora_names = {
        n for n, _ in model.named_parameters() if any(m in n for m in LORA_PARAM_MARKERS)
    }
    lora_missing = [n for n in model_lora_names if n in missing]
    lora_unexpected = [n for n in unexpected if any(m in n for m in LORA_PARAM_MARKERS)]
    logger.info(
        f"LoRA load: {len(model_lora_names)} model params; "
        f"{len(lora_missing)} missing from ckpt; "
        f"{len(lora_unexpected)} unexpected in ckpt"
    )
    if lora_missing:
        raise RuntimeError(
            f"LoRA params missing from ckpt: {lora_missing[:5]} ... "
            f"(total {len(lora_missing)}) — ckpt does not have trained LoRA weights"
        )
    if lora_unexpected:
        logger.warning(
            f"LoRA keys in ckpt not in model: {lora_unexpected[:5]} ..."
        )

    # Verify the loaded LoRA B weights are NON-ZERO (proof the trained delta
    # was actually loaded; init is all-zero so any non-zero value = ckpt load).
    nz_b_count = 0
    nz_b_total = 0
    for n, p in model.named_parameters():
        if "lora_B" in n:
            nz_b_total += 1
            if p.abs().max().item() > 1e-10:
                nz_b_count += 1
    logger.info(f"LoRA B layers with non-zero weights: {nz_b_count}/{nz_b_total}")
    if nz_b_count < nz_b_total:
        raise RuntimeError(
            f"After ckpt load, only {nz_b_count}/{nz_b_total} LoRA B matrices "
            f"are non-zero — expected ALL to be trained (Slot 1 step2000)"
        )

    return len(lora_ckpt_keys), len(model_lora_names) - len(lora_missing), len(head_ckpt_keys)


def _reinit_head(model: EffortDetector) -> None:
    """Mirror train_sweep.py's head_only_retrain.reinit_head: true path."""
    head_module = getattr(model, "head", None)
    if head_module is None:
        raise RuntimeError("model has no `head` attribute")
    if not isinstance(head_module, nn.Linear):
        raise RuntimeError(
            f"head is {type(head_module).__name__}, not nn.Linear — "
            "smoke only covers the standard Linear head path"
        )
    pre_w = head_module.weight.detach().clone()
    nn.init.kaiming_uniform_(head_module.weight, a=5 ** 0.5)
    if head_module.bias is not None:
        fan_in = head_module.weight.shape[1]
        bound = 1.0 / (fan_in ** 0.5) if fan_in > 0 else 0.0
        nn.init.uniform_(head_module.bias, -bound, bound)
    diff = (head_module.weight - pre_w).abs().max().item()
    logger.info(f"[head_only_retrain] re-initialised head; max delta vs pre-init: {diff:.4f}")
    if diff < 1e-8:
        raise RuntimeError("re-init produced no change in head weights")


def _freeze_all_but_head(model: EffortDetector, head_substrings) -> tuple[int, int]:
    """Mirror train_sweep.py's head_only_retrain freeze block."""
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        keep = any(sub in name for sub in head_substrings)
        param.requires_grad_(keep)
        if keep:
            trainable += param.numel()
    pct = 100.0 * trainable / total if total else 0.0
    logger.info(
        f"[head_only_retrain] froze every non-head param | "
        f"trainable={trainable:,} of {total:,} ({pct:.6f}%) | "
        f"selectors={head_substrings}"
    )
    return trainable, total


def _verify_only_head_trainable(model: EffortDetector) -> list[str]:
    """List every trainable param name — must be ONLY head.weight / head.bias."""
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    logger.info(f"Trainable params ({len(trainable_names)}): {trainable_names}")
    expected = {"head.weight", "head.bias"}
    unexpected = [n for n in trainable_names if n not in expected]
    missing = [n for n in expected if n not in trainable_names]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable params: {unexpected}")
    if missing:
        raise RuntimeError(f"Expected trainable params missing: {missing}")
    return trainable_names


def _verify_optimizer_head_only(model: EffortDetector, config: dict) -> None:
    optimizer = choose_optimizer(model, config)
    logger.info(f"Optimizer has {len(optimizer.param_groups)} param groups")
    total_active = 0
    for g in optimizer.param_groups:
        n_params = sum(p.numel() for p in g["params"])
        logger.info(
            f"  group='{g.get('name', '?')}' lr={g['lr']:g} weight_decay={g['weight_decay']} params={n_params:,}"
        )
        total_active += n_params
    # Total active params should equal trainable params (only head).
    expected_head_total = sum(
        p.numel() for n, p in model.named_parameters() if p.requires_grad
    )
    if total_active != expected_head_total:
        raise RuntimeError(
            f"Optimizer holds {total_active} params; expected {expected_head_total} (head only)"
        )
    logger.info(f"PASS: optimizer covers exactly the {total_active} head params")


def _verify_forward_pass(model: EffortDetector) -> None:
    logger.info("Running a forward pass on a random 224×224 input ...")
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model.backbone(x)
        # The head ingests pooler_output via model.head(...)
        pooled = out["pooler_output"] if isinstance(out, dict) else out
        logits = model.head(pooled)
    assert torch.isfinite(pooled).all(), "Backbone produced non-finite output"
    assert torch.isfinite(logits).all(), "Head produced non-finite output"
    logger.info(
        f"Forward OK: pooler_output shape={tuple(pooled.shape)} "
        f"logits shape={tuple(logits.shape)} all-finite"
    )


def _verify_backward_only_head_grads(model: EffortDetector) -> None:
    """Train mode + one backward; only head params should get gradients."""
    model.train()
    for p in model.parameters():
        p.grad = None
    x = torch.randn(2, 3, 224, 224)
    out = model.backbone(x)
    pooled = out["pooler_output"] if isinstance(out, dict) else out
    logits = model.head(pooled)
    loss = logits.sum()
    loss.backward()

    head_has_grad = []
    nonhead_has_grad = []
    for name, param in model.named_parameters():
        nonzero_grad = (param.grad is not None) and (param.grad.abs().sum().item() > 0)
        if nonzero_grad:
            (head_has_grad if name in ("head.weight", "head.bias") else nonhead_has_grad).append(name)
    logger.info(
        f"Backward grads: head={head_has_grad}; non-head with grad={nonhead_has_grad[:5]}"
    )
    if nonhead_has_grad:
        raise RuntimeError(
            f"{len(nonhead_has_grad)} non-head params received gradients — freeze failed"
        )
    if not head_has_grad:
        raise RuntimeError("No head params received gradients — head not actually in loss path")
    logger.info("PASS: ONLY head params received gradients")


def main() -> int:
    config = _build_minimal_config()
    model = _build_model(config)

    # Step 1: pre-install LoRA modules (mirrors train_sweep `_head_only_enabled`
    # pre-load LoRA block).
    _preinstall_lora(model, config)

    # Step 2: (optional) load Slot 1's ckpt if available locally — verifies the
    # trained LoRA weights load correctly with strict=False.
    if SLOT1_CKPT and os.path.exists(SLOT1_CKPT):
        n_ckpt_lora, n_loaded_lora, n_head = _load_slot1_ckpt(model, SLOT1_CKPT)
        if n_ckpt_lora != n_loaded_lora:
            raise RuntimeError(
                f"Loaded {n_loaded_lora}/{n_ckpt_lora} LoRA keys — gap suggests mis-named modules"
            )
        logger.info(f"PASS: all {n_loaded_lora} LoRA keys loaded, {n_head} head keys loaded")
    else:
        logger.info(
            f"SLOT1_CKPT not set or path missing — skipping ckpt-load verification. "
            "(Set SLOT1_CKPT=/path/to/slot1.pth to enable.)"
        )

    # Step 3: re-init head (matches reinit_head: true).
    _reinit_head(model)

    # Step 4: freeze every non-head param.
    head_substrings = list(config["head_only_retrain"]["trainable_param_substrings"])
    trainable, total = _freeze_all_but_head(model, head_substrings)
    pct = 100.0 * trainable / total if total else 0.0
    if pct > 0.01:
        raise RuntimeError(
            f"Trainable-fraction {pct:.6f}% > 0.01% — head-only freeze is loose"
        )
    logger.info(f"PASS: trainable fraction {pct:.6f}% < 0.01%")

    # Step 5: only `head.weight` / `head.bias` are trainable.
    _verify_only_head_trainable(model)

    # Step 6: optimizer holds ONLY head params.
    _verify_optimizer_head_only(model, config)

    # Step 7: forward + backward sanity.
    _verify_forward_pass(model)
    _verify_backward_only_head_grads(model)

    logger.info("ALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
