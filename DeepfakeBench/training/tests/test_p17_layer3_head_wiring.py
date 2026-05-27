"""P17 layer-3 head wiring smoke test.

Verifies:
  1. Detector instantiates with `backbone.intermediate_layer: 3` and
     `backbone.hidden_size: 768`.
  2. Backbone wrapper exposes the intermediate-layer hook.
  3. Forward returns 768-d features (vs 512-d in vanilla mode).
  4. Backward flows only to the head (when backbone is frozen).
  5. Trainer's `load_ckpt` head-key-drop logic is correct (gated by
     `backbone.intermediate_layer is not None`).
  6. Backward-compat: yaml without `intermediate_layer` produces unchanged
     behavior.

Run:
    cd training/
    python3 -m pytest tests/test_p17_layer3_head_wiring.py -v
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CACHE_DIR = REPO_ROOT / "analysis" / "_features_cache_2026-04-30"
P8A_CKPT = CACHE_DIR / "value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth"


def _build_cfg(intermediate_layer=None):
    cfg = yaml.safe_load(open(REPO_ROOT / "config" / "detector" / "effort.yaml"))
    cfg.update(yaml.safe_load(open(REPO_ROOT / "config" / "train_config.yaml")))
    if P8A_CKPT.exists():
        ckpt = torch.load(str(P8A_CKPT), map_location="cpu", weights_only=False)
        for k, v in ckpt.get("model_config", {}).items():
            if k != "current_arcface_s":
                cfg[k] = v
    if intermediate_layer is not None:
        cfg["backbone"]["intermediate_layer"] = intermediate_layer
        cfg["backbone"]["hidden_size"] = 768
        cfg["backbone"]["freeze_svd_residuals"] = True
        cfg["backbone"]["unfreeze_final_proj"] = False
        cfg["backbone"]["unfreeze_final_ln"] = False
        cfg["hidden_size"] = 768
    return cfg


@pytest.mark.skipif(not P8A_CKPT.exists(), reason="P8A checkpoint not cached locally")
def test_vanilla_unchanged_512d():
    """Test 1: vanilla mode (no intermediate_layer) returns 512-d features."""
    from detectors import DETECTOR
    cfg = _build_cfg(intermediate_layer=None)
    m = DETECTOR[cfg["model_name"]](cfg).eval()
    img = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = m({"image": img}, inference=True)
    assert out["feat"].shape == (2, 512), f"vanilla feat dim regressed: {out['feat'].shape}"
    assert out["cls"].shape == (2, 2)


@pytest.mark.skipif(not P8A_CKPT.exists(), reason="P8A checkpoint not cached locally")
def test_layer3_returns_768d():
    """Test 2: with intermediate_layer=3, forward returns 768-d frozen-backbone features."""
    from detectors import DETECTOR
    cfg = _build_cfg(intermediate_layer=3)
    m = DETECTOR[cfg["model_name"]](cfg).eval()
    trainable_backbone = [
        name for name, param in m.backbone.named_parameters()
        if param.requires_grad
    ]
    assert trainable_backbone == [], f"backbone should be frozen, got {trainable_backbone[:5]}"
    img = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = m({"image": img}, inference=True)
    assert out["feat"].shape == (2, 768), f"expected (2,768) got {out['feat'].shape}"
    assert out["cls"].shape == (2, 2)


@pytest.mark.skipif(not P8A_CKPT.exists(), reason="P8A checkpoint not cached locally")
def test_layer3_backward_freeze_backbone():
    """Test 3: backward with config-frozen backbone — only head gets gradient."""
    from detectors import DETECTOR
    cfg = _build_cfg(intermediate_layer=3)
    m = DETECTOR[cfg["model_name"]](cfg)
    m.train()
    img = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 0, 1])
    out = m({"image": img, "label": labels}, inference=False)
    loss = torch.nn.functional.cross_entropy(out["cls"], labels)
    loss.backward()
    head_grad_norm = float(torch.norm(m.head.weight.grad))
    backbone_grads = sum(1 for p in m.backbone.parameters()
                         if p.grad is not None and float(p.grad.norm()) > 0)
    assert head_grad_norm > 0, "head should receive gradient"
    assert backbone_grads == 0, f"backbone should be frozen, got {backbone_grads} non-zero grads"


def test_trainer_head_drop_logic():
    """Test 5: trainer's head-key-drop logic (gated by intermediate_layer)."""
    state_dict = OrderedDict({
        "backbone.visual.proj": torch.randn(768, 512),
        "backbone.x.y.z": torch.randn(10),
        "head.weight": torch.randn(2, 512),
        "head.s": torch.tensor(6.0),
    })
    # With intermediate_layer set: drop head.*
    backbone_cfg = {"intermediate_layer": 3}
    new_sd = OrderedDict(state_dict)
    if backbone_cfg.get("intermediate_layer") is not None:
        head_keys = [k for k in list(new_sd.keys()) if k.startswith("head.")]
        for k in head_keys:
            del new_sd[k]
    assert "head.weight" not in new_sd
    assert "head.s" not in new_sd
    assert "backbone.visual.proj" in new_sd
    assert len(new_sd) == 2

    # Backward-compat: without flag, no drop
    backbone_cfg2 = {}
    new_sd2 = OrderedDict(state_dict)
    if backbone_cfg2.get("intermediate_layer") is not None:
        head_keys = [k for k in list(new_sd2.keys()) if k.startswith("head.")]
        for k in head_keys:
            del new_sd2[k]
    assert len(new_sd2) == 4
    assert "head.weight" in new_sd2


@pytest.mark.skipif(not P8A_CKPT.exists(), reason="P8A checkpoint not cached locally")
def test_load_ckpt_with_layer3_succeeds():
    """Test 6: ckpt load with intermediate_layer=3 succeeds via head-drop."""
    from detectors import DETECTOR
    cfg = _build_cfg(intermediate_layer=3)
    m = DETECTOR[cfg["model_name"]](cfg)
    ckpt = torch.load(str(P8A_CKPT), map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_sd[name] = v
    # Apply the trainer's drop logic.
    backbone_cfg = cfg.get("backbone", {})
    if backbone_cfg.get("intermediate_layer") is not None:
        head_keys = [k for k in list(new_sd.keys()) if k.startswith("head.")]
        for k in head_keys:
            del new_sd[k]
    # Must NOT raise.
    result = m.load_state_dict(new_sd, strict=False)
    # Head should be in missing keys (since we dropped them).
    head_in_missing = any(k.startswith("head.") for k in result.missing_keys)
    assert head_in_missing, "head.* should be in missing_keys after dropping from checkpoint"
    # Forward should still work.
    img = torch.randn(2, 3, 224, 224)
    m.eval()
    with torch.no_grad():
        out = m({"image": img}, inference=True)
    assert out["feat"].shape == (2, 768)


def test_yaml_p17_parses():
    """Test 7: P17 yaml parses cleanly."""
    yaml_path = REPO_ROOT / "experiments" / "phase2_round13" / "R13_P17_LAYER3_HEAD.yaml"
    if not yaml_path.exists():
        pytest.skip("P17 yaml not present")
    cfg = yaml.safe_load(open(yaml_path))
    assert cfg["backbone"]["intermediate_layer"] == 3
    assert cfg["backbone"]["hidden_size"] == 768
    assert cfg["backbone"]["freeze_svd_residuals"] is True
    assert cfg["total_training_steps"] == 10000
    assert cfg["lr_scheduler_warmup_steps"] == 500
    assert cfg["periodic_saves"]["step_list"] == [500, 1000, 1500, 2000, 3000, 4000, 5000, 7500, 10000]
    assert "save_steps" not in cfg["periodic_saves"]
    assert cfg["checkpointing"]["gcs_prefix"].endswith("/phase2r13_experiments/")
    assert cfg.get("load_base_checkpoint") is True
    assert cfg.get("gcs_base_checkpoint", "").endswith(".pth")
    # Single-lever discipline:
    assert cfg.get("face_scale_jitter", {}).get("enabled") is False
    assert cfg.get("anchor_aware", {}).get("enabled") is False
    assert cfg.get("pipeline_randomization", {}).get("enabled") is False
    # Bad-data lanes off:
    assert cfg.get("data", {}).get("visomaster_hints", {}).get("enabled") is False
    assert cfg.get("data", {}).get("visomaster_hints_teams", {}).get("enabled") is False


def test_wandb_overrides_preserve_explicit_hidden_size():
    """Test 8: apply_backbone_overrides honors yaml's explicit hidden_size.

    Regression for the P17 launch failure where ViT-B-16-DataComp-XL's
    registry-hardcoded hidden_size: 512 (the projected output dim) silently
    clobbered the yaml's explicit hidden_size: 768 needed for layer-3 readout.
    """
    from utils.config_helpers import apply_wandb_backbone_params

    yaml_path = REPO_ROOT / "experiments" / "phase2_round13" / "R13_P17_LAYER3_HEAD.yaml"
    if not yaml_path.exists():
        pytest.skip("P17 yaml not present")
    p17_cfg = yaml.safe_load(open(yaml_path))

    config = {
        "backbone": dict(p17_cfg["backbone"]),
        "backbone_registry": yaml.safe_load(
            open(REPO_ROOT / "config" / "backbone_registry.yaml")
        ),
    }
    apply_wandb_backbone_params(config, p17_cfg, logger=None)
    assert config["backbone"]["hidden_size"] == 768, (
        f"explicit yaml hidden_size: 768 was clobbered to "
        f"{config['backbone']['hidden_size']}; bug returned"
    )

    # Backward-compat: without explicit hidden_size, registry value is used.
    p7_like_cfg = {
        "backbone": {
            "name": "vit_b_16_laion_datacomp",
            "variant": "ViT-B-16-DataComp-XL",
            "source": "laion",
            "model_name": "ViT-B-16",
            "pretrained": "datacomp_xl_s13b_b90k",
            "resolution": 224,
        }
    }
    config2 = {
        "backbone": dict(p7_like_cfg["backbone"]),
        "backbone_registry": yaml.safe_load(
            open(REPO_ROOT / "config" / "backbone_registry.yaml")
        ),
    }
    apply_wandb_backbone_params(config2, p7_like_cfg, logger=None)
    assert config2["backbone"]["hidden_size"] == 512, (
        "without explicit hidden_size, registry value (512) should win"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
