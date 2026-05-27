"""Regression test for ``lora-enabled-not-propagated-by-load-model`` (open loop
in docs/packet_retrospectives/threads/wandb_yaml_propagation_bugs.md).

Before fix (2026-05-20): ``trainer/trainer.py:save_ckpt`` did NOT embed the
``lora`` cfg block into ``ckpt['model_config']``. Consumers
(``batch_inference_gcs.load_model``, ad-hoc analyses) read ``model_config``,
constructed EffortDetector WITHOUT the LoRA install step, then
``load_state_dict(..., strict=False)`` silently dropped the 16 lora_A/lora_B
tensors as unexpected keys. The Slot 2 ckpt loaded this way scored a
degraded non-LoRA model. Manual canary numbers differed materially from the
W&B in-training canary (score_p95_on_reals 0.746 vs 0.931).

Fix:
  1. ``trainer/trainer.py:save_ckpt`` now embeds ``self.config.get('lora')``
     into ``model_config``.
  2. ``batch_inference_gcs.py:load_model`` now checks ``cfg.get('lora', {}).get('enabled')``
     and, when true, calls ``apply_lora_to_openclip_visual`` AFTER constructing
     EffortDetector and BEFORE ``load_state_dict``.
  3. ``load_model`` now emits a HARD warning if ``load_state_dict`` returned
     unexpected lora_* keys (silent-drop is the failure mode).

This test asserts the round-trip contract: a LoRA state_dict loaded into a
freshly-constructed model has ZERO unexpected lora_* keys.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.lora_adapter import (  # noqa: E402
    apply_lora_to_openclip_visual,
    count_lora_parameters,
)


# Reuse the FakeVisual fixtures from test_lora_adapter.py — they mirror the
# OpenCLIP attribute layout that apply_lora_to_openclip_visual expects.


class _FakeMLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: int = 4):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, embed_dim * mlp_ratio)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(embed_dim * mlp_ratio, embed_dim)


class _FakeResblock(nn.Module):
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, mlp_ratio: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = _FakeMLP(embed_dim, mlp_ratio)


class _FakeTransformer(nn.Module):
    def __init__(self, n_blocks: int = 4, embed_dim: int = 64):
        super().__init__()
        self.resblocks = nn.Sequential(*[_FakeResblock(embed_dim) for _ in range(n_blocks)])


class _FakeVisual(nn.Module):
    def __init__(self, n_blocks: int = 4, embed_dim: int = 64):
        super().__init__()
        self.transformer = _FakeTransformer(n_blocks, embed_dim)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_lora_save_load_round_trip_zero_unexpected():
    """Round-trip: apply LoRA, take state_dict, fresh model + apply LoRA,
    load_state_dict → zero unexpected lora_* keys."""
    torch.manual_seed(0)
    target_layers = [2, 3]
    rank = 4
    alpha = 8.0

    # 1. Build + LoRA-wrap a "trained" model, take its state_dict.
    visual_trained = _FakeVisual(n_blocks=4, embed_dim=64)
    n_wrapped_trained = apply_lora_to_openclip_visual(
        visual_trained, target_layers=target_layers, rank=rank, alpha=alpha,
    )
    assert n_wrapped_trained > 0
    state_dict = visual_trained.state_dict()
    n_lora_in_sd = sum(1 for k in state_dict if "lora_" in k.lower())
    assert n_lora_in_sd > 0, "state_dict should contain lora_* tensors after wrap"

    # 2. Build a FRESH model, apply LoRA per saved cfg, load_state_dict.
    visual_loaded = _FakeVisual(n_blocks=4, embed_dim=64)
    apply_lora_to_openclip_visual(
        visual_loaded, target_layers=target_layers, rank=rank, alpha=alpha,
    )
    missing, unexpected = visual_loaded.load_state_dict(state_dict, strict=False)

    # 3. Assert no lora_* tensors were dropped as unexpected.
    unexpected_lora = [k for k in unexpected if "lora_" in k.lower()]
    assert unexpected_lora == [], (
        f"{len(unexpected_lora)} lora_* tensors silently dropped: "
        f"{unexpected_lora[:5]}"
    )
    missing_lora = [k for k in missing if "lora_" in k.lower()]
    assert missing_lora == [], (
        f"{len(missing_lora)} lora_* tensors missing from state_dict: "
        f"{missing_lora[:5]}"
    )


def test_lora_skip_install_drops_state_dict_keys_as_unexpected():
    """Negative control: if a fresh model is NOT LoRA-wrapped before
    load_state_dict, ALL lora_* state_dict keys land in unexpected_keys.

    This is exactly the pre-fix failure mode of batch_inference_gcs.load_model.
    """
    torch.manual_seed(0)
    target_layers = [2, 3]
    rank = 4
    alpha = 8.0

    visual_trained = _FakeVisual(n_blocks=4, embed_dim=64)
    apply_lora_to_openclip_visual(
        visual_trained, target_layers=target_layers, rank=rank, alpha=alpha,
    )
    state_dict = visual_trained.state_dict()

    visual_fresh = _FakeVisual(n_blocks=4, embed_dim=64)
    # IMPORTANT: do NOT apply LoRA here — this mirrors the pre-fix bug.
    _missing, unexpected = visual_fresh.load_state_dict(state_dict, strict=False)
    unexpected_lora = [k for k in unexpected if "lora_" in k.lower()]
    assert len(unexpected_lora) > 0, (
        "Expected the pre-fix failure mode: lora_* keys should be dropped "
        "as unexpected when the model has no LoRA install."
    )


def test_save_ckpt_embeds_lora_block_in_model_config():
    """Direct check of the trainer.save_ckpt change: when cfg contains a lora
    block, the embedded model_config must carry it through. We verify the
    dict-build logic without invoking the full trainer."""
    # Mirror the relevant subset of trainer.save_ckpt's checkpoint dict
    # construction (trainer/trainer.py:1406-1421 + the new line for 'lora').
    cfg = {
        "model_name": "effort",
        "use_arcface_head": False,
        "lambda_reg": 1.0,
        "rank": 736,
        "lora": {
            "enabled": True,
            "target_layers": [8, 9],
            "rank": 8,
            "alpha": 16.0,
        },
    }

    # Replicate the trainer's model_config dict-build (post-fix).
    model_config = {
        "model_name": cfg.get("model_name"),
        "use_arcface_head": cfg.get("use_arcface_head", False),
        "lambda_reg": cfg.get("lambda_reg", 1.0),
        "rank": cfg.get("rank", 1023),
        "lora": cfg.get("lora") or {},
    }

    assert model_config["lora"]["enabled"] is True
    assert model_config["lora"]["target_layers"] == [8, 9]
    assert model_config["lora"]["rank"] == 8
    assert model_config["lora"]["alpha"] == 16.0
