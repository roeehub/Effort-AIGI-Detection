"""Unit tests for ``detectors/lora_adapter.py`` (LoRA-on-layers-10-11 task).

Three core tests per task spec §3.1:
  1. Zero-init parity — LoRA delta is mathematically zero at init.
  2. Gradient targeting — only ``lora_A`` / ``lora_B`` receive gradients.
  3. Rank scaling — higher rank produces more trainable parameters.

Plus one robustness test:
  4. Stacking on SVD — LoRA composes with the existing ``_svd_in_proj`` routing.

Tests use a minimal fake resblock (``FakeResblock``) that mirrors the structure
``apply_lora_to_resblock`` expects (``attn`` is ``nn.MultiheadAttention``;
``mlp`` has ``c_fc`` / ``c_proj``). No external weights or downloads required.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn


# Allow `import detectors.lora_adapter` when tests are run with the repo root
# on the path (e.g., via `python -m pytest tests/`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.lora_adapter import (  # noqa: E402
    DEFAULT_TARGET_MODULES,
    LoRAInProjModule,
    LoRALinear,
    LORA_PARAM_MARKERS,
    apply_lora_to_openclip_visual,
    apply_lora_to_resblock,
    count_lora_parameters,
    freeze_base_clip_encoder,
)


# ---------------------------------------------------------------------------
# Test fixtures: minimal stand-in for an OpenCLIP ResidualAttentionBlock.
# ---------------------------------------------------------------------------


class _FakeMLP(nn.Module):
    """Mirrors OpenCLIP's MLP: named ``c_fc`` and ``c_proj`` linears."""

    def __init__(self, embed_dim: int, mlp_ratio: int = 4):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, embed_dim * mlp_ratio)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(embed_dim * mlp_ratio, embed_dim)

    def forward(self, x):  # noqa: D401
        return self.c_proj(self.gelu(self.c_fc(x)))


class _FakeResblock(nn.Module):
    """Minimal OpenCLIP-style ``ResidualAttentionBlock``.

    Has the attribute layout (``attn``, ``ln_1``, ``ln_2``, ``mlp.c_fc``,
    ``mlp.c_proj``) that :func:`apply_lora_to_resblock` reaches into. Avoids
    pulling in ``open_clip`` for unit tests.
    """

    def __init__(self, embed_dim: int = 64, num_heads: int = 4, mlp_ratio: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = _FakeMLP(embed_dim, mlp_ratio)

    def forward(self, x):  # noqa: D401
        x_norm = self.ln_1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x


class _FakeTransformer(nn.Module):
    def __init__(self, n_blocks: int = 4, embed_dim: int = 64):
        super().__init__()
        self.resblocks = nn.Sequential(*[_FakeResblock(embed_dim) for _ in range(n_blocks)])


class _FakeVisual(nn.Module):
    def __init__(self, n_blocks: int = 4, embed_dim: int = 64):
        super().__init__()
        self.transformer = _FakeTransformer(n_blocks, embed_dim)


# ---------------------------------------------------------------------------
# Test 1: Zero-init parity — LoRA delta is exactly zero at init.
# ---------------------------------------------------------------------------


def test_zero_init_lora_delta_is_exactly_zero():
    """``B`` is zero-initialised, so ``B @ A * scaling = 0`` everywhere at init."""
    torch.manual_seed(0)
    resblock = _FakeResblock(embed_dim=64, num_heads=4)
    n_wrapped = apply_lora_to_resblock(resblock, rank=8, alpha=16)
    assert n_wrapped == len(DEFAULT_TARGET_MODULES)

    # in_proj LoRA
    in_proj = resblock.attn._lora_in_proj
    assert isinstance(in_proj, LoRAInProjModule)
    assert in_proj.delta.abs().max().item() == 0.0

    # out_proj / MLP LoRA wrappers
    for layer in (resblock.attn.out_proj, resblock.mlp.c_fc, resblock.mlp.c_proj):
        assert isinstance(layer, LoRALinear)
        delta = (layer.lora_B.weight @ layer.lora_A.weight) * layer.scaling
        assert delta.abs().max().item() == 0.0


def test_zero_init_forward_output_matches_base_layer_weight():
    """``LoRALinear.weight`` equals base ``.weight`` exactly when LoRA delta is zero."""
    torch.manual_seed(0)
    base = nn.Linear(32, 16)
    wrapped = LoRALinear(base, rank=4, alpha=8)
    # At init, B=0 so wrapped.weight should equal base.weight exactly.
    assert torch.allclose(wrapped.weight, base.weight, atol=0.0, rtol=0.0)
    if base.bias is not None:
        assert torch.equal(wrapped.bias, base.bias)


def test_zero_init_full_resblock_forward_parity():
    """End-to-end: a freshly LoRA-wrapped resblock's forward output should
    match a non-wrapped identical resblock's output, to within slow-path vs
    fast-path numerical noise from ``nn.MultiheadAttention``.
    """
    torch.manual_seed(42)
    resblock_a = _FakeResblock(embed_dim=64, num_heads=4)
    resblock_b = _FakeResblock(embed_dim=64, num_heads=4)
    resblock_b.load_state_dict(resblock_a.state_dict())
    resblock_a.eval()
    resblock_b.eval()

    apply_lora_to_resblock(resblock_b, rank=8, alpha=16)

    x = torch.randn(8, 2, 64)
    with torch.no_grad():
        y_a = resblock_a(x)
        y_b = resblock_b(x)

    max_diff = (y_a - y_b).abs().max().item()
    # Tolerance accommodates that the monkey-patched MHA forward uses the slow
    # path (``F.multi_head_attention_forward``) regardless of PyTorch's
    # fast-path heuristics; the math is identical, but reduction order may
    # differ by a small amount.
    assert max_diff < 1e-4, (
        f"LoRA at init should be a near-no-op, got max diff {max_diff:.3e}"
    )


# ---------------------------------------------------------------------------
# Test 2: Gradient targeting — only LoRA params receive gradients post-freeze.
# ---------------------------------------------------------------------------


def test_only_lora_params_receive_gradients_after_freeze():
    torch.manual_seed(0)
    resblock = _FakeResblock(embed_dim=64, num_heads=4)
    apply_lora_to_resblock(resblock, rank=8, alpha=16)

    # Freeze every non-LoRA param under the resblock (mirrors what
    # ``freeze_base_clip_encoder`` does at the visual-encoder scope).
    for name, p in resblock.named_parameters():
        if any(marker in name for marker in LORA_PARAM_MARKERS):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    # At init, B=0 means dL/dA = 0 (vacuously). Perturb B by a small amount so
    # the backward pass produces non-trivial gradients on A as well.
    with torch.no_grad():
        for name, p in resblock.named_parameters():
            if "lora_B" in name:
                p.add_(0.01)

    for p in resblock.parameters():
        p.grad = None

    x = torch.randn(8, 2, 64)
    loss = resblock(x).sum()
    loss.backward()

    has_grad_lora = []
    has_grad_base = []
    for name, p in resblock.named_parameters():
        is_lora = any(marker in name for marker in LORA_PARAM_MARKERS)
        grad_nonzero = (p.grad is not None) and (p.grad.abs().sum().item() > 0)
        if grad_nonzero:
            (has_grad_lora if is_lora else has_grad_base).append(name)

    assert len(has_grad_lora) > 0, "Expected at least one LoRA param to receive a gradient"
    assert len(has_grad_base) == 0, (
        f"Frozen base params received gradients: {has_grad_base}"
    )


# ---------------------------------------------------------------------------
# Test 3: Rank scaling sanity.
# ---------------------------------------------------------------------------


def _count_lora_params(module: nn.Module) -> int:
    return sum(
        p.numel()
        for n, p in module.named_parameters()
        if any(marker in n for marker in LORA_PARAM_MARKERS)
    )


def test_higher_rank_yields_more_lora_parameters():
    """LoRA params scale linearly with rank (rank=32 has exactly 8× more than rank=4)."""
    block_low = _FakeResblock(embed_dim=64, num_heads=4)
    apply_lora_to_resblock(block_low, rank=4, alpha=8)
    n_low = _count_lora_params(block_low)

    block_high = _FakeResblock(embed_dim=64, num_heads=4)
    apply_lora_to_resblock(block_high, rank=32, alpha=64)
    n_high = _count_lora_params(block_high)

    assert n_low > 0
    assert n_high > n_low
    # Each LoRA pair (A, B) scales linearly with rank, so total LoRA params
    # for rank=32 should be exactly 8× the rank=4 count.
    assert n_high == n_low * 8, (
        f"Expected exact 8× scaling rank=4 -> rank=32, got {n_low} -> {n_high}"
    )


# ---------------------------------------------------------------------------
# Test 4: Composition with SVD-in-proj routing.
# ---------------------------------------------------------------------------


def _install_fake_svd_in_proj(attn: nn.MultiheadAttention) -> None:
    """Minimal stand-in for the real SVD-in-proj routing: install a marker and
    a tiny module with a ``weight`` property and a ``svd_q.weight_main``
    parameter so ``apply_lora_to_resblock`` can find the dtype/device.
    """
    import types as _types

    class _FakeSVDInProj(nn.Module):
        def __init__(self, embed_dim: int):
            super().__init__()
            # SVD residuals are stored as nn.Parameter; the real
            # SVDInProjLinear nests `svd_q.weight_main` similarly.
            self.svd_q = nn.Module()
            self.svd_q.weight_main = nn.Parameter(
                torch.zeros(embed_dim, embed_dim), requires_grad=False
            )
            self._fused = nn.Parameter(
                torch.randn(3 * embed_dim, embed_dim) * 0.02, requires_grad=False
            )

        @property
        def weight(self):
            return self._fused

    attn._svd_in_proj = _FakeSVDInProj(int(attn.embed_dim))
    attn._svd_in_proj_routing_active = True

    # Also install a slow-path forward so the test exercises the same code
    # path the LoRA installer will overwrite.
    def _forward(self, query, key=None, value=None, **_kwargs):
        if key is None:
            key = query
        if value is None:
            value = query
        return torch.nn.functional.multi_head_attention_forward(
            query, key, value,
            self.embed_dim, self.num_heads,
            self._svd_in_proj.weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            need_weights=False,
            use_separate_proj_weight=False,
        )

    attn.forward = _types.MethodType(_forward, attn)


def test_lora_stacks_on_svd_in_proj_routing():
    """When SVD-in-proj routing is already installed, LoRA's installer must
    detect it and add its delta on top of the SVD-routed weight.
    """
    resblock = _FakeResblock(embed_dim=64, num_heads=4)
    _install_fake_svd_in_proj(resblock.attn)
    assert getattr(resblock.attn, "_svd_in_proj_routing_active", False) is True

    apply_lora_to_resblock(resblock, rank=8, alpha=16)
    assert getattr(resblock.attn, "_lora_in_proj_routing_active", False) is True

    # Forward should run end-to-end without exceptions and produce a finite
    # tensor; at init the LoRA delta is zero so behaviour is governed by the
    # SVD-routed base weight.
    resblock.eval()
    x = torch.randn(8, 2, 64)
    with torch.no_grad():
        y = resblock(x)
    assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# Test 5: visual-encoder-level helpers.
# ---------------------------------------------------------------------------


def test_apply_lora_to_openclip_visual_indexes_by_layer():
    visual = _FakeVisual(n_blocks=4, embed_dim=64)
    n_wrapped = apply_lora_to_openclip_visual(
        visual, target_layers=[2, 3], rank=8, alpha=16
    )
    assert n_wrapped == 2 * len(DEFAULT_TARGET_MODULES)

    # Blocks 0 and 1 should NOT have LoRA installed.
    for idx in (0, 1):
        block = visual.transformer.resblocks[idx]
        assert not isinstance(block.attn.out_proj, LoRALinear)
        assert not hasattr(block.attn, "_lora_in_proj")

    # Blocks 2 and 3 should have LoRA installed.
    for idx in (2, 3):
        block = visual.transformer.resblocks[idx]
        assert isinstance(block.attn.out_proj, LoRALinear)
        assert hasattr(block.attn, "_lora_in_proj")


def test_freeze_base_clip_encoder_freezes_everything_but_lora():
    visual = _FakeVisual(n_blocks=4, embed_dim=64)
    apply_lora_to_openclip_visual(visual, target_layers=[2, 3], rank=8, alpha=16)
    trainable, total = freeze_base_clip_encoder(visual)
    assert 0 < trainable < total

    for name, p in visual.named_parameters():
        is_lora = any(marker in name for marker in LORA_PARAM_MARKERS)
        if is_lora:
            assert p.requires_grad, f"{name} (LoRA) should be trainable"
        else:
            assert not p.requires_grad, f"{name} (base) should be frozen"

    lora_count, total2 = count_lora_parameters(visual)
    assert total2 == total
    assert lora_count == trainable


def test_apply_lora_rejects_out_of_range_layer():
    visual = _FakeVisual(n_blocks=4, embed_dim=64)
    with pytest.raises(IndexError):
        apply_lora_to_openclip_visual(visual, target_layers=[7], rank=8, alpha=16)


def test_apply_lora_rejects_double_install():
    resblock = _FakeResblock(embed_dim=64, num_heads=4)
    apply_lora_to_resblock(resblock, rank=8, alpha=16)
    with pytest.raises(RuntimeError):
        apply_lora_to_resblock(resblock, rank=8, alpha=16)
