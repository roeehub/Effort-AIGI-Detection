"""
SVD residual gradient audit (Phase 0.2 of the shortcut-learning plan).

Verifies that *all* SVD residual parameters in the effort_detector
architecture actually receive non-zero gradients from a synthetic
classification-style loss after one forward+backward pass.

BACKGROUND (2026-04-26)
-----------------------
The original ``apply_svd_to_in_proj`` path used a ``forward_pre_hook`` that
did ``module.in_proj_weight.data.copy_(module._svd_in_proj.weight)``. Because
``.data.copy_`` writes values in-place into a frozen leaf parameter, autograd
never propagated gradients back to the SVD residuals. Every R12 / RLP / P-*
run with this flag enabled was training the q/k/v residuals on regularizer
gradient only.

The fix is in ``detectors/effort_detector.py::_install_svd_in_proj_routing``:
the MHA forward is rewritten to call ``F.multi_head_attention_forward`` with
``self._svd_in_proj.weight`` (an autograd-tracked ``@property``) instead of
the frozen leaf ``self.in_proj_weight``.

This script is the regression test for that fix. Run it after any change to
the SVD/attention plumbing.

USAGE
-----
    cd /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training
    python analysis/probe_battery_2026-04-26/grad_audit.py

Exit code 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

# Ensure the training package is importable.
HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if TRAINING_ROOT not in sys.path:
    sys.path.insert(0, TRAINING_ROOT)

from detectors.effort_detector import (  # noqa: E402
    SVDInProjLinear,
    _install_svd_in_proj_routing,
    replace_with_svd_residual,
)


def _residual_param_names(module: nn.Module) -> list[str]:
    return [
        n for n, p in module.named_parameters()
        if any(tag in n for tag in ("S_residual", "U_residual", "V_residual"))
        and p.requires_grad
    ]


def _grad_summary(module: nn.Module) -> dict:
    """Return {param_name: {is_none, abs_sum, mean, max}}."""
    out = {}
    for name, param in module.named_parameters():
        if not any(tag in name for tag in ("S_residual", "U_residual", "V_residual")):
            continue
        if not param.requires_grad:
            continue
        g = param.grad
        if g is None:
            out[name] = {"is_none": True, "abs_sum": 0.0, "mean": 0.0, "max": 0.0}
        else:
            out[name] = {
                "is_none": False,
                "abs_sum": float(g.abs().sum().item()),
                "mean": float(g.abs().mean().item()),
                "max": float(g.abs().max().item()),
            }
    return out


def audit_in_proj_path(embed_dim: int = 64, num_heads: int = 4, rank: int = 32) -> bool:
    """
    Build a single nn.MultiheadAttention, install the SVD-in-proj routing,
    then forward+backward a synthetic batch and check residual gradients.
    """
    print("=" * 60)
    print("AUDIT 1/2: in_proj SVD path (was BROKEN, fixed 2026-04-26)")
    print("=" * 60)

    mha = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)

    svd_in_proj = SVDInProjLinear(
        mha.in_proj_weight.data,
        mha.in_proj_bias.data if mha.in_proj_bias is not None else None,
        embed_dim,
        rank,
    )
    mha._svd_in_proj = svd_in_proj
    _install_svd_in_proj_routing(mha)

    # Match the freeze pattern from apply_svd_residual_to_openclip_attn.
    mha.in_proj_weight.requires_grad = False
    if mha.in_proj_bias is not None:
        mha.in_proj_bias.requires_grad = False
    if svd_in_proj.bias is not None:
        svd_in_proj.bias.requires_grad = False

    residual_names = _residual_param_names(svd_in_proj)
    print(f"trainable residual params: {len(residual_names)}")
    for n in residual_names:
        print(f"  - {n}")

    x = torch.randn(2, 10, embed_dim)
    out, _ = mha(x, x, x, need_weights=False)
    loss = out.pow(2).mean()
    loss.backward()

    summary = _grad_summary(svd_in_proj)
    failures = [n for n, info in summary.items() if info["is_none"] or info["abs_sum"] == 0.0]

    print("\nresidual gradients:")
    for name, info in sorted(summary.items()):
        status = "PASS" if (not info["is_none"] and info["abs_sum"] > 0) else "FAIL"
        print(f"  [{status}] {name}: abs_sum={info['abs_sum']:.6e} max={info['max']:.6e}")

    if failures:
        print(f"\nFAIL: {len(failures)} in_proj residual(s) received zero or None gradient.")
        return False

    print(f"\nPASS: all {len(summary)} in_proj residual params received non-zero gradient.")
    return True


def audit_linear_svd_path(in_features: int = 64, out_features: int = 128, rank: int = 32) -> bool:
    """
    Sanity check on the MLP / out_proj path (uses SVDResidualLinear). This
    path was always correct; this just confirms the audit infrastructure works.
    """
    print()
    print("=" * 60)
    print("AUDIT 2/2: SVDResidualLinear path (used for MLP / out_proj)")
    print("=" * 60)

    base = nn.Linear(in_features, out_features)
    svd = replace_with_svd_residual(base, rank)

    residual_names = _residual_param_names(svd)
    print(f"trainable residual params: {len(residual_names)}")
    for n in residual_names:
        print(f"  - {n}")

    x = torch.randn(8, in_features)
    out = svd(x)
    loss = out.pow(2).mean()
    loss.backward()

    summary = _grad_summary(svd)
    failures = [n for n, info in summary.items() if info["is_none"] or info["abs_sum"] == 0.0]

    print("\nresidual gradients:")
    for name, info in sorted(summary.items()):
        status = "PASS" if (not info["is_none"] and info["abs_sum"] > 0) else "FAIL"
        print(f"  [{status}] {name}: abs_sum={info['abs_sum']:.6e} max={info['max']:.6e}")

    if failures:
        print(f"\nFAIL: {len(failures)} residual(s) received zero or None gradient.")
        return False

    print(f"\nPASS: all {len(summary)} residual params received non-zero gradient.")
    return True


def main() -> int:
    torch.manual_seed(0)
    in_proj_ok = audit_in_proj_path()
    linear_ok = audit_linear_svd_path()

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  in_proj path:  {'PASS' if in_proj_ok else 'FAIL'}")
    print(f"  linear path:   {'PASS' if linear_ok else 'FAIL'}")

    if in_proj_ok and linear_ok:
        print("\nAll SVD residual paths receive classification-loss gradient.")
        return 0
    print("\nFAILURE: at least one SVD path is broken.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
