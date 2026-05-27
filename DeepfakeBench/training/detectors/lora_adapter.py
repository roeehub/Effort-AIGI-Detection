"""LoRA (Low-Rank Adaptation) adapter for OpenCLIP visual transformer resblocks.

Pattern mirrors the existing SVD residual machinery in
``detectors/effort_detector.py`` (``SVDResidualLinear`` 1850,
``SVDInProjLinear`` 2207, ``_install_svd_in_proj_routing`` 2328):

* Wrap a frozen base linear layer; expose ``.weight`` / ``.bias`` properties
  so ``F.multi_head_attention_forward`` keeps working.
* For ``nn.MultiheadAttention.in_proj_weight`` (a leaf ``nn.Parameter``, not a
  module), monkey-patch the MHA instance's ``forward`` to add the LoRA delta
  on top of the existing in_proj path.

Stacks cleanly on top of frozen SVD: ``LoRALinear``'s base layer can be an
``SVDResidualLinear``; ``_install_lora_in_proj_routing`` detects an active
SVD routing (``_svd_in_proj_routing_active`` marker) and pulls the base
weight from ``_svd_in_proj.weight`` instead of from the raw leaf parameter.

See ``docs/packet_retrospectives/LORA_LAYERS_10_11_TASK_2026-05-12.md`` §9b
for the architectural decisions this module implements.
"""

from __future__ import annotations

import logging
import types
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


DEFAULT_TARGET_MODULES: Tuple[str, ...] = (
    "attn.in_proj",
    "attn.out_proj",
    "mlp.c_fc",
    "mlp.c_proj",
)

# Substring markers used to identify LoRA parameters in ``named_parameters``.
LORA_PARAM_MARKERS: Tuple[str, ...] = ("lora_A", "lora_B")


class LoRALinear(nn.Module):
    """LoRA adapter wrapping a frozen base linear-like module.

    Computes ``y = base(x) + B(A(x)) * (alpha / rank)``.

    The wrapper exposes ``.weight`` and ``.bias`` properties so it is a drop-in
    replacement for ``nn.Linear`` (or ``SVDResidualLinear``) inside
    ``nn.MultiheadAttention``'s ``out_proj`` slot, which is read as
    ``out_proj.weight`` by ``F.multi_head_attention_forward``.

    ``A`` is initialised ``N(0, 1/rank)``; ``B`` is initialised to zero. The
    LoRA delta is therefore exactly zero at step 0 — the model behaves
    identically to the frozen base before any optimisation steps.
    """

    def __init__(self, base_layer: nn.Module, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if not hasattr(base_layer, "weight"):
            raise TypeError(
                "LoRALinear base_layer must expose a .weight property "
                f"(got {type(base_layer).__name__})"
            )

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        base_weight = base_layer.weight
        out_features, in_features = int(base_weight.shape[0]), int(base_weight.shape[1])

        self.lora_A = nn.Linear(in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, out_features, bias=False)

        # Match base dtype/device so .to() propagation and AMP both work.
        target_dtype = base_weight.dtype
        target_device = base_weight.device
        with torch.no_grad():
            self.lora_A.weight = nn.Parameter(
                self.lora_A.weight.to(dtype=target_dtype, device=target_device)
            )
            self.lora_B.weight = nn.Parameter(
                self.lora_B.weight.to(dtype=target_dtype, device=target_device)
            )
            nn.init.normal_(self.lora_A.weight, mean=0.0, std=1.0 / self.rank)
            nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self) -> torch.Tensor:
        """Effective weight: ``base.weight + (B @ A) * scaling``.

        Required for ``F.multi_head_attention_forward`` compatibility when
        this module replaces ``nn.MultiheadAttention.out_proj``.
        """
        return self.base_layer.weight + (self.lora_B.weight @ self.lora_A.weight) * self.scaling

    @property
    def bias(self):
        return getattr(self.base_layer, "bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_B(self.lora_A(x)) * self.scaling

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:g}"


class LoRAInProjModule(nn.Module):
    """LoRA A/B for the fused ``in_proj_weight`` of ``nn.MultiheadAttention``.

    The fused in-projection weight has shape ``[3 * embed_dim, embed_dim]``.
    LoRA decomposes that whole matrix at once (Q/K/V are not separated). See
    LORA_LAYERS_10_11_TASK §7 for the design choice.
    """

    def __init__(
        self,
        embed_dim: int,
        rank: int,
        alpha: float,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.embed_dim = int(embed_dim)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        out_features = 3 * self.embed_dim
        self.lora_A = nn.Linear(self.embed_dim, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, out_features, bias=False)

        with torch.no_grad():
            self.lora_A.weight = nn.Parameter(self.lora_A.weight.to(dtype=dtype, device=device))
            self.lora_B.weight = nn.Parameter(self.lora_B.weight.to(dtype=dtype, device=device))
            nn.init.normal_(self.lora_A.weight, mean=0.0, std=1.0 / self.rank)
            nn.init.zeros_(self.lora_B.weight)

    @property
    def delta(self) -> torch.Tensor:
        """LoRA delta of shape ``[3 * embed_dim, embed_dim]``."""
        return (self.lora_B.weight @ self.lora_A.weight) * self.scaling

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, rank={self.rank}, alpha={self.alpha}"


def _install_lora_in_proj_routing(mha_module: nn.MultiheadAttention) -> None:
    """Patch ``nn.MultiheadAttention.forward`` to add the LoRA in_proj delta.

    Stacks on top of either:
      * the raw frozen ``in_proj_weight`` leaf parameter, OR
      * the SVD-routed ``_svd_in_proj.weight`` autograd-tracked property
        (when SVD-in-proj routing is also active).

    The override re-implements the slow path of ``nn.MultiheadAttention.forward``
    and sets the marker attribute ``_lora_in_proj_routing_active = True`` so
    audits / checkpoint loaders can detect that LoRA is installed.

    Mirrors ``detectors/effort_detector.py:_install_svd_in_proj_routing`` (2328).
    """
    if not hasattr(mha_module, "_lora_in_proj"):
        raise RuntimeError(
            "LoRAInProjModule must be attached as mha._lora_in_proj before "
            "_install_lora_in_proj_routing is called"
        )

    svd_routing_active = bool(getattr(mha_module, "_svd_in_proj_routing_active", False))

    def forward(
        self,
        query,
        key=None,
        value=None,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        if key is None:
            key = query
        if value is None:
            value = query

        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        if svd_routing_active:
            base_in_proj_weight = self._svd_in_proj.weight
        else:
            base_in_proj_weight = self.in_proj_weight
        in_proj_weight = base_in_proj_weight + self._lora_in_proj.delta
        in_proj_bias = self.in_proj_bias

        # out_proj.weight is an autograd-tracked property when out_proj is
        # either SVDResidualLinear or LoRALinear; both expose `.weight` and
        # `.bias` for this purpose.
        out_proj_weight = self.out_proj.weight
        out_proj_bias = self.out_proj.bias

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query, key, value,
            self.embed_dim, self.num_heads,
            in_proj_weight, in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, out_proj_weight, out_proj_bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=False,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0)

        return attn_output, attn_output_weights

    mha_module.forward = types.MethodType(forward, mha_module)
    mha_module._lora_in_proj_routing_active = True


def apply_lora_to_resblock(
    resblock: nn.Module,
    rank: int,
    alpha: float,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
) -> int:
    """Apply LoRA adapters inside one OpenCLIP ``ResidualAttentionBlock``.

    Args:
        resblock: Expects ``resblock.attn`` (``nn.MultiheadAttention``) and
                  ``resblock.mlp`` (with ``c_fc`` / ``c_proj`` linears).
        rank: LoRA rank.
        alpha: LoRA scaling factor; effective scale is ``alpha / rank``.
        target_modules: Subset of ``DEFAULT_TARGET_MODULES`` to adapt.

    Returns:
        Number of layers actually wrapped (for sanity assertions).

    Raises:
        RuntimeError: If a target layer already has LoRA installed (avoids
                      accidental double-stacking).
    """
    wrapped = 0
    targets = set(target_modules)

    if "attn.in_proj" in targets:
        attn = resblock.attn
        if not isinstance(attn, nn.MultiheadAttention):
            raise TypeError(
                f"Expected resblock.attn to be nn.MultiheadAttention, "
                f"got {type(attn).__name__}"
            )
        if getattr(attn, "_lora_in_proj_routing_active", False):
            raise RuntimeError("attn.in_proj already has LoRA routing installed")

        embed_dim = int(attn.embed_dim)
        # Inherit dtype/device from whatever currently provides the in_proj
        # weight: SVD's frozen `weight_main` if SVD routing is active, else
        # the raw `in_proj_weight` leaf parameter.
        if hasattr(attn, "_svd_in_proj"):
            ref_w = attn._svd_in_proj.svd_q.weight_main
        else:
            ref_w = attn.in_proj_weight
        attn._lora_in_proj = LoRAInProjModule(
            embed_dim=embed_dim,
            rank=rank,
            alpha=alpha,
            dtype=ref_w.dtype,
            device=ref_w.device,
        )
        _install_lora_in_proj_routing(attn)
        wrapped += 1

    if "attn.out_proj" in targets:
        attn = resblock.attn
        base = attn.out_proj
        if isinstance(base, LoRALinear):
            raise RuntimeError("attn.out_proj already has LoRA installed")
        attn.out_proj = LoRALinear(base, rank=rank, alpha=alpha)
        wrapped += 1

    if "mlp.c_fc" in targets:
        base = resblock.mlp.c_fc
        if isinstance(base, LoRALinear):
            raise RuntimeError("mlp.c_fc already has LoRA installed")
        resblock.mlp.c_fc = LoRALinear(base, rank=rank, alpha=alpha)
        wrapped += 1

    if "mlp.c_proj" in targets:
        base = resblock.mlp.c_proj
        if isinstance(base, LoRALinear):
            raise RuntimeError("mlp.c_proj already has LoRA installed")
        resblock.mlp.c_proj = LoRALinear(base, rank=rank, alpha=alpha)
        wrapped += 1

    return wrapped


def apply_lora_to_openclip_visual(
    visual: nn.Module,
    target_layers: Sequence[int],
    rank: int,
    alpha: float,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
) -> int:
    """Apply LoRA at the specified resblock indices of an OpenCLIP visual encoder.

    Args:
        visual: OpenCLIP visual transformer (``EffortDetector.backbone.visual``).
                Must expose ``.transformer.resblocks`` (an ``nn.Sequential``).
        target_layers: Resblock indices to adapt (e.g. ``[10, 11]``).
        rank, alpha, target_modules: forwarded to :func:`apply_lora_to_resblock`.

    Returns:
        Total number of layers wrapped across the targeted resblocks.

    Raises:
        IndexError: If a target index is out of range.
        AttributeError: If ``visual`` does not have the expected structure.
    """
    if not hasattr(visual, "transformer") or not hasattr(visual.transformer, "resblocks"):
        raise AttributeError(
            "apply_lora_to_openclip_visual expected `visual.transformer.resblocks`; "
            f"got {type(visual).__name__}"
        )
    resblocks = visual.transformer.resblocks
    n_blocks = len(resblocks)
    total = 0
    for idx in target_layers:
        if not (0 <= idx < n_blocks):
            raise IndexError(
                f"LoRA target_layer index {idx} out of range for visual with "
                f"{n_blocks} resblocks"
            )
        total += apply_lora_to_resblock(
            resblocks[idx], rank=rank, alpha=alpha, target_modules=target_modules
        )
    return total


def freeze_base_clip_encoder(visual: nn.Module) -> Tuple[int, int]:
    """Freeze every parameter under ``visual`` whose name does NOT contain a
    LoRA marker (``lora_A`` / ``lora_B``).

    Must be called AFTER LoRA injection so the new LoRA parameters are
    registered and discoverable via ``named_parameters``.

    Args:
        visual: The OpenCLIP visual encoder (e.g.
                ``EffortDetector.backbone.visual``).

    Returns:
        ``(trainable_param_count, total_param_count)`` for the encoder.
    """
    trainable = 0
    total = 0
    for name, param in visual.named_parameters():
        total += param.numel()
        is_lora = any(marker in name for marker in LORA_PARAM_MARKERS)
        if is_lora:
            param.requires_grad_(True)
            trainable += param.numel()
        else:
            param.requires_grad_(False)
    return trainable, total


def count_lora_parameters(visual: nn.Module) -> Tuple[int, int]:
    """Return ``(lora_param_count, total_param_count)`` for ``visual``.

    Useful for the ``[LoRA] enabled ...`` log line in the trainer wiring.
    """
    lora = 0
    total = 0
    for name, param in visual.named_parameters():
        total += param.numel()
        if any(marker in name for marker in LORA_PARAM_MARKERS):
            lora += param.numel()
    return lora, total
