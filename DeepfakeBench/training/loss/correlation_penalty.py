"""Correlation-penalty loss for shortcut avoidance.

Adds a regularization term `lambda_ * sum_axis |Pearson_batch(score, axis)|`
to the classification loss, forcing the model's score to be uncorrelated
with specified per-frame nuisance features (sharpness, luma, etc.) within
each training batch.

Background: 2026-05-04/05 audit established that R13 detectors learn
capture-condition shortcuts (sharpness, luma, face_area, capture_mode).
Symmetric augmentation (PC packet) destroyed the discriminative signal —
direct evidence the model relies on those axes. This loss is the explicit
form of the shortcut-avoidance objective: penalize the score's batch-level
correlation with each nuisance axis.

A frozen-head Phase 2 prototype (analysis/corr_penalty_prototype_2026-05-05/)
showed:
  - Mechanism converges across lambda in {0, 0.1, 1, 10, 100}; lambda=10
    starts oscillating, lambda=100 collapses to chance.
  - At lambda=1 on a 50/50 test population: |r_sharp| 0.156 -> 0.053 (-66%),
    |r_luma| 0.233 -> 0.071 (-69%); recall@FPR=10% INCREASED 0.355 -> 0.491
    (+13.6pp). On a frozen head, untargeted axes (face_area, is_webcam)
    rebound — shortcut shifting. Encoder fine-tune is positioned to fix it.

See:
  - docs/packet_retrospectives/threads/processing_signature_shortcut.md
  - analysis/shortcut_audit_2026-05-05/
  - analysis/corr_penalty_prototype_2026-05-05/
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def batch_pearson(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable Pearson correlation across a batch.

    Args:
        x: [B] tensor (e.g., model score)
        y: [B] tensor (e.g., per-frame nuisance axis value)
        eps: stabilizer for the denominator

    Returns:
        scalar tensor: Pearson correlation in [-1, 1]
    """
    if x.numel() < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    xc = x - x.mean()
    yc = y - y.mean()
    num = (xc * yc).sum()
    denom = xc.norm() * yc.norm() + eps
    return num / denom


def compute_pixel_axes(image: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute per-frame nuisance axes derivable from the input image alone.

    Two axes returned:
      - `sharpness_laplacian`: variance of the Laplacian on the luma channel.
        Standard image-sharpness metric; matches the lockbox-tagging parquet's
        column (modulo full-vs-face-crop caveat from `sharpness_metric_bug`).
      - `luma_mean`: mean Y-channel value via Rec. 601 weights.

    Args:
        image: [B, 3, H, W] OR [B, T, 3, H, W] float tensor. The combined-paired
            collate emits a video-style 5-D tensor; we flatten to per-frame
            so axis values match the per-frame score that get_losses computes
            after pred.softmax(dim=-1)[:, 1] over the [B*T, 2] logits.
            Range can be [0, 1] or normalized ImageNet stats — the correlation
            is scale-invariant so absolute range does not matter.

    Returns:
        dict with keys 'sharpness_laplacian', 'luma_mean'; each value is a
        [B] (or [B*T] for 5-D input) tensor.
    """
    if image.dim() == 5:
        # [B, T, 3, H, W] -> [B*T, 3, H, W]
        bsz, tlen = image.shape[0], image.shape[1]
        image = image.reshape(bsz * tlen, *image.shape[2:])
    if image.dim() != 4 or image.shape[1] != 3:
        raise ValueError(f"compute_pixel_axes expects [B, 3, H, W] or [B, T, 3, H, W]; got {tuple(image.shape)}")

    # Rec. 601 luma — works on either [0,1] or normalized inputs
    luma = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]  # [B, H, W]
    luma_mean = luma.mean(dim=(1, 2))  # [B]

    # Laplacian via 3x3 4-neighbor kernel (cv2.Laplacian default)
    lap_kernel = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    luma_4d = luma.unsqueeze(1)  # [B, 1, H, W]
    lap = F.conv2d(luma_4d, lap_kernel, padding=1)  # [B, 1, H, W]
    sharpness = lap.var(dim=(1, 2, 3))  # [B]

    return {
        "sharpness_laplacian": sharpness,
        "luma_mean": luma_mean,
    }


class CorrelationPenalty:
    """Sum of |Pearson(score, axis)| over a configured list of axes, scaled by lambda.

    Usage from the detector's loss assembly:

        if self.corr_penalty is not None:
            score = pred_dict['cls'].softmax(dim=-1)[:, 1]  # P(fake)
            axes = compute_pixel_axes(data_dict['image'])
            corr_loss, per_axis_r = self.corr_penalty(score, axes)
            losses['correlation_penalty'] = corr_loss
            losses['overall'] = losses['overall'] + corr_loss
            for ax_name, r_val in per_axis_r.items():
                losses[f'corr_r_{ax_name}'] = r_val.abs()
    """

    def __init__(self, axes: List[str], lambda_: float):
        if lambda_ < 0:
            raise ValueError(f"lambda_ must be non-negative; got {lambda_}")
        self.axes = list(axes)
        self.lambda_ = float(lambda_)

    def __call__(
        self,
        score: torch.Tensor,
        axis_values: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the penalty + per-axis raw Pearson values for diagnostics.

        Args:
            score: [B] tensor — the model's per-frame score (probability of fake).
            axis_values: dict mapping axis name -> [B] tensor.

        Returns:
            (penalty_loss, per_axis_pearson_dict)
            - penalty_loss is `lambda_ * sum_axis |r|`, scalar tensor
            - per_axis_pearson_dict maps axis name -> scalar tensor of r
              (NOT |r|) so calling code can log signed values for diagnostics
        """
        device = score.device
        per_axis_r: Dict[str, torch.Tensor] = {}
        if self.lambda_ == 0.0 or len(self.axes) == 0:
            zero = torch.zeros((), device=device, dtype=score.dtype)
            for ax in self.axes:
                per_axis_r[ax] = zero
            return zero, per_axis_r

        sum_abs_r = torch.zeros((), device=device, dtype=score.dtype)
        for ax in self.axes:
            if ax not in axis_values:
                raise KeyError(
                    f"CorrelationPenalty axis {ax!r} not in axis_values "
                    f"(have: {sorted(axis_values.keys())})"
                )
            r = batch_pearson(score, axis_values[ax])
            per_axis_r[ax] = r
            sum_abs_r = sum_abs_r + r.abs()

        return self.lambda_ * sum_abs_r, per_axis_r


def build_correlation_penalty_from_config(
    config: Optional[dict],
) -> Optional[CorrelationPenalty]:
    """Build a CorrelationPenalty from a yaml config block.

    Expected yaml shape:

        correlation_penalty:
          enabled: true
          lambda: 1.0
          axes:
            - sharpness_laplacian
            - luma_mean

    Returns None if the block is missing or `enabled: false`.
    """
    if not config:
        return None
    block = config.get("correlation_penalty") if isinstance(config, dict) else None
    if not block or not block.get("enabled", False):
        return None
    axes = block.get("axes") or ["sharpness_laplacian", "luma_mean"]
    lam = float(block.get("lambda", 1.0))
    return CorrelationPenalty(axes=axes, lambda_=lam)
