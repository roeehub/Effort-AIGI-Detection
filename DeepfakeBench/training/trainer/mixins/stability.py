"""
Stability Regularization Mixin — Perturbation Consistency Loss.

Adds an input-perturbation consistency regularisation term to the training
objective.  For each batch, a slightly perturbed copy is generated (Gaussian
noise + small spatial shift) and the KL-divergence between perturbed and
clean predictions is penalised.  This teaches the model to be invariant to
codec jitter and face-crop instability — the two leading causes of frame-to-
frame score instability observed in R8.

Config keys (all optional — disabled when ``stability_lambda == 0``):
    stability_lambda:       float  Weight of the consistency loss (default 0.0).
    stability_noise_std:    float  Std-dev of Gaussian pixel noise (default 0.02).
    stability_crop_jitter:  float  Fraction of spatial crop-and-resize shift (default 0.03).
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class StabilityRegMixin:
    """Mixin that adds perturbation-consistency regularisation."""

    # ------------------------------------------------------------------
    # Initialisation (called from Trainer.__init__)
    # ------------------------------------------------------------------

    def init_stability_reg(self):
        self.stability_lambda = float(self.config.get("stability_lambda", 0.0))
        self.stability_noise_std = float(self.config.get("stability_noise_std", 0.02))
        self.stability_crop_jitter = float(self.config.get("stability_crop_jitter", 0.03))

        if self.stability_lambda > 0:
            self.logger.info(
                "StabilityReg ENABLED: lambda=%.4f  noise_std=%.4f  crop_jitter=%.4f",
                self.stability_lambda,
                self.stability_noise_std,
                self.stability_crop_jitter,
            )
        else:
            self.logger.info("StabilityReg DISABLED (stability_lambda=0)")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def compute_stability_loss(
        self,
        model: torch.nn.Module,
        data_dict: dict,
        predictions: dict,
    ) -> torch.Tensor:
        """
        Compute ``stability_lambda * KL(perturbed || clean.detach())``.

        Uses **raw (unpenalised) logits** on both sides so that the
        comparison is fair regardless of whether ArcFace is active.

        Args:
            model:       The current model (may be DDP-wrapped).
            data_dict:   The original batch dict (must contain ``'image'``).
            predictions: Output dict from the clean forward pass (must
                         contain ``'raw_logits'``).

        Returns:
            Scalar loss tensor.  Returns ``torch.tensor(0.)`` when disabled.
        """
        images = data_dict["image"]
        if self.stability_lambda <= 0:
            return torch.tensor(0.0, device=images.device)

        logits_clean = predictions["raw_logits"]

        perturbed = self._generate_perturbation(images)

        # Forward pass on perturbed inputs in inference mode so that
        # ArcFace returns raw logits (no margin penalty, no label needed).
        with torch.cuda.amp.autocast():
            perturbed_data = {"image": perturbed}
            perturbed_preds = model(perturbed_data, inference=True)
            logits_perturbed = perturbed_preds["raw_logits"]

        # KL-divergence: D_KL(perturbed || clean_detached)
        log_p_perturbed = F.log_softmax(logits_perturbed, dim=-1)
        p_clean = F.softmax(logits_clean.detach(), dim=-1)

        kl = F.kl_div(log_p_perturbed, p_clean, reduction="batchmean")

        return self.stability_lambda * kl

    # ------------------------------------------------------------------
    # Perturbation generation
    # ------------------------------------------------------------------

    def _generate_perturbation(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply small Gaussian noise + random crop-and-resize to *images*.

        The perturbations mimic the jitter introduced by codec re-encoding
        and face-detector bounding-box instability.
        """
        perturbed = images.clone()

        # 1. Additive Gaussian noise
        if self.stability_noise_std > 0:
            noise = torch.randn_like(perturbed) * self.stability_noise_std
            perturbed = perturbed + noise

        # 2. Random spatial crop-and-resize (simulates bbox jitter)
        if self.stability_crop_jitter > 0 and perturbed.dim() == 4:
            B, C, H, W = perturbed.shape
            j = self.stability_crop_jitter
            # Random offsets in [-j, +j] fraction of spatial dims
            top = int(torch.randint(0, max(1, int(H * j)), (1,)).item())
            left = int(torch.randint(0, max(1, int(W * j)), (1,)).item())
            crop_h = H - top
            crop_w = W - left
            if crop_h > 4 and crop_w > 4:
                perturbed = perturbed[:, :, top : top + crop_h, left : left + crop_w]
                perturbed = F.interpolate(
                    perturbed, size=(H, W), mode="bilinear", align_corners=False
                )

        # NOTE: Do NOT clamp to [0, 1] — images are CLIP-normalised (approx
        # range [-2, +3]).  Clamping would destroy the image content and make
        # the stability loss train against a clamp artefact.
        return perturbed
