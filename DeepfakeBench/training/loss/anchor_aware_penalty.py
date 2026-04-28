"""Anchor-aware penalty — push prob_fake on false-flag real pools toward target.

Background. The Dor/Roee anchor pools at ``analysis/teams_pool_rescore.py``
designate a set of real frames the model wrongly scores as fake (the
"false-flag" pools — most reliably ``dor-real-webcam-false-flag-no-virtual-bg``).
Those false flags are downstream of three documented shortcuts (camera/pipeline
signature, face pixel-area, webcam-mode). At validation time the trainer
already monitors ``anchor_mean`` on these pools (trainer.py:2080); this module
adds a *training-time* penalty so the gradient sees the false-flag mistake on
every step rather than only between validations.

The penalty is a hinge on the mean probability of fake across a sampled
mini-batch of cached anchor frames:

    penalty = weight * max(0, mean(prob_fake) - target)^2

When the model already keeps anchor_mean below the target (default 0.10),
the penalty is zero and adds no gradient. As soon as the model drifts above,
the penalty grows quadratically.

Config (all under ``anchor_aware:`` block in the run yaml):

    enabled:           bool   default False
    weight:            float  default 5.0
    target_mean_prob:  float  default 0.10
    samples_per_step:  int    default 16
    pool_names:        list   default [ANCHOR_POOL] (single most-reliable pool)
"""
from __future__ import annotations

import logging
from typing import Optional

import torch


class AnchorAwarePenalty:
    """Stateful penalty: caches anchor frames once, samples a mini-batch per step.

    Mirrors the shape of ``trainer.mixins.stability.compute_stability_loss``
    but lives outside the mixin tree because its state (cached frame tensor)
    doesn't fit the stateless mixin pattern.
    """

    def __init__(
        self,
        config: dict,
        anchor_cache_dir: str,
        logger: Optional[logging.Logger] = None,
    ):
        cfg = config or {}
        self.enabled = bool(cfg.get('enabled', False))
        self.weight = float(cfg.get('weight', 5.0))
        self.target = float(cfg.get('target_mean_prob', 0.10))
        self.samples_per_step = int(cfg.get('samples_per_step', 16))
        self.pool_names = cfg.get('pool_names', None)
        self.logger = logger or logging.getLogger(__name__)
        self._frames_cpu: Optional[torch.Tensor] = None
        self._device_cache: Optional[torch.device] = None
        self._frames_device: Optional[torch.Tensor] = None

        if not self.enabled or self.weight <= 0:
            self.logger.info(
                "AnchorAwarePenalty DISABLED (enabled=%s weight=%s)",
                self.enabled, self.weight,
            )
            return
        self._load_cache(anchor_cache_dir)

    def _load_cache(self, cache_dir: str) -> None:
        from analysis.teams_pool_rescore import (
            cache_anchor_pools_locally,
            ANCHOR_POOL,
            _LocalAnchorDataset,
        )

        pools = self.pool_names or [ANCHOR_POOL]
        try:
            index = cache_anchor_pools_locally(cache_dir, pools=pools)
        except Exception as exc:
            self.logger.warning(
                "AnchorAwarePenalty: cache load failed (%s); disabling", exc,
            )
            self.enabled = False
            return

        items = []
        for p in pools:
            items.extend(index.get(p, []))
        if not items:
            self.logger.warning(
                "AnchorAwarePenalty: no frames found for pools=%s; disabling", pools,
            )
            self.enabled = False
            return

        ds = _LocalAnchorDataset(items)
        frames = []
        for i in range(len(ds)):
            tensor, _ = ds[i]
            frames.append(tensor)
        self._frames_cpu = torch.stack(frames)
        self.logger.info(
            "AnchorAwarePenalty ENABLED: weight=%.3f target=%.3f samples_per_step=%d "
            "n_frames=%d pools=%s",
            self.weight, self.target, self.samples_per_step,
            self._frames_cpu.shape[0], pools,
        )

    def _frames_on(self, device: torch.device) -> torch.Tensor:
        if self._device_cache is None or self._device_cache != device:
            self._frames_device = self._frames_cpu.to(device, non_blocking=True)
            self._device_cache = device
        return self._frames_device

    def compute(self, model: torch.nn.Module, device: torch.device) -> torch.Tensor:
        """Forward sample, return scalar penalty tensor on ``device``.

        Returns ``torch.tensor(0., device=device)`` when disabled or when the
        cache is empty — safe to add to ``losses['overall']`` unconditionally.
        """
        if not self.enabled or self._frames_cpu is None:
            return torch.zeros((), device=device)

        frames_dev = self._frames_on(device)
        n = frames_dev.shape[0]
        k = min(self.samples_per_step, n)
        idx = torch.randperm(n, device=device)[:k]
        batch = frames_dev[idx]

        outputs = model({"image": batch}, inference=True)
        prob = outputs["prob"]
        if prob.dim() == 2:
            prob_fake = prob[:, 1]
        else:
            prob_fake = prob.view(-1)

        mean_p = prob_fake.mean()
        excess = torch.clamp(mean_p - self.target, min=0.0)
        return self.weight * (excess ** 2)
