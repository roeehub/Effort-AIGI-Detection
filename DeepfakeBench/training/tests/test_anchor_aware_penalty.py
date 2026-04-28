"""Smoke test for loss/anchor_aware_penalty.py — AnchorAwarePenalty.

Covers:
  - disabled config → zero scalar (no GPU/cache work)
  - empty cache (no frames discovered) → self-disables, returns zero
  - hinge below target → zero; above target → positive quadratic
  - shape: returns scalar tensor on the requested device
"""
from __future__ import annotations

import logging

import torch

from loss.anchor_aware_penalty import AnchorAwarePenalty


class _MockModel(torch.nn.Module):
    """Minimal stand-in: returns a fixed prob_fake on every call."""

    def __init__(self, prob_fake: float):
        super().__init__()
        self.fixed = float(prob_fake)
        # One trainable param so gradient bookkeeping is real.
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, data_dict, inference: bool = False):
        x = data_dict["image"]
        n = x.shape[0]
        # prob shape [N, 2] with col 1 = prob_fake — matches Effort detector.
        prob_fake = torch.full((n,), self.fixed, device=x.device) + self.dummy
        prob_real = 1.0 - prob_fake
        prob = torch.stack([prob_real, prob_fake], dim=1)
        return {"prob": prob, "raw_logits": prob}

    def __call__(self, data_dict, inference: bool = False):
        return self.forward(data_dict, inference=inference)


def _make_penalty_with_synth_cache(n_frames: int, **cfg_overrides) -> AnchorAwarePenalty:
    """Build an AnchorAwarePenalty whose cache is pre-populated with synthetic frames."""
    cfg = {
        'enabled': True,
        'weight': 5.0,
        'target_mean_prob': 0.10,
        'samples_per_step': 8,
    }
    cfg.update(cfg_overrides)
    # Construct disabled, then inject synthetic frames to bypass GCS.
    p = AnchorAwarePenalty(
        config={'enabled': False},  # skip _load_cache during construction
        anchor_cache_dir='/nonexistent',
        logger=logging.getLogger('test'),
    )
    p.enabled = bool(cfg['enabled'])
    p.weight = float(cfg['weight'])
    p.target = float(cfg['target_mean_prob'])
    p.samples_per_step = int(cfg['samples_per_step'])
    p._frames_cpu = torch.randn(n_frames, 3, 224, 224)
    return p


class TestAnchorAwarePenalty:
    def test_disabled_returns_zero(self):
        p = AnchorAwarePenalty(
            config={'enabled': False},
            anchor_cache_dir='/nonexistent',
            logger=logging.getLogger('test'),
        )
        device = torch.device('cpu')
        model = _MockModel(prob_fake=0.99)
        loss = p.compute(model, device)
        assert loss.shape == ()
        assert loss.item() == 0.0
        assert loss.device == device

    def test_zero_weight_disables(self):
        p = AnchorAwarePenalty(
            config={'enabled': True, 'weight': 0.0},
            anchor_cache_dir='/nonexistent',
            logger=logging.getLogger('test'),
        )
        device = torch.device('cpu')
        model = _MockModel(prob_fake=0.99)
        loss = p.compute(model, device)
        assert loss.item() == 0.0

    def test_missing_cache_dir_self_disables(self):
        # Real cache dir doesn't exist → cache_anchor_pools_locally will try
        # to fetch from GCS; we expect it to fail and the penalty to disable
        # itself gracefully. Hard to test online without GCS creds, but
        # importantly the constructor must not crash.
        p = AnchorAwarePenalty(
            config={'enabled': True, 'weight': 5.0},
            anchor_cache_dir='/tmp/nonexistent_anchor_cache_for_smoke_test_xxxxx',
            logger=logging.getLogger('test'),
        )
        # Either it disabled itself, or it pulled from GCS and is now usable.
        # Either outcome is acceptable; just verify no crash on compute().
        device = torch.device('cpu')
        model = _MockModel(prob_fake=0.5)
        _ = p.compute(model, device)  # must not raise

    def test_hinge_below_target_returns_zero(self):
        p = _make_penalty_with_synth_cache(
            n_frames=32, target_mean_prob=0.50, weight=5.0,
        )
        device = torch.device('cpu')
        model = _MockModel(prob_fake=0.10)  # well below 0.50
        loss = p.compute(model, device)
        assert loss.item() == 0.0
        assert loss.device == device

    def test_hinge_above_target_is_positive(self):
        p = _make_penalty_with_synth_cache(
            n_frames=32, target_mean_prob=0.10, weight=5.0,
        )
        device = torch.device('cpu')
        model = _MockModel(prob_fake=0.50)  # excess = 0.40
        loss = p.compute(model, device)
        # weight * excess^2 = 5.0 * 0.16 = 0.80
        assert loss.item() > 0.5
        assert loss.item() < 1.0

    def test_returns_scalar_tensor(self):
        p = _make_penalty_with_synth_cache(n_frames=16)
        device = torch.device('cpu')
        model = _MockModel(prob_fake=0.30)
        loss = p.compute(model, device)
        assert loss.dim() == 0
        assert loss.requires_grad  # gradient must flow through model param

    def test_gradient_flows(self):
        p = _make_penalty_with_synth_cache(
            n_frames=16, target_mean_prob=0.10, weight=5.0,
        )
        device = torch.device('cpu')
        model = _MockModel(prob_fake=0.40)
        loss = p.compute(model, device)
        loss.backward()
        # Gradient should be non-zero on the dummy param since
        # excess > 0 and the prob expression includes dummy.
        assert model.dummy.grad is not None
        assert torch.abs(model.dummy.grad).item() > 0.0

    def test_samples_per_step_capped_by_cache_size(self):
        p = _make_penalty_with_synth_cache(
            n_frames=4, samples_per_step=16, target_mean_prob=0.10,
        )
        device = torch.device('cpu')
        model = _MockModel(prob_fake=0.50)
        # Should not crash even though samples_per_step > n_frames.
        loss = p.compute(model, device)
        assert loss.item() > 0
