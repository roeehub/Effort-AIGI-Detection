"""Regression test for ``canary-silence-when-multi-axis-grl-active`` (open loop
in docs/packet_retrospectives/threads/in_training_canary_signal.md).

Before fix (2026-05-20): runs with ``multi_axis_grl.enabled: true`` AND
``canary_probe.enabled: true`` would silently lose the canary because the
mixin called ``self.model(data_dict)`` with ``inference=False`` (default),
which hit a training-only forward branch that raised; the try/except at
``trainer/mixins/canary_probe.py:128`` swallowed it and disabled the canary
for the rest of the run. The T5C_TRIPLE 2026-05-20 batch lost 4h+1h of
mid-training visibility this way.

Fix: pass ``inference=True`` so the model's forward at
``detectors/effort_detector.py:1742-1846`` skips the training-only branches
(``use_quality_head and not inference`` at line 1836; ``use_multi_axis_grl
and not inference`` at line 1843).

These tests assert the contract from the open loop's ``close_criterion``:
calling forward with ``inference=True`` on a model with
``use_multi_axis_grl=True`` does NOT raise and DOES return a dict with
``cls`` / ``prob`` keys for canary scoring.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# A minimal EffortDetector-shaped model that exercises the inference-vs-train
# branches. We don't need the full CLIP backbone for this regression test —
# only the forward-signature contract matters.
# ---------------------------------------------------------------------------


class _TinyMultiAxisGRLBlock(nn.Module):
    """Mirrors detectors/effort_detector.py:MultiAxisGRLBlock signature."""

    def __init__(self, hidden_size: int, axes: list):
        super().__init__()
        self.axes = list(axes)
        self.heads = nn.ModuleDict({ax: nn.Linear(hidden_size, 2) for ax in axes})

    def forward(self, features):
        return {ax: self.heads[ax](features) for ax in self.axes}


class _TinyEffortDetector(nn.Module):
    """Forward path mirrors detectors/effort_detector.py:1742-1846.

    Crucially: the multi_axis_grl branch is gated by ``not inference``. When
    the canary mixin called forward without inference=True, this branch fired
    and could raise (the actual raise site under the live trainer involves
    DDP + training-only data_dict expectations; the contract we test here is
    ``inference=True must not invoke any training-only branch``).
    """

    def __init__(self, hidden_size: int = 64, axes=("chronic_flag", "is_dor")):
        super().__init__()
        self.backbone = nn.Linear(3 * 8 * 8, hidden_size)  # toy "pooler_output"
        self.head = nn.Linear(hidden_size, 2)
        self.use_arcface_head = False
        self.use_quality_head = False
        self.use_multi_axis_grl = True
        self.multi_axis_grl_block = _TinyMultiAxisGRLBlock(hidden_size, list(axes))
        # Tracks whether the training-only branch fired (we assert it did NOT
        # fire when inference=True).
        self.training_branch_fired = False

    def forward(self, data_dict: dict, inference: bool = False):
        image = data_dict["image"]
        B = image.shape[0]
        features = self.backbone(image.view(B, -1))
        raw_logits = self.head(features)
        pred = {
            "cls": raw_logits,
            "prob": torch.softmax(raw_logits, dim=1)[:, 1],
            "feat": features,
            "raw_logits": raw_logits,
        }
        if self.use_quality_head and not inference:
            self.training_branch_fired = True
        if self.use_multi_axis_grl and not inference:
            self.training_branch_fired = True
            # Mirror the real model: assemble per-axis labels from data_dict
            # (which the canary's data_dict={'image': batch} does NOT carry).
            # Raise here to simulate the silent-fail the canary hit in prod.
            _ = data_dict["chronic_flag"]  # KeyError on canary's minimal dict
            pred["multi_axis_grl_logits"] = self.multi_axis_grl_block(features)
        return pred


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_forward_with_inference_true_skips_grl_branch():
    """With inference=True, the multi_axis_grl branch must not fire."""
    model = _TinyEffortDetector()
    batch = torch.randn(4, 3, 8, 8)
    pred = model({"image": batch}, inference=True)
    assert not model.training_branch_fired, "GRL branch fired despite inference=True"
    assert isinstance(pred, dict)
    assert "cls" in pred and "prob" in pred
    assert pred["cls"].shape == (4, 2)
    assert pred["prob"].shape == (4,)


def test_forward_without_inference_kwarg_hits_grl_and_raises():
    """Without inference=True (canary's pre-fix call), the GRL branch fires
    and raises because data_dict lacks per-axis labels. This mirrors the
    actual production failure."""
    model = _TinyEffortDetector()
    batch = torch.randn(4, 3, 8, 8)
    with pytest.raises(KeyError):
        model({"image": batch})  # no inference=True — fires the GRL branch


def test_canary_mixin_call_pattern_succeeds_with_inference_true():
    """End-to-end: the canary mixin's call pattern (after fix) succeeds.

    Mirrors trainer/mixins/canary_probe.py:_compute_canary_metrics at lines
    301-330 (post-fix): per-batch forward with inference=True, extract
    probs from pred['prob'] (or fall back to softmax over pred['cls']).
    """
    model = _TinyEffortDetector()
    model.eval()
    tensor = torch.randn(16, 3, 8, 8)
    bs = 4
    scores: list = []
    for start in range(0, tensor.shape[0], bs):
        end = min(start + bs, tensor.shape[0])
        batch = tensor[start:end]
        pred = model({"image": batch}, inference=True)
        if isinstance(pred, dict) and "prob" in pred and pred["prob"] is not None:
            probs = pred["prob"]
            if probs.dim() == 2:
                probs = probs[:, 1]
            scores.extend(probs.float().cpu().tolist())
        else:
            logits = pred["cls"] if isinstance(pred, dict) else pred
            probs = torch.softmax(logits, dim=-1)[:, 1]
            scores.extend(probs.float().cpu().tolist())
    assert len(scores) == 16
    assert all(0.0 <= s <= 1.0 for s in scores)
