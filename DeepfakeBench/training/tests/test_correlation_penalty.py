"""Tests for loss/correlation_penalty.py — the shortcut-avoidance regularizer.

Coverage:
  - batch_pearson is differentiable + matches numpy reference within tolerance
  - compute_pixel_axes returns expected shapes / non-degenerate values
  - CorrelationPenalty: returns 0 when lambda=0 or empty axes
  - CorrelationPenalty: gradient flows through the score variable
  - CorrelationPenalty: builder respects `enabled: false` and missing config
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from loss.correlation_penalty import (
    CorrelationPenalty,
    batch_pearson,
    build_correlation_penalty_from_config,
    compute_pixel_axes,
)


class TestBatchPearson:
    def test_perfect_positive(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
        r = batch_pearson(x, y)
        assert torch.isclose(r, torch.tensor(1.0), atol=1e-5)

    def test_perfect_negative(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        r = batch_pearson(x, y)
        assert torch.isclose(r, torch.tensor(-1.0), atol=1e-5)

    def test_uncorrelated(self):
        torch.manual_seed(0)
        x = torch.randn(1000)
        y = torch.randn(1000)
        r = batch_pearson(x, y)
        assert abs(r.item()) < 0.1, f"random vectors should be ~uncorrelated, got r={r.item()}"

    def test_matches_numpy(self):
        rng = np.random.default_rng(42)
        x_np = rng.normal(size=64)
        y_np = rng.normal(loc=x_np * 0.7, size=64)  # correlated
        r_np = float(np.corrcoef(x_np, y_np)[0, 1])
        r_torch = batch_pearson(torch.from_numpy(x_np), torch.from_numpy(y_np)).item()
        assert abs(r_np - r_torch) < 1e-6

    def test_gradient_flows(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        y = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0])
        r = batch_pearson(x, y)
        r.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestComputePixelAxes:
    def test_shape_correct(self):
        img = torch.rand(4, 3, 32, 32)
        axes = compute_pixel_axes(img)
        assert "sharpness_laplacian" in axes
        assert "luma_mean" in axes
        assert axes["sharpness_laplacian"].shape == (4,)
        assert axes["luma_mean"].shape == (4,)

    def test_uniform_image_low_sharpness(self):
        # Constant image → laplacian variance ≈ 0 (small border-padding artifact)
        img = torch.full((2, 3, 16, 16), 0.5)
        axes = compute_pixel_axes(img)
        # Border padding produces tiny non-zero variance; checkerboard test verifies
        # the metric ranks high-freq much higher. Keep tolerance loose here.
        assert (axes["sharpness_laplacian"] < 0.1).all()
        assert torch.allclose(axes["luma_mean"], torch.full((2,), 0.5), atol=1e-5)

    def test_high_freq_image_high_sharpness(self):
        # Checkerboard → very high laplacian variance
        img = torch.zeros(1, 3, 16, 16)
        for i in range(16):
            for j in range(16):
                if (i + j) % 2 == 0:
                    img[0, :, i, j] = 1.0
        axes = compute_pixel_axes(img)
        assert axes["sharpness_laplacian"][0].item() > 1.0

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            compute_pixel_axes(torch.rand(3, 32, 32))  # missing batch dim


class TestCorrelationPenalty:
    def test_zero_lambda_returns_zero(self):
        cp = CorrelationPenalty(axes=["a"], lambda_=0.0)
        score = torch.tensor([0.1, 0.5, 0.9])
        axes = {"a": torch.tensor([1.0, 2.0, 3.0])}
        loss, _ = cp(score, axes)
        assert loss.item() == 0.0

    def test_empty_axes_returns_zero(self):
        cp = CorrelationPenalty(axes=[], lambda_=1.0)
        score = torch.tensor([0.1, 0.5, 0.9])
        loss, _ = cp(score, {})
        assert loss.item() == 0.0

    def test_lambda_scales_loss(self):
        cp1 = CorrelationPenalty(axes=["a"], lambda_=1.0)
        cp2 = CorrelationPenalty(axes=["a"], lambda_=3.0)
        score = torch.tensor([0.1, 0.4, 0.6, 0.9])
        axes = {"a": torch.tensor([1.0, 2.0, 3.0, 4.0])}
        l1, _ = cp1(score, axes)
        l2, _ = cp2(score, axes)
        assert torch.isclose(l2 / 3.0, l1, atol=1e-6)

    def test_per_axis_r_returned(self):
        cp = CorrelationPenalty(axes=["sharp", "luma"], lambda_=1.0)
        score = torch.tensor([0.1, 0.4, 0.7, 0.9])
        axes = {
            "sharp": torch.tensor([1.0, 2.0, 3.0, 4.0]),  # positively correlated
            "luma": torch.tensor([4.0, 3.0, 2.0, 1.0]),   # negatively correlated
        }
        loss, per_axis_r = cp(score, axes)
        assert "sharp" in per_axis_r
        assert "luma" in per_axis_r
        assert per_axis_r["sharp"].item() > 0.9
        assert per_axis_r["luma"].item() < -0.9
        # loss = lambda * (|r_sharp| + |r_luma|)
        expected = (per_axis_r["sharp"].abs() + per_axis_r["luma"].abs())
        assert torch.isclose(loss, expected, atol=1e-6)

    def test_missing_axis_raises(self):
        cp = CorrelationPenalty(axes=["a", "missing"], lambda_=1.0)
        with pytest.raises(KeyError, match="missing"):
            cp(torch.tensor([0.1, 0.5]), {"a": torch.tensor([1.0, 2.0])})

    def test_gradient_flows_through_score(self):
        score = torch.tensor([0.1, 0.4, 0.7, 0.9], requires_grad=True)
        axes = {"a": torch.tensor([1.0, 2.0, 3.0, 4.0])}
        cp = CorrelationPenalty(axes=["a"], lambda_=1.0)
        loss, _ = cp(score, axes)
        loss.backward()
        assert score.grad is not None
        assert torch.isfinite(score.grad).all()
        assert (score.grad != 0).any()

    def test_negative_lambda_rejected(self):
        with pytest.raises(ValueError):
            CorrelationPenalty(axes=["a"], lambda_=-0.5)


class TestBuilder:
    def test_none_config(self):
        assert build_correlation_penalty_from_config(None) is None
        assert build_correlation_penalty_from_config({}) is None

    def test_disabled_returns_none(self):
        config = {"correlation_penalty": {"enabled": False, "lambda": 1.0}}
        assert build_correlation_penalty_from_config(config) is None

    def test_enabled_returns_instance(self):
        config = {
            "correlation_penalty": {
                "enabled": True,
                "lambda": 2.5,
                "axes": ["sharpness_laplacian", "luma_mean"],
            }
        }
        cp = build_correlation_penalty_from_config(config)
        assert cp is not None
        assert cp.lambda_ == 2.5
        assert cp.axes == ["sharpness_laplacian", "luma_mean"]

    def test_default_axes_when_missing(self):
        config = {"correlation_penalty": {"enabled": True, "lambda": 1.0}}
        cp = build_correlation_penalty_from_config(config)
        assert cp.axes == ["sharpness_laplacian", "luma_mean"]
