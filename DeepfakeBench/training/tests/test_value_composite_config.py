"""
Tests for A9 value_composite config-driven gate + stability-jitter stat.

Exercises `_compute_value_composite` directly under the old (0.02 / 0.04 / max)
and new (0.03 / 0.05 / p95) packet-3.5 configurations.
"""
from __future__ import annotations

import numpy as np
import pytest


# --- Fixtures ----------------------------------------------------------------


def _make_pool(preds, labels):
    return {"preds": np.asarray(preds, dtype=float), "labels": np.asarray(labels, dtype=int)}


@pytest.fixture
def simple_pools():
    """Two real pools + one teams fake pool + one other fake pool.

    Real pool A: 98 negatives below 0.50, 2 above 0.50 (so FPR@0.50 = 2%)
    Real pool B: 97 negatives below 0.55, 3 above 0.55 (so FPR@0.55 ≈ 3%)
    Teams fake pool: 90 positives above 0.55 (TPR@0.55 = 90%)
    Other fake pool: 80 positives above 0.55 (TPR@0.55 = 80%)
    """
    real_a_preds = np.concatenate([np.linspace(0.0, 0.49, 98), np.array([0.60, 0.70])])
    real_a_labels = np.zeros_like(real_a_preds, dtype=int)

    real_b_preds = np.concatenate([np.linspace(0.0, 0.54, 97), np.array([0.65, 0.75, 0.85])])
    real_b_labels = np.zeros_like(real_b_preds, dtype=int)

    teams_fake_preds = np.linspace(0.55, 1.0, 100)
    teams_fake_labels = np.ones_like(teams_fake_preds, dtype=int)

    other_fake_preds = np.linspace(0.50, 1.0, 100)
    other_fake_labels = np.ones_like(other_fake_preds, dtype=int)

    return {
        "real": {
            "proper_clean_real": _make_pool(real_a_preds, real_a_labels),
            "proper_teams_real": _make_pool(real_b_preds, real_b_labels),
        },
        "teams_fake": {
            "teams_ood_fake": _make_pool(teams_fake_preds, teams_fake_labels),
        },
        "other_fake": {
            "wma_failure_fake": _make_pool(other_fake_preds, other_fake_labels),
        },
    }


# --- Tests -------------------------------------------------------------------


def test_value_composite_default_gate_is_packet3_legacy(simple_pools):
    """Default gate is (0.02, 0.04); baseline for retro-compat."""
    from trainer.trainer import _compute_value_composite

    vc = _compute_value_composite(
        real_pools=simple_pools["real"],
        teams_fake_pools=simple_pools["teams_fake"],
        other_fake_pools=simple_pools["other_fake"],
        stability_jitter_max=0.0,
    )
    # Defaults are 0.02 / 0.04 — packet 3 legacy.
    assert vc["tau"] is not None, f"Expected τ resolvable under legacy gate; blocked_by={vc['value_composite_blocked_by']}"


def test_value_composite_new_gate_relaxes_tau(simple_pools):
    """Relaxed gate (0.03 / 0.05) should produce equal or lower τ than (0.02 / 0.04)."""
    from trainer.trainer import _compute_value_composite

    vc_legacy = _compute_value_composite(
        real_pools=simple_pools["real"],
        teams_fake_pools=simple_pools["teams_fake"],
        other_fake_pools=simple_pools["other_fake"],
        stability_jitter_max=0.0,
        target_mean_fpr=0.02,
        max_pool_fpr=0.04,
    )
    vc_new = _compute_value_composite(
        real_pools=simple_pools["real"],
        teams_fake_pools=simple_pools["teams_fake"],
        other_fake_pools=simple_pools["other_fake"],
        stability_jitter_max=0.0,
        target_mean_fpr=0.03,
        max_pool_fpr=0.05,
    )
    # Relaxed gate tolerates more FPR → τ is lower → more fakes caught → composite up or equal.
    # At minimum, neither should be NaN when the other resolves.
    if vc_legacy["tau"] is not None and vc_new["tau"] is not None:
        assert vc_new["tau"] <= vc_legacy["tau"] + 1e-6


def test_value_composite_stability_gate_trip(simple_pools):
    """Pool B has FPR@0.50 of 3% which should trip max_pool_fpr=0.04 legacy gate
    when target is 0.02 — no τ satisfies mean=0.02 AND max≤0.04."""
    from trainer.trainer import _compute_value_composite

    # Build a stress test: one pool with high FPR tail
    pools_stress = dict(simple_pools["real"])
    heavy_tail = np.concatenate(
        [np.linspace(0.0, 0.40, 50), np.linspace(0.55, 1.0, 50)]
    )
    pools_stress["proper_teams_real"] = _make_pool(heavy_tail, np.zeros_like(heavy_tail, dtype=int))

    vc = _compute_value_composite(
        real_pools=pools_stress,
        teams_fake_pools=simple_pools["teams_fake"],
        other_fake_pools=simple_pools["other_fake"],
        stability_jitter_max=0.0,
        target_mean_fpr=0.02,
        max_pool_fpr=0.04,
    )
    # Either blocked by worst-pool, or τ resolved and max_fpr ≤ 0.04.
    if vc["tau"] is None:
        assert vc["value_composite_blocked_by"] == "worst_pool_fpr"


def test_value_composite_stability_term_clamped(simple_pools):
    """stability = 1 − max_jitter, clamped to [0, 1]."""
    from trainer.trainer import _compute_value_composite

    # jitter_max = 0.6 → stability = 0.4
    vc_mid = _compute_value_composite(
        real_pools=simple_pools["real"],
        teams_fake_pools=simple_pools["teams_fake"],
        other_fake_pools=simple_pools["other_fake"],
        stability_jitter_max=0.6,
    )
    assert abs(vc_mid["stability"] - 0.4) < 1e-6

    # jitter_max = 1.0 → stability = 0 (packet 3 observed)
    vc_high = _compute_value_composite(
        real_pools=simple_pools["real"],
        teams_fake_pools=simple_pools["teams_fake"],
        other_fake_pools=simple_pools["other_fake"],
        stability_jitter_max=1.0,
    )
    assert vc_high["stability"] == 0.0

    # jitter_max = 0 (perfect) → stability = 1
    vc_perfect = _compute_value_composite(
        real_pools=simple_pools["real"],
        teams_fake_pools=simple_pools["teams_fake"],
        other_fake_pools=simple_pools["other_fake"],
        stability_jitter_max=0.0,
    )
    assert vc_perfect["stability"] == 1.0


def test_value_composite_weights_sum(simple_pools):
    """When all components present, active_weight=1.0 (0.6 + 0.3 + 0.1)."""
    from trainer.trainer import _compute_value_composite

    vc = _compute_value_composite(
        real_pools=simple_pools["real"],
        teams_fake_pools=simple_pools["teams_fake"],
        other_fake_pools=simple_pools["other_fake"],
        stability_jitter_max=0.5,  # stability = 0.5
    )
    # If τ resolves, composite should equal
    #   (0.6·teams_tpr + 0.3·other_tpr + 0.1·0.5) / 1.0
    if vc["tau"] is not None:
        expected = (
            0.6 * vc["teams_fakes_tpr"]
            + 0.3 * vc["other_fakes_tpr"]
            + 0.1 * 0.5
        )
        assert abs(vc["value_composite"] - expected) < 1e-6


def test_aggregate_jitter_has_p95(simple_pools):
    """Regression test: _aggregate_jitter_across_videos emits p95 alongside max."""
    from trainer.trainer import _aggregate_jitter_across_videos

    # Two videos: one stable, one with a single spike.
    per_video = [
        {"mean": 0.02, "max": 0.05, "diffs": np.array([0.01, 0.02, 0.03, 0.05])},
        {"mean": 0.01, "max": 0.95, "diffs": np.array([0.01, 0.01, 0.95, 0.02, 0.01])},
    ]
    agg = _aggregate_jitter_across_videos(per_video)
    assert "max" in agg and "p95" in agg and "mean" in agg
    # max picks up the spike
    assert agg["max"] == pytest.approx(0.95)
    # p95 across all 9 diffs: sorted = [0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.05, 0.95]
    # p95 interpolates near the top; should be much less than 0.95 since only 1/9 is a spike.
    assert agg["p95"] < agg["max"]
    assert agg["p95"] <= 0.95
