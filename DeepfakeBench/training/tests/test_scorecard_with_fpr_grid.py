"""Test the P21 reporting wrapper (arena/scorecard_with_fpr_grid.py)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from arena.scorecard_with_fpr_grid import (
    DEFAULT_FPR_FLOORS,
    build_operating_point_table,
    select_threshold_at_fpr,
)


def _toy_grid() -> pd.DataFrame:
    """Synthetic monotone grid: as τ drops, FPR rises and recall rises."""
    rows = []
    for ckpt in ("CKPT_A", "CKPT_B"):
        for tau, fpr, recall_a, recall_b in [
            (1.00, 0.000, 0.000, 0.000),
            (0.99, 0.020, 0.350, 0.020),  # contract op point
            (0.97, 0.050, 0.500, 0.080),
            (0.90, 0.100, 0.700, 0.260),
            (0.50, 0.200, 0.870, 0.560),
            (0.10, 0.500, 0.990, 0.900),
        ]:
            # CKPT_B has scaled-down recall by 0.5 for testability
            scale = 0.5 if ckpt == "CKPT_B" else 1.0
            rows.append({
                "checkpoint_key": ckpt,
                "checkpoint_path": f"/tmp/{ckpt}.pth",
                "threshold": tau,
                "teams_real_all_dev__real_fpr": fpr,
                "teams_real_all_dev__fake_recall": None,
                "teams_fake_all_dev__real_fpr": None,
                "teams_fake_all_dev__fake_recall": recall_a * scale,
                "visomaster_enhanced_macro_dev__real_fpr": None,
                "visomaster_enhanced_macro_dev__fake_recall": recall_b * scale,
                "deeplive_enhanced_dev__real_fpr": None,
                "deeplive_enhanced_dev__fake_recall": recall_a * scale * 0.8,
                "dev_primary_real_fpr": fpr,
                "dev_worst_real_stress_fpr": fpr * 1.2,
                "dev_fake_macro_recall": (recall_a + recall_b) / 2 * scale,
            })
    return pd.DataFrame(rows)


class TestSelectThresholdAtFpr:
    def test_picks_minimum_tau_within_constraint(self):
        # Lowest τ s.t. FPR ≤ 0.05 is τ=0.97 (FPR=0.050).
        sub = _toy_grid()
        sub = sub[sub.checkpoint_key == "CKPT_A"]
        chosen = select_threshold_at_fpr(sub, "dev_primary_real_fpr", 0.05)
        assert chosen is not None
        assert chosen["threshold"] == pytest.approx(0.97)
        assert chosen["dev_primary_real_fpr"] == pytest.approx(0.05)

    def test_does_not_pick_trivial_tau_one(self):
        # The bug we're guarding against: picking τ=1.00 (recall=0) when
        # τ=0.99 (recall=0.35) is also within FPR=0.02.
        sub = _toy_grid()
        sub = sub[sub.checkpoint_key == "CKPT_A"]
        chosen = select_threshold_at_fpr(sub, "dev_primary_real_fpr", 0.02)
        assert chosen["threshold"] != pytest.approx(1.00)
        assert chosen["teams_fake_all_dev__fake_recall"] > 0.3

    def test_falls_back_to_min_fpr_if_no_valid_row(self):
        # Floor below smallest available FPR — should fall back to τ=1.00 (FPR=0).
        sub = _toy_grid()
        sub = sub[sub.checkpoint_key == "CKPT_A"]
        chosen = select_threshold_at_fpr(sub, "dev_primary_real_fpr", 0.0001)
        assert chosen["dev_primary_real_fpr"] == pytest.approx(0.0)


class TestBuildOperatingPointTable:
    def test_one_row_per_checkpoint_per_floor(self):
        out = build_operating_point_table(_toy_grid(), DEFAULT_FPR_FLOORS)
        assert len(out) == 2 * len(DEFAULT_FPR_FLOORS)
        assert set(out.checkpoint_key) == {"CKPT_A", "CKPT_B"}

    def test_recall_columns_for_every_fake_suite(self):
        out = build_operating_point_table(_toy_grid())
        for s in ("teams_fake_all_dev", "visomaster_enhanced_macro_dev", "deeplive_enhanced_dev"):
            assert f"{s}_recall" in out.columns

    def test_recall_monotone_with_fpr_floor(self):
        # As we relax the FPR floor, recall should not drop.
        out = build_operating_point_table(_toy_grid())
        for ckpt, sub in out.groupby("checkpoint_key"):
            sub = sub.sort_values("fpr_floor")
            recalls = sub["teams_fake_all_dev_recall"].to_numpy()
            assert all(recalls[i] <= recalls[i+1] + 1e-9 for i in range(len(recalls) - 1)), \
                f"non-monotone recall on {ckpt}: {recalls}"

    def test_b_strictly_below_a(self):
        # CKPT_B was constructed with halved recall; expect B < A at every floor.
        out = build_operating_point_table(_toy_grid())
        a = out[out.checkpoint_key == "CKPT_A"].sort_values("fpr_floor").reset_index(drop=True)
        b = out[out.checkpoint_key == "CKPT_B"].sort_values("fpr_floor").reset_index(drop=True)
        for col in ("teams_fake_all_dev_recall", "visomaster_enhanced_macro_dev_recall"):
            for i in range(len(a)):
                if a[col].iloc[i] > 0:
                    assert b[col].iloc[i] < a[col].iloc[i] + 1e-9


class TestWithRealScorecard:
    def test_p18_scorecard_has_three_checkpoints(self):
        path = Path("analysis/p18_probe_2026-05-01/d_results/promotion_contract/threshold_grid.csv")
        if not path.exists():
            pytest.skip(f"{path} not present")
        out = build_operating_point_table(pd.read_csv(path))
        ckpts = set(out.checkpoint_key)
        assert "P8A_REFERENCE_STEP5000" in ckpts
        # Confirm wrapper picks meaningful τ < 1 at fpr_floor=0.02
        p8a_2pct = out[(out.checkpoint_key == "P8A_REFERENCE_STEP5000") & (out.fpr_floor == 0.02)]
        assert len(p8a_2pct) == 1
        assert p8a_2pct.threshold.iloc[0] < 1.0
        assert p8a_2pct.threshold.iloc[0] > 0.9
