"""Tier-1 P21 reporting standard — operating-point grid wrapper.

Given a `promotion_contract` scorecard directory (containing `threshold_grid.csv`
as produced by `arena/score_teams_promotion_contract.py`), produce a
multi-FPR-floor recall table per checkpoint.

This implements the recommendation in
`analysis/cpu_decision_2026-05-02_pm_late/FINDINGS_AND_DECISION.md` §1:

> Adopt P21 as the reporting standard. Every future contract scorecard should
> report recall at FPR ∈ {2, 5, 10, 20}%.

Usage:
    python -m arena.scorecard_with_fpr_grid \
        --grid analysis/p18_probe_2026-05-01/d_results/promotion_contract/threshold_grid.csv \
        --out  analysis/p18_probe_2026-05-01/d_results/promotion_contract/operating_point_grid.csv

The output CSV has one row per (checkpoint, fpr_floor) with columns:
  checkpoint_key, fpr_floor, threshold, dev_primary_real_fpr,
  <fake_suite>_recall (one column per fake suite),
  <real_stress_suite>_fpr (one column per real stress suite),
  notes (optional human-readable comment).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd

DEFAULT_FPR_FLOORS = (0.02, 0.05, 0.10, 0.20)


def _suite_columns(grid: pd.DataFrame) -> tuple[List[str], List[str]]:
    fake_suites: List[str] = []
    real_suites: List[str] = []
    for col in grid.columns:
        m = re.match(r"^(.+)__fake_recall$", col)
        if m:
            fake_suites.append(m.group(1))
            continue
        m = re.match(r"^(.+)__real_fpr$", col)
        if m:
            real_suites.append(m.group(1))
    return fake_suites, real_suites


def select_threshold_at_fpr(grid: pd.DataFrame, primary_fpr_col: str, fpr_floor: float) -> pd.Series | None:
    """Return the row from `grid` whose primary FPR ≤ floor and whose τ is *minimal*
    among such rows.

    Lower τ within the constraint = higher recall. Higher τ within the constraint
    is the trivial "predict no fakes" answer at τ ≥ 1, which we don't want.

    Falls back to the row with smallest primary FPR if no row meets the floor.
    """
    valid = grid[grid[primary_fpr_col] <= fpr_floor + 1e-12]
    if len(valid) == 0:
        return grid.loc[grid[primary_fpr_col].idxmin()]
    return valid.loc[valid["threshold"].idxmin()]


def build_operating_point_table(
    grid: pd.DataFrame,
    fpr_floors: Iterable[float] = DEFAULT_FPR_FLOORS,
) -> pd.DataFrame:
    fake_suites, real_suites = _suite_columns(grid)
    primary_fpr_col = "dev_primary_real_fpr"
    if primary_fpr_col not in grid.columns:
        raise ValueError(f"grid is missing required column '{primary_fpr_col}'; got {list(grid.columns)}")

    rows: List[dict] = []
    for ckpt, sub in grid.groupby("checkpoint_key"):
        sub = sub.sort_values("threshold", ascending=False).reset_index(drop=True)
        for floor in fpr_floors:
            best = select_threshold_at_fpr(sub, primary_fpr_col, floor)
            if best is None:
                continue
            row = {
                "checkpoint_key": ckpt,
                "fpr_floor": floor,
                "threshold": float(best["threshold"]),
                "dev_primary_real_fpr": float(best[primary_fpr_col]),
            }
            if "dev_worst_real_stress_fpr" in best.index:
                row["dev_worst_real_stress_fpr"] = float(best["dev_worst_real_stress_fpr"])
            if "dev_fake_macro_recall" in best.index:
                row["dev_fake_macro_recall"] = float(best["dev_fake_macro_recall"])
            for s in fake_suites:
                col = f"{s}__fake_recall"
                val = best.get(col)
                row[f"{s}_recall"] = float(val) if pd.notna(val) else None
            for s in real_suites:
                col = f"{s}__real_fpr"
                val = best.get(col)
                row[f"{s}_FPR"] = float(val) if pd.notna(val) else None
            # Sanity comment
            achieved_fpr = row["dev_primary_real_fpr"]
            row["notes"] = (
                f"τ chosen so primary FPR={achieved_fpr:.4f} (≤ floor {floor:.2f})"
                if achieved_fpr <= floor + 1e-9
                else f"floor {floor:.2f} infeasible; report shows minimum-FPR τ "
                     f"(primary FPR={achieved_fpr:.4f})"
            )
            rows.append(row)
    return pd.DataFrame(rows)


def format_pretty(table: pd.DataFrame, fake_suites: Iterable[str]) -> str:
    cols = ["checkpoint_key", "fpr_floor", "threshold", "dev_primary_real_fpr"]
    cols.extend([f"{s}_recall" for s in fake_suites])
    cols.append("notes")
    cols = [c for c in cols if c in table.columns]
    fmt: dict = {}
    for c in cols:
        if "recall" in c or "fpr" in c.lower() or c == "fpr_floor" or c == "threshold":
            fmt[c] = lambda x: f"{x:.4f}" if pd.notna(x) else "—"
    return table[cols].to_string(index=False, formatters=fmt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True, help="Path to threshold_grid.csv")
    parser.add_argument("--out", required=True, help="Where to write operating_point_grid.csv")
    parser.add_argument(
        "--fpr_floors",
        nargs="+",
        type=float,
        default=list(DEFAULT_FPR_FLOORS),
        help="FPR floors to report (default: 0.02 0.05 0.10 0.20)",
    )
    args = parser.parse_args()

    grid = pd.read_csv(args.grid)
    out = build_operating_point_table(grid, args.fpr_floors)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(out)} rows)")
    print()
    fake_suites, _ = _suite_columns(grid)
    # Restrict pretty print to a few key fake suites
    key_suites = [s for s in fake_suites if any(
        kw in s for kw in ("teams_fake_all", "visomaster_enhanced_macro", "deeplive_enhanced")
    )]
    print(format_pretty(out, key_suites))


if __name__ == "__main__":
    main()
