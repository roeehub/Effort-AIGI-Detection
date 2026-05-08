#!/usr/bin/env python3
"""Build a candidate-cutoff lookup for the IQ atlas.

For each metric and each pool, report the fraction of frames below a series
of candidate cutoffs. This lets the user read off "if I gate at cutoff X,
how many frames in pool Y do I drop?" without re-querying the parquet.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
THIS = ROOT / "analysis" / "iq_data_atlas_2026-05-08"
OUT = THIS / "outputs"


METRIC_CUTOFFS = {
    # min_dim cutoffs (px). Production-typical lower bound is ~150-200; canary
    # chronic shows p05=82.
    "min_dim": [80, 100, 128, 150, 180, 200, 224, 256, 300],
    # lap_var cutoffs (sharpness; higher = sharper). Lockbox cam_test_s33
    # was reported around 10-20; IQ-shortcut memory says train is ~120+.
    "lap_var": [5, 10, 15, 20, 30, 50, 80, 150, 250],
    # luma_mean cutoffs (brightness; very low = very dark)
    "luma_mean": [40, 60, 80, 100, 120, 140, 160, 180],
    # color_b_dev cutoffs
    "color_b_dev": [5, 8, 10, 12, 15, 20, 30, 50],
    # edge_mag cutoffs
    "edge_mag": [10, 15, 20, 25, 30, 40, 50, 70, 100],
    # bytes
    "bytes": [10_000, 20_000, 30_000, 50_000, 70_000, 100_000, 150_000, 250_000],
}


def main():
    df = pd.read_parquet(OUT / "per_frame.parquet")
    rows = []
    for metric, cutoffs in METRIC_CUTOFFS.items():
        if metric not in df.columns:
            continue
        for pool, g in df.groupby("pool"):
            s = g[metric].dropna()
            if s.empty:
                continue
            for c in cutoffs:
                rows.append({
                    "metric": metric,
                    "pool": pool,
                    "role": g["role"].iloc[0],
                    "cutoff": c,
                    "n": len(s),
                    "n_below": int((s < c).sum()),
                    "frac_below": float((s < c).mean()),
                })
    out = pd.DataFrame(rows).sort_values(["metric", "cutoff", "pool"])
    out.to_csv(OUT / "cutoff_lookup.csv", index=False)
    print(f"wrote {OUT / 'cutoff_lookup.csv'}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
