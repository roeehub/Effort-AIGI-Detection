#!/usr/bin/env python3
"""Finalize the IQ atlas: aggregate all per-pool parquets into the global
per_frame.parquet, run the canary + production-reference fold-ins, build
summary CSVs, and regenerate figures.

Designed to be run any time after `build_iq_atlas.py` has produced at least
one cached parquet — useful for partial reruns or after a kill+restart.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
THIS = ROOT / "analysis" / "iq_data_atlas_2026-05-08"
OUT = THIS / "outputs"
FIG = THIS / "figs"
CACHE = THIS / "_cache"

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO, datefmt="%H:%M:%S")
log = logging.getLogger("finalize")

# Reuse the helpers from the main build script.
sys.path.insert(0, str(THIS))
from build_iq_atlas import (
    fold_canary, fold_production_reference, per_pool_summary, cross_pool_compare,
    make_figures,
)


def main():
    parquet_files = sorted(CACHE.glob("*.parquet"))
    log.info("Found %d cached parquets", len(parquet_files))

    per_frame = []
    for f in parquet_files:
        try:
            df = pd.read_parquet(f)
            per_frame.append(df)
            log.info("  loaded %s: %d rows", f.name, len(df))
        except Exception as e:
            log.warning("  failed to load %s: %s", f.name, e)

    fold_canary(per_frame)
    fold_production_reference(per_frame)

    if not per_frame:
        log.error("No data — abort")
        sys.exit(1)

    df_all = pd.concat(per_frame, ignore_index=True)
    df_all.to_parquet(OUT / "per_frame.parquet", index=False)
    log.info("Wrote per_frame.parquet (%d rows, %d pools)",
             len(df_all), df_all["pool"].nunique())

    summary = per_pool_summary(df_all)
    summary.to_csv(OUT / "per_pool_summary.csv", index=False)
    log.info("Wrote per_pool_summary.csv (%d pools)", len(summary))

    cross = cross_pool_compare(df_all)
    cross.to_csv(OUT / "cross_pool_compare.csv", index=False)
    log.info("Wrote cross_pool_compare.csv")

    make_figures(df_all)
    log.info("Built figures")

    # Print a quick headline for convenience
    print()
    print("=== Summary by group / metric (p50) ===")
    df_all["group"] = df_all["role"]
    for metric in ["min_dim", "lap_var", "luma_mean", "color_b_dev"]:
        if metric not in df_all.columns:
            continue
        print(f"\n  {metric}:")
        for grp, g in df_all.groupby("group"):
            s = g[metric].dropna()
            if len(s) < 5:
                continue
            print(f"    {grp:30s}  n={len(s):4d}  p05={s.quantile(0.05):.1f}  "
                  f"p50={s.quantile(0.50):.1f}  p95={s.quantile(0.95):.1f}")


if __name__ == "__main__":
    main()
