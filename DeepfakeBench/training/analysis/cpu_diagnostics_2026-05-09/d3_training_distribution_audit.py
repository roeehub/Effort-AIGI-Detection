"""D3 — Training-pool distribution audit.

Reads `analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet`,
filters to training-pool frames, characterizes the joint distribution of
(label × IQ axes × method × face_size × identity).

Outputs:
  outputs/d3_training_pool_iq_summary.csv
  outputs/d3_training_real_vs_fake_axis_table.csv
  outputs/d3_training_method_iq_signature.csv

Reading: shows whether a within-batch sampler that wants to balance some
shortcut axis can actually find paired (real, fake) within each bin.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
OUTPUTS = THIS_DIR / "outputs"

ATLAS_PARQUET = (
    REPO_ROOT / "analysis" / "iq_data_atlas_2026-05-08" / "outputs" / "per_frame.parquet"
)

logger = logging.getLogger("d3-training-audit")


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    df = pd.read_parquet(ATLAS_PARQUET)
    logger.info("atlas rows=%d, cols=%s", len(df), list(df.columns))

    # Identify training pools (pool name starts with "train_").
    if "pool" not in df.columns:
        logger.error("atlas missing 'pool' column; cols=%s", list(df.columns))
        return 2
    train_df = df[df["pool"].astype(str).str.startswith("train_")].copy()
    logger.info("training-pool rows=%d (of %d)", len(train_df), len(df))
    if len(train_df) == 0:
        # Fall back: rows whose pool contains "train" anywhere.
        train_df = df[df["pool"].astype(str).str.contains("train", case=False, na=False)].copy()
        logger.info("fallback training-pool rows=%d", len(train_df))

    # Per-pool IQ summary.
    iq_axes = ["lap_var", "min_dim", "luma_mean", "color_b_dev", "edge_mag", "skin_frac"]
    avail_axes = [a for a in iq_axes if a in train_df.columns]
    rows = []
    for pool, sub in train_df.groupby("pool"):
        for axis in avail_axes:
            vals = sub[axis].dropna().values
            if len(vals) == 0:
                continue
            rows.append({
                "pool": pool, "axis": axis,
                "n": int(len(vals)),
                "p10": float(np.percentile(vals, 10)),
                "p50": float(np.percentile(vals, 50)),
                "p90": float(np.percentile(vals, 90)),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
            })
    pool_summary = pd.DataFrame(rows)
    pool_summary.to_csv(OUTPUTS / "d3_training_pool_iq_summary.csv", index=False)
    logger.info("wrote d3_training_pool_iq_summary.csv (%d rows)", len(pool_summary))

    # Real-vs-fake split table (per pool).
    rows = []
    if "label" in train_df.columns or "label_str" in train_df.columns:
        label_col = "label" if "label" in train_df.columns else "label_str"
        for pool, sub in train_df.groupby("pool"):
            real = sub[sub[label_col].astype(str).isin(["real", "0"])]
            fake = sub[sub[label_col].astype(str).isin(["fake", "1"])]
            for axis in avail_axes:
                rv = real[axis].dropna().values
                fv = fake[axis].dropna().values
                rows.append({
                    "pool": pool, "axis": axis,
                    "n_real": len(rv), "n_fake": len(fv),
                    "real_p50": float(np.median(rv)) if len(rv) else float("nan"),
                    "fake_p50": float(np.median(fv)) if len(fv) else float("nan"),
                    "delta_p50": float(np.median(fv) - np.median(rv))
                                 if (len(rv) and len(fv)) else float("nan"),
                })
        rvf = pd.DataFrame(rows)
        rvf.to_csv(OUTPUTS / "d3_training_real_vs_fake_axis_table.csv", index=False)
        logger.info("wrote d3_training_real_vs_fake_axis_table.csv (%d rows)", len(rvf))

    # Method × IQ signature on training fakes (per face-swap method).
    if "method" in train_df.columns or "fake_method" in train_df.columns:
        method_col = "method" if "method" in train_df.columns else "fake_method"
        fake_only = train_df
        if "label" in train_df.columns:
            fake_only = train_df[train_df["label"].astype(str).isin(["fake", "1"])]
        rows = []
        for method, sub in fake_only.groupby(method_col):
            for axis in avail_axes:
                vals = sub[axis].dropna().values
                if len(vals) < 20:
                    continue
                rows.append({
                    "method": method, "axis": axis,
                    "n": int(len(vals)),
                    "p10": float(np.percentile(vals, 10)),
                    "p50": float(np.percentile(vals, 50)),
                    "p90": float(np.percentile(vals, 90)),
                })
        msig = pd.DataFrame(rows)
        msig.to_csv(OUTPUTS / "d3_training_method_iq_signature.csv", index=False)
        logger.info("wrote d3_training_method_iq_signature.csv (%d rows)", len(msig))

    # Console headlines.
    print("\n" + "=" * 80)
    print("D3 — Training-pool IQ summary (lap_var p50 across pools)")
    print("=" * 80)
    if "lap_var" in avail_axes and len(pool_summary) > 0:
        lap = pool_summary[pool_summary["axis"] == "lap_var"][["pool", "n", "p50"]]
        print(lap.sort_values("p50").to_string(index=False))

    print("\n" + "=" * 80)
    print("D3 — Real vs Fake IQ asymmetry per training pool (lap_var)")
    print("=" * 80)
    if 'rvf' in locals() and len(rvf) > 0:
        sub = rvf[rvf["axis"] == "lap_var"][["pool", "n_real", "n_fake",
                                              "real_p50", "fake_p50", "delta_p50"]]
        pd.options.display.float_format = "{:.2f}".format
        print(sub.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
