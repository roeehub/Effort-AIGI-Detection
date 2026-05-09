"""
Build the Slot 1 Teams REAL keep-list: bottom 75% of train_teams_real frames
ranked by per-frame lap_var. Removes the very-sharp = real signal.

Output:
  analysis/cpu_diagnostics_2026-05-09/slot1_teams_real_keep_list_2026-05-09.csv
  Columns: frame_uri, lap_var

Idempotent: re-running produces byte-identical output (deterministic sort).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("slot1")


def main() -> int:
    cache_path = ROOT / "analysis/cpu_diagnostics_2026-05-09/_cache/teams_real_uri_lap_var.parquet"
    out_path = ROOT / "analysis/cpu_diagnostics_2026-05-09/slot1_teams_real_keep_list_2026-05-09.csv"

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Run _build_teams_real_lap_var.py first to produce {cache_path}"
        )

    df = pd.read_parquet(cache_path)
    df = df.dropna(subset=["lap_var"]).reset_index(drop=True)
    n_in = len(df)
    logger.info("Input: %d teams_real frames", n_in)

    # Drop top 25% by lap_var. Keep bottom 75%.
    threshold = float(df["lap_var"].quantile(0.75))
    keep = df[df["lap_var"] <= threshold].copy()
    keep = keep.sort_values(["frame_uri"]).reset_index(drop=True)
    logger.info(
        "Threshold (p75 of lap_var) = %.3f; keeping %d / %d frames (%.1f%%)",
        threshold, len(keep), n_in, 100.0 * len(keep) / max(n_in, 1),
    )

    # Compute KS vs active fakes (pooled).
    ks_stat = _compute_ks_vs_fakes(keep["lap_var"].to_numpy())
    logger.info("KS(real_kept_lap_var, active_fake_lap_var) = %.4f", ks_stat)

    # Slot 1 fake distribution is unchanged (we did not filter fakes).
    out_df = keep[["frame_uri", "lap_var"]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, lineterminator="\n", float_format="%.6f")
    logger.info("Wrote %s (%d rows)", out_path, len(out_df))

    print("====== SLOT 1 SUMMARY ======")
    print(f"input_teams_real_frames:   {n_in}")
    print(f"output_teams_real_frames:  {len(out_df)} ({100*len(out_df)/n_in:.1f}%)")
    print(f"drop_threshold_lap_var:    {threshold:.3f}")
    print(f"KS(real_kept, active_fake): {ks_stat:.4f}")
    print(f"output_path:               {out_path}")
    return 0


def _compute_ks_vs_fakes(real_lap_var: np.ndarray) -> float:
    """KS between kept reals and ACTIVE training fake distribution from atlas.

    Atlas coverage of active training fakes (verified 2026-05-09): 6 DF40 methods
    + train_teams_fake_pool + visomaster_enhanced_v2_all are present;
    train_df40_mobileswap and the deeplive non-enhanced training pool are NOT
    in the atlas. KS is computed against the present subset (~3,500 fake frames).
    """
    from scipy.stats import ks_2samp
    fakes = load_active_fake_lap_var()
    if len(fakes) == 0:
        return float("nan")
    ks = ks_2samp(real_lap_var, fakes)
    return float(ks.statistic)


def load_active_fake_lap_var() -> np.ndarray:
    fake_pools = [
        # 6 of 7 active DF40 methods present in the atlas (mobileswap missing).
        "train_df40_simswap", "train_df40_facedancer", "train_df40_blendface",
        "train_df40_e4s", "train_df40_inswap", "train_df40_uniface",
        # Active teams + visomaster fake training pools.
        "train_teams_fake_pool",
        "visomaster_enhanced_v2_all",
    ]
    cache_dir = ROOT / "analysis/iq_data_atlas_2026-05-08/_cache"
    arrs = []
    for m in fake_pools:
        p = cache_dir / f"{m}.parquet"
        if not p.exists():
            logger.warning("Missing atlas pool: %s", p)
            continue
        d = pd.read_parquet(p)
        if "lap_var" in d.columns:
            arrs.append(d["lap_var"].dropna().to_numpy())
    if not arrs:
        return np.array([], dtype=float)
    return np.concatenate(arrs)


if __name__ == "__main__":
    raise SystemExit(main())
