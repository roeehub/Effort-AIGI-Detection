"""
Slot 3 — Per-method IQ-matched reals.

For each fake method M:
  - Compute [p10_M, p90_M] from M's training fake lap_var distribution.
  - Build the kept set of reals (FF++ identities + teams_real frames) whose
    lap_var falls in [p10_M, p90_M] for THAT method.

DF40 side: produce a UNIFIED filtered pair JSON where each pair satisfies
its method's window. Teams side: produce a per-method keep-list CSV with
columns (method, frame_uri, lap_var); the loader's
`teams_real_frame_keep_list_per_method` field activates it.

Outputs:
  - dataset/df40_pairs/df40-pair-matching__per_method_iq_2026-05-09.json
  - analysis/cpu_diagnostics_2026-05-09/slot3_teams_real_per_method_keep_list_2026-05-09.csv

Random state = 9103 (no random sampling actually used; included for spec
compliance and any tie-breaks).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("slot3")

RANDOM_STATE = 9103
ACTIVE_DF40_METHODS = (
    "simswap", "facedancer", "blendface", "e4s",
    "inswap", "mobileswap", "uniface",
)


def load_method_window(method: str) -> Tuple[float, float]:
    """Return (p10, p90) for a given fake method's lap_var distribution.

    Sources (see _load_*_lap_var helpers below):
      - DF40 methods: atlas parquet `train_df40_<m>.parquet` (or our mobileswap cache).
      - deeplive_teams_<strategy>: teams_fake_uri_lap_var.parquet grouped by method.
    """
    df = _load_method_lap_var(method)
    if df is None or len(df) == 0:
        raise RuntimeError(f"No fake lap_var data for method={method}")
    return float(df["lap_var"].quantile(0.10)), float(df["lap_var"].quantile(0.90))


def _load_method_lap_var(method: str) -> pd.DataFrame:
    cache_dir = ROOT / "analysis/iq_data_atlas_2026-05-08/_cache"
    if method in {"simswap", "facedancer", "blendface", "e4s", "inswap", "uniface"}:
        return pd.read_parquet(cache_dir / f"train_df40_{method}.parquet")
    if method == "mobileswap":
        return pd.read_parquet(
            ROOT / "analysis/cpu_diagnostics_2026-05-09/_cache/train_df40_mobileswap_iq.parquet"
        )
    if method.startswith("deeplive_teams_"):
        df = pd.read_parquet(
            ROOT / "analysis/cpu_diagnostics_2026-05-09/_cache/teams_fake_uri_lap_var.parquet"
        )
        return df[df["method"] == method]
    raise ValueError(f"Unknown method: {method}")


def main() -> int:
    pair_json_path = ROOT / "dataset/df40_pairs/df40-pair-matching.json"
    out_pair_json = ROOT / "dataset/df40_pairs/df40-pair-matching__per_method_iq_2026-05-09.json"
    out_keep_csv = ROOT / "analysis/cpu_diagnostics_2026-05-09/slot3_teams_real_per_method_keep_list_2026-05-09.csv"

    ff_df = pd.read_parquet(ROOT / "analysis/cpu_diagnostics_2026-05-09/ff_per_identity_lap_var_2026-05-09.parquet")
    teams_df = pd.read_parquet(ROOT / "analysis/cpu_diagnostics_2026-05-09/_cache/teams_real_uri_lap_var.parquet")
    teams_df = teams_df.dropna(subset=["lap_var"]).reset_index(drop=True)
    logger.info("FF++ identities: %d, Teams real frames: %d", len(ff_df), len(teams_df))

    # 1) DF40 pair filtering.
    with open(pair_json_path, "r") as f:
        data = json.load(f)

    method_windows: Dict[str, Tuple[float, float]] = {}
    for m in ACTIVE_DF40_METHODS:
        method_windows[m] = load_method_window(m)
        logger.info("Method %s window: [%.2f, %.2f]", m, *method_windows[m])

    # Build identity -> lap_var lookup from FF++ measurements.
    id_lv = dict(zip(ff_df["identity"], ff_df["lap_var"]))

    new_pairs = []
    method_kept: Dict[str, int] = {m: 0 for m in ACTIVE_DF40_METHODS}
    method_total: Dict[str, int] = {m: 0 for m in ACTIVE_DF40_METHODS}
    for pair in data["pairs"]:
        m = pair["method"]
        if m not in ACTIVE_DF40_METHODS:
            new_pairs.append(pair)  # unaffected non-active methods preserved
            continue
        method_total[m] += 1
        real = pair["real"]
        ident = f"df40_{real['source']}_{real['identity']}"
        lv = id_lv.get(ident)
        if lv is None:
            continue
        lo, hi = method_windows[m]
        if lo <= lv <= hi:
            new_pairs.append(pair)
            method_kept[m] += 1

    new_data = dict(data)
    new_data["pairs"] = new_pairs
    new_summary = dict(data.get("summary", {}))
    new_summary["pairs_per_method__after_per_method_iq_2026-05-09"] = method_kept
    new_summary["total_pairs__after_per_method_iq_2026-05-09"] = sum(method_kept.values())
    new_data["summary"] = new_summary
    new_data.setdefault("filters", []).append({
        "filter": "per_method_iq_2026-05-09",
        "random_state": RANDOM_STATE,
        "method_windows": {m: list(w) for m, w in method_windows.items()},
        "method_kept": method_kept,
        "method_total": method_total,
    })
    out_pair_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pair_json, "w") as f:
        json.dump(new_data, f, indent=2)
        f.write("\n")
    logger.info("Wrote %s (%d pairs total)", out_pair_json, len(new_pairs))

    # 2) Per-method teams keep-list.
    teams_methods_present = sorted(
        m for m in pd.read_parquet(
            ROOT / "analysis/cpu_diagnostics_2026-05-09/_cache/teams_fake_uri_lap_var.parquet"
        )["method"].unique()
    )
    rows = []
    teams_kept_summary = {}
    for tm in teams_methods_present:
        lo, hi = load_method_window(tm)
        method_windows[tm] = (lo, hi)
        sub = teams_df[(teams_df["lap_var"] >= lo) & (teams_df["lap_var"] <= hi)]
        sub = sub.sort_values(["frame_uri"]).reset_index(drop=True)
        teams_kept_summary[tm] = len(sub)
        for _, row in sub.iterrows():
            rows.append({
                "method": tm,
                "frame_uri": row["frame_uri"],
                "lap_var": float(row["lap_var"]),
            })
        logger.info(
            "Method %s window=[%.2f,%.2f] teams_real kept=%d / %d",
            tm, lo, hi, len(sub), len(teams_df),
        )

    out_csv_df = pd.DataFrame(rows, columns=["method", "frame_uri", "lap_var"])
    out_csv_df = out_csv_df.sort_values(["method", "frame_uri"]).reset_index(drop=True)
    out_keep_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv_df.to_csv(out_keep_csv, index=False, lineterminator="\n", float_format="%.6f")
    logger.info("Wrote %s (%d rows)", out_keep_csv, len(out_csv_df))

    print("====== SLOT 3 SUMMARY ======")
    print("DF40 per-method retention:")
    for m in ACTIVE_DF40_METHODS:
        kept = method_kept[m]
        total = method_total[m]
        pct = 100.0 * kept / max(total, 1)
        flag = " <100" if kept < 100 else ""
        print(f"  {m:14s} {kept:5d}/{total:5d}  ({pct:5.1f}%){flag}")
    print("Teams per-method retention (real-frame side):")
    for m in teams_methods_present:
        n = teams_kept_summary[m]
        flag = " <100" if n < 100 else ""
        print(f"  {m:38s} {n:5d}/{len(teams_df):5d}{flag}")
    print(f"output_pair_json:        {out_pair_json}")
    print(f"output_teams_keep_csv:   {out_keep_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
