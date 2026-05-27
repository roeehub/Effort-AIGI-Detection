"""
Compute lap_var for ~500 mobileswap fake frames (atlas missing). One frame
per pair. Output to atlas-style cache parquet.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _compute_iq import measure_uris  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mobileswap_iq")


def main() -> int:
    pair_json = ROOT / "dataset/df40_pairs/df40-pair-matching.json"
    out_path = ROOT / "analysis/cpu_diagnostics_2026-05-09/_cache/train_df40_mobileswap_iq.parquet"

    with open(pair_json, "r") as f:
        data = json.load(f)
    uris = []
    for pair in data["pairs"]:
        if pair["method"] != "mobileswap":
            continue
        fake = pair["fake"]
        path = fake["path"]
        if not path.endswith("/"):
            path = path + "/"
        # Pick first frame from the pair real list (fake frames mirror real frame names).
        frames = pair["real"]["frames"]
        if not frames:
            continue
        first = frames[0]
        uris.append(path + first)
    logger.info("Found %d mobileswap pair URIs", len(uris))

    measured = measure_uris(
        uris=uris,
        cache_path=out_path,
        cache_key="frame_uri",
        n_workers=32,
    )
    logger.info(
        "mobileswap lap_var: N=%d p10=%.2f p50=%.2f p90=%.2f",
        len(measured),
        measured["lap_var"].quantile(0.10),
        measured["lap_var"].quantile(0.50),
        measured["lap_var"].quantile(0.90),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
