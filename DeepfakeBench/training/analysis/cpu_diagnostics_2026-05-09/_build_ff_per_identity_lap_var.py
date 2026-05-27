"""
Compute per-identity lap_var for all FF++ paired-real identities used in the
active DF40 pair JSON. One representative frame per identity is downloaded
from gs://df40-frames-recropped-rfa85.

Output: analysis/cpu_diagnostics_2026-05-09/ff_per_identity_lap_var_2026-05-09.parquet
Columns: identity, source, lap_var, frame_uri
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _compute_iq import measure_uris  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ff_per_id")


def main() -> int:
    pair_json = ROOT / "dataset/df40_pairs/df40-pair-matching.json"
    out_path = ROOT / "analysis/cpu_diagnostics_2026-05-09/ff_per_identity_lap_var_2026-05-09.parquet"
    cache_path = ROOT / "analysis/cpu_diagnostics_2026-05-09/_cache/ff_identity_uri_lap_var.parquet"

    with open(pair_json, "r") as f:
        data = json.load(f)

    # Active methods used by training (per baseline R13_S2 yaml). Identities are
    # FF++-only by atlas verification, but we filter to the active method set
    # so we never compute for unused real identities.
    active_methods = {
        "simswap", "facedancer", "blendface", "e4s",
        "inswap", "mobileswap", "uniface",
    }
    identities = {}
    for pair in data["pairs"]:
        if pair["method"] not in active_methods:
            continue
        real = pair["real"]
        # Identity key: source + identity-id (matches DF40 identity scheme).
        ident = f"df40_{real['source']}_{real['identity']}"
        if ident in identities:
            continue
        # Pick the first frame of the real prefix as the representative.
        if not real.get("frames"):
            continue
        first_frame = real["frames"][0]
        # `path` is the directory; the actual frame URI is path + frame name.
        path = real["path"]
        if not path.endswith("/"):
            path = path + "/"
        frame_uri = path + first_frame
        identities[ident] = {
            "identity": ident,
            "source": real["source"],
            "frame_uri": frame_uri,
        }

    rows = list(identities.values())
    logger.info("Total active FF++ identities: %d", len(rows))

    uris = [r["frame_uri"] for r in rows]
    measured = measure_uris(
        uris=uris,
        cache_path=cache_path,
        cache_key="frame_uri",
        n_workers=32,
    )

    # Join measured back with identity info.
    import pandas as pd
    base = pd.DataFrame(rows)
    out = base.merge(measured, on="frame_uri", how="left")
    out = out.dropna(subset=["lap_var"]).reset_index(drop=True)

    out.to_parquet(out_path, index=False)
    logger.info(
        "Wrote %s (%d identities, lap_var p10=%.2f / p50=%.2f / p90=%.2f)",
        out_path, len(out),
        out["lap_var"].quantile(0.10),
        out["lap_var"].quantile(0.50),
        out["lap_var"].quantile(0.90),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
