"""
Compute per-frame lap_var for ALL Teams-passthrough FAKE frames at the active
training anchor indices, broken down by strategy. Used for Slot 3 per-method
windows.

Output cache: analysis/cpu_diagnostics_2026-05-09/_cache/teams_fake_uri_lap_var.parquet
Columns: frame_uri, lap_var, sample_id, strategy, frame_idx, method (= "deeplive_teams_<strategy>")
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _compute_iq import measure_uris  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("teams_fake_iq")

BUCKET = "live-deepfake-methods-real-and-fake-frames-cropped-teams"
ANCHOR_INDICES = (0, 2, 4, 6, 8, 10, 12, 14)


def main() -> int:
    cache_path = ROOT / "analysis/cpu_diagnostics_2026-05-09/_cache/teams_fake_uri_lap_var.parquet"
    sample_dirs_path = ROOT / "analysis/cpu_diagnostics_2026-05-09/_cache/teams_real_sample_dirs.txt"

    # Reuse the sample-id list from the real-side discovery.
    if not sample_dirs_path.exists():
        raise FileNotFoundError(
            f"Run _build_teams_real_lap_var.py first to populate {sample_dirs_path}"
        )
    sample_ids = [s.strip() for s in sample_dirs_path.read_text().splitlines() if s.strip()]
    logger.info("Sample dirs: %d", len(sample_ids))

    uris = []
    sample_meta = []
    for sid in sample_ids:
        m = re.match(r"^(.*)_(\d{4})$", sid)
        strategy = m.group(1) if m else "unknown"
        for idx in ANCHOR_INDICES:
            uri = f"gs://{BUCKET}/samples/{sid}/frames/fake/frame_{idx:04d}.jpg"
            uris.append(uri)
            sample_meta.append((sid, strategy, idx))

    logger.info("Total Teams-fake URIs at anchor indices: %d", len(uris))

    measured = measure_uris(
        uris=uris,
        cache_path=cache_path.with_name("teams_fake_uri_lap_var__raw.parquet"),
        cache_key="frame_uri",
        n_workers=32,
    )

    import pandas as pd
    base = pd.DataFrame(uris, columns=["frame_uri"])
    base["sample_id"] = [m[0] for m in sample_meta]
    base["strategy"] = [m[1] for m in sample_meta]
    base["frame_idx"] = [m[2] for m in sample_meta]
    base["method"] = "deeplive_teams_" + base["strategy"]
    out = base.merge(measured, on="frame_uri", how="left")
    out = out.dropna(subset=["lap_var"]).reset_index(drop=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path, index=False)
    logger.info("Wrote %s (%d rows)", cache_path, len(out))
    summary = out.groupby("method")["lap_var"].agg(
        ["count",
         lambda s: s.quantile(0.10),
         lambda s: s.quantile(0.50),
         lambda s: s.quantile(0.90)],
    )
    summary.columns = ["n", "p10", "p50", "p90"]
    print(summary.round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
