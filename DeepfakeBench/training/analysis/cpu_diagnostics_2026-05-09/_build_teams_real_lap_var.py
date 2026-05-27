"""
Compute per-frame lap_var for ALL Teams-passthrough real frames at the active
training anchor indices [0, 2, 4, 6, 8, 10, 12, 14] in
gs://live-deepfake-methods-real-and-fake-frames-cropped-teams.

Output cache: analysis/cpu_diagnostics_2026-05-09/_cache/teams_real_uri_lap_var.parquet
Columns: frame_uri, lap_var, sample_id, strategy, frame_idx
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _compute_iq import measure_uris, list_blobs  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("teams_real_iq")

BUCKET = "live-deepfake-methods-real-and-fake-frames-cropped-teams"
ANCHOR_INDICES = (0, 2, 4, 6, 8, 10, 12, 14)


def main() -> int:
    cache_path = ROOT / "analysis/cpu_diagnostics_2026-05-09/_cache/teams_real_uri_lap_var.parquet"
    sample_dirs_path = ROOT / "analysis/cpu_diagnostics_2026-05-09/_cache/teams_real_sample_dirs.txt"

    # Step 1 — discover all sample IDs (sample_dirs / "<id>/frames/real/").
    if sample_dirs_path.exists():
        sample_ids = [s.strip() for s in sample_dirs_path.read_text().splitlines() if s.strip()]
        logger.info("Loaded %d sample ids from cache", len(sample_ids))
    else:
        from google.cloud import storage
        client = storage.Client(project="train-cvit2")
        bucket = client.bucket(BUCKET)
        manifest_blobs = list(bucket.list_blobs(prefix="samples/", match_glob="**/manifest.json"))
        sample_ids = []
        for blob in manifest_blobs:
            # samples/<id>/manifest.json
            m = re.match(r"samples/([^/]+)/manifest\.json$", blob.name)
            if not m:
                continue
            sample_ids.append(m.group(1))
        sample_dirs_path.parent.mkdir(parents=True, exist_ok=True)
        sample_dirs_path.write_text("\n".join(sample_ids))
        logger.info("Discovered %d sample dirs", len(sample_ids))

    # Step 2 — build the URI list for each sample × anchor index.
    uris = []
    sample_meta = []  # parallel list with [(sample_id, strategy, frame_idx)]
    for sid in sample_ids:
        # Strategy is the leading token before the underscore-NNNN.
        # e.g. edge_cases_0007 -> "edge_cases"; minimal_processing_0010 -> "minimal_processing"
        m = re.match(r"^(.*)_(\d{4})$", sid)
        strategy = m.group(1) if m else "unknown"
        for idx in ANCHOR_INDICES:
            uri = f"gs://{BUCKET}/samples/{sid}/frames/real/frame_{idx:04d}.jpg"
            uris.append(uri)
            sample_meta.append((sid, strategy, idx))

    logger.info("Total Teams-real URIs at anchor indices: %d", len(uris))

    # Step 3 — measure lap_var (cached) and join metadata.
    measured = measure_uris(
        uris=uris,
        cache_path=cache_path.with_name("teams_real_uri_lap_var__raw.parquet"),
        cache_key="frame_uri",
        n_workers=32,
    )

    import pandas as pd
    base = pd.DataFrame(uris, columns=["frame_uri"])
    base["sample_id"] = [m[0] for m in sample_meta]
    base["strategy"] = [m[1] for m in sample_meta]
    base["frame_idx"] = [m[2] for m in sample_meta]
    out = base.merge(measured, on="frame_uri", how="left")
    out = out.dropna(subset=["lap_var"]).reset_index(drop=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path, index=False)
    logger.info(
        "Wrote %s (%d frames; lap_var p10=%.2f / p50=%.2f / p90=%.2f)",
        cache_path, len(out),
        out["lap_var"].quantile(0.10),
        out["lap_var"].quantile(0.50),
        out["lap_var"].quantile(0.90),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
