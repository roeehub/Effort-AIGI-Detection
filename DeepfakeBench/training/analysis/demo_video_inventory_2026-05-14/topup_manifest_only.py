"""Top up the two visomaster 'none' methods (instyleswapper256_c, simswap512)
that have NO valid fake.mp4 in the bucket fallback.

For these, the only source is the proper_visomaster manifest (HDTF/QuickClips
clean fakes). Sample everything not already downloaded.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
INV_DIR = ROOT / "analysis" / "demo_video_inventory_2026-05-14"
OUT_DIR = ROOT / "dataset" / "demo_videos_2026-05-14"
MANIFEST = OUT_DIR / "MANIFEST.csv"
PARALLELISM = 16

AFFECTED_SWAPS = {"instyleswapper256_c", "simswap512"}


def existing_files(swap: str) -> set[str]:
    d = OUT_DIR / "visomaster" / swap / "none"
    if not d.exists():
        return set()
    return {p.name for p in d.glob("*.mp4")}


def download_one(gcs_uri: str, local_path: str) -> tuple[str, bool, str]:
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(local_path).exists() and Path(local_path).stat().st_size > 0:
        return (gcs_uri, True, "skip")
    res = subprocess.run(
        ["gsutil", "-q", "cp", gcs_uri, local_path],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        return (gcs_uri, False, res.stderr.strip()[:200])
    return (gcs_uri, True, "ok")


def main() -> None:
    new_rows: list[tuple[str, str, str, str]] = []
    download_jobs: list[tuple[str, str]] = []

    with (INV_DIR / "visomaster_videos.csv").open() as f:
        reader = list(csv.DictReader(f))

    for swap in AFFECTED_SWAPS:
        present = existing_files(swap)
        candidates = [
            r for r in reader
            if r["transport"] == "clean"
            and r["swap_model"] == swap
            and r["restorer"] == "none"
            and Path(r["video_mp4_gcs_uri"]).name not in present
        ]
        print(f"{swap}: have {len(present)}, manifest candidates not yet downloaded: {len(candidates)}", file=sys.stderr)
        for r in candidates:
            fname = Path(r["video_mp4_gcs_uri"]).name
            local = OUT_DIR / "visomaster" / swap / "none" / fname
            local_rel = str(local.relative_to(OUT_DIR))
            new_rows.append((local_rel, r["method"], r["base_capture_id"], r["video_mp4_gcs_uri"]))
            download_jobs.append((r["video_mp4_gcs_uri"], str(local)))

    print(f"total downloads: {len(download_jobs)}", file=sys.stderr)
    if not download_jobs:
        return

    failed: list[tuple[str, str]] = []
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLELISM) as ex:
        futs = {ex.submit(download_one, uri, local): (uri, local) for uri, local in download_jobs}
        for fut in as_completed(futs):
            uri, ok, msg = fut.result()
            done += 1
            if not ok:
                failed.append((uri, msg))
    elapsed = time.time() - t0
    print(f"  {done}/{len(download_jobs)} ({elapsed:.0f}s, {len(failed)} failed)", file=sys.stderr)

    successful = {uri for uri, _ in download_jobs} - {uri for uri, _ in failed}
    appended = 0
    with MANIFEST.open("a", newline="") as f:
        w = csv.writer(f)
        for local_rel, method, sid, uri in new_rows:
            if uri in successful:
                w.writerow([local_rel, method, sid, uri])
                appended += 1
    print(f"appended {appended} rows to MANIFEST.csv", file=sys.stderr)
    if failed:
        for uri, msg in failed[:5]:
            print(f"  fail: {uri}: {msg}", file=sys.stderr)


if __name__ == "__main__":
    main()
