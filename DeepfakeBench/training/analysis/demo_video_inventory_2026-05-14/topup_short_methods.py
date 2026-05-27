"""Top up the visomaster 'none' methods that came up short after the initial
download because the bucket-fallback source has many empty sample dirs in
certain ranges.

For each affected swap, list the actual existing fake.mp4 files via gsutil
wildcard, subtract what's already downloaded, sample the gap, download, and
append new entries to MANIFEST.csv.
"""

from __future__ import annotations

import csv
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT_DIR = ROOT / "dataset" / "demo_videos_2026-05-14"
MANIFEST = OUT_DIR / "MANIFEST.csv"
PARALLELISM = 32
SEED = 738  # different from initial run so top-up sampling is independent

# (local-dir swap name, bucket-side prefix)
AFFECTED = [
    ("instyleswapper256_b", "visomaster_InStyleSwapper256-B"),
    ("instyleswapper256_c", "visomaster_InStyleSwapper256-C"),
    ("inswapper128", "visomaster_Inswapper128"),
    ("simswap512", "visomaster_SimSwap512"),
]

TARGET = 100  # final per-method count for 'none'


def list_valid_fakes(bucket_prefix: str) -> list[str]:
    """Return all existing fake.mp4 URIs for a given bucket prefix."""
    pattern = f"gs://live-deepfake-methods-real-and-fake-videos/samples/{bucket_prefix}_*/fake.mp4"
    res = subprocess.run(
        ["gsutil", "ls", pattern],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0 and "matched no objects" not in (res.stderr or ""):
        print(f"WARN: ls failed for {bucket_prefix}: {res.stderr[:200]}", file=sys.stderr)
    uris = [ln.strip() for ln in res.stdout.splitlines() if ln.strip().endswith("/fake.mp4")]
    return uris


def already_downloaded_sample_ids(swap: str) -> set[str]:
    d = OUT_DIR / "visomaster" / swap / "none"
    if not d.exists():
        return set()
    ids = set()
    for p in d.glob("*__fake.mp4"):
        # filename: visomaster_GhostFace-v1_02544__fake.mp4 -> sample_id = visomaster_GhostFace-v1_02544
        ids.add(p.name[: -len("__fake.mp4")])
    return ids


def sample_id_from_uri(uri: str) -> str:
    # gs://bucket/samples/<sample_id>/fake.mp4
    return uri.rsplit("/", 2)[-2]


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
    rng = random.Random(SEED)
    new_rows: list[tuple[str, str, str, str]] = []  # (local_rel, method, source_id, uri)
    download_jobs: list[tuple[str, str]] = []

    for swap, bucket_prefix in AFFECTED:
        present = already_downloaded_sample_ids(swap)
        # Also count manifest-origin files in the dir (e.g. HDTF20260416_*.mp4)
        # — they don't follow the __fake.mp4 pattern, so we count via dir listing.
        d = OUT_DIR / "visomaster" / swap / "none"
        existing = len(list(d.glob("*.mp4"))) if d.exists() else 0
        gap = TARGET - existing
        print(f"{swap}: have {existing}, gap {gap}", file=sys.stderr)
        if gap <= 0:
            continue

        valid = list_valid_fakes(bucket_prefix)
        valid_ids = [sample_id_from_uri(u) for u in valid]
        pairs = [(sid, u) for sid, u in zip(valid_ids, valid) if sid not in present]
        print(f"  valid fake.mp4 in bucket: {len(valid)}; not yet downloaded: {len(pairs)}", file=sys.stderr)
        if not pairs:
            print(f"  WARN: no candidates available for {swap}", file=sys.stderr)
            continue
        n = min(gap, len(pairs))
        picks = rng.sample(pairs, n) if len(pairs) >= n else pairs
        method = f"visomaster_bucket_clean__{swap}"
        for sid, uri in picks:
            fname = f"{sid}__fake.mp4"
            local = OUT_DIR / "visomaster" / swap / "none" / fname
            local_rel = str(local.relative_to(OUT_DIR))
            new_rows.append((local_rel, method, sid, uri))
            download_jobs.append((uri, str(local)))

    print(f"total top-up downloads: {len(download_jobs)}", file=sys.stderr)
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
            if done % 50 == 0 or done == len(download_jobs):
                elapsed = time.time() - t0
                print(f"  {done}/{len(download_jobs)} ({elapsed:.0f}s, {len(failed)} failed)", file=sys.stderr)

    # Append successful rows to manifest
    successful = {uri for uri, _ in download_jobs} - {uri for uri, _ in failed}
    appended = 0
    with MANIFEST.open("a", newline="") as f:
        w = csv.writer(f)
        for local_rel, method, sid, uri in new_rows:
            if uri in successful:
                w.writerow([local_rel, method, sid, uri])
                appended += 1
    print(f"appended {appended} rows to MANIFEST.csv ({len(failed)} failed)", file=sys.stderr)
    if failed:
        for uri, msg in failed[:5]:
            print(f"  fail: {uri}: {msg}", file=sys.stderr)


if __name__ == "__main__":
    main()
