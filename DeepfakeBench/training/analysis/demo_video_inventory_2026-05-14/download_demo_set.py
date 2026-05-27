"""Sample the demo set per the 2026-05-14 spec, then download it locally.

Spec:
- Drop all CSCS (clean + every restorer).
- For each of the 8 remaining swap models: 100 random "none" (no restorer) videos
  drawn from union of manifest + bucket fallback.
- For each of the 8 swap models: ALL videos for each of the 7 restorer combos
  (manifest-only, since bucket fallback has no restorer labels).
- Deeplive: 100 random "regular" + 100 random "enhanced", fake.mp4 only.
- Reals: 400 random HDTF, 386 random QuickClips (cap; bucket has 386 mp4s).

Output:
  dataset/demo_videos_2026-05-14/
    visomaster/{swap}/{restorer-or-none}/{base_capture_id_or_sample_id}.mp4
    deeplive/{regular,enhanced}/{sample_id}__fake.mp4
    reals/{hdtf,quickclips}/{real_id}.mp4
    MANIFEST.csv
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
INV_DIR = ROOT / "analysis" / "demo_video_inventory_2026-05-14"
OUT_DIR = ROOT / "dataset" / "demo_videos_2026-05-14"

SEED = 737
RNG = random.Random(SEED)

EXCLUDE_SWAP = {"cscs"}
NONE_PER_SWAP = 100
DEEPLIVE_PER_METHOD = 100
HDTF_REALS = 400
QCLIP_REALS_CAP = 400  # will be capped at actual bucket size

PARALLELISM = 32


# ---------------------------------------------------------------------------
# Build the download plan
# ---------------------------------------------------------------------------


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def plan_visomaster() -> list[tuple[str, str, str, str]]:
    """Return list of (gcs_uri, local_path, method, source_id).

    For "none": sample 100 per swap from union of manifest-clean + bucket-fallback.
    For restorer combos: take ALL from manifest-clean.
    """
    items: list[tuple[str, str, str, str]] = []

    manifest_rows = read_csv(INV_DIR / "visomaster_videos.csv")
    bucket_rows = read_csv(INV_DIR / "visomaster_bucket_videos.csv")

    # Group manifest clean rows by (swap, restorer)
    manifest_by_key: dict[tuple[str, str], list[dict]] = {}
    for r in manifest_rows:
        if r["transport"] != "clean":
            continue
        if r["swap_model"] in EXCLUDE_SWAP:
            continue
        key = (r["swap_model"], r["restorer"])
        manifest_by_key.setdefault(key, []).append(r)

    # Group bucket-fallback rows by swap (these are all "none" / no-restorer)
    bucket_by_swap: dict[str, list[dict]] = {}
    for r in bucket_rows:
        swap = r["method"].split("__", 1)[1]
        if swap in EXCLUDE_SWAP:
            continue
        bucket_by_swap.setdefault(swap, []).append(r)

    # Build per-(swap, restorer) entries
    swap_models = sorted(set(k[0] for k in manifest_by_key.keys()))
    for swap in swap_models:
        # NONE: sample 100 from union of (uri, method, source_id, local_filename)
        # For manifest entries, source_id is base_capture_id and filename is "<base>.mp4".
        # For bucket-fallback entries, source_id is the sample_id (which ends in _NNNNN)
        # and the URI basename is just "fake.mp4" -- must rename to "<sample_id>__fake.mp4".
        none_pool: list[tuple[str, str, str, str]] = []
        for r in manifest_by_key.get((swap, "none"), []):
            none_pool.append(
                (r["video_mp4_gcs_uri"], r["method"], r["base_capture_id"], f"{r['base_capture_id']}.mp4")
            )
        for r in bucket_by_swap.get(swap, []):
            none_pool.append(
                (r["fake_mp4_gcs_uri"], r["method"], r["sample_id"], f"{r['sample_id']}__fake.mp4")
            )
        if len(none_pool) < NONE_PER_SWAP:
            print(f"WARN: {swap} none pool only has {len(none_pool)}, taking all", file=sys.stderr)
            picks = none_pool
        else:
            picks = RNG.sample(none_pool, NONE_PER_SWAP)
        for uri, method, src, fname in picks:
            local = OUT_DIR / "visomaster" / swap / "none" / fname
            items.append((uri, str(local), method, src))

        # Restorers: take all
        restorers = sorted({k[1] for k in manifest_by_key.keys() if k[0] == swap and k[1] != "none"})
        for rest in restorers:
            rows = manifest_by_key.get((swap, rest), [])
            for r in rows:
                local = OUT_DIR / "visomaster" / swap / rest / Path(r["video_mp4_gcs_uri"]).name
                items.append((r["video_mp4_gcs_uri"], str(local), r["method"], r["base_capture_id"]))

    return items


def plan_deeplive() -> list[tuple[str, str, str, str]]:
    items: list[tuple[str, str, str, str]] = []
    rows = read_csv(INV_DIR / "deeplive_videos.csv")
    regular = [r for r in rows if r["method"] == "deeplive_regular"]
    enhanced = [r for r in rows if r["method"] == "deeplive_enhanced"]
    print(f"deeplive_regular pool: {len(regular)}", file=sys.stderr)
    print(f"deeplive_enhanced pool: {len(enhanced)}", file=sys.stderr)
    reg_picks = RNG.sample(regular, DEEPLIVE_PER_METHOD)
    enh_picks = RNG.sample(enhanced, DEEPLIVE_PER_METHOD)
    for r in reg_picks:
        local = OUT_DIR / "deeplive" / "regular" / f"{r['sample_id']}__fake.mp4"
        items.append((r["fake_mp4_gcs_uri"], str(local), "deeplive_regular", r["sample_id"]))
    for r in enh_picks:
        local = OUT_DIR / "deeplive" / "enhanced" / f"{r['sample_id']}__fake.mp4"
        items.append((r["fake_mp4_gcs_uri"], str(local), "deeplive_enhanced", r["sample_id"]))
    return items


def list_bucket_mp4s(uri_prefix: str) -> list[str]:
    """Return all .mp4 URIs under uri_prefix."""
    result = subprocess.run(
        ["gsutil", "ls", f"{uri_prefix}*.mp4"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [ln.strip() for ln in result.stdout.splitlines() if ln.strip().endswith(".mp4")]


def plan_reals() -> list[tuple[str, str, str, str]]:
    items: list[tuple[str, str, str, str]] = []

    print("Listing HDTF reals...", file=sys.stderr)
    hdtf = list_bucket_mp4s("gs://hdtf_visomaster_videos/reals/")
    print(f"  {len(hdtf)} HDTF reals", file=sys.stderr)
    hdtf_picks = RNG.sample(hdtf, HDTF_REALS) if len(hdtf) >= HDTF_REALS else hdtf
    for uri in hdtf_picks:
        local = OUT_DIR / "reals" / "hdtf" / Path(uri).name
        items.append((uri, str(local), "real_hdtf", Path(uri).stem))

    print("Listing QuickClips reals...", file=sys.stderr)
    qclip = list_bucket_mp4s("gs://quickclips_visomaster_videos/reals/")
    print(f"  {len(qclip)} QCLIP reals", file=sys.stderr)
    target = min(QCLIP_REALS_CAP, len(qclip))
    qclip_picks = RNG.sample(qclip, target) if len(qclip) >= target else qclip
    for uri in qclip_picks:
        local = OUT_DIR / "reals" / "quickclips" / Path(uri).name
        items.append((uri, str(local), "real_quickclips", Path(uri).stem))

    return items


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


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


def download_all(items: list[tuple[str, str, str, str]]) -> int:
    n = len(items)
    print(f"Downloading {n} files with parallelism={PARALLELISM}...", file=sys.stderr)
    failed: list[tuple[str, str]] = []
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLELISM) as ex:
        futs = {ex.submit(download_one, uri, local): (uri, local) for uri, local, _, _ in items}
        for fut in as_completed(futs):
            uri, ok, msg = fut.result()
            done += 1
            if not ok:
                failed.append((uri, msg))
            if done % 100 == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  {done}/{n} ({rate:.1f}/s, {elapsed:.0f}s elapsed, {len(failed)} failed)", file=sys.stderr)
    if failed:
        print(f"FAILURES: {len(failed)}", file=sys.stderr)
        for uri, msg in failed[:10]:
            print(f"  {uri}: {msg}", file=sys.stderr)
    return len(failed)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def write_manifest(items: list[tuple[str, str, str, str]]) -> None:
    path = OUT_DIR / "MANIFEST.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["local_path", "method", "source_id", "gcs_uri"])
        for uri, local, method, src in items:
            rel = Path(local).relative_to(OUT_DIR)
            w.writerow([str(rel), method, src, uri])
    print(f"wrote {path} ({len(items)} rows)", file=sys.stderr)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    viso = plan_visomaster()
    deeplive = plan_deeplive()
    reals = plan_reals()
    items = viso + deeplive + reals

    print(f"plan: visomaster={len(viso)}, deeplive={len(deeplive)}, reals={len(reals)}, total={len(items)}", file=sys.stderr)

    write_manifest(items)

    if "--plan-only" in sys.argv:
        print("plan-only mode; not downloading", file=sys.stderr)
        return

    nfail = download_all(items)
    if nfail:
        sys.exit(2)
    print("done.", file=sys.stderr)


if __name__ == "__main__":
    main()
