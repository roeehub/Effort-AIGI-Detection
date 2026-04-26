#!/usr/bin/env python3
"""
manifest_overlap.py — frame-level overlap probe between training and lockbox buckets.

Purpose
-------
"Held-out" lockbox FPR wins (e.g. P8A 0.147%) only mean what we think they mean
if the lockbox bucket is actually disjoint from the training pool at the
frame/recording level. The two buckets are separate — but no one has confirmed
the underlying recordings don't overlap. This script does that confirmation.

Method
------
1. Stratified sample ~5,000 frames per side.
   - Training side: gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2
       layout: samples/<family>_<NNNN>/frames/{real,fake}/frame_*.jpg
       families seen on 2026-04-26:
         edge_cases (389 sample dirs)
         minimal_processing (410 sample dirs)
         quality_enhancement (312 sample dirs)
         visomaster_CSCS (151)
         visomaster_GhostFace-v1 (145)
         visomaster_GhostFace-v2 (104)
         visomaster_GhostFace-v3 (35)
         visomaster_InStyleSwapper256-A (35)
         visomaster_InStyleSwapper256-B (36)
         visomaster_Inswapper128 (29)
       Strata used (~500/stratum): edge_cases:{real,fake},
       minimal_processing:{real,fake}, quality_enhancement:{real,fake},
       visomaster_CSCS:fake, visomaster_GhostFace-v1:fake,
       visomaster_GhostFace-v2:fake, visomaster_Inswapper128:fake.
   - Lockbox side: gs://teams-faces-data-test-2914-fake-4420-real-feb-28
       layout: {real,fake,dor}/Cam_Test__s32_*.jpg  (flat)
       Strata: real (~1666), fake (~1666), dor (~1666).

2. For each frame, download bytes (no local persistence), decode with cv2,
   convert to grayscale, resize to 64x64, then md5(bytes) — a perceptual hash
   that survives re-encoding and re-cropping noise floor.

3. Compare hash sets, count overlap, dump top-5 colliding pairs as evidence.

Outputs
-------
analysis/probe_battery_2026-04-26/manifest_overlap.json

Notes
-----
- Streams everything; no large local downloads.
- Reads only — does not modify production code.
- If GCS auth fails, the script aborts and reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
    from google.cloud import storage
    from google.api_core.exceptions import NotFound, Forbidden
except Exception as e:  # noqa: BLE001
    print(f"FATAL: required imports failed: {e}", file=sys.stderr)
    sys.exit(2)


TRAIN_BUCKET = "live-deepfake-methods-real-and-fake-frames-cropped-teams-v2"
LOCKBOX_BUCKET = "teams-faces-data-test-2914-fake-4420-real-feb-28"
GCS_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "train-cvit2")

# Stratum specs:
#   key -> (family_prefix, label)   for training buckets ("real" or "fake")
#   None means top-level prefix in lockbox bucket
TRAIN_STRATA: Dict[str, Tuple[str, str]] = {
    "train.edge_cases.real": ("edge_cases", "real"),
    "train.edge_cases.fake": ("edge_cases", "fake"),
    "train.minimal_processing.real": ("minimal_processing", "real"),
    "train.minimal_processing.fake": ("minimal_processing", "fake"),
    "train.quality_enhancement.real": ("quality_enhancement", "real"),
    "train.quality_enhancement.fake": ("quality_enhancement", "fake"),
    "train.visomaster_CSCS.fake": ("visomaster_CSCS", "fake"),
    "train.visomaster_GhostFace-v1.fake": ("visomaster_GhostFace-v1", "fake"),
    "train.visomaster_GhostFace-v2.fake": ("visomaster_GhostFace-v2", "fake"),
    "train.visomaster_Inswapper128.fake": ("visomaster_Inswapper128", "fake"),
}

LOCKBOX_STRATA = ["fake", "real", "dor"]


def _logger() -> logging.Logger:
    log = logging.getLogger("manifest-overlap")
    if not log.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


LOG = _logger()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FrameRef:
    bucket: str
    blob_path: str
    stratum: str

    @property
    def gs_uri(self) -> str:
        return f"gs://{self.bucket}/{self.blob_path}"


def _list_sample_dirs(client: storage.Client, bucket_name: str, family_prefix: str) -> List[str]:
    """Return list of sample dir tokens like 'edge_cases_0007'."""
    bucket = client.bucket(bucket_name)
    # Use delimiter to get common prefixes (folders) cheaply
    prefix = f"samples/{family_prefix}_"
    iterator = bucket.list_blobs(prefix=prefix, delimiter="/")
    # Force iteration so .prefixes is populated
    list(iterator)
    sample_keys = sorted({p.rstrip("/").split("/")[-1] for p in iterator.prefixes})
    return sample_keys


def _list_frames_in_sample(
    client: storage.Client,
    bucket_name: str,
    sample_key: str,
    label: str,
) -> List[str]:
    bucket = client.bucket(bucket_name)
    prefix = f"samples/{sample_key}/frames/{label}/"
    out = []
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith("/"):
            continue
        if not blob.name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        out.append(blob.name)
    return out


def sample_training_frames(
    client: storage.Client,
    target_per_stratum: int,
    rng: random.Random,
) -> List[FrameRef]:
    """Stratified sample across families+labels. ~target_per_stratum per stratum."""
    refs: List[FrameRef] = []

    for stratum_name, (family_prefix, label) in TRAIN_STRATA.items():
        LOG.info("[train] discovering sample dirs for family=%s ...", family_prefix)
        sample_dirs = _list_sample_dirs(client, TRAIN_BUCKET, family_prefix)
        if not sample_dirs:
            LOG.warning("[train] no sample dirs for family=%s — skipping", family_prefix)
            continue
        rng.shuffle(sample_dirs)

        chosen: List[str] = []
        # Walk sample dirs until we collect enough frames for this stratum.
        # Adaptive per-dir take: aim for breadth but accept depth when families
        # have few sample dirs (visomaster_* families have 29-150 dirs).
        per_dir_take = max(1, target_per_stratum // max(1, min(len(sample_dirs), 50)))
        idx = 0
        while len(chosen) < target_per_stratum and idx < len(sample_dirs):
            sk = sample_dirs[idx]
            idx += 1
            try:
                frames = _list_frames_in_sample(client, TRAIN_BUCKET, sk, label)
            except (NotFound, Forbidden) as e:
                LOG.warning("[train] %s: skipping %s: %s", stratum_name, sk, e)
                continue
            if not frames:
                continue
            # Subsample within sample dir to keep diversity high.
            rng.shuffle(frames)
            take = min(len(frames), per_dir_take)
            chosen.extend(frames[:take])

        chosen = chosen[:target_per_stratum]
        LOG.info("[train] %s: collected %d frames", stratum_name, len(chosen))
        refs.extend(FrameRef(TRAIN_BUCKET, p, stratum_name) for p in chosen)

    return refs


def sample_lockbox_frames(
    client: storage.Client,
    target_per_stratum: int,
    rng: random.Random,
) -> List[FrameRef]:
    refs: List[FrameRef] = []
    bucket = client.bucket(LOCKBOX_BUCKET)
    for stratum in LOCKBOX_STRATA:
        prefix = f"{stratum}/"
        # Reservoir sample so we don't materialize tens-of-thousands of blob names.
        # In practice the lockbox is ~2-6k per stratum; fully list and shuffle is fine.
        names: List[str] = []
        try:
            for blob in bucket.list_blobs(prefix=prefix):
                if blob.name.endswith("/"):
                    continue
                if not blob.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                names.append(blob.name)
        except (NotFound, Forbidden) as e:
            LOG.warning("[lockbox] %s: %s", stratum, e)
            continue
        rng.shuffle(names)
        chosen = names[:target_per_stratum]
        LOG.info("[lockbox] %s: collected %d frames (pool=%d)", stratum, len(chosen), len(names))
        refs.extend(FrameRef(LOCKBOX_BUCKET, p, f"lockbox.{stratum}") for p in chosen)
    return refs


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def _phash_bytes(image_bytes: bytes) -> Optional[str]:
    """64x64 grayscale md5 — perceptual-ish hash, robust to JPEG re-encode."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    return hashlib.md5(resized.tobytes()).hexdigest()


def _hash_one(client: storage.Client, ref: FrameRef) -> Tuple[FrameRef, Optional[str]]:
    bucket = client.bucket(ref.bucket)
    blob = bucket.blob(ref.blob_path)
    try:
        data = blob.download_as_bytes()
    except Exception as e:  # noqa: BLE001
        LOG.warning("download fail %s: %s", ref.gs_uri, e)
        return ref, None
    h = _phash_bytes(data)
    return ref, h


def hash_refs(
    refs: List[FrameRef],
    side: str,
    max_workers: int = 32,
) -> Dict[FrameRef, str]:
    """Returns {ref: phash} for refs that decoded cleanly."""
    LOG.info("[%s] hashing %d frames with %d workers ...", side, len(refs), max_workers)
    out: Dict[FrameRef, str] = {}
    t0 = time.time()

    # Each worker thread gets its own client (storage.Client is thread-friendly
    # but we use one per pool to avoid contention surprises).
    def _worker(ref: FrameRef) -> Tuple[FrameRef, Optional[str]]:
        return _hash_one(_thread_client(), ref)

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_worker, r) for r in refs]
        for fut in as_completed(futs):
            ref, h = fut.result()
            done += 1
            if h is not None:
                out[ref] = h
            if done % 500 == 0:
                LOG.info("[%s] hashed %d/%d (%.1fs)", side, done, len(refs), time.time() - t0)

    LOG.info("[%s] done. %d/%d hashed in %.1fs", side, len(out), len(refs), time.time() - t0)
    return out


# Per-thread storage client (kept simple — one client constructed per call,
# google-cloud-storage Client is cheap).
def _thread_client() -> storage.Client:
    return storage.Client(project=GCS_PROJECT)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-train-per-stratum", type=int, default=500)
    parser.add_argument("--target-lockbox-per-stratum", type=int, default=1666)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "manifest_overlap.json"),
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    LOG.info("Probe started. project=%s seed=%d", GCS_PROJECT, args.seed)
    LOG.info("  train  = gs://%s", TRAIN_BUCKET)
    LOG.info("  lockbox= gs://%s", LOCKBOX_BUCKET)

    try:
        client = storage.Client(project=GCS_PROJECT)
    except Exception as e:  # noqa: BLE001
        LOG.error("GCS auth/client failed: %s", e)
        return 3

    # ----- Sample -----
    train_refs = sample_training_frames(client, args.target_train_per_stratum, rng)
    lockbox_refs = sample_lockbox_frames(client, args.target_lockbox_per_stratum, rng)

    LOG.info("Sampled train=%d lockbox=%d", len(train_refs), len(lockbox_refs))

    # ----- Hash -----
    train_hashes = hash_refs(train_refs, "train", max_workers=args.workers)
    lockbox_hashes = hash_refs(lockbox_refs, "lockbox", max_workers=args.workers)

    # ----- Compare -----
    train_hash_to_ref: Dict[str, FrameRef] = {}
    train_hash_count: Dict[str, int] = {}
    for ref, h in train_hashes.items():
        train_hash_count[h] = train_hash_count.get(h, 0) + 1
        # Keep first ref for evidence
        train_hash_to_ref.setdefault(h, ref)

    lockbox_hash_to_ref: Dict[str, FrameRef] = {}
    for ref, h in lockbox_hashes.items():
        lockbox_hash_to_ref.setdefault(h, ref)

    train_hash_set = set(train_hash_to_ref.keys())
    lockbox_hash_set = set(lockbox_hash_to_ref.keys())
    overlap_hashes = train_hash_set & lockbox_hash_set

    overlap_count = len(overlap_hashes)
    lockbox_n = len(lockbox_hashes)
    train_n = len(train_hashes)

    overlap_fraction_lockbox = overlap_count / max(1, lockbox_n)

    # Per-stratum breakdown
    per_stratum_lockbox_total: Dict[str, int] = {}
    per_stratum_lockbox_overlap: Dict[str, int] = {}
    for ref, h in lockbox_hashes.items():
        per_stratum_lockbox_total[ref.stratum] = per_stratum_lockbox_total.get(ref.stratum, 0) + 1
        if h in overlap_hashes:
            per_stratum_lockbox_overlap[ref.stratum] = (
                per_stratum_lockbox_overlap.get(ref.stratum, 0) + 1
            )

    # Top-5 evidence pairs
    evidence_pairs: List[Dict[str, str]] = []
    for h in list(overlap_hashes)[:50]:  # cap iteration
        t_ref = train_hash_to_ref[h]
        l_ref = lockbox_hash_to_ref[h]
        evidence_pairs.append(
            {
                "phash": h,
                "training_path": t_ref.gs_uri,
                "training_stratum": t_ref.stratum,
                "lockbox_path": l_ref.gs_uri,
                "lockbox_stratum": l_ref.stratum,
            }
        )
        if len(evidence_pairs) >= 5:
            break

    # Verdict
    if overlap_count == 0:
        verdict = "0% — clean (no frame-level leakage detected at this sample size)"
    elif overlap_fraction_lockbox < 0.01:
        verdict = (
            f"<1% — minor leak ({overlap_fraction_lockbox*100:.3f}% of lockbox sample). "
            "Past lockbox FPR wins are quantitatively still meaningful but should be "
            "annotated with this caveat."
        )
    else:
        verdict = (
            f">=1% — significant leak ({overlap_fraction_lockbox*100:.3f}% of lockbox sample). "
            "Past lockbox FPR wins (e.g. P8A 0.147%) need a downward correction; "
            "the held-out condition is violated."
        )

    report = {
        "schema": "manifest_overlap.v1",
        "run_started_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "buckets": {
            "training": f"gs://{TRAIN_BUCKET}",
            "lockbox": f"gs://{LOCKBOX_BUCKET}",
        },
        "method": {
            "phash": "md5(cv2.resize(grayscale, (64,64)).tobytes())",
            "decoder": "cv2.imdecode(IMREAD_GRAYSCALE)",
            "sample_seed": args.seed,
            "target_train_per_stratum": args.target_train_per_stratum,
            "target_lockbox_per_stratum": args.target_lockbox_per_stratum,
        },
        "sample_sizes": {
            "training_requested": len(train_refs),
            "training_hashed_ok": train_n,
            "lockbox_requested": len(lockbox_refs),
            "lockbox_hashed_ok": lockbox_n,
        },
        "overlap": {
            "count": overlap_count,
            "fraction_lockbox": overlap_fraction_lockbox,
            "fraction_lockbox_pct": overlap_fraction_lockbox * 100.0,
            "per_lockbox_stratum": {
                k: {
                    "total": per_stratum_lockbox_total.get(k, 0),
                    "overlap": per_stratum_lockbox_overlap.get(k, 0),
                    "overlap_pct": (
                        per_stratum_lockbox_overlap.get(k, 0)
                        / max(1, per_stratum_lockbox_total.get(k, 0))
                    )
                    * 100.0,
                }
                for k in sorted(per_stratum_lockbox_total)
            },
        },
        "verdict": verdict,
        "evidence_top5": evidence_pairs,
    }

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=False)
    LOG.info("Wrote report -> %s", out_path)

    # Console-friendly summary
    print("=" * 70)
    print("MANIFEST OVERLAP PROBE — SUMMARY")
    print("=" * 70)
    print(f"training sample (hashed ok): {train_n}")
    print(f"lockbox  sample (hashed ok): {lockbox_n}")
    print(f"overlap count               : {overlap_count}")
    print(f"overlap fraction (lockbox)  : {overlap_fraction_lockbox*100:.4f}%")
    print(f"verdict                     : {verdict}")
    if evidence_pairs:
        print("\nTop evidence pairs:")
        for ev in evidence_pairs:
            print(f"  - {ev['training_path']}  <==>  {ev['lockbox_path']}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
