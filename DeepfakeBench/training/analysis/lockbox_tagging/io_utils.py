"""Manifest + predictions loaders, GCS frame fetcher with on-disk cache."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
MANIFEST = REPO / "arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json"
PRED_CSV = REPO / "inference_results/r9a_run1__teams-faces-data-test-2914-fake-4420-real-feb-28.csv"
CACHE_ROOT = REPO / "analysis/lockbox_tagging/_frame_cache"


@dataclass(frozen=True)
class FrameRow:
    gcs_uri: str
    blob_path: str
    bucket: str
    label: str  # "real" | "fake"
    split: str  # "lockbox" | "dev" | "unknown"
    identity_key: str | None
    session_id: str | None
    video_id: str
    method: str
    prob_fake: float | None  # may be None if frame is not in predictions CSV


def _split_gs(uri: str) -> tuple[str, str]:
    assert uri.startswith("gs://"), uri
    rest = uri[len("gs://"):]
    bucket, _, blob = rest.partition("/")
    return bucket, blob


def load_manifest_index() -> dict[str, dict]:
    """gcs_uri -> {identity_key, session_id, video_id, method, label, split}."""
    with open(MANIFEST) as f:
        manifest = json.load(f)
    idx: dict[str, dict] = {}
    for v in manifest["videos"]:
        meta = {
            "identity_key": v.get("identity_key"),
            "session_id": v.get("session_id"),
            "video_id": v["video_id"],
            "method": v.get("method"),
            "label": v["label"],
            "split": v["split"],
        }
        for fp in v["frame_paths"]:
            idx[fp] = meta
    return idx


def load_predictions() -> dict[str, dict]:
    """gcs_uri -> {prob_fake, label_str, method, video_id, status}."""
    out: dict[str, dict] = {}
    with open(PRED_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                p = float(row["prob_fake"]) if row["prob_fake"] else None
            except ValueError:
                p = None
            out[row["gcs_uri"]] = {
                "prob_fake": p,
                "label_str": row["label_str"],
                "method": row["method"],
                "video_id": row["video_id"],
                "status": row["status"],
                "blob_path": row["blob_path"],
                "bucket": row["bucket"],
            }
    return out


def select_frames(
    scope: str = "lockbox_with_preds",
    limit: int | None = None,
    seed: int = 0,
) -> list[FrameRow]:
    """Return frames to tag.

    scope:
      - "lockbox_with_preds": lockbox frames that also have a prediction (default; 839 frames)
      - "all_preds":          every frame in predictions CSV (~7334)
      - "lockbox_all":        every lockbox frame in manifest (1843, no prediction join)

    When `limit` is set, the candidate list is deterministically shuffled (seed=`seed`)
    before truncation, so the sample is representative across identities rather than
    alphabetically clustered.
    """
    import random

    manifest_idx = load_manifest_index()
    preds = load_predictions()
    rows: list[FrameRow] = []
    if scope == "lockbox_with_preds":
        candidates = [u for u, m in manifest_idx.items() if m["split"] == "lockbox" and u in preds]
    elif scope == "lockbox_all":
        candidates = [u for u, m in manifest_idx.items() if m["split"] == "lockbox"]
    elif scope == "all_preds":
        candidates = list(preds.keys())
    else:
        raise ValueError(f"unknown scope {scope!r}")

    candidates.sort()
    if limit is not None:
        rng = random.Random(seed)
        rng.shuffle(candidates)
        candidates = candidates[:limit]
        candidates.sort()  # re-sort the sampled subset so output ordering is deterministic
    for u in candidates:
        m = manifest_idx.get(u, {})
        p = preds.get(u, {})
        bucket, blob = _split_gs(u)
        rows.append(
            FrameRow(
                gcs_uri=u,
                blob_path=p.get("blob_path") or blob,
                bucket=p.get("bucket") or bucket,
                label=m.get("label") or p.get("label_str") or "unknown",
                split=m.get("split", "unknown"),
                identity_key=m.get("identity_key"),
                session_id=m.get("session_id"),
                video_id=m.get("video_id") or p.get("video_id") or "",
                method=m.get("method") or p.get("method") or "",
                prob_fake=p.get("prob_fake"),
            )
        )
    return rows


def _cache_path(blob_path: str) -> Path:
    h = hashlib.md5(blob_path.encode()).hexdigest()
    return CACHE_ROOT / h[:2] / h[2:4] / Path(blob_path).name


def fetch_frame(row: FrameRow) -> Path:
    """Download (once) and return local cache path for a frame."""
    target = _cache_path(row.blob_path)
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    # Lazy import so module is cheap to import.
    from google.cloud import storage  # type: ignore

    client = _gcs_client()
    bucket = client.bucket(row.bucket)
    blob = bucket.blob(row.blob_path)
    tmp = target.with_suffix(target.suffix + ".part")
    blob.download_to_filename(str(tmp))
    os.replace(tmp, target)
    return target


_gcs_client_singleton = None


def _gcs_client():
    global _gcs_client_singleton
    if _gcs_client_singleton is None:
        from google.cloud import storage  # type: ignore

        _gcs_client_singleton = storage.Client()
    return _gcs_client_singleton


def fetch_frames_parallel(rows: Sequence[FrameRow], workers: int = 16) -> list[Path]:
    """Download all frames in parallel; return local paths in input order."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fetch_frame, rows))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="lockbox_with_preds")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    rows = select_frames(scope=args.scope, limit=args.limit)
    print(f"selected {len(rows)} frames")
    if rows:
        print(f"first: {rows[0]}")
        print(f"last:  {rows[-1]}")
