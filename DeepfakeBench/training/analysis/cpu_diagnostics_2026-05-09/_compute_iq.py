"""
Shared IQ computation utilities for slot1/slot2/slot3 sidecar builders.

Computes per-image lap_var (variance of Laplacian on grayscale) using the
SAME formula as the IQ atlas builder:
    cv2.Laplacian(gray, cv2.CV_64F).var()

GCS downloads use google-cloud-storage with thread-pool parallelism.
Aggressively caches per-frame measurements as parquet so re-runs are fast.
"""
from __future__ import annotations

import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from google.cloud import storage

logger = logging.getLogger("iq_compute")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _split_uri(uri: str) -> Tuple[str, str]:
    assert uri.startswith("gs://"), f"Bad URI: {uri}"
    rest = uri[len("gs://"):]
    bucket, _, blob = rest.partition("/")
    return bucket, blob


def _get_client(project: str = "train-cvit2") -> storage.Client:
    return storage.Client(project=project)


def _decode_lap_var(buf: bytes) -> Optional[float]:
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def measure_uris(
    uris: Sequence[str],
    project: str = "train-cvit2",
    n_workers: int = 24,
    cache_path: Optional[Path] = None,
    cache_key: str = "frame_uri",
) -> pd.DataFrame:
    """Measure lap_var for each gs:// URI. Returns DataFrame[frame_uri, lap_var]."""
    cache_df: Optional[pd.DataFrame] = None
    if cache_path is not None and cache_path.exists():
        cache_df = pd.read_parquet(cache_path)
        logger.info("Loaded cache %s with %d rows", cache_path, len(cache_df))

    seen = set(cache_df[cache_key].tolist()) if cache_df is not None else set()
    todo = [u for u in uris if u not in seen]
    logger.info(
        "Measuring %d URIs (%d cached, %d new)",
        len(uris), len(uris) - len(todo), len(todo),
    )
    if not todo:
        return cache_df if cache_df is not None else pd.DataFrame(
            {cache_key: list(uris), "lap_var": [np.nan] * len(uris)}
        )

    client = _get_client(project)
    bucket_cache: Dict[str, storage.Bucket] = {}

    def _measure_one(uri: str) -> Tuple[str, Optional[float]]:
        try:
            b, blob_path = _split_uri(uri)
            if b not in bucket_cache:
                bucket_cache[b] = client.bucket(b)
            buf = bucket_cache[b].blob(blob_path).download_as_bytes()
            return uri, _decode_lap_var(buf)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Measure failed for %s: %s", uri, exc)
            return uri, None

    rows: List[Tuple[str, Optional[float]]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="iq-meas") as pool:
        futures = {pool.submit(_measure_one, u): u for u in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rows.append(fut.result())
            except Exception as exc:
                logger.debug("future error: %s", exc)
            if i % 500 == 0 or i == len(todo):
                rate = i / max(time.time() - t0, 1e-3)
                eta = (len(todo) - i) / max(rate, 1e-3)
                logger.info("  progress %d/%d (%.1f/s, eta %.0fs)", i, len(todo), rate, eta)

    new_df = pd.DataFrame(rows, columns=[cache_key, "lap_var"])
    if cache_df is not None:
        out = pd.concat([cache_df, new_df], ignore_index=True)
        out = out.drop_duplicates(subset=[cache_key], keep="last")
    else:
        out = new_df

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path, index=False)
        logger.info("Wrote cache %s (%d rows)", cache_path, len(out))

    return out


def list_blobs(
    bucket: str,
    prefix: str,
    project: str = "train-cvit2",
    suffix: Optional[str] = None,
) -> List[str]:
    client = _get_client(project)
    out: List[str] = []
    for blob in client.bucket(bucket).list_blobs(prefix=prefix):
        if suffix is None or blob.name.endswith(suffix):
            out.append(f"gs://{bucket}/{blob.name}")
    return out
