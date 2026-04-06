#!/usr/bin/env python3
"""
Batch inference via HTTP API on GCS test buckets — per-frame CSV output.

Streams frames from GCS with concurrent downloads, sends to check_frame_batch
API with parallel consumers, writes CSV rows as results arrive.

Architecture:
  ThreadPoolExecutor(12) → bounded Queue(200) → 3 API worker threads → CSV

All in-memory — no temp files. Natural backpressure via bounded queue.

Usage:
  python batch_inference_api.py \
    --buckets poc-phase-1-test \
              teams-faces-data-test-2914-fake-4420-real-feb-28 \
              live-deepfake-methods-real-and-fake-frames-cropped-teams \
    --output_dir ./inference_results

  # Resume a previously interrupted run (skips frames already in CSV):
  python batch_inference_api.py --buckets poc-phase-1-test --resume
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from google.cloud import storage

logger = logging.getLogger("batch-inference-api")

# =============================================================================
# Data structures
# =============================================================================
@dataclass
class FrameRecord:
    bucket: str
    blob_path: str
    label: int              # 0=real, 1=fake
    method: str
    video_id: str
    frame_name: str
    strategy: str = ""

# Sentinel to signal all downloads are done
_DONE = object()

# =============================================================================
# GCS bucket discovery
# =============================================================================
def _extract_method_from_folder(folder_name: str) -> str:
    known = [
        "Deeplivecam", "Face2Face", "FaceShifter", "FaceSwap",
        "MRAA", "NeuralTextures", "facevid2vid", "fomm", "fsgan",
        "hyperreenact", "inswap", "lia", "mcnet", "mobileswap",
        "oneshot", "pirender", "sadtalker", "simswap", "tpsm",
    ]
    for m in known:
        if folder_name.startswith(m + "_") or folder_name == m:
            return m
    return folder_name.split("_")[0]


def _extract_teams_segment(filename: str) -> str:
    try:
        idx_s32 = filename.index("s32_") + 4
        idx_frame = filename.index("_frame_")
        return f"seg_{filename[idx_s32:idx_frame]}"
    except (ValueError, IndexError):
        return "unknown"


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def discover_poc_phase1(client: storage.Client, bucket_name: str) -> List[FrameRecord]:
    records = []
    bucket = client.bucket(bucket_name)
    for label_str, label_int in [("fake", 1), ("real", 0)]:
        for blob in bucket.list_blobs(prefix=f"{label_str}/"):
            if blob.name.endswith("/"):
                continue
            if os.path.splitext(blob.name)[1].lower() not in IMAGE_EXTS:
                continue
            rel = blob.name[len(f"{label_str}/"):]
            parts = rel.split("/")
            if len(parts) < 2:
                continue
            folder_name = parts[0]
            method = _extract_method_from_folder(folder_name) if label_int == 1 else "real"
            records.append(FrameRecord(
                bucket=bucket_name, blob_path=blob.name, label=label_int,
                method=method, video_id=folder_name, frame_name=parts[-1],
            ))
    return records


def discover_teams_flat(client: storage.Client, bucket_name: str) -> List[FrameRecord]:
    records = []
    bucket = client.bucket(bucket_name)
    for label_str, label_int in [("fake", 1), ("real", 0)]:
        for blob in bucket.list_blobs(prefix=f"{label_str}/"):
            if blob.name.endswith("/"):
                continue
            if os.path.splitext(blob.name)[1].lower() not in IMAGE_EXTS:
                continue
            fname = os.path.basename(blob.name)
            records.append(FrameRecord(
                bucket=bucket_name, blob_path=blob.name, label=label_int,
                method="teams_passthrough" if label_int == 1 else "real",
                video_id=_extract_teams_segment(fname), frame_name=fname,
            ))
    return records


def discover_teams_paired(client: storage.Client, bucket_name: str) -> List[FrameRecord]:
    records = []
    bucket = client.bucket(bucket_name)
    manifests: Dict[str, dict] = {}
    frame_blobs: List[Tuple[str, str]] = []

    for blob in bucket.list_blobs(prefix="samples/"):
        if blob.name.endswith("/"):
            continue
        parts = blob.name.split("/")
        if len(parts) < 2:
            continue
        sample_key = parts[1]
        if blob.name.endswith("manifest.json"):
            try:
                manifests[sample_key] = json.loads(blob.download_as_text())
            except Exception:
                pass
        elif os.path.splitext(blob.name)[1].lower() in IMAGE_EXTS:
            frame_blobs.append((sample_key, blob.name))

    for sample_key, blob_name in frame_blobs:
        parts = blob_name.split("/")
        if "frames" not in parts:
            continue
        fi = parts.index("frames")
        if fi + 1 >= len(parts):
            continue
        label_str = parts[fi + 1]
        if label_str not in ("fake", "real"):
            continue
        label_int = 1 if label_str == "fake" else 0
        strategy = "_".join(sample_key.split("_")[:-1])
        records.append(FrameRecord(
            bucket=bucket_name, blob_path=blob_name, label=label_int,
            method=f"teams_{strategy}" if label_int == 1 else "real",
            video_id=sample_key, frame_name=parts[-1], strategy=strategy,
        ))
    return records


def discover_bucket(client: storage.Client, bucket_name: str) -> List[FrameRecord]:
    if bucket_name == "poc-phase-1-test":
        return discover_poc_phase1(client, bucket_name)
    elif bucket_name.startswith("teams-faces-data-test"):
        return discover_teams_flat(client, bucket_name)
    elif "live-deepfake-methods" in bucket_name:
        return discover_teams_paired(client, bucket_name)
    return discover_poc_phase1(client, bucket_name)


# =============================================================================
# Producer: concurrent GCS downloads → bounded queue (in-memory bytes)
# =============================================================================
def producer_download(
    client: storage.Client,
    records: List[FrameRecord],
    q: "queue.Queue",
    num_workers: int = 12,
):
    """
    Concurrent GCS downloads. Each completed download puts (idx, bytes, ext, err)
    into the bounded queue. q.put() blocks when queue is full, which blocks the
    pool worker thread — providing natural backpressure.
    """
    def download_and_enqueue(idx_rec: Tuple[int, FrameRecord]):
        idx, rec = idx_rec
        try:
            blob = client.bucket(rec.bucket).blob(rec.blob_path)
            data = blob.download_as_bytes()
            ext = os.path.splitext(rec.blob_path)[1] or ".jpg"
            q.put((idx, data, ext, None))  # blocks when queue full
        except Exception as e:
            q.put((idx, None, None, str(e)))

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        # Submit all tasks; only num_workers run concurrently.
        # Blocking q.put() inside each task provides backpressure.
        futures = [pool.submit(download_and_enqueue, (i, r))
                   for i, r in enumerate(records)]
        # Wait for all to finish
        for f in futures:
            f.result()

    q.put(_DONE)


# =============================================================================
# API call (in-memory, no disk)
# =============================================================================
def send_batch_to_api(
    api_url: str,
    batch_items: List[Tuple[str, bytes, str]],  # (filename, data, ext)
    threshold: float = 0.5,
    timeout: int = 120,
) -> List[Optional[float]]:
    """Send batch of in-memory images to API. Returns list of per-frame probs."""
    files = []
    for fname, data, ext in batch_items:
        mime = "image/png" if ext == ".png" else "image/jpeg"
        files.append(("files", (fname, io.BytesIO(data), mime)))

    try:
        resp = requests.post(
            api_url,
            params={"threshold": threshold},
            files=files,
            timeout=timeout,
        )
        resp.raise_for_status()
        probs = resp.json().get("probs", [])
        if len(probs) != len(batch_items):
            logger.warning(
                "API returned %d probs for %d files — padding with None",
                len(probs), len(batch_items),
            )
            probs.extend([None] * (len(batch_items) - len(probs)))
        return probs
    except Exception as e:
        logger.error("API error: %s", e)
        return [None] * len(batch_items)


# =============================================================================
# Consumer: API worker threads
# =============================================================================
def api_worker(
    worker_id: int,
    q: "queue.Queue",
    records: List[FrameRecord],
    csv_writer: csv.DictWriter,
    csv_lock: threading.Lock,
    stats: Dict[str, Any],
    args: argparse.Namespace,
):
    """
    Pull items from queue, accumulate a batch, send to API, write CSV rows.
    Multiple workers can run in parallel for higher throughput.
    """
    while True:
        batch_indices: List[int] = []
        batch_items: List[Tuple[str, bytes, str]] = []

        while len(batch_indices) < args.batch_size:
            try:
                item = q.get(timeout=300)
            except queue.Empty:
                logger.warning("Worker %d: queue timeout after 5 min", worker_id)
                break

            if item is _DONE:
                q.put(_DONE)  # re-signal for other workers
                break

            idx, data, ext, err = item

            if err or data is None:
                # Download failed — record immediately
                rec = records[idx]
                row = _make_row(rec, prob=None, status=f"download_error: {err}")
                with csv_lock:
                    csv_writer.writerow(row)
                    stats["fail"] += 1
                    stats["processed"] += 1
                continue

            batch_indices.append(idx)
            batch_items.append((records[idx].frame_name, data, ext))

        if not batch_indices:
            return  # No more work

        # Send batch to API
        probs = send_batch_to_api(args.api_url, batch_items, args.threshold)

        with csv_lock:
            for rec_idx, prob in zip(batch_indices, probs):
                rec = records[rec_idx]
                if prob is not None:
                    row = _make_row(rec, prob=prob, status="ok")
                    csv_writer.writerow(row)
                    stats["ok"] += 1
                else:
                    row = _make_row(rec, prob=None, status="api_error")
                    csv_writer.writerow(row)
                    stats["fail"] += 1
                stats["processed"] += 1


def _make_row(rec: FrameRecord, prob: Optional[float], status: str) -> Dict[str, str]:
    return {
        "bucket": rec.bucket,
        "gcs_uri": f"gs://{rec.bucket}/{rec.blob_path}",
        "blob_path": rec.blob_path,
        "label": str(rec.label),
        "label_str": "fake" if rec.label == 1 else "real",
        "method": rec.method,
        "video_id": rec.video_id,
        "frame_name": rec.frame_name,
        "strategy": rec.strategy,
        "prob_fake": f"{prob:.8f}" if prob is not None else "",
        "status": status,
    }


# =============================================================================
# Progress ticker
# =============================================================================
def progress_ticker(stats: Dict[str, Any], interval: float = 15.0):
    """Prints progress every `interval` seconds until stats["done"] is set."""
    while not stats.get("done"):
        time.sleep(interval)
        elapsed = time.time() - stats["start"]
        ok = stats["ok"]
        fail = stats["fail"]
        proc = stats["processed"]
        total = stats["total"]
        fps = ok / max(elapsed, 0.001)
        eta = (total - proc) / max(fps, 0.01)
        logger.info(
            "  %d/%d (%.1f%%) | ok=%d fail=%d | %.1f fps | elapsed %.0fs | ETA %.0fs",
            proc, total, 100 * proc / max(total, 1),
            ok, fail, fps, elapsed, eta,
        )


# =============================================================================
# CSV fields
# =============================================================================
CSV_FIELDS = [
    "bucket", "gcs_uri", "blob_path",
    "label", "label_str", "method", "video_id",
    "frame_name", "strategy", "prob_fake", "status",
]


# =============================================================================
# Resume support
# =============================================================================
def load_completed_blobs(csv_path: str) -> Set[str]:
    """Load blob_path values already in the CSV (for resume)."""
    completed: Set[str] = set()
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "ok":
                completed.add(row["blob_path"])
    return completed


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Batch inference via HTTP API on GCS test buckets"
    )
    parser.add_argument(
        "--buckets", nargs="+", required=True,
        help="GCS bucket names to process",
    )
    parser.add_argument(
        "--api_url", type=str,
        default="http://34.16.217.28:8999/check_frame_batch",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./inference_results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Frames per API request (default: 32)",
    )
    parser.add_argument(
        "--max_queue", type=int, default=200,
        help="Max items in download queue (bounds memory; each item ~150KB)",
    )
    parser.add_argument(
        "--download_workers", type=int, default=12,
        help="Concurrent GCS download threads (default: 12)",
    )
    parser.add_argument(
        "--api_workers", type=int, default=3,
        help="Concurrent API consumer threads (default: 3)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Classification threshold passed to API (affects pred_label only)",
    )
    parser.add_argument(
        "--run_id", type=str, default="",
        help="Optional run identifier for output naming",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip frames already present (status=ok) in existing CSV",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)

    # Quick API health check
    try:
        r = requests.get(
            args.api_url.replace("/check_frame_batch", "/health"), timeout=5,
        )
        logger.info("API health: %s", r.text[:200])
    except Exception:
        logger.warning("Could not reach API health endpoint — proceeding anyway")

    gcs_client = storage.Client()

    for bucket_name in args.buckets:
        logger.info("=" * 70)
        logger.info("Bucket: %s", bucket_name)
        logger.info("=" * 70)

        # CSV path
        safe_name = bucket_name.replace("/", "_")
        csv_path = os.path.join(args.output_dir, f"{run_id}__{safe_name}.csv")

        # Discover frames
        t0 = time.time()
        all_records = discover_bucket(gcs_client, bucket_name)
        if not all_records:
            logger.warning("No frames in %s — skipping", bucket_name)
            continue

        # Resume: filter out already-processed frames
        if args.resume:
            completed = load_completed_blobs(csv_path)
            if completed:
                logger.info("Resume: found %d already-processed frames in %s",
                            len(completed), csv_path)
                records = [r for r in all_records if r.blob_path not in completed]
                logger.info("Resume: %d remaining (skipped %d)",
                            len(records), len(all_records) - len(records))
            else:
                records = all_records
        else:
            records = all_records

        fake_n = sum(1 for r in records if r.label == 1)
        real_n = sum(1 for r in records if r.label == 0)
        methods = sorted(set(r.method for r in records))
        logger.info(
            "Discovered %d frames (fake=%d real=%d) %d methods in %.1fs",
            len(records), fake_n, real_n, len(methods), time.time() - t0,
        )
        logger.info("Methods: %s", ", ".join(methods))

        if not records:
            logger.info("Nothing to process — all frames already done")
            continue

        # Open CSV for streaming writes (append if resuming, else new)
        csv_mode = "a" if (args.resume and os.path.exists(csv_path)) else "w"
        csv_file = open(csv_path, csv_mode, newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if csv_mode == "w":
            csv_writer.writeheader()
        csv_lock = threading.Lock()

        # Bounded queue for download → API pipeline
        q: queue.Queue = queue.Queue(maxsize=args.max_queue)

        # Shared stats (protected by csv_lock)
        stats: Dict[str, Any] = {
            "ok": 0, "fail": 0, "processed": 0,
            "total": len(records), "start": time.time(), "done": False,
        }

        # Start progress ticker
        ticker = threading.Thread(
            target=progress_ticker, args=(stats, 15.0), daemon=True,
        )
        ticker.start()

        # Start producer (concurrent downloads)
        producer = threading.Thread(
            target=producer_download,
            args=(gcs_client, records, q, args.download_workers),
            daemon=True,
        )
        producer.start()

        # Start API consumer workers
        workers = []
        for i in range(args.api_workers):
            t = threading.Thread(
                target=api_worker,
                args=(i, q, records, csv_writer, csv_lock, stats, args),
                daemon=True,
            )
            t.start()
            workers.append(t)

        # Wait for all API workers to finish
        for t in workers:
            t.join()

        producer.join(timeout=10)

        stats["done"] = True
        csv_file.flush()
        csv_file.close()

        elapsed = time.time() - stats["start"]
        logger.info(
            "Done: %d ok, %d failed in %.1fs (%.1f fps)",
            stats["ok"], stats["fail"], elapsed,
            stats["ok"] / max(elapsed, 0.001),
        )
        logger.info("CSV → %s", csv_path)

        # Quick accuracy summary
        try:
            import numpy as np
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                rows_by_label: Dict[str, List[float]] = {"fake": [], "real": []}
                for row in reader:
                    if row["status"] == "ok" and row["prob_fake"]:
                        rows_by_label.get(row["label_str"], []).append(
                            float(row["prob_fake"])
                        )
            for lbl in ["fake", "real"]:
                probs = rows_by_label[lbl]
                if not probs:
                    continue
                correct = sum(1 for p in probs if (p >= 0.5) == (lbl == "fake"))
                logger.info(
                    "  %s: n=%d mean_prob=%.4f acc@0.5=%.3f",
                    lbl, len(probs), float(np.mean(probs)), correct / len(probs),
                )
        except Exception as e:
            logger.warning("Summary stats failed: %s", e)

    logger.info("All done.")


if __name__ == "__main__":
    main()
