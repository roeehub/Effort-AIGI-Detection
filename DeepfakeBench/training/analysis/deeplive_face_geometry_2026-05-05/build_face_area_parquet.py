"""Compute face_area_fraction for every frame in the 5 deeplive training strategies.

Output:
    analysis/deeplive_face_geometry_2026-05-05/face_area.parquet
    analysis/deeplive_face_geometry_2026-05-05/summary.json

Architecture:
    - Enumerate all frame URIs once via GCS list_blobs (one pass per strategy).
    - Parallelize compute via a process pool (≤8 workers; mediapipe holds GIL on CPU).
      Each worker downloads its own frame to a per-process temp directory, runs
      compute_face_geometry, deletes the temp file. We do NOT keep frames cached
      on disk — 64K PNGs would be ~30 GB.
    - The face mesh singleton lives per-worker; cv2 + mediapipe load once per process.

Hard rules:
    - Per project memory feedback_sklearn_njobs.md: never n_jobs=-1; use ≤8 workers.
    - On per-frame failure (decode crash / mediapipe crash), log + skip.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT_DIR = REPO / "analysis/deeplive_face_geometry_2026-05-05"
BUCKET = "live-deepfake-methods-real-and-fake-frames-cropped"

# Short-circuit gcloud project lookup. Without this, every storage.Client() call may fork a
# `gcloud config get project` subprocess (smoke-test 2026-05-05 hit 13+ concurrent gcloud
# forks before fix). Setting this env var makes storage.Client() pick the project directly.
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "train-cvit2")
STRATEGIES = (
    "edge_cases",
    "edge_cases_enhanced",
    "minimal_processing",
    "minimal_processing_enhanced",
    "quality_enhancement",
)

# Make the layers/ module importable.
sys.path.insert(0, str(REPO / "analysis"))


@dataclass(frozen=True)
class FrameJob:
    gcs_uri: str
    blob_path: str
    strategy: str
    label: int  # 0 real, 1 fake


# ---------- frame enumeration (main process) ----------

def enumerate_frames() -> list[FrameJob]:
    """List all blobs under samples/<strat>_NNNN/frames/{fake,real}/*.png for the 5 strategies."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(BUCKET)

    # Step 1: get the set of folder names under samples/ matching each strategy. One list_blobs
    # per call with delimiter='/' returns folder prefixes cheaply.
    print(f"[enumerate] listing folders under samples/ ...", flush=True)
    t0 = time.time()
    iterator = client.list_blobs(bucket, prefix="samples/", delimiter="/")
    folders: set[str] = set()
    for page in iterator.pages:
        folders.update(page.prefixes)
    print(f"[enumerate] {len(folders)} folders in {time.time()-t0:.1f}s", flush=True)

    # Bucket folders by strategy via regex (so 'edge_cases_0000' doesn't get pulled in by
    # 'edge_cases_enhanced' prefix and vice versa).
    pats = {s: re.compile(rf"^samples/{re.escape(s)}_(\d+)/$") for s in STRATEGIES}
    folders_by_strat: dict[str, list[str]] = {s: [] for s in STRATEGIES}
    for f in folders:
        for s, pat in pats.items():
            if pat.match(f):
                folders_by_strat[s].append(f)
                break

    for s in STRATEGIES:
        print(f"[enumerate]   {s}: {len(folders_by_strat[s])} folders", flush=True)

    # Step 2: list frames/{fake,real}/*.png for each folder. We do this in parallel threads
    # (network-bound, GIL is fine). 16 threads is reasonable for GCS list calls.
    #
    # IMPORTANT: reuse the parent `client` and `bucket` across threads. Instantiating a fresh
    # storage.Client() inside the worker forks a `gcloud config get project` subprocess per
    # call (smoke-test 2026-05-05 stalled on this — 2040 forks × seconds each). The Google
    # Cloud client is documented as thread-safe.
    from concurrent.futures import ThreadPoolExecutor

    def list_frames_in_folder(folder: str) -> list[tuple[str, int]]:
        # Returns list of (blob_path, label_int)
        out: list[tuple[str, int]] = []
        for label_str, label_int in (("fake", 1), ("real", 0)):
            prefix = folder + f"frames/{label_str}/"
            for b in client.list_blobs(bucket, prefix=prefix):
                if b.name.endswith(".png"):
                    out.append((b.name, label_int))
        return out

    print(f"[enumerate] listing frames per folder (16 threads) ...", flush=True)
    t1 = time.time()
    jobs: list[FrameJob] = []
    folder_to_strat: dict[str, str] = {}
    all_folders: list[str] = []
    for s, fs in folders_by_strat.items():
        for f in fs:
            folder_to_strat[f] = s
            all_folders.append(f)
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(list_frames_in_folder, f): f for f in all_folders}
        for i, fut in enumerate(as_completed(futures)):
            folder = futures[fut]
            strat = folder_to_strat[folder]
            for blob_path, label in fut.result():
                uri = f"gs://{BUCKET}/{blob_path}"
                jobs.append(FrameJob(uri, blob_path, strat, label))
            if (i + 1) % 200 == 0:
                print(f"[enumerate]   listed {i+1}/{len(all_folders)} folders, "
                      f"{len(jobs)} frames so far ({time.time()-t1:.1f}s)", flush=True)
    print(f"[enumerate] DONE — {len(jobs)} frames in {time.time()-t1:.1f}s", flush=True)
    return jobs


# ---------- per-worker state + compute ----------

# These are module-level singletons; each worker-process initializes them once.
_worker_storage_client = None
_worker_bucket = None
_worker_tempdir: Path | None = None
_worker_face_geom = None  # the compute_face_geometry function
_worker_pid = None


def _worker_init():
    """Process-pool initializer: warm the GCS client, mediapipe, mediapipe singleton, and tempdir."""
    global _worker_storage_client, _worker_bucket, _worker_tempdir, _worker_face_geom, _worker_pid
    from google.cloud import storage

    _worker_pid = os.getpid()
    _worker_storage_client = storage.Client()
    _worker_bucket = _worker_storage_client.bucket(BUCKET)
    _worker_tempdir = Path(tempfile.mkdtemp(prefix=f"deeplive_face_{_worker_pid}_"))

    # Warm the face_mesh singleton inside this process.
    from lockbox_tagging.layers.face_geometry import compute_face_geometry, _face_mesh
    _face_mesh()  # initialize MediaPipe FaceMesh once
    _worker_face_geom = compute_face_geometry


def _worker_process_one(job: FrameJob) -> dict:
    """Download → compute → delete. Returns a dict row.

    On any failure, returns a row with face_area_fraction=NaN and an error string.
    """
    out: dict = {
        "frame_path": job.gcs_uri,
        "face_area_fraction": float("nan"),
        "strategy": job.strategy,
        "label": job.label,
        "error": None,
    }
    local_path = _worker_tempdir / f"{_worker_pid}_{os.urandom(6).hex()}.png"
    try:
        try:
            blob = _worker_bucket.blob(job.blob_path)
            blob.download_to_filename(str(local_path))
        except Exception as e:
            out["error"] = f"download:{type(e).__name__}:{str(e)[:120]}"
            return out
        try:
            geom = _worker_face_geom(local_path)
        except Exception as e:
            out["error"] = f"compute:{type(e).__name__}:{str(e)[:120]}"
            return out
        ratio = geom.get("face_area_ratio")
        if ratio is None:
            # No face detected — represent as NaN.
            out["face_area_fraction"] = float("nan")
        else:
            out["face_area_fraction"] = float(ratio)
        return out
    finally:
        try:
            if local_path.exists():
                local_path.unlink()
        except Exception:
            pass


# ---------- main driver ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on jobs (for smoke-test).")
    ap.add_argument("--log-every", type=int, default=5000)
    ap.add_argument("--checkpoint-every", type=int, default=10000,
                    help="Persist a partial parquet every N completed frames.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_parquet = OUT_DIR / "face_area.parquet"
    out_summary = OUT_DIR / "summary.json"
    out_log = OUT_DIR / "run.log"

    log_fh = open(out_log, "w")

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    log(f"workers={args.workers} limit={args.limit}")
    log(f"output={out_parquet}")

    overall_t0 = time.time()

    # ---- enumerate ----
    jobs = enumerate_frames()
    if args.limit is not None:
        jobs = jobs[: args.limit]
    log(f"total frames to process: {len(jobs)}")

    # Pre-flight: counts per strategy × label
    counts: dict[tuple[str, int], int] = {}
    for j in jobs:
        counts[(j.strategy, j.label)] = counts.get((j.strategy, j.label), 0) + 1
    for (s, lbl), n in sorted(counts.items()):
        log(f"  {s} label={lbl}: {n}")

    # ---- compute ----
    rows: list[dict] = []
    error_rows: list[dict] = []
    n_total = len(jobs)
    t_compute_start = time.time()
    last_log_n = 0
    last_chk_n = 0

    log(f"spinning up process pool (workers={args.workers}) ...")
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as ex:
        # Submit all upfront; rely on as_completed for streaming.
        futures = [ex.submit(_worker_process_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures)):
            try:
                row = fut.result()
            except Exception as e:
                # Should rarely happen given the worker traps everything.
                row = {
                    "frame_path": "<unknown>",
                    "face_area_fraction": float("nan"),
                    "strategy": "<unknown>",
                    "label": -1,
                    "error": f"future:{type(e).__name__}:{str(e)[:120]}",
                }
            if row.get("error"):
                error_rows.append({k: row[k] for k in ("frame_path", "strategy", "label", "error")})
            rows.append({k: row[k] for k in ("frame_path", "face_area_fraction", "strategy", "label")})

            done = i + 1
            if done - last_log_n >= args.log_every:
                elapsed = time.time() - t_compute_start
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (n_total - done) / rate if rate > 0 else float("inf")
                n_err = len(error_rows)
                n_face = sum(1 for r in rows if not (
                    isinstance(r["face_area_fraction"], float) and np.isnan(r["face_area_fraction"])
                ))
                log(f"  progress {done}/{n_total} ({100*done/n_total:.1f}%) "
                    f"rate={rate:.1f} fps elapsed={elapsed:.0f}s eta={eta:.0f}s "
                    f"err={n_err} face_detected={n_face}")
                last_log_n = done

            if done - last_chk_n >= args.checkpoint_every:
                # Write a partial parquet so we have a recoverable checkpoint.
                df_partial = pd.DataFrame(rows)
                df_partial.to_parquet(out_parquet, index=False)
                log(f"  checkpoint: wrote {len(df_partial)} rows -> {out_parquet}")
                last_chk_n = done

    compute_elapsed = time.time() - t_compute_start
    log(f"compute complete: {len(rows)} rows in {compute_elapsed:.1f}s "
        f"({len(rows)/compute_elapsed:.1f} fps)")

    # ---- write final parquet ----
    df = pd.DataFrame(rows)
    df.to_parquet(out_parquet, index=False)
    log(f"wrote {len(df)} rows -> {out_parquet}")

    # Sanity: schema
    log(f"schema: {dict(df.dtypes)}")

    # ---- summary ----
    n_total_proc = len(df)
    n_with_face = int(df["face_area_fraction"].notna().sum())
    coverage = n_with_face / n_total_proc if n_total_proc > 0 else 0.0

    by_strat_label: dict = {}
    for (s, lbl), grp in df.groupby(["strategy", "label"], dropna=False):
        n = len(grp)
        n_f = int(grp["face_area_fraction"].notna().sum())
        by_strat_label[f"{s}|label={int(lbl)}"] = {
            "n_frames": int(n),
            "n_with_face": n_f,
            "coverage": (n_f / n) if n else 0.0,
        }

    by_strat: dict = {}
    for s, grp in df.groupby("strategy"):
        valid = grp["face_area_fraction"].dropna()
        if len(valid) == 0:
            by_strat[s] = {
                "n_frames": int(len(grp)),
                "n_with_face": 0,
                "coverage": 0.0,
                "median": None, "p10": None, "p90": None, "p25": None, "p75": None,
            }
            continue
        by_strat[s] = {
            "n_frames": int(len(grp)),
            "n_with_face": int(len(valid)),
            "coverage": float(len(valid) / len(grp)),
            "median": float(valid.median()),
            "p10": float(valid.quantile(0.10)),
            "p25": float(valid.quantile(0.25)),
            "p75": float(valid.quantile(0.75)),
            "p90": float(valid.quantile(0.90)),
        }

    # Error breakdown
    err_kind_counts: dict[str, int] = {}
    for er in error_rows:
        kind = (er.get("error") or "").split(":", 1)[0]
        err_kind_counts[kind] = err_kind_counts.get(kind, 0) + 1
    sample_errs = error_rows[:20]

    summary = {
        "wall_clock_seconds": round(time.time() - overall_t0, 1),
        "compute_seconds": round(compute_elapsed, 1),
        "n_total_processed": n_total_proc,
        "n_with_face": n_with_face,
        "coverage_overall": round(coverage, 6),
        "n_errors": len(error_rows),
        "error_kind_counts": err_kind_counts,
        "sample_errors": sample_errs,
        "by_strategy_label": by_strat_label,
        "by_strategy": by_strat,
        "workers": args.workers,
        "bucket": BUCKET,
        "strategies": list(STRATEGIES),
        "output_parquet": str(out_parquet),
        "schema": {k: str(v) for k, v in df.dtypes.items()},
    }

    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"wrote summary -> {out_summary}")

    log_fh.close()


if __name__ == "__main__":
    main()
