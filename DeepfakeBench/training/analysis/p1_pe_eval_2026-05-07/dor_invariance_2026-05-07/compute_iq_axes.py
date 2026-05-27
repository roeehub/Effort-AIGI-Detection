"""
Compute per-frame IQ-axis values (min_dim, color_b_dev, sharpness)
for the 180-frame dor invariance probe.

Tests whether P1's wins/regressions correlate with IQ axes that
the prevailing memory says drove 2026-05-06 may6 drift.

Reads frames once from local _axis_cache/ (downloads if missing).
Writes axis_values_per_frame.csv and axis_correlation_summary.csv.
"""
from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from google.cloud import storage

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO, datefmt="%H:%M:%S")
log = logging.getLogger("iq_axes")

THIS_DIR = Path(__file__).resolve().parent
MANIFEST = THIS_DIR / "manifest.csv"
SCORES = THIS_DIR / "scores_full.csv"
CACHE_DIR = THIS_DIR / "_axis_cache"
CACHE_DIR.mkdir(exist_ok=True)

CKPTS = ["P8A", "E2B", "P1_BUNDLE_step4000", "P1_PAIRRANK_step6750"]
SCORE_COLS = [f"score_{c}" for c in CKPTS]


def parse_gcs(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(uri)
    bucket, blob = uri[5:].split("/", 1)
    return bucket, blob


def cache_path_for(blob_path: str) -> Path:
    # Use blob path's filename + a stable hash-free key. Use full blob path
    # converted to filesystem-safe.
    safe = blob_path.replace("/", "__").replace(" ", "_")
    return CACHE_DIR / safe


def download_one(client: storage.Client, bucket_name: str, blob_path: str) -> Path:
    out = cache_path_for(blob_path)
    if out.exists() and out.stat().st_size > 0:
        return out
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    data = blob.download_as_bytes()
    out.write_bytes(data)
    return out


def download_all(df: pd.DataFrame) -> dict[int, Path]:
    """Parallel download. Returns {row_idx: local_path}."""
    client = storage.Client()
    by_idx: dict[int, Path] = {}

    def _job(idx: int, uri: str) -> tuple[int, Path]:
        bucket, blob = parse_gcs(uri)
        return idx, download_one(client, bucket, blob)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(_job, i, row["frame_path"]) for i, row in df.iterrows()]
        for n, fut in enumerate(as_completed(futs), 1):
            i, p = fut.result()
            by_idx[i] = p
            if n % 20 == 0 or n == len(futs):
                log.info("downloaded %d / %d (%.1fs)", n, len(futs), time.time() - t0)
    return by_idx


def compute_axes(local_path: Path) -> dict:
    """Decode image and compute IQ axes.

    color_b_dev: std of B-channel pixel values, treating decoded array
                 as RGB. (PIL decodes to RGB; cv2.imdecode uses BGR.)
                 We use cv2 imdecode then convert to RGB so 'B' channel
                 means BLUE in the natural sense, matching task spec.
    sharpness:   variance of cv2.Laplacian(gray, CV_64F).
    min_dim:     min(width, height) of decoded image.
    """
    data = np.frombuffer(local_path.read_bytes(), dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"decode failed: {local_path}")
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    b_channel = img_rgb[:, :, 2]  # B = blue (in RGB index 2)
    color_b_dev = float(np.std(b_channel))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "width": int(w),
        "height": int(h),
        "min_dim": int(min(w, h)),
        "color_b_dev": color_b_dev,
        "sharpness": sharpness,
    }


def pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r with NaN safety. Returns NaN if degenerate."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    scores = pd.read_csv(SCORES)
    log.info("scores: %d rows, variants=%s", len(scores),
             sorted(scores["variant"].unique()))

    log.info("downloading frames into %s", CACHE_DIR)
    local_paths = download_all(scores)

    log.info("computing axes")
    axes_rows = []
    for i, row in scores.iterrows():
        lp = local_paths[i]
        ax = compute_axes(lp)
        axes_rows.append({
            "variant": row["variant"],
            "frame_path": row["frame_path"],
            **ax,
        })
    axes_df = pd.DataFrame(axes_rows)

    full = scores.merge(axes_df, on=["variant", "frame_path"])
    assert len(full) == 180, f"expected 180 rows, got {len(full)}"

    # Score deltas vs P8A
    for ck in CKPTS:
        if ck == "P8A":
            continue
        full[f"delta_{ck}"] = full[f"score_{ck}"] - full["score_P8A"]

    out_per_frame = THIS_DIR / "axis_values_per_frame.csv"
    full.to_csv(out_per_frame, index=False)
    log.info("wrote %s (n=%d)", out_per_frame, len(full))

    # Correlations: per (axis, ckpt, target, scope)
    AXES = ["min_dim", "color_b_dev", "sharpness"]
    TARGETS = []
    for ck in CKPTS:
        TARGETS.append((f"score_{ck}", ck, "raw"))
    for ck in CKPTS:
        if ck == "P8A":
            continue
        TARGETS.append((f"delta_{ck}", ck, "delta_vs_P8A"))

    rows = []
    variants = sorted(full["variant"].unique())
    for axis in AXES:
        for col, ck, target_kind in TARGETS:
            # Combined
            r = pearson_safe(full[axis].values, full[col].values)
            rows.append({
                "axis": axis, "ckpt": ck, "target_kind": target_kind,
                "scope": "ALL", "n": len(full), "pearson_r": r,
            })
            # Per variant
            for v in variants:
                sub = full[full["variant"] == v]
                r_v = pearson_safe(sub[axis].values, sub[col].values)
                rows.append({
                    "axis": axis, "ckpt": ck, "target_kind": target_kind,
                    "scope": v, "n": len(sub), "pearson_r": r_v,
                })

    corr_df = pd.DataFrame(rows)
    out_corr = THIS_DIR / "axis_correlation_summary.csv"
    corr_df.to_csv(out_corr, index=False)
    log.info("wrote %s (n=%d)", out_corr, len(corr_df))

    # Console summary
    print()
    print("=" * 100)
    print("AXIS DESCRIPTIVES per variant (mean ± std)")
    print("=" * 100)
    desc = full.groupby("variant").agg(
        n=("min_dim", "size"),
        min_dim_mean=("min_dim", "mean"),
        min_dim_std=("min_dim", "std"),
        color_b_dev_mean=("color_b_dev", "mean"),
        color_b_dev_std=("color_b_dev", "std"),
        sharpness_mean=("sharpness", "mean"),
        sharpness_std=("sharpness", "std"),
    )
    print(desc.round(2).to_string())

    # Compact correlation table — combined, raw scores
    print()
    print("=" * 100)
    print("PEARSON r — RAW SCORE vs AXIS  (combined ALL n=180)")
    print("=" * 100)
    pivot_raw = corr_df[(corr_df["scope"] == "ALL") &
                        (corr_df["target_kind"] == "raw")].pivot(
        index="ckpt", columns="axis", values="pearson_r"
    )
    print(pivot_raw.round(3).to_string())

    print()
    print("=" * 100)
    print("PEARSON r — DELTA vs P8A vs AXIS  (combined ALL n=180)")
    print("=" * 100)
    pivot_delta = corr_df[(corr_df["scope"] == "ALL") &
                          (corr_df["target_kind"] == "delta_vs_P8A")].pivot(
        index="ckpt", columns="axis", values="pearson_r"
    )
    print(pivot_delta.round(3).to_string())

    # Per-variant correlation tables for the two flagged variants
    for v in ["dor_webcam_with_vbg", "dor_session_0424"]:
        print()
        print("=" * 100)
        print(f"PEARSON r — DELTA vs P8A vs AXIS  (scope={v} n=30)")
        print("=" * 100)
        sub = corr_df[(corr_df["scope"] == v) &
                      (corr_df["target_kind"] == "delta_vs_P8A")].pivot(
            index="ckpt", columns="axis", values="pearson_r"
        )
        print(sub.round(3).to_string())

        print()
        print(f"--- raw score r — scope={v} ---")
        sub2 = corr_df[(corr_df["scope"] == v) &
                       (corr_df["target_kind"] == "raw")].pivot(
            index="ckpt", columns="axis", values="pearson_r"
        )
        print(sub2.round(3).to_string())

    # Per-variant mean scores re-printed for context
    print()
    print("=" * 100)
    print("MEAN SCORE per variant (cross-check)")
    print("=" * 100)
    means = full.groupby("variant")[SCORE_COLS].mean().round(3)
    print(means.to_string())


if __name__ == "__main__":
    main()
