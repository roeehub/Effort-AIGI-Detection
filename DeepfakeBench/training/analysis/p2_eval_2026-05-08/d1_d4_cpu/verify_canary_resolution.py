#!/usr/bin/env python3
"""Resolution + sharpness audit on the 800-frame canary substrate.

Question (raised by user 2026-05-08 after eyeballing chronic-6 contact sheets):
the displayed crops look pixelated. Is that a viewer-side downscale, or is the
image that goes into the model actually that low-resolution? If the latter,
the canary's chronic-real failures may live below where production frames will
ever sit, and a deployment-side IQ gate is the right policy.

Method:
  1. Stat every cached crop in `_frame_cache/` (pixel dims, JPEG bytes).
  2. Compute Laplacian variance (sharpness proxy) per cohort.
  3. Spot-check 3 frames against the GCS source to verify the cached file is
     the saved-on-disk version (no further downscale at cache time).
  4. Tabulate per-cohort distribution + the chronic-6-vs-others split, since
     the user's eyeball was specifically on the chronic-6 cohort.

Output:
  - outputs/canary_resolution_per_frame.csv  (one row per cached file)
  - outputs/canary_resolution_per_cohort.csv (per-cohort summary)
  - prints a side-by-side comparison + the headline numbers.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO,
                    datefmt="%H:%M:%S")
log = logging.getLogger("verify_res")

THIS = Path(__file__).resolve().parent
SCORES = THIS / "scores"
CACHE = THIS / "_frame_cache"
OUT = THIS / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def laplacian_var(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def main():
    meta = pd.read_csv(SCORES / "_canary_meta.csv")
    log.info("canary meta: %d rows × %d cols", *meta.shape)

    rows = []
    missing = 0
    for _, r in meta.iterrows():
        bn = r["frame_path"].split("/")[-1]
        f = CACHE / bn
        if not f.exists():
            missing += 1
            continue
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            missing += 1
            continue
        h, w = img.shape[:2]
        rows.append({
            "frame_idx": int(r["frame_idx"]),
            "cohort": r["cohort"],
            "base_identity": r["base_identity"],
            "label": int(r["label"]),
            "h": h, "w": w,
            "min_dim": min(h, w),
            "max_dim": max(h, w),
            "lap_var": laplacian_var(img),
            "bytes": f.stat().st_size,
        })
    log.info("read %d / %d cached frames (missing=%d)", len(rows), len(meta), missing)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "canary_resolution_per_frame.csv", index=False)

    # Headline distribution.
    print()
    print("=== Headline pixel-resolution + sharpness across all 800 canary frames ===")
    for col in ["min_dim", "max_dim", "lap_var"]:
        s = df[col]
        print(f"  {col:10s}  p05={s.quantile(0.05):.1f}  p25={s.quantile(0.25):.1f}  "
              f"p50={s.quantile(0.50):.1f}  p75={s.quantile(0.75):.1f}  "
              f"p95={s.quantile(0.95):.1f}  mean={s.mean():.1f}")

    # Chronic-6 vs others.
    chronic_ids = {
        "PC_Generator__s22", "PC_Generator__s45", "Q__s6", "Roy_D",
        "bla_bla_chow", "bla_bla_chow__s2",
    }
    df["chronic"] = df["base_identity"].isin(chronic_ids)
    print()
    print("=== Chronic-6 vs other-real vs fake split ===")
    grp = df.assign(group=np.where(
        df["chronic"], "chronic_real",
        np.where(df["label"] == 0, "other_real", "fake")
    )).groupby("group").agg(
        n=("frame_idx", "count"),
        min_dim_p50=("min_dim", lambda x: x.quantile(0.50)),
        min_dim_p05=("min_dim", lambda x: x.quantile(0.05)),
        lap_var_p50=("lap_var", lambda x: x.quantile(0.50)),
        lap_var_p05=("lap_var", lambda x: x.quantile(0.05)),
    ).reset_index()
    print(grp.round(1).to_string(index=False))

    # Per-cohort summary.
    print()
    print("=== Per-cohort summary (chronic-6 detail) ===")
    chr_grp = df[df["chronic"]].groupby("base_identity").agg(
        n=("frame_idx", "count"),
        min_dim_p50=("min_dim", lambda x: x.quantile(0.50)),
        min_dim_min=("min_dim", "min"),
        lap_var_p50=("lap_var", lambda x: x.quantile(0.50)),
        lap_var_p05=("lap_var", lambda x: x.quantile(0.05)),
    ).reset_index()
    print(chr_grp.round(1).to_string(index=False))
    chr_grp.to_csv(OUT / "canary_resolution_per_cohort.csv", index=False)

    # Spot-check: pull the GCS source for one chronic frame and one fake to
    # confirm the cached file matches the source dims (no downscale-at-cache).
    print()
    print("=== Spot-check: cached vs GCS-source dims ===")
    spot_idxs = []
    for ident in ["PC_Generator__s22", "Roy_D"]:
        m = df[(df["base_identity"] == ident)].sort_values("frame_idx").iloc[0]
        spot_idxs.append(int(m["frame_idx"]))
    fake_row = meta[meta["label"] == 1].iloc[0]
    spot_idxs.append(int(fake_row["frame_idx"]))

    for fidx in spot_idxs:
        meta_row = meta[meta["frame_idx"] == fidx].iloc[0]
        bn = meta_row["frame_path"].split("/")[-1]
        cached = CACHE / bn
        cached_img = cv2.imread(str(cached), cv2.IMREAD_COLOR)
        ch, cw = cached_img.shape[:2]
        # Pull GCS source to /tmp.
        tmp = Path(f"/tmp/_canary_verify_{fidx}.jpg")
        if not tmp.exists():
            r = subprocess.run(["gsutil", "cp", meta_row["frame_path"], str(tmp)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  frame {fidx}: gsutil failed: {r.stderr.strip()[:200]}")
                continue
        gcs_img = cv2.imread(str(tmp), cv2.IMREAD_COLOR)
        gh, gw = gcs_img.shape[:2]
        match = "MATCH" if (ch, cw) == (gh, gw) else "DIFFER"
        cached_lap = laplacian_var(cached_img)
        gcs_lap = laplacian_var(gcs_img)
        print(f"  frame {fidx} ({meta_row['base_identity']}): "
              f"cached {cw}×{ch} (lap={cached_lap:.0f})  "
              f"gcs {gw}×{gh} (lap={gcs_lap:.0f})  [{match}]")

    print()
    print(f"wrote {OUT / 'canary_resolution_per_frame.csv'}")
    print(f"wrote {OUT / 'canary_resolution_per_cohort.csv'}")


if __name__ == "__main__":
    main()
