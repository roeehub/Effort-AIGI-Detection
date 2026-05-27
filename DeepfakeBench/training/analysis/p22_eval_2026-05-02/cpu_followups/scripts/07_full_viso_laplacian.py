"""Job I — full-viso F1 Pearson r recomputation.

The original F1 falsifier was computed on n=63 viso frames in
cross_suite_attributes.csv. The full viso suite is n=550. Compute Laplacian
variance on the remaining frames so we can re-evaluate F1 at full sample size.

Strategy: download the viso dev frames from GCS in parallel, compute
laplacian_var with cv2, then merge with per-frame scorecard and recompute
Pearson.
"""
from __future__ import annotations

import sys; sys.path.insert(0, "analysis/p22_eval_2026-05-02/cpu_followups/scripts")

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import io

import cv2
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from google.cloud import storage

from _common import OUT, FIG, ATTRS_CSV, load_per_frame_scores

CKPTS = ["P8A_REFERENCE_STEP5000", "P18T_GRL_TREATMENT_STEP4000",
         "P22_AUG_STEP1000", "P22_AUG_STEP4000", "P22_AUG_STEP8000"]


def laplacian_var_from_bytes(b: bytes) -> float | None:
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def gs_to_bucket_blob(uri: str):
    assert uri.startswith("gs://")
    rest = uri[5:]
    bucket, _, blob = rest.partition("/")
    return bucket, blob


def fetch_and_compute(client_storage, frame_path: str):
    try:
        bucket_name, blob_name = gs_to_bucket_blob(frame_path)
        bucket = client_storage.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        b = blob.download_as_bytes()
        return frame_path, laplacian_var_from_bytes(b)
    except Exception as e:
        return frame_path, None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    # Use P8A's per-frame viso CSV as the source of viso frame paths
    viso_p8a = load_per_frame_scores("P8A_REFERENCE_STEP5000",
                                      "visomaster_enhanced_macro_dev")
    if viso_p8a is None:
        print("ERROR: no viso P8A CSV found")
        return
    print(f"Total viso frames: {len(viso_p8a)}")

    # Skip frames that already have laplacian in attrs
    attrs = pd.read_csv(ATTRS_CSV)
    have_lap = attrs[["frame_path", "laplacian_var"]].dropna()
    paths_done = set(have_lap.frame_path.tolist())
    paths_to_fetch = [p for p in viso_p8a.frame_path.unique() if p not in paths_done]
    print(f"Already have lap on {len(paths_done & set(viso_p8a.frame_path))} viso frames; "
          f"need to fetch {len(paths_to_fetch)}")

    if paths_to_fetch:
        client = storage.Client(project="train-cvit2")
        results = []
        with ThreadPoolExecutor(max_workers=16) as ex:
            futures = [ex.submit(fetch_and_compute, client, p) for p in paths_to_fetch]
            for i, fut in enumerate(futures):
                r = fut.result()
                results.append(r)
                if (i + 1) % 50 == 0:
                    print(f"  fetched {i+1}/{len(paths_to_fetch)}")
        new_lap = pd.DataFrame(results, columns=["frame_path", "laplacian_var"])
        new_lap = new_lap.dropna(subset=["laplacian_var"])
        print(f"Successfully fetched lap for {len(new_lap)} viso frames")
        # Save fetched results so we can re-use
        out_path = OUT / "07_viso_laplacian_fetched.csv"
        new_lap.to_csv(out_path, index=False)
        print(f"Wrote: {out_path}")
        # Combine with existing
        all_lap = pd.concat([have_lap, new_lap], ignore_index=True)
    else:
        all_lap = have_lap

    print(f"Total viso frames with lap: "
          f"{(all_lap.frame_path.isin(viso_p8a.frame_path)).sum()}")

    # Recompute Pearson r per ckpt on full viso
    rows = []
    for ckpt in CKPTS:
        viso_scores = load_per_frame_scores(ckpt, "visomaster_enhanced_macro_dev")
        if viso_scores is None: continue
        merged = viso_scores.merge(all_lap, on="frame_path", how="inner")
        merged = merged.dropna(subset=["frame_prob", "laplacian_var"])
        if len(merged) < 30:
            print(f"  {ckpt}: too few merged ({len(merged)}), skipping")
            continue
        r, p = pearsonr(merged["frame_prob"], merged["laplacian_var"])
        rows.append({"ckpt": ckpt, "n": len(merged),
                     "pearson_r_score_lap": float(r),
                     "pearson_p": float(p)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "07_full_viso_pearson.csv", index=False)

    print("\n" + "=" * 80)
    print("Full-viso F1 — Pearson r(score, laplacian_var) per checkpoint")
    print("=" * 80)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))

    if "P8A_REFERENCE_STEP5000" in df.ckpt.values:
        p8a = df[df.ckpt == "P8A_REFERENCE_STEP5000"].pearson_r_score_lap.iloc[0]
        print(f"\nP8A baseline r = {p8a:+.4f}, |r| = {abs(p8a):.4f}")
        for _, row in df.iterrows():
            if row["ckpt"] == "P8A_REFERENCE_STEP5000": continue
            delta_mag = abs(p8a) - abs(row.pearson_r_score_lap)
            verdict = "PASS" if delta_mag >= 0.15 else "FAIL"
            print(f"  {row['ckpt']:32s}  r={row.pearson_r_score_lap:+.4f}  "
                  f"|Δr|={delta_mag:+.4f}  → F1 {verdict}")


if __name__ == "__main__":
    main()
