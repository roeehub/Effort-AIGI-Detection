"""Build a synthetic tags parquet where `clip_capture_mode` is replaced by
IQ-feature K-means cluster labels.

Pass the output parquet to `per_substrate_tau_calibration_2026-05-05/run_calibration.py`
via `--tags-parquet <out>` and `--substrates iqA0 iqA1 ...`.

Phase 1 of the IQ-substrate τ feasibility test.
See `/Users/roeedar/.claude/plans/cheeky-roaming-riddle.md`.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
DEFAULT_TAGS = REPO / "analysis/lockbox_tagging/full_tags_2026-04-27.parquet"

# Phase 1 feature set — all already in the parquet, all deployment-observable
# (computable per-frame from raw image).
# NOTE: jpeg_qf_estimate is all-NaN in full_tags_2026-04-27.parquet (estimation
# failed); dropping. brightness_v_std is the contrast-adjacent feature also
# available and used here as luma-spread.
FEATURES_PARQUET = [
    "sharpness_laplacian",   # 2026-05-19 lap_var
    "brightness_v_mean",      # luma
    "brightness_v_std",       # luma std (related to contrast)
    "contrast_rms",           # contrast
    "saturation_s_mean",      # saturation
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags-in", default=str(DEFAULT_TAGS),
                    help="Source tags parquet (lockbox-tagging output)")
    ap.add_argument("--k", type=int, default=4, help="Number of K-means clusters")
    ap.add_argument("--seed", type=int, default=9921)
    ap.add_argument("--out", required=True,
                    help="Output directory (will contain synthetic_tags.parquet + kmeans.pkl + summary.json)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading {args.tags_in} ...")
    cols = ["gcs_uri", "clip_capture_mode", "split"] + FEATURES_PARQUET
    tags = pd.read_parquet(args.tags_in, columns=cols)
    print(f"  loaded {len(tags)} rows")

    print(f"[2/5] Inspecting feature availability ...")
    for c in FEATURES_PARQUET:
        n_nan = tags[c].isna().sum()
        print(f"  {c}: {n_nan}/{len(tags)} NaN, range [{tags[c].min():.2f}, {tags[c].max():.2f}], mean {tags[c].mean():.2f}")

    # Drop rows with any NaN feature (rare; would mess up K-means)
    valid_mask = tags[FEATURES_PARQUET].notna().all(axis=1)
    valid = tags[valid_mask].copy()
    print(f"  {valid_mask.sum()}/{len(tags)} rows have all features non-NaN")

    # Use dev split for clustering (per plan); apply to all
    is_dev = valid["split"] == "dev"
    print(f"  dev split: {is_dev.sum()} rows; lockbox: {(valid['split']=='lockbox').sum()}; other: {(~is_dev & (valid['split']!='lockbox')).sum()}")

    print(f"[3/5] Fitting StandardScaler + KMeans(k={args.k}) on dev rows ...")
    X_dev = valid.loc[is_dev, FEATURES_PARQUET].values
    scaler = StandardScaler()
    X_dev_scaled = scaler.fit_transform(X_dev)
    kmeans = KMeans(n_clusters=args.k, random_state=args.seed, n_init=10)
    kmeans.fit(X_dev_scaled)

    # Apply to all valid rows
    X_all_scaled = scaler.transform(valid[FEATURES_PARQUET].values)
    cluster_ids = kmeans.predict(X_all_scaled)
    valid["iq_cluster"] = [f"iqA{c}" for c in cluster_ids]

    print(f"[4/5] Cluster summary on dev rows:")
    dev_clusters = valid.loc[is_dev, "iq_cluster"].value_counts().sort_index()
    print(f"  dev cluster counts: {dict(dev_clusters)}")
    lockbox_clusters = valid.loc[valid['split']=='lockbox', "iq_cluster"].value_counts().sort_index()
    print(f"  lockbox cluster counts: {dict(lockbox_clusters)}")

    # Cluster centers in original units (un-scale)
    centers_scaled = kmeans.cluster_centers_
    centers_orig = scaler.inverse_transform(centers_scaled)
    centers_df = pd.DataFrame(centers_orig, columns=FEATURES_PARQUET,
                              index=[f"iqA{i}" for i in range(args.k)])
    print(f"  Cluster centers (original scale):")
    print(centers_df.round(2))

    # Cross-tabulation: how does iq_cluster correlate with clip_capture_mode?
    valid_with_mode = valid.dropna(subset=["clip_capture_mode"])
    if len(valid_with_mode) > 0:
        crosstab = pd.crosstab(valid_with_mode["iq_cluster"],
                               valid_with_mode["clip_capture_mode"],
                               normalize="index")
        print(f"\n  IQ-cluster x clip_capture_mode cross-tab (row-normalized):")
        print(crosstab.round(3))

    # Build synthetic parquet — REPLACE clip_capture_mode with iq_cluster.
    # The calibration framework reads cols ['gcs_uri', 'clip_capture_mode', 'split'].
    synthetic = valid[["gcs_uri", "split", "iq_cluster"]].copy()
    synthetic = synthetic.rename(columns={"iq_cluster": "clip_capture_mode"})
    synth_path = out_dir / "synthetic_tags.parquet"
    synthetic.to_parquet(synth_path, index=False)
    print(f"\n[5/5] Wrote {synth_path}: {len(synthetic)} rows, "
          f"{synthetic['clip_capture_mode'].nunique()} unique substrate labels")

    # Persist k-means + scaler for Phase 4 (natural-experiment retest)
    with open(out_dir / "kmeans_model.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "kmeans": kmeans,
                     "features": FEATURES_PARQUET, "k": args.k,
                     "labels": [f"iqA{i}" for i in range(args.k)]}, f)

    # Summary JSON
    summary = {
        "k": args.k,
        "seed": args.seed,
        "n_dev_rows": int(is_dev.sum()),
        "n_lockbox_rows": int((valid['split']=='lockbox').sum()),
        "features_used": FEATURES_PARQUET,
        "cluster_centers_original_scale": centers_df.to_dict(orient="index"),
        "dev_cluster_counts": {str(k): int(v) for k, v in dev_clusters.items()},
        "lockbox_cluster_counts": {str(k): int(v) for k, v in lockbox_clusters.items()},
        "tags_parquet_source": args.tags_in,
        "synthetic_parquet_path": str(synth_path),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {out_dir / 'summary.json'}")
    print(f"  wrote {out_dir / 'kmeans_model.pkl'}")

    # Print the substrate-name list for convenience
    labels = [f"iqA{i}" for i in range(args.k)]
    print(f"\nFor calibration, use:")
    print(f"  --tags-parquet {synth_path}")
    print(f"  --substrates {' '.join(labels)}")


if __name__ == "__main__":
    main()
