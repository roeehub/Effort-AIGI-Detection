"""Slice analysis on the tagged-lockbox parquet.

Run as a script for a quick textual report; or import the helper functions from
a notebook for charts. The point of this module is to answer:
  *how much of the model's error budget is concentrated in slices defined by
   our per-image tags*. If a flag separates errors strongly, that's a data
   issue. If errors are uniform across flags, the model is genuinely weak.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TAGS = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis/lockbox_tagging/lockbox_tags_2026-04-27.parquet"
)

FLAG_COLUMNS = [
    "is_low_quality",
    "is_pose_extreme",
    "is_no_face",
    "is_likely_screen_capture",
    "is_arcface_nn_outlier",  # added by compute_arcface_outlier_score below
]
ARCFACE_NN_PURITY_THRESHOLD = 0.4  # if <40% of 5-NN share label, flag as outlier


def add_outcome_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Add pred_fake / correct / fn / fp columns based on prob_fake."""
    df = df.copy()
    df["pred_fake"] = df["prob_fake"] >= threshold
    df["is_fake"] = df["label"] == "fake"
    df["correct"] = (df["pred_fake"] & df["is_fake"]) | (~df["pred_fake"] & ~df["is_fake"])
    df["fn"] = df["is_fake"] & ~df["pred_fake"]   # missed fake
    df["fp"] = ~df["is_fake"] & df["pred_fake"]   # false alarm on real
    return df


def overall_metrics(df: pd.DataFrame) -> dict:
    n_real = int((~df["is_fake"]).sum())
    n_fake = int(df["is_fake"].sum())
    fpr = float(df["fp"].sum() / max(n_real, 1))
    recall = float((df["is_fake"] & df["pred_fake"]).sum() / max(n_fake, 1))
    return {
        "n_total": int(len(df)),
        "n_real": n_real,
        "n_fake": n_fake,
        "fpr": fpr,
        "fake_recall": recall,
    }


def slice_table(df: pd.DataFrame, flags: list[str] = FLAG_COLUMNS) -> pd.DataFrame:
    """For each flag, compare in-flag vs out-of-flag recall/FPR/n."""
    rows = []
    for flag in flags:
        if flag not in df.columns:
            continue
        for slice_label, sub in [("flag=True", df[df[flag]]), ("flag=False", df[~df[flag]])]:
            n_real = int((~sub["is_fake"]).sum())
            n_fake = int(sub["is_fake"].sum())
            fpr = float(sub["fp"].sum() / max(n_real, 1)) if n_real else float("nan")
            recall = (
                float((sub["is_fake"] & sub["pred_fake"]).sum() / max(n_fake, 1))
                if n_fake
                else float("nan")
            )
            rows.append({
                "flag": flag,
                "slice": slice_label,
                "n": len(sub),
                "n_real": n_real,
                "n_fake": n_fake,
                "fpr": fpr,
                "fake_recall": recall,
            })
    return pd.DataFrame(rows)


def error_budget_share(df: pd.DataFrame, flag: str) -> dict:
    """What fraction of FNs and FPs fall inside the flag=True slice?"""
    if flag not in df.columns:
        return {}
    n_fn = int(df["fn"].sum())
    n_fp = int(df["fp"].sum())
    fn_in = int(df.loc[df[flag], "fn"].sum())
    fp_in = int(df.loc[df[flag], "fp"].sum())
    flag_share = float(df[flag].mean())
    return {
        "flag": flag,
        "flag_share": flag_share,
        "fn_share_in_flag": fn_in / max(n_fn, 1),
        "fp_share_in_flag": fp_in / max(n_fp, 1),
        "fn_lift": (fn_in / max(n_fn, 1)) / max(flag_share, 1e-6),
        "fp_lift": (fp_in / max(n_fp, 1)) / max(flag_share, 1e-6),
    }


def quantile_buckets(
    df: pd.DataFrame,
    column: str,
    n_bins: int = 4,
) -> pd.DataFrame:
    """Per-quantile-bucket recall/FPR for a continuous column. NaNs go in their own bucket."""
    if column not in df.columns:
        return pd.DataFrame()
    s = df[column]
    out_rows = []
    if s.notna().any():
        bins = pd.qcut(s, q=n_bins, duplicates="drop")
        for bucket, sub in df.groupby(bins, observed=True):
            n_real = int((~sub["is_fake"]).sum())
            n_fake = int(sub["is_fake"].sum())
            out_rows.append({
                "column": column,
                "bucket": str(bucket),
                "n": len(sub),
                "fpr": sub["fp"].sum() / max(n_real, 1) if n_real else float("nan"),
                "fake_recall": (sub["is_fake"] & sub["pred_fake"]).sum() / max(n_fake, 1)
                if n_fake else float("nan"),
            })
    nan_sub = df[s.isna()]
    if len(nan_sub):
        n_real = int((~nan_sub["is_fake"]).sum())
        n_fake = int(nan_sub["is_fake"].sum())
        out_rows.append({
            "column": column,
            "bucket": "NaN",
            "n": len(nan_sub),
            "fpr": nan_sub["fp"].sum() / max(n_real, 1) if n_real else float("nan"),
            "fake_recall": (nan_sub["is_fake"] & nan_sub["pred_fake"]).sum() / max(n_fake, 1)
            if n_fake else float("nan"),
        })
    return pd.DataFrame(out_rows)


def compute_arcface_outlier_score(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """For each frame, find the k nearest ArcFace neighbors over the whole dataset
    and return:
      - arcface_nn_label_purity: fraction of k-NN that share the frame's label.
        Low purity = visual neighbors are mostly the opposite label, i.e. the
        frame is a property-level outlier.
      - arcface_nn_dist_p50: median cosine distance to the k-NN, i.e. how far the
        frame sits from its visual cohort overall.

    These are *property* signals built from the embedding — they make no use of
    manifest identity_key (which is unreliable on this dataset)."""
    if "arcface_embed" not in df.columns:
        return df
    has = df["arcface_embed"].apply(lambda v: v is not None and len(v) == 512)
    sub = df[has]
    if len(sub) < k + 2:
        return df

    embeds = np.stack([np.asarray(v, dtype=np.float32) for v in sub["arcface_embed"]])
    # Already unit-normed, so dot product = cosine similarity.
    sims = embeds @ embeds.T
    np.fill_diagonal(sims, -np.inf)
    # k+1 to skip the self that's still at sims[i][i] in case of dup, but we already masked.
    knn_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    is_fake = (sub["label"].values == "fake")

    purities = np.zeros(len(sub))
    dists = np.zeros(len(sub))
    for i in range(len(sub)):
        ni = knn_idx[i]
        purities[i] = float(np.mean(is_fake[ni] == is_fake[i]))
        dists[i] = float(np.median(1.0 - sims[i, ni]))

    df = df.copy()
    df.loc[has, "arcface_nn_label_purity"] = purities
    df.loc[has, "arcface_nn_dist_p50"] = dists
    df["is_arcface_nn_outlier"] = (
        df.get("arcface_nn_label_purity", pd.Series(dtype=float)) < ARCFACE_NN_PURITY_THRESHOLD
    ).fillna(False)
    return df


def threshold_sweep(df: pd.DataFrame) -> pd.DataFrame:
    n_real = int((df["label"] == "real").sum())
    n_fake = int((df["label"] == "fake").sum())
    rows = []
    for t in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]:
        pred = df["prob_fake"] >= t
        fpr = (pred & (df["label"] == "real")).sum() / max(n_real, 1)
        recall = (pred & (df["label"] == "fake")).sum() / max(n_fake, 1)
        rows.append({"threshold": t, "fpr": float(fpr), "fake_recall": float(recall)})
    return pd.DataFrame(rows)


def report(parquet: Path = DEFAULT_TAGS, threshold: float = 0.5) -> None:
    df = pd.read_parquet(parquet)
    df = compute_arcface_outlier_score(df)
    df = add_outcome_columns(df, threshold=threshold)
    print("=" * 70)
    print(f"Tagged report   (parquet: {parquet.name}, n={len(df)}, threshold={threshold})")
    print("=" * 70)
    print()
    print("Overall:", overall_metrics(df))
    print()

    print("--- Threshold sweep ---")
    print(threshold_sweep(df).to_string(index=False))
    print()

    print("--- Per-flag recall/FPR comparison ---")
    print(slice_table(df).to_string(index=False))
    print()

    print("--- Error budget share by flag (lift > 1.0 means flag concentrates errors) ---")
    eb = pd.DataFrame([error_budget_share(df, f) for f in FLAG_COLUMNS])
    print(eb.to_string(index=False))
    print()

    cols = [
        "face_pixel_area",
        "sharpness_laplacian",
        "yaw_deg",
        "pitch_deg",
        "brightness_v_mean",
        "arcface_nn_label_purity",
        "arcface_nn_dist_p50",
    ]
    for col in cols:
        if col not in df.columns:
            continue
        print(f"--- Quantile buckets for {col} ---")
        print(quantile_buckets(df, col).to_string(index=False))
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=DEFAULT_TAGS)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    report(args.parquet, args.threshold)
