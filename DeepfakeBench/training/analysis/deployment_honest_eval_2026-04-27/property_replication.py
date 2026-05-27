"""Replicate the parallel agent's property-level findings on the full
7,334-row tagged parquet (R9A predictions).

Goal: verify each headline before relying on it. Then extend with two pieces
the agent didn't compute but we need:
- per-method recall in each property bucket (so we know the recall side
  doesn't quietly collapse for any specific deepfake family)
- per-(label, split) decomposition so we can see e.g. lockbox-real-only FPR
  by property bucket, since the headline mixes dev+lockbox

Run from training/:
  python3 -m analysis.deployment_honest_eval_2026-04-27.property_replication
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
PARQUET = REPO / "analysis/lockbox_tagging/full_tags_2026-04-27.parquet"
OUT_DIR = REPO / "analysis/deployment_honest_eval_2026-04-27"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def add_outcome(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["pred_fake"] = df["prob_fake"] >= threshold
    df["is_fake"] = df["label"] == "fake"
    df["fp"] = (~df["is_fake"]) & df["pred_fake"]
    df["fn"] = df["is_fake"] & (~df["pred_fake"])
    df["correct"] = ((df["is_fake"]) & (df["pred_fake"])) | ((~df["is_fake"]) & (~df["pred_fake"]))
    return df


def metrics(df: pd.DataFrame) -> dict:
    n_real = int((~df["is_fake"]).sum())
    n_fake = int(df["is_fake"].sum())
    return {
        "n": int(len(df)),
        "n_real": n_real,
        "n_fake": n_fake,
        "fpr": float(df["fp"].sum() / max(n_real, 1)),
        "recall": float((df["is_fake"] & df["pred_fake"]).sum() / max(n_fake, 1)),
    }


def quartile_table(df: pd.DataFrame, col: str, n_bins: int = 4) -> pd.DataFrame:
    s = df[col]
    if not s.notna().any():
        return pd.DataFrame()
    # qcut with duplicates="drop" — produces ≤n_bins quantile bins
    bins = pd.qcut(s, q=n_bins, duplicates="drop")
    rows = []
    for bucket, sub in df.groupby(bins, observed=True):
        m = metrics(sub)
        m["column"] = col
        m["bucket"] = str(bucket)
        rows.append(m)
    nan_sub = df[s.isna()]
    if len(nan_sub):
        m = metrics(nan_sub)
        m["column"] = col
        m["bucket"] = "NaN"
        rows.append(m)
    return pd.DataFrame(rows)


def per_method_recall_in_bucket(df: pd.DataFrame, mask: pd.Series, threshold: float) -> dict:
    """For each fake-method family, recall on frames passing the mask."""
    out = {}
    sub = df[mask & (df["label"] == "fake")]
    for m, g in sub.groupby("method"):
        if len(g) < 5:
            continue
        rec = float((g["prob_fake"] >= threshold).sum() / len(g))
        out[m] = {"n": int(len(g)), "recall": rec}
    return out


def main() -> None:
    df = pd.read_parquet(PARQUET)
    print(f"[replicate] loaded {len(df)} rows  ({(df['split']=='dev').sum()} dev, {(df['split']=='lockbox').sum()} lockbox)")
    print(f"[replicate] labels: {df['label'].value_counts().to_dict()}")

    # ------- Part A: cross-split threshold sweep (validate "lockbox is harder") -------
    print("\n=== A. Threshold sweep, per split ===")
    rows = []
    for split in ["dev", "lockbox"]:
        sub = df[df["split"] == split]
        n_real = int((sub["label"] == "real").sum())
        n_fake = int((sub["label"] == "fake").sum())
        for t in [0.5, 0.7, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995]:
            pred = sub["prob_fake"] >= t
            fpr = float(((sub["label"] == "real") & pred).sum() / max(n_real, 1))
            rec = float(((sub["label"] == "fake") & pred).sum() / max(n_fake, 1))
            rows.append({"split": split, "n": len(sub), "n_real": n_real, "n_fake": n_fake,
                         "threshold": t, "fpr": fpr, "recall": rec})
    a_df = pd.DataFrame(rows)
    print(a_df.to_string(index=False))
    a_df.to_csv(OUT_DIR / "A_threshold_sweep_per_split.csv", index=False)

    # ------- Part B: property quartiles at threshold 0.5 (the agent's main finding) -------
    print("\n=== B. Property quartiles @ threshold=0.5, full set ===")
    df_t = add_outcome(df, threshold=0.5)
    for col in ["face_pixel_area", "sharpness_laplacian", "brightness_v_mean", "pitch_deg", "yaw_deg"]:
        print(f"\n--- {col} (threshold=0.5) ---")
        t = quartile_table(df_t, col)
        if len(t):
            print(t.to_string(index=False))
            t.to_csv(OUT_DIR / f"B_quartile_{col}_thr0.5.csv", index=False)

    # ------- Part C: face_pixel_area decomposed by split -------
    # Confirms whether the "small faces → high FPR" pattern is dev-specific or also in lockbox.
    print("\n=== C. face_pixel_area quartiles by split @ threshold=0.5 ===")
    for split in ["dev", "lockbox"]:
        sub = df_t[df_t["split"] == split]
        if len(sub) == 0:
            continue
        # Use *full-data* quantile cuts so the comparison is on a common scale.
        all_pa = df_t["face_pixel_area"]
        bins = pd.qcut(all_pa, q=4, duplicates="drop")
        sub_b = bins.reindex(sub.index)
        rows = []
        for bucket, sg in sub.groupby(sub_b, observed=True):
            m = metrics(sg)
            m["split"] = split
            m["bucket"] = str(bucket)
            rows.append(m)
        out_c = pd.DataFrame(rows)
        print(f"\n--- {split} ---")
        print(out_c.to_string(index=False))
        out_c.to_csv(OUT_DIR / f"C_facearea_{split}_thr0.5.csv", index=False)

    # ------- Part D: per-method recall by face_pixel_area quartile -------
    # The critical extension: if the "recall stays 93-99% across buckets" claim has
    # asymmetry across visomaster vs deeplive vs teams_fake_all, we need to know.
    print("\n=== D. Per-method recall by face_pixel_area quartile @ threshold=0.5 ===")
    pa_bins = pd.qcut(df["face_pixel_area"], q=4, duplicates="drop")
    rows = []
    for bucket, sg in df_t.groupby(pa_bins, observed=True):
        # Try to coerce 'method' to a family group when possible
        for m_name, mg in sg[sg["label"] == "fake"].groupby("method"):
            if len(mg) < 10:
                continue
            rec = float((mg["prob_fake"] >= 0.5).sum() / len(mg))
            rows.append({"bucket": str(bucket), "method": m_name, "n": len(mg), "recall": rec})
    d_df = pd.DataFrame(rows)
    print(d_df.sort_values(["method", "bucket"]).to_string(index=False))
    d_df.to_csv(OUT_DIR / "D_per_method_recall_by_facearea.csv", index=False)

    # ------- Part E: simple "deployment-honest cut" sweep -------
    # Apply face_pixel_area >= cut and report aggregate FPR/recall + per-method recall
    print("\n=== E. Deployment-honest cut sweep (face_pixel_area >= cutoff) ===")
    cuts = [10000, 20000, 30000, 40000, 50000, 57200, 70000, 90000]
    rows = []
    for cut in cuts:
        sub = df[df["face_pixel_area"] >= cut]
        for split in ["dev", "lockbox"]:
            ssub = sub[sub["split"] == split]
            n_real = int((ssub["label"] == "real").sum())
            n_fake = int((ssub["label"] == "fake").sum())
            for t in [0.5, 0.9, 0.97]:
                pred = ssub["prob_fake"] >= t
                fpr = float(((ssub["label"] == "real") & pred).sum() / max(n_real, 1))
                rec = float(((ssub["label"] == "fake") & pred).sum() / max(n_fake, 1))
                rows.append({
                    "cut_face_pixel_area_min": cut,
                    "split": split,
                    "n_real": n_real,
                    "n_fake": n_fake,
                    "threshold": t,
                    "fpr": fpr,
                    "recall": rec,
                    "frac_kept_real": n_real / max((df[df["split"] == split]["label"] == "real").sum(), 1),
                    "frac_kept_fake": n_fake / max((df[df["split"] == split]["label"] == "fake").sum(), 1),
                })
    e_df = pd.DataFrame(rows)
    print(e_df.to_string(index=False))
    e_df.to_csv(OUT_DIR / "E_deployment_cut_sweep.csv", index=False)

    # ------- Part F: dor_shkedi face_pixel_area distribution -------
    # Sanity check: is dor_shkedi failure size-driven (would fall under the cut)?
    print("\n=== F. dor_shkedi face_pixel_area distribution (lockbox real, R9A 0% accuracy slice) ===")
    dor = df[df["identity_key"] == "dor_shkedi"]
    if len(dor) > 0:
        pa = dor["face_pixel_area"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        print(pa.to_string())
        print(f"\nFraction of dor_shkedi frames with face_pixel_area < 57000: "
              f"{(dor['face_pixel_area'] < 57000).mean():.2%}")
        print(f"Fraction with face_pixel_area < 30000: "
              f"{(dor['face_pixel_area'] < 30000).mean():.2%}")

    # Save a single-page summary for the next agent.
    summary = {
        "generated_at": "2026-04-27 18:00 CEST",
        "parquet": str(PARQUET),
        "model_used_for_predictions": "R9A_run1 (top_n_effort_20260228_step6000_auc0.9891_eer0.0457.pth)",
        "n_rows": int(len(df)),
        "n_dev": int((df["split"] == "dev").sum()),
        "n_lockbox": int((df["split"] == "lockbox").sum()),
        "key_finding_face_size_at_t05_full": {
            "smallest_q1_fpr": "see B_quartile_face_pixel_area_thr0.5.csv",
            "largest_q4_fpr": "see B_quartile_face_pixel_area_thr0.5.csv",
        },
        "outputs": [str(p.name) for p in sorted(OUT_DIR.glob("*.csv"))],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[replicate] outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
