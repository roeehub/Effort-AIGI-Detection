"""Deployment-honest evaluation of P8A using the Vertex batch_inference output.

After `launch_batch_inference.sh` finishes, two CSVs land at:
  gs://training-job-outputs/batch_inference_results/
    p8a_step5000_deployment_honest__teams-faces-data-test-2914-fake-4420-real-feb-28.csv
    p8a_step5000_deployment_honest__teams-faces-data-test-prod-honest-180-2026-04-27.csv

This script:
  - Pulls both CSVs locally
  - Joins the test-bucket predictions with the property-tagged parquet
    (lockbox_tagging/full_tags_2026-04-27.parquet) by gcs_uri so we can
    re-run face_pixel_area + sharpness sweeps on P8A scores instead of R9A
  - Decodes the production-honest pool's regime (OK/FAIL) + tag from the
    encoded filename (we uploaded as `real/<tag>__<participant>__<frame>.png`)
  - Sweeps τ and produces:
      A. Production-honest FPR by τ, broken out by regime (OK / FAIL) and
         per-tag.
      B. Dev fake recall by method @ τ chosen such that production FPR = 5%
         (and at τ ∈ {0.5, 0.7, 0.9, 0.95, 0.99}).
      C. Property-bucket FPR/recall on dev (replicates the R9A finding on P8A).

Run from training/:
  python3 -m analysis.deployment_honest_eval_2026-04-27.run_p8a_deployment_honest
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT_DIR = REPO / "analysis/deployment_honest_eval_2026-04-27"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GCS_RESULTS_PREFIX = "gs://training-job-outputs/batch_inference_results"
RUN_ID = "p8a_step5000_deployment_honest"

CSV_TEST = OUT_DIR / f"{RUN_ID}__teams-faces-data-test-2914-fake-4420-real-feb-28.csv"
CSV_PROD = OUT_DIR / f"{RUN_ID}__teams-faces-data-test-prod-honest-180-2026-04-27.csv"

PROP_PARQUET = REPO / "analysis/lockbox_tagging/full_tags_2026-04-27.parquet"


def fetch_csv(name: str, local: Path) -> bool:
    if local.exists() and local.stat().st_size > 0:
        return True
    uri = f"{GCS_RESULTS_PREFIX}/{name}"
    rc = os.system(f"gsutil -q cp '{uri}' '{local}'")
    return rc == 0 and local.exists()


def fetch_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    ok_test = fetch_csv(CSV_TEST.name, CSV_TEST)
    ok_prod = fetch_csv(CSV_PROD.name, CSV_PROD)
    if not (ok_test and ok_prod):
        raise SystemExit(
            "[deploy-honest] predictions not yet on disk:\n"
            f"  test: exists={ok_test} ({CSV_TEST})\n"
            f"  prod: exists={ok_prod} ({CSV_PROD})\n"
            "Wait for the Vertex job to finish, then re-run."
        )
    df_t = pd.read_csv(CSV_TEST)
    df_p = pd.read_csv(CSV_PROD)
    print(f"[deploy-honest] test bucket: {len(df_t)} rows ({(df_t['label']==1).sum()} fake / {(df_t['label']==0).sum()} real)")
    print(f"[deploy-honest] prod bucket: {len(df_p)} rows (all real, false-flag pool)")
    return df_t, df_p


def decode_regime(frame_name: str) -> tuple[str, str]:
    """Extract (tag, regime) from `<tag>__<participant>__<frame>.png`."""
    tag = frame_name.split("__")[0]
    regime = "FAIL" if "false-flag" in tag else "OK"
    return tag, regime


def production_sweep(df_p: pd.DataFrame) -> pd.DataFrame:
    df = df_p.copy()
    decoded = df["frame_name"].apply(decode_regime)
    df["tag"] = decoded.apply(lambda x: x[0])
    df["regime"] = decoded.apply(lambda x: x[1])

    rows = []
    for t in [0.30, 0.50, 0.70, 0.90, 0.95, 0.97, 0.99, 0.995]:
        for grp_label, sub in [("ALL", df), ("OK", df[df["regime"] == "OK"]), ("FAIL", df[df["regime"] == "FAIL"])]:
            n = len(sub)
            if n == 0:
                continue
            fp = (sub["prob_fake"] >= t).sum()
            rows.append({"threshold": t, "regime": grp_label, "n": n, "fp": int(fp), "fpr": float(fp / n)})
    out = pd.DataFrame(rows)
    print("\n=== Production-honest FPR sweep (180-frame pool) ===")
    print(out.to_string(index=False))
    out.to_csv(OUT_DIR / f"{RUN_ID}_production_fpr_sweep.csv", index=False)

    # Per-tag breakdown at canonical thresholds
    rows = []
    for t in [0.50, 0.90, 0.97, 0.99]:
        for tag, sub in df.groupby("tag"):
            n = len(sub)
            fp = int((sub["prob_fake"] >= t).sum())
            rows.append({
                "threshold": t,
                "tag": tag,
                "n": n,
                "fp": fp,
                "fpr": fp / n,
                "mean_prob_fake": float(sub["prob_fake"].mean()),
            })
    pertag = pd.DataFrame(rows)
    print("\n=== Per-tag production FPR (P8A) ===")
    print(pertag.sort_values(["threshold", "tag"]).to_string(index=False))
    pertag.to_csv(OUT_DIR / f"{RUN_ID}_production_per_tag.csv", index=False)
    return df


def find_threshold_at_fpr(df_prod: pd.DataFrame, target_fpr: float) -> float:
    """Find the smallest τ where FPR_prod ≤ target_fpr."""
    probs = sorted(df_prod["prob_fake"].values, reverse=True)
    n = len(probs)
    max_fp = int(target_fpr * n)
    # The (max_fp+1)-th highest score is the threshold above which we have ≤max_fp FPs
    if max_fp + 1 > n:
        return 0.0
    return float(probs[max_fp])


def dev_fake_recall_at_thresholds(df_t: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    df = df_t[df_t["label"] == 1].copy()  # fakes only
    rows = []
    for t in thresholds:
        for method, sub in df.groupby("method"):
            rec = float((sub["prob_fake"] >= t).sum() / len(sub))
            rows.append({"threshold": t, "method": method, "n": len(sub), "recall": rec})
        # also aggregate
        rec_all = float((df["prob_fake"] >= t).sum() / len(df))
        rows.append({"threshold": t, "method": "ALL", "n": len(df), "recall": rec_all})
    out = pd.DataFrame(rows)
    print("\n=== Dev fake recall by method (P8A) ===")
    print(out.to_string(index=False))
    out.to_csv(OUT_DIR / f"{RUN_ID}_dev_fake_recall.csv", index=False)
    return out


def merge_test_with_property_tags(df_t: pd.DataFrame) -> pd.DataFrame:
    if not PROP_PARQUET.exists():
        print(f"[deploy-honest] property parquet missing at {PROP_PARQUET}; skip property sweep")
        return pd.DataFrame()
    props = pd.read_parquet(PROP_PARQUET)
    # P8A test CSV uses gcs_uri keyed; the parquet also has gcs_uri
    if "gcs_uri" not in props.columns:
        print("[deploy-honest] property parquet missing gcs_uri column; skip")
        return pd.DataFrame()
    keep = [c for c in [
        "gcs_uri", "split", "face_pixel_area", "sharpness_laplacian",
        "brightness_v_mean", "pitch_deg", "yaw_deg", "is_likely_screen_capture",
    ] if c in props.columns]
    merged = df_t.merge(props[keep], on="gcs_uri", how="left")
    matched = merged["face_pixel_area"].notna().sum()
    print(f"[deploy-honest] property-merge: {matched}/{len(merged)} test rows have property tags")
    return merged


def property_quartile_table(df: pd.DataFrame, col: str, threshold: float, n_bins: int = 4) -> pd.DataFrame:
    if col not in df.columns or df[col].isna().all():
        return pd.DataFrame()
    bins = pd.qcut(df[col], q=n_bins, duplicates="drop")
    rows = []
    for bucket, sub in df.groupby(bins, observed=True):
        is_fake = sub["label"] == 1
        n_real = int((~is_fake).sum())
        n_fake = int(is_fake.sum())
        fp = int(((sub["prob_fake"] >= threshold) & ~is_fake).sum())
        tp = int(((sub["prob_fake"] >= threshold) & is_fake).sum())
        rows.append({
            "column": col, "bucket": str(bucket),
            "n": int(len(sub)), "n_real": n_real, "n_fake": n_fake,
            "fpr": fp / max(n_real, 1), "recall": tp / max(n_fake, 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df_t, df_p = fetch_predictions()
    df_p_decoded = production_sweep(df_p)

    # Find τ_5pct on production pool
    tau_5 = find_threshold_at_fpr(df_p_decoded, 0.05)
    tau_3 = find_threshold_at_fpr(df_p_decoded, 0.03)
    print(f"\n[deploy-honest] τ for 5% production FPR: {tau_5:.4f}")
    print(f"[deploy-honest] τ for 3% production FPR: {tau_3:.4f}")

    thresholds = [0.50, 0.70, 0.90, 0.95, 0.97, 0.99, tau_5, tau_3]
    thresholds = sorted(set(round(t, 4) for t in thresholds))

    rec = dev_fake_recall_at_thresholds(df_t, thresholds)

    # Property sweep on dev
    merged = merge_test_with_property_tags(df_t)
    if len(merged):
        for col in ["face_pixel_area", "sharpness_laplacian", "brightness_v_mean", "pitch_deg"]:
            for t in [0.50, tau_5]:
                tab = property_quartile_table(merged, col, threshold=t)
                if len(tab):
                    print(f"\n--- {col} @ τ={t:.4f} ---")
                    print(tab.to_string(index=False))
                    tab.to_csv(OUT_DIR / f"{RUN_ID}_property_{col}_thr{t:.4f}.csv", index=False)

    # Decision table at τ_5pct
    rec_5 = rec[rec["threshold"] == round(tau_5, 4)].set_index("method")["recall"].to_dict()
    rec_50 = rec[rec["threshold"] == 0.50].set_index("method")["recall"].to_dict()
    summary = {
        "checkpoint": "P8A_step5000",
        "n_dev_test_frames": int(len(df_t)),
        "n_production_frames": int(len(df_p)),
        "tau_for_5pct_production_fpr": tau_5,
        "tau_for_3pct_production_fpr": tau_3,
        "production_fpr_at_tau_0.50": float((df_p["prob_fake"] >= 0.50).mean()),
        "production_fpr_at_tau_0.99": float((df_p["prob_fake"] >= 0.99).mean()),
        "dev_recall_at_tau_5pct_per_method": rec_5,
        "dev_recall_at_tau_0.50_per_method": rec_50,
    }
    (OUT_DIR / f"{RUN_ID}_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Decision summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
