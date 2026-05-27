"""Per-method recall on P8A predictions, joined with the property parquet's
method labels (the test-bucket discovery flattens everything to
`teams_passthrough` so we have to merge to recover real method names).

Run from training/:
  python3 -m analysis.deployment_honest_eval_2026-04-27.p8a_per_method
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT = REPO / "analysis/deployment_honest_eval_2026-04-27"
CSV_TEST = OUT / "p8a_step5000_deployment_honest__teams-faces-data-test-2914-fake-4420-real-feb-28.csv"
PROD_CSV = OUT / "p8a_step5000_deployment_honest__teams-faces-data-test-prod-honest-180-2026-04-27.csv"
PARQUET = REPO / "analysis/lockbox_tagging/full_tags_2026-04-27.parquet"

# Method-family aggregations matching the contract scorer suites.
TEAMS_FAKE_FAMILIES = [
    "teams_capture_cam_test_s32", "teams_capture_cam_test_s33",
    "teams_capture_cam_test_s35", "teams_capture_cam_test_s38",
    "teams_capture_cam_test_s46", "teams_capture_test_cam_s53",
    "teams_capture_test_cam_s73", "teams_capture_test_cam_s76",
    "teams_capture_pc_generator_s3", "teams_capture_pc_generator_s4",
    "teams_capture_pc_generator_s9", "teams_capture_pc_generator_s15",
    "teams_capture_dor_shkedi_s16", "teams_capture_noyn_sharker_s23",
    "teams_flat_xiang_xiang2_feng",
]


def main() -> None:
    df = pd.read_csv(CSV_TEST)
    prop = pd.read_parquet(PARQUET)
    prop_keep = prop[["gcs_uri", "method", "label", "split"]].drop_duplicates("gcs_uri")

    # Drop CSV's method (it's bucket-derived "teams_passthrough"/"real") and use the parquet's.
    df_keep = df.drop(columns=["method"], errors="ignore")
    if "label" in df_keep.columns:
        df_keep = df_keep.rename(columns={"label": "label_int"})  # parquet label is str
    merged = df_keep.merge(prop_keep, on="gcs_uri", how="inner")
    print(f"[per-method] merged: {len(merged)} rows (test_csv={len(df)}, parquet={len(prop_keep)})")
    print(f"[per-method] split counts: {merged['split'].value_counts().to_dict()}")
    print(f"[per-method] method top:\n{merged['method'].value_counts().head(20)}")

    # Load production predictions for τ_5pct computation
    prod = pd.read_csv(PROD_CSV)
    probs_prod = sorted(prod["prob_fake"].values, reverse=True)
    n_prod = len(probs_prod)
    tau_5 = float(probs_prod[max(0, int(0.05 * n_prod))])
    tau_3 = float(probs_prod[max(0, int(0.03 * n_prod))])
    print(f"\n[per-method] τ_5pct = {tau_5:.4f}, τ_3pct = {tau_3:.4f}")

    # Per-method recall at canonical thresholds
    fakes = merged[merged["label"] == "fake"]
    rows = []
    for t in [0.50, 0.70, 0.90, 0.95, 0.97, tau_5, 0.99]:
        for m, sub in fakes.groupby("method"):
            n = len(sub)
            rec = float((sub["prob_fake"] >= t).sum() / n)
            rows.append({"threshold": round(t, 4), "method": m, "n": n, "recall": rec})
        # teams_fake_all aggregate (matches contract scorer's group)
        sub_all = fakes[fakes["method"].isin(TEAMS_FAKE_FAMILIES)]
        if len(sub_all):
            rec = float((sub_all["prob_fake"] >= t).sum() / len(sub_all))
            rows.append({"threshold": round(t, 4), "method": "teams_fake_all_AGG", "n": len(sub_all), "recall": rec})
        # deeplive_enhanced single
        sub_dl = fakes[fakes["method"] == "deeplive_enhanced"]
        if len(sub_dl):
            rec = float((sub_dl["prob_fake"] >= t).sum() / len(sub_dl))
            rows.append({"threshold": round(t, 4), "method": "deeplive_enhanced", "n": len(sub_dl), "recall": rec})
    out = pd.DataFrame(rows).drop_duplicates(["threshold", "method"])
    print("\n=== P8A per-method recall ===")
    print(out.sort_values(["method", "threshold"]).to_string(index=False))
    out.to_csv(OUT / "p8a_per_method_recall.csv", index=False)

    # Compact decision table at three operational thresholds
    decision_rows = []
    decision_methods = ["deeplive_enhanced", "teams_fake_all_AGG"] + TEAMS_FAKE_FAMILIES
    for t in [0.50, 0.97, tau_5, 0.99]:
        row = {"threshold": round(t, 4)}
        for m in decision_methods:
            sub = fakes[fakes["method"] == m] if m != "teams_fake_all_AGG" else fakes[fakes["method"].isin(TEAMS_FAKE_FAMILIES)]
            row[m] = float((sub["prob_fake"] >= t).sum() / max(len(sub), 1))
        decision_rows.append(row)
    dec = pd.DataFrame(decision_rows).set_index("threshold")
    print("\n=== P8A: per-method recall by τ (decision table) ===")
    print(dec.T.to_string(float_format=lambda x: f"{x:.3f}"))
    dec.T.to_csv(OUT / "p8a_decision_table.csv")

    # Also: dev real FPR (sanity)
    reals = merged[merged["label"] == "real"]
    fpr_rows = []
    for t in [0.50, 0.97, tau_5, 0.99]:
        fpr = float((reals["prob_fake"] >= t).sum() / len(reals))
        fpr_rows.append({"threshold": round(t, 4), "n_real_dev": len(reals), "fpr_dev_real": fpr})
    print("\n=== Dev real FPR ===")
    print(pd.DataFrame(fpr_rows).to_string(index=False))

    # Final summary
    summary = {
        "checkpoint": "P8A_step5000",
        "tau_5pct_production": tau_5,
        "tau_3pct_production": tau_3,
        "tau_0.99_production_fpr": float((prod["prob_fake"] >= 0.99).mean()),
        "decision_table": dec.T.to_dict(orient="index"),
        "n_test_merged": int(len(merged)),
        "n_test_fakes": int(len(fakes)),
    }
    (OUT / "p8a_per_method_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
