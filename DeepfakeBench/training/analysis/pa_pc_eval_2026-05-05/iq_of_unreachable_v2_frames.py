#!/usr/bin/env python3
"""
What do the 364 unreachable v2 viso fakes share?

Per Job 12 (memory project_job12_ensemble_ceiling_2026-05-04): 364 of 550 viso
fakes are uncaught by P8A_step5000, E2B_top_n_step3200, E3_top_n_step6600 even
with each ckpt at its own FPR=10% budget (loose-OR Venn). This script
characterizes the IQ properties of those 364 vs the 186 caught.

Outputs: analysis/pa_pc_eval_2026-05-05/iq_unreachable_v2_findings.json + .csv
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
RAW = ROOT / "analysis/cpu_followups_2026-05-04/raw_reports"
PARQUET = ROOT / "analysis/lockbox_tagging/full_tags_2026-04-27.parquet"
OUT = ROOT / "analysis/pa_pc_eval_2026-05-05"

CKPTS = {
    "P8A": {
        "real_dev_csv": RAW / "teams_real_all_dev_p8a_reference_step5000_frames_report.csv",
        "viso_csv": RAW / "visomaster_enhanced_macro_dev_p8a_reference_step5000_frames_report.csv",
    },
    "E2B": {
        "real_dev_csv": RAW / "teams_real_all_dev_e2b_top_n_step3200_frames_report.csv",
        "viso_csv": RAW / "visomaster_enhanced_macro_dev_e2b_top_n_step3200_frames_report.csv",
    },
    "E3": {
        "real_dev_csv": RAW / "teams_real_all_dev_e3_top_n_step6600_frames_report.csv",
        "viso_csv": RAW / "visomaster_enhanced_macro_dev_e3_top_n_step6600_frames_report.csv",
    },
}
TARGET_FPR = 0.10


def calibrate_tau(real_csv, target_fpr):
    """Find smallest tau s.t. fpr(real, tau) <= target_fpr."""
    df = pd.read_csv(real_csv)
    scores = sorted(df["frame_prob"].dropna().tolist(), reverse=True)
    n = len(scores)
    cap = int(n * target_fpr)
    if cap <= 0:
        return 1.0
    return scores[cap - 1]


def main():
    # Load real_dev for tau calibration
    catches = {}
    taus = {}
    for ck, paths in CKPTS.items():
        tau = calibrate_tau(paths["real_dev_csv"], TARGET_FPR)
        taus[ck] = tau
        viso = pd.read_csv(paths["viso_csv"])
        viso = viso[viso["label"] == 1] if "label" in viso.columns else viso
        viso["caught"] = (viso["frame_prob"] >= tau).astype(int)
        catches[ck] = viso[["frame_path", "frame_prob", "caught"]].rename(
            columns={"frame_prob": f"score_{ck}", "caught": f"caught_{ck}"}
        )
        print(f"{ck}: tau={tau:.4f}, n_viso={len(viso)}, n_caught={int(viso['caught'].sum())} ({100*viso['caught'].mean():.1f}%)")

    # Merge per-frame
    merged = catches["P8A"].merge(catches["E2B"], on="frame_path", how="outer").merge(catches["E3"], on="frame_path", how="outer")
    merged["caught_any"] = (
        merged["caught_P8A"].fillna(0).astype(int)
        | merged["caught_E2B"].fillna(0).astype(int)
        | merged["caught_E3"].fillna(0).astype(int)
    )
    merged["caught_all"] = (
        merged["caught_P8A"].fillna(0).astype(int)
        & merged["caught_E2B"].fillna(0).astype(int)
        & merged["caught_E3"].fillna(0).astype(int)
    )
    n = len(merged)
    n_any = int(merged["caught_any"].sum())
    n_all = int(merged["caught_all"].sum())
    n_uncaught = n - n_any
    print(f"\nN={n}; caught_any (loose OR): {n_any} ({100*n_any/n:.1f}%); caught_all (intersection): {n_all} ({100*n_all/n:.1f}%); uncaught_by_any: {n_uncaught} ({100*n_uncaught/n:.1f}%)")

    # Load IQ tags from parquet (parquet uses gcs_uri + sharpness_laplacian etc.)
    tags = pd.read_parquet(PARQUET)
    interesting = ["gcs_uri", "sharpness_laplacian", "brightness_v_mean", "brightness_v_std",
                   "contrast_rms", "saturation_s_mean", "width", "height", "face_pixel_area",
                   "face_area_ratio", "is_no_face", "clip_capture_mode", "identity_key"]
    tag_cols = [c for c in interesting if c in tags.columns]
    print(f"\nParquet has {len(tags)} rows, tag cols available: {tag_cols}")
    tags_keep = tags[tag_cols].copy()
    tags_keep = tags_keep.rename(columns={"gcs_uri": "frame_path"})
    merged = merged.merge(tags_keep, on="frame_path", how="left")
    iq_keys = [c for c in ["sharpness_laplacian", "brightness_v_mean", "brightness_v_std",
                            "contrast_rms", "saturation_s_mean", "width", "height",
                            "face_pixel_area", "face_area_ratio"] if c in merged.columns]
    n_with_tags = int(merged[iq_keys[0]].notna().sum()) if iq_keys else 0
    print(f"Frames with IQ tags: {n_with_tags} of {n}")
    summary = {"n_total": n, "n_caught_any": n_any, "n_uncaught": n_uncaught, "taus": taus, "iq_compare": {}}
    for k in iq_keys:
        caught_subset = merged[merged["caught_any"] == 1][k].dropna()
        uncaught_subset = merged[merged["caught_any"] == 0][k].dropna()
        if len(caught_subset) < 5 or len(uncaught_subset) < 5:
            continue
        from scipy.stats import mannwhitneyu
        try:
            u, p = mannwhitneyu(caught_subset, uncaught_subset, alternative="two-sided")
            summary["iq_compare"][k] = {
                "n_caught": int(len(caught_subset)),
                "n_uncaught": int(len(uncaught_subset)),
                "median_caught": float(caught_subset.median()),
                "median_uncaught": float(uncaught_subset.median()),
                "mean_caught": float(caught_subset.mean()),
                "mean_uncaught": float(uncaught_subset.mean()),
                "ratio_caught_over_uncaught": float(caught_subset.median() / max(uncaught_subset.median(), 1e-6)),
                "mannwhitney_u": float(u),
                "mannwhitney_p": float(p),
            }
        except Exception as e:
            summary["iq_compare"][k] = {"err": str(e)}

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "iq_unreachable_v2_findings.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    merged.to_csv(OUT / "iq_unreachable_v2_per_frame.csv", index=False)
    print(f"\nWrote {OUT/'iq_unreachable_v2_findings.json'}")
    print(f"Wrote {OUT/'iq_unreachable_v2_per_frame.csv'}")

    # Print headline
    print("\n=== HEADLINE: caught vs uncaught IQ medians ===")
    for k, d in summary["iq_compare"].items():
        if "median_caught" not in d:
            continue
        print(f"{k:20s}: caught {d['median_caught']:.2f} | uncaught {d['median_uncaught']:.2f} | ratio {d['ratio_caught_over_uncaught']:.2f}× | p={d['mannwhitney_p']:.2e}")

    # Capture mode breakdown
    if "clip_capture_mode" in merged.columns:
        print("\n=== Capture mode breakdown of uncaught ===")
        mode_counts = merged[merged["caught_any"] == 0]["clip_capture_mode"].value_counts(dropna=False)
        print(mode_counts.to_string())


if __name__ == "__main__":
    main()
