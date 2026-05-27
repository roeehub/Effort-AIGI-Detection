"""Empirical face-size filter analysis on P8A's lockbox + dev predictions.

Joins P8A predictions to the lockbox-tagging parquet, computes baseline FPR/recall,
then sweeps face-size and sharpness thresholds to find a "well-conditioned crops"
subset that materially drops FPR without crushing fake recall.

This answers Plan v3 §A.5c: does data hygiene alone solve the FPR side?
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
PARQUET = REPO / "analysis/lockbox_tagging/full_tags_2026-04-27.parquet"
P8A_CSV = REPO / "analysis/deployment_honest_eval_2026-04-27/p8a_step5000_deployment_honest__teams-faces-data-test-2914-fake-4420-real-feb-28.csv"
OUT_DIR = REPO / "analysis/crop_shortcut_2026-04-27"


def fpr_recall(df: pd.DataFrame, tau: float = 0.5) -> dict:
    pred = (df["prob_fake_p8a"] >= tau).astype(int)
    real = df[df["label"] == "real"]
    fake = df[df["label"] == "fake"]
    return {
        "n_real": len(real),
        "n_fake": len(fake),
        "fpr": float((real["prob_fake_p8a"] >= tau).mean()) if len(real) else float("nan"),
        "recall": float((fake["prob_fake_p8a"] >= tau).mean()) if len(fake) else float("nan"),
    }


def main() -> None:
    tags = pd.read_parquet(PARQUET)
    print(f"[analysis] tags rows: {len(tags)}, cols: {len(tags.columns)}")

    p8a = pd.read_csv(P8A_CSV)
    p8a = p8a.rename(columns={"prob_fake": "prob_fake_p8a"})
    print(f"[analysis] p8a rows: {len(p8a)}")

    # Inner join on gcs_uri.
    df = tags.merge(p8a[["gcs_uri", "prob_fake_p8a"]], on="gcs_uri", how="inner")
    print(f"[analysis] joined rows: {len(df)}")
    print(f"[analysis] split x label after join:")
    print(df.groupby(["split", "label"]).size())

    # =================== BASELINE ===================
    print("\n=== BASELINE (no filter) ===")
    for split in ("dev", "lockbox"):
        sub = df[df["split"] == split]
        m = fpr_recall(sub, 0.5)
        print(f"  {split}: n_real={m['n_real']:>4}, n_fake={m['n_fake']:>4}, "
              f"FPR={m['fpr']:.3%}, recall={m['recall']:.3%}")

    # =================== FILTER SWEEPS ===================
    # 1. Face-size filter (face_pixel_area floor).
    print("\n=== FILTER A: face_pixel_area floor sweep (lockbox) ===")
    print(f"{'min_face_px²':<14} {'n_real':<7} {'n_fake':<7} {'%real_kept':<11} {'%fake_kept':<11} {'FPR':<8} {'recall':<8}")
    lb = df[df["split"] == "lockbox"].copy()
    lb_n_real_all = (lb["label"] == "real").sum()
    lb_n_fake_all = (lb["label"] == "fake").sum()
    for min_fpa in [0, 5000, 10000, 20000, 30000, 50000, 75000, 100000]:
        f = lb[lb["face_pixel_area"] >= min_fpa]
        m = fpr_recall(f, 0.5)
        n_real_kept = m["n_real"]
        n_fake_kept = m["n_fake"]
        print(f"{min_fpa:<14} {n_real_kept:<7} {n_fake_kept:<7} "
              f"{n_real_kept / lb_n_real_all:<11.1%} {n_fake_kept / lb_n_fake_all:<11.1%} "
              f"{m['fpr']:<8.2%} {m['recall']:<8.2%}")

    # Same for dev (~6500 frames is the meaningful sample).
    print("\n=== FILTER A: face_pixel_area floor sweep (dev) ===")
    dev = df[df["split"] == "dev"].copy()
    dev_n_real_all = (dev["label"] == "real").sum()
    dev_n_fake_all = (dev["label"] == "fake").sum()
    print(f"{'min_face_px²':<14} {'n_real':<7} {'n_fake':<7} {'%real_kept':<11} {'%fake_kept':<11} {'FPR':<8} {'recall':<8}")
    for min_fpa in [0, 5000, 10000, 20000, 30000, 50000, 75000, 100000]:
        f = dev[dev["face_pixel_area"] >= min_fpa]
        m = fpr_recall(f, 0.5)
        n_real_kept = m["n_real"]
        n_fake_kept = m["n_fake"]
        print(f"{min_fpa:<14} {n_real_kept:<7} {n_fake_kept:<7} "
              f"{n_real_kept / dev_n_real_all:<11.1%} {n_fake_kept / dev_n_fake_all:<11.1%} "
              f"{m['fpr']:<8.2%} {m['recall']:<8.2%}")

    # 2. Sharpness floor sweep.
    print("\n=== FILTER B: sharpness_laplacian floor sweep (lockbox) ===")
    print(f"{'min_sharp':<10} {'n_real':<7} {'n_fake':<7} {'%real_kept':<11} {'%fake_kept':<11} {'FPR':<8} {'recall':<8}")
    for min_sh in [0, 5, 10, 20, 50, 100, 200]:
        f = lb[lb["sharpness_laplacian"] >= min_sh]
        m = fpr_recall(f, 0.5)
        n_real_kept = m["n_real"]
        n_fake_kept = m["n_fake"]
        print(f"{min_sh:<10} {n_real_kept:<7} {n_fake_kept:<7} "
              f"{n_real_kept / lb_n_real_all:<11.1%} {n_fake_kept / lb_n_fake_all:<11.1%} "
              f"{m['fpr']:<8.2%} {m['recall']:<8.2%}")

    # 3. is_likely_screen_capture flag (the agent's signal).
    print("\n=== FILTER C: is_likely_screen_capture (lockbox + dev real only) ===")
    if "is_likely_screen_capture" in df.columns:
        for split in ("dev", "lockbox"):
            sub = df[df["split"] == split]
            for screencap in (True, False, "all"):
                if screencap == "all":
                    f = sub
                    label = "all"
                else:
                    f = sub[sub["is_likely_screen_capture"] == screencap]
                    label = "yes" if screencap else "no"
                m = fpr_recall(f, 0.5)
                print(f"  {split:<8} screencap={label:<4}  n_real={m['n_real']:<5} n_fake={m['n_fake']:<5} "
                      f"FPR={m['fpr']:.2%} recall={m['recall']:.2%}")

    # 4. clip_capture_mode breakdown.
    if "clip_capture_mode" in df.columns:
        print("\n=== FILTER D: clip_capture_mode breakdown (lockbox real) ===")
        lb_real = lb[lb["label"] == "real"]
        modes = lb_real["clip_capture_mode"].value_counts()
        print(f"  modes seen: {modes.to_dict()}")
        for mode, _ in modes.items():
            f = lb_real[lb_real["clip_capture_mode"] == mode]
            fpr = float((f["prob_fake_p8a"] >= 0.5).mean())
            print(f"    {mode:<25}  n={len(f)}  FPR={fpr:.2%}")

    # 5. Combined "well-conditioned" filter: face_pixel_area >= 20000 AND sharpness >= 5
    print("\n=== FILTER E: COMBINED — face_pixel_area >= 20000 AND sharpness >= 5 ===")
    for split in ("dev", "lockbox"):
        sub = df[df["split"] == split]
        f = sub[(sub["face_pixel_area"] >= 20000) & (sub["sharpness_laplacian"] >= 5)]
        m_all = fpr_recall(sub, 0.5)
        m = fpr_recall(f, 0.5)
        print(f"  {split:<8} ALL: n_real={m_all['n_real']:<5} FPR={m_all['fpr']:.2%}, "
              f"recall={m_all['recall']:.2%}")
        print(f"  {split:<8} FILTERED: n_real={m['n_real']:<5} FPR={m['fpr']:.2%}, "
              f"recall={m['recall']:.2%}  ({m['n_real']}/{m_all['n_real']} real, "
              f"{m['n_fake']}/{m_all['n_fake']} fake kept)")

    # 6. Per-identity FPR (lockbox only, 5 identities).
    print("\n=== Per-identity FPR (lockbox real) ===")
    for ikey, g in lb[lb["label"] == "real"].groupby("identity_key"):
        fpr = float((g["prob_fake_p8a"] >= 0.5).mean())
        med_fpa = g["face_pixel_area"].median()
        print(f"  {str(ikey)[:30]:<32}  n={len(g):<4}  FPR={fpr:.2%}  median_face_px²={med_fpa:.0f}")

    # Save merged frame for further analysis.
    out_csv = OUT_DIR / "p8a_lockbox_join_2026-04-27.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[analysis] saved joined data: {out_csv}")


if __name__ == "__main__":
    main()
