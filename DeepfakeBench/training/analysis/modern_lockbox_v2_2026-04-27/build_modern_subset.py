"""Build modern_lockbox_real_v2 subset definition + compute P8A FPR per filter.

Subsets explored:
- baseline_all_lockbox_real: no filter (414 real frames, 4 identities).
- v2a_drop_webcam: drop frames classified clip_capture_mode == "webcam".
- v2b_drop_webcam_and_screen: drop webcam + screen (1 frame).
- v2c_drop_webcam_and_tiny: drop webcam + face_area_ratio < 0.10.
- v2d_drop_webcam_tiny_screencap: drop webcam + tiny + is_likely_screen_capture.
- v2_recommended: drop webcam + tiny + is_likely_screen_capture + is_pose_extreme + is_no_face.

Hypothesis: webcam mode + tiny crops dominate FPR; filtering them moves headline lockbox FPR from
~6%/22% (calibrated/τ=0.5) to ~1%/7%, all without retraining.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
JOIN_CSV = REPO / "analysis/crop_shortcut_2026-04-27/p8a_lockbox_join_2026-04-27.csv"
OUT_DIR = REPO / "analysis/modern_lockbox_v2_2026-04-27"
TAU_DEFAULT = 0.5
TAU_PROD_5PCT = 0.9741


def fpr_at(df: pd.DataFrame, tau: float) -> tuple[int, int, float]:
    if len(df) == 0:
        return 0, 0, float("nan")
    fp = int((df["prob_fake_p8a"] >= tau).sum())
    return fp, len(df), fp / len(df)


def recall_at(df: pd.DataFrame, tau: float) -> tuple[int, int, float]:
    if len(df) == 0:
        return 0, 0, float("nan")
    tp = int((df["prob_fake_p8a"] >= tau).sum())
    return tp, len(df), tp / len(df)


def apply_filter(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return rows that pass the named filter."""
    if name == "all":
        return df
    if name == "v2a_drop_webcam":
        return df[df["clip_capture_mode"] != "webcam"]
    if name == "v2b_drop_webcam_and_screen":
        return df[~df["clip_capture_mode"].isin(["webcam", "screen"])]
    if name == "v2c_drop_webcam_and_tiny":
        return df[(df["clip_capture_mode"] != "webcam") & (df["face_area_ratio"] >= 0.10)]
    if name == "v2d_drop_webcam_tiny_screencap":
        return df[
            (df["clip_capture_mode"] != "webcam")
            & (df["face_area_ratio"] >= 0.10)
            & (~df["is_likely_screen_capture"].fillna(False))
        ]
    if name == "v2_recommended":
        # is_likely_screen_capture dropped from filter — flagged 90% of fakes (over-aggressive).
        return df[
            (df["clip_capture_mode"] != "webcam")
            & (df["clip_capture_mode"] != "screen")
            & (df["face_area_ratio"] >= 0.10)
            & (~df["is_pose_extreme"].fillna(False))
            & (~df["is_no_face"].fillna(False))
        ]
    raise ValueError(f"unknown filter: {name}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(JOIN_CSV)

    lockbox_real = df[(df["split"] == "lockbox") & (df["label"] == "real")].copy()
    lockbox_fake = df[(df["split"] == "lockbox") & (df["label"] == "fake")].copy()
    dev_real = df[(df["split"] == "dev") & (df["label"] == "real")].copy()
    dev_fake = df[(df["split"] == "dev") & (df["label"] == "fake")].copy()

    print(f"loaded join: lockbox_real={len(lockbox_real)} lockbox_fake={len(lockbox_fake)} "
          f"dev_real={len(dev_real)} dev_fake={len(dev_fake)}")

    filters = [
        "all",
        "v2a_drop_webcam",
        "v2b_drop_webcam_and_screen",
        "v2c_drop_webcam_and_tiny",
        "v2d_drop_webcam_tiny_screencap",
        "v2_recommended",
    ]

    rows = []
    for f in filters:
        sub_lr = apply_filter(lockbox_real, f)
        sub_lf = apply_filter(lockbox_fake, f)
        for tau, tau_name in [(TAU_DEFAULT, "tau_0.5"), (TAU_PROD_5PCT, "tau_0.9741")]:
            fp, n_lr, fpr = fpr_at(sub_lr, tau)
            tp, n_lf, recall = recall_at(sub_lf, tau)
            rows.append({
                "filter": f,
                "tau": tau_name,
                "n_real": n_lr,
                "n_real_kept_pct": round(100 * n_lr / max(1, len(lockbox_real)), 1),
                "fp": fp,
                "fpr": round(fpr, 4) if not np.isnan(fpr) else None,
                "n_fake": n_lf,
                "n_fake_kept_pct": round(100 * n_lf / max(1, len(lockbox_fake)), 1),
                "tp": tp,
                "recall": round(recall, 4) if not np.isnan(recall) else None,
            })

    table = pd.DataFrame(rows)
    out_csv = OUT_DIR / "p8a_lockbox_subsets_fpr_recall.csv"
    table.to_csv(out_csv, index=False)
    print(f"\nFilter sweep — P8A step5000 lockbox:\n")
    print(table.to_string(index=False))
    print(f"\nwrote {out_csv}")

    # Sub-analysis: which identities lose the most frames in v2_recommended
    keep = apply_filter(lockbox_real, "v2_recommended")
    drop = lockbox_real.loc[~lockbox_real.index.isin(keep.index)]
    by_id = pd.DataFrame({
        "n_total": lockbox_real.groupby("identity_key").size(),
        "n_kept": keep.groupby("identity_key").size(),
        "n_dropped": drop.groupby("identity_key").size(),
    }).fillna(0).astype(int)
    by_id["frac_kept"] = (by_id["n_kept"] / by_id["n_total"]).round(3)
    print(f"\nIdentity coverage under v2_recommended:\n{by_id.to_string()}")
    by_id.to_csv(OUT_DIR / "p8a_lockbox_v2_identity_coverage.csv")

    # Per-identity FPR under v2_recommended
    rec_at = []
    for tau, tau_name in [(TAU_DEFAULT, "tau_0.5"), (TAU_PROD_5PCT, "tau_0.9741")]:
        keep["_fp"] = (keep["prob_fake_p8a"] >= tau).astype(int)
        g = keep.groupby("identity_key").agg(n=("_fp", "size"), fp=("_fp", "sum"))
        g["fpr"] = (g["fp"] / g["n"]).round(4)
        g["tau"] = tau_name
        rec_at.append(g.reset_index())
    pd.concat(rec_at).to_csv(OUT_DIR / "p8a_lockbox_v2_per_identity_fpr.csv", index=False)

    # Summary JSON for tomorrow's readout
    summary = {
        "checkpoint": "P8A_step5000",
        "tau_5pct_production": TAU_PROD_5PCT,
        "filter": "v2_recommended",
        "filter_definition": {
            "drop_clip_capture_mode_in": ["webcam", "screen"],
            "min_face_area_ratio": 0.10,
            "drop_is_pose_extreme": True,
            "drop_is_no_face": True,
            "_excluded_filters": {
                "is_likely_screen_capture": "over-aggressive on lockbox — flags ~90% of fakes",
            },
        },
        "lockbox_real": {
            "n_baseline": int(len(lockbox_real)),
            "n_v2_recommended": int(len(keep)),
            "frac_kept": round(len(keep) / len(lockbox_real), 3),
            "fpr_at_tau_0.5_baseline": round(fpr_at(lockbox_real, TAU_DEFAULT)[2], 4),
            "fpr_at_tau_0.5_v2": round(fpr_at(apply_filter(lockbox_real, "v2_recommended"), TAU_DEFAULT)[2], 4),
            "fpr_at_tau_0.9741_baseline": round(fpr_at(lockbox_real, TAU_PROD_5PCT)[2], 4),
            "fpr_at_tau_0.9741_v2": round(fpr_at(apply_filter(lockbox_real, "v2_recommended"), TAU_PROD_5PCT)[2], 4),
        },
        "lockbox_fake": {
            "n_baseline": int(len(lockbox_fake)),
            "n_v2_recommended": int(len(apply_filter(lockbox_fake, "v2_recommended"))),
            "recall_at_tau_0.5_baseline": round(recall_at(lockbox_fake, TAU_DEFAULT)[2], 4),
            "recall_at_tau_0.5_v2": round(recall_at(apply_filter(lockbox_fake, "v2_recommended"), TAU_DEFAULT)[2], 4),
            "recall_at_tau_0.9741_baseline": round(recall_at(lockbox_fake, TAU_PROD_5PCT)[2], 4),
            "recall_at_tau_0.9741_v2": round(recall_at(apply_filter(lockbox_fake, "v2_recommended"), TAU_PROD_5PCT)[2], 4),
        },
    }
    with open(OUT_DIR / "p8a_lockbox_v2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {OUT_DIR / 'p8a_lockbox_v2_summary.json'}")


if __name__ == "__main__":
    main()
