"""Calibration probe for P1 BUNDLE_step500 and PAIRRANK_step500.

Per-frame Platt and isotonic rescaling on identity-stratified hold-out split.

Output:
- calibration_results.csv : full sweep table (raw, Platt, isotonic) x (best_recall_at_FPR<=10%, best_tau)
- per_frame_rescaled.csv  : per-frame raw/Platt/isotonic scores + train/test partition for both ckpts

Constraints:
- CPU only, n_jobs=1, seed=42
- sklearn>=1.0
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit


# ----------------------------- config ---------------------------------------

SEED = 42
HOLDOUT_TEST_FRAC = 0.50  # 50/50 stratified by identity

REPORT_DIR = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/"
    "DeepfakeBench/training/analysis/p1_pe_eval_2026-05-07/raw_reports/phase_a"
)
OUT_DIR = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/"
    "DeepfakeBench/training/analysis/p1_pe_eval_2026-05-07/"
    "calibration_probe_2026-05-07"
)

CKPTS = {
    "p1_bundle_periodic_step500": {
        "real": "teams_real_all_lockbox_p1_bundle_periodic_step500_frames_report.csv",
        "fake": "teams_fake_all_lockbox_p1_bundle_periodic_step500_frames_report.csv",
    },
    "p1_pairrank_periodic_step500": {
        "real": "teams_real_all_lockbox_p1_pairrank_periodic_step500_frames_report.csv",
        "fake": "teams_fake_all_lockbox_p1_pairrank_periodic_step500_frames_report.csv",
    },
}

TAU_GRID = np.linspace(0.05, 0.95, 50)
# Wider grid used only for raw scores so we can compare against the existing
# lockbox_roc_curves.csv. BUNDLE_step500 raw is known to have its FPR<=10%
# feasible region at tau > 0.95 and isn't representable on the 50-point grid
# of the calibrated variants.
TAU_GRID_RAW = np.concatenate(
    [
        np.linspace(0.05, 0.95, 50),
        np.linspace(0.96, 0.999, 40),
    ]
)
FPR_TARGET = 0.10  # F1 close criterion
RECALL_TARGET = 0.90


# --------------------------- helpers ----------------------------------------


def base_identity(video_id: str) -> str:
    return video_id.split("__s")[0]


def load_ckpt_frames(real_csv: Path, fake_csv: Path) -> pd.DataFrame:
    real = pd.read_csv(real_csv)
    fake = pd.read_csv(fake_csv)
    df = pd.concat([real, fake], ignore_index=True)
    df["base_identity"] = df["video_id"].map(base_identity)
    df["score"] = df["frame_prob"].astype(float)
    df["label"] = df["label"].astype(int)
    return df[["video_id", "frame_path", "base_identity", "label", "score"]]


def stratified_split(df: pd.DataFrame, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratify by (label, base_identity) so chronic-6 identities split between cal and test sets.

    Returns boolean masks (calibration_mask, test_mask) for the dataframe.
    """
    # Stratification key combines label and identity so each (identity, label) cell
    # is split independently. This guarantees both reals and fakes from each
    # identity appear in both halves whenever cell size >= 2.
    y = df["label"].astype(str) + "__" + df["base_identity"]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    cal_idx, test_idx = next(sss.split(df.index.to_numpy(), y))
    cal_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)
    cal_mask[cal_idx] = True
    test_mask[test_idx] = True
    return cal_mask, test_mask


def best_recall_at_fpr_le(
    scores: np.ndarray, labels: np.ndarray, tau_grid: np.ndarray, fpr_max: float
) -> tuple[float, float, float, bool]:
    """Sweep tau and return (best_recall, best_tau, fpr_at_best, feasible).

    Picks the tau on the grid that maximises recall subject to FPR <= fpr_max.
    Ties broken by lower FPR, then lower tau. `feasible` is False when no tau on
    the grid satisfies the FPR constraint -- in that case the returned best
    is computed over the closest-to-target slice of the grid (the smallest-FPR
    tau on the grid, which is the most conservative achievable point).
    """
    real_mask = labels == 0
    fake_mask = labels == 1
    n_real = int(real_mask.sum())
    n_fake = int(fake_mask.sum())
    if n_real == 0 or n_fake == 0:
        return float("nan"), float("nan"), float("nan"), False

    rows = []
    for tau in tau_grid:
        preds = scores >= tau
        fp = int((preds & real_mask).sum())
        tp = int((preds & fake_mask).sum())
        fpr = fp / n_real
        recall = tp / n_fake
        rows.append((float(tau), float(fpr), float(recall)))

    feasible = [r for r in rows if r[1] <= fpr_max + 1e-12]
    has_feasible = len(feasible) > 0
    if not has_feasible:
        # Fall back to the smallest-FPR tau on the grid (most conservative)
        feasible = sorted(rows, key=lambda r: (r[1], -r[2], r[0]))[:1]
    # max recall, tie-break by lower fpr, then lower tau
    feasible.sort(key=lambda r: (-r[2], r[1], r[0]))
    best = feasible[0]
    return best[2], best[0], best[1], has_feasible


# --------------------------- main -------------------------------------------


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)

    # Top-level summary table
    summary_rows = []
    per_frame_rows = []

    for ckpt, paths in CKPTS.items():
        real_csv = REPORT_DIR / paths["real"]
        fake_csv = REPORT_DIR / paths["fake"]
        df = load_ckpt_frames(real_csv, fake_csv)

        cal_mask, test_mask = stratified_split(df, HOLDOUT_TEST_FRAC, SEED)

        cal = df.loc[cal_mask].reset_index(drop=True)
        test = df.loc[test_mask].reset_index(drop=True)

        # Sanity: identity coverage on each side
        cal_ids = set(cal["base_identity"].unique())
        test_ids = set(test["base_identity"].unique())
        n_cal_real = int((cal["label"] == 0).sum())
        n_cal_fake = int((cal["label"] == 1).sum())
        n_test_real = int((test["label"] == 0).sum())
        n_test_fake = int((test["label"] == 1).sum())

        print(f"\n[{ckpt}]")
        print(
            f"  cal: n={len(cal)} (real={n_cal_real}, fake={n_cal_fake}); "
            f"test: n={len(test)} (real={n_test_real}, fake={n_test_fake})"
        )
        print(f"  cal identities: {sorted(cal_ids)}")
        print(f"  test identities: {sorted(test_ids)}")
        print(f"  identity overlap (chronic-6 stratification): {sorted(cal_ids & test_ids)}")

        cal_scores = cal["score"].to_numpy()
        cal_labels = cal["label"].to_numpy()
        test_scores = test["score"].to_numpy()
        test_labels = test["label"].to_numpy()

        # ------- Fit calibrators on cal set -------
        # Platt: logistic regression on raw scores -> sigmoid rescaling.
        platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, n_jobs=1)
        platt.fit(cal_scores.reshape(-1, 1), cal_labels)
        # Isotonic: monotone non-parametric calibrator
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(cal_scores, cal_labels)

        # Apply on test
        test_platt = platt.predict_proba(test_scores.reshape(-1, 1))[:, 1]
        test_iso = iso.predict(test_scores)

        # Sweep test
        for variant, scrs in [("raw", test_scores), ("platt", test_platt), ("isotonic", test_iso)]:
            grid = TAU_GRID_RAW if variant == "raw" else TAU_GRID
            recall, tau, fpr, feas = best_recall_at_fpr_le(
                scrs, test_labels, grid, FPR_TARGET
            )
            summary_rows.append(
                {
                    "ckpt": ckpt,
                    "variant": variant,
                    "best_recall_at_FPR_le_10": recall,
                    "best_tau": tau,
                    "fpr_at_best": fpr,
                    "feasible_on_grid": feas,
                    "tau_in_contract_friendly_range_0p5_0p85": (
                        feas and 0.5 <= tau <= 0.85
                    ),
                    "test_n": len(test),
                    "test_n_real": n_test_real,
                    "test_n_fake": n_test_fake,
                    "cal_n": len(cal),
                    "cal_n_real": n_cal_real,
                    "cal_n_fake": n_cal_fake,
                }
            )

        # Per-frame export (useful for follow-up)
        for partition, sub_df, sub_idx in [
            ("cal", cal, np.where(cal_mask)[0]),
            ("test", test, np.where(test_mask)[0]),
        ]:
            sub_scores = sub_df["score"].to_numpy()
            sub_platt = platt.predict_proba(sub_scores.reshape(-1, 1))[:, 1]
            sub_iso = iso.predict(sub_scores)
            for i in range(len(sub_df)):
                per_frame_rows.append(
                    {
                        "ckpt": ckpt,
                        "partition": partition,
                        "row_idx": int(sub_idx[i]),
                        "video_id": sub_df["video_id"].iat[i],
                        "base_identity": sub_df["base_identity"].iat[i],
                        "label": int(sub_df["label"].iat[i]),
                        "raw_score": float(sub_scores[i]),
                        "platt_score": float(sub_platt[i]),
                        "iso_score": float(sub_iso[i]),
                    }
                )

        # Also dump full ROC sweep table for each variant on test set
        roc_rows = []
        for variant, scrs in [("raw", test_scores), ("platt", test_platt), ("isotonic", test_iso)]:
            real_mask = test_labels == 0
            fake_mask = test_labels == 1
            n_real = int(real_mask.sum())
            n_fake = int(fake_mask.sum())
            grid = TAU_GRID_RAW if variant == "raw" else TAU_GRID
            for tau in grid:
                preds = scrs >= tau
                fp = int((preds & real_mask).sum())
                tp = int((preds & fake_mask).sum())
                roc_rows.append(
                    {
                        "ckpt": ckpt,
                        "variant": variant,
                        "tau": float(tau),
                        "fpr": fp / n_real,
                        "recall": tp / n_fake,
                        "fp": fp,
                        "tp": tp,
                        "n_real": n_real,
                        "n_fake": n_fake,
                    }
                )
        roc_df = pd.DataFrame(roc_rows)
        roc_df.to_csv(OUT_DIR / f"roc_sweep_{ckpt}.csv", index=False)

    # Save summary + per-frame
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "calibration_results.csv", index=False)
    pd.DataFrame(per_frame_rows).to_csv(OUT_DIR / "per_frame_rescaled.csv", index=False)

    print("\n=== summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {OUT_DIR / 'calibration_results.csv'}")
    print(f"Saved: {OUT_DIR / 'per_frame_rescaled.csv'}")


if __name__ == "__main__":
    main()
