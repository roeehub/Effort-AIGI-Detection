#!/usr/bin/env python3
"""
Supplementary: per-ckpt-independent thresholds calibrated so the *union* (OR)
on teams_real_all_dev hits the FPR target. This is the apples-to-apples ceiling
for "any ckpt fires" detection at the same dev FPR budget.

Strategy: equal-FPR-share — each ckpt gets τ at fpr_target/3 marginal; then
walk τ jointly down until union FPR hits target. We use a simple grid search.
"""
import os
import numpy as np
import pandas as pd

ROOT = "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis"
RAW = os.path.join(ROOT, "cpu_followups_2026-05-04", "raw_reports")
OUT = os.path.join(ROOT, "job_12_ensemble_ceiling_2026-05-04")

CKPTS = ["P8A", "E2B", "E3"]
CKPT_FILE_TAGS = {
    "P8A": "p8a_reference_step5000",
    "E2B": "e2b_top_n_step3200",
    "E3":  "e3_top_n_step6600",
}
FAKE_SUITES = [
    "visomaster_enhanced_macro_dev",
    "deeplive_enhanced_dev",
    "teams_fake_all_dev",
    "teams_fake_all_lockbox",
]
REAL_SUITES = ["teams_real_all_dev", "teams_real_all_lockbox"]
ALL_SUITES = REAL_SUITES + FAKE_SUITES
CALIB_REAL_SUITE = "teams_real_all_dev"

FPR_TARGETS = [0.02, 0.05, 0.10, 0.20]


def load_suite(suite: str) -> pd.DataFrame:
    dfs = {}
    for ckpt in CKPTS:
        path = os.path.join(RAW, f"{suite}_{CKPT_FILE_TAGS[ckpt]}_frames_report.csv")
        df = pd.read_csv(path, usecols=["frame_path", "frame_prob", "label"])
        df = df.rename(columns={"frame_prob": f"prob_{ckpt}"})
        dfs[ckpt] = df
    base = dfs["P8A"][["frame_path", "label", "prob_P8A"]].copy()
    for ckpt in ["E2B", "E3"]:
        base = base.merge(dfs[ckpt][["frame_path", f"prob_{ckpt}"]], on="frame_path", how="inner")
    return base


print("[Job 12 OR] Loading...")
DATA = {s: load_suite(s) for s in ALL_SUITES}
calib = DATA[CALIB_REAL_SUITE]
n_calib = len(calib)


def fpr_for_taus(scores_p, scores_e2, scores_e3, tp, te2, te3):
    fired = (scores_p >= tp) | (scores_e2 >= te2) | (scores_e3 >= te3)
    return float(fired.mean()), int(fired.sum())


def recall_for_taus(df, tp, te2, te3):
    fired = (df["prob_P8A"].values >= tp) | (df["prob_E2B"].values >= te2) | (df["prob_E3"].values >= te3)
    return float(fired.mean()), int(fired.sum()), len(df)


# Strategy A — equal marginal FPR share: each τ at quantile (1 - fpr_target/3)
# Strategy B — equal scaled τ: same τ_quantile applied to each ckpt's own quantile
# We pick A as the simple symmetric apples-to-apples union baseline.
records = []
for fpr_target in FPR_TARGETS:
    # Strategy A: marginal share
    margin = fpr_target / 3.0
    tA = {ckpt: float(np.quantile(calib[f"prob_{ckpt}"].values, 1.0 - margin)) for ckpt in CKPTS}
    union_fpr_A, _ = fpr_for_taus(
        calib["prob_P8A"].values, calib["prob_E2B"].values, calib["prob_E3"].values,
        tA["P8A"], tA["E2B"], tA["E3"],
    )

    # Strategy B: search the *single shared quantile q* such that union FPR == target.
    # Apply q to each ckpt's own marginal: tau_ckpt = quantile(real_ckpt, 1-q).
    # binary search over q in [target/3, target]
    lo, hi = max(fpr_target / 3.0, 1e-4), min(fpr_target, 0.5)
    for _ in range(60):
        mid = (lo + hi) / 2
        taus = {ckpt: float(np.quantile(calib[f"prob_{ckpt}"].values, 1.0 - mid)) for ckpt in CKPTS}
        u, _ = fpr_for_taus(
            calib["prob_P8A"].values, calib["prob_E2B"].values, calib["prob_E3"].values,
            taus["P8A"], taus["E2B"], taus["E3"],
        )
        if u > fpr_target:
            hi = mid
        else:
            lo = mid
    qB = (lo + hi) / 2
    tB = {ckpt: float(np.quantile(calib[f"prob_{ckpt}"].values, 1.0 - qB)) for ckpt in CKPTS}
    union_fpr_B, _ = fpr_for_taus(
        calib["prob_P8A"].values, calib["prob_E2B"].values, calib["prob_E3"].values,
        tB["P8A"], tB["E2B"], tB["E3"],
    )

    for strat_name, taus, calib_fpr in [
        ("OR_EQUAL_MARGIN", tA, union_fpr_A),
        ("OR_CALIBRATED", tB, union_fpr_B),
    ]:
        for suite in ALL_SUITES:
            df = DATA[suite]
            r, c, n = recall_for_taus(df, taus["P8A"], taus["E2B"], taus["E3"])
            records.append({
                "strategy": strat_name,
                "fpr_target": fpr_target,
                "tau_P8A": taus["P8A"],
                "tau_E2B": taus["E2B"],
                "tau_E3": taus["E3"],
                "calib_actual_fpr_dev": calib_fpr,
                "eval_suite": suite,
                "is_real_suite": suite in REAL_SUITES,
                "recall_or_fpr": r,
                "caught": c,
                "n": n,
            })

or_df = pd.DataFrame(records)
or_df.to_csv(os.path.join(OUT, "or_ensemble_per_ckpt_thresholds.csv"), index=False)
print(f"  wrote or_ensemble_per_ckpt_thresholds.csv ({len(or_df)} rows)")

# Print key table at FPR=10%
sub = or_df[or_df["fpr_target"] == 0.10]
piv = sub.pivot_table(index="strategy", columns="eval_suite", values="recall_or_fpr")
print("\n==== OR-ensemble at FPR_target=10% (per-ckpt thresholds, union calibrated) ====")
print(piv.round(4).to_string())
