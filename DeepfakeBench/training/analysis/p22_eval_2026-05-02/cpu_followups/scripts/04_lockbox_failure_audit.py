"""Job D — lockbox failure-mode audit for P22 step8k.

Are P22 step8k's lockbox real false-positives concentrated on the same attrs
as P8A's (image-quality shortcut hitting webcam reals) or different?

For each (P8A, P22 step1k, P22 step8k) on lockbox real:
  - Take false-positives at FPR=2% τ
  - Compare attribute distributions of FPs to TNs (true-negatives)
  - Identify whether FPs cluster at low-laplacian / high-luma / low-skin
"""
from __future__ import annotations

import sys; sys.path.insert(0, "analysis/p22_eval_2026-05-02/cpu_followups/scripts")

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

from _common import OUT, ATTRS_CSV, load_per_frame_scores

CKPTS = ["P8A_REFERENCE_STEP5000", "P18T_GRL_TREATMENT_STEP4000",
         "P22_AUG_STEP1000", "P22_AUG_STEP4000", "P22_AUG_STEP8000"]
ATTR_COLS = ["laplacian_var", "luma_mean", "skin_frac"]


def fp_audit(ckpt, attrs, suite="teams_real_all_lockbox", floor=0.02):
    df = load_per_frame_scores(ckpt, suite)
    if df is None: return None
    real_dev = load_per_frame_scores(ckpt, "teams_real_all_dev")
    if real_dev is None: return None
    # Calibrated τ at FPR=floor on dev
    rd = real_dev["frame_prob"].to_numpy()
    rd_sorted = np.sort(rd)
    if len(rd_sorted) < 10:
        return None
    idx = int(np.ceil(len(rd_sorted) * (1 - floor)))
    tau = float(rd_sorted[min(idx, len(rd_sorted) - 1)])

    df = df.merge(attrs[["frame_path"] + ATTR_COLS], on="frame_path", how="left")

    # FP = real frame scored ≥ τ
    fp = df[df["frame_prob"] >= tau].copy()
    tn = df[df["frame_prob"] < tau].copy()

    # Attribute stats on FPs vs TNs (only frames with attrs available)
    fp_attrs = fp.dropna(subset=ATTR_COLS)
    tn_attrs = tn.dropna(subset=ATTR_COLS)

    out = {"ckpt": ckpt, "tau_at_dev_fpr_0.02": tau,
           "n_total": len(df), "n_FP": len(fp), "n_TN": len(tn),
           "lockbox_FPR_at_dev_calibrated_tau": len(fp) / len(df)}

    for col in ATTR_COLS:
        fp_vals = fp_attrs[col].dropna().to_numpy()
        tn_vals = tn_attrs[col].dropna().to_numpy()
        if len(fp_vals) >= 5 and len(tn_vals) >= 5:
            U, p = mannwhitneyu(fp_vals, tn_vals, alternative="two-sided")
            out[f"{col}_FP_median"] = float(np.median(fp_vals))
            out[f"{col}_TN_median"] = float(np.median(tn_vals))
            out[f"{col}_p_FP_lower_TN"] = float(p)
            out[f"{col}_n_FP_with_attrs"] = int(len(fp_vals))
            out[f"{col}_n_TN_with_attrs"] = int(len(tn_vals))
        else:
            out[f"{col}_FP_median"] = float("nan")
            out[f"{col}_TN_median"] = float("nan")
            out[f"{col}_p_FP_lower_TN"] = float("nan")
            out[f"{col}_n_FP_with_attrs"] = int(len(fp_vals))
            out[f"{col}_n_TN_with_attrs"] = int(len(tn_vals))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    attrs = pd.read_csv(ATTRS_CSV)
    rows = [fp_audit(c, attrs) for c in CKPTS]
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "04_lockbox_fp_audit.csv", index=False)

    print("=" * 80)
    print("Lockbox real-suite false-positive audit at dev-calibrated τ (FPR=2% on dev)")
    print("=" * 80)
    print(df[["ckpt", "tau_at_dev_fpr_0.02", "n_FP", "n_TN",
              "lockbox_FPR_at_dev_calibrated_tau"]].to_string(index=False,
              float_format=lambda x: f"{x:.4f}"))

    print("\nAttribute medians: FPs vs TNs (low n_with_attrs in lockbox CSV — only n=50)")
    for col in ATTR_COLS:
        print(f"\n  {col}:")
        cols = ["ckpt", f"{col}_FP_median", f"{col}_TN_median",
                f"{col}_n_FP_with_attrs", f"{col}_n_TN_with_attrs",
                f"{col}_p_FP_lower_TN"]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
