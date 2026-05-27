"""D1 — P8A IQ alignment regression on chronic-6 reals.

Tests whether P8A's chronic-6 invariance is explained by 6 IQ axes via:
  A1 — joint regression on chronic-6 reals
  A2 — per-axis univariate regressions on chronic-6 reals
  A3 — per-identity-within-chronic-6 sub-decomposition
  A4 — joint regression on healthy reals (baseline)

CPU only. n_jobs=1 enforced. Pure pandas/numpy/sklearn/statsmodels.
"""
from __future__ import annotations

import os
import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
INPUT_CSV = ROOT / "analysis/cpu_diagnostics_2026-05-12_stage_a/outputs/unified_frame_matrix.csv"
OUT_DIR = ROOT / "analysis/cpu_diagnostics_2026-05-12_d1_p8a_iq_alignment"
OUT_OUTPUTS = OUT_DIR / "outputs"
OUT_OUTPUTS.mkdir(parents=True, exist_ok=True)

IQ_AXES = [
    "min_dim",
    "lap_var",
    "luma_mean",
    "saturation_mean",
    "color_a_dev",
    "color_b_dev",
]

CHRONIC_6_PATTERNS = [
    "Roy_D",
    "PC_Generator",
    "bla_bla_chow",
    "Md_noyn_Sharker",
    "dor_shkedi",
    "healthy_dor",
]

EPS = 1e-6


def to_logit(p: np.ndarray) -> np.ndarray:
    """logit(p) with clipping at [eps, 1-eps]."""
    clipped = np.clip(p, EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def extract_identity(frame_path: str) -> str:
    """basename split by '__' [0]."""
    if not isinstance(frame_path, str):
        return ""
    basename = frame_path.split("/")[-1]
    parts = basename.split("__")
    return parts[0] if parts else ""


def joint_regression(df: pd.DataFrame, axes: list[str], y_col: str) -> dict:
    """Run OLS with standardized features. Return dict with n, R2, coefs, CIs, dominant axis."""
    sub = df[axes + [y_col]].dropna().copy()
    n = len(sub)
    if n < (len(axes) + 2):
        return {"n": n, "r2": np.nan, "coefs": {a: np.nan for a in axes}, "ci_low": {a: np.nan for a in axes},
                "ci_high": {a: np.nan for a in axes}, "dominant_axis": None, "dominant_coef": np.nan,
                "intercept": np.nan, "pvalues": {a: np.nan for a in axes}}

    X = sub[axes].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)

    # Standardize features (z-score) using sample means/stds
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0)
    stds_safe = np.where(stds == 0, 1.0, stds)
    Xz = (X - means) / stds_safe

    # sklearn fit for R^2
    lr = LinearRegression(n_jobs=1)
    lr.fit(Xz, y)
    r2 = float(lr.score(Xz, y))

    # Closed-form OLS with CIs + p-values via numpy/scipy (statsmodels-equivalent).
    # Design matrix with intercept.
    k = Xz.shape[1]  # number of axes
    Xd = np.hstack([np.ones((n, 1)), Xz])  # n x (k+1)
    # Solve normal equations
    XtX = Xd.T @ Xd
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (Xd.T @ y)  # length k+1
    y_pred = Xd @ beta
    resid = y - y_pred
    dof = n - (k + 1)
    if dof <= 0:
        sigma2 = np.nan
        se = np.full(k + 1, np.nan)
        tvals = np.full(k + 1, np.nan)
        pvals_all = np.full(k + 1, np.nan)
        tcrit = np.nan
    else:
        sigma2 = float((resid @ resid) / dof)
        var_beta = sigma2 * np.diag(XtX_inv)
        se = np.sqrt(np.clip(var_beta, 0.0, None))
        tvals = beta / np.where(se == 0, np.nan, se)
        # two-sided p-value
        pvals_all = 2.0 * (1.0 - scipy_stats.t.cdf(np.abs(tvals), df=dof))
        tcrit = float(scipy_stats.t.ppf(0.975, df=dof))

    coefs = {a: float(beta[i + 1]) for i, a in enumerate(axes)}
    if not np.isnan(tcrit):
        ci_low = {a: float(beta[i + 1] - tcrit * se[i + 1]) for i, a in enumerate(axes)}
        ci_high = {a: float(beta[i + 1] + tcrit * se[i + 1]) for i, a in enumerate(axes)}
    else:
        ci_low = {a: np.nan for a in axes}
        ci_high = {a: np.nan for a in axes}
    pvalues = {a: float(pvals_all[i + 1]) for i, a in enumerate(axes)}
    intercept = float(beta[0])

    dom_axis = max(coefs, key=lambda a: abs(coefs[a]))
    dom_coef = coefs[dom_axis]

    return {
        "n": n, "r2": r2, "coefs": coefs, "ci_low": ci_low, "ci_high": ci_high,
        "pvalues": pvalues, "intercept": intercept,
        "dominant_axis": dom_axis, "dominant_coef": dom_coef,
    }


def univariate_table(df: pd.DataFrame, axes: list[str], y_col: str) -> pd.DataFrame:
    """Per-axis Pearson r and r^2 with y_col."""
    rows = []
    for a in axes:
        sub = df[[a, y_col]].dropna()
        n = len(sub)
        if n < 3:
            rows.append({"axis": a, "n": n, "pearson_r": np.nan, "r2": np.nan})
            continue
        x = sub[a].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)
        if x.std() == 0:
            r = np.nan
        else:
            r = float(np.corrcoef(x, y)[0, 1])
        rows.append({"axis": a, "n": n, "pearson_r": r, "r2": (r * r) if not math.isnan(r) else np.nan})
    out = pd.DataFrame(rows).sort_values(by="r2", ascending=False, na_position="last").reset_index(drop=True)
    return out


def coef_rows(label: str, result: dict, axes: list[str]) -> list[dict]:
    """Convert a joint_regression result to per-axis rows for CSV output."""
    rows = []
    for a in axes:
        rows.append({
            "cohort": label,
            "n": result["n"],
            "r2": result["r2"],
            "axis": a,
            "std_coef": result["coefs"][a],
            "ci_low_95": result["ci_low"][a],
            "ci_high_95": result["ci_high"][a],
            "pvalue": result["pvalues"][a],
            "dominant_axis": result["dominant_axis"],
            "dominant_std_coef": result["dominant_coef"],
            "intercept": result["intercept"],
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not INPUT_CSV.exists():
        sys.exit(f"ERROR: input not found: {INPUT_CSV}")

    print(f"[d1] reading {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"[d1] total rows: {len(df)}")

    # Ensure required columns
    required = ["P8A", "label", "is_chronic_6", "frame_path"] + IQ_AXES
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: missing columns: {missing}")

    df["P8A_logit"] = to_logit(df["P8A"].to_numpy(dtype=float))
    df["identity"] = df["frame_path"].apply(extract_identity)

    # IQ non-null mask
    iq_mask = df[IQ_AXES].notna().all(axis=1)
    print(f"[d1] rows with all IQ axes non-null: {int(iq_mask.sum())} / {len(df)}")

    # -----------------------------------------------------------------------
    # A1 — joint regression on chronic-6 reals
    # -----------------------------------------------------------------------
    mask_a1 = (df["label"] == 0) & (df["is_chronic_6"] == 1) & iq_mask
    df_a1 = df.loc[mask_a1].copy()
    print(f"[d1] A1 cohort (chronic-6 reals, non-null IQ): n={len(df_a1)}")
    a1 = joint_regression(df_a1, IQ_AXES, "P8A_logit")
    print(f"[d1] A1: n={a1['n']} R2={a1['r2']:.4f} dom={a1['dominant_axis']} (beta={a1['dominant_coef']:+.4f})")

    a1_rows = coef_rows("chronic6_reals", a1, IQ_AXES)
    pd.DataFrame(a1_rows).to_csv(OUT_OUTPUTS / "joint_regression_chronic6.csv", index=False)

    # -----------------------------------------------------------------------
    # A2 — univariate per axis on chronic-6 reals
    # -----------------------------------------------------------------------
    a2_table = univariate_table(df_a1, IQ_AXES, "P8A_logit")
    a2_table.to_csv(OUT_OUTPUTS / "univariate_per_axis.csv", index=False)
    print("[d1] A2 univariate (top):")
    print(a2_table.to_string(index=False))

    # -----------------------------------------------------------------------
    # A3 — per-identity within chronic-6
    # -----------------------------------------------------------------------
    a3_rows = []
    a3_coef_rows = []  # per-axis-per-identity if we want; we emit summary only
    print("[d1] A3 per-identity:")
    for pat in CHRONIC_6_PATTERNS:
        # Identity match: case-sensitive contains on extracted identity (basename split __ [0]).
        sub_mask = df_a1["identity"].str.contains(pat, case=True, na=False, regex=False)
        sub = df_a1.loc[sub_mask].copy()
        n_total = len(sub)
        if n_total < 30:
            a3_rows.append({
                "identity_pattern": pat, "n": n_total, "r2": np.nan,
                "dominant_axis": None, "dominant_std_coef": np.nan,
                "note": "n<30, skipped",
            })
            print(f"  {pat}: n={n_total} (skipped, n<30)")
            continue
        res = joint_regression(sub, IQ_AXES, "P8A_logit")
        a3_rows.append({
            "identity_pattern": pat,
            "n": res["n"], "r2": res["r2"],
            "dominant_axis": res["dominant_axis"],
            "dominant_std_coef": res["dominant_coef"],
            "note": "",
        })
        # Also write per-axis coefs to a long-form file
        for a in IQ_AXES:
            a3_coef_rows.append({
                "identity_pattern": pat,
                "n": res["n"], "r2": res["r2"],
                "axis": a,
                "std_coef": res["coefs"][a],
                "ci_low_95": res["ci_low"][a],
                "ci_high_95": res["ci_high"][a],
                "pvalue": res["pvalues"][a],
            })
        print(f"  {pat}: n={res['n']} R2={res['r2']:.4f} dom={res['dominant_axis']} (beta={res['dominant_coef']:+.4f})")

    a3_summary = pd.DataFrame(a3_rows)
    a3_summary.to_csv(OUT_OUTPUTS / "per_identity_regression.csv", index=False)
    a3_coefs = pd.DataFrame(a3_coef_rows)
    a3_coefs.to_csv(OUT_OUTPUTS / "per_identity_regression_long.csv", index=False)

    # -----------------------------------------------------------------------
    # A4 — joint regression on healthy reals (is_chronic_6 == 0)
    # -----------------------------------------------------------------------
    mask_a4 = (df["label"] == 0) & (df["is_chronic_6"] == 0) & iq_mask
    df_a4 = df.loc[mask_a4].copy()
    print(f"[d1] A4 cohort (healthy reals, non-null IQ): n={len(df_a4)}")
    a4 = joint_regression(df_a4, IQ_AXES, "P8A_logit")
    print(f"[d1] A4: n={a4['n']} R2={a4['r2']:.4f} dom={a4['dominant_axis']} (beta={a4['dominant_coef']:+.4f})")

    a4_rows = coef_rows("healthy_reals", a4, IQ_AXES)
    pd.DataFrame(a4_rows).to_csv(OUT_OUTPUTS / "joint_regression_healthy.csv", index=False)

    # -----------------------------------------------------------------------
    # Save summary JSON-ish for the FACTS doc
    # -----------------------------------------------------------------------
    import json
    summary = {
        "A1_chronic6": {
            "n": a1["n"], "r2": a1["r2"],
            "dominant_axis": a1["dominant_axis"],
            "dominant_std_coef": a1["dominant_coef"],
            "coefs": a1["coefs"],
            "ci_low": a1["ci_low"], "ci_high": a1["ci_high"],
            "pvalues": a1["pvalues"], "intercept": a1["intercept"],
        },
        "A4_healthy": {
            "n": a4["n"], "r2": a4["r2"],
            "dominant_axis": a4["dominant_axis"],
            "dominant_std_coef": a4["dominant_coef"],
            "coefs": a4["coefs"],
            "ci_low": a4["ci_low"], "ci_high": a4["ci_high"],
            "pvalues": a4["pvalues"], "intercept": a4["intercept"],
        },
        "A3_per_identity": a3_rows,
        "A2_univariate_chronic6": a2_table.to_dict(orient="records"),
    }
    (OUT_OUTPUTS / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("[d1] done. outputs at:", OUT_OUTPUTS)


if __name__ == "__main__":
    main()
