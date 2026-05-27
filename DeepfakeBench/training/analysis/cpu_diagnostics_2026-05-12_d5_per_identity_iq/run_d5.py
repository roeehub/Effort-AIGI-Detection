"""
D5 — Per-identity mean(score) explained by per-identity mean(IQ axes)?

Question: Are per-identity FPRs mostly explained by per-identity mean-IQ profile?
If yes, "identity" is a bag of IQ values from the model's perspective and
identity-level interventions are downstream of axis-level interventions.

Inputs (local only — no GCS, no internet):
    analysis/cpu_diagnostics_2026-05-12_stage_a/outputs/unified_frame_matrix.csv

Outputs (this folder):
    outputs/per_identity_aggregates.csv
    outputs/per_ckpt_regression.csv
    outputs/per_ckpt_residuals_top10.csv

Method:
    B1 — per-identity aggregation on real frames with non-null IQ atlas join.
    B2 — per-ckpt linear regression of identity mean-logit(score) ~ z(6 IQ axes),
         with closed-form OLS for R^2 and standardized betas + 95% CI.
    B3 — top-10 absolute residuals per ckpt (which identities don't fit).

Constraints:
    - n_jobs=1 (per feedback_sklearn_njobs.md).
    - statsmodels is broken in the local env (scipy._lazywhere import error),
      so OLS is implemented in closed form using numpy + scipy.stats.t for CIs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

REPO_ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
STAGE_A_MATRIX = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-12_stage_a/outputs/unified_frame_matrix.csv"
OUT_DIR = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-12_d5_per_identity_iq/outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPTS = ["P8A", "T5C_step3500", "T3_S1_step1500"]
IQ_AXES = ["min_dim", "lap_var", "luma_mean", "saturation_mean", "color_a_dev", "color_b_dev"]
MIN_N_FRAMES = 30
LOGIT_EPS = 1e-3  # clip so log(m/(1-m)) is finite

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def identity_from_path(p: str) -> str:
    """Identity := basename(frame_path).split('__')[0]."""
    return p.split("/")[-1].split("__")[0]


def safe_logit(m: np.ndarray, eps: float = LOGIT_EPS) -> np.ndarray:
    """Clip then logit."""
    m = np.clip(m, eps, 1.0 - eps)
    return np.log(m / (1.0 - m))


def ols_fit(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Closed-form OLS with intercept.

    Returns dict with:
        beta: (k,) — coefficients on columns of X (excluding intercept).
        intercept: float
        ci_low: (k,), ci_high: (k,) — 95% CIs for the k slope coeffs.
        r2: float — coefficient of determination.
        y_pred: (n,) — fitted values.
        residuals: (n,) — y - y_pred.
        n, k, dof.
    """
    n, k = X.shape
    X_aug = np.column_stack([np.ones(n), X])  # intercept + slopes
    XtX = X_aug.T @ X_aug
    XtX_inv = np.linalg.pinv(XtX)
    beta_all = XtX_inv @ X_aug.T @ y  # shape (k+1,)
    y_pred = X_aug @ beta_all
    residuals = y - y_pred

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    dof = max(n - (k + 1), 1)
    sigma2 = ss_res / dof
    cov_beta = sigma2 * XtX_inv  # shape (k+1, k+1)
    se = np.sqrt(np.diag(cov_beta))  # (k+1,)
    t_crit = stats.t.ppf(0.975, df=dof)

    intercept = float(beta_all[0])
    slopes = beta_all[1:]
    ci_low = slopes - t_crit * se[1:]
    ci_high = slopes + t_crit * se[1:]

    return {
        "beta": slopes,
        "intercept": intercept,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "r2": r2,
        "y_pred": y_pred,
        "residuals": residuals,
        "n": n,
        "k": k,
        "dof": dof,
    }


# ----------------------------------------------------------------------
# Load and filter
# ----------------------------------------------------------------------


def load_filtered() -> pd.DataFrame:
    """Load unified matrix, filter to real frames with all 6 IQ axes non-null."""
    df = pd.read_csv(STAGE_A_MATRIX)
    n0 = len(df)
    df = df[df["label"] == 0].copy()
    n_real = len(df)
    df = df.dropna(subset=IQ_AXES).copy()
    n_real_with_iq = len(df)
    df["identity"] = df["frame_path"].apply(identity_from_path)
    print(f"[load] total rows: {n0}")
    print(f"[load] real rows: {n_real}")
    print(f"[load] real rows with all 6 IQ axes non-null: {n_real_with_iq}")
    print(f"[load] unique identities (real, non-null IQ): {df['identity'].nunique()}")
    return df


# ----------------------------------------------------------------------
# B1 — per-identity aggregation
# ----------------------------------------------------------------------


def b1_aggregate(real: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per identity. Returns one row per identity."""
    agg_specs = {
        "n_frames": ("identity", "size"),
        "is_chronic_6": ("is_chronic_6", "max"),
        "is_dor": ("is_dor", "max"),
        "is_roy_d": ("is_roy_d", "max"),
    }
    for ckpt in CKPTS:
        agg_specs[f"mean_{ckpt}"] = (ckpt, "mean")
        agg_specs[f"p90_{ckpt}"] = (ckpt, lambda s: s.quantile(0.90))
    for axis in IQ_AXES:
        agg_specs[f"mean_{axis}"] = (axis, "mean")

    g = real.groupby("identity").agg(**agg_specs).reset_index()
    g = g.sort_values("n_frames", ascending=False).reset_index(drop=True)
    return g


# ----------------------------------------------------------------------
# B2 — per-ckpt regression
# ----------------------------------------------------------------------


def b2_regress(agg: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict]]:
    """
    For each ckpt, fit mean-logit(score) ~ z(6 IQ axes).
    Returns:
        long-form per-ckpt-axis dataframe with beta + 95% CI + R^2_full.
        dict of per-ckpt fit info (used by B3 for residuals).
    """
    iq_cols = [f"mean_{a}" for a in IQ_AXES]
    X_raw = agg[iq_cols].to_numpy(dtype=float)  # (n_id, 6)
    # z-score columns (population SD; ddof=0).
    mu = X_raw.mean(axis=0)
    sd = X_raw.std(axis=0, ddof=0)
    sd_safe = np.where(sd == 0, 1.0, sd)
    Z = (X_raw - mu) / sd_safe

    rows: list[dict] = []
    fits: dict[str, dict] = {}
    for ckpt in CKPTS:
        m = agg[f"mean_{ckpt}"].to_numpy(dtype=float)
        y = safe_logit(m)
        fit = ols_fit(Z, y)
        fits[ckpt] = {
            "fit": fit,
            "mean_score": m,
            "logit_obs": y,
            "X_z": Z,
            "iq_mu": mu,
            "iq_sd": sd,
        }
        for j, axis in enumerate(IQ_AXES):
            rows.append({
                "ckpt": ckpt,
                "axis": axis,
                "beta_standardized": float(fit["beta"][j]),
                "ci_low": float(fit["ci_low"][j]),
                "ci_high": float(fit["ci_high"][j]),
                "R2_full_model": float(fit["r2"]),
                "n_identities": int(fit["n"]),
                "dof": int(fit["dof"]),
                "intercept": float(fit["intercept"]),
            })
    long_df = pd.DataFrame(rows)
    return long_df, fits


# ----------------------------------------------------------------------
# B3 — top residuals
# ----------------------------------------------------------------------


def b3_residuals_top10(agg: pd.DataFrame, fits: dict[str, dict]) -> pd.DataFrame:
    """For each ckpt, top 10 identities by |observed - predicted| (in logit space)."""
    out_rows: list[dict] = []
    for ckpt in CKPTS:
        info = fits[ckpt]
        fit = info["fit"]
        observed_logit = info["logit_obs"]
        predicted_logit = fit["y_pred"]
        residuals = fit["residuals"]
        m_obs = info["mean_score"]
        # Convert predicted logit back to probability for reporting.
        m_pred = 1.0 / (1.0 + np.exp(-predicted_logit))
        order = np.argsort(-np.abs(residuals))[:10]
        for rank, idx in enumerate(order, start=1):
            out_rows.append({
                "ckpt": ckpt,
                "rank_by_abs_residual": rank,
                "identity": agg.iloc[idx]["identity"],
                "n_frames": int(agg.iloc[idx]["n_frames"]),
                "is_chronic_6": int(agg.iloc[idx]["is_chronic_6"]),
                "observed_mean_score": float(m_obs[idx]),
                "predicted_mean_score": float(m_pred[idx]),
                "observed_logit": float(observed_logit[idx]),
                "predicted_logit": float(predicted_logit[idx]),
                "residual_logit": float(residuals[idx]),
            })
    return pd.DataFrame(out_rows)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    print(f"[paths] STAGE_A_MATRIX = {STAGE_A_MATRIX}")
    print(f"[paths] OUT_DIR        = {OUT_DIR}")

    real = load_filtered()

    # B1
    full_agg = b1_aggregate(real)
    full_agg_path = OUT_DIR / "per_identity_aggregates.csv"
    full_agg.to_csv(full_agg_path, index=False)
    print(f"[B1] wrote {full_agg_path} — {len(full_agg)} identities total")

    agg_for_reg = full_agg[full_agg["n_frames"] >= MIN_N_FRAMES].copy().reset_index(drop=True)
    n_total = len(agg_for_reg)
    n_chronic = int(agg_for_reg["is_chronic_6"].sum())
    n_nonchronic = n_total - n_chronic
    print(f"[B1] identities with n_frames >= {MIN_N_FRAMES}: {n_total} "
          f"(chronic_6 = {n_chronic}, non-chronic = {n_nonchronic})")

    # B2
    reg_df, fits = b2_regress(agg_for_reg)
    reg_path = OUT_DIR / "per_ckpt_regression.csv"
    reg_df.to_csv(reg_path, index=False)
    print(f"[B2] wrote {reg_path}")
    for ckpt in CKPTS:
        r2 = fits[ckpt]["fit"]["r2"]
        # Find top |beta| axis.
        betas = fits[ckpt]["fit"]["beta"]
        top_idx = int(np.argmax(np.abs(betas)))
        print(f"[B2] {ckpt}: R^2 = {r2:.4f}; top axis = {IQ_AXES[top_idx]} "
              f"(β_std = {betas[top_idx]:+.4f})")

    # B3
    resid_df = b3_residuals_top10(agg_for_reg, fits)
    resid_path = OUT_DIR / "per_ckpt_residuals_top10.csv"
    resid_df.to_csv(resid_path, index=False)
    print(f"[B3] wrote {resid_path}")
    for ckpt in CKPTS:
        sub = resid_df[resid_df["ckpt"] == ckpt].head(1).iloc[0]
        print(f"[B3] {ckpt} top outlier: {sub['identity']} "
              f"(obs_mean={sub['observed_mean_score']:.4f}, pred_mean={sub['predicted_mean_score']:.4f}, "
              f"resid_logit={sub['residual_logit']:+.4f}, n={sub['n_frames']}, chronic={sub['is_chronic_6']})")

    # Echo summary so the agent's report can quote numbers directly.
    summary = {
        "n_identities_total": int(len(full_agg)),
        "n_identities_n_ge_30": int(n_total),
        "n_identities_n_ge_30_chronic_6": int(n_chronic),
        "n_identities_n_ge_30_non_chronic": int(n_nonchronic),
        "per_ckpt_R2": {ckpt: float(fits[ckpt]["fit"]["r2"]) for ckpt in CKPTS},
        "per_ckpt_top_axis": {
            ckpt: {
                "axis": IQ_AXES[int(np.argmax(np.abs(fits[ckpt]["fit"]["beta"])))],
                "beta_standardized": float(
                    fits[ckpt]["fit"]["beta"][int(np.argmax(np.abs(fits[ckpt]["fit"]["beta"])))]
                ),
            }
            for ckpt in CKPTS
        },
        "per_ckpt_top_residual": {
            ckpt: {
                "identity": resid_df[resid_df["ckpt"] == ckpt].iloc[0]["identity"],
                "observed_mean_score": float(resid_df[resid_df["ckpt"] == ckpt].iloc[0]["observed_mean_score"]),
                "predicted_mean_score": float(resid_df[resid_df["ckpt"] == ckpt].iloc[0]["predicted_mean_score"]),
                "residual_logit": float(resid_df[resid_df["ckpt"] == ckpt].iloc[0]["residual_logit"]),
                "n_frames": int(resid_df[resid_df["ckpt"] == ckpt].iloc[0]["n_frames"]),
                "is_chronic_6": int(resid_df[resid_df["ckpt"] == ckpt].iloc[0]["is_chronic_6"]),
            }
            for ckpt in CKPTS
        },
    }
    summary_path = OUT_DIR / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[summary] wrote {summary_path}")


if __name__ == "__main__":
    main()
