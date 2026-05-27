"""Statistical analysis: Cohen's d, t-tests, score correlations, linear classifier.

Reads per_frame_features.csv. Writes:
  axis_comparison.csv, score_axis_correlations.csv, falseflag_classifier.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis/xinhe_cross_camera_audit_2026-05-06"
)
OUT = ROOT / "outputs"

# Axes considered for cross-population comparison.
# Image-level features are universally available; face-* features may be NaN
# but extraction reported 0 skips, so all should be present.
AXES = [
    "lap_var_full",
    "lap_var_face",
    "sobel_mean_full",
    "sobel_mean_face",
    "hf_ratio_full",
    "hf_ratio_face",
    "luma_mean",
    "luma_std",
    "luma_mean_face",
    "sat_mean",
    "sat_std",
    "hue_mean",
    "r_mean",
    "g_mean",
    "b_mean",
    "r_std",
    "g_std",
    "b_std",
    "width",
    "height",
    "face_w",
    "face_h",
    "face_area",
    "face_area_frac",
]


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference (a - b) using pooled SD."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    sa, sb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((a.size - 1) * sa + (b.size - 1) * sb) / (a.size + b.size - 2))
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def main() -> int:
    df = pd.read_csv(OUT / "per_frame_features.csv")
    may6 = df[df["population"] == "may6_falseflag"].copy()
    may5 = df[df["population"] == "may5_correct"].copy()
    print(f"may6: {len(may6)}  may5: {len(may5)}")

    # 1. Per-axis comparison
    rows: list[dict] = []
    for ax in AXES:
        if ax not in df.columns:
            continue
        a = may6[ax].astype(float).values
        b = may5[ax].astype(float).values
        a_clean = a[np.isfinite(a)]
        b_clean = b[np.isfinite(b)]
        rec: dict = {"axis": ax}
        rec["n_may6"] = int(a_clean.size)
        rec["n_may5"] = int(b_clean.size)
        rec["mean_may6"] = float(np.mean(a_clean)) if a_clean.size else np.nan
        rec["mean_may5"] = float(np.mean(b_clean)) if b_clean.size else np.nan
        rec["std_may6"] = float(np.std(a_clean, ddof=1)) if a_clean.size > 1 else np.nan
        rec["std_may5"] = float(np.std(b_clean, ddof=1)) if b_clean.size > 1 else np.nan
        rec["cohen_d"] = cohens_d(a_clean, b_clean)
        if a_clean.size > 1 and b_clean.size > 1:
            t = stats.ttest_ind(a_clean, b_clean, equal_var=False)
            rec["t_stat"] = float(t.statistic)
            rec["t_pvalue"] = float(t.pvalue)
            u = stats.mannwhitneyu(a_clean, b_clean, alternative="two-sided")
            rec["mwu_stat"] = float(u.statistic)
            rec["mwu_pvalue"] = float(u.pvalue)
        else:
            rec["t_stat"] = rec["t_pvalue"] = rec["mwu_stat"] = rec["mwu_pvalue"] = np.nan
        rows.append(rec)

    cmp_df = pd.DataFrame(rows)
    cmp_df["abs_cohen_d"] = cmp_df["cohen_d"].abs()
    cmp_df = cmp_df.sort_values("abs_cohen_d", ascending=False).reset_index(drop=True)
    cmp_df.to_csv(OUT / "axis_comparison.csv", index=False)
    print("\nTop 10 axes by |Cohen d|:")
    print(cmp_df[["axis", "cohen_d", "mean_may6", "mean_may5", "t_pvalue"]].head(10).to_string(index=False))

    # 2. Within-may6 score correlations
    score = may6["deploy_score"].astype(float).values
    score_rows: list[dict] = []
    for ax in AXES:
        if ax not in may6.columns:
            continue
        x = may6[ax].astype(float).values
        mask = np.isfinite(x) & np.isfinite(score)
        if mask.sum() < 5:
            continue
        r, p = stats.pearsonr(x[mask], score[mask])
        rs, ps = stats.spearmanr(x[mask], score[mask])
        score_rows.append(
            {
                "axis": ax,
                "n": int(mask.sum()),
                "pearson_r": float(r),
                "pearson_p": float(p),
                "spearman_r": float(rs),
                "spearman_p": float(ps),
            }
        )
    score_df = pd.DataFrame(score_rows)
    score_df["abs_pearson_r"] = score_df["pearson_r"].abs()
    score_df = score_df.sort_values("abs_pearson_r", ascending=False).reset_index(drop=True)
    score_df.to_csv(OUT / "score_axis_correlations.csv", index=False)
    print("\nTop 10 within-may6 score correlations:")
    print(score_df[["axis", "pearson_r", "pearson_p", "spearman_r"]].head(10).to_string(index=False))

    # 3. Linear classifier — logistic regression with 5-fold CV
    feat_axes = [a for a in AXES if a in df.columns]
    X = df[feat_axes].astype(float).values
    y = (df["population"] == "may6_falseflag").astype(int).values
    # NaN-safe — replace with column mean
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(
            max_iter=5000,
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            n_jobs=1,
        )
        clf.fit(Xtr, y[tr])
        oof[te] = clf.predict_proba(Xte)[:, 1]
    cv_auc = float(roc_auc_score(y, oof))
    print(f"\n5-fold CV AUC (may6 vs may5): {cv_auc:.4f}")

    # final fit for coefficient extraction
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf_full = LogisticRegression(max_iter=5000, C=1.0, penalty="l2", solver="lbfgs", n_jobs=1)
    clf_full.fit(Xs, y)
    coefs = list(zip(feat_axes, clf_full.coef_[0].tolist()))
    coefs_sorted = sorted(coefs, key=lambda kv: abs(kv[1]), reverse=True)
    print("\nTop 10 standardized coefficients (positive = pushes toward may6):")
    for nm, c in coefs_sorted[:10]:
        print(f"  {nm:20s} {c:+.4f}")

    out = {
        "cv_auc": cv_auc,
        "n_may6": int((y == 1).sum()),
        "n_may5": int((y == 0).sum()),
        "features": feat_axes,
        "intercept": float(clf_full.intercept_[0]),
        "coefficients_standardized": [
            {"axis": nm, "coef": float(c), "abs_coef": float(abs(c))}
            for nm, c in coefs_sorted
        ],
        "scaler_mean": dict(zip(feat_axes, scaler.mean_.tolist())),
        "scaler_scale": dict(zip(feat_axes, scaler.scale_.tolist())),
    }
    with open(OUT / "falseflag_classifier.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT / 'falseflag_classifier.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
