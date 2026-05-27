"""
Subtype-stratified re-analysis of visomaster_enhanced_macro_dev (550 frames):
raw (275) vs teams (275). Three ckpts: P8A, E2B_3200, E3_6600.
Pure CPU; pandas / numpy / scipy / sklearn (n_jobs=1).

Outputs land in this script's directory.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
RAW = ROOT / "analysis" / "cpu_followups_2026-05-04" / "raw_reports"
OUT = ROOT / "analysis" / "viso_subtype_analysis_2026-05-04"
OUT.mkdir(parents=True, exist_ok=True)

VISO_JOINED = ROOT / "analysis" / "cpu_followups_2026-05-04" / "outputs" / "viso_per_frame_with_all_ckpts.csv"
CROP_ATTR = ROOT / "analysis" / "score_distribution_2026-05-02" / "outputs" / "crop_attributes.csv"

CKPTS = {
    "P8A": "p8a_reference_step5000",
    "E2B_3200": "e2b_top_n_step3200",
    "E3_6600": "e3_top_n_step6600",
}

CKPT_SCORE_COL = {
    "P8A": "P8A_score",
    "E2B_3200": "E2B_3200_score",
    "E3_6600": "E3_6600_score",
}

FPR_TARGETS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]

IQ_FEATURES = [
    "luma_p10",
    "laplacian_var",
    "sobel_edge_mean",
    "luma_p90",
    "luma_mean",
    "skin_frac",
    "saturation_mean",
    "luma_std",
    "h",
    "w",
]


def load_viso_with_subtype() -> pd.DataFrame:
    df = pd.read_csv(VISO_JOINED)
    df["filename"] = df["frame_path"].str.split("/").str[-1]
    df["subtype"] = df["filename"].apply(
        lambda f: "teams"
        if "visomaster_enhanced_teams" in f
        else ("raw" if "visomaster_enhanced_raw" in f else None)
    )
    assert df["subtype"].notna().all(), f"untagged frames: {df[df.subtype.isna()].filename.tolist()[:5]}"
    return df


def load_real_scores() -> dict[str, np.ndarray]:
    """Load teams_real_all_dev scores per ckpt for FPR calibration."""
    out = {}
    for ckpt_name, ckpt_tag in CKPTS.items():
        path = RAW / f"teams_real_all_dev_{ckpt_tag}_frames_report.csv"
        df = pd.read_csv(path)
        # Only label==0 (real) — but they should all be reals in this suite
        if "label" in df.columns:
            df = df[df["label"] == 0]
        scores = df["frame_prob"].values.astype(float)
        out[ckpt_name] = scores
    return out


def thresholds_at_fpr(real_scores: np.ndarray, fpr_targets: list[float]) -> dict[float, float]:
    """For each FPR target, return τ such that mean(real_scores >= τ) ≈ fpr_target.
    Use the (1 - fpr) quantile of reals.
    """
    out = {}
    for fpr in fpr_targets:
        tau = float(np.quantile(real_scores, 1.0 - fpr))
        out[fpr] = tau
    return out


def step2_recall_sweep(viso: pd.DataFrame, real_scores: dict[str, np.ndarray]):
    rows = []
    for ckpt_name in CKPTS:
        score_col = CKPT_SCORE_COL[ckpt_name]
        taus = thresholds_at_fpr(real_scores[ckpt_name], FPR_TARGETS)
        for subtype, sub_df in viso.groupby("subtype"):
            scores = sub_df[score_col].values.astype(float)
            n = len(scores)
            for fpr, tau in taus.items():
                caught = int((scores >= tau).sum())
                recall = caught / n if n > 0 else 0.0
                rows.append(
                    {
                        "subtype": subtype,
                        "ckpt": ckpt_name,
                        "fpr_target": fpr,
                        "tau": tau,
                        "n": n,
                        "n_caught": caught,
                        "recall": recall,
                    }
                )
    pd.DataFrame(rows).to_csv(OUT / "subtype_recall_by_fpr.csv", index=False)
    return pd.DataFrame(rows)


def step3_score_stats(viso: pd.DataFrame):
    rows = []
    for ckpt_name in CKPTS:
        score_col = CKPT_SCORE_COL[ckpt_name]
        for subtype, sub_df in viso.groupby("subtype"):
            s = sub_df[score_col].values.astype(float)
            rows.append(
                {
                    "subtype": subtype,
                    "ckpt": ckpt_name,
                    "n": len(s),
                    "mean": float(np.mean(s)),
                    "std": float(np.std(s, ddof=1)),
                    "p10": float(np.quantile(s, 0.10)),
                    "p25": float(np.quantile(s, 0.25)),
                    "p50": float(np.quantile(s, 0.50)),
                    "p75": float(np.quantile(s, 0.75)),
                    "p90": float(np.quantile(s, 0.90)),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "subtype_score_stats.csv", index=False)
    return pd.DataFrame(rows)


def step4_correlations(viso: pd.DataFrame):
    rows = []
    for subtype, sub_df in viso.groupby("subtype"):
        for a, b in combinations(CKPTS.keys(), 2):
            sa = sub_df[CKPT_SCORE_COL[a]].values.astype(float)
            sb = sub_df[CKPT_SCORE_COL[b]].values.astype(float)
            pear, _ = stats.pearsonr(sa, sb)
            spe, _ = stats.spearmanr(sa, sb)
            rows.append(
                {
                    "subtype": subtype,
                    "ckpt_a": a,
                    "ckpt_b": b,
                    "pearson": float(pear),
                    "spearman": float(spe),
                    "n": len(sa),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "subtype_correlations.csv", index=False)
    return pd.DataFrame(rows)


def caught_subset_label(p8a_c: int, e2b_c: int, e3_c: int) -> str:
    parts = []
    if p8a_c:
        parts.append("P8A")
    if e2b_c:
        parts.append("E2B_3200")
    if e3_c:
        parts.append("E3_6600")
    if not parts:
        return "NONE"
    return "+".join(parts)


def step5_frame_coverage(viso: pd.DataFrame, real_scores: dict[str, np.ndarray]):
    """Per-subtype Venn at FPR=10% (and also report at FPR=5% for completeness)."""
    taus_per_fpr = {
        ckpt: thresholds_at_fpr(real_scores[ckpt], FPR_TARGETS) for ckpt in CKPTS
    }
    rows = []
    for fpr in [0.05, 0.10]:
        for subtype, sub_df in viso.groupby("subtype"):
            tau_p8a = taus_per_fpr["P8A"][fpr]
            tau_e2b = taus_per_fpr["E2B_3200"][fpr]
            tau_e3 = taus_per_fpr["E3_6600"][fpr]
            c_p8a = (sub_df["P8A_score"] >= tau_p8a).astype(int)
            c_e2b = (sub_df["E2B_3200_score"] >= tau_e2b).astype(int)
            c_e3 = (sub_df["E3_6600_score"] >= tau_e3).astype(int)
            labels = [
                caught_subset_label(a, b, c)
                for a, b, c in zip(c_p8a, c_e2b, c_e3)
            ]
            counts = pd.Series(labels).value_counts().to_dict()
            n_total = len(sub_df)
            for label_str, n in counts.items():
                rows.append(
                    {
                        "subtype": subtype,
                        "fpr_target": fpr,
                        "caught_by": label_str,
                        "n_frames": int(n),
                        "fraction": n / n_total,
                        "n_total": n_total,
                    }
                )
    pd.DataFrame(rows).to_csv(OUT / "subtype_frame_coverage.csv", index=False)
    return pd.DataFrame(rows)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    if pooled <= 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled))


def step6a_iq_distribution(viso_with_iq: pd.DataFrame):
    rows = []
    raw = viso_with_iq[viso_with_iq.subtype == "raw"]
    teams = viso_with_iq[viso_with_iq.subtype == "teams"]
    for feat in IQ_FEATURES:
        a = raw[feat].values.astype(float)
        b = teams[feat].values.astype(float)
        t, p = stats.ttest_ind(a, b, equal_var=False)
        d = cohens_d(a, b)  # raw - teams
        rows.append(
            {
                "feature": feat,
                "raw_mean": float(np.mean(a)),
                "raw_std": float(np.std(a, ddof=1)),
                "teams_mean": float(np.mean(b)),
                "teams_std": float(np.std(b, ddof=1)),
                "cohens_d_raw_minus_teams": d,
                "abs_cohens_d": abs(d) if not np.isnan(d) else float("nan"),
                "t_stat": float(t),
                "p_value": float(p),
                "n_raw": len(a),
                "n_teams": len(b),
            }
        )
    df = pd.DataFrame(rows).sort_values("abs_cohens_d", ascending=False)
    df.to_csv(OUT / "subtype_iq_distribution.csv", index=False)
    return df


def step6b_caught_vs_uncaught(viso_with_iq: pd.DataFrame, real_scores: dict[str, np.ndarray]):
    """Caught = (P8A>=τ_p8a OR E2B>=τ_e2b OR E3>=τ_e3) at FPR=10%."""
    tau_p8a = thresholds_at_fpr(real_scores["P8A"], [0.10])[0.10]
    tau_e2b = thresholds_at_fpr(real_scores["E2B_3200"], [0.10])[0.10]
    tau_e3 = thresholds_at_fpr(real_scores["E3_6600"], [0.10])[0.10]
    df = viso_with_iq.copy()
    df["is_caught"] = (
        (df["P8A_score"] >= tau_p8a)
        | (df["E2B_3200_score"] >= tau_e2b)
        | (df["E3_6600_score"] >= tau_e3)
    ).astype(int)

    rows = []
    for subtype, sub_df in df.groupby("subtype"):
        caught = sub_df[sub_df.is_caught == 1]
        uncaught = sub_df[sub_df.is_caught == 0]
        for feat in IQ_FEATURES:
            a = caught[feat].values.astype(float)
            b = uncaught[feat].values.astype(float)
            if len(a) < 2 or len(b) < 2:
                rows.append(
                    {
                        "subtype": subtype,
                        "feature": feat,
                        "caught_mean": float(np.mean(a)) if len(a) else float("nan"),
                        "uncaught_mean": float(np.mean(b)) if len(b) else float("nan"),
                        "cohens_d_caught_minus_uncaught": float("nan"),
                        "abs_cohens_d": float("nan"),
                        "t_stat": float("nan"),
                        "p_value": float("nan"),
                        "n_caught": len(a),
                        "n_uncaught": len(b),
                    }
                )
                continue
            t, p = stats.ttest_ind(a, b, equal_var=False)
            d = cohens_d(a, b)
            rows.append(
                {
                    "subtype": subtype,
                    "feature": feat,
                    "caught_mean": float(np.mean(a)),
                    "uncaught_mean": float(np.mean(b)),
                    "cohens_d_caught_minus_uncaught": d,
                    "abs_cohens_d": abs(d) if not np.isnan(d) else float("nan"),
                    "t_stat": float(t),
                    "p_value": float(p),
                    "n_caught": len(a),
                    "n_uncaught": len(b),
                }
            )
    out_df = pd.DataFrame(rows).sort_values(
        ["subtype", "abs_cohens_d"], ascending=[True, False]
    )
    out_df.to_csv(OUT / "subtype_iq_caught_vs_uncaught.csv", index=False)
    return out_df, df


def step6c_logistic_within(df_caught: pd.DataFrame):
    """L1 logistic predicting is_caught from standardized IQ features, within each subtype.
    C=0.5, solver=liblinear, max_iter=1000, n_jobs=1, 5-fold CV accuracy.
    """
    rows = []
    for subtype, sub_df in df_caught.groupby("subtype"):
        X = sub_df[IQ_FEATURES].values.astype(float)
        y = sub_df["is_caught"].values.astype(int)
        # standardize on full subtype set
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        # CV accuracy
        if y.sum() < 2 or (len(y) - y.sum()) < 2:
            cv_acc = float("nan")
        else:
            n_pos = int(y.sum())
            n_neg = int(len(y) - y.sum())
            n_splits = min(5, n_pos, n_neg)
            if n_splits < 2:
                cv_acc = float("nan")
            else:
                clf_cv = LogisticRegression(
                    penalty="l1",
                    C=0.5,
                    solver="liblinear",
                    max_iter=1000,
                )
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                scores = cross_val_score(clf_cv, Xs, y, cv=skf, scoring="accuracy", n_jobs=1)
                cv_acc = float(np.mean(scores))
        # Fit on all for coefficients
        clf = LogisticRegression(
            penalty="l1",
            C=0.5,
            solver="liblinear",
            max_iter=1000,
        )
        clf.fit(Xs, y)
        coefs = clf.coef_.flatten()
        for feat, c in zip(IQ_FEATURES, coefs):
            rows.append(
                {
                    "subtype": subtype,
                    "feature": feat,
                    "std_coefficient": float(c),
                    "abs_coef": float(abs(c)),
                    "n_caught": int(y.sum()),
                    "n_uncaught": int(len(y) - y.sum()),
                }
            )
        # cv accuracy row
        rows.append(
            {
                "subtype": subtype,
                "feature": "__cv_accuracy__",
                "std_coefficient": cv_acc,
                "abs_coef": float("nan"),
                "n_caught": int(y.sum()),
                "n_uncaught": int(len(y) - y.sum()),
            }
        )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT / "subtype_logistic_within.csv", index=False)
    return out_df


def main():
    print("Loading viso joined ...")
    viso = load_viso_with_subtype()
    print(f"  shape={viso.shape}; subtype counts={viso['subtype'].value_counts().to_dict()}")

    print("Loading real scores for τ calibration ...")
    real_scores = load_real_scores()
    for k, v in real_scores.items():
        print(f"  {k}: n_real={len(v)}, mean={np.mean(v):.4f}")

    print("Step 2: per-subtype × per-ckpt recall sweep ...")
    rec = step2_recall_sweep(viso, real_scores)
    print(rec[rec.fpr_target == 0.10][["subtype", "ckpt", "n_caught", "n", "recall"]].to_string(index=False))

    print("Step 3: per-subtype score stats ...")
    step3_score_stats(viso)

    print("Step 4: per-subtype cross-ckpt correlations ...")
    step4_correlations(viso)

    print("Step 5: per-subtype frame coverage Venn ...")
    step5_frame_coverage(viso, real_scores)

    # Step 6: load IQ features
    print("Loading crop_attributes for IQ features ...")
    iq = pd.read_csv(CROP_ATTR)
    # build join key on filename
    iq["filename"] = iq["filename"].astype(str)
    viso["filename"] = viso["filename"].astype(str)
    viso_iq = viso.merge(iq[["filename", "subtype"] + IQ_FEATURES], on="filename", how="inner", suffixes=("", "_crop"))
    # subtype from the two sides should match — verify
    if "subtype_crop" in viso_iq.columns:
        mismatch = (viso_iq["subtype"] != viso_iq["subtype_crop"]).sum()
        if mismatch:
            print(f"  WARN: {mismatch} subtype mismatches between viso and crop_attributes")
        viso_iq = viso_iq.drop(columns=["subtype_crop"])
    print(f"  joined shape: {viso_iq.shape}")
    if len(viso_iq) != len(viso):
        print(f"  WARN: {len(viso) - len(viso_iq)} viso frames missing IQ features")

    print("Step 6a: IQ distribution by subtype ...")
    iq_dist = step6a_iq_distribution(viso_iq)
    print(iq_dist[["feature", "raw_mean", "teams_mean", "cohens_d_raw_minus_teams", "p_value"]].to_string(index=False))

    print("Step 6b: caught vs uncaught (within subtype) IQ ...")
    iq_cvu, df_caught = step6b_caught_vs_uncaught(viso_iq, real_scores)

    print("Step 6c: L1 logistic within subtype ...")
    step6c_logistic_within(df_caught)

    print("Done. Outputs in", OUT)


if __name__ == "__main__":
    main()
