"""
Job 8 — Multi-ckpt anchor audit (2026-05-04).

Three orthogonal lenses + headroom across P8A / E2B_3200 / E3_6600.
NEUTRAL framing — symmetric candidate treatment, no winner declared.

Pure CPU. Reuses cached scores in
analysis/cpu_followups_2026-05-04/raw_reports/. No GPU, no model inference.

Outputs land in this script's directory.
random_state=42 throughout.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
)
RAW = ROOT / "analysis" / "cpu_followups_2026-05-04" / "raw_reports"
OUT = ROOT / "analysis" / "job_8_anchor_audit_2026-05-04"
OUT.mkdir(parents=True, exist_ok=True)

CROP_ATTR = ROOT / "analysis" / "score_distribution_2026-05-02" / "outputs" / "crop_attributes.csv"

CKPTS = {
    "P8A": "p8a_reference_step5000",
    "E2B_3200": "e2b_top_n_step3200",
    "E3_6600": "e3_top_n_step6600",
}

FAKE_SUITES = [
    "visomaster_enhanced_macro_dev",
    "deeplive_enhanced_dev",
    "teams_fake_all_dev",
    "teams_fake_all_lockbox",
]

REAL_DEV = "teams_real_all_dev"
REAL_LOCKBOX = "teams_real_all_lockbox"
FAKE_DEV = "teams_fake_all_dev"
FAKE_LOCKBOX = "teams_fake_all_lockbox"

FPR_TARGETS = [0.02, 0.05, 0.10, 0.20]

IQ_FEATURES = [
    "luma_p10",
    "luma_p90",
    "luma_mean",
    "laplacian_var",
    "sobel_edge_mean",
    "saturation_mean",
    "skin_frac",
]

RANDOM_STATE = 42


def load_scores(suite: str, ckpt_tag: str) -> pd.DataFrame:
    path = RAW / f"{suite}_{ckpt_tag}_frames_report.csv"
    df = pd.read_csv(path)
    return df


def quantile_dict(scores: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(scores)),
        "mean": float(np.mean(scores)),
        "p10": float(np.quantile(scores, 0.10)),
        "p25": float(np.quantile(scores, 0.25)),
        "p50": float(np.quantile(scores, 0.50)),
        "p75": float(np.quantile(scores, 0.75)),
        "p90": float(np.quantile(scores, 0.90)),
        "std": float(np.std(scores)),
    }


def lens1_substrate_invariance() -> pd.DataFrame:
    """Lens 1: distributional similarity dev vs lockbox (reals AND fakes)."""
    rows = []
    for ckpt_name, ckpt_tag in CKPTS.items():
        # Reals
        real_dev = load_scores(REAL_DEV, ckpt_tag)["frame_prob"].values
        real_lock = load_scores(REAL_LOCKBOX, ckpt_tag)["frame_prob"].values
        rd = quantile_dict(real_dev)
        rl = quantile_dict(real_lock)
        ks_real = stats.ks_2samp(real_dev, real_lock)

        # Fakes
        fake_dev = load_scores(FAKE_DEV, ckpt_tag)["frame_prob"].values
        fake_lock = load_scores(FAKE_LOCKBOX, ckpt_tag)["frame_prob"].values
        fd = quantile_dict(fake_dev)
        fl = quantile_dict(fake_lock)
        ks_fake = stats.ks_2samp(fake_dev, fake_lock)

        row = {
            "ckpt": ckpt_name,
            # Reals — dev
            "real_dev_n": rd["n"],
            "real_dev_mean": rd["mean"],
            "real_dev_p10": rd["p10"],
            "real_dev_p25": rd["p25"],
            "real_dev_p50": rd["p50"],
            "real_dev_p75": rd["p75"],
            "real_dev_p90": rd["p90"],
            # Reals — lockbox
            "real_lockbox_n": rl["n"],
            "real_lockbox_mean": rl["mean"],
            "real_lockbox_p10": rl["p10"],
            "real_lockbox_p25": rl["p25"],
            "real_lockbox_p50": rl["p50"],
            "real_lockbox_p75": rl["p75"],
            "real_lockbox_p90": rl["p90"],
            # Real deltas
            "real_delta_mean": abs(rl["mean"] - rd["mean"]),
            "real_delta_p50": abs(rl["p50"] - rd["p50"]),
            "real_delta_p90": abs(rl["p90"] - rd["p90"]),
            "real_ks_stat": float(ks_real.statistic),
            "real_ks_pvalue": float(ks_real.pvalue),
            # Fakes — dev
            "fake_dev_n": fd["n"],
            "fake_dev_mean": fd["mean"],
            "fake_dev_p50": fd["p50"],
            # Fakes — lockbox
            "fake_lockbox_n": fl["n"],
            "fake_lockbox_mean": fl["mean"],
            "fake_lockbox_p50": fl["p50"],
            # Fake deltas
            "fake_delta_mean": abs(fl["mean"] - fd["mean"]),
            "fake_delta_p50": abs(fl["p50"] - fd["p50"]),
            "fake_ks_stat": float(ks_fake.statistic),
            "fake_ks_pvalue": float(ks_fake.pvalue),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "substrate_invariance_per_ckpt.csv", index=False)
    print(f"[Lens 1] Wrote substrate_invariance_per_ckpt.csv ({len(df)} rows)")
    return df


def lens2_calibration_lift() -> pd.DataFrame:
    """Lens 2: dev-cal vs lockbox-cal recall on each fake suite."""
    rows = []
    for ckpt_name, ckpt_tag in CKPTS.items():
        real_dev_scores = load_scores(REAL_DEV, ckpt_tag)["frame_prob"].values
        real_lock_scores = load_scores(REAL_LOCKBOX, ckpt_tag)["frame_prob"].values

        for suite in FAKE_SUITES:
            fake_scores = load_scores(suite, ckpt_tag)["frame_prob"].values
            for fpr in FPR_TARGETS:
                tau_dev = float(np.quantile(real_dev_scores, 1.0 - fpr))
                tau_lock = float(np.quantile(real_lock_scores, 1.0 - fpr))
                recall_dev = float(np.mean(fake_scores > tau_dev))
                recall_lock = float(np.mean(fake_scores > tau_lock))
                rows.append(
                    {
                        "ckpt": ckpt_name,
                        "fake_suite": suite,
                        "fpr_target": fpr,
                        "tau_dev_cal": tau_dev,
                        "tau_lockbox_cal": tau_lock,
                        "recall_dev_cal": recall_dev,
                        "recall_lockbox_cal": recall_lock,
                        "lift": recall_lock - recall_dev,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "calibration_lift_per_ckpt.csv", index=False)
    print(f"[Lens 2] Wrote calibration_lift_per_ckpt.csv ({len(df)} rows)")
    return df


def lens3_iq_shortcut() -> pd.DataFrame:
    """Lens 3: L1 logreg on 7 IQ features predicting caught/uncaught (viso)."""
    crop = pd.read_csv(CROP_ATTR)  # 550 viso frames
    # crop has filename ending in .png; frame_path in scores ends in .png too.
    rows = []
    for ckpt_name, ckpt_tag in CKPTS.items():
        scores_df = load_scores("visomaster_enhanced_macro_dev", ckpt_tag).copy()
        scores_df["filename"] = scores_df["frame_path"].str.split("/").str[-1]
        # Calibrate τ at FPR=10% on teams_real_all_dev
        real_dev = load_scores(REAL_DEV, ckpt_tag)["frame_prob"].values
        tau_10 = float(np.quantile(real_dev, 0.90))
        scores_df["is_caught"] = (scores_df["frame_prob"] > tau_10).astype(int)

        merged = scores_df.merge(crop, on="filename", how="inner", validate="one_to_one")
        assert len(merged) == 550, f"merge mismatch: {len(merged)}"

        X = merged[IQ_FEATURES].values
        y = merged["is_caught"].values
        baseline = max(y.mean(), 1 - y.mean())
        caught_n = int(y.sum())
        total_n = int(len(y))

        if y.sum() < 5 or (1 - y.mean()) < 0.05:
            # Degenerate — skip CV, store baseline only
            rows.append(
                {
                    "ckpt": ckpt_name,
                    "fake_suite": "visomaster_enhanced_macro_dev",
                    "n": total_n,
                    "n_caught": caught_n,
                    "majority_baseline_acc": baseline,
                    "cv_acc_mean": np.nan,
                    "cv_acc_std": np.nan,
                    "cv_auc_mean": np.nan,
                    "cv_auc_std": np.nan,
                    "top1_feature": "",
                    "top1_coef": np.nan,
                    "top2_feature": "",
                    "top2_coef": np.nan,
                    "top3_feature": "",
                    "top3_coef": np.nan,
                    "note": "degenerate_class_balance",
                }
            )
            continue

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        clf = LogisticRegression(
            penalty="l1",
            C=0.5,
            solver="liblinear",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        acc = cross_val_score(clf, Xs, y, cv=skf, scoring="accuracy", n_jobs=1)
        # AUC may fail if a fold has only one class — guard
        try:
            auc = cross_val_score(clf, Xs, y, cv=skf, scoring="roc_auc", n_jobs=1)
            auc_mean = float(auc.mean())
            auc_std = float(auc.std())
        except Exception:
            auc_mean = np.nan
            auc_std = np.nan

        # Refit on all data to extract std-coefficients
        clf.fit(Xs, y)
        coefs = clf.coef_.ravel()
        feat_coefs = sorted(
            zip(IQ_FEATURES, coefs), key=lambda kv: abs(kv[1]), reverse=True
        )
        top3 = feat_coefs[:3]

        rows.append(
            {
                "ckpt": ckpt_name,
                "fake_suite": "visomaster_enhanced_macro_dev",
                "n": total_n,
                "n_caught": caught_n,
                "majority_baseline_acc": float(baseline),
                "cv_acc_mean": float(acc.mean()),
                "cv_acc_std": float(acc.std()),
                "cv_auc_mean": auc_mean,
                "cv_auc_std": auc_std,
                "top1_feature": top3[0][0],
                "top1_coef": float(top3[0][1]),
                "top2_feature": top3[1][0],
                "top2_coef": float(top3[1][1]),
                "top3_feature": top3[2][0],
                "top3_coef": float(top3[2][1]),
                "note": "",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "iq_shortcut_per_ckpt.csv", index=False)
    print(f"[Lens 3] Wrote iq_shortcut_per_ckpt.csv ({len(df)} rows)")
    return df


def lens4_headroom() -> pd.DataFrame:
    """Lens 4: AUC vs operational recall (FPR=10% and FPR=50%) per ckpt × suite."""
    rows = []
    for ckpt_name, ckpt_tag in CKPTS.items():
        real_dev = load_scores(REAL_DEV, ckpt_tag)["frame_prob"].values
        for suite in FAKE_SUITES:
            fake_scores = load_scores(suite, ckpt_tag)["frame_prob"].values
            y_true = np.concatenate([np.zeros(len(real_dev)), np.ones(len(fake_scores))])
            y_score = np.concatenate([real_dev, fake_scores])
            auc = float(roc_auc_score(y_true, y_score))
            tau_10 = float(np.quantile(real_dev, 0.90))
            tau_50 = float(np.quantile(real_dev, 0.50))
            recall_10 = float(np.mean(fake_scores > tau_10))
            recall_50 = float(np.mean(fake_scores > tau_50))
            gap = (auc * 100.0) - (recall_10 * 100.0)
            rows.append(
                {
                    "ckpt": ckpt_name,
                    "fake_suite": suite,
                    "n_real": int(len(real_dev)),
                    "n_fake": int(len(fake_scores)),
                    "auc": auc,
                    "recall_at_FPR10": recall_10,
                    "recall_at_FPR50": recall_50,
                    "gap_AUC_minus_recall_FPR10_pp": gap,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "headroom_per_ckpt.csv", index=False)
    print(f"[Lens 4] Wrote headroom_per_ckpt.csv ({len(df)} rows)")
    return df


def build_decision_matrix(
    lens1: pd.DataFrame,
    lens2: pd.DataFrame,
    lens3: pd.DataFrame,
    lens4: pd.DataFrame,
) -> pd.DataFrame:
    """One row per ckpt — neutral metric columns."""
    rows = []
    for ckpt_name in CKPTS:
        l1 = lens1[lens1["ckpt"] == ckpt_name].iloc[0]
        l3 = lens3[lens3["ckpt"] == ckpt_name].iloc[0] if (lens3["ckpt"] == ckpt_name).any() else None

        # Lens 2 summaries — average lift across all 4 suites at FPR=10%
        l2_fpr10 = lens2[(lens2["ckpt"] == ckpt_name) & (lens2["fpr_target"] == 0.10)]
        # Lens 2 specific: teams_fake_all_lockbox at FPR=10%
        l2_lock_fpr10 = l2_fpr10[l2_fpr10["fake_suite"] == "teams_fake_all_lockbox"]
        # Lens 2 specific: viso at FPR=10%
        l2_viso_fpr10 = l2_fpr10[l2_fpr10["fake_suite"] == "visomaster_enhanced_macro_dev"]
        # Lens 2 specific: deeplive at FPR=10%
        l2_dl_fpr10 = l2_fpr10[l2_fpr10["fake_suite"] == "deeplive_enhanced_dev"]

        # Lens 4 summaries — viso, deeplive, teams_fake_lockbox AUCs
        l4 = lens4[lens4["ckpt"] == ckpt_name]
        l4_viso = l4[l4["fake_suite"] == "visomaster_enhanced_macro_dev"].iloc[0]
        l4_dl = l4[l4["fake_suite"] == "deeplive_enhanced_dev"].iloc[0]
        l4_tflock = l4[l4["fake_suite"] == "teams_fake_all_lockbox"].iloc[0]

        row = {
            "ckpt": ckpt_name,
            # Lens 1 — substrate invariance
            "L1_real_delta_mean": l1["real_delta_mean"],
            "L1_real_delta_p50": l1["real_delta_p50"],
            "L1_real_ks_stat": l1["real_ks_stat"],
            "L1_fake_delta_mean": l1["fake_delta_mean"],
            "L1_fake_delta_p50": l1["fake_delta_p50"],
            "L1_fake_ks_stat": l1["fake_ks_stat"],
            # Lens 2 — calibration lift at FPR=10%
            "L2_lift_viso_fpr10": (
                float(l2_viso_fpr10["lift"].iloc[0]) if len(l2_viso_fpr10) else np.nan
            ),
            "L2_lift_deeplive_fpr10": (
                float(l2_dl_fpr10["lift"].iloc[0]) if len(l2_dl_fpr10) else np.nan
            ),
            "L2_lift_teams_fake_lockbox_fpr10": (
                float(l2_lock_fpr10["lift"].iloc[0]) if len(l2_lock_fpr10) else np.nan
            ),
            "L2_recall_lockbox_cal_teams_fake_lockbox_fpr10": (
                float(l2_lock_fpr10["recall_lockbox_cal"].iloc[0])
                if len(l2_lock_fpr10)
                else np.nan
            ),
            "L2_recall_dev_cal_teams_fake_lockbox_fpr10": (
                float(l2_lock_fpr10["recall_dev_cal"].iloc[0])
                if len(l2_lock_fpr10)
                else np.nan
            ),
            # Lens 3 — IQ shortcut on viso
            "L3_iq_cv_acc_mean_viso": (
                float(l3["cv_acc_mean"]) if l3 is not None else np.nan
            ),
            "L3_iq_cv_auc_mean_viso": (
                float(l3["cv_auc_mean"]) if l3 is not None else np.nan
            ),
            "L3_iq_majority_baseline_viso": (
                float(l3["majority_baseline_acc"]) if l3 is not None else np.nan
            ),
            "L3_iq_top1_feature_viso": (
                str(l3["top1_feature"]) if l3 is not None else ""
            ),
            # Lens 4 — headroom
            "L4_auc_viso": float(l4_viso["auc"]),
            "L4_recall_FPR10_viso": float(l4_viso["recall_at_FPR10"]),
            "L4_gap_pp_viso": float(l4_viso["gap_AUC_minus_recall_FPR10_pp"]),
            "L4_auc_deeplive": float(l4_dl["auc"]),
            "L4_recall_FPR10_deeplive": float(l4_dl["recall_at_FPR10"]),
            "L4_gap_pp_deeplive": float(l4_dl["gap_AUC_minus_recall_FPR10_pp"]),
            "L4_auc_teams_fake_lockbox": float(l4_tflock["auc"]),
            "L4_recall_FPR10_teams_fake_lockbox": float(l4_tflock["recall_at_FPR10"]),
            "L4_gap_pp_teams_fake_lockbox": float(l4_tflock["gap_AUC_minus_recall_FPR10_pp"]),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "anchor_decision_matrix.csv", index=False)
    print(f"[Matrix] Wrote anchor_decision_matrix.csv ({len(df)} rows)")
    return df


def main():
    print("Job 8 anchor audit — starting")
    print(f"Random state: {RANDOM_STATE}")
    print(f"OUT: {OUT}")

    lens1 = lens1_substrate_invariance()
    lens2 = lens2_calibration_lift()
    lens3 = lens3_iq_shortcut()
    lens4 = lens4_headroom()
    matrix = build_decision_matrix(lens1, lens2, lens3, lens4)

    print("\n=== Decision matrix ===")
    print(matrix.to_string(index=False))


if __name__ == "__main__":
    main()
