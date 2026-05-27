"""Xinhe-fake cohort mechanism analysis.

What distinguishes the cohorts where all ckpts catch <30% at mode B
(xinhe-fake-1, -2, -3) from the cohort where all ckpts catch ≥95%
(xinhe-fake-7)?

Uses:
  - per-frame scores from team-identity readout (5 ckpts × 1099 Xinhe-fake frames)
  - frozen-CLIP L11 features (768-d) cached per frame

Analyses:
  1. Per-cohort score distributions across 5 ckpts
  2. Per-cohort centroid in frozen-CLIP feature space
  3. Cohort-pairwise centroid distances + nearest-neighbor structure
  4. Top PCA components of cohort centroids
  5. Probe: does a single axis (in CLIP space) explain the easy/hard split?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
PER_FRAME = ROOT / "analysis/team_identity_deploy_readout_expanded_2026-05-23/outputs/per_frame_full.csv"
CLIP_FEATS = ROOT / "analysis/frozen_clip_team_identity_baseline_2026-05-23/outputs/clip_frozen_l11__team_identity_n5941.npz"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

CKPTS = ["P8A", "E2B", "T5C", "SlotAv2_CLS", "SlotAv2_FACE"]

HARD_COHORTS = ["live_prod__xinhe-fake-1", "live_prod__xinhe-fake-2", "live_prod__xinhe-fake-3"]
EASY_COHORTS = ["live_prod__xinhe-fake-7"]  # per readout §6.2
MED_COHORTS = ["live_prod__xinhe-fake-4", "live_prod__xinhe-fake-5", "live_prod__xinhe-fake-6",
               "live_prod__xinhe-fake-8", "live_prod__xinhe-fake-9-glasses",
               "live_prod__xinhe-fake-10-glasses", "live_prod__xinhe-fake-11-glasses",
               "live_prod__xinhe-fake-8-glasses"]


def main() -> None:
    df = pd.read_csv(PER_FRAME)
    xinhe = df[df["role"] == "fake_target_Xinhe"].copy()
    print(f"Xinhe-fake total frames: {len(xinhe)}")

    feats_data = np.load(CLIP_FEATS, allow_pickle=True)
    feat_frame_paths = feats_data["frame_paths"]
    feat_array = feats_data["features"]
    print(f"frozen-CLIP feature cache: {feat_array.shape}")

    # Build path → feature index
    fp_to_idx = {fp: i for i, fp in enumerate(feat_frame_paths)}
    xinhe["_clip_idx"] = xinhe["frame_path"].map(fp_to_idx)
    n_missing = xinhe["_clip_idx"].isna().sum()
    print(f"frames without CLIP feature: {n_missing} / {len(xinhe)}")
    xinhe = xinhe.dropna(subset=["_clip_idx"]).copy()
    xinhe["_clip_idx"] = xinhe["_clip_idx"].astype(int)

    # ---- 1) Per-cohort score distributions ----
    score_summary = []
    for cohort, group in xinhe.groupby("base_identity"):
        n = len(group)
        row: dict = {"cohort": cohort, "n": n}
        for ckpt in CKPTS:
            col = f"prob_{ckpt}"
            row[f"{ckpt}_mean"] = float(group[col].mean())
            row[f"{ckpt}_std"] = float(group[col].std())
            row[f"{ckpt}_p25"] = float(group[col].quantile(0.25))
            row[f"{ckpt}_p75"] = float(group[col].quantile(0.75))
        # Cohort difficulty: mean fake-recall at τ=0.5 averaged across ckpts
        recalls = [(group[f"prob_{c}"] >= 0.5).mean() for c in CKPTS]
        row["mean_recall_05"] = float(np.mean(recalls))
        row["min_recall_05"] = float(np.min(recalls))
        row["max_recall_05"] = float(np.max(recalls))
        # difficulty class
        if cohort in HARD_COHORTS:
            row["difficulty"] = "hard"
        elif cohort in EASY_COHORTS:
            row["difficulty"] = "easy"
        elif cohort in MED_COHORTS:
            row["difficulty"] = "medium"
        else:
            row["difficulty"] = "other"
        score_summary.append(row)
    score_df = pd.DataFrame(score_summary).sort_values("mean_recall_05")
    score_df.to_csv(OUT / "per_cohort_score_summary.csv", index=False)

    print("\n=== Per-cohort score summary (sorted by mean recall @τ=0.5) ===")
    cols_to_show = ["cohort", "difficulty", "n", "mean_recall_05",
                    "P8A_mean", "E2B_mean", "T5C_mean", "SlotAv2_CLS_mean", "SlotAv2_FACE_mean"]
    print(score_df[cols_to_show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- 2) Per-cohort centroids in CLIP space ----
    centroids: dict = {}
    cohort_sizes: dict = {}
    for cohort, group in xinhe.groupby("base_identity"):
        idx = group["_clip_idx"].to_numpy()
        feats = feat_array[idx]
        centroids[cohort] = feats.mean(axis=0)
        cohort_sizes[cohort] = len(group)
    cohort_names = list(centroids.keys())

    # ---- 3) Pairwise centroid distances ----
    n_coh = len(cohort_names)
    cent_mat = np.array([centroids[c] for c in cohort_names])
    # Cosine distance
    norms = np.linalg.norm(cent_mat, axis=1)
    cosine_sim = cent_mat @ cent_mat.T / np.outer(norms, norms)
    dist_df = pd.DataFrame(1 - cosine_sim, index=cohort_names, columns=cohort_names)
    dist_df.to_csv(OUT / "centroid_cosine_distance.csv")

    # For each cohort, identify the 3 nearest cohort-centroids
    print("\n=== Nearest cohorts by centroid cosine distance ===")
    nn_rows = []
    for c in cohort_names:
        d = dist_df.loc[c].drop(c).sort_values()
        nn_rows.append({"cohort": c, "nn1": d.index[0], "nn1_dist": d.iloc[0],
                        "nn2": d.index[1], "nn2_dist": d.iloc[1],
                        "nn3": d.index[2], "nn3_dist": d.iloc[2]})
    nn_df = pd.DataFrame(nn_rows)
    nn_df.to_csv(OUT / "centroid_nn.csv", index=False)
    print(nn_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- 4) Hard vs easy: probe whether the centroids carve a linear separation ----
    # Train LR to predict 'hard' vs 'easy' from per-frame CLIP features
    hard_mask = xinhe["base_identity"].isin(HARD_COHORTS)
    easy_mask = xinhe["base_identity"].isin(EASY_COHORTS)
    he_mask = hard_mask | easy_mask
    X = feat_array[xinhe.loc[he_mask, "_clip_idx"].to_numpy()]
    y = hard_mask.loc[he_mask].astype(int).to_numpy()
    print(f"\nHard vs easy probe: hard={y.sum()}, easy={len(y) - y.sum()}")
    if len(y) and 0 < y.sum() < len(y):
        lr = LogisticRegression(C=1.0, max_iter=2000, n_jobs=1)
        lr.fit(X, y)
        in_pred = lr.predict_proba(X)[:, 1]
        in_auc = roc_auc_score(y, in_pred)
        print(f"In-sample LR hard-vs-easy AUC: {in_auc:.4f}")
        # 5-fold CV
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(lr, X, y, cv=5, scoring="roc_auc", n_jobs=1)
        print(f"5-fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        # Save the LR coefficient (the 'hard' axis in CLIP space)
        np.save(OUT / "hard_vs_easy_lr_coef.npy", lr.coef_[0])

        # Project ALL Xinhe-fake cohorts on this axis (using LR signed margin) and report per-cohort mean
        all_feats = feat_array[xinhe["_clip_idx"].to_numpy()]
        margins = lr.decision_function(all_feats)
        xinhe_with_margin = xinhe.assign(hard_axis_margin=margins)
        axis_summary = xinhe_with_margin.groupby("base_identity").agg(
            n=("hard_axis_margin", "size"),
            mean_hard_axis=("hard_axis_margin", "mean"),
            std_hard_axis=("hard_axis_margin", "std"),
        ).reset_index()
        axis_summary["difficulty"] = axis_summary["cohort" if "cohort" in axis_summary.columns else "base_identity"].apply(
            lambda c: "hard" if c in HARD_COHORTS else "easy" if c in EASY_COHORTS else "medium" if c in MED_COHORTS else "other"
        )
        axis_summary = axis_summary.sort_values("mean_hard_axis", ascending=False)
        axis_summary.to_csv(OUT / "hard_axis_per_cohort.csv", index=False)
        print("\n=== Projection onto LR hard-vs-easy axis (per-cohort mean margin) ===")
        print(axis_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- 5) PCA on cohort centroids — find dominant axes of cohort variation ----
    if n_coh >= 3:
        pca = PCA(n_components=min(5, n_coh - 1))
        cent_centered = cent_mat - cent_mat.mean(axis=0, keepdims=True)
        proj = pca.fit_transform(cent_centered)
        print(f"\n=== PCA on Xinhe-fake cohort centroids: explained variance ratio ===")
        print(pca.explained_variance_ratio_)
        proj_df = pd.DataFrame(proj, index=cohort_names,
                               columns=[f"PC{i+1}" for i in range(proj.shape[1])])
        proj_df["difficulty"] = [
            "hard" if c in HARD_COHORTS else "easy" if c in EASY_COHORTS
            else "medium" if c in MED_COHORTS else "other"
            for c in cohort_names
        ]
        proj_df["n"] = [cohort_sizes[c] for c in cohort_names]
        proj_df.to_csv(OUT / "cohort_pca_projection.csv")
        print("\nCohort projections:")
        print(proj_df.to_string(float_format=lambda x: f"{x:.3f}"))

    # ---- 6) Aggregate hard vs easy vs medium: per-cohort score means ----
    print("\n=== Aggregate score means by difficulty class (each ckpt) ===")
    diff_groups = {
        "hard": HARD_COHORTS,
        "easy": EASY_COHORTS,
        "medium": MED_COHORTS,
    }
    for ckpt in CKPTS:
        col = f"prob_{ckpt}"
        means = {}
        for label, cohorts in diff_groups.items():
            sub = xinhe[xinhe["base_identity"].isin(cohorts)][col]
            means[label] = (sub.mean(), sub.std(), len(sub))
        print(f"  {ckpt:13s}  hard={means['hard'][0]:.3f}±{means['hard'][1]:.3f} (n={means['hard'][2]})  "
              f"easy={means['easy'][0]:.3f}±{means['easy'][1]:.3f} (n={means['easy'][2]})  "
              f"medium={means['medium'][0]:.3f}±{means['medium'][1]:.3f} (n={means['medium'][2]})")


if __name__ == "__main__":
    main()
