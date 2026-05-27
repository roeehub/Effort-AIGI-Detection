"""KLIEP-style substrate matching check (discriminator-based density-ratio proxy).

Comparisons:
  1. OPTB-real ↔ team-identity-real (deploy-relevant): does training-corpus look like deploy?
  2. OPTB-fake ↔ team-identity-fake: same on fake side
  3. Per-human: OPTB-real ↔ team-id-real-{Noyn, Roee_W, Xiang, Xinhe, dor}
  4. Intra-team-identity: between-human reals — are team humans CLIP-distinguishable?

Method: LogisticRegression discriminator on frozen-CLIP L11 (768-d) features.
- 5-fold stratified CV accuracy + AUC = "how distinguishable are the pools?"
- 50% accuracy ≈ identical distributions; 100% ≈ completely separable
- Effective sample size estimate (Kanamori-style): for LR-density-ratio
  w(x) = (1 - p(x)) / p(x) * (n_target / n_source), ESS = (sum w)^2 / sum(w^2)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
TEAM_FEATS = ROOT / "analysis/frozen_clip_team_identity_baseline_2026-05-23/outputs/clip_frozen_l11__team_identity_n5941.npz"
OPTB_FEATS = ROOT / "analysis/frozen_clip_team_identity_baseline_2026-05-23/outputs/clip_frozen_l11__training_optionB_n6000.npz"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_CV = 5


def compare_pools(X_a: np.ndarray, X_b: np.ndarray, name_a: str, name_b: str) -> dict:
    """LR discriminator A vs B. A=label 0, B=label 1."""
    X = np.concatenate([X_a, X_b], axis=0)
    y = np.concatenate([np.zeros(len(X_a), dtype=int), np.ones(len(X_b), dtype=int)])
    lr = LogisticRegression(C=1.0, max_iter=2000, n_jobs=1, solver="lbfgs")
    skf = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=SEED)
    acc_scores = cross_val_score(lr, X, y, cv=skf, scoring="accuracy", n_jobs=1)
    auc_scores = cross_val_score(lr, X, y, cv=skf, scoring="roc_auc", n_jobs=1)

    # In-sample for the importance-ratio
    lr.fit(X, y)
    p = lr.predict_proba(X_a)[:, 1]  # p(target | x) for source points
    # density ratio w = p / (1 - p) — drop the n_target / n_source scaler since we report ESS as fraction of n_a
    p_clipped = np.clip(p, 1e-6, 1 - 1e-6)
    w = p_clipped / (1 - p_clipped)
    ess = float((w.sum() ** 2) / (w ** 2).sum())
    ess_frac = ess / len(w)

    return {
        "name_a": name_a, "name_b": name_b,
        "n_a": len(X_a), "n_b": len(X_b),
        "cv_acc_mean": float(acc_scores.mean()), "cv_acc_std": float(acc_scores.std()),
        "cv_auc_mean": float(auc_scores.mean()), "cv_auc_std": float(auc_scores.std()),
        "ess_on_a": ess, "ess_frac_on_a": ess_frac,
        # interpretation hint
        "interpretation": (
            "near-identical (≤55%)" if acc_scores.mean() <= 0.55 else
            "well-matched (55-65%)" if acc_scores.mean() <= 0.65 else
            "moderately-mismatched (65-75%)" if acc_scores.mean() <= 0.75 else
            "well-mismatched (75-85%)" if acc_scores.mean() <= 0.85 else
            "highly-mismatched (>85%)"
        ),
    }


def main() -> None:
    print("Loading caches...")
    team_data = np.load(TEAM_FEATS, allow_pickle=True)
    team_feats = team_data["features"]
    team_labels = team_data["labels"]
    team_humans = team_data["humans"]
    team_roles = team_data["roles"]
    team_deploy = team_data["deploy_relevant"]

    optb_data = np.load(OPTB_FEATS, allow_pickle=True)
    optb_feats = optb_data["features"]
    optb_labels = optb_data["labels"]

    # Subset team-identity to deploy-relevant
    deploy_mask = team_deploy.astype(bool)
    team_real_mask = deploy_mask & (np.array([r == "real" for r in team_roles]))
    # Fakes have roles like "fake_target_dor", "fake_target_Xinhe", etc.
    team_fake_mask = deploy_mask & (~np.array([r == "real" for r in team_roles]))

    team_real = team_feats[team_real_mask]
    team_fake = team_feats[team_fake_mask]
    print(f"team-identity-real (deploy): {len(team_real)}")
    print(f"team-identity-fake (deploy): {len(team_fake)}")

    optb_real = optb_feats[optb_labels == 0]
    optb_fake = optb_feats[optb_labels == 1]
    print(f"OPTB-real: {len(optb_real)}")
    print(f"OPTB-fake: {len(optb_fake)}")

    rows: list[dict] = []

    print("\n=== Aggregate comparisons ===")
    # OPTB-real vs team-identity-real
    r = compare_pools(optb_real, team_real, "OPTB_real", "team_id_real")
    rows.append(r)
    print(f"  OPTB-real vs team-id-real:  cv_acc={r['cv_acc_mean']:.3f}±{r['cv_acc_std']:.3f}  cv_auc={r['cv_auc_mean']:.3f}  ESS_on_OPTB={r['ess_frac_on_a']:.3f}  [{r['interpretation']}]")

    # OPTB-fake vs team-identity-fake
    r = compare_pools(optb_fake, team_fake, "OPTB_fake", "team_id_fake")
    rows.append(r)
    print(f"  OPTB-fake vs team-id-fake:  cv_acc={r['cv_acc_mean']:.3f}±{r['cv_acc_std']:.3f}  cv_auc={r['cv_auc_mean']:.3f}  ESS_on_OPTB={r['ess_frac_on_a']:.3f}  [{r['interpretation']}]")

    # Sanity check: OPTB-real vs OPTB-fake (should be highly distinguishable in CLIP space)
    r = compare_pools(optb_real, optb_fake, "OPTB_real", "OPTB_fake")
    rows.append(r)
    print(f"  OPTB-real vs OPTB-fake (sanity): cv_acc={r['cv_acc_mean']:.3f}")

    # team-real vs team-fake (also a sanity check)
    r = compare_pools(team_real, team_fake, "team_id_real", "team_id_fake")
    rows.append(r)
    print(f"  team-id-real vs team-id-fake (sanity): cv_acc={r['cv_acc_mean']:.3f}")

    print("\n=== OPTB-real vs each team-human's real cohort ===")
    for human in ["Noyn", "Roee_Windows", "Xiang", "Xinhe", "dor"]:
        h_mask = deploy_mask & (team_humans == human) & (np.array([r == "real" for r in team_roles]))
        h_feats = team_feats[h_mask]
        if len(h_feats) < 30:
            print(f"  {human:15s}  n={len(h_feats)} (too small, skip)")
            continue
        r = compare_pools(optb_real, h_feats, "OPTB_real", f"team_id_real_{human}")
        rows.append(r)
        print(f"  OPTB-real vs {human:15s} n={len(h_feats):4d}  cv_acc={r['cv_acc_mean']:.3f}±{r['cv_acc_std']:.3f}  cv_auc={r['cv_auc_mean']:.3f}  ESS_on_OPTB={r['ess_frac_on_a']:.3f}  [{r['interpretation']}]")

    print("\n=== Intra-team-identity: between-human reals ===")
    humans = ["Noyn", "Roee_Windows", "Xiang", "Xinhe", "dor"]
    h_feats: dict = {}
    for human in humans:
        h_mask = deploy_mask & (team_humans == human) & (np.array([r == "real" for r in team_roles]))
        h_feats[human] = team_feats[h_mask]
    pairs_done = set()
    for ha in humans:
        for hb in humans:
            if ha == hb:
                continue
            pair = tuple(sorted([ha, hb]))
            if pair in pairs_done:
                continue
            pairs_done.add(pair)
            if len(h_feats[pair[0]]) < 30 or len(h_feats[pair[1]]) < 30:
                continue
            r = compare_pools(h_feats[pair[0]], h_feats[pair[1]], pair[0], pair[1])
            rows.append(r)
            print(f"  {pair[0]:15s} vs {pair[1]:15s}  cv_acc={r['cv_acc_mean']:.3f}  cv_auc={r['cv_auc_mean']:.3f}  [{r['interpretation']}]")

    pd.DataFrame(rows).to_csv(OUT / "discriminator_results.csv", index=False)
    print(f"\nSaved {len(rows)} rows to outputs/discriminator_results.csv")


if __name__ == "__main__":
    main()
