"""A2 follow-up — strong-regularization + PCA-projection probes.

Full-lockbox + PC_Generator-only probes both hit AUC=1.0 — likely memorization
given dim=768 vs N=77-453. This script sweeps C∈{1e-3, 1e-2, 1e-1, 1, 10} and
also projects to PCA-50 first.

If even strong regularization keeps AUC near 1.0, the encoder representation
truly admits clean linear separation; if AUC drops below trained-head AUC at
strong reg, the result was memorization.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT_DIR = REPO_ROOT / "analysis" / "cpu_diagnostics_2026-05-11_a2_linear_probe"
CACHE_DIR = OUT_DIR / "_cache"

CKPTS = ["T4_LAMBDA1_TOP_N_STEP10500", "P8A_REFERENCE_STEP5000"]
RANDOM_SEED = 42

logger = logging.getLogger("a2-reg")


def probe_fold(feats: np.ndarray, labels: np.ndarray, C: float, pca_dim: int | None) -> list:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score

    rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for fold, (tr, te) in enumerate(skf.split(feats, labels)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(feats[tr])
        X_te = scaler.transform(feats[te])
        if pca_dim is not None:
            pca = PCA(n_components=pca_dim, random_state=RANDOM_SEED)
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)
        clf = LogisticRegression(C=C, max_iter=4000, n_jobs=1)
        clf.fit(X_tr, labels[tr])
        scores = clf.predict_proba(X_te)[:, 1]
        try:
            auc = roc_auc_score(labels[te], scores)
        except ValueError:
            auc = float("nan")
        rows.append({"fold": fold, "auc": float(auc)})
    return rows


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    grid_rows = []
    C_grid = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    pca_grid = [None, 50, 20, 10]
    for ckpt in CKPTS:
        cache_file = CACHE_DIR / f"video_feats__{ckpt}__L11.npz"
        data = np.load(cache_file, allow_pickle=True)
        feats = data["features"]
        labels = data["labels"]
        for C in C_grid:
            for pca in pca_grid:
                rows = probe_fold(feats, labels, C=C, pca_dim=pca)
                aucs = [r["auc"] for r in rows]
                grid_rows.append({"ckpt": ckpt, "C": C, "pca": str(pca),
                                  "mean_auc": np.mean(aucs), "std_auc": np.std(aucs),
                                  "min_auc": np.min(aucs), "max_auc": np.max(aucs)})
                logger.info("[%s] C=%g pca=%s  mean_AUC=%.4f ± %.4f",
                            ckpt, C, pca, np.mean(aucs), np.std(aucs))
    df = pd.DataFrame(grid_rows)
    df.to_csv(OUT_DIR / "lockbox_probe_regularized_grid.csv", index=False)
    print(df.pivot(index=["ckpt", "C"], columns="pca", values="mean_auc"))


if __name__ == "__main__":
    raise SystemExit(main())
