"""A2 follow-up — restrict probe to PC_Generator-only subset (only source with both reals + fakes).

Full-lockbox probe gave AUC=1.0 for both ckpts but this is identity-driven:
reals = mostly dor_shkedi+Chikara, fakes = mostly Cam_Test. The probe doesn't
distinguish encoder-feature degradation from identity recovery.

This script re-runs the StratifiedKFold probe on PC_Generator-only videos
(real PC_Generator + fake PC_Generator), which removes the identity-confound.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT_DIR = REPO_ROOT / "analysis" / "cpu_diagnostics_2026-05-11_a2_linear_probe"
CACHE_DIR = OUT_DIR / "_cache"

CKPTS = ["T4_LAMBDA1_TOP_N_STEP10500", "P8A_REFERENCE_STEP5000"]
RANDOM_SEED = 42

logger = logging.getLogger("a2-pcgen")


def run_probe(feats: np.ndarray, labels: np.ndarray, label_for_log: str, n_splits: int = 5) -> list:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    rows = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    for fold, (tr, te) in enumerate(skf.split(feats, labels)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(feats[tr])
        X_te = scaler.transform(feats[te])
        clf = LogisticRegression(C=1.0, max_iter=2000, n_jobs=1)
        clf.fit(X_tr, labels[tr])
        scores = clf.predict_proba(X_te)[:, 1]
        try:
            auc = roc_auc_score(labels[te], scores)
        except ValueError:
            auc = float("nan")
        n_r = int((labels[te] == 0).sum())
        n_f = int((labels[te] == 1).sum())
        rows.append({"ckpt": label_for_log, "subset": "pcgen_only", "fold": fold,
                     "auc": float(auc), "n_real": n_r, "n_fake": n_f})
        logger.info("[%s] fold %d AUC=%.4f", label_for_log, fold, auc)
    return rows


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    all_results = []
    for ckpt in CKPTS:
        cache_file = CACHE_DIR / f"video_feats__{ckpt}__L11.npz"
        data = np.load(cache_file, allow_pickle=True)
        feats = data["features"]
        labels = data["labels"]
        sources = data["sources"]
        mask = sources == "PC_Generator"
        sub_feats = feats[mask]
        sub_labels = labels[mask]
        n_real = (sub_labels == 0).sum()
        n_fake = (sub_labels == 1).sum()
        logger.info("[%s] PC_Generator subset: n_real=%d n_fake=%d", ckpt, n_real, n_fake)
        if n_real < 5 or n_fake < 5:
            logger.warning("too few samples, skipping")
            continue
        # n_splits = min(5, n_real, n_fake)
        n_splits = min(5, int(n_real), int(n_fake))
        rows = run_probe(sub_feats, sub_labels, ckpt, n_splits=n_splits)
        all_results.extend(rows)
    df = pd.DataFrame(all_results)
    out_csv = OUT_DIR / "lockbox_probe_auc_pcgen_only.csv"
    df.to_csv(out_csv, index=False)
    logger.info("wrote %s", out_csv)
    print("\n=== PC_GENERATOR-ONLY SUMMARY ===")
    for ckpt, grp in df.groupby("ckpt"):
        aucs = grp["auc"].values
        print(f"{ckpt}: mean_AUC={aucs.mean():.4f} ± {aucs.std():.4f} (folds={len(aucs)})")
        print(f"  per_fold: {[f'{a:.4f}' for a in aucs]}")


if __name__ == "__main__":
    raise SystemExit(main())
