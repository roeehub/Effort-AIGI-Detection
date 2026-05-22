"""Probe 1 — KLIEP re-fit sanity probe on trained-encoder L11 features.

For each of the 3 ckpts (P8A_step5000, SlotAv2_step3500, T5C_step3500):
  1) Load cached L11 features (clean + teams).
  2) Fit substrate logistic regression (y=0 clean, y=1 teams), 80/20 split,
     C=1.0, max_iter=2000, lbfgs, n_jobs=1, class_weight='balanced',
     mirroring run_d10.py:fit_kliep_discriminator.
  3) Save unit-norm coefficient as `_trained_encoder_substrate_axis_{ckpt}.npy`.
  4) Compute:
       - held-out 20% accuracy of the per-ckpt classifier
       - cosine(per-ckpt axis, frozen-CLIP KLIEP axis)
       - mean & std of pair-direction projection on the per-ckpt axis
  5) Write `_probe1_kliep_refit_results.json` with the aggregate schema.

Reuses cached features only. NO model load, NO GCS calls.

Usage:
    python run_probe1_kliep_refit.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

OUT_DIR = Path(__file__).resolve().parent
CKPT_KEYS: List[str] = ["P8A_step5000", "SlotAv2_step3500", "T5C_step3500"]

SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(OUT_DIR / "_probe1.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("probe1")


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n


def fit_substrate_axis(
    feats_clean: np.ndarray, feats_teams: np.ndarray
) -> Dict:
    """Fit y={0 clean, 1 teams} logistic regression, 80/20 split.

    Mirrors `run_d10.py:fit_kliep_discriminator` (C=1.0, max_iter=2000,
    solver=lbfgs, n_jobs=1, class_weight=balanced).

    Returns dict with:
      w_hat  (768,)  unit-norm coefficient vector
      acc_train, acc_test
      n_train, n_test
      coef_norm
      b
    """
    X_all = np.concatenate([feats_clean, feats_teams], axis=0).astype(np.float64)
    y_all = np.concatenate(
        [np.zeros(len(feats_clean), dtype=np.int64),
         np.ones(len(feats_teams), dtype=np.int64)]
    )
    X_all = l2_normalize_rows(X_all)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all,
    )

    clf = LogisticRegression(
        C=1.0, max_iter=2000, solver="lbfgs", n_jobs=1, class_weight="balanced",
    )
    clf.fit(X_tr, y_tr)
    w = clf.coef_[0].astype(np.float64)
    b = float(clf.intercept_[0])
    coef_norm = float(np.linalg.norm(w))
    w_hat = w / (coef_norm + 1e-12)

    acc_train = float(clf.score(X_tr, y_tr))
    acc_test = float(clf.score(X_te, y_te))

    return {
        "w_hat": w_hat,
        "b": b,
        "coef_norm": coef_norm,
        "acc_train": acc_train,
        "acc_test": acc_test,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
    }


def main() -> int:
    t0_total = time.time()
    feats_dir = OUT_DIR / "feats"
    frozen_axis_path = OUT_DIR / "_kliep_w_hat.npy"
    assert frozen_axis_path.exists(), f"missing frozen KLIEP axis: {frozen_axis_path}"
    w_frozen = np.load(frozen_axis_path).astype(np.float64)
    # already unit-norm but enforce
    w_frozen = w_frozen / (np.linalg.norm(w_frozen) + 1e-12)
    logger.info("frozen KLIEP axis: shape=%s norm=%.6f", w_frozen.shape, float(np.linalg.norm(w_frozen)))

    # Load matched-pair meta to align (clean, teams) by pair_id
    meta_clean = pd.read_parquet(OUT_DIR / "_cache_frames_meta_clean.parquet")
    meta_teams = pd.read_parquet(OUT_DIR / "_cache_frames_meta_teams.parquet")
    logger.info("meta: clean=%d teams=%d", len(meta_clean), len(meta_teams))

    # For pair-direction projection, take frame_idx=0 within each pair_id
    clean_groups = meta_clean.groupby("pair_id").indices
    teams_groups = meta_teams.groupby("pair_id").indices
    common = sorted(set(clean_groups.keys()) & set(teams_groups.keys()))

    # Pre-compute the per-pair row indices into the *clean/teams feature arrays*
    clean_idx = []
    teams_idx = []
    for pid in common:
        ci = clean_groups[pid]
        ti = teams_groups[pid]
        c0 = ci[np.argmin(meta_clean.iloc[ci]["frame_idx"].to_numpy())]
        t0 = ti[np.argmin(meta_teams.iloc[ti]["frame_idx"].to_numpy())]
        clean_idx.append(c0)
        teams_idx.append(t0)
    clean_idx = np.asarray(clean_idx, dtype=np.int64)
    teams_idx = np.asarray(teams_idx, dtype=np.int64)
    n_pairs = len(clean_idx)
    logger.info("matched pairs: %d", n_pairs)

    results = {
        "frozen_clip_kliep": {
            "acc": 0.9908864954432477,  # from RESULTS_FACTS doc + _gate_verdict.json
            "source": "_kliep_w_hat.npy",
        },
        "ckpt_axes": {},
    }

    for ckpt_key in CKPT_KEYS:
        t0_ckpt = time.time()
        clean_path = feats_dir / f"{ckpt_key}_L11_clean.npy"
        teams_path = feats_dir / f"{ckpt_key}_L11_teams.npy"
        assert clean_path.exists(), f"missing: {clean_path}"
        assert teams_path.exists(), f"missing: {teams_path}"

        Fc = np.load(clean_path)
        Ft = np.load(teams_path)
        logger.info(
            "[%s] L11 features: clean=%s teams=%s",
            ckpt_key, Fc.shape, Ft.shape,
        )

        fit = fit_substrate_axis(Fc, Ft)
        w_hat = fit["w_hat"]

        # Save axis
        out_axis = OUT_DIR / f"_trained_encoder_substrate_axis_{ckpt_key}.npy"
        np.save(out_axis, w_hat)
        logger.info("[%s] axis saved -> %s", ckpt_key, out_axis)

        # Cosine with frozen-CLIP axis
        cos_with_frozen = float(np.dot(w_hat, w_frozen))

        # Per-pair direction projection on the per-ckpt axis
        Fc_n = l2_normalize_rows(Fc.astype(np.float64))
        Ft_n = l2_normalize_rows(Ft.astype(np.float64))
        diff = Ft_n[teams_idx] - Fc_n[clean_idx]
        proj_per_ckpt = diff @ w_hat
        proj_frozen = diff @ w_frozen

        pair_proj_mean = float(np.mean(proj_per_ckpt))
        pair_proj_std = float(np.std(proj_per_ckpt, ddof=1))
        pair_proj_abs_mean = float(np.mean(np.abs(proj_per_ckpt)))
        pair_proj_frozen_mean = float(np.mean(proj_frozen))
        pair_proj_frozen_std = float(np.std(proj_frozen, ddof=1))

        results["ckpt_axes"][ckpt_key] = {
            "acc": fit["acc_test"],
            "acc_train": fit["acc_train"],
            "n_train": fit["n_train"],
            "n_test": fit["n_test"],
            "coef_norm": fit["coef_norm"],
            "b": fit["b"],
            "cos_with_frozen": cos_with_frozen,
            "pair_proj_mean": pair_proj_mean,
            "pair_proj_std": pair_proj_std,
            "pair_proj_abs_mean": pair_proj_abs_mean,
            "pair_proj_frozen_mean": pair_proj_frozen_mean,
            "pair_proj_frozen_std": pair_proj_frozen_std,
            "n_pairs": int(n_pairs),
        }

        logger.info(
            "[%s] acc_train=%.4f acc_test=%.4f cos_with_frozen=%+.4f "
            "pair_proj(trained)=%.4f±%.4f pair_proj(frozen)=%.4f±%.4f (%.1fs)",
            ckpt_key, fit["acc_train"], fit["acc_test"], cos_with_frozen,
            pair_proj_mean, pair_proj_std,
            pair_proj_frozen_mean, pair_proj_frozen_std,
            time.time() - t0_ckpt,
        )

    results["meta"] = {
        "seed": SEED,
        "n_pairs": int(n_pairs),
        "wall_seconds": round(time.time() - t0_total, 1),
        "split": "stratified 80/20 random_state=42",
    }

    out_path = OUT_DIR / "_probe1_kliep_refit_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("wrote %s (wall=%.1fs)", out_path, time.time() - t0_total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
