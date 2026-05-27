#!/usr/bin/env python3
"""Job 7 — Stage 2 + 3: train candidate heads on frozen P8A features and
evaluate them at calibrated FPR thresholds. Pure CPU.

Heads:
  H0  baseline LogisticRegression(C=1.0)
  H1  LR with IQ-residualised features (regress 7-d IQ out of each feat dim)
  H2  LR with adversarial debiasing (PyTorch; minimise BCE, maximise IQ MSE)
  H3  LR with sample reweighting on chronic-FP-prone identities (3x)
  H4  Single-hidden-layer MLP (sklearn MLPClassifier)
  H5  XGBoost (depth=4, n_estimators=200, n_jobs=1)

5-fold StratifiedKFold trained on:
  reals  : teams_real_all_dev (4564)
  fakes  : visomaster_enhanced_macro_dev (550) + deeplive_enhanced_dev (545)
           + teams_fake_all_dev (3039 — if extracted)

Out-of-fold (OOF) fake scores are concatenated to compute per-suite recall at
calibrated FPR (calibrated on OOF reals only). For OUT suites (lockbox + v2),
the head is fit on the full train pool, then applied to the held-out suite
features. For lockbox FPR calibration we use teams_real_all_dev OOF.

Outputs (all under analysis/job7_head_retrain_2026-05-04/outputs/):
  head_objectives_summary.csv            per (head, fold) train/eval metrics
  per_suite_recall_summary.csv           head x suite x FPR_target -> recall
  v2_held_out_per_family.csv             head x v2 family -> recall, AUC, n
  per_identity_fpr_after_retrain.csv     head x identity -> FPR
  iq_quartile_lift.csv                   head x IQ_quartile -> recall (viso)
  subtype_stratified_recall.csv          head x subtype x FPR_target -> recall
  oof_scores_per_head.csv                per-frame OOF scores for diagnostics
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
HERE = REPO / "analysis" / "job7_head_retrain_2026-05-04"
FROZEN = HERE / "frozen_features"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
LOG_PATH = HERE / "run_train.log"

logger = logging.getLogger("job7-train")

IQ_COLS = ["luma_mean", "luma_p10", "luma_p90", "laplacian_var",
           "sobel_edge_mean", "saturation_mean", "skin_frac"]

# Train suites (reals + multiple fake suites for breadth).
TRAIN_REAL_SUITE = "teams_real_all_dev"
TRAIN_FAKE_SUITES = [
    "visomaster_enhanced_macro_dev",
    "deeplive_enhanced_dev",
    "teams_fake_all_dev",  # optional; skipped silently if missing.
]
HELD_OUT_SUITES = [
    "teams_real_all_lockbox",
    "teams_fake_all_lockbox",
    "visomaster_enhanced_v2",
]

CHRONIC_IDENTS = {"bla_bla_chow", "bla_bla_chow__s2", "PC_Generator__s22",
                  "PC_Generator__s45", "Roy_D", "Q__s6"}
FPR_TARGETS = [0.02, 0.05, 0.10, 0.20, 0.30]
SEED = 737


# -----------------------------------------------------------------------------
def setup_logging():
    fh = logging.FileHandler(LOG_PATH); sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s %(levelname)s :: %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.handlers = [fh, sh]; logger.setLevel(logging.INFO)


def load_suite(name: str) -> Optional[Dict[str, np.ndarray]]:
    p = FROZEN / f"{name}_p8a_features.npz"
    if not p.exists():
        logger.warning("missing feature file %s", p)
        return None
    z = np.load(p, allow_pickle=True)
    return {k: np.asarray(z[k]) for k in z.files}


# -----------------------------------------------------------------------------
# Recall @ FPR helpers.
# -----------------------------------------------------------------------------
def recall_at_fpr(scores_pos: np.ndarray, scores_neg: np.ndarray,
                  fpr_target: float) -> Tuple[float, float]:
    """Returns (recall, threshold). Threshold = upper-quantile of negatives at
    (1 - fpr_target) — closest to recall_at_fpr_max convention."""
    if len(scores_neg) == 0 or len(scores_pos) == 0:
        return float("nan"), float("nan")
    tau = float(np.quantile(scores_neg, 1.0 - fpr_target))
    recall = float((scores_pos >= tau).mean())
    return recall, tau


# -----------------------------------------------------------------------------
# Head implementations.
# -----------------------------------------------------------------------------
class IQResidualiser:
    """Train: regress IQ -> each feature dim via lstsq; produce residuals."""
    def __init__(self):
        self.beta = None  # (D, K+1) including intercept on K IQ dims
        self.k = None

    def fit(self, X: np.ndarray, IQ: np.ndarray):
        self.k = IQ.shape[1]
        IQ_aug = np.concatenate([IQ, np.ones((IQ.shape[0], 1), dtype=np.float32)], axis=1)
        # Solve in float64 for numerical stability.
        beta, *_ = np.linalg.lstsq(IQ_aug.astype(np.float64),
                                    X.astype(np.float64), rcond=None)
        self.beta = beta.astype(np.float32)  # (K+1, D)
        return self

    def transform(self, X: np.ndarray, IQ: np.ndarray) -> np.ndarray:
        IQ_aug = np.concatenate([IQ, np.ones((IQ.shape[0], 1), dtype=np.float32)], axis=1)
        pred = IQ_aug @ self.beta
        return X - pred


def fit_lr(Xtr, ytr, sample_weight=None, C=1.0):
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    clf = LogisticRegression(max_iter=2000, n_jobs=1, C=C, solver="lbfgs",
                              random_state=SEED)
    clf.fit(Xtr_s, ytr, sample_weight=sample_weight)
    return sc, clf


def predict_lr(sc, clf, X):
    return clf.predict_proba(sc.transform(X))[:, 1]


# -----------------------------------------------------------------------------
# H2 — adversarial debiasing.
# -----------------------------------------------------------------------------
class AdvHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.head(x).squeeze(-1)


class IQAdversary(nn.Module):
    def __init__(self, in_dim: int, n_iq: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, n_iq),
        )

    def forward(self, x):
        return self.net(x)


def fit_adv_head(Xtr: np.ndarray, ytr: np.ndarray, IQtr: np.ndarray,
                  n_steps: int = 1500, lam: float = 1.0, lr: float = 1e-3):
    """Train head + IQ adversary jointly. Returns (scaler, iq_scaler, head).

    Loss(head)  = BCE(label) - lam * MSE(IQ_pred, IQ_true)
    Loss(adv)   = MSE(IQ_pred, IQ_true)
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    sc = StandardScaler().fit(Xtr)
    iq_sc = StandardScaler().fit(IQtr)
    Xtr_s = sc.transform(Xtr).astype(np.float32)
    IQtr_s = iq_sc.transform(IQtr).astype(np.float32)
    n, d = Xtr_s.shape; n_iq = IQtr_s.shape[1]
    head = AdvHead(d); adv = IQAdversary(d, n_iq)
    opt_h = torch.optim.Adam(head.parameters(), lr=lr)
    opt_a = torch.optim.Adam(adv.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(Xtr_s); yt = torch.from_numpy(ytr.astype(np.float32))
    iqt = torch.from_numpy(IQtr_s)
    bs = 256
    rng = np.random.default_rng(SEED)
    for step in range(n_steps):
        idx = rng.integers(0, n, size=bs)
        xb = Xt[idx]; yb = yt[idx]; iqb = iqt[idx]
        # 1) Train adversary on current head input.
        opt_a.zero_grad()
        iq_pred = adv(xb.detach() if False else xb)
        # adversary trained directly on raw features (not head reps) since head
        # is linear projection; this approximates "predict IQ from input".
        loss_a = ((iq_pred - iqb) ** 2).mean()
        loss_a.backward()
        opt_a.step()
        # 2) Head: classify + simultaneously reduce IQ predictability.
        opt_h.zero_grad()
        logit = head(xb)
        loss_cls = bce(logit, yb)
        # Use frozen adv to compute IQ predictability of current logit-shaped
        # feature; simplest is to push the linear combination of features used
        # by head to be IQ-orthogonal: penalise correlation between logit and
        # each IQ dim.
        logit_centered = logit - logit.mean()
        iqb_centered = iqb - iqb.mean(0, keepdim=True)
        std_l = logit_centered.std() + 1e-6
        std_iq = iqb_centered.std(0) + 1e-6
        corr = (logit_centered.unsqueeze(1) * iqb_centered).mean(0) / (std_l * std_iq)
        loss_iq = (corr ** 2).mean()
        loss = loss_cls + lam * loss_iq
        loss.backward()
        opt_h.step()
        if step % 300 == 0:
            logger.info("    [adv] step %d cls=%.4f iqcorr=%.4f", step,
                        float(loss_cls), float(loss_iq))
    return sc, iq_sc, head


def predict_adv(sc, head, X):
    head.eval()
    with torch.no_grad():
        Xs = torch.from_numpy(sc.transform(X).astype(np.float32))
        logit = head(Xs).numpy()
    return 1.0 / (1.0 + np.exp(-logit))


# -----------------------------------------------------------------------------
# Train + evaluate everything.
# -----------------------------------------------------------------------------
def main() -> int:
    setup_logging()

    # Load all suites.
    suites: Dict[str, Dict[str, np.ndarray]] = {}
    for s in [TRAIN_REAL_SUITE] + TRAIN_FAKE_SUITES + HELD_OUT_SUITES:
        d = load_suite(s)
        if d is not None:
            suites[s] = d
            logger.info("loaded %s n=%d feat_dim=%d", s, len(d["labels"]), d["features"].shape[1])

    # Build train pool: reals + available fake suites.
    real = suites[TRAIN_REAL_SUITE]
    fake_suites_loaded = [s for s in TRAIN_FAKE_SUITES if s in suites]
    # Skip 0-dim arrays (e.g., 'iq_cols', 'suite' which are scalar metadata).
    fake_concat = {
        k: np.concatenate([suites[s][k] for s in fake_suites_loaded], axis=0)
        for k in real.keys() if k in suites[fake_suites_loaded[0]] and real[k].ndim >= 1
    }
    # Build origin-suite tag for each fake row.
    fake_origin = np.concatenate([
        np.array([s] * len(suites[s]["labels"])) for s in fake_suites_loaded
    ], axis=0)

    X_real = real["features"].astype(np.float32)
    IQ_real = real["iq_features"].astype(np.float32)
    X_fake = fake_concat["features"].astype(np.float32)
    IQ_fake = fake_concat["iq_features"].astype(np.float32)
    X = np.concatenate([X_real, X_fake], axis=0)
    IQ = np.concatenate([IQ_real, IQ_fake], axis=0)
    y = np.concatenate([np.zeros(len(X_real), dtype=np.int32),
                        np.ones(len(X_fake), dtype=np.int32)], axis=0)
    fp = np.concatenate([real["frame_paths"], fake_concat["frame_paths"]], axis=0)
    ident = np.concatenate([real["identities"], fake_concat["identities"]], axis=0)
    suite_origin = np.concatenate([
        np.array([TRAIN_REAL_SUITE] * len(X_real)), fake_origin,
    ], axis=0)
    logger.info("Train pool: n=%d (reals=%d, fakes=%d) feat=%d",
                len(y), len(X_real), len(X_fake), X.shape[1])
    logger.info("  fake breakdown: %s",
                {s: int((suite_origin == s).sum()) for s in fake_suites_loaded})

    # Sample weights for H3 (chronic identity reals upweighted 3x).
    w_h3 = np.ones(len(y), dtype=np.float32)
    chronic_mask = np.isin(ident, list(CHRONIC_IDENTS)) & (y == 0)
    w_h3[chronic_mask] = 3.0
    logger.info("H3 chronic-real upweighted rows: %d", int(chronic_mask.sum()))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))

    head_names = ["H0_LR", "H1_LR_IQ_resid", "H2_LR_adv", "H3_LR_chronic_3x",
                  "H4_MLP", "H5_XGB"]

    # OOF scores per head (length = len(y)).
    oof = {h: np.full(len(y), np.nan, dtype=np.float32) for h in head_names}

    # Final-fit (whole-pool) predictors per head, used on held-out suites.
    final_preds = {}

    # Per-fold metrics.
    rows_summary = []

    for fi, (tr, te) in enumerate(folds):
        logger.info("=" * 70)
        logger.info("Fold %d/%d  ntrain=%d  ntest=%d", fi + 1, 5, len(tr), len(te))

        # ===== H0 baseline LR =====
        sc, clf = fit_lr(X[tr], y[tr])
        s_te = predict_lr(sc, clf, X[te])
        oof["H0_LR"][te] = s_te
        rows_summary.append(_row("H0_LR", fi, y[te], s_te))

        # ===== H1 IQ-residualised =====
        resid = IQResidualiser().fit(X[tr], IQ[tr])
        Xtr_r = resid.transform(X[tr], IQ[tr]); Xte_r = resid.transform(X[te], IQ[te])
        sc1, clf1 = fit_lr(Xtr_r, y[tr])
        s_te = predict_lr(sc1, clf1, Xte_r)
        oof["H1_LR_IQ_resid"][te] = s_te
        rows_summary.append(_row("H1_LR_IQ_resid", fi, y[te], s_te))

        # ===== H2 adversarial =====
        sc2, iq_sc2, head2 = fit_adv_head(X[tr], y[tr], IQ[tr], n_steps=1500, lam=1.0)
        s_te = predict_adv(sc2, head2, X[te])
        oof["H2_LR_adv"][te] = s_te
        rows_summary.append(_row("H2_LR_adv", fi, y[te], s_te))

        # ===== H3 chronic upweight =====
        sc3, clf3 = fit_lr(X[tr], y[tr], sample_weight=w_h3[tr])
        s_te = predict_lr(sc3, clf3, X[te])
        oof["H3_LR_chronic_3x"][te] = s_te
        rows_summary.append(_row("H3_LR_chronic_3x", fi, y[te], s_te))

        # ===== H4 MLP =====
        sc4 = StandardScaler().fit(X[tr])
        mlp = MLPClassifier(hidden_layer_sizes=(256,), activation="relu",
                             alpha=1e-4, max_iter=200, random_state=SEED,
                             early_stopping=False, batch_size=256, verbose=False)
        mlp.fit(sc4.transform(X[tr]), y[tr])
        s_te = mlp.predict_proba(sc4.transform(X[te]))[:, 1]
        oof["H4_MLP"][te] = s_te
        rows_summary.append(_row("H4_MLP", fi, y[te], s_te))

        # ===== H5 XGBoost =====
        xgbc = xgb.XGBClassifier(max_depth=4, n_estimators=200, n_jobs=1,
                                  random_state=SEED, eval_metric="logloss",
                                  use_label_encoder=False, tree_method="hist")
        xgbc.fit(X[tr], y[tr])
        s_te = xgbc.predict_proba(X[te])[:, 1]
        oof["H5_XGB"][te] = s_te
        rows_summary.append(_row("H5_XGB", fi, y[te], s_te))

    pd.DataFrame(rows_summary).to_csv(OUT / "head_objectives_summary.csv", index=False)
    logger.info("Wrote head_objectives_summary.csv")

    # OOF scores frame (writeable).
    df_oof = pd.DataFrame({
        "frame_path": fp,
        "label": y,
        "identity": ident,
        "suite_origin": suite_origin,
    })
    for h in head_names:
        df_oof[h] = oof[h]
    df_oof.to_csv(OUT / "oof_scores_per_head.csv", index=False)
    logger.info("Wrote oof_scores_per_head.csv (%d rows)", len(df_oof))

    # Compute per-suite OOF recall@FPR (using teams_real_all_dev OOF reals
    # to calibrate τ, then apply to each fake suite).
    rows_recall = []
    rows_iq = []
    rows_subtype = []
    is_real = (y == 0)
    for h in head_names:
        s = oof[h]
        # Calibrate τ on reals (OOF).
        s_real = s[is_real]
        for fpr in FPR_TARGETS:
            tau = float(np.quantile(s_real, 1.0 - fpr))
            for fake_suite in fake_suites_loaded:
                m = (suite_origin == fake_suite) & (y == 1)
                rec = float((s[m] >= tau).mean()) if m.sum() else float("nan")
                rows_recall.append({"head": h, "suite": fake_suite,
                                    "fpr_target": fpr, "tau": tau,
                                    "recall": rec, "n": int(m.sum())})
            # Also reals on the same threshold (sanity).
            rows_recall.append({"head": h, "suite": TRAIN_REAL_SUITE,
                                 "fpr_target": fpr, "tau": tau,
                                 "recall": float((s[is_real] >= tau).mean()),
                                 "n": int(is_real.sum()),
                                 "_note": "FPR_actual"})
        # IQ quartile lift on viso fakes.
        viso_mask = (suite_origin == "visomaster_enhanced_macro_dev") & (y == 1)
        viso_iq = IQ[viso_mask, IQ_COLS.index("laplacian_var")]
        viso_s = s[viso_mask]
        if viso_mask.sum() > 0:
            qcuts = np.quantile(viso_iq, [0.25, 0.50, 0.75])
            tau_10 = float(np.quantile(s_real, 0.90))
            for q_idx in range(4):
                lo = -np.inf if q_idx == 0 else qcuts[q_idx - 1]
                hi = +np.inf if q_idx == 3 else qcuts[q_idx]
                m_q = (viso_iq > lo) & (viso_iq <= hi)
                rec = float((viso_s[m_q] >= tau_10).mean()) if m_q.sum() else float("nan")
                rows_iq.append({"head": h, "iq_quartile": q_idx + 1,
                                 "lap_var_lo": float(lo) if np.isfinite(lo) else None,
                                 "lap_var_hi": float(hi) if np.isfinite(hi) else None,
                                 "n": int(m_q.sum()), "recall_at_10pct": rec})
            # Subtype split (raw / teams).
            viso_fp = fp[viso_mask]
            for subtype in ["raw", "teams"]:
                tag = f"visomaster_enhanced_{subtype}__"
                sub_m = np.array([tag in str(p) for p in viso_fp])
                for fpr in FPR_TARGETS:
                    tau = float(np.quantile(s_real, 1.0 - fpr))
                    rec = float((viso_s[sub_m] >= tau).mean()) if sub_m.sum() else float("nan")
                    rows_subtype.append({"head": h, "subtype": subtype,
                                          "fpr_target": fpr,
                                          "n": int(sub_m.sum()),
                                          "recall": rec})

    pd.DataFrame(rows_recall).to_csv(OUT / "per_suite_recall_summary.csv", index=False)
    pd.DataFrame(rows_iq).to_csv(OUT / "iq_quartile_lift.csv", index=False)
    pd.DataFrame(rows_subtype).to_csv(OUT / "subtype_stratified_recall.csv", index=False)
    logger.info("Wrote per_suite_recall_summary.csv, iq_quartile_lift.csv, subtype_stratified_recall.csv")

    # Per-identity FPR (OOF, on teams_real_all_dev only) at FPR=0.10.
    rows_id = []
    for h in head_names:
        s = oof[h]
        s_real = s[is_real]
        tau_10 = float(np.quantile(s_real, 0.90))
        ident_real = ident[is_real]
        scores_real = s[is_real]
        df_id = pd.DataFrame({"ident": ident_real, "score": scores_real})
        agg = df_id.groupby("ident").agg(
            n=("score", "size"),
            n_fp=("score", lambda x: int((x >= tau_10).sum())),
            mean_score=("score", "mean"),
        ).reset_index()
        agg["fpr"] = agg["n_fp"] / agg["n"]
        agg["head"] = h
        rows_id.append(agg)
    pd.concat(rows_id, axis=0, ignore_index=True).to_csv(
        OUT / "per_identity_fpr_after_retrain.csv", index=False
    )
    logger.info("Wrote per_identity_fpr_after_retrain.csv")

    # Held-out suites: refit each head on the full pool, score lockbox + v2.
    logger.info("=" * 70)
    logger.info("Final fit on full train pool, then score held-out suites")
    out_held = []
    out_v2_fam = []
    # Recompute full-pool fits for each head.
    sc_full, clf_full = fit_lr(X, y)
    final_preds["H0_LR"] = lambda Xq, IQq, sc=sc_full, c=clf_full: predict_lr(sc, c, Xq)

    resid_full = IQResidualiser().fit(X, IQ)
    sc1_full, clf1_full = fit_lr(resid_full.transform(X, IQ), y)
    def pred_h1(Xq, IQq, sc=sc1_full, c=clf1_full, r=resid_full):
        return predict_lr(sc, c, r.transform(Xq, IQq))
    final_preds["H1_LR_IQ_resid"] = pred_h1

    sc2_full, iq_sc2_full, head2_full = fit_adv_head(X, y, IQ, n_steps=1500, lam=1.0)
    final_preds["H2_LR_adv"] = lambda Xq, IQq, sc=sc2_full, h=head2_full: predict_adv(sc, h, Xq)

    sc3_full, clf3_full = fit_lr(X, y, sample_weight=w_h3)
    final_preds["H3_LR_chronic_3x"] = lambda Xq, IQq, sc=sc3_full, c=clf3_full: predict_lr(sc, c, Xq)

    sc4_full = StandardScaler().fit(X)
    mlp_full = MLPClassifier(hidden_layer_sizes=(256,), activation="relu",
                              alpha=1e-4, max_iter=200, random_state=SEED,
                              batch_size=256, verbose=False)
    mlp_full.fit(sc4_full.transform(X), y)
    final_preds["H4_MLP"] = lambda Xq, IQq, sc=sc4_full, m=mlp_full: m.predict_proba(sc.transform(Xq))[:, 1]

    xgb_full = xgb.XGBClassifier(max_depth=4, n_estimators=200, n_jobs=1,
                                  random_state=SEED, eval_metric="logloss",
                                  use_label_encoder=False, tree_method="hist")
    xgb_full.fit(X, y)
    final_preds["H5_XGB"] = lambda Xq, IQq, m=xgb_full: m.predict_proba(Xq)[:, 1]

    # Pick calibration source = OOF reals on teams_real_all_dev, consistent
    # with above. Build the per-head τ map from oof[h][is_real].
    tau_map: Dict[str, Dict[float, float]] = {}
    for h in head_names:
        s_real_oof = oof[h][is_real]
        tau_map[h] = {fpr: float(np.quantile(s_real_oof, 1.0 - fpr)) for fpr in FPR_TARGETS}

    for suite in HELD_OUT_SUITES:
        if suite not in suites:
            logger.warning("held-out suite %s missing — skipping", suite)
            continue
        Sd = suites[suite]
        Xq = Sd["features"].astype(np.float32)
        IQq = Sd["iq_features"].astype(np.float32)
        yq = Sd["labels"].astype(np.int32)
        for h in head_names:
            s_q = final_preds[h](Xq, IQq)
            for fpr in FPR_TARGETS:
                tau = tau_map[h][fpr]
                if (yq == 1).sum():
                    rec = float((s_q[yq == 1] >= tau).mean())
                else:
                    rec = float("nan")
                if (yq == 0).sum():
                    fpr_actual = float((s_q[yq == 0] >= tau).mean())
                else:
                    fpr_actual = float("nan")
                out_held.append({"head": h, "suite": suite,
                                  "fpr_target": fpr, "tau": tau,
                                  "recall_pos": rec,
                                  "fpr_actual": fpr_actual,
                                  "n_pos": int((yq == 1).sum()),
                                  "n_neg": int((yq == 0).sum())})
            # v2 per-family.
            if suite == "visomaster_enhanced_v2":
                fam = Sd["family_keys"]
                for f in np.unique(fam):
                    fm = (fam == f)
                    fm_pos = fm & (yq == 1)
                    if fm_pos.sum() == 0:
                        continue
                    auc = roc_auc_score(yq[fm], s_q[fm]) if (len(np.unique(yq[fm])) > 1) else float("nan")
                    for fpr in FPR_TARGETS:
                        tau = tau_map[h][fpr]
                        rec = float((s_q[fm_pos] >= tau).mean())
                        out_v2_fam.append({"head": h, "v2_family": str(f),
                                            "fpr_target": fpr, "tau": tau,
                                            "n_frames": int(fm_pos.sum()),
                                            "recall": rec, "auc": auc})

    # Append held-out rows to per_suite_recall_summary.csv.
    df_held = pd.DataFrame(out_held)
    df_held.to_csv(OUT / "held_out_per_suite_recall.csv", index=False)
    pd.DataFrame(out_v2_fam).to_csv(OUT / "v2_held_out_per_family.csv", index=False)
    logger.info("Wrote held_out_per_suite_recall.csv and v2_held_out_per_family.csv")

    # Build a combined "per_suite_recall_summary.csv" that includes both OOF
    # train suites and held-out suites, plus the existing-head P8A baseline
    # for comparison.
    df_dev = pd.read_csv(OUT / "per_suite_recall_summary.csv")
    df_dev["fold"] = "OOF"
    rows_full = df_dev.to_dict("records")
    for r in out_held:
        rr = dict(r); rr["fold"] = "held_out"
        rr["recall"] = rr.pop("recall_pos")
        rows_full.append(rr)

    # Existing-head baseline rows (using scores_p8a_existing on every suite).
    for s in [TRAIN_REAL_SUITE] + fake_suites_loaded + HELD_OUT_SUITES:
        if s not in suites: continue
        ss = suites[s]
        sc_p = ss["scores_p8a_existing"]
        yy = ss["labels"]
        # Calibrate τ on teams_real_all_dev existing scores.
        tau_src = suites[TRAIN_REAL_SUITE]["scores_p8a_existing"]
        for fpr in FPR_TARGETS:
            tau = float(np.quantile(tau_src, 1.0 - fpr))
            if (yy == 1).sum():
                rec = float((sc_p[yy == 1] >= tau).mean())
            elif (yy == 0).sum():
                rec = float((sc_p[yy == 0] >= tau).mean())
            else:
                rec = float("nan")
            rows_full.append({
                "head": "EXISTING_P8A_HEAD",
                "suite": s,
                "fpr_target": fpr,
                "tau": tau,
                "recall": rec,
                "n": int(len(yy)),
                "fold": "held_out" if s in HELD_OUT_SUITES else "existing-baseline",
            })
    pd.DataFrame(rows_full).to_csv(OUT / "per_suite_recall_summary.csv", index=False)
    logger.info("Re-wrote per_suite_recall_summary.csv with held-out and baseline")

    return 0


def _row(head, fold, y_true, y_score):
    auc = roc_auc_score(y_true, y_score) if (len(np.unique(y_true)) > 1) else float("nan")
    rec_at = {}
    s_real = y_score[y_true == 0]; s_fake = y_score[y_true == 1]
    for fpr in FPR_TARGETS:
        if len(s_real):
            tau = float(np.quantile(s_real, 1.0 - fpr))
            rec_at[f"recall@FPR={fpr:.2f}"] = float((s_fake >= tau).mean()) if len(s_fake) else float("nan")
        else:
            rec_at[f"recall@FPR={fpr:.2f}"] = float("nan")
    return {"head": head, "fold": fold, "n_test": len(y_true),
            "n_test_pos": int((y_true == 1).sum()),
            "n_test_neg": int((y_true == 0).sum()),
            "auc": auc, **rec_at}


if __name__ == "__main__":
    raise SystemExit(main())
