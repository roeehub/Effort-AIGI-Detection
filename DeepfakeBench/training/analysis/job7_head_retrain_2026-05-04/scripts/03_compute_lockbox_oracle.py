#!/usr/bin/env python3
"""Job 7 Stage 3 — compute lockbox-self-calibrated and joint-calibrated
recall numbers for each retrained head, plus head score-distribution stats
across substrates. Outputs lockbox_oracle_summary.csv and
score_distribution_per_head_substrate.csv.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
HERE = REPO / "analysis" / "job7_head_retrain_2026-05-04"
F = HERE / "frozen_features"
OUT = HERE / "outputs"
SEED = 737

CHRONIC = {"bla_bla_chow", "bla_bla_chow__s2", "PC_Generator__s22",
           "PC_Generator__s45", "Roy_D", "Q__s6"}


def load_npz(p):
    z = np.load(p, allow_pickle=True)
    return {k: np.asarray(z[k]) for k in z.files}


def fit_lr(X, y, sample_weight=None):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, n_jobs=1, solver="lbfgs",
                             random_state=SEED).fit(sc.transform(X), y, sample_weight=sample_weight)
    return sc, clf


def predict_lr(sc, clf, X):
    return clf.predict_proba(sc.transform(X))[:, 1]


class IQResid:
    def __init__(self):
        self.beta = None

    def fit(self, X, IQ):
        IQ_aug = np.concatenate([IQ, np.ones((IQ.shape[0], 1), dtype=np.float32)], axis=1)
        beta, *_ = np.linalg.lstsq(IQ_aug.astype(np.float64), X.astype(np.float64), rcond=None)
        self.beta = beta.astype(np.float32)
        return self

    def transform(self, X, IQ):
        IQ_aug = np.concatenate([IQ, np.ones((IQ.shape[0], 1), dtype=np.float32)], axis=1)
        return X - IQ_aug @ self.beta


class AdvHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        return self.head(x).squeeze(-1)


def fit_adv_head(X, y, IQ, n_steps=1500, lr=1e-3, lam=1.0):
    torch.manual_seed(SEED); np.random.seed(SEED)
    sc = StandardScaler().fit(X); iq_sc = StandardScaler().fit(IQ)
    Xs = sc.transform(X).astype(np.float32); IQs = iq_sc.transform(IQ).astype(np.float32)
    n, d = Xs.shape; n_iq = IQs.shape[1]
    head = AdvHead(d)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(Xs); yt = torch.from_numpy(y.astype(np.float32))
    iqt = torch.from_numpy(IQs)
    bs = 256; rng = np.random.default_rng(SEED)
    for step in range(n_steps):
        idx = rng.integers(0, n, size=bs)
        xb = Xt[idx]; yb = yt[idx]; iqb = iqt[idx]
        opt.zero_grad()
        logit = head(xb)
        loss_cls = bce(logit, yb)
        logit_centered = logit - logit.mean()
        iqb_centered = iqb - iqb.mean(0, keepdim=True)
        std_l = logit_centered.std() + 1e-6
        std_iq = iqb_centered.std(0) + 1e-6
        corr = (logit_centered.unsqueeze(1) * iqb_centered).mean(0) / (std_l * std_iq)
        loss_iq = (corr ** 2).mean()
        (loss_cls + lam * loss_iq).backward()
        opt.step()
    return sc, head


def predict_adv(sc, head, X):
    head.eval()
    with torch.no_grad():
        logit = head(torch.from_numpy(sc.transform(X).astype(np.float32))).numpy()
    return 1.0 / (1.0 + np.exp(-logit))


def main():
    real_dev = load_npz(F / "teams_real_all_dev_p8a_features.npz")
    viso = load_npz(F / "visomaster_enhanced_macro_dev_p8a_features.npz")
    deeplive = load_npz(F / "deeplive_enhanced_dev_p8a_features.npz")
    fake_dev = load_npz(F / "teams_fake_all_dev_p8a_features.npz")
    real_lock = load_npz(F / "teams_real_all_lockbox_p8a_features.npz")
    fake_lock = load_npz(F / "teams_fake_all_lockbox_p8a_features.npz")
    v2 = load_npz(F / "visomaster_enhanced_v2_p8a_features.npz")

    X = np.concatenate([real_dev["features"], viso["features"], deeplive["features"], fake_dev["features"]], axis=0).astype(np.float32)
    IQ = np.concatenate([real_dev["iq_features"], viso["iq_features"], deeplive["iq_features"], fake_dev["iq_features"]], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(len(real_dev["labels"]), int),
                        np.ones(len(viso["labels"]), int),
                        np.ones(len(deeplive["labels"]), int),
                        np.ones(len(fake_dev["labels"]), int)])
    ident_train = np.concatenate([real_dev["identities"], viso["identities"], deeplive["identities"], fake_dev["identities"]])

    # Sample weights for H3.
    w = np.ones(len(y), dtype=np.float32)
    chronic_mask = np.isin(ident_train, list(CHRONIC)) & (y == 0)
    w[chronic_mask] = 3.0

    # H0
    sc0, clf0 = fit_lr(X, y)
    # H1
    resid = IQResid().fit(X, IQ)
    sc1, clf1 = fit_lr(resid.transform(X, IQ), y)
    # H2
    sc2, head2 = fit_adv_head(X, y, IQ, n_steps=1500, lam=1.0)
    # H3
    sc3, clf3 = fit_lr(X, y, sample_weight=w)
    # H4
    sc4 = StandardScaler().fit(X)
    mlp = MLPClassifier(hidden_layer_sizes=(256,), activation="relu", max_iter=200,
                        batch_size=256, random_state=SEED, alpha=1e-4).fit(sc4.transform(X), y)
    # H5
    xgbc = xgb.XGBClassifier(max_depth=4, n_estimators=200, n_jobs=1, random_state=SEED,
                              eval_metric="logloss", tree_method="hist").fit(X, y)

    heads = ["H0_LR", "H1_LR_IQ_resid", "H2_LR_adv", "H3_LR_chronic_3x", "H4_MLP", "H5_XGB"]

    def predict_for(head, Xq, IQq):
        if head == "H0_LR":
            return predict_lr(sc0, clf0, Xq)
        if head == "H1_LR_IQ_resid":
            return predict_lr(sc1, clf1, resid.transform(Xq, IQq))
        if head == "H2_LR_adv":
            return predict_adv(sc2, head2, Xq)
        if head == "H3_LR_chronic_3x":
            return predict_lr(sc3, clf3, Xq)
        if head == "H4_MLP":
            return mlp.predict_proba(sc4.transform(Xq))[:, 1]
        if head == "H5_XGB":
            return xgbc.predict_proba(Xq)[:, 1]
        raise KeyError(head)

    # 1) Score-distribution-per-head-per-substrate.
    rows_dist = []
    substrates = {
        "real_dev": (real_dev, 0),
        "real_lockbox": (real_lock, 0),
        "fake_lockbox": (fake_lock, 1),
        "viso_macro_dev_FAKE": (viso, 1),
        "deeplive_FAKE": (deeplive, 1),
        "teams_fake_dev_FAKE": (fake_dev, 1),
        "v2_held_out_FAKE": (v2, 1),
    }
    rows_existing = []
    for h in heads + ["EXISTING_P8A_HEAD"]:
        for sub_name, (sd, lab) in substrates.items():
            if h == "EXISTING_P8A_HEAD":
                s = sd["scores_p8a_existing"]
            else:
                s = predict_for(h, sd["features"].astype(np.float32),
                                sd["iq_features"].astype(np.float32))
            rows_dist.append({
                "head": h, "substrate": sub_name, "label": lab, "n": len(s),
                "mean": float(s.mean()), "p10": float(np.quantile(s, 0.10)),
                "p50": float(np.median(s)), "p90": float(np.quantile(s, 0.90)),
                "p99": float(np.quantile(s, 0.99)),
            })
    pd.DataFrame(rows_dist).to_csv(OUT / "score_distribution_per_head_substrate.csv", index=False)
    print("wrote score_distribution_per_head_substrate.csv")

    # 2) Lockbox oracle (calibrate τ on lockbox reals) + joint calibration.
    rows_oracle = []
    for h in heads + ["EXISTING_P8A_HEAD"]:
        if h == "EXISTING_P8A_HEAD":
            s_real_dev = real_dev["scores_p8a_existing"]
            s_real_lock = real_lock["scores_p8a_existing"]
            s_fake_lock = fake_lock["scores_p8a_existing"]
            s_viso = viso["scores_p8a_existing"]
            s_deeplive = deeplive["scores_p8a_existing"]
            s_fake_dev = fake_dev["scores_p8a_existing"]
            # v2 has no existing P8A scores in feed (zeros); skip
            s_v2 = v2["scores_p8a_existing"]
        else:
            s_real_dev = predict_for(h, real_dev["features"].astype(np.float32), real_dev["iq_features"].astype(np.float32))
            s_real_lock = predict_for(h, real_lock["features"].astype(np.float32), real_lock["iq_features"].astype(np.float32))
            s_fake_lock = predict_for(h, fake_lock["features"].astype(np.float32), fake_lock["iq_features"].astype(np.float32))
            s_viso = predict_for(h, viso["features"].astype(np.float32), viso["iq_features"].astype(np.float32))
            s_deeplive = predict_for(h, deeplive["features"].astype(np.float32), deeplive["iq_features"].astype(np.float32))
            s_fake_dev = predict_for(h, fake_dev["features"].astype(np.float32), fake_dev["iq_features"].astype(np.float32))
            s_v2 = predict_for(h, v2["features"].astype(np.float32), v2["iq_features"].astype(np.float32))

        all_real = np.concatenate([s_real_dev, s_real_lock])
        for fpr in [0.02, 0.05, 0.10, 0.20]:
            tau_dev = float(np.quantile(s_real_dev, 1.0 - fpr))
            tau_lock = float(np.quantile(s_real_lock, 1.0 - fpr))
            tau_joint = float(np.quantile(all_real, 1.0 - fpr))
            for label, tau in [("dev", tau_dev), ("lockbox_oracle", tau_lock), ("joint_dev_lockbox", tau_joint)]:
                rows_oracle.append({
                    "head": h, "fpr_target": fpr,
                    "calib_source": label, "tau": tau,
                    "real_dev_fpr": float((s_real_dev >= tau).mean()),
                    "real_lockbox_fpr": float((s_real_lock >= tau).mean()),
                    "fake_lockbox_recall": float((s_fake_lock >= tau).mean()),
                    "viso_macro_dev_recall": float((s_viso >= tau).mean()),
                    "deeplive_recall": float((s_deeplive >= tau).mean()),
                    "teams_fake_dev_recall": float((s_fake_dev >= tau).mean()),
                    "v2_held_out_recall": float((s_v2 >= tau).mean()),
                })
    pd.DataFrame(rows_oracle).to_csv(OUT / "lockbox_oracle_calibration.csv", index=False)
    print("wrote lockbox_oracle_calibration.csv")

    # 3) Save per-head OOD score CSV with all suites for downstream FACTS:
    # we'll embed in score_distribution above. Done.


if __name__ == "__main__":
    raise SystemExit(main())
