#!/usr/bin/env python3
"""Score-correlation-penalty prototype on frozen P8A features (CPU).

Tests whether `L = BCE + lambda * sum_axis |pearson_batch(score, axis)|` converges
on a linear head and drives |pearson| down without collapsing recall.

Penalty axes (continuous unless noted):
  - sharpness_laplacian   (npz iq_features, log1p)
  - luma_mean              (npz iq_features)
  - face_skin_frac         (npz iq_features; proxy for face_area_fraction since
                            parquet face_area_ratio has 0% coverage on viso suite)
  - is_webcam              (parquet clip_capture_mode == 'webcam'; binary;
                            limited coverage so penalty is masked-mean)

Lambda sweep: 0, 0.1, 1.0, 10.0, 100.0
Optimizer: Adam(lr=1e-3), batch=512, epochs=30, seed=737
Identity-disjoint train/test split 80/20.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
FROZEN = REPO / "analysis" / "job7_head_retrain_2026-05-04" / "frozen_features"
PARQUET_PATH = REPO / "analysis" / "lockbox_tagging" / "full_tags_2026-04-27.parquet"
OUT = REPO / "analysis" / "corr_penalty_prototype_2026-05-05"
OUT.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT / "run.log"

SEED = 737
LAMBDAS = [0.0, 0.1, 1.0, 10.0, 100.0]
EPOCHS = 30
BATCH_SIZE = 512
LR = 1e-3

# Train suites: reals + fake suites (mirrors job7 conventions).
TRAIN_REAL_SUITE = "teams_real_all_dev"
TRAIN_FAKE_SUITES = [
    "visomaster_enhanced_macro_dev",
    "deeplive_enhanced_dev",
    "teams_fake_all_dev",
]

logger = logging.getLogger("corr-penalty")


def setup_logging():
    fh = logging.FileHandler(LOG_PATH)
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s %(levelname)s :: %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.handlers = [fh, sh]
    logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Data loading.
# -----------------------------------------------------------------------------
def load_suite(name: str) -> Optional[Dict[str, np.ndarray]]:
    p = FROZEN / f"{name}_p8a_features.npz"
    if not p.exists():
        logger.warning("missing feature file %s", p)
        return None
    z = np.load(p, allow_pickle=True)
    return {k: np.asarray(z[k]) for k in z.files}


def build_dataset() -> Dict[str, np.ndarray]:
    """Combine real + fake suites, derive penalty axes, do identity-disjoint
    80/20 split."""
    real = load_suite(TRAIN_REAL_SUITE)
    fake_loaded = []
    for s in TRAIN_FAKE_SUITES:
        d = load_suite(s)
        if d is not None:
            fake_loaded.append((s, d))
    assert real is not None and len(fake_loaded) > 0

    iq_cols = list(real["iq_cols"])
    lap_idx = iq_cols.index("laplacian_var")
    luma_idx = iq_cols.index("luma_mean")
    skin_idx = iq_cols.index("skin_frac")

    def grab(d, key):
        return np.asarray(d[key])

    X_list = [real["features"].astype(np.float32)]
    iq_list = [real["iq_features"].astype(np.float32)]
    fp_list = [grab(real, "frame_paths")]
    ident_list = [grab(real, "identities")]
    label_list = [np.zeros(len(real["features"]), dtype=np.int32)]
    suite_list = [np.array([TRAIN_REAL_SUITE] * len(real["features"]))]

    for s, d in fake_loaded:
        X_list.append(d["features"].astype(np.float32))
        iq_list.append(d["iq_features"].astype(np.float32))
        fp_list.append(grab(d, "frame_paths"))
        ident_list.append(grab(d, "identities"))
        label_list.append(np.ones(len(d["features"]), dtype=np.int32))
        suite_list.append(np.array([s] * len(d["features"])))

    X = np.concatenate(X_list, axis=0)
    IQ = np.concatenate(iq_list, axis=0)
    fp = np.concatenate(fp_list, axis=0)
    ident = np.concatenate(ident_list, axis=0)
    y = np.concatenate(label_list, axis=0)
    suite = np.concatenate(suite_list, axis=0)
    logger.info("Total pool: n=%d (real=%d fake=%d) feat_dim=%d", len(y),
                int((y == 0).sum()), int((y == 1).sum()), X.shape[1])
    for s in [TRAIN_REAL_SUITE] + [s for s, _ in fake_loaded]:
        logger.info("  %s : %d", s, int((suite == s).sum()))

    # Penalty axes from npz (universal coverage).
    lap_raw = IQ[:, lap_idx]
    sharp_axis = np.log1p(np.maximum(lap_raw, 0.0)).astype(np.float32)
    luma_axis = IQ[:, luma_idx].astype(np.float32)
    face_axis = IQ[:, skin_idx].astype(np.float32)

    # is_webcam from parquet via gcs_uri join.
    df = pd.read_parquet(PARQUET_PATH)
    fp_to_mode = dict(zip(df["gcs_uri"].tolist(), df["clip_capture_mode"].tolist()))
    webcam_axis = np.full(len(fp), np.nan, dtype=np.float32)
    n_match = 0
    for i, p in enumerate(fp):
        m = fp_to_mode.get(str(p))
        if m is not None and isinstance(m, str):
            webcam_axis[i] = 1.0 if m == "webcam" else 0.0
            n_match += 1
    logger.info("is_webcam coverage: %d/%d (%.1f%%)", n_match, len(fp),
                n_match / len(fp) * 100.0)

    # Identity-disjoint 80/20 split.
    rng = np.random.default_rng(SEED)
    uniq_idents = np.array(sorted(set(ident.tolist())))
    rng.shuffle(uniq_idents)
    n_test_id = max(1, int(0.20 * len(uniq_idents)))
    test_idents = set(uniq_idents[:n_test_id].tolist())
    is_test = np.array([str(i) in test_idents for i in ident])
    is_train = ~is_test
    logger.info("Identity-disjoint split: %d train ids, %d test ids; rows train=%d test=%d",
                len(uniq_idents) - n_test_id, n_test_id,
                int(is_train.sum()), int(is_test.sum()))

    # Sanity: each split has both classes.
    for name, mask in [("train", is_train), ("test", is_test)]:
        n_pos = int((y[mask] == 1).sum())
        n_neg = int((y[mask] == 0).sum())
        logger.info("  %s: pos=%d neg=%d", name, n_pos, n_neg)

    return dict(
        X=X, y=y, ident=ident, suite=suite, fp=fp,
        sharp_axis=sharp_axis, luma_axis=luma_axis,
        face_axis=face_axis, webcam_axis=webcam_axis,
        is_train=is_train, is_test=is_test,
    )


# -----------------------------------------------------------------------------
# Pearson on a batch — differentiable.
# -----------------------------------------------------------------------------
def pearson_diff(x: torch.Tensor, y: torch.Tensor,
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Returns a scalar |pearson(x, y)| over the batch (masked if provided).

    If too few valid entries, returns 0 (no signal). Uses centered cosine.
    """
    if mask is not None:
        # mask shape (n,), bool.
        if mask.sum() < 3:
            return torch.tensor(0.0, dtype=x.dtype, device=x.device)
        x = x[mask]
        y = y[mask]
    if x.numel() < 3:
        return torch.tensor(0.0, dtype=x.dtype, device=x.device)
    xc = x - x.mean()
    yc = y - y.mean()
    num = (xc * yc).sum()
    den = xc.norm() * yc.norm() + 1e-8
    return torch.abs(num / den)


# -----------------------------------------------------------------------------
# Training loop.
# -----------------------------------------------------------------------------
def train_one(lam: float, data: Dict[str, np.ndarray],
              epochs: int = EPOCHS) -> Dict:
    """Trains a linear head with BCE + lam * sum_axis |Pearson(score, axis)|.
    Returns dict: 'model_logit_fn', 'history' (list of epoch dicts), 'final_metrics'.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    X = data["X"]; y = data["y"]
    is_train = data["is_train"]
    sharp = data["sharp_axis"]; luma = data["luma_axis"]
    face = data["face_axis"]; webcam = data["webcam_axis"]

    Xtr = X[is_train]; ytr = y[is_train]
    sharp_tr = sharp[is_train]; luma_tr = luma[is_train]
    face_tr = face[is_train]; webcam_tr = webcam[is_train]

    # Standardize features on train.
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr).astype(np.float32)

    # Standardize axes on train (so penalty magnitudes comparable; though the
    # |pearson| metric is scale-invariant, we standardize for numerical hygiene).
    def _z(a):
        m = float(np.nanmean(a)); s = float(np.nanstd(a) + 1e-8)
        return ((a - m) / s).astype(np.float32), m, s
    sharp_tr_z, sm, ss = _z(sharp_tr)
    luma_tr_z, lm, ls = _z(luma_tr)
    face_tr_z, fm, fs = _z(face_tr)
    # webcam is binary; standardize as is (it's already 0/1, mean-center it).
    webcam_valid = ~np.isnan(webcam_tr)
    webcam_tr_z = webcam_tr.copy()
    if webcam_valid.sum() > 1:
        wmean = float(webcam_tr[webcam_valid].mean())
        webcam_tr_z[webcam_valid] = webcam_tr[webcam_valid] - wmean

    # To tensors.
    Xt = torch.from_numpy(Xtr_s)
    yt = torch.from_numpy(ytr.astype(np.float32))
    sharp_t = torch.from_numpy(sharp_tr_z)
    luma_t = torch.from_numpy(luma_tr_z)
    face_t = torch.from_numpy(face_tr_z)
    webcam_t = torch.from_numpy(webcam_tr_z)
    webcam_mask_t = torch.from_numpy(webcam_valid)

    n, d = Xt.shape

    head = nn.Linear(d, 1)
    opt = optim.Adam(head.parameters(), lr=LR)
    bce_fn = nn.BCEWithLogitsLoss()

    rng = np.random.default_rng(SEED)
    history = []
    n_steps_per_epoch = max(1, n // BATCH_SIZE)
    diverged = False

    t0 = time.time()
    for epoch in range(epochs):
        perm = rng.permutation(n)
        bce_running = 0.0; pen_running = 0.0; total_running = 0.0
        head.train()
        for step in range(n_steps_per_epoch):
            idx = perm[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
            if len(idx) < 8:
                continue
            xb = Xt[idx]; yb = yt[idx]
            sharp_b = sharp_t[idx]; luma_b = luma_t[idx]
            face_b = face_t[idx]; web_b = webcam_t[idx]
            web_m = webcam_mask_t[idx]

            opt.zero_grad()
            logit = head(xb).squeeze(-1)
            score = torch.sigmoid(logit)
            loss_cls = bce_fn(logit, yb)
            # Verify on first iteration that gradient flows.
            if epoch == 0 and step == 0 and lam > 0:
                # touch graph by retaining for inspection.
                pass

            # Compute penalty.
            p_sharp = pearson_diff(score, sharp_b)
            p_luma = pearson_diff(score, luma_b)
            p_face = pearson_diff(score, face_b)
            p_web = pearson_diff(score, web_b, mask=web_m)
            penalty = p_sharp + p_luma + p_face + p_web

            loss = loss_cls + lam * penalty
            loss.backward()

            if not torch.isfinite(loss):
                logger.error("[lam=%.3g] non-finite loss at epoch %d step %d", lam, epoch, step)
                diverged = True
                break
            # Optional: gradient clip to keep things stable for large lam.
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=10.0)
            opt.step()

            bce_running += float(loss_cls.detach())
            pen_running += float(penalty.detach())
            total_running += float(loss.detach())

        if diverged:
            break

        n_b = max(1, n_steps_per_epoch)
        bce_avg = bce_running / n_b
        pen_avg = pen_running / n_b
        total_avg = total_running / n_b

        # Evaluate at epoch end on train (logging only).
        with torch.no_grad():
            head.eval()
            logit_all = head(Xt).squeeze(-1)
            score_all = torch.sigmoid(logit_all)
            try:
                auc_tr = float(roc_auc_score(ytr, score_all.numpy()))
            except Exception:
                auc_tr = float("nan")

        history.append({
            "lambda": lam, "epoch": epoch, "bce_loss": bce_avg,
            "penalty_loss": pen_avg, "total_loss": total_avg,
            "train_auc": auc_tr,
        })
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 5 == 0:
            logger.info("[lam=%.3g] ep %2d  bce=%.4f  pen=%.4f  tot=%.4f  train_auc=%.4f",
                        lam, epoch, bce_avg, pen_avg, total_avg, auc_tr)

    elapsed = time.time() - t0

    # Build a callable that predicts logits on raw X (uses scaler).
    head.eval()
    def predict(Xq: np.ndarray) -> np.ndarray:
        Xq_s = sc.transform(Xq).astype(np.float32)
        with torch.no_grad():
            logit = head(torch.from_numpy(Xq_s)).squeeze(-1).numpy()
        return 1.0 / (1.0 + np.exp(-logit))

    return dict(
        history=history, predict=predict, scaler=sc, head=head,
        diverged=diverged, elapsed=elapsed,
    )


# -----------------------------------------------------------------------------
# Test-set metrics, including post-training |pearson| measurements.
# -----------------------------------------------------------------------------
def evaluate(predict_fn, data: Dict[str, np.ndarray]) -> Dict:
    X = data["X"]; y = data["y"]
    is_test = data["is_test"]
    Xte = X[is_test]; yte = y[is_test]
    sharp_te = data["sharp_axis"][is_test]
    luma_te = data["luma_axis"][is_test]
    face_te = data["face_axis"][is_test]
    web_te = data["webcam_axis"][is_test]
    score_te = predict_fn(Xte)

    out: Dict = {}
    out["n_test"] = int(len(yte))
    out["n_test_pos"] = int((yte == 1).sum())
    out["n_test_neg"] = int((yte == 0).sum())
    try:
        out["auc"] = float(roc_auc_score(yte, score_te))
    except Exception:
        out["auc"] = float("nan")

    s_real = score_te[yte == 0]
    s_fake = score_te[yte == 1]
    for fpr in [0.05, 0.10]:
        if len(s_real) > 0:
            tau = float(np.quantile(s_real, 1.0 - fpr))
            rec = float((s_fake >= tau).mean()) if len(s_fake) else float("nan")
            tau_actual = float((s_real >= tau).mean())
        else:
            rec = float("nan"); tau = float("nan"); tau_actual = float("nan")
        out[f"tau_FPR={fpr:.2f}"] = tau
        out[f"recall@FPR={fpr:.2f}"] = rec
        out[f"FPR_actual@target={fpr:.2f}"] = tau_actual

    # Post-training |Pearson| per axis on TEST scores.
    def _pearson_np(a, b, mask=None):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if mask is not None:
            a = a[mask]; b = b[mask]
        if len(a) < 3:
            return float("nan")
        ac = a - a.mean(); bc = b - b.mean()
        den = np.linalg.norm(ac) * np.linalg.norm(bc) + 1e-8
        return float(abs((ac * bc).sum() / den))

    out["abs_r_sharp"] = _pearson_np(score_te, sharp_te)
    out["abs_r_luma"] = _pearson_np(score_te, luma_te)
    out["abs_r_face"] = _pearson_np(score_te, face_te)
    web_mask = ~np.isnan(web_te)
    out["abs_r_webcam"] = _pearson_np(score_te, web_te, mask=web_mask)

    # Per-axis recall variance: stratify recall by axis quartile (test reals
    # only used to set tau, but recall is computed on test fakes within each
    # quartile of the FAKE axis).
    def _recall_var(axis_vals, fakes_mask):
        a = axis_vals[fakes_mask]; s = score_te[fakes_mask]
        if len(a) == 0:
            return float("nan")
        if np.all(np.isnan(a)):
            return float("nan")
        a = np.where(np.isnan(a), np.nanmedian(a), a)
        qs = np.quantile(a, [0.25, 0.50, 0.75])
        recs = []
        tau10 = float(np.quantile(s_real, 0.90)) if len(s_real) else float("nan")
        for qi in range(4):
            lo = -np.inf if qi == 0 else qs[qi - 1]
            hi = +np.inf if qi == 3 else qs[qi]
            m = (a > lo) & (a <= hi)
            if m.sum() == 0:
                recs.append(np.nan)
            else:
                recs.append(float((s[m] >= tau10).mean()))
        recs = [r for r in recs if not np.isnan(r)]
        if len(recs) < 2:
            return float("nan")
        return float(max(recs) - min(recs))

    fakes_mask = (yte == 1)
    out["recall_var_sharp"] = _recall_var(sharp_te, fakes_mask)
    out["recall_var_luma"] = _recall_var(luma_te, fakes_mask)
    out["recall_var_face"] = _recall_var(face_te, fakes_mask)
    web_te_filled = np.where(np.isnan(web_te), 0.0, web_te)
    out["recall_var_webcam"] = _recall_var(web_te_filled, fakes_mask & web_mask)
    return out


# -----------------------------------------------------------------------------
def main() -> int:
    setup_logging()
    logger.info("Building dataset")
    data = build_dataset()

    # Sanity: gradient flow at lam>0 (manual check on a tiny subset).
    logger.info("Sanity: verifying gradient flows through penalty term")
    torch.manual_seed(SEED)
    head = nn.Linear(data["X"].shape[1], 1)
    sc = StandardScaler().fit(data["X"][data["is_train"]][:500])
    xs = torch.from_numpy(sc.transform(data["X"][data["is_train"]][:64]).astype(np.float32))
    ys = torch.from_numpy(data["y"][data["is_train"]][:64].astype(np.float32))
    sharp_b = torch.from_numpy(data["sharp_axis"][data["is_train"]][:64].astype(np.float32))
    sharp_b = (sharp_b - sharp_b.mean()) / (sharp_b.std() + 1e-8)
    logit = head(xs).squeeze(-1)
    score = torch.sigmoid(logit)
    loss = nn.functional.binary_cross_entropy_with_logits(logit, ys) + 1.0 * pearson_diff(score, sharp_b)
    loss.backward()
    grad_norm = float(head.weight.grad.norm())
    logger.info("  loss=%.4f  weight_grad_norm=%.4f", float(loss), grad_norm)
    if not (grad_norm > 0):
        logger.error("FATAL: gradient norm is zero through penalty path")
        return 1

    sweep_rows = []
    curve_rows = []
    for lam in LAMBDAS:
        logger.info("=" * 70)
        logger.info("lambda = %.3g", lam)
        result = train_one(lam, data, epochs=EPOCHS)
        for h in result["history"]:
            curve_rows.append(h)
        eval_metrics = evaluate(result["predict"], data)
        eval_metrics["lambda"] = lam
        eval_metrics["diverged"] = bool(result["diverged"])
        eval_metrics["elapsed_sec"] = result["elapsed"]
        sweep_rows.append(eval_metrics)
        logger.info("  AUC=%.4f rec@5=%.4f rec@10=%.4f |r_sharp|=%.4f |r_luma|=%.4f |r_face|=%.4f |r_webcam|=%.4f",
                    eval_metrics["auc"],
                    eval_metrics["recall@FPR=0.05"], eval_metrics["recall@FPR=0.10"],
                    eval_metrics["abs_r_sharp"], eval_metrics["abs_r_luma"],
                    eval_metrics["abs_r_face"], eval_metrics["abs_r_webcam"])

    # Write CSVs.
    sweep_df = pd.DataFrame(sweep_rows)
    cols = ["lambda", "auc", "recall@FPR=0.05", "recall@FPR=0.10",
            "abs_r_sharp", "abs_r_luma", "abs_r_face", "abs_r_webcam",
            "recall_var_sharp", "recall_var_luma", "recall_var_face", "recall_var_webcam",
            "tau_FPR=0.05", "tau_FPR=0.10",
            "FPR_actual@target=0.05", "FPR_actual@target=0.10",
            "n_test", "n_test_pos", "n_test_neg", "diverged", "elapsed_sec"]
    sweep_df = sweep_df[cols]
    sweep_path = OUT / "lambda_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    logger.info("Wrote %s", sweep_path)

    curve_df = pd.DataFrame(curve_rows)
    curve_path = OUT / "loss_curves.csv"
    curve_df.to_csv(curve_path, index=False)
    logger.info("Wrote %s", curve_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
