#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video-level fusion & evaluation for specialist models from per-frame CSVs.

Usage (4 models):
  python run_video_level_fusion.py \
    --csvs /path/run1.csv /path/run2.csv /path/run3.csv /path/run4.csv \
    --names "ModelA" "ModelB" "ModelC" "ModelD" \
    --outdir ./out_4_models

Usage (bottleneck & key method analysis):
  python run_video_level_fusion.py \
    --csvs /path/run1.csv /path/run2.csv /path/run3.csv \
    --names "FaceSwap" "Reenact" "EFS" \
    --key-methods "dfdc_fake" "deep_live_cam_fake" \
    --outdir ./out_with_analysis

Requires: pandas, numpy, scikit-learn, tqdm, matplotlib, scipy, pyarrow
"""
import os
import re
import math
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_fscore_support, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, logit
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


# ---------------------------
# Utils & parsing
# ---------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_outdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def extract_clip_key(frame_path: str) -> str:
    """Return the directory of the frame (without the frame filename), e.g. gs://.../FaceShifter/629_618"""
    # Assume '/' separator in GCS path
    # We want one level up from frame file: .../<method>/<clip_id>/<frame.png> -> .../<method>/<clip_id>
    if not isinstance(frame_path, str):
        return ""
    frame_path = frame_path.rstrip("/")
    return "/".join(frame_path.split("/")[:-1])


def extract_frame_index(frame_path: str) -> int:
    """Try to extract a numeric frame index from the filename; fallback to 0 if not parseable."""
    try:
        fname = frame_path.split("/")[-1]
        num = re.sub(r"\D+", "", os.path.splitext(fname)[0])
        return int(num) if num != "" else 0
    except Exception:
        return 0


def safe_entropy(pvals: np.ndarray, bins: int = 20) -> float:
    """Shannon entropy (approx via histogram) on probabilities [0,1]."""
    if pvals.size == 0:
        return 0.0
    hist, _ = np.histogram(pvals, bins=bins, range=(0.0, 1.0), density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    probs = hist / np.sum(hist)
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def longest_streak_above(arr: np.ndarray, thresh: float) -> int:
    if arr.size == 0:
        return 0
    streak = best = 0
    for v in arr:
        if v > thresh:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def softmax_pool(x: np.ndarray, beta: float) -> float:
    if x.size == 0:
        return 0.0
    # stabilized log-sum-exp
    m = np.max(beta * x)
    return float((np.log(np.mean(np.exp(beta * x - m))) + m) / beta)


def topk_mean(x: np.ndarray, k: int) -> float:
    if x.size == 0:
        return 0.0
    k = max(1, min(k, x.size))
    idx = np.argpartition(x, -k)[-k:]
    return float(np.mean(x[idx]))


def trimmed_mean(x: np.ndarray, trim: float = 0.1) -> float:
    if x.size == 0:
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    t = int(math.floor(trim * n))
    if 2 * t >= n:
        return float(np.mean(x_sorted))
    return float(np.mean(x_sorted[t: n - t]))


def macro_auc_by_method(y_true: np.ndarray, y_score: np.ndarray, methods: np.ndarray) -> Tuple[float, Dict[str, float]]:
    by_m = {}
    for m in np.unique(methods):
        mask = (methods == m)
        y_m = y_true[mask]
        s_m = y_score[mask]
        # need both classes
        if len(np.unique(y_m)) < 2:
            continue
        try:
            by_m[m] = roc_auc_score(y_m, s_m)
        except Exception:
            pass
    if len(by_m) == 0:
        return float('nan'), by_m
    return float(np.mean(list(by_m.values()))), by_m


def find_threshold_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_target: float) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_score)
    ok = np.where(fpr <= fpr_target)[0]
    if ok.size == 0:
        return float(thr[0])
    thr_sel = float(thr[ok[-1]])
    if not np.isfinite(thr_sel):
        thr_sel = float(np.nextafter(1.0, 0.0))
    return thr_sel


def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


def eer_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float(thr[idx])


# ---------------------------
# Loading & aggregation
# ---------------------------
def load_frame_csv(path: str, model_name: str) -> pd.DataFrame:
    usecols = ["method", "label", "frame_path", "frame_prob"]
    dtypes = {"method": "string", "label": "int8", "frame_path": "string", "frame_prob": "float32"}
    logging.info(f"Reading {model_name} from: {path}")
    df = pd.read_csv(path, usecols=usecols, dtype=dtypes)
    df["clip_key"] = df["frame_path"].map(extract_clip_key)
    df["frame_idx"] = df["frame_path"].map(extract_frame_index).astype("int32")
    df["model"] = model_name
    return df


def sample_clip_keys_intersection(dfs: List[pd.DataFrame], per_method: int, seed: int) -> List[str]:
    keys_sets = [set(df["clip_key"].unique()) for df in dfs]
    clip_common = list(set.intersection(*keys_sets))
    ref = dfs[0][["clip_key", "method"]].drop_duplicates()
    ref = ref[ref["clip_key"].isin(clip_common)]
    rng = np.random.default_rng(seed)
    sampled = []
    for m in ref["method"].unique():
        cands = ref.loc[ref["method"] == m, "clip_key"].values
        if cands.size == 0:
            continue
        take = min(per_method, cands.size)
        sampled.extend(list(rng.choice(cands, size=take, replace=False)))
    return sampled


def aggregate_per_video(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    gb = df.sort_values(["clip_key", "frame_idx"]).groupby("clip_key", sort=False)
    for clip, g in tqdm(
            gb,
            total=gb.ngroups,
            desc=f"Aggregating videos ({gb.ngroups} clips)"
    ):
        label_set = g["label"].unique()
        method_set = g["method"].unique()
        if len(label_set) != 1 or len(method_set) != 1:
            logging.warning(f"Inconsistent labels/methods within clip {clip}: labels={label_set}, methods={method_set}")
        label = int(label_set[0])
        method = str(method_set[0])
        probs = g["frame_prob"].to_numpy(dtype=np.float64)
        aggs = {
            "mean": float(np.mean(probs)) if probs.size else 0.0,
            "max": float(np.max(probs)) if probs.size else 0.0,
            "median": float(np.median(probs)) if probs.size else 0.0,
            "softmax_b5": softmax_pool(probs, 5.0),
            "softmax_b10": softmax_pool(probs, 10.0),
            "softmax_b20": softmax_pool(probs, 20.0),
            "topk1": topk_mean(probs, 1),
            "topk2": topk_mean(probs, 2),
            "topk4": topk_mean(probs, 4),
            "tmean10": trimmed_mean(probs, 0.10),
        }
        q90 = float(np.quantile(probs, 0.90)) if probs.size else 0.0
        std = float(np.std(probs)) if probs.size else 0.0
        ent = safe_entropy(probs)
        frac_hi = float(np.mean(probs > 0.9)) if probs.size else 0.0
        frac_lo = float(np.mean(probs < 0.1)) if probs.size else 0.0
        streak = longest_streak_above(probs, 0.9)
        row = {
            "clip_key": clip,
            "label": label,
            "method": method,
            **aggs,
            "q90": q90,
            "std": std,
            "entropy": ent,
            "frac_above_0.9": frac_hi,
            "frac_below_0.1": frac_lo,
            "streak_hi_0.9": streak,
            "n_frames": int(len(g)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------
# Aggregator selection & calibration
# ---------------------------
CANDIDATE_AGGS = ["mean", "max", "median", "softmax_b5", "softmax_b10", "softmax_b20", "topk1", "topk2", "topk4",
                  "tmean10"]


def pick_best_aggregator(df: pd.DataFrame) -> str:
    y = df["label"].to_numpy()
    best_name, best_score = None, -1.0
    used_metric = "auc"

    for name in CANDIDATE_AGGS:
        if name not in df.columns:
            continue
        s = df[name].to_numpy()
        try:
            score = roc_auc_score(y, s)
            metric = "auc"
        except Exception:
            score = average_precision_score(y, s)
            metric = "ap"

        if not np.isnan(score) and score > best_score:
            best_score = score
            best_name = name
            used_metric = metric

    if best_name is None:
        best_name = "mean"
        logging.warning("No aggregator produced a valid metric; falling back to 'mean'.")

    logging.info(f"Selected aggregator: {best_name} (best {used_metric}={best_score:.4f})")
    return best_name


def calibrate_isotonic_oof(scores: np.ndarray, labels: np.ndarray, n_splits: int = 5, random_state: int = 42) -> Tuple[
    np.ndarray, List[IsotonicRegression]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    probs = np.zeros_like(scores, dtype=float)
    calibrators = []
    for tr, va in skf.split(scores, labels):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(scores[tr], labels[tr])
        probs[va] = ir.transform(scores[va])
        calibrators.append(ir)
    return probs, calibrators


# ---------------------------
# Fusion
# ---------------------------
def fusion_noisyor(p_mat: np.ndarray) -> np.ndarray:
    return 1.0 - np.prod(1.0 - p_mat, axis=1)


def fusion_sum_logits(p_mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    logits = np.log(np.clip(p_mat, eps, 1 - eps) / np.clip(1 - p_mat, eps, 1 - eps))
    s = np.sum(logits, axis=1)
    return expit(s)


def stacked_logistic_oof(X: np.ndarray, y: np.ndarray, nonneg: bool = True, n_splits: int = 5,
                         random_state: int = 42) -> Tuple[np.ndarray, Dict]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(X.shape[0], dtype=float)
    weights = []
    for tr, va in skf.split(X, y):
        Xtr, ytr = X[tr], y[tr]
        Xva = X[va]

        if not nonneg:
            lr = LogisticRegression(max_iter=1000, solver="lbfgs")
            lr.fit(Xtr, ytr)
            oof[va] = lr.predict_proba(Xva)[:, 1]
            weights.append({"coef": lr.coef_.ravel().tolist(), "intercept": float(lr.intercept_[0])})
        else:
            nfeat = Xtr.shape[1]

            def neg_logloss(params):
                w = params[:nfeat]
                b = params[-1]
                z = Xtr @ w + b
                p = expit(z)
                p = np.clip(p, 1e-6, 1 - 1e-6)
                return -float(np.mean(ytr * np.log(p) + (1 - ytr) * np.log(1 - p)))

            x0 = np.zeros(nfeat + 1, dtype=float)
            bounds = [(0.0, None)] * nfeat + [(None, None)]
            res = minimize(neg_logloss, x0=x0, method="L-BFGS-B", bounds=bounds)
            w = res.x[:nfeat]
            b = res.x[-1]
            oof[va] = expit(Xva @ w + b)
            weights.append(
                {"coef": w.tolist(), "intercept": float(b), "success": bool(res.success), "fun": float(res.fun)})
    return oof, {"fold_weights": weights}


# ---------------------------
# Evaluation & Analysis
# ---------------------------
# --- NEW FUNCTION START (For Bottleneck Analysis) ---
def analyze_bottlenecks(y_true: np.ndarray, scores_dict: Dict[str, np.ndarray], methods: np.ndarray,
                        max_remove: int = 5) -> pd.DataFrame:
    """
    Identifies 'worst-offender' fake methods and shows how performance improves when they are excluded.

    Args:
        y_true: Ground truth labels (N,).
        scores_dict: Dictionary of fusion model names to their scores (N,).
        methods: Array of method names for each sample (N,).
        max_remove: The maximum number of worst offenders to remove iteratively.

    Returns:
        A pandas DataFrame summarizing the performance at each step of removal.
    """
    logging.info("Starting bottleneck analysis to find worst-offending methods...")
    results = []
    real_mask = (y_true == 0)
    fake_methods = np.unique(methods[y_true == 1])

    for fusion_name, scores in scores_dict.items():
        # 1. Rank fake methods by their per-method AUC
        method_aucs = {}
        for method in fake_methods:
            method_mask = (methods == method)
            # Create a balanced dataset of this method vs all reals
            y_subset = np.concatenate([y_true[real_mask], y_true[method_mask]])
            s_subset = np.concatenate([scores[real_mask], scores[method_mask]])
            if len(np.unique(y_subset)) < 2:
                continue
            try:
                method_aucs[method] = roc_auc_score(y_subset, s_subset)
            except ValueError:
                method_aucs[method] = 0.5  # Assign neutral score if AUC fails

        # Sort methods by AUC, ascending (worst first)
        worst_offenders = sorted(method_aucs.keys(), key=lambda m: method_aucs[m])

        # 2. Iteratively remove worst offenders and re-evaluate
        for k in range(max_remove + 1):
            if k > len(worst_offenders):
                break

            removed_methods = worst_offenders[:k]

            # Create a mask to exclude the removed methods
            if k == 0:
                eval_mask = np.ones_like(y_true, dtype=bool)
            else:
                eval_mask = ~np.isin(methods, removed_methods)

            y_eval, s_eval, m_eval = y_true[eval_mask], scores[eval_mask], methods[eval_mask]

            if len(y_eval) < 10 or len(np.unique(y_eval)) < 2:
                continue

            overall_auc = roc_auc_score(y_eval, s_eval)
            macro_auc, _ = macro_auc_by_method(y_eval, s_eval, m_eval)

            results.append({
                "fusion_model": fusion_name,
                "num_removed": k,
                "removed_methods": ", ".join(removed_methods) if k > 0 else "None",
                "remaining_videos": len(y_eval),
                "overall_auc": overall_auc,
                "macro_auc": macro_auc,
            })

    logging.info("Bottleneck analysis complete.")
    return pd.DataFrame(results)


# --- NEW FUNCTION END ---


# --- NEW FUNCTION START (For Key Method Performance) ---
def report_key_method_performance(y_true: np.ndarray, scores_dict: Dict[str, np.ndarray], methods: np.ndarray,
                                  key_methods: List[str]) -> pd.DataFrame:
    """
    Reports the performance (TPR) on a specific list of key methods at a globally optimal threshold.

    Args:
        y_true: Ground truth labels (N,).
        scores_dict: Dictionary of fusion model names to their scores (N,).
        methods: Array of method names for each sample (N,).
        key_methods: A list of method names to evaluate.

    Returns:
        A pandas DataFrame summarizing performance on the key methods.
    """
    if not key_methods:
        return pd.DataFrame()
    logging.info(f"Reporting performance for key methods: {key_methods}")
    results = []

    for fusion_name, scores in scores_dict.items():
        # Use Youden's J to find a single, globally optimal threshold
        threshold = youden_threshold(y_true, scores)
        y_pred = (scores >= threshold).astype(int)

        for method in key_methods:
            mask = (methods == method)
            if not np.any(mask):
                logging.warning(f"Key method '{method}' not found in the dataset.")
                continue

            y_m, yp_m = y_true[mask], y_pred[mask]

            # Assuming key methods are fakes (label=1), TPR is the mean of predictions
            if y_m[0] == 1:
                tpr = np.mean(yp_m)
                metric_name = "TPR"
                metric_val = tpr
            else:  # Or if it's a real source, we care about TNR
                tnr = 1 - np.mean(yp_m)
                metric_name = "TNR"
                metric_val = tnr

            results.append({
                "fusion_model": fusion_name,
                "key_method": method,
                "threshold": threshold,
                metric_name: metric_val,
                "count": len(y_m)
            })
    return pd.DataFrame(results)


# --- NEW FUNCTION END ---


def summarize_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float, methods: np.ndarray) -> Dict:
    y_pred = (y_score >= thr).astype(int)
    acc = float(np.mean(y_pred == y_true))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fpr = float(fp / max(1, (fp + tn)))
    fnr = float(fn / max(1, (fn + tp)))

    try:
        overall_auc = roc_auc_score(y_true, y_score)
    except Exception:
        overall_auc = float("nan")
    try:
        ap = average_precision_score(y_true, y_score)
    except Exception:
        ap = float("nan")

    return {
        "threshold": thr,
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "fpr": fpr,
        "fnr": fnr,
        "auc": float(overall_auc),
        "ap": float(ap),
    }


def apply_exclusion(methods: np.ndarray, exclude: str) -> np.ndarray:
    mlow = np.char.lower(methods.astype(str))
    if exclude == "faceforensics++":
        return ~(mlow == "faceforensics++")
    elif exclude == "deepfakedetection":
        return ~(mlow == "deepfakedetection")
    elif exclude == "both":
        return ~((mlow == "faceforensics++") | (mlow == "deepfakedetection"))
    else:
        return np.ones_like(methods, dtype=bool)


def per_method_rates(y_true: np.ndarray, y_pred: np.ndarray, methods: np.ndarray) -> pd.DataFrame:
    """Computes TPR for fake methods and TNR for real sources."""
    out = []
    for m in np.unique(methods):
        mask = (methods == m)
        y_m, yp_m = y_true[mask], y_pred[mask]
        if y_m.size == 0:
            continue
        if y_m[0] == 1:  # fake method
            tpr = float((yp_m == 1).mean())
            out.append({"method": m, "label": "fake", "TPR": tpr, "count": int(y_m.size)})
        else:  # real source
            tnr = float((yp_m == 0).mean())
            out.append({"method": m, "label": "real", "TNR": tnr, "count": int(y_m.size)})
    return pd.DataFrame(out)


def evaluate_scenarios(y: np.ndarray, scores_dict: Dict[str, np.ndarray], methods: np.ndarray, out_csv: str):
    records = []
    scenarios = [
        ("none", None),
        ("fpr_le_1pct", 0.01),
        ("fpr_le_0.2pct", 0.002),
        ("exclude_faceforensics++", "exclude_ffpp"),
        ("exclude_deepfakedetection", "exclude_dfd"),
        ("exclude_both", "exclude_both"),
    ]
    for name, scores in scores_dict.items():
        for scen, param in scenarios:
            mask = np.ones_like(y, dtype=bool)
            if scen == "exclude_faceforensics++":
                mask = apply_exclusion(methods, "faceforensics++")
            elif scen == "exclude_deepfakedetection":
                mask = apply_exclusion(methods, "deepfakedetection")
            elif scen == "exclude_both":
                mask = apply_exclusion(methods, "both")

            y_m = y[mask]
            s_m = scores[mask]
            m_m = methods[mask]

            if y_m.size < 10:
                continue

            thr_eer = eer_threshold(y_m, s_m)
            thr_youden = youden_threshold(y_m, s_m)
            res = None

            if scen == "none":
                res_eer = summarize_at_threshold(y_m, s_m, thr_eer, m_m)
                res_eer.update({"fusion": name, "scenario": scen, "constraint": "EER"})
                records.append(res_eer)
                res = summarize_at_threshold(y_m, s_m, thr_youden, m_m)
                res.update({"fusion": name, "scenario": scen, "constraint": "Youden"})
                records.append(res)
            else:
                if isinstance(param, float):
                    thr = find_threshold_at_fpr(y_m, s_m, param)
                else:
                    thr = thr_youden
                res = summarize_at_threshold(y_m, s_m, thr, m_m)
                res.update({"fusion": name, "scenario": scen,
                            "constraint": f"thr@{param}" if isinstance(param, float) else "Youden"})
                records.append(res)

            if res is not None:
                yhat = (s_m >= res["threshold"]).astype(int)
                pm = per_method_rates(y_m, yhat, m_m)
                pm_out_path = out_csv.replace("summary_metrics.csv", f"per_method_{name}_{scen}.csv")
                pm.to_csv(pm_out_path, index=False)

    pd.DataFrame.from_records(records).to_csv(out_csv, index=False)


def det_grid(y: np.ndarray, s: np.ndarray, fprs: Tuple[float, ...] = (0.002, 0.005, 0.01, 0.02, 0.05, 0.10)):
    rows = []
    fpr_curve, tpr_curve, thr_curve = roc_curve(y, s)
    for target in fprs:
        ok = np.where(fpr_curve <= target)[0]
        if ok.size:
            i = ok[-1]
            rows.append({"fpr_target": target, "thr": float(thr_curve[i]), "tpr": float(tpr_curve[i]),
                         "actual_fpr": float(fpr_curve[i])})
    return pd.DataFrame(rows)


# ---------------------------
# Plotting
# ---------------------------
def plot_roc(y: np.ndarray, scores_dict: Dict[str, np.ndarray], out_png: str):
    plt.figure(figsize=(7, 6))
    for name, s in scores_dict.items():
        fpr, tpr, _ = roc_curve(y, s)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "--", lw=1, alpha=0.5)
    plt.xlim([0.0, 0.1])
    plt.ylim([0.0, 1.0])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC (zoom to low-FPR)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_reliability(y: np.ndarray, scores_dict: Dict[str, np.ndarray], out_png: str, title: str):
    plt.figure(figsize=(7, 6))
    for name, s in scores_dict.items():
        prob_true, prob_pred = calibration_curve(y, s, n_bins=15, strategy="quantile")
        plt.plot(prob_pred, prob_true, marker="o", label=name)
    plt.plot([0, 1], [0, 1], "--", alpha=0.5, label="Perfectly calibrated")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical frequency")
    plt.title(f"Reliability: {title}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_score_hists(y: np.ndarray, scores_dict: Dict[str, np.ndarray], out_png: str):
    plt.figure(figsize=(8, 6))
    n_plots = len(scores_dict)
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols
    for i, (name, s) in enumerate(scores_dict.items()):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.hist(s[y == 0], bins=50, alpha=0.6, label="real", density=True)
        plt.hist(s[y == 1], bins=50, alpha=0.6, label="fake", density=True)
        plt.title(name)
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# ---------------------------
# Main
# ---------------------------
def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    # --- MODIFICATION START (Support N models) ---
    parser.add_argument("--csvs", nargs='+', required=True, help="Paths to the per-frame CSVs (one per model).")
    parser.add_argument("--names", nargs='+', required=True, help="Model names, same order as --csvs.")
    # --- MODIFICATION END ---
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument("--sample", action="store_true", help="Enable sample mode (3 videos per method by default).")
    parser.add_argument("--sample-per-method", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    # --- MODIFICATION START (Support N models) ---
    parser.add_argument("--fixed-aggs", nargs='+', default=None,
                        help="Optional: set aggregators per model, skips auto-selection.")
    # --- MODIFICATION END ---
    # --- NEW ARGUMENT START ---
    parser.add_argument("--key-methods", nargs='+', default=None,
                        help="Optional: list of key methods to report specific performance on.")
    # --- NEW ARGUMENT END ---
    args = parser.parse_args()

    # --- NEW VALIDATION START ---
    if len(args.csvs) != len(args.names):
        parser.error(
            f"--csvs and --names must have the same number of arguments ({len(args.csvs)} vs {len(args.names)}).")
    if args.fixed_aggs and len(args.csvs) != len(args.fixed_aggs):
        parser.error(
            f"--fixed-aggs must have the same number of arguments as --csvs ({len(args.fixed_aggs)} vs {len(args.csvs)}).")
    # --- NEW VALIDATION END ---

    ensure_outdir(args.outdir)
    logging.info(f"Starting analysis for {len(args.names)} models: {args.names}")

    # 1) Load
    dfs = []
    for path, name in zip(args.csvs, args.names):
        dfs.append(load_frame_csv(path, name))

    # 2) Align / Sample
    keys_intersection = set(dfs[0]["clip_key"].unique())
    for df in dfs[1:]:
        keys_intersection &= set(df["clip_key"].unique())
    logging.info(f"Common clips across all models: {len(keys_intersection):,}")

    if args.sample:
        sampled_keys = sample_clip_keys_intersection(dfs, args.sample_per_method, args.seed)
        logging.info(f"Sample mode: keeping {len(sampled_keys):,} clips")
        keys_intersection = set(sampled_keys)

    dfs = [df[df["clip_key"].isin(keys_intersection)].copy() for df in dfs]

    logging.info("Performing strict frame alignment...")
    common_frames = dfs[0][["clip_key", "frame_idx"]].drop_duplicates()
    for df in dfs[1:]:
        common_frames = common_frames.merge(df[["clip_key", "frame_idx"]].drop_duplicates(),
                                            on=["clip_key", "frame_idx"], how="inner")
    total_frames_before = sum(len(df) for df in dfs)
    for i in range(len(dfs)):
        dfs[i] = dfs[i].merge(common_frames, on=["clip_key", "frame_idx"], how="inner")
    total_frames_after = sum(len(df) for df in dfs)
    logging.info(f"Frame alignment reduced total frames from {total_frames_before:,} to {total_frames_after:,}")

    # 3) Aggregate per video per model
    per_model_video = []
    for df, name in zip(dfs, args.names):
        logging.info(f"Aggregating video features for {name} …")
        vdf = aggregate_per_video(df)
        vdf.columns = [f"{name}::{c}" if c not in ("clip_key", "label", "method") else c for c in vdf.columns]
        per_model_video.append(vdf)

    # 4) Merge per-model video tables
    logging.info("Merging per-model tables …")
    base = per_model_video[0]
    for vdf in per_model_video[1:]:
        base = base.merge(vdf, on=["clip_key", "label", "method"], how="inner")
    logging.info(f"Merged videos: {len(base):,}")
    logging.info(
        f"Methods in data: {base['method'].nunique()} | Reals: {int((base['label'] == 0).sum())} | Fakes: {int((base['label'] == 1).sum())}")

    # 5) Pick best aggregator per model
    best_agg = {}
    if args.fixed_aggs:
        for name, agg in zip(args.names, args.fixed_aggs):
            best_agg[name] = agg
            logging.info(f"Forcing aggregator for {name}: {agg}")
    else:
        for name in args.names:
            cols = [c for c in base.columns if c.startswith(f"{name}::")]
            dfm = base[["clip_key", "label", "method"] + cols].copy()
            local = dfm.rename(
                columns={f"{name}::{agg}": agg for agg in CANDIDATE_AGGS if f"{name}::{agg}" in dfm.columns})
            agg_name = pick_best_aggregator(local)
            best_agg[name] = agg_name
    logging.info(f"Best aggregator per model: {best_agg}")

    # 6) Build per-model raw scores and calibrate OOF
    y = base["label"].to_numpy(dtype=int)
    methods = base["method"].astype(str).to_numpy()

    raw_scores = {}
    cal_scores = {}
    for name in args.names:
        agg_col = f"{name}::{best_agg[name]}"
        s = base[agg_col].to_numpy(dtype=float)
        raw_scores[name] = s
        oof_cal, calibrators = calibrate_isotonic_oof(s, y, n_splits=5, random_state=args.seed)
        cal_scores[name] = oof_cal

    # 7) Fusion OOF
    p_mat = np.stack([cal_scores[n] for n in args.names], axis=1)
    fusion = {}
    fusion["noisy_or"] = fusion_noisyor(p_mat)
    fusion["sum_logits"] = fusion_sum_logits(p_mat)

    maxp = np.max(p_mat, axis=1)
    stdp = np.std(p_mat, axis=1)
    top2 = np.sort(p_mat, axis=1)[:, -2:].mean(axis=1) if p_mat.shape[1] >= 2 else maxp
    X_meta = np.column_stack([p_mat, maxp, stdp, top2])
    oof_stack, stack_info = stacked_logistic_oof(X_meta, y, nonneg=True, n_splits=5, random_state=args.seed)
    fusion["stacked_logit_nonneg"] = oof_stack

    # 8) Save intermediate tables
    base_out = base[["clip_key", "label", "method"] + [c for c in base.columns if "::" in c]]
    base_out.to_parquet(os.path.join(args.outdir, "per_video_features.parquet"))

    oof_df = pd.DataFrame({"clip_key": base["clip_key"], "label": y, "method": methods})
    for name in args.names:
        oof_df[f"{name}_calib_prob"] = cal_scores[name]
        oof_df[f"{name}_raw_{best_agg[name]}"] = raw_scores[name]
    oof_df.to_parquet(os.path.join(args.outdir, "oof_calibrated_probs.parquet"))

    fusion_df = pd.DataFrame({"clip_key": base["clip_key"], "label": y, "method": methods, **fusion})
    fusion_df.to_parquet(os.path.join(args.outdir, "fusion_oof_scores.parquet"))
    with open(os.path.join(args.outdir, "fusion_meta.json"), "w") as f:
        json.dump({"best_agg": best_agg, "stacker_info": stack_info}, f, indent=2)

    # --- NEW BLOCK START (Analysis) ---
    # 9a) Bottleneck Analysis
    bottleneck_df = analyze_bottlenecks(y, fusion, methods, max_remove=5)
    bottleneck_df.to_csv(os.path.join(args.outdir, "bottleneck_analysis.csv"), index=False)
    logging.info(f"Saved bottleneck analysis to: {os.path.join(args.outdir, 'bottleneck_analysis.csv')}")

    # 9b) Key Method Performance
    if args.key_methods:
        key_method_df = report_key_method_performance(y, fusion, methods, args.key_methods)
        key_method_df.to_csv(os.path.join(args.outdir, "key_methods_performance.csv"), index=False)
        logging.info(f"Saved key method performance to: {os.path.join(args.outdir, 'key_methods_performance.csv')}")
    # --- NEW BLOCK END ---

    # 9c) General Evaluation across scenarios
    eval_scores = {**cal_scores, **fusion}
    evaluate_scenarios(y, eval_scores, methods, out_csv=os.path.join(args.outdir, "summary_metrics.csv"))

    logging.info("Generating DET grids...")
    for name, s in eval_scores.items():
        det_df = det_grid(y, s)
        det_df.to_csv(os.path.join(args.outdir, f"det_grid_{name}.csv"), index=False)

    # 10) Plots
    plot_roc(y, fusion, os.path.join(args.outdir, "roc_low_fpr_fusion.png"))
    for name in args.names:
        plot_reliability(y,
                         scores_dict={"raw": raw_scores[name], "calibrated": cal_scores[name]},
                         out_png=os.path.join(args.outdir, f"reliability_{name}.png"),
                         title=name)
    best_fusion_name = max(fusion.items(), key=lambda item: roc_auc_score(y, item[1]))[0]
    best_fusion_scores = fusion[best_fusion_name]
    plot_reliability(y,
                     scores_dict={best_fusion_name: best_fusion_scores},
                     out_png=os.path.join(args.outdir, "reliability_best_fusion.png"),
                     title=f"Best Fusion: {best_fusion_name}")
    plot_score_hists(y, fusion, os.path.join(args.outdir, "score_histograms_fusion.png"))

    # 11) Final console summary (topline)
    logging.info("Topline AUCs (OOF):")
    for k, v in eval_scores.items():
        logging.info(f"  {k:>25s}: AUC={roc_auc_score(y, v):.4f}")
    logging.info("Done. Artifacts saved to: %s", args.outdir)


if __name__ == "__main__":
    main()
