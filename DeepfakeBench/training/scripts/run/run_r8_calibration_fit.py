#!/usr/bin/env python3
"""
Fit and evaluate post-hoc calibration for R8 using identity-safe splits.

Input CSV is expected to contain at least:
  - label (0/1)
  - identity
  - raw_prob (model output probability before calibration)

What this script does:
1) Split by identity into calibration/eval sets (or use provided split CSV).
2) Fit calibrators on calibration split:
   - raw (identity mapping)
   - platt: sigmoid(a * logit(p) + b)
   - isotonic regression
3) Evaluate each calibrator on eval split (frame-level + identity-mean level).
4) Choose threshold(s) with FP-first objective and fake-TPR constraint.
5) Export calibrator bundle + detailed report.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit


EPS = 1e-6


@dataclass
class SplitResult:
    cal_identities: List[str]
    eval_identities: List[str]


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_label(value: str) -> int:
    text = str(value).strip().lower()
    if text in {"0", "real", "r", "false", "negative"}:
        return 0
    if text in {"1", "fake", "f", "true", "positive"}:
        return 1
    try:
        v = int(float(text))
    except Exception as exc:
        raise ValueError(f"Unsupported label value: {value}") from exc
    if v not in (0, 1):
        raise ValueError(f"Label must be 0 or 1, got: {value}")
    return v


def _clip_probs(x: np.ndarray) -> np.ndarray:
    return np.clip(x, EPS, 1.0 - EPS)


def _logit(p: np.ndarray) -> np.ndarray:
    pp = _clip_probs(p)
    return np.log(pp / (1.0 - pp))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        w = float(np.sum(mask)) / max(1.0, float(n))
        ece += w * abs(acc - conf)
    return float(ece)


def _eer_and_threshold(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y, p, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float(fpr[idx]), float(thr[idx])


def _metrics_basic(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["n"] = int(len(y))
    out["n_real"] = int(np.sum(y == 0))
    out["n_fake"] = int(np.sum(y == 1))

    if len(np.unique(y)) < 2:
        out.update({
            "auc": float("nan"),
            "brier": float("nan"),
            "logloss": float("nan"),
            "ece": float("nan"),
            "eer": float("nan"),
            "eer_threshold": float("nan"),
        })
        return out

    out["auc"] = float(roc_auc_score(y, p))
    out["brier"] = float(brier_score_loss(y, p))
    out["logloss"] = float(log_loss(y, p, labels=[0, 1]))
    out["ece"] = float(_ece(y, p))
    eer, eer_thr = _eer_and_threshold(y, p)
    out["eer"] = eer
    out["eer_threshold"] = eer_thr
    return out


def _threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (p >= threshold).astype(np.int32)
    real_mask = y == 0
    fake_mask = y == 1

    real_fpr = float(np.mean(pred[real_mask] == 1)) if np.any(real_mask) else float("nan")
    fake_tpr = float(np.mean(pred[fake_mask] == 1)) if np.any(fake_mask) else float("nan")

    out = {
        "threshold": float(threshold),
        "real_fpr": real_fpr,
        "fake_tpr": fake_tpr,
    }
    if np.isfinite(real_fpr) and np.isfinite(fake_tpr):
        out["balanced_acc"] = float(np.mean([1.0 - real_fpr, fake_tpr]))
    else:
        out["balanced_acc"] = float("nan")
    return out


def _select_threshold(
    y: np.ndarray,
    p: np.ndarray,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    min_fake_tpr: float,
    fallback_lambda: float,
) -> Dict[str, float]:
    thresholds = np.arange(threshold_min, threshold_max + 1e-12, threshold_step)
    thresholds = np.unique(np.clip(np.round(thresholds, 6), 0.0, 1.0))

    grid = [_threshold_metrics(y, p, float(t)) for t in thresholds]

    feasible = [g for g in grid if np.isfinite(g["fake_tpr"]) and g["fake_tpr"] >= min_fake_tpr]
    if feasible:
        feasible.sort(key=lambda g: (g["real_fpr"], -g["fake_tpr"], -g["threshold"]))
        best = dict(feasible[0])
        best["selection_mode"] = "constraint_min_fpr"
        best["feasible_count"] = len(feasible)
        return best

    scored: List[Dict[str, float]] = []
    for g in grid:
        if not np.isfinite(g["real_fpr"]) or not np.isfinite(g["fake_tpr"]):
            continue
        utility = g["fake_tpr"] - fallback_lambda * g["real_fpr"]
        gg = dict(g)
        gg["utility"] = float(utility)
        scored.append(gg)

    if not scored:
        raise RuntimeError("No valid threshold candidates")

    scored.sort(key=lambda g: (-g["utility"], g["real_fpr"], -g["fake_tpr"], -g["threshold"]))
    best = dict(scored[0])
    best["selection_mode"] = "fallback_utility"
    best["feasible_count"] = 0
    return best


def _thresholds_at_target_fpr(
    y: np.ndarray,
    p: np.ndarray,
    targets: Sequence[float],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if len(np.unique(y)) < 2:
        for t in targets:
            out[str(t)] = {"threshold": float("nan"), "fake_tpr": float("nan")}
        return out

    fpr, tpr, thr = roc_curve(y, p, pos_label=1)
    for target in targets:
        idxs = np.where(fpr <= target)[0]
        key = str(target)
        if len(idxs) == 0:
            out[key] = {"threshold": 1.0, "fake_tpr": 0.0}
            continue
        idx = int(idxs[-1])
        out[key] = {
            "threshold": float(thr[idx]),
            "fake_tpr": float(tpr[idx]),
        }
    return out


def _identity_majority_labels(identities: np.ndarray, labels: np.ndarray) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for identity in sorted(set(identities.tolist())):
        mask = identities == identity
        vals = labels[mask]
        zeros = int(np.sum(vals == 0))
        ones = int(np.sum(vals == 1))
        out[str(identity)] = 1 if ones >= zeros else 0
    return out


def _split_identities_auto(
    identities: np.ndarray,
    labels: np.ndarray,
    holdout_fraction: float,
    seed: int,
) -> SplitResult:
    id_to_label = _identity_majority_labels(identities, labels)
    id_list = np.array(sorted(id_to_label.keys()))
    id_labels = np.array([id_to_label[i] for i in id_list], dtype=np.int32)

    # Prefer stratified split when possible.
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=holdout_fraction, random_state=seed)
        tr_idx, te_idx = next(sss.split(id_list, id_labels))
        cal_ids = id_list[tr_idx].tolist()
        eval_ids = id_list[te_idx].tolist()
    except Exception:
        rng = np.random.default_rng(seed)
        shuffled = id_list.copy()
        rng.shuffle(shuffled)
        n_eval = max(1, int(round(len(shuffled) * holdout_fraction)))
        eval_ids = shuffled[:n_eval].tolist()
        cal_ids = shuffled[n_eval:].tolist()

    if not cal_ids or not eval_ids:
        raise RuntimeError("Identity split failed to produce both calibration and eval sets")

    return SplitResult(cal_identities=sorted(cal_ids), eval_identities=sorted(eval_ids))


def _split_identities_from_csv(path: str) -> SplitResult:
    rows = _read_csv(path)
    if "identity" not in rows[0] or "split" not in rows[0]:
        raise ValueError("split_csv must include columns: identity,split")

    cal_ids = sorted({r["identity"] for r in rows if r["split"].strip().lower() in {"cal", "calibration", "train"}})
    eval_ids = sorted({r["identity"] for r in rows if r["split"].strip().lower() in {"eval", "val", "test"}})

    if not cal_ids or not eval_ids:
        raise ValueError("split_csv must contain both calibration/train and eval/val/test identities")

    overlap = set(cal_ids) & set(eval_ids)
    if overlap:
        raise ValueError(f"split_csv has overlapping identities across splits: {sorted(overlap)[:5]}")

    return SplitResult(cal_identities=cal_ids, eval_identities=eval_ids)


def _fit_platt(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    x = _logit(scores).reshape(-1, 1)
    y = labels.astype(np.int32)

    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    lr.fit(x, y)

    a = float(lr.coef_[0][0])
    b = float(lr.intercept_[0])
    return a, b


def _apply_platt(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    z = a * _logit(scores) + b
    return _clip_probs(_sigmoid(z))


def _fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(scores, labels)
    return ir


def _aggregate_identity(
    identities: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    agg_prob: List[float] = []
    agg_label: List[int] = []

    for identity in sorted(set(identities.tolist())):
        mask = identities == identity
        p = float(np.mean(probs[mask]))
        lvals = labels[mask]
        zeros = int(np.sum(lvals == 0))
        ones = int(np.sum(lvals == 1))
        l = 1 if ones >= zeros else 0
        agg_prob.append(p)
        agg_label.append(l)

    return np.array(agg_label, dtype=np.int32), np.array(agg_prob, dtype=np.float64)


def _safe_float(x: float) -> float:
    if x is None:
        return float("nan")
    try:
        v = float(x)
    except Exception:
        return float("nan")
    if math.isinf(v) or math.isnan(v):
        return float("nan")
    return v


def _build_bundle(
    *,
    best_model: str,
    platt_a: float,
    platt_b: float,
    isotonic: IsotonicRegression,
    recommendations: Dict[str, Dict[str, object]],
    target_fpr_map: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, object]:
    x_th = [float(v) for v in getattr(isotonic, "X_thresholds_", [])]
    y_th = [float(v) for v in getattr(isotonic, "y_thresholds_", [])]

    return {
        "version": 1,
        "created_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "best_model": best_model,
        "models": {
            "raw": {"type": "identity"},
            "platt": {
                "type": "platt",
                "input": "raw_prob",
                "equation": "sigmoid(a*logit(p)+b)",
                "a": float(platt_a),
                "b": float(platt_b),
            },
            "isotonic": {
                "type": "isotonic",
                "input": "raw_prob",
                "x_thresholds": x_th,
                "y_thresholds": y_th,
            },
        },
        "recommended_thresholds": recommendations,
        "thresholds_at_target_fpr": target_fpr_map,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit R8 threshold calibration")
    parser.add_argument("--scores_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--identity_col", type=str, default="identity")
    parser.add_argument("--score_col", type=str, default="raw_prob")

    parser.add_argument("--split_csv", type=str, default="",
                        help="Optional identity split CSV with columns: identity,split")
    parser.add_argument("--holdout_fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--threshold_min", type=float, default=0.01)
    parser.add_argument("--threshold_max", type=float, default=0.99)
    parser.add_argument("--threshold_step", type=float, default=0.001)
    parser.add_argument("--min_fake_tpr", type=float, default=0.85)
    parser.add_argument("--fallback_lambda", type=float, default=2.0)
    parser.add_argument("--target_fprs", type=str, default="0.005,0.01,0.02,0.05")
    args = parser.parse_args()

    if not (0.0 < args.holdout_fraction < 1.0):
        raise ValueError("holdout_fraction must be in (0,1)")
    if not (0.0 <= args.threshold_min < args.threshold_max <= 1.0):
        raise ValueError("Invalid threshold range")
    if args.threshold_step <= 0:
        raise ValueError("threshold_step must be > 0")

    target_fprs: List[float] = []
    for part in args.target_fprs.split(","):
        part = part.strip()
        if not part:
            continue
        target_fprs.append(float(part))
    if not target_fprs:
        raise ValueError("No valid target_fprs provided")

    rows = _read_csv(args.scores_csv)
    for col in (args.label_col, args.identity_col, args.score_col):
        if col not in rows[0]:
            raise ValueError(f"Missing required column: {col}")

    labels = np.array([_parse_label(r[args.label_col]) for r in rows], dtype=np.int32)
    identities = np.array([str(r[args.identity_col]) for r in rows])
    scores = _clip_probs(np.array([float(r[args.score_col]) for r in rows], dtype=np.float64))

    if len(np.unique(labels)) < 2:
        raise RuntimeError("scores_csv must contain both real and fake samples")

    if args.split_csv:
        split = _split_identities_from_csv(args.split_csv)
    else:
        split = _split_identities_auto(
            identities=identities,
            labels=labels,
            holdout_fraction=args.holdout_fraction,
            seed=args.seed,
        )

    cal_set = set(split.cal_identities)
    eval_set = set(split.eval_identities)

    mask_cal = np.array([x in cal_set for x in identities])
    mask_eval = np.array([x in eval_set for x in identities])

    if not np.any(mask_cal) or not np.any(mask_eval):
        raise RuntimeError("Split produced empty calibration or eval subset")

    y_cal = labels[mask_cal]
    p_cal_raw = scores[mask_cal]

    y_eval = labels[mask_eval]
    p_eval_raw = scores[mask_eval]
    id_eval = identities[mask_eval]

    # Fit calibrators on calibration split only.
    platt_a, platt_b = _fit_platt(p_cal_raw, y_cal)
    isotonic = _fit_isotonic(p_cal_raw, y_cal)

    p_eval_platt = _apply_platt(p_eval_raw, platt_a, platt_b)
    p_eval_iso = _clip_probs(isotonic.transform(p_eval_raw))

    models_eval = {
        "raw": p_eval_raw,
        "platt": p_eval_platt,
        "isotonic": p_eval_iso,
    }

    model_report: Dict[str, Dict[str, object]] = {}
    recommendations: Dict[str, Dict[str, object]] = {}
    target_fpr_map: Dict[str, Dict[str, Dict[str, float]]] = {}

    scored_eval_rows: List[Dict[str, object]] = []

    for model_name, p_eval in models_eval.items():
        frame_metrics = _metrics_basic(y_eval, p_eval)
        frame_threshold = _select_threshold(
            y=y_eval,
            p=p_eval,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
            min_fake_tpr=args.min_fake_tpr,
            fallback_lambda=args.fallback_lambda,
        )
        frame_fpr_targets = _thresholds_at_target_fpr(y_eval, p_eval, target_fprs)

        y_eval_id, p_eval_id = _aggregate_identity(id_eval, y_eval, p_eval)
        id_metrics = _metrics_basic(y_eval_id, p_eval_id)
        id_threshold = _select_threshold(
            y=y_eval_id,
            p=p_eval_id,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
            min_fake_tpr=args.min_fake_tpr,
            fallback_lambda=args.fallback_lambda,
        )
        id_fpr_targets = _thresholds_at_target_fpr(y_eval_id, p_eval_id, target_fprs)

        model_report[model_name] = {
            "frame": {
                "metrics": frame_metrics,
                "recommended_threshold": frame_threshold,
                "thresholds_at_target_fpr": frame_fpr_targets,
            },
            "identity_mean": {
                "metrics": id_metrics,
                "recommended_threshold": id_threshold,
                "thresholds_at_target_fpr": id_fpr_targets,
            },
        }

        recommendations[model_name] = {
            "frame": frame_threshold,
            "identity_mean": id_threshold,
        }

        target_fpr_map[model_name] = {
            "frame": frame_fpr_targets,
            "identity_mean": id_fpr_targets,
        }

    # Pick best calibrator by eval frame FP-first objective, with brier tie-break.
    def _rank_key(m: str) -> Tuple[float, float, float, float]:
        rec = recommendations[m]["frame"]
        metrics = model_report[m]["frame"]["metrics"]
        real_fpr = _safe_float(rec.get("real_fpr", float("inf")))
        fake_tpr = _safe_float(rec.get("fake_tpr", float("nan")))
        brier = _safe_float(metrics.get("brier", float("inf")))
        ece = _safe_float(metrics.get("ece", float("inf")))
        # lower real_fpr better, higher fake_tpr better, then lower brier/ece
        return (real_fpr, -fake_tpr, brier, ece)

    ranked_models = sorted(models_eval.keys(), key=_rank_key)
    best_model = ranked_models[0]

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write split CSV for reproducibility.
    split_rows = [{"identity": x, "split": "calibration"} for x in split.cal_identities]
    split_rows += [{"identity": x, "split": "eval"} for x in split.eval_identities]
    split_rows.sort(key=lambda r: (r["split"], r["identity"]))
    _write_csv(
        str(out_dir / "split_identities.csv"),
        split_rows,
        fieldnames=["identity", "split"],
    )

    # Detailed eval rows.
    eval_indices = np.where(mask_eval)[0]
    for idx, base_idx in enumerate(eval_indices):
        row = dict(rows[int(base_idx)])
        row["label_int"] = int(y_eval[idx])
        row["raw_prob_eval"] = float(p_eval_raw[idx])
        row["platt_prob_eval"] = float(p_eval_platt[idx])
        row["isotonic_prob_eval"] = float(p_eval_iso[idx])
        scored_eval_rows.append(row)

    eval_fields = list(scored_eval_rows[0].keys()) if scored_eval_rows else []
    if eval_fields:
        _write_csv(str(out_dir / "eval_scored_with_calibration.csv"), scored_eval_rows, eval_fields)

    bundle = _build_bundle(
        best_model=best_model,
        platt_a=platt_a,
        platt_b=platt_b,
        isotonic=isotonic,
        recommendations=recommendations,
        target_fpr_map=target_fpr_map,
    )

    with (out_dir / "calibrator_bundle.json").open("w") as f:
        json.dump(bundle, f, indent=2)

    report = {
        "created_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "input": {
            "scores_csv": str(Path(args.scores_csv).resolve()),
            "label_col": args.label_col,
            "identity_col": args.identity_col,
            "score_col": args.score_col,
        },
        "data_summary": {
            "n_rows_total": int(len(rows)),
            "n_identities_total": int(len(set(identities.tolist()))),
            "n_rows_calibration": int(np.sum(mask_cal)),
            "n_rows_eval": int(np.sum(mask_eval)),
            "n_identities_calibration": int(len(split.cal_identities)),
            "n_identities_eval": int(len(split.eval_identities)),
            "n_real_eval": int(np.sum(y_eval == 0)),
            "n_fake_eval": int(np.sum(y_eval == 1)),
        },
        "settings": {
            "holdout_fraction": args.holdout_fraction,
            "seed": args.seed,
            "threshold_min": args.threshold_min,
            "threshold_max": args.threshold_max,
            "threshold_step": args.threshold_step,
            "min_fake_tpr": args.min_fake_tpr,
            "fallback_lambda": args.fallback_lambda,
            "target_fprs": target_fprs,
        },
        "best_model": best_model,
        "model_ranking": ranked_models,
        "model_report": model_report,
    }

    with (out_dir / "calibration_report.json").open("w") as f:
        json.dump(report, f, indent=2)

    print("[run_r8_calibration_fit] done")
    print(f"  output_dir     : {out_dir}")
    print(f"  best_model     : {best_model}")
    print(
        "  frame threshold: "
        f"{recommendations[best_model]['frame']['threshold']:.4f} "
        f"(real_fpr={recommendations[best_model]['frame']['real_fpr']:.4f}, "
        f"fake_tpr={recommendations[best_model]['frame']['fake_tpr']:.4f})"
    )


if __name__ == "__main__":
    main()
