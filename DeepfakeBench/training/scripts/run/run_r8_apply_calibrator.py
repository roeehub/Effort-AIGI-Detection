#!/usr/bin/env python3
"""
Apply an exported R8 calibrator bundle to a CSV of raw scores.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


EPS = 1e-6


def _clip_probs(x: np.ndarray) -> np.ndarray:
    return np.clip(x, EPS, 1.0 - EPS)


def _logit(p: np.ndarray) -> np.ndarray:
    pp = _clip_probs(p)
    return np.log(pp / (1.0 - pp))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _write_csv(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _apply_platt(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    return _clip_probs(_sigmoid(a * _logit(scores) + b))


def _apply_isotonic(scores: np.ndarray, x: List[float], y: List[float]) -> np.ndarray:
    if len(x) == 0 or len(y) == 0:
        return scores.copy()
    return _clip_probs(np.interp(scores, np.array(x), np.array(y), left=y[0], right=y[-1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply R8 calibrator bundle to score CSV")
    parser.add_argument("--scores_csv", type=str, required=True)
    parser.add_argument("--bundle_json", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)

    parser.add_argument("--score_col", type=str, default="raw_prob")
    parser.add_argument("--model", type=str, default="best", choices=["best", "raw", "platt", "isotonic"])
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold. If omitted, uses bundle best-model frame threshold.")
    args = parser.parse_args()

    rows = _read_csv(args.scores_csv)
    if args.score_col not in rows[0]:
        raise ValueError(f"Missing score column in CSV: {args.score_col}")

    with open(args.bundle_json, "r") as f:
        bundle = json.load(f)

    model = bundle.get("best_model") if args.model == "best" else args.model
    if model not in {"raw", "platt", "isotonic"}:
        raise ValueError(f"Unsupported model in bundle: {model}")

    raw_scores = _clip_probs(np.array([float(r[args.score_col]) for r in rows], dtype=np.float64))

    if model == "raw":
        calibrated = raw_scores
    elif model == "platt":
        m = bundle["models"]["platt"]
        calibrated = _apply_platt(raw_scores, float(m["a"]), float(m["b"]))
    else:
        m = bundle["models"]["isotonic"]
        calibrated = _apply_isotonic(raw_scores, m.get("x_thresholds", []), m.get("y_thresholds", []))

    if args.threshold is not None:
        threshold = float(args.threshold)
    else:
        threshold = float(
            bundle["recommended_thresholds"][bundle["best_model"]]["frame"]["threshold"]
        )

    output_rows: List[Dict[str, str]] = []
    for row, p in zip(rows, calibrated):
        out = dict(row)
        out["calibrated_model"] = model
        out["calibrated_prob"] = f"{float(p):.8f}"
        out["decision_threshold"] = f"{threshold:.6f}"
        out["decision"] = "FAKE" if float(p) >= threshold else "REAL"
        output_rows.append(out)

    fieldnames = list(output_rows[0].keys())
    _write_csv(args.output_csv, output_rows, fieldnames)

    print("[run_r8_apply_calibrator] done")
    print(f"  model         : {model}")
    print(f"  threshold     : {threshold:.6f}")
    print(f"  output_csv    : {Path(args.output_csv).resolve()}")


if __name__ == "__main__":
    main()
