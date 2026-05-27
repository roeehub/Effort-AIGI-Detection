#!/usr/bin/env python3
"""Linear-probe runner.

Loads one or more ``features.npz`` files produced by
``extract_features_for_probes.py``, trains a stratified 80/20 logistic-regression
probe per checkpoint, and writes per-checkpoint + per-class accuracy plus a
PASS/FAIL verdict against the §6 threshold from the master plan.

PASS semantics: low probe accuracy means the corresponding label (source bucket,
camera, identity, …) is NOT linearly readable from the model's features. High
probe accuracy is the failure mode — it says the model encodes the shortcut.

Usage (locally, CPU only, no GPU needed):
    python3 analysis/probe_battery_2026-04-26/run_linear_probes.py \\
        --features_npz gs://.../checkpoint_A/features.npz=ckpt_A \\
        --features_npz gs://.../checkpoint_B/features.npz=ckpt_B \\
        --label_field source_bucket \\
        --threshold 0.25 \\
        --output_dir analysis/probe_battery_2026-04-26/source_probe_run_2026-04-27/

GCS URIs are downloaded to a local cache; local paths work too.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("linear-probes")

SEED = 737


def _download_if_gcs(uri: str) -> str:
    if not uri.startswith("gs://"):
        return uri
    from google.cloud import storage  # local import — keep top of file dep-free
    bucket_name, _, blob_name = uri[5:].partition("/")
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    local_dir = tempfile.mkdtemp(prefix="probe_npz_")
    local_path = os.path.join(local_dir, Path(blob_name).name)
    logger.info("Downloading %s → %s", uri, local_path)
    blob.download_to_filename(local_path)
    return local_path


def _parse_features_arg(arg: str) -> Tuple[str, str]:
    """Parse ``URI=name`` or just ``URI``; name defaults to the parent dir."""
    if "=" in arg:
        uri, name = arg.split("=", 1)
        return uri, name
    parent = Path(arg.rstrip("/")).parent.name
    return arg, parent or "checkpoint"


def _load_npz(uri: str, label_field: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    local = _download_if_gcs(uri)
    with np.load(local, allow_pickle=True) as data:
        X = data["features"].astype(np.float32)
        if label_field not in data.files:
            raise KeyError(
                f"label_field '{label_field}' not found in {uri}. "
                f"Available: {sorted(data.files)}"
            )
        y_raw = np.array([str(v) for v in data[label_field]])
        # 'label' field exists for every npz produced by extract_features_for_probes
        y_realfake = np.array([str(v) for v in data["label"]]) if "label" in data.files else np.array([])
    return X, y_raw, y_realfake


def _stratified_split(X: np.ndarray, y: np.ndarray, test_frac: float, seed: int):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_frac, stratify=y, random_state=seed)


def _fit_and_eval(X_tr, y_tr, X_te, y_te) -> Dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
    )
    clf = LogisticRegression(C=1.0, max_iter=2000, n_jobs=1)
    clf.fit(X_tr, y_tr)
    y_pred_tr = clf.predict(X_tr)
    y_pred_te = clf.predict(X_te)
    acc_tr = float(accuracy_score(y_tr, y_pred_tr))
    acc_te = float(accuracy_score(y_te, y_pred_te))
    classes = sorted(set(np.concatenate([y_tr, y_te]).tolist()))
    cm = confusion_matrix(y_te, y_pred_te, labels=classes).tolist()
    report = classification_report(
        y_te, y_pred_te, labels=classes, zero_division=0, output_dict=True
    )
    return {
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_classes": len(classes),
        "classes": classes,
        "train_accuracy": acc_tr,
        "test_accuracy": acc_te,
        "chance_accuracy": 1.0 / max(len(classes), 1),
        "confusion_matrix": cm,
        "per_class": {
            c: {
                "precision": float(report[c]["precision"]),
                "recall": float(report[c]["recall"]),
                "f1": float(report[c]["f1-score"]),
                "support": int(report[c]["support"]),
            }
            for c in classes
        },
    }


def _verdict(test_accuracy: float, threshold: float) -> str:
    return "PASS" if test_accuracy <= threshold else "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--features_npz", action="append", required=True,
        help="Repeatable: URI[=name] of a features.npz from extract_features_for_probes.py.",
    )
    ap.add_argument("--label_field", default="source_bucket",
                    help="Which label column to probe (source_bucket | provenance | source_name).")
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="PASS verdict if test accuracy <= threshold.")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--output_dir", required=True,
                    help="Directory to write per-checkpoint JSON + summary.csv.")
    ap.add_argument("--min_per_class", type=int, default=2,
                    help="Drop classes with fewer samples than this before splitting.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict] = []

    for spec in args.features_npz:
        uri, name = _parse_features_arg(spec)
        logger.info("=== %s (%s) ===", name, uri)
        X, y, y_realfake = _load_npz(uri, args.label_field)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"feature/label length mismatch in {uri}: {X.shape[0]} vs {y.shape[0]}")

        counts = Counter(y.tolist())
        keep_classes = {c for c, n in counts.items() if n >= args.min_per_class}
        dropped = sorted(set(counts) - keep_classes)
        if dropped:
            logger.warning("Dropping %d underpopulated class(es) (< %d samples): %s",
                           len(dropped), args.min_per_class, dropped)
        mask = np.array([c in keep_classes for c in y])
        X, y = X[mask], y[mask]

        if len(set(y)) < 2:
            logger.error("Skipping %s: fewer than 2 classes after filtering.", name)
            continue

        X_tr, X_te, y_tr, y_te = _stratified_split(X, y, args.test_frac, args.seed)
        result = _fit_and_eval(X_tr, y_tr, X_te, y_te)
        result["checkpoint_name"] = name
        result["features_uri"] = uri
        result["label_field"] = args.label_field
        result["threshold"] = args.threshold
        result["verdict"] = _verdict(result["test_accuracy"], args.threshold)
        if y_realfake.size:
            result["realfake_distribution"] = dict(Counter(y_realfake.tolist()))

        out_json = out_dir / f"{name}.{args.label_field}.json"
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(
            "%s — test_acc=%.3f (chance=%.3f) → %s — wrote %s",
            name, result["test_accuracy"], result["chance_accuracy"],
            result["verdict"], out_json,
        )

        summary_rows.append({
            "checkpoint_name": name,
            "label_field": args.label_field,
            "n_classes": result["n_classes"],
            "n_train": result["n_train"],
            "n_test": result["n_test"],
            "chance_accuracy": result["chance_accuracy"],
            "train_accuracy": result["train_accuracy"],
            "test_accuracy": result["test_accuracy"],
            "threshold": args.threshold,
            "verdict": result["verdict"],
        })

    summary_csv = out_dir / f"summary.{args.label_field}.csv"
    if summary_rows:
        cols = list(summary_rows[0].keys())
        with open(summary_csv, "w") as f:
            f.write(",".join(cols) + "\n")
            for row in summary_rows:
                f.write(",".join(
                    f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])
                    for c in cols
                ) + "\n")
        logger.info("Wrote summary → %s", summary_csv)
        logger.info("\n%-40s  %-15s  %-15s  %s",
                    "checkpoint", "test_accuracy", "chance", "verdict")
        for row in summary_rows:
            logger.info("%-40s  %-15.4f  %-15.4f  %s",
                        row["checkpoint_name"], row["test_accuracy"],
                        row["chance_accuracy"], row["verdict"])
    else:
        logger.warning("No probes were trained; summary not written.")

    return 0 if all(r["verdict"] == "PASS" for r in summary_rows) else 0  # always 0 — verdict is in the JSON


if __name__ == "__main__":
    raise SystemExit(main())
