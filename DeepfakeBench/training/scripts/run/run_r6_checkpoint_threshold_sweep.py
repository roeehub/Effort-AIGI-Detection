"""
R6 checkpoint + threshold overnight sweep runner.

What this script does:
1) Discovers checkpoint files from a GCS prefix (default: top_n checkpoints only).
2) Runs `run_target_domain_validation_sequential.py` across all discovered checkpoints
   and all suites in a manifest.
3) Pulls generated `videos_report.csv` files from GCS.
4) Sweeps thresholds and selects:
   - best threshold per checkpoint
   - best checkpoint overall
   with a false-positive-minimizing objective and fake-TPR constraints.
5) Writes analysis artifacts locally and uploads them to GCS.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
from google.cloud import storage


def _training_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _load_manifest(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        text = f.read().strip()

    if not text:
        raise ValueError(f"Suite manifest is empty: {path}")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import yaml  # noqa

        data = yaml.safe_load(text)

    if isinstance(data, dict):
        suites = data.get("suites", [])
    elif isinstance(data, list):
        suites = data
    else:
        suites = []

    if not suites:
        raise ValueError(f"No suites found in manifest: {path}")

    normalized: List[Dict[str, Any]] = []
    for idx, suite in enumerate(suites, 1):
        if not isinstance(suite, dict):
            raise ValueError(f"Suite #{idx} must be a mapping.")
        name = str(suite.get("name", "")).strip()
        if not name:
            raise ValueError(f"Suite #{idx} missing required key 'name'.")
        normalized.append(suite)
    return normalized


def _to_gs_uri(path: str) -> str:
    if path.startswith("gs://"):
        return path
    return f"gs://{path}"


def _parse_gs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    no_scheme = uri[5:]
    parts = no_scheme.split("/", 1)
    bucket = parts[0]
    blob = parts[1] if len(parts) > 1 else ""
    return bucket, blob


def _extract_ckpt_sort_key(ckpt_uri: str) -> Tuple[int, float, str]:
    name = os.path.basename(ckpt_uri)
    step_match = re.search(r"step(\d+)", name)
    auc_match = re.search(r"_auc([0-9.]+)", name)
    step = int(step_match.group(1)) if step_match else -1
    auc = float(auc_match.group(1)) if auc_match else -1.0
    return step, auc, name


def _discover_checkpoints(prefix: str, pattern: str, max_checkpoints: int | None) -> List[str]:
    prefix = prefix.rstrip("/")
    if not prefix.startswith("gs://"):
        raise ValueError(f"checkpoint prefix must be gs:// URI, got: {prefix}")

    bucket_name, prefix_blob = _parse_gs_uri(prefix)
    # List everything under prefix and filter with fnmatch against basename.
    import fnmatch

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = client.list_blobs(bucket, prefix=prefix_blob.rstrip("/") + "/")

    paths: List[str] = []
    for blob in blobs:
        name = os.path.basename(blob.name)
        if fnmatch.fnmatch(name, pattern):
            paths.append(f"gs://{bucket_name}/{blob.name}")

    ckpts = sorted(paths, key=_extract_ckpt_sort_key)
    if max_checkpoints and max_checkpoints > 0:
        ckpts = ckpts[-max_checkpoints:]
    return ckpts


def _write_checkpoint_map(checkpoints: List[str]) -> Tuple[str, List[str], Dict[str, str]]:
    if len(checkpoints) > 8:
        raise ValueError(
            f"Found {len(checkpoints)} checkpoints but runner keys are limited to FT1..FT8. "
            "Use --max_checkpoints 8."
        )

    key_map: Dict[str, str] = {}
    keys: List[str] = []
    for idx, ckpt in enumerate(checkpoints, 1):
        key = f"FT{idx}"
        keys.append(key)
        key_map[key] = ckpt

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(key_map, tmp, indent=2)
    tmp.flush()
    tmp.close()
    return tmp.name, keys, key_map


def _run_validation_jobs(
    *,
    checkpoint_keys: List[str],
    checkpoint_map_path: str,
    suite_manifest: str,
    output_gcs_folder: str,
    wandb_project: str,
    frames_per_video: int,
    dry_run: bool,
) -> None:
    cmd = [
        sys.executable,
        "-u",
        "run_target_domain_validation_sequential.py",
        "--checkpoints",
        ",".join(checkpoint_keys),
        "--checkpoint_map",
        checkpoint_map_path,
        "--suite_manifest",
        suite_manifest,
        "--output_gcs_folder",
        output_gcs_folder,
        "--wandb_project",
        wandb_project,
        "--frames_per_video",
        str(frames_per_video),
        "--detailed_reports",
    ]
    if dry_run:
        cmd.append("--dry-run")

    print("Launching validation stage:")
    print(" ".join(cmd))
    if dry_run:
        return

    subprocess.run(cmd, cwd=_training_dir(), check=True)


@dataclass
class EvalRow:
    checkpoint_key: str
    checkpoint_path: str
    suite: str
    method: str
    label: int
    prob: float


def _read_videos_report(gcs_uri: str) -> List[Dict[str, str]]:
    bucket_name, blob_name = _parse_gs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not blob.exists(client=client):
        raise FileNotFoundError(f"Missing report in GCS: {gcs_uri}")
    text = blob.download_as_text()
    reader = csv.DictReader(text.splitlines())
    return list(reader)


def _load_eval_rows(
    *,
    suites: List[Dict[str, Any]],
    checkpoint_keys: List[str],
    checkpoint_map: Dict[str, str],
    output_gcs_folder: str,
) -> List[EvalRow]:
    rows: List[EvalRow] = []
    output_gcs_folder = output_gcs_folder.rstrip("/")
    for suite in suites:
        suite_name = str(suite["name"])
        for ckpt_key in checkpoint_keys:
            prefix = f"{suite_name}_{ckpt_key.lower()}_"
            report_uri = f"{output_gcs_folder}/{prefix}videos_report.csv"
            records = _read_videos_report(report_uri)
            for rec in records:
                try:
                    label = int(float(rec["label"]))
                    prob = float(rec["avg_video_prob"])
                    method = str(rec["method"])
                except Exception as exc:
                    raise ValueError(f"Malformed row in {report_uri}: {rec}") from exc
                rows.append(
                    EvalRow(
                        checkpoint_key=ckpt_key,
                        checkpoint_path=checkpoint_map[ckpt_key],
                        suite=suite_name,
                        method=method,
                        label=label,
                        prob=prob,
                    )
                )
    return rows


def _eer_threshold(y: np.ndarray, s: np.ndarray) -> float:
    # Local ROC implementation to avoid adding more dependencies
    # Steps:
    # 1) sort unique thresholds descending
    # 2) compute FPR/TPR for each threshold
    unique = np.unique(s)
    thresholds = np.concatenate(([math.inf], unique[::-1], [-math.inf]))
    best_thr = 0.5
    best_gap = float("inf")

    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    for thr in thresholds:
        pred = s >= thr
        tp = int(np.sum(pred & pos))
        fp = int(np.sum(pred & neg))
        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        fnr = 1.0 - tpr
        gap = abs(fpr - fnr)
        if gap < best_gap:
            best_gap = gap
            best_thr = float(thr if np.isfinite(thr) else 1.0)
    return best_thr


def _compute_threshold_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    method_keys: List[str],
    threshold: float,
) -> Dict[str, float]:
    pred = (probs >= threshold).astype(np.int32)

    real_mask = labels == 0
    fake_mask = labels == 1
    real_n = int(real_mask.sum())
    fake_n = int(fake_mask.sum())

    real_fpr = float(np.mean(pred[real_mask] == 1)) if real_n > 0 else float("nan")
    fake_tpr = float(np.mean(pred[fake_mask] == 1)) if fake_n > 0 else float("nan")
    real_tnr = 1.0 - real_fpr if np.isfinite(real_fpr) else float("nan")
    bal_acc = (
        float(np.nanmean([real_tnr, fake_tpr]))
        if np.isfinite(real_tnr) and np.isfinite(fake_tpr)
        else float("nan")
    )

    # Macro metrics across suite::method groups so no single source dominates.
    group_to_idx: Dict[str, List[int]] = {}
    for i, mk in enumerate(method_keys):
        group_to_idx.setdefault(mk, []).append(i)

    real_group_fprs: List[float] = []
    fake_group_tprs: List[float] = []
    for _, idxs in group_to_idx.items():
        g_labels = labels[idxs]
        g_pred = pred[idxs]
        if len(g_labels) == 0:
            continue
        if np.all(g_labels == 0):
            real_group_fprs.append(float(np.mean(g_pred == 1)))
        elif np.all(g_labels == 1):
            fake_group_tprs.append(float(np.mean(g_pred == 1)))
        else:
            # Mixed-label group should be rare with current reporting; skip for macro.
            continue

    macro_real_fpr = float(np.mean(real_group_fprs)) if real_group_fprs else float("nan")
    worst_real_fpr = float(np.max(real_group_fprs)) if real_group_fprs else float("nan")
    macro_fake_tpr = float(np.mean(fake_group_tprs)) if fake_group_tprs else float("nan")
    worst_fake_tpr = float(np.min(fake_group_tprs)) if fake_group_tprs else float("nan")

    return {
        "threshold": float(threshold),
        "n_total": int(labels.size),
        "n_real": real_n,
        "n_fake": fake_n,
        "real_fpr": real_fpr,
        "fake_tpr": fake_tpr,
        "balanced_acc": bal_acc,
        "macro_real_fpr": macro_real_fpr,
        "worst_real_fpr": worst_real_fpr,
        "macro_fake_tpr": macro_fake_tpr,
        "worst_fake_tpr": worst_fake_tpr,
    }


def _select_best_threshold(
    metrics_grid: List[Dict[str, float]],
    min_macro_fake_tpr: float,
    min_worst_fake_tpr: float,
    fallback_lambda: float,
) -> Dict[str, float]:
    feasible = [
        m
        for m in metrics_grid
        if np.isfinite(m["macro_fake_tpr"])
        and np.isfinite(m["macro_real_fpr"])
        and m["macro_fake_tpr"] >= min_macro_fake_tpr
        and m["worst_fake_tpr"] >= min_worst_fake_tpr
    ]
    if feasible:
        feasible.sort(
            key=lambda m: (
                m["macro_real_fpr"],
                m["worst_real_fpr"],
                -m["macro_fake_tpr"],
                -m["worst_fake_tpr"],
            )
        )
        best = dict(feasible[0])
        best["selection_mode"] = "feasible_min_fpr"
        best["feasible_count"] = len(feasible)
        return best

    # Fallback utility if constraints are too strict.
    scored = []
    for m in metrics_grid:
        if not np.isfinite(m["macro_fake_tpr"]) or not np.isfinite(m["macro_real_fpr"]):
            continue
        utility = m["macro_fake_tpr"] - fallback_lambda * m["macro_real_fpr"]
        mm = dict(m)
        mm["utility"] = float(utility)
        scored.append(mm)

    if not scored:
        raise RuntimeError("No valid threshold candidates produced.")

    scored.sort(key=lambda m: (-m["utility"], m["macro_real_fpr"], -m["macro_fake_tpr"]))
    best = dict(scored[0])
    best["selection_mode"] = "fallback_utility"
    best["feasible_count"] = 0
    return best


def _write_local_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _upload_file_to_gcs(local_path: str, gcs_uri: str) -> None:
    bucket_name, blob_name = _parse_gs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)


def _run_analysis(
    *,
    eval_rows: List[EvalRow],
    checkpoint_keys: List[str],
    suites: List[Dict[str, Any]],
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    min_macro_fake_tpr: float,
    min_worst_fake_tpr: float,
    fallback_lambda: float,
) -> Dict[str, Any]:
    by_ckpt: Dict[str, List[EvalRow]] = {}
    for r in eval_rows:
        by_ckpt.setdefault(r.checkpoint_key, []).append(r)

    candidate_rows: List[Dict[str, Any]] = []
    suite_eer_rows: List[Dict[str, Any]] = []
    top_grid_rows: List[Dict[str, Any]] = []

    thresholds = np.arange(threshold_min, threshold_max + 1e-12, threshold_step)
    thresholds = np.clip(thresholds, 0.0, 1.0)
    thresholds = np.unique(np.round(thresholds, 6))

    for ckpt_key in checkpoint_keys:
        rows = by_ckpt.get(ckpt_key, [])
        if not rows:
            continue

        labels = np.array([r.label for r in rows], dtype=np.int32)
        probs = np.array([r.prob for r in rows], dtype=np.float64)
        method_keys = [f"{r.suite}::{r.method}" for r in rows]

        # Per-suite EER thresholds
        for suite in suites:
            suite_name = str(suite["name"])
            suite_rows = [r for r in rows if r.suite == suite_name]
            if not suite_rows:
                continue
            y = np.array([r.label for r in suite_rows], dtype=np.int32)
            s = np.array([r.prob for r in suite_rows], dtype=np.float64)
            eer_thr = _eer_threshold(y, s) if len(np.unique(y)) > 1 else float("nan")
            suite_eer_rows.append(
                {
                    "checkpoint_key": ckpt_key,
                    "suite": suite_name,
                    "n_samples": int(y.size),
                    "n_real": int(np.sum(y == 0)),
                    "n_fake": int(np.sum(y == 1)),
                    "eer_threshold": float(eer_thr) if np.isfinite(eer_thr) else "",
                }
            )

        grid = [
            _compute_threshold_metrics(labels=labels, probs=probs, method_keys=method_keys, threshold=float(t))
            for t in thresholds
        ]
        best = _select_best_threshold(
            metrics_grid=grid,
            min_macro_fake_tpr=min_macro_fake_tpr,
            min_worst_fake_tpr=min_worst_fake_tpr,
            fallback_lambda=fallback_lambda,
        )
        best["checkpoint_key"] = ckpt_key
        best["checkpoint_path"] = rows[0].checkpoint_path
        best["global_eer_threshold"] = _eer_threshold(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
        candidate_rows.append(best)

        # Keep top threshold rows for debugging (lowest macro FPR among strong fake TPR first).
        sortable = [
            m
            for m in grid
            if np.isfinite(m["macro_real_fpr"]) and np.isfinite(m["macro_fake_tpr"])
        ]
        sortable.sort(
            key=lambda m: (
                m["macro_real_fpr"],
                m["worst_real_fpr"],
                -m["macro_fake_tpr"],
                -m["worst_fake_tpr"],
            )
        )
        for m in sortable[:20]:
            row = dict(m)
            row["checkpoint_key"] = ckpt_key
            row["checkpoint_path"] = rows[0].checkpoint_path
            top_grid_rows.append(row)

    if not candidate_rows:
        raise RuntimeError("No checkpoint candidates were analyzed.")

    # Pick best checkpoint by FP-first policy (with fake-TPR constraints already baked in).
    candidate_rows.sort(
        key=lambda r: (
            r["macro_real_fpr"],
            r["worst_real_fpr"],
            -r["macro_fake_tpr"],
            -r["worst_fake_tpr"],
        )
    )
    winner = candidate_rows[0]

    return {
        "winner": winner,
        "checkpoint_candidates": candidate_rows,
        "suite_eer_thresholds": suite_eer_rows,
        "top_threshold_grid_rows": top_grid_rows,
    }


def main() -> None:
    full_suite_manifest_default = "experiments/phase2_round6/R6_THRESHOLD_SWEEP_SUITES.yaml"
    smoke_suite_manifest_default = "experiments/phase2_round6/R6_THRESHOLD_SWEEP_SUITES_SMOKE.yaml"

    parser = argparse.ArgumentParser(description="R6 checkpoint + threshold overnight sweep")
    parser.add_argument(
        "--checkpoint_prefix_gcs",
        type=str,
        required=True,
        help="GCS prefix containing checkpoint files, e.g. gs://.../phase2r6_experiments/<run_id>",
    )
    parser.add_argument(
        "--checkpoint_pattern",
        type=str,
        default="top_n_effort_*.pth",
        help="Glob pattern under checkpoint prefix (default top_n checkpoints only).",
    )
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=8,
        help="Max number of checkpoints to evaluate (must be <= 8 with current FT keying).",
    )
    parser.add_argument(
        "--suite_manifest",
        type=str,
        default=full_suite_manifest_default,
        help="Path to suite manifest used by run_target_domain_validation_sequential.py",
    )
    parser.add_argument(
        "--smoke_suite_manifest",
        type=str,
        default=smoke_suite_manifest_default,
        help="Suite manifest used when --smoke is enabled and --suite_manifest is left at default.",
    )
    parser.add_argument(
        "--output_gcs_folder",
        type=str,
        default="",
        help="Where validation reports and analysis outputs are written. "
        "If empty, auto-creates under gs://training-job-outputs/test_results/",
    )
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "phase2-round6"))
    parser.add_argument("--frames_per_video", type=int, default=8)

    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.99)
    parser.add_argument("--threshold_step", type=float, default=0.002)
    parser.add_argument("--min_macro_fake_tpr", type=float, default=0.85)
    parser.add_argument("--min_worst_fake_tpr", type=float, default=0.65)
    parser.add_argument("--fallback_lambda", type=float, default=2.0)

    parser.add_argument("--skip_validation", action="store_true", default=False)
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="Fast preflight mode: uses smoke suites and lighter defaults unless explicitly overridden.",
    )
    parser.add_argument("--dry_run", action="store_true", default=False)
    args = parser.parse_args()

    if args.smoke:
        # Swap to smoke suites only if caller did not explicitly set a custom manifest.
        if args.suite_manifest == full_suite_manifest_default:
            args.suite_manifest = args.smoke_suite_manifest
        # Keep smoke quick unless caller already tightened this.
        if args.max_checkpoints == 8:
            args.max_checkpoints = 1
        if abs(args.threshold_step - 0.002) < 1e-12:
            args.threshold_step = 0.02
        if abs(args.min_macro_fake_tpr - 0.85) < 1e-12:
            args.min_macro_fake_tpr = 0.70
        if abs(args.min_worst_fake_tpr - 0.65) < 1e-12:
            args.min_worst_fake_tpr = 0.50

    if args.max_checkpoints < 1 or args.max_checkpoints > 8:
        raise ValueError("--max_checkpoints must be in [1, 8]")
    if not (0.0 <= args.threshold_min < args.threshold_max <= 1.0):
        raise ValueError("Invalid threshold range")
    if args.threshold_step <= 0:
        raise ValueError("--threshold_step must be > 0")

    checkpoint_prefix_gcs = args.checkpoint_prefix_gcs.rstrip("/")
    suite_manifest = args.suite_manifest
    if not os.path.isabs(suite_manifest):
        suite_manifest = os.path.join(_training_dir(), suite_manifest)

    suites = _load_manifest(suite_manifest)
    checkpoints = _discover_checkpoints(
        prefix=checkpoint_prefix_gcs,
        pattern=args.checkpoint_pattern,
        max_checkpoints=args.max_checkpoints,
    )
    if not checkpoints:
        raise RuntimeError(
            f"No checkpoints found at {checkpoint_prefix_gcs} with pattern {args.checkpoint_pattern}"
        )

    ts = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_id = checkpoint_prefix_gcs.rstrip("/").split("/")[-1]
    output_gcs_folder = args.output_gcs_folder.strip()
    if not output_gcs_folder:
        output_gcs_folder = (
            f"gs://training-job-outputs/test_results/"
            f"r6_ckpt_threshold_sweep_{run_id}_{ts}"
        )

    ckpt_map_path, checkpoint_keys, checkpoint_map = _write_checkpoint_map(checkpoints)
    print("Discovered checkpoints:")
    for key in checkpoint_keys:
        print(f"  {key}: {checkpoint_map[key]}")
    print(f"Suite manifest: {suite_manifest}")
    print(f"Output folder: {output_gcs_folder}")
    print(f"W&B project: {args.wandb_project}")

    try:
        if not args.skip_validation:
            _run_validation_jobs(
                checkpoint_keys=checkpoint_keys,
                checkpoint_map_path=ckpt_map_path,
                suite_manifest=suite_manifest,
                output_gcs_folder=output_gcs_folder,
                wandb_project=args.wandb_project,
                frames_per_video=args.frames_per_video,
                dry_run=args.dry_run,
            )
        elif args.dry_run:
            print("skip_validation + dry_run: skipping execution")

        if args.dry_run:
            return

        eval_rows = _load_eval_rows(
            suites=suites,
            checkpoint_keys=checkpoint_keys,
            checkpoint_map=checkpoint_map,
            output_gcs_folder=output_gcs_folder,
        )
        analysis = _run_analysis(
            eval_rows=eval_rows,
            checkpoint_keys=checkpoint_keys,
            suites=suites,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
            min_macro_fake_tpr=args.min_macro_fake_tpr,
            min_worst_fake_tpr=args.min_worst_fake_tpr,
            fallback_lambda=args.fallback_lambda,
        )

        local_out_dir = os.path.join(_training_dir(), "analysis_results", f"r6_ckpt_threshold_sweep_{run_id}_{ts}")
        os.makedirs(local_out_dir, exist_ok=True)

        candidates_csv = os.path.join(local_out_dir, "checkpoint_candidates.csv")
        suite_eer_csv = os.path.join(local_out_dir, "suite_eer_thresholds.csv")
        grid_csv = os.path.join(local_out_dir, "threshold_grid_top20_per_checkpoint.csv")
        summary_json = os.path.join(local_out_dir, "winner_summary.json")

        _write_local_csv(candidates_csv, analysis["checkpoint_candidates"])
        _write_local_csv(suite_eer_csv, analysis["suite_eer_thresholds"])
        _write_local_csv(grid_csv, analysis["top_threshold_grid_rows"])
        with open(summary_json, "w") as f:
            json.dump(analysis["winner"], f, indent=2)

        # Upload analysis artifacts
        analysis_gcs_dir = output_gcs_folder.rstrip("/") + "/analysis"
        _upload_file_to_gcs(candidates_csv, analysis_gcs_dir + "/checkpoint_candidates.csv")
        _upload_file_to_gcs(suite_eer_csv, analysis_gcs_dir + "/suite_eer_thresholds.csv")
        _upload_file_to_gcs(grid_csv, analysis_gcs_dir + "/threshold_grid_top20_per_checkpoint.csv")
        _upload_file_to_gcs(summary_json, analysis_gcs_dir + "/winner_summary.json")

        winner = analysis["winner"]
        print("\n=== WINNER (FP-first policy) ===")
        print(f"Checkpoint key: {winner['checkpoint_key']}")
        print(f"Checkpoint path: {winner['checkpoint_path']}")
        print(f"Recommended threshold: {winner['threshold']:.4f}")
        print(f"Macro real FPR: {winner['macro_real_fpr']:.4f}")
        print(f"Worst real FPR: {winner['worst_real_fpr']:.4f}")
        print(f"Macro fake TPR: {winner['macro_fake_tpr']:.4f}")
        print(f"Worst fake TPR: {winner['worst_fake_tpr']:.4f}")
        print(f"Selection mode: {winner.get('selection_mode')}")
        print(f"\nAnalysis uploaded to: {analysis_gcs_dir}")

    finally:
        try:
            os.remove(ckpt_map_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
