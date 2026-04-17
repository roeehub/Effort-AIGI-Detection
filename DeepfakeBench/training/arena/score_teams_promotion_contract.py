#!/usr/bin/env python3
"""
Score the threshold-calibrated Teams promotion contract from video-report artifacts.

This script consumes ``*_videos_report.csv`` files emitted by
``arena/run_target_domain_validation_sequential.py``. For each checkpoint it:

1. Loads the required dev and lockbox suites from an existing reports root.
2. Sweeps candidate thresholds on the dev suites only.
3. Selects the best threshold with the documented low-FP lexicographic policy:
   - minimize ``teams_real_all_dev`` FPR
   - then minimize the worst FPR across the real stress dev suites
   - then maximize fake recall on the dev fake suites
4. Freezes that threshold.
5. Reads out lockbox real/fake metrics at the frozen threshold.

The output is a compact promotion scorecard plus the threshold grid used to
select each checkpoint's frozen threshold.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _split_gs_uri(uri: str) -> Tuple[str, str]:
    stripped = uri.replace("gs://", "", 1)
    if "/" not in stripped:
        return stripped, ""
    bucket, blob = stripped.split("/", 1)
    return bucket, blob


def _read_text_from_path(path: str) -> str:
    if path.startswith("gs://"):
        try:
            from google.cloud import storage
        except ImportError as exc:  # pragma: no cover - depends on runtime image.
            raise RuntimeError(
                "Reading gs:// paths requires google-cloud-storage in this runtime."
            ) from exc

        bucket_name, blob_path = _split_gs_uri(path)
        client = storage.Client()
        return client.bucket(bucket_name).blob(blob_path).download_as_text()

    return Path(path).read_text()


def _write_text_to_path(path: str, text: str) -> None:
    if path.startswith("gs://"):
        try:
            from google.cloud import storage
        except ImportError as exc:  # pragma: no cover - depends on runtime image.
            raise RuntimeError(
                "Writing gs:// paths requires google-cloud-storage in this runtime."
            ) from exc

        bucket_name, blob_path = _split_gs_uri(path)
        client = storage.Client()
        client.bucket(bucket_name).blob(blob_path).upload_from_string(
            text,
            content_type="text/plain; charset=utf-8",
        )
        return

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)


def _join_path(root: str, filename: str) -> str:
    if root.startswith("gs://"):
        return f"{root.rstrip('/')}/{filename}"
    return str(Path(root) / filename)


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    text = _read_text_from_path(path)
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _write_dict_rows_to_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for CSV export: {path}")

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    _write_text_to_path(path, buffer.getvalue())


def _csv_list(value: str | None) -> Tuple[str, ...]:
    if value is None:
        return ()
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _parse_flat_key_value_mapping(text: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key_text = key.strip()
        value_text = value.strip().strip('"').strip("'")
        if key_text and value_text:
            result[key_text] = value_text
    return result


def _load_checkpoint_map(path: str) -> Dict[str, str]:
    text = _read_text_from_path(path).strip()
    if not text:
        return {}

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError:
            data = _parse_flat_key_value_mapping(text)
        else:
            data = yaml.safe_load(text) or {}

    result: Dict[str, str] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            key_text = str(key).strip().upper()
            value_text = str(value).strip()
            if key_text and value_text:
                result[key_text] = value_text
    return result


def _resolve_requested_checkpoint_keys(
    checkpoints_arg: str,
    checkpoint_map_path: str,
) -> List[str]:
    checkpoint_map = _load_checkpoint_map(checkpoint_map_path)
    available = sorted(checkpoint_map.keys())
    requested = [part.strip().upper() for part in str(checkpoints_arg).split(",") if part.strip()]
    if not requested:
        raise ValueError("No checkpoints requested.")
    if len(requested) == 1 and requested[0] == "ALL":
        if not available:
            raise ValueError(f"Checkpoint map is empty: {checkpoint_map_path}")
        return available
    return requested


def _resolve_checkpoints(selected_keys: Sequence[str], checkpoint_map_path: str) -> Dict[str, str]:
    checkpoint_map = _load_checkpoint_map(checkpoint_map_path)
    resolved: Dict[str, str] = {}
    missing: List[str] = []
    for key in selected_keys:
        key_text = str(key).strip().upper()
        path = checkpoint_map.get(key_text, "")
        if path:
            resolved[key_text] = path
        else:
            missing.append(key_text)
    if missing:
        raise ValueError(
            "Missing checkpoint path(s) for: "
            + ", ".join(missing)
            + f". Check checkpoint map: {checkpoint_map_path}"
        )
    return resolved


def _report_path_for_job(report_root: str, checkpoint_key: str, suite_name: str) -> str:
    filename = f"{suite_name}_{checkpoint_key.lower()}_videos_report.csv"
    return _join_path(report_root, filename)


def _parse_label(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"0", "real", "false", "negative"}:
        return 0
    if text in {"1", "fake", "true", "positive"}:
        return 1
    raise ValueError(f"Unsupported label value: {value!r}")


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * q
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]

    lower = ordered[lower_index]
    upper = ordered[upper_index]
    fraction = position - lower_index
    return lower + (upper - lower) * fraction


def _mean(values: Iterable[float]) -> float:
    ordered = [float(value) for value in values]
    if not ordered:
        return 0.0
    return sum(ordered) / float(len(ordered))


def _sort_number(value: Any, *, higher_is_better: bool = False) -> float:
    if value is None:
        return float("inf")
    try:
        number = float(value)
    except Exception:
        return float("inf")
    if not math.isfinite(number):
        return float("inf")
    return -number if higher_is_better else number


@dataclass(frozen=True)
class ContractConfig:
    dev_real_suite: str
    dev_real_stress_suites: Tuple[str, ...]
    dev_fake_suites: Tuple[str, ...]
    lockbox_real_suite: str
    lockbox_fake_suite: str

    def all_suites(self) -> Tuple[str, ...]:
        ordered: List[str] = []
        for suite_name in (
            self.dev_real_suite,
            *self.dev_real_stress_suites,
            *self.dev_fake_suites,
            self.lockbox_real_suite,
            self.lockbox_fake_suite,
        ):
            if suite_name and suite_name not in ordered:
                ordered.append(suite_name)
        return tuple(ordered)


@dataclass(frozen=True)
class VideoScore:
    label: int
    prob: float
    method: str
    video_id: str


def _load_video_scores(report_path: str) -> List[VideoScore]:
    rows = _read_csv_rows(report_path)
    parsed: List[VideoScore] = []
    for row in rows:
        parsed.append(
            VideoScore(
                label=_parse_label(row.get("label", "")),
                prob=float(str(row.get("avg_video_prob", "0")).strip() or "0"),
                method=str(row.get("method", "")).strip(),
                video_id=str(row.get("video_id", "")).strip(),
            )
        )
    if not parsed:
        raise ValueError(f"Video report contains no rows: {report_path}")
    return parsed


def _suite_metrics(rows: Sequence[VideoScore], threshold: float) -> Dict[str, Any]:
    probs = [float(row.prob) for row in rows]
    labels = [int(row.label) for row in rows]
    predictions = [1 if prob >= threshold else 0 for prob in probs]

    tn = fp = fn = tp = 0
    for label, prediction in zip(labels, predictions):
        if label == 0 and prediction == 0:
            tn += 1
        elif label == 0 and prediction == 1:
            fp += 1
        elif label == 1 and prediction == 0:
            fn += 1
        elif label == 1 and prediction == 1:
            tp += 1

    n_videos = len(rows)
    n_real = tn + fp
    n_fake = tp + fn
    accuracy = (tn + tp) / n_videos if n_videos > 0 else 0.0
    real_fpr = (fp / n_real) if n_real > 0 else None
    real_tnr = (tn / n_real) if n_real > 0 else None
    fake_recall = (tp / n_fake) if n_fake > 0 else None
    fake_fnr = (fn / n_fake) if n_fake > 0 else None

    return {
        "threshold": round(float(threshold), 6),
        "n_videos": int(n_videos),
        "n_real": int(n_real),
        "n_fake": int(n_fake),
        "accuracy": round(float(accuracy), 6),
        "mean_prob": round(float(_mean(probs)), 6),
        "p50_prob": round(float(_quantile(probs, 0.50)), 6),
        "p90_prob": round(float(_quantile(probs, 0.90)), 6),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "real_fpr": None if real_fpr is None else round(float(real_fpr), 6),
        "real_tnr": None if real_tnr is None else round(float(real_tnr), 6),
        "fake_recall": None if fake_recall is None else round(float(fake_recall), 6),
        "fake_fnr": None if fake_fnr is None else round(float(fake_fnr), 6),
    }


def _require_real_metric(metrics: Dict[str, Any], suite_name: str, checkpoint_key: str) -> float:
    value = metrics.get("real_fpr")
    if value is None:
        raise ValueError(
            f"Suite {suite_name} for checkpoint {checkpoint_key} has no real examples; "
            "expected a real-only or mixed real-bearing suite."
        )
    return float(value)


def _require_fake_metric(metrics: Dict[str, Any], suite_name: str, checkpoint_key: str) -> float:
    value = metrics.get("fake_recall")
    if value is None:
        raise ValueError(
            f"Suite {suite_name} for checkpoint {checkpoint_key} has no fake examples; "
            "expected a fake-only or mixed fake-bearing suite."
        )
    return float(value)


def _candidate_thresholds(report_rows_by_suite: Dict[str, List[VideoScore]], contract: ContractConfig) -> List[float]:
    values = {0.0, 1.0}
    dev_suites = [
        contract.dev_real_suite,
        *contract.dev_real_stress_suites,
        *contract.dev_fake_suites,
    ]
    for suite_name in dev_suites:
        rows = report_rows_by_suite.get(suite_name, [])
        for row in rows:
            values.add(round(float(row.prob), 8))
    return sorted(values)


def _build_threshold_grid_row(
    checkpoint_key: str,
    checkpoint_path: str,
    threshold: float,
    report_rows_by_suite: Dict[str, List[VideoScore]],
    contract: ContractConfig,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "checkpoint_key": checkpoint_key,
        "checkpoint_path": checkpoint_path,
        "threshold": round(float(threshold), 6),
    }

    stress_fprs: List[float] = []
    fake_recalls: List[float] = []

    dev_suite_names = (
        contract.dev_real_suite,
        *contract.dev_real_stress_suites,
        *contract.dev_fake_suites,
    )
    for suite_name in dev_suite_names:
        metrics = _suite_metrics(report_rows_by_suite[suite_name], threshold)
        row[f"{suite_name}__n_videos"] = metrics["n_videos"]
        row[f"{suite_name}__real_fpr"] = metrics["real_fpr"]
        row[f"{suite_name}__fake_recall"] = metrics["fake_recall"]

    primary_metrics = _suite_metrics(report_rows_by_suite[contract.dev_real_suite], threshold)
    row["dev_primary_real_fpr"] = _require_real_metric(primary_metrics, contract.dev_real_suite, checkpoint_key)

    for suite_name in contract.dev_real_stress_suites:
        metrics = _suite_metrics(report_rows_by_suite[suite_name], threshold)
        stress_fprs.append(_require_real_metric(metrics, suite_name, checkpoint_key))
    row["dev_worst_real_stress_fpr"] = round(max(stress_fprs), 6) if stress_fprs else None

    for suite_name in contract.dev_fake_suites:
        metrics = _suite_metrics(report_rows_by_suite[suite_name], threshold)
        fake_recalls.append(_require_fake_metric(metrics, suite_name, checkpoint_key))
    row["dev_fake_macro_recall"] = round(_mean(fake_recalls), 6) if fake_recalls else None

    return row


def _threshold_sort_key(row: Dict[str, Any], contract: ContractConfig) -> Tuple[float, ...]:
    return (
        _sort_number(row.get("dev_primary_real_fpr")),
        _sort_number(row.get("dev_worst_real_stress_fpr")),
        *[
            _sort_number(row.get(f"{suite_name}__fake_recall"), higher_is_better=True)
            for suite_name in contract.dev_fake_suites
        ],
        _sort_number(row.get("threshold"), higher_is_better=True),
    )


def _promotion_summary_sort_key(row: Dict[str, Any], contract: ContractConfig) -> Tuple[float, ...]:
    return (
        _sort_number(row.get("lockbox_real_fpr")),
        _sort_number(row.get("lockbox_fake_recall"), higher_is_better=True),
        _sort_number(row.get("dev_primary_real_fpr")),
        _sort_number(row.get("dev_worst_real_stress_fpr")),
        *[
            _sort_number(row.get(f"{suite_name}__fake_recall"), higher_is_better=True)
            for suite_name in contract.dev_fake_suites
        ],
        _sort_number(row.get("selected_threshold"), higher_is_better=True),
    )


def _build_selected_threshold_scorecard_rows(
    checkpoint_key: str,
    checkpoint_path: str,
    selected_threshold: float,
    report_rows_by_suite: Dict[str, List[VideoScore]],
    report_paths_by_suite: Dict[str, str],
    contract: ContractConfig,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for suite_name in contract.all_suites():
        metrics = _suite_metrics(report_rows_by_suite[suite_name], selected_threshold)
        rows.append(
            {
                "checkpoint_key": checkpoint_key,
                "checkpoint_path": checkpoint_path,
                "suite_name": suite_name,
                "report_path": report_paths_by_suite[suite_name],
                **metrics,
            }
        )
    return rows


def score_promotion_contract(
    *,
    report_root: str,
    checkpoint_map_path: str,
    checkpoints_arg: str,
    contract: ContractConfig,
) -> Dict[str, Any]:
    selected_keys = _resolve_requested_checkpoint_keys(checkpoints_arg, checkpoint_map_path)
    checkpoints = _resolve_checkpoints(selected_keys, checkpoint_map_path)

    threshold_grid_rows: List[Dict[str, Any]] = []
    selected_threshold_scorecard_rows: List[Dict[str, Any]] = []
    checkpoint_summary_rows: List[Dict[str, Any]] = []

    for checkpoint_key in selected_keys:
        checkpoint_path = checkpoints[checkpoint_key]
        report_rows_by_suite: Dict[str, List[VideoScore]] = {}
        report_paths_by_suite: Dict[str, str] = {}

        for suite_name in contract.all_suites():
            report_path = _report_path_for_job(report_root, checkpoint_key, suite_name)
            report_paths_by_suite[suite_name] = report_path
            report_rows_by_suite[suite_name] = _load_video_scores(report_path)

        candidate_thresholds = _candidate_thresholds(report_rows_by_suite, contract)
        checkpoint_grid_rows = [
            _build_threshold_grid_row(
                checkpoint_key=checkpoint_key,
                checkpoint_path=checkpoint_path,
                threshold=threshold,
                report_rows_by_suite=report_rows_by_suite,
                contract=contract,
            )
            for threshold in candidate_thresholds
        ]
        checkpoint_grid_rows.sort(key=lambda row: _threshold_sort_key(row, contract))
        best_threshold_row = checkpoint_grid_rows[0]
        selected_threshold = float(best_threshold_row["threshold"])

        threshold_grid_rows.extend(checkpoint_grid_rows)

        suite_rows = _build_selected_threshold_scorecard_rows(
            checkpoint_key=checkpoint_key,
            checkpoint_path=checkpoint_path,
            selected_threshold=selected_threshold,
            report_rows_by_suite=report_rows_by_suite,
            report_paths_by_suite=report_paths_by_suite,
            contract=contract,
        )
        selected_threshold_scorecard_rows.extend(suite_rows)

        suite_row_map = {row["suite_name"]: row for row in suite_rows}
        lockbox_real_row = suite_row_map[contract.lockbox_real_suite]
        lockbox_fake_row = suite_row_map[contract.lockbox_fake_suite]

        summary_row: Dict[str, Any] = {
            "checkpoint_key": checkpoint_key,
            "checkpoint_path": checkpoint_path,
            "selected_threshold": round(selected_threshold, 6),
            "threshold_candidate_count": len(candidate_thresholds),
            "dev_primary_real_fpr": best_threshold_row["dev_primary_real_fpr"],
            "dev_worst_real_stress_fpr": best_threshold_row["dev_worst_real_stress_fpr"],
            "dev_fake_macro_recall": best_threshold_row["dev_fake_macro_recall"],
            "lockbox_real_suite": contract.lockbox_real_suite,
            "lockbox_real_n_videos": lockbox_real_row["n_videos"],
            "lockbox_real_fpr": _require_real_metric(
                lockbox_real_row,
                contract.lockbox_real_suite,
                checkpoint_key,
            ),
            "lockbox_fake_suite": contract.lockbox_fake_suite,
            "lockbox_fake_n_videos": lockbox_fake_row["n_videos"],
            "lockbox_fake_recall": _require_fake_metric(
                lockbox_fake_row,
                contract.lockbox_fake_suite,
                checkpoint_key,
            ),
        }
        for suite_name in contract.dev_fake_suites:
            summary_row[f"{suite_name}__fake_recall"] = best_threshold_row[f"{suite_name}__fake_recall"]
        checkpoint_summary_rows.append(summary_row)

    checkpoint_summary_rows.sort(key=lambda row: _promotion_summary_sort_key(row, contract))
    for rank, row in enumerate(checkpoint_summary_rows, 1):
        row["promotion_rank"] = rank

    winner = checkpoint_summary_rows[0] if checkpoint_summary_rows else None
    return {
        "contract": {
            "dev_real_suite": contract.dev_real_suite,
            "dev_real_stress_suites": list(contract.dev_real_stress_suites),
            "dev_fake_suites": list(contract.dev_fake_suites),
            "lockbox_real_suite": contract.lockbox_real_suite,
            "lockbox_fake_suite": contract.lockbox_fake_suite,
        },
        "threshold_grid_rows": threshold_grid_rows,
        "selected_threshold_scorecard_rows": selected_threshold_scorecard_rows,
        "checkpoint_summary_rows": checkpoint_summary_rows,
        "winner": winner,
    }


def write_promotion_contract_outputs(output_root: str, payload: Dict[str, Any]) -> Dict[str, str]:
    output_root = str(output_root).rstrip("/")
    grid_csv = _join_path(output_root, "threshold_grid.csv")
    scorecard_csv = _join_path(output_root, "selected_threshold_scorecard.csv")
    summary_csv = _join_path(output_root, "checkpoint_summary.csv")
    payload_json = _join_path(output_root, "promotion_contract.json")
    winner_json = _join_path(output_root, "promotion_winner.json")

    _write_dict_rows_to_csv(grid_csv, payload["threshold_grid_rows"])
    _write_dict_rows_to_csv(scorecard_csv, payload["selected_threshold_scorecard_rows"])
    _write_dict_rows_to_csv(summary_csv, payload["checkpoint_summary_rows"])
    _write_text_to_path(payload_json, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_text_to_path(
        winner_json,
        json.dumps(
            {
                "contract": payload["contract"],
                "winner": payload["winner"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    return {
        "threshold_grid_csv": grid_csv,
        "selected_threshold_scorecard_csv": scorecard_csv,
        "checkpoint_summary_csv": summary_csv,
        "promotion_contract_json": payload_json,
        "promotion_winner_json": winner_json,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score the calibrated low-FP Teams promotion contract from video reports"
    )
    parser.add_argument(
        "--report_root",
        required=True,
        help=(
            "Directory or gs:// prefix containing "
            "<suite>_<checkpoint>_videos_report.csv artifacts."
        ),
    )
    parser.add_argument("--checkpoint_map", required=True, help="YAML/JSON checkpoint alias map.")
    parser.add_argument(
        "--checkpoints",
        default="ALL",
        help="Comma-separated checkpoint aliases to score (default: ALL).",
    )
    parser.add_argument("--output_dir", required=True, help="Local or gs:// output directory.")
    parser.add_argument("--dev_real_suite", default="teams_real_all_dev")
    parser.add_argument(
        "--dev_real_stress_suites",
        default="teams_real_poor_quality_dev,teams_real_lighting_extreme_dev",
    )
    parser.add_argument(
        "--dev_fake_suites",
        default="teams_fake_all_dev,visomaster_enhanced_macro_dev,deeplive_enhanced_dev",
    )
    parser.add_argument("--lockbox_real_suite", default="teams_real_all_lockbox")
    parser.add_argument("--lockbox_fake_suite", default="teams_fake_all_lockbox")
    args = parser.parse_args()

    contract = ContractConfig(
        dev_real_suite=str(args.dev_real_suite).strip(),
        dev_real_stress_suites=_csv_list(args.dev_real_stress_suites),
        dev_fake_suites=_csv_list(args.dev_fake_suites),
        lockbox_real_suite=str(args.lockbox_real_suite).strip(),
        lockbox_fake_suite=str(args.lockbox_fake_suite).strip(),
    )

    payload = score_promotion_contract(
        report_root=str(args.report_root).strip(),
        checkpoint_map_path=str(args.checkpoint_map).strip(),
        checkpoints_arg=str(args.checkpoints).strip(),
        contract=contract,
    )

    output_paths = write_promotion_contract_outputs(str(args.output_dir).strip(), payload)

    winner = payload["winner"]
    print("Teams promotion contract scored.")
    print(f"  threshold_grid.csv            : {output_paths['threshold_grid_csv']}")
    print(
        "  selected_threshold_scorecard.csv : "
        f"{output_paths['selected_threshold_scorecard_csv']}"
    )
    print(f"  checkpoint_summary.csv        : {output_paths['checkpoint_summary_csv']}")
    print(f"  promotion_contract.json       : {output_paths['promotion_contract_json']}")
    print(f"  promotion_winner.json         : {output_paths['promotion_winner_json']}")
    if winner:
        print(
            "  winner                        : "
            f"{winner['checkpoint_key']} "
            f"(lockbox_real_fpr={winner['lockbox_real_fpr']}, "
            f"lockbox_fake_recall={winner['lockbox_fake_recall']}, "
            f"threshold={winner['selected_threshold']})"
        )


if __name__ == "__main__":
    main()
