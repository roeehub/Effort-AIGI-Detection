"""
Sequential target-domain validation runner.

Runs validate_custom_sources.py for selected checkpoints across a configurable
set of target-domain suites (Zoom-like real/fake buckets, stress variants, etc).

The optional scorecard exports from this runner are fixed-threshold (`0.5`)
diagnostic artifacts. Promotion decisions should use the calibrated contract
implemented in ``arena/score_teams_promotion_contract.py``.

Suite manifest format (JSON or YAML):

suites:
  - name: zoom_real
    df40_mode: none
    external_real_bucket: effort-collected-data/real/zoom_real
    external_real_method: zoom_real
    max_external_real: 2000

  - name: zoom_wma_flat
    df40_mode: none
    external_fake_bucket: effort-collected-data
    external_fake_prefix: wma_validation/enhanced_fake
    external_fake_method: wma_failure_fake
    external_fake_grouping: per_image
    external_fake_deterministic: true
    max_external_fake: 1202
"""

from __future__ import annotations

import argparse
import csv
import io
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

RUNNER_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = RUNNER_DIR.parent
VALIDATE_CUSTOM_SOURCES_SCRIPT = TRAINING_ROOT / "validate_custom_sources.py"


def _split_gs_uri(uri: str) -> Tuple[str, str]:
    stripped = uri.replace("gs://", "", 1)
    if "/" not in stripped:
        return stripped, ""
    return tuple(stripped.split("/", 1))


def _read_text_from_path(path: str) -> str:
    if path.startswith("gs://"):
        from google.cloud import storage

        bucket_name, blob_path = _split_gs_uri(path)
        client = storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))
        return client.bucket(bucket_name).blob(blob_path).download_as_text()

    with open(path, "r") as f:
        return f.read()


def _write_text_to_path(path: str, text: str) -> None:
    if path.startswith("gs://"):
        from google.cloud import storage

        bucket_name, blob_path = _split_gs_uri(path)
        client = storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))
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


def _join_present(values: List[Any]) -> str:
    ordered: List[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in ordered:
            continue
        ordered.append(text)
    return "|".join(ordered)


def _parse_scalar_text(value: str) -> Any:
    text = str(value).strip()
    if not text:
        return ""
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]

    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "~"}:
        return None

    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


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
        value_text = str(_parse_scalar_text(value)).strip()
        if key_text and value_text:
            result[key_text] = value_text
    return result


def _parse_simple_suite_manifest_yaml(text: str) -> Dict[str, Any]:
    suites: List[Dict[str, Any]] = []
    current_suite: Dict[str, Any] | None = None
    in_suites = False

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        stripped = raw_line.strip()
        if not in_suites:
            if stripped == "suites:":
                in_suites = True
            continue

        if raw_line.lstrip().startswith("- "):
            current_suite = {}
            suites.append(current_suite)
            payload = raw_line.lstrip()[2:].strip()
            if payload:
                if ":" not in payload:
                    raise ValueError(f"Unsupported suite item line: {raw_line}")
                key, value = payload.split(":", 1)
                current_suite[key.strip()] = _parse_scalar_text(value)
            continue

        if current_suite is None:
            continue
        if ":" not in stripped:
            raise ValueError(f"Unsupported suite manifest line: {raw_line}")
        key, value = stripped.split(":", 1)
        current_suite[key.strip()] = _parse_scalar_text(value)

    if not suites:
        raise ValueError("Suite manifest contains no suites.")
    return {"suites": suites}


def _suite_label_mode(suite: Dict[str, Any]) -> str:
    has_real = bool(suite.get("external_real_manifest") or suite.get("external_real_bucket"))
    has_fake = bool(suite.get("external_fake_manifest") or suite.get("external_fake_bucket"))
    if has_real and has_fake:
        return "mixed"
    if has_real:
        return "real_only"
    if has_fake:
        return "fake_only"
    return "unknown"


def _suite_split_hint(suite: Dict[str, Any]) -> str:
    return _join_present(
        [
            suite.get("external_real_manifest_split"),
            suite.get("external_fake_manifest_split"),
        ]
    )


def _suite_slice_hint(suite: Dict[str, Any]) -> str:
    return _join_present(
        [
            suite.get("external_real_manifest_slices"),
            suite.get("external_fake_manifest_slices"),
            suite.get("external_real_prefix"),
            suite.get("external_fake_prefix"),
        ]
    )


def _suite_method_hint(suite: Dict[str, Any]) -> str:
    return _join_present(
        [
            suite.get("external_real_method"),
            suite.get("external_fake_method"),
        ]
    )


def _checkpoint_precision(checkpoint_key: str) -> str:
    upper = str(checkpoint_key).strip().upper()
    if upper.endswith("_FP32"):
        return "fp32"
    if upper.endswith("_INT8"):
        return "int8"
    return "other"


def _checkpoint_pair_key(checkpoint_key: str) -> str:
    upper = str(checkpoint_key).strip().upper()
    for suffix in ("_FP32", "_INT8"):
        if upper.endswith(suffix):
            return upper[: -len(suffix)]
    return upper


def _metric_direction(metric_name: str) -> str:
    if str(metric_name).strip() in {"real_fpr_at_0p5", "fake_fnr_at_0p5"}:
        return "lower_is_better"
    return "higher_is_better"


def _quantile(values: List[float], q: float) -> float:
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


def _report_path_for_job(output_root: str, checkpoint_key: str, suite_name: str) -> str:
    filename = f"{suite_name}_{checkpoint_key.lower()}_videos_report.csv"
    return _join_path(output_root, filename)


def _read_text_with_retries(path: str, attempts: int = 3, retry_delay_seconds: float = 5.0) -> str:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _read_text_from_path(path)
        except Exception as exc:  # pragma: no cover - exercised in integration runs
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(retry_delay_seconds)
    if last_error is None:
        raise RuntimeError(f"Failed to read report from {path}")
    raise last_error


def _load_video_report_rows(report_path: str) -> List[Dict[str, Any]]:
    text = _read_text_with_retries(report_path)
    reader = csv.DictReader(io.StringIO(text))
    rows: List[Dict[str, Any]] = []
    for raw_row in reader:
        rows.append(
            {
                "method": str(raw_row.get("method", "")).strip(),
                "label": int(float(str(raw_row.get("label", "0")).strip() or "0")),
                "video_id": str(raw_row.get("video_id", "")).strip(),
                "avg_video_prob": float(str(raw_row.get("avg_video_prob", "0")).strip() or "0"),
                "prediction": int(float(str(raw_row.get("prediction", "0")).strip() or "0")),
                "group_key": str(raw_row.get("group_key", "")).strip(),
                "family_key": str(raw_row.get("family_key", "")).strip(),
            }
        )

    if not rows:
        raise ValueError(f"Video report at {report_path} contained no rows.")

    return rows


def _build_scorecard_row(
    checkpoint_key: str,
    checkpoint_path: str,
    suite: Dict[str, Any],
    report_path: str,
    video_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    label_mode = _suite_label_mode(suite)
    suite_name = str(suite["name"])

    tn = fp = fn = tp = 0
    probs: List[float] = []
    methods_seen: List[str] = []
    for row in video_rows:
        label = int(row["label"])
        prediction = int(row["prediction"])
        prob = float(row["avg_video_prob"])
        probs.append(prob)

        method = str(row.get("method", "")).strip()
        if method and method not in methods_seen:
            methods_seen.append(method)

        if label == 0 and prediction == 0:
            tn += 1
        elif label == 0 and prediction == 1:
            fp += 1
        elif label == 1 and prediction == 0:
            fn += 1
        elif label == 1 and prediction == 1:
            tp += 1

    n_videos = len(video_rows)
    real_total = tn + fp
    fake_total = tp + fn
    accuracy = (tp + tn) / n_videos if n_videos > 0 else 0.0
    real_fpr = fp / real_total if real_total > 0 else None
    real_tnr = tn / real_total if real_total > 0 else None
    fake_recall = tp / fake_total if fake_total > 0 else None
    fake_fnr = fn / fake_total if fake_total > 0 else None

    if label_mode == "real_only":
        score_metric_name = "real_fpr_at_0p5"
        score_metric_value = real_fpr
    elif label_mode == "fake_only":
        score_metric_name = "fake_recall_at_0p5"
        score_metric_value = fake_recall
    else:
        score_metric_name = "accuracy_at_0p5"
        score_metric_value = accuracy

    methods_seen_text = "|".join(methods_seen)
    explicit_method_hint = _suite_method_hint(suite)

    return {
        "checkpoint_key": checkpoint_key,
        "checkpoint_path": checkpoint_path,
        "checkpoint_precision": _checkpoint_precision(checkpoint_key),
        "checkpoint_pair_key": _checkpoint_pair_key(checkpoint_key),
        "suite_name": suite_name,
        "label_mode": label_mode,
        "split_hint": _suite_split_hint(suite),
        "slice_hint": _suite_slice_hint(suite),
        "method_hint": explicit_method_hint or methods_seen_text,
        "methods_seen": methods_seen_text,
        "report_path": report_path,
        "n_videos": n_videos,
        "accuracy_at_0p5": round(float(accuracy), 6),
        "mean_prob": round(float(sum(probs) / n_videos), 6),
        "p50_prob": round(float(_quantile(probs, 0.50)), 6),
        "p90_prob": round(float(_quantile(probs, 0.90)), 6),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "real_fpr_at_0p5": None if real_fpr is None else round(float(real_fpr), 6),
        "real_tnr_at_0p5": None if real_tnr is None else round(float(real_tnr), 6),
        "fake_recall_at_0p5": None if fake_recall is None else round(float(fake_recall), 6),
        "fake_fnr_at_0p5": None if fake_fnr is None else round(float(fake_fnr), 6),
        "score_metric_name": score_metric_name,
        "score_metric_value": None if score_metric_value is None else round(float(score_metric_value), 6),
    }


def _load_scorecard_row(
    checkpoint_key: str,
    checkpoint_path: str,
    suite: Dict[str, Any],
    output_root: str,
) -> Dict[str, Any]:
    report_path = _report_path_for_job(
        output_root=output_root,
        checkpoint_key=checkpoint_key,
        suite_name=str(suite["name"]),
    )
    video_rows = _load_video_report_rows(report_path)
    return _build_scorecard_row(
        checkpoint_key=checkpoint_key,
        checkpoint_path=checkpoint_path,
        suite=suite,
        report_path=report_path,
        video_rows=video_rows,
    )


def _csv_list(value: str | None) -> List[str]:
    if value is None:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _scorecard_requested(args: argparse.Namespace) -> bool:
    return bool(
        args.scorecard_csv
        or args.scorecard_wide_csv
        or args.scorecard_delta_csv
        or args.scorecard_json
    )


def _promotion_contract_requested(args: argparse.Namespace) -> bool:
    return bool(args.promotion_contract_dir)


def _load_promotion_contract_module():
    module_name = "teams_promotion_contract_runtime"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = RUNNER_DIR / "score_teams_promotion_contract.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load promotion contract module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_promotion_contract_outputs(
    args: argparse.Namespace,
    selected_checkpoint_keys: List[str],
) -> None:
    promotion = _load_promotion_contract_module()
    contract = promotion.ContractConfig(
        dev_real_suite=str(args.promotion_dev_real_suite).strip(),
        dev_real_stress_suites=tuple(_csv_list(args.promotion_dev_real_stress_suites)),
        dev_fake_suites=tuple(_csv_list(args.promotion_dev_fake_suites)),
        lockbox_real_suite=str(args.promotion_lockbox_real_suite).strip(),
        lockbox_fake_suite=str(args.promotion_lockbox_fake_suite).strip(),
    )
    payload = promotion.score_promotion_contract(
        report_root=str(args.output_gcs_folder).strip(),
        checkpoint_map_path=str(args.checkpoint_map).strip(),
        checkpoints_arg=",".join(selected_checkpoint_keys),
        contract=contract,
    )
    output_paths = promotion.write_promotion_contract_outputs(
        str(args.promotion_contract_dir).strip(),
        payload,
    )
    winner = payload.get("winner")
    print("Promotion contract artifacts written to:")
    print(f"  {output_paths['threshold_grid_csv']}")
    print(f"  {output_paths['selected_threshold_scorecard_csv']}")
    print(f"  {output_paths['checkpoint_summary_csv']}")
    print(f"  {output_paths['promotion_contract_json']}")
    print(f"  {output_paths['promotion_winner_json']}")
    if winner:
        print(
            "Promotion winner: "
            f"{winner['checkpoint_key']} "
            f"(lockbox_real_fpr={winner['lockbox_real_fpr']}, "
            f"lockbox_fake_recall={winner['lockbox_fake_recall']}, "
            f"threshold={winner['selected_threshold']})"
        )


def _build_wide_scorecard_rows(scorecard_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_checkpoint: Dict[str, Dict[str, Any]] = {}
    for row in scorecard_rows:
        checkpoint_key = str(row["checkpoint_key"])
        wide_row = by_checkpoint.setdefault(
            checkpoint_key,
            {
                "checkpoint_key": checkpoint_key,
                "checkpoint_path": row["checkpoint_path"],
                "checkpoint_precision": row.get("checkpoint_precision", "other"),
                "checkpoint_pair_key": row.get("checkpoint_pair_key", checkpoint_key),
            },
        )
        suite_name = str(row["suite_name"])
        wide_row[f"{suite_name}__score_metric_name"] = row["score_metric_name"]
        if row.get("score_metric_value") is not None:
            wide_row[f"{suite_name}__{row['score_metric_name']}"] = row["score_metric_value"]
        wide_row[f"{suite_name}__accuracy_at_0p5"] = row["accuracy_at_0p5"]
        wide_row[f"{suite_name}__n_videos"] = row["n_videos"]

    return [by_checkpoint[key] for key in sorted(by_checkpoint.keys())]


def _build_pair_delta_rows(scorecard_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pair_rows: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    for row in scorecard_rows:
        precision = str(row.get("checkpoint_precision") or _checkpoint_precision(str(row["checkpoint_key"])))
        if precision not in {"fp32", "int8"}:
            continue

        pair_key = str(row.get("checkpoint_pair_key") or _checkpoint_pair_key(str(row["checkpoint_key"])))
        suite_name = str(row["suite_name"])
        entry = pair_rows.setdefault((pair_key, suite_name), {})
        entry[precision] = row

    delta_rows: List[Dict[str, Any]] = []
    for pair_key, suite_name in sorted(pair_rows.keys()):
        paired = pair_rows[(pair_key, suite_name)]
        fp32_row = paired.get("fp32")
        int8_row = paired.get("int8")
        if fp32_row is None or int8_row is None:
            continue

        fp32_metric_name = str(fp32_row.get("score_metric_name", "")).strip()
        int8_metric_name = str(int8_row.get("score_metric_name", "")).strip()
        metric_names_match = fp32_metric_name == int8_metric_name
        metric_direction = _metric_direction(fp32_metric_name) if metric_names_match else ""

        fp32_metric_value = fp32_row.get("score_metric_value")
        int8_metric_value = int8_row.get("score_metric_value")
        raw_delta = None
        directional_delta = None
        if (
            metric_names_match
            and fp32_metric_value is not None
            and int8_metric_value is not None
        ):
            raw_delta = round(float(int8_metric_value) - float(fp32_metric_value), 6)
            if metric_direction == "lower_is_better":
                directional_delta = round(float(fp32_metric_value) - float(int8_metric_value), 6)
            else:
                directional_delta = raw_delta

        accuracy_delta = None
        if (
            fp32_row.get("accuracy_at_0p5") is not None
            and int8_row.get("accuracy_at_0p5") is not None
        ):
            accuracy_delta = round(
                float(int8_row["accuracy_at_0p5"]) - float(fp32_row["accuracy_at_0p5"]),
                6,
            )

        delta_rows.append(
            {
                "checkpoint_pair_key": pair_key,
                "suite_name": suite_name,
                "label_mode": fp32_row.get("label_mode", ""),
                "split_hint": fp32_row.get("split_hint", ""),
                "slice_hint": fp32_row.get("slice_hint", ""),
                "method_hint": fp32_row.get("method_hint", ""),
                "metric_names_match": metric_names_match,
                "score_metric_name": fp32_metric_name if metric_names_match else "",
                "metric_direction": metric_direction,
                "fp32_checkpoint_key": fp32_row["checkpoint_key"],
                "fp32_checkpoint_path": fp32_row["checkpoint_path"],
                "fp32_score_metric_value": fp32_metric_value,
                "fp32_accuracy_at_0p5": fp32_row.get("accuracy_at_0p5"),
                "fp32_n_videos": fp32_row.get("n_videos"),
                "int8_checkpoint_key": int8_row["checkpoint_key"],
                "int8_checkpoint_path": int8_row["checkpoint_path"],
                "int8_score_metric_value": int8_metric_value,
                "int8_accuracy_at_0p5": int8_row.get("accuracy_at_0p5"),
                "int8_n_videos": int8_row.get("n_videos"),
                "n_videos_match": fp32_row.get("n_videos") == int8_row.get("n_videos"),
                "score_metric_delta_int8_minus_fp32": raw_delta,
                "score_metric_directional_delta": directional_delta,
                "accuracy_at_0p5_delta_int8_minus_fp32": accuracy_delta,
            }
        )

    return delta_rows


def _write_dict_rows_to_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for scorecard export: {path}")

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


def _write_scorecard_outputs(args: argparse.Namespace, scorecard_rows: List[Dict[str, Any]]) -> None:
    wide_rows = _build_wide_scorecard_rows(scorecard_rows)
    pair_delta_rows = _build_pair_delta_rows(scorecard_rows)

    if args.scorecard_csv:
        _write_dict_rows_to_csv(args.scorecard_csv, scorecard_rows)
        print(f"Scorecard CSV written to: {args.scorecard_csv}")

    if args.scorecard_wide_csv:
        _write_dict_rows_to_csv(args.scorecard_wide_csv, wide_rows)
        print(f"Wide scorecard CSV written to: {args.scorecard_wide_csv}")

    if args.scorecard_delta_csv:
        if pair_delta_rows:
            _write_dict_rows_to_csv(args.scorecard_delta_csv, pair_delta_rows)
            print(f"Pair delta CSV written to: {args.scorecard_delta_csv}")
        else:
            _write_text_to_path(args.scorecard_delta_csv, "")
            print(
                "Pair delta CSV requested but no FP32/INT8 pairs were available: "
                f"{args.scorecard_delta_csv}"
            )

    if args.scorecard_json:
        payload = {
            "scorecard_rows": scorecard_rows,
            "wide_rows": wide_rows,
            "pair_delta_rows": pair_delta_rows,
        }
        _write_text_to_path(args.scorecard_json, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"Scorecard JSON written to: {args.scorecard_json}")


def _default_checkpoint_map() -> Dict[str, str]:
    mapping = {}
    for idx in range(1, 9):
        key = f"FT{idx}"
        env_key = f"R4_{key}_CKPT"
        mapping[key] = os.environ.get(env_key, "").strip()
    return mapping


def _merged_checkpoint_map(checkpoint_map_path: str | None) -> Dict[str, str]:
    checkpoint_map = _default_checkpoint_map()
    if checkpoint_map_path:
        checkpoint_map.update(_load_checkpoint_map(checkpoint_map_path))
    return checkpoint_map


def _load_checkpoint_map(path: str) -> Dict[str, str]:
    with open(path, "r") as f:
        text = f.read().strip()

    if not text:
        return {}

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # noqa
        except ImportError:
            data = _parse_flat_key_value_mapping(text)
        else:
            data = yaml.safe_load(text) or {}

    result: Dict[str, str] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            result[str(key).strip().upper()] = str(value).strip()
    return result


def _resolve_checkpoints(selected_keys: List[str], checkpoint_map_path: str | None) -> Dict[str, str]:
    checkpoint_map = _merged_checkpoint_map(checkpoint_map_path)

    resolved: Dict[str, str] = {}
    missing: List[str] = []
    for key in selected_keys:
        path = checkpoint_map.get(key, "")
        if path:
            resolved[key] = path
        else:
            missing.append(key)

    if missing:
        raise ValueError(
            "Missing checkpoint path(s) for: "
            + ", ".join(missing)
            + ". Provide --checkpoint_map or env vars R4_FT*_CKPT."
        )

    return resolved


def _resolve_requested_checkpoint_keys(
    checkpoints_arg: str,
    checkpoint_map_path: str | None,
) -> List[str]:
    selected = [part.strip().upper() for part in checkpoints_arg.split(",") if part.strip()]
    if not selected:
        raise ValueError("No checkpoint keys were requested.")

    checkpoint_map = _merged_checkpoint_map(checkpoint_map_path)
    available_with_paths = sorted(
        key for key, value in checkpoint_map.items() if str(value).strip()
    )

    if selected == ["ALL"]:
        if not available_with_paths:
            raise ValueError(
                "No checkpoint paths are available for --checkpoints ALL. "
                "Provide --checkpoint_map or env vars R4_FT*_CKPT."
            )
        return available_with_paths

    if checkpoint_map_path:
        available_keys = sorted(checkpoint_map.keys())
        invalid = [key for key in selected if key not in checkpoint_map]
        if invalid:
            raise ValueError(
                f"Invalid checkpoint key(s): {invalid}. "
                f"Available keys from checkpoint map/env: {available_keys}"
            )
        return selected

    valid = {f"FT{i}" for i in range(1, 9)}
    invalid = [key for key in selected if key not in valid]
    if invalid:
        raise ValueError(
            f"Invalid checkpoint key(s): {invalid}. "
            f"Valid default keys: {sorted(valid)}. "
            "Provide --checkpoint_map for custom aliases."
        )
    return selected


def _load_suites(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        text = f.read()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # noqa
        except ImportError:
            data = _parse_simple_suite_manifest_yaml(text)
        else:
            data = yaml.safe_load(text)

    if isinstance(data, dict):
        suites = data.get("suites", [])
    elif isinstance(data, list):
        suites = data
    else:
        suites = []

    if not suites:
        raise ValueError("Suite manifest contains no suites.")

    normalized = []
    for idx, suite in enumerate(suites, 1):
        if not isinstance(suite, dict):
            raise ValueError(f"Suite #{idx} must be a mapping.")
        if not suite.get("name"):
            raise ValueError(f"Suite #{idx} missing required key 'name'.")
        normalized.append(suite)

    return normalized


def _append_if_present(args: List[str], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    args.extend([key, str(value)])


def _build_job_args(
    checkpoint_key: str,
    checkpoint_path: str,
    suite: Dict[str, Any],
    common: argparse.Namespace,
) -> List[str]:
    suite_name = str(suite["name"])
    local_ckpt = f"./weights/validation_checkpoint_{checkpoint_key}_{suite_name}.pth"

    args = [
        "--checkpoint_gcs_path", checkpoint_path,
        "--checkpoint_local_path", local_ckpt,
        "--log_prefix", f"td_{suite_name}_{checkpoint_key.lower()}",
        "--run_name", f"TargetDomain {suite_name} {checkpoint_key}",
        "--output_gcs_folder", common.output_gcs_folder,
        "--output_filename_prefix", f"{suite_name}_{checkpoint_key.lower()}_",
        "--wandb_project", common.wandb_project,
        "--frames_per_video", str(int(suite.get("frames_per_video", common.frames_per_video))),
    ]

    if common.detailed_reports:
        args.append("--detailed_reports")
    else:
        args.append("--no_detailed_reports")

    # DF40
    df40_mode = str(suite.get("df40_mode", common.df40_mode))
    args.extend(["--df40_mode", df40_mode])
    if df40_mode != "none":
        args.extend(["--df40_orientation", str(suite.get("df40_orientation", common.df40_orientation))])
        _append_if_present(args, "--df40_methods", suite.get("df40_methods"))

    _append_if_present(args, "--eval_num_workers", suite.get("eval_num_workers"))

    # DeepLive
    _append_if_present(args, "--deeplive_bucket", suite.get("deeplive_bucket"))
    if suite.get("deeplive_bucket"):
        args.extend(["--deeplive_split", str(suite.get("deeplive_split", "all"))])
        _append_if_present(args, "--deeplive_strategies", suite.get("deeplive_strategies"))

    # VisoMaster
    _append_if_present(args, "--visomaster_bucket", suite.get("visomaster_bucket"))
    if suite.get("visomaster_bucket"):
        _append_if_present(args, "--visomaster_frames_bucket", suite.get("visomaster_frames_bucket"))
        _append_if_present(args, "--visomaster_swap_models", suite.get("visomaster_swap_models"))
        _append_if_present(args, "--visomaster_tiers", suite.get("visomaster_tiers"))

    # External real
    _append_if_present(args, "--external_real_manifest", suite.get("external_real_manifest"))
    _append_if_present(args, "--external_real_manifest_split", suite.get("external_real_manifest_split"))
    _append_if_present(args, "--external_real_manifest_slices", suite.get("external_real_manifest_slices"))
    _append_if_present(args, "--external_real_bucket", suite.get("external_real_bucket"))
    if suite.get("external_real_manifest") or suite.get("external_real_bucket"):
        _append_if_present(args, "--external_real_prefix", suite.get("external_real_prefix"))
        _append_if_present(args, "--external_real_method", suite.get("external_real_method"))
        _append_if_present(args, "--external_real_cache", suite.get("external_real_cache"))
        _append_if_present(args, "--max_external_real", suite.get("max_external_real"))
        _append_if_present(args, "--external_real_seed", suite.get("external_real_seed"))
        if bool(suite.get("external_real_deterministic", False)):
            args.append("--external_real_deterministic")

    # External fake
    _append_if_present(args, "--external_fake_manifest", suite.get("external_fake_manifest"))
    _append_if_present(args, "--external_fake_manifest_split", suite.get("external_fake_manifest_split"))
    _append_if_present(args, "--external_fake_manifest_slices", suite.get("external_fake_manifest_slices"))
    _append_if_present(args, "--external_fake_bucket", suite.get("external_fake_bucket"))
    if suite.get("external_fake_manifest") or suite.get("external_fake_bucket"):
        _append_if_present(args, "--external_fake_prefix", suite.get("external_fake_prefix"))
        _append_if_present(args, "--external_fake_method", suite.get("external_fake_method"))
        _append_if_present(args, "--external_fake_cache", suite.get("external_fake_cache"))
        _append_if_present(args, "--external_fake_grouping", suite.get("external_fake_grouping"))
        _append_if_present(args, "--max_external_fake", suite.get("max_external_fake"))
        _append_if_present(args, "--external_fake_seed", suite.get("external_fake_seed"))
        if bool(suite.get("external_fake_deterministic", False)):
            args.append("--external_fake_deterministic")

    return args


def _build_validation_command(job_args: List[str]) -> List[str]:
    return [sys.executable, "-u", str(VALIDATE_CUSTOM_SOURCES_SCRIPT)] + job_args


def _run_job(index: int, total: int, job_name: str, cmd: List[str], dry_run: bool) -> bool:
    print(f"\n{'=' * 88}")
    print(f"[{index}/{total}] {job_name}")
    print(f"{'=' * 88}")
    print("Command:")
    print(" ".join(cmd))

    if dry_run:
        print("[DRY RUN] Skipping execution")
        return True

    start = time.time()
    result = subprocess.run(cmd, cwd=str(TRAINING_ROOT))
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"FAILED {job_name} (exit={result.returncode}) after {elapsed:.1f}s")
        return False

    print(f"Completed {job_name} in {elapsed:.1f}s")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sequential target-domain validation runner with diagnostic fixed-threshold "
            "scorecards and optional calibrated promotion-contract exports"
        )
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="FT7",
        help=(
            "Comma-separated checkpoint aliases to run. "
            "Use --checkpoint_map for custom aliases, or ALL to run every available alias."
        ),
    )
    parser.add_argument("--checkpoint_map", type=str, default=None)
    parser.add_argument("--suite_manifest", type=str, required=True,
                        help="JSON/YAML manifest containing validation suites.")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "phase2-experiments"))
    parser.add_argument("--output_gcs_folder", type=str,
                        default="gs://training-job-outputs/test_results/target_domain_validation")
    parser.add_argument("--frames_per_video", type=int, default=8)
    parser.add_argument("--detailed_reports", dest="detailed_reports", action="store_true")
    parser.add_argument("--no_detailed_reports", dest="detailed_reports", action="store_false")
    parser.set_defaults(detailed_reports=True)
    parser.add_argument(
        "--scorecard_csv",
        type=str,
        default=None,
        help=(
            "Optional long-form diagnostic scorecard CSV output path "
            "(fixed threshold 0.5; local or gs://)."
        ),
    )
    parser.add_argument(
        "--scorecard_wide_csv",
        type=str,
        default=None,
        help=(
            "Optional checkpoint-by-suite diagnostic scorecard CSV output path "
            "(fixed threshold 0.5; local or gs://)."
        ),
    )
    parser.add_argument(
        "--scorecard_json",
        type=str,
        default=None,
        help=(
            "Optional JSON payload containing the diagnostic long, wide, and "
            "pair-delta scorecard tables."
        ),
    )
    parser.add_argument(
        "--scorecard_delta_csv",
        type=str,
        default=None,
        help=(
            "Optional diagnostic FP32-vs-INT8 pair-delta CSV output path "
            "(local or gs://)."
        ),
    )
    parser.add_argument(
        "--promotion_contract_dir",
        type=str,
        default=None,
        help=(
            "Optional calibrated promotion-contract output directory. "
            "Requires --checkpoint_map and detailed reports."
        ),
    )
    parser.add_argument("--promotion_dev_real_suite", type=str, default="teams_real_all_dev")
    parser.add_argument(
        "--promotion_dev_real_stress_suites",
        type=str,
        default="teams_real_poor_quality_dev,teams_real_lighting_extreme_dev",
    )
    parser.add_argument(
        "--promotion_dev_fake_suites",
        type=str,
        default="teams_fake_all_dev,visomaster_enhanced_macro_dev,deeplive_enhanced_dev",
    )
    parser.add_argument("--promotion_lockbox_real_suite", type=str, default="teams_real_all_lockbox")
    parser.add_argument("--promotion_lockbox_fake_suite", type=str, default="teams_fake_all_lockbox")

    # Defaults for optional suite keys
    parser.add_argument("--df40_mode", type=str, default="none",
                        choices=["paired", "fake_only", "real_only", "none"])
    parser.add_argument("--df40_orientation", type=str, default="target_source",
                        choices=["all", "target_source", "source_target"])

    args = parser.parse_args()

    selected = _resolve_requested_checkpoint_keys(
        checkpoints_arg=args.checkpoints,
        checkpoint_map_path=args.checkpoint_map,
    )
    if (_scorecard_requested(args) or _promotion_contract_requested(args)) and not args.detailed_reports:
        raise ValueError(
            "Scorecard or promotion-contract export requires detailed reports to stay enabled."
        )
    if _promotion_contract_requested(args) and not args.checkpoint_map:
        raise ValueError(
            "Promotion contract export requires --checkpoint_map so checkpoint aliases "
            "resolve reproducibly."
        )

    checkpoints = _resolve_checkpoints(selected, args.checkpoint_map)
    suites = _load_suites(args.suite_manifest)

    jobs: List[Dict[str, Any]] = []
    for suite in suites:
        suite_name = str(suite["name"])
        for ckpt_key in selected:
            ckpt_path = checkpoints[ckpt_key]
            cmd = _build_validation_command(_build_job_args(
                checkpoint_key=ckpt_key,
                checkpoint_path=ckpt_path,
                suite=suite,
                common=args,
            ))
            jobs.append(
                {
                    "job_name": f"suite={suite_name} checkpoint={ckpt_key}",
                    "cmd": cmd,
                    "suite": suite,
                    "suite_name": suite_name,
                    "checkpoint_key": ckpt_key,
                    "checkpoint_path": ckpt_path,
                }
            )

    print("=" * 88)
    print("Target-domain validation plan")
    print(f"Checkpoints: {selected}")
    print(f"Suites: {[s['name'] for s in suites]}")
    print(f"Output folder: {args.output_gcs_folder}")
    print(f"W&B project: {args.wandb_project}")
    if _scorecard_requested(args):
        print("Diagnostic scorecards: enabled (fixed threshold 0.5)")
    if _promotion_contract_requested(args):
        print(f"Promotion contract dir: {args.promotion_contract_dir}")
    print("=" * 88)

    results = []
    scorecard_rows: List[Dict[str, Any]] = []
    scorecard_failures: List[str] = []
    global_start = time.time()
    for idx, job in enumerate(jobs, 1):
        ok = _run_job(idx, len(jobs), job["job_name"], job["cmd"], args.dry_run)
        results.append((job["job_name"], ok))
        if ok and _scorecard_requested(args) and not args.dry_run:
            try:
                row = _load_scorecard_row(
                    checkpoint_key=job["checkpoint_key"],
                    checkpoint_path=job["checkpoint_path"],
                    suite=job["suite"],
                    output_root=args.output_gcs_folder,
                )
                scorecard_rows.append(row)
                metric_value = row.get("score_metric_value")
                metric_text = "n/a" if metric_value is None else f"{metric_value:.4f}"
                print(
                    "Scorecard "
                    f"{job['checkpoint_key']} {job['suite_name']}: "
                    f"{row['score_metric_name']}={metric_text}"
                )
            except Exception as exc:
                message = (
                    "FAILED to build scorecard row for "
                    f"suite={job['suite_name']} checkpoint={job['checkpoint_key']}: {exc}"
                )
                print(message)
                scorecard_failures.append(message)

    elapsed = time.time() - global_start
    failed = [job_name for job_name, ok in results if not ok]

    print(f"\n{'=' * 88}")
    print(f"Target-domain summary ({len(jobs)} jobs, {elapsed:.1f}s)")
    print(f"{'=' * 88}")
    for job_name, ok in results:
        print(f"{'OK ' if ok else 'ERR'} {job_name}")

    if _scorecard_requested(args) and not args.dry_run and scorecard_rows:
        _write_scorecard_outputs(args, scorecard_rows)
    if _promotion_contract_requested(args) and not args.dry_run:
        _write_promotion_contract_outputs(args, selected)

    if failed:
        print(f"\nFailed jobs: {failed}")
        raise SystemExit(1)
    if scorecard_failures:
        print(f"\nScorecard failures: {scorecard_failures}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
