#!/usr/bin/env python3
"""
Common helpers for WT-D Teams decision-system report analysis.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ROUND_DIGITS = 6


def split_gs_uri(uri: str) -> Tuple[str, str]:
    stripped = uri.replace("gs://", "", 1)
    if "/" not in stripped:
        return stripped, ""
    return tuple(stripped.split("/", 1))


def read_text_from_path(path: str) -> str:
    if path.startswith("gs://"):
        try:
            from google.cloud import storage
        except ImportError as exc:  # pragma: no cover - depends on runtime.
            raise RuntimeError(
                "Reading gs:// paths requires google-cloud-storage in this runtime."
            ) from exc

        bucket_name, blob_path = split_gs_uri(path)
        client = storage.Client()
        return client.bucket(bucket_name).blob(blob_path).download_as_text()

    return Path(path).read_text()


def write_text_to_path(path: str, text: str) -> None:
    if path.startswith("gs://"):
        try:
            from google.cloud import storage
        except ImportError as exc:  # pragma: no cover - depends on runtime.
            raise RuntimeError(
                "Writing gs:// paths requires google-cloud-storage in this runtime."
            ) from exc

        bucket_name, blob_path = split_gs_uri(path)
        client = storage.Client()
        client.bucket(bucket_name).blob(blob_path).upload_from_string(
            text,
            content_type="text/plain; charset=utf-8",
        )
        return

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)


def join_path(root: str, filename: str) -> str:
    if root.startswith("gs://"):
        return f"{root.rstrip('/')}/{filename}"
    return str(Path(root) / filename)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    rows = list(csv.DictReader(io.StringIO(read_text_from_path(path))))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def write_dict_rows_to_csv(path: str, rows: List[Dict[str, Any]], *, allow_empty: bool = False) -> None:
    if not rows:
        if allow_empty:
            write_text_to_path(path, "")
            return
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
    write_text_to_path(path, buffer.getvalue())


def csv_list(value: str | None) -> Tuple[str, ...]:
    if value is None:
        return ()
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def parse_flat_key_value_mapping(text: str) -> Dict[str, str]:
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


def load_checkpoint_map(path: str) -> Dict[str, str]:
    text = read_text_from_path(path).strip()
    if not text:
        return {}

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError:
            data = parse_flat_key_value_mapping(text)
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


def resolve_requested_checkpoint_keys(
    checkpoints_arg: str,
    checkpoint_map_path: str,
) -> List[str]:
    checkpoint_map = load_checkpoint_map(checkpoint_map_path)
    available = sorted(checkpoint_map.keys())
    requested = [part.strip().upper() for part in str(checkpoints_arg).split(",") if part.strip()]
    if not requested:
        raise ValueError("No checkpoints requested.")
    if len(requested) == 1 and requested[0] == "ALL":
        if not available:
            raise ValueError(f"Checkpoint map is empty: {checkpoint_map_path}")
        return available
    return requested


def resolve_checkpoints(selected_keys: Sequence[str], checkpoint_map_path: str) -> Dict[str, str]:
    checkpoint_map = load_checkpoint_map(checkpoint_map_path)
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


def report_path_for_job(report_root: str, checkpoint_key: str, suite_name: str, report_kind: str) -> str:
    filename = f"{suite_name}_{checkpoint_key.lower()}_{report_kind}.csv"
    return join_path(report_root, filename)


def parse_label(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"0", "real", "false", "negative"}:
        return 0
    if text in {"1", "fake", "true", "positive"}:
        return 1
    raise ValueError(f"Unsupported label value: {value!r}")


def round_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, ROUND_DIGITS)


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def mean(values: Iterable[float]) -> float:
    ordered = [float(value) for value in values]
    if not ordered:
        return 0.0
    return sum(ordered) / float(len(ordered))


def optional_mean(values: Iterable[float | None]) -> float | None:
    ordered = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not ordered:
        return None
    return sum(ordered) / float(len(ordered))


def quantile(values: Sequence[float], q: float) -> float:
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


def stddev(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((float(value) - mu) ** 2 for value in values) / float(len(values)))


def sort_number(value: Any, *, higher_is_better: bool = False) -> float:
    if value is None:
        return float("inf")
    try:
        number = float(value)
    except Exception:
        return float("inf")
    if not math.isfinite(number):
        return float("inf")
    return -number if higher_is_better else number


def format_float_slug(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "neg").replace(".", "p")


@dataclass(frozen=True)
class ContractConfig:
    dev_real_suite: str
    dev_real_stress_suites: Tuple[str, ...]
    dev_fake_suites: Tuple[str, ...]
    lockbox_real_suite: str = ""
    lockbox_fake_suite: str = ""

    def dev_suites(self) -> Tuple[str, ...]:
        ordered: List[str] = []
        for suite_name in (
            self.dev_real_suite,
            *self.dev_real_stress_suites,
            *self.dev_fake_suites,
        ):
            if suite_name and suite_name not in ordered:
                ordered.append(suite_name)
        return tuple(ordered)

    def all_suites(self) -> Tuple[str, ...]:
        ordered = list(self.dev_suites())
        for suite_name in (self.lockbox_real_suite, self.lockbox_fake_suite):
            if suite_name and suite_name not in ordered:
                ordered.append(suite_name)
        return tuple(ordered)


@dataclass(frozen=True)
class VideoScore:
    label: int
    prob: float
    method: str
    video_id: str


def load_video_scores(report_path: str) -> List[VideoScore]:
    rows = read_csv_rows(report_path)
    parsed: List[VideoScore] = []
    for row in rows:
        parsed.append(
            VideoScore(
                label=parse_label(row.get("label", "")),
                prob=float(str(row.get("avg_video_prob", "0")).strip() or "0"),
                method=str(row.get("method", "")).strip(),
                video_id=str(row.get("video_id", "")).strip(),
            )
        )
    if not parsed:
        raise ValueError(f"Video report contains no rows: {report_path}")
    return parsed


@dataclass(frozen=True)
class FrameScore:
    label: int
    prob: float
    method: str
    video_id: str
    frame_path: str
    group_key: str
    family_key: str


def load_frame_scores(report_path: str) -> List[FrameScore]:
    rows = read_csv_rows(report_path)
    parsed: List[FrameScore] = []
    for row in rows:
        prob_text = (
            row.get("frame_prob")
            or row.get("probability")
            or row.get("prob_fake")
            or "0"
        )
        parsed.append(
            FrameScore(
                label=parse_label(row.get("label", "")),
                prob=float(str(prob_text).strip() or "0"),
                method=str(row.get("method", "")).strip(),
                video_id=str(row.get("video_id", "")).strip(),
                frame_path=str(row.get("frame_path", row.get("frame_name", ""))).strip(),
                group_key=str(row.get("group_key", "")).strip(),
                family_key=str(row.get("family_key", "")).strip(),
            )
        )
    if not parsed:
        raise ValueError(f"Frame report contains no rows: {report_path}")
    return parsed


def group_frame_scores_by_video(rows: Sequence[FrameScore]) -> Dict[str, List[FrameScore]]:
    grouped: Dict[str, List[FrameScore]] = {}
    for row in rows:
        grouped.setdefault(row.video_id, []).append(row)
    for video_id, frames in grouped.items():
        grouped[video_id] = sorted(frames, key=lambda item: item.frame_path)
    return grouped


def score_decisions(labels: Sequence[int], decisions: Sequence[int | None]) -> Dict[str, Any]:
    if len(labels) != len(decisions):
        raise ValueError("Labels and decisions must have the same length.")

    tn = fp = fn = tp = 0
    n_uncertain = 0
    n_decided_real = n_decided_fake = 0

    for label, decision in zip(labels, decisions):
        if decision is None:
            n_uncertain += 1
            continue
        if label == 0:
            n_decided_real += 1
        else:
            n_decided_fake += 1

        if label == 0 and decision == 0:
            tn += 1
        elif label == 0 and decision == 1:
            fp += 1
        elif label == 1 and decision == 0:
            fn += 1
        elif label == 1 and decision == 1:
            tp += 1

    n_total = len(labels)
    n_real = sum(1 for label in labels if label == 0)
    n_fake = sum(1 for label in labels if label == 1)
    n_decided = n_total - n_uncertain

    accuracy = safe_rate(tp + tn, n_total) or 0.0
    clean_accuracy = safe_rate(tp + tn, n_decided)
    real_fpr = safe_rate(fp, n_real)
    fake_recall = safe_rate(tp, n_fake)
    real_coverage = safe_rate(n_decided_real, n_real)
    fake_coverage = safe_rate(n_decided_fake, n_fake)
    decided_rate = safe_rate(n_decided, n_total) or 0.0
    uncertain_rate = safe_rate(n_uncertain, n_total) or 0.0

    return {
        "n_videos": n_total,
        "n_real": n_real,
        "n_fake": n_fake,
        "n_decided": n_decided,
        "n_uncertain": n_uncertain,
        "decided_rate": round_float(decided_rate),
        "uncertain_rate": round_float(uncertain_rate),
        "accuracy": round_float(accuracy),
        "clean_accuracy": round_float(clean_accuracy),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "real_fpr": round_float(real_fpr),
        "fake_recall": round_float(fake_recall),
        "real_coverage": round_float(real_coverage),
        "fake_coverage": round_float(fake_coverage),
    }
