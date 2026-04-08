#!/usr/bin/env python3
"""
Build a deterministic target-domain manifest for the Teams benchmark bucket.

The manifest is intended for validation, not training. It freezes:
  - one row per discovered target-domain video
  - a stable ``dev`` / ``lockbox`` split keyed by identity/session
  - core real-only slices:
      * teams_real_all
      * teams_real_poor_quality
      * teams_real_lighting_extreme

Fake-family provenance in the flat Teams bucket is incomplete. This builder
ships a minimal prefix-rule hook so known prefixes can be tagged explicitly
without changing the grouping/splitting logic.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import logging
import os
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
DEFAULT_METRIC_WORKERS = min(16, max(4, os.cpu_count() or 4))

log = logging.getLogger(__name__)
_THREAD_LOCAL = threading.local()

_STRUCTURED_CAPTURE_RE = re.compile(
    r"^(?P<prefix>.+?)__s(?P<session>[^_]+)_(?P<segment>[^_]+)_frame_"
    r"(?P<frame>\d+)_crop_(?P<crop>\d+)__(?P<uid>[^.]+)\.(?P<ext>\w+)$"
)
_SEQUENCE_RE = re.compile(
    r"^(?P<prefix>.+?)__frame_(?P<frame>\d+)_seq(?P<seq>\d+)"
    r"(?:__(?P<uid>[^.]+))?\.(?P<ext>\w+)$"
)

_DEFAULT_PREFIX_RULES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "real": {},
    "fake": {
        "visomaster_enhanced_raw": {
            "method": "visomaster_enhanced_macro",
            "slices": ["visomaster_enhanced_macro"],
        },
        "visomaster_enhanced_teams": {
            "method": "visomaster_enhanced_macro",
            "slices": ["visomaster_enhanced_macro"],
        },
        # This prefix comes from the quality-enhancement DeepLive upload lane.
        "deeplive_dor": {
            "method": "deeplive_enhanced",
            "slices": ["deeplive_enhanced"],
        },
    },
}


@dataclass
class DiscoveredVideo:
    label: str
    prefix: str
    video_id: str
    identity_key: str
    frame_paths: List[str] = field(default_factory=list)
    session_id: Optional[str] = None
    segment_id: Optional[str] = None
    sequence_id: Optional[str] = None
    source_kind: str = "unknown"


def _split_gs_uri(uri: str) -> Tuple[str, str]:
    stripped = uri.replace("gs://", "", 1)
    if "/" not in stripped:
        return stripped, ""
    bucket, blob = stripped.split("/", 1)
    return bucket, blob


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
            text, content_type="application/json"
        )
        return

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)


def _load_structured_data(path: str) -> Dict[str, Any]:
    text = _read_text_from_path(path)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import yaml

        data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ValueError(f"Expected rule file {path} to contain a mapping.")
    return data


def _merge_rules(
    base: Dict[str, Dict[str, Dict[str, Any]]],
    override: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    merged: Dict[str, Dict[str, Dict[str, Any]]] = {
        "real": {**base.get("real", {})},
        "fake": {**base.get("fake", {})},
    }
    for label in ("real", "fake"):
        label_rules = override.get(label, {}) or {}
        if not isinstance(label_rules, dict):
            raise ValueError(f"Rule section '{label}' must be a mapping.")
        for key, value in label_rules.items():
            if not isinstance(value, dict):
                raise ValueError(f"Rule '{label}.{key}' must be a mapping.")
            merged[label][str(key)] = dict(value)
    return merged


def _load_prefix_rules(path: Optional[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    rules = {
        "real": {**_DEFAULT_PREFIX_RULES["real"]},
        "fake": {**_DEFAULT_PREFIX_RULES["fake"]},
    }
    if not path:
        return rules
    override = _load_structured_data(path)
    return _merge_rules(rules, override)


def _stable_int_hash(value: str) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def _stable_fraction(key: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


class StorageReader:
    def __init__(self, source_root: str):
        self.source_root = source_root.rstrip("/")
        self.is_gcs = self.source_root.startswith("gs://")
        if self.is_gcs:
            from google.cloud import storage

            self.bucket_name, self.root_prefix = _split_gs_uri(self.source_root)
            self.client = storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))
            self.bucket = self.client.bucket(self.bucket_name)
        else:
            self.root_path = Path(self.source_root).expanduser().resolve()

    def iter_label_paths(self, label: str) -> Iterable[str]:
        if self.is_gcs:
            prefix = f"{self.root_prefix}/{label}/".strip("/")
            for blob in self.client.list_blobs(self.bucket_name, prefix=prefix):
                if blob.name.endswith("/"):
                    continue
                if Path(blob.name).suffix.lower() not in IMAGE_EXTS:
                    continue
                yield f"gs://{self.bucket_name}/{blob.name}"
            return

        base = self.root_path / label
        if not base.exists():
            return
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                yield str(path.resolve())

    def read_bytes(self, path: str) -> bytes:
        if path.startswith("gs://"):
            bucket_name, blob_path = _split_gs_uri(path)
            return self.client.bucket(bucket_name).blob(blob_path).download_as_bytes()
        return Path(path).read_bytes()


def _parse_discovery_metadata(filename: str, label: str) -> Dict[str, Any]:
    match = _STRUCTURED_CAPTURE_RE.match(filename)
    if match:
        prefix = match.group("prefix")
        session_id = f"s{match.group('session')}"
        segment_id = match.group("segment")
        base_video_key = f"{prefix}__{session_id}__seg_{segment_id}"
        return {
            "prefix": prefix,
            "session_id": session_id,
            "segment_id": segment_id,
            "sequence_id": None,
            "source_kind": "session_capture",
            "video_id": f"{base_video_key}__{label}",
            "identity_key": f"{prefix}__{session_id}",
            "frame_sort_key": (float(segment_id), int(match.group("frame")), int(match.group("crop"))),
        }

    match = _SEQUENCE_RE.match(filename)
    if match:
        prefix = match.group("prefix")
        sequence_id = f"seq{match.group('seq')}"
        base_video_key = f"{prefix}__{sequence_id}"
        return {
            "prefix": prefix,
            "session_id": None,
            "segment_id": None,
            "sequence_id": sequence_id,
            "source_kind": "flat_upload",
            "video_id": f"{base_video_key}__{label}",
            "identity_key": prefix,
            "frame_sort_key": (int(match.group("seq")), int(match.group("frame"))),
        }

    stem = Path(filename).stem
    prefix = filename.split("__", 1)[0] if "__" in filename else stem
    return {
        "prefix": prefix,
        "session_id": None,
        "segment_id": None,
        "sequence_id": None,
        "source_kind": "fallback_single",
        "video_id": f"{stem}__{label}",
        "identity_key": prefix,
        "frame_sort_key": (stem,),
    }


def discover_videos(source_root: str) -> List[DiscoveredVideo]:
    reader = StorageReader(source_root)
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for label in ("real", "fake"):
        for path in reader.iter_label_paths(label):
            filename = os.path.basename(path)
            meta = _parse_discovery_metadata(filename, label=label)
            key = (label, meta["video_id"])
            group = grouped.setdefault(
                key,
                {
                    "label": label,
                    "prefix": meta["prefix"],
                    "video_id": meta["video_id"],
                    "identity_key": meta["identity_key"],
                    "session_id": meta["session_id"],
                    "segment_id": meta["segment_id"],
                    "sequence_id": meta["sequence_id"],
                    "source_kind": meta["source_kind"],
                    "items": [],
                },
            )
            group["items"].append((meta["frame_sort_key"], path))

    videos: List[DiscoveredVideo] = []
    for group in grouped.values():
        ordered_paths = [path for _, path in sorted(group["items"], key=lambda item: item[0])]
        videos.append(
            DiscoveredVideo(
                label=group["label"],
                prefix=group["prefix"],
                video_id=group["video_id"],
                identity_key=group["identity_key"],
                frame_paths=ordered_paths,
                session_id=group["session_id"],
                segment_id=group["segment_id"],
                sequence_id=group["sequence_id"],
                source_kind=group["source_kind"],
            )
        )

    videos.sort(key=lambda item: (item.label, item.video_id))
    return videos


def _select_stat_paths(frame_paths: List[str], sample_count: int) -> List[str]:
    if sample_count <= 0 or len(frame_paths) <= sample_count:
        return list(frame_paths)
    indices = np.linspace(0, len(frame_paths) - 1, num=sample_count, dtype=int)
    return [frame_paths[idx] for idx in sorted(set(indices.tolist()))]


def _compute_frame_metrics(reader: StorageReader, path: str) -> Dict[str, float]:
    image = Image.open(io.BytesIO(reader.read_bytes(path))).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    grad_y, grad_x = np.gradient(gray)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    return {
        "mean_brightness": float(gray.mean()),
        "brightness_std": float(gray.std()),
        "contrast_rms": float(gray.std() / (gray.mean() + 1e-6)),
        "rb_ratio": float((rgb[:, :, 0].mean() + 1e-6) / (rgb[:, :, 2].mean() + 1e-6)),
        "sharpness_tenengrad": float(grad_mag.mean()),
    }


def _compute_video_metrics(
    reader: StorageReader,
    frame_paths: List[str],
    sample_count: int,
) -> Dict[str, float]:
    selected = _select_stat_paths(frame_paths, sample_count=sample_count)
    per_frame = [_compute_frame_metrics(reader, path) for path in selected]
    keys = per_frame[0].keys()
    return {
        key: round(float(np.mean([frame[key] for frame in per_frame])), 6)
        for key in keys
    }


def _get_thread_reader(source_root: str) -> StorageReader:
    reader = getattr(_THREAD_LOCAL, "reader", None)
    reader_root = getattr(_THREAD_LOCAL, "reader_root", None)
    if reader is None or reader_root != source_root:
        reader = StorageReader(source_root)
        _THREAD_LOCAL.reader = reader
        _THREAD_LOCAL.reader_root = source_root
    return reader


def _compute_real_row_metrics(
    source_root: str,
    frame_paths: List[str],
    sample_count: int,
) -> Dict[str, float]:
    reader = _get_thread_reader(source_root)
    return _compute_video_metrics(reader, frame_paths, sample_count=sample_count)


def _quantile(values: List[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float32), q))


def _resolve_rule(
    rules: Dict[str, Dict[str, Dict[str, Any]]],
    label: str,
    prefix: str,
    session_id: Optional[str],
) -> Tuple[Optional[str], Dict[str, Any]]:
    label_rules = rules.get(label, {})
    candidates = []
    if session_id:
        candidates.append(f"{prefix}__{session_id}")
    candidates.append(prefix)
    for key in candidates:
        if key in label_rules:
            return key, dict(label_rules[key])
    return None, {}


def _annotate_real_slices(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    if not rows:
        return {}

    sharpness = [row["quality_metrics"]["sharpness_tenengrad"] for row in rows]
    contrast = [row["quality_metrics"]["contrast_rms"] for row in rows]
    brightness_std = [row["quality_metrics"]["brightness_std"] for row in rows]
    mean_brightness = [row["quality_metrics"]["mean_brightness"] for row in rows]
    rb_ratio = [row["quality_metrics"]["rb_ratio"] for row in rows]

    thresholds = {
        "sharpness_tenengrad_q25": round(_quantile(sharpness, 0.25), 6),
        "contrast_rms_q25": round(_quantile(contrast, 0.25), 6),
        "brightness_std_q25": round(_quantile(brightness_std, 0.25), 6),
        "mean_brightness_q10": round(_quantile(mean_brightness, 0.10), 6),
        "mean_brightness_q90": round(_quantile(mean_brightness, 0.90), 6),
        "rb_ratio_q10": round(_quantile(rb_ratio, 0.10), 6),
        "rb_ratio_q90": round(_quantile(rb_ratio, 0.90), 6),
    }

    for row in rows:
        metrics = row["quality_metrics"]
        poor_quality_score = sum(
            [
                metrics["sharpness_tenengrad"] <= thresholds["sharpness_tenengrad_q25"],
                metrics["contrast_rms"] <= thresholds["contrast_rms_q25"],
                metrics["brightness_std"] <= thresholds["brightness_std_q25"],
            ]
        )
        lighting_extreme = (
            metrics["mean_brightness"] <= thresholds["mean_brightness_q10"]
            or metrics["mean_brightness"] >= thresholds["mean_brightness_q90"]
            or metrics["rb_ratio"] <= thresholds["rb_ratio_q10"]
            or metrics["rb_ratio"] >= thresholds["rb_ratio_q90"]
        )

        if poor_quality_score >= 2:
            row["slices"].append("teams_real_poor_quality")
        if lighting_extreme:
            row["slices"].append("teams_real_lighting_extreme")
        row["slices"] = sorted(set(row["slices"]))

    return {
        "poor_quality": {
            "sharpness_tenengrad_q25": thresholds["sharpness_tenengrad_q25"],
            "contrast_rms_q25": thresholds["contrast_rms_q25"],
            "brightness_std_q25": thresholds["brightness_std_q25"],
        },
        "lighting_extreme": {
            "mean_brightness_q10": thresholds["mean_brightness_q10"],
            "mean_brightness_q90": thresholds["mean_brightness_q90"],
            "rb_ratio_q10": thresholds["rb_ratio_q10"],
            "rb_ratio_q90": thresholds["rb_ratio_q90"],
        },
    }


def build_manifest(
    source_root: str,
    lockbox_ratio: float = 0.20,
    split_seed: int = 737,
    stats_frames_per_video: int = 3,
    prefix_rules_path: Optional[str] = None,
) -> Dict[str, Any]:
    if not 0.0 < lockbox_ratio < 1.0:
        raise ValueError(f"lockbox_ratio must be in (0, 1), got {lockbox_ratio}")
    if stats_frames_per_video <= 0:
        raise ValueError(
            f"stats_frames_per_video must be > 0, got {stats_frames_per_video}"
        )

    reader = StorageReader(source_root)
    rules = _load_prefix_rules(prefix_rules_path)
    videos = discover_videos(source_root)

    rows: List[Dict[str, Any]] = []
    real_row_indices: List[int] = []

    for video in videos:
        matched_rule_key, rule = _resolve_rule(
            rules,
            label=video.label,
            prefix=video.prefix,
            session_id=video.session_id,
        )

        if video.label == "real":
            method = str(rule.get("method") or "teams_real")
            slices = ["teams_real_all"] + [str(item) for item in rule.get("slices", [])]
        else:
            method = str(rule.get("method") or "teams_fake_unknown")
            slices = ["teams_fake_all"] + [str(item) for item in rule.get("slices", [])]

        split = (
            "lockbox"
            if _stable_fraction(video.identity_key, split_seed) < lockbox_ratio
            else "dev"
        )

        row: Dict[str, Any] = {
            "label": video.label,
            "method": method,
            "video_id": video.video_id,
            "frame_paths": video.frame_paths,
            "identity": _stable_int_hash(video.identity_key),
            "identity_key": video.identity_key,
            "split": split,
            "slices": sorted(set(slices)),
            "prefix": video.prefix,
            "session_id": video.session_id,
            "segment_id": video.segment_id,
            "sequence_id": video.sequence_id,
            "source_kind": video.source_kind,
            "matched_rule": matched_rule_key,
        }

        if video.label == "real":
            real_row_indices.append(len(rows))

        rows.append(row)

    if real_row_indices:
        worker_count = DEFAULT_METRIC_WORKERS
        log.info(
            "Computing real-video quality metrics for %d rows with %d workers",
            len(real_row_indices),
            worker_count,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(
                    _compute_real_row_metrics,
                    source_root,
                    rows[row_index]["frame_paths"],
                    stats_frames_per_video,
                ): row_index
                for row_index in real_row_indices
            }
            completed = 0
            for future in concurrent.futures.as_completed(future_to_index):
                row_index = future_to_index[future]
                rows[row_index]["quality_metrics"] = future.result()
                completed += 1
                if completed % 250 == 0 or completed == len(real_row_indices):
                    log.info(
                        "Computed real-video quality metrics: %d/%d",
                        completed,
                        len(real_row_indices),
                    )

    real_rows = [rows[row_index] for row_index in real_row_indices]
    thresholds = _annotate_real_slices(real_rows)

    label_counts = Counter(row["label"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    method_counts = Counter(row["method"] for row in rows)
    slice_counts = Counter(slice_name for row in rows for slice_name in row["slices"])

    return {
        "manifest_version": 1,
        "source_root": source_root,
        "split_seed": split_seed,
        "lockbox_ratio": lockbox_ratio,
        "stats_frames_per_video": stats_frames_per_video,
        "prefix_rules_path": prefix_rules_path,
        "quality_thresholds": thresholds,
        "summary": {
            "videos_total": len(rows),
            "label_counts": dict(sorted(label_counts.items())),
            "split_counts": dict(sorted(split_counts.items())),
            "method_counts": dict(sorted(method_counts.items())),
            "slice_counts": dict(sorted(slice_counts.items())),
        },
        "videos": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic Teams target-domain manifest"
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default="gs://teams-faces-data-test-2914-fake-4420-real-feb-28",
        help="Root directory or gs:// bucket containing real/ and fake/ children.",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--lockbox-ratio", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=737)
    parser.add_argument("--stats-frames-per-video", type=int, default=3)
    parser.add_argument(
        "--prefix-rules",
        type=str,
        default=None,
        help="Optional JSON/YAML mapping for known prefix or prefix__session provenance tags.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    manifest = build_manifest(
        source_root=args.source_root,
        lockbox_ratio=args.lockbox_ratio,
        split_seed=args.split_seed,
        stats_frames_per_video=args.stats_frames_per_video,
        prefix_rules_path=args.prefix_rules,
    )
    payload = json.dumps(manifest, indent=2, sort_keys=True)
    _write_text_to_path(args.output, payload + "\n")

    summary = manifest["summary"]
    log.info("Wrote manifest to %s", args.output)
    log.info("Videos total: %d", summary["videos_total"])
    log.info("Labels: %s", summary["label_counts"])
    log.info("Splits: %s", summary["split_counts"])
    log.info("Methods: %s", summary["method_counts"])


if __name__ == "__main__":
    main()
