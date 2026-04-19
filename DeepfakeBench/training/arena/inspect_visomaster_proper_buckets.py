#!/usr/bin/env python3
"""
Inspect VisoMaster proper-data candidate buckets before training integration.

This utility supports two workflows:

1. `census`
   Read manifest rows from one or more clean or Teams buckets and emit a stable
   summary of dataset keys, swap models, enhancers, frame-layout consistency,
   and join-key quality.

2. `validate-join`
   Compare two manifest sources and measure how well candidate keys line up
   across them. This is intended for clean-versus-Teams validation once the
   parallel buckets arrive.

The parser is intentionally manifest-driven. It does not assume the legacy
`visomaster_<swap_model>_<...>` sample naming pattern used by the older
training loader.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_JOIN_CANDIDATES: Tuple[str, ...] = (
    "real_id",
    "target_video_name",
    "clip_stem",
    "sample_id",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_gs_uri(uri: str) -> Tuple[str, str]:
    stripped = uri.replace("gs://", "", 1)
    if "/" not in stripped:
        return stripped, ""
    bucket, blob = stripped.split("/", 1)
    return bucket, blob


def _normalize_source_root(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Source root must be non-empty.")
    if text.startswith("gs://"):
        return text.rstrip("/")

    local_path = Path(text).expanduser()
    if local_path.exists():
        return str(local_path.resolve())

    return f"gs://{text.rstrip('/')}"


def _source_label(source_root: str) -> str:
    if source_root.startswith("gs://"):
        bucket, prefix = _split_gs_uri(source_root)
        if prefix:
            return f"{bucket}/{prefix}"
        return bucket

    path = Path(source_root)
    return path.name or str(path)


def _coalesce_string(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _sample_id_from_manifest_ref(manifest_ref: str) -> str:
    stripped = manifest_ref.replace("gs://", "", 1)
    parts = [part for part in stripped.split("/") if part]
    for index, part in enumerate(parts):
        if part == "samples" and index + 1 < len(parts):
            return parts[index + 1]
    return ""


def _sample_prefix(sample_id: str) -> str:
    text = str(sample_id or "").strip()
    if not text:
        return ""
    if "_" not in text:
        return text
    return text.split("_", 1)[0]


def _normalize_ext(filename: str) -> str:
    suffix = Path(str(filename or "")).suffix.lower()
    return suffix or ""


def _ext_list(frame_names: Sequence[str]) -> Tuple[str, ...]:
    exts = sorted({_normalize_ext(name) for name in frame_names if _normalize_ext(name)})
    return tuple(exts)


def _derive_identity_from_video_name(video_name: str) -> str:
    text = str(video_name or "").strip()
    if not text:
        return ""
    if "__" in text:
        return text.split("__", 1)[0]
    return ""


def _count_example_rows(
    mapping: Mapping[str, Sequence[str]],
    *,
    limit: int = 5,
    sample_limit: int = 5,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ordered = sorted(
        (
            (key, list(sample_ids))
            for key, sample_ids in mapping.items()
            if key and len(sample_ids) > 1
        ),
        key=lambda item: (-len(item[1]), item[0]),
    )
    for key, sample_ids in ordered[:limit]:
        rows.append(
            {
                "key": key,
                "count": len(sample_ids),
                "sample_ids": sample_ids[:sample_limit],
            }
        )
    return rows


def _counter_to_dict(counter: Counter[str]) -> Dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


@dataclass
class ProperBucketRecord:
    source_root: str
    source_label: str
    manifest_ref: str
    sample_id: str
    sample_prefix: str
    real_id: str
    dataset_key: str
    strategy: str
    task_strategy: str
    swap_model: str
    enhancer: str
    combo_key: str
    target_video_name: str
    clip_stem: str
    real_identity: str
    expected_total_per_stream: int
    cropped_real_count: int
    cropped_fake_count: int
    real_exts: Tuple[str, ...]
    fake_exts: Tuple[str, ...]
    candidate_keys: Dict[str, str]
    real_frame_names: Tuple[str, ...] = ()
    fake_frame_names: Tuple[str, ...] = ()


class ManifestSource:
    def __init__(self, source_root: str, project: Optional[str] = None):
        self.source_root = _normalize_source_root(source_root)
        self.source_label = _source_label(self.source_root)
        self.is_gcs = self.source_root.startswith("gs://")
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT") or "train-cvit2"

        if self.is_gcs:
            self.bucket_name, self.root_prefix = _split_gs_uri(self.source_root)
            self.root_prefix = self.root_prefix.strip("/")
            self._thread_local = threading.local()
            self._client_lock = threading.Lock()
            self._storage_clients: List[Any] = []
        else:
            self.root_path = Path(self.source_root)

    def _storage_client(self):
        from google.cloud import storage

        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = storage.Client(project=self.project)
            self._thread_local.client = client
            with self._client_lock:
                self._storage_clients.append(client)
        return client

    def close(self) -> None:
        if not self.is_gcs:
            return
        with self._client_lock:
            clients = list(self._storage_clients)
            self._storage_clients.clear()
        for client in clients:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def list_manifest_refs(self) -> List[str]:
        refs: List[str] = []

        if self.is_gcs:
            client = self._storage_client()
            prefix = "/".join(part for part in (self.root_prefix, "samples") if part).rstrip("/")
            prefix = f"{prefix}/" if prefix else "samples/"
            for blob in client.list_blobs(self.bucket_name, prefix=prefix):
                if blob.name.endswith("/manifest.json"):
                    refs.append(f"gs://{self.bucket_name}/{blob.name}")
            return sorted(refs)

        samples_root = self.root_path / "samples"
        search_root = samples_root if samples_root.exists() else self.root_path
        for path in sorted(search_root.rglob("manifest.json")):
            refs.append(str(path.resolve()))
        return refs

    def read_json(self, manifest_ref: str) -> Dict[str, Any]:
        if manifest_ref.startswith("gs://"):
            client = self._storage_client()
            bucket_name, blob_path = _split_gs_uri(manifest_ref)
            text = client.bucket(bucket_name).blob(blob_path).download_as_text()
            payload = json.loads(text)
        else:
            payload = json.loads(Path(manifest_ref).read_text())

        if not isinstance(payload, dict):
            raise ValueError(
                f"Manifest {manifest_ref} must decode to an object, got {type(payload).__name__}."
            )
        return payload


def extract_manifest_record(
    manifest: Mapping[str, Any],
    *,
    manifest_ref: str,
    source_root: str,
    source_label: str,
) -> ProperBucketRecord:
    original_manifest = (
        manifest.get("original_manifest", {})
        if isinstance(manifest.get("original_manifest"), dict)
        else {}
    )
    semantic_manifest: Mapping[str, Any] = original_manifest or manifest

    pair = (
        semantic_manifest.get("pair", {})
        if isinstance(semantic_manifest.get("pair"), dict)
        else {}
    )
    metadata = pair.get("metadata", {}) if isinstance(pair.get("metadata"), dict) else {}
    parameters = (
        metadata.get("parameters", {}) if isinstance(metadata.get("parameters"), dict) else {}
    )
    origin = metadata.get("origin", {}) if isinstance(metadata.get("origin"), dict) else {}
    target_meta = (
        metadata.get("target_metadata", {})
        if isinstance(metadata.get("target_metadata"), dict)
        else {}
    )
    real_video = (
        metadata.get("real_video", {}) if isinstance(metadata.get("real_video"), dict) else {}
    )
    fake_video = (
        metadata.get("fake_video", {}) if isinstance(metadata.get("fake_video"), dict) else {}
    )
    bucket_prep = (
        metadata.get("bucket_prep", {}) if isinstance(metadata.get("bucket_prep"), dict) else {}
    )
    frame_counts_top = (
        manifest.get("frame_counts", {}) if isinstance(manifest.get("frame_counts"), dict) else {}
    )
    frame_counts_semantic = (
        semantic_manifest.get("frame_counts", {})
        if isinstance(semantic_manifest.get("frame_counts"), dict)
        else {}
    )
    frame_files = (
        manifest.get("frame_files", {}) if isinstance(manifest.get("frame_files"), dict) else {}
    )

    real_files = [str(name).strip() for name in frame_files.get("real", []) if str(name).strip()]
    fake_files = [str(name).strip() for name in frame_files.get("fake", []) if str(name).strip()]

    sample_id = _coalesce_string(
        manifest.get("sample_id"),
        semantic_manifest.get("sample_id"),
        pair.get("sample_id"),
        origin.get("task_id"),
        _sample_id_from_manifest_ref(manifest_ref),
    )
    real_id = _coalesce_string(
        manifest.get("real_id"),
        semantic_manifest.get("real_id"),
        real_video.get("real_id"),
        bucket_prep.get("real_id"),
    )
    dataset_key = _coalesce_string(
        manifest.get("dataset_key"),
        semantic_manifest.get("dataset_key"),
        origin.get("dataset_key"),
        bucket_prep.get("dataset_key"),
    )
    strategy = _coalesce_string(
        pair.get("strategy"),
        semantic_manifest.get("strategy"),
        manifest.get("strategy"),
        dataset_key,
    )
    task_strategy = _coalesce_string(origin.get("task_strategy"), strategy)
    method_name = _coalesce_string(manifest.get("method_name"))
    method_swap_model = method_name.split("__", 1)[0] if "__" in method_name else method_name
    method_enhancer = method_name.split("__", 1)[1] if "__" in method_name else ""
    swap_model = _coalesce_string(
        manifest.get("swap_model"),
        semantic_manifest.get("swap_model"),
        parameters.get("swap_model"),
        origin.get("swap_model"),
        fake_video.get("swap_model"),
        method_swap_model,
    )
    enhancer = _coalesce_string(
        parameters.get("enhancer"),
        origin.get("enhancer"),
        fake_video.get("enhancer"),
        method_enhancer,
        "None",
    )
    combo_key = _coalesce_string(
        origin.get("combo_key"),
        fake_video.get("combo_key"),
        f"{swap_model}|{enhancer}" if swap_model else "",
    )
    target_video_name = _coalesce_string(
        target_meta.get("video_name"),
        fake_video.get("video_name"),
        real_video.get("clip_stem"),
    )
    clip_stem = _coalesce_string(real_video.get("clip_stem"), target_video_name)
    real_identity = _coalesce_string(
        target_meta.get("real_identity"),
        _derive_identity_from_video_name(target_video_name),
    )
    expected_total_per_stream = _safe_int(
        frame_counts_top.get("target_per_stream"),
        default=_safe_int(frame_counts_semantic.get("expected_total_per_stream"), default=0),
    )
    cropped_real_count = _safe_int(
        frame_counts_top.get("selected_real"),
        default=_safe_int(frame_counts_semantic.get("cropped_real"), default=len(real_files)),
    )
    cropped_fake_count = _safe_int(
        frame_counts_top.get("selected_fake"),
        default=_safe_int(frame_counts_semantic.get("cropped_fake"), default=len(fake_files)),
    )

    candidate_keys = {
        "sample_id": sample_id,
        "real_id": real_id,
        "target_video_name": target_video_name,
        "clip_stem": clip_stem,
    }

    return ProperBucketRecord(
        source_root=source_root,
        source_label=source_label,
        manifest_ref=manifest_ref,
        sample_id=sample_id,
        sample_prefix=_sample_prefix(sample_id),
        real_id=real_id,
        dataset_key=dataset_key,
        strategy=strategy,
        task_strategy=task_strategy,
        swap_model=swap_model,
        enhancer=enhancer,
        combo_key=combo_key,
        target_video_name=target_video_name,
        clip_stem=clip_stem,
        real_identity=real_identity,
        expected_total_per_stream=expected_total_per_stream,
        cropped_real_count=cropped_real_count,
        cropped_fake_count=cropped_fake_count,
        real_exts=_ext_list(real_files),
        fake_exts=_ext_list(fake_files),
        real_frame_names=tuple(real_files),
        fake_frame_names=tuple(fake_files),
        candidate_keys=candidate_keys,
    )


def load_manifest_records(
    source_root: str,
    *,
    project: Optional[str] = None,
) -> Dict[str, Any]:
    source = ManifestSource(source_root, project=project)
    manifest_refs = source.list_manifest_refs()
    records: List[ProperBucketRecord] = []
    load_errors: List[Dict[str, str]] = []

    def _load_one(manifest_ref: str) -> ProperBucketRecord:
        manifest = source.read_json(manifest_ref)
        return extract_manifest_record(
            manifest,
            manifest_ref=manifest_ref,
            source_root=source.source_root,
            source_label=source.source_label,
        )

    def _capture_error(manifest_ref: str, exc: Exception) -> None:
        load_errors.append({"manifest_ref": manifest_ref, "error": str(exc)})

    max_workers = min(16, max(4, os.cpu_count() or 4), len(manifest_refs) or 1)

    try:
        if max_workers <= 1:
            for manifest_ref in manifest_refs:
                try:
                    records.append(_load_one(manifest_ref))
                except Exception as exc:
                    _capture_error(manifest_ref, exc)
        else:
            future_to_ref = {}
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for manifest_ref in manifest_refs:
                    future_to_ref[pool.submit(_load_one, manifest_ref)] = manifest_ref
                for future in as_completed(future_to_ref):
                    manifest_ref = future_to_ref[future]
                    try:
                        records.append(future.result())
                    except Exception as exc:
                        _capture_error(manifest_ref, exc)
    finally:
        source.close()

    records.sort(key=lambda record: (record.sample_id, record.manifest_ref))
    load_errors.sort(key=lambda row: row["manifest_ref"])

    return {
        "source_root": source.source_root,
        "source_label": source.source_label,
        "manifest_count": len(manifest_refs),
        "records": records,
        "load_errors": load_errors,
    }


def _candidate_key_stats(
    records: Sequence[ProperBucketRecord],
    candidate_key: str,
) -> Dict[str, Any]:
    key_to_sample_ids: DefaultDict[str, List[str]] = defaultdict(list)
    missing_sample_ids: List[str] = []

    for record in records:
        key_value = str(record.candidate_keys.get(candidate_key, "")).strip()
        if not key_value:
            missing_sample_ids.append(record.sample_id)
            continue
        key_to_sample_ids[key_value].append(record.sample_id)

    duplicate_key_count = sum(1 for sample_ids in key_to_sample_ids.values() if len(sample_ids) > 1)
    max_samples_per_key = max((len(sample_ids) for sample_ids in key_to_sample_ids.values()), default=0)
    present_count = len(records) - len(missing_sample_ids)

    return {
        "present_count": present_count,
        "missing_count": len(missing_sample_ids),
        "unique_key_count": len(key_to_sample_ids),
        "duplicate_key_count": duplicate_key_count,
        "max_samples_per_key": max_samples_per_key,
        "stable_one_to_one": (
            len(records) > 0
            and present_count == len(records)
            and len(key_to_sample_ids) == len(records)
            and duplicate_key_count == 0
        ),
        "duplicate_examples": _count_example_rows(key_to_sample_ids),
        "missing_sample_ids": missing_sample_ids[:5],
    }


def _example_records(records: Sequence[ProperBucketRecord], limit: int = 3) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records[:limit]:
        rows.append(
            {
                "sample_id": record.sample_id,
                "real_id": record.real_id,
                "dataset_key": record.dataset_key,
                "swap_model": record.swap_model,
                "enhancer": record.enhancer,
                "target_video_name": record.target_video_name,
            }
        )
    return rows


def build_census_report(
    source_roots: Sequence[str],
    *,
    project: Optional[str] = None,
    candidate_keys: Sequence[str] = DEFAULT_JOIN_CANDIDATES,
) -> Dict[str, Any]:
    sources: List[Dict[str, Any]] = []

    for source_root in source_roots:
        load_result = load_manifest_records(source_root, project=project)
        records = load_result["records"]

        dataset_keys: Counter[str] = Counter()
        strategies: Counter[str] = Counter()
        task_strategies: Counter[str] = Counter()
        swap_models: Counter[str] = Counter()
        enhancers: Counter[str] = Counter()
        combo_keys: Counter[str] = Counter()
        sample_prefixes: Counter[str] = Counter()
        real_frame_extensions: Counter[str] = Counter()
        fake_frame_extensions: Counter[str] = Counter()
        real_frame_counts: Counter[str] = Counter()
        fake_frame_counts: Counter[str] = Counter()
        missing_field_counts: Counter[str] = Counter()

        real_fake_count_equal = 0
        real_fake_count_mismatch = 0
        expected_total_matches_both = 0
        expected_total_mismatch = 0

        for record in records:
            if record.dataset_key:
                dataset_keys[record.dataset_key] += 1
            else:
                missing_field_counts["dataset_key"] += 1

            if record.strategy:
                strategies[record.strategy] += 1
            else:
                missing_field_counts["strategy"] += 1

            if record.task_strategy:
                task_strategies[record.task_strategy] += 1
            else:
                missing_field_counts["task_strategy"] += 1

            if record.swap_model:
                swap_models[record.swap_model] += 1
            else:
                missing_field_counts["swap_model"] += 1

            if record.enhancer:
                enhancers[record.enhancer] += 1
            else:
                missing_field_counts["enhancer"] += 1

            if record.combo_key:
                combo_keys[record.combo_key] += 1
            else:
                missing_field_counts["combo_key"] += 1

            if record.sample_prefix:
                sample_prefixes[record.sample_prefix] += 1
            else:
                missing_field_counts["sample_prefix"] += 1

            if not record.sample_id:
                missing_field_counts["sample_id"] += 1
            if not record.real_id:
                missing_field_counts["real_id"] += 1
            if not record.target_video_name:
                missing_field_counts["target_video_name"] += 1
            if not record.clip_stem:
                missing_field_counts["clip_stem"] += 1
            if not record.real_identity:
                missing_field_counts["real_identity"] += 1

            if record.cropped_real_count == record.cropped_fake_count:
                real_fake_count_equal += 1
            else:
                real_fake_count_mismatch += 1

            if (
                record.expected_total_per_stream > 0
                and record.cropped_real_count == record.expected_total_per_stream
                and record.cropped_fake_count == record.expected_total_per_stream
            ):
                expected_total_matches_both += 1
            elif record.expected_total_per_stream > 0:
                expected_total_mismatch += 1

            real_frame_counts[str(record.cropped_real_count)] += 1
            fake_frame_counts[str(record.cropped_fake_count)] += 1

            for ext in record.real_exts:
                real_frame_extensions[ext] += 1
            for ext in record.fake_exts:
                fake_frame_extensions[ext] += 1

        sources.append(
            {
                "source_root": load_result["source_root"],
                "source_label": load_result["source_label"],
                "manifest_count": load_result["manifest_count"],
                "sample_count": len(records),
                "load_error_count": len(load_result["load_errors"]),
                "load_errors": load_result["load_errors"][:10],
                "examples": _example_records(records),
                "counts": {
                    "dataset_keys": _counter_to_dict(dataset_keys),
                    "strategies": _counter_to_dict(strategies),
                    "task_strategies": _counter_to_dict(task_strategies),
                    "swap_models": _counter_to_dict(swap_models),
                    "enhancers": _counter_to_dict(enhancers),
                    "combo_keys": _counter_to_dict(combo_keys),
                    "sample_prefixes": _counter_to_dict(sample_prefixes),
                    "real_frame_extensions": _counter_to_dict(real_frame_extensions),
                    "fake_frame_extensions": _counter_to_dict(fake_frame_extensions),
                    "real_frame_counts": _counter_to_dict(real_frame_counts),
                    "fake_frame_counts": _counter_to_dict(fake_frame_counts),
                },
                "frame_layout": {
                    "real_fake_count_equal": real_fake_count_equal,
                    "real_fake_count_mismatch": real_fake_count_mismatch,
                    "expected_total_matches_both": expected_total_matches_both,
                    "expected_total_mismatch": expected_total_mismatch,
                },
                "missing_field_counts": _counter_to_dict(missing_field_counts),
                "join_key_candidates": {
                    candidate_key: _candidate_key_stats(records, candidate_key)
                    for candidate_key in candidate_keys
                },
            }
        )

    return {
        "schema_name": "visomaster_proper_bucket_census_v1",
        "generated_at": _now_iso(),
        "candidate_keys": list(candidate_keys),
        "sources": sources,
    }


def _join_candidate_report(
    left_records: Sequence[ProperBucketRecord],
    right_records: Sequence[ProperBucketRecord],
    *,
    candidate_key: str,
) -> Dict[str, Any]:
    left_map: DefaultDict[str, List[str]] = defaultdict(list)
    right_map: DefaultDict[str, List[str]] = defaultdict(list)
    left_missing: List[str] = []
    right_missing: List[str] = []

    for record in left_records:
        key_value = str(record.candidate_keys.get(candidate_key, "")).strip()
        if not key_value:
            left_missing.append(record.sample_id)
            continue
        left_map[key_value].append(record.sample_id)

    for record in right_records:
        key_value = str(record.candidate_keys.get(candidate_key, "")).strip()
        if not key_value:
            right_missing.append(record.sample_id)
            continue
        right_map[key_value].append(record.sample_id)

    overlap_keys = sorted(set(left_map) & set(right_map))
    left_only_keys = sorted(set(left_map) - set(right_map))
    right_only_keys = sorted(set(right_map) - set(left_map))
    one_to_one_overlap_keys = [
        key
        for key in overlap_keys
        if len(left_map[key]) == 1 and len(right_map[key]) == 1
    ]
    ambiguous_overlap_keys = [
        key
        for key in overlap_keys
        if len(left_map[key]) != 1 or len(right_map[key]) != 1
    ]

    left_duplicate_key_count = sum(1 for sample_ids in left_map.values() if len(sample_ids) > 1)
    right_duplicate_key_count = sum(1 for sample_ids in right_map.values() if len(sample_ids) > 1)

    left_stable = (
        len(left_records) > 0
        and not left_missing
        and len(left_map) == len(left_records)
        and left_duplicate_key_count == 0
    )
    right_stable = (
        len(right_records) > 0
        and not right_missing
        and len(right_map) == len(right_records)
        and right_duplicate_key_count == 0
    )

    smaller_unique_set = min(len(left_map), len(right_map)) if left_map and right_map else 0
    match_rate_vs_smaller_unique_set = (
        len(one_to_one_overlap_keys) / smaller_unique_set
        if smaller_unique_set > 0
        else 0.0
    )

    return {
        "candidate": candidate_key,
        "left_present_count": len(left_records) - len(left_missing),
        "right_present_count": len(right_records) - len(right_missing),
        "left_missing_count": len(left_missing),
        "right_missing_count": len(right_missing),
        "left_unique_key_count": len(left_map),
        "right_unique_key_count": len(right_map),
        "left_duplicate_key_count": left_duplicate_key_count,
        "right_duplicate_key_count": right_duplicate_key_count,
        "left_stable_one_to_one": left_stable,
        "right_stable_one_to_one": right_stable,
        "overlap_key_count": len(overlap_keys),
        "one_to_one_overlap_key_count": len(one_to_one_overlap_keys),
        "ambiguous_overlap_key_count": len(ambiguous_overlap_keys),
        "left_only_key_count": len(left_only_keys),
        "right_only_key_count": len(right_only_keys),
        "match_rate_vs_smaller_unique_set": match_rate_vs_smaller_unique_set,
        "sample_overlap_keys": overlap_keys[:5],
        "sample_left_only_keys": left_only_keys[:5],
        "sample_right_only_keys": right_only_keys[:5],
        "left_duplicate_examples": _count_example_rows(left_map),
        "right_duplicate_examples": _count_example_rows(right_map),
        "left_missing_sample_ids": left_missing[:5],
        "right_missing_sample_ids": right_missing[:5],
    }


def _recommend_join_candidate(candidate_reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ranked = sorted(
        candidate_reports,
        key=lambda report: (
            int(report["left_stable_one_to_one"] and report["right_stable_one_to_one"]),
            float(report["match_rate_vs_smaller_unique_set"]),
            int(report["one_to_one_overlap_key_count"]),
            -int(report["ambiguous_overlap_key_count"]),
            -int(report["left_missing_count"] + report["right_missing_count"]),
        ),
        reverse=True,
    )
    if not ranked:
        return {}

    best = ranked[0]
    if best["one_to_one_overlap_key_count"] <= 0:
        return {
            "candidate": best["candidate"],
            "confidence": "none",
            "reason": "No one-to-one overlap was found for any candidate key.",
        }

    if best["left_stable_one_to_one"] and best["right_stable_one_to_one"]:
        confidence = "strong"
        reason = (
            "Candidate is one-to-one inside both sources and has the strongest "
            "cross-source overlap."
        )
    else:
        confidence = "provisional"
        reason = (
            "Candidate has the strongest overlap, but at least one side has "
            "missing or duplicate values."
        )

    return {
        "candidate": best["candidate"],
        "confidence": confidence,
        "reason": reason,
    }


def build_join_validation_report(
    left_source_root: str,
    right_source_root: str,
    *,
    project: Optional[str] = None,
    candidate_keys: Sequence[str] = DEFAULT_JOIN_CANDIDATES,
) -> Dict[str, Any]:
    left = load_manifest_records(left_source_root, project=project)
    right = load_manifest_records(right_source_root, project=project)

    candidate_reports = [
        _join_candidate_report(
            left["records"],
            right["records"],
            candidate_key=candidate_key,
        )
        for candidate_key in candidate_keys
    ]

    return {
        "schema_name": "visomaster_proper_bucket_join_validation_v1",
        "generated_at": _now_iso(),
        "candidate_keys": list(candidate_keys),
        "left_source": {
            "source_root": left["source_root"],
            "source_label": left["source_label"],
            "manifest_count": left["manifest_count"],
            "sample_count": len(left["records"]),
            "load_error_count": len(left["load_errors"]),
            "load_errors": left["load_errors"][:10],
        },
        "right_source": {
            "source_root": right["source_root"],
            "source_label": right["source_label"],
            "manifest_count": right["manifest_count"],
            "sample_count": len(right["records"]),
            "load_error_count": len(right["load_errors"]),
            "load_errors": right["load_errors"][:10],
        },
        "candidate_reports": candidate_reports,
        "recommended_candidate": _recommend_join_candidate(candidate_reports),
    }


def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _print_census_summary(report: Mapping[str, Any], output: Optional[str]) -> None:
    if output:
        print(f"Wrote {output}")
    for source in report.get("sources", []):
        print(
            f"{source['source_label']}: "
            f"{source['sample_count']} samples, "
            f"{source['load_error_count']} load errors"
        )
        for candidate_key, stats in source.get("join_key_candidates", {}).items():
            stable = "stable" if stats.get("stable_one_to_one") else "not_stable"
            print(
                f"  {candidate_key}: "
                f"present={stats['present_count']} "
                f"unique={stats['unique_key_count']} "
                f"duplicates={stats['duplicate_key_count']} "
                f"{stable}"
            )


def _print_join_summary(report: Mapping[str, Any], output: Optional[str]) -> None:
    if output:
        print(f"Wrote {output}")
    left = report.get("left_source", {})
    right = report.get("right_source", {})
    print(
        f"{left.get('source_label', 'left')} -> {right.get('source_label', 'right')}: "
        f"{left.get('sample_count', 0)} vs {right.get('sample_count', 0)} samples"
    )
    for candidate_report in report.get("candidate_reports", []):
        print(
            f"  {candidate_report['candidate']}: "
            f"one_to_one_overlap={candidate_report['one_to_one_overlap_key_count']} "
            f"ambiguous_overlap={candidate_report['ambiguous_overlap_key_count']} "
            f"match_rate={candidate_report['match_rate_vs_smaller_unique_set']:.4f}"
        )
    recommendation = report.get("recommended_candidate", {})
    if recommendation:
        print(
            "Recommended candidate: "
            f"{recommendation.get('candidate', 'n/a')} "
            f"({recommendation.get('confidence', 'n/a')})"
        )
        reason = recommendation.get("reason")
        if reason:
            print(f"  {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    census_parser = subparsers.add_parser(
        "census",
        help="Inspect one or more manifest sources and emit a bucket census.",
    )
    census_parser.add_argument(
        "--source",
        dest="sources",
        action="append",
        required=True,
        help=(
            "Bucket name, gs:// URI, or local path. Repeat for multiple sources. "
            "If a local path contains a `samples/` directory it will be used automatically."
        ),
    )
    census_parser.add_argument(
        "--project",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT") or "train-cvit2",
        help="GCP project for the storage client when reading from GCS.",
    )
    census_parser.add_argument(
        "--output",
        help="Optional JSON output path. If omitted, the report is printed to stdout.",
    )

    join_parser = subparsers.add_parser(
        "validate-join",
        help="Compare two manifest sources and measure join-key overlap.",
    )
    join_parser.add_argument("--left-source", required=True, help="Left bucket, gs:// URI, or local path.")
    join_parser.add_argument("--right-source", required=True, help="Right bucket, gs:// URI, or local path.")
    join_parser.add_argument(
        "--project",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT") or "train-cvit2",
        help="GCP project for the storage client when reading from GCS.",
    )
    join_parser.add_argument(
        "--candidate-key",
        dest="candidate_keys",
        action="append",
        choices=list(DEFAULT_JOIN_CANDIDATES),
        help="Override the candidate keys to compare. Repeat to add more.",
    )
    join_parser.add_argument(
        "--output",
        help="Optional JSON output path. If omitted, the report is printed to stdout.",
    )

    args = parser.parse_args()

    if args.command == "census":
        report = build_census_report(args.sources, project=args.project)
        if args.output:
            _write_json(args.output, report)
        else:
            print(json.dumps(report, indent=2, sort_keys=False))
        _print_census_summary(report, args.output)
        return

    if args.command == "validate-join":
        candidate_keys = tuple(args.candidate_keys or DEFAULT_JOIN_CANDIDATES)
        report = build_join_validation_report(
            args.left_source,
            args.right_source,
            project=args.project,
            candidate_keys=candidate_keys,
        )
        if args.output:
            _write_json(args.output, report)
        else:
            print(json.dumps(report, indent=2, sort_keys=False))
        _print_join_summary(report, args.output)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
