"""
Runtime helpers for inventory-backed proper-data training samples.

This loader reads the explicit WT-F inventory contract and turns each
transport-specific fake variant into one paired training sample. Real variants
stay explicit in the inventory, but the training path pairs each fake variant
with the matching real transport so the new lanes remain honest without being
collapsed into legacy VisoMaster aliases.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_PARALLEL_FRAME_DOWNLOAD_WORKERS = 4


@dataclass(frozen=True)
class ProperDataPairedSample:
    """Paired clean/Teams proper-data sample derived from one fake variant."""

    sample_id: str
    base_capture_id: str
    identity_id: str
    capture_session_id: str
    split_group_id: str
    source: str
    method: str
    transport: str
    enhancement: str
    generator_family: str
    generator_method: str
    quality_band: str
    face_scale_band: str
    real_frame_paths: Tuple[str, ...]
    fake_frame_paths: Tuple[str, ...]
    inventory_path: str = ""
    manifest_path: str = ""
    wave_id: str = ""


def _split_gs_uri(uri: str) -> Tuple[str, str]:
    stripped = str(uri or "").replace("gs://", "", 1)
    if "/" not in stripped:
        return stripped, ""
    bucket, blob = stripped.split("/", 1)
    return bucket, blob


def _normalize_method_name(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _slugify(value: str) -> str:
    safe = []
    previous_sep = False
    for char in str(value or "").strip().lower():
        if char.isalnum():
            safe.append(char)
            previous_sep = False
            continue
        if not previous_sep:
            safe.append("_")
            previous_sep = True
    return "".join(safe).strip("_") or "unknown"


def _normalize_path_token(path: str) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    if text.startswith("gs://"):
        return text.rstrip("/")

    candidate = Path(text).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return str(candidate)


def _find_path_subsequence(parts: Tuple[str, ...], needle: Tuple[str, ...]) -> int:
    if not parts or not needle or len(needle) > len(parts):
        return -1
    for index in range(len(parts) - len(needle) + 1):
        if tuple(parts[index : index + len(needle)]) == needle:
            return index
    return -1


def _local_path_aliases(path: str) -> set[str]:
    text = str(path or "").strip()
    if not text:
        return set()

    aliases: set[str] = set()

    def _record(candidate: str) -> None:
        candidate_text = str(candidate or "").strip()
        if not candidate_text:
            return
        aliases.add(candidate_text.rstrip("/") or candidate_text)

    def _record_repo_variants(candidate: str) -> None:
        candidate_text = str(candidate or "").strip()
        if not candidate_text or candidate_text.startswith("gs://"):
            return

        path_obj = Path(candidate_text)
        parts = tuple(path_obj.parts)
        repo_anchor = ("DeepfakeBench", "training")
        repo_index = _find_path_subsequence(parts, repo_anchor)
        if repo_index >= 0:
            repo_relative = Path(*parts[repo_index + len(repo_anchor) :])
            if str(repo_relative):
                _record(str(repo_relative))
                _record(str(Path("/workspace") / repo_relative))

        if candidate_text == "/workspace":
            return
        workspace_prefix = "/workspace/"
        if candidate_text.startswith(workspace_prefix):
            workspace_relative = candidate_text[len(workspace_prefix) :]
            if workspace_relative:
                _record(workspace_relative)
                _record(str(Path("DeepfakeBench") / "training" / workspace_relative))

    _record(text)
    candidate = Path(text).expanduser()
    _record(str(candidate))
    if candidate.exists():
        resolved = str(candidate.resolve())
        _record(resolved)

    snapshot = list(aliases)
    for alias in snapshot:
        _record_repo_variants(alias)

    return aliases


def _path_tokens_match(expected_path: str, observed_path: str) -> bool:
    expected = _normalize_path_token(expected_path)
    observed = _normalize_path_token(observed_path)
    if expected == observed:
        return True

    if expected.startswith("gs://") or observed.startswith("gs://"):
        return False

    expected_aliases = _local_path_aliases(expected_path)
    observed_aliases = _local_path_aliases(observed_path)
    return not expected_aliases.isdisjoint(observed_aliases)


def _read_text_from_path(path: str) -> str:
    if str(path).startswith("gs://"):
        from google.cloud import storage

        bucket_name, blob_path = _split_gs_uri(path)
        client = storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))
        return client.bucket(bucket_name).blob(blob_path).download_as_text()

    return Path(path).read_text()


def _load_structured_data(path: str) -> Dict[str, Any]:
    raw = _read_text_from_path(path)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = yaml.safe_load(raw)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping at {path}, got {type(payload).__name__}.")
    return payload


def _list_images(root: str) -> List[str]:
    root = str(root or "").strip()
    if not root:
        return []

    if root.startswith("gs://"):
        from google.cloud import storage

        bucket_name, prefix = _split_gs_uri(root.rstrip("/"))
        client = storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))
        results: List[str] = []
        for blob in client.list_blobs(bucket_name, prefix=prefix.rstrip("/") + "/"):
            if blob.name.endswith("/"):
                continue
            if Path(blob.name).suffix.lower() not in IMAGE_EXTS:
                continue
            results.append(f"gs://{bucket_name}/{blob.name}")
        return sorted(results)

    base = Path(root).expanduser().resolve()
    if not base.exists():
        return []
    return sorted(
        str(path.resolve())
        for path in base.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def _coerce_string(value: Any, *, field_name: str, context: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"Missing required field '{field_name}' in {context}.")
    return text


def _coerce_enum(
    value: Any,
    *,
    allowed: Sequence[str],
    field_name: str,
    context: str,
) -> str:
    text = _coerce_string(value, field_name=field_name, context=context).lower()
    if text not in set(allowed):
        raise ValueError(
            f"Invalid value for '{field_name}' in {context}: {text!r}. "
            f"Expected one of {sorted(set(allowed))}."
        )
    return text


def _resolve_frame_paths(
    variant: Mapping[str, Any],
    *,
    context: str,
) -> Tuple[str, ...]:
    has_root = bool(str(variant.get("frame_root") or "").strip())
    has_paths = variant.get("frame_paths") is not None

    if has_root and has_paths:
        raise ValueError(
            f"{context} must set exactly one of 'frame_root' or 'frame_paths', not both."
        )
    if not has_root and not has_paths:
        raise ValueError(f"{context} must set one of 'frame_root' or 'frame_paths'.")

    if has_root:
        frame_paths = _list_images(
            _coerce_string(variant.get("frame_root"), field_name="frame_root", context=context)
        )
    else:
        raw_paths = variant.get("frame_paths")
        if not isinstance(raw_paths, list):
            raise ValueError(f"{context}.frame_paths must be a list.")
        frame_paths = []
        for index, raw in enumerate(raw_paths):
            text = str(raw).strip() if raw is not None else ""
            if not text:
                raise ValueError(f"{context}.frame_paths[{index}] must be a non-empty string.")
            frame_paths.append(text)

    if not frame_paths:
        raise ValueError(f"{context} resolved zero frames.")
    return tuple(sorted(frame_paths))


def _lane_for_fake_variant(
    *,
    transport: str,
    enhancement: str,
    generator_family: str,
    context: str,
) -> str:
    if generator_family != "visomaster":
        raise ValueError(
            f"{context} must declare generator_family='visomaster' for fake proper-data variants."
        )

    if transport == "clean" and enhancement == "none":
        return "proper_visomaster_clean"
    if transport == "teams" and enhancement == "none":
        return "proper_visomaster_teams"
    if transport == "clean" and enhancement == "enhanced":
        return "proper_visomaster_enhanced_clean"
    if transport == "teams" and enhancement == "enhanced":
        return "proper_visomaster_enhanced_teams"
    raise ValueError(f"Unsupported proper-data fake lane in {context}.")


def _method_name(lane: str, generator_method: str) -> str:
    return f"{lane}__{_slugify(generator_method)}"


def _validate_manifest_reference(
    manifest: Mapping[str, Any],
    *,
    inventory: Mapping[str, Any],
    inventory_path: str,
    manifest_path: str,
) -> Dict[str, Any]:
    wave_id = str(manifest.get("wave_id") or "").strip()
    inventory_wave_id = str(inventory.get("wave_id") or "").strip()
    if wave_id and inventory_wave_id and wave_id != inventory_wave_id:
        raise ValueError(
            f"Proper-data manifest wave_id mismatch: inventory={inventory_wave_id!r} "
            f"manifest={wave_id!r} ({manifest_path})."
        )

    manifest_inventory = str(manifest.get("source_inventory") or "").strip()
    if manifest_inventory:
        expected = _normalize_path_token(inventory_path)
        observed = _normalize_path_token(manifest_inventory)
        if not _path_tokens_match(inventory_path, manifest_inventory):
            raise ValueError(
                f"Proper-data manifest source_inventory mismatch: expected {expected!r}, "
                f"got {observed!r} ({manifest_path})."
            )

    summary = manifest.get("summary") or {}
    if summary and not isinstance(summary, dict):
        raise ValueError(
            f"Proper-data manifest summary must be a mapping when present ({manifest_path})."
        )
    return dict(summary) if isinstance(summary, dict) else {}


def _build_samples_from_inventory(
    inventory: Mapping[str, Any],
    *,
    inventory_path: str,
    manifest_path: str,
    include_lanes: Optional[Sequence[str]] = None,
) -> Tuple[List[ProperDataPairedSample], Dict[str, Any]]:
    inventory_version = int(inventory.get("inventory_version", 0) or 0)
    if inventory_version != 1:
        raise ValueError(
            f"Unsupported proper-data inventory_version={inventory_version!r}; expected 1."
        )

    wave_id = _coerce_string(inventory.get("wave_id"), field_name="wave_id", context="inventory")
    captures = inventory.get("captures")
    if not isinstance(captures, list) or not captures:
        raise ValueError("Proper-data inventory must contain a non-empty 'captures' list.")

    include_lane_set = {
        _normalize_method_name(lane)
        for lane in (include_lanes or [])
        if str(lane).strip()
    }

    samples: List[ProperDataPairedSample] = []
    seen_sample_ids = set()
    ragged_pairs: List[Dict[str, Any]] = []
    discovered_lane_counts: Counter[str] = Counter()

    for capture_index, capture in enumerate(captures):
        if not isinstance(capture, dict):
            raise ValueError(f"captures[{capture_index}] must be a mapping.")
        capture_context = f"captures[{capture_index}]"

        base_capture_id = _coerce_string(
            capture.get("base_capture_id"),
            field_name="base_capture_id",
            context=capture_context,
        )
        identity_id = _coerce_string(
            capture.get("identity_id"),
            field_name="identity_id",
            context=capture_context,
        )
        capture_session_id = _coerce_string(
            capture.get("capture_session_id"),
            field_name="capture_session_id",
            context=capture_context,
        )
        split_group_id = _coerce_string(
            capture.get("split_group_id"),
            field_name="split_group_id",
            context=capture_context,
        )
        quality_band = _coerce_enum(
            capture.get("quality_band"),
            allowed=("high", "medium", "low"),
            field_name="quality_band",
            context=capture_context,
        )
        face_scale_band = _coerce_enum(
            capture.get("face_scale_band"),
            allowed=("big_face", "standard"),
            field_name="face_scale_band",
            context=capture_context,
        )

        variants = capture.get("variants")
        if not isinstance(variants, list) or not variants:
            raise ValueError(f"{capture_context} must contain a non-empty 'variants' list.")

        real_by_transport: Dict[str, Tuple[str, ...]] = {}
        ordered_fake_variants: List[Tuple[Mapping[str, Any], str, str, str, str]] = []

        for variant_index, variant in enumerate(variants):
            if not isinstance(variant, dict):
                raise ValueError(f"{capture_context}.variants[{variant_index}] must be a mapping.")
            variant_context = f"{capture_context}.variants[{variant_index}]"
            label = _coerce_enum(
                variant.get("label"),
                allowed=("real", "fake"),
                field_name="label",
                context=variant_context,
            )
            transport = _coerce_enum(
                variant.get("transport"),
                allowed=("clean", "teams"),
                field_name="transport",
                context=variant_context,
            )
            enhancement = _coerce_enum(
                variant.get("enhancement"),
                allowed=("none", "enhanced"),
                field_name="enhancement",
                context=variant_context,
            )
            frame_paths = _resolve_frame_paths(variant, context=variant_context)

            if label == "real":
                if enhancement != "none":
                    raise ValueError(
                        f"{variant_context} is real but enhancement={enhancement!r}; real variants must use 'none'."
                    )
                if transport in real_by_transport:
                    raise ValueError(
                        f"{capture_context} declares multiple real variants for transport={transport!r}."
                    )
                real_by_transport[transport] = frame_paths
                continue

            generator_family = _coerce_enum(
                variant.get("generator_family"),
                allowed=("visomaster",),
                field_name="generator_family",
                context=variant_context,
            )
            generator_method = _coerce_string(
                variant.get("generator_method"),
                field_name="generator_method",
                context=variant_context,
            )
            ordered_fake_variants.append(
                (variant, transport, enhancement, generator_family, generator_method)
            )

        for variant, transport, enhancement, generator_family, generator_method in ordered_fake_variants:
            variant_context = f"{capture_context}::{variant.get('variant_id', 'unknown_variant')}"
            if transport not in real_by_transport:
                raise ValueError(
                    f"{variant_context} has no matching real variant for transport={transport!r}."
                )

            lane = _lane_for_fake_variant(
                transport=transport,
                enhancement=enhancement,
                generator_family=generator_family,
                context=variant_context,
            )
            discovered_lane_counts[lane] += 1

            if include_lane_set and _normalize_method_name(lane) not in include_lane_set:
                continue

            sample_id = _coerce_string(
                variant.get("variant_id"),
                field_name="variant_id",
                context=variant_context,
            )
            if sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate proper-data sample_id={sample_id!r}.")
            seen_sample_ids.add(sample_id)

            real_frame_paths = real_by_transport[transport]
            fake_frame_paths = _resolve_frame_paths(variant, context=variant_context)
            if len(real_frame_paths) != len(fake_frame_paths):
                ragged_pairs.append(
                    {
                        "sample_id": sample_id,
                        "lane": lane,
                        "real_frame_count": len(real_frame_paths),
                        "fake_frame_count": len(fake_frame_paths),
                    }
                )

            samples.append(
                ProperDataPairedSample(
                    sample_id=sample_id,
                    base_capture_id=base_capture_id,
                    identity_id=identity_id,
                    capture_session_id=capture_session_id,
                    split_group_id=split_group_id,
                    source=lane,
                    method=_method_name(lane, generator_method),
                    transport=transport,
                    enhancement=enhancement,
                    generator_family=generator_family,
                    generator_method=generator_method,
                    quality_band=quality_band,
                    face_scale_band=face_scale_band,
                    real_frame_paths=real_frame_paths,
                    fake_frame_paths=fake_frame_paths,
                    inventory_path=inventory_path,
                    manifest_path=manifest_path,
                    wave_id=wave_id,
                )
            )

    discovery_summary = {
        "wave_id": wave_id,
        "capture_count": len(captures),
        "discovered_paired_sample_count": len(samples),
        "discovered_lane_counts": dict(sorted(discovered_lane_counts.items())),
        "ragged_pair_count": len(ragged_pairs),
        "ragged_pair_examples": ragged_pairs[:10],
    }
    return sorted(samples, key=lambda sample: (sample.source, sample.sample_id)), discovery_summary


def _apply_sample_caps(
    samples: Sequence[ProperDataPairedSample],
    *,
    max_samples_per_lane: Optional[int],
    max_samples_total: Optional[int],
) -> List[ProperDataPairedSample]:
    capped: List[ProperDataPairedSample] = list(samples)

    if max_samples_per_lane is not None:
        lane_limit = max(0, int(max_samples_per_lane))
        by_lane: Dict[str, List[ProperDataPairedSample]] = defaultdict(list)
        for sample in capped:
            by_lane[sample.source].append(sample)
        capped = []
        for lane in sorted(by_lane):
            capped.extend(sorted(by_lane[lane], key=lambda sample: sample.sample_id)[:lane_limit])

    if max_samples_total is not None:
        total_limit = max(0, int(max_samples_total))
        capped = sorted(capped, key=lambda sample: (sample.source, sample.sample_id))[:total_limit]

    return capped


def discover_proper_data_samples(
    *,
    inventory_uri: str,
    manifest_uri: Optional[str] = None,
    include_lanes: Optional[Sequence[str]] = None,
    max_samples_per_lane: Optional[int] = None,
    max_samples_total: Optional[int] = None,
    log: Optional[logging.Logger] = None,
) -> Tuple[List[ProperDataPairedSample], Dict[str, Any]]:
    """Discover proper-data paired samples from an explicit WT-F inventory."""

    logger = log or logging.getLogger(__name__)
    inventory = _load_structured_data(inventory_uri)
    manifest_summary: Dict[str, Any] = {}
    if manifest_uri:
        manifest = _load_structured_data(manifest_uri)
        manifest_summary = _validate_manifest_reference(
            manifest,
            inventory=inventory,
            inventory_path=str(inventory_uri),
            manifest_path=str(manifest_uri),
        )

    raw_samples, discovery_summary = _build_samples_from_inventory(
        inventory,
        inventory_path=str(inventory_uri),
        manifest_path=str(manifest_uri or ""),
        include_lanes=include_lanes,
    )
    capped_samples = _apply_sample_caps(
        raw_samples,
        max_samples_per_lane=max_samples_per_lane,
        max_samples_total=max_samples_total,
    )

    lane_counts = Counter(sample.source for sample in capped_samples)
    method_counts = Counter(sample.method for sample in capped_samples)
    transport_counts = Counter(sample.transport for sample in capped_samples)
    enhancement_counts = Counter(sample.enhancement for sample in capped_samples)
    quality_band_counts = Counter(sample.quality_band for sample in capped_samples)
    face_scale_band_counts = Counter(sample.face_scale_band for sample in capped_samples)

    summary = {
        "inventory_path": str(inventory_uri),
        "manifest_path": str(manifest_uri or ""),
        "wave_id": discovery_summary["wave_id"],
        "capture_count": discovery_summary["capture_count"],
        "discovered_paired_sample_count": discovery_summary["discovered_paired_sample_count"],
        "paired_sample_count": len(capped_samples),
        "discovered_lane_counts": discovery_summary["discovered_lane_counts"],
        "lane_counts": dict(sorted(lane_counts.items())),
        "method_counts": dict(sorted(method_counts.items())),
        "transport_counts": dict(sorted(transport_counts.items())),
        "enhancement_counts": dict(sorted(enhancement_counts.items())),
        "quality_band_counts": dict(sorted(quality_band_counts.items())),
        "face_scale_band_counts": dict(sorted(face_scale_band_counts.items())),
        "ragged_pair_count": discovery_summary["ragged_pair_count"],
        "ragged_pair_examples": discovery_summary["ragged_pair_examples"],
        "max_samples_per_lane": (
            int(max_samples_per_lane) if max_samples_per_lane is not None else None
        ),
        "max_samples_total": int(max_samples_total) if max_samples_total is not None else None,
        "manifest_summary": manifest_summary,
    }

    logger.info(
        "Proper-data discovery: wave=%s paired_samples=%d lane_counts=%s ragged_pairs=%d",
        summary["wave_id"],
        summary["paired_sample_count"],
        summary["lane_counts"],
        summary["ragged_pair_count"],
    )
    if manifest_uri and manifest_summary:
        logger.info(
            "Proper-data manifest summary: lane_counts=%s split_counts=%s",
            manifest_summary.get("lane_counts", {}),
            manifest_summary.get("split_counts", {}),
        )

    return capped_samples, summary


def _load_image_from_uri(
    frame_path: str,
    *,
    as_array: bool,
    client: Optional[Any] = None,
) -> Any:
    import cv2
    import numpy as np

    if str(frame_path).startswith("gs://"):
        from google.cloud import storage as gcs_storage

        gcs_client = client or gcs_storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))
        bucket_name, blob_path = _split_gs_uri(frame_path)
        blob_bytes = gcs_client.bucket(bucket_name).blob(blob_path).download_as_bytes()
    else:
        blob_bytes = Path(frame_path).read_bytes()

    image = cv2.imdecode(np.frombuffer(blob_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode proper-data frame: {frame_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if as_array:
        return image

    from PIL import Image

    return Image.fromarray(image)


def load_proper_data_frames(
    sample: ProperDataPairedSample,
    anchor_indices: Sequence[int],
    *,
    as_array: bool = True,
    client: Optional[Any] = None,
    executor: Optional[Any] = None,
    parallel_download_workers: int = DEFAULT_PARALLEL_FRAME_DOWNLOAD_WORKERS,
) -> Tuple[List[Optional[Any]], List[Optional[Any]]]:
    """Load aligned real/fake frames for a proper-data sample."""

    valid_indices = [
        frame_idx
        for frame_idx in anchor_indices
        if frame_idx < len(sample.real_frame_paths) and frame_idx < len(sample.fake_frame_paths)
    ]
    real_by_idx: Dict[int, Any] = {}
    fake_by_idx: Dict[int, Any] = {}

    def _load_pair(frame_idx: int) -> Tuple[int, Any, Any]:
        return (
            frame_idx,
            _load_image_from_uri(sample.real_frame_paths[frame_idx], as_array=as_array, client=client),
            _load_image_from_uri(sample.fake_frame_paths[frame_idx], as_array=as_array, client=client),
        )

    def _record_result(frame_idx: int, real_img: Any, fake_img: Any) -> None:
        real_by_idx[frame_idx] = real_img
        fake_by_idx[frame_idx] = fake_img

    def _load_sequential() -> None:
        for frame_idx in valid_indices:
            frame_id, real_img, fake_img = _load_pair(frame_idx)
            _record_result(frame_id, real_img, fake_img)

    if executor is None:
        max_workers = max(1, int(parallel_download_workers or 1))
        if max_workers <= 1 or len(valid_indices) <= 1:
            _load_sequential()
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(valid_indices))) as pool:
                future_to_idx = {
                    pool.submit(_load_pair, frame_idx): frame_idx
                    for frame_idx in valid_indices
                }
                for future in as_completed(future_to_idx):
                    frame_id, real_img, fake_img = future.result()
                    _record_result(frame_id, real_img, fake_img)
    else:
        future_to_idx = {
            executor.submit(_load_pair, frame_idx): frame_idx
            for frame_idx in valid_indices
        }
        for future in as_completed(future_to_idx):
            frame_id, real_img, fake_img = future.result()
            _record_result(frame_id, real_img, fake_img)

    return (
        [real_by_idx.get(frame_idx) for frame_idx in anchor_indices],
        [fake_by_idx.get(frame_idx) for frame_idx in anchor_indices],
    )
