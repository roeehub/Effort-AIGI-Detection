#!/usr/bin/env python3
"""
Build a provenance-clean future proper-data manifest from an explicit inventory.

Unlike the current Teams manifest builder, this file does not infer semantic
truth from bucket prefixes, session names, or merged convenience lanes. Future
proper data must arrive through an inventory that declares the exact condition
for every variant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
ALLOWED_LABELS = {"real", "fake"}
ALLOWED_TRANSPORTS = {"clean", "teams"}
ALLOWED_QUALITY_BANDS = {"high", "medium", "low"}
ALLOWED_FACE_SCALE_BANDS = {"big_face", "standard"}
ALLOWED_ENHANCEMENTS = {"none", "enhanced"}


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return text or "unknown"


def _stable_int_hash(value: str) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def _stable_fraction(key: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


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
    return Path(path).read_text()


def _write_text_to_path(path: str, text: str) -> None:
    if path.startswith("gs://"):
        from google.cloud import storage

        bucket_name, blob_path = _split_gs_uri(path)
        client = storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))
        client.bucket(bucket_name).blob(blob_path).upload_from_string(
            text,
            content_type="application/json",
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
        raise ValueError(f"Expected mapping at {path}, got {type(data).__name__}.")
    return data


class StorageReader:
    def list_images(self, root: str) -> List[str]:
        root = str(root).strip()
        if not root:
            return []
        if root.startswith("gs://"):
            from google.cloud import storage

            bucket_name, prefix = _split_gs_uri(root.rstrip("/"))
            client = storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))
            bucket = client.bucket(bucket_name)
            results = []
            for blob in client.list_blobs(bucket, prefix=prefix.rstrip("/") + "/"):
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


def _coerce_optional_string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _coerce_enum(
    value: Any,
    *,
    allowed: Sequence[str],
    field_name: str,
    context: str,
    default: Optional[str] = None,
) -> str:
    raw = value if value is not None else default
    text = _coerce_string(raw, field_name=field_name, context=context).lower()
    if text not in set(allowed):
        raise ValueError(
            f"Invalid value for '{field_name}' in {context}: {text!r}. "
            f"Expected one of {sorted(set(allowed))}."
        )
    return text


def _coerce_source_logs(value: Any) -> Dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Top-level 'source_logs' must be a mapping.")
    result: Dict[str, str] = {}
    for key, raw in value.items():
        text = _coerce_optional_string(raw)
        if text:
            result[str(key).strip()] = text
    return result


def _resolve_frame_paths(
    variant: Dict[str, Any],
    *,
    context: str,
    storage: StorageReader,
) -> List[str]:
    has_root = "frame_root" in variant and _coerce_optional_string(variant.get("frame_root"))
    has_paths = "frame_paths" in variant and variant.get("frame_paths") is not None
    if has_root and has_paths:
        raise ValueError(
            f"{context} must set exactly one of 'frame_root' or 'frame_paths', not both."
        )
    if not has_root and not has_paths:
        raise ValueError(
            f"{context} must set one of 'frame_root' or 'frame_paths'."
        )

    if has_root:
        frame_root = _coerce_string(
            variant.get("frame_root"),
            field_name="frame_root",
            context=context,
        )
        frame_paths = storage.list_images(frame_root)
    else:
        raw_paths = variant.get("frame_paths")
        if not isinstance(raw_paths, list):
            raise ValueError(f"{context}.frame_paths must be a list.")
        frame_paths = []
        for index, raw in enumerate(raw_paths):
            text = _coerce_optional_string(raw)
            if not text:
                raise ValueError(
                    f"{context}.frame_paths[{index}] must be a non-empty string."
                )
            frame_paths.append(text)

    if not frame_paths:
        raise ValueError(f"{context} resolved zero frames.")
    return sorted(frame_paths)


def _lane_for_variant(
    *,
    label: str,
    transport: str,
    enhancement: str,
    generator_family: str,
    generator_method: str,
    context: str,
) -> str:
    if label == "real":
        if enhancement != "none":
            raise ValueError(
                f"{context} is real but enhancement={enhancement!r}; real variants must use 'none'."
            )
        if generator_family not in {"", "none"}:
            raise ValueError(
                f"{context} is real but generator_family={generator_family!r}; real variants cannot declare a fake family."
            )
        if generator_method:
            raise ValueError(
                f"{context} is real but generator_method={generator_method!r}; real variants cannot declare a fake method."
            )
        return "proper_real_teams" if transport == "teams" else "proper_real_clean"

    if generator_family != "visomaster":
        raise ValueError(
            f"{context} must declare generator_family='visomaster' for fake proper data."
        )
    if not generator_method:
        raise ValueError(f"{context} is fake but missing generator_method.")

    if enhancement == "none":
        return "proper_visomaster_teams" if transport == "teams" else "proper_visomaster_clean"
    if enhancement == "enhanced":
        return (
            "proper_visomaster_enhanced_teams"
            if transport == "teams"
            else "proper_visomaster_enhanced_clean"
        )
    raise ValueError(f"Unsupported enhancement value {enhancement!r} in {context}.")


def _method_name(*, label: str, lane: str, generator_method: str) -> str:
    if label == "real":
        return lane
    return f"{lane}__{_slugify(generator_method)}"


def _slices_for_variant(
    *,
    label: str,
    lane: str,
    transport: str,
    quality_band: str,
    face_scale_band: str,
    generator_method: str,
) -> List[str]:
    slices = [
        lane,
        f"proper_quality_{quality_band}",
        f"proper_face_scale_{face_scale_band}",
    ]
    if label == "real":
        slices.append("proper_real_all")
    else:
        slices.append("proper_fake_all")
        slices.append("proper_fake_teams_all" if transport == "teams" else "proper_fake_clean_all")
        slices.append(f"{lane}__{_slugify(generator_method)}")
    return sorted(set(slices))


def _recommended_contract() -> Dict[str, Any]:
    return {
        "dev_real_suite": "proper_real_teams_dev",
        "dev_real_stress_suites": [],
        "dev_fake_suites": [
            "proper_fake_teams_all_dev",
            "proper_visomaster_teams_dev",
            "proper_visomaster_enhanced_teams_dev",
        ],
        "lockbox_real_suite": "proper_real_teams_lockbox",
        "lockbox_fake_suite": "proper_fake_teams_all_lockbox",
    }


def build_manifest(
    inventory: Dict[str, Any],
    *,
    source_inventory_path: Optional[str] = None,
    split_seed_override: Optional[int] = None,
    lockbox_ratio_override: Optional[float] = None,
) -> Dict[str, Any]:
    version = int(inventory.get("inventory_version", 0) or 0)
    if version != 1:
        raise ValueError(
            f"Unsupported inventory_version={version!r}; expected 1."
        )

    wave_id = _coerce_string(inventory.get("wave_id"), field_name="wave_id", context="inventory")
    split_seed = (
        int(split_seed_override)
        if split_seed_override is not None
        else int(inventory.get("split_seed", 737))
    )
    lockbox_ratio = (
        float(lockbox_ratio_override)
        if lockbox_ratio_override is not None
        else float(inventory.get("lockbox_ratio", 0.20))
    )
    if not 0.0 < lockbox_ratio < 1.0:
        raise ValueError(f"lockbox_ratio must be in (0, 1), got {lockbox_ratio}")

    notes = inventory.get("notes") or []
    if notes and not isinstance(notes, list):
        raise ValueError("Top-level 'notes' must be a list when provided.")

    captures = inventory.get("captures")
    if not isinstance(captures, list) or not captures:
        raise ValueError("Inventory must contain a non-empty 'captures' list.")

    source_logs = _coerce_source_logs(inventory.get("source_logs"))

    storage = StorageReader()
    rows: List[Dict[str, Any]] = []
    seen_variant_ids = set()
    seen_capture_ids = set()

    for capture_index, capture in enumerate(captures):
        if not isinstance(capture, dict):
            raise ValueError(f"captures[{capture_index}] must be a mapping.")
        capture_context = f"captures[{capture_index}]"

        base_capture_id = _coerce_string(
            capture.get("base_capture_id"),
            field_name="base_capture_id",
            context=capture_context,
        )
        if base_capture_id in seen_capture_ids:
            raise ValueError(f"Duplicate base_capture_id={base_capture_id!r}.")
        seen_capture_ids.add(base_capture_id)

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
            allowed=ALLOWED_QUALITY_BANDS,
            field_name="quality_band",
            context=capture_context,
        )
        face_scale_band = _coerce_enum(
            capture.get("face_scale_band"),
            allowed=ALLOWED_FACE_SCALE_BANDS,
            field_name="face_scale_band",
            context=capture_context,
        )
        variants = capture.get("variants")
        if not isinstance(variants, list) or not variants:
            raise ValueError(f"{capture_context} must contain a non-empty 'variants' list.")

        split = (
            "lockbox"
            if _stable_fraction(split_group_id, split_seed) < lockbox_ratio
            else "dev"
        )

        seen_variant_keys = set()
        for variant_index, variant in enumerate(variants):
            if not isinstance(variant, dict):
                raise ValueError(
                    f"{capture_context}.variants[{variant_index}] must be a mapping."
                )
            variant_context = f"{capture_context}.variants[{variant_index}]"

            variant_id = _coerce_string(
                variant.get("variant_id"),
                field_name="variant_id",
                context=variant_context,
            )
            if variant_id in seen_variant_ids:
                raise ValueError(f"Duplicate variant_id={variant_id!r}.")
            seen_variant_ids.add(variant_id)

            label = _coerce_enum(
                variant.get("label"),
                allowed=ALLOWED_LABELS,
                field_name="label",
                context=variant_context,
            )
            transport = _coerce_enum(
                variant.get("transport"),
                allowed=ALLOWED_TRANSPORTS,
                field_name="transport",
                context=variant_context,
            )
            enhancement = _coerce_enum(
                variant.get("enhancement"),
                allowed=ALLOWED_ENHANCEMENTS,
                field_name="enhancement",
                context=variant_context,
                default="none",
            )
            generator_family = _coerce_optional_string(variant.get("generator_family")).lower()
            generator_method = _coerce_optional_string(variant.get("generator_method"))
            playback_path = _coerce_string(
                variant.get("playback_path"),
                field_name="playback_path",
                context=variant_context,
            )
            frame_paths = _resolve_frame_paths(
                variant,
                context=variant_context,
                storage=storage,
            )

            lane = _lane_for_variant(
                label=label,
                transport=transport,
                enhancement=enhancement,
                generator_family=generator_family,
                generator_method=generator_method,
                context=variant_context,
            )
            variant_key = (lane, _slugify(generator_method or "real"), transport)
            if variant_key in seen_variant_keys:
                raise ValueError(
                    f"{capture_context} declares duplicate lane/method transport combo {variant_key!r}."
                )
            seen_variant_keys.add(variant_key)

            row = {
                "label": label,
                "lane": lane,
                "method": _method_name(
                    label=label,
                    lane=lane,
                    generator_method=generator_method,
                ),
                "video_id": variant_id,
                "frame_paths": frame_paths,
                "identity": _stable_int_hash(split_group_id),
                "identity_key": split_group_id,
                "split": split,
                "slices": _slices_for_variant(
                    label=label,
                    lane=lane,
                    transport=transport,
                    quality_band=quality_band,
                    face_scale_band=face_scale_band,
                    generator_method=generator_method,
                ),
                "base_capture_id": base_capture_id,
                "identity_id": identity_id,
                "capture_session_id": capture_session_id,
                "quality_band": quality_band,
                "face_scale_band": face_scale_band,
                "transport": transport,
                "generator_family": generator_family or "none",
                "generator_method": generator_method or None,
                "enhancement": enhancement,
                "playback_path": playback_path,
                "source_kind": "future_proper_inventory",
            }

            notes_text = _coerce_optional_string(variant.get("notes"))
            if notes_text:
                row["notes"] = notes_text

            rows.append(row)

    rows.sort(key=lambda row: (str(row["base_capture_id"]), str(row["video_id"])))

    label_counts = Counter(str(row["label"]) for row in rows)
    split_counts = Counter(str(row["split"]) for row in rows)
    lane_counts = Counter(str(row["lane"]) for row in rows)
    method_counts = Counter(str(row["method"]) for row in rows)
    slice_counts = Counter(slice_name for row in rows for slice_name in row["slices"])
    quality_counts = Counter(str(row["quality_band"]) for row in rows)
    face_scale_counts = Counter(str(row["face_scale_band"]) for row in rows)

    return {
        "manifest_version": 1,
        "schema_name": "future_proper_target_domain_v1",
        "source_inventory": source_inventory_path,
        "wave_id": wave_id,
        "split_seed": split_seed,
        "lockbox_ratio": lockbox_ratio,
        "source_logs": source_logs,
        "notes": list(notes),
        "recommended_contract": _recommended_contract(),
        "summary": {
            "captures_total": len(captures),
            "videos_total": len(rows),
            "label_counts": dict(sorted(label_counts.items())),
            "split_counts": dict(sorted(split_counts.items())),
            "lane_counts": dict(sorted(lane_counts.items())),
            "method_counts": dict(sorted(method_counts.items())),
            "slice_counts": dict(sorted(slice_counts.items())),
            "quality_band_counts": dict(sorted(quality_counts.items())),
            "face_scale_band_counts": dict(sorted(face_scale_counts.items())),
        },
        "videos": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        required=True,
        help="YAML/JSON inventory file declaring future proper-data captures and variants.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Local or gs:// JSON output path for the built manifest.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Optional override for the inventory split_seed.",
    )
    parser.add_argument(
        "--lockbox-ratio",
        type=float,
        default=None,
        help="Optional override for the inventory lockbox_ratio.",
    )
    args = parser.parse_args()

    inventory = _load_structured_data(args.inventory)
    manifest = build_manifest(
        inventory,
        source_inventory_path=str(args.inventory).strip(),
        split_seed_override=args.split_seed,
        lockbox_ratio_override=args.lockbox_ratio,
    )
    _write_text_to_path(
        args.output,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    summary = manifest["summary"]
    print(f"Wrote manifest: {args.output}")
    print(f"Captures: {summary['captures_total']}")
    print(f"Videos: {summary['videos_total']}")
    print(f"Lane counts: {summary['lane_counts']}")


if __name__ == "__main__":
    main()
