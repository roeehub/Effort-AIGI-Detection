#!/usr/bin/env python3
"""
Build provisional WT-F proper-data artifacts from the incoming VisoMaster buckets.

This converter does not route the new buckets through legacy training loaders.
Instead it materializes:

1. an explicit future proper-data inventory snapshot
2. a future proper-data target-domain manifest
3. a concrete target-domain suite YAML with the manifest path filled in
4. a build report that records what was kept and what was filtered

The current wave is intentionally provisional:
- clean-versus-Teams pairing is driven by exact `sample_id` intersection
- Teams rows are filtered to exact fixed-frame selections before inclusion
- `quality_band` / `face_scale_band` are currently assigned by explicit
  dataset-level defaults because those fields are not present in the bucket
  manifests yet
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml


ARENA_ROOT = Path(__file__).resolve().parent


DEFAULT_WAVE_ID = "proper_visomaster_wave_2026_04_19_provisional"
DEFAULT_DATASET_BANDS: Dict[str, Dict[str, str]] = {
    "hdtf_20260416": {
        "quality_band": "high",
        "face_scale_band": "big_face",
    },
    "quickclips_20260417_20260418_combined": {
        "quality_band": "medium",
        "face_scale_band": "standard",
    },
}
DEFAULT_SOURCE_LOGS: Dict[str, str] = {
    "clean_census_report": "DeepfakeBench/training/arena/reports/visomaster_proper_clean_bucket_census_2026-04-19.json",
    "teams_census_report": "DeepfakeBench/training/arena/reports/visomaster_proper_teams_bucket_census_2026-04-19.json",
    "hdtf_join_report": "DeepfakeBench/training/arena/reports/hdtf_visomaster_clean_vs_teams_join_2026-04-19.json",
    "quickclips_join_report": "DeepfakeBench/training/arena/reports/quickclips_visomaster_clean_vs_teams_join_2026-04-19.json",
    "handoff_doc": "DeepfakeBench/training/docs/relaunch_handoffs/NEW_DATA_LOADER_AND_EXPERIMENT_HANDOFF_2026-04-19.md",
}
DEFAULT_NOTES: Tuple[str, ...] = (
    "Provisional snapshot from the live 2026-04-19 Teams propagation state; regenerate after the remaining Teams data lands.",
    "Clean-versus-Teams alignment is driven by exact sample_id overlap because base-capture keys are ambiguous across multiple fake variants.",
    "Only exact 16/16 clean and Teams rows are included in this snapshot.",
    "quality_band and face_scale_band are provisional dataset-level defaults until upstream manifests carry explicit per-capture band metadata.",
)


@dataclass(frozen=True)
class SourcePair:
    name: str
    clean_source: str
    teams_source: str


DEFAULT_SOURCE_PAIRS: Tuple[SourcePair, ...] = (
    SourcePair(
        name="hdtf",
        clean_source="gs://hdtf_visomaster_cropped_frames",
        teams_source="gs://hdtf_visomaster_cropped_frames_teams",
    ),
    SourcePair(
        name="quickclips",
        clean_source="gs://quickclips_visomaster_cropped_frames",
        teams_source="gs://quickclips_visomaster_cropped_frames_teams",
    ),
)


_MODULE_CACHE: Dict[str, Any] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_local_module(module_name: str, filename: str):
    cache_key = f"{module_name}:{filename}"
    if cache_key in _MODULE_CACHE:
        return _MODULE_CACHE[cache_key]

    module_path = ARENA_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _MODULE_CACHE[cache_key] = module
    return module


def _inspect_module():
    return _load_local_module(
        "visomaster_proper_bucket_inspector_runtime",
        "inspect_visomaster_proper_buckets.py",
    )


def _future_manifest_module():
    return _load_local_module(
        "future_proper_target_domain_manifest_runtime",
        "build_future_proper_target_domain_manifest.py",
    )


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return text or "unknown"


def _parse_wave_id_tokens(wave_id: str) -> Optional[Dict[str, str]]:
    match = re.match(
        r"^(?P<prefix>.+?)_wave_(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})(?:_(?P<suffix>.+))?$",
        str(wave_id or "").strip(),
    )
    if match is None:
        return None
    tokens = match.groupdict()
    return {
        "prefix": str(tokens.get("prefix") or "").strip(),
        "date_slug": (
            f"{tokens.get('year')}-{tokens.get('month')}-{tokens.get('day')}"
        ),
        "suffix": str(tokens.get("suffix") or "").strip(),
    }


def _normalize_none_token(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "None"
    if text.lower() in {"none", "null", "nan"}:
        return "None"
    return text


def _is_enhanced(enhancer: Any) -> bool:
    return _normalize_none_token(enhancer) != "None"


def _fake_enhancement(record: Any) -> str:
    return "enhanced" if _is_enhanced(record.enhancer) else "none"


def _generator_method(record: Any) -> str:
    swap_model = str(record.swap_model or "").strip() or "unknown_visomaster"
    enhancer = _normalize_none_token(record.enhancer)
    if enhancer == "None":
        return swap_model
    return f"{swap_model}__{enhancer}"


def _source_path(root: str, *parts: str) -> str:
    base = str(root).rstrip("/")
    pieces = [str(part).strip("/") for part in parts if str(part).strip("/")]
    if base.startswith("gs://"):
        return "/".join([base] + pieces)
    return str((Path(base).joinpath(*pieces)).resolve())


def _frame_paths_for_record(record: Any, side: str) -> List[str]:
    if side not in {"real", "fake"}:
        raise ValueError(f"Unsupported side {side!r}")

    frame_names = (
        tuple(record.real_frame_names)
        if side == "real"
        else tuple(record.fake_frame_names)
    )
    if not frame_names:
        raise ValueError(
            f"Missing explicit frame_files for sample_id={record.sample_id!r} side={side!r}. "
            "The provisional proper-data builder refuses to synthesize fallback frame paths."
        )

    frame_root = _source_path(record.source_root, "samples", record.sample_id, "frames", side)
    return [_source_path(frame_root, frame_name) for frame_name in frame_names]


def _assert_explicit_frame_files(record: Any) -> None:
    if not tuple(record.real_frame_names):
        raise ValueError(
            f"Missing explicit frame_files for sample_id={record.sample_id!r} side='real'. "
            "The provisional proper-data builder refuses to synthesize fallback frame paths."
        )
    if not tuple(record.fake_frame_names):
        raise ValueError(
            f"Missing explicit frame_files for sample_id={record.sample_id!r} side='fake'. "
            "The provisional proper-data builder refuses to synthesize fallback frame paths."
        )


def _record_is_fixed(record: Any, target_frame_count: int) -> bool:
    expected = int(record.expected_total_per_stream or 0)
    if expected not in {0, target_frame_count}:
        return False
    if int(record.cropped_real_count or 0) != target_frame_count:
        return False
    if int(record.cropped_fake_count or 0) != target_frame_count:
        return False
    if len(tuple(record.real_frame_names)) != target_frame_count:
        return False
    if len(tuple(record.fake_frame_names)) != target_frame_count:
        return False
    return True


def _clean_record_is_fixed(record: Any, target_frame_count: int) -> bool:
    return _record_is_fixed(record, target_frame_count)


def _teams_record_is_fixed(record: Any, target_frame_count: int) -> bool:
    return _record_is_fixed(record, target_frame_count)


def _derive_identity_and_session(record: Any) -> Tuple[str, str]:
    clip_key = (
        str(record.clip_stem or "")
        or str(record.target_video_name or "")
        or str(record.real_id or "")
        or str(record.sample_id or "")
    ).strip()
    prefix = clip_key.split("__rank", 1)[0] if "__rank" in clip_key else clip_key

    if "__" in prefix:
        identity_id, capture_session_id = prefix.split("__", 1)
        identity_id = identity_id.strip() or prefix
        capture_session_id = capture_session_id.strip() or prefix
        return identity_id, capture_session_id

    match = re.match(r"(.+)_\d{3}$", prefix)
    if match:
        return match.group(1), prefix

    if prefix:
        return prefix, prefix
    sample_id = str(record.sample_id or "").strip() or "unknown_capture"
    return sample_id, sample_id


def _resolve_dataset_bands(
    dataset_key: str,
    dataset_band_defaults: Mapping[str, Mapping[str, str]],
) -> Dict[str, str]:
    resolved = dataset_band_defaults.get(str(dataset_key).strip())
    if resolved is None:
        raise ValueError(
            f"No dataset band defaults configured for dataset_key={dataset_key!r}."
        )
    quality_band = str(resolved.get("quality_band") or "").strip()
    face_scale_band = str(resolved.get("face_scale_band") or "").strip()
    if not quality_band or not face_scale_band:
        raise ValueError(
            f"Incomplete band defaults for dataset_key={dataset_key!r}: {resolved!r}"
        )
    return {
        "quality_band": quality_band,
        "face_scale_band": face_scale_band,
    }


def _raise_on_load_errors(payload: Mapping[str, Any]) -> None:
    errors = list(payload.get("load_errors") or [])
    if not errors:
        return
    first = errors[0]
    raise RuntimeError(
        f"Failed to load {len(errors)} manifests from {payload.get('source_root')}: "
        f"{first.get('manifest_ref')}: {first.get('error')}"
    )


def _index_by_sample_id(records: Sequence[Any], context: str) -> Dict[str, Any]:
    indexed: Dict[str, Any] = {}
    for record in records:
        sample_id = str(record.sample_id or "").strip()
        if not sample_id:
            raise ValueError(f"Encountered empty sample_id in {context}.")
        if sample_id in indexed:
            raise ValueError(f"Duplicate sample_id={sample_id!r} in {context}.")
        indexed[sample_id] = record
    return indexed


def _assert_record_alignment(clean_record: Any, teams_record: Any) -> None:
    comparisons = [
        ("dataset_key", clean_record.dataset_key, teams_record.dataset_key),
        ("real_id", clean_record.real_id, teams_record.real_id),
        ("swap_model", clean_record.swap_model, teams_record.swap_model),
        (
            "enhancer",
            _normalize_none_token(clean_record.enhancer),
            _normalize_none_token(teams_record.enhancer),
        ),
    ]
    mismatches = [
        f"{field}: clean={left!r} teams={right!r}"
        for field, left, right in comparisons
        if str(left or "").strip() != str(right or "").strip()
    ]
    if mismatches:
        raise ValueError("; ".join(mismatches))


def _build_capture(
    clean_record: Any,
    teams_record: Any,
    *,
    dataset_band_defaults: Mapping[str, Mapping[str, str]],
) -> Dict[str, Any]:
    _assert_record_alignment(clean_record, teams_record)

    bands = _resolve_dataset_bands(clean_record.dataset_key, dataset_band_defaults)
    identity_id, capture_session_id = _derive_identity_and_session(clean_record)
    split_group_id = f"{identity_id}__{capture_session_id}"
    base_capture_id = str(clean_record.sample_id).strip()
    generator_method = _generator_method(clean_record)
    enhancement = _fake_enhancement(clean_record)
    fake_slug = _slugify(generator_method)

    variants = [
        {
            "variant_id": f"{base_capture_id}__real_clean",
            "label": "real",
            "transport": "clean",
            "enhancement": "none",
            "playback_path": "direct_capture",
            "frame_paths": _frame_paths_for_record(clean_record, "real"),
        },
        {
            "variant_id": f"{base_capture_id}__real_teams",
            "label": "real",
            "transport": "teams",
            "enhancement": "none",
            "playback_path": "obs_virtual_cam_to_teams",
            "frame_paths": _frame_paths_for_record(teams_record, "real"),
        },
        {
            "variant_id": f"{base_capture_id}__{fake_slug}__clean",
            "label": "fake",
            "transport": "clean",
            "generator_family": "visomaster",
            "generator_method": generator_method,
            "enhancement": enhancement,
            "playback_path": "direct_capture",
            "frame_paths": _frame_paths_for_record(clean_record, "fake"),
        },
        {
            "variant_id": f"{base_capture_id}__{fake_slug}__teams",
            "label": "fake",
            "transport": "teams",
            "generator_family": "visomaster",
            "generator_method": generator_method,
            "enhancement": enhancement,
            "playback_path": "obs_virtual_cam_to_teams",
            "frame_paths": _frame_paths_for_record(teams_record, "fake"),
        },
    ]

    return {
        "base_capture_id": base_capture_id,
        "identity_id": identity_id,
        "capture_session_id": capture_session_id,
        "split_group_id": split_group_id,
        "quality_band": bands["quality_band"],
        "face_scale_band": bands["face_scale_band"],
        "variants": variants,
    }


def build_inventory_snapshot(
    source_pairs: Sequence[SourcePair],
    *,
    wave_id: str = DEFAULT_WAVE_ID,
    split_seed: int = 737,
    lockbox_ratio: float = 0.20,
    teams_target_frame_count: int = 16,
    allow_ragged_clean: bool = False,
    dataset_band_defaults: Optional[Mapping[str, Mapping[str, str]]] = None,
    source_logs: Optional[Mapping[str, str]] = None,
    notes: Optional[Iterable[str]] = None,
    project: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    inspect_module = _inspect_module()
    dataset_band_defaults = dict(dataset_band_defaults or DEFAULT_DATASET_BANDS)

    captures: List[Dict[str, Any]] = []
    pair_reports: List[Dict[str, Any]] = []

    for pair in source_pairs:
        clean_payload = inspect_module.load_manifest_records(pair.clean_source, project=project)
        teams_payload = inspect_module.load_manifest_records(pair.teams_source, project=project)
        _raise_on_load_errors(clean_payload)
        _raise_on_load_errors(teams_payload)

        clean_by_sample = _index_by_sample_id(clean_payload["records"], f"{pair.name} clean source")
        teams_by_sample = _index_by_sample_id(teams_payload["records"], f"{pair.name} teams source")

        overlap_sample_ids = sorted(set(clean_by_sample).intersection(teams_by_sample))
        clean_only_sample_ids = sorted(set(clean_by_sample).difference(teams_by_sample))
        teams_only_sample_ids = sorted(set(teams_by_sample).difference(clean_by_sample))

        kept_for_pair = 0
        kept_enhancers: Counter[str] = Counter()
        kept_dataset_keys: Counter[str] = Counter()
        skipped_alignment: List[Dict[str, Any]] = []
        skipped_ragged_clean: List[Dict[str, Any]] = []
        skipped_ragged_teams: List[Dict[str, Any]] = []

        for sample_id in overlap_sample_ids:
            clean_record = clean_by_sample[sample_id]
            teams_record = teams_by_sample[sample_id]

            try:
                _assert_record_alignment(clean_record, teams_record)
            except ValueError as exc:
                skipped_alignment.append(
                    {
                        "sample_id": sample_id,
                        "reason": str(exc),
                    }
                )
                continue

            _assert_explicit_frame_files(clean_record)
            _assert_explicit_frame_files(teams_record)

            if not allow_ragged_clean and not _clean_record_is_fixed(
                clean_record, teams_target_frame_count
            ):
                skipped_ragged_clean.append(
                    {
                        "sample_id": sample_id,
                        "dataset_key": clean_record.dataset_key,
                        "expected_total_per_stream": int(clean_record.expected_total_per_stream or 0),
                        "selected_real": int(clean_record.cropped_real_count or 0),
                        "selected_fake": int(clean_record.cropped_fake_count or 0),
                    }
                )
                continue

            if not _teams_record_is_fixed(teams_record, teams_target_frame_count):
                skipped_ragged_teams.append(
                    {
                        "sample_id": sample_id,
                        "dataset_key": clean_record.dataset_key,
                        "expected_total_per_stream": int(teams_record.expected_total_per_stream or 0),
                        "selected_real": int(teams_record.cropped_real_count or 0),
                        "selected_fake": int(teams_record.cropped_fake_count or 0),
                    }
                )
                continue

            capture = _build_capture(
                clean_record,
                teams_record,
                dataset_band_defaults=dataset_band_defaults,
            )
            captures.append(capture)
            kept_for_pair += 1
            kept_enhancers[_normalize_none_token(clean_record.enhancer)] += 1
            kept_dataset_keys[str(clean_record.dataset_key)] += 1

        pair_reports.append(
            {
                "name": pair.name,
                "clean_source": clean_payload["source_root"],
                "teams_source": teams_payload["source_root"],
                "clean_manifest_count": int(clean_payload["manifest_count"]),
                "teams_manifest_count": int(teams_payload["manifest_count"]),
                "clean_sample_count": len(clean_by_sample),
                "teams_sample_count": len(teams_by_sample),
                "overlap_sample_count": len(overlap_sample_ids),
                "kept_sample_count": kept_for_pair,
                "clean_only_sample_count": len(clean_only_sample_ids),
                "teams_only_sample_count": len(teams_only_sample_ids),
                "skipped_alignment_count": len(skipped_alignment),
                "skipped_alignment_examples": skipped_alignment[:10],
                "skipped_ragged_clean_count": len(skipped_ragged_clean),
                "skipped_ragged_clean_examples": skipped_ragged_clean[:10],
                "skipped_ragged_teams_count": len(skipped_ragged_teams),
                "skipped_ragged_teams_examples": skipped_ragged_teams[:10],
                "kept_dataset_keys": {key: int(kept_dataset_keys[key]) for key in sorted(kept_dataset_keys)},
                "kept_enhancers": {key: int(kept_enhancers[key]) for key in sorted(kept_enhancers)},
                "clean_only_sample_examples": clean_only_sample_ids[:10],
                "teams_only_sample_examples": teams_only_sample_ids[:10],
            }
        )

    inventory = {
        "inventory_version": 1,
        "wave_id": wave_id,
        "split_seed": int(split_seed),
        "lockbox_ratio": float(lockbox_ratio),
        "source_logs": dict(source_logs or DEFAULT_SOURCE_LOGS),
        "notes": list(DEFAULT_NOTES) + [str(note) for note in (notes or []) if str(note).strip()],
        "captures": captures,
    }

    report = {
        "schema_name": "visomaster_proper_data_artifact_build_v1",
        "generated_at": _now_iso(),
        "wave_id": wave_id,
        "teams_target_frame_count": int(teams_target_frame_count),
        "allow_ragged_clean": bool(allow_ragged_clean),
        "dataset_band_defaults": {
            key: {
                "quality_band": str(value.get("quality_band")),
                "face_scale_band": str(value.get("face_scale_band")),
            }
            for key, value in sorted(dataset_band_defaults.items())
        },
        "source_pairs": pair_reports,
        "inventory_capture_count": len(captures),
    }
    return inventory, report


def render_suite_template(
    manifest_path_for_suite: str,
    *,
    template_path: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    template = Path(template_path).resolve() if template_path else ARENA_ROOT / "target_domain_suites.proper_data_future.template.yaml"
    text = template.read_text()
    rendered = text.replace("<manifest-path>", str(manifest_path_for_suite))
    parsed = yaml.safe_load(rendered)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("suites"), list):
        raise ValueError(f"Rendered suite template from {template} did not produce a valid suites document.")
    return rendered, parsed


def summarize_suite_occupancy(
    manifest: Mapping[str, Any],
    suites_doc: Mapping[str, Any],
) -> Dict[str, Any]:
    videos = list(manifest.get("videos") or [])
    suite_rows: List[Dict[str, Any]] = []
    empty_suite_names: List[str] = []

    for suite in suites_doc.get("suites", []) or []:
        name = str(suite.get("name") or "").strip()
        split = str(
            suite.get("external_real_manifest_split")
            or suite.get("external_fake_manifest_split")
            or ""
        ).strip()
        slice_name = str(
            suite.get("external_real_manifest_slices")
            or suite.get("external_fake_manifest_slices")
            or ""
        ).strip()

        count = sum(
            1
            for row in videos
            if str(row.get("split") or "").strip() == split
            and slice_name in set(row.get("slices") or [])
        )
        suite_rows.append(
            {
                "name": name,
                "split": split,
                "slice": slice_name,
                "video_count": int(count),
            }
        )
        if count == 0 and name:
            empty_suite_names.append(name)

    return {
        "suite_counts": suite_rows,
        "empty_suite_names": empty_suite_names,
    }


def build_artifacts(
    source_pairs: Sequence[SourcePair],
    *,
    inventory_output_path: str,
    manifest_output_path: str,
    suite_output_path: str,
    report_output_path: str,
    manifest_path_for_suite: Optional[str] = None,
    wave_id: str = DEFAULT_WAVE_ID,
    split_seed: int = 737,
    lockbox_ratio: float = 0.20,
    teams_target_frame_count: int = 16,
    allow_ragged_clean: bool = False,
    dataset_band_defaults: Optional[Mapping[str, Mapping[str, str]]] = None,
    source_logs: Optional[Mapping[str, str]] = None,
    notes: Optional[Iterable[str]] = None,
    project: Optional[str] = None,
    suite_template_path: Optional[str] = None,
) -> Dict[str, Any]:
    future_manifest_module = _future_manifest_module()
    write_text = future_manifest_module._write_text_to_path

    inventory, report = build_inventory_snapshot(
        source_pairs,
        wave_id=wave_id,
        split_seed=split_seed,
        lockbox_ratio=lockbox_ratio,
        teams_target_frame_count=teams_target_frame_count,
        allow_ragged_clean=allow_ragged_clean,
        dataset_band_defaults=dataset_band_defaults,
        source_logs=source_logs,
        notes=notes,
        project=project,
    )

    inventory_yaml = yaml.safe_dump(
        inventory,
        sort_keys=False,
        default_flow_style=False,
        width=120,
    )
    write_text(str(inventory_output_path), inventory_yaml)

    manifest = future_manifest_module.build_manifest(
        inventory,
        source_inventory_path=str(inventory_output_path),
        split_seed_override=split_seed,
        lockbox_ratio_override=lockbox_ratio,
    )
    write_text(str(manifest_output_path), json.dumps(manifest, indent=2))

    suite_manifest_path = str(manifest_path_for_suite or manifest_output_path)
    suite_text, suites_doc = render_suite_template(
        suite_manifest_path,
        template_path=suite_template_path,
    )
    write_text(str(suite_output_path), suite_text)

    suite_summary = summarize_suite_occupancy(manifest, suites_doc)
    report.update(
        {
            "artifact_paths": {
                "inventory": str(inventory_output_path),
                "manifest": str(manifest_output_path),
                "suite_manifest": str(suite_output_path),
                "build_report": str(report_output_path),
            },
            "manifest_summary": manifest.get("summary", {}),
            "recommended_contract": manifest.get("recommended_contract", {}),
            "suite_summary": suite_summary,
        }
    )
    write_text(str(report_output_path), json.dumps(report, indent=2))

    if suite_summary["empty_suite_names"]:
        raise ValueError(
            "Generated suite manifest contains empty suites: "
            + ", ".join(sorted(suite_summary["empty_suite_names"]))
        )

    return {
        "inventory": inventory,
        "manifest": manifest,
        "suite_doc": suites_doc,
        "report": report,
    }


def _parse_source_pair(value: str) -> SourcePair:
    text = str(value or "").strip()
    if not text:
        raise argparse.ArgumentTypeError("Source pair must be non-empty.")

    if "=" in text:
        name, remainder = text.split("=", 1)
    else:
        name = f"pair_{_slugify(text)}"
        remainder = text

    if "::" not in remainder:
        raise argparse.ArgumentTypeError(
            "Source pair must use NAME=CLEAN_SOURCE::TEAMS_SOURCE."
        )

    clean_source, teams_source = remainder.split("::", 1)
    clean_source = clean_source.strip()
    teams_source = teams_source.strip()
    if not clean_source or not teams_source:
        raise argparse.ArgumentTypeError(
            "Both CLEAN_SOURCE and TEAMS_SOURCE must be non-empty."
        )
    return SourcePair(name=name.strip() or f"pair_{_slugify(remainder)}", clean_source=clean_source, teams_source=teams_source)


def _parse_key_value(items: Optional[Sequence[str]]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in items or []:
        text = str(item or "").strip()
        if "=" not in text:
            raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got {item!r}")
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got {item!r}")
        result[key] = value
    return result


def _default_output_paths(wave_id: str) -> Dict[str, str]:
    parsed_wave_id = _parse_wave_id_tokens(wave_id)
    if parsed_wave_id is None:
        manifest_name = f"{wave_id}_target_domain_manifest.json"
        suite_name = f"target_domain_suites.{wave_id}.yaml"
    else:
        prefix = parsed_wave_id["prefix"] or "proper_data"
        date_slug = parsed_wave_id["date_slug"] or "snapshot"
        suffix = parsed_wave_id["suffix"]
        manifest_name = f"{prefix}_target_domain_manifest_{date_slug}"
        suite_tail = date_slug
        if suffix:
            manifest_name = f"{manifest_name}_{suffix}"
            suite_tail = f"{suffix}_{date_slug}"
        manifest_name = f"{manifest_name}.json"
        suite_name = f"target_domain_suites.proper_data_future.{suite_tail}.yaml"
    return {
        "inventory": str(ARENA_ROOT / "inventories" / f"{wave_id}.yaml"),
        "manifest": str(ARENA_ROOT / "manifests" / manifest_name),
        "suite": str(ARENA_ROOT / suite_name),
        "report": str(ARENA_ROOT / "reports" / f"{wave_id}_build_report.json"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build provisional WT-F proper-data artifacts from VisoMaster clean/Teams buckets.",
    )
    parser.add_argument(
        "--source-pair",
        action="append",
        type=_parse_source_pair,
        help="Source pair in NAME=CLEAN_SOURCE::TEAMS_SOURCE form. Defaults to the current HDTF and quickclips bucket pairs.",
    )
    parser.add_argument(
        "--wave-id",
        default=DEFAULT_WAVE_ID,
        help=f"Wave identifier to write into the inventory. Default: {DEFAULT_WAVE_ID}",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=737,
        help="Split seed for the future proper-data manifest builder. Default: 737",
    )
    parser.add_argument(
        "--lockbox-ratio",
        type=float,
        default=0.20,
        help="Lockbox ratio for the future proper-data manifest builder. Default: 0.20",
    )
    parser.add_argument(
        "--teams-target-frame-count",
        type=int,
        default=16,
        help="Exact Teams real/fake frame count required for inclusion. Default: 16",
    )
    parser.add_argument(
        "--allow-ragged-clean",
        action="store_true",
        help="Keep clean-side rows even when they are not exact fixed-frame pairs. Default is strict clean-side 16/16 filtering.",
    )
    parser.add_argument(
        "--inventory-output",
        help="Path for the generated inventory YAML.",
    )
    parser.add_argument(
        "--manifest-output",
        help="Path for the generated target-domain manifest JSON.",
    )
    parser.add_argument(
        "--suite-output",
        help="Path for the rendered suite YAML.",
    )
    parser.add_argument(
        "--report-output",
        help="Path for the build report JSON.",
    )
    parser.add_argument(
        "--manifest-path-for-suite",
        help="Manifest path string to inject into the suite template. Defaults to --manifest-output.",
    )
    parser.add_argument(
        "--suite-template",
        help="Optional alternative suite template path. Defaults to arena/target_domain_suites.proper_data_future.template.yaml",
    )
    parser.add_argument(
        "--source-log",
        action="append",
        metavar="KEY=VALUE",
        help="Extra source log mapping to include in the inventory. May be repeated.",
    )
    parser.add_argument(
        "--dataset-band",
        action="append",
        metavar="DATASET_KEY=QUALITY:FACESCALE",
        help="Override dataset-level band defaults, e.g. hdtf_20260416=high:big_face",
    )
    parser.add_argument(
        "--note",
        action="append",
        help="Extra note to append to the inventory.",
    )
    parser.add_argument(
        "--project",
        help="Optional Google Cloud project override for manifest reads.",
    )

    args = parser.parse_args(argv)
    source_pairs = tuple(args.source_pair or DEFAULT_SOURCE_PAIRS)

    output_defaults = _default_output_paths(args.wave_id)
    inventory_output = args.inventory_output or output_defaults["inventory"]
    manifest_output = args.manifest_output or output_defaults["manifest"]
    suite_output = args.suite_output or output_defaults["suite"]
    report_output = args.report_output or output_defaults["report"]

    source_logs = dict(DEFAULT_SOURCE_LOGS)
    source_logs.update(_parse_key_value(args.source_log))

    dataset_bands = dict(DEFAULT_DATASET_BANDS)
    for item in args.dataset_band or []:
        text = str(item or "").strip()
        if "=" not in text or ":" not in text:
            raise argparse.ArgumentTypeError(
                f"Expected DATASET_KEY=QUALITY:FACESCALE, got {item!r}"
            )
        dataset_key, remainder = text.split("=", 1)
        quality_band, face_scale_band = remainder.split(":", 1)
        dataset_bands[dataset_key.strip()] = {
            "quality_band": quality_band.strip(),
            "face_scale_band": face_scale_band.strip(),
        }

    build_artifacts(
        source_pairs,
        inventory_output_path=inventory_output,
        manifest_output_path=manifest_output,
        suite_output_path=suite_output,
        report_output_path=report_output,
        manifest_path_for_suite=args.manifest_path_for_suite or manifest_output,
        wave_id=args.wave_id,
        split_seed=args.split_seed,
        lockbox_ratio=args.lockbox_ratio,
        teams_target_frame_count=args.teams_target_frame_count,
        allow_ragged_clean=bool(args.allow_ragged_clean),
        dataset_band_defaults=dataset_bands,
        source_logs=source_logs,
        notes=args.note,
        project=args.project,
        suite_template_path=args.suite_template,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
