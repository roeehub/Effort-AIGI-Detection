#!/usr/bin/env python3
"""
Build a reusable target-domain suite manifest from a frozen Teams manifest.

Default behavior:
- keep the core real suites on `dev` and `lockbox`
- emit `teams_fake_all` on `dev`
- emit fake family slices on `dev`
- emit fake method-level slices on `dev`

This complements `build_teams_target_domain_manifest.py`: the frozen manifest
captures discovered videos, while this script turns that frozen artifact into a
scorecard-ready suite manifest with consistent naming.
"""

from __future__ import annotations

import argparse
import json
import os
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
            content_type="text/plain; charset=utf-8",
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


def _csv_list(value: str | None, default: Sequence[str]) -> List[str]:
    if value is None:
        return [str(item).strip() for item in default if str(item).strip()]
    items = [part.strip() for part in str(value).split(",") if part.strip()]
    return items or [str(item).strip() for item in default if str(item).strip()]


def _coerce_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_label(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"0", "real"}:
        return "real"
    if text in {"1", "fake"}:
        return "fake"
    return text


def _sorted_by_count_then_name(names: Iterable[str], counts: Dict[str, int]) -> List[str]:
    return sorted(
        {str(name).strip() for name in names if str(name).strip()},
        key=lambda name: (-int(counts.get(name, 0)), name),
    )


def _build_real_suite(manifest_path: str, split: str, slice_name: str) -> Dict[str, Any]:
    return {
        "name": f"{slice_name}_{split}",
        "df40_mode": "none",
        "external_real_manifest": manifest_path,
        "external_real_manifest_split": split,
        "external_real_manifest_slices": slice_name,
        "external_real_method": "teams_real",
        "external_real_deterministic": True,
    }


def _build_fake_suite(
    manifest_path: str,
    split: str,
    slice_name: str,
    method_name: str | None = None,
) -> Dict[str, Any]:
    suite = {
        "name": f"{slice_name}_{split}",
        "df40_mode": "none",
        "external_fake_manifest": manifest_path,
        "external_fake_manifest_split": split,
        "external_fake_manifest_slices": slice_name,
        "external_fake_deterministic": True,
    }
    if method_name:
        suite["external_fake_method"] = method_name
    return suite


def build_suite_manifest(
    manifest: Dict[str, Any],
    manifest_path: str,
    real_splits: Sequence[str] | None = None,
    fake_splits: Sequence[str] | None = None,
    real_slices: Sequence[str] | None = None,
    include_fake_all: bool = True,
    include_fake_family_slices: bool = True,
    include_fake_method_slices: bool = True,
) -> Dict[str, Any]:
    videos = manifest.get("videos", [])
    if not isinstance(videos, list) or not videos:
        raise ValueError("Manifest contains no videos.")

    requested_real_splits = [str(item).strip() for item in (real_splits or ["dev", "lockbox"]) if str(item).strip()]
    requested_fake_splits = [str(item).strip() for item in (fake_splits or ["dev"]) if str(item).strip()]
    requested_real_slices = [
        str(item).strip()
        for item in (real_slices or ["teams_real_all", "teams_real_poor_quality", "teams_real_lighting_extreme"])
        if str(item).strip()
    ]

    rows: List[Dict[str, Any]] = []
    for row in videos:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "label": _normalize_label(row.get("label")),
                "split": str(row.get("split", "")).strip(),
                "method": str(row.get("method", "")).strip(),
                "slices": [
                    str(item).strip()
                    for item in _coerce_list(row.get("slices"))
                    if str(item).strip()
                ],
            }
        )

    available_real_splits = {row["split"] for row in rows if row["label"] == "real" and row["split"]}
    available_fake_splits = {row["split"] for row in rows if row["label"] == "fake" and row["split"]}

    missing_real_splits = [split for split in requested_real_splits if split not in available_real_splits]
    if missing_real_splits:
        raise ValueError(f"Requested real split(s) not found in manifest: {missing_real_splits}")

    missing_fake_splits = [split for split in requested_fake_splits if split not in available_fake_splits]
    if missing_fake_splits:
        raise ValueError(f"Requested fake split(s) not found in manifest: {missing_fake_splits}")

    real_slice_counts: Dict[Tuple[str, str], int] = {}
    fake_slice_counts: Dict[Tuple[str, str], int] = {}
    fake_method_counts: Dict[Tuple[str, str], int] = {}

    for row in rows:
        split = row["split"]
        if row["label"] == "real":
            for slice_name in row["slices"]:
                real_slice_counts[(split, slice_name)] = real_slice_counts.get((split, slice_name), 0) + 1
        elif row["label"] == "fake":
            for slice_name in row["slices"]:
                fake_slice_counts[(split, slice_name)] = fake_slice_counts.get((split, slice_name), 0) + 1
            if row["method"]:
                fake_method_counts[(split, row["method"])] = fake_method_counts.get((split, row["method"]), 0) + 1

    suites: List[Dict[str, Any]] = []
    seen_suite_names = set()

    def add_suite(suite: Dict[str, Any]) -> None:
        suite_name = str(suite.get("name", "")).strip()
        if not suite_name or suite_name in seen_suite_names:
            return
        seen_suite_names.add(suite_name)
        suites.append(suite)

    for split in requested_real_splits:
        for slice_name in requested_real_slices:
            if real_slice_counts.get((split, slice_name), 0) <= 0:
                continue
            add_suite(_build_real_suite(manifest_path=manifest_path, split=split, slice_name=slice_name))

    for split in requested_fake_splits:
        split_fake_methods = {
            method_name
            for split_name, method_name in fake_method_counts.keys()
            if split_name == split
        }
        split_fake_slices = {
            slice_name
            for split_name, slice_name in fake_slice_counts.keys()
            if split_name == split
        }

        if include_fake_all and fake_slice_counts.get((split, "teams_fake_all"), 0) > 0:
            add_suite(_build_fake_suite(manifest_path=manifest_path, split=split, slice_name="teams_fake_all"))

        if include_fake_family_slices:
            family_slices = split_fake_slices - split_fake_methods - {"teams_fake_all"}
            family_count_map = {
                name: fake_slice_counts.get((split, name), 0)
                for name in family_slices
            }
            for slice_name in _sorted_by_count_then_name(family_slices, family_count_map):
                add_suite(_build_fake_suite(manifest_path=manifest_path, split=split, slice_name=slice_name))

        if include_fake_method_slices:
            method_count_map = {
                name: fake_method_counts.get((split, name), 0)
                for name in split_fake_methods
            }
            for method_name in _sorted_by_count_then_name(split_fake_methods, method_count_map):
                add_suite(
                    _build_fake_suite(
                        manifest_path=manifest_path,
                        split=split,
                        slice_name=method_name,
                        method_name=method_name,
                    )
                )

    return {
        "generated_from_manifest": manifest_path,
        "real_splits": list(requested_real_splits),
        "fake_splits": list(requested_fake_splits),
        "include_fake_all": bool(include_fake_all),
        "include_fake_family_slices": bool(include_fake_family_slices),
        "include_fake_method_slices": bool(include_fake_method_slices),
        "suites": suites,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Teams target-domain scorecard suite manifest")
    parser.add_argument(
        "--manifest",
        required=True,
        help=(
            "Frozen manifest path (local or gs://). This path is also written "
            "verbatim into the emitted suites, so pass the same path form the "
            "runner will later consume."
        ),
    )
    parser.add_argument("--output", required=True, help="Output suite-manifest path (local or gs://).")
    parser.add_argument(
        "--real-splits",
        default="dev,lockbox",
        help="Comma-separated real splits to emit. Default: dev,lockbox",
    )
    parser.add_argument(
        "--fake-splits",
        default="dev",
        help="Comma-separated fake splits to emit. Default: dev",
    )
    parser.add_argument(
        "--real-slices",
        default="teams_real_all,teams_real_poor_quality,teams_real_lighting_extreme",
        help="Comma-separated real slices to emit. Default: teams_real_all,teams_real_poor_quality,teams_real_lighting_extreme",
    )
    parser.add_argument("--include-fake-all", dest="include_fake_all", action="store_true")
    parser.add_argument("--no-include-fake-all", dest="include_fake_all", action="store_false")
    parser.set_defaults(include_fake_all=True)
    parser.add_argument("--include-fake-family-slices", dest="include_fake_family_slices", action="store_true")
    parser.add_argument("--no-include-fake-family-slices", dest="include_fake_family_slices", action="store_false")
    parser.set_defaults(include_fake_family_slices=True)
    parser.add_argument("--include-fake-method-slices", dest="include_fake_method_slices", action="store_true")
    parser.add_argument("--no-include-fake-method-slices", dest="include_fake_method_slices", action="store_false")
    parser.set_defaults(include_fake_method_slices=True)
    args = parser.parse_args()

    manifest = _load_structured_data(args.manifest)
    payload = build_suite_manifest(
        manifest=manifest,
        manifest_path=args.manifest,
        real_splits=_csv_list(args.real_splits, ["dev", "lockbox"]),
        fake_splits=_csv_list(args.fake_splits, ["dev"]),
        real_slices=_csv_list(
            args.real_slices,
            ["teams_real_all", "teams_real_poor_quality", "teams_real_lighting_extreme"],
        ),
        include_fake_all=args.include_fake_all,
        include_fake_family_slices=args.include_fake_family_slices,
        include_fake_method_slices=args.include_fake_method_slices,
    )

    import yaml

    _write_text_to_path(args.output, yaml.safe_dump(payload, sort_keys=False))
    print(f"Generated {len(payload['suites'])} suites -> {args.output}")


if __name__ == "__main__":
    main()
