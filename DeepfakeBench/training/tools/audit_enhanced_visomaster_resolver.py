#!/usr/bin/env python3
"""
Audit and resolve companion linkage for gs://enhanced-visomaster-cropped.

This turns the raw per-sample manifests into a training-safe resolver manifest
with one row per base ``sample_id``. The key rule is that companion buckets are
resolved by existence checks, not by trusting the enhanced manifest metadata.

Outputs:
- JSON manifest with rows + summary metadata
- Optional CSV export with the same per-sample rows

Typical usage:
    python tools/audit_enhanced_visomaster_resolver.py \
        --output-json /tmp/enhanced_visomaster_resolver_2026-04-06.json \
        --output-csv /tmp/enhanced_visomaster_resolver_2026-04-06.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, List, Tuple

from fsspec.core import url_to_fs
from google.cloud import storage


EXPECTED_ENHANCERS: Tuple[str, ...] = (
    "codeformer",
    "gfpgan",
    "gpen-1024",
    "gpen-2048",
    "gpen-256",
    "gpen-512",
    "restoreformer++",
    "vqfr-v2",
)
FRAME_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

logger = logging.getLogger("audit-enhanced-visomaster-resolver")


def _looks_like_frame(blob_name: str) -> bool:
    return PurePosixPath(blob_name).suffix.lower() in FRAME_SUFFIXES


def _write_json_uri(uri: str, payload: Dict[str, Any]) -> None:
    fs, path = url_to_fs(uri)
    parent = str(PurePosixPath(path).parent)
    if parent and parent not in {".", ""} and not fs.exists(parent):
        fs.makedirs(parent, exist_ok=True)
    with fs.open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def _write_csv_uri(uri: str, rows: List[Dict[str, Any]]) -> None:
    fs, path = url_to_fs(uri)
    parent = str(PurePosixPath(path).parent)
    if parent and parent not in {".", ""} and not fs.exists(parent):
        fs.makedirs(parent, exist_ok=True)

    fieldnames = [
        "sample_id",
        "strategy",
        "claimed_real_fake_bucket",
        "claimed_real_fake_path",
        "resolved_companion_bucket",
        "resolved_companion_path",
        "resolution_status",
        "teams_v2_real_exists",
        "teams_v2_fake_exists",
        "clean_real_exists",
        "clean_fake_exists",
        "resolved_real_frame_count",
        "resolved_fake_frame_count",
        "resolved_total_frame_count",
        "resolved_extensions",
        "available_enhancers",
        "available_enhancer_count",
        "expected_enhancer_count",
        "all_expected_enhancers_present",
        "missing_expected_enhancers",
        "enhancer_frame_counts",
        "enhanced_total_frame_count",
        "manifest_enhancer_count",
        "manifest_all_enhancers_present",
        "manifest_total_frames",
        "manifest_uploaded_at",
        "manifest_pipeline_version",
        "claimed_bucket_matches_resolution",
    ]

    def _csv_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True)
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    with fs.open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})


def _discover_enhanced_bucket(
    bucket: storage.Bucket,
    sample_limit: int | None = None,
) -> tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
    """
    Scan the enhanced bucket once.

    Returns:
    - sample_id -> manifest blob name
    - sample_id -> enhancer -> frame count
    """
    manifest_paths: Dict[str, str] = {}
    enhancer_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    for blob in bucket.list_blobs(prefix="samples/"):
        name = blob.name
        parts = name.split("/")
        if len(parts) < 3 or parts[0] != "samples":
            continue

        sample_id = parts[1]
        if sample_limit is not None and sample_id not in manifest_paths and len(manifest_paths) >= sample_limit:
            # Preserve frames for already-admitted samples, but stop admitting new ones.
            continue

        if name.endswith("manifest.json"):
            manifest_paths[sample_id] = name
            continue

        if not _looks_like_frame(name):
            continue

        if len(parts) < 5 or parts[2] != "frames":
            continue

        enhancer = parts[3]
        enhancer_counts[sample_id][enhancer] += 1

    frame_counts = {
        sample_id: dict(sorted(counter.items()))
        for sample_id, counter in enhancer_counts.items()
    }
    return manifest_paths, frame_counts


def _probe_side_exists(bucket: storage.Bucket, sample_id: str, side: str) -> bool:
    prefix = f"samples/{sample_id}/frames/{side}/"
    for blob in bucket.list_blobs(prefix=prefix, max_results=8):
        if _looks_like_frame(blob.name):
            return True
    return False


def _count_side_frames(bucket: storage.Bucket, sample_id: str, side: str) -> tuple[int, Dict[str, int]]:
    prefix = f"samples/{sample_id}/frames/{side}/"
    frame_count = 0
    suffix_counts: Counter[str] = Counter()
    for blob in bucket.list_blobs(prefix=prefix):
        if not _looks_like_frame(blob.name):
            continue
        frame_count += 1
        suffix_counts[PurePosixPath(blob.name).suffix.lower()] += 1
    return frame_count, dict(sorted(suffix_counts.items()))


def _resolve_companion(
    sample_id: str,
    teams_bucket: storage.Bucket,
    clean_bucket: storage.Bucket,
    teams_bucket_name: str,
    clean_bucket_name: str,
) -> Dict[str, Any]:
    teams_real_exists = _probe_side_exists(teams_bucket, sample_id, "real")
    teams_fake_exists = _probe_side_exists(teams_bucket, sample_id, "fake")
    clean_real_exists = _probe_side_exists(clean_bucket, sample_id, "real")
    clean_fake_exists = _probe_side_exists(clean_bucket, sample_id, "fake")

    teams_pair_exists = teams_real_exists and teams_fake_exists
    clean_pair_exists = clean_real_exists and clean_fake_exists

    if teams_pair_exists:
        resolved_bucket_name = teams_bucket_name
        resolved_bucket = teams_bucket
        resolution_status = "teams_v2_companion"
    elif clean_pair_exists:
        resolved_bucket_name = clean_bucket_name
        resolved_bucket = clean_bucket
        resolution_status = "clean_companion_only"
    else:
        resolved_bucket_name = ""
        resolved_bucket = None
        resolution_status = "missing_companion"

    resolved_real_count = 0
    resolved_fake_count = 0
    resolved_extensions: Dict[str, Dict[str, int]] = {"real": {}, "fake": {}}
    if resolved_bucket is not None:
        resolved_real_count, resolved_extensions["real"] = _count_side_frames(
            resolved_bucket, sample_id, "real"
        )
        resolved_fake_count, resolved_extensions["fake"] = _count_side_frames(
            resolved_bucket, sample_id, "fake"
        )

    return {
        "resolution_status": resolution_status,
        "resolved_companion_bucket": resolved_bucket_name,
        "resolved_companion_path": f"samples/{sample_id}/" if resolved_bucket_name else "",
        "teams_v2_real_exists": teams_real_exists,
        "teams_v2_fake_exists": teams_fake_exists,
        "clean_real_exists": clean_real_exists,
        "clean_fake_exists": clean_fake_exists,
        "resolved_real_frame_count": resolved_real_count,
        "resolved_fake_frame_count": resolved_fake_count,
        "resolved_total_frame_count": resolved_real_count + resolved_fake_count,
        "resolved_extensions": resolved_extensions,
    }


def _load_manifest(bucket: storage.Bucket, manifest_path: str) -> Dict[str, Any]:
    return json.loads(bucket.blob(manifest_path).download_as_text())


def _build_row(
    sample_id: str,
    manifest: Dict[str, Any],
    enhancer_frame_counts: Dict[str, int],
    companion_resolution: Dict[str, Any],
) -> Dict[str, Any]:
    claimed_bucket = str(manifest.get("real_fake_bucket") or "")
    claimed_path = str(manifest.get("real_fake_path") or "")
    manifest_enhancers = manifest.get("enhancers") or {}
    manifest_enhancer_count = int(manifest.get("enhancer_count") or len(manifest_enhancers))

    available_enhancers = sorted(enhancer_frame_counts.keys())
    expected_set = set(EXPECTED_ENHANCERS)
    available_set = set(available_enhancers)
    missing_expected = sorted(expected_set - available_set)

    resolved_bucket = companion_resolution["resolved_companion_bucket"]
    claimed_bucket_matches_resolution = bool(resolved_bucket) and claimed_bucket == resolved_bucket

    return {
        "sample_id": sample_id,
        "strategy": manifest.get("strategy", ""),
        "claimed_real_fake_bucket": claimed_bucket,
        "claimed_real_fake_path": claimed_path,
        "resolved_companion_bucket": resolved_bucket,
        "resolved_companion_path": companion_resolution["resolved_companion_path"],
        "resolution_status": companion_resolution["resolution_status"],
        "teams_v2_real_exists": companion_resolution["teams_v2_real_exists"],
        "teams_v2_fake_exists": companion_resolution["teams_v2_fake_exists"],
        "clean_real_exists": companion_resolution["clean_real_exists"],
        "clean_fake_exists": companion_resolution["clean_fake_exists"],
        "resolved_real_frame_count": companion_resolution["resolved_real_frame_count"],
        "resolved_fake_frame_count": companion_resolution["resolved_fake_frame_count"],
        "resolved_total_frame_count": companion_resolution["resolved_total_frame_count"],
        "resolved_extensions": companion_resolution["resolved_extensions"],
        "available_enhancers": available_enhancers,
        "available_enhancer_count": len(available_enhancers),
        "expected_enhancer_count": len(EXPECTED_ENHANCERS),
        "all_expected_enhancers_present": len(missing_expected) == 0,
        "missing_expected_enhancers": missing_expected,
        "enhancer_frame_counts": enhancer_frame_counts,
        "enhanced_total_frame_count": sum(enhancer_frame_counts.values()),
        "manifest_enhancer_count": manifest_enhancer_count,
        "manifest_all_enhancers_present": bool(manifest.get("all_enhancers_present", False)),
        "manifest_total_frames": int(manifest.get("total_frames") or 0),
        "manifest_uploaded_at": manifest.get("uploaded_at", ""),
        "manifest_pipeline_version": manifest.get("pipeline_version", ""),
        "claimed_bucket_matches_resolution": claimed_bucket_matches_resolution,
    }


def _build_summary(
    rows: List[Dict[str, Any]],
    teams_bucket_name: str,
    clean_bucket_name: str,
) -> Dict[str, Any]:
    status_counts = Counter(row["resolution_status"] for row in rows)
    claimed_bucket_counts = Counter(row["claimed_real_fake_bucket"] for row in rows)

    claimed_bucket_mismatch_count = 0
    teams_claim_but_clean_only = 0
    for row in rows:
        resolved_bucket = row["resolved_companion_bucket"]
        claimed_bucket = row["claimed_real_fake_bucket"]
        if resolved_bucket and claimed_bucket != resolved_bucket:
            claimed_bucket_mismatch_count += 1
        if (
            claimed_bucket == teams_bucket_name
            and row["resolution_status"] == "clean_companion_only"
        ):
            teams_claim_but_clean_only += 1

    examples = [
        row["sample_id"]
        for row in rows
        if row["claimed_real_fake_bucket"] == teams_bucket_name
        and row["resolution_status"] == "clean_companion_only"
    ][:20]

    return {
        "sample_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "claimed_real_fake_bucket_counts": dict(sorted(claimed_bucket_counts.items())),
        "all_expected_enhancers_present_count": sum(
            1 for row in rows if row["all_expected_enhancers_present"]
        ),
        "claimed_bucket_mismatch_count": claimed_bucket_mismatch_count,
        "teams_claim_but_clean_only_count": teams_claim_but_clean_only,
        "teams_claim_but_clean_only_examples": examples,
        "resolved_to_teams_v2_count": sum(
            1 for row in rows if row["resolved_companion_bucket"] == teams_bucket_name
        ),
        "resolved_to_clean_count": sum(
            1 for row in rows if row["resolved_companion_bucket"] == clean_bucket_name
        ),
        "missing_companion_count": sum(
            1 for row in rows if row["resolution_status"] == "missing_companion"
        ),
    }


def _log_summary(summary: Dict[str, Any]) -> None:
    logger.info("Resolver audit summary:")
    logger.info("  - Sample count: %d", summary["sample_count"])
    logger.info("  - Status counts: %s", summary["status_counts"])
    logger.info(
        "  - All expected enhancers present: %d",
        summary["all_expected_enhancers_present_count"],
    )
    logger.info(
        "  - Claimed bucket mismatches: %d",
        summary["claimed_bucket_mismatch_count"],
    )
    logger.info(
        "  - Teams-claim but clean-only: %d",
        summary["teams_claim_but_clean_only_count"],
    )
    if summary["teams_claim_but_clean_only_examples"]:
        logger.info(
            "  - Example teams-claim/clean-only sample_ids: %s",
            summary["teams_claim_but_clean_only_examples"][:10],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit companion resolution for enhanced-visomaster-cropped."
    )
    parser.add_argument(
        "--enhanced-bucket",
        type=str,
        default="enhanced-visomaster-cropped",
    )
    parser.add_argument(
        "--teams-bucket",
        type=str,
        default="live-deepfake-methods-real-and-fake-frames-cropped-teams-v2",
    )
    parser.add_argument(
        "--clean-bucket",
        type=str,
        default="live-deepfake-methods-real-and-fake-frames-cropped",
    )
    parser.add_argument(
        "--gcs-project",
        type=str,
        default="train-cvit2",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Local path or URI for the JSON resolver manifest.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Local path or URI for a flattened CSV export.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Process only the first N sample_ids after discovery (debugging).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    client = storage.Client(project=args.gcs_project)
    enhanced_bucket = client.bucket(args.enhanced_bucket)
    teams_bucket = client.bucket(args.teams_bucket)
    clean_bucket = client.bucket(args.clean_bucket)

    logger.info("Scanning enhanced bucket: gs://%s", args.enhanced_bucket)
    manifest_paths, enhancer_frame_counts = _discover_enhanced_bucket(
        enhanced_bucket,
        sample_limit=args.sample_limit,
    )

    sample_ids = sorted(manifest_paths.keys())
    logger.info("Discovered %d manifest-backed samples", len(sample_ids))

    rows: List[Dict[str, Any]] = []
    for idx, sample_id in enumerate(sample_ids, start=1):
        if idx == 1 or idx % 100 == 0 or idx == len(sample_ids):
            logger.info("Resolving companion sample %d/%d: %s", idx, len(sample_ids), sample_id)

        manifest = _load_manifest(enhanced_bucket, manifest_paths[sample_id])
        companion_resolution = _resolve_companion(
            sample_id=sample_id,
            teams_bucket=teams_bucket,
            clean_bucket=clean_bucket,
            teams_bucket_name=args.teams_bucket,
            clean_bucket_name=args.clean_bucket,
        )
        row = _build_row(
            sample_id=sample_id,
            manifest=manifest,
            enhancer_frame_counts=enhancer_frame_counts.get(sample_id, {}),
            companion_resolution=companion_resolution,
        )
        rows.append(row)

    summary = _build_summary(rows, args.teams_bucket, args.clean_bucket)
    _log_summary(summary)

    payload = {
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "enhanced_bucket": args.enhanced_bucket,
            "teams_bucket": args.teams_bucket,
            "clean_bucket": args.clean_bucket,
            "gcs_project": args.gcs_project,
            "expected_enhancers": list(EXPECTED_ENHANCERS),
        },
        "summary": summary,
        "rows": rows,
    }

    if args.output_json:
        _write_json_uri(args.output_json, payload)
        logger.info("Wrote JSON resolver manifest: %s", args.output_json)
    if args.output_csv:
        _write_csv_uri(args.output_csv, rows)
        logger.info("Wrote CSV resolver manifest: %s", args.output_csv)


if __name__ == "__main__":
    main()
