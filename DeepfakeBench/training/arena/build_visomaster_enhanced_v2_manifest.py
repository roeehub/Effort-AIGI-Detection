#!/usr/bin/env python3
"""
Build a validation manifest for the flat fake-only VisoMaster enhanced v2 bucket.

The bucket layout is:

    gs://visomaster-enhanced-face-cropped-v2/
      fake/<model_folder>/<image>.png

Each image becomes a single manifest row so validation can preserve per-folder
method labels inside one suite. Downstream evaluation may deterministically pad
these single-frame samples to `frames_per_video` if desired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


_VALID_SUFFIXES = (".png", ".jpg", ".jpeg")
_SEQ_RE = re.compile(r"_seq(?P<seq>\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)


def _stable_identity(*parts: str) -> int:
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def _iter_rows(bucket_name: str, project: str) -> Iterable[Dict[str, object]]:
    from google.cloud import storage

    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    method_counts: Counter[str] = Counter()

    for blob in client.list_blobs(bucket, prefix="fake/"):
        name = blob.name
        parts = name.split("/")
        if len(parts) != 3:
            continue

        _, model_folder, filename = parts
        if not filename.lower().endswith(_VALID_SUFFIXES):
            continue

        method = f"visomaster_enhanced_v2_{model_folder}"
        seq_match = _SEQ_RE.search(filename)
        sequence_id = seq_match.group("seq") if seq_match else None
        gs_path = f"gs://{bucket_name}/{name}"

        method_counts[method] += 1
        stem = Path(filename).stem
        yield {
            "video_id": f"{model_folder}__{stem}",
            "label": "fake",
            "method": method,
            "split": "all",
            "frame_paths": [gs_path],
            "identity": _stable_identity(bucket_name, model_folder, filename),
            "slices": [
                "visomaster_enhanced_v2_all",
                method,
            ],
            "metadata": {
                "bucket_path": name,
                "model_folder": model_folder,
                "filename": filename,
                "sequence_id": sequence_id,
            },
        }


def build_manifest(bucket_name: str, project: str) -> Dict[str, object]:
    rows: List[Dict[str, object]] = list(_iter_rows(bucket_name=bucket_name, project=project))
    rows.sort(key=lambda row: (str(row["method"]), str(row["video_id"])))

    method_counts = Counter(str(row["method"]) for row in rows)
    unique_frames = sum(len(row.get("frame_paths", [])) for row in rows)

    return {
        "manifest_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_bucket": bucket_name,
        "layout": "flat_fake_only_by_model_folder",
        "notes": (
            "Each image is represented as a single fake sample. Suites that use "
            "this manifest may deterministically pad to the requested "
            "frames_per_video for video-shaped inference."
        ),
        "summary": {
            "videos": len(rows),
            "unique_source_frames": unique_frames,
            "method_counts": dict(sorted(method_counts.items())),
        },
        "videos": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket",
        default="visomaster-enhanced-face-cropped-v2",
        help="GCS bucket name containing the flat fake-only VisoMaster enhanced v2 set.",
    )
    parser.add_argument(
        "--project",
        default="train-cvit2",
        help="GCP project used for the storage client.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Local JSON path to write.",
    )
    args = parser.parse_args()

    manifest = build_manifest(bucket_name=args.bucket, project=args.project)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")

    summary = manifest["summary"]
    print(f"Wrote {output_path}")
    print(f"Videos: {summary['videos']}")
    print(f"Unique source frames: {summary['unique_source_frames']}")
    print("Per-method counts:")
    for method, count in summary["method_counts"].items():
        print(f"  {method}: {count}")


if __name__ == "__main__":
    main()
