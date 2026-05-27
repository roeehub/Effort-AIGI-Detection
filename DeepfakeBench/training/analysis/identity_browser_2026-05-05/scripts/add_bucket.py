#!/usr/bin/env python3
"""Add a new bucket / suite of frames to the identity browser.

This is a thin wrapper around `build_browser.py`.  Two ways to use it:

(1) You already have a fully-formed scope-manifest CSV with the columns:
    suite, video_id, frame_path, label, is_lockbox, bucket
    Then just point build_browser.py at all manifests:

        python scripts/build_browser.py \\
            --manifest scope_manifest.csv \\
            --manifest /path/to/new_bucket_manifest.csv

(2) You have raw inputs that need to be converted into a manifest first.
    Edit the `build_manifest_from_*` example below to plug in your source
    (e.g. a scorecard CSV from the promotion contract launcher), then run:

        python scripts/add_bucket.py --new-manifest /tmp/my_bucket.csv \\
            --then-rebuild

Either way, the build is idempotent: existing frames/thumbs are reused.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

OUTPUT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = OUTPUT_ROOT / "scope_manifest.csv"


def build_manifest_from_scorecard_csv(
    scorecard_csv: Path,
    suite_name: str,
    is_lockbox: bool,
    out_csv: Path,
    bucket: str = "teams-faces-data-test-2914-fake-4420-real-feb-28",
) -> Path:
    """EXAMPLE: turn a per-frame scorecard CSV into a scope-manifest CSV.

    Expected scorecard columns (typical for promotion-contract scorecards):
      video_id, frame_path, label

    Adjust the column mapping below if your CSV has different names.
    """

    df = pd.read_csv(scorecard_csv)
    needed = {"video_id", "frame_path", "label"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"scorecard {scorecard_csv} missing columns: {missing}")
    out = pd.DataFrame(
        {
            "suite": suite_name,
            "video_id": df["video_id"],
            "frame_path": df["frame_path"],
            "label": df["label"].astype(int),
            "is_lockbox": bool(is_lockbox),
            "bucket": bucket,
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--new-manifest",
        type=Path,
        help="Path to a NEW scope-manifest CSV to register alongside the bundled one.",
    )
    ap.add_argument(
        "--from-scorecard",
        type=Path,
        help="OPTIONAL: build a new manifest from this scorecard CSV first.",
    )
    ap.add_argument(
        "--suite-name",
        type=str,
        default="custom_suite",
        help="Suite label to use when building from --from-scorecard.",
    )
    ap.add_argument(
        "--is-lockbox",
        action="store_true",
        help="Mark the new suite as lockbox (default: dev).",
    )
    ap.add_argument(
        "--then-rebuild",
        action="store_true",
        help="After registering the manifest, run build_browser.py.",
    )
    args = ap.parse_args()

    if args.from_scorecard:
        out_csv = args.new_manifest or (
            OUTPUT_ROOT / "data" / f"manifest_{args.suite_name}.csv"
        )
        build_manifest_from_scorecard_csv(
            args.from_scorecard,
            suite_name=args.suite_name,
            is_lockbox=args.is_lockbox,
            out_csv=out_csv,
        )
        print(f"Built manifest: {out_csv}")
        args.new_manifest = out_csv

    if args.then_rebuild:
        build_script = OUTPUT_ROOT / "scripts" / "build_browser.py"
        cmd = [sys.executable, str(build_script), "--manifest", str(DEFAULT_MANIFEST)]
        if args.new_manifest:
            cmd += ["--manifest", str(args.new_manifest)]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
