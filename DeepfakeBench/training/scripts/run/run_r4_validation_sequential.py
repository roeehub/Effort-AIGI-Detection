"""
Sequential R4 sidecar validation runner.

Runs validate_custom_sources.py once per R4 checkpoint (FT1..FT8) with the
full evidence bundle required for decision making:
- DF40 paired validation (target_source)
- DeepLive split=all (including enhanced strategies)
- VisoMaster validation
- External real gate set
- WMA failure fake gate set
- Detailed report artifacts enabled
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List


def _default_checkpoint_map() -> Dict[str, str]:
    """Read checkpoint paths from environment variables when available."""
    mapping = {}
    for idx in range(1, 9):
        key = f"FT{idx}"
        env_key = f"R4_{key}_CKPT"
        mapping[key] = os.environ.get(env_key, "").strip()
    return mapping


def _load_checkpoint_map(path: str) -> Dict[str, str]:
    """Load checkpoint map from JSON or YAML-like JSON file."""
    with open(path, "r") as f:
        text = f.read().strip()

    if not text:
        return {}

    # Prefer JSON to keep parser dependency minimal.
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import yaml  # noqa

        data = yaml.safe_load(text) or {}

    result: Dict[str, str] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            result[str(key).strip().upper()] = str(value).strip()
    return result


def _resolve_checkpoints(selected_keys: List[str], checkpoint_map_path: str | None) -> Dict[str, str]:
    checkpoint_map = _default_checkpoint_map()
    if checkpoint_map_path:
        checkpoint_map.update(_load_checkpoint_map(checkpoint_map_path))

    resolved: Dict[str, str] = {}
    missing: List[str] = []
    for key in selected_keys:
        path = checkpoint_map.get(key, "")
        if path:
            resolved[key] = path
        else:
            missing.append(key)

    if missing:
        raise ValueError(
            "Missing checkpoint path(s) for: "
            + ", ".join(missing)
            + ". Provide --checkpoint_map or env vars R4_FT*_CKPT."
        )

    return resolved


def _build_job_args(key: str, ckpt_path: str, args: argparse.Namespace) -> List[str]:
    local_ckpt = f"./weights/validation_checkpoint_{key}.pth"
    deeplive_strategies = [
        "edge_cases",
        "minimal_processing",
        "quality_enhancement",
        "edge_cases_enhanced",
        "minimal_processing_enhanced",
    ]

    job_args = [
        "--checkpoint_gcs_path", ckpt_path,
        "--checkpoint_local_path", local_ckpt,
        "--df40_mode", "paired",
        "--df40_orientation", args.df40_orientation,
        "--deeplive_bucket", args.deeplive_bucket,
        "--deeplive_split", "all",
        "--deeplive_strategies", ",".join(deeplive_strategies),
        "--visomaster_bucket", args.visomaster_bucket,
        "--visomaster_frames_bucket", args.visomaster_frames_bucket,
        "--external_real_bucket", args.external_real_bucket,
        "--max_external_real", str(args.max_external_real),
        "--external_fake_bucket", args.external_fake_bucket,
        "--external_fake_prefix", args.external_fake_prefix,
        "--external_fake_method", args.external_fake_method,
        "--external_fake_grouping", args.external_fake_grouping,
        "--max_external_fake", str(args.max_external_fake),
        "--log_prefix", f"R4_{key.lower()}_sidecar",
        "--run_name", f"R4 {key} sidecar validation",
        "--output_gcs_folder", args.output_gcs_folder,
        "--output_filename_prefix", f"R4_{key.lower()}_",
        "--wandb_project", args.wandb_project,
        "--detailed_reports",
    ]

    if args.external_real_deterministic:
        job_args.append("--external_real_deterministic")
    if args.external_fake_deterministic:
        job_args.append("--external_fake_deterministic")

    return job_args


def _run_job(index: int, total: int, key: str, cmd: List[str], dry_run: bool) -> bool:
    print(f"\n{'=' * 78}")
    print(f"[{index}/{total}] Running sidecar validation for {key}")
    print(f"{'=' * 78}")
    print("Command:")
    print(" ".join(cmd))

    if dry_run:
        print("[DRY RUN] Skipping execution")
        return True

    start = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"FAILED {key} (exit={result.returncode}) after {elapsed:.1f}s")
        return False

    print(f"Completed {key} in {elapsed:.1f}s")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential R4 sidecar validation runner")
    parser.add_argument("--checkpoints", type=str, default="FT1,FT2,FT3,FT4,FT5,FT6,FT7,FT8")
    parser.add_argument("--checkpoint_map", type=str, default=None,
                        help="Optional JSON/YAML map of checkpoint keys to GCS paths.")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "phase2-experiments"))
    parser.add_argument("--output_gcs_folder", type=str,
                        default="gs://training-job-outputs/test_results/r4_validation")

    parser.add_argument("--df40_orientation", type=str, default="target_source",
                        choices=["all", "target_source", "source_target"])

    parser.add_argument("--deeplive_bucket", type=str,
                        default="live-deepfake-methods-real-and-fake-frames-cropped")
    parser.add_argument("--visomaster_bucket", type=str,
                        default="live-deepfake-methods-real-and-fake-frames-cropped")
    parser.add_argument("--visomaster_frames_bucket", type=str,
                        default="live-deepfake-methods-real-and-fake-frames")

    parser.add_argument("--external_real_bucket", type=str,
                        default="effort-collected-data/real/external_youtube_avspeech")
    parser.add_argument("--max_external_real", type=int, default=2000)
    parser.add_argument("--external_real_deterministic", action="store_true", default=False,
                        help="Deterministically shape external real samples to --frames_per_video.")

    parser.add_argument("--external_fake_bucket", type=str, default="effort-collected-data")
    parser.add_argument("--external_fake_prefix", type=str, default="wma_validation/enhanced_fake")
    parser.add_argument("--external_fake_method", type=str, default="wma_failure_fake")
    parser.add_argument("--external_fake_grouping", type=str, default="by_folder",
                        choices=["by_folder", "per_image"],
                        help="Grouping mode for external fake source. "
                             "'per_image' is recommended for canonical WMA flat reporting.")
    parser.add_argument("--external_fake_deterministic", action="store_true", default=False,
                        help="Deterministically shape external fake samples to --frames_per_video.")
    parser.add_argument("--max_external_fake", type=int, default=1202)

    args = parser.parse_args()

    selected = [part.strip().upper() for part in args.checkpoints.split(",") if part.strip()]
    valid = {f"FT{i}" for i in range(1, 9)}
    invalid = [key for key in selected if key not in valid]
    if invalid:
        raise ValueError(f"Invalid checkpoint key(s): {invalid}. Valid keys: {sorted(valid)}")

    checkpoints = _resolve_checkpoints(selected, args.checkpoint_map)

    jobs = []
    for key in selected:
        ckpt_path = checkpoints[key]
        cmd = [sys.executable, "-u", "validate_custom_sources.py"] + _build_job_args(key, ckpt_path, args)
        jobs.append((key, cmd))

    print("=" * 78)
    print("R4 sidecar validation plan")
    print(f"Checkpoints: {selected}")
    print(f"Output folder: {args.output_gcs_folder}")
    print(f"W&B project: {args.wandb_project}")
    print("=" * 78)

    results = []
    global_start = time.time()
    for idx, (key, cmd) in enumerate(jobs, 1):
        ok = _run_job(idx, len(jobs), key, cmd, args.dry_run)
        results.append((key, ok))

    elapsed = time.time() - global_start
    failed = [key for key, ok in results if not ok]

    print(f"\n{'=' * 78}")
    print(f"R4 sidecar summary ({len(jobs)} jobs, {elapsed:.1f}s)")
    print(f"{'=' * 78}")
    for key, ok in results:
        print(f"{'OK ' if ok else 'ERR'} {key}")

    if failed:
        print(f"\nFailed checkpoints: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
