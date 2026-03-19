"""
Sequential target-domain validation runner.

Runs validate_custom_sources.py for selected checkpoints across a configurable
set of target-domain suites (Zoom-like real/fake buckets, stress variants, etc).

Suite manifest format (JSON or YAML):

suites:
  - name: zoom_real
    df40_mode: none
    external_real_bucket: effort-collected-data/real/zoom_real
    external_real_method: zoom_real
    max_external_real: 2000

  - name: zoom_wma_flat
    df40_mode: none
    external_fake_bucket: effort-collected-data
    external_fake_prefix: wma_validation/enhanced_fake
    external_fake_method: wma_failure_fake
    external_fake_grouping: per_image
    external_fake_deterministic: true
    max_external_fake: 1202
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Any


def _default_checkpoint_map() -> Dict[str, str]:
    mapping = {}
    for idx in range(1, 9):
        key = f"FT{idx}"
        env_key = f"R4_{key}_CKPT"
        mapping[key] = os.environ.get(env_key, "").strip()
    return mapping


def _load_checkpoint_map(path: str) -> Dict[str, str]:
    with open(path, "r") as f:
        text = f.read().strip()

    if not text:
        return {}

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


def _load_suites(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        text = f.read()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import yaml  # noqa

        data = yaml.safe_load(text)

    if isinstance(data, dict):
        suites = data.get("suites", [])
    elif isinstance(data, list):
        suites = data
    else:
        suites = []

    if not suites:
        raise ValueError("Suite manifest contains no suites.")

    normalized = []
    for idx, suite in enumerate(suites, 1):
        if not isinstance(suite, dict):
            raise ValueError(f"Suite #{idx} must be a mapping.")
        if not suite.get("name"):
            raise ValueError(f"Suite #{idx} missing required key 'name'.")
        normalized.append(suite)

    return normalized


def _append_if_present(args: List[str], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    args.extend([key, str(value)])


def _build_job_args(
    checkpoint_key: str,
    checkpoint_path: str,
    suite: Dict[str, Any],
    common: argparse.Namespace,
) -> List[str]:
    suite_name = str(suite["name"])
    local_ckpt = f"./weights/validation_checkpoint_{checkpoint_key}_{suite_name}.pth"

    args = [
        "--checkpoint_gcs_path", checkpoint_path,
        "--checkpoint_local_path", local_ckpt,
        "--log_prefix", f"td_{suite_name}_{checkpoint_key.lower()}",
        "--run_name", f"TargetDomain {suite_name} {checkpoint_key}",
        "--output_gcs_folder", common.output_gcs_folder,
        "--output_filename_prefix", f"{suite_name}_{checkpoint_key.lower()}_",
        "--wandb_project", common.wandb_project,
        "--frames_per_video", str(int(suite.get("frames_per_video", common.frames_per_video))),
    ]

    if common.detailed_reports:
        args.append("--detailed_reports")
    else:
        args.append("--no_detailed_reports")

    # DF40
    df40_mode = str(suite.get("df40_mode", common.df40_mode))
    args.extend(["--df40_mode", df40_mode])
    if df40_mode != "none":
        args.extend(["--df40_orientation", str(suite.get("df40_orientation", common.df40_orientation))])
        _append_if_present(args, "--df40_methods", suite.get("df40_methods"))

    _append_if_present(args, "--eval_num_workers", suite.get("eval_num_workers"))

    # DeepLive
    _append_if_present(args, "--deeplive_bucket", suite.get("deeplive_bucket"))
    if suite.get("deeplive_bucket"):
        args.extend(["--deeplive_split", str(suite.get("deeplive_split", "all"))])
        _append_if_present(args, "--deeplive_strategies", suite.get("deeplive_strategies"))

    # VisoMaster
    _append_if_present(args, "--visomaster_bucket", suite.get("visomaster_bucket"))
    if suite.get("visomaster_bucket"):
        _append_if_present(args, "--visomaster_frames_bucket", suite.get("visomaster_frames_bucket"))
        _append_if_present(args, "--visomaster_swap_models", suite.get("visomaster_swap_models"))
        _append_if_present(args, "--visomaster_tiers", suite.get("visomaster_tiers"))

    # External real
    _append_if_present(args, "--external_real_bucket", suite.get("external_real_bucket"))
    if suite.get("external_real_bucket"):
        _append_if_present(args, "--external_real_prefix", suite.get("external_real_prefix"))
        _append_if_present(args, "--external_real_method", suite.get("external_real_method"))
        _append_if_present(args, "--external_real_cache", suite.get("external_real_cache"))
        _append_if_present(args, "--max_external_real", suite.get("max_external_real"))
        _append_if_present(args, "--external_real_seed", suite.get("external_real_seed"))
        if bool(suite.get("external_real_deterministic", False)):
            args.append("--external_real_deterministic")

    # External fake
    _append_if_present(args, "--external_fake_bucket", suite.get("external_fake_bucket"))
    if suite.get("external_fake_bucket"):
        _append_if_present(args, "--external_fake_prefix", suite.get("external_fake_prefix"))
        _append_if_present(args, "--external_fake_method", suite.get("external_fake_method"))
        _append_if_present(args, "--external_fake_cache", suite.get("external_fake_cache"))
        _append_if_present(args, "--external_fake_grouping", suite.get("external_fake_grouping"))
        _append_if_present(args, "--max_external_fake", suite.get("max_external_fake"))
        _append_if_present(args, "--external_fake_seed", suite.get("external_fake_seed"))
        if bool(suite.get("external_fake_deterministic", False)):
            args.append("--external_fake_deterministic")

    return args


def _run_job(index: int, total: int, job_name: str, cmd: List[str], dry_run: bool) -> bool:
    print(f"\n{'=' * 88}")
    print(f"[{index}/{total}] {job_name}")
    print(f"{'=' * 88}")
    print("Command:")
    print(" ".join(cmd))

    if dry_run:
        print("[DRY RUN] Skipping execution")
        return True

    start = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"FAILED {job_name} (exit={result.returncode}) after {elapsed:.1f}s")
        return False

    print(f"Completed {job_name} in {elapsed:.1f}s")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential target-domain validation runner")
    parser.add_argument("--checkpoints", type=str, default="FT7")
    parser.add_argument("--checkpoint_map", type=str, default=None)
    parser.add_argument("--suite_manifest", type=str, required=True,
                        help="JSON/YAML manifest containing validation suites.")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "phase2-experiments"))
    parser.add_argument("--output_gcs_folder", type=str,
                        default="gs://training-job-outputs/test_results/target_domain_validation")
    parser.add_argument("--frames_per_video", type=int, default=8)
    parser.add_argument("--detailed_reports", dest="detailed_reports", action="store_true")
    parser.add_argument("--no_detailed_reports", dest="detailed_reports", action="store_false")
    parser.set_defaults(detailed_reports=True)

    # Defaults for optional suite keys
    parser.add_argument("--df40_mode", type=str, default="none",
                        choices=["paired", "fake_only", "real_only", "none"])
    parser.add_argument("--df40_orientation", type=str, default="target_source",
                        choices=["all", "target_source", "source_target"])

    args = parser.parse_args()

    selected = [part.strip().upper() for part in args.checkpoints.split(",") if part.strip()]
    valid = {f"FT{i}" for i in range(1, 9)}
    invalid = [key for key in selected if key not in valid]
    if invalid:
        raise ValueError(f"Invalid checkpoint key(s): {invalid}. Valid keys: {sorted(valid)}")

    checkpoints = _resolve_checkpoints(selected, args.checkpoint_map)
    suites = _load_suites(args.suite_manifest)

    jobs = []
    for suite in suites:
        suite_name = str(suite["name"])
        for ckpt_key in selected:
            ckpt_path = checkpoints[ckpt_key]
            cmd = [sys.executable, "-u", "validate_custom_sources.py"] + _build_job_args(
                checkpoint_key=ckpt_key,
                checkpoint_path=ckpt_path,
                suite=suite,
                common=args,
            )
            jobs.append((f"suite={suite_name} checkpoint={ckpt_key}", cmd))

    print("=" * 88)
    print("Target-domain validation plan")
    print(f"Checkpoints: {selected}")
    print(f"Suites: {[s['name'] for s in suites]}")
    print(f"Output folder: {args.output_gcs_folder}")
    print(f"W&B project: {args.wandb_project}")
    print("=" * 88)

    results = []
    global_start = time.time()
    for idx, (job_name, cmd) in enumerate(jobs, 1):
        ok = _run_job(idx, len(jobs), job_name, cmd, args.dry_run)
        results.append((job_name, ok))

    elapsed = time.time() - global_start
    failed = [job_name for job_name, ok in results if not ok]

    print(f"\n{'=' * 88}")
    print(f"Target-domain summary ({len(jobs)} jobs, {elapsed:.1f}s)")
    print(f"{'=' * 88}")
    for job_name, ok in results:
        print(f"{'OK ' if ok else 'ERR'} {job_name}")

    if failed:
        print(f"\nFailed jobs: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
