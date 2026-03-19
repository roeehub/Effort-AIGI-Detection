"""
Sequential R3 Validation: R3_S1 vs R3_S2 vs R25_F1 (baseline)
==============================================================
Runs validate_custom_sources.py 9 times in subprocess calls:
  3 checkpoints × 3 data sources (DeepLive, ExtReal, QualityEnhancement)

Each invocation is a fully independent Python process — no shared state,
no model persistence between runs. Identical to running 9 Vertex AI jobs,
just sequential in one container.

Usage (in Vertex AI via entrypoint):
  python -u run_r3_validation_sequential.py

Or locally:
  python run_r3_validation_sequential.py [--dry-run]
"""

import argparse
import os
import subprocess
import sys
import time


# ── Checkpoints ──────────────────────────────────────────────────────────────
CHECKPOINTS = {
    "S1": {
        "path": "gs://training-job-outputs/phase2r3_experiments/zjftx8ny/"
                "top_n_effort_20260213_step6500_auc0.9931_eer0.0420.pth",
        "desc": "R3_S1 (scratch, qr_moderate)",
    },
    "S2": {
        "path": "gs://training-job-outputs/phase2r3_experiments/qgcp25lr/"
                "top_n_effort_20260213_step6000_auc0.9934_eer0.0267.pth",
        "desc": "R3_S2 (curriculum phase1 light)",
    },
    "F1": {
        "path": "gs://training-job-outputs/phase2r2_experiments/5w453our/"
                "top_n_effort_20260212_step18500_auc0.9893_eer0.0356.pth",
        "desc": "R25_F1 (baseline)",
    },
}

# ── Data sources ─────────────────────────────────────────────────────────────
DEEPLIVE_BUCKET = "live-deepfake-methods-real-and-fake-frames-cropped"
EXTERNAL_REAL_BUCKET = "effort-collected-data/real/external_youtube_avspeech"

OUTPUT_FOLDER = "gs://training-job-outputs/test_results/r3_S1_vs_S2_vs_F1_validation"
WANDB_PROJECT = "phase2-experiments"


def build_jobs():
    """Build the list of 9 (checkpoint × datasource) jobs."""
    jobs = []

    for ckpt_key, ckpt in CHECKPOINTS.items():
        # Use a unique local path per checkpoint to avoid caching bugs
        ckpt_local_path = f"./weights/validation_checkpoint_{ckpt_key}.pth"

        # 1) DeepLive ALL (edge_cases + minimal_processing)
        jobs.append({
            "name": f"{ckpt_key} × DeepLive ALL",
            "args": [
                "--checkpoint_gcs_path", ckpt["path"],
                "--checkpoint_local_path", ckpt_local_path,
                "--df40_mode", "none",
                "--deeplive_bucket", DEEPLIVE_BUCKET,
                "--deeplive_split", "all",
                "--log_prefix", f"{ckpt_key}_deeplive_all",
                "--run_name", f"{ckpt['desc']} × DeepLive ALL (~860)",
                "--output_gcs_folder", OUTPUT_FOLDER,
                "--output_filename_prefix", f"{ckpt_key}_deeplive_",
                "--wandb_project", WANDB_PROJECT,
            ],
        })

        # 2) External Real
        jobs.append({
            "name": f"{ckpt_key} × External Real",
            "args": [
                "--checkpoint_gcs_path", ckpt["path"],
                "--checkpoint_local_path", ckpt_local_path,
                "--df40_mode", "none",
                "--external_real_bucket", EXTERNAL_REAL_BUCKET,
                "--log_prefix", f"{ckpt_key}_extreal",
                "--run_name", f"{ckpt['desc']} × External Real (~8k)",
                "--output_gcs_folder", OUTPUT_FOLDER,
                "--output_filename_prefix", f"{ckpt_key}_extreal_",
                "--wandb_project", WANDB_PROJECT,
            ],
        })

        # 3) Quality Enhancement ONLY
        jobs.append({
            "name": f"{ckpt_key} × Quality Enhancement",
            "args": [
                "--checkpoint_gcs_path", ckpt["path"],
                "--checkpoint_local_path", ckpt_local_path,
                "--df40_mode", "none",
                "--deeplive_bucket", DEEPLIVE_BUCKET,
                "--deeplive_split", "all",
                "--deeplive_strategies", "quality_enhancement",
                "--log_prefix", f"{ckpt_key}_qualenhance",
                "--run_name", f"{ckpt['desc']} × Quality Enhancement (~320)",
                "--output_gcs_folder", OUTPUT_FOLDER,
                "--output_filename_prefix", f"{ckpt_key}_qualenhance_",
                "--wandb_project", WANDB_PROJECT,
            ],
        })

    return jobs


def run_job(job_num, total, job, dry_run=False):
    """Run a single validate_custom_sources.py invocation as a subprocess."""
    cmd = [sys.executable, "-u", "validate_custom_sources.py"] + job["args"]

    print(f"\n{'=' * 70}")
    print(f"  [{job_num}/{total}] {job['name']}")
    print(f"{'=' * 70}")
    print(f"  Command: {' '.join(cmd)}")
    print()

    if dry_run:
        print("  [DRY RUN] Skipping execution.")
        return True

    start = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  ❌ FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False
    else:
        print(f"\n  ✅ Completed in {elapsed:.1f}s")
        return True


def main():
    parser = argparse.ArgumentParser(description="Sequential R3 validation (9 jobs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing.")
    args = parser.parse_args()

    jobs = build_jobs()

    print("=" * 70)
    print("  R3 Sequential Validation")
    print("  3 checkpoints × 3 data sources = 9 evaluations")
    print("=" * 70)
    print()
    print("Checkpoints:")
    for key, ckpt in CHECKPOINTS.items():
        print(f"  {key}: {ckpt['desc']}")
    print()
    print("Data sources:")
    print("  1. DeepLive ALL (edge_cases + minimal_processing, ~860 samples)")
    print("  2. External Real YouTube AVSpeech (~8k videos)")
    print("  3. Quality Enhancement ONLY (~320 videos)")
    print()
    print(f"Output: {OUTPUT_FOLDER}")
    print(f"W&B:    {WANDB_PROJECT}")
    print()

    total_start = time.time()
    results = []

    for i, job in enumerate(jobs, 1):
        ok = run_job(i, len(jobs), job, dry_run=args.dry_run)
        results.append((job["name"], ok))

    total_elapsed = time.time() - total_start

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY — {len(jobs)} jobs in {total_elapsed:.1f}s")
    print(f"{'=' * 70}")
    for name, ok in results:
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")

    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"\n⚠️  {len(failed)} job(s) failed!")
        sys.exit(1)
    else:
        print(f"\n✅ All {len(jobs)} jobs completed successfully.")


if __name__ == "__main__":
    main()
