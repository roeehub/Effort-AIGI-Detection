"""
Sequential R3 Fine-Tune Validation: FT1, FT2, FT3, FT4 vs R25_F1 (baseline)
=============================================================================
Runs validate_custom_sources.py 15 times in subprocess calls:
  5 checkpoints × 3 data sources (DeepLive, ExtReal, QualityEnhancement)

Same evaluation protocol as run_r3_validation_sequential.py (S1/S2/F1),
now extended to the R3 fine-tune models.

Each invocation is a fully independent Python process — no shared state,
no model persistence between runs. Identical to running 15 Vertex AI jobs,
just sequential in one container.

Usage (in Vertex AI via entrypoint):
  python -u run_r3ft_validation_sequential.py

Or locally:
  python run_r3ft_validation_sequential.py [--dry-run]

To run only FT3 (the top candidate):
  python run_r3ft_validation_sequential.py --checkpoints FT3

To run FT3 + F1 baseline comparison:
  python run_r3ft_validation_sequential.py --checkpoints FT3,F1
"""

import argparse
import os
import subprocess
import sys
import time


# ── Checkpoints ──────────────────────────────────────────────────────────────
CHECKPOINTS = {
    "FT1": {
        "path": "gs://training-job-outputs/phase2r3_experiments/w7wi9lpj/"
                "top_n_effort_20260213_step500_auc0.9962_eer0.0191.pth",
        "desc": "R3_FT1 (qr_moderate, from F1)",
    },
    "FT2": {
        "path": "gs://training-job-outputs/phase2r3_experiments/3cxpwxgn/"
                "top_n_effort_20260213_step500_auc0.9954_eer0.0229.pth",
        "desc": "R3_FT2 (qr_light, from F1)",
    },
    "FT3": {
        "path": "gs://training-job-outputs/phase2r3_experiments/kzfu116l/"
                "top_n_effort_20260213_step2500_auc0.9966_eer0.0191.pth",
        "desc": "R3_FT3 (base_only, +QE data, from F1)",
    },
    "FT4": {
        "path": "gs://training-job-outputs/phase2r3_experiments/3k2jzoe8/"
                "top_n_effort_20260213_step500_auc0.9962_eer0.0191.pth",
        "desc": "R3_FT4 (qr_moderate+QE, from F1)",
    },
    "F1": {
        "path": "gs://training-job-outputs/phase2r2_experiments/5w453our/"
                "top_n_effort_20260212_step18500_auc0.9893_eer0.0356.pth",
        "desc": "R25_F1 (baseline, no aug, no QE)",
    },
}

# ── Data sources ─────────────────────────────────────────────────────────────
DEEPLIVE_BUCKET = "live-deepfake-methods-real-and-fake-frames-cropped"
EXTERNAL_REAL_BUCKET = "effort-collected-data/real/external_youtube_avspeech"

OUTPUT_FOLDER = "gs://training-job-outputs/test_results/r3_FT_validation"
WANDB_PROJECT = "phase2-experiments"


def build_jobs(selected_checkpoints=None):
    """Build the list of (checkpoint × datasource) jobs.
    
    Args:
        selected_checkpoints: Optional list of checkpoint keys to include.
                              If None, all checkpoints are used.
    """
    jobs = []

    checkpoints = CHECKPOINTS
    if selected_checkpoints:
        checkpoints = {k: v for k, v in CHECKPOINTS.items() 
                       if k in selected_checkpoints}
        missing = set(selected_checkpoints) - set(CHECKPOINTS.keys())
        if missing:
            print(f"WARNING: Unknown checkpoint keys: {missing}")
            print(f"  Available: {list(CHECKPOINTS.keys())}")

    for ckpt_key, ckpt in checkpoints.items():
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
    parser = argparse.ArgumentParser(
        description="Sequential R3 Fine-Tune validation (up to 15 jobs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing.")
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Comma-separated list of checkpoint keys to run. "
                             "Default: all (FT1,FT2,FT3,FT4,F1). "
                             "Example: --checkpoints FT3,F1")
    args = parser.parse_args()

    selected = None
    if args.checkpoints:
        selected = [c.strip() for c in args.checkpoints.split(",")]

    jobs = build_jobs(selected_checkpoints=selected)

    ckpts_used = selected or list(CHECKPOINTS.keys())
    n_ckpts = len(ckpts_used)
    n_jobs = len(jobs)

    print("=" * 70)
    print("  R3 Fine-Tune Sequential Validation")
    print(f"  {n_ckpts} checkpoints × 3 data sources = {n_jobs} evaluations")
    print("=" * 70)
    print()
    print("Checkpoints:")
    for key in ckpts_used:
        ckpt = CHECKPOINTS.get(key)
        if ckpt:
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
