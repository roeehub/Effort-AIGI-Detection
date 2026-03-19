#!/bin/bash
# Launch overnight checkpoint + threshold sweep for R6.
#
# This runs:
#   run_r6_checkpoint_threshold_sweep.py
# which:
#   1) validates all discovered top_n checkpoints for a run
#   2) sweeps thresholds
#   3) picks best checkpoint + threshold with an FP-first objective
#
# Usage:
#   export IMAGE_URI=us-docker.pkg.dev/<project>/<repo>/<image>:<tag>
#   ./launch_r6_checkpoint_threshold_sweep.sh <WANDB_PROJECT> <REGION> <RUN_ID> [--smoke]
#
# Example:
#   ./launch_r6_checkpoint_threshold_sweep.sh phase2-round6 asia-southeast1 xqkbkfgg

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <WANDB_PROJECT> <REGION> <RUN_ID> [--smoke]"
  echo "Example (full):  $0 phase2-round6 asia-southeast1 xqkbkfgg"
  echo "Example (smoke): $0 phase2-round6 asia-southeast1 xqkbkfgg --smoke"
  exit 1
fi

WANDB_PROJECT="$1"
REGION="$2"
RUN_ID="$3"
SMOKE_MODE=0
if [[ $# -eq 4 ]]; then
  if [[ "$4" == "--smoke" ]]; then
    SMOKE_MODE=1
  else
    echo "Unknown fourth argument: $4 (expected --smoke)"
    exit 1
  fi
fi

PROJECT="${PROJECT:-train-cvit2}"
SUITE_MANIFEST="experiments/phase2_round6/R6_THRESHOLD_SWEEP_SUITES.yaml"
CKPT_PREFIX="gs://training-job-outputs/phase2r6_experiments/${RUN_ID}"
MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-3}"
OUTPUT_SUFFIX=""
SMOKE_ARG=""

if [[ "$SMOKE_MODE" -eq 1 ]]; then
  MAX_CHECKPOINTS="${MAX_CHECKPOINTS_SMOKE:-1}"
  OUTPUT_SUFFIX="_smoke"
  SMOKE_ARG="--smoke"
fi

OUTPUT_FOLDER="gs://training-job-outputs/test_results/r6_ckpt_threshold_sweep_${RUN_ID}${OUTPUT_SUFFIX}_$(date +%Y%m%d-%H%M%S)"

if [[ -z "${IMAGE_URI:-}" ]]; then
  echo "ERROR: IMAGE_URI is required."
  echo "Set it first, e.g.:"
  echo "  export IMAGE_URI=us-docker.pkg.dev/${PROJECT}/effort-detector/effort-detector:<tag>"
  exit 1
fi

echo "=== R6 Checkpoint + Threshold Sweep ==="
echo "Project        : ${PROJECT}"
echo "Region         : ${REGION}"
echo "Run ID         : ${RUN_ID}"
echo "Mode           : $([[ "$SMOKE_MODE" -eq 1 ]] && echo smoke || echo full)"
echo "Checkpoint dir : ${CKPT_PREFIX}"
echo "Suite manifest : ${SUITE_MANIFEST}"
echo "Max checkpoints: ${MAX_CHECKPOINTS}"
echo "Output folder  : ${OUTPUT_FOLDER}"
echo "W&B project    : ${WANDB_PROJECT}"
echo "Image          : ${IMAGE_URI}"
echo ""

./launch_experiment_jobs.sh \
  --mode train \
  --job-name "r6-threshold-sweep-${RUN_ID}-$(date +%Y%m%d-%H%M%S)" \
  --project "${PROJECT}" \
  --regions "${REGION}" \
  --image-uri "${IMAGE_URI}" \
  --gpu-type NVIDIA_TESLA_A100 \
  --gpu-count 1 \
  --main-script run_r6_checkpoint_threshold_sweep.py \
  -- --checkpoint_prefix_gcs "${CKPT_PREFIX}" \
     --suite_manifest "${SUITE_MANIFEST}" \
     --output_gcs_folder "${OUTPUT_FOLDER}" \
     --wandb_project "${WANDB_PROJECT}" \
     --max_checkpoints "${MAX_CHECKPOINTS}" \
     --threshold_min 0.05 \
     --threshold_max 0.99 \
     --threshold_step 0.002 \
     --min_macro_fake_tpr 0.85 \
     --min_worst_fake_tpr 0.65 \
     --fallback_lambda 2.0 \
     ${SMOKE_ARG:+$SMOKE_ARG}

echo ""
echo "Job submitted."
echo "Validation + analysis artifacts will land under:"
echo "  ${OUTPUT_FOLDER}"
echo "Final winner summary:"
echo "  ${OUTPUT_FOLDER}/analysis/winner_summary.json"
