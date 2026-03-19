#!/bin/bash
# =============================================================================
# R3 Fine-Tune Validation: FT1, FT2, FT3, FT4, F1 (single Vertex AI job)
# =============================================================================
# Submits ONE Vertex AI job that runs up to 15 evaluations sequentially:
#   5 checkpoints × 3 data sources
#
#   Checkpoints: R3_FT1, R3_FT2, R3_FT3, R3_FT4, R25_F1 (baseline)
#   Data:        DeepLive ALL, External Real, Quality Enhancement
#
# Same protocol as launch_r3_validation.sh (S1/S2/F1), now for FT models.
#
# Options:
#   --ft3-only     Run only FT3 + F1 baseline (6 evals, ~1 hour)
#   (default)      Run all 5 checkpoints (15 evals, ~2-3 hours)
#
# Output → gs://training-job-outputs/test_results/r3_FT_validation/
# =============================================================================

set -e

PROJECT="${PROJECT:-train-cvit2}"
REGION="us-central1"
CHECKPOINTS_ARG=""

# Parse our own args
for arg in "$@"; do
  case $arg in
    --ft3-only)
      CHECKPOINTS_ARG="FT3,F1"
      echo "Mode: FT3 + F1 baseline only (6 evaluations)"
      shift
      ;;
  esac
done

if [[ -z "$IMAGE_URI" ]]; then
  echo "ERROR: IMAGE_URI environment variable is required."
  echo "  Set it after building the Docker image, e.g.:"
  echo "  export IMAGE_URI=us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:<tag>"
  exit 1
fi

# Build the extra args to forward to the Python script
EXTRA_ARGS=""
if [[ -n "$CHECKPOINTS_ARG" ]]; then
  EXTRA_ARGS="--checkpoints ${CHECKPOINTS_ARG}"
  N_EVALS=6
  DESC="FT3 + F1 baseline"
else
  N_EVALS=15
  DESC="FT1, FT2, FT3, FT4, F1"
fi

echo "=========================================="
echo "  R3 Fine-Tune Validation"
echo "=========================================="
echo "  Image:       $IMAGE_URI"
echo "  Project:     $PROJECT"
echo "  Region:      $REGION"
echo "  Checkpoints: $DESC"
echo ""
echo "  ${N_EVALS} evaluations (checkpoints × 3 data sources)"
echo "  Estimated time: ~$((N_EVALS * 12)) minutes"
echo ""

TS=$(date +%Y%m%d-%H%M%S)

if [[ -n "$EXTRA_ARGS" ]]; then
  ./launch_experiment_jobs.sh \
    --mode train \
    --job-name validate-r3ft-${TS} \
    --project "${PROJECT}" \
    --regions "${REGION}" \
    --image-uri "${IMAGE_URI}" \
    --gpu-type NVIDIA_TESLA_A100 \
    --gpu-count 1 \
    --main-script run_r3ft_validation_sequential.py \
    -- ${EXTRA_ARGS}
else
  ./launch_experiment_jobs.sh \
    --mode train \
    --job-name validate-r3ft-${TS} \
    --project "${PROJECT}" \
    --regions "${REGION}" \
    --image-uri "${IMAGE_URI}" \
    --gpu-type NVIDIA_TESLA_A100 \
    --gpu-count 1 \
    --main-script run_r3ft_validation_sequential.py
fi

echo ""
echo "✅ Submitted job with ${N_EVALS} sequential evaluations."
echo ""
echo "Output: gs://training-job-outputs/test_results/r3_FT_validation/"
echo "W&B:    https://wandb.ai/dtect-vision/phase2-experiments"
echo ""
echo "Expected output files:"
if [[ -n "$CHECKPOINTS_ARG" ]]; then
  echo "  FT3_deeplive_*   FT3_extreal_*   FT3_qualenhance_*"
  echo "  F1_deeplive_*    F1_extreal_*     F1_qualenhance_*"
else
  echo "  FT1_deeplive_*   FT1_extreal_*   FT1_qualenhance_*"
  echo "  FT2_deeplive_*   FT2_extreal_*   FT2_qualenhance_*"
  echo "  FT3_deeplive_*   FT3_extreal_*   FT3_qualenhance_*"
  echo "  FT4_deeplive_*   FT4_extreal_*   FT4_qualenhance_*"
  echo "  F1_deeplive_*    F1_extreal_*     F1_qualenhance_*"
fi
