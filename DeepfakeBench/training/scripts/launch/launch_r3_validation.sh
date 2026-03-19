#!/bin/bash
# =============================================================================
# R3 Validation: R3_S1 vs R3_S2 vs R25_F1 (single Vertex AI job)
# =============================================================================
# Submits ONE Vertex AI job that runs 9 evaluations sequentially:
#   3 checkpoints × 3 data sources
#
#   Checkpoints: R3_S1, R3_S2, R25_F1
#   Data:        DeepLive ALL, External Real, Quality Enhancement
#
# Each evaluation runs validate_custom_sources.py as a fresh subprocess,
# so there's no state leakage between runs.
#
# Output → gs://training-job-outputs/test_results/r3_S1_vs_S2_vs_F1_validation/
# =============================================================================

set -e

PROJECT="${PROJECT:-train-cvit2}"
REGION="us-central1"

if [[ -z "$IMAGE_URI" ]]; then
  echo "ERROR: IMAGE_URI environment variable is required."
  echo "  Set it after building the Docker image, e.g.:"
  echo "  export IMAGE_URI=us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:<tag>"
  exit 1
fi

echo "=========================================="
echo "  R3 Validation (single sequential job)"
echo "=========================================="
echo "  Image:   $IMAGE_URI"
echo "  Project: $PROJECT"
echo "  Region:  $REGION"
echo ""
echo "  9 evaluations (3 checkpoints × 3 data sources)"
echo "  Estimated time: ~2-3 hours total"
echo ""

TS=$(date +%Y%m%d-%H%M%S)

./launch_experiment_jobs.sh \
  --mode train \
  --job-name validate-r3-sequential-${TS} \
  --project "${PROJECT}" \
  --regions "${REGION}" \
  --image-uri "${IMAGE_URI}" \
  --gpu-type NVIDIA_TESLA_A100 \
  --gpu-count 1 \
  --main-script run_r3_validation_sequential.py

echo ""
echo "✅ Submitted single job with 9 sequential evaluations."
echo ""
echo "Output: gs://training-job-outputs/test_results/r3_S1_vs_S2_vs_F1_validation/"
echo "W&B:    https://wandb.ai/dtect-vision/phase2-experiments"
echo ""
echo "Expected output files:"
echo "  S1_deeplive_*     S1_extreal_*     S1_qualenhance_*"
echo "  S2_deeplive_*     S2_extreal_*     S2_qualenhance_*"
echo "  F1_deeplive_*     F1_extreal_*     F1_qualenhance_*"
