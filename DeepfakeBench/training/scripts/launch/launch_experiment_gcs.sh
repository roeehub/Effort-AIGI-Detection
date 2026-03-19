#!/usr/bin/env bash
# launch_experiment_gcs.sh - Launch experiments with config from GCS (no rebuild needed!)
#
# Usage:
#   ./launch_experiment_gcs.sh <WANDB_PROJECT> <REGION> <GCS_CONFIG_PATH>
#
# Example:
#   ./launch_experiment_gcs.sh effort-dec2025 asia-southeast1 gs://experiment-configs/sanity_exp.yaml
#
# This script downloads the experiment config from GCS at runtime, so you can
# create new experiments without rebuilding the Docker image!
#
# To upload a config:
#   gsutil cp orgenize_training/sanity_exp.yaml gs://experiment-configs/sanity_exp.yaml

set -euo pipefail

# ==============================================
# USAGE
# ==============================================
usage() {
    cat << EOF
Usage: $0 <WANDB_PROJECT> <REGION> <GCS_CONFIG_PATH>

Arguments:
  WANDB_PROJECT    W&B project name (e.g., "effort-baseline-dec2025")
  REGION           GCP region (e.g., "asia-southeast1", "us-central1")
  GCS_CONFIG_PATH  GCS path to experiment config YAML
                   (e.g., "gs://experiment-configs/sanity_exp.yaml")

Example:
  $0 effort-dec2025 asia-southeast1 gs://experiment-configs/sanity_exp.yaml

To upload a config first:
  gsutil cp orgenize_training/my_exp.yaml gs://experiment-configs/my_exp.yaml

Environment variables (optional overrides):
PROJECT          GCP project (default: train-cvit2)
  VERSION          Docker image version (default: from VERSION file)
  GPU_TYPE         GPU type (default: NVIDIA_TESLA_A100)
  GPU_COUNT        Number of GPUs (default: 1)
  CONFIG_BUCKET    GCS bucket for configs (default: experiment-configs)

EOF
}

# Check for required arguments
if [ $# -lt 3 ]; then
    echo "Error: Missing required arguments"
    echo ""
    usage
    exit 1
fi

# ==============================================
# CONFIGURATION
# ==============================================

# Required arguments
WANDB_PROJECT="$1"
REGION="$2"
GCS_CONFIG_PATH="$3"

# Validate GCS path
if [[ ! "${GCS_CONFIG_PATH}" =~ ^gs:// ]]; then
    echo "Error: GCS_CONFIG_PATH must start with 'gs://'"
    echo "  Got: ${GCS_CONFIG_PATH}"
    echo ""
    usage
    exit 1
fi

# GCP Project (can be overridden via env var)
PROJECT="${PROJECT:-train-cvit2}"

# Weights & Biases (can be overridden via env vars)
WANDB_API_KEY="${WANDB_API_KEY:-bb5a8ea4a27ebe45917587df8c46674d26e43966}"
WANDB_ENTITY="${WANDB_ENTITY:-dtect-vision}"

# Docker image version (read from VERSION file if not set)
if [ -z "${VERSION:-}" ]; then
    VERSION="$(cat VERSION)"
fi
IMAGE_URI="${IMAGE_URI:-us-docker.pkg.dev/${PROJECT}/effort-detector/effort-detector:${VERSION}}"

# GPU configuration (can be overridden via env vars)
GPU_TYPE="${GPU_TYPE:-NVIDIA_TESLA_A100}"
GPU_COUNT="${GPU_COUNT:-1}"

# Job configuration
CONFIG_BASENAME=$(basename "${GCS_CONFIG_PATH}" .yaml)
JOB_NAME="exp-${CONFIG_BASENAME}-$(date +%Y%m%d-%H%M%S)"
MODE="train"

# ==============================================
# VERIFY CONFIG EXISTS
# ==============================================
echo "Verifying config exists at: ${GCS_CONFIG_PATH}"
if ! gsutil -q stat "${GCS_CONFIG_PATH}"; then
    echo "Error: Config not found at ${GCS_CONFIG_PATH}"
    echo ""
    echo "To upload a config:"
    echo "  gsutil cp your_config.yaml ${GCS_CONFIG_PATH}"
    exit 1
fi
echo "✅ Config found!"
echo ""

# ==============================================
# EXPORT ENVIRONMENT VARIABLES
# ==============================================
export PROJECT
export WANDB_API_KEY
export WANDB_ENTITY
export WANDB_PROJECT
export VERSION
export IMAGE_URI
export REGIONS="${REGION}"

# ==============================================
# DISPLAY CONFIGURATION
# ==============================================

echo "=============================================="
echo "Launching Training Experiment (GCS Config)"
echo "=============================================="
echo "Project:       ${PROJECT}"
echo "W&B Entity:    ${WANDB_ENTITY}"
echo "W&B Project:   ${WANDB_PROJECT}"
echo "Image:         ${IMAGE_URI}"
echo "Job Name:      ${JOB_NAME}"
echo "GCS Config:    ${GCS_CONFIG_PATH}"
echo "GPU:           ${GPU_TYPE} x ${GPU_COUNT}"
echo "Region:        ${REGION}"
echo "=============================================="
echo ""
echo "NOTE: Config is loaded from GCS at runtime - no rebuild needed!"
echo ""

# Confirm before launching
read -p "Launch this job? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Export GCS_CONFIG so it gets passed as an environment variable
export GCS_CONFIG="${GCS_CONFIG_PATH}"

# Launch the job (GCS_CONFIG will be injected as env var in template)
./launch_experiment_jobs.sh \
    --mode "${MODE}" \
    --job-name "${JOB_NAME}" \
    --project "${PROJECT}" \
    --regions "${REGION}" \
    --image-uri "${IMAGE_URI}" \
    --gpu-type "${GPU_TYPE}" \
    --gpu-count "${GPU_COUNT}" \
    --main-script "train_sweep.py"

echo ""
echo "=============================================="
echo "Job submitted: ${JOB_NAME}"
echo "=============================================="
echo ""
echo "Monitor job:"
echo "  gcloud ai custom-jobs describe ${JOB_NAME} --region=${REGION} --project=${PROJECT}"
echo ""
echo "View logs:"
echo "  gcloud ai custom-jobs stream-logs ${JOB_NAME} --region=${REGION} --project=${PROJECT}"
echo ""
echo "W&B dashboard:"
echo "  https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
echo ""
