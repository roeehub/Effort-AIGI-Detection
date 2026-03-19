#!/usr/bin/env bash
# launch_deeplive.sh - Launch DeepLive training on Vertex AI with GPU
#
# Usage:
#   ./launch_deeplive.sh <experiment_name> [region]
#
# Examples:
#   ./launch_deeplive.sh deeplive_vit_B16
#   ./launch_deeplive.sh deeplive_vit_L14 us-central1
#   ./launch_deeplive.sh deeplive_vit_B32 asia-southeast1
#
# Available experiments:
#   - deeplive_vit_B16      (ViT-B-16 OpenAI CLIP)
#   - deeplive_vit_L14      (ViT-L-14 OpenAI CLIP)
#   - deeplive_vit_B32      (ViT-B-32 OpenAI CLIP)
#   - deeplive_vit_B16_laion (ViT-B-16 LAION DataComp)

set -euo pipefail

# ==============================================
# USAGE
# ==============================================
usage() {
    cat << EOF
Usage: $0 <experiment_name> [region]

Arguments:
  experiment_name   Name of experiment config (without .yaml extension)
                    Available: deeplive_vit_B16, deeplive_vit_L14, 
                               deeplive_vit_B32, deeplive_vit_B16_laion
  region            GCP region (default: us-central1)
                    Options: us-central1, europe-west4, asia-southeast1

Examples:
  $0 deeplive_vit_B16
  $0 deeplive_vit_L14 asia-southeast1

Environment variables (optional overrides):
  PROJECT           GCP project (default: train-cvit2)
  GPU_TYPE          GPU type (default: NVIDIA_TESLA_A100)
  GPU_COUNT         Number of GPUs (default: 1)
  WANDB_PROJECT     W&B project name (default: deeplive-experiments)

EOF
}

# Check for required arguments
if [ $# -lt 1 ]; then
    echo "Error: Missing experiment name"
    echo ""
    usage
    exit 1
fi

# ==============================================
# CONFIGURATION
# ==============================================

EXPERIMENT="$1"
REGION="${2:-asia-southeast1}"

# Validate experiment config exists
CONFIG_PATH="experiments/${EXPERIMENT}.yaml"
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Error: Experiment config not found: ${CONFIG_PATH}"
    echo ""
    echo "Available experiments:"
    ls -1 experiments/deeplive_*.yaml 2>/dev/null | sed 's|experiments/||; s|\.yaml||; s|^|  - |'
    exit 1
fi

# GCP Project
PROJECT="${PROJECT:-train-cvit2}"

# Weights & Biases
WANDB_API_KEY="${WANDB_API_KEY:-bb5a8ea4a27ebe45917587df8c46674d26e43966}"
WANDB_ENTITY="${WANDB_ENTITY:-dtect-vision}"
WANDB_PROJECT="${WANDB_PROJECT:-deeplive-experiments}"

# Docker image version
VERSION="$(cat VERSION)"
IMAGE_URI="us-docker.pkg.dev/${PROJECT}/effort-detector/effort-detector:${VERSION}"

# GPU configuration
GPU_TYPE="${GPU_TYPE:-NVIDIA_TESLA_A100}"
GPU_COUNT="${GPU_COUNT:-1}"

# Job name
JOB_NAME="${EXPERIMENT}-$(date +%Y%m%d-%H%M%S)"

# ==============================================
# DISPLAY CONFIGURATION
# ==============================================

echo "=============================================="
echo "Launching DeepLive Training on Vertex AI"
echo "=============================================="
echo "Experiment:    ${EXPERIMENT}"
echo "Config:        ${CONFIG_PATH}"
echo "Project:       ${PROJECT}"
echo "Region:        ${REGION}"
echo "Image:         ${IMAGE_URI}"
echo "GPU:           ${GPU_TYPE} x ${GPU_COUNT}"
echo "W&B Project:   ${WANDB_PROJECT}"
echo "Job Name:      ${JOB_NAME}"
echo "=============================================="
echo ""

# Show config summary
echo "--- Experiment Config Summary ---"
grep -E "^name:|^backbone:|^  name:|^  variant:|^  hidden_size:|^nEpochs:" "${CONFIG_PATH}" | head -10
echo "---"
echo ""

# Confirm before launching
read -p "Launch this job? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# ==============================================
# UPLOAD CONFIG TO GCS
# ==============================================
GCS_CONFIG_PATH="gs://experiment-configs/deeplive/${EXPERIMENT}.yaml"
echo ""
echo "Uploading config to GCS: ${GCS_CONFIG_PATH}"
gsutil cp "${CONFIG_PATH}" "${GCS_CONFIG_PATH}"

# ==============================================
# LAUNCH JOB
# ==============================================
echo ""
echo "Launching Vertex AI job..."

# Export environment variables for the job
export PROJECT
export WANDB_API_KEY
export WANDB_ENTITY
export WANDB_PROJECT
export VERSION
export IMAGE_URI
export GCS_CONFIG="${GCS_CONFIG_PATH}"

# Launch using the existing job launcher
# Note: GCS_CONFIG env var triggers entrypoint to download config and add --param-config
# We only pass --wandb flags here, not --config (entrypoint handles that)
./launch_experiment_jobs.sh \
    --mode train \
    --job-name "${JOB_NAME}" \
    --project "${PROJECT}" \
    --regions "${REGION}" \
    --image-uri "${IMAGE_URI}" \
    --gpu-type "${GPU_TYPE}" \
    --gpu-count "${GPU_COUNT}" \
    --main-script "train_deeplive.py" \
    -- --wandb --wandb-project "${WANDB_PROJECT}"

echo ""
echo "=============================================="
echo "Job submitted: ${JOB_NAME}"
echo "=============================================="
echo ""
echo "Monitor job:"
echo "  gcloud ai custom-jobs describe ${JOB_NAME} --region=${REGION} --project=${PROJECT}"
echo ""
echo "Stream logs:"
echo "  gcloud ai custom-jobs stream-logs ${JOB_NAME} --region=${REGION} --project=${PROJECT}"
echo ""
echo "W&B dashboard:"
echo "  https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
echo ""
