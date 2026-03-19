#!/usr/bin/env bash
# launch_experiment.sh - Generic script to launch training experiments on GCP Vertex AI
#
# Usage:
#   ./launch_experiment.sh <WANDB_PROJECT> <REGION> <PARAM_CONFIG>
#
# Example:
#   ./launch_experiment.sh effort-baseline-dec2025 asia-southeast1 orgenize_training/sanity_exp.yaml

set -euo pipefail

# ==============================================
# USAGE
# ==============================================
usage() {
    cat << EOF
Usage: $0 [-y] <WANDB_PROJECT> <REGION> <PARAM_CONFIG>

Arguments:
  WANDB_PROJECT    W&B project name (e.g., "effort-baseline-dec2025")
  REGION           GCP region (e.g., "asia-southeast1", "us-central1")
  PARAM_CONFIG     Path to experiment config YAML (relative to repo root)
                   (e.g., "orgenize_training/sanity_exp.yaml")

Options:
  -y, --yes        Auto-confirm (skip confirmation prompt)

Example:
  $0 effort-baseline-dec2025 asia-southeast1 orgenize_training/sanity_exp.yaml
  $0 -y effort-baseline-dec2025 asia-southeast1 orgenize_training/sanity_exp.yaml

Environment variables (optional overrides):
  PROJECT          GCP project (default: train-cvit2)
  WANDB_API_KEY    W&B API key (default: from hardcoded value)
  WANDB_ENTITY     W&B entity (default: dtect-vision)
  VERSION          Docker image version (default: from VERSION file)
  GPU_TYPE         GPU type (default: NVIDIA_TESLA_A100)
  GPU_COUNT        Number of GPUs (default: 1)

EOF
}

# ==============================================
# CONFIGURATION
# ==============================================

# Check for -y flag (auto-confirm)
AUTO_CONFIRM=false
POSITIONAL_ARGS=()
for arg in "$@"; do
    case $arg in
        -y|--yes)
            AUTO_CONFIRM=true
            ;;
        *)
            POSITIONAL_ARGS+=("$arg")
            ;;
    esac
done

# Restore positional args
set -- "${POSITIONAL_ARGS[@]}"

# Check for required arguments (after flag parsing)
if [ $# -lt 3 ]; then
    echo "Error: Missing required arguments"
    echo ""
    usage
    exit 1
fi

# Required arguments
WANDB_PROJECT="$1"
REGION="$2"
PARAM_CONFIG_INPUT="$3"

# GCP Project (can be overridden via env var)
PROJECT="${PROJECT:-train-cvit2}"

# Weights & Biases (can be overridden via env vars)
WANDB_API_KEY="${WANDB_API_KEY:-bb5a8ea4a27ebe45917587df8c46674d26e43966}"
WANDB_ENTITY="${WANDB_ENTITY:-dtect-vision}"

# Docker image version (read from VERSION file if not set)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${VERSION:-}" ]; then
    VERSION="$(cat "${SCRIPT_DIR}/VERSION")"
fi
# Always rebuild IMAGE_URI from VERSION to avoid stale env vars
IMAGE_URI="us-docker.pkg.dev/${PROJECT}/effort-detector/effort-detector:${VERSION}"

# GPU configuration (can be overridden via env vars)
GPU_TYPE="${GPU_TYPE:-NVIDIA_TESLA_A100}"
GPU_COUNT="${GPU_COUNT:-1}"

# Job configuration
MODE="train"  # single training run (not sweep)
JOB_NAME="exp-$(basename ${PARAM_CONFIG_INPUT} .yaml)-$(date +%Y%m%d-%H%M%S)"

# Convert param config to container path (add /workspace prefix if not absolute)
if [[ "${PARAM_CONFIG_INPUT}" = /* ]]; then
    PARAM_CONFIG="${PARAM_CONFIG_INPUT}"
else
    PARAM_CONFIG="/workspace/${PARAM_CONFIG_INPUT}"
fi

# ==============================================
# EXPORT ENVIRONMENT VARIABLES
# ==============================================
export PROJECT
export WANDB_API_KEY
export WANDB_ENTITY
export WANDB_PROJECT
export VERSION
export IMAGE_URI
export REGIONS="${REGION}"  # Single region for this job

# ==============================================
# DISPLAY CONFIGURATION
# ==============================================

echo "=============================================="
echo "Launching Training Experiment"
echo "=============================================="
echo "Project:       ${PROJECT}"
echo "W&B Entity:    ${WANDB_ENTITY}"
echo "W&B Project:   ${WANDB_PROJECT}"
echo "Image:         ${IMAGE_URI}"
echo "Job Name:      ${JOB_NAME}"
echo "Config:        ${PARAM_CONFIG}"
echo "GPU:           ${GPU_TYPE} x ${GPU_COUNT}"
echo "Region:        ${REGION}"
echo "=============================================="
echo ""

# Confirm before launching (skip if -y flag)
if [[ "$AUTO_CONFIRM" == true ]]; then
    echo "Auto-confirming (-y flag)"
else
    read -p "Launch this job? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Launch the job
./launch_experiment_jobs.sh \
    --mode "${MODE}" \
    --job-name "${JOB_NAME}" \
    --project "${PROJECT}" \
    --regions "${REGION}" \
    --image-uri "${IMAGE_URI}" \
    --gpu-type "${GPU_TYPE}" \
    --gpu-count "${GPU_COUNT}" \
    --main-script "train_sweep.py" \
    -- --param-config "${PARAM_CONFIG}"

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
