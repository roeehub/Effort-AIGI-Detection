#!/usr/bin/env bash
# launch_retro_score.sh — Launch a single Vertex AI retro-score job that runs
# retro_score_value_composite.py against one P3 checkpoint with one yaml.
#
# This is a retro-score cousin of launch_experiment.sh: it piggybacks on
# launch_experiment_jobs.sh + the standard Vertex template, but swaps the
# --main-script from train_sweep.py to retro_score_value_composite.py and
# forwards retro-specific CLI args to the container.
#
# Usage:
#   ./launch_retro_score.sh [-y] <WANDB_PROJECT> <REGION> <CONFIG_YAML> <CHECKPOINT_GCS_PATH> <WANDB_RUN_NAME>
#
# Example:
#   ./launch_retro_score.sh -y enhanced-aug-test us-central1 \
#     experiments/phase2_round13/R13_RLP3_02_FT_proper_main__NEW_GATES_FOR_RETRO.yaml \
#     gs://training-job-outputs/phase2r13_experiments/zifvogm6/value_composite_effort_20260422_step4500_auc0.9889_eer0.0463.pth \
#     R13_RLP3_02_main_NEW_0422

set -euo pipefail

usage() {
    cat << EOF
Usage:
  $0 [-y] <WANDB_PROJECT> <REGION> <CONFIG_YAML> <CHECKPOINT_GCS_PATH> <WANDB_RUN_NAME>

Arguments:
  WANDB_PROJECT         W&B project name (e.g., "enhanced-aug-test")
  REGION                GCP region (e.g., "us-central1")
  CONFIG_YAML           Path to retro-score yaml, relative to repo root
                        (e.g., "experiments/phase2_round13/..__NEW_GATES_FOR_RETRO.yaml")
  CHECKPOINT_GCS_PATH   gs:// URL to checkpoint (.pth) to retro-score
  WANDB_RUN_NAME        Suffix for the W&B run name (script prefixes "retro_")

Options:
  -y, --yes             Auto-confirm (skip confirmation prompt)

Env overrides:
  PROJECT         GCP project (default: train-cvit2)
  WANDB_ENTITY    W&B entity (default: dtect-vision)
  VERSION         Docker image version (default: from VERSION file)
  GPU_TYPE        (default: NVIDIA_TESLA_A100)
  GPU_COUNT       (default: 1)
EOF
}

AUTO_CONFIRM=false
POSITIONAL_ARGS=()
for arg in "$@"; do
    case $arg in
        -y|--yes) AUTO_CONFIRM=true ;;
        -h|--help) usage; exit 0 ;;
        *) POSITIONAL_ARGS+=("$arg") ;;
    esac
done
if [ ${#POSITIONAL_ARGS[@]} -gt 0 ]; then
    set -- "${POSITIONAL_ARGS[@]}"
else
    set --
fi

if [ $# -ne 5 ]; then
    echo "Error: Expected 5 positional arguments, got $#"
    echo ""
    usage
    exit 1
fi

WANDB_PROJECT="$1"
REGION="$2"
CONFIG_YAML_INPUT="$3"
CHECKPOINT_GCS_PATH="$4"
WANDB_RUN_NAME="$5"

PROJECT="${PROJECT:-train-cvit2}"
WANDB_API_KEY="${WANDB_API_KEY:-bb5a8ea4a27ebe45917587df8c46674d26e43966}"
WANDB_ENTITY="${WANDB_ENTITY:-dtect-vision}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [ -z "${VERSION:-}" ]; then
    VERSION="$(cat "${TRAINING_DIR}/VERSION")"
fi
IMAGE_URI="us-docker.pkg.dev/${PROJECT}/effort-detector/effort-detector:${VERSION}"

GPU_TYPE="${GPU_TYPE:-NVIDIA_TESLA_A100}"
GPU_COUNT="${GPU_COUNT:-1}"

# Container-side path for the yaml (launch_experiment pattern)
if [[ "${CONFIG_YAML_INPUT}" = /* ]]; then
    CONFIG_YAML="${CONFIG_YAML_INPUT}"
else
    CONFIG_YAML="/workspace/${CONFIG_YAML_INPUT}"
fi

JOB_NAME="retro-$(basename ${CONFIG_YAML_INPUT} .yaml)-$(date +%Y%m%d-%H%M%S)"

export PROJECT
export WANDB_API_KEY
export WANDB_ENTITY
export WANDB_PROJECT
export VERSION
export IMAGE_URI
export REGIONS="${REGION}"

echo "=============================================="
echo "Launching RETRO-SCORE Job"
echo "=============================================="
echo "Project:       ${PROJECT}"
echo "W&B Entity:    ${WANDB_ENTITY}"
echo "W&B Project:   ${WANDB_PROJECT}"
echo "W&B Run Name:  retro_${WANDB_RUN_NAME}"
echo "Image:         ${IMAGE_URI}"
echo "Job Name:      ${JOB_NAME}"
echo "Config YAML:   ${CONFIG_YAML}"
echo "Checkpoint:    ${CHECKPOINT_GCS_PATH}"
echo "GPU:           ${GPU_TYPE} x ${GPU_COUNT}"
echo "Region:        ${REGION}"
echo "=============================================="

if [[ "$AUTO_CONFIRM" == true ]]; then
    echo "Auto-confirming (-y flag)"
else
    read -p "Launch this retro-score job? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

"${SCRIPT_DIR}/launch_experiment_jobs.sh" \
    --mode "train" \
    --job-name "${JOB_NAME}" \
    --project "${PROJECT}" \
    --regions "${REGION}" \
    --image-uri "${IMAGE_URI}" \
    --gpu-type "${GPU_TYPE}" \
    --gpu-count "${GPU_COUNT}" \
    --main-script "retro_score_value_composite.py" \
    -- \
        --checkpoint_gcs_path "${CHECKPOINT_GCS_PATH}" \
        --config_yaml "${CONFIG_YAML}" \
        --wandb_run_name "${WANDB_RUN_NAME}" \
        --wandb_project "${WANDB_PROJECT}"

echo ""
echo "=============================================="
echo "Job submitted: ${JOB_NAME}"
echo "=============================================="
echo ""
echo "Monitor job:"
echo "  gcloud ai custom-jobs list --region=${REGION} --project=${PROJECT} --filter=\"displayName=${JOB_NAME}\" --format=\"value(name,state)\""
echo ""
echo "Stream logs (once it runs):"
echo "  gcloud ai custom-jobs stream-logs <JOB_ID> --region=${REGION} --project=${PROJECT}"
echo ""
echo "W&B dashboard:"
echo "  https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
