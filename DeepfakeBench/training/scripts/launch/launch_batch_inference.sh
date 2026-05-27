#!/usr/bin/env bash
# launch_batch_inference.sh — Launch batch inference on Vertex AI with A100
#
# Usage:
#   ./launch_batch_inference.sh [--checkpoint GS_URI] [--buckets BUCKET1 BUCKET2 ...] [--region REGION]
#
# Defaults run R9_A on all 3 test buckets.
#
# Examples:
#   # Run R9_A on all 3 test buckets (default):
#   ./launch_batch_inference.sh
#
#   # Run specific checkpoint on one bucket:
#   ./launch_batch_inference.sh \
#     --checkpoint gs://training-job-outputs/phase2r8_experiments/hu7cen3m/top_n_effort_20260224_step8000_auc0.9925_eer0.0207.pth \
#     --buckets poc-phase-1-test
#
#   # Custom run ID and region:
#   ./launch_batch_inference.sh --run-id R9A_final --region europe-west4

set -euo pipefail

# ==========================================================================
# Defaults
# ==========================================================================
CHECKPOINT="${CHECKPOINT:-gs://training-job-outputs/phase2r9_experiments/1551zxa8/top_n_effort_20260228_step6000_auc0.9891_eer0.0457.pth}"
BUCKETS=(
    "poc-phase-1-test"
    "teams-faces-data-test-2914-fake-4420-real-feb-28"
    "live-deepfake-methods-real-and-fake-frames-cropped-teams"
)
REGION="${REGION:-us-central1}"
RUN_ID=""
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PROJECT="${PROJECT:-train-cvit2}"
WANDB_API_KEY="${WANDB_API_KEY:-bb5a8ea4a27ebe45917587df8c46674d26e43966}"
WANDB_ENTITY="${WANDB_ENTITY:-dtect-vision}"

# Docker image
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VERSION="${VERSION:-$(cat "${TRAINING_DIR}/VERSION")}"
IMAGE_URI="us-docker.pkg.dev/${PROJECT}/effort-detector/effort-detector:${VERSION}"

GCS_OUTPUT_BUCKET="training-job-outputs"
GCS_OUTPUT_PREFIX="batch_inference_results"
AUTO_CONFIRM=false

# ==========================================================================
# Parse CLI args
# ==========================================================================
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)      CHECKPOINT="$2"; shift 2 ;;
        --checkpoint=*)    CHECKPOINT="${1#*=}"; shift ;;
        --buckets)         shift; BUCKETS=(); while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do BUCKETS+=("$1"); shift; done ;;
        --region)          REGION="$2"; shift 2 ;;
        --region=*)        REGION="${1#*=}"; shift ;;
        --run-id)          RUN_ID="$2"; shift 2 ;;
        --run-id=*)        RUN_ID="${1#*=}"; shift ;;
        --batch-size)      BATCH_SIZE="$2"; shift 2 ;;
        --batch-size=*)    BATCH_SIZE="${1#*=}"; shift ;;
        --num-workers)     NUM_WORKERS="$2"; shift 2 ;;
        --num-workers=*)   NUM_WORKERS="${1#*=}"; shift ;;
        --version)         VERSION="$2"; IMAGE_URI="us-docker.pkg.dev/${PROJECT}/effort-detector/effort-detector:${VERSION}"; shift 2 ;;
        --version=*)       VERSION="${1#*=}"; IMAGE_URI="us-docker.pkg.dev/${PROJECT}/effort-detector/effort-detector:${VERSION}"; shift ;;
        -y|--yes)          AUTO_CONFIRM=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--checkpoint GS_URI] [--buckets B1 B2 ...] [--region REGION] [--run-id ID] [--batch-size N] [--num-workers N] [--version V] [-y]"
            exit 0
            ;;
        *)                 echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ==========================================================================
# Build container args for batch_inference_gcs.py
# ==========================================================================
PY_ARGS=(
    "--checkpoint" "${CHECKPOINT}"
    "--buckets" "${BUCKETS[@]}"
    "--output_dir" "/workspace/inference_results"
    "--batch_size" "${BATCH_SIZE}"
    "--num_workers" "${NUM_WORKERS}"
    "--upload_results"
    "--gcs_output_bucket" "${GCS_OUTPUT_BUCKET}"
    "--gcs_output_prefix" "${GCS_OUTPUT_PREFIX}"
)

if [[ -n "$RUN_ID" ]]; then
    PY_ARGS+=("--run_id" "${RUN_ID}")
fi

# ==========================================================================
# Build the Vertex AI job name
# ==========================================================================
CKPT_BASENAME="$(basename "${CHECKPOINT}" .pth)"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
JOB_NAME="batch-infer-${TIMESTAMP}"

# Service account
SERVICE_ACCOUNT="vertex-job-runner-train-cvit2@${PROJECT}.iam.gserviceaccount.com"

# ==========================================================================
# Display configuration
# ==========================================================================
echo "=============================================="
echo "Launching Batch Inference on Vertex AI"
echo "=============================================="
echo "Project:       ${PROJECT}"
echo "Image:         ${IMAGE_URI}"
echo "Job Name:      ${JOB_NAME}"
echo "Region:        ${REGION}"
echo "Checkpoint:    ${CHECKPOINT}"
echo "Buckets:       ${BUCKETS[*]}"
echo "Batch Size:    ${BATCH_SIZE}"
echo "Workers:       ${NUM_WORKERS}"
echo "Output:        gs://${GCS_OUTPUT_BUCKET}/${GCS_OUTPUT_PREFIX}/"
if [[ -n "$RUN_ID" ]]; then
    echo "Run ID:        ${RUN_ID}"
fi
echo "=============================================="
echo ""

# Confirm
if [[ "$AUTO_CONFIRM" != true ]]; then
    read -p "Launch this inference job? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# ==========================================================================
# Build YAML from template and submit
# ==========================================================================
# We'll use the same template but override MODE and pass our args

# Build entrypoint args string for YAML
# Format: ["--mode", "batch_infer", "--", "--checkpoint", "gs://...", ...]
YAML_ARGS='["--mode", "batch_infer", "--"'
for arg in "${PY_ARGS[@]}"; do
    # Escape double quotes in arg
    escaped_arg="${arg//\\/\\\\}"
    escaped_arg="${escaped_arg//\"/\\\"}"
    YAML_ARGS+=", \"${escaped_arg}\""
done
YAML_ARGS+=']'

# Create a temporary job spec YAML
TMP_YAML="$(mktemp)"

cat > "${TMP_YAML}" << YAMLEOF
# Batch inference job spec (auto-generated)

baseOutputDirectory:
  outputUriPrefix: gs://${GCS_OUTPUT_BUCKET}/vertex-output/${JOB_NAME}

scheduling:
  timeout: 86400s
  restartJobOnWorkerRestart: false

workerPoolSpecs:
  - machineSpec:
      machineType: a2-highgpu-1g
      acceleratorType: "NVIDIA_TESLA_A100"
      acceleratorCount: 1
    replicaCount: 1
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: 200
    containerSpec:
      imageUri: "${IMAGE_URI}"
      command: ["/bin/bash", "/workspace/entrypoint.sh"]
      args: ${YAML_ARGS}
      env:
        - name: WANDB_API_KEY
          value: "${WANDB_API_KEY}"
        - name: WANDB_ENTITY
          value: "${WANDB_ENTITY}"
        - name: WANDB_PROJECT
          value: "batch-inference"
        - name: JOB_MODE
          value: "batch_infer"
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: TOKENIZERS_PARALLELISM
          value: "false"

serviceAccount: "${SERVICE_ACCOUNT}"
YAMLEOF

echo "Generated job spec: ${TMP_YAML}"
echo ""

# Submit to Vertex AI (try multiple regions if needed)
IFS=',' read -ra RLIST <<< "${REGION}"
SUBMITTED=false
for region in "${RLIST[@]}"; do
    echo "Submitting '${JOB_NAME}' to region ${region}..."
    if gcloud ai custom-jobs create \
        --project "${PROJECT}" \
        --region "${region}" \
        --display-name "${JOB_NAME}" \
        --config "${TMP_YAML}"; then
        SUBMITTED=true
        FINAL_REGION="${region}"
        break
    else
        echo "Region ${region} failed; trying next…"
    fi
done

rm -f "${TMP_YAML}"

if [[ "${SUBMITTED}" != true ]]; then
    echo "ERROR: Failed to submit job to any region."
    exit 1
fi

echo ""
echo "=============================================="
echo "Job submitted: ${JOB_NAME}"
echo "=============================================="
echo ""
echo "Monitor job:"
echo "  gcloud ai custom-jobs describe ${JOB_NAME} --region=${FINAL_REGION} --project=${PROJECT}"
echo ""
echo "Stream logs:"
echo "  gcloud ai custom-jobs stream-logs ${JOB_NAME} --region=${FINAL_REGION} --project=${PROJECT}"
echo ""
echo "Results will be at:"
echo "  gs://${GCS_OUTPUT_BUCKET}/${GCS_OUTPUT_PREFIX}/"
echo ""
