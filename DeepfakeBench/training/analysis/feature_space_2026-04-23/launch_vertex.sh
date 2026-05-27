#!/usr/bin/env bash
# Launch feature-space extraction on Vertex AI with A100.
#
# Usage:
#   ./launch_vertex.sh [--checkpoint GS_URI] [--output-prefix GS_URI] [--region REGION] [--n N] [-y]
#
# Defaults: run against packet-5 slot 07 checkpoint (new leader candidate, 0.7736),
# output to gs://training-job-outputs/feature_space_analysis/<timestamp>/, us-central1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PROJECT="${PROJECT:-train-cvit2}"
REGION="${REGION:-us-central1}"
VERSION="${VERSION:-$(cat "${TRAINING_DIR}/VERSION")}"
IMAGE_URI="us-docker.pkg.dev/${PROJECT}/effort-detector/effort-detector:${VERSION}"
SERVICE_ACCOUNT="vertex-job-runner-train-cvit2@${PROJECT}.iam.gserviceaccount.com"
WANDB_API_KEY="${WANDB_API_KEY:-bb5a8ea4a27ebe45917587df8c46674d26e43966}"
WANDB_ENTITY="${WANDB_ENTITY:-dtect-vision}"

# Slot-07 run id = 6jwwb526 (best_value_composite=0.7736 @ step 20500).
# Default to the exact value_composite checkpoint at the winning step.
CHECKPOINT="${CHECKPOINT:-gs://training-job-outputs/phase2r13_experiments/6jwwb526/value_composite_effort_20260423_step20500_auc0.9908_eer0.0304.pth}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_TAG="${RUN_TAG:-rlp5_07_${TIMESTAMP}}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-gs://training-job-outputs/feature_space_analysis/${RUN_TAG}/}"
N_PER_SOURCE="${N_PER_SOURCE:-150}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AUTO_CONFIRM=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)      CHECKPOINT="$2"; shift 2 ;;
    --output-prefix)   OUTPUT_PREFIX="$2"; shift 2 ;;
    --region)          REGION="$2"; shift 2 ;;
    --n)               N_PER_SOURCE="$2"; shift 2 ;;
    --batch-size)      BATCH_SIZE="$2"; shift 2 ;;
    --num-workers)     NUM_WORKERS="$2"; shift 2 ;;
    --version)         VERSION="$2";
                       IMAGE_URI="us-docker.pkg.dev/${PROJECT}/effort-detector/effort-detector:${VERSION}";
                       shift 2 ;;
    -y|--yes)          AUTO_CONFIRM=true; shift ;;
    -h|--help)         echo "Usage: $0 [--checkpoint GS_URI] [--output-prefix GS_URI] [--region REGION] [--n N] [--batch-size N] [--num-workers N] [--version V] [-y]"; exit 0 ;;
    *)                 echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ "${OUTPUT_PREFIX}" != */ ]]; then OUTPUT_PREFIX="${OUTPUT_PREFIX}/"; fi

JOB_NAME="feature-space-${TIMESTAMP}"
ARGS_ARR=("--mode" "feature_space" "--" \
          "--checkpoint" "${CHECKPOINT}" \
          "--output_prefix" "${OUTPUT_PREFIX}" \
          "--n_per_source" "${N_PER_SOURCE}" \
          "--batch_size" "${BATCH_SIZE}" \
          "--num_workers" "${NUM_WORKERS}")
YAML_ARGS='['
for i in "${!ARGS_ARR[@]}"; do
    [[ $i -gt 0 ]] && YAML_ARGS+=', '
    escaped="${ARGS_ARR[$i]//\\/\\\\}"
    escaped="${escaped//\"/\\\"}"
    YAML_ARGS+="\"${escaped}\""
done
YAML_ARGS+=']'

echo "=============================================="
echo "Feature-space extraction on Vertex AI"
echo "=============================================="
echo "Project:       ${PROJECT}"
echo "Image:         ${IMAGE_URI}"
echo "Region:        ${REGION}"
echo "Job name:      ${JOB_NAME}"
echo "Checkpoint:    ${CHECKPOINT}"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo "N per source:  ${N_PER_SOURCE}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Num workers:   ${NUM_WORKERS}"
echo "=============================================="

if [[ "$AUTO_CONFIRM" != true ]]; then
  read -p "Launch? (y/N) " -n 1 -r; echo
  [[ ! $REPLY =~ ^[Yy]$ ]] && { echo "Aborted."; exit 0; }
fi

TMP_YAML="$(mktemp)"
cat > "${TMP_YAML}" << YAMLEOF
baseOutputDirectory:
  outputUriPrefix: gs://training-job-outputs/vertex-output/${JOB_NAME}

scheduling:
  timeout: 7200s
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
        - name: JOB_MODE
          value: "feature_space"
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: TOKENIZERS_PARALLELISM
          value: "false"

serviceAccount: "${SERVICE_ACCOUNT}"
YAMLEOF

echo "Generated job spec: ${TMP_YAML}"
cat "${TMP_YAML}"
echo ""

IFS=',' read -ra RLIST <<< "${REGION}"
SUBMITTED=false
for region in "${RLIST[@]}"; do
  echo "Submitting '${JOB_NAME}' to region ${region}…"
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
echo "Job submitted: ${JOB_NAME} (region=${FINAL_REGION})"
echo "=============================================="
echo ""
echo "Stream logs:"
echo "  gcloud ai custom-jobs stream-logs --region=${FINAL_REGION} --project=${PROJECT} \$(gcloud ai custom-jobs list --region=${FINAL_REGION} --filter=\"displayName=${JOB_NAME}\" --format='value(name)' --limit=1)"
echo ""
echo "Results will be at:"
echo "  ${OUTPUT_PREFIX}"
echo ""
