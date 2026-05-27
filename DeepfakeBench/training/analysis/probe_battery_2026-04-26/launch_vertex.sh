#!/usr/bin/env bash
# Launch probe-battery feature extraction on Vertex AI with A100.
#
# Pairs with extract_features_for_probes.py (per-frame features + source_bucket
# labels) and run_linear_probes.py (sklearn LogisticRegression locally).
#
# Usage:
#   ./launch_vertex.sh --checkpoint GS_URI --run-tag NAME [--region REGION] [--n N] [-y]
#
# Defaults to us-central1; override with --region us-east1 / us-west4 / us-multi (CSV).
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

CHECKPOINT="${CHECKPOINT:-}"
RUN_TAG="${RUN_TAG:-}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
N_PER_SOURCE="${N_PER_SOURCE:-150}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AUTO_CONFIRM=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)      CHECKPOINT="$2"; shift 2 ;;
    --run-tag)         RUN_TAG="$2"; shift 2 ;;
    --region)          REGION="$2"; shift 2 ;;
    --n)               N_PER_SOURCE="$2"; shift 2 ;;
    --batch-size)      BATCH_SIZE="$2"; shift 2 ;;
    --num-workers)     NUM_WORKERS="$2"; shift 2 ;;
    --version)         VERSION="$2";
                       IMAGE_URI="us-docker.pkg.dev/${PROJECT}/effort-detector/effort-detector:${VERSION}";
                       shift 2 ;;
    -y|--yes)          AUTO_CONFIRM=true; shift ;;
    -h|--help)
      echo "Usage: $0 --checkpoint GS_URI --run-tag NAME [--region REGION] [--n N] [--batch-size N] [--num-workers N] [--version V] [-y]";
      exit 0 ;;
    *)                 echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "${CHECKPOINT}" || -z "${RUN_TAG}" ]]; then
  echo "ERROR: --checkpoint and --run-tag are required."; exit 1
fi

OUTPUT_PREFIX="gs://training-job-outputs/probe_battery_2026-04-26/${RUN_TAG}/"
JOB_NAME="probe-features-${RUN_TAG}-${TIMESTAMP}"

ARGS_ARR=("--mode" "probe_features" "--" \
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
echo "Probe-battery feature extraction on Vertex AI"
echo "=============================================="
echo "Project:       ${PROJECT}"
echo "Image:         ${IMAGE_URI}"
echo "Region:        ${REGION}"
echo "Job name:      ${JOB_NAME}"
echo "Run tag:       ${RUN_TAG}"
echo "Checkpoint:    ${CHECKPOINT}"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo "N per source:  ${N_PER_SOURCE}"
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
          value: "probe_features"
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
echo "Results:"
echo "  ${OUTPUT_PREFIX}features.npz"
echo "  ${OUTPUT_PREFIX}sampling_manifest.json"
echo ""
echo "Then run probes locally:"
echo "  python3 analysis/probe_battery_2026-04-26/run_linear_probes.py \\"
echo "    --features_npz ${OUTPUT_PREFIX}features.npz=${RUN_TAG} \\"
echo "    --label_field source_bucket --threshold 0.25 \\"
echo "    --output_dir analysis/probe_battery_2026-04-26/source_probe_run_$(date +%Y-%m-%d)/"
