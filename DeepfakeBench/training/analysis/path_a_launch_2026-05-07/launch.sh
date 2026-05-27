#!/usr/bin/env bash
# PATH_A_LAUNCH_2026-05-07 — submit the combined Phase 0h + 0j paired-feature
# extraction job to Vertex AI.
#
# Defaults to us-east1 (per CLAUDE.md region rules).
#
# Usage:
#   ./launch.sh [--region REGION] [-y]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PROJECT="${PROJECT:-train-cvit2}"
REGION="${REGION:-us-east1}"
# NOTE: VERSION file may point to an unbuilt tag (auto-bumped ahead of build).
# Default to 1.3.267 (latest built as of 2026-05-06 at job submission time);
# override with --version <tag> if a newer build is available.
VERSION="${VERSION:-1.3.267}"
IMAGE_URI="${IMAGE_URI:-us-docker.pkg.dev/${PROJECT}/effort-detector/effort-detector:${VERSION}}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-vertex-job-runner-train-cvit2@${PROJECT}.iam.gserviceaccount.com}"

# Inputs already uploaded to GCS by the parent agent.
SCRIPT_GCS="gs://training-job-outputs/path_a_inputs_2026-05-07/extract_paired_features.py"
PAIR_GAPS_CSV_GCS="gs://training-job-outputs/path_a_inputs_2026-05-07/pair_gaps.csv"
OUTPUT_PREFIX="gs://training-job-outputs/analysis_outputs/frozen_pair_features_2026-05-07/"

# Target checkpoints (per memory project_p8a_breakthrough.md and
# project_deployment_is_e2b_2026-05-06.md).
P8A_CKPT="${P8A_CKPT:-gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth}"
E2B_CKPT="${E2B_CKPT:-gs://training-job-outputs/best_checkpoints/rmat8lwx/top_n_effort_20260503_step3200_auc0.9863_eer0.0310.pth}"

INCLUDE_CLIP_RAW="${INCLUDE_CLIP_RAW:-true}"   # raw OpenCLIP B16 baseline
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AUTO_CONFIRM="${AUTO_CONFIRM:-false}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)             REGION="$2"; shift 2 ;;
    --version)            VERSION="$2";
                          IMAGE_URI="us-docker.pkg.dev/${PROJECT}/effort-detector/effort-detector:${VERSION}";
                          shift 2 ;;
    --batch-size)         BATCH_SIZE="$2"; shift 2 ;;
    --num-workers)        NUM_WORKERS="$2"; shift 2 ;;
    --no-clip-raw)        INCLUDE_CLIP_RAW="false"; shift ;;
    -y|--yes)             AUTO_CONFIRM=true; shift ;;
    -h|--help)
      echo "Usage: $0 [--region REGION] [--version V] [--batch-size N] [--num-workers N] [--no-clip-raw] [-y]"
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ "${INCLUDE_CLIP_RAW}" == "true" ]]; then
  INCLUDE_CLIP_RAW_FLAG="--include_clip_raw"
else
  INCLUDE_CLIP_RAW_FLAG=""
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
JOB_NAME="path-a-extract-${TIMESTAMP}"

echo "=============================================="
echo "PATH_A_LAUNCH_2026-05-07"
echo "=============================================="
echo "Project:        ${PROJECT}"
echo "Region:         ${REGION}"
echo "Image:          ${IMAGE_URI}"
echo "Job name:       ${JOB_NAME}"
echo "Service acct:   ${SERVICE_ACCOUNT}"
echo "Output prefix:  ${OUTPUT_PREFIX}"
echo "P8A ckpt:       ${P8A_CKPT}"
echo "E2B ckpt:       ${E2B_CKPT}"
echo "CLIP raw:       ${INCLUDE_CLIP_RAW}"
echo "Batch size:     ${BATCH_SIZE}"
echo "Num workers:    ${NUM_WORKERS}"
echo "Pair gaps CSV:  ${PAIR_GAPS_CSV_GCS}"
echo "Script GCS:     ${SCRIPT_GCS}"
echo "=============================================="

if [[ "$AUTO_CONFIRM" != true ]]; then
  read -p "Submit? (y/N) " -n 1 -r; echo
  [[ ! $REPLY =~ ^[Yy]$ ]] && { echo "Aborted."; exit 0; }
fi

# Render YAML from template via envsubst.
TMP_YAML="$(mktemp --suffix=.yaml 2>/dev/null || mktemp -t path_a_yaml.XXXXXX)"
export JOB_NAME IMAGE_URI SCRIPT_GCS PAIR_GAPS_CSV_GCS OUTPUT_PREFIX \
       P8A_CKPT E2B_CKPT INCLUDE_CLIP_RAW_FLAG BATCH_SIZE NUM_WORKERS \
       SERVICE_ACCOUNT
envsubst < "${SCRIPT_DIR}/launch_yaml/extraction_job.yaml.template" > "${TMP_YAML}"

echo "Rendered job spec at: ${TMP_YAML}"
echo "----- BEGIN SPEC -----"
cat "${TMP_YAML}"
echo "----- END SPEC -----"
echo ""

echo "Submitting to ${REGION}..."
# NOTE: do NOT capture into a subshell-substitution + `set -e` will swallow
# the error code on submit failure. Run gcloud directly so stderr surfaces.
set +e
gcloud ai custom-jobs create \
    --project "${PROJECT}" \
    --region "${REGION}" \
    --display-name "${JOB_NAME}" \
    --config "${TMP_YAML}" 2>&1 | tee /tmp/path_a_submit_out.txt
SUBMIT_RC="${PIPESTATUS[0]}"
set -e
SUBMIT_OUT="$(cat /tmp/path_a_submit_out.txt)"
if [[ "${SUBMIT_RC}" -ne 0 ]]; then
  echo "ERROR: gcloud submit failed with exit code ${SUBMIT_RC}"
  exit "${SUBMIT_RC}"
fi

# Persist rendered spec alongside outputs (useful for postmortem).
cp "${TMP_YAML}" "${SCRIPT_DIR}/launch_yaml/extraction_job.${TIMESTAMP}.rendered.yaml"
rm -f "${TMP_YAML}"

JOB_RESOURCE="$(echo "${SUBMIT_OUT}" | grep -oE 'projects/[0-9]+/locations/[a-z0-9-]+/customJobs/[0-9]+' | head -1)"
JOB_ID="$(echo "${JOB_RESOURCE}" | awk -F/ '{print $NF}')"

echo ""
echo "=============================================="
echo "Submitted: ${JOB_NAME}"
echo "Resource:  ${JOB_RESOURCE}"
echo "Job ID:    ${JOB_ID}"
echo "Region:    ${REGION}"
echo "=============================================="
echo ""
echo "Monitor:"
echo "  gcloud ai custom-jobs describe ${JOB_ID} --region=${REGION} --project=${PROJECT}"
echo ""
echo "Stream logs:"
echo "  gcloud ai custom-jobs stream-logs ${JOB_ID} --region=${REGION} --project=${PROJECT}"
echo ""
echo "Outputs will land at:"
echo "  ${OUTPUT_PREFIX}p8a_paired_features.npz"
echo "  ${OUTPUT_PREFIX}e2b_paired_features.npz"
if [[ "${INCLUDE_CLIP_RAW}" == "true" ]]; then
  echo "  ${OUTPUT_PREFIX}clip_b16_raw_paired_features.npz"
fi
echo "  ${OUTPUT_PREFIX}frame_manifest.csv"
