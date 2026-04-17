#!/bin/bash
# ============================================================================
# launch_r13_best_megaval.sh
#
# Launch a mixed Teams + source-regression evaluation on Vertex AI.
# This wraps the sequential validation runner with:
# - compact Teams target-domain slices
# - full DeepLive source coverage
# - full VisoMaster source coverage
# - the fake-only VisoMaster enhanced v2 bucket
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"

PROJECT="${PROJECT:-train-cvit2}"
IMAGE_URI="us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:$(cat "${TRAINING_DIR}/VERSION")"
REGION="${REGION:-asia-southeast1}"
GPU_TYPE="${GPU_TYPE:-NVIDIA_TESLA_A100}"
GPU_COUNT="${GPU_COUNT:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-phase2-experiments}"

SUITE_MANIFEST="arena/target_domain_suites.r13_best_megaval_2026-04-13.yaml"
CHECKPOINT_MAP="arena/checkpoint_maps/r13_best_eval_2026-04-13.yaml"
CHECKPOINTS="R12_G_FP32,R13_A_STEP15500,R13_FT7_FP32,R13_FT8_FP32,R13_FT9_FP32,R13_FT10_FP32,R13_E_BESTSOFAR"
OUTPUT_GCS_ROOT="gs://training-job-outputs/test_results/r13_best_megaval"
JOB_NAME=""
DRY_RUN=""

usage() {
    cat <<EOF
Usage:
  $(basename "$0") [options]

Options:
  --suite-manifest PATH       Repo-relative or /workspace path to the suite manifest baked into the image.
  --checkpoint-map PATH       Repo-relative or /workspace path to the checkpoint map baked into the image.
  --checkpoints CSV           Comma-separated checkpoint aliases (default: ${CHECKPOINTS})
  --output-gcs-root URI       GCS root for reports + scorecards (default: ${OUTPUT_GCS_ROOT})
  --wandb-project NAME        W&B project name (default: ${WANDB_PROJECT})
  --job-name NAME             Override Vertex display name / output suffix.
  --region REGION             Vertex region (default: ${REGION})
  --image-uri URI             Container image URI (default: versioned effort-detector image)
  --gpu-type TYPE             Accelerator type passed to the shared launcher.
  --gpu-count N               Accelerator count passed to the shared launcher.
  --dry-run                   Print the launch plan without submitting a Vertex job.
EOF
}

normalize_container_local_path() {
    local path="$1"
    if [[ "$path" == gs://* ]]; then
        echo "Suite/checkpoint map paths must be local inside the image, not gs:// paths: $path" >&2
        exit 2
    fi
    if [[ "$path" == /workspace/* ]]; then
        printf '%s\n' "$path"
        return 0
    fi
    if [[ "$path" == /* ]]; then
        printf '%s\n' "$path"
        return 0
    fi
    path="${path#./}"
    printf '/workspace/%s\n' "$path"
}

print_cmd() {
    printf 'Command:'
    printf ' %q' "$@"
    printf '\n'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite-manifest)
            SUITE_MANIFEST="$2"; shift 2 ;;
        --checkpoint-map)
            CHECKPOINT_MAP="$2"; shift 2 ;;
        --checkpoints)
            CHECKPOINTS="$2"; shift 2 ;;
        --output-gcs-root)
            OUTPUT_GCS_ROOT="$2"; shift 2 ;;
        --wandb-project)
            WANDB_PROJECT="$2"; shift 2 ;;
        --job-name)
            JOB_NAME="$2"; shift 2 ;;
        --region)
            REGION="$2"; shift 2 ;;
        --image-uri)
            IMAGE_URI="$2"; shift 2 ;;
        --gpu-type)
            GPU_TYPE="$2"; shift 2 ;;
        --gpu-count)
            GPU_COUNT="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN="1"; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 2 ;;
    esac
done

SUITE_MANIFEST="$(normalize_container_local_path "${SUITE_MANIFEST}")"
CHECKPOINT_MAP="$(normalize_container_local_path "${CHECKPOINT_MAP}")"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
if [[ -z "${JOB_NAME}" ]]; then
    JOB_NAME="r13-best-megaval-${TIMESTAMP}"
fi

RUN_ROOT="${OUTPUT_GCS_ROOT%/}/${JOB_NAME}"
REPORTS_GCS_FOLDER="${RUN_ROOT}/reports"
SCORECARD_GCS_DIR="${RUN_ROOT}/scorecards"

RUNNER_ARGS=(
    --checkpoints "${CHECKPOINTS}"
    --checkpoint_map "${CHECKPOINT_MAP}"
    --suite_manifest "${SUITE_MANIFEST}"
    --output_gcs_folder "${REPORTS_GCS_FOLDER}"
    --wandb_project "${WANDB_PROJECT}"
    --scorecard_csv "${SCORECARD_GCS_DIR}/scorecard.csv"
    --scorecard_wide_csv "${SCORECARD_GCS_DIR}/scorecard.wide.csv"
    --scorecard_delta_csv "${SCORECARD_GCS_DIR}/scorecard.int8_delta.csv"
    --scorecard_json "${SCORECARD_GCS_DIR}/scorecard.json"
)

LAUNCH_CMD=(
    "${TRAINING_DIR}/scripts/launch/launch_experiment_jobs.sh"
    --mode train
    --job-name "${JOB_NAME}"
    --project "${PROJECT}"
    --regions "${REGION}"
    --image-uri "${IMAGE_URI}"
    --gpu-type "${GPU_TYPE}"
    --gpu-count "${GPU_COUNT}"
    --main-script arena/run_target_domain_validation_sequential.py
    --
    "${RUNNER_ARGS[@]}"
)

echo "============================================================"
echo "R13 Best Mega Eval — Vertex AI Launch"
echo "============================================================"
echo "Image:              ${IMAGE_URI}"
echo "Project:            ${PROJECT}"
echo "Region:             ${REGION}"
echo "Job Name:           ${JOB_NAME}"
echo "Suite Manifest:     ${SUITE_MANIFEST}"
echo "Checkpoint Map:     ${CHECKPOINT_MAP}"
echo "Checkpoints:        ${CHECKPOINTS}"
echo "W&B Project:        ${WANDB_PROJECT}"
echo "Vertex Output:      gs://training-job-outputs/vertex-output/${JOB_NAME}/"
echo "Detailed Reports:   ${REPORTS_GCS_FOLDER}/"
echo "Scorecards:         ${SCORECARD_GCS_DIR}/"
echo "============================================================"
print_cmd "${LAUNCH_CMD[@]}"

if [[ -n "${DRY_RUN}" ]]; then
    echo "DRY RUN — no Vertex job submitted."
    exit 0
fi

"${LAUNCH_CMD[@]}"

echo ""
echo "============================================================"
echo "Mega eval job submitted!"
echo "Vertex system output:"
echo "  gs://training-job-outputs/vertex-output/${JOB_NAME}/"
echo "Detailed validation reports:"
echo "  ${REPORTS_GCS_FOLDER}/"
echo "Scorecard artifacts:"
echo "  ${SCORECARD_GCS_DIR}/scorecard.csv"
echo "  ${SCORECARD_GCS_DIR}/scorecard.wide.csv"
echo "  ${SCORECARD_GCS_DIR}/scorecard.int8_delta.csv"
echo "  ${SCORECARD_GCS_DIR}/scorecard.json"
echo "============================================================"
