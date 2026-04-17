#!/usr/bin/env bash
#
# Launch DeepLive + VisoMaster validation for the current R13 finalists.
#
# This is the broad source-regression lane, not the Teams target-domain scorecard.
# It is meant to answer:
# - which FT candidate preserves or improves source-family behavior vs R12_G
# - whether the best scratch follow-up is still worth considering
#
# Usage:
#   cd DeepfakeBench/training
#   ./scripts/launch/launch_r13_finalist_source_validation.sh
#   ./scripts/launch/launch_r13_finalist_source_validation.sh --checkpoints R12_G_FP32,R13_FT7_FP32,R13_FT8_FP32
#   ./scripts/launch/launch_r13_finalist_source_validation.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PROJECT="${PROJECT:-train-cvit2}"
REGION="${REGION:-asia-southeast1}"
WANDB_PROJECT="${WANDB_PROJECT:-phase2-experiments}"
IMAGE_URI="${IMAGE_URI:-us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:$(cat "${TRAINING_DIR}/VERSION")}"

DEEPLIVE_BUCKET="live-deepfake-methods-real-and-fake-frames-cropped"
VISOMASTER_CROPPED_BUCKET="live-deepfake-methods-real-and-fake-frames-cropped"
VISOMASTER_FRAMES_BUCKET="live-deepfake-methods-real-and-fake-frames"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-gs://training-job-outputs/test_results/r13_finalist_source_validation/${TIMESTAMP}}"

ALL_CHECKPOINTS=(
  "R12_G_FP32"
  "R13_FT7_FP32"
  "R13_FT8_FP32"
  "R13_FT9_FP32"
  "R13_FT10_FP32"
  "R13_E_BESTSOFAR"
)

checkpoint_path_for() {
  case "$1" in
    R12_G_FP32)
      printf '%s\n' "gs://training-job-outputs/phase2r12_experiments/0xxqwhxg/top_n_effort_20260310_step14000_auc0.9930_eer0.0253.pth"
      ;;
    R13_FT7_FP32)
      printf '%s\n' "gs://training-job-outputs/phase2r13_experiments/w4n9ejic/ood_composite_effort_20260411_step6000_auc0.9901_eer0.0194.pth"
      ;;
    R13_FT8_FP32)
      printf '%s\n' "gs://training-job-outputs/phase2r13_experiments/fpdcvzhf/ood_composite_effort_20260411_step6000_auc0.9901_eer0.0194.pth"
      ;;
    R13_FT9_FP32)
      printf '%s\n' "gs://training-job-outputs/phase2r13_experiments/irzf5ymv/ood_composite_effort_20260412_step8000_auc0.9906_eer0.0194.pth"
      ;;
    R13_FT10_FP32)
      printf '%s\n' "gs://training-job-outputs/phase2r13_experiments/ctcz09ko/ood_composite_effort_20260412_step8000_auc0.9906_eer0.0194.pth"
      ;;
    R13_E_BESTSOFAR)
      printf '%s\n' "gs://training-job-outputs/phase2r13_experiments/14d5exx0/ood_composite_effort_20260412_step12000_auc0.9844_eer0.0369.pth"
      ;;
    *)
      echo "Unknown checkpoint alias: $1" >&2
      return 1
      ;;
  esac
}

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [options]

Options:
  --checkpoints CSV   Comma-separated aliases to launch.
                      Default: ${ALL_CHECKPOINTS[*]}
  --output-folder URI Override the shared GCS output folder.
  --dry-run           Print the launch plan without submitting jobs.

Known aliases:
  ${ALL_CHECKPOINTS[*]}
EOF
}

SELECTED_CHECKPOINTS=("${ALL_CHECKPOINTS[@]}")
DRY_RUN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoints)
      IFS=',' read -r -a SELECTED_CHECKPOINTS <<< "$2"
      shift 2
      ;;
    --output-folder)
      OUTPUT_FOLDER="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

echo "============================================================"
echo "R13 Finalist Source Validation"
echo "============================================================"
echo "Image:         ${IMAGE_URI}"
echo "Project:       ${PROJECT}"
echo "Region:        ${REGION}"
echo "W&B Project:   ${WANDB_PROJECT}"
echo "Output Folder: ${OUTPUT_FOLDER}"
echo "DeepLive:      ${DEEPLIVE_BUCKET} (split=all)"
echo "VisoMaster:    ${VISOMASTER_CROPPED_BUCKET}"
echo "Aliases:       ${SELECTED_CHECKPOINTS[*]}"
if [[ -n "${DRY_RUN}" ]]; then
  echo "Mode:          DRY RUN"
fi
echo "============================================================"

for alias in "${SELECTED_CHECKPOINTS[@]}"; do
  alias="${alias// /}"
  [[ -n "${alias}" ]] || continue

  checkpoint_path="$(checkpoint_path_for "${alias}")"
  alias_lower="$(printf '%s' "${alias}" | tr '[:upper:]' '[:lower:]')"

  CMD=(
    "${TRAINING_DIR}/scripts/launch/launch_experiment_jobs.sh"
    --mode train
    --job-name "r13-sourceval-${alias_lower}-${TIMESTAMP}"
    --project "${PROJECT}"
    --regions "${REGION}"
    --image-uri "${IMAGE_URI}"
    --gpu-type NVIDIA_TESLA_A100
    --gpu-count 1
    --main-script validate_custom_sources.py
    --
    --checkpoint_gcs_path "${checkpoint_path}"
    --df40_mode none
    --deeplive_bucket "${DEEPLIVE_BUCKET}"
    --deeplive_split all
    --visomaster_bucket "${VISOMASTER_CROPPED_BUCKET}"
    --visomaster_frames_bucket "${VISOMASTER_FRAMES_BUCKET}"
    --visomaster_include_tier_methods
    --frames_per_video 8
    --detailed_reports
    --output_gcs_folder "${OUTPUT_FOLDER}"
    --output_filename_prefix "${alias_lower}_"
    --log_prefix "${alias_lower}_deeplive_visomaster"
    --run_name "${alias} DeepLive+VisoMaster full evaluation"
    --wandb_project "${WANDB_PROJECT}"
  )

  printf 'Command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'

  if [[ -z "${DRY_RUN}" ]]; then
    "${CMD[@]}"
    sleep 2
  fi
done

if [[ -z "${DRY_RUN}" ]]; then
  echo ""
  echo "Submitted source-validation jobs."
  echo "Results will land under:"
  echo "  ${OUTPUT_FOLDER}/"
fi
