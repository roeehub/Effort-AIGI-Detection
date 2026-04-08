#!/bin/bash
# ============================================================================
# run_teams_target_domain_scorecard.sh
#
# One-command wrapper for the frozen Teams target-domain scorecard.
#
# Usage:
#   bash arena/run_teams_target_domain_scorecard.sh \
#     --checkpoint-map arena/checkpoint_maps/teams_target_domain.template.yaml \
#     --checkpoints R12_G_FP32,R12_G_INT8 \
#     --dry-run
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"

CHECKPOINT_MAP=""
CHECKPOINTS="ALL"
SUITE_MANIFEST="arena/target_domain_suites.teams_manifest.frozen_2026-04-06.yaml"
OUTPUT_GCS_FOLDER="gs://training-job-outputs/test_results/target_domain_validation"
SCORECARD_DIR="arena/scorecards"
WANDB_PROJECT="${WANDB_PROJECT:-phase2-experiments}"
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-map)
            CHECKPOINT_MAP="$2"; shift 2 ;;
        --checkpoints)
            CHECKPOINTS="$2"; shift 2 ;;
        --suite-manifest)
            SUITE_MANIFEST="$2"; shift 2 ;;
        --output-gcs-folder)
            OUTPUT_GCS_FOLDER="$2"; shift 2 ;;
        --scorecard-dir)
            SCORECARD_DIR="$2"; shift 2 ;;
        --wandb-project)
            WANDB_PROJECT="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN="--dry-run"; shift ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1 ;;
    esac
done

if [[ -z "${CHECKPOINT_MAP}" ]]; then
    echo "--checkpoint-map is required" >&2
    exit 1
fi

TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOCAL_SCORECARD_DIR="${TRAINING_DIR}/${SCORECARD_DIR}/${TIMESTAMP}"
mkdir -p "${LOCAL_SCORECARD_DIR}"

LONG_CSV="${LOCAL_SCORECARD_DIR}/scorecard.csv"
WIDE_CSV="${LOCAL_SCORECARD_DIR}/scorecard.wide.csv"
DELTA_CSV="${LOCAL_SCORECARD_DIR}/scorecard.int8_delta.csv"
JSON_OUT="${LOCAL_SCORECARD_DIR}/scorecard.json"

cd "${TRAINING_DIR}"

CMD=(
    python arena/run_target_domain_validation_sequential.py
    --checkpoints "${CHECKPOINTS}"
    --checkpoint_map "${CHECKPOINT_MAP}"
    --suite_manifest "${SUITE_MANIFEST}"
    --output_gcs_folder "${OUTPUT_GCS_FOLDER}"
    --wandb_project "${WANDB_PROJECT}"
    --scorecard_csv "${LONG_CSV}"
    --scorecard_wide_csv "${WIDE_CSV}"
    --scorecard_delta_csv "${DELTA_CSV}"
    --scorecard_json "${JSON_OUT}"
)

if [[ -n "${DRY_RUN}" ]]; then
    CMD+=("${DRY_RUN}")
fi

echo "============================================================"
echo "Teams Target-Domain Scorecard"
echo "============================================================"
echo "Checkpoint map:   ${CHECKPOINT_MAP}"
echo "Checkpoints:      ${CHECKPOINTS}"
echo "Suite manifest:   ${SUITE_MANIFEST}"
echo "Output GCS:       ${OUTPUT_GCS_FOLDER}"
echo "Local scorecards: ${LOCAL_SCORECARD_DIR}"
echo "W&B project:      ${WANDB_PROJECT}"
if [[ -n "${DRY_RUN}" ]]; then
    echo "Mode:             DRY RUN"
fi
echo "============================================================"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

if [[ -z "${DRY_RUN}" ]]; then
    echo "Scorecard artifacts written under:"
    echo "  ${LOCAL_SCORECARD_DIR}/"
fi
