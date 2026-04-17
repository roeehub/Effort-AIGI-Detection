#!/bin/bash
# ============================================================================
# run_teams_promotion_contract.sh
#
# One-command wrapper for the authoritative calibrated Teams promotion contract.
#
# This runs the frozen promotion-authoritative suite set, writes the usual
# fixed-threshold scorecard artifacts as diagnostic sidecars, and then writes
# the calibrated promotion-contract artifacts that should decide promotion.
#
# Usage:
#   bash arena/run_teams_promotion_contract.sh --dry-run
#   bash arena/run_teams_promotion_contract.sh --checkpoints ALL
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        PYTHON_BIN="python"
    fi
fi

CHECKPOINT_MAP="arena/checkpoint_maps/teams_target_domain.promotion_shortlist_2026-04-17.yaml"
CHECKPOINTS="ALL"
SUITE_MANIFEST="arena/target_domain_suites.teams_promotion_contract_2026-04-17.yaml"
OUTPUT_GCS_ROOT="gs://training-job-outputs/test_results/teams_promotion_contract"
ARTIFACT_DIR="arena/promotion_contracts"
WANDB_PROJECT="${WANDB_PROJECT:-phase2-experiments}"
RUN_NAME=""
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-map)
            CHECKPOINT_MAP="$2"; shift 2 ;;
        --checkpoints)
            CHECKPOINTS="$2"; shift 2 ;;
        --suite-manifest)
            SUITE_MANIFEST="$2"; shift 2 ;;
        --output-gcs-root)
            OUTPUT_GCS_ROOT="$2"; shift 2 ;;
        --artifact-dir)
            ARTIFACT_DIR="$2"; shift 2 ;;
        --wandb-project)
            WANDB_PROJECT="$2"; shift 2 ;;
        --run-name)
            RUN_NAME="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN="--dry-run"; shift ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1 ;;
    esac
done

TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
if [[ -z "${RUN_NAME}" ]]; then
    RUN_NAME="teams-promotion-contract-${TIMESTAMP}"
fi

LOCAL_RUN_DIR="${TRAINING_DIR}/${ARTIFACT_DIR}/${TIMESTAMP}"
DIAGNOSTIC_DIR="${LOCAL_RUN_DIR}/diagnostic_scorecard"
CONTRACT_DIR="${LOCAL_RUN_DIR}/promotion_contract"
REPORTS_GCS_FOLDER="${OUTPUT_GCS_ROOT%/}/${RUN_NAME}/reports"

mkdir -p "${DIAGNOSTIC_DIR}" "${CONTRACT_DIR}"

cd "${TRAINING_DIR}"

CMD=(
    "${PYTHON_BIN}" arena/run_target_domain_validation_sequential.py
    --checkpoints "${CHECKPOINTS}"
    --checkpoint_map "${CHECKPOINT_MAP}"
    --suite_manifest "${SUITE_MANIFEST}"
    --output_gcs_folder "${REPORTS_GCS_FOLDER}"
    --wandb_project "${WANDB_PROJECT}"
    --scorecard_csv "${DIAGNOSTIC_DIR}/scorecard.csv"
    --scorecard_wide_csv "${DIAGNOSTIC_DIR}/scorecard.wide.csv"
    --scorecard_delta_csv "${DIAGNOSTIC_DIR}/scorecard.int8_delta.csv"
    --scorecard_json "${DIAGNOSTIC_DIR}/scorecard.json"
    --promotion_contract_dir "${CONTRACT_DIR}"
)

if [[ -n "${DRY_RUN}" ]]; then
    CMD+=("${DRY_RUN}")
fi

echo "============================================================"
echo "Teams Promotion Contract"
echo "============================================================"
echo "Checkpoint map:          ${CHECKPOINT_MAP}"
echo "Checkpoints:             ${CHECKPOINTS}"
echo "Suite manifest:          ${SUITE_MANIFEST}"
echo "Reports GCS:             ${REPORTS_GCS_FOLDER}"
echo "Local diagnostic scores: ${DIAGNOSTIC_DIR}"
echo "Local contract outputs:  ${CONTRACT_DIR}"
echo "W&B project:             ${WANDB_PROJECT}"
echo "Promotion authority:     calibrated contract"
echo "Diagnostic-only sidecar: fixed-threshold 0.5 scorecard"
if [[ -n "${DRY_RUN}" ]]; then
    echo "Mode:                    DRY RUN"
fi
echo "============================================================"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

if [[ -z "${DRY_RUN}" ]]; then
    echo "Artifacts written under:"
    echo "  ${LOCAL_RUN_DIR}/"
fi
