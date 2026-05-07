#!/bin/bash
# ============================================================================
# launch_teams_promotion_contract.sh
# Launch the authoritative calibrated Teams promotion contract on Vertex AI.
#
# This runs the promotion-authoritative suite manifest, writes the usual
# fixed-threshold scorecards as diagnostic sidecars, and emits the calibrated
# promotion-contract artifacts that should decide promotion.
#
# Usage:
#   ./arena/launch_teams_promotion_contract.sh
#   ./arena/launch_teams_promotion_contract.sh --checkpoints ALL --dry-run
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

SUITE_MANIFEST="arena/target_domain_suites.teams_promotion_contract_2026-04-17.yaml"
CHECKPOINT_MAP="arena/checkpoint_maps/teams_target_domain.promotion_shortlist_2026-04-17.yaml"
CHECKPOINTS="ALL"
OUTPUT_GCS_ROOT="gs://training-job-outputs/test_results/teams_promotion_contract"
JOB_NAME=""
DRY_RUN=""

# Default to the 500GB-disk scorecard template — the 200GB default exhausted
# during PD's run (per pd_scorecard_artifacts_2026-05-06/README §Caveats and
# NEXT_STEPS_PLAN_2026-05-06.md §12). Override with --yaml-template if needed.
YAML_TEMPLATE="${TRAINING_DIR}/infra/cloudbuild/vertex_job_template_scorecard.yaml"

# Promotion-contract policy budgets passed through to the runner.
# Defaults match the v3-fix design (recall floor + bumped FPR budgets); they
# close the `contract-policy-bug-fix-not-committed` open loop's "scorecard
# run with --promotion_target_fake_recall_min ... selected τ via the
# recall-floor path" component. See:
#   docs/packet_retrospectives/threads/contract_policy_bug.md
#   docs/packet_retrospectives/threads/promotion_contract_evolution.md
# Pass --promotion-target-fake-recall-min 0.0 explicitly for legacy-policy
# comparison runs.
PROMOTION_TARGET_REAL_FPR="${PROMOTION_TARGET_REAL_FPR:-0.07}"
PROMOTION_TARGET_STRESS_FPR="${PROMOTION_TARGET_STRESS_FPR:-0.10}"
PROMOTION_TARGET_FAKE_RECALL_MIN="${PROMOTION_TARGET_FAKE_RECALL_MIN:-0.30}"

usage() {
    cat <<EOF
Usage:
  $(basename "$0") [options]

Options:
  --suite-manifest PATH       Repo-relative or /workspace path to the baked-in suite manifest.
  --checkpoint-map PATH       Repo-relative or /workspace path to the baked-in checkpoint map.
  --checkpoints CSV           Comma-separated checkpoint aliases (default: ${CHECKPOINTS})
  --output-gcs-root URI       GCS root for reports + artifacts (default: ${OUTPUT_GCS_ROOT})
  --wandb-project NAME        W&B project name (default: ${WANDB_PROJECT})
  --job-name NAME             Override Vertex display name / output suffix.
  --region REGION             Vertex region (default: ${REGION})
  --image-uri URI             Container image URI (default: versioned effort-detector image)
  --gpu-type TYPE             Accelerator type passed to the shared launcher.
  --gpu-count N               Accelerator count passed to the shared launcher.
  --promotion-target-real-fpr FLOAT
                              FPR budget on primary real dev suite for τ selection
                              (default: ${PROMOTION_TARGET_REAL_FPR}; v3-fix design).
  --promotion-target-stress-fpr FLOAT
                              FPR budget on worst real stress dev suite for τ selection
                              (default: ${PROMOTION_TARGET_STRESS_FPR}; v3-fix design).
  --promotion-target-fake-recall-min FLOAT
                              Recall floor on dev_fake_macro_recall. Candidates below
                              the floor rank in a worse tier (within-ckpt + cross-ckpt).
                              (default: ${PROMOTION_TARGET_FAKE_RECALL_MIN}; v3-fix design.
                              Pass 0.0 explicitly for legacy-policy comparison.)
  --yaml-template PATH        Vertex job template (default: 500GB-disk scorecard template
                              at infra/cloudbuild/vertex_job_template_scorecard.yaml).
  --dry-run                   Print the launch plan without submitting a Vertex job.

Artifacts:
  - reports:
      <output-gcs-root>/<job-name>/reports/
  - diagnostic fixed-threshold scorecards:
      <output-gcs-root>/<job-name>/diagnostic_scorecard/
  - calibrated promotion-contract outputs:
      <output-gcs-root>/<job-name>/promotion_contract/
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
        --promotion-target-real-fpr)
            PROMOTION_TARGET_REAL_FPR="$2"; shift 2 ;;
        --promotion-target-stress-fpr)
            PROMOTION_TARGET_STRESS_FPR="$2"; shift 2 ;;
        --promotion-target-fake-recall-min)
            PROMOTION_TARGET_FAKE_RECALL_MIN="$2"; shift 2 ;;
        --yaml-template)
            YAML_TEMPLATE="$2"; shift 2 ;;
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

# Image-currency pre-launch check (PCP #2 from plan-v2 LOG): verify both
# the suite manifest and checkpoint map are older than the image push time.
"${TRAINING_DIR}/scripts/launch/check_image_currency.sh" "${IMAGE_URI}" "${SUITE_MANIFEST}" "${CHECKPOINT_MAP}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
if [[ -z "${JOB_NAME}" ]]; then
    JOB_NAME="teams-promotion-contract-${TIMESTAMP}"
fi

RUN_ROOT="${OUTPUT_GCS_ROOT%/}/${JOB_NAME}"
REPORTS_GCS_FOLDER="${RUN_ROOT}/reports"
DIAGNOSTIC_GCS_DIR="${RUN_ROOT}/diagnostic_scorecard"
CONTRACT_GCS_DIR="${RUN_ROOT}/promotion_contract"

RUNNER_ARGS=(
    --checkpoints "${CHECKPOINTS}"
    --checkpoint_map "${CHECKPOINT_MAP}"
    --suite_manifest "${SUITE_MANIFEST}"
    --output_gcs_folder "${REPORTS_GCS_FOLDER}"
    --wandb_project "${WANDB_PROJECT}"
    --scorecard_csv "${DIAGNOSTIC_GCS_DIR}/scorecard.csv"
    --scorecard_wide_csv "${DIAGNOSTIC_GCS_DIR}/scorecard.wide.csv"
    --scorecard_delta_csv "${DIAGNOSTIC_GCS_DIR}/scorecard.int8_delta.csv"
    --scorecard_json "${DIAGNOSTIC_GCS_DIR}/scorecard.json"
    --promotion_contract_dir "${CONTRACT_GCS_DIR}"
    --promotion_target_real_fpr "${PROMOTION_TARGET_REAL_FPR}"
    --promotion_target_stress_fpr "${PROMOTION_TARGET_STRESS_FPR}"
    --promotion_target_fake_recall_min "${PROMOTION_TARGET_FAKE_RECALL_MIN}"
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
    --yaml-template "${YAML_TEMPLATE}"
    --main-script arena/run_target_domain_validation_sequential.py
    --
    "${RUNNER_ARGS[@]}"
)

echo "============================================================"
echo "Teams Promotion Contract — Vertex AI Launch"
echo "============================================================"
echo "Image:                   ${IMAGE_URI}"
echo "Project:                 ${PROJECT}"
echo "Region:                  ${REGION}"
echo "Job Name:                ${JOB_NAME}"
echo "Suite Manifest:          ${SUITE_MANIFEST}"
echo "Checkpoint Map:          ${CHECKPOINT_MAP}"
echo "Checkpoints:             ${CHECKPOINTS}"
echo "W&B Project:             ${WANDB_PROJECT}"
echo "Promotion policy:        target_real_fpr=${PROMOTION_TARGET_REAL_FPR}, target_stress_fpr=${PROMOTION_TARGET_STRESS_FPR}, target_fake_recall_min=${PROMOTION_TARGET_FAKE_RECALL_MIN}"
echo "Vertex Template:         ${YAML_TEMPLATE}"
echo "Vertex Output:           gs://training-job-outputs/vertex-output/${JOB_NAME}/"
echo "Detailed Reports:        ${REPORTS_GCS_FOLDER}/"
echo "Diagnostic Scorecards:   ${DIAGNOSTIC_GCS_DIR}/"
echo "Promotion Contract:      ${CONTRACT_GCS_DIR}/"
echo "Promotion authority:     calibrated contract"
echo "Diagnostic-only sidecar: fixed-threshold 0.5 scorecard"
echo "============================================================"
print_cmd "${LAUNCH_CMD[@]}"

if [[ -n "${DRY_RUN}" ]]; then
    echo "DRY RUN — no Vertex job submitted."
    exit 0
fi

"${LAUNCH_CMD[@]}"

echo ""
echo "============================================================"
echo "Promotion contract job submitted!"
echo "Vertex system output:"
echo "  gs://training-job-outputs/vertex-output/${JOB_NAME}/"
echo "Detailed validation reports:"
echo "  ${REPORTS_GCS_FOLDER}/"
echo "Diagnostic scorecard artifacts:"
echo "  ${DIAGNOSTIC_GCS_DIR}/scorecard.csv"
echo "  ${DIAGNOSTIC_GCS_DIR}/scorecard.wide.csv"
echo "  ${DIAGNOSTIC_GCS_DIR}/scorecard.int8_delta.csv"
echo "  ${DIAGNOSTIC_GCS_DIR}/scorecard.json"
echo "Promotion contract artifacts:"
echo "  ${CONTRACT_GCS_DIR}/threshold_grid.csv"
echo "  ${CONTRACT_GCS_DIR}/selected_threshold_scorecard.csv"
echo "  ${CONTRACT_GCS_DIR}/checkpoint_summary.csv"
echo "  ${CONTRACT_GCS_DIR}/promotion_contract.json"
echo "  ${CONTRACT_GCS_DIR}/promotion_winner.json"
echo "============================================================"
