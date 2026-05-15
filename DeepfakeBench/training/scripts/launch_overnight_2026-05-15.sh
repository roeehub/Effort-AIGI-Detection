#!/usr/bin/env bash
# ============================================================================
# Launch the 2026-05-15 overnight pair:
#   Slot α: R13_T5C_RESCHAIN_2026-05-15  -> us-east1
#   Slot β: R13_T5C_6AXIS_GRL_2026-05-15 -> us-west4
#
# Pre-req: ./dev.sh build-prod -y must have bumped VERSION and pushed image.
# Per CLAUDE.md: US regions only, never asia/europe without user authorization.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

WANDB_PROJECT="phase2r13-experiments"

SLOT_A_YAML="experiments/phase2_round13/R13_T5C_RESCHAIN_2026-05-15.yaml"
SLOT_B_YAML="experiments/phase2_round13/R13_T5C_6AXIS_GRL_2026-05-15.yaml"

SLOT_A_REGION="us-east1"
SLOT_B_REGION="us-west4"

LOG_DIR="/tmp/overnight_2026-05-15"
mkdir -p "${LOG_DIR}"

echo "=== Launching Slot α (resolution_chain_aug) to ${SLOT_A_REGION} ==="
./launch_experiment.sh -y "${WANDB_PROJECT}" "${SLOT_A_REGION}" "${SLOT_A_YAML}" 2>&1 | tee "${LOG_DIR}/slot_a_launch.log"

echo ""
echo "=== Launching Slot β (6-axis GRL) to ${SLOT_B_REGION} ==="
./launch_experiment.sh -y "${WANDB_PROJECT}" "${SLOT_B_REGION}" "${SLOT_B_YAML}" 2>&1 | tee "${LOG_DIR}/slot_b_launch.log"

echo ""
echo "=== Both jobs submitted. Extracting job IDs/names ==="
SLOT_A_JOB=$(grep -E "^Job Name:" "${LOG_DIR}/slot_a_launch.log" | head -1 | awk '{print $NF}')
SLOT_B_JOB=$(grep -E "^Job Name:" "${LOG_DIR}/slot_b_launch.log" | head -1 | awk '{print $NF}')
echo "Slot α: ${SLOT_A_JOB} in ${SLOT_A_REGION}"
echo "Slot β: ${SLOT_B_JOB} in ${SLOT_B_REGION}"

cat > "${LOG_DIR}/job_metadata.env" <<EOF
SLOT_A_JOB="${SLOT_A_JOB}"
SLOT_A_REGION="${SLOT_A_REGION}"
SLOT_A_YAML="${SLOT_A_YAML}"
SLOT_B_JOB="${SLOT_B_JOB}"
SLOT_B_REGION="${SLOT_B_REGION}"
SLOT_B_YAML="${SLOT_B_YAML}"
EOF
echo ""
echo "Metadata written to ${LOG_DIR}/job_metadata.env"
