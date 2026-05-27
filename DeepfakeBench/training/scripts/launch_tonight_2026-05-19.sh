#!/usr/bin/env bash
# ============================================================================
# Launch tonight's 3-slot batch (2026-05-19):
#
#   Slot 1: R13_T5C_6AXIS_PLUS_ANCHOR_2026-05-19  -> us-west4
#           (Slot β recipe + anchor_aware stacked; targets dor_shkedi FPR)
#   Slot 2: R13_LORA_T5C_L8_L9_R8_2026-05-19      -> us-east1
#           (LoRA layer-relocation ablation, L8-L9 vs L10-L11)
#   Slot 3: R13_T5C_5AXIS_NOLUMA_2026-05-19       -> us-central1
#           (5-axis sister of Slot β; isolates color_b vs luma contribution)
#
# Pre-req: pre-launch checks pass (see plan-next-3-gpu...).
# Image:  cat VERSION = 1.3.293 (no code changes; reuses last built image).
# Per CLAUDE.md: US regions only. 30-min PENDING -> switch regions.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

WANDB_PROJECT="phase2r13-experiments"

SLOT_1_YAML="experiments/phase2_round13/R13_T5C_6AXIS_PLUS_ANCHOR_2026-05-19.yaml"
SLOT_2_YAML="experiments/phase2_round13/R13_LORA_T5C_L8_L9_R8_2026-05-19.yaml"
SLOT_3_YAML="experiments/phase2_round13/R13_T5C_5AXIS_NOLUMA_2026-05-19.yaml"

SLOT_1_REGION="us-west4"
SLOT_2_REGION="us-east1"
SLOT_3_REGION="us-central1"

LOG_DIR="/tmp/tonight_2026-05-19"
mkdir -p "${LOG_DIR}"

echo "=== VERSION (image tag) ==="
cat VERSION
echo ""

echo "=== Launching Slot 1 (6-axis GRL + anchor stacked) to ${SLOT_1_REGION} ==="
./launch_experiment.sh -y "${WANDB_PROJECT}" "${SLOT_1_REGION}" "${SLOT_1_YAML}" 2>&1 | tee "${LOG_DIR}/slot_1_launch.log"

echo ""
echo "=== Launching Slot 2 (LoRA L8-L9 ablation) to ${SLOT_2_REGION} ==="
./launch_experiment.sh -y "${WANDB_PROJECT}" "${SLOT_2_REGION}" "${SLOT_2_YAML}" 2>&1 | tee "${LOG_DIR}/slot_2_launch.log"

echo ""
echo "=== Launching Slot 3 (5-axis sister, drop luma) to ${SLOT_3_REGION} ==="
./launch_experiment.sh -y "${WANDB_PROJECT}" "${SLOT_3_REGION}" "${SLOT_3_YAML}" 2>&1 | tee "${LOG_DIR}/slot_3_launch.log"

echo ""
echo "=== All 3 jobs submitted. Extracting job IDs/names ==="
SLOT_1_JOB=$(grep -E "^Job Name:" "${LOG_DIR}/slot_1_launch.log" | head -1 | awk '{print $NF}' || echo "PARSE_FAILED")
SLOT_2_JOB=$(grep -E "^Job Name:" "${LOG_DIR}/slot_2_launch.log" | head -1 | awk '{print $NF}' || echo "PARSE_FAILED")
SLOT_3_JOB=$(grep -E "^Job Name:" "${LOG_DIR}/slot_3_launch.log" | head -1 | awk '{print $NF}' || echo "PARSE_FAILED")

echo "Slot 1: ${SLOT_1_JOB} in ${SLOT_1_REGION}"
echo "Slot 2: ${SLOT_2_JOB} in ${SLOT_2_REGION}"
echo "Slot 3: ${SLOT_3_JOB} in ${SLOT_3_REGION}"

cat > "${LOG_DIR}/job_metadata.env" <<EOF
SLOT_1_JOB="${SLOT_1_JOB}"
SLOT_1_REGION="${SLOT_1_REGION}"
SLOT_1_YAML="${SLOT_1_YAML}"
SLOT_2_JOB="${SLOT_2_JOB}"
SLOT_2_REGION="${SLOT_2_REGION}"
SLOT_2_YAML="${SLOT_2_YAML}"
SLOT_3_JOB="${SLOT_3_JOB}"
SLOT_3_REGION="${SLOT_3_REGION}"
SLOT_3_YAML="${SLOT_3_YAML}"
EOF
echo ""
echo "Metadata written to ${LOG_DIR}/job_metadata.env"
