#!/usr/bin/env bash
# ============================================================================
# Launch 2026-05-21 codec-restoration triple:
#
#   Slot 1: R13_T5C_ANCHOR_AWARE_PLUS_CODEC_2026-05-21         -> us-west4
#           (anchor + codec p=0.40 = P8A-matched dose; FT from Slot A v2)
#   Slot 2: R13_T5C_ANCHOR_AWARE_PLUS_CODEC_LIGHT_2026-05-21   -> us-east1
#           (anchor + codec p=0.20 = half dose; FT from Slot A v2; dose-response)
#   Slot 3: R13_T5C_CODEC_ONLY_NO_ANCHOR_2026-05-21            -> us-central1
#           (NO anchor + codec p=0.40; FT from T5C; anchor-necessity control)
#
# 2×2 design:                  codec_p=0.0         codec_p=0.40
#   anchor=ON  (Slot A v2)     reference (already)    Slot 1
#   anchor=OFF (T5C)           reference (already)    Slot 3
#                              codec_p=0.20 sweep: Slot 2 (anchor=ON)
#
# Image: cat VERSION = 1.3.295 (rebuilt 2026-05-20T22:18 UTC with the 3 new yamls).
# Per CLAUDE.md: US regions only. 30-min PENDING -> switch regions per rules.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

WANDB_PROJECT="phase2r13-experiments"

SLOT_1_YAML="experiments/phase2_round13/R13_T5C_ANCHOR_AWARE_PLUS_CODEC_2026-05-21.yaml"
SLOT_2_YAML="experiments/phase2_round13/R13_T5C_ANCHOR_AWARE_PLUS_CODEC_LIGHT_2026-05-21.yaml"
SLOT_3_YAML="experiments/phase2_round13/R13_T5C_CODEC_ONLY_NO_ANCHOR_2026-05-21.yaml"

SLOT_1_REGION="us-west4"
SLOT_2_REGION="us-east1"
SLOT_3_REGION="us-central1"

LOG_DIR="/tmp/tonight_2026-05-21"
mkdir -p "${LOG_DIR}"

echo "=== VERSION (image tag) ==="
cat VERSION
echo ""

echo "=== Launching Slot 1 (anchor + codec p=0.40) to ${SLOT_1_REGION} ==="
./launch_experiment.sh -y "${WANDB_PROJECT}" "${SLOT_1_REGION}" "${SLOT_1_YAML}" 2>&1 | tee "${LOG_DIR}/slot_1_launch.log"

echo ""
echo "=== Launching Slot 2 (anchor + codec p=0.20 light) to ${SLOT_2_REGION} ==="
./launch_experiment.sh -y "${WANDB_PROJECT}" "${SLOT_2_REGION}" "${SLOT_2_YAML}" 2>&1 | tee "${LOG_DIR}/slot_2_launch.log"

echo ""
echo "=== Launching Slot 3 (NO anchor + codec p=0.40 control) to ${SLOT_3_REGION} ==="
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

echo "Wrote: ${LOG_DIR}/job_metadata.env"
