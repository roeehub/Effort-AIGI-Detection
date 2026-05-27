#!/usr/bin/env bash
# P22 post-launch monitor.
#
# Usage:
#   ./08_monitor_p22_launch.sh <JOB_NAME> [REGION]
#
# Polls Vertex job status; alerts when transition into RUNNING (or fails / queues
# > 30 min, per CLAUDE.md region-capacity playbook).

set -euo pipefail

JOB_NAME="${1:-}"
REGION="${2:-us-west4}"
PROJECT="${PROJECT:-train-cvit2}"

if [[ -z "${JOB_NAME}" ]]; then
    echo "Usage: $0 <JOB_NAME> [REGION]"
    exit 1
fi

START_TS=$(date +%s)
DEADLINE_PENDING_S=1800  # 30 min PENDING → relaunch elsewhere per CLAUDE.md

echo "Monitoring Vertex job ${JOB_NAME} in ${REGION}..."
while :; do
    STATE=$(gcloud ai custom-jobs describe "${JOB_NAME}" \
        --region="${REGION}" --project="${PROJECT}" \
        --format="value(state)" 2>/dev/null || echo "UNKNOWN")
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TS))
    printf "  [%4ds]  %s\n" "${ELAPSED}" "${STATE}"

    case "${STATE}" in
        JOB_STATE_RUNNING)
            echo "✅ RUNNING after ${ELAPSED}s"
            break
            ;;
        JOB_STATE_SUCCEEDED|JOB_STATE_FAILED|JOB_STATE_CANCELLED)
            echo "⚠️ Terminal state ${STATE} after ${ELAPSED}s"
            exit 1
            ;;
        JOB_STATE_QUEUED|JOB_STATE_PENDING)
            if (( ELAPSED > DEADLINE_PENDING_S )); then
                echo "⏰ ${ELAPSED}s in pending state (> ${DEADLINE_PENDING_S}s)"
                echo "   CLAUDE.md says: switch regions. Suggest us-east1 or us-central1."
                echo "   But CONFIRM WITH USER before launching elsewhere or cancelling current."
                exit 2
            fi
            ;;
    esac
    sleep 60
done

echo
echo "Job is RUNNING. Streaming first 50 log lines:"
gcloud ai custom-jobs stream-logs "${JOB_NAME}" \
    --region="${REGION}" --project="${PROJECT}" 2>&1 | head -50 || true
