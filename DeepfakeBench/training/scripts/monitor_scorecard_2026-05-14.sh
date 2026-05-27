#!/usr/bin/env bash
# ============================================================================
# Monitor the slotab-scorecard-2026-05-14 Vertex job; signal on completion.
# Background usage: ./scripts/monitor_scorecard_2026-05-14.sh &
# ============================================================================
set -uo pipefail

JOB_ID=1924697647042527232
REGION=us-east1
LOG=/tmp/monitor_scorecard_2026-05-14.log
STATE_FILE=/tmp/monitor_scorecard_2026-05-14.state
MAX_HOURS=10
POLL_INTERVAL=300  # 5 minutes

START=$(date +%s)
echo "=== Scorecard monitor started $(date -u) ===" >> "$LOG"
echo "  Job: $JOB_ID in $REGION" >> "$LOG"

last_state=""
while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))
    if [ $ELAPSED -gt $((MAX_HOURS * 3600)) ]; then
        echo "[$(date -u)] TIMEOUT after ${MAX_HOURS}h" >> "$LOG"
        echo "TIMEOUT" > "$STATE_FILE"
        exit 1
    fi

    STATE=$(gcloud ai custom-jobs describe "$JOB_ID" --region="$REGION" \
            --format='value(state)' 2>/dev/null || echo "ERR")

    if [ "$STATE" != "$last_state" ]; then
        echo "[$(date -u)] state=$STATE  (elapsed ${ELAPSED}s)" >> "$LOG"
        last_state="$STATE"
    fi

    case "$STATE" in
        JOB_STATE_SUCCEEDED)
            echo "[$(date -u)] SUCCEEDED — scorecard complete" >> "$LOG"
            echo "SUCCEEDED" > "$STATE_FILE"
            exit 0
            ;;
        JOB_STATE_FAILED|JOB_STATE_CANCELLED)
            echo "[$(date -u)] $STATE" >> "$LOG"
            echo "$STATE" > "$STATE_FILE"
            exit 2
            ;;
    esac

    sleep "$POLL_INTERVAL"
done
