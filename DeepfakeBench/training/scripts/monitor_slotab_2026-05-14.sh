#!/usr/bin/env bash
# ============================================================================
# Monitor Slot α + Slot β training jobs and signal when both have terminated.
# Background usage: ./scripts/monitor_slotab_2026-05-14.sh &
# ============================================================================
set -uo pipefail

SLOT_A_JOB=4834133713400889344
SLOT_A_REGION=us-west4
SLOT_B_JOB=6512036079984443392
SLOT_B_REGION=us-east1

LOG=/tmp/monitor_slotab_2026-05-14.log
STATE_FILE=/tmp/monitor_slotab_2026-05-14.state
MAX_HOURS=8
POLL_INTERVAL=300  # 5 minutes

START=$(date +%s)
echo "=== Monitor started $(date -u) ===" >> "$LOG"
echo "  Slot α: $SLOT_A_JOB in $SLOT_A_REGION" >> "$LOG"
echo "  Slot β: $SLOT_B_JOB in $SLOT_B_REGION" >> "$LOG"

last_state_a=""
last_state_b=""

while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))
    if [ $ELAPSED -gt $((MAX_HOURS * 3600)) ]; then
        echo "[$(date -u)] TIMEOUT after ${MAX_HOURS}h (elapsed ${ELAPSED}s)" >> "$LOG"
        echo "TIMEOUT" > "$STATE_FILE"
        exit 1
    fi

    STATE_A=$(gcloud ai custom-jobs describe "$SLOT_A_JOB" --region="$SLOT_A_REGION" \
              --format='value(state)' 2>/dev/null || echo "ERR")
    STATE_B=$(gcloud ai custom-jobs describe "$SLOT_B_JOB" --region="$SLOT_B_REGION" \
              --format='value(state)' 2>/dev/null || echo "ERR")

    # Log state transitions only (not every poll)
    if [ "$STATE_A" != "$last_state_a" ] || [ "$STATE_B" != "$last_state_b" ]; then
        echo "[$(date -u)] α=$STATE_A  β=$STATE_B  (elapsed ${ELAPSED}s)" >> "$LOG"
        last_state_a="$STATE_A"
        last_state_b="$STATE_B"
    fi

    # Terminal states for both → exit
    case "$STATE_A:$STATE_B" in
        JOB_STATE_SUCCEEDED:JOB_STATE_SUCCEEDED)
            echo "[$(date -u)] BOTH SUCCEEDED — handing off for scorecard launch" >> "$LOG"
            echo "BOTH_SUCCEEDED" > "$STATE_FILE"
            exit 0
            ;;
        JOB_STATE_FAILED:*|*:JOB_STATE_FAILED|JOB_STATE_CANCELLED:*|*:JOB_STATE_CANCELLED)
            echo "[$(date -u)] FAILURE/CANCEL — α=$STATE_A β=$STATE_B" >> "$LOG"
            echo "FAILED" > "$STATE_FILE"
            exit 2
            ;;
    esac

    sleep "$POLL_INTERVAL"
done
