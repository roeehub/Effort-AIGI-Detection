#!/usr/bin/env bash
# ============================================================================
# Monitor Slot α + Slot β overnight 2026-05-15 jobs. Signals on terminal state.
# Reads job IDs from /tmp/overnight_2026-05-15/job_metadata.env (created by
# launch_overnight_2026-05-15.sh).
#
# Background usage: ./scripts/monitor_overnight_2026-05-15.sh &
# ============================================================================
set -uo pipefail

META=/tmp/overnight_2026-05-15/job_metadata.env
if [ ! -f "$META" ]; then
    echo "ERR: metadata file $META missing — launch jobs first" >&2
    exit 1
fi
source "$META"

LOG=/tmp/overnight_2026-05-15/monitor.log
STATE_FILE=/tmp/overnight_2026-05-15/monitor.state
MAX_HOURS=8
POLL_INTERVAL=300  # 5 minutes

START=$(date +%s)
echo "=== Overnight monitor started $(date -u) ===" >> "$LOG"
echo "  Slot α: ${SLOT_A_JOB} in ${SLOT_A_REGION}" >> "$LOG"
echo "  Slot β: ${SLOT_B_JOB} in ${SLOT_B_REGION}" >> "$LOG"

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

    if [ "$STATE_A" != "$last_state_a" ] || [ "$STATE_B" != "$last_state_b" ]; then
        echo "[$(date -u)] α=$STATE_A  β=$STATE_B  (elapsed ${ELAPSED}s)" >> "$LOG"
        last_state_a="$STATE_A"
        last_state_b="$STATE_B"
    fi

    # Flag both PENDING > 30min on launch (CLAUDE.md region-capacity rule);
    # do NOT auto-switch since user is offline.
    if [ "$ELAPSED" -gt 1800 ] && \
       [ "$STATE_A" = "JOB_STATE_PENDING" ] && \
       [ "$STATE_B" = "JOB_STATE_PENDING" ] && \
       [ ! -f /tmp/overnight_2026-05-15/pending_warning_logged ]; then
        echo "[$(date -u)] WARNING: both jobs PENDING > 30min in their respective US regions; not auto-switching (user offline)" >> "$LOG"
        touch /tmp/overnight_2026-05-15/pending_warning_logged
    fi

    case "$STATE_A:$STATE_B" in
        JOB_STATE_SUCCEEDED:JOB_STATE_SUCCEEDED)
            echo "[$(date -u)] BOTH SUCCEEDED — ready for scorecard" >> "$LOG"
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
