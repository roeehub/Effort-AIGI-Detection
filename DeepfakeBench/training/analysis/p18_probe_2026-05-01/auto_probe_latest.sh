#!/usr/bin/env bash
# Auto-probe new P18 checkpoints as they land in GCS.
#
# Usage:
#   ./auto_probe_latest.sh <treatment_run_id> <control_run_id>
#
# Polls gs://training-job-outputs/phase2r13_experiments/<run_id>/ for new
# `value_composite_effort_*.pth` and `top_n_effort_*.pth` files. For each
# new file, downloads to /tmp/p18_ckpts/ and runs probe_p18_ckpt.py.
#
# Designed to be invoked by Monitor on a 5-min interval, OR run as a foreground
# loop. Idempotent — won't re-probe a ckpt that's already in the trajectory CSV.

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <treatment_run_id> <control_run_id>" >&2
    exit 1
fi

T_RUN="$1"
C_RUN="$2"
REPO_ROOT="/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
LOCAL_CACHE="/tmp/p18_ckpts"
TRAJECTORY_CSV="${REPO_ROOT}/analysis/p18_probe_2026-05-01/outputs/trajectory.csv"

mkdir -p "$LOCAL_CACHE"

probe_run() {
    local run_id="$1"
    local arm="$2"
    local prefix="gs://training-job-outputs/phase2r13_experiments/${run_id}/"

    # List recent ckpts
    local ckpts
    ckpts=$(gsutil ls "${prefix}*.pth" 2>/dev/null | grep -E "(value_composite|top_n|first_best)" || true)
    if [ -z "$ckpts" ]; then
        return 0
    fi

    while IFS= read -r gcs_uri; do
        [ -z "$gcs_uri" ] && continue
        local fname
        fname=$(basename "$gcs_uri")
        local label="${run_id}__${fname%.pth}"

        # Skip if already probed (label appears in trajectory.csv)
        if [ -f "$TRAJECTORY_CSV" ] && grep -qF "$label" "$TRAJECTORY_CSV" 2>/dev/null; then
            continue
        fi

        local local_path="${LOCAL_CACHE}/${run_id}__${fname}"
        if [ ! -f "$local_path" ]; then
            echo "[$(date '+%H:%M:%S')] downloading ${gcs_uri}..."
            if ! gsutil cp "$gcs_uri" "$local_path" 2>&1 | tail -2; then
                echo "[$(date '+%H:%M:%S')] download FAILED for $gcs_uri — skipping"
                continue
            fi
        fi

        echo "[$(date '+%H:%M:%S')] probing ${arm} ${label}..."
        cd "$REPO_ROOT"
        if python3 analysis/p18_probe_2026-05-01/probe_p18_ckpt.py \
            --ckpt "$local_path" --label "$label" --arm "$arm" 2>&1 | tail -15; then
            echo "[$(date '+%H:%M:%S')] PROBED: $label"
        else
            echo "[$(date '+%H:%M:%S')] PROBE FAILED: $label"
        fi
    done <<< "$ckpts"
}

probe_run "$T_RUN" "treatment"
probe_run "$C_RUN" "control"
