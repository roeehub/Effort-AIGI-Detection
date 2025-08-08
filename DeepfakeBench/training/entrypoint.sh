#!/usr/bin/env bash
# entrypoint.sh
# Thin wrapper so the same image can:
#   • run a single training job
#   • host a W&B sweep agent
#   • execute a Vertex HPT trial (env vars HP_* → CLI flags)

set -euo pipefail

JOB_MODE="${JOB_MODE:-train}"   # train | sweep | vertex_hpt
MAIN_SCRIPT="train_sweep.py"

echo "[entrypoint] JOB_MODE=$JOB_MODE"
echo "[entrypoint] Hostname: $(hostname) | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"not set"}"

case "$JOB_MODE" in
  train)
        echo "[entrypoint] Starting single training run..."
        python -u "$MAIN_SCRIPT" "$@"
        ;;
  sweep)
        if [[ -z "${SWEEP_ID:-}" ]]; then
            echo "[entrypoint] ERROR: SWEEP_ID env-var not set"
            exit 1
        fi
        echo "[entrypoint] Launching W&B sweep agent for $SWEEP_ID"
        wandb agent "$SWEEP_ID"
        ;;
  vertex_hpt)
        echo "[entrypoint] Running Vertex HPT trial…"
        # Convert any HP_* env-vars to --<key> <value> CLI flags
        PARAMS=()
        for v in $(compgen -e | grep '^HP_'); do
            key=$(echo "$v" | sed 's/^HP_//' | tr '[:upper:]' '[:lower:]')
            PARAMS+=("--$key" "${!v}")
        done
        python -u "$MAIN_SCRIPT" "${PARAMS[@]}" "$@"
        ;;
  *)
        echo "[entrypoint] ERROR: Unknown JOB_MODE '$JOB_MODE'."
        exit 1
        ;;
esac
