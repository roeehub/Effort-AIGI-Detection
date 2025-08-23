#!/usr/bin/env bash
# /workspace/entrypoint.sh
set -euo pipefail

# Defaults (env fallbacks)
JOB_MODE_ENV="${JOB_MODE:-train}"        # train | sweep | vertex_hpt
MAIN_SCRIPT_ENV="${MAIN_SCRIPT:-train_sweep.py}"
SWEEP_ID_ENV="${SWEEP_ID:-}"
COUNT_ENV="${SWEEP_COUNT:-5}"

# Apply CLI overrides
JOB_MODE="$JOB_MODE_ENV"
MAIN_SCRIPT="$MAIN_SCRIPT_ENV"
SWEEP_ID="$SWEEP_ID_ENV"
COUNT="$COUNT_ENV"

print_help() {
  cat <<EOF
Usage:
  /workspace/entrypoint.sh [--mode train|sweep|vertex_hpt] [--sweep-id <id>] [--count N] [--main-script PATH] [--] [extra args...]
Notes:
  - CLI flags override env vars (JOB_MODE, SWEEP_ID, SWEEP_COUNT, MAIN_SCRIPT)
  - Extra args after "--" are passed to the Python script (train / vertex_hpt)
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)           JOB_MODE="$2"; shift 2 ;;
    --mode=*)         JOB_MODE="${1#*=}"; shift ;;
    -m)               JOB_MODE="$2"; shift 2 ;;
    --sweep-id)       SWEEP_ID="$2"; shift 2 ;;
    --sweep-id=*)     SWEEP_ID="${1#*=}"; shift ;;
    --count)          COUNT="$2"; shift 2 ;;
    --count=*)        COUNT="${1#*=}"; shift ;;
    --main-script)    MAIN_SCRIPT="$2"; shift 2 ;;
    --main-script=*)  MAIN_SCRIPT="${1#*=}"; shift ;;
    -h|--help)        print_help; exit 0 ;;
    --)               shift; EXTRA_ARGS+=("$@"); break ;;
    *)                EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# Allow PY_ARGS env to inject extra flags too
if [[ -n "${PY_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS+=(${PY_ARGS})
fi

echo "[entrypoint] JOB_MODE=$JOB_MODE"
echo "[entrypoint] Hostname: $(hostname) | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"

case "$JOB_MODE" in
  train)
    echo "[entrypoint] Starting single training run…"
    echo "[entrypoint] Script: $MAIN_SCRIPT"
    python -u "$MAIN_SCRIPT" "${EXTRA_ARGS[@]}"
    ;;
  sweep)
    if [[ -z "$SWEEP_ID" ]]; then
      echo "[entrypoint] ERROR: --sweep-id (or SWEEP_ID) is required for sweep mode"
      exit 1
    fi
    echo "[entrypoint] Launching W&B agent: SWEEP_ID=$SWEEP_ID | count=$COUNT"
    wandb agent "$SWEEP_ID" --count "$COUNT"
    ;;
  vertex_hpt|vertex-hpt)
    echo "[entrypoint] Running Vertex HPT trial…"
    PARAMS=()
    while IFS='=' read -r k v; do
      k="${k#HP_}"
      PARAMS+=("--${k,,}")
      PARAMS+=("$v")
    done < <(printenv | grep '^HP_')
    echo "[entrypoint] Script: $MAIN_SCRIPT"
    echo "[entrypoint] Params from HP_*: ${PARAMS[*]:-(none)}"
    python -u "$MAIN_SCRIPT" "${PARAMS[@]}" "${EXTRA_ARGS[@]}"
    ;;
  *)
    echo "[entrypoint] ERROR: Unknown mode '$JOB_MODE'"; print_help; exit 1 ;;
esac
