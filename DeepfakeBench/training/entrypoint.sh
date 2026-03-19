#!/usr/bin/env bash
# /workspace/entrypoint.sh
set -euo pipefail

# Defaults (env fallbacks)
JOB_MODE_ENV="${JOB_MODE:-train}"        # train | sweep | vertex_hpt
MAIN_SCRIPT_ENV="${MAIN_SCRIPT:-train_sweep.py}"
SWEEP_ID_ENV="${SWEEP_ID:-}"
COUNT_ENV="${SWEEP_COUNT:-5}"
GCS_CONFIG_ENV="${GCS_CONFIG:-}"         # NEW: GCS path to experiment config

# Apply CLI overrides
JOB_MODE="$JOB_MODE_ENV"
MAIN_SCRIPT="$MAIN_SCRIPT_ENV"
SWEEP_ID="$SWEEP_ID_ENV"
COUNT="$COUNT_ENV"
GCS_CONFIG="$GCS_CONFIG_ENV"

print_help() {
  cat <<EOF
Usage:
  /workspace/entrypoint.sh [--mode train|sweep|vertex_hpt] [--sweep-id <id>] [--count N] [--main-script PATH] [--gcs-config GCS_PATH] [--] [extra args...]
Notes:
  - CLI flags override env vars (JOB_MODE, SWEEP_ID, SWEEP_COUNT, MAIN_SCRIPT, GCS_CONFIG)
  - Extra args after "--" are passed to the Python script (train / vertex_hpt)
  - --gcs-config: Download experiment YAML from GCS (e.g., gs://my-bucket/configs/exp.yaml)
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
    --gcs-config)     GCS_CONFIG="$2"; shift 2 ;;
    --gcs-config=*)   GCS_CONFIG="${1#*=}"; shift ;;
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

# ============================================================
# Download experiment config from GCS if specified
# ============================================================
LOCAL_CONFIG_PATH=""
if [[ -n "$GCS_CONFIG" ]]; then
  echo "[entrypoint] Downloading experiment config from GCS: $GCS_CONFIG"
  LOCAL_CONFIG_PATH="/workspace/runtime_config.yaml"
  
  # Use Python SDK instead of gsutil (Vertex AI provides ADC for Python but not gsutil)
  if ! python -c "
from google.cloud import storage
import sys

gcs_path = '$GCS_CONFIG'
local_path = '$LOCAL_CONFIG_PATH'

# Parse gs://bucket/path
if not gcs_path.startswith('gs://'):
    print(f'Invalid GCS path: {gcs_path}', file=sys.stderr)
    sys.exit(1)

path_without_prefix = gcs_path[5:]  # Remove 'gs://'
parts = path_without_prefix.split('/', 1)
bucket_name = parts[0]
blob_name = parts[1] if len(parts) > 1 else ''

print(f'Downloading from bucket={bucket_name}, blob={blob_name}')
client = storage.Client()
bucket = client.bucket(bucket_name)
blob = bucket.blob(blob_name)
blob.download_to_filename(local_path)
print(f'Successfully downloaded to {local_path}')
"; then
    echo "[entrypoint] ERROR: Failed to download config from $GCS_CONFIG"
    exit 1
  fi
  
  echo "[entrypoint] Downloaded config to: $LOCAL_CONFIG_PATH"
  echo "[entrypoint] Config file exists: $(ls -la $LOCAL_CONFIG_PATH)"
  # Inject --param-config into EXTRA_ARGS if not already present
  if [[ ! " ${EXTRA_ARGS[*]:-} " =~ " --param-config " ]]; then
    EXTRA_ARGS+=("--param-config" "$LOCAL_CONFIG_PATH")
    echo "[entrypoint] Added --param-config to EXTRA_ARGS"
  fi
  echo "[entrypoint] EXTRA_ARGS now: ${EXTRA_ARGS[*]:-}"
else
  echo "[entrypoint] No GCS_CONFIG specified, skipping config download"
fi

echo "[entrypoint] JOB_MODE=$JOB_MODE"
echo "[entrypoint] Hostname: $(hostname) | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"

case "$JOB_MODE" in
  train)
    echo "[entrypoint] Starting single training run…"
    echo "[entrypoint] Script: $MAIN_SCRIPT"
    echo "[entrypoint] Full command: python -u $MAIN_SCRIPT ${EXTRA_ARGS[*]:-}"
    python -u "$MAIN_SCRIPT" "${EXTRA_ARGS[@]}"
    ;;
  ceiling)
    echo "[entrypoint] Starting B16 capacity ceiling experiment…"
    CEILING_SCRIPT="train_capacity_ceiling.py"
    echo "[entrypoint] Script: $CEILING_SCRIPT"
    echo "[entrypoint] Full command: python -u $CEILING_SCRIPT ${EXTRA_ARGS[*]:-}"
    python -u "$CEILING_SCRIPT" "${EXTRA_ARGS[@]}"
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
  batch_infer|batch-infer)
    echo "[entrypoint] Running batch inference on GCS buckets…"
    INFER_SCRIPT="batch_inference_gcs.py"
    echo "[entrypoint] Script: $INFER_SCRIPT"
    echo "[entrypoint] Full command: python -u $INFER_SCRIPT ${EXTRA_ARGS[*]:-}"
    python -u "$INFER_SCRIPT" "${EXTRA_ARGS[@]}"
    ;;
  *)
    echo "[entrypoint] ERROR: Unknown mode '$JOB_MODE'"; print_help; exit 1 ;;
esac
