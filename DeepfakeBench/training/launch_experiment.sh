#!/usr/bin/env bash
# launch_experiment.sh
# Helper: fills template, checks quota, falls back to next region if needed.
set -euo pipefail

# ─── default params ────────────────────────────────────────────────────────────
REGIONS=(us-central1 us-east4 europe-west4 asia-northeast1)
GPU_TYPE="NVIDIA_A100_40GB"
GPU_COUNT=1
JOB_MODE="train"
JOB_NAME="effort-run-$(date +%Y%m%d-%H%M%S)"
YAML_TEMPLATE="vertex_job_template.yaml"

# ─── CLI flags ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --region)   REGIONS=("$2"); shift 2;;
    --gpu)      GPU_TYPE="$2"; shift 2;;
    --job-name) JOB_NAME="$2"; shift 2;;
    --mode)     JOB_MODE="$2"; shift 2;;
    *)          echo "Unknown flag $1"; exit 1;;
  esac
done

# ─── env & sanity checks ───────────────────────────────────────────────────────
IMAGE_URI="us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:latest"
SERVICE_ACCOUNT="$(gcloud config get-value account)"
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

[[ -z "$WANDB_API_KEY" ]] && { echo "[launcher] WANDB_API_KEY env-var not set"; exit 1; }
[[ -z "$WANDB_PROJECT" ]] && { echo "[launcher] WANDB_PROJECT env-var not set"; exit 1; }
[[ -z "$WANDB_ENTITY"  ]] && { echo "[launcher] WANDB_ENTITY  env-var not set"; exit 1; }

export IMAGE_URI JOB_NAME JOB_MODE GPU_TYPE SERVICE_ACCOUNT WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY

# ─── cost map (USD/h) ──────────────────────────────────────────────────────────
GPU_RATE=""
case "$GPU_TYPE" in
  NVIDIA_A100_40GB) GPU_RATE="2.45" ;;
  NVIDIA_A100_80GB) GPU_RATE="4.12" ;;
  NVIDIA_H100_80GB) GPU_RATE="4.70" ;;
  *)                GPU_RATE=""      ;;  # unknown type
esac

# ─── iterate regions until quota available ─────────────────────────────────────
for REGION in "${REGIONS[@]}"; do
  echo "[launcher] Trying region $REGION ..."
  if gcloud ai locations describe "$REGION" --format="value(locationId)" >/dev/null 2>&1; then
    # Check concurrent-job quota (simplified)
    AVAIL=$(gcloud ai regions describe "$REGION" \
              --format="value(quotas[metric=custom-jobs-concurrent-capacity].available)")
    if (( AVAIL > 0 )); then
      echo "[launcher]  $AVAIL job slots free – launching here."
      envsubst < "$YAML_TEMPLATE" | \
        gcloud ai custom-jobs create --region "$REGION" --quiet --file=-
      if [[ -n "$GPU_RATE" ]]; then
        echo "[launcher] 🚀 Estimated cost/hour: \$${GPU_RATE}"
      else
        echo "[launcher] (cost rate for $GPU_TYPE not in map)"
      fi
      exit 0
    else
      echo "[launcher]  No slots in $REGION."
    fi
  fi
done

echo "[launcher] ERROR: No region had free CustomJob quota." >&2
exit 1
