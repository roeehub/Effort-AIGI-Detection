#!/usr/bin/env bash
# launch_experiment.sh
# Final version: Handles the complete YAML template with all environment variables.
set -euo pipefail

# --- Default parameters ---
REGIONS=(us-central1 us-east4 europe-west4)
GPU_TYPE="NVIDIA_A100_40GB"
GPU_COUNT=1
JOB_NAME="test-run-$(date +%Y%m%d-%H%M%S)"
JOB_MODE="train" # Default job mode
YAML_TEMPLATE="vertex_job_template.yaml"
TEMP_YAML_CONFIG="/tmp/vertex_job_config_$$_${JOB_NAME}.yaml"

# --- Cleanup Function ---
trap 'rm -f "$TEMP_YAML_CONFIG"' EXIT

# --- CLI flags ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --job-name) JOB_NAME="$2"; shift 2;;
    --gpu)      GPU_TYPE="$2"; shift 2;;
    --mode)     JOB_MODE="$2"; shift 2;; # Allow overriding job mode
    *)          echo "Unknown flag $1"; exit 1;;
  esac
done

# --- Environment and Sanity Checks ---
PROJECT_ID="$(gcloud config get-value project)"
IMAGE_URI="us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:latest"
VERTEX_JOB_SERVICE_ACCOUNT="vertex-job-runner-train-cvit2@${PROJECT_ID}.iam.gserviceaccount.com"
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

# --- Map user-friendly GPU names to the required API identifiers ---
case "$GPU_TYPE" in
  "NVIDIA_A100_40GB") API_GPU_TYPE="NVIDIA_TESLA_A100" ;;
  "NVIDIA_A100_80GB") API_GPU_TYPE="NVIDIA_A100_80GB" ;;
  "NVIDIA_H100_80GB") API_GPU_TYPE="NVIDIA_H100_80GB" ;;
  *) echo "[launcher] ERROR: Unrecognized GPU type '$GPU_TYPE'."; exit 1 ;;
esac

[[ -z "$PROJECT_ID" ]] && { echo "[launcher] ERROR: gcloud project not set."; exit 1; }
[[ ! -f "$YAML_TEMPLATE" ]] && { echo "[launcher] ERROR: Template file '$YAML_TEMPLATE' not found."; exit 1; }
if grep -q "YOUR_BUCKET_NAME_HERE" "$YAML_TEMPLATE"; then
  echo "[launcher] ERROR: You must replace 'gs://YOUR_BUCKET_NAME_HERE' in '$YAML_TEMPLATE' with your actual GCS bucket path."
  exit 1
fi

# --- Main Loop ---
for REGION in "${REGIONS[@]}"; do
  echo "[launcher] Attempting to launch in region: $REGION ..."

  # This 'sed' command is now updated to handle all your environment variables.
  sed -e "s|{{JOB_NAME}}|${JOB_NAME}|g" \
      -e "s|{{IMAGE_URI}}|${IMAGE_URI}|g" \
      -e "s|{{GPU_TYPE}}|${API_GPU_TYPE}|g" \
      -e "s|{{GPU_COUNT}}|${GPU_COUNT}|g" \
      -e "s|{{SERVICE_ACCOUNT}}|${VERTEX_JOB_SERVICE_ACCOUNT}|g" \
      -e "s|{{WANDB_API_KEY}}|${WANDB_API_KEY}|g" \
      -e "s|{{WANDB_PROJECT}}|${WANDB_PROJECT}|g" \
      -e "s|{{WANDB_ENTITY}}|${WANDB_ENTITY}|g" \
      -e "s|{{JOB_MODE}}|${JOB_MODE}|g" \
      "$YAML_TEMPLATE" > "$TEMP_YAML_CONFIG"

  # The command structure remains the same and is known to work.
  if gcloud ai custom-jobs create \
        --project="$PROJECT_ID" \
        --region="$REGION" \
        --display-name="$JOB_NAME" \
        --config="$TEMP_YAML_CONFIG" \
        --quiet ; then

    JOB_ID=$(gcloud ai custom-jobs list --project="$PROJECT_ID" --region="$REGION" --filter="displayName=$JOB_NAME" --sort-by=~creationTime --limit=1 --format="value(name)")
    echo "[launcher] ✅ SUCCESS! Job '$JOB_NAME' launched in region $REGION."
    echo "[launcher] View job at: https://console.cloud.google.com/vertex-ai/locations/$REGION/training/$JOB_ID?project=$PROJECT_ID"
    exit 0
  else
    echo "[launcher] ❌ FAILED in $REGION. Trying next region..."
    echo "----------------------------------------------------------------------"
  fi
done

echo "[launcher] 🔥 FINAL ERROR: Failed to launch job in all specified regions." >&2
exit 1