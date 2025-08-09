##!/usr/bin/env bash
## launch_experiment_jobs.sh
## Final version: Handles the complete YAML template with all environment variables.
#set -euo pipefail
#
#--- Default parameters ---
#REGIONS=(us-central1 us-east4 europe-west4)
#GPU_TYPE="NVIDIA_A100_40GB"
#GPU_COUNT=1
#JOB_NAME="test-run-$(date +%Y%m%d-%H%M%S)" # Unique job name
#YAML_TEMPLATE="vertex_job_template.yaml" # Assuming your template file is named this
#TEMP_YAML_CONFIG="/tmp/vertex_job_config_$$_${JOB_NAME}.yaml"
#
## --- NEW: Add SWEEP_ID and default JOB_MODE ---
#SWEEP_ID="" # This will be passed as a CLI argument or env var
#JOB_MODE="train" # Default to train, can be overridden to sweep
#
## --- Cleanup Function ---
#trap 'rm -f "$TEMP_YAML_CONFIG"' EXIT
#
## --- CLI flags ---
#while [[ $# -gt 0 ]]; do
#  case "$1" in
#    --job-name) JOB_NAME="$2"; shift 2;;
#    --gpu)      GPU_TYPE="$2"; shift 2;;
#    --count)    GPU_COUNT="$2"; shift 2;; # Ensure this flag is handled if you intend to vary GPU_COUNT via CLI
#    --mode)     JOB_MODE="$2"; shift 2;; # Allow overriding job mode
#    --sweep-id) SWEEP_ID="$2"; shift 2;; # NEW: For passing sweep ID
#    *)          echo "Unknown flag $1"; exit 1;;
#  esac
#done
#
## --- Environment and Sanity Checks ---
#PROJECT_ID="$(gcloud config get-value project)"
#IMAGE_URI="us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:latest"
#VERTEX_JOB_SERVICE_ACCOUNT="vertex-job-runner-train-cvit2@${PROJECT_ID}.iam.gserviceaccount.com"
#WANDB_API_KEY="${WANDB_API_KEY:-}" # Expect this to be set in the shell env
#WANDB_PROJECT="${WANDB_PROJECT:-}" # Expect this to be set in the shell env
#WANDB_ENTITY="${WANDB_ENTITY:-}"   # Expect this to be set in the shell env
#
#if [[ -z "$WANDB_API_KEY" ]]; then
#    echo "[launcher] ERROR: WANDB_API_KEY environment variable is not set." >&2
#    exit 1
#fi
#if [[ -z "$WANDB_PROJECT" ]]; then
#    echo "[launcher] ERROR: WANDB_PROJECT environment variable is not set." >&2
#    exit 1
#fi
#if [[ -z "$WANDB_ENTITY" ]]; then
#    echo "[launcher] ERROR: WANDB_ENTITY environment variable is not set." >&2
#    exit 1
#fi
#
## Ensure SWEEP_ID is set if JOB_MODE is sweep
#if [[ "$JOB_MODE" == "sweep" && -z "$SWEEP_ID" ]]; then
#    echo "[launcher] ERROR: --sweep-id must be provided when --mode is 'sweep'." >&2
#    exit 1
#fi
#
## --- Map user-friendly GPU names to the required API identifiers ---
#
#case "${GPU_TYPE}" in
#    "NVIDIA_A100_40GB") API_GPU_TYPE="NVIDIA_TESLA_A100";;
#    "NVIDIA_A100_80GB") API_GPU_TYPE="NVIDIA_A100_80GB";; # Note: Vertex AI usually has NVIDIA_TESLA_A100 for 40GB, A100_80GB for 80GB
#    "NVIDIA_H100_80GB") API_GPU_TYPE="NVIDIA_H100_80GB";;
#    *) echo "[launcher] ERROR: Unrecognized GPU type '${GPU_TYPE}'."; exit 1 ;;
#esac
#
#if [[ ! -f "$YAML_TEMPLATE" ]]; then
#    echo "[launcher] ERROR: Template file '$YAML_TEMPLATE' not found."
#    exit 1
#fi
#
## Check for GCS bucket placeholder in the template (good practice)
#if grep -q "gs://YOUR_BUCKET_NAME_HERE" "$YAML_TEMPLATE"; then
#    echo "[launcher] ERROR: You must replace 'gs://YOUR_BUCKET_NAME_HERE' in '$YAML_TEMPLATE' with your actual GCS bucket path."
#    exit 1
#fi
#
## --- Main Loop ---
#for REGION in "${REGIONS[@]}"; do
#    echo "[launcher] Attempting to launch in region: $REGION ..."
#
#    sed -e "s|{{JOB_NAME}}|${JOB_NAME}|g" \
#        -e "s|{{IMAGE_URI}}|${IMAGE_URI}|g" \
#        -e "s|{{GPU_TYPE}}|${API_GPU_TYPE}|g" \
#        -e "s|{{GPU_COUNT}}|${GPU_COUNT}|g" \
#        -e "s|{{SERVICE_ACCOUNT}}|${VERTEX_JOB_SERVICE_ACCOUNT}|g" \
#        -e "s|{{WANDB_API_KEY}}|${WANDB_API_KEY}|g" \
#        -e "s|{{WANDB_PROJECT}}|${WANDB_PROJECT}|g" \
#        -e "s|{{WANDB_ENTITY}}|${WANDB_ENTITY}|g" \
#        -e "s|{{JOB_MODE}}|${JOB_MODE}|g" \
#        -e "s|{{SWEEP_ID}}|${SWEEP_ID}|g" \
#        "$YAML_TEMPLATE" > "$TEMP_YAML_CONFIG"
#
#  # The command structure remains the same and is known to work.
#    if gcloud ai custom-jobs create \
#        --project="${PROJECT_ID}" \
#        --region="${REGION}" \
#        --display-name="${JOB_NAME}" \
#        --config="${TEMP_YAML_CONFIG}" \
#        --quiet ; then
#
#        JOB_ID=$(gcloud ai custom-jobs list --project="$PROJECT_ID" --region="$REGION" --filter="displayName=$JOB_NAME" --sort-by=~creationTime --limit=1 --format="value(name)")
#        echo "[launcher] ✅ SUCCESS! Job '$JOB_NAME' launched in region $REGION."
#        echo "[launcher] View job at: https://console.cloud.google.com/vertex-ai/locations/$REGION/training/$JOB_ID?project=$PROJECT_ID"
#        # Since you want to launch 8 jobs, you'd typically remove 'exit 0' here
#        # if you intend this script to launch all 8 sequentially in one run.
#        # For separate `gcloud` commands, each will exit.
#        break # Exit loop on first successful launch if you only want one job per script execution
#    else
#        echo "[launcher] ❌ FAILED in $REGION. Trying next region..."
#        echo "----------------------------------------------------------------------"
#    fi
#done
#
#if [[ "$?" -ne 0 ]]; then # Check if the last command failed (i.e., no job launched successfully)
#    echo "[launcher] 🔥 FINAL ERROR: Failed to launch job in all specified regions." >&2
#    exit 1
#fi