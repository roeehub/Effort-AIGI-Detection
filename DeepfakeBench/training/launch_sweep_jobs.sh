#!/bin/bash

export WANDB_API_KEY=bb5a8ea4a27ebe45917587df8c46674d26e43966
export WANDB_PROJECT="Effort-Deepfake"
export WANDB_ENTITY="dtect-vision"
export ACTUAL_SWEEP_ID="dtect-vision/Effort-Deepfake/3ovkgus1" # REPLACE THIS WITH YOUR ACTUAL SWEEP ID

# Loop to launch 6 jobs
for i in $(seq 1 6); do
    JOB_NAME="job-${i}-$(date +%Y%m%d%H%M%S)"
    echo "Launching job ${JOB_NAME}..."
    ./launch_experiment.sh \
        --job-name "${JOB_NAME}" \
        --mode "sweep" \
        --sweep-id "${ACTUAL_SWEEP_ID}" & # The '&' runs each launch in the background
    sleep 5 # Small delay between launches to avoid API rate limits or similar
done

echo "All 8 jobs launched. Monitor them in the GCP console and W&B dashboard."