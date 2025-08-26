#!/bin/zsh

# This script runs the launch_experiment_jobs.sh command sequentially
# for a predefined list of configuration files.

# Define the array of configuration file names
CONFIG_FILES=(
  "night1_8x8.yaml"
  "night1_16x4.yaml"
  "night1_32x2.yaml"
  "night1_64x1_full_data.yaml"
  "night1_64x1_lre4.yaml"
  "night1_64x1_lre5.yaml"
)

# Base path for the configuration files
CONFIG_BASE_PATH="/workspace/experiments_configs/night1"

# Loop through each configuration file and run the command
for config_file in "${CONFIG_FILES[@]}"; do
  echo "-----------------------------------------------------"
  echo "Starting experiment with config: ${config_file}"
  echo "-----------------------------------------------------"

  ./launch_experiment_jobs.sh --mode train \
    --project "${PROJECT}" \
    --regions "${REGIONS}" \
    --image-uri "${IMAGE_URI}" \
    --main-script train_sweep.py \
    -- \
    --param-config "${CONFIG_BASE_PATH}/${config_file}"

  echo "-----------------------------------------------------"
  echo "Finished experiment with config: ${config_file}"
  echo "-----------------------------------------------------"
  echo # Add a newline for better readability
done

echo "All experiments have been launched."