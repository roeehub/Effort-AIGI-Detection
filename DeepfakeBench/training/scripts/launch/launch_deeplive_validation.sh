#!/bin/bash
# Launch DeepLive validation jobs for B16 and L14 models
# Appends results to existing test_results folders

set -e

PROJECT="${PROJECT:-train-cvit2}"
IMAGE_URI="us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:$(cat VERSION)"

echo "Using image: $IMAGE_URI"
echo "Project: $PROJECT"
echo ""

# B16 checkpoint and output folder
B16_CHECKPOINT="gs://training-job-outputs/best_checkpoints/corrected/top_n_effort_20260119_step14000_auc0.9947_eer0.0218_B16_LAION.patched.pth"
B16_OUTPUT_FOLDER="gs://training-job-outputs/test_results/2026-01-20_14-28-06_B16_df40_source_target_extreal"

# L14 checkpoint and output folder  
L14_CHECKPOINT="gs://training-job-outputs/best_checkpoints/corrected/top_n_effort_20260113_step21000_auc0.9950_eer0.0175.patched.pth"
L14_OUTPUT_FOLDER="gs://training-job-outputs/test_results/2026-01-20_15-03-38_L14_df40_source_target_extreal"

# DeepLive bucket
DEEPLIVE_BUCKET="live-deepfake-methods-real-and-fake-frames-cropped"

echo "=== Launching B16 DeepLive Validation ==="
./launch_experiment_jobs.sh \
  --mode train \
  --job-name validate-b16-deeplive-$(date +%Y%m%d-%H%M%S) \
  --project "${PROJECT}" \
  --regions asia-southeast1 \
  --image-uri "${IMAGE_URI}" \
  --gpu-type NVIDIA_TESLA_A100 \
  --gpu-count 1 \
  --main-script validate_custom_sources.py \
  -- --checkpoint_gcs_path "$B16_CHECKPOINT" \
     --df40_mode none \
     --deeplive_bucket "$DEEPLIVE_BUCKET" \
     --deeplive_split val \
     --log_prefix B16_deeplive \
     --run_name "B16 DeepLive validation (val split)" \
     --output_gcs_folder "$B16_OUTPUT_FOLDER" \
     --output_filename_prefix deeplive_

echo ""
echo "=== Launching L14 DeepLive Validation ==="
./launch_experiment_jobs.sh \
  --mode train \
  --job-name validate-l14-deeplive-$(date +%Y%m%d-%H%M%S) \
  --project "${PROJECT}" \
  --regions asia-southeast1 \
  --image-uri "${IMAGE_URI}" \
  --gpu-type NVIDIA_TESLA_A100 \
  --gpu-count 1 \
  --main-script validate_custom_sources.py \
  -- --checkpoint_gcs_path "$L14_CHECKPOINT" \
     --df40_mode none \
     --deeplive_bucket "$DEEPLIVE_BUCKET" \
     --deeplive_split val \
     --log_prefix L14_deeplive \
     --run_name "L14 DeepLive validation (val split)" \
     --output_gcs_folder "$L14_OUTPUT_FOLDER" \
     --output_filename_prefix deeplive_

echo ""
echo "=== Both jobs submitted ==="
echo "B16 output: $B16_OUTPUT_FOLDER"
echo "L14 output: $L14_OUTPUT_FOLDER"
