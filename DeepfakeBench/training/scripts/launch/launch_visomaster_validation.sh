#!/bin/bash
# Launch VisoMaster validation for B16 model
# Validates against all 9 VisoMaster swap models with per-tier and per-model reporting
#
# This validates the B16 LAION checkpoint against the new VisoMaster data
# which sits in the same cropped bucket as DeepLiveCam data.
# Tier data (STRONG/MODERATE/MINIMAL) is enriched from the full-frames bucket.

set -e

PROJECT="${PROJECT:-train-cvit2}"
IMAGE_URI="us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:$(cat VERSION)"

echo "Using image: $IMAGE_URI"
echo "Project: $PROJECT"
echo ""

# B16 checkpoint
B16_CHECKPOINT="gs://training-job-outputs/best_checkpoints/corrected/top_n_effort_20260119_step14000_auc0.9947_eer0.0218_B16_LAION.patched.pth"
B16_OUTPUT_FOLDER="gs://training-job-outputs/test_results/2026-01-20_14-28-06_B16_df40_source_target_extreal"

# VisoMaster buckets
VISOMASTER_CROPPED_BUCKET="live-deepfake-methods-real-and-fake-frames-cropped"
VISOMASTER_FRAMES_BUCKET="live-deepfake-methods-real-and-fake-frames"

echo "=== Launching B16 VisoMaster Validation ==="
echo "  Checkpoint: $B16_CHECKPOINT"
echo "  Cropped bucket: $VISOMASTER_CROPPED_BUCKET"
echo "  Frames bucket (tier data): $VISOMASTER_FRAMES_BUCKET"
echo "  Output folder: $B16_OUTPUT_FOLDER"
echo ""

./launch_experiment_jobs.sh \
  --mode train \
  --job-name validate-b16-visomaster-$(date +%Y%m%d-%H%M%S) \
  --project "${PROJECT}" \
  --regions asia-southeast1 \
  --image-uri "${IMAGE_URI}" \
  --gpu-type NVIDIA_TESLA_A100 \
  --gpu-count 1 \
  --main-script validate_custom_sources.py \
  -- --checkpoint_gcs_path "$B16_CHECKPOINT" \
     --df40_mode none \
     --visomaster_bucket "$VISOMASTER_CROPPED_BUCKET" \
     --visomaster_frames_bucket "$VISOMASTER_FRAMES_BUCKET" \
     --visomaster_include_tier_methods \
     --log_prefix B16_visomaster \
     --run_name "B16 VisoMaster validation (all models, all tiers)" \
     --output_gcs_folder "$B16_OUTPUT_FOLDER" \
     --output_filename_prefix visomaster_

echo ""
echo "=== Job submitted ==="
echo "B16 output: $B16_OUTPUT_FOLDER"
echo ""
echo "Expected per-model methods:  visomaster_CSCS, visomaster_GhostFace-v1, etc."
echo "Expected per-tier methods:   visomaster_tier_STRONG, visomaster_tier_MODERATE, visomaster_tier_MINIMAL"
echo "Expected real method:        visomaster_real"
