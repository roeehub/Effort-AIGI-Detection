#!/bin/bash
# Launch old B16 checkpoint validation on DeepLive + External Real
# For comparison against R25_F1 checkpoint results.
#
# Old B16: AUC=0.9947, EER=0.0218 (Phase 1 best, DF40-only training)
# New F1:  AUC=0.9893, EER=0.0356 (Phase 2 best, combined training)
#
# This submits TWO Vertex AI jobs:
#   1. DeepLive ALL (860 samples: edge_cases + minimal_processing)
#   2. External Real YouTube AVSpeech (~7,869 real videos)

set -e

PROJECT="${PROJECT:-train-cvit2}"

if [[ -z "$IMAGE_URI" ]]; then
  echo "ERROR: IMAGE_URI environment variable is required."
  echo "  Set it after building the Docker image, e.g.:"
  echo "  export IMAGE_URI=us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:<tag>"
  exit 1
fi

echo "Using image: $IMAGE_URI"
echo "Project: $PROJECT"
echo ""

# Old B16 checkpoint
B16_CHECKPOINT="gs://training-job-outputs/best_checkpoints/corrected/top_n_effort_20260119_step14000_auc0.9947_eer0.0218_B16_LAION.patched.pth"
B16_OUTPUT_FOLDER="gs://training-job-outputs/test_results/B16_old_vs_F1_comparison"

# Data sources
DEEPLIVE_BUCKET="live-deepfake-methods-real-and-fake-frames-cropped"
EXTERNAL_REAL_BUCKET="effort-collected-data/real/external_youtube_avspeech"

echo "============================================"
echo "  Old B16 Checkpoint Comparison Evaluation"
echo "============================================"
echo "  Checkpoint: $B16_CHECKPOINT"
echo "  Output:     $B16_OUTPUT_FOLDER"
echo ""

# --- Job 1: DeepLive ALL ---
echo "=== [1/2] Launching B16 DeepLive ALL Validation ==="
./launch_experiment_jobs.sh \
  --mode train \
  --job-name validate-b16old-deeplive-$(date +%Y%m%d-%H%M%S) \
  --project "${PROJECT}" \
  --regions asia-southeast1 \
  --image-uri "${IMAGE_URI}" \
  --gpu-type NVIDIA_TESLA_A100 \
  --gpu-count 1 \
  --main-script validate_custom_sources.py \
  -- --checkpoint_gcs_path "$B16_CHECKPOINT" \
     --df40_mode none \
     --deeplive_bucket "$DEEPLIVE_BUCKET" \
     --deeplive_split all \
     --log_prefix B16old_deeplive_all \
     --run_name "B16-old DeepLive FULL (all 860 samples) - comparison" \
     --output_gcs_folder "$B16_OUTPUT_FOLDER" \
     --output_filename_prefix deeplive_all_ \
     --wandb_project phase2-experiments

echo ""
sleep 3

# --- Job 2: External Real ---
echo "=== [2/2] Launching B16 External Real Validation ==="
./launch_experiment_jobs.sh \
  --mode train \
  --job-name validate-b16old-extreal-$(date +%Y%m%d-%H%M%S) \
  --project "${PROJECT}" \
  --regions asia-southeast1 \
  --image-uri "${IMAGE_URI}" \
  --gpu-type NVIDIA_TESLA_A100 \
  --gpu-count 1 \
  --main-script validate_custom_sources.py \
  -- --checkpoint_gcs_path "$B16_CHECKPOINT" \
     --df40_mode none \
     --external_real_bucket "$EXTERNAL_REAL_BUCKET" \
     --max_external_real 2000 \
     --log_prefix B16old_extreal \
     --run_name "B16-old External Real (2k sample, seed=737) - comparison" \
     --output_gcs_folder "$B16_OUTPUT_FOLDER" \
     --output_filename_prefix extreal_ \
     --wandb_project phase2-experiments

echo ""
echo "=== Both jobs submitted ==="
echo "Output: $B16_OUTPUT_FOLDER"
echo ""
echo "Comparison plan:"
echo "  B16-old DeepLive  vs  R25_F1 DeepLive"
echo "  B16-old ExtReal   vs  R25_F1 ExtReal"
echo ""
echo "Check W&B: https://wandb.ai/dtect-vision/phase2-experiments"
