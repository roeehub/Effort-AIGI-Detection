#!/bin/bash
# =============================================================================
# 2x2 Checkpoint Comparison: R25_F1 (new) vs B16-old
# =============================================================================
# Submits 4 Vertex AI jobs to evaluate both checkpoints on:
#   1. DeepLive ALL (860 samples: edge_cases + minimal_processing)
#   2. External Real YouTube AVSpeech (~7,869 videos, all)
#
# Comparison matrix:
#   ┌──────────────┬───────────────────┬───────────────────┐
#   │              │ DeepLive (fake)   │ ExtReal (real)    │
#   ├──────────────┼───────────────────┼───────────────────┤
#   │ R25_F1 (new) │ AUC/EER/thresh    │ FPR at thresholds │
#   │ B16-old      │ AUC/EER/thresh    │ FPR at thresholds │
#   └──────────────┴───────────────────┴───────────────────┘
#
# All results go to the same GCS output folder with distinct prefixes,
# and all log to W&B phase2-experiments for easy side-by-side comparison.
# =============================================================================

set -e

PROJECT="${PROJECT:-train-cvit2}"
REGION="us-central1"

if [[ -z "$IMAGE_URI" ]]; then
  echo "ERROR: IMAGE_URI environment variable is required."
  echo "  Set it after building the Docker image, e.g.:"
  echo "  export IMAGE_URI=us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:<tag>"
  exit 1
fi

echo "=========================================="
echo "  2x2 Checkpoint Comparison Evaluation"
echo "=========================================="
echo "  Image:   $IMAGE_URI"
echo "  Project: $PROJECT"
echo "  Region:  $REGION"
echo ""

# --- Checkpoints ---
F1_CHECKPOINT="gs://training-job-outputs/phase2r2_experiments/5w453our/top_n_effort_20260212_step18500_auc0.9893_eer0.0356.pth"
B16_CHECKPOINT="gs://training-job-outputs/best_checkpoints/corrected/top_n_effort_20260119_step14000_auc0.9947_eer0.0218_B16_LAION.patched.pth"

# --- Output ---
OUTPUT_FOLDER="gs://training-job-outputs/test_results/2x2_F1_vs_B16old_comparison"

# --- Data sources ---
DEEPLIVE_BUCKET="live-deepfake-methods-real-and-fake-frames-cropped"
EXTERNAL_REAL_BUCKET="effort-collected-data/real/external_youtube_avspeech"

WANDB_PROJ="phase2-experiments"
TS=$(date +%Y%m%d-%H%M%S)

echo "Checkpoints:"
echo "  NEW (R25_F1): ...auc0.9893_eer0.0356.pth"
echo "  OLD (B16):    ...auc0.9947_eer0.0218_B16_LAION.patched.pth"
echo ""
echo "Data:"
echo "  DeepLive:  860 samples (edge_cases + minimal_processing, all splits)"
echo "  ExtReal:   ~7,869 videos (all)"
echo ""
echo "Output: $OUTPUT_FOLDER"
echo ""

# ─────────────────────────────────────────────────────────
# Job 1/4: R25_F1 × DeepLive ALL
# ─────────────────────────────────────────────────────────
echo "=== [1/4] R25_F1 × DeepLive ALL ==="
./launch_experiment_jobs.sh \
  --mode train \
  --job-name validate-f1-deeplive-${TS} \
  --project "${PROJECT}" \
  --regions "${REGION}" \
  --image-uri "${IMAGE_URI}" \
  --gpu-type NVIDIA_TESLA_A100 \
  --gpu-count 1 \
  --main-script validate_custom_sources.py \
  -- --checkpoint_gcs_path "$F1_CHECKPOINT" \
     --df40_mode none \
     --deeplive_bucket "$DEEPLIVE_BUCKET" \
     --deeplive_split all \
     --log_prefix F1_deeplive_all \
     --run_name "R25_F1 × DeepLive ALL (860 samples)" \
     --output_gcs_folder "$OUTPUT_FOLDER" \
     --output_filename_prefix F1_deeplive_ \
     --wandb_project "$WANDB_PROJ"
echo "  ✅ Submitted"
sleep 3

# ─────────────────────────────────────────────────────────
# Job 2/4: R25_F1 × External Real (2k sample)
# ─────────────────────────────────────────────────────────
echo "=== [2/4] R25_F1 × External Real (all ~8k) ==="
./launch_experiment_jobs.sh \
  --mode train \
  --job-name validate-f1-extreal-${TS} \
  --project "${PROJECT}" \
  --regions "${REGION}" \
  --image-uri "${IMAGE_URI}" \
  --gpu-type NVIDIA_TESLA_A100 \
  --gpu-count 1 \
  --main-script validate_custom_sources.py \
  -- --checkpoint_gcs_path "$F1_CHECKPOINT" \
     --df40_mode none \
     --external_real_bucket "$EXTERNAL_REAL_BUCKET" \
     --log_prefix F1_extreal \
     --run_name "R25_F1 × External Real (all ~8k)" \
     --output_gcs_folder "$OUTPUT_FOLDER" \
     --output_filename_prefix F1_extreal_ \
     --wandb_project "$WANDB_PROJ"
echo "  ✅ Submitted"
sleep 3

# ─────────────────────────────────────────────────────────
# Job 3/4: B16-old × DeepLive ALL
# ─────────────────────────────────────────────────────────
echo "=== [3/4] B16-old × DeepLive ALL ==="
./launch_experiment_jobs.sh \
  --mode train \
  --job-name validate-b16old-deeplive-${TS} \
  --project "${PROJECT}" \
  --regions "${REGION}" \
  --image-uri "${IMAGE_URI}" \
  --gpu-type NVIDIA_TESLA_A100 \
  --gpu-count 1 \
  --main-script validate_custom_sources.py \
  -- --checkpoint_gcs_path "$B16_CHECKPOINT" \
     --df40_mode none \
     --deeplive_bucket "$DEEPLIVE_BUCKET" \
     --deeplive_split all \
     --log_prefix B16old_deeplive_all \
     --run_name "B16-old × DeepLive ALL (860 samples)" \
     --output_gcs_folder "$OUTPUT_FOLDER" \
     --output_filename_prefix B16old_deeplive_ \
     --wandb_project "$WANDB_PROJ"
echo "  ✅ Submitted"
sleep 3

# ─────────────────────────────────────────────────────────
# Job 4/4: B16-old × External Real (2k sample)
# ─────────────────────────────────────────────────────────
echo "=== [4/4] B16-old × External Real (all ~8k) ==="
./launch_experiment_jobs.sh \
  --mode train \
  --job-name validate-b16old-extreal-${TS} \
  --project "${PROJECT}" \
  --regions "${REGION}" \
  --image-uri "${IMAGE_URI}" \
  --gpu-type NVIDIA_TESLA_A100 \
  --gpu-count 1 \
  --main-script validate_custom_sources.py \
  -- --checkpoint_gcs_path "$B16_CHECKPOINT" \
     --df40_mode none \
     --external_real_bucket "$EXTERNAL_REAL_BUCKET" \
     --log_prefix B16old_extreal \
     --run_name "B16-old × External Real (all ~8k)" \
     --output_gcs_folder "$OUTPUT_FOLDER" \
     --output_filename_prefix B16old_extreal_ \
     --wandb_project "$WANDB_PROJ"
echo "  ✅ Submitted"

echo ""
echo "=========================================="
echo "  All 4 jobs submitted!"
echo "=========================================="
echo ""
echo "Output: $OUTPUT_FOLDER"
echo "  F1_deeplive_frames_report.csv      F1_deeplive_videos_report.csv"
echo "  F1_extreal_frames_report.csv       F1_extreal_videos_report.csv"
echo "  B16old_deeplive_frames_report.csv   B16old_deeplive_videos_report.csv"
echo "  B16old_extreal_frames_report.csv    B16old_extreal_videos_report.csv"
echo ""
echo "W&B: https://wandb.ai/dtect-vision/phase2-experiments"
echo ""
echo "Comparison plan:"
echo "  ┌──────────────┬───────────────────┬───────────────────┐"
echo "  │              │ DeepLive (fake)   │ ExtReal (real)    │"
echo "  ├──────────────┼───────────────────┼───────────────────┤"
echo "  │ R25_F1 (new) │ F1_deeplive_*     │ F1_extreal_*      │"
echo "  │ B16-old      │ B16old_deeplive_* │ B16old_extreal_*  │"
echo "  └──────────────┴───────────────────┴───────────────────┘"
