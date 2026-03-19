#!/bin/bash
# Launch DeepLive-only validation for R25_F1 (best Phase 2 checkpoint)
# Evaluates on ALL 860 DeepLive samples (edge_cases + minimal_processing)
# No VisoMaster, no DF40 — pure DeepLive evaluation for threshold calibration.
#
# Expected data:
#   - 435 edge_cases samples  → 435 real + 435 fake
#   - 425 minimal_processing  → 425 real + 425 fake
#   - Total: 1,720 VideoInfo entries
#
# Output: per-method CSV reports + overall AUC/EER/thresholds → GCS

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

# R25_F1 checkpoint (best Phase 2 model: AUC=0.9893, rank=736, k=32)
F1_CHECKPOINT="gs://training-job-outputs/phase2r2_experiments/5w453our/top_n_effort_20260212_step18500_auc0.9893_eer0.0356.pth"
F1_OUTPUT_FOLDER="gs://training-job-outputs/test_results/R25_F1_deeplive_evaluation"

# DeepLive bucket (shared with VisoMaster, but load_deeplive_validation filters
# to only edge_cases + minimal_processing strategies)
DEEPLIVE_BUCKET="live-deepfake-methods-real-and-fake-frames-cropped"

echo "=== Launching R25_F1 DeepLive FULL Validation ==="
echo "  Checkpoint: $F1_CHECKPOINT"
echo "  DeepLive bucket: $DEEPLIVE_BUCKET"
echo "  Split: ALL (train + val = 860 samples)"
echo "  Output folder: $F1_OUTPUT_FOLDER"
echo ""

./launch_experiment_jobs.sh \
  --mode train \
  --job-name validate-f1-deeplive-all-$(date +%Y%m%d-%H%M%S) \
  --project "${PROJECT}" \
  --regions asia-southeast1 \
  --image-uri "${IMAGE_URI}" \
  --gpu-type NVIDIA_TESLA_A100 \
  --gpu-count 1 \
  --main-script validate_custom_sources.py \
  -- --checkpoint_gcs_path "$F1_CHECKPOINT" \
     --df40_mode none \
     --deeplive_bucket "$DEEPLIVE_BUCKET" \
     --deeplive_split all \
     --log_prefix F1_deeplive_all \
     --run_name "R25_F1 DeepLive FULL evaluation (all 860 samples)" \
     --output_gcs_folder "$F1_OUTPUT_FOLDER" \
     --output_filename_prefix deeplive_all_ \
     --wandb_project phase2-experiments

echo ""
echo "=== Job submitted ==="
echo "Output: $F1_OUTPUT_FOLDER"
echo ""
echo "Expected methods in report:"
echo "  - deeplive_edge_cases (435 samples)"
echo "  - deeplive_minimal_processing (425 samples)"
echo ""
echo "Check W&B: https://wandb.ai/dtect-vision/phase2-experiments"
