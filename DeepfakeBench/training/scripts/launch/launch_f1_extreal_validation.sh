#!/bin/bash
# Launch external real (YouTube AVSpeech) validation for R25_F1
# Evaluates ONLY on real videos to measure false positive rate at various thresholds.
#
# Expected data:
#   - ~7,869 real videos from effort-collected-data/real/external_youtube_avspeech
#   - All labeled "real" — this is a false-positive-rate test
#
# Key metrics to watch:
#   - What % of real videos are correctly classified at the EER threshold (0.5601)?
#   - FPR at various operating points
#
# Output: per-frame and per-video CSV reports → GCS

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
F1_OUTPUT_FOLDER="gs://training-job-outputs/test_results/R25_F1_evaluation"

# External real data
EXTERNAL_REAL_BUCKET="effort-collected-data/real/external_youtube_avspeech"

echo "=== Launching R25_F1 External Real (YouTube AVSpeech) Validation ==="
echo "  Checkpoint: $F1_CHECKPOINT"
echo "  External real: gs://$EXTERNAL_REAL_BUCKET"
echo "  Sampling: 2,000 / ~7,869 real videos (seed=737)"
echo "  Output folder: $F1_OUTPUT_FOLDER"
echo ""

./launch_experiment_jobs.sh \
  --mode train \
  --job-name validate-f1-extreal-$(date +%Y%m%d-%H%M%S) \
  --project "${PROJECT}" \
  --regions asia-southeast1 \
  --image-uri "${IMAGE_URI}" \
  --gpu-type NVIDIA_TESLA_A100 \
  --gpu-count 1 \
  --main-script validate_custom_sources.py \
  -- --checkpoint_gcs_path "$F1_CHECKPOINT" \
     --df40_mode none \
     --external_real_bucket "$EXTERNAL_REAL_BUCKET" \
     --max_external_real 2000 \
     --log_prefix F1_extreal \
     --run_name "R25_F1 External Real (2k sample, seed=737) - FPR test" \
     --output_gcs_folder "$F1_OUTPUT_FOLDER" \
     --output_filename_prefix extreal_ \
     --wandb_project phase2-experiments

echo ""
echo "=== Job submitted ==="
echo "Output: $F1_OUTPUT_FOLDER"
echo ""
echo "This is a REAL-ONLY evaluation (false positive rate test)."
echo "Watch for: % of real videos misclassified as fake at various thresholds."
echo ""
echo "Check W&B: https://wandb.ai/dtect-vision/phase2-experiments"
