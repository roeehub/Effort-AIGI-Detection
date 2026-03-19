#!/bin/bash
# =============================================================================
# launch_failure_analysis.sh
# =============================================================================
# Launches 5 Vertex AI validation jobs (one per R2.5 checkpoint) on the SAME
# evaluation data, then tells you how to run the local failure_analysis.py
# cross-comparison.
#
# Two phases:
#   Phase 1 (Vertex AI):  Run validate_custom_sources.py × 5 checkpoints
#   Phase 2 (local):      Download CSVs → run failure_analysis.py
#
# Usage:
#   export IMAGE_URI=us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:1.3.113
#   ./launch_failure_analysis.sh
#
# This does NOT require the new quality_robust Docker image — it uses the
# current production image since we're only running inference with existing
# checkpoints (no new augmentation code needed).
# =============================================================================

set -e

PROJECT="${PROJECT:-train-cvit2}"
REGION="${REGION:-us-central1}"

if [[ -z "$IMAGE_URI" ]]; then
  echo "ERROR: IMAGE_URI environment variable is required."
  echo "  Set it to the CURRENT production image, e.g.:"
  echo "  export IMAGE_URI=us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:1.3.113"
  exit 1
fi

echo "=================================================================="
echo "  Round 2.5 — Cross-Checkpoint Failure Analysis"
echo "  5 checkpoints × 4 data sources = 5 parallel Vertex AI jobs"
echo "=================================================================="
echo "  Image:   $IMAGE_URI"
echo "  Project: $PROJECT"
echo "  Region:  $REGION"
echo ""

# ─────────────────────────────────────────────
# Checkpoints (parallel arrays — bash 3.x compatible)
# ─────────────────────────────────────────────
CKPT_ORDER=(R25_F1 R25_F2 R25_F5 R25_F3 R25_F6)
CKPT_PATHS=(
  "gs://training-job-outputs/phase2r2_experiments/5w453our/top_n_effort_20260212_step18500_auc0.9893_eer0.0356.pth"
  "gs://training-job-outputs/phase2r2_experiments/ug8n870r/top_n_effort_20260212_step13500_auc0.9887_eer0.0249.pth"
  "gs://training-job-outputs/phase2r2_experiments/tkh09be0/top_n_effort_20260212_step18500_auc0.9885_eer0.0320.pth"
  "gs://training-job-outputs/phase2r2_experiments/6md2py50/top_n_effort_20260212_step18500_auc0.9884_eer0.0427.pth"
  "gs://training-job-outputs/phase2r2_experiments/x8lktnbc/top_n_effort_20260212_step22000_auc0.9882_eer0.0391.pth"
)

# ─────────────────────────────────────────────
# Data sources — same for ALL checkpoints
# ─────────────────────────────────────────────
DEEPLIVE_BUCKET="live-deepfake-methods-real-and-fake-frames-cropped"
EXTERNAL_REAL_BUCKET="effort-collected-data/real/external_youtube_avspeech"
VISOMASTER_BUCKET="live-deepfake-methods-real-and-fake-frames-cropped-visomaster"

# ─────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────
TS=$(date +%Y%m%d-%H%M%S)
OUTPUT_FOLDER="gs://training-job-outputs/failure_analysis/${TS}"
WANDB_PROJ="phase2-experiments"

echo "Checkpoints:"
for i in "${!CKPT_ORDER[@]}"; do
  echo "  ${CKPT_ORDER[$i]}: ...$(basename ${CKPT_PATHS[$i]})"
done
echo ""
echo "Evaluation data (same for all):"
echo "  • DF40 target→source (paired fakes + faceforensics++ reals)"
echo "  • DeepLive val split (fakes + reals, all strategies)"
echo "  • External YouTube AVSpeech reals (~7.3k)"
echo "  • VisoMaster (train swap models + held-out OOD models)"
echo ""
echo "Output: $OUTPUT_FOLDER"
echo ""

# Confirm
read -p "Launch 5 Vertex AI validation jobs? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 0
fi

# ─────────────────────────────────────────────
# Launch one job per checkpoint
# ─────────────────────────────────────────────
JOB_COUNT=0
TOTAL_JOBS=${#CKPT_ORDER[@]}

for i in "${!CKPT_ORDER[@]}"; do
  name="${CKPT_ORDER[$i]}"
  JOB_COUNT=$((JOB_COUNT + 1))
  CKPT="${CKPT_PATHS[$i]}"

  name_lower=$(echo "$name" | tr '[:upper:]' '[:lower:]')

  echo ""
  echo "=== [$JOB_COUNT/$TOTAL_JOBS] $name ==="
  echo "  Checkpoint: $(basename $CKPT)"

  ./launch_experiment_jobs.sh \
    --mode train \
    --job-name "fa-${name_lower}-${TS}" \
    --project "${PROJECT}" \
    --regions "${REGION}" \
    --image-uri "${IMAGE_URI}" \
    --gpu-type NVIDIA_TESLA_A100 \
    --gpu-count 1 \
    --main-script validate_custom_sources.py \
    -- --checkpoint_gcs_path "$CKPT" \
       --df40_orientation target_source \
       --df40_mode paired \
       --deeplive_bucket "$DEEPLIVE_BUCKET" \
       --deeplive_split val \
       --external_real_bucket "$EXTERNAL_REAL_BUCKET" \
       --visomaster_bucket "$VISOMASTER_BUCKET" \
       --visomaster_held_out_models "GhostFace-v3,InStyleSwapper256-C" \
       --frames_per_video 8 \
       --detailed_reports \
       --output_gcs_folder "$OUTPUT_FOLDER" \
       --output_filename_prefix "${name}_" \
       --log_prefix "${name}_validation" \
       --run_name "FailureAnalysis: ${name}" \
       --wandb_project "$WANDB_PROJ"

  echo "  ✅ $name submitted"
  sleep 3
done

echo ""
echo "=================================================================="
echo "  All $TOTAL_JOBS jobs submitted!"
echo "=================================================================="
echo ""
echo "Output folder: $OUTPUT_FOLDER"
echo ""
echo "Expected output files:"
for name in "${CKPT_ORDER[@]}"; do
  echo "  ${name}_frames_report.csv"
  echo "  ${name}_videos_report.csv"
done
echo ""
echo "=================================================================="
echo "  NEXT STEPS (after all 5 jobs finish):"
echo "=================================================================="
echo ""
echo "  # 1. Download all CSVs locally:"
echo "  mkdir -p ./failure_analysis_data"
echo "  gsutil -m cp '${OUTPUT_FOLDER}/*.csv' ./failure_analysis_data/"
echo ""
echo "  # 2. Run cross-checkpoint failure analysis:"
echo "  cd $(dirname $0)/../analysis_results"
echo "  python failure_analysis.py \\"
echo "    --data_dir ../DeepfakeBench/training/failure_analysis_data \\"
echo "    --models R25_F1,R25_F2,R25_F5,R25_F3,R25_F6 \\"
echo "    --output_dir ./failure_analysis_output"
echo ""
echo "  # 3. Review results:"
echo "  #    - consensus_failures.csv    → videos ALL models get wrong"
echo "  #    - discriminating_failures.csv → videos models disagree on"
echo "  #    - per_method_summary.csv     → side-by-side accuracy per method"
echo ""
echo "W&B: https://wandb.ai/dtect-vision/phase2-experiments"
