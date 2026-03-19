#!/bin/bash
# ============================================================================
# launch_r10_diagnostics.sh
# Launch R10 Pre-Training Diagnostic Experiments on Vertex AI
#
# Runs Experiments 0A (JPEG sensitivity), 0B (Lighting profile), 0C (Teams v2)
# against the R9_A production champion checkpoint.
#
# Usage:
#   ./launch_r10_diagnostics.sh                    # Run all 3 experiments
#   ./launch_r10_diagnostics.sh --experiments 0A   # Run only JPEG sensitivity
#   ./launch_r10_diagnostics.sh --experiments 0B   # Run only lighting profile
#   ./launch_r10_diagnostics.sh --experiments 0C   # Run only Teams v2 check
# ============================================================================

set -euo pipefail

# ── Defaults ──
PROJECT="${PROJECT:-train-cvit2}"
IMAGE_URI="us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:$(cat VERSION)"
REGION="${REGION:-asia-southeast1}"

# ── Parse --experiments flag (supports: --experiments 0C, or bare 0C, or nothing) ──
EXPERIMENTS_VALUE="0A,0B,0C"  # default: all
while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiments)
            if [[ -z "${2:-}" || "$2" == --* ]]; then
                echo "ERROR: --experiments requires a value (e.g., --experiments 0C)" >&2
                exit 1
            fi
            EXPERIMENTS_VALUE="$2"
            shift 2
            ;;
        *)
            # Bare positional: treat as experiment list
            EXPERIMENTS_VALUE="$1"
            shift
            ;;
    esac
done

# ── R9_A Champion Checkpoint ──
R9_A_CHECKPOINT="gs://training-job-outputs/phase2r9_experiments/1551zxa8/top_n_effort_20260228_step6000_auc0.9891_eer0.0457.pth"

# ── Output folder (timestamped) ──
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_GCS_FOLDER="gs://training-job-outputs/r10_diagnostics/${TIMESTAMP}_R9A_pre_r10"

# ── Teams v2 bucket ──
TEAMS_V2_BUCKET="live-deepfake-methods-real-and-fake-frames-cropped-teams-v2"

echo "============================================================"
echo "R10 Pre-Training Diagnostics"
echo "============================================================"
echo "Image:      ${IMAGE_URI}"
echo "Project:    ${PROJECT}"
echo "Region:     ${REGION}"
echo "Checkpoint: ${R9_A_CHECKPOINT}"
echo "Output:     ${OUTPUT_GCS_FOLDER}"
echo "Experiments: ${EXPERIMENTS_VALUE}"
echo ""

echo "Launching diagnostic job..."

./launch_experiment_jobs.sh \
  --mode train \
  --job-name "r10-diagnostics-${TIMESTAMP}" \
  --project "${PROJECT}" \
  --regions "${REGION}" \
  --image-uri "${IMAGE_URI}" \
  --gpu-type NVIDIA_TESLA_A100 \
  --gpu-count 1 \
  --main-script run_r10_diagnostics.py \
  -- --checkpoint_gcs_path "${R9_A_CHECKPOINT}" \
     --experiments "${EXPERIMENTS_VALUE}" \
     --output_gcs_folder "${OUTPUT_GCS_FOLDER}" \
     --teams_v2_bucket "${TEAMS_V2_BUCKET}" \
     --0a_per_method 30 \
     --0b_per_method 5 \
     --0c_max_samples 100 \
     --wandb_project r10-diagnostics

echo ""
echo "============================================================"
echo "Job submitted!"
echo ""
echo "Output files will be at:"
echo "  ${OUTPUT_GCS_FOLDER}/"
echo ""
echo "Expected outputs:"
echo "  Experiment 0A (JPEG Sensitivity):"
echo "    - 0A_jpeg_sensitivity_frames.csv     (per-frame scores at each JPEG quality)"
echo "    - 0A_jpeg_sensitivity_summary.csv    (AUC/accuracy per quality level)"
echo "    - 0A_jpeg_sensitivity_per_method.csv (per-method breakdown)"
echo ""
echo "  Experiment 0B (Lighting Profile):"
echo "    - 0B_lighting_sensitivity_frames.csv    (per-sample × perturbation scores)"
echo "    - 0B_lighting_sensitivity_summary.csv   (mean delta, flip rate per perturbation)"
echo "    - 0B_cosine_distance_vs_perturbation.csv (embedding distance under perturbation)"
echo ""
echo "  Experiment 0C (Teams v2 Sanity):"
echo "    - 0C_teams_v2_frames.csv          (per-frame scores)"
echo "    - 0C_teams_v2_videos.csv          (per-video aggregated scores)"
echo "    - 0C_teams_v2_summary.csv         (per-strategy breakdown)"
echo "    - 0C_teams_v2_score_histogram.csv (for distribution plotting)"
echo ""
echo "  Master:"
echo "    - diagnostics_summary.json (all results + timing)"
echo "============================================================"
