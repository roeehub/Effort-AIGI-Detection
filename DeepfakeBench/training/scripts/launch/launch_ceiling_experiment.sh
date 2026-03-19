#!/usr/bin/env bash
# launch_ceiling_experiment.sh - Launch B16 capacity ceiling experiments on GCP Vertex AI
#
# Usage:
#   ./launch_ceiling_experiment.sh <EXPERIMENT_NUMBER>
#
# Examples:
#   ./launch_ceiling_experiment.sh 2          # Unfreeze last 2 blocks
#   ./launch_ceiling_experiment.sh 4          # Full finetune (ceiling test)
#   ./launch_ceiling_experiment.sh all        # Run all experiments sequentially

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
PROJECT="${PROJECT:-train-cvit2}"
REGION="${REGION:-asia-southeast1}"
WANDB_PROJECT="B16-capacity-ceiling"
WANDB_ENTITY="${WANDB_ENTITY:-dtect-vision}"
VERSION="$(cat VERSION)"
IMAGE_URI="us-docker.pkg.dev/${PROJECT}/effort-detector/effort-detector:${VERSION}"

# GPU config
GPU_TYPE="${GPU_TYPE:-NVIDIA_TESLA_A100}"
GPU_COUNT="${GPU_COUNT:-1}"

# Export for launch_experiment_jobs.sh
export WANDB_API_KEY="${WANDB_API_KEY:-bb5a8ea4a27ebe45917587df8c46674d26e43966}"
export WANDB_ENTITY
export WANDB_PROJECT

usage() {
    cat << EOF
Usage: $0 <EXPERIMENT_NUMBER>

B16 Capacity Ceiling Experiments:
  1  - Baseline Effort (control, use regular train_sweep.py)
  2  - Unfreeze last 2 blocks
  3  - Unfreeze last 4 blocks  
  4  - Full finetune (THE CEILING TEST)
  all - Run experiments 2, 3, 4 sequentially

Example:
  $0 4        # Run full finetune experiment
  $0 all      # Run all capacity tests

Note: Experiment 1 (baseline) should use the regular training pipeline
      for fair comparison with existing results.
EOF
}

launch_experiment() {
    local exp_num="$1"
    local config_pattern="experiments/B16_capacity_ceiling/${exp_num}_*.yaml"
    
    # Find the config file
    local config_file
    config_file=$(ls $config_pattern 2>/dev/null | head -1) || true
    if [[ -z "$config_file" ]]; then
        echo "❌ Config file not found for experiment $exp_num"
        exit 1
    fi
    
    echo "=============================================="
    echo "🚀 Launching B16 Ceiling Experiment $exp_num"
    echo "   Config: $config_file"
    echo "   Region: $REGION"
    echo "   GPU: $GPU_TYPE x $GPU_COUNT"
    echo "=============================================="
    
    local job_name="b16-ceiling-exp${exp_num}-$(date +%Y%m%d-%H%M%S)"
    
    # Container path for config
    local container_config="/workspace/${config_file}"
    
    # Use the existing launch_experiment_jobs.sh which handles env vars properly
    ./launch_experiment_jobs.sh \
        --mode "ceiling" \
        --job-name "$job_name" \
        --project "$PROJECT" \
        --regions "$REGION" \
        --image-uri "$IMAGE_URI" \
        --gpu-type "$GPU_TYPE" \
        --gpu-count "$GPU_COUNT" \
        --main-script "train_capacity_ceiling.py" \
        -- --config "$container_config"
    
    echo "✅ Job submitted: $job_name"
    echo ""
}

# Parse arguments
if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

exp="$1"

case "$exp" in
    1)
        echo "⚠️  Experiment 1 (baseline) should use regular train_sweep.py"
        echo "   Run: ./launch_experiment.sh B16-capacity-ceiling $REGION experiments/B16_capacity_ceiling/1_baseline_effort.yaml"
        ;;
    2|3|4)
        launch_experiment "$exp"
        ;;
    all)
        echo "🔄 Running all ceiling experiments (2, 3, 4)..."
        for e in 2 3 4; do
            launch_experiment "$e"
            echo "⏳ Waiting 30s before next submission..."
            sleep 30
        done
        echo "✅ All experiments submitted!"
        ;;
    *)
        echo "❌ Unknown experiment: $exp"
        usage
        exit 1
        ;;
esac
