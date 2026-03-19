#!/bin/bash
# Launch B16 SVD Rank Sweep Experiments
#
# Usage:
#   ./launch_svd_sweep.sh all [REGION] [PROJECT]
#   ./launch_svd_sweep.sh pack_a [REGION] [PROJECT]
#   ./launch_svd_sweep.sh pack_b [REGION] [PROJECT]
#   ./launch_svd_sweep.sh k1 [REGION] [PROJECT]
#
# Examples:
#   ./launch_svd_sweep.sh all                            # Use defaults
#   ./launch_svd_sweep.sh all asia-southeast1            # Custom region
#   ./launch_svd_sweep.sh all asia-southeast1 my-project # Custom region and project
#
# Reference: experiments/B16_svd_rank_sweep/README.md

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
EXPERIMENT_TYPE="${1:-all}"
REGION="${2:-${GCP_REGION:-asia-southeast1}}"
PROJECT="${3:-${WANDB_PROJECT:-B16-svd-sweep}}"

# Experiment configs
PACK_A_CONFIGS=(
    "experiments/B16_svd_rank_sweep/rank_k1_baseline.yaml"
    "experiments/B16_svd_rank_sweep/rank_k2.yaml"
    "experiments/B16_svd_rank_sweep/rank_k4.yaml"
    "experiments/B16_svd_rank_sweep/rank_k8.yaml"
    "experiments/B16_svd_rank_sweep/rank_k16.yaml"
)

PACK_B_CONFIGS=(
    "experiments/B16_svd_rank_sweep/rank_k8_late_3_blocks.yaml"
    "experiments/B16_svd_rank_sweep/rank_k8_late_5_blocks.yaml"
)

launch_experiment() {
    local config=$1
    echo "==========================================="
    echo "Launching: $config"
    echo "Project: $PROJECT"
    echo "Region: $REGION"
    echo "==========================================="
    ./launch_experiment.sh "$PROJECT" "$REGION" "$config"
    echo ""
}

case "$EXPERIMENT_TYPE" in
    all)
        echo "🚀 Launching ALL Experiments: Pack A + Pack B (7 experiments)"
        echo ""
        echo "Pack A: SVD Rank Sweep (k=1, 2, 4, 8, 16)"
        for config in "${PACK_A_CONFIGS[@]}"; do
            launch_experiment "$config"
        done
        echo ""
        echo "Pack B: Late Layers (last 3 and 5 blocks)"
        for config in "${PACK_B_CONFIGS[@]}"; do
            launch_experiment "$config"
        done
        echo "✅ All 7 experiments launched!"
        ;;
    pack_a)
        echo "🚀 Launching Pack A: SVD Rank Sweep (5 experiments)"
        echo ""
        for config in "${PACK_A_CONFIGS[@]}"; do
            launch_experiment "$config"
        done
        echo "✅ All Pack A experiments launched!"
        ;;
    pack_b)
        echo "🚀 Launching Pack B: Late Layers Experiments (2 experiments)"
        echo ""
        for config in "${PACK_B_CONFIGS[@]}"; do
            launch_experiment "$config"
        done
        echo "✅ All Pack B experiments launched!"
        ;;
    k1)
        launch_experiment "experiments/B16_svd_rank_sweep/rank_k1_baseline.yaml"
        ;;
    k2)
        launch_experiment "experiments/B16_svd_rank_sweep/rank_k2.yaml"
        ;;
    k4)
        launch_experiment "experiments/B16_svd_rank_sweep/rank_k4.yaml"
        ;;
    k8)
        launch_experiment "experiments/B16_svd_rank_sweep/rank_k8.yaml"
        ;;
    k16)
        launch_experiment "experiments/B16_svd_rank_sweep/rank_k16.yaml"
        ;;
    late3)
        launch_experiment "experiments/B16_svd_rank_sweep/rank_k8_late_3_blocks.yaml"
        ;;
    late5)
        launch_experiment "experiments/B16_svd_rank_sweep/rank_k8_late_5_blocks.yaml"
        ;;
    *)
        echo "Usage: $0 {all|pack_a|pack_b|k1|k2|k4|k8|k16|late3|late5} [REGION] [PROJECT]"
        echo ""
        echo "Arguments:"
        echo "  EXPERIMENT_TYPE  - Which experiments to launch (required)"
        echo "  REGION           - GCP region (default: asia-southeast1)"
        echo "  PROJECT          - W&B project name (default: B16-svd-sweep)"
        echo ""
        echo "Options:"
        echo "  all          - Launch ALL experiments (Pack A + Pack B = 7 total)"
        echo "  pack_a       - Launch Pack A only (k=1,2,4,8,16)"
        echo "  pack_b       - Launch Pack B only (late layers)"
        echo "  k1-k16       - Launch individual rank experiment"
        echo "  late3, late5 - Launch individual late layers experiment"
        echo ""
        echo "Examples:"
        echo "  $0 all"
        echo "  $0 all asia-southeast1"
        echo "  $0 k8 us-central1 my-project"
        exit 1
        ;;
esac
