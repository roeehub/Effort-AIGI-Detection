#!/bin/bash
# Quick local test for DeepLive training
# Usage: ./test_train_deeplive.sh [experiment_name]
# Example: ./test_train_deeplive.sh deeplive_vit_B16

set -e

EXPERIMENT=${1:-"deeplive_vit_B16"}
MAX_STEPS=${2:-100}  # Default: 100 steps for quick test

echo "========================================"
echo "Testing DeepLive Training"
echo "========================================"
echo "Experiment: $EXPERIMENT"
echo "Max steps: $MAX_STEPS"
echo ""

# Check if config exists
CONFIG_PATH="experiments/${EXPERIMENT}.yaml"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config not found: $CONFIG_PATH"
    echo ""
    echo "Available experiments:"
    ls -1 experiments/deeplive_*.yaml 2>/dev/null || echo "  (none found)"
    exit 1
fi

echo "Config: $CONFIG_PATH"
echo ""

# Run dry-run first
echo "--- Step 1: Dry run ---"
python train_deeplive.py \
    --config "$CONFIG_PATH" \
    --dry-run

echo ""
echo "--- Step 2: Short training run ($MAX_STEPS steps) ---"
python train_deeplive.py \
    --config "$CONFIG_PATH" \
    --max-steps "$MAX_STEPS" \
    --output-dir "outputs/test_${EXPERIMENT}"

echo ""
echo "========================================"
echo "Test completed successfully!"
echo "========================================"
echo "Outputs saved to: outputs/test_${EXPERIMENT}/"
