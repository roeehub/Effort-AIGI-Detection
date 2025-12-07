#!/bin/bash
# Quick test script for four-model ensemble with real servers

echo "=================================="
echo "Four-Model Ensemble - Real Server Test"
echo "=================================="
echo ""

# Test with real frames
python test_four_model_fusion.py \
    --participant-name "Test User" \
    --num-frames 27 \
    --image-dir "../../data_collection/real_data_collection/qa_frames"

echo ""
echo "Test complete! Check the logs above for results."
