#!/usr/bin/env bash
# Launch the Training Data Viewer
# Usage: ./run_viewer.sh [experiment_yaml] [port]
#
# Example:
#   ./run_viewer.sh experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml 8501
#
# All downloaded frames are cached in .viewer_cache/ — delete with:
#   rm -rf .viewer_cache/

set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${1:-experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml}"
PORT="${2:-8501}"

echo "=== Training Data Viewer ==="
echo "Config: $CONFIG"
echo "URL:    http://127.0.0.1:$PORT"
echo "Cache:  .viewer_cache/"
echo ""
echo "To clear downloaded frames: rm -rf .viewer_cache/"
echo "========================================="

python -m viewer.server --config "$CONFIG" --port "$PORT"
