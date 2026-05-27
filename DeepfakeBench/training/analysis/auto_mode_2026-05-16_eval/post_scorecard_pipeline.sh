#!/bin/bash
# Post-scorecard analysis pipeline — auto-mode 2026-05-16.
#
# Runs after the Vertex scorecard job completes.
# Steps:
#   1. Download per-ckpt summary + per-frame reports from GCS
#   2. Compute per-identity decomposition for Slot A / Slot B / baselines
#   3. Compute band-shortcut readout on new ckpts
#   4. Re-embed Roy_D / dor_shkedi / etc. through new trained encoders
#   5. Write RESULTS_FACTS + AGENT_PROPOSAL retro
#
# Usage: bash post_scorecard_pipeline.sh <SCORECARD_GCS_PREFIX>

set -euo pipefail

SCORECARD_PREFIX="${1:-gs://training-job-outputs/test_results/teams_promotion_contract/auto-mode-2026-05-16/}"
ANALYSIS_DIR="/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis/auto_mode_2026-05-16_eval"
OUT="$ANALYSIS_DIR/scorecard_outputs"
mkdir -p "$OUT"

echo "=== Step 1: Download scorecard artifacts ==="
gsutil -m cp -r "${SCORECARD_PREFIX}promotion_contract" "$OUT/" 2>&1 | tail -5
gsutil -m cp "${SCORECARD_PREFIX}reports/teams_real_all_lockbox_*_frames_report.csv" "$OUT/" 2>&1 | tail -3
gsutil -m cp "${SCORECARD_PREFIX}reports/teams_real_all_dev_*_frames_report.csv" "$OUT/" 2>&1 | tail -3
gsutil -m cp "${SCORECARD_PREFIX}reports/teams_real_dor_dev_*_frames_report.csv" "$OUT/" 2>&1 | tail -3 || true

echo ""
echo "=== Step 2: Per-identity decomposition ==="
python3 "$ANALYSIS_DIR/per_identity_decomp_for_auto_mode.py" --inputs "$OUT"

echo ""
echo "=== Step 3: Band-shortcut readout on new ckpts ==="
python3 "$ANALYSIS_DIR/band_shortcut_new_ckpts.py" --inputs "$OUT"

echo ""
echo "=== Step 4: Re-embed cohorts through Slot A v2 + Slot B trained encoders ==="
# Download Slot A v2 ckpt + (if needed) Slot B ckpt
gsutil cp "gs://training-job-outputs/best_checkpoints/hp35c51p/periodic_effort_*_step3500_*.pth" /tmp/slot_a_v2_ckpt.pth 2>&1 | tail -2 || true
# Note: requires running on Mac with the SVD-aware encoder probe
python3 "$ANALYSIS_DIR/trained_encoder_probe_for_auto_mode.py"

echo ""
echo "=== Done. Outputs at $OUT ==="
ls -la "$OUT"
