#!/usr/bin/env bash
# run_periodic_ckpt_scorecard.sh — score each HEAD periodic ckpt with the
# face-pool scorer + the promotion contract (lex + composite λ=1.0).
#
# Usage:
#   ./run_periodic_ckpt_scorecard.sh <RUN_ID> <SUITES_ROOT> <REPORTS_ROOT>
#
# RUN_ID is the W&B run ID from the HEAD full-launch (e.g. abc123).
# SUITES_ROOT is the local dir with the 9 manifest CSVs (re-use
#   analysis/face_pool_scorecard_2026-05-22/_tmp/).
# REPORTS_ROOT is where to write per-ckpt videos_report.csv files.
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <RUN_ID> <SUITES_ROOT> <REPORTS_ROOT>" >&2
  exit 2
fi

RUN_ID="$1"
SUITES_ROOT="$2"
REPORTS_ROOT="$3"
PERIODIC_STEPS=(100 500 1000 1500 2500)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$REPORTS_ROOT"
LOG_DIR="$(dirname "$REPORTS_ROOT")/_score_logs"
mkdir -p "$LOG_DIR"

CKPT_DOWNLOAD_DIR="$REPORTS_ROOT/_ckpts"
mkdir -p "$CKPT_DOWNLOAD_DIR"

# 1. List all periodic ckpts in this run's GCS folder.
echo "Listing ckpts for run $RUN_ID..."
ALL_CKPTS=$(gsutil ls "gs://training-job-outputs/best_checkpoints/$RUN_ID/" 2>/dev/null || true)
echo "$ALL_CKPTS"

for STEP in "${PERIODIC_STEPS[@]}"; do
  CKPT_URI=$(echo "$ALL_CKPTS" | grep -E "periodic_effort_.*_step${STEP}_" | head -1 || true)
  if [ -z "$CKPT_URI" ]; then
    echo "WARN: no ckpt found for step=${STEP} in run ${RUN_ID}; skipping" >&2
    continue
  fi
  LOCAL_CKPT="$CKPT_DOWNLOAD_DIR/$(basename "$CKPT_URI")"
  if [ ! -f "$LOCAL_CKPT" ]; then
    echo "Downloading $CKPT_URI -> $LOCAL_CKPT ..."
    gsutil -m cp "$CKPT_URI" "$LOCAL_CKPT"
  else
    echo "Already local: $LOCAL_CKPT"
  fi
  KEY="HEAD_${RUN_ID}_STEP${STEP}"
  echo "=== scoring step=${STEP} key=${KEY} ==="
  python "$REPO_ROOT/analysis/face_pool_scorecard_2026-05-22/score_face_pool_suites.py" \
    --report-root "$REPORTS_ROOT" \
    --ckpt "$LOCAL_CKPT" \
    --checkpoint-key "$KEY" \
    --suites-root "$SUITES_ROOT" \
    --log-dir "$LOG_DIR" \
    --batch-size 32 \
    --num-workers 4
done

echo "All periodic ckpts scored. Reports under: $REPORTS_ROOT"
