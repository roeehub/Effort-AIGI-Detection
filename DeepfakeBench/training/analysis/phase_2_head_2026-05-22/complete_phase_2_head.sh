#!/usr/bin/env bash
# complete_phase_2_head.sh — run after full training reaches JOB_STATE_SUCCEEDED.
#
# 1. Discovers periodic ckpts in gs://training-job-outputs/best_checkpoints/<RUN_ID>/
# 2. Downloads them to _ckpts/
# 3. Scores each on 9 promotion-contract suites via score_face_pool_suites.py
# 4. Applies score_teams_promotion_contract.py under lex AND composite λ=1.0
# 5. Writes _phase_2_head_complete.json sentinel
set -euo pipefail

RUN_ID="${1:-kwhju7im}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ANALYSIS_DIR="$REPO_ROOT/analysis/phase_2_head_2026-05-22"
SUITES_ROOT="$REPO_ROOT/analysis/face_pool_scorecard_2026-05-22/_tmp"
REPORTS_DIR="$ANALYSIS_DIR/reports"
CKPT_DIR="$ANALYSIS_DIR/_ckpts"
LOG_DIR="$ANALYSIS_DIR/_score_logs"

mkdir -p "$REPORTS_DIR" "$CKPT_DIR" "$LOG_DIR"

echo "[$(date +%H:%M:%S)] Listing ckpts for run $RUN_ID..."
ALL_CKPTS=$(gsutil ls "gs://training-job-outputs/best_checkpoints/$RUN_ID/" 2>/dev/null || true)
echo "$ALL_CKPTS"

KEYS=()
for STEP in 100 500 1000 1500 2500; do
  CKPT_URI=$(echo "$ALL_CKPTS" | grep -E "periodic_effort_.*_step${STEP}_" | head -1 || true)
  if [ -z "$CKPT_URI" ]; then
    echo "[$(date +%H:%M:%S)] step=$STEP: not found; skipping"
    continue
  fi
  LOCAL="$CKPT_DIR/$(basename "$CKPT_URI")"
  if [ ! -f "$LOCAL" ]; then
    echo "[$(date +%H:%M:%S)] downloading step=$STEP"
    gsutil -m cp "$CKPT_URI" "$LOCAL"
  fi
  KEY="HEAD_${RUN_ID}_STEP${STEP}"
  KEYS+=("$KEY")
  echo "[$(date +%H:%M:%S)] === scoring step=$STEP key=$KEY ==="
  cd "$REPO_ROOT"
  python "$REPO_ROOT/analysis/face_pool_scorecard_2026-05-22/score_face_pool_suites.py" \
    --report-root "$REPORTS_DIR" \
    --ckpt "$LOCAL" \
    --checkpoint-key "$KEY" \
    --suites-root "$SUITES_ROOT" \
    --log-dir "$LOG_DIR" \
    --batch-size 32 \
    --num-workers 4 \
    2>&1 | tee "$LOG_DIR/${KEY}.score.log" | tail -30
done

if [ ${#KEYS[@]} -eq 0 ]; then
  echo "[$(date +%H:%M:%S)] No ckpts scored; aborting."
  exit 2
fi

KEYS_CSV=$(IFS=,; echo "${KEYS[*]}")

# Build a checkpoint map yaml on the fly
CKPT_MAP="$REPORTS_DIR/_head_checkpoint_map.yaml"
{
  echo "# Auto-generated checkpoint map for Phase 2 HEAD"
  echo "# Run: $RUN_ID"
  echo "# Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for KEY in "${KEYS[@]}"; do
    STEP_NUM=$(echo "$KEY" | grep -oE 'STEP[0-9]+' | grep -oE '[0-9]+')
    URI=$(echo "$ALL_CKPTS" | grep -E "periodic_effort_.*_step${STEP_NUM}_" | head -1 || true)
    if [ -n "$URI" ]; then echo "${KEY}: \"${URI}\""; fi
  done
} > "$CKPT_MAP"

# Lex policy
LEX_OUT="$REPORTS_DIR/contract_lex"
mkdir -p "$LEX_OUT"
echo "[$(date +%H:%M:%S)] === contract lex ==="
python "$REPO_ROOT/arena/score_teams_promotion_contract.py" \
  --report_root "$REPORTS_DIR" \
  --checkpoint_map "$CKPT_MAP" \
  --checkpoints "$KEYS_CSV" \
  --output_dir "$LEX_OUT" \
  --tiebreak_policy lex \
  2>&1 | tee "$LOG_DIR/contract_lex.log" | tail -50 || echo "LEX FAIL"

# Composite λ=1.0
COMP_OUT="$REPORTS_DIR/contract_composite_lambda_1.0"
mkdir -p "$COMP_OUT"
echo "[$(date +%H:%M:%S)] === contract composite λ=1.0 ==="
python "$REPO_ROOT/arena/score_teams_promotion_contract.py" \
  --report_root "$REPORTS_DIR" \
  --checkpoint_map "$CKPT_MAP" \
  --checkpoints "$KEYS_CSV" \
  --output_dir "$COMP_OUT" \
  --tiebreak_policy composite \
  --tiebreak_lambda 1.0 \
  2>&1 | tee "$LOG_DIR/contract_composite.log" | tail -50 || echo "COMPOSITE FAIL"

# Write sentinel
SENTINEL="$ANALYSIS_DIR/_phase_2_head_complete.json"
cat <<JSON > "$SENTINEL"
{
  "phase": "phase_2_head",
  "status": "scoring_complete",
  "completed_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "wandb_run_id": "$RUN_ID",
  "wandb_url": "https://wandb.ai/dtect-vision/effort-r13-phase2/runs/$RUN_ID",
  "checkpoints_scored": [$(printf '"%s",' "${KEYS[@]}" | sed 's/,$//' )],
  "checkpoint_map": "$CKPT_MAP",
  "lex_output_dir": "$LEX_OUT",
  "composite_output_dir": "$COMP_OUT",
  "reports_dir": "$REPORTS_DIR",
  "log_dir": "$LOG_DIR"
}
JSON
echo "[$(date +%H:%M:%S)] Sentinel: $SENTINEL"
echo "[$(date +%H:%M:%S)] Done."
