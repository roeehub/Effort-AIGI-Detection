#!/usr/bin/env bash
# complete_phase_2_backbone.sh — score a BACKBONE run's periodic ckpts on the
# 9-suite contract, then run lex + composite λ=1.0 contract scoring.
#
# Usage: ./complete_phase_2_backbone.sh <RUN_ID> <ARM>
#   ARM = "slotav2" or "t5c"
set -euo pipefail

RUN_ID="${1:?need RUN_ID}"
ARM="${2:?need ARM (slotav2|t5c)}"
STEPS_CSV="${3:-3500}"  # default: score only final ckpt; override with e.g. "500,2500,3500"
REPO="/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
ANALYSIS_DIR="$REPO/analysis/phase_2_backbone_2026-05-22"
SUITES_ROOT="$REPO/analysis/face_pool_scorecard_2026-05-22/_tmp"
REPORTS_DIR="$ANALYSIS_DIR/reports_${ARM}_${RUN_ID}"
CKPT_DIR="$ANALYSIS_DIR/_ckpts_${ARM}_${RUN_ID}"
LOG_DIR="$ANALYSIS_DIR/_score_logs"

mkdir -p "$REPORTS_DIR" "$CKPT_DIR" "$LOG_DIR"
cd "$REPO"

echo "[$(date -u +%FT%TZ)] Listing ckpts for run $RUN_ID (arm=$ARM)..."
ALL_CKPTS=$(gsutil ls "gs://training-job-outputs/best_checkpoints/$RUN_ID/" 2>/dev/null || true)
echo "$ALL_CKPTS"

KEYS=()
KEY_PREFIX="BACKBONE_$(echo $ARM | tr a-z A-Z)_$(echo $RUN_ID | tr a-z A-Z)"

# Steps from $STEPS_CSV (default 3500, the final periodic ckpt)
IFS=',' read -ra STEPS <<< "$STEPS_CSV"
for STEP in "${STEPS[@]}"; do
  CKPT_URI=$(echo "$ALL_CKPTS" | grep -E "periodic_effort_.*_step${STEP}_" | head -1 || true)
  if [ -z "$CKPT_URI" ]; then
    echo "[$(date -u +%FT%TZ)] step=$STEP: not found; skipping"
    continue
  fi
  LOCAL="$CKPT_DIR/$(basename "$CKPT_URI")"
  if [ ! -f "$LOCAL" ]; then
    echo "[$(date -u +%FT%TZ)] downloading step=$STEP"
    gsutil -m cp "$CKPT_URI" "$LOCAL"
  fi
  KEY="${KEY_PREFIX}_STEP${STEP}"
  KEYS+=("$KEY")
  echo "[$(date -u +%FT%TZ)] === scoring step=$STEP key=$KEY ==="
  python "$REPO/analysis/cls_pool_scorer_2026-05-22/score_cls_pool_suites.py" \
    --report-root "$REPORTS_DIR" \
    --ckpt "$LOCAL" \
    --checkpoint-key "$KEY" \
    --suites-root "$SUITES_ROOT" \
    --log-dir "$LOG_DIR" \
    --batch-size 32 \
    --num-workers 4 \
    2>&1 | tee "$LOG_DIR/${KEY}.score.log" | tail -25
done

if [ ${#KEYS[@]} -eq 0 ]; then
  echo "[$(date -u +%FT%TZ)] No ckpts scored; aborting."
  exit 2
fi

KEYS_CSV=$(IFS=,; echo "${KEYS[*]}")

# Build a checkpoint map yaml on the fly
CKPT_MAP="$REPORTS_DIR/_backbone_${ARM}_checkpoint_map.yaml"
{
  echo "# Auto-generated checkpoint map for Phase 2 BACKBONE-$ARM"
  echo "# Run: $RUN_ID"
  echo "# Generated: $(date -u +%FT%TZ)"
  for KEY in "${KEYS[@]}"; do
    STEP_NUM=$(echo "$KEY" | grep -oE 'STEP[0-9]+' | grep -oE '[0-9]+')
    URI=$(echo "$ALL_CKPTS" | grep -E "periodic_effort_.*_step${STEP_NUM}_" | head -1 || true)
    if [ -n "$URI" ]; then echo "${KEY}: \"${URI}\""; fi
  done
} > "$CKPT_MAP"

# Lex policy
LEX_OUT="$REPORTS_DIR/contract_lex"
mkdir -p "$LEX_OUT"
echo "[$(date -u +%FT%TZ)] === contract lex ==="
python "$REPO/arena/score_teams_promotion_contract.py" \
  --report_root "$REPORTS_DIR" \
  --checkpoint_map "$CKPT_MAP" \
  --checkpoints "$KEYS_CSV" \
  --output_dir "$LEX_OUT" \
  --tiebreak_policy lex \
  2>&1 | tee "$LOG_DIR/${ARM}_${RUN_ID}_contract_lex.log" | tail -50 || echo "LEX FAIL"

# Composite λ=1.0
COMP_OUT="$REPORTS_DIR/contract_composite_lambda_1.0"
mkdir -p "$COMP_OUT"
echo "[$(date -u +%FT%TZ)] === contract composite λ=1.0 ==="
python "$REPO/arena/score_teams_promotion_contract.py" \
  --report_root "$REPORTS_DIR" \
  --checkpoint_map "$CKPT_MAP" \
  --checkpoints "$KEYS_CSV" \
  --output_dir "$COMP_OUT" \
  --tiebreak_policy composite \
  --tiebreak_lambda 1.0 \
  2>&1 | tee "$LOG_DIR/${ARM}_${RUN_ID}_contract_composite.log" | tail -50 || echo "COMPOSITE FAIL"

# Sentinel
SENTINEL="$ANALYSIS_DIR/_phase_2_backbone_${ARM}_scoring_complete.json"
cat <<JSON > "$SENTINEL"
{
  "phase": "phase_2_backbone_${ARM}_scoring",
  "status": "scoring_complete",
  "completed_at_utc": "$(date -u +%FT%TZ)",
  "run_id": "$RUN_ID",
  "arm": "$ARM",
  "checkpoints_scored": [$(printf '"%s",' "${KEYS[@]}" | sed 's/,$//')],
  "checkpoint_map": "$CKPT_MAP",
  "lex_output_dir": "$LEX_OUT",
  "composite_output_dir": "$COMP_OUT"
}
JSON
echo "[$(date -u +%FT%TZ)] Sentinel: $SENTINEL"
