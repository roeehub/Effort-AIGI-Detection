#!/usr/bin/env bash
# run_contract_scoring.sh — apply the promotion contract scorer to the
# face-pool scoring reports under BOTH lex and composite λ=1.0 policies.
#
# Usage:
#   ./run_contract_scoring.sh <RUN_ID> <REPORTS_ROOT> <CHECKPOINTS_CSV>
#
# REPORTS_ROOT is the same dir passed to run_periodic_ckpt_scorecard.sh
# CHECKPOINTS_CSV is a comma-separated list of HEAD_<runid>_STEP<step> keys.
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <RUN_ID> <REPORTS_ROOT> <CHECKPOINTS_CSV>" >&2
  exit 2
fi

RUN_ID="$1"
REPORTS_ROOT="$2"
CHECKPOINTS_CSV="$3"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Build a checkpoint map yaml on the fly from the run + step list.
CKPT_MAP="$REPORTS_ROOT/_head_checkpoint_map.yaml"
echo "# Auto-generated checkpoint map for Phase 2 HEAD" > "$CKPT_MAP"
echo "# Run: $RUN_ID" >> "$CKPT_MAP"
echo "# Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$CKPT_MAP"
IFS=',' read -ra KEYS <<< "$CHECKPOINTS_CSV"
for KEY in "${KEYS[@]}"; do
  STEP_NUM=$(echo "$KEY" | grep -oE 'STEP[0-9]+' | grep -oE '[0-9]+' || true)
  if [ -z "$STEP_NUM" ]; then continue; fi
  # The checkpoint URI matches what was on GCS
  CKPT_URI=$(gsutil ls "gs://training-job-outputs/best_checkpoints/$RUN_ID/" 2>/dev/null | grep -E "periodic_effort_.*_step${STEP_NUM}_" | head -1 || true)
  if [ -n "$CKPT_URI" ]; then
    echo "${KEY}: \"${CKPT_URI}\"" >> "$CKPT_MAP"
  fi
done
echo "Checkpoint map written: $CKPT_MAP"
cat "$CKPT_MAP"

# Run contract scoring under lex policy
LEX_OUT="$REPORTS_ROOT/contract_lex"
mkdir -p "$LEX_OUT"
echo "=== contract lex ==="
python "$REPO_ROOT/arena/score_teams_promotion_contract.py" \
  --report_root "$REPORTS_ROOT" \
  --checkpoint_map "$CKPT_MAP" \
  --checkpoints "$CHECKPOINTS_CSV" \
  --output_dir "$LEX_OUT" \
  --tiebreak_policy lex || echo "LEX FAIL"

# Run contract scoring under composite λ=1.0 policy
COMP_OUT="$REPORTS_ROOT/contract_composite_lambda_1.0"
mkdir -p "$COMP_OUT"
echo "=== contract composite λ=1.0 ==="
python "$REPO_ROOT/arena/score_teams_promotion_contract.py" \
  --report_root "$REPORTS_ROOT" \
  --checkpoint_map "$CKPT_MAP" \
  --checkpoints "$CHECKPOINTS_CSV" \
  --output_dir "$COMP_OUT" \
  --tiebreak_policy composite \
  --tiebreak_lambda 1.0 || echo "COMPOSITE FAIL"

echo "Done. Lex outputs: $LEX_OUT  Composite outputs: $COMP_OUT"
