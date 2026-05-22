#!/bin/bash
# Post-processing for the face-pool full scorecard rerun (2026-05-22).
#
# Waits for the face-pool scoring summary JSON to appear (signals all 9
# videos_reports are written), then:
#   1. Runs arena/score_teams_promotion_contract.py under lex policy
#   2. Runs the same scorer with --tiebreak_policy=composite --tiebreak_lambda 1.0
#   3. Writes RESULTS_FACTS_2026-05-22.md (CLS vs face-pool diff)
#   4. Writes AGENT_PROPOSAL_2026-05-22.md
#   5. Writes _scorecard_complete.json sentinel LAST
#
# Reads neither stdin nor stdout-monitor. Designed for nohup + disown.

set -u
set -o pipefail

REPO=/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training
WORK=$REPO/analysis/face_pool_scorecard_2026-05-22
REPORT_ROOT=$WORK/reports
SCORE_LEX=$WORK/scorecard/lex
SCORE_COMP=$WORK/scorecard/composite_lambda_1.0
CKPT_MAP=$REPO/arena/checkpoint_maps/teams_target_domain.slot_a_v2_validation_2026-05-20.yaml
CKPT_KEY=SLOT_A_ANCHOR_AWARE_STEP3500
SUMMARY_JSON=$REPORT_ROOT/_face_pool_scoring_summary_slot_a_anchor_aware_step3500.json
LOGDIR=$WORK/_logs
mkdir -p "$SCORE_LEX" "$SCORE_COMP" "$LOGDIR"

T0=$(date +%s)
echo "[postproc] $(date) — waiting for ALL 9 videos_reports + summary JSON" >> "$LOGDIR/postprocess.log"

# Need: 9 *_videos_report.csv + the summary JSON. The scoring script writes
# the summary AFTER the last suite's videos_report, so checking for both
# {summary AND 9 reports} avoids a race where summary exists from an earlier
# partial run.
REQUIRED_SUITES=(
  teams_real_all_dev
  teams_real_poor_quality_dev
  teams_real_lighting_extreme_dev
  teams_fake_all_dev
  visomaster_enhanced_macro_dev
  deeplive_enhanced_dev
  teams_real_all_lockbox
  teams_fake_all_lockbox
  teams_real_dor_dev
)

# Poll for completion. Hard timeout 12h. Check every 60s.
DEADLINE=$(( T0 + 43200 ))
while true; do
  NOW=$(date +%s)
  # Count present videos_reports
  PRESENT=0
  for s in "${REQUIRED_SUITES[@]}"; do
    if [ -f "$REPORT_ROOT/${s}_slot_a_anchor_aware_step3500_videos_report.csv" ]; then
      PRESENT=$(( PRESENT + 1 ))
    fi
  done
  if [ -f "$SUMMARY_JSON" ] && [ "$PRESENT" -eq 9 ]; then
    break
  fi
  if [ "$NOW" -ge "$DEADLINE" ]; then
    echo "[postproc] TIMEOUT after 12h; reports present=$PRESENT/9; ABORT" >> "$LOGDIR/postprocess.log"
    cat > "$WORK/_scorecard_complete.json" <<EOF
{"status": "aborted", "wall_seconds": $(( NOW - T0 )), "verdict_summary": "scoring did not finish within 12h (reports present=$PRESENT/9)"}
EOF
    exit 2
  fi
  sleep 60
done

echo "[postproc] $(date) — summary JSON present, running scorers" >> "$LOGDIR/postprocess.log"

cd "$REPO"

# 1. Lex policy scorer
python3 arena/score_teams_promotion_contract.py \
  --report_root "$REPORT_ROOT" \
  --checkpoint_map "$CKPT_MAP" \
  --checkpoints "$CKPT_KEY" \
  --output_dir "$SCORE_LEX" \
  --tiebreak_policy lex \
  >> "$LOGDIR/score_lex.log" 2>&1
LEX_RC=$?

# 2. Composite policy scorer
python3 arena/score_teams_promotion_contract.py \
  --report_root "$REPORT_ROOT" \
  --checkpoint_map "$CKPT_MAP" \
  --checkpoints "$CKPT_KEY" \
  --output_dir "$SCORE_COMP" \
  --tiebreak_policy composite \
  --tiebreak_lambda 1.0 \
  >> "$LOGDIR/score_composite.log" 2>&1
COMP_RC=$?

echo "[postproc] $(date) — scorer exit codes lex=$LEX_RC composite=$COMP_RC" >> "$LOGDIR/postprocess.log"

# 3-4. Run the analysis/docs writer
python3 "$WORK/write_results_and_proposal.py" \
  --workdir "$WORK" \
  --cls-baseline-scorecard-gs gs://training-job-outputs/test_results/teams_promotion_contract/slot-a-v2-validation-2026-05-20/promotion_contract/ \
  --ckpt-key "$CKPT_KEY" \
  >> "$LOGDIR/write_results.log" 2>&1
WRITE_RC=$?

T1=$(date +%s)
echo "[postproc] $(date) — write_rc=$WRITE_RC total_wall=$(( T1 - T0 ))s" >> "$LOGDIR/postprocess.log"

# 5. Sentinel — write LAST
STATUS=done
if [ "$LEX_RC" -ne 0 ] || [ "$COMP_RC" -ne 0 ] || [ "$WRITE_RC" -ne 0 ]; then
  STATUS=aborted
fi

# Read verdict snippet from write_results output if available
VERDICT=""
if [ -f "$WORK/_verdict.txt" ]; then
  VERDICT=$(head -1 "$WORK/_verdict.txt" | tr -d '"')
fi
if [ -z "$VERDICT" ]; then
  VERDICT="see scorecard/ and RESULTS_FACTS_2026-05-22.md"
fi

cat > "$WORK/_scorecard_complete.json" <<EOF
{
  "status": "$STATUS",
  "wall_seconds": $(( T1 - T0 )),
  "lex_rc": $LEX_RC,
  "composite_rc": $COMP_RC,
  "write_rc": $WRITE_RC,
  "verdict_summary": "$VERDICT"
}
EOF
echo "[postproc] sentinel written; status=$STATUS" >> "$LOGDIR/postprocess.log"
exit 0
