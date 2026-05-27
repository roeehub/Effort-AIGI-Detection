#!/bin/bash
# ============================================================================
# LAUNCH_MORNING_JOBS.sh
# One-click launcher for the two prepared GPU jobs from the 2026-05-13
# overnight push. Run after waking up.
#
# Usage:
#   ./analysis/best_candidate_search_2026-05-13/LAUNCH_MORNING_JOBS.sh hdtf
#   ./analysis/best_candidate_search_2026-05-13/LAUNCH_MORNING_JOBS.sh t5d
#   ./analysis/best_candidate_search_2026-05-13/LAUNCH_MORNING_JOBS.sh both
# ============================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

JOB="${1:-help}"

case "$JOB" in
  hdtf)
    echo "=== Launching T5C HDTF cross-substrate scorecard ==="
    echo "Estimated cost: \$15-25. Estimated runtime: 4-6h."
    echo "Region: us-east1 (per US-bucket-locality rule)"
    echo
    ./arena/launch_teams_promotion_contract.sh \
        --checkpoint-map arena/checkpoint_maps/teams_target_domain.t5c_hdtf_followup_2026-05-14.yaml \
        --suite-manifest arena/target_domain_suites.proper_data_future.provisional_2026-04-19.yaml \
        --checkpoints T5C_PERIODIC_STEP3500,P8A_REFERENCE_STEP5000,E2B_TOP_N_STEP3200,P2_D_FOURIER_PERIODIC_STEP3000 \
        --job-name t5c-hdtf-followup-2026-05-14 \
        --region us-east1
    ;;

  t5d)
    echo "=== Launching T5D = T5C + Fourier-band-aug improvement training ==="
    echo "Estimated cost: \$30-50. Estimated runtime: 3-5h."
    echo "Region: us-east1"
    echo "Falsifier criteria pre-registered in the yaml header."
    echo
    scripts/launch/launch_experiment.sh -y phase2-experiments us-east1 \
        experiments/phase2_round13/R13_T5D_T5C_PLUS_FOURIER_2026-05-14.yaml
    ;;

  both)
    echo "=== Launching BOTH jobs in parallel (HDTF + T5D training) ==="
    echo "Combined estimated cost: \$45-75"
    echo
    # HDTF first (slot 1)
    ./arena/launch_teams_promotion_contract.sh \
        --checkpoint-map arena/checkpoint_maps/teams_target_domain.t5c_hdtf_followup_2026-05-14.yaml \
        --suite-manifest arena/target_domain_suites.proper_data_future.provisional_2026-04-19.yaml \
        --checkpoints T5C_PERIODIC_STEP3500,P8A_REFERENCE_STEP5000,E2B_TOP_N_STEP3200,P2_D_FOURIER_PERIODIC_STEP3000 \
        --job-name t5c-hdtf-followup-2026-05-14 \
        --region us-east1 &
    # T5D training (slot 2)
    scripts/launch/launch_experiment.sh -y phase2-experiments us-east1 \
        experiments/phase2_round13/R13_T5D_T5C_PLUS_FOURIER_2026-05-14.yaml &
    wait
    ;;

  help|*)
    cat <<EOF
Usage: $0 [hdtf|t5d|both]

  hdtf  — T5C HDTF cross-substrate scorecard. ~\$15-25, 4-6h.
          Closes the one missing data point in the current recommendation.

  t5d   — T5D training: T5C step3500 base + Fourier-band aug.
          Single-lever improvement attempt. ~\$30-50, 3-5h.
          See experiments/phase2_round13/R13_T5D_T5C_PLUS_FOURIER_2026-05-14.yaml
          for the pre-registered falsifier criteria.

  both  — Launch both jobs in parallel (uses both GPU slots).

Each invocation prints the cost+runtime estimate and asks for
final confirmation (per the launch script's --yes flag handling).

Authoritative recommendation: analysis/best_candidate_search_2026-05-13/00_MORNING_RECOMMENDATION.md

To verify everything before launching:
  cat arena/checkpoint_maps/teams_target_domain.t5c_hdtf_followup_2026-05-14.yaml
  cat experiments/phase2_round13/R13_T5D_T5C_PLUS_FOURIER_2026-05-14.yaml
EOF
    ;;
esac
