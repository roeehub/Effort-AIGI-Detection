#!/usr/bin/env bash
# Launch overnight 2026-05-16 scorecard — Slot α + Slot β + anchors.
# Mode "full" = all 5 ckpts in the map × all 9 suites in the minimal manifest.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

./arena/launch_teams_promotion_contract.sh \
    --mode full \
    --checkpoint-map arena/checkpoint_maps/teams_target_domain.overnight_2026-05-16.yaml \
    --suite-manifest arena/target_domain_suites.teams_promotion_contract_minimal_9suite_2026-05-14.yaml \
    --region us-east1 \
    --wandb-project phase2r13-experiments \
    --job-name overnight-scorecard-2026-05-16
