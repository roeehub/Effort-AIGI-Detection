#!/usr/bin/env bash
set -euo pipefail

# Launch six R5 overnight experiments sequentially.
# Usage:
#   ./launch_r5_overnight_6.sh <WANDB_PROJECT> [REGION] [VERSION_OVERRIDE]
# Example:
#   ./launch_r5_overnight_6.sh enhanced-aug-test asia-southeast1

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <WANDB_PROJECT> [REGION]"
  exit 1
fi

WANDB_PROJECT="$1"
REGION="${2:-asia-southeast1}"
VERSION_OVERRIDE="${3:-}"
PROJECT="${PROJECT:-train-cvit2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION_FILE="${SCRIPT_DIR}/VERSION"
if [[ ! -f "${VERSION_FILE}" ]]; then
  echo "ERROR: VERSION file not found: ${VERSION_FILE}"
  exit 1
fi
FILE_VERSION="$(tr -d '[:space:]' < "${VERSION_FILE}")"
RUN_VERSION="${VERSION_OVERRIDE:-${FILE_VERSION}}"

if [[ -n "${VERSION:-}" && "${VERSION}" != "${RUN_VERSION}" ]]; then
  echo "WARN: Overriding shell VERSION=${VERSION} with RUN_VERSION=${RUN_VERSION}"
fi
export VERSION="${RUN_VERSION}"
unset IMAGE_URI
IMAGE_URI="us-docker.pkg.dev/${PROJECT}/effort-detector/effort-detector:${VERSION}"

CONFIGS=(
  "experiments/phase2_round5/R5_S1_scratch_baseline_ft7mix.yaml"
  "experiments/phase2_round5/R5_S2_scratch_ft7mix_weighted.yaml"
  "experiments/phase2_round5/R5_S3_scratch_ft7mix_weighted_targetdomain_aug.yaml"
  "experiments/phase2_round5/R5_S1_scratch_baseline_ft7mix_seed1337.yaml"
  "experiments/phase2_round5/R5_S2_scratch_ft7mix_weighted_seed1337.yaml"
  "experiments/phase2_round5/R5_S3_scratch_ft7mix_weighted_targetdomain_aug_seed1337.yaml"
)

echo "Launching 6 R5 overnight runs"
echo "W&B project: ${WANDB_PROJECT}"
echo "Region: ${REGION}"
echo "Image version: ${VERSION}"
echo "Image URI: ${IMAGE_URI}"

if ! gcloud artifacts docker images describe "${IMAGE_URI}" >/dev/null 2>&1; then
  echo "ERROR: Image not found in Artifact Registry: ${IMAGE_URI}"
  echo "Build and push that tag first, or pass an explicit VERSION override as arg #3."
  exit 1
fi

for cfg in "${CONFIGS[@]}"; do
  echo "------------------------------------------------------------"
  echo "Launching: ${cfg}"
  ./launch_experiment.sh -y "${WANDB_PROJECT}" "${REGION}" "${cfg}"
done

echo "------------------------------------------------------------"
echo "Submitted all 6 runs."
