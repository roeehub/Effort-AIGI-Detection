#!/bin/bash
# ============================================================================
# launch_arena.sh
# Launch Model Arena on Vertex AI — cross-checkpoint evaluation
#
# Runs model_arena.py on a GPU instance with all configured checkpoints
# and data sources.
#
# Usage:
#   ./launch_arena.sh                              # Full pipeline (inference + analysis + strategy)
#   ./launch_arena.sh --phase inference             # Inference only
#   ./launch_arena.sh --phase analysis --inference-dir gs://...  # Analysis on pre-existing CSVs
#   ./launch_arena.sh --dry-run                     # Show plan locally (no Vertex AI)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"

# ── Defaults ──
PROJECT="${PROJECT:-train-cvit2}"
IMAGE_URI="us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:$(cat "${TRAINING_DIR}/VERSION")"
REGION="${REGION:-asia-southeast1}"
PHASE="all"
DRY_RUN=""
SMOKE_TEST=""
RESUME=""
INFERENCE_DIR=""
CONFIG="arena_config.yaml"
CHECKPOINTS=""

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)
            PHASE="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN="--dry-run"; shift ;;
        --smoke-test)
            SMOKE_TEST="--smoke-test"; shift ;;
        --resume)
            RESUME="$2"; shift 2 ;;
        --inference-dir)
            INFERENCE_DIR="$2"; shift 2 ;;
        --config)
            CONFIG="$2"; shift 2 ;;
        --checkpoints)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                CHECKPOINTS="${CHECKPOINTS} $1"; shift
            done ;;
        --region)
            REGION="$2"; shift 2 ;;
        --image-uri)
            IMAGE_URI="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

# ── Dry-run: just run locally (no GPU needed) ──
if [[ -n "${DRY_RUN}" ]]; then
    echo "DRY RUN — showing plan locally"
    python "${SCRIPT_DIR}/model_arena.py" --config "${SCRIPT_DIR}/${CONFIG}" --dry-run
    exit 0
fi

# ── Build entrypoint args ──
ARENA_ARGS="--config /workspace/arena/${CONFIG} --phase ${PHASE} --verbose"

if [[ -n "${SMOKE_TEST}" ]]; then
    ARENA_ARGS="${ARENA_ARGS} --smoke-test"
fi

if [[ -n "${RESUME}" ]]; then
    ARENA_ARGS="${ARENA_ARGS} --resume ${RESUME}"
fi

if [[ -n "${INFERENCE_DIR}" ]]; then
    ARENA_ARGS="${ARENA_ARGS} --inference-dir ${INFERENCE_DIR}"
fi

if [[ -n "${CHECKPOINTS}" ]]; then
    ARENA_ARGS="${ARENA_ARGS} --checkpoints ${CHECKPOINTS}"
fi

echo "============================================================"
echo "Model Arena — Vertex AI Launch"
echo "============================================================"
echo "Image:      ${IMAGE_URI}"
echo "Project:    ${PROJECT}"
echo "Region:     ${REGION}"
echo "Phase:      ${PHASE}"
echo "Config:     ${CONFIG}"
echo "Timestamp:  ${TIMESTAMP}"
if [[ -n "${RESUME}" ]]; then
    echo "Resume:     ${RESUME}"
fi
if [[ -n "${INFERENCE_DIR}" ]]; then
    echo "Inf. Dir:   ${INFERENCE_DIR}"
fi
echo ""

"${TRAINING_DIR}/launch_experiment_jobs.sh" \
  --mode train \
  --job-name "arena-${TIMESTAMP}" \
  --project "${PROJECT}" \
  --regions "${REGION}" \
  --image-uri "${IMAGE_URI}" \
  --gpu-type NVIDIA_TESLA_A100 \
  --gpu-count 1 \
  --main-script arena/model_arena.py \
  -- ${ARENA_ARGS}

echo ""
echo "============================================================"
echo "Arena job submitted!"
echo ""
echo "Output will be at:"
echo "  gs://training-job-outputs/arena_results/${TIMESTAMP}/"
echo ""
echo "Expected outputs:"
echo "  inference/     — Per-frame CSVs (model__source.csv)"
echo "  embeddings/    — Backbone feature NPZ files"
echo "  analysis/"
echo "    prob_health.csv              — AUC, EER, threshold health per model×source"
echo "    stability.csv                — Frame-to-frame jitter metrics"
echo "    per_method.csv               — Per-method TPR/AUC breakdown"
echo "    leaderboard.csv              — Composite model ranking"
echo "    arena_report.txt             — Full text report"
echo "  strategy/"
echo "    strategy_grid_all.csv        — All W×T×K combos per model"
echo "    best_strategy_per_model.csv  — Optimal strategy per model"
echo "    strategy_report.txt          — Strategy analysis report"
echo "============================================================"
