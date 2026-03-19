#!/usr/bin/env bash
# End-to-end R8 calibration pipeline:
#   1) (optional) build manifest from folder tree
#   2) score manifest with checkpoint
#   3) fit calibrator + thresholds

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./launch_r8_calibration.sh \
    --checkpoint <local_or_gs://checkpoint.pth> \
    --out_dir <output_dir> \
    [--manifest_csv <existing_manifest.csv> | --data_root <folder_with_real_fake>] \
    [--device auto|cuda|cpu|mps] \
    [--recrop] \
    [--max_frames_per_identity N] \
    [--holdout_fraction 0.25] \
    [--min_fake_tpr 0.85]

Examples:
  ./launch_r8_calibration.sh \
    --checkpoint gs://training-job-outputs/phase2r8_experiments/hu7cen3m/top_n_effort_*.pth \
    --data_root /data/r8_calibration_set \
    --out_dir ./analysis_results/r8_calibration

  ./launch_r8_calibration.sh \
    --checkpoint ./weights/r8e_best.pth \
    --manifest_csv ./analysis_results/r8_manifest.csv \
    --out_dir ./analysis_results/r8_calibration \
    --device cuda
EOF
}

CHECKPOINT=""
OUT_DIR=""
MANIFEST_CSV=""
DATA_ROOT=""
DEVICE="auto"
RECROP=0
MAX_FRAMES_PER_ID=""
HOLDOUT_FRACTION="0.25"
MIN_FAKE_TPR="0.85"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT="$2"; shift 2 ;;
    --out_dir)
      OUT_DIR="$2"; shift 2 ;;
    --manifest_csv)
      MANIFEST_CSV="$2"; shift 2 ;;
    --data_root)
      DATA_ROOT="$2"; shift 2 ;;
    --device)
      DEVICE="$2"; shift 2 ;;
    --recrop)
      RECROP=1; shift 1 ;;
    --max_frames_per_identity)
      MAX_FRAMES_PER_ID="$2"; shift 2 ;;
    --holdout_fraction)
      HOLDOUT_FRACTION="$2"; shift 2 ;;
    --min_fake_tpr)
      MIN_FAKE_TPR="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1 ;;
  esac
done

if [[ -z "$CHECKPOINT" || -z "$OUT_DIR" ]]; then
  echo "ERROR: --checkpoint and --out_dir are required"
  usage
  exit 1
fi

if [[ -z "$MANIFEST_CSV" && -z "$DATA_ROOT" ]]; then
  echo "ERROR: provide either --manifest_csv or --data_root"
  usage
  exit 1
fi

if [[ -n "$MANIFEST_CSV" && -n "$DATA_ROOT" ]]; then
  echo "ERROR: provide only one of --manifest_csv or --data_root"
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$(python - <<'PY' "$OUT_DIR"
import os,sys
print(os.path.abspath(sys.argv[1]))
PY
)"
mkdir -p "$OUT_DIR"

if [[ -n "$DATA_ROOT" ]]; then
  MANIFEST_CSV="$OUT_DIR/calibration_manifest.csv"
  BUILD_CMD=(
    python "$SCRIPT_DIR/build_r8_calibration_manifest.py"
    --root_dir "$DATA_ROOT"
    --output_csv "$MANIFEST_CSV"
  )
  if [[ -n "$MAX_FRAMES_PER_ID" ]]; then
    BUILD_CMD+=(--max_frames_per_identity "$MAX_FRAMES_PER_ID")
  fi

  echo "[1/3] Building manifest..."
  "${BUILD_CMD[@]}"
else
  echo "[1/3] Using existing manifest: $MANIFEST_CSV"
fi

SCORES_CSV="$OUT_DIR/calibration_scores.csv"
FAILURES_CSV="$OUT_DIR/calibration_score_failures.csv"

echo "[2/3] Scoring manifest with checkpoint..."
SCORE_CMD=(
  python "$SCRIPT_DIR/run_r8_score_manifest.py"
  --manifest_csv "$MANIFEST_CSV"
  --output_csv "$SCORES_CSV"
  --failures_csv "$FAILURES_CSV"
  --checkpoint "$CHECKPOINT"
  --device "$DEVICE"
)
if [[ "$RECROP" -eq 1 ]]; then
  SCORE_CMD+=(--recrop)
fi
"${SCORE_CMD[@]}"

echo "[3/3] Fitting calibrator and thresholds..."
python "$SCRIPT_DIR/run_r8_calibration_fit.py" \
  --scores_csv "$SCORES_CSV" \
  --output_dir "$OUT_DIR" \
  --holdout_fraction "$HOLDOUT_FRACTION" \
  --min_fake_tpr "$MIN_FAKE_TPR"

echo

echo "Calibration pipeline complete."
echo "Artifacts:"
echo "  - $OUT_DIR/calibrator_bundle.json"
echo "  - $OUT_DIR/calibration_report.json"
echo "  - $OUT_DIR/split_identities.csv"
echo "  - $OUT_DIR/eval_scored_with_calibration.csv"
echo "  - $SCORES_CSV"
echo "  - $FAILURES_CSV"
