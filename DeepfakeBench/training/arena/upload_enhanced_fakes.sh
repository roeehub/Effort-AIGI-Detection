#!/usr/bin/env bash
# =============================================================================
# Upload VisoMaster-enhanced fake faces (raw + Teams-augmented) to the
# teams_test arena bucket.
#
# Usage:
#   bash upload_enhanced_fakes.sh
#
# Reads from:
#   RAW_DIR  — original VisoMaster-enhanced frames
#   TEAMS_DIR — Teams-codec-augmented copies (produced by apply_teams_augmentation.py)
#
# Uploads to:
#   gs://teams-faces-data-test-2914-fake-4420-real-feb-28/fake/
#
# Naming convention (double-underscore separator, matches existing uploads):
#   visomaster_enhanced_raw__frame_NNNNNN_seqNNNNN.png
#   visomaster_enhanced_teams__frame_NNNNNN_seqNNNNN.png
# =============================================================================
set -euo pipefail

BUCKET="teams-faces-data-test-2914-fake-4420-real-feb-28"
GCS_FAKE="gs://${BUCKET}/fake"

RAW_DIR="${1:-/Users/roeedar/Downloads/enhanced faces viso master/dor shkedi enhanced viso}"
TEAMS_DIR="${2:-/tmp/visomaster_enhanced_teams}"

STAGING="/tmp/enhanced_fakes_staging"

echo "=== Upload VisoMaster Enhanced Fakes ==="
echo "Raw dir:   ${RAW_DIR}"
echo "Teams dir: ${TEAMS_DIR}"
echo "Staging:   ${STAGING}"
echo "Dest:      ${GCS_FAKE}"
echo ""

# Validate inputs
if [ ! -d "${RAW_DIR}" ]; then
    echo "ERROR: Raw directory does not exist: ${RAW_DIR}" >&2
    exit 1
fi
if [ ! -d "${TEAMS_DIR}" ]; then
    echo "ERROR: Teams directory does not exist: ${TEAMS_DIR}" >&2
    echo "Run apply_teams_augmentation.py first." >&2
    exit 1
fi

# Clean staging
rm -rf "${STAGING}"
mkdir -p "${STAGING}"

# Stage raw files with prefix
echo "Staging raw files..."
raw_count=0
for f in "${RAW_DIR}"/*.png "${RAW_DIR}"/*.jpg; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    cp "$f" "${STAGING}/visomaster_enhanced_raw__${base}"
    raw_count=$((raw_count + 1))
done
echo "  → ${raw_count} raw files staged"

# Stage Teams-augmented files with prefix
echo "Staging Teams-augmented files..."
teams_count=0
for f in "${TEAMS_DIR}"/*.png "${TEAMS_DIR}"/*.jpg; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    cp "$f" "${STAGING}/visomaster_enhanced_teams__${base}"
    teams_count=$((teams_count + 1))
done
echo "  → ${teams_count} Teams-augmented files staged"

total=$((raw_count + teams_count))
echo ""
echo "Total to upload: ${total} files"
echo ""

# Upload in parallel
echo "Uploading to ${GCS_FAKE} ..."
gsutil -m cp "${STAGING}"/* "${GCS_FAKE}/"
echo "Upload complete."

# Show final counts
echo ""
echo "=== Post-upload verification ==="
echo "New enhanced fakes:"
gsutil ls "${GCS_FAKE}/visomaster_enhanced_*" 2>/dev/null | wc -l | xargs echo "  visomaster_enhanced_* count:"

echo ""
echo "Total fake count in bucket:"
gsutil ls "${GCS_FAKE}/**" 2>/dev/null | wc -l | xargs echo "  total fakes:"

# Clean up staging
rm -rf "${STAGING}"
echo ""
echo "Done. Staging cleaned up."
