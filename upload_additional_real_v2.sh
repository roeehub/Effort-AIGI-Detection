#!/bin/bash
set -euo pipefail

# Create staging directory with renamed symlinks, then parallel upload
STAGING="/tmp/faces_upload_staging"
rm -rf "$STAGING"
mkdir -p "$STAGING"

BUCKET="gs://teams-faces-data-test-2914-fake-4420-real-feb-28"

link_images() {
    local src_dir="$1"
    local prefix="$2"
    local count=0
    while IFS= read -r -d '' file; do
        basename=$(basename "$file")
        dest_name="${prefix}__${basename}"
        # Use hard links (same fs) or cp for cross-fs
        cp "$file" "${STAGING}/${dest_name}"
        count=$((count + 1))
    done < <(find "$src_dir" -type f \( -iname '*.jpg' -o -iname '*.png' -o -iname '*.jpeg' \) -print0)
    echo "Staged ${count} images with prefix '${prefix}'"
}

echo "=== Staging files ==="
link_images '/Users/roeedar/Downloads/faces 4 - rael dor not flaged' 'real_dor'
link_images '/Users/roeedar/Downloads/faces 4/Roy D' 'Roy_D'
link_images '/Users/roeedar/Downloads/faces 4/dor shkedi' 'dor_shkedi'
link_images '/Users/roeedar/Downloads/faces 3/dor' 'dor'
link_images '/Users/roeedar/Downloads/faces 3/xiang' 'xiang'

STAGED_COUNT=$(ls -1 "$STAGING" | wc -l | tr -d ' ')
echo ""
echo "Total staged: ${STAGED_COUNT}"
echo ""

echo "=== Uploading in parallel with gsutil -m ==="
gsutil -m cp "${STAGING}/"* "${BUCKET}/real/"

echo ""
echo "=== Upload complete ==="

# Cleanup
rm -rf "$STAGING"

# Create marker file
echo "=== Creating marker file ==="
REAL_COUNT=$(gsutil ls "${BUCKET}/real/" | wc -l | tr -d ' ')
FAKE_COUNT=$(gsutil ls "${BUCKET}/fake/" | wc -l | tr -d ' ')
MARKER_NAME="${FAKE_COUNT}_FAKE_${REAL_COUNT}_REAL.txt"
echo -n "" | gsutil cp - "${BUCKET}/${MARKER_NAME}"
echo "Created marker: ${MARKER_NAME}"

echo ""
echo "=== Final counts ==="
echo "Real: ${REAL_COUNT}"
echo "Fake: ${FAKE_COUNT}"
