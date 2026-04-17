#!/bin/bash
set -euo pipefail

BUCKET="gs://teams-faces-data-test-2914-fake-4420-real-feb-28"
DEST="${BUCKET}/real"

TOTAL=0

upload_dir() {
    local src_dir="$1"
    local prefix="$2"
    local count=0

    while IFS= read -r -d '' file; do
        basename=$(basename "$file")
        dest_name="${prefix}__${basename}"
        gsutil -q cp "$file" "${DEST}/${dest_name}"
        count=$((count + 1))
        echo "  [${count}] ${dest_name}"
    done < <(find "$src_dir" -type f \( -iname '*.jpg' -o -iname '*.png' -o -iname '*.jpeg' \) -print0)

    echo "=> Uploaded ${count} images with prefix '${prefix}'"
    TOTAL=$((TOTAL + count))
}

echo "=== 1/5: faces 4 - rael dor not flaged (as real_dor) ==="
upload_dir '/Users/roeedar/Downloads/faces 4 - rael dor not flaged' 'real_dor'

echo ""
echo "=== 2/5: faces 4 / Roy D ==="
upload_dir '/Users/roeedar/Downloads/faces 4/Roy D' 'Roy_D'

echo ""
echo "=== 3/5: faces 4 / dor shkedi ==="
upload_dir '/Users/roeedar/Downloads/faces 4/dor shkedi' 'dor_shkedi'

echo ""
echo "=== 4/5: faces 3 / dor ==="
upload_dir '/Users/roeedar/Downloads/faces 3/dor' 'dor'

echo ""
echo "=== 5/5: faces 3 / xiang ==="
upload_dir '/Users/roeedar/Downloads/faces 3/xiang' 'xiang'

echo ""
echo "==============================="
echo "TOTAL UPLOADED: ${TOTAL}"
echo "==============================="
