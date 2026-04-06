#!/usr/bin/env bash
# ============================================================================
# download_audit_data.sh — Pull face crops from GCS for lighting audit
#
# Creates two directories:
#   audit_data/df40_crops/     → ~250 DF40 training crops (real + 7 fake methods)
#   audit_data/teams_real/     → ~600 Teams-v2 real frames (production proxy)
#
# Usage:
#   cd DeepfakeBench/training
#   bash tools/download_audit_data.sh [OUTPUT_DIR]
# ============================================================================
set -uo pipefail

OUT="${1:-audit_data}"

# Conservative gsutil parallelism to avoid 429 rate limits
GSUTIL_OPTS=(-o "GSUtil:parallel_thread_count=4" -o "GSUtil:parallel_process_count=2")

DF40_BUCKET="gs://df40-frames-recropped-rfa85"
TEAMS_BUCKET="gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2"

DF40_METHODS=(simswap facedancer blendface e4s inswap mobileswap uniface)

echo "=== Lighting Audit Data Download ==="
echo "Output directory: ${OUT}"
echo ""

mkdir -p "${OUT}/df40_crops" "${OUT}/teams_real"

# ------------------------------------------------------------------
# Helper: download files from a GCS folder with unique local names.
# Copies to a temp staging dir, renames with a tag prefix, moves to dest.
#
# Usage: download_prefixed TAG GCS_FOLDER DEST_DIR [MAX_FILES]
# Echoes the count downloaded.
# ------------------------------------------------------------------
download_prefixed() {
    local tag="$1"
    local gcs_folder="$2"  # must end with /
    local dest_dir="$3"
    local max_files="${4:-999}"
    local tmpdir="${dest_dir}/.dl_${tag}_$$"
    mkdir -p "${tmpdir}"

    # Try both png and jpg; suppress errors for missing patterns
    gsutil "${GSUTIL_OPTS[@]}" -m cp "${gcs_folder}*.png" "${tmpdir}/" 2>/dev/null || true
    gsutil "${GSUTIL_OPTS[@]}" -m cp "${gcs_folder}*.jpg" "${tmpdir}/" 2>/dev/null || true

    # Rename with tag prefix and move to dest
    local count=0
    for f in "${tmpdir}"/*; do
        [[ -f "${f}" ]] || continue
        [[ ${count} -ge ${max_files} ]] && break
        mv "${f}" "${dest_dir}/${tag}__$(basename "${f}")"
        count=$((count + 1))
    done
    rm -rf "${tmpdir}"
    echo "${count}"
}

# ==================================================================
# 1. DF40 REAL FRAMES (~150)
# ==================================================================
echo "[1/3] DF40 real frames..."

# The bucket has: real/FaceForensics++/, real/YouTube-real/, etc.
# List dataset dirs first
echo "      Listing real dataset dirs..."
REAL_DATASETS=$(gsutil ls "${DF40_BUCKET}/real/" 2>/dev/null || true)
echo "      Datasets: $(echo "${REAL_DATASETS}" | grep -c '/') found"

REAL_TOTAL=0
IDENTITIES_DONE=0
TARGET_IDENTITIES=10
FRAMES_PER_ID=15

for dataset_dir in ${REAL_DATASETS}; do
    [[ ${IDENTITIES_DONE} -ge ${TARGET_IDENTITIES} ]] && break

    dataset_name=$(basename "${dataset_dir}")
    echo "      Dataset: ${dataset_name}"

    # List identity folders, sample some
    REMAINING=$((TARGET_IDENTITIES - IDENTITIES_DONE))
    ID_DIRS=$(gsutil ls "${dataset_dir}" 2>/dev/null | cat \
        | awk -v s=42 'BEGIN{srand(s)} {print rand(), $0}' \
        | sort -k1,1n | head -n "${REMAINING}" | cut -d' ' -f2- || true)

    for id_dir in ${ID_DIRS}; do
        [[ ${IDENTITIES_DONE} -ge ${TARGET_IDENTITIES} ]] && break
        id_name=$(basename "${id_dir}")
        tag="real_${dataset_name}_${id_name}"

        n=$(download_prefixed "${tag}" "${id_dir}" "${OUT}/df40_crops" "${FRAMES_PER_ID}")
        REAL_TOTAL=$((REAL_TOTAL + n))
        IDENTITIES_DONE=$((IDENTITIES_DONE + 1))
        echo "        ${id_name}: ${n} frames"
        sleep 0.5
    done
done

echo "      Real total: ${REAL_TOTAL}"

# ==================================================================
# 2. DF40 FAKE FRAMES (~105, 15 per method)
# ==================================================================
echo ""
echo "[2/3] DF40 fake frames (7 methods × ~15)..."

FAKE_TOTAL=0
for method in "${DF40_METHODS[@]}"; do
    echo "      ${method}..."

    # List identity-pair folders, sample 3
    ID_DIRS=$(gsutil ls "${DF40_BUCKET}/fake/${method}/" 2>/dev/null | cat \
        | awk -v s=42 'BEGIN{srand(s)} {print rand(), $0}' \
        | sort -k1,1n | head -n 3 | cut -d' ' -f2- || true)

    for id_dir in ${ID_DIRS}; do
        id_name=$(basename "${id_dir}")
        tag="fake_${method}_${id_name}"

        n=$(download_prefixed "${tag}" "${id_dir}" "${OUT}/df40_crops" 5)
        FAKE_TOTAL=$((FAKE_TOTAL + n))
        echo "        ${id_name}: ${n} frames"
    done
    sleep 1  # brief pause between methods
done

echo "      Fake total: ${FAKE_TOTAL}"
DF40_TOTAL=$(find "${OUT}/df40_crops" -type f | wc -l | tr -d ' ')
echo ""
echo "      DF40 combined: ${DF40_TOTAL} crops"

# ==================================================================
# 3. TEAMS REAL FRAMES (~600 from 30 diverse samples)
# ==================================================================
echo ""
echo "[3/3] Teams-v2 real frames (30 diverse samples)..."

echo "      Listing all sample dirs..."
ALL_SAMPLES=$(gsutil ls "${TEAMS_BUCKET}/samples/" 2>/dev/null || true)
TOTAL_SAMPLES=$(echo "${ALL_SAMPLES}" | grep -c '/' || echo 0)
echo "      Found ${TOTAL_SAMPLES} samples"

if [[ ${TOTAL_SAMPLES} -eq 0 ]]; then
    echo "      WARNING: No samples found — check bucket access."
    TEAMS_TOTAL=0
else
    # Deterministic sample of 30
    SAMPLED=$(echo "${ALL_SAMPLES}" \
        | awk -v s=42 'BEGIN{srand(s)} {print rand(), $0}' \
        | sort -k1,1n | head -n 30 | cut -d' ' -f2- || true)

    TEAMS_TOTAL=0
    SIDX=0
    for sample_dir in ${SAMPLED}; do
        SIDX=$((SIDX + 1))
        sample_name=$(basename "${sample_dir}")

        n=$(download_prefixed "${sample_name}" "${sample_dir}frames/real/" "${OUT}/teams_real" 999)
        TEAMS_TOTAL=$((TEAMS_TOTAL + n))
        echo "      [${SIDX}/30] ${sample_name}: ${n} real frames"
        sleep 0.5
    done
fi

echo "      Teams real total: ${TEAMS_TOTAL}"

# ==================================================================
# Summary
# ==================================================================
echo ""
echo "=== Download Complete ==="
echo "  DF40 training crops:  ${DF40_TOTAL} files → ${OUT}/df40_crops/"
echo "  Teams real frames:    ${TEAMS_TOTAL} files → ${OUT}/teams_real/"
echo ""
echo "Next: run the audit"
echo "  python tools/lighting_showcase.py audit \\"
echo "    --images-dir ${OUT}/df40_crops \\"
echo "    --real-captures-dir ${OUT}/teams_real \\"
echo "    --shadow-p 0.10 --gamma-up-p 0.12 \\"
echo "    --output ${OUT}/r13_lighting_audit.png"
