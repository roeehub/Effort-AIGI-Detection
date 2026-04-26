#!/usr/bin/env bash
# Pre-launch check: verify referenced config files are older than the image's
# last push time. Catches the class of bug where a new yaml or checkpoint-map
# is added AFTER the most recent image build — it would not be inside the
# container and would FileNotFoundError at runtime.
#
# Reference: PLAN CHANGE PROPOSAL #2 in april-26-training-master-plan-v2.LOG.md
# (incident: Phase A.2 v1 launch on image 1.3.216 with a newer checkpoint map).
#
# Usage:
#   check_image_currency.sh <IMAGE_URI> <FILE1> [<FILE2> ...]
#
# Each FILE may be a repo-relative path, an absolute host path, or a
# /workspace/-prefixed container path (the prefix is stripped and resolved
# against the training root). gs:// paths are skipped.
#
# Exit codes:
#   0  all files older than image push time, OR check skipped (no gcloud answer,
#      missing files, or SKIP_IMAGE_CURRENCY_CHECK=1).
#   1  at least one file is newer than image push time — launch should abort.
#   2  bad invocation.

set -euo pipefail

if [[ "${SKIP_IMAGE_CURRENCY_CHECK:-0}" == "1" ]]; then
    echo "[image-currency] SKIP_IMAGE_CURRENCY_CHECK=1; skipping check." >&2
    exit 0
fi

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <IMAGE_URI> <FILE1> [<FILE2> ...]" >&2
    exit 2
fi

IMAGE_URI="$1"; shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Split IMAGE_URI into "package" and "tag" for the list query.
# Format is normally <registry>/<project>/<repo>/<image>:<tag>.
image_pkg="${IMAGE_URI%:*}"
image_tag="${IMAGE_URI##*:}"
if [[ "${image_pkg}" == "${IMAGE_URI}" || -z "${image_tag}" ]]; then
    echo "[image-currency] WARN: ${IMAGE_URI} is missing a tag; skipping check." >&2
    exit 0
fi

image_push_iso="$(gcloud artifacts docker images list "${image_pkg}" --include-tags --filter="tags:${image_tag}" --format='value(createTime)' 2>/dev/null | head -n1 || true)"
if [[ -z "${image_push_iso}" ]]; then
    echo "[image-currency] WARN: could not query push time for ${IMAGE_URI}; skipping check." >&2
    exit 0
fi

image_push_epoch="$(python3 -c "import datetime,sys; print(int(datetime.datetime.fromisoformat(sys.argv[1].replace('Z','+00:00')).timestamp()))" "${image_push_iso}" 2>/dev/null || echo "")"
if [[ -z "${image_push_epoch}" ]]; then
    echo "[image-currency] WARN: could not parse image push timestamp '${image_push_iso}'; skipping check." >&2
    exit 0
fi

resolve_local_path() {
    local raw="$1"
    if [[ "${raw}" == gs://* ]]; then
        printf ''
        return 0
    fi
    if [[ "${raw}" == /workspace/* ]]; then
        printf '%s' "${TRAINING_DIR}/${raw#/workspace/}"
        return 0
    fi
    if [[ "${raw}" == /* ]]; then
        printf '%s' "${raw}"
        return 0
    fi
    printf '%s' "${TRAINING_DIR}/${raw}"
}

format_epoch() {
    local epoch="$1"
    date -r "${epoch}" '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null \
        || date -d "@${epoch}" '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null \
        || echo "${epoch}"
}

bad=0
checked=0
for raw in "$@"; do
    local_path="$(resolve_local_path "${raw}")"
    if [[ -z "${local_path}" ]]; then
        continue
    fi
    if [[ ! -f "${local_path}" ]]; then
        echo "[image-currency] WARN: ${local_path} not found locally — cannot verify; assuming inside image." >&2
        continue
    fi
    file_mtime_epoch="$(stat -f %m "${local_path}" 2>/dev/null || stat -c %Y "${local_path}")"
    checked=$((checked + 1))
    if [[ "${file_mtime_epoch}" -gt "${image_push_epoch}" ]]; then
        echo "[image-currency] ERROR: ${local_path}" >&2
        echo "                  mtime $(format_epoch "${file_mtime_epoch}")" >&2
        echo "                  image $(format_epoch "${image_push_epoch}")  (${IMAGE_URI})" >&2
        echo "                  File is NEWER than image push time — it is NOT inside the image." >&2
        bad=1
    fi
done

if [[ "${bad}" -eq 1 ]]; then
    echo "" >&2
    echo "[image-currency] One or more config files are newer than the image." >&2
    echo "[image-currency] Run: ./dev.sh build-prod -y    # then retry the launch." >&2
    echo "[image-currency] Override (advanced): export SKIP_IMAGE_CURRENCY_CHECK=1" >&2
    exit 1
fi

if [[ "${checked}" -gt 0 ]]; then
    echo "[image-currency] OK: ${checked} config file(s) older than image push time." >&2
fi
exit 0
