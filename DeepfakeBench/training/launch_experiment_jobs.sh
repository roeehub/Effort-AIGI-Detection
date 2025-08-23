#!/usr/bin/env bash
# training/launch_experiment_jobs.sh
# One launcher for single runs and W&B sweeps on Vertex AI.
# Now supports passing extra entrypoint args after a literal `--`
# (they become container args, e.g. `-- --param-config /workspace/train_parameters.yaml`).

set -euo pipefail

# -------- defaults (override via flags) --------
MODE="train"                       # train | sweep
AGENTS=1                           # number of jobs to submit (parallel sweep agents)
DELAY=5                            # seconds between submissions when AGENTS>1
COUNT=5                            # trials per agent (sweep only)
JOB_NAME=""                        # if empty, auto-generate
PREFIX="job"                       # used if JOB_NAME is empty
MAIN_SCRIPT="train_sweep.py"       # program to run in 'train' mode
SWEEP_ID=""                        # required for 'sweep' mode

PROJECT=""                         # --project for gcloud
REGIONS="us-central1,europe-west4,asia-southeast1"
YAML_TEMPLATE="./vertex_job_template.yaml"

IMAGE_URI=""                       # required (or set in your env)
GPU_TYPE="NVIDIA_TESLA_A100"       # matches a2-highgpu-1g
GPU_COUNT="1"
SERVICE_ACCOUNT=""                 # optional; if empty and PROJECT set, we’ll try a sane default

# Extra args that should be forwarded to the container entrypoint
EXTRA_ARGS=()                      # everything after a standalone `--` goes here

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --mode train|sweep [options] [-- <entrypoint extra args>]

Common options:
  --project ID                GCP project for the job
  --regions LST               Comma-separated regions to try (default: $REGIONS)
  --yaml-template PATH        Vertex job template YAML (default: $YAML_TEMPLATE)
  --image-uri URI             Container image URI (required if not baked in)
  --gpu-type TYPE             Vertex accelerator type (default: $GPU_TYPE)
  --gpu-count N               Accelerators per worker (default: $GPU_COUNT)
  --service-account EMAIL     Service account to run the job
  --job-name NAME             Explicit display name (else auto: <prefix>-<timestamp>-<n>)
  --prefix STR                Name prefix for auto names (default: $PREFIX)
  --delay SEC                 Delay between submissions when --agents>1 (default: $DELAY)

Train mode:
  --mode train
  --main-script PATH          Python entrypoint (default: $MAIN_SCRIPT)
  # Pass extra python flags AFTER a literal -- so they reach the container entrypoint, e.g.:
  #   $(basename "$0") --mode train ... -- --param-config /workspace/train_parameters.yaml

Sweep mode:
  --mode sweep
  --sweep-id ID               W&B sweep id (entity/project/XXXX)  [REQUIRED]
  --count N                   Trials per agent (default: $COUNT)
  --agents N                  How many parallel sweep agents to launch (default: $AGENTS)

Env expected (no secrets in files):
  WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT
EOF
}

# -------- parse args --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --mode=*) MODE="${1#*=}"; shift ;;
    --agents) AGENTS="$2"; shift 2 ;;
    --agents=*) AGENTS="${1#*=}"; shift ;;
    --delay) DELAY="$2"; shift 2 ;;
    --delay=*) DELAY="${1#*=}"; shift ;;
    --count) COUNT="$2"; shift 2 ;;
    --count=*) COUNT="${1#*=}"; shift ;;
    --sweep-id) SWEEP_ID="$2"; shift 2 ;;
    --sweep-id=*) SWEEP_ID="${1#*=}"; shift ;;
    --main-script) MAIN_SCRIPT="$2"; shift 2 ;;
    --main-script=*) MAIN_SCRIPT="${1#*=}"; shift ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --job-name=*) JOB_NAME="${1#*=}"; shift ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --prefix=*) PREFIX="${1#*=}"; shift ;;
    --project) PROJECT="$2"; shift 2 ;;
    --project=*) PROJECT="${1#*=}"; shift ;;
    --regions) REGIONS="$2"; shift 2 ;;
    --regions=*) REGIONS="${1#*=}"; shift ;;
    --yaml-template) YAML_TEMPLATE="$2"; shift 2 ;;
    --yaml-template=*) YAML_TEMPLATE="${1#*=}"; shift ;;
    --image-uri) IMAGE_URI="$2"; shift 2 ;;
    --image-uri=*) IMAGE_URI="${1#*=}"; shift ;;
    --gpu-type) GPU_TYPE="$2"; shift 2 ;;
    --gpu-type=*) GPU_TYPE="${1#*=}"; shift ;;
    --gpu-count) GPU_COUNT="$2"; shift 2 ;;
    --gpu-count=*) GPU_COUNT="${1#*=}"; shift ;;
    --service-account) SERVICE_ACCOUNT="$2"; shift 2 ;;
    --service-account=*) SERVICE_ACCOUNT="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;   # everything after -- goes to container args
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
done

# -------- validate --------
[[ -n "$IMAGE_URI" ]] || { echo "ERROR: --image-uri is required"; exit 2; }
case "$MODE" in
  sweep) [[ -n "$SWEEP_ID" ]] || { echo "ERROR: --sweep-id is required for --mode sweep"; exit 2; } ;;
  train) : ;;
  *) echo "ERROR: --mode must be 'train' or 'sweep'"; exit 2 ;;
esac

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WARN: WANDB_API_KEY not set; your container must still get it (template env or Secret Manager)."
fi
if [[ -z "$SERVICE_ACCOUNT" && -n "$PROJECT" ]]; then
  # Use a sensible default if you follow the convention; override with --service-account to customize.
  SERVICE_ACCOUNT="vertex-job-runner-train-cvit2@${PROJECT}.iam.gserviceaccount.com"
fi

# -------- helpers --------
ts() { date +%Y%m%d-%H%M%S; }

# Escape a single argument for YAML double-quoted string
yaml_quote_arg() {
  local s="$1"
  s="${s//\\/\\\\}"   # backslashes
  s="${s//\"/\\\"}"   # double quotes
  printf '"%s"' "$s"
}

render_yaml() {
  local in="$1" out="$2" jobname="$3"
  sed \
    -e "s|{{JOB_NAME}}|$jobname|g" \
    -e "s|{{IMAGE_URI}}|$IMAGE_URI|g" \
    -e "s|{{GPU_TYPE}}|$GPU_TYPE|g" \
    -e "s|{{GPU_COUNT}}|$GPU_COUNT|g" \
    -e "s|{{WANDB_API_KEY}}|${WANDB_API_KEY:-}|g" \
    -e "s|{{WANDB_ENTITY}}|${WANDB_ENTITY:-}|g" \
    -e "s|{{WANDB_PROJECT}}|${WANDB_PROJECT:-}|g" \
    -e "s|{{JOB_MODE}}|$MODE|g" \
    -e "s|{{SWEEP_ID}}|${SWEEP_ID:-"N/A"}|g" \
    -e "s|{{SWEEP_COUNT}}|$COUNT|g" \
    -e "s|{{MAIN_SCRIPT}}|$MAIN_SCRIPT|g" \
    -e "s|{{SERVICE_ACCOUNT}}|$SERVICE_ACCOUNT|g" \
    "$in" > "$out"

  # If extra args were provided, inject them into containerSpec.args
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    local arr=( "--" "${EXTRA_ARGS[@]}" )
    local yaml_list=""
    for a in "${arr[@]}"; do
      if [[ -n "$yaml_list" ]]; then yaml_list+=", "; fi
      yaml_list+="$(yaml_quote_arg "$a")"
    done
    # Replace the placeholder empty list: args: []
    if grep -qE '^[[:space:]]*args:[[:space:]]*\[[[:space:]]*\]' "$out"; then
      sed -i.bak -e "s|^\([[:space:]]*args:\)[[:space:]]*\[[[:space:]]*\]|\1 [ ${yaml_list} ]|" "$out"
      rm -f "$out.bak"
    else
      # Fallback: append under containerSpec if args:[] not present
      awk -v yaml_list="$yaml_list" '
        { print }
        /^ *containerSpec:/ { in_cs=1; next }
        in_cs && /^ *env:/ {
          print "      args: [ " yaml_list " ]"
          in_cs=0
        }
      ' "$out" > "$out.tmp" && mv "$out.tmp" "$out"
    fi
  fi
}

submit_job_once() {
  local name="$1"
  local tmp
  tmp="$(mktemp)"
  render_yaml "$YAML_TEMPLATE" "$tmp" "$name"
  local rc=1
  IFS=',' read -ra RLIST <<< "$REGIONS"
  for region in "${RLIST[@]}"; do
    echo "Submitting '$name' to region $region..."
    if gcloud ai custom-jobs create \
        --project "$PROJECT" \
        --region "$region" \
        --display-name "$name" \
        --config "$tmp"; then
      rc=0; break
    else
      echo "Region $region failed; trying next…"
    fi
  done
  rm -f "$tmp"
  return $rc
}

# -------- main --------
echo "Mode: $MODE  Agents: $AGENTS  Count(per-agent): $COUNT  Image: $IMAGE_URI"
[[ -n "$PROJECT" ]] && echo "Project: $PROJECT"
echo "Template: $YAML_TEMPLATE"
echo "Regions: $REGIONS"
[[ "$MODE" == "train" ]] && echo "Main script: $MAIN_SCRIPT"
[[ "$MODE" == "sweep" ]] && echo "Sweep ID: $SWEEP_ID"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "Entrypoint extra args: ${EXTRA_ARGS[*]}"
fi

pids=()
for i in $(seq 1 "$AGENTS"); do
  n="${JOB_NAME:-${PREFIX}-$(ts)-$i}"
  submit_job_once "$n" &
  pids+=($!)
  (( i < AGENTS )) && sleep "$DELAY"
done

# wait for submissions to finish
fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=1
done
[[ $fail -eq 0 ]] && echo "All $AGENTS job(s) submitted." || { echo "One or more submissions failed."; exit 1; }
