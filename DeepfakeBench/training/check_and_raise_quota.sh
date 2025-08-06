#!/usr/bin/env bash
# Lists CustomJob concurrency + GPU quotas and opens the
# web form to request more if any are zero.
set -euo pipefail
PROJECT=$(gcloud config get-value project)

echo "== Vertex AI CustomJob concurrency =="
gcloud ai regions list --format="table(locationId, quotas[metric=custom-jobs-concurrent-capacity].limit, quotas[metric=custom-jobs-concurrent-capacity].usage)"

echo -e "\n== GPU quotas =="
gcloud compute regions list --format="value(name)" | while read -r R; do
  Q=$(gcloud compute regions describe "$R" --format="value(quotas[metric=ACCELERATOR_GPU_A100].limit)")
  [[ -n $Q ]] && echo "$R : $Q"
done

read -p $'\nRequest quota increase? (y/N) ' yn
if [[ $yn =~ ^[Yy]$ ]]; then
  xdg-open "https://console.cloud.google.com/iam-admin/quotas?project=${PROJECT}"
fi
