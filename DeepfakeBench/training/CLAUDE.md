# Project Notes for Claude

## ML Experiment Tracking

Operational rules for managing Vertex AI training/eval jobs.

### Region capacity

If a Vertex job is `PENDING` in a region for **more than 30 minutes**, switch regions:

1. Relaunch the same job in another eligible region.
2. Monitor the new submission until it is actually `RUNNING` (not just `PENDING`).
3. Once the new run is `RUNNING`, cancel the original pending job.

Do not cancel the original until the replacement has reached `RUNNING` state — avoids losing the slot if the new region also queues. Eligible US regions for A100s on this project: `us-west4`, `us-east1`, `us-central1` (us-multi-region buckets keep throughput high — see memory `project_gcs_region_locality.md`).
