# Project Notes for Claude

## ML Experiment Tracking

Operational rules for managing Vertex AI training/eval jobs.

### Region preference: always US

**Strongly prefer US regions.** Project data lives in US multi-region GCS buckets, so non-US compute eats both cross-region egress charges and a ~10× throughput penalty (US ~3.4 it/s vs asia ~0.31 it/s — see memory `project_gcs_region_locality.md`).

Eligible US regions for A100s on this project, in default-trial order: `us-west4`, `us-east1`, `us-central1`.

Do **not** launch in `asia-*` or `europe-*` regions even when US capacity is tight — instead, queue/retry across the three US options or run jobs in parallel across two US regions and cancel the loser (see below). Falling back to a non-US region requires explicit user authorization and acceptance of the cost/throughput hit.

### Region capacity

If a Vertex job is `PENDING` in a US region for **more than 30 minutes**, switch regions:

1. Relaunch the same job in **another US region** (`us-west4` ↔ `us-east1` ↔ `us-central1`).
2. Monitor the new submission until it is actually `RUNNING` (not just `PENDING`).
3. Once the new run is `RUNNING`, cancel the original pending job.

Do not cancel the original until the replacement has reached `RUNNING` state — avoids losing the slot if the new region also queues.

Optional faster pattern when capacity is contested: launch the same job in two US regions in parallel from the start (with distinct `--job-name` so output paths don't collide); once one transitions to `RUNNING`, cancel whichever is still `PENDING` (or the later-starter if both reached `RUNNING`). Always confirm with the user before cancelling.
