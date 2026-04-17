# Proper Data Capture, Inventory, And Provenance

**Date:** April 17, 2026

## Purpose

Translate the current coordinated capture workflow into a repo-ingestion
contract that preserves future proper data truth.

This document is not a capture-operations guide. It defines what must exist
**before** future proper data is allowed to become a checked-in manifest or a
training/evaluation lane.

## What From The Current Capture Flow Is Reusable

Useful raw artifacts from the current coordinated capture stack:

- `playlist.json`
- `playback_log.jsonl`
- `session_log.json`
- the frame directories produced by `receiver_server.py`

What is **not** sufficient by itself:

- `teams_dataset/<sample_id>/<real|fake>/...`
- any upload that preserves only `sample_id` and side
- any upload step that assumes the missing `upload_teams_to_gcs.py`

Reason:

`sample_id + real|fake` does not preserve:

- clean versus Teams transport
- enhancement status
- generator method
- clean / Teams parallelism for the same base capture
- split hygiene for future lockbox

## Required Artifact Set Per Incoming Wave

Every future proper-data wave should preserve, at minimum:

1. one explicit inventory file using
   `arena/future_proper_data_inventory.template.yaml`
2. raw frame storage for every variant
3. sender playlist
4. sender playback log
5. receiver session log
6. a wave identifier such as `proper_visomaster_wave_2026_04_17`

Without that set, the data is not provenance-clean enough to become a
promotion-relevant manifest.

## Normalization Rule: `sample_id` Is Not The Final Key

Current coordinated capture output should be normalized like this:

- raw `sample_id` -> becomes or maps into `base_capture_id`
- `real|fake` -> becomes only the label field
- transport -> explicit `clean` or `teams`
- enhancement -> explicit `none` or `enhanced`
- fake method -> explicit `generator_method`
- playback route -> explicit `playback_path`

That means one raw sample can expand into multiple future proper variants:

- clean real
- Teams real
- clean fake
- clean enhanced fake
- Teams fake
- Teams enhanced fake

The normalized inventory is the ground truth. The raw capture directory remains
supporting evidence only.

## Split Hygiene Rules

Mandatory rules:

1. assign `split_group_id` before expanding variants
2. keep all variants derived from the same base capture under the same split
3. do not let clean and Teams siblings leak across `dev` and `lockbox`
4. if one identity appears in multiple sessions, decide the grouping rule
   explicitly and write it into `split_group_id`

Practical recommendation:

- when in doubt, use the most conservative grouping that still makes the future
  lockbox meaningful
- default to a group key that includes both identity and clean capture session

## Recommended Storage Layout

The repo does not need to assume one bucket forever, but the path structure
should keep the exact lane explicit.

Recommended pattern:

```text
gs://<bucket>/proper_data/<wave_id>/<base_capture_id>/<lane>/<variant_id>/frame_0000.jpg
```

Why this layout is safer than `sample_id/real|fake`:

- the exact lane is visible in the path
- the variant remains distinct from the base capture
- future clean and Teams variants do not collapse into one folder

## Required Provenance Fields

These fields must survive from capture to manifest:

- `wave_id`
- `base_capture_id`
- `variant_id`
- `identity_id`
- `capture_session_id`
- `split_group_id`
- `label`
- `transport`
- `generator_family`
- `generator_method`
- `enhancement`
- `playback_path`
- `quality_band`
- `face_scale_band`

## Minimal Ingestion Checklist

Before future proper data is accepted:

1. inventory validates through
   `arena/build_future_proper_target_domain_manifest.py`
2. resulting rows use only `proper_*` exact lanes
3. clean and Teams variants remain explicit
4. no row lands in hint lanes or merged VTE aliases
5. future suite names come directly from the canonical slice names in
   `arena/target_domain_suites.proper_data_future.template.yaml`

If any of those checks fail, the wave should stay in raw-capture staging rather
than entering the repo as target-domain truth.
