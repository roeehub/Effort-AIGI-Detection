# Sampler pair-grouping fix — sanity probe FACTS (2026-05-23)

Probe entry point: `analysis/sampler_pair_grouping_fix_2026-05-23/run_pair_grouping_probe.py`
Wall seconds: 1.16

## Setup

- Inventory rows loaded: 1826
- UnifiedPairedSample wrappers materialized: 3652
- Identities with (clean, teams) partner duo registered in sampler: 705
- Configured pair_fraction: 0.25 (matches R13_PAIR_LOSS_ASYM_T5C_2026-05-26.yaml)
- batch_size: 32; frames_per_video: 8; n_workers: 0

## Coverage

- Batches run: 50
- Frames consumed: 1600
- Matched-pair batches: 23 / 50 (46.0%)
- substrate_pair_asymmetric_loss > 0 steps: 23 / 50 (46.0%)
- Mean loss value on matched batches: 0.12

## Interpretation (mechanical)

- The asymmetric pair-loss fires on 23 of 50 probed steps. In the pre-fix code path the loss returned 0 on every step across 3 smokes.
- Matched-pair batch fraction (46.0%) is at or above the configured pair_fraction (25%), confirming the sampler emits paired identities as a co-occurring duo.

## Status

- PASS: matched-pair coverage and loss firing both above zero.
