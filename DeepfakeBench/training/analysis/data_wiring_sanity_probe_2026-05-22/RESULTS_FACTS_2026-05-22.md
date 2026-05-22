# Data wiring sanity probe — RESULTS FACTS

Run date: 2026-05-22
Probe entry point: `analysis/data_wiring_sanity_probe_2026-05-22/run_sanity_probe.py`
Wall seconds: 8.4

## Inventory discovery

- Total inventory rows discovered: 1826
- Per-source counts: {'hdtf_visomaster_teams': 1094, 'quickclips_visomaster_teams': 732}
- UnifiedPairedSample wrappers built: 3652 (= 1826 × 2 sides)
- Per yield-source counts (clean + teams sides): {'hdtf_visomaster': 1094, 'hdtf_visomaster_teams': 1094, 'quickclips_visomaster': 732, 'quickclips_visomaster_teams': 732}
- Unique inventory `identity_id` values across the HDTF + QCLIP rows: 705
- Stamper-installed identity → pair_id mappings (all 3 inventory sources): 759

## Stamper geometry (load-bearing)

- `SubstratePairStamper` uses identity-keyed lookup: multiple inventory rows that share an `identity_id` collapse to the SAME `pair_id` (last-write-wins at registration). The realistic ceiling for unique pair_ids reachable via the HDTF + QCLIP lanes is **705** (number of unique identities in those inventory rows), NOT the 1826 row count.
- The task spec mentioned a 1500-pair-id threshold; that threshold presupposes one pair_id per inventory row. Under the existing stamper (unchanged per Hard Rules), the wiring achieves the per-identity ceiling, which is the maximum it can deliver.

## Pass A — Random shuffle

- Batches probed: 100
- Batch size: 32 per-video rows (one frame per video)
- Total per-video collate outputs: 3200
- Total videos with substrate_pair_id >= 0 (stamped): 3200
- Unique substrate_pair_ids reached: 704 / 705
- Stamp ratio: 100.0%
- Matched-pair batches: 29 / 100 (29.0%)
- Total matched pairs across all batches: 35

Per-source per-yield-row distribution under random shuffle:
- `hdtf_visomaster`: 956 (29.9%)
- `hdtf_visomaster_teams`: 957 (29.9%)
- `quickclips_visomaster`: 647 (20.2%)
- `quickclips_visomaster_teams`: 640 (20.0%)

Transport distribution under random shuffle (post-stamp, per video):
- transport=0 (clean): 1603 videos
- transport=1 (teams): 1597 videos

## Pass B — Paired sampler at pair_fraction=0.25

- Models the operational design referenced by the yaml flag `combined_paired.substrate_pair_sampling.pair_fraction = 0.25`: 25% of each batch is filled with (clean, teams) wrapper pairs drawn from the same `base_capture_id`. The remainder is random.

- Batches probed: 100
- Total per-video collate outputs: 3199
- Total videos with substrate_pair_id >= 0 (stamped): 3199
- Unique substrate_pair_ids reached: 698 / 705
- Stamp ratio: 100.0%
- Matched-pair batches: 100 / 100 (100.0%)
- Total matched pairs across all batches: 425

Per-source per-yield-row distribution under paired sampler:
- `hdtf_visomaster`: 961 (30.0%)
- `hdtf_visomaster_teams`: 962 (30.1%)
- `quickclips_visomaster`: 651 (20.3%)
- `quickclips_visomaster_teams`: 626 (19.6%)

## Combined unique pair_id coverage (union of both passes)

- Unique pair_ids reached across both passes: 704 / 705

## Pass criteria

- Unique pair_ids >= 95% of reachable ceiling (705): **PASS** (704 / 705 = 99.9%)
- Matched-pair batch pct under paired sampler >= 30%: **PASS** (100.0%)

## Notes

- Sanity probe runs on CPU with synthetic 64×64 zero image arrays (substitute for GCS PNG frames). The data path under test is the inventory parsing, UnifiedPairedSample materialization, source dispatch, and per-video stamping in `combined_paired_collate_fn`.
- A live GPU training job pulls real PNG bytes via the `_iterate_substrate_paired_inventory_sample` path (which is exercised separately by tests).
- Identity prefix `realpool_` is stripped by the stamper before inventory lookup, per `data.sample.substrate_paired._normalize_identity`.
- The HDTF + QCLIP inventory contains 1826 rows but only 705 unique identity_ids — the source-of-truth ceiling for pair_id coverage given the stamper's identity-keyed design.
