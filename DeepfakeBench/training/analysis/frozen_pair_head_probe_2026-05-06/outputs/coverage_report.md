# Cached-feature coverage report — FROZEN_PAIR_HEAD_PROBE_2026-05-06

## Summary verdict

**INSUFFICIENT_COVERAGE.** No on-disk cached frozen features cover same-source
`(sample_id, frame_idx)` opposite-label pairs at the density required for the
frozen-head pair-rank probe (target ≥200 pairs with both real and fake feature
vectors present).

The probe cannot be executed against existing caches; a single fresh
feature-extraction pass over a paired manifest is required (blueprint below).

## What was inspected

### `analysis/clip_vs_p8a_viso_2026-05-03/outputs/`

| File | n | Has `frame_path` | Pair-frame overlap |
|---|---|---|---|
| `p8a__features.npz` | 1100 (550 fake + 550 real) | yes | 131 reals only, 0 fakes |
| `clip_b16_raw__features.npz` | 1100 (same frames as P8A) | yes | identical to P8A |

The 1100 frames are split: 550 `viso_fake` (visomaster_enhanced, no co-located
real partner) + 550 `teams_real_dev` (broad teams real pool, no fake partner).
The `frame_path` keys exist, but the fake side draws from the
`visomaster-enhanced-face-cropped-v2` bucket without any matching real-side
partner in the cache. The 131 cached reals overlap with the pair-audit real
pool but the corresponding fake partners were never extracted.

Subject distribution of the 131 overlapping reals:
- xiang_xiang2_feng: 45
- test_cam__s73: 34
- test_cam__s76: 18
- cam_test__s38: 15
- cam_test__s32: 15
- dor_shkedi__s16: 2
- pc_generator__s4: 2

### `analysis/_features_cache_2026-04-30/`

31 `.npz` files covering 800-frame slices for ckpts P8A, P18T, P18C, MCLIOEXB,
SLOT2_GRL, SLOT3_JITTER (final_cls, intermediate layer 0/3/6/9/11, domain_probe,
triptych, prod_honest_180, layer_validation).

**Schema:** all files store only `(features, valid_idx)` — `valid_idx` is
positional within an external sampling manifest. **No `frame_path` is stored
inline.** The companion sampling manifest (typically uploaded under
`output_prefix + "sampling_manifest.json"` in `extract_features_for_probes.py`)
is not present locally for this cache directory; the path-to-feature mapping is
lost. These caches cannot be used for pair lookup without a fresh manifest
join.

The original sampler (`feature_space_2026-04-23/extract_features.py`) draws
`DEFAULT_N_PER_SOURCE = 150` per source bucket from the proper-visomaster
target-domain manifest — by construction it is **not paired**: it samples by
source bucket, not by `(sample_id, frame_idx)` opposite-label pairs.

### `analysis/probe_battery_2026-04-26/`

Holds `extract_features_for_probes.py` and `run_linear_probes.py` — these were
the scripts that produced the n800 caches. They reuse the per-source extraction
pattern, which never preserves the pair structure required here.

## Pair-data structure (what we WOULD need)

Source: `analysis/pair_gap_audit_2026-05-06/outputs/pair_gaps.csv` — 37,327
cross-product pairs over 11 canonical subjects, but the cross-product was
constructed at the **canonical_subject** level (not at `(sample_id,
frame_idx)`), so each row references one of 1812 unique real_paths × 3499
unique fake_paths (5311 unique frames total).

Subject-level pair counts (transport-matched, capped at 4000 cross-product per
subject):

| canonical_subject | n_real | n_fake | n_pairs |
|---|---|---|---|
| dor_local | 568 | 2167 | 4000 |
| xiang_xiang2_feng | 301 | 135 | 4000 |
| test_cam__s73 | 251 | 101 | 4000 |
| test_cam__s76 | 214 | 138 | 4000 |
| extra_xiang | 224 | 73 | 4000 |
| cam_test__s38 | 85 | 85 | 4000 |
| cam_test__s32 | 81 | 235 | 4000 |
| extra_xinghe | 19 | 366 | 4000 |
| pc_generator__s15 | 29 | 91 | 2639 |
| dor_shkedi__s16 | 31 | 78 | 2418 |
| pc_generator__s4 | 9 | 30 | 270 |

**Caveat from the pair-audit summary itself:** "Pairs are CROSS-PRODUCT within
canonical_subject (not frame-level). The inference manifest does not carry
frame-level paired structure (teams fake/real are different sessions;
visomaster_v2_dor has no co-bucketed real)." Therefore, even with full feature
extraction over these 5311 unique paths, the resulting "pairs" remain
canonical-subject-level cross products, not the strict
trainer-style `(sample_id, frame_idx)` semantic-equality pairs.

## Implication for the probe

Three of the probe's three required inputs are missing in the on-disk cache:

1. Frozen features for **fake-side paired frames** — 0 of 3499 unique fake
   paths covered.
2. Frozen features for the full **real-side paired frame pool** — 131 of 1812
   covered.
3. A manifest join for the n800 caches that maps `valid_idx` to `frame_path` —
   not present locally.

Therefore, frozen-feature pair-rank head training cannot be evaluated against
existing caches. Falling back to a CE-only head trained on the existing
1100-frame substrate would not test the question (no pair gradient), and the
existing 1100-frame substrate is the same one already used for `score_*`
columns in the pair-gap audit, where pair-gap behaviour is observed at the
encoder+head level (not the frozen-feature-only level).

See `FINDINGS.md` for the extraction blueprint and the implication for P1.
