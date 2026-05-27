# Pairing semantics in `combined_paired.py`

What "pair" means in the active training loader, plain-English. Useful
for the PE_PAIR_RANK_DRO recipe.

## What gets yielded

Every lane (DF40, DeepLive, VisoMaster, VisoMaster Enhanced, VisoMaster
TeamsEnhanced, proper_data, Teams passthrough) goes through one
of the `_iterate_*_sample` methods of
`CombinedPairedIterableDataset.__iter__`. Every single one of those
methods has the same structural skeleton:

```python
real_frames, fake_frames = <load matched real and fake frames at frame_indices>

for i, frame_idx in enumerate(frame_indices):
    real_img = real_frames[i]
    yield {'image': real_img, 'label': 0, 'sample_id': ..., 'frame_idx': frame_idx, ...}

    fake_img = fake_frames[i]
    yield {'image': fake_img, 'label': 1, 'sample_id': ..., 'frame_idx': frame_idx, ...}
```

So a "pair" is exactly:

> The pair `(real_img, fake_img)` whose dict entries share the same
> `sample_id` AND the same `frame_idx`. Real has `label=0`; fake has
> `label=1`. Both are yielded back-to-back in iteration order.

## Pair key for PE_PAIR_RANK_DRO

The natural pair-id is `(sample_id, frame_idx)`. Two yielded items in a
batch with the same `(sample_id, frame_idx)` and opposite `label` are a
matched pair.

The current `combined_paired_collate_fn` (line 3503) groups by
`(sample_id, label)` for video-style batching but does NOT preserve
the explicit real-fake link as a first-class field. The pair-rank
recipe needs to add a tag (or rely on `(sample_id, frame_idx)` matching
in-batch) so the loss can find matched pairs.

## What is NOT a pair

The `external_training_reals` lane (`UnifiedUnpairedRealSample` →
`_iterate_unpaired_real_sample`) yields ONLY real frames with
`is_unpaired_real=True`. There is no `(sample_id, frame_idx)` that has a
matching `label=1` partner. Pair-rank cannot fire on these. The
collate function and loss path already handle this case: the iterator
sets `label=0` only and the loss continues with standard CE on those
items.

## Loaders that DO yield pairs

| Lane                          | iterator method                                    | pair construction                                                                                                                          |
| ----------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| DF40                          | `_iterate_df40_sample`                             | `df40_dataset.load_sample_frames(sample, frame_indices=...)` returns aligned real/fake frame lists.                                        |
| DeepLive                      | `_iterate_deeplive_sample`                         | `deeplive_dataset.load_sample_frames(sample, frame_indices=...)` returns aligned real/fake frame lists.                                    |
| VisoMaster (V1 base)          | `_iterate_visomaster_sample`                       | `load_visomaster_frames(sample, frame_indices, ...)` returns aligned real/fake frame lists from `gs://<bucket>/samples/<id>/frames/{real,fake}/frame_NNNN.png`. |
| VisoMaster Enhanced           | `_iterate_visomaster_enhanced_sample`              | Cross-bucket: real frames from `original_bucket/samples/<orig_id>/frames/real/...`, enhanced fake frames from `enhanced_bucket/samples/<id>/frames/fake/...`. Matched by `frame_idx`. |
| VisoMaster TeamsEnhanced      | `_iterate_visomaster_teams_enhanced_sample`        | Real side from companion bucket (teams_v2 OR clean fallback); fake side picked at iteration time from one of N+1 branches (original or any of N enhancers). Both real and fake are aligned by `frame_idx`. The fake branch is randomly chosen per epoch with `p_original=0.5`. |
| VisoMaster ResVariant         | `_iterate_visomaster_res_variant_sample`           | Same as VisoMaster Enhanced cross-bucket pattern.                                                                                          |
| Proper Data                   | `_iterate_proper_data_sample`                      | `load_proper_data_frames` returns aligned real/fake from explicit per-capture inventory paths (local or `gs://`).                           |
| Teams Passthrough             | `_iterate_teams_sample`                            | Both real and fake JPGs loaded via `_load_teams_frame_map` from `samples/<id>/frames/{real,fake}/frame_NNNN.jpg`.                          |

Every paired lane uses the same `(sample_id, frame_idx)` pair key.

## Loader that does NOT yield pairs

| Lane                       | iterator method                          | structure                                                       |
| -------------------------- | ---------------------------------------- | --------------------------------------------------------------- |
| `external_training_reals`  | `_iterate_unpaired_real_sample`          | Only `label=0` frames. No `frame_idx`-matched fake counterpart. |

## Subtle points

1. **Per-frame_idx pairing is approximate semantic equality, not
   pixel-equality.** Real and fake frames at the same `frame_idx`
   correspond to the *same source video timestamp* but different
   manipulation paths. Hold approximately constant: identity, source
   video, pose, lighting, crop, capture. Differ: manipulation
   absent/present, plus minor face-detection / resampling drift.

2. **VisoMaster TeamsEnhanced lane is not Teams-transport-pure.** The
   resolver from 2026-04-06 found that of 999 base samples claiming
   teams_v2 transport, only 54 (5.4%) actually resolved to a teams_v2
   real-side companion bucket; the remaining 943 fell back to the
   clean companion. So while the lane yields paired (real, fake), the
   real side is mostly `clean` not `teams_v2`. The companion_domain
   tag in the yielded dict (`'teams_v2'` vs `'clean_fallback'`) marks
   which is which.

3. **The fake-branch of VisoMaster TeamsEnhanced rotates per epoch.**
   Each base sample has up to N enhancers + 1 original-fake branch
   available. The iterator picks 1 per epoch with probability
   p_original for original. So a single base sample yields, across
   epochs, multiple distinct (real, fake) pairs — but only ONE of them
   per forward-pass batch. Pair-rank fires on whichever branch is
   chosen; over training, all branches are seen.

4. **DF40 has 11k pairs total but only 7k active.** The yaml enables
   7 of 17 methods (`simswap`, `facedancer`, `blendface`, `e4s`,
   `inswap`, `mobileswap`, `uniface`). The 10 disabled methods
   (faceswap, MRAA, danet, facevid2vid, fomm, fsgan, lia, mcnet,
   one_shot_free, pirender) account for ~4.7k of the 11.5k pairs.
   Pair-rank loss does not see those pairs — they're filtered out at
   loader-construction time (per `df40_paired.py:methods` check).

5. **Identity-balanced sampling means one base sample per identity per
   epoch.** With ~3000 unique identities across all paired lanes (and
   ~80 in unpaired external), the sampler picks ~3080 base samples
   per epoch. Each paired sample yields 16 frames (8 real + 8 fake).
   So each epoch has roughly 49k frames of which 3.6% are unpaired
   externally-real frames (~80 ids × ~6 frames). On family-weight
   share, the unpaired share is closer to 6.7% per yaml.

## Implications for pair-rank loss design

- **Pair detection.** In-batch, find items that share
  `(sample_id, frame_idx)` AND have opposite label. Both must be
  present in the same forward pass for the loss to fire.
- **Batching constraint.** The current `combined_paired_collate_fn`
  groups by `(sample_id, label)`. This works WITH pair-rank as long
  as the batch construction keeps real and fake from the same
  `sample_id` in the same batch (which is the default — every base
  sample contributes both labels in iteration order). But once the
  collate splits batches, we need a guard: skip the loss when no pair
  is detected.
- **Skip rule.** If a batch contains an unpaired real
  (`is_unpaired_real=True`) or no matching real for a fake, the
  pair-rank term contributes 0 for those items. CE still applies.
- **Per-batch coverage.** With ~93% of batch content being paired,
  the loss fires on the vast majority of items every step. This is
  good — the regularizer has signal on essentially every batch.

## File locations referenced

- `data/sources/combined_paired.py:152-199` — `UnifiedPairedSample` /
  `UnifiedUnpairedRealSample` dataclasses.
- `data/sources/combined_paired.py:2643-2696` —
  `CombinedPairedIterableDataset.__iter__` dispatch.
- `data/sources/combined_paired.py:2813-3438` — per-lane iterators.
- `data/sources/combined_paired.py:3440-3496` —
  `_iterate_unpaired_real_sample` (unpaired path).
- `data/sources/combined_paired.py:3503-...` —
  `combined_paired_collate_fn`.
- `data/sources/visomaster.py:740-797` —
  `VisoMasterTeamsEnhancedSample`.
