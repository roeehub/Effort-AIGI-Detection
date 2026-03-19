# VisoMaster Enhanced Data Integration — Design Document

> **Status:** Implementation complete (Steps A–G). Experiment config (Step H) pending.
>
> **Goal:** Integrate ~8,923 post-hoc face-enhanced VisoMaster deepfake samples
> into the training pipeline so the model learns to detect enhanced fakes,
> addressing a known weakness of the current best checkpoint (R12_G).

---

## 1. Problem Statement

The current best model (R12_G, AUC 0.9950, OOD 0.9789) is weak on
face-enhanced deepfakes.  Post-hoc enhancement (GFPGAN, CodeFormer, GPEN-BFR,
RestoreFormer++, VQFR) smooths GAN artifacts and increases perceived quality,
causing the model to confuse enhanced fakes with real images.

A new dataset of ~8,923 enhanced samples (8 enhancers × multiple VisoMaster
swap models) is available in GCS.  Training on this data should make the model
robust to enhancement without degrading in-distribution or OOD performance.

## 2. Data Source

| Property | Value |
|---|---|
| Enhanced fakes bucket | `gs://visomaster-enhanced-face-cropped` (project `train-cvit2`) |
| Real frames bucket | `gs://live-deepfake-methods-real-and-fake-frames-cropped` (cross-reference) |
| Samples | ~8,923 enhanced sample directories |
| Enhancers (8) | GFPGAN v1.4, CodeFormer, GPEN-BFR 256/512/1024/2048, RestoreFormer++, VQFR v2 |
| Tier schema | `ARTIFACT`, `STRONG`, `MODERATE`, `MINIMAL` |
| Default excluded tiers | `ARTIFACT` (identity-destroying enhancement) |
| Frames per sample | ~16 |
| Total blobs | ~148,685 |

### Bucket Layout

```
gs://visomaster-enhanced-face-cropped/
  samples/
    visomaster_{SwapModel}_{NNNNN}_enhanced_{enhancer}/
      manifest.json          # tier_data, gcs_paths (cross-refs), enhancer, ...
      frames/
        fake/
          frame_0000.png
          frame_0001.png
          ...
```

Real frames are loaded via cross-bucket reference from the original bucket:
```
gs://live-deepfake-methods-real-and-fake-frames-cropped/
  samples/
    visomaster_{SwapModel}_{NNNNN}/
      frames/
        real/
          frame_0000.png
          ...
```

## 3. Architecture Decisions

### 3.1 Single Family (`visomaster_enhanced_fake`)

All 8 enhancers map to one family (`visomaster_enhanced_fake`) for weighted
sampling.  Individual enhancers are preserved as separate **methods**
(`visomaster_enhanced_gfpgan`, `visomaster_enhanced_codeformer`, etc.) so
per-enhancer metrics are available during validation.

**Rationale:** A single family weight is simpler to tune, and the 8 enhancers
share the same augmentation treatment (heavy degradation to counter smoothing).

### 3.2 Tier Filtering

The enhanced data has a new tier schema where `ARTIFACT` represents
identity-destroying enhancement (face too distorted to be useful).  By default,
`ARTIFACT` samples are excluded.

**Config:** `combined_paired.visomaster_enhanced.exclude_tiers: ["ARTIFACT"]`

### 3.3 Identity Sharing with `realpool_` Prefix

Enhanced samples use the same identity as their original VisoMaster source.
The identity is prefixed with `realpool_` to share the identity namespace with
DeepLive and VisoMaster originals, preventing train/val/test split leakage.

**Example:** If original sample has identity `person_A`, enhanced sample also
gets `realpool_person_A`, ensuring all variants of that person stay in the
same split.

### 3.4 Cross-Bucket Frame Loading

Unlike existing data sources where real and fake frames live in the same bucket,
enhanced data uses **two buckets**:
- Enhanced fake frames from `gs://visomaster-enhanced-face-cropped`
- Real frames from `gs://live-deepfake-methods-real-and-fake-frames-cropped`

The `load_visomaster_enhanced_frames()` function handles this with separate
GCS client handles.

### 3.5 Paired-Only Training (No Triplets)

Only paired training (real + enhanced-fake) is used.  No triplet training
(real + original-fake + enhanced-fake) — keeps the training loop simple.

## 4. Files Changed

### `data/sources/visomaster.py` — Data Model & Discovery

| Addition | Purpose |
|---|---|
| `VisoMasterEnhancedSample` dataclass | Data model for enhanced samples with `enhanced_bucket` + `original_bucket` fields |
| `discover_visomaster_enhanced_samples()` | GCS discovery with tier filtering, enhancer filtering, and manifest-based caching |
| `_load_visomaster_enhanced_cache()` / `_save_visomaster_enhanced_cache()` | Cache helpers (version=2, <12h by default) |
| `load_visomaster_enhanced_frames()` | Cross-bucket frame loading (real from original bucket, fake from enhanced bucket) |

### `data/sources/combined_paired.py` — Pipeline Integration

| Addition | Purpose |
|---|---|
| `create_unified_samples_from_visomaster_enhanced()` | Converts enhanced samples to `UnifiedPairedSample` with identity, method, source |
| Discovery block in `create_combined_paired_pipeline()` | Reads `combined_paired.visomaster_enhanced` config, discovers samples, adds to `all_samples` |
| `_iterate_visomaster_enhanced_sample()` method | Cross-bucket frame loading & yielding in `CombinedPairedDataset.__iter__` |
| `QUALITY_DOMAIN_MAP["visomaster_enhanced"] = 2` | Quality domain assignment (studio_capture) |
| `CombinedBatchingConfig.visomaster_enhanced_sparse_indices` | Configurable frame indices (default: `[0, 2, 4, 6, 8, 10, 12, 14]`) |
| `data_stats["visomaster_enhanced_samples"]` | Stats reporting |

### `utils/grouping.py` — Family Taxonomy

| Addition | Purpose |
|---|---|
| `visomaster_enhanced` routing block (before generic visomaster) | Group key: `visomaster_enhanced_fake` / `visomaster_enhanced_real` |
| Family key mapping | `visomaster_enhanced_fake` → `visomaster_enhanced_fake`, `visomaster_enhanced_real` → `realpool_real` |

### `data/augmentations/pipelines.py` — Augmentation

| Addition | Purpose |
|---|---|
| `visomaster_enhanced_fake` pipeline in `_build_family_quality_pipeline()` | Aggressive degradation (heavy compression, downscale, blur) to counter enhancement smoothing |
| Pipeline registered in `QualityTargetedFamilyRouter._pipelines` | Automatic routing during training |

### `config/defaults.yaml` — Label Dictionary

Added label mappings for all 8 enhanced methods:
```yaml
visomaster_enhanced_real: 0
visomaster_enhanced_gfpgan: 1
visomaster_enhanced_codeformer: 1
visomaster_enhanced_gpen_bfr_256: 1
# ... etc.
```

### `tests/test_phase4_family_pipeline.py` — Tests

| Test | What it verifies |
|---|---|
| Updated `test_group_key_mapping_representative_cases` | Enhanced methods route to correct group/family keys |
| Updated `test_context_variation_forward_pass_all_families` | Augmentation pipeline includes `visomaster_enhanced_fake` |
| Updated `test_vcd_targeted_router_forward_pass_with_context_variation` | Router handles enhanced meta correctly |
| `test_visomaster_enhanced_augmentation_pipeline_all_presets` | Pipeline works for all strength presets |
| `test_visomaster_enhanced_grouping_does_not_collide_with_visomaster` | Enhanced → `visomaster_enhanced_fake`, original → `visomaster_fake` |
| `test_visomaster_enhanced_real_maps_to_realpool` | Real frames share `realpool_real` family |
| `test_visomaster_enhanced_quality_domain_map` | `QUALITY_DOMAIN_MAP` contains `visomaster_enhanced` |
| `test_visomaster_enhanced_router_registered` | Router's `_pipelines` dict includes `visomaster_enhanced_fake` |

## 5. Augmentation Strategy

The `visomaster_enhanced_fake` augmentation pipeline applies **heavier degradation**
than regular `visomaster_fake` because:

1. **Enhancement smooths GAN artifacts** — the model needs to look past the smooth
   surface to find underlying fake structure
2. **Enhanced fakes look more like reals** — aggressive compression/downscale/blur
   creates training diversity that prevents the model from relying on quality cues

Key parameters compared to other families:

| Parameter | `visomaster_fake` | `visomaster_enhanced_fake` | `deeplive_enhanced_fake` |
|---|---|---|---|
| Downscale min | `downscale_min - 0.08` | `downscale_min - 0.10` | `downscale_min - 0.10` |
| JPEG quality lower | `jpeg_lower - 8` | `jpeg_lower - 12` / `jpeg_lower - 16` | `jpeg_lower - 10` / `jpeg_lower - 14` |
| Blur limit upper | `blur_limit[1]` | `blur_limit[1] + 4` | `blur_limit[1] + 2` |
| Degradation probability | `quality_p + 0.12` | `quality_p + 0.24` | `quality_p + 0.22` |
| Sharpen probability | `0.14` | `0.06` | `0.08` |

The enhanced fake pipeline is intentionally the most aggressive because multi-enhancer
outputs (8 different enhancement algorithms) create the most diverse quality space.

## 6. Configuration Interface

### Experiment YAML

```yaml
combined_paired:
  visomaster_enhanced:
    enabled: true
    gcs_bucket: "visomaster-enhanced-face-cropped"
    original_bucket: "live-deepfake-methods-real-and-fake-frames-cropped"
    gcs_project: "train-cvit2"
    exclude_tiers: ["ARTIFACT"]
    enhancers: null  # null = all 8 enhancers
    cache_manifest_uri: "gs://visomaster-enhanced-face-cropped/cache/discovery_manifest.json"
    cache_max_age_hours: 12.0

  sampling:
    strategy: "identity_resample_weighted"
    family_weights:
      # Existing families
      df40_fake: 0.15
      visomaster_fake: 2.5
      deeplive_non_enhanced_fake: 2.5
      deeplive_enhanced_fake: 3.0
      deeplive_teams_fake: 5.0
      deeplive_teams_real: 4.0
      df40_real: 0.4
      realpool_real: 2.0
      external_real: 2.5
      # New family
      visomaster_enhanced_fake: 3.5  # Tune this weight
```

### Config Keys Reference

| Key | Type | Default | Description |
|---|---|---|---|
| `visomaster_enhanced.enabled` | bool | `false` | Enable/disable enhanced data |
| `visomaster_enhanced.gcs_bucket` | str | `visomaster-enhanced-face-cropped` | Enhanced frames bucket |
| `visomaster_enhanced.original_bucket` | str | `live-deepfake-methods-real-and-fake-frames-cropped` | Real frames bucket |
| `visomaster_enhanced.gcs_project` | str | `train-cvit2` | GCP project |
| `visomaster_enhanced.exclude_tiers` | list | `["ARTIFACT"]` | Tiers to exclude |
| `visomaster_enhanced.enhancers` | list/null | `null` (all) | Filter to specific enhancers |
| `visomaster_enhanced.cache_manifest_uri` | str/null | `null` | Discovery cache URI |
| `visomaster_enhanced.cache_max_age_hours` | float | `12.0` | Cache TTL |

## 7. Step H — Experiment Design Notes

When creating the experiment config:

1. **Start from R12_G as base** — the current best checkpoint
2. **Suggested initial family weight: `3.0–4.0`** for `visomaster_enhanced_fake`
   (similar to `deeplive_enhanced_fake`)
3. **Consider increasing `realpool_real` weight slightly** to compensate for the
   additional fake data
4. **Monitor per-enhancer metrics** — the method-level breakdown will show which
   enhancers the model struggles with most
5. **Val/test split follows natural identity split** — enhanced samples end up in
   the same split as their originals (no manual split control needed)
6. **Discovery cache is recommended** for faster startup on Vertex AI:
   ```yaml
   cache_manifest_uri: "gs://visomaster-enhanced-face-cropped/cache/discovery_manifest.json"
   ```

### Suggested Experiment Variants

| Variant | Family Weight | Description |
|---|---|---|
| Baseline | `3.0` | Light enhanced exposure |
| Moderate | `4.0` | Balanced with existing enhanced families |
| Aggressive | `5.5` | Heavy focus on enhanced robustness |
| Sweep | `[2.5, 3.5, 4.5, 5.5]` | W&B sweep to find optimal weight |

## 8. Data Flow Diagram

```
                    ┌─────────────────────────────────────┐
                    │  combined_paired.visomaster_enhanced │
                    │           (experiment YAML)          │
                    └───────────────┬─────────────────────┘
                                    │ enabled=true
                                    ▼
                    ┌─────────────────────────────────────┐
                    │ discover_visomaster_enhanced_samples │
                    │        (visomaster.py)               │
                    │   - GCS list / cached manifest       │
                    │   - tier filtering                   │
                    │   - enhancer filtering               │
                    └───────────────┬─────────────────────┘
                                    │ List[VisoMasterEnhancedSample]
                                    ▼
                    ┌─────────────────────────────────────┐
                    │ create_unified_samples_from_         │
                    │   visomaster_enhanced                │
                    │ - identity = realpool_{person}       │
                    │ - method = visomaster_enhanced_{enh} │
                    │ - source = visomaster_enhanced       │
                    └───────────────┬─────────────────────┘
                                    │ List[UnifiedPairedSample]
                                    ▼
                    ┌─────────────────────────────────────┐
                    │        Identity-based splitting      │
                    │    (shared with DeepLive/VisoMaster) │
                    └───────────────┬─────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
                 train           val            test
                    │
                    ▼
        ┌──────────────────────────────┐
        │ CombinedPairedDataset.__iter__│
        │   route: visomaster_enhanced │
        │   → _iterate_visomaster_     │
        │     enhanced_sample()        │
        └──────────────┬───────────────┘
                       │
            ┌──────────┴──────────┐
            │                     │
            ▼                     ▼
    ┌──────────────┐    ┌──────────────────┐
    │ Original GCS │    │   Enhanced GCS   │
    │  real frames │    │   fake frames    │
    │ (cross-ref)  │    │                  │
    └──────────────┘    └──────────────────┘
            │                     │
            └──────────┬──────────┘
                       ▼
            ┌──────────────────────┐
            │  Augmentation Router │
            │  family_key =        │
            │  visomaster_enhanced │
            │  _fake               │
            │ (heavy degradation)  │
            └──────────────────────┘
```
