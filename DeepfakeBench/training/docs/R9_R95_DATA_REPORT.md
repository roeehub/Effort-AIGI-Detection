# Round 9 & Round 9.5 — Comprehensive Data Report

> **Purpose:** Full inventory of all training data used in R9 and R9.5 experiments.
> Covers: GCS buckets, structure, metadata, image types, methods/labels, sample counts, identity splitting, weighted sampling, and how it all composes into dataloaders.

---

## Table of Contents

1. [GCS Buckets Overview](#1-gcs-buckets-overview)
2. [Data Sources in Detail](#2-data-sources-in-detail)
   - [DF40 (Academic Deepfakes)](#21-df40-academic-deepfakes)
   - [DeepLive (Studio Capture)](#22-deeplive-studio-capture)
   - [VisoMaster (Studio Capture — Face-Swap Models)](#23-visomaster-studio-capture--face-swap-models)
   - [Teams Passthrough (Codec-degraded)](#24-teams-passthrough-codec-degraded)
   - [External Training Reals (VCD Webcam)](#25-external-training-reals-vcd-webcam)
3. [OOD Monitoring Data (Validation Only)](#3-ood-monitoring-data-validation-only)
4. [Identity System & Splitting](#4-identity-system--splitting)
5. [Family Taxonomy & Weighted Sampling](#5-family-taxonomy--weighted-sampling)
6. [Dataloader Mechanics](#6-dataloader-mechanics)
7. [Experiment Matrix: R9 vs R9.5](#7-experiment-matrix-r9-vs-r95)
8. [Known Issues & Gotchas](#8-known-issues--gotchas)

---

## 1. GCS Buckets Overview

| # | Bucket Name | Role | Image Format | Used in Training |
|---|------------|------|:---:|:---:|
| 1 | `df40-frames-recropped-rfa85` | DF40 academic deepfake frames | PNG | ✅ |
| 2 | `live-deepfake-methods-real-and-fake-frames-cropped` | DeepLive + VisoMaster cropped frames | PNG | ✅ |
| 3 | `live-deepfake-methods-real-and-fake-frames` | Full (uncropped) frames — VisoMaster tier metadata | PNG | Metadata only |
| 4 | `live-teams-deepfake-methods-real-and-fake-frames-cropped` | Teams passthrough frames | **JPG** | ✅ (when enabled) |
| 5 | `effort-collected-data` | External reals (VCD) + OOD monitoring sets | PNG | ✅ (VCD reals) |
| 6 | `base-checkpoints` | CLIP backbone weights | — | Init only |
| 7 | `training-job-outputs` | Training checkpoints (R8_E base, R9/R9.5 outputs) | — | Fine-tune init |

---

## 2. Data Sources in Detail

### 2.1 DF40 (Academic Deepfakes)

**Bucket:** `gs://df40-frames-recropped-rfa85`

**Structure:**
```
real/{source_id}/{target_id}/frame_00000.png ... frame_00031.png
fake/{method}/{target_id}_{source_id}/frame_00000.png ... frame_00031.png
```

**Metadata:** `dataset/df40_pairs/df40-pair-matching.json` (local file, schema v1.0)
 - Contains 11,479 pairs across 17 methods
 - Each pair entry: `real_path`, `fake_path`, `target_identity`, `source_identity`, `orientation`, frame count (32)

**Image type:** PNG, 32 sequential frames per video pair

**Methods used in R9/R9.5 (7 of 17):**

| Method | Orientation | Pairs | Notes |
|--------|:-----------:|------:|-------|
| simswap | target_source | 944 | Largest |
| mobileswap | target_source | 682 | |
| facedancer | target_source | 680 | |
| e4s | target_source | 681 | |
| inswap | target_source | 678 | |
| blendface | target_source | 676 | |
| uniface | target_source | 357 | Smallest |
| **Total** | | **4,698** | |

**Unused methods (10):** fomm, lia, mcnet, one_shot_free, MRAA, pirender, facevid2vid, faceswap, danet, fsgan (all `source_target` orientation except faceswap)

**Sparse sampling:** Indices `[0, 4, 8, 12, 16, 20, 24, 28]` → **8 frames selected** from 32 per pair

**Labels:** Each pair yields 8 real frames (label=0) + 8 fake frames (label=1) → 16 training items per sample

**Identity prefix:** `df40_{target_identity}` — separate identity pool from DeepLive/VisoMaster

**Quality domain:** `0` (clean_academic)

**Landmarks:** ❌ Not available for DF40

---

### 2.2 DeepLive (Studio Capture)

**Bucket:** `gs://live-deepfake-methods-real-and-fake-frames-cropped`

**Structure:**
```
samples/{sample_id}/manifest.json
samples/{sample_id}/frames/real/frame_0000.png ... frame_0015.png
samples/{sample_id}/frames/fake/frame_0000.png ... frame_0015.png
```

**Metadata:** Per-sample `manifest.json` files discovered via GCS listing of `samples/` prefix. Contains: sample_id, strategy, video pairs, landmark availability.

**Image type:** PNG, up to 16 frames per sample

**Strategies used in R9/R9.5 (5):**

| Strategy | Type | Family | Min enforced |
|----------|:----:|--------|:---:|
| edge_cases | Non-enhanced | `deeplive_non_enhanced_fake` | — |
| minimal_processing | Non-enhanced | `deeplive_non_enhanced_fake` | — |
| quality_enhancement | Enhanced | `deeplive_enhanced_fake` | — |
| edge_cases_enhanced | Enhanced | `deeplive_enhanced_fake` | ≥400 |
| minimal_processing_enhanced | Enhanced | `deeplive_enhanced_fake` | ≥390 |

**Approximate sample count:** ~1,000 samples, ~920 unique identities

**Sparse sampling:** Indices `[0, 2, 4, 6, 8, 10, 12, 14]` → **8 frames** from 16

**Labels:** Each sample yields 8 real + 8 fake = 16 training items

**Identity prefix:** `realpool_{video_name}` — **shared** with VisoMaster and Teams

**Quality domain:** `2` (studio_capture)

**Landmarks:** Available (but `use_landmarks: false` in R9/R9.5 configs)

---

### 2.3 VisoMaster (Studio Capture — Face-Swap Models)

**Bucket:** `gs://live-deepfake-methods-real-and-fake-frames-cropped` (same as DeepLive)

**Structure:**
```
samples/visomaster_{swap_model}_{sample_id}/manifest.json
samples/visomaster_{swap_model}_{sample_id}/frames/real/frame_0000.png ... frame_0015.png
samples/visomaster_{swap_model}_{sample_id}/frames/fake/frame_0000.png ... frame_0015.png
```

**Metadata:** Manifest files in GCS with `visomaster_` prefix; tier metadata enriched from the full-frames bucket (`live-deepfake-methods-real-and-fake-frames`)

**Image type:** PNG, up to 16 frames per sample

**Swap models used in R9/R9.5 (all 9):**

| Swap Model | Notes |
|-----------|-------|
| CSCS | |
| GhostFace-v1 | |
| GhostFace-v2 | |
| GhostFace-v3 | |
| InStyleSwapper256-A | |
| InStyleSwapper256-B | |
| InStyleSwapper256-C | |
| Inswapper128 | |
| SimSwap512 | |

**Tier system:** MINIMAL / MODERATE / STRONG — config uses `tiers: null` (all tiers)

**Sparse sampling:** Indices `[0, 2, 4, 6, 8, 10, 12, 14]` → **8 frames** from 16

**Labels:** 8 real + 8 fake = 16 training items per sample

**Identity prefix:** `realpool_{video_name}` — shared with DeepLive and Teams (critical for leak prevention)

**Quality domain:** `2` (studio_capture)

---

### 2.4 Teams Passthrough (Codec-degraded)

**Bucket:** `gs://live-deepfake-methods-real-and-fake-frames-cropped-teams`

**Structure:**
```
samples/{sample_id}/manifest.json
samples/{sample_id}/frames/real/frame_0000.jpg ... frame_NNNN.jpg
samples/{sample_id}/frames/fake/frame_0000.jpg ... frame_NNNN.jpg
```

**Metadata:** Per-sample `manifest.json` with `pair_complete` flag. Only samples where `pair_complete: true` are used (config `require_pair_complete: true`).

**Image type:** ⚠️ **JPG** (unlike all other sources which are PNG) — Teams codec introduces lossy compression artifacts, which is the whole point of this source

**Labels:** Paired real + fake, same sparse sampling as DeepLive

**Sparse sampling:** Indices `[0, 2, 4, 6, 8, 10, 12, 14]` → 8 frames per side

**Identity prefix:** `realpool_{video_name}` — shared pool

**Quality domain:** `1` (webcam_codec)

**Teams-disabled runs:** R9_D (stability-only ablation) and R9_F (sim-aug only)

---

### 2.5 External Training Reals (VCD Webcam)

**Bucket:** `gs://effort-collected-data`

**Structure:**
```
real/VCD/{md5_hash}_frame_{N}.png
```

**Metadata:** Identity extracted via regex: `real__VCD__(?P<md5>[a-f0-9]{32})_`

**Image type:** PNG

**Configuration (all R9/R9.5 runs):**
| Parameter | Value |
|-----------|-------|
| `identity_train_fraction` | 0.40 (40% of VCD identities → training) |
| `identity_split_seed` | 737 |
| `max_frames_per_identity` | 15 |
| `max_total_samples` | 1,200 |
| `deterministic` | true |

**Labels:** Real only (label=0), **unpaired** — no fake counterpart

**Identity prefix:** `external_vcd_{md5}` — separate identity pool

**Quality domain:** `1` (webcam_codec)

---

## 3. OOD Monitoring Data (Validation Only)

These are **not** used in training — they serve as out-of-distribution monitoring during validation:

| Source | Bucket | Prefix | Max Items | Label | Grouping |
|--------|--------|--------|----------:|:-----:|----------|
| YouTube AVSpeech | `effort-collected-data` | `real/external_youtube_avspeech` | 200 | real | by_folder |
| VCD zoom (monitoring) | `effort-collected-data` | `real/VCD` | 1,200 | real | per_image |
| WMA failure fakes | `effort-collected-data` | `wma_validation/enhanced_fake` | 1,202 | fake | per_image |

- **VCD monitoring** excludes the 40% of identities already used in training (`exclude_training_identities: true`)
- Counts are strictly enforced via preflight checks

---

## 4. Identity System & Splitting

### Identity Prefixes

| Prefix | Sources | Prevents |
|--------|---------|----------|
| `df40_{target_id}` | DF40 pairs | DF40 identities leak between splits |
| `realpool_{video_name}` | DeepLive + VisoMaster + Teams | Same real person appearing across splits via different sources |
| `external_vcd_{md5}` | VCD webcam reals | VCD leakage between train & OOD monitoring |

The `realpool_` prefix is the key design choice: DeepLive, VisoMaster, and Teams all share real video sources. A person filmed in DeepLive that also has a VisoMaster face-swap **must** land in the same split, otherwise the model would see their real face in training and their swapped face in validation.

### Split Ratios (all R9/R9.5)

| Split | Fraction | Purpose |
|-------|:--------:|---------|
| Train | 85% of identities | Main training pool |
| Val (in-distribution) | 10% of identities | In-dist validation |
| Test (holdout) | 5% of identities | Not used during training |

- **Split mode:** `identity` (identities are shuffled, then partitioned; all samples for an identity go to the same split)
- **Split seed:** 737 (deterministic across all runs)
- **Leakage check:** Pipeline verifies zero identity overlap between splits

### What Goes Into Each Split

All samples from an identity move together. For example, if identity `realpool_john_001` has:
- 1 DeepLive edge_cases pair
- 1 DeepLive quality_enhancement pair
- 3 VisoMaster swaps (CSCS, GhostFace-v1, Inswapper128)
- 1 Teams pair

...then **all 6 samples** go to the same split (train, val, or test).

---

## 5. Family Taxonomy & Weighted Sampling

### Group → Family Mapping

Samples are first classified into **groups** (fine-grained), then grouped into **families** (coarse) for weighted sampling:

| Group Key | Family Key | Description |
|-----------|-----------|-------------|
| `df40_real` | `df40_real` | DF40 real frames |
| `df40_fake` | `df40_fake` | DF40 fake frames (any of 7 methods) |
| `deeplive_edge_cases_fake` | `deeplive_non_enhanced_fake` | DeepLive non-enhanced |
| `deeplive_minimal_processing_fake` | `deeplive_non_enhanced_fake` | DeepLive non-enhanced |
| `deeplive_quality_enhancement_fake` | `deeplive_enhanced_fake` | DeepLive enhanced |
| `deeplive_edge_cases_enhanced_fake` | `deeplive_enhanced_fake` | DeepLive enhanced |
| `deeplive_minimal_processing_enhanced_fake` | `deeplive_enhanced_fake` | DeepLive enhanced |
| `visomaster_fake` | `visomaster_fake` | All VisoMaster swap models |
| `deeplive_teams_fake` | `deeplive_teams_fake` | Teams passthrough fakes |
| `deeplive_teams_real` | `deeplive_teams_real` | Teams passthrough reals |
| `visomaster_real` / `deeplive_*_real` | `realpool_real` | All non-Teams reals from realpool |
| `external_real` | `external_real` | VCD webcam reals |

### Family Weights by Experiment

**Standard weights (R9_A/B/C/D/E/F/G, R95_A/B/C/D/E):**

| Family Key | Weight | Interpretation |
|-----------|:------:|:---------------|
| `df40_fake` | **0.2** | De-emphasized (academic, less relevant) |
| `df40_real` | **0.5** | |
| `visomaster_fake` | **2.0** | Medium priority |
| `deeplive_non_enhanced_fake` | **2.5** | Medium-high |
| `deeplive_enhanced_fake` | **3.0** | High (enhanced = harder) |
| `deeplive_teams_fake` | **7.0** | Very high (deployment target) |
| `deeplive_teams_real` | **5.0** | High (deployment target) |
| `realpool_real` | **1.5** | Baseline |
| `external_real` | **2.0** | Webcam diversity |

**R9_H (teams extreme):** `deeplive_teams_fake: 10.0`, `deeplive_teams_real: 7.0` (bump Teams even more)

**R95_F (DF40 upweight):** `df40_fake: 1.0`, `df40_real: 1.0` (5× / 2× the standard)

### How Weights Actually Work

⚠️ **Critical detail:** Weights are applied **within each identity**, not globally.

The sampling strategy is `identity_resample_weighted`:

1. **Every epoch**, iterate over ALL training identities (one sample per identity per epoch)
2. For each identity that has **multiple samples** (e.g., 3 VisoMaster swaps + 1 DeepLive + 1 Teams):
   - Each sample's family_key is looked up
   - The family weight is used as probabilities in `rng.choices(..., weights=weights, k=1)`
   - **One sample is selected** for this identity in this epoch
3. For identities with only **one sample** → that sample is always used (weight irrelevant)
4. After selecting one sample per identity, the list is shuffled and sharded across dataloader workers

**Net effect:**
- DF40 identities (each with 1–7 paired fakes across methods) → weight selects which method's pair is used this epoch
- `realpool` identities with both DeepLive and VisoMaster → weight selects whether DeepLive or VisoMaster is shown
- Teams weight of 7.0 means: when an identity has both a Teams pair and other data, Teams is chosen ~7/(7+2+3+2.5) = ~48% of the time

---

## 6. Dataloader Mechanics

### Data Flow

```
GCS Buckets
    ↓
Discovery (enumerate manifests / pair JSON)
    ↓
UnifiedPairedSample pool (all sources)
    ↓
Identity-stratified split (85/10/5)
    ↓
CombinedPairedIterableDataset
    ↓
Identity-balanced sampling (1 sample per identity per epoch)
    ↓ (for each selected sample)
_iterate_{source}_sample()
    ↓
8 frame indices × 2 (real + fake) = 16 dicts per paired sample
8 frame indices × 1 (real only) = 8 dicts per unpaired real sample
    ↓
Each dict: {image, label, quality_domain, method, identity, ...}
    ↓
DataLoader with batch_size = frames_per_batch = 32
```

### Frame Loading Per Source

| Source | Frames Stored | Sparse Indices | Frames Selected | Dicts Yielded |
|--------|:---:|:----|:---:|:---:|
| DF40 | 32 | `[0,4,8,12,16,20,24,28]` | 8 | 16 (8 real + 8 fake) |
| DeepLive | 16 | `[0,2,4,6,8,10,12,14]` | 8 | 16 |
| VisoMaster | 16 | `[0,2,4,6,8,10,12,14]` | 8 | 16 |
| Teams | Varies | `[0,2,4,6,8,10,12,14]` | 8 | 16 |
| VCD Reals | Varies | N/A (up to 8) | ≤8 | ≤8 (real only) |

### Epoch Size

- **Samples per epoch** = number of training identities (~85% of total)
- **Frames per epoch** ≈ `n_training_identities × ~15` (avg paired=16, unpaired=8)
- Training does **not** use epoch-based termination — it runs for `total_training_steps` optimizer steps
- **Batch size (frames):** 32
- So **steps per epoch** ≈ `(n_training_identities × ~15) / 32`

### Worker Sharding

- Multi-worker DataLoader shards by identity: worker `i` gets every `num_workers`-th identity from the shuffled list
- Each worker has deterministic RNG: `seed + epoch * 1000 + worker_id`

---

## 7. Experiment Matrix: R9 vs R9.5

### Context

- **R9** experiments had a **bug** where `stability_lambda` and `label_smoothing` were specified in config but did NOT propagate to the model (config_helpers.py issue)
- **R9.5** = "R9 rerun with stability bug fixed" — the configs are nearly identical but stability params actually work

### R9 Experiments

| Run | Purpose | Teams | Checkpoint | LR | Steps | Label Smooth | Stability λ | Codec Sim | Teams Weight |
|-----|---------|:-----:|-----------|:---:|:-----:|:---:|:---:|:---:|:---:|
| **R9_A** | Baseline + stability | ✅ | R8_E | 5e-5 | 10K | 0.05* | 0.3* | — | 7/5 |
| **R9_B** | No stability fixes | ✅ | R8_E | 5e-5 | 10K | 0.0 | 0.0 | — | 7/5 |
| **R9_C** | Scratch + Teams | ✅ | None | 2e-4 | 12K | 0.05* | 0.3* | — | 7/5 |
| **R9_D** | Stability only (no Teams) | ❌ | R8_E | 5e-5 | 10K | 0.05* | 0.3* | — | — |
| **R9_E** | Heavy regularization | ✅ | R8_E | 5e-5 | 10K | 0.1* | 0.5* | — | 7/5 |
| **R9_F** | Sim aug (no Teams data) | ❌ | R8_E | 5e-5 | 10K | 0.05* | 0.3* | ✅ p=0.15 all | — |
| **R9_G** | Teams + codec sim | ✅ | R8_E | 5e-5 | 10K | 0.05* | 0.3* | ✅ p=0.15 excl teams | 7/5 |
| **R9_H** | Fast adapt + high Teams wt | ✅ | R8_E | 1e-4 | 8K | 0.05* | 0.3* | — | **10/7** |

> \* = These values were **configured but did NOT actually apply** due to the stability bug. All R9 runs effectively ran with `label_smoothing=0` and `stability_lambda=0`, making R9_B the only honest config.

### R9.5 Experiments (Bug Fixed)

| Run | Purpose | Teams | Checkpoint | LR | Steps | Label Smooth | Stability λ | Noise σ | DF40 Weight |
|-----|---------|:-----:|-----------|:---:|:-----:|:---:|:---:|:---:|:---:|
| **R95_A** | True baseline (R9_A fix) | ✅ | R8_E | 5e-5 | 10K | 0.05 | 0.3 | 0.02 | 0.2 |
| **R95_B** | Light stability | ✅ | R8_E | 5e-5 | 10K | 0.0 | 0.1 | 0.02 | 0.2 |
| **R95_C** | Heavy stability | ✅ | R8_E | 5e-5 | 10K | 0.1 | 0.5 | 0.03 | 0.2 |
| **R95_D** | Scratch (no pretrain) | ✅ | None | 2e-4 | 12K | 0.05 | 0.3 | 0.02 | 0.2 |
| **R95_E** | Scale test (baseline) | ✅ | R8_E | 5e-5 | 10K | 0.05 | 0.3 | 0.02 | 0.2 |
| **R95_F** | DF40 upweight | ✅ | R8_E | 5e-5 | 10K | 0.05 | 0.3 | 0.02 | **1.0** |

### Data Config Differences

All R9 and R9.5 runs share **identical** data source configurations:
- Same 7 DF40 methods (target_source orientation)
- Same 5 DeepLive strategies (with enhanced)
- Same 9 VisoMaster swap models (all, no tier filter)
- Same VCD external reals: 40% of identities, max 1200 samples, 15 frames/identity
- Same split seed (737), same split ratios (85/10/5)

**Exceptions:**
| Parameter | Non-default in |
|-----------|---------------|
| Teams disabled | R9_D, R9_F |
| Teams weight bumped to 10/7 | R9_H |
| DF40 weight bumped to 1.0 | R95_F |
| Codec simulation enabled | R9_F (all families), R9_G (excluding teams) |

---

## 8. Known Issues & Gotchas

### Stability Bug (R9)
All R9 experiments (except R9_B which set them to 0 anyway) had `label_smoothing` and `stability_lambda` configured but not propagated. R9.5 is the corrected rerun.

### JPG vs PNG
Teams passthrough is the **only** source using JPG frames. This means the CLIP backbone sees JPG compression artifacts only from Teams (and potentially from codec simulation augmentation in R9_F/R9_G). All other sources provide clean PNG.

### DF40 Has No Landmarks
If you add augmentation pipelines that depend on facial landmarks, they **cannot** be used with DF40 samples. The config already sets `use_landmarks: false` globally for R9/R9.5.

### DF40 Orientations
The 7 selected methods are all `target_source` orientation. The remaining 10 methods (mostly `source_target`) are excluded. If adding new DF40 methods, verify the orientation.

### Shared realpool_ Identity Pool
DeepLive, VisoMaster, and Teams share the `realpool_` prefix. Any new data source using the same real volunteers **must** use this prefix to prevent identity leakage.

### Family Weights Act Within-Identity
Weights don't control global proportions — they control **which sample is selected** when an identity has multiple options. An identity with only DF40 data will always contribute DF40 frames regardless of the 0.2 weight. The weight only matters for mixed identities.

### External Reals Are Unpaired
VCD reals yield only 8 real dicts per sample (vs 16 for paired sources), slightly diluting the fake/real balance.

### Quality Domain Map
Every yielded frame carries a `quality_domain` integer for the optional gradient-reversal head:
| Domain ID | Label | Sources |
|:---------:|-------|---------|
| 0 | clean_academic | DF40 |
| 1 | webcam_codec | VCD reals, Teams passthrough |
| 2 | studio_capture | DeepLive, VisoMaster |
| 3 | social_media | YouTube AVSpeech (OOD only) |

---

## Appendix: Quick Reference — Sample Counts

| Source | Pairs/Samples in Bucket | Used Methods | Est. Training Samples (85% split) | Frames per Sample | Dicts per Sample |
|--------|:-----------------------:|:---:|:---------:|:---:|:---:|
| DF40 | 4,698 pairs (7 methods) | 7 | ~3,990 | 8 real + 8 fake | 16 |
| DeepLive | ~1,000 samples | 5 strategies | ~850 | 8 real + 8 fake | 16 |
| VisoMaster | Dynamic (9 models) | 9 | Dynamic | 8 real + 8 fake | 16 |
| Teams | Dynamic (from bucket) | 1 | Dynamic | 8 real + 8 fake | 16 |
| VCD Reals | ≤1,200 frames | 1 | ≤1,200 | ≤8 real | ≤8 |

> **"Dynamic"** = discovered at runtime from GCS bucket listing. Counts depend on what's been collected and uploaded. Run a smoke test or dry-run to get exact counts.
