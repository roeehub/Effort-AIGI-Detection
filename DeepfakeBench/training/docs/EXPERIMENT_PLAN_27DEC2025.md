# Experiment Plan - December 27, 2025

**Goal:** Run two parallel experiments to validate refactoring and test new data pipeline.

---

## Experiment A: Baseline Validation (Post-Refactoring Sanity Check)

**Purpose:** Verify that the refactored codebase works with existing setup.

### Configuration
- **Config File:** `orgenize_training/sanity_exp.yaml`
- **Backbone:** CLIP ViT-L-14 (default, 224px)
- **Data Strategy:** Property-based balancing
- **Data Source:** Existing GCS bucket (`df40-frames-recropped-rfa85`)
- **Expected Result:** Should work exactly as before refactoring

### Requirements
✅ No code changes needed (should work out of box)
- Uses existing dataloader: `property_balancing`
- Uses existing augmentation: version 7
- Uses existing methods/weights from `sanity_exp.yaml`

### Action Items
1. [ ] Run dry-run validation: `python train_simple.py --param-config orgenize_training/sanity_exp.yaml --dry-run`
2. [ ] Launch short training run (local or GCP)
3. [ ] Verify metrics/checkpoints are saved correctly
4. [ ] Compare with pre-refactoring baseline (if logs available)

### Success Criteria
- ✅ Training starts without errors
- ✅ Dataloader loads samples correctly
- ✅ Metrics are logged to W&B
- ✅ Checkpoints are saved to GCS
- ✅ Lesson gate logic functions properly

---

## Experiment B: New Data Pipeline (DeepLive Dataset)

**Purpose:** Integrate new paired real/fake frame dataset with custom dataloader.

### Configuration
- **Backbone:** TBD (need specifications from user)
  - Model name: ?
  - GCS path: ?
  - Resolution: ?
  - Hidden size: ?
- **Data Source:** `live-deepfake-methods-real-and-fake-frames` (new GCS bucket)
- **Data Strategy:** NEW custom dataloader (to be implemented)
- **Augmentation:** NEW custom pipeline (to be implemented)

### Data Characteristics (from DEEPLIVE_PIPELINE_DATA_BUCKET.md)
```
Structure:
  gs://live-deepfake-methods-real-and-fake-frames/
  └── samples/{sample_id}/
      ├── manifest.json
      ├── frames/real/frame_XXXX.png (16 frames)
      ├── frames/fake/frame_XXXX.png (16 frames)
      └── landmarks/{real|fake}_landmarks.json
```

**Key Features:**
- 16 frames per sample (8 anchor + 8 consecutive)
- Paired real/fake frames (same temporal moment)
- Face landmarks available (MediaPipe)
- Blendshape features (52D facial expressions)
- Strategies: edge_cases, minimal_processing, quality_enhancement, random_mixed

### Requirements

#### 1. Backbone Configuration
**Status:** 📋 Awaiting user input

Need to specify:
- `backbone.type`: ? (clip, openclip, siglip, dinov2)
- `backbone.variant`: ?
- `backbone.source`: ?
- `backbone.resolution`: ?
- GCS path for weights (if not in backbone_registry)

#### 2. New Dataloader Implementation
**Status:** 🔧 TO IMPLEMENT

**File to create:** `data/batching/strategies/deeplive_strategy.py`

**Requirements from user:**
- Sampling strategy (anchor-only? all 16? pairs?)
- Batch composition (how many samples per batch?)
- Use landmarks for anything? (face cropping, attention masks?)
- Strategy filtering (which strategies to use?)
- Temporal pairing logic (compare anchor vs consecutive?)

**Integration points:**
- Register in `data/batching/registry.py`
- Add strategy name to config (e.g., `dataloader_strategy: deeplive_paired`)

#### 3. New Augmentation Pipeline
**Status:** 🔧 TO IMPLEMENT

**File to create:** `data/augmentations/pipelines/deeplive_augmentations.py`

**Requirements from user:**
- Which augmentations to apply?
- Landmark-guided augmentations? (e.g., eye region manipulation)
- Should preserve facial structure more carefully?
- Different augmentation for real vs fake?
- Use blendshape features for anything?

**Integration points:**
- Register in `data/augmentations/registry.py`
- Add pipeline name to config (e.g., `augmentation_version: deeplive_v1`)

#### 4. Dataset Class
**Status:** 🔧 TO IMPLEMENT

**File to create:** `dataset/deeplive_dataset.py`

**Features needed:**
- GCS frame loading (similar to existing `FrameDataset`)
- Manifest parsing for sample metadata
- Landmark loading (optional)
- Strategy filtering
- Frame pair support (temporal analysis)

#### 5. Configuration Updates
**Status:** 🔧 TO IMPLEMENT

**Files to modify:**
- `config/defaults.yaml` - Add deeplive data source config
- `utils/config_helpers.py` - Add deeplive-specific overrides (if needed)

**New config sections:**
```yaml
# In experiment config
backbone:
  type: TBD
  variant: TBD
  source: TBD
  resolution: TBD

data_source:
  bucket_name: "live-deepfake-methods-real-and-fake-frames"
  type: "deeplive"
  strategies: ["edge_cases", "minimal_processing"]  # Filter by strategy
  use_landmarks: true
  frame_sampling: "sparse"  # or "full" or "pairs"

dataloader_strategy: "deeplive_paired"  # New strategy name

augmentation_version: "deeplive_v1"  # New pipeline name
```

---

## Implementation Order

### Phase 1: Validate Baseline (Experiment A)
**Priority:** 🔴 HIGH - Must work first

1. Test sanity_exp.yaml dry-run
2. Fix any issues from refactoring
3. Run short training to completion
4. Confirm all systems (checkpointing, metrics, lesson gate) work

### Phase 2: Gather Requirements (Experiment B)
**Priority:** 🟡 MEDIUM - Need user input

1. Get backbone specifications from user
2. Get dataloader requirements (sampling strategy, batch composition)
3. Get augmentation requirements (what transforms to apply)

### Phase 3: Implement DeepLive Support (Experiment B)
**Priority:** 🟢 LOW - After Phase 1 & 2 complete

1. Implement `DeepLiveDataset` class
2. Implement dataloader strategy
3. Implement augmentation pipeline
4. Register components
5. Test with dry-run
6. Launch training

---

## Questions for User (Experiment B)

### Backbone
1. Which backbone model do you want to use?
   - Options: CLIP variant, OpenCLIP, SigLIP, DINOv2, custom?
2. What resolution? (224, 336, 384?)
3. Where are the weights? (GCS path or local path?)

### Dataloader
1. **Frame sampling:** Use all 16 frames or just 8 anchors? Or temporal pairs?
2. **Batch composition:** How many samples per batch? Mix of strategies?
3. **Landmark usage:** Should we use landmarks for face cropping or feature extraction?
4. **Temporal logic:** Should we compare anchor vs consecutive frames in loss?

### Augmentation
1. **Augmentation intensity:** Light (preserve faces) or aggressive?
2. **Landmark-guided:** Should augmentations avoid face regions (using landmarks)?
3. **Real vs Fake:** Same augmentation for both or different?
4. **Temporal consistency:** Should consecutive frames get consistent augmentation?

### Data Filtering
1. **Strategies:** Use all strategies or filter? (edge_cases, minimal_processing, etc.)
2. **Frame quality:** Filter samples where face detection failed?

---

## Current Status

| Task | Status | Notes |
|------|--------|-------|
| **Experiment A** | | |
| Dry-run test | ⏳ Pending | Need to run `train_simple.py --dry-run` |
| Training test | 🔴 FAILED | Job crashed during GCS download |
| **Debug Progress** | | |
| Checkpoint exists | ✅ Verified | `gsutil ls` shows file exists |
| Permissions | ✅ Verified | Service account has Storage Admin |
| Error handling fix | ✅ Applied | Added try/except to `utils/gcs.py` |
| Return value check | ✅ Applied | Added check in `train_sweep.py` |
| **Experiment B** | | |
| Backbone specs | ❓ Need input | Awaiting user |
| Dataloader requirements | ❓ Need input | Awaiting user |
| Augmentation requirements | ❓ Need input | Awaiting user |
| Dataset implementation | ⏸️ Blocked | Waiting on requirements |
| Strategy implementation | ⏸️ Blocked | Waiting on requirements |
| Pipeline implementation | ⏸️ Blocked | Waiting on requirements |

---

## Debug Log (December 27, 2025)

### Issue: Experiment A crashed during GCS asset download

**Error observed:**
```
2025-12-27 18:57:23,703 - INFO - Source: gs://training-job-outputs/best_checkpoints/7a19nea9/top_n_effort_20251003_ep1_auc0.9460_eer0.1222.pth
2025-12-27 18:57:23,703 - INFO - Destination: ./weights/base.pth
ERROR ... The replica workerpool0-0 exited with a non-zero status of 1.
```

**Investigation:**
1. ✅ Checkpoint file exists (verified with `gsutil ls`)
2. ✅ Downloads locally (verified with `gsutil cp`)
3. ✅ IAM permissions look correct (Storage Admin)

**Root cause hypothesis:**
- The download function in `utils/gcs.py` lacked proper exception handling for single-file downloads
- If download failed, it would raise an unhandled exception
- Additionally, `train_sweep.py` didn't check the return value of `download_assets_from_gcs()`

**Fixes applied:**
1. `utils/gcs.py`: Added try/except around `blob.download_to_filename()` for single files
2. `train_sweep.py`: Added check for `None` return value from `download_assets_from_gcs()`
3. `requirements.txt`: Pinned `transformers==4.44.2` and `tokenizers==0.19.1` (4.54+ requires PyTorch 2.6+ due to CVE-2025-32434)

**Next step:** Rebuild image and retry

---

## Dev Efficiency Improvements (December 27, 2025)

### Problem
Each code or config change requires rebuilding the Docker image (~minutes).

### Solutions Implemented

#### 1. `.gcloudignore` - Faster uploads
Added `.gcloudignore` to exclude unnecessary files from Cloud Build uploads:
- Git files, Python cache, IDE files, docs, local data, logs
- **Impact:** Faster upload to Cloud Build

#### 2. GCS Config Injection - No rebuild for new experiments! ✅
New feature: Load experiment configs from GCS at runtime instead of baking them into the image.

**How to use:**
```bash
# 1. Upload your config to GCS (one-time per config)
gsutil cp orgenize_training/sanity_exp.yaml gs://experiment-configs/sanity_exp.yaml

# 2. Launch with the new script (no rebuild needed!)
./launch_experiment_gcs.sh \
    effort-dec2025 \
    asia-southeast1 \
    gs://experiment-configs/sanity_exp.yaml
```

**Files changed:**
- `entrypoint.sh` - Added `--gcs-config` flag to download config at runtime
- `launch_experiment_gcs.sh` - New launch script for GCS configs
- `Dockerfile` - Added gsutil for runtime downloads
- `.gcloudignore` - Exclude unnecessary files from builds

**Workflow comparison:**

| Old Workflow | New Workflow |
|--------------|--------------|
| 1. Edit config | 1. Edit config |
| 2. Build image | 2. Upload to GCS |
| 3. Wait ~5min | 3. Launch (instant) |
| 4. Launch | |

---

## Next Steps

1. **Immediate:** Run Experiment A dry-run to validate baseline
2. **User Input:** Get specifications for Experiment B (see questions above)
3. **Implementation:** Once requirements clear, implement DeepLive components
4. **Testing:** Test both experiments in parallel

---

## Files to Create/Modify

### New Files
- [ ] `dataset/deeplive_dataset.py` - Dataset class for new data
- [ ] `data/batching/strategies/deeplive_strategy.py` - Custom batching
- [ ] `data/augmentations/pipelines/deeplive_augmentations.py` - Custom augmentations
- [ ] `orgenize_training/deeplive_exp.yaml` - Config for Experiment B

### Modified Files
- [ ] `data/batching/registry.py` - Register new strategy
- [ ] `data/augmentations/registry.py` - Register new pipeline
- [ ] `config/defaults.yaml` - Add deeplive config section (if needed)
- [ ] `utils/config_helpers.py` - Add deeplive overrides (if needed)

---

**Last Updated:** December 27, 2025 - Initial planning
