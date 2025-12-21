# Refactoring Checklist & Progress Tracker

> **Purpose:** Track all refactoring tasks, their status, and testing requirements  
> **Last Updated:** December 2024

---

## Status Legend

- ⬜ **Not Started** - Task identified but work not begun
- 🟡 **In Progress** - Currently being worked on
- 🟢 **Complete** - Code changes done
- ✅ **Verified** - Tested and confirmed working
- ❌ **Blocked** - Cannot proceed due to dependency
- 🔄 **Needs Review** - Changes made, needs verification

---

## Phase 0: Planning & Documentation

| ID | Task | Status | Notes |
|----|------|--------|-------|
| P0.1 | Create main organization document | ✅ | `main.md` |
| P0.2 | Document batching strategies | ✅ | `batching_strategies.md` |
| P0.3 | Map configuration system | ✅ | `config_mapping.md` |
| P0.4 | Document augmentation pipelines | ✅ | `augmentations.md` |
| P0.5 | Create checklist system | ✅ | This file |
| P0.6 | Document logging conventions | ⬜ | `logging_conventions.md` |
| P0.7 | Document future research directions | ⬜ | `future_features.md` |
| P0.8 | Create testing plan | ⬜ | `testing_plan.md` |

---

## Phase 1: Low-Risk Extractions

### 1.1 Augmentation Pipeline Extraction

| ID | Task | Status | File(s) Affected | Test |
|----|------|--------|------------------|------|
| A1.1 | Create `data/augmentations/` directory | ⬜ | New | - |
| A1.2 | Extract `CustomUnsharpMask` transform | ⬜ | `dataloaders.py` → `augmentations/transforms/` | Unit test |
| A1.3 | Extract `NoOp` transform | ⬜ | `dataloaders.py` → `augmentations/transforms/` | Unit test |
| A1.4 | Create augmentation registry | ⬜ | New `augmentations/registry.py` | Unit test |
| A1.5 | Extract V3 pipeline | ⬜ | `dataloaders.py` → `augmentations/pipelines/` | Unit test |
| A1.6 | Extract V4 pipeline | ⬜ | `dataloaders.py` → `augmentations/pipelines/` | Unit test |
| A1.7 | Extract V5 pipeline | ⬜ | `dataloaders.py` → `augmentations/pipelines/` | Unit test |
| A1.8 | Extract V6 pipeline | ⬜ | `dataloaders.py` → `augmentations/pipelines/` | Unit test |
| A1.9 | Extract V7 pipeline | ⬜ | `dataloaders.py` → `augmentations/pipelines/` | Unit test |
| A1.10 | Extract surgical pipeline | ⬜ | `dataloaders.py` → `augmentations/pipelines/` | Unit test |
| A1.11 | Extract helper pipelines (degrade, enhance, social_media) | ⬜ | `dataloaders.py` → `augmentations/helpers/` | Unit test |
| A1.12 | Update `dataloaders.py` to use registry | ⬜ | `dataloaders.py` | Integration test |
| A1.13 | **CHECKPOINT: Run training with aug changes** | ⬜ | - | Full training run |

### 1.2 Utility Extraction

| ID | Task | Status | File(s) Affected | Test |
|----|------|--------|------------------|------|
| U1.1 | Create `utils/` directory | ⬜ | New | - |
| U1.2 | Extract `download_gcs_asset` | ⬜ | `train_sweep.py` → `utils/gcs.py` | Unit test |
| U1.3 | Extract `download_assets_from_gcs` | ⬜ | `train_sweep.py` → `utils/gcs.py` | Unit test |
| U1.4 | Extract `init_seed` | ⬜ | `train_sweep.py` → `utils/setup.py` | Unit test |
| U1.5 | Extract `choose_optimizer` | ⬜ | `train_sweep.py` → `utils/setup.py` | Unit test |
| U1.6 | Extract `choose_scheduler` | ⬜ | `train_sweep.py` → `utils/setup.py` | Unit test |
| U1.7 | Extract `choose_metric` | ⬜ | `train_sweep.py` → `utils/setup.py` | Unit test |
| U1.8 | Update `train_sweep.py` imports | ⬜ | `train_sweep.py` | Integration test |
| U1.9 | **CHECKPOINT: Run training with utility changes** | ⬜ | - | Full training run |

---

## Phase 2: Configuration System

| ID | Task | Status | File(s) Affected | Test |
|----|------|--------|------------------|------|
| C2.1 | Create `config/` directory structure | ⬜ | New | - |
| C2.2 | Define config dataclasses/schema | ⬜ | New `config/schema.py` | Unit test |
| C2.3 | Create config loader | ⬜ | New `config/loader.py` | Unit test |
| C2.4 | Create default config template | ⬜ | New `config/defaults.yaml` | - |
| C2.5 | Add W&B override mapping | ⬜ | `config/loader.py` | Unit test |
| C2.6 | Add config validation | ⬜ | `config/loader.py` | Unit test |
| C2.7 | Update `train_sweep.py` to use new loader | ⬜ | `train_sweep.py` | Integration test |
| C2.8 | Deprecate old config files (keep for reference) | ⬜ | Move to `config/legacy/` | - |
| C2.9 | **CHECKPOINT: Run training with new config system** | ⬜ | - | Full training run |

---

## Phase 3: Data Pipeline Refactoring

### 3.1 Splitting Logic

| ID | Task | Status | File(s) Affected | Test |
|----|------|--------|------------------|------|
| S3.1 | Create `data/splitting/` directory | ⬜ | New | - |
| S3.2 | Define `Splitter` interface | ⬜ | New `data/splitting/base.py` | - |
| S3.3 | Extract legacy splitting | ⬜ | `prepare_splits.py` → `data/splitting/legacy.py` | Unit test |
| S3.4 | Extract property-based splitting | ⬜ | `prepare_splits.py` → `data/splitting/property_based.py` | Unit test |
| S3.5 | Create splitter factory | ⬜ | New `data/splitting/__init__.py` | Unit test |
| S3.6 | Update `train_sweep.py` to use factory | ⬜ | `train_sweep.py` | Integration test |
| S3.7 | **CHECKPOINT: Run training with splitting changes** | ⬜ | - | Full training run |

### 3.2 Batching Strategies

| ID | Task | Status | File(s) Affected | Test |
|----|------|--------|------------------|------|
| B3.1 | Create `data/batching/` directory | ⬜ | New | - |
| B3.2 | Define `BatchingStrategy` interface | ⬜ | New `data/batching/base.py` | - |
| B3.3 | Extract custom DataPipes | ⬜ | `dataloaders.py` → `data/datapipes.py` | Unit test |
| B3.4 | Extract frame-level strategy | ⬜ | `dataloaders.py` → `data/batching/frame_level.py` | Unit test |
| B3.5 | Extract video-level strategy | ⬜ | `dataloaders.py` → `data/batching/video_level.py` | Unit test |
| B3.6 | Extract per-method strategy | ⬜ | `dataloaders.py` → `data/batching/per_method.py` | Unit test |
| B3.7 | Extract property-balanced strategy | ⬜ | `dataloaders.py` → `data/batching/property_balanced.py` | Unit test |
| B3.8 | Create batching factory | ⬜ | New `data/batching/__init__.py` | Unit test |
| B3.9 | Update `create_dataloaders` to use factory | ⬜ | `dataloaders.py` | Integration test |
| B3.10 | **CHECKPOINT: Run training with batching changes** | ⬜ | - | Full training run |

### 3.3 Frame Loading

| ID | Task | Status | File(s) Affected | Test |
|----|------|--------|------------------|------|
| L3.1 | Create `data/loaders/` directory | ⬜ | New | - |
| L3.2 | Create unified frame loader | ⬜ | New `data/loaders/frame_loader.py` | Unit test |
| L3.3 | Extract video loader | ⬜ | `dataloaders.py` → `data/loaders/video_loader.py` | Unit test |
| L3.4 | Create GCS/fsspec helpers | ⬜ | New `data/loaders/gcs_utils.py` | Unit test |
| L3.5 | Update batching strategies to use loaders | ⬜ | `data/batching/*.py` | Integration test |
| L3.6 | **CHECKPOINT: Run training with loader changes** | ⬜ | - | Full training run |

---

## Phase 4: Trainer Refactoring

| ID | Task | Status | File(s) Affected | Test |
|----|------|--------|------------------|------|
| T4.1 | Create `trainer/mixins/` directory | ⬜ | New | - |
| T4.2 | Extract checkpointing mixin | ⬜ | `trainer.py` → `trainer/mixins/checkpointing.py` | Unit test |
| T4.3 | Extract early stopping mixin | ⬜ | `trainer.py` → `trainer/mixins/early_stopping.py` | Unit test |
| T4.4 | Extract Group-DRO mixin | ⬜ | `trainer.py` → `trainer/mixins/group_dro.py` | Unit test |
| T4.5 | Extract curriculum/lesson gate mixin | ⬜ | `trainer.py` → `trainer/mixins/curriculum.py` | Unit test |
| T4.6 | Extract validation logic | ⬜ | `trainer.py` → `trainer/validation.py` | Unit test |
| T4.7 | Extract reporting logic | ⬜ | `trainer.py` → `trainer/reporting.py` | Unit test |
| T4.8 | Simplify core `Trainer` class | ⬜ | `trainer.py` | Integration test |
| T4.9 | **CHECKPOINT: Run training with trainer changes** | ⬜ | - | Full training run |

---

## Phase 5: Entry Point Simplification

| ID | Task | Status | File(s) Affected | Test |
|----|------|--------|------------------|------|
| E5.1 | Create simplified `train.py` | ⬜ | New (or rewrite existing) | - |
| E5.2 | Create high-level pipeline interface | ⬜ | New `data/pipeline.py` | Unit test |
| E5.3 | Deprecate/rename `train_sweep.py` | ⬜ | Rename to `train_sweep_legacy.py` | - |
| E5.4 | Update documentation | ⬜ | README, docstrings | - |
| E5.5 | **FINAL CHECKPOINT: Full end-to-end test** | ⬜ | - | Full training run + validation |

---

## Future Features (Phase 6+)

| ID | Feature | Status | Priority | Notes |
|----|---------|--------|----------|-------|
| F6.1 | Configurable CLIP backbone selection | ⬜ | High | Currently hardcoded |
| F6.2 | Paired real/fake video batching | ⬜ | High | For new dataset structure |
| F6.3 | AugMix augmentation | ⬜ | Medium | New augmentation type |
| F6.4 | Facial landmark-based augmentation | ⬜ | Medium | Occlude mouth/eyes/nose |
| F6.5 | Temporal frame sequences | ⬜ | Medium | For future architecture |
| F6.6 | Quality matching (fake→real) preprocessing | ⬜ | High | Remove obvious fake artifacts |
| F6.7 | Pipeline visualization/debugging tools | ⬜ | High | See what's happening |
| F6.8 | Metadata-aware data loading | ⬜ | High | Real source tracking |

---

## Testing Checkpoints

Each checkpoint should verify:

1. **Training starts** - No import errors, config loads
2. **Data loads** - Batches are created correctly
3. **Forward pass** - Model processes batches
4. **Backward pass** - Gradients flow correctly
5. **Checkpointing** - Model saves/loads correctly
6. **Metrics** - Validation metrics computed correctly
7. **W&B logging** - Metrics logged to W&B

### Checkpoint Test Script Template
```bash
# Quick sanity check (5 epochs, small subset)
python train.py \
  --config config/test_config.yaml \
  --data.splitting.data_subset_percentage 0.01 \
  --training.epochs 5 \
  --dry_run
```

---

## Rollback Plan

If any phase introduces bugs:

1. **Git tags** - Tag before each phase: `git tag pre-phase-X`
2. **Legacy imports** - Keep old code importable during transition
3. **Feature flags** - Use config flags to toggle old/new code paths
4. **Checkpoint tests** - Run after each subtask, not just phases

---

## Notes & Issues Log

| Date | Issue | Resolution |
|------|-------|------------|
| 2024-12-21 | Initial planning complete | N/A |
| | | |
| | | |

---

## Sign-off

- [ ] All Phase 1 tasks complete and verified
- [ ] All Phase 2 tasks complete and verified
- [ ] All Phase 3 tasks complete and verified
- [ ] All Phase 4 tasks complete and verified
- [ ] All Phase 5 tasks complete and verified
- [ ] Full end-to-end training run successful
- [ ] Documentation updated
