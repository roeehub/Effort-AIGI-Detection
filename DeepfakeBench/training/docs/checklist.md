# Refactoring Checklist & Progress Tracker

> **Purpose:** Track all refactoring tasks, their status, and testing requirements  
> **Last Updated:** January 2026  
> **Status:** � **FIXES APPLIED - Needs Testing**

---

## ⚠️ January 2026 Audit Results & Fixes

**Original Issue:** `train_deeplive.py` bypassed the entire Trainer class and implemented its own training loop.

**Fixes Applied:**
- ✅ Created `train_deeplive_v2.py` that properly uses the Trainer class
- ✅ Added 'deeplive' strategy handling to `trainer/trainer.py`
- ✅ Added CLIP normalization to `data/batching/deeplive.py`
- ✅ Added W&B config override support (`apply_wandb_overrides_deeplive()`)
- ✅ Added LR scheduler support (uses `choose_scheduler()`)

**Next Step:** Run actual training to verify everything works end-to-end.

See `refactoring_audit_jan2026.md` for full details.

### Core Principle (Restated)
> The refactoring should **preserve identical functionality** while organizing code into modules.
> New features (backbones, datasets, augmentations) should plug into the **SAME** pipeline.
> We should NOT create separate training scripts with different behavior.

---

## Status Legend

- ⬜ **Not Started** - Task identified but work not begun
- 🟡 **In Progress** - Currently being worked on
- 🟢 **Complete** - Code changes done (but NOT verified end-to-end)
- ✅ **Verified** - Tested and confirmed working with actual training run
- ❌ **Blocked** - Cannot proceed due to dependency
- 🔄 **Needs Review** - Changes made, needs verification
- 🔴 **Broken** - Previously marked complete but found to be broken

---

## Phase 0: Planning & Documentation

| ID | Task | Status | Notes |
|----|------|--------|-------|
| P0.1 | Create main organization document | ✅ | `main.md` |
| P0.2 | Document batching strategies | ✅ | `batching_strategies.md` |
| P0.3 | Map configuration system | ✅ | `config_mapping.md` |
| P0.4 | Document augmentation pipelines | ✅ | `augmentations.md` |
| P0.5 | Create checklist system | ✅ | This file |
| P0.6 | Document logging conventions | ✅ | `logging_conventions.md` |
| P0.7 | Document future research directions | ✅ | `future_features.md` |
| P0.8 | Create testing plan | ✅ | `testing_plan.md` |

---

## Phase 1: Low-Risk Extractions

### 1.1 Augmentation Pipeline Extraction

| ID | Task | Status | File(s) Affected | Test |
|----|------|--------|------------------|------|
| A1.1 | Create `data/augmentations/` directory | 🟢 | New | - |
| A1.2 | Extract `CustomUnsharpMask` transform | 🟢 | `dataloaders.py` → `augmentations/transforms.py` | Unit test |
| A1.3 | Extract `NoOp` transform | 🟢 | `dataloaders.py` → `augmentations/transforms.py` | Unit test |
| A1.4 | Create augmentation registry | 🟢 | New `augmentations/registry.py` | Unit test |
| A1.5 | Extract V3 pipeline | 🟢 | `dataloaders.py` → `augmentations/pipelines.py` | Unit test |
| A1.6 | Extract V4 pipeline | 🟢 | `dataloaders.py` → `augmentations/pipelines.py` | Unit test |
| A1.7 | Extract V5 pipeline | 🟢 | `dataloaders.py` → `augmentations/pipelines.py` | Unit test |
| A1.8 | Extract V6 pipeline | 🟢 | `dataloaders.py` → `augmentations/pipelines.py` | Unit test |
| A1.9 | Extract V7 pipeline | 🟢 | `dataloaders.py` → `augmentations/pipelines.py` | Unit test |
| A1.10 | Extract surgical pipeline | 🟢 | `dataloaders.py` → `augmentations/pipelines.py` | Unit test |
| A1.11 | Extract helper pipelines (degrade, enhance, social_media) | 🟢 | `dataloaders.py` → `augmentations/pipelines.py` | Unit test |
| A1.12 | Update `dataloaders.py` to use imports | 🟢 | `dataloaders.py` | Integration test |
| A1.13 | **CHECKPOINT: Run training with aug changes** | ⬜ | - | Full training run |

### 1.2 Utility Extraction

| ID | Task | Status | File(s) Affected | Test |
|----|------|--------|------------------|------|
| U1.1 | Create `utils/` directory | 🟢 | New | - |
| U1.2 | Extract `download_gcs_asset` | 🟢 | `train_sweep.py` → `utils/gcs.py` | Unit test |
| U1.3 | Extract `download_assets_from_gcs` | 🟢 | `train_sweep.py` → `utils/gcs.py` | Unit test |
| U1.4 | Extract `init_seed` | 🟢 | `train_sweep.py` → `utils/setup.py` | Unit test |
| U1.5 | Extract `choose_optimizer` | 🟢 | `train_sweep.py` → `utils/setup.py` | Unit test |
| U1.6 | Extract `choose_scheduler` | 🟢 | `train_sweep.py` → `utils/setup.py` | Unit test |
| U1.7 | Extract `choose_metric` | 🟢 | `train_sweep.py` → `utils/setup.py` | Unit test |
| U1.8 | Update `train_sweep.py` imports | 🟢 | `train_sweep.py` | Integration test |
| U1.9 | **CHECKPOINT: Run training with utility changes** | ⬜ | - | Full training run |

---

## Phase 2: Configuration System

| ID | Task | Status | File(s) Affected | Test |
|----|------|--------|------------------|------|
| C2.1 | Create `config_system/` directory structure | 🟢 | New `config_system/` | - |
| C2.2 | Define config dataclasses/schema | 🟢 | New `config_system/schema.py` | Unit test |
| C2.3 | Create config loader | 🟢 | New `config_system/loader.py` | Unit test |
| C2.4 | Create default config template | 🟢 | New `config/defaults.yaml` | - |
| C2.5 | Add W&B override mapping | 🟢 | `config_system/loader.py` | Unit test |
| C2.6 | Add config validation | 🟢 | `config_system/validation.py` | Unit test |
| C2.7 | Update `train_sweep.py` to use new loader | ⬜ | `train_sweep.py` | Integration test |
| C2.8 | Deprecate old config files (keep for reference) | ⬜ | Move to `config/legacy/` | - |
| C2.9 | **CHECKPOINT: Run training with new config system** | ⬜ | - | Full training run |

---

## Phase 3: Data Pipeline Refactoring

### 3.1 Splitting Logic

| ID | Task | Status | File(s) Affected | Test |
|----|------|--------|------------------|------|
| S3.1 | Create `data/splitting/` directory | 🟢 | New | - |
| S3.2 | Define `Splitter` interface | 🟢 | New `data/splitting/base.py` | - |
| S3.3 | Extract legacy splitting | 🟢 | `prepare_splits.py` → `data/splitting/splitters.py` | Unit test |
| S3.4 | Extract property-based splitting | 🟢 | `prepare_splits.py` → `data/splitting/splitters.py` | Unit test |
| S3.5 | Create splitter factory | 🟢 | `data/splitting/__init__.py` + `get_splitter()` | Unit test |
| S3.6 | Add bridge function in `prepare_splits.py` | 🟢 | `prepare_splits.py` (use_refactored flag) | Integration test |
| S3.7 | **CHECKPOINT: Run training with splitting changes** | ⬜ | - | Full training run |

### 3.2 Batching Strategies

| ID | Task | Status | File(s) Affected | Test |
|----|------|--------|------------------|------|
| B3.1 | Create `data/batching/` directory | 🟢 | New | - |
| B3.2 | Define `BatchingStrategy` interface | 🟢 | New `data/batching/base.py` | - |
| B3.3 | Extract custom DataPipes | 🟢 | `dataloaders.py` → `data/batching/datapipes.py` | Unit test |
| B3.4 | Extract frame-level strategy | 🟢 | `dataloaders.py` → `data/batching/frame_level.py` | Unit test |
| B3.5 | Extract video-level strategy | 🟢 | `dataloaders.py` → `data/batching/video_level.py` | Unit test |
| B3.6 | Extract per-method strategy | 🟢 | `dataloaders.py` → `data/batching/per_method.py` | Unit test |
| B3.7 | Extract property-balanced strategy | 🟢 | `dataloaders.py` → `data/batching/property_balanced.py` | Unit test |
| B3.8 | Create batching factory | 🟢 | New `data/batching/factory.py` | Unit test |
| B3.9 | Update `create_dataloaders` to use factory | 🟢 | `dataloaders.py` (backward compat imports) | Integration test |
| B3.10 | **CHECKPOINT: Run training with batching changes** | ⬜ | - | Full training run |

### 3.3 Frame Loading

| ID | Task | Status | File(s) Affected | Test |
|----|------|--------|------------------|------|
| L3.1 | ~~Create `data/loaders/` directory~~ | 🟢 | Moved to `data/batching/loaders.py` | - |
| L3.2 | Create unified frame loader | 🟢 | `data/batching/loaders.py` | Unit test |
| L3.3 | Extract video loader | 🟢 | `data/batching/loaders.py` | Unit test |
| L3.4 | Create GCS/fsspec helpers | 🟢 | Inline in loaders (fsspec handles GCS) | Unit test |
| L3.5 | Update batching strategies to use loaders | 🟢 | `data/batching/*.py` imports from loaders | Integration test |
| L3.6 | **CHECKPOINT: Run training with loader changes** | ⬜ | - | Full training run |

> **Note:** Frame loading was consolidated into `data/batching/loaders.py` rather than a separate directory, as the loaders are tightly coupled with batching strategies.

---

## Phase 4: Trainer Refactoring

| ID | Task | Status | File(s) Affected | Test |
|----|------|--------|------------------|------|
| T4.1 | Create `trainer/mixins/` directory | 🟢 | New | - |
| T4.2 | Extract checkpointing mixin | 🟢 | `trainer.py` → `trainer/mixins/checkpointing.py` | Unit test |
| T4.3 | Extract early stopping mixin | 🟢 | `trainer.py` → `trainer/mixins/early_stopping.py` | Unit test |
| T4.4 | Extract Group-DRO mixin | 🟢 | `trainer.py` → `trainer/mixins/group_dro.py` | Unit test |
| T4.5 | Extract curriculum/lesson gate mixin | 🟢 | `trainer.py` → `trainer/mixins/curriculum.py` | Unit test |
| T4.6 | Extract validation logic | 🟢 | `trainer.py` → `trainer/mixins/validation.py` | Unit test |
| T4.7 | Extract reporting logic | 🟢 | `trainer.py` → `trainer/mixins/reporting.py` | Unit test |
| T4.8 | Simplify core `Trainer` class | 🟢 | `trainer.py` | Integration test |
| T4.9 | **CHECKPOINT: Run training with trainer changes** | ⬜ | - | Full training run |

---

## Phase 5: Entry Point Simplification

> **🔴 CRITICAL: This phase was attempted but FAILED.**  
> `train_deeplive.py` was created but it **bypasses the Trainer class entirely**.  
> See `refactoring_audit_jan2026.md` for details.

| ID | Task | Status | File(s) Affected | Test |
|----|------|--------|------------------|------|
| E5.1 | Create simplified `train.py` | 🔴 BROKEN | `train_deeplive.py` exists but wrong | - |
| E5.2 | Create high-level pipeline interface | ⬜ | New `data/pipeline.py` | Unit test |
| E5.3 | ~~Deprecate/rename `train_sweep.py`~~ | ❌ BLOCKED | DO NOT deprecate - it's the working version! | - |
| E5.4 | Update documentation | ⬜ | README, docstrings | - |
| E5.5 | **FINAL CHECKPOINT: Full end-to-end test** | ⬜ | - | Full training run + validation |

### Phase 5 Fix Tasks (Updated - January 2026)

| ID | Task | Status | File(s) Affected | Test |
|----|------|--------|------------------|------|
| E5.F1 | Fix CLIP normalization in deeplive collate_fn | ✅ | `data/batching/deeplive.py` | Unit test |
| E5.F2 | Refactor to use Trainer class | ✅ | `train_deeplive_v2.py` (NEW), `trainer/trainer.py` | Integration test |
| E5.F3 | Add W&B config override support | ✅ | `train_deeplive_v2.py` | Integration test |
| E5.F4 | Add LR scheduler | ✅ | `train_deeplive_v2.py` (uses choose_scheduler) | Integration test |
| E5.F5 | Verify all Trainer features work with DeepLive | ⬜ | - | Full training run |
| E5.F6 | **CHECKPOINT: Run full DeepLive training with Trainer** | ⬜ | - | W&B logs match old format |

**Files Created/Modified:**
- `train_deeplive_v2.py` (NEW) - Properly uses Trainer class
- `train_deeplive.py` (DEPRECATED) - Keep as backup, do not use
- `data/batching/deeplive.py` - Added CLIP normalization
- `trainer/trainer.py` - Added 'deeplive' strategy handling

---

## Local Development Infrastructure

| ID | Task | Status | File(s) Affected | Notes |
|----|------|--------|------------------|-------|
| D1 | Create `Dockerfile.dev` (CPU-only, fast build) | ✅ | `Dockerfile.dev` | Uses python:3.10-slim, ~5min build |
| D2 | Create `docker-compose.yml` | ✅ | `docker-compose.yml` | Volume mounts, dev services |
| D3 | Create `dev.sh` helper script | ✅ | `dev.sh` | build/verify/shell/test/debug commands |
| D4 | Create verification script | ✅ | `scripts/verify_refactoring.py` | 17 tests for refactored modules |
| D5 | VS Code debug configurations | ✅ | `.vscode/launch.json` | Attach to Docker debugpy |
| D6 | Create test debug config | ✅ | `config/test_debug.yaml` | Minimal data, no GPU, no W&B |
| D7 | Document local dev workflow | ✅ | `docs/LOCAL_DEVELOPMENT.md` | Setup, debugging, testing guide |
| D8 | Create `.dockerignore` | ✅ | `.dockerignore` | Excludes weights, logs, etc. |

### Quick Reference
```bash
# Build dev container (first time or after requirements change)
./dev.sh build

# Verify all refactored modules work
./dev.sh verify

# Open interactive shell
./dev.sh shell

# Run tests
./dev.sh test

# Debug with VS Code (attach to localhost:5678)
./dev.sh debug
```

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
| 2024-12-21 | `libgl1-mesa-glx` package renamed in Debian trixie | Changed to `libgl1` in Dockerfile.dev |
| 2024-12-21 | Module imports failing in Docker | Added PYTHONPATH=/workspace to docker-compose |
| 2024-12-21 | All 17 verification tests passing ✅ | Phases 1-3 refactoring verified working |
| **2026-01-01** | **AUDIT: train_deeplive.py bypasses Trainer** | **See refactoring_audit_jan2026.md** |
| **2026-01-01** | **AUDIT: Missing CLIP normalization** | **Pending fix** |
| **2026-01-01** | **AUDIT: Missing LR scheduler** | **Pending fix** |
| **2026-01-01** | **AUDIT: Missing W&B config overrides** | **Pending fix** |

---

## Sign-off

> **⚠️ DO NOT sign off until ALL checkpoints have actual training runs**

- [x] All Phase 1 tasks complete (code changes only - NOT verified with training)
- [x] All Phase 2 tasks complete (code changes only - NOT verified with training)
- [x] All Phase 3 tasks complete (code changes only - NOT verified with training)
- [x] All Phase 4 tasks complete (code changes only - NOT verified with training)
- [ ] All Phase 5 tasks complete - **🔴 FAILED - train_deeplive.py is broken**
- [ ] Full end-to-end training run successful with OLD flow (train_sweep.py)
- [ ] Full end-to-end training run successful with NEW flow (train_deeplive.py)
- [ ] W&B logs look identical between old and new flows
- [ ] Documentation updated

### Final Verification Criteria

Before marking the refactoring as complete, the following must ALL be true:

1. `train_deeplive.py` uses the `Trainer` class (not custom loop)
2. All Trainer features work: ArcFace annealing, curriculum, checkpointing, etc.
3. W&B logging produces the same metrics structure as `train_sweep.py`
4. GCS checkpoint upload works
5. Can run W&B sweeps with the new flow
6. Adding a new dataset requires ONLY:
   - A new experiment YAML
   - A new batching strategy (if data format differs)
   - **NOT** a new training script
