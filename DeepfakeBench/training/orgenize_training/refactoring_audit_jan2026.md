# Refactoring Audit - January 2026

> **Status:** 🔴 **INCOMPLETE - Critical Issues Found**  
> **Audited By:** Code Review  
> **Date:** January 1, 2026

---

## Executive Summary

**The refactoring is NOT complete.** While the modular structure was created (`data/batching/`, `data/augmentations/`, `trainer/mixins/`, `utils/`), the new entry point `train_deeplive.py` **bypasses the entire production pipeline** and implements its own simplified training loop from scratch.

This defeats the purpose of the refactoring.

---

## 🎯 The Actual Goal (Restated for Clarity)

### What the Refactoring Was Supposed to Do

1. **Break up monolithic files** into smaller, organized modules
2. **Preserve identical functionality** - the training should behave exactly the same
3. **Make future additions easier** - new backbones, datasets, augmentations should plug into the SAME pipeline
4. **Keep all parameters, logging, and behavior** feeling the same

### What the Refactoring Was NOT Supposed to Do

- ❌ Create new training scripts with different behavior
- ❌ Skip features that were working (ArcFace annealing, curriculum learning, etc.)
- ❌ Change how W&B logging works
- ❌ Create a "simpler" alternative pipeline
- ❌ Remove functionality to make code "cleaner"

### The Correct Mental Model

```
BEFORE REFACTORING:
┌────────────────────────────────────────────────────────────────┐
│                    train_sweep.py (900 lines)                  │
│  - Config loading, data prep, model creation, training loop    │
│  - Everything in one file, messy but WORKING                   │
└────────────────────────────────────────────────────────────────┘

AFTER REFACTORING (Intended):
┌────────────────────────────────────────────────────────────────┐
│                    train_sweep.py (simplified)                 │
│  - Orchestration only, delegates to modules                    │
├────────────────────────────────────────────────────────────────┤
│  utils/           │ data/batching/    │ trainer/mixins/        │
│  - gcs.py         │ - base.py         │ - checkpointing.py     │
│  - setup.py       │ - frame_level.py  │ - early_stopping.py    │
│  - config.py      │ - video_level.py  │ - curriculum.py        │
│                   │ - property_bal.py │ - arcface.py           │
│                   │ - deeplive.py     │ - validation.py        │
└────────────────────────────────────────────────────────────────┘
                    ↑
         ALL USING THE SAME TRAINER CLASS
         ALL SAME BEHAVIOR AS BEFORE
```

### What Actually Happened (Wrong)

```
AFTER REFACTORING (Actual):
┌─────────────────────────┐    ┌─────────────────────────────────┐
│   train_sweep.py        │    │   train_deeplive.py             │
│   (old pipeline)        │    │   (NEW SEPARATE PIPELINE!)      │
│   - Uses Trainer        │    │   - Own training loop           │
│   - All features work   │    │   - Missing most features       │
│   - Battle-tested       │    │   - Different behavior          │
└─────────────────────────┘    └─────────────────────────────────┘
        ↓                              ↓
   PRODUCTION READY              INCOMPLETE PROTOTYPE
```

---

## 🚨 Critical Issues Found

### Issue #1: `train_deeplive.py` Bypasses the Trainer Class

**Severity:** 🔴 Critical

**Location:** `train_deeplive.py` lines 244-536

**Problem:** Instead of using the existing `Trainer` class (which has all the battle-tested functionality), `train_deeplive.py` implements its own training loop from scratch:

```python
# train_deeplive.py - WRONG APPROACH
def train_epoch(model, train_loader, optimizer, criterion, device, epoch, max_steps=None):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model({'image': images, 'label': labels})
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()  # That's it - no bells and whistles
```

**What's Missing:**
| Feature | In Trainer | In train_deeplive.py |
|---------|-----------|---------------------|
| ArcFace parameter annealing | ✅ `_update_arcface_s()` | ❌ Not implemented |
| Curriculum Learning / Lesson Gates | ✅ `_check_lesson_gate()` | ❌ Not implemented |
| Group-DRO loss calculation | ✅ `calculate_group_dro_loss()` | ❌ Not implemented |
| Mixed precision (AMP/GradScaler) | ✅ `self.scaler` | ❌ Not implemented |
| Gradient clipping | ✅ `gradient_clip_val` | ❌ Not implemented |
| Step-based training control | ✅ `max_train_steps`, `evaluate_every_steps` | ❌ Not implemented |
| Per-method validation metrics | ✅ Detailed breakdown | ❌ Just overall accuracy |
| Confusion matrix logging | ✅ | ❌ |
| Video-level aggregation | ✅ | ❌ |
| OOD monitoring | ✅ `_run_ood_monitoring()` | ❌ Not implemented |
| GCS checkpoint upload | ✅ `_upload_to_gcs()` | ❌ Not implemented |
| Top-N checkpoint management | ✅ | ❌ |
| Rich W&B logging | ✅ Tables, histograms, etc. | ⚠️ Basic scalars only |

---

### Issue #2: Missing CLIP Normalization

**Severity:** 🔴 Critical (will cause bad model performance)

**Location:** `data/batching/deeplive.py` lines 213-244

**Problem:** The DeepLive collate function does basic `/255` normalization but doesn't apply CLIP's expected normalization:

```python
# Current code (WRONG)
img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

# Should be (CORRECT)
img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
img_tensor = (img_tensor - mean) / std
```

**Impact:** The model receives incorrectly normalized inputs, leading to degraded performance.

---

### Issue #3: No Learning Rate Scheduler

**Severity:** 🟡 High

**Location:** `train_deeplive.py` lines 446-450

**Problem:** The experiment config specifies a cosine scheduler with warmup, but it's never created:

```yaml
# In deeplive_vit_B16.yaml
lr_scheduler: "cosine_with_warmup"
total_training_steps: 5000
lr_scheduler_warmup_steps: 500
```

```python
# In train_deeplive.py - scheduler is NEVER CREATED
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# No scheduler = flat learning rate for entire run
```

---

### Issue #4: No W&B Config Override System

**Severity:** 🟡 High

**Location:** `train_deeplive.py` lines 348-367

**Problem:** The old flow has sophisticated config merging that allows W&B sweeps to override any parameter. The new flow just loads the YAML directly:

```python
# Old flow (train_sweep.py) - CORRECT
apply_all_wandb_overrides(config, data_config, wandb.config, logger)

# New flow (train_deeplive.py) - NO OVERRIDE SYSTEM
config = load_experiment_config(args.config)  # That's it
```

**Impact:** Cannot use W&B sweeps with the new flow.

---

### Issue #5: No Base Checkpoint Loading

**Severity:** 🟡 High

**Location:** `train_deeplive.py`

**Problem:** The old flow supports loading a pre-trained checkpoint and continuing training (crucial for curriculum learning). The new flow starts from scratch every time.

---

### Issue #6: Different Data Pipeline (Intentional but Incomplete)

**Severity:** 🟡 Medium

**Details:** The DeepLive dataset (`live-deepfake-methods-real-and-fake-frames` bucket) has a different structure than the old dataset (`df40-frames`). This is intentional - new data format. However, the pipeline should still flow through the same Trainer, use the same logging, same checkpointing, etc.

---

## 📋 Checklist Updates

The original `checklist.md` marked many items as 🟢 Complete. This was **premature**. Here's the corrected status:

### Phase 1-3: Module Extraction - ✅ Actually Complete
The modular structure IS properly extracted:
- `data/augmentations/` - ✅ Working
- `data/batching/` - ✅ Working
- `trainer/mixins/` - ✅ Working
- `utils/` - ✅ Working

### Phase 4: Trainer Refactoring - ⚠️ Partially Complete
- Mixins extracted: ✅
- **But train_deeplive.py doesn't use them!** ❌

### Phase 5: Entry Point Simplification - ❌ NOT Complete
- `train_deeplive.py` was supposed to use the refactored modules
- Instead it implements everything from scratch
- This is the core failure

### Missing Checkpoints - Never Run
- ⬜ "CHECKPOINT: Run training with aug changes" - Never verified
- ⬜ "CHECKPOINT: Run training with batching changes" - Never verified
- ⬜ "FINAL CHECKPOINT: Full end-to-end test" - Never verified

---

## 🔧 Fix Plan

### Goal
Make `train_deeplive.py` use the existing `Trainer` class and all its functionality, just with the DeepLive dataset and batching strategy.

### Step 1: Fix CLIP Normalization (Quick Win)
Add proper normalization to `deeplive_collate_fn` in `data/batching/deeplive.py`.

### Step 2: Refactor train_deeplive.py to Use Trainer
Replace the custom training loop with Trainer instantiation:

```python
# CURRENT (wrong)
for epoch in range(1, n_epochs + 1):
    train_metrics = train_epoch(model, train_loader, ...)
    val_metrics = evaluate(model, val_loader, ...)

# TARGET (correct)
trainer = Trainer(
    config=config,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    logger=logger,
    val_in_dist_loader=val_loader,
    val_holdout_loader=test_loader,
    metric_scoring='auc',
    wandb_run=wandb_run,
)

for epoch in range(config['nEpochs']):
    trainer.train_epoch(train_loader, epoch, train_videos=train_samples)
```

### Step 3: Add W&B Config Override Support
Use the same `apply_all_wandb_overrides()` function for consistency.

### Step 4: Add Scheduler Creation
Use `choose_scheduler()` from `utils/setup.py`.

### Step 5: Verify End-to-End
Run actual training and verify:
- All metrics log correctly to W&B
- Checkpoints save to GCS
- ArcFace annealing works (if enabled)
- Early stopping works
- Validation metrics match expected format

---

## ✅ FIXES APPLIED (January 2026)

### Fix 1: CLIP Normalization ✅
**File:** `data/batching/deeplive.py`

Added CLIP normalization to `deeplive_collate_fn`:
```python
# Apply CLIP normalization (same as old pipeline)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
img_tensor = (img_tensor - CLIP_MEAN) / CLIP_STD
```

### Fix 2: Trainer Integration ✅
**Files:** 
- Created `train_deeplive_v2.py` (uses Trainer class properly)
- Modified `trainer/trainer.py` (added 'deeplive' strategy handling)

Key changes:
1. `train_deeplive_v2.py` creates Trainer instance and delegates to it
2. Added `strategy == 'deeplive'` branch in `Trainer.train_epoch()` for epoch length calculation
3. Added 'deeplive' to the list of strategies in training loop branch

### Fix 3: W&B Config Override Support ✅
**File:** `train_deeplive_v2.py`

Added `apply_wandb_overrides_deeplive()` function that supports:
- Learning rate and optimizer params
- Batch size params
- ArcFace params (s, m, annealing)
- Loss params (focal loss)
- Training params (epochs, steps)
- Backbone params (rank, resolution)
- DeepLive-specific params (sampling_mode, train_split)
- Augmentation params
- Scheduler params

### Fix 4: LR Scheduler ✅
**File:** `train_deeplive_v2.py`

Uses `choose_scheduler()` from utils, same as train_sweep.py.

---

## 📝 Files Modified

| File | Change | Status |
|------|--------|--------|
| `data/batching/deeplive.py` | Added CLIP normalization to collate_fn | ✅ Done |
| `train_deeplive_v2.py` | NEW - Uses Trainer class properly | ✅ Created |
| `trainer/trainer.py` | Added 'deeplive' strategy handling | ✅ Done |
| `train_deeplive.py` | Keep as backup, deprecated | ⚠️ Deprecated |

---

## 🏁 Success Criteria

The refactoring will be considered **complete** when:

1. ✅ `train_deeplive_v2.py` uses the `Trainer` class
2. ✅ All Trainer features work (ArcFace, curriculum, checkpointing, etc.)
3. ⏳ W&B logging looks identical to old flow - **NEEDS TESTING**
4. ⏳ Checkpoint management works (GCS upload, top-N) - **NEEDS TESTING**
5. ✅ Can run W&B sweeps with the new flow (override support added)
6. ✅ Adding a new backbone/dataset requires only:
   - A new experiment YAML
   - A new batching strategy (if needed)
   - **NOT** a new training script

---

## 🧪 Testing Needed

To fully verify the fixes, run:

```bash
# Test without W&B
cd DeepfakeBench/training
python train_deeplive_v2.py --config experiments/deeplive_vit_B16.yaml --dry-run

# Test with W&B
python train_deeplive_v2.py --config experiments/deeplive_vit_B16.yaml --wandb --wandb-project test-deeplive

# Full training run
python train_deeplive_v2.py --config experiments/deeplive_vit_B16.yaml --wandb --wandb-project deeplive-experiments
```

---

## Historical Note

The previous documentation (`checklist.md`) marked items as complete without running the verification checkpoints. **Do not mark anything complete until it's tested end-to-end.**

The phrase "Run training with X changes" means actually running a full training job and verifying outputs, not just checking that imports work.

---

## Next Steps

1. ~~Review this document and confirm understanding~~ ✅
2. ~~Start with Step 1 (CLIP normalization fix)~~ ✅
3. ~~Proceed to Step 2 (Trainer integration)~~ ✅
4. **Run actual training to verify** ⏳
5. Update checklist with real completion status


