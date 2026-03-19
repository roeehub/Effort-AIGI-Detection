# Training Pipeline Fixes - Jan 9, 2026

## Executive Summary

This document details critical fixes to the Effort-AIGI training pipeline addressing **training collapse** observed in the DF40 paired experiment. The root causes were:

1. **Identity leakage**: Same person appeared in train/val/test splits
2. **Identity frequency imbalance**: Identities with more fake methods dominated training

We implemented three key fixes and created a new combined data source that merges DF40 and DeepLive datasets.

---

## Problem Analysis

### Observed Failure

The ViT-B-16 LAION experiment on DF40 paired data showed training collapse around step 3,000-4,000:

| Metric | Expected | Observed | Interpretation |
|--------|----------|----------|----------------|
| `val_in_dist/auc` | 0.7+ stable | 0.75 → 0.4 crash | Model lost discrimination ability |
| `train/pred_balance/fake_ratio` | ~0.5 | Dropped to 0.0 | Model predicts ALL samples as "real" |
| `svd/near_zero_layers` | 0 | 5 → 6 → 7 | SVD layers died |
| `train/grad_norm` | Stable | Spiked to 50+ | Gradient explosion |

### Root Cause 1: Identity Leakage

**The split was done by pair, not by identity.**

The DF40 dataset contains:
- ~954 unique identities (real people)
- ~5,379 pairs (real + fake video combinations)
- 8 different fake generation methods

Original splitting:
```
Sample A (identity X, simswap) → Train
Sample B (identity X, facedancer) → Val  ← LEAKED!
Sample C (identity X, e4s) → Test  ← LEAKED!
```

The model learned to recognize **faces** (which identity), not **artifacts** (real vs fake). During validation, it saw the same faces and "recognized" them, giving false confidence.

### Root Cause 2: Identity Frequency Imbalance

Some identities have 8 fake methods (8 pairs), others have only 1 method (1 pair):

```
Identity A: 8 pairs → seen 8x per epoch
Identity B: 1 pair → seen 1x per epoch
```

This causes:
- Model focuses on high-frequency identities
- Rare identities under-represented in gradients
- Amplifies identity memorization problem

---

## Fixes Implemented

### Fix 1: Identity-Stratified Splitting

**File:** `data/sources/df40_paired.py` - `_split_samples()` function

**Before:**
```python
# Split by pairs (WRONG!)
random.shuffle(samples)
train = samples[:n_train]
val = samples[n_train:n_train+n_val]
test = samples[n_train+n_val:]
```

**After:**
```python
# Group by identity
by_identity = defaultdict(list)
for sample in samples:
    by_identity[sample.target_identity].append(sample)

# Split IDENTITIES (not pairs)
identities = list(by_identity.keys())
rng.shuffle(identities)
n_train_ids = int(len(identities) * train_split)
n_val_ids = int(len(identities) * val_split)

train_identities = set(identities[:n_train_ids])
val_identities = set(identities[n_train_ids:n_train_ids + n_val_ids])
test_identities = set(identities[n_train_ids + n_val_ids:])

# Assign ALL pairs per identity to that identity's split
for sample in samples:
    if sample.target_identity in train_identities:
        train_samples.append(sample)
    elif sample.target_identity in val_identities:
        val_samples.append(sample)
    else:
        test_samples.append(sample)

# Sanity check: verify no overlap
train_val_overlap = train_ids_check & val_ids_check
if train_val_overlap:
    raise ValueError("Identity leakage detected!")
```

**Effect:** 80% of identities in train, 10% in val, 10% in test. NO identity appears in multiple splits.

---

### Fix 2: Identity-Balanced Sampling

**File:** `data/batching/df40_paired.py` - `DF40PairedIterableDataset`

**Before:**
```python
# Iterate all samples (WRONG!)
for sample in self.samples:
    yield sample  # Identity X seen 8x, Identity Y seen 1x
```

**After:**
```python
# Group by identity at init time
self._samples_by_identity = defaultdict(list)
for sample in samples:
    self._samples_by_identity[sample.target_identity].append(sample)
self._identities = list(self._samples_by_identity.keys())

# Each epoch: ONE random method per identity
def _get_identity_balanced_samples(self, rng, worker_id, num_workers):
    selected = []
    for identity in self._identities:
        # Randomly pick ONE method for this identity this epoch
        sample = rng.choice(self._samples_by_identity[identity])
        selected.append(sample)
    rng.shuffle(selected)
    return selected[worker_id::num_workers]

def set_epoch(self, epoch: int):
    """Called by trainer at epoch start to vary random selection."""
    self._epoch = epoch
```

**Effect:** Every identity is seen exactly once per epoch. Different methods are randomly selected across epochs.

**Config option:**
```yaml
df40_paired:
  identity_balanced_sampling: true  # Default
```

---

### Fix 3: Combined Data Source

**File:** `data/sources/combined_paired.py` (NEW ~993 lines)

Combines DF40 (~954 identities) + DeepLive (~920 identities) = ~1,874 unique identities.

**Key components:**

#### 3.1 Unified Sample Wrapper
```python
@dataclass
class UnifiedPairedSample:
    identity: str           # Prefixed: "df40_XXX" or "deeplive_YYY"
    source: str             # 'df40' or 'deeplive'
    original_sample: Any    # DF40PairedSample or DeepLiveSample
    method: str             # Generation method name
    has_landmarks: bool     # True for DeepLive, False for DF40
    sample_id: str          # Unique identifier
```

#### 3.2 Identity Extraction

**DF40:** Uses `sample.target_identity` directly (parsed from video filename)

**DeepLive:** Extracts from `original_video_name` field:
```python
def extract_deeplive_identity(sample):
    # original_video_name: "cropped_ABC123xyz.mp4"
    # → identity: "ABC123xyz"
    name = sample.original_video_name
    if name.startswith('cropped_'):
        name = name[8:]
    if name.endswith('.mp4'):
        name = name[:-4]
    return name
```

#### 3.3 Cross-Source Split
```python
def split_samples_by_identity(unified_samples, train_split, val_split, seed, logger):
    # Pool ALL identities from both sources
    by_identity = defaultdict(list)
    for sample in unified_samples:
        by_identity[sample.identity].append(sample)
    
    # Split at identity level (same as Fix 1)
    identities = list(by_identity.keys())
    rng.shuffle(identities)
    # ... split and assign
```

#### 3.4 Combined Iterable Dataset
```python
class CombinedPairedIterableDataset(IterableDataset):
    """Handles both DF40 and DeepLive samples with proper landmark handling."""
    
    def _iterate_single_sample(self, unified_sample, rng):
        if unified_sample.source == 'df40':
            # Load from DF40 dataset, landmarks=None
            yield from self._iterate_df40_sample(unified_sample, rng)
        else:
            # Load from DeepLive dataset, landmarks available
            yield from self._iterate_deeplive_sample(unified_sample, rng)
```

---

### Supporting Changes

#### Trainer Updates (`trainer/trainer.py`)

1. **Epoch notification for varying random selection:**
```python
# At epoch start
if hasattr(train_loader.dataset, 'set_epoch'):
    train_loader.dataset.set_epoch(epoch)
```

2. **Correct epoch length calculation:**
```python
if strategy in ['df40_paired', 'combined_paired']:
    if hasattr(train_loader.dataset, '_identities'):
        # Use unique identities, not total samples
        epoch_len = len(train_loader.dataset._identities) * frames_per_sample // batch_size
```

#### DeepLive Dataset Update (`dataset/deeplive_dataset.py`)

Added `original_video_name` field to `DeepLiveSample`:
```python
@dataclass
class DeepLiveSample:
    # ... existing fields ...
    original_video_name: str = ""  # NEW: For identity extraction
```

Updated `discover_samples()` to populate from manifest JSON.

---

## New Experiment Configs

Created 4 new configs for the combined data source:

| Config File | Backbone | Hidden Size | Rank |
|-------------|----------|-------------|------|
| `combined_paired_vit_B16.yaml` | OpenAI ViT-B-16 | 768 | 767 |
| `combined_paired_vit_B16_laion.yaml` | LAION DataComp ViT-B-16 | 512 | 511 |
| `combined_paired_vit_B32.yaml` | OpenAI ViT-B-32 | 768 | 767 |
| `combined_paired_vit_L14.yaml` | OpenAI ViT-L-14 | 1024 | 1023 |

**Key config changes:**
```yaml
data_source: combined_paired  # New data source
learning_rate: 5.0e-5         # Reduced from 1e-4 for stability
evaluate_every_steps: 500     # Reasonable for debugging

combined_paired:
  identity_balanced_sampling: true  # One method per identity per epoch
  df40:
    pair_json: "dataset/df40_pairs/df40-pair-matching.json"
    gcs_bucket: "df40-frames-recropped-rfa85"
  deeplive:
    gcs_bucket: "live-deepfake-methods-real-and-fake-frames-cropped"
```

---

## Expected Training Behavior

### Metrics to Monitor

| Metric | Expected (Fixed) | Failure Mode |
|--------|------------------|--------------|
| `val_in_dist/auc` | Stable 0.6-0.8, gradual improvement | Sharp drop (identity leakage) |
| `train/pred_balance/fake_ratio` | Stable ~0.5 | Drops to 0 or 1 (collapse) |
| `svd/near_zero_layers` | 0 throughout | Increasing (SVD dying) |
| `train/loss` | Gradual decrease | Spikes or explosion |
| `train/grad_norm` | <5, stable | Spikes >50 |

### Logging to Expect

At pipeline initialization:
```
Combined Paired Data Source: Initializing
  - DF40 samples: 5,379 pairs from 954 identities
  - DeepLive samples: ~1,000 from ~920 identities
  - Total unified samples: ~6,300
  - Total unique identities: ~1,874

Data split BY IDENTITY (seed=737):
  - Train identities: 1,499 (80.0%)
  - Val identities: 187 (10.0%)
  - Test identities: 188 (10.0%)
  ✓ No identity overlap between splits
```

At each epoch:
```
Epoch 0 method distribution (worker 0): {'simswap': 180, 'facedancer': 175, ...}
```

---

## Files Changed Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `data/sources/df40_paired.py` | Modified | Identity-stratified `_split_samples()` |
| `data/batching/df40_paired.py` | Modified | Identity-balanced sampling in `DF40PairedIterableDataset` |
| `data/sources/combined_paired.py` | **New** | Combined DF40+DeepLive data source |
| `data/sources/__init__.py` | Modified | Registered `combined_paired` |
| `dataset/deeplive_dataset.py` | Modified | Added `original_video_name` to `DeepLiveSample` |
| `trainer/trainer.py` | Modified | `set_epoch()` call, epoch_len for combined_paired |
| `experiments/combined_paired_vit_B16.yaml` | **New** | OpenAI B16 config |
| `experiments/combined_paired_vit_B16_laion.yaml` | **New** | LAION B16 config |
| `experiments/combined_paired_vit_B32.yaml` | **New** | OpenAI B32 config |
| `experiments/combined_paired_vit_L14.yaml` | **New** | OpenAI L14 config |

---

## Debugging Checklist

If training still fails, check:

1. **Identity extraction working?**
   - Look for log: `Total unique identities: ~1,874`
   - If much lower, identity extraction may be broken

2. **Split done by identity?**
   - Look for log: `✓ No identity overlap between splits`
   - If missing or error, split logic not running

3. **Identity-balanced sampling active?**
   - Look for log: `Identity-balanced sampling: True`
   - Check `Samples per epoch: X (one method per identity)` not `(all pairs)`

4. **Epoch varying correctly?**
   - Method distribution should differ between epochs
   - If identical, `set_epoch()` not being called

5. **Frame loading working?**
   - Check for GCS errors in logs
   - Verify bucket access permissions

---

## Launch Commands

```bash
# From DeepfakeBench/training/

# Test locally (dry run)
./dev.sh shell-prod
python train_sweep.py --config experiments/combined_paired_vit_B16_laion.yaml --dry-run

# Launch on Vertex AI
./launch_experiment.sh combined-paired-experiments asia-southeast1 experiments/combined_paired_vit_B16_laion.yaml
```

---

## Version History

| Date | Author | Changes |
|------|--------|---------|
| Jan 9, 2026 | Copilot | Initial implementation of all fixes |

---

## References

- Failed experiment W&B run: (link to be added)
- DF40 dataset paper: [link]
- Effort method (ICML 2025): [link]
