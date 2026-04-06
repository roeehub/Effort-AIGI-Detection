# Phase 2: Comprehensive Training Investigation Plan

**Date:** February 10, 2026  
**Backbone:** B16 LAION (ViT-B-16-DataComp-XL)  
**Method:** Effort (ICML 2025) — SVD orthogonal subspace decomposition  
**Goal:** Maximize both in-distribution AND out-of-distribution deepfake detection

---

## Table of Contents

1. [Background & Motivation](#1-background--motivation)
2. [Current State of Results](#2-current-state-of-results)
3. [Data Inventory & Audit](#3-data-inventory--audit)
4. [Architecture of the Training System](#4-architecture-of-the-training-system)
5. [Experiment Plan: Phase 2 Sweep](#5-experiment-plan-phase-2-sweep)
6. [Code Changes Required](#6-code-changes-required)
7. [Validation & Testing Protocol](#7-validation--testing-protocol)
8. [Risk Analysis & Mitigations](#8-risk-analysis--mitigations)
9. [Timeline & Execution Order](#9-timeline--execution-order)

---

## 1. Background & Motivation

### What We've Built

The Effort method fine-tunes CLIP backbones by decomposing each linear layer via SVD: the top-r singular components are **frozen** (preserving CLIP's pretrained representations), while only the residual k = embed_dim − rank components are trained. An orthogonal loss ($\lambda_{\text{reg}}$) constrains the residual subspace to stay orthogonal to the frozen subspace.

### The Phase 1 Discovery

The B16 lambda sweep (January 18, 2026) established a critical finding: **λ=1.0 was pinning the solution**, preventing the model from learning. Relaxing λ to 0.01 or 0 unlocked training. Four successful configurations emerged:

| Config | λ | Head | ArcFace | Result |
|--------|---|------|---------|--------|
| `P1_lambda001` | 0.01 | CE | — | ✅ Strong baseline |
| `P1_lambda0` | 0.0 | CE | — | ✅ Upper bound (no constraint) |
| `P1_cosine_softmax` | 0.01 | ArcFace (m=0) | s: 10→18 | ✅ Cosine geometry baseline |
| `P1_arcface_conservative` | 0.01 | ArcFace (m=0.15) | s: 8→12 | ✅ Margins viable |

The winning checkpoint: `top_n_effort_20260119_step14000_auc0.9947_eer0.0218_B16_LAION.patched.pth`

### Why Phase 2?

The Phase 1 model is **excellent in-distribution** (AUC=0.9947 on DF40+DeepLive) but shows significant degradation on **out-of-distribution** faceswaps:

| Metric | DF40 In-Dist | VisoMaster OOD |
|--------|:---:|:---:|
| Best balanced accuracy | ~92.5% | 83.7% |
| Real preservation @ best | ~97% | 82.4% |
| Fake detection @ best | ~97% | 85.1% |
| Tier gradient | N/A | STRONG 91.7% > MOD 89.3% > MIN 79.7% |

The **tier gradient on VisoMaster** (a dataset never seen during training) proves the model IS learning real manipulation artifacts — it just needs more diverse training data to sharpen its decision boundary for harder swap methods.

---

## 2. Current State of Results

### 2.1 In-Distribution Performance (DF40 + DeepLive)

**Checkpoint:** `top_n_effort_20260119_step14000_auc0.9947_eer0.0218_B16_LAION.patched.pth`

Best in-dist threshold strategy: `prob=0.1, vote=5/8` → 92.5% accuracy (89.0% real / 97.2% fake)

Conservative production threshold: `prob=0.9, vote=2/8` → 91.2% real / 94.5% fake

### 2.2 Out-of-Distribution: VisoMaster (9 swap models, 3 tiers, 14,858 videos)

**At default threshold (0.5/4-of-8):**
- Real: 96.1% ✓ (great preservation)
- Fake: 56.9% ✗ (barely above chance for many models)

**At optimal OOD threshold (0.1/2-of-8):**
- Real: 82.4% | Fake: 85.1% | Balanced: 83.7%

**Per-model detection @ optimal threshold:**
| Model | Accuracy | Note |
|-------|:--------:|------|
| CSCS | 96.3% | SimSwap family — already well-covered by DF40 |
| Inswapper128 | 89.2% | InsightFace swapper |
| GhostFace-v3 | 84.4% | Novel architecture |
| SimSwap512 | 84.7% | Higher res SimSwap |
| InStyleSwapper256-C | 84.3% | StyleGAN-based |
| InStyleSwapper256-A | 83.0% | StyleGAN-based |
| GhostFace-v1 | 82.8% | Novel architecture |
| InStyleSwapper256-B | 81.9% | StyleGAN-based |
| GhostFace-v2 | 80.3% | Hardest model |

**Key Insight:** All models are ≥80% at the right threshold — the signal IS there, just weak. Training on this data should amplify it.

### 2.3 What We Don't Yet Know

- [ ] Which of the 4 Phase 1 configs produced the winning checkpoint (need to check W&B)
- [ ] Exact number of training steps for that checkpoint (filename says step14000)
- [ ] Per-method breakdown on DF40 in-distribution (which DF40 methods are weakest?)
- [ ] How much of the real data we're actually using vs. available

---

## 3. Data Inventory & Audit

### 3.1 Available Data Sources

| Source | Type | GCS Bucket | Samples | Methods | Landmarks | Currently Used |
|--------|------|------------|:-------:|:-------:|:---------:|:--------------:|
| **DF40 Paired** | Train+Val | `df40-frames-recropped-rfa85` | ~5,379 pairs | 8 FS methods | ❌ | ✅ Training |
| **DeepLive** | Train+Val | `live-deepfake-methods-real-and-fake-frames-cropped` | ~1,000 | 2 strategies | ✅ | ✅ Training |
| **VisoMaster** | Val only | `visomaster-cropped-frames` | ~14,858 videos | 9 swap models | ❌ | ✅ Validation only |
| **External Real** | Val only | `external-youtube-avspeech-frames` | ? | Real only | ❌ | ⚠️ Not confirmed |
| **DF40 Full** | Available | `df40-frames-recropped-rfa85` | ~40 methods | FR+EFS+FE too | ❌ | ❌ Unused |

### 3.2 DF40 Method Breakdown

The DF40 dataset has **40+ methods** across 4 categories. Currently only **Face Swapping (FS)** methods are used:

**Face Swapping (FS) — Currently in training:**
- e4s, facedancer, faceswap, fsgan, inswap, simswap, uniface, blendface, mobileswap, deepfacelab

**Face Reenactment (FR) — NOT used yet:**
- fomm, hyperreenact, MRAA, one_shot_free, pirender, facevid2vid, lia, mcnet, sadtalker, wav2lip, tpsm, danet

**Entire Face Synthesis GAN (EFS-GAN) — NOT used yet:**
- StyleGAN2, StyleGAN3, StyleGANXL, VQGAN, stargan, starganv2, styleclip, e4e

**Entire Face Synthesis Diffusion (EFS-Diff) — NOT used yet:**
- ddim, DiT, SiT, RDDM, Collaborative_Diffusion, pixart, sd1.5, sd2.1

**Face Editing (FE) — NOT used yet:**
- DALLE2-face, MidJourney

### 3.3 Data Split Architecture (Current: `combined_paired`)

```
combined_paired data source:
├── DF40 (~5,379 pairs, ~954 identities)
│   ├── 8 fake methods (FS only)
│   ├── Paired real frames (same identity)
│   └── 32 frames/pair, sparse to 8
├── DeepLive (~1,000 samples, ~920 identities)
│   ├── 2 strategies (edge_cases, minimal_processing)
│   ├── Paired real frames
│   └── 16 frames/sample, sparse to 8
└── Split: 90% train / 10% val / 0% test
    └── Identity-stratified (no identity leakage)
```

### 3.4 Critical Data Questions

| Question | Status | Action |
|----------|--------|--------|
| Are we using ALL DF40 FS methods? | ⚠️ `methods: null` in config → loads all from pair JSON | Verify pair JSON contents |
| Are we using FR/EFS methods from DF40? | ❌ Not paired; pair JSON only has FS | Could add unpaired loading |
| How much real data in DF40 paired? | ~954 unique identities (paired with fakes) | Not the full DF40 real set |
| External real data for training? | ❌ Only used in validation | Could add `external_youtube_avspeech` |
| VisoMaster for training? | ❌ No data source registered | **Major opportunity** |
| DeepLive strategies used? | Only `edge_cases`, `minimal_processing` | Check if more exist |
| Test split? | `test_split: 0.0` in configs | No held-out test set! |

---

## 4. Architecture of the Training System

### 4.1 Training Flow

```
train_sweep.py (entry point)
    │
    ├── load_base_configs()         ← defaults.yaml + effort.yaml
    ├── wandb.init(config=single_cfg)  ← experiment YAML
    ├── apply_all_wandb_overrides() ← merge flat W&B keys
    ├── download_assets_from_gcs()  ← backbone weights, checkpoints
    │
    ├── create_data_pipeline()      ← factory pattern
    │   ├── 'manifest'             ← legacy; JSON manifest + parquet
    │   ├── 'deeplive'             ← GCS bucket + landmarks
    │   ├── 'df40_paired'          ← pair JSON + GCS
    │   └── 'combined_paired' ✓    ← DF40 + DeepLive unified
    │
    ├── DETECTOR['effort']()        ← EffortDetector
    │   ├── Load CLIP backbone (HF or OpenCLIP)
    │   ├── apply_svd_openclip_attn() or apply_svd_hf_attn()
    │   │   └── Replace nn.Linear → SVDResidualLinear (per layer)
    │   └── Optional ArcFace head
    │
    └── Trainer(model, train_loader, val_loader, ...)
        ├── CheckpointingMixin   ← top-N GCS checkpoints
        ├── EarlyStoppingMixin   ← patience-based
        ├── GroupDROMixin        ← per-method loss reweighting
        ├── CurriculumMixin      ← lesson gates
        ├── ArcFaceMixin         ← s/m annealing
        ├── ValidationMixin      ← state tracking
        └── ReportingMixin       ← CSV/TXT reports
```

### 4.2 Key Training Parameters (from Phase 1 winning configs)

| Parameter | Phase 1 Value | Notes |
|-----------|:---:|-------|
| `data_source` | `combined_paired` | DF40 + DeepLive |
| `rank` | 760 | k=8 trainable directions (embed_dim=768) |
| `lambda_reg` | 0.01 or 0.0 | Relaxed orthogonal constraint |
| `learning_rate` | 2e-4 | With cosine warmup |
| `weight_decay` | 0.05 | |
| `frames_per_batch` | 32 | |
| `frames_per_video` | 8 | |
| `total_training_steps` | 65,000 | ~30 epochs |
| `lr_scheduler` | cosine_with_warmup | 1000 warmup steps |
| `augmentation` | base_only | No landmarks (DF40 constraint) |
| `train_split` | 0.9 | Identity-stratified |

### 4.3 Available Training Mechanisms (Mixins)

| Mechanism | Status | Relevance |
|-----------|--------|-----------|
| **Group DRO** | ✅ Implemented | Critical for multi-source training with method imbalance |
| **Curriculum** | ✅ Implemented | Lesson gates with metric thresholds + plateau detection |
| **ArcFace annealing** | ✅ Implemented | s_start→s_end, margin curriculum |
| **Lambda annealing** | ✅ Implemented | λ_start→λ_end over N steps |
| **Early stopping** | ✅ Implemented | Patience-based |
| **Identity-balanced sampling** | ✅ In combined_paired | One method/identity/epoch |
| **Property balancing** | ✅ In manifest source | Category-weighted sampling |
| **Checkpoint resume** | ✅ `load_base_checkpoint` | Continue from Phase 1 checkpoint |

---

## 5. Experiment Plan: Phase 2 Sweep

### 5.1 Design Principles

1. **Data diversity is the #1 lever** — more methods > more tuning
2. **Always preserve real accuracy** — must stay ≥90% on all real sources
3. **Test what you validate on** — held-out test set required
4. **Curriculum over brute-force** — introduce hard data gradually
5. **Group DRO for fairness** — no method left behind

### 5.2 Data Strategy

#### Split Design (Train / Val / Test)

```
TRAIN DATA:
├── DF40 Paired (FS methods)      ← existing, identity-stratified 90% train
├── DeepLive (all strategies)     ← existing, identity-stratified 90% train
└── VisoMaster (NEW — subset)     ← 70% of VisoMaster identities for training
    ├── 6 of 9 swap models (train set)
    │   CSCS, GhostFace-v1, GhostFace-v2, InStyleSwapper256-A,
    │   InStyleSwapper256-B, Inswapper128
    └── All tiers (MINIMAL, MODERATE, STRONG)

IN-DIST VALIDATION:
├── DF40 Paired (10% val split)
├── DeepLive (10% val split)
└── VisoMaster (15% of identities, SAME 6 models as train)

OOD TEST (NEVER trained on):
├── VisoMaster held-out models (3 models never seen):
│   GhostFace-v3, InStyleSwapper256-C, SimSwap512
├── VisoMaster held-out identities (15% of identities, all models)
└── External YouTube real (unseen real distribution)
```

**Rationale for VisoMaster model holdout:**
- GhostFace-v3 is mid-difficulty (84.4% at optimal threshold) — good OOD test
- InStyleSwapper256-C is similar to A/B but held out for generalization test
- SimSwap512 is a variant of SimSwap (already in DF40) — tests within-family generalization
- This gives us a true OOD signal: "can the model detect swap models it's never seen?"

### 5.3 Experiment Matrix

#### Group A: Data Composition (which data to use)

| ID | Name | Train Data | Purpose |
|----|------|-----------|---------|
| **A1** | `df40_only` | DF40 paired (FS) only | Baseline — isolate DF40 contribution |
| **A2** | `df40_deeplive` | DF40 + DeepLive (current) | Reproduce Phase 1 baseline |
| **A3** | `df40_deeplive_viso` | DF40 + DeepLive + VisoMaster (6 models) | **Primary experiment** |
| **A4** | `df40_viso_no_deeplive` | DF40 + VisoMaster (no DeepLive) | Isolate VisoMaster contribution vs DeepLive |

All Group A experiments use the Phase 1 best hyperparameters (λ=0.01, CE head, lr=2e-4, rank=760).

#### Group B: Training Strategy (how to train)

Starting from the best data composition (expected: A3), vary the training approach:

| ID | Name | Strategy | Purpose |
|----|------|----------|---------|
| **B1** | `flat_training` | All data from step 0, uniform sampling | Simplest baseline |
| **B2** | `group_dro` | All data + Group DRO (β=3.0) | Per-method fairness |
| **B3** | `curriculum_easy_first` | Phase 1→add VisoMaster STRONG→MOD→MIN | Curriculum by tier difficulty |
| **B4** | `curriculum_method_first` | Phase 1→add easy models→hard models | Curriculum by model difficulty |
| **B5** | `checkpoint_resume` | Load Phase 1 checkpoint, train on new data | Transfer learning |
| **B6** | `checkpoint_resume_dro` | Load Phase 1 checkpoint + Group DRO | Best of both worlds |

#### Group C: Loss & Regularization

| ID | Name | Changes | Purpose |
|----|------|---------|---------|
| **C1** | `ce_lambda001` | CE + λ=0.01 (Phase 1 winner) | Baseline loss |
| **C2** | `arcface_m015_s12` | ArcFace m=0.15, s=8→12, λ=0.01 | Conservative margin |
| **C3** | `cosine_softmax` | ArcFace m=0.0, s=10→18, λ=0.01 | Cosine geometry only |
| **C4** | `lambda0_arcface` | ArcFace m=0.15 + λ=0.0 | Maximum flexibility |
| **C5** | `lambda_anneal_arcface` | λ: 0.01→0.001 + ArcFace m=0.15 | Gradual relaxation |

#### Group D: Scale & Efficiency

| ID | Name | Changes | Purpose |
|----|------|---------|---------|
| **D1** | `rank760_k8` | rank=760 (Phase 1) | Baseline k |
| **D2** | `rank752_k16` | rank=752, more trainable directions | More capacity |
| **D3** | `rank744_k24` | rank=744 | Even more capacity |
| **D4** | `batch64_video` | 8 videos × 8 frames | Video-level batching |
| **D5** | `batch128_accum` | 32 frames + 4× gradient accumulation | Larger effective batch |

### 5.4 Priority Execution Order

**Night 1: Data composition validation (4 runs)**
→ A1, A2, A3, A4 — all with same hyperparameters, differ only in data
→ Answer: "Does adding VisoMaster help? Does DeepLive help on top of DF40?"

**Night 2: Training strategy (3-4 runs)**
→ B1, B2, B5, B6 — using best data from Night 1
→ Answer: "Is Group DRO needed? Does checkpoint resume beat training from scratch?"

**Night 3: Loss functions (3 runs)**
→ C1, C2, C3 — using best data + strategy from Nights 1-2
→ Answer: "Does ArcFace help now that we have more diverse data?"

**Night 4: Curriculum (2 runs)**
→ B3, B4 — requires lesson gate configs and potentially new data control logic
→ Answer: "Does curriculum help, or is flat training sufficient?"

**Night 5: Scale & fine-tuning (2-3 runs)**
→ D1-D5 selected based on prior nights
→ Answer: "Can we squeeze more from larger k or larger batches?"

### 5.5 Success Criteria

| Metric | Phase 1 Baseline | Target |
|--------|:---:|:---:|
| DF40 in-dist AUC | 0.9947 | ≥ 0.99 (maintain) |
| DF40 in-dist balanced acc | ~92.5% | ≥ 90% (maintain) |
| VisoMaster OOD balanced acc (all models) | 83.7% | **≥ 88%** |
| VisoMaster held-out models (OOD) | ~82% (est.) | **≥ 85%** |
| VisoMaster tier MINIMAL | 79.7% | **≥ 85%** |
| Real accuracy (all sources) | 96.6% | ≥ 93% |
| Per-model minimum fake accuracy | 80.3% (GhostFace-v2) | **≥ 85%** |

---

## 6. Code Changes Required

### 6.1 NEW: VisoMaster Training Data Source

**Priority: 🔴 CRITICAL — blocks all Group A experiments with VisoMaster**

**File to create:** `data/sources/visomaster.py`

**Scope:**
- Register as `@register_data_source('visomaster')` 
- Load VisoMaster samples from GCS cropped bucket (same as validation source)
- Support filtering by `swap_models` list (for train/OOD split)
- Support filtering by `tiers` list (for curriculum)
- Identity-stratified splitting (extract identity from sample_id)
- Return `DataPipelineResult` with train/val loaders
- No landmarks (like DF40)

**Key design decision:** We need both a standalone source AND integration into `combined_paired`. Options:
1. **Standalone `visomaster` source** — simplest, works for A4
2. **Extend `combined_paired` to support 3+ sources** — needed for A3
3. **New `multi_paired` source** — most general

**Recommendation:** Option 2 (extend `combined_paired`) is best — minimal new code, reuses identity-stratified splitting.

**Estimated effort:** ~200 lines (adaptation of existing validation source + paired dataset pattern)

### 6.2 EXTEND: `combined_paired.py` → Support VisoMaster as 3rd Source

**Priority: 🔴 CRITICAL**

**Changes needed:**
```python
# In combined_paired config:
combined_paired:
  identity_balanced_sampling: true
  
  df40:
    enabled: true
    pair_json: "dataset/df40_pairs/df40-pair-matching.json"
    gcs_bucket: "df40-frames-recropped-rfa85"
    ...
  
  deeplive:
    enabled: true
    gcs_bucket: "live-deepfake-methods-real-and-fake-frames-cropped"
    ...
  
  visomaster:           # ← NEW
    enabled: true
    gcs_bucket: "visomaster-cropped-frames"
    swap_models: [CSCS, GhostFace-v1, GhostFace-v2, ...]  # training subset
    tiers: null         # all tiers
    ...
  
  train_split: 0.85
  val_split: 0.10
  test_split: 0.05     # ← NEW: actually hold out test data
```

**Functions to modify:**
- `create_combined_paired_pipeline()` — add VisoMaster sample loading
- `create_unified_samples_from_visomaster()` — new helper (analogous to deeplive/df40)
- `CombinedPairedIterableDataset.__iter__()` — handle VisoMaster samples (no landmarks)
- Identity-stratified split — works unchanged (just more identities)

### 6.3 NEW: Held-Out Model OOD Validation

**Priority: 🟡 HIGH — needed for proper evaluation**

**File to modify:** `validate_custom_sources.py`

**Changes needed:**
- Add `--visomaster_held_out_models` flag to specify OOD models
- OR: create separate validation configs for "train models" vs "OOD models"
- Ensure validation reports clearly separate in-dist vs OOD performance

### 6.4 EXTEND: Curriculum Lesson Data Control for VisoMaster

**Priority: 🟡 HIGH — needed for Group B experiments**

**Current `CurriculumMixin`** supports lesson gates (threshold checks) but lesson data switching needs:
- `lesson_data_control` config already parsed in `train_sweep.py`
- Need to implement actual data source switching between lessons
- For VisoMaster tiers: lesson 1 = STRONG only, lesson 2 = +MODERATE, lesson 3 = +MINIMAL

**File to modify:** `trainer/trainer.py` (train_one_epoch method) and potentially `data/sources/combined_paired.py` (dynamic tier filtering)

### 6.5 EXTEND: Group DRO Method Mapping for 3-Source Data

**Priority: 🟡 HIGH — needed for Group B experiments**

**Current state:** `GroupDROMixin` expects `method_mapping` dict from data pipeline.

**Changes needed:**
- `combined_paired.py` should emit method_mapping that includes VisoMaster swap models
- Format: `{"df40_simswap": 0, "df40_e4s": 1, ..., "deeplive_edge_cases": 10, "visomaster_CSCS": 12, ...}`
- Group DRO will then reweight per-method loss to avoid ignoring hard VisoMaster models

### 6.6 CONFIG: Experiment YAML Templates

**Priority: 🟢 MEDIUM — needed before launch**

Create experiment YAML files for each Group A-D experiment. Each needs:
- Data source configuration
- Hyperparameters
- Training schedule
- Evaluation settings
- GCS asset paths

**Location:** `experiments/phase2/`

### 6.7 EXTEND: Add Test Split Support

**Priority: 🟡 HIGH**

**Current issue:** All configs use `test_split: 0.0` — there's no held-out test set.

**Changes needed:**
- Identity-stratified splitting already supports 3-way split — just set `test_split: 0.05`
- Add test evaluation pass after training completes
- Report test metrics separately from validation

---

## 7. Validation & Testing Protocol

### 7.1 Multi-Level Evaluation

Every experiment will be evaluated at 3 levels:

| Level | Data | When | Purpose |
|-------|------|------|---------|
| **In-Dist Val** | DF40 val + DeepLive val + VisoMaster val (same models) | Every 500 steps | Training signal |
| **OOD Val** | VisoMaster held-out models (GhostFace-v3, InStyleSwapper256-C, SimSwap512) | Every 2000 steps | Generalization signal |
| **Final Test** | External YouTube real + all VisoMaster models + all tiers | End of training | Final metric |

### 7.2 Metric Dashboard

For each run, track in W&B:

**Primary metrics (for checkpointing):**
- `val/auc` — in-dist AUC (for early stopping / checkpoint selection)
- `val/balanced_accuracy` — in-dist balanced accuracy
- `ood/balanced_accuracy` — OOD balanced accuracy

**Per-method breakdown:**
- `val/method/{method_name}/accuracy` — for each DF40, DeepLive, VisoMaster method
- `ood/method/{model_name}/accuracy` — for held-out VisoMaster models

**Diagnostic:**
- `group_dro/ema_loss/{method}` — if using Group DRO
- `svd/S_residual_mean` — SVD residual health
- `train/collapse_warning` — logit collapse detection

### 7.3 Threshold Analysis Protocol

After each run:
1. Download frames report CSV from GCS
2. Run `analyze_visomaster.py` grid search (prob × vote thresholds)
3. Find optimal operating point at real_acc ≥ 90% and ≥ 95%
4. Compare across all experiments at the **same** threshold

---

## 8. Risk Analysis & Mitigations

### Risk 1: Adding VisoMaster Degrades DF40 Performance
**Probability:** Medium  
**Impact:** High  
**Mitigation:** 
- Group DRO ensures no source dominates
- Monitor DF40 in-dist AUC every 500 steps
- Early stopping on combined metric (0.5 × in-dist + 0.5 × OOD)
- A1 (DF40 only) baseline will quantify any regression

### Risk 2: Identity Leakage Across Sources
**Probability:** Low  
**Impact:** Critical  
**Mitigation:**
- Identity prefixing (`df40_`, `deeplive_`, `visomaster_`) prevents cross-source collision
- Identity-stratified splitting with explicit overlap check
- VisoMaster IDs are YouTube video IDs (different from DF40's celebrity IDs)

### Risk 3: VisoMaster Overwhelming Smaller Sources
**Probability:** High — VisoMaster has 9,458 fake videos vs DF40's 5,379 pairs  
**Impact:** Medium  
**Mitigation:**
- Identity-balanced sampling (one sample/identity/epoch)
- `source_weights` config to control sampling ratio
- Group DRO per-method reweighting

### Risk 4: Curriculum Complexity Causing Training Instability
**Probability:** Medium  
**Impact:** Medium  
**Mitigation:**
- B1 (flat training) as baseline — only try curriculum if flat isn't enough
- Conservative lesson gates (wait for plateau, not aggressive thresholds)
- Lambda annealing for smooth transitions

### Risk 5: Overfitting to VisoMaster at Expense of Generalization
**Probability:** Low-Medium  
**Impact:** High  
**Mitigation:**
- 3 VisoMaster models completely held out from training
- 15% of identities held out even for training models
- External YouTube real as unseen real distribution test
- If overfit: reduce VisoMaster sampling weight, increase augmentation

---

## 9. Timeline & Execution Order

### Pre-Requisite: Code Changes (2-3 days)

| Task | Priority | Effort | Blocks |
|------|----------|--------|--------|
| Create `visomaster` data source registration | 🔴 | 1 day | All A3/A4 experiments |
| Extend `combined_paired` for 3 sources | 🔴 | 1 day | A3 experiments |
| Add test split support | 🟡 | 0.5 day | Proper evaluation |
| Create Phase 2 experiment YAMLs | 🟡 | 0.5 day | All launches |
| Extend Group DRO method mapping | 🟡 | 0.5 day | B2/B6 experiments |
| Extend curriculum for tier control | 🟢 | 1 day | B3/B4 experiments |

### Execution: Experiment Nights

| Night | Experiments | Goal | Dependencies |
|-------|-----------|------|-------------|
| **1** | A1, A2, A3, A4 | Data composition | VisoMaster source ready |
| **2** | B1, B2, B5, B6 | Training strategy | Night 1 results |
| **3** | C1, C2, C3 | Loss functions | Night 2 results |
| **4** | B3, B4 | Curriculum | Curriculum code + Night 2-3 results |
| **5** | D1-D5 (selected) | Scale/capacity | Night 3-4 results |
| **6** | Final champion run | Best combination | All prior results |

### Post-Experiment: Analysis (1-2 days)

1. Grid search on all checkpoints (prob × vote threshold)
2. Per-model accuracy comparison across all experiments
3. In-dist vs OOD scatter plot
4. Champion model selection
5. Production threshold recommendation

---

## Appendix A: Phase 1 Config Reference

The 4 successful Phase 1 configs all share:
```yaml
data_source: combined_paired
backbone:
  variant: "ViT-B-16-DataComp-XL"
  source: "laion"
  hidden_size: 512
  resolution: 224
  apply_svd_to_in_proj: true
rank: 760                         # k=8 trainable directions
learning_rate: 2.0e-4
weight_decay: 0.05
lr_scheduler: "cosine_with_warmup"
lr_scheduler_warmup_steps: 1000
total_training_steps: 65000
nEpochs: 30
frames_per_batch: 32
frames_per_video: 8
augmentation:
  version: "base_only"
combined_paired:
  identity_balanced_sampling: true
  df40:
    gcs_bucket: "df40-frames-recropped-rfa85"
    methods: null                 # All FS methods from pair JSON
  deeplive:
    gcs_bucket: "live-deepfake-methods-real-and-fake-frames-cropped"
  train_split: 0.9
  val_split: 0.1
  test_split: 0.0
```

Differences between the 4 configs:

| Config | λ | ArcFace | m | s_start→s_end |
|--------|---|---------|---|:---:|
| P1_lambda001 | 0.01 | ❌ CE | — | — |
| P1_lambda0 | 0.00 | ❌ CE | — | — |
| P1_cosine_softmax | 0.01 | ✅ (m=0) | 0.0 | 10→18 |
| P1_arcface_conservative | 0.01 | ✅ | 0.15 | 8→12 |

## Appendix B: GCS Bucket Reference

| Bucket | Content | Used For |
|--------|---------|----------|
| `df40-frames-recropped-rfa85` | DF40 cropped frames (RFA 0.85) | Training (DF40 paired) |
| `live-deepfake-methods-real-and-fake-frames-cropped` | DeepLive paired frames | Training (DeepLive) |
| `visomaster-cropped-frames` | VisoMaster cropped frames | Validation → **Training (Phase 2)** |
| `visomaster-full-frames` | VisoMaster full frames + tier data | Tier metadata |
| `external-youtube-avspeech-frames` | YouTube real faces | Validation (external real) |
| `base-checkpoints/effort-aigi/` | CLIP backbone weights | Model initialization |
| `training-job-outputs/best_checkpoints/` | Training checkpoints | Checkpoint resume |

## Appendix C: Code File Reference

| File | Purpose | Phase 2 Changes |
|------|---------|----------------|
| `data/sources/__init__.py` | Data source registry | No change |
| `data/sources/combined_paired.py` | DF40+DeepLive pipeline | **Extend for VisoMaster** |
| `data/sources/visomaster.py` | **NEW** VisoMaster training source | **Create** |
| `data/validation_sources.py` | Validation data loading | Extend for OOD reporting |
| `validate_custom_sources.py` | Validation entry point | Add OOD model flags |
| `trainer/trainer.py` | Main training logic | Minor: curriculum data switching |
| `trainer/mixins/curriculum_mixin.py` | Lesson gates | Extend for tier-based curriculum |
| `trainer/mixins/group_dro_mixin.py` | Per-method DRO | No change (just new method mapping) |
| `detectors/effort_detector.py` | Effort model | No change |
| `train_sweep.py` | Training entry point | Minor: test split handling |
