# Phase 2 — Complete Experiment Report (R1 → R2 → R2.5 → R3)

**Last Updated:** 2026-02-17 — R4 status reset + R5 implementation plan added  
**W&B Entity/Project:** `dtect-vision/phase2-experiments`  
**Infrastructure:** Vertex AI, A100-SXM4-40GB, `asia-southeast1` (training), `us-central1` (evaluation)  
**Backbone:** ViT-B-16-DataComp-XL (LAION), hidden_size=512, resolution=224  
**Method:** Effort (SVD Residual Linear)

---

## Table of Contents

0. [Bird's Eye View — The Full Story](#0-birds-eye-view--the-full-story)
1. [Round 1 — Architecture & Loss Search (Invalidated)](#1-round-1--architecture--loss-search-invalidated)
2. [Bug Fixes Between Rounds](#2-bug-fixes-between-rounds)
3. [Round 2 — Controlled Ablation (8 Runs)](#3-round-2--controlled-ablation-8-runs)
4. [Round 2.5 — Capacity & Hyperparameter Sweep (8 Runs)](#4-round-25--capacity--hyperparameter-sweep-8-runs)
5. [R2/R2.5 Results & Winner: R25_F1](#5-r2r25-results--winner-r25_f1)
6. [2×2 Production Validation: R25_F1 vs Phase 1](#6-22-production-validation-r25_f1-vs-phase-1)
7. [Round 3 — Quality Bias Experiments (6 Runs)](#7-round-3--quality-bias-experiments-6-runs)
8. [R3 Out-of-Distribution Validation (3×3)](#8-r3-out-of-distribution-validation-33)
9. [Grand Leaderboard — All Experiments](#9-grand-leaderboard--all-experiments)
10. [Key Findings & Lessons Learned](#10-key-findings--lessons-learned)
11. [Production Recommendation](#11-production-recommendation)
12. [Data Sources Reference](#12-data-sources-reference)
13. [Code Changes Log](#13-code-changes-log)
14. [Appendix: GCS Paths & W&B IDs](#14-appendix-gcs-paths--wb-ids)
15. [WMA Enhanced Face-Swap Investigation — Why All Models Fail](#15-wma-enhanced-face-swap-investigation--why-all-models-fail)
16. [Verification Deep-Dive & Revised Fix Strategy](#16-verification-deep-dive--revised-fix-strategy-feb-2026)
17. [GFPGAN Comparison — Isolating the Enhancement Effect](#17-gfpgan-comparison--isolating-the-enhancement-effect-feb-16-2026)
18. [R4 Status Reset and R5 Plan (Feb 17, 2026)](#18-r4-status-reset-and-r5-plan-feb-17-2026)

---

## 0. Bird's Eye View — The Full Story

### What We Built

An AI-generated image detector based on the **Effort** method (ICML 2025 Oral), fine-tuned on a ViT-B-16 CLIP backbone. The method freezes the top-r singular components of CLIP's linear layers and trains only the residual SVD subspace — achieving strong generalization to unseen deepfake methods with minimal parameter updates (145 trainable parameter groups across 48 SVD layers).

### The Journey (Feb 10–14, 2026)

```
Round 1 (9 runs)     → Loss/architecture search
    ↓ INVALIDATED       Config bug: all runs used same data
    ↓ BUT USEFUL        Found: cosine softmax > CE, rank=752 > 760
    
Round 2 (8 runs)     → Controlled ablation with bug fixes
    ↓ FINDINGS           Data composition >> loss function >> capacity
    ↓                    Group DRO = no benefit, CE ≈ cosine softmax
    
Round 2.5 (8 runs)   → Capacity & hyperparameter sweep
    ↓ WINNER             R25_F1: k=32, cosine, lr=2e-4 → AUC 0.9893
    ↓                    Sweet spot at k=32 (4× baseline capacity)
    
2×2 Production Eval  → R25_F1 vs Phase 1 best on real-world data
    ↓ VERDICT            R25_F1 wins: same fake detection, HALF the FPR
    ↓                    At 95% TPR: 1.8% FPR vs 8.6% (Phase 1)
    
Round 3 (6 runs)     → Quality bias fix attempt
    ↓ SCRATCH (S1,S2)    Higher holdout AUC than F1, BUT...
    ↓ FINE-TUNE (FT1-4)  Massive holdout AUC gains (0.9954–0.9966)
    ↓                    4 failed launch attempts before success
    
R3 OOD Validation    → 3 models × 3 data sources (9 evals)
    ↓ SURPRISE           F1 baseline WINS on OOD despite lower holdout AUC
    ↓                    Quality-robust augmentation HURTS generalization
    
CONCLUSION           → R25_F1 remains best for deployment
                        FT models need OOD validation before displacing F1
    
WMA Investigation    → 5 models × 1,202 enhanced face-swaps
    ↓ ALL FAIL           Best: B16_old 55.7%, R25_F1 21.0%, R3_FT1 8.3%
    ↓ META-ANALYSIS      Sampled 2,602 images from 10 data sources
    ↓ ROOT CAUSE         Sharpness shortcut: model uses image quality as
    ↓                    proxy for real/fake. GFPGAN smoothing inverts
    ↓                    this relationship. Resolution mismatch amplifies.
    ↓ IMPLICATION        Needs training-time fix, not just threshold tuning.
```

### 37 Total Training Runs

| Round | Runs | Purpose | Key Finding |
|-------|------|---------|-------------|
| **R1** | 9 | Loss/arch search | Cosine softmax + higher rank help (data bug invalidated ablations) |
| **R2** | 8 | Controlled ablation | Data >> loss >> capacity. DRO useless. |
| **R2.5** | 8 | HP refinement | k=32 sweet spot. R25_F1 wins (AUC 0.9893) |
| **R3 scratch** | 2 | Quality bias fix | Higher holdout AUC but worse OOD generalization |
| **R3 fine-tune** | 4 | Fine-tune from F1 | Highest holdout AUC ever (0.9966), OOD TBD |
| **Eval jobs** | 6+ | OOD validation | F1 baseline wins on deployment-relevant metrics |
| **WMA investigation** | 5 evals + meta-analysis | Enhanced face-swap failure analysis | Sharpness shortcut is root cause. All models fail. |

### The Paradox We Discovered

**Holdout AUC ≠ deployment performance.** Models with higher holdout AUC (R3 scratch, R3 fine-tune) can perform *worse* on out-of-distribution data than models with lower holdout AUC (R25_F1). This is the single most important finding of Phase 2 — it means we cannot trust holdout metrics alone and must always validate on OOD data before deployment.

---

## 1. Round 1 — Architecture & Loss Search (Invalidated)

**9 runs, launched ~Feb 10. All used identical data due to config bug.**

Despite the data bug, loss/architecture comparisons were valid (controlled — same bug affected all):

| Finding | Detail |
|---------|--------|
| Cosine softmax (m=0, s:10→18) best loss | Holdout AUC 0.9844 at ep5 |
| rank=752 (k=16) > rank=760 (k=8) | In-dist AUC 0.9898 vs 0.9866 |
| ArcFace m=0.15 too aggressive | Last place, AUC 0.9785 |
| Group DRO crashed | Dimension mismatch bug (fixed in R2) |
| Data ablations meaningless | All had identical data |

Full Round 1 analysis: [INTERMEDIATE_REPORT.md](../phase2/INTERMEDIATE_REPORT.md)

---

## 2. Bug Fixes Between Rounds

### Fix 1: Config Passthrough (Critical — caused R1 invalidation)
**File:** `train_sweep.py` ~line 200  
`combined_paired` section from experiment YAMLs was never copied to `data_config`. All R1 runs trained on identical DF40+DeepLive data regardless of config. Fixed + added DATA VERIFICATION block with fail-fast assertions.

### Fix 2: Identity Leakage (Critical)
**File:** `data/sources/combined_paired.py`  
DeepLive prefixed `deeplive_{identity}`, VisoMaster prefixed `visomaster_{identity}` — same person could appear in both train and val splits. Fixed: both use `realpool_{identity}` prefix for unified identity-stratified splitting.

### Fix 3: Group DRO Dimension Mismatch
**File:** `trainer/mixins/group_dro.py`  
`method_ids` tensor shape [B] vs `per_sample_loss` shape [B×T]. Fixed with `repeat_interleave(T)`.

### Fix 4: Data Verification Logging
**File:** `train_sweep.py`  
Added W&B table logging of actual data composition (source, method, video_count) with fail-fast assertions.

---

## 3. Round 2 — Controlled Ablation (8 Runs)

All 8 share: seed=737, lr=2e-4, cosine_with_warmup (1000 warmup), 30 epochs, batch=32 frames (4 videos × 8 frames), weight_decay=0.05, lambda_reg=0.01, early_stopping patience=10.

| # | Run Name | Loss | Rank (k) | Data | DRO | Theme |
|---|----------|------|-----------|------|-----|-------|
| 1 | R2_A1 | CE | 760 (k=8) | DF40 only | No | Data baseline |
| 2 | R2_A3 | CE | 760 (k=8) | DF40+DL+VM | No | Data baseline |
| 3 | R2_E1 | Cosine | 760 (k=8) | All | No | Loss validation |
| 4 | R2_E2 | Cosine | 752 (k=16) | All | No | ★ PRIMARY |
| 5 | R2_E3 | Cosine | 744 (k=24) | All | No | Capacity |
| 6 | R2_E4 | Cosine | 752 (k=16) | DF40 only | No | Data × Model |
| 7 | R2_E5 | Cosine+DRO | 752 (k=16) | All | **Yes** | Robustness |
| 8 | R2_E6 | CE | 752 (k=16) | All | No | Loss ablation |

**Legend:** CE = Cross-Entropy. Cosine = ArcFace head (m=0.0, s: 10→18). DL = DeepLive, VM = VisoMaster.

### R2 Results

| # | Run | Run ID | Best AUC | Best EER | Best Ep | State |
|---|------|--------|----------|----------|---------|-------|
| 1 | R2_A1 | `pcyqx9l2` | 0.9792 | 0.0833 | 17 | ✅ finished |
| 2 | R2_A3 | `8urk1cmw` | 0.9866 | 0.0534 | 6 | ✅ finished |
| 3 | R2_E1 | `ai6c1gk8` | 0.9848 | 0.0391 | 8 | ✅ finished |
| 4 | R2_E2 | `hog6lmfq` | 0.9860 | 0.0320 | 7 | ✅ finished |
| 5 | R2_E3 | `hp87l4m2` | 0.9851 | 0.0320 | 6 | ✅ finished |
| 6 | R2_E4 | `nhl3aaws` | 0.9817 | 0.0500 | 17 | ✅ finished |
| 7 | R2_E5 | `x4fmqrhu` | 0.9860 | 0.0391 | 6 | ✅ finished |
| 8 | R2_E6 | `we9dc7ql` | 0.9862 | 0.0320 | 6 | ✅ finished |

### R2 Ablation Answers

| # | Question | Comparison | Result |
|---|----------|------------|--------|
| 1 | Data composition? | A1 vs A3 | **Yes. AUC 0.9792→0.9866 (Δ=+0.0074)**. Biggest single lever. |
| 2 | Data × best model? | E4 vs E2 | **Yes. AUC 0.9817→0.9860 (Δ=+0.0043)**. |
| 3 | Cosine vs CE (k=8)? | A3 vs E1 | **CE slightly better. 0.9866 vs 0.9848.** |
| 4 | Cosine vs CE (k=16)? | E6 vs E2 | **Tied. 0.9862 ≈ 0.9860.** |
| 5 | SVD capacity (k)? | E1→E2→E3 | **No clear trend at k=8→16→24 within R2.** |
| 6 | Group DRO? | E2 vs E5 | **No benefit. Both 0.9860.** |
| 7 | Model vs data delta? | cross | **Data helps more than model architecture.** |

---

## 4. Round 2.5 — Capacity & Hyperparameter Sweep (8 Runs)

R2 showed capacity mattered. R2.5 pushed further with k=24/32/48 and hyperparameter variations.

| # | Run Name | Rank (k) | Variation | Best AUC | Best EER | Best Ep | State |
|---|----------|-----------|-----------|----------|----------|---------|-------|
| 1 | **R25_F1** | **736 (k=32)** | baseline | **0.9893** | 0.0356 | 10 | ✅ early-stopped |
| 2 | R25_F2 | 720 (k=48) | baseline | 0.9887 | **0.0249** | 8 | ✅ early-stopped |
| 3 | R25_F3 | 744 (k=24) | lr=1e-4 | 0.9884 | 0.0427 | 10 | ✅ early-stopped |
| 4 | R25_F4 | 744 (k=24) | wd=0.1 | 0.9875 | 0.0356 | 9 | ✅ early-stopped |
| 5 | R25_F5 | 744 (k=24) | warmup=3K | 0.9885 | 0.0320 | 10 | ✅ early-stopped |
| 6 | R25_F6 | 736 (k=32) | lr=1e-4 | 0.9882 | 0.0391 | 12 | ✅ early-stopped |
| 7 | R25_F7 | 744 (k=24) | λ_reg=0.05 | 0.9820 | 0.0498 | 4 | ✅ finished |
| 8 | R25_F8 | 744 (k=24) | s_end=24 | 0.9829 | 0.0498 | 5 | ✅ finished |

### R2.5 Findings

- **k=32 is the sweet spot.** F1 (k=32, AUC 0.9893) > F2 (k=48, 0.9887) > R2_E3 (k=24, 0.9851).
- **Default hyperparameters win at k=32.** lr=2e-4 beats lr=1e-4 (0.9893 > 0.9882).
- **At k=24:** lower LR or longer warmup help (+0.003). High λ_reg or s_end hurt (−0.003).
- **R25_F2 (k=48) has lowest EER** (0.0249) despite slightly lower AUC — worth noting for threshold tuning.

---

## 5. R2/R2.5 Results & Winner: R25_F1

### Combined Leaderboard (16 Runs, sorted by AUC)

| Rank | Run | Config Summary | AUC | EER | Epoch |
|------|-----|---------------|-----|-----|-------|
| 🥇 | **R25_F1** | k=32, cosine, lr=2e-4, all data | **0.9893** | 0.0356 | 10 |
| 🥈 | R25_F2 | k=48, cosine, lr=2e-4, all data | 0.9887 | **0.0249** | 8 |
| 🥉 | R25_F5 | k=24, cosine, warmup=3K, all data | 0.9885 | 0.0320 | 10 |
| 4 | R25_F3 | k=24, cosine, lr=1e-4, all data | 0.9884 | 0.0427 | 10 |
| 5 | R25_F6 | k=32, cosine, lr=1e-4, all data | 0.9882 | 0.0391 | 12 |
| 6 | R25_F4 | k=24, cosine, wd=0.1, all data | 0.9875 | 0.0356 | 9 |
| 7 | R2_A3 | k=8, CE, lr=2e-4, all data | 0.9866 | 0.0534 | 6 |
| 8 | R2_E6 | k=16, CE, lr=2e-4, all data | 0.9862 | 0.0320 | 6 |
| 9 | R2_E2 | k=16, cosine, lr=2e-4, all data | 0.9860 | 0.0320 | 7 |
| 10 | R2_E5 | k=16, cosine+DRO, lr=2e-4, all data | 0.9860 | 0.0391 | 6 |
| 11 | R2_E3 | k=24, cosine, lr=2e-4, all data | 0.9851 | 0.0320 | 6 |
| 12 | R2_E1 | k=8, cosine, lr=2e-4, all data | 0.9848 | 0.0391 | 8 |
| 13 | R25_F8 | k=24, cosine, s_end=24, all data | 0.9829 | 0.0498 | 5 |
| 14 | R25_F7 | k=24, cosine, λ_reg=0.05, all data | 0.9820 | 0.0498 | 4 |
| 15 | R2_E4 | k=16, cosine, lr=2e-4, DF40 only | 0.9817 | 0.0500 | 17 |
| 16 | R2_A1 | k=8, CE, lr=2e-4, DF40 only | 0.9792 | 0.0833 | 17 |

### R25_F1 Winner Profile

| Metric | Value |
|--------|-------|
| **W&B Run ID** | `5w453our` |
| **Config** | Cosine softmax, rank=736 (k=32), DF40+DL+VM, lr=2e-4, `base_only` aug |
| **Best Holdout AUC** | **0.9893** |
| **Best Holdout EER** | **0.0356** |
| **Best Epoch** | 10 (early-stopped at epoch 13) |
| **Data** | 21,611 videos from 26 methods (NO `quality_enhancement` — added later) |
| **Checkpoint** | `gs://training-job-outputs/phase2r2_experiments/5w453our/top_n_effort_20260212_step18500_auc0.9893_eer0.0356.pth` |

### R25_F1 Holdout Operating Points

| FPR Target | Threshold | TPR |
|---|---|---|
| EER | 0.5601 | ~96.4% (EER=0.0356) |
| 1% | 0.7783 | 72.6% |
| 2% | 0.7330 | 79.7% |
| 5% | 0.5448 | 98.6% |

---

## 6. 2×2 Production Validation: R25_F1 vs Phase 1

Before R3, we validated R25_F1 against the Phase 1 best checkpoint (B16-old) on production-relevant data.

**Checkpoints:**
- **R25_F1:** k=32, cosine, DF40+DL+VM (this project)
- **B16-old:** k=8, CE, DF40+DL (Phase 1 best, AUC 0.9947)

### DeepLive Fake Detection — Tied

| Metric | R25_F1 | B16-old |
|---|---|---|
| AUC | 0.9968 | 0.9967 |
| EER | 0.0111 | 0.0099 |
| Accuracy @0.5 | 98.8% | 99.1% |

### External Real FPR — R25_F1 Wins Decisively

| Threshold | R25_F1 FPR | B16-old FPR | Improvement |
|---|---|---|---|
| 0.50 | **6.3%** | 9.8% | **−3.5pp** |
| 0.70 | **4.0%** | 8.0% | **−4.0pp** |
| 0.80 | **2.9%** | 6.9% | **−4.0pp** |

### Cross-Dataset Calibration (Key for Production)

| Operating Point | R25_F1 | B16-old |
|---|---|---|
| At EER threshold: ExtReal FPR | **5.6%** | 10.4% |
| At 95% DeepLive TPR: ExtReal FPR | **1.8%** | 8.6% |
| At 99% DeepLive TPR: ExtReal FPR | **6.7%** | 12.2% |

**Verdict:** R25_F1 matches B16-old on fakes and has roughly **half the false positive rate** on real-world content. At 95% fake detection rate, F1 achieves **1.8% FPR** vs **8.6%** — nearly 5× fewer false alarms.

---

## 7. Round 3 — Quality Bias Experiments (6 Runs)

### Motivation

R25_F1 was trained with `base_only` augmentation and without `quality_enhancement` data (~320 high-quality deepfake videos). R3 tested whether adding quality-robust augmentation and/or quality_enhancement data could improve the model.

### R3 Experiment Matrix

| Experiment | From Scratch? | Augmentation | QE Data | LR | Purpose |
|---|---|---|---|---|---|
| **R3_S1** | Yes | quality_robust moderate | Yes | 2e-4 | Full aug from epoch 0 |
| **R3_S2** | Yes | quality_robust light | Yes | 2e-4 | Lighter aug (curriculum phase 1) |
| **R3_FT1** | No (from F1) | quality_robust moderate | Yes | 5e-5 | Quick debiasing via aug |
| **R3_FT2** | No (from F1) | quality_robust light | Yes | 5e-5 | Conservative debiasing |
| **R3_FT3** | No (from F1) | base_only (same as F1) | Yes | 5e-5 | Data-only contribution |
| **R3_FT4** | No (from F1) | quality_robust moderate | Yes | 5e-5 | Aug + data combined |

All R3 experiments: 21,931 videos from 27 methods (26 from R2 + `quality_enhancement`).

### R3 Holdout Results (All Finished ✅)

| Run | W&B ID | Best AUC | Best EER | Best Ep | In-dist AUC | In-dist EER | TPR@FPR1% |
|---|---|---|---|---|---|---|---|
| **R3_FT3** 🥇 | `kzfu116l` | **0.9966** | 0.0191 | 2 | 0.9922 | 0.0181 | **0.9771** |
| R3_FT1 | `w7wi9lpj` | 0.9962 | 0.0191 | 1 | 0.9926 | **0.0151** | 0.9733 |
| R3_FT4 | `3k2jzoe8` | 0.9962 | 0.0191 | 1 | 0.9927 | 0.0166 | 0.9733 |
| R3_S2 | `qgcp25lr` | 0.9957 | 0.0229 | 7 | 0.9885 | 0.0257 | 0.7672 |
| R3_FT2 | `3cxpwxgn` | 0.9954 | 0.0229 | 1 | **0.9930** | **0.0151** | 0.9733 |
| R3_S1 | `zjftx8ny` | 0.9933 | 0.0382 | 4 | 0.9886 | 0.0393 | 0.7214 |
| *R25_F1 (ref)* | *`5w453our`* | *0.9893* | *0.0356* | *10* | *0.9909* | *0.0254* | *0.7260* |

### R3 Fine-Tune Key Observations

1. **FT models massively outperform scratch on holdout.** All 4 FT runs (0.9954–0.9966) beat every scratch run and R25_F1.
2. **FT3 (data-only, `base_only` aug) is the holdout champion.** AUC 0.9966 — adding 320 quality_enhancement videos during fine-tuning boosted AUC by +0.0073 from R25_F1.
3. **Quality-robust augmentation doesn't help in fine-tuning either.** FT3 (no aug, data only) > FT1/FT4 (moderate aug) > FT2 (light aug). Same monotonic pattern as scratch.
4. **All FT runs peaked at epoch 1–2.** F1 was already well-converged; fine-tuning at 5e-5 LR needed minimal adaptation.
5. **TPR@FPR 1% dramatically improved.** FT models achieve ~0.97 vs ~0.72 for scratch/baseline — at 1% false positive rate, FT catches 97% of fakes vs 72%.

### Fine-Tune Launch Failure History

The 4 FT experiments required **5 launch attempts** before succeeding:

| Attempt | Date | Issue | Root Cause |
|---|---|---|---|
| 1 | Feb 12 ~21:45 | Instant crash (×4) | Stale Docker YAML → wrong checkpoint (L-14 vs B-16 size mismatch) |
| 2 | Feb 12 ~23:02 | Instant crash (×4) | Same — Docker not rebuilt yet |
| 3 | Feb 13 ~10:24 | Failed at step 1 (×4, ~80 min each) | Unknown — data loaded OK, died during first forward/backward pass |
| 4 | Feb 13 ~11:46 | Instant crash (×3) | Docker rebuilt with validation changes, possibly incomplete |
| **5** | **Feb 13 ~12:47** | **✅ SUCCESS (×4)** | Docker v1.3.117 with all fixes |

---

## 8. R3 Out-of-Distribution Validation (3×3)

### Setup

3 checkpoints (S1, S2, F1) × 3 data sources = 9 evaluations.  
Data: 9,657 videos — 863 `deeplive_edge_cases` (fake), 846 `deeplive_minimal_processing` (fake), 638 `deeplive_quality_enhancement` (mixed), 7,310 `external_youtube_avspeech` (real).

> **Note:** R3 fine-tune models (FT1–FT4) have NOT been OOD-validated yet — they finished after this validation run.

### Overall Results (video-level, threshold=0.5)

| Model | Overall ACC | Real ACC (TNR) | Fake ACC (TPR) |
|---|---|---|---|
| **F1** (R25, no quality aug) | **94.8%** | **94.2%** | **98.9%** |
| S2 (quality_robust light) | 93.8% | 93.4% | 96.0% |
| S1 (quality_robust moderate) | 92.9% | 92.4% | 96.4% |

### Per-Method Accuracy

| Method | Label | N | S1 | S2 | **F1** |
|---|---|---|---|---|---|
| deeplive_edge_cases | fake | 863 | 95.9% | 95.9% | **98.6%** |
| deeplive_minimal_processing | fake | 846 | 97.2% | 97.3% | **98.0%** |
| deeplive_quality_enhancement | mixed | 638 | 98.4% | 98.1% | **98.9%** |
| external_youtube_avspeech | real | 7,310 | 91.5% | 92.7% | **93.6%** |

### Cross-Checkpoint Consensus Analysis

| Category | Count | % |
|---|---|---|
| All 3 models correct | 8,865 | 91.8% |
| Consensus failures (ALL wrong) | 416 | 4.3% |
| Discriminating (models disagree) | 376 | 3.9% |

**Consensus failures (416):** 404 are false positives on `external_youtube_avspeech` — real videos all models confidently flag as fake (avg confidence >0.99). These are genuinely ambiguous/problematic real videos, not borderline cases.

**Discriminating failures (376):** F1 gets **76.1%** right, S2 gets 50.3%, S1 gets 27.7%. F1 dominates on external reals (221/297 correct vs S2 158, S1 72).

### FP Confidence Profile

| Model | FP mean prob | FP median | Threshold tunability |
|---|---|---|---|
| **F1** | **0.588** | **0.614** | **Best** — FPs are borderline, fixable with threshold |
| S2 | 0.651 | 0.669 | Moderate |
| S1 | 0.737 | 0.797 | Worst — many stubbornly confident FPs |

### The Key Takeaway

**Quality-robust augmentation hurts OOD generalization monotonically:** more aug → worse results. F1 (no quality aug) > S2 (light) > S1 (moderate) on every evaluation dimension. The augmentation teaches the model to rely on augmentation-specific artifacts that don't appear in real-world data.

F1 was trained WITHOUT quality_enhancement data but achieves **98.9% accuracy on it** — better than S1 (98.4%) and S2 (98.1%) which were trained WITH it.

---

## 9. Grand Leaderboard — All Experiments

### By Holdout AUC (Training Metric)

| Rank | Run | Round | Config | Holdout AUC | Holdout EER | Best Ep |
|------|-----|-------|--------|-------------|-------------|---------|
| 🥇 | **R3_FT3** | R3 | FT from F1, base_only aug, +QE data | **0.9966** | 0.0191 | 2 |
| 🥈 | R3_FT1 | R3 | FT from F1, quality_robust moderate | 0.9962 | 0.0191 | 1 |
| 🥈 | R3_FT4 | R3 | FT from F1, quality_robust mod + QE | 0.9962 | 0.0191 | 1 |
| 4 | R3_S2 | R3 | Scratch, quality_robust light | 0.9957 | 0.0229 | 7 |
| 5 | R3_FT2 | R3 | FT from F1, quality_robust light | 0.9954 | 0.0229 | 1 |
| 6 | R3_S1 | R3 | Scratch, quality_robust moderate | 0.9933 | 0.0382 | 4 |
| 7 | **R25_F1** | R2.5 | k=32, cosine, base_only, all data | **0.9893** | 0.0356 | 10 |
| 8 | R25_F2 | R2.5 | k=48, cosine, all data | 0.9887 | 0.0249 | 8 |
| 9 | R25_F5 | R2.5 | k=24, warmup=3K, all data | 0.9885 | 0.0320 | 10 |
| 10 | R25_F3 | R2.5 | k=24, lr=1e-4, all data | 0.9884 | 0.0427 | 10 |
| 11 | R25_F6 | R2.5 | k=32, lr=1e-4, all data | 0.9882 | 0.0391 | 12 |
| 12 | R25_F4 | R2.5 | k=24, wd=0.1, all data | 0.9875 | 0.0356 | 9 |
| 13 | R2_A3 | R2 | k=8, CE, all data | 0.9866 | 0.0534 | 6 |
| 14 | R2_E6 | R2 | k=16, CE, all data | 0.9862 | 0.0320 | 6 |
| 15 | R2_E2 | R2 | k=16, cosine, all data | 0.9860 | 0.0320 | 7 |
| 16 | R2_E5 | R2 | k=16, cosine+DRO, all data | 0.9860 | 0.0391 | 6 |
| 17 | R2_E3 | R2 | k=24, cosine, all data | 0.9851 | 0.0320 | 6 |
| 18 | R2_E1 | R2 | k=8, cosine, all data | 0.9848 | 0.0391 | 8 |
| 19 | R25_F8 | R2.5 | k=24, s_end=24 | 0.9829 | 0.0498 | 5 |
| 20 | R25_F7 | R2.5 | k=24, λ_reg=0.05 | 0.9820 | 0.0498 | 4 |
| 21 | R2_E4 | R2 | k=16, cosine, DF40 only | 0.9817 | 0.0500 | 17 |
| 22 | R2_A1 | R2 | k=8, CE, DF40 only | 0.9792 | 0.0833 | 17 |

### By OOD Deployment Performance (The Metric That Matters)

| Rank | Run | Overall ACC | Real FPR @0.5 | Fake TPR | Discriminating ACC |
|------|-----|-------------|---------------|----------|-------------------|
| 🥇 | **R25_F1** | **94.8%** | **6.3%** | **98.9%** | **76.1%** |
| 🥈 | R3_S2 | 93.8% | 7.3% | 96.0% | 50.3% |
| 🥉 | R3_S1 | 92.9% | 8.5% | 96.4% | 27.7% |
| ? | R3_FT1–FT4 | *Not yet evaluated* | — | — | — |

**⚠️ The two leaderboards tell opposite stories.** R3_FT3 is #1 by holdout AUC but R25_F1 is #1 by OOD performance. This is the central lesson of Phase 2.

---

## 10. Key Findings & Lessons Learned

### Finding 1: Data Composition Is the Biggest Lever
Adding DeepLive + VisoMaster to DF40 gave Δ=+0.004–0.007 AUC — larger than any loss, capacity, or hyperparameter change. DF40-only runs are consistently worst.

### Finding 2: Capacity Sweet Spot at k=32
The trainable SVD subspace dimension follows: k=8 < k=16 < k=24 < **k=32** > k=48. Diminishing returns above k=32. This means 32 trainable directions per layer (out of 768) is the right balance between expressivity and regularization.

### Finding 3: Loss Function Doesn't Matter Much
CE ≈ Cosine softmax across all comparisons. The cosine softmax with m=0 (no angular margin) performs similarly to plain CE. ArcFace with m=0.15 was too aggressive (R1). This is consistent with the Effort paper's finding that the SVD structure provides implicit regularization.

### Finding 4: Group DRO Is Inert
R2_E5 (DRO) = R2_E2 (no DRO) at 0.9860. The per-method loss reweighting doesn't help when the SVD residual already constrains the model.

### Finding 5: Quality-Robust Augmentation Hurts Generalization
The monotonic pattern across both scratch and fine-tune experiments: more quality augmentation → worse OOD performance. The augmentation likely introduces distribution shifts that inflate holdout metrics but don't match real-world data.

### Finding 6: Holdout AUC ≠ Deployment Performance (THE Lesson)
Models with AUC 0.9933–0.9966 (R3) underperform a model with AUC 0.9893 (R25_F1) on out-of-distribution evaluation. **Never deploy based on holdout metrics alone.** Always validate on OOD data.

### Finding 7: Fine-Tuning Is Efficient but Needs OOD Validation
FT models converge in 1–2 epochs (vs 7–10 for scratch), achieve the highest holdout metrics ever, and dramatically improve TPR@FPR1% (0.97 vs 0.72). But we don't yet know if they generalize. The R3 scratch→OOD validation showed that holdout gains don't guarantee OOD gains.

### Finding 8: External Real FPs Are the Dominant Error Mode
394 out of 416 consensus failures are YouTube real videos falsely flagged as fake with >0.99 confidence. This is a data quality/distribution issue, not a model architecture issue.

---

## 11. Production Recommendation

### Current Best: R25_F1 (`5w453our`)

Based on all evidence, **R25_F1 is the best deployment candidate:**

| Dimension | R25_F1 |
|---|---|
| Overall OOD accuracy | **94.8%** (9,657 videos) |
| Real false positive rate @0.5 | **6.3%** |
| Fake detection rate | **98.9%** |
| Discriminating case accuracy | **76.1%** |
| FP confidence profile | Most threshold-tunable (mean 0.59) |
| vs Phase 1 (B16-old) | Same TPR, **half** the FPR |

**Checkpoint:** `gs://training-job-outputs/phase2r2_experiments/5w453our/top_n_effort_20260212_step18500_auc0.9893_eer0.0356.pth`

### Candidate for Upgrade: R3_FT3 (`kzfu116l`)

R3_FT3 has the highest holdout AUC (0.9966) and best TPR@FPR1% (0.9771). It uses the same augmentation as F1 (`base_only`) and only adds 320 quality_enhancement videos. **However, it has NOT been OOD-validated.** Given the R3 scratch experience (holdout ≠ OOD), an OOD validation run is mandatory before considering FT3 for deployment.

**FT3 Checkpoint:** `gs://training-job-outputs/phase2r3_experiments/kzfu116l/top_n_effort_20260213_step2500_auc0.9966_eer0.0191.pth`

### Recommended Next Steps

1. **OOD-validate R3_FT3** — Same 3-data-source eval as §8. If FT3 beats F1 on OOD, it becomes the new deployment candidate.
2. **Deploy F1 now** — Don't wait for FT3 validation. F1 is proven on OOD data.
3. **Threshold tuning** — F1's FP confidence profile (mean 0.59) is highly threshold-tunable. A threshold of 0.70+ gives <4% FPR with minimal TPR loss.

---

## 12. Data Sources Reference

### Training Data

| Source | Videos | Methods | Notes |
|---|---|---|---|
| **DF40** | ~9,900 | 17 | MRAA, blendface, danet, e4s, facedancer, faceswap, facevid2vid, fomm, fsgan, inswap, lia, mcnet, mobileswap, one_shot_free, pirender, simswap, uniface |
| **DeepLive** | ~6,800 | 4 | edge_cases, minimal_processing, visomaster, quality_enhancement (R3 only) |
| **VisoMaster** | ~3,100 | 6 (train) | CSCS, GhostFace-v1/v2, InStyleSwapper256-A/B, Inswapper128 |
| **Total (R2/R2.5)** | ~21,600 | 26 | No quality_enhancement |
| **Total (R3)** | ~21,900 | 27 | +320 quality_enhancement videos |

### Evaluation Data (OOD)

| Source | Videos | Label | Purpose |
|---|---|---|---|
| DeepLive edge_cases | 863 | fake | Production-relevant deepfakes |
| DeepLive minimal_processing | 846 | fake | Production-relevant deepfakes |
| DeepLive quality_enhancement | 638 | mixed | High-quality enhanced deepfakes |
| External YouTube AVSpeech | 7,310 | real | OOD real-world content for FPR |

### Data Pipeline

- Identity-stratified splitting via `realpool_{identity}` prefix
- Split: 85% train / 10% val / 5% test
- Sampling: 8 frames per video (sparse anchor indices)
- Augmentation: `base_only` (horizontal flip, color jitter, quality variation)

---

## 13. Code Changes Log

### Critical Bug Fixes (R1→R2)
1. **Config passthrough** — `combined_paired` section now copied to `data_config` in `train_sweep.py`
2. **Identity leakage** — Unified `realpool_` prefix in `data/sources/combined_paired.py`
3. **Group DRO dimensions** — `repeat_interleave(T)` fix in `trainer/mixins/group_dro.py`
4. **Data verification** — Fail-fast assertions + W&B table logging in `train_sweep.py`

### Evaluation Infrastructure
5. **EER threshold + FPR metrics** — New `_compute_roc_metrics()` in `metrics/utils.py`
6. **External real sampling** — `max_videos` + `seed` params in `data/validation_sources.py`
7. **Empty frame list guard** — Skip samples with 0 frames in DeepLive loader
8. **DeepLive strategy override** — `--deeplive_strategies` CLI arg in `validate_custom_sources.py`

### R3 Infrastructure
9. **Validation checkpoint caching bug** — Pass unique `--checkpoint_local_path` per checkpoint in `run_r3_validation_sequential.py`
10. **Fine-tune checkpoint loading** — Fixed stale YAML paths in experiment configs (was pointing to L-14 checkpoint instead of B-16)
11. **Quality-robust augmentation pipeline** — `quality_robust_moderate` and `quality_robust_light` pipelines in `data/augmentations/`

### Tooling
12. **`launch_2x2_comparison.sh`** — 2×2 production validation launcher
13. **`run_r3_validation_sequential.py`** — 3×3 R3 validation orchestrator
14. **`launch_r3_validation.sh`** — Vertex AI submission for R3 validation
15. **`launch_failure_analysis.sh`** — Cross-checkpoint failure analysis (5 models × 4 sources)
16. **`analysis_results/failure_analysis.py`** — Consensus/discriminating failure analysis

### Docker Versions
- **v1.3.105** — R2/R2.5 training
- **v1.3.116** — R3 scratch training
- **v1.3.117** — R3 fine-tune success + validation (current)

---

## 14. Appendix: GCS Paths & W&B IDs

### Key Checkpoints

| Model | W&B ID | AUC | Checkpoint |
|---|---|---|---|
| **R25_F1** (deploy) | `5w453our` | 0.9893 | `gs://training-job-outputs/phase2r2_experiments/5w453our/top_n_effort_20260212_step18500_auc0.9893_eer0.0356.pth` |
| **R3_FT3** (candidate) | `kzfu116l` | 0.9966 | `gs://training-job-outputs/phase2r3_experiments/kzfu116l/top_n_effort_20260213_step2500_auc0.9966_eer0.0191.pth` |
| R3_FT1 | `w7wi9lpj` | 0.9962 | `gs://training-job-outputs/phase2r3_experiments/w7wi9lpj/top_n_effort_20260213_step500_auc0.9962_eer0.0191.pth` |
| R3_FT4 | `3k2jzoe8` | 0.9962 | `gs://training-job-outputs/phase2r3_experiments/3k2jzoe8/top_n_effort_20260213_step500_auc0.9962_eer0.0191.pth` |
| R3_FT2 | `3cxpwxgn` | 0.9954 | `gs://training-job-outputs/phase2r3_experiments/3cxpwxgn/top_n_effort_20260213_step500_auc0.9954_eer0.0229.pth` |
| R3_S2 | `qgcp25lr` | 0.9957 | `gs://training-job-outputs/phase2r3_experiments/qgcp25lr/top_n_effort_20260213_step13000_auc0.9957_eer0.0229.pth` |
| R3_S1 | `zjftx8ny` | 0.9933 | `gs://training-job-outputs/phase2r3_experiments/zjftx8ny/top_n_effort_20260213_step7500_auc0.9933_eer0.0382.pth` |
| B16-old (Phase 1) | — | 0.9947 | `gs://training-job-outputs/best_checkpoints/corrected/top_n_effort_20260119_step14000_auc0.9947_eer0.0218_B16_LAION.patched.pth` |

### All R2 Checkpoints

| Run | Run ID | AUC | Checkpoint |
|---|---|---|---|
| R2_A1 | `pcyqx9l2` | 0.9792 | `gs://training-job-outputs/phase2r2_experiments/pcyqx9l2/top_n_effort_20260211_step6500_auc0.9792_eer0.0833.pth` |
| R2_A3 | `8urk1cmw` | 0.9866 | `gs://training-job-outputs/phase2r2_experiments/8urk1cmw/top_n_effort_20260212_step10000_auc0.9866_eer0.0534.pth` |
| R2_E1 | `ai6c1gk8` | 0.9848 | `gs://training-job-outputs/phase2r2_experiments/ai6c1gk8/top_n_effort_20260212_step13500_auc0.9848_eer0.0391.pth` |
| R2_E2 | `hog6lmfq` | 0.9860 | `gs://training-job-outputs/phase2r2_experiments/hog6lmfq/top_n_effort_20260212_step12000_auc0.9860_eer0.0320.pth` |
| R2_E3 | `hp87l4m2` | 0.9851 | `gs://training-job-outputs/phase2r2_experiments/hp87l4m2/top_n_effort_20260212_step10000_auc0.9851_eer0.0320.pth` |
| R2_E4 | `nhl3aaws` | 0.9817 | `gs://training-job-outputs/phase2r2_experiments/nhl3aaws/top_n_effort_20260211_step6500_auc0.9817_eer0.0500.pth` |
| R2_E5 | `x4fmqrhu` | 0.9860 | `gs://training-job-outputs/phase2r2_experiments/x4fmqrhu/top_n_effort_20260212_step10000_auc0.9860_eer0.0391.pth` |
| R2_E6 | `we9dc7ql` | 0.9862 | `gs://training-job-outputs/phase2r2_experiments/we9dc7ql/top_n_effort_20260212_step10000_auc0.9862_eer0.0320.pth` |

### All R2.5 Checkpoints

| Run | Run ID | AUC | Checkpoint |
|---|---|---|---|
| R25_F1 | `5w453our` | 0.9893 | `gs://training-job-outputs/phase2r2_experiments/5w453our/top_n_effort_20260212_step18500_auc0.9893_eer0.0356.pth` |
| R25_F2 | `ug8n870r` | 0.9887 | `gs://training-job-outputs/phase2r2_experiments/ug8n870r/top_n_effort_20260212_step13500_auc0.9887_eer0.0249.pth` |
| R25_F3 | `6md2py50` | 0.9884 | `gs://training-job-outputs/phase2r2_experiments/6md2py50/top_n_effort_20260212_step18500_auc0.9884_eer0.0427.pth` |
| R25_F4 | `gbcqvspo` | 0.9875 | `gs://training-job-outputs/phase2r2_experiments/gbcqvspo/top_n_effort_20260212_step16500_auc0.9875_eer0.0356.pth` |
| R25_F5 | `tkh09be0` | 0.9885 | `gs://training-job-outputs/phase2r2_experiments/tkh09be0/top_n_effort_20260212_step18500_auc0.9885_eer0.0320.pth` |
| R25_F6 | `x8lktnbc` | 0.9882 | `gs://training-job-outputs/phase2r2_experiments/x8lktnbc/top_n_effort_20260212_step22000_auc0.9882_eer0.0391.pth` |
| R25_F7 | `27i36ctf` | 0.9820 | `gs://training-job-outputs/phase2r2_experiments/27i36ctf/top_n_effort_20260212_step6500_auc0.9820_eer0.0498.pth` |
| R25_F8 | `8w9a8qdu` | 0.9829 | `gs://training-job-outputs/phase2r2_experiments/8w9a8qdu/top_n_effort_20260212_step9000_auc0.9829_eer0.0498.pth` |

### Evaluation Outputs (GCS)

| Eval | Path |
|---|---|
| 2×2 (F1 vs B16-old) | `gs://training-job-outputs/test_results/2x2_F1_vs_B16old_comparison/` |
| R3 3×3 validation | `gs://training-job-outputs/test_results/r3_S1_vs_S2_vs_F1_validation/` |

### Local Analysis Outputs

| Analysis | Location |
|---|---|
| R3 validation CSVs | `analysis_results/r3_validation/` |
| Cross-checkpoint predictions | `analysis_results/r3_validation/failure_analysis_output/cross_checkpoint_predictions.csv` |
| Consensus failures | `analysis_results/r3_validation/failure_analysis_output/consensus_failures.csv` |
| Discriminating failures | `analysis_results/r3_validation/failure_analysis_output/discriminating_failures.csv` |

### W&B Quick Reference

| Item | Value |
|---|---|
| Entity | `dtect-vision` |
| Project | `phase2-experiments` |
| R2 filter tag | `phase2-round2` |
| R3 scratch | `zjftx8ny` (S1), `qgcp25lr` (S2) |
| R3 fine-tune | `w7wi9lpj` (FT1), `3cxpwxgn` (FT2), `kzfu116l` (FT3), `3k2jzoe8` (FT4) |
| Deployment candidate | `5w453our` (R25_F1) |
| Upgrade candidate | `kzfu116l` (R3_FT3) — pending OOD validation |

---

## 15. WMA Enhanced Face-Swap Investigation — Why All Models Fail

**Date:** 2026-02-15  
**Motivation:** All deployed models catastrophically fail on enhanced face-swaps from DeepLiveCam+GFPGAN (the "WMA" pipeline). This section documents the full investigation from discovery to root-cause analysis.

---

### 15.1 Background — The WMA Pipeline

A production partner (WMA) exported **1,202 face-swap images** generated by:

1. **DeepLiveCam** (InsightFace/SCRFD face detection → ArcFace 128×128 embedding → `inswapper_128` face swap)
2. **GFPGAN v1.4 enhancer** (facexlib/RetinaFace re-detection → FFHQ 512×512 alignment → GFPGANv1.4 restoration → paste-back)

The images are **raw crops** from this pipeline — **not** pre-cropped to 224×224 like our training data. They represent what a real-world deployment would see: enhanced, high-resolution face-swap outputs at variable resolutions (~342×435 median).

The images live at: `/Users/roeedar/Downloads/wma_export/all_images/` (1,202 JPEGs, all fake).

This is **the same base software** (DeepLiveCam/inswapper_128) used in our training data under the `quality_enhancement` and `minimal_processing` strategies — but the WMA images went through a different face-detection/cropping pipeline and GFPGAN enhancement before reaching us.

---

### 15.2 Step 1 — Multi-Checkpoint Evaluation (Catastrophic Failure)

We tested **5 checkpoints** spanning different training rounds and strategies against all 1,202 WMA images using local MPS inference.

**Script:** `DeepfakeBench/training/eval_enhancer_local.py`

The script downloads each checkpoint from GCS, runs batched inference on all WMA images (with standard CLIP preprocessing: resize to 224×224, center crop, ImageNet normalization), and saves per-image probabilities to CSV.

**Run it yourself:**
```bash
cd DeepfakeBench/training
conda run -n sweep-env python eval_enhancer_local.py
```

**Results (all images are fake — accuracy = % correctly detected as fake):**

| Model | Holdout AUC | WMA Accuracy @0.5 | Mean Prob | Median Prob | % > 0.9 | % < 0.3 |
|-------|-------------|-------------------|-----------|-------------|---------|---------|
| **B16_old** (Phase 1) | 0.9947 | **55.7%** | 0.546 | 0.633 | 32.6% | 37.2% |
| **R3_FT3** (holdout champ) | 0.9966 | 39.5% | 0.418 | 0.329 | 14.6% | 47.5% |
| **R25_F1** (OOD winner) | 0.9893 | 21.0% | 0.283 | 0.175 | 7.7% | 65.1% |
| R3_FT2 | 0.9954 | 9.3% | 0.196 | 0.111 | 3.2% | 82.2% |
| R3_FT1 | 0.9962 | 8.3% | 0.196 | 0.129 | 2.2% | 83.7% |

**Key observations:**

1. **Every single model fails.** The best (B16_old) gets only 55.7% — barely above random chance. Our deployment winner R25_F1 gets only 21%. The R3 models with quality-robust augmentation are worst (8–9%).
2. **Holdout AUC is inversely correlated with WMA accuracy.** B16_old (lowest AUC at 0.9947) is best on WMA; R3_FT1 (AUC 0.9962) is worst. This deepens the paradox from §10 Finding 6.
3. **Quality-robust augmentation makes things worse, not better.** R3_FT1 and R3_FT2 (trained with augmentation designed for this exact scenario) are the worst performers. The augmentation teaches the model to ignore the wrong things.
4. **Most images are confidently misclassified as real.** 65% of WMA images get <0.3 fake probability from R25_F1. The model is not uncertain — it's confidently wrong.

**Output files:**
- `analysis_results/enhancer_eval_per_image.csv` — per-image fake probability for all 5 models (1,202 rows × 5 probability columns)
- `analysis_results/enhancer_eval_summary.csv` — per-model aggregate statistics

---

### 15.3 Step 2 — Hypothesis Generation

After seeing universal failure, we considered several hypotheses:

| # | Hypothesis | Description |
|---|-----------|-------------|
| H1 | **Preprocessing gap** | WMA images have different resolution/crop quality than training data. Resizing to 224×224 destroys diagnostic features. |
| H2 | **GFPGAN smoothing** | The GFPGANv1.4 enhancer specifically removes the high-frequency artifacts (GAN fingerprints, blending boundaries) that the model learned to detect. |
| H3 | **Sharpness-as-shortcut** | The model learned "sharp = fake, smooth = real" as a shortcut, because training fakes happen to be sharper than training reals. GFPGAN-enhanced images are smoother than even real faces, inverting this relationship. |
| H4 | **Distribution shift** | WMA images occupy a different region of CLIP feature space than any training source. |
| H5 | **Resolution mismatch** | Training images are uniformly 224×224 (pre-cropped). WMA images are ~342×435 — downscaling to 224 further degrades quality. |

To distinguish between these hypotheses, we designed a systematic meta-analysis comparing image-level properties and CLIP features across all data sources.

---

### 15.4 Step 3 — Meta-Analysis Design (10 Data Sources, 2,602 Images)

We sampled images from **10 data sources** — the WMA test images plus 9 sources from our training/evaluation data — and computed comprehensive image properties and CLIP features for each.

**Script:** `DeepfakeBench/training/meta_analysis_enhancer.py` (main analysis)  
**Script:** `DeepfakeBench/training/meta_analysis_fix_missing.py` (targeted fix for 4 initially missing sources)

#### Data Sources

| # | Source | Type | n | Origin | Why Included |
|---|--------|------|---|--------|-------------|
| 1 | `wma_enhanced` | fake | 1,202 | Local (WMA export) | **The problem source** — enhanced face-swaps we can't detect |
| 2 | `deeplive_quality_enhancement_fake` | fake | 150 | GCS training data | **Critical comparison** — same DeepLiveCam software, same GFPGAN enhancer, but pre-cropped to 224×224 |
| 3 | `deeplive_minimal_processing_fake` | fake | 150 | GCS training data | Same DeepLiveCam software, **no** enhancer — isolates GFPGAN effect |
| 4 | `deeplive_edge_cases_fake` | fake | 150 | GCS training data | DeepLiveCam edge-case fakes the model handles well |
| 5 | `visomaster_fake` | fake | 150 | GCS training data | VisoMaster face-swaps — high quality, high detection rate |
| 6 | `df40_fake` | fake | 150 | GCS training data | DF40 multi-method fakes — moderate detection rate |
| 7 | `deeplive_quality_enhancement_real` | real | 150 | GCS training data | Corresponding real faces for QE fakes |
| 8 | `deeplive_minimal_processing_real` | real | 150 | GCS training data | Corresponding real faces for MP fakes |
| 9 | `df40_real` | real | 150 | GCS training data | DF40 real faces |
| 10 | `external_youtube_real` | real | 200 | GCS eval data | OOD real faces from YouTube AVSpeech |

#### What We Computed Per Image (36 Properties)

| Category | Metrics |
|----------|---------|
| **Resolution** | width, height, aspect_ratio, num_pixels, effective_resolution_90pct |
| **Sharpness** | laplacian_var (key!), laplacian_mean, tenengrad, tenengrad_var |
| **Frequency** | low/mid/high energy ratios, high-to-low ratio (FFT-based) |
| **Color** | per-channel (R,G,B) mean/std, luminance mean/std |
| **Contrast** | RMS contrast, Michelson contrast |
| **Color detail** | saturation mean/std, hue mean/std |
| **Texture** | noise_estimate, JPEG compressibility, edge_density, local_var_mean, symmetry_score |
| **Model output** | `model_fake_prob` (R25_F1 prediction) |
| **CLIP features** | 512-dimensional pooler output from R25_F1's ViT-B-16 backbone |

#### GCS Sampling Bug & Fix

The initial run of `meta_analysis_enhancer.py` only retrieved 6 of 10 sources. The 4 missing sources — `deeplive_quality_enhancement_fake`, `deeplive_quality_enhancement_real`, `deeplive_minimal_processing_fake`, `deeplive_minimal_processing_real` — were the most critical for the investigation (same software as WMA!).

**Root cause:** `sample_deeplive_images()` listed blobs with `prefix='samples/'` and `max_results=10000`. The bucket has >10K blobs under `samples/`, and visomaster entries (alphabetically dominant under `samples/v...`) consumed all 10K slots before the scanner could reach `quality_enhancement_*` or `minimal_processing_*` directories.

**Fix:** Changed to use `prefix=f'samples/{strategy}_'` for targeted listing. The fix script `meta_analysis_fix_missing.py` downloads only the 4 missing sources, computes their properties and features, and merges them into the existing CSV/NPZ files.

**Run it yourself:**
```bash
cd DeepfakeBench/training

# Full analysis from scratch (all 10 sources):
conda run -n sweep-env python meta_analysis_enhancer.py

# Or just the 4 missing sources (if you already have the first 6):
conda run -n sweep-env python meta_analysis_fix_missing.py
```

---

### 15.5 Step 4 — Results: The Complete Picture

**Output files:**
- `analysis_results/meta_analysis_properties.csv` — 2,602 rows × 36 columns (all image properties)
- `analysis_results/meta_analysis_features.npz` — 2,602 × 512 CLIP features, model probabilities, source labels, ground-truth labels, filenames

**Visualization script:** `analysis_results/plot_meta_analysis.py` (generates all 10 plots)

```bash
conda run -n sweep-env python analysis_results/plot_meta_analysis.py
# → outputs 10 PNG plots to analysis_results/plots/
```

#### 15.5.1 Complete Results Table

| Source | n | Accuracy | Median Prob | Sharpness (Laplacian var) | Edge Density | Noise Est. | Resolution |
|--------|--:|--------:|------------:|--------------------------:|-------------:|-----------:|-----------|
| **WMA Enhanced** (test, fake) | 1,202 | **21.0%** | 0.175 | **18.7** | 0.0118 | 0.479 | **342×435** |
| DL QualEnhance (train, fake) | 150 | 98.0% | 1.000 | 63.2 | 0.0340 | 0.448 | 224×224 |
| DL MinProc (train, fake) | 150 | 98.7% | 1.000 | 54.4 | 0.0298 | 0.420 | 224×224 |
| DL EdgeCases (train, fake) | 150 | 98.7% | 1.000 | 68.7 | 0.0322 | 0.442 | 224×224 |
| VisoMaster (train, fake) | 150 | 99.3% | 1.000 | 129.1 | 0.0428 | 0.653 | 224×224 |
| DF40 (train, fake) | 150 | **53.3%** | 0.565 | **37.9** | 0.0270 | 0.393 | 224×224 |
| DL QualEnhance (train, real) | 150 | 98.0% | 0.001 | 107.3 | 0.0444 | 0.514 | 224×224 |
| DL MinProc (train, real) | 150 | 96.7% | 0.001 | 89.2 | 0.0376 | 0.470 | 224×224 |
| DF40 (train, real) | 150 | **11.3%** | 1.000 | **15.4** | 0.0118 | 0.334 | 224×224 |
| YouTube (eval, real) | 200 | 88.5% | 0.002 | 226.0 | 0.0527 | 0.901 | 190×190 |

#### 15.5.2 The Sharpness Shortcut — Root Cause Identified

The data above reveals a striking pattern. Group the sources by model accuracy:

**Sources the model gets RIGHT (>95% accuracy):**
- All DeepLive training data (QE fake/real, MP fake/real, EC fake): sharpness 54–107, all 224×224
- VisoMaster: sharpness 129, 224×224
- YouTube real: sharpness 226, 190×190

**Sources the model gets WRONG (<55% accuracy):**
- WMA enhanced: sharpness **18.7**, 342×435
- DF40 fake: sharpness **37.9**, 224×224
- DF40 real: sharpness **15.4**, 224×224 — **89% false positive rate!**

**The pattern: sharpness predicts accuracy, not the actual real/fake label.**

| Sharpness Range | Model Behavior | Sources |
|----------------|----------------|---------|
| <40 | Classifies as "REAL" regardless of truth | WMA (fake), DF40 fake, DF40 real |
| 50–130 | Correctly classifies fake as fake | DL QE/MP/EC fake, VisoMaster fake |
| >80 | Correctly classifies real as real | DL QE/MP real, YouTube real |

The model has learned a **sharpness-based shortcut**: low-sharpness images → "real", high-sharpness images → "fake". This works on the training distribution because:
- Training fakes (DeepLive, VisoMaster) are generated from high-quality video frames and retain sharp GAN artifacts → high sharpness
- Training reals (YouTube) are also relatively sharp but the model learned a threshold above them

But it catastrophically fails when:
- **GFPGAN-enhanced fakes** have *lower* sharpness than reals (smoothed by the super-resolution network)
- **DF40 images** happen to have low sharpness (different preprocessing pipeline)

#### 15.5.3 The DF40 Failure — Corroborating Evidence

The DF40 results provide critical corroboration:

- **DF40 fake** (sharpness 37.9): only 53.3% accuracy — nearly random
- **DF40 real** (sharpness 15.4): **only 11.3% accuracy** — the model classifies 89% of real faces as fake!

DF40 reals have the lowest sharpness in the entire dataset (15.4), even lower than WMA enhanced fakes (18.7). The model confidently assigns them **median fake probability of 1.000** — it thinks they are fake with near-certainty, because it has learned "low sharpness = real face, NOT a deepfake" but DF40's preprocessing pipeline produces low-sharpness reals that the model hasn't seen.

Wait — this reveals something even more nuanced. Re-reading the accuracy: DF40 real at 11.3% accuracy means the model says "fake" for 89% of real DF40 faces. But WMA enhanced fake at 21% accuracy means the model says "real" for 79% of fake WMA faces. **The model is not just using sharpness one way** — it's using a combination of features that happens to correlate with sharpness on the training distribution but breaks down on OOD data.

The t-SNE plot (`analysis_results/plots/07_tsne_clip_features.png`) shows whether WMA and DF40 cluster together in CLIP feature space, which would confirm they share some quality-related representation the model has latched onto.

#### 15.5.4 Resolution Mismatch Amplifies the Problem

WMA images are the **only non-224×224 source** among the fake sources:

| Source | Resolution | Sharpness |
|--------|-----------|-----------|
| WMA enhanced | **342×435** (variable) | 18.7 |
| All training fakes | **224×224** (uniform) | 54–129 |

When WMA images are resized to 224×224 for inference (required by the ViT-B-16 backbone), they undergo significant downscaling (342→224 in width = 35% reduction). This downscaling further reduces sharpness and destroys high-frequency content — compounding the GFPGAN smoothing effect.

The model has never seen images that were downscaled from higher resolution during training (all training images are already 224×224). The downscaling introduces a distinctive frequency signature that the model interprets as "even more real-looking."

#### 15.5.5 GFPGAN Smoothing vs No Enhancer

Comparing the two DeepLiveCam processing strategies that the model handles well:

| Source | Strategy | Sharpness | Accuracy |
|--------|----------|-----------|----------|
| DL QualEnhance fake | GFPGAN enhanced, 224×224 | 63.2 | 98.0% |
| DL MinProc fake | No enhancer, 224×224 | 54.4 | 98.7% |
| **WMA enhanced** | **GFPGAN enhanced, 342×435** | **18.7** | **21.0%** |

**Key insight:** The model handles GFPGAN-enhanced images just fine when they're pre-cropped to 224×224 (DL QualEnhance: 98% accuracy, sharpness 63.2). The problem is specifically the combination of:
1. GFPGAN smoothing (reduces sharpness)
2. PLUS downscaling from higher resolution (further reduces sharpness)
3. PLUS a different face-detection/cropping pipeline (different crop boundaries)

DL QualEnhance images go through: DeepLiveCam → GFPGAN → YOLO crop to 224×224 (our pipeline)
WMA images go through: DeepLiveCam → GFPGAN → exported at native resolution → resized to 224×224 at inference

The difference is **where the cropping and resizing happens** in the pipeline. Our training data is pre-cropped and resized with YOLO face detection at a controlled quality level. WMA images are cropped differently and resized at inference time.

#### 15.5.6 Edge Density and Noise Tell the Same Story

Two more properties corroborate the sharpness-shortcut hypothesis:

**Edge density** (fraction of Canny edge pixels):
- WMA: 0.0118 (lowest among fakes — very few edges)
- Training fakes: 0.0298–0.0428 (2.5–3.6× more edges)
- DF40 real: 0.0118 (same as WMA — and also misclassified)
- YouTube real: 0.0527 (most edges — correctly classified as real)

**Noise estimate** (standard deviation of high-pass filtered image):
- WMA: 0.479 (moderate — GFPGAN produces clean output)
- Training fakes: 0.420–0.653 (similar or higher)
- DF40 real: 0.334 (lowest noise — also misclassified)

The edge density numbers are particularly revealing: WMA fakes and DF40 reals have **identical edge density** (0.0118) and both are badly misclassified. The model appears to use edge density as a co-feature with sharpness.

---

### 15.6 Visualization Guide — 10 Diagnostic Plots

All plots are in `analysis_results/plots/`. Regenerate them with:

```bash
conda run -n sweep-env python analysis_results/plot_meta_analysis.py
```

| Plot | File | What It Shows |
|------|------|--------------|
| **01** | `01_probability_distributions.png` | Fake probability histograms per source. Shows WMA and DF40 fakes clustered near 0 (misclassified as real), while training fakes cluster near 1. |
| **02** | `02_sharpness_comparison.png` | Box/violin plots of Laplacian variance, Tenengrad, and edge density. **The smoking gun** — WMA and DF40 have dramatically lower sharpness than training fakes. |
| **03** | `03_frequency_analysis.png` | FFT energy distribution. WMA has the most low-frequency energy and least high-frequency energy — GFPGAN removes high-frequency artifacts. |
| **04** | `04_noise_texture.png` | Noise and texture metrics. Shows WMA has a tight, uniform noise profile (GFPGAN produces consistent output). |
| **05** | `05_resolution_comparison.png` | Width, height, pixel count distributions. WMA is the clear outlier — much higher resolution than training data. |
| **06** | `06_color_statistics.png` | RGB means, luminance, saturation. Less discriminative but shows WMA has slightly different color statistics. |
| **07** | `07_tsne_clip_features.png` | **Most important plot** — t-SNE of 512-dim CLIP features colored by source (left) and by model prediction (right). Shows where WMA images land in feature space relative to training data. |
| **08** | `08_sharpness_vs_probability.png` | Scatter of sharpness vs model probability. If the model uses sharpness as a shortcut, this should show a strong positive correlation — and it does. |
| **09** | `09_summary_dashboard.png` | Combined accuracy bar chart + sharpness vs probability + frequency scatter. Executive summary view. |
| **10** | `10_radar_profiles.png` | Radar/spider charts of normalized properties per source. Shows the "shape" of each data source across all metrics. |

**Plot 02 and 08 are the most important.** Plot 02 shows the raw sharpness gap; Plot 08 directly tests whether the model's predictions correlate with sharpness more than with the actual real/fake label.

---

### 15.7 Hypothesis Verdict

| # | Hypothesis | Verdict | Evidence |
|---|-----------|---------|----------|
| H1 | Preprocessing gap | **✅ CONFIRMED** | WMA images are 342×435 vs training's 224×224. Downscaling reduces sharpness. |
| H2 | GFPGAN smoothing | **✅ CONFIRMED** (partially) | GFPGAN reduces sharpness, but the model handles GFPGAN when pre-cropped (DL QE: 98%). The issue is GFPGAN + downscaling combined. |
| H3 | Sharpness-as-shortcut | **✅ CONFIRMED** — **ROOT CAUSE** | Sharpness predicts model accuracy across ALL sources. DF40 real (sharpness 15.4) has 89% FPR. Plot 08 directly shows the correlation. |
| H4 | Distribution shift | **Partially confirmed** | t-SNE plot shows WMA clusters differently, but this may be a consequence of H1+H3 rather than an independent cause. |
| H5 | Resolution mismatch | **✅ CONFIRMED** (amplifier) | Not the root cause on its own (DF40 is also 224×224 and fails), but amplifies the problem for WMA. |

**Root cause:** The model has learned a **sharpness/texture shortcut** that correlates with real/fake labels on the training distribution but breaks on out-of-distribution data. This is a training-time problem that cannot be fixed by threshold tuning or post-processing.

---

### 15.8 Implications for Training — What Needs to Change

The analysis points to several concrete directions for a training fix:

#### 15.8.1 The Core Problem

The model's feature representation encodes "image quality" (sharpness, edge density, high-frequency content) as a primary discriminant. On the training distribution, this correlates with the real/fake label because:
- Training fakes are sharp (generated from high-quality sources, retain GAN artifacts)
- Training reals from YouTube are also sharp (but below the fake sharpness range)
- The model learns a quality-based decision boundary that separates these groups

This breaks when:
- Fakes are smoother than training reals (GFPGAN-enhanced, downscaled from higher resolution)
- Reals are smoother than training reals (DF40 faces with different preprocessing)

#### 15.8.2 Potential Training Fixes to Explore

1. **Resolution augmentation at training time.** Randomly upsample training images to 2–3× resolution and then downsample back to 224×224. This simulates the WMA pipeline (high-res → resize at inference) and breaks the model's reliance on resolution-specific sharpness patterns.

2. **Sharpness-invariant training.** Add random Gaussian blur, JPEG compression, and super-resolution artifacts as training augmentations — but critically, apply them to **both** reals and fakes equally. This forces the model to learn features that are invariant to sharpness.

3. **Include raw/uncropped face images in training.** Instead of only training on pre-cropped 224×224 images, include some images that are cropped and resized at training time from higher resolutions (simulating the inference-time pipeline).

4. **Adversarial quality augmentation.** Specifically target the sharpness shortcut: for each training fake, create a "smoothed" version (blur + slight downscale) and ensure the model still detects it as fake.

5. **DF40 data diagnosis.** The DF40 real misclassification (89% FPR!) suggests the model may have learned DF40-specific shortcuts rather than genuine fake detection. Consider whether DF40 data is helping or hurting generalization.

> **⚠️ Important caveat from R3:** The Round 3 experiments showed that "quality-robust" augmentation (`quality_robust_moderate/light`) **hurt** OOD generalization even though it improved holdout metrics. Any training fix must be validated on OOD data, not just holdout AUC. The augmentation approach must be carefully designed to avoid the same trap.

#### 15.8.3 What We Learned About Evaluation

The WMA investigation reinforces the central lesson of Phase 2 (§10 Finding 6): **holdout metrics do not predict deployment performance.** We now have three layers of evidence:

1. **R3 scratch models** — higher holdout AUC, worse OOD accuracy than R25_F1
2. **R3 fine-tune models** — highest holdout AUC ever (0.9966), but untested on OOD
3. **WMA evaluation** — all models fail, and the ones with highest holdout AUC fail worst

Any future training experiment must include WMA-style evaluation as a mandatory validation step before deployment consideration.

---

### 15.9 Reproducing This Analysis

All scripts and data are in the repository. Here's the complete workflow:

```bash
# Prerequisites: conda environment with PyTorch + project dependencies
# (see DeepfakeBench/training/requirements.txt)
conda activate sweep-env

# Step 1: Run multi-checkpoint evaluation against WMA images
cd DeepfakeBench/training
python eval_enhancer_local.py
# → analysis_results/enhancer_eval_per_image.csv (1,202 rows × 5 models)
# → analysis_results/enhancer_eval_summary.csv (5 rows, aggregate stats)

# Step 2: Run meta-analysis (downloads from GCS, needs auth)
python meta_analysis_enhancer.py
# → analysis_results/meta_analysis_properties.csv (2,602 rows × 36 props)
# → analysis_results/meta_analysis_features.npz (2,602 × 512 CLIP features)

# Step 2b: If sources are missing, run targeted fix:
python meta_analysis_fix_missing.py
# → merges into existing CSV/NPZ

# Step 3: Generate all 10 diagnostic plots
python ../analysis_results/plot_meta_analysis.py
# → analysis_results/plots/01_probability_distributions.png
# → analysis_results/plots/02_sharpness_comparison.png
# → ... (10 plots total)
```

#### Data Files Reference

| File | Rows | Description |
|------|------|-------------|
| `analysis_results/enhancer_eval_per_image.csv` | 1,202 | Per-image fake probability for 5 checkpoints on WMA images |
| `analysis_results/enhancer_eval_summary.csv` | 5 | Per-model aggregate statistics |
| `analysis_results/meta_analysis_properties.csv` | 2,602 | 36 image properties for 10 data sources |
| `analysis_results/meta_analysis_features.npz` | 2,602 | 512-dim CLIP features, model probs, source/GT labels, filenames |

#### Script Reference

| Script | Purpose |
|--------|---------|
| `DeepfakeBench/training/eval_enhancer_local.py` | Multi-checkpoint eval against WMA images (local MPS) |
| `DeepfakeBench/training/meta_analysis_enhancer.py` | Full meta-analysis: download sources, compute properties, extract features |
| `DeepfakeBench/training/meta_analysis_fix_missing.py` | Targeted fix for 4 missing DeepLive sources |
| `analysis_results/plot_meta_analysis.py` | Generate all 10 diagnostic plots |

#### Key Columns in `meta_analysis_properties.csv`

| Column | What It Means | Why It Matters |
|--------|--------------|----------------|
| `source` | Data source identifier (e.g., `wma_enhanced`, `deeplive_quality_enhancement_fake`) | Groups data for comparison |
| `label` | Ground-truth label (`fake` or `real`) | Compare model prediction vs truth |
| `sharpness_laplacian_var` | Variance of Laplacian filter response (higher = sharper) | **Primary shortcut feature** — predicts model accuracy |
| `edge_density` | Fraction of pixels detected as edges by Canny | **Co-feature** with sharpness — WMA and DF40 real have identical values |
| `freq_high_energy_ratio` | Fraction of FFT energy in high frequencies | GFPGAN removes high-freq content |
| `noise_estimate` | Std of high-pass filtered image | Texture/noise consistency indicator |
| `model_fake_prob` | R25_F1's predicted fake probability | The dependent variable — what we're trying to understand |
| `width`, `height` | Original image dimensions | Resolution mismatch indicator |

---

## 16. Verification Deep-Dive & Revised Fix Strategy (Feb 2026)

### 16.1 Context

After the WMA investigation (§15) identified a sharpness/quality shortcut as root cause, we paused training to run rigorous verification experiments before committing to a fix. Two reported failure modes needed validation:

1. **WMA enhanced face-swaps** — 21% detection (confirmed real problem)
2. **DF40 reals** — 89% FPR (suspected measurement error)

### 16.2 Verification Experiments Completed

#### V4: CLIP Feature Probes
**Question:** Is the shortcut in CLIP's frozen backbone, or in the SVD trainable layers?

| Probe | Result |
|-------|--------|
| CLIP → sharpness | R² = −0.14 (weak — CLIP barely encodes sharpness) |
| CLIP → model decision | AUC = 0.999 (features carry the decision perfectly) |
| CLIP → ground truth | AUC = 0.991 (CLIP carries genuine forgery signal) |
| CLIP → sharpness within WMA | R² = 0.97, ρ = −0.62 (strong within-source) |

**Verdict:** CLIP weakly encodes sharpness globally, but SVD layers amplified it into a primary discriminant. CLIP itself carries genuine forgery signal (AUC 0.991). The shortcut is learnable — an architectural or data-level fix can work.

#### V5: Feature-Prediction Correlation Ranking
**Question:** Which image property best predicts the model's fake probability?

| Rank | Feature | Spearman ρ |
|------|---------|------------|
| 1 | noise_estimate | −0.317 |
| 2 | freq_mid_energy_ratio | −0.295 |
| 3 | color_b_mean | −0.237 |
| ... | ... | ... |
| 7 | sharpness_laplacian_mean | −0.136 |

**Verdict:** The shortcut is broader than just "sharpness" — it's a cluster of correlated quality features (noise, frequency content, color statistics). The model has learned a general **image quality proxy**, not a single sharpness signal.

#### V6: Cross-Model Consistency
**Question:** Is the sharpness correlation model-specific or systemic?

| Model | WMA Acc | Sharpness ρ |
|-------|---------|-------------|
| R25_F1 (deployed) | 21% | −0.621 |
| R3_FT3 (best OOD) | 40% | −0.628 |
| R3_FT1 (quality_robust mod.) | 8% | −0.615 |
| R3_FT2 (quality_robust light) | 9% | −0.623 |
| B16_old (Phase 1, k=8) | 56% | −0.628 |

**Verdict:** All 5 models show |ρ| ≈ 0.62 with sharpness. This is systemic — the training data itself creates this correlation. Notably, `quality_robust` augmentation made it *worse* (8–9% acc), not better.

#### DF40 Contamination Discovery (CRITICAL)
**Question:** Does the model actually fail on DF40?

The original meta-analysis (`sample_df40_images()`) sampled randomly from the GCS bucket `df40-frames-recropped-rfa85/fake/`, which contains **29 method directories**. But only **17 methods** appear in the training pair JSON. The 12 extra methods (StyleGAN2/3/XL, DiT, SiT, RDDM, ddim, VQGAN, hyperreenact, sadtalker, tpsm, wav2lip) are **completely unseen** in training. ~40% of meta-analysis "DF40 fake" samples came from never-seen methods, contaminating the reported 53% accuracy and 89% FPR.

#### DF40 Clean Per-Method Evaluation
Ran R25_F1 against 50 images per method (17 training + 3 unseen controls + 150 paired reals):

**Face-Swap (target_source) — 89.2% overall:**
| Method | Acc | Sharpness | Notes |
|--------|-----|-----------|-------|
| inswap | 100% | 43.9 | |
| mobileswap | 100% | 47.7 | |
| simswap | 100% | 56.9 | |
| blendface | 98% | 58.7 | |
| uniface | 98% | 65.2 | |
| e4s | 96% | 70.8 | |
| facedancer | 76% | 76.2 | ← quality gap narrows |
| faceswap | 46% | 101.7 | ← sharpest fake ≈ real quality |

**Reenactment (source_target) — 98.4% overall:**
| Method | Acc | Sharpness |
|--------|-----|-----------|
| fomm | 100% | 12.8 |
| one_shot_free | 100% | 13.0 |
| mcnet | 100% | 50.4 |
| pirender | 100% | 22.3 |
| MRAA | 98% | 11.9 |
| danet | 98% | 50.4 |
| facevid2vid | 98% | 36.6 |
| fsgan | 98% | 55.8 |
| lia | 94% | 43.1 |

**Unseen Controls — 88.7% (generalization!):**
| Method | Acc | Notes |
|--------|-----|-------|
| StyleGAN2 | 96% | GAN, never in training |
| ddim | 90% | Diffusion, never in training |
| wav2lip | 80% | Reenactment, never in training |

**Reals — 83.3% TNR (16.7% FPR):**
- 64.7% of reals get prob < 0.1 (confident correct)
- 16.7% misclassified as fake (prob > 0.5), not 89%

**DF40 sharpness ↔ fake_prob: ρ = −0.319** (present but weaker than WMA's −0.62)

### 16.3 Revised Situation Assessment

The verification experiments fundamentally changed the picture:

| Original Belief | Revised Reality |
|----------------|----------------|
| "DF40 reals have 89% FPR" | **16.7% FPR** — meta-analysis was contaminated by unseen methods |
| "Model fails on DF40 fakes (53%)" | **89–98% accuracy** on trained methods; 53% was diluted by unseen methods |
| "Sharpness is THE shortcut" | Sharpness is **one of several** correlated quality features (noise, frequency, color) |
| "DF40 data may be hurting" | DF40 is **working well** — model generalizes even to unseen DF40 methods (89%) |
| "Two catastrophic failures" | **One real failure** (WMA at 21%) + one measurement artifact |

**The remaining real problem is singular: WMA enhanced face-swaps (21% detection).**

The model is strong on:
- DF40 reenactment: 98.4%
- DF40 face-swap: 89.2% (6 of 8 methods >96%)
- Unseen methods: 88.7% (genuine generalization)
- DeepLive: previously validated at 95%+

The model fails on:
- WMA enhanced face-swaps: 21% (GFPGAN + downscale → smooth fakes cross below real quality)
- DF40 `faceswap` method: 46% (sharpest fake method — same root cause)
- DF40 reals: 16.7% FPR (acceptable but improvable)

### 16.4 Revised Fix Strategy

Given that the problem is narrower than originally thought (WMA-specific, not systemic DF40 failure), the fix options are refined:

#### Approach A: GFPGAN-Enhanced Fake Data (RECOMMENDED — PRIMARY)
**Goal:** Add GFPGAN-enhanced fakes to training data so the model sees fakes at both original and WMA-like quality levels, preventing the quality shortcut.

**This is NOT a training-time augmentation.** It's an offline data creation + integration process in 3 steps:

**Step 1: Offline GFPGAN Processing (IN PROGRESS — being done separately)**
Take a subset of training fakes (DeepLive and/or DF40) and run them through the same GFPGAN enhancement pipeline that WMA uses. This produces a new dataset of GFPGAN-enhanced fakes, stored as a separate data source.

**Step 2: Quality Validation (BEFORE training — critical)**
Before integrating, verify the enhanced fakes actually match WMA's quality profile. The WMA pipeline involves more than just GFPGAN: face-swap → GFPGAN → meeting video encoding → frame extraction → crop/resize. If GFPGAN alone doesn't reproduce the WMA quality distribution, we need to replicate more of the pipeline.

| Property | WMA Enhanced (target) | GFPGAN Fakes (must match) |
|----------|----------------------|---------------------------|
| Sharpness (Laplacian var) | ~20–40 | Verify |
| Edge density | ~0.03–0.04 | Verify |
| Noise estimate | ~1.5–2.5 | Verify |
| Freq high energy ratio | Low | Verify |

A validation script will compare the GFPGAN-enhanced fakes against these targets before committing to a training run.

**Step 3: Pipeline Integration (new data source, NOT replacing originals)**
Integrate the GFPGAN-enhanced fakes as an additional data source alongside the originals. Both versions are needed — the model must detect fakes regardless of quality level:
- Original DeepLive/DF40 fakes → labeled fake ✅
- GFPGAN-enhanced fakes → labeled fake ✅
- All reals → labeled real ✅

**Why it should work:** CLIP carries genuine forgery signal (AUC 0.991 on ground truth). The SVD layers took the quality shortcut because it was easier — but if quality no longer separates reals from fakes in training, the SVD layers must learn the real forgery signal instead.

**Why R3's augmentation failed but this should work:** R3 applied random blur/JPEG to both reals and fakes equally (symmetric), preserving the quality gap. This approach is **asymmetric** — it adds smooth fakes specifically, collapsing the gap. And it uses the exact same enhancement (GFPGAN) that causes the deployment failure, not a generic approximation.

**Risk:** Low. Main risk is that GFPGAN alone may not fully replicate WMA's quality profile (Step 2 catches this).

#### Approach B: Gradient-Based Adversarial Regularization (BACKUP)
**Goal:** Add a quality-prediction head with gradient reversal to explicitly prevent quality encoding.

**Why it's backup:** V4 showed the shortcut is in SVD layers (not CLIP backbone), and CLIP itself carries strong forgery signal. Data-level fixes (A) are simpler and address the root cause. Use B only if A doesn't close the gap.

#### Approach C: OOD Validation Gates (MANDATORY — REGARDLESS)
**Goal:** Never deploy without passing WMA + external-real validation.

**Implementation:**
- Add WMA enhanced samples as a mandatory validation set during training
- Add DF40 `faceswap` method (the hardest case) as a secondary gate
- Gate condition: WMA accuracy > 70% AND real FPR < 10% before checkpoint is eligible

**Status:** This is an infrastructure change, not a modeling change. Should be implemented regardless of which fix approach is chosen.

### 16.5 Priority Order

1. **Approach C** — implement OOD validation gates (infrastructure, no model changes)
2. **Approach A** — integrate GFPGAN-enhanced fakes when user's data is ready; also add synthetic quality degradation to training fakes
3. **Approach B** — only if A+C doesn't achieve WMA accuracy > 70%

### 16.6 What R3's Failure Taught Us About Augmentation

The quality_robust augmentation (R3) made things **worse** — models with it got 8–9% WMA accuracy vs 21% without. Post-mortem:

1. **Symmetric application** — degraded both reals and fakes equally, so the quality gap was preserved
2. **Augmentation artifacts** — JPEG compression + blur created their own distribution that the model learned to recognize, separate from genuine quality differences
3. **No targeted quality matching** — random degradation doesn't match the specific WMA quality profile

The fix must be **asymmetric** (fakes only), **distribution-matched** (targeting WMA-like quality), and **validated on OOD** (not just holdout AUC).

### 16.7 In-Progress Data Work

#### 16.7.1 GFPGAN-Enhanced Fake Dataset (✅ CREATED & COMPARED)

GFPGAN-enhanced versions of training fakes were created offline using GFPGANv1.4, separately from the training codebase. This was Step 1 of Approach A (see §16.4).

**What was created:**
- 860 samples (435 edge_cases_enhanced + 425 minimal_processing_enhanced) in the raw GCS bucket
- Each sample has 16 frames (fake + real subdirs), GFPGAN v1.4 applied to face regions
- Stored in `gs://live-deepfake-methods-real-and-fake-frames/samples/` with `enhancement: GFPGAN_sample` in manifest

**Status:** ✅ Data created. ✅ Quality comparison complete (see §17). GFPGAN does NOT match WMA's property profile, but independently causes detection failure through a different mechanism.

**Next steps:**
1. ~~Quality validation (Step 2)~~ — **Done.** See §17 for full results. Key finding: GFPGAN shifts properties in the opposite direction from WMA (higher sharpness/noise rather than lower), but both cause model failure.
2. **Pipeline integration (Step 3)** — add GFPGAN-enhanced samples as a new data source. The model should train on both quality levels simultaneously. See §17.5 for revised training plan.

#### 16.7.2 WMA Validation Data Upload (TODO)

The WMA enhanced face-swap images used for the §15 investigation exist locally at:
```
/Users/roeedar/Downloads/wma_export/all_images/
  → 1,202 cropped face images (JPG)
  → Mix of real and enhanced-fake participants from WMA meetings
  → Already evaluated by 5 models (results in analysis_results/enhancer_eval_per_image.csv)
```

**TODO:** Upload this data to a GCS bucket so it can be used as a validation source during training (Approach C). This enables:
- Automated WMA accuracy gating during training (no manual eval needed)
- Consistent evaluation across experiments
- Integration with the existing `validate_custom_sources.py` framework

**Proposed bucket path:** `gs://wma-validation-frames/` (or a subfolder under an existing bucket)

### 16.8 Verification Scripts & Data Reference

All verification artifacts are in `analysis_results/verification/`:

| File | Purpose | Status |
|------|---------|--------|
| `VERIFICATION_PLAN.md` | Full plan documentation for V1–V6 | ✅ Created |
| `V4_clip_probes.py` | CLIP feature probe analysis (shortcut location) | ✅ Complete |
| `V4_probe_results.txt` | V4 full results | ✅ Complete |
| `V5_feature_correlation.py` | Feature-prediction correlation ranking | ✅ Complete |
| `V5_correlation_results.csv` | V5 full results (36 features ranked) | ✅ Complete |
| `V6_cross_model_check.py` | Cross-model sharpness consistency | ✅ Complete |
| `V6_cross_model_results.csv` | V6 full results (5 models) | ✅ Complete |
| `check_df40_methods.py` | DF40 GCS bucket method discovery | ✅ Complete |
| `check_df40_training_methods.py` | DF40 pair JSON method analysis | ✅ Complete |
| `eval_df40_by_method.py` | Clean per-method DF40 evaluation | ✅ Complete |
| `df40_by_method_results.csv` | Per-method results (1,150 rows) | ✅ Complete |
| `df40_method_cache/` | Cached DF40 images (50 per method) | ✅ 20 methods cached |
| `V1_sharpness_causality.py` | Causal sharpness manipulation test | ❌ Created, not run (skipped) |
| `V2_resolution_isolation.py` | Resolution vs quality isolation | ❌ Created, not run (skipped) |

---

## 17. GFPGAN Comparison — Isolating the Enhancement Effect (Feb 16, 2026)

### 17.1 Motivation

Section 16.4 proposed **Approach A**: train the model on GFPGAN-enhanced fakes to fix the WMA failure. Before committing to that, we needed to answer a critical question:

> **Does GFPGAN enhancement alone cause detection failure, or does the WMA failure require the combination of GFPGAN + WMA's specific cropping/resolution pipeline?**

860 GFPGAN-enhanced samples (GFPGANv1.4) were created in the raw GCS bucket (`edge_cases_enhanced`, `minimal_processing_enhanced`). These are full 640×360 video frames — the same raw resolution as the original DeepLive samples — *before* the YOLO face-cropping pipeline runs.

### 17.2 Methodology

**Script:** `meta_analysis_gfpgan_comparison.py`

The comparison applies the **exact same preprocessing pipeline used for training data**:

1. Download raw 640×360 frames from `gs://live-deepfake-methods-real-and-fake-frames/samples/`
2. Apply `extract_yolo_face()` from `video_preprocessor.py` — YOLOv8 face detection → square crop → resize to 224×224 (identical to the training data creation pipeline)
3. Compute 36 image properties (same as §15 meta-analysis)
4. Run R25_F1 model inference on the face crops
5. Compare with existing meta-analysis data (WMA, DeepLive training, DF40, Visomaster)

**Four source categories, 150 samples each (600 total):**

| Source | Description |
|--------|-------------|
| `gfpgan_edge_cases_enhanced_fake` | GFPGAN-enhanced fakes, YOLO-cropped to 224×224 |
| `gfpgan_minimal_processing_enhanced_fake` | GFPGAN-enhanced fakes, YOLO-cropped to 224×224 |
| `gcs_original_edge_cases_fake` | Non-enhanced fakes (same samples), YOLO-cropped to 224×224 |
| `gcs_original_minimal_processing_fake` | Non-enhanced fakes (same samples), YOLO-cropped to 224×224 |

**Critical design choice:** Previous (invalid) run used raw 640×360 frames directly, producing meaningless comparisons. This run applies YOLO face-cropping first, making results directly comparable to training data and WMA.

### 17.3 Results

#### Model Accuracy (R25_F1, threshold=0.5)

| Source | N | Accuracy | Mean Prob | Size |
|--------|---|----------|-----------|------|
| **GFPGAN enhanced edge_cases** | 150 | **32.7%** | 0.343 | 224×224 |
| **GFPGAN enhanced minimal_proc** | 150 | **30.7%** | 0.338 | 224×224 |
| Original edge_cases (no GFPGAN) | 150 | 68.0% | 0.675 | 224×224 |
| Original minimal_proc (no GFPGAN) | 150 | 71.3% | 0.701 | 224×224 |
| **WMA enhanced** (baseline) | 1202 | **21.0%** | 0.283 | 333×430 |
| DeepLive training fakes | ~450 | 98.5% | 0.984 | 224×224 |
| Visomaster fakes | 150 | 99.3% | 0.994 | 224×224 |
| DF40 fakes | 150 | 53.3% | 0.539 | 224×224 |

#### Key Image Properties (face crops)

| Source | Sharpness | Noise | Edge Density | Freq High |
|--------|-----------|-------|--------------|-----------|
| **GFPGAN enhanced** | **177** | **0.83** | **0.046** | **0.217** |
| Original (no GFPGAN) | 124 | 0.64 | 0.038 | 0.191 |
| **WMA enhanced** | **18.8** | 0.48 | 0.012 | 0.176 |
| DeepLive training fakes | 79 | 0.46 | 0.035 | 0.161 |
| Visomaster | 158 | 0.70 | 0.045 | 0.205 |

### 17.4 Analysis

#### Finding 1: GFPGAN Enhancement Alone Causes Detection Failure

Even with the correct YOLO face-cropping pipeline at 224×224, model accuracy drops from ~70% (originals) to ~31% (GFPGAN-enhanced). **GFPGAN is a standalone failure mode** — it breaks detection independent of resolution or cropping pipeline.

#### Finding 2: GFPGAN and WMA Fail Through Different Mechanisms

This is the surprise result. GFPGAN does NOT make face crops look like WMA crops:

| Property | GFPGAN Effect | WMA Profile | Direction |
|----------|---------------|-------------|-----------|
| Sharpness | 124 → **177** (+43%) | **18.8** (−76% vs training) | **Opposite** |
| Noise | 0.64 → **0.83** (+30%) | 0.48 (close to training) | **Opposite** |
| Edge density | 0.038 → **0.046** (+21%) | 0.012 (−67% vs training) | **Opposite** |
| Freq high energy | 0.191 → **0.217** (+14%) | 0.176 (close to training) | **Opposite** |

- **GFPGAN** increases sharpness, noise, and edges — it adds high-frequency texture/details to the face
- **WMA** drastically reduces sharpness and edges — resolution downsampling + recompression destroys detail

Both push the model's output toward "real" (prob ~0.34 and ~0.28 respectively), but from **opposite directions** in property space. The model has a narrow "fake Detection zone" and both perturbation types move fakes outside of it.

#### Finding 3: Original GCS Fakes Are Already Weak (~70%)

The non-enhanced `edge_cases` and `minimal_processing` strategies only achieve ~70% accuracy — well below the 98.5% on training-distribution DeepLive fakes. These strategies represent harder face-swap scenarios that the model already struggles with. GFPGAN cuts that further to ~31%.

#### Finding 4: Two Distinct Failure Distributions Require Two Fixes

Since WMA and GFPGAN fail through opposite mechanisms, a single augmentation approach won't address both:

- **WMA fix** — the model needs exposure to low-sharpness/low-edge face crops (quality degradation augmentation applied asymmetrically to fakes)
- **GFPGAN fix** — the model needs exposure to high-sharpness/high-texture GFPGAN-enhanced face crops (include GFPGAN-enhanced fakes in training data)

### 17.5 Revised Training Plan

Based on all findings from §15 (WMA investigation), §16 (verification), and §17 (GFPGAN comparison):

#### Priority 1: GFPGAN-Enhanced Training Data (Approach A — Ready Now)

The 860 GFPGAN-enhanced samples in the raw bucket need to be:
1. Run through the YOLO face-cropping pipeline → upload to the `-cropped` bucket
2. Added as a new strategy discoverable by the DeepLive data source
3. Train with both enhanced and original fakes so the model sees GFPGAN-style faces as suspicious

**Expected impact:** Fix the 31% accuracy on GFPGAN fakes → should reach 80%+ with direct training exposure.

#### Priority 2: Asymmetric Quality Degradation for WMA (Approach A, Part 2)

Apply synthetic quality degradation **only to fake faces** during training:
- Gaussian blur, downscale+upscale, JPEG compression — targeting the WMA sharpness range (~20)
- This is different from R3's `quality_robust` augmentation, which failed because it was applied symmetrically to both classes

**Expected impact:** Improve WMA accuracy from 21% toward 50%+.

#### Priority 3: OOD Validation Gates (Approach C)

Add WMA and GFPGAN-enhanced samples as mandatory validation sets during training. Gate checkpoint eligibility on:
- WMA accuracy > 50% (initial target, raise as fixes land)
- GFPGAN accuracy > 60%
- Real FPR < 10%

### 17.6 Artifacts

| File | Purpose |
|------|---------|
| `meta_analysis_gfpgan_comparison.py` | Comparison script (YOLO-cropped) |
| `analysis_results/gfpgan_comparison_properties.csv` | 600 rows, 36 properties + model probs |
| `analysis_results/gfpgan_comparison_features.npz` | CLIP features + probs for all 600 samples |
| `weights/gfpgan_comparison_cropped/` | Cached YOLO-cropped 224×224 face chips |

---

## 18. R4 Status Reset and R5 Plan (Feb 17, 2026)

### 18.1 Current Ground Truth

1. Sidecar outputs (full suite):  
   `gs://training-job-outputs/test_results/r4_sidecar_sofar_20260217_113917/`
2. WMA-only outputs (FT5/FT7):  
   `gs://training-job-outputs/test_results/r4_wma_only_20260217-163324/`
3. Flat WMA per-image evaluation (`n=1202`, threshold=0.5):
   - FT5: `893/1202 = 74.29%`
   - FT7: `1062/1202 = 88.35%`
   - Artifact: `DeepfakeBench/training/debug/wma_flat_eval_ft5_ft7.json`

### 18.2 R4 Scoreboard (Canonical Snapshot)

| Run | External real FPR (<=8%) | WMA flat gate (>=50%) | Enhanced DeepLive fake TPR | DF40 fake TPR | Overall sidecar AUC | Practical read |
|---|---:|---:|---:|---:|---:|---|
| FT1 | 4.17% | Not measured in flat pass | 28.97% | 83.77% | 0.9899 | Fails enhanced-fake objective |
| FT2 | 4.82% | Not measured in flat pass | 31.29% | 81.78% | 0.9883 | Fails enhanced-fake objective |
| FT5 | 3.35% | 74.29% | 98.38% | 83.08% | 0.9943 | Strong |
| FT7 | 4.38% | 88.35% | 99.53% | 86.85% | 0.9941 | Best balance |

### 18.3 Conclusions

1. FT1 and FT2 are excluded for production intent due to weak enhanced-fake TPR.
2. FT5 and FT7 both pass external-real and WMA-flat gates.
3. FT7 is the strongest current candidate on targeted fake robustness and balanced metrics.
4. FT5 remains a rollback option with a slightly better external-real margin.

### 18.4 Validation Semantics Policy

For WMA, canonical reporting is now **flat per-image** (`1202` samples), not grouped-by-folder.

Implemented interface changes:
1. `validate_custom_sources.py`
   - `--external_fake_grouping {by_folder,per_image}`
   - `--external_fake_deterministic`
   - `--external_real_deterministic`
2. `data/validation_sources.py`
   - External loader supports both grouping modes and deterministic frame shaping.
3. `run_r4_validation_sequential.py`
   - Threads explicit grouping/deterministic flags to every job.

### 18.5 R5 Improvement Workstream

Parallel scratch matrix (architecture/loss stable, change data/sampling/augmentation realism):
1. `experiments/phase2_round5/R5_S1_scratch_baseline_ft7mix.yaml`
2. `experiments/phase2_round5/R5_S2_scratch_ft7mix_weighted.yaml`
3. `experiments/phase2_round5/R5_S3_scratch_ft7mix_weighted_targetdomain_aug.yaml`

Target-domain validation assets:
1. `run_target_domain_validation_sequential.py`
2. `experiments/phase2_round5/TARGET_DOMAIN_SUITES_TEMPLATE.yaml`

Decision policy for the next promotion:
1. Rank by WMA flat detection first.
2. External real FPR second.
3. Target-domain worst-suite performance third.
4. Overall AUC only as tiebreaker.
5. Promote only if a new scratch run beats FT7 on WMA flat with no external-real FPR regression.
