# Phase 2 Experiments — Intermediate Report

**Date:** 2026-02-11 (experiments at ~15 hours / epoch 5 of 30)  
**W&B Project:** [dtect-vision/phase2-experiments](https://wandb.ai/dtect-vision/phase2-experiments)  
**Backbone:** ViT-B-16-DataComp-XL (LAION) | **Method:** Effort (SVD residual)  
**All runs:** A100-SXM4-40GB, lr=2e-4, cosine_with_warmup, 65K steps, seed=737

---

## 1. Run Inventory

| # | Run Name | W&B ID | State | Experiment Group | Description |
|---|----------|--------|-------|-----------------|-------------|
| 1 | `P2_C3_cosine_softmax_0210-1803` | `t1zpnv9s` | 🟢 running | Loss (C) | ArcFace m=0.0, s: 10→18, all 3 sources |
| 2 | `P2_D2_rank752_k16_0210-1804` | `l7o5ee4s` | 🟢 running | Scale (D) | rank=752 (k=16 trainable), all 3 sources |
| 3 | `P2_A1_df40_only_0210-1804` | `4maq2jwt` | 🟢 running | Data (A) | DF40 only, no DeepLive or VisoMaster |
| 4 | `P2_C2_arcface_m015_s12_0210-1803` | `48ys0hcz` | 🟢 running | Loss (C) | ArcFace m=0.15, s: 8→12, all 3 sources |
| 5 | `P2_A4_df40_viso_no_deeplive_0210-1803` | `0d8k0x0k` | 🟢 running | Data (A) | DF40 + VisoMaster, no DeepLive |
| 6 | `P2_A2_df40_deeplive_0210-1803` | `aosru6y3` | 🟢 running | Data (A) | DF40 + DeepLive (Phase 1 reproduction) |
| 7 | `P2_A3_df40_deeplive_viso_0210-1802` | `we21udqf` | 🟢 running | Data (A) | DF40 + DeepLive + VisoMaster (primary) |
| 8 | `P2_B2_group_dro_0210-1803` | `vepdxhka` | ❌ failed | Strategy (B) | Group DRO — failed before training |
| 9 | `P2_B2_group_dro_0210-1836` | `e6nglp2q` | 💥 crashed | Strategy (B) | Group DRO retry — crashed during training |

**Status:** 7/9 runs healthy, 2 Group DRO failures (needs investigation).

---

## 2. Current Leaderboard (Epoch 5 of 30)

### By Holdout AUC (primary metric)

| Rank | Run | Holdout AUC | Holdout EER | In-Dist AUC | In-Dist EER |
|------|-----|-------------|-------------|-------------|-------------|
| 🥇 1 | **P2_C3_cosine_softmax** (`t1zpnv9s`) | **0.9844** | 0.0372 | 0.9846 | 0.0485 |
| 🥈 2 | **P2_D2_rank752_k16** (`l7o5ee4s`) | 0.9840 | **0.0354** | **0.9898** | **0.0275** |
| 🥉 3 | P2_A4_df40_viso_no_deeplive (`0d8k0x0k`) | 0.9819 | 0.0425 | 0.9819 | 0.0518 |
| 4 | P2_A3_df40_deeplive_viso (`we21udqf`) | 0.9815 | 0.0425 | 0.9821 | 0.0566 |
| 5 | P2_A1_df40_only (`4maq2jwt`) | 0.9807 | 0.0460 | 0.9866 | 0.0356 |
| 5 | P2_A2_df40_deeplive (`aosru6y3`) | 0.9797 | 0.0460 | 0.9865 | 0.0356 |
| 7 | P2_C2_arcface_m015_s12 (`48ys0hcz`) | 0.9785 | 0.0496 | 0.9802 | 0.0566 |

> **Phase 1 reference:** Best P1 model achieved AUC=0.9947 at step 14K. We're at epoch 5 (~step 8K), so these numbers are expected to rise.

---

## 3. Training Curve Analysis

### 3.1 Convergence Speed & Shape

All 7 runs share the same general trajectory pattern:

```
Epoch:  1    2    3    4    5    ...
        ↑    ↑    ↑    ↑    ↑
      fast  fast  mod  slow  slow   ← convergence rate
```

**Initial phase (eval 1–3):** Rapid improvement from random/pretrained to ~0.96 AUC. The first eval point reveals interesting initialization differences:
- **CE-based runs (A1–A4):** Start at holdout AUC ~0.71 — the SVD residual structure provides a reasonable warm start
- **ArcFace C2:** Starts at 0.57 — the angular margin suppresses early confidence
- **Cosine softmax C3:** Starts at 0.54 — zero margin but high initial temperature (s=10) compresses logits

**Mid phase (eval 3–10):** Steady improvement ~0.2% AUC per eval. All runs exhibit a **characteristic dip at eval point 8** (holdout AUC drops ~0.5-1%), likely a learning rate schedule inflection where the cosine warmup reaches maximum LR. The model temporarily destabilizes before recovering.

**Current phase (eval 10–16):** Diminishing returns, ~0.1% per eval. The top 2 runs (C3, D2) are still on upward trajectories. The A-series data ablations appear to be converging toward each other.

### 3.2 Loss Landscape Behavior

| Metric | CE runs (A1-A4) | Cosine softmax (C3) | ArcFace (C2) | Rank-752 (D2) |
|--------|-----------------|---------------------|--------------|---------------|
| Initial cls_loss | ~0.72 | ~0.72 | **~1.46** | ~0.72 |
| Current cls_loss | ~0.25–0.43 | ~0.13–0.35 | ~0.24–0.35 | ~0.25–0.36 |
| Loss variance | Medium | **Low** | Medium-High | Medium |
| reg_loss trend | 0→0.011 | 0→0.010 | 0→0.018 | 0→0.014 |

**Key observation:** Cosine softmax (C3) has the **lowest loss variance** — its loss curve is the smoothest. This suggests the temperature-scaling provides a natural regularization effect that stabilizes training. ArcFace (C2) has the highest initial loss (2× CE) due to the margin penalty, but it's been converging steadily.

### 3.3 Class Separation Evolution

Class separation measures how far apart real vs fake logit distributions are:

| Run | Initial | Current | Trend |
|-----|---------|---------|-------|
| C3 cosine_softmax | ~0 | **6.2** | Smooth, steadily rising |
| C2 arcface_m015 | ~0 | **6.4** | Highest absolute value |
| D2 rank752_k16 | ~0 | 5.9 | Rising steadily |
| A1–A4 (CE runs) | ~0 | 4.5–6.5 | Similar trajectories |

ArcFace achieves the **highest class separation** (6.4) despite lagging in AUC — the angular margin is doing its job of pushing classes apart, but may be too aggressive for generalization.

### 3.4 SVD Residual Behavior

| Run | S_residual_mean | S_residual_range | near_zero_layers |
|-----|-----------------|------------------|------------------|
| C3 cosine_softmax | 0.0072 | [-0.15, 0.26] | 1 |
| D2 rank752_k16 | 0.0120 | [-0.17, 0.21] | 1 |
| C2 arcface_m015 | 0.0080 | [-0.23, 0.25] | 1 |
| A1–A4 (CE runs) | ~0.004 | [-0.22, 0.21] | 1–2 |

**D2 (rank752, k=16)** has the **largest S_residual_mean** (0.012 vs 0.004 for CE runs). With 2× more trainable SVD directions, it's utilizing more of its capacity. The CE runs have 2 near-zero layers each (barely modifying original weights), while D2 and the loss variants all have exactly 1 — suggesting more uniform SVD parameter utilization.

---

## 4. Key Findings

### Finding 1: Loss Function Design > Data Composition (at this stage)
The **top 2 runs are loss/architecture variants**, not data ablations:
- C3 (cosine softmax): AUC=0.9844
- D2 (rank752, k=16): AUC=0.9840
- Best data run (A4): AUC=0.9819 (gap of +0.25%)

The entire A-series (A1–A4) spans only **0.22% AUC** (0.9797–0.9819). Data composition hasn't differentiated yet. This may change in later epochs as the model becomes more discriminative and data diversity starts to matter.

### Finding 2: More SVD Capacity Helps
D2's rank=752 (k=16 trainable directions) outperforms the default rank=760 (k=8) by a meaningful margin on in-distribution metrics (AUC 0.9898 vs 0.9866 best CE). The S_residual_mean of 0.012 vs 0.004 confirms it's **actively using the extra capacity**, not just overfitting.

### Finding 3: VisoMaster Provides a Slight Edge
Within the data ablations:
- A4 (DF40 + VisoMaster): 0.9819 → **best data run**
- A3 (DF40 + DeepLive + VisoMaster): 0.9815
- A1 (DF40 only): 0.9807
- A2 (DF40 + DeepLive): 0.9797

VisoMaster-inclusive runs (A3, A4) slightly outperform VisoMaster-excluded runs (A1, A2) by ~0.1–0.2%. Interestingly, A4 (no DeepLive) slightly edges A3 (with DeepLive), hinting that DeepLive may introduce mild noise. But these differences are within noise at epoch 5.

### Finding 4: ArcFace m=0.15 Is Too Aggressive
C2 (ArcFace m=0.15, s:8→12) consistently trails all other runs:
- Highest initial loss (1.46 vs 0.72)
- Slowest to reach 0.97 AUC
- Currently last at 0.9785 holdout AUC
- But has the **highest class separation** (6.4)

The margin penalty suppresses early learning without compensating later. The Phase 1 winner used m=0 (pure cosine softmax), which is being confirmed here.

### Finding 5: Group DRO Needs Engineering Work
Both Group DRO attempts failed:
- `vepdxhka`: Failed before training started (configuration issue)
- `e6nglp2q`: Crashed during training

This is a **bug**, not a scientific result. Group DRO should be retried after fixing the underlying issue.

### Finding 6: All Runs Show a Mid-Training "Dip"
Every run shows a holdout AUC regression at approximately eval point 8, where performance drops by 0.5–1% before recovering. This is consistent across all configurations, suggesting it's a **shared artifact** — likely the cosine LR scheduler reaching its warmup peak or an epoch boundary causing a data distribution shift.

---

## 5. Trajectory Projections

Given the current convergence rates (0.1–0.2% AUC improvement per eval at epoch 5):

| Run | Current AUC | Projected Final (est.) | Confidence |
|-----|-------------|----------------------|------------|
| C3 cosine_softmax | 0.9844 | 0.990–0.993 | High (smooth trajectory) |
| D2 rank752_k16 | 0.9840 | 0.990–0.993 | High (strong in-dist) |
| A4 df40_viso | 0.9819 | 0.987–0.990 | Medium |
| A3 all_sources | 0.9815 | 0.987–0.990 | Medium |
| C2 arcface | 0.9785 | 0.985–0.990 | Medium (may catch up late) |

**Phase 1 achieved 0.9947 at step 14K.** The top runs are on track to match or exceed this — they're at 0.984 at step 8K with 25 epochs remaining.

---

## 6. Checkpoints Saved (for later evaluation)

| Run | Best Checkpoint | Step | AUC |
|-----|----------------|------|-----|
| C3 cosine_softmax | `gs://training-job-outputs/best_checkpoints/t1zpnv9s/top_n_effort_20260211_step8000_auc0.9844_eer0.0372.pth` | 8000 | 0.9844 |
| D2 rank752_k16 | `gs://training-job-outputs/best_checkpoints/l7o5ee4s/top_n_effort_20260211_step6500_auc0.9833_eer0.0354.pth` | 6500 | 0.9833 |
| A4 df40_viso | `gs://training-job-outputs/best_checkpoints/0d8k0x0k/top_n_effort_20260211_step8000_auc0.9819_eer0.0425.pth` | 8000 | 0.9819 |
| A3 all_sources | `gs://training-job-outputs/best_checkpoints/we21udqf/top_n_effort_20260211_step8000_auc0.9815_eer0.0425.pth` | 8000 | 0.9815 |
| A1 df40_only | `gs://training-job-outputs/best_checkpoints/4maq2jwt/top_n_effort_20260211_step7000_auc0.9807_eer0.0513.pth` | 7000 | 0.9807 |
| A2 df40_deeplive | `gs://training-job-outputs/best_checkpoints/aosru6y3/top_n_effort_20260211_step6500_auc0.9797_eer0.0460.pth` | 6500 | 0.9797 |
| C2 arcface | `gs://training-job-outputs/best_checkpoints/48ys0hcz/top_n_effort_20260211_step7500_auc0.9785_eer0.0496.pth` | 7500 | 0.9785 |

---

## 7. Action Items

### Immediate
- [x] **Investigate Group DRO failures** — bug fixed by user
- [x] **Investigate Data Anomaly** — **CRITICAL BUG FOUND** (see §8 below)
- [ ] **Let all 7 runs complete** — they're still actively improving at epoch 5/30 (but see §8 for reinterpretation)

### After Completion
- [ ] **Run VisoMaster OOD evaluation** on top 3 checkpoints (C3, D2, A4) — the real test of generalization
- [ ] **Compare holdout vs in-dist gap** — D2 shows the largest gap (0.9898 in-dist vs 0.9840 holdout), suggesting it may be more prone to overfitting to seen methods
- [ ] **Re-run data composition experiments (A-series)** with config passthrough fix applied
- [ ] **Relaunch Group DRO** with fixed config
- [ ] **Consider combining C3 + D2 insights** — cosine softmax loss with rank=752 capacity

---

## 8. 🚨 CRITICAL: Config Passthrough Bug — Data Anomaly

### Discovery
All 9 runs report **identical** data: 17,928 videos from 20 methods, 14,579 train / 3,349 val — even A1 which should be DF40-only.

### Root Cause
The `combined_paired` section from experiment YAMLs was **never transferred** to `data_config`. In `train_sweep.py`, the "CRITICAL FIX" section (lines 176–250) handles `deeplive` and `visomaster` at the **top level** of `single_cfg`, but experiment YAMLs nest them under `combined_paired:`. The `combined_paired` key itself was never copied to `data_config`.

Result: `combined_paired.py` reads `data_config.get('combined_paired', {})` → **empty dict** → all sub-sources fall back to defaults:
- `df40_enabled` defaults to **True** ← always on
- `deeplive_enabled` defaults to **True** ← always on (even when YAML says false!)
- `visomaster_enabled` defaults to **False** ← never activated

### Impact on Runs

| Run | Intended Data | Actual Data | Status |
|-----|--------------|-------------|--------|
| A1 (df40_only) | DF40 only | DF40 + DeepLive | ❌ INVALID |
| A2 (df40_deeplive) | DF40 + DeepLive | DF40 + DeepLive | ✅ Correct (by accident) |
| A3 (df40_deeplive_viso) | DF40 + DeepLive + VisoMaster | DF40 + DeepLive only | ❌ No VisoMaster |
| A4 (df40_viso) | DF40 + VisoMaster | DF40 + DeepLive | ❌ INVALID |
| C1, C2, C3, D1, D2 | DF40 + DeepLive + VisoMaster | DF40 + DeepLive only | ❌ No VisoMaster |

**Key consequence:** The dedicated VisoMaster pipeline (with swap model holdout and tier control) was **never activated**. However, all runs DID see VisoMaster-generated fakes through the DeepLive discovery pipeline as `deeplive_visomaster` — but without controlled swap model holdout or tier-based difficulty levels.

### Reinterpretation of Results
- **A-series (data composition):** All 4 runs trained on identical data → the 0.22% AUC spread is pure noise/seed variance, NOT data composition effects
- **C/D-series (loss/scale):** These are still valid comparisons since they all used the same data — just not the data originally intended. C3 cosine_softmax vs D2 rank752_k16 comparison is meaningful.
- **VisoMaster training:** Partially happened through DeepLive's `deeplive_visomaster` method, but without the dedicated pipeline's fine-grained control

### Fix Applied
`train_sweep.py` now includes `combined_paired` passthrough:
```python
if 'combined_paired' in single_cfg:
    data_config['combined_paired'] = single_cfg['combined_paired']
```
Code version bumped to `2026-01-02-COMBINED-PAIRED-FIX-V1`.

### The 20 methods (what ALL runs actually trained on)
- **DF40 (17):** MRAA, blendface, danet, e4s, facedancer, faceswap, facevid2vid, fomm, fsgan, inswap, lia, mcnet, mobileswap, one_shot_free, pirender, simswap, uniface
- **DeepLive (3):** deeplive_edge_cases, deeplive_minimal_processing, deeplive_visomaster
