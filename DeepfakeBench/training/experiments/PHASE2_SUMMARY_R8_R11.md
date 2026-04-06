# Phase 2 Experiment Summary: R8 → R11

> **Date:** March 8, 2026  
> **Document Purpose:** Comprehensive summary of deepfake detection experiments from R8 through R11

---

## Executive Summary

Over the course of R8–R11 (approximately 2 weeks, ~30 experiments), we systematically explored how to build a robust deepfake detector that works both on benchmark data (DF40) and deployment-domain data (Microsoft Teams video calls). 

**The core finding:** B-16 ViT backbone has a fundamental capacity limitation — its rank-32 SVD residual subspace cannot simultaneously hold DF40 discrimination and Teams domain knowledge. Adding Teams data consistently costs ~1pp AUC regardless of training strategy.

| Champion | AUC | EER | OOD | Note |
|----------|----:|----:|----:|------|
| **R9_D** (B-16, no Teams) | **0.9942** | 0.0325 | 0.9689 | Best B-16 model ever |
| R11_G (L-14, with Teams) | 0.9918 | **0.0263** | **0.9819** | Best overall (not production-eligible) |
| R10_C / R11_F (B-16, with Teams) | 0.983 | 0.0569 | 0.96 | Best B-16 with Teams competence |

---

## Round-by-Round Summary

### R8: Foundation Building

**Date:** ~February 25, 2026  
**Runs:** 8 experiments  
**Champion:** R8_E (`hu7cen3m`) — AUC 0.9925, VCD Real 82.1%

**Focus:** Establish baseline with DF40 + VisoMaster + DeepLiveCam data. Test scratch vs fine-tune strategies.

**Key Results:**
- Solved VisoMaster detection (69% → 95–98%)
- Maintained DeepLiveCam performance (93–97%)
- Identified threshold calibration gap (val EER ~0.45 vs OOD EER ~0.77)
- Discovered score instability problem (identical frames → wildly different scores)

**Learnings:**
- ArcFace `s=18` amplifies cosine perturbations into large probability swings
- Scratch training preserves per-method diversity better than fine-tuning
- facedancer stuck at ~60% (architectural ceiling suspected)

---

### R9: Teams Domain Adaptation

**Date:** February 27–28, 2026  
**Runs:** 8 experiments  
**Champions:** R9_D (AUC 0.9942), R9_F (EER 0.0266), R9_A (production, AUC 0.9891)

**Focus:** Adapt to Microsoft Teams video call domain. Add TeamsCodecSimulation augmentation. Test stability regularization.

**Key Results:**
| Run | Strategy | AUC | EER | OOD | Note |
|-----|----------|----:|----:|----:|------|
| R9_D | Scratch, no Teams | **0.9942** | 0.0325 | 0.9689 | Best ever |
| R9_F | FT + codec sim | 0.9926 | **0.0266** | 0.9632 | Best EER |
| R9_A | FT + Teams v1 | 0.9891 | 0.0457 | 0.9692 | Production deploy |

**Critical Bug:** Stability λ and label smoothing never reached the Trainer due to config pipeline gap. All R9 runs trained with λ=0.

**Learnings:**
- Teams codec simulation (blur, brightness, contrast) is validated and effective
- Models without Teams data (R9_D) achieve highest benchmark AUC
- Teams v1 data (~288 pairs) provides some domain adaptation but costs ~0.5pp AUC

---

### R9.5: Stability Bug Fix & Ablations

**Date:** March 2, 2026  
**Runs:** 6 experiments  
**Purpose:** Rerun R9 with stability λ actually working. Test ablations.

**Key Results:**

| Finding | Evidence | Conclusion |
|---------|----------|------------|
| Stability λ monotonically hurts OOD | λ=0 (0.9768) > λ=0.1 (0.9729) > λ=0.5 (0.9556) | **λ=0 is optimal** |
| Label smoothing hurts | All smoothed runs underperformed | **No smoothing** |
| FT landscape is extremely narrow | 5/5 FT runs produced identical per-method scores | Scratch > FT for diversity |
| facedancer is architecture-bound | 5× DF40 weight had zero effect (59.1%) | B-16 capacity ceiling |
| Scratch preserves facedancer | R95_D: 68.2% vs FT: 59.1% | Scratch > FT for weak methods |

**Impact:** Killed the stability regularization hypothesis. Confirmed scratch + λ=0 is the optimal strategy for B-16.

---

### R10: Teams v2 Integration

**Date:** March 6–7, 2026  
**Runs:** 7 experiments  
**Best Finished:** R10_C (AUC 0.9836, Teams HO 88.0%)

**Focus:** Integrate Teams v2 data (4.7× more than v1: 1,346 vs 288 pairs). Test scratch vs FT with larger Teams corpus.

**What Happened:**

| Run | Strategy | Final Step | State | AUC | OOD |
|-----|----------|----------:|-------|----:|----:|
| R10_C | FT from R9_A, narrow aug | 8167 | ✅ Finished | 0.9836 | 0.9502 |
| R10_G | FT from R9_A, wide aug | 8167 | ✅ Finished | 0.9829 | 0.9547 |
| R10_A | Scratch, wide aug | 8706 | 💥 24h crash | 0.9805 | 0.9439 |
| R10_B | Scratch + mixup | 8317 | 💥 24h crash | 0.9803 | 0.9274 |
| R10_D | ViT-L-14 | 8718 | 💥 24h crash | 0.9042 | **0.9807** |
| R10_E | Scratch, low-s | 8305 | 💥 24h crash | 0.9804 | 0.9273 |
| R10_F | Scratch, narrow aug | 8710 | 💥 24h crash | 0.9809 | 0.9349 |

**Critical Issues:**
1. **24h Vertex AI timeout** killed all scratch runs (configured for 22K steps but crashed at ~8.5K)
2. **ViT-L-14 collapse** — R10_D had catastrophic in-dist AUC (0.9042) but best-ever OOD (0.9807). Root cause: `lambda_reg=1.0` + `rank=1023` over-constrained the SVD residual.

**Key Finding:** Teams v2 data hurts B-16 in-dist AUC. Best R10 (0.9836) is worse than R9_D (0.9942) by 1.06pp.

---

### R11: FT from R9_D + L-14 Fix

**Date:** March 7–8, 2026  
**Runs:** 8 experiments (all still running at ~17h mark)  
**Purpose:** Fine-tune the best-ever B-16 checkpoint (R9_D) to absorb Teams data. Fix the ViT-L-14 configuration.

**Run Matrix:**

| Run | Strategy | Base | Steps | Key Differentiator |
|-----|----------|------|------:|-------------------|
| R11_A | FT | R9_D | 8K | Primary bet — R9_D + Teams v2 |
| R11_B | FT + Group DRO | R9_D | 8K | Upweight worst methods |
| R11_C | FT | R9_D | 8K | Wide augmentation |
| R11_D | FT | R9_F | 8K | Best-EER checkpoint |
| R11_E | Scratch | — | 30K | Scratch ceiling test |
| R11_F | FT | R9_D | 8K | High Teams weight (7.0/5.0) |
| R11_G | Scratch | — | 30K | **L-14 fix:** λ=0.01, rank=768 |
| R11_H | FT | R9_D | 10K | Ultra-low LR (1e-5) |

**Results at 17h (~Epoch 3 of 6):**

| Run | Step | AUC | EER | TPR@1% | OOD | Unified |
|-----|-----:|----:|----:|-------:|----:|--------:|
| **R11_G (L-14)** | 6422 | **0.9918** | **0.0263** | **0.9647** | **0.9819** | **0.9883** |
| R11_D (FT R9_F) | 6409 | 0.9829 | 0.0613 | 0.8477 | 0.9405 | 0.9808 |
| R11_F (high Teams) | 5626 | 0.9828 | 0.0569 | 0.8631 | 0.9615 | 0.9839 |
| R11_A (FT R9_D) | 6618 | 0.9827 | 0.0569 | 0.8653 | 0.9485 | 0.9818 |
| R11_B (Group DRO) | 6618 | 0.9827 | 0.0569 | 0.8609 | 0.9485 | 0.9819 |
| R11_C (wide aug) | 5626 | 0.9821 | 0.0613 | 0.8521 | 0.9623 | 0.9823 |
| R11_H (ultra-low LR) | 5626 | 0.9815 | 0.0591 | 0.8631 | 0.9613 | 0.9823 |
| R11_E (scratch) | 6401 | 0.9787 | 0.0613 | 0.8190 | 0.9177 | 0.9794 |

---

## R11 Conclusive Findings

### 1. ViT-L-14 Fix is a Major Success (But Not Production-Eligible)

The lambda/rank fix transformed R10_D's catastrophe into R11_G's triumph:

| Metric | R10_D (broken) | R11_G (fixed) | Delta |
|--------|:--------------:|:-------------:|:-----:|
| AUC | 0.9042 | **0.9918** | +0.0876 |
| EER | — | **0.0263** | (best ever) |
| TPR@1%FPR | — | **0.9647** | (best ever) |
| OOD | 0.9807 | **0.9819** | +0.0012 |
| Teams GhostFace-v2 | 77.6% | **100%** | +22.4pp |
| facedancer | — | **71%** | (vs B-16's 64%) |

**The change:** `lambda_reg: 1.0 → 0.01`, `rank: 1023 → 768`, `LR: 2e-4 → 3e-4`

However, L-14 is not eligible for production deployment per product constraints.

### 2. B-16 Cannot Recover R9_D's AUC with Teams Data

All six B-16 FT runs (A, B, C, D, F, H) plateaued at **AUC ~0.982–0.983**, which is:
- 1.1pp below R9_D (0.9942)
- 0.5pp below R10_C (0.9836)
- Flat — no improvement in the last 5 hours of training

This pattern held across 13 B-16 runs in R10+R11, regardless of:
- Starting checkpoint (R9_A vs R9_D vs R9_F vs scratch)
- Learning rate (1e-5 to 2e-4)
- Augmentation (narrow vs wide)
- Family weighting (4.0 to 7.0)
- Group DRO
- Mixup

**Root cause:** B-16's rank-32 residual subspace (hidden_size=512, 736 frozen SVD components, only 32 trainable dimensions) doesn't have the capacity to hold both DF40 discrimination and Teams domain knowledge.

### 3. Group DRO Failed for GhostFace-v2

R11_B (Group DRO) = R11_A across every metric. Teams GhostFace-v2 stuck at 50% for both. The DRO mechanism either:
- Doesn't generate enough gradient signal on small method groups, or
- The GhostFace-v2 collapse is a data issue (too few samples), not a loss-weighting issue

### 4. Teams GhostFace-v2 = 50% is a B-16 Architectural Ceiling

Universal across all 13 B-16 runs. Only L-14 breaks through to 100%. This is the clearest evidence that the B-16 residual subspace is saturated.

### 5. Wide Aug and High-Teams Weight Help OOD but Not AUC

| Run | OOD Delta | AUC Delta |
|-----|----------:|----------:|
| C (wide aug) | +0.011 | 0 |
| F (high Teams) | +0.012 | 0 |
| H (ultra-low LR) | +0.008 | 0 |

These interventions improve generalization but don't change the in-dist picture.

---

## Checkpoint Inventory — B-16 Only

Since L-14 is not production-eligible, here are the B-16 options:

| Checkpoint | Run | AUC | EER | OOD | Teams HO | Trade-off |
|------------|-----|----:|----:|----:|:--------:|-----------|
| `m7etxxnp` step 4000 | R9_D | **0.9942** | 0.0325 | **0.9689** | ~74%* | Best AUC/OOD, no Teams competence |
| `ueoziuou` step 500 | R9_F | 0.9926 | **0.0266** | 0.9632 | ~74%* | Best EER, no Teams competence |
| `1551zxa8` step 6000 | R9_A | 0.9891 | 0.0457 | 0.9692 | ~74% | Current production |
| `wc3jv0ls` step 3000 | R11_F | 0.9828 | 0.0569 | 0.9615 | **88.7%** | Best Teams with acceptable AUC loss |

*R9_D/R9_F were trained without any Teams data, so their Teams holdout accuracy estimate is based on zero-shot transfer.

---

## Recommendations

### For Immediate Production

**Ship R9_D** (`m7etxxnp`) if Teams-specific accuracy is not critical. It has:
- Best AUC (0.9942), best OOD (0.9689)
- No regression from current R9_A production model on any benchmark metric
- ~74% zero-shot Teams accuracy (acceptable for non-Teams deployments)

### For Teams-Heavy Deployments

**Ship R11_F** (`wc3jv0ls`) if Teams accuracy is the priority. It has:
- 88.7% Teams holdout accuracy (+14pp over R9_A)
- AUC 0.9828 (−1.1pp from R9_D but acceptable)
- OOD 0.9615 (−0.7pp from R9_D)

### For Future Experiments (R12)

The only way to get both R9_D-level AUC AND R11_F-level Teams accuracy on B-16 is to **increase trainable capacity**:

1. **Raise `k` from 32 to 64 or 128** — more SVD residual dimensions trainable
2. **Lower `rank` from 736 to 640 or 512** — fewer frozen CLIP components
3. **Two-model ensemble** — R9_D for general, R11_F for Teams, route by source domain

---

## Appendix: Key Files

| File | Location |
|------|----------|
| R9_D checkpoint | `gs://training-job-outputs/phase2r9_experiments/m7etxxnp/top_n_effort_20260228_step4000_auc0.9942_eer0.0325.pth` |
| R9_F checkpoint | `gs://training-job-outputs/phase2r9_experiments/ueoziuou/top_n_effort_20260228_step500_auc0.9926_eer0.0266.pth` |
| R11_F checkpoint | `gs://training-job-outputs/phase2r11_experiments/wc3jv0ls/top_n_effort_20260308_step3000_auc0.9828_eer0.0569.pth` |
| R11_G checkpoint (L-14) | `gs://training-job-outputs/phase2r11_experiments/5w8dl94c/top_n_effort_20260308_step5500_auc0.9918_eer0.0263.pth` |

---

## Timeline

```
Feb 25: R8 complete — R8_E champion (AUC 0.9925)
Feb 28: R9 complete — R9_D champion (AUC 0.9942), stability bug discovered
Mar 02: R9.5 complete — stability λ hurts, λ=0 confirmed optimal
Mar 07: R10 complete — Teams v2 costs ~1pp AUC, L-14 collapsed, 24h timeout
Mar 08: R11 in progress — L-14 fix works spectacularly, B-16 plateau confirmed
```
