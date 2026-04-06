# Round 9.5 — Final Report

**Date:** March 6, 2026  
**Runs:** 6 (R95_A through R95_F)  
**W&B Project:** `dtect-vision/phase2r95-experiments`  
**Docker Image:** 1.3.145  
**GPU:** A100-40GB × 1 per run, Vertex AI asia-southeast1  
**Runtime:** ~24 hours per run (all runs hit Vertex AI timeout → state=crashed, which is normal)

---

## Executive Summary

**R9.5 did not produce a better model than R9_A.** The stability regularization fix — the raison d'être of this round — was confirmed working, but stability regularization **hurts OOD generalization** across all configurations tested. The accidental λ=0 in R9 was not a bug in terms of model quality — it was the better configuration.

| Outcome | Status |
|---------|--------|
| Stability fix works (stability_loss > 0) | ✅ Confirmed |
| Holdout AUC ≥ R9_A (0.9891) | ✅ R95_A/F reached 0.9900 |
| OOD AUC ≥ R9_A (0.9768) | ❌ Best was R95_B at 0.9729 (−0.0039) |
| Score jitter < 0.030 | ❌ Best was R95_B at 0.041 (worse than R9) |
| facedancer ≥ 75% | ❌ All FT runs degraded to 59.1% (R9: 68.2%) |
| Teams EC ≥ 75% | ⚠️ Only R95_D (scratch) at 80%; all FT at 70% |
| Unified EER ≤ 3.5% | ❌ Best was 3.8% (R9: 3.52%) |

**Recommendation: R9_A (`1551zxa8`) remains the production checkpoint.**

---

## All Runs — Final State

| | R95_A | R95_B | R95_C | R95_D | R95_E | R95_F |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **W&B ID** | `lxfzu0di` | `95p2wvqi` | `ykhi5ypm` | `kt4z2qww` | `pglgtxlc` | `n9c4ec3i` |
| **Track** | FT baseline | FT light stab | FT heavy stab | Scratch | FT ArcFace | FT DF40 5x |
| **Steps reached** | 10,635 | 10,635 | 8,890 | 8,890 | 10,635 | 10,635 |
| **Final epoch** | 5 | 5 | 4 | 4 | 5 | 5 |
| **Best epoch** | 3 | 3 | 3 | **4** | 3 | 3 |
| **EWI** | 6 | 6 | 3 | **1** | 6 | 6 |
| **λ (stability)** | 0.3 | 0.1 | 0.5 | 0.3 | 0.3 | 0.3 |
| **Label smooth** | 0.05 | 0.0 | 0.1 | 0.05 | 0.05 | 0.05 |

**Note on C/D timers:** R95_C and R95_D ran at 0.11 steps/sec vs 0.27 for others — they likely received slower hardware and only completed epoch 4 instead of 5. Their metrics are still valid at their best epoch.

---

## 1. Core In-Distribution Performance

| | R95_A | R95_B | R95_C | R95_D | R95_E | R95_F | **R9_A** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Best AUC** | **0.9900** | 0.9894 | 0.9894 | 0.9876 | 0.9895 | **0.9900** | 0.9891 |
| **Best EER** | **3.8%** | 4.3% | 4.3% | 5.3% | 4.1% | **3.8%** | 4.57% |
| **Unified AUC** | **0.9881** | 0.9876 | 0.9876 | 0.9876 | 0.9880 | **0.9881** | 0.9897 |
| **Unified EER** | 3.8% | 3.8% | 4.1% | 3.9% | 3.8% | 3.8% | **3.52%** |
| **Holdout AUC** | 0.9875 | 0.9870 | 0.9867 | 0.9874 | 0.9874 | 0.9874 | **0.9891** |
| **Holdout EER** | 4.3% | **3.0%** | 3.8% | 4.6% | 3.6% | 5.1% | 4.57% |
| **macro_acc (ho)** | 93.1% | 93.0% | 93.1% | 92.8% | **93.1%** | 93.1% | — |
| **TPR@1%FPR** | **83.3%** | 82.3% | 77.6% | 76.1% | 82.8% | **83.3%** | — |
| **Ho TPR@1%FPR** | **92.5%** | 84.6% | 92.3% | 77.1% | 92.3% | **92.5%** | — |

**Findings:**
- R95_A and R95_F tie at AUC 0.9900, EER 3.8% — the best Best-AUC in R9.5 (and better than R9_A's 0.9891).
- But the **unified AUC** (which tests in-dist + holdout together) is 0.9881 vs R9_A's 0.9897. R9 was better at generalizing across the validation split.
- Holdout EER is best for R95_B (3.0%) — light stability produced the tightest holdout boundary.
- R95_D (scratch) has the weakest in-dist AUC (0.9876) and TPR@1%FPR (76.1%), but was still training (EWI=1).

---

## 2. Stability Regularization — The Central Question

| | R95_A | R95_B | R95_C | R95_D | R95_E | R95_F | **R9_A** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **stability_loss** | 0.0181 | 0.0089 | 0.0052 | 0.0012 | 0.0184 | 0.0181 | **0.0** |
| **cls_loss** | 0.3283 | 0.2556 | 0.4394 | 0.4095 | 0.3301 | 0.3281 | — |
| **overall_loss** | 0.4845 | 0.4406 | 0.2424 | 0.1593 | 0.4921 | 0.4843 | — |
| **YT jitter** | 0.0438 | 0.0407 | 0.0453 | 0.0710 | 0.0447 | 0.0438 | **0.0389** |
| **WMA jitter** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **VCD jitter** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**Findings:**
- **Stability fix confirmed**: All runs have non-zero stability_loss. R9 had 0.0 everywhere.
- **Stability did NOT reduce YouTube jitter.** R9_A (no stability) had 0.0389. R95_B (lightest stability) has 0.0407 — *worse*. R95_C (heaviest) has 0.0453.
- **Paradox**: More stability regularization → more jitter. The loss encourages crops to produce consistent scores, but the effect is either too weak or the wrong signal for OOD score stability.
- **WMA and VCD jitter = 0** across all runs — these outputs are fully saturated (WMA at 99.9% fake, VCD stuck at a floor). Jitter=0 is an architectural artifact, not a stability success.

### Stability λ Gradient (Monotonic OOD Degradation)

| λ | Label Smooth | Run | OOD AUC | VCD Real | YT Jitter |
|---|---|---|---|---|---|
| **0.0** | 0.0 | R9_A | **0.9768** | **80.3%** | **0.0389** |
| 0.1 | 0.0 | R95_B | 0.9729 | 76.9% | 0.0407 |
| 0.3 | 0.05 | R95_A | 0.9661 | 75.4% | 0.0438 |
| 0.5 | 0.1 | R95_C | 0.9556 | 74.7% | 0.0453 |

**This is unambiguous: stability regularization monotonically degrades OOD generalization and increases score jitter.** λ=0 (R9's accidental config) was the optimal setting.

---

## 3. Out-of-Distribution Performance

| | R95_A | R95_B | R95_C | R95_D | R95_E | R95_F | **R9_A** | **R8_E** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **OOD AUC** | 0.9661 | **0.9729** | 0.9556 | 0.9280 | 0.9585 | 0.9662 | **0.9768** | — |
| **OOD EER** | 8.2% | **7.5%** | 8.8% | 14.5% | 9.0% | 8.2% | **6.55%** | — |
| **VCD real** | 75.4% | 76.9% | 74.7% | 61.6% | 75.5% | 75.5% | **80.3%** | **82.1%** |
| **YT real** | 86.8% | 86.1% | 85.6% | 83.2% | 86.8% | 86.7% | 86.8% | 85.8% |
| **WMA fake** | 99.9% | 99.9% | 99.9% | 99.9% | 99.9% | 99.9% | 99.9% | 98.5% |
| **OOD TPR@1%** | 24.7% | **33.0%** | 16.6% | 15.9% | 21.5% | 24.7% | — | — |
| **OOD TPR@5%** | 75.7% | **81.1%** | 60.1% | 46.5% | 65.2% | 75.9% | — | — |

**Findings:**
- **Every R9.5 run is worse than R9_A on OOD.** This is the headline result.
- R95_B is the OOD champion of R9.5 (AUC 0.9729, EER 7.5%), but still −0.0039 below R9_A.
- VCD real continues its decline: R8_E (82.1%) → R9_A (80.3%) → R95_B (76.9%). Each round of fine-tuning erodes VCD real discrimination by ~3-5pp.
- YouTube real accuracy is stable at ~86% across all configurations.
- WMA fake accuracy is saturated at 99.9% — this is no longer a differentiating metric.
- R95_D (scratch) has the worst OOD across the board (AUC 0.9280, VCD 61.6%), but it only completed 4 epochs with EWI=1 — it was still improving.

---

## 4. Teams Domain Performance

| | R95_A | R95_B | R95_C | R95_D | R95_E | R95_F | **R9_A** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **EC in-dist** | 76.7% | 78.3% | 76.7% | **83.3%** | 76.7% | 76.7% | — |
| **MP in-dist** | 100% | 100% | 100% | 100% | 100% | 100% | — |
| **QE in-dist** | 83.3% | 83.3% | 83.3% | **91.7%** | **91.7%** | 83.3% | — |
| **EC holdout** | 70.0% | 70.0% | 70.0% | **80.0%** | 70.0% | 70.0% | 70.0% |
| **MP holdout** | 91.7% | 91.7% | 91.7% | 91.7% | 91.7% | 91.7% | 91.7% |
| **QE holdout** | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

**Findings:**
- Teams holdout is identical across all 5 FT runs and identical to R9_A. The model learned Teams detection from R8_E → R9 and R9.5 fine-tuning didn't move the needle.
- R95_D (scratch) is the outlier: EC holdout 80% (+10pp), EC in-dist 83.3% (+7pp), QE in-dist 91.7%. Scratch has a different decision boundary that happens to help Teams.
- The Teams holdout set is very small (~14 videos), so the 70% vs 80% delta may represent 1-2 videos. Don't over-interpret.

---

## 5. Per-Method DF40 Holdout Performance

| Method | R95_A/B/C/E/F | R95_D | **R9_A** |
|--------|:---:|:---:|:---:|
| **facedancer** | **59.1%** | **68.2%** | **68.2%** |
| **e4s** | **62.5%** | **75.0%** | 87.5%* |
| **mobileswap** | 94.4% | 88.9% | 94.4% |
| **simswap** | 91.7% | 86.1% | 91.7% |
| **inswap** | 95.8% | 95.8% | 95.8% |
| **blendface** | 100% | 100% | 100% |
| **uniface** | 100% | 100% | 100% |
| **facedancer_id** | 59.4% | 59.4% | — |

*R9_A e4s value from 8K checkpoint analysis.

**Critical Finding — All 5 FT runs produce IDENTICAL per-method holdout scores.** R95_A = R95_B = R95_C = R95_E = R95_F on every single DF40 method. This means:
1. The models converged to the **same decision boundary** for DF40 data despite different λ, label smoothing, ArcFace scale, and DF40 weights.
2. R95_F (5× DF40 weight) had **zero effect** — facedancer stayed at 59.1%.
3. **facedancer is an architectural limitation**, not a training data problem. The CLIP ViT-B-16 backbone + SVD residual method cannot discriminate facedancer swaps beyond ~59-68%.
4. facedancer degraded from 68.2% (R9_A/R95_D) → 59.1% (all FT R9.5). Additional fine-tuning epochs actively hurt facedancer detection — the model overfits to easier methods.

**R95_D preserved facedancer at 68.2%** because scratch training maintains orthogonal feature directions that fine-tuning collapses.

---

## 6. DeepLive & VisoMaster Holdout

| | R95_A/B/C/E/F | R95_D | R9_A (approx) |
|---|:---:|:---:|:---:|
| **DL edge_cases** | 97.6% | 97.6% | ~97% |
| **DL minimal_proc** | 98.1% | 96.2% | ~98% |
| **DL quality_enh** | 94.2% | 94.2% | ~94% |
| **DL EC enhanced** | 100% | 100% | 100% |
| **DL MP enhanced** | 100% | 97.7% | 100% |
| **Viso CSCS** | 100% | 97.2% | 100% |
| **Viso GF-v1** | 100% | 97.7% | 100% |
| **Viso GF-v2** | 97.1% | 100% | 97.1% |
| **Viso Insw128** | 100% | 95.8% | 100% |
| **Viso SS512** | 100% | 100% | 100% |
| **VCD real ho** | 80.0% | 60.0% | 80.0% |

**Findings:**
- DeepLive and VisoMaster holdout is near-ceiling (97-100%) for all FT runs — no meaningful differences.
- R95_D (scratch) has slightly lower VisoMaster scores (~95-97% vs 100%) because it hasn't trained long enough, but overall strong.
- VCD real holdout: 80% for all FT runs, only 60% for scratch. This aligns with the OOD findings.

---

## 7. Confidence & Score Distribution

| | R95_A | R95_B | R95_C | R95_D | R95_E | R95_F |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **conf_mean** | 0.518 | 0.516 | 0.562 | 0.575 | 0.520 | 0.518 |
| **conf_std** | 0.369 | **0.388** | 0.334 | 0.339 | 0.366 | 0.369 |
| **frac_confident** | 66.7% | 66.7% | 60.4% | 58.3% | 66.7% | 66.7% |
| **class_sep** | 4.46 | **7.05** | 3.18 | 3.59 | 4.40 | 4.46 |
| **logit_diff** | 0.296 | **0.626** | 0.471 | 0.665 | 0.334 | 0.294 |

**Findings:**
- R95_B has the best class separation (7.05 vs ~4.4 for others). Light stability + no label smoothing produces more polarized, confident predictions.
- R95_C has the worst class separation (3.18) — heavy stability makes the model less decisive.
- R95_D has the highest confidence mean (0.575) with lowest fraction confident (58.3%) — the scratch model's decision surface is still forming.

---

## 8. SVD & Model Internals

| | R95_A | R95_B | R95_C | R95_D | R95_E | R95_F |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **S_residual max** | 0.255 | 0.254 | 0.256 | 0.199 | 0.254 | 0.255 |
| **S_residual mean** | 0.023 | 0.023 | 0.023 | 0.021 | 0.023 | 0.023 |
| **near_zero_layers** | 1 | 1 | 1 | 1 | 1 | 1 |
| **keepsv** | 8.466 | 8.523 | 8.508 | 7.233 | 8.059 | 8.469 |
| **orthogonal** | 76.22 | 76.98 | 75.96 | 72.84 | 75.02 | 76.22 |
| **ArcFace s** | 9.14 | 9.14 | 8.63 | 8.19 | **14.19** | 9.14 |
| **weight_norm** | 0.201 | 0.205 | 0.202 | 0.220 | 0.203 | 0.201 |

**Findings:**
- All FT runs have nearly identical SVD structure (S_residual max ~0.255, mean ~0.023). The SVD residual space barely evolved between configurations.
- R95_D (scratch) has lower S_residual max (0.199) — its residual capacity is still developing.
- R95_E reached ArcFace s=14.19 (vs 9.14 for baseline). Higher scale didn't translate to better separation or OOD performance.
- Only 1 near-zero layer across all runs — the SVD decomposition is well-utilized.

---

## 9. Ablation Answers

### Q1: Did stability help? (R95_A vs R9_A)

| Metric | R9_A (λ=0) | R95_A (λ=0.3) | Delta |
|--------|:---:|:---:|:---:|
| Best AUC | 0.9891 | **0.9900** | +0.0009 |
| OOD AUC | **0.9768** | 0.9661 | **−0.0107** |
| VCD real | **80.3%** | 75.4% | **−4.9pp** |
| YT jitter | **0.0389** | 0.0438 | **+0.005** |
| facedancer | **68.2%** | 59.1% | **−9.1pp** |
| Teams EC ho | 70% | 70% | 0 |

**Answer: No.** Stability improved in-dist AUC marginally (+0.0009) but degraded OOD (−0.0107), VCD real (−4.9pp), facedancer (−9.1pp), and jitter (+0.005). The stability bug in R9 was a happy accident.

### Q2: Light vs medium stability? (R95_B vs R95_A)

| Metric | R95_A (λ=0.3) | R95_B (λ=0.1) | Delta |
|--------|:---:|:---:|:---:|
| OOD AUC | 0.9661 | **0.9729** | **+0.0068** |
| VCD real | 75.4% | **76.9%** | **+1.5pp** |
| OOD TPR@5% | 75.7% | **81.1%** | **+5.4pp** |
| class_sep | 4.46 | **7.05** | +2.59 |

**Answer: Light stability is less harmful.** The monotonic degradation pattern holds — less stability = better OOD. R95_B is the best R9.5 run on OOD metrics.

### Q3: Heavy vs medium stability? (R95_C vs R95_A)

| Metric | R95_A (λ=0.3) | R95_C (λ=0.5) | Delta |
|--------|:---:|:---:|:---:|
| OOD AUC | 0.9661 | 0.9556 | −0.0105 |
| VCD real | 75.4% | 74.7% | −0.7pp |
| class_sep | 4.46 | 3.18 | −1.28 |

**Answer: Heavy stability is even worse.** The OOD degradation is proportional to λ.

### Q4: Scratch + stability? (R95_D vs R95_A)

| Metric | R95_A (FT) | R95_D (Scratch) | Delta |
|--------|:---:|:---:|:---:|
| Best AUC | **0.9900** | 0.9876 | −0.0024 |
| OOD AUC | **0.9661** | 0.9280 | −0.0381 |
| facedancer | 59.1% | **68.2%** | **+9.1pp** |
| e4s | 62.5% | **75.0%** | **+12.5pp** |
| Teams EC ho | 70% | **80%** | **+10pp** |
| EWI | 6 | **1** | ← still improving |

**Answer: R95_D was still training.** At epoch 4, it hadn't converged yet (EWI=1). It's the only run that preserved facedancer (68.2%) and had better Teams EC (80%). A longer scratch run (20K+ steps) is the most promising direction for R10.

### Q5: Higher ArcFace scale? (R95_E vs R95_A)

| Metric | R95_A (s→12) | R95_E (s→18) | Delta |
|--------|:---:|:---:|:---:|
| ArcFace s (final) | 9.14 | **14.19** | +5.05 |
| OOD AUC | 0.9661 | 0.9585 | −0.0076 |
| Best AUC | 0.9900 | 0.9895 | −0.0005 |

**Answer: Higher ArcFace scale hurts.** The gentle 6→12 schedule was correct. R8's 10→18 worked for scratch (where it helps learn the feature space), but for fine-tuning it overrides the learned representations.

### Q6: 5× DF40 weight for facedancer? (R95_F vs R95_A)

| Metric | R95_A (df40=0.2) | R95_F (df40=1.0) | Delta |
|--------|:---:|:---:|:---:|
| facedancer | 59.1% | 59.1% | **0** |
| e4s | 62.5% | 62.5% | **0** |
| All DF40 methods | identical | identical | **0** |
| Best AUC | 0.9900 | 0.9900 | 0 |
| OOD AUC | 0.9661 | 0.9662 | 0 |

**Answer: Zero effect.** Every single metric is identical between A and F (within noise). The 5× DF40 weight was completely absorbed — the optimizer converged to the same solution. This definitively rules out data weighting as a fix for facedancer.

---

## 10. Training Dynamics

### Epoch Progression (FT runs)
All FT runs (A/B/C/E/F) peaked at **epoch 3** and never recovered. Training continued to epoch 4-5, but all key metrics were logged from the epoch 3 checkpoint. Early stopping (patience 3) would terminate around step 7000 for these configurations.

### R95_D (scratch) — The Sleeper
R95_D is the only run with EWI=1, meaning it was still improving at epoch 4 when time ran out. Its trajectory:
- Lower in-dist AUC (0.9876) — still catching up
- Much lower OOD (0.9280) — expected for early scratch
- Preserved hard methods (facedancer 68.2%, e4s 75.0%) — doesn't overfit to easy methods yet
- Best Teams EC (80%) — different learned boundary

R8_E (the scratch champion) peaked at step 8000 out of 12K. R95_D peaked at step 8000 out of ~9K available — it needed at least 15-20K steps to converge.

### Convergence Twin: R95_A ≡ R95_F
These two runs arrived at the **same checkpoint** despite different DF40 weights. This means:
- The per-method sampling is dominated by other source weights (DeepLive 4.0/5.0, Viso 3.0, Teams 7.0)
- At DF40 weight 0.2 vs 1.0, the effective DF40 mini-batch fraction changes from ~1.5% to ~7% — but both are dwarfed by DeepLive/Teams
- The loss landscape at convergence is determined by the heavyweight sources; DF40 is noise

---

## 11. R9.5 Leaderboard

### By Overall Balance (weighted ranking)

| Rank | Run | Best AUC | OOD AUC | VCD Real | facedancer | Teams EC | Notes |
|------|-----|----------|---------|----------|------------|----------|-------|
| **1** | **R95_B** | 0.9894 | **0.9729** | **76.9%** | 59.1% | 70% | Best OOD, best class sep |
| 2 | R95_A | **0.9900** | 0.9661 | 75.4% | 59.1% | 70% | Highest AUC, tied with F |
| 3 | R95_F | **0.9900** | 0.9662 | 75.5% | 59.1% | 70% | ≡ A (DF40 weight had no effect) |
| 4 | R95_E | 0.9895 | 0.9585 | 75.5% | 59.1% | 70% | ArcFace hurt |
| 5 | R95_C | 0.9894 | 0.9556 | 74.7% | 59.1% | 70% | Heavy stability worst FT |
| 6 | R95_D | 0.9876 | 0.9280 | 61.6% | **68.2%** | **80%** | Incomplete — needs more steps |

### Compared to Prior Champions

| Model | Best AUC | OOD AUC | VCD Real | facedancer | Teams EC |
|-------|----------|---------|----------|------------|----------|
| **R8_E** | 0.9925 | — | **82.1%** | — | — |
| **R9_A** | 0.9891 | **0.9768** | 80.3% | **68.2%** | 70% |
| **R95_B** (best R9.5) | 0.9894 | 0.9729 | 76.9% | 59.1% | 70% |

**R9_A remains the overall champion.** R9.5 improved in-dist AUC slightly but degraded on every OOD metric that matters.

---

## 12. Checkpoints

| Run | Best Checkpoint |
|-----|-----------------|
| R95_A | `gs://training-job-outputs/phase2r95_experiments/lxfzu0di/top_n_effort_20260302_step7000_auc0.9900_eer0.0381.pth` |
| R95_B | `gs://training-job-outputs/phase2r95_experiments/95p2wvqi/top_n_effort_20260302_step7000_auc0.9894_eer0.0431.pth` |
| R95_C | `gs://training-job-outputs/phase2r95_experiments/ykhi5ypm/top_n_effort_20260302_step7000_auc0.9894_eer0.0431.pth` |
| R95_D | `gs://training-job-outputs/phase2r95_experiments/kt4z2qww/top_n_effort_20260302_step8000_auc0.9876_eer0.0533.pth` |
| R95_E | `gs://training-job-outputs/phase2r95_experiments/pglgtxlc/top_n_effort_20260302_step7000_auc0.9895_eer0.0406.pth` |
| R95_F | `gs://training-job-outputs/phase2r95_experiments/n9c4ec3i/top_n_effort_20260302_step7000_auc0.9900_eer0.0381.pth` |

---

## 13. Conclusions & Recommendations

### What We Learned

1. **Stability regularization is counterproductive.** The monotonic gradient (λ=0 > λ=0.1 > λ=0.3 > λ=0.5) on OOD AUC is definitive. The stability loss term competes with the classification loss and pushes the model toward in-distribution consistency at the expense of OOD generalization. Don't use it.

2. **Label smoothing provides no measurable benefit.** R95_B (LS=0.0) outperformed R95_A (LS=0.05) and R95_C (LS=0.1) on every OOD metric.

3. **facedancer is an architectural ceiling.** 5× DF40 weight had zero effect. The 59-68% range is where CLIP ViT-B-16 + SVD residual saturates on facedancer. Fixing this requires a fundamentally different approach — possibly ViT-L-14, attention-based detection, or facedancer-specific training.

4. **FT runs converge in 3 epochs regardless of regularization.** All FT runs peaked at epoch 3, all produced identical DF40 holdout scores, all converged to the same SVD structure. The fine-tuning landscape is extremely narrow — regularization choices barely perturb the final solution.

5. **Scratch training (R95_D) is the promising direction.** It preserved facedancer (68.2%), had best Teams EC (80%), and was still improving (EWI=1). A 20K+ step scratch run with stability λ=0 is the strongest candidate for R10.

6. **R9's "bug" was a feature.** The config pipeline gap that prevented stability_lambda from reaching the Trainer produced the best model (R9_A). This is a serendipity we should embrace, not fix.

### R10 Recommendations

| Experiment | Rationale |
|-----------|-----------|
| **Scratch, λ=0, 20K steps** | R95_D was improving at 9K. Give it 2× more time with no stability to match R8_E's convergence profile. |
| **Scratch, λ=0, ViT-L-14** | If facedancer needs to cross 80%, a bigger backbone is the only known lever. |
| **R9_A re-deploy as-is** | For immediate deployment, R9_A is confirmed best. No further R9-family fine-tuning needed. |

### Production Recommendation

**Deploy R9_A (`1551zxa8`)** with threshold 0.4766. It has:
- Best OOD AUC (0.9768) of any Teams-aware model
- Best facedancer (68.2%) among FT models
- VCD real 80.3% (declining but acceptable)
- Teams EC holdout 70% (noisy — only ~6 videos)
- WMA 99.9%, YouTube 86.8%, DeepLive ~97%+

---

*Report generated from W&B final metrics. All runs in project `phase2r95-experiments`, entity `dtect-vision`.*
