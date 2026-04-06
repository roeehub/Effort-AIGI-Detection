# Winning Runs Registry — Phase 1 through R12

**Last updated:** March 11, 2026  
**Scope:** Every "winner" or promoted checkpoint from each experiment round, with full identification and context.

---

> **⚠️ OOD Comparability Note (R10+ vs R8/R9)**  
> Starting in R10, the training data expanded significantly (Teams v2 codec-degraded data, more VisoMaster swap models).
> Starting in R12, the OOD evaluation suite itself expanded — adding **teams_ood_real (300)** and **teams_ood_fake (300)** sources on top of the original {youtube_avspeech, zoom_vcd_real, wma_failure_fake}. This makes OOD AUC in R12 **not directly comparable** to R8/R9 numbers.
> We are also deliberately making the task harder each round — more diverse fakes, harder real-domain sources, codec-degraded inputs. **Similar or slightly lower numbers on the harder R10+ benchmarks should be considered genuine progress** relative to the R8/R9 results on the easier eval suite.

---

## Quick Reference — The Winners

| Round | Date | Winner | W&B ID | Holdout AUC | Key OOD Metric | Checkpoint |
|-------|------|--------|--------|-------------|----------------|------------|
| **Phase 1** | ~Jan 19 | B16-old | — | 0.9947 | ExtReal FPR 8.6% @95%TPR | `gs://training-job-outputs/best_checkpoints/corrected/top_n_effort_20260119_step14000_auc0.9947_eer0.0218_B16_LAION.patched.pth` |
| **R1** | Feb 10–11 | P2_C3 (cosine softmax) | `t1zpnv9s` | 0.9844 (ep5) | *(invalidated — data bug)* | `gs://training-job-outputs/best_checkpoints/t1zpnv9s/top_n_effort_20260211_step8000_auc0.9844_eer0.0372.pth` |
| **R2** | Feb 11–12 | R2_A3 (all data, CE) | `8urk1cmw` | 0.9866 | — | `gs://training-job-outputs/phase2r2_experiments/8urk1cmw/top_n_effort_20260212_step10000_auc0.9866_eer0.0534.pth` |
| **R2.5** | Feb 12 | **R25_F1** (k=32) | `5w453our` | **0.9893** | OOD ACC 94.8%, ExtReal FPR 1.8% @95%TPR | `gs://training-job-outputs/phase2r2_experiments/5w453our/top_n_effort_20260212_step18500_auc0.9893_eer0.0356.pth` |
| **R3** | Feb 13 | R3_FT3 (holdout champ) | `kzfu116l` | **0.9966** | *(OOD not validated — F1 still preferred)* | `gs://training-job-outputs/phase2r3_experiments/kzfu116l/top_n_effort_20260213_step2500_auc0.9966_eer0.0191.pth` |
| **R4** | Feb 17 | **R4_FT7** (deploy candidate) | `udgwsu7o` | 0.9935 | WMA 88.35%, ExtReal FPR 4.38%, DeepLive TPR 99.53% | `gs://training-job-outputs/phase2r4_experiments/udgwsu7o/top_n_effort_20260217_step500_auc0.9935_eer0.0088.pth` |
| **R5** | Feb 17–18 | *(FT7 retained)* | — | — | R5 scratch did not displace FT7 | — |
| **R6** | Feb 19–20 | **R6_S1** (VCD reals + aug) | `s3tx3fk4` | 0.9691 (@6K) | VCD real acc 0.698, OOD AUC 0.954 | `gs://training-job-outputs/phase2r6_experiments/best_checkpoint.pth` |
| **R7** | Feb 20–22 | *(subsumed by R8)* | — | — | R7 configs ran but R8 superseded | — |
| **R8** | Feb 23–24 | **R8_E** (scratch, target-heavy) | `hu7cen3m` | **0.9925** | VCD real 82.1%, Viso 97.1%, DeepLive 93.3% | `gs://training-job-outputs/phase2r8_experiments/hu7cen3m/top_n_effort_20260224_step8000_auc0.9925_eer0.0207.pth` |
| **R9** | Feb 28 | **R9_A** (Teams FT + stability*) | `1551zxa8` | 0.9891 | OOD AUC 0.9768, Teams EC 70%, WMA 99.9% | `gs://training-job-outputs/phase2r9_experiments/1551zxa8/` |
| **R9.5** | Mar 2–6 | **No new winner** (R9_A retained) | — | 0.9894 (B, best) | Stability λ hurts OOD monotonically; R95_B best OOD AUC 0.9729 but < R9_A's 0.9768 | — |
| **R10** | Mar 6–7 | **R10_C** (FT R9_A + Teams v2) | `R10_C` | 0.9836 | OOD AUC 0.9502†, VCD real 74.4%, WMA 99.7% | `gs://training-job-outputs/phase2r10_experiments/` |
| **R11** | Mar 7–8 | **R11_G** (ViT-L/14) ⚠️ crashed | `R11_G` | **0.9932** | OOD AUC **0.9873**†, VCD real **92.0%**, FD **92.9%** | *(crashed — checkpoint may be partial)* |
| **R12** | Mar 8–11 | **R12_G** (scratch, seed=737) | `0xxqwhxg` | **0.9917** | OOD AUC 0.9607‡, Composite **0.9738** | `gs://training-job-outputs/phase2r12_experiments/0xxqwhxg/ood_composite_effort_20260310_step12500_auc0.9923_eer0.0253.pth` |
| | | **R12_A** (scratch, OOD pick) | `4v8av986` | 0.9852 | OOD AUC **0.9658**‡, VCD Real **83.9%** | `gs://training-job-outputs/phase2r12_experiments/4v8av986/ood_composite_effort_20260309_step7500_auc0.9705_eer0.0744.pth` |

†R10/R11 OOD = {youtube_avspeech, zoom_vcd_real, wma_failure_fake} — same as R8/R9.  
‡R12 OOD = above + {teams_ood_real, teams_ood_fake} — harder eval, not directly comparable.

---

## Detailed Round-by-Round

### Phase 1 — B16-old (Baseline)

| Field | Value |
|-------|-------|
| **Name** | B16-old (Phase 1 best) |
| **Backbone** | ViT-B-16-DataComp-XL (LAION), hidden_size=512 |
| **SVD rank** | k=8 trainable directions |
| **Loss** | Cross-Entropy |
| **Data** | DF40 + DeepLive (no VisoMaster) |
| **Checkpoint** | `gs://training-job-outputs/best_checkpoints/corrected/top_n_effort_20260119_step14000_auc0.9947_eer0.0218_B16_LAION.patched.pth` |
| **Best step** | 14,000 |
| **Holdout AUC** | 0.9947 |
| **Holdout EER** | 0.0218 |
| **ExtReal FPR @95%TPR** | 8.6% |
| **Status** | Superseded by R25_F1 (same fake TPR, half the FPR) |

---

### Round 1 — P2_C3 cosine_softmax *(Invalidated)*

| Field | Value |
|-------|-------|
| **W&B ID** | `t1zpnv9s` |
| **Config** | Cosine softmax (m=0, s: 10→18), rank=760 (k=8) |
| **Data** | All sources (but config bug = all runs used identical DF40+DL data) |
| **Checkpoint** | `gs://training-job-outputs/best_checkpoints/t1zpnv9s/top_n_effort_20260211_step8000_auc0.9844_eer0.0372.pth` |
| **Best step** | 8,000 |
| **Holdout AUC** | 0.9844 (epoch 5, still training) |
| **Status** | **Invalidated** — config passthrough bug meant all R1 runs trained on identical data. Loss/architecture comparisons were still valid (cosine softmax > CE, more rank helps). |
| **Lesson** | Cosine softmax and rank=752+ were validated as directions for R2. |

---

### Round 2 — R2_A3 (all data, CE)

| Field | Value |
|-------|-------|
| **W&B ID** | `8urk1cmw` |
| **Config** | CE loss, rank=760 (k=8), DF40+DeepLive+VisoMaster |
| **Checkpoint** | `gs://training-job-outputs/phase2r2_experiments/8urk1cmw/top_n_effort_20260212_step10000_auc0.9866_eer0.0534.pth` |
| **Best step/epoch** | 10,000 / epoch 6 |
| **Holdout AUC** | 0.9866 |
| **Holdout EER** | 0.0534 |
| **Key finding** | Data composition (adding DeepLive+VisoMaster) was the biggest single lever (+0.007 AUC vs DF40-only). |
| **Status** | Superseded by R25_F1 in the same cycle. |

---

### Round 2.5 — R25_F1 (k=32 sweet spot) **★ MILESTONE WINNER**

| Field | Value |
|-------|-------|
| **W&B ID** | `5w453our` |
| **Config** | Cosine softmax (m=0, s: 10→18), rank=736 (k=32), lr=2e-4, `base_only` aug |
| **Data** | DF40 + DeepLive + VisoMaster (21,611 videos, 26 methods, NO quality_enhancement) |
| **Checkpoint** | `gs://training-job-outputs/phase2r2_experiments/5w453our/top_n_effort_20260212_step18500_auc0.9893_eer0.0356.pth` |
| **Best step/epoch** | 18,500 / epoch 10 (early-stopped at epoch 13) |
| **Holdout AUC** | **0.9893** |
| **Holdout EER** | 0.0356 |
| **OOD overall ACC** | **94.8%** (9,657 videos) |
| **ExtReal FPR @0.5** | 6.3% |
| **ExtReal FPR @95%TPR** | **1.8%** (vs Phase 1's 8.6% — nearly 5× fewer false alarms) |
| **Fake TPR** | 98.9% |
| **Discriminating case ACC** | 76.1% (best among all models tested) |
| **Operating points** | EER thresh 0.560 → TPR ~96.4%; @FPR5% thresh 0.545 → TPR 98.6% |
| **Status** | Long-standing OOD champion for general deployment. Superseded by FT7 for enhanced-fake-aware deployment. |
| **Why it won** | k=32 SVD capacity was the sweet spot; cosine softmax + `base_only` augmentation prevented quality shortcuts. Beat Phase 1 on equal fake detection with half the FPR. |

---

### Round 3 — R3_FT3 (holdout AUC champion) ⚠️ OOD UNVALIDATED

| Field | Value |
|-------|-------|
| **W&B ID** | `kzfu116l` |
| **Config** | Fine-tuned from R25_F1, `base_only` aug (same as F1), +320 quality_enhancement videos, lr=5e-5 |
| **Data** | 21,931 videos, 27 methods (26 + quality_enhancement) |
| **Checkpoint** | `gs://training-job-outputs/phase2r3_experiments/kzfu116l/top_n_effort_20260213_step2500_auc0.9966_eer0.0191.pth` |
| **Best step/epoch** | 2,500 / epoch 2 |
| **Holdout AUC** | **0.9966** (highest ever recorded) |
| **Holdout EER** | 0.0191 |
| **TPR@FPR1%** | 0.9771 (vs R25_F1's 0.7260 — massive improvement) |
| **OOD performance** | **Never validated on OOD data** |
| **Status** | Holdout champion but never deployed. R3 OOD validation (on scratch models S1/S2 vs F1) showed holdout AUC ≠ OOD performance. R25_F1 won on OOD despite lower holdout AUC. |
| **The paradox** | R3 was the experiment that proved **holdout AUC ≠ deployment performance** — the single most important finding of Phase 2. |

Other notable R3 checkpoints:
| Run | W&B ID | AUC | Checkpoint |
|-----|--------|-----|------------|
| R3_FT1 | `w7wi9lpj` | 0.9962 | `gs://training-job-outputs/phase2r3_experiments/w7wi9lpj/top_n_effort_20260213_step500_auc0.9962_eer0.0191.pth` |
| R3_FT4 | `3k2jzoe8` | 0.9962 | `gs://training-job-outputs/phase2r3_experiments/3k2jzoe8/top_n_effort_20260213_step500_auc0.9962_eer0.0191.pth` |
| R3_FT2 | `3cxpwxgn` | 0.9954 | `gs://training-job-outputs/phase2r3_experiments/3cxpwxgn/top_n_effort_20260213_step500_auc0.9954_eer0.0229.pth` |
| R3_S2 | `qgcp25lr` | 0.9957 | `gs://training-job-outputs/phase2r3_experiments/qgcp25lr/top_n_effort_20260213_step13000_auc0.9957_eer0.0229.pth` |
| R3_S1 | `zjftx8ny` | 0.9933 | `gs://training-job-outputs/phase2r3_experiments/zjftx8ny/top_n_effort_20260213_step7500_auc0.9933_eer0.0382.pth` |

---

### Round 4 — R4_FT7 (deployment candidate) **★ MILESTONE WINNER**

| Field | Value |
|-------|-------|
| **W&B ID** | `udgwsu7o` |
| **Config** | Fine-tuned from R25_F1, `quality_targeted_family` light aug, with enhanced DeepLive strategies (860-set), `identity_resample_weighted` with DeepLive-upweighted families |
| **Checkpoint** | `gs://training-job-outputs/phase2r4_experiments/udgwsu7o/top_n_effort_20260217_step500_auc0.9935_eer0.0088.pth` |
| **Best step** | 500 |
| **Holdout AUC** | 0.9935 |
| **Holdout EER** | 0.0088 |
| **DF40 fake TPR** | 86.85% |
| **Enhanced DeepLive TPR** | **99.53%** |
| **WMA flat per-image ACC** | **88.35%** (1,062/1,202) |
| **ExtReal FPR** | 4.38% (passes ≤8% gate) |
| **Overall sidecar AUC** | 0.9941 |
| **YouTube real ACC** | ~96% |
| **VCD real ACC** | ~51–60% (the remaining weakness) |
| **Status** | Deployment candidate through R5 and into R6. Best on enhanced-fake and WMA gates. |
| **Why it won** | Family-aware augmentation + enhanced DeepLive data + weighted sampling solved the enhanced face-swap failure. Maintained external-real FPR under gate. |

R4 runner-up:
| Run | W&B ID | WMA ACC | ExtReal FPR | Enhanced DL TPR | Checkpoint |
|-----|--------|---------|-------------|-----------------|------------|
| FT5 (rollback) | `tu0zeofr` | 74.29% | 3.35% | 98.38% | `gs://training-job-outputs/phase2r4_experiments/tu0zeofr/top_n_effort_20260217_step500_auc0.9920_eer0.0122.pth` |

---

### Round 5 — No new winner (FT7 retained)

| Field | Value |
|-------|-------|
| **Runs launched** | 6 scratch runs (S1/S2/S3 × seeds 737/1337) |
| **Purpose** | Scratch retraining cycle to try to beat FT7 with cleaner data mix and sampling |
| **Outcome** | FT7 remained the provisional deployment candidate. R5 scratch runs did not displace FT7 on the canonical gates (WMA flat ≥50%, ExtReal FPR ≤8%). |
| **Status** | Transitioned to R6, which shifted focus to VCD real robustness. |

---

### Round 6 — R6_S1 (VCD reals + aug) **★ VCD ROBUSTNESS WINNER**

| Field | Value |
|-------|-------|
| **W&B ID** | `s3tx3fk4` |
| **Config** | Scratch, VCD reals in training (800 samples, 20% identity split), `vcd_targeted` augmentation, no GRL |
| **Checkpoint** | `gs://training-job-outputs/phase2r6_experiments/best_checkpoint.pth` |
| **Holdout AUC** | 0.9691 (@6K steps) |
| **VCD Real ACC** | **0.698** (up from ~0.55 baseline — **+14.8pp improvement**) |
| **YouTube Real ACC** | 0.853 |
| **OOD AUC** | 0.954 |
| **In-dist AUC** | 0.9911 |
| **WMA fake ACC** | ~95–96% |
| **Status** | Winner of R6. Chosen as base for R7 fine-tuning. |
| **Why it won** | Simplest effective config — VCD reals + targeted augmentation without the (buggy) GRL head matched the full-stack S3 while being cleaner. S1 ≈ S3 ≈ S6 ≈ S7 (Tier 1), but S1 was simplest. |

Key R6 findings:
- VCD real accuracy improved from ~55% (R5 baseline) to ~70% (R6 Tier 1 runs)
- GRL was broken throughout R6 (quality_domain_loss = 0 due to label propagation bug)
- Plateau/rollback observed at ~7K steps (some runs peaked at ~5K then regressed)
- S8 (more VCD identities) counterintuitively performed worse
- R6 Tier 1 runs (S1/S3/S6/S7) were all roughly equivalent

R6 reference runs (Tier 1 — all equivalent):
| Run | W&B ID | VCD Real ACC | YouTube ACC | Holdout AUC |
|-----|--------|-------------|-------------|-------------|
| S1 (VCD+aug) | `s3tx3fk4` | 0.698 | 0.853 | 0.9691 |
| S3 (full stack GRL) | `iwk4pe1j` | 0.698 | 0.853 | 0.9694 |
| S6 (high VCD weight) | `rlyym6yn` | 0.701 | 0.853 | 0.9692 |
| S7 (strong GRL) | `z54lwhda` | 0.700 | 0.853 | 0.9689 |

---

### Round 7 — Subsumed by R8

| Field | Value |
|-------|-------|
| **Status** | R7 configs ran but were superseded by R8's distribution inversion approach |
| **Focus** | Context variation + faceswap exclusion |
| **Outcome** | R7 baseline results (VisoMaster 69%, VCD real ~70%) became the starting point for R8's redesign |

---

### Round 8 — R8_E (scratch, target-heavy) **★ DEPLOYMENT-TARGET BREAKTHROUGH**

| Field | Value |
|-------|-------|
| **W&B ID** | `hu7cen3m` |
| **Config** | Scratch from CLIP weights, target-heavy distribution (DF40 weight 0.2, Viso 3.0, DeepLive 4.0/5.0), all 9 VisoMaster swap models, 40% VCD identities in training, ArcFace s: 10→18, `quality_targeted_family` + `vcd_targeted` aug |
| **Data** | DF40 (downweighted) + DeepLive + all 9 VisoMaster swap models + VCD reals (40% identities) |
| **Checkpoint** | `gs://training-job-outputs/phase2r8_experiments/hu7cen3m/top_n_effort_20260224_step8000_auc0.9925_eer0.0207.pth` |
| **Best step** | 8,000 / 12K |
| **LR** | 2e-4 (scratch schedule) |
| **Holdout AUC** | **0.9925** |
| **Holdout EER** | 0.0207 |
| **VCD Real ACC** | **82.1%** (best ever — up from 70% in R6, target 85%) |
| **YouTube Real ACC** | 85.8% |
| **VisoMaster overall** | **97.1%** (up from 69% in R7 — **SOLVED**) |
| **DeepLiveCam avg** | 93.3% (gate ≥95% — close) |
| **WMA Fake ACC** | 98.5% |
| **Status** | R8 champion. Used as base checkpoint for all R9 fine-tuning runs. |
| **Why it won** | Inverted the training distribution: massive upweight of deployment-target data (DeepLive, VisoMaster) while downweighting DF40 academic data. Scratch training avoided inherited biases from prior checkpoints. All 9 VisoMaster swap models + identity-based splitting solved the VisoMaster problem entirely. |
| **Known issues** | (1) Threshold calibration gap: val EER thresh ~0.45 vs OOD EER thresh ~0.77. (2) Seed sensitivity: R8_H (seed 1337) got only 69.4% VCD Real vs R8_E's 82.1%. (3) DeepLive edge_cases (non-enhanced) at 76.5% — weakest sub-category. |

R8 leaderboard (all 8 runs):

| Rank | Run | Track | AUC | EER | VCD Real | Viso Overall | DeepLive Avg |
|------|-----|-------|-----|-----|----------|--------------|--------------|
| **1** | **R8_E** | Scratch | 0.9912* | 0.0325* | **82.1%** | 97.1% | 93.3% |
| 2 | R8_C | FT | 0.9921 | 0.0355 | 77.8% | 96.6% | 94.0% |
| 3 | R8_A | FT | 0.9921 | 0.0355 | 77.4% | 96.6% | 95.6% |
| 4 | R8_D | FT | 0.9892 | 0.0419 | 73.4% | 97.6% | 94.3% |
| 5 | R8_F | Scratch | 0.9870 | 0.0419 | 72.8% | 96.0% | 95.0% |
| 6 | R8_G | Scratch | 0.9925 | 0.0296 | 71.2% | 95.2% | 97.2% |
| 7 | R8_H | Scratch | 0.9886 | 0.0297 | 69.4% | 96.5% | 85.3% |
| 8 | R8_B | FT | 0.9936 | 0.0197 | 69.2% | 96.5% | 97.6% |

*Mid-run metrics at step 7109. Final checkpoint at step 8000: AUC 0.9925, EER 0.0207.

Key R8 ablation insights:
1. **DF40-zero hurts** — R8_B/R8_G (no DF40) had worst VCD Real. DF40 provides useful negative anchors.
2. **Smart tiers don't help** — Excluding STRONG tier (R8_D/R8_F) performed worse. Keep all tiers.
3. **Scratch beats fine-tune on VCD Real** — R8_E (scratch) > R8_A (FT) by 4.7pp on VCD Real.
4. **Seed sensitivity** — R8_E vs R8_H: 82.1% vs 69.4% VCD Real with different seeds.

---

## The Arc — How Each Winner Led to the Next

```
Phase 1 (B16-old, k=8, CE, AUC 0.9947)
    │
    │ Problem: 8.6% FPR on external reals
    ▼
R1 (invalidated by config bug, but validated cosine softmax + higher rank)
    │
    ▼
R2 (controlled ablation: data composition >> loss >> capacity)
    │
    ▼
R2.5 ★ R25_F1 (k=32 sweet spot, AUC 0.9893, FPR cut to 1.8%)
    │
    │ Problem: WMA enhanced fakes = 21% detection
    ▼
R3 (holdout AUC 0.9966 but OOD WORSE → proved holdout ≠ deployment)
    │
    │ Lesson: quality-robust aug hurts OOD generalization
    ▼
R4 ★ FT7 (enhanced DeepLive + family aug → WMA 88.35%, DL TPR 99.53%)
    │
    │ Problem: VCD real accuracy ~51-60%
    ▼
R5 (scratch retraining, FT7 retained as best)
    │
    ▼
R6 ★ S1 (VCD reals + targeted aug → VCD real 70%, +15pp improvement)
    │
    │ Problem: VCD still at 70% (target ≥85%), faceswap weak, GRL was broken
    ▼
R7 (context variation + faceswap exclusion — subsumed by R8)
    │
    │ Problem: VisoMaster still 69%, VCD real still 70%
    ▼
R8 ★ R8_E (scratch, target-heavy → Viso 97%, VCD real 82%, AUC 0.9925)
    │
    │ Problem: VCD real 82.1% (target 85%), threshold calibration gap,
    │          score instability, no Teams codec in training
    ▼
R9 ★ R9_A (Teams FT from R8_E → Teams 70-100%, OOD AUC 0.9768, unified EER 3.52%)
    │
    │ ⚠️ BUG: stability_lambda & label_smoothing never reached Trainer
    │ Problem: facedancer 68%, VCD real 80%, score stability untested
    ▼
R9.5 (stability fix + λ sweep — STABILITY HURTS OOD, R9_A RETAINED)
    │
    │ Finding: λ=0 > λ=0.1 > λ=0.3 > λ=0.5 on OOD AUC (monotonic)
    │ Finding: facedancer is architectural ceiling (5× DF40 weight = zero effect)
    │ Finding: R95_D scratch still improving at EWI=1 — needs 20K steps
    ▼
R10 ★ R10_C (Teams v2 + VisoMaster expansion, FT from R9_A → AUC 0.9836, OOD AUC 0.9502)
    │
    │ Problem: VCD real regressed to 74.4%; facedancer still 63.6% FT / 68.2% scratch
    │ Signal: R10_D (ViT-L-14) crashed but had OOD AUC 0.9709 — L-14 has potential
    ▼
R11 ★ R11_G (ViT-L-14 BREAKOUT → AUC 0.9932, OOD 0.9873, facedancer 92.9%) ⚠️ CRASHED
    │
    │ Finding: L-14 solves facedancer (92.9% vs 63.6% B-16 FT). Capacity was the bottleneck.
    │ Finding: Group DRO (R11_B) no benefit. Scratch needs 20K+ (R11_E: 71.4% FD, climbing).
    │ Problem: R11_G checkpoint may be unusable — need stable L-14 rerun.
    │ Problem: All B-16 FT runs converged to same ceiling (~0.9827 AUC).
    ▼
R12 ★ R12_G (scratch seed=737 → Composite 0.9738, AUC 0.9917, FD 100%) + R12_A (OOD pick)
    │
    │ Finding: Seed=737 (R12_G) came from behind — worst OOD at epoch 3 (0.9126) → tied-best by epoch 5 (0.9607).
    │ Finding: GRL 0.1 (D/F) provided tiny OOD gain; GRL 0.3 (H) actively hurt. GRL verdict: not worth the complexity.
    │ Finding: k=64 (R12_B) = no benefit over k=32. FT runs (C/D/F) hit ceiling fast.
    │ Finding: R12_A has best raw OOD AUC (0.9658) and VCD Real (83.9%) — the OOD/real-data champion.
    │ Finding: After epoch 3, holdout AUC kept climbing but OOD eroded for A/B — classic in-dist overfit signal.
    │ Status: 4 finished/stopped (C,D,F,H), 4 ran to epoch 5 (A,B,E,G). Best composite checkpoints already banked.
    │ Signal: GRL 0.1 helps OOD (+0.5pp), GRL 0.3 hurts. Seed sensitivity confirmed.
    │ ⚠️ OOD eval now includes Teams OOD — numbers not comparable to R8-R11.
    ▼
R12.5 🔄 (lighting augmentation ablation — 3 runs, targeting production brightness gap)
    │
    │ (see R12.5 section below for full details)
    ▼
R13 (TBD — stable ViT-L-14 run, GRL tuning, possibly combine L-14 + GRL)
```

---

### Round 9 — R9_A (Teams FT + stability*) **★ TEAMS DOMAIN WINNER** ⚠️ STABILITY BUG

| Field | Value |
|-------|-------|
| **W&B ID** | `1551zxa8` |
| **Config** | Fine-tuned from R8_E, Teams passthrough data (278 videos), ArcFace s: 6→12, stability_lambda=0.3* (see bug note), label_smoothing=0.05* |
| **Data** | DF40 + DeepLive + all 9 VisoMaster swap models + Teams (3 strategies) + VCD reals (40% identities) — 12,668 videos, 25 methods |
| **Checkpoint** | `gs://training-job-outputs/phase2r9_experiments/1551zxa8/` |
| **Best step/epoch** | ~8,000 / epoch 3 |
| **LR** | 5e-5 (fine-tune schedule) |
| **Holdout AUC** | 0.9891 |
| **Holdout EER** | 0.0457 |
| **Unified AUC** | **0.9897** |
| **Unified EER** | **0.0352** |
| **OOD AUC** | 0.9768 |
| **OOD EER** | 0.0655 |
| **Teams edge_cases (holdout)** | 70.0% (~6 videos) |
| **Teams minimal_proc (holdout)** | 91.7% (~5 videos) |
| **Teams quality_enh (holdout)** | 100% (~3 videos) |
| **facedancer (holdout)** | 68.2% (worst method — universal across all runs) |
| **WMA fake (OOD)** | **99.9%** |
| **VCD real (OOD)** | 80.3% (regressed from R8_E's 82.1%) |
| **YouTube real (OOD)** | 86.8% |
| **Score jitter (YouTube)** | 0.0389 |
| **Status** | Teams domain winner. Stability was NOT active due to config bug. |
| **Why it won** | Best balance of Teams detection + OOD generalization + low unified EER. Tighter worst-case floor than R9_H (facedancer 68.2% vs 63.6%, e4s 87.5% vs 75%). |

**⚠️ Critical Bug — Stability Never Activated:**

All 8 R9 runs had `train/loss/stability = 0` because `stability_lambda` and `label_smoothing` (flat YAML keys) were never propagated through the config pipeline to the Trainer. The pipeline only handled nested keys. This means:
- R9_A through R9_H all trained **without stability regularization** (λ=0)
- R9_A through R9_H all trained **without label smoothing** (LS=0)
- R9_E (designed as "heavy reg" λ=0.5, LS=0.1) was identical to R9_B (designed as "no fixes" λ=0, LS=0)

**Fix applied (ready for Docker 1.3.145):**
1. `config_helpers.py`: Added `apply_wandb_stability_params()` function for `stability_lambda`, `stability_noise_std`, `stability_crop_jitter`
2. `config_helpers.py`: Added `label_smoothing` to `apply_wandb_loss_params()`
3. `config_helpers.py`: Wired into `apply_all_wandb_overrides()`
4. `train_sweep.py`: Added flat-key passthrough in `single_cfg` direct-apply block

**Additional fix — Unified threshold metrics:**
- `metrics/utils.py`: Added `metrics_at_threshold()` function
- `trainer/trainer.py`: Extracts val_in_dist EER threshold, applies to holdout → `val_holdout/at_indist/*`, applies to OOD → `ood/at_indist/*`

R9 leaderboard (all 8 runs):

| Rank | Run | Track | Holdout AUC | Unified AUC | OOD AUC | Teams EC | facedancer |
|------|-----|-------|-------------|-------------|---------|----------|------------|
| **1** | **R9_A** | FT+Teams | 0.9891 | **0.9897** | 0.9768 | 70% | **68.2%** |
| 2 | R9_H | FT+Teams (high LR) | 0.9887 | 0.9895 | **0.9773** | 70% | 63.6% |
| 3 | R9_C | Scratch+Teams | 0.9865 | — | **0.9801** | 75% | — |
| 4 | R9_D | FT no Teams | **0.9942** | — | 0.9673 | — | — |
| 5 | R9_E | FT+Teams (heavy reg*) | 0.9890 | — | 0.9715 | — | — |
| 6 | R9_B | FT+Teams (no fixes*) | 0.9888 | — | 0.9715 | — | — |
| 7 | R9_G | FT+Teams+CodecSim | 0.9883 | — | 0.9678 | 75% | — |
| 8 | R9_F | FT+CodecSim (no Teams) | 0.9926 | — | 0.9632 | — | — |

*E and B were functionally identical due to stability bug.

Key R9 findings:
1. **Teams data helps OOD by +1-2pp AUC** — R9_A (with Teams) 0.9768 vs R9_D (no Teams) 0.9673.
2. **Scratch still beats FT on OOD** — R9_C (scratch) had best OOD AUC at 0.9801, consistent with R8 finding.
3. **facedancer is universally weakest** at 63-68% across all runs. The DF40 weight (0.2) is too low.
4. **Teams edge_cases is hardest** Teams strategy at 70-82%. Only ~6 holdout videos — noisy signal.
5. **VCD real regressed** from 82.1% (R8_E) to ~80% — adding Teams data slightly diluted real discrimination.
6. **Threshold calibration gap persists** — val EER threshold ~0.46 vs OOD EER threshold ~0.94.

---

### Round 9.5 — Stability Bug Fix Rerun (No new winner — R9_A retained) ⚠️ STABILITY HURTS OOD

| Field | Value |
|-------|-------|
| **Purpose** | Rerun R9 with stability config fix + 6 ablations (λ sweep, ArcFace, DF40 weight, scratch) |
| **Docker image** | 1.3.145 (stability config fix + unified threshold metrics) |
| **Runs** | 6 (R95_A through R95_F) |
| **Data** | Same as R9 (DF40 + DeepLive + VisoMaster + Teams + VCD reals) |
| **All runs best epoch** | 3 (FT) or 4 (scratch R95_D) |
| **Stability fix confirmed** | ✅ All runs have stability_loss > 0 (R9 had 0.0 everywhere) |

**Central finding:** Stability regularization **monotonically degrades OOD generalization**:

| λ | Label Smooth | Run | OOD AUC | VCD Real | YT Jitter |
|---|---|---|---|---|---|
| **0.0** | 0.0 | R9_A | **0.9768** | **80.3%** | **0.0389** |
| 0.1 | 0.0 | R95_B | 0.9729 | 76.9% | 0.0407 |
| 0.3 | 0.05 | R95_A | 0.9661 | 75.4% | 0.0438 |
| 0.5 | 0.1 | R95_C | 0.9556 | 74.7% | 0.0453 |

R9.5 leaderboard:

| Rank | Run | W&B ID | Best AUC | OOD AUC | VCD Real | facedancer | Teams EC |
|------|-----|--------|----------|---------|----------|------------|----------|
| 1 | R95_B (light λ) | `95p2wvqi` | 0.9894 | **0.9729** | **76.9%** | 59.1% | 70% |
| 2 | R95_A (baseline) | `lxfzu0di` | **0.9900** | 0.9661 | 75.4% | 59.1% | 70% |
| 3 | R95_F (DF40 5×) | `n9c4ec3i` | **0.9900** | 0.9662 | 75.5% | 59.1% | 70% |
| 4 | R95_E (ArcFace) | `pglgtxlc` | 0.9895 | 0.9585 | 75.5% | 59.1% | 70% |
| 5 | R95_C (heavy λ) | `ykhi5ypm` | 0.9894 | 0.9556 | 74.7% | 59.1% | 70% |
| 6 | R95_D (scratch) | `kt4z2qww` | 0.9876 | 0.9280 | 61.6% | **68.2%** | **80%** |

Key R9.5 findings:
1. **All FT runs identical on DF40 holdout** — facedancer 59.1%, e4s 62.5% etc. across A/B/C/E/F. Different λ, LS, ArcFace scale, DF40 weight all converged to the same decision boundary.
2. **R95_F (5× DF40 weight) ≡ R95_A** — zero effect on any metric. facedancer is an architectural ceiling, not a data problem.
3. **R95_D (scratch) preserved facedancer (68.2%)** and had best Teams EC (80%), but EWI=1 — still improving at epoch 4. Needed 20K+ steps.
4. **All FT runs peaked at epoch 3** with EWI 3-6. Early stopping at 7K steps would be optimal.
5. **R9_A remains champion** — its accidental λ=0 was the optimal config. No R9.5 run improved on R9_A's OOD AUC.

---

### Round 10 — Teams v2 + VisoMaster Expansion (R10_C wins, marginal)

| Field | Value |
|-------|-------|
| **Purpose** | First round with Teams v2 codec-degraded data + expanded VisoMaster swap sources; test FT vs scratch and ViT-L/14 |
| **Docker image** | 1.3.150+ |
| **Runs** | 17 total (7 with metrics, 10 failed retries of R10_C/R10_G) |
| **Data** | DF40 + DeepLive + VisoMaster (expanded) + Teams v2 + VCD reals |
| **Base checkpoint (FT)** | R9_A (`1551zxa8`) |
| **Config** | `experiments/phase2_round10/` — no CCT/GRL, standard augmentation |

**R10 Leaderboard (B-16 runs):**

| Rank | Run | Type | Best AUC | EER | OOD AUC | VCD Real | facedancer | WMA |
|------|-----|------|----------|-----|---------|----------|------------|-----|
| 1 | **R10_C** | FT (cheapest) | **0.9836** | 0.055 | **0.9502** | 74.4% | 64.3% | 99.7% |
| 2 | R10_G | FT (wide aug) | 0.9829 | 0.059 | 0.9547 | 74.8% | 64.3% | 99.7% |
| 3 | R10_A | Scratch (wide aug) | 0.9807 | 0.053 | 0.9388 | 65.6% | 78.6% | 99.9% |
| 4 | R10_E | Scratch (low s) | 0.9804 | 0.057 | 0.9445 | 75.8% | 57.1% | 98.9% |
| 5 | R10_B | Scratch (mixup) | 0.9803 | 0.059 | 0.9427 | 75.9% | 57.1% | 98.5% |
| 6 | R10_F | Scratch (narrow) | 0.9823 | 0.059 | 0.9253 | 68.8% | 64.3% | 99.7% |

**ViT-L/14 (R10_D — crashed early):**

| Run | Backbone | AUC | OOD AUC | VCD Real | facedancer | Note |
|-----|----------|-----|---------|----------|------------|------|
| R10_D | ViT-L-14 | 0.9042 | **0.9709** | 85.3% | 45.5% | Crashed ~epoch 2. Best OOD in round but in-dist only 0.90 |

Key R10 findings:
1. **Teams v2 data integrated successfully** — no regression from R9_A baseline.
2. **R10_C (cheapest FT) best in-dist AUC** — minimal config, just FT from R9_A. But R10_G had better OOD AUC (0.9547 vs 0.9502). Simplest approach still competitive.
3. **VCD Real regressed further** — 82.1% (R8_E) → 80.3% (R9_A) → 74.4% (R10_C). More fake diversity dilutes real discrimination.
4. **ViT-L/14 flagged first promise** — R10_D had OOD AUC 0.9709 (best in round) despite crashing early and only 0.90 in-dist. L-14 backbone warranted dedicated exploration.
5. **10 of 17 runs were failed retries** — infrastructure instability, GCS mount issues. R10_C and R10_G each had 5 retry attempts.
6. **Scratch facedancer varied wildly** — R10_A had 78.6% while R10_E had 57.1%, suggesting augmentation matters more than previously thought for facedancer detection.

---

### Round 11 — ViT-L/14 Breakout + Group DRO Test (R11_G dominates, then crashes)

| Field | Value |
|-------|-------|
| **Purpose** | Dedicated ViT-L/14 test (R11_G); Group DRO fairness (R11_B); scratch ceiling test at 30K steps (R11_E); FT variations |
| **Docker image** | 1.3.155+ |
| **Runs** | 8 (R11_A through R11_H — R11_H failed) |
| **Data** | Same as R10 (DF40 + DeepLive + VisoMaster + Teams v2 + VCD reals) |
| **Base checkpoint (FT)** | R8_E (`hu7cen3m`) for B-16 runs; scratch for R11_G (L-14) and R11_E |
| **Config** | `experiments/phase2_round11/` — includes Group DRO (R11_B), ViT-L-14 (R11_G), 30K max steps (R11_E) |

**R11 Leaderboard:**

| Rank | Run | Backbone | Type | Best AUC | EER | OOD AUC | VCD Real | facedancer | WMA |
|------|-----|----------|------|----------|-----|---------|----------|------------|-----|
| 1 | **R11_G** ⚠️ | **ViT-L-14** | Scratch | **0.9932** | **0.011** | **0.9873** | **92.0%** | **92.9%** | 95.9% |
| 2 | R11_H | B-16 | FT R8_E (ultra-low LR) | 0.9815 | 0.059 | 0.9566 | **76.1%** | 63.6% | 99.8% |
| 3 | R11_C | B-16 | FT R8_E (wide) | 0.9821 | 0.061 | 0.9565 | 75.6% | 63.6% | 99.8% |
| 4 | R11_F | B-16 | FT R8_E (high Teams wt) | 0.9828 | 0.057 | 0.9527 | 74.8% | 63.6% | 99.7% |
| 5 | R11_B | B-16 | FT + Group DRO | 0.9827 | 0.057 | 0.9527 | 75.2% | 63.6% | 99.7% |
| 6 | R11_A | B-16 | FT R8_E | 0.9827 | 0.057 | 0.9526 | 74.8% | 63.6% | 99.7% |
| 7 | R11_D | B-16 | FT R8_E (narrow) | **0.9829** | 0.061 | 0.9449 | 73.8% | 63.6% | 99.7% |
| 8 | R11_E | B-16 | Scratch (30K) | 0.9803 | 0.059 | 0.9353 | 82.2% | **71.4%** | 91.0% |

⚠️ **R11_G crashed** after achieving these metrics. Checkpoint may be partial/unusable.

Key R11 findings:
1. **ViT-L/14 is a game-changer** — R11_G obliterated every B-16 run on every metric: AUC +1pp, OOD AUC +3pp, VCD Real +16pp, facedancer **92.9%** (vs 64.3% for B-16 FT, vs 71.4% for B-16 scratch). The larger backbone has enough capacity to solve facedancer.
2. **R11_G crashed** — this is devastating. The best model we've ever trained may not have a usable checkpoint. L-14 needs a dedicated stable run.
3. **Group DRO (R11_B) no benefit** — OOD AUC 0.9527 identical to R11_F (0.9527) and R11_A (0.9526). Fairness regularization may need different tuning or more epochs.
4. **All B-16 FT runs converged tightly** — R11_A/B/C/D/F/H span only 0.12pp on OOD AUC (0.9449–0.9566). Ultra-low LR (R11_H) and wide aug (R11_C) surprisingly had best B-16 OOD.
5. **Scratch ceiling test (R11_E, 30K)** — facedancer improved to 71.4% (vs 64.3% FT) and VCD Real hit 82.2% (best among B-16 runs). But WMA dropped to 91.0% and OOD AUC only 0.9353. Scratch needs more steps and benefits different metrics.
6. **B-16 FT from R8_E ≈ FT from R9_A** — R10 (FT R9_A) and R11 (FT R8_E) produced nearly identical B-16 results, confirming checkpoint source is not a major factor for FT.

---

### Round 12 — Compound Augmentation + GRL Quality Domain Head (🔄 IN PROGRESS)

| Field | Value |
|-------|-------|
| **Purpose** | Test compound augmentation (CCT color temp, asymmetric brightness, hue shift) + GRL quality domain head for codec-invariant features + OOD composite checkpointing |
| **Docker image** | 1.3.160+ |
| **Runs** | 8 (R12_A through R12_H — 4 ran to epoch 5, 1 finished early, 3 crashed/stopped) |
| **Data** | DF40 + DeepLive + VisoMaster + Teams v2 + VCD reals (same training as R10/R11) |
| **OOD eval (NEW)** | {youtube_avspeech, zoom_vcd_real, wma_failure_fake} **+ teams_ood_real (300) + teams_ood_fake (300)** |
| **Base checkpoint (FT)** | R8_E (`hu7cen3m`) for R12_C/D/F; scratch for R12_A/B/E/G/H |
| **Config** | `experiments/phase2_round12/` |

**New R12 features:**
- **GRL (Gradient Reversal Layer) quality domain head** — adversarial training to learn codec-invariant features. Weights tested: 0.0 (R12_C), 0.1 (R12_D/F), 0.3 (R12_H)
- **Compound augmentation** — CCT color temperature shift, asymmetric brightness [-0.2, 0.6], hue_shift=20°
- **OOD composite checkpointing** — checkpoint selection based on composite of in-dist AUC + OOD AUC (not just in-dist)
- **Teams OOD monitoring** — teams_ood_real and teams_ood_fake added to eval suite as OOD sources

**R12 Epoch 3 (~8.5K steps, ~19h) — Initial Snapshot:**

| Rank | Run | Type | GRL | AUC | OOD AUC | Composite | Note |
|------|-----|------|-----|-----|---------|-----------|------|
| 1 | R12_F | FT R8_E | 0.1 | 0.9832 | 0.9450 | 0.9672 | Best composite at ep3 |
| 2 | R12_D | FT R8_E | 0.1 | 0.9832 | 0.9467 | 0.9672 | Identical to F |
| 3 | R12_C | FT R8_E | 0.0 | 0.9832 | 0.9411 | 0.9645 | FT baseline |
| 4 | R12_E | Scratch | 0.1 | 0.9819 | 0.9614 | 0.9693 | GRL+Teams OOD |
| 5 | R12_A | Scratch | 0.0 | 0.9832 | 0.9658 | 0.9681 | Best raw OOD |
| 6 | R12_B | Scratch | 0.0 | 0.9830 | 0.9656 | 0.9730 | k=64 |
| 7 | R12_H | Scratch | 0.3 | 0.9804 | 0.9604 | 0.9687 | Strong GRL |
| 8 | R12_G | Scratch | 0.0 | 0.9850 | 0.9440 | 0.9712 | seed=737, worst OOD at ep3 |

**R12 Final (~12.5K steps, ~35h) — Updated March 11:**

| Rank | Run | State | Ep | H.AUC | OOD AUC | Composite | VCD Real | FD | EWI | Key Note |
|------|-----|-------|-----|-------|---------|-----------|----------|-----|-----|----------|
| 1 | **R12_G** | ran to ep5 | 4 | **0.9917** | 0.9607 | **0.9738** | 77.8% | **100%** | 0 | ★ New composite leader. Surged from worst OOD at ep3 to tied-best |
| 2 | **R12_B** | ran to ep5 | 5 | 0.9872 | 0.9607 | 0.9730 | 82.5% | 78.6% | 0 | k=64, VCD holding |
| 3 | R12_E | ran to ep4 | 4 | 0.9819 | 0.9467 | 0.9693 | 78.0% | 50.0% | **11** | GRL+Teams OOD — stalled, EWI=11 |
| 4 | R12_H | crashed ep3 | 3 | 0.9804 | 0.9464 | 0.9687 | 70.9% | 71.4% | 5 | GRL 0.3 — stopped |
| 5 | **R12_A** | ran to ep5 | 5 | 0.9852 | 0.9547 | 0.9681 | **79.8%** | 85.7% | 0 | ★ OOD pick (best checkpoint at step 7500: OOD 0.9658, VCD 83.9%) |
| 6 | R12_D | crashed ep3 | 3 | 0.9832 | 0.9467 | 0.9672 | 73.8% | 64.3% | 4 | FT+GRL 0.1 — stopped |
| 7 | R12_F | crashed ep3 | 3 | 0.9832 | 0.9450 | 0.9672 | 73.5% | 64.3% | 4 | FT+GRL 0.1 — crashed |
| 8 | R12_C | finished ep3 | 3 | 0.9832 | 0.9411 | 0.9645 | 72.9% | 64.3% | 10 | FT baseline — early stopped |

**Best OOD Composite Checkpoints (final):**

| Run | Composite | H.AUC | OOD AUC | Step | GCS Path |
|-----|-----------|-------|---------|------|----------|
| **R12_G** | **0.9738** | 0.9923 | 0.9607 | 12,500 | `0xxqwhxg/ood_composite_effort_20260310_step12500_auc0.9923_eer0.0253.pth` |
| R12_B | 0.9730 | 0.9805 | 0.9656 | 7,500 | `wupk4909/ood_composite_...step7500...pth` |
| R12_E | 0.9693 | 0.9774 | 0.9614 | 7,500 | `ys8z1div/ood_composite_...step7500...pth` |
| **R12_A** | 0.9681 | 0.9705 | **0.9658** | 7,500 | `4v8av986/ood_composite_effort_20260309_step7500_auc0.9705_eer0.0744.pth` |

> All paths under `gs://training-job-outputs/phase2r12_experiments/`

> **⚠️ R12 OOD numbers include Teams OOD sources** — not directly comparable to R8-R11 OOD AUC values. See comparability note at top of document.

Key R12 observations (final, epochs 3–5):
1. **R12_G is the surprise winner.** Worst OOD at epoch 3 (0.9440) → tied-best by epoch 5 (0.9607). Composite 0.9738 = best in R12. 100% FaceDancer. The seed=737 run needed longer to converge but found a better basin.
2. **R12_A is the OOD/real-data champion.** Best raw OOD AUC 0.9658 and VCD Real 83.9% (both at step 7,500 checkpoint). After step 7,500, holdout kept climbing but OOD eroded — classic in-dist overfit after the sweet spot.
3. **GRL verdict: not worth it.** GRL 0.1 (D/F) gave marginal gains at epoch 3 but runs crashed/stalled. GRL 0.3 (H) actively hurt. R12_E (GRL+Teams OOD signal) stalled at EWI=11. The no-GRL scratch runs (A, B, G) are the clear winners.
4. **k=64 (R12_B) no benefit over k=32.** B finished with composite 0.9730 vs G's 0.9738 (k=32). Extra capacity is wasted.
5. **FT from R8_E hit ceiling immediately.** C/D/F all converged to 0.9832 holdout AUC by epoch 3 and never improved. Scratch runs overtook them by epoch 5.
6. **After epoch 3, holdout improves but OOD erodes.** A: OOD -0.0111, VCD -4.1pp. B: OOD -0.0049. Only G improved both holdout AND composite after epoch 3.
7. **OOD source breakdown:** YT Real ~87-88%, WMA Fake 95-99%, Teams OOD ~94-97%. VCD Real remains the hardest source (71-83%).

---

### Round 12.5 — Lighting Augmentation Ablation

| Field | Value |
|-------|-------|
| **Purpose** | Close the production brightness gap identified in the LIGHTING_ROBUSTNESS_REPORT. Training mean brightness ~104, production reaches 160–205. Test whether targeted lighting augmentation (GammaUp, DirectionalShadow, wider CCT) improves OOD without hurting in-dist. |
| **Motivation** | The lighting report (`docs/LIGHTING_ROBUSTNESS_REPORT.md`) showed that real-world captures fall **entirely outside** the training brightness distribution. Only 45% of production captures overlap with training's brightness range. Current augmentation shifts images *down* more than *up*. Two new transforms (GammaUp, DirectionalShadow) were implemented in `data/augmentations/transforms.py` but left OFF (p=0.0) in R12 to establish a clean baseline. R12.5 turns them on in a controlled ablation. |
| **Base config** | R12_A (`experiments/phase2_round12/R12_A_scratch_aug_fix.yaml`) — scratch, k=32, seed=737, no GRL |
| **Docker image** | Same as R12 (1.3.160+) |
| **Data** | Same as R12 (DF40 + DeepLive + VisoMaster + Teams v2 + VCD reals) |
| **OOD eval** | Same as R12 (5-source: youtube_avspeech, zoom_vcd_real, wma_failure_fake, teams_ood_real, teams_ood_fake) |
| **Configs** | `experiments/phase2_round12_5/` (3 YAML files) |
| **Checkpoints** | `gs://training-job-outputs/phase2r125_experiments/` |
| **W&B project** | `dtect-vision/phase2r125-experiments` (or similar — set at launch) |

**The 3 Runs (clean ablation chain):**

| Run | Config File | Delta from R12_A | What it tests |
|-----|-------------|-----------------|---------------|
| **R125_A** | `R125_A_gamma_up.yaml` | `gamma_up_p: 0.15`, `gamma_up_range: [0.45, 0.85]` | GammaUp isolation — always-brighten gamma (γ<1) to push training images toward production brightness range |
| **R125_B** | `R125_B_gamma_up_shadow.yaml` | R125_A + `shadow_p: 0.10`, `shadow_intensity: [0.15, 0.45]`, `shadow_softness: [0.20, 0.50]` | GammaUp + DirectionalShadow — adds non-uniform spatial lighting (shadows from random angles). Training data is mostly evenly-lit studio/webcam. |
| **R125_C** | `R125_C_max_lighting_envelope.yaml` | R125_A + `cct_p: 0.25` (was 0.15), `cct_range: [2200, 9500]` (was [2700, 8000]), `brightness: [-0.15, 0.70]` (was [-0.20, 0.60]), `individual_p: 0.20` (was 0.15) | Maximum lighting envelope — pushes CCT and brightness as far as possible. Boundary test: how far can augmentation go before in-dist regresses? |

**Ablation logic:**
```
R12_A (baseline, no lighting push)
  └→ R125_A (+GammaUp only)           → isolates brightness upward push
       ├→ R125_B (+DirectionalShadow)  → isolates spatial shadow effect
       └→ R125_C (+wider CCT/brightness) → isolates colour temp / range push
```

All 3 runs: scratch, k=32, seed=737, no GRL, 30K steps max, cosine LR with warmup.

**Launch commands:**
```bash
./launch_experiment.sh -y <WANDB_PROJECT> asia-southeast1 experiments/phase2_round12_5/R125_A_gamma_up.yaml
./launch_experiment.sh -y <WANDB_PROJECT> asia-southeast1 experiments/phase2_round12_5/R125_B_gamma_up_shadow.yaml
./launch_experiment.sh -y <WANDB_PROJECT> asia-southeast1 experiments/phase2_round12_5/R125_C_max_lighting_envelope.yaml
```

**What to look for:**
- Does GammaUp alone (R125_A) improve VCD Real accuracy? (R12_A's best was 83.9%)
- Does DirectionalShadow add value on top of GammaUp, or does it hurt in-dist?
- Is there an OOD ceiling from augmentation? (R125_C pushes to the limit)
- Key comparison: R125_A vs R12_G (same seed, same everything except GammaUp) will show the pure lighting-augmentation effect.

---

## All Checkpoint Paths (Copy-Paste Ready)

```bash
# Phase 1
gs://training-job-outputs/best_checkpoints/corrected/top_n_effort_20260119_step14000_auc0.9947_eer0.0218_B16_LAION.patched.pth

# R2.5 — R25_F1 (OOD champion)
gs://training-job-outputs/phase2r2_experiments/5w453our/top_n_effort_20260212_step18500_auc0.9893_eer0.0356.pth

# R3 — FT3 (holdout champion, OOD unvalidated)
gs://training-job-outputs/phase2r3_experiments/kzfu116l/top_n_effort_20260213_step2500_auc0.9966_eer0.0191.pth

# R4 — FT7 (deployment candidate)
gs://training-job-outputs/phase2r4_experiments/udgwsu7o/top_n_effort_20260217_step500_auc0.9935_eer0.0088.pth

# R4 — FT5 (rollback candidate)
gs://training-job-outputs/phase2r4_experiments/tu0zeofr/top_n_effort_20260217_step500_auc0.9920_eer0.0122.pth

# R6 — S1 (VCD robustness winner)
gs://training-job-outputs/phase2r6_experiments/best_checkpoint.pth

# R8 — R8_E (target-heavy scratch champion)
gs://training-job-outputs/phase2r8_experiments/hu7cen3m/top_n_effort_20260224_step8000_auc0.9925_eer0.0207.pth
# R8_E alias used in R9 configs:
gs://training-job-outputs/phase2r8_experiments/R8_E_scratch_target_heavy/best_checkpoint.pth

# R9 — R9_A (Teams FT, stability bug — still best Teams run)
gs://training-job-outputs/phase2r9_experiments/1551zxa8/

# R9.5 — No new winner (R9_A retained). Best R9.5 checkpoints for reference:
# R95_B (best OOD in R9.5)
gs://training-job-outputs/phase2r95_experiments/95p2wvqi/top_n_effort_20260302_step7000_auc0.9894_eer0.0431.pth
# R95_A (best in-dist AUC in R9.5)
gs://training-job-outputs/phase2r95_experiments/lxfzu0di/top_n_effort_20260302_step7000_auc0.9900_eer0.0381.pth
# R95_D (scratch — incomplete, EWI=1)
gs://training-job-outputs/phase2r95_experiments/kt4z2qww/top_n_effort_20260302_step8000_auc0.9876_eer0.0533.pth

# R10 — R10_C (FT from R9_A, cheapest config, Teams v2 winner)
gs://training-job-outputs/phase2r10_experiments/

# R11 — R11_G ⚠️ CRASHED (ViT-L-14, AUC 0.9932, OOD AUC 0.9873 — checkpoint may be partial)
# R11_D (best finished B-16 FT)
gs://training-job-outputs/phase2r11_experiments/

# R12 — R12_G ★ (composite leader, scratch seed=737)
gs://training-job-outputs/phase2r12_experiments/0xxqwhxg/ood_composite_effort_20260310_step12500_auc0.9923_eer0.0253.pth
# R12_A (OOD/real-data pick, best VCD Real 83.9%)
gs://training-job-outputs/phase2r12_experiments/4v8av986/ood_composite_effort_20260309_step7500_auc0.9705_eer0.0744.pth
```

---

## W&B Quick Reference

| Winner | W&B Entity/Project | Run ID |
|--------|-------------------|--------|
| R25_F1 | `dtect-vision/phase2-experiments` | `5w453our` |
| R3_FT3 | `dtect-vision/phase2-experiments` | `kzfu116l` |
| R4_FT7 | `dtect-vision/phase2-experiments` | `udgwsu7o` |
| R4_FT5 | `dtect-vision/phase2-experiments` | `tu0zeofr` |
| R6_S1 | `dtect-vision/phase2-round6` | `s3tx3fk4` |
| R6_S3 | `dtect-vision/phase2-round6` | `iwk4pe1j` |
| R8_E | `dtect-vision/phase2-round8` | `hu7cen3m` |
| R9_A | `dtect-vision/phase2r9-experiments` | `1551zxa8` |
| R9_H | `dtect-vision/phase2r9-experiments` | `8q17507q` |
| R9_C | `dtect-vision/phase2r9-experiments` | `1gmroh48` |
| R95_B | `dtect-vision/phase2r95-experiments` | `95p2wvqi` |
| R95_A | `dtect-vision/phase2r95-experiments` | `lxfzu0di` |
| R95_D | `dtect-vision/phase2r95-experiments` | `kt4z2qww` |
| R10_C | `dtect-vision/phase2r10-experiments` | `R10_C` |
| R10_D | `dtect-vision/phase2r10-experiments` | `R10_D` |
| R11_G | `dtect-vision/phase2r11-experiments` | `R11_G` |
| R11_D | `dtect-vision/phase2r11-experiments` | `R11_D` |
| R12_G ★ | `dtect-vision/phase2r12-experiments` | `0xxqwhxg` |
| R12_A (OOD) | `dtect-vision/phase2r12-experiments` | `4v8av986` |
| R12_B | `dtect-vision/phase2r12-experiments` | `wupk4909` |
| R12_E | `dtect-vision/phase2r12-experiments` | `ys8z1div` |

---

## Key Lessons Encoded in the Winners

1. **R25_F1**: Capacity matters (k=32 > k=8). Simple loss and augmentation beat complex ones.
2. **R3_FT3**: Holdout AUC ≠ deployment performance. Never deploy without OOD validation.
3. **R4_FT7**: Target-domain data + family-aware augmentation fix specific failure modes (WMA enhanced fakes).
4. **R6_S1**: Adding real target-domain data (VCD) directly to training is the strongest lever for real-domain robustness.
5. **Across all rounds**: Quality-robust augmentation consistently hurt OOD generalization (R3, R6_S5). Data-level fixes outperform augmentation-level fixes.
6. **R8_E**: When the training distribution shifts dramatically, starting from scratch avoids inherited biases. Inverting the data balance (target-heavy, DF40-light) solved VisoMaster (69%→97%) and pushed VCD Real to a new high (70%→82%).
7. **R9_A**: Always verify config keys reach the Trainer — flat YAML scalars can silently drop. Teams passthrough data at high weight + FT from best checkpoint is the winning Teams formula. Facedancer (DF40, weight 0.2) remains universally weakest — data weight matters.
8. **Across R8+R9**: Scratch training consistently wins on OOD AUC (R8_E > R8 FT runs, R9_C > R9 FT runs). Fine-tuning wins on holdout AUC. The deployment-relevant metric is OOD.
9. **R9.5**: Stability regularization monotonically degrades OOD (λ=0 > 0.1 > 0.3 > 0.5). Data weighting (5× DF40) has zero effect on facedancer. All FT runs converge to the same DF40 holdout boundary. Scratch (R95_D) preserves hard-method detection but needs 20K+ steps. R9_A's accidental λ=0 was the optimal configuration.
10. **R10**: Simplest FT config wins again (R10_C = cheapest, beat heavy configs). Teams v2 codec-degraded data integrates smoothly. VCD Real continues regressing as fake diversity grows (82% → 80% → 74%). ViT-L/14 shows OOD promise (0.9709) even when crashing early — warrants dedicated exploration.
11. **R11**: **ViT-L/14 is the single biggest lever discovered since R6 VCD reals.** R11_G solved facedancer (92.9%, up from 63.6%), boosted VCD real to 92.0%, and achieved best-ever OOD AUC 0.9873 — all on scratch training. The B-16 backbone has a hard capacity ceiling on hard fake methods. Group DRO provided no benefit. Unfortunately R11_G crashed and may not have a usable checkpoint.
12. **R12 (final)**: The big story is **seed sensitivity and late convergence**. R12_G (seed=737) was dead last on OOD at epoch 3 (0.9126) but surged to #1 composite by epoch 5 (0.9738). R12_A peaked on OOD at step 7,500 then eroded — proving that best-composite checkpointing captures the right moment. GRL verdict is negative (0.1 = negligible, 0.3 = harmful, GRL+Teams OOD = stalled). k=64 offered nothing over k=32. FT from R8_E converged instantly and plateaued. **Scratch training with compound lighting augmentation and patient training is the R12 formula.** The expanded OOD eval (with Teams OOD) makes numbers incomparable to earlier rounds — this is deliberate.
