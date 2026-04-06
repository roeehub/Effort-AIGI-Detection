# R11 Strategic Plan

> **Date:** 2026-03-08
> **R9 Champion (no Teams):** R9_D (`m7etxxnp`) — AUC 0.9942, EER 0.0325, OOD 0.9689, Unified 0.9928
> **R9 Best EER:** R9_F (`ueoziuou`) — AUC 0.9926, EER 0.0266, OOD 0.9632, Unified 0.9919
> **R9 Prod (Teams v1):** R9_A (`1551zxa8`) — AUC 0.9891, EER 0.0457, OOD 0.9692, Unified 0.9875
> **R10 Best (Teams v2, FT from R9_A):** R10_C — AUC 0.9836, EER 0.0547, OOD 0.9502, Unified 0.9833, Teams HO avg 88.0%

---

## 1. R10 Conclusions — What We Learned

### 1a. Final R10 Results

| Run | Strategy | Step | AUC | EER | TPR@1% | OOD | Unified | Teams HO |
|-----|----------|-----:|----:|----:|-------:|----:|--------:|---------:|
| R10_C | FT narrow | 8167 ✅ | 0.9836 | 0.0547 | 0.8609 | 0.9502 | 0.9833 | 88.0% |
| R10_G | FT wide | 8167 ✅ | 0.9829 | 0.0591 | — | 0.9547 | 0.9825 | 86.8% |
| R10_A | Scratch wide | 8706 💥 | 0.9805 | — | — | 0.9439 | — | 87.0% |
| R10_F | Scratch narrow | 8710 💥 | 0.9809 | — | — | 0.9349 | — | 86.8% |
| R10_E | Scratch low-s | 8305 💥 | 0.9804 | — | — | 0.9273 | — | 87.0% |
| R10_B | Scratch mixup | 8317 💥 | 0.9803 | — | — | 0.9274 | — | 87.0% |
| R10_D | ViT-L-14 | 8718 💥 | 0.9042 | — | — | **0.9807** | — | 77.6% |

✅ = finished, 💥 = crashed at Vertex AI 24h timeout

### 1b. Key Findings

**Finding 1: Teams v2 data hurts in-dist AUC vs R9 champions.**
Best R10 AUC = 0.9836 (R10_C) vs R9_D = 0.9942 (−1.06pp). Best R10 OOD = 0.9547 vs R9_D = 0.9689 (−1.4pp). The 4.7× more Teams data diluted in-distribution performance.

**Finding 2: Teams accuracy massively improved.**
R10 runs averaged ~87–88% Teams holdout accuracy. R9_A (Teams v1) was ~74%. The Teams v2 data clearly helps domain-specific performance, just at a cost to the DF40 in-dist frontier.

**Finding 3: FT from R9_A plateaued early, limited headroom.**
R10_C's best AUC (0.9836) appeared by step ~6K and didn't improve. Starting from R9_A (0.9891) rather than R9_D (0.9942) left performance on the table.

**Finding 4: All scratch ablations were indistinguishable.**
A (wide aug), B (mixup), E (low-s), F (narrow aug) — all clustered at AUC 0.980 ± 0.0005 at ~8.5K steps. Mixup, low-s, and wide-vs-narrow aug produced zero measurable differentiation. (Caveat: all crashed at 24h before reaching 22K target.)

**Finding 5: ViT-L-14 — terrible in-dist, best-ever OOD.**
AUC = 0.9042 (catastrophically low), yet OOD = 0.9807 (best across R9–R10). Root cause: `lambda_reg=1.0` with `rank=1023` left the SVD residual subspace too constrained. `S_res_max = 0.032`, `cosine_sim = −0.992` — the model barely modified the frozen CLIP features. The L-14 backbone has strong OOD representations natively; we just need to unlock the trainable subspace.

**Finding 6: Universal weak spots persist.**
GhostFace-v2 = 50%, InStyleSwapper256-B = 50%, facedancer = 57–64% — across ALL B-16 runs. These are architectural limitations at B-16 capacity, not training failures.

**Finding 7: 24h Vertex AI timeout killed all scratch runs.**
All 5 scratch runs (22K steps) hit exactly 24.0h wall-clock before reaching their target. Only the 8K-step FT runs (C, G) completed. R11 must budget steps vs wall-clock more carefully.

### 1c. The Core Insight for R11

**Fine-tune R9_D, not R9_A.** R9_D is the best checkpoint ever produced (AUC 0.9942, OOD 0.9689) — it was trained without Teams data. The strategy is to use R11 to inject Teams v2 competence into R9_D's superior feature space rather than trying to simultaneously learn DF40 discrimination and Teams adaptation from scratch. This gives us a starting AUC 1.06pp higher than R10_C's starting point.

---

## 2. R11 Objectives

1. **Primary:** Absorb Teams v2 data into R9_D without regressing in-dist AUC below 0.990
2. **Secondary:** Fix ViT-L-14 in-dist collapse while preserving its OOD advantage
3. **Tertiary:** Test if Group DRO can lift GhostFace-v2 / InStyleSwapper256-B from 50%
4. **Exploration:** Find scratch ceiling with enough steps (30K) and proper timeout budget

---

## 3. R11 Experiment Matrix

### Design Principles

1. **FT from R9_D (0.9942)** for most runs — the best foundation we have
2. **8K steps for FT runs** — proven sufficient in R10 (C, G both converged within 8K)
3. **Low LR (3e-5)** for FT — gentle adaptation, don't destroy R9_D's features
4. **Teams v2 bucket** — same as R10
5. **Lambda_reg = 0** for B-16, **0.01** for L-14 — heavily relaxed from the catastrophic 1.0
6. **Budget wall-clock** — scratch gets 30K steps (should still fit 24h based on ~8.7K/24h R10 rate for B-16... tight. L-14 may need monitoring)

### Family Weights

Two weight profiles tested:

| Family | Standard (A/B/C/D/H) | High-Teams (F) |
|--------|:--------------------:|:--------------:|
| `df40_fake` | 0.15 | **0.10** |
| `df40_real` | 0.4 | 0.4 |
| `visomaster_fake` | 2.5 | 2.5 |
| `deeplive_non_enhanced_fake` | 2.5 | 2.5 |
| `deeplive_enhanced_fake` | 3.0 | 3.0 |
| `deeplive_teams_fake` | **5.0** | **7.0** |
| `deeplive_teams_real` | **4.0** | **5.0** |
| `realpool_real` | 2.0 | 2.0 |
| `external_real` | 2.5 | 2.5 |

### Run Matrix

| Run | Strategy | Base Ckpt | Steps | Backbone | Key Differentiator |
|-----|----------|-----------|------:|----------|-------------------|
| **R11_A** | FT | R9_D | 8K | B16-LAION | **Primary bet** — R9_D + Teams v2, narrow aug |
| **R11_B** | FT + Group DRO | R9_D | 8K | B16-LAION | Group DRO upweighting for GhostFace-v2, InStyleSwapper-B |
| **R11_C** | FT | R9_D | 8K | B16-LAION | Wide augmentation absorption test |
| **R11_D** | FT | R9_F | 8K | B16-LAION | Tighter boundary transfer (R9_F had best EER 0.0266) |
| **R11_E** | Scratch | — | 30K | B16-LAION | Scratch ceiling w/ full budget |
| **R11_F** | FT | R9_D | 8K | B16-LAION | Max Teams emphasis (weights 7.0/5.0) |
| **R11_G** | Scratch | — | 30K | **ViT-L-14** | L-14 fix: lambda=0.01, rank=768, LR=3e-4 |
| **R11_H** | FT | R9_D | 10K | B16-LAION | Ultra-conservative LR=1e-5 |

**Clean ablation axes:**
- **A vs B:** Effect of Group DRO (same config otherwise)
- **A vs C:** Effect of wider augmentation in FT from R9_D
- **A vs D:** Effect of starting checkpoint (R9_D AUC=0.9942 vs R9_F AUC=0.9926/EER=0.0266)
- **A vs F:** Effect of aggressive Teams family weighting
- **A vs H:** Effect of 3× lower learning rate (3e-5 vs 1e-5)
- **A vs E:** Scratch vs fine-tune strategy with enough steps
- **G vs R10_D:** Effect of fixed lambda/rank on L-14

---

## 4. Run Details

### R11_A — FT from R9_D, Narrow Aug (Primary Bet)

The central hypothesis: R9_D's 0.9942 AUC feature space can absorb Teams v2 data with minimal regression if fine-tuned gently.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Checkpoint | R9_D (`m7etxxnp`) step 4000 | Best-ever AUC |
| LR | 3e-5, cosine_with_warmup | Same as R10_C which worked |
| Warmup | 500 steps | Standard for FT |
| Steps | 8K | Proven convergence window |
| ArcFace s | 6→12 | Same schedule as R10_C, moderate amplification |
| Aug | Narrow (R9-style) | Don't confound two variables |
| Teams weight | 5.0/4.0 | Moderate — higher than R10's 4.0/3.0 to push absorption |
| Eval freq | Every 500 steps | Dense checkpointing |

**Success criteria:** AUC ≥ 0.990, Teams HO ≥ 85%, OOD ≥ 0.965

### R11_B — FT from R9_D + Group DRO

Same as R11_A, with `use_group_dro: true`. Group DRO upweights the loss on the worst-performing generation methods, combating the GhostFace-v2 = 50% and InStyleSwapper256-B = 50% collapse observed universally in R10.

| Parameter | Value |
|-----------|-------|
| Group DRO β | 3.0 |
| Group DRO step size | 0.01 |
| Everything else | Same as R11_A |

**What to watch:** Does GhostFace-v2 holdout accuracy rise above 50%? Does overall AUC regress vs A?

### R11_C — FT from R9_D, Wide Aug

Same as R11_A, but with wider augmentation to test whether R9_D can absorb both Teams data and wider light/color perturbations simultaneously.

| Parameter | Value |
|-----------|-------|
| gamma_limit | [50, 150] (vs A's [80, 120]) |
| brightness | 0.40 (vs A's 0.25) |
| contrast | 0.35 (vs A's 0.25) |
| oneof_p | 0.50 (vs A's 0.30) |
| rotate | 12 (vs A's 10) |
| Everything else | Same as R11_A |

**What to watch:** Does wider aug improve OOD without hurting in-dist AUC?

### R11_D — FT from R9_F (Best EER Base)

Tests whether R9_F's tighter decision boundary (EER = 0.0266, the best we've seen) transfers better than R9_D's higher AUC.

| Parameter | Value |
|-----------|-------|
| Checkpoint | R9_F (`ueoziuou`) step 500 | 
| Everything else | Same as R11_A |

**Rationale:** R9_F's EER superiority (0.0266 vs 0.0325) means its real/fake separator is sharper. If it maintains that edge after Teams data injection, it's a better production model (lower false positive rate).

### R11_E — Scratch 30K Steps

Pure scratch run with enough budget to find the ceiling. R10 scratch runs were at ~8.5K when they crashed — they were clearly still improving. This run pushes to 30K.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Checkpoint | None (CLIP init) | Scratch |
| LR | 2e-4 | Standard for scratch |
| Warmup | 1500 steps | Longer warmup for longer run |
| Steps | 30K | 3.4× what R10 achieved before crashing |
| ArcFace s | 10→14 | Same as R10 scratch runs |
| Aug | Narrow | Isolate the step-count variable |
| Patience | 15 epochs | Longer patience for slower scratch convergence |

**Risk:** 30K steps at ~360 steps/hour = ~83 hours. This **will exceed 24h**. Either needs a larger machine type or checkpoint-resume. Monitor closely.

### R11_F — FT from R9_D, High Teams Weight

Tests whether aggressively weighting Teams data (7.0/5.0) accelerates Teams absorption at the cost of DF40 in-dist.

| Parameter | Value |
|-----------|-------|
| teams_fake weight | 7.0 (vs A's 5.0) |
| teams_real weight | 5.0 (vs A's 4.0) |
| df40_fake weight | 0.10 (vs A's 0.15) |
| Everything else | Same as R11_A |

**Trade hypothesis:** Higher Teams weight → higher Teams HO accuracy, lower DF40 AUC. The question is whether the trade is worth it for production (where Teams-like video calls are the deployment domain).

### R11_G — ViT-L-14 Fixed (The Fix)

R10_D proved L-14 has spectacular OOD ability (0.9807) but catastrophic in-dist (0.9042). The root cause was `lambda_reg=1.0` choking the SVD residual, plus `rank=1023` (almost full rank) leaving near-zero trainable capacity.

| Parameter | R10_D (broken) | R11_G (fixed) |
|-----------|:--------------:|:-------------:|
| lambda_reg | 1.0 | **0.01** |
| rank (frozen SVD components) | 1023 | **768** |
| k (trainable residual dim) | 1 | **256** |
| LR | 2e-4 | **3e-4** |
| hidden_size | 1024 | 1024 |
| Steps | 22K | **30K** |

**The fix explained:** `lambda_reg = 0.01` (100× reduction) allows the orthogonal loss to be nearly ignored — the model can freely modify features. `rank = 768` freezes only the top 768 singular components, leaving 256 fully trainable dimensions (vs 1 in R10_D). `LR = 3e-4` is slightly higher to compensate for the larger trainable subspace.

**Success criteria:** AUC > 0.980 (vs R10_D's 0.9042). If OOD stays > 0.960, this becomes the new champion architecture.

### R11_H — FT from R9_D, Ultra-Low LR

Tests whether 3× slower learning rate (1e-5 vs 3e-5) preserves more of R9_D's features during Teams absorption.

| Parameter | Value |
|-----------|-------|
| LR | 1e-5 (vs A's 3e-5) |
| Steps | 10K (2K more to compensate for slower learning) |
| s_end | 10 (vs A's 12) | 
| Everything else | Same as R11_A |

**Hypothesis:** If 3e-5 is too aggressive and destroys R9_D's features early, this run should show higher final AUC but slower convergence.

---

## 5. Early Results (12h Check-In, ~Epoch 2)

All 8 runs launched successfully on 2026-03-07 ~22:40 UTC. Status at 2026-03-08 ~10:30 UTC:

| Run | Step | AUC | EER | TPR@1% | OOD | Unified | Teams Avg |
|-----|-----:|----:|----:|-------:|----:|--------:|----------:|
| **R11_G (L-14 fixed)** | 4593 | **0.9912** | **0.0263** | 0.8764 | 0.9055 | **0.9907** | **96.4%** |
| R11_D (FT R9_F) | 4593 | 0.9829 | 0.0613 | 0.8477 | 0.9399 | 0.9820 | 85.7% |
| R11_F (high Teams) | 4079 | 0.9828 | 0.0569 | 0.8631 | 0.9491 | 0.9797 | 88.7% |
| R11_A (FT R9_D) | 4597 | 0.9827 | 0.0569 | 0.8653 | 0.9496 | 0.9834 | 85.7% |
| R11_B (Group DRO) | 4597 | 0.9827 | 0.0569 | 0.8609 | 0.9497 | 0.9831 | 85.9% |
| R11_C (wide aug) | 4079 | 0.9821 | 0.0613 | 0.8521 | 0.9511 | 0.9802 | 87.8% |
| R11_H (ultra-low LR) | 4079 | 0.9810 | 0.0569 | 0.8322 | **0.9532** | 0.9794 | 88.1% |
| R11_E (scratch 30K) | 4593 | 0.9687 | 0.0788 | 0.7395 | 0.8941 | 0.9667 | 81.5% |

### Early Observations

**R11_G is the clear standout.** The ViT-L-14 lambda/rank fix is working spectacularly:
- AUC 0.9912 (vs R10_D's 0.9042 — a +8.7pp recovery)
- EER 0.0263 (already better than R9_F's 0.0266, the prior best)
- Teams avg 96.4%, with GhostFace-v2 at 100% (was 50% on ALL B-16 runs)
- facedancer at 92.86% (was 57–64% on B-16 — capacity ceiling lifted!)
- Only weakness: OOD = 0.9055 (was 0.9807 in R10_D, but expected to rise with more training)

**B-16 FT runs are tightly clustered at AUC ~0.982.** No run has yet beaten R10_C's 0.9836, and all are well below R9_D's 0.9942. These are still early (epoch 2 of ~6) — the fine-tuning runs should improve through epoch 3–4.

**Group DRO (B) hasn't differentiated from A.** Identical AUC (0.9827 vs 0.9827), Teams GhostFace-v2 still at 50% for both. May need more epochs to kick in, or the DRO grouping doesn't have enough signal yet.

**High Teams weight (F) shows Teams benefit.** InStyleSwapper256-B at 90% (vs 80% in A/B/D). Also best Teams avg among B-16 runs (88.7%). But no AUC advantage.

**Ultra-low LR (H) has highest OOD so far** (0.9532) among B-16 runs — preserving more of R9_D's OOD features as hypothesized. But AUC is lagging (0.9810) and needs to catch up.

**R11_E (scratch) tracking as expected** — at 4.6K of 30K steps, AUC 0.9687 trails the FT runs by ~1.4pp. At risk of the same 24h timeout as R10 scratch runs.

### Teams Method Breakdown (B-16 Runs)

| Teams Method | A | B | C | D | F | H |
|-------------|---:|---:|---:|---:|---:|---:|
| GhostFace-v2 | 50% | 50% | 50% | 50% | 50% | 50% |
| InStyleSwapper256-B | 80% | 80% | 80% | 80% | **90%** | **90%** |
| CSCS | 83% | 83% | **100%** | 83% | **100%** | **100%** |
| edge_cases | 85% | 87% | 88% | 87% | 87% | 85% |
| GhostFace-v3 | 88% | 88% | **100%** | 88% | **100%** | **100%** |
| minimal_processing | 89% | 89% | 87% | 87% | 87% | 87% |
| quality_enhancement | 97% | 97% | 97% | 97% | 97% | 94% |
| GhostFace-v1 | 100% | 100% | 100% | 100% | 100% | 100% |
| InStyleSwapper256-A | 100% | 100% | 88% | 100% | 88% | 88% |

GhostFace-v2 Teams = 50% is universal across all B-16 runs. Only R11_G (L-14) breaks through.

### R10-Weak Method Recovery

| Method | A | B | C | D | F | H | **G (L-14)** |
|--------|---:|---:|---:|---:|---:|---:|---:|
| GhostFace-v2 (holdout) | 97% | 93% | 93% | 97% | 93% | 93% | 93% |
| InStyleSwapper256-B (holdout) | 98% | 98% | 98% | 98% | 98% | 98% | 98% |
| facedancer | 64% | 64% | 57% | 64% | 57% | 57% | **93%** |

Note: The "holdout" GhostFace-v2 and InStyleSwapper256-B are *non-Teams* versions from the DF40/visomaster datasets, which perform well. The *Teams* variants of GhostFace-v2 collapse to 50%. This suggests a Teams-specific domain issue rather than a model capacity problem for B-16.

---

## 6. What to Watch Next

1. **R11_G OOD trajectory** — If OOD rises above 0.95 by step 10K while AUC stays >0.988, L-14 becomes the new champion architecture
2. **B-16 AUC convergence** — Do A/B/C/D/F/H reach R9_D levels (0.994) or plateau at ~0.985?
3. **Group DRO activation** — Does B start diverging from A after epoch 3?
4. **R11_E wall-clock** — Will it survive the 24h timeout? At ~360 steps/h, 30K = 83h. It will crash at ~8.7K again
5. **R11_H AUC catch-up** — Ultra-low LR should be slower but may end higher
