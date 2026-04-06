# The Experiment Journey — Feb 10 to Feb 22, 2026

> A chronological narrative of the Effort AIGI detector's Phase 2 experiment arc,
> reconstructed from the 11 markdown documents written along the way.

---

## Document Timeline

| Date | File | Role |
|:-----|:-----|:-----|
| Feb 10 | `PHASE2_EXPERIMENT_PLAN.md` | Origin — the grand design |
| Feb 11 | `phase2/INTERMEDIATE_REPORT.md` | R1 results, config bug discovery |
| Feb 17 17:00 | `phase2_round4/SESSION_HANDOFF_SUMMARY.md` | R4 handoff, FT7 as provisional winner |
| Feb 17 17:01 | `phase2_round4/README.md` | R4 run matrix |
| Feb 17 22:09 | `phase2_round5/README.md` | R5 scratch cycle plan |
| Feb 18 09:00 | `phase2_round2/PROGRESS_REPORT.md` | Mega doc: R1→R2→R2.5→R3 full story |
| Feb 18 16:43 | `REAL_ROBUSTNESS_PLAN.md` | Bridge plan — VCD root cause + R6 blueprint |
| Feb 19 15:20 | `phase2_round6/R6_EXPERIMENT_REPORT.md` | R6 results at ~6K steps |
| Feb 19 15:20 | `phase2_round6/README.md` | R6 experiment matrix |
| Feb 20 22:10 | `phase2_round6/R6_TO_R7_ROBUSTNESS_RECAP.md` | R6→R7 transition recap |
| Feb 22 20:50 | `WINNING_RUNS_REGISTRY.md` | Consolidated winner registry |

---

## Act I: The Grand Design (Feb 10)

**Document**: `PHASE2_EXPERIMENT_PLAN.md` (664 lines)

Phase 1 had just delivered AUC **0.9947** on in-dist DF40, but only **83.7%** balanced accuracy on VisoMaster OOD — and an **8.6% false positive rate**. The model was strong on its training distribution and brittle everywhere else.

The plan was sweeping: **5 experiment groups** run over **5 sequential nights**, each informing the next:

| Night | Group | Question |
|:------|:------|:---------|
| 1 | **A: Data Composition** (4 runs) | Does adding VisoMaster help? Does DeepLive help on top of DF40? |
| 2 | **B: Training Strategy** (4 runs) | Is Group DRO needed? Does checkpoint resume beat scratch? |
| 3 | **C: Loss Functions** (3 runs) | Does ArcFace help with more diverse data? |
| 4 | **B (cont): Curriculum** (2 runs) | Does curriculum help, or is flat training sufficient? |
| 5 | **D: Scale & Efficiency** (3 runs) | Can we squeeze more from larger k or larger batches? |

Key design bets:
- Add VisoMaster to training with a **3-model OOD holdout** (GhostFace-v3, InStyleSwapper256-C, SimSwap512)
- Introduce **Group DRO** for per-method fairness
- Trial **ArcFace/cosine softmax** heads
- Test **curriculum** by tier difficulty (STRONG → MODERATE → MINIMAL)

The plan also identified **2-3 days of prerequisite code changes**: creating a `visomaster` data source, extending `combined_paired` to support 3 sources, adding test splits, and wiring Group DRO method mappings.

Success criteria were explicit:

| Metric | Phase 1 Baseline | Target |
|:-------|:---:|:---:|
| DF40 in-dist AUC | 0.9947 | ≥ 0.99 |
| VisoMaster OOD balanced acc | 83.7% | **≥ 88%** |
| VisoMaster tier MINIMAL | 79.7% | **≥ 85%** |
| Real accuracy (all sources) | 96.6% | ≥ 93% |
| Per-model minimum fake acc | 80.3% | **≥ 85%** |

The document radiates methodical confidence — risk matrices, rollback criteria, GCS bucket references, appendices with exact config diffs. It's the "where we thought we were going."

---

## Act II: First Blood, First Surprise (Feb 11)

**Document**: `phase2/INTERMEDIATE_REPORT.md`

R1 launches. At epoch 5/30, a **critical config bug** is discovered: the backbone was loading **ViT-L-14** instead of **ViT-B-16**. The "Phase 1 winner" had been wrong about what it was even running.

The pivot to B-16 with LAION DataComp-XL weights began here. The 4 Phase 1 configs (CE λ=0.01, CE λ=0, cosine softmax, ArcFace conservative) were re-run on the correct backbone.

This is the moment where the tidy 5-night plan first contacts reality. Plans rarely survive, but the reflexes — detect, diagnose, adapt — were fast.

---

## Act III: The Great Acceleration (Feb 17-18)

**Documents**: `PROGRESS_REPORT.md` (the mega doc), `SESSION_HANDOFF_SUMMARY.md`, `phase2_round4/README.md`, `phase2_round5/README.md`

The most compressed period: **R1→R2→R2.5→R3→R4→R5 in under a week.** The PROGRESS_REPORT (written Feb 18) is the longest document — a retrospective war diary covering 8 days of dense iteration.

### R2: Expanding the Data (Feb 12-13)

First attempt to add VisoMaster to training (the Group A experiments from the plan). Results were mixed — VisoMaster improved OOD VisoMaster accuracy but the model was still fragile on unseen real distributions.

### R2.5: The OOD Breakthrough (Feb 13-14)

The unexpected hero: **`property_balanced` batching strategy.**

`R25_F1` achieved AUC **0.9893** with FPR slashed from 8.6% to **1.8%**. Property-balanced sampling ensured every generation method category was seen every epoch, preventing the model from "forgetting" rare methods. The model generalized for the first time.

**Winner**: `R25_F1` (`5w453our`) — the OOD champion.

### R3: The Paradox (Feb 14-15)

`R3_FT3` (fine-tuned from R2.5) hit AUC **0.9966** on holdout — new all-time record. But OOD performance didn't improve. Sometimes it regressed.

This crystallized the **central lesson of the entire project**:

> **Holdout AUC ≠ deployment performance.**

The metric you optimize isn't necessarily the metric that matters. The holdout set (DF40 val split) shared the same quality profile as training. High holdout AUC just meant the model was better at the in-distribution task, not that it would generalize to webcam captures, enhanced fakes, or other real-world distributions.

This insight would echo through every subsequent round.

### R4: The Deployment Candidate (Feb 16-17)

FT7 emerged by fine-tuning from R2.5's checkpoint with:
- **Cosine softmax head** (ArcFace m=0.0, s=10→18)
- **WMA training data** (GFPGAN-enhanced face-swap fakes)
- `quality_targeted_family` augmentation

Results:

| Metric | FT7 |
|:-------|:---:|
| WMA detection | **88.35%** |
| Enhanced DeepLive TPR | **99.53%** |
| DF40 fake TPR | 86.85% |
| External real FPR | 4.38% |
| In-dist AUC | 0.9941 |

FT7 became the **deployment candidate** — the first model you'd trust in production for most use cases.

**Winner**: FT7 (`udgwsu7o`)

### R5: Can Scratch Beat Fine-Tuning? (Feb 17)

Three scratch-training configs — starting from pretrained CLIP, not from any fine-tuned checkpoint. None beat FT7. The fine-tuned checkpoint lineage (Phase 1 → R2.5 → R4) carried accumulated knowledge that scratch couldn't match in the same step budget.

FT7 retained its crown.

---

## Act IV: The Real Problem Revealed (Feb 18)

**Document**: `REAL_ROBUSTNESS_PLAN.md` (1,135 lines — the longest single document)

Written the same day as the progress report, this is the **intellectual pivot** of the entire project. FT7 could detect fakes well. But it had one glaring failure: **VCD real webcam faces classified as fake at ~40-49% rate.**

### Root Cause: The Quality-Property Shortcut

A forensic analysis of image quality metrics across all real-face sources revealed the problem:

| Metric | DF40 Reals (training) | VCD Reals (failing) | Ratio |
|:---|---:|---:|---:|
| Laplacian variance (sharpness) | 38.9 | 295.3 | **7.6×** |
| Texture local variance | 3.6 | 23.3 | **6.5×** |
| Edge density | 0.022 | 0.044 | **2.0×** |
| HF noise std | 1.97 | 4.49 | **2.3×** |
| Effective resolution (90th pct) | 0.210 | 0.405 | **1.9×** |

The model learned **"soft, smooth, low-noise = real"** because that's what DF40 reals look like. VCD webcam captures are sharp, noisy, textured — they look like *fakes* to the model.

Codec simulation had been tried but made images **blurrier** (wrong direction for VCD, which is *sharper* than DF40). The gap couldn't be augmented away with existing tools.

### The Three-Pronged Intervention

The plan designed three interventions, each addressing the problem from a different angle:

**Part A — Add VCD Reals to Training (~800 frames)**
- New `UnifiedUnpairedRealSample` dataclass (no paired fake side)
- Identity-split: 20% of VCD identities for training, 80% reserved for OOD eval
- Modest family weight (0.6) to avoid overwhelming fake detection signal
- Full implementation spec: 10 steps (A1-A10) with code snippets, line references, routing changes

**Part B — `vcd_targeted` Augmentation Preset**
- Sharpen reals (push DF40 toward VCD's profile): alpha 0.30-0.70, p=0.60
- Add noise to reals (VCD has 2.3× higher HF noise)
- Degrade fakes (prevent new shortcut: "sharp+noisy = real")
- **Do NOT** increase codec simulation — it goes the wrong direction

**Part C — Gradient Reversal Quality-Domain Head (DANN-style)**
- Classify quality domain (clean_academic / webcam_codec / social_media / enhanced)
- Reverse gradients through GRL so backbone features become quality-invariant
- Sigmoid lambda schedule: starts at 0, reaches ~1.0 at 70% of training
- The most theoretically principled solution (Ganin & Lempitsky 2015)

### Experiment Matrix

| Config | VCD Reals | Augmentation | GRL Head | Purpose |
|:---|:---:|:---:|:---:|:---|
| R6_S1 | 800 | `vcd_targeted` | No | Data + augmentation |
| R6_S2 | None | `vcd_targeted` | No | Ablation: augmentation alone |
| R6_S3 | 800 | `vcd_targeted` | Yes | Full stack |

Success criteria: VCD real accuracy >80% (target), >70% (hard minimum). Regression gates: WMA ≥50%, DeepLive TPR ≥90%, DF40 TPR ≥80%.

---

## Act V: R6 Delivers (Feb 19)

**Documents**: `phase2_round6/R6_EXPERIMENT_REPORT.md`, `phase2_round6/README.md`

Three configs launched. Results at ~6K steps:

| Config | VCD Real Acc | WMA | DeepLive TPR |
|:---|---:|---:|---:|
| **S1** (VCD reals + augmentation) | **0.698** | ≥85% | ≥95% |
| S2 (augmentation only) | ~0.60 | — | — |
| S3 (full stack with GRL) | ~0.65 | — | — |

**S1 won.** Simply adding diverse real data + targeted augmentation improved VCD accuracy by **+15 percentage points** over FT7's ~51-60% baseline.

Two key findings:

1. **Data diversity was the dominant lever.** S2 (augmentation only, no VCD reals) underperformed S1. The model needed to *see* sharp, noisy real faces during training — you can't fully simulate them.

2. **The gradient reversal head didn't help much.** S3 (full stack) performed worse than S1. The adversarial quality head may have been fighting the classification signal rather than complementing it — or it just needed more training steps to converge. Either way, the simplest intervention (add the data) beat the most theoretically principled one.

This echoes the plan's own Design Principle #1: *"Data diversity is the #1 lever — more methods > more tuning."*

**Winner**: S1 (`s3tx3fk4`)

---

## Act VI: The Bridge to R7 (Feb 20)

**Document**: `phase2_round6/R6_TO_R7_ROBUSTNESS_RECAP.md`

0.698 is progress, but not 0.80. The recap analyzes what worked and what's still missing, then designs **6 new R7 configs** to push further:

- More VCD samples (increase from 800 to 1200+)
- Stronger augmentation variants
- Label smoothing to soften overconfident predictions
- Different family weight ratios (increase `external_real` weight)
- Longer training with patience

The target shifts from *"can we move the needle?"* to *"can we close the gap to 80%?"*

---

## Act VII: The Registry (Feb 22)

**Document**: `WINNING_RUNS_REGISTRY.md`

The consolidation. Every winner from every round, with W&B run IDs, GCS checkpoint paths, and the metric that mattered in each context:

| Round | Winner | W&B ID | Defining Metric |
|:------|:-------|:-------|:----------------|
| Phase 1 | B16-old | — | AUC 0.9947 (but 8.6% FPR) |
| R2.5 | R25_F1 | `5w453our` | FPR **1.8%** — OOD champion |
| R3 | R3_FT3 | `kzfu116l` | AUC **0.9966** — holdout champion (but OOD unvalidated) |
| R4 | FT7 | `udgwsu7o` | WMA **88.35%** — deployment candidate |
| R5 | (none) | — | FT7 retained |
| R6 | S1 | `s3tx3fk4` | VCD real acc **0.698** — +15pp robustness improvement |
| R7 | (pending) | — | 6 configs designed, target >0.80 |

---

## The Arc

Reading these documents in chronological order tells a clear story with a recurring pattern:

```
Ambition → Reality → Diagnosis → Intervention → Partial Victory → Repeat
```

### What Was Planned vs What Happened

The original Phase 2 plan imagined a **tidy sequential progression**: data → strategy → loss → curriculum → scale, with each night building cleanly on the last.

What actually happened was **messier and more instructive** — a tight loop of:

```
Run → Analyze failure mode → Diagnose root cause → Design targeted intervention → Run again
```

The 5-night plan became 7+ rounds over 12 days. Many of the planned experiments (Groups C-D) were never run because the failure modes discovered along the way demanded different experiments than the ones anticipated.

### The Three Big Insights

1. **Holdout AUC ≠ Deployment Performance** (R3's paradox)
   - R3_FT3 achieved the highest AUC ever (0.9966) but didn't improve OOD
   - The holdout set shared the training distribution's quality profile
   - Optimizing for holdout just makes you better at the in-distribution task

2. **Data Diversity > Model Sophistication** (R6's lesson)
   - The gradient reversal head (DANN, theoretically principled) was outperformed by simply adding 800 VCD real frames
   - Property-balanced batching (R2.5) mattered more than loss function design
   - Design Principle #1 proved itself repeatedly: *"Data diversity is the #1 lever"*

3. **Quality-Property Shortcuts Are the Core Failure Mode** (Feb 18 diagnosis)
   - The model learned texture/sharpness/noise as proxy features for real/fake
   - DF40 reals are anomalously soft (laplacian variance 38.9 vs VCD's 295.3)
   - This isn't a model architecture problem — it's a training distribution problem
   - The fix is distributional: show the model diverse real-world quality profiles

### The Checkpoint Lineage

The winning checkpoints form a chain of accumulated knowledge:

```
CLIP ViT-B-16 (pretrained)
    │
    ▼
Phase 1 B16-old ── AUC 0.9947, FPR 8.6%
    │                (config bug — was loading ViT-L-14)
    ▼
R2.5 R25_F1 ────── AUC 0.9893, FPR 1.8%
    │                (property_balanced batching breakthrough)
    ├──► R3_FT3 ─── AUC 0.9966 (holdout record, but OOD stagnant)
    │                (proved: holdout ≠ deployment)
    ▼
R4 FT7 ──────────── WMA 88.35%, DeepLive TPR 99.53%
    │                (cosine softmax + WMA data)
    ▼
R6 S1 ───────────── VCD real acc 0.698 (+15pp)
    │                (VCD reals + vcd_targeted augmentation)
    ▼
R7 ──────────────── Target: VCD real acc >0.80 (in progress)
```

Each checkpoint inherits the learned features of its parent, and each round addresses a failure mode the previous round couldn't solve. The lineage matters — R5 proved that scratch training can't match this accumulated knowledge in the same step budget.

---

*Generated Feb 23, 2026. Source documents in `DeepfakeBench/training/experiments/`.*
