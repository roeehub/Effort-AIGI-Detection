# R6 -> R7 Robustness Recap and Execution Guide

**Date**: February 20, 2026  
**Scope**: Consolidated summary of what we learned in the R6 discussion and what to do next for robust real-video generalization (especially conferencing-like data such as VCD).

---

## 1. Why We Are Doing This

The core deployment risk is unchanged: the detector performs strongly in-distribution, but real conferencing data can still be misclassified as fake due to quality-domain shortcuts.

The objective is not just "higher average AUC." The objective is:

1. High fake detection performance.
2. High real accuracy across *different* real domains (conference codec, social video, clean captures).
3. Low sensitivity to quality/style shortcuts.

---

## 2. Key Findings From Current R6 Runs

These findings use the active R6 cohort (`R6_S1..S8_0219-2007/2008`) at ~7k-8k steps and comparison against R5.

### 2.1 What improved vs R5

1. **VCD real accuracy improved materially**.
   - R6 median: `0.6396`
   - R5 median: `0.5504`
   - Delta: `+0.0892` (+8.9pp)
2. **OOD and holdout AUC improved**.
   - OOD AUC median: R6 `0.9627` vs R5 `0.9272`
   - Holdout AUC median: R6 `0.9558` vs R5 `0.9296`

### 2.2 Tradeoffs observed

1. YouTube real accuracy dropped vs R5 (median delta about `-0.0242`).
2. In-dist AUC dropped slightly vs R5 (median delta about `-0.0058`).

Interpretation: R6 moved the boundary in the right direction for VCD but introduced a mild generalization tradeoff.

### 2.3 Plateau/rollback pattern

Across OOD checkpoints (~1k, ~3k, ~5k, ~7k), several runs peaked near ~5k then regressed on VCD by ~7k:

1. S1: `0.699 -> 0.658` (5k->7k: `-0.041`)
2. S3: `0.681 -> 0.612` (`-0.069`)
3. S6: `0.683 -> 0.613` (`-0.069`)
4. S7: `0.679 -> 0.604` (`-0.075`)
5. S8 stayed stable near peak (`0.752 -> 0.748`)

Implication: checkpoint selection must prioritize domain metrics (especially VCD), not just final-step model.

### 2.4 GRL status now

In the current 0219 cohort, GRL is active (quality loss is non-zero in GRL-enabled runs, labels/logits present).  
But stronger GRL weight alone did not consistently improve VCD.

Implication: GRL is no longer "broken," but it is not currently the dominant performance lever.

---

## 3. Method-Level Risk We Must Address

A single fake method is consistently weak: **`faceswap`**.

### 3.1 Evidence

1. R6 in-dist method accuracy for `faceswap`: mean ~`0.543` (often `0.50`).
2. R5 also showed poor `faceswap` performance on larger holdout counts:
   - mean ~`0.494` across runs with ~264-268 samples per run.
3. `faceswap` is present in training and not underrepresented (high train counts).

### 3.2 Interpretation

This is likely a chronic method-specific difficulty (or label quality / low-manipulation subset issue), not random variance.

Actionable hypothesis:

1. Treat `faceswap` as a dedicated hard group.
2. Audit likely noisy/hard subset before deciding whether to downweight, relabel, or split.

---

## 4. Data Strategy Conclusions

### 4.1 If no additional true conferencing data is available

You can still make progress with:

1. **VoxCeleb faces** (primary source): realistic video dynamics, varied quality.
2. **FFHQ faces** (secondary source): excellent identity diversity but too clean as raw input.
3. Structured conferencing-like codec augmentation ladder.

This is a strong bridge strategy, but not a perfect substitute for true conferencing captures.

### 4.2 Why this is still valid

The model currently overreacts to quality-domain signals. Adding broader real-source diversity and realistic codec perturbations expands the real manifold and reduces shortcut pressure.

---

## 5. Pairing Constraint: Practical Resolution

Concern: training often uses paired real/fake frames from the same timeline, but new data is unpaired.

Conclusion: unpaired reals are supported and valid in the current pipeline.

1. `combined_paired` already supports unpaired real samples.
2. Loss is per-sample classification (not strict pairwise contrastive-only).
3. Mixed paired + unpaired batches are already handled.

So unpaired external reals are a practical and correct addition.

---

## 6. Augmentation Policy to Avoid New Shortcuts

Yes, we should augment fakes too, but not as aggressively as reals.

### 6.1 Principle

If only reals get conferencing-like augmentations, the model can learn a new shortcut:  
"conference artifact signature = real."

### 6.2 Recommended pattern

1. Keep a clean fraction for both classes.
2. Apply shared codec-style transforms to both real and fake.
3. Apply extra real-targeted transforms on a subset (to expand real support).
4. Apply mild fake degradation on a smaller subset (to reduce quality-label coupling).
5. Avoid over-degrading fakes to the point manipulation signal is erased.

---

## 7. FFmpeg: Offline vs On-the-Fly

Question: Should full conferencing-style augmentation run on-the-fly during A100 training jobs?

Recommendation: **No** for full FFmpeg re-encoding.

Reason:

1. FFmpeg re-encode is CPU/I/O-heavy and can starve GPU utilization.
2. It increases runtime variance and startup latency.
3. It hurts throughput and reproducibility.

Use this instead:

1. Precompute codec ladder variants offline (batch preprocessing).
2. Store variants + manifest in GCS.
3. Sample those variants during training.
4. Keep only lightweight photometric/noise/sharpen transforms on-the-fly.

---

## 8. What to Optimize For (Not Just One Metric)

Use a composite objective for checkpointing/model choice:

1. Worst real-domain accuracy (VCD-like + YouTube-like + any new source).
2. OOD overall AUC.
3. Holdout AUC.
4. Weak fake method floor (`faceswap` in particular).

Practical near-term milestone:

1. `min(real-domain-acc) >= 0.80`
2. `VCD real acc >= 0.85` while avoiding major YouTube regression
3. `ood/overall/auc >= 0.96`
4. `faceswap` materially above chance

---

## 9. Immediate R7 Execution Plan

1. **Checkpoint selection first**:
   - choose by composite domain-aware metric
   - do not assume final-step checkpoint is best
2. **Data expansion**:
   - VoxCeleb as primary real source
   - FFHQ as supplemental source after realistic degradation
3. **Codec ladder preprocessing**:
   - offline FFmpeg pipeline + manifest
4. **Balanced augmentation policy**:
   - shared quality transforms across classes
   - controlled class-specific variants
5. **Faceswap hard-group workstream**:
   - isolate, audit, and handle as dedicated risk
6. **GRL stance**:
   - keep moderate settings
   - avoid betting on stronger GRL alone

---

## 10. Final Position

The R6 interventions showed real progress, especially on VCD, but also exposed a stabilization problem and method-specific fragility.

The best path forward is:

1. Better real-domain coverage (even if synthetic-conference via preprocessing).
2. Shortcut-resistant augmentation design across both classes.
3. Domain-aware checkpoint/model selection.
4. Dedicated remediation for `faceswap`.

This gives the highest probability of a robust detector under conferencing-like deployment conditions without sacrificing core fake-detection capability.
