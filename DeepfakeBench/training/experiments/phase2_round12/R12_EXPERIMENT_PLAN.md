# R12 Experiment Plan

> **Date:** March 8, 2026
> **Experiments:** 8 (R12_A through R12_H)
> **Estimated runtime:** Scratch runs ~48–60h, FT runs ~22h
> **Docker:** Current production image (ensure R12 code changes included)

---

## Background & Motivation

### The Problem

Our EFFORT detector achieves **AUC=0.9942** on the holdout validation set (R9_D), but the EER threshold shifts dramatically in deployment: **0.45 in-distribution → 0.77 on Microsoft Teams**.  The root cause is a domain gap between training data (mostly DF40 academic datasets + DeepLive studio captures) and the production domain (Teams video calls with webcam compression, bright lighting, different color profiles).

**Key domain statistics:**
- Mean brightness: training ~104, Teams ~160
- R/B color ratio: training ~1.6, Teams ~1.2
- Codec artifacts: Teams uses webRTC/H.264 webcam pipeline vs DF40's pristine frames

### What R8–R11 Taught Us

Over ~30 experiments in R8–R11, we established:

1. **B-16 capacity bottleneck** — All 13 B-16 fine-tune runs plateaued at AUC 0.982–0.983, 1.1pp below R9_D.  Teams GhostFace-v2 detection stuck at 50% universally.  Only L-14 (256 trainable dims) broke through.  The PHASE2_SUMMARY concluded: *"B-16's rank-32 residual subspace doesn't have the capacity to hold both DF40 discrimination and Teams domain knowledge."*

2. **Stability/label smoothing definitively hurt** — R9.5 (the corrected rerun after fixing the R9 config pipeline bug) showed stability lambda monotonically degrades OOD: λ=0.0 → AUC 0.9768, λ=0.5 → AUC 0.9556.  Paradoxically, more stability → more jitter.

3. **GRL was never properly tested** — The Gradient Reversal Layer was broken throughout R6 (quality_domain_loss=0.0 in all runs).  The conclusion that "GRL didn't help" was based on a bug.

4. **Scratch beats fine-tune on generalization** — R8_E (scratch) outperformed all FT runs on VCD Real by 4.7pp.  R9_C (scratch+Teams) had the best OOD AUC at 0.9801.

5. **Seed sensitivity is alarming** — R8_E (seed 737) got 82.1% VCD Real, R8_H (seed 1337) got 69.4% — a 12.7pp gap from seed alone.

### What Changed in R12

R12 introduces targeted augmentation fixes to directly address the domain gap:

| Fix | Detail | Rationale |
|-----|--------|-----------|
| **OneOf removal** | Independent context variation transforms at p=0.15 each (was mutually exclusive) | Each training sample can now receive multiple lighting changes simultaneously |
| **CCT simulation** | Correlated Color Temperature shift 2700–8000K, p=0.15 | Directly simulates webcam white-balance variation seen in Teams |
| **Asymmetric brightness** | Range [-0.20, +0.60] (was symmetric ±0.20) | Teams is systematically brighter; training must see more brightness upward |
| **Wider gamma** | [70, 130] (was [80, 120]) | More extreme gamma correction covers webcam auto-exposure behavior |
| **Teams v2 training data** | 1,346 video pairs from Teams bucket | Direct domain exposure during training |
| **Teams OOD monitoring** | 300 real + 300 fake videos held out for deployment-domain eval | First time we can track Teams accuracy during training |
| **OOD composite checkpointing** | Saves top-3 checkpoints by hmean of AUC + OOD metrics | Previous rounds selected checkpoints only by holdout AUC |

---

## Experiment Design

### Design Principles

1. **Test one thing at a time** — Each experiment varies exactly one axis from its control
2. **Evidence-based pruning** — k=128 and label smoothing dropped based on prior evidence
3. **Reproducibility** — Seed control (R12_G) to establish variance floor
4. **FT from R8_E, not R9_D** — R8_E is the last strong scratch checkpoint before DF40-specific fine-tuning narrowed the subspace.  R9_D's only earlier checkpoints are step 500 (≈R8_E) and step 3500 (already converged); R8_E provides maximum plasticity.

### Full Experiment Matrix

| Slot | Name | Init | k | GRL | Seed | Key Question |
|:----:|------|------|:-:|:---:|:----:|-------------|
| **A** | `R12_A_scratch_aug_fix` | Scratch | 32 | — | 737 | Do R12 aug fixes alone close the domain gap? |
| **B** | `R12_B_scratch_k64` | Scratch | 64 | — | 737 | Does doubling SVD capacity help? |
| **C** | `R12_C_ft_r8e_aug_fix` | FT R8_E | 32 | — | 737 | Does FT from a plastic checkpoint + new aug help? |
| **D** | `R12_D_ft_r8e_grl` | FT R8_E | 32 | 0.1 | 737 | Does GRL help fine-tuning? (control pair for F) |
| **E** | `R12_E_grl_teams_ood` | Scratch | 32 | 0.1 | 737 | Does GRL help scratch? (first proper GRL test ever) |
| **F** | `R12_F_ft_r8e_grl` | FT R8_E | 32 | 0.1 | 737 | Does GRL help fine-tuning? (pairs with D) |
| **G** | `R12_G_scratch_seed_control` | Scratch | 32 | — | 1337 | Is R12_A's result robust to seed? (was 12.7pp gap in R8) |
| **H** | `R12_H_scratch_grl_strong` | Scratch | 32 | 0.3 | 737 | GRL dosage: is 0.3 better than 0.1? |

### Ablation Graph

```
                   R12_A (scratch, k=32, no GRL, seed 737)
                  /        |         \            \
          capacity?     GRL?       FT?         seed?
            /              |          \            \
        R12_B           R12_E       R12_C        R12_G
      (k=64)        (GRL 0.1)   (FT R8_E)   (seed 1337)
                        |            \
                    dosage?        FT+GRL?
                        |          /      \
                     R12_H     R12_D    R12_F
                   (GRL 0.3)  (control)  (pair)
```

### Training Configuration Summary

| Config | Scratch runs (A, B, E, G, H) | FT runs (C, D, F) |
|--------|-------------------------------|---------------------|
| Steps | 30,000 | 8,000 |
| LR | 2e-4 | 3e-5 |
| Warmup | 1,500 steps | 400 steps |
| ArcFace s | 10→14 (anneal over 30K) | 6→12 (anneal over 20K) |
| Checkpoint | — | R8_E `hu7cen3m` step 8000 |
| Early stop patience | 15 | 10 |

### Common Configuration (all 8 experiments)

- **Backbone:** ViT-B-16-DataComp-XL (LAION, hidden_size=512)
- **Data:** DF40 (7 methods) + DeepLive (5 strategies) + VisoMaster (9 swap models) + Teams v2 + VCD reals
- **Augmentation:** quality_targeted_family with R12 fixes (CCT, asymmetric brightness, wider gamma)
- **Sampling weights:** Teams fake=5.0, Teams real=4.0, DeepLive enhanced=3.0, VisoMaster=2.5
- **label_smoothing=0.0, stability_lambda=0.0** (definitively shown to hurt in R9.5)
- **OOD monitoring:** YouTube (200), VCD real (1200), WMA fake (1202), Teams real (300), Teams fake (300)
- **OOD composite checkpointing:** top-3 by hmean of AUC + OOD metrics

---

## What Was Dropped (and Why)

### k=128 (was R12_D)
- **Risk:** 4× more trainable params per SVD layer.  Overfitting risk without corresponding data increase.
- **Reason:** k=64 (R12_B) already probes the capacity hypothesis.  If k=64 helps, we can try k=128 in R13.  If k=64 doesn't help, k=128 won't either — the bottleneck is elsewhere.

### Label smoothing / stability lambda
- **R9.5 evidence is definitive:** λ monotonically degrades OOD (0.9768 → 0.9556). LS added nothing.
- **Stability paradox:** More stability → more jitter, not less.

### k=64 + GRL interaction (was R12_G)
- **Premature:** We need to first confirm that GRL (R12_E) and k=64 (R12_B) individually help before testing their interaction.
- **Replaced with:** Seed sensitivity control (R12_G) — a more immediately informative experiment.

### FT from R9_D step 4000
- **Problem:** R9_D is R8_E + 4000 steps of DF40-only fine-tuning. Already "spent" residual capacity on discriminating DF40 methods.
- **Only earlier checkpoints available:** step 500 (≈R8_E anyway) and step 3500 (already converged, only 500 steps before winner).
- **Solution:** FT directly from R8_E — the parent checkpoint with maximum plasticity.

---

## Key Metrics to Watch

### Primary (OOD — deployment readiness)
- **Teams OOD AUC** — NEW metric, first time tracked during training
- **OOD composite hmean** — holistic production readiness score
- **VCD Real accuracy** — real-video misclassification rate

### Secondary (in-distribution quality)
- **Holdout AUC** — must stay above 0.98 (R9_D baseline: 0.9942)
- **Per-method breakdown** — especially facedancer (floor ~60%), GhostFace-v2 (stuck at 50% on B-16)
- **EER threshold** — gap between val EER and OOD EER thresh should shrink

### Controls
- **R12_A vs R12_G** — seed sensitivity gap.  Target: <3pp (was 12.7pp in R8)
- **R12_D vs R12_F** — FT+GRL reproducibility.  Should agree within ~1pp.

---

## Checkpoint Lineage

```
CLIP ViT-B-16-DataComp (pretrained)
    │
    │  R8_E: scratch, 8000 steps, AUC 0.9925
    │  gs://training-job-outputs/phase2r8_experiments/hu7cen3m/
    │     top_n_effort_20260224_step8000_auc0.9925_eer0.0207.pth
    │
    ├── R12_A, B, E, G, H (scratch — start from CLIP pretrained)
    │
    └── R12_C, D, F (fine-tune from R8_E)
```

---

## Launch Commands

```bash
cd DeepfakeBench/training

for cfg in \
  experiments/phase2_round12/R12_A_scratch_aug_fix.yaml \
  experiments/phase2_round12/R12_B_scratch_k64.yaml \
  experiments/phase2_round12/R12_C_ft_r8e_aug_fix.yaml \
  experiments/phase2_round12/R12_D_ft_r8e_grl.yaml \
  experiments/phase2_round12/R12_E_grl_teams_ood.yaml \
  experiments/phase2_round12/R12_F_ft_r8e_grl.yaml \
  experiments/phase2_round12/R12_G_scratch_seed_control.yaml \
  experiments/phase2_round12/R12_H_scratch_grl_strong.yaml; do
  ./launch_experiment.sh -y phase2r12-experiments asia-southeast1 "$cfg"
  sleep 5
done
```

---

## Decision Framework (Post-Results)

After results come in, evaluate on this priority:

1. **Did aug fixes help?** Compare R12_A vs R11_E baseline.  If Teams OOD AUC improves >2pp, the aug fixes are working.
2. **Does GRL work?** Compare R12_E vs R12_A.  If OOD improves without >0.5pp AUC drop, GRL is validated for real.
3. **Does capacity help?** Compare R12_B vs R12_A.  If both AUC and OOD improve, k=64 is the path forward.
4. **Is FT from R8_E viable?** Compare R12_C vs R12_A.  If R12_C matches or beats on OOD with similar AUC, FT is a faster path.
5. **Are results robust?** Compare R12_G vs R12_A.  If gap <3pp, we can trust single-seed results.

**Best case:** Aug fixes + GRL both help → R13 would be k=64 + GRL 0.1 + R12 augs.
**Worst case:** Nothing helps → problem is data volume or backbone, not augmentation.
