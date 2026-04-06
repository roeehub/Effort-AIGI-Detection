# Round 13 Experiment Plan — B-16 Production Backbone + VisoMaster Enhanced

**Date:** March 14, 2026  
**Backbone:** ViT-B-16-DataComp-XL (LAION) — production backbone, all 8 runs  
**New data:** ~8,923 VisoMaster Enhanced samples (8 enhancers, ARTIFACT tier excluded)  
**Goal:** Integrate enhanced fakes into the B-16 production model. Compare scratch
training vs fine-tuning. Determine optimal weighting for enhanced + Teams data.

---

## Why This Design — 5 Scratch + 3 Fine-Tune

### Scratch vs FT: Two Complementary Strategies

**Scratch (5 runs):** Rebuild from CLIP weights. Historically wins on OOD AUC
but takes 20K+ steps. Lets us ablate enhanced weight, Teams weight, SVD capacity,
and seed sensitivity cleanly.

**Fine-tune (3 runs):** Start from R12_G champion checkpoint (0.9950 H.AUC,
0.9789 OOD AUC). Converges in ~3 epochs. The model already generalizes — we
just teach it the new enhanced fakes. Lower risk per run, faster results.

Historical context:
- FT consistently converges fast (~3 epochs) but has previously plateaued below
  scratch on OOD (R12: FT ceiling at 0.9832 vs scratch R12_G at 0.9950).
- BUT prior FT rounds didn't have significant new data to absorb. With ~8,923
  new enhanced samples as the main signal, FT may close the gap.
- R4_FT7 is the success story: FT + heavy reweighting solved WMA enhanced detection
  (21% → 99.53%). This is exactly the pattern we're repeating.

### Starting Point: R12_G (Current Champion)

| Metric | R12_G | R12_B | R12_A |
|--------|-------|-------|-------|
| **Composite** | **0.9869** | 0.9844 | 0.9816 |
| H.AUC | **0.9950** | 0.9903 | 0.9884 |
| OOD AUC | **0.9789** | 0.9787 | 0.9705 |
| VCD Real | 84.1% | **84.5%** | 76.7% |
| FD | **90.0%** | 64.3% | 78.6% |

---

## The 8 Runs

### Scratch Runs (5)

| Run | k | Seed | Enhanced Wt | Teams F/R | realpool/ext | Purpose |
|-----|---|------|-------------|-----------|-------------|---------|
| **R13_A** | 32 | 737 | 3.5 | 5.0/4.0 | 2.5/2.5 | Champion + enhanced (baseline) |
| **R13_B** | 64 | 737 | 3.5 | 5.0/4.0 | 2.5/2.5 | k=64 capacity test |
| **R13_C** | 32 | 737 | **5.0** | 5.0/4.0 | 2.5/2.5 | Enhanced weight ablation |
| **R13_D** | 32 | 737 | 3.5 | **7.0/5.5** | 2.5/2.5 | Teams weight ablation |
| **R13_G** | 32 | **1337** | 3.5 | 5.0/4.0 | 2.5/2.5 | Seed insurance (k=32) |

### Fine-Tune Runs (3) — from R12_G checkpoint

| Run | LR | Steps | Enhanced Wt | Teams F/R | realpool/ext | Purpose |
|-----|-----|-------|-------------|-----------|-------------|---------|
| **R13_FT1** | 3e-5 | 10K | 3.5 | 5.0/4.0 | 2.5/2.5 | Conservative FT baseline |
| **R13_FT2** | 5e-5 | 12K | **6.0** | 5.0/4.0 | 3.0/3.0 | Heavy enhanced FT |
| **R13_FT3** | 3e-5 | 10K | **5.0** | **7.0/5.5** | 3.0/3.0 | Production-optimized FT |

All FT runs: k=32, seed=737, ArcFace s=6→12 (gentler), warmup=400 steps.

### What's shared across ALL 8 runs (proven recipe)

- ArcFace m=0.0, compound augmentation, GammaUp p=0.15
- No GRL, no stability lambda
- VisoMaster Enhanced enabled, ARTIFACT tier excluded
- OOD composite checkpointing, Teams OOD monitoring
- Same data sources (DF40, DeepLive, VisoMaster, Enhanced, Teams, external reals)

---

## The 4 Scratch Axes

**1. VisoMaster Enhanced weight (3.5 vs 5.0) — A vs C**
- R4_FT7: upweighting enhanced to 5.0 solved WMA detection (21% → 99.53%)
- R8_E: target-heavy distribution fixed VisoMaster (69% → 97%)

**2. Teams weight (5.0/4.0 vs 7.0/5.5) — A vs D**
- R9_A: Teams at weight 5.0 → +1-2pp OOD AUC (largest single-factor gain since R6)

**3. SVD capacity k=32 vs k=64 — A vs B**
- R12_B (k=64) had best VCD Real at 84.5%, but FD only 64.3%
- Capacity benefits only visible with extended training (R12_G lesson)

**4. Seed (737 vs 1337) — A vs G**
- R8: 12.7pp VCD Real gap between seeds, same config
- R12: R12_G (seed=1337) went from worst OOD at epoch 3 to best by epoch 8

---

## Ablation Reading Guide

### Scratch Pure Isolations

| Question | Compare | What differs |
|----------|---------|-------------|
| Does 5.0 enhanced weight beat 3.5? | **A vs C** | enhanced_wt only |
| Does boosted Teams help? | **A vs D** | teams_wt only |
| Does k=64 help with enhanced data? | **A vs B** | rank only |
| Does seed matter with enhanced data? | **A vs G** | seed only |

### Scratch vs FT (the key new dimension)

| Question | Compare | What differs |
|----------|---------|-------------|
| Scratch vs FT, same data mix? | **A vs FT1** | Training mode only |
| Heavy enhanced: scratch vs FT? | **C vs FT2** | Training mode + LR + weight tuning |
| Boosted teams: scratch vs FT? | **D vs FT3** | Training mode + LR |

### FT Internal Comparisons

| Question | Compare | What differs |
|----------|---------|-------------|
| Conservative vs heavy enhanced FT? | **FT1 vs FT2** | LR + enhanced weight |
| Conservative vs production-push FT? | **FT1 vs FT3** | Enhanced + Teams weights |
| Enhanced-only vs production-optimized? | **FT2 vs FT3** | Enhanced vs Teams emphasis |

---

## Launch Commands

```bash
cd DeepfakeBench/training

# === SCRATCH RUNS (5) ===
./launch_experiment.sh -y phase2r13-experiments asia-southeast1 experiments/phase2_round13/R13_A_champion_plus_enhanced.yaml
./launch_experiment.sh -y phase2r13-experiments asia-southeast1 experiments/phase2_round13/R13_B_k64_capacity.yaml
./launch_experiment.sh -y phase2r13-experiments asia-southeast1 experiments/phase2_round13/R13_C_heavy_enhanced.yaml
./launch_experiment.sh -y phase2r13-experiments asia-southeast1 experiments/phase2_round13/R13_D_boosted_teams.yaml
./launch_experiment.sh -y phase2r13-experiments asia-southeast1 experiments/phase2_round13/R13_G_seed_insurance_k32.yaml

# === FINE-TUNE RUNS (3) — from R12_G champion ===
./launch_experiment.sh -y phase2r13-experiments asia-southeast1 experiments/phase2_round13/R13_FT1_conservative_enhanced.yaml
./launch_experiment.sh -y phase2r13-experiments asia-southeast1 experiments/phase2_round13/R13_FT2_heavy_enhanced.yaml
./launch_experiment.sh -y phase2r13-experiments asia-southeast1 experiments/phase2_round13/R13_FT3_enhanced_boosted_teams.yaml
```

---

## Monitoring Guide

### FT Runs — Watch First (fast convergence)

FT runs should show results within **2-3 epochs** (~3K-6K steps):
- Check enhanced fake per-enhancer TPR early — this is the main FT signal
- Compare FT OOD vs R12_G baseline (0.9789) — any degradation = catastrophic forgetting
- If FT1 matches or beats scratch A by step 5K, FT is the right approach for this data

### Scratch Runs — Be Patient

- **DO NOT** judge OOD before epoch 5 — R12_G was WORST at epoch 3 (0.9440)
- Compare A vs C (enhanced weight) starting epoch 4
- Seed comparison (A vs G) only meaningful after epoch 6+

### Key Metrics to Beat (R12_G)

| Metric | Target | Champion |
|--------|--------|----------|
| Composite | > 0.9869 | R12_G |
| OOD AUC | > 0.9789 | R12_G |
| VCD Real | ≥ 85% | R12_B had 84.5% |
| Teams OOD | stable or better | — |
| WMA | ≥ 95% | — |

### Post-Round Decisions

1. **FT beats scratch on OOD**: Ship the FT model — faster training, lower cost, and
   the architecture supports iterative data ingestion (add new data → quick FT cycle).
2. **Scratch beats FT on OOD**: Confirmed historical trend. Use scratch for final model
   but keep FT for rapid iteration during development.
3. **FT2 (heavy enhanced) wins**: Heavy reweighting during FT is the right pattern
   for absorbing new fake families. Use this for future data integrations.
4. **FT3 (boosted teams) wins**: Production-domain emphasis during FT is the priority.
   Production model should always FT with boosted Teams weights.
5. **A or G beats R12_G**: Enhanced data helps from scratch too. Next round can
   combine insights from both tracks.
