# Round 9.5 Plan — Stability Bug Fix Rerun + Ablations

**Date:** March 2, 2026  
**Status:** READY TO LAUNCH  
**Predecessor:** R9 (8 runs — stability_lambda never activated due to config pipeline bug)  
**Docker image required:** 1.3.145 (must build before launching — includes stability config fix + unified threshold metrics)

---

## Why R9.5?

R9 had a **critical bug**: `stability_lambda` and `label_smoothing` never reached the Trainer due to a gap in the config pipeline. All 8 R9 runs trained with **no stability regularization and no label smoothing**, regardless of what the YAML configs specified.

We fixed this in `config_helpers.py` and `train_sweep.py` — but the fix hasn't shipped yet. R9.5 runs the exact same experiments with the fix, plus targeted ablations.

---

## Experiment Design — 6 Runs

Each experiment changes **exactly one thing** vs R95_A (the baseline). This gives clean ablation signal.

| Config | Base | What Changes | Key Question |
|--------|------|-------------|--------------|
| **R95_A** | R8_E FT | Nothing — this IS the baseline | What R9_A should have been (stability actually working) |
| **R95_B** | R8_E FT | λ=0.1, LS=0.0 (vs λ=0.3, LS=0.05) | Is light stability better than medium? |
| **R95_C** | R8_E FT | λ=0.5, LS=0.1, noise=0.03 (vs λ=0.3, LS=0.05, noise=0.02) | Is heavy stability better? Upper bound test. |
| **R95_D** | Scratch | No checkpoint, LR=2e-4, 12K steps | Does stability help scratch? (R9_C had best OOD without it) |
| **R95_E** | R8_E FT | ArcFace s: 10→18 (vs 6→12) | Was R9's gentle scale too conservative? R8 used 10→18. |
| **R95_F** | R8_E FT | df40_fake=1.0, df40_real=1.0 (vs 0.2, 0.5) | 5x DF40 weight — can we close the facedancer gap? |

### Detailed Config Matrix

| Param | R95_A | R95_B | R95_C | R95_D | R95_E | R95_F |
|-------|-------|-------|-------|-------|-------|-------|
| **LR** | 5e-5 | 5e-5 | 5e-5 | 2e-4 | 5e-5 | 5e-5 |
| **Steps** | 10K | 10K | 10K | 12K | 10K | 10K |
| **Warmup** | 500 | 500 | 500 | 1000 | 500 | 500 |
| **Checkpoint** | R8_E | R8_E | R8_E | None | R8_E | R8_E |
| **stability_lambda** | 0.3 | **0.1** | **0.5** | 0.3 | 0.3 | 0.3 |
| **label_smoothing** | 0.05 | **0.0** | **0.1** | 0.05 | 0.05 | 0.05 |
| **noise_std** | 0.02 | 0.02 | **0.03** | 0.02 | 0.02 | 0.02 |
| **ArcFace s** | 6→12 | 6→12 | 6→12 | 6→12 | **10→18** | 6→12 |
| **df40_fake wt** | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | **1.0** |
| **df40_real wt** | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | **1.0** |
| **Teams wt** | 7.0 | 7.0 | 7.0 | 7.0 | 7.0 | 7.0 |

---

## What Each Run Tells Us

1. **R95_A vs R9_A** — The impact of actually enabling stability. If R95_A >> R9_A on jitter metrics, stability was worth the effort. If AUC is similar or better, stability is free.

2. **R95_B vs R95_A** — If B wins, light stability (λ=0.1) is enough and label smoothing is unnecessary. Simpler is better.

3. **R95_C vs R95_A** — If C wins, we should push harder on stability. If C loses badly, 0.3 is near the ceiling.

4. **R95_D vs R95_A** — Scratch + stability vs FT + stability. R9_C (scratch, no stability) had the best OOD AUC (0.9801). Does adding stability make scratch even better? If so, scratch + stability may be the winning combo.

5. **R95_E vs R95_A** — ArcFace scale test. R8_E used 10→18 and performed well; R9 switched to 6→12. If E wins, we were too conservative. The downside: higher scale = more score instability (which is why stability reg exists).

6. **R95_F vs R95_A** — facedancer is at 65-68%. By giving DF40 5x more weight, we should push facedancer and e4s up significantly. The cost: less per-epoch exposure to DeepLive/Visomaster. Worth it if facedancer crosses 80%.

---

## Success Criteria

| Metric | R9_A Baseline | R95 Target | Notes |
|--------|--------------|------------|-------|
| Holdout AUC | 0.9891 | ≥0.9890 | Don't regress |
| OOD AUC | 0.9768 | ≥0.9770 | Don't regress |
| Stability loss | 0.0 (broken!) | >0.0 (any value) | Proof the fix works |
| Score jitter (youtube) | 0.0389 | <0.030 | Stability reg should reduce this |
| facedancer holdout | 68.2% | ≥75% (R95_F) | Close the gap |
| Teams edge_cases | 70% | ≥75% | Incremental improvement |
| unified EER | 3.52% | ≤3.5% | Hold the line |

---

## Pre-Launch Checklist

- [ ] Build Docker image 1.3.145 (`./dev.sh build-prod`)
- [ ] Verify `stability_lambda` appears in Trainer logs (dry-run test)
- [ ] Verify `train/loss/stability > 0` in first few steps of R95_A
- [ ] Launch all 6 experiments

---

## Runtime Estimate

- FT runs (A/B/C/E/F): ~10K steps × ~1.5s/step = ~4.2 hours each
- Scratch run (D): ~12K steps × ~1.5s/step = ~5 hours
- **Total wall time (6 parallel A100s):** ~5 hours
- **All 6 should complete in one overnight session.**
