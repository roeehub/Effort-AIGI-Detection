# E2b Final Verdict — 2026-05-04 00:30 Paris

**Status:** Contract scorecard complete. Job `5524802937504661504` SUCCEEDED 22:14 UTC.

## TL;DR

**E2B_TOP_N_STEP3200 cracks the deeplive ceiling spectacularly (43% → 88% at FPR=10%, 2.04× lift) and lifts teams_fake too. Viso ceiling is NOT broken — STEP3200 viso 7% vs P8A 27% at FPR=10%.**

This is a **fundamentally different** outcome than E1 (which lost everything except teams) and changes the picture for B16 deployment options.

## Multi-FPR-floor table (apples-to-apples, same threshold grid)

```
ckpt                    FPR=2%       FPR=5%        FPR=10%
                        viso/dl/tf   viso/dl/tf    viso/dl/tf

P8A_REFERENCE_5000      1.1/2.4/37   7.8/15.0/47   27.1/42.9/64
E2B_TOP_N_STEP3200      4.2/43.3/53  4.9/70.1/64   7.1/87.5/72   ← best deeplive
E2B_STEP3000            2.6/4.4/34   3.6/43.9/52   5.1/65.3/62
E2B_STEP6000            3.5/1.1/32   5.5/22.0/47   10.6/64.2/66  ← contract winner
E2B_TOP_N_STEP4400      2.9/3.9/34   4.9/43.9/54   6.9/75.1/68
E2B_TOP_N_STEP6800      2.9/0.0/31   4.7/9.7/43    7.3/44.8/59
E2B_TOP_N_STEP7400      2.0/0.0/28   4.4/8.3/39    6.7/45.7/57
```

(viso = visomaster_enhanced_macro_dev, dl = deeplive_enhanced_dev, tf = teams_fake_all_dev — all `*_recall`)

## Key findings

1. **STEP3200 is the lone deeplive breakthrough.** Other E2b ckpts (3400, 4400, 6400, 6800, 7400) all retreated on deeplive after step3200. Classic "trained past the sweet spot" pattern.

2. **The contract winner (STEP6000) is NOT the operational best.** Contract ranks STEP6000 first because of low dev_worst_real_stress_fpr (3.25%) and zero lockbox_real_FPR. But STEP3200 dominates STEP6000 on every fake-recall suite at every FPR floor.

3. **No E2b ckpt breaks viso ceiling.** Best E2b viso at FPR=10% is STEP6000 at 10.55%, vs P8A's 27.09%. Viso ceiling is **STRUCTURAL** — not architecture, not loss, not chain-ossification.

4. **P8A is the unique viso ckpt.** Across 11+ packets in the R13 chain, only P8A breaks 20% viso at FPR=10%.

5. **STEP3200 dominates P8A on deeplive at every FPR floor.** 18× at FPR=2%, 4.7× at FPR=5%, 2.04× at FPR=10%.

## Deployment implications

We now have **two complementary B16 ckpts**:
- **P8A_step5000** — best for viso-heavy deployment (27% viso @ FPR=10%, but 43% deeplive)
- **E2B_TOP_N_STEP3200** — best for deeplive/teams deployment (88% deeplive @ FPR=10%, but 7% viso)

**Possible ensemble paths** (consistent with `project_p22_cpu_followups_reframe_2026-05-02` pattern):
- OR-rule (max score): combines both detection regions, but lifts FPR
- Min-rule: preserves FPR, drops recall
- Weighted average: tunable trade-off
- Rank-fusion: rank scores per-frame from each, average ranks

The P22 finding showed P8A+P22-step1k gave "18× viso, 51× deeplive at $0". A P8A+E2B-STEP3200 ensemble should plausibly give:
- viso ~25-27% (held by P8A)
- deeplive ~85-88% (held by STEP3200)
- teams_fake ~70-75% (both high)

## What this refutes / supports

**Refuted:**
- "B16 SCRATCH + CrossEntropy breaks the viso ceiling" — NO. Viso regresses from 27% to 7%.
- "FT-chain ossification was the binding constraint on viso" — NO. Scratch doesn't rescue viso.

**Supported / extended:**
- "Viso ceiling is structural" (`project_viso_ceiling_unbroken_10_packets`) — extended to a 12th packet.
- "B16 is structurally viable for scratch" (E2 collapsed under ArcFace, but E2b CE is clean) — confirmed.
- "Different ckpts specialize in different methods" — newly confirmed at large magnitude (43% → 88% deeplive lift in one packet).

## Open verdict

**Does L14 (E3) break the viso ceiling?** Pending. E3 trim4 is on suite 3 of 8, ETA verdict ~04:00-05:00 Paris. If L14 also doesn't lift viso, the binding viso constraint is data/eval-substrate, not encoder capacity.

## Budget

E2b scorecard: ~$15 (5.7h on us-west4 A100).
Cumulative today: ~$95 of $200.

## Ensemble analysis — CPU run, 2026-05-04 00:45 Paris

Per-frame ensemble of P8A + E2B_TOP_N_STEP3200 on pooled dev real (7609 frames), calibrated to FPR=10%, evaluated on the 3 dev fake suites:

```
strategy                     viso     deeplive   teams_fake
P8A alone                    31.1%    47.2%      72.7%
E2B_3200 alone                7.3%    88.6%      77.5%
max-rule (P8A∨STEP3200)      22.7%    68.1%      76.7%   ← best joint trade-off
percentile-rank avg           9.6%    75.6%      77.1%
z-score avg                  11.8%    71.2%      76.6%
per-ckpt OR (half-FPR each)  11.5%    70.8%      74.9%
```

**No ensemble simultaneously gets BOTH maxima.** P8A's viso strength and STEP3200's deeplive strength are score-distribution-bound — they can't be combined to give 31% viso AND 88% deeplive at FPR=10%.

**Best ensemble**: max-rule. Loses 8pp viso vs P8A, gains 21pp deeplive. Net win for "general detector" deployment.

Lockbox readout (max-rule, dev-calibrated):
- max-rule @ dev_FPR=10%: lockbox_real_FPR=3.67%, lockbox_fake_recall=58.4% (vs P8A contract 23.7%)

## Recommended next moves (for morning)

1. **If you want a single B16 model now**: P8A_step5000 for viso-priority, E2B_TOP_N_STEP3200 for deeplive-priority, max-rule ensemble for balance.
2. **Wait for E3 verdict** (ETA 04:00-05:00 Paris) before launching another GPU experiment.
3. **If E3 breaks viso**: clear path is L14-based packet portfolio (encoder capacity hypothesis confirmed).
4. **If E3 doesn't break viso**: viso bottleneck is upstream of training. Recommend pivoting to:
   - Eval substrate redesign (per `project_eval_production_crop_tightness_gap`)
   - Per-method specialization rather than monolithic training
   - Production-side ensembling (deploy P8A + STEP3200 with output max)
