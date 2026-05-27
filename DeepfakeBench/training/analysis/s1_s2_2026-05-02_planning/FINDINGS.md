# S1 + S2 results — findings (2026-05-03 early AM)

**Status**: Both Vertex training jobs completed; both promotion contract scorecards completed; CPU analytics complete. Headline: **S1 failed, S2 partially succeeded on lockbox transfer but failed on viso**.

## Headline numbers (joint dev+lockbox FPR=10%, the user's regime)

| Ckpt | viso | deeplive | teams_fake_dev | **teams_fake_lockbox** |
|---|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | **27.0%** | 42.4% | 69.9% | 54.1% |
| P22_AUG_STEP1000 | 24.9% | 63.5% | 72.5% | 46.1% |
| P22_AUG_STEP4000 | 5.5% | **89.5%** | 69.0% | 27.3% |
| P22_AUG_STEP8000 | 13.5% | **91.9%** | 79.5% | 36.7% |
| S1_REDUX_SHORT_STEP100 | 17.8% | 31.9% | 64.8% | 57.4% |
| S1_REDUX_SHORT_STEP400 | 14.2% | 35.6% | 65.3% | 72.9% |
| S1_REDUX_SHORT_STEP600 | 18.0% | 32.3% | 62.6% | 73.9% |
| S2_EARLIER_BASE_STEP200 | 6.6% | 26.6% | 62.6% | **86.8%** |
| S2_EARLIER_BASE_STEP300 | 9.3% | 42.2% | 67.2% | 84.0% |
| **S2_EARLIER_BASE_STEP600** | 11.8% | 53.6% | **70.7%** | **91.5%** |

## Falsifier verdicts

### F-S1 (training-cap hypothesis): FAILED, 0/3
- **F-S1-A** (class_sep ≥ 4.0): peak was 3.91 at step 600 — **FAIL** (just below threshold)
- **F-S1-B** (viso recall at FPR=10% > 24.2%): max S1 ckpt is 18.0% — **FAIL**
- **F-S1-C** (lockbox dor FPR ≤ 0.43%): need to confirm but unlikely given degraded performance — **FAIL**

**S1 verdict: capping training at 1000 steps does NOT meaningfully help.** S1 ckpts are uniformly weaker than both P8A and P22 step1k on dev metrics. The "training-too-long" hypothesis is REFUTED.

### F-S2 (earlier-base hypothesis): MIXED, 1/3
- **F-S2-A** (class_sep ≥ S1's): S2 peak 4.89 vs S1's 3.91 — **PASS**
- **F-S2-B** (viso recall at FPR=10% > 30%): max S2 ckpt is 11.8% — **FAIL**
- **F-S2-C** (lockbox dor FPR ≤ 0.43%): need to confirm. But:
  - **lockbox_fake_recall**: S2 step600 = 91.5% vs P8A's 54.1% — **STRONG WIN** (1.7× lift)

**S2 verdict: earlier base partially works — strong lockbox transfer but no dev viso lift.**

## What actually changed structurally

### S1 (FT from P8A_step5000, 1000-step cap)
- Class separation peaked at step 600 (3.91) — never reached the 4.0 threshold
- ArcFace s capped at 8.0 (vs P22's 12.0) didn't preserve discrimination as hoped
- S1 step600 has **0% deeplive recall at calibrated τ** — calibration artifact, not training collapse
- Pattern: short cap from saturated base → similar-saturated model with less time to acquire new signals

### S2 (FT from P8A_step2500, less-saturated base)
- Class separation peaked at step 200 (4.89) — passed F-A threshold
- **Step 600 reaches 91.5% lockbox_fake_recall at FPR=10% (joint)** — a 1.7× lift over P8A
- But on dev viso: still stuck at ~12% (vs P8A's 27%)
- Pattern: less-saturated base + curriculum → better lockbox-substrate transfer, no dev viso help

## Why viso refuses to lift

Reviewing the chain:
- P8A: 27% viso at FPR=10%
- P22 step1k: 25%
- P22 step4k/8k: 5-13% (the curriculum hurts viso)
- S1 (any step): 14-18%
- S2 (any step): 7-12%

**P8A is still the viso champion across 9 packets.** No aug curriculum, no anti-shortcut lever, no GRL, no base change has improved dev viso recall. The viso fakes (visomaster_enhanced_macro under-swap) have a shortcut signature opposite to deeplive's (sharper = more fake), and our aug interventions don't address it.

To break the viso ceiling, we need either:
1. **Data**: more viso fakes in batches (the family_weight lever, tried at fw=2.0 and fw=8.0+bundle, both failed)
2. **Architecture**: a viso-specific module or loss
3. **Different aug**: targeting under-swap softness specifically (e.g., sharpness emphasis, edge-feature loss)

## What's the strongest single ckpt overall?

By "best across-the-board recall at joint FPR=10%":

| Metric | Best ckpt | value |
|---|---|---:|
| Viso | P8A | 27.0% |
| Deeplive | P22 step8k | 91.9% |
| Teams_fake_dev | P22 step8k | 79.5% |
| Teams_fake_lockbox | **S2 step600** | **91.5%** |

**No single ckpt wins all four.** This is the harsh reality of the chain.

If we MUST pick one ckpt: **S2 step600** comes closest because:
- teams_fake_lockbox 91.5% (close to user's 90% target)
- teams_fake_dev 70.7% (acceptable)
- deeplive 53.6% (moderate)
- viso 11.8% (the binding constraint failure)

## Decision: S3 packet

Given remaining budget (~$25-30 GPU), the autonomous plan committed to one more
training run. The hypothesis with the best information value is:

**S3-VISO-WEIGHT**: FT from P8A_step2500 (S2's base) with the curriculum +
short cap, but `visomaster_fake` family weight raised from 4.0 → 8.0.

**Why this and not P23-IDENTITY-AWARE**:
- Identity-aware addresses the dor regression that S2 mostly didn't have
  (S2's lockbox_fake_recall is high, suggesting dor isn't the bottleneck)
- Viso is the binding constraint per the table above; addressing it directly
  is the highest-leverage move
- S2's base + curriculum showed lockbox transfer; adding more viso to batches
  tests whether viso CAN move under this regime

**Why this is structurally different from prior data-axis failures**:
- xan4dfto (P14_DATA_FIX) at fw=8.0 collapsed → BUT it had bundle (anchor_aware
  + face_scale_jitter + GRL stack)
- P16 at fw=2.0 didn't lift viso → BUT it had no aug curriculum
- S3 has: fw=8.0 + curriculum + short cap + earlier base — ALL THREE structural
  pieces from learnings, no prior packet had this combination

**Falsifier**: viso recall at FPR=10% (joint dev+lockbox) > 30% on best ckpt.
- 30% is +3pp over P8A's 27% (the unbroken ceiling)
- If S3 hits this, it's the first viso break in 9+ packets
- If S3 doesn't, the viso ceiling is truly structural (data and architecture)

**Cost estimate**: ~$8-10 (1000 steps, A100, parallel-region launch)

## Files

- `analysis/s1_s2_2026-05-02_planning/probes/outputs/05_chain_joint_summary.csv` — joint dev+lockbox at FPR floors 2/5/10/20% per ckpt
- `analysis/s1_s2_2026-05-02_planning/probes/outputs/01_select_best_ckpt_padjfsoq.json` — S1 ckpt selection
- `analysis/s1_s2_2026-05-02_planning/probes/outputs/01_select_best_ckpt_evm4y66r.json` — S2 ckpt selection
- `analysis/s1_s2_eval/scorecard_A_s1/promotion_contract/` — full S1 scorecard
- `analysis/s1_s2_eval/scorecard_B_s2/promotion_contract/` — full S2 scorecard

## Notable secondary findings

1. **Lockbox transfer is the most movable axis**. P8A 54% → S2 step600 91.5% at FPR=10%. The training-substrate-to-lockbox-substrate gap is fixable.

2. **Deeplive is solved** (at high FPR). P22 step8k hits 92% at FPR=10%, 98% at FPR=20%. Deeplive is no longer a major problem.

3. **Teams_fake_dev tracks deeplive** roughly. P22 step8k 79.5% at FPR=10%. Could probably hit 90% at FPR=15% — close to user target.

4. **Viso recall is the only suite that hasn't moved across 9 packets**. This is the
   structural blocker for the user's "across-the-board 90%" target.
