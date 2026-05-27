# E2b Preliminary Verdict — 2026-05-03 23:00 Paris

> **⚠ SUPERSEDED 2026-05-04 by `E2B_FINAL_VERDICT_2026-05-04.md`.**
> The "regresses BELOW P8A on viso" headline below is correct but the headline missed that **E2b breaks the DEEPLIVE ceiling spectacularly** (43%→88% at FPR=10%). See final verdict for full multi-FPR comparison + ensemble analysis.


**Status:** Suite 6 of 8 complete (deeplive). 2 lockbox suites + contract calibration remaining.

## Apples-to-apples comparison at threshold 0.5 (uncalibrated, same suite, same threshold)

P8A vs best E2b ckpts on each suite (acc / recall as appropriate):

| Suite | P8A_REFERENCE_STEP5000 | E2B_TOP_N_STEP3200 | E2B_STEP6000 | E2B_TOP_N_STEP6400 |
|-------|------------------------|--------------------| -------------|--------------------|
| teams_real_all_dev (acc) | 0.879 (FPR 12.1%) | 0.880 (FPR 12.0%) | 0.843 (FPR 15.7%) | 0.830 (FPR 17.0%) |
| teams_real_poor_quality_dev | 0.919 | — | — | — |
| teams_real_lighting_extreme_dev | 0.889 | — | — | — |
| **teams_fake_all_dev (recall)** | **0.702** | **0.748** | **0.762** | 0.724 |
| **visomaster_enhanced_macro_dev (recall)** | **0.356** | **0.084** | **0.169** | **0.149** |
| **deeplive_enhanced_dev (recall)** | (pending) | **0.945** | **0.883** | 0.785 |

## Read

**E2b TRADES viso recall for Teams + deeplive gains.** Different specialization than P8A.

- TOP_N_STEP3200 has nearly identical real-FPR calibration to P8A (12.0% vs 12.1%), enabling direct comparison without needing contract calibration to disambiguate.
- At that matched calibration: viso 8.4% (E2b) vs 35.6% (P8A) = **~4× viso recall regression**.
- Same matched calibration: teams_fake 74.8% (E2b) vs 70.2% (P8A) = **+4.6pp teams gain**.
- Deeplive 94.5% (E2b) vs P8A pending — likely E2b also wins.

## Implication for the user's question

**"Does scratch B16 + CrossEntropy break the viso ceiling?" → NO.** It actively regresses viso, even though training was clean (val AUC 0.9944).

This refutes the "FT-chain ossification" hypothesis: P8A's accumulated FT structure is *load-bearing* for viso. Throwing it away (scratch) and rebuilding with the same aug curriculum does NOT recover viso — it recovers a *different* operating point that's better for Teams/deeplive but worse for viso.

## What this means for E3 (L14 trim4 still running)

If E3 also regresses viso → architecture/loss/data axis is not the lever; viso bottleneck is something else (data, label, eval substrate gap).

If E3 lifts viso significantly → encoder capacity (304M vs 86M) was the missing piece, and L14 should be the new base.

E3 trim4 ETA verdict: ~04:00-06:00 Paris.

## Caveats (will update when full contract scorecard lands)

- Threshold 0.5 isn't the contract τ. The contract calibrates each ckpt to a fixed real-FPR target. Calibrated comparison may shift these numbers — but the ratio of recall to real-FPR is structurally bounded, and E2b's ratio on viso is ~4× worse than P8A's at matched FPR. Calibration won't fix that.
- 2 lockbox suites remaining — E1 showed a lockbox-fake lift even when viso regressed. Possible E2b also has a lockbox lift.
- Contract scorecard ETA: 23:30-00:00 Paris.

## Honest read for tomorrow morning

**E2b is not the lever for viso.** Best-case interpretation is "alternative deployment point with better Teams/deeplive at the cost of viso." Worst-case interpretation is "scratch threw away P8A's accumulated viso-relevant structure, confirming we shouldn't restart from CLIP again."

Either way: **the viso ceiling is not an architecture or loss problem on B16.** The remaining axes are:
1. Encoder capacity (E3 L14 — pending verdict)
2. Data axis revisited with structurally different lever than P14_DATA_FIX
3. Eval substrate change (the production crops are tighter than eval crops; per `project_eval_production_crop_tightness_gap`)
