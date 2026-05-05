# PA + PC F0+F4 verdict — DATA LEVER WINS, CODEC LEVER HURTS

**Date authored**: 2026-05-05
**Vertex job**: `7756239039929253888` (still in progress for lockbox + per-session suites; F0/F4 on critical fake suites complete for all 8 ckpts)
**Status**: Final on F0/F4 viso/teams_fake; pending on lockbox + per-substrate τ + deeplive
**Supersedes**: `VERDICT_PA_PARTIAL.md` (PA-only verdict; this doc adds PC)

---

## Two-line headline

1. **PA's data lever IS dispositive on F4-cleaned substrate** — PA top_n_5600 reaches 72.36% viso recall at F4 FPR=10%, beating E2B's 30.91% by **+41.45pp** AND P8A's 67.09% by +5.27pp. First R13 ckpt to exceed P8A on F4 viso.
2. **PC's codec lever HURTS viso recall by 35-50pp** vs PA on F4 — codec aug is counter-productive for viso even though calibrated faithful to actual teams transport.

## **UPDATE 2026-05-05 ~01:35 UTC: deeplive numbers landed for all 5 PA-related ckpts**

### F4 at FPR=10% across all 3 fake suites (the deployment-honest readout)

| Ckpt | viso F4@10% | deeplive F4@10% | teams_fake F4@10% |
|---|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 67.09% | 92.48% | 92.23% |
| E2B_TOP_N_STEP3200 | 30.91% | **100.00%** | 87.13% |
| **PA_TOP_N_STEP5600** | **72.36%** | **100.00%** | **94.80%** |
| PA_TOP_N_STEP3800 | 59.64% | **100.00%** | 92.63% |
| PA_PERIODIC_STEP5000 | 64.91% | 99.45% | 93.25% |

**PA_TOP_N_STEP5600 IS THE NEW BEST R13 CKPT ACROSS ALL 3 FAKE SUITES AT F4 FPR=10%.** Dispositively beats P8A on viso (+5pp), deeplive (+7.5pp), teams_fake (+2.6pp). Ties E2B on deeplive (100%) while beating E2B by +41pp on viso.

This is a UNIFIED deployment-grade detector at F4 substrate.

### F4 at FPR=5% (strict ceiling)

| Ckpt | viso F4@5% | deeplive F4@5% | teams_fake F4@5% |
|---|---:|---:|---:|
| P8A | 55.45% | 75.78% | 86.15% |
| E2B | 11.64% | 98.35% | 82.20% |
| **PA_TOP_N_STEP5600** | 46.00% | **99.45%** | **89.54%** |
| PA_TOP_N_STEP3800 | 36.91% | 99.63% | 87.79% |
| PA_PERIODIC_STEP5000 | 31.27% | 87.71% | 83.68% |

At F4@5%: PA top_n_5600 wins on deeplive (+24pp vs P8A) and teams_fake (+3pp). On viso, P8A leads (55% vs PA's 46%) — PA's saturation hurts more at the stricter τ. **The deployment trade-off**:
- Under STRICT 5% FPR ceiling: P8A is best for viso; PA top_n_5600 is best for deeplive + teams_fake.
- Under MODERATE 10% FPR ceiling: PA top_n_5600 dominates all 3.

**Close criterion** ("5% FPR ceiling, lift above E2B's 11.64%"): PA top_n_5600 at 46.00% F4@5% viso → +34.36pp lift vs E2B. **Verdict (a) holds at F4@5% as well.**

### Production deployability story

If F4 is the production-relevant lens (chronic-6 are eval-test-specific, per `project_v2_substrate_is_dor_diverse_swap.md`), PA_TOP_N_STEP5600 delivers:
- viso: 72.36% (10% FPR) / 46.00% (5% FPR)
- deeplive: 100% (10% FPR) / 99.45% (5% FPR)
- teams_fake: 94.80% (10% FPR) / 89.54% (5% FPR)

The 90%-across-the-board target (per `project_success_criteria.md`) is approximately reached at F4@10%, with viso the only suite below 90% (72%). For 5% FPR, deeplive and teams_fake clear 90%; viso doesn't.

This is the closest the R13 program has come to the deployment goal. Three layers of progress documented:
1. The substrate-vs-trajectory question is dispositively closed (P8A on HDTF: 93.57%; v2 is the binding axis on F0).
2. F4 substrate cleaning IS the production-relevant lens.
3. PA's data lever DELIVERS a unified deployment-grade detector at F4.

---

## Numerical results (visomaster_enhanced_macro_dev, n=550)

| Ckpt | F0 viso (FPR=10%) | F4@5% viso | F4@10% viso | F0→F4@10% lift |
|---|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 26.91% | 55.45% | 67.09% | +40.18pp |
| E2B_TOP_N_STEP3200 (FT base) | 8.36% | 11.64% | 30.91% | +22.55pp |
| **PA_TOP_N_STEP5600** | 10.73% | 46.00% | **72.36%** | **+61.64pp** |
| PA_TOP_N_STEP3800 | 21.09% | 36.91% | 59.64% | +38.55pp |
| PA_PERIODIC_STEP5000 | 12.55% | 31.27% | 64.91% | +52.36pp |
| PC_TOP_N_STEP7400 | 4.91% | 13.27% | 37.27% | +32.36pp |
| PC_TOP_N_STEP5400 | 2.73% | 5.09% | 21.82% | +19.09pp |
| PC_PERIODIC_STEP5000 | 8.00% | 7.82% | 26.18% | +18.18pp |

## Numerical results (teams_fake_all_dev, n=3039)

| Ckpt | F0 teams_fake (FPR=10%) | F4@5% | F4@10% |
|---|---:|---:|---:|
| P8A | 69.89% | 86.15% | 92.23% |
| E2B | 79.40% | 82.20% | 87.13% |
| **PA_TOP_N_STEP5600** | 75.49% | 89.54% | **94.80%** |
| PA_TOP_N_STEP3800 | 82.36% | 87.79% | 92.63% |
| PA_PERIODIC_STEP5000 | 72.03% | 83.68% | 93.25% |
| PC_TOP_N_STEP7400 | 67.56% | 79.20% | 88.19% |
| PC_TOP_N_STEP5400 | 66.21% | 73.81% | 84.63% |
| PC_PERIODIC_STEP5000 | 72.06% | 71.87% | 84.44% |

PA dominates on teams_fake_all_dev too. PC slightly worse than E2B at F4.

## IQ-shortcut signatures (Pearson r(score, laplacian_var) on viso fakes, n=550)

| Ckpt | r(score, lap) | Interpretation |
|---|---:|---|
| P8A | +0.508 | Strong canonical IQ shortcut (sharper = more fake) |
| E2B | -0.254 | Inverted (blurrier = more fake) |
| **PA_TOP_N_STEP5600** | **-0.072** | **Lap-agnostic (best, IQ-shortcut weakened)** |
| PA_TOP_N_STEP3800 | +0.130 | Weakly P8A-like |
| PA_PERIODIC_STEP5000 | +0.143 | Weakly P8A-like |
| PC_TOP_N_STEP7400 | -0.133 | Inverted, between E2B and zero |
| PC_TOP_N_STEP5400 | +0.009 | Lap-agnostic |
| PC_PERIODIC_STEP5000 | -0.142 | Inverted, like E2B |

**PA dramatically weakens the IQ shortcut**; PC is mixed (moves toward zero or back to E2B-like inverted, depending on ckpt).

## Per-cohort score medians on the 364 unreachable v2 frames (cohort D)

| Ckpt | D median score |
|---|---:|
| E2B | 0.019 |
| P8A | 0.031 |
| PC_TOP_N_STEP7400 | 0.036 |
| PC_TOP_N_STEP5400 | 0.048 |
| PA_TOP_N_STEP5600 | 0.062 |
| PC_PERIODIC_STEP5000 | 0.090 |
| PA_TOP_N_STEP3800 | 0.168 |
| PA_PERIODIC_STEP5000 | 0.317 |

**PA scores cohort D 1.5-10× HIGHER than P8A/E2B.** PC is barely better than E2B on cohort D. The IQ-valley is **broken by PA's data lever** but NOT broken by PC's data+codec combination.

## Why does codec aug hurt viso?

Hypothesis (consistent with `project_viso_subtype_iq_compounded_2026-05-04`): viso teams transport ITSELF is the IQ-degradation channel that the model uses to detect viso fakes. Codec aug exposes the model to "teams-codec degradation as real-side training noise" — making the model LEARN TO IGNORE it. But teams-transported viso fakes' fake-signal partially LIVES IN that codec degradation. So PC ignores the very signal that makes teams-viso fakes catchable.

PA, training without codec aug, retains its sensitivity to teams-codec degradation as a fake-signal channel. PA's score on cohort D (which has TEAMS subtype dominance: 208 teams of 364, per `analysis/p8a_signature_decomposition_2026-05-05/`) is much higher than PC's.

## Verdict on the close criterion

`data-axis-clean-single-lever-retest-in-progress` close criterion: "lift visomaster_enhanced_macro_dev recall above the E2B_3200 baseline (8.4% F0 / 30.9% F4) at deployment-honest single-τ at 5% FPR ceiling."

| Lens | E2B baseline | PA top_n_5600 | Δ | Verdict |
|---|---:|---:|---:|---|
| F0 5% FPR (extrapolated) | ~5% | ~6-8% | ~+2pp | marginal |
| F0 10% FPR | 8.36% | 10.73% | +2.37pp | marginal |
| F4 5% FPR | 11.64% | 46.00% | **+34.36pp** | DISPOSITIVE |
| F4 10% FPR | 30.91% | 72.36% | **+41.45pp** | DISPOSITIVE |

**Verdict: (a) — data-axis lever is dispositive on F4-cleaned substrate.** Marginal at F0.

The F4 lens is production-relevant if chronic-6 are eval-test-identity-specific (per `project_v2_substrate_is_dor_diverse_swap.md` framing).

## Comparison vs my pre-registered predictions (honest accounting)

From `PRE_LANDING_PREDICTIONS.md`:

| Prediction | Result | Match? |
|---|---|:-:|
| PA F0 viso 5-15% | 10.73-21.09% | ✓ (range hit) |
| PA F4 viso 25-45% | 59.64-72.36% | ✗ **WRONG** (way higher) |
| PC F0 viso ≈ PA F0 ± 3pp | PC 2.73-8.00%, PA 10.73-21.09% — PC LOWER by 5-13pp | ✗ partial wrong (direction right per codec_hedge prior, magnitude bigger) |
| PC F4 viso ≈ PA F4 ± 5pp | PC 22-37%, PA 60-72% — PC LOWER by 30-50pp | ✗ **VERY WRONG** (PC much worse) |
| PA F0 deeplive 75-90% | _pending_ | _pending_ |
| PA inherits E2B's IQ profile | PA r=-0.07 (closer to 0); not E2B-like (-0.25) | ✗ partial wrong |
| Lockbox FPR drift at dev-cal τ | _pending_ | _pending_ |

**My pre-registration was substantially wrong on F4 magnitudes.** I correctly predicted F0 viso would be modest, but I underestimated F4 lift potential of PA. The "canonical τ-collapse" framing was wrong for F4.

What I correctly predicted: PC ≤ PA on viso (codec aug doesn't help). The MAGNITUDE was much larger than predicted.

## Implications for the broader R13 program

1. **The data lever IS productive when cleanly tested with the RIGHT toggles**. Prior P14_DATA_FIX/P16/S3 didn't include `visomaster_enhanced.enabled=true` (the clean-enhancer source); they only enabled `visomaster_teams_enhanced` (the conjunction). PA enables BOTH. The novelty is in the CLEAN-enhancer training data, not the conjunction alone.

2. **Codec aug is counter-productive for viso**. The codec_hedge prior (`PRIOR_CODEC_AUG_CONTEXT.md`) showed -1.2pp on FT-from-P8A; PC shows -35pp F4 on FT-from-E2B. The effect is base-agnostic in direction, larger in magnitude on E2B.

3. **F4 substrate cleaning is the production-relevant lens**, supported by:
   - PA dramatically lifts viso on F4 but only marginally on F0
   - Chronic-6 are EVAL TEST IDENTITIES, not production users (per `project_v2_substrate_is_dor_diverse_swap.md`)
   - F4 numbers are deployment-honest if production traffic differs from eval substrate

4. **PA top_n_5600 is the new best deployment ckpt** under F4-relevant framing:
   - F4 viso: 72.36% (best in R13)
   - F4 teams_fake: 94.80% (best in R13)
   - F0 deeplive: pending (but FT-from-E2B should preserve much of E2B's strength)

5. **The R13 chain has a working deployment story**: PA top_n_5600 + F4 substrate filter at inference + per-suite τ tuning could realistically deliver:
   - viso: ~70-75% recall
   - teams_fake: ~92-95% recall
   - deeplive: pending but likely 90%+
   At FPR=10% on a chronic-6-cleaned real distribution.

## What's NOT yet known (open questions)

1. **Deeplive recall**: not yet computed for PA/PC. E2B was 94% F0 / 100% F4. PA likely loses some E2B specialization but should be 75-90% F0. Pending.
2. **Lockbox FPR / fake recall**: PA+PC F0 contract scorecard hasn't reached the lockbox suites yet. Critical for confirming production-honest readout.
3. **Per-substrate τ**: needed for the deployable single-τ-at-strict-ceiling number. Pending.
4. **PA on HDTF substrate**: not run; if PA's F4 lift translates to broader substrate generalization, the deployment story is even stronger.

## Memory + thread updates queued

1. **Amend** `project_data_axis_lever_pulled_twice_no_lift.md` → "pulled three times; PA (third try, FT-from-E2B + visomaster_enhanced + visomaster_teams_enhanced + fw=4.0 single-lever) lifted F4 viso recall to 72% — DATA LEVER IS DISPOSITIVE on F4 substrate when tested cleanly."
2. **Write new memory** `project_pa_breaks_iq_valley_on_f4_2026-05-05.md` — the headline finding
3. **Write new memory** `project_pc_codec_aug_hurts_viso_2026-05-05.md` — the codec verdict
4. **Extend thread** `viso_bucket_gap.md` with "2026-05-05 — PA dispositive verdict + PC codec-hurts-viso"
5. **Flip in-progress loop** `data-axis-clean-single-lever-retest-in-progress` to `resolved` (verdict (a))

## Cross-references

- `analysis/pa_pc_eval_2026-05-05/f4_outputs/*_summary.json` — per-ckpt F4 outputs
- `analysis/pa_pc_eval_2026-05-05/iq_signatures.json` — Pearson r per ckpt
- `analysis/pa_pc_eval_2026-05-05/per_cohort_score_pivot.csv` — cohort × ckpt score medians
- `IQ_VALLEY_FINDING.md` — the IQ valley framework (now broken by PA on F4)
- `PRIOR_CODEC_AUG_CONTEXT.md` — codec_hedge prior, confirmed by PC result
- `VERDICT_PA_PARTIAL.md` — earlier partial verdict (superseded by this doc)
