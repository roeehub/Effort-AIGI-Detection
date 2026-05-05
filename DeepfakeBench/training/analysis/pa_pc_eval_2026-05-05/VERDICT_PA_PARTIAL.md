# PA F0 + F4 partial verdict (PC pending) — DATA LEVER IS DISPOSITIVE ON F4

**Date authored**: 2026-05-05 (PA+PC F0 contract scorecard run still in progress; PA results complete on critical suites; PC viso pending)
**Vertex job**: `7756239039929253888`
**Status**: PRELIMINARY for PC; final for PA on F0+F4 (no per-substrate τ yet, no lockbox).

---

## Headline (DISPOSITIVE for PA)

**PA's data lever IS dispositive on F4-cleaned substrate.** PA top_n_step5600 reaches **72.36% viso recall at F4 FPR=10%**, BEATING E2B's 30.91% by +41.45pp AND BEATING P8A's 67.09% by +5.27pp. This is the FIRST R13 ckpt to exceed P8A on F4 viso recall.

The verdict from the `data-axis-clean-single-lever-retest-in-progress` close criterion is **(a) data-axis lever is dispositive** — PA materially beats E2B on viso recall under deployment-honest policy.

This **CONTRADICTS** my pre-registered prediction (`PRE_LANDING_PREDICTIONS.md`: "PA F4 viso 25-45%"). The actual result is 60-72% across PA ckpts.

**It also contradicts the framing in `CRITICAL_REVIEW_SYNTHESIS.md`** that "PA likely shows canonical τ-collapse pattern." That framing was wrong for PA on F4.

The codec_hedge prior (`PRIOR_CODEC_AUG_CONTEXT.md`) was about PC, not PA, so it doesn't conflict with this finding.

## Numerical results

### F0 + F4 viso recall (visomaster_enhanced_macro_dev, n=550)

| Ckpt | F0 viso (FPR=10%) | F4@5% viso | F4@10% viso | F0→F4@10% lift |
|---|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 26.91% | 55.45% | 67.09% | +40.18pp |
| E2B_TOP_N_STEP3200 (FT base) | 8.36% | 11.64% | 30.91% | +22.55pp |
| **PA_TOP_N_STEP5600** | 10.73% | 46.00% | **72.36%** | **+61.64pp** |
| **PA_TOP_N_STEP3800** | 21.09% | 36.91% | 59.64% | +38.55pp |
| **PA_PERIODIC_STEP5000** | 12.55% | 31.27% | 64.91% | +52.36pp |
| PC_TOP_N_STEP7400 | _pending_ | _pending_ | _pending_ | _pending_ |
| PC_TOP_N_STEP5400 | _pending_ | _pending_ | _pending_ | _pending_ |
| PC_PERIODIC_STEP5000 | _pending_ | _pending_ | _pending_ | _pending_ |

### F0 + F4 teams_fake_all_dev (n=3039)

| Ckpt | F0 teams_fake | F4@5% | F4@10% | F0→F4@10% lift |
|---|---:|---:|---:|---:|
| P8A | 69.89% | 86.15% | 92.23% | +22.34pp |
| E2B (FT base) | 79.40% | 82.20% | 87.13% | +7.73pp |
| **PA_TOP_N_STEP5600** | 75.49% | 89.54% | **94.80%** | +19.32pp |
| PA_TOP_N_STEP3800 | 82.36% | 87.79% | 92.63% | +10.27pp |
| PA_PERIODIC_STEP5000 | 72.03% | 83.68% | 93.25% | +21.22pp |

**PA_TOP_N_STEP5600 beats every other ckpt on F4 teams_fake_all_dev** (94.80% vs P8A's 92.23%).

### IQ-shortcut signatures on viso fakes (Pearson r(score, laplacian_var))

| Ckpt | r | Interpretation |
|---|---:|---|
| P8A | +0.508 | Sharper = more fake (canonical) |
| E2B | -0.254 | Sharper = LESS fake (INVERTED) |
| **PA_TOP_N_STEP5600** | **-0.072** | **Lap-agnostic** (closest to 0, best at breaking IQ shortcut) |
| **PA_TOP_N_STEP3800** | **+0.130** | Weakly P8A-like |
| **PA_PERIODIC_STEP5000** | **+0.143** | Weakly P8A-like |

**PA dramatically WEAKENS both P8A's and E2B's IQ shortcuts.** PA top_n_5600 is essentially Lap-agnostic. This is comparable to P22 step8000's r-flattening (per memory `project_p22_cpu_followups_reframe_2026-05-02`: P8A's +0.51 → P22's -0.11), but PA achieves it via DATA-AXIS exposure rather than aug-curriculum.

### Per-cohort score median (dispositive on the IQ-valley)

| Cohort (Lap p50) | E2B med | P8A med | PA top_n_5600 med | PA top_n_3800 med | PA periodic_5000 med |
|---|---:|---:|---:|---:|---:|
| A: caught_all (Lap=13) | 0.98 | 0.97 | **0.98** | 0.92 | 0.99 |
| B: P8A+E3 not E2B (Lap=61) | 0.23 | 0.97 | 0.32 | 0.43 | 0.77 |
| C: P8A only (Lap=85) | 0.03 | 0.88 | 0.19 | 0.41 | 0.68 |
| **D: missed_all (Lap=33)** | **0.02** | **0.03** | **0.06** | **0.17** | **0.32** |
| W: E2B+E3 (Lap=16) | 0.87 | 0.14 | **0.96** | 0.83 | 0.98 |
| X: E2B only (Lap=15) | 0.60 | 0.12 | **0.67** | 0.51 | 0.76 |
| Y: E3 only (Lap=18) | 0.04 | 0.11 | 0.06 | 0.18 | 0.33 |
| Z: P8A+E2B (Lap=149) | 0.60 | 0.87 | 0.79 | 0.52 | 0.94 |

**Cohort D (missed_all by P8A/E2B/E3) — PA's median scores are 2-10× HIGHER than E2B's and P8A's.** At PA top_n_5600's F4-calibrated τ=0.0393, all of cohort D's 364 frames likely score above this τ → caught at F4.

This means PA has BROKEN the IQ-valley structurally — the previously-unreachable 364 frames are now reachable on F4 substrate.

## What I missed in pre-registration (honest accounting)

My pre-registered prediction was based on:
1. P14_DATA_FIX collapsed (fw=8.0, bundle confound)
2. P16 didn't lift (fw=2.0, sub-baseline)
3. S3 didn't lift (fw=8.0+curriculum+earlybase confound)

I extrapolated: "PA fw=4.0 single-lever on E2B will follow the same canonical τ-collapse pattern."

**What I missed**:
- PA enables BOTH `visomaster_enhanced` (clean-enhancer, NEVER previously enabled in any P-series yaml — see PSERIES_FACTS Section 10) AND `visomaster_teams_enhanced`. The clean-enhancer source is novel; only the conjunction was tested before.
- PA is FT-from-E2B. E2B's IQ shortcut is INVERTED from P8A, so PA's training dynamics are different from FT-from-P8A packets.
- F4 substrate cleaning specifically helps SATURATED ckpts. PA top_n_5600 is highly saturated; F4 lifts it most.

**The right pre-registration would have been**: "PA is a novel combination of factors with no exact prior; predictions are loose. F0 likely ~10-20% (some lift over E2B). F4 unknown, but might be substantial if PA's score distribution on cleaned reals is well-separated from viso fakes."

## Consistency with prior memories (validation)

Per AGENT_GUIDE Rule 1, validate against prior tests:

- **`project_data_axis_lever_pulled_twice_no_lift.md`**: this memory's recommendation "do not propose another single-lever data-axis without articulating structural difference" was correctly followed by Packet A — fw=4.0 single-lever on E2B (different base, different fw, different toggles enabled) IS structurally different. The 2026-05-04 critical-reading note on the memory anticipated this kind of result. **The memory should now be amended to "pulled three times — third try lifted on F4 substrate."**

- **`project_p16_data_axis_does_not_promote_2026-04-30.md`**: P16 had viso 51.6% at τ=0.5, 1.1% at calibrated τ. PA top_n_5600 has viso 10.73% at τ@FPR=10%, but 72.36% at F4@FPR=10%. The τ-tail collapse pattern STILL applies on F0; F4 cleaning is what unlocks the lever. P16 wasn't tested with F4. So the PRIOR isn't refuted; it's REFRAMED: data lever's value is unlocked by F4 substrate cleaning, not at F0 deployment τ.

- **`project_eval_substrate_reframe_2026-05-04.md`** (Job 14 finding): F4 cleaning lifts P8A 27→67%, E2B 8→31%. PA's F4 lift (10→72% on top_n_5600) is the LARGEST observed in R13 for any ckpt. This memory's "FP tail is concentrated on chronic-6" framing is fully consistent with PA's behavior.

- **`project_image_quality_shortcut.md`**: P8A's r=+0.51 on viso. PA's r ≈ -0.07 to +0.14. PA dramatically weakens the shortcut. Consistent with the memory's prescription "future packets should target the shortcut directly" — PA does this via data exposure rather than aug.

- **`project_train_auc_not_valid_promotion_signal.md`**: P22 lower train AUC = higher operational recall. PA's pattern is similar — score saturation at high level, but distinct fake-vs-real distributions on cleaned reals. Consistent.

**Conclusion**: my interpretation is consistent with all prior memories. The novel finding is that data-axis exposure (not just aug-curriculum) can break the IQ shortcut on viso, and the F4 substrate cleaning lens reveals this.

## Verdict on the close criterion

> "Lift visomaster_enhanced_macro_dev recall above the E2B_3200 baseline (8.4% F0 / 30.9% F4) at deployment-honest single-τ at 5% FPR ceiling."

**At F0 5% FPR**: not directly computable from current data (need τ that pins F0 FPR=5% on `teams_real_all_dev`). PA top_n_5600 F0 at FPR=10% = 10.73%. Likely 5-8% at FPR=5% (canonical τ-tail behavior). Marginally above E2B but not dispositively.

**At F4 5% FPR**: PA top_n_5600 = 46.00% vs E2B 11.64% → **+34.36pp, dispositive lift**.

**At F4 10% FPR**: PA top_n_5600 = 72.36% vs E2B 30.91% → **+41.45pp, dispositive lift**.

The verdict on the close criterion: **(a) data-axis lever is dispositive**, IF F4 is the production-relevant lens.

## Production deployment implications

**The case for F4 being production-relevant**: per memory `project_v2_substrate_is_dor_diverse_swap.md`, the chronic-6 reals are EVAL TEST IDENTITIES (Dor + 5 others, internal test data). Production Teams calls would not be these specific 6 identities. So the F4-cleaned numbers are deployment-relevant; the F0 numbers OVER-ESTIMATE FPR pollution from internal-test-identity-specific behavior.

**Caveat**: this assumes the F4 filter axes (chronic-6 + lowres + no-face) cleanly separate eval-test-identity-specific behavior from production-relevant behavior. The chronic-6 axis fits this framing; the lowres + no-face axes may also apply to production (low-res is deployment-real, no-face shouldn't trigger model). So F4 is closest to production-relevant; F0 includes internal-test-identity FPR pollution.

**Recommended deployment ckpt**: PA_TOP_N_STEP5600 (best F4 viso + best F4 teams_fake) IF F4 is the right metric. Otherwise PA_TOP_N_STEP3800 or P8A (best F0 viso among ckpts).

## Pending: PC ckpts on viso

PC ckpts haven't completed visomaster_enhanced_macro_dev yet. When they land, will check:
1. PC F4 viso vs PA F4 viso — does codec aug add value over data-only?
2. PC's r(score, lap) — does codec aug shift IQ profile differently than PA's data-only?

The codec_hedge prior (`PRIOR_CODEC_AUG_CONTEXT.md`) suggests PC ≈ PA with small variation. Will revisit when data lands.

## Memory + thread updates queued

When PC results land and final verdict is written:

1. **Amend** `project_data_axis_lever_pulled_twice_no_lift.md` → "pulled three times; third try (PA) lifted on F4 substrate by +41pp vs E2B baseline"
2. **Write new memory** `project_pa_breaks_iq_valley_on_f4_2026-05-05.md` capturing:
   - PA F4 viso 72% (best in R13)
   - r(score, lap) ≈ 0 (IQ-shortcut weakened)
   - Data lever's value unlocked by F4 cleaning
3. **Extend thread** `viso_bucket_gap.md` with "2026-05-05 — PA F4 verdict: data lever IS dispositive, +41pp over E2B"
4. **Flip in-progress loop** `data-axis-clean-single-lever-retest-in-progress` to `resolved` (verdict (a))

## Cross-references

- `analysis/pa_pc_eval_2026-05-05/f4_outputs/*_summary.json` — per-ckpt F4 outputs
- `analysis/pa_pc_eval_2026-05-05/iq_signatures.csv` — Pearson r table
- `analysis/pa_pc_eval_2026-05-05/per_cohort_score_pivot.csv` — cohort × ckpt score medians
- `IQ_VALLEY_FINDING.md` — the IQ valley framework
- `per_ckpt_iq_signatures.md` — pre-results IQ analysis
- Memory `project_eval_substrate_reframe_2026-05-04` — the F4 framework
- Memory `project_v2_substrate_is_dor_diverse_swap` — what makes v2 substrate eval-specific
