# CRITICAL WALKBACK — PA does NOT generalize off v2 substrate

**Date authored**: 2026-05-05 ~03:00 UTC
**Status**: DISPOSITIVE empirical result; supersedes earlier "PA wins" framing
**Vertex job**: `4985399369189556224` (PA + E2B on HDTF substrate)

---

## Headline (REVERSE the prior verdict)

**PA_TOP_N_STEP5600 catastrophically REGRESSES on HDTF substrate.** On `proper_visomaster_enhanced_teams_dev` (n=1182), PA catches only 7.87% (93/1182) at τ=0.5 — vs P8A's 93.57% (1106/1182) on the same suite (per `analysis/p8a_on_hdtf_2026-05-05/FINDINGS.md`).

**E2B_TOP_N_STEP3200 also regresses** on HDTF: 6.94% (82/1182).

P8A retains 93.57% recall; PA + E2B both collapse to ~7%.

**The "data lever IS dispositive" verdict from `VERDICT_FINAL_F0_F4.md` is wrong as a general-purpose claim.** The verdict holds only on the v2 substrate PA was trained on. PA does NOT generalize to substrates outside its training distribution.

## The numbers (PA vs P8A vs RLP6_04 on HDTF, all at τ=0.5)

| Suite | P8A on HDTF | RLP6_04 on HDTF | **PA top_n_5600 on HDTF** | E2B on HDTF |
|---|---:|---:|---:|---:|
| proper_real_clean_dev FPR (n=1443) | 0.42% | 1.11% | 0.69% | 0.28% |
| proper_real_teams_dev FPR (n=1444) | 0.97% | 1.18% | 0.07% | 0.21% |
| proper_visomaster_teams_dev recall (n=262) | 94.27% | 93.51% | **54.96%** | 50.38% |
| **proper_visomaster_enhanced_teams_dev recall (n=1182)** | **93.57%** | 85.36% | **7.87%** | **6.94%** |
| proper_visomaster_enhanced_teams_lockbox recall (n=302) | 95.03% | 89.74% | **6.95%** | _pending_ |
| proper_fake_teams_all_dev recall (n=1444) | 93.70% | 86.84% | **16.41%** | 14.82% |

PA + E2B (FT-from-CLIP-scratch family) **collapse on HDTF teams substrate**. P8A (FT chain through R12G) holds.

## Reconciliation with the F4 v2 verdict

PA's F4 v2 viso recall = 72.36% (per `VERDICT_FINAL_F0_F4.md`). Same model, different substrate (HDTF) = 7.87% recall. **The 64.5pp gap is NOT a substrate measurement artifact — it's PA's trained pattern not generalizing.**

Possible mechanisms (not mutually exclusive):

1. **Training data leak / memorization**: PA enables `visomaster_enhanced` (clean-enhancer source) and `visomaster_teams_enhanced` (conjunction). Both source from `visomaster-enhanced-face-cropped-v2` (per `data/sources/visomaster.py`). The eval bucket `visomaster_enhanced_macro_dev` reads from `teams-faces-data-test-2914-fake-4420-real-feb-28` (different bucket, but per `project_v2_substrate_is_dor_diverse_swap.md`, v2 substrate is Dor-dominant — same identity space). PA's training likely SAW Dor-like enhanced viso during training, and the F4 v2 recall is partially memorization of identity + setup + enhancement patterns.
2. **Substrate overfitting**: PA learned v2-specific patterns (Dor's standard setup, specific 16 swap models, specific enhancement chains). HDTF has different identities, capture conditions, and possibly different enhancement chains. PA doesn't recognize HDTF enhanced+teams as "fake."
3. **Distribution shift in IQ**: per `project_image_quality_shortcut`, eval viso v2 has Lap 36-60; HDTF training viso has Lap 132-407. PA's training on v2 shifted its IQ profile to expect low-Lap fakes. HDTF (high-Lap) doesn't fit PA's learned fake-pattern.

The IQ-shortcut framing alone is insufficient: PA's r(score, lap) on v2 viso is -0.072 (Lap-agnostic). So it's not just IQ. It's something more specific.

## Implications for the verdict

The verdict on close criterion `data-axis-clean-single-lever-retest-in-progress` was (a) — data lever IS dispositive on F4 v2. **This still holds for the v2 substrate ONLY.** It does NOT support the claim "PA is a deployment-ready detector."

For production deployment:
- If production traffic = v2-like (Dor in standard setup with various swap models): PA top_n_5600 wins at 72% F4 viso.
- If production traffic = HDTF-like (varied identities + varied capture): PA wins at 7.87%, P8A wins at 93.57%.
- **In production, traffic is unknown but DEFINITELY NOT v2 (v2 is internal test data with 1 identity)**.

So in practice, P8A_REFERENCE_STEP5000 remains the more deployable detector, despite its lower F4 v2 viso recall.

## What the original F4 verdict was actually measuring

PA's F4 v2 viso recall = "PA's ability to detect fakes structurally similar to its training data, after dropping eval-test-specific identity FPR pollution." This is a CIRCULAR success — PA is good at detecting what it trained on.

P8A's F4 v2 viso recall = "P8A's ability to detect v2 fakes despite NOT having visomaster_enhanced or visomaster_teams_enhanced enabled in training." This is GENUINE generalization. P8A's 67% is more impressive than PA's 72% because P8A wasn't trained on the specific substrate.

Per `project_data_axis_lever_pulled_twice_no_lift.md` critical-reading note (now amended to resolved): the strong claim "data lever IS dispositive" rested on PA's F4 number. **The walkback**: PA's F4 number is partially circular; the model trained on substantially-similar data and now detects on a held-out slice from the same distribution. That's overfitting, not breakthrough.

## What this means for the user's "drive forward" mandate

Walking back the dispositive claim. The honest reading:

1. **The data-axis lever's value is BOUNDED by training-data-distribution-overlap with eval substrate**. PA's F4 lift on v2 is real but does NOT translate to HDTF.
2. **P8A REMAINS the best general-purpose detector** (best on HDTF; viso F4 v2 = 67% which is comparable to PA's 72%; doesn't crash on out-of-distribution substrates).
3. **Production deployment**: stick with P8A_REFERENCE_STEP5000. PA top_n_5600 carries deployment risk because we don't know how production traffic will look — if it's not v2-like, PA could be much worse than P8A.

## Memory updates queued

I need to:
1. **Walkback** `project_pa_breaks_iq_valley_on_f4_2026-05-05.md` with a critical-reading note: PA's lift is v2-substrate-specific, does NOT generalize to HDTF (PA 7.87% vs P8A 93.57% on HDTF same cell).
2. **Walkback** `project_data_axis_lever_pulled_twice_no_lift.md` resolution amendment: PA's F4 v2 lift is real BUT v2-substrate-bound, not generalized.
3. **Write new memory** `project_pa_does_not_generalize_to_hdtf_2026-05-05.md` documenting the cross-substrate test result.
4. **Re-flip** the `data-axis-clean-single-lever-retest-in-progress` loop? Actually no — the verdict on the close criterion (lift v2 viso recall above E2B baseline) is still (a) on v2. The HDTF result is a SEPARATE finding: PA's lift is v2-specific. The loop closure stands.

## Honest accounting

This is now the SECOND major reversal in this autonomous session:
- I pre-registered "PA F4 viso 25-45%" → actual 72%, surprised UP
- I post-registered "PA dispositive on F4" → HDTF data shows the lift doesn't generalize, surprised DOWN

Both reversals point to the user's warning: "you propose theories but conflict with prior tests." Specifically:
- The first reversal: I missed that PA enabled `visomaster_enhanced.enabled=true` for the first time (PSERIES_FACTS Section 10).
- The second reversal: I missed that PA's training data overlaps with v2 substrate. The F4 lift on v2 may be partially memorization.

For the user's deployment decision: P8A is the safer choice. PA's apparent gain on v2 is plausibly distribution-overlap-driven and the cross-substrate test refutes the generalization claim.

## What I should have done pre-results

Per AGENT_GUIDE Rule 1: validate-before-suggest. I should have:
1. Audited the bucket overlap: does `visomaster-enhanced-face-cropped-v2` (PA training source) overlap with `teams-faces-data-test-2914-fake-4420-real-feb-28` (v2 eval bucket) by identity?
2. Looked for ANY HDTF-or-non-v2 eval before celebrating the F4 v2 lift.

I did the second only AFTER launching PA on HDTF. The first was never done. Both are clear gaps.

## Cross-references

- `analysis/p8a_on_hdtf_2026-05-05/FINDINGS.md` — P8A on HDTF (the comparison anchor)
- `analysis/job_b_pre_rlp604_2026-05-04/FINDINGS.md` — RLP6_04 on HDTF
- `analysis/pa_pc_eval_2026-05-05/VERDICT_FINAL_F0_F4.md` — F4 v2 verdict (still correct on v2; needs scoping)
- Memory `project_pa_breaks_iq_valley_on_f4_2026-05-05.md` — needs walkback
- Memory `project_v2_substrate_is_dor_diverse_swap.md` — the substrate-overlap mechanism
