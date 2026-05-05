# PA dispositive verdict — handoff for user (and next agent)

**Date authored**: 2026-05-05 (during overnight autonomous work session)
**Purpose**: clear synthesis of overnight findings + recommendations for next session
**Author**: agent acting under user authorization to "drive forward" with $30 GPU budget

---

> **⚠ CRITICAL WALKBACK ADDED 2026-05-05 ~03:00 UTC**: The "PA dispositive" verdict in this handoff was REVERSED later in the same session. PA + E2B on HDTF substrate (job `4985399369189556224`) showed PA top_n_5600 catches only **7.87% on `proper_visomaster_enhanced_teams_dev` (n=1182)** vs P8A's 93.57% on the same cell. **PA's F4 v2 lift is v2-substrate-bound, likely partial memorization** (PA's `visomaster_enhanced.enabled` source overlaps with v2 substrate's identity space, Dor-dominant). For deployment, **P8A_REFERENCE_STEP5000 remains the safer choice**. See `analysis/pa_pc_eval_2026-05-05/CRITICAL_WALKBACK_PA_DOES_NOT_GENERALIZE.md` for the full walkback. The TL;DR below is HISTORICAL; read the walkback for the corrected reading.

## TL;DR (corrected 2026-05-05 ~03:00 UTC)

1. PA top_n_5600 wins F4@10% **on v2 only**: viso 72%, deeplive 100%, teams_fake 95%. **DOES NOT generalize**: on HDTF substrate, PA viso = 7.87% vs P8A's 93.57%.
2. The data lever's apparent F4 lift on v2 is partly substrate-overlap (PA trained on visomaster_enhanced which sources from v2's bucket; eval bucket shares Dor-dominant substrate per `project_v2_substrate_is_dor_diverse_swap`).
3. PC codec aug HURTS viso 35-50pp vs PA on F4 v2 (still empirically refuted on viso, regardless of substrate framing).
4. **For deployment: P8A_REFERENCE_STEP5000 remains the best generalist**. PA carries cross-distribution risk.
5. **For future packets**: cross-substrate validation MUST be part of any data-axis close criterion. F4 v2 lift alone is insufficient evidence of generalizable improvement.

## ORIGINAL TL;DR (HISTORICAL — read with walkback above)

1. **PA_TOP_N_STEP5600 is the new best R13 ckpt** — beats P8A on viso (+5pp), deeplive (+7.5pp), and teams_fake (+2.6pp) at F4 FPR=10%. First ckpt to clear all 3 fake suites simultaneously.
2. **The data lever IS dispositive** when cleanly tested — but only on F4-cleaned substrate; F0 lift is marginal because chronic-6 FPR pollution dominates F0.
3. **PC codec aug HURTS viso** by 35-50pp on F4 vs PA — codec aug as a single lever is empirically refuted.

## The deployment number

Under F4 framing (chronic-6 are eval-test-specific identities, NOT production users — per memory `project_v2_substrate_is_dor_diverse_swap.md`), PA_TOP_N_STEP5600 delivers:

| Metric | At F4 FPR=10% | At F4 FPR=5% |
|---|---:|---:|
| viso recall | **72.36%** | 46.00% |
| deeplive recall | **100.00%** | 99.45% |
| teams_fake recall | **94.80%** | 89.54% |

Compare to E2B_TOP_N_STEP3200 (FT base): viso 30.91% / 100.00% / 87.13% at F4@10%.

Compare to P8A_REFERENCE_STEP5000 (production anchor, prior best): viso 67.09% / 92.48% / 92.23% at F4@10%.

## Why PA worked when 3 prior data-axis tests failed

PA is the FIRST P-series yaml to enable `combined_paired.visomaster_enhanced.enabled=true` (clean-enhancer source, NEVER previously tested per PSERIES_FACTS-2026-05-02 Section 10). Prior tests (P14_DATA_FIX, P16, S3) only enabled `visomaster_teams_enhanced` (the conjunction). The clean-enhancer source is the load-bearing addition.

PA also: FT-from-E2B (different IQ profile vs P8A), fw=4.0 (documented baseline), no anti-shortcut bundle (single-lever discipline).

## What PA learned (mechanism)

PA's IQ-shortcut signature on viso fakes is r(score, lap) = -0.072 (Lap-agnostic), vs P8A's +0.508 / E2B's -0.254. PA broke the IQ-shortcut.

PA's score on cohort D (the 364 unreachable v2 frames at Lap p50=33, the IQ valley between P8A's high-Lap and E2B's low-Lap sweet spots) = 0.062 — 2-10× higher than P8A's 0.031 or E2B's 0.019. At F4-calibrated τ=0.039, cohort D becomes catchable.

PA lifted the score floor on midrange-Lap viso fakes — the structural IQ valley of the v2 substrate.

## What's pending (still in flight at this writeup)

1. **Lockbox suites** (PA+PC F0 contract scorecard, ~3-5h more): does PA's F4 lift survive on a different real population? If yes, the framework is robust. If no, F4 is dev-substrate-specific and weaker generalization story.
2. **PA on HDTF** (not yet launched; recommend $5-10 spend): cross-substrate test. PA on HDTF should give comparable to P8A's 93.57% if PA is a true unified detector. Tests whether PA's lift is v2-specific or generalizable.
3. **Per-substrate τ** (CPU only, blocked on lockbox suites): deployable single-τ readout.
4. **PC ckpts on remaining suites** (deeplive completing now; lockbox suites pending). PC F4 viso is dispositively worse than PA — already documented.

## Pre-registered predictions vs actual (honest accounting)

From `analysis/pa_pc_eval_2026-05-05/PRE_LANDING_PREDICTIONS.md`:

| Prediction | Result | Match? |
|---|---|:-:|
| PA F0 viso 5-15% | 10.7-21.1% | ✓ (range hit) |
| **PA F4 viso 25-45%** | **59.6-72.4%** | ✗ **WRONG** (way higher) |
| PC F0 viso ≈ PA F0 ± 3pp | PC LOWER by 5-13pp | partial wrong |
| PC F4 viso ≈ PA F4 ± 5pp | PC LOWER by 30-50pp | very wrong |
| PA inherits E2B's IQ profile | PA r=-0.07 (closer to 0 than E2B) | partial wrong |

**My pre-registration substantially under-estimated PA's F4 lift potential.** I correctly anticipated PA might marginally beat E2B on F0, but missed the F4 cleaning's amplification effect on PA's saturated score distribution. The "canonical τ-collapse" framing was wrong for F4.

What I correctly predicted: PC ≤ PA on viso (codec aug doesn't help viso). Magnitude was much larger than predicted.

## What was DISPOSITIVELY tested (factual claims)

- **The "viso ceiling structural across architectures" framing was conditional on v2 substrate.** P8A on HDTF = 93.57% on enhanced+teams (vs 27% on v2 production substrate). 3.5× recall difference purely from substrate. Job B + P8A-on-HDTF dispositively closed this.
- **F4 substrate cleaning lifts every ckpt by 22-62pp on viso.** PA top_n_5600's lift is the largest at +61.64pp (10.73% F0 → 72.36% F4). Job 14 + this run.
- **Codec aug is counter-productive for viso recall** at the proposal-as-scoped (PC) and prior-test-scope (codec_hedge). Both negative; PC is much larger magnitude.
- **PA's data lever (clean-enhancer + conjunction, fw=4.0, FT-from-E2B, single-lever) IS dispositive on F4.** First clean test of the data axis to lift in 3 attempts.

## Recommendations for next session

### CPU-first follow-ups (free)

1. **Per-substrate τ on PA top_n_5600** (when lockbox suites complete, ~1-2h): compute the deployable single-τ recall at strict/moderate/loose ceilings.
2. **PA per-method viso recall breakdown**: which swap-model families does PA catch? Confirms the data lever exposed PA to specific swap-model artifacts.
3. **PA's chronic-6 FPR characterization**: does PA fix chronic-6 at training (good) or just at F4 cleaning (less good)? Measurable from teams_real_all_dev frame_report.

### GPU experiments (≤$30 budget)

In priority order:

1. **PA_TOP_N_STEP5600 on HDTF substrate** (~$5-10): cross-substrate test. Direct extension of P8A-on-HDTF. Uses existing checkpoint_map (`pa_pc_2026-05-05.yaml`) + existing suite_manifest. Same launcher pattern as P8A-on-HDTF. Will tell us:
   - viso recall on HDTF (expected 70-90%+ if PA generalizes)
   - clean variants (expected ≥98%)
   - teams variants
   - Validates the F4-as-production-relevant story.
2. **PA + jitter@0.50 packet** (~$15-25 for one Vertex training): combine PA's data lever with jitter (per `project_face_scale_jitter_load_bearing.md`, jitter@0.50 is independently productive). Tests whether two productive levers stack.
3. **PA + chronic-6 hard-neg mining** (~$25-30): explicitly weight chronic-6 reals as hard negatives in training. Tests whether F0 → F4 gap can be closed training-side. Higher complexity (need to identify chronic-6 frames in training data), so save for a focused effort.

**Recommended first move**: launch (1) PA on HDTF when ready. $5-10. Gives confidence in deployment story before spending more.

## Memory + thread state

The following memories were updated/written in this session:

- **Amended**: `project_data_axis_lever_pulled_twice_no_lift.md` → resolved with "pulled three times; PA succeeded" amendment
- **Amended**: `project_viso_ceiling_unbroken_10_packets.md` → scoping note upgraded from inference to head-to-head measured
- **Amended**: `project_job_b_findings_universal_vs_trajectory_2026-05-04.md` → P8A on HDTF datapoint added
- **New**: `project_pa_breaks_iq_valley_on_f4_2026-05-05.md` (the dispositive finding)
- **New**: `project_pc_codec_aug_hurts_viso_2026-05-05.md` (codec lever counter-productive)

Loop `data-axis-clean-single-lever-retest-in-progress` flipped to `resolved` (verdict (a)).

OPEN_LOOPS regenerated (in-progress 4 → 3, resolved 8 → 9).

Thread `viso_bucket_gap.md` extended with full PA verdict subsection.

TIMELINE has 4 new entries for this session: 
- 2026-05-05 early morning (Job B analysis + launches)
- 2026-05-05 morning (P8A on HDTF dispositive)
- 2026-05-05 mid-morning (PA dispositive verdict)

## Analysis artifacts (key files for next agent)

- `analysis/pa_pc_eval_2026-05-05/VERDICT_FINAL_F0_F4.md` — full verdict writeup
- `analysis/pa_pc_eval_2026-05-05/IQ_VALLEY_FINDING.md` — the IQ valley framework + PA's break
- `analysis/pa_pc_eval_2026-05-05/per_ckpt_iq_signatures.md` — Pearson r per ckpt
- `analysis/pa_pc_eval_2026-05-05/per_cohort_score_pivot.csv` — cohort × ckpt score medians
- `analysis/pa_pc_eval_2026-05-05/preliminary_score_distributions.md` — score distribution shapes
- `analysis/pa_pc_eval_2026-05-05/CRITICAL_REVIEW_SYNTHESIS.md` — pre-results theory (mostly wrong on F4 magnitudes — note in `VERDICT_FINAL` honestly accounts for this)
- `analysis/pa_pc_eval_2026-05-05/PRE_LANDING_PREDICTIONS.md` — pre-registered predictions
- `analysis/pa_pc_eval_2026-05-05/PRIOR_CODEC_AUG_CONTEXT.md` — codec_hedge prior surfaced post-prediction
- `analysis/p8a_on_hdtf_2026-05-05/FINDINGS.md` — P8A on HDTF dispositive substrate finding
- `analysis/job_b_pre_rlp604_2026-05-04/FINDINGS.md` — pre-RLP6_04 chain on HDTF (companion)

## Cost summary (for budget tracking)

This session's spend (estimate):
- PA training (`26u8bn1t`, 3h25m on A100): ~$15-20
- PC training (`0ujswaad`, 3h55m on A100): ~$20-25
- P8A-on-HDTF (Vertex `4232735281465262080`, ~1.5h): ~$5-10
- PA+PC F0 contract scorecard (Vertex `7756239039929253888`, in flight ~10h ETA): ~$30-45
- Image rebuild (Cloud Build 9763bb40, 7m52s): ~$1-2

Cumulative ~$70-100 spent on Vertex/CB before "$30 GPU budget" remaining.

The "$30 GPU budget" remaining: not yet spent. Recommendation: PA on HDTF ($5-10) first, then re-evaluate.

## What I did NOT do (transparency)

- Did NOT launch any additional GPU jobs (the $30 budget remains untouched).
- Did NOT run per-substrate τ analysis on PA (blocked on lockbox suites; CPU-ready when they complete).
- Did NOT update viewer (`viewer/server.py` + `model_dashboard_runs.yaml`) per AGENT_GUIDE Rule 4. Should be done in next session.
- Did NOT compute deployable single-τ-substrate-aware numbers for PA (the dispositive deployment readout per memory `feedback_per_mode_tau_not_deployable.md`).
- Did NOT validate that chronic-6 are TRULY internal-test-specific (needed for the F4-as-production-relevant claim).

These are reasonable next-session priorities.

## A note about the user warning

The user explicitly warned: "you propose theories but conflict with prior tests. Work in a loop: propose a theory and then rigorously examine it until you have a theory that makes sense and can actually drive us forward."

My pre-registered theory ("PA likely shows canonical τ-collapse pattern") was wrong on F4. Specifically: I extrapolated from P14_DATA_FIX/P16/S3 patterns, but missed that PA's `visomaster_enhanced.enabled=true` toggle is NOVEL (PSERIES_FACTS Section 10 had this; I didn't read it carefully enough pre-prediction).

The current theory ("PA's data lever + F4 cleaning unlocks unified deployment-grade detector") is empirically supported and consistent with all prior memories. The remaining open question is whether F4 framing generalizes (lockbox + HDTF will tell us).

I tried to be honest in the pre-registered → post-results comparison; see VERDICT_FINAL_F0_F4.md "What I missed in pre-registration" section.
