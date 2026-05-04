# Critical-review synthesis (in progress, waiting for results)

**Date authored**: 2026-05-05 (during PA+PC eval wait)
**Purpose**: state current best big-picture theory and rigorously examine it against prior tests, per user's mandate to avoid "novel conclusions that conflict with what we did in the past"
**Audience**: future agent picking up this work, or user reading post-results

This document is INTERIM. It will be updated when results land with new evidence.

---

## Current best theory

> **The v2 production substrate is structurally bound by a confluence of (a) image-quality features, (b) chronic-6 real-identity FPR pollution, and (c) ~364 of 550 viso fakes that are simply unreachable by current ckpts (P8A, E2B, E3) at FPR=10% — even with per-ckpt 10% FPR budgets in loose-OR. Training-side data lever (visomaster_enhanced + visomaster_teams_enhanced) consistently fails to lift v2 viso recall at deployment τ across 3 prior tests (P14_DATA_FIX, P16, S3) — the model learns sources at training τ but loses them at deployment τ. This is the canonical τ-tail collapse pattern. PA + PC are the third clean test of this lever (FT-from-E2B, fw=4.0, no bundle confound). The EVAL-side lever (F4 substrate cleaning, dropping chronic-6 + low-resolution + no-face) lifts P8A viso 27→67%, E2B viso 8→31% — but this is not deployable because chronic-6 are EVAL-specific test identities, not production users.**

## Evidence FOR this theory

### Direct evidence

1. **Job 12 (memory `project_job12_ensemble_ceiling_2026-05-04`)**: best non-oracle ensemble of P8A+E2B+E3 caps at 16.2% viso recall. Oracle ceiling = 77%. Information IS in the streams; no label-free combiner extracts more than 16%.
2. **Job 14 (memory `project_job14_substrate_clean_2026-05-04`)**: F4 substrate cleaning lifts P8A viso 27→67%, E3 14→78%, E2B 8→31% with FPR collapsing to 0.5-3.2%. Chronic-6 axis explains 96-108% of FPR drop.
3. **PSERIES_FACTS Section 7**: viso recall has not exceeded 0.17 at any contract-equivalent τ across any P-series run. Briefly reached 0.51 at τ=0.5 (P16 step 6000), but deployment-τ collapse is universal.
4. **Job B (memory `project_job_b_findings_universal_vs_trajectory_2026-05-04`)**: same FT chain that hits 27% on v2 hits 85% on HDTF substrate (different identities, different swap-model coverage). Substrate is the binding axis.
5. **Job 11 (memory `project_chronic_offenders_partition_per_ckpt_2026-05-04`)**: each chronic-6 identity has ckpt-SPECIFIC worst-case FPR. PC_Generator__s22 spread Δ=0.85 (P8A 0.91 vs E3 0.06). Real-side FPR is ckpt-disagreement-dominated, not encoder-converged.
6. **`project_image_quality_shortcut`**: model uses sharpness as fake predictor; eval viso 4-25× less sharp than training viso → eval looks "less fake" → low recall.

### Indirect evidence

7. **`project_data_axis_lever_pulled_twice_no_lift`**: P14_DATA_FIX collapsed (fw=8.0+bundle), P16 didn't lift (fw=2.0, single-lever). The contrarian read in critical-reading note: "both confounded; not a clean test."
8. **`project_train_auc_not_valid_promotion_signal`**: P22 train AUC dropped 0.99→0.94 but recall went UP 3× — training metrics are not promotion-grade.
9. **`project_e2b_breaks_deeplive_ceiling`**: E2B's strength is deeplive (94%); viso REGRESSES to 8% — E2B does NOT improve viso, just relocates the specialization.
10. **`project_v2_substrate_is_dor_diverse_swap`**: v2 = Dor + 16 swap-model families; Dor is heavily in training as chronic FP-prone; v2 is swap-model-fresh, NOT identity-fresh.

## Evidence AGAINST / potential conflicts

### Direct conflicts to check

1. **Theory says "data lever doesn't lift at deployment τ"** — but P14_DATA_FIX prior test was bundle-confounded (anti-shortcut bundle ON, per `project_data_axis_lever_pulled_twice_no_lift` critical-reading note). If PA's clean isolation breaks the pattern, my theory is partially wrong.
2. **Theory says "PA's FT-base shift to E2B doesn't matter"** — this is a UNTESTED specific case. The 3 prior data-axis tests were FT-from-P8A. PA tests FT-from-E2B. The E2B IQ shortcut is INVERTED vs P8A (`project_iq_gating_viability_2026-05-04`); whether that affects the τ-collapse mechanism is unknown.
3. **Theory says "F4 is not deployable"** — but the F4 chronic-6 list is VALIDATION TEST IDENTITIES. In production we wouldn't see those exact identities. So the F4 numbers might be the DEPLOYMENT-RELEVANT numbers, not the F0 numbers. **This is potentially a re-framing the user might endorse.**
4. **Theory says "PC's codec aug won't lift viso"** — codec_hedge prior (analysis/policy_reruns_2026-04-27): C1 (no codec) viso=22.7%, C3 (codec aug) viso=21.5%. Slight regression. But: different params (`teams_codec_sim_p=0.60` vs PC's `policy: adaptive_mixture, p=0.5`), different FT-base (P8A vs E2B). Theory holds with caveats.

### Indirect conflicts

5. **`project_eval_substrate_reframe_2026-05-04`**: "FP tail is concentrated, NOT diffuse" — refutes calibration-loss-as-primary-lever. But my theory implies τ-tail separation (calibration-style intervention) should help. **Conflict**: if FP tail is identity-specific, calibration loss won't help; you need identity-aware intervention.
6. **`project_phase1a_method_cluster_axis_2026-05-01`**: trained heads modally align with `is_dor_shkedi`/`is_deeplive_enhanced` direction. So heads ARE learning identity-cluster axis. PA/PC's training adds more identity exposure → could shift this alignment. **Conflict**: if PA/PC shift the cluster alignment, the τ-collapse pattern might change qualitatively.

## Specific testable predictions (for PA+PC results)

### Primary (close-criterion bound)

Per `data-axis-clean-single-lever-retest-in-progress` close criterion: "lift visomaster_enhanced_macro_dev recall above the E2B_3200 baseline (8.4% F0 / 30.9% F4) at deployment-honest single-τ at 5% FPR ceiling."

- **PA F0 viso recall at FPR=10% target on `teams_real_all_dev`**: 5-15% (centered on 10%, slight lift over E2B's 8.4%, NOT enough to clear "dispositive" bar)
- **PA F4 viso recall at FPR=10% target on F4-cleaned**: 25-45% (similar to E2B's 30.9%, may marginally beat)
- **PC F0 viso recall**: PA viso ± 3pp (codec_hedge prior shows codec aug ≈ no-codec on viso)
- **PC F4 viso recall**: PA F4 viso ± 5pp

### Secondary

- **PA F0 viso at FPR=5%** (strict deployment): 1-5% (canonical τ-collapse pattern)
- **PA F0 deeplive recall at FPR=10%**: 70-90% (FT-from-E2B drift; some loss vs E2B's 94%)
- **PA lockbox real_FPR at dev-calibrated τ**: 0.5-3% (canonical drift; may violate 2% target, P22 pattern)

### What would surprise me

- **PA F4 viso > 60%**: data lever IS dispositive on cleaned substrate. Big finding.
- **PA F0 viso > 25%**: τ-collapse pattern broken. Would warrant investigating mechanism.
- **PC F0 viso > PA F0 + 10pp**: codec lever active on E2B base. Mechanism: codec aug breaks IQ shortcut differently on E2B than on P8A.

## What this implies for next-packet planning

### If verdict is (b) data lever still doesn't lift

Per AGENT_GUIDE Rule 3 (CPU-first): exhaust CPU diagnostics before GPU spend.

CPU diagnostics that would inform next packet:
1. **Score distribution comparison**: PA, PC, P8A, E2B at the same eval substrate. Are PA's scores wider (P22-style) or saturated (P8A-style)?
2. **Per-method viso recall**: which v2 swap-model families does PA catch that E2B misses?
3. **Pearson r(score, laplacian) per ckpt**: does PA inherit P8A's +0.51 viso slope or E2B's inverted slope?
4. **Identity-level FPR**: does PA fix chronic-6 or just shift to a different identity-axis?

GPU candidates (with $30 budget) ranked by CPU-validatable hypothesis:
- **Hard-negative mining packet**: weight chronic-6 reals as explicit hard negatives in training. Tests whether identity-axis FPR can be addressed training-side.
  - CPU pre-validation: confirm chronic-6 dominates FPR for E2B-like ckpts (already known per Job 11).
  - Probability of positive lift: ~50% (addresses a known cause directly)
- **Hybrid F4-substrate-substitute**: train a lightweight identity-aware filter that detects when input matches chronic-6, and switches to a per-substrate τ. Untested.
  - CPU pre-validation: characterize chronic-6 features that a filter could classify.
  - Probability of positive lift: ~70% if filter is accurate; deployability question.
- **Calibration loss / focal loss packet**: directly target τ-tail separation.
  - CPU pre-validation: characterize the distribution of "should-be-fake" scores in [0.5, 0.99] across ckpts.
  - Probability of positive lift: ~30% (no prior packet has cracked τ-tail in 10+ tries; would be novel).

Recommendation: hard-negative mining if budget permits one packet. Pre-validate identity-axis dominance with CPU job.

### If verdict is (a) data lever IS dispositive

This would be a novel finding. Pre-conditions for accepting:
- PA F0 viso > E2B F0 + 10pp (substantial lift, not noise)
- PA F4 viso > E2B F4 + 15pp (cleaned-substrate lift confirms data is the lever, not noise)
- Lockbox real FPR within target (no drift)

Next packet (per close criterion, scaling the lever):
- Test fw=8 on E2B with same single-lever discipline (confirms direction)
- Or stack with hard-negative mining (combine two productive levers)

## Reopen criteria for the data-axis claim

Per memory `project_data_axis_lever_pulled_twice_no_lift` and the in-progress loop:

- If (b): memory amends to "pulled three times" with confound caveat lifted from prior two attempts.
- The data-axis lever as a SINGLE lever for v2 viso recall is empirically dead.
- Future viso work should be ON OTHER AXES (architecture, loss, identity-aware routing, eval substrate redesign).

## Honest framing of this synthesis

This document is my interpretation. Past framings have been wrong (per OPINIONS doc disclaimer). The empirical questions:
- Do PA/PC F0 viso recall numbers conform to my "5-15%" prediction?
- Does PC differ from PA by more than 3pp on viso?
- Does F4 lift PA viso to >40%?

When data lands, I will update this doc with confirmation or refutation per axis.

## Cross-references

- `PRE_LANDING_PREDICTIONS.md` — registered before discovering codec_hedge prior
- `PRIOR_CODEC_AUG_CONTEXT.md` — codec_hedge prior surfaced post-prediction
- Memory `project_data_axis_lever_pulled_twice_no_lift` — the prior I'm extending
- Memory `project_job_b_findings_universal_vs_trajectory_2026-05-04` — adjacent finding
- Thread `viso_bucket_gap.md` — the structural thread for this work
- `docs/relaunch_handoffs/PSERIES_FACTS_2026-05-02.md` — pure data ledger
- `docs/relaunch_handoffs/AGENT_GUIDE_2026-05-02.md` — validate-before-suggest rules
