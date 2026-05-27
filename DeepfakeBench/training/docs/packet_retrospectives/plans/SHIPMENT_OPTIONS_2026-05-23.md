# Shipment Options — P8A_REFERENCE_STEP5000

**Date**: 2026-05-23 (PM)
**Status**: operator-facing decision menu
**Ckpt**: P8A_REFERENCE_STEP5000 (`analysis/manual_canary_2026-05-20/ckpts/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth`)
**Currently deployed**: T5C_PERIODIC_STEP3500 (fails per-ckpt-calibrated team-identity bar — should be switched out)

This doc captures the per-frame-τ choice for shipping P8A. The model is fixed; the τ is the runtime knob. Aggregation policy (majority vote over N frames per window) lives downstream of the model and is the user's existing implementation choice.

---

## TL;DR — REVISED 2026-05-23 PM after score-shift sensitivity analysis

**Recommendation reversed.** The supermajority-70% option (Option 0) is Pareto-optimal on the static measurement but **brittle to production score-shift**. Under the score-shift sensitivity analysis (`analysis/mv_window_weighting_2026-05-23/`), Option 0 breaks at only +0.10 score-shift on Mac-OOD cohorts; Option 1 stays robust up to +0.30.

User's specific concern: "some users in production might be at 50-70% per-frame FPR — majority vote breaks at >50%, and I don't know if the 5-human sample is representative." This is precisely the score-shift scenario. Option 1 has 3-6× more headroom against this drift.

Operating points ranked by RECOMMENDED order (per-segment metrics, N=20 frame window):

| Rank | Option | per-frame τ | Vote rule | Per-segment Xinhe recall | OOD-capture robustness (worst Mac cohort score-shift to MV-break) | Engineering |
|---:|---|---:|---|---:|---|---|
| **1** | **τ=0.59 + simple-majority** ✱ | **0.59** | flag if >50% over τ | **75%** | **robust to +0.30+ shift** on most cohorts (Roy_D breaks at +0.125) | None — direct ship |
| 2 | τ=0.20 + supermajority-70% | 0.20 | flag if >70% over τ | 84% | breaks at +0.10 on Roy_D, +0.15 on b_b_c__s2 | Vote-rule change |
| 3 | τ=0.15 + supermajority-70% | 0.15 | flag if >70% over τ | 91% | breaks at +0.05 on Roy_D, +0.10 on b_b_c__s2 | Vote-rule change |
| 4 | τ=0.10 + abstain rule | 0.10 | flag if 50-70%, abstain if >70% | 100% | requires fraction-over-τ heuristic to abstain | Abstain logic |
| 5 | Two-tier τ (0.59 confirm, 0.10 escalate) | 0.59 + 0.10 | per-segment confidence tier | Flexible | Inherits Option 1 robustness for HIGH-conf flags + escalation safety net for LOW-conf | Most engineering |

✱ **Option 1 is now the recommended ship.** It accepts the 75% Xinhe recall in exchange for robustness against production drift. Options 2-3 are upgrades that become available once production telemetry confirms your user population is well-represented by the 5-human sample.

**Key insight**: high-τ + simple-majority is robust to drift because the high τ filters out the drift-affected score tail. Low-τ + supermajority gives better static recall but lives at the edge of the score distribution where small drifts dominate.

---

## Why P8A is the model

Per `analysis/per_ckpt_tau_recal_2026-05-23/RESULTS_FACTS_2026-05-23.md` and `analysis/per_human_bootstrap_ci_2026-05-23/RESULTS_FACTS_2026-05-23.md`:

- P8A is the only ckpt that mechanically passes the team-identity bar (per-human FPR ≤5% AND per-human fake recall ≥50% at the same τ) under per-ckpt-calibrated τ.
- E2B also passes per-ckpt-calibrated, but has 35.9% Xinhe-may6 FPR at its τ (E2B's known structural fragility on Xinhe).
- T5C fails: 8σ-below-floor Xinhe recall, structural — not a calibration issue.
- Slot A v2 (CLS): definitively fails Xinhe recall (CI [0.429, 0.489] entirely below 0.50).
- Slot A v2 (face-pool): 0.1pp boundary miss on dor recall (within sample noise) BUT 15-91% Xinhe-may6 FPR depending on τ.

P8A at τ=0.59 is also 0% Xinhe-may6 FPR on both may5 and may6 cohorts (`analysis/xinhe_may6_per_ckpt_tau_2026-05-23/README.md`). No other ckpt achieves this.

---

## Option 1 — τ=0.59 (recall-conservative; RECOMMENDED)

### Operating point
- per-frame τ = 0.59 (midpoint of P8A's widest 5%/50% passing range [0.427, 0.753])
- per-human max real-FPR: 3.7% (dor)
- per-human min fake recall: 60.3% (Xinhe)
- Xinhe-may6 false-flag cohort: 0.0% (0/92 frames)

### Per-human breakdown
| Human | Real FPR | Real-FPR CI | Fake recall | Fake-recall CI |
|---|---:|---|---:|---|
| Noyn | 1.4% | — | (no fakes) | — |
| Roee_Windows | 0.9% | — | (no fakes) | — |
| Xiang | 2.2% | — | 88.1% | — |
| Xinhe | 1.3% | [0.0%, 3.8%] (n=79 small) | 60.3% | [57.4%, 63.1%] |
| dor | 3.7% | [2.3%, 5.2%] | 79.6% | [77.9%, 81.2%] |

### Majority-vote (N=20 frame window) behavior

| Cohort regime | per-frame FPR | MV failure rate |
|---|---:|---:|
| All 5 team humans, all 18 deploy-relevant cohorts | ≤12.0% (worst: dor_morning) | **0.0%** across the board |
| Mac-Roee (out-of-scope per spec) Roy_D | 43.8% | 21.9% |
| Mac-Roee bla_bla_chow__s2 | 24.7% | 0.5% |
| Mac-Roee bla_bla_chow | 10.7% | 0.0% |

### Risk profile

- **Bulletproof on in-distribution users (Windows captures, all 5 team humans).** MV failure rate 0% across every sampled cohort.
- **Reasonable on partial OOD** (Mac-Roee bla_bla_chow*): MV recovers most cases.
- **One residual risk**: Roy_D-like users (Mac webcam, certain capture conditions) — MV fails 22% of windows. If your production population includes Mac-laptop users in this regime, ~1 in 5 of their session-windows will be wrongly flagged.

### What it costs
- Recall is moderate: Xinhe fakes caught 60% per frame; dor 80%; Xiang 88%.
- With N=20 MV at majority threshold, per-segment fake-catch rate is higher than per-frame (because real fakes likely cluster strongly above τ over a window) but is not directly measured here.

---

## Option 2 — τ=0.10 + deployment-side abstain rule

### Operating point
- per-frame τ = 0.10 (low end of P8A's 15%/50% passing range [0.100, 0.753])
- per-human max real-FPR: 15.0% (dor; exactly at the cap)
- per-human min fake recall: 86.5% (Xinhe — massive lift over Option 1's 60%)
- Xinhe-may6 false-flag cohort: 4.3% (4/92 frames; under any reasonable safety ceiling)

### Per-human breakdown
| Human | Real FPR | Fake recall |
|---|---:|---:|
| Noyn | 9.5% | (no fakes) |
| Roee_Windows | 2.1% | (no fakes) |
| Xiang | 7.9% | 96.5% |
| Xinhe | 6.3% | **86.5%** |
| dor | 15.0% | 91.2% |

### Why it needs an abstain rule

Without an abstain rule, τ=0.10 breaks on OOD captures:
- Mac-Roee Roy_D: per-frame 67.7% → **MV failure 92.4%** (catastrophic)
- Mac-Roee bla_bla_chow__s2: per-frame 64.0% → **MV failure 86.5%** (catastrophic)

**The MV-safety boundary is per-frame FPR < 50%.** Once a user's per-frame FPR crosses 50%, majority vote *amplifies* the error instead of averaging it. Roy_D crosses this boundary at τ=0.10.

### The abstain rule (proposed)

At deployment, for each N-frame window:
1. Compute `frac_over_tau = fraction of frames in window with prob_fake ≥ 0.10`
2. If `frac_over_tau > 0.70`: **ABSTAIN** (output: "uncertain / no decision")
3. If `0.50 < frac_over_tau ≤ 0.70`: **FLAG** as fake
4. If `frac_over_tau ≤ 0.50`: **PASS** (real)

The abstain branch catches the Mac-OOD case (frames score uniformly high because the capture pipeline is OOD, not because the speaker is a deepfake). It converts catastrophic false-flag into "no decision" — which is a much better operational outcome.

### Risk profile

- **High recall on in-distribution users** (86%+ per-human min recall).
- **Safe on heavily OOD users** ONLY with the abstain rule. Without it, catastrophic.
- **In-distribution cohorts will occasionally hit the abstain threshold**: dor_morning at 36.7% per-frame, with N=20, has ~8% of windows where MV would fail without abstain. With the abstain rule, these become abstains, not false-flags. Cost: some real users get "uncertain" responses occasionally.

### What it costs
- Deployment-side engineering: the abstain logic + a UX for "uncertain" responses.
- Some fraction of in-distribution windows produce "uncertain" instead of "real."
- Without the abstain rule, this option is unsafe and should not be shipped.

---

## Option 3 — Two-tier τ (0.59 confirm + 0.10 escalate)

### Operating point
Use both thresholds per segment:
- If majority of frames in window ≥ 0.59: **HIGH-CONFIDENCE FAKE**
- Else if majority of frames ≥ 0.10 AND fraction < 0.70 (abstain guard): **LOW-CONFIDENCE FAKE / ESCALATE TO HUMAN REVIEW**
- Else: **PASS**

### Why it's interesting

- Captures the high-recall benefit of τ=0.10 (catches Xinhe attacks at 86.5% per-frame) without sacrificing the high-precision benefit of τ=0.59
- The "escalate to human review" branch acknowledges that low-confidence flags need a second look (rather than auto-flagging or auto-passing)
- Compliant with MODEL_GOALS NO-ENSEMBLE rule: this is "scalar policy on top of one model's score" (per MODEL_GOALS §"Single model"), not multi-model fusion

### Risk profile

- Most engineering complexity of the three options.
- Requires UX/product alignment on what "escalate to human review" means in your product.
- Robust to OOD if combined with the abstain rule.

---

## Common evidence (all three options share)

- Per-ckpt τ-recalibration: `analysis/per_ckpt_tau_recal_2026-05-23/`
- Bootstrap CIs: `analysis/per_human_bootstrap_ci_2026-05-23/`
- Xinhe-may6 sanity at per-ckpt τ: `analysis/xinhe_may6_per_ckpt_tau_2026-05-23/`
- Per-cohort majority-vote simulation: `analysis/majority_vote_robustness_2026-05-23/outputs/`
- KLIEP substrate-match (intra-team CV-separability): `analysis/kliep_substrate_match_2026-05-23/`

---

## My recommendation

**Ship Option 1 (τ=0.59) first.** Then collect production telemetry for 1-2 weeks:
1. Distribution of per-frame scores across real users
2. Distribution of per-window MV outcomes
3. Per-user frame-count and per-user score distributions (for the heavy-tail users)

If telemetry shows:
- Most users have per-frame FPR < 5% (= no MV concerns): ship as-is, consider Option 2 only if you want higher recall
- Some users have per-frame FPR 5-30% (= MV mostly works, occasional false-flags): consider Option 3 for those
- Some users have per-frame FPR > 50% (= MV breaks; Mac-OOD-like): MUST migrate to Option 2 with abstain rule before shipping anything more aggressive

The migration is reversible and the τ is a runtime knob; you're not locked in.

---

## Production telemetry I'd ask for

If you can wire telemetry into the deployment, the highest-value signals to log:

1. **Per-frame `prob_fake` distribution per user** (histogram bucketed at 0.0, 0.1, 0.2, ..., 0.9, 1.0). Tells you the deployed distribution of per-frame FPR per user — directly answers the "are some users at 50%+ per-frame FPR?" question we cannot answer from current data alone.
2. **Per-window MV outcome counts** (flagged / passed / abstained if you ship Option 2 or 3).
3. **Per-user session-count** so the histogram is per-user-normalized.
4. **Capture-pipeline metadata if available** (Windows vs Mac, integrated webcam vs external, codec hints). Lets you correlate high-FPR users with capture conditions and identify whether the Roy_D-like population is large.

---

## Caveats

1. **Xinhe real cohort n=79 is the binding sample-size limit.** Xinhe real-FPR CIs are ±5pp; if Xinhe-like users are common in your production, the true Xinhe-real-FPR could be 11% (vs the 6% point estimate at τ=0.10). Manageable but worth noting.
2. **Mac-Roee was the only OOD-capture data point available.** Whether Mac-laptop users in production behave like Roy_D, or like bla_bla_chow (the milder OOD), or like the in-distribution cohorts — we don't know. Production telemetry is the only way to find out.
3. **The MV-safety boundary (per-frame FPR < 50%) assumes N=20 frame windows with simple majority threshold.** Larger windows are more robust but slower-reacting; weighted voting (e.g., higher weight to confident scores) could shift the boundary. Your aggregation policy is downstream.
4. **All numbers in this doc come from the 6,439-frame team-identity cohort.** They don't account for the broader real-world distribution of Teams users. Production telemetry would expand and tighten these estimates.

---

## Supporting analyses

- `analysis/per_ckpt_tau_recal_2026-05-23/` — per-ckpt τ-recalibration verdicts
- `analysis/per_human_bootstrap_ci_2026-05-23/` — bootstrap CIs on per-human metrics
- `analysis/xinhe_may6_per_ckpt_tau_2026-05-23/` — Xinhe-may6 false-flag check at per-ckpt τ
- `analysis/majority_vote_robustness_2026-05-23/` — initial MV failure analysis at τ=0.10 vs τ=0.59
- `analysis/mv_window_weighting_2026-05-23/` — full τ × vote-rule grid (the basis for Option 0)
- `analysis/device_ood_detector_2026-05-23/` — CLIP-based device-OOD detector (REJECTED, identity-confounded)
- `analysis/iq_device_detector_2026-05-23/RESULTS_FACTS_2026-05-23.md` — IQ-based device-OOD detector (generalizes on REALS)
- `analysis/iq_device_detector_2026-05-23/RESULTS_FAKES_2026-05-24.md` — IQ-detector cross-check on FAKES (REJECTED for hard-abstain; see updated note below)

## Note on Option 2's abstain trigger — UPDATED 2026-05-24 after fake-frame cross-check

The abstain rule in Option 2 uses a fraction-over-τ heuristic (abstain if >70% of frames in a window cross τ). Two device-OOD detectors were investigated as **alternative or complementary** abstain triggers:

1. **CLIP-feature detector** (`analysis/device_ood_detector_2026-05-23/`): does NOT generalize — overfit to "Roee on his specific Windows laptop." All other team-humans' Windows captures classified as "Mac-like." **REJECTED.**

2. **IQ-feature detector** (`analysis/iq_device_detector_2026-05-23/`): generalizes across people (11 of 15 non-Roee in-dist cohorts correctly classified as Windows-like) but **DOES NOT generalize cleanly across reals vs fakes**. See `RESULTS_FAKES_2026-05-24.md` for the cross-check.

### IQ-abstain verdict (2026-05-24): REJECTED for hard-abstain use

Cross-check on 4,120 fake frames found:

- **fake_target_dor is bimodal** — 61% Mac-like, 18.6% extreme-Mac (margin>+3). At T=+3 abstain, **7 specific dor fake-cohorts would become COMPLETELY undetectable**: dor_shkedi__s16, dor_fake_trump_regular, dor_fake_bill_gates_regular, dor_fake_chinese_regular, dor_fake_face_1_enhanced, dor_fake_face_1_regular, dor_fake_chinese_enhanced. All 330 frames are margin>+6 AND 95-100% currently detected by P8A.
- **Per-decile structural pathology**: in decile 8 (margin +1.86 to +7.35), real-FPR is 40% AND fake-recall is 100%. The abstain rule drops both signals together — IQ-margin is **not orthogonal** to true fakeness.
- **Per-frame Spearman ρ within fakes = +0.40** (same direction as within reals, ρ=+0.35-0.52). Mac-like fakes are P8A's most-confident fake detections, not borderline cases. Abstaining on them silences P8A's strongest fake signals.
- Aggregate trade looks tolerable on the surface (40% FPR reduction for 4% recall reduction at T=+2, deploy-relevant subset), but the recall loss is **concentrated**, not uniform — total blindness to specific fake methodologies.

### Acceptable uses of the IQ signal (soft, non-abstain)

The IQ axis is still informative. Don't waste it. Acceptable uses:

1. **Operator telemetry / alert** — flag when a deployed user's frames score high IQ-margin so the operator knows the model is operating OOD on that user. Don't take automatic action.
2. **Soft down-weighting in majority vote** — instead of dropping Mac-like frames, weight them by `1 / (1 + exp(iq_margin - 2))` in MV aggregation. Marginal frames count less but aren't silenced.
3. **Per-user τ shift** — for Mac-like frames, apply τ + 0.05 so they need more confidence to flag. Preserves recall on extreme-Mac fakes while reducing FPR on extreme-Mac reals.
4. **Per-user calibration aid** — collect IQ-margin per user; if user X is persistently Mac-like, recalibrate τ for that user from longitudinal telemetry.

### Implication for Option 2

The IQ-abstain enhancement to Option 2 is **withdrawn**. Option 2 falls back to its original form (fraction-over-τ as the only abstain trigger). The original fraction-over-τ abstain is unaffected by this analysis — it's still safe to combine with τ=0.10 as the recall-aggressive option.

**This means the recommendation stays at Option 1 (τ=0.59 simple-majority) for first ship**, with Option 2 (fraction-over-τ abstain only) available as the recall-aggressive upgrade once production telemetry validates the user population.

## Sibling docs

- `TRAINING_DIRECTIONS_OPTIONS_2026-05-23.md` — original training-program plan
- `TRAINING_DIRECTIONS_REVIEW_2026-05-23.md` — independent review + working log (this session)
- `STRUCTURAL_REFRAME_PROPOSAL_2026-05-23.md` — morning structural-reframe proposal
- `STRUCTURAL_REFRAME_REVIEW_2026-05-23.md` — morning's adversarial review
