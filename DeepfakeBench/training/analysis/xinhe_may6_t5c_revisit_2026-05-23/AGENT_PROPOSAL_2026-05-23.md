# AGENT_PROPOSAL — Xinhe may6 T5C revisit

Date: 2026-05-23. Interpretive doc to accompany `RESULTS_FACTS_2026-05-23.md`. This is the SINGLE opinion doc; opinion verbs are unconstrained here.

This probe answers the operationally-urgent question raised after the 2026-05-23 production-deployment-switch confirmation: **does T5C exhibit the may6-catastrophic regression that E2B did?**

---

## 1. Operational verdict — the headline

**T5C does NOT exhibit the E2B-may6 catastrophic regression.** It exhibits a mild, deployment-tolerable regression that is uniform across the may6 cohort but does NOT cross the deployment-mode-B threshold for the vast majority of frames.

| Cohort | metric | E2B (the 2026-05-06 worst case) | T5C (current production) | Slot A v2 CLS | P8A |
|---|---|---:|---:|---:|---:|
| may6 | mean prob_fake | **0.511** | 0.264 | 0.153 | 0.021 |
| may6 | fpr @ τ=0.50 (original claim) | **57.6%** (53/92) | 17.4% (16/92) | 4.3% (4/92) | 0% |
| may6 | fpr @ mode B (τ=0.78) | 31.5% (29/92) | **3.3% (3/92)** | 1.1% (1/92) | 0% |
| may6 | fpr @ mode C (τ=0.87) | 20.7% (19/92) | 0% | 0% | 0% |
| may6 | high-confidence flags (prob > 0.9) | 16/92 | **0/92** | 0/92 | 0/92 |
| may5 (reference) | fpr @ τ=0.5 | 1.7% | 0% | 0% | 0% |

The mode-B answer is the production-relevant one. At τ=0.78, **T5C false-flags 3 of 92 may6 frames vs E2B's 29 of 92** — T5C is ~10x better than E2B at the deployment operating point on this specific cohort.

The "did production break on Xinhe-may6" question maps directly to the mode-B fpr. **T5C is at 3.3% — over the 2% mode-B target but well under any "operationally broken" threshold.** The original E2B-was-broken framing required 31-58% fpr, which T5C does not produce.

Sub-answer to T5.1 ("does T5C show the E2B-may6 catastrophic regression?"): **NO**. Mean prob_fake is 0.264 (not "near 0.5"); fpr at τ=0.5 is 17.4% (not "near 57%"); fpr at mode B is 3.3% (above the 2% mode-B target but ~10x below E2B's mode-B fpr).

Sub-answer to T5.2/T5.3: the YES branch does not apply. The NO branch applies: **T5C is at-least-may6-stable; the structural-reframe conversation can proceed at deliberate pace without operational pressure to switch the production ckpt.**

Sub-answer to T5.4 ("what does Slot A v2 step3500 do on may6"): **materially better than T5C on this cohort**. Slot A v2 CLS shows fpr@0.5 = 4.3%, fpr@modeB = 1.1% (1/92), mean drift +0.082 vs T5C's +0.178. Slot A v2 face-pool is uninterpretable at fixed τ but its internal-calibrated drift (+0.156) is comparable to T5C's. On the may6 cohort specifically, Slot A v2 CLS appears to be the tightest deploy-grade ckpt below P8A.

---

## 2. Where T5C sits in the lineage

The 2026-05-06 may6 audit + the 2026-05-10 retest with T3 ckpts established a partial ordering of "how badly does this ckpt drift on may6":

| ckpt | may6 mean drift | fpr@0.5 may6 |
|---|---:|---:|
| P8A | +0.000 | 0% |
| T3_S1_step1500 (cached 2026-05-10) | +0.096 | 6.5% |
| **Slot A v2 CLS** (2026-05-23 new) | **+0.082** | **4.3%** |
| **T5C step3500** (2026-05-23 new) | **+0.178** | **17.4%** |
| PA_3800 (cached 2026-05-06) | +0.215 | 16.3% |
| E2B step3200 | +0.459 | 57.6% |
| T3_S1_step2500 (cached 2026-05-10) | +0.670 | 77.2% |

T5C sits roughly where PA_3800 sat — substantially better than E2B and T3_S1_step2500, substantially worse than P8A and the T3_S1_step1500 / Slot A v2 CLS pair. The "PA_3800-class drift on may6" framing is a reasonable interpretation: T5C inherits enough robustness to avoid the E2B failure mode while not closing the gap to P8A.

The CLS-pool branch of the lineage (P8A → ... → Slot A v2 CLS) appears to be the more may6-stable lineage; the periodic-training branch (T5C) shows ~2× the drift of Slot A v2 CLS at the same training-step bracket. This is one new data point, not a generalization.

---

## 3. The operational decision under no urgency

Given the NO verdict on T5C-is-may6-broken, the production-rotation question rests on the broader scorecard, NOT the Xinhe-may6 single point. The expanded team-identity readout (`analysis/team_identity_deploy_readout_expanded_2026-05-23/AGENT_PROPOSAL_2026-05-23.md`) concluded that on the 5-team-human cohort + 12 Xinhe-fake cohorts at the dual gate (real ≤5%, fake ≥50% per human at the same τ), **P8A is the only ckpt that mechanically clears both bars at mode A**, while T5C "fails real-side (dor 0.215 mode-A) and fails fake-side (Xinhe 0.493 mode-A; Xinhe 0.247 mode-B)".

That broader readout's conclusion now has to be reconciled with this probe's finding:
- The expanded readout said T5C is over the dor-mode-A real-FPR floor (0.215 vs 0.050).
- This probe says T5C is at-least-may6-stable on Xinhe reals.
- Both are true. The constraint is dor reals, not Xinhe reals.

**So the operational decision is: T5C is keepable on Xinhe-real grounds but is questionable on dor-real grounds.** The Xinhe-may6 question is closed for now (T5C is not broken on Xinhe-may6 reals). The dor-real chronic-FP question is still open in the expanded-readout doc and is the actual binding constraint, not Xinhe.

This redirects the "is T5C OK to keep deployed" question from "is Xinhe-may6 broken?" (answered: no) to "is dor-real chronic-FP broken?" (answered, in the expanded readout: yes at mode A, ~tolerable at mode B). The latter is the conversation worth having; the former is closed.

---

## 4. Where the Slot A v2 CLS finding matters

This probe shows Slot A v2 CLS is the tightest deploy-grade ckpt on may6 — fpr@0.5 of 4.3% (4/92), modeB fpr of 1.1% (1/92), drift of +0.082. This is **better than T5C on may6**, which is the inverse of the relative ranking on Slot A v2's known regressions:

- Slot A v2 CLS regressed Roy_D from 29.2% → 81.5% on Mac-Roee lockbox (`project_overnight_3_packets_2026_05_20`)
- Slot A v2 CLS introduced bla_bla_chow chronic 0% → 16.2% on Mac-Roee lockbox (`project_band_shortcut_ood_hypothesis_2026-05-16`)
- Slot A v2 CLS reduced visomaster recall by 10pp at the 9-suite contract (`project_face_pool_scorecard_pareto_2026-05-22`)

The 2026-05-23 team-identity refinement (`project_team_identities_multi_labeled_2026-05-23`) excludes Mac-Roee from the deploy gate, which neutralizes the Roy_D + bla_bla_chow regressions for deploy purposes. So Slot A v2 CLS comes back into play if Mac is out of scope. This Xinhe-may6 probe adds another datapoint that Slot A v2 CLS is, at least on this specific cohort, MORE may6-stable than T5C.

This does NOT promote Slot A v2 CLS over T5C overall — the dor-fake-side recall gap at mode B (Slot A v2 CLS dor 0.414, T5C dor higher) per the expanded readout still holds. But for the specific "Xinhe-real chronic FP" lever, Slot A v2 CLS is the strongest of the four post-P8A ckpts on this cohort.

---

## 5. Why this probe was fast and unambiguous

The 2026-05-06 audit infrastructure (raw frames + per-ckpt scoring script + cached scores) survived intact at 2026-05-23. No re-derivation, no checksum, no re-download was needed. Reproduction on P8A + E2B was bytewise-exact (every per-frame |Δprob| = 0.0000) on MPS scoring at 2026-05-23 vs CPU scoring at 2026-05-06. The two scoring backends agree to floating-point identity on this preprocessing path.

Total cost: $0 (local MPS), ~5 min wall time including analyzer. The "operational urgency to verify a 13-day-old finding under a production switch" loop closes cleanly when the original audit was instrumented well.

---

## 6. Self-correction log

**No mid-task corrections.** The reproduction check passed on the first run with bytewise-exact agreement on both P8A and E2B; the T5C/Slot A v2 numbers were obtained on a single straight-through pass.

**One subtle interpretation step.** The face-pool ckpt's fpr@0.5 of 26.7% on may5 and 91.3% on may6 looked alarming on first read. Reading `score_canary_face_pool.py:11-23` and the project memory `project_face_pool_scorecard_pareto_2026-05-22` clarified that the head was trained on CLS-pool features and substituting face-pool shifts the absolute regime; the right comparison is internal calibration (may5 p95 as the operating threshold), not absolute τ. The internal-calibrated face-pool drift is +0.156 — comparable to T5C, not catastrophic. This caveat is documented in RESULTS_FACTS §2 (caveat) and §3 (alternative readout).

**One claim in the task that this probe leaves un-answered.** The task asked about "the structural-reframe conversation can proceed at deliberate pace, and the next experiment can be designed without operational pressure." This probe closes the operational-pressure question for the **Xinhe-may6** cohort. It does NOT close it for the dor cohort, the Roee-Windows cohort, or the broader 5-team-human cohort. Those are scoped in the expanded team-identity readout, not here.

---

## 7. Memory updates and thread amendments

Suggested memory updates:

1. **Update** `project_production_is_t5c_not_e2b_2026-05-23.md`: append the may6 finding — "T5C is at-least-may6-stable: fpr@0.5 17.4%, mode-B fpr 3.3%, mean prob_fake 0.264. NOT the E2B catastrophic regression. The 'is current production broken on Xinhe-may6?' question is closed NO at 2026-05-23."

2. **Update** `project_xinhe_may6_falseflag_2026-05-06.md`: append a 2026-05-23 row to the per-ckpt table:
   ```
   | T5C_PERIODIC_STEP3500 | 0.086 | 0.264 | +0.178 | 16/92 | 6/92 | 0/92 |
   | SLOT_A_V2_CLS_STEP3500 | 0.071 | 0.153 | +0.082 | 4/92 | 1/92 | 0/92 |
   | SLOT_A_V2_FACE_POOL | 0.443 | 0.599 | +0.156 | 84/92 | 6/92 | 0/92 |
   ```
   With the note: "Slot A v2 face-pool fixed-τ readouts are uninterpretable due to head-pool/calibration mismatch; internal-calibrated may6 fpr (at per-ckpt may5 p95) is 76.1% face-pool, 42.4% Slot A v2 CLS, 46.7% T5C, 65.2% E2B, 4.3% P8A."

3. **Optional new memory**: `project_t5c_may6_stable_at_modeB_2026-05-23.md` if the operator wants this as a standalone reference. The headline would be: "T5C step3500 is at-least-may6-stable. 3/92 false-flag at mode B (τ=0.78), 16/92 at τ=0.5, mean 0.264. Closer to PA_3800 drift (+0.215) than to E2B drift (+0.459). NOT the E2B 57.6%-may6 failure mode."

Thread retrospectives potentially affected:
- `docs/packet_retrospectives/threads/processing_signature_shortcut.md` — appears to reference may6 / E2B findings; should note that the 2026-05-23 production (T5C) does not inherit the may6 failure mode.
