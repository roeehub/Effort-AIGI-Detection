# AGENT_PROPOSAL — Per-ckpt τ-recalibration on team-identity bar

Date: 2026-05-23. Interpretive doc to accompany `RESULTS_FACTS_2026-05-23.md`. Per `docs/packet_retrospectives/AGENTS.md` §"Eval-folder authoring contract": this is the SINGLE opinion doc; opinion verbs are unconstrained here.

This readout was the first CPU task in the sequence `1→2→7→3→4` from `TRAINING_DIRECTIONS_REVIEW_2026-05-23.md`. It is the prerequisite for the production-switch decision and for any interpretation of the morning's plan's "structural ceiling" framing.

---

## 1. Headline

**The "no ckpt passes the team-identity bar" framing in `TRAINING_DIRECTIONS_OPTIONS_2026-05-23.md` §1 was a τ-calibration artifact.**

Under per-ckpt fair calibration:
- **P8A passes the strict 5%/50% bar at τ ∈ [0.427, 0.753]** — a wide 0.33-unit passing range with comfortable margin (max-FPR 3.7%, min-recall 60.3% at midpoint).
- **E2B passes at τ ∈ [0.687, 0.748]** — a tight 0.06-unit range.
- **Slot A v2 face-pool misses by 0.1pp** at the boundary (dor recall 0.499 vs 0.50 floor) — within sample noise.
- **Slot A v2 CLS passes the relaxed (5%/40%) gate** at τ ∈ [0.562, 0.634].
- **T5C is structurally weak** and passes NO gate evaluated.

The plan reports "no ckpt passes both gates simultaneously at mode B" as a load-bearing input for structural-reframe urgency. That conclusion is largely an artifact of using τ=0.78 (Slot A v2's calibrated mode B) across all ckpts. The correct cross-ckpt τs differ by up to 0.22 units (P8A 0.590 vs cross-const 0.78).

---

## 2. What this changes for the plan

### 2.1 Production switch (`TRAINING_DIRECTIONS_OPTIONS §2`)

**Switch production to P8A at τ=0.59.** This is a same-day, $0, no-GPU-cost decision. The recommendation:
- **Primary**: P8A at τ=0.59 (midpoint of widest passing range; max real FPR 3.7%, min fake recall 60.3%).
- **τ-flexibility**: P8A passes the 5%/50% bar across τ ∈ [0.427, 0.753]. If product wants more recall, drop τ toward 0.43 (Xinhe recall lifts but stays under the 5% FPR cap on each human). If product wants tighter FPR, raise τ toward 0.75.
- **Wide passing range means low τ-sensitivity** — the deploy isn't fragile to small calibration drift in production.

The current production (T5C step3500) is failing per-ckpt-calibrated bars too — it cannot reach 50% Xinhe recall under a 5% FPR cap at any τ. Combined with the Xinhe-may6 elevation (17.4% @ τ=0.5; per `xinhe_may6_t5c_revisit`), there is no defensible argument for keeping T5C in production once P8A is calibrated.

### 2.2 Structural reframe urgency (`TRAINING_DIRECTIONS_OPTIONS §1, §3`)

**The structural-reframe diagnosis is weakened, but not refuted.**

Weakened how:
- The "6 weeks of no movement" framing was partly retrospective bookkeeping under a recently-defined bar with calibration that disfavored multiple ckpts. P8A (April 2026, the earliest in the comparison set) does pass.
- The "every ckpt fails" headline depended on cross-ckpt-constant τ. That doesn't make the structural-shortcut taxonomy in §3 of the proposal wrong; it does mean the gate that triggered "we are at a structural ceiling NOW" is less binding than presented.

Not refuted because:
- T5C (the most recent production ckpt) still fails per-ckpt-calibrated bars. The newer model is worse on Xinhe.
- The strict (2%/60%) gate isn't met by any ckpt. If product needs tighter operation, the plan's structural-reframe direction remains the right next step.
- Slot A v2 (anchor_aware, the post-P8A "one win") passes only the relaxed (5%/40%) gate. Its theoretical improvement over P8A on the chronic-FP axis doesn't translate to passing the strict gate; on Xinhe it under-performs P8A by ~15pp.

Net: the urgency framing in the plan should be downgraded. Week 1 GPU spend can be more exploratory and less "we must find a passing ckpt because none exists." The recommended Week 1 from the review is unchanged in shape (sequential clean attribution), but the time pressure is gone.

### 2.3 The relaxed (5%/40%) gate enables ranking nuance

Under the relaxed gate (which the team-deploy AGENT_PROPOSAL §6 explicitly flagged as user-revisable), the picture is:
- **4 of 5 ckpts pass** (P8A, E2B, Slot A v2 CLS, Slot A v2 face-pool); T5C still fails
- **Slot A v2 face-pool is the most recall-leaning** (Xinhe recall 0.801 at the relaxed best-τ) — strongest on Xinhe attacks specifically
- **P8A is the most balanced** — min fake recall 0.560 across all three humans

If the user adopts the relaxed gate, the production decision becomes a Pareto tradeoff: P8A for balance vs Slot A v2 face-pool for Xinhe-attack specialization vs E2B for status-quo-continuity.

---

## 3. The T5C structural problem persists

T5C is the only ckpt failing the relaxed gate (5%/40%). Its Xinhe fake recall at the best 5%-FPR-capped τ is 0.237 — about 2× worse than P8A and 3× worse than E2B.

This is consistent with `xinhe_may6_t5c_revisit/RESULTS §3` which showed T5C may6 FPR is 46.7% at the per-ckpt p95 threshold. T5C has elevated false-flag pressure on Xinhe-distribution frames AND under-catches Xinhe-attack frames. The score distribution is compressed in the wrong direction for Xinhe specifically — frames around 0.2-0.3 prob include both Xinhe-attack frames and dor real frames, so no τ separates them.

Mechanistically this is likely the T5C 256→1024 hidden-dim change (per `project_t6_t7_t5c_scorecard_2026-05-12`) producing a head that's more "confident" on a broader range of features but loses the Xinhe-fake discrimination axis. The fix is upstream of the head — at the encoder — which is exactly the structural-reframe direction. So T5C is not evidence against the structural diagnosis; it's evidence that the most recent FT attempt got worse on the binding human.

---

## 4. Slot A v2 face-pool: not catastrophic, just barely-misses

The team-deploy AGENT_PROPOSAL §3 says face-pool's visomaster-class dor regression "is real and load-bearing" and the Xinhe-may6 readout calls it "catastrophic" (91.3% @ τ=0.5). Per-ckpt-calibrated, the picture is more nuanced:
- At user_bar best-τ (0.681): dor recall 0.499 (0.1pp under 50% floor), Xinhe recall 0.849, Xiang recall 0.978.
- At relaxed best-τ (0.700): dor recall 0.449, Xinhe recall 0.801, Xiang recall 0.971 — passes.
- At recall_lean best-τ (0.663): dor recall 0.539, Xinhe recall 0.875, Xiang recall 0.985 — passes.

Face-pool's "failure" on the strict gate is a 0.1pp dor-recall miss, not a fundamental incapacity. It's a contender for "recall-leaning deployment" specifically because of its Xinhe-attack strength (0.849-0.875 across modes vs P8A's 0.603-0.875). The catastrophic Xinhe-may6 finding is at the wrong τ — under face-pool-calibrated τ around 0.68-0.70, may6 behavior would need separate measurement (current readout uses cross-ckpt τs that don't apply).

**Followup**: the Xinhe-may6 revisit should be re-run at face-pool's per-ckpt-calibrated τ (~0.69) to confirm the 91% number is a τ-artifact or a genuine OOD-distribution problem. This is a 5-minute CPU job using existing scored frames.

---

## 5. Recommended actions (operator picks)

### 5.1 Same-day, $0

1. **Switch production T5C → P8A at τ=0.59.** Per-ckpt calibration places this in the middle of P8A's wide passing range. Document the τ choice and the per-human breakdown from §3.1 of the FACTS doc.
2. **Adopt the per-ckpt τ-calibration discipline** for any future canary readout. Document the cross-ckpt-constant-τ comparison as the wrong default for cross-ckpt verdicts.
3. **Re-evaluate the Xinhe-may6 FPR for all 5 ckpts at per-ckpt τs** (5-min CPU job). The cross-const τ=0.5 column in the xinhe_may6 readout is unfairly comparing across ckpts with different calibration scales.

### 5.2 Decision for user

4. **Should the 5%/50% floor be relaxed to 5%/40%?** The relaxed gate has 4-of-5 ckpts passing and gives a richer operational tradeoff space. Adopting it would weaken the structural-reframe urgency but not change the structural-reframe direction.

### 5.3 Week 1 GPU plan (input to `TRAINING_DIRECTIONS_REVIEW`)

This readout argues for:
- **Lower urgency on Week 1 GPU work.** P8A passes; the deploy can ship while exploratory packets run.
- **Re-rank Week 1 candidates by EV.** With urgency lower, the OPTB-negative-result-driven case for B.II.3 (substrate-pair contrastive pretrain) becomes more attractive vs the in-paradigm levers — there's time to be deliberate.
- **The output-preservation aux loss spec doc (Task #5)** should still proceed; the spec is decoupled from urgency.

---

## 6. What this readout does NOT do

- Does not re-evaluate the 9-suite contract under per-ckpt τ. P8A passing the team-identity bar at τ=0.59 does not automatically imply it passes the conventional contract; that's a separate analysis (similar τ-sweep on different cohorts).
- Does not measure Mac-Roee (out-of-scope per spec).
- Does not score the missing dor-webcam-false-flag pools (excluded from `grouped_manifest_v2`); the production switch recommendation here would need to be cross-checked against those pools before commit if they're considered production-relevant.
- Does not reproduce the Xinhe-may6 cohort at per-ckpt τ (deferred to action 5.1.3 above).

---

## 7. Self-correction log

- **Initial framing**: I expected per-ckpt calibration to lift P8A through the gate (the morning reviewer predicted this for the 0.013pp miss) but did NOT expect 2 ckpts to pass with margin. The plan's "no ckpt passes" framing collapsed harder than I anticipated.
- **Slot A v2 face-pool 0.499 dor recall**: I initially noted this as "fails by 0.001" and considered it strictly failing. On second thought, sample noise at n=2443 (±~1pp at 95% CI) makes the 0.1pp gap genuinely indistinguishable. Reported as both (failing by mechanical bar, within noise by statistical bar).
- **T5C diagnosis**: I initially considered whether T5C's failure could also be τ-calibration. It isn't — T5C's score distribution simply doesn't separate Xinhe-fake from dor-real. The structural verdict on T5C is firm.
- **Did not investigate**: whether there exists any (ckpt, τ) cell that satisfies stricter floors like (3%, 55%) — could be checked in a follow-up if user wants finer-grained Pareto exploration.

---

## 8. Followups (TODOs for user-decided application)

These are NOT applied; per scope guard rails I am leaving them for the user.

### Memory updates

- **NEW**: `project_per_ckpt_tau_recal_reframes_team_identity_verdict_2026-05-23.md` — "Per-ckpt τ-recalibration on team-identity bar shows P8A passes strict 5%/50% at τ∈[0.427,0.753] (best τ=0.59); E2B passes at τ∈[0.687,0.748]; Slot A v2 face-pool misses by 0.1pp at boundary (sample noise); T5C fails. Plan's 'no ckpt passes' headline was τ-calibration artifact (used Slot A v2's 0.78 across all ckpts; correct P8A τ is 0.59). Production switch: T5C→P8A at τ=0.59, same-day $0. Structural-reframe urgency downgraded; direction unchanged."
- **`project_production_is_t5c_not_e2b_2026-05-23`** — append: "Per-ckpt τ-recal 2026-05-23 PM shows T5C fails team-identity 5%/50% at all τ; cannot reach 50% Xinhe recall under 5% FPR cap. P8A passes at τ=0.59. Recommended production switch."
- **`project_face_pool_scorecard_pareto_2026-05-22`** — append: "Per-ckpt τ=0.681, face-pool dor recall=0.499 (0.1pp under strict 50% floor; within ±1pp sample noise at n=2443). Xinhe recall 0.849. Face-pool is recall-leaning option at relaxed gate; the cross-const-τ failure in the team-deploy readout was largely calibration artifact."

### Threads to amend

- `docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md` — append a section noting that the team-identity gate failure framing was partly τ-calibration; structural diagnosis remains directionally correct but lower urgency.

### OPEN_LOOPS

- Close: "Production switch decision (T5C / P8A / Slot A v2 face-pool)" — verdict P8A at τ=0.59.
- Open: "Re-evaluate 9-suite contract under per-ckpt τ" — moderate-cost CPU follow-up (~1 hr).
- Open: "Re-evaluate Xinhe-may6 at per-ckpt τ for all 5 ckpts" — 5-min CPU follow-up.

### TIMELINE

- Append: `2026-05-23 PM — per-ckpt τ-recalibration on team-identity bar (analysis/per_ckpt_tau_recal_2026-05-23/) — P8A passes strict 5%/50% at τ∈[0.427,0.753] best 0.59; E2B passes at [0.687,0.748]; Slot A v2 face-pool misses by 0.1pp (sample noise); T5C fails all gates; SlotAv2 CLS passes relaxed 5%/40%. Plan's "no ckpt passes" headline was τ-calibration artifact. Production switch recommendation: P8A at τ=0.59.`

---

## 9. Gaps and blockers

- **9-suite contract not re-evaluated.** P8A passing the team-identity bar at τ=0.59 doesn't automatically clear the conventional contract — that's an independent τ-sweep on a different cohort.
- **dor-webcam-false-flag pools missing** (per the source readout §14). If P8A is switched into production, these pools should be scored to confirm the τ=0.59 choice doesn't regress on the dor-webcam corner that motivated some prior verdicts.
- **Xinhe real n=79** carries ±5pp CI on the per-human FPR cell. Task #2 (bootstrap CI) will tighten this.
- **The 50% floor itself is user-discretionary** — the per-ckpt readout is robust to either 50% or 40% as the floor; user needs to pick.
