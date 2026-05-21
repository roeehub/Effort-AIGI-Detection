# Memory caveats — claims requiring revision — frozen 2026-05-21

> Auto-memory entries (in `~/.claude/projects/.../memory/*.md`) accumulate
> session-summary OPINIONS as one-line indexes. Several of those summaries
> contain quantitative claims that subsequent analysis has invalidated or
> partially revised. This document records the specific deltas so a fresh
> agent doesn't act on a wrong number.
>
> This is FACTS-only — each entry states (a) what the memory says, (b) what
> later analysis showed, (c) where to read the correction.

## C1 — "Per-substrate τ has 21pp lockbox recall lift available on P8A"

**Memory entry**: `project_job7_head_retrain_REFUTED_2026-05-04.md` — claims
"21pp lockbox recall lift available on P8A via per-substrate τ-calibration
alone, no retraining."

**Status**: **OVERSTATED**. The 21pp number was computed comparing two
policies at **different lockbox FPR operating points**:
- Per-mode oracle dev-calibrated → lands at 16.9% lockbox FPR with 78.6% recall
- Single global τ dev-calibrated → lands at 4.3% lockbox FPR with 54.1% recall

At the **same** lockbox FPR (16.9%), naive global τ lockbox-calibrated achieves
~85.2% recall — **6.6pp MORE than per-mode oracle**.

The "lift" is a calibration-mismatch artifact, not a policy-quality lift.
Per-mode τ does NOT beat naive global τ at matched FPR.

**Where to read the correction**:
`analysis/iq_substrate_tau_2026-05-21/FACTS_2026-05-21.md` §"Apples-to-apples
Pareto comparison" and §"Why the May 5 '24.5pp lift' was misleading".

**Implication for strategy**: "per-account τ at deployment" should be
downgraded as a candidate lever. The remaining viable form is **discrete
per-account metadata signal from Teams' SDK** (not the noisy continuous
per-frame IQ proxy refuted today).

---

## C2 — "modern_v2 filter cuts headline FPR 4.6% → 0.71% at calibrated 5% τ"

**Memory entry**: `project_lockbox_fpr_dominated_by_webcam_mode.md`.

**Status**: **NEEDS CAVEAT**. The "filter" referenced is the same per-substrate
τ policy class as C1; its lift is similarly subject to the apples-to-oranges
issue. The specific 4.6% → 0.71% number IS a real reduction at that operating
point — but it requires a stricter τ that also reduces fake recall. Without
quoting recall at the same point, the FPR reduction alone is incomplete.

**Where to read**: Same as C1.

---

## C3 — "Slot A v2 ranks rank-2 only by a 0.07pp tiebreak"

**Memory entry**: `project_band_shortcut_ood_hypothesis_2026-05-16.md` + the
2026-05-20 7-job CPU evidence batch.

**Status**: **STILL TRUE on the 29-suite contract** (P8A 0.0184 lockbox_real_fpr
vs Slot A v2 0.0191; Slot A v2 wins lockbox_fake_recall 0.688 vs 0.387). But
the 2026-05-21 partial Pareto computation shows **Slot A v2 dominates P8A at
every lockbox FPR operating point from 1% to 20%** when τ is lockbox-calibrated
directly. So:

- On the 29-suite contract with its fixed dev-calibrated τ rule: P8A rank-1
  by 0.07pp (per Job B, this 0.07pp is inside sampling noise — 95% CI
  [-0.008, +0.010] covers 0; P=0.519 that P8A is truly better).
- On the lockbox-calibrated Pareto: Slot A v2 wins at every operating point.

Both are true. They reflect different calibration regimes. The contract is
the formal promotion criterion; the Pareto is the deployment-relevant
operational curve.

**Where to read the partial Pareto**:
`analysis/iq_substrate_tau_2026-05-21/slot_a_v2_lockbox_pareto.md`.

**Where to read the contract sampling-noise analysis**:
`analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/RESULTS_FACTS_2026-05-20.md`
§Job B.

---

## C4 — "Inference-side remediation DEAD for T5C"

**Memory entry**: `project_blend_unsharp_lever_2026-05-14.md`.

**Status**: **TRUE for score-altering remediations** (blend / blur / desat /
CLAHE / TTA — all 10 tested fail F1/F2/F3). Distinct from **per-substrate τ
adjustment** (which is a DIFFERENT class of inference-side intervention —
adjust threshold conditional on observable property, not transform the frame
score). 

The C1 finding (per-substrate τ refuted at matched FPR) is independent: it
refutes a SEPARATE class of inference-side intervention that this memory
entry didn't cover.

**Net**: both classes of inference-side intervention are now refuted on P8A,
for different reasons.

---

## C5 — "Codec aug works; restore P8A's teams_codec_sim_p=0.40"

**Pre-2026-05-21 expectation** (implicit in conversation framing leading up
to the codec restoration packets): restoring P8A's codec aug on top of
P22's pipeline_randomization would close the 2026-05-19 per-Teams-account
transport gap.

**Status (verdict 2026-05-21 night)**: **PARTIALLY refuted**. Codec aug DOES
bite the transport axis (37-60% Δ reduction on natural-experiment crops) but:
- Mechanism is **asymmetric**: encoder regularizes by uplifting Guest's score
  toward Roy_D's (both move into FAKE region), not by suppressing Roy_D's
  toward Guest's.
- All 3 packets regress may6 production-drift false-flag rate vs Slot A v2
  base (0.043 → 0.065/0.087/0.120).
- Dose-response **non-monotone** (p=0.20 gives best Δ closure but worse
  may6 than p=0.40 — suggests the lever is fighting itself).

Single-lever codec aug as a deployment fix on top of Slot A v2 is refuted.

**Where to read**:
`analysis/codec_restoration_gates_2026-05-21/VERDICT_FACTS_2026-05-21.md`.

---

## C6 — "Roy_D anchor pool extension is a load-bearing next step"

**Memory entry**: `project_band_shortcut_ood_hypothesis_2026-05-16.md` —
proposed Roy_D-specific anchor pool extension as "next packet."

**Status**: **REMAINS OPEN** but with two updates:
1. Slot A v2 already addressed dor-cluster chronic-FP via the anchor mechanism
   on a single pool (`dor-real-webcam-false-flag-no-virtual-bg`). Roy_D
   cluster is structurally different from dor — anchor mechanism may not
   generalize.
2. 2026-05-20 Job D measured Roy_D dev FPR going 0.30 → 0.84 (+0.54) on
   Slot A v2 — the regression is on DEV, not lockbox, and is invisible to
   the contract because Roy_D is not in the lockbox cohort.

The Roy_D anchor pool packet is still the highest-confidence training-side
option that hasn't been tried, but it requires bucket-prefix setup + frames
upload + registry edit in `analysis/teams_pool_rescore.py` (~60-90 min infra
with silent-failure risk per Plan agent's prior assessment).

---

## C7 — General caveat on packet-retro "verdicts"

Memory entries summarize at-the-time packet verdicts. Several of those
verdicts compared against an at-the-time-deployed ckpt or contract policy
that has since shifted:
- "ship T5C step3500 + τ=0.49 + G1+G2(110) + Option 3 per-identity rule"
  (memory `project_blend_unsharp_lever_2026-05-14`) was the **2026-05-14**
  deployment recommendation. Slot A v2 step3500 supersedes T5C step3500 for
  most operating points per the 2026-05-21 partial Pareto.
- The 2026-05-09 T3 SLOT1 step1500 verdict ("first substantive lockbox lift
  over P8A in many packets") was true at the time; subsequent T4/T5C/Slot A v2
  packets in the T-lineage produced larger lifts.

**Net**: treat memory verdicts as session snapshots, not standing claims.
The ledger at `docs/R13_LEVER_LEDGER_2026-05-21.md` re-states verdicts in
the language of "what was measured" rather than "what was decided."

---

## How to use this addendum

When reading auto-memory, hold this in your head: anything claiming a
specific quantitative lift over P8A (or any pre-Slot-A-v2 baseline) requires
checking whether the comparison was apples-to-apples on lockbox FPR. The
canonical comparison surface from 2026-05-21 onward is:
- 29-suite contract scorecard (`arena/score_teams_promotion_contract.py`) for
  formal promotion
- Lockbox-calibrated Pareto curve (per-substrate-population recall@FPR) for
  operational deployment decisions

If a memory's lift claim isn't framed in those terms, treat it as a session
hypothesis until you've re-derived the number under one of those two surfaces.
