# Agent Proposal — T3 deployment recommendation (2026-05-10)

> **Status: OPINION doc.** This document is a single agent's interpretation of
> the FACTS at `ROBUSTNESS_FACTS_2026-05-10.md` and the prior packet retro at
> `docs/packet_retrospectives/packets/T3.md`. The reader is free to disagree.
>
> **Authoring**: written 2026-05-10 by the agent that ran the post-T3
> robustness diagnostics. Held separately per the user's request — the user
> intends to compare this opinion against an independent fresh-agent take
> before adopting either.
>
> **Companion docs (FACTS, no interpretation)**: `ROBUSTNESS_FACTS_2026-05-10.md`,
> `REPORT_2026-05-10.md`. Driver scripts and per-bin CSVs at
> `analysis/cpu_diagnostics_2026-05-10/scripts/` and `outputs/`.

---

## TL;DR (my opinion)

**T3_SLOT1_PERIODIC_STEP1500 is the right primary deployment candidate.** Not
P8A (which I'd argued for earlier in this session before the diagnostics
landed) and not T3_SLOT1_PERIODIC_STEP2500 (which the prior session's morning
brief had recommended).

The case is asymmetric:
- **vs the deployed E2B** — step1500 wins on every dimension I measured. Not close.
- **vs P8A** — step1500 wins on more cells than it loses, by larger margins than its losses, and the losses are bounded.
- **vs step2500** — step1500 has dramatically better real-side robustness; step2500's only edge is fake-recall on the hardest cells, paid for with empirically-confirmed catastrophic-tail behavior on production-drift frames.

Step2500 is a research success — it proves the data-axis lever can drive
substantial fake-recall lift — but it should not ship in its current form.
T4 (Slot 1 keep-list + face_scale_jitter@0.50) is the natural next training
packet to test whether jitter resolves step2500's color-axis fragility while
preserving its fake-recall lift.

---

## Why step1500 wins overall — the case

### 1. Step1500 is a clear net improvement over P8A on cells that matter for production

| metric | P8A | step1500 | Δ | what it means |
|---|---:|---:|---:|---|
| F0 lockbox fake recall | 38.74% | **77.47%** | **+38.7pp** | catches 2× more lockbox fakes |
| F0 dev_fake_macro_recall | 30.03% | **37.63%** | +7.6pp | passes contract floor by 8pp |
| F4 viso recall @10% FPR | 67.09% | **73.27%** | +6.2pp | breaks viso ceiling |
| F4 deeplive recall @10% FPR | 92.48% | **100.00%** | +7.5pp | perfect deeplive coverage |
| Deeplive catastrophic-miss rate (<0.3) | 37.1% | **10.5%** | -27pp | step1500 catches 27pp more |
| Pre-gate FPR @ τ=0.5 | 10.70% | **8.43%** | better | narrower false-flag tail |
| Post-medium-IQ-gate FPR | 4.21% | **2.65%** | better | gate works for both, more for step1500 |
| Unique catastrophic-FP frames | 11 | **0** | better | step1500 has no "alone over-fires" cases |
| Per-identity FPR (PC_Generator) | 24.1% | **12.8%** | -11.3pp | substantial chronic-id improvement |
| Per-identity FPR (non-chronic) | 6.5% | **2.0%** | -4.5pp | broad-population improvement |
| Outlier-high rate on disputed reals | 36% | **0%** | better | step1500 never aggressive-alone on reals |

That's 11 cells where step1500 beats P8A, several by very large margins. The
+38.7pp on lockbox fake recall is by itself a major capability lift.

### 2. Step1500's losses vs P8A are real but bounded

| metric | P8A | step1500 | Δ |
|---|---:|---:|---:|
| HDTF visomaster_enhanced_teams_dev (FPR-cal 5%) | 94.62% | 77.38% | -17.2pp |
| may6 production-drift false-flag rate | 0/92 (0%) | 6/92 (6.5%) | +6.5pp |
| Day-to-day drift (may5→may6 mean) | +0.000 | +0.096 | +0.096 |
| Stress FPR (lighting_extreme dev_worst) | 6.85% | 9.92% | +3.1pp |

The HDTF teams transport gap is the biggest. It only matters proportional to
how much of the production attack mix is HDTF-style (different identities
through Teams pipeline with GFPGAN enhancement). If production traffic is
dominated by **clean** transport (most non-HDTF Teams calls), step1500 wins;
if it's dominated by **teams-transport-after-HDTF-style-enhancement**, P8A
wins this cell.

The may6 regression is small in absolute terms (6 frames out of 92, none
above 0.9). It's measurable, but it doesn't reproduce the catastrophic-tail
behavior we see on E2B (53/92) or step2500 (71/92).

The stress FPR delta is structurally interesting: per the per-identity
analysis in the morning brief, T3's 9.92% stress FPR is **entirely Roy_D
on lighting_extreme**. Ex-Roy_D, T3 (3.2%) beats P8A (5.0%) on stress.
Whether this matters in production depends on whether production users
have Roy_D-equivalent IQ profiles (color-warm + lap_var ~90).

### 3. The catastrophic-tail asymmetry is the real story

The single most-load-bearing finding of this diagnostic run is the unique
catastrophic-FP count:

| ckpt | unique catastrophic FPs |
|---|---:|
| **step1500** | 0 |
| E2B | 2 |
| P8A | 11 |
| step2500 | 60 |

These are real frames where one ckpt scores >0.9 AND all other 3 ckpts
score <0.5. Step1500 has none — every frame it false-flags is also
false-flagged by at least one other ckpt. P8A has 11 that no other ckpt
shares. Step2500 has 60.

This is the empirical signature of "robustness across diverse conditions."
A model with 0 unique catastrophic FPs is one that doesn't have a
characteristic failure mode that the other models avoid. A model with 60
has a structural weakness that's specific to that model. **Step1500 is
the only ckpt of the 4 with no characteristic failure mode.**

### 4. The disagreement-structure data confirms step1500's conservatism

Job D found that on disputed real frames (range > 0.7 across ckpts):
- step1500 is NEVER the outlier-high (0% — never aggressively false-flags alone)
- step2500 is NEVER the outlier-low (0% — always aggressive when ckpts disagree)
- P8A is outlier-high 36% of the time
- E2B is outlier-low 51% of the time (under-confident on real frames that look like fakes)

Step1500 has the most "agreement-with-the-room" behavior. Combined with
the lowest catastrophic-tail width and the zero unique catastrophic FPs,
this is a coherent picture of a calibration-conservative model that
preserves real-side discrimination across diverse axes.

---

## What's good about step1500

The above is the structural case. Beyond that, step1500 has several
properties that make it operationally attractive:

1. **Single-lever recipe.** FT-from-P8A with one additive change (drop top-25%
   high-Lap teams reals via keep-list) on top of PA's data sources. Easy to
   reason about; easy to attribute lift.

2. **Inherits P8A's substrate-invariance.** The drop-list lever is additive on
   top of P8A's representation. Pearson r=0.825 between P8A and step1500
   scores on the broad real cohort — the highest pairwise correlation I
   measured. Step1500 looks like "P8A plus a small targeted modification,"
   not a structurally different model.

3. **Passes the F0 contract floor cleanly.** F0 dev_fake_macro_recall=37.63%
   (vs 30% floor). Step2500 fails this floor, P8A barely passes it (30.03%).
   Step1500 has the most headroom of the 3.

4. **Generalizes to HDTF** — partially, but enough to matter. 77.4% on the
   HDTF visomaster_enhanced_teams_dev cell is far from PA's 7.87% collapse
   on the same cell. Whatever the lever does, it's representational and
   not v2-bound.

5. **No catastrophic miss on deeplive.** P8A misses 37% of `deeplive_enhanced_dev`
   fakes at score < 0.3. Step1500 misses 10.5%. The step1500 deeplive coverage
   is a meaningful Pillar 1 improvement on a target attack class.

---

## What's still problematic

I'm not arguing step1500 is the final answer — only that it's the right
primary candidate today, and the right operating point to ship from
this packet's outputs. Real concerns:

1. **The 17pp HDTF visomaster_enhanced_teams gap vs P8A is the elephant.**
   77.4% recall is well above the 30% promotion floor, but if the production
   attack class is dominated by HDTF-style teams transport, P8A is just
   better on Pillar 1. We don't know the production attack mix, and the
   diagnostics can't infer it.

2. **The may6 regression is small but not zero.** 6/92 at >0.5, +0.096 mean
   drift. This is a real production-frame fragility, just much smaller than
   E2B's or step2500's. If the user's production environment has frequent
   capture-pipeline drift (subtle day-to-day camera changes), step1500 will
   occasionally over-fire where P8A wouldn't.

3. **The Roy_D color-axis regression is the universal T3 obstacle.** Step1500
   has 14.3% FPR on color_a_dev Q4 (warm-color reals) vs P8A's 4.7%. Job C2
   shows the IQ gate at medium threshold reduces this to 11.5% (vs P8A's
   13.3%) — so it's *not* worse than P8A on warm-color frames *post-gate*,
   but it's measurably worse pre-gate. The whole T3 family has this signature;
   step1500 has the smallest version of it.

4. **Stress FPR 9.92% is at the contract limit.** Within the 10% target
   budget, but a single Roy_D-equivalent identity in production stress
   conditions (lighting_extreme) would push it over.

5. **The mechanism is not fully understood.** We know the lever (drop high-Lap
   teams reals) shifts the model's real-anchor signal from color_a_dev
   toward luma_mean. We don't know whether this would generalize to other
   chronic-real populations the eval substrate doesn't sample. The Roy_D
   regression may be a category — there could be Roy_D-equivalent users in
   production we haven't seen.

6. **Step1500 hasn't been HDTF-evaluated past τ=0.5 on most suites.** Some
   HDTF cells are only available at fixed τ=0.5, not FPR-calibrated. The
   proper comparison would need re-running the HDTF promotion-contract scorecard
   with the suite-name-map bug fix at `score_teams_promotion_contract.py:748`.

7. **Lockbox + stress + dor reports for step2500 don't exist locally.** I
   compared all 4 ckpts on `teams_real_all_dev` (the only suite with
   per-frame data for all 4). Stress + lockbox + dor reports for step1500
   exist; step2500's don't. So the catastrophic-tail comparison is on
   broad-cohort data, not lockbox or stress-substrate data.

---

## How this informs next experiments

### Priority 1 — Cheap, high-value diagnostics (close before T4)

**1.1 — Fix HDTF promotion-contract scoring bug + re-run for E2B + P8A** (~$15-20).
Closes the deployment-switch decision with proper formal artifacts. Currently
blocked by the suite-name-map bug at `score_teams_promotion_contract.py:748`
(hardcoded `--dev_real_suite teams_real_all_dev` not present in HDTF manifest).
One-line fix.

**1.2 — Roy_D mechanism attribution diagnostic** (free, CPU-only). Open loop
`roy-d-mechanism-not-fully-diagnosed`. Two probes:
- (a) ablate keep-list on `visomaster_enhanced` data sources alone — does the
  Roy_D regression persist without the keep-list?
- (b) probe whether disabling `visomaster_teams_enhanced` (PA's data source)
  restores Roy_D handling on a frozen P8A.

If (a) shows the regression is keep-list-driven → T4 should refine the keep-list.
If (b) shows the regression is data-source-driven → T4 should rebalance data,
not the keep-list.

**1.3 — Score step2500 on lockbox + stress + dor cohorts**. T3 ckpts only
have per-frame reports for 4 suites. Adding the missing 5 suites would
let me extend Jobs A/B/C2 to lockbox + stress, completing the robustness
picture. Either run a full T3 scorecard at the proper image version
(~$15-20) or score locally on Mac (free, ~30 min).

### Priority 2 — T4 packet (the next training run)

**T4 = Slot 1 keep-list + face_scale_jitter@0.50.** Single-lever discipline.
Cost ~$50-70 on Vertex.

Hypothesis: face_scale_jitter forces the model to discount identity-specific
cues — including the color signature that drives Roy_D regression. Stacking
it with the Slot 1 keep-list should produce a ckpt with step1500's real-side
robustness AND closer-to-step2500's fake recall.

Why this is the right structural lever:
- The score-IQ correlation analysis showed T3 weakened ρ(score, color_a_dev)
  from −0.59 to −0.45 and added ρ(score, luma_mean) = +0.49. The model
  shifted away from color as a real-anchor; jitter would break the
  *identity-specific* color signature that fills the gap.
- face_scale_jitter@0.50 was the load-bearing P14 winner per memory
  `project_face_scale_jitter_load_bearing` (mclioexb won value_composite=0.661
  vs full bundle 0.116; jitter alone was the lift).
- It's a single lever on top of an already-validated single lever — clean
  attribution.

**Close criteria for T4** (do not promote without all 4):
1. Match step1500 on catastrophic-tail width (≤9.03% reals > 0.5 on broad cohort)
2. Match step2500 on F4 viso recall (≥79% at FPR=10%)
3. Reduce color_a_dev Q4 FPR to ≤10% (vs step1500's 14.3%, step2500's 20.4%)
4. Preserve 0 unique catastrophic-FP signature (or ≤2)

If T4 hits 4/4 → it's the deployment candidate, replaces step1500 as primary.
If T4 hits 1-3 of 4 → it's a refinement, not a winner; iterate.
If T4 hits 0 of 4 → mechanism reading was wrong; revisit before another run.

### Priority 3 — Production shadow-deploy (when feasible)

The HDTF visomaster_enhanced_teams gap is the hardest question to resolve
offline. A shadow-deployment of step1500 vs P8A (and possibly E2B for
baseline) on actual production traffic would:
- Reveal the production attack class mix
- Measure may6-style drift incidents in real conditions
- Resolve the step1500-vs-P8A primary-candidate question definitively

Cost is ops, not GPU.

### Priority 4 — Step2500 trajectory exploration (low priority)

Score steps {3500, 4500} of T3_SLOT1 on F4 + HDTF (~$15-30 if Vertex; free
if MPS). Tests whether the lift saturates at 2500 or continues. Mostly a
research question — these later steps aren't deployment candidates because
they share step2500's catastrophic-tail behavior at minimum.

---

## Confusion post-mortem — what I got wrong, in what order

Five distinct framing errors during this session, in order of when they
mattered:

### 1. Initial recommendation overweighted HDTF visomaster_enhanced_teams_dev

When I first read the FACTS docs (Tier 2 of the prescribed reading order),
my read was "P8A dominates the production-target attack class on HDTF —
ship P8A." I framed this as a Pillar 1 verdict.

What I missed: HDTF visomaster_enhanced_teams_dev is one substrate; the
production traffic distribution is unknown. Optimizing for it is optimizing
for a hypothetical attack mix. The macro recall numbers don't capture
real-side fragility, which is what the user's anecdote was about.

**Lesson**: Don't recommend a deployment based on a single substrate's
fake-recall metric. Multi-substrate + multi-axis robustness measurement
is required.

### 2. Used τ=0.5 readings for cross-substrate comparison

In my first HDTF analysis I read step1500 at 28.8% on visomaster_enhanced_teams_dev
(τ=0.5) and called it "broken." The prior agent's morning brief used
FPR-calibrated τ and got 77.4% — a 50pp difference. The τ=0.5 reading made
step1500 look much weaker than it actually is, because step1500 calibrates
to a much lower τ (0.682 on F0 contract; ~0.010 on HDTF reals).

**Lesson**: Always use each ckpt's own FPR-calibrated τ for cross-substrate
comparison. Fixed-τ comparisons across ckpts with different score
distributions are misleading.

### 3. Misattributed the may6 false-flagging to P8A

The memory entry `project_xinhe_may6_falseflag_2026-05-06` says "92 fresh real
Xinhe frames false-flagged at deploy 0.83-0.93." I read "deploy" as
ambiguous — and when the user said "P8A feels fragile," I assumed the may6
incident was P8A's behavior. The memory `project_deployment_is_e2b_2026-05-06`
explicitly says the deployed model is E2B (Pearson r=+1.000 between deploy
and E2B local), but I didn't connect the two in my initial framing.

The may6 retest today shows P8A scores 0/92 may6 frames > 0.5. E2B scores
53/92 (57.6%). The may6 false-flagging was E2B's behavior. The user's
"P8A feels fragile" anecdote is real but maps onto a different frame
population (the 11 unique catastrophic-FP frames Job B surfaced on the
broad eval cohort) — not onto may6.

**Lesson**: When two memory entries have related framings ("[deployed] does X"
and "deployed model is Y"), substitute Y into X explicitly before reasoning.
And when a user says "model X feels Y" and a documented incident mentions
"deploy did Y," verify whether "deploy" = X or "deploy" = something else.

### 4. Pivoted prematurely on the IQ-gate filterability question

Before running Job C, I asserted "the IQ gate operates on sharpness + min_dim;
it doesn't filter Roy_D's color signature." That's structurally true. But
when Job C ran, it showed the medium gate filters 93.7% of Roy_D's eval-substrate
frames — because Roy_D in the eval substrate happens to ALSO be lap_var-low
(89.6, just below the 100 threshold). The gate works by accident on this
substrate.

I then ran Job C2 to look at *production-realistic* warm-color frames
(color_a_dev Q4 that PASSES the gate). That showed the strict gate
*amplifies* step2500's failure on those frames (FPR jumps to 50%). So
my structural concern was right *at deployment time on diverse production
traffic*, but I had to reason through the eval-substrate accident to get there.

**Lesson**: When a gate-filtering claim depends on an axis the gate doesn't
operate on, check whether the axis is correlated with axes the gate DOES
operate on in the substrate where the claim is being made. Gate filtering
that "works by accident" on the eval substrate doesn't necessarily transfer
to production.

### 5. Underweighted my own Job B finding before may6

Job B surfaced step1500 with 0 unique catastrophic FPs and the narrowest
broad-cohort tail. My initial synthesis said "step1500 is the most
production-robust" based on this. Then the may6 retest showed step1500
has 6.5% over-flag rate on production-drift frames vs P8A's 0%. I
over-corrected to "ship P8A primary" — under-weighting that step1500's
broad-cohort robustness wins are larger than its may6 regression in
absolute terms, AND that step1500's may6 regression is dramatically
smaller than E2B's or step2500's.

The right read: both metrics are real; both ckpts win different cells;
step1500 wins more cells overall and by larger margins.

**Lesson**: When a single new data point flips a recommendation, check
whether the new data point is actually larger in magnitude than the
existing data points it's overriding. Over-correction is a real risk when
new findings land late in a session.

---

## Documentation changes I'm proposing

In the spirit of "let the next agent see clearly without bias," I propose
these documentation edits — all FACTS-additions, no opinions:

### Memory amendments
1. **Amend `project_xinhe_may6_falseflag_2026-05-06`**: replace "false-flagged
   at deploy 0.83-0.93" with "false-flagged at deploy (= E2B per
   `project_deployment_is_e2b_2026-05-06`) at scores 0.83-0.93." Make the
   attribution explicit.
2. **New memory `project_t3_robustness_diagnostics_2026-05-10`**: documents
   what was measured (catastrophic-tail count + IQ-gate filterability +
   per-axis variance + may6 retest), the 4 ckpts compared, and where the
   data lives. No deployment recommendation in the entry.

### Wiki updates
3. **Append to `packets/T3.md`**: a "Post-packet diagnostics 2026-05-10"
   section pointing at `analysis/cpu_diagnostics_2026-05-10/`. FACTS only.
4. **Append to `STATE.md`**: a current-state paragraph noting the
   diagnostics ran and where the artifacts live. Do not pre-empt the next
   agent's deployment read.
5. **Append to `TIMELINE.md`**: one line for this session.

### Methodology guidance (so the next agent doesn't repeat my mistakes)
6. **Add to `MODEL_GOALS.md` or `SCORECARD_GUIDE.md`**: a "cross-substrate
   comparison protocol" note: each ckpt's FPR-calibrated τ should be used
   for cross-substrate fake recall comparisons; τ=0.5 cross-ckpt comparisons
   are misleading when ckpts have different score distributions.
7. **Add to `AGENT_GUIDE.md`**: a "before recommending deployment" checklist:
   (a) catastrophic-tail count on broad real cohort; (b) production-frame
   retest if available (may6/canary); (c) post-IQ-gate FPR per ckpt;
   (d) per-axis FPR variance. Macro recall metrics alone don't capture
   fragility.

These changes don't push my conclusion. They give the next agent the same
data and the same methodology I had — and point out the specific
attribution and methodology pitfalls I fell into. The next agent should
be able to read everything and reach their own conclusion.

---

## What would change my mind

I'd revise the step1500-primary recommendation if:

1. **Production traffic turns out to be HDTF-visomaster-enhanced-teams-dominated.**
   Then P8A's 17pp HDTF lead becomes load-bearing and step1500 is the wrong
   primary. Resolution: shadow-deploy or production-traffic mix audit.

2. **The Roy_D mechanism diagnostic shows the regression is from PA's data
   sources, not the keep-list.** Then the lever is fragile (we'd need to
   rebalance data, not refine the keep-list), and the T4 design above is
   wrong-target.

3. **More extensive production-frame retests reveal step1500's may6
   regression generalizes.** If 5+ production-day retests show step1500
   has consistent +5-10% drift on diverse production frames, the
   broad-cohort win may not transfer.

4. **A trajectory exploration finds an even better operating point.** Step3500
   or step4500 on HDTF + F4 hasn't been measured. If one of them has
   step1500's robustness AND step2500's fake recall, that ckpt would be
   the new primary candidate.

5. **T4 lands cleanly.** If T4 hits all 4 close criteria, T4 replaces
   step1500 as the primary candidate. If T4 misses by a lot, the lever
   class is actually narrower than I'm framing.

Any of these would shift the recommendation. None would invalidate the
"switch from E2B" finding — that's empirically robust across all the
diagnostics today.
