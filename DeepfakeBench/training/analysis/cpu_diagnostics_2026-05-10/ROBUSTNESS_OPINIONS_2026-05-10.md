# Robustness diagnostics OPINIONS (2026-05-10)

> **Status: interpretation.** This document reads the `ROBUSTNESS_FACTS_2026-05-10.md`
> numbers and proposes what they mean for deployment. The reader is free to disagree.
>
> **Authoring**: independent take requested by user after T3 packet landed.
> Driven by the user's anecdotal observation that "P8A feels very fragile and sensitive"
> and the question "which model is the most robust to different conditions while
> keeping a high fake recall in all of these diversity?"

---

## 1. The headline reframe

**Step1500 is empirically the most production-robust ckpt of the 4 candidates,
not P8A.**

This is a substantive revision of both my initial take ("ship P8A") and the prior
agent's take ("ship step2500"). Three independent diagnostics converge:

1. **Catastrophic-tail width** (Job B): step1500 has the narrowest tail of reals
   scoring above any deployment threshold (9.03% > 0.5, vs P8A 12.20%, step2500 19.96%)
   AND has **zero unique catastrophic FPs** (real frames it false-flags that no other
   ckpt false-flags) — vs P8A 11, step2500 60. P8A's "feels fragile" anecdote is
   directly corroborated by the 11 unique catastrophic-FP signature.

2. **Post-IQ-gate FPR** (Job C2): at the medium IQ gate (lap_var≥100, min_dim≥200),
   step1500 hits 2.65% FPR @ τ=0.5 vs P8A 4.21%, E2B 10.76%, step2500 14.20%.
   The IQ gate REDUCES step1500's FPR 3.18× (the largest reduction of any ckpt),
   but only reduces step2500's FPR 1.46× — meaning step2500's failures are
   structurally NOT IQ-correlated and the gate doesn't shield them.

3. **Disagreement-structure conservatism** (Job D): step1500 is **NEVER the
   outlier-high** on disputed real frames (n=474 controversial frames, 9.8% of
   cohort) — meaning it never aggressively false-flags reals when the other 3 ckpts
   agree the frame is real. P8A is outlier-high 36% of the time; step2500 is 53%.

## 2. Each ckpt has a distinctive fragility axis

Job A's per-axis-bin FPR matrix surfaces a per-ckpt fragility signature that
the macro metrics don't show. At calibrated 5% overall FPR:

| ckpt | worst axis-bin | FPR there | mechanism |
|---|---|---:|---|
| P8A | min_dim Q1 (lowest res) | 10.4% | low-resolution reals over-fire |
| E2B | lap_var Q1 (blurriest) | 13.1% | blurry reals over-fire |
| T3_S1_step1500 | color_a_dev Q4 (warmest) | 14.3% | warm-color reals over-fire |
| T3_S1_step2500 | color_a_dev Q4 (warmest) | 20.4% | same axis, twice as bad |

Critically: **P8A's and E2B's worst-bin failures are on axes the proposed IQ
gate filters (resolution, sharpness). T3 step1500/2500's worst-bin failures
are on color — which the IQ gate does not filter.** This is the structural
deployment-decision pivot.

But Job C2 shows the IQ gate's filtering is more nuanced than the prior agent's
framing ("color is not an IQ axis, gate doesn't help"). The medium gate happens
to filter 70.9% of color-axis Q4 frames — not because it filters by color, but
because warm-color frames in the eval substrate happen to also be lower-sharpness
or lower-resolution. The gate works by accident on the eval substrate.

In production, this correlation may not hold: a warm-color user with sharp
resolution would pass the gate. Job C2's "strict gate + color_q4" cell shows
this risk: at the strict gate, step2500's FPR on warm-color frames JUMPS to
50%, and E2B's jumps to 55.4% — because the strict gate selectively retains
the sharp+large warm-color frames that those ckpts over-fire on most. **The
IQ gate doesn't shield step2500's color-axis fragility; it concentrates it.**

## 3. Each ckpt's "fragility" anecdote has a different mechanism

The user's observation that "P8A feels fragile and sensitive" maps onto Job B's
finding of 11 unique catastrophic-FP frames + Job A's min_dim Q1 fragility.
P8A is fragile on **low-resolution real frames** specifically. Production
scenarios with small face crops (distant cameras, multi-person framing) would
trigger P8A's fragility.

Step2500's fragility is on **warm-color real frames** (Roy_D-type production
users with tinted lighting, color-graded skin tones, or cosmetic warmth).
Job D shows step2500 is the outlier-high on 53% of controversial frames — when
ckpts disagree, step2500 is most often the one over-firing alone.

E2B's fragility is on **viso fake recall** — 86.5% of visomaster_enhanced fakes
score below 0.3 at any reasonable threshold. The currently-deployed model is
weakest on the production-target attack class, regardless of any T3 promotion
question. This is the most-urgent operational finding in the data.

Step1500's fragility is on the warm-color axis but at half step2500's magnitude
(14.3% FPR vs 20.4% in Job A, 11.5% vs 33.6% post-medium-gate in Job C2).
It's the same axis as step2500, just less severe.

## 4. The fake-recall trade-off is real

Step1500's robustness on reals comes at the cost of weaker fake recall on the
hardest cells. The trade-offs at FPR-cal τ:

| metric | P8A | E2B | step1500 | step2500 |
|---|---:|---:|---:|---:|
| F0 dev_fake_macro_recall (contract floor 30%) | 30.0% ✓ | 50.9% ✓ | 37.6% ✓ | 23.0% ✗ |
| HDTF visomaster_enhanced_teams_dev (n=1182) | **94.6%** | 43.5% | 77.4% | 85.0% |
| F4 viso recall @10% FPR | 67.1% | 30.9% | 73.3% | **79.3%** |
| F4 deeplive recall @10% FPR | 92.5% | 100.0% | **100.0%** | 100.0% |
| dev_worst_real_stress_fpr (cap 10%) | **6.85%** ✓ | 9.99% ✓ | 9.92% ✓ | 9.99% ✓ |

Step1500 sits in a real sweet spot:
- Beats P8A on F4 viso (+6pp), F0 dev_fake_macro (+8pp), HDTF clean substrates (+1-2pp)
- Within 17pp of P8A on HDTF visomaster_enhanced_teams_dev (the production-honest
  attack-class cell)
- Has the most-robust real-side behavior of all 4 candidates per Jobs A/B/C/D

P8A has the strongest HDTF visomaster_enhanced_teams_dev recall (94.6%) — that
remains its unique strength. If production traffic is dominated by that exact
attack class on HDTF-like identities, P8A still wins on Pillar 1.

## 5. Where I disagree with my own prior take

My initial recommendation was "ship P8A" based on the macro HDTF
visomaster_enhanced_teams_dev win. The CPU diagnostics today refute the
load-bearing premise of that recommendation:

- **Premise**: "P8A is the most production-robust because it dominates the
  most HDTF cells."
- **Refutation**: HDTF cells measure aggregate fake recall at FPR-cal τ.
  They don't measure real-side catastrophic-tail behavior, which is the
  production failure mode the user is describing. On the catastrophic-tail
  metric (Job B), P8A is the SECOND-MOST-FRAGILE ckpt of the 4, with 11 unique
  catastrophic FPs and 12.2% reals scoring > 0.5.

My initial framing missed that P8A's and step1500's strengths are on different
dimensions: P8A optimizes Pillar 1 (fake recall on the hardest cells); step1500
optimizes Pillar 2 (real-side robustness across diverse conditions). Both are
load-bearing for Teams deployment per `MODEL_GOALS.md`, but they trade off
against each other in this candidate set.

## 6. Where I disagree with the prior agent

The prior agent's recommendation was "ship step2500 with FPR-calibrated τ on
production-realistic real cohort + face_scale_jitter refinement packet."
The CPU diagnostics today substantively complicate this:

- **Prior agent's premise**: "Under your IQ-gate deployment policy, [step2500's
  Roy_D regression] artifact disappears."
- **Refutation**: Job C2 shows the IQ gate at strict threshold AMPLIFIES
  step2500's color-axis FPR to 50%. The gate doesn't shield step2500's
  fragility; for the warm-color sub-population, it concentrates the failures
  onto the gate-passing frames.

The prior agent acknowledged this gap as an open question in §9 T4_d but
recommended step2500 anyway. With the actual gate-filtering numbers in hand,
step2500 is not as deployable under IQ-gate policy as the prior agent's
framing suggested.

## 7. Revised deployment recommendation

The deployment decision turns on the production cost function:

### Option A — Ship step1500 (recommended for production-robustness)

Best Pillar 2 + Pillar 3 of the 4 candidates. Worst Pillar 1 on HDTF
visomaster_enhanced_teams (77% vs P8A 94%) but still substantially above
E2B (43%). Zero unique catastrophic FPs; lowest false-flag tail; robust
across IQ-gate thresholds; conservative on disputed reals.

**Trade-off**: Gives up 17pp on HDTF visomaster_enhanced_teams recall vs P8A
to gain real-side robustness on the broad production population.

### Option B — Ship P8A (recommended if HDTF teams transport is the production attack)

Best Pillar 1 on HDTF visomaster_enhanced_teams_dev (94.6%). Mid-tier Pillar 2
(11 unique catastrophic FPs, 12.2% > 0.5 tail). Has a deeplive blind spot
(37% catastrophic miss on `deeplive_enhanced_dev`).

**Trade-off**: Strongest on the attack class but has the resolution-fragility
the user's anecdote describes.

### Option C — Ship step2500 (NOT recommended without addressing color-axis fragility)

Best Pillar 1 on F4 viso + step2500's fake recall (62.4% F4@5% viso, 79.3%
F4@10% viso, 100% deeplive at every FPR threshold). Worst Pillar 2 (60 unique
catastrophic FPs, 20% > 0.5 tail). Color-axis fragility not shielded by IQ gate.

**Trade-off**: Highest fake recall but the most production-fragile real-side.
Recommend AGAINST until color-axis regression is addressed (see T4 design below).

### Switch from currently-deployed E2B regardless

E2B's Pillar 1 on viso_teams transport is 43.5% (HDTF) and 0.16% on F0 viso
at 5% FPR. Whatever deployment ckpt is chosen, E2B is the weakest on the
production-target attack class and should be retired. This is independent
of the T3 promotion debate.

## 8. T4 packet design (when ready)

When the user is ready for the next training packet, the diagnostics here
support a specific structural intervention:

**T4 = Slot 1 keep-list + face_scale_jitter @0.50** (the prior agent's T4_a
proposal, now empirically motivated). Mechanism: face_scale_jitter forces the
model to discount identity-specific cues, including the color signature that
drives Roy_D regression in step1500/step2500. The hypothesis is that adding
jitter on top of step1500's data lever produces a ckpt with step1500's
real-robustness AND closer-to-step2500's fake-recall.

This is testable in one Vertex training run (~$50-70). Cheap relative to
the deployment-improvement value.

## 9. Open questions

1. **Production-frame retest on may6**: T3 ckpts haven't been scored against
   the Xinhe-may6 false-flag frames (memory `project_xinhe_may6_falseflag_2026-05-06`).
   This would directly empirically test whether step1500/step2500 reproduce
   P8A's may6 fragility on the same frames. Cost: ~5 minutes MPS time after
   GCS download. Worth running.

2. **HDTF promotion-contract re-run for E2B**: the open loop
   `hdtf-promotion-contract-failure-recurrence` blocks the formal contract verdict.
   With the suite-name-map bug fix (one-line change at
   `score_teams_promotion_contract.py:748`), an E2B + P8A scorecard on HDTF
   would close the deployment-switch decision with proper artifacts. Cost: ~$15-20.

3. **Step2500 mechanism attribution**: open loop `roy-d-mechanism-not-fully-diagnosed`
   asks whether the color-axis regression comes from the keep-list lever or
   from PA's inherited `visomaster_enhanced` data sources. CPU-diagnostic-only,
   $0. Determines whether T4 should refine the keep-list or rebalance the
   inherited data.
