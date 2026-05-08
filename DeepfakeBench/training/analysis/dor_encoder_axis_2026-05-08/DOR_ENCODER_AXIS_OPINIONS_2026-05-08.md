# Dor encoder-axis characterization — OPINIONS (2026-05-08)

> **Status: OPINION.** Past framings on this codebase have been demonstrably
> wrong (P18 GRL "biting" verdict reversed; P14 bundle 5.7× weaker than
> isolated lever; viso ceiling reframed twice). Read the FACTS doc
> (`DOR_ENCODER_AXIS_FACTS_2026-05-08.md`) for numbers. This document
> reasons about what those numbers mean for INTERPRETATION 1 vs 2 and the
> Stage 2a packet design.
>
> **Audience**: a fresh agent reading the FACTS docs without prior bias —
> per the user's framing: "We don't want to feed the next agent bias, but
> we also don't want to deprive it from the information." Alternative
> readings are explicitly marked.
>
> **Author**: agent that ran checks (a) and (c) on 2026-05-08.

---

## 1. Headline reading (provisional)

The data points to a **MIXED INTERPRETATION 1+2 collapse**, not a clean
either/or split. Specifically: P2-D-step3000 has constructed a "Dor identity"
sub-region in feature space where:

- Both Dor-real and Dor-fake centroids cluster together
  (final-CLS cosine distance 0.112, vs P8A 0.952; ~8.5× compression).
- Dor reals are pushed AWAY from non-Dor reals
  (final-CLS distance 0.342, vs P8A 0.062; 5.5× farther).
- Dor fakes are pushed AWAY from non-Dor fakes
  (final-CLS distance 1.077, vs P8A 0.003; 350× farther).
- Per-frame median score for Dor reals: 0.181 (P8A) → 0.436 (P2D).
- Per-frame median score for Dor fakes: 0.986 (P8A) → 0.489 (P2D).
- Score gap between Dor real p50 and Dor fake p50:
  P8A 0.805, E2B 0.554, P2D **0.053**.

This is **bidirectional** (reals up, fakes down), **identity-specific**
(only Dor; non-Dor cohorts retain reasonable separation), and **score-space
convergent** (both classes converge on ~0.45–0.49 mean score).

I read this as Hypothesis-1-real-side **simultaneously** with
Hypothesis-2-real-side: the encoder treats Dor identity as a special cluster
and within that cluster the real/fake separation collapses. The two
hypotheses originally framed as alternatives turn out to be the same
phenomenon viewed from different sides.

---

## 2. Interpretation 1 vs 2 — explicit reading

### 2.1 INTERPRETATION 1 (CONFUSED) — STRONG SUPPORT

**Signature**: Dor reals + Dor fakes share a confused region. Centroids
close. Margin shrinks.

Evidence aligned with Interpretation 1:

- P2D `dor_real_vs_dor_fake` cosine distance = 0.112 vs P8A 0.952 (FACTS §2).
  Centroids are 8.5× closer.
- P2D Dor real p50–Dor fake p50 score gap = 0.053 vs P8A 0.805 (FACTS §4.3).
  Distributions overlap heavily.
- P2D DOR_REAL_DEV score std = 0.176 vs P8A 0.347 (FACTS §4.1). Variance is
  compressed by ~50% — frames are pushed toward a common centroid.

This is the dominant signal in the data. Dor frames live in a degenerate
region where the model can't separate real from fake.

### 2.2 INTERPRETATION 2 (SHIFTED) — PARTIAL SUPPORT, REAL SIDE ONLY

**Signature**: Dor reals shifted toward fake centroid; axis-specific
failure that may correlate with a non-IQ feature.

Evidence aligned with Interpretation 2 (real side):

- P2D Dor real frames sit at median cos-distance 0.214 to non-Dor fake
  centroid, vs P8A 0.646 (FACTS §5.1). Reals pushed toward generic
  fake region.
- P2D DOR_REAL_DEV mean score 0.471 vs P8A 0.331 (FACTS §4.1). Reals
  fire higher.
- P2D DOR_REAL_DEV FPR at τ=0.5 is 0.36 (vs P8A 0.30) — modest in this
  sample, but the J4 bucket-FPR at the contract-selected τ goes to 0.46.

Evidence AGAINST Interpretation 2 (fake side; this is where the simple
"shift toward fake" reading breaks):

- P2D Dor fakes sit far from non-Dor fakes (cos-distance 1.077 vs P8A
  0.003 — a 350× increase). The Dor fakes did NOT collapse onto the
  fake centroid; they moved AWAY from it.
- P2D DOR_FAKE_DEV mean score 0.492 vs P8A 0.945 (FACTS §4.3). Fakes
  fire LOWER.

So the data does NOT support a simple "Dor reals shifted toward fake
centroid; Dor fakes still recognized" reading. Both Dor classes were moved.

### 2.3 The "third" reading — IDENTITY CLUSTER COLLAPSE

(Best-fit reading per my read of the data; explicitly opinion.)

P2-D-step3000 has constructed an "is-Dor" cluster in feature space, distinct
from both the non-Dor real and non-Dor fake regions. Within this Dor
cluster, the real/fake direction is severely compressed. This is consistent
with:

- The encoder learning an identity-specific feature that overrides the
  real/fake feature.
- The new identity-specific feature being orthogonal to the IQ axes
  (IQ R² for `score ~ IQ` on Dor cohort is weak for P2D — see §3).

This is Interpretation 1 (confused region) with the additional structural
property that the confused region is **identity-bounded** — non-Dor frames
preserve their normal real/fake separation.

The phrasing "Dor regression" in J4 is accurate at the bucket level
(real_FPR ↑, fake_recall ↓) but the underlying mechanism is bilateral —
both Dor real AND Dor fake scoring degraded relative to non-Dor cohorts.

### 2.4 Alternative reading the next agent could honestly take

**Alternative A** (Interpretation 1 only — "encoder couldn't learn Dor
identity-specific structure"): Per the FT-from-CLIP-scratch family memory
`project_pa_does_not_generalize_to_hdtf_2026-05-05`, P2D belongs to the
E2B family which is FT-from-CLIP-scratch (not FT-from-P8A). P8A had
already learned Dor's identity-specific structure during R12G; P2D's
chain may not have had the same exposure trajectory. The encoder didn't
learn to keep Dor identity invariant; the resulting region is "confused"
because the model has no good representation for these frames. This
reading is testable by checking whether E2B (the precursor) shows similar
collapse — and the FACTS doc data shows E2B's `dor_real_vs_dor_fake`
distance is 1.976 (vs P2D's 0.112), so E2B did NOT show this collapse.
Therefore, the collapse is P2D-specific, not E2B-family-wide.

**Alternative B** (the new training objective — Fourier amp aug — caused
specific Dor disruption): per memory `project_fourier_band_overlap_2026-05-06`,
the P2-D recipe applies band-limited amp randomization. Dor frames may
have a distinctive amp-band signature that the augmentation specifically
disrupted, while not affecting other identities the same way. Testable
via per-identity Fourier amp signature analysis (not done here).

**Alternative C** (the "deepfake" assumption was wrong — the J4 numbers
are an artifact): per FACTS §5, the per-cohort distance medians from the
388-frame sample line up with J4's bucket-level FPR pattern. Unlikely.

Of A, B, C and "third reading" (§2.3), my opinion is the third reading
plus alternative B (training-objective-specific identity disruption) is
the most consistent with all 388 data points. Alternative A is partially
refuted by the E2B comparison but might still apply for the
Fourier-aug-specific changes. Alternative C is the null hypothesis and is
contradicted by the centroid pattern.

---

## 3. The IQ regression on Dor is WEAK on P2D

(This is the load-bearing observation that prompted check (c) in the
first place: Stage 1 IQ R² did not include Dor in any high-R² cell.)

From FACTS §6: Pearson r between per-frame `score` and per-frame IQ
features within the Dor cohorts. P8A has stronger lap_var correlation on
Dor real lockbox (r = +0.46) than P2D does (r = +0.19). This direction is
the OPPOSITE of what you'd expect if P2D's regression were IQ-mediated.

The two largest P2D−P8A score-vs-IQ correlation deltas are:
- DOR_FAKE_DEV `edge_mag`: P8A r = -0.24, P2D r = +0.61 (Δ +0.85). Sharper
  edges → higher fake score on P2D.
- DOR_REAL_DEV `min_dim`: P8A r = -0.33, P2D r = +0.28 (Δ +0.61). Larger
  frames → higher score on P2D.

**These are sign-flipped from the LOCKBOX IQ-shortcut direction.** Stage 1
reported the LOCKBOX `min_dim` coefficient as NEGATIVE (smaller frames →
higher score). On Dor for P2D, `min_dim` flips to POSITIVE (larger frames
→ higher score).

Reading: P2D's Dor regression cannot be explained by the LOCKBOX IQ-shortcut
direction. Whatever feature axis is responsible for the Dor collapse, it
is not in the same multivariate IQ direction the LOCKBOX cell is.

---

## 4. The encoder-vs-projection split

(Side observation; not load-bearing for the headline reading.)

FACTS §3 vs §2: at L11 (encoder pre-projection), P2D's
`dor_real_vs_dor_fake` distance is 0.229 — already 1.8× closer than P8A's
0.419 at L11. After projection (final-CLS), it compresses further to 0.112
(2.0× smaller). Both layers contribute to the collapse, but the encoder
already does most of the work.

Implication for Stage 2a: a GRL on the LATE-block CLS (e.g. layer 11) might
target the right region, but a GRL on the post-projection final-CLS could
also work. The data here doesn't strongly distinguish between those two
hook locations on the Dor signature alone — both layers carry the Dor
collapse signal.

---

## 5. What this analysis cannot tell us

(Self-correction-log honesty; the next agent should know what this probe
did NOT measure.)

1. **Whether the Dor collapse is causal of the FPR regression.** The
   centroid distances + score distributions + per-frame distances all
   describe the SAME feature-space pattern under different metrics. They
   don't establish causality. A GRL that "fixes" the centroid distance
   without also fixing scores would be a no-op for J4's FPR regression.

2. **Whether P2D's Dor collapse is rooted in the encoder, the head, or
   both.** §4 above shows the encoder already does most of the
   collapse, but the head/projection amplifies it. A counterfactual where
   we keep the P2D encoder + a P8A-trained head was not tested.

3. **Whether a Stage 2a IQ-GRL packet would touch the Dor axis.** The Dor
   axis appears (per §3) to be NOT in the named IQ direction. An IQ-GRL
   packet might leave the Dor regression intact even if it removes the
   LOCKBOX IQ-shortcut. Conversely, an identity-axis GRL (akin to P18's
   12-class method-conditional GRL but applied to identity instead of
   method) might address the Dor collapse but is not in the current
   Stage 2a proposal.

4. **Whether the same collapse happens at OTHER identities for P2D.** This
   probe focused on Dor because J4 highlighted it. Per J4 §5.2, the
   D-step3000 fake recall regression is Dor-only (non-Dor capture suites
   are at parity or ahead). So the identity-level analysis on Dor may not
   generalize.

5. **Whether E2B has a milder version of the same effect.** §3.1 shows
   E2B's `dor_fake_vs_non_dor_fake` distance at L11 is 0.419 (vs P8A
   0.051). E2B also separates Dor fakes from non-Dor fakes more than P8A
   does, just less aggressively than P2D. The collapse direction (Dor
   sub-region) may be a property of the E2B FT-from-CLIP-scratch family,
   amplified by the P2D Fourier-aug.

---

## 6. Implications for Stage 2a packet design

(Opinion-with-tradeoffs; the user reserves the decision.)

### 6.1 If Stage 2a is IQ-axis GRL only:

Risk: doesn't address the Dor identity-cluster collapse. The Dor regression
is on a non-IQ axis. A pure IQ-GRL packet might reduce LOCKBOX R² (the
target signal) without preventing further Dor regressions in the future
(Dor would remain a non-protected identity that future training disrupts).

### 6.2 If Stage 2a is IQ-axis GRL **plus** an identity-protection mechanism:

This stacks two levers, contrary to the AGENT_GUIDE single-lever discipline.
But the data suggests that without an identity-axis protection, IQ-GRL
might restore LOCKBOX behavior while opening new identity-specific
regressions.

### 6.3 If Stage 2a is identity-axis GRL only:

A 4-class GRL (Dor, PC_Generator, Roy_D, "other") might prevent identity-
cluster collapse. This would NOT touch the LOCKBOX IQ-shortcut but might
preserve P8A's per-identity invariance pattern as new training is applied.

### 6.4 My read on what the FACTS doc + check (a) data jointly indicate:

Check (a) shows IQ representation peaks at L6 (`min_dim` AUC 0.98) and
plateaus through L11 (avg per-feat R² 0.78 for E2B/P2D). The Dor identity
cluster collapses at L11+projection (FACTS §3-4). Both sit at the same
late-block locus. A GRL hook at L11 could theoretically address both — but
the GRL target axis matters: IQ-features GRL would not affect identity-
cluster collapse, and identity-features GRL would not affect IQ-shortcut.

I think the right Stage 2 design depends on the user's prioritization:
- If LOCKBOX-FPR reduction is the binding constraint → IQ-GRL Stage 2a
  per the proposal.
- If preserving P8A's per-identity invariance is the binding constraint
  → identity-axis intervention (Stage 2c, not currently proposed).
- If both → a sister-variant ablation comparing IQ-GRL alone vs
  identity-GRL alone vs both, run in parallel as 3 separate packets.

The data doesn't pre-decide between these; it surfaces the trade-off.

---

## 7. Concrete next-step proposals (for user authorization)

Three paths the user can choose between:

**Path A — proceed with Stage 2a IQ-GRL as proposed.**
Per the proposal §4.2 + sister-agent's recommendation. The IQ probe (check
(a)) suggests a layer 6 hook would target the IQ-feature peak. Risk: Dor
collapse continues / worsens because IQ-GRL doesn't touch the identity
axis.

**Path B — Stage 2c identity-axis intervention BEFORE 2a.**
Run a 4-class identity-conditional GRL (Dor, PC_Generator, Roy_D, "other")
as a single-lever packet. This addresses the J4 regression directly. Risk:
LOCKBOX FPR remains because IQ-shortcut is left untouched.

**Path C — sister-variant ablation: 2a IQ-GRL ∥ 2c identity-GRL ∥ both.**
Three GPU packets in parallel. Highest information yield; highest GPU
spend. Decision criterion at conclusion: which packet's per-substrate
close criterion (LOCKBOX R² ≤ 0.30 AND Dor real FPR ≤ 0.15 AND Dor fake
recall ≥ 0.80) is met by which packet.

I lean toward **Path C** for clarity, but the GPU spend (~$45–75) is the
user's call. **No GPU spend authorized for me.** The Stage 2 decision is
the user's regardless.

---

## 8. Cross-references

- FACTS: `DOR_ENCODER_AXIS_FACTS_2026-05-08.md` (this directory).
- Check (a) FACTS:
  `analysis/iq_perlayer_probe_2026-05-08/IQ_PERLAYER_PROBE_FACTS_2026-05-08.md`.
- Stage 1: `analysis/iq_shortcut_decomp_2026-05-08/IQ_DECOMP_FACTS_2026-05-08.md`.
- J4 source: `analysis/p2_eval_2026-05-08/P2_DEEPER_ANALYSIS_FACTS_2026-05-08.md` §5.
- Stage 1 OPINIONS:
  `analysis/iq_shortcut_decomp_2026-05-08/IQ_DECOMP_OPINIONS_2026-05-08.md`.
- Memory:
  - `project_p8a_breakthrough.md` (P8A's signature dor invariance).
  - `project_phase1a_method_cluster_axis_2026-05-01.md` (method-cluster axis
    as a precedent for identity-cluster axes).
  - `project_p18_diagnostics_complete_2026-05-02.md` (P8A 0/1170
    real_lockbox_dor_FPR baseline).
  - `project_pa_does_not_generalize_to_hdtf_2026-05-05.md` (FT-from-CLIP
    family generalization pattern).
  - `feedback_decision_points.md` (user reserves Stage 2 decision).
- Predecessor frames superseded:
  - The original "Interpretation 1 vs 2" framing in the task prompt: the
    data shows BOTH happening simultaneously in a third pattern
    (identity-cluster collapse). The framing was helpful starting point;
    the data refines it.
