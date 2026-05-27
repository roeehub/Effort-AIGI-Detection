# IQ-shortcut R² decomposition — OPINIONS (2026-05-08)

> **Status: OPINION.** Past framings on this codebase have been demonstrably
> wrong (e.g. the P18 GRL "biting" verdict was reversed; the P14 bundle was
> 5.7× weaker than the isolated lever; the "viso ceiling" reframed twice).
> Read the FACTS doc (`IQ_DECOMP_FACTS_2026-05-08.md`) for numbers. This
> document reasons about what those numbers mean for the Hypothesis A/B/C
> frame and the proposal §4 Stage 2 decision.
>
> **Audience**: the user, who reserves judgment calls at decision points
> (memory `feedback_decision_points.md`). The Stage 2 GPU spend authorization
> is the user's; this doc proposes a reading and a recommendation.
>
> **Author**: agent that picked up `HANDOFF_IQ_DECONVOLUTION_2026-05-08.md`.

---

## 1. Headline reading

The IQ-shortcut load is **substrate-conditional**, not uniform across cells as
the proposal §4.1 decision criterion implicitly assumed. R² ranges from
**0.016 (HDTF_TEAMS_DEV, P8A) to 0.594 (DEV_DEEPLIVE_VS_REAL, E2B)** across the
23 cells.

Two structural patterns visible across all three checkpoints (P8A, E2B,
P2-D-step3000):

- **HDTF cells** (8 cells across P8A + E2B): R² = 0.016–0.090 (median 0.046),
  residual AUC 0.71–0.99 (median 0.96). The model's score on HDTF is
  **not well-explained by IQ features**. Removing the IQ component does not
  meaningfully reduce fake-vs-real discriminability.
- **LOCKBOX_TEAMS** (3 cells, one per ckpt): R² = 0.40–0.54, residual AUC
  0.63–0.76 (drop of 0.21–0.29 from raw AUC). The lockbox score is the
  **single most IQ-explained cell** in every ckpt. Roughly 40–55% of the
  score variance on lockbox is captured by the 6 named IQ features alone.
- DEV cells sit in the middle (R² 0.02–0.59) with one outlier:
  E2B and P2-D have R² ~0.5–0.6 on DEV_DEEPLIVE_VS_REAL (the other ckpts'
  DEV cells are at 0.12–0.27). E2B/P2-D's deeplive-detection capability is
  **substantially IQ-driven**.
- **DEV_VISO_VS_REAL** R² is uniformly low (0.02–0.14), but raw AUC is also
  the lowest of any DEV cell (0.71–0.87). The model under-fires on viso at
  contract τ; what little signal it has is mostly NOT IQ-explainable.

## 2. Hypothesis A/B/C reading per substrate

The proposal §3.1 frames three competing hypotheses. The cell-by-cell numbers
do not all land on a single hypothesis. The substrate axis is the load-bearing
axis:

| substrate | observed pattern | best-fit hypothesis | confidence |
|---|---|---|---|
| HDTF (all 4 variants) | low R² + high raw AUC + high resid AUC | **B with content dominant** | high — confirms §3.3 reasoning |
| LOCKBOX_TEAMS | high R² + high raw AUC + moderate resid AUC | **B with IQ dominant** but content channel exists | high |
| DEV_TEAMS_PRIMARY | low-moderate R² (0.16–0.26) + high raw AUC + high resid AUC | **B with content dominant** | medium |
| DEV_DEEPLIVE (E2B/P2-D) | high R² (0.53–0.59) + high raw AUC + moderate resid AUC | **B with IQ dominant** | medium-high |
| DEV_VISO | low R² + lower raw AUC + lower resid AUC | inconclusive (model under-fires; little signal of either type) | low |

**No cell** shows the Hypothesis-A signature (high R² + low residual AUC).
Even the most IQ-dominant cell (LOCKBOX, P2-D, residual AUC 0.633) keeps
substantially-better-than-chance fake-vs-real discriminability after removing
the IQ component. Hypothesis A can be **provisionally rejected**: the encoder
is not "almost entirely an IQ classifier" on any substrate I measured.

**No cell** shows the Hypothesis-C signature in a way that is testable from
this probe (data inconsistency across training-real pools is upstream of the
score; this measurement is on the score side).

The picture across substrates is **Hypothesis B**, with the IQ-vs-content
balance shifting strongly by substrate. This refines the proposal's framing
rather than refuting it.

## 3. The Stage 2 decision per the proposal §4.1

The decision criterion as written (R² > 0.5 → 2a; ∈ [0.3, 0.5] → 2b; < 0.2
→ pivot) was specified "across most ckpt × pool cells". My data:

- **R² > 0.5**: 4 cells (3 LOCKBOX cells + DEV_DEEPLIVE for E2B and P2-D).
  4 / 23 = 17%.
- **R² ∈ [0.3, 0.5]**: 1 cell (LOCKBOX_TEAMS for P8A on primary_6, but ≥0.3
  jumps to 6 cells with expanded_10 features).
- **R² < 0.2**: 12 cells (all 8 HDTF cells, 4 of 5 DEV cells excl. STRESS).

If applied literally, the criterion says "pivot" (R² < 0.2 across most cells).
That reading conflicts with the LOCKBOX picture, where the IQ shortcut
clearly carries 40–55% of the score variance and ~21–29 AUC points.

**My read of what the data actually says about Stage 2**:

The criterion was written on the implicit assumption that R² is roughly
uniform across substrates. The data instead shows a substrate-conditional
pattern where the same encoder is heavily IQ-driven on one substrate and
content-driven on another. The right Stage 2 question is:

> "Where does the IQ shortcut bind, and is that the substrate that matters
> for production?"

Production deployment (memory `project_deployment_is_e2b_2026-05-06.md`,
`MODEL_GOALS.md`) is E2B serving Teams calls. The lockbox substrate is the
closest cross-validation analog to production capture conditions. **The IQ
shortcut is most operative on the substrate closest to production**, which
is exactly where Pillar 3 (robustness across capture conditions) lives.

So the binding question is not "should we GRL the IQ axis somewhere" — it's
"can a Stage 2 packet reduce LOCKBOX R² without breaking HDTF resid AUC".

## 4. Recommendation: Stage 2a (IQ GRL) — but with a substrate-conditional close criterion

### 4.1 Why GRL over IQ-balanced sampler (2a vs 2b)

I recommend **Stage 2a (IQ GRL)** rather than 2b (IQ-balanced sampler).
Reasons (opinion):

1. **The IQ axes are continuous, multi-dimensional, and correlated**. The
   regression with 10 IQ features captures up to R²=0.61 (E2B LOCKBOX) but
   the "primary 6" already captures R²=0.45. The shortcut isn't carried by
   a single feature; it's carried by a multivariate IQ direction. GRL on a
   multi-output IQ regressor head is structurally aligned with this; bin-based
   IQ-balanced sampling would have to bin in a high-dimensional IQ space and
   risk batch composition artifacts.
2. **Memory `project_iq_gating_viability_2026-05-04.md`** records that P8A
   recall is monotonically increasing with sharpness while E2B's recall is
   INVERTED. Bin-based sampling that naively re-weights by sharpness alone
   would not catch the multi-axis direction the encoder uses; GRL on a
   multivariate IQ regressor would.
3. **GRL machinery exists**. P15 (capture-mode GRL) and P18 (12-class
   method-conditional GRL) are in the codebase. Reusing it for a fresh axis
   is a smaller engineering ask than building a stratified sampler.

That said, 2b is a viable parallel option and the proposal §4.2's Failure
Modes hold for either: **the encoder may regress on Pillar 1 if IQ is a
dominant signal it relied on**. This is the central risk of any Stage 2
intervention, and it is not eliminated by choosing GRL over sampling.

### 4.2 Pre-launch CPU-first gate

Per `AGENT_GUIDE.md` Rule 3 (CPU-first-then-GPU): before authorizing the GPU
launch I'd run one more CPU diagnostic.

**CPU diagnostic D1** (~$0, ~2-3 hours): take the existing P8A and E2B
features (already extracted in `analysis/path_a_launch_2026-05-07/` and
`analysis/intermediate_layer_probe_2026-04-30/`), fit a multivariate linear
probe `IQ_axes ← CLS_features` per layer (resblocks 0–11). The proposal §3.4
hypothesizes that the method-cluster axis (Phase 1A) is partly an IQ axis.
This probe tells us:
- Whether the encoder represents IQ explicitly (probe AUC high at layer L)
  or as a side-effect of content features (probe AUC moderate at every L).
- Where in the encoder the IQ representation is concentrated (gives the GRL
  hook a target layer rather than the default CLS).

If the IQ probe is high at layers 8–11 (where memory
`project_per_layer_divergence_2026-05-06.md` says P8A/E2B diverge), the GRL
hook is best at the late-block CLS. If IQ is uniformly representable at every
layer (including layer 0 which is uninstructive — input pixel reads —
and the early blocks), the GRL has structural complexity ahead.

**This is a 1–2 day CPU-only addition before the GPU spend**. It refines
2a's design rather than gating it; the decision to authorize 2a is the
user's regardless.

### 4.3 Close criterion the user can hold the agent to

A Stage 2a packet should bind on a **substrate-conditional close criterion**:

| substrate | requirement (vs P8A baseline) |
|---|---|
| LOCKBOX_TEAMS | R² drops by ≥ 0.10 absolute (0.40 → ≤ 0.30); residual AUC stays ≥ 0.65 |
| DEV_TEAMS_PRIMARY | residual AUC stays ≥ 0.85 (P8A baseline 0.86) |
| HDTF_CLEAN_DEV | residual AUC stays ≥ 0.95 (P8A baseline 0.98) |
| DEV_DEEPLIVE | raw AUC does not regress > 0.05 |

A packet that drops LOCKBOX R² but tanks HDTF residual AUC would be
"worse-on-Pillar-3" not "better-on-Pillar-3" — even though the named
intervention "reduced the shortcut".

## 5. Caveat: the substrate axis is not separable from the data axis

The atlas's §2.5 records HDTF as the only substrate without IQ artifacts on
both real/fake sides. The HDTF residual AUC stays high (0.71–0.99) because
the IQ regression has no predictive power there — there's no IQ shift to
predict against. **The HDTF residual AUC is not an upper bound on what a
GRL'd model could achieve**; it's a signal that the IQ shortcut is upstream
of HDTF measurement and downstream of LOCKBOX measurement.

This means: **a Stage 2a packet evaluated only on LOCKBOX would over-credit
the IQ debiasing**; one evaluated only on HDTF would under-credit it. The
close criterion in §4.3 holds the packet to both.

## 6. What this analysis cannot tell us

(Self-correction-log honesty.)

- **Whether the IQ shortcut on LOCKBOX is the production blocker**. Memory
  `project_lockbox_fpr_dominated_by_webcam_mode.md` says webcam mode
  dominates lockbox FPR; the user policy memory
  `project_canary_below_production_resolution_2026-05-08.md` says some
  chronic-FP resolution levels would be IQ-gated at deployment. If a
  meaningful chunk of LOCKBOX FPR sits below the deployment IQ gate, the
  Stage 2a payoff goes to deployment-relevant frames only after the gate
  threshold is fixed.
- **Whether the regression captures the right "IQ"**. The 6 features are a
  hand-picked set; the encoder may use IQ-correlated nonlinear patterns
  (e.g. high-frequency Fourier band amplitudes per Probe 6) that the linear
  regression on these features misses. The expanded_10 set adds ~0.04 mean
  R²; a kernel regression or learned linear probe on encoder features might
  show higher R² and reframe LOCKBOX from "moderate IQ-shortcut" to
  "near-total IQ-shortcut".
- **Whether content-channel residual AUC is causally usable**. The residual
  is the projection of score onto the IQ-orthogonal subspace; it is not the
  prediction of a "content-only model". A trained content-only model might
  perform very differently from the residual.

## 7. Concrete next step proposal (for user authorization)

Two paths the user can choose between:

**Path A — gather one more CPU signal first** (recommended).
Run the per-layer IQ probe (§4.2 CPU diagnostic D1) to refine the GRL hook
location. ~1-2 days CPU. Cost ~$0. Output: a FACTS doc + probe AUC table that
sharpens the Stage 2a yaml.

**Path B — authorize Stage 2a directly**.
Skip D1; design Stage 2a per `iq_shortcut_deconvolution_program_2026-05-08.md`
§4.2 with the GRL hook on the layer-11 CLS by default. ~$30–50 Vertex spend.
Risk: GRL applied at the wrong layer either fails to bite (P15 outcome) or
catastrophically regresses Pillar 1 (P18C outcome). The D1 probe is what
distinguishes those failure modes.

I don't have GPU spend authorization either way. The Stage 1 measurement is
done; the next step is the user's call.

## 8. Cross-references

- FACTS: `IQ_DECOMP_FACTS_2026-05-08.md` (this directory).
- Proposal: `docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md`.
- Memory:
  - `project_image_quality_shortcut.md`
  - `project_iq_gating_viability_2026-05-04.md`
  - `project_per_layer_divergence_2026-05-06.md`
  - `project_dor_drift_named_axes_2026-05-06.md`
  - `project_p18_diagnostics_complete_2026-05-02.md`
  - `feedback_decision_points.md`
- Predecessor frames superseded:
  - The proposal §4.1 binary R²-threshold criterion (uniform across cells)
    is replaced here with a substrate-conditional close criterion. The
    proposal's framing was a starting point that the data refined; not a
    framing that turned out to be wrong, but a framing that needs the
    substrate axis added before it is actionable.
