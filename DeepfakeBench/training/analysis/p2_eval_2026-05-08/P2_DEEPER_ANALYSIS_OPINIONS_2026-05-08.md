# P2 Phase A — synthesis OPINIONS (2026-05-08)

> **Status: OPINION + SUGGESTION.** Reasoning, mechanism debate, next-experiment
> proposal. Not a verdict. Past framings on this codebase have been reversed
> (P14 bundle 5.7×, P18 GRL "biting" verdict, viso ceiling reframed twice).
>
> **Companion FACTS docs**:
> - [`P2_PHASE_A_VERDICT_FACTS_2026-05-08.md`](P2_PHASE_A_VERDICT_FACTS_2026-05-08.md) — promotion contract scorecard
> - [`P2_DEEPER_ANALYSIS_FACTS_2026-05-08.md`](P2_DEEPER_ANALYSIS_FACTS_2026-05-08.md) — J1-J5 characterization
> - [`../iq_shortcut_decomp_2026-05-08/IQ_DECOMP_FACTS_2026-05-08.md`](../iq_shortcut_decomp_2026-05-08/IQ_DECOMP_FACTS_2026-05-08.md) — Stage 1 R² probe
> - [`../iq_shortcut_decomp_2026-05-08/IQ_DECOMP_OPINIONS_2026-05-08.md`](../iq_shortcut_decomp_2026-05-08/IQ_DECOMP_OPINIONS_2026-05-08.md) — sister-agent's Stage 2 reading
>
> **Wiki anchors**: [`MODEL_GOALS.md`](../../docs/packet_retrospectives/MODEL_GOALS.md),
> [`SCORECARD_GUIDE.md`](../../docs/packet_retrospectives/SCORECARD_GUIDE.md),
> [`threads/iq_shortcut_deconvolution_program_2026-05-08.md`](../../docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md).

---

## 1. Synthesis — what we now know (post-convergence)

**Three independent measurements converged**:
1. **P2 Phase A verdict**: P8A holds production rank 1; D step3000 has rank-4 with the strongest recall signal (dev_fake_macro_recall 0.530, lockbox_fake_recall 0.889) AND a Pillar-2 regression (lockbox_real_fpr 0.164, 9× E2B; teams_real_dor_dev real_fpr 0.460, ~6× P8A).
2. **J1-J5 deeper analysis**: D step3000's behavior is non-uniform across identities (regressions on Dor + Noyn_Sharker; lifts on cam_test + lockbox-fake) and non-uniform across substrates (recall lift concentrated on smooth lockbox; preserved at FPR-matched τ).
3. **Stage 1 IQ R² probe**: Hypothesis A (model is mostly an IQ classifier) is **rejected** — residual AUC stays high (>0.6) on every cell. Hypothesis B (IQ + content compose) is **confirmed** with substrate-conditional structure: HDTF R² 1-9% (content-dominant), LOCKBOX R² 40-54% (IQ-dominant), DEV R² 16-26% (middle).

**Key unifying finding** (opinion): D step3000's lockbox R² is **the highest of any ckpt** (0.542 vs E2B 0.454 vs P8A 0.397). The Fourier intervention amplified the encoder's IQ-axis sensitivity on the lockbox substrate. The recall lift on lockbox (38.7% → 88.9%) is partially "real content channel improvement" (residual AUC 0.633 stays above chance) and partially "IQ shortcut firing harder on a smooth substrate" (R² jumps from P8A's 0.40 to D's 0.54).

This **falsifies the original framing** that interpreted the verdict as
either "Fourier produced real content lift" (mechanism A) or "Fourier
amplified IQ" (mechanism B) as alternatives. The data says **both
mechanisms are in play, with weights that vary by substrate**.

## 2. The Dor regression — a separate signal

J4 surfaced a finding that's NOT explained by the IQ-shortcut frame:

- D step3000 `teams_real_dor_dev` real FPR: 46% (vs P8A 8%, vs E2B 12%) — **5.75× P8A**
- D step3000 `teams_capture_dor_shkedi_dev` fake recall: 62% (vs P8A 86%) — **DROPS 24pp**

The Dor cohort fails BOTH directions: more false positives on real Dor frames AND fewer correct catches on Dor fakes. This is not "encoder more aggressive on smooth = fake" (which would only inflate FPR, not also drop recall). **D step3000 specifically lost something about Dor identity discrimination** that P8A had.

Memory `project_p8a_breakthrough.md` records P8A's signature property as
`0/1170 real_lockbox_dor_FPR` — Dor invariance was P8A's unique strength.
**D step3000 partially destroyed this signature property.** Whether the
Fourier intervention caused the loss, or the Fourier-from-scratch training
(no P8A FT init) lost it, is undetermined from current data — but it's a
SEPARATE failure mode from the IQ-shortcut amplification.

The Dor regression is upstream of any IQ-axis intervention — Stage 2a
(IQ GRL) is unlikely to fix it because Dor isn't where the IQ-shortcut
binds (LOCKBOX is).

## 3. The viso-ceiling reframe

D step3000 broke the 13+ packet viso recall ceiling at contract τ
(visomaster_enhanced_macro_dev recall 16.5% vs P8A 13.5% — first
material lift). This was the original motivation for celebrating D step3000.

**Stage 1 IQ-decomp DEV_VISO_VS_REAL R² for D step3000 = 0.142** (low).
This is NOT the high-IQ-shortcut substrate. The viso-ceiling break is
**mostly content-channel-real**, not IQ-shortcut.

But: there's a critical missing measurement. **Phase C HDTF was not run
for P2-D**. The viso recall on HDTF (where reals + fakes are uniformly IQ-clean)
would distinguish "viso ceiling broken at the content level" from
"viso ceiling broken on an IQ-noisy substrate". Memory
`project_pa_does_not_generalize_to_hdtf_2026-05-05.md` records PA's
v2-substrate lift collapsing on HDTF — same hazard applies here.

## 4. Mechanism summary across substrates

| substrate | D step3000 mechanism | confidence |
|---|---|---|
| Lockbox fake (smooth) | IQ amplification dominant + content channel real | high |
| Lockbox real | IQ amplification = FPR cost | high |
| visomaster_enhanced_macro_dev | Content-channel-real lift (low R²) | moderate (no HDTF readout) |
| deeplive_enhanced_dev | Content-channel-real (low R²) but D step3000 doesn't beat E2B here | moderate |
| Dor cohort | Identity-specific regression, not IQ-mediated | moderate |
| Most capture_<identity>_dev | Comparable to P8A or marginally better | high |
| HDTF | Unknown — not measured for P2-D | (gap) |

## 5. Sister-agent's Stage 2 recommendation — refined

The Stage 1 OPINIONS doc (`IQ_DECOMP_OPINIONS_2026-05-08.md` §4) recommends
**Stage 2a (IQ GRL)** with a substrate-conditional close criterion:
- LOCKBOX R² drops ≥0.10
- DEV residual AUC stays ≥0.85
- HDTF residual AUC stays ≥0.95
- DEV_DEEPLIVE raw AUC doesn't regress >0.05

The sister-agent also recommends a **Path A pre-launch CPU diagnostic**:
per-layer IQ probe (where in the encoder is IQ represented?) to refine
the GRL hook's target layer.

I agree with both. **I would add two additional pre-Stage-2a checks**:

### Check 1 — D step3000 Phase C HDTF (small GPU spend, ~$15-25)

Without HDTF readout for D step3000, the viso-ceiling break is unverified
on the IQ-clean substrate. PA's v2-lift collapsed on HDTF
(`project_pa_does_not_generalize_to_hdtf_2026-05-05.md`); the same hazard
is live for D step3000. **HDTF Phase C for D step3000 is the cheapest
single-ckpt measurement that disambiguates "v2-substrate-bound lift" vs
"general content-channel lift" before any Stage 2a packet design.**

If D step3000 retains its viso lift on HDTF: Fourier is a real content
intervention; Stage 2a starts from D step3000.
If D step3000 collapses on HDTF (PA-style): Fourier is substrate-bound;
Stage 2a should start from E2B base instead, OR pivot to a different
data-curation lever (Stage 4 from the proposal: drop celeb_real /
youtube_real from training).

### Check 2 — Dor cohort per-frame characterization (CPU, ~$0)

The Dor regression is structural and not IQ-explained. Before launching a
Stage 2a packet from D step3000, we should know:
- Is the regression on D step3000 unique to step3000, or present at all
  D ckpts?
- Is it present at all P2 ckpts (does the from-scratch C step3000 also
  regress on Dor)?
- Is the Dor representation in D step3000's encoder shifted relative to
  P8A's, and on which axes?

J4's per-identity recall table partially answers Q1+Q2 (D step8000/19000
also drop on dor; C step3000 drops MORE on dor than D step3000). The
encoder-axis question (Q3) needs CPU work — extract per-frame features for
Dor frames from D step3000 vs P8A and compare.

If Dor regression is a from-scratch artifact that all P2 ckpts share, FT
from P8A may be required for any Stage 2a packet. **This affects FT-base
selection for Stage 2a directly.**

## 6. Convergence — proposed sequence

The user authorized continuing this path. Both my J1-J5 and the sister-
agent's Stage 1 are in. Three artifacts gate Stage 2a:

| step | type | ETA | $ | informs |
|---|---|---|---|---|
| (a) Per-layer IQ probe (sister-agent's §4.2) | CPU | 1-2 days | $0 | GRL hook layer choice |
| (b) D step3000 Phase C HDTF | GPU | ~2-4h | ~$15-25 | FT-base selection (D vs E2B) |
| (c) Dor encoder-axis characterization | CPU | half-day | $0 | FT-base selection (FT-from-P8A vs from-scratch) |

These three are independent and can run in parallel (a and c on local CPU
while b runs on Vertex). After all three land, Stage 2a packet design has
its remaining unknowns resolved.

### Stage 2a packet design (after a/b/c complete)

**Single-lever discipline** (per `project_face_scale_jitter_load_bearing.md`
and AGENT_GUIDE Rule 1): each Stage 2a slot tests ONE delta vs a
controlled baseline.

**Proposed slate** (3 slots, ~$60-80 total):

- **Slot 1 — STAGE2A_BASE**: FT-from-best-base (D step3000 OR E2B,
  per (b)/(c) verdict) with Fourier band-amp aug retained from D step3000's
  recipe. **No GRL.** This is the baseline for Slot 2.
- **Slot 2 — STAGE2A_IQ_GRL**: Slot 1 + IQ-axis GRL on the layer chosen
  by (a). Single-lever delta. Close criterion per sister-agent's §4.3.
- **Slot 3 — STAGE2A_IQ_BALANCED_SAMPLER**: From-scratch (or FT) with
  IQ-balanced batch sampler (sister-agent's §4.1 alternative). Tests
  whether the sampler-level intervention beats the GRL-level intervention.
  Single-lever delta from Slot 1.

If (b) shows D step3000 collapses on HDTF → Slot 1's FT-base is E2B
instead of D step3000.

If (c) shows the Dor regression is a from-scratch artifact → all three
slots use FT-from-P8A as the base and start from there with the lever.

### What this is NOT

- **Not committing to Stage 2a today.** The decision is the user's; the
  three checks (a/b/c) are CPU-mostly with one small GPU and produce the
  evidence the user needs to authorize Stage 2a's $60-80.
- **Not abandoning the original plan.** Fourier is in the slate (Slot 1).
  Stage 2a tests "does adding IQ debiasing on top of Fourier improve the
  FPR while preserving the recall lift?" — directly answering the question
  of the verdict.
- **Not solving the Dor regression with a single packet.** Dor is a
  separate axis; Stage 2a may co-incidentally help (if IQ debiasing
  reduces over-firing) or not. A dedicated Dor-axis intervention may be
  Stage 3.

## 7. Risk inventory

(Honest. Each carries a CPU-cheap diagnostic that would update before GPU.)

| risk | how it manifests | diagnostic that catches it |
|---|---|---|
| GRL targets wrong layer | P15-style "didn't bite" outcome on Slot 2 | check (a) per-layer IQ probe |
| D step3000 is substrate-bound (PA-like) | Stage 2a from D base inherits the substrate-binding | check (b) D HDTF Phase C |
| Dor regression baked into from-scratch P2 lineage | Stage 2a from D OR C step3000 inherits the Dor regression | check (c) Dor encoder-axis characterization |
| IQ debiasing destroys Pillar 1 recall | Slot 2 dev_fake_macro_recall drops below E2B's 0.509 | sister-agent's close criterion §4.3 |
| Sampler-bin artifacts | Slot 3 batch composition collapses to one IQ regime | implementation review pre-launch |
| Fourier + IQ-GRL cancel | Slot 2 ties Slot 1 | Slot 1 vs Slot 2 single-lever ablation IS the test |

## 8. The bigger frame

If Stage 2a Slot 2 succeeds (LOCKBOX R² drops ≥0.10, HDTF residual AUC
stays ≥0.95, DEV residual AUC stays ≥0.85): **we have validated the
distribution-level intervention thesis** — frame-level augmentation has
a co-equal at the loss level for IQ-axis robustness. Future packets gain
a new lever class.

If Stage 2a Slot 2 fails (cancellation with Slot 1 OR Pillar 1 collapse):
**Stage 2b (IQ-balanced sampler) becomes the next move** OR Stage 4
(data curation: drop smooth training reals). The R² probe results say
the IQ-shortcut is real on lockbox; if GRL doesn't bite, the sampler-
or-data path is the alternative.

If both Stage 2a AND 2b fail: the IQ-deconvolution program enters
"residual content channel is what we have; eval substrate must shift to
HDTF-canonical to read it correctly" mode. That's Stage 3 (HDTF as
canonical eval per the proposal §4.3) — a paradigm reframe more than a
new packet, with implications across the entire packet retrospective set.

## 9. Cross-references

- **Companion FACTS docs**: `P2_PHASE_A_VERDICT_FACTS_2026-05-08.md`,
  `P2_DEEPER_ANALYSIS_FACTS_2026-05-08.md`,
  `../iq_shortcut_decomp_2026-05-08/IQ_DECOMP_FACTS_2026-05-08.md`,
  `../iq_shortcut_decomp_2026-05-08/IQ_DECOMP_OPINIONS_2026-05-08.md`.
- **Memory entries that should be updated post-Stage-2a**:
  - `project_p8a_breakthrough.md` — Dor invariance signature loss in P2-D.
  - `project_image_quality_shortcut.md` — substrate-conditional R² structure.
  - `project_iq_shortcut_decomp_stage1_2026-05-08.md` (just added by
    sister-agent).
- **Wiki**:
  - [`threads/iq_shortcut_deconvolution_program_2026-05-08.md`](../../docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md) §7 — append entry for this synthesis.
  - [`STATE.md`](../../docs/packet_retrospectives/STATE.md) — update "Where we stand" if user proceeds to Stage 2a.
  - [`SCORECARD_GUIDE.md`](../../docs/packet_retrospectives/SCORECARD_GUIDE.md) — Stage 2a will use trajectory mode on the resulting ckpts.
