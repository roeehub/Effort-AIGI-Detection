# Best-candidate search 2026-05-13 — VERDICT

## TL;DR

Yes, there IS a strong candidate, and it is **T5C_step3500** — but the headline
"≥90% fake recall at ≤5% FPR" only holds if we accept dropping a small
chronic-identity cohort that is largely already filtered out by the production
face-detector + min-resolution pipeline. Specifically:

| Condition (what we drop from eval) | T5C macro fake recall @ FPR=5% | T5C macro fake recall @ FPR=10% |
|---|---|---|
| Nothing (F0 raw, 4564 frames)                                                     | 42.84% | 76.62% |
| Production filter only (min(W,H)<200, is_no_face — 2527 frames remain)            | **32.91%** | 82.25% |
| Production filter + drop bla_bla_chow (sharp screen-capture, 312 frames)          | 33.67% | 89.24% |
| Production filter + drop bla_bla_chow + bla_bla_chow__s2 (already lowres-dropped) | — | — |
| Production filter + drop bla_bla_chow + roy_d (2 chronic identities, 442 frames)  | 90.56% | 95.02% |

The jump from 33.67% to 90.56% at FPR=5% comes from dropping **roy_d**. Why is
roy_d binding?

- roy_d has **no parquet IQ coverage** (we have not characterized its substrate
  on the IQ axes).
- On T5C_step3500, roy_d real-frames score **96.9% FPR at deployment tau** — i.e.,
  the encoder essentially classifies all roy_d reals as fake. This is a known
  recipe-specific failure: the T3 SLOT1 data lever (drop-top-25%-high-IQ-teams-reals)
  collapses the roy_d cluster on T3/T5C-FT families; P8A and SLOT1-LoRA-from-P8A
  (today's batch) handle roy_d cleanly (38.5% / 13.8% FPR respectively).

So the "T5C hits 90% at 5% FPR" claim is technically: **conditional on production
not seeing roy_d-substrate-equivalent frames.** Since roy_d represents 2.85% of
dev pool with unknown parquet substrate, this assumption needs a Pillar-3
robustness verdict before deployment.

## How extreme are the chronic identities?

| Identity | n in dev | %pool | min(W,H) p50 | %<200 px | Capture mode | What makes it extreme |
|---|---|---|---|---|---|---|
| pc_generator__s22 | 227 | 5.0% | 88  | 100% | webcam        | TINY face (88 px) — out of distribution on face_pixel_area; already filtered by production min-res |
| pc_generator__s45 |  91 | 2.0% | 90  | 100% | webcam        | TINY face (90 px) — same; already filtered |
| q__s6             |  54 | 1.2% | 94  | 100% | webcam        | TINY face (94 px) — same; already filtered |
| bla_bla_chow__s2  | 180 | 3.9% | 146 |  96% | phone_screen  | sub-200 px (production-filtered) + low sharpness (55 Laplacian) |
| bla_bla_chow      | 311 | 6.8% | 399 |   2% | screen        | OPPOSITE-end OOD: very wide face, very sharp (525 Laplacian), screen-capture mode |
| roy_d             | 130 | 2.9% | NA  |  NA  | (no parquet)  | recipe-specific failure on T3-FT family; P8A handles it fine |

**Production-filter overlap:** F2 (min(W,H)<200) already drops all 3 tiny-face
chronics and bla_bla_chow__s2 — so 4 of the 6 chronic identities are
**production-automatic-drops** (the face-detector + min-resolution filter
already exclude them before they reach the model).

**Net "voluntary" identity drops** that the F4 contract adds beyond F2:
- bla_bla_chow (screen-capture, sharp, wide face) — 311 frames
- roy_d (no parquet) — 130 frames

These 441 frames (9.7% of dev real pool) are the entire delta between
"production-realistic substrate" and "F4-cleaned substrate."

## Full finalist scoreboard (deep-dive)

Sorted by F4 macro recall @ FPR=5%.

| Ckpt | F0 raw recall @5% / @10% | P2 (lowres+noface) recall @5% / @10% | F4 recall @5% / @10% | Pearson F4 (shortcut, |r|→0 best) | AUC F4 (forgery signal) | Min FPR-F4 for 90% recall |
|---|---|---|---|---|---|---|
| **T5C_step3500**      | 42.8 / 76.6 | 32.9 / 82.2 | **90.6 / 95.0** | **-0.011** | **0.982** | **4.73%** |
| P2D_fourier_step3000  | 49.3 / 71.2 | 49.0 / 72.1 | 89.2 / 94.1 | +0.049 | 0.974 | 5.50% |
| P1_pairrank_step6750  | 37.2 / 68.3 | 29.8 / 73.9 | 88.3 / 93.4 | -0.057 | 0.975 | 6.70% |
| P1_pairrank_step6000  | 36.6 / 63.9 | 27.8 / 70.9 | 87.9 / 92.2 | -0.027 | 0.974 | 7.22% |
| P1_bundle_step4000    | 35.9 / 62.2 | 19.1 / 59.6 | 85.3 / 90.6 | +0.049 | 0.967 | 9.28% |
| T3_SLOT1_step1500     | 33.7 / 60.7 | 40.2 / 76.8 | 85.1 / 89.5 | -0.064 | 0.958 | 10.76% |
| T3_SLOT1_step2500     | 23.3 / 57.0 | 17.0 / 62.2 | 84.3 / 91.8 | -0.102 | 0.958 | 7.27% |
| P1_bundle_step3750    | 32.0 / 57.7 | 14.5 / 51.9 | 84.1 / 90.0 | +0.054 | 0.964 | 10.19% |
| E3_step6600           | 52.8 / 61.5 | 52.2 / 60.5 | 73.5 / 91.2 | -0.077 | 0.944 | 9.85% |
| P2D_fourier_step8000  | 29.0 / 55.5 | 20.4 / 42.4 | 82.9 / 92.0 | -0.027 | 0.966 | 7.41% |
| **P8A_step5000**      | 23.2 / 46.4 | 54.1 / 72.5 | 72.5 / 83.9 | -0.116 | 0.941 | 17.17% |
| **E2B_step3200**      | 50.9 / 60.6 | 51.1 / 60.6 | 64.1 / 72.7 | -0.118 | 0.939 | 16.36% |

**Read this table by pillar:**

1. **Shortcut avoidance (Pearson F4, |r| → 0 is best):**
   T5C −0.011 < P2D_step8000 −0.027 = P1_pairrank_6000 −0.027 < P2D_3000 +0.049
   < P1_pairrank_6750 −0.057 < T3_SLOT1_1500 −0.064 < E3_6600 −0.077 < T3_SLOT1_2500 −0.102
   < **P8A −0.116** < **E2B −0.118**.
   **T5C is the LEAST shortcut-driven** on F4-cleaned reals — its score does
   not track face-size at all. P8A and E2B both show stronger negative
   correlation, meaning they DO use face-size as a fake predictor on the
   production-realistic substrate.

2. **Forgery signal (AUC on F4 reals vs full fakes):**
   T5C 0.982 > P1_pairrank_6750 0.975 ≈ P2D_3000 0.974 ≈ P1_pairrank_6000 0.974
   > P1_bundle_4000 0.967 ≈ P2D_8000 0.966 > P1_bundle_3750 0.964 > T3_SLOT1_*
   0.958 > E3_6600 0.944 > P8A 0.941 > E2B 0.939.
   **T5C has the strongest forgery signal.**

3. **Recall at low FPR on F4:**
   T5C is the **only candidate ≥ 90% macro recall at FPR ≤ 5%** on the F4
   substrate. The next-best (P2D_fourier_step3000) needs 5.50% FPR; T5C needs
   only 4.73%.

## Substrate-invariance trade-off

The view changes when we widen the substrate. On **P2 (lowres+noface only, no
identity drops — production-realistic)**:

| Ckpt | viso @5% | viso @10% | deeplive @5% | deeplive @10% | teams_fake @5% | teams_fake @10% | macro @5% | macro @10% |
|---|---|---|---|---|---|---|---|---|
| **T5C_step3500**     | 4.2  | 61.1 | 37.6 | 96.9 | 56.9 | 88.8 | **32.9** | **82.2** |
| **P8A_step5000**     | 34.9 | 55.5 | 52.1 | 75.8 | 75.3 | 86.2 | **54.1** | 72.5 |
| P2D_fourier_step3000 | 12.9 | 41.1 | 68.8 | 91.7 | 65.4 | 83.4 | 49.0 | 72.1 |
| T3_SLOT1_step1500    | 12.4 | 46.4 | 42.0 | 95.4 | 66.3 | 88.5 | 40.2 | 76.8 |
| E3_step6600          |  7.6 | 13.5 | 76.0 | 89.0 | 73.1 | 79.2 | 52.2 | 60.6 |
| E2B_step3200         |  4.9 |  8.4 | 76.0 | 93.9 | 72.4 | 79.4 | 51.1 | 60.6 |

On the production-realistic P2 substrate:
- At **FPR=5%**: **P8A wins** (54.1%). T5C drops to 32.9% because its score on
  bla_bla_chow + roy_d eats into the FPR budget. P8A is genuinely
  substrate-invariant.
- At **FPR=10%**: **T5C wins** (82.2%). The FPR budget is wide enough that
  bla_bla_chow + roy_d false-positives are absorbed.

This is the actual deployment trade-off. T5C wins at 10% FPR even without
identity drops; P8A wins at 5% FPR without identity drops; T5C wins both at
F4-cleaned substrate.

## What "deployable" looks like

Three honest framings:

### Framing A — strict F4 substrate (drop chronic-6)
**Pick T5C_step3500. Hits 90.6% macro fake recall at 4.73% FPR.**
- Production filtering: assume face-detector excludes tiny faces (<200 px), is_no_face frames, AND no production sources resemble bla_bla_chow (sharp screen-capture, wide face) or roy_d (T3-FT recipe failure).
- Per-suite breakdown at FPR=5%: viso 78.0%, deeplive 98.9%, teams_fake 94.8%.
- Risk: if production traffic includes a bla_bla_chow-like screen-capture
  identity, expected FPR rises to ~17% (P0 raw) instead of 5%.

### Framing B — production-realistic P2 substrate, FPR=10% budget
**Pick T5C_step3500. Hits 82.2% macro fake recall at 10% FPR.**
- Production filtering: only the standard face-detector + min-resolution + no-face filter.
- Per-suite: viso 61.1%, deeplive 96.9%, teams_fake 88.8%.
- Risk: FPR may run hot to ~17% on the in-eval substrate; in deployment depends on actual real-traffic shape.

### Framing C — production-realistic P2 substrate, FPR=5% budget
**Pick P8A_step5000. Hits 54.1% macro fake recall at 5% FPR.**
- Most substrate-invariant; doesn't degrade catastrophically with novel
  substrate.
- Per-suite at FPR=5%: viso 34.9%, deeplive 52.1%, teams_fake 75.3%.
- This is essentially "currently best invariance-bound" — the answer that
  matches the conservative "you can't assume substrate" reading.

## The non-obvious finding

The 3 most-extreme chronic identities (pc_generator__s22 / pc_generator__s45 /
q__s6) **do not require any explicit "drop list" intervention** to make T5C
work — they are already removed by the standard production min-resolution
filter. The only identities the F4 contract drops *beyond* production filtering
are **bla_bla_chow** (sharp wide-face screen-capture) and **roy_d** (no IQ
characterization yet). These two identities account for the entire gap between
Framing B (82.2%) and Framing A (95.0%) recall at FPR=10%.

In other words: **the lift from "33% to 90%" at FPR=5% is almost entirely
driven by roy_d alone** (because dropping just bla_bla_chow gives 33.67% →
33.67% range, while adding roy_d drop gives the 90.56% jump). T5C's
deployment readiness hinges on whether production traffic contains roy_d-like
substrate or not.

## Recommended next steps

1. **Characterize roy_d substrate** (face_pixel_area, sharpness, capture_mode)
   to determine if production traffic would contain it. Currently no parquet
   coverage. If roy_d is found to be a production-realistic substrate type,
   T5C is NOT deployable at FPR=5%.
2. **Run T5C through Pillar-3 robustness checks** (cross-camera/lighting/codec
   gauntlet from `project_success_criteria`). The 90.56% recall@5% FPR is
   only one pillar.
3. **Independent verification on HDTF substrate** (per project_pa_does_not_generalize_to_hdtf)
   to confirm T5C generalizes beyond v2 substrate. This is the trap that broke
   PA: same-substrate gain that did not cross-validate.
4. **Compare T5C against the deployed E2B** under the same per-substrate
   scoring (per-substrate τ-calibration from `project_per_substrate_tau_2026-05-04`)
   to estimate concrete production FPR/recall change.

## Per-stage artifacts

- `_stage1/STAGE1_SCOREBOARD.csv` — 38-candidate F4 sweep
- `_stage2/CHRONIC_IDENTITY_EXTREMENESS.csv` — per-identity extremeness + per-ckpt FPR contribution
- `_stage3/STAGE3_PROGRESSIVE.csv` — full progressive-drop matrix (P0..P6 × per-suite × FPR={5,10})
- `_stage4/STAGE4_FINALISTS.csv` — finalists with shortcut/AUC/min-FPR-for-90%-recall

## Final answer to the user's question

**"No deployable candidate from today's batch"** was a wrong default conclusion
on my part — the batch's reapply candidate (today's SLOT1_LORA) isn't strong,
but **T5C_step3500 from yesterday IS a strong deployable candidate under
relaxed substrate conditions**. The decision hinges on:

- Whether dropping bla_bla_chow (sharp screen-capture) and roy_d (recipe-bad)
  from the FPR-calibration substrate is defensible. The 3 tiny-face chronics
  do not require justification — they're already filtered by production.
- Whether T5C's substrate-invariance gap vs P8A on novel substrates is
  acceptable, given that P8A on F4 substrate hits 72.5% recall (vs T5C 90.6%)
  at the same FPR.

The choice between Framing A (T5C), Framing B (T5C), and Framing C (P8A) is a
deployment-substrate-assumption call, not a model-quality call. T5C is the
strongest model by every pillar metric on F4-cleaned data; P8A is the strongest
on widest-substrate-with-tight-FPR data.
