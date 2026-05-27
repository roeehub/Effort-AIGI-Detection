# AGENT_PROPOSAL_2026-05-16 — what the property analysis says, and what it doesn't

> **OPINIONS.** Same agent who ran the property analysis. Interpretation, not facts.

## §1. The user was right; my prior framing was wrong

I framed the lockbox over-fires as "identity-localized". That framing missed
the actual mechanism. The diagnostic confirms the user's reframe:

- The same human (`dor`) appears as `dor_shkedi` (9.9% over-fire) AND as
  `real_dor` (0% over-fire). The model is not keying on the person.
- The two tagged sets have **wildly different image properties** (Cohen's d
  −3.74 on lab_b_std; same person, different tag).
- Within `dor_shkedi` itself, the over-fires concentrate on a specific
  capture-style subset (`.jpg` rectangular crops: 14.9% rate vs `.png`
  square crops: 8.4%).

What I called "chronic-6 identities" should have been called "frames matching
a specific capture/processing signature that happens to be over-represented
in the dor_shkedi and bla_bla_chow tags". That's the actual axis.

## §2. The specific shortcut, and why it's worse than I thought

### What the data says

`dor_shkedi.jpg` crops have all of:
- **Rectangular aspect ratio** (median 499×394, never square).
- **JPEG compression** (median 292kB file size — 3× the typical .png).
- **Very low color variation** (lab_b_dev median 3.9 vs png 13.8 vs real_dor 26.0).
- **Very low skin fraction** (0.19 vs png 0.60 vs real_dor 0.67 — meaning these crops
  have lots of non-face content / background / hair).
- **High sharpness** (Laplacian median 422 vs png 201).

The model fires hardest on this subset (14.9% over-fire). The `.png` square
crops of the same person have moderate values across the board and a
moderate over-fire rate. `real_dor` crops are all `.png`, square, with the
HIGHEST color variation and 0% over-fire.

### Why this is worse than expected

I had assumed the shortcut was a continuous property axis (sharpness, luma,
color stats) that we could augment against. The data says it's a
**multi-property capture-signature** that includes aspect ratio, compression
type, and crop-content composition. These are discrete pipeline
characteristics, not continuous augmentable axes.

The cross-identity classifier transfer test (RESULTS_FACTS §5) makes this
even sharper: a continuous-feature classifier trained on dor_shkedi
over-fires predicts 96-98% over-fire probability on `real_dor` and
`PC_Generator` — where actual is 0%. The 11 continuous properties I
extracted CANNOT distinguish over-firing from non-over-firing cohorts
across identities. The actual signal lives in higher-order features
(compression artifacts at JPEG quantization boundaries, the implicit
edge structure of rectangular crops resampled to 224×224, etc.) that
the deep model is reading and my hand-crafted features are missing.

## §3. What this means for production

### Inference-side rescue is now dead-dead

The earlier rule-tweak idea (`count_above_0.92` instead of `count_above_0.9`)
would rescue the 5-identity lockbox sample but does NOT address the actual
shortcut. The shortcut is "this set of crops looks like a specific capture
pipeline" — and that capture pipeline (rectangular JPEG with low color
content) is exactly what you'd see in production webcam captures with
non-square video frames and JPEG transport. A wisdom rule that suppresses
overfires on the lockbox won't help when production traffic has the same
capture signature.

### Slot β as a deployment candidate is more fragile than it looked

In §1 of the prior `AGENT_PROPOSAL_2026-05-16.md` (the slot_b_per_identity
folder), I framed Slot β + tweaked rule as "near-shippable". The capture-
signature finding makes me less confident about this. Slot β's lockbox
overfires concentrate on the capture-signature pattern that production WILL
see (rectangular jpg crops with low color content), and the rescue
mechanism only works because the lockbox lets us match by known identity.

In production, we won't have "this frame is from dor_shkedi" as a label —
we'll have whatever capture signature comes off the user's webcam. If that
signature matches the dor_shkedi.jpg profile, Slot β will over-fire and the
rule won't catch it.

### Training-side options

The actual training-side fix needs to disrupt **capture-signature encoding**,
not individual continuous properties. Three candidate levers:

**Lever 1: Aspect-ratio + compression augmentation.** Equivalent to
`resolution_chain_aug` but on the capture-pipeline axis. Random
padding/cropping to varying aspect ratios + JPEG re-encode at random
quality. Could compose with the existing `resolution_chain_aug` from
2026-05-15.

**Lever 2: Bucket-level domain randomization.** During training, randomly
re-encode crops through different capture pipelines (vary aspect, quality,
chroma subsampling) so the model can't use pipeline signatures.

**Lever 3: Architectural — invariance pretext task.** Train the model to
produce the same encoder output for a frame across many capture pipelines
(contrastive loss across re-encoded variants). This is the "true
invariance" version of what `multi_axis_grl` was trying to do but on the
right axis.

All three are GPU work. I do not recommend launching any without first
addressing Caveat A below.

## §4. Caveats on this very finding

### Caveat A: only 5 identities in the lockbox

The whole property analysis is on 5 identities, only 2 of which have any
over-fires. The capture-signature hypothesis is built on contrasting
`dor_shkedi.jpg` (n=275) vs `dor_shkedi.png` (n=895) vs `real_dor.png`
(n=109) — three subsets. Before any training-side action, this should be
cross-validated on a wider real cohort. Specifically:

- Run the same property-decomposition on `teams_real_all_dev` real cohort
  (~2000 frames, broader identity coverage).
- See whether non-chronic identities with `.jpg` rectangular crops at
  similar property values also over-fire at elevated rates on Slot β.

If yes → capture-signature axis is real and a single training-side packet
is justified. If no → the property pattern is dor_shkedi-specific and the
mechanism may be subtler still.

### Caveat B: file extension is a proxy, not the actual signal

The `.jpg`/`.png` split tracks aspect ratio + compression in this lockbox,
but file extension itself isn't what the model reads. The actual signals
the model uses are JPEG quantization artifacts, aspect-distorted resampling
patterns when the model resizes to 224×224, and possibly the implicit
luma/chroma down-sampling that JPEG does. These need to be characterized
on actual image data before we can design the right augmentation.

### Caveat C: P8A doesn't show the same pattern

P8A also has chronic-FPs (`Chikara_Takahashi`, `PC_Generator`) but those are
single-extension (`.jpg`) populations and the property pattern §4 found for
Slot β doesn't apply to P8A (sharpness ρ flips: dor +0.28 vs Chikara
−0.60). Whatever shortcut P8A is using is different from Slot β's. We
should not assume one fix addresses both.

## §5. Recommended next steps, prioritized

These are options for you. I am not autonomously authorized to take any.

**Step 1 — Cross-validate on teams_real_all_dev (~$0, ~3h CPU).**
Same property analysis on the broader dev cohort. If we see the same
"rectangular JPG = elevated over-fire on Slot β" pattern across more
identities, the capture-signature hypothesis becomes shippable as a packet
spec. If not, we found a 5-identity quirk and the work is informative but
not actionable. This is the only step I'd run before any GPU work.

**Step 2 — Property analysis on training data (~$0, ~4h CPU).**
Compute the same 11+aspect+extension properties on a sampled training
batch. If the training data's REAL frames are ~100% square `.png`, that
explains the shortcut: the model never saw rectangular `.jpg` real crops
during training and learned to flag them as out-of-distribution. This
would point to a data-augmentation packet (Lever 1 above) as the right
intervention.

**Step 3 — Single-lever aspect+compression augmentation packet (1 GPU slot,
~5h).** Only after Steps 1 and 2 confirm the mechanism and the gap. Same
recipe as `resolution_chain_aug` but on capture-pipeline axis. FT from
T5C step3500, single-lever, with the per-step canary probe enabled (to
catch the recall-floor regression earlier than we did with Slot α). The
yaml-template canary-default open loop becomes load-bearing here.

**Step 4 — Stress test on production traffic if available.** If we have any
real production-captured frames or screen-recording footage from actual
Teams sessions, run all three current ckpts (P8A, T5C, Slot β) on them
and check whether the capture-signature shortcut transfers. This is the
deployment ground-truth and would change priority on Steps 1-3.

## §6. What I do NOT recommend

- Shipping Slot β + tweaked rule. The rule fixes the lockbox sample but
  not the actual mechanism that will recur in production.
- Launching another resolution_chain_aug variant. The capture-signature
  axis is orthogonal to resolution-chain; a different aug is needed.
- Launching another GRL-axis-extension. The 4 → 6 axis extension (Slot β)
  added color_b but that's the axis where dor_shkedi already has the
  LOWEST values; the GRL didn't decorrelate the encoder from the axis it
  was supposed to neutralize.

## §7. Self-correction log

In this session I framed two things that the analysis subsequently invalidated:

- **"Slot β over-fires are identity-localized; per-identity rule rescues."**
  → AMENDED. The over-fires are localized to a *capture signature* that
  happens to be concentrated in 2 identity tags but is not actually about
  identity. The rule rescue works only because the lockbox has known identity
  labels; in production it would not work.
- **"Slot β + tweaked rule is near-shippable."** → WEAKENED. The mechanism
  the rule rescues from is the same mechanism that will hit production
  traffic with no identity label to match. Shipping Slot β depends on
  whether the capture-signature shortcut survives on the broader cohort
  (Step 1).
- **"Hand-crafted properties might be the shortcut axis."** → REFUTED.
  Continuous properties don't transfer (RESULTS_FACTS §5). The signal is
  in higher-order capture-pipeline features that need direct
  characterization, not 1-11 univariate axes.

## §8. Memory update suggested

A new memory entry is warranted: the "image-quality shortcut" finding
(memory `project_image_quality_shortcut`) should be amended — the shortcut
isn't sharpness/luma alone, it's a multi-property **capture-pipeline
signature** including aspect ratio + compression artifacts that hand-crafted
property extraction does not fully capture. This is upstream of the
sharpness/luma observation and constrains what training-side fixes are
viable.
