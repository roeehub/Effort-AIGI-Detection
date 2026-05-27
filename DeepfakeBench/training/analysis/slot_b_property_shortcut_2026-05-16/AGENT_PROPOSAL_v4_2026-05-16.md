# AGENT_PROPOSAL_v4 — the mechanism, the cause, the lever

> **OPINIONS.** Final synthesis after v1 (identity reframe), v2 (PNG reframe),
> v3 (band-shortcuts found), v4 (training-distribution cause located).

## §1. Bottom line

The chronic-FP shortcut is the **multi-band intersection of low-level
properties** (lab_a_dev > 16, min_dim > 266, sharpness < 142, skin_frac >
0.88, etc.). It's caused by training data having essentially **zero real
examples in the 6+ band intersection region**. All three production
ckpts (P8A, T5C, Slot β) inherit this shortcut; their over-fire rates
differ because of how broadly each ckpt's decision boundary extends from
the "fake" side into the unanchored region.

This refutes my v2 framing ("multi-mode shortcut, hand-craftable property
gating won't help"). The shortcut IS structurally hand-craftable —
**through joint band membership, not single-property thresholds**.

## §2. The diagnostic chain

1. **v3**: 9 single-property bands each have 4-10× higher over-fire rate
   than outside. Combined, frames hitting 6+ bands have 65-100% Slot β
   over-fire and 22-50% P8A over-fire.
2. **v4**: Training reals are essentially absent (0% at n_bands=6, 0%
   at n_bands=8, 0.2% at n_bands=7) from the multi-band intersection
   region. Training fakes have some presence (0.7% at n_bands=6).
3. **The Bayes-rule P(real | n_bands=k) on training data drops from 0.66
   at n_bands=0 to 0.00 at n_bands=6.** The model's learned prior is
   exactly correct from a training-data perspective; the model is failing
   because the production distribution puts identities like Roy_D in a
   region where training never showed reals.

## §3. What this means for production

### Bad news

The shortcut is structural and built into the training data sampling. It
will recur on any model trained on this data with any reasonable
classification loss. Slot β's "regression" relative to P8A is just
"broader decision boundary into the unanchored zone" — P8A is less wrong
because its fake-side confidence is narrower, not because P8A is using
better features.

This explains why **13+ packets have failed to break the viso ceiling**
on the chronic-6 cohort. The chronic-6 happens to be the test cohort
that lives in the unanchored 6+ band region. Augmentation changes,
GRL extensions, LoRA, scratch+CE — none of them changed the
training-data distribution in the multi-band space.

### Good news

The lever now points at something concrete that hasn't been tried:
**compositional augmentation that produces training reals in the
6+ band region**. The current `pipeline_randomization` and `vcd_targeted`
augs operate on individual properties with independent probabilities;
they do not push real frames TOWARD high-density multi-band combinations.

## §4. The proposed intervention class

### What it is

A new augmentation that explicitly walks training real frames INTO the
6+ band intersection region by applying a chain of correlated transforms:
- Reduce sharpness (gaussian blur) → enters "sharpness < 142" band
- Push lab_a away from neutral (chroma rotation in a fixed direction) →
  enters "lab_a_dev > 16" band
- Push luma down → enters "luma_mean < 130" band
- Upscale via interpolation → enters "min_dim > 266" band
- etc.

Applied at training time with some probability (e.g., 10-20%) to
specifically populate the 6+ band region with REAL labels.

### What it is NOT

- It is NOT additional invariance training (Slot β tried that with GRL on 6
  axes; it didn't help because the encoder doesn't use the binarized
  axis representation).
- It is NOT a property-gate at inference (the gate would treat Roy_D as
  noise; deployment cost is high).
- It is NOT a single-property aug (sharpness alone, color alone). The
  signal IS the joint, so the aug must be joint.

### Why it might work

The training data has 470/470 reals concentrated in n_bands ≤ 5. The
model has no reason to assign "real" to anything in the 6+ region
because it never saw such reals. If we add even 50-100 training reals
that hit 6+ bands (via deliberate augmentation), the loss will start
penalizing fake-classification in that region.

P8A's 50% over-fire on 9-band frames suggests even ~50% real-anchor in
that region might be enough to halve the FPR. This is a very different
intervention than past packets.

### Why it might fail

- Augmentation-pushed reals might be visually unrealistic (oversharpened
  + over-saturated + dim), and the model may learn to ignore them or
  treat them as a distinct "augmented real" class. Need to validate by
  spot-checking augmented samples look like Roy_D-style frames, not like
  cartoon glitches.
- The augmentation might shift the FAKE band distribution too if applied
  symmetrically. Need to ensure real-only application, or that fakes are
  augmented through ADifferent path so the band signature is real-specific.
- The 11 properties might not capture the actual underlying feature the
  encoder uses. We've shown decision tree AUC of 0.89-0.99 on these 11,
  but the encoder might be reading higher-order features (JPEG quantization
  artifacts, micro-texture patterns) that happen to correlate with the
  bands. If so, hand-crafted band-targeted augmentation won't fix it.

## §5. Cheap CPU-only follow-ups before any GPU work

I do not recommend launching a GPU packet without first running these
two CPU diagnostics. Both ~$0, total ~3h:

**Follow-up 1**: Verify that Roy_D specifically (and not just "the band
region in general") is rescuable by anchoring. Take the existing 470
training reals + add 50 synthetic Roy_D-shaped reals (via the proposed
aug applied to high-band training reals). Run encoder embeddings on
both sets, check whether the augmented samples cluster near Roy_D in
embedding space. If they cluster near Roy_D, the aug is producing the
target distribution. If they cluster apart, the aug isn't capturing
the actual feature the encoder uses.

**Follow-up 2**: Probe the encoder embedding directly — does Roy_D vs
ilan/orel separate in T5C / Slot β encoder feature space? If yes, the
encoder IS using a learnable axis to distinguish them, and adding real
anchors in that region should work. If they don't separate (i.e., the
score difference comes purely from the head's threshold), the
intervention is in the head, not the encoder, and augmentation won't help.

## §6. What to retract / amend from earlier in this session

| earlier claim | status | replacement |
|---|---|---|
| "There's a single image-property shortcut Slot β fires on" | RETRACTED (v2) | It's joint-band, not single-axis |
| "No single property axis explains it" | NUANCED (v3) | True for univariate; refuted for joint band |
| "Hand-crafted properties miss the shortcut" | RETRACTED (v3-v4) | They capture it at AUC 0.89-0.99 with bands not means |
| "Roy_D is OOD on lab_a_dev" | REFUTED (v4) | Roy_D is at p95 of training reals on this axis |
| "Slot β + tweaked rule is shippable" | RETRACTED (v2) | Production has no identity labels |
| "Slot β is fragile because of capture-signature shortcut" | PARTIALLY RETAINED | Capture signature is one band among many; the joint matters more |

## §7. The structural framing for you

The reason 13+ packets have failed is now mechanically explainable: every
packet has tried to change the encoder's INVARIANCE properties (GRL,
augmentation invariance, LoRA fine-tuning) but none has changed the
underlying training DISTRIBUTION of real labels in property space.

If the actual lever is "add real-label anchors in the high-band region",
that's a data-construction packet, not a loss-design packet. Past memory
flags `project_data_axis_lever_pulled_twice_no_lift` as a refuted
direction, but those packets pulled different levers (bucket-gap, fw
weighting) — not "explicitly populate the multi-band intersection with
augmented reals".

I am not recommending we launch this packet without your sign-off. The
mechanism story is now clean enough that I think it's worth your
consideration as a distinct lever from anything previously tried, and
the CPU follow-ups in §5 would let us pre-test the aug before a GPU
spend.
