# AGENT_PROPOSAL_v5 — refined synthesis after encoder probe

> **OPINIONS.** Final synthesis. v4's proposed augmentation intervention is
> mechanistically refuted by encoder evidence. Replacement options below.

## §1. What just changed

In v4 I proposed compositional augmentation to push training reals into the
6+ band region. The Follow-up 1 + Follow-up 2 results overnight refute this:

- Follow-up 1: the aug CAN produce frames in the 6+ band region (49% reach
  it). But at unrealistic property values (sharpness ~5 vs Roy_D ~67).
- Follow-up 2: more decisively, the augmented samples do NOT move into
  Roy_D's encoder-embedding region. PC1=+3.1 (aug) vs +3.5 (original training
  reals) vs -3.5 (Roy_D). Only 6% improvement in the share of augmented
  samples that's closer to Roy_D than to clean controls.

The encoder is keying on something the 11 hand-crafted properties are
**correlated** with but not equivalent to. Augmenting the surface property
values does not move the underlying encoder feature representation.

## §2. What the evidence base now supports

Hard finding: The vanilla pre-trained OpenCLIP ViT-B-16 encoder ALONE — before
any fine-tuning — already separates Roy_D from ilan/orel at AUC 1.00 and
predicts Slot β over-fire at AUC 0.88. The "shortcut" is not introduced by
training; it's inherited from pretraining.

Implication: any model fine-tuned on this backbone will inherit this axis.
The training procedure can only sharpen or dull the decision boundary that
USES this axis; it cannot remove the axis itself.

## §3. Revised intervention options

Given that the encoder's discriminating axis is inherited from pretraining
and not addressable by hand-crafted-property augmentation:

### Option A — data acquisition (the strongest mechanistic lever)

Add real video data of identities similar to Roy_D / dor_shkedi / bla_bla_chow
to the training pool. NOT augmentation of existing reals — actual fresh
captures with the same lighting, capture pipeline, identity diversity that
makes Roy_D fail. This directly populates the encoder embedding region with
real labels.

Cost: depends on access to additional video sources matching the chronic-FP
profile. Could be small (10-50 new identities recorded in similar conditions)
or large depending on diversity required.

### Option B — contrastive anchor loss tying same-person variants

`dor_shkedi`, `dor`, and `real_dor` are the same human. They produce different
encoder embeddings. A contrastive loss that pulls these together (using known
identity-pair information) would force the encoder to NOT discriminate
between them and reduce the Roy_D-style axis.

Cost: training-side packet, ~$5-8 GPU. Requires identity-pair labels in the
training data (memory `project_clean_teams_same_identity` says these exist).

### Option C — different backbone

OpenCLIP datacomp_xl was the pre-training. A backbone pre-trained on a more
diverse + balanced face dataset would not have learned the Roy_D-style
discrimination axis. Candidates: face-specific backbones (ArcFace, FaceNet,
CLIP-FaceX) trained on identity-balanced face data.

Cost: requires re-training from a new backbone, ~$20-50 GPU.

### Option D (acknowledge limits)

Accept that the chronic-FP cohort will continue to over-fire and add
identity-level handling at the deployment / wisdom-rule layer. This is
operationally what's already happening — the wisdom rule per memory
`project_blend_unsharp_lever_2026-05-14` is doing this.

Cost: zero. Risk: production identities not in the wisdom rule will still
over-fire (Roy_D is in dev but production traffic will have new identities
in similar profiles).

## §4. Which option I think is highest-EV

**Option B (contrastive same-person anchor loss)**, gated on Option A being
unavailable. Reasoning:

- The encoder DOES distinguish Roy_D from clean, by AUC 1.0. The
  representation gap is large.
- The encoder also distinguishes dor_shkedi from real_dor (same person) —
  this is a pure spurious axis with no semantic basis.
- A contrastive loss that says "dor_shkedi and real_dor must have the
  same encoding" would force the encoder to collapse the axis that
  distinguishes them. If that axis is what Roy_D occupies, the model
  would also stop firing on Roy_D-style frames.

This is mechanistically different from past packets:
- Past GRL packets tried to make the encoder invariant to BINARIZED
  properties via gradient reversal. They failed (memory `project_lora_l10_l11_chronic_fp_hard_coupled_2026-05-15`).
- The contrastive same-person anchor uses ACTUAL same-identity labels as
  the invariance constraint, not binarized properties. The supervision
  signal is much stronger.

It requires reading the training data to confirm enough same-person pairs
exist across capture conditions. Memory says this is in place
(`project_clean_teams_same_identity`).

## §5. CPU-only diagnostic that should run before Option B

Before launching a GPU packet for Option B, one more CPU diagnostic:

**Probe whether existing training data has enough same-person variation
across capture conditions to anchor the loss.** Pull a sample of training
samples, group by identity, compute encoder embeddings per-frame, measure
within-identity / across-condition variance. If within-identity variance
is already low (i.e., the same person looks the same in all their training
crops because the training pipeline doesn't vary capture style enough),
then Option B's contrastive supervision signal isn't strong enough.

Cost: ~$0, ~2h CPU.

## §6. Cumulative retraction log for this session

| earlier claim | current status |
|---|---|
| "Slot β is identity-localized" | False — capture-property bands |
| "Hand-crafted properties miss the shortcut" | False — they predict at AUC 0.89 |
| "OOD-band hypothesis explains it" | Partial — bands are correlated, training has zero reals there |
| "Compositional augmentation walks frames into Roy_D region" | **False** — encoder doesn't see them as Roy_D-like |
| "Augmentation packet is the next lever" | **False** — encoder evidence refutes this |
| "Contrastive same-person anchor loss is the next lever" | Provisional — needs §5 probe |

## §7. The honest summary

The chronic-FP problem is structurally tied to a pretrained-encoder axis
that distinguishes capture conditions / identities / poses. We can describe
this axis statistically (the 11-property decision tree at AUC 0.89 captures
its surface manifestation). We CANNOT manipulate it with hand-crafted
augmentation on those surface properties (Follow-up 2 confirms this).

The remaining levers are upstream: change the training data (Option A),
change the training loss to use stronger invariance supervision (Option B),
or change the backbone (Option C). The cheapest among them that has
mechanistic support is Option B, gated on §5's CPU diagnostic.

Past 13+ packets have all worked at a level below this — augmentation
hyperparameters, GRL coefficients, LoRA fine-tuning. None of them touched
the encoder's primary discrimination axis because none of them had
identity-pair supervision.

## §8. What I am NOT proposing

- Compositional augmentation into the band region (v4 proposal refuted).
- Another GRL extension or LoRA configuration tweak (these all operate
  below the encoder-axis level).
- Per-identity wisdom-rule tightening (this is Option D, already operational).
- Re-training from scratch (cost-prohibitive for a likely incremental fix).

## §9. Open question for you

Before I propose anything actionable, the structural question is whether
you accept the encoder-axis framing. The diagnostic chain is:

1. v3: bands predict over-fire at AUC 0.89 (handcrafted features ARE
   shortcut-correlated)
2. v4: training has no reals at 6+ bands (training distribution does NOT
   anchor the band region as real)
3. v5: but augmenting into the bands doesn't move the encoder representation
   (bands are not the encoder's axis)

If (3) is right, the band story (which we just spent 4h building) is true
but operationally not actionable. The actual lever is in the encoder's
inherited representation, not the surface property bands.

I have one residual doubt about (3): the probe used VANILLA openclip
ViT-B-16. The actual T5C / Slot β encoder may have shifted via FT. To
definitively confirm or refute (3), I'd need to build the SVD-aware
encoder loader and re-run the probe with the trained ckpt's encoder.
~3-4h additional engineering. Whether to do that depends on how much
weight you put on the Option B framing.
