# AGENT_PROPOSAL_v2_2026-05-16 — Roy_D reframe + multi-mode shortcut

> **OPINIONS.** Supersedes `AGENT_PROPOSAL_2026-05-16.md` after PNG-only
> reframe + dev cross-validation. Self-correction log in §6.

## §1. What this round of CPU work actually showed

1. **PNG is the production-relevant subset, and Slot β still over-fires
   ~37× more than P8A on production-relevant PNG lockbox (7.47% vs 0.20%).**
   The JPG/PNG framing flagged earlier is a useful diagnostic but doesn't
   make Slot β's PNG problem go away.

2. **Roy_D is the catastrophic production-relevant chronic FP.** On `teams_real_all_dev`
   PNG, Slot β over-fires on 79% of Roy_D frames (T5C 86%, **P8A 29%**). All three
   ckpts fail on Roy_D — this is not a Slot β-specific problem; it's a
   baseline-since-P8A problem that 13+ packets have failed to fix.

3. **The shortcut is NOT a single image-property axis.** Roy_D (dim, blurry,
   high-skin, high-color-saturation) and dor_shkedi.png (bright, sharper,
   low-color, less-skin) have OPPOSING property profiles on every measured
   axis but BOTH over-fire on Slot β. Within-cohort contrasts also reverse:
   Roy_D over-firing frames have MORE edges, dor_shkedi.png over-firing
   frames have FEWER edges.

This is mechanistically the most important finding of the night. The model
isn't using one hand-craftable image-quality shortcut — it's failing on
**multiple disjoint image regions** for different reasons that hand-crafted
property extraction does not capture as a single axis.

## §2. Why this constrains the option space

### Inference-side rescue is dead-dead-dead

I've now retracted "inference-side rescue" 3 times in 2 days:

1. First retraction (DEEP_DIVE §6, 2026-05-16 morning): inference-side
   remediation (blend/blur/CLAHE) fails the F1/F2/F3 falsifier — memory
   `project_blend_unsharp_lever_2026-05-14`.
2. Second retraction (AGENT_PROPOSAL §3, 2026-05-16 afternoon): per-identity
   rule rescue depends on identity labels that production doesn't have.
3. Third retraction (this doc): hand-crafted property gating can't even
   distinguish Roy_D from ilan/orel because the relevant features aren't
   in the 11 properties I extracted. They're in higher-order capture-
   pipeline features the model has learned and I haven't.

I should stop proposing inference-side fixes for this class of problem.

### Training-side single-lever attacks are also constrained

The 6-axis GRL Slot β experiment specifically had `color_a_approx_dev_high`
and `color_b_dev_high` as 2 of its 6 axes. Roy_D has lab_a_dev=18.0 (well
into the "high" bin per training_metadata.py thresholds). If the GRL had
bitten, Roy_D should show LESS over-firing on Slot β than on T5C. The actual
numbers:

- T5C Roy_D: 86% over-fire
- Slot β Roy_D: 79% over-fire (3.0pp better, but mostly noise on n=130)

The GRL is doing essentially nothing for Roy_D. This is the binarized-axis
problem from memory `project_phase1a_method_cluster_axis_2026-05-01` — the
encoder isn't using the binarized axis as the actual representation, so
gradient reversal on it doesn't decorrelate the real features.

## §3. What I think is actually going on

This is opinion. I think the model has learned **capture-style and
identity-cluster encoding** as a primary representation, and chronic-6
identities (Roy_D, dor_shkedi, bla_bla_chow, etc.) are clusters in this
encoding that happen to sit near the fake decision boundary. The "shortcut"
isn't a low-level property — it's a high-level capture-style + identity
cluster representation that emerges 1000+ steps into training.

Evidence:
- Property contrasts have opposite signs across cohorts (§4 of RESULTS_FACTS_v2)
- Continuous property classifier doesn't transfer across identities (96-98%
  false positive on real_dor + PC_Generator)
- All chronic-6 over-fires recur across architecturally distinct ckpts
  (B16 vs L14) per memory `project_l14_does_not_break_viso_ceiling`
- Memory `project_p17_trained_head_destroys_substrate_invariance`: even a
  layer-3 readout retrains AWAY from invariance after step 1000

If this is right, the fix isn't "augment property X harder" — it's
**preventing the encoder from forming the identity-cluster axis in the
first place**. That's a fundamentally different intervention than
`resolution_chain_aug` or `multi_axis_grl` extension.

Candidate levers that target the structural mechanism (untested in
this session, listed for option-completeness, not recommendation):

- **Hard chronic-identity oversampling**: include Roy_D, dor_shkedi,
  bla_bla_chow REAL frames in EVERY training batch with weights chosen to
  block clustering. Memory `project_lockbox_identity_looseness` notes these
  identities already appear in training as both real + fake; perhaps the
  weighting at the batch level is the lever.
- **Stronger contrastive anchor loss between same-person variants** (e.g.,
  dor_shkedi and real_dor should have similar encoder representations).
  Anchor-loss was proposed at memory `project_stage2_all_levers_regress_p8a_2026-05-09`
  but never landed.
- **Identity-shuffle augmentation in fake generation**: ensure every chronic
  identity's REAL frames are also in the fake bucket via swap (`*_clean_teams`
  pair-transport per memory `project_clean_teams_same_identity`) so the
  encoder can't use identity-cluster as a fake predictor.

These all require GPU work and reading the trainer code first to confirm
feasibility. I am NOT recommending we launch any of them yet.

## §4. What CPU work would close the next loop

The single highest-EV CPU diagnostic from here:

**Hold-out probe — encoder embedding similarity within and across same-person variants.**
Compute the T5C and Slot β encoder embeddings for:
- dor_shkedi frames (lockbox)
- real_dor frames (lockbox, same person)
- dor frames (dev, same person)
- Roy_D frames (dev)
- ilan, orel frames (dev, clean controls)

Then measure cosine similarity within-tag and across-tags-same-person. If
dor_shkedi embeddings cluster tightly together but far from real_dor (same
person, different capture), the encoder IS using capture-style as a
primary axis. If embeddings are dispersed and overlap across tags, then the
problem is downstream (head/decision boundary, not encoder).

Cost: ~$0, ~3-4h CPU (forward pass through B16 encoder on ~1500 frames + sklearn
PCA/UMAP).

This would tell us whether the right intervention is encoder-level
(invariance training) or head-level (decision-boundary calibration with
contrastive anchor).

## §5. What I am NOT going to propose

- Another property-based packet. The dev cohort dispatched that hypothesis.
- A jpeg-aug or aspect-aug packet. Roy_D is PNG + square + already in the
  training distribution geometry; the shortcut isn't there.
- Slot β + rule + tighter threshold as a ship candidate. The dev cohort
  shows Slot β is 79% wrong on Roy_D (a production-relevant identity), and
  the per-identity rule cannot rescue that — Roy_D's frac_above_0.6 is
  catastrophic; the rule will say FAKE on Roy_D.
- Re-investigating the lockbox_real_fpr tiebreak. The PNG-only reframe
  shows Slot β's actual production-relevant lockbox over-firing is 7.47%
  vs P8A 0.20%. The tiebreak isn't masking a deployability story — it's
  reading the right signal on PNG-only data.

## §6. Self-correction log

In this session I have made several framings that subsequent diagnostics
have invalidated. Cumulative retraction list:

- "Slot β + per-identity rule rescues lockbox over-fires." → RETRACTED.
  Production doesn't have identity labels (Slot β per-identity 2026-05-16).
- "JPG vs PNG is the capture-pipeline shortcut." → SCOPE-LIMITED. JPG/PNG
  is one observable signature of the model's capture-style sensitivity but
  it's not the binding constraint. Roy_D (PNG, square) is 79% over-fire on
  Slot β and 29% on P8A — same shortcut without JPG involvement.
- "Slot β is near-shippable with a small rule tweak." → RETRACTED.
  Production-relevant PNG-only lockbox FPR is 7.47% Slot β vs 0.20% P8A;
  Roy_D in dev is 79% over-fire. Slot β is not shippable.
- "Hand-crafted properties are the right diagnostic axis." → RETRACTED.
  Roy_D and dor_shkedi.png have opposing property profiles but both
  over-fire. The shortcut is in higher-order features.

## §7. The structural question for you

Given that:
- P8A has 29% over-fire on Roy_D PNG dev
- T5C has 86% over-fire
- Slot β has 79% over-fire
- All three are catastrophic
- The chronic-6 axis was identified 2026-05-04 and 13+ packets have not
  moved the needle on Roy_D

…is Roy_D in scope for the production deployment, or is it a "known
chronic-6, downstream identity-block rule covers it" identity? If the
latter, we may already be done with the resolution-chain + 6-axis-GRL
program and the choice between P8A and Slot β reduces to the existing
scorecard tradeoff. If the former, we need to address Roy_D directly,
which is GPU work targeting the encoder, not the augmentation layer.

I do not have this context; you do.
