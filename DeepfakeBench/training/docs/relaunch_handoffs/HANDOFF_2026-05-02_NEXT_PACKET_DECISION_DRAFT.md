# Next-packet decision draft (post-P18 diagnostics, pre-D)

**Date drafted**: 2026-05-02 morning, while Diagnostic D is running on Vertex
(`projects/700371397073/locations/us-east1/customJobs/4550151094264659968`).

This is the next-packet recommendation matrix. Will be finalized once D returns
and the rows are filled with the actual contract numbers.

## Summary of what we know without D

CPU-only diagnostics (`HANDOFF_2026-05-02_P18_DIAGNOSTICS_COMPLETE.md`) establish:

- **P8A wins lockbox aggregate** (AUC 0.79) on the 87-frame substrate lockbox subset.
- **P18T regresses ~12pp lockbox AUC vs P8A** but is a substantial improvement over P18C
  (which dropped to AUC 0.50 ≈ chance).
- **GRL's role is defensive against FT-induced regression**, not additive over P8A.
  Without GRL, FT-only on the heavy `deeplive_teams_fake` weight (7.0) collapses lockbox
  discriminability.
- **The dor_shkedi shortcut comes from training-data composition**: bucket 3
  (`deeplive_teams`) carries the highest family weight (7.0 fake, 5.0 real). The GRL
  fights against this 7× data signal.
- **P18 trade-off vs P8A at deployment τ=0.92**: −5pp aggregate FPR, −9pp aggregate recall.

## Decision matrix — three candidate next packets

The right next packet depends on D's outcome.

### Candidate I: **P19 — same recipe but rebalance family_weights**

**Hypothesis**: GRL is biting (Diagnostic B confirmed) but is being out-competed by the
data signal. If we lower `deeplive_teams_fake: 7.0 → 1.0`, the GRL's effective gradient
on the dor axis grows relative to the data signal.

**Cost**: ~$60-100 / ~12-18h Vertex. Smallest yaml change in the 2026-04-29-onwards
packet sequence.

**Expected outcome**: P19T's lockbox AUC ≥ P18T's AUC; possibly approaches P8A.
Specifically, P19T's lockbox FPR @ τ=0.92 ≤ 0.07 AND recall ≥ 0.40 would beat P18T.

**Trigger conditions for I**:
- D shows P18T promotes OR comes very close to v3 contract floor (`target_fake_recall_min=0.70`).
- AND D shows P18T has materially better dor_shkedi suite FPR than P8A.

### Candidate II: **Move 4 — paired same-identity contrastive**

**Hypothesis**: The shortcut is identity-as-data-signal (one identity always-fake, another
always-real). Constructing training batches where the SAME identity appears as both real
(original) and fake (swapped) by construction forbids identity-as-shortcut.

**Cost**: ~2-3 days new code + ~$80-120 Vertex. Requires
- New paired-identity batch sampler that pulls (real_X, fake_X_swapped_to_Y) pairs
- Possibly contrastive loss term (cosine margin between paired samples)
- Validation that batch composition is identity-balanced

**Expected outcome**: Eliminates identity-as-shortcut by construction; should fix the
dor_shkedi pattern without requiring data-weight tuning.

**Trigger conditions for II**:
- D shows P18T does NOT promote AND is not close to v3 floor.
- AND family_weights tuning (Candidate I) is judged unlikely to close the gap (e.g.,
  if D shows P18T's recall is mostly capped by the deeplive_teams content NOT being
  in the substrate at all).
- AND Move 4 prerequisites are in place: paired identity manifests exist (memory:
  `project_clean_teams_same_identity` confirms 705 paired identities cached).

### Candidate III: **P19-Lite — same data + stronger GRL λ**

**Hypothesis**: P18 used λ ramped 0 → 0.95. Stronger λ (1.5? 2.0?) might force more
encoder change. The bucket-data-pressure is constant; doubling λ might compete better.

**Cost**: ~$80-120 / ~12h Vertex. Just a yaml flag change.

**Expected outcome**: GRL bites harder, dor shortcut shrinks more, but possibly at
recall cost (the encoder's discriminative power on Teams content gets more aggressively
scrubbed).

**Trigger conditions for III**:
- D shows P18T's bucket pattern looks correct directionally but undercrushed (e.g.,
  P18T improves on dor specifically more than P8A but not enough to swing aggregate).
- AND we want a cheap "+λ" probe before committing to Move 4.

### Candidate "0": **No new packet — declare P8A is the production model and stop FT'ing**

**Hypothesis**: Every FT-from-P8A on the current heavy-Teams-weight data has regressed
relative to P8A on lockbox aggregate. The current data weighting is incompatible with
preserving P8A's invariance through FT. The right move is to ship P8A and stop trying
to improve via FT.

**Cost**: $0 Vertex. Some org work (deployment plumbing).

**Trigger conditions for 0**:
- D confirms no arm beats P8A on the v3 contract.
- AND any apparent improvement on individual buckets is within noise.

## Recommendation (provisional, pre-D)

**Default**: spend $60-100 on **Candidate I** (rebalance family_weights). It's the
cheapest direct test of the data-composition hypothesis identified in this session.
If Candidate I improves over P8A on the contract, we have a clear next packet.
If Candidate I plateaus at P18T's level, Move 4 becomes the right move.

**Skip if**: D shows P8A unambiguously beats both P18 arms across all contract buckets,
including bucket-3-specific FPR, with no operating point where P18T is preferable.
In that case, Candidate 0 (or jumping straight to Move 4) is the right move.

---

## Post-D update plan

When D returns, fill the following table from `promotion_contract.json`:

| Arm | dev_fake_macro_recall@selected_τ | selected_τ | lockbox_FPR | lockbox_fake_recall | dor_dev_FPR | webcam_lockbox_FPR | promotes? |
|---|---:|---:|---:|---:|---:|---:|:---:|
| P8A_REFERENCE_STEP5000 | | | | | | | |
| P18T_GRL_TREATMENT_STEP4000 | | | | | | | |
| P18C_NO_GRL_CONTROL_STEP4000 | | | | | | | |

Use these to discriminate between candidates I/II/III/0 above.
