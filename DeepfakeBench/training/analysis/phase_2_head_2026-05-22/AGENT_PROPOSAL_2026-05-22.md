# Phase 2 HEAD — agent proposal (opinion doc)

Status: **PENDING full-run completion**. Decision gate per
`/Users/roeedar/.claude/plans/ok-so-we-don-t-valiant-quasar.md` Phase 2 HEAD
section. This doc will be populated once the full Vertex job reaches
`JOB_STATE_SUCCEEDED` and the per-ckpt scorecard runs.

This document IS opinion-bearing — it makes a recommendation. FACTS live in
`RESULTS_FACTS_2026-05-22.md`.

---

## Decision matrix (from plan)

| Outcome | Criterion (all four MUST hold for a single periodic ckpt) |
|---|---|
| Deploy candidate | lockbox_real_fpr ≤ 0.016 AND lockbox_fake_recall ≥ 0.74 AND visomaster_enhanced_macro_dev recall ≥ 0.15 AND composite (λ=1.0) ≤ 0.20 |
| Iterate (HEAD ALT later) | lockbox kept but viso < 0.15 |
| Abort | lockbox regresses vs face-pool inference baseline (0.0154 / 0.7668) |

---

## Recommendation (to be filled post-scoring)

_TBD._

### Argument for deploy

_TBD — depends on whether any periodic ckpt clears all four gates._

### Argument for iterate

_TBD — depends on whether CPU-1 outcome (visomaster non-face-localization at
gamma per `analysis/viso_fake_signature_localization_2026-05-23/`) justifies
deferring to Phase 4 HEAD ALT instead of immediate iteration._

### Argument for abort

_TBD — depends on whether the head-fitting on face-pool features regresses
lockbox vs the face-pool inference baseline (0.0154 / 0.7668)._

---

## Risk caveats (independent of outcome)

1. **head_only_retrain freezes the encoder entirely.** With `freeze_base=true`
   on the SVD residuals + `head_only_retrain.enabled=true`, the encoder
   that produced Slot A v2 step3500's substrate-invariance is preserved
   exactly. This was the load-bearing P8A property
   (`project_job7_head_retrain_REFUTED_2026-05-04.md`). The
   risk profile here is OPPOSITE to Job 7 because the FEATURES the head
   sees are different (face_pool, not CLS) and substrate-invariant by
   construction at L11 per Probe 2.

2. **anchor_aware penalty still computes with face_pool features.** With
   the encoder frozen, the anchor_aware penalty term only contributes via
   the head's gradient. The penalty still reads dor false-flag pool frames
   through the face_pool wrapper. Expected behavior: head learns to put
   prob_fake on dor pool near target=0.10. If the face-pool features for
   dor are NOT separable from non-chronic reals, the anchor term will
   pressure the head into a flatter decision boundary — could help or hurt
   depending on whether dor's face features carry enough fake signature.

3. **multi_axis_grl has no effect with frozen encoder.** GRL only acts via
   encoder gradient reversal; frozen encoder => GRL classifier trains
   but its gradient never reaches the encoder. Kept enabled for trainer
   stat-logging parity with Slot A v2 ancestor, not for any learning
   contribution.

4. **face_pool inference baseline already Pareto-dominates Slot A v2 CLS
   on three of four gates** (lockbox_real_fpr, lockbox_fake_recall,
   composite). The only failing gate is `visomaster_enhanced_macro_dev`
   recall (0.0673 vs 0.15 floor). The hypothesis is that head-fitting on
   face_pool features can rescue viso recall while keeping the lockbox
   gains. If it does NOT, then the viso signature is structurally outside
   the centered 7x7 face region (CPU-1 outcome was gamma; non-face mass
   72.6% suggestive but not decisive at α). Phase 4 HEAD ALT
   (dual-readout face + non-face) would then be the right next step.

5. **Smoke yaml bug discovered late** (`max_train_steps` not propagated)
   forced cancellation of smoke at 57 min ($2.94 cost) instead of the
   planned 30 min. Lesson: every new training yaml MUST include
   `max_train_steps: N` matching `total_training_steps`. The smoke
   ckpts (step 100, 200) are preserved on GCS.

---

## Cost ledger (as of full-run submission)

| Item | Wall | Cost |
|---|---:|---:|
| Cloud Build 1.3.297 + 1.3.298 | ~15 min | ~$0 (free tier) |
| Smoke (cancelled at 57 min) | 57 min | ~$2.94 |
| Full (RUNNING) | _pending — projected ~2.5-3 h_ | _projected ~$7.30-9.00_ |
| Scoring on local MPS | _pending — projected ~2-2.5 h_ | $0 |
| **Total projected (Phase 2 HEAD)** | _projected ~5-6 h_ | _projected ~$10-12_ |

Plan budget for HEAD: $25-32. Actual projected ~$10-12 — under budget by
50-60%. Smoke overrun ($2.94 vs $3 plan) was offset by smoke A100 being
faster per dollar than the original A10G plan estimate, and by no
us-east1 PENDING delay on the full launch.
