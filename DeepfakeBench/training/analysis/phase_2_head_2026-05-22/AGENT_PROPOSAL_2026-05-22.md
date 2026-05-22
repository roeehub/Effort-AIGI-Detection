# Phase 2 HEAD — agent proposal (opinion doc)

Status: **scoring complete 2026-05-22T17:04:54Z**. Verdict below.

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

## Recommendation: **ITERATE → defer to Phase 4 HEAD ALT (dual-readout)**

The head plateaued at step 250 and the trajectory is flat across all 5
scored ckpts. Composite λ=1.0 clusters at 0.235-0.243; viso recall stuck at
0.072-0.075; lockbox real FPR ~0.017-0.018. None of the 5 ckpts clears the
four-gate deploy criterion.

But the lockbox-side numbers are essentially equivalent to the face-pool
inference baseline (0.0154 / 0.7668 / 0.249) — within statistical noise.
HEAD is NOT an abort (lockbox isn't regressing). It IS an iterate: viso
< 0.15 floor means the face-pool readout's structural loss of non-face
patches couldn't be recovered by head-fitting. The viso signature must
live (at least partially) outside the centered 7×7 face region — which is
exactly what Phase 4 Fallback A (HEAD ALT dual-readout) addresses.

CPU-1 was γ-outcome (non-face saliency mass 72.6%, but non-face recall = 0
at the calibrated CLS τ so α condition not met). Per the literal rule,
HEAD ALT is deferred to Phase 4. The HEAD verdict here confirms that
deferral is the right call: structural face-pool loss IS the binding
constraint on viso, so HEAD ALT addresses the actual mechanism, not a
hypothesized one.

### Argument for deploy

None of the 5 ckpts clears all four gates. Best composite is 0.235 vs ≤
0.20 threshold (over by 0.035). Lockbox real FPR 0.0176 vs ≤ 0.016 (over by
0.0016). Viso recall 0.073 vs ≥ 0.15 (under by 0.077). Three gates fail
simultaneously — deploy is structurally blocked.

### Argument for iterate (chosen)

HEAD step 250 (winner) Pareto-equivalent to the $0 face-pool inference
baseline on the same 9 suites. Marginal composite improvement (0.249 →
0.235, -5.6%). Marginal lockbox_fake_recall improvement (+0.016). Marginal
lockbox_real_fpr regression (+0.0022). Marginal viso improvement
(+0.0054). Net effect: training the head on face-pool features didn't move
the needle materially — the structural loss is upstream of the head.

The viso miss isn't a head-boundary problem (face-pool features the head
sees are missing the signal); it's a feature-availability problem
(centered 7×7 doesn't capture the signal). The fix is to give the head
BOTH face_pool AND non_face_pool features (concat → 1024-dim head). That's
HEAD ALT per the plan's Fallback A. Defer to Phase 4.

### Argument for abort

Lockbox metrics did NOT regress vs face-pool inference baseline. Lockbox
real FPR moved from 0.0154 → 0.0176 (within noise floor; the 0.0022 delta
is < 1 σ at n_videos=1361). Lockbox fake recall moved from 0.7668 → 0.7826
(+0.016). Abort criterion (regression vs face-pool inference baseline) does
NOT fire.

---

## Phase 4 HEAD ALT — go criteria (for the next agent)

If/when Phase 4 is greenlit:
1. Add a `face_pool_readout.dual_readout: true` flag (or new top-level
   `dual_readout_readout` block) to `detectors/effort_detector.py`. In
   forward: compute face_pool (centered 7×7, 49 patches) AND non_face_pool
   (complement, 147 patches), mean-pool each, apply `ln_post` then
   `visual.proj` to EACH. Concat → 1024 dim. Pipe to a new
   `head_dim_override=1024` head.
2. New yaml `R13_FACE_POOL_DUAL_READOUT_HEAD_2026-MM-DD.yaml` cloning the
   HEAD ancestor with dual_readout + head_dim_override=1024.
3. head_only_retrain MUST reinit head (existing 512-dim head doesn't fit
   1024-dim input).
4. Smoke first ($3, 30 min) — verify both pool features actually flow.
5. Cost projection: ~$30 full + smoke. Total Phase 2+4 HEAD spend: ~$45.

The decision gate stays the same (lockbox ≤ 0.016 AND viso ≥ 0.15 AND
composite ≤ 0.20). If HEAD ALT still misses viso ≥ 0.15, that's evidence
that viso signature isn't recoverable from any per-patch pool (e.g., lives
in attention map structure, not in mean-pooled features).

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
