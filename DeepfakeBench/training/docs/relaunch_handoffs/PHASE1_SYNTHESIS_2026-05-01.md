# Phase 1 synthesis — what the diagnostics tell us, and the Phase 3 packet recommendation

**Date**: 2026-05-01 (post-P17 verdict, after C-track diagnostic battery)
**Author**: working agent, after sub-agent dispatches
**Status**: All Phase 1 probes complete. Strategic recommendation pending user decision.

> Companion docs (read alongside this one):
> - `P17_FINAL_VERDICT_2026-05-01.md` — what falsified the layer-3 readout idea.
> - `PHASE1A_SUBSTRATE_DIRECTION_FINDING_2026-05-01.md` — the 1A probe in detail.
> - `analysis/move1_frozen_probe_2026-05-01/outputs/probe_results.json` — Move 1 numbers.
> - `analysis/move1_5_production_recrop_2026-05-01/outputs/recrop_summary.json` — Move 1.5 numbers + REPORT.md.

---

## TL;DR

The three Phase 1 diagnostics paint a coherent picture that is **different from the working assumption going in**:

1. **Phase 1A**: trained P17 heads do NOT use capture-mode features (cosines vs `is_webcam`/`is_phone_screen`/`is_normal_photo` all |c|<0.08). They modally use the **identity-fake-method cluster** (`is_dor_shkedi` + `is_deeplive_enhanced`, 86% co-aligned, head-cosine +0.10–+0.14 monotonic across training).
2. **Move 1.5**: re-cropping eval substrate at **production tightness (RFA=0.85) INCREASES FPR by ~10 pp** at every τ tested, +20 pp on the FPR-dominant webcam slice. Webcam-real median prob_fake jumps **22×** (0.025 → 0.55). Production translates to *higher* FPR than eval, not lower. Eval-substrate FPR underestimates production FPR.
3. **Move 1**: training-bucket viso vs eval-bucket viso are discriminable at AUC 0.916 with grouped split, but **identity-only control AUC = 0.987** — bucket distinction is identity-confounded. Cross-bucket fake/real transfer AUC = 0.929 (vs train-bucket within-fake/real OOF AUC 0.968 and eval-bucket within OOF AUC 0.886). Per PLAN.md §9 P1 outcome ladder this is **AMBIGUOUS** — does not justify P14_DATA_FIX-style bucket lever.

**Three structural updates to the strategic landscape:**

A. The previously-proposed "ramped GRL on capture-mode quality-domain head" (`use_quality_domain_head: true`, the family-level 4-domain mapping in `combined_paired.py:66`) is targeting **the wrong axis**. The trained head wasn't using capture-mode at static λ; ramping λ on the same axis won't introduce alignment that wasn't there. A more dispositive linear-probe AUC = 0.999 across P8A and SLOT2_GRL (`analysis/domain_confusion_probe_2026-04-30/outputs/probe_p8a_slot2_slot3/summary.json`) corroborates: GRL on this axis didn't bite, regardless of λ.

B. Production-tight training crops are now load-bearing. Eval substrate's looser crop tightness has been giving us ~10 pp FPR + 20 pp on webcam underestimates. The fix is not re-cropping at inference (which makes things worse for the same model), but **retraining at production tightness** (deterministic RFA=0.85, NOT the existing `face_scale_jitter@0.50` augmentation which is jitter-around-some-default).

C. P14_DATA_FIX-style bucket lever (already failed once at run `xan4dfto`, value_composite=0.126) does NOT regain support from Move 1. The bucket gap is largely identity-overlap-deficit, not a meaningful distribution shift in P8A's feature space. The right data-side intervention is identity-axis (Move 4 paired same-identity), not bucket-axis.

---

## Detailed evidence

### Phase 1A — what direction did the trained head pick?

The probe trained logistic regression on cached P8A layer-3 [CLS] features (`intermediate__P8A__layer03__n800.npz`) for multiple substrate axes, then computed cosine similarity in normalized feature space between (substrate-classifier direction, trained-head decision direction). For 8 P17 trajectory ckpts (4 ArcFace, 4 LINEAR), the most-aligned axis was uniformly `is_dor_shkedi`, with monotonic alignment growth across training:

| Step | ArcFace cos(head_dir, dor_shkedi) | LINEAR cos(head_dir, dor_shkedi) |
|---|---:|---:|
| ep1 (random) | -0.010 | -0.009 |
| ~step 1000 | +0.092 | +0.070 |
| ~step 1200 | +0.120 | +0.089 |
| ~step 1700-2100 | +0.142 | +0.102 |

**Capture-mode axes:** uniformly |c|<0.08, no monotonic trend. **Face-size axis:** uniformly |c|<0.022.

`cos(is_dor_shkedi, is_deeplive_enhanced) = +0.86` — these form one tight identity-fake-method cluster in feature space. The trained head is amplifying this cluster.

**Mechanistic explanation for why P15 GRL@λ=0.20 didn't bite:** the encoder didn't have a "capture-mode classifier" component for the GRL adversarial gradient to reverse. The shortcut runs along an axis the GRL head wasn't classifying.

---

### Move 1.5 — does production-tight cropping make FPR better or worse?

`recrop_dataset.py:39` defines production tightness as `TARGET_RFA = 0.85` (face area / image area). 190 frames stratified by capture mode, scored on P8A both with original eval crop ("asis") and re-cropped at RFA=0.85 ("prod").

| τ | Pooled FPR (asis → prod) | Δ FPR | Pooled Recall (asis → prod) | Δ Recall |
|---|---|---:|---|---:|
| 0.50 | 16.24% → 30.77% | **+14.53 pp** | 78.08% → 87.67% | +9.59 pp |
| 0.92 | 7.69% → 17.95% | **+10.26 pp** | 57.53% → 67.12% | +9.59 pp |
| 0.9741 | 2.56% → 12.82% | **+10.26 pp** | 47.95% → 58.90% | +10.96 pp |

Webcam-mode reals (n=35, FPR-dominant slice per `project_lockbox_fpr_dominated_by_webcam_mode.md`):
- τ=0.92: FPR 20% → 40% (+20 pp)
- τ=0.9741: FPR 8.6% → 28.6% (+20 pp)
- Webcam-real median prob_fake: 0.025 → 0.553 (×22 inflation)

Real-frame mean prob_fake doubles overall (0.172 → 0.344). Fake-frame mean is already saturated (0.766 → 0.832); recall gain is at the τ=0.97 tail.

**Key implication:** every reported FPR number on the eval substrate is a **lower bound** on production FPR. Modern_v2's 0.71% lockbox FPR (`project_lockbox_fpr_dominated_by_webcam_mode.md`) is on eval-tightness crops; production crops would push it higher. This compounds with the mclioexb non-promotion: deployment is even harder than the scorecard already showed.

Native eval `face_area_ratio` median = 0.38, vs production target 0.85. **Eval faces cover ~45% of what production crops will cover.** 24% of probe frames had RFA=0.85 crop clamped at the image edge (face was already near boundary).

---

### Move 1 — is the bucket gap real and not identity-confounded?

| Metric | Value |
|---|---:|
| Bucket discrimination AUC (train viso vs eval viso, GroupShuffleSplit by (identity, session)) | 0.916 ± 0.019 |
| **Identity-only control AUC (multi-class OVR, min_per_class=2)** | **0.987** |
| Within train-bucket fake/real OOF AUC (GroupKFold by sample_id) | 0.968 |
| Within eval-bucket fake/real OOF AUC (GroupKFold by identity) | 0.886 |
| Cross-bucket transfer AUC (train-bucket fake/real LR → eval-bucket) | 0.929 |

The grouped bucket discrimination is high (0.916), but the identity-only control is even higher (0.987) — almost all of what the bucket-LR is fitting is identity, not bucket-distribution shift. **Per PLAN.md §9 Priority 1 outcome ladder: AMBIGUOUS** ("identity-only control ≥ 0.70 → probe results are confounded; do NOT use as P14_DATA_FIX gate"). Identity-only AUC ≥ 0.90 specifically calls for "stricter group exclusion or larger holdout."

The within-bucket fake/real OOF AUCs are healthier than the trained-model frame-level AUC (0.886 eval-bucket OOF vs ~0.75 from `project_p8a_frame_level_auc_2026-04-29.md`). **The frozen P8A features carry more usable fake/real signal than P8A's deployed head reads out** — consistent with Phase 1A's finding that the head is using a sub-optimal direction.

Cross-bucket transfer AUC 0.929 is *higher* than within-eval-bucket OOF AUC 0.886 (small N effect or composition difference), and lower than within-train-bucket 0.968 — so there *is* a 0.04–0.08 cross-bucket gap, but it's not catastrophic and it's largely identity-driven.

---

## What this collectively means for the next packet

**Reframed problem statement (post-Phase-1):**
> The trained head learns identity-fake-method cluster discrimination on training-substrate-loose-crops because that's the easiest gradient. Eval substrate (looser crops, mostly teams_real) HIDES the model's failure mode by giving it lots of "easy" real frames. Production substrate (tight crops, more diverse identities, more webcam) UNHIDES the failure: the same identity-cluster shortcut blows up FPR by 10–20 pp, especially on webcam-mode reals.

The fix has to attack THREE things at once because each alone has been refuted:
1. **The identity-cluster shortcut** (Phase 1A; refutes capture-mode-GRL singularly).
2. **The crop-tightness gap** (Move 1.5; refutes "re-crop at inference" + "current eval substrate is honest enough").
3. **The training-data identity overlap with eval** (Move 1; refutes naive bucket-axis swaps).

---

## Recommended Phase 3 packet design

Two viable architectures, in order of confidence:

### Option I (recommended) — production-tight + method-conditional GRL FT-from-P8A

**Levers (single-axis stacked, each with isolation ablation per the bundle-decomposition discipline):**

1. **Deterministic production-tight crops at RFA=0.85** (NOT face_scale_jitter; replaces the augmentation with a fixed crop spec at training time). Closes the eval→production crop-tightness gap by training under deployment conditions.
2. **Method-conditional GRL** with finer-grained method labels (e.g., 8–12 method classes: `df40`, `deeplive_basic`, `deeplive_enhanced`, `deeplive_teams`, `visomaster_simswap`, `visomaster_inswapper`, `visomaster_ghost`, `teams_capture`, `external/VCD`, ...). Repurposes existing `use_quality_domain_head` machinery in `detectors/effort_detector.py:236-274` with a wider QUALITY_DOMAIN_MAP (or a new `method_domain_head` field). Targets the dor_shkedi/deeplive_enhanced cluster directly.
3. **Ramped λ** (`set_lambda(t)` ramp 0 → λ_target over warmup; existing in `GradientReversalLayer` per P15 readiness note, never wired to trainer caller).
4. **Periodic saves at every 250 steps** + **in-trainer kill gate**: probe each save's lockbox-substrate-direction AUC (≈P17's recipe) and abort if AUC drops below 0.5 by step 1000. Codifies the P17 trajectory-probe methodology into trainer-side observability.

**Cost estimate:** ~$60–$100 / 8h on us-west4 A100. Plus a no-GRL-but-tight-crop control (~$60) to disambiguate "tight crops alone" vs "tight crops + method GRL" contributions.

**Pre-launch CPU gates (each ~$0):**
- Method-domain count audit: per-(method_class, label) sample count ≥ minimum threshold; no dominant-source confound (PLAN.md §7.4 round-1 escalation).
- Smoke-load 200 steps of training-bucket viso + frozen-feature method-LR AUC ≥ 0.85 (confirms method labels are real signal).
- Domain-confusion probe analog of `domain_probe.py` but with method classes — to know what we're trying to flatten.

**Why this over alternatives:**
- The P17 verdict said "fix must be UPSTREAM of the head." This is upstream (data-tightness) + adversarial-encoder (method GRL). Phase 1A pinpointed the right axis (method/identity-cluster, not capture-mode).
- Cost is moderate; existing infrastructure (effort_detector + GRL machinery + face_scale_jitter knobs) is mostly reusable.
- Single-lever ablation respect (mclioexb lesson): each contributing knob can be turned off in a sister yaml.

### Option II — Move 4 paired same-identity contrastive

**Levers:**
1. Production-tight crops (same as Option I).
2. Pair-aware sampler that yields anchor + positive (same identity, opposite label) + negative (different identity, same label) per training step. Uses proper-data wave's 705 paired identities + companion_bucket plumbing in `data/sources/visomaster.py:1044`.
3. Triplet contrastive loss (existing `loss/contrastive_regularization.py:39` `ContrastiveLoss` is for UCF/disentanglement; needs an effort_detector-compatible variant).

**Cost estimate:** ~$60–$100 Vertex (one ~8h FT-from-P8A run) PLUS ~2–3 days of new-code: paired sampler integration into `combined_paired.py`'s `combined_paired` strategy, contrastive-loss head exposure on the effort detector, trainer forward-pass changes, unit tests.

**Why second-pick:**
- Higher engineering risk. Significant new code that needs end-to-end testing.
- Theoretically the most-direct attack on the dor_shkedi cluster shortcut (paired same-identity examples mean identity is no longer a viable shortcut by construction).
- Only worth pursuing if Option I doesn't move the needle.

### What I would NOT do

- **Re-launch P15 with ramped λ on capture-mode quality-domain head.** Phase 1A directly refutes that this axis is what the encoder is using. The 5-line yaml change has the wrong domain mapping.
- **Re-attempt P14_DATA_FIX with conjunction source.** Move 1 is AMBIGUOUS; the bucket gap is identity-driven, not bucket-driven. Won't escape the τ-tail Pareto failure that mclioexb hit.
- **Sweep more head/layer variants.** P17 closed that chapter; the head can't be retrained out of using the dominant gradient.
- **Treat current eval substrate as honest production proxy without explicit retag.** Move 1.5 refutes that; every FPR number carries a +10pp production caveat now.

---

## Pre-Phase-3 work that has to happen first

These are *also* CPU-only / mechanical and load-bearing for whichever Phase 3 option lands:

1. **Commit the contract-policy v3 fix + safety patch + launch-wrapper audit** (PLAN.md §8.2, §5.4 expanded). Three "fix attempts" in 6 days, none committed. The mclioexb scorecard didn't even exercise v3 because the launcher omitted `--promotion_target_fake_recall_min 0.30`. Until v3 is committed AND the runner default flips to 0.30 (or fail-fast assertion is in place), every future scorecard read is unreliable. Estimated effort: half-day mechanical + ~$10 / 3h Vertex to verify on P8A baseline.

2. **Method-class label audit** in `combined_paired.py` — confirm sample.method values, count distribution, decide on the 8–12 method buckets for Option I's GRL. ~30 min CPU.

3. **Eval-substrate retag at production tightness** — extend Move 1.5's recrop probe to all of lockbox, not just 190 frames; produce a v3-substrate eval CSV that future scorecards can read against. ~1–2h CPU. Critical for honest reporting.

---

## Decision request for user

1. **Option I (production-tight + method-GRL) or Option II (Move 4 paired contrastive) as the Phase 3 packet?** Recommendation: I, with II reserved for round-2 if I doesn't move the needle.
2. **Authorize the contract-policy v3 commit + safety patch + image rebuild + P8A scorecard re-run with `--promotion_target_fake_recall_min 0.30` engaged?** Mechanical, ~$10. Recommendation: yes, blocking precondition for any Phase 3 launch.
3. **Authorize the eval-substrate retag at production tightness?** ~1–2h CPU; closes the eval-FPR honesty gap. Recommendation: yes.
4. **Re-evaluate "no-GRL control" at the same scope** (PLAN.md §10.7 escalation): we should pair every Phase 3 packet with a control yaml that has identical data + tight crops but no GRL, to disambiguate which lever moved what. Recommendation: yes — same cost (~$60).
