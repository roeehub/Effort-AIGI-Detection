# Phase 1+2 complete — comprehensive status, recommended next packet, decisions still pending

**Date**: 2026-05-01 (post-P17 verdict, after Phase 1 diagnostic battery + Phase 2 readiness work)
**Branch**: `teams-relaunch-root-2026-04-17`
**Status**: All Phase 1 probes done. Phase 2 readiness work substantively done; Vertex verify of contract v3 in flight (~3h).

> Companion docs (read in order):
> 1. `P17_FINAL_VERDICT_2026-05-01.md` — what falsified the layer-3 readout idea.
> 2. `PHASE1A_SUBSTRATE_DIRECTION_FINDING_2026-05-01.md` — capture-mode is wrong axis; identity/method-cluster is right.
> 3. `PHASE1_SYNTHESIS_2026-05-01.md` — three-probe synthesis + Option I/II framing.
> 4. `PHASE2D_CODE_CHANGE_CHECKLIST_2026-05-01.md` — what code has to land before P18 launch.
> 5. This document — full picture, all decisions, what to do next.

---

## TL;DR

The journey-level updates from this session:

1. **Capture-mode GRL is the wrong axis.** P17's open question 1 probe shows the trained heads modally align with the `is_dor_shkedi` / `is_deeplive_enhanced` cluster (cos +0.10–+0.14 monotonic across training), and are essentially orthogonal to capture-mode (|cos|<0.08). This mechanistically explains why P15 GRL @ static λ=0.20 didn't bite (probe AUC 0.9994). Ramped λ on the same axis would not bite either.
2. **Eval substrate FPR underestimates production FPR by ~10–13 pp pooled, +40 pp on webcam.** Move 1.5 (190 frames) and the full-lockbox replication (839 frames) both confirm: re-cropping at production tightness (RFA=0.85) inflates FPR uniformly. `PC_Generator__s15` reals jump 20.69% → 89.66% (+69 pp). modern_v2 filter (drops webcam+screen) absorbs most of it down to +2.14 pp.
3. **The viso bucket gap is identity-confounded.** Move 1 grouped probe gives bucket-discrimination AUC 0.916 but identity-only control 0.987 — AMBIGUOUS verdict per PLAN.md §9 P1. P14_DATA_FIX-style data-side bucket lever has no probe support; the right data-side intervention is identity-axis (Move 4 paired same-identity), not bucket-axis.
4. **Contract policy v3 was already committed** (commit `974e033`, 2026-04-29; default flipped to 0.70). Image at 1.3.239 includes it. Launch-wrapper audit confirms no wrappers pass legacy flags. Vertex verify run on P8A_REFERENCE_step5000 submitted (job `5427016015462531072`, us-east1) — pending.
5. **Phase 3 packet (P18) is yaml-ready but blocked on ~half-day of code changes.** R13_P18_METHOD_DOMAIN_GRL.yaml + R13_P18_NO_GRL_CONTROL.yaml drafted. Treatment uses 2C's 12-bucket method-domain map that splits `deeplive` into basic/enhanced/teams (isolating the `is_deeplive_enhanced` axis Phase 1A pinpointed). Code changes required first: combined_paired.py method-aware lookup, effort_detector.py domain-map mirror, fix the test_unpaired_reals_and_grl.py import bug.

---

## What Phase 1 + Phase 2 produced

### Phase 1 — diagnostic battery (CPU only, ~2h total)

| Probe | Outcome | Strategic update |
|---|---|---|
| **1A: Substrate-classifier direction probe** | Trained heads modally align with `is_dor_shkedi` (cos +0.07→+0.14 monotonic); orthogonal to capture-mode (|cos|<0.08) | Capture-mode GRL targets the wrong axis. Ramped λ on capture-mode won't bite. |
| **1.5: Production-tight re-crop on 190 frames** | +10–14 pp FPR at every τ; +20 pp on webcam reals | Eval-FPR understates production-FPR. Re-cropping at inference makes things worse — needs retraining. |
| **2B: Full-lockbox v3 retag (839 frames)** | Replicates 1.5 at scale: +13.8 pp pooled FPR, +40 pp on webcam, modern_v2 absorbs to +2.14 pp; `PC_Generator__s15` reals 20.69%→89.66% | Production-tightness is structurally load-bearing; processing-signature shortcut surfaces on a 2nd identity (PC_Gen) beyond the canonical dor_shkedi case. |
| **1C: Move 1 grouped frozen-feature probe** | Bucket AUC 0.916 grouped; identity-only AUC 0.987 → AMBIGUOUS | Bucket gap is identity-confounded. P14_DATA_FIX-style bucket lever not justified. |

### Phase 2 — readiness work for next packet

| Sub-step | Outcome | Status |
|---|---|---|
| **2A: Contract policy v3** | Already committed (974e033, 2026-04-29). Default flipped to 0.70. Tests pass. Image at 1.3.239 includes it. Launch wrappers audited — none pass legacy flags. | ✅ done modulo Vertex verify (~3h pending) |
| **2B: Full-lockbox v3 retag** | Listed above under Phase 1. Outputs at `analysis/eval_substrate_v3_retag_2026-05-01/`. | ✅ done |
| **2C: Method-class label audit** | 12-bucket map at `analysis/method_class_audit_2026-05-01/proposed_method_domain_map.py`. 14/14 lookup tests pass. | ✅ done |
| **2D: Phase 3 yaml drafts** | `R13_P18_METHOD_DOMAIN_GRL.yaml` + `R13_P18_NO_GRL_CONTROL.yaml` drafted; code-change checklist at `PHASE2D_CODE_CHANGE_CHECKLIST_2026-05-01.md`. | ✅ done — yamls are DRAFTS pending code changes |
| **2E: Final documentation** | This document. | in progress |

---

## Reframed problem statement

The model has multiple shortcut signals (camera signature, face-pixel-area, processing pipeline, crop tightness, codec, identity-fake-method clustering) that are:
1. **Strongly present in training data** — the trained head amplifies the `is_dor_shkedi`/`is_deeplive_enhanced` cluster axis monotonically across 1000+ steps of training.
2. **Strongly correlated with fake/real on dev** — fresh LR on dev features achieves AUC 0.94; the trained head's deployed direction is *orthogonal* (cos +0.03 to +0.09) to that direction.
3. **NOT correlated with fake/real in deployment** — fresh LR on layer-3 features achieves 0.95 lockbox transfer, but the trained head collapses to 0.04–0.19 lockbox AUC by step 1000.
4. **Distributionally amplified by the eval substrate's loose crops** — production-tight crops UNHIDE the failure (Move 1.5 +10–13 pp FPR pooled; +40 pp webcam).

The pattern across the last five packets (P14 bundle, mclioexb, P14_DATA_FIX, P15 GRL @ static λ=0.20, P16 data-axis, P17 layer-3 readout) is consistent: each lever fires structurally, the trainer-side composite moves, the deployment scorecard says "no promote." Each Vertex packet costs $60–$70 to produce negative data. P17 was the cleanest exception — a $0 CPU probe gave a more dispositive answer than any scorecard would have. Phase 1's full battery follows that same pattern.

---

## Recommended Phase 3 packet — Option I-α (drafted, blocked on code)

**Treatment** (`R13_P18_METHOD_DOMAIN_GRL.yaml`):
- FT-from-P8A_step5000 (same init as P14 / P15)
- 12-class method-conditional GRL (using 2C's bucket map; splits `deeplive` into basic/enhanced/teams)
- λ=0.20 static (ramped λ deferred to v2)
- ANTI-SHORTCUT BUNDLE DISABLED (anchor_aware/face_scale_jitter/pipeline_random all OFF) per mclioexb single-axis discipline
- 4000 training steps
- Periodic saves at every 500 steps
- `external_real` family weight lowered 2.0 → 1.0 (Phase 2C label-confound mitigation)

**Control** (`R13_P18_NO_GRL_CONTROL.yaml`):
- Identical to treatment except `use_quality_domain_head: false`
- For clean attribution per Phase 1 synthesis user authorization

**Cost**: ~$120 / 1 day Vertex (treatment + control).

**Pre-launch blockers** (per `PHASE2D_CODE_CHANGE_CHECKLIST_2026-05-01.md`):
1. Update `data/sources/combined_paired.py` to use method-aware lookup (replaces 21+ call sites).
2. Update `detectors/effort_detector.py:280` `QualityDomainHead.DOMAIN_MAP` to mirror 12-bucket map.
3. Fix `tests/test_unpaired_reals_and_grl.py` import error (PLAN.md §10.7).
4. Pre-launch CPU smoke gates (GRL domain-population gate, frozen-feature method-LR baseline, 200-step Vertex smoke).

**Estimated engineering effort to launch-ready**: ~½ day to 1 day.

**Outcome lattice**:
- α (passes): post-training domain-confusion AUC drops materially on 12-class basis (0.95 → 0.65); contract scorecard recall lifts > 5 pp on `visomaster_enhanced_macro_dev` vs P8A. → ship P18.
- β (partial bite): domain-confusion AUC drops but contract under-promotes. → next packet adds ramped λ + production-tight retraining.
- γ (no bite): domain-confusion AUC stays ~0.95 (like SLOT2_GRL did). → 12-class GRL is ALSO wrong axis; pivot to identity-conditional GRL or Move 4 paired same-identity contrastive (Option II).

---

## Vertex job in flight

**Job ID**: `5427016015462531072` (us-east1).
**Purpose**: Verify scorecard on P8A_REFERENCE_step5000 with v3 contract policy ENGAGED (the launcher omits the flag, relying on the 0.70 default flipped in commit 974e033). Closes PLAN.md §8.2 (`contract-policy-bug-fix-not-committed`) loop's "scorecard verifies new policy in image" criterion.
**Submitted**: 2026-05-01 19:07 CEST.
**Initial state**: `JOB_STATE_PENDING`.
**ETA**: ~3 hours from submission.
**Monitor**: `by2cnqbc7` (state-transition poll, 60s interval).

Output paths (when ready):
```
gs://training-job-outputs/test_results/teams_promotion_contract/p8a_v3_verify_2026-05-01/
  reports/                          # validation reports per suite
  diagnostic_scorecard/scorecard.{csv,wide.csv,int8_delta.csv,json}
  promotion_contract/
    threshold_grid.csv              # full τ grid Pareto curve
    selected_threshold_scorecard.csv
    checkpoint_summary.csv
    promotion_contract.json
    promotion_winner.json
```

**What success looks like**: P8A_step5000 selected τ ≈ 0.916 (recall-floor path), with viso recall ~13.6%, deeplive ~23.9%, teams_fake ~52.6%, lockbox FPR ≤ 5%. Per `contract_policy_bug.md:25-66`. If the result matches local re-score (`/tmp/p13_repolicy/recall_floor_30/` reference), the v3 fix in image is verified. If results diverge from local re-score, investigate before launching subsequent P-series scorecards.

---

## Decisions still pending user authorization

1. **Authorize the code-change work** for P18 launch — per PHASE2D_CODE_CHANGE_CHECKLIST: combined_paired.py method-aware lookup, effort_detector.py mirror, test import fix. ~½ day engineering. Recommendation: yes, this is the necessary engineering to test Phase 1A's hypothesis at scale.

2. **Authorize the P18 launch itself** (treatment + control, ~$120 / 1 day Vertex) AFTER code lands and pre-launch smoke gates pass. Recommendation: yes, conditional on smoke gates passing.

3. **Disposition on processing-signature shortcut on PC_Generator__s15** (new finding from 2B). Same shortcut pattern as canonical dor_shkedi case but on a different identity. Should we (a) add a per-identity FPR audit to every future scorecard, (b) launch a contact-sheet investigation similar to the slot-07 sanity, OR (c) treat it as another data point that motivates Move 4 (paired same-identity training)? Recommendation: (c) — Move 4 by construction prevents identity-as-shortcut; the contact-sheet would be informative but not actionable beyond what Move 4 already addresses.

4. **Promotion-grade scorecard re-run on existing leaders** with v3 substrate (after Phase 2B retag). The 2B retag produces an honest production-FPR-translation surface; re-running scorecards on P8A and `mclioexb` against the v3 substrate would tell us "what is honest deployment-FPR for our existing leaders?" Recommendation: yes for P8A only (~$10), to update the baseline floor we judge P18 against. mclioexb non-promotion is already locked in; not worth re-scoring.

---

## What I would NOT do (per Phase 1 finding)

- **Re-launch P15 with ramped λ on capture-mode quality-domain head** — Phase 1A directly refutes that this axis is the encoder's shortcut. Wastes ~$60–120 to confirm what we already know.
- **Re-attempt P14_DATA_FIX with conjunction source** — Move 1 says bucket gap is identity-confounded. Same data-axis lever has now failed twice (DATA_FIX collapse 2026-04-30, P16 non-promotion 2026-04-30) and the new probe says it's not the right axis.
- **Sweep more head/layer variants** — P17 is dispositive. Head architecture isn't the destructor.
- **Stack levers in another bundle without single-axis ablation** — mclioexb taught us bundle < strongest single component (5.7× gap).
- **Launch any Vertex packet against the existing eval substrate** as the deployment proxy without acknowledging the +10–13 pp production-FPR caveat from 2B.

---

## Files written / committed this session

```
A  analysis/intermediate_layer_probe_2026-04-30/substrate_classifier_direction_2026-05-01.py
A  analysis/intermediate_layer_probe_2026-04-30/outputs/substrate_classifier_direction_2026-05-01.{json,csv}
A  analysis/move1_frozen_probe_2026-05-01/finish_analysis.py
A  analysis/move1_frozen_probe_2026-05-01/outputs/probe_results.{json,csv}
A  analysis/move1_5_production_recrop_2026-05-01/run_probe.py
A  analysis/move1_5_production_recrop_2026-05-01/outputs/{recrop_per_frame.csv,recrop_summary.json,score_shift_distribution.png}
A  analysis/move1_5_production_recrop_2026-05-01/REPORT.md
A  analysis/eval_substrate_v3_retag_2026-05-01/run_v3_retag.py
A  analysis/eval_substrate_v3_retag_2026-05-01/outputs/{eval_substrate_v3_retag.parquet,eval_substrate_v3_retag.csv,aggregate_metrics_v3_retag.json}
A  analysis/method_class_audit_2026-05-01/proposed_method_domain_map.py
A  analysis/method_class_audit_2026-05-01/method_inventory.csv
A  experiments/phase2_round13/R13_P18_METHOD_DOMAIN_GRL.yaml
A  experiments/phase2_round13/R13_P18_NO_GRL_CONTROL.yaml
A  docs/relaunch_handoffs/PHASE1A_SUBSTRATE_DIRECTION_FINDING_2026-05-01.md
A  docs/relaunch_handoffs/PHASE1_SYNTHESIS_2026-05-01.md
A  docs/relaunch_handoffs/PHASE2D_CODE_CHANGE_CHECKLIST_2026-05-01.md
A  docs/relaunch_handoffs/PHASE1_2_COMPLETE_STATUS_2026-05-01.md   (this document)
```

All uncommitted per project pattern. Contract v3 itself was committed earlier (commit `974e033`, 2026-04-29).

---

## Updated journey re-frame

The cumulative diagnosis is unusually clean now:

1. **Architectural layer (P7→P8A)**: solved. P8A is the strongest base model.
2. **Augmentation layer (P13–P16)**: exhausted. mclioexb's bundle-decomposition lesson + P16's data-axis non-promotion + Phase 1's diagnostic verdict together establish that augmentation/bundle stacking is not the binding constraint.
3. **Head/layer-readout layer (P17)**: structurally dead. The head can't be retrained out of using the dominant gradient.
4. **Diagnostic + measurement layer (Phase 1)**: now in hand. We can falsify Phase 3 levers on CPU before spending $60–$120 each. P17's $0 verdict + Phase 1's $0 substrate-classifier probe + Move 1.5's $0 (8 min) production-tight retag set the methodology.
5. **The destructor**: the encoder + training data + standard cross-entropy loss conspire to learn an identity-fake-method-cluster shortcut. Standard ML doctrine for this is invariance penalty (DANN/GRL on the right axis) or substrate-balanced data (paired same-identity).

The Phase 3 packet is the first attempt at the "GRL on the right axis" intervention, with single-axis isolation respect for the bundle-decomposition lesson, paired with a no-GRL control for clean attribution. If P18 doesn't bite, Move 4 paired same-identity contrastive (Option II in the synthesis) is the next architectural address.

The path to a "strong, robust, no-shortcut model" is now concrete:
- (1) P18 method-GRL @ 12-class — testing whether finer-grained GRL bites where coarse 3-class didn't (~½ day engineering + $120 Vertex).
- (2) If P18 doesn't bite → Move 4 paired same-identity contrastive (~2–3 days new code + $60 Vertex).
- (3) Eval substrate retag at production tightness — already done (Phase 2B); future scorecards should read against the v3 substrate.
- (4) Production-tight training crops — needs new bbox-aware training-time crop transform; deferred to v2 packet.
