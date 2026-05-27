# 4-Day Recall-First Sprint — Plan v3

**Author:** Claude Opus 4.7 (1M context)
**Date:** 2026-04-27
**Working dir:** `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/`
**Branch:** `teams-relaunch-root-2026-04-17`
**Replaces:** `april-26-training-master-plan-v2` (recipe-tuning hypothesis space)
**Companion logs:** `april-26-training-master-plan-v2.LOG.md`, `april-26-training-master-plan-v2.RESULTS.md` (continue appending — same project, same artifact thread)

---

## 1. Context — Why This Plan Succeeds Plan v2

Plan v2 closed three A.2-style validations and a probe battery. Outcomes:

- **Recipe-tuning is exhausted.** P8A reference dominates lockbox; no recipe variant we tried (P10_SYM, codec_hedge, RLP6_04 fork-back) clears the plan-v2 §4 Phase D gates.
- **P8A is partially shortcut-leaky.** Source-bucket probe at 4.6× chance (vs §6 ≤2.5× PASS gate). Codec aug pulls out ~9% of the leakage; not enough to clear gate alone.
- **Calibrated P8A under corrected 7%/10% policy:** lockbox 1.84% FPR / 38.7% recall. At τ=0.5: 6.17% FPR / 65.2% recall. **Gap to user's 90/5 target:** 25–55 pp on recall depending on operating point.
- **Per-method dev recall is the binding metric**, not aggregate lockbox: P8A τ=0.5 hits visomaster_enhanced_macro **35.6%**, deeplive_enhanced **53.0%**, teams_fake_all_dev **70.2%**. User's bar is **90% on each of these three families**.

**Two new levers v2 didn't exploit:**

1. **Modern-target-domain lockbox subset.** A parallel agent began frame-level tagging of the lockbox (`analysis/lockbox_tagging/`, created 2026-04-27 ~15:30). The lockbox includes captures from old setups not representative of current Teams deployment. A subset gated on modern conditions is likely to have materially lower P8A FPR than the raw 6.2%.

2. **Data-axis levers untouched in Plan v2.** Three exist and are ready to wire: (a) `loss/contrastive_regularization.py` is registered but inactive; (b) `data/augmentations/teams_simulation.py` supports heavier `teams_codec_sim_p`; (c) `R13_P9_05_real_codec_uplift.yaml` carries the real-side codec uplift flag. None of those have been combined in a single FT recipe with the in_proj-SVD bug now fixed.

**User bar this sprint (locked):** ≥90% recall on visomaster_enhanced_macro + deeplive_enhanced + teams_fake_all_dev simultaneously, ≤5% FPR on modern Teams capture conditions, validated against Dor sanity. Extension allowed if needed.

---

## 1.7 Identity-Corruption Audit Resolved — 2026-04-27 17:30 CEST

Hard data has resolved §1.6's open questions. Findings (full numbers in `april-26-training-master-plan-v2.RESULTS.md` 17:30 entry):

1. **Lockbox 5-identity slice is 96.5% identity-clean.** ArcFace clustered 4 of 5 labels as 1 person; the only multi-cluster label is PC_Generator__s15 where the split is real-vs-fake source mismatch (91 fakes of person A, 29 reals of person B sharing one identity_key).
2. **dor_shkedi is one ArcFace identity, 275 frames.** The agent's 0% accuracy finding is **genuine model failure**, not label corruption. Track A.5d down-weighted accordingly.
3. **Dev split has session-aggregation corruption.** PC_Generator__s13 sample (n=30) contains ≥3 distinct people, all labeled real (intra-cluster sim 0.86 vs inter-cluster sim 0.19). Manifest confirms 141 distinct video segments under that label.
4. **Training-set identities are video-derived, not session-derived.** Read of `data/sources/{combined_paired,df40_paired,visomaster}.py` confirms: DF40 uses `df40_{target_identity}`, DeepLive uses `realpool_{original_video_name}`, Visomaster uses videoID portion of sample_id. Identity-stratified split has explicit overlap checks. Session-aggregation corruption does NOT propagate to training.

**Plan-v3 amendments propagating from this audit:**
- §1.6 P11-launch hold-conditions are FALSIFIED. P11_TARGETED launch is GREEN once authorized — training-pipeline assumptions hold.
- §1.5 absolute claims need re-reading through the lens of model-failure-not-label-corruption (especially the "dor_shkedi 0% accuracy" being genuine model weakness, not data hygiene).
- §7 Validation gains a Day-2 sub-track: explode dev session labels into ArcFace-derived clusters before computing per-method FPR; PC_Generator__s13 counts as 3 evaluation units, not 1.
- Modern subset definition (Track A.5b/c) stays useful for capture-condition filtering but loses the within-identity outlier headroom §1.5 anticipated.

**Optimism update:** the §2 ranges hold. The FPR-side hope shifts slightly — less likely that "data hygiene + outlier filter" alone collapses lockbox FPR by ~half (because dor_shkedi failures are the model's, not the labels'). But identity-balanced sampling + contrastive loss in P11_TARGETED are now confirmed safe to use, which preserves the training lever.

---

## 1.5 New Diagnostic Signal — Identity-Level Failure Mode (2026-04-27 evening)

A parallel agent running the lockbox-image-tagging-pipeline session shared findings that materially change the strategy. Their analysis is on a 5-identity slice of the lockbox using the R9A_run1 predictions CSV (`inference_results/r9a_run1__teams-faces-data-test-2914-fake-4420-real-feb-28.csv`), **NOT P8A** — so the absolute numbers don't apply to our current best, but the structural patterns are likely robust.

**Headline patterns to integrate:**

1. **The lockbox FPR is identity-driven, not condition-driven.** On the 5-identity slice they analyzed, errors concentrate sharply per identity:
   - `dor_shkedi` (real, n=275): 0% accuracy, median P(fake)=0.95. Sharp 640px frames but tiny ~144² face crops. **One identity drives 33% of the slice and 100% of its real-side errors.**
   - `bla_bla_chow__s1` (real, n=68): 6% accuracy, blurry sharpness=56.
   - `PC_Generator__s15` (real, n=29): 21% accuracy, very tiny ~48² crops, very sharp.
   - `Chikara_Takahashi__s22` (real, n=42): 81% accuracy — **works**.
   - `Cam_Test__s33` (fake, n=334): 90% accuracy — **works**.
   - `PC_Generator__s15` (fake, n=91): 98% accuracy — **works**.

   *Interpretation:* the "outlier capture conditions" hypothesis is correct, but the right axis to filter on is **within-identity face-crop typicality** (face size, sharpness vs blur, crop framing) — NOT just camera/lighting tags. A "modern subset" filter that drops the dor_shkedi tiny-crop outliers could collapse the headline FPR by ~half on its own.

2. **Screen-capture-style fakes are the recall long pole.** `is_likely_screen_capture` CLIP signal has lift 1.95 for FN — **97% of missed-fakes look screen-recording-style**. This is a concrete data-axis intervention: train against synthetic screen-capture overlays on real samples to harden detection of this specific failure mode.

3. **Threshold mismatch — possible plumbing gap.** Their per-frame threshold sweep on this 5-identity slice doesn't reach 5% FPR / 90% recall at any τ. Likely explanations: (a) our scorer aggregates per-video and theirs is per-frame, (b) their predictions are R9A not P8A, (c) different subset. Worth ~1h to reconcile Day 1; if it's per-video aggregation, that itself is a deployment-relevant insight.

4. **Lockbox tagging output is on disk already.** `analysis/lockbox_tagging/lockbox_tags_2026-04-27.parquet` exists — Track A.1's dependency is **unblocked**.

5. **Plumbing notes from agent:** `.jpg` files are PNG-RGBA but uniformly across labels (not a leak); InsightFace identity ArcFace (Layer 3 — within-identity outlier scoring) was deferred because `onnxruntime` not installed. **Layer 3 is the most valuable next add** — it would quantify "this dor_shkedi crop doesn't look like the rest of dor_shkedi" with one number, giving us a principled outlier filter.

**Caveat the agent didn't account for:** all our data is per-frame from per-video captures, with each video having a single capture setup. Per-video aggregation IS the correct evaluation level (frames within a video share conditions). The agent's per-frame analysis is diagnostically useful but may not be the right operating-level for FPR/recall headlines.

**Implications for this plan (integrated below):**
- Track A.5 (NEW): identity-level diagnostics, within-identity outlier scoring, threshold reconciliation. Highest Day-1 leverage.
- Track B variant (NEW): P11_TARGETED gains a synthetic screen-capture augmentation flag if buildable cheaply, OR we run a parallel hedge variant P11_SCRCAP that adds it.
- Modern-subset definition becomes: capture-condition filter ∧ within-identity typicality filter (using either Layer 3 ArcFace outlier score or a cheaper proxy: min face-crop size + sharpness range).

---

## 2. TL;DR + Optimism

**The bridge is three tracks running in parallel, converging Day 4:**

| Track | What it buys | Cost |
|---|---|---|
| **A — Reframe evaluation** | Likely converts 6.2% lockbox FPR → ~3-4% on modern subset; surfaces per-method AUC ceiling so we know if 90% is reachable | ~6h dev |
| **B — Train P11_TARGETED** | New recipe combining contrastive loss + heavier codec_sim + real-codec-uplift on P8A base; targets the per-method recall gap | ~12h Vertex (~$60) |
| **C — Inference-time stack** | Calibration + ensemble (P8A + C3 + P11) + TTA; expected +5–15 pp recall lift with bounded FPR cost | ~8h dev |

**Optimism: 55% by Day 4, 78% by Day 6 — bumped from prior 50/75 after the parallel agent's findings (§1.5).**

- *55% by Day 4:* The identity-level finding (dor_shkedi alone driving 33% of FPR on the diagnostic slice) is a strong positive signal that the FPR side of the gate is largely a data-hygiene artifact. If A.5c (well-conditioned crops filter) confirms this on P8A, the FPR side may be solved by Day 1 evening. Ensemble + calibration + screen-capture-targeted training then has to deliver the recall side.
- *78% by Day 6:* Day 5–6 absorbs (a) P11_TARGETED converging slowly, (b) ArcFace Layer 3 needing more time, (c) one targeted training run if a single family is the long pole.
- *Floor risk (~22%, down from 25%):* per-method AUC computed Day 1.8 will tell us if the substrate is bounded. If all three families have AUC <0.92 on dev, no operating-point or ensemble trick gets to 90% recall — needs multi-week scratch retrain. The agent's screen-capture finding (97% lift on FN) suggests there's headroom most likely accessible via training not inference.

---

## 3. Targets & Constraints

**Hard targets (the bar):**

| Suite | Metric | Target |
|---|---|---|
| `visomaster_enhanced_macro_dev` | recall at chosen τ | ≥0.90 |
| `deeplive_enhanced_dev` | recall at chosen τ | ≥0.90 |
| `teams_fake_all_dev` | recall at chosen τ | ≥0.90 |
| `modern_lockbox_real` (subset) | FPR at chosen τ | ≤0.05 |
| `teams_real_all_dev` | FPR at chosen τ | ≤0.07 (hard cap) |
| `teams_real_dor_dev` (sanity) | FPR at chosen τ | ≤0.10 (sanity, not gating) |
| Source-bucket probe (plan v2 §6) | test_acc | ≤0.30 (relaxed from 0.25; flag if >0.50) |

**Operational constraints:**
- Must use corrected policy (`arena/score_teams_promotion_contract.py` with FPR 7%/10%, uncommitted but validated).
- Image rebuilds before any Vertex launch with new in-image files (yamls, checkpoint maps). Use `SKIP_IMAGE_CURRENCY_CHECK=1` only as a temporary override.
- US-region training only (us-east1 / us-west4 / us-central1). Region switch after 30 min PENDING.
- No cancelling Vertex training jobs without explicit user authorization.
- `n_jobs=1` enforced on sklearn/joblib (memory `feedback_sklearn_njobs.md`).
- LOG + RESULTS files updated at session start and end.

---

## 4. Strategy — Three Parallel Tracks

### Track A — Reframe evaluation (Days 1–2)

**A.1 — Modern target subset (depends on parallel agent).**
The lockbox-tagging agent is producing per-frame quality + geometry tags at `analysis/lockbox_tagging/`. As soon as their output lands (expected Day 1 evening / Day 2 morning):
- Define `modern_lockbox_real` subset: frames passing tags for (i) JPEG QF in modern-Teams range, (ii) brightness/contrast in standard range, (iii) face geometry within deployment bounds (frontal, well-lit).
- Build `arena/inventories/modern_target_lockbox_subset_2026-04-28.yaml`.
- Re-score all 4 existing candidates (P8A, C3 VC, C3 OOD, C1) against modern subset.
- **Fallback if parallel agent slips:** crude tagging (JPEG QF + brightness via OpenCV) — we built the same primitives at `analysis/lockbox_tagging/layers/quality.py`; can adapt locally in ~2h.

**A.2 — Per-method τ + per-method AUC.**
Extend `arena/score_teams_promotion_contract.py` to emit:
- AUC per fake suite (visomaster_enhanced_macro, deeplive_enhanced, teams_fake_all) at each candidate.
- Optional per-method τ selection (each suite calibrated independently). Reports the FPR cost on dev_real_all of meeting per-method 90% recall.
- This answers the empirical question **"is 90% recall on each family achievable at any τ, given our current substrate?"** before we spend GPU.

**A.3 — Dor sanity integration.**
`teams_real_dor_dev` (50 dor_shkedi videos) is already wired as a readout-only suite at `arena/score_teams_promotion_contract.py:767`. Add to the Day-4 final scorecard at gate-status (warn-only ≤10% FPR; investigate if >10%).

**A.4 — Calibration (Platt + isotonic).**
Use `scripts/run/run_r8_calibration_fit.py` to fit calibrators on dev for P8A_REFERENCE_STEP5000 and C3_CODEC_HEDGE_VC_STEP2000. Apply via `scripts/run/run_r8_apply_calibrator.py` and re-score. Often free 3–8 pp recall lift at fixed FPR.

**A.5 — Identity-level diagnostics + within-identity outlier filter (NEW, highest Day-1 leverage).**
The parallel agent's finding that dor_shkedi alone drives 33% of FPR on a 5-identity slice forces a refined modern-subset definition.
- **A.5a — Reconcile predictions:** the agent used R9A_run1 predictions, not P8A. Re-run P8A inference on these 5 identities (cheap — small slice; reuse existing inference scaffolding `analysis/teams_pool_rescore.py`) to confirm/deny the identity pattern on our current best.
- **A.5b — Reconcile threshold mismatch:** check whether `arena/score_teams_promotion_contract.py` aggregates per-video or per-frame. The agent's per-frame sweep can't reach 5%/90% on the slice; if our scorer is per-video, that's a major reason our headline numbers look better. Write up the answer to RESULTS.
- **A.5c — Within-identity outlier filter (cheap proxy):** use the existing `analysis/lockbox_tagging/lockbox_tags_2026-04-27.parquet` columns for face crop size + sharpness. Build a "well-conditioned crops only" filter (e.g., face_h × face_w ≥ 96² AND sharpness in [normal_capture_range]). Subset the modern lockbox by this.
- **A.5d — Identity ArcFace (gold standard, optional Day 2):** install `onnxruntime` (~30 MB), score all 7,334 frames for within-identity ArcFace outlierness. Filter the modern subset on outlier z-score ≤ 2σ. Per agent: this is the highest-value diagnostic add.
- **A.5e — Per-video FPR table** for transparency: compute FPR by (identity, video) so we know which videos are pathological vs which are typical. Inform the dor sanity gate setting.

This sub-track may eclipse A.1 (capture-condition tagging) as the FPR-side headliner. If A.5c shows the headline FPR collapses by ≥3 pp under "well-conditioned crops only", the FPR side of the gate is essentially solved by data hygiene.

### Track B — Train P11_TARGETED (launch Day 1)

**Hypothesis:** the per-method recall gap is from (i) symmetric codec exposure missing (P9_05 axis), (ii) shortcut leakage from camera-signature (contrastive loss directly attacks), (iii) codec aug undertrained on the dev split (C3 axis under-uplifted on lockbox). A combined recipe addresses all three.

**P11_TARGETED yaml** (new, `experiments/phase2_round13/R13_P11_TARGETED.yaml`):
- **Base checkpoint:** `gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth` (P8A reference step 5000 — same lockbox-side dominator we keep returning to).
- **Inherits P8A flags:** `unfreeze_final_proj=true`, `unfreeze_final_ln=true`, `apply_svd_to_mlp=true`, `apply_svd_to_in_proj=true` (now actually trains thanks to bug fix in commit `2feea58`).
- **Adds (the deltas from P8A):**
  - `augmentation.real_codec_uplift: true` (P9_05 axis — symmetrize codec exposure across real/fake).
  - `augmentation.teams_codec_sim_p: 0.65` (vs P8A 0.50; from C3 codec_hedge axis).
  - `augmentation.teams_codec_sim_quality: [18, 60]` (slightly wider than C3's [15, 55] for dev coverage).
  - `loss.contrastive_regularization.enabled: true`, `loss.contrastive_regularization.margin: 0.5`, `loss.contrastive_regularization.weight: 0.10`.
  - `dataloader_strategy: combined_paired` with `identity_balanced_sampling: true` (already default).
  - `optimizer.adam.backbone_lr_mult: 0.5` (between P8A's 1.0 and P9_01's 0.3 — slightly soften but not freeze).
- **Steps:** 8000 (vs P8A's 5000 winner — give contrastive loss room to settle).
- **Region:** us-east1 (us-west4 fallback if PENDING > 30 min).
- **Image:** must rebuild after yaml is committed; expected 1.3.221.

**Optional Day-1 hedge variant — P11_SCRCAP** (only launched if Day 1.2 confirms `screen_capture` augmentation can be wired into the existing pipeline cheaply, ≤2h dev):
- Same recipe as P11_TARGETED + new `augmentation.screen_capture_sim_p: 0.30` applied to real samples during training.
- Targets the agent's finding that 97% of missed fakes are screen-capture-style — by exposing real samples to synthetic screen-capture overlays during training, the model should stop using "looks-screen-recorded" as a fake signal.
- If the augmentation primitive doesn't exist in the codebase: build it (cheap — basic monitor refresh patterns + light JPEG re-encode + slight scale + brightness compression). Reuse `data/augmentations/teams_simulation.py` skeleton.
- Cost: parallel ~12h Vertex (~$60). **Only launch if A.5 doesn't already deliver the FPR side and recall is still the gating concern.**

**Mid-run kill switches** (read from W&B `https://wandb.ai/dtect-vision/enhanced-aug-test`):
- Step 1500: anchor_mean > 0.85 → kill (substrate broken; revert to P8A).
- Step 3000: visomaster_enhanced_macro_dev recall < 0.40 → kill (regression below P8A baseline 0.36).
- Step 5000: anchor_mean > 0.65 AND viso recall < 0.55 → kill.
- Step 6500: visomaster_enhanced_macro_dev recall > 0.65 AND lockbox FPR < 8% → strong candidate; let finish.

### Track C — Inference-time stack (Days 2–3)

**C.1 — Multi-checkpoint ensemble.**
Adapt `wma/test_four_model_fusion.py` (existing noisy-OR PoC) into a scorer-compatible wrapper at `analysis/ensemble_2026-04-28/noisy_or_ensemble.py`. Inputs: 2–3 checkpoint score CSVs from existing reports. Outputs: ensemble scores → standard scoring pipeline.

**C.2 — Test-time augmentation.**
For each frame, run 3 forward passes: (a) original, (b) horizontal flip, (c) Teams-codec-sim applied to original at p=1.0. Average probabilities. Check FPR/recall delta.

**C.3 — Per-method τ deployment.**
Extension of A.2: at deployment time, route based on detected source family, apply per-family τ.

---

## 5. Day-by-Day Execution

### Day 1 (today, 2026-04-27 evening → 2026-04-28 morning)

- [ ] **1.1 — Commit policy fix + probe scripts** (~15 min). Stage `arena/score_teams_promotion_contract.py` and the four `analysis/probe_battery_2026-04-26/` files (per HANDOFF.md "Not Yet Done").
- [ ] **1.2 — Track A.5a/b — reconcile predictions and threshold mismatch** (~1.5h, **highest leverage**). Read `analysis/lockbox_tagging/lockbox_tags_2026-04-27.parquet` and the R9A predictions CSV. Re-run P8A inference on the 5 identities the agent flagged (use existing arena pool-rescore scaffolding). Determine per-video vs per-frame aggregation in `arena/score_teams_promotion_contract.py`. Write findings to RESULTS.
- [ ] **1.3 — Track A.5c — well-conditioned crops subset** (~1.5h). Use the parquet's face-size + sharpness columns. Build `arena/inventories/modern_target_lockbox_subset_2026-04-28.yaml` filtered on (face_h × face_w ≥ 96²) AND sharpness in capture range. Re-aggregate P8A's existing report frames against this subset.
- [ ] **1.4 — Author P11_TARGETED yaml** (~30 min). New file `experiments/phase2_round13/R13_P11_TARGETED.yaml`. Base on `R13_P9_05_real_codec_uplift.yaml`; layer in contrastive loss + heavier codec_sim. Verify all referenced flags exist in `data/augmentations/pipelines.py` and `loss/contrastive_regularization.py` before committing.
- [ ] **1.5 — Investigate screen-capture augmentation feasibility** (~30 min). Grep `data/augmentations/` for any screen-capture or monitor-refresh primitive. If exists and ≤2h to wire in: add `screen_capture_sim_p` flag to P11_TARGETED. If not: defer P11_SCRCAP hedge variant until Day 2.
- [ ] **1.6 — Image rebuild** (~5 min build + ~3 min push, cache hit). `./dev.sh build-prod -y` → expected 1.3.221.
- [ ] **1.7 — Launch P11_TARGETED on Vertex** (~5 min). `./scripts/launch/launch_experiment.sh -y enhanced-aug-test us-east1 experiments/phase2_round13/R13_P11_TARGETED.yaml`. Confirm RUNNING within 30 min; switch to us-west4 otherwise.
- [ ] **1.8 — Track A.2 scorer extension** (~3h, can run after launch). Add per-method τ + per-method AUC to `arena/score_teams_promotion_contract.py`. Tests: re-run on existing codec_hedge reports; per-method numbers reconcile against existing aggregates within rounding.
- [ ] **1.9 — Track A.4 calibration fit** (~1h). Run `scripts/run/run_r8_calibration_fit.py` on P8A's existing dev predictions. Output: Platt + isotonic calibrator JSON for P8A and C3.
- [ ] **1.10 — Optional: install onnxruntime + start Layer 3 ArcFace scoring** (~1h to install + ~2h to run on all 7,334 frames). Defer to Day 2 if Day 1 schedule slips. The agent flagged this as the highest-value diagnostic add.
- [ ] **1.11 — Write LOG + RESULTS entries.**

**Exit criterion Day 1:**
- P11_TARGETED RUNNING on Vertex.
- A.5a/b/c done: identity-level finding confirmed/denied on P8A; threshold-aggregation question answered; well-conditioned-crops subset built.
- Per-method τ + AUC scorer extension green; calibrators fitted.
- LOG + RESULTS updated.

### Day 2 (2026-04-28)

- [ ] **2.1 — Layer 3 ArcFace within-identity outlier scoring** (if not done Day 1.10). Score all 7,334 lockbox+dev frames; merge into `lockbox_tags_2026-04-27.parquet`. Define refined modern subset filter using ArcFace outlier z-score ≤ 2σ.
- [ ] **2.2 — Final modern subset definition.** Combine A.1 (capture conditions from parquet) + A.5c (face-crop typicality) + A.5d (ArcFace outlier z). Output: `modern_target_lockbox_subset_v2_2026-04-28.yaml`.
- [ ] **2.3 — Re-score 4 existing candidates against modern subset (v2).** Output: a 4×4 table (P8A, C3-VC, C3-OOD, C1) × (raw lockbox, modern_v1, modern_v2, dor_dev) FPR + per-fake-method recall + per-method AUC.
- [ ] **2.4 — Apply calibrators + per-method τ on existing candidates.** Output: scorecard at calibrated τ + per-method τ. Decision-grade table.
- [ ] **2.5 — Build noisy-OR ensemble of P8A + C3.** Score against modern subset + dev fake suites + dor sanity. Compare to single-model baselines.
- [ ] **2.6 — Decision gate (end of Day 2 evening).** Three possible verdicts:
  - **(α) Already there:** ensemble @ calibrated per-method τ hits 90/5 on modern subset v2. → Day 3–4 become validation + ship. P11_TARGETED becomes upside / regression-baseline.
  - **(β) Close but no:** within 5–10 pp on at least one family. → Day 3 validates P11_TARGETED; if it adds the missing pp, we ship the 3-way ensemble.
  - **(γ) Big gap:** >10 pp short on a family OR per-method AUC <0.92 on any family (ceiling concern). → invoke extension protocol; consider screen-capture hedge launch (P11_SCRCAP).
- [ ] **2.7 — If verdict is γ AND screen-capture aug is buildable (Day 1.5 outcome positive):** launch P11_SCRCAP as hedge variant Day 2 evening (parallel ~12h Vertex).
- [ ] **2.8 — RESULTS + LOG update.**

### Day 3 (2026-04-29)

- [ ] **3.1 — P11_TARGETED finishes** (expected mid-day per 12h ETA from Day 1 evening launch). Pull best `value_composite` checkpoint per W&B summary.
- [ ] **3.2 — Probe extraction on P11_TARGETED.** Use `analysis/probe_battery_2026-04-26/launch_vertex.sh --checkpoint <gs://...> --run-tag p11_targeted`. ~30 min Vertex; ~3 min local sklearn.
- [ ] **3.3 — Full A.2-style validation on P11_TARGETED.** Same protocol as codec_hedge validation 2026-04-27. Output: contract scorecard + diagnostic τ=0.5 readout + dev fake per-method recall.
- [ ] **3.4 — Update ensemble: add P11_TARGETED.** Score 3-way noisy-OR (P8A + C3 + P11). If P11 fails probe gate badly (>0.50 source_bucket), exclude from ensemble.
- [ ] **3.5 — TTA evaluation** (if Day 2 verdict was β or γ). Implement 3-augmentation TTA on the best ensemble; measure FPR/recall delta.
- [ ] **3.6 — Decision gate:** is best candidate (single or ensemble, possibly TTA) at 90/5? If yes → Day 4 final validation + ship. If no → invoke extension protocol (Day 5–6).
- [ ] **3.7 — RESULTS + LOG update.**

### Day 4 (2026-04-30)

- [ ] **4.1 — Final validation.** Run best candidate against full target-domain suite: modern lockbox + raw lockbox + dor sanity + dev fake families + heldout/OOD (treat OOD as informational). Verify probe verdict on the deployed checkpoint or ensemble component.
- [ ] **4.2 — Build deployment scorecard.** Single document with: per-suite FPR + recall, calibration diagnostics, probe verdict, threshold(s) deployed, regression check vs P8A baseline.
- [ ] **4.3 — Decision: ship / extend / retreat.**
  - **Ship:** all hard targets met. Crown candidate. Document recipe + thresholds.
  - **Extend:** within 3 pp on one family; launch one targeted P12 training (e.g., visomaster-heavy hard-example mining if visomaster is the long pole). Day 5–6.
  - **Retreat:** more than 3 pp short on any family AND probe analysis shows substrate-level ceiling. Document floor + propose multi-week scratch retrain plan.
- [ ] **4.4 — RESULTS + LOG update + final HANDOFF.md refresh.**

### Days 5–6 (extension, only if Day 4 verdict = extend)

- [ ] **5.1 — Targeted P12 training** based on the failure mode identified Day 4. Examples:
  - If visomaster recall is the long pole: P12_VISO = P11_TARGETED + 2× visomaster sampling weight + visomaster-only contrastive negatives.
  - If deeplive recall is the long pole: P12_LIVE = P11_TARGETED + deeplive enhancement strategies in identity-balanced sampling.
  - If real_lockbox FPR is the long pole and modern subset is also failing: P12_REAL_HEDGE = P11_TARGETED + heavier real-side augmentation specifically on capture conditions where modern subset still fails.
- [ ] **5.2 — Validate Day 6.** Final ship/retreat decision Day 6 EOD.

---

## 6. Decision Gates (quantitative)

**Day 2 Gate (end of A+C track):**
- IF best ensemble @ calibrated per-method τ achieves all three:
  - viso recall ≥ 0.90 AND deeplive recall ≥ 0.90 AND teams_fake_all recall ≥ 0.90
  - modern_lockbox FPR ≤ 0.05
  - teams_real_all_dev FPR ≤ 0.07
- THEN → defer P11_TARGETED to fallback; spend Day 3 on robustness validation + Day 4 on ship.
- ELSE → continue P11; Day 3 ensemble adds P11.

**Day 3 Gate:**
- IF best 3-way ensemble (P8A + C3 + P11) + per-method τ + calibration achieves 90/5 → Day 4 ship.
- ELSE IF closest within 5 pp on one family → invoke Day 5–6 extension with targeted P12.
- ELSE → flag substrate-ceiling concern; Day 4 produces honest report + multi-week plan.

**Day 4 Final Gate:**
- ALL of: viso ≥0.90, deeplive ≥0.90, teams_fake_all ≥0.90, modern_lockbox ≤0.05, dor sanity ≤0.10, regression vs P8A baseline within tolerance.
- ANY missing → user picks ship-with-caveat / extend / retreat.

---

## 7. Validation Framework

| Suite | Source | Used for | Notes |
|---|---|---|---|
| `modern_lockbox_real_v1` | `analysis/lockbox_tagging/lockbox_tags_2026-04-27.parquet` (capture-conditions filter) | FPR gate v1 (target ≤5%) | Day 1.3 |
| `modern_lockbox_real_v2` | v1 + within-identity outlier filter (Layer 3 ArcFace or face-size proxy) | FPR gate v2 (target ≤5%) | Day 2.2 |
| `modern_lockbox_fake` | same source | recall floor (informational) | New; built Day 1–2 |
| `teams_real_dor_dev` | existing manifest 2026-04-23 | sanity gate (target ≤10%) | Already wired |
| `visomaster_enhanced_macro_dev` | existing | recall gate (target ≥90%) | Already in scorer |
| `deeplive_enhanced_dev` | existing | recall gate (target ≥90%) | Already in scorer |
| `teams_fake_all_dev` | existing | recall gate (target ≥90%) | Already in scorer |
| Source-bucket probe | existing infra | shortcut-leak diagnostic | Plan v2 §6, relaxed gate |
| Raw lockbox (FPR + recall) | existing | transparency / regression | Reported but not gating |
| Heldout/OOD | existing (broken — investigate) | informational only | Don't invest in fixing |

---

## 8. Critical Files

### Existing — read-only / use as-is
- `detectors/effort_detector.py` — has in_proj-SVD fix (committed `2feea58`).
- `arena/score_teams_promotion_contract.py` — corrected policy (uncommitted; commit Day 1.1).
- `experiments/phase2_round13/R13_RLP8_01_unfreeze_clip_codec.yaml` — P8A reference recipe.
- `experiments/phase2_round13/R13_P9_05_real_codec_uplift.yaml` — base for P11 yaml.
- `data/sources/combined_paired.py:1-141` — paired loader; identity-stratified.
- `data/batching/df40_paired.py:66-405` — paired batch composition (collate fn).
- `data/augmentations/teams_simulation.py:1-150` — Teams codec sim (`teams_codec_sim_p`, `teams_codec_sim_quality`).
- `loss/contrastive_regularization.py:38-78` — `ContrastiveLoss` (registered, needs activation in yaml).
- `trainer/trainer.py:1141,1276-1335` — paired strategy detection + frame budgeting.
- `analysis/probe_battery_2026-04-26/run_linear_probes.py` — probe scorer (uncommitted; commit Day 1.1).
- `analysis/probe_battery_2026-04-26/extract_features_for_probes.py` — feature extractor.
- `analysis/probe_battery_2026-04-26/launch_vertex.sh` — Vertex launcher for probe features.
- `scripts/run/run_r8_calibration_fit.py` — Platt + isotonic calibration fit.
- `scripts/run/run_r8_apply_calibrator.py` — calibrator application.
- `wma/test_four_model_fusion.py` — noisy-OR fusion PoC (adapt for ensemble).
- `analysis/lockbox_tagging/io_utils.py`, `analysis/lockbox_tagging/layers/quality.py`, `analysis/lockbox_tagging/layers/face_geometry.py`, `analysis/lockbox_tagging/run_tagging.py`, `analysis/lockbox_tagging/analyze.py` — parallel agent's infra. **Output already on disk:** `analysis/lockbox_tagging/lockbox_tags_2026-04-27.parquet`.
- `inference_results/r9a_run1__teams-faces-data-test-2914-fake-4420-real-feb-28.csv` — the predictions CSV the agent used (R9A run, NOT P8A); needed for cross-checking the identity-level finding.
- `arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json` — has `teams_real_dor_dev` (50 dor_shkedi videos).
- `arena/launch_teams_promotion_contract.sh` — scorer launcher (must export WANDB_*).

### To create
- `experiments/phase2_round13/R13_P11_TARGETED.yaml` — Day 1.2 (~250 lines).
- `arena/inventories/modern_target_lockbox_subset_2026-04-28.yaml` — Day 2.1 (consumes parallel agent's tags).
- `arena/checkpoint_maps/teams_target_domain.day4_final_2026-04-30.yaml` — Day 4.
- `analysis/per_method_threshold_2026-04-28/scorer_extension.py` — Day 1.5 (per-method τ + AUC extension).
- `analysis/ensemble_2026-04-28/noisy_or_ensemble.py` — Day 2.4 (~150 lines, adapts `wma/test_four_model_fusion.py`).
- `analysis/tta_2026-04-29/tta_inference.py` — Day 3.5 (~100 lines, only if needed).

### To modify (small, additive)
- `arena/score_teams_promotion_contract.py` — Day 1.5: add per-method τ selection + per-method AUC reporting; back-compat with global τ.
- `experiments/phase2_round13/R13_P11_TARGETED.yaml` may need wiring of `loss.contrastive_regularization.enabled` if not already first-class in the yaml schema (verify via `trainer/trainer.py` config parsing).

---

## 9. Risks & Mitigations

1. **P11_TARGETED could underperform / regress like C3.** Mitigation: kill-switches at step 1500/3000/5000. Ensemble fallback ready if P11 fails.
2. **Lockbox-tagging output may slip past Day 2.** Mitigation: crude tagger fallback using JPEG QF + brightness primitives we can build in 2h from existing `analysis/lockbox_tagging/layers/quality.py`.
3. **Per-method τ may explode dev_real_all FPR even if it lifts per-family recall.** Mitigation: hard cap dev_real_all_dev FPR at 7% in the per-method optimizer; reject solutions that breach.
4. **Ensemble of leaky candidates could compound shortcut.** Mitigation: probe each candidate first; weight by 1 / probe_test_acc OR exclude any candidate above 0.55.
5. **The 90/all-three target may be substrate-bounded.** Mitigation: per-method AUC computed Day 1; if AUC <0.92 on any family, escalate the substrate-ceiling concern early so user can decide ship-with-caveat vs multi-week.
6. **Contrastive loss may interact badly with the in_proj-SVD fix.** Mitigation: monitor `train/loss/contrastive` and `train/loss/cls` ratio at step 500; if contrastive >5× cls, weight is too high — kill and relaunch with weight 0.05.
7. **Image rebuild cost.** Each yaml/checkpoint-map change requires a rebuild. Mitigation: bundle all Day 1 in-image changes into one rebuild (P11 yaml + any auxiliary inventory files).
8. **Dev-vs-lockbox mismatch.** All recall targets are on dev; lockbox recall may be lower. Mitigation: Day 4 transparency table reports both.
9. **Identity-level finding may not transfer to P8A.** Agent used R9A predictions; P8A may distribute errors differently. Mitigation: Day 1.2 re-runs P8A on the 5-identity slice; if pattern doesn't hold, A.5c filter is less impactful and we revert to capture-condition-only modern subset.
10. **Within-identity outlier filter could over-prune the lockbox real pool.** If we drop too many "outlier" frames, the FPR denominator shrinks and the metric stops being meaningful. Mitigation: cap pruning at 30% of any individual identity's frames; report fraction-pruned per identity in scorecard for transparency.
11. **Threshold mismatch (per-frame vs per-video) might reveal our existing headlines depend on per-video aggregation.** That would be deployment-relevant info — Teams operates per-frame at inference. Mitigation: if true, a "deployment-honest" per-frame metric column added to the Day 4 scorecard.
12. **Screen-capture augmentation is novel territory.** No primitive exists in the codebase. Mitigation: keep it as an optional Day-2 hedge (P11_SCRCAP), not a Day-1 critical-path dependency.

---

## 10. Open Questions / Dependencies

- **Lockbox-tagging output is on disk already** (`lockbox_tags_2026-04-27.parquet`). Schema needs to be inspected Day 1.2 — column names, frame-id keying, completeness across the 7,334 frames the agent referenced.
- **Per-video vs per-frame aggregation** — answered by Day 1.2 inspection of `arena/score_teams_promotion_contract.py`. Result feeds the deployment-honesty conversation.
- **InsightFace ArcFace install on this machine** — `pip install onnxruntime` is ~30 MB; user can authorize Day 1 evening or defer.
- **Ensemble weighting policy** — equal weights vs probe-cleanliness-weighted vs dev-recall-weighted. Decision: start with noisy-OR equal; test weighted only if equal underperforms.
- **TTA augmentation set** — flip + Teams-codec is the minimal set; expand to color jitter only if base TTA shows lift.
- **P11_SCRCAP launch decision** — depends on Day 1.5 (does screen-capture aug exist or build cheap?) AND Day 2 verdict (is recall the gating concern?). Decision deferred to Day 2 evening.

---

## 11. Coordinator Protocol

- **LOG file:** `april-26-training-master-plan-v2.LOG.md` — append entry per session.
- **RESULTS file:** `april-26-training-master-plan-v2.RESULTS.md` — append entry per measurement.
- **Plan file (this):** read-only for execution agents. Plan changes go in LOG with `PLAN CHANGE PROPOSAL` block; user authorizes.
- **Memory updates:** if a discovery generalizes beyond this sprint (e.g., "real_codec_uplift was the missing P9 lever"), file under `~/.claude/projects/.../memory/`.
- **Commit cadence:** small, frequent. Yaml + scorer extensions in separate commits from training launches.
- **HANDOFF.md** refresh at end of each working day.

---

## 12. Resume Instructions (next agent)

1. Read this plan + last LOG entry + last RESULTS entry.
2. Read `MEMORY.md` and the listed memory files (especially `project_promotion_contract.md`, `project_contract_policy_bug.md`, `project_p8a_breakthrough.md`, `feedback_sklearn_njobs.md`, `feedback_no_cancelling_vertex_jobs.md`).
3. Run the state-drift check from prior HANDOFF.md (Section "Resume Instructions" item 1).
4. Identify which day we're on; pick next checkbox from §5; execute.
5. End session with LOG entry + (if measurements made) RESULTS entry.

---

## 13. Honest Read on Optimism

Putting raw probabilities on the wall for the user:

| Outcome | P |
|---|---|
| 90/5 met by Day 4 | 0.55 |
| 90/5 met by Day 6 (with extension) | 0.78 |
| FPR side ≤5% solved by Day 2 via well-conditioned subset (no model change) | 0.65 |
| Within 5 pp of 90 on at least 2 of 3 families by Day 4 | 0.85 |
| Substrate ceiling rules out 90/5 even with 2-week effort | 0.18 |
| Big positive surprise (identity-cleanup or screen-capture aug single-handedly closes a major gap) | 0.70 |

**The most likely failure mode** is visomaster_enhanced_macro recall stalling at 70–80% — that's the family with the lowest current recall (35.6% at τ=0.5) and the smallest dev sample. If that's the long pole, Day 5–6 extension targeting visomaster is the default.

**The most likely positive surprise** has shifted with §1.5: instead of "modern-subset reframe converts 6.2% FPR to 3-4%," it's now plausibly "well-conditioned crops + within-identity outlier filter converts 6.2% to ~1-2%" — because if dor_shkedi alone is 33% of the FPR-relevant slice and at 0% accuracy, dropping the pathological-crop subset of dor (or filtering all identities by face-crop quality) collapses the headline.

The recipe-tuning era is over; this plan is honest about that. The data-hygiene + data-axis training + inference-time stack is what's left, and §1.5 confirms there's real headroom in data hygiene that nobody had quantified before today. Four days is tight but plausibly enough.

The recipe-tuning era is over; this plan is honest about that. The data-axis + inference-time stack is what's left, and it's exactly what's been queued in the codebase but never executed end-to-end. Four days is tight but not unreasonable for the parallel-track strategy. If the substrate isn't there, we'll know cleanly by Day 4 and ship that signal honestly.
