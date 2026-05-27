# Recall-First Sprint — Plan v4

**Author:** Claude Opus 4.7 (1M context), post-Option-A pivot
**Date:** 2026-04-28 (end of Day 2 of original 4-day sprint, extension authorized)
**Working dir:** `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/`
**Branch:** `teams-relaunch-root-2026-04-17`
**Replaces:** Plan v3 §4 (all three tracks closed). Plan v3 §1.5 / §1.6 / §6 / §11 / §12 still valid and inherited.
**Companion logs:** `april-26-training-master-plan-v2.LOG.md`, `april-26-training-master-plan-v2.RESULTS.md` (continue appending — same artifact thread).

---

## 1. Context — Why Plan v3 Reached Its Ceiling

Plan v3 closed three pieces of evidence that together force a redesign:

1. **Plan v3 Track B (P11 / P12) — recipe-tuning at FT-extension is OUT.** P11_HEAVY's anchor regression at step 1000 is real and persistent; the P12_HEAVY_LONG trajectory (every-500-step granularity) shows recovery at ~0.04 anchor_mean per 1000 steps — would need ~17000 more steps to reach P8A's level. Pure FT-extension of HEAVY does not close 90/5 within any sane budget.
2. **Plan v3 Track A (Option A — inference-time stack) — ensemble + calibration + per-method τ is OUT.** The subagent run on 2026-04-28 evaluated 30 single-candidate variants × 3 calibrations + 45 noisy-OR ensembles + 10 weighted ensembles. **0 of 85 configurations cleared the gate.** The binding constraint is **AUC**: best ensemble AUC against modern_v2 real on visomaster_enhanced_macro = **0.647**. To hit 90% recall at 5% FPR you need AUC ≈ 0.95. Calibration moves probabilities, not AUC; ensembles of same-deficit candidates cannot break the substrate ceiling.
3. **The trainer's `periodic_saves` patch silently no-ops.** ONE-LINE FIX is identified but **untested**. Burned ~$25 + 4h on P12 for nothing. Any future training launch must apply + smoke-test the fix first.

**Implication:** Plan v3 Verdict §6 = γ. The substrate is bounded under the current (P8A FT + P11 family) recipe space. We need a **substrate-redesign training run** with mechanisms that haven't been tried, or an honest substrate-ceiling report and a multi-week scratch-retrain plan.

What hasn't been tried (and is buildable inside the sprint budget):

- **Anchor-aware loss term** — explicit penalty on anchor-pool mean prob. Stops the recall-vs-anchor trade-off from punishing anchor every time HEAVY recipe lifts recall.
- **Visomaster-targeted oversampling at 5×** — viso is the long pole (best ensemble recall 25.8%); HEAVY's 3.0 weight wasn't enough. Also: visomaster-only contrastive negatives.
- **Real-side capture-condition matching** — training reals are not distribution-matched to modern_v2 (newer Teams capture); covariate shift is part of why FPR is high on modern_v2 even at moderate τ.
- **Screen-capture synthetic augmentation on real samples** — agent's lockbox-tagging finding: 97% of FN on the diagnostic slice look screen-recorded. This is a concrete data-axis intervention that no recipe to date has touched.
- **AUC-saturated early-stopping replaced by anchor-aware patience** — current early_stopping fires at step 6000 of an 8000-step run because val_holdout AUC saturates at step 1000.

---

## 2. TL;DR + Optimism

**The bridge to 90/5 in this sprint, if it exists, runs through one focused redesign training run plus a parallel data-hygiene track. Multi-week scratch-retrain is now the credible fallback.**

| Track | What it buys | Cost | When |
|---|---|---|---|
| **G — Substrate redesign (P13_ANCHOR_AWARE)** | Anchor-aware loss + viso-5× + screen-capture aug + working periodic_saves; targets the AUC-ceiling problem head-on | ~12h Vertex (~$45) | Launch Day 3 evening, results Day 4 morning |
| **H — Data-axis investigation (parallel, no GPU)** | ArcFace cluster training set (label corruption test); modern_v2 filter audit; scratch-retrain scope doc | ~8h dev | Day 3–4 mornings |
| **I — Honest report + scratch-retrain proposal** | If P13 doesn't clear: a deployable summary + a credible 2–3 week plan to fix the substrate properly | ~4h dev | Day 4 evening / Day 5 |

**Optimism (revised honestly after Option A's γ verdict):**

| Outcome | P (was, plan v3) | P (revised) |
|---|---|---|
| 90/5 met by Day 4 (sprint deadline) | 0.55 | **0.18** |
| 90/5 met by Day 6 (extension) | 0.78 | **0.32** |
| Within 5pp on at least 2 of 3 families by Day 4 | 0.85 | **0.55** |
| Substrate ceiling confirmed → ship-with-caveat by Day 5 | n/a | **0.55** |
| Multi-week scratch retrain succeeds (90/5) | n/a | **~0.65** (informed estimate) |
| P13 single-handedly closes the gap | n/a | **0.20** |

**The honest read:** Option A confirmed the substrate ceiling on the current candidate set. P13 has to do two things at once — preserve HEAVY's recall lifts while breaking the anchor regression — and previous recipes that tried less than this all failed. The recipe direction is right (HEAVY > MILD on every recall metric); the question is whether one anchor-aware mechanism + viso-5× + screen-capture aug is enough additional levers. If P13 fails, ship-with-caveat at the current operating point and pivot to a multi-week clean-data retrain becomes the recommendation. Day 4 is the gate; Day 5–6 absorbs either polish (if P13 clears) or scratch-retrain prep.

---

## 3. Targets & Constraints (unchanged from Plan v3)

**Hard targets:**

| Suite | Metric | Target |
|---|---|---|
| `visomaster_enhanced_macro_dev` | recall at chosen τ | ≥0.90 |
| `deeplive_enhanced_dev` | recall at chosen τ | ≥0.90 |
| `teams_fake_all_dev` | recall at chosen τ | ≥0.90 |
| `modern_lockbox_real_v2` (281 frames) | FPR at chosen τ | ≤0.05 |
| `teams_real_all_dev` | FPR at chosen τ | ≤0.07 (hard cap) |
| `teams_real_dor_dev` (sanity) | FPR at chosen τ | ≤0.10 (warn-only) |

**Operational constraints (new in Plan v4):**

- **Trainer bug fix is mandatory before any Vertex launch.** Apply the `isinstance(dict)` defense at `trainer/trainer.py:2326-2329`. Add a `self.logger.info(f"periodic_saves check: type={type(...)} val={...}")` debug log. Run a CPU smoke test (load yaml, instantiate trainer, simulate one validation call) and confirm "periodic_save triggered" log fires. Only then rebuild image + launch. **No exceptions.**
- **Don't cancel Vertex training jobs without explicit user authorization** (memory `feedback_no_cancelling_vertex_jobs.md`).
- **`n_jobs=1` everywhere on this Mac** (memory `feedback_sklearn_njobs.md`).
- **US-region only** (us-east1 / us-west4 / us-central1). Switch after 30 min PENDING.
- **Image rebuild before Vertex launch with new in-image files** (yaml, checkpoint maps, trainer/ patches).

---

## 4. Strategy — Three Tracks

### Track G — Substrate Redesign (P13_ANCHOR_AWARE) — primary

**Hypothesis:** The recall-vs-anchor trade-off in HEAVY is a side-effect of training without an explicit anchor objective. Adding (a) an anchor-loss term that penalizes high mean prob on the anchor pool, (b) visomaster oversampling at 5× (over HEAVY's 3.0), (c) screen-capture synthetic aug on real samples, and (d) working periodic_saves at 9 step granularity, will preserve HEAVY's recall lifts (+16/+25/+9 pp on viso/deeplive/teams_fake at step 1000 vs P8A) while restoring anchor to P8A-comparable levels by step 4000–5000.

**Recipe — `experiments/phase2_round13/R13_P13_ANCHOR_AWARE.yaml` (new, ~280 lines):**

- **Base checkpoint:** `gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth` (P8A reference — same anchor-best lockbox-stable base we keep returning to).
- **Inherits HEAVY recipe deltas** (the recall direction that worked):
  - `unfreeze_final_proj=true`, `unfreeze_final_ln=true`, `apply_svd_to_mlp=true`, `apply_svd_to_in_proj=true` (in-proj-SVD bug now fixed; commit `2feea58`).
  - `augmentation.real_codec_uplift: true` (P9_05 axis).
  - `augmentation.teams_codec_sim_p: 0.65`, `teams_codec_sim_quality: [18, 60]`.
  - `loss.contrastive_regularization.enabled: true`, `margin: 0.5`, `weight: 0.10`.
  - `dataloader_strategy: combined_paired`, `identity_balanced_sampling: true`.
  - `optimizer.adam.backbone_lr_mult: 0.5`.
- **NEW deltas (the v4 levers):**
  - `loss.anchor_aware_penalty.enabled: true`, `target_mean_prob: 0.10`, `weight: 5.0`. (Add penalty term `weight * max(0, anchor_pool_mean_prob - target_mean_prob)` computed every N=200 steps on the cached anchor pool.) **Requires implementation in `loss/` — see §8.**
  - `dataloader.method_oversample.visomaster_enhanced: 5.0` (HEAVY had 3.0).
  - `dataloader.method_oversample.deeplive_enhanced: 3.0` (keep HEAVY default — HEAVY_DEEPLIVE 6× was the dud per HANDOFF Failed Approach #5).
  - `augmentation.screen_capture_sim_p: 0.30` on real samples ONLY. **Requires implementation in `data/augmentations/` — see §8.** Cheap primitive: monitor refresh patterns + light JPEG re-encode + slight downscale + slight brightness compression.
  - `early_stopping_patience: 50` (was implicit 10) — prevents AUC-saturation truncation.
  - `periodic_saves.enabled: true`, `step_list: [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000]`. **Trainer bug must be fixed first.**
- **Steps:** 6000 (was 8000 in P12 — but P12 truncated at 6000 due to early-stopping, and the marginal data after step 4000 is the load-bearing range). Total run ~9h.
- **Region:** us-east1 (us-west4 fallback if PENDING > 30 min).

**Mid-run kill switches (read W&B `https://wandb.ai/dtect-vision/enhanced-aug-test`):**
- Step 500: `train/loss/total > 1.5 × P8A_baseline` → kill (anchor-aware penalty weight too high; relaunch with weight 2.0).
- Step 1000: `anchor/anchor_mean > 0.95` AND viso recall < HEAVY_step1000_baseline (0.520) → kill (substrate broken in both directions).
- Step 2500: `anchor/anchor_mean > 0.65 AND viso recall < 0.55` → kill (no improvement over P11_HEAVY trajectory).
- Step 4000: if `anchor/anchor_mean ≤ 0.30 AND viso recall ≥ 0.55` → strong candidate; let finish.
- Step 5000–6000: if `anchor/anchor_mean ≤ 0.20 AND viso recall ≥ 0.65` → P13 succeeded structurally; ship-or-extend decision Day 4.

### Track H — Data-Axis Investigation (parallel, no GPU)

**Hypothesis:** The substrate ceiling reported by Option A may be partly a measurement artifact (modern_v2 filter pruning meaningful real distribution variation) AND/OR partly a label-corruption artifact in training (per Plan v3 §1.6). Track H quantifies both before any multi-week proposal commits resources.

**H.1 — Modern_v2 filter audit (Day 3 morning, ~3h).**
- Schema-inspect `analysis/lockbox_tagging/lockbox_tags_2026-04-27.parquet`.
- Re-derive modern_v2 with relaxed thresholds: face crop ≥ 64² (vs current 96²), sharpness band widened to capture range 90th percentile (vs current 50–95th). Compare modern_v2 vs modern_v2_relaxed on the 5 candidate scorecards.
- If modern_v2_relaxed has **materially different** P8A FPR (Δ ≥ 1.5pp), the current modern_v2 is over-pruning and the AUC-ceiling claim from Option A is partially measurement-induced. Re-run Option A's analysis on modern_v2_relaxed.

**H.2 — Training-set ArcFace identity audit (Day 3 afternoon, ~3h).**
- Install `onnxruntime` if not present (~30 MB).
- Sample 30 identities from training set. Cluster face embeddings within each "identity" label.
- If average within-identity cluster purity < 0.85, the training labels are corrupted (per the Plan v3 §1.6 hypothesis). This invalidates `identity_balanced_sampling` and the contrastive loss's identity-grouping; the camera-signature shortcut becomes a likely artifact of mislabeled identities.
- If corruption confirmed: P13 should NOT use contrastive_regularization. Add to plan v4 §6 decision gate.

**H.3 — Per-frame vs per-video aggregation reconciliation (Day 3 morning, ~30 min).**
- Inspect `arena/score_teams_promotion_contract.py` for aggregation logic. If per-video, document the per-frame variant for deployment honesty.

**H.4 — Scratch-retrain scope doc (Day 4 morning, ~3h, only if Track G unclear).**
- Catalog clean training data: which sources have verified labels, which have ArcFace-clustered identity audits.
- Recipe outline: which P8A flags survive, what additional axes (e.g., 4× more visomaster fakes, modern Teams capture diversity).
- Compute estimate, time estimate, success-probability estimate. Deliverable that lets the user decide ship-with-caveat vs commit-multi-week on Day 5.

### Track I — Honest Report + Scratch-Retrain Proposal (Day 4 evening / Day 5)

Triggered only if Track G fails to clear 90/5 by Day 4 evening.

- **I.1** — Final Day-4 scorecard with P13 + ensemble + per-method τ included. Document the operating-point ceiling clearly: viso recall achievable at modern_v2 FPR 5% = X%, etc. The user reads this to make ship/no-ship decision.
- **I.2** — Multi-week scratch-retrain proposal absorbing Track H findings. Time, cost, success-probability bounds. Hand off to user; this becomes a separate sprint.

---

## 5. Day-by-Day Execution

### Day 3 (2026-04-29)

- [ ] **3.1 — Periodic_saves bug fix + smoke test (~2h, BLOCKING).** Apply `isinstance(dict)` guard at `trainer/trainer.py:2326-2329`. Add debug log line above the gate. Local CPU smoke: load `R13_P12_HEAVY_LONG.yaml`, instantiate trainer (no GPU forward), call the periodic-saves block at simulated `step_cnt` in `step_list`. Confirm "periodic_save triggered at step=..." log fires. **Only after passing smoke test:** commit trainer.py + bug-fix.
- [ ] **3.2 — Implement anchor-aware loss term (~3h).** New file `loss/anchor_aware_penalty.py` (~80 lines). Function: `compute_anchor_penalty(anchor_pool_probs, target=0.10) -> max(0, mean(probs) - target) * weight`. Wire into trainer's main loss aggregation. Verify the term is logged to W&B as `train/loss/anchor_aware`. Local smoke test on dummy anchor pool.
- [ ] **3.3 — Implement screen-capture aug primitive (~2h).** New file `data/augmentations/screen_capture_sim.py` (~120 lines). Steps: (a) random 1–3px horizontal/vertical scan-line streaks at random spatial frequencies; (b) JPEG re-encode at quality 60–85; (c) random downscale to 0.85–1.0× then upscale; (d) slight brightness compression (multiply by 0.92–1.0). Wire into `data/augmentations/pipelines.py` under flag `screen_capture_sim_p`. Visual smoke: write 10 augmented samples to `analysis/screen_capture_smoke_2026-04-29/` and visually verify they look screen-recorded.
- [ ] **3.4 — Track H.1 modern_v2 filter audit (~3h).** Per §4.H.1. Output: `analysis/modern_v2_audit_2026-04-29/` with diagnostics CSV + decision: keep modern_v2 or switch to modern_v2_relaxed for Day-4 scoring.
- [ ] **3.5 — Author P13_ANCHOR_AWARE yaml (~30 min).** New file `experiments/phase2_round13/R13_P13_ANCHOR_AWARE.yaml`. Reference all newly-implemented flags. Commit yaml + supporting flag schema if any.
- [ ] **3.6 — Image rebuild (~5 min build + ~3 min push).** `./dev.sh build-prod -y` → expected 1.3.224.
- [ ] **3.7 — User authorization for P13 launch.** Per Plan v3 §11 / memory `feedback_no_cancelling_vertex_jobs.md`. Wait for explicit OK.
- [ ] **3.8 — Launch P13_ANCHOR_AWARE (~5 min).** `./scripts/launch/launch_experiment.sh -y enhanced-aug-test us-east1 experiments/phase2_round13/R13_P13_ANCHOR_AWARE.yaml`. Confirm RUNNING within 30 min; switch to us-west4 otherwise.
- [ ] **3.9 — Track H.2 ArcFace audit kicks off (~3h, parallel to GPU run).**
- [ ] **3.10 — LOG + RESULTS entries.**

**Exit criterion Day 3:** P13 RUNNING. Bug fix committed + tested. Anchor-aware loss + screen-capture aug landed in code. Modern_v2 audit complete. ArcFace audit underway or complete.

### Day 4 (2026-04-30, original sprint deadline)

- [ ] **4.1 — P13 finishes (expected 09:00–11:00 CEST).** Pull all 9 periodic ckpts (steps 500–6000) from GCS.
- [ ] **4.2 — Score P13 ckpts on full validation suite.** Use `arena/launch_teams_promotion_contract.sh` over the 9 P13 ckpts × 8 suites. Build P13 scorecard analogous to `analysis/modern_lockbox_v2_2026-04-27/p11_overnight_modern_v2_scorecard.csv`.
- [ ] **4.3 — Ensemble P13 with P8A.** Add P13's best-anchor ckpt to the noisy-OR ensemble from Option A. Re-run inference-time stack analysis. Expected runtime ~30 min.
- [ ] **4.4 — Per-method τ + calibration on P13 best ckpt.** Apply `scripts/run/run_r8_calibration_fit.py` + Option A's per-method τ optimizer. Output: P13 calibrated scorecard.
- [ ] **4.5 — Decision gate (Day-4 evening, ~17:00 CEST):**
  - **(α) P13 + ensemble at 90/5:** validate on dor sanity + lockbox raw → ship Day 5. Update HANDOFF.md as "ship".
  - **(β) Within 5pp on 1–2 families, modern_v2 FPR ≤ 5%:** invoke extension protocol → Track G' (one targeted hard-example mining run on the failing family) Day 5–6.
  - **(γ) Substrate ceiling confirmed (best ensemble AUC < 0.85 on all three families):** Track I activates Day 5. Honest report to user + scratch-retrain proposal.
- [ ] **4.6 — LOG + RESULTS update + HANDOFF.md refresh.**

### Day 5 (2026-05-01, extension Day 1)

- [ ] **5.1 — Per Day-4 verdict:**
  - α path: final ship validation; deployment scorecard; close out the sprint.
  - β path: launch P14_HARD (one targeted training run, e.g., 2× viso oversample + viso-only contrastive negatives).
  - γ path: write Track I.2 scratch-retrain proposal; user decision.
- [ ] **5.2 — Track H.4 if not done.**
- [ ] **5.3 — LOG + RESULTS update.**

### Day 6 (2026-05-02, extension Day 2 — final)

- [ ] **6.1 — Final ship/extend/retreat decision.**
- [ ] **6.2 — Comprehensive HANDOFF.md update for next phase (whether deploy-validation or multi-week-retrain).**

---

## 6. Decision Gates (quantitative)

**Day-4 Gate (P13 final + ensemble):**

```
α (ship)  ALL of: viso_dev recall ≥ 0.90, deeplive_dev recall ≥ 0.90,
          teams_fake_all_dev recall ≥ 0.90, modern_v2 FPR ≤ 0.05,
          teams_real_all_dev FPR ≤ 0.07, dor sanity ≤ 0.10
β (extend) Within 5pp on 1–2 of recall families AND modern_v2 FPR ≤ 0.05
γ (retreat) Best ensemble AUC < 0.85 on any family (substrate ceiling)
           OR best ensemble fails modern_v2 FPR ≤ 0.05 at any tau
           that gives recall ≥ 0.50 on all three families
```

**ArcFace audit (Track H.2) gate:**

```
clean (within-identity purity ≥ 0.85): training labels usable; contrastive loss survives in P14 if needed
corrupted (purity < 0.85): training labels NOT trustable — recommend scratch-retrain pivot;
                          remove contrastive loss from any future recipe;
                          modern_v2-only evaluation tightening becomes load-bearing
```

**modern_v2 audit (Track H.1) gate:**

```
filter-confirmed (Δ FPR < 1.5pp on relaxed): keep modern_v2 as authoritative
filter-suspect (Δ ≥ 1.5pp): switch to modern_v2_relaxed for Day-4 scoring;
                            update Plan v3 §6 verdict retrospectively
```

---

## 7. Validation Framework (unchanged from Plan v3 + audit additions)

| Suite | Source | Used for | Notes |
|---|---|---|---|
| `modern_lockbox_real_v2` | `analysis/lockbox_tagging/lockbox_tags_2026-04-27.parquet` | FPR gate (target ≤5%) | 281 frames; under audit Track H.1 |
| `modern_lockbox_real_v2_relaxed` | derived in Track H.1 | FPR gate (alt) | Built Day 3 |
| `teams_real_dor_dev` | existing manifest 2026-04-23 | sanity gate (target ≤10%) | already wired |
| `visomaster_enhanced_macro_dev` | existing | recall gate (target ≥90%) | best current 0.40 calibrated |
| `deeplive_enhanced_dev` | existing | recall gate (target ≥90%) | best current 0.60 calibrated |
| `teams_fake_all_dev` | existing | recall gate (target ≥90%) | best current 0.74 calibrated |
| `teams_real_all_dev` | existing | hard FPR cap (≤7%) | binding; Option A showed 7%/5% gates conflict |
| `teams_real_all_lockbox` | existing | transparency / regression | reported, not gating |
| ArcFace within-identity purity (training set) | new Day-3 H.2 | training-data hygiene | binary gate per §6 |

---

## 8. Critical Files

### Existing — read-only / use as-is

- `detectors/effort_detector.py` — has in_proj-SVD fix.
- `arena/score_teams_promotion_contract.py` — corrected policy; commit pending.
- `experiments/phase2_round13/R13_P12_HEAVY_LONG.yaml` — base for the P13 recipe (HEAVY + viso oversample); copy + extend.
- `experiments/phase2_round13/R13_RLP8_01_unfreeze_clip_codec.yaml` — P8A reference recipe.
- `data/sources/combined_paired.py:1-141` — paired loader.
- `data/augmentations/teams_simulation.py:1-150` — codec sim.
- `data/augmentations/pipelines.py` — aug aggregation; **Track G adds `screen_capture_sim_p` here**.
- `loss/contrastive_regularization.py:38-78` — survives Plan v4 unless ArcFace audit invalidates.
- `trainer/trainer.py:392-393` — the `isinstance(dict)` defense pattern; **Plan v4 G.1 fix references this**.
- `trainer/trainer.py:2319-2362` — broken periodic_saves; **Plan v4 G.1 fixes this**.
- `analysis/option_a_ensemble_2026-04-28/` — Option A subagent output; reference for AUC-ceiling argument.
- `analysis/modern_lockbox_v2_2026-04-27/p11_overnight_modern_v2_scorecard.csv` — P11 baseline scorecard.
- `analysis/lockbox_tagging/lockbox_tags_2026-04-27.parquet` — ground truth for modern_v2 filter audit.
- `arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json` — has `teams_real_dor_dev`.
- `arena/launch_teams_promotion_contract.sh` — scorer launcher.
- `scripts/run/run_r8_calibration_fit.py` — calibration; reused for P13.

### To create (Day 3)

- `loss/anchor_aware_penalty.py` (~80 lines) — anchor-loss term implementation.
- `data/augmentations/screen_capture_sim.py` (~120 lines) — screen-capture aug primitive.
- `experiments/phase2_round13/R13_P13_ANCHOR_AWARE.yaml` (~280 lines) — P13 recipe.
- `analysis/modern_v2_audit_2026-04-29/` — Track H.1 outputs.
- `analysis/arcface_training_audit_2026-04-29/` — Track H.2 outputs.
- `arena/checkpoint_maps/teams_target_domain.p13_2026-04-30.yaml` — Day-4 ckpt map (9 periodic ckpts).
- `analysis/p13_validation_2026-04-30/` — Day-4 P13 scorecard.

### To modify (additive)

- `trainer/trainer.py` — apply `isinstance(dict)` fix at line 2326 + add anchor-aware loss aggregation.
- `data/augmentations/pipelines.py` — wire `screen_capture_sim_p` flag.
- `loss/__init__.py` (or registry) — register anchor-aware loss.

---

## 9. Risks & Mitigations

1. **Anchor-aware loss term may over-suppress fake recall.** Mitigation: weight = 5.0 starting point; kill switch at step 500 if total loss > 1.5× baseline. Relaunch with weight 2.0 if needed.
2. **Screen-capture aug primitive may introduce NaN like webcam_harden did.** Mitigation: write 10 augmented samples to disk and visually inspect; clip output values to [0, 1]; gradient clip in trainer (already in place).
3. **Periodic_saves bug fix may not be the right fix.** Mitigation: smoke test mandatory; if it doesn't fire on the smoke test, re-investigate before launching. Cost-of-being-wrong: $45 wasted.
4. **AUC-saturated early-stopping fires anyway despite patience=50.** Mitigation: also add `early_stopping_metric: anchor_composite` to yaml (verify trainer supports the override); fallback `early_stopping_enabled: false`.
5. **ArcFace audit reveals training-label corruption.** Mitigation: Track H.4 scratch-retrain proposal; user decides ship-with-caveat vs multi-week.
6. **P13 doesn't break the AUC ceiling.** Likely outcome (~50%). Mitigation: Track I.2 scratch-retrain proposal Day 5.
7. **Image rebuild contains untested code.** Mitigation: smoke tests for trainer + aug primitive before rebuild. Don't combine the bug fix and the new aug into a single launch without separately verifying both.
8. **us-east1 / us-west4 / us-central1 all PEND.** Mitigation: 30-min PENDING → switch regions (per CLAUDE.md). If all three pend, escalate to user.
9. **Day-4 P13 inference takes longer than 30 min.** Mitigation: Day 4 schedule has 6h between expected P13 finish and decision gate.
10. **Modern_v2 audit reveals filter is mis-built.** Mitigation: re-run Option A's analysis on modern_v2_relaxed; update Day-4 P13 scoring substrate; update HANDOFF.md retroactive interpretation of P11/Option-A results.

---

## 10. Open Questions / Dependencies

- **Anchor pool source for the loss term.** Use existing `analysis/teams_pool_rescore.py` cached pool (P8A rescored). Confirm Day 3.2 the pool URL and frame-id schema.
- **Anchor-aware loss frequency.** Compute every-200-steps or every-step? Latter risks instability + memory; former gives a "lagged" gradient signal. Recommendation: every 200 steps, hold gradient for next 200, OR compute as a periodic auxiliary loss summed only every 200 steps. Decide Day 3.2.
- **Visomaster oversample 5.0 vs 4.0.** HEAVY's 3.0 was insufficient; 6.0 broke pair structure (HEAVY_DEEPLIVE precedent). 5.0 is the bet. Consider 4.0 as fallback if P13 step-500 loss is unstable.
- **Screen-capture aug at p=0.30 vs higher.** Start at 0.30 (matches teams_codec_sim_p band); if step-1500 anchor still high, P14 could escalate to 0.50.
- **Ensemble weighting for P8A + P13** — equal noisy-OR or weighted? Equal first (Option A's default); test weighted only if equal underperforms.

---

## 11. Coordinator Protocol

- **LOG file:** `april-26-training-master-plan-v2.LOG.md` — append entry per session.
- **RESULTS file:** `april-26-training-master-plan-v2.RESULTS.md` — append entry per measurement.
- **Plan file (this):** `april-26-training-master-plan-v4.md` at training/ root. Read-only for execution agents. Plan changes go in LOG with `PLAN CHANGE PROPOSAL` block.
- **HANDOFF.md:** refresh end of Day 3 + end of Day 4.
- **Memory updates:** if discoveries generalize beyond this sprint (e.g., anchor-aware loss is a general lever), file under `~/.claude/projects/.../memory/`.
- **Commit cadence:** small + frequent. Trainer bug fix in its own commit (one-liner). Anchor-aware loss + screen-capture aug each in their own commits. P13 yaml in its own commit. Image rebuild after all four are committed.

---

## 12. Resume Instructions (next agent)

1. Read this plan + HANDOFF.md + last LOG entry + last RESULTS entry.
2. Read `MEMORY.md` and the listed memory files (especially `project_promotion_contract.md`, `feedback_no_cancelling_vertex_jobs.md`, `project_p8a_breakthrough.md`, `feedback_sklearn_njobs.md`, `project_lockbox_fpr_dominated_by_webcam_mode.md`).
3. Run `git status` and `git diff --stat HEAD` to confirm tree state.
4. Identify which day we're on; pick next checkbox from §5; execute.
5. **For trainer.py changes:** smoke test before commit. **For Vertex launches:** user authorization required.
6. End session with LOG + RESULTS entry + HANDOFF.md refresh.

---

## 13. Honest Read on Optimism

Putting raw probabilities on the wall:

| Outcome | P |
|---|---|
| 90/5 met by Day 4 (P13 single-handedly closes) | 0.18 |
| 90/5 met by Day 6 (P13 + targeted P14 hard mining) | 0.32 |
| Within 5pp of 90 on at least 2 of 3 families by Day 4 | 0.55 |
| modern_v2 audit reveals filter is over-pruning (changes the verdict) | 0.30 |
| ArcFace audit reveals training-label corruption | 0.40 |
| Substrate ceiling confirmed; ship-with-caveat by Day 5 | 0.55 |
| Multi-week scratch-retrain succeeds (90/5) | ~0.65 |
| The user wants to extend the sprint past Day 6 | 0.40 |

**The most likely path:**

- Day 3: bug fix + recipe redesign + audits land. P13 launches Day 3 evening.
- Day 4: P13 results show recall preserved, anchor improved but not P8A-level; modern_v2 FPR slightly elevated. Best ensemble (P8A + P13) gets within 3–5pp on viso, hits gate on deeplive + teams_fake at modern_v2 FPR ≈ 6–7%. **β verdict.** modern_v2 audit may turn this into α retroactively if the filter is over-pruning.
- Day 5: targeted P14 viso-hard-mining run launched. Multi-week proposal drafted in parallel.
- Day 6: ship/no-ship decision. ~50% it's a ship-with-caveat; ~30% full ship; ~20% retreat to multi-week.

**The most likely positive surprise** has shifted again: the modern_v2 filter audit (Track H.1) is a 30%-likelihood "this changes everything" — if the current modern_v2 over-prunes meaningful real-side variation, then P11_HEAVY's headline FPR was already an underestimate of the true target-domain FPR, AND the existing best-ensemble configurations may already be deployable on a properly-defined modern subset. This is the cheapest, fastest experiment in the plan and runs first thing Day 3.

**The most likely negative surprise** is the anchor-aware loss term not converging cleanly. The codebase has no precedent for anchor-aware training; it's a from-scratch implementation under sprint pressure. Mitigation is small initial weight + kill switches + fallback to weight=2.0.

The recipe-tuning era is fully closed. The data-axis + training-mechanism era is what's left, and it's bounded but not exhausted. Four-to-six days is tight but the plan has clear branch points and credible fallbacks at each.
