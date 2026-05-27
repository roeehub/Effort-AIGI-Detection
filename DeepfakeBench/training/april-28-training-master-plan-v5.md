# Path Forward — 3 Days, ~$1000, Anti-Shortcut Detector

**Author:** Claude Opus 4.7 (1M context), 2026-04-28 13:30 CEST
**Branch:** `teams-relaunch-root-2026-04-17`
**Working dir:** `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/`
**Extends, not replaces:** `april-26-training-master-plan-v4.md` (P13_ANCHOR_AWARE recipe). Plan-v4's spine is sound; this v5 sharpens the anti-shortcut focus, adds pipeline-randomization across BOTH labels, adds face-size canonicalization at the data loader, and adds a clean-eval + shortcut-probe evaluation axis to the Day-4 gate.

---

## Context — Why This Plan, In One Paragraph

The user's framing is broader than the sprint's 90/5 metric: a deployable detector must "actually learn the features we care about and avoid shortcut learning, be robust to different conditions, minimize FP and maximize TP." Three memory files document concrete shortcuts the model has learned (camera/processing-pipeline signature, face pixel-area as fake predictor, webcam-mode FPR concentration). The Option A subagent's verdict (2026-04-28 12:59 CEST: 0/85 ensemble configurations pass; AUC ceiling 0.647 against modern_v2; 7%/5% FPR gates conflict on the existing candidate set) confirms the substrate is shortcut-bounded under the current training distribution. **Adding more training on the same shortcuts will not produce a robust model.** Plan-v4's anchor-aware loss is necessary but not sufficient; this plan adds two structural shortcut-breakers (pipeline-randomization, face-size canonicalization) plus a deployment-honest eval set, keeps Plan-v4's audits, and locks the budget envelope at ~$220 of the ~$1000 available.

---

## 1. Where We Stand (end of Day 2 of original 4-day sprint)

### 1.1 Best two checkpoints, headline metrics

| Candidate | Suite (τ=0.5 raw) | viso recall | deeplive recall | teams_fake_all recall | modern_v2 FPR | dev_real_all FPR | lockbox FPR |
|---|---|---|---|---|---|---|---|
| **P8A reference step5000** (lockbox-stable champion) | dev | **35.6%** | 53.0% | 75.6% | **3.6%** ✓ | 12.2% ✗ | 6.8% |
| **P11_HEAVY step1000** (recall-strong, anchor-broken) | dev | **52.0%** | **77.8%** | **84.7%** | 27.0% ✗ | 15.1% ✗ | 20.7% |

At a calibrated 5%-prod τ on P8A (τ=0.9741):
- P8A: viso 6.5%, deeplive 11.9%, teams_fake_all 53.4%, modern_v2 0.36% ✓, dev_real_all 5.0% ✓
- P11_HEAVY: viso 10.7%, deeplive 40.0%, teams_fake_all 61.7%, modern_v2 1.07%, dev_real_all 6.6% ✓

**Best operating point we have today: P11_HEAVY raw τ≈0.905 → modern_v2 4.98% ✓, but viso 33.4% / deeplive 57.6% / teams_fake_all 71.6% — all below 90%.**

### 1.2 The 4 P11 overnight runs (2026-04-27 night → 2026-04-28 morning)

| run | recipe delta vs P8A | viso τ=0.5 | deeplive τ=0.5 | teams_fake τ=0.5 | modern_v2 τ=0.5 | verdict |
|---|---|---|---|---|---|---|
| MILD | context_var 0.30 + codec_p 0.55 | 44.0% | 71.0% | 81.7% | 18.1% | β/γ — anchor-broken at step 1000 |
| **HEAVY** | context_var 0.50 + real_codec_uplift + codec_p 0.65 | **52.0%** | 77.8% | 84.7% | 27.0% | β/γ — best recall, biggest anchor break |
| HEAVY_DEEPLIVE | HEAVY + deeplive 6× weight | 44.5% | **80.0%** | 83.6% | 26.7% | DROP — viso regressed -7.5pp; HEAVY 3× is right |
| WEBCAM_HARDEN | HEAVY + webcam_harden aug | 23.3% | 55.4% | 75.1% | 6.0% | DROP — NaN at step 5683; aug too aggressive |

### 1.3 P12_HEAVY_LONG anchor trajectory (no ckpts saved due to bug — load-bearing)

| step (eval) | anchor_mean | anchor_composite |
|---|---|---|
| ~500 | 0.821 | — |
| ~1000 | 0.731 | 0.221 |
| ~1500 | 0.851 | 0.119 |
| ~2500 | 0.735 | 0.161 |
| ~6000 | **0.601** | 0.152 |

Recovery rate ~0.04/1000 step. To reach P8A's anchor_mean ≈ 0.10 would need ~17000 more steps. **Pure FT-extension of HEAVY is OUT.**

### 1.4 Option A inference-time stack — 0/85 configs pass

- 30 single-candidate (5 ckpts × 3 calibrations × 2 ops): **0 pass.**
- 45 noisy-OR ensembles (4 combos × 3 calibrations × 5 ops): **0 pass.**
- 10 weighted (P8A + P11_HEAVY) noisy-ORs: **0 pass.**
- Best ensemble overall: P8A + P11_HEAVY isotonic noisy-OR, τ=0.846 → viso 31.3%, deeplive 58.2%, teams_fake_all 75.0%, modern_v2 4.98% ✓, dev_real_all 10.9% ✗.
- **AUC(viso fake vs modern_v2 real) max across all candidates and ensembles = 0.647.** To hit 90% recall at 5% FPR you need AUC ≈ 0.95.
- **The 7%/5% gates conflict**: every operating point ≤ 5% modern_v2 FPR breaches teams_real_all_dev > 7%.

### 1.5 Critical bug — trainer.py:2326 periodic_saves

Uncommitted patch lines 2319-2362. `self.config.get('periodic_saves')` likely returns a wandb.Config wrapper, not a plain dict; `.get('enabled', False)` silently returns False; the entire periodic-saves block is silently skipped. **One-line fix:** apply the `isinstance(dict)` defense pattern from line 392-393. **Untested.** Cost of getting it wrong on next launch: ~$25-60 + 8h. Smoke-test mandatory before any further training.

---

## 2. Hypothesis Retro — What We Tried, What's Still Standing

| # | Hypothesis | Source | Status | Evidence |
|---|---|---|---|---|
| H1 | Recipe-tuning + FT can close 90/5 | Plan v2 | **FALSE** | All P10 / RLP6_04 / codec_hedge variants failed Phase D gates |
| H2 | in_proj-SVD bug fix unlocks features | Plan v3 | **PARTIALLY TRUE** | Fix in commit 2feea58 enables the lever; P11 shows recall lift; alone insufficient |
| H3 | HEAVY recipe lifts recall without breaking anchor | Plan v3 D1 | **PARTIAL** | Recall lifts +16/+25/+9 pp confirmed. Anchor regression also confirmed. Tradeoff is structural, not transient |
| H4 | Pure FT extension restores anchor | Plan v3 D1 | **FALSE** | P12 trajectory: 0.04 anchor_mean recovery / 1000 steps. ~17000 steps needed |
| H5 | Inference-time stack closes the gap | Plan v3 Track A | **FALSE** | Option A: 0/85 configs pass; AUC ceiling at 0.647 |
| H6 | WEBCAM_HARDEN aug improves robustness | Plan v3 D1 hedge | **FAILED** | NaN at step 5683 (GaussNoise var 20-70 + aggressive stack); drop axis |
| H7 | HEAVY_DEEPLIVE 6× weight pushes deeplive recall | Plan v3 D1 hedge | **FAILED** | viso -7.5pp; broke pair structure |
| H8 | modern_v2 filter is correctly defined | Plan v3 D1 evening | **NOT TESTED** | 30% likelihood it over-prunes (Plan v4 H.1) |
| H9 | Training-set identity labels are clean | Plan v3 §1.6 (assumption) | **NOT TESTED** | 40% likelihood corrupted (Plan v4 H.2) |
| H10 | Anchor-aware loss term restores anchor | Plan v4 G | **NOT TESTED** | Implementation Day 3 |

### 2.1 The shortcuts the model has actually learned (memory-confirmed)

| Shortcut | Memory file | Mechanism | Why metric-chasing won't fix it |
|---|---|---|---|
| **Camera/pipeline signature** | `project_signature_shortcut_finding.md` | Same person flips real↔fake under different processing pipelines (dor_shkedi 0.457 vs real_dor 0.038) | The "fake-pipeline" features are the discriminator, not face artifacts. Production captures use the wrong pipeline → flips |
| **Face pixel-area** | `project_face_size_label_leak.md` | deeplive 22-25k px² (single-point), teams_capture varies, reals span wider; Cohen's d ≈ 0.37 | A frame >75k is 1.7× more likely fake; <10k is 12× more likely real. Production crops outside the trained band fail |
| **Webcam-mode FPR concentration** | `project_lockbox_fpr_dominated_by_webcam_mode.md` | clip_capture_mode==webcam → 65.7% FPR vs 8.3% normal, 1.5% phone_screen | Model learned "webcam look = fake". modern_v2 hides this by filtering webcam frames |

### 2.2 Diagnosis

**The model is not detecting deepfake artifacts; it is detecting pipeline-conditional features.** Three converging pieces of evidence:
1. Memory `project_signature_shortcut_finding.md` is causal: same face, different pipeline, opposite prediction.
2. Memory `project_face_size_label_leak.md` is correlative: face size correlates with fake method.
3. Option A's AUC ceiling at 0.647 against modern_v2 (which is 75% one identity) is consistent with a model whose features don't generalize beyond its training-pipeline distribution.

**Implication:** even if P13_ANCHOR_AWARE hits 90/5 on the existing benchmark, it could be exploiting a (different) shortcut. The Plan v4 anchor-aware loss term is a necessary but insufficient mechanism — it pressures FPR down but doesn't break the underlying pipeline-signature discrimination.

---

## 3. Path Forward — 3 Days, ~$1000

### 3.1 Strategic principles

1. **Treat the disease, not the symptom.** Each known shortcut gets a structural intervention.
2. **Validate against a clean substrate.** modern_v2 (281 frames, 75% one identity) is necessary but insufficient. Build a 30-50-frame-per-identity ArcFace-clustered eval set with diverse capture conditions.
3. **Pipeline-randomization aug applied to BOTH real and fake.** Plan v4's screen_capture_sim on real-only does not break the pipeline-label correlation; it just shifts where the shortcut lands. The aug must be symmetric across labels.
4. **Face-size canonicalization must be scale-jitter, not fixed resize.** Per memory `project_face_size_label_leak.md`: random scale-crop aug with wide range (0.7-1.3 of nominal) symmetric across labels is the direct counter.
5. **Budget envelope ~$220 committed of $1000.** Reserve ~$780 for contingency (region switches, P14 if β verdict, debug iterations).
6. **Bug fix is critical path.** Smoke test mandatory before launch. Cost of getting it wrong: $25-60 per attempt.

### 3.2 The candidate run: P13_ANTI_SHORTCUT

Inherits Plan-v4 P13_ANCHOR_AWARE recipe (anchor-aware loss + viso-4× + working periodic_saves + early_stopping_patience 50) and extends with shortcut-breakers:

| Lever | Magnitude | Targets shortcut | Source |
|---|---|---|---|
| **Anchor-aware loss term** | weight=5.0, target_mean_prob=0.10 | anchor regression (FPR side) | Plan v4 G — implement `loss/anchor_aware_penalty.py` |
| **Pipeline-randomization aug stack — symmetric across labels** | p=0.55 real, p=0.45 fake | camera/pipeline signature | NEW. JPEG roundtrip [40, 95] + downscale-upscale [0.85, 1.0] + chroma blur (existing primitive in `data/augmentations/teams_simulation.py`) + RGB→YUV→RGB roundtrip + mild gamma jitter [0.92, 1.08] |
| **Face scale-jitter at data loader** | scale ∈ [0.75, 1.25] of nominal, symmetric across labels | face pixel-area | NEW. Inject before the hardcoded 224×224 resize in `data/batching/df40_paired.py:362` (collate fn) and `data/sources/combined_paired.py` resize |
| **Visomaster oversample** | 4.0 (HEAVY had 3.0; HEAVY_DEEPLIVE's 6.0 broke pair structure) | viso recall long pole | Plan v4 G — `dataloader.method_oversample.visomaster_enhanced: 4.0` |
| **Working periodic_saves** | step_list [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000] | step-1000 anchor-worst-moment artifact | Plan v4 G — apply `isinstance(dict)` fix at trainer.py:2326 |
| **Early-stopping patience** | 50 (was implicit 10, AUC saturates step 1000) | spurious truncation at step 6000 | Plan v4 G |

**Augmentations explicitly NOT used (lessons from H6/H7):**
- MotionBlur > 5px (was 3-5, OK; do not increase)
- ImageCompression quality < 30 (webcam_harden's 18-50 floor was too low)
- GaussNoise var > 30 (webcam_harden's 20-70 caused NaN)
- HueSaturationValue compounded with all of the above (webcam_harden stacked all four — too aggressive)

### 3.3 Day-by-day execution

#### Day 3 (2026-04-29) — setup, audit, launch

**Morning block (~5h):**
- [ ] **3.1 Bug fix + smoke test** (~2h, BLOCKING). Apply isinstance(dict) at trainer.py:2326-2329; add debug log inside the gate. Local CPU smoke: load `R13_P12_HEAVY_LONG.yaml`, instantiate trainer (no GPU), simulate `_run_validation` call at `step_cnt` ∈ step_list. Confirm "periodic_save triggered at step=..." log fires. Commit only after smoke passes.
- [ ] **3.2 Implement anchor-aware loss term** (~3h). New file `loss/anchor_aware_penalty.py` (~80 lines). Wire at trainer.py:1485 (loss aggregation injection point per Explore audit) using the existing `compute_stability_loss` pattern. Verify W&B logs `train/loss/anchor_aware`. Local smoke on dummy anchor pool.

**Midday block (~5h):**
- [ ] **3.3 Implement pipeline-randomization aug stack** (~3h). New file `data/augmentations/pipeline_randomization.py` (~150 lines). Reuses existing `_apply_jpeg_roundtrip` and `_apply_chroma_blur` primitives in `data/augmentations/teams_simulation.py`. Adds: RGB→YUV→RGB roundtrip, scale-jitter [0.85, 1.0] downscale-upscale, gamma jitter [0.92, 1.08]. **Critical guard: clip output to [0, 1]; defensive NaN check after each step.** Wire under flag `pipeline_random_p_real` and `pipeline_random_p_fake` (separately controlled, label-aware) in `data/augmentations/pipelines.py`. Visual smoke: write 20 augmented samples (10 real, 10 fake) to `analysis/pipeline_random_smoke_2026-04-29/` and inspect manually.
- [ ] **3.4 Face scale-jitter at data loader** (~1h). Inject `RandomScale(scale_limit=0.25)` (i.e., 0.75-1.25 of nominal face crop) BEFORE the hardcoded 224×224 resize at `data/batching/df40_paired.py:362` and the equivalent in `data/sources/combined_paired.py`. Symmetric across labels (no label-conditioning). Smoke test: dump 10 examples and confirm visible scale variation.
- [ ] **3.5 Track H.1 modern_v2 filter audit** (~1.5h, parallel). Inspect `analysis/lockbox_tagging/lockbox_tags_2026-04-27.parquet` schema. Re-derive modern_v2_relaxed (face crop ≥ 64² vs current 96², sharpness band 90th pct vs 50-95th). Compare modern_v2 vs modern_v2_relaxed P8A FPR. If Δ ≥ 1.5pp → switch to relaxed for Day-4 scoring.

**Afternoon block (~3h):**
- [ ] **3.6 Build clean-eval set + shortcut-probe set** (~2h). NEW. Source: `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` (n=7334).
  - **clean_eval_v1** (~30-50 frames per identity, balanced across modern Teams capture conditions, NOT dor_shkedi-skewed). Target ~12-15 identities × ~3-4 frames each = ~50 frames.
  - **shortcut_probe_v1**: paired same-face-different-pipeline frames. Source pairs: `(dor_shkedi, real_dor)`, `(teams_capture_cam_test_s33 fake, Cam_Test__s33 real)`, etc. Target ~20 pairs.
  - Save URI lists to `analysis/clean_eval_2026-04-29/`.
- [ ] **3.7 Author P13_ANTI_SHORTCUT yaml** (~30 min). New file `experiments/phase2_round13/R13_P13_ANTI_SHORTCUT.yaml` (~280 lines). Base on `R13_P12_HEAVY_LONG.yaml`; layer in anchor-aware loss + pipeline-random + face scale-jitter + viso 4×.
- [ ] **3.8 Image rebuild** (~5 min build + ~3 min push) → expected 1.3.224.

**Evening block (~1h):**
- [ ] **3.9 User authorization gate.** Discuss recipe with user before launch. ~$60 commitment.
- [ ] **3.10 Launch P13_ANTI_SHORTCUT** us-east1, switch us-west4 if PENDING > 30 min. Confirm RUNNING within 30 min.
- [ ] **3.11 Track H.2 ArcFace audit** (~3h, parallel to GPU run). Sample 30 training-set identities; cluster face embeddings within each label; compute within-identity purity. If avg purity < 0.85 → label corruption confirmed → drop contrastive_regularization in P14.
- [ ] **3.12 LOG + RESULTS update.**

**Day 3 spend:** ~$60 P13 training + ~$5 audits + ~$1 image rebuild = **~$66**.

#### Day 4 (2026-04-30) — score, decide

- [ ] **4.1 P13 finishes** (expected 09:00-12:00 CEST). Pull all 9 periodic ckpts.
- [ ] **4.2 Score P13 on full validation suite** + clean_eval_v1 + shortcut_probe_v1. ~$10 in inference jobs.
- [ ] **4.3 Ensemble P13 with P8A.** Re-run Option A's scorer with P13 best-anchor ckpt added.
- [ ] **4.4 Per-method τ + calibration on P13 best ckpt.**
- [ ] **4.5 Day-4 triple-axis decision** (Day 4 evening, ~17:00 CEST). Best ckpt = best balance across:
  - **Axis 1 (90/5 metric, existing benchmark):** viso ≥ 90 AND deeplive ≥ 90 AND teams_fake_all ≥ 90 AND modern_v2 FPR ≤ 5 AND teams_real_all_dev FPR ≤ 7
  - **Axis 2 (clean_eval, deployment-honest):** recall ≥ 80 on the new clean_eval_v1
  - **Axis 3 (shortcut-probe):** max-min Δprob across same-face-different-pipeline pairs ≤ 0.15

  Verdict gates:
  - **(α) PASS** all three axes → ship Day 5
  - **(β) PARTIAL** at least one axis falls 5-15pp short → Day 5: targeted P14 (e.g., viso 5× + harder pipeline aug + same anchor-aware loss)
  - **(γ) FAIL** all three axes ≥ 15pp short OR shortcut-probe > 0.30 → Day 5: write multi-week scratch-retrain proposal + ship-with-caveat preparation
- [ ] **4.6 LOG + RESULTS + HANDOFF.md refresh.**

**Day 4 spend:** ~$10. **Cumulative through Day 4:** ~$76.

#### Day 5 (2026-05-01, extension Day 1) — branch by verdict

- [ ] **5.α Ship validation.** Build deployment scorecard with all three axes. Document operating point + per-pipeline FPR + per-identity FPR + caveat-ready disclaimers. ~$5 in scoring.
- [ ] **5.β Launch P14_HARDER** (~$60). Tweaks: pipeline-aug p=0.75/0.65 (was 0.55/0.45), viso 5× (was 4×), keep anchor-aware loss. Single targeted run.
- [ ] **5.γ Track I.2 multi-week scratch-retrain proposal.** Deliverable: data hygiene plan, recipe outline, time + cost + success-probability bounds. User decides ship-with-caveat vs commit-to-multi-week.

**Day 5 max spend (β path):** ~$60-75.

**Cumulative through Day 5:** **~$140-150 of $1000.** Buffer ~$850.

### 3.4 Budget envelope summary

| Item | Cost |
|---|---|
| P13_ANTI_SHORTCUT training (Day 3 night) | ~$60 |
| P13 inference + scoring (Day 4) | ~$10 |
| Image rebuilds (~3-4 ×) | ~$3 |
| Audits / probe / clean eval (Day 3) | ~$5 |
| (β path only) P14_HARDER training (Day 5) | ~$60 |
| (β path only) P14 inference (Day 6) | ~$15 |
| Buffer for region switches / debug iterations | ~$50 |
| **Subtotal committed** | **~$203** |
| **Reserve** | **~$797** |

Reserve is intentionally large — it buys one full additional 9h training run on Day 6 (~$60) plus debug iterations if anti-shortcut interventions need re-tuning. Total committed across the 3-day window stays well under 25% of budget.

### 3.5 Anti-shortcut mechanism table

| Shortcut (memory) | Plan v4 mechanism | Plan v5 (this) addition | Why v5 is stronger |
|---|---|---|---|
| Camera/pipeline signature | (none direct) | Pipeline-randomization on BOTH labels (p=0.55 real, 0.45 fake) | Forces model to find features that survive pipeline jitter |
| Face pixel-area | (none) | Scale-jitter [0.75, 1.25] symmetric across labels | Direct counter recommended in `project_face_size_label_leak.md` |
| Webcam-mode FPR | screen_capture_sim on real-only | Pipeline aug on both labels + clean_eval set | Symmetric label aug breaks the correlation; clean_eval validates beyond webcam-filter trick |
| Anchor regression | anchor-aware loss | Same (weight 5, target 0.10) | Inherited |
| Identity-label corruption (suspected) | ArcFace audit Day 3 | Same | Inherited |
| Saturating AUC early-stop | patience=50 | Same | Inherited |
| Step-1000 anchor-worst artifact | periodic_saves | Same (with isinstance(dict) fix smoke-tested) | Inherited |

### 3.6 Deltas from Plan v4 (additive only)

1. **Pipeline-randomization aug applied to BOTH real and fake** (Plan v4: real-only screen_capture_sim, narrower scope).
2. **Face scale-jitter [0.75, 1.25] at data loader** (Plan v4: implicit via existing crop pipeline; the existing pipeline does NOT canonicalize before resize per Explore audit).
3. **clean_eval_v1 construction** (Plan v4 H.1: only audits existing modern_v2, doesn't build a new substrate).
4. **shortcut_probe_v1 evaluation** (Plan v4: not present).
5. **Triple-axis Day-4 decision** (Plan v4: single-axis 90/5).
6. **Explicit budget envelope** (Plan v4: $45 for P13 only; this plan: $203 committed of $1000).
7. **Visomaster oversample 4×** (Plan v4: 5× — split-the-difference between HEAVY's failed 3× and HEAVY_DEEPLIVE's failed 6×).
8. **Pipeline-aug NaN guards** (clip [0,1], defensive NaN check after each step) given webcam_harden precedent.

---

## 4. Verification — How We Know We're Done

### Day-3 exit (P13 RUNNING)
- Bug fix smoke-tested locally — log "periodic_save triggered at step=..." fires.
- Anchor-aware loss + pipeline-random + face scale-jitter all have visual / unit smokes passing.
- modern_v2 audit decision documented (keep or relaxed).
- ArcFace identity-purity audit underway or complete.
- clean_eval_v1 + shortcut_probe_v1 frame URI lists committed.

### Day-4 exit (decision made)
- P13 9-ckpt scorecard against all 8 dev/lockbox suites + clean_eval_v1 + shortcut_probe_v1.
- Best ckpt identified per triple-axis decision.
- Verdict α/β/γ written into LOG + RESULTS + HANDOFF.md.
- Image rebuild + P14 yaml stand-by IF β.
- Multi-week proposal stand-by IF γ.

### Day-5 exit (sprint terminates or extension begins)
- α: ship scorecard + deployment caveats + per-pipeline FPR table + sign-off.
- β: P14 launched + ship-prep continuing in parallel.
- γ: multi-week proposal in user's hands; ship-with-caveat substrate documented.

### Robustness criteria (a "good model" beyond 90/5)
- shortcut_probe_v1 max-min Δprob ≤ 0.15 (same face same prediction across pipelines).
- clean_eval_v1 recall ≥ 80% (deployment-domain-honest).
- Per-pipeline FPR breakdown reported (no single pipeline driving headline).
- Per-identity FPR no single identity > 2× the median.

---

## 5. Risks & Mitigations (delta from Plan v4 §9)

R1. **Pipeline-randomization aug on fake samples weakens fake recall.** Mitigation: lower p on fake (0.45) than real (0.55). Kill switch step 500 if total loss > 1.5× baseline OR teams_fake recall < 0.70.

R2. **Face scale-jitter could hide real deepfake patterns.** Mitigation: visual inspection of 20 augmented examples (10 real, 10 fake) before image rebuild. Verify deepfake "tells" (eye boundary, hairline) survive at scale 0.75 and 1.25.

R3. **Pipeline-aug NaN like webcam_harden.** Mitigation: defensive `np.nan_to_num` + `np.clip(x, 0, 1)` after each sub-aug; NaN check before yielding from collate. Caps: JPEG quality ≥ 40 (webcam_harden was 18), no aggressive GaussNoise (was 20-70 var).

R4. **Building clean_eval_v1 is time-bounded.** If we can't get 12-15 identities × 3-4 frames by end of Day 3.6 (~2h budget), fall back to existing modern_v2 + dor_dev as eval target. clean_eval_v1 is upgrade, not blocker.

R5. **shortcut_probe_v1 pairs are sparse.** dor_shkedi vs real_dor is one obvious pair. Pair construction may yield only 5-10 high-confidence same-face-different-pipeline pairs. Acceptable — this is a directional signal, not a primary metric.

R6. **isinstance(dict) fix is wrong root cause.** Mitigation: smoke test will catch this. If smoke test fails, inspect `type(self.config.get('periodic_saves'))` directly and adjust. Don't rebuild image until smoke passes.

R7. **Pipeline aug + face scale-jitter compounding NaN risk.** Mitigation: enable each independently in 200-step micro-runs locally before full launch; verify loss curves are stable.

R8. **β verdict but P14 also fails.** Mitigation: γ-pivot still available Day 6 (multi-week proposal + ship-with-caveat).

---

## 6. Open Questions for User (decision points)

These are explicit decision points before Day 3 starts. The plan executes one way or another based on user input.

Q1. **Pipeline-randomization aug applied to BOTH real and fake — confirm scope.** Plan v4 only applied screen_capture_sim to real samples. v5 makes the aug symmetric across labels. The argument: if the aug is real-only, the model learns "real has pipeline-random artifacts → label real" — that's still a pipeline shortcut, just inverted. Symmetric application is the load-bearing anti-shortcut intervention. **OK to proceed?**

Q2. **Visomaster oversample 4× vs Plan v4's 5×.** HEAVY's 3× was insufficient; HEAVY_DEEPLIVE's 6× broke pair structure (-7.5pp viso). 4× is mid-bet (closer to HEAVY); 5× is slightly more aggressive. **User pick?**

Q3. **Day-4 triple-axis α gate is strictly harder than 90/5 alone** (90/5 + clean_eval ≥ 80 + shortcut_probe ≤ 0.15). This means a result that hits 90/5 but fails clean_eval or shortcut_probe is β, not α. Goal is anti-shortcut robustness, not benchmark-passing alone. **Confirm this α gate?**

Q4. **Reserve allocation $797 of $1000.** This buys one extra training run + debug iterations on Day 6 if extension is taken. **Comfortable with this conservative split, or prefer to commit more upfront (e.g., launch P14 in parallel with P13 to hedge)?**

---

## 7. Critical Files (delta from Plan v4 §8)

### Existing (use as-is)
- `data/augmentations/teams_simulation.py` — has `_apply_jpeg_roundtrip`, `_apply_chroma_blur` primitives. Reuse, don't reinvent.
- `data/batching/df40_paired.py:362` — collate fn `cv2.resize(img, (224, 224))` is the post-aug resize. Inject scale-jitter BEFORE this line.
- `data/sources/combined_paired.py` — has equivalent resize call. Same pattern.
- `trainer/trainer.py:1485` — main loss aggregation point (Explore audit). Inject anchor-aware loss aggregation here, mirroring `stability_loss` pattern at line 1495.
- `trainer/trainer.py:392-393` — isinstance(dict) defense pattern. Apply to line 2326.
- `analysis/lockbox_tagging/lockbox_tags_2026-04-27.parquet` — 7334-row tag table for clean_eval_v1 + shortcut_probe_v1 construction.
- `analysis/option_a_ensemble_2026-04-28/run_option_a_analysis.py` — adapt for Day-4 P13 + ensemble re-scoring.

### To create (Day 3)
- `loss/anchor_aware_penalty.py` (~80 lines).
- `data/augmentations/pipeline_randomization.py` (~150 lines).
- `experiments/phase2_round13/R13_P13_ANTI_SHORTCUT.yaml` (~280 lines).
- `analysis/clean_eval_2026-04-29/clean_eval_v1_frames.yaml` (~50 frame URIs).
- `analysis/clean_eval_2026-04-29/shortcut_probe_v1_pairs.yaml` (~10-20 pairs).
- `analysis/modern_v2_audit_2026-04-29/` — Track H.1 outputs.
- `analysis/arcface_training_audit_2026-04-29/` — Track H.2 outputs.
- `arena/checkpoint_maps/teams_target_domain.p13_2026-04-30.yaml` — Day-4 ckpt map (9 periodic ckpts).

### To modify (additive only)
- `trainer/trainer.py` — isinstance(dict) fix at 2326-2329 + anchor-aware loss aggregation near line 1485.
- `data/augmentations/pipelines.py` — wire `pipeline_random_p_real` + `pipeline_random_p_fake` flags + `face_scale_jitter` flag.
- `data/batching/df40_paired.py` — inject scale-jitter before line 362 resize.
- `data/sources/combined_paired.py` — same scale-jitter injection.
- `loss/__init__.py` (or registry) — register anchor-aware loss.

---

## 8. Resume Instructions (next agent, when this plan is approved)

1. Read this plan + `april-26-training-master-plan-v4.md` + last LOG entry + last RESULTS entry.
2. Read MEMORY.md and these key files: `project_signature_shortcut_finding.md`, `project_face_size_label_leak.md`, `project_lockbox_fpr_dominated_by_webcam_mode.md`, `feedback_no_cancelling_vertex_jobs.md`, `feedback_sklearn_njobs.md`, `project_in_proj_svd_gradient_bug.md`.
3. Run `git status` + `git diff --stat HEAD` for state-drift check.
4. Pick next checkbox from §3.3 Day 3 — start with 3.1 (bug fix + smoke test, BLOCKING).
5. **For trainer.py changes:** smoke test before commit.
6. **For Vertex launches:** user authorization required (per `feedback_no_cancelling_vertex_jobs.md`).
7. End session with LOG + RESULTS + HANDOFF.md refresh.

---

## 9. Honest Read on Probability of Success

| Outcome | P (Plan v3) | P (Plan v4) | P (Plan v5, this) |
|---|---|---|---|
| 90/5 met by Day 4 (P13 single-handedly) | 0.55 | 0.18 | **0.20** |
| 90/5 + triple-axis α by Day 4 | n/a | n/a | **0.10** |
| 90/5 met by Day 6 (P13 + P14) | 0.78 | 0.32 | **0.32** |
| Within 5pp on at least 2 of 3 families by Day 4 | 0.85 | 0.55 | **0.55** |
| modern_v2 audit reveals over-pruning | n/a | 0.30 | 0.30 |
| ArcFace audit reveals label corruption | n/a | 0.40 | 0.40 |
| Substrate ceiling confirmed → ship-with-caveat | n/a | 0.55 | **0.55** |
| Multi-week scratch retrain succeeds (90/5) | n/a | ~0.65 | ~0.65 |
| **shortcut-probe ≤ 0.15 on best ckpt** | n/a | n/a | **0.40** |

The triple-axis α has lower probability than Plan v4's single-axis 90/5 (0.10 vs 0.18) by design — it's a stricter gate. Trade: lower probability of "ship by Day 4" in exchange for higher confidence that a Day-4 ship is genuinely robust rather than another shortcut-passing artifact. **The point of v5 is not to maximize probability of meeting the metric; it's to reduce the probability of shipping a brittle model that meets the metric but fails in deployment.**

Most likely path: P13 lifts recall ~5pp on each family vs P11_HEAVY (anchor-aware loss preserves more recall than naive FT-extension); modern_v2 FPR settles at ~6-8% (close-but-not-clearing); shortcut-probe shows partial robustness (Δprob 0.18-0.25). β verdict on Day 4. P14 launched Day 5 with harder pipeline-aug. Day 6 is ship-with-caveat or extension to Week 2.
