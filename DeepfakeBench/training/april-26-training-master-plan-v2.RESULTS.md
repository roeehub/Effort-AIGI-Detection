# Experiment Results — april-26-training-master-plan-v2

Append-only log of measurement outputs tied to this master plan. Code changes go in the LOG, not here.

## How to use this file

- **One entry per measurement** (scorecard, probe, rescore, etc.).
- **Always link to the source artifact** (JSON/CSV/summary path) so the result is reproducible.
- **Compare to the §4 Phase D gate** from the master plan when applicable.
- **Never delete entries.** If a result is corrected or superseded, append a new entry referencing the old one.

## Entry template

````
### YYYY-MM-DD — <checkpoint or experiment label>
**Source artifact(s):** <relative path(s)>
**Phase/step:** <e.g., A.2 promotion-contract scorecard>
**Headline numbers:**
- <number 1>
- <number 2>
**Gate results (vs §4 Phase D):**
- <gate>: <PASS / FAIL / N/A>
**Notes:**
- <contract policy bug fired? selected_threshold value? sample size? known caveats?>
**Next implication:**
- <what this means for plan progression>
````

---

## Results

### 2026-04-26 — P10_SYM_on_P8A step 2500 (pool rescore)
**Source artifact:** `analysis/pool_rescore_p10_sym_on_p8a_step2500.summary.json`
**Phase/step:** Pre-Phase-A (the provocation cited in plan §1 and §2)
**Headline numbers:**
- `anchor_mean = 0.6051`
- `frac_gt_0.9 = 0.133`
- `max_real_correct_mean = 0.0290`
**Gate results (vs §4 Phase D — anchor-pool subset only):**
- `anchor_mean ≤ 0.70` — **PASS** (0.6051)
- `frac_gt_0.9 ≤ 0.30` — **PASS** (0.133)
- `max_real_correct_mean ≤ 0.04` — **PASS** (0.0290)
- Lockbox real_fpr / fake_recall — NOT YET MEASURED
- visomaster_enhanced_macro recall, deeplive_enhanced recall — NOT YET MEASURED
- Source / camera / identity probes — NOT YET MEASURED
- Contract τ (`selected_threshold`) — NOT YET MEASURED
**Notes:**
- Trained under the broken-in_proj-SVD code path (zero CLS gradient on q/k/v residuals). See plan §1 and memory `project_in_proj_svd_gradient_bug.md`.
- This result is what triggered the M-vs-V decision in plan §3.
**Next implication:**
- Phase A.1 should check for later checkpoints of the same run; if step-N > 2500 exists, prefer it.
- Phase A.2 contract scorecard + Phase A.4 source-bucket probe still required before this checkpoint can be considered for crowning under either path.

---

### 2026-04-26 — A.2 v3 promotion-contract scorecard (recovery, image 1.3.219)
**Source artifact(s):**
- `gs://training-job-outputs/test_results/teams_promotion_contract/phase-a2-p10sym-on-p8a-step2500-v3-20260427/promotion_contract/checkpoint_summary.csv`
- `gs://training-job-outputs/test_results/teams_promotion_contract/phase-a2-p10sym-on-p8a-step2500-v3-20260427/promotion_contract/promotion_winner.json`
- `gs://training-job-outputs/test_results/teams_promotion_contract/phase-a2-p10sym-on-p8a-step2500-v3-20260427/diagnostic_scorecard/scorecard.wide.csv`
**Phase/step:** A.2 promotion-contract scorecard (post-fix re-run after A.2 v2 scorer crash)
**Headline numbers (calibrated contract — τ chosen by FPR-budget policy):**

| Checkpoint | rank | selected_threshold | lockbox_real_fpr | lockbox_fake_recall |
|---|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 (winner) | 1 | 0.99095 | 0.00147 | 0.2372 |
| RLP6_04_BASELINE_STEP23500 | 2 | 0.99191 | 0.00441 | 0.2332 |
| P10_SYM_ON_P8A_STEP2500 | 3 | 0.98475 | 0.00441 | 0.2095 |

**Headline numbers (diagnostic τ=0.5 — the trustworthy readout):**

| Checkpoint | dev_real_fpr | dev_real_poor_quality_fpr | dev_real_lighting_extreme_fpr | teams_fake_all_dev recall | visomaster_enhanced_macro recall | deeplive_enhanced recall | lockbox_real_fpr | lockbox_fake_recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 0.121 | 0.081 | 0.111 | 0.702 | 0.356 | 0.530 | 0.062 | 0.652 |
| P10_SYM_ON_P8A_STEP2500 | 0.152 | 0.130 | 0.135 | 0.751 | 0.396 | 0.653 | 0.104 | 0.593 |
| RLP6_04_BASELINE_STEP23500 | 0.159 | 0.133 | 0.158 | 0.839 | 0.573 | 0.796 | 0.393 | 0.842 |

**Gate results (vs §4 Phase D):**
- Calibrated contract τ ≈ 0.99 across all 3 → **contract policy bug fired** (memory `project_contract_policy_bug.md`). Calibrated lockbox_fake_recall numbers are NOT promotion-grade; defer to τ=0.5 readout.
- P8A reference @ τ=0.5: lockbox_real_fpr 6.2 %, lockbox_fake_recall 65.2 % — closest to gate.
- P10_SYM_ON_P8A step 2500 @ τ=0.5: dominated by P8A reference on **both** axes (FPR 10.4 % vs 6.2 %, recall 59.3 % vs 65.2 %). §C.1 hypothesis (sym+GRL improves on P8A) **NOT supported at this checkpoint**.
- RLP6_04 baseline: highest fake recall (84.2 %) but unacceptable real FPR (39.3 %) — confirms camera-signature shortcut.

**Notes:**
- Recovery from A.2 v2 scorer crash (404 on `teams_real_dor_dev`). Fix in commit `ed6f53b` adds `_path_exists` pre-flight + drops missing readout-only suites; same numbers as local recovery run at `/tmp/a2_v2_local_recovery/`.
- Image 1.3.219 (Cloud Build `7e30c915-748e-4ca5-8f30-dfec60816ee7`).
- Vertex job `889157804793790464` in us-east1 (started 22:21 UTC, artifacts landed 23:46 UTC).

**Next implication:**
- P10_SYM_on_P8A at step 2500 is **not the recipe**: P8A reference dominates it. Either (a) crown P8A reference itself, or (b) hunt for a better P10_SYM checkpoint / variant.
- The codec_hedge slate run (C.3) is the most promising P10_SYM variant — its anchor/composite is +30 % vs canonical (see slate synthesis below); deserves an A.2-style validation pass to get lockbox numbers.

---

### 2026-04-26 — Phase C overnight slate (5-corner read, training-time metrics)
**Source artifact(s):**
- W&B project `dtect-vision/enhanced-aug-test`, runs:
  - C.1 canonical: `R13_P10_SYM_on_P8A_0426-1940` (`6tzvre0k`)
  - C-ablation: `R13_P10_SYM_on_P8A_ablation_no_in_proj_0426-2121`
  - C.3 codec hedge: `R13_P10_SYM_on_P8A_codec_hedge_0426-2050` (`bm7xxwqo`)
  - C.4a one-step-back: `R13_P10_SYM_on_RLP6_04_step23500_0426-2121`
  - C.4b two-effects-back: `R13_P10_SYM_on_RLP6_04_step8000_0426-2121`
**Phase/step:** §4 Phase C — 5-corner read of slate completion (W&B summaries; lockbox not yet measured for these new checkpoints)
**Headline numbers:**

| Run | best_anchor/composite (step) | best_value_composite (step) | best_ood_composite (step) | heldout_ood AUC (by_value) | anchor_mean (last) | max_correct_real_mean |
|---|---:|---:|---:|---:|---:|---:|
| C.1 canonical (P8A base, fixed in_proj) | **0.2180** (s=2500) | 0.9181 (s=2000) | 0.9861 (s=1500) | 0.9618 | 0.682 | 0.137 |
| C-ablation (P8A base, in_proj-SVD off) | 0.2187 (s=6000) | 0.9186 (s=6500) | 0.9855 (s=2000) | 0.8609 | 0.612 | 0.283 |
| **C.3 codec hedge** (P8A base, codec aug) | **0.2842** (s=3000) | 0.9234 (s=2500) | 0.9858 (s=2500) | 0.9660 | 0.477 | 0.272 |
| C.4a one-step-back (RLP6_04 step 23500 base) | 0.1725 (s=5500) | 0.9140 (s=5000) | 0.9871 (s=3000) | 0.9834 | 0.857 | 0.082 |
| C.4b two-effects-back (RLP6_04 step 8000 base) | 0.1732 (s=6000) | 0.9078 (s=4000) | 0.9888 (s=2500) | 0.9842 | 0.872 | 0.047 |

For reference, A.2 v2/v3 lockbox numbers @ τ=0.5 on P10_SYM step 2500 (broken-bug baseline): lockbox real FPR 10.4 %, lockbox fake recall 59.3 %.

**Gate results (vs §4 Phase D — slate is training-time only; lockbox gates not directly measurable here):**
- Mid-run kill switches (plan §C.2):
  - Step 1500 `anchor_mean > 0.90` — N/A (no run hit this; no kills issued).
  - Step 3000 `anchor_mean > 0.85` — both RLP6_04-based runs (C.4a 0.857, C.4b 0.872) hit it; not killed (plan-v2 PCP #3a left §C.4 outside §C.2 mid-run kill scope).
  - Step 5000 `anchor_mean ≤ 0.55` AND recall preserved — C.3 codec hedge (anchor_mean 0.477) is closest to satisfying this signature.

**5-corner synthesis:**

1. **C.1 vs C-ablation (in_proj-SVD bug fix attribution)**: virtually identical (best_anchor/composite 0.218 vs 0.219; best_value 0.918 vs 0.919). The freshly-ungated in_proj-SVD residual lever (commit `2feea58`, 2026-04-26) is **not pulling weight in this configuration**. Heldout_ood by_value diverges (0.962 vs 0.861), but ood-composite path (the actual selection metric) is tied. → bug fix unlocked nothing material; either lever capacity is too small at this rank or regularization needs retuning to leverage it.

2. **C.3 vs C.1 (codec axis)**: C.3 wins decisively on anchor (+0.066 best, +0.070 last-step) with similar value/ood composite and slightly better heldout_ood AUC (0.966 vs 0.962). Trades anchor_mean down (0.682 → 0.477) for max_correct_real lift (0.137 → 0.272) — flatter, more balanced anchor profile, not just "better". → **codec augmentation is the slate's clear positive direction**.

3. **C.4a vs C.1 (P8A recipe necessity)**: C.4a's anchor composite (0.173) is materially worse than C.1's (0.218) despite RLP6_04 base having more pretrain steps. Heldout_ood AUC is higher (0.983 vs 0.962) but mostly because RLP6_04's high anchor_mean (0.857) inherits the camera-signature shortcut and makes everything look "well-separated". → P8A is a strictly better base than RLP6_04 for this recipe.

4. **C.4b vs C.4a (§C-γ crystallization trigger)**: essentially **a tie within noise** — best_anchor/composite 0.1732 vs 0.1725 (Δ=0.0007); marginal trades on every other metric. Both anchor_mean ≈ 0.86–0.87 (the RLP6_04 ceiling). → §C-γ trigger condition (C.4b > C.4a anchor) **NOT met**. Plain-CLIP-scratch (~48 h, ~$$$) is not warranted by this evidence.

5. **Cross-recipe ranking on the only training-time metric that tracks the deployment goal (best_anchor/composite, P8A base only)**: **C.3 (0.284) > C-ablation (0.219) ≈ C.1 (0.218) >> C.4a (0.173) ≈ C.4b (0.173)**.

**Notes:**
- All 5 runs trained on image 1.3.218 (C.1) or 1.3.218 (others); all post-`2feea58` so in_proj-SVD residuals receive proper CLS gradient.
- These are training-time anchor/composite metrics, NOT post-hoc lockbox numbers. Direct comparison to A.2 lockbox readouts requires a follow-up A.2-style validation pass on the new checkpoints.
- The "lockbox-equivalent" anchor `anchor_mean` ceiling of ~0.86 for any RLP6_04-based FT independently confirms memory `project_shortcut_is_upstream.md` (camera-signature shortcut lives upstream of RLP6_04).

**Next implication:**
- **§C-γ plain-CLIP-scratch trigger: NOT FIRING.** C.4b is not materially better than C.4a; the substrate-vs-crystallization question doesn't have a clean answer from this slate, but the smaller-reward + larger-cost combination kills the autonomous launch case.
- **Promote C.3 codec_hedge to A.2-style validation** as the highest-priority follow-up. Suggested checkpoint map:
  - C.3 best_value_composite: `gs://training-job-outputs/phase2r13_experiments/bm7xxwqo/value_composite_effort_20260426_step2500_auc0.9897_eer0.0372.pth`
  - C.1 best_value_composite (for fixed-bug baseline): `gs://training-job-outputs/phase2r13_experiments/6tzvre0k/value_composite_effort_20260426_step2000_auc0.9877_eer0.0473.pth`
  - P8A reference for reuse: `gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth`
  - Requires new yaml + image rebuild (image-currency check is active per PCP #2). ~30 min build + ~1.5 h validation. Not launched autonomously this session — surfaced for user confirmation.
- **Drop the in_proj-SVD lever experiments** as a near-term priority — bug fix produced no material lift (C.1 vs C-ablation ≈ tie). Revisit only when a different recipe tries to actually use the unlocked residual capacity.

---

### 2026-04-27 — Codec_hedge A.2-style validation (image 1.3.220)
**Source artifact(s):**
- `gs://training-job-outputs/test_results/teams_promotion_contract/codec-hedge-validation-20260427/promotion_contract/checkpoint_summary.csv`
- `gs://training-job-outputs/test_results/teams_promotion_contract/codec-hedge-validation-20260427/promotion_contract/promotion_winner.json`
- `gs://training-job-outputs/test_results/teams_promotion_contract/codec-hedge-validation-20260427/promotion_contract/selected_threshold_scorecard.csv`
**Phase/step:** A.2-style follow-up promotion-contract scorecard on the C.3 codec_hedge slate winner (vs C.1 canonical baseline + P8A reference dominator).
**Headline numbers (calibrated contract — τ chosen by FPR-budget policy):**

| Checkpoint | rank | selected_threshold | lockbox_real_fpr | lockbox_fake_recall | dev_fake_macro_recall | deeplive_enhanced_dev recall |
|---|---:|---:|---:|---:|---:|---:|
| **P8A_REFERENCE_STEP5000** (winner) | 1 | 0.99095 | 0.00147 | 0.2372 | 0.1358 | 0.0239 |
| C3_CODEC_HEDGE_VC_STEP2000 | 2 | 0.97487 | 0.00514 | 0.2016 | 0.1714 | 0.1174 |
| C3_CODEC_HEDGE_OOD_STEP2500 | 3 | 0.98117 | 0.00588 | 0.1818 | 0.1786 | 0.1468 |
| C1_CANONICAL_VC_STEP2000 | 4 | 0.97835 | 0.00882 | 0.2253 | 0.2032 | 0.1963 |

**Headline numbers (diagnostic τ=0.5 — the trustworthy readout per memory `project_contract_policy_bug.md`):**

| Checkpoint | dev_real_all FPR | dev_real_poor_quality FPR | dev_real_lighting_extreme FPR | teams_fake_all_dev recall | visomaster_enhanced_macro recall | deeplive_enhanced recall | lockbox_real_fpr | lockbox_fake_recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **P8A_REFERENCE_STEP5000** | 0.121 | 0.081 | 0.111 | **0.702** | **0.356** | **0.530** | **0.062** | **0.652** |
| C3_CODEC_HEDGE_VC_STEP2000 | 0.121 | 0.062 | 0.106 | 0.776 | 0.421 | 0.714 | 0.133 | 0.514 |
| C3_CODEC_HEDGE_OOD_STEP2500 | 0.148 | 0.078 | 0.122 | 0.800 | 0.450 | 0.767 | 0.169 | 0.617 |
| C1_CANONICAL_VC_STEP2000 | 0.140 | 0.068 | 0.115 | 0.807 | 0.463 | 0.798 | 0.177 | 0.419 |

**Gate results (vs §4 Phase D — calibrated):**
- All 4 selected_threshold ∈ [0.975, 0.991] → **contract policy bug fired on all 4** (`< 0.99` gate fails for P8A; the others squeak under 0.99 but at the cost of even lower recall). Calibrated lockbox numbers are NOT promotion-grade for any candidate.
- P8A reference passes calibrated `lockbox_real_fpr ≤ 0.441 %` (0.147 %) and `lockbox_fake_recall ≥ 0.213` (23.7 %), but only because τ ≈ 0.99 has crushed recall floor-to-ceiling and the gate happens to be just below the crushed level.

**Gate results (vs §4 Phase D — diagnostic τ=0.5, trustworthy readout):**
- P8A reference: lockbox_real_fpr 6.2 % vs gate 0.441 % → **FAIL by 14×**. lockbox_fake_recall 65.2 % vs gate 21.3 % → **PASS**.
- C3 codec hedge VC: lockbox_real_fpr 13.3 % → **FAIL by 30×**. lockbox_fake_recall 51.4 % → **PASS**.
- C3 codec hedge OOD: lockbox_real_fpr 16.9 % → **FAIL by 38×**. lockbox_fake_recall 61.7 % → **PASS**.
- C1 canonical: lockbox_real_fpr 17.7 % → **FAIL by 40×**. lockbox_fake_recall 41.9 % → **PASS**.
- visomaster_enhanced_macro recall ≥ 0.30: P8A 0.36 / C.3 VC 0.42 / C.3 OOD 0.45 / C.1 0.46 → **all PASS**.
- deeplive_enhanced recall ≥ 0.034: all four PASS at τ=0.5; only P8A at calibrated τ also PASSes (0.024 → wait, fails — but the calibrated number is artifactual).

**Synthesis — codec_hedge axis:**
- C3 codec_hedge VC delivers an interesting trade vs P8A: **better fake recall on all dev fake suites** (deeplive 0.71 vs 0.53; visomaster_macro 0.42 vs 0.36; teams_fake_all 0.78 vs 0.70) but **2.2× worse lockbox real FPR** (13.3 % vs 6.2 %) and **0.84× lockbox fake recall** (0.51 vs 0.65). C3 OOD step 2500 sits between C3 VC and P8A on both axes (lockbox 0.62 recall / 0.169 FPR).
- The trainer's `best_anchor/composite` advantage (+30 % vs C.1; documented in 2026-04-26 5-corner read) translated to a real **dev-side lift** but **does not translate to lockbox** — codec hedge's residual generalization gain is consumed by the lockbox real-pool composition, which P8A handles better.
- **C.3 codec_hedge is NOT a Phase D promotion candidate.** P8A reference remains the standalone dominator on lockbox.

**Synthesis — overall Phase A/C status:**
- **Three A.2-style validations now agree**: P8A reference is the only lockbox-side dominator we have. The C-slate exhausted the immediate recipe-tuning hypothesis space (in_proj-SVD activation, codec aug, base-checkpoint choice, FT-step choice).
- **No candidate clears §4 Phase D `lockbox_real_fpr ≤ 0.441 %` at τ=0.5.** The calibrated path passes only when the contract policy bug forces τ ≈ 0.99 and crushes recall. **The plan's deployment gate is structurally unreachable in its current form** without (a) fixing the contract policy, (b) accepting a higher real-FPR floor, or (c) finding a fundamentally different recipe.

**Notes:**
- Vertex job `4993397216769998848` in us-east1; created 07:14 UTC, RUNNING by 07:17 UTC (no region switch needed), SUCCEEDED by ~09:43 UTC. ~2.5 h end-to-end, 32 reports + 5 contract artifacts.
- Image 1.3.220 (Cloud Build `5a4c19d7-0d1c-42a3-8698-4153567953fd`, 2 m 9 s).
- Image-currency check passed at launch (PCP #2 working as designed).
- 4-checkpoint yaml: `arena/checkpoint_maps/teams_target_domain.codec_hedge_2026-04-27.yaml`.

**Next implication:**
- §9 "no candidate" branch is now the live path. **Probe battery becomes the deliverable**, not a side-quest.
- Probe code is ready (image 1.3.220). Highest-leverage immediate run: probe-feature extraction on P8A_REFERENCE_STEP5000 (best lockbox), C3_CODEC_HEDGE_VC_STEP2000 (best dev fake recall + good real FPR), and ideally a control like RLP6_04_BASELINE_STEP23500 (known-shortcut baseline) to anchor the probe-accuracy scale.
- **Plan steering proposal (NOT applied to plan text per user instruction):** the §4 Phase D `lockbox_real_fpr ≤ 0.441 %` gate is unreachable at τ=0.5 by any candidate produced so far. Either (a) raise the FPR target to ~5–7 % (matches what P8A actually delivers), (b) escalate the contract policy bug fix from optional to blocking, or (c) commit to the §9 "no candidate" methodology deliverable + a data-axis Phase E. Surfaced for user decision.

### 2026-04-27 — Probe-battery readout + policy-fix end-to-end revalidation
**Source artifact(s):**
- Probe outputs (local): `analysis/probe_battery_2026-04-26/results/{p8a_reference_step5000,c3_codec_hedge_vc_step2000,rlp6_04_baseline_step23500}.source_bucket.json` + `summary.source_bucket.csv`
- Probe features (GCS): `gs://training-job-outputs/probe_battery_2026-04-26/{p8a_reference_step5000,c3_codec_hedge_vc_step2000,rlp6_04_baseline_step23500}/features.npz`
- Policy reruns (local): `analysis/policy_reruns_2026-04-27/{default,recall_floor_30}/{promotion_winner.json,checkpoint_summary.csv,selected_threshold_scorecard.csv,threshold_grid.csv}`
- Policy fix (uncommitted): `arena/score_teams_promotion_contract.py` (+61/-16) — defaults bumped to FPR 7%/10%, new `--target_fake_recall_min` flag, tier-based sort.
- Underlying contract reports (GCS, from this morning's codec_hedge validation): `gs://training-job-outputs/test_results/teams_promotion_contract/codec-hedge-validation-20260427/reports/`

**Phase/step:** Plan §9 "no candidate" deliverable (probe battery) + fork (ii) policy revalidation. See LOG entry `2026-04-27 14:55 UTC — forks-ii-iii-execution`.

**Headline numbers — probe battery (source_bucket, plan §6 PASS gate ≤ 0.25, chance = 0.10):**

| Checkpoint | train_acc | test_acc | × chance | §6 verdict |
|---|---:|---:|---:|---|
| RLP6_04_BASELINE_STEP23500 (known-shortcut control) | 0.997 | **0.970** | 9.7× | FAIL (memorized buckets) |
| P8A_REFERENCE_STEP5000 | 0.468 | **0.461** | 4.6× | FAIL |
| C3_CODEC_HEDGE_VC_STEP2000 | 0.428 | **0.416** | 4.2× | FAIL |

n_classes=10 (source_bucket), test_frac=0.20, sklearn LogisticRegression(C=1.0, max_iter=2000, n_jobs=1) per memory `feedback_sklearn_njobs.md`.

**Headline numbers — policy revalidation (codec_hedge reports re-scored under corrected 7%/10% policy):**

Variant A (default, no recall floor):

| Checkpoint | rank | selected_threshold | dev_primary_real_fpr | dev_fake_macro_recall | lockbox_real_fpr | lockbox_fake_recall |
|---|---:|---:|---:|---:|---:|---:|
| **P8A_REFERENCE_STEP5000** | 1 | 0.9156 | 0.0695 | 0.300 | **0.0184** | **0.387** |
| C3_CODEC_HEDGE_VC_STEP2000 | 2 | 0.8456 | 0.0698 | 0.426 | 0.0309 | 0.300 |
| C3_CODEC_HEDGE_OOD_STEP2500 | 3 | 0.9254 | 0.0698 | 0.392 | 0.0331 | 0.281 |
| C1_CANONICAL_VC_STEP2000 | 4 | 0.9300 | 0.0689 | 0.421 | 0.0382 | 0.253 |

Variant B (`--target_fake_recall_min 0.30`): byte-identical to Variant A. P8A's `dev_fake_macro_recall` is exactly 0.300; floor is not binding. No new information at this scorecard.

**Operating-point comparison (P8A reference, calibrated vs diagnostic):**

| τ | dev_primary_real_fpr | lockbox_real_fpr | lockbox_fake_recall | dev_fake_macro_recall | deeplive_enhanced_dev recall |
|---|---:|---:|---:|---:|---:|
| 0.916 (calibrated, 7%/10% policy) | 6.95% | **1.84%** | **38.7%** | 30.0% | 23.9% |
| 0.5 (diagnostic) | 12.11% | 6.17%¹ | 65.22%¹ | 53.1% | 53.0%¹ |

¹ from this morning's codec_hedge readout RESULTS entry. τ=0.5 row in `analysis/policy_reruns_2026-04-27/default/threshold_grid.csv` confirms `dev_primary_real_fpr=0.121`, `dev_fake_macro_recall=0.531`.

**Gate results — vs plan §6 (probe PASS ≤ 0.25 on source_bucket):**
- All three checkpoints **FAIL**. P8A and C3 are 1.7-1.8× over the gate; RLP6_04 control is 3.9× over.
- The **gradient** is informative: codec aug (C3) is ~9% relative cleaner than P8A on the source_bucket axis, and both are ~2× cleaner than the known-shortcut RLP6_04 control. Codec aug *is* a partial lever — just not enough to clear the gate alone.

**Gate results — vs plan §4 Phase D (corrected 7%/10% policy):**
- `lockbox_real_fpr ≤ 0.441%`: P8A 1.84% → **FAIL by 4.2×**. (Was 0.147% under buggy policy — that PASS was artifactual; τ ≈ 0.99 had crushed recall, and the gate happened to sit just under the crushed real FPR.)
- `lockbox_fake_recall ≥ 0.213`: P8A 38.7% → **PASS** by ~1.8×.
- The §4 Phase D `lockbox_real_fpr` gate remains structurally unreachable under recipe-tuning. Three A.2-style validations now agree; the corrected policy did not change the headline.

**Gate results — vs user's "operationally OK with 7% FPR if recall ≥ 95% after further training" stance:**
- 38.7% lockbox recall at calibrated τ — recall is 56 pp short of target.
- 65.2% lockbox recall at τ=0.5 (no calibration) — still 30 pp short, plus partially shortcut-attributable per probe verdict.
- The 95% recall target is not on the recipe-tuning trajectory.

**Synthesis — probe battery as methodology deliverable (plan §9):**
- The probe battery cleanly distinguishes a known-shortcut model (RLP6_04 at 0.97) from recipe-tuned models (P8A/C3 at 0.42-0.46). The control validates the probe scale.
- P8A is *partially* shortcut-leaky: 4.6× chance accuracy on source_bucket. Genuine fake-recall capacity is some unknown fraction of measured recall; the probe alone can't quantify which fraction without counterfactual data.
- Codec aug (C3 vs P8A) reduces source-bucket leakage by ~9% relative. Statistically meaningful direction; not enough magnitude to clear the §6 PASS gate.
- The §9 "no candidate" branch deliverable is well-supported. The plan can credibly claim: "we've ruled out recipe-tuning under our hypothesis space; the substrate has measurable shortcut leakage; the next sprint is data-axis (counterfactual same-content cross-pipeline pairs)."

**Synthesis — policy fix:**
- Tier-based sort (budget-OK > budget-OK-but-floor-failed > budget-violated) produces deterministic, plan-aligned τ selection. Defaults bumped at the source level so future runs auto-use the corrected policy. Memory `project_contract_policy_bug.md` is closed for this configuration.
- The *ranking* of checkpoints is robust to the fix (same as buggy-policy ranking). What changed is the *operating point* (τ ≈ 0.92 instead of ~0.99) and the headline lockbox recall (~38% instead of ~24%).
- The recall floor knob exists but is not binding for the codec_hedge scorecard. Future use: a checkpoint with high dev recall but poor calibration would force the policy to skip non-meeting tiers. Useful as a safety belt.

**Notes:**
- All probe runs local, n_jobs=1 enforced (memory `feedback_sklearn_njobs.md`; n_jobs=-1 caused 3 system reboots on 2026-04-27 — bugfix in place at `analysis/probe_battery_2026-04-26/run_linear_probes.py:91`, file uncommitted).
- Probe runs covered only `source_bucket`. Optional follow-ups (~3 min each, local): `--label_field provenance` (real vs fake — sanity check on training objective) and `--label_field source_name` (20 sources, finer than buckets). Not run today.
- Recall_floor variant from prior session was lost to /tmp wipe; re-validation now lives on durable storage at `analysis/policy_reruns_2026-04-27/`.

**Next implication / Plan steering (NOT applied):**
- **PCP — amend §4 Phase D gate** to require *probe PASS* (source_bucket ≤ 0.25) in addition to FPR/recall budgets. Without that, calibrated recall numbers can be shortcut-inflated. AWAITING USER DECISION.
- **PCP — promote §9 Phase E** (data-axis counterfactual same-content cross-pipeline pairs) from "if no candidate" branch to active work. The "no candidate" condition is met under any reasonable interpretation of the FPR-gate fork (i). AWAITING USER DECISION.
- User is consulting another agent on "where we go now" with full context (this entry, LOG entry 14:55 UTC, refreshed `HANDOFF.md`).

---

### 2026-04-27 17:30 CEST — agent: Claude Opus 4.7 (1M context) / autonomous diagnostic
**Phase/step:** Plan-v3 Day-1 §1.6 — identity-corruption audit
**Inputs measured:**
- `analysis/lockbox_tagging/lockbox_tags_2026-04-27.parquet` (839 frames, 5 lockbox identity_keys, R9A predictions joined)
- 30-frame sample of `PC_Generator__s13` from dev split (downloaded from GCS, ArcFace embedded, clustered)
- Read of `data/sources/{combined_paired,df40_paired,visomaster}.py` to characterize training-set identity construction
- ArcFace ONNX `~/.insightface/models/buffalo_l/w600k_r50.onnx` (already on disk; CoreML EP active)

**Method:** ArcFace embeddings (512-d, unit-normed) + greedy single-link clustering at cosine sim ≥ 0.45 (deliberately permissive — same-person ArcFace pairs ≥0.6 in normal lighting; threshold 0.45 only flags labels where pair sims drop clearly into the cross-identity range).

**Lockbox 5-identity slice:**

| identity_key | n_frames | n_clusters | largest_share | cluster_sizes | verdict |
|---|---|---|---|---|---|
| dor_shkedi (real) | 275 | 1 | 1.000 | [275] | clean — single identity |
| Cam_Test__s33 (fake) | 334 | 2 | 0.997 | [333, 1] | clean (1 outlier) |
| Chikara_Takahashi__s22 (fake) | 42 | 1 | 1.000 | [42] | clean |
| bla_bla_chow__s1 (real) | 68 | 1 | 1.000 | [68] | clean |
| PC_Generator__s15 (mixed) | 120 | 2 | 0.758 | [91 fake, 29 real] | real-vs-fake source mismatch |

- **Frame corruption rate (lockbox 5-id slice):** 29 / 839 = **3.5%** in non-dominant cluster.
- **Cluster-vs-label crosstab on PC_Generator__s15:** cluster 0 = 91 fake / 0 real; cluster 1 = 0 fake / 29 real. Two distinct humans share the label, one is the fake-source and one is the real subject.
- **Dor sanity finding:** dor_shkedi is a single ArcFace identity. The parallel agent's "0% accuracy on dor_shkedi" finding (R9A_run1 model) is **genuine model failure**, not label corruption.

**Dev-split label PC_Generator__s13 (n=30 sampled, all label=real):**

| metric | value |
|---|---|
| n embedded | 30 |
| n_clusters | **3** |
| cluster_sizes | [17, 10, 3] |
| intra-cluster sim mean | 0.859 |
| intra-cluster sim p10 | 0.778 |
| inter-cluster sim mean | 0.186 |
| inter-cluster sim p90 | 0.293 |
| inter-cluster sim max | 0.403 |

- **Verdict:** PC_Generator__s13 contains ≥3 distinct humans within its all-real frames in dev. User's image (which suggested ≥2) is a lower bound. The label is a *capture session*, not a person. Manifest confirms: 141 distinct video segments under that one identity_key.

**Manifest-level structural check (no GCS download):**

| split | n_idents | mean videos/ident | max | p90 |
|---|---|---|---|---|
| lockbox | 6 | 269 | 1138 | 1138 |
| dev | 34 | 168 | 545 | 409 |

- Many identities aggregate hundreds of video segments. The mechanism (capture-session label spans an entire pipeline run that played many source videos) is structural, not isolated.

**Training-set identity construction (read of source code):**

| dataset | identity rule | per-video? |
|---|---|---|
| DF40 | `df40_{sample.target_identity}` | yes |
| DeepLive | `realpool_{original_video_name}` | yes |
| Visomaster | videoID portion of sample_id | yes |

- Identity-stratified split is enforced (`df40_paired.py:301-380` does explicit overlap-check; logs "✓ No identity overlap between splits (leakage check passed)").
- **Verdict:** training-set identities are video-derived; session-aggregation corruption does NOT propagate. P11_TARGETED's `identity_balanced_sampling: True` and `loss.contrastive_regularization` operate on clean groupings.

**Implications — propagating into plan v3 framings:**

1. **§1.5 "dor_shkedi 0% accuracy = 33% of FPR slice" is now established as genuine model failure** (R9A model) on a single real identity. Track A.5d (within-identity ArcFace outlier filter) is *down-weighted* — there is no within-identity outlier to find when ArcFace says the identity is one person. Track A.5c (well-conditioned crops only) still useful for tiny-crop hygiene but won't gain the full headroom §1.5 implied.

2. **§1.5 "PC_Generator__s15 21% real vs 98% fake accuracy" reframed:** the 91-frame fake cluster and 29-frame real cluster are different humans. The model is correctly classifying both. This is a *naming pathology*, not a model pathology; but it does mean per-identity diagnostics can confound source-face mismatch with model failure.

3. **§1.6 amendment — training-set-corruption hypothesis FALSIFIED.** Identity-balanced sampling and contrastive loss in P11_TARGETED operate on clean groupings. **P11 launch decision: GREEN to proceed when authorized.**

4. **§7 Validation amendment — dev session labels need ArcFace explosion.** The right denominator for dev FPR is "per ArcFace cluster", not "per session label". PC_Generator__s13 should count as 3 evaluation units, weighted by their respective cluster sizes. Day 2 morning addition.

5. **Recall numerator is unaffected.** Per-method recall on dev fake suites is `(label==fake & pred==fake) / (label==fake)`, both denominator and numerator use sample-level labels not identity grouping. Plan v3 §3 hard targets (visomaster_enhanced_macro / deeplive_enhanced / teams_fake_all_dev recall ≥ 0.90) remain measurable as defined.

**Outputs:**
- `analysis/identity_corruption_audit_2026-04-27/{per_label_summary.csv,per_label_summary.json,verdict.json,report.txt,frame_clusters.parquet}`
- `analysis/identity_corruption_audit_2026-04-27/s13_dev/{s13_clusters.csv,s13_summary.json}`
- `analysis/identity_corruption_audit_2026-04-27.py` (lockbox audit script)
- `analysis/identity_corruption_audit_2026-04-27/audit_s13_dev.py` (dev-sample audit script)

**Notes on cost:**
- All compute local on M1 Pro. ArcFace embeds via CoreML EP, ~8-10ms / frame. 839 + 30 frames in <1 minute total.
- 30-frame GCS download for s13 dev sample completed in ~6s.
- No GPU/Vertex spend. No commits yet — analysis scripts uncommitted alongside policy fix and probe scripts; will bundle into Day-1.1 housekeeping commit when authorized.

---

### 2026-04-27 18:06 CEST — agent: Claude Opus 4.7 (1M context) / autonomous diagnostic
**Phase/step:** Plan-v3 — deployment-honest evaluation framework
**Inputs measured:**
- `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` (7,334 frames, R9A predictions, full property tagging including ArcFace embeddings)
- `gs://real-teams-dor-roee/session_20260424_combined_tags_121458_121007/` (180 production frames + sidecar JSON with deployment-time model scores)
- Read of source manifests for training-set identity construction (separate audit, see 17:30 entry)
- Read of `arena/score_teams_promotion_contract.py` to confirm scorer aggregation level (per-video; see 17:30 entry on the dev session label corruption)

**Method:**
- Replicated parallel agent's per-frame property analysis on the full 7,334-row parquet — face_pixel_area, sharpness_laplacian, brightness_v_mean, yaw, pitch — at τ=0.5.
- Decomposed by split (dev / lockbox) to test whether the property pattern is split-uniform.
- Downloaded 180 production frames; re-ran quality + face_geometry tagging; compared OK regime vs FAIL regime on each property.
- Computed deployment-honest gate: production-pool FPR per τ × dev-fake per-method recall per τ, looking for a τ where both sides clear.

**Key results — A. Threshold sweep, per split (R9A, full parquet):**

| split | n | n_real | n_fake | τ=0.5 | τ=0.9 | τ=0.95 | τ=0.97 | τ=0.99 |
|---|---|---|---|---|---|---|---|---|
| dev | 6495 | 4006 | 2489 | FPR 19.1% / R 97.6% | FPR 12.3% / R 90.2% | FPR 10.7% / R 85.0% | FPR 9.9% / R 79.5% | FPR 8.1% / R 63.1% |
| lockbox | 839 | 414 | 425 | **FPR 89.4%** / R 91.5% | FPR 65.9% / R 64.5% | FPR 40.3% / R 51.5% | FPR 27.1% / R 43.1% | FPR 7.5% / R 24.2% |

Lockbox is approximately 9× harder than dev at τ=0.5 — the model is essentially flagging all lockbox real frames as fake in that regime. Frame-level (not per-video aggregation) per the parquet's natural granularity.

**Key results — B. Property quartiles (R9A, full set, τ=0.5):**

| column | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| face_pixel_area | n=1779 FPR 37.6% R 93.6% | n=1779 FPR 29.7% R 99.1% | n=1778 FPR 25.1% R 99.2% | n=1779 **FPR 1.0% R 95.1%** |
| sharpness_laplacian | <44.6 FPR 44.3% R 96.9% | 44.6-157 FPR 12.9% R 91.5% | 157-402 FPR 12.5% R 100% | >402 FPR 44.6% R 100% |
| brightness_v_mean | <136 FPR 35.4% | 136-148 FPR 31.6% | 148-168 FPR 22.5% | >168 FPR 15.3% |
| pitch_deg | <-5.4° FPR 12.3% | -5.4-9.5° FPR 24.5% | 9.5-15.4° FPR 37.6% | >15.4° FPR 33.7% |

face_pixel_area shows a clean 38× FPR drop in Q4 with recall preserved — this replicates the agent's headline.

**Key results — C. face_pixel_area quartile by SPLIT (R9A, full set, τ=0.5):**

| split | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| dev | n=1680 FPR 35.5% R 93.6% | n=1474 FPR 10.5% R 99.3% | n=1679 FPR 20.0% R 99.2% | n=1443 **FPR 1.0% R 97.4%** |
| lockbox | n=99 (n_real 99 / n_fake 0) FPR 64.6% | n=305 FPR 99.6% | n=99 FPR 89.5% | n=336 (n_real 0 / n_fake 336) **FPR 0% but UNDEFINED** |

**Critical observation: lockbox has zero real frames in Q4** (face_pixel_area > 57k). The lockbox real pool maxes out at 31k px²; the dev split has real frames spanning all four quartiles. The "deployment-honest face-area filter" is *applicable to the dev split* but *empty on the lockbox split*.

**Key results — F. dor_shkedi face area (R9A, lockbox real, 100% accuracy=0%):**

| metric | value |
|---|---|
| n | 275 |
| mean | 20,646 |
| p10 | 2,703 |
| p50 | 21,597 |
| p90 | 26,774 |
| max | 28,521 |
| % below 30,000 | 100% |
| % below 57,000 | 100% |

100% of dor_shkedi frames are below the 30k cutoff. The model's "0% accuracy" on dor_shkedi is fully explained by the small-face FPR property (~30% in this size range) compounded across 275 frames. **This is genuine property-driven model failure**, not label corruption (per 17:30 entry) and not capture-condition outliers in any deployment-irrelevant sense.

**Key results — Production deployment characterization (180 frames from `gs://real-teams-dor-roee/`):**

Tag-level per-regime statistics (deployed model's score and verdict from session sidecar):

| tag | n | regime | mean score | sharpness | face_area | pitch | verdict majority |
|---|---|---|---|---|---|---|---|
| dor-laptop-whiteish | 30 | OK | 0.018 | 426 | 16,163 | -11° | GREEN ×30 |
| dor-laptop-yellowish | 30 | OK | 0.040 | 329 | 14,416 | -11° | GREEN ×30 |
| roee-windows-laptop | 30 | OK | 0.007 | **1064** | 12,674 | +12° | GREEN ×29 / YELLOW ×1 |
| dor-webcam-false-flag | 30 | FAIL | 0.878 | **77** | 26,529 | +17° | RED ×16 / YELLOW ×12 / GREEN ×2 |
| dor-webcam-no-vbg | 30 | FAIL | 0.940 | 141 | 14,035 | +16° | RED ×24 / YELLOW ×6 |
| roee-mac-laptop-vbg-false-flag | 30 | FAIL | 0.900 | 300 | 23,295 | 0° / yaw 49° | RED ×7 / YELLOW ×1 / GREEN ×22 |

**Aggregate regime contrast (n=90 each):**

| property | OK mean | FAIL mean | ratio / Δ |
|---|---|---|---|
| sharpness_laplacian | 606 | 173 | **3.5× lower** |
| brightness_v_mean | 192 | 165 | -27 |
| pitch_deg | -3° | +11° | sign flip |
| yaw_deg | +1.6° | +6.8° | similar |
| face_pixel_area | 14,418 | 21,286 | FAIL is *larger* |
| score | 0.022 | 0.906 | 41× higher |

**Sharpness is the production-relevant axis. Face area is NOT.** Face area in production is bounded ≤ 31k for both regimes. The webcam-blur regime is what drives false-positives at deployment time.

**Production FPR by τ (all 180 frames):**

| τ | n flagged | FPR |
|---|---|---|
| 0.50 | 89 | 49.4% |
| 0.90 | 64 | 35.6% |
| 0.95 | 49 | 27.2% |
| 0.97 | 32 | 17.8% |
| 0.98 | 25 | 13.9% |
| **0.99** | **9** | **5.0%** |
| 0.995 | 1 | 0.6% |

**At τ=0.99: production FPR exactly 5% (matches gate). All 9 false-flags are within the FAIL regime tags.**

OK regime FPR at every τ in [0.5, 0.99]: **0%**.
FAIL regime FPR at τ=0.99: 9/90 = 10%.

**R9A dev fake per-method recall at τ=0.99:**

| method | n | recall at τ=0.99 |
|---|---|---|
| deeplive_enhanced | 545 | **51%** |
| teams_capture_dor_shkedi_s16 | 78 | **2.6%** |
| teams_capture_noyn_sharker_s23 | 324 | **26%** |
| teams_flat_xiang_xiang2_feng | 135 | **21%** |
| teams_capture_test_cam_s53 | 165 | 66% |
| teams_capture_pc_generator_s3 | 118 | 67% |
| teams_capture_cam_test_s46 | 124 | 53% |
| teams_capture_pc_generator_s9 | 46 | 78% |
| teams_capture_cam_test_s35 | 365 | 89% |
| teams_capture_test_cam_s76 | 138 | 92% |
| teams_capture_test_cam_s73 | 101 | 96% |
| teams_capture_cam_test_s32 | 235 | 96% |
| teams_capture_pc_generator_s4 | 30 | 97% |
| teams_capture_cam_test_s38 | 85 | 96% |

**Bimodal recall: 4 methods at the τ=0.99 operating point clear the 90% gate (cam_test_s38, test_cam_s73, cam_test_s32, pc_generator_s4); 4 methods miss by >35 pp (deeplive, dor_shkedi_s16, noyn_sharker, xiang_feng). visomaster predictions absent from R9A on this bucket — would need separate inference.**

**Synthesis — "no single-τ candidate" verdict for R9A on deployment-honest framing:**

R9A meets the production-FPR side of the gate at τ=0.99. It fails the per-method recall side by ≥39 pp on at least 4 of 14 fake methods. A single global τ cannot close this. Per-method τ would help but isn't deployable without method-aware inference.

**Synthesis — production failure-mode mechanism:**

The 90 false-flagged production frames split cleanly into 3 known-failure tags. The discriminating axis is **sharpness_laplacian** (3.5× drop) plus **looking-down pitch** (sign flip). This is consistent with the "webcam captures with virtual-bg processing under varying compression" deployment surface. Training augmentations that simulate this (motion blur + webcam codec + downward camera angle) are concrete data-axis levers.

**Synthesis — the lockbox is structurally inadequate as a deployment FPR proxy, but P8A may not need a "deployment-honest filter" because it has the production-FPR pool directly:**

The 180-frame production set IS the deployment FPR pool. We don't need to subset the lockbox; we just need to run P8A against the production set + dev fakes and read out the gate. The current bottleneck: P8A per-frame predictions don't exist on disk. Running `batch_inference_gcs.py` on Vertex (~30 min, ~$5-10) gives us this end-to-end.

**Caveats:**
- All numbers above are R9A. P8A is stronger by training AUC (0.9926 vs 0.9891) and lockbox AUC; per-method recall should be uniformly better, possibly substantially. The bimodal recall pattern may collapse to single-mode under P8A.
- 180 production frames is small. The OK/FAIL regimes are deliberately balanced 90/90, not a natural distribution. The 5%-at-τ=0.99 number is a property of THIS sample, not deployment as a whole.
- visomaster_enhanced predictions are absent from this analysis. Verifying P8A on visomaster requires either a separate batch_inference run on `gs://visomaster-enhanced-face-cropped-v2` or use of suite-aggregate metrics from `policy_reruns_2026-04-27/`.
- The "deployed model's sidecar scores" on the production frames is not necessarily P8A. Could be an earlier shipped checkpoint. Score distribution (OK ~0.02, FAIL ~0.9) is a typical Effort-detector profile and is consistent with a well-trained model that just has this specific failure mode.

---

## 2026-04-27 18:40 CEST — P8A deployment-honest evaluation (HARD DATA)

**Source:** Vertex `batch-infer-20260427-182324` (us-east1, ran in 3 min). Outputs:
- `gs://training-job-outputs/batch_inference_results/p8a_step5000_deployment_honest__teams-faces-data-test-2914-fake-4420-real-feb-28.csv` (9446 rows)
- `gs://training-job-outputs/batch_inference_results/p8a_step5000_deployment_honest__teams-faces-data-test-prod-honest-180-2026-04-27.csv` (180 rows)
- Local copies in `analysis/deployment_honest_eval_2026-04-27/`.

### A. Production-honest FPR (180 frames from gs://real-teams-dor-roee/...)

| τ | ALL FPR | OK regime | FAIL regime |
|---|---|---|---|
| 0.50 | 30.0% (54/180) | **0.0%** | 60.0% |
| 0.90 | 16.7% | 0.0% | 33.3% |
| 0.95 | 10.0% | 0.0% | 20.0% |
| 0.97 | 6.1% | 0.0% | 12.2% |
| **0.9741** | **5.0%** | 0.0% | 10.0% |
| 0.99 | **0.6%** | 0.0% | 1.1% |

**P8A solves the OK regime perfectly** (deployment-correct crops with whiteish/yellowish lighting, looking forward). Even at τ=0.30, OK regime is 0% FPR.

**FAIL regime (deployment-correct crops, R9A-flagged) still hard.** At τ=0.50, 60% of FAIL frames classified fake. Drops to 10% at τ=0.9741 and 1% at τ=0.99.

### B. Per-tag (6 tags, 30 frames each) at τ=0.50

| Tag | Regime | FPR | Mean prob_fake |
|---|---|---|---|
| dor-real-laptop-correct-no-virtual-bg-whiteish | OK | 0.0% | 0.006 |
| dor-real-laptop-correct-no-virtual-bg-yellowish | OK | 0.0% | 0.007 |
| roee-real-windows-laptop-correct | OK | 0.0% | 0.005 |
| roee-mac-laptop-false-flag-virtual-bg | FAIL | 23.3% | 0.353 |
| dor-real-webcam-false-flag | FAIL | **76.7%** | 0.731 |
| dor-real-webcam-false-flag-no-virtual-bg | FAIL | **80.0%** | 0.744 |

**Dor's webcam frames at proper crop are the load-bearing failure.** Roee's mac-laptop FAIL is much milder.

### C. Per-method dev recall (joined with parquet)

τ_5pct (production) = **0.9741**. Decision table:

| Method | n | τ=0.50 | τ=0.97 | τ=0.9741 | τ=0.99 |
|---|---|---|---|---|---|
| **deeplive_enhanced** | 545 | **0.591** | 0.167 | **0.132** | 0.042 |
| teams_fake_all_AGG | 2369 | **0.900** | 0.729 | 0.719 | 0.634 |
| teams_capture_cam_test_s32 | 235 | 1.000 | 0.987 | 0.979 | 0.945 |
| teams_capture_cam_test_s33 | 334 | 0.620 | 0.183 | 0.168 | 0.072 |
| teams_capture_cam_test_s35 | 365 | 0.978 | 0.868 | 0.855 | 0.753 |
| teams_capture_cam_test_s38 | 85 | 1.000 | 0.965 | 0.941 | 0.847 |
| teams_capture_cam_test_s46 | 124 | 1.000 | 0.984 | 0.984 | 0.968 |
| teams_capture_test_cam_s53 | 165 | 1.000 | 0.921 | 0.921 | 0.770 |
| teams_capture_test_cam_s73 | 101 | 1.000 | 1.000 | 1.000 | 1.000 |
| teams_capture_test_cam_s76 | 138 | 0.993 | 0.993 | 0.993 | 0.986 |
| teams_capture_pc_generator_s3 | 118 | 1.000 | 0.949 | 0.949 | 0.898 |
| teams_capture_pc_generator_s4 | 30 | 1.000 | 1.000 | 1.000 | 0.967 |
| teams_capture_pc_generator_s9 | 46 | 1.000 | 1.000 | 1.000 | 0.978 |
| teams_capture_pc_generator_s15 | 91 | 0.989 | 0.945 | 0.923 | 0.868 |
| teams_capture_dor_shkedi_s16 | 78 | 0.987 | 0.654 | 0.641 | 0.449 |
| teams_capture_noyn_sharker_s23 | 324 | 0.907 | 0.599 | 0.580 | 0.401 |
| teams_flat_xiang_xiang2_feng | 135 | 0.496 | 0.030 | 0.030 | 0.000 |

### D. Lockbox per-identity FPR/recall (P8A)

| Identity | n_real | n_fake | FPR @0.50 | Recall @0.50 | FPR @0.9741 | Recall @0.9741 |
|---|---|---|---|---|---|---|
| Cam_Test__s33 | 0 | 334 | — | 0.620 | — | 0.168 |
| Chikara_Takahashi__s22 | 42 | 0 | **0.833** | — | 0.214 | — |
| PC_Generator__s15 | 29 | 91 | 0.897 | 0.989 | 0.207 | 0.923 |
| bla_bla_chow__s1 | 68 | 0 | 0.015 | — | 0.000 | — |
| **dor_shkedi** | 275 | 0 | **0.105** | — | **0.015** | — |

P8A vs R9A on dor_shkedi lockbox: R9A had 100% FPR (mean prob 0.95). P8A drops to **10.5% at τ=0.5 and 1.5% at τ=0.9741**. Major improvement from the unfreeze + in_proj-SVD combo.

PC_Generator__s15 line is unreliable (confirmed corrupt: 91 fakes and 29 reals are different humans per identity audit 2026-04-27).

### E. Property quartiles on dev (P8A) at τ=0.9741

**face_pixel_area:**
| Q | range | FPR | Recall |
|---|---|---|---|
| Q1 | 1k-17.6k | 17.9% | 51.0% |
| Q2 | 17.6k-27.5k | 1.0% | 27.8% |
| Q3 | 27.5k-57.2k | 0.3% | **86.6%** |
| Q4 | 57.2k-283k | 0.0% | 69.0% |

**sharpness_laplacian:**
| Q | range | FPR | Recall |
|---|---|---|---|
| Q1 | 4-44 (blurry) | 3.5% | 78.7% |
| Q2 | 44-157 | 1.3% | 62.7% |
| Q3 | 157-402 | 0.7% | **13.3%** |
| Q4 | 402-3184 (sharp) | 13.8% | 27.0% |

**pitch_deg:**
| Q | range | FPR | Recall |
|---|---|---|---|
| Q1 | -64..-5 (looking up) | 1.0% | 51.8% |
| Q2 | -5..9 (forward) | 3.7% | 53.0% |
| Q3 | 9..15 (down) | **13.2%** | 63.8% |
| Q4 | 15..65 (down hard) | 8.1% | 72.5% |

**Dev real FPR at canonical thresholds:**
| τ | FPR (n=4420) |
|---|---|
| 0.50 | 14.0% |
| 0.97 | 6.2% |
| 0.9741 | 6.0% |
| 0.99 | 4.0% |

### Verdict vs plan-v3 hard targets

| Target | Status |
|---|---|
| modern_lockbox_real FPR ≤ 5% | **MET** at τ=0.9741 (production: 5.0%); MET at τ=0.99 (0.6%) |
| dev_real_all FPR ≤ 7% | MET at τ=0.97 and above |
| **deeplive_enhanced recall ≥ 90%** | **FAIL** — 13.2% at τ=0.9741, 59% at τ=0.50 |
| teams_fake_all recall ≥ 90% | **FAIL** at τ_5pct (72%); MET at τ=0.50 (90%) |
| visomaster_enhanced_macro recall ≥ 90% | **NOT MEASURED** — bucket not in this job |

### Plan-v3 verdict gate γ ("big gap")

Triggered. Recipe-tuning era was already declared over by plan-v2. The remaining levers per plan-v3:

1. **Ensemble + per-method τ + calibration** (Track A.4, C.1) — could close some of the gap on deeplive_enhanced if other models compensate, but with only one substrate (P8A) and one available co-candidate (C3 codec_hedge), the lift ceiling is small.
2. **Train P11_TARGETED with new FAIL-regime-targeted augmentations** (Track B). Recipe revision needed based on deployment-honest mechanism (sharpness drop / downward pitch / brightness compression / per-method oversampling for deeplive_enhanced).
3. **Inference-time TTA** (Track C.2) — flip + Teams-codec-sim. Bounded lift.

---

## 2026-04-28 — P11 Overnight Slate Promotion Contract Scorecard

**Job:** `7291147820403261440` (us-east1), 159 min wall, 160/160 reports landed at `gs://training-job-outputs/test_results/teams_promotion_contract/p11-overnight-2026-04-28/reports/`.
**Scorer:** `analysis/modern_lockbox_v2_2026-04-27/score_p11_modern_v2.py`. Outputs at `analysis/modern_lockbox_v2_2026-04-27/p11_overnight_modern_v2_scorecard.csv` + `_summary.json`.
**modern_v2 subset:** 281 real / 367 fake frames (the modern-Teams-conditions subset of full lockbox).

### Headline scorecard at τ=0.5

| Candidate | viso_dev | deeplive_dev | teams_fake_dev | modern_v2 FPR | lockbox FPR | teams_real_dev FPR |
|---|---|---|---|---|---|---|
| **P8A reference** (step 5000) | 0.356 | 0.530 | 0.756 | **0.036** | **0.068** | **0.122** |
| P11_MILD step1000 | 0.440 | 0.710 | 0.817 | 0.181 | 0.147 | 0.138 |
| **P11_HEAVY step1000** | **0.520** | **0.778** | **0.847** | 0.270 | 0.207 | 0.151 |
| P11_HEAVY_DEEPLIVE step1000 | 0.445 | 0.800 | 0.836 | 0.267 | 0.186 | 0.149 |
| P11_WEBCAM_HARDEN step1000 | 0.233 | 0.554 | 0.751 | 0.060 | 0.092 | 0.140 |

### Calibrated τ=0.9741 (5%-prod target on P8A)

| Candidate | viso_dev | deeplive_dev | teams_fake_dev | modern_v2 FPR | lockbox FPR | teams_real_dev FPR |
|---|---|---|---|---|---|---|
| P8A reference | 0.065 | 0.119 | 0.534 | **0.0036** | 0.0092 | 0.050 |
| P11_MILD | 0.093 | 0.316 | 0.589 | 0.0107 | 0.0219 | 0.062 |
| **P11_HEAVY** | **0.107** | **0.400** | **0.617** | 0.0107 | 0.0261 | 0.066 |
| P11_HEAVY_DEEPLIVE | 0.100 | 0.422 | 0.614 | 0.0107 | 0.0169 | 0.061 |
| P11_WEBCAM_HARDEN | 0.027 | 0.136 | 0.526 | **0.0036** | 0.0099 | 0.047 |

### Plan v3 §6 verdict per candidate
All five candidates classified **β/γ ASSESS** by the scorer's automatic gate (90/5 not met on any candidate at any tested τ). 

### Reads
1. **Recipe direction confirmed.** P11_HEAVY at τ=0.5 lifts deeplive_enhanced recall +25pp over P8A (0.530 → 0.778) — the family that had the lowest baseline. Visomaster +16pp, teams_fake_all +9pp. This is the largest single-recipe recall lift we've seen.
2. **FPR cost is the step-1000-anchor-worst-moment artifact.** modern_v2 FPR at τ=0.5: HEAVY 27% vs P8A 3.6%. At calibrated τ=0.9741 it collapses to ~1% for all P11 candidates and ~0.4% for P8A — the modern_v2 substrate is doing exactly what's claimed.
3. **Recall–FPR tradeoff is broken at step 1000** because anchor hasn't recovered (P8A's anchor_mean ~0.10; P11_HEAVY at step 1000 anchor_mean 0.918). Calibrated τ that solves FPR also crushes recall.
4. **WEBCAM_HARDEN had lowest FPR but worst recall.** Aug overhardened the model. **Drop axis from any next-wave.**
5. **HEAVY_DEEPLIVE edges HEAVY on deeplive only** (+2.2pp), loses on viso (-7.5pp). 6× weight broke loss balance. **Drop axis from any next-wave.**

---

## 2026-04-28 — P12_HEAVY_LONG Anchor Trajectory (W&B run v6237h5e)

P12_HEAVY_LONG inherited P11_HEAVY recipe + `total_training_steps: 8000` + (broken) `periodic_saves` config. Early-stopped at step 6000 (epoch 3). Anchor monitor data captured at every-500-step eval cadence. **Critical for next-plan strategy:**

| step (approx) | anchor_mean | anchor_composite |
|---|---|---|
| ~500 | 0.821 | — |
| ~1000 | 0.731 | 0.221 |
| ~1500 | 0.851 | 0.119 |
| ~2500 | 0.735 | 0.161 |
| ~6000 | **0.601** | 0.152 |

**Per-1000-step recovery rate: ~0.04.** From step 6000 (anchor_mean 0.601) to P8A's level (~0.10), need ~17000 more steps. Pure HEAVY recipe FT-extension is not a path to 90/5.

### What this rules out vs validates
- **Rules out:** "Just train P11_HEAVY longer" — anchor recovery rate is too slow.
- **Validates:** the recipe direction (heavier `context_variation_scale`, `real_codec_uplift`, higher `teams_codec_sim_p`) does drive recall up. Future training must combine this direction with anchor-aware regularization (loss term, freeze policy, or counter-augmentation) — NOT just longer training.

### Trainer patch failure mode (load-bearing for any future agent)
- Patch added to `trainer/trainer.py:2319-2362` at the end of `_run_validation()`. Logic: at every eval where `step_cnt in step_list`, call `save_ckpt(prefix='periodic')`.
- **Patch never fired in P12.** Yaml `periodic_saves.enabled: true` was present; image 1.3.223 used; debug log confirmed `periodic_saves` is in wandb.config keys.
- **Likely root cause** (high confidence, unconfirmed): `self.config.get('periodic_saves')` returns a wandb.Config wrapper, not a plain dict. Codebase has `isinstance(self.config.get(...), dict)` defenses elsewhere (e.g., `trainer/trainer.py:392-393`); my patch lacks that guard. Fix is one line. **Must be unit-tested locally before any relaunch.**

### Cost incurred
- Inference (P11): ~$15.
- Training (P12, truncated to 161 min): ~$25.
- Total Day 2: ~$41.

---

## 2026-04-28 13:30 CEST — Option A inference-time stack analysis (verdict γ)

Subagent (general-purpose, agentId a30ca1638125bef61) ran Plan v3 Track A: calibration + per-method τ + noisy-OR ensemble on the 5 cached frame-level prediction CSVs (P8A reference + P11_HEAVY + P11_HEAVY_DEEPLIVE + P11_MILD + P11_WEBCAM_HARDEN, all step_1000). No GPU spend. Output at `analysis/option_a_ensemble_2026-04-28/`.

### Configuration sweep

| Strategy | Configs | Passed gate |
|---|---|---|
| Single candidate × 3 calibrations × 5 ops | 30 | **0** |
| Noisy-OR ensemble × 3 calibrations × 3 ops | 45 | **0** |
| Weighted noisy-OR (P8A + P11_HEAVY) × 10 weights | 10 | **0** |
| **Total** | **85** | **0** |

### Best operating point found across all 85

`p8a_reference + p11_heavy_step1000` isotonic-noisy-OR, τ=0.846:

| Metric | Value | Target | Status |
|---|---|---|---|
| viso recall | 0.313 | ≥ 0.90 | fails (-58.7pp) |
| deeplive recall | 0.582 | ≥ 0.90 | fails (-31.8pp) |
| teams_fake_all recall | 0.750 | ≥ 0.90 | fails (-15.0pp) |
| modern_v2 FPR | 0.0498 | ≤ 0.05 | **passes** |
| teams_real_all_dev FPR | 0.109 | ≤ 0.07 | fails (+3.9pp) |

### Best single-candidate at modern_v2 FPR ≈ 5%

P11_HEAVY raw, τ=0.905:
- viso 0.334, deeplive 0.576, teams_fake_all 0.716
- modern_v2 FPR 0.0498
- teams_real_all_dev FPR 0.092 (fails 7% cap)

### The binding constraint — AUC ceiling

Best AUC(visomaster fake vs modern_v2 real) across all candidates and all ensembles: **0.647**. To hit 90% TPR at < 5% FPR requires AUC ≈ 0.95. Calibration is rank-preserving — it cannot lift AUC. Noisy-OR / weighted ensembles of same-deficit candidates also do not lift AUC materially (confirmed: ensemble AUC 0.647 vs best-single 0.62–0.65 range).

### The 7%/5% gate conflict

Every operating point that holds modern_v2 FPR ≤ 5% breaks teams_real_all_dev FPR > 7%. This means the **two FPR gates conflict on the current candidate set** — the modern_v2 substrate (281 frames, narrow distribution) and the teams_real_all_dev substrate (4564 frames, broader distribution) are not jointly satisfiable at any τ for any candidate.

### Calibration coefficients (Platt fit, n=7603 each)

| Candidate | Platt a | Platt b |
|---|---|---|
| p8a_reference_step5000 | 0.456 | -0.029 |
| p11_heavy_step1000 | 0.599 | -0.460 |
| p11_heavy_deeplive_step1000 | 0.579 | -0.406 |
| p11_mild_step1000 | 0.568 | -0.300 |
| p11_webcam_harden_step1000 | 0.510 | -0.127 |

(Isotonic also fit; produced same gate-failure pattern.)

### Cache size verification

The subagent flagged a brief discrepancy: `teams_real_all_dev` is n=4564, NOT 1418 as a downstream brief had said. (1418 is `teams_real_all_lockbox`.) Document for any future agent re-running the analysis: `cache_sizes.csv` is authoritative.

### Reads

1. **Substrate ceiling confirmed.** No inference-time tuning on the current 5 candidates clears 90/5. AUC is the binding constraint, not τ.
2. **The recipe direction (HEAVY) is right but step 1000 has the wrong anchor.** The β/γ split in Plan v3 §6 collapses to γ on this substrate — recall families are 15–60pp short.
3. **Visomaster is the long pole.** Best viso recall achievable at modern_v2 FPR ≤ 5% = 33.4%. Even with infinite computational budget on inference-time tricks, this candidate set cannot reach 90% viso recall at any FPR target ≤ 50%.
4. **Plan v4 must redesign training, not inference.** This is what triggered the Plan v4 authorship Day 2 evening.

---

## 2026-04-28 evening — Track H.1: modern_v2 relaxed-filter audit

User approved the H.1 subagent's proposal: drop the `is_pose_extreme` gate and lower `face_area_ratio` 0.10 → 0.0625. Audit script `analysis/modern_v2_audit_2026-04-29/run_audit.py` executed; outputs at `comparison.csv`, `findings.txt`, plus relaxed-substrate yamls.

### Substrate change

| Side | current | relaxed | delta |
|---|---|---|---|
| real frames | 281 | 308 | +27 |
| fake frames | 367 | 390 | +23 |

Capture-mode gate preserved (still drops webcam/screen). Geometry floor 0.0625 (≈ 64×64 face in 256×256 crop) only adds 7 of the 27 real frames; the bulk come from removing the pose-extreme gate (mostly dor_shkedi off-axis normal-photo captures).

### FPR / recall comparison (all 5 deployable candidates)

| Candidate | τ_cur5pct | FPR cur | FPR relax | ΔFPR pp | AUC viso cur | AUC viso relax |
|---|---|---|---|---|---|---|
| p8a_reference_step5000 | 0.400 | 0.0534 | 0.0584 | +0.51 | 0.653 | 0.650 |
| p11_heavy_step1000 | 0.902 | 0.0534 | 0.0584 | +0.51 | 0.633 | 0.627 |
| p11_heavy_deeplive_step1000 | 0.908 | 0.0534 | 0.0584 | +0.51 | 0.580 | 0.574 |
| p11_mild_step1000 | 0.834 | 0.0534 | 0.0584 | +0.51 | 0.628 | 0.622 |
| p11_webcam_harden_step1000 | 0.520 | 0.0534 | 0.0552 | +0.18 | 0.554 | 0.552 |

P11 ensemble (mean of 4) visomaster AUC: 0.6075 cur → 0.6020 relax.

**Verdict (Plan v4 §6 gate):** `filter_confirmed`. max |ΔFPR| at matched-τ = 0.51 pp. The 281-frame substrate is not over-pruning meaningful real-side variation; relaxing the geometry/pose gates does not move FPR materially.

### Reads

1. **The "modern_v2 hides our failures" hypothesis does not hold.** The 27 added frames (mostly dor_shkedi pose-extreme normal-photos) shift FPR by ≤0.66pp at any threshold and ≤0.51pp at the calibrated 5% τ. The current substrate represents the deployable real-side fairly.
2. **The 0.647 AUC ceiling is intrinsic to the model, not the filter.** Visomaster AUC is essentially identical across substrates (0.65 ± 0.005 single-model; 0.61 ± 0.006 ensemble). No data-hygiene relaxation changes the substrate-ceiling story from the Option A configuration sweep.
3. **Recall side improves slightly under relaxation.** Adding 23 fake frames lifts P8A recall +1.7pp at τ_cur5pct (0.6649 → 0.6821) and +2.0pp at τ=0.5. Net positive but not load-bearing.
4. **For Plan v4 / experiment planning: the model side is the entire problem.** Filter-side levers are spent. Track G (P13_ANCHOR_AWARE) and Track I (scratch-retrain) remain the open paths to clearing 90/5.

