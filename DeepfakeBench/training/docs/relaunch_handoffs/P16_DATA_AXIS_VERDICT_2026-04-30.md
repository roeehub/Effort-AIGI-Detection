# Verdict: R13_P16_DATA_AXIS

**Generated**: 2026-05-01 ~00:45 local
**Run**: `rmic6wrc` (W&B project `enhanced-aug-test`) — JOB SUCCEEDED 2026-04-30 16:34 UTC
**Scorecard job**: `8895157920358989824` (us-west4) — JOB SUCCEEDED 2026-04-30 21:09 UTC (4h 39m wall-clock)
**Scorecard outputs (GCS)**: `gs://training-job-outputs/test_results/teams_promotion_contract/r13-p16-data-axis-promotion-20260430-182818/`
**Scorecard mirror (local)**: `analysis/scorecard_p16_data_axis_2026-04-30/`
**Checkpoint map**: `arena/checkpoint_maps/teams_target_domain.r13_p16_data_axis_2026-04-30.yaml`

---

## TL;DR

**P16 does NOT promote under the production contract (target_real_fpr=0.02, target_fake_recall_min=0.70).** No checkpoint — including the P8A_step5000 baseline — meets the contract gates. P8A ranks #1 by a tiny margin; all 8 P16 ckpts rank 2-9.

**At a more diagnostic τ@5%-dev-FPR policy, P16 step 7000 IS net better than P8A** (+7.9pp dev macro fake recall, +26pp deeplive recall) — but **viso recall, the pool the data-axis lever targeted, did not improve** at the operating point (P16 step 7000 viso 3.6% vs P8A 7.1%).

**Pattern (now confirmed 2× in two packets)**:
- mclioexb (P14 face_scale_jitter@0.50 isolated): trainer composite 0.661 → fails contract.
- rmic6wrc (P16 data-axis lever): trainer composite 0.674 → fails contract, *worse* than P8A on lockbox.

Trainer-side `value_composite` continues to be non-predictive of deployment-grade scorecards. Two consecutive single-lever packets failed to crack the cross-domain calibration ceiling at low real-FPR operating points.

**Recommended next direction**: pivot away from data-axis / single-lever augmentation experiments. The binding constraint is **τ-tail separation under a hard real-FPR ceiling**. Future packets should target this directly — margin/contrastive losses, real-fake separation regularizers, or hard-negative mining on the τ-tail viso/deeplive pools.

---

## Production Contract Verdict (target_real_fpr=0.02, target_fake_recall_min=0.70)

`promotion_winner.json` — winner = **P8A_REFERENCE_STEP5000** (rank 1).

| Ckpt | τ | viso↑ | dlive↑ | tfake_dev↑ | tfake_lockbox↑ | dev_real_FPR↓ | macro_fake↑ | rank |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| **gates** | — | ≥0.70 | ≥0.70 | ≥0.70 | — | ≤0.02 | ≥0.70 | — |
| **P8A_REFERENCE_STEP5000** | 0.991 | 0.011 | 0.024 | 0.373 | **0.237** | 0.020 | 0.136 | **1** |
| P16_DATA_AXIS_STEP500 | 0.988 | 0.007 | 0.002 | 0.349 | 0.186 | 0.020 | 0.119 | 2 |
| P16_DATA_AXIS_STEP1500 | 0.983 | 0.009 | 0.046 | 0.333 | 0.158 | 0.020 | 0.129 | 3 |
| P16_DATA_AXIS_STEP2500 | 0.979 | 0.009 | 0.139 | 0.341 | 0.150 | 0.020 | 0.163 | 4 |
| P16_DATA_AXIS_STEP4000 | 0.986 | 0.009 | 0.138 | 0.330 | 0.146 | 0.019 | 0.159 | 5 |
| P16_DATA_AXIS_STEP3000 | 0.978 | 0.009 | 0.139 | 0.329 | 0.146 | 0.020 | 0.159 | 6 |
| P16_DATA_AXIS_STEP5500 | 0.811 | 0.009 | 0.161 | 0.330 | 0.146 | 0.020 | 0.167 | 7 |
| P16_DATA_AXIS_STEP6000 | 0.694 | 0.009 | 0.171 | 0.328 | 0.146 | 0.020 | 0.169 | 8 |
| P16_DATA_AXIS_STEP7000 | 0.668 | 0.009 | 0.174 | 0.330 | 0.146 | 0.020 | 0.171 | 9 |

**Observations:**
1. **No ckpt approaches the 70% recall floor** on any pool except the unknown_fake / teams_capture group inside teams_fake_all_dev (which is ~98% raw at τ=0.5 and not the bottleneck).
2. **viso pool is severely τ-collapsed** for every ckpt (0.7-1.1%) — the bucket gap is not the binding issue at this τ tail.
3. **P16 hurt lockbox fake recall by ~9pp** (P8A 0.237 → P16 best 0.186, P16 worst 0.146). The data-axis lever generalized to dev pools but **not** to held-out lockbox.
4. **τ ranges from 0.668 (P16 step 7000) to 0.991 (P8A)**. P16's later ckpts have qualitatively different τ behavior — recall is moving more steeply with τ — but that doesn't help under the hard 2% FPR ceiling.

---

## Diagnostic Calibration: τ@5%-dev-FPR

Re-calibrating to the looser 5%-FPR policy (sanity check that the verdict isn't an artifact of the strict 2% gate):

| Ckpt | τ@5% | viso↑ | dlive↑ | tfake_dev↑ | dev_macro↑ |
|---|---:|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 0.967 | **0.071** | 0.145 | 0.466 | 0.227 |
| P16_DATA_AXIS_STEP500 | 0.986 | 0.025 | 0.092 | 0.421 | 0.179 |
| P16_DATA_AXIS_STEP1500 | 0.964 | 0.047 | 0.246 | 0.452 | 0.248 |
| P16_DATA_AXIS_STEP2500 | 0.944 | 0.056 | 0.380 | 0.491 | 0.309 |
| P16_DATA_AXIS_STEP3000 | 0.947 | 0.040 | 0.349 | 0.469 | 0.286 |
| P16_DATA_AXIS_STEP4000 | 0.963 | 0.044 | 0.343 | 0.469 | 0.285 |
| P16_DATA_AXIS_STEP5500 | 0.750 | 0.035 | 0.367 | 0.466 | 0.289 |
| P16_DATA_AXIS_STEP6000 | 0.651 | 0.036 | 0.387 | 0.470 | 0.298 |
| **P16_DATA_AXIS_STEP7000** | **0.623** | 0.036 | **0.406** | **0.477** | **0.306** |

**Observations under τ@5%:**

- **P16 step 7000 wins macro by +7.9pp vs P8A** (0.306 vs 0.227). At a more practical operating point, P16 IS a real improvement.
- **deeplive recall lifted +26pp** (0.145 → 0.406) — P16 is dramatically better on deeplive at this τ.
- **teams_fake_dev lifted modestly** (0.466 → 0.477).
- **viso recall actually regressed** (-3.5pp): 0.071 → 0.036. The pool the data-axis lever was supposed to target is **worse** than baseline at this calibration.

This is the load-bearing diagnostic finding: the visomaster_teams_enhanced fw=2.0 lever shifted bulk of viso confidence at τ=0.5 (raw recall 35.6% → 54.6% at step 1500) but **did not improve viso separation in the high-precision tail**. Whatever changed in the model's τ-tail behavior produced a deeplive transfer bonus, not a viso lift.

---

## τ=0.5 Readout (no calibration; for reference)

Per-suite raw recall at default τ=0.5:

| Ckpt | viso | dlive | tfake_dev | teams_real_dev_FPR |
|---|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 0.356 | 0.530 | n/a | 0.121 |
| P16_DATA_AXIS_STEP1500 | **0.546** | 0.943 | 0.875 | 0.203 |
| P16_DATA_AXIS_STEP6000 | 0.516 | **0.969** | 0.878 | 0.235 |
| P16_DATA_AXIS_STEP7000 | 0.378 | 0.941 | 0.799 | 0.201 |

P16 step 6000 hits 96.9% raw deeplive recall — a massive lift over P8A's 53%. Raw viso lift is less dramatic (35.6 → 54.6%) and concentrated at step 1500. **Real-FPR cost is +5-10pp** vs P8A at default τ.

---

## Trainer-Side Metrics (for context only — non-predictive)

W&B run `rmic6wrc`:
- `value_composite`: 0.674 (best in series — vs mclioexb 0.661, vs in-tree top recipes ~0.5-0.6)
- `val_primary/best_metric`: 0.99637 at epoch 3 (~step 7000)
- `val_holdout` peak AUC: 0.9964 step 7000; best EER 0.0167 step 1500
- Anchor composite: positive flip step 1500 (+0.0175); peaked +0.1679 epoch 2; sustained +0.04-0.05 late epochs
- Held-out target-fake recall (deeplive_teams): 100/100/100 at step 1500; slipped to 83% by step 10500 (mild rolling overfit)

The trainer composite improved over mclioexb yet the contract verdict is *worse* than mclioexb on lockbox-vs-baseline. **Trainer composite is a leading indicator with the wrong sign for deployment-grade calibration.**

---

## Why The Data-Axis Lever Didn't Crack The Contract

Three interacting factors:

1. **τ-tail collapse under hard real-FPR ceiling.** The model's probability separation is poor in the high-confidence regime required for ≤2% real FPR. Both raw recalls and 5%-FPR calibrated recalls show that improvements at default τ don't translate to the contract operating point. This is a **calibration / margin** problem, not a representation problem (frame-level AUC was 0.99+ on dev).

2. **Lockbox identity gap.** P16 hurt lockbox fake recall (-9pp vs P8A) while improving dev macro at τ@5%. The visomaster_teams_enhanced bucket fw=2.0 weighting biased the model toward dev-set distributions in a way that didn't transfer to lockbox identities.

3. **Viso bucket gap unchanged at the τ-tail.** Despite directly oversampling visomaster_teams_enhanced, the high-confidence viso fakes didn't move. Memory `project_move1_bucket_gap_refuted.md` already established that viso bucket discriminability is fine (probe AUC 0.92+); the bucket gap manifests as **shortcut on reals**, not viso non-discriminability. Oversampling the viso bucket can't close a real-side shortcut.

---

## Recommended Next Direction

**Stop pursuing single-lever data-axis / augmentation packets.** Two consecutive failures with stronger trainer-side composites both lost the contract:
- mclioexb (P14 jitter@0.50 isolated): trainer 0.661 → fails contract
- rmic6wrc (P16 data-axis fw=2.0 isolated): trainer 0.674 → fails contract, regresses on lockbox

The cross-domain calibration ceiling at low real-FPR is the binding constraint. Candidate directions for P17+:

1. **τ-tail separation regularizer**: contrastive / margin loss between high-confidence reals and high-confidence fakes (push apart the τ-tail directly).
2. **Hard-negative mining on viso bucket**: target the actual viso fakes that score below τ-tail, not the bucket distribution as a whole.
3. **Feature-level decomposition**: `analysis/intermediate_layer_probe_2026-04-30/` was authored as part of the P16 launch package to disambiguate which layers move during FT. Run this to inform whether the next intervention should target backbone layers or just the head.
4. **Anchor-aware loss revisit**: P14 explored anchor-aware loss in a bundle that turned out net-negative; the *isolated* anchor-aware ablation has not been run.

User decision point: pick one of the four above (or propose alternative). All four require structural rather than data-shape changes, which is the load-bearing pivot.

---

## Files To Know

| Path | What it is |
|---|---|
| `analysis/scorecard_p16_data_axis_2026-04-30/promotion_winner.json` | Contract winner (P8A) + per-ckpt selected_threshold + macro_fake recall. |
| `analysis/scorecard_p16_data_axis_2026-04-30/checkpoint_summary.csv` | One row per ckpt with all contract metrics. |
| `analysis/scorecard_p16_data_axis_2026-04-30/selected_threshold_scorecard.csv` | Per-suite metrics at the contract-selected τ. |
| `analysis/scorecard_p16_data_axis_2026-04-30/threshold_grid.csv` | Full τ-grid (5400-5600 candidates per ckpt) — source for τ@5% diagnostic. |
| `arena/checkpoint_maps/teams_target_domain.r13_p16_data_axis_2026-04-30.yaml` | Resolved GCS paths for all 9 ckpts. |
| `experiments/phase2_round13/R13_P16_DATA_AXIS.yaml` | Training config (single-lever discipline, anti-shortcut bundle DISABLED, fw=2.0). |
| `analysis/intermediate_layer_probe_2026-04-30/intermediate_layer_probe.py` | Authored, not yet run. Hooks resblocks [0,3,6,9,11] of EffortDetector backbone. Useful for the "where did FT actually move things" question. |

---

## Cost Summary (P16 packet end-to-end)

| Item | Cost | Wall-clock |
|---|---:|---:|
| Cloud Build image rebuild (1.3.234, 1.3.235) | ~$2 | ~5 min |
| Vertex training (rmic6wrc, A100 us-west4) | ~$70 | 4h |
| Vertex scorecard (8895157920358989824) | ~$15 | 4h 39m |
| **Total** | **~$87** | ~9h |

---

## Memory Updates (logged)

- `project_p16_data_axis_does_not_promote_2026-04-30.md` — verdict + numbers + pattern (added).
- Existing: `project_mclioexb_does_not_promote_2026-04-30.md` (sibling; same pattern).
- Existing: `project_promotion_contract.md` (load-bearing: trainer composite ≠ deployment).
- Existing: `project_face_scale_jitter_load_bearing.md` (P14 packet result).
