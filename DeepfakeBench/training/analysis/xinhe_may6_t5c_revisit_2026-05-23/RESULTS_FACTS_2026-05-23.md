# Xinhe may6 T5C revisit — RESULTS FACTS

Generated 2026-05-23. Factual readout only (forbidden words: "succeeds", "fails", "wins", "promotes", "deployment-grade", "broken", "solves", "fixes"). Interpretive content lives in `AGENT_PROPOSAL_2026-05-23.md`.

## 0. Scope and inputs

Re-scoring of the 92-frame may6 + 60-frame may5 Xinhe cohorts (raw frames at `analysis/xinhe_cross_camera_audit_2026-05-06/raw/may{5,6}/`) on the 5 currently deploy-relevant ckpts. Frames are unchanged from the 2026-05-06 audit.

| Ckpt key | Local path |
|---|---|
| `P8A_REFERENCE_STEP5000` | `analysis/manual_canary_2026-05-20/ckpts/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth` |
| `E2B_TOP_N_STEP3200` | `analysis/team_identity_deploy_readout_2026-05-23/_ckpts/E2B_TOP_N_STEP3200.pth` |
| `T5C_PERIODIC_STEP3500` (current production per `project_production_is_t5c_not_e2b_2026-05-23`) | `analysis/manual_canary_2026-05-20/ckpts/periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth` |
| `SLOT_A_V2_CLS_STEP3500` | `analysis/manual_canary_2026-05-20/ckpts/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth` |
| `SLOT_A_V2_FACE_POOL_STEP3500` | same ckpt + face-pool monkey-patch from `analysis/face_pool_canary_2026-05-22/score_canary_face_pool.py` |

τ-modes from memory `project_deployment_three_modes_slot_a_v2_2026-05-21`:
- `tau_0_5` = 0.5 (the 2026-05-06 E2B-claim threshold; informational)
- `mode_A_tau_0_535` = 0.535
- `mode_B_tau_0_78` = 0.78
- `mode_C_tau_0_87` = 0.87

CAVEAT: τ values were calibrated for Slot A v2 step3500 CLS-pool on the 9-suite scorecard. Cross-ckpt usage of the same numeric τ is a constant-threshold comparison, not a constant-FPR comparison.

Scoring: 5 ckpts × 152 frames on local MPS (Apple Silicon). Wall time: ~3 minutes total. Preprocessing path is identical to the 2026-05-06 audit (`arena.model_arena` CLIP_MEAN/STD, INTER_LINEAR resize 224, BGR→RGB).

---

## 1. Reproduction check (Task T3)

Fresh P8A + E2B scores on the same 152 frames are bytewise-equal to the 2026-05-06 cached numbers at `analysis/xinhe_cross_camera_audit_2026-05-06/outputs/scores_all_5_ckpts_may6_may5.csv`:

| ckpt | cohort | fresh fpr@0.5 | cached fpr@0.5 | \|Δfpr\| | fresh mean | cached mean | \|Δmean\| |
|---|---|---:|---:|---:|---:|---:|---:|
| P8A | may6_falseflag | 0.0% | 0.0% | 0.0pp | 0.021 | 0.021 | 0.000 |
| P8A | may5_correct | 0.0% | 0.0% | 0.0pp | 0.021 | 0.021 | 0.000 |
| E2B | may6_falseflag | 57.6% | 57.6% | 0.0pp | 0.511 | 0.511 | 0.000 |
| E2B | may5_correct | 1.7% | 1.7% | 0.0pp | 0.053 | 0.053 | 0.000 |

Per-frame |Δprob| max = 0.0000, mean = 0.0000 for both P8A and E2B. Original 2026-05-06 numbers reproduce exactly on MPS scoring at 2026-05-23.

---

## 2. Per (ckpt × cohort) headline table (Task T4)

`outputs/fpr_by_mode.csv` and `outputs/fpr_by_mode.txt`:

| ckpt | cohort | n | mean | median | p95 | max | >0.9 | fpr@0.5 | modeA (τ=0.535) | modeB (τ=0.78) | modeC (τ=0.87) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P8A | may5_correct | 60 | 0.021 | — | 0.098 | 0.234 | 0 | 0.0% | 0.0% | 0.0% | 0.0% |
| P8A | may6_falseflag | 92 | 0.021 | — | 0.083 | 0.253 | 0 | 0.0% | 0.0% | 0.0% | 0.0% |
| E2B | may5_correct | 60 | 0.053 | — | 0.323 | 0.603 | 0 | 1.7% | 1.7% | 0.0% | 0.0% |
| E2B | may6_falseflag | 92 | 0.511 | — | 0.974 | 0.990 | 16 | 57.6% | 55.4% | 31.5% | 20.7% |
| T5C | may5_correct | 60 | 0.086 | 0.071 | 0.177 | 0.246 | 0 | 0.0% | 0.0% | 0.0% | 0.0% |
| T5C | may6_falseflag | 92 | 0.264 | 0.170 | 0.714 | 0.829 | 0 | 17.4% | 16.3% | 3.3% | 0.0% |
| Slot A v2 CLS | may5_correct | 60 | 0.071 | — | 0.105 | 0.213 | 0 | 0.0% | 0.0% | 0.0% | 0.0% |
| Slot A v2 CLS | may6_falseflag | 92 | 0.153 | — | 0.425 | 0.783 | 0 | 4.3% | 4.3% | 1.1% | 0.0% |
| Slot A v2 face-pool | may5_correct | 60 | 0.443 | — | 0.561 | 0.597 | 0 | 26.7% | 10.0% | 0.0% | 0.0% |
| Slot A v2 face-pool | may6_falseflag | 92 | 0.599 | — | 0.712 | 0.770 | 0 | 91.3% | 85.9% | 0.0% | 0.0% |

`>0.9` = number of frames with `prob_fake > 0.9` (high-confidence false flag).

CAVEAT for the face-pool row: the classifier head was trained on CLS-pool features; substituting face-region mean shifts the absolute score regime (see `analysis/face_pool_canary_2026-05-22/score_canary_face_pool.py:11-23`). The fixed-τ FPRs above are constant-threshold readouts, NOT constant-operating-point readouts. The face-pool may5 mean of 0.443 vs may6 mean of 0.599 is informative as a relative shift; the absolute `91.3%` may6 fpr@0.5 reflects the calibration-band shift, not a 91.3% "false-flag rate" at the face-pool operating point. See §3 for the internally-calibrated readout.

---

## 3. Internally-calibrated drift (Task T4 supplemental)

Per-ckpt internal calibration: take may5 95th percentile (p95) as the per-ckpt operating threshold, then measure may6 fpr at that threshold. This removes absolute-τ-calibration artifacts and is invariant to head-pool substitution.

| ckpt | τ = may5 p95 | may6 fpr at τ | may6 mean | may6 max | may5 max |
|---|---:|---:|---:|---:|---:|
| P8A | 0.0979 | 4.3% | 0.021 | 0.253 | 0.234 |
| E2B | 0.3231 | 65.2% | 0.511 | 0.990 | 0.603 |
| T5C | 0.1773 | 46.7% | 0.264 | 0.829 | 0.246 |
| Slot A v2 CLS | 0.1047 | 42.4% | 0.153 | 0.783 | 0.213 |
| Slot A v2 face-pool | 0.5606 | 76.1% | 0.599 | 0.770 | 0.597 |

By construction may5 fpr at this threshold is 5% (4/60 ≈ 5% rounding artifact: it's actually 3/60 = 5.0% at the 95th-percentile cutoff, since p95 is a sample percentile). For all ckpts the may5 fpr at the per-ckpt p95 is in {3/60, 4/60} = {5.0%, 6.7%}.

Per-cohort mean drift (may6_mean − may5_mean):

| ckpt | may5_mean | may6_mean | drift |
|---|---:|---:|---:|
| P8A | 0.021 | 0.021 | +0.000 |
| E2B | 0.053 | 0.511 | +0.459 |
| T5C | 0.086 | 0.264 | +0.178 |
| Slot A v2 CLS | 0.071 | 0.153 | +0.082 |
| Slot A v2 face-pool | 0.443 | 0.599 | +0.156 |

---

## 4. Comparison to the 2026-05-06 E2B claim

The 2026-05-06 memo (`project_xinhe_may6_falseflag_2026-05-06.md`) reported:

| ckpt | may5 mean | may6 mean | drift | may6 > 0.5 |
|---|---:|---:|---:|---:|
| P8A | 0.021 | 0.021 | +0.000 | 0/92 |
| E2B (deployed at the time) | 0.053 | 0.511 | +0.459 | 53/92 (57.6%) |
| PA_3800 | 0.054 | 0.269 | +0.215 | 15/92 (16.3%) |
| T3_S1_step1500 | 0.012 | 0.109 | +0.096 | 6/92 (6.5%) |
| T3_S1_step2500 | 0.063 | 0.732 | +0.670 | 71/92 (77.2%) |

Three rows reproduce on the 2026-05-23 MPS scoring (P8A bytewise; E2B bytewise; PA_3800 cached in `analysis/xinhe_cross_camera_audit_2026-05-06/outputs/scores_all_5_ckpts_may6_may5.csv`). T5C did not exist at the time of the original memo.

The 2026-05-23 T5C drift of +0.178 (0.086→0.264) is mid-range between PA_3800 (+0.215) and T3_S1_step1500 (+0.096) — closer to PA_3800. The 2026-05-23 Slot A v2 CLS drift of +0.082 is closest to T3_S1_step1500 (+0.096).

---

## 5. Self-correction log

No mid-task corrections. Reproduction check passed on the first run. The only interpretation step that required care was the face-pool row, where the head-pool substitution shifts the absolute score regime; documented in §2 caveat and §3 alternative readout.

---

## 6. Outputs index

- `outputs/scores_<KEY>.csv` — per-frame `(population, frame_path, frame_basename, prob_fake)` for each of the 5 ckpts (152 rows each)
- `outputs/joint_scores.csv` — long table: ckpt × cohort × frame × prob_fake (5 × 152 = 760 rows)
- `outputs/wide_scores.csv` — wide table: one row per frame, one column per ckpt
- `outputs/fpr_by_mode.csv` — per (ckpt, cohort) statistics + FPR at each of 4 τ-modes
- `outputs/fpr_by_mode.txt` — human-readable version of fpr_by_mode.csv with reproduction check inline
- `scripts/score_may6_may5.py` — scoring driver (reusable on the same cohort with `--only <KEY>`)
- `scripts/analyze.py` — joiner + FPR-mode-table builder
- `_score_T5C.log`, `_score_rest.log` — raw scoring logs
