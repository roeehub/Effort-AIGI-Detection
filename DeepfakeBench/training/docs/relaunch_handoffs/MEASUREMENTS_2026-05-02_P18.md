# P18 Measurements — pure data, no interpretation

**Date**: 2026-05-02
**Branch**: `teams-relaunch-root-2026-04-17`
**Scope**: All measurements from this session's diagnostics + Vertex contract scorecard for P18 (treatment `xpbvc1e4`, control `rgt4kw2u`) and P8A (`9lmvb5b4`) baseline.

This document contains **only numbers and the methods that produced them**. It deliberately omits interpretation, framing, and recommendations. For interpretation, see the dated `HANDOFF_2026-05-02_*` documents.

---

## 0. Subjects under test

| Label | W&B run | Vertex job | Local ckpt | GCS ckpt | ArcFace s |
|---|---|---|---|---|---:|
| **P8A** | `9lmvb5b4` | n/a | `analysis/_features_cache_2026-04-30/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth` | `gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth` | 9.749 |
| **P18T (treatment)** | `xpbvc1e4` | `820959496569356288` | `/tmp/p18_ckpts/xpbvc1e4__periodic_step4000.pth` | `gs://training-job-outputs/phase2r13_experiments/xpbvc1e4/periodic_effort_20260501_step4000_auc0.9907_eer0.0279.pth` | 12.000 |
| **P18C (control)** | `rgt4kw2u` | `456167926752346112` | `/tmp/p18_ckpts/rgt4kw2u__periodic_step4000.pth` | `gs://training-job-outputs/phase2r13_experiments/rgt4kw2u/periodic_effort_20260501_step4000_auc0.9871_eer0.0330.pth` | 12.000 |

P18 yaml: `experiments/phase2_round13/R13_P18_METHOD_DOMAIN_GRL.yaml` (treatment), `experiments/phase2_round13/R13_P18_NO_GRL_CONTROL.yaml` (control).
ArcFace `s` annealed `s_start=6.0 → s_end=12.0` over `anneal_steps`. P8A used `anneal_steps=8000`, captured at step 5000 (s = 6 + 5000/8000 × 6 = 9.75). P18T/C used `anneal_steps=4000`, captured at step 4000 (s = 12.0).

---

## 1. Diagnostic D — Vertex promotion contract scorecard

**Vertex job**: `projects/700371397073/locations/us-east1/customJobs/4550151094264659968`
**Image**: `1.3.241`
**Region**: `us-east1`
**Submitted**: 2026-05-02 08:31Z
**RUNNING**: 08:35Z
**SUCCEEDED**: 10:04Z (1h33m)
**Suite manifest**: `arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml`
**Checkpoint map**: `arena/checkpoint_maps/teams_target_domain.p18_corrective_2026-05-02.yaml`
**Output GCS**: `gs://training-job-outputs/test_results/teams_promotion_contract/p18-corrective-contract-20260502-103127/`
**Local pull**: `analysis/p18_probe_2026-05-01/d_results/`

### 1.1 Contract config used

```
target_fake_recall_min:    0.7
target_real_fpr:           0.02
target_stress_fpr:         0.05

dev_fake_suites:           [teams_fake_all_dev, visomaster_enhanced_macro_dev, deeplive_enhanced_dev]
dev_real_stress_suites:    [teams_real_poor_quality_dev, teams_real_lighting_extreme_dev]
dev_real_suite:            teams_real_all_dev
lockbox_fake_suite:        teams_fake_all_lockbox
lockbox_real_suite:        teams_real_all_lockbox
```

### 1.2 checkpoint_summary.csv (selected τ per arm)

| checkpoint_key | promotion_rank | selected_threshold | dev_primary_real_fpr | dev_worst_real_stress_fpr | dev_fake_macro_recall | lockbox_real_n | lockbox_real_fpr | lockbox_fake_n | lockbox_fake_recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 1 | 0.990946 | 0.019982 | 0.015703 | 0.135844 | 1361 | 0.001470 | 253 | 0.237154 |
| P18T_GRL_TREATMENT_STEP4000 | 2 | 0.993590 | 0.019982 | 0.029979 | 0.150550 | 1361 | 0.002939 | 253 | 0.177866 |
| P18C_NO_GRL_CONTROL_STEP4000 | 3 | 0.994625 | 0.019674 | 0.034975 | 0.179269 | 1361 | 0.008082 | 253 | 0.173913 |

Per-fake-suite recalls (at the selected τ):

| arm | teams_fake_all_dev (n=2409) | visomaster_enhanced_macro_dev (n=550) | deeplive_enhanced_dev (n=545) |
|---|---:|---:|---:|
| P8A | 0.372769 | 0.010909 | 0.023853 |
| P18T | 0.358240 | 0.018182 | 0.075229 |
| P18C | 0.328767 | 0.014545 | 0.194495 |

### 1.3 selected_threshold_scorecard.csv — per-suite REAL FPR at selected τ

| suite | n_real | P8A | P18T | P18C |
|---|---:|---:|---:|---:|
| teams_real_all_dev | 3253 | 0.019982 | 0.019982 | 0.019674 |
| teams_real_poor_quality_dev | 923 | 0.003250 | 0.002167 | 0.002167 |
| teams_real_lighting_extreme_dev | 1401 | 0.015703 | 0.029979 | 0.034975 |
| teams_real_all_lockbox | 1361 | 0.001470 | 0.002939 | 0.008082 |
| teams_real_dor_dev | 50 | 0.000 (fp=0/50) | 0.040 (fp=2/50) | 0.040 (fp=2/50) |

### 1.4 selected_threshold_scorecard.csv — per-suite FAKE RECALL at selected τ

| suite | n_fake | P8A | P18T | P18C |
|---|---:|---:|---:|---:|
| teams_fake_all_dev | 2409 | 0.372769 (898 of 2409) | 0.358240 (862 of 2409) | 0.328767 (792 of 2409) |
| visomaster_enhanced_macro_dev | 550 | 0.010909 (6 of 550) | 0.018182 (10 of 550) | 0.014545 (8 of 550) |
| deeplive_enhanced_dev | 545 | 0.023853 (13 of 545) | 0.075229 (41 of 545) | 0.194495 (106 of 545) |
| teams_fake_all_lockbox | 253 | 0.237154 (60 of 253) | 0.177866 (45 of 253) | 0.173913 (44 of 253) |

### 1.5 threshold_grid.csv cross-section at fixed dev_primary_real_fpr ≈ 0.07

(Selected per-arm row from 16,481-row `threshold_grid.csv` minimizing `|dev_primary_real_fpr − 0.07|`.)

| arm | τ | dev_primary_real_fpr | dev_worst_real_stress_fpr | dev_fake_macro_recall | teams_fake_all_dev recall | visomaster_enhanced_macro_dev recall | deeplive_enhanced_dev recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| P8A | 0.913585 | 0.070089 | 0.069950 | 0.300418 | 0.526359 | 0.136364 | 0.238532 |
| P18T | 0.951889 | 0.070089 | 0.089222 | 0.333610 | 0.531756 | 0.070909 | 0.398165 |
| P18C | 0.972216 | 0.070089 | 0.092791 | 0.430281 | 0.599834 | 0.080000 | 0.611009 |

(`threshold_grid.csv` does NOT contain lockbox columns — lockbox is held out and only scored at the contract-selected τ from §1.2.)

---

## 2. CPU-only diagnostics

All scripts under `analysis/p18_probe_2026-05-01/`. Outputs under `outputs/`.
Eval substrate: `analysis/embedding_triptych_2026-04-30/outputs/triptych_p8a_slot2_slot3/sampled_frames.csv` (first 800 rows; all have local cached crops).
Counts in this substrate: 800 frames total = 87 lockbox + 713 dev = 324 fake + 476 real. Lockbox split: 40 reals (all `teams_real`) + 47 fakes (36 `teams_capture_cam_test_s33` + 11 `teams_capture_pc_generator_s15`); 0 lockbox `deeplive_enhanced` in this substrate.
Within-substrate dor_shkedi counts: 9 in bucket-3 fake/real (used by within-bucket LR), 25 lockbox dor reals, 15 lockbox other-Teams reals.

### 2.1 Diagnostic G — L3 hook fidelity (`verify_l3_hook.py`)

Compared L3 features extracted via the forward hook in `corrective_probes.py:extract_l3_features` to cached features in `analysis/_features_cache_2026-04-30/intermediate__P8A__layer03__n800.npz`, on 64 frames of P8A.

| metric | value |
|---|---|
| max-of-max abs diff per-frame | 1.91e-06 |
| mean-of-max abs diff per-frame | 7.92e-07 |
| median-of-max abs diff per-frame | 8.94e-07 |
| fraction of frames with max abs diff < 1e-3 | 1.000 |
| fraction of frames with max abs diff < 1e-4 | 1.000 |
| min cosine similarity to cached features | 1.000000 |
| mean cosine similarity to cached features | 1.000000 |

### 2.2 Diagnostic F — corrective_probes.py on P8A baseline

Output: `outputs/corrective_probes__P8A_step5000_FINAL.json`

| metric | P8A | P18T | P18C |
|---|---:|---:|---:|
| Fresh-LR L3 dev→lockbox AUC | 0.9495 | 0.9404 | 0.9527 |
| Within-bucket-3 dor-vs-other LR @ FINAL [CLS] AUC (StandardScaler full-set) | 0.9826 | 0.9536 | 0.9965 |
| Within-bucket-3 dor-vs-other LR @ L3 AUC (l2norm row) | 0.8194 | 0.8194 | 0.8194 |
| cos(fresh-LR direction, is_lockbox) | +0.5503 | +0.5487 | +0.5524 |
| cos(fresh-LR direction, is_webcam) | −0.1706 | −0.1730 | −0.1576 |
| cos(fresh-LR direction, is_dor_shkedi) | +0.2898 | +0.2862 | +0.2891 |
| cos(fresh-LR direction, is_deeplive_enhanced) | +0.2431 | +0.2476 | +0.2425 |
| cos(fresh-LR direction, is_teams_capture) | +0.6767 | +0.6740 | +0.6752 |
| Lockbox dor real (n=25) mean prob_fake (no s-scaling) | 0.4385 | 0.4940 | 0.5492 |
| Lockbox dor real (n=25) median prob_fake | 0.4368 | 0.4917 | 0.5508 |
| Lockbox other-Teams real (n=15) mean prob_fake | 0.5010 | 0.5149 | 0.5353 |
| Lockbox dor real FPR @ τ=0.5 | 0.12 | 0.40 | 0.92 |
| Lockbox other-Teams real FPR @ τ=0.5 | 0.5333 | 0.6000 | 0.8000 |
| Lockbox dor real FPR @ τ=0.92 (no s-scaling) | 0.0 | 0.0 | 0.0 |
| Lockbox dor real FPR @ τ=0.974 (no s-scaling) | 0.0 | 0.0 | 0.0 |

### 2.3 Diagnostic A — Bootstrap CIs on within-bucket-3 LR AUC @ CLS

Method: 5-fold StratifiedKFold OOF predictions; bootstrap 1000 iters resampling `(oof_proba, y)` pairs with replacement; report 2.5%/97.5% percentiles. n_dor=9, n_other=256.

| arm | point AUC | 95% CI lo | 95% CI hi | std | n_valid_iters |
|---|---:|---:|---:|---:|---:|
| P8A | 0.9939 | 0.9776 | 1.0000 | 0.0066 | 1000 |
| P18T | 0.9679 | 0.8964 | 1.0000 | 0.0295 | 1000 |
| P18C | 0.9996 | 0.9974 | 1.0000 | 0.0008 | 1000 |

Paired delta CIs (same resampled indices each iter):

| pair | point Δ | 95% CI lo | 95% CI hi | zero in CI |
|---|---:|---:|---:|:---:|
| P18T − P18C | −0.0317 | −0.1036 | +0.0000 | yes |
| P18T − P8A | −0.0260 | −0.0803 | +0.0000 | yes |
| P18C − P8A | +0.0056 | −0.0019 | +0.0224 | yes |

### 2.4 Diagnostic B — per-frame paired Wilcoxon signed-rank tests

n_dor_lockbox_real = 25; n_nondor_teams_lockbox_real = 15.

**Unscaled** scores (sigmoid(margin), no s):

| subset | pair | median Δ p_fake (a−b) | Wilcoxon W | p (two-sided) | n |
|---|---|---:|---:|---:|---:|
| dor | P18T − P18C | −0.0577 | 0.0 | < 0.0001 | 25 |
| dor | P18T − P8A | +0.0590 | 1.0 | < 0.0001 | 25 |
| dor | P18C − P8A | +0.1139 | 0.0 | < 0.0001 | 25 |
| nondor_teams | P18T − P18C | −0.0205 | 15.0 | 0.0084 | 15 |
| nondor_teams | P18T − P8A | −0.0022 | 43.0 | 0.3591 | 15 |
| nondor_teams | P18C − P8A | +0.0316 | 31.0 | 0.1070 | 15 |

**s-scaled** scores (sigmoid(s × margin), s = 9.749 for P8A, 12.000 for P18T/C):

| subset | pair | median Δ p_fake (a−b) | Wilcoxon W | p (two-sided) | n |
|---|---|---:|---:|---:|---:|
| dor | P18T − P18C | −0.3872 | 0.0 | < 0.0001 | 25 |
| dor | P18T − P8A | +0.3095 | 0.0 | < 0.0001 | 25 |
| dor | P18C − P8A | +0.7616 | 0.0 | < 0.0001 | 25 |
| nondor_teams | P18T − P18C | −0.0414 | 22.0 | 0.0302 | 15 |
| nondor_teams | P18T − P8A | +0.0178 | 33.0 | 0.1354 | 15 |
| nondor_teams | P18C − P8A | +0.0369 | 29.0 | 0.0833 | 15 |

### 2.5 Diagnostic C — ArcFace s-scaling and FPR @ deployment τ

`prob_fake = sigmoid(s × (cos(fake) − cos(real)))`. Cosines computed via head.weight l2-norm × CLS-feature l2-norm dot product. s from §0 above. Mask: lockbox reals.

| arm | s | subset | n | mean_prob unscaled | mean_prob scaled | FPR @ τ=0.5 unscaled | FPR @ τ=0.5 scaled | FPR @ τ=0.92 unscaled | FPR @ τ=0.92 scaled | FPR @ τ=0.974 unscaled | FPR @ τ=0.974 scaled |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P8A | 9.75 | dor lockbox real | 25 | 0.4385 | 0.1536 | 0.12 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 |
| P8A | 9.75 | nondor Teams lockbox real | 15 | 0.5010 | 0.5250 | 0.53 | 0.53 | 0.00 | 0.33 | 0.00 | 0.13 |
| P18T | 12.00 | dor lockbox real | 25 | 0.4940 | 0.4410 | 0.40 | 0.40 | 0.00 | 0.00 | 0.00 | 0.00 |
| P18T | 12.00 | nondor Teams lockbox real | 15 | 0.5149 | 0.5932 | 0.60 | 0.60 | 0.00 | 0.20 | 0.00 | 0.07 |
| P18C | 12.00 | dor lockbox real | 25 | 0.5492 | 0.8390 | 0.92 | 0.92 | 0.00 | 0.52 | 0.00 | 0.16 |
| P18C | 12.00 | nondor Teams lockbox real | 15 | 0.5353 | 0.7466 | 0.80 | 0.80 | 0.00 | 0.20 | 0.00 | 0.20 |

### 2.6 Diagnostic E — within-bucket-3 dor-vs-other LR at intermediate layers (l2norm row, 5-fold OOF)

| layer | P8A AUC | P8A 95% CI | P18T AUC | P18T 95% CI | P18C AUC | P18C 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| L3 | 0.8194 | [0.7586, 0.8816] | 0.8194 | [0.7586, 0.8816] | 0.8194 | [0.7586, 0.8816] |
| L6 | 1.0000 | [1.0000, 1.0000] | 1.0000 | [1.0000, 1.0000] | 1.0000 | [1.0000, 1.0000] |
| L9 | 1.0000 | [1.0000, 1.0000] | 1.0000 | [1.0000, 1.0000] | 0.9952 | [0.9862, 1.0000] |
| L11 | 0.9635 | [0.8867, 1.0000] | 0.9440 | [0.8132, 1.0000] | 0.9931 | [0.9818, 1.0000] |

Per-layer paired delta CIs (same resampled indices each iter, all show "zero in CI"):

| layer | P18T − P18C Δ | 95% CI | P18T − P8A Δ | 95% CI | P18C − P8A Δ | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| L3 | +0.0000 | [−0.0000, +0.0000] | +0.0000 | [−0.0000, +0.0000] | +0.0000 | [−0.0000, +0.0000] |
| L6 | +0.0000 | [−0.0000, +0.0000] | +0.0000 | [−0.0000, +0.0000] | +0.0000 | [−0.0000, +0.0000] |
| L9 | +0.0048 | [+0.0000, +0.0138] | +0.0000 | [−0.0000, +0.0000] | −0.0048 | [−0.0138, +0.0000] |
| L11 | −0.0490 | [−0.1694, +0.0103] | −0.0195 | [−0.0690, +0.0062] | +0.0295 | [−0.0097, +0.1026] |

### 2.7 Mini-scorecard on full eval substrate (`run_minicard.py`)

**Substrate inventory**:
- 800 frames total; 87 lockbox + 713 dev; 324 fake + 476 real; 40 lockbox reals + 47 lockbox fakes
- Methods (top): teams_real n=476 (lockbox 40), deeplive_enhanced n=59 (lockbox 0), teams_capture_cam_test_s35 n=41 (lockbox 0), teams_capture_noyn_sharker_s23 n=36 (lockbox 0), teams_capture_cam_test_s33 n=36 (lockbox 36), teams_capture_pc_generator_s15 n=11 (lockbox 11)
- Capture modes: normal_photo n=264, phone_screen n=255, webcam n=238, screen n=27, screen_recording n=16

**Per-method lockbox real FPR (n_lockbox_real_per_method)** at scaled τ:

| method | n | τ=0.5 | τ=0.92 | τ=0.974 |
|---|---:|---|---|---|
| teams_real | 40 | P8A 0.28 / P18T 0.47 / P18C 0.88 | P8A 0.12 / P18T 0.07 / P18C 0.40 | P8A 0.05 / P18T 0.03 / P18C 0.17 |

**Per capture-mode lockbox real FPR** at scaled τ:

| capture_mode | n | τ=0.5 | τ=0.92 | τ=0.974 |
|---|---:|---|---|---|
| normal_photo | 20 | P8A 0.05 / P18T 0.35 / P18C 0.90 | P8A 0.00 / P18T 0.00 / P18C 0.50 | P8A 0.00 / P18T 0.00 / P18C 0.05 |
| webcam | 14 | P8A 0.71 / P18T 0.79 / P18C 0.93 | P8A 0.36 / P18T 0.21 / P18C 0.43 | P8A 0.14 / P18T 0.07 / P18C 0.43 |
| phone_screen | 6 | P8A 0.00 / P18T 0.17 / P18C 0.67 | P8A 0.00 / P18T 0.00 / P18C 0.00 | P8A 0.00 / P18T 0.00 / P18C 0.00 |

**Per-method lockbox fake recall** at scaled τ:

| method | n | τ=0.5 | τ=0.92 | τ=0.974 |
|---|---:|---|---|---|
| teams_capture_cam_test_s33 | 36 | P8A 0.58 / P18T 0.64 / P18C 0.75 | P8A 0.31 / P18T 0.22 / P18C 0.25 | P8A 0.17 / P18T 0.14 / P18C 0.11 |
| teams_capture_pc_generator_s15 | 11 | P8A 1.00 / P18T 1.00 / P18C 1.00 | P8A 1.00 / P18T 0.91 / P18C 1.00 | P8A 0.91 / P18T 0.82 / P18C 0.82 |

**Aggregate lockbox** at scaled τ:

| metric | n | τ=0.5 | τ=0.92 | τ=0.974 |
|---|---:|---|---|---|
| All lockbox real FPR | 40 | P8A 0.28 / P18T 0.47 / P18C 0.88 | P8A 0.12 / P18T 0.07 / P18C 0.40 | P8A 0.05 / P18T 0.03 / P18C 0.17 |
| All lockbox fake RECALL | 47 | P8A 0.68 / P18T 0.72 / P18C 0.81 | P8A 0.47 / P18T 0.38 / P18C 0.43 | P8A 0.34 / P18T 0.30 / P18C 0.28 |

### 2.8 ROC analysis on lockbox subset (`run_roc_curves.py`)

Lockbox-only AUC (n_real=40, n_fake=47):

| arm | lockbox AUC |
|---|---:|
| P8A | 0.7926 |
| P18T | 0.6734 |
| P18C | 0.4973 |

Recall at fixed FPR floors:

| FPR floor | P8A recall | P8A τ | P18T recall | P18T τ | P18C recall | P18C τ |
|---|---:|---:|---:|---:|---:|---:|
| ≤ 0.01 | 0.298 | 0.9895 | 0.298 | 0.9792 | 0.149 | 0.9959 |
| ≤ 0.02 | 0.298 | 0.9895 | 0.298 | 0.9792 | 0.149 | 0.9959 |
| ≤ 0.05 | 0.383 | 0.9704 | 0.319 | 0.9574 | 0.170 | 0.9955 |
| ≤ 0.10 | 0.447 | 0.9584 | 0.383 | 0.9346 | 0.191 | 0.9926 |
| ≤ 0.15 | 0.468 | 0.9232 | 0.426 | 0.8913 | 0.234 | 0.9889 |
| ≤ 0.20 | 0.532 | 0.7493 | 0.468 | 0.8639 | 0.277 | 0.9844 |
| ≤ 0.30 | 0.745 | 0.3603 | 0.660 | 0.6606 | 0.362 | 0.9554 |

### 2.9 Lockbox AUC decomposition by dor / nondor reals (`decompose_lockbox_auc.py`)

| subset | n_real / n_fake | P8A AUC | P18T AUC | P18C AUC |
|---|---|---:|---:|---:|
| lockbox all | 40 / 47 | 0.7926 | 0.6734 | 0.4973 |
| lockbox minus dor reals | 15 / 47 | 0.6511 | 0.5929 | 0.5560 |
| lockbox only-dor reals + all fakes | 25 / 47 | 0.8774 | 0.7217 | 0.4621 |
| lockbox only-nondor reals + all fakes | 15 / 47 | 0.6511 | 0.5929 | 0.5560 |

---

## 3. P18 training data composition (from `R13_P18_METHOD_DOMAIN_GRL.yaml` `combined_paired.family_weights`)

```
df40_fake:                       0.2
visomaster_fake:                 4.0
deeplive_non_enhanced_fake:      2.5
deeplive_enhanced_fake:          3.0
deeplive_teams_fake:             7.0
deeplive_teams_real:             5.0
df40_real:                       0.5
realpool_real:                   1.5
external_real:                   1.0   (P18: lowered from 2.0; bucket 10 risk per Phase 2C audit)
```

P14 (`R13_P14_FACE_SCALE_JITTER_ISOLATED.yaml`) used identical family_weights except `external_real: 2.0`.

P18 12-bucket method-domain map (from yaml header comments, sourced from `analysis/method_class_audit_2026-05-01/proposed_method_domain_map.py`):

| bucket | label | sources |
|---|---|---|
| 0 | df40 | academic clean (17 methods collapsed) |
| 1 | deeplive_basic | edge_cases + minimal_processing |
| 2 | deeplive_enhanced | quality_enhancement + *_enhanced |
| 3 | deeplive_teams | Teams passthrough (incl. dor_shkedi) |
| 4 | visomaster_inswapper | Inswapper128 + SimSwap512 |
| 5 | visomaster_ghost | GhostFace v1/v2/v3 |
| 6 | visomaster_other | CSCS + InStyleSwapper256-A/B/C |
| 7 | visomaster_enhanced | RESERVED (disabled in P14/P15/P18) |
| 8 | visomaster_teams_recap | RESERVED (disabled) |
| 9 | proper_visomaster_clean | RESERVED (disabled) |
| 10 | external_vcd_real | (label-confound risk) |
| 11 | realpool_real | |

GRL `λ` ramp: P18 had `quality_grl_lambda` ramped 0.245 → 0.948 across training (per W&B `train/quality_grl_lambda` history). `quality_domain_loss` was non-zero throughout. P18C identical recipe minus `use_quality_domain_head`.

---

## 4. Best-val-AUC checkpoint inventory (P18 GCS ckpts)

P18T (`xpbvc1e4`):
- step8000 cumulative (= step3000 FT): val_AUC 0.9926, EER 0.0228 (highest val AUC)
- step7500: val_AUC 0.9922, EER 0.0254
- step4000 (= step-final periodic): val_AUC 0.9907, EER 0.0279 ← used in D
- step1500: val_AUC 0.9884, EER 0.0228

P18C (`rgt4kw2u`):
- step6500: val_AUC 0.9915, EER 0.0330 (highest val AUC)
- step6000: val_AUC 0.9914, EER 0.0330
- step4000 (= step-final periodic): val_AUC 0.9871, EER 0.0330 ← used in D
- step1500: val_AUC 0.9874, EER 0.0533

D scored only the periodic_step4000 ckpts; best-val-AUC ckpts not scored.

---

## 5. Source data files for these measurements

| Data source | Path | Description |
|---|---|---|
| Eval substrate | `analysis/embedding_triptych_2026-04-30/outputs/triptych_p8a_slot2_slot3/sampled_frames.csv` | 800 frames; first 800 rows used |
| Cached intermediate L3-L11 features | `analysis/_features_cache_2026-04-30/intermediate__{P8A,P18T,P18C}__layer{03,06,09,11}__n800.npz` | per-arm features at 4 layers |
| Cached final CLS features | `analysis/_features_cache_2026-04-30/final_cls__{P8A,P18T,P18C}__n800.npz` | per-arm 512-dim CLS features |
| P18 corrective probe outputs | `analysis/p18_probe_2026-05-01/outputs/corrective_probes__{P8A_step5000,xpbvc1e4__step4000,rgt4kw2u__step4000}_FINAL.json` | per-arm probe outputs |
| Diagnostics A/B/C/E results | `analysis/p18_probe_2026-05-01/outputs/diagnostics_abce_2026-05-02.json` | full bootstrap + Wilcoxon |
| Mini-scorecard | `analysis/p18_probe_2026-05-01/outputs/mini_scorecard_2026-05-02.json` | per-bucket FPR/recall |
| ROC curves | `analysis/p18_probe_2026-05-01/outputs/roc_curves_2026-05-02.json` | lockbox AUC + FPR floors |
| D contract artifacts | `analysis/p18_probe_2026-05-01/d_results/promotion_contract/{promotion_contract.json, promotion_winner.json, checkpoint_summary.csv, selected_threshold_scorecard.csv, threshold_grid.csv}` | full Vertex output |
| D scripts | `analysis/p18_probe_2026-05-01/{verify_l3_hook,extract_all_arms_layers,run_diagnostics_abce,run_minicard,run_roc_curves,decompose_lockbox_auc,corrective_probes,analyze_d_results}.py` | reproducible measurement code |

---

## 6. Cross-references from prior sessions (not re-measured here, cited)

- P8A v3 verify on Vertex (`5427016015462531072`, 2026-05-01 19:49 SUCCEEDED): under v3 default τ=0.991, dev_fake_macro_recall=0.136, lockbox_fake_recall=0.237. Matches §1.2 P8A row exactly.
- mclioexb v3 scorecard (2026-04-30 evening, memory `project_mclioexb_does_not_promote_2026-04-30.md`): zero τ in 5549-pt grid meets contract.
- P16 v3 scorecard (2026-04-30, memory `project_p16_data_axis_does_not_promote_2026-04-30.md`): all 8 P16 ckpts rank 2-9, P8A rank 1.
- Phase 1A finding (memory `project_phase1a_method_cluster_axis_2026-05-01.md`): trained P17 heads modally align with `is_dor_shkedi` direction (cos +0.07 → +0.14 across training).
- Move 1 grouped probe (memory `project_move1_bucket_gap_refuted.md`): bucket-LR AUC 0.916, identity-only control AUC 0.987 — bucket gap is identity-confounded.
- Eval-vs-production crop tightness gap (memory `project_eval_production_crop_tightness_gap.md`).

---

## 7. Operational state at end of measurement session

- 0 active Vertex jobs (in us-east1, us-west4, asia-southeast1).
- Image: 1.3.241 in GCR (Cloud Build `16a2455d-50d2-4d0e-aacc-b3c8780ed19a`, 4m14s, SUCCESS, 2026-05-02).
- Local /tmp/p18_ckpts/ ≈ 17 GB; /tmp/p17_ckpts/ ≈ 8 GB; downloadable from GCS.
- Working tree uncommitted (per project convention; many `.py`/`.md`/`.npz`/`.yaml` files added).
- VERSION file: 1.3.241.
- Spend this session: ~$10-15 of $80 GPU budget (Cloud Build $0.50 + D Vertex $8-12).
