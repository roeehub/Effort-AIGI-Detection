# T4 promotion-contract + post-hoc diagnostics — FACTS (2026-05-11)

> **Status: factual-only.** Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, unfortunately, remarkably.
> Numbers + tables + cross-references only. Interpretation belongs in `AGENT_PROPOSAL_2026-05-11.md`.
>
> **Scope**: T4 (multi-axis-L11-GRL packet) promotion-contract scorecard verdict + 3 follow-up CPU diagnostics comparing T4_LAMBDA1_TOP_N_STEP10500 against P8A_REFERENCE_STEP5000 and T4_LAMBDA2_PERIODIC_STEP1500 on lockbox/dev cells.
>
> **Inputs**:
> - Promotion-contract artifacts: `gs://training-job-outputs/test_results/teams_promotion_contract/teams-promotion-contract-20260511-002920/promotion_contract/` (5 files) and `.../reports/*_videos_report.csv` (per-ckpt × per-suite, video-level avg_video_prob).
> - Local copies under `_t4_scorecard_local/` (gitignored).
> - Scripts in `analysis/cpu_diagnostics_2026-05-10/scripts/`: `digest_t4_scorecard.py`, `t4_p8a_tau_sweep.py`, `t4_calibration_probe.py`.

---

## 1. Question

For the T4 packet (multi-axis-L11-GRL), tabulate:

1. Formal promotion-contract verdict — rank-1 ckpt + per-ckpt metrics — Job 1
2. Per-suite metrics at the contract-calibrated τ — Job 2
3. τ-sensitivity for T4_LAMBDA1_TOP_N_STEP10500 on lockbox/dor — Job 3
4. Score-distribution percentiles on lockbox + dor cells — Job 4
5. Per-suite ROC AUC including held-out lockbox — Job 5
6. Effect of dev-fit isotonic calibration applied to lockbox — Job 6

## 2. Method

All jobs share inputs:
- 7 ckpts in scorecard: `P8A_REFERENCE_STEP5000`, `E2B_TOP_N_STEP3200`, `T4_LAMBDA1_TOP_N_STEP10500`, `T4_LAMBDA1_TOP_N_STEP9000`, `T4_LAMBDA1_PERIODIC_STEP5000`, `T4_LAMBDA2_PERIODIC_STEP1500`, `T4_LAMBDA2_PERIODIC_STEP2500`.
- 9 suites: `teams_real_all_dev`, `teams_real_poor_quality_dev`, `teams_real_lighting_extreme_dev`, `teams_fake_all_dev`, `visomaster_enhanced_macro_dev`, `deeplive_enhanced_dev`, `teams_real_all_lockbox`, `teams_fake_all_lockbox`, `teams_real_dor_dev`.
- Promotion-contract policy version `v3-fix` (recall-floor 0.30, target_real_fpr 0.07, target_stress_fpr 0.10).

τ-sweep computed over video-level `avg_video_prob`. Threshold-at-target-FPR uses `tau = sort_asc(real_scores)[n_real - int(target * n_real)]` and counts both reals/fakes at `score ≥ tau`.

AUC computed via Mann-Whitney U on video-level `avg_video_prob` (positive = fake, negative = real).

Isotonic regression fit on pooled dev (real + fake video-level scores) using pool-adjacent-violators with block-mean output; applied to lockbox by `isotonic_apply(model, raw_score) = mean_label_of_rightmost_block_with_threshold_≤_score`. **Note**: this implementation produces tied calibrated values across blocks; when computing fake_recall at target real_FPR on calibrated scores, ties at the threshold cause inclusion of any fakes at that exact value, so the reported "calibrated_fake_recall" overestimates true fake recall at the target FPR (see §6 caveat). AUC is invariant under any monotonic transform.

## 3. Job 1 — Promotion-contract verdict

Source: `_t4_scorecard_local/promotion_winner.json`, `_t4_scorecard_local/checkpoint_summary.csv`.

### 3.1 Rank-1 ckpt

`P8A_REFERENCE_STEP5000` (promotion_rank = 1).
- `selected_threshold` = 0.915605
- `dev_primary_real_fpr` = 0.069474
- `dev_worst_real_stress_fpr` = 0.068522
- `dev_fake_macro_recall` = 0.300280
- `lockbox_real_fpr` = 0.018369
- `lockbox_fake_recall` = 0.387352

### 3.2 All 7 ckpts ranked

| Rank | checkpoint_key | τ | dev_real_fpr | stress_fpr | dev_recall | lb_real_fpr | lb_fake_recall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | P8A_REFERENCE_STEP5000 | 0.9156 | 0.0695 | 0.0685 | 0.3003 | 0.0184 | 0.3874 |
| 2 | E2B_TOP_N_STEP3200 | 0.7108 | 0.0667 | 0.0999 | 0.5085 | 0.0235 | 0.6285 |
| 3 | T4_LAMBDA1_TOP_N_STEP10500 | 0.8440 | 0.0553 | 0.0999 | 0.4166 | 0.0536 | 0.3676 |
| 4 | T4_LAMBDA2_PERIODIC_STEP2500 | 0.6613 | 0.0572 | 0.0992 | 0.4909 | 0.0551 | 0.4308 |
| 5 | T4_LAMBDA1_PERIODIC_STEP5000 | 0.8561 | 0.0667 | 0.0999 | 0.4900 | 0.0867 | 0.4269 |
| 6 | T4_LAMBDA1_TOP_N_STEP9000 | 0.8823 | 0.0682 | 0.0999 | 0.5426 | 0.0867 | 0.2767 |
| 7 | T4_LAMBDA2_PERIODIC_STEP1500 | 0.7846 | 0.0590 | 0.0999 | 0.1596 | 0.0404 | 0.7708 |

Gates per contract: real_fpr ≤ 0.07; stress_fpr ≤ 0.10; dev_recall ≥ 0.30.

### 3.3 Per-gate pass/miss

| Rank | Ckpt | real_fpr ≤ 0.07 | stress ≤ 0.10 | recall ≥ 0.30 |
|---:|---|:-:|:-:|:-:|
| 1 | P8A | ✓ | ✓ | ✓ (0.0003 above floor) |
| 2 | E2B | ✓ | ✓ (0.0001 below ceiling) | ✓ |
| 3 | T4_L1_step10500 | ✓ | ✓ (0.0001 below ceiling) | ✓ |
| 4 | T4_L2_step2500 | ✓ | ✓ (0.0008 below ceiling) | ✓ |
| 5 | T4_L1_step5000 | ✓ | ✓ (0.0001 below ceiling) | ✓ |
| 6 | T4_L1_step9000 | ✓ | ✓ (0.0001 below ceiling) | ✓ |
| 7 | T4_L2_step1500 | ✓ | ✓ (0.0001 below ceiling) | ✗ (0.1404 below floor) |

## 4. Job 2 — Per-suite metrics at calibrated τ

Source: `_t4_scorecard_local/selected_threshold_scorecard.csv`.

P8A τ = 0.9156. T4_L1_step10500 τ = 0.8440. T4_L2_step1500 τ = 0.7846.

### 4.1 Per-suite FPR / recall

| Suite | P8A FPR | T4_L1_step10500 FPR | T4_L2_step1500 FPR |
|---|---:|---:|---:|
| teams_real_all_dev | 0.0695 | 0.0553 | 0.0590 |
| teams_real_poor_quality_dev | 0.0260 | 0.0184 | 0.0217 |
| teams_real_lighting_extreme_dev | 0.0685 | 0.0999 | 0.0999 |
| teams_real_all_lockbox | 0.0184 | 0.0536 | 0.0404 |
| teams_real_dor_dev (n=50) | 0.0800 | 0.1800 | 0.3200 |

| Suite | P8A recall | T4_L1_step10500 recall | T4_L2_step1500 recall |
|---|---:|---:|---:|
| teams_fake_all_dev | 0.5260 | 0.5832 | 0.4350 |
| visomaster_enhanced_macro_dev | 0.1345 | 0.1418 | 0.0436 |
| deeplive_enhanced_dev | 0.2385 | 0.5248 | 0.0000 |
| teams_fake_all_lockbox | 0.3874 | 0.3676 | 0.7708 |

### 4.2 dev_fake_macro_recall decomposition

T4_L1_step10500 dev_fake_macro_recall = (0.5832 + 0.1418 + 0.5248) / 3 = 0.4166.
P8A dev_fake_macro_recall = (0.5260 + 0.1345 + 0.2385) / 3 = 0.2997.

Per-cell Δ vs P8A: teams_fake_all_dev +0.0572, visomaster_enhanced_macro_dev +0.0073, deeplive_enhanced_dev +0.2863.

## 5. Job 3 — τ-sensitivity for T4_LAMBDA1_TOP_N_STEP10500

Source: `analysis/cpu_diagnostics_2026-05-10/scripts/t4_p8a_tau_sweep.py`.

### 5.1 Lockbox real_fpr + fake_recall + dor_fpr across τ

| τ | lb_real_fpr | lb_fake_recall | dor_fpr |
|---:|---:|---:|---:|
| 0.50 | 0.4629 | 0.7826 | 0.6600 |
| 0.60 | 0.3578 | 0.7075 | 0.5400 |
| 0.70 | 0.2337 | 0.6126 | 0.3800 |
| 0.75 | 0.1609 | 0.5771 | 0.2600 |
| 0.80 | 0.1007 | 0.4862 | 0.2000 |
| 0.82 | 0.0794 | 0.4506 | 0.1800 |
| 0.83 | 0.0661 | 0.4032 | 0.1800 |
| 0.85 | 0.0478 | 0.3557 | 0.1800 |
| 0.90 | 0.0073 | 0.1621 | 0.0800 |
| 0.95 | 0.0000 | 0.0000 | 0.0000 |

### 5.2 Comparison vs P8A target operating point

P8A target: lockbox_real_fpr = 0.0184 (rank-1 point), lockbox_fake_recall = 0.3874.

Across τ ∈ [0.50, 0.99] for T4_L1_step10500: no τ satisfies both (lb_real_fpr ≤ 0.0184 AND lb_fake_recall ≥ 0.3874). Closest: at τ=0.85 (lb_real_fpr=0.048, lb_fake_recall=0.356).

## 6. Job 4 — Score-distribution percentiles

Source: `analysis/cpu_diagnostics_2026-05-10/scripts/t4_p8a_tau_sweep.py` (video-level avg_video_prob).

### 6.1 teams_real_all_lockbox (n=1361 reals, label=0)

| ckpt | p10 | p25 | p50 | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|
| P8A | 0.0061 | 0.0082 | 0.0157 | 0.0541 | 0.2541 | 0.9944 |
| T4_L1_step10500 | 0.1431 | 0.2600 | 0.4722 | 0.6852 | 0.8008 | 0.9173 |
| T4_L2_step1500 | 0.4904 | 0.5823 | 0.6582 | 0.7074 | 0.7501 | 0.8487 |

### 6.2 teams_fake_all_lockbox (n=253 fakes, label=1)

| ckpt | p10 | p25 | p50 | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|
| P8A | 0.0867 | 0.3438 | 0.8042 | 0.9896 | 0.9946 | 0.9946 |
| T4_L1_step10500 | 0.2996 | 0.5536 | 0.7936 | 0.8773 | 0.9094 | 0.9297 |
| T4_L2_step1500 | 0.7547 | 0.7881 | 0.8190 | 0.8447 | 0.8558 | 0.8797 |

### 6.3 teams_real_dor_dev (n=50 reals, label=0)

| ckpt | p10 | p25 | p50 | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|
| P8A | 0.0130 | 0.0319 | 0.1801 | 0.6637 | 0.9040 | 0.9845 |
| T4_L1_step10500 | 0.3081 | 0.4288 | 0.6195 | 0.7935 | 0.8836 | 0.9214 |
| T4_L2_step1500 | 0.7028 | 0.7269 | 0.7646 | 0.7972 | 0.8291 | 0.8699 |

### 6.4 Ratio of fake-p50 to real-p50 (lockbox)

| ckpt | p50_fake / p50_real |
|---|---:|
| P8A | 51.2 |
| T4_L1_step10500 | 1.68 |
| T4_L2_step1500 | 1.24 |

## 7. Job 5 — Per-suite AUC

Source: `analysis/cpu_diagnostics_2026-05-10/scripts/t4_calibration_probe.py` (Mann-Whitney AUC on video-level scores).

### 7.1 AUC table

| Suite (positive_label_suite vs real_suite) | P8A AUC | T4_L1_step10500 AUC | Δ |
|---|---:|---:|---:|
| teams_fake_all_dev vs teams_real_all_dev | 0.8896 | 0.9265 | +0.0369 |
| deeplive_enhanced_dev vs teams_real_all_dev | 0.8565 | 0.9352 | +0.0787 |
| visomaster_enhanced_macro_dev vs teams_real_all_dev | 0.7403 | 0.8218 | +0.0815 |
| teams_fake_all_lockbox vs teams_real_all_lockbox | 0.9355 | 0.7619 | −0.1736 |

n_fake / n_real per cell: dev 2409/3253; deeplive 545/3253; viso 550/3253; lockbox 253/1361.

## 8. Job 6 — Dev-fit isotonic calibration applied to lockbox

Source: `analysis/cpu_diagnostics_2026-05-10/scripts/t4_calibration_probe.py`.

### 8.1 Setup

Isotonic regression fit on n=6757 dev videos (3504 fake label=1, 3253 real label=0). Applied to 1361 lockbox real + 253 lockbox fake videos. Reported "calibrated_fake_recall" at target lockbox_real_fpr is the raw count of fakes with `calibrated_score ≥ τ_cal` where τ_cal is selected to admit ≤ target * n_real reals; see §2 caveat about block-tie inclusion inflating recall at target FPR.

### 8.2 Calibrated vs raw fake recall at target lockbox real_FPR (with block-tie caveat)

| target_real_fpr | ckpt | raw_fake_recall | calibrated_fake_recall |
|---:|---|---:|---:|
| 0.018 | P8A | 0.3794 | 0.5771 |
| 0.020 | P8A | 0.4032 | 0.5771 |
| 0.050 | P8A | 0.6206 | 0.6482 |
| 0.018 | T4_L1_step10500 | 0.2174 | 0.4743 |
| 0.020 | T4_L1_step10500 | 0.2174 | 0.4743 |
| 0.050 | T4_L1_step10500 | 0.3557 | 0.4743 |

### 8.3 AUC under calibration

Isotonic regression is monotonic; AUC is invariant under any monotonic transform. The lockbox AUCs in §7.1 (P8A 0.9355, T4_L1_step10500 0.7619) are unchanged by the calibration step.

## 9. Output artifacts and scripts

- `_t4_scorecard_local/` — local cache of GCS scorecard artifacts (gitignored).
- `analysis/cpu_diagnostics_2026-05-10/scripts/digest_t4_scorecard.py` — emits formal verdict digest.
- `analysis/cpu_diagnostics_2026-05-10/scripts/t4_p8a_tau_sweep.py` — Jobs 3 and 4.
- `analysis/cpu_diagnostics_2026-05-10/scripts/t4_calibration_probe.py` — Jobs 5 and 6.
- `analysis/cpu_diagnostics_2026-05-10/T4_SCORECARD_DIGEST_2026-05-11.md` — auto-generated digest of `promotion_winner.json` and `checkpoint_summary.csv` (regeneratable; large file from contract.json dump).

## 10. Direct observations

1. T4_L1_step10500 dev_fake_macro_recall Δ vs P8A (+0.117 absolute) decomposes to +0.286 on `deeplive_enhanced_dev` and ≤ +0.057 on the other two dev fake suites (§4.2).
2. Each of the 5 T4 ckpts has `dev_worst_real_stress_fpr` within 0.001 of the 0.100 gate ceiling; P8A is at 0.0685 (§3.2).
3. T4_L1_step10500 `lockbox_real_fpr` (0.0536) is 2.92× P8A's (0.0184); T4_L1_step10500 `teams_real_dor_dev` FPR (0.18) is 2.25× P8A's (0.08) (§4.1).
4. For T4_L1_step10500, no τ ∈ [0.50, 0.99] is found where both `lockbox_real_fpr ≤ 0.0184` and `lockbox_fake_recall ≥ 0.3874` (§5.2).
5. P8A lockbox real-vs-fake median ratio is 51.2; T4_L1_step10500's is 1.68; T4_L2_step1500's is 1.24 (§6.4).
6. T4_L1_step10500 dev AUCs are higher than P8A's across all three dev fake suites (Δ ∈ [+0.037, +0.082]); lockbox AUC is lower by 0.1736 absolute (§7.1).
7. AUC is invariant under monotonic calibration; T4_L1_step10500's lockbox AUC ceiling is 0.7619 (§8.3).
