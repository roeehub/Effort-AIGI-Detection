# P2 Phase A Scorecard — VERDICT FACTS (2026-05-08)

> **Status: factual-only.** No interpretation. Forbidden words: succeeds, fails,
> wins, promotes, deployment-grade.
>
> **Source**: Vertex job `6226906326623059968` (us-east1, image 1.3.273), display
> name `p2-scratch-scorecard-2026-05-08`, terminal state `JOB_STATE_SUCCEEDED`.
> Outputs at `gs://training-job-outputs/test_results/teams_promotion_contract/p2-scratch-scorecard-2026-05-08/`.
>
> **Run config**: 7-ckpt curated set (per `arena/checkpoint_maps/teams_target_domain.p2_scratch_2026-05-08.yaml`),
> 29-suite `target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml`,
> contract policy `target_real_fpr=0.07`, `target_stress_fpr=0.10`,
> `target_fake_recall_min=0.30`.

## 1. Promotion-contract ranking

From `promotion_contract/checkpoint_summary.csv` (rank ascending = better;
lexicographic τ-pick per contract policy).

| rank | checkpoint | wandb run | selected τ | dev_primary_real_fpr | dev_worst_real_stress_fpr | dev_fake_macro_recall | lockbox_real_fpr | lockbox_fake_recall |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | P8A_REFERENCE_STEP5000 | `9lmvb5b4` | 0.9156 | 0.0695 | 0.0685 | 0.3003 | 0.0184 | 0.3874 |
| 2 | E2B_TOP_N_STEP3200 | `rmat8lwx` | 0.7108 | 0.0667 | 0.0999 | 0.5085 | 0.0235 | 0.6285 |
| 3 | P2_C_PAIRRANK_PERIODIC_STEP3000 | `oaur8odo` | 0.7809 | 0.0566 | 0.0999 | 0.5247 | 0.1198 | 0.1265 |
| 4 | P2_D_FOURIER_PERIODIC_STEP3000 | `89tt9xyz` | 0.4600 | 0.0695 | 0.0999 | 0.5300 | **0.1639** | **0.8893** |
| 5 | P2_D_FOURIER_TOP_N_STEP19000 | `89tt9xyz` | 0.9889 | 0.0541 | 0.0999 | 0.2463 | 0.0051 | 0.1621 |
| 6 | P2_D_FOURIER_PERIODIC_STEP8000 | `89tt9xyz` | 0.9664 | 0.0504 | 0.0999 | 0.2029 | 0.0059 | 0.1186 |
| 7 | P2_C_PAIRRANK_TOP_N_STEP7000 | `oaur8odo` | 0.9975 | 0.0581 | 0.0999 | 0.1918 | 0.0110 | 0.1344 |

Promotion winner (`promotion_contract/promotion_winner.json`): `P8A_REFERENCE_STEP5000`.

## 2. Per-suite real FPR / fake recall at the contract τ

From `promotion_contract/selected_threshold_scorecard.csv`. The 9 contract
suites are reported here. Each ckpt evaluated at its own selected τ.

### 2.1 P8A_REFERENCE_STEP5000 (τ = 0.9156)

| suite | role | n | real_fpr | fake_recall |
|---|---|---:|---:|---:|
| teams_real_all_dev | dev_real | 3253 | 0.0695 | — |
| teams_real_poor_quality_dev | dev_real_stress | 923 | 0.0260 | — |
| teams_real_lighting_extreme_dev | dev_real_stress | 1401 | 0.0685 | — |
| teams_real_dor_dev | dev_real_dor | 50 | 0.0800 | — |
| teams_real_all_lockbox | lockbox_real | 1361 | 0.0184 | — |
| teams_fake_all_dev | dev_fake | 2409 | — | 0.5255 |
| visomaster_enhanced_macro_dev | dev_fake | 550 | — | 0.1345 |
| deeplive_enhanced_dev | dev_fake | 545 | — | 0.2385 |
| teams_fake_all_lockbox | lockbox_fake | 253 | — | 0.3874 |

### 2.2 E2B_TOP_N_STEP3200 (τ = 0.7108)

| suite | role | n | real_fpr | fake_recall |
|---|---|---:|---:|---:|
| teams_real_all_dev | dev_real | 3253 | 0.0667 | — |
| teams_real_poor_quality_dev | dev_real_stress | 923 | 0.0813 | — |
| teams_real_lighting_extreme_dev | dev_real_stress | 1401 | 0.0999 | — |
| teams_real_dor_dev | dev_real_dor | 50 | 0.1200 | — |
| teams_real_all_lockbox | lockbox_real | 1361 | 0.0235 | — |
| teams_fake_all_dev | dev_fake | 2409 | — | 0.6783 |
| visomaster_enhanced_macro_dev | dev_fake | 550 | — | 0.0509 |
| deeplive_enhanced_dev | dev_fake | 545 | — | 0.7963 |
| teams_fake_all_lockbox | lockbox_fake | 253 | — | 0.6285 |

### 2.3 P2_D_FOURIER_PERIODIC_STEP3000 (τ = 0.4600)

| suite | role | n | real_fpr | fake_recall |
|---|---|---:|---:|---:|
| teams_real_all_dev | dev_real | 3253 | 0.0695 | — |
| teams_real_poor_quality_dev | dev_real_stress | 923 | 0.0834 | — |
| teams_real_lighting_extreme_dev | dev_real_stress | 1401 | 0.0999 | — |
| teams_real_dor_dev | dev_real_dor | 50 | **0.4600** | — |
| teams_real_all_lockbox | lockbox_real | 1361 | **0.1639** | — |
| teams_fake_all_dev | dev_fake | 2409 | — | 0.6534 |
| visomaster_enhanced_macro_dev | dev_fake | 550 | — | 0.1655 |
| deeplive_enhanced_dev | dev_fake | 545 | — | 0.7688 |
| teams_fake_all_lockbox | lockbox_fake | 253 | — | 0.8893 |

## 3. Headline cross-checkpoint deltas (vs E2B as production deployment)

From §1 + §2.1-2.3.

| metric | P8A_REFERENCE | E2B_TOP_N (deployed) | P2_D_FOURIER_periodic_3000 |
|---|---:|---:|---:|
| dev_fake_macro_recall | 0.300 | 0.508 | 0.530 (+0.022 vs E2B) |
| teams_fake_all_dev fake_recall | 0.525 | 0.678 | 0.653 (−0.025 vs E2B) |
| visomaster_enhanced_macro_dev fake_recall | 0.135 | 0.051 | 0.166 (+0.115 vs E2B, +0.031 vs P8A) |
| deeplive_enhanced_dev fake_recall | 0.239 | 0.796 | 0.769 (−0.027 vs E2B) |
| teams_fake_all_lockbox fake_recall | 0.387 | 0.628 | 0.889 (+0.261 vs E2B) |
| teams_real_dor_dev real_fpr | 0.080 | 0.120 | **0.460** (+0.340 vs E2B) |
| teams_real_all_lockbox real_fpr | 0.018 | 0.024 | **0.164** (+0.140 vs E2B) |
| teams_real_lighting_extreme_dev real_fpr | 0.069 | 0.100 | 0.100 (=stress ceiling) |
| dev_primary_real_fpr | 0.069 | 0.067 | 0.069 (≈ E2B) |

## 4. Other P2 ckpts (rank 3 / 5 / 6 / 7) — headline deltas

| ckpt | rank | dev_fake_macro_recall | lockbox_real_fpr | lockbox_fake_recall |
|---|---:|---:|---:|---:|
| P2_C_PAIRRANK_PERIODIC_STEP3000 | 3 | 0.525 | 0.120 | 0.126 |
| P2_D_FOURIER_TOP_N_STEP19000 | 5 | 0.246 | 0.005 | 0.162 |
| P2_D_FOURIER_PERIODIC_STEP8000 | 6 | 0.203 | 0.006 | 0.119 |
| P2_C_PAIRRANK_TOP_N_STEP7000 | 7 | 0.192 | 0.011 | 0.134 |

The two P2-D late-checkpoint variants (top_n_step19000, periodic_step8000) sit
at very small `lockbox_real_fpr` (0.005-0.006) and small `dev_fake_macro_recall`
(0.20-0.25) at large τ ≥ 0.97. The two P2-C variants sit between.

## 5. Notes on contract τ search

- All seven ckpts had `threshold_candidate_count` between 5363 and 5610 (search
  resolved a fine grid).
- P2_D_FOURIER_PERIODIC_STEP3000's selected τ = 0.460 is the lowest among the 7;
  it is the only ckpt whose τ < 0.5.
- P8A's selected τ = 0.916 matches the post-2026-04-29 contract policy bug fix
  (memory `project_contract_policy_bug.md` records `0.92` as P8A's deployment τ;
  numerical equivalence within 0.5pp).

## 6. Cross-references

- `RESULTS_FACTS_2026-05-08.md` — per-slot training outcome (4 slots).
- `CANARY_TRAJECTORY_FACTS_2026-05-08.md` — in-training canary trajectories.
- `d1_d4_cpu/D5_CKPT_MAPPING_FACTS_2026-05-08.md` — D5 identified
  `D_periodic_step3000` as the highest `lockbox_recall_at_FPR_10pct` checkpoint
  across all 6 saved D ckpts on the 800-frame canary (0.37 vs 0.06-0.20 at later
  ckpts). The Phase A scorecard (this doc) corroborates direction at scale on the
  full lockbox suite (0.889 lockbox_fake_recall at τ=0.460) at the cost of
  +0.140 lockbox_real_fpr.
- `arena/checkpoint_maps/teams_target_domain.p2_scratch_2026-05-08.yaml` —
  ckpt-set definition.
- Memory `project_deployment_is_e2b_2026-05-06.md` — E2B is the deployment.

## 7. Artifacts

- `gs://training-job-outputs/test_results/teams_promotion_contract/p2-scratch-scorecard-2026-05-08/promotion_contract/promotion_winner.json`
- `…/promotion_contract/checkpoint_summary.csv`
- `…/promotion_contract/selected_threshold_scorecard.csv`
- `…/promotion_contract/promotion_contract.json` (50 MB; includes per-ckpt × per-suite per-frame distributional summaries)
- `…/promotion_contract/threshold_grid.csv` (9.9 MB; full τ-grid scan)
- `…/diagnostic_scorecard/scorecard.{csv,wide.csv,json}` — fixed-τ-0.5 sidecar
- `…/reports/{suite}_{ckpt}_{frames,group_metrics,videos,summary}_report.csv` — per-frame and per-video reports for all 7 ckpts × 29 suites = 812 files
