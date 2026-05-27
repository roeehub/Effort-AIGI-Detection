# T6 / T7 / T5C promotion-contract scorecard — FACTS (2026-05-12)

> **Status: factual-only.** Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, unfortunately, remarkably.
> Numbers + tables + cross-references only. Interpretation belongs in `AGENT_PROPOSAL_2026-05-12.md`.
>
> **Scope**: combined promotion-contract scorecard verdict for the T6 (T3 + face_scale_jitter), T7 (T4 + face_scale_jitter), and T5C (T4 + classifier hidden_dim 1024) packets — 12 ckpts (9 candidates + 3 anchors) on the standing v3-fix policy.
>
> **Inputs**:
> - Promotion-contract artifacts: `gs://training-job-outputs/test_results/teams_promotion_contract/t67-t5c-scorecard-2026-05-11/promotion_contract/` (5 files) + `.../reports/` (per-ckpt × per-suite `_videos_report.csv`, `_frames_report.csv`, `_group_metrics.csv`, `_summary_report.txt`; 348 video-reports landed = 12 ckpts × 29 sub-suites).
> - Local copies under `_scorecard_local/` (gitignored).
> - Checkpoint map: `arena/checkpoint_maps/teams_target_domain.t67_t5c_2026-05-11.yaml`.
> - Suite manifest (target-domain 9-suite contract): same as T4 packet scorecard (`target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml`).
> - Policy: standing v3-fix (target_fake_recall_min=0.30, target_real_fpr=0.07, target_stress_fpr=0.10).
>
> **Vertex job**: `6455763074874867712` (us-east1, image `1.3.283`, display name `t67-t5c-scorecard-2026-05-11`). Started 2026-05-11 17:18:54 UTC; SUCCEEDED 2026-05-12 03:13:17 UTC. Runtime 9h 54m.

---

## 1. Question

For the combined T6 / T7 / T5C packet (12-ckpt scorecard), tabulate:

1. Formal promotion-contract verdict — rank-1 ckpt + per-ckpt metrics — Job 1
2. Per-ckpt × per-suite FPR + recall at the contract-calibrated τ — Job 2
3. Per-gate pass/miss decomposition (recall floor 0.30, real_fpr ≤0.07, stress_fpr ≤0.10) — Job 3
4. dev_fake_macro_recall decomposition for the candidate ckpts — Job 4
5. Lockbox-side numbers (lockbox_real_fpr + lockbox_fake_recall) — Job 5
6. teams_real_dor_dev FPR per ckpt (small-N=50 chronic-identity cohort) — Job 6

## 2. Method

Per-ckpt selected_threshold is computed by the scorer per v3-fix policy: it finds the threshold on `teams_real_all_dev` that yields `real_fpr ≤ target_real_fpr (=0.07)`, then checks the recall/stress-FPR floors. All 9 contract suites are scored at video-level `avg_video_prob`.

Per-suite metrics from `selected_threshold_scorecard.csv`. Per-ckpt summary metrics from `checkpoint_summary.csv`. Rank from `promotion_winner.json` + `checkpoint_summary.csv` `promotion_rank` column (lexicographic policy implemented in scorer).

Suite n_videos (constant across all 12 ckpts):

| suite | n_videos | n_real | n_fake |
|---|---:|---:|---:|
| teams_real_all_dev | 3253 | 3253 | 0 |
| teams_real_poor_quality_dev | 923 | 923 | 0 |
| teams_real_lighting_extreme_dev | 1401 | 1401 | 0 |
| teams_fake_all_dev | 2409 | 0 | 2409 |
| visomaster_enhanced_macro_dev | 550 | 0 | 550 |
| deeplive_enhanced_dev | 545 | 0 | 545 |
| teams_real_all_lockbox | 1361 | 1361 | 0 |
| teams_fake_all_lockbox | 253 | 0 | 253 |
| teams_real_dor_dev | 50 | 50 | 0 |

## 3. Job 1 — Promotion-contract verdict

Source: `_scorecard_local/promotion_winner.json`, `_scorecard_local/checkpoint_summary.csv`.

### 3.1 Rank-1 ckpt

`P8A_REFERENCE_STEP5000` (promotion_rank = 1).
- `selected_threshold` = 0.915605
- `dev_primary_real_fpr` = 0.069474
- `dev_worst_real_stress_fpr` = 0.068522
- `dev_fake_macro_recall` = 0.300280
- `lockbox_real_fpr` = 0.018369
- `lockbox_fake_recall` = 0.387352
- `threshold_candidate_count` = 5363

### 3.2 All 12 ckpts ranked

Per `checkpoint_summary.csv` (sorted by `promotion_rank` ascending):

| Rank | checkpoint_key | τ | dev_real_fpr | stress_fpr | dev_macro_recall | lb_real_fpr | lb_fake_recall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | P8A_REFERENCE_STEP5000 | 0.9156 | 0.0695 | 0.0685 | 0.3003 | 0.0184 | 0.3874 |
| 2 | E2B_TOP_N_STEP3200 | 0.7108 | 0.0667 | 0.0999 | 0.5085 | 0.0235 | 0.6285 |
| 3 | T5C_PERIODIC_STEP3500 | 0.8309 | 0.0652 | 0.0999 | 0.4589 | 0.0279 | 0.6601 |
| 4 | T3_SLOT1_PERIODIC_STEP1500 | 0.6817 | 0.0676 | 0.0992 | 0.3763 | 0.0309 | 0.7747 |
| 5 | T5C_TOP_N_STEP3750 | 0.8240 | 0.0661 | 0.0999 | 0.4732 | 0.0309 | 0.4308 |
| 6 | T5C_PERIODIC_STEP1500 | 0.9303 | 0.0698 | 0.0949 | 0.6833 | 0.3204 | 0.6640 |
| 7 | T6_PERIODIC_STEP3500 | 0.9374 | 0.0572 | 0.0999 | 0.2461 | 0.0147 | 0.2292 |
| 8 | T7_TOP_N_STEP4750 | 0.8553 | 0.0550 | 0.0999 | 0.2228 | 0.0169 | 0.1976 |
| 9 | T6_TOP_N_STEP10250 | 0.9598 | 0.0627 | 0.0999 | 0.2725 | 0.0176 | 0.3557 |
| 10 | T7_PERIODIC_STEP2500 | 0.8639 | 0.0630 | 0.0999 | 0.2158 | 0.0250 | 0.4822 |
| 11 | T6_PERIODIC_STEP1500 | 0.7811 | 0.0636 | 0.0992 | 0.2598 | 0.0441 | 0.8696 |
| 12 | T7_PERIODIC_STEP3500 | 0.9379 | 0.0630 | 0.0999 | 0.2277 | 0.0617 | 0.2609 |

### 3.3 Anchor identification

`P8A_REFERENCE_STEP5000`: production anchor (substrate-invariance reference; rank-1 across all R13 scorecards in 2026-05).
`E2B_TOP_N_STEP3200`: currently deployed model (per memory `project_deployment_is_e2b_2026-05-06`).
`T3_SLOT1_PERIODIC_STEP1500`: T3 packet's F0-passing periodic ckpt (last week's strongest new candidate).

## 4. Job 2 — Per-suite FPR at calibrated τ

Source: `_scorecard_local/selected_threshold_scorecard.csv`.

### 4.1 Per-suite real_fpr (rows = ckpts; cols = real suites)

| ckpt | teams_real_all_dev | poor_quality_dev | lighting_extreme_dev | real_all_lockbox | real_dor_dev (n=50) |
|---|---:|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 0.0695 | 0.0260 | 0.0685 | 0.0184 | 0.0800 |
| E2B_TOP_N_STEP3200 | 0.0667 | 0.0812 | 0.0999 | 0.0235 | 0.1200 |
| T3_SLOT1_PERIODIC_STEP1500 | 0.0676 | 0.0238 | 0.0992 | 0.0309 | 0.2000 |
| T5C_PERIODIC_STEP1500 | 0.0698 | 0.0617 | 0.0949 | 0.3204 | 0.6400 |
| T5C_PERIODIC_STEP3500 | 0.0652 | 0.0390 | 0.0999 | 0.0279 | 0.0600 |
| T5C_TOP_N_STEP3750 | 0.0661 | 0.0357 | 0.0999 | 0.0309 | 0.1000 |
| T6_PERIODIC_STEP1500 | 0.0636 | 0.0530 | 0.0992 | 0.0441 | 0.1800 |
| T6_PERIODIC_STEP3500 | 0.0572 | 0.0282 | 0.0999 | 0.0147 | 0.0400 |
| T6_TOP_N_STEP10250 | 0.0627 | 0.0498 | 0.0999 | 0.0176 | 0.0400 |
| T7_PERIODIC_STEP2500 | 0.0630 | 0.0509 | 0.0999 | 0.0250 | 0.0400 |
| T7_PERIODIC_STEP3500 | 0.0630 | 0.0390 | 0.0999 | 0.0617 | 0.1200 |
| T7_TOP_N_STEP4750 | 0.0550 | 0.0260 | 0.0999 | 0.0169 | 0.0400 |

### 4.2 Per-suite fake_recall (rows = ckpts; cols = fake suites)

| ckpt | teams_fake_all_dev | visomaster_enh_macro_dev | deeplive_enh_dev | teams_fake_all_lockbox |
|---|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 0.5255 | 0.1345 | 0.2385 | 0.3874 |
| E2B_TOP_N_STEP3200 | 0.6782 | 0.0509 | 0.7963 | 0.6285 |
| T3_SLOT1_PERIODIC_STEP1500 | 0.5907 | 0.1218 | 0.4165 | 0.7747 |
| T5C_PERIODIC_STEP1500 | 0.7306 | 0.6036 | 0.7156 | 0.6640 |
| T5C_PERIODIC_STEP3500 | 0.6122 | 0.1382 | 0.6239 | 0.6601 |
| T5C_TOP_N_STEP3750 | 0.6168 | 0.1545 | 0.6477 | 0.4308 |
| T6_PERIODIC_STEP1500 | 0.4902 | 0.1309 | 0.1560 | 0.8696 |
| T6_PERIODIC_STEP3500 | 0.4612 | 0.0509 | 0.2239 | 0.2292 |
| T6_TOP_N_STEP10250 | 0.4985 | 0.0327 | 0.2862 | 0.3557 |
| T7_PERIODIC_STEP2500 | 0.4840 | 0.0291 | 0.1339 | 0.4822 |
| T7_PERIODIC_STEP3500 | 0.4574 | 0.0345 | 0.1908 | 0.2609 |
| T7_TOP_N_STEP4750 | 0.4412 | 0.0273 | 0.2000 | 0.1976 |

## 5. Job 3 — Per-gate pass/miss

Gates per contract v3-fix: real_fpr ≤ 0.07; stress_fpr ≤ 0.10; recall ≥ 0.30.

### 5.1 Pass-all matrix

Margin column = Δ to gate boundary; "+x" = passes by x; "−x" = misses by x.

| Rank | Ckpt | real_fpr ≤0.07 (margin) | stress ≤0.10 (margin) | recall ≥0.30 (margin) | all pass? |
|---:|---|---|---|---|:-:|
| 1 | P8A | ✓ (+0.0005) | ✓ (+0.0315) | ✓ (+0.0003) | ✓ |
| 2 | E2B | ✓ (+0.0033) | ✓ (+0.0001) | ✓ (+0.2085) | ✓ |
| 3 | T5C step3500 | ✓ (+0.0048) | ✓ (+0.0001) | ✓ (+0.1589) | ✓ |
| 4 | T3_SLOT1 step1500 | ✓ (+0.0024) | ✓ (+0.0008) | ✓ (+0.0763) | ✓ |
| 5 | T5C step3750 | ✓ (+0.0039) | ✓ (+0.0001) | ✓ (+0.1732) | ✓ |
| 6 | T5C step1500 | ✓ (+0.0002) | ✓ (+0.0051) | ✓ (+0.3833) | ✓ |
| 7 | T6 step3500 | ✓ (+0.0128) | ✓ (+0.0001) | ✗ (−0.0539) | ✗ |
| 8 | T7 step4750 | ✓ (+0.0150) | ✓ (+0.0001) | ✗ (−0.0772) | ✗ |
| 9 | T6 step10250 | ✓ (+0.0073) | ✓ (+0.0001) | ✗ (−0.0275) | ✗ |
| 10 | T7 step2500 | ✓ (+0.0070) | ✓ (+0.0001) | ✗ (−0.0842) | ✗ |
| 11 | T6 step1500 | ✓ (+0.0064) | ✓ (+0.0008) | ✗ (−0.0402) | ✗ |
| 12 | T7 step3500 | ✓ (+0.0070) | ✓ (+0.0001) | ✗ (−0.0723) | ✗ |

### 5.2 Observations on the rank ordering

Among the 6 all-pass ckpts (ranks 1-6), the ordering does not follow `dev_fake_macro_recall` descending (T5C step1500 has the highest dev_macro_recall = 0.6833 but ranks 6). The ordering is consistent with ascending `lockbox_real_fpr` (P8A 0.0184 < E2B 0.0235 < T5C step3500 0.0279 < T3_SLOT1 0.0309 = T5C step3750 0.0309 < T5C step1500 0.3204).

The 6 ckpts that fail the recall floor (ranks 7-12) are T6 ×3 and T7 ×3. All 6 hit `stress_fpr` within 0.001 of the 0.100 ceiling (consistent with the scorer pushing τ upward to satisfy `real_fpr ≤ 0.07` until `stress_fpr` saturates the ceiling).

## 6. Job 4 — dev_fake_macro_recall decomposition

Source: §4.2.

For each ckpt: dev_macro = (teams_fake_all_dev + visomaster_enh_macro_dev + deeplive_enh_dev) / 3.

| Ckpt | teams_fake_all_dev | visomaster_enh | deeplive_enh | macro |
|---|---:|---:|---:|---:|
| P8A | 0.5255 | 0.1345 | 0.2385 | 0.2995 |
| E2B | 0.6782 | 0.0509 | 0.7963 | 0.5085 |
| T3_SLOT1 step1500 | 0.5907 | 0.1218 | 0.4165 | 0.3763 |
| T5C step1500 | 0.7306 | 0.6036 | 0.7156 | 0.6832 |
| T5C step3500 | 0.6122 | 0.1382 | 0.6239 | 0.4581 |
| T5C step3750 | 0.6168 | 0.1545 | 0.6477 | 0.4730 |
| T6 step3500 | 0.4612 | 0.0509 | 0.2239 | 0.2453 |
| T7 step4750 | 0.4412 | 0.0273 | 0.2000 | 0.2228 |

Per-cell Δ vs P8A (for the 6 all-pass ckpts):

| Ckpt | Δ teams_fake_all_dev | Δ visomaster_enh | Δ deeplive_enh |
|---|---:|---:|---:|
| E2B | +0.1527 | −0.0836 | +0.5578 |
| T3_SLOT1 step1500 | +0.0652 | −0.0127 | +0.1780 |
| T5C step1500 | +0.2051 | +0.4691 | +0.4771 |
| T5C step3500 | +0.0867 | +0.0037 | +0.3853 |
| T5C step3750 | +0.0914 | +0.0200 | +0.4092 |

## 7. Job 5 — Lockbox-side metrics

Source: §3.2 (rank table).

### 7.1 Lockbox real_fpr + fake_recall, all 12 ckpts

| Ckpt | lb_real_fpr | lb_fake_recall | n_real_lockbox | n_fake_lockbox |
|---|---:|---:|---:|---:|
| P8A | 0.0184 | 0.3874 | 1361 | 253 |
| E2B | 0.0235 | 0.6285 | 1361 | 253 |
| T5C step3500 | 0.0279 | 0.6601 | 1361 | 253 |
| T3_SLOT1 step1500 | 0.0309 | 0.7747 | 1361 | 253 |
| T5C step3750 | 0.0309 | 0.4308 | 1361 | 253 |
| T5C step1500 | 0.3204 | 0.6640 | 1361 | 253 |
| T6 step3500 | 0.0147 | 0.2292 | 1361 | 253 |
| T7 step4750 | 0.0169 | 0.1976 | 1361 | 253 |
| T6 step10250 | 0.0176 | 0.3557 | 1361 | 253 |
| T7 step2500 | 0.0250 | 0.4822 | 1361 | 253 |
| T6 step1500 | 0.0441 | 0.8696 | 1361 | 253 |
| T7 step3500 | 0.0617 | 0.2609 | 1361 | 253 |

### 7.2 Direct comparison: candidate ckpts vs anchors on lockbox

vs P8A baseline (lb_real_fpr 0.0184, lb_fake_recall 0.3874):
- E2B: +0.0051 real_fpr, +0.2411 fake_recall
- T5C step3500: +0.0095 real_fpr, +0.2727 fake_recall
- T3_SLOT1: +0.0125 real_fpr, +0.3873 fake_recall
- T5C step1500: +0.3020 real_fpr (note: 17× P8A), +0.2766 fake_recall

vs E2B baseline (lb_real_fpr 0.0235, lb_fake_recall 0.6285):
- T5C step3500: +0.0044 real_fpr, +0.0316 fake_recall
- T3_SLOT1: +0.0074 real_fpr, +0.1462 fake_recall

## 8. Job 6 — teams_real_dor_dev FPR (n=50 chronic-identity cohort)

Source: §4.1 last column.

n=50 reals. Below all 12 ckpts; ascending:

| Ckpt | dor_fpr (n_fp / 50) |
|---|---:|
| T6 step3500 | 0.04 (2/50) |
| T6 step10250 | 0.04 (2/50) |
| T7 step2500 | 0.04 (2/50) |
| T7 step4750 | 0.04 (2/50) |
| T5C step3500 | 0.06 (3/50) |
| P8A | 0.08 (4/50) |
| T5C step3750 | 0.10 (5/50) |
| E2B | 0.12 (6/50) |
| T7 step3500 | 0.12 (6/50) |
| T6 step1500 | 0.18 (9/50) |
| T3_SLOT1 step1500 | 0.20 (10/50) |
| T5C step1500 | 0.64 (32/50) |

Cohort size n=50 → 95% CI for FPR estimate ≈ ±0.14 absolute at p=0.10. Rank order is interpretable above the noise band only for large gaps (e.g., T5C step1500 0.64 vs others ≤0.20).

## 9. Output artifacts and scripts

- `_scorecard_local/` — local cache of `promotion_winner.json`, `promotion_contract.json` (52 MB), `checkpoint_summary.csv`, `selected_threshold_scorecard.csv` (44 KB), `threshold_grid.csv` (16 MB). Gitignored.
- `build_tables.py` — re-runnable script that emits the §4-§8 tables from the local cache. (To be added; placeholder until the agent commits.)
- Per-ckpt × per-suite raw reports under `gs://training-job-outputs/test_results/teams_promotion_contract/t67-t5c-scorecard-2026-05-11/reports/` (348 files; not locally mirrored).
- Checkpoint map: `arena/checkpoint_maps/teams_target_domain.t67_t5c_2026-05-11.yaml`.
- Yamls: `experiments/phase2_round13/R13_T6_T3_PLUS_JITTER_2026-05-11.yaml`, `R13_T7_T4_PLUS_JITTER_2026-05-11.yaml`, `R13_T5C_T4_BIG_CLASSIFIER_2026-05-11.yaml`.

## 10. Direct observations

1. P8A retains `promotion_rank` = 1 on the v3-fix contract; no T6 / T7 / T5C ckpt displaces it (§3.2).
2. Among the 12 scored ckpts, 6 pass all three contract gates (P8A, E2B, T5C step3500, T3_SLOT1 step1500, T5C step3750, T5C step1500); 6 fail the recall floor (all T6 ckpts ×3 and all T7 ckpts ×3) (§5.1).
3. Among all-pass ckpts, rank ordering is consistent with ascending `lockbox_real_fpr` (P8A 0.0184 → T5C step1500 0.3204) (§5.2).
4. T5C step3500 is the highest-ranking new-packet ckpt (rank 3); it ranks above the T3_SLOT1 anchor (rank 4) (§3.2).
5. T5C step1500 has the highest dev_fake_macro_recall (0.6832) but its `lockbox_real_fpr` (0.3204) is 17× P8A's (§7.2).
6. T5C step3500 dev_macro lift vs P8A (+0.1586) decomposes to +0.0867 / +0.0037 / +0.3853 across teams_fake_all_dev / visomaster_enh / deeplive_enh (§6); the dev lift is concentrated on `deeplive_enhanced_dev`.
7. All T7 ckpts and all T6 ckpts (except T6 step1500) hit `stress_fpr` within 0.001 of the 0.100 ceiling (§3.2); for these ckpts the scorer's τ choice is bounded by the stress-FPR ceiling at the candidate-τ frontier.
8. On the teams_real_dor_dev (n=50) cohort, T6 step3500, T6 step10250, T7 step2500, T7 step4750 all hit dor_fpr = 0.04 (lower than P8A's 0.08) while their dev_macro_recall is below the 0.30 floor (§8 + §3.2). The dor-FPR ranking within the n=50 noise band is not interpretable except for the T5C step1500 (0.64) catastrophic case.
9. T6 step1500 has the highest `lockbox_fake_recall` (0.8696) of any scored ckpt; its `lockbox_real_fpr` is 0.0441 and dev_macro_recall is 0.2598 (below floor) (§3.2).
10. T3_SLOT1 step1500's `lockbox_fake_recall` is 0.7747 — the highest among the 6 all-pass ckpts — at `lockbox_real_fpr` 0.0309 (§7.1).
