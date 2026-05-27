# T5C/T6 candidate-ckpt scoring — FACTS (2026-05-12)

> **Status: factual-only.** Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, unfortunately, remarkably.
> Numbers + tables + cross-references only. Interpretation belongs in `../cpu_diagnostics_2026-05-12_stage_a/STAGE_A_PLUS_PACKET_PLAN_OPINIONS_2026-05-12.md`.
>
> **Scope**: supplement to `T67_T5C_PROBE_FACTS_2026-05-11.md` (which abstained for network reasons on the candidate ckpts). On 2026-05-12, 3 candidate ckpts were downloaded and scored on the same 952-frame cohort matrix, then L11 atlas features extracted. This doc tabulates the scoring numbers; `INV_MEAN_FACTS_2026-05-12.md` (sibling) tabulates the atlas readouts.
>
> **Inputs**:
> - Ckpt files (downloaded 2026-05-12 12:31-12:33 UTC, gcloud storage cp from `gs://training-job-outputs/best_checkpoints/{jrlldtem,smmcn6tj}/`):
>   - `_ckpts/t5c/periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth` (942 MB)
>   - `_ckpts/t5c/periodic_effort_20260511_step1500_auc0.9874_eer0.0570.pth` (940 MB)
>   - `_ckpts/t6/periodic_effort_20260511_step1500_auc0.9903_eer0.0340.pth` (940 MB)
> - Cohort table: `analysis/cpu_diagnostics_2026-05-11_t67_t5c_probe/outputs/per_ckpt_cohort_scores.csv` — 952 frames × {P8A, E2B, T3_S1_step1500, T6_periodic_step1500, T5C_periodic_step1500, T5C_periodic_step3500}
> - Anchor ckpts (already scored 2026-05-11): P8A, E2B, T3_S1_step1500
> - Cohort composition (same as `T67_T5C_PROBE_FACTS_2026-05-11.md` §"Scope and methodology"):
>   - DOR sub-cohorts (388): 50 dev_real, 78 dev_fake, 100 lockbox_real, 80 NON_DOR dev_real, 80 NON_DOR dev_fake
>   - ROY_D (130 frames): chronic_6 reals from `stage2_cpu_2026-05-09/_roy_d_frames/`
>   - MAY5 (60), MAY6 (92): production-drift reals from `xinhe_cross_camera_audit_2026-05-06/raw/`
>   - CHRONIC6 from triptych (282 frames): 207 dev_real, 34 lockbox_real, 30 dev_fake, 11 lockbox_fake
> - Scoring code: `analysis/cpu_diagnostics_2026-05-11_t67_t5c_probe/run_score_incremental.py` (MPS device)

---

## 1. Per-cohort FPR @ τ=0.5 (real cohorts)

Source: `outputs/per_ckpt_cohort_scores.csv`. FPR = (scores ≥ 0.5).mean() per cohort.

| Cohort | n | P8A | E2B | T3_S1_step1500 | T5C_step3500 | T5C_step1500 | T6_step1500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAY5 | 60 | 0/60 (0.0%) | 1/60 (1.7%) | 0/60 (0.0%) | 0/60 (0.0%) | 3/60 (5.0%) | 0/60 (0.0%) |
| MAY6 | 92 | 0/92 (0.0%) | 53/92 (57.6%) | 6/92 (6.5%) | 16/92 (17.4%) | 71/92 (77.2%) | 29/92 (31.5%) |
| DOR_REAL_DEV | 50 | 21/50 (42.0%) | 23/50 (46.0%) | 21/50 (42.0%) | 31/50 (62.0%) | 50/50 (100.0%) | 28/50 (56.0%) |
| DOR_REAL_LOCKBOX | 100 | 6/100 (6.0%) | 10/100 (10.0%) | 12/100 (12.0%) | 61/100 (61.0%) | 100/100 (100.0%) | 15/100 (15.0%) |
| CHRONIC6_REAL_DEV | 207 | 32/207 (15.5%) | 31/207 (15.0%) | 20/207 (9.7%) | 31/207 (15.0%) | 62/207 (30.0%) | 14/207 (6.8%) |
| CHRONIC6_REAL_LOCKBOX | 34 | 6/34 (17.6%) | 17/34 (50.0%) | 9/34 (26.5%) | 26/34 (76.5%) | 34/34 (100.0%) | 11/34 (32.4%) |
| ROY_D | 130 | 62/130 (47.7%) | 46/130 (35.4%) | 119/130 (91.5%) | 130/130 (100.0%) | 129/130 (99.2%) | 124/130 (95.4%) |
| NON_DOR_REAL_DEV | 80 | 15/80 (18.8%) | 15/80 (18.8%) | 10/80 (12.5%) | 13/80 (16.2%) | 30/80 (37.5%) | 12/80 (15.0%) |

## 2. Per-cohort FPR @ τ=0.9 (real cohorts; deployment-stringent threshold)

| Cohort | n | P8A | E2B | T3_S1_step1500 | T5C_step3500 | T5C_step1500 | T6_step1500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAY5 | 60 | 0/60 (0.0%) | 0/60 (0.0%) | 0/60 (0.0%) | 0/60 (0.0%) | 0/60 (0.0%) | 0/60 (0.0%) |
| MAY6 | 92 | 0/92 (0.0%) | 16/92 (17.4%) | 0/92 (0.0%) | 0/92 (0.0%) | 14/92 (15.2%) | 12/92 (13.0%) |
| DOR_REAL_DEV | 50 | 12/50 (24.0%) | 10/50 (20.0%) | 8/50 (16.0%) | 2/50 (4.0%) | 45/50 (90.0%) | 6/50 (12.0%) |
| DOR_REAL_LOCKBOX | 100 | 1/100 (1.0%) | 1/100 (1.0%) | 1/100 (1.0%) | 0/100 (0.0%) | 70/100 (70.0%) | 0/100 (0.0%) |
| CHRONIC6_REAL_DEV | 207 | 22/207 (10.6%) | 12/207 (5.8%) | 5/207 (2.4%) | 3/207 (1.4%) | 16/207 (7.7%) | 3/207 (1.4%) |
| CHRONIC6_REAL_LOCKBOX | 34 | 2/34 (5.9%) | 1/34 (2.9%) | 0/34 (0.0%) | 0/34 (0.0%) | 16/34 (47.1%) | 4/34 (11.8%) |
| ROY_D | 130 | 40/130 (30.8%) | 9/130 (6.9%) | 89/130 (68.5%) | 78/130 (60.0%) | 118/130 (90.8%) | 94/130 (72.3%) |
| NON_DOR_REAL_DEV | 80 | 9/80 (11.2%) | 6/80 (7.5%) | 3/80 (3.8%) | 3/80 (3.8%) | 12/80 (15.0%) | 5/80 (6.2%) |

## 3. Per-cohort fake recall @ τ=0.5

| Cohort | n | P8A | E2B | T3_S1_step1500 | T5C_step3500 | T5C_step1500 | T6_step1500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DOR_FAKE_DEV | 78 | 77/78 (98.7%) | 62/78 (79.5%) | 58/78 (74.4%) | 72/78 (92.3%) | 78/78 (100.0%) | 52/78 (66.7%) |
| NON_DOR_FAKE_DEV | 80 | 62/80 (77.5%) | 62/80 (77.5%) | 62/80 (77.5%) | 72/80 (90.0%) | 79/80 (98.8%) | 65/80 (81.2%) |
| CHRONIC6_FAKE_DEV | 30 | 30/30 (100.0%) | 29/30 (96.7%) | 29/30 (96.7%) | 30/30 (100.0%) | 30/30 (100.0%) | 27/30 (90.0%) |
| CHRONIC6_FAKE_LOCKBOX | 11 | 11/11 (100.0%) | 11/11 (100.0%) | 9/11 (81.8%) | 10/11 (90.9%) | 11/11 (100.0%) | 10/11 (90.9%) |

## 4. MAY6 score distribution (mean / p50 / p90 / max)

n=92. Source: `outputs/per_ckpt_cohort_scores.csv` filtered to cohort=MAY6.

| Ckpt | mean | p50 | p90 | max |
|---|---:|---:|---:|---:|
| P8A | 0.0206 | 0.0093 | 0.0368 | 0.2535 |
| E2B | 0.5113 | 0.5794 | 0.9451 | 0.9896 |
| T3_S1_step1500 | 0.1087 | 0.0197 | 0.3171 | 0.8076 |
| T5C_step3500 | 0.2641 | 0.1702 | 0.5818 | 0.8287 |
| T5C_step1500 | 0.6806 | 0.7416 | 0.9215 | 0.9367 |
| T6_step1500 | 0.3799 | 0.2725 | 0.9394 | 0.9730 |

### 4.1 may6 − may5 mean-score drift (production-fragility signature)

| Ckpt | mean(may5) | mean(may6) | Δ = may6 − may5 |
|---|---:|---:|---:|
| P8A | 0.0207 | 0.0206 | −0.0001 |
| E2B | 0.0527 | 0.5113 | +0.4586 |
| T3_S1_step1500 | 0.0124 | 0.1087 | +0.0963 |
| T5C_step3500 | 0.0862 | 0.2641 | +0.1779 |
| T5C_step1500 | 0.2755 | 0.6806 | +0.4051 |
| T6_step1500 | 0.0785 | 0.3799 | +0.3014 |

## 5. FPR-calibrated τ comparison (calibrated to 5% on `teams_real_all_dev`)

τ values from `analysis/cpu_diagnostics_2026-05-12_stage_a/synthesize_facts.py` output (using the 13,636-frame contract data):

| Ckpt | calibrated τ |
|---|---:|
| P8A | 0.9752 |
| T3_S1_step1500 | 0.7729 |
| T5C_step3500 | 0.8465 |

Per-cohort behavior at these τ values is in `../cpu_diagnostics_2026-05-12_stage_a/STAGE_A_FACTS_2026-05-12.md` §4 and §2.4.

## 6. Output artifacts

- `outputs/per_ckpt_cohort_scores.csv` — 952 rows × {frame_path, label, cohort, cohort_group, 6 ckpt score columns}. Extended this session with T5C_periodic_step3500, T5C_periodic_step1500, T6_periodic_step1500 columns (P8A, E2B, T3_S1_step1500 columns pre-existed from 2026-05-11).
- `_ckpts/t5c/`, `_ckpts/t6/` — local copies of the 3 candidate ckpts (gitignored).

## 7. Caveats

- Cohort sizes vary: n=11 (CHRONIC6_FAKE_LOCKBOX), n=30 (CHRONIC6_FAKE_DEV), n=34 (CHRONIC6_REAL_LOCKBOX), n=50 (DOR_REAL_DEV), n=60 (MAY5) are small. 95% CI for FPR estimate ≈ ±0.14 absolute at p=0.10 for n=50.
- ROY_D cohort lives in `analysis/stage2_cpu_2026-05-09/_roy_d_frames/`, not a held-out lockbox cohort — its FPR readings characterize a specific axis (color_a_dev Q4 + warm-color shortcut) rather than overall lockbox FPR.
- All scoring used MPS device; preprocessing identical to `cpu_diagnostics_2026-05-09/run_t3_score_probe.py` (INTER_LINEAR resize to 224, BGR2RGB, CLIP normalization).
- The anchor ckpt scoring (P8A, E2B, T3_S1_step1500) was performed 2026-05-11 and matches memory facts (`project_xinhe_may6_falseflag_2026-05-06`, etc.) to within 1 frame per cohort.

## 8. Direct observations

1. P8A retains 0/92 may6 FPR at both τ=0.5 and τ=0.9 (§1, §2).
2. T5C_step3500 has 16/92 (17.4%) may6 FPR at τ=0.5, and 0/92 at τ=0.9 (§1, §2).
3. T5C_step3500 has 61/100 (61.0%) DOR_REAL_LOCKBOX FPR at τ=0.5, and 0/100 (0.0%) at τ=0.9 (§1, §2).
4. T5C_step3500 has 4.0% DOR_REAL_DEV FPR at τ=0.9 (vs P8A 24.0%) and 90.9% CHRONIC6_FAKE_LOCKBOX recall at τ=0.5 (vs P8A 100.0%) (§2, §3).
5. T5C_step1500 has 100% DOR_REAL_DEV FPR at τ=0.5 and 70/100 (70.0%) DOR_REAL_LOCKBOX FPR at τ=0.9 (§1, §2).
6. T6_step1500 has 29/92 (31.5%) may6 FPR at τ=0.5 and 12/92 (13.0%) at τ=0.9 (§1, §2).
7. T5C_step3500 may6 mean drift over may5 is +0.1779 (vs P8A −0.0001, T3_S1_step1500 +0.0963, E2B +0.4586) (§4.1).
8. T5C_step3500 catches 72/80 (90.0%) NON_DOR_FAKE_DEV at τ=0.5 (vs P8A 62/80 = 77.5%) (§3).
9. T6_step1500 (the highest-lockbox-fake-recall ckpt in the formal scorecard) catches 52/78 (66.7%) DOR_FAKE_DEV at τ=0.5 (vs P8A 98.7%) on the 952-frame probe (§3).
