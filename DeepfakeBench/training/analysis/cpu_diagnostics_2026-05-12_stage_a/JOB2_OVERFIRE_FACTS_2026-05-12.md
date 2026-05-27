# CPU Job 2 — T5C step3500 vs P8A overfire population audit — FACTS (2026-05-12)

> **Status: factual-only.** Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, unfortunately, remarkably.
> Numbers + tables + cross-references only. Interpretation belongs in `STAGE_A_PLUS_PACKET_PLAN_OPINIONS_2026-05-12.md`.
>
> **Scope**: characterize the 766 real-cohort frames where T5C_step3500 scored ≥ 0.5 higher than P8A — by suite, identity, IQ profile. The 766-frame population is the candidate target cohort for an anchor-loss intervention.
>
> **Inputs**:
> - Unified per-frame matrix: `outputs/unified_frame_matrix.csv` (13,636 frames × 9 contract suites × {P8A, T5C_step3500, T3_S1_step1500} score columns; IQ atlas joined)
> - Real subset (label==0): 9,077 frames
> - Compute code: `job2_overfire_audit.py`

---

## 1. Method

For each real frame `i` in the 9,077-frame contract real cohort:
- `delta(i) = T5C_step3500(i) − P8A(i)`
- `overfire_T5C(i) = 1 if delta(i) > 0.5 else 0`

Identity extraction: `identity = filename.split("__")[0]` (filenames follow `<identity>__s<session>_<frame>_crop_<crop>__<hash>.jpg`).

## 2. Overfire frequency

| Statistic | Value |
|---|---:|
| Total real frames | 9,077 |
| Overfire frames (delta > 0.5) | 766 |
| Overfire rate | 8.44% |

## 3. Overfire by suite

| Suite | n | overfires | rate | mean(delta) |
|---|---:|---:|---:|---:|
| teams_real_all_dev | 4564 | 176 | 3.86% | +0.090 |
| teams_real_all_lockbox | 1418 | 429 | 30.25% | +0.373 |
| teams_real_dor_dev | 50 | 2 | 4.00% | +0.188 |
| teams_real_lighting_extreme_dev | 1742 | 126 | 7.23% | +0.147 |
| teams_real_poor_quality_dev | 1303 | 33 | 2.53% | +0.113 |

## 4. Overfire by identity (top 20 by absolute count)

| Identity | n | overfires | rate |
|---|---:|---:|---:|
| dor_shkedi | 1263 | 380 | 30.09% |
| bla_bla_chow | 1071 | 186 | 17.37% |
| Roy_D | 246 | 109 | 44.31% |
| xiang | 325 | 30 | 9.23% |
| dor | 395 | 26 | 6.58% |
| PC_Generator | 1729 | 20 | 1.16% |
| Xiang_Xiang2_Feng | 753 | 12 | 1.59% |
| orel | 68 | 3 | 4.41% |
| Cam_Test | 176 | 0 | 0.00% |
| Chikara_Takahashi | 42 | 0 | 0.00% |
| Md_noyn_Sharker | 1014 | 0 | 0.00% |
| Q | 54 | 0 | 0.00% |
| Test_Cam | 1784 | 0 | 0.00% |
| ilan | 48 | 0 | 0.00% |
| real_dor | 109 | 0 | 0.00% |

## 5. Cumulative overfire coverage by identity rank

| Top-k identities | overfires covered | % of total (766) |
|---:|---:|---:|
| 1 (dor_shkedi) | 380 | 49.6% |
| 2 (+ bla_bla_chow) | 566 | 73.9% |
| 3 (+ Roy_D) | 675 | 88.1% |
| 4 (+ xiang) | 705 | 92.0% |
| **5 (+ dor)** | **731** | **95.4%** |
| 10 | 766 | 100.0% |

| Coverage target | # identities required |
|---|---:|
| 50% of overfires | 2 |
| 80% of overfires | 3 |
| 95% of overfires | 5 |

## 6. IQ profile of overfires vs non-overfires

Means + medians on the joined IQ atlas subset (n_joined / 9,077 reals).

| Axis | overfire mean | overfire median | non-overfire mean | non-overfire median |
|---|---:|---:|---:|---:|
| min_dim | 277.21 | 248.00 | 252.92 | 239.00 |
| lap_var | 219.77 | 186.62 | 307.04 | 176.83 |
| color_a_dev | 12.06 | 10.03 | 9.78 | 9.24 |
| saturation_mean | 90.49 | 80.57 | 79.20 | 78.21 |
| luma_mean | 157.71 | 156.19 | 161.50 | 162.34 |

## 7. Chronic_6 share

`chronic_6` = identity matches `Roy_D | PC_Generator | bla_bla_chow | Md_noyn_Sharker | dor_shkedi | healthy_dor`.

| Quantity | Value |
|---|---:|
| Chronic_6 share of overfires (766) | 697 (91.0%) |
| Chronic_6 share of all reals (9,077) | 5,373 (59.2%) |

## 8. Per-cohort real-side score distribution (median / p90)

| Cohort | n | P8A p50 | P8A p90 | T5C_step3500 p50 | T5C_step3500 p90 |
|---|---:|---:|---:|---:|---:|
| Chronic_6 reals | 5,373 | 0.0150 | 0.8206 | 0.2723 | 0.8051 |
| Healthy reals | 3,704 | 0.0059 | 0.0902 | 0.0933 | 0.3473 |

## 9. Output artifacts

- `outputs/real_with_overfire_flag.csv` — 9,077 rows; real-cohort frames with score columns, delta, overfire_T5C flag, identity, is_chronic_6, suite, IQ atlas axes.

## 10. Caveats

- Identity extraction is filename-based and inherits whatever conventions the upstream data pipeline used. The chronic_6 set is fixed at 6 patterns per memory `project_chronic_offenders_partition_per_ckpt_2026-05-04`.
- IQ atlas join is 6,905 / 13,636 across the full 9-suite matrix (50.6% coverage). The §6 IQ profile uses only the joined subset; the joined-vs-unjoined difference is documented in `STAGE_A_FACTS_2026-05-12.md` §1.
- The 766-frame overfire population is restricted to "delta > 0.5" — a threshold chosen for inspection of the load-bearing tail. The full delta distribution lives in `real_with_overfire_flag.csv` and can be re-thresholded.

## 11. Direct observations

1. 766 of 9,077 real frames (8.44%) have T5C_step3500 − P8A > 0.5 (§2).
2. 429 of the 766 overfires (56.0%) are on `teams_real_all_lockbox`, where the suite-rate is 30.25% (§3).
3. The top 5 identities (dor_shkedi, bla_bla_chow, Roy_D, xiang, dor) cover 731 of 766 overfires (95.4%); the top 2 cover 49.6%, top 3 cover 88.1% (§5).
4. The chronic_6 set (5,373 / 9,077 = 59.2% of reals) accounts for 697 / 766 = 91.0% of overfires (§7).
5. Overfire frames have higher mean min_dim (277.21 vs 252.92), lower mean lap_var (219.77 vs 307.04), higher mean color_a_dev (12.06 vs 9.78), higher mean saturation_mean (90.49 vs 79.20) than non-overfire reals (§6).
6. Chronic_6 real cohort: P8A score p50 = 0.0150 vs T5C_step3500 p50 = 0.2723; healthy real cohort: P8A p50 = 0.0059 vs T5C_step3500 p50 = 0.0933 (§8).
