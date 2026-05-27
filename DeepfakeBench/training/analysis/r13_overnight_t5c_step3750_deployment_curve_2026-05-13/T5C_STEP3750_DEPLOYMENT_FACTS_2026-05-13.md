# T5C_TOP_N_STEP3750 deployment τ-tradeoff curve — FACTS (2026-05-13)

> **Status: factual-only.** Numbers + tables + cross-references. No interpretation, no verdict language.
> Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, unfortunately, remarkably.

## 0. Scope and provenance

- **Question**: build a Pareto data sheet of `(real_fpr, fake_recall)` at multiple τ values for `T5C_TOP_N_STEP3750` to enable a ship-tonight decision; provide matched-τ comparison rows for `T5C_PERIODIC_STEP3500` (sister ckpt), `P8A_REFERENCE_STEP5000` (production anchor), and currently-deployed `E2B_TOP_N_STEP3200`.
- **Trigger**: T5C step3750 measured 9/92 on may6 production-drift frames (P8A-class Pillar 1 per `xinhe_cross_camera_audit_2026-05-06`); the 2026-05-12 T6/T7/T5C scorecard had it at rank-5 (all 3 contract gates passed) with dev_macro_recall=0.4732 and lockbox_real_fpr=0.0309. Task asked whether the per-identity profile matches sister ckpt step3500's catastrophic Roy_D FPR (87% at recommended τ per `T5C_STEP3500_DEPLOYMENT_FACTS_2026-05-13.md` §5).
- **Inputs**:
  - Video-level CSVs at `./_reports_cache/<suite>_<ckpt>_videos_report.csv` (10 suites × 4 ckpts = 40 files).
  - Source: `gs://training-job-outputs/test_results/teams_promotion_contract/t67-t5c-scorecard-2026-05-11/reports/`.
  - Calibrated τ values from `analysis/t6_t7_t5c_scorecard_eval_2026-05-12/RESULTS_FACTS_2026-05-12.md` §3.2 (v3-fix policy with stress-binding τ-bump).
- **Score basis**: video-level `avg_video_prob` (matches contract scorer's video-level decision).
- **Code**: `run_t5c_step3750_deployment_curve.py` in this folder. Re-runnable.

## 1. Anchor τ values and sample sizes

| Ckpt | Calibrated τ | dev_real n | lockbox_real n | lockbox_fake n | dor_dev n |
|---|---:|---:|---:|---:|---:|
| `T5C_TOP_N_STEP3750` | 0.8240 | 3253 | 1361 | 253 | 50 |
| `T5C_PERIODIC_STEP3500` | 0.8309 | 3253 | 1361 | 253 | 50 |
| `P8A_REFERENCE_STEP5000` | 0.9156 | 3253 | 1361 | 253 | 50 |
| `E2B_TOP_N_STEP3200` | 0.7108 | 3253 | 1361 | 253 | 50 |

Same eval substrate across ckpts (identical manifests).

## 2. T5C step3750 Pareto curve

Video-level metrics at user-specified τ grid plus step3750's own calibrated τ.

| τ | dev_real_fpr | lockbox_real_fpr | dor_dev_fpr | poor_q_dev_fpr | lighting_ext_dev_fpr | teams_fake_R | viso_enh_R | deeplive_enh_R | dev_macro_R | lockbox_fake_R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5000 | 0.1737 | 0.4592 | 0.5800 | 0.1387 | 0.2098 | 0.9108 | 0.6836 | 0.9890 | 0.8611 | 0.8261 |
| 0.7000 | 0.1137 | 0.1425 | 0.2400 | 0.0834 | 0.1499 | 0.7999 | 0.4691 | 0.9248 | 0.7313 | 0.6601 |
| 0.8000 | 0.0750 | 0.0514 | 0.1600 | 0.0466 | 0.1135 | 0.6658 | 0.2418 | 0.7284 | 0.5454 | 0.4704 |
| 0.8200 | 0.0673 | 0.0331 | 0.1000 | 0.0411 | 0.1056 | 0.6260 | 0.1727 | 0.6624 | 0.4870 | 0.4348 |
| **0.8240 (cal)** | **0.0661** | **0.0309** | **0.1000** | 0.0357 | 0.0999 | 0.6164 | 0.1545 | 0.6477 | **0.4729** | **0.4308** |
| **0.8301 (rec)** | **0.0625** | **0.0250** | **0.0800** | 0.0357 | 0.0999 | 0.6011 | 0.1364 | 0.5972 | **0.4449** | **0.4071** |
| 0.8400 | 0.0599 | 0.0206 | 0.0600 | 0.0314 | 0.0942 | 0.5733 | 0.1109 | 0.5505 | 0.4115 | 0.3992 |
| 0.8600 | 0.0504 | 0.0118 | 0.0600 | 0.0228 | 0.0792 | 0.5160 | 0.0582 | 0.4220 | 0.3321 | 0.3439 |
| 0.8800 | 0.0372 | 0.0073 | 0.0400 | 0.0119 | 0.0671 | 0.4624 | 0.0236 | 0.2972 | 0.2611 | 0.2727 |
| 0.9000 | 0.0255 | 0.0015 | 0.0400 | 0.0054 | 0.0521 | 0.3944 | 0.0073 | 0.1706 | 0.1908 | 0.1858 |
| 0.9200 | 0.0074 | 0.0000 | 0.0000 | 0.0011 | 0.0250 | 0.2233 | 0.0000 | 0.0275 | 0.0836 | 0.0474 |
| 0.9400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Source: `_t5c_step3750_pareto_2026-05-13.csv`.

### 2.1 Observations from the curve

- `lockbox_real_fpr` drops monotonically 45.9% → 0.0% over τ ∈ [0.50, 0.94].
- `lockbox_fake_recall` drops monotonically 82.6% → 0.0% over the same range.
- Crossing `lockbox_real_fpr ≤ 0.025` happens between τ=0.8240 (0.0309) and τ=0.8400 (0.0206). Precise τ that yields exactly 34 FP out of 1361 reals (= 0.0250): **τ = 0.830102**.
- Above τ=0.94 all metrics collapse to 0 (no probs in that range).

## 3. Recommended τ for T5C step3750 (lb_real_fpr ≤ 0.025 = E2B-comparable)

**τ_recommended = 0.830102**. Metrics at τ_rec for all 4 ckpts:

| Ckpt | τ | dev_real_fpr | lockbox_real_fpr | lockbox_fake_R | dev_macro_R | dor_dev_fpr | teams_fake_R | viso_enh_R | deeplive_enh_R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `T5C_STEP3750` @ rec | 0.8301 | 0.0642 | **0.0250** | **0.4071** | **0.4449** | **0.0800** | 0.5965 | 0.1273 | 0.6110 |
| `T5C_STEP3500` @ rec | 0.8301 | 0.0655 | 0.0279 | 0.6601 | 0.4596 | 0.0600 | 0.6131 | 0.1382 | 0.6275 |
| `P8A_STEP5000` @ rec | 0.8301 | 0.0833 | 0.0272 | 0.4664 | 0.3738 | 0.1600 | 0.5841 | 0.2145 | 0.3229 |
| `E2B_STEP3200` @ rec | 0.8301 | 0.0387 | 0.0118 | 0.4783 | 0.4354 | 0.0800 | 0.6169 | 0.0473 | 0.6422 |

Source: `_recommended_tau_2026-05-13.csv`.

## 4. Each ckpt at its OWN calibrated τ (operational points)

| Ckpt | τ_cal | lockbox_real_fpr | lockbox_fake_R | dev_macro_R | dor_dev_fpr | viso_enh_R | deeplive_enh_R |
|---|---:|---:|---:|---:|---:|---:|---:|
| `T5C_STEP3750` | 0.8240 | 0.0309 | 0.4308 | 0.4729 | 0.1000 | 0.1545 | 0.6477 |
| `T5C_STEP3500` | 0.8309 | 0.0279 | 0.6601 | 0.4581 | 0.0600 | 0.1382 | 0.6239 |
| `P8A_STEP5000` (anchor) | 0.9156 | 0.0184 | 0.3874 | 0.3003 | 0.0800 | 0.1345 | 0.2385 |
| `E2B_STEP3200` (deployed) | 0.7108 | 0.0235 | 0.6285 | 0.5078 | 0.1200 | 0.0509 | 0.7963 |

Cross-check vs `t6_t7_t5c_scorecard_eval_2026-05-12/RESULTS_FACTS_2026-05-12.md` §3.2:
- T5C_STEP3750 at cal-τ: lockbox_real_fpr=0.0309 matches scorecard's 0.0309. dev_macro_recall=0.4729 matches scorecard's 0.4730 (rounding).
- T5C_STEP3500 at cal-τ: lockbox_real_fpr=0.0279 matches scorecard's 0.0279. lockbox_fake_recall=0.6601 matches scorecard's 0.6601.
- P8A: lockbox_real_fpr=0.0184 matches scorecard's 0.0184. lockbox_fake_recall=0.3874 matches scorecard's 0.3874.

## 5. Per-identity sanity check at recommended τ

### 5.1 Lockbox-only (5 distinct identities)

At T5C step3750 τ=0.8301:

| Identity | Chronic-6 | n_videos | n_fp | FPR |
|---|:-:|---:|---:|---:|
| `dor_shkedi` | Y | 1138 | 33 | 0.0290 |
| `bla_bla_chow` | Y | 61 | 1 | 0.0164 |
| `Chikara_Takahashi` | | 25 | 0 | 0.0000 |
| `PC_Generator` | Y | 28 | 0 | 0.0000 |
| `real_dor` | | 109 | 0 | 0.0000 |

Lockbox-only: **0 identities with FPR > 30%** at recommended τ. **0 identities with FPR > 20%**.

Source: `_per_identity_lockbox_2026-05-13.csv`.

### 5.2 Pooled across all real suites (15 distinct identities)

Pooling `teams_real_all_lockbox + teams_real_all_dev + teams_real_poor_quality_dev + teams_real_lighting_extreme_dev + teams_real_dor_dev + teams_real_poor_quality_lockbox` at T5C step3750 τ=0.8301:

| Identity | Chronic-6 | n_pooled | n_fp | FPR | Per-suite breakdown |
|---|:-:|---:|---:|---:|---|
| **`Roy_D`** | Y | **246** | **209** | **0.8496** | dev=107/130; poor_q_dev=3/3; lighting_ext_dev=99/113 |
| `Q` | | 36 | 8 | 0.2222 | dev=8/36 |
| `bla_bla_chow` | Y | 1019 | 92 | 0.0903 | lockbox=1/61; dev=46/467; poor_q_dev=21/95; lighting_ext_dev=24/394 |
| `xiang` | Y | 325 | 18 | 0.0554 | dev=9/159; poor_q_dev=4/86; lighting_ext_dev=5/80 |
| `PC_Generator` | Y | 1102 | 50 | 0.0454 | lockbox=0/28; dev=38/525; poor_q_dev=2/241; lighting_ext_dev=10/308 |
| `dor_shkedi` | Y | 1250 | 37 | 0.0296 | lockbox=33/1138; dor_dev=4/50; (other suites 0) |
| `dor` | Y | 345 | 2 | 0.0058 | dev=1/269; poor_q_dev=1/74 |
| (8 others, all 0.0000) | | | | 0.0000 | |

Pooled (n≥5): **1 identity with FPR > 30%: `Roy_D` at 0.8496 (209/246).**

**Top 5 worst identities by FPR**:
1. Roy_D — 0.8496 (chronic-6)
2. Q — 0.2222 (not chronic-6, n=36)
3. bla_bla_chow — 0.0903 (chronic-6)
4. xiang — 0.0554 (chronic-6)
5. PC_Generator — 0.0454 (chronic-6)

Only Roy_D crosses the 30% gate.

Source: `_per_identity_pooled_2026-05-13.csv`.

### 5.3 Cross-ckpt per-identity FPR comparison at recommended τ (= 0.8301)

Same pooled real suites, at τ=0.8301, all four ckpts. Higher = worse.

| Identity | T5C_3750@rec | T5C_3500@rec | P8A@rec | E2B@rec | E2B@cal (0.7108) |
|---|---:|---:|---:|---:|---:|
| `Roy_D` | **0.8496 (209/246)** | **0.8862 (218/246)** | 0.3740 (92/246) | 0.1138 (28/246) | **0.1707 (42/246)** |
| `Q` | 0.2222 (8/36) | 0.2222 (8/36) | 0.9444 (34/36) | 0.0556 (2/36) | 0.2222 (8/36) |
| `bla_bla_chow` | 0.0903 (92/1019) | 0.1246 (127/1019) | 0.0736 (75/1019) | 0.1708 (174/1019) | 0.2718 (277/1019) |
| `xiang` | 0.0554 (18/325) | 0.0400 (13/325) | 0.0123 (4/325) | 0.1138 (37/325) | 0.1877 (61/325) |
| `PC_Generator` | 0.0454 (50/1102) | 0.0263 (29/1102) | 0.1969 (217/1102) | 0.0127 (14/1102) | 0.0408 (45/1102) |
| `dor_shkedi` | 0.0296 (37/1250) | 0.0264 (33/1250) | 0.0160 (20/1250) | 0.0144 (18/1250) | 0.0248 (31/1250) |
| `Chikara_Takahashi` | 0.0000 (0/40) | 0.0000 (0/40) | 0.3750 (15/40) | 0.0000 (0/40) | 0.0000 (0/40) |
| `dor` | 0.0058 (2/345) | 0.0087 (3/345) | 0.0000 (0/345) | 0.0000 (0/345) | 0.0000 (0/345) |
| (7 others, all 0.0000 across all ckpts) | | | | | |

Roy_D-specific cross-cell:
- T5C step3750 @ T5C-3750-cal-τ (0.8240): 0.8496 (209/246) — same as rec-τ
- T5C step3750 @ rec-τ (0.8301): 0.8496 (209/246)
- T5C step3500 @ rec-τ (0.8301): 0.8862 (218/246)
- T5C step3500 @ T5C-3500-cal-τ (0.8309): 0.8862 (218/246)
- P8A @ rec-τ (0.8301): 0.3740 (92/246)
- P8A @ P8A-cal-τ (0.9156): 0.3171 (78/246)
- E2B @ rec-τ (0.8301): 0.1138 (28/246)
- E2B @ E2B-cal-τ (0.7108): 0.1707 (42/246)

Source: `_per_identity_cross_ckpt_2026-05-13.csv`.

## 6. Headline tradeoff matrix

### 6.1 T5C step3750 @ rec-τ vs E2B @ E2B-cal-τ (deployed)

| Metric | T5C step3750 @ τ=0.8301 | E2B @ τ=0.7108 (deployed) | Δ (T5C_3750 − E2B) |
|---|---:|---:|---:|
| `lockbox_real_fpr` | 2.50% | 2.35% | **+0.15 pp** |
| `lockbox_fake_recall` | 40.71% | 62.85% | **−22.13 pp** |
| `dev_macro_recall` | 44.49% | 50.78% | **−6.28 pp** |
| `dor_dev_fpr` | 8.00% | 12.00% | **−4.00 pp** |
| `teams_fake_dev_recall` | 60.11% | 67.82% | **−7.71 pp** |
| `viso_enh_dev_recall` | 13.64% | 5.09% | **+8.55 pp** |
| `deeplive_enh_dev_recall` | 59.72% | 79.63% | **−19.91 pp** |
| `Roy_D` pooled FPR | 84.96% | 17.07% | **+67.89 pp** |

### 6.2 T5C step3750 @ rec-τ vs T5C step3500 @ T5C-3500-cal-τ (sister ckpt)

| Metric | T5C_3750 @ τ=0.8301 | T5C_3500 @ τ=0.8309 | Δ (T5C_3750 − T5C_3500) |
|---|---:|---:|---:|
| `lockbox_real_fpr` | 2.50% | 2.79% | **−0.29 pp** |
| `lockbox_fake_recall` | 40.71% | 66.01% | **−25.30 pp** |
| `dev_macro_recall` | 44.49% | 45.81% | **−1.32 pp** |
| `dor_dev_fpr` | 8.00% | 6.00% | **+2.00 pp** |
| `Roy_D` pooled FPR | 84.96% | 88.62% | **−3.66 pp** |

### 6.3 T5C step3750 @ rec-τ vs P8A @ P8A-cal-τ (production anchor)

| Metric | T5C_3750 @ τ=0.8301 | P8A @ τ=0.9156 | Δ (T5C_3750 − P8A) |
|---|---:|---:|---:|
| `lockbox_real_fpr` | 2.50% | 1.84% | **+0.66 pp** |
| `lockbox_fake_recall` | 40.71% | 38.74% | **+1.98 pp** |
| `dev_macro_recall` | 44.49% | 30.03% | **+14.47 pp** |
| `dor_dev_fpr` | 8.00% | 8.00% | **+0.00 pp** |
| `Roy_D` pooled FPR | 84.96% | 31.71% | **+53.25 pp** |

### 6.4 T5C step3750 @ T5C-3750-cal-τ (own operational point) vs comparators

| Metric | T5C_3750 @ τ=0.8240 (cal) | E2B @ deployed (0.7108) | P8A @ cal (0.9156) | T5C_3500 @ cal (0.8309) |
|---|---:|---:|---:|---:|
| `lockbox_real_fpr` | 3.09% | 2.35% | 1.84% | 2.79% |
| `lockbox_fake_recall` | 43.08% | 62.85% | 38.74% | 66.01% |
| `dev_macro_recall` | 47.29% | 50.78% | 30.03% | 45.81% |
| `dor_dev_fpr` | 10.00% | 12.00% | 8.00% | 6.00% |
| `Roy_D` pooled FPR | 84.96% | 17.07% | 31.71% | 88.62% |

(Roy_D pooled FPR at T5C-3750-cal-τ = 0.8240 is the same 209/246 = 0.8496 as at rec-τ = 0.8301, because the same 209 Roy_D videos score above both thresholds. Roy_D scores are deeply over-confident on this ckpt.)

## 7. Score-distribution observation on Roy_D under step3750

Roy_D pooled across `teams_real_all_dev + teams_real_poor_quality_dev + teams_real_lighting_extreme_dev` has n=246 reals. Of these, 209/246 score ≥ 0.8240 (cal-τ) and 209/246 score ≥ 0.8301 (rec-τ). The implied score percentile structure on Roy_D: at least 85th-percentile of Roy_D reals score in [0.8240, 0.8301], far above the score mass on other identities at the same τ.

## 8. v3-fix gate evaluation at recommended τ for T5C step3750

Gates per contract v3-fix: real_fpr ≤ 0.07; stress_fpr ≤ 0.10; recall ≥ 0.30. Plus MODEL_GOALS' "no single identity FPR > 30%" rule.

| Gate | Threshold | T5C step3750 @ rec-τ | Pass? |
|---|---|---:|:-:|
| dev_real_fpr | ≤ 0.07 | 0.0625 | ✓ |
| stress_fpr (max poor_q, lighting_ext) | ≤ 0.10 | max(0.0357, 0.0999) = 0.0999 | ✓ |
| dev_macro_recall | ≥ 0.30 | 0.4449 | ✓ |
| MODEL_GOALS single-identity FPR | ≤ 0.30 (~ 30%) | **Roy_D 0.8496** | **✗** |

## 9. File index

| File | Content |
|---|---|
| `run_t5c_step3750_deployment_curve.py` | source script |
| `_t5c_step3750_pareto_2026-05-13.csv` | T5C step3750 metrics at 11 τ values |
| `_comparison_2026-05-13.csv` | All 4 ckpts at 10 grid τ + each at own cal-τ + each at recommended τ |
| `_recommended_tau_2026-05-13.csv` | All 4 ckpts at τ=0.8301 (single row each) |
| `_per_identity_lockbox_2026-05-13.csv` | T5C step3750 @ rec-τ lockbox-only (5 ids) |
| `_per_identity_pooled_2026-05-13.csv` | T5C step3750 @ rec-τ pooled across 6 real suites (15 ids) |
| `_per_identity_cross_ckpt_2026-05-13.csv` | per-identity FPR for all 4 ckpts at both rec-τ and each cal-τ |
| `_results_2026-05-13.json` | full JSON dump of all above |

## 10. Direct observations

1. **Roy_D pooled FPR at T5C step3750 rec-τ = 0.8496 (209/246).** This is the only identity above 30% in the pooled-across-real-suites view. Roy_D is in the chronic-6 set per `project_chronic_offenders_partition_per_ckpt_2026-05-04`.
2. Roy_D pooled FPR is **84.96% (T5C step3750) vs 88.62% (T5C step3500) vs 31.71% (P8A) vs 17.07% (E2B at deployed-τ)** — both T5C variants share the Roy_D catastrophe within 4 pp of each other.
3. Roy_D pooled FPR at T5C step3750 cal-τ (0.8240) = 0.8496 — same as rec-τ. Tightening τ from 0.8240 to 0.8301 catches 0 additional Roy_D videos; the catastrophe is concentrated in score band > 0.83.
4. Lockbox-only (n=5 identities) shows **0 identities > 30% and 0 identities > 20%** at rec-τ. The chronic-identity catastrophe is concentrated in the dev real suites (where Roy_D has n=246 reals), not in lockbox real (where Roy_D has n=0).
5. T5C step3750 vs E2B (deployed) at matched lb_real_fpr ≈ 2.5%: lockbox_fake_recall **−22.13 pp**, dev_macro_recall **−6.28 pp**, deeplive_enh_dev_recall **−19.91 pp**, viso_enh_dev_recall **+8.55 pp**, Roy_D pooled FPR **+67.89 pp**.
6. T5C step3750 vs T5C step3500 at matched τ=0.8301: step3750 has lower `lockbox_real_fpr` (2.50% vs 2.79%) but lower `lockbox_fake_recall` (40.71% vs 66.01%) and equivalent dev_macro (44.49% vs 45.96%); the −25 pp lockbox-recall gap is the dominant difference.
7. T5C step3750 cross-ckpt at rec-τ vs P8A on `Q` (n=36): T5C_3750 0.2222 vs P8A 0.9444. P8A handles Roy_D better (0.3740) than T5C variants (0.85-0.89), but P8A handles Q worse (0.9444) than T5C variants (0.2222).
8. v3-fix gates (real_fpr ≤ 0.07, stress ≤ 0.10, recall ≥ 0.30) all pass for T5C step3750 at rec-τ. The MODEL_GOALS single-identity gate (Roy_D > 30%) does not pass.
9. At cal-τ (0.8240), the partial scorecard verdict was rank-5 with all 3 contract gates passed. The cross-ckpt comparison at rec-τ does not change the gate status (still all 3 pass) but exposes the Roy_D dev-suite pooled FPR.
