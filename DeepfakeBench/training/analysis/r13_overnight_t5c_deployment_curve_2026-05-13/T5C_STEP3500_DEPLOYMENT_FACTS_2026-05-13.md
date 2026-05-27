# T5C_PERIODIC_STEP3500 deployment τ-tradeoff curve — FACTS (2026-05-13)

> **Status: factual-only.** Numbers + tables + cross-references. No interpretation, no verdict language.
> Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, unfortunately, remarkably.

## 0. Scope and provenance

- **Question**: build a Pareto data sheet of `(real_fpr, fake_recall)` at multiple τ values for `T5C_PERIODIC_STEP3500` so a ship-tonight decision can be made; provide matched-τ comparison rows for `P8A_REFERENCE_STEP5000` and currently-deployed `E2B_TOP_N_STEP3200`.
- **Trigger**: T5C step3500 is rank-3 on the partial scorecard (v3-fix policy) with `lockbox_fake_recall=0.6719` (1.74× P8A_REFERENCE 0.3874) and `lockbox_real_fpr=0.0309` close to E2B's 0.0250.
- **Inputs**:
  - Video-level CSVs at `../r13_overnight_partial_scorecard_2026-05-13/_reports_cache/<suite>_<ckpt>_videos_report.csv`.
  - 10 suites loaded per ckpt: 6 real, 4 fake. The 9 contract suites in the scorecard plus `teams_real_poor_quality_lockbox` (which is referenced but has very small `n_real=22`).
  - Calibrated τ values from `_partial_scorecard_2026-05-13.csv` (dev-cal at `real_fpr ≤ 0.07` on `teams_real_all_dev`, midpoint convention).
- **Score basis**: video-level `avg_video_prob` (matches contract scorer's video-level decision).
- **Code**: `run_t5c_deployment_curve.py` in this folder. Re-runnable.

## 1. Anchor τ values and sample sizes

| Ckpt | Calibrated τ | dev_real n | lockbox_real n | lockbox_fake n | dor_dev n |
|---|---:|---:|---:|---:|---:|
| `T5C_PERIODIC_STEP3500` | 0.821989 | 3253 | 1361 | 253 | 50 |
| `P8A_REFERENCE_STEP5000` | 0.914263 | 3253 | 1361 | 253 | 50 |
| `E2B_TOP_N_STEP3200` | 0.695595 | 3253 | 1361 | 253 | 50 |

(Sample sizes identical across ckpts; same evaluation manifests.)

## 2. T5C step3500 Pareto curve

Video-level metrics at user-specified τ grid plus T5C's own calibrated τ. Columns: real-FPR and fake-recall per suite + macro aggregates.

| τ | dev_real_fpr | lockbox_real_fpr | dor_dev_fpr | poor_q_dev_fpr | lighting_ext_dev_fpr | teams_fake_R | viso_enh_R | deeplive_enh_R | dev_macro_R | lockbox_fake_R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5000 | 0.1752 | 0.4431 | 0.4600 | 0.1398 | 0.2256 | 0.9207 | 0.7273 | 0.9890 | 0.8790 | 0.9368 |
| 0.6000 | 0.1433 | 0.2667 | 0.3200 | 0.1083 | 0.1870 | 0.8804 | 0.6327 | 0.9706 | 0.8279 | 0.9091 |
| 0.7000 | 0.1122 | 0.1293 | 0.2400 | 0.0834 | 0.1556 | 0.8120 | 0.5036 | 0.9138 | 0.7431 | 0.8458 |
| 0.7500 | 0.0968 | 0.0801 | 0.1800 | 0.0693 | 0.1349 | 0.7522 | 0.3727 | 0.8477 | 0.6575 | 0.7984 |
| 0.8000 | 0.0787 | 0.0478 | 0.1400 | 0.0509 | 0.1163 | 0.6775 | 0.2564 | 0.7266 | 0.5535 | 0.7154 |
| **0.8220 (T5C cal)** | **0.0698** | **0.0309** | **0.1000** | 0.0433 | 0.1056 | 0.6335 | 0.1782 | 0.6661 | **0.4926** | **0.6719** |
| 0.8300 | 0.0661 | 0.0279 | 0.0600 | 0.0401 | 0.1014 | 0.6131 | 0.1382 | 0.6275 | 0.4596 | 0.6601 |
| **0.8417 (rec)** | **0.0624** | **0.0250** | **0.0600** | 0.0379 | 0.0989 | 0.5928 | 0.1145 | 0.5605 | **0.4227** | **0.6245** |
| 0.8500 | 0.0593 | 0.0220 | 0.0600 | 0.0379 | 0.0942 | 0.5704 | 0.0982 | 0.5266 | 0.3984 | 0.6047 |
| 0.8700 | 0.0486 | 0.0110 | 0.0600 | 0.0271 | 0.0835 | 0.5073 | 0.0436 | 0.3817 | 0.3109 | 0.5336 |
| 0.9000 | 0.0350 | 0.0029 | 0.0400 | 0.0141 | 0.0635 | 0.4263 | 0.0091 | 0.2239 | 0.2198 | 0.3676 |
| 0.9200 | 0.0154 | 0.0007 | 0.0200 | 0.0043 | 0.0335 | 0.3304 | 0.0000 | 0.0697 | 0.1334 | 0.1462 |
| 0.9400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0677 | 0.0000 | 0.0000 | 0.0226 | 0.0000 |
| 0.9600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Source: `_t5c_pareto_2026-05-13.csv` (+ row for τ=0.8417 derived directly from script for the recommended-τ block; CSV does not include rec-τ as a separate row).

### 2.1 Observations from the curve

- `lockbox_real_fpr` drops monotonically 44.3% → 0.0% as τ goes 0.50 → 0.96.
- `lockbox_fake_recall` drops monotonically 93.7% → 0.0% over the same range.
- Crossing `lockbox_real_fpr ≤ 0.025` happens between τ=0.83 (0.0279) and τ=0.85 (0.0220). The precise τ that puts `lockbox_real_fpr` exactly at 0.0250 is τ=0.841699 (computed from sorted lockbox real probs).
- Score range covered: above τ=0.94 all metrics collapse to 0 (no probs in that range).

## 3. Comparison at common τ grid — T5C vs P8A vs E2B

Same τ values applied to each ckpt's video-level probs. Columns: `lockbox_real_fpr / lockbox_fake_recall / dev_macro_recall / dor_dev_fpr`.

| τ | T5C lb_fpr | T5C lb_R | T5C macro_R | T5C dor_fpr | P8A lb_fpr | P8A lb_R | P8A macro_R | P8A dor_fpr | E2B lb_fpr | E2B lb_R | E2B macro_R | E2B dor_fpr |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5000 | 0.4431 | 0.9368 | 0.8790 | 0.4600 | 0.0617 | 0.6522 | 0.5297 | 0.3000 | 0.0647 | 0.8261 | 0.5923 | 0.2400 |
| 0.6000 | 0.2667 | 0.9091 | 0.8279 | 0.3200 | 0.0470 | 0.6047 | 0.4864 | 0.3000 | 0.0419 | 0.7470 | 0.5518 | 0.1800 |
| 0.7000 | 0.1293 | 0.8458 | 0.7431 | 0.2400 | 0.0411 | 0.5415 | 0.4438 | 0.2400 | 0.0250 | 0.6403 | 0.5140 | 0.1400 |
| 0.7500 | 0.0801 | 0.7984 | 0.6575 | 0.1800 | 0.0338 | 0.5296 | 0.4240 | 0.2400 | 0.0198 | 0.5771 | 0.4866 | 0.1200 |
| 0.8000 | 0.0478 | 0.7154 | 0.5535 | 0.1400 | 0.0287 | 0.5020 | 0.3918 | 0.1800 | 0.0162 | 0.5336 | 0.4572 | 0.1200 |
| 0.8300 | 0.0279 | 0.6601 | 0.4596 | 0.0600 | 0.0272 | 0.4664 | 0.3747 | 0.1600 | 0.0118 | 0.4783 | 0.4354 | 0.0800 |
| 0.8500 | 0.0220 | 0.6047 | 0.3984 | 0.0600 | 0.0250 | 0.4427 | 0.3612 | 0.1400 | 0.0118 | 0.4269 | 0.4156 | 0.0800 |
| 0.8700 | 0.0110 | 0.5336 | 0.3109 | 0.0600 | 0.0235 | 0.4348 | 0.3435 | 0.1200 | 0.0103 | 0.3913 | 0.3962 | 0.0600 |
| 0.9000 | 0.0029 | 0.3676 | 0.2198 | 0.0400 | 0.0198 | 0.4111 | 0.3109 | 0.1200 | 0.0088 | 0.3636 | 0.3627 | 0.0600 |
| 0.9200 | 0.0007 | 0.1462 | 0.1334 | 0.0200 | 0.0176 | 0.3874 | 0.2976 | 0.0800 | 0.0073 | 0.3202 | 0.3303 | 0.0600 |
| 0.9400 | 0.0000 | 0.0000 | 0.0226 | 0.0000 | 0.0132 | 0.3636 | 0.2800 | 0.0600 | 0.0044 | 0.2846 | 0.2610 | 0.0200 |
| 0.9600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0110 | 0.3281 | 0.2517 | 0.0600 | 0.0007 | 0.2411 | 0.1902 | 0.0200 |

Source: `_comparison_2026-05-13.csv`.

### 3.1 Each ckpt at its OWN calibrated τ (the operational points)

| Ckpt | τ_cal | lockbox_real_fpr | lockbox_fake_R | dev_macro_R | dor_dev_fpr |
|---|---:|---:|---:|---:|---:|
| `T5C_STEP3500` | 0.8220 | 0.0309 | 0.6719 | 0.4926 | 0.1000 |
| `P8A_STEP5000` | 0.9143 | 0.0184 | 0.3874 | 0.3003 | 0.0800 |
| `E2B_STEP3200` (deployed) | 0.6956 | 0.0250 | 0.6482 | 0.5151 | 0.1400 |

Cross-checks against the partial scorecard:
- T5C: `lockbox_fake_recall=0.6719` matches scorecard's `0.6719`.
- T5C: `lockbox_real_fpr=0.0309` matches scorecard's `0.0309`.
- P8A: `lockbox_fake_recall=0.3874` matches scorecard's `0.3874`.
- E2B: `lockbox_fake_recall=0.6482` matches scorecard's `0.6482`.

(The task statement's `lockbox_real_fpr 0.0279` value for T5C corresponds to τ=0.83, not τ_cal=0.82.)

## 4. Recommended τ for T5C (lb_real_fpr ≤ 0.025 = E2B-comparable)

Precise τ that puts T5C lockbox_real_fpr at the 0.025 ceiling: **τ = 0.841699** (midpoint between 34th and 35th sorted lockbox real probs; n=1361 reals; max_n_fp = 34).

Metrics at τ_recommended:

| Ckpt | τ | lockbox_real_fpr | lockbox_fake_R | dev_macro_R | dor_dev_fpr | teams_fake_R | viso_enh_R | deeplive_enh_R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `T5C_STEP3500` @ rec | 0.8417 | **0.0250** | **0.6245** | **0.4227** | **0.0600** | 0.5928 | 0.1145 | 0.5605 |
| `P8A_STEP5000` @ rec | 0.8417 | 0.0250 | 0.4625 | 0.3653 | 0.1000 | 0.5256 | 0.0764 | 0.4936 |
| `E2B_STEP3200` @ rec | 0.8417 | 0.0118 | 0.4427 | 0.4190 | 0.0800 | 0.5048 | 0.0709 | 0.4807 |

Source: `_recommended_tau_2026-05-13.csv`.

## 5. Per-identity sanity check at recommended τ

### 5.1 Lockbox-only (5 distinct identities)

Lockbox real has only 5 distinct identities. At T5C τ=0.8417:

| Identity | Chronic-6 | n_videos | n_fp | FPR |
|---|:-:|---:|---:|---:|
| `bla_bla_chow` | Y | 61 | 8 | **0.1311** |
| `dor_shkedi` | Y | 1138 | 26 | 0.0228 |
| `Chikara_Takahashi` | | 25 | 0 | 0.0000 |
| `PC_Generator` | Y | 28 | 0 | 0.0000 |
| `real_dor` | | 109 | 0 | 0.0000 |

Lockbox-only: **0 identities with FPR > 30%** at recommended τ.

Source: `_per_identity_lockbox_2026-05-13.csv`.

### 5.2 Pooled across all real suites (15 distinct identities)

Pooling `teams_real_all_lockbox + teams_real_all_dev + teams_real_poor_quality_dev + teams_real_lighting_extreme_dev + teams_real_dor_dev + teams_real_poor_quality_lockbox` at T5C τ=0.8417:

| Identity | Chronic-6 | n_pooled | n_fp | FPR | Per-suite breakdown |
|---|:-:|---:|---:|---:|---|
| **`Roy_D`** | Y | **246** | **214** | **0.8699** | dev=110/130; poor_q_dev=3/3; lighting_ext_dev=101/113 |
| `Q` | | 36 | 8 | 0.2222 | dev=8/36 |
| `bla_bla_chow` | Y | 1019 | 123 | 0.1207 | lockbox=8/61; dev=60/467; poor_q_dev=26/95; lighting_ext_dev=29/394 |
| `xiang` | Y | 325 | 12 | 0.0369 | dev=6/159; poor_q_dev=3/86; lighting_ext_dev=3/80 |
| `dor_shkedi` | Y | 1250 | 29 | 0.0232 | lockbox=26/1138; dor_dev=3/50; (other suites 0) |
| `PC_Generator` | Y | 1102 | 24 | 0.0218 | dev=17/525; poor_q_dev=2/241; lighting_ext_dev=5/308 |
| `dor` | Y | 345 | 3 | 0.0087 | dev=2/269; poor_q_dev=1/74 |
| (10 others) | | | | 0.0000 | |

Pooled (n≥5): **1 identity with FPR > 30%: `Roy_D` at 0.8699 (214/246).**

Source: `_per_identity_pooled_2026-05-13.csv`.

### 5.3 Cross-ckpt per-identity FPR comparison at recommended τ

Same pooled real suites, at τ=0.8417, all three ckpts:

| Identity | T5C@rec | P8A@rec | E2B@rec | E2B@cal (0.6956) |
|---|---:|---:|---:|---:|
| `Roy_D` | **0.8699 (214/246)** | 0.3659 (90/246) | 0.1138 (28/246) | **0.1707 (42/246)** |
| `Q` | 0.2222 (8/36) | 0.9167 (33/36) | 0.0278 (1/36) | 0.2500 (9/36) |
| `bla_bla_chow` | 0.1207 (123/1019) | 0.0726 (74/1019) | 0.1619 (165/1019) | 0.2875 (293/1019) |
| `xiang` | 0.0369 (12/325) | 0.0123 (4/325) | 0.0954 (31/325) | 0.1877 (61/325) |
| `dor_shkedi` | 0.0232 (29/1250) | 0.0136 (17/1250) | 0.0144 (18/1250) | 0.0272 (34/1250) |
| `PC_Generator` | 0.0218 (24/1102) | 0.1915 (211/1102) | 0.0109 (12/1102) | 0.0408 (45/1102) |
| `Chikara_Takahashi` | 0.0000 (0/40) | 0.3750 (15/40) | 0.0000 (0/40) | 0.0000 (0/40) |
| `dor` | 0.0087 (3/345) | 0.0000 (0/345) | 0.0000 (0/345) | 0.0058 (2/345) |
| (7 others, all 0.0000 across all ckpts) | | | | |

Roy_D-specific cross-cell:
- T5C @ T5C-cal-τ (0.8220): 0.8984 (221/246)
- T5C @ rec-τ (0.8417): 0.8699 (214/246)
- P8A @ rec-τ (0.8417): 0.3659 (90/246)
- P8A @ P8A-cal-τ (0.9143): 0.3252 (80/246)
- E2B @ rec-τ (0.8417): 0.1138 (28/246)
- E2B @ E2B-cal-τ (0.6956): 0.1707 (42/246)

Source: `_per_identity_cross_ckpt_2026-05-13.csv`.

## 6. Headline tradeoff: T5C @ rec-τ vs E2B @ E2B-cal-τ (production)

Subtract E2B-at-its-deployment-τ from T5C-at-recommended-τ:

| Metric | T5C @ τ=0.8417 | E2B @ τ=0.6956 (deployed) | Δ (T5C − E2B) |
|---|---:|---:|---:|
| `lockbox_real_fpr` | 2.50% | 2.50% | **+0.00 pp** |
| `lockbox_fake_recall` | 62.45% | 64.82% | **−2.37 pp** |
| `dev_macro_recall` | 42.27% | 51.51% | **−9.24 pp** |
| `dor_dev_fpr` | 6.00% | 14.00% | **−8.00 pp** |
| `teams_fake_dev_recall` | 59.28% | 68.33% | **−9.05 pp** |
| `viso_enh_dev_recall` | 11.45% | 5.27% | **+6.18 pp** |
| `deeplive_enh_dev_recall` | 56.05% | 80.92% | **−24.87 pp** |
| `Roy_D` pooled FPR | 86.99% | 17.07% | **+69.92 pp** |

### 6.1 Alternative: T5C @ T5C-cal-τ vs E2B @ E2B-cal-τ

(Same comparison but T5C at its native calibrated τ=0.8220 instead of forced-to-0.025-ceiling τ=0.8417.)

| Metric | T5C @ τ=0.8220 (cal) | E2B @ τ=0.6956 (deployed) | Δ |
|---|---:|---:|---:|
| `lockbox_real_fpr` | 3.09% | 2.50% | +0.59 pp |
| `lockbox_fake_recall` | 67.19% | 64.82% | +2.37 pp |
| `dev_macro_recall` | 49.26% | 51.51% | −2.25 pp |
| `dor_dev_fpr` | 10.00% | 14.00% | −4.00 pp |
| `teams_fake_dev_recall` | 63.35% | 68.33% | −4.98 pp |
| `viso_enh_dev_recall` | 17.82% | 5.27% | +12.55 pp |
| `deeplive_enh_dev_recall` | 66.61% | 80.92% | −14.31 pp |
| `Roy_D` pooled FPR | 89.84% | 17.07% | +72.77 pp |

## 7. File index

| File | Content |
|---|---|
| `run_t5c_deployment_curve.py` | source script |
| `_t5c_pareto_2026-05-13.csv` | T5C step3500 metrics at 13 τ values |
| `_comparison_2026-05-13.csv` | T5C/P8A/E2B at 12 grid τ + each at own cal-τ + each at recommended τ |
| `_recommended_tau_2026-05-13.csv` | T5C/P8A/E2B at τ=0.8417 (single row each) |
| `_per_identity_lockbox_2026-05-13.csv` | T5C @ rec-τ lockbox-only (5 ids) |
| `_per_identity_pooled_2026-05-13.csv` | T5C @ rec-τ pooled across 6 real suites (15 ids) |
| `_per_identity_cross_ckpt_2026-05-13.csv` | per-identity FPR for T5C/P8A/E2B at both rec-τ and each cal-τ |
| `_results_2026-05-13.json` | full JSON dump of all above |
