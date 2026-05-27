# Step250 (Slot 1 head-retrain) deployment τ-tradeoff curve — FACTS (2026-05-13)

> **Status: factual-only.** Numbers + tables + cross-references. No interpretation, no verdict language.
> Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, unfortunately, remarkably.

## 0. Scope and provenance

- **Question**: build the same deployment-curve / Pareto data sheet as
  `../r13_overnight_t5c_deployment_curve_2026-05-13/` for the Slot 1
  head-retrain `top_n_step250` checkpoint (W&B run `fz84lq5k`).
  Open the verdict on Roy_D pooled FPR, Chikara_Takahashi__s22 context, and fake-recall preservation at calibrated τ.
- **Ckpt**: `gs://training-job-outputs/best_checkpoints/fz84lq5k/top_n_effort_20260513_step250_auc0.9847_eer0.0437.pth`
- **W&B run**: `fz84lq5k`
- **LoRA wrap**: rank=16, alpha=32, target_layers=[10,11], target_modules=[attn.in_proj/out_proj, mlp.c_fc/c_proj]; applied BEFORE state_dict load; load reports 0 missing / 0 unexpected (parameters populate from ckpt with non-zero values).
- **Inputs**:
  - step250 video-level scores derived by aggregating per-frame CSVs at `./outputs/scores_step250_<suite>.csv` (avg of `prob_fake` over frames per video_id).
  - P8A, E2B, T5C video-level CSVs come from `../r13_overnight_partial_scorecard_2026-05-13/_reports_cache/<suite>_<ckpt>_videos_report.csv` (per-video).
- **Score basis**: video-level `avg_video_prob`.
- **Code**: `score_step250_remaining_suites.py` + `run_step250_deployment_curve.py` in this folder. Re-runnable.

### 0.1 Identity-parsing note

`run_step250_deployment_curve.py::parse_identity` splits anchor `video_id` strings (which lack an explicit `identity_key` column) by the first `__seg_` or `__seq` marker, yielding e.g. `PC_Generator__s22` from `PC_Generator__s22__seg_100.0__real`. The T5C deployment-curve harness used a first-token `__`-split, which collapsed `Q__s6`→`Q`, `Chikara_Takahashi__s22`→`Chikara_Takahashi`, etc. Both schemes agree on canonical chronic-6 names (Roy_D, dor_shkedi). Per-identity counts below use the marker-aware parse for consistency with the manifest's `identity_key` field on step250's own CSVs.

## 1. Anchor τ values and sample sizes (video-level)

| Ckpt | Calibrated τ | dev_real n | poor_q_dev n | lighting_ext_dev n | lockbox_real n | lockbox_fake n | dor_dev n | teams_fake_dev n | viso_enh n | dl_enh n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `step250` | 0.779558 | 3253 | 923 | 1401 | 1361 | 253 | 50 | 2409 | 550 | 545 |
| `P8A_REFERENCE_STEP5000` | 0.914263 | 3253 | 923 | 1401 | 1361 | 253 | 50 | 2409 | 550 | 545 |
| `E2B_TOP_N_STEP3200` (deployed) | 0.695595 | 3253 | 923 | 1401 | 1361 | 253 | 50 | 2409 | 550 | 545 |
| `T5C_PERIODIC_STEP3500` | 0.821989 | 3253 | 923 | 1401 | 1361 | 253 | 50 | 2409 | 550 | 545 |

step250's calibrated τ comes from this folder's own data: smallest τ s.t. `teams_real_all_dev` video-level real_fpr ≤ 0.07 (midpoint convention; n=3253; max_n_fp=227). Anchor τ values are imported from `../r13_overnight_partial_scorecard_2026-05-13/_partial_scorecard_2026-05-13.csv`.

## 2. step250 Pareto curve

Video-level metrics at user-specified τ grid plus step250's calibrated τ and the recommended τ. Columns: real-FPR and fake-recall per suite + macro aggregates.

| τ | dev_real_fpr | lockbox_real_fpr | dor_dev_fpr | poor_q_dev_fpr | lighting_ext_dev_fpr | teams_fake_R | viso_enh_R | deeplive_enh_R | dev_macro_R | lockbox_fake_R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5000 | 0.1869 | 0.0860 | 0.4400 | 0.1582 | 0.1892 | 0.7136 | 0.2836 | 0.5706 | 0.5226 | 0.9170 |
| 0.6000 | 0.1586 | 0.0625 | 0.3400 | 0.1246 | 0.1627 | 0.6592 | 0.2018 | 0.4807 | 0.4472 | 0.8498 |
| 0.7000 | 0.1340 | 0.0463 | 0.3000 | 0.0997 | 0.1435 | 0.5890 | 0.1273 | 0.3339 | 0.3501 | 0.7826 |
| 0.7500 | 0.1091 | 0.0301 | 0.1800 | 0.0845 | 0.1213 | 0.5214 | 0.0709 | 0.1945 | 0.2623 | 0.6917 |
| **0.7694 (rec)** | **0.0882** | **0.0250** | **0.1600** | 0.0672 | 0.0999 | 0.4641 | 0.0382 | 0.1028 | **0.2017** | **0.6087** |
| **0.7796 (cal)** | **0.0698** | **0.0154** | **0.1200** | 0.0488 | 0.0799 | 0.4126 | 0.0200 | 0.0477 | **0.1601** | **0.4704** |
| 0.7800 | 0.0686 | 0.0154 | 0.1200 | 0.0488 | 0.0799 | 0.4076 | 0.0164 | 0.0440 | 0.1560 | 0.4664 |
| 0.8000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.8200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.8500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.9000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Source: `_step250_pareto_2026-05-13.csv`.

### 2.1 Observations from the curve

- All step250 metrics collapse to 0.0000 at τ ≥ 0.80 — the per-video `avg_video_prob` distribution is bounded above ~0.79.
- `lockbox_real_fpr ≤ 0.025` is reached at τ = 0.769443 (max_n_fp = 34 of n=1361; midpoint between sorted-prob ranks 34/35).
- Above the rec-τ, both real-FPRs and fake-recalls drop sharply: at τ=0.7796 (cal-τ), lockbox_fake_recall = 0.4704 (already down 14 pp from rec-τ's 0.6087); at τ=0.7800, recall is 0.4664. The "deployable window" lies in [0.769, 0.800).

## 3. Comparison at common τ grid — step250 vs P8A vs E2B vs T5C

Same τ values applied to each ckpt's video-level probs. Columns: `lockbox_real_fpr / lockbox_fake_recall / dev_macro_recall / dor_dev_fpr`.

| τ | s250 lb_fpr | s250 lb_R | s250 macro_R | s250 dor_fpr | P8A lb_fpr | P8A lb_R | P8A macro_R | P8A dor_fpr | E2B lb_fpr | E2B lb_R | E2B macro_R | E2B dor_fpr | T5C lb_fpr | T5C lb_R | T5C macro_R | T5C dor_fpr |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5000 | 0.0860 | 0.9170 | 0.5226 | 0.4400 | 0.0617 | 0.6522 | 0.5297 | 0.3000 | 0.0647 | 0.8261 | 0.5923 | 0.2400 | 0.4431 | 0.9368 | 0.8790 | 0.4600 |
| 0.6000 | 0.0625 | 0.8498 | 0.4472 | 0.3400 | 0.0470 | 0.6047 | 0.4864 | 0.3000 | 0.0419 | 0.7470 | 0.5518 | 0.1800 | 0.2667 | 0.9091 | 0.8279 | 0.3200 |
| 0.7000 | 0.0463 | 0.7826 | 0.3501 | 0.3000 | 0.0411 | 0.5415 | 0.4438 | 0.2400 | 0.0250 | 0.6403 | 0.5140 | 0.1400 | 0.1293 | 0.8458 | 0.7431 | 0.2400 |
| 0.7500 | 0.0301 | 0.6917 | 0.2623 | 0.1800 | 0.0338 | 0.5296 | 0.4240 | 0.2400 | 0.0198 | 0.5771 | 0.4866 | 0.1200 | 0.0801 | 0.7984 | 0.6575 | 0.1800 |
| 0.7800 | 0.0154 | 0.4664 | 0.1560 | 0.1200 | 0.0316 | 0.5138 | 0.4050 | 0.2000 | 0.0176 | 0.5534 | 0.4663 | 0.1200 | 0.0632 | 0.7470 | 0.5924 | 0.1400 |
| 0.8000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0287 | 0.5020 | 0.3918 | 0.1800 | 0.0162 | 0.5336 | 0.4572 | 0.1200 | 0.0478 | 0.7154 | 0.5535 | 0.1400 |
| 0.8200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0272 | 0.4980 | 0.3793 | 0.1800 | 0.0118 | 0.4980 | 0.4427 | 0.0800 | 0.0309 | 0.6798 | 0.4995 | 0.1000 |
| 0.8500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0250 | 0.4427 | 0.3612 | 0.1400 | 0.0118 | 0.4269 | 0.4156 | 0.0800 | 0.0220 | 0.6047 | 0.3984 | 0.0600 |
| 0.9000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0198 | 0.4111 | 0.3109 | 0.1200 | 0.0088 | 0.3636 | 0.3627 | 0.0600 | 0.0029 | 0.3676 | 0.2198 | 0.0400 |

Source: `_comparison_2026-05-13.csv`.

### 3.1 Each ckpt at its OWN calibrated τ (operational points)

| Ckpt | τ_cal | lockbox_real_fpr | lockbox_fake_R | dev_macro_R | dor_dev_fpr | teams_fake_R | viso_R | dl_R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `step250` | 0.7796 | **0.0154** | **0.4704** | **0.1601** | **0.1200** | 0.4126 | 0.0200 | 0.0477 |
| `P8A_STEP5000` | 0.9143 | 0.0184 | 0.3874 | 0.3003 | 0.0800 | 0.5259 | 0.1364 | 0.2385 |
| `E2B_STEP3200` (deployed) | 0.6956 | 0.0250 | 0.6482 | 0.5151 | 0.1400 | 0.6833 | 0.0527 | 0.8092 |
| `T5C_STEP3500` | 0.8220 | 0.0309 | 0.6719 | 0.4926 | 0.1000 | 0.6335 | 0.1782 | 0.6661 |

Cross-checks against the partial scorecard:
- step250 dev_real_fpr at cal-τ = 0.0698 matches the quick eval doc's 6.95% (226/3253).
- step250 lockbox_all FPR at cal-τ = 0.0154 (21/1361) matches the quick eval's 1.54%.
- P8A: `lockbox_fake_recall=0.3874` matches scorecard 0.3874.
- E2B: `lockbox_fake_recall=0.6482` matches scorecard 0.6482.

## 4. Recommended τ for step250 (lb_real_fpr ≤ 0.025 = E2B-comparable ceiling)

Precise τ that puts step250 lockbox_real_fpr at 0.025: **τ = 0.769443** (midpoint between rank 34 and 35 of sorted lockbox real probs; n=1361; max_n_fp = 34).

Metrics at τ_recommended (all 4 ckpts):

| Ckpt | τ | lockbox_real_fpr | lockbox_fake_R | dev_macro_R | dor_dev_fpr | teams_fake_R | viso_R | dl_R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `step250` @ rec | 0.7694 | **0.0250** | **0.6087** | **0.2017** | **0.1600** | 0.4641 | 0.0382 | 0.1028 |
| `P8A_STEP5000` @ rec | 0.7694 | 0.0316 | 0.5217 | 0.4101 | 0.2000 | 0.6106 | 0.2455 | 0.3743 |
| `E2B_STEP3200` @ rec | 0.7694 | 0.0191 | 0.5613 | 0.4729 | 0.1200 | 0.6505 | 0.0491 | 0.7193 |
| `T5C_STEP3500` @ rec | 0.7694 | 0.0698 | 0.7549 | 0.6160 | 0.1400 | 0.7227 | 0.3327 | 0.7927 |

Source: `_recommended_tau_2026-05-13.csv`.

## 5. step250 vs E2B (deployed) — matched-ceiling headline

Subtract E2B-at-its-own-cal-τ (= operational deployment) from step250-at-recommended-τ. Both ckpts at lb_real_fpr = 0.0250.

| Metric | step250 @ τ=0.7694 | E2B @ τ=0.6956 (deployed) | Δ (s250 − E2B) |
|---|---:|---:|---:|
| `lockbox_real_fpr` | 2.50% | 2.50% | **+0.00 pp** |
| `lockbox_fake_recall` | 60.87% | 64.82% | **−3.95 pp** |
| `dev_macro_recall` | 20.17% | 51.51% | **−31.34 pp** |
| `dor_dev_fpr` | 16.00% | 14.00% | **+2.00 pp** |
| `teams_fake_dev_recall` | 46.41% | 68.33% | **−21.92 pp** |
| `viso_enh_dev_recall` | 3.82% | 5.27% | **−1.45 pp** |
| `deeplive_enh_dev_recall` | 10.28% | 80.92% | **−70.64 pp** |
| `poor_quality_dev_fpr` | 6.72% | 8.34% | **−1.63 pp** |
| `lighting_extreme_dev_fpr` | 9.99% | 10.49% | **−0.50 pp** |

### 5.1 Alternative: step250 @ step250-cal-τ vs E2B @ E2B-cal-τ

(step250 at native τ=0.7796 instead of forced-to-0.025 τ=0.7694.)

| Metric | step250 @ τ=0.7796 (cal) | E2B @ τ=0.6956 (deployed) | Δ |
|---|---:|---:|---:|
| `lockbox_real_fpr` | 1.54% | 2.50% | **−0.96 pp** |
| `lockbox_fake_recall` | 47.04% | 64.82% | **−17.78 pp** |
| `dev_macro_recall` | 16.01% | 51.51% | **−35.50 pp** |
| `dor_dev_fpr` | 12.00% | 14.00% | **−2.00 pp** |
| `teams_fake_dev_recall` | 41.26% | 68.33% | **−27.07 pp** |
| `viso_enh_dev_recall` | 2.00% | 5.27% | **−3.27 pp** |
| `deeplive_enh_dev_recall` | 4.77% | 80.92% | **−76.15 pp** |

## 6. Per-identity sanity check at recommended τ (step250)

### 6.1 Lockbox-only (5 distinct identities)

Lockbox-real has only 5 distinct identities. At step250 τ=0.7694:

| Identity | Chronic-6 | n_videos | n_fp | FPR |
|---|:-:|---:|---:|---:|
| `Chikara_Takahashi__s22` | | 25 | 19 | **0.7600** |
| `PC_Generator__s15` | Y | 28 | 10 | **0.3571** |
| `bla_bla_chow__s1` | Y | 61 | 0 | 0.0000 |
| `dor_shkedi` | Y | 1138 | 3 | 0.0026 |
| `real_dor` | | 109 | 0 | 0.0000 |

(Source: `_per_identity_lockbox_2026-05-13.csv`.) Lockbox-only counts 2 of 5 identities with FPR > 30%.

### 6.2 Pooled across 5 real suites (15 distinct identities, n≥5)

Pooling `teams_real_all_lockbox + teams_real_all_dev + teams_real_poor_quality_dev + teams_real_lighting_extreme_dev + teams_real_dor_dev` at step250 τ=0.7694:

| Identity | Chronic-6 | n_pooled | n_fp | FPR | Per-suite breakdown |
|---|:-:|---:|---:|---:|---|
| `Q__s6` | | 36 | 33 | **0.9167** | dev=33/36 |
| `Chikara_Takahashi__s22` | | 25 | 19 | **0.7600** | lockbox=19/25 |
| `PC_Generator__s22` | Y | 195 | 132 | **0.6769** | dev=90/143; lighting_ext_dev=42/52 |
| `PC_Generator__s45` | Y | 103 | 66 | **0.6408** | dev=31/59; poor_q_dev=30/37; lighting_ext_dev=5/7 |
| **`Roy_D`** | Y | **246** | **141** | **0.5732** | dev=70/130; poor_q_dev=2/3; lighting_ext_dev=69/113 |
| `PC_Generator__s15` | Y | 28 | 10 | **0.3571** | lockbox=10/28 |
| `bla_bla_chow__s2` | Y | 343 | 73 | 0.2128 | dev=43/160; poor_q_dev=14/90; lighting_ext_dev=16/93 |
| `dor_shkedi_real` | | 50 | 8 | 0.1600 | dor_dev=8/50 (note: this is the `dor` slice cohort; identity_key on manifest = `dor_shkedi_real`) |
| `Xiang_Xiang2_Feng__s23` | | 67 | 3 | 0.0448 | dev=3/63; lighting_ext_dev=0/4 |
| `bla_bla_chow` | Y | 613 | 26 | 0.0424 | dev=2/307; lighting_ext_dev=24/301; poor_q_dev=0/5 |
| `xiang` | Y | 325 | 12 | 0.0369 | dev=6/159; poor_q_dev=3/86; lighting_ext_dev=3/80 |
| `Test_Cam__s73` | | 220 | 2 | 0.0091 | dev=2/141; poor_q_dev=0/79 |
| `dor_shkedi` | Y | 1138 | 5 | 0.0044 | lockbox=5/1138 (note: quick-eval at cal-τ 0.7796 = 3/1138) |
| `dor` | Y | 345 | 1 | 0.0029 | poor_q_dev=1/74; dev=0/269 |
| (16 others, all 0.0000 at n≥5) | | | | |

Pooled (n≥5): **6 identities with FPR > 30% at recommended τ.**

Source: `_per_identity_pooled_2026-05-13.csv`.

### 6.3 Cross-ckpt per-identity FPR comparison at step250's recommended τ

Same pooled real suites, at τ=0.7694, all four ckpts:

| Identity | step250 @ rec | P8A @ rec | E2B @ rec | T5C @ rec | E2B @ E2B-cal (deployed) |
|---|---:|---:|---:|---:|---:|
| `Q__s6` | **0.9167 (33/36)** | 0.9444 (34/36) | 0.0833 (3/36) | 0.2778 (10/36) | 0.2500 (9/36) |
| `Chikara_Takahashi__s22` | **0.7600 (19/25)** | 0.4400 (11/25) | 0.0000 (0/25) | 0.0800 (2/25) | 0.0000 (0/25) |
| `PC_Generator__s22` | **0.6769 (132/195)** | 0.8821 (172/195) | 0.0205 (4/195) | 0.2513 (49/195) | 0.0359 (7/195) |
| `PC_Generator__s45` | **0.6408 (66/103)** | 0.4466 (46/103) | 0.1650 (17/103) | 0.2427 (25/103) | 0.2524 (26/103) |
| **`Roy_D`** | **0.5732 (141/246)** | 0.4065 (100/246) | 0.1463 (36/246) | 0.9472 (233/246) | 0.1707 (42/246) |
| `PC_Generator__s15` | 0.3571 (10/28) | 0.6071 (17/28) | 0.1429 (4/28) | 0.0357 (1/28) | 0.2143 (6/28) |
| `bla_bla_chow__s2` | 0.2128 (73/343) | 0.1487 (51/343) | 0.2332 (80/343) | 0.2478 (85/343) | 0.4111 (141/343) |
| `dor_shkedi_real` | 0.1600 (8/50) | 0.0000 (0/50) | 0.0000 (0/50) | 0.0000 (0/50) | 0.0000 (0/50) |
| `bla_bla_chow` | 0.0424 (26/613) | 0.0522 (32/613) | 0.2284 (140/613) | 0.1452 (89/613) | 0.2545 (156/613) |
| `xiang` | 0.0369 (12/325) | 0.0123 (4/325) | 0.1538 (50/325) | 0.0554 (18/325) | 0.1877 (61/325) |
| `dor_shkedi` (combined w/ dor_dev) | 0.0109 (13/1188) | 0.0210 (25/1188) | 0.0227 (27/1188) | 0.0673 (80/1188) | 0.0286 (34/1188) |
| `dor_shkedi` (lockbox only) | 0.0044 (5/1138) | 0.0061 (7/1138) | 0.0114 (13/1138) | 0.0220 (25/1138) | 0.0272 (31/1138) |

Source: `_per_identity_cross_ckpt_2026-05-13.csv`.

Roy_D-specific cross-cell:
- step250 @ rec-τ (0.7694): **0.5732 (141/246)** — pillar-1 chronic axis exceeds the 30% bar.
- step250 @ cal-τ (0.7796): 0.5366 (132/246) — modest improvement at higher τ; Roy_D probs still cluster within the 0.77-0.80 band.
- P8A @ rec-τ (0.7694): 0.4065 (100/246).
- E2B @ rec-τ (0.7694): 0.1463 (36/246).
- E2B @ E2B-cal-τ (0.6956): 0.1707 (42/246).
- T5C @ rec-τ (0.7694): 0.9472 (233/246).
- T5C @ T5C-cal-τ (0.8220): 0.8699 (214/246).

## 7. Chikara_Takahashi__s22 context (lockbox-only)

n=25 video clips of one identity not present in the chronic-6 set.

| Ckpt | τ_label | τ | n_fp / n | FPR | prob_max | prob_p50 |
|---|---|---:|---:|---:|---:|---:|
| `step250` | rec | 0.7694 | 19/25 | **76.00%** | 0.79 | 0.78 |
| `step250` | cal | 0.7796 | 16/25 | 64.00% | 0.79 | 0.78 |
| `P8A_REFERENCE_STEP5000` | rec | 0.7694 | 11/25 | 44.00% | 0.99 | 0.76 |
| `P8A_REFERENCE_STEP5000` | cal | 0.9143 | 9/25 | 36.00% | 0.99 | 0.76 |
| `E2B_TOP_N_STEP3200` | rec | 0.7694 | 0/25 | 0.00% | 0.35 | 0.02 |
| `E2B_TOP_N_STEP3200` | cal | 0.6956 | 0/25 | 0.00% | 0.35 | 0.02 |
| `T5C_PERIODIC_STEP3500` | rec | 0.7694 | 2/25 | 8.00% | 0.79 | 0.34 |
| `T5C_PERIODIC_STEP3500` | cal | 0.8220 | 0/25 | 0.00% | 0.79 | 0.34 |

Source: `_chikara_context_2026-05-13.csv`.

Score-distribution diagnostics on Chikara_Takahashi__s22 (lockbox; n=25):
- step250 video-level probs cluster near 0.78-0.79 (median 0.78, max 0.79). No video has a prob below ~0.10; the identity is uniformly scored "fake-side" by step250.
- P8A: probs span 0.006-0.989; median 0.76. High-variance distribution.
- E2B: probs span 0.006-0.349; median 0.025. Uniformly scored "real-side".
- T5C: probs span 0.063-0.786; median 0.34. Bimodal with a low-end cluster.

## 8. Sub-identity cross-cohort table at rec-τ (compact)

| Identity | step250 | P8A | E2B | T5C |
|---|---:|---:|---:|---:|
| Q__s6 (n=36) | 91.7% | 94.4% | 8.3% | 27.8% |
| Chikara_Takahashi__s22 (n=25) | 76.0% | 44.0% | 0.0% | 8.0% |
| PC_Generator__s22 (n=195) | 67.7% | 88.2% | 2.1% | 25.1% |
| PC_Generator__s45 (n=103) | 64.1% | 44.7% | 16.5% | 24.3% |
| Roy_D (n=246) | 57.3% | 40.7% | 14.6% | 94.7% |
| PC_Generator__s15 (n=28) | 35.7% | 60.7% | 14.3% | 3.6% |
| bla_bla_chow__s2 (n=343) | 21.3% | 14.9% | 23.3% | 24.8% |
| dor_shkedi_real (n=50) | 16.0% | 0.0% | 0.0% | 0.0% |
| dor_shkedi (n=1138) | 0.4% | 0.6% | 1.1% | 2.2% |

## 9. Sample-size / sanity sums

- Pooled FPR across all real suites at step250 rec-τ: **531 / 6988 = 7.60%** (across 31 distinct identities).
- Pooled FPR across all real suites at step250 cal-τ (0.7796): aggregate count not computed in this script; per-suite FPRs in §2 sum compatible with cell-level numbers.
- Total may6 firings (step250 @ τ=0.5, from quick eval): 5/92 (median prob 0.20).

## 10. File index

| File | Content |
|---|---|
| `score_step250_remaining_suites.py` | Score step250 on the 7 contract suites not already scored by `run_step250_eval.py` |
| `run_step250_deployment_curve.py` | Build the τ-tradeoff curve + per-identity tables + Chikara context |
| `outputs/scores_step250_*.csv` | Per-frame scores (9 contract suites + may6) |
| `_step250_pareto_2026-05-13.csv` | step250 metrics at 11 τ values |
| `_comparison_2026-05-13.csv` | step250/P8A/E2B/T5C at grid τ + each at own cal-τ + each at recommended τ |
| `_recommended_tau_2026-05-13.csv` | step250/P8A/E2B/T5C at τ=0.7694 (single row each) |
| `_per_identity_lockbox_2026-05-13.csv` | step250 @ rec-τ lockbox-only (5 ids) |
| `_per_identity_pooled_2026-05-13.csv` | step250 @ rec-τ pooled across 5 real suites (31 ids) |
| `_per_identity_cross_ckpt_2026-05-13.csv` | per-identity FPR for step250/P8A/E2B/T5C at both rec-τ and each cal-τ |
| `_chikara_context_2026-05-13.csv` | Chikara_Takahashi__s22 FPR across ckpts × τ |
| `_results_2026-05-13.json` | Full JSON dump of all above |
