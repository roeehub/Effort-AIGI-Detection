# P8A v3-Substrate Scorecard Analog (2026-05-01)

## TL;DR

Frame-level promotion-contract analog applied to P8A_step5000 lockbox (839 frames) on the v3-substrate (production-tight RFA=0.85 crop).

- **Default-floor (0.70) selection on prod arm**: tier=1 (FAILS contract). tau=0.9945 recall=0.1106 fpr=0.0676.
- **Relaxed-floor (0.30) selection on prod arm**: tier=1 (FAILS contract). tau=0.9945 recall=0.1106 fpr=0.0676.
- **Production-FPR gap at recall=0.30**: asis min_fpr=0.0362 -> prod min_fpr=0.1787 (delta=+0.1425).
- **Recall=0.70 reachability**: asis achievable (min_fpr=0.2198); prod achievable (min_fpr=0.4155).
- **Canonical (video-level) reference**: tau=0.9156 lockbox_real_fpr=0.0184 lockbox_fake_recall=0.387 dev_fake_macro_recall=0.300 (FAILS 0.70 floor).

## Method

Loaded the v3-retag parquet (839 rows) and merged with `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` on `original_local_path` to bring in `is_pose_extreme`, `is_no_face`, `face_area_ratio`. Built a quantile-spaced threshold grid of up to 5000 points over the union of asis+prod prob distributions.

For each tau, computed FPR (over reals) and recall (over fakes). Selection mirrors `arena/score_teams_promotion_contract._threshold_sort_key`: tier 0 = both FPR<=0.07 and recall>=floor; tier 1 = FPR<=0.07 only; tier 2 = FPR violated. Within tier, max recall, then highest tau, then lowest fpr.

**Caveat**: canonical scorer is video-level on dev pool then reads out lockbox. This analog is frame-level on lockbox-only — same underlying logic, but absolute numbers are not directly comparable to the canonical scorecard. Asis-vs-prod *delta* is the load-bearing signal.

## Asis-vs-Prod Pareto Headline (full lockbox)

| recall_target | asis min_fpr | prod min_fpr | delta (prod - asis) |
|---|---|---|---|
| 0.20 | 0.0072 | 0.1546 | +0.1473 |
| 0.30 | 0.0362 | 0.1787 | +0.1425 |
| 0.50 | 0.1135 | 0.3043 | +0.1908 |
| 0.70 | 0.2198 | 0.4155 | +0.1957 |

### modern_v2 subset

| recall_target | asis min_fpr | prod min_fpr | delta |
|---|---|---|---|
| 0.20 | 0.0071 | 0.0391 | +0.0320 |
| 0.30 | 0.0071 | 0.0890 | +0.0819 |
| 0.50 | 0.0214 | 0.2206 | +0.1993 |
| 0.70 | 0.0712 | 0.3345 | +0.2633 |

## Per-Capture-Mode (asis vs prod, full lockbox)

| mode | n_real | recall | asis min_fpr | prod min_fpr | delta |
|---|---|---|---|---|---|
| normal_photo | n=240 | 0.20 | 0.0000 | 0.0375 | +0.0375 |
| normal_photo | n=240 | 0.30 | 0.0083 | 0.0625 | +0.0542 |
| normal_photo | n=240 | 0.50 | 0.0208 | 0.2292 | +0.2083 |
| normal_photo | n=240 | 0.70 | 0.0833 | 0.3583 | +0.2750 |
| phone_screen | n=68 | 0.20 | 0.0000 | 0.0000 | +0.0000 |
| phone_screen | n=68 | 0.30 | 0.0000 | 0.0147 | +0.0147 |
| phone_screen | n=68 | 0.50 | 0.0000 | 0.0588 | +0.0588 |
| phone_screen | n=68 | 0.70 | 0.0147 | 0.1324 | +0.1176 |
| webcam | n=105 | 0.20 | 0.0286 | 0.5238 | +0.4952 |
| webcam | n=105 | 0.30 | 0.1143 | 0.5524 | +0.4381 |
| webcam | n=105 | 0.50 | 0.3905 | 0.6381 | +0.2476 |
| webcam | n=105 | 0.70 | 0.6571 | 0.7333 | +0.0762 |

## Per-Method Recall at Selected Taus (full lockbox)

| method | n_fake | arm | floor | tau | recall |
|---|---|---|---|---|---|
| teams_capture_cam_test_s33 | 334 | prob_fake_asis | default_0.70 | 0.9524 | 0.2605 |
| teams_capture_cam_test_s33 | 334 | prob_fake_asis | relaxed_0.30 | 0.9524 | 0.2605 |
| teams_capture_cam_test_s33 | 334 | prob_fake_prod | default_0.70 | 0.9945 | 0.0000 |
| teams_capture_cam_test_s33 | 334 | prob_fake_prod | relaxed_0.30 | 0.9945 | 0.0000 |
| teams_capture_pc_generator_s15 | 91 | prob_fake_asis | default_0.70 | 0.9524 | 0.9451 |
| teams_capture_pc_generator_s15 | 91 | prob_fake_asis | relaxed_0.30 | 0.9524 | 0.9451 |
| teams_capture_pc_generator_s15 | 91 | prob_fake_prod | default_0.70 | 0.9945 | 0.5165 |
| teams_capture_pc_generator_s15 | 91 | prob_fake_prod | relaxed_0.30 | 0.9945 | 0.5165 |

## Implication for P18 Baseline Floor

The asis-vs-prod delta isolates how much the canonical scorer overstates P8A's production-FPR. Any P18 candidate must (a) close the asis-prod FPR gap (or beat the prod-arm number directly), and (b) achieve recall=0.30 at fpr<=prod_full_30 to be a credible improvement over P8A on production-honest crops. Numbers above set that floor.
