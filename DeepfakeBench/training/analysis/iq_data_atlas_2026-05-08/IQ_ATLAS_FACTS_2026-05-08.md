# IQ data atlas FACTS (2026-05-08)

> **Status: factual-only.** No interpretation. Forbidden words: succeeds, fails,
> wins, promotes, deployment-grade.
>
> Source data: `outputs/per_frame.parquet`, `outputs/per_pool_summary.csv`,
> `outputs/cross_pool_compare.csv` produced by `build_iq_atlas.py` on
> 2026-05-08 (CPU-only; multiprocessing.Pool(8); gsutil parallel download via
> ThreadPoolExecutor).
>
> Companion figures: `figs/histogram_<metric>_all_pools.png`,
> `figs/pool_<pool>_quad.png`,
> `figs/train_vs_eval_vs_lockbox_<metric>.png`.

## 1. Question

User eyeballed chronic-FP frames in the canary substrate and noticed they look
pixelated. The pixelation was confirmed to be in the data (cached crops match
GCS-source dims; not viewer-side downscale). See
`analysis/p2_eval_2026-05-08/d1_d4_cpu/CANARY_RESOLUTION_FACTS_2026-05-08.md`.

The user requested a comprehensive cross-pool IQ atlas to inform three
decisions:

1. Whether to filter "ultra-bad" frames out of TRAINING.
2. Whether to standardize on F4-style filtering for EVAL/lockbox readouts.
3. Where to set the deployment-side IQ-gate threshold.

This document is the cross-pool measurement that informs those decisions. It
contains numbers + tables only. No recommendations.

## 2. Method

**Features.** Per-frame IQ panel computed by
`build_iq_atlas.py:per_frame_attrs`. Feature formulas match
`analysis/dor_drift_mechanism_2026-05-06/run_analysis.py:compute_iq_axes`
exactly so the production-reference rows fold in directly without unit
mismatch:

- `h`, `w`, `min_dim`, `max_dim`, `aspect_ratio`
- `lap_var` — `cv2.Laplacian(gray).var()` (sharpness proxy)
- `luma_mean`, `luma_std` — HSV V channel mean / std
- `saturation_mean` — HSV S channel mean
- `contrast_l` — LAB L channel std
- `edge_mag` — `cv2.Sobel` magnitude mean (ksize=3)
- `color_a_dev` — `mean(|LAB.A - 128|)` (LAB A deviation from neutral)
- `color_b_dev` — `mean(|LAB.B - 128|)` (LAB B deviation from neutral)
- `skin_frac` — YCbCr skin range fraction
- `bytes` — JPEG / PNG file size

**Sampling.** Random sample of N=500 frames per pool (oversample 1.5× to
absorb download / decode loss). Manifest-based pools enumerate frame URIs from
the three manifests in `arena/manifests/`. Training pools enumerate via
`gsutil ls` over `gs://teams-faces-data-test-2914-fake-4420-real-feb-28/{real,fake}/`
and df40 method directories, then sample uniformly within.

**Speed-ups.** ThreadPoolExecutor over `gsutil cp` (24 workers) for download;
`multiprocessing.Pool(8)` for CPU decode + feature compute. Pool dirs deleted
after measurement to bound disk usage. Per-pool parquet caches in `_cache/`
make the script idempotent — re-runs skip already-measured pools.

**Frame counts.**

- Total pools sampled: **35**
- Total frames measured: **15,236**
- Worker count: 8 (constrained per project memory `feedback_sklearn_njobs.md`)

## 3. Pool inventory

| pool | role | n_frames |
|---|---|---:|
| canary_chronic_real | canary_chronic_real | 300 |
| canary_fake | canary_fake | 200 |
| canary_other_real | canary_other_real | 300 |
| deeplive_enhanced_dev | dev_fake | 500 |
| teams_fake_all_dev | dev_fake | 500 |
| visomaster_enhanced_macro_dev | dev_fake | 500 |
| teams_real_all_dev | dev_real | 500 |
| teams_real_dor_dev | dev_real | 81 |
| teams_real_lighting_extreme_dev | dev_real | 500 |
| teams_real_poor_quality_dev | dev_real | 500 |
| hdtf_fake_clean_dev | hdtf_fake | 500 |
| hdtf_fake_clean_lockbox | hdtf_fake | 500 |
| hdtf_fake_teams_dev | hdtf_fake | 500 |
| hdtf_fake_teams_lockbox | hdtf_fake | 500 |
| hdtf_real_clean_dev | hdtf_real | 500 |
| hdtf_real_clean_lockbox | hdtf_real | 500 |
| hdtf_real_teams_dev | hdtf_real | 500 |
| hdtf_real_teams_lockbox | hdtf_real | 500 |
| teams_fake_all_lockbox | lockbox_fake | 425 |
| teams_real_all_lockbox | lockbox_real | 500 |
| prod_ref_dor_evening_local | prod_ref_real | 200 |
| prod_ref_dor_may5_teams | prod_ref_real | 30 |
| prod_ref_dor_morning_local | prod_ref_real | 200 |
| train_df40_blendface | train_fake | 500 |
| train_df40_e4s | train_fake | 500 |
| train_df40_facedancer | train_fake | 500 |
| train_df40_inswap | train_fake | 500 |
| train_df40_simswap | train_fake | 500 |
| train_df40_uniface | train_fake | 500 |
| train_teams_fake_pool | train_fake | 500 |
| visomaster_enhanced_v2_all | train_fake | 500 |
| train_df40_real_celeb_real | train_real | 500 |
| train_df40_real_faceforensics | train_real | 500 |
| train_df40_real_youtube_real | train_real | 500 |
| train_teams_real_pool | train_real | 500 |

## 4. Headline cross-pool table — by group

For each property: p05 / p50 / p95 across the pool groups (one row per group).


### min_dim

| group | n | p05 | p50 | p95 | mean |
|---|---:|---:|---:|---:|---:|
| TRAIN_REAL | 2000 | 165.9 | 224.0 | 313.1 | 227.8 |
| TRAIN_FAKE | 4000 | 188.0 | 224.0 | 368.0 | 231.1 |
| DEV_REAL | 1581 | 124.0 | 239.0 | 409.0 | 254.7 |
| DEV_FAKE | 1500 | 185.0 | 390.0 | 435.0 | 368.4 |
| LOCKBOX_REAL | 500 | 159.9 | 241.0 | 398.0 | 258.0 |
| LOCKBOX_FAKE | 425 | 179.2 | 291.0 | 325.0 | 276.9 |
| HDTF_REAL | 2000 | 211.0 | 224.0 | 283.0 | 235.8 |
| HDTF_FAKE | 2000 | 212.0 | 224.0 | 286.0 | 236.6 |
| CANARY_CHRONIC_REAL | 300 | 82.0 | 130.0 | 410.2 | 184.0 |
| CANARY_OTHER_REAL | 300 | 144.0 | 223.0 | 384.3 | 234.5 |
| CANARY_FAKE | 200 | 189.0 | 357.5 | 415.0 | 335.9 |
| PROD_REF_REAL | 430 | 175.4 | 230.5 | 512.0 | 287.8 |

### lap_var

| group | n | p05 | p50 | p95 | mean |
|---|---:|---:|---:|---:|---:|
| TRAIN_REAL | 2000 | 7.5 | 33.7 | 492.0 | 104.1 |
| TRAIN_FAKE | 4000 | 14.8 | 53.7 | 190.1 | 74.2 |
| DEV_REAL | 1581 | 29.2 | 156.4 | 967.0 | 297.2 |
| DEV_FAKE | 1500 | 12.8 | 81.5 | 259.5 | 151.5 |
| LOCKBOX_REAL | 500 | 62.3 | 213.5 | 667.9 | 302.0 |
| LOCKBOX_FAKE | 425 | 8.5 | 11.2 | 63.1 | 19.7 |
| HDTF_REAL | 2000 | 54.7 | 155.0 | 548.3 | 211.9 |
| HDTF_FAKE | 2000 | 40.6 | 171.9 | 479.3 | 205.4 |
| CANARY_CHRONIC_REAL | 300 | 48.1 | 491.0 | 1,088 | 485.0 |
| CANARY_OTHER_REAL | 300 | 79.8 | 236.2 | 1,017 | 358.7 |
| CANARY_FAKE | 200 | 9.0 | 20.7 | 255.6 | 84.6 |
| PROD_REF_REAL | 430 | 11.3 | 97.6 | 432.8 | 149.8 |

### luma_mean

| group | n | p05 | p50 | p95 | mean |
|---|---:|---:|---:|---:|---:|
| TRAIN_REAL | 2000 | 88.6 | 131.5 | 182.7 | 134.1 |
| TRAIN_FAKE | 4000 | 105.1 | 148.1 | 192.7 | 148.1 |
| DEV_REAL | 1581 | 119.3 | 162.7 | 200.0 | 161.4 |
| DEV_FAKE | 1500 | 134.1 | 152.1 | 200.0 | 160.2 |
| LOCKBOX_REAL | 500 | 128.6 | 158.7 | 175.9 | 155.6 |
| LOCKBOX_FAKE | 425 | 130.6 | 142.7 | 182.7 | 148.8 |
| HDTF_REAL | 2000 | 100.3 | 139.1 | 178.3 | 139.6 |
| HDTF_FAKE | 2000 | 96.8 | 136.8 | 176.2 | 136.8 |
| PROD_REF_REAL | 430 | 138.7 | 155.4 | 176.1 | 156.8 |

### color_b_dev

| group | n | p05 | p50 | p95 | mean |
|---|---:|---:|---:|---:|---:|
| TRAIN_REAL | 2000 | 6.4 | 13.7 | 24.5 | 14.1 |
| TRAIN_FAKE | 4000 | 6.3 | 15.6 | 27.3 | 16.3 |
| DEV_REAL | 1581 | 5.8 | 14.0 | 29.3 | 15.5 |
| DEV_FAKE | 1500 | 7.2 | 11.9 | 19.6 | 12.9 |
| LOCKBOX_REAL | 500 | 7.3 | 14.3 | 26.0 | 14.1 |
| LOCKBOX_FAKE | 425 | 8.1 | 9.9 | 17.3 | 11.1 |
| HDTF_REAL | 2000 | 5.6 | 12.1 | 21.2 | 12.8 |
| HDTF_FAKE | 2000 | 5.8 | 12.5 | 21.3 | 13.0 |
| PROD_REF_REAL | 430 | 9.8 | 12.3 | 14.7 | 12.3 |

### edge_mag

| group | n | p05 | p50 | p95 | mean |
|---|---:|---:|---:|---:|---:|
| TRAIN_REAL | 2000 | 21.4 | 34.8 | 54.9 | 36.7 |
| TRAIN_FAKE | 4000 | 25.1 | 40.3 | 55.5 | 40.1 |
| DEV_REAL | 1581 | 25.0 | 34.6 | 84.4 | 41.6 |
| DEV_FAKE | 1500 | 22.1 | 32.0 | 36.7 | 31.8 |
| LOCKBOX_REAL | 500 | 35.1 | 44.9 | 53.0 | 44.8 |
| LOCKBOX_FAKE | 425 | 21.1 | 24.8 | 45.3 | 27.4 |
| HDTF_REAL | 2000 | 29.3 | 43.3 | 63.6 | 44.4 |
| HDTF_FAKE | 2000 | 29.1 | 44.8 | 64.0 | 45.6 |
| PROD_REF_REAL | 430 | 21.9 | 39.3 | 56.6 | 37.8 |

### aspect_ratio

| group | n | p05 | p50 | p95 | mean |
|---|---:|---:|---:|---:|---:|
| TRAIN_REAL | 2000 | 1.0 | 1.0 | 1.3 | 1.0 |
| TRAIN_FAKE | 4000 | 1.0 | 1.0 | 1.3 | 1.0 |
| DEV_REAL | 1581 | 1.0 | 1.2 | 1.4 | 1.2 |
| DEV_FAKE | 1500 | 1.0 | 1.1 | 1.3 | 1.1 |
| LOCKBOX_REAL | 500 | 1.0 | 1.0 | 1.3 | 1.1 |
| LOCKBOX_FAKE | 425 | 1.2 | 1.4 | 1.5 | 1.4 |
| HDTF_REAL | 2000 | 1.0 | 1.0 | 1.0 | 1.0 |
| HDTF_FAKE | 2000 | 1.0 | 1.0 | 1.0 | 1.0 |
| CANARY_CHRONIC_REAL | 300 | 1.0 | 1.1 | 1.4 | 1.1 |
| CANARY_OTHER_REAL | 300 | 1.0 | 1.2 | 1.4 | 1.2 |
| CANARY_FAKE | 200 | 1.0 | 1.2 | 1.5 | 1.2 |
| PROD_REF_REAL | 430 | 1.0 | 1.0 | 1.0 | 1.0 |

### bytes

| group | n | p05 | p50 | p95 | mean |
|---|---:|---:|---:|---:|---:|
| TRAIN_REAL | 2000 | 7,149 | 9,165 | 146,704 | 33,330 |
| TRAIN_FAKE | 4000 | 8,066 | 10,092 | 191,017 | 35,791 |
| DEV_REAL | 1581 | 18,360 | 89,272 | 265,028 | 118,824 |
| DEV_FAKE | 1500 | 66,173 | 209,004 | 312,475 | 214,131 |
| LOCKBOX_REAL | 500 | 52,720 | 98,496 | 296,411 | 127,080 |
| LOCKBOX_FAKE | 425 | 62,173 | 133,444 | 158,116 | 125,468 |
| HDTF_REAL | 2000 | 67,370 | 85,056 | 126,471 | 89,589 |
| HDTF_FAKE | 2000 | 67,120 | 85,621 | 128,686 | 90,598 |
| CANARY_CHRONIC_REAL | 300 | 16,041 | 30,345 | 204,212 | 68,242 |
| CANARY_OTHER_REAL | 300 | 15,181 | 78,642 | 204,318 | 91,231 |
| CANARY_FAKE | 200 | 68,662 | 161,232 | 287,583 | 181,873 |

## 5. Per-property findings

For each metric: train-vs-eval-vs-lockbox p50, real-vs-fake p50, and the
fraction of each group's frames that fall below the reference p05.

Reference for "below" cutoff: **PROD_REF_REAL p05 (dor_evening + dor_morning + dor_may5_teams)**.


### min_dim

- TRAIN_REAL p50 = 224.0 | DEV_REAL p50 = 239.0 | LOCKBOX_REAL p50 = 241.0 | HDTF_REAL p50 = 224.0 | PROD_REF_REAL p50 = 230.5
- TRAIN_FAKE p50 = 224.0 | DEV_FAKE p50 = 390.0 | LOCKBOX_FAKE p50 = 291.0 | HDTF_FAKE p50 = 224.0
- Reference PROD_REF_REAL p05 = 175.4. Frac frames below this:
    - TRAIN_REAL: 6.0% (n=2000)
    - TRAIN_FAKE: 1.3% (n=4000)
    - DEV_REAL: 25.4% (n=1581)
    - DEV_FAKE: 4.5% (n=1500)
    - LOCKBOX_REAL: 6.2% (n=500)
    - LOCKBOX_FAKE: 2.4% (n=425)
    - HDTF_REAL: 0.4% (n=2000)
    - HDTF_FAKE: 0.1% (n=2000)
    - CANARY_CHRONIC_REAL: 64.3% (n=300)

### lap_var

- TRAIN_REAL p50 = 33.7 | DEV_REAL p50 = 156.4 | LOCKBOX_REAL p50 = 213.5 | HDTF_REAL p50 = 155.0 | PROD_REF_REAL p50 = 97.6
- TRAIN_FAKE p50 = 53.7 | DEV_FAKE p50 = 81.5 | LOCKBOX_FAKE p50 = 11.2 | HDTF_FAKE p50 = 171.9
- Reference PROD_REF_REAL p05 = 11.3. Frac frames below this:
    - TRAIN_REAL: 12.8% (n=2000)
    - TRAIN_FAKE: 2.7% (n=4000)
    - DEV_REAL: 0.1% (n=1581)
    - DEV_FAKE: 3.4% (n=1500)
    - LOCKBOX_REAL: 0.0% (n=500)
    - LOCKBOX_FAKE: 52.9% (n=425)
    - HDTF_REAL: 0.0% (n=2000)
    - HDTF_FAKE: 0.0% (n=2000)
    - CANARY_CHRONIC_REAL: 0.0% (n=300)

### luma_mean

- TRAIN_REAL p50 = 131.5 | DEV_REAL p50 = 162.7 | LOCKBOX_REAL p50 = 158.7 | HDTF_REAL p50 = 139.1 | PROD_REF_REAL p50 = 155.4
- TRAIN_FAKE p50 = 148.1 | DEV_FAKE p50 = 152.1 | LOCKBOX_FAKE p50 = 142.7 | HDTF_FAKE p50 = 136.8
- Reference PROD_REF_REAL p05 = 138.7. Frac frames below this:
    - TRAIN_REAL: 59.4% (n=2000)
    - TRAIN_FAKE: 37.2% (n=4000)
    - DEV_REAL: 22.8% (n=1581)
    - DEV_FAKE: 8.9% (n=1500)
    - LOCKBOX_REAL: 17.0% (n=500)
    - LOCKBOX_FAKE: 28.7% (n=425)
    - HDTF_REAL: 49.3% (n=2000)
    - HDTF_FAKE: 53.6% (n=2000)

### color_b_dev

- TRAIN_REAL p50 = 13.7 | DEV_REAL p50 = 14.0 | LOCKBOX_REAL p50 = 14.3 | HDTF_REAL p50 = 12.1 | PROD_REF_REAL p50 = 12.3
- TRAIN_FAKE p50 = 15.6 | DEV_FAKE p50 = 11.9 | LOCKBOX_FAKE p50 = 9.9 | HDTF_FAKE p50 = 12.5
- Reference PROD_REF_REAL p05 = 9.8. Frac frames below this:
    - TRAIN_REAL: 18.9% (n=2000)
    - TRAIN_FAKE: 14.1% (n=4000)
    - DEV_REAL: 26.2% (n=1581)
    - DEV_FAKE: 7.9% (n=1500)
    - LOCKBOX_REAL: 23.6% (n=500)
    - LOCKBOX_FAKE: 48.2% (n=425)
    - HDTF_REAL: 27.0% (n=2000)
    - HDTF_FAKE: 26.7% (n=2000)

### edge_mag

- TRAIN_REAL p50 = 34.8 | DEV_REAL p50 = 34.6 | LOCKBOX_REAL p50 = 44.9 | HDTF_REAL p50 = 43.3 | PROD_REF_REAL p50 = 39.3
- TRAIN_FAKE p50 = 40.3 | DEV_FAKE p50 = 32.0 | LOCKBOX_FAKE p50 = 24.8 | HDTF_FAKE p50 = 44.8
- Reference PROD_REF_REAL p05 = 21.9. Frac frames below this:
    - TRAIN_REAL: 5.5% (n=2000)
    - TRAIN_FAKE: 1.7% (n=4000)
    - DEV_REAL: 1.5% (n=1581)
    - DEV_FAKE: 4.9% (n=1500)
    - LOCKBOX_REAL: 0.0% (n=500)
    - LOCKBOX_FAKE: 10.4% (n=425)
    - HDTF_REAL: 0.3% (n=2000)
    - HDTF_FAKE: 0.1% (n=2000)

### aspect_ratio

- TRAIN_REAL p50 = 1.0 | DEV_REAL p50 = 1.2 | LOCKBOX_REAL p50 = 1.0 | HDTF_REAL p50 = 1.0 | PROD_REF_REAL p50 = 1.0
- TRAIN_FAKE p50 = 1.0 | DEV_FAKE p50 = 1.1 | LOCKBOX_FAKE p50 = 1.4 | HDTF_FAKE p50 = 1.0
- Reference PROD_REF_REAL p05 = 1.0. Frac frames below this:
    - TRAIN_REAL: 0.0% (n=2000)
    - TRAIN_FAKE: 0.0% (n=4000)
    - DEV_REAL: 0.0% (n=1581)
    - DEV_FAKE: 0.0% (n=1500)
    - LOCKBOX_REAL: 0.0% (n=500)
    - LOCKBOX_FAKE: 0.0% (n=425)
    - HDTF_REAL: 0.0% (n=2000)
    - HDTF_FAKE: 0.0% (n=2000)
    - CANARY_CHRONIC_REAL: 0.0% (n=300)

### bytes

- TRAIN_REAL p50 = 9,165 | DEV_REAL p50 = 89,272 | LOCKBOX_REAL p50 = 98,496 | HDTF_REAL p50 = 85,056 | PROD_REF_REAL p50 = N/A
- TRAIN_FAKE p50 = 10,092 | DEV_FAKE p50 = 209,004 | LOCKBOX_FAKE p50 = 133,444 | HDTF_FAKE p50 = 85,621
- Reference PROD_REF_REAL p05 = N/A. Frac frames below this:
    - TRAIN_REAL: 0.0% (n=2000)
    - TRAIN_FAKE: 0.0% (n=4000)
    - DEV_REAL: 0.0% (n=1581)
    - DEV_FAKE: 0.0% (n=1500)
    - LOCKBOX_REAL: 0.0% (n=500)
    - LOCKBOX_FAKE: 0.0% (n=425)
    - HDTF_REAL: 0.0% (n=2000)
    - HDTF_FAKE: 0.0% (n=2000)
    - CANARY_CHRONIC_REAL: 0.0% (n=300)

## 6. Identified ultra-bad tails (per-pool detail)

### 6.1 min_dim (resolution)

Reference: PROD_REF_REAL p05 (dor_evening + dor_morning + dor_may5_teams) = 175.4 px.

| pool | role | n | metric_p05 | metric_p50 | frac_below_ref |
|---|---|---:|---:|---:|---:|
| canary_chronic_real | canary_chronic_real | 300 | 82.0 | 130.0 | 64.3% |
| canary_fake | canary_fake | 200 | 189.0 | 357.5 | 0.0% |
| canary_other_real | canary_other_real | 300 | 144.0 | 223.0 | 23.0% |
| deeplive_enhanced_dev | dev_fake | 500 | 408.0 | 413.0 | 0.6% |
| hdtf_fake_clean_dev | hdtf_fake | 500 | 224.0 | 224.0 | 0.0% |
| hdtf_fake_clean_lockbox | hdtf_fake | 500 | 224.0 | 224.0 | 0.0% |
| hdtf_fake_teams_dev | hdtf_fake | 500 | 202.9 | 248.0 | 0.0% |
| hdtf_fake_teams_lockbox | hdtf_fake | 500 | 206.0 | 246.0 | 0.6% |
| hdtf_real_clean_dev | hdtf_real | 500 | 224.0 | 224.0 | 0.0% |
| hdtf_real_clean_lockbox | hdtf_real | 500 | 224.0 | 224.0 | 0.0% |
| hdtf_real_teams_dev | hdtf_real | 500 | 205.0 | 247.0 | 1.0% |
| hdtf_real_teams_lockbox | hdtf_real | 500 | 204.9 | 246.0 | 0.4% |
| prod_ref_dor_evening_local | prod_ref_real | 200 | 200.9 | 220.0 | 0.0% |
| prod_ref_dor_may5_teams | prod_ref_real | 30 | 162.4 | 171.5 | 73.3% |
| prod_ref_dor_morning_local | prod_ref_real | 200 | 280.9 | 373.0 | 0.0% |
| teams_fake_all_dev | dev_fake | 500 | 141.0 | 345.5 | 12.8% |
| teams_fake_all_lockbox | lockbox_fake | 425 | 179.2 | 291.0 | 2.4% |
| teams_real_all_dev | dev_real | 500 | 99.9 | 222.0 | 21.6% |
| teams_real_all_lockbox | lockbox_real | 500 | 159.9 | 241.0 | 6.2% |
| teams_real_dor_dev | dev_real | 81 | 172.0 | 189.0 | 28.4% |
| teams_real_lighting_extreme_dev | dev_real | 500 | 119.9 | 260.0 | 35.0% |
| teams_real_poor_quality_dev | dev_real | 500 | 130.0 | 250.0 | 19.2% |
| train_df40_blendface | train_fake | 500 | 224.0 | 224.0 | 0.0% |
| train_df40_e4s | train_fake | 500 | 224.0 | 224.0 | 0.0% |
| train_df40_facedancer | train_fake | 500 | 224.0 | 224.0 | 0.0% |
| train_df40_inswap | train_fake | 500 | 224.0 | 224.0 | 0.0% |
| train_df40_real_celeb_real | train_real | 500 | 224.0 | 224.0 | 0.0% |
| train_df40_real_faceforensics | train_real | 500 | 224.0 | 224.0 | 0.0% |
| train_df40_real_youtube_real | train_real | 500 | 224.0 | 224.0 | 0.0% |
| train_df40_simswap | train_fake | 500 | 224.0 | 224.0 | 0.0% |
| train_df40_uniface | train_fake | 500 | 224.0 | 224.0 | 0.0% |
| train_teams_fake_pool | train_fake | 500 | 146.0 | 327.5 | 10.0% |
| train_teams_real_pool | train_real | 500 | 97.9 | 232.0 | 24.0% |
| visomaster_enhanced_macro_dev | dev_fake | 500 | 358.9 | 379.0 | 0.0% |
| visomaster_enhanced_v2_all | train_fake | 500 | 180.0 | 194.0 | 0.4% |

### 6.2 lap_var (sharpness proxy)

Reference: PROD_REF_REAL p05 (dor_evening + dor_morning + dor_may5_teams) = 11.3.

| pool | role | n | metric_p05 | metric_p50 | frac_below_ref |
|---|---|---:|---:|---:|---:|
| canary_chronic_real | canary_chronic_real | 300 | 48.1 | 491.0 | 0.0% |
| canary_fake | canary_fake | 200 | 9.0 | 20.7 | 26.0% |
| canary_other_real | canary_other_real | 300 | 79.8 | 236.2 | 0.0% |
| deeplive_enhanced_dev | dev_fake | 500 | 98.2 | 241.8 | 0.0% |
| hdtf_fake_clean_dev | hdtf_fake | 500 | 38.3 | 208.7 | 0.0% |
| hdtf_fake_clean_lockbox | hdtf_fake | 500 | 54.2 | 209.9 | 0.0% |
| hdtf_fake_teams_dev | hdtf_fake | 500 | 42.7 | 147.5 | 0.0% |
| hdtf_fake_teams_lockbox | hdtf_fake | 500 | 39.9 | 140.3 | 0.0% |
| hdtf_real_clean_dev | hdtf_real | 500 | 63.0 | 194.7 | 0.0% |
| hdtf_real_clean_lockbox | hdtf_real | 500 | 64.4 | 189.0 | 0.0% |
| hdtf_real_teams_dev | hdtf_real | 500 | 54.4 | 122.5 | 0.0% |
| hdtf_real_teams_lockbox | hdtf_real | 500 | 43.8 | 125.1 | 0.0% |
| prod_ref_dor_evening_local | prod_ref_real | 200 | 72.5 | 202.8 | 0.0% |
| prod_ref_dor_may5_teams | prod_ref_real | 30 | 205.8 | 237.4 | 0.0% |
| prod_ref_dor_morning_local | prod_ref_real | 200 | 10.9 | 53.2 | 11.0% |
| teams_fake_all_dev | dev_fake | 500 | 9.4 | 31.6 | 9.0% |
| teams_fake_all_lockbox | lockbox_fake | 425 | 8.5 | 11.2 | 52.9% |
| teams_real_all_dev | dev_real | 500 | 36.2 | 179.9 | 0.0% |
| teams_real_all_lockbox | lockbox_real | 500 | 62.3 | 213.5 | 0.0% |
| teams_real_dor_dev | dev_real | 81 | 56.0 | 214.1 | 0.0% |
| teams_real_lighting_extreme_dev | dev_real | 500 | 30.3 | 366.0 | 0.2% |
| teams_real_poor_quality_dev | dev_real | 500 | 27.8 | 119.9 | 0.2% |
| train_df40_blendface | train_fake | 500 | 17.5 | 41.6 | 0.0% |
| train_df40_e4s | train_fake | 500 | 35.0 | 71.6 | 0.0% |
| train_df40_facedancer | train_fake | 500 | 18.8 | 46.3 | 0.8% |
| train_df40_inswap | train_fake | 500 | 11.1 | 31.8 | 6.8% |
| train_df40_real_celeb_real | train_real | 500 | 7.2 | 12.8 | 29.2% |
| train_df40_real_faceforensics | train_real | 500 | 19.4 | 49.9 | 0.2% |
| train_df40_real_youtube_real | train_real | 500 | 6.1 | 21.6 | 21.8% |
| train_df40_simswap | train_fake | 500 | 20.4 | 49.1 | 0.0% |
| train_df40_uniface | train_fake | 500 | 23.3 | 50.3 | 0.0% |
| train_teams_fake_pool | train_fake | 500 | 9.5 | 26.7 | 13.8% |
| train_teams_real_pool | train_real | 500 | 51.7 | 203.3 | 0.0% |
| visomaster_enhanced_macro_dev | dev_fake | 500 | 13.9 | 37.7 | 1.2% |
| visomaster_enhanced_v2_all | train_fake | 500 | 60.0 | 96.7 | 0.0% |

### 6.3 Caveat

We do not have a raw production-frame substrate. The PROD_REF_REAL group is a
proxy built from three local recordings of one identity (Dor): the
`dor_morning_local`, `dor_evening_local` (laptop captures, n=200 each), and
`dor_may5_teams` (Teams capture, n=30) sessions from
`analysis/dor_drift_mechanism_2026-05-06/outputs/per_frame_features.csv`.

This is one identity in three sessions, not a production-traffic sample. If a
real production-frame log becomes available, recompute §6 against it.

## 7. Open observations (factual)

The following are observable from the headline tables. No interpretation; no
ranking; no recommendation.

1. **Resolution gap between training pools and HDTF.** TRAIN_REAL / TRAIN_FAKE
   p50 vs HDTF_REAL / HDTF_FAKE p50 on `min_dim` — see §4 `min_dim`.
2. **Resolution gap between training pools and CANARY_CHRONIC_REAL.** Three of
   the six chronic identities have min_dim p50 ~ 90 px (see
   `CANARY_RESOLUTION_FACTS_2026-05-08.md`); see §6.1 for atlas-wide cross-pool
   distribution.
3. **lap_var inversion between fake and real on the canary substrate.** Per
   `CANARY_RESOLUTION_FACTS_2026-05-08.md`: fake group p50 lap_var = 20.7;
   chronic_real p50 = 491. Atlas confirms direction at scale — see §4.
4. **TRAIN_FAKE vs TRAIN_REAL gap** on `min_dim` and `lap_var` — see §4.
5. **HDTF resolution distribution** — clean vs teams subtypes — see §4 / §6.

(Caller may extend / contest. These are starting points only.)

## 8. Cross-references (existing memories + threads)

The following project memories share evidence with or constrain
interpretations of this atlas. Do not collapse them into recommendations
inside this FACTS doc; they are listed for the user to consult during the
decision step that follows this document.

- `project_image_quality_shortcut.md` — score correlates negatively with
  Laplacian variance / luminance / skin_frac across most suites; eval lockbox
  cam_test_s33 is 14-25× less sharp than training data; deployment-τ
  relaxation 2%→10% real_FPR unlocks 4-77% recall depending on suite.
- `project_face_size_label_leak.md` — each fake method clusters at a tight
  face-size band; reals span wider; model uses face size as a fake predictor.
- `project_dor_drift_named_axes_2026-05-06.md` — endpoint-union ridge
  captures 90% of drift on P8A from named pixel-domain IQ axes; top drivers
  min_dim 53-68%, color_b_dev 27-38%, edge_mag 44%.
- `project_eval_production_crop_tightness_gap.md` — eval frames carry more
  background context around the face than production crops do; structurally
  upstream of the camera-signature shortcut + face-size leak + webcam FPR.
- `project_lockbox_fpr_dominated_by_webcam_mode.md` — lockbox FPR is
  dominated by webcam-style captures (clip_capture_mode==webcam = 65.7% FPR);
  modern_v2 filter cuts headline 4.6% → 0.71%.
- `project_iq_gating_viability_2026-05-04.md` — P8A recall increases
  monotonically with sharpness (Q1 36% → Q4 98%); E2B is INVERTED (Q1 52% →
  Q3 5%). IQ gating is a P8A lever, non-starter for E2B.
- `project_canary_below_production_resolution_2026-05-08.md` — user policy
  thread on resolution-based gating.
- Companion FACTS:
  `analysis/p2_eval_2026-05-08/d1_d4_cpu/CANARY_RESOLUTION_FACTS_2026-05-08.md`.

## 9. Artifacts

- `outputs/per_frame.parquet` — 15,236 rows × all features
- `outputs/per_pool_summary.csv` — per-pool p05/p50/p95/mean for all metrics
- `outputs/cross_pool_compare.csv` — pivoted: rows=pool, columns=metric x p05/p50/p95
- `figs/histogram_<metric>_all_pools.png` — overlaid histograms (7 metrics)
- `figs/pool_<pool>_quad.png` — 4-panel per pool
- `figs/train_vs_eval_vs_lockbox_<metric>.png` — group overlay (3 metrics)
- `build_iq_atlas.py` — driver (re-runnable, idempotent)
- `build_facts_doc.py` — this doc generator
