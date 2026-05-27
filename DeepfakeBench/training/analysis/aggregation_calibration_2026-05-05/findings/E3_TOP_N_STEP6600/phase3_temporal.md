# Phase 3 — temporal autocorrelation — E3_TOP_N_STEP6600

Streams with >= 8 frames only. lag1 autocorrelation tells us if 32 frames in a window are effectively independent. Strong positive lag1 (e.g., > 0.5) means the effective sample size is <<32 — sliding windows behave very differently from random subsampling.

| suite | label | n_streams≥8 | lag1 p25 | lag1 p50 | lag1 p75 | lag3 p50 | lag5 p50 | slope p50 | max_run≥0.5 p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| teams_fake_all_dev | fake | 12 | 0.230 | 0.301 | 0.430 | 0.162 | 0.089 | 2.989e-07 | 293 |
| teams_fake_all_lockbox | fake | 2 | 0.040 | 0.067 | 0.094 | 0.012 | 0.044 | 1.185e-05 | 108 |
| teams_real_all_dev | real | 17 | -0.012 | 0.114 | 0.249 | 0.062 | 0.001 | 2.607e-07 | 20 |
| teams_real_all_lockbox | real | 3 | 0.263 | 0.351 | 0.442 | 0.104 | 0.014 | 0.001654 | 53 |
| teams_real_lighting_extreme_dev | real | 11 | -0.011 | 0.009 | 0.315 | -0.002 | 0.004 | 5.556e-07 | 29 |
| teams_real_poor_quality_dev | real | 11 | -0.056 | 0.232 | 0.333 | 0.062 | -0.015 | 4.238e-06 | 24 |

_Wall time: 2.3s_