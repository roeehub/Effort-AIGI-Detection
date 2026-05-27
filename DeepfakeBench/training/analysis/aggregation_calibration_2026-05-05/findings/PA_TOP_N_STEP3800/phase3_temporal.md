# Phase 3 — temporal autocorrelation — PA_TOP_N_STEP3800

Streams with >= 8 frames only. lag1 autocorrelation tells us if 32 frames in a window are effectively independent. Strong positive lag1 (e.g., > 0.5) means the effective sample size is <<32 — sliding windows behave very differently from random subsampling.

| suite | label | n_streams≥8 | lag1 p25 | lag1 p50 | lag1 p75 | lag3 p50 | lag5 p50 | slope p50 | max_run≥0.5 p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| teams_fake_all_dev | fake | 12 | 0.131 | 0.320 | 0.478 | 0.059 | 0.019 | -7.258e-06 | 293 |
| teams_fake_all_lockbox | fake | 2 | 0.259 | 0.302 | 0.345 | 0.098 | 0.092 | 0.0004393 | 88 |
| teams_real_all_dev | real | 17 | 0.066 | 0.195 | 0.434 | 0.083 | 0.065 | 2.633e-05 | 15 |
| teams_real_all_lockbox | real | 3 | 0.198 | 0.238 | 0.360 | -0.098 | 0.175 | 0.001581 | 4 |
| teams_real_lighting_extreme_dev | real | 11 | 0.064 | 0.139 | 0.195 | 0.094 | 0.106 | 1.215e-05 | 9 |
| teams_real_poor_quality_dev | real | 11 | 0.132 | 0.179 | 0.318 | 0.083 | 0.081 | 0.0001579 | 34 |

_Wall time: 1.6s_