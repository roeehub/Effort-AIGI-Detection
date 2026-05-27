# Phase 3 — temporal autocorrelation — P8A_REFERENCE_STEP5000

Streams with >= 8 frames only. lag1 autocorrelation tells us if 32 frames in a window are effectively independent. Strong positive lag1 (e.g., > 0.5) means the effective sample size is <<32 — sliding windows behave very differently from random subsampling.

| suite | label | n_streams≥8 | lag1 p25 | lag1 p50 | lag1 p75 | lag3 p50 | lag5 p50 | slope p50 | max_run≥0.5 p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| teams_fake_all_dev | fake | 12 | 0.011 | 0.104 | 0.583 | 0.008 | 0.036 | 9.98e-06 | 230 |
| teams_fake_all_lockbox | fake | 2 | 0.161 | 0.209 | 0.258 | 0.021 | 0.067 | 0.0002166 | 87 |
| teams_real_all_dev | real | 17 | -0.008 | 0.144 | 0.387 | 0.007 | 0.048 | -1.366e-06 | 31 |
| teams_real_all_lockbox | real | 3 | 0.113 | 0.223 | 0.313 | 0.029 | 0.052 | 0.000547 | 11 |
| teams_real_lighting_extreme_dev | real | 11 | -0.066 | 0.009 | 0.173 | 0.010 | 0.030 | 2.338e-05 | 19 |
| teams_real_poor_quality_dev | real | 11 | 0.014 | 0.274 | 0.389 | 0.080 | 0.018 | 0.0001015 | 6 |

_Wall time: 1.6s_