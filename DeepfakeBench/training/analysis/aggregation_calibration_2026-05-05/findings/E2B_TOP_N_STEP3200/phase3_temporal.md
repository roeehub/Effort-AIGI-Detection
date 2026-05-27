# Phase 3 — temporal autocorrelation — E2B_TOP_N_STEP3200

Streams with >= 8 frames only. lag1 autocorrelation tells us if 32 frames in a window are effectively independent. Strong positive lag1 (e.g., > 0.5) means the effective sample size is <<32 — sliding windows behave very differently from random subsampling.

| suite | label | n_streams≥8 | lag1 p25 | lag1 p50 | lag1 p75 | lag3 p50 | lag5 p50 | slope p50 | max_run≥0.5 p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| teams_fake_all_dev | fake | 12 | 0.164 | 0.264 | 0.490 | 0.078 | 0.019 | -1.052e-05 | 230 |
| teams_fake_all_lockbox | fake | 2 | 0.257 | 0.303 | 0.348 | 0.103 | 0.239 | 0.000406 | 88 |
| teams_real_all_dev | real | 17 | 0.020 | 0.158 | 0.299 | 0.052 | 0.039 | 3.854e-05 | 11 |
| teams_real_all_lockbox | real | 3 | 0.170 | 0.218 | 0.219 | -0.070 | 0.208 | -0.001227 | 4 |
| teams_real_lighting_extreme_dev | real | 11 | -0.031 | 0.125 | 0.301 | 0.069 | 0.096 | 0.0005492 | 6 |
| teams_real_poor_quality_dev | real | 11 | 0.009 | 0.262 | 0.408 | 0.079 | 0.078 | 0.0002621 | 13 |

_Wall time: 1.7s_