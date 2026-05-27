# Phase 1 — frame-level distributions — PA_TOP_N_STEP3800

## Per-suite stats

| suite | label | n | mean | p50 | p95 | p99 | muddy[0.2,0.8] | muddy[0.1,0.9] | frac>=0.98 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| teams_real_all_dev | real | 4564 | 0.140 | 0.016 | 0.718 | 0.973 | 0.190 | 0.277 | 0.007 |
| teams_real_all_lockbox | real | 1418 | 0.313 | 0.261 | 0.821 | 0.973 | 0.574 | 0.815 | 0.004 |
| teams_real_dor_dev | real | 50 | 0.665 | 0.663 | 0.960 | 0.991 | 0.640 | 0.800 | 0.040 |
| teams_real_poor_quality_dev | real | 1303 | 0.164 | 0.020 | 0.878 | 0.970 | 0.178 | 0.295 | 0.005 |
| teams_real_lighting_extreme_dev | real | 1742 | 0.255 | 0.099 | 0.927 | 0.985 | 0.308 | 0.439 | 0.015 |
| teams_fake_all_dev | fake | 3039 | 0.788 | 0.935 | 0.994 | 0.995 | 0.234 | 0.376 | 0.274 |
| teams_fake_all_lockbox | fake | 425 | 0.717 | 0.778 | 0.982 | 0.987 | 0.504 | 0.668 | 0.068 |
| deeplive_enhanced_dev | fake | 545 | 0.809 | 0.870 | 0.978 | 0.986 | 0.378 | 0.600 | 0.037 |

## Real-vs-fake separation

| real_suite | fake_suite | n_real | n_fake | KL(r→f) | W1 | bimod_union | AUC(fake+) |
|---|---|---:|---:|---:|---:|---:|---:|
| teams_real_all_dev | teams_fake_all_dev | 4564 | 3039 | 2.434 | 0.648 | 0.827 | 0.941 |
| teams_real_all_lockbox | teams_fake_all_lockbox | 1418 | 425 | 2.984 | 0.405 | 0.633 | 0.877 |
| teams_real_all_dev | deeplive_enhanced_dev | 4564 | 545 | 20.950 | 0.669 | 0.842 | 0.960 |

_Wall time: 0.6s_