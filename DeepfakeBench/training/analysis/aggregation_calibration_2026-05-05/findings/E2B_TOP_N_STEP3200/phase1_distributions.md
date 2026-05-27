# Phase 1 — frame-level distributions — E2B_TOP_N_STEP3200

## Per-suite stats

| suite | label | n | mean | p50 | p95 | p99 | muddy[0.2,0.8] | muddy[0.1,0.9] | frac>=0.98 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| teams_real_all_dev | real | 4564 | 0.120 | 0.007 | 0.741 | 0.947 | 0.148 | 0.215 | 0.003 |
| teams_real_all_lockbox | real | 1418 | 0.104 | 0.020 | 0.556 | 0.856 | 0.153 | 0.224 | 0.001 |
| teams_real_dor_dev | real | 50 | 0.300 | 0.223 | 0.895 | 0.961 | 0.400 | 0.500 | 0.020 |
| teams_real_poor_quality_dev | real | 1303 | 0.140 | 0.007 | 0.808 | 0.959 | 0.163 | 0.243 | 0.004 |
| teams_real_lighting_extreme_dev | real | 1742 | 0.176 | 0.015 | 0.820 | 0.954 | 0.219 | 0.314 | 0.005 |
| teams_fake_all_dev | fake | 3039 | 0.762 | 0.961 | 0.996 | 0.996 | 0.145 | 0.237 | 0.418 |
| teams_fake_all_lockbox | fake | 425 | 0.750 | 0.827 | 0.993 | 0.994 | 0.424 | 0.602 | 0.169 |
| deeplive_enhanced_dev | fake | 545 | 0.824 | 0.893 | 0.973 | 0.982 | 0.306 | 0.512 | 0.015 |

## Real-vs-fake separation

| real_suite | fake_suite | n_real | n_fake | KL(r→f) | W1 | bimod_union | AUC(fake+) |
|---|---|---:|---:|---:|---:|---:|---:|
| teams_real_all_dev | teams_fake_all_dev | 4564 | 3039 | 1.507 | 0.642 | 0.879 | 0.926 |
| teams_real_all_lockbox | teams_fake_all_lockbox | 1418 | 425 | 5.267 | 0.647 | 0.839 | 0.966 |
| teams_real_all_dev | deeplive_enhanced_dev | 4564 | 545 | 21.384 | 0.704 | 0.884 | 0.965 |

_Wall time: 0.6s_