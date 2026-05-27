# Phase 1 — frame-level distributions — E3_TOP_N_STEP6600

## Per-suite stats

| suite | label | n | mean | p50 | p95 | p99 | muddy[0.2,0.8] | muddy[0.1,0.9] | frac>=0.98 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| teams_real_all_dev | real | 4564 | 0.161 | 0.006 | 0.988 | 0.994 | 0.091 | 0.140 | 0.058 |
| teams_real_all_lockbox | real | 1418 | 0.096 | 0.006 | 0.910 | 0.993 | 0.051 | 0.075 | 0.025 |
| teams_real_dor_dev | real | 50 | 0.025 | 0.006 | 0.065 | 0.424 | 0.020 | 0.040 | 0.000 |
| teams_real_poor_quality_dev | real | 1303 | 0.140 | 0.006 | 0.974 | 0.994 | 0.092 | 0.137 | 0.045 |
| teams_real_lighting_extreme_dev | real | 1742 | 0.286 | 0.007 | 0.994 | 0.994 | 0.124 | 0.200 | 0.120 |
| teams_fake_all_dev | fake | 3039 | 0.843 | 0.994 | 0.994 | 0.994 | 0.075 | 0.115 | 0.754 |
| teams_fake_all_lockbox | fake | 425 | 0.968 | 0.994 | 0.994 | 0.994 | 0.045 | 0.068 | 0.887 |
| deeplive_enhanced_dev | fake | 545 | 0.945 | 0.994 | 0.994 | 0.994 | 0.061 | 0.103 | 0.809 |

## Real-vs-fake separation

| real_suite | fake_suite | n_real | n_fake | KL(r→f) | W1 | bimod_union | AUC(fake+) |
|---|---|---:|---:|---:|---:|---:|---:|
| teams_real_all_dev | teams_fake_all_dev | 4564 | 3039 | 1.779 | 0.681 | 0.924 | 0.923 |
| teams_real_all_lockbox | teams_fake_all_lockbox | 1418 | 425 | 23.733 | 0.873 | 0.961 | 0.988 |
| teams_real_all_dev | deeplive_enhanced_dev | 4564 | 545 | 21.123 | 0.784 | 0.936 | 0.955 |

_Wall time: 0.6s_