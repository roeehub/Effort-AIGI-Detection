# Phase 1 — frame-level distributions — P8A_REFERENCE_STEP5000

## Per-suite stats

| suite | label | n | mean | p50 | p95 | p99 | muddy[0.2,0.8] | muddy[0.1,0.9] | frac>=0.98 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| teams_real_all_dev | real | 4564 | 0.134 | 0.007 | 0.975 | 0.994 | 0.071 | 0.120 | 0.047 |
| teams_real_all_lockbox | real | 1418 | 0.096 | 0.016 | 0.629 | 0.968 | 0.093 | 0.162 | 0.008 |
| teams_real_dor_dev | real | 50 | 0.331 | 0.181 | 0.955 | 0.983 | 0.260 | 0.440 | 0.040 |
| teams_real_poor_quality_dev | real | 1303 | 0.099 | 0.007 | 0.730 | 0.984 | 0.092 | 0.147 | 0.012 |
| teams_real_lighting_extreme_dev | real | 1742 | 0.126 | 0.008 | 0.963 | 0.994 | 0.070 | 0.120 | 0.040 |
| teams_fake_all_dev | fake | 3039 | 0.745 | 0.985 | 0.995 | 0.995 | 0.165 | 0.258 | 0.519 |
| teams_fake_all_lockbox | fake | 425 | 0.648 | 0.788 | 0.995 | 0.995 | 0.308 | 0.447 | 0.294 |
| deeplive_enhanced_dev | fake | 545 | 0.522 | 0.534 | 0.988 | 0.993 | 0.339 | 0.527 | 0.099 |

## Real-vs-fake separation

| real_suite | fake_suite | n_real | n_fake | KL(r→f) | W1 | bimod_union | AUC(fake+) |
|---|---|---:|---:|---:|---:|---:|---:|
| teams_real_all_dev | teams_fake_all_dev | 4564 | 3039 | 1.445 | 0.610 | 0.908 | 0.911 |
| teams_real_all_lockbox | teams_fake_all_lockbox | 1418 | 425 | 1.929 | 0.552 | 0.891 | 0.919 |
| teams_real_all_dev | deeplive_enhanced_dev | 4564 | 545 | 1.474 | 0.388 | 0.922 | 0.861 |

_Wall time: 0.6s_