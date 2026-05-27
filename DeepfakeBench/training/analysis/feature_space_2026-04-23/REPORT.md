# Feature-space distance analysis — detector perspective

Sources: 20. Pairwise distances over 512-dim backbone features.

## Per-source prob_fake (what the detector thinks)

| source | n | label | prob mean | prob med | prob>0.5 rate |
|---|---:|---|---:|---:|---:|
| `deeplive_enh_fake` | 128 | fake | 0.996 | 0.996 | 1.000 |
| `deeplive_enh_real` | 128 | real | 0.021 | 0.007 | 0.008 |
| `deeplive_non_enh_fake` | 128 | fake | 0.870 | 0.995 | 0.867 |
| `deeplive_non_enh_real` | 128 | real | 0.018 | 0.007 | 0.008 |
| `dl_bucket_visomaster_fake` | 128 | fake | 0.929 | 0.995 | 0.953 |
| `dl_bucket_visomaster_real` | 113 | real | 0.069 | 0.008 | 0.044 |
| `external_vcd_real` | 256 | real | 0.138 | 0.014 | 0.129 |
| `external_youtube_avspeech_real` | 277 | real | 0.221 | 0.009 | 0.238 |
| `proper_real_clean__paired` | 300 | real | 0.032 | 0.007 | 0.017 |
| `proper_real_teams__paired` | 300 | real | 0.029 | 0.007 | 0.007 |
| `proper_visomaster_clean_fake` | 300 | fake | 0.977 | 0.995 | 0.987 |
| `proper_visomaster_enhanced_clean_fake` | 300 | fake | 0.988 | 0.995 | 0.997 |
| `proper_visomaster_enhanced_teams_fake` | 300 | fake | 0.949 | 0.993 | 0.960 |
| `proper_visomaster_teams_fake` | 300 | fake | 0.917 | 0.993 | 0.937 |
| `tv2_deeplive_fake` | 156 | fake | 0.989 | 0.995 | 1.000 |
| `tv2_deeplive_real` | 163 | real | 0.079 | 0.009 | 0.074 |
| `tv2_visomaster_fake` | 38 | fake | 0.494 | 0.429 | 0.421 |
| `tv2_visomaster_real` | 134 | real | 0.074 | 0.008 | 0.060 |
| `visomaster_enhanced_v2_fake` | 300 | fake | 0.842 | 0.984 | 0.867 |
| `wma_failure_fake` | 111 | fake | 0.994 | 0.995 | 1.000 |

## Top-15 largest Fréchet gaps (bigger = more distributionally distant in feature space)

| pair | Fréchet |
|---|---:|
| `deeplive_enh_fake` ↔ `deeplive_enh_real` | 994.77 |
| `deeplive_enh_fake` ↔ `deeplive_non_enh_real` | 985.44 |
| `deeplive_enh_fake` ↔ `proper_real_clean__paired` | 960.07 |
| `deeplive_enh_fake` ↔ `proper_real_teams__paired` | 931.53 |
| `deeplive_enh_real` ↔ `wma_failure_fake` | 927.57 |
| `deeplive_enh_real` ↔ `tv2_deeplive_fake` | 920.78 |
| `deeplive_non_enh_real` ↔ `wma_failure_fake` | 918.65 |
| `deeplive_non_enh_real` ↔ `tv2_deeplive_fake` | 911.59 |
| `deeplive_enh_fake` ↔ `dl_bucket_visomaster_real` | 910.34 |
| `deeplive_enh_fake` ↔ `tv2_visomaster_real` | 900.04 |
| `proper_real_clean__paired` ↔ `wma_failure_fake` | 892.93 |
| `deeplive_enh_real` ↔ `proper_visomaster_clean_fake` | 889.85 |
| `proper_real_clean__paired` ↔ `tv2_deeplive_fake` | 886.56 |
| `deeplive_non_enh_real` ↔ `proper_visomaster_clean_fake` | 881.14 |
| `deeplive_enh_real` ↔ `proper_visomaster_enhanced_clean_fake` | 880.70 |

## Top-15 largest MMD² (RBF) gaps

| pair | MMD² |
|---|---:|
| `deeplive_enh_fake` ↔ `deeplive_enh_real` | 1.3959 |
| `deeplive_enh_real` ↔ `tv2_deeplive_fake` | 1.3899 |
| `deeplive_enh_real` ↔ `wma_failure_fake` | 1.3890 |
| `proper_real_clean__paired` ↔ `tv2_deeplive_fake` | 1.3890 |
| `deeplive_enh_real` ↔ `proper_visomaster_enhanced_clean_fake` | 1.3793 |
| `deeplive_enh_real` ↔ `proper_visomaster_clean_fake` | 1.3660 |
| `deeplive_enh_fake` ↔ `tv2_deeplive_real` | 1.3554 |
| `proper_real_clean__paired` ↔ `proper_visomaster_enhanced_clean_fake` | 1.3443 |
| `deeplive_non_enh_real` ↔ `tv2_deeplive_fake` | 1.3410 |
| `proper_real_teams__paired` ↔ `tv2_deeplive_fake` | 1.3339 |
| `proper_real_clean__paired` ↔ `proper_visomaster_clean_fake` | 1.3327 |
| `tv2_deeplive_real` ↔ `wma_failure_fake` | 1.3319 |
| `proper_real_clean__paired` ↔ `wma_failure_fake` | 1.3228 |
| `proper_real_teams__paired` ↔ `wma_failure_fake` | 1.3176 |
| `deeplive_non_enh_real` ↔ `proper_visomaster_clean_fake` | 1.3173 |

## Top-15 largest centroid Euclidean gaps

| pair | ‖μ_a − μ_b‖ |
|---|---:|
| `deeplive_enh_fake` ↔ `deeplive_enh_real` | 31.181 |
| `deeplive_enh_fake` ↔ `deeplive_non_enh_real` | 31.021 |
| `deeplive_enh_fake` ↔ `proper_real_clean__paired` | 30.469 |
| `deeplive_enh_real` ↔ `wma_failure_fake` | 30.107 |
| `deeplive_enh_fake` ↔ `proper_real_teams__paired` | 30.016 |
| `deeplive_non_enh_real` ↔ `wma_failure_fake` | 29.944 |
| `deeplive_enh_real` ↔ `tv2_deeplive_fake` | 29.929 |
| `deeplive_non_enh_real` ↔ `tv2_deeplive_fake` | 29.762 |
| `deeplive_enh_real` ↔ `proper_visomaster_clean_fake` | 29.423 |
| `proper_real_clean__paired` ↔ `wma_failure_fake` | 29.383 |
| `deeplive_enh_fake` ↔ `dl_bucket_visomaster_real` | 29.342 |
| `deeplive_enh_real` ↔ `proper_visomaster_enhanced_clean_fake` | 29.279 |
| `deeplive_non_enh_real` ↔ `proper_visomaster_clean_fake` | 29.257 |
| `proper_real_clean__paired` ↔ `tv2_deeplive_fake` | 29.249 |
| `deeplive_non_enh_real` ↔ `proper_visomaster_enhanced_clean_fake` | 29.118 |

## Training-vs-OOD gate drivers

The OOD gate grades `worst_pool_fpr` against the fake pools listed under `ood_monitoring.external_fake_sources` (currently `wma_failure_fake` + `teams_ood_fake`) and the real pools under `external_real_sources` (incl. `external_vcd_real`, `external_youtube_avspeech_real`, `teams_ood_real`). The relevant comparisons for generalization are therefore between the **training-side** fake/real sources and these gate pools.

### Gate-fake pools vs training-fake sources (Fréchet)

| gate pool | closest training fake | Fréchet | farthest training fake | Fréchet |
|---|---|---:|---|---:|
| `wma_failure_fake` | `deeplive_enh_fake` | 14.55 | `tv2_visomaster_fake` | 339.49 |

## Files
- `distance_matrix_frechet.csv`
- `distance_matrix_mmd_rbf.csv`
- `distance_matrix_centroid.csv`
- `per_source_summary.csv`