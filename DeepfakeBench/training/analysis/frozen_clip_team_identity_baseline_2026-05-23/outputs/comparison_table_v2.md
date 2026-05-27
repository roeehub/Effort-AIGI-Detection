# Joint-calibrated comparison @ team-real-FPR = 5%

Each head τ set so that fraction(team-real >= τ) = 0.05 on the 1,821 deploy-relevant real frames.
Fake recall reported per-human on fake-attack cohorts. Higher = better.

| Head | Mean real prob | τ | dor recall | Xinhe recall | Xiang recall | min recall | mean recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| P8A | 0.047 | 0.2388 | 0.870 | 0.786 | 0.936 | 0.786 | 0.864 |
| E2B | 0.067 | 0.4551 | 0.700 | 0.813 | 0.945 | 0.700 | 0.819 |
| SlotAv2_FACE | 0.445 | 0.6529 | 0.562 | 0.905 | 0.988 | 0.562 | 0.818 |
| SlotAv2_CLS | 0.140 | 0.4608 | 0.622 | 0.546 | 0.964 | 0.546 | 0.710 |
| T5C | 0.210 | 0.6571 | 0.759 | 0.379 | 0.948 | 0.379 | 0.695 |
| MLP_DEV_LB | 0.205 | 0.9134 | 0.527 | 0.205 | 0.465 | 0.205 | 0.399 |
| MLP_DEV | 0.274 | 0.9748 | 0.292 | 0.160 | 0.159 | 0.159 | 0.204 |
| LR_DEV_LB | 0.203 | 0.9750 | 0.422 | 0.153 | 0.381 | 0.153 | 0.318 |
| LR_DEV | 0.255 | 0.9931 | 0.289 | 0.055 | 0.197 | 0.055 | 0.180 |
| MLP_OPTB_DEV | 0.212 | 0.9995 | 0.171 | 0.037 | 0.303 | 0.037 | 0.170 |
| LR_OPTB_DEV | 0.332 | 0.9951 | 0.160 | 0.018 | 0.268 | 0.018 | 0.149 |
| MLP_OPTB | 0.296 | 0.9856 | 0.092 | 0.014 | 0.005 | 0.005 | 0.037 |
| LR_OPTB | 0.382 | 0.9926 | 0.071 | 0.002 | 0.000 | 0.000 | 0.024 |
