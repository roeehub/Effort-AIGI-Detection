# Phase 5 — cross-condition robustness — P8A_REFERENCE_STEP5000

**Chosen policy**: `majority_vote` W=16 params={'threshold': 0.7, 'vote_majority': 0.4} override=`none`

## Per-stratum vs global rate

Δ shows stratum_rate − global_rate. For real suites: Δ>+5pp = FPR rises (bad); for fake suites: Δ<-10pp = recall drops (bad).


### teams_real_all_dev (real) — global rate = 0.064 (n=1247)

| stratum_col | stratum_val | n | rate | Δ vs global |
|---|---|---:|---:|---:|
| clip_capture_mode | normal_photo | 50 | 0.000 | -0.064 |
| clip_capture_mode | phone_screen | 138 | 0.043 | -0.021 |
| clip_capture_mode | screen | 185 | 0.086 | +0.022 |
| clip_capture_mode | screen_recording | 40 | 0.000 | -0.064 |
| clip_capture_mode | webcam | 276 | 0.022 | -0.042 |
| clip_lighting | dim | 1 | 0.000 | -0.064 |
| clip_lighting | harsh | 4 | 0.000 | -0.064 |
| clip_lighting | normal | 684 | 0.041 | -0.023 |
| face_area_quartile | Q1 | 98 | 0.071 | +0.007 |
| face_area_quartile | Q2 | 88 | 0.091 | +0.027 |
| face_area_quartile | Q3 | 476 | 0.019 | -0.045 |
| face_area_quartile | Q4 | 13 | 0.077 | +0.013 |
| is_low_quality | False | 214 | 0.000 | -0.064 |
| is_low_quality | True | 475 | 0.059 | -0.005 |
| is_no_face | False | 671 | 0.036 | -0.028 |
| is_no_face | True | 18 | 0.222 | +0.158 |
| sharpness_quartile | Q1 | 44 | 0.114 | +0.049 |
| sharpness_quartile | Q2 | 50 | 0.020 | -0.044 |
| sharpness_quartile | Q3 | 118 | 0.034 | -0.030 |
| sharpness_quartile | Q4 | 477 | 0.038 | -0.026 |

### teams_real_all_lockbox (real) — global rate = 0.018 (n=1250)

| stratum_col | stratum_val | n | rate | Δ vs global |
|---|---|---:|---:|---:|
| clip_capture_mode | normal_photo | 233 | 0.021 | +0.003 |
| clip_capture_mode | phone_screen | 1 | 0.000 | -0.018 |
| clip_capture_mode | webcam | 40 | 0.200 | +0.182 |
| clip_lighting | backlit | 23 | 0.174 | +0.156 |
| clip_lighting | harsh | 1 | 0.000 | -0.018 |
| clip_lighting | normal | 250 | 0.036 | +0.018 |
| face_area_quartile | Q1 | 36 | 0.250 | +0.232 |
| face_area_quartile | Q2 | 233 | 0.017 | -0.001 |
| face_area_quartile | Q3 | 5 | 0.000 | -0.018 |
| is_low_quality | False | 187 | 0.011 | -0.008 |
| is_low_quality | True | 87 | 0.126 | +0.108 |
| is_no_face | False | 274 | 0.047 | +0.029 |
| sharpness_quartile | Q1 | 1 | 0.000 | -0.018 |
| sharpness_quartile | Q2 | 3 | 0.333 | +0.315 |
| sharpness_quartile | Q3 | 84 | 0.000 | -0.018 |
| sharpness_quartile | Q4 | 186 | 0.065 | +0.046 |

### teams_real_dor_dev (real) — global rate = 0.240 (n=50)

| stratum_col | stratum_val | n | rate | Δ vs global |
|---|---|---:|---:|---:|

### teams_real_poor_quality_dev (real) — global rate = 0.029 (n=312)

| stratum_col | stratum_val | n | rate | Δ vs global |
|---|---|---:|---:|---:|
| clip_capture_mode | normal_photo | 19 | 0.053 | +0.024 |
| clip_capture_mode | phone_screen | 37 | 0.108 | +0.079 |
| clip_capture_mode | screen_recording | 33 | 0.000 | -0.029 |
| clip_capture_mode | webcam | 60 | 0.017 | -0.012 |
| clip_lighting | normal | 149 | 0.040 | +0.011 |
| face_area_quartile | Q1 | 10 | 0.200 | +0.171 |
| face_area_quartile | Q2 | 3 | 0.000 | -0.029 |
| face_area_quartile | Q3 | 124 | 0.008 | -0.021 |
| face_area_quartile | Q4 | 7 | 0.143 | +0.114 |
| is_low_quality | False | 107 | 0.000 | -0.029 |
| is_low_quality | True | 42 | 0.143 | +0.114 |
| is_no_face | False | 142 | 0.028 | -0.001 |
| is_no_face | True | 7 | 0.286 | +0.257 |
| sharpness_quartile | Q1 | 13 | 0.154 | +0.125 |
| sharpness_quartile | Q2 | 17 | 0.118 | +0.089 |
| sharpness_quartile | Q3 | 23 | 0.043 | +0.015 |
| sharpness_quartile | Q4 | 96 | 0.010 | -0.018 |

### teams_real_lighting_extreme_dev (real) — global rate = 0.095 (n=775)

| stratum_col | stratum_val | n | rate | Δ vs global |
|---|---|---:|---:|---:|
| clip_capture_mode | normal_photo | 35 | 0.029 | -0.067 |
| clip_capture_mode | phone_screen | 94 | 0.021 | -0.074 |
| clip_capture_mode | screen | 185 | 0.086 | -0.009 |
| clip_capture_mode | screen_recording | 39 | 0.000 | -0.095 |
| clip_capture_mode | webcam | 227 | 0.018 | -0.078 |
| clip_lighting | normal | 580 | 0.040 | -0.056 |
| face_area_quartile | Q1 | 88 | 0.045 | -0.050 |
| face_area_quartile | Q2 | 37 | 0.243 | +0.148 |
| face_area_quartile | Q3 | 435 | 0.018 | -0.077 |
| face_area_quartile | Q4 | 9 | 0.000 | -0.095 |
| is_low_quality | False | 144 | 0.000 | -0.095 |
| is_low_quality | True | 436 | 0.053 | -0.043 |
| is_no_face | False | 569 | 0.037 | -0.059 |
| is_no_face | True | 11 | 0.182 | +0.086 |
| sharpness_quartile | Q1 | 39 | 0.077 | -0.019 |
| sharpness_quartile | Q2 | 41 | 0.024 | -0.071 |
| sharpness_quartile | Q3 | 106 | 0.028 | -0.067 |
| sharpness_quartile | Q4 | 394 | 0.041 | -0.055 |

### teams_fake_all_dev (fake) — global rate = 0.347 (n=1242)

| stratum_col | stratum_val | n | rate | Δ vs global |
|---|---|---:|---:|---:|
| clip_capture_mode | normal_photo | 235 | 0.349 | +0.002 |
| clip_capture_mode | phone_screen | 35 | 0.257 | -0.090 |
| clip_capture_mode | screen_recording | 93 | 0.344 | -0.003 |
| clip_capture_mode | webcam | 329 | 0.486 | +0.139 |
| clip_lighting | backlit | 1 | 0.000 | -0.347 |
| clip_lighting | normal | 691 | 0.410 | +0.063 |
| face_area_quartile | Q1 | 37 | 0.135 | -0.212 |
| face_area_quartile | Q2 | 546 | 0.429 | +0.082 |
| face_area_quartile | Q3 | 77 | 0.403 | +0.056 |
| face_area_quartile | Q4 | 23 | 0.565 | +0.218 |
| is_low_quality | False | 444 | 0.358 | +0.011 |
| is_low_quality | True | 248 | 0.500 | +0.153 |
| is_no_face | False | 683 | 0.414 | +0.067 |
| is_no_face | True | 9 | 0.000 | -0.347 |
| sharpness_quartile | Q1 | 21 | 0.714 | +0.367 |
| sharpness_quartile | Q2 | 47 | 0.255 | -0.092 |
| sharpness_quartile | Q3 | 546 | 0.425 | +0.078 |
| sharpness_quartile | Q4 | 78 | 0.308 | -0.039 |

### teams_fake_all_lockbox (fake) — global rate = 1.000 (n=2)

| stratum_col | stratum_val | n | rate | Δ vs global |
|---|---|---:|---:|---:|
| clip_capture_mode | phone_screen | 2 | 1.000 | +0.000 |
| clip_lighting | normal | 2 | 1.000 | +0.000 |
| face_area_quartile | Q2 | 1 | 1.000 | +0.000 |
| face_area_quartile | Q4 | 1 | 1.000 | +0.000 |
| is_low_quality | True | 2 | 1.000 | +0.000 |
| is_no_face | False | 2 | 1.000 | +0.000 |
| sharpness_quartile | Q1 | 1 | 1.000 | +0.000 |
| sharpness_quartile | Q2 | 1 | 1.000 | +0.000 |

### deeplive_enhanced_dev (fake) — global rate = 0.424 (n=545)

| stratum_col | stratum_val | n | rate | Δ vs global |
|---|---|---:|---:|---:|
| clip_capture_mode | normal_photo | 229 | 0.332 | -0.092 |
| clip_capture_mode | phone_screen | 3 | 0.000 | -0.424 |
| clip_capture_mode | screen_recording | 8 | 0.125 | -0.299 |
| clip_capture_mode | webcam | 305 | 0.505 | +0.081 |
| clip_lighting | backlit | 1 | 0.000 | -0.424 |
| clip_lighting | normal | 544 | 0.425 | +0.001 |
| face_area_quartile | Q2 | 536 | 0.431 | +0.007 |
| is_low_quality | False | 377 | 0.347 | -0.076 |
| is_low_quality | True | 168 | 0.595 | +0.171 |
| is_no_face | False | 536 | 0.431 | +0.007 |
| is_no_face | True | 9 | 0.000 | -0.424 |
| sharpness_quartile | Q2 | 29 | 0.276 | -0.148 |
| sharpness_quartile | Q3 | 516 | 0.432 | +0.008 |

## Robustness flags

| suite | stratum_col | stratum_val | issue | Δ |
|---|---|---|---|---:|
| teams_real_all_dev | is_no_face | True | FPR rises >5pp | +0.158 |
| teams_real_all_lockbox | clip_capture_mode | webcam | FPR rises >5pp | +0.182 |
| teams_real_all_lockbox | clip_lighting | backlit | FPR rises >5pp | +0.156 |
| teams_real_all_lockbox | face_area_quartile | Q1 | FPR rises >5pp | +0.232 |
| teams_real_all_lockbox | is_low_quality | True | FPR rises >5pp | +0.108 |
| teams_real_all_lockbox | sharpness_quartile | Q2 | FPR rises >5pp | +0.315 |
| teams_real_poor_quality_dev | clip_capture_mode | phone_screen | FPR rises >5pp | +0.079 |
| teams_real_poor_quality_dev | face_area_quartile | Q1 | FPR rises >5pp | +0.171 |
| teams_real_poor_quality_dev | face_area_quartile | Q4 | FPR rises >5pp | +0.114 |
| teams_real_poor_quality_dev | is_low_quality | True | FPR rises >5pp | +0.114 |
| teams_real_poor_quality_dev | is_no_face | True | FPR rises >5pp | +0.257 |
| teams_real_poor_quality_dev | sharpness_quartile | Q1 | FPR rises >5pp | +0.125 |
| teams_real_poor_quality_dev | sharpness_quartile | Q2 | FPR rises >5pp | +0.089 |
| teams_real_lighting_extreme_dev | face_area_quartile | Q2 | FPR rises >5pp | +0.148 |
| teams_real_lighting_extreme_dev | is_no_face | True | FPR rises >5pp | +0.086 |
| teams_fake_all_dev | clip_lighting | backlit | recall drops >10pp | -0.347 |
| teams_fake_all_dev | face_area_quartile | Q1 | recall drops >10pp | -0.212 |
| teams_fake_all_dev | is_no_face | True | recall drops >10pp | -0.347 |
| deeplive_enhanced_dev | clip_capture_mode | phone_screen | recall drops >10pp | -0.424 |
| deeplive_enhanced_dev | clip_capture_mode | screen_recording | recall drops >10pp | -0.299 |
| deeplive_enhanced_dev | clip_lighting | backlit | recall drops >10pp | -0.424 |
| deeplive_enhanced_dev | is_no_face | True | recall drops >10pp | -0.424 |
| deeplive_enhanced_dev | sharpness_quartile | Q2 | recall drops >10pp | -0.148 |

## Parquet join coverage

| suite | strat_col | coverage |
|---|---|---:|
| teams_real_all_dev | clip_capture_mode | 0.878 |
| teams_real_all_dev | face_area_quartile | 0.840 |
| teams_real_all_dev | sharpness_quartile | 0.878 |
| teams_real_all_dev | clip_lighting | 0.878 |
| teams_real_all_dev | is_low_quality | 0.878 |
| teams_real_all_dev | is_no_face | 0.878 |
| teams_real_all_lockbox | clip_capture_mode | 0.292 |
| teams_real_all_lockbox | face_area_quartile | 0.292 |
| teams_real_all_lockbox | sharpness_quartile | 0.292 |
| teams_real_all_lockbox | clip_lighting | 0.292 |
| teams_real_all_lockbox | is_low_quality | 0.292 |
| teams_real_all_lockbox | is_no_face | 0.292 |
| teams_real_dor_dev | clip_capture_mode | 0.000 |
| teams_real_dor_dev | face_area_quartile | 0.000 |
| teams_real_dor_dev | sharpness_quartile | 0.000 |
| teams_real_dor_dev | clip_lighting | 0.000 |
| teams_real_dor_dev | is_low_quality | 0.000 |
| teams_real_dor_dev | is_no_face | 0.000 |
| teams_real_poor_quality_dev | clip_capture_mode | 0.875 |
| teams_real_poor_quality_dev | face_area_quartile | 0.814 |
| teams_real_poor_quality_dev | sharpness_quartile | 0.875 |
| teams_real_poor_quality_dev | clip_lighting | 0.875 |
| teams_real_poor_quality_dev | is_low_quality | 0.875 |
| teams_real_poor_quality_dev | is_no_face | 0.875 |
| teams_real_lighting_extreme_dev | clip_capture_mode | 0.888 |
| teams_real_lighting_extreme_dev | face_area_quartile | 0.871 |
| teams_real_lighting_extreme_dev | sharpness_quartile | 0.888 |
| teams_real_lighting_extreme_dev | clip_lighting | 0.888 |
| teams_real_lighting_extreme_dev | is_low_quality | 0.888 |
| teams_real_lighting_extreme_dev | is_no_face | 0.888 |
| teams_fake_all_dev | clip_capture_mode | 0.819 |
| teams_fake_all_dev | face_area_quartile | 0.804 |
| teams_fake_all_dev | sharpness_quartile | 0.819 |
| teams_fake_all_dev | clip_lighting | 0.819 |
| teams_fake_all_dev | is_low_quality | 0.819 |
| teams_fake_all_dev | is_no_face | 0.819 |
| teams_fake_all_lockbox | clip_capture_mode | 1.000 |
| teams_fake_all_lockbox | face_area_quartile | 1.000 |
| teams_fake_all_lockbox | sharpness_quartile | 1.000 |
| teams_fake_all_lockbox | clip_lighting | 1.000 |
| teams_fake_all_lockbox | is_low_quality | 1.000 |
| teams_fake_all_lockbox | is_no_face | 1.000 |
| deeplive_enhanced_dev | clip_capture_mode | 1.000 |
| deeplive_enhanced_dev | face_area_quartile | 0.983 |
| deeplive_enhanced_dev | sharpness_quartile | 1.000 |
| deeplive_enhanced_dev | clip_lighting | 1.000 |
| deeplive_enhanced_dev | is_low_quality | 1.000 |
| deeplive_enhanced_dev | is_no_face | 1.000 |

_Wall time: 4.2s_