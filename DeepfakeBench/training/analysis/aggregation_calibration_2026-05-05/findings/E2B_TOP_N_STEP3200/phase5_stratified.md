# Phase 5 — cross-condition robustness — E2B_TOP_N_STEP3200

**Chosen policy**: `run_length` W=16 params={'M': 5, 'threshold': 0.9} override=`three_consec_frames_above_0.95`

## Per-stratum vs global rate

Δ shows stratum_rate − global_rate. For real suites: Δ>+5pp = FPR rises (bad); for fake suites: Δ<-10pp = recall drops (bad).


### teams_real_all_dev (real) — global rate = 0.037 (n=1247)

| stratum_col | stratum_val | n | rate | Δ vs global |
|---|---|---:|---:|---:|
| clip_capture_mode | normal_photo | 50 | 0.000 | -0.037 |
| clip_capture_mode | phone_screen | 138 | 0.022 | -0.015 |
| clip_capture_mode | screen | 185 | 0.119 | +0.082 |
| clip_capture_mode | screen_recording | 40 | 0.000 | -0.037 |
| clip_capture_mode | webcam | 276 | 0.018 | -0.019 |
| clip_lighting | dim | 1 | 0.000 | -0.037 |
| clip_lighting | harsh | 4 | 0.000 | -0.037 |
| clip_lighting | normal | 684 | 0.044 | +0.007 |
| face_area_quartile | Q1 | 98 | 0.041 | +0.004 |
| face_area_quartile | Q2 | 88 | 0.102 | +0.065 |
| face_area_quartile | Q3 | 476 | 0.036 | -0.001 |
| face_area_quartile | Q4 | 13 | 0.000 | -0.037 |
| is_low_quality | False | 214 | 0.000 | -0.037 |
| is_low_quality | True | 475 | 0.063 | +0.026 |
| is_no_face | False | 671 | 0.045 | +0.008 |
| is_no_face | True | 18 | 0.000 | -0.037 |
| sharpness_quartile | Q1 | 44 | 0.023 | -0.014 |
| sharpness_quartile | Q2 | 50 | 0.020 | -0.017 |
| sharpness_quartile | Q3 | 118 | 0.025 | -0.011 |
| sharpness_quartile | Q4 | 477 | 0.052 | +0.016 |

### teams_real_all_lockbox (real) — global rate = 0.008 (n=1250)

| stratum_col | stratum_val | n | rate | Δ vs global |
|---|---|---:|---:|---:|
| clip_capture_mode | normal_photo | 233 | 0.004 | -0.004 |
| clip_capture_mode | phone_screen | 1 | 0.000 | -0.008 |
| clip_capture_mode | webcam | 40 | 0.225 | +0.217 |
| clip_lighting | backlit | 23 | 0.217 | +0.209 |
| clip_lighting | harsh | 1 | 0.000 | -0.008 |
| clip_lighting | normal | 250 | 0.020 | +0.012 |
| face_area_quartile | Q1 | 36 | 0.250 | +0.242 |
| face_area_quartile | Q2 | 233 | 0.004 | -0.004 |
| face_area_quartile | Q3 | 5 | 0.000 | -0.008 |
| is_low_quality | False | 187 | 0.000 | -0.008 |
| is_low_quality | True | 87 | 0.115 | +0.107 |
| is_no_face | False | 274 | 0.036 | +0.028 |
| sharpness_quartile | Q1 | 1 | 0.000 | -0.008 |
| sharpness_quartile | Q2 | 3 | 0.000 | -0.008 |
| sharpness_quartile | Q3 | 84 | 0.000 | -0.008 |
| sharpness_quartile | Q4 | 186 | 0.054 | +0.046 |

### teams_real_dor_dev (real) — global rate = 0.060 (n=50)

| stratum_col | stratum_val | n | rate | Δ vs global |
|---|---|---:|---:|---:|

### teams_real_poor_quality_dev (real) — global rate = 0.022 (n=312)

| stratum_col | stratum_val | n | rate | Δ vs global |
|---|---|---:|---:|---:|
| clip_capture_mode | normal_photo | 19 | 0.000 | -0.022 |
| clip_capture_mode | phone_screen | 37 | 0.000 | -0.022 |
| clip_capture_mode | screen_recording | 33 | 0.000 | -0.022 |
| clip_capture_mode | webcam | 60 | 0.000 | -0.022 |
| clip_lighting | normal | 149 | 0.000 | -0.022 |
| face_area_quartile | Q1 | 10 | 0.000 | -0.022 |
| face_area_quartile | Q2 | 3 | 0.000 | -0.022 |
| face_area_quartile | Q3 | 124 | 0.000 | -0.022 |
| face_area_quartile | Q4 | 7 | 0.000 | -0.022 |
| is_low_quality | False | 107 | 0.000 | -0.022 |
| is_low_quality | True | 42 | 0.000 | -0.022 |
| is_no_face | False | 142 | 0.000 | -0.022 |
| is_no_face | True | 7 | 0.000 | -0.022 |
| sharpness_quartile | Q1 | 13 | 0.000 | -0.022 |
| sharpness_quartile | Q2 | 17 | 0.000 | -0.022 |
| sharpness_quartile | Q3 | 23 | 0.000 | -0.022 |
| sharpness_quartile | Q4 | 96 | 0.000 | -0.022 |

### teams_real_lighting_extreme_dev (real) — global rate = 0.057 (n=775)

| stratum_col | stratum_val | n | rate | Δ vs global |
|---|---|---:|---:|---:|
| clip_capture_mode | normal_photo | 35 | 0.000 | -0.057 |
| clip_capture_mode | phone_screen | 94 | 0.021 | -0.035 |
| clip_capture_mode | screen | 185 | 0.119 | +0.062 |
| clip_capture_mode | screen_recording | 39 | 0.000 | -0.057 |
| clip_capture_mode | webcam | 227 | 0.022 | -0.035 |
| clip_lighting | normal | 580 | 0.050 | -0.007 |
| face_area_quartile | Q1 | 88 | 0.034 | -0.023 |
| face_area_quartile | Q2 | 37 | 0.243 | +0.186 |
| face_area_quartile | Q3 | 435 | 0.039 | -0.018 |
| face_area_quartile | Q4 | 9 | 0.000 | -0.057 |
| is_low_quality | False | 144 | 0.000 | -0.057 |
| is_low_quality | True | 436 | 0.067 | +0.010 |
| is_no_face | False | 569 | 0.051 | -0.006 |
| is_no_face | True | 11 | 0.000 | -0.057 |
| sharpness_quartile | Q1 | 39 | 0.026 | -0.031 |
| sharpness_quartile | Q2 | 41 | 0.000 | -0.057 |
| sharpness_quartile | Q3 | 106 | 0.028 | -0.028 |
| sharpness_quartile | Q4 | 394 | 0.063 | +0.007 |

### teams_fake_all_dev (fake) — global rate = 0.262 (n=1242)

| stratum_col | stratum_val | n | rate | Δ vs global |
|---|---|---:|---:|---:|
| clip_capture_mode | normal_photo | 235 | 0.532 | +0.270 |
| clip_capture_mode | phone_screen | 35 | 0.171 | -0.090 |
| clip_capture_mode | screen_recording | 93 | 0.172 | -0.090 |
| clip_capture_mode | webcam | 329 | 0.465 | +0.203 |
| clip_lighting | backlit | 1 | 0.000 | -0.262 |
| clip_lighting | normal | 691 | 0.434 | +0.172 |
| face_area_quartile | Q1 | 37 | 0.189 | -0.072 |
| face_area_quartile | Q2 | 546 | 0.487 | +0.226 |
| face_area_quartile | Q3 | 77 | 0.247 | -0.015 |
| face_area_quartile | Q4 | 23 | 0.348 | +0.086 |
| is_low_quality | False | 444 | 0.455 | +0.193 |
| is_low_quality | True | 248 | 0.395 | +0.133 |
| is_no_face | False | 683 | 0.439 | +0.178 |
| is_no_face | True | 9 | 0.000 | -0.262 |
| sharpness_quartile | Q1 | 21 | 0.524 | +0.262 |
| sharpness_quartile | Q2 | 47 | 0.085 | -0.177 |
| sharpness_quartile | Q3 | 546 | 0.489 | +0.227 |
| sharpness_quartile | Q4 | 78 | 0.231 | -0.031 |

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

### deeplive_enhanced_dev (fake) — global rate = 0.488 (n=545)

| stratum_col | stratum_val | n | rate | Δ vs global |
|---|---|---:|---:|---:|
| clip_capture_mode | normal_photo | 229 | 0.524 | +0.036 |
| clip_capture_mode | phone_screen | 3 | 0.000 | -0.488 |
| clip_capture_mode | screen_recording | 8 | 0.000 | -0.488 |
| clip_capture_mode | webcam | 305 | 0.479 | -0.009 |
| clip_lighting | backlit | 1 | 0.000 | -0.488 |
| clip_lighting | normal | 544 | 0.489 | +0.001 |
| face_area_quartile | Q2 | 536 | 0.496 | +0.008 |
| is_low_quality | False | 377 | 0.499 | +0.011 |
| is_low_quality | True | 168 | 0.464 | -0.024 |
| is_no_face | False | 536 | 0.496 | +0.008 |
| is_no_face | True | 9 | 0.000 | -0.488 |
| sharpness_quartile | Q2 | 29 | 0.069 | -0.419 |
| sharpness_quartile | Q3 | 516 | 0.512 | +0.024 |

## Robustness flags

| suite | stratum_col | stratum_val | issue | Δ |
|---|---|---|---|---:|
| teams_real_all_dev | clip_capture_mode | screen | FPR rises >5pp | +0.082 |
| teams_real_all_dev | face_area_quartile | Q2 | FPR rises >5pp | +0.065 |
| teams_real_all_lockbox | clip_capture_mode | webcam | FPR rises >5pp | +0.217 |
| teams_real_all_lockbox | clip_lighting | backlit | FPR rises >5pp | +0.209 |
| teams_real_all_lockbox | face_area_quartile | Q1 | FPR rises >5pp | +0.242 |
| teams_real_all_lockbox | is_low_quality | True | FPR rises >5pp | +0.107 |
| teams_real_lighting_extreme_dev | clip_capture_mode | screen | FPR rises >5pp | +0.062 |
| teams_real_lighting_extreme_dev | face_area_quartile | Q2 | FPR rises >5pp | +0.186 |
| teams_fake_all_dev | clip_lighting | backlit | recall drops >10pp | -0.262 |
| teams_fake_all_dev | is_no_face | True | recall drops >10pp | -0.262 |
| teams_fake_all_dev | sharpness_quartile | Q2 | recall drops >10pp | -0.177 |
| deeplive_enhanced_dev | clip_capture_mode | phone_screen | recall drops >10pp | -0.488 |
| deeplive_enhanced_dev | clip_capture_mode | screen_recording | recall drops >10pp | -0.488 |
| deeplive_enhanced_dev | clip_lighting | backlit | recall drops >10pp | -0.488 |
| deeplive_enhanced_dev | is_no_face | True | recall drops >10pp | -0.488 |
| deeplive_enhanced_dev | sharpness_quartile | Q2 | recall drops >10pp | -0.419 |

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

_Wall time: 4.8s_