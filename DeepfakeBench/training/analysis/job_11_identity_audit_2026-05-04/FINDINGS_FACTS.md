# Job 11 — Per-identity per-ckpt FPR / fake-recall audit (FACTS)

Date: 2026-05-04

Ckpts: P8A_step5000, E2B_TOP_N_step3200, E3_TOP_N_step6600

τ@FPR=10% on teams_real_all_dev: P8A=0.7052, E2B_3200=0.5075, E3_6600=0.8526

Identity extraction: substring-strip on video_id (KEEPS __s\d+ as part of identity, mirroring Job 3). Reused identity column from Job 3 cohort csv for real-side suites.


## 1. Chronic-6 per-ckpt FPR (teams_real_all_dev)

| Identity | n_frames | n_FP P8A | n_FP E2B_3200 | n_FP E3_6600 | FPR P8A | FPR E2B_3200 | FPR E3_6600 | max-min Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bla_bla_chow | 311 | 22 | 152 | 165 | 0.071 | 0.489 | 0.531 | 0.460 |
| bla_bla_chow__s2 | 180 | 32 | 67 | 135 | 0.178 | 0.372 | 0.750 | 0.572 |
| PC_Generator__s22 | 227 | 206 | 26 | 13 | 0.907 | 0.115 | 0.057 | 0.850 |
| PC_Generator__s45 | 91 | 52 | 45 | 16 | 0.571 | 0.495 | 0.176 | 0.396 |
| roy_d | 130 | 50 | 42 | 103 | 0.385 | 0.323 | 0.792 | 0.469 |
| Q__s6 | 54 | 51 | 32 | 9 | 0.944 | 0.593 | 0.167 | 0.778 |

## 2. Largest cross-ckpt FPR spread on teams_real_all_dev (top 10)

| Identity | n_frames | FPR P8A | FPR E2B_3200 | FPR E3_6600 | max-min Δ |
|---|---:|---:|---:|---:|---:|
| PC_Generator__s22 | 227 | 0.907 | 0.115 | 0.057 | 0.850 |
| Q__s6 | 54 | 0.944 | 0.593 | 0.167 | 0.778 |
| bla_bla_chow__s2 | 180 | 0.178 | 0.372 | 0.750 | 0.572 |
| roy_d | 130 | 0.385 | 0.323 | 0.792 | 0.469 |
| bla_bla_chow | 311 | 0.071 | 0.489 | 0.531 | 0.460 |
| PC_Generator__s45 | 91 | 0.571 | 0.495 | 0.176 | 0.396 |
| xiang | 159 | 0.013 | 0.358 | 0.057 | 0.346 |
| orel | 35 | 0.000 | 0.114 | 0.057 | 0.114 |
| PC_Generator__s13 | 212 | 0.000 | 0.080 | 0.000 | 0.080 |
| Test_Cam__s73 | 251 | 0.044 | 0.000 | 0.000 | 0.044 |

## 3. Chronic-6 per-ckpt fake-side recall

Recall at the same dev-calibrated τ (FPR=10% on teams_real_all_dev). Empty rows = identity is not present in fake suite.

### 3a. teams_fake_all_dev
| Identity | n_frames | recall P8A | recall E2B_3200 | recall E3_6600 | max-min Δ |
|---|---:|---:|---:|---:|---:|
| bla_bla_chow | (not in fake_dev) |  |  |  |  |
| bla_bla_chow__s2 | (not in fake_dev) |  |  |  |  |
| PC_Generator__s22 | (not in fake_dev) |  |  |  |  |
| PC_Generator__s45 | (not in fake_dev) |  |  |  |  |
| roy_d | (not in fake_dev) |  |  |  |  |
| Q__s6 | (not in fake_dev) |  |  |  |  |

### 3a-base. teams_fake_all_dev (base-identity, __s\d+ collapsed)

Chronic-6 share these base names: `bla_bla_chow`, `pc_generator`, `roy_d`, `q`. Detailed-identity (with `__s\d+`) lookups in Section 3a returned zero matches because the fake-suite session IDs (`__s3`, `__s4`, `__s9`, `__s15`) differ from the real-suite session IDs (`__s22`, `__s45`, `__s2`). Per memory `project_lockbox_identity_looseness.md` identity leakage is intentional at the BASE-name level.

| Base identity | n_frames | recall P8A | recall E2B_3200 | recall E3_6600 | max-min Δ |
|---|---:|---:|---:|---:|---:|
| bla_bla_chow | (not in fake_dev) |  |  |  |  |
| pc_generator | 194 | 1.000 | 0.979 | 0.861 | 0.139 |
| roy_d | (not in fake_dev) |  |  |  |  |
| q | (not in fake_dev) |  |  |  |  |

### 3b. teams_fake_all_lockbox
| Identity | n_frames | recall P8A | recall E2B_3200 | recall E3_6600 | max-min Δ |
|---|---:|---:|---:|---:|---:|
| bla_bla_chow | (not in fake_lockbox) |  |  |  |  |
| bla_bla_chow__s2 | (not in fake_lockbox) |  |  |  |  |
| PC_Generator__s22 | (not in fake_lockbox) |  |  |  |  |
| PC_Generator__s45 | (not in fake_lockbox) |  |  |  |  |
| roy_d | (not in fake_lockbox) |  |  |  |  |
| Q__s6 | (not in fake_lockbox) |  |  |  |  |

### 3b-base. teams_fake_all_lockbox (base-identity)

| Base identity | n_frames | recall P8A | recall E2B_3200 | recall E3_6600 | max-min Δ |
|---|---:|---:|---:|---:|---:|
| bla_bla_chow | (not in fake_lockbox) |  |  |  |  |
| pc_generator | 91 | 1.000 | 1.000 | 1.000 | 0.000 |
| roy_d | (not in fake_lockbox) |  |  |  |  |
| q | (not in fake_lockbox) |  |  |  |  |

## 4. Capture-mode dominance per chronic / top-10 identity

Source: parquet `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` joined on gcs_uri. Coverage% = fraction of dev frames with parquet meta.

| Identity | n_frames_dev | meta coverage | dominant capture_mode | dominant share |
|---|---:|---:|:---|---:|
| Md_noyn_Sharker__s15 | 682 | 100% | webcam | 0.726 |
| PC_Generator__s13 | 212 | 100% | normal_photo | 0.604 |
| PC_Generator__s14 | 103 | 100% | phone_screen | 1.000 |
| PC_Generator__s22 (chronic-6) | 227 | 100% | webcam | 0.974 |
| PC_Generator__s45 (chronic-6) | 91 | 100% | webcam | 0.670 |
| Q__s6 (chronic-6) | 54 | 100% | webcam | 0.685 |
| Test_Cam__s41 | 815 | 100% | phone_screen | 0.465 |
| Test_Cam__s73 | 251 | 100% | normal_photo | 0.936 |
| bla_bla_chow (chronic-6) | 311 | 100% | screen | 0.598 |
| bla_bla_chow__s2 (chronic-6) | 180 | 100% | phone_screen | 0.956 |
| orel | 35 | 100% | phone_screen | 0.743 |
| roy_d (chronic-6) | 130 | 0% | NO_PARQUET_COVERAGE | N/A |
| xiang | 159 | 0% | NO_PARQUET_COVERAGE | N/A |

## 5. Uniquely-offending identities per ckpt (teams_real_all_dev)

Definition: identity-FPR for this ckpt > 2× median ckpt-FPR for the same identity, and ckpt-FPR > 0. Or ckpt-FPR > 0 while median = 0.

Counts: P8A=6, E2B_3200=6, E3_6600=2

### Per-ckpt list, sorted by FPR desc (top 12 each)

**P8A:**

| Identity | n_frames | FPR (this ckpt) | median FPR | × median | chronic-6? |
|---|---:|---:|---:|---:|:---:|
| PC_Generator__s22 | 227 | 0.907 | 0.115 | 7.92× | Y |
| Test_Cam__s73 | 251 | 0.044 | 0.000 | ∞ |  |
| Xiang_Xiang2_Feng__s23 | 102 | 0.039 | 0.000 | ∞ |  |
| Test_Cam__s41 | 815 | 0.017 | 0.007 | 2.33× |  |
| Md_noyn_Sharker__s15 | 682 | 0.010 | 0.000 | ∞ |  |
| Test_Cam__s76 | 214 | 0.005 | 0.000 | ∞ |  |

**E2B_3200:**

| Identity | n_frames | FPR (this ckpt) | median FPR | × median | chronic-6? |
|---|---:|---:|---:|---:|:---:|
| xiang | 159 | 0.358 | 0.057 | 6.33× |  |
| orel | 35 | 0.114 | 0.057 | 2.00× |  |
| PC_Generator__s13 | 212 | 0.080 | 0.000 | ∞ |  |
| PC_Generator__s34 | 92 | 0.011 | 0.000 | ∞ |  |
| Xiang_Xiang2_Feng | 301 | 0.010 | 0.000 | ∞ |  |
| dor | 269 | 0.004 | 0.000 | ∞ |  |

**E3_6600:**

| Identity | n_frames | FPR (this ckpt) | median FPR | × median | chronic-6? |
|---|---:|---:|---:|---:|:---:|
| roy_d | 130 | 0.792 | 0.385 | 2.06× | Y |
| bla_bla_chow__s2 | 180 | 0.750 | 0.372 | 2.01× | Y |

## 6. Chronic-6 FPR across real suites beyond teams_real_all_dev


### teams_real_all_lockbox
| Identity | n_frames | FPR P8A | FPR E2B_3200 | FPR E3_6600 |
|---|---:|---:|---:|---:|
| bla_bla_chow | (not present) |  |  |  |
| bla_bla_chow__s2 | (not present) |  |  |  |
| PC_Generator__s22 | (not present) |  |  |  |
| PC_Generator__s45 | (not present) |  |  |  |
| roy_d | (not present) |  |  |  |
| Q__s6 | (not present) |  |  |  |

### teams_real_dor_dev
| Identity | n_frames | FPR P8A | FPR E2B_3200 | FPR E3_6600 |
|---|---:|---:|---:|---:|
| bla_bla_chow | (not present) |  |  |  |
| bla_bla_chow__s2 | (not present) |  |  |  |
| PC_Generator__s22 | (not present) |  |  |  |
| PC_Generator__s45 | (not present) |  |  |  |
| roy_d | (not present) |  |  |  |
| Q__s6 | (not present) |  |  |  |

### teams_real_poor_quality_dev
| Identity | n_frames | FPR P8A | FPR E2B_3200 | FPR E3_6600 |
|---|---:|---:|---:|---:|
| bla_bla_chow | 7 | 0.286 | 0.714 | 0.286 |
| bla_bla_chow__s2 | 100 | 0.230 | 0.540 | 0.780 |
| PC_Generator__s22 | (not present) |  |  |  |
| PC_Generator__s45 | 53 | 0.396 | 0.566 | 0.208 |
| roy_d | 3 | 0.667 | 1.000 | 1.000 |
| Q__s6 | (not present) |  |  |  |

### teams_real_lighting_extreme_dev
| Identity | n_frames | FPR P8A | FPR E2B_3200 | FPR E3_6600 |
|---|---:|---:|---:|---:|
| bla_bla_chow | 301 | 0.066 | 0.485 | 0.538 |
| bla_bla_chow__s2 | 104 | 0.087 | 0.173 | 0.769 |
| PC_Generator__s22 | 78 | 0.910 | 0.077 | 0.051 |
| PC_Generator__s45 | 11 | 0.273 | 0.182 | 0.091 |
| roy_d | 113 | 0.442 | 0.363 | 0.876 |
| Q__s6 | (not present) |  |  |  |

## 7. Caveats

- Identity extraction strips standard suffixes; `__s\d+` retained (treats `PC_Generator__s22` and `PC_Generator__s45` as distinct identities, matching Job 3).
- Lockbox parquet coverage on teams_real_all_dev frames is partial; capture_mode in Section 4 reflects only covered frames.
- "Uniquely-offending" threshold is a heuristic (>2× median or median=0 with this ckpt > 0). Sensitivity to alternative definitions not characterized.
- Fake-side recall in Section 3 uses dev-calibrated τ from teams_real_all_dev, not a per-suite τ.
