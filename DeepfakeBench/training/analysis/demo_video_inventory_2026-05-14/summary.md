# Demo Video Inventory — Per-Method Counts (2026-05-14)

Inventory of full-frame fake videos labeled by method.

Sources:
- `visomaster_videos.csv` — from proper_visomaster_target_domain_manifest_2026-04-19_provisional.json. Has the 70-method (swap x restorer) enumeration. Clean transport has mp4 URIs; teams transport has only cropped frames (mp4 URI empty).
- `deeplive_videos.csv` — from gsutil ls on the deeplive videos bucket. Strategies: edge_cases + minimal_processing -> regular, quality_enhancement -> enhanced.
- `visomaster_bucket_videos.csv` — bonus pool of CLEAN visomaster videos in the deeplive bucket under `visomaster_{generator}_*` prefixes. No restorer info available, covers only the 9 base swap models.

Threshold for showcase target: **>= 20 videos per method**.

## Deeplive (live-deepfake-methods-real-and-fake-videos bucket)

| method | videos | clears >=20 |
|---|---|---|
| deeplive_enhanced | 320 | YES |
| deeplive_regular | 859 | YES |

## Visomaster CLEAN (full-frame mp4 available)

| method | videos | clears >=20 |
|---|---|---|
| proper_visomaster_clean__cscs | 12 | no |
| proper_visomaster_clean__ghostface_v1 | 41 | YES |
| proper_visomaster_clean__ghostface_v2 | 41 | YES |
| proper_visomaster_clean__ghostface_v3 | 38 | YES |
| proper_visomaster_clean__instyleswapper256_a | 41 | YES |
| proper_visomaster_clean__instyleswapper256_b | 41 | YES |
| proper_visomaster_clean__instyleswapper256_c | 43 | YES |
| proper_visomaster_clean__inswapper128 | 42 | YES |
| proper_visomaster_clean__simswap512 | 43 | YES |
| proper_visomaster_enhanced_clean__cscs_codeformer | 2 | no |
| proper_visomaster_enhanced_clean__cscs_gfpgan_v1_4 | 11 | no |
| proper_visomaster_enhanced_clean__cscs_gpen_1024 | 3 | no |
| proper_visomaster_enhanced_clean__cscs_gpen_2048 | 4 | no |
| proper_visomaster_enhanced_clean__cscs_gpen_256 | 10 | no |
| proper_visomaster_enhanced_clean__cscs_gpen_512 | 7 | no |
| proper_visomaster_enhanced_clean__cscs_restoreformer | 9 | no |
| proper_visomaster_enhanced_clean__ghostface_v1_codeformer | 21 | YES |
| proper_visomaster_enhanced_clean__ghostface_v1_gfpgan_v1_4 | 26 | YES |
| proper_visomaster_enhanced_clean__ghostface_v1_gpen_1024 | 27 | YES |
| proper_visomaster_enhanced_clean__ghostface_v1_gpen_2048 | 28 | YES |
| proper_visomaster_enhanced_clean__ghostface_v1_gpen_256 | 28 | YES |
| proper_visomaster_enhanced_clean__ghostface_v1_gpen_512 | 23 | YES |
| proper_visomaster_enhanced_clean__ghostface_v1_restoreformer | 25 | YES |
| proper_visomaster_enhanced_clean__ghostface_v2_codeformer | 25 | YES |
| proper_visomaster_enhanced_clean__ghostface_v2_gfpgan_v1_4 | 25 | YES |
| proper_visomaster_enhanced_clean__ghostface_v2_gpen_1024 | 28 | YES |
| proper_visomaster_enhanced_clean__ghostface_v2_gpen_2048 | 24 | YES |
| proper_visomaster_enhanced_clean__ghostface_v2_gpen_256 | 25 | YES |
| proper_visomaster_enhanced_clean__ghostface_v2_gpen_512 | 28 | YES |
| proper_visomaster_enhanced_clean__ghostface_v2_restoreformer | 23 | YES |
| proper_visomaster_enhanced_clean__ghostface_v3_codeformer | 29 | YES |
| proper_visomaster_enhanced_clean__ghostface_v3_gfpgan_v1_4 | 27 | YES |
| proper_visomaster_enhanced_clean__ghostface_v3_gpen_1024 | 25 | YES |
| proper_visomaster_enhanced_clean__ghostface_v3_gpen_2048 | 26 | YES |
| proper_visomaster_enhanced_clean__ghostface_v3_gpen_256 | 27 | YES |
| proper_visomaster_enhanced_clean__ghostface_v3_gpen_512 | 25 | YES |
| proper_visomaster_enhanced_clean__ghostface_v3_restoreformer | 24 | YES |
| proper_visomaster_enhanced_clean__instyleswapper256_a_codeformer | 26 | YES |
| proper_visomaster_enhanced_clean__instyleswapper256_a_gfpgan_v1_4 | 28 | YES |
| proper_visomaster_enhanced_clean__instyleswapper256_a_gpen_1024 | 25 | YES |
| proper_visomaster_enhanced_clean__instyleswapper256_a_gpen_2048 | 27 | YES |
| proper_visomaster_enhanced_clean__instyleswapper256_a_gpen_256 | 25 | YES |
| proper_visomaster_enhanced_clean__instyleswapper256_a_gpen_512 | 26 | YES |
| proper_visomaster_enhanced_clean__instyleswapper256_a_restoreformer | 26 | YES |
| proper_visomaster_enhanced_clean__instyleswapper256_b_codeformer | 26 | YES |
| proper_visomaster_enhanced_clean__instyleswapper256_b_gfpgan_v1_4 | 28 | YES |
| proper_visomaster_enhanced_clean__instyleswapper256_b_gpen_1024 | 25 | YES |
| proper_visomaster_enhanced_clean__instyleswapper256_b_gpen_2048 | 24 | YES |
| proper_visomaster_enhanced_clean__instyleswapper256_b_gpen_256 | 24 | YES |
| proper_visomaster_enhanced_clean__instyleswapper256_b_gpen_512 | 26 | YES |
| proper_visomaster_enhanced_clean__instyleswapper256_b_restoreformer | 26 | YES |
| proper_visomaster_enhanced_clean__instyleswapper256_c_codeformer | 28 | YES |
| proper_visomaster_enhanced_clean__instyleswapper256_c_gfpgan_v1_4 | 27 | YES |
| proper_visomaster_enhanced_clean__instyleswapper256_c_gpen_1024 | 28 | YES |
| proper_visomaster_enhanced_clean__instyleswapper256_c_gpen_2048 | 26 | YES |
| proper_visomaster_enhanced_clean__instyleswapper256_c_gpen_256 | 27 | YES |
| proper_visomaster_enhanced_clean__instyleswapper256_c_gpen_512 | 26 | YES |
| proper_visomaster_enhanced_clean__instyleswapper256_c_restoreformer | 25 | YES |
| proper_visomaster_enhanced_clean__inswapper128_codeformer | 23 | YES |
| proper_visomaster_enhanced_clean__inswapper128_gfpgan_v1_4 | 24 | YES |
| proper_visomaster_enhanced_clean__inswapper128_gpen_1024 | 29 | YES |
| proper_visomaster_enhanced_clean__inswapper128_gpen_2048 | 24 | YES |
| proper_visomaster_enhanced_clean__inswapper128_gpen_256 | 24 | YES |
| proper_visomaster_enhanced_clean__inswapper128_gpen_512 | 29 | YES |
| proper_visomaster_enhanced_clean__inswapper128_restoreformer | 24 | YES |
| proper_visomaster_enhanced_clean__simswap512_codeformer | 27 | YES |
| proper_visomaster_enhanced_clean__simswap512_gfpgan_v1_4 | 22 | YES |
| proper_visomaster_enhanced_clean__simswap512_gpen_1024 | 25 | YES |
| proper_visomaster_enhanced_clean__simswap512_gpen_2048 | 26 | YES |
| proper_visomaster_enhanced_clean__simswap512_gpen_256 | 26 | YES |
| proper_visomaster_enhanced_clean__simswap512_gpen_512 | 23 | YES |
| proper_visomaster_enhanced_clean__simswap512_restoreformer | 24 | YES |

## Visomaster TEAMS (cropped frames only, no full mp4)

| method | videos | clears >=20 |
|---|---|---|
| proper_visomaster_enhanced_teams__cscs_codeformer | 2 | no |
| proper_visomaster_enhanced_teams__cscs_gfpgan_v1_4 | 11 | no |
| proper_visomaster_enhanced_teams__cscs_gpen_1024 | 3 | no |
| proper_visomaster_enhanced_teams__cscs_gpen_2048 | 4 | no |
| proper_visomaster_enhanced_teams__cscs_gpen_256 | 10 | no |
| proper_visomaster_enhanced_teams__cscs_gpen_512 | 7 | no |
| proper_visomaster_enhanced_teams__cscs_restoreformer | 9 | no |
| proper_visomaster_enhanced_teams__ghostface_v1_codeformer | 21 | YES |
| proper_visomaster_enhanced_teams__ghostface_v1_gfpgan_v1_4 | 26 | YES |
| proper_visomaster_enhanced_teams__ghostface_v1_gpen_1024 | 27 | YES |
| proper_visomaster_enhanced_teams__ghostface_v1_gpen_2048 | 28 | YES |
| proper_visomaster_enhanced_teams__ghostface_v1_gpen_256 | 28 | YES |
| proper_visomaster_enhanced_teams__ghostface_v1_gpen_512 | 23 | YES |
| proper_visomaster_enhanced_teams__ghostface_v1_restoreformer | 25 | YES |
| proper_visomaster_enhanced_teams__ghostface_v2_codeformer | 25 | YES |
| proper_visomaster_enhanced_teams__ghostface_v2_gfpgan_v1_4 | 25 | YES |
| proper_visomaster_enhanced_teams__ghostface_v2_gpen_1024 | 28 | YES |
| proper_visomaster_enhanced_teams__ghostface_v2_gpen_2048 | 24 | YES |
| proper_visomaster_enhanced_teams__ghostface_v2_gpen_256 | 25 | YES |
| proper_visomaster_enhanced_teams__ghostface_v2_gpen_512 | 28 | YES |
| proper_visomaster_enhanced_teams__ghostface_v2_restoreformer | 23 | YES |
| proper_visomaster_enhanced_teams__ghostface_v3_codeformer | 29 | YES |
| proper_visomaster_enhanced_teams__ghostface_v3_gfpgan_v1_4 | 27 | YES |
| proper_visomaster_enhanced_teams__ghostface_v3_gpen_1024 | 25 | YES |
| proper_visomaster_enhanced_teams__ghostface_v3_gpen_2048 | 26 | YES |
| proper_visomaster_enhanced_teams__ghostface_v3_gpen_256 | 27 | YES |
| proper_visomaster_enhanced_teams__ghostface_v3_gpen_512 | 25 | YES |
| proper_visomaster_enhanced_teams__ghostface_v3_restoreformer | 24 | YES |
| proper_visomaster_enhanced_teams__instyleswapper256_a_codeformer | 26 | YES |
| proper_visomaster_enhanced_teams__instyleswapper256_a_gfpgan_v1_4 | 28 | YES |
| proper_visomaster_enhanced_teams__instyleswapper256_a_gpen_1024 | 25 | YES |
| proper_visomaster_enhanced_teams__instyleswapper256_a_gpen_2048 | 27 | YES |
| proper_visomaster_enhanced_teams__instyleswapper256_a_gpen_256 | 25 | YES |
| proper_visomaster_enhanced_teams__instyleswapper256_a_gpen_512 | 26 | YES |
| proper_visomaster_enhanced_teams__instyleswapper256_a_restoreformer | 26 | YES |
| proper_visomaster_enhanced_teams__instyleswapper256_b_codeformer | 26 | YES |
| proper_visomaster_enhanced_teams__instyleswapper256_b_gfpgan_v1_4 | 28 | YES |
| proper_visomaster_enhanced_teams__instyleswapper256_b_gpen_1024 | 25 | YES |
| proper_visomaster_enhanced_teams__instyleswapper256_b_gpen_2048 | 24 | YES |
| proper_visomaster_enhanced_teams__instyleswapper256_b_gpen_256 | 24 | YES |
| proper_visomaster_enhanced_teams__instyleswapper256_b_gpen_512 | 26 | YES |
| proper_visomaster_enhanced_teams__instyleswapper256_b_restoreformer | 26 | YES |
| proper_visomaster_enhanced_teams__instyleswapper256_c_codeformer | 28 | YES |
| proper_visomaster_enhanced_teams__instyleswapper256_c_gfpgan_v1_4 | 27 | YES |
| proper_visomaster_enhanced_teams__instyleswapper256_c_gpen_1024 | 28 | YES |
| proper_visomaster_enhanced_teams__instyleswapper256_c_gpen_2048 | 26 | YES |
| proper_visomaster_enhanced_teams__instyleswapper256_c_gpen_256 | 27 | YES |
| proper_visomaster_enhanced_teams__instyleswapper256_c_gpen_512 | 26 | YES |
| proper_visomaster_enhanced_teams__instyleswapper256_c_restoreformer | 25 | YES |
| proper_visomaster_enhanced_teams__inswapper128_codeformer | 23 | YES |
| proper_visomaster_enhanced_teams__inswapper128_gfpgan_v1_4 | 24 | YES |
| proper_visomaster_enhanced_teams__inswapper128_gpen_1024 | 29 | YES |
| proper_visomaster_enhanced_teams__inswapper128_gpen_2048 | 24 | YES |
| proper_visomaster_enhanced_teams__inswapper128_gpen_256 | 24 | YES |
| proper_visomaster_enhanced_teams__inswapper128_gpen_512 | 29 | YES |
| proper_visomaster_enhanced_teams__inswapper128_restoreformer | 24 | YES |
| proper_visomaster_enhanced_teams__simswap512_codeformer | 27 | YES |
| proper_visomaster_enhanced_teams__simswap512_gfpgan_v1_4 | 22 | YES |
| proper_visomaster_enhanced_teams__simswap512_gpen_1024 | 25 | YES |
| proper_visomaster_enhanced_teams__simswap512_gpen_2048 | 26 | YES |
| proper_visomaster_enhanced_teams__simswap512_gpen_256 | 26 | YES |
| proper_visomaster_enhanced_teams__simswap512_gpen_512 | 23 | YES |
| proper_visomaster_enhanced_teams__simswap512_restoreformer | 24 | YES |
| proper_visomaster_teams__cscs | 12 | no |
| proper_visomaster_teams__ghostface_v1 | 41 | YES |
| proper_visomaster_teams__ghostface_v2 | 41 | YES |
| proper_visomaster_teams__ghostface_v3 | 38 | YES |
| proper_visomaster_teams__instyleswapper256_a | 41 | YES |
| proper_visomaster_teams__instyleswapper256_b | 41 | YES |
| proper_visomaster_teams__instyleswapper256_c | 43 | YES |
| proper_visomaster_teams__inswapper128 | 42 | YES |
| proper_visomaster_teams__simswap512 | 43 | YES |

## Visomaster bucket fallback (9 base swap models, clean, NO restorer breakdown)

| method | videos | clears >=20 |
|---|---|---|
| visomaster_bucket_clean__cscs | 602 | YES |
| visomaster_bucket_clean__ghostface_v1 | 602 | YES |
| visomaster_bucket_clean__ghostface_v2 | 602 | YES |
| visomaster_bucket_clean__ghostface_v3 | 602 | YES |
| visomaster_bucket_clean__instyleswapper256_a | 602 | YES |
| visomaster_bucket_clean__instyleswapper256_b | 673 | YES |
| visomaster_bucket_clean__instyleswapper256_c | 702 | YES |
| visomaster_bucket_clean__inswapper128 | 607 | YES |
| visomaster_bucket_clean__simswap512 | 602 | YES |

## Totals

- Visomaster methods >= 20: **128 / 144**
- Visomaster CLEAN methods >= 20: **64 / 72**
- Deeplive methods >= 20: **2 / 2**
- Visomaster bucket fallback methods >= 20: **9 / 9**
