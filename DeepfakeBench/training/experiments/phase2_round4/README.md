# Phase 2 Round 4 (Family-Aware Quality Augmentation)

All runs fine-tune from the R25_F1 checkpoint and keep `albumentations==0.4.6` compatibility.

## Run Matrix

- `R4_FT1_base_noenhanced.yaml`: `base_only`, no `*_enhanced` DeepLive strategies.
- `R4_FT2_family_light_noenhanced.yaml`: `quality_targeted_family` + `light`, no `*_enhanced` strategies.
- `R4_FT3_family_moderate_noenhanced.yaml`: `quality_targeted_family` + `moderate`, no `*_enhanced` strategies.
- `R4_FT4_base_withenhanced.yaml`: `base_only`, includes enhanced DeepLive strategies (860-set).
- `R4_FT5_family_light_withenhanced.yaml`: `quality_targeted_family` + `light`, includes enhanced strategies.
- `R4_FT6_family_moderate_withenhanced.yaml`: `quality_targeted_family` + `moderate`, includes enhanced strategies.
- `R4_FT7_family_light_withenhanced_weighted.yaml`: same as FT5 + `identity_resample_weighted` with DeepLive-upweighted families.
- `R4_FT8_family_moderate_withenhanced_weighted.yaml`: same as FT6 + stronger DeepLive-upweighted family sampling.

## Gates

- External real FPR `<= 8%`
- WMA failure fake detection `>= 50%`

Use `validate_custom_sources.py` with detailed reports enabled to generate:

- `<prefix>frames_report.csv`
- `<prefix>videos_report.csv`
- `<prefix>group_metrics.csv`
- `<prefix>summary_report.txt`

For sequential sidecar validation across FT1-FT8, use:

- `run_r4_validation_sequential.py`

For canonical WMA gate reporting (flat per-image on 1202 images), use:

- `--external_fake_grouping per_image`
- `--external_fake_deterministic`
- `--max_external_fake 1202`
