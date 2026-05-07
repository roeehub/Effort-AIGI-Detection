# F3 untargeted-axis audit — color_b_dev (B-channel std)

## Method

Read 8698 (ckpt-shared) per-frame Phase A reports across 8 ckpts × 4 suites at `analysis/p1_pe_eval_2026-05-07/raw_reports/phase_a/`. Downloaded 7603 unique source frames into local cache (`_frame_cache/` and reused `dor_invariance_2026-05-07/_axis_cache/`); decoded with `cv2.imdecode` (BGR uint8 0-255) and computed `color_b_dev = std(B_channel)`. Per (ckpt, suite) Pearson r between `frame_prob` and `color_b_dev` computed via `numpy.corrcoef` after dropping NaNs.

No subsampling was performed (full per-suite frame sets used).

## Per-ckpt mean |r| on real-side and fake-side

| ckpt | mean_abs_r_real | mean_abs_r_fake |
| --- | --- | --- |
| p8a | 0.0911 | 0.1970 |
| e2b | 0.0245 | 0.1764 |
| p1_bundle_periodic_500 | 0.0138 | 0.1454 |
| p1_bundle_top_n_3750 | 0.0224 | 0.1216 |
| p1_bundle_top_n_4000 | 0.0151 | 0.1159 |
| p1_pairrank_periodic_500 | 0.0163 | 0.1643 |
| p1_pairrank_top_n_6000 | 0.0043 | 0.1290 |
| p1_pairrank_top_n_6750 | 0.0002 | 0.1324 |

## Δ |r| pct vs P8A

| ckpt | delta_real_pct_vs_p8a | delta_fake_pct_vs_p8a |
| --- | --- | --- |
| p8a | 0.00 | 0.00 |
| e2b | -73.08 | -10.46 |
| p1_bundle_periodic_500 | -84.82 | -26.20 |
| p1_bundle_top_n_3750 | -75.38 | -38.27 |
| p1_bundle_top_n_4000 | -83.42 | -41.16 |
| p1_pairrank_periodic_500 | -82.05 | -16.58 |
| p1_pairrank_top_n_6000 | -95.26 | -34.51 |
| p1_pairrank_top_n_6750 | -99.77 | -32.82 |

## Per-(ckpt × suite) Pearson r

| ckpt | suite | kind | n | pearson_r |
| --- | --- | --- | --- | --- |
| p8a | visomaster_enhanced_macro_dev | fake | 550 | -0.1569 |
| p8a | deeplive_enhanced_dev | fake | 545 | +0.0187 |
| p8a | teams_fake_all_dev | fake | 3039 | -0.4154 |
| p8a | teams_real_all_dev | real | 4564 | -0.0911 |
| e2b | visomaster_enhanced_macro_dev | fake | 550 | +0.1799 |
| e2b | deeplive_enhanced_dev | fake | 545 | +0.1791 |
| e2b | teams_fake_all_dev | fake | 3039 | -0.1703 |
| e2b | teams_real_all_dev | real | 4564 | +0.0245 |
| p1_bundle_periodic_500 | visomaster_enhanced_macro_dev | fake | 550 | -0.1157 |
| p1_bundle_periodic_500 | deeplive_enhanced_dev | fake | 545 | +0.1559 |
| p1_bundle_periodic_500 | teams_fake_all_dev | fake | 3039 | -0.1646 |
| p1_bundle_periodic_500 | teams_real_all_dev | real | 4564 | -0.0138 |
| p1_bundle_top_n_3750 | visomaster_enhanced_macro_dev | fake | 550 | -0.0756 |
| p1_bundle_top_n_3750 | deeplive_enhanced_dev | fake | 545 | +0.2116 |
| p1_bundle_top_n_3750 | teams_fake_all_dev | fake | 3039 | -0.0776 |
| p1_bundle_top_n_3750 | teams_real_all_dev | real | 4564 | +0.0224 |
| p1_bundle_top_n_4000 | visomaster_enhanced_macro_dev | fake | 550 | -0.0911 |
| p1_bundle_top_n_4000 | deeplive_enhanced_dev | fake | 545 | +0.1752 |
| p1_bundle_top_n_4000 | teams_fake_all_dev | fake | 3039 | -0.0815 |
| p1_bundle_top_n_4000 | teams_real_all_dev | real | 4564 | +0.0151 |
| p1_pairrank_periodic_500 | visomaster_enhanced_macro_dev | fake | 550 | -0.2212 |
| p1_pairrank_periodic_500 | deeplive_enhanced_dev | fake | 545 | +0.0352 |
| p1_pairrank_periodic_500 | teams_fake_all_dev | fake | 3039 | -0.2367 |
| p1_pairrank_periodic_500 | teams_real_all_dev | real | 4564 | +0.0163 |
| p1_pairrank_top_n_6000 | visomaster_enhanced_macro_dev | fake | 550 | -0.0607 |
| p1_pairrank_top_n_6000 | deeplive_enhanced_dev | fake | 545 | +0.2012 |
| p1_pairrank_top_n_6000 | teams_fake_all_dev | fake | 3039 | -0.1252 |
| p1_pairrank_top_n_6000 | teams_real_all_dev | real | 4564 | +0.0043 |
| p1_pairrank_top_n_6750 | visomaster_enhanced_macro_dev | fake | 550 | -0.0347 |
| p1_pairrank_top_n_6750 | deeplive_enhanced_dev | fake | 545 | +0.2982 |
| p1_pairrank_top_n_6750 | teams_fake_all_dev | fake | 3039 | -0.0641 |
| p1_pairrank_top_n_6750 | teams_real_all_dev | real | 4564 | +0.0002 |

## Subsampling

None. All Phase A frames per suite were used.

## Coverage and caveats

- Total unique URIs across 4 suites: **7603**
- Successfully fetched: **7603**
- Download failures: **0** (0.00%)
- Decode failures (image corrupt / unsupported): **0**

## F3 close criterion

F3 close criterion: "no untargeted axis amplifies +50%". Mechanical pass/fail per ckpt against `color_b_dev`:

| ckpt | delta_real_pct_vs_p8a | delta_fake_pct_vs_p8a | real_amplifies_>50% | fake_amplifies_>50% |
| --- | --- | --- | --- | --- |
| e2b | -73.08 | -10.46 | no | no |
| p1_bundle_periodic_500 | -84.82 | -26.20 | no | no |
| p1_bundle_top_n_3750 | -75.38 | -38.27 | no | no |
| p1_bundle_top_n_4000 | -83.42 | -41.16 | no | no |
| p1_pairrank_periodic_500 | -82.05 | -16.58 | no | no |
| p1_pairrank_top_n_6000 | -95.26 | -34.51 | no | no |
| p1_pairrank_top_n_6750 | -99.77 | -32.82 | no | no |
