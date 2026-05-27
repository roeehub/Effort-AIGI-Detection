# Phase 2 — within-stream score spread — PA_TOP_N_STEP3800

**NOTE on grouping**: per-`video_id` rows in the score reports are short segments (1-4 frames). True within-stream variance comes from grouping by session (e.g., 'Cam_Test__s32') and ordering by (seg_id, frame_idx). This phase reports stream-level stats at the session granularity. For deeplive_enhanced_dev where all 545 frames share a single session, treat each `video_id` as its own length-1 stream.

## Suite-level summary

| suite | label | n_streams | frames p50 | frames p95 | median p50 | IQR p50 | std p50 | frac>=0.5 p50 | frac>=0.5 p95 | pct_w/_any>=0.95 | pct_w/_any>=0.98 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| teams_real_all_dev | real | 1247 | 1 | 1 | 0.148 | 0.000 | 0.000 | 0.000 | 1.000 | 0.046 | 0.020 |
| teams_real_all_lockbox | real | 1250 | 1 | 1 | 0.254 | 0.000 | 0.000 | 0.000 | 1.000 | 0.020 | 0.004 |
| teams_real_dor_dev | real | 50 | 1 | 1 | 0.663 | 0.000 | 0.000 | 1.000 | 1.000 | 0.060 | 0.040 |
| teams_real_poor_quality_dev | real | 312 | 1 | 1 | 0.163 | 0.000 | 0.000 | 0.000 | 1.000 | 0.016 | 0.013 |
| teams_real_lighting_extreme_dev | real | 775 | 1 | 1 | 0.348 | 0.000 | 0.000 | 0.000 | 1.000 | 0.080 | 0.035 |
| teams_fake_all_dev | fake | 1242 | 1 | 1 | 0.628 | 0.000 | 0.000 | 1.000 | 1.000 | 0.103 | 0.028 |
| teams_fake_all_lockbox | fake | 2 | 212 | 322 | 0.827 | 0.205 | 0.148 | 0.868 | 0.987 | 1.000 | 1.000 |
| deeplive_enhanced_dev | fake | 545 | 1 | 1 | 0.870 | 0.000 | 0.000 | 1.000 | 1.000 | 0.191 | 0.037 |

## Override-rule pre-check

`P(any frame above τ | label_class)` — high gap between fake and real means the override is safe.

| suite | label | τ | n | P(any≥τ) |
|---|---|---:|---:|---:|
| deeplive_enhanced_dev | fake | 0.95 | 545 | 0.191 |
| teams_fake_all_dev | fake | 0.95 | 1242 | 0.103 |
| teams_fake_all_lockbox | fake | 0.95 | 2 | 1.000 |
| teams_real_all_dev | real | 0.95 | 1247 | 0.046 |
| teams_real_all_lockbox | real | 0.95 | 1250 | 0.020 |
| teams_real_dor_dev | real | 0.95 | 50 | 0.060 |
| teams_real_lighting_extreme_dev | real | 0.95 | 775 | 0.080 |
| teams_real_poor_quality_dev | real | 0.95 | 312 | 0.016 |
| deeplive_enhanced_dev | fake | 0.98 | 545 | 0.037 |
| teams_fake_all_dev | fake | 0.98 | 1242 | 0.028 |
| teams_fake_all_lockbox | fake | 0.98 | 2 | 1.000 |
| teams_real_all_dev | real | 0.98 | 1247 | 0.020 |
| teams_real_all_lockbox | real | 0.98 | 1250 | 0.004 |
| teams_real_dor_dev | real | 0.98 | 50 | 0.040 |
| teams_real_lighting_extreme_dev | real | 0.98 | 775 | 0.035 |
| teams_real_poor_quality_dev | real | 0.98 | 312 | 0.013 |

_Wall time: 4.2s_