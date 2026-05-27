# Spike robustness — E2B_TOP_N_STEP3200

Inject K consecutive frames at 0.99 into REAL multi-frame streams (>=32 frames). Measure the FPR delta. Policy is robust if delta is small (< 5pp).

| suite | policy | K_spike | n | baseline_FPR | spiked_FPR | Δ |
|---|---|---:|---:|---:|---:|---:|
| teams_real_all_dev | vanilla | 4 | 15 | 0.200 | 0.267 | +0.067 |
| teams_real_all_dev | chosen | 4 | 15 | 0.067 | 1.000 | +0.933 |
| teams_real_all_dev | chosen_no_override | 4 | 15 | 0.067 | 0.133 | +0.067 |
| teams_real_all_dev | vanilla | 8 | 15 | 0.200 | 0.267 | +0.067 |
| teams_real_all_dev | chosen | 8 | 15 | 0.067 | 1.000 | +0.933 |
| teams_real_all_dev | chosen_no_override | 8 | 15 | 0.067 | 1.000 | +0.933 |
| teams_real_all_dev | vanilla | 12 | 15 | 0.200 | 0.267 | +0.067 |
| teams_real_all_dev | chosen | 12 | 15 | 0.067 | 1.000 | +0.933 |
| teams_real_all_dev | chosen_no_override | 12 | 15 | 0.067 | 1.000 | +0.933 |
| teams_real_all_lockbox | vanilla | 4 | 2 | 0.000 | 0.000 | +0.000 |
| teams_real_all_lockbox | chosen | 4 | 2 | 0.000 | 1.000 | +1.000 |
| teams_real_all_lockbox | chosen_no_override | 4 | 2 | 0.000 | 0.000 | +0.000 |
| teams_real_all_lockbox | vanilla | 8 | 2 | 0.000 | 0.000 | +0.000 |
| teams_real_all_lockbox | chosen | 8 | 2 | 0.000 | 1.000 | +1.000 |
| teams_real_all_lockbox | chosen_no_override | 8 | 2 | 0.000 | 1.000 | +1.000 |
| teams_real_all_lockbox | vanilla | 12 | 2 | 0.000 | 0.000 | +0.000 |
| teams_real_all_lockbox | chosen | 12 | 2 | 0.000 | 1.000 | +1.000 |
| teams_real_all_lockbox | chosen_no_override | 12 | 2 | 0.000 | 1.000 | +1.000 |
| teams_real_dor_dev | vanilla | 4 | 1 | 0.000 | 0.000 | +0.000 |
| teams_real_dor_dev | chosen | 4 | 1 | 0.000 | 1.000 | +1.000 |
| teams_real_dor_dev | chosen_no_override | 4 | 1 | 0.000 | 0.000 | +0.000 |
| teams_real_dor_dev | vanilla | 8 | 1 | 0.000 | 0.000 | +0.000 |
| teams_real_dor_dev | chosen | 8 | 1 | 0.000 | 1.000 | +1.000 |
| teams_real_dor_dev | chosen_no_override | 8 | 1 | 0.000 | 1.000 | +1.000 |
| teams_real_dor_dev | vanilla | 12 | 1 | 0.000 | 1.000 | +1.000 |
| teams_real_dor_dev | chosen | 12 | 1 | 0.000 | 1.000 | +1.000 |
| teams_real_dor_dev | chosen_no_override | 12 | 1 | 0.000 | 1.000 | +1.000 |
| teams_real_poor_quality_dev | vanilla | 4 | 8 | 0.250 | 0.250 | +0.000 |
| teams_real_poor_quality_dev | chosen | 4 | 8 | 0.000 | 1.000 | +1.000 |
| teams_real_poor_quality_dev | chosen_no_override | 4 | 8 | 0.000 | 0.000 | +0.000 |
| teams_real_poor_quality_dev | vanilla | 8 | 8 | 0.250 | 0.250 | +0.000 |
| teams_real_poor_quality_dev | chosen | 8 | 8 | 0.000 | 1.000 | +1.000 |
| teams_real_poor_quality_dev | chosen_no_override | 8 | 8 | 0.000 | 1.000 | +1.000 |
| teams_real_poor_quality_dev | vanilla | 12 | 8 | 0.250 | 0.250 | +0.000 |
| teams_real_poor_quality_dev | chosen | 12 | 8 | 0.000 | 1.000 | +1.000 |
| teams_real_poor_quality_dev | chosen_no_override | 12 | 8 | 0.000 | 1.000 | +1.000 |
| teams_real_lighting_extreme_dev | vanilla | 4 | 6 | 0.000 | 0.000 | +0.000 |
| teams_real_lighting_extreme_dev | chosen | 4 | 6 | 0.000 | 1.000 | +1.000 |
| teams_real_lighting_extreme_dev | chosen_no_override | 4 | 6 | 0.000 | 0.000 | +0.000 |
| teams_real_lighting_extreme_dev | vanilla | 8 | 6 | 0.000 | 0.000 | +0.000 |
| teams_real_lighting_extreme_dev | chosen | 8 | 6 | 0.000 | 1.000 | +1.000 |
| teams_real_lighting_extreme_dev | chosen_no_override | 8 | 6 | 0.000 | 1.000 | +1.000 |
| teams_real_lighting_extreme_dev | vanilla | 12 | 6 | 0.000 | 0.333 | +0.333 |
| teams_real_lighting_extreme_dev | chosen | 12 | 6 | 0.000 | 1.000 | +1.000 |
| teams_real_lighting_extreme_dev | chosen_no_override | 12 | 6 | 0.000 | 1.000 | +1.000 |