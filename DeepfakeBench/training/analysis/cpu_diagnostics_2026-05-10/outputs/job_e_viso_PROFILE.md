# Job E — Why viso is harder than deeplive across ckpts

For each ckpt, characterize the score distribution + IQ profile of MISSED fake frames (score < 0.5) vs CAUGHT fake frames on viso vs deeplive.

## P8A

### visomaster_enhanced_macro_dev — per-method miss rate at τ=0.5

| group_key | n | miss_rate | score_p50 |
|---|---:|---:|---:|
| visomaster_enhanced_fake | 550 | 64% | 0.170 |

### P8A — viso miss profile vs deeplive miss profile

| axis | viso_caught_p50 | viso_missed_p50 | deeplive_caught_p50 | deeplive_missed_p50 |
|---|---:|---:|---:|---:|
| lap_var | 74.3 | 28.0 | 240.7 | 245.7 |
| min_dim | 384.0 | 377.0 | 414.0 | 413.0 |
| color_a_dev | 9.0 | 8.6 | 5.7 | 5.6 |
| color_b_dev | 13.6 | 13.9 | 11.3 | 11.1 |
| luma_mean | 171.1 | 188.0 | 143.8 | 143.5 |
| saturation_mean | 67.3 | 68.7 | 70.8 | 69.8 |

## E2B

### visomaster_enhanced_macro_dev — per-method miss rate at τ=0.5

| group_key | n | miss_rate | score_p50 |
|---|---:|---:|---:|
| visomaster_enhanced_fake | 550 | 92% | 0.025 |

### E2B — viso miss profile vs deeplive miss profile

| axis | viso_caught_p50 | viso_missed_p50 | deeplive_caught_p50 | deeplive_missed_p50 |
|---|---:|---:|---:|---:|
| lap_var | 15.6 | 40.2 | 242.0 | 236.7 |
| min_dim | 384.0 | 379.0 | 414.0 | 411.5 |
| color_a_dev | 7.7 | 8.8 | 5.6 | 5.7 |
| color_b_dev | 14.1 | 13.8 | 11.1 | 11.5 |
| luma_mean | 192.7 | 175.4 | 143.7 | 144.3 |
| saturation_mean | 63.6 | 68.6 | 70.1 | 70.8 |

## T3_S1_step1500

### visomaster_enhanced_macro_dev — per-method miss rate at τ=0.5

| group_key | n | miss_rate | score_p50 |
|---|---:|---:|---:|
| visomaster_enhanced_fake | 550 | 74% | 0.170 |

### T3_S1_step1500 — viso miss profile vs deeplive miss profile

| axis | viso_caught_p50 | viso_missed_p50 | deeplive_caught_p50 | deeplive_missed_p50 |
|---|---:|---:|---:|---:|
| lap_var | 74.8 | 32.3 | 242.4 | 239.5 |
| min_dim | 385.0 | 378.0 | 413.0 | 413.0 |
| color_a_dev | 9.1 | 8.7 | 5.6 | 5.6 |
| color_b_dev | 13.5 | 13.9 | 11.2 | 11.1 |
| luma_mean | 171.6 | 185.4 | 143.7 | 143.8 |
| saturation_mean | 67.8 | 68.6 | 70.5 | 69.8 |

## T3_S1_step2500

### visomaster_enhanced_macro_dev — per-method miss rate at τ=0.5

| group_key | n | miss_rate | score_p50 |
|---|---:|---:|---:|
| visomaster_enhanced_fake | 550 | 35% | 0.717 |

### T3_S1_step2500 — viso miss profile vs deeplive miss profile

| axis | viso_caught_p50 | viso_missed_p50 | deeplive_caught_p50 | deeplive_missed_p50 |
|---|---:|---:|---:|---:|
| lap_var | 56.9 | 25.7 | 242.1 | 100.8 |
| min_dim | 381.5 | 376.0 | 413.0 | 413.0 |
| color_a_dev | 8.6 | 8.9 | 5.6 | 4.7 |
| color_b_dev | 13.8 | 13.9 | 11.2 | 11.0 |
| luma_mean | 187.0 | 173.5 | 143.7 | 145.2 |
| saturation_mean | 67.1 | 69.6 | 70.2 | 66.8 |
