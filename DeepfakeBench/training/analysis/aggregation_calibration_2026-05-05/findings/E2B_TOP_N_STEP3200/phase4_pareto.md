# Phase 4 — Pareto sweep — E2B_TOP_N_STEP3200

**FPR cap used (primary)**: 0.1

**Chosen policy** (#948): `run_length` W=16 params={'M': 5, 'threshold': 0.9} override=`three_consec_frames_above_0.95`
- fake_macro_recall: **0.446**
- real_max_fpr: **0.057**

**Per-suite rates**:
- deeplive_enhanced_dev|fake: 0.647
- teams_fake_all_dev|fake: 0.262
- teams_fake_all_lockbox|fake: 0.429
- teams_real_all_dev|real: 0.037
- teams_real_all_lockbox|real: 0.008
- teams_real_dor_dev|real: 0.000
- teams_real_lighting_extreme_dev|real: 0.057
- teams_real_poor_quality_dev|real: 0.022

**Top-3 at primary cap**:

| rank | strategy | W | params | override | fake_macro_R | real_max_FPR |
|---:|---|---:|---|---|---:|---:|
| 1 | run_length | 16 | {'M': 5, 'threshold': 0.9} | three_consec_frames_above_0.95 | 0.446 | 0.057 |
| 2 | run_length | 32 | {'M': 5, 'threshold': 0.9} | three_consec_frames_above_0.95 | 0.446 | 0.057 |
| 3 | run_length | 48 | {'M': 5, 'threshold': 0.9} | three_consec_frames_above_0.95 | 0.446 | 0.057 |

## Best policy at each FPR cap (any strategy / override)

| FPR cap | strategy | W | params | override | fake_macro_R | real_max_FPR | per-suite (deeplive / fake_dev / fake_lockbox) |
|---|---|---:|---|---|---:|---:|---|
| FPR<=0.10_best | run_length | 16 | {'M': 5, 'threshold': 0.9} | three_consec_frames_above_0.95 | 0.446 | 0.057 | 0.647 / 0.262 / 0.429 |
| FPR<=0.15_best | run_length | 16 | {'M': 5, 'threshold': 0.9} | three_consec_frames_above_0.95 | 0.446 | 0.057 | 0.647 / 0.262 / 0.429 |
| FPR<=0.20_best | majority_vote | 16 | {'threshold': 0.7, 'vote_majority': 0.4} | none | 0.813 | 0.172 | 1.000 / 0.440 / 1.000 |

## Best SPIKE-RESISTANT policy at each FPR cap

(excluding `run_length` strategy and `*_consec_*` overrides — these are vulnerable to short bursts of high-prob frames)

| FPR cap | strategy | W | params | override | fake_macro_R | real_max_FPR | per-suite (deeplive / fake_dev / fake_lockbox) |
|---|---|---:|---|---|---:|---:|---|
| FPR<=0.20_best_spike_resistant | majority_vote | 16 | {'threshold': 0.7, 'vote_majority': 0.4} | none | 0.813 | 0.172 | 1.000 / 0.440 / 1.000 |

_Wall time: 19.1s, 8040 work units, 8 workers_