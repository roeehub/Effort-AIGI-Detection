# Phase 4 — Pareto sweep — P8A_REFERENCE_STEP5000

**FPR cap used (primary)**: 0.1

**Chosen policy** (#240): `majority_vote` W=16 params={'threshold': 0.7, 'vote_majority': 0.4} override=`none`
- fake_macro_recall: **0.621**
- real_max_fpr: **0.095**

**Per-suite rates**:
- deeplive_enhanced_dev|fake: 0.588
- teams_fake_all_dev|fake: 0.347
- teams_fake_all_lockbox|fake: 0.929
- teams_real_all_dev|real: 0.064
- teams_real_all_lockbox|real: 0.018
- teams_real_dor_dev|real: 0.000
- teams_real_lighting_extreme_dev|real: 0.095
- teams_real_poor_quality_dev|real: 0.029

**Top-3 at primary cap**:

| rank | strategy | W | params | override | fake_macro_R | real_max_FPR |
|---:|---|---:|---|---|---:|---:|
| 1 | majority_vote | 16 | {'threshold': 0.7, 'vote_majority': 0.4} | none | 0.621 | 0.095 |
| 2 | majority_vote | 16 | {'threshold': 0.7, 'vote_majority': 0.4} | three_consec_frames_above_0.95 | 0.621 | 0.095 |
| 3 | majority_vote | 16 | {'threshold': 0.7, 'vote_majority': 0.4} | five_consec_frames_above_0.90 | 0.621 | 0.095 |

## Best policy at each FPR cap (any strategy / override)

| FPR cap | strategy | W | params | override | fake_macro_R | real_max_FPR | per-suite (deeplive / fake_dev / fake_lockbox) |
|---|---|---:|---|---|---:|---:|---|
| FPR<=0.10_best | majority_vote | 16 | {'threshold': 0.7, 'vote_majority': 0.4} | none | 0.621 | 0.095 | 0.588 / 0.347 / 0.929 |
| FPR<=0.15_best | majority_vote | 32 | {'threshold': 0.5, 'vote_majority': 0.5} | none | 0.679 | 0.121 | 0.588 / 0.449 / 1.000 |
| FPR<=0.20_best | majority_vote | 16 | {'threshold': 0.3, 'vote_majority': 0.7} | three_consec_frames_above_0.95 | 0.680 | 0.155 | 0.647 / 0.537 / 0.857 |

## Best SPIKE-RESISTANT policy at each FPR cap

(excluding `run_length` strategy and `*_consec_*` overrides — these are vulnerable to short bursts of high-prob frames)

| FPR cap | strategy | W | params | override | fake_macro_R | real_max_FPR | per-suite (deeplive / fake_dev / fake_lockbox) |
|---|---|---:|---|---|---:|---:|---|
| FPR<=0.10_best_spike_resistant | majority_vote | 16 | {'threshold': 0.7, 'vote_majority': 0.4} | none | 0.621 | 0.095 | 0.588 / 0.347 / 0.929 |
| FPR<=0.15_best_spike_resistant | majority_vote | 32 | {'threshold': 0.5, 'vote_majority': 0.5} | none | 0.679 | 0.121 | 0.588 / 0.449 / 1.000 |
| FPR<=0.20_best_spike_resistant | majority_vote | 32 | {'threshold': 0.5, 'vote_majority': 0.5} | none | 0.679 | 0.121 | 0.588 / 0.449 / 1.000 |

_Wall time: 16.3s, 8040 work units, 8 workers_