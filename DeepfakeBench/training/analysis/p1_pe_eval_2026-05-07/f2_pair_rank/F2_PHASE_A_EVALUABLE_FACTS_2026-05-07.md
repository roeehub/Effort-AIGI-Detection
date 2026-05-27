# F2 Pair-Rank, Phase-A-Evaluable Reformulation — P1 vs P8A

Status: factual-only.

## Why this reformulation exists

### Question answered

What does F2 ("≥30% relative pair-rank lift on ≥2 of 6 paired training lanes among previously-missed fakes") look like when re-expressed against axes Phase A actually scores, instead of yaml-named training lanes that have no Phase A coverage?

### Method

Phase A's `analysis/p1_pe_eval_2026-05-07/raw_reports/phase_a/` contains per-frame scores for 6 P1 ckpts plus `p8a_reference_step5000` plus `e2b_top_n_step3200` across 29 suites. Of the 6 paired training lanes named in `R13_P1_BUNDLE_FT_FROM_P8A.yaml`, 5 (`df40`, `deeplive`, `visomaster_v1_base`, `visomaster_enhanced`, `visomaster_teams_enhanced`) have no Phase A frame coverage. The 6th (`deeplive_teams`) maps to the eval-substrate teams_passthrough cross-product but every active sub-lane there has P8A baseline frac ≥ 0.97 (per `F2_PAIR_RANK_FACTS_2026-05-07.md` §"P8A baseline (reference)"), so the 30%-relative bar is structurally unreachable on that lane.

The reformulation replaces the yaml-named lane axis with the Phase A fake-suite axis:
1. For each Phase A fake suite F, identify P8A-missed fakes (P8A `frame_prob < 0.5`).
2. Real pool = union of all label==0 frames in 7 real suites (`teams_real_all_dev`, `teams_real_all_lockbox`, `teams_real_dor_dev`, `teams_real_lighting_extreme_dev`, `teams_real_lighting_extreme_lockbox`, `teams_real_poor_quality_dev`, `teams_real_poor_quality_lockbox`).
3. Per-frame metric for missed fake `f`: `mean over reals r of 1[score(f) > score(r)]`.
4. Suite-level metric: mean of the per-frame quantity across all P8A-missed fakes in F.
5. Lift vs P8A baseline: relative percent change in the suite-level metric.

Pre-stated bar: ≥30% relative lift on ≥2 of N evaluable fake suites.

### Output artifacts

- `f2_pair_rank/run_f2_phase_a_evaluable.py` — script (re-runnable; reads Phase A reports directly, no caches).
- `f2_pair_rank/phase_a_evaluable_pairing.csv` — long-form per-(suite, ckpt, fake_path) pair-gap probability values.
- `f2_pair_rank/phase_a_evaluable_per_suite_per_ckpt_lift.csv` — per-(suite, ckpt) summary with `mean_pair_gap_prob`, `lift_abs`, `lift_rel_pct`, `passes_30pct`.
- `f2_pair_rank/phase_a_evaluable_pass_summary.csv` — per-ckpt count of evaluable suites passing the 30% bar.

## Suite-axis definitions

### Fake suites (5)

These are the top-level non-overlapping fake suites scored on Phase A. Per-subject `teams_capture_*_dev` and `teams_capture_*_s*_dev` suites are subsets of `teams_fake_all_dev` (and lockbox analogue) and were excluded to avoid double-counting (`f2_pair_rank/phase_a_evaluable_per_suite_per_ckpt_lift.csv`).

| Fake suite | Total fakes (Phase A) | P8A-missed (frame_prob < 0.5) | %% missed |
|---|---:|---:|---:|
| deeplive_enhanced_dev | 545 | 256 | 47.0% |
| teams_fake_all_dev | 3039 | 740 | 24.4% |
| teams_fake_all_lockbox | 425 | 143 | 33.6% |
| teams_flat_xiang_xiang2_feng_dev | 135 | 74 | 54.8% |
| visomaster_enhanced_macro_dev | 550 | 354 | 64.4% |

### Real pool (7 suites, union)

| Real suite | Frames |
|---|---:|
| teams_real_all_dev | 4564 |
| teams_real_all_lockbox | 1418 |
| teams_real_dor_dev | 50 |
| teams_real_lighting_extreme_dev | 1742 |
| teams_real_lighting_extreme_lockbox | 214 |
| teams_real_poor_quality_dev | 1303 |
| teams_real_poor_quality_lockbox | 31 |
| **Union (deduplicated by frame_path)** | **6032** |

The union total (6032) is less than the sum (9322) because real-pool suites overlap by frame_path (e.g., `teams_real_lighting_extreme_dev` and `teams_real_poor_quality_dev` are subsets of `teams_real_all_dev`). The script deduplicates by `frame_path` via dict-merge.

## P8A baseline (reference) — `mean_pair_gap_prob`

Per `f2_pair_rank/phase_a_evaluable_per_suite_per_ckpt_lift.csv` rows where `ckpt = p8a_reference_step5000`.

| Suite | n_missed | mean_pair_gap_prob |
|---|---:|---:|
| deeplive_enhanced_dev | 256 | 0.780332 |
| teams_fake_all_dev | 740 | 0.712664 |
| teams_fake_all_lockbox | 143 | 0.798776 |
| teams_flat_xiang_xiang2_feng_dev | 74 | 0.797779 |
| visomaster_enhanced_macro_dev | 354 | 0.627094 |

All 5 baselines are strictly between 0 and 1 (no 0-saturated baseline; no 1-saturated baseline). Maximum reachable `lift_rel_pct` per suite under the [0, 1] range:

| Suite | P8A baseline | Max lift_rel_pct (= (1 - p8a)/p8a × 100) |
|---|---:|---:|
| deeplive_enhanced_dev | 0.780332 | +28.15% |
| teams_fake_all_dev | 0.712664 | +40.32% |
| teams_fake_all_lockbox | 0.798776 | +25.19% |
| teams_flat_xiang_xiang2_feng_dev | 0.797779 | +25.35% |
| visomaster_enhanced_macro_dev | 0.627094 | +59.46% |

The 30%-relative bar is reachable on 2 of 5 suites at the upper bound of the metric range (`teams_fake_all_dev` and `visomaster_enhanced_macro_dev`); the other 3 cap at +28%, +25%, +25% (below the 30% bar). The metric is structurally limited from above by the unmissed-fakes ratio in the real pool.

## P1 ckpt × suite lift table

Per `f2_pair_rank/phase_a_evaluable_per_suite_per_ckpt_lift.csv`. `lift_rel_pct` rounded to 2 decimals.

### deeplive_enhanced_dev (n_missed=256, P8A=0.7803)

| Ckpt | mean_pair_gap_prob | lift_abs | lift_rel_pct | passes 30%% bar |
|---|---:|---:|---:|:---:|
| p1_bundle_periodic_step500 | 0.717700 | -0.062632 | -8.03 | no |
| p1_bundle_top_n_step3750 | 0.856284 | +0.075952 | +9.73 | no |
| p1_bundle_top_n_step4000 | 0.865356 | +0.085024 | +10.90 | no |
| p1_pairrank_periodic_step500 | 0.874752 | +0.094420 | +12.10 | no |
| p1_pairrank_top_n_step6000 | 0.897673 | +0.117341 | +15.04 | no |
| p1_pairrank_top_n_step6750 | 0.904211 | +0.123880 | +15.88 | no |

### teams_fake_all_dev (n_missed=740, P8A=0.7127)

| Ckpt | mean_pair_gap_prob | lift_abs | lift_rel_pct | passes 30%% bar |
|---|---:|---:|---:|:---:|
| p1_bundle_periodic_step500 | 0.675878 | -0.036785 | -5.16 | no |
| p1_bundle_top_n_step3750 | 0.775257 | +0.062593 | +8.78 | no |
| p1_bundle_top_n_step4000 | 0.784218 | +0.071554 | +10.04 | no |
| p1_pairrank_periodic_step500 | 0.773737 | +0.061074 | +8.57 | no |
| p1_pairrank_top_n_step6000 | 0.802211 | +0.089547 | +12.57 | no |
| p1_pairrank_top_n_step6750 | 0.801259 | +0.088596 | +12.43 | no |

### teams_fake_all_lockbox (n_missed=143, P8A=0.7988)

| Ckpt | mean_pair_gap_prob | lift_abs | lift_rel_pct | passes 30%% bar |
|---|---:|---:|---:|:---:|
| p1_bundle_periodic_step500 | 0.963231 | +0.164455 | +20.59 | no |
| p1_bundle_top_n_step3750 | 0.746825 | -0.051951 | -6.50 | no |
| p1_bundle_top_n_step4000 | 0.783582 | -0.015194 | -1.90 | no |
| p1_pairrank_periodic_step500 | 0.916196 | +0.117420 | +14.70 | no |
| p1_pairrank_top_n_step6000 | 0.744543 | -0.054233 | -6.79 | no |
| p1_pairrank_top_n_step6750 | 0.860425 | +0.061649 | +7.72 | no |

### teams_flat_xiang_xiang2_feng_dev (n_missed=74, P8A=0.7978)

| Ckpt | mean_pair_gap_prob | lift_abs | lift_rel_pct | passes 30%% bar |
|---|---:|---:|---:|:---:|
| p1_bundle_periodic_step500 | 0.893093 | +0.095314 | +11.95 | no |
| p1_bundle_top_n_step3750 | 0.893966 | +0.096187 | +12.06 | no |
| p1_bundle_top_n_step4000 | 0.892468 | +0.094689 | +11.87 | no |
| p1_pairrank_periodic_step500 | 0.850404 | +0.052625 | +6.60 | no |
| p1_pairrank_top_n_step6000 | 0.874615 | +0.076836 | +9.63 | no |
| p1_pairrank_top_n_step6750 | 0.818233 | +0.020454 | +2.56 | no |

### visomaster_enhanced_macro_dev (n_missed=354, P8A=0.6271)

| Ckpt | mean_pair_gap_prob | lift_abs | lift_rel_pct | passes 30%% bar |
|---|---:|---:|---:|:---:|
| p1_bundle_periodic_step500 | 0.585621 | -0.041474 | -6.61 | no |
| p1_bundle_top_n_step3750 | 0.688015 | +0.060920 | +9.71 | no |
| p1_bundle_top_n_step4000 | 0.700089 | +0.072994 | +11.64 | no |
| p1_pairrank_periodic_step500 | 0.694681 | +0.067587 | +10.78 | no |
| p1_pairrank_top_n_step6000 | 0.710347 | +0.083253 | +13.28 | no |
| p1_pairrank_top_n_step6750 | 0.723664 | +0.096569 | +15.40 | no |

## Pass count per ckpt at the 30%-relative bar

Per `f2_pair_rank/phase_a_evaluable_pass_summary.csv`.

| Ckpt | n suites passing 30%% bar | n evaluable suites | passes ≥2-of-N |
|---|---:|---:|:---:|
| p1_bundle_periodic_step500 | 0 | 5 | no |
| p1_bundle_top_n_step3750 | 0 | 5 | no |
| p1_bundle_top_n_step4000 | 0 | 5 | no |
| p1_pairrank_periodic_step500 | 0 | 5 | no |
| p1_pairrank_top_n_step6000 | 0 | 5 | no |
| p1_pairrank_top_n_step6750 | 0 | 5 | no |

## Direct observations

1. Across all 5 fake suites × 6 P1 ckpts (30 cells), the highest `lift_rel_pct` observed is **+20.59%** (`p1_bundle_periodic_step500` on `teams_fake_all_lockbox`). No cell crosses +30%.
2. 24 of 30 cells have `lift_rel_pct > 0` (positive lift relative to P8A); 6 of 30 cells have `lift_rel_pct < 0` (negative lift). Negative lifts: `p1_bundle_periodic_step500` on 3 of 5 suites (`deeplive_enhanced_dev`, `teams_fake_all_dev`, `visomaster_enhanced_macro_dev`); `p1_bundle_top_n_step3750` and `p1_bundle_top_n_step4000` on `teams_fake_all_lockbox`; `p1_pairrank_top_n_step6000` on `teams_fake_all_lockbox`.
3. Per-ckpt average `lift_rel_pct` across the 5 suites: `p1_bundle_periodic_step500` +2.55%, `p1_bundle_top_n_step3750` +6.76%, `p1_bundle_top_n_step4000` +8.51%, `p1_pairrank_periodic_step500` +10.55%, `p1_pairrank_top_n_step6000` +8.74%, `p1_pairrank_top_n_step6750` +10.80%.
4. The 30%-relative bar is structurally above the metric ceiling on 3 of 5 suites at P8A baseline: `deeplive_enhanced_dev` ceiling +28.15%, `teams_fake_all_lockbox` ceiling +25.19%, `teams_flat_xiang_xiang2_feng_dev` ceiling +25.35%. On those 3 suites the bar is unreachable for any ckpt regardless of its scores.
5. The 2 suites where the bar is reachable in principle (`teams_fake_all_dev` ceiling +40.32%, `visomaster_enhanced_macro_dev` ceiling +59.46%) had max observed lift +12.57% and +15.40% respectively; both below the bar.
6. P8A-missed fake counts per suite range from 74 (`teams_flat_xiang_xiang2_feng_dev`) to 740 (`teams_fake_all_dev`); summed across 5 suites = 1567 missed fakes, all included in the pairing CSV.
7. Real-pool size used for every per-frame metric is 6032 unique frame_paths.
8. 0 of 6 P1 ckpts meet the ≥2-of-N pass criterion at the 30%-relative bar across the 5 evaluable fake suites.
9. The pairrank arm (3 ckpts) has higher mean lift than the bundle arm (3 ckpts): mean over (suite × pairrank ckpts) = +10.03% vs mean over (suite × bundle ckpts) = +5.94%; difference = +4.09 percentage points.
10. The `e2b_top_n_step3200` ckpt was not included in this run (per task scope: "P1 ckpts vs P8A baseline").

## Caveats

1. **Suite axis is not the same as the yaml's training-lane axis.** The 5 fake suites here are eval-substrate scopes determined by the Phase A scorecard, not training-loader paired lanes. Specifically: `df40`, `visomaster_v1_base`, and `visomaster_teams_enhanced` (3 of the yaml's 6 training lanes) have no Phase A frame coverage and are not represented at all. `deeplive` (training lane 2) is partially proxied by `deeplive_enhanced_dev`, but the eval substrate's deeplive scope is `deeplive_enhanced_*` not `deeplive` (clean). `visomaster_enhanced` (training lane 4) is partially proxied by `visomaster_enhanced_macro_dev`. `deeplive_teams` (training lane 6) is partially represented within `teams_fake_all_dev` / `teams_fake_all_lockbox`.
2. **The metric ceiling is suite-dependent.** Three of five suites (`deeplive_enhanced_dev`, `teams_fake_all_lockbox`, `teams_flat_xiang_xiang2_feng_dev`) have P8A baselines high enough that +30% relative is mathematically unreachable on those suites at any ckpt. This is a property of the metric formulation, not of the ckpts.
3. **"Previously-missed" filter uses a fixed P8A frame_prob < 0.5 threshold.** No alternative thresholds (e.g., < 0.3 or < 0.7) were tried in this run. A lower threshold would tighten the missed set toward more confident-wrong fakes; a higher threshold would relax it.
4. **The pair-gap probability uses the entire 6032-frame real pool per fake.** The script computes `frac_real_below_fake = mean over all reals r of 1[fake_score > real_score]` strictly (ties counted as not-greater). This differs from a sampled-pair construction; it is exact given the union-deduplicated real pool.
5. **Real-pool union deduplicates by `frame_path`.** Frames that appear in multiple real suites (e.g., a frame in both `teams_real_all_dev` and `teams_real_lighting_extreme_dev`) contribute only once to the metric. The 6032 figure reflects that.
6. **`teams_capture_*_dev` and `teams_capture_*_s*_dev` per-subject fake suites are excluded.** They are subsets of `teams_fake_all_dev` (verified: same `frame_path` domain). Including them would inflate the suite count without adding distinct evidence.
7. **No statistical significance test is reported.** The lift values are point estimates per ckpt × suite cell. Per-frame variability and confidence intervals around `mean_pair_gap_prob` were not computed in this run.
8. **`e2b_top_n_step3200` was not evaluated here** despite being present in `phase_a/`. Adding it would test the lift of a non-P1 ckpt; out of scope of this task.

## Output files

- `f2_pair_rank/run_f2_phase_a_evaluable.py`
- `f2_pair_rank/phase_a_evaluable_pairing.csv` — 10,969 rows of long-form pairing data (1567 P8A-missed fake_paths × 7 ckpts).
- `f2_pair_rank/phase_a_evaluable_per_suite_per_ckpt_lift.csv` — 5 suites × 7 ckpts = 35 rows.
- `f2_pair_rank/phase_a_evaluable_pass_summary.csv` — 6 rows (one per P1 ckpt).
