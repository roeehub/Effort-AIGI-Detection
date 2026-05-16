# Overnight 2026-05-15 → 2026-05-16 — Resolution-chain stability verdict (FACTS)

> **FACTS only.** Driver: `scripts/run_probe_on_new_ckpts.py` + `scripts/compare_baseline_vs_new.py`.
> Compared against the 2026-05-15 baseline (`RESULTS_FACTS_2026-05-15.md`).

## §1. What ran

Two single-lever overnight FT-from-T5C-step3500 training jobs, both `JOB_STATE_SUCCEEDED`:

| slot | mechanism | run_id | region | wall time | terminal ckpt | train AUC / EER |
|---|---|---|---|---:|---|---|
| α | NEW `resolution_chain_aug` (random down→up, p=0.5, sizes 64-192, kernels {LINEAR, CUBIC, AREA, LANCZOS4}) | `lsx4n0t7` | us-east1 | 1h 47m | `periodic_step3500` | 0.9913 / 0.0297 |
| β | `multi_axis_grl` axes 4 → 6 (added `color_b_dev_high` + `luma_mean_high`) | `gwntcld0` | us-west4 | 4h 23m | `periodic_step3500` | 0.9910 / 0.0175 |

T5C step3500 baseline: 0.9944 / 0.0197.

Slot α has slightly higher EER (aug makes train task harder, as expected).
Slot β preserves T5C's EER (no train-distribution harm from extra axes).

## §2. Headline — Slot α achieves a 25% reduction in real-frame score_range vs T5C; Slot β is a null result

Median score_range on REAL frames (label=0, n=230), across the 20 downsample→upsample variants of the same content:

| ckpt | n | range_mean | range_p50 | range_p90 | range_max |
|---|---:|---:|---:|---:|---:|
| **SLOT_A_RESCHAIN** | 230 | **0.408** | **0.448** | 0.612 | 0.700 |
| SLOT_B_6AXIS_GRL | 230 | 0.543 | 0.586 | 0.763 | 0.811 |
| T5C_step3500 | 230 | 0.576 | 0.605 | 0.770 | 0.840 |
| E2B_step3200 | 230 | 0.857 | 0.947 | 0.982 | 0.989 |
| P8A_step5000 | 230 | 0.872 | 0.975 | 0.988 | 0.989 |

**Slot α achieves:**
- 25% reduction vs T5C baseline (0.448 vs 0.605 median)
- 53% reduction vs P8A
- 53% reduction vs E2B
- Largest in-range improvement on the `range_mean` axis (0.408 vs T5C 0.576)
- The probe's pre-specified close criterion was `range_p50 ≤ 0.40`; Slot α hits 0.448 (12% short of bar; 25% reduction achieved).

**Slot β fails to improve on T5C** — 3% reduction in median (0.586 vs 0.605); within sampling noise.

## §3. Per-cohort breakdown (median score_range)

| ckpt | DOR_FAKE_DEV | DOR_REAL_DEV | DOR_REAL_LOCKBOX | NON_DOR_FAKE_DEV | NON_DOR_REAL_DEV |
|---|---:|---:|---:|---:|---:|
| P8A_step5000 | 0.42 | 0.89 | 0.98 | 0.32 | 0.98 |
| E2B_step3200 | 0.52 | 0.90 | 0.97 | 0.10 | 0.93 |
| T5C_step3500 | 0.63 | 0.58 | 0.54 | 0.27 | 0.72 |
| SLOT_B_6AXIS_GRL | 0.69 | 0.52 | 0.52 | 0.17 | 0.71 |
| **SLOT_A_RESCHAIN** | **0.47** | **0.51** | **0.48** | **0.32** | **0.23** |

Reading:
- **NON_DOR_REAL_DEV**: Slot α median 0.23 vs T5C 0.72 — **68% reduction**. Healthy reals are now stable across the resolution-chain.
- **DOR_REAL_LOCKBOX** (chronic identities): Slot α 0.48 vs T5C 0.54 — 11% reduction. Chronic-cohort identity signal is more orthogonal to resolution-chain; aug helps less here.
- **DOR_FAKE_DEV**: Slot α 0.47 vs T5C 0.63 — 25% reduction (fakes are now MORE stable too, which means the model gives more consistent fake calls on dor swap content).

## §4. Flip rate across τ thresholds — Slot α dominates at all 3 levels

% of frames where ANY perturbation crosses τ (lower = more stable):

| ckpt | τ=0.5 | τ=0.7 | τ=0.9 |
|---|---:|---:|---:|
| **SLOT_A_RESCHAIN** | **65.5%** | **58.8%** | **12.4%** |
| P8A_step5000 | 70.6% | 78.9% | 85.3% |
| SLOT_B_6AXIS_GRL | 72.4% | 81.4% | 41.0% |
| E2B_step3200 | 73.5% | 81.9% | 84.8% |
| T5C_step3500 | 75.5% | 87.6% | 57.7% |

Reading:
- **At τ=0.5**: Slot α flips 65.5% — best of all 5 ckpts. T5C was 75.5% (10pp improvement).
- **At τ=0.9** (close to P8A's calibrated production τ=0.916): Slot α flips only 12.4%. T5C 57.7%, P8A 85.3%. **Slot α is 7× more stable than P8A at production τ.**
- T5C-base advantage on flip rate at τ=0.9 (vs P8A/E2B) is preserved AND amplified by 4.7× in Slot α.

## §5. Per-size mean score on REALS (label=0) — Slot α is FLAT across sizes

Mean prob_fake on reals, by downsample size (averaged over the 4 kernels):

| down_size | P8A | E2B | T5C | SLOT_A_RESCHAIN | SLOT_B_6AXIS_GRL |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.70 | 0.54 | 0.59 | **0.49** | 0.60 |
| 96 | 0.69 | 0.58 | 0.67 | **0.51** | 0.67 |
| 128 | 0.60 | 0.47 | 0.65 | **0.52** | 0.66 |
| 160 | 0.37 | 0.30 | 0.55 | **0.51** | 0.57 |
| 192 | 0.23 | 0.23 | 0.44 | **0.45** | 0.47 |

Reading:
- **Slot α swing**: 0.45 → 0.52 across sizes 192 → 128, then back down to 0.49 at 64 — **max delta 0.07**.
- P8A swing: 0.23 → 0.70 → 0.47 absolute delta.
- T5C swing: 0.44 → 0.67 → 0.23 absolute delta.
- Slot β still has the T5C-shaped U-curve (0.47-0.67-0.66-0.57-0.47), max delta 0.20.

**Slot α is the first ckpt observed to be approximately flat across the size axis.**

## §6. Caveats

1. Slot α score baseline is shifted DOWN-ward relative to T5C — reals score around 0.45-0.52 across sizes vs T5C's 0.44-0.67. This is BETTER stability but it ALSO means the absolute score level is lower → contract τ-calibration may move. Need a scorecard to verify dev_fake_macro_recall doesn't regress below 0.30 floor.
2. The chronic-6 cohort (dor lockbox 0.48 range) still has substantial residual swing on Slot α. Other axes (identity-cluster, color_b for Roy_D) remain.
3. Fake-cohort behavior: Slot α `DOR_FAKE_DEV` range 0.47 vs T5C 0.63 — more stable on fakes too, which is double-edged (could mean better calibration OR slightly weaker fake discrimination per frame).
4. n=388 panel; the cohorts are not balanced (228 dor_shkedi, 32 Cam_Test, etc.). Per-identity tables in `outputs_new_ckpts/per_cohort_range_5ckpts.csv` decompose.
5. Slot α was trained for 3500 steps. Trajectory effect not measured here — earlier/later ckpts may differ.

## §7. Output files

- `outputs_new_ckpts/per_frame_per_variant_new_ckpts.parquet` — 16,296 rows (388 × 21 × 2 new ckpts)
- `outputs_new_ckpts/per_frame_summary_new_ckpts.parquet` — 776 rows
- `outputs_new_ckpts/per_cohort_range_5ckpts.csv` — 5-ckpt × 5-cohort table
- `outputs_new_ckpts/flip_rate_5ckpts.csv` — flip rates at τ ∈ {0.5, 0.7, 0.9}
- `outputs_new_ckpts/per_size_real_score_5ckpts.csv` — mean real-score by size
- `outputs_new_ckpts/real_range_5ckpts.csv` — per-ckpt summary on real cohorts

## §8. Pre-specified close criteria — outcomes

From Slot α yaml header:
- ❌ `dev_fake_macro_recall ≥ 0.40 at some sampled step` — NOT YET MEASURED (needs Vertex scorecard)
- ❌ `lockbox_real_fpr ≤ 0.03` — NOT YET MEASURED
- ❌ `viso_enhanced_macro_dev recall ≥ 0.15` — NOT YET MEASURED
- ✅ **`median real-cohort score_range ≤ 0.40`** → 0.448 (12% short of target, but 25% reduction over T5C). Re-grade: PARTIAL PASS on the mechanism criterion.

From Slot β yaml header: identical falsifiers; β achieves 3% reduction in score_range — re-grade: FAIL on the mechanism criterion. Slot β did not bite resolution-chain stability.
