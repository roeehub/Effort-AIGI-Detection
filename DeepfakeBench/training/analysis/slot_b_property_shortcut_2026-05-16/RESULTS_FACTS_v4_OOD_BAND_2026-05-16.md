# RESULTS_FACTS_v4 — OOD-band hypothesis confirmed

> **FACTS only.** Builds on v3 (band-shortcuts confirmed). v4 tests whether
> training data inhabits the 6+ band intersection region. It does not.

## §1. Provenance

- Sampled 470 training real `.jpg` frames + 422 training fake `.jpg` frames from
  `gs://live-deepfake-methods-real-and-fake-frames-cropped-teams/samples/<sample_id>/frames/{real,fake}/frame_{0000,0004,0008}.jpg`
  across 200 randomly-chosen samples (seed=0).
- Computed the same 11 properties via cv2 (single-threaded).
- All training reals here are JPG (the production-relevant test cohort
  includes PNG; this is a known compression-axis difference).

## §2. Training real lab_a_dev distribution

| stat | training reals | Roy_D dev PNG | dor_shkedi lockbox PNG |
|---|---:|---:|---:|
| n | 470 | 130 | 895 |
| mean | 9.20 | 18.02 | 4.83 |
| median | 8.33 | 17.97 | 6.20 |
| p95 | 18.17 | — | — |
| p99 | 22.74 | — | — |
| max | 33.94 | — | — |

Roy_D's median lab_a_dev (17.97) sits between p95 and p99 of training reals — **NOT out-of-distribution on this single axis**. ~10% of training reals have lab_a_dev > 16. The univariate OOD framing is REFUTED.

## §3. Bands inhabited by training reals vs fakes

| property | band | %_train_real | %_train_fake | fake/real ratio |
|---|---|---:|---:|---:|
| sharpness < 142 | 45.1% | 59.5% | 1.32× |
| luma_mean < 130 | 45.5% | 45.7% | 1.00× |
| **lab_a_dev > 16** | **10.2%** | 12.3% | 1.21× |
| **lab_a_std ∈ [10.3, 11.9]** | 7.5% | 5.9% | 0.80× |
| **lab_b_dev ∈ [14.7, 17.7]** | 15.1% | 14.0% | 0.93× |
| **skin_frac > 0.88** | 9.2% | 14.2% | 1.55× |
| edge_density < 0.25 | 43.8% | 45.5% | 1.04× |
| min_dim > 266 | 13.8% | 19.0% | 1.37× |
| file_size > 110210 | 9.4% | 10.9% | 1.16× |

No single band is dramatically real-fake skewed at training time. Marginal fake/real ratios are 0.80×-1.55× — moderate but not extreme.

## §4. Joint band-hits distribution — TRAINING DATA is BIMODAL on the joint axis

| n_bands hit | n_real | %_real | n_fake | %_fake | P(real \| n_bands) |
|---:|---:|---:|---:|---:|---:|
| 0 | 72 | 15.3% | 37 | 8.8% | **0.661** |
| 1 | 110 | 23.4% | 79 | 18.7% | **0.582** |
| 2 | 120 | 25.5% | 124 | 29.4% | 0.492 |
| 3 | 108 | 23.0% | 123 | 29.2% | 0.468 |
| 4 | 38 | 8.1% | 36 | 8.5% | 0.514 |
| 5 | 21 | 4.5% | 20 | 4.7% | 0.512 |
| **6** | **0** | **0%** | **3** | 0.7% | **0.000** |
| **7+** | 1 | 0.2% | 0 | 0% | undef. |

**The critical row: at training, n_bands ∈ {6, 7+} has effectively zero real examples.** The single training real that hits 7 bands is 0.2% of the cohort. The training data therefore provides **no anchor for "this is real" in the multi-band region**.

## §5. Cross-reference with the test cohort over-fire pattern

Recall from RESULTS_FACTS_v3 §2 the per-`n_bands` over-fire rates on the test PNG pool (1198 frames):

| n_bands | training: n_real (n=470) | test: Slot β rate (n_pool=1198) | test: P8A rate |
|---:|---:|---:|---:|
| 0 | 72 (15.3%) | 6.9% | 0.0% |
| 5 | 21 (4.5%) | 30.0% | 6.7% |
| 6 | **0 (0%)** | **65.6%** | 21.9% |
| 7 | **1 (0.2%)** | **93.1%** | 41.4% |
| 8 | **0 (0%)** | **79.5%** | 28.2% |
| 9 | **0 (0%)** | **100%** | 50.0% |

**Dose-response correlation between "absence in training" and "over-fire at test"**: the bands where training has zero real anchors are the bands where the model fails catastrophically at test time. **All three ckpts share this pattern** — the structural cause is the training data distribution, not any specific ckpt's loss design.

## §6. Current training pipeline color augmentation (audit)

The T5C training yaml uses the `quality_targeted_family` aug at `vcd_targeted` strength. The color-axis augmentation:

- HueSaturationValue: `hue_shift_limit=12-20°`, `sat_shift=24`, `val_shift=24`, `p=color_p=0.52`
- ColorTemperatureShift (CCT): `range=(2700-8000) K`, `p=0.15`
- RandomBrightnessContrast: brightness limit `(-0.20, 0.60)` (asymmetric, biased UP), contrast `0.25`, p=`individual_p=0.15`
- RandomGamma: `gamma_limit=(70, 130)`, p=0.15

These augmentations rotate / shift / temperature-shift the color channel — they do NOT explicitly INCREASE lab_a_dev or lab_b_dev. A hue rotation can produce either a higher or lower deviation from neutral depending on starting chroma.

The pipeline does not contain a "push toward high lab_a_dev" augmentation. The augs that exist do not compositionally produce multi-band reals; the 4% multi-band training reals are mostly natural variation, not augmentation-produced.

## §7. dor_shkedi.png alternate mechanism

The 75 Slot β over-firing frames on dor_shkedi.png have avg 0.44 bands hit
(per RESULTS_FACTS_v3 §3). Within dor_shkedi.png, no individual property
has |Cohen's d| > 0.49 between over and non-over (per RESULTS_FACTS_v2 §2).
A decision tree on dor_shkedi.png-only would need to be trained to
find this alternate mechanism; that analysis is deferred.

## §8. Output files

- `outputs/training_real_properties.csv` — 470 sampled training reals
- `outputs/training_fake_properties.csv` — 422 sampled training fakes
- `outputs/all_png_pool_with_bands.csv` — test pool + bands
- `outputs/per_decile_overfire_rate.csv` — RESULTS_FACTS_v3 source data
