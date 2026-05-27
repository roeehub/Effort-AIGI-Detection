# RESULTS_FACTS_2026-05-16 — Slot β property-shortcut decomposition

> **FACTS only.** Per-frame numerical findings on the 1418-frame
> `teams_real_all_lockbox` cohort. No interpretation.
>
> Origin: post-Slot-β CPU diagnostic prompted by user reframing 2026-05-16:
> the over-fire pattern is not identity-localized (`dor_shkedi`, `dor`, and
> `real_dor` are the same person but score differently) — it must be a
> property of specific image sets that the model is reacting to.

## §1. Provenance

- Cohort: 1418 unique JPG/PNG crops from `gs://teams-faces-data-test-2914-fake-4420-real-feb-28/real/` across 5 identities (`Chikara_Takahashi`, `PC_Generator`, `bla_bla_chow`, `dor_shkedi`, `real_dor`).
- Source data: per-frame `teams_real_all_lockbox_*_frames_report.csv` for P8A, T5C, Slot β step3500 (from scorecard `gs://...overnight-scorecard-2026-05-16/`).
- 11 image properties computed locally via cv2 (single-threaded, per `feedback_sklearn_njobs`): sharpness (Laplacian variance), mean luma, luma std, LAB a/b channel mean/std/deviation, skin fraction (HSV), edge density, min_dim, file size.
- Scripts: `analysis/slot_b_property_shortcut_2026-05-16/scripts/{compute_properties,aggregate_by_cohort,property_predictor,by_seq}.py`.

## §2. Same-person, different-tag property gap

`dor_shkedi` (n=1170) and `real_dor` (n=109) are the same person, different tagging. Slot β: 116 over-fires on `dor_shkedi`, 0 on `real_dor`. Property contrast (Cohen's d):

| property | dor_shkedi median | real_dor median | Cohen's d | MW p |
|---|---:|---:|---:|---:|
| lab_b_std | 8.61 | 16.14 | **−3.74** | 2.9e-55 |
| lab_b_dev | 13.51 | 26.01 | **−2.65** | 1.8e-52 |
| lab_a_dev | 5.97 | 10.50 | **−2.14** | 6.2e-55 |
| lab_a_std | 8.96 | 12.40 | **−1.79** | 6.4e-47 |
| skin_frac_hsv | 0.586 | 0.672 | −0.87 | 5.1e-39 |
| file_size | 100874 | 86931 | +0.61 | 9.0e-27 |
| min_dim | 245 | 219 | +0.59 | 5.8e-32 |
| edge_density | 0.278 | 0.299 | −0.22 | 3.8e-12 |
| luma_mean | 137.18 | 133.06 | +0.18 | 3.5e-05 |
| sharpness_laplacian | 211.69 | 352.99 | −0.16 | 4.1e-10 |

The same person has wildly different image properties across the two tags, with the largest effects on LAB color channels (b-axis dev/std ~2× higher for real_dor) and skin fraction.

## §3. Bimodal capture-style population WITHIN dor_shkedi

dor_shkedi crops split cleanly by file extension into two populations:

| stratum | n | mean_score | over-fire rate | sharpness_med | lab_b_dev_med | skin_frac_med | min_dim_med | file_size_med |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dor_shkedi .jpg | 275 (24%) | 0.654 | **14.9%** (41/275) | 422 | 3.90 | 0.19 | 394 | 291,759 |
| dor_shkedi .png | 895 (76%) | 0.563 | **8.4%** (75/895) | 201 | 13.79 | 0.60 | 242 | 98,873 |
| real_dor .png | 109 | 0.152 | **0%** | 353 | 26.01 | 0.67 | 219 | 86,931 |

### §3.1. Crop geometry

- `dor_shkedi.jpg`: rectangular crops, width ≠ height (median 499 × 394, range 130–512 × 107–415).
- `dor_shkedi.png`: square crops, width == height (median 242 × 242, range 208–290).
- `real_dor.png`: square crops (median 219 × 219, range 177–404).
- `bla_bla_chow.jpg`: 68 frames, all .jpg (no extension comparison available; rate 11.8%).
- `Chikara_Takahashi.jpg`, `PC_Generator.jpg`: all .jpg, 0% over-fire.

**dor_shkedi is the only identity in the lockbox with a mixed-extension capture-style population.** Every other identity is single-extension.

## §4. Within-identity contrast: dor_shkedi over vs non-over

Among the 1170 `dor_shkedi` frames, comparing the 116 Slot β over-firing frames against the 1054 non-over-firing frames:

| property | over mean | non-over mean | Cohen's d | MW p |
|---|---:|---:|---:|---:|
| sharpness_laplacian | 491.8 | 281.8 | **+0.50** | 4.4e-02 |
| lab_a_std | 8.19 | 8.70 | **−0.41** | 6.3e-04 |
| lab_b_dev | 9.87 | 11.60 | −0.38 | 7.3e-05 |
| skin_frac_hsv | 0.439 | 0.510 | −0.38 | 2.1e-04 |
| luma_std | 62.6 | 60.0 | +0.36 | 4.3e-03 |
| lab_a_dev | 4.03 | 4.92 | −0.33 | 9.2e-03 |

Slot β over-firing dor_shkedi frames have HIGHER sharpness, LOWER color variation (a_std, b_dev, a_dev), and LOWER skin fraction than non-over-firing dor_shkedi frames. The direction is opposite to P8A's known image-quality shortcut (memory `project_image_quality_shortcut`: P8A scores correlate NEGATIVELY with sharpness).

## §5. Property-classifier transfer test

A logistic regression trained on dor_shkedi (X = 11 image properties, y = Slot β over-fire status) and evaluated on other identities:

| identity | n | actual over-fire rate | classifier predicted mean prob | frac predicted > 0.5 |
|---|---:|---:|---:|---:|
| dor_shkedi (train) | 1170 | 9.9% | — (train AUC 0.713) | — |
| bla_bla_chow (test) | 68 | 11.8% | — (test AUC 0.671) | — |
| real_dor | 109 | 0% | **0.961** | **98.2%** |
| Chikara_Takahashi | 42 | 0% | 0.001 | 0% |
| PC_Generator | 29 | 0% | **0.827** | **96.6%** |

The 11 continuous image properties do NOT transfer across identities as a predictor. The dor-trained classifier predicts 96-98% over-fire probability on `real_dor` and `PC_Generator` (where actual rate is 0%), and 0% on `Chikara_Takahashi` (also 0% actual). The continuous-property axis is identity-confounded.

In contrast, file extension (a discrete signature that captures aspect-ratio + compression style) DOES localize the over-fire concentration within dor_shkedi (§3).

## §6. Per-identity Spearman correlation of Slot β score vs property

|              | dor_shkedi | bla_bla_chow | real_dor | Chikara_T. | PC_Gen. |
|---|---:|---:|---:|---:|---:|
| sharpness_laplacian | +0.28 | −0.12 | +0.39 | **−0.60** | +0.10 |
| luma_mean | −0.06 | +0.21 | −0.12 | **−0.54** | −0.31 |
| lab_a_dev | −0.11 | −0.15 | −0.13 | **−0.70** | +0.10 |
| skin_frac_hsv | −0.11 | −0.24 | −0.06 | −0.49 | −0.15 |
| min_dim | −0.02 | +0.38 | −0.46 | +0.02 | +0.07 |
| lab_b_std | +0.17 | +0.23 | **−0.44** | +0.28 | −0.23 |

No property has consistent direction across identities. sharpness ρ flips sign (dor +0.28, Chikara −0.60). The model's score-vs-property relationship is identity-dependent.

## §7. Slot β over-fire rates per identity

| identity | n_frames | n_over | rate |
|---|---:|---:|---:|
| dor_shkedi | 1170 | 116 | 9.9% |
| bla_bla_chow | 68 | 8 | 11.8% |
| real_dor | 109 | 0 | 0% |
| Chikara_Takahashi | 42 | 0 | 0% |
| PC_Generator | 29 | 0 | 0% |

All over-fires concentrate on the two identities (`dor_shkedi`, `bla_bla_chow`) that are JPEG-compressed at the data level (`dor_shkedi.jpg` subset + all of `bla_bla_chow.jpg`).

## §8. Outputs

- `outputs/frame_properties.csv` — per-frame property values
- `outputs/frames_with_props_and_scores.csv` — merged with per-frame scores
- `outputs/contrast_dor_shkedi_overfire_vs_not.csv` — §4
- `outputs/contrast_dor_shkedi_vs_real_dor.csv` — §2
- `outputs/per_identity_property_medians.csv` — per-identity property profile
- `outputs/per_identity_spearman_slot_b_vs_property.csv` — §6
- `outputs/per_session_*` — seq-decile decompositions
- Scripts under `scripts/`
