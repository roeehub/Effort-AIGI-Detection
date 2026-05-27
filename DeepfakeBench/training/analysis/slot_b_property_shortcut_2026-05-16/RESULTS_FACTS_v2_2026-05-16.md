# RESULTS_FACTS_v2_2026-05-16 — PNG-only reframe + dev cohort cross-validation

> **FACTS only.** Supersedes `RESULTS_FACTS_2026-05-16.md` §3-7 conclusions
> after user input that production deployment is PNG-only (JPG cohorts are
> legacy data, still useful as model-behavior probes but not as production
> readouts).
>
> Origin: post-Slot-β cross-validation on the dev cohort 2026-05-16.

## §1. PNG-only lockbox FPR re-eval

Same lockbox real cohort, restricted to `.png` extension:

| ckpt | full FPR | **PNG FPR (prod-relevant)** | JPG FPR (legacy) |
|---|---:|---:|---:|
| P8A | 1.90% (27/1418) | **0.20%** (2/1004) | 6.04% (25/414) |
| T5C | 3.10% (44/1418) | **2.49%** (25/1004) | 4.59% (19/414) |
| Slot β | 8.74% (124/1418) | **7.47%** (75/1004) | 11.84% (49/414) |

P8A's lockbox advantage was 92% on legacy JPG data (25/27 JPG, 2/27 PNG). On production-relevant PNG: P8A 0.20% vs Slot β 7.47% — Slot β still over-fires ~37× more.

## §2. Lockbox PNG identity composition

| identity | n PNG | Slot β over | rate |
|---|---:|---:|---:|
| dor_shkedi | 895 | 75 | **8.4%** |
| real_dor | 109 | 0 | 0% |

All lockbox PNG over-fires concentrate on `dor_shkedi` (same person as `real_dor`, 0%). Within dor_shkedi.png, the within-cohort contrast on Slot β over vs non-over (n=75 vs 820) is more modest than the JPG-mixed contrast:

| property | over_med | non-over_med | Cohen's d |
|---|---:|---:|---:|
| edge_density | 0.272 | 0.280 | **−0.49** |
| lab_b_dev | 13.62 | 13.82 | −0.36 |
| skin_frac_hsv | 0.59 | 0.60 | −0.36 |
| luma_std | 57.02 | 56.92 | +0.35 |
| lab_a_dev | 6.15 | 6.21 | −0.28 |

Effect sizes within PNG are 0.2-0.5 Cohen's d; the dor_shkedi.png signal is weaker than the JPG/PNG split suggested.

## §3. Dev cohort cross-validation — Roy_D is the production-relevant problem

`teams_real_all_dev` has 4564 frames, of which only **194 are PNG**: 130 Roy_D, 29 ilan, 35 orel. The rest is legacy JPG.

### §3.1. Per-identity over-fire rates on PNG dev

| identity | n | P8A rate | T5C rate | Slot β rate | P8A p50 | T5C p50 | Slot β p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Roy_D** | 130 | **29.2%** | **86.2%** | **79.2%** | 0.44 | **0.91** | **0.89** |
| ilan | 29 | 0% | 0% | 0% | 0.005 | 0.07 | 0.09 |
| orel | 35 | 0% | 0% | 0% | 0.007 | 0.14 | 0.10 |

**Roy_D is catastrophically over-fired on by ALL three ckpts.** P8A scores median 0.44 with p95=0.99 (long-tail above τ). T5C and Slot β over-fire on 80%+ of frames — the entire identity is essentially flagged as fake. ilan and orel are clean on all 3 ckpts.

### §3.2. Roy_D vs ilan/orel property contrast

| property | Roy_D med | ilan+orel med | Cohen's d | MW p |
|---|---:|---:|---:|---:|
| lab_a_std | 11.52 | 7.07 | **+5.25** | 1e-29 |
| lab_a_dev | 17.97 | 11.49 | **+4.20** | 2e-29 |
| file_size | 111,664 | 61,019 | +3.36 | 2e-27 |
| min_dim | 277.5 | 194.5 | +2.93 | 4e-28 |
| lab_b_std | 8.07 | 4.82 | +2.55 | 2e-26 |
| lab_b_dev | 14.91 | 10.85 | +1.09 | 4e-08 |
| skin_frac_hsv | 0.902 | 0.766 | +1.04 | 4e-05 |
| sharpness | 67.4 | 133.1 | **−0.73** | 3e-11 |
| luma_std | 51.3 | 62.8 | −0.64 | 1e-04 |

Roy_D crops are: square (278×278 always), large, heavily color-saturated on the a-axis (red-green skew), high skin coverage, blurry (sharpness 67 vs 133+ for ilan/orel), with low luma variation.

### §3.3. Roy_D vs dor_shkedi.png — same Slot β failure, OPPOSITE properties

Both Roy_D (dev PNG, 79% over-fire) and dor_shkedi.png (lockbox PNG, 8% over-fire) are PNG square crops where Slot β over-fires. Their property profiles are **opposite on every axis**:

| property | Roy_D med | dor_shkedi.png med | Cohen's d |
|---|---:|---:|---:|
| lab_a_dev | 17.97 | 6.20 | +8.33 |
| skin_frac_hsv | 0.902 | 0.598 | +6.78 |
| luma_mean | 108 | 139 | **−5.58** |
| edge_density | 0.202 | 0.280 | −3.57 |
| sharpness | 67 | 201 | **−1.33** |
| lab_b_dev | 14.91 | 13.79 | +0.85 |
| luma_std | 51.3 | 56.9 | −1.21 |

Roy_D over-firing frames are dark, blurry, color-saturated, high-skin. dor_shkedi.png over-firing frames are bright, sharper, low-color, lower-skin. Both over-fire on the same model. **There is no single property axis that explains the over-firing pattern.**

### §3.4. Roy_D within-identity over vs non-over

Within Roy_D (n=130), Slot β over-fires on 103, doesn't on 27. The within-cohort contrast:

| property | over_med | non-over_med | Cohen's d | MW p |
|---|---:|---:|---:|---:|
| luma_std | 50.06 | 57.30 | **−1.50** | 2.3e-08 |
| luma_mean | 107.33 | 112.77 | **−1.20** | 5.9e-07 |
| edge_density | 0.210 | 0.179 | **+1.08** | 1.2e-05 |
| lab_b_dev | 15.73 | 13.75 | +0.66 | 5.9e-04 |
| min_dim | 274 | 291 | −0.48 | 2.9e-03 |

Within Roy_D, over-firing frames are: dimmer (lower luma_mean and luma_std), MORE edges, higher b-axis color dev, smaller crops. This is **opposite to the dor_shkedi.png within-cohort contrast**, which was: less edges, lower color dev.

## §4. Summary across cohorts

Slot β over-firing PNG cohorts have **opposing property profiles**:

| cohort | n | over-fire rate | luma_mean | edge_density | lab_a_dev | sharpness | skin_frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| Roy_D dev | 130 | 79.2% | 108 ↓ | 0.20 ↓ | 18.0 ↑ | 67 ↓ | 0.90 ↑ |
| dor_shkedi lockbox | 895 | 8.4% | 139 ↑ | 0.28 ↓ | 6.2 ↓ | 201 ↑ | 0.60 ↓ |
| real_dor lockbox | 109 | 0% | 133 ↑ | 0.30 | 10.5 | 353 ↑↑ | 0.67 |
| ilan dev | 29 | 0% | 101 | 0.26 | 11.8 | 163 | 0.63 |
| orel dev | 35 | 0% | 112 | 0.07 | 10.3 | 108 | 0.91 |

There is no single property direction that distinguishes over-firing PNG cohorts from clean PNG cohorts. The shortcut, if there is one, is multi-modal.

## §5. Outputs

- `outputs/dev_png_frame_properties.csv` — per-frame properties for 194 dev PNG frames
- `outputs/dev_png_frames_with_props_and_scores.csv` — merged with scores
- `outputs/dev_png_per_identity_medians.csv` — §3.1
- `outputs/dev_png_roy_d_overfire_contrast.csv` — §3.4
- `outputs/dev_png_roy_d_vs_clean_contrast.csv` — §3.2
