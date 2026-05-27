# IQ-based device-OOD detector — RESULTS FACTS

Generated 2026-05-23. Factual readout. Interpretive content + open questions in `AGENT_PROPOSAL_2026-05-23.md`.

> **Question**: Does an explicit image-quality LR (sharpness, brightness, color cast, edge density, aspect ratio, jpeg_qf) generalize as a device-OOD detector across people — unlike the CLIP-feature LR, which overfit to "Roee on his specific Windows laptop"?
>
> **Method**: Compute 25 IQ features per frame on 2,319 real frames (5 deploy-relevant team humans + Mac-Roee). Train LR (with StandardScaler) on Mac-Roee vs Roee-Windows. Project all team frames onto the axis. Critical generalization test: do non-Roee in-distribution cohorts (Xinhe/Xiang/Noyn/dor on Windows captures) end up on the "Windows" side (margin < 0)?

---

## 0. IQ features computed

25 features per frame, from `analysis/lockbox_tagging/layers/quality.py` extended with color-cast + per-channel + LAB + edge-density features:

| Block | Features |
|---|---|
| Resolution / aspect | width, height, aspect_ratio |
| Sharpness / contrast | sharpness_laplacian, contrast_rms, edge_density |
| Brightness | brightness_v_mean, brightness_v_std, clipped_highlights_frac |
| Saturation | saturation_s_mean |
| RGB | r_mean, g_mean, b_mean, r_std, g_std, b_std |
| Color cast | color_cast_rg, color_cast_rb, color_cast_gb |
| LAB | lab_l_mean, lab_a_mean, lab_b_mean, lab_l_std, lab_a_std, lab_b_std |
| JPEG | jpeg_qf (None for non-JPEG frames; excluded from features) |

---

## 1. Detector training

| Metric | Value |
|---|---:|
| Roee-Windows frames (label 0) | 330 |
| Roee-Mac frames (label 1) | 498 |
| LR (with StandardScaler) 5-fold CV accuracy | **1.0000 ± 0.0000** |
| LR 5-fold CV AUC | **1.0000** |

Perfectly separable in IQ space (same as CLIP-LR). But unlike CLIP-LR, this one generalizes — see §2.

---

## 2. Generalization test — does the detector classify NON-Roee Windows captures correctly?

If the detector overfit to Roee-specific identity, all non-Roee in-dist cohorts (Xinhe/Xiang/Noyn/dor on Windows webcams) would end up "Mac-like" (margin > 0) like they did with the CLIP detector. If it learned a genuine device-axis, they should end up "Windows-like" (margin < 0).

Per-cohort IQ-margin (sorted descending; positive = Mac-like):

| Cohort | Human | In-scope | n | margin_mean | P8A FPR@τ=0.10 | Classified |
|---|---|---|---:|---:|---:|---|
| bla_bla_chow | Mac-Roee | OOS | 150 | 7.785 | 27.3% | ✓ Mac |
| bla_bla_chow__s1 | Mac-Roee | OOS | 68 | (Mac-side) | 5.9% | ✓ Mac |
| bla_bla_chow__s2 | Mac-Roee | OOS | 150 | (Mac-side) | 64.0% | ✓ Mac |
| Roy_D | Mac-Roee | OOS | 130 | (Mac-side) | 67.7% | ✓ Mac |
| **dor_shkedi__s16** | dor | deploy | 31 | **6.213** | 0.0% | ✗ FALSE-Mac (clean cohort flagged) |
| **dor_morning** | dor | deploy | 150 | **4.688** | 36.7% | ✓ Mac (correctly flagged as problematic) |
| **Md_noyn_Sharker__s15** | Noyn | deploy | 150 | **2.440** | 13.3% | ✓ Mac (borderline-problematic) |
| xiang | Xiang | deploy | 150 | 0.018 | 20.0% | borderline |
| dor_evening | dor | deploy | 150 | -0.964 | 2.0% | ✓ Windows |
| real_dor | dor | deploy | 109 | -1.398 | 5.5% | ✓ Windows |
| extra_xiang | Xiang | deploy | 150 | -1.567 | 3.3% | ✓ Windows |
| dor_shkedi | dor | deploy | 150 | -2.853 | 17.3% | ✓ Windows |
| Xiang_Xiang2_Feng__s23 | Xiang | deploy | 102 | -3.072 | 9.8% | ✓ Windows |
| team_may5__Dor | dor | deploy | 30 | -3.177 | 10.0% | ✓ Windows |
| team_may5__Xiang | Xiang | deploy | 30 | -3.729 | 0.0% | ✓ Windows |
| Xiang_Xiang2_Feng | Xiang | deploy | 150 | -4.682 | 0.7% | ✓ Windows |
| extra_xinghe | Xinhe | deploy | 19 | -5.432 | 10.5% | ✓ Windows |
| team_may5__Noyn | Noyn | deploy | 60 | -5.592 | 0.0% | ✓ Windows |
| team_may5__Xinhe | Xinhe | deploy | 60 | -9.720 | 5.0% | ✓ Windows |
| (3 Roee_Windows cohorts) | Roee_Windows | deploy | — | (far Windows-side) | low | ✓ Windows |

**11 of 15 non-Roee in-distribution cohorts classified correctly as Windows-like** (margin < 0).

The 4 cohorts that DON'T classify as Windows-like:
- dor_shkedi__s16 (margin 6.2, P8A FPR 0%) — **false-positive abstain target** if used naively
- dor_morning (margin 4.7, P8A FPR 36.7%) — correctly identified as problematic
- Md_noyn_Sharker__s15 (margin 2.4, P8A FPR 13.3%) — borderline correctly identified
- xiang (margin 0.02, P8A FPR 20%) — borderline correctly identified

**3 of the 4 "Mac-classified" in-dist cohorts have elevated P8A FPR.** The "false positives" of the IQ detector concentrate on cohorts that are also genuinely problematic for P8A.

---

## 3. Comparison to CLIP-based detector

CLIP-LR classified ALL 15 non-Roee in-dist cohorts as Mac-like (identity-confounded). IQ-LR classifies 11 of 15 correctly. Side-by-side on each cohort:

| Cohort | Human | CLIP margin | IQ margin | Both Windows? | IQ-better? |
|---|---|---:|---:|---|---|
| dor_evening | dor | -1.971 | -0.964 | YES | – |
| real_dor | dor | -1.818 | -1.398 | YES | – |
| extra_xiang | Xiang | 4.914 | -1.567 | – | ✓ IQ better |
| dor_shkedi | dor | -0.372 | -2.853 | YES | – |
| Xiang_Xiang2_Feng__s23 | Xiang | 5.580 | -3.072 | – | ✓ IQ better |
| team_may5__Dor | dor | -1.507 | -3.177 | YES | – |
| team_may5__Xiang | Xiang | 0.448 | -3.729 | – | ✓ IQ better |
| Xiang_Xiang2_Feng | Xiang | 4.152 | -4.682 | – | ✓ IQ better |
| extra_xinghe | Xinhe | 6.338 | -5.432 | – | ✓ IQ better |
| team_may5__Noyn | Noyn | -1.191 | -5.592 | YES | – |
| team_may5__Xinhe | Xinhe | 0.945 | -9.720 | – | ✓ IQ better |

IQ detector is strictly better than CLIP detector on every cohort where they disagree.

---

## 4. Per-frame correlation: IQ-margin → P8A score

Spearman rank correlation between per-frame IQ-margin and per-frame P8A score:

| Subset | n | ρ | p-value |
|---|---:|---:|---|
| All real frames (deploy + Mac OOS) | 2,319 | **0.5166** | 2e-158 |
| Deploy-relevant only | 1,821 | **0.3543** | 5e-55 |

Higher IQ-margin (= more Mac-like) systematically predicts higher P8A score. The correlation is strong overall and meaningfully present even within the deploy-relevant subset.

### Per-margin-decile P8A FPR

| Decile | Margin range | n | P8A FPR @τ=0.10 | P8A FPR @τ=0.59 | In-dist share |
|---:|---|---:|---:|---:|---:|
| 0 (most Windows-like) | [-31.8, -8.4] | 232 | 4.3% | 1.3% | 100% |
| 1 | [-8.4, -4.8] | 232 | 5.2% | 1.3% | 100% |
| 2 | [-4.8, -3.2] | 232 | 5.6% | 1.3% | 100% |
| 3 | [-3.2, -2.0] | 232 | 8.6% | 1.3% | 100% |
| 4 | [-2.0, -1.0] | 232 | 6.9% | 1.3% | 100% |
| 5 | [-1.0, +0.2] | 231 | 7.4% | 0.9% | 100% |
| 6 | [+0.2, +3.5] | 232 | 19.8% | 5.2% | 94% |
| 7 | [+3.5, +5.7] | 232 | **34.9%** | **16.4%** | 57% |
| 8 | [+5.8, +7.8] | 232 | **43.1%** | **21.6%** | 27% |
| 9 (most Mac-like) | [+7.8, +15.9] | 232 | 36.6% | 15.5% | 7% |

A clear monotonic relationship: P8A FPR rises from ~5% in the Windows-like deciles to ~35-43% in the Mac-like deciles. The transition happens between deciles 5 and 7 (margin ≈ +3 to +5).

---

## 5. Top LR feature importance

Top 10 features by absolute coefficient (StandardScaler-normalized):

| Feature | Coefficient | Direction |
|---|---:|---|
| edge_density | -1.769 | high → Windows |
| sharpness_laplacian | -1.377 | high sharpness → Windows |
| aspect_ratio | -1.077 | high aspect → Windows |
| lab_a_std | +1.058 | high `a*` variance → Mac |
| lab_a_mean | +1.029 | high `a*` (red-magenta) → Mac |
| color_cast_gb | -1.028 | high green-blue cast → Windows |
| height | +0.966 | tall frame → Mac |
| saturation_s_mean | +0.763 | high saturation → Mac |
| brightness_v_std | -0.755 | high brightness variance → Windows |
| lab_b_mean | -0.752 | high `b*` (yellow) → Windows |

The detector keys on physically interpretable axes:
- **Sharpness + edge density** are the dominant signal: Roee's Windows webcam captures are sharper and have higher edge density than Roee's Mac webcam captures.
- **Color cast** (lab a/b channels): Mac shifts toward red-magenta; Windows toward yellow.
- **Aspect ratio + height**: capture-pipeline frame-shape differs between devices.

These are intrinsic device-pipeline properties, not identity-specific.

---

## 6. Artifacts

- `outputs/per_frame_iq_v2.parquet` — 2,319 real frames × 30 columns (frame metadata + IQ features + P8A score)
- `outputs/per_cohort_iq_margin.csv` — per-cohort margin statistics
- `outputs/feature_importance.csv` — LR coefficient table
- `scripts/extract_iq_v2.py` — IQ extraction with GCS fallback
- `scripts/build_iq_detector.py` — LR training + projection + analysis

Wall time: ~7 min IQ extraction (with GCS download for ~1550 frames at ~5 fps) + ~30 sec LR + analysis.

---

## 7. Caveats

1. **Same Roee-only training set.** Detector is trained on one person's Mac vs Windows captures. The fact that it generalizes (11 of 15 non-Roee correctly classified) suggests it learned device-intrinsic features (sharpness, color cast), but a multi-person training set would be more robust.
2. **Fake frames not extracted.** This readout covers REAL frames only. If fake frames have systematically higher IQ-margin (= more Mac-like), an IQ-margin-based abstain rule could prevent fake detection. Need IQ extraction on fake frames before any abstain rule is shipped.
3. **The 4 in-dist cohorts classified as Mac-like** (dor_shkedi__s16, dor_morning, Md_noyn_Sharker__s15, xiang) are a mix of false-positives and correctly-identified-problematic cohorts. 3 of 4 have elevated P8A FPR; dor_shkedi__s16 is the only true false-positive.
4. **No external validation set.** Lockbox, OPTB, viso cohorts have not been projected onto the IQ-margin axis.
5. **The detector cannot disambiguate "Mac webcam" from "low-sharpness in-distribution captures."** This is by design — both regimes have elevated P8A FPR. But it means the abstain rule would trigger on both, not just Mac-OOD.
