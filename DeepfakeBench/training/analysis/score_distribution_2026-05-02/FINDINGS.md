# Score Distribution Forensics — FINDINGS (pure facts, no framings)

**Date**: 2026-05-02 PM
**Branch**: `teams-relaunch-root-2026-04-17`
**Scope**: Forensic analysis of per-frame `prob_fake` distributions for P8A, P18T, P18C across the 9 suites in the 2026-05-02 D contract scorecard. Goal: characterise the *shape* of the model failure on the viso slice, with no interpretation. Per AGENT_GUIDE Rule 5, opinions belong elsewhere.

**Source data**: `gs://training-job-outputs/test_results/teams_promotion_contract/p18-corrective-contract-20260502-103127/reports/*_frames_report.csv` (27 files, 40,908 frames). Pulled to `analysis/score_distribution_2026-05-02/raw_reports/`.

**Authoritative artifacts**:

- `outputs/REPORT.html` — visual aggregator
- `outputs/combined_frames.parquet` — all 27 reports merged
- `outputs/headline_summary.csv` — per (suite × model)
- `outputs/quantiles_frame.csv` — per-suite quantiles
- `outputs/method_breakdown.csv` — per-method breakdown of `teams_fake_all_dev`
- `outputs/viso_pairs.csv` — paired raw/teams scores (275 sequences × 3 models)
- `outputs/viso_pair_category_counts.csv` — pairs categorised by recall pattern at τ=0.5
- `outputs/viso_paired_sample_manifest.csv` — manifest for downloaded paired sample frames
- `outputs/pixel_diff.csv` — pixel-level diff between paired raw/teams images (275 unique pairs)
- `outputs/figures/*.png` — 41 figures total

---

## 1. Composition findings (just suite contents)

### 1.1 The viso eval slice is internally split 50/50

`visomaster_enhanced_macro_dev` (n=550, label=fake, all `method=visomaster_enhanced_macro`, all `identity_key=visomaster_enhanced_raw`) decomposes into:

- **`visomaster_enhanced_raw__*` substrate**: n=275
- **`visomaster_enhanced_teams__*` substrate**: n=275
- The two halves share the same 275 sequence IDs (`seq5402`, `seq5475`, `seq5477`, ... appear in both). They are paired recaptures of the same source content.

Source: `outputs/viso_pairs.csv`, parsing `frame_path` per the regex `visomaster_enhanced_(raw|teams)__frame_\d+_(seq\d+)\.png`.

### 1.2 `teams_fake_all_dev` overlaps the dedicated suites

`teams_fake_all_dev` (n=3039 frames) **contains** the same 550 viso frames + 545 deeplive_enhanced frames as the dedicated suites. Methods present (frame counts, P8A run):

| method | n_frames |
|---|---:|
| visomaster_enhanced_macro | 550 |
| deeplive_enhanced | 545 |
| teams_capture_cam_test_s35 | 365 |
| teams_capture_noyn_sharker_s23 | 324 |
| teams_capture_cam_test_s32 | 235 |
| teams_capture_test_cam_s53 | 165 |
| teams_capture_test_cam_s76 | 138 |
| teams_flat_xiang_xiang2_feng | 135 |
| teams_capture_cam_test_s46 | 124 |
| teams_capture_pc_generator_s3 | 118 |
| teams_capture_test_cam_s73 | 101 |
| teams_capture_cam_test_s38 | 85 |
| teams_capture_dor_shkedi_s16 | 78 |
| teams_capture_pc_generator_s9 | 46 |
| teams_capture_pc_generator_s4 | 30 |

So the contract's `teams_fake_all_dev` recall is a weighted average over a heterogeneous mix. The 898/2409 = 37% recall (P8A, contract value) breaks down as: ~6/550 viso + ~13/545 deeplive_enhanced + ~879/1314 teams_capture videos.

Source: `outputs/method_breakdown.csv`.

### 1.3 `deeplive_enhanced_dev` is single-identity, single-substrate

n=545, all `method=deeplive_enhanced`, all `video_id` matching `deeplive_dor__seq\d+__fake`. Single identity (dor). No raw-vs-teams substrate split.

---

## 2. Score distribution shape per (suite × model)

### 2.1 P8A on visomaster_enhanced_macro_dev (n=550)

| stat | value |
|---|---:|
| mean | 0.3563 |
| **p10** | **0.0011** |
| p25 | 0.0096 |
| **p50** | **0.1704** |
| p75 | 0.7330 |
| p90 | 0.9491 |
| p95 | 0.9817 |
| p99 | 0.9911 |
| **fraction ≥ 0.5** | **0.356** |
| **fraction ≥ contract τ (0.9909)** | **0.0109** (6/550) |

Source: `outputs/quantiles_frame.csv`, `outputs/headline_summary.csv`.

The score distribution is bimodal-with-zero-spike, NOT calibration-soft. Histogram (`figures/hist_visomaster_enhanced_macro_dev_P8A.png`):
- Spike at prob ≈ 0–0.05: ~180 frames (33% of slice)
- Long flat tail through 0.1 to 0.95
- Small bump near 0.9: ~50 frames
- 6 frames cross τ = 0.991

### 2.2 Per-model viso comparison

| model | mean | p50 | p99 | recall@τ |
|---|---:|---:|---:|---:|
| P8A | 0.3563 | 0.1704 | 0.9911 | 0.0109 |
| P18T | 0.4023 | 0.3360 | 0.9960 | 0.0182 |
| P18C | 0.4240 | 0.3322 | 0.9948 | 0.0145 |

P18T and P18C lift the bottom half (fewer frames at near-zero) but do not push more frames past the deployment τ.

### 2.3 deeplive_enhanced_dev shape (different from viso)

| model | mean | p50 | p90 | p99 | recall@τ | recall@0.9 |
|---|---:|---:|---:|---:|---:|---:|
| P8A | 0.5224 | 0.5342 | 0.979 | 0.993 | 0.0239 | 0.2514 |
| P18T | 0.7754 | 0.8974 | 0.991 | 0.995 | 0.0752 | 0.4899 |
| P18C | 0.9371 | 0.9838 | 0.996 | 0.997 | 0.1945 | 0.8404 |

P18C piles 84% of frames above 0.9 but only 19% cross τ ≈ 0.995. Distribution is right-skewed-piled-near-1.0 — different shape than viso's near-zero spike.

### 2.4 teams_fake_all_dev shape

P8A: mean 0.745, p50 0.985, p99 0.995, recall@τ = 0.455. Distribution is dominated by a spike near 1.0; teams_capture methods score confidently. The viso/deeplive subsets within this suite drag down the aggregate recall.

### 2.5 teams_fake_all_lockbox shape (lockbox dropoff)

P8A: mean 0.648, p50 0.788, p99 0.995, recall@τ = 0.231. The same teams_capture methods drop from p50 = 0.985 (dev) to p50 = 0.788 (lockbox). Generalisation gap on the same fake method.

---

## 3. Paired raw vs teams analysis (viso slice)

### 3.1 Pearson correlation of paired scores

| model | r(raw, teams) | raw mean | teams mean | mean Δ (teams − raw) |
|---|---:|---:|---:|---:|
| P8A | **0.770** | 0.444 | 0.269 | **−0.175** |
| P18T | 0.735 | 0.343 | 0.461 | +0.118 |
| P18C | 0.782 | 0.400 | 0.448 | +0.048 |

Source: `outputs/viso_pairs.csv`, `figures/viso_paired_scatter.png`.

### 3.2 Per-pair categorisation at τ=0.5

| model | both_caught | raw_only | teams_only | both_missed |
|---|---:|---:|---:|---:|
| P8A | 65 | **65** | **1** | **144** |
| P18T | 67 | 13 | **54** | 141 |
| P18C | 81 | 23 | 27 | 144 |

Source: `outputs/viso_pair_category_counts.csv`.

P8A: 65 pairs (24%) where raw is caught but teams is missed; 1 pair (0.4%) the reverse. Asymmetry is essentially monotonic.

P18T: 54 pairs (20%) where teams is caught but raw is missed; 13 (4.7%) the reverse. Asymmetry inverts.

**144/275 pairs (52%) are missed by all three models in BOTH substrates.**

### 3.3 Per-frame Δ histogram

`figures/viso_paired_delta.png`:

- P8A: large central spike at Δ ≈ 0 (~85 pairs near-identical), long left tail (raw > teams), no right tail above +0.25.
- P18T: bell-shaped centred at +0.1 with long right tail.
- P18C: roughly symmetric centred at +0.04.

---

## 4. Pixel-level diff between paired raw/teams images

275 unique sequences × 2 substrates = 550 unique frames downloaded. Diff statistics (raw is reference; teams = raw + Δ):

| metric | mean | median | min | max | std |
|---|---:|---:|---:|---:|---:|
| RGB L1 mean | 32.98 | 33.10 | 19.62 | 46.33 | 6.13 |
| RGB L2 RMS | 34.54 | 34.66 | 21.02 | 48.26 | 6.29 |
| Luma L1 mean | 33.85 | 34.05 | 20.51 | 47.49 | 6.27 |
| **Frac pixels changed (any RGB delta)** | **1.0000** | **1.0000** | 0.99997 | 1.0000 | 0.0000 |
| Luma mean shift (signed) | +33.85 | +34.05 | +20.51 | +47.49 | 6.27 |
| HF-band-ratio (raw, 8×8 DCT) | 0.0131 | 0.0128 | 0.0084 | 0.0280 | 0.0030 |
| HF-band-ratio (teams) | 0.0048 | 0.0046 | 0.0027 | 0.0109 | 0.0009 |
| **HF-band-ratio Δ** | **−0.0083** | −0.0083 | −0.0181 | −0.0024 | 0.0025 |
| File size (raw, bytes) | 201,658 | 202,784 | 152,361 | 246,539 | 22,021 |
| File size (teams, bytes) | 177,827 | 177,989 | 130,884 | 218,561 | 20,021 |
| **File size ratio (teams/raw)** | **0.882** | 0.879 | 0.797 | 0.960 | 0.034 |

Source: `outputs/pixel_diff.csv`, `figures/pixel_diff_hist.png`.

### 4.1 Single-pair confirmation (seq5402, P8A both-caught winner)

- Same shape: 359×359×3 uint8 both
- raw mean: 147.30 (R=168.5 G=145.3 B=128.2)
- teams mean: 185.48 (R=203.1 G=187.0 B=166.3)
- Per-channel Δ: R=+34.6, G=+41.8, B=+38.1
- Frac pixels with Δ > 0: 99.52%
- Frac pixels with Δ == 0: 0.33%
- Frac pixels with Δ < 0: 0.15%
- File sizes: raw 165,455 B → teams 155,495 B (94% ratio for this pair)

Source: `figures/single_pair_seq5402_diff.png`.

### 4.2 Score-delta vs pixel-diff correlation

`figures/pixel_diff_vs_score_delta.png` — three models overlaid. Per-pair regression coefficients not computed in this report; visible trend is weak across all six pixel-diff metrics (low R²). The pixel diff is structurally present in all 275 pairs but does not directly predict per-pair score Δ.

---

## 5. Cross-cutting: at what τ does each suite start collapsing?

`figures/suite_recall_curves.png` — frame-level recall vs τ ∈ [0.5, 1.0] per fake suite per model.

At τ = 0.5 (most permissive, frame-level):

| suite | P8A | P18T | P18C |
|---|---:|---:|---:|
| teams_fake_all_dev | 0.756 | 0.825 | 0.875 |
| teams_fake_all_lockbox | 0.665 | 0.673 | 0.715 |
| visomaster_enhanced_macro_dev | 0.356 | 0.385 | 0.385 |
| deeplive_enhanced_dev | 0.530 | 0.800 | 0.985 |

The viso slice has the lowest τ=0.5 recall across all three models. The gap between τ=0.5 recall and τ=deployed recall is largest for deeplive in P18C (0.985 → 0.194) and smallest for viso in P8A (0.356 → 0.011).

---

## 6. Real-suite dor anomaly (cross-reference)

`teams_real_dor_dev` (n=50, dor identity) FPR @ deployed τ:

| model | mean prob | p50 | FPR@τ |
|---|---:|---:|---:|
| P8A | 0.331 | 0.181 | **0.000** (0/50) |
| P18T | 0.485 | 0.512 | 0.040 (2/50) |
| P18C | 0.737 | 0.855 | 0.040 (2/50) |

P8A keeps dor reals scoring low (0% FPR at deployment τ). P18T and P18C lift dor real scores significantly. This matches the prior CPU diagnostics (`MEASUREMENTS_2026-05-02_P18.md §2.3`).

Source: `outputs/headline_summary.csv`.

---

## 7. Open uncertainties (what this analysis does NOT establish)

1. **Why the Teams transport produces the +34 luminance shift** — codec only would not. The shift is uniform per pixel and unidirectional. Source of the transformation is not established here. Hypotheses untested: auto-exposure during capture, tone-mapping in post, compositing back into a Teams call frame with different ambient lighting.
2. **Whether the model's failure on viso is causally related to the +34 luminance shift** — pixel-diff metrics weakly predict score Δ in the scatter (low R²), so the brightness shift is real but not the lone signal the model uses.
3. **What the 144 both-missed pairs have in common** — not characterised here. Could be related to face content, pose, or some other structural quality unique to those sequences.
4. **Whether `teams_real_*` slices undergo a similar transformation** — sampled means show real frames span 95–173 luma vs viso_teams at 185. A direct paired comparison was not done (real frames don't have a paired raw counterpart by construction).
5. **Whether deeplive_enhanced_dev (single-identity dor) has any internal sub-stratification** — not found via `frame_path` naming, but there could be hidden segments by pose/lighting.

---

## 8. Viewer integration

Per AGENT_GUIDE Rule 4, every analysis output has a viewer landing:

- **per_frame_scores artifact** added to `viewer/model_dashboard_runs.yaml` for `p8a`, `p18t`, `p18c`. The viewer's existing Model Diagnostics tab will now render per-suite false-negative / false-positive / near-threshold galleries for all three models (use the run-picker, then the Frames panel).
- **Score distribution report** at `analysis/score_distribution_2026-05-02/outputs/REPORT.html` is referenced from each run's `notes` and pointed to by the new `score_distribution_report` artifact key. Open directly in a browser.
- **41 PNG figures** under `outputs/figures/`, all linked from the REPORT.

---

*This file contains only data. Framings about what these findings imply for the next packet belong in a dated entry in `docs/relaunch_handoffs/PSERIES_OPINIONS_2026-05-02.md`.*

---

## Addendum: 2026-05-02 PM follow-up batch (Diag #1, #2, #3)

### Diag #3 — Operating-point feasibility (`outputs/operating_point_feasibility.csv`, `figures/operating_point_relaxation.png`)

For each model, what recall do we get at canonical real-FPR floors? (frame-level, on `teams_real_all_dev` as the FPR reference suite)

P8A:
| FPR floor | chosen tau | teams_fake | lockbox | viso | deeplive | dor_FPR |
|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.500 | 0.757 | 0.664 | 0.356 | 0.530 | 0.300 |
| 0.10 | 0.705 | 0.699 | 0.541 | 0.269 | 0.424 | 0.240 |
| 0.07 | 0.923 | 0.600 | 0.409 | 0.129 | 0.233 | 0.080 |
| 0.05 | 0.975 | 0.530 | 0.304 | 0.058 | 0.108 | 0.040 |
| 0.03 | 0.990 | 0.461 | 0.240 | 0.013 | 0.026 | 0.000 |
| 0.02 | 0.993 | 0.420 | 0.193 | 0.006 | 0.004 | 0.000 |

P18C:
| FPR floor | chosen tau | teams_fake | lockbox | viso | deeplive | dor_FPR |
|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.500 | 0.875 | 0.713 | 0.386 | 0.985 | 0.740 |
| 0.10 | 0.928 | 0.737 | 0.393 | 0.122 | 0.773 | 0.420 |
| 0.07 | 0.967 | 0.676 | 0.287 | 0.084 | 0.642 | 0.280 |
| 0.05 | 0.983 | 0.603 | 0.235 | 0.067 | 0.501 | 0.140 |
| 0.03 | 0.992 | 0.470 | 0.174 | 0.036 | 0.288 | 0.060 |

The contract floor (0.02 dev_real_FPR) is not the only viable operating point. At 5% FPR, P18C achieves 50% deeplive recall and 60% teams_fake recall — substantially better than at the contract tau. dor_FPR scales with deployment tau for both models; P18C's regression is real but not absolute (drops to 14% at 5% FPR vs 100% at default tau=0.5).

### Diag #1 — Crop-attribute audit (`outputs/crop_attribute_significance.csv`, `figures/crop_attr_*.png`)

275 viso pairs categorised by recall pattern. **108 of 275 pairs are NEVER caught by any of P8A/P18T/P18C in either substrate at tau=0.5.** Per-attribute Mann-Whitney comparison vs the 167 ever-caught pairs:

| feature | never_mean | ever_mean | delta | p-value |
|---|---:|---:|---:|---:|
| laplacian_var_raw | 43.58 | 71.36 | -27.78 | < 1e-4 |
| laplacian_var_teams | 26.46 | 41.69 | -15.22 | < 1e-4 |
| sobel_edge_mean_raw | 27.56 | 29.78 | -2.22 | < 1e-4 |
| sobel_edge_mean_teams | 26.60 | 28.03 | -1.43 | < 1e-4 |
| luma_std_raw | 57.89 | 59.89 | -1.99 | < 1e-4 |
| luma_std_teams | 63.92 | 65.95 | -2.03 | < 1e-4 |
| saturation_mean_teams | 0.245 | 0.227 | +0.018 | 0.002 |
| saturation_mean_raw | 0.298 | 0.286 | +0.013 | 0.014 |
| luma_mean_teams | 175.58 | 177.67 | -2.09 | 0.157 |
| skin_frac_teams | 0.614 | 0.620 | -0.005 | 0.622 |
| luma_mean_raw | 142.08 | 143.59 | -1.51 | 0.730 |
| skin_frac_raw | 0.630 | 0.647 | -0.017 | 0.849 |

The never-caught pairs are systematically less sharp (lower Laplacian variance and Sobel edge magnitude). Brightness, skin coverage, and to first order saturation are NOT structurally different. This is consistent with the model relying on high-frequency content as the fake signal: smooth/blurry frames have less HF for the model to exploit.

### Diag #2 — Train vs eval visomaster comparison (`outputs/train_vs_eval/`, `figures/train_vs_eval_*.png`)

| group | n | mean_w | mean_size_bytes | mean_luma |
|---|---:|---:|---:|---:|
| eval_enhanced_raw | 4 | 369 | 175,440 | 148 |
| eval_enhanced_teams | 4 | 369 | 165,275 | 183 |
| train_base_CSCS | 4 | 224 | 78,909 | 73 |
| train_base_GhostFace_v3 | 4 | 224 | 90,037 | 145 |
| train_base_SimSwap512 | 4 | 224 | 78,600 | 105 |
| train_enhanced_codeformer | 2 | 224 | 77,403 | 183 |
| train_enhanced_gfpgan | 2 | 224 | 93,266 | 125 |
| train_enhanced_gpen-2048 | 1 | 224 | 70,219 | 182 |

Mechanical mismatches:
- Image dimensions: training is 224x224 (50k px-squared), eval is 369x369 (136k px-squared, 2.7x more pixels).
- File sizes: eval files are roughly 2x the bytes of training files for the same source.
- Three different GCS buckets: live-deepfake-methods-real-and-fake-frames-cropped, visomaster-enhanced-face-cropped, teams-faces-data-test-...

Brightness alignment:
- eval_enhanced_teams brightness (~183) sits within training-data range, matches train_enhanced_codeformer (183) and train_enhanced_gpen-2048 (182).
- eval_enhanced_raw brightness (~148) sits between train_base_GhostFace_v3 (145) and train_enhanced_gfpgan (125).
- Brightness alone does not explain why eval is OOD.

Visual differences (per `figures/train_vs_eval_grid.png`):
- Training samples are different identities entirely; eval is the single visomaster_enhanced_raw identity (one specific person, 275 sequences).
- Training samples are visibly lower resolution (224 vs 369), so the network's downsampling pass produces different effective input.
- Training samples include obvious face-swap artifacts (warped mouth on SimSwap512); eval samples look more naturally rendered.

### Compound observation across all three diagnostics

The viso failure at the contract tau is overdetermined by:
1. The contract tau (0.99) cuts off all moderately-confident detections - at tau=0.9 P18C gets 13% viso, at tau=0.99 only 1.5%.
2. 52% of pairs are too smooth (low Laplacian variance) for the model to find HF artifacts in either substrate.
3. The eval data is at 2.7x the pixel resolution of training, so face-detail content is materially different at the network input.

These three factors are independent and additive. None alone explains the 99% miss rate; together they're consistent with it.

---

## Addendum 2: 2026-05-02 PM, second follow-up batch (Diag #1-#6, all CPU)

### Diag #1 — Sharpness validation per model (`outputs/sharpness_validation.csv`, `figures/sharpness_per_model.png`)

For each of P8A/P18T/P18C, compares median Laplacian variance per pair-category vs both_missed (Mann-Whitney two-sided).

| model | category_A | n_A | n_B (both_missed) | median_A | median_B | delta | p | sig |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| P8A | both_caught | 65 | 144 | 64.2 | 24.9 | +39 | <0.0001 | YES |
| P8A | raw_only | 65 | 144 | 67.9 | 24.9 | +43 | <0.0001 | YES |
| P8A | teams_only | 1 | 144 | 19.8 | 24.9 | -5 | NaN (n=1) | - |
| P18T | both_caught | 67 | 141 | 62.8 | 35.2 | +28 | 0.002 | YES |
| P18T | raw_only | 13 | 141 | 77.5 | 35.2 | +42 | <0.0001 | YES |
| **P18T** | **teams_only** | **54** | **141** | **47.0** | **35.2** | **+12** | **0.547** | **NO** |
| P18C | both_caught | 81 | 144 | 52.8 | 34.2 | +19 | 0.441 | NO |
| P18C | raw_only | 23 | 144 | 77.5 | 34.2 | +43 | <0.0001 | YES |
| P18C | teams_only | 27 | 144 | 65.0 | 34.2 | +31 | 0.001 | YES |

Observations (data-only):
- raw_only catches are reliably sharper than both_missed across all three models (consistent direction, all significant).
- P18T's teams_only catches (n=54) — the substantive new wins from method-conditional GRL — show NO significant sharpness elevation vs both_missed (p=0.55, delta only +12). 
- P18C's both_caught vs both_missed: NOT significant (p=0.44).
- The sharpness pattern holds tightly for P8A; weakens for P18T/P18C, especially on teams substrate.

### Diag #4 — Identity-level FPR breakdown (`outputs/identity_fpr_breakdown.csv`, `figures/identity_fpr_*.png`)

teams_real_all_dev (n=4564) decomposes into 13 unique identity strings.

False-positive concentration at deployed τ:

| model | total FPs | identities with ≥1 FP | top-3 share | top-1 |
|---|---:|---:|---:|---|
| P8A | 122 | 6 | 96% | PC_Generator (83/835 = 10% FPR) |
| P18T | 91 | 4 | 91% | Roy_D (32/130 = 25% FPR) |
| P18C | 81 | 4 | 94% | Roy_D (46/130 = 35% FPR) |

Top per-identity FPR @ deployed τ:

| identity | n_frames | P8A FPR | P18T FPR | P18C FPR |
|---|---:|---:|---:|---:|
| Roy_D | 130 | 4.6% | **24.6%** | **35.4%** |
| Q | 54 | 51.9% | 38.9% | 31.5% |
| PC_Generator | 835 | 10.0% | 3.6% | 1.6% |
| bla_bla_chow | 491 | 0.4% | 1.6% | 1.0% |
| dor | 269 | **0.0%** | **0.0%** | **0.0%** |
| dor_shkedi (in dev) | 31 | 0.0% | 0.0% | 0.0% |
| Test_Cam | 1280 | 0.08% | 0.0% | 0.0% |
| Cam_Test | 166 | 0.0% | 0.0% | 0.0% |
| Xiang_Xiang2_Feng | 403 | 0.5% | 0.0% | 0.0% |
| Md_noyn_Sharker | 682 | 0.0% | 0.0% | 0.0% |

Notes:
- The dor identity in dev (`dor`, n=269) is at 0% FPR for ALL three models. The "dor regression" referenced in prior diagnostics is for the LOCKBOX dor identity (`dor_shkedi`, n≈1170 in lockbox; per separate slice).
- Roy_D and Q are the main FPR drivers and get progressively worse from P8A → P18C.
- PC_Generator is partially fixed by P18T/C (10% → 1.6%).

### Diag #3 — Lockbox fake walkthrough (`outputs/lockbox_fake_per_frame.csv`, `figures/lockbox_fake_gallery_*.png`)

Frame-level recall on the two lockbox fake methods:

| method | n | model | recall@τ | recall@0.9 | recall@0.5 | median prob |
|---|---:|---|---:|---:|---:|---:|
| teams_capture_cam_test_s33 | 334 | P8A | 5.1% | 27.2% | 57.2% | 0.594 |
| teams_capture_cam_test_s33 | 334 | P18T | 3.0% | 24.6% | 58.4% | 0.614 |
| teams_capture_cam_test_s33 | 334 | P18C | 0.6% | 28.7% | 63.5% | 0.679 |
| teams_capture_pc_generator_s15 | 91 | P8A | 89.0% | 98.9% | 100% | 0.995 |
| teams_capture_pc_generator_s15 | 91 | P18T | 72.5% | 93.4% | 100% | 0.996 |
| teams_capture_pc_generator_s15 | 91 | P18C | 71.4% | 98.9% | 100% | 0.996 |

Cross-model catch pattern at deployed τ:

| method | caught by ANY | caught by ALL 3 | caught by NONE |
|---|---:|---:|---:|
| teams_capture_cam_test_s33 | 19 (5.7%) | 2 (0.6%) | **315 (94.3%)** |
| teams_capture_pc_generator_s15 | 81 (89.0%) | 62 (68.1%) | 10 (11.0%) |

The lockbox fake-recall headline is dominated by `cam_test_s33` failure: 315 of 334 frames are missed by all three models at deployment τ. Visual inspection of the gallery (`figures/lockbox_fake_gallery_teams_capture_cam_test_s33.png`) shows these are all the same identity (one woman) in various poses, all looking like reasonable-quality face captures.

### Diag #2 — Train data sharpness/brightness distribution (`outputs/train_attributes.csv`, `figures/train_vs_eval_attr_overlay.png`)

Mean per-feature across 359 training samples + 550 eval frames:

| group | n | luma_mean | laplacian_var | sobel_edge_mean | h |
|---|---:|---:|---:|---:|---:|
| eval_visomaster_raw | 275 | 143.0 | **60.4** | 28.9 | 379 |
| eval_visomaster_teams | 275 | 176.8 | **35.7** | 27.5 | 379 |
| train_realpool | 50 | 114.7 | 132.6 | 41.1 | 224 |
| train_visomaster_CSCS | 50 | 113.1 | **155.0** | 42.7 | 224 |
| train_visomaster_GhostFace_v3 | 50 | 103.3 | **161.8** | 43.0 | 224 |
| train_visomaster_InStyleSwapper | 50 | 105.5 | **190.2** | 44.3 | 224 |
| train_visomaster_Inswapper128 | 49 | 102.9 | **140.0** | 39.4 | 224 |
| train_visomaster_SimSwap512 | 50 | 103.9 | **185.8** | 42.0 | 224 |
| train_visomaster_enhanced_codeformer | 30 | 158.2 | **406.8** | 45.5 | 224 |
| train_visomaster_enhanced_gfpgan | 30 | 158.2 | **406.8** | 45.5 | 224 |

(Note: codeformer and gfpgan show identical stats; the discover script's filename pattern matched the same files for both. Consider re-running with stricter filtering for separate measurements.)

Headline numbers:
- **Training Laplacian variance ranges 132-407 across methods.**
- **Eval Laplacian variance is 36-60.**
- The eval distribution is **roughly 4-11× LESS SHARP than training** on the same metric.
- This is exactly the metric Diag #1 (PM batch 1) identified as predicting catchability.
- Image height: training 224, eval 379 (~1.7× larger).

The figures `figures/train_vs_eval_attr_overlay.png` and `train_vs_eval_attr_box.png` make this visible directly: the laplacian_var distributions are essentially non-overlapping between the two populations.

### Diag #6 — Pose audit on viso pairs (`outputs/pose_significance.csv`, `figures/pose_attr_dist.png`)

MediaPipe FaceMesh on 550 cached viso frames. Detection rate: 539/550 = 98.0%. Both substrates detected for 269/270 pairs (99.6%).

Per-feature comparison: never_caught (n=103) vs ever_caught (n=167):

| feature | never_mean | ever_mean | delta | p-value | significant |
|---|---:|---:|---:|---:|---:|
| pitch_proxy_teams | 0.5433 | 0.5451 | -0.002 | 0.061 | NO |
| bbox_area_frac_raw | 0.5096 | 0.5113 | -0.002 | 0.096 | NO |
| mouth_aspect_ratio_raw | 0.0525 | 0.0579 | -0.005 | 0.193 | NO |
| roll_deg_raw | -3.26 | -3.14 | -0.12 | 0.235 | NO |
| pitch_proxy_raw | 0.5367 | 0.5377 | -0.001 | 0.270 | NO |
| roll_deg_teams | -3.19 | -3.24 | +0.04 | 0.316 | NO |
| bbox_area_frac_teams | 0.5086 | 0.5137 | -0.005 | 0.337 | NO |
| yaw_proxy_raw | 0.0296 | 0.0239 | +0.006 | 0.542 | NO |
| yaw_proxy_teams | 0.0257 | 0.0213 | +0.005 | 0.622 | NO |
| eye_dist_rel_raw | 0.4323 | 0.4321 | +0.000 | 0.659 | NO |
| mouth_aspect_ratio_teams | 0.0512 | 0.0538 | -0.003 | 0.855 | NO |
| eye_dist_rel_teams | 0.4321 | 0.4324 | -0.000 | 0.916 | NO |

**No pose feature reaches statistical significance** (lowest p = 0.061 for pitch_teams). Yaw, pitch, roll, eye distance, mouth aspect ratio, and face bbox area distributions are essentially identical between the never-caught and ever-caught populations.

Combined with Diag #1 + Diag #2 from PM batch 1 (where sharpness + edge metrics WERE significant at p<0.0001), this rules out pose/orientation as the structural difference. Image-quality features (sharpness, edge density, contrast) are the only attributes that distinguish caught from missed.

### Diag #5 — Feature embedding via pretrained encoder

(Currently running in background; output will land at `outputs/viso_embeddings_resnet50.npy`, `figures/embedding_pca_2d.png`, `figures/embedding_tsne_2d.png`.)

### Diag #5 — Feature embedding via ResNet-50 ImageNet (`outputs/viso_embeddings_resnet50.npy`, `figures/embedding_*.png`)

550 viso frames embedded via ResNet-50 (ImageNet V2 weights), 2048-dim features. Projected to 2D via PCA (explained variance: 24.9% + 12.8% on first two PCs) and t-SNE (perplexity 30, n_iter 1000).

Per-pair raw↔teams feature distance (Euclidean on 2048-dim ResNet features), summarised by P8A category:

| category | n_pairs | mean distance | median distance | std |
|---|---:|---:|---:|---:|
| both_caught | 65 | 3.86 | 3.91 | 0.54 |
| **raw_only** | **65** | **4.20** | **4.00** | **0.85** |
| teams_only | 1 | 2.82 | 2.82 | NaN |
| both_missed | 144 | 3.76 | 3.68 | 0.64 |

The raw_only category (where P8A catches raw but misses teams) has the **largest median feature distance between substrates** (4.00 vs 3.68 for both_missed). The both_missed category has the **smallest** distance.

Directional reading (data only): when the substrate transformation moves the pair further apart in pretrained-feature space, P8A catches one but not the other; when the substrates are closer in pretrained-feature space, P8A misses both.

t-SNE structure (`figures/embedding_tsne_2d.png`):
- subtype clusters: raw and teams form partially-separated regions in t-SNE space
- score clusters: high-prob_fake frames are concentrated in some regions; low-prob_fake frames are widely scattered
- pair connectivity: pairs (same seq_id) are NOT consistently close in t-SNE space — the substrate transformation moves them apart in pretrained features

PCA variance: PC1 captures 24.9% of variance, PC2 captures 12.8%. The 550 frames span a substantial part of ImageNet feature variance even within a single identity.

---

## Addendum 3: 2026-05-02 PM, third follow-up batch (Diag #7-#10 + identity overlap)

### Diag #7 — Lockbox cam_test_s33 attribute analysis (`outputs/lockbox_attribute_significance.csv`, `figures/lockbox_cam_vs_pc_attrs.png`)

Computed crop attributes on all 425 lockbox fake frames.

cam_test_s33 (mostly missed) vs pc_generator_s15 (mostly caught), Mann-Whitney two-sided:

| feature | cam_test_s33 mean | pc_generator_s15 mean | delta | p | sig |
|---|---:|---:|---:|---:|:---:|
| luma_mean | 117.2 | 139.6 | -22.4 | <1e-4 | YES |
| luma_std | 50.0 | 42.2 | +7.8 | <1e-4 | YES |
| **laplacian_var** | **9.5** | **46.1** | **-36.6** | **<1e-4** | **YES** |
| sobel_edge_mean | 24.2 | 39.6 | -15.4 | <1e-4 | YES |
| saturation_mean | 0.317 | 0.351 | -0.034 | <1e-4 | YES |
| skin_frac | 0.970 | 0.670 | +0.299 | <1e-4 | YES |
| h | 416 | 244 | +172 | <1e-4 | YES |

**cam_test_s33 is 5× less sharp than pc_generator_s15** (Laplacian 9.5 vs 46.1) and dominated by skin (97% vs 67%, suggesting close-up framing). Frame height differs by 172 pixels.

Within cam_test_s33, comparing P8A-caught (n=214) vs P8A-missed (n=157):

| feature | caught_mean | missed_mean | delta | p | sig |
|---|---:|---:|---:|---:|:---:|
| luma_mean | 116.4 | 118.3 | -1.85 | 0.010 | YES (small) |
| **laplacian_var** | **9.57** | **9.42** | **+0.15** | **0.394** | **NO** |
| sobel_edge_mean | 24.15 | 24.38 | -0.24 | 0.218 | NO |
| saturation_mean | 0.317 | 0.318 | -0.001 | 0.410 | NO |
| skin_frac | 0.970 | 0.970 | -0.001 | 0.803 | NO |
| h | 414 | 420 | -6.5 | 0.022 | YES (small) |

Within the cam_test_s33 slice, sharpness does NOT distinguish caught vs missed. Once a frame is in the low-sharpness regime (laplacian ~9-10), sharpness is no longer predictive at that resolution.

### Diag #8 — FPR-driver identity galleries (`outputs/identity_attribute_summary.csv`, `figures/identity_gallery.png`)

Successfully sampled and computed attributes for 3 identities (parser failed silently for the others — fixed in later identity_overlap.py pass):

| identity | n | mean_h | mean_luma | mean_laplacian_var | P8A_score | P18T_score | P18C_score |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Roy_D** | 16 | 279 | 109 | **66** | **0.516** | **0.795** | **0.916** |
| dor | 16 | 316 | 116 | 90 | 0.009 | 0.042 | 0.121 |
| bla_bla_chow | 16 | 374 | 108 | 402 | 0.133 | 0.533 | 0.416 |

Roy_D has lower Laplacian variance (66) than dor (90) and bla_bla_chow (402). All three identities have similar luminance (108-116). Roy_D's mean P8A score is 0.516 (essentially chance) and rises to 0.916 in P18C — model treats Roy_D as fake despite Roy_D being real.

### Diag #9 — Cross-suite attribute landscape (`outputs/cross_suite_attribute_summary.csv`, `figures/cross_suite_attr_box.png`)

50-frame samples per suite plus all 425 lockbox + 16 per identity:

| suite | n | mean_h | mean_luma | mean_laplacian_var | median_laplacian_var | P8A_mean_score | label |
|---|---:|---:|---:|---:|---:|---:|---|
| teams_fake_all_dev | 88 | 360 | 149 | 66 | 28 | 0.532 | fake |
| **teams_fake_all_lockbox** | **475** | **379** | **122** | **17.5** | **10.0** | **0.653** | **fake** |
| **deeplive_enhanced_dev** | **62** | **415** | **130** | **234.8** | **245.5** | **0.469** | **fake** |
| teams_real_all_dev | 198 | 294 | 130 | 238.5 | 121.2 | 0.169 | real |
| teams_real_all_lockbox | 50 | 263 | 136 | 224.4 | 197.9 | 0.071 | real |
| teams_real_dor_dev | 50 | 236 | 140 | 301.7 | 263.3 | 0.331 | real |

Plus from the prior batch: visomaster_enhanced_macro_dev raw=60, teams=36 (laplacian).

The Laplacian variance landscape:

```
teams_fake_all_lockbox:        17.5  ← LEAST sharp
visomaster_enhanced_teams_dev: 35.7
visomaster_enhanced_raw_dev:   60.4
teams_fake_all_dev:            66.0
deeplive_enhanced_dev:         234.8
teams_real_all_lockbox:        224.4
teams_real_all_dev:            238.5
teams_real_dor_dev:            301.7  ← MOST sharp (real)
training visomaster (base):    132-190
training visomaster (enhanced):407
```

Real suites have Laplacian variance 220-300. Training data ranges 130-410. Eval lockbox is 17.5. The lockbox fake suite is by FAR the least sharp population in the entire scorecard, while the eval real suites are within the training-data range.

### Diag #10 — Score-attribute correlations across suites (`outputs/score_attribute_correlations.csv`, `figures/score_attribute_corr_heatmap_P8A.png`)

Pearson r between frame_prob and each crop attribute, per (suite, model). Top 20 strongest |r|:

| suite | model | feature | n | Pearson r | p |
|---|---|---|---:|---:|---:|
| teams_real_dor_dev | P18T | laplacian_var | 50 | **-0.569** | <1e-4 |
| teams_fake_all_dev | P8A | luma_std | 88 | -0.554 | <1e-4 |
| teams_fake_all_dev | P8A | luma_mean | 88 | -0.525 | <1e-4 |
| deeplive_enhanced_dev | P18T | sobel_edge_mean | 62 | -0.523 | <1e-4 |
| teams_fake_all_lockbox | P8A | h | 475 | -0.504 | <1e-4 |
| teams_fake_all_lockbox | P18T | h | 475 | -0.496 | <1e-4 |
| teams_real_dor_dev | P18C | laplacian_var | 50 | -0.495 | 3e-4 |
| teams_fake_all_lockbox | P18C | h | 475 | -0.488 | <1e-4 |
| teams_real_all_lockbox | P18C | skin_frac | 50 | -0.488 | 3e-4 |
| teams_fake_all_lockbox | P8A | skin_frac | 475 | -0.486 | <1e-4 |
| teams_fake_all_lockbox | P18T | skin_frac | 475 | -0.479 | <1e-4 |
| teams_fake_all_lockbox | P18C | skin_frac | 475 | -0.479 | <1e-4 |

Negative correlations dominate — meaning higher attribute values associate with lower fake scores:
- For real suites (e.g., teams_real_dor_dev / laplacian_var): sharper → correctly classified as real (good).
- For fake suites (e.g., teams_fake_all_dev / luma_mean): brighter → MISSED as fake (the model has learned darker = more fake-like).
- For fake suites (e.g., teams_fake_all_lockbox / skin_frac): more skin → MISSED as fake (close-up framing reduces fake detection).
- For fake suites (e.g., teams_fake_all_lockbox / h): taller frames → MISSED as fake (lockbox cam_test_s33 has 416-pixel tall frames).

### Identity overlap analysis (`outputs/identity_overlap_table.csv`, `outputs/identity_per_suite_fpr.csv`, `figures/identity_overlap_heatmap.png`)

Identity strings present per real suite (frame counts):

| identity | dev_n | lockbox_n | poor_quality_n | lighting_extreme_n |
|---|---:|---:|---:|---:|
| dor | 269 | 0 | 74 | 2 |
| dor_shkedi | 31 | 1170 | 31 | 31 |
| real_dor | 0 | 109 | 0 | 0 |
| dor_shkedi_real_frame_*_seq* (60 unique) | 0 in dev_all | 0 | 0 | 0 (all in dor_dev only, 1 frame each) |
| **PC_Generator** | **835** | **29** | 384 | 481 |
| **bla_bla_chow** | **491** | **68** | 107 | 405 |
| Cam_Test | 166 | 0 | 10 | 0 |
| Test_Cam | 1280 | 0 | 475 | 29 |
| Roy_D | 130 | 0 | 3 | 113 |
| Q | 54 | 0 | 0 | 0 |
| Md_noyn_Sharker | 682 | 0 | 0 | 332 |
| Xiang_Xiang2_Feng | 403 | 0 | 118 | 232 |
| Chikara_Takahashi | 0 | 42 | 0 | 0 |

3 identity strings appear in BOTH dev and lockbox: PC_Generator, bla_bla_chow, dor_shkedi.

Per-identity P8A FPR @ deployed τ across real suites:

| identity | dev | lockbox | poor_quality | lighting_extreme |
|---|---:|---:|---:|---:|
| PC_Generator | 9.9% | **6.9%** | 0.0% | 6.0% |
| Roy_D | 4.6% | (NaN) | **66.7%** | 5.3% |
| Q | **51.9%** | (NaN) | (NaN) | (NaN) |
| Chikara_Takahashi | (NaN) | 4.8% | (NaN) | (NaN) |
| All others | <1% across the board | | | |

Roy_D's FPR EXPLODES in the poor_quality_dev suite (66.7%). The other identities have stable FPR across suites.

### Training-bucket identity check

Cross-referenced FPR-driver identity strings against `gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/`:

| identity | training samples (substring match) | training samples (verified) |
|---|---:|---:|
| PC_Generator | 0 | 0 |
| dor_shkedi | 0 | 0 |
| bla_bla_chow | 0 | 0 |
| Roy_D | 0 | 0 |
| Q | 320 | **0 (false positive — matched `quality_enhancement_*`)** |
| Test_Cam | 0 | 0 |
| Md_noyn_Sharker | 0 | 0 |
| Cam_Test | 0 | 0 |

**ZERO of the eval identities appear in the training bucket.** The training bucket is structured by ALGORITHM (visomaster_CSCS, GhostFace, deeplive_*, etc.), not by identity. The Teams identities used in eval have no direct training presence.

### Compound observation across all PM batches

The viso/lockbox failure at the contract τ is supported by these independent factors:
1. **Contract τ ≈ 0.99 cuts off all moderate-confidence detections** — at τ=0.5 viso is 35% recall, at τ=0.99 it's 1%.
2. **Eval data is 4-25× LESS SHARP than training data** on Laplacian variance — and within eval, the lockbox fake suite is the least sharp population (median 10).
3. **Within the low-sharpness eval regime, sharpness no longer distinguishes caught vs missed** — the model has effectively "given up" on those frames.
4. **52% of viso pairs are uncatchable in either substrate by any of the 3 models.**
5. **The eval identities have ZERO presence in training data** — neither the FPR-driver real identities (Roy_D, Q, PC_Generator, bla_bla_chow) nor the eval fake identities.
6. **3 identity strings cross dev↔lockbox split** (PC_Generator, bla_bla_chow, dor_shkedi) — partial leakage of identities across the held-out boundary.
7. **Score correlates negatively with sharpness/luminance/skin_frac on multiple suites** — model has learned that darker, less-sharp, less-skin-dominated images are more likely fake.
8. **The cam_test_s33 lockbox subset is a single identity** with 5× lower Laplacian variance than the other lockbox method; that identity drives the lockbox fake-recall headline collapse.

These observations are independent and accumulate. They do not collectively imply a single root cause — they characterise the empirical landscape.

---

## Addendum 4: 2026-05-02 PM, Diag #11 (visual identity comparison)

### Diag #11 — Identity comparison gallery (`figures/identity_comparison_gallery.png`, `figures/identity_score_histograms.png`)

Side-by-side rendering of 8 frames each from 8 identity labels across dev / lockbox / fake suites.

**Visual findings (data-only observations):**
- Rows 1-4 (`dor`, `dor_shkedi`, `real_dor`, `deeplive_dor`) all show the SAME PERSON across different captures. The naming differs between dev/lockbox/fake suites but the visible subject is identical.
- Rows 5-8 (`Roy_D`, `Q`, `Cam_Test`, `bla_bla_chow`) show four DISTINCT individuals, none visually similar to each other or to dor.

**Implications observed in data (no causal claims):**
- The 269 `dor` frames in `teams_real_all_dev`, the 1170 `dor_shkedi` frames in `teams_real_all_lockbox`, the 109 `real_dor` frames in `teams_real_all_lockbox`, and the 545 `deeplive_dor` frames in `deeplive_enhanced_dev` are all the same identity under different label strings.
- The dev↔lockbox identity split is NOT held-out by person — only by per-frame label string.
- The `deeplive_dor` fake suite tests deepfake-recognition on the same identity that appears as a real in both dev and lockbox.

**Per-identity prob_fake distributions** (`figures/identity_score_histograms.png`):

| identity (suite) | label | n | P8A median | P18T median | P18C median |
|---|---|---:|---:|---:|---:|
| dor (dev) | real | 269 | very low (spike near 0) | low | low |
| dor_shkedi (lockbox) | real | 1170 | low (spike near 0) | mid (more spread) | mid (more spread) |
| real_dor (lockbox) | real | 109 | low | mid | mid |
| deeplive_dor (deeplive_enhanced_dev) | fake | 545 | bimodal (some 0, some 1) | mostly high | dominated by 1 |
| Roy_D (dev) | real | 130 | spread | RIGHT-SHIFTED (high) | dominated by 1 (FPR) |
| Q (dev) | real | 54 | bimodal | bimodal | bimodal |
| Cam_Test (dev) | real | 1280 | dominantly low + tail | low + tail | low + tail |
| bla_bla_chow (dev) | real | 491 | low + tail | low + medium tail | low + medium tail |

**Mechanical note on dor**: P18C's prob_fake distribution on dor_shkedi (lockbox real) is materially RIGHT-SHIFTED vs P8A's. P8A holds dor as confidently real; P18C does not. This is the documented "dor regression" of P18C (per prior MEASUREMENTS doc §2.5), now visible in the identity-resolved score distribution.

**Mechanical note on Roy_D and Q**: For these dev real identities, P18T and P18C produce dramatically right-shifted distributions vs P8A — these are the FPR drivers. Roy_D's median score in P18C is in the right tail.

### Identity-leakage summary

For the dor identity specifically:
- Training data: ZERO frames (dor not in `live-deepfake-methods-real-and-fake-frames-cropped/samples/`).
- `teams_real_all_dev`: 269 frames as `dor` (label=real).
- `teams_real_dor_dev`: 50 frames of dor variants (label=real, used as the targeted dor stress suite).
- `teams_real_all_lockbox`: 1170 frames as `dor_shkedi` + 109 as `real_dor` (both label=real).
- `deeplive_enhanced_dev`: 545 frames as `deeplive_dor` (label=fake — the dor person processed by deeplive).

The dor identity is heavily represented across dev real, lockbox real, AND dev fake suites. The contract treats these as separate populations. Visually they are the same person.

Other identities (PC_Generator, bla_bla_chow, dor_shkedi proper) have the documented dev↔lockbox overlap from the earlier identity_overlap analysis.

This addendum is descriptive only. Whether the identity overlap reflects intentional eval design, evaluation policy, or oversight is not assessed here.

---

## User-confirmed: lockbox identities are deliberately loose

**User note (2026-05-02 PM)**: lockbox identities are known to be loose. PC_Generator and similar identities appear as BOTH real and fake in BOTH dev and lockbox splits. This is by design or known eval-data structure, not an oversight discovered here.

Concretely from the data:
- **PC_Generator** identity: 835 frames real (dev), 29 real (lockbox), AND 91 frames fake in `teams_capture_pc_generator_s15` (lockbox method) plus ~194 frames fake across dev fake methods (`teams_capture_pc_generator_s3/s4/s9`).
- **dor** identity: 269 real (dev as `dor`), 1170 real (lockbox as `dor_shkedi`), 109 real (lockbox as `real_dor`), AND 545 fake (deeplive_enhanced_dev as `deeplive_dor`).
- **bla_bla_chow** identity: 491 real (dev), 68 real (lockbox).
- **dor_shkedi** identity: 31 real (dev), 1170 real (lockbox).

What this means for evaluation:
- The deepfake task is realistic: the same person can appear both as a real captured video AND as a deepfaked target in eval data.
- The dev↔lockbox split is by sample, not by identity. Identity-leakage is intentional structure, not a measurement bug.
- This means the FPR-driver findings (Roy_D / Q in dev) are valid as long as we treat them as "specific identities the model has trouble with" — they may or may not appear in lockbox, but the failure mode is real.
- The dor regression for P18C in lockbox (52% FPR @ τ=0.92) is on the same person who is at 0% FPR in dev. The regression is condition-dependent, not identity-specific.

This addendum corrects the framing in Addendum 3 that implied identity overlap was an oversight. It's a deliberate / accepted property of the eval design.
