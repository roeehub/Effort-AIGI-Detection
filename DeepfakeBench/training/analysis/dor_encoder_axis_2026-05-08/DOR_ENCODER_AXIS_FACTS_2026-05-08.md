# Dor encoder-axis characterization — FACTS (2026-05-08)

> **Status: factual-only.** No interpretation. Forbidden words: succeeds,
> fails, wins, promotes, deployment-grade.
>
> **Check (c)** of the IQ-deconvolution program's pre-Stage-2a checks.
> CPU-only (MPS-accelerated). No GPU spend.
>
> **Goal**: characterize WHY P2-D-step3000 lost P8A's signature dor invariance
> (J4 in `analysis/p2_eval_2026-05-08/P2_DEEPER_ANALYSIS_FACTS_2026-05-08.md`:
> teams_real_dor_dev FPR 8.0% → 46.0% (5.75×); teams_capture_dor_shkedi_dev
> recall 85.9% → 61.5%).

---

## 1. Method

### 1.1 Cohort

Per-cohort frame counts after image-decode validation
(`analysis/dor_encoder_axis_2026-05-08/_cache/cohort_manifest.csv`):

| cohort | label | n |
|---|---:|---:|
| DOR_REAL_DEV | 0 | 50 |
| DOR_REAL_LOCKBOX | 0 | 100 |
| NON_DOR_REAL_DEV | 0 | 80 |
| DOR_FAKE_DEV | 1 | 78 |
| NON_DOR_FAKE_DEV | 1 | 80 |

Source: P2 scoreboard
(`analysis/p2_eval_2026-05-08/p2_deeper_analysis/outputs/scoreboard.parquet`).

- DOR_REAL_DEV: 50 frames in `teams_real_dor_dev` (the J4-load-bearing real
  cohort).
- DOR_FAKE_DEV: 78 frames in `teams_capture_dor_shkedi_dev` (the J4-load-bearing
  fake cohort).
- DOR_REAL_LOCKBOX: 100-frame random subset of the 1170 Dor frames in
  `teams_real_all_lockbox` (random_state=0).
- NON_DOR_REAL_DEV: 80-frame random subset of `teams_real_all_dev` excluding
  any frame_path containing "dor" (random_state=0).
- NON_DOR_FAKE_DEV: 80-frame random subset of `teams_fake_all_dev` excluding
  any frame_path containing "dor" (random_state=0).

Total: 388 unique frames.

### 1.2 Per-frame feature extraction

For each ckpt (P8A_REFERENCE_STEP5000, E2B_TOP_N_STEP3200,
P2_D_FOURIER_PERIODIC_STEP3000), forward each cohort frame through the
EffortDetector backbone and capture:

- `final_cls`: 512-dim CLS output post `backbone.proj` (the head input).
- `layer11_cls`: 768-dim CLS output at `transformer.resblocks[11]` via hook
  (pre-projection, last-block representation).

Image preprocessing: cv2 BGR→RGB, resize 224×224 INTER_LINEAR, CLIP
mean/std normalization. MPS device. Same recipe as
`analysis/intermediate_layer_probe_2026-04-30/intermediate_layer_probe.py`.

Cached at `_cache/cohort_features__{label}.npz`. Re-runnable; no GCS reads
on cache hit.

### 1.3 Score source

Per-frame `frame_prob` from the P2 Phase A scoreboard:
- P8A: column `p8a_reference_step5000`
- E2B: column `e2b_top_n_step3200`
- P2D: column `p2_d_fourier_periodic_step3000`

These are the same scores J4 used.

### 1.4 Centroid distances

For each (ckpt × feature_layer ∈ {final, layer11}):
- Compute per-cohort centroid as mean of un-normalized feature vectors.
- Cosine distance between centroids = `1 - cos_similarity(C_a, C_b)`.

Six pair-distances reported per (ckpt × feature_layer):
- `dor_real_vs_non_dor_real`: how Dor reals position relative to non-Dor reals.
- `dor_fake_vs_non_dor_fake`: how Dor fakes position relative to non-Dor fakes.
- `dor_real_vs_dor_fake`: Dor real-fake separation.
- `non_dor_real_vs_non_dor_fake`: non-Dor real-fake separation (control).
- `dor_real_vs_non_dor_fake`: Dor reals vs non-Dor fake centroid (cross-class).
- `non_dor_real_vs_dor_fake`: non-Dor reals vs Dor fake centroid (cross-class).

### 1.5 Per-frame distance from each frame to each centroid

For cohort frames F_c, compute `cos_dist(f, C)` to each of the 4 anchor
centroids: `C_dor_real`, `C_dor_fake`, `C_non_dor_real`, `C_non_dor_fake`.
Reports p10/p50/p90 per cohort × ckpt.

### 1.6 IQ-feature × score Pearson r per cohort

For each (ckpt × cohort × IQ feature ∈ primary_6), Pearson r between
per-frame `frame_prob` and per-frame IQ feature. IQ features sourced from
the cross-pool atlas where available (176/388 frames); inline-computed
from local files for the remaining 212 frames using the same cv2/LAB/YCbCr
formulas as `build_iq_atlas.py:per_frame_attrs`.

---

## 2. Centroid distances — final-CLS (512-d, post-projection)

Cosine distance (`1 - cos_similarity`). Higher = farther. Scale roughly 0–2.

| pair | P8A | E2B | P2D |
|---|---:|---:|---:|
| dor_real_vs_non_dor_real | 0.062 | 0.001 | **0.342** |
| dor_fake_vs_non_dor_fake | 0.003 | 0.007 | **1.077** |
| dor_real_vs_dor_fake | 0.952 | 1.976 | **0.112** |
| non_dor_real_vs_non_dor_fake | 1.362 | 1.998 | 1.986 |
| dor_real_vs_non_dor_fake | 1.020 | 1.995 | 1.524 |
| non_dor_real_vs_dor_fake | 1.299 | 1.984 | 0.759 |

### 2.1 P2D-vs-P8A directly comparable observations

- **Dor real-fake separation**: P8A 0.952 → P2D **0.112** (8.5× closer).
- **Dor real centroid distance from non-Dor real**: P8A 0.062 → P2D 0.342
  (5.5× farther).
- **Dor fake centroid distance from non-Dor fake**: P8A 0.003 → P2D 1.077
  (P2D's Dor fakes are far from non-Dor fakes; P8A's were nearly co-located).

### 2.2 Cross-class distances under P2D

- DOR reals to NON_DOR fakes: 1.524 (still substantial)
- DOR reals to NON_DOR reals: 0.342
- DOR fakes to DOR reals: 0.112 (closest)
- DOR fakes to NON_DOR fakes: 1.077

DOR reals' nearest "anchor" centroid by cosine distance:
- P8A: NON_DOR real (0.062) ← closest
- E2B: NON_DOR real (0.001) ← closest
- P2D: NON_DOR real (0.342) but DOR fake is 0.112 ← Dor fake is now CLOSEST

DOR fakes' nearest anchor centroid:
- P8A: NON_DOR fake (0.003) ← closest
- E2B: NON_DOR fake (0.007) ← closest
- P2D: DOR real (0.112) ← closest

---

## 3. Centroid distances — layer-11 (768-d, encoder pre-projection)

| pair | P8A | E2B | P2D |
|---|---:|---:|---:|
| dor_real_vs_non_dor_real | 0.150 | 0.371 | 0.624 |
| dor_fake_vs_non_dor_fake | 0.051 | 0.419 | 0.281 |
| dor_real_vs_dor_fake | 0.419 | 0.396 | 0.229 |
| non_dor_real_vs_non_dor_fake | 0.746 | 1.303 | 0.987 |
| dor_real_vs_non_dor_fake | 0.532 | 0.844 | 0.305 |
| non_dor_real_vs_dor_fake | 0.680 | 0.914 | 0.688 |

### 3.1 Encoder-vs-projection comparison

- **dor_real_vs_dor_fake**: at L11 P2D = 0.229, at final P2D = 0.112.
  Encoder already has Dor reals and fakes closer than expected; projection
  amplifies it (~2× compression ratio).
- **dor_real_vs_non_dor_real**: at L11 P2D = 0.624, at final P2D = 0.342.
  P2D's Dor reals are pushed away from non-Dor reals at L11 (encoder),
  then partly absorbed by the projection.

---

## 4. Per-cohort score statistics

At each ckpt's per-frame `frame_prob` (raw scores from the scoreboard).

### 4.1 Mean / median / spread

| cohort | metric | P8A | E2B | P2D |
|---|---|---:|---:|---:|
| DOR_REAL_DEV | mean | 0.331 | 0.300 | 0.471 |
| DOR_REAL_DEV | p50 | 0.181 | 0.223 | 0.436 |
| DOR_REAL_DEV | std | 0.347 | 0.294 | 0.176 |
| DOR_REAL_DEV | p90 | 0.905 | 0.810 | 0.676 |
| DOR_FAKE_DEV | mean | 0.945 | 0.722 | 0.492 |
| DOR_FAKE_DEV | p50 | 0.986 | 0.777 | 0.489 |
| DOR_FAKE_DEV | std | 0.115 | 0.207 | 0.124 |
| DOR_FAKE_DEV | p10 | 0.889 | 0.422 | 0.330 |
| DOR_REAL_LOCKBOX | mean | 0.084 | 0.095 | 0.286 |
| DOR_REAL_LOCKBOX | p50 | 0.019 | 0.020 | 0.266 |
| DOR_REAL_LOCKBOX | std | 0.158 | 0.172 | 0.166 |
| NON_DOR_REAL_DEV | mean | 0.184 | 0.156 | 0.136 |
| NON_DOR_REAL_DEV | p50 | 0.010 | 0.013 | 0.034 |
| NON_DOR_FAKE_DEV | mean | 0.782 | 0.755 | 0.649 |
| NON_DOR_FAKE_DEV | p50 | 0.994 | 0.989 | 0.745 |

### 4.2 Distribution compression on Dor cohorts (P2D)

P2D vs P8A score distribution width on Dor cohorts (p10–p90 spread):

| cohort | P8A p10–p90 spread | P2D p10–p90 spread | Ratio (P2D/P8A) |
|---|---:|---:|---:|
| DOR_REAL_DEV | 0.013 → 0.905 (Δ 0.892) | 0.262 → 0.676 (Δ 0.414) | 0.46 |
| DOR_FAKE_DEV | 0.889 → 0.994 (Δ 0.105) | 0.330 → 0.659 (Δ 0.329) | 3.13 |
| DOR_REAL_LOCKBOX | 0.007 → 0.268 (Δ 0.262) | 0.100 → 0.528 (Δ 0.428) | 1.63 |

DOR_REAL_DEV: P2D's spread is 0.46× P8A's (compressed).
DOR_FAKE_DEV: P2D's spread is 3.13× P8A's (expanded; P8A's was tight at high values).

### 4.3 P2D bidirectional shift on Dor

- DOR_REAL_DEV mean: 0.331 (P8A) → 0.471 (P2D); +0.140 (reals pushed up).
- DOR_FAKE_DEV mean: 0.945 (P8A) → 0.492 (P2D); −0.453 (fakes pulled down).
- DOR_REAL_DEV p50 vs DOR_FAKE_DEV p50:
  - P8A: real 0.181, fake 0.986; gap 0.805
  - E2B: real 0.223, fake 0.777; gap 0.554
  - P2D: real 0.436, fake 0.489; gap **0.053**

Real and fake Dor median scores converge to 0.05 apart on P2D.

### 4.4 Threshold-crossing fractions

Fraction with score > 0.5 (typical "fake" classifier threshold):

| cohort | P8A | E2B | P2D |
|---|---:|---:|---:|
| DOR_REAL_DEV | 0.300 | 0.240 | 0.360 |
| DOR_FAKE_DEV | 0.987 | 0.821 | 0.474 |
| DOR_REAL_LOCKBOX | 0.040 | 0.040 | 0.140 |
| NON_DOR_REAL_DEV | 0.188 | 0.163 | 0.100 |
| NON_DOR_FAKE_DEV | 0.787 | 0.762 | 0.662 |

Fraction with score > 0.92 (P8A's deployment τ from
`P2_PHASE_A_VERDICT_FACTS_2026-05-08.md`):

| cohort | P8A | E2B | P2D |
|---|---:|---:|---:|
| DOR_REAL_DEV | 0.080 | 0.060 | 0.040 |
| DOR_FAKE_DEV | 0.859 | 0.154 | 0.000 |
| DOR_REAL_LOCKBOX | 0.010 | 0.000 | 0.000 |
| NON_DOR_REAL_DEV | 0.075 | 0.025 | 0.000 |
| NON_DOR_FAKE_DEV | 0.662 | 0.700 | 0.263 |

P2D recall on DOR_FAKE_DEV at τ=0.92 is **zero** (vs P8A 0.86); recall on
NON_DOR_FAKE_DEV is 0.26 (vs P8A 0.66).

---

## 5. Per-frame distance distributions

### 5.1 DOR_REAL_DEV (50 frames; J4's chronic regression)

Per-frame cosine distance to each anchor centroid (median):

| ckpt | d→non_dor_real | d→non_dor_fake | d→dor_fake | d→dor_real |
|---|---:|---:|---:|---:|
| P8A | 0.259 | 0.646 | 0.583 | (own) |
| E2B | 0.458 | 1.503 | 1.419 | (own) |
| P2D | 1.674 | 0.214 | 0.626 | (own) |

P2D's Dor real frames sit far from non-Dor reals (median 1.67), close to
non-Dor fakes (median 0.21), and moderately far from Dor fakes (0.63).

### 5.2 DOR_FAKE_DEV (78 frames; J4's chronic regression)

Per-frame cosine distance medians:

| ckpt | d→non_dor_fake | d→non_dor_real | d→dor_real | d→dor_fake |
|---|---:|---:|---:|---:|
| P8A | 0.050 | 1.325 | 0.980 | (own) |
| E2B | 0.007 | 1.985 | 1.977 | (own) |
| P2D | 1.006 | 0.832 | 0.272 | (own) |

P2D's Dor fake frames sit far from non-Dor fakes (median 1.01) and close
to Dor reals (median 0.27).

---

## 6. IQ-feature × score Pearson r per cohort × ckpt

For each (ckpt, cohort, IQ feature) where n ≥ 10 per cohort. Source:
`outputs/iq_correlation_dor.csv`.

### 6.1 DOR_REAL_DEV

| iq_feature | P8A | E2B | P2D | Δ(P2D − P8A) |
|---|---:|---:|---:|---:|
| color_b_dev | -0.087 | 0.020 | 0.074 | +0.162 |
| edge_mag | -0.040 | -0.256 | -0.477 | -0.437 |
| lap_var | -0.406 | -0.428 | -0.115 | +0.291 |
| luma_mean | 0.037 | 0.095 | 0.127 | +0.090 |
| min_dim | -0.325 | -0.199 | 0.283 | **+0.608** |
| skin_frac | -0.238 | -0.079 | 0.265 | +0.503 |

### 6.2 DOR_FAKE_DEV

| iq_feature | P8A | E2B | P2D | Δ(P2D − P8A) |
|---|---:|---:|---:|---:|
| color_b_dev | 0.175 | -0.335 | -0.107 | -0.282 |
| edge_mag | -0.241 | 0.649 | 0.608 | **+0.849** |
| lap_var | 0.030 | 0.163 | 0.193 | +0.163 |
| luma_mean | 0.007 | 0.530 | 0.516 | +0.509 |
| min_dim | 0.197 | -0.004 | -0.157 | -0.354 |
| skin_frac | -0.029 | 0.082 | 0.064 | +0.093 |

### 6.3 DOR_REAL_LOCKBOX

| iq_feature | P8A | E2B | P2D | Δ(P2D − P8A) |
|---|---:|---:|---:|---:|
| color_b_dev | -0.276 | -0.698 | 0.028 | +0.304 |
| edge_mag | 0.220 | 0.255 | 0.220 | 0.000 |
| lap_var | 0.457 | 0.674 | 0.194 | -0.263 |
| luma_mean | -0.104 | -0.451 | 0.085 | +0.189 |
| min_dim | 0.050 | 0.292 | -0.166 | -0.216 |
| skin_frac | -0.296 | -0.683 | 0.037 | +0.333 |

### 6.4 Direction comparisons

The largest score-vs-IQ Pearson r changes from P8A to P2D, sorted by absolute
delta:

| cohort | iq_feature | P8A | P2D | Δ(P2D − P8A) |
|---|---|---:|---:|---:|
| DOR_FAKE_DEV | edge_mag | -0.241 | 0.608 | +0.849 |
| DOR_REAL_DEV | min_dim | -0.325 | 0.283 | +0.608 |
| DOR_FAKE_DEV | luma_mean | 0.007 | 0.516 | +0.509 |
| DOR_REAL_DEV | skin_frac | -0.238 | 0.265 | +0.503 |
| DOR_REAL_DEV | edge_mag | -0.040 | -0.477 | -0.437 |
| DOR_FAKE_DEV | min_dim | 0.197 | -0.157 | -0.354 |
| DOR_REAL_LOCKBOX | skin_frac | -0.296 | 0.037 | +0.333 |
| DOR_REAL_LOCKBOX | color_b_dev | -0.276 | 0.028 | +0.304 |

---

## 7. Cross-checks against Stage 1 IQ R² findings

`IQ_DECOMP_FACTS_2026-05-08.md` reports per-cell R² for `score ~ IQ_features`
on substrate-level pool groups. Dor was not isolated as a pool group there.
On the LOCKBOX_TEAMS pool group (which contains the 1170 Dor frames), the
table reports:

| ckpt | LOCKBOX_TEAMS R² | LOCKBOX_TEAMS residual AUC |
|---|---:|---:|
| P8A | 0.397 | 0.677 |
| E2B | 0.454 | 0.757 |
| P2D | 0.542 | 0.633 |

Note: the LOCKBOX_TEAMS R² is computed on a mixed cohort
(`teams_real_all_lockbox` + `teams_fake_all_lockbox`), which is dominated
by chronic-6 + non-Dor frames, not by Dor frames alone.

The Dor-cohort-specific Pearson correlations in §6 are scalar (single-feature)
correlations within a single cohort with the other label fixed; they are
NOT directly comparable to a multivariate cross-class R² of `score ~
IQ_features`.

---

## 8. Coverage caveats

1. **DOR_REAL_LOCKBOX subset.** 100 of the 1170 lockbox Dor frames sampled
   (random_state=0). Adequate for centroid estimates but may miss
   sub-population structure within the 1170.

2. **DOR_FAKE_DEV n=78.** Same physical 78 frames are present in both
   `teams_capture_dor_shkedi_dev` and `teams_capture_dor_shkedi_s16_dev`
   (per J4: identical recall numbers). Only `teams_capture_dor_shkedi_dev`
   was sampled as DOR_FAKE_DEV — the s16 variant was not double-counted.

3. **deeplive_dor / deeplive_enhanced.** The triptych sample contains 59
   frames labeled `deeplive_dor` (deeplive_enhanced fakes for Dor identity).
   These were NOT included in the DOR_FAKE_DEV cohort — they belong to the
   `deeplive_enhanced_dev` suite, not `teams_capture_dor_shkedi_dev`. The
   present analysis is on the J4-load-bearing teams_capture cohort only.

4. **Single-ckpt-step.** P2D = P2_D_FOURIER_PERIODIC_STEP3000 only. The other
   P2 D variants (step8000, step19000) were not analyzed; J4 noted their
   Dor real FPR is 0.04 and 0.02 respectively — different regression
   pattern from step3000 (0.46).

5. **Linear-regression IQ correlations.** Pearson r captures linear
   association only; nonlinear IQ–score relationships in the Dor cohort
   would not surface here.

---

## 9. Cross-references

- J4 (per-identity FPR table):
  `analysis/p2_eval_2026-05-08/P2_DEEPER_ANALYSIS_FACTS_2026-05-08.md` §5.
- Stage 1 IQ R² decomposition:
  `analysis/iq_shortcut_decomp_2026-05-08/IQ_DECOMP_FACTS_2026-05-08.md`.
- IQ data atlas: `analysis/iq_data_atlas_2026-05-08/IQ_ATLAS_FACTS_2026-05-08.md`.
- Per-layer IQ probe (check (a)):
  `analysis/iq_perlayer_probe_2026-05-08/IQ_PERLAYER_PROBE_FACTS_2026-05-08.md`.
- P8A signature dor invariance (memory):
  `project_p8a_breakthrough.md`, `project_p18_diagnostics_complete_2026-05-02.md`.
- Per-layer P8A vs E2B divergence (memory):
  `project_per_layer_divergence_2026-05-06.md`.

---

## 10. Artifacts

- `_cache/cohort_features__{P8A,E2B,P2D}.npz` — per-ckpt feature blob with
  `final_cls` (388, 512), `layer11_cls` (388, 768), `valid_mask` (388,).
- `_cache/cohort_manifest.csv` — 388-row manifest with frame_path, label,
  per-ckpt scores, cohort assignment.
- `_cache/frames/` — 388 cached GCS-downloaded crops (~30 MB total).
- `outputs/dor_axis_centroids.csv` — per-(ckpt × layer × cohort) centroid
  norm + dim. 30 rows.
- `outputs/dor_axis_pair_distances.csv` — per-(ckpt × layer × pair)
  cosine + L2 distance. 36 rows.
- `outputs/dor_axis_per_frame.csv` — per-(ckpt × layer × cohort × frame)
  distances + score. ~2,328 rows.
- `outputs/dor_axis_score_stats.csv` — per-(ckpt × cohort) score percentiles.
- `outputs/iq_correlation_dor.csv` — per-(ckpt × cohort × IQ feature)
  Pearson r.
- `outputs/dor_axis_summary.json` — JSON of headline numbers.
