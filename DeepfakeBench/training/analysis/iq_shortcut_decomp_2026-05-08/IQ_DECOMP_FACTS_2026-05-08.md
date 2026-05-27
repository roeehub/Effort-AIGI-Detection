# IQ-shortcut R² decomposition — FACTS (2026-05-08)

> **Status: factual-only.** No interpretation. Forbidden words: succeeds, fails,
> wins, promotes, deployment-grade.
>
> **Stage 1** of the IQ-shortcut deconvolution program (proposal thread:
> `docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md`).
> CPU-only.

## 1. Method

Per the proposal §4.1, for each (ckpt × pool-group) cell the driver
(`decompose.py`) fits an OLS regression on standardized IQ features with
target = per-frame model score, then reports R² and the AUC of the residual.

- **Score source**: per-frame `frame_prob` from
  `gs://training-job-outputs/test_results/teams_promotion_contract/p2-scratch-scorecard-2026-05-08/reports/` (Phase A) and
  `…/p1-pe-hdtf-scorecard-2026-05-07/reports/` (Phase C HDTF).
- **IQ feature source**: `analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet`
  (15,236 frames × 13 IQ features sampled at N=500 per pool).
- **Joiner**: inner join on `(atlas_pool, frame_path)`. Atlas is a 500-frame
  sample per pool; the join keeps only rows that have both an IQ feature panel
  and a Phase-A/C model score.
- **Regression**: `sklearn.linear_model.LinearRegression(n_jobs=1)`. Feature
  set "primary_6" = `[lap_var, min_dim, luma_mean, color_b_dev, edge_mag, skin_frac]`.
  Sensitivity-check feature set "expanded_10" adds
  `[color_a_dev, luma_std, contrast_l, saturation_mean]`. R² and residual AUC
  reported below are from "primary_6"; expanded_10 numbers in §6.
- **Residual AUC**: AUC of `(score − OLS-predicted score)` against the binary
  fake/real label, computed only on cells where both classes are present.
  This is the proposal's "content-channel-only predictor" estimate.
- **Pool-groups** (substrate-coherent, both classes present per cell):

| pool_group | atlas pools | n_real (atlas) | n_fake (atlas) |
|---|---|---:|---:|
| `DEV_TEAMS_PRIMARY` | `teams_real_all_dev` + `teams_fake_all_dev` | 500 | 500 |
| `DEV_DEEPLIVE_VS_REAL` | `teams_real_all_dev` + `deeplive_enhanced_dev` | 500 | 500 |
| `DEV_VISO_VS_REAL` | `teams_real_all_dev` + `visomaster_enhanced_macro_dev` | 500 | 500 |
| `DEV_TEAMS_STRESS_VS_FAKE` | `teams_real_lighting_extreme_dev` + `teams_real_poor_quality_dev` + `teams_fake_all_dev` | 1000 | 500 |
| `LOCKBOX_TEAMS` | `teams_real_all_lockbox` + `teams_fake_all_lockbox` | 500 | 425 |
| `HDTF_CLEAN_DEV` | `hdtf_real_clean_dev` + `hdtf_fake_clean_dev` | 250 | 252 |
| `HDTF_CLEAN_LOCKBOX` | `hdtf_real_clean_lockbox` + `hdtf_fake_clean_lockbox` | 243 | 247 |
| `HDTF_TEAMS_DEV` | `hdtf_real_teams_dev` + `hdtf_fake_teams_dev` | 237 | 247 |
| `HDTF_TEAMS_LOCKBOX` | `hdtf_real_teams_lockbox` + `hdtf_fake_teams_lockbox` | 249 | 242 |

**Coverage caveat**: the Phase C (HDTF) scorecard ran for P8A and E2B only.
P2_D_FOURIER_PERIODIC_STEP3000 has no HDTF per-frame scores (no Phase C run for
P2 as of 2026-05-08), so its HDTF cells are absent.

**Total cells**: 23 (P8A 9 cells + E2B 9 cells + P2-D 5 cells, primary_6).

## 2. Headline table — R² and residual AUC

Per (ckpt × pool-group). `r2` is the OLS R² of `score ~ IQ_features`.
`raw_auc` is fake-vs-real AUC on the raw `frame_prob`. `resid_auc` is
fake-vs-real AUC on `(frame_prob − OLS-predicted frame_prob)`.

### 2.1 P8A_REFERENCE_STEP5000

| pool_group | n | n_real | n_fake | R² | raw AUC | residual AUC | Δ AUC (raw − resid) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DEV_TEAMS_PRIMARY | 1000 | 500 | 500 | 0.180 | 0.923 | 0.858 | 0.066 |
| DEV_DEEPLIVE_VS_REAL | 1000 | 500 | 500 | 0.251 | 0.890 | 0.567 | 0.323 |
| DEV_VISO_VS_REAL | 1000 | 500 | 500 | 0.119 | 0.785 | 0.599 | 0.186 |
| DEV_TEAMS_STRESS_VS_FAKE | 1500 | 1000 | 500 | 0.239 | 0.908 | 0.827 | 0.082 |
| LOCKBOX_TEAMS | 925 | 500 | 425 | 0.397 | 0.918 | 0.677 | 0.241 |
| HDTF_CLEAN_DEV | 502 | 250 | 252 | 0.035 | 0.996 | 0.977 | 0.019 |
| HDTF_CLEAN_LOCKBOX | 490 | 243 | 247 | 0.073 | 1.000 | 0.994 | 0.006 |
| HDTF_TEAMS_DEV | 484 | 237 | 247 | 0.016 | 0.988 | 0.963 | 0.025 |
| HDTF_TEAMS_LOCKBOX | 491 | 249 | 242 | 0.037 | 0.996 | 0.986 | 0.010 |

### 2.2 E2B_TOP_N_STEP3200

| pool_group | n | n_real | n_fake | R² | raw AUC | residual AUC | Δ AUC (raw − resid) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DEV_TEAMS_PRIMARY | 1000 | 500 | 500 | 0.163 | 0.929 | 0.854 | 0.075 |
| DEV_DEEPLIVE_VS_REAL | 1000 | 500 | 500 | 0.592 | 0.970 | 0.737 | 0.233 |
| DEV_VISO_VS_REAL | 1000 | 500 | 500 | 0.020 | 0.708 | 0.594 | 0.114 |
| DEV_TEAMS_STRESS_VS_FAKE | 1500 | 1000 | 500 | 0.169 | 0.899 | 0.824 | 0.075 |
| LOCKBOX_TEAMS | 925 | 500 | 425 | 0.454 | 0.965 | 0.757 | 0.208 |
| HDTF_CLEAN_DEV | 502 | 250 | 252 | 0.035 | 0.994 | 0.970 | 0.024 |
| HDTF_CLEAN_LOCKBOX | 490 | 243 | 247 | 0.064 | 0.995 | 0.960 | 0.035 |
| HDTF_TEAMS_DEV | 484 | 237 | 247 | 0.090 | 0.844 | 0.722 | 0.122 |
| HDTF_TEAMS_LOCKBOX | 491 | 249 | 242 | 0.059 | 0.805 | 0.714 | 0.091 |

### 2.3 P2_D_FOURIER_PERIODIC_STEP3000

| pool_group | n | n_real | n_fake | R² | raw AUC | residual AUC | Δ AUC (raw − resid) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DEV_TEAMS_PRIMARY | 1000 | 500 | 500 | 0.261 | 0.951 | 0.881 | 0.070 |
| DEV_DEEPLIVE_VS_REAL | 1000 | 500 | 500 | 0.527 | 0.958 | 0.720 | 0.237 |
| DEV_VISO_VS_REAL | 1000 | 500 | 500 | 0.142 | 0.871 | 0.680 | 0.190 |
| DEV_TEAMS_STRESS_VS_FAKE | 1500 | 1000 | 500 | 0.266 | 0.919 | 0.859 | 0.060 |
| LOCKBOX_TEAMS | 925 | 500 | 425 | 0.542 | 0.923 | 0.633 | 0.290 |

### 2.4 HDTF cells — UNIFIED 2026-05-08 evening run (`add_p2d_hdtf_2026-05-08.py`)

After the `p2-d-step3000-hdtf-2026-05-08` Vertex run landed (2026-05-08 PM,
recovered locally per `analysis/p2_d_hdtf_2026-05-08/P2_D_HDTF_FACTS_2026-05-08.md`),
the 4 HDTF pool groups are recomputable for all 3 ckpts on a single
2026-05-08 source. The earlier P8A/E2B HDTF rows in §2.1-2.2 were sourced
from the 2026-05-07 P1 PE HDTF run and are kept in those tables for
historical reference. The unified table below is the consistent
2026-05-08 measurement.

Source: `outputs/iq_decomp_hdtf_unified_2026-05-08.csv` (24 rows × 2 feature
sets). Joined panel size: 5,901 frames after inner join with atlas.

#### R² (primary_6)

| pool_group | P8A | E2B | P2D |
|---|---:|---:|---:|
| HDTF_CLEAN_DEV | 0.035 | 0.035 | 0.036 |
| HDTF_CLEAN_LOCKBOX | 0.073 | 0.064 | 0.052 |
| HDTF_TEAMS_DEV | 0.016 | 0.090 | 0.034 |
| HDTF_TEAMS_LOCKBOX | 0.037 | 0.059 | 0.035 |

#### Residual AUC (primary_6)

| pool_group | P8A | E2B | P2D |
|---|---:|---:|---:|
| HDTF_CLEAN_DEV | 0.977 | 0.970 | 0.987 |
| HDTF_CLEAN_LOCKBOX | 0.994 | 0.960 | 0.989 |
| HDTF_TEAMS_DEV | 0.963 | 0.722 | 0.800 |
| HDTF_TEAMS_LOCKBOX | 0.986 | 0.714 | 0.737 |

#### Direct observations (factual; no interpretation)

1. P2D's HDTF R² (0.034-0.052) is in the same range as P8A's
   (0.016-0.073). It is NOT materially higher than P8A's HDTF R² on any
   of the 4 cells.
2. P2D's HDTF residual AUC on the 4 cells: 0.737-0.989. P2D's residual
   AUC exceeds E2B's on all 4 HDTF cells; matches or exceeds P8A's on
   the 2 clean cells (0.987 vs 0.977; 0.989 vs 0.994).
3. On the 2 HDTF teams-transport cells, P2D's residual AUC is
   0.737-0.800; P8A's is 0.963-0.986. P2D residual AUC < P8A residual
   AUC by 0.16-0.25.
4. The HDTF macro_fake_recall pattern from `P2_D_HDTF_FACTS_2026-05-08.md`
   §3.3 (P8A 0.896, E2B 0.484, P2D 0.540 at Phase A contract τ) does NOT
   match the HDTF residual AUC ordering (P8A ~ P2D > E2B on clean;
   P8A > P2D > E2B on teams).

## 3. R² and Δ AUC summary statistics by ckpt and group-class

Group-classes: DEV_* (excl. STRESS), STRESS, LOCKBOX, HDTF_*.

### 3.1 R² by group-class

| ckpt | DEV (3 cells) | STRESS (1) | LOCKBOX (1) | HDTF (4 / 0) |
|---|:---:|:---:|:---:|:---:|
| P8A_REFERENCE | min 0.119 / median 0.180 / max 0.251 | 0.239 | **0.397** | min 0.016 / median 0.036 / max 0.073 |
| E2B_TOP_N | min 0.020 / median 0.163 / max **0.592** | 0.169 | **0.454** | min 0.035 / median 0.061 / max 0.090 |
| P2_D_FOURIER_periodic_3000 | min 0.142 / median 0.261 / max 0.527 | 0.266 | **0.542** | n/a |

### 3.2 Residual AUC by group-class

| ckpt | DEV (3 cells, median) | STRESS | LOCKBOX | HDTF (4 cells, median) |
|---|:---:|:---:|:---:|:---:|
| P8A_REFERENCE | 0.599 | 0.827 | 0.677 | **0.982** |
| E2B_TOP_N | 0.737 | 0.824 | 0.757 | **0.846** |
| P2_D_FOURIER_periodic_3000 | 0.720 | 0.859 | 0.633 | n/a |

### 3.3 Δ AUC (raw − residual) by group-class

| ckpt | DEV (3 cells, median) | STRESS | LOCKBOX | HDTF (4 cells, median) |
|---|:---:|:---:|:---:|:---:|
| P8A_REFERENCE | 0.186 | 0.082 | **0.241** | 0.014 |
| E2B_TOP_N | 0.114 | 0.075 | **0.208** | 0.030 |
| P2_D_FOURIER_periodic_3000 | 0.190 | 0.060 | **0.290** | n/a |

## 4. Top three IQ axes by |β · σ_x| per cell

Standardized contribution: each coefficient times the feature's standard
deviation in the joined panel, signed. Positive ⇒ IQ axis pushes score upward;
negative ⇒ downward. Multiple IQ features are correlated, so partial
coefficients in the multivariate regression can take signs that differ from
the marginal direction.

| ckpt | pool_group | top 1 (β·σ) | top 2 | top 3 |
|---|---|---|---|---|
| P8A_REFERENCE | DEV_TEAMS_PRIMARY | skin_frac (+0.146) | luma_mean (−0.129) | color_b_dev (−0.091) |
| P8A_REFERENCE | DEV_DEEPLIVE_VS_REAL | min_dim (+0.127) | luma_mean (−0.114) | edge_mag (+0.090) |
| P8A_REFERENCE | DEV_VISO_VS_REAL | edge_mag (+0.129) | min_dim (+0.123) | color_b_dev (−0.087) |
| P8A_REFERENCE | DEV_TEAMS_STRESS_VS_FAKE | luma_mean (−0.162) | skin_frac (+0.126) | color_b_dev (−0.088) |
| P8A_REFERENCE | LOCKBOX_TEAMS | edge_mag (−0.647) | min_dim (−0.549) | lap_var (+0.159) |
| P8A_REFERENCE | HDTF_CLEAN_DEV | edge_mag (+0.126) | lap_var (−0.090) | luma_mean (−0.080) |
| P8A_REFERENCE | HDTF_CLEAN_LOCKBOX | edge_mag (+0.217) | lap_var (−0.181) | luma_mean (−0.073) |
| P8A_REFERENCE | HDTF_TEAMS_DEV | lap_var (+0.144) | color_b_dev (+0.047) | skin_frac (−0.046) |
| P8A_REFERENCE | HDTF_TEAMS_LOCKBOX | color_b_dev (+0.096) | min_dim (−0.060) | skin_frac (−0.045) |
| E2B_TOP_N | DEV_TEAMS_PRIMARY | luma_mean (−0.139) | edge_mag (−0.127) | color_b_dev (−0.090) |
| E2B_TOP_N | DEV_DEEPLIVE_VS_REAL | min_dim (+0.208) | edge_mag (+0.113) | luma_mean (−0.093) |
| E2B_TOP_N | DEV_VISO_VS_REAL | luma_mean (+0.039) | min_dim (−0.030) | skin_frac (−0.018) |
| E2B_TOP_N | DEV_TEAMS_STRESS_VS_FAKE | luma_mean (−0.138) | edge_mag (−0.108) | color_b_dev (−0.073) |
| E2B_TOP_N | LOCKBOX_TEAMS | edge_mag (−0.652) | min_dim (−0.419) | lap_var (+0.230) |
| E2B_TOP_N | HDTF_CLEAN_DEV | edge_mag (+0.111) | lap_var (−0.096) | luma_mean (−0.078) |
| E2B_TOP_N | HDTF_CLEAN_LOCKBOX | edge_mag (+0.184) | lap_var (−0.165) | luma_mean (−0.057) |
| E2B_TOP_N | HDTF_TEAMS_DEV | lap_var (−0.065) | min_dim (−0.063) | skin_frac (−0.057) |
| E2B_TOP_N | HDTF_TEAMS_LOCKBOX | skin_frac (−0.053) | color_b_dev (+0.038) | min_dim (−0.023) |
| P2_D_FOURIER_periodic_3000 | DEV_TEAMS_PRIMARY | skin_frac (+0.123) | luma_mean (−0.121) | edge_mag (−0.112) |
| P2_D_FOURIER_periodic_3000 | DEV_DEEPLIVE_VS_REAL | min_dim (+0.129) | luma_mean (−0.077) | color_b_dev (−0.058) |
| P2_D_FOURIER_periodic_3000 | DEV_VISO_VS_REAL | min_dim (+0.056) | edge_mag (+0.040) | luma_mean (+0.037) |
| P2_D_FOURIER_periodic_3000 | DEV_TEAMS_STRESS_VS_FAKE | luma_mean (−0.134) | skin_frac (+0.105) | edge_mag (−0.079) |
| P2_D_FOURIER_periodic_3000 | LOCKBOX_TEAMS | edge_mag (−0.225) | min_dim (−0.129) | color_b_dev (−0.110) |

## 5. Score distribution sanity check

Per-cell median raw scores (real vs fake) — confirms the joined panel's
score distribution matches the Phase A scorecard headlines.

| ckpt | pool_group | p50 score (real) | p50 score (fake) |
|---|---|---:|---:|
| P8A_REFERENCE | DEV_TEAMS_PRIMARY | 0.0066 | 0.9825 |
| P8A_REFERENCE | LOCKBOX_TEAMS | 0.0149 | 0.7882 |
| P8A_REFERENCE | HDTF_CLEAN_DEV | 0.0054 | 0.9945 |
| P8A_REFERENCE | HDTF_TEAMS_DEV | 0.0056 | 0.9942 |
| E2B_TOP_N | DEV_TEAMS_PRIMARY | 0.0070 | 0.9612 |
| E2B_TOP_N | DEV_DEEPLIVE_VS_REAL | 0.0070 | 0.8975 |
| E2B_TOP_N | DEV_VISO_VS_REAL | 0.0070 | 0.0258 |
| E2B_TOP_N | LOCKBOX_TEAMS | 0.0174 | 0.8273 |
| E2B_TOP_N | HDTF_TEAMS_DEV | 0.0057 | 0.0205 |
| P2_D_FOURIER_periodic_3000 | DEV_TEAMS_PRIMARY | 0.0265 | 0.6165 |
| P2_D_FOURIER_periodic_3000 | LOCKBOX_TEAMS | 0.2347 | 0.7241 |

## 6. Sensitivity check — expanded_10 feature set

Same analysis with the 10-feature set
`[lap_var, min_dim, luma_mean, color_b_dev, edge_mag, skin_frac, color_a_dev,
luma_std, contrast_l, saturation_mean]`.

| ckpt | pool_group | R² (p6) | R² (e10) | resid AUC (p6) | resid AUC (e10) | ΔR² | Δresid AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| P8A_REFERENCE | DEV_TEAMS_PRIMARY | 0.180 | 0.199 | 0.858 | 0.858 | +0.018 | +0.001 |
| P8A_REFERENCE | DEV_DEEPLIVE_VS_REAL | 0.251 | 0.301 | 0.567 | 0.548 | +0.050 | −0.019 |
| P8A_REFERENCE | DEV_VISO_VS_REAL | 0.119 | 0.177 | 0.599 | 0.510 | +0.058 | −0.089 |
| P8A_REFERENCE | DEV_TEAMS_STRESS_VS_FAKE | 0.239 | 0.245 | 0.827 | 0.816 | +0.006 | −0.011 |
| P8A_REFERENCE | LOCKBOX_TEAMS | **0.397** | **0.534** | 0.677 | 0.631 | **+0.137** | −0.046 |
| P8A_REFERENCE | HDTF_CLEAN_DEV | 0.035 | 0.061 | 0.977 | 0.978 | +0.026 | +0.002 |
| P8A_REFERENCE | HDTF_CLEAN_LOCKBOX | 0.073 | 0.085 | 0.994 | 0.994 | +0.012 | +0.000 |
| P8A_REFERENCE | HDTF_TEAMS_DEV | 0.016 | 0.029 | 0.963 | 0.960 | +0.014 | −0.003 |
| P8A_REFERENCE | HDTF_TEAMS_LOCKBOX | 0.037 | 0.041 | 0.986 | 0.985 | +0.004 | −0.001 |
| E2B_TOP_N | DEV_TEAMS_PRIMARY | 0.163 | 0.196 | 0.854 | 0.871 | +0.033 | +0.017 |
| E2B_TOP_N | DEV_DEEPLIVE_VS_REAL | 0.592 | 0.660 | 0.737 | 0.712 | +0.068 | −0.025 |
| E2B_TOP_N | DEV_VISO_VS_REAL | 0.020 | 0.055 | 0.594 | 0.566 | +0.034 | −0.029 |
| E2B_TOP_N | DEV_TEAMS_STRESS_VS_FAKE | 0.169 | 0.205 | 0.824 | 0.822 | +0.036 | −0.003 |
| E2B_TOP_N | LOCKBOX_TEAMS | **0.454** | **0.614** | 0.757 | 0.689 | **+0.160** | −0.068 |
| E2B_TOP_N | HDTF_CLEAN_DEV | 0.035 | 0.069 | 0.970 | 0.967 | +0.034 | −0.003 |
| E2B_TOP_N | HDTF_CLEAN_LOCKBOX | 0.064 | 0.089 | 0.960 | 0.956 | +0.025 | −0.004 |
| E2B_TOP_N | HDTF_TEAMS_DEV | 0.090 | 0.096 | 0.722 | 0.709 | +0.006 | −0.013 |
| E2B_TOP_N | HDTF_TEAMS_LOCKBOX | 0.059 | 0.070 | 0.714 | 0.718 | +0.011 | +0.004 |
| P2_D_FOURIER_3000 | DEV_TEAMS_PRIMARY | 0.261 | 0.269 | 0.881 | 0.875 | +0.008 | −0.006 |
| P2_D_FOURIER_3000 | DEV_DEEPLIVE_VS_REAL | 0.527 | 0.602 | 0.720 | 0.691 | +0.075 | −0.029 |
| P2_D_FOURIER_3000 | DEV_VISO_VS_REAL | 0.142 | 0.206 | 0.680 | 0.617 | +0.063 | −0.064 |
| P2_D_FOURIER_3000 | DEV_TEAMS_STRESS_VS_FAKE | 0.266 | 0.303 | 0.859 | 0.854 | +0.037 | −0.005 |
| P2_D_FOURIER_3000 | LOCKBOX_TEAMS | **0.542** | **0.569** | 0.633 | 0.607 | **+0.027** | −0.026 |

Across all 23 cells: max ΔR² = +0.160 (E2B LOCKBOX_TEAMS); mean ΔR² = +0.041;
max |Δresid AUC| = 0.089 (P8A DEV_VISO_VS_REAL); mean |Δresid AUC| = 0.020.
The qualitative HDTF-low / LOCKBOX-high R² structure is preserved across both
feature sets.

## 7. Artifacts

- `decompose.py` — driver (idempotent; cache at `scores_cache/`).
- `outputs/iq_decomp.csv` — 46 rows: 23 cells × 2 feature sets.
- `outputs/iq_decomp.json` — same with full coefficient vectors.
- `outputs/joined_panel.parquet` — per-frame joined panel
  (rows = ckpt × atlas_pool × frame_path; 15,859 rows).
- `outputs/per_frame_residuals.parquet` — per-frame `(frame_path, ckpt, pool_group, frame_prob, score_resid, label)` (20,209 rows). Reusable for downstream per-frame analyses (e.g. residual heatmaps in the viewer).
- `scores_cache/{ckpt}__{atlas_pool}.csv` — 43 cached frame-level reports.

## 8. Cross-references

- Proposal: `docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md` §4.1.
- Atlas: `analysis/iq_data_atlas_2026-05-08/IQ_ATLAS_FACTS_2026-05-08.md`.
- P2 verdict: `analysis/p2_eval_2026-05-08/P2_PHASE_A_VERDICT_FACTS_2026-05-08.md`.
- Memory: `project_image_quality_shortcut.md`,
  `project_dor_drift_named_axes_2026-05-06.md`,
  `project_iq_gating_viability_2026-05-04.md`.
