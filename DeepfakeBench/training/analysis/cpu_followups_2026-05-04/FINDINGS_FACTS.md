# CPU follow-ups — FACTS only (2026-05-04)

This document records data observations from the 2026-05-04 CPU analysis pass on
E1/E2b/E3 packets and the visomaster fake suite. **Conclusions and interpretations
are intentionally not included** — they belong in OPINIONS docs / morning briefings.

Source CSVs are in `analysis/cpu_followups_2026-05-04/outputs/`. Source script for
each finding is named.

## Section 1 — Source data and ckpts under analysis

Three checkpoints compared throughout (operational best of each packet):

| Alias | W&B run | Path | Val AUC | Val EER |
|-------|---------|------|--------:|--------:|
| P8A | 9lmvb5b4 | `value_composite_step5000_auc0.9926_eer0.0270.pth` | 0.9926 | 0.0270 |
| E2B_3200 | rmat8lwx | `top_n_step3200_auc0.9863_eer0.0310.pth` | 0.9863 | 0.0310 |
| E3_6600 | jzroefab | `top_n_step6600_auc0.9972_eer0.0093.pth` | 0.9972 | 0.0093 |

Frame-level CSVs joined across all 3 ckpts on `frame_path`. Suite coverage:

| Suite | n_real | n_fake | n_joined |
|-------|-------:|-------:|---------:|
| teams_real_all_dev | 4564 | 0 | 4564 |
| teams_real_poor_quality_dev | 1303 | 0 | 1303 |
| teams_real_lighting_extreme_dev | 1742 | 0 | 1742 |
| teams_real_all_lockbox | 1418 | 0 | 1418 |
| teams_real_dor_dev | 50 | 0 | 50 |
| visomaster_enhanced_macro_dev | 0 | 550 | 550 |
| deeplive_enhanced_dev | 0 | 545 | 545 |
| teams_fake_all_dev | 0 | 3039 | 3039 |
| teams_fake_all_lockbox | 0 | 425 | 425 |

Calibration thresholds at FPR=10% on `teams_real_all_dev`:
- P8A: τ = 0.7052
- E2B_3200: τ = 0.5075
- E3_6600: τ = 0.8526

(τ at FPR=2% are the contract-selected values: P8A 0.991, E2B_3200 0.917, E3_6600 0.994.)

## Section 2 — Cross-suite AUC table

Source: `outputs/02b_cross_suite_auc.csv` (script `02b_cross_suite_auc.py`)
AUC computed by joining each fake suite vs `teams_real_all_dev` as negative class.

| ckpt | viso AUC | deeplive AUC | teams_fake_dev AUC | teams_fake_lockbox AUC |
|------|---------:|-------------:|-------------------:|-----------------------:|
| P8A | 0.7527 | 0.8614 | 0.9106 | 0.8982 |
| E2B_3200 | 0.6815 | 0.9654 | 0.9261 | 0.9479 |
| E3_6600 | 0.7404 | 0.9547 | 0.9229 | 0.9653 |

Mean separation (fake_mean - real_mean) at frame level:

| ckpt | viso gap | deeplive gap | teams_fake_dev gap | teams_fake_lockbox gap |
|------|---------:|-------------:|-------------------:|-----------------------:|
| P8A | +0.222 | +0.388 | +0.610 | +0.514 |
| E2B_3200 | +0.009 | +0.704 | +0.642 | +0.630 |
| E3_6600 | +0.123 | +0.784 | +0.681 | +0.807 |

## Section 3 — Threshold-relaxation curves on viso

Source: `outputs/11_threshold_relaxation_curves.csv` (script `11_threshold_relaxation_curves.py`)
Recall on visomaster_enhanced_macro_dev at multiple FPR floors:

| FPR target | P8A recall | E2B_3200 recall | E3_6600 recall |
|-----------:|-----------:|----------------:|---------------:|
| 0.01 | 0.002 | 0.038 | 0.031 |
| 0.02 | 0.005 | 0.045 | 0.040 |
| 0.05 | 0.058 | 0.049 | 0.078 |
| 0.10 | 0.269 | 0.084 | 0.138 |
| 0.20 | 0.576 | 0.213 | 0.385 |
| 0.30 | 0.689 | 0.398 | 0.875 |
| 0.50 | 0.858 | 0.900 | 0.911 |

## Section 4 — Per-frame score correlation

Source: `outputs/04_per_frame_score_correlation.csv` (script `run_all_analyses.py`)
Pearson r on per-frame scores between ckpts within each suite. Only the fake-only
correlations are shown (real-only also computed in CSV).

| Suite | P8A↔E2B | P8A↔E3 | E2B↔E3 |
|-------|--------:|-------:|-------:|
| visomaster_enhanced_macro_dev | 0.295 | 0.411 | 0.587 |
| deeplive_enhanced_dev | 0.432 | 0.300 | 0.172 |
| teams_fake_all_dev | 0.631 | 0.579 | 0.826 |
| teams_fake_all_lockbox | 0.713 | 0.234 | 0.266 |

Real-only correlations (on `teams_real_all_dev`, n=4564):
- P8A↔E2B: 0.525
- P8A↔E3: 0.375
- E2B↔E3: 0.673

## Section 5 — Frame coverage by ckpt-subset at FPR=10%

Source: `outputs/10_frame_coverage_all_suites.csv` (script `10_frame_coverage_all_suites.py`)
For each fake suite, count of frames caught by each subset of {P8A, E2B_3200, E3_6600}
at the FPR=10% calibrated thresholds (per Section 1).

### visomaster_enhanced_macro_dev (550 fakes)

| caught_by | n_frames | fraction |
|-----------|---------:|---------:|
| uncaught | 364 | 0.662 |
| P8A | 93 | 0.169 |
| E3_6600+P8A | 26 | 0.047 |
| E2B_3200+E3_6600+P8A | 25 | 0.045 |
| E3_6600 | 21 | 0.038 |
| E2B_3200 | 13 | 0.024 |
| E2B_3200+P8A | 4 | 0.007 |
| E2B_3200+E3_6600 | 4 | 0.007 |

### deeplive_enhanced_dev (545 fakes)

| caught_by | n_frames | fraction |
|-----------|---------:|---------:|
| E2B_3200+E3_6600 | 234 | 0.429 |
| E2B_3200+E3_6600+P8A | 229 | 0.420 |
| E2B_3200 | 48 | 0.088 |
| E3_6600 | 31 | 0.057 |
| uncaught | 1 | 0.002 |
| E2B_3200+P8A | 1 | 0.002 |
| E3_6600+P8A | 1 | 0.002 |

### teams_fake_all_dev (3039 fakes)

| caught_by | n_frames | fraction |
|-----------|---------:|---------:|
| E2B_3200+E3_6600+P8A | 1915 | 0.630 |
| uncaught | 387 | 0.127 |
| E2B_3200+E3_6600 | 370 | 0.122 |
| P8A | 96 | 0.032 |
| E2B_3200 | 80 | 0.026 |
| E3_6600 | 78 | 0.026 |
| E3_6600+P8A | 65 | 0.021 |
| E2B_3200+P8A | 48 | 0.016 |

### teams_fake_all_lockbox (425 fakes)

| caught_by | n_frames | fraction |
|-----------|---------:|---------:|
| E2B_3200+E3_6600+P8A | 220 | 0.518 |
| E2B_3200+E3_6600 | 117 | 0.275 |
| E3_6600 | 57 | 0.134 |
| E2B_3200 | 13 | 0.031 |
| uncaught | 8 | 0.019 |
| E3_6600+P8A | 7 | 0.016 |
| E2B_3200+P8A | 2 | 0.005 |
| P8A | 1 | 0.002 |

## Section 6 — Cluster image-quality statistics on visomaster_enhanced_macro_dev

Source: `outputs/14*.csv` (script `14_cluster_statistical_analysis.py`)
Joined viso fakes with `analysis/score_distribution_2026-05-02/outputs/crop_attributes.csv` on filename. Image features available: h, w, luma_mean, luma_std, luma_p10, luma_p90, laplacian_var, sobel_edge_mean, saturation_mean, skin_frac.

Joined size: 550/550 frames.

### 6a. Caught vs uncaught — Welch t-test, sorted by |Cohen's d|

| feature | caught_mean (n=186) | uncaught_mean (n=364) | delta_mean | Cohen's d | p-value |
|---------|--------------------:|----------------------:|-----------:|----------:|--------:|
| luma_p10 | 51.94 | 59.83 | -7.89 | -0.583 | 1.6e-10 |
| laplacian_var | 60.64 | 41.66 | +18.98 | +0.547 | 3.2e-08 |
| sobel_edge_mean | 29.07 | 27.73 | +1.34 | +0.474 | 8.9e-07 |
| luma_p90 | 216.87 | 224.30 | -7.43 | -0.389 | 1.3e-05 |
| luma_mean | 155.68 | 162.09 | -6.41 | -0.344 | 1.1e-04 |
| skin_frac | 0.638 | 0.624 | +0.013 | +0.255 | 1.9e-03 |
| luma_std | (see CSV) | (see CSV) | | | |
| saturation_mean | (see CSV) | (see CSV) | | | |
| h | (see CSV) | (see CSV) | | | |
| w | (see CSV) | (see CSV) | | | |

### 6b. L1 logistic regression — predicting `is_caught` from standardized features

5-fold cross-validation accuracy: **0.705 ± 0.076** (baseline always-uncaught: 0.662).

L1 coefficients (standardized features, sorted by |coef|):

| feature | std_coefficient | direction |
|---------|----------------:|-----------|
| saturation_mean | -0.806 | ↑ feature → ↓ caught |
| luma_p10 | -0.758 | ↑ feature → ↓ caught |
| laplacian_var | +0.739 | ↑ feature → ↑ caught |
| skin_frac | +0.639 | ↑ feature → ↑ caught |
| sobel_edge_mean | -0.348 | ↑ feature → ↓ caught |
| h | +0.315 | ↑ feature → ↑ caught |
| (others zeroed by L1 — see CSV) | | |

Note: sobel_edge_mean has positive Cohen's d in 6a (caught has higher edges) but
negative coefficient in 6b. This is because the L1 logistic conditions on the
correlated features (laplacian_var, h) — sign reflects partial association.

### 6c. Pairwise cluster comparisons — top distinguishing feature per pair

For pairs with both clusters n≥5. Source: `14d_pairwise_cluster_comparison.csv`.

| cluster_a | cluster_b | n_a | n_b | top feature | Cohen's d | p-value |
|-----------|-----------|----:|----:|-------------|----------:|--------:|
| E2B_3200 | E2B_3200+E3_6600+P8A | 13 | 25 | luma_std | +0.96 | 1.0e-02 |
| E2B_3200 | E3_6600 | 13 | 21 | luma_p90 | +5.03 | 8.0e-13 |
| E2B_3200 | E3_6600+P8A | 13 | 26 | luma_p90 | +6.31 | 8.6e-13 |
| E2B_3200 | P8A | 13 | 93 | sobel_edge_mean | -4.19 | 1.4e-26 |
| E2B_3200 | uncaught | 13 | 364 | laplacian_var | -1.39 | 9.8e-35 |
| E2B_3200+E3_6600+P8A | E3_6600 | 25 | 21 | luma_p10 | +1.97 | 2.8e-08 |
| E2B_3200+E3_6600+P8A | E3_6600+P8A | 25 | 26 | luma_mean | +2.10 | 1.7e-08 |
| E2B_3200+E3_6600+P8A | P8A | 25 | 93 | sobel_edge_mean | -4.07 | 2.0e-27 |
| E2B_3200+E3_6600+P8A | uncaught | 25 | 364 | sobel_edge_mean | -1.40 | 5.9e-12 |
| E3_6600 | E3_6600+P8A | 21 | 26 | sobel_edge_mean | -1.10 | 4.3e-04 |
| E3_6600 | P8A | 21 | 93 | sobel_edge_mean | -2.97 | 3.1e-15 |
| E3_6600 | uncaught | 21 | 364 | luma_p10 | -1.60 | 7.6e-10 |
| E3_6600+P8A | P8A | 26 | 93 | luma_p90 | -1.17 | 3.5e-11 |
| E3_6600+P8A | uncaught | 26 | 364 | luma_p10 | -1.59 | 7.1e-14 |
| P8A | uncaught | 93 | 364 | sobel_edge_mean | +1.68 | 2.3e-36 |

All n smaller than 5 omitted (E2B_3200+E3_6600 n=4, E2B_3200+P8A n=4).

### 6d. Subtype breakdown

Viso filenames split into 2 subtypes by string match. Source: `14e_method_subtype_breakdown.csv`.

Counts of frames per (subtype, caught_by_subset):

| subtype | uncaught | P8A | E3_6600+P8A | E2B+E3+P8A | E3_6600 | E2B_3200 | E2B+P8A | E2B+E3 | total |
|---------|---------:|----:|------------:|-----------:|--------:|---------:|--------:|-------:|------:|
| visomaster_enhanced_raw | 156 | 73 | 16 | 13 | 13 | 0 | 4 | 0 | 275 |
| visomaster_enhanced_teams | 208 | 20 | 10 | 12 | 8 | 13 | 0 | 4 | 275 |

Uncaught fractions: raw 156/275 = 0.567; teams 208/275 = 0.756.

## Section 7 — Per-method recall table (E2B_3200, E3_6600 vs P8A)

Source: `outputs/01_per_method_recall.csv` (script `run_all_analyses.py`)
171 rows total (per-suite × per-method × per-FPR-target × per-ckpt).
At FPR=10%, dev fake suites:

| suite × method | P8A recall | E2B_3200 recall | E3_6600 recall |
|----------------|-----------:|----------------:|---------------:|
| visomaster_enhanced_macro_dev × visomaster_enhanced_macro | 0.271 | 0.071 | 0.116 |
| deeplive_enhanced_dev × deeplive_enhanced | 0.424 | 0.939 | 0.908 |
| teams_fake_all_dev × (multiple methods — see CSV) | (varies) | (varies) | (varies) |
| teams_fake_all_lockbox × (multiple methods — see CSV) | (varies) | (varies) | (varies) |

Full per-method table in `01_per_method_recall.csv` (171 rows).

## Section 8 — Per-identity FPR concentration (real suites)

Source: `outputs/07_per_identity_fpr.csv` (script `run_all_analyses.py`)
For each ckpt × real suite, computed at the per-ckpt FPR=10% τ from Section 1:

| ckpt | suite | n_identities | n_offending | fraction_offending | top10pct_share |
|------|-------|-------------:|------------:|-------------------:|---------------:|
| (data in CSV) | | | | | |

(Header for reference; full numbers in CSV.)

## Section 9 — Top per-video disagreements

Source: `outputs/06_per_video_disagreements.csv` (script `run_all_analyses.py`)
200 rows, ranked by max(P8A, E2B_3200, E3_6600 mean-score) - min(...). Per-video
mean of frame scores aggregated within each suite.

## Section 10 — Visualizations generated

PNGs in `figures/` (29 total):
- `score_histogram_<suite>.png` (9): real (green) and fake (red) score densities per ckpt
- `roc_curve_<fake_suite>.png` (4): ROC for the 3 ckpts vs `teams_real_all_dev`
- `recall_vs_fpr_<fake_suite>.png` (4): operational view 0-20% FPR with markers at 2/5/10%
- `scatter_<fake_suite>_<ckpt_a>_vs_<ckpt_b>.png` (12): per-frame scatter, colored by label
- `viso_score_space.png` (1): t-SNE in score-space + (P8A vs max(E2B,E3)) scatter

## Section 11 — Viewer integration

Files added to `viewer/model_dashboard_runs.yaml`:
- `e2b_3200` run entry (B16 SCRATCH+CE+aug operational best)
- `e3_6600` run entry (L14 SCRATCH+CE+aug operational best)
- `viso_uncaught` pseudo-run entry (550 viso fakes browser with 3-ckpt scores)

All three reference `analysis/cpu_followups_2026-05-04/INDEX.html` as
`score_distribution_report` artifact.

Two viewer bugs fixed in `viewer/model_dashboard.py` and `viewer/templates/index.html`:
1. `_load_manifold` did not compose GCS proxy URLs for manifold points (only set image_url for locally-registered frames).
2. Added `caught_by_subset` color option to manifold dropdown + `_color_value` handler.

CSV-loader gotcha: `_coerce_scalar` silently nulls strings "NONE", "NULL", "NaN" (case-insensitive) on CSV read. Worked around by labeling the no-catch group as "uncaught" instead of "NONE".

## Section 12 — Files reference

| Output file | Source script | Description |
|-------------|---------------|-------------|
| `outputs/01_per_method_recall.csv` | `run_all_analyses.py` | per-suite × per-method recall, 171 rows |
| `outputs/02_per_ckpt_per_suite_auc_eer.csv` | `run_all_analyses.py` | per-ckpt × per-suite metrics, 27 rows |
| `outputs/02b_cross_suite_auc.csv` | `02b_cross_suite_auc.py` | proper binary AUC, 12 rows |
| `outputs/03_score_distribution_stats.csv` | `run_all_analyses.py` | mean/std/percentiles per ckpt × suite × label, 27 rows |
| `outputs/04_per_frame_score_correlation.csv` | `run_all_analyses.py` | Pearson r matrix per suite, 27 rows |
| `outputs/05_method_champion.csv` | `run_all_analyses.py` | ckpt winner per (suite, method, FPR) |
| `outputs/06_per_video_disagreements.csv` | `run_all_analyses.py` | top 200 most-divergent videos |
| `outputs/07_per_identity_fpr.csv` | `run_all_analyses.py` | per-identity FPR concentration, 15 rows |
| `outputs/09_deeplive_per_method_recall.csv` | `09_deeplive_deep_dive.py` | deeplive sub-method breakdown |
| `outputs/09b_deeplive_frame_coverage_by_subset.csv` | `09_deeplive_deep_dive.py` | per-deeplive-frame ckpt-subset coverage |
| `outputs/10_frame_coverage_all_suites.csv` | `10_frame_coverage_all_suites.py` | frame coverage by ckpt-subset, all fake suites |
| `outputs/11_threshold_relaxation_curves.csv` | `11_threshold_relaxation_curves.py` | recall at FPR=1/2/5/10/20/30/50% |
| `outputs/14a_cluster_feature_stats.csv` | `14_cluster_statistical_analysis.py` | per-cluster feature mean/std/median |
| `outputs/14b_uncaught_vs_caught_ttest.csv` | `14_cluster_statistical_analysis.py` | Welch t-test + Cohen's d, caught vs uncaught |
| `outputs/14c_logistic_feature_importance.csv` | `14_cluster_statistical_analysis.py` | L1 logistic coefficients |
| `outputs/14d_pairwise_cluster_comparison.csv` | `14_cluster_statistical_analysis.py` | pairwise t-tests across all caught_by clusters |
| `outputs/14e_method_subtype_breakdown.csv` | `14_cluster_statistical_analysis.py` | viso subtype × caught_by counts |
| `outputs/viso_per_frame_with_all_ckpts.csv` | `12_build_viso_browser_artifacts.py` | 550 viso fakes joined with 3-ckpt scores + caught_by |
| `outputs/viso_uncaught_at_fpr10.csv` | `12_build_viso_browser_artifacts.py` | 364 uncaught viso fakes |
| `outputs/viso_score_space_tsne.csv` | `12_build_viso_browser_artifacts.py` | 2D t-SNE on (P8A, E2B, E3) score space, 550 points |
| `viewer_artifacts/viso_uncaught_browse/viso_browse_*_frames_report.csv` | `12_build_viso_browser_artifacts.py` | per-ckpt viso+real frames for viewer browse |

## Section 13 — Methodology notes

- Calibration uses `teams_real_all_dev` real frames as the negative class.
  This matches the contract's `dev_primary_real_fpr` definition.
- All Cohen's d values use pooled standard deviation across the two groups.
- Cross-suite AUC: each fake suite's fake frames are joined against `teams_real_all_dev`'s real frames as a binary task.
- Per-frame scatter Pearson r computed only on frames present in BOTH ckpts' reports (inner join on `frame_path`). Sample sizes match the suite frame counts in Section 1.
- t-SNE in `viso_score_space_tsne.csv` uses 3-vector `(P8A_score, E2B_score, E3_score)` as input, perplexity=30, random_state=42, n_jobs=1 (per `feedback_sklearn_njobs`).
- L1 logistic uses `C=0.5`, `solver=liblinear`, `max_iter=1000`, `random_state=42`. Features standardized via `StandardScaler`. 5-fold CV via `cross_val_score`, default split.
- Image-quality features sourced from existing `analysis/score_distribution_2026-05-02/outputs/crop_attributes.csv` (550 viso fakes, 14 columns). Joined to score data via filename inner-merge.

## Section 14 — Cross-references

- Morning briefing: `docs/relaunch_handoffs/MORNING_BRIEFING_2026-05-04.md` (interpretive)
- Per-packet verdicts: `docs/relaunch_handoffs/E_PACKET_VERDICT_2026-05-03.md` (E1), `E2B_FINAL_VERDICT_2026-05-04.md`, `E3_L14_FINAL_VERDICT_2026-05-04.md`
- Handoff for next agent: `docs/relaunch_handoffs/HANDOFF_NEXT_STEPS_2026-05-04.md`
- Pre-existing FACTS doc (broader R-chain): `docs/relaunch_handoffs/PSERIES_FACTS_2026-05-02.md`
- Pre-existing OPINIONS doc: `docs/relaunch_handoffs/PSERIES_OPINIONS_2026-05-02.md`
- Viewer-served HTML report: `analysis/cpu_followups_2026-05-04/INDEX.html`
- Memory entries (`/Users/roeedar/.claude/projects/.../memory/`):
  - `project_e2b_breaks_deeplive_ceiling.md`
  - `project_l14_does_not_break_viso_ceiling.md`
  - `project_viso_ceiling_unbroken_10_packets.md` (now 13+ packets)
  - `project_image_quality_shortcut.md` (pre-existing)
  - `project_eval_production_crop_tightness_gap.md` (pre-existing)
  - `project_face_size_label_leak.md` (pre-existing)
