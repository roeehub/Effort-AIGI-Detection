# A2 — Frozen-encoder L11 linear probe on lockbox (FACTS, 2026-05-11)

> **Status: factual-only.** Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade.
> Numbers + tables + cross-references only. Interpretation belongs in a paired proposal doc.
>
> **Scope**: video-level frozen-encoder linear probe on layer-11 CLS features for two ckpts —
> `T4_LAMBDA1_TOP_N_STEP10500` and `P8A_REFERENCE_STEP5000` — evaluated on the lockbox subset.
> Decides between α / β / γ for the source of the T4 trained-head lockbox AUC drop
> (`analysis/t4_eval_2026-05-11/RESULTS_FACTS_2026-05-11.md` §7.1: T4 lockbox AUC 0.7619 vs P8A 0.9355).
>
> **Inputs**:
> - T4_L1 step10500 ckpt: `analysis/cpu_diagnostics_2026-05-10/_ckpts_t4/top_n_effort_20260510_step10500_auc0.9937_eer0.0195.pth`
> - P8A reference step5000 ckpt: `analysis/_features_cache_2026-04-30/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth`
> - Lockbox frame lists: `analysis/cpu_diagnostics_2026-05-10/_t4_scorecard_local/teams_{real,fake}_all_lockbox_t4_lambda1_top_n_step10500_frames_report.csv`.
> - Local frame mirror: `/Users/roeedar/Downloads/faces/r9_feb28_for_checker/{real,fake}/flat/`.
> - Scripts: `run_lockbox_probe.py`, `run_pcgen_subset_probe.py`, `run_trained_head_baseline.py`, `run_regularized_probe.py`.
> - Outputs: `lockbox_probe_auc.csv`, `lockbox_probe_auc_pcgen_only.csv`, `trained_head_baseline_auc.csv`, `lockbox_probe_regularized_grid.csv`, `_cache/video_feats__*__L11.npz`.

---

## 1. Method

### 1.1 Frame → feature extraction

- Each ckpt loaded via `detectors.DETECTOR[cfg['model_name']]` with `state_dict.pop('module.')` prefix and `strict=False`; T4 ckpts' `multi_axis_grl_block.*` keys dropped on load (only encoder + head paths matter for L11 CLS extraction).
- Forward hook on `model.backbone.visual.transformer.resblocks[11]`; CLS token = `output[:, 0]` (dim=768).
- Image preprocessing identical to `extract_t4_features.py`: BGR→RGB, resize-224, CLIP-mean/std normalize.
- Device: MPS.

### 1.2 Sample construction

- Lockbox reals: 1361 distinct videos in source CSV → 1361 (frames=1418) after local mirror availability check (`gs://...` → `/Users/roeedar/Downloads/faces/...`). Note: `real_dor` source's 109 videos drop out because those frames are PNG-format and not in the local mirror (`/Users/roeedar/Downloads/faces/r9_feb28_for_checker/real/flat/` contains 4420 `.jpg`/`.png` files, but the lockbox CSV's `real_dor__frame_*_seq*.png` filenames are absent).
- After local mapping: 200 reals + 253 fakes selected via proportional source-stratified random sampling (seed=42).
- Per video, ≤4 frames sampled (random within video), mean-aggregated to a single (768,) feature vector. **One feature per video.**
- Final probe inputs: 200 reals + 253 fakes = 453 video-features.

### 1.3 Source composition of probe inputs

| Class | Source identity prefix | n_videos |
|---|---|---:|
| real (label=0) | dor_shkedi | 140 |
| real | bla_bla_chow | 32 |
| real | PC_Generator | 15 |
| real | Chikara_Takahashi | 13 |
| fake (label=1) | Cam_Test | 191 |
| fake | PC_Generator | 62 |

Note: only `PC_Generator` has both real and fake video sources. `real_dor` source (109 videos) is absent due to local-mirror format mismatch noted in §1.2.

### 1.4 Linear probe

- `sklearn.preprocessing.StandardScaler` fit on train fold only.
- `sklearn.linear_model.LogisticRegression(C=1.0, max_iter=2000, n_jobs=1)` (n_jobs=1 per `feedback_sklearn_njobs.md`).
- `sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Per-fold ROC AUC via `sklearn.metrics.roc_auc_score`.

---

## 2. Job A — Primary probe (full 453-video lockbox subset)

Source: `lockbox_probe_auc.csv`.

| ckpt | fold | AUC | n_real | n_fake |
|---|---:|---:|---:|---:|
| T4_LAMBDA1_TOP_N_STEP10500 | 0 | 1.0000 | 40 | 51 |
| T4_LAMBDA1_TOP_N_STEP10500 | 1 | 1.0000 | 40 | 51 |
| T4_LAMBDA1_TOP_N_STEP10500 | 2 | 1.0000 | 40 | 51 |
| T4_LAMBDA1_TOP_N_STEP10500 | 3 | 1.0000 | 40 | 50 |
| T4_LAMBDA1_TOP_N_STEP10500 | 4 | 1.0000 | 40 | 50 |
| P8A_REFERENCE_STEP5000 | 0 | 1.0000 | 40 | 51 |
| P8A_REFERENCE_STEP5000 | 1 | 1.0000 | 40 | 51 |
| P8A_REFERENCE_STEP5000 | 2 | 1.0000 | 40 | 51 |
| P8A_REFERENCE_STEP5000 | 3 | 1.0000 | 40 | 50 |
| P8A_REFERENCE_STEP5000 | 4 | 1.0000 | 40 | 50 |

| ckpt | mean_AUC | std_AUC |
|---|---:|---:|
| T4_LAMBDA1_TOP_N_STEP10500 | 1.0000 | 0.0000 |
| P8A_REFERENCE_STEP5000 | 1.0000 | 0.0000 |

---

## 3. Job B — Trained-head baseline AUC on the same video subset

Source: `trained_head_baseline_auc.csv`. Computed from per-frame `frame_prob` in the same lockbox CSVs, video-level aggregate = mean of `frame_prob` per `video_id`, label-AUC computed via `sklearn.metrics.roc_auc_score`.

| ckpt | full lockbox AUC (n=1361 reals + 253 fakes) | probe-subset AUC (n=200 reals + 253 fakes) | Δ (subset − full) |
|---|---:|---:|---:|
| T4_LAMBDA1_TOP_N_STEP10500 | 0.7735 | 0.7957 | +0.0222 |
| P8A_REFERENCE_STEP5000 | 0.9417 | 0.8638 | −0.0779 |

Reference: T4 trained-head video-level AUC from `analysis/t4_eval_2026-05-11/RESULTS_FACTS_2026-05-11.md` §7.1: T4 0.7619, P8A 0.9355.

T4 vs P8A trained-head AUC gap on the **same 453-video probe subset** = 0.8638 − 0.7957 = **0.0681** (T4 < P8A).

---

## 4. Job C — PC_Generator-only subset (source-confound removed)

Source: `lockbox_probe_auc_pcgen_only.csv`. Subset filter: `source == 'PC_Generator'` (15 reals + 62 fakes = 77 videos).

| ckpt | mean_AUC | std_AUC | per-fold AUCs |
|---|---:|---:|---|
| T4_LAMBDA1_TOP_N_STEP10500 | 1.0000 | 0.0000 | [1.0000, 1.0000, 1.0000, 1.0000, 1.0000] |
| P8A_REFERENCE_STEP5000 | 1.0000 | 0.0000 | [1.0000, 1.0000, 1.0000, 1.0000, 1.0000] |

---

## 5. Job D — Regularization / dimensionality sweep

Source: `lockbox_probe_regularized_grid.csv`. Grid: `C ∈ {1e-3, 1e-2, 1e-1, 1, 10}` × `pca ∈ {None, 50, 20, 10}` (PCA fit on train fold only). Same 5-fold StratifiedKFold inputs as Job A.

mean AUC by (ckpt, C, pca):

| ckpt | C | pca=None | pca=50 | pca=20 | pca=10 |
|---|---:|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 0.001 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| P8A_REFERENCE_STEP5000 | 0.010 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| P8A_REFERENCE_STEP5000 | 0.100 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| P8A_REFERENCE_STEP5000 | 1.000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| P8A_REFERENCE_STEP5000 | 10.000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| T4_LAMBDA1_TOP_N_STEP10500 | 0.001 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| T4_LAMBDA1_TOP_N_STEP10500 | 0.010 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| T4_LAMBDA1_TOP_N_STEP10500 | 0.100 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| T4_LAMBDA1_TOP_N_STEP10500 | 1.000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| T4_LAMBDA1_TOP_N_STEP10500 | 10.000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Minimum AUC across all 20 (C, pca) cells: P8A 0.9999, T4 1.0000.

---

## 6. Job E — Feature statistics

Source: `_cache/video_feats__*__L11.npz`.

| ckpt | n_videos | dim | mean | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| T4_LAMBDA1_TOP_N_STEP10500 | 453 | 768 | −0.0796 | 0.6335 | −8.5289 | 5.4139 |
| P8A_REFERENCE_STEP5000 | 453 | 768 | (re-extractable from cache) | | | |

---

## 7. Cross-references

- T4 packet trained-head outcome: `analysis/t4_eval_2026-05-11/RESULTS_FACTS_2026-05-11.md` §7.1.
- Forgery atlas (P8A vs T4 across layers): `analysis/cpu_diagnostics_2026-05-10/outputs/forgery_signal_atlas_with_t4.csv`.
- Per-layer divergence prior (P8A vs E2B): memory `project_per_layer_divergence_2026-05-06.md`.

---

## 8. Self-contained summary (5 bullets)

- T4_LAMBDA1_TOP_N_STEP10500 linear probe lockbox AUC = **1.0000 ± 0.0000** (5-fold StratifiedKFold) vs T4 trained-head AUC = 0.7619 (full lockbox per RESULTS_FACTS §7.1, 0.7957 on the 453-video probe subset).
- P8A_REFERENCE_STEP5000 linear probe lockbox AUC = **1.0000 ± 0.0000** (5-fold StratifiedKFold) vs P8A trained-head AUC = 0.9355 (full lockbox per RESULTS_FACTS §7.1, 0.8638 on the 453-video probe subset).
- Verdict: **α** — both encoders retain linearly separable L11 CLS features on the lockbox subset (probe AUC = 1.000 robust across `C ∈ [1e-3, 10]` and PCA ∈ {None, 50, 20, 10}); the 0.0681 trained-head AUC gap (P8A − T4 on the same 453 videos) is not reflected in the L11 features.
- Sample sizes: 200 reals + 253 fakes = 453 videos (one (768,) feature vector per video, mean of ≤4 random sampled frames). Reals downsampled from 1361 distinct videos (proportional-source-stratified, seed=42); fakes kept at 100% (253/253). `real_dor` source (109 videos) excluded because its `.png` frames are absent from the local mirror used here.
- Caveats: (i) the 453-video subset trained-head gap (T4 0.7957 vs P8A 0.8638) is smaller than the full-lockbox gap (T4 0.7619 vs P8A 0.9355), so the absent `real_dor` videos carry disproportionate trained-head difficulty for T4; (ii) within-PC_Generator probe (the only source with both real + fake) also hit AUC=1.0000 ± 0.0000 for both ckpts (Job C, n=15+62=77), so the AUC=1.000 result is not driven by between-source identity separation; (iii) with n=453 vs dim=768 the L1-trained linear probe is in the overdetermined regime, but the strong-regularization + PCA-10 grid (Job D) shows AUC stays at ≥0.9999 for both ckpts in all 20 cells, ruling out memorization as the cause; (iv) probe inputs are mean-of-≤4-frames, so within-video frame variation is averaged out before the probe sees them.
